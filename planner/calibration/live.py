"""Closed-window selection for an ongoing replay; never waits for engines to idle.

Callers supply complete JSONL records cut at the declared warmup deadline and a
closed telemetry snapshot. Worker clock correction and physical identity must
be explicit, rather than inferred from a generation directory name.
"""
from collections import defaultdict
import math
from .restore import counter_bracket, read_restore_counters
from .restore_audit import include_scrape_collection_time


def prefill_windows(rows, requests, source_nodes, telemetry, *, replicas,
                    start, end, clock_pad_s):
    """Return auditable 60-second windows and exclusion counts.

    Invalid/missing request joins invalidate their whole closed busy episode.
    A continuing busy interval can also supply fully observed interior windows:
    every included forward must have completed-request evidence. No work before
    the first observed launch or after an unjoined forward/gap is inferred.
    """
    if end <= start:
        raise ValueError('Empty calibration window')
    identity=defaultdict(lambda: dict(participants=0,started=0,cached=0,finished_input_lens=[]))
    for node,records in requests.items():
        seen=set()
        for req in records:
            if req['rid'] in seen:
                raise ValueError('Duplicate completed request identity')
            seen.add(req['rid'])
            steps=req['prefill_iters']
            for step in steps:
                identity[node,step]['participants']+=1
            if steps:
                identity[node,steps[0]]['started']+=1
                identity[node,steps[0]]['cached']+=req['cached_tokens']
                if (req.get('requested_output_len') == 1 and req.get('output_len') == 1 and req.get('decode_ct') == 0
                        and req.get('spec_verify_ct') == 0
                        and type(req.get('input_len')) is int and req['input_len'] > 0):
                    identity[node,steps[-1]]['finished_input_lens'].append(req['input_len'])
    by_node=defaultdict(list)
    for row in rows:
        if row['source'] not in source_nodes:
            raise ValueError('Unmapped physical worker stream')
        node=source_nodes[row['source']]
        # Overlap can launch one decode before the last prefill result finishes
        # a one-token request. Its unused result still consumed worker time.
        # Accept it only when completed requests from the immediately preceding
        # prefill prove the exact batch size and launch context; genuine decode
        # and missing records must continue to invalidate a P window.
        finished=identity[node,row['forward_iter']-1]['finished_input_lens']
        auxiliary=(row['mode']=='decode' and row.get('launch_context_source')=='batch_sum'
                   and bool(finished) and len(finished)==row.get('launch_bs')
                   and sum(finished)==row.get('launch_seq_lens_sum'))
        if (row['mode']=='extend' or auxiliary) and row.get('launch_bs'):
            r=dict(row,node=node,launch_wall=row['ts']-row['dur_ms']/1000,
                   auxiliary_prefill_step=auxiliary)
            if start <= r['launch_wall'] and row['ts'] <= end:
                by_node[r['node']].append(r)
    meters=read_restore_counters(telemetry,replicas)
    has_vision = any(r.get('vision_matmul_flops', 0) for events in by_node.values() for r in events)
    windows=[];coverage={}
    for node,events in by_node.items():
        events.sort(key=lambda r:r['forward_iter'])
        if len({r['forward_iter'] for r in events})!=len(events):
            raise ValueError('Multiple worker generations in one calibration stream')
        complete=[];invalid=[];group=None;previous=None;unknown_start=None
        for row in events:
            # An onboarding snapshot can begin while P is already busy. Its
            # first observed full forward anchors a bounded interval; this
            # does not declare the unobserved capture edge idle.
            if previous is None:
                group=[]
            gap=previous is not None and row['forward_iter']!=previous['forward_iter']+1
            if gap:
                origin=group[0]['launch_wall'] if group else previous['launch_wall']
                unknown_start=origin if unknown_start is None else min(origin,unknown_start)
                group=None
            if row['after_idle']:
                if unknown_start is not None:
                    invalid.append((unknown_start,row['launch_wall']))
                    unknown_start=None
                if group:
                    a,b=group[0]['launch_wall'],group[-1]['ts']
                    joined=all(r['auxiliary_prefill_step'] or
                               identity[node,r['forward_iter']]['participants']==r['launch_bs'] for r in group)
                    if a < b <= row['launch_wall'] and joined:
                        complete.append((a,b,group))
                    else:
                        invalid.append((a,max(b,row['launch_wall'])))
                group=[]
            if group is not None:
                group.append(row)
            previous=row
        if unknown_start is not None:
            invalid.append((unknown_start,end))
        # Saturated P workers may never become idle during warmup. Retain only
        # the completed, joined prefix of their ongoing interval. A request
        # still in flight at the cutoff cannot contribute features or time.
        observed=[]
        if group:
            prefix=[]
            for row in group:
                if (not row['auxiliary_prefill_step'] and
                        identity[node,row['forward_iter']]['participants']!=row['launch_bs']):
                    # Overlap can have started this unjoined forward before
                    # the preceding result was observed. Trim that overlap.
                    while prefix and prefix[-1]['ts']>row['launch_wall']:
                        prefix.pop()
                    break
                prefix.append(row)
            if prefix:
                a,b=prefix[0]['launch_wall'],prefix[-1]['ts']
                if a < b:
                    observed.append((a,b,prefix))
        audit=dict(total_steps=len(events),closed_episodes=len(complete),
                   invalid_episodes_or_gaps=len(invalid),open_steps=len(group or []),
                   observed_busy_prefixes=len(observed),
                   observed_busy_prefix_steps=sum(len(g) for _,_,g in observed),
                   eligible_steps=0,windows=0,excluded_windows=0)
        coverage[node]=audit
        measured=complete+observed
        if not measured:
            continue
        horizon=(measured[0][0],measured[-1][1])
        for index in range(int((end-start)//60)):
            a,b=start+60*index,start+60*(index+1)
            if not horizon[0]<=a<b<=horizon[1] or any(x<b and y>a for x,y in invalid):
                audit['excluded_windows']+=1
                continue
            steps=[r for _,_,g in measured for r in g if a<=r['launch_wall']<b]
            seconds=sum(max(0,min(y,b)-max(x,a)) for x,y,_ in measured)
            if not steps or seconds<=0:
                audit['excluded_windows']+=1
                continue
            try:
                ca,alo,ahi,agap=counter_bracket(meters[node],a)
                cb,blo,bhi,bgap=counter_bracket(meters[node],b)
                sample=dict(node=node,start=a,end=b,seconds=seconds,steps=len(steps),
                    requests=sum(identity[node,r['forward_iter']]['started'] for r in steps),
                    started_cached_tokens=sum(identity[node,r['forward_iter']]['cached'] for r in steps),
                    features=[len(steps),sum(r['new_prefill_tokens'] for r in steps),
                              sum(r['prefill_attention_pairs'] for r in steps
                                  if not r['auxiliary_prefill_step']),cb-ca],
                    restore_lo=max(0,blo-ahi),restore_hi=bhi-alo,
                    max_scrape_gap_s=max(agap,bgap))
                if any(not math.isfinite(v) or v<0 for v in sample['features']):
                    raise ValueError('Invalid window features')
                if has_vision:
                    vision_work = sum(r.get('vision_matmul_flops', 0) for r in steps)
                    if not math.isfinite(vision_work) or vision_work < 0:
                        raise ValueError('Invalid vision work')
                    sample['features'].append(vision_work)
                windows.append(sample)
            except ValueError:
                audit['excluded_windows']+=1
        selected=[w for w in windows if w['node']==node]
        audit['windows']=len(selected)
        audit['eligible_steps']=sum(w['steps'] for w in selected)
    # Full HTTP observation brackets, not request-completion timestamps alone.
    excluded=include_scrape_collection_time(windows,telemetry,replicas,clock_pad_s,
                                           exclude_unbracketed=True)
    for node,audit in coverage.items():
        rejected=[w for w in excluded if w['node']==node]
        selected=[w for w in windows if w['node']==node]
        audit['excluded_windows']+=len(rejected)
        audit['restore_counter_excluded_windows']=len(rejected)
        audit['restore_counter_exclusions']=rejected
        audit['windows']=len(selected)
        audit['eligible_steps']=sum(w['steps'] for w in selected)
    return sorted(windows,key=lambda w:(w['end'],w['node'])),coverage
