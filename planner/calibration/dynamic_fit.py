"""Once-per-warmup parameter fitting; no engine stop, role change, or cap change."""
from .nodes import fit_prefill, fit_prefill_scale, fit_decode


def fit_bundle(*, windows, decode_rows, workers, start, end, experts, top_k,
               predeclared_prefill_shape):
    """Keep per-node fitted models and explicitly identify missing-role priors.

    P coefficients include vision work when present. An insufficient or singular
    sample falls back to ONE scale on the predeclared shape; high validation
    error alone never selects a different model. Unfit observed roles are
    retained as excluded evidence and use same-run role donors, not invented
    per-node measurements. No donors for either role is a hard failure.
    """
    if end <= start or any(w['start'] < start or w['end'] > end for w in windows):
        raise ValueError('P fit window lies outside frozen learning interval')
    if any(w['node'] not in workers for w in windows) or any(n not in workers for n in decode_rows):
        raise ValueError('Physical worker identity mismatch')
    if len(predeclared_prefill_shape) == 5:
        # A window without screenshots did zero vision work; a text-only interval still
        # fits the five-term shape (the vision term is absent from its features).
        windows=[dict(w,features=list(w['features'])+[0.]) if len(w['features'])==4 else w
                 for w in windows]
    split=start+.7*(end-start)
    profiles={};failures={}
    for node,worker in workers.items():
        own={}
        rows=[w for w in windows if w['node']==node]
        if rows:
            try:
                try:
                    model=fit_prefill(rows,split_time=split)
                    model['method']=('per_node_five_term_nnls' if len(model['coefficients']) == 5
                                     else 'per_node_four_term_nnls')
                except ValueError as exc:
                    model=fit_prefill_scale(rows,split_time=split,
                        reference_coefficients=predeclared_prefill_shape)
                    model['full_fit_unavailable']=str(exc)
                selected=[w for w in rows if w['end']<=split or w['start']>=split]
                sums=[sum(w['features'][i] for w in selected) for i in range(4)]
                cached=sum(w['started_cached_tokens'] for w in selected)
                if sums[0]<=0 or sums[1]<=0 or (cached<=0 and sums[3]>0):
                    raise ValueError('Cannot identify packing or restore/cached ratio')
                model.update(tokens_per_step=sums[1]/sums[0],
                             restored_per_cached_token=sums[3]/cached if cached else 0.,
                             calibration_node=node)
                own['prefill']=model
            except ValueError as exc:
                failures[node+':prefill']=str(exc)
        if decode_rows.get(node):
            try:
                model=fit_decode(decode_rows[node],start=start,end=end,experts=experts,top_k=top_k)
                model['calibration_node']=node
                own['decode']=model
            except ValueError as exc:
                failures[node+':decode']=str(exc)
        profiles[node]=own
    donors={role:[p[role] for p in profiles.values() if role in p]
            for role in ('prefill','decode')}
    if any(not d for d in donors.values()):
        raise ValueError('Missing current-run calibrated role donors: '+repr(failures))
    result={}
    for node,worker in workers.items():
        result[node]={k:worker[k] for k in ('kv_capacity_tokens','max_running_requests')}
        for role in donors:
            own=profiles[node].get(role)
            result[node][role]=dict(origin='measured' if own else 'same_run_role_prior',
                                   models=[own] if own else donors[role])
    return dict(workers=result,fit_failures=failures,window_start=start,window_end=end,
        split_time=split,experts=experts,top_k=top_k,parameters_frozen_after_warmup=True,
        limitations=[
            'Only the initial role is measured on each physical node; other roles use declared donors.',
            'Validation residuals are observations, not statistical confidence intervals.',
            'Packing and restore/cached ratio are warmup averages; placement can change them.',
            'Decode context has no independent fitted term; unobserved batches are extrapolations.',
            'Mean KV occupancy and finite-population capacity do not guarantee tail latency.',
        ])


def snapshot_measurements(manifest, *, replicas):
    """Reuse the same checked, closed windows for calibration and later updates."""
    import hashlib
    from pathlib import Path
    from .trace import json_lines, load_steps
    from .live import prefill_windows
    root=Path(manifest['telemetry']).parent
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Calibration snapshot changed: '+name)
    start,end,pad=(manifest[k] for k in ('start','end','clock_pad_s'))
    p_rows=[];requests={};source_nodes={};d_rows={}
    for node,worker in manifest['workers'].items():
        rows=load_steps([worker['steps']],start+pad,end-pad)
        if worker['role']=='prefill':
            p_rows.extend(rows)
            requests[node]=list(json_lines(worker['requests']))
            source_nodes[worker['steps']]=node
        elif worker['role']=='decode':
            # Completion inside the trimmed interval is insufficient: a first
            # launch can precede it. Trim whole launch intervals while keeping
            # the original chronological70/30 split used by fit_bundle.
            d_rows[node]=[r for r in rows
                if start+pad <= r['ts']-r['dur_ms']/1000
                and r['ts']-r['dur_ms']/1000+(r.get('busy_interval_s') or 0) <= end-pad]
        else:
            raise ValueError('Warmup must have an exclusive declared initial role')
    windows,coverage=prefill_windows(p_rows,requests,source_nodes,manifest['telemetry'],
        replicas=replicas,start=start,end=end,clock_pad_s=pad)
    return windows, d_rows, coverage


def fit_snapshot(manifest, *, capacities, replicas, experts, top_k, predeclared_prefill_shape):
    """Connect the immutable snapshot to fits with nodeN identities throughout."""
    if set(manifest['workers'])!=set(capacities):
        raise ValueError('Snapshot/capacity worker membership differs')
    windows,d_rows,coverage=snapshot_measurements(manifest,replicas=replicas)
    start,end=manifest['start'],manifest['end']
    fitted=fit_bundle(windows=windows,decode_rows=d_rows,workers=capacities,
        start=start,end=end,experts=experts,top_k=top_k,
        predeclared_prefill_shape=predeclared_prefill_shape)
    fitted.update(prefill_windows=windows,prefill_coverage=coverage)
    return fitted
