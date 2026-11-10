"""Low-frequency capacity decisions from frozen, per-worker warmup fits.

This predicts aggregate service capacity, not request finish times. Measured
roles retain their own coefficients; unobserved roles use the median prediction
of the explicitly listed same-run donors. No output-length oracle is consumed.
"""
import math
from statistics import fmean, median, stdev

from .guards import EVICT_RELEASE_GAIN, EVICT_RELEASE_P95_S, floor_restore_ratios

REQUIRED_STANDARD_ERRORS = 2.
# D batch 'measured': a window is memory-bound when p_completed - running >= max(MIN, FRACTION * p_completed).
MEMORY_BOUND_MIN_REQUESTS = 2.
MEMORY_BOUND_FRACTION = .2
FORMULA_KV_FRACTION = .95


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('Nonpositive/nonfinite ' + name)
    return value


def workload_delta(previous, current):
    """Difference two cumulative completed-usage snapshots from one router epoch."""
    keys = ('requests', 'input_tokens', 'uncached_tokens', 'causal_attention_pairs',
            'output_tokens', 'verified_output_tokens', 'verify_steps')
    delta = {key: current[key] - previous[key] for key in keys}
    delta['vision_matmul_flops'] = current.get('vision_matmul_flops', 0) - previous.get('vision_matmul_flops', 0)
    delta['image_attention_pairs'] = current.get('image_attention_pairs', 0) - previous.get('image_attention_pairs', 0)
    if any(not math.isfinite(v) or v < 0 for v in delta.values()):
        raise ValueError('Completed workload counters reset or became invalid')
    n = positive(delta['requests'], 'completed requests')
    if delta['uncached_tokens'] > delta['input_tokens']:
        raise ValueError('Uncached tokens exceed input')
    output = positive(delta['output_tokens'] / n, 'observed output length')
    return dict(requests=n, uncached=delta['uncached_tokens']/n,
                cached=(delta['input_tokens']-delta['uncached_tokens'])/n,
                pairs=(delta['causal_attention_pairs']+delta['image_attention_pairs'])/n, output=output,
                vision_matmul_flops=delta['vision_matmul_flops']/n,
                context=delta['input_tokens']/n+output/2,
                acceptance=(delta['verified_output_tokens']/delta['verify_steps']
                            if delta['verify_steps'] else 1.))


def prefill_seconds(profile, work, onboarding=None):
    coefficients = profile['coefficients']
    if len(coefficients) not in (4, 5):
        raise ValueError('Expected measured P steps/tokens/attention-pairs/restore/vision schema')
    vision = work.get('vision_matmul_flops', 0)
    if not math.isfinite(vision) or vision < 0:
        raise ValueError('Invalid vision work')
    if vision > 0 and len(coefficients) == 4:
        raise ValueError('Image work requires a measured vision coefficient')
    # All coefficients, including vision and restore, remain those of the
    # initial calibration. Onboarding observes the current restore ratio and
    # packing (new tokens per forward step); a node without an observed packing
    # keeps its warmup packing.
    restore_fraction = (profile['restored_per_cached_token'] if onboarding is None
                        else onboarding['restored_per_cached_token'])
    packing = (onboarding or {}).get('tokens_per_step', profile['tokens_per_step'])
    features = [work['uncached']/positive(packing, 'packing'),
                work['uncached'], work['pairs'],
                work['cached']*restore_fraction]
    if len(coefficients) == 5:
        features.append(vision)
    if any(not math.isfinite(x) or x < 0 for x in coefficients+features):
        raise ValueError('Invalid P model/work')
    # Zero-work cache hits are permitted; node aggregate work must remain >0.
    return positive(sum(c*x for c,x in zip(coefficients, features)), 'P demand')


def decode_capacity(profile, worker, work, experts, top_k, population_per_d=None, kv_fraction=None):
    cap = min(int(worker['max_running_requests']),
              math.floor(FORMULA_KV_FRACTION*worker['kv_capacity_tokens']/positive(work['context'], 'context')))
    if kv_fraction is not None and cap >= 1:
        # D batch 'measured': the KV cap uses the learned effective KV fraction. Like the population bound below it
        # is a fluid operating point (a time-weighted running batch), not an integer engine batch.
        cap = max(1., min(float(worker['max_running_requests']),
                          kv_fraction*worker['kv_capacity_tokens']/work['context']))
    duty = 1.
    if population_per_d is not None:
        if not math.isfinite(population_per_d) or population_per_d < 0:
            raise ValueError('Invalid D population')
        duty = min(1., population_per_d)
        # A mean population is fractional. Flooring independently on every D
        # invents capacity gains when the number of D workers changes (e.g. 13
        # requests across seven versus eight nodes). This is a fluid operating
        # point in the continuous step formula, not an actual engine batch.
        cap = min(cap, max(1., population_per_d))
    if cap < 1:
        raise ValueError('Mean request cannot fit worker KV capacity')
    coeff = profile['coefficients']
    if len(coeff) != 3 or any(not math.isfinite(c) or c < 0 for c in coeff):
        raise ValueError('Invalid D step model')
    if not 0 < top_k <= experts:
        raise ValueError('Invalid expert counts')
    fraction = 1. if top_k == experts else -math.expm1(
        cap*profile['draft_tokens']*math.log1p(-top_k/experts))
    step = positive(coeff[0]+coeff[1]*cap+coeff[2]*fraction, 'D step seconds')
    acceptance = positive(work['acceptance'], 'observed acceptance')
    return dict(batch=cap, step_s=step, duty_fraction=duty,
                rps=duty*cap*acceptance/(step*positive(work['output'], 'output')),
                supported_batch=cap in profile['batch_support'])


def capacities(workers, roles, work, *, experts, top_k, decode_population=None,
               onboarding=None, kv_fraction=None):
    """P sum assumes balancing; RR D is constrained by the slowest D.

    An optional measured decode population bounds average batch size. This is a
    quasi-steady estimate, not an inference that every engine runs full batches.
    Both bound and no-bound predictions should be recorded in decisions.
    """
    if set(workers) != set(roles) or set(roles.values()) != {'prefill', 'decode'}:
        raise ValueError('Complete split role map required')
    p_capacity, d_rates, evidence = 0., [], {}
    d_count = sum(r == 'decode' for r in roles.values())
    for key, worker in workers.items():
        role = roles[key]
        profiles = worker[role]['models']
        if not profiles:
            raise ValueError('Missing measured model or declared role prior')
        if role == 'prefill':
            restore = (onboarding or {}).get(key)
            demands = [prefill_seconds(model, work, restore) for model in profiles]
            seconds = median(demands)
            p_capacity += 1/seconds
            evidence[key] = dict(role=role, seconds=seconds, origin=worker[role]['origin'])
            evidence[key]['onboarding'] = restore
        else:
            points = [decode_capacity(model, worker, work, experts, top_k,
                        None if decode_population is None else decode_population/d_count, kv_fraction)
                      for model in profiles]
            rate = median(p['rps'] for p in points)
            d_rates.append(rate)
            evidence[key] = dict(role=role, rps=rate, operating_points=points,
                                 origin=worker[role]['origin'])
    d_capacity = d_count*min(d_rates)
    return dict(prefill_rps=p_capacity, decode_rr_rps=d_capacity,
                system_rps=min(p_capacity,d_capacity), workers=evidence,
                population_bound=decode_population)


def _role_maps(workers, roles, work, *, decode_population, **kwargs):
    """One role map for every other P count, each with the fewest changes.

    Walking away from the current count in both directions, each step changes
    one more node: the one whose new map has the highest whole-interval
    estimate (system, then total capacity, then node name). A node whose map
    cannot be estimated is passed over; if none is left, that direction ends.
    """
    maps = []
    for source, target in (('decode', 'prefill'), ('prefill', 'decode')):
        candidate, changed = dict(roles), []
        while sum(r == source for r in candidate.values()) > 1:
            best = None
            for node in sorted(n for n, r in candidate.items() if r == source):
                trial = {**candidate, node: target}
                try:
                    e = capacities(workers, trial, work, decode_population=decode_population, **kwargs)
                except ValueError:
                    continue
                key = (e['system_rps'], e['prefill_rps'] + e['decode_rr_rps'])
                if best is None or key > best[0]:
                    best = (key, node, trial)
            if best is None:
                maps.append(dict(changed=sorted(changed), from_roles=candidate,
                                 error='No estimable ' + source + ' node left to change'))
                break
            candidate, changed = best[2], [*changed, best[1]]
            maps.append(dict(changed=sorted(changed), roles=candidate))
    return maps


def decide(workers, roles, work, *, minutes, experts, top_k,
           decode_population=None, onboarding=None, guards=None, queue=None, queue_added=False,
           decode_batch=None, evict_pressure=None, first_decision=False):
    """Move to the P count whose minute-by-minute gain is consistent and largest.

    Every other P count is a candidate, reached from the current roles with the
    fewest changes (`_role_maps`), so one decision may change several nodes.

    `work` and `decode_population` describe the whole interval and are recorded.
    `minutes` holds the same inputs for consecutive slices of that interval.
    Within a slice a candidate and the current roles share one workload, so
    variation common to both cancels. A candidate is eligible when its mean gain
    exceeds REQUIRED_STANDARD_ERRORS standard errors of that mean (about 95%
    confidence): the margin is measured from each interval, not configured. A
    model bias common to every slice, or a later workload change, is not seen.
    Equal means prefer fewer changes.

    Current assignment is always a candidate; same-count permutations and
    data-dependent model selection are deliberately outside this controller.

    `guards` (evolving_infra.planner.calibration.guards.PlannerGuards) enables opt-in guards; with
    None or all guards off the decision is unchanged and records nothing new.
    `queue` is the live P_queue observation for the queue add-P guard
    (p95_s, n, window_s, at, since_switch_s). `queue_added` tells the queue
    removal hold whether a switch selected by the queue add-P guard was
    confirmed earlier in the run; the caller keeps that state. `decode_batch`
    holds the measured inputs of the D batch option: running ({D node:
    {batch, passes, dur_s}} over the decision window), p_completed and
    population of the same window, and kv_fraction, the last learned fraction
    (None before the first). `evict_pressure` is the eviction-pressure
    hold's observation (HostEvictionMonitor.observe: tokens, signal,
    complete, per-node tokens); `first_decision` (the run's first periodic
    decision) matters only for its 'first_decision' release. With any guard
    on, the decision records under 'guards' what each guard saw and what it
    changed.
    """
    common = dict(minutes=minutes, experts=experts, top_k=top_k, decode_population=decode_population)
    if guards is None or not guards.enabled:
        return _decide(workers, roles, work, onboarding=onboarding, **common)
    used, floor = floor_restore_ratios(onboarding) if guards.restore_floor else (onboarding, None)
    batch, kv = None, {}
    if guards.decode_batch_source == 'measured':
        batch = _decode_kv_fraction(workers, roles, work, decode_batch)
        if batch['applied']:
            kv = dict(kv_fraction=batch['kv_fraction'])
    decision = _decide(workers, roles, work, onboarding=used,
                       min_relative_gain=guards.min_relative_gain, **common, **kv)
    selections = dict(without_guards=_selection(
        _decide(workers, roles, work, onboarding=onboarding, **common)))
    if guards.restore_floor and guards.min_relative_gain:
        selections['restore_floor_only'] = _selection(
            _decide(workers, roles, work, onboarding=used, **common, **kv))
        selections['min_relative_gain_only'] = _selection(
            _decide(workers, roles, work, onboarding=onboarding,
                    min_relative_gain=guards.min_relative_gain, **common, **kv))
    selections['standard'] = _selection(decision)
    if batch is not None:
        # Counterfactual: the same guards with the formula's decode cap.
        formula = _decide(workers, roles, work, onboarding=used, min_relative_gain=guards.min_relative_gain, **common)
        selections['decode_formula'] = _selection(formula)
        batch.update(estimate=_estimate_summary(decision['candidates'][0]),
                     formula_estimate=_estimate_summary(formula['candidates'][0]))
    record = dict(options=guards.as_dict(), restore_floor=floor, min_relative_gain=None,
                  selections=selections, queue_add_prefill=None)
    if guards.min_relative_gain:
        record['min_relative_gain'] = dict(threshold=guards.min_relative_gain, blocked=[
            _gain_summary(o) for o in decision['candidates'][1:]
            if 'minute_gain' in o and o['minute_gain']['eligible']
            and not o['minute_gain']['passes_min_relative_gain']])
    hold, evict, record['queue_add_prefill'] = _removal_and_add_guards(
        decision, roles, guards, queue, queue_added, evict_pressure, first_decision)
    if hold is not None:
        record['queue_removal_hold'] = hold
    if evict is not None:
        record['evict_pressure_hold'] = evict
    if batch is not None:
        record['decode_batch'] = batch
    held = bool((record.get('queue_removal_hold') or {}).get('blocked'))
    evict_held = bool((evict or {}).get('blocked'))
    record['selected_by'] = ('queue_add_prefill' if (record['queue_add_prefill'] or {}).get('fired')
                             else 'queue_removal_hold' if held
                             else 'evict_pressure_hold' if evict_held
                             else 'minute_standard_error' if decision['action'] == 'switch' else None)
    record['fired'] = [name for name, fired in (
        ('restore_floor', bool(floor and floor['applied'])),
        ('min_relative_gain', bool(record['min_relative_gain'] and record['min_relative_gain']['blocked'])),
        ('queue_removal_hold', held),
        ('evict_pressure_hold', evict_held),
        ('decode_batch', batch is not None and (selections['standard']['action'], selections['standard']['changed'])
         != (selections['decode_formula']['action'], selections['decode_formula']['changed'])),
        ('queue_add_prefill', record['selected_by'] == 'queue_add_prefill')) if fired]
    final = _selection(decision)
    record['changed_decision'] = ((final['action'], final['changed'])
                                  != (selections['without_guards']['action'],
                                      selections['without_guards']['changed']))
    decision['guards'] = record
    return decision


def _prefill_count(roles):
    return sum(role == 'prefill' for role in roles.values())


def _gain_summary(option):
    gain = option['minute_gain']
    return dict(changed=option['changed'], prefill_nodes=_prefill_count(option['roles']),
                mean_rps=gain['mean_rps'], standard_error_rps=gain['standard_error_rps'],
                mean_relative=gain['mean_relative'])


def _selection(decision):
    selected = decision['selected']
    gain = selected.get('minute_gain') or {}
    return dict(action=decision['action'], changed=sorted(selected['changed']),
                prefill_nodes=_prefill_count(selected['roles']),
                mean_rps=gain.get('mean_rps'), mean_relative=gain.get('mean_relative'))


def _estimate_summary(option):
    estimate = option['estimate']
    return {k: estimate[k] for k in ('prefill_rps', 'decode_rr_rps', 'system_rps')}


def _decode_kv_fraction(workers, roles, work, decode_batch):
    """D batch 'measured': the effective KV fraction of the decode cap for every role map, and its record.

    Memory-bound: the requests whose P leg has completed (the router's p_completed, time-weighted over the window)
    exceed the measured running decode requests of the current D nodes by at least max(2, 0.2 * p_completed). The
    full decode population (p_completed + p_pending) is not used: it also counts requests still queued or computing
    at P, which is large when P is the bottleneck. In a memory-bound window f = median over D nodes of
    running * context / kv_capacity_tokens (at most 1); otherwise the last learned f is kept, so a
    supply-limited window never lowers the cap. Before the first learned f the formula's floored cap stays in use
    ('applied' False): the fluid 0.95 cap could exceed it. A window in which some, but not all, current D nodes have no
    decode pass (idle, stalled or no trace) has no running measurement: counting such a node as 0 could make a
    supply-limited window look memory-bound and learn a low f.
    """
    inputs = decode_batch or {}
    running = inputs.get('running') or {}
    previous = inputs.get('kv_fraction')
    completed = inputs.get('p_completed')
    context = positive(work['context'], 'context')
    d_nodes = sorted(node for node, role in roles.items() if role == 'decode')
    nodes = {node: running[node] for node in d_nodes if node in running}
    batches = {node: nodes[node]['batch'] for node in nodes}
    idle = sorted(node for node in nodes if not (nodes[node].get('passes') or 0) > 0)
    partly_idle = bool(idle) and len(idle) < len(d_nodes)
    total = sum(batches.values()) if len(nodes) == len(d_nodes) and not partly_idle else None
    formula = {node: min(int(workers[node]['max_running_requests']),
                         math.floor(FORMULA_KV_FRACTION*workers[node]['kv_capacity_tokens']/context))
               for node in d_nodes}
    margin = None if completed is None else max(MEMORY_BOUND_MIN_REQUESTS, MEMORY_BOUND_FRACTION*completed)
    memory_bound, raw = None, None
    if total is None:
        reason = ('no decode pass in the window on D node ' + ', '.join(idle)
                  if partly_idle and len(nodes) == len(d_nodes) else 'no running measurement for every D node')
    elif completed is None:
        reason = 'no p_completed population'
    elif not total > 0:
        reason = 'no running decode request'
    else:
        memory_bound = completed-total >= margin
        reason = 'memory-bound' if memory_bound else 'not memory-bound'
        if memory_bound:
            raw = median(batches[node]*context/workers[node]['kv_capacity_tokens'] for node in d_nodes)
            if not raw > 0:
                memory_bound, reason = None, 'median running decode batch is 0'
    learned = bool(memory_bound)
    fraction = min(1., raw) if learned else FORMULA_KV_FRACTION if previous is None else previous
    applied = learned or previous is not None
    used = {node: (max(1., min(float(workers[node]['max_running_requests']),
                               fraction*workers[node]['kv_capacity_tokens']/context)) if applied else formula[node])
            if formula[node] >= 1 else None for node in d_nodes}
    return dict(source='measured', context=context, p_completed=completed, population=inputs.get('population'),
                running_per_d=dict(median=median(batches.values()) if batches else None, nodes=batches),
                running_total=total, passes={node: nodes[node].get('passes') for node in nodes},
                margin=margin, memory_bound=memory_bound, reason=reason, kv_fraction=fraction,
                kv_fraction_measured=raw, learned=learned, previous_kv_fraction=previous, applied=applied,
                idle_d_nodes=idle,
                cap_formula=dict(median=median(formula.values()) if formula else None, nodes=formula),
                cap_used=dict(median=median(v for v in used.values() if v is not None) if any(
                    v is not None for v in used.values()) else None, nodes=used))


def _queue_guards(decision, roles, guards, queue, queue_added):
    """Q-hold, then Q', on the standard decision, which is updated in place; returns their records.

    Both read the standard rule's choice: Q' fires exactly as with Q-hold off (Q' is never blocked) and its +1 P
    overrides Q-hold's choice. Without either guard its record is None. (E-hold, if on, sees no signal here.)
    """
    hold, _, add = _removal_and_add_guards(decision, roles, guards, queue, queue_added, None)
    return hold, add


def _removal_and_add_guards(decision, roles, guards, queue, queue_added, evict_pressure, first_decision=False):
    """Q-hold and E-hold, then Q', on the standard decision, which is updated in place; returns their records.

    All three read the standard rule's choice. A P removal blocked by either hold is replaced by the best remaining
    selectable candidate without P removal (the same for both), else a hold. Q' fires exactly as with both holds off
    (it is never blocked) and its +1 P overrides the holds' choice. Without a guard its record is None.
    """
    standard = dict(decision)
    hold = evict = add = None
    blocked = False
    if guards.queue_removal_floor_s:
        hold, kept = _queue_removal_hold(standard, roles, guards, queue, queue_added)
        blocked = hold['blocked']
    if guards.evict_pressure_hold:
        evict, kept_evict = _evict_pressure_hold(standard, roles, guards, evict_pressure, queue, first_decision)
        if not blocked:
            kept = kept_evict
        blocked = blocked or evict['blocked']
    if blocked:
        decision.update(action='hold' if kept is None else 'switch',
                        selected=decision['candidates'][0] if kept is None else kept)
    if guards.queue_p95_s:
        add, plus = _queue_add_prefill(standard, roles, guards, queue)
        if add['fired']:
            decision.update(action='switch', selected=plus)
    return hold, evict, add


def _no_p95_reason(queue):
    """Why an observation has no P_queue p95: no observation, or (ungated signal) only gated P legs."""
    if queue is not None and queue.get('signal') == 'ungated' and queue.get('n_all'):
        return 'no ungated P leg in the window'
    return 'no live P_queue observation'


def _queue_add_prefill(decision, roles, guards, queue):
    """Q': the +1 P candidate when the live P_queue p95 is above the threshold."""
    current = _prefill_count(roles)
    plus = next((o for o in decision['candidates'][1:]
                 if 'minute_gain' in o and _prefill_count(o['roles']) == current+1), None)
    p95 = None if queue is None else queue.get('p95_s')
    since = None if queue is None else queue.get('since_switch_s')
    if p95 is None:
        reason = _no_p95_reason(queue)
    elif not p95 > guards.queue_p95_s:
        reason = 'P_queue p95 not above threshold'
    elif since is not None and since < guards.queue_min_since_switch_s:
        reason = 'too soon after the last role switch'
    elif decision['action'] == 'switch' and _prefill_count(decision['selected']['roles']) > current:
        reason = 'standard rule already adds a P'
    elif plus is None:
        reason = 'no estimable +1 P candidate'
    elif not plus['minute_gain']['mean_rps'] > 0:
        reason = '+1 P mean minute gain not positive'
    else:
        reason = 'fired'
    return dict(threshold_s=guards.queue_p95_s, min_since_switch_s=guards.queue_min_since_switch_s,
                observation=queue, candidate=None if plus is None else _gain_summary(plus),
                fired=reason == 'fired', reason=reason), plus


def _preference(option):
    """The standard order of `_decide`: largest mean minute gain, then fewest changes."""
    return option['minute_gain']['mean_rps'], -len(option['changed'])


def _selectable(option):
    """Whether the standard rule (two standard errors, and G3 when on) admits the candidate."""
    gain = option.get('minute_gain')
    return bool(gain) and gain['eligible'] and gain.get('passes_min_relative_gain', True)


def _queue_removal_hold(decision, roles, guards, queue, queue_added):
    """Q-hold: after a confirmed Q' switch, no P removal while the live P_queue p95 is at or above the floor.

    Returns (record, kept). When the standard selection is blocked, `kept` is
    the best remaining selectable candidate without P removal, or None (hold).
    """
    current = _prefill_count(roles)
    selected = decision['selected']
    removes = decision['action'] == 'switch' and _prefill_count(selected['roles']) < current
    p95 = None if queue is None else queue.get('p95_s')
    blocked, kept = False, None
    if not removes:
        reason = 'standard rule does not remove a P'
    elif not queue_added:
        reason = 'no queue add-P switch confirmed earlier in the run'
    elif p95 is None:
        reason = _no_p95_reason(queue)
    elif math.isnan(p95):
        reason = 'P_queue p95 is NaN'
    elif not p95 >= guards.queue_removal_floor_s:
        reason = 'P_queue p95 below the floor'
    else:
        blocked = True
        kept = _best_without_removal(decision, current)
        reason = ('P removal blocked, best remaining candidate selected' if kept is not None
                  else 'P removal blocked, no remaining candidate: hold')
    return dict(floor_s=guards.queue_removal_floor_s, queue_added=bool(queue_added), p95_s=p95,
                blocked=blocked, candidate=_gain_summary(selected) if removes else None,
                selected=None if kept is None else _gain_summary(kept), reason=reason), kept


def _best_without_removal(decision, current):
    """The best standard-selectable candidate with at least `current` P (standard order), or None."""
    return max((o for o in decision['candidates'][1:]
                if _selectable(o) and _prefill_count(o['roles']) >= current), key=_preference, default=None)


def _evict_release(decision, guards, queue, first_decision):
    """The release bound of a removal that the eviction pressure would block: {mode, value, limit, released, reason}."""
    mode = guards.evict_pressure_release
    if mode == 'queue_p95':
        # The all-leg P_queue p95 of the Q' observation (p95_all when the observation is ungated).
        p95 = None if queue is None else queue.get('p95_all', queue.get('p95_s'))
        if p95 is None or isinstance(p95, bool) or not isinstance(p95, (int, float)) or math.isnan(p95):
            return dict(mode=mode, value=p95, limit=EVICT_RELEASE_P95_S, released=False,
                        reason='no all-leg P_queue p95: no release')
        released = p95 < EVICT_RELEASE_P95_S
        return dict(mode=mode, value=p95, limit=EVICT_RELEASE_P95_S, released=released,
                    reason='all-leg P_queue p95 below the limit' if released else 'all-leg P_queue p95 at or above the limit')
    if mode == 'gain':
        gain = (decision['selected'].get('minute_gain') or {}).get('mean_relative')
        released = gain is not None and gain >= EVICT_RELEASE_GAIN
        return dict(mode=mode, value=gain, limit=EVICT_RELEASE_GAIN, released=released,
                    reason='predicted relative gain at or above the limit' if released
                    else 'predicted relative gain below the limit')
    if mode == 'first_decision':
        released = bool(first_decision)
        return dict(mode=mode, value=released, limit=None, released=released,
                    reason='the first decision of the run' if released else 'not the first decision')
    return dict(mode=mode, value=None, limit=None, released=False, reason='no release bound')


def _evict_pressure_hold(decision, roles, guards, observation, queue=None, first_decision=False):
    """E-hold: no P removal while the P nodes' class-1 host-only evictions in the window exceed the threshold.

    Returns (record, kept) like Q-hold. No observation, no signal (no current P node with a fresh sample) or a
    non-finite total does not block. With some current P nodes unobserved the visible total is a lower bound: it
    blocks only when it exceeds the threshold. The release bound can still allow the removal. In shadow mode the
    would-block result is recorded (`would_block`, `selected`) and nothing is blocked.
    """
    current = _prefill_count(roles)
    selected = decision['selected']
    removes = decision['action'] == 'switch' and _prefill_count(selected['roles']) < current
    tokens = None if observation is None else observation.get('tokens')
    complete = None if observation is None else bool(observation.get('complete'))
    shadow = guards.evict_pressure_hold == 'shadow'
    would, kept, release = False, None, None
    if not removes:
        reason = 'standard rule does not remove a P'
    elif (observation is None or not observation.get('signal') or isinstance(tokens, bool)
          or not isinstance(tokens, (int, float)) or not math.isfinite(tokens)):
        reason = 'no eviction signal'
    elif not tokens > guards.evict_pressure_tokens:
        reason = ('P class-1 host evictions not above the threshold' if complete
                  else 'eviction signal incomplete, visible total not above the threshold')
    else:
        release = _evict_release(decision, guards, queue, first_decision)
        if release['released']:
            reason = 'P class-1 host evictions above the threshold, removal released: '+release['reason']
        else:
            would = True
            kept = _best_without_removal(decision, current)
            reason = ('P removal blocked, best remaining candidate selected' if kept is not None
                      else 'P removal blocked, no remaining candidate: hold')
    blocked = would and not shadow
    return dict(mode='shadow' if shadow else 'on', threshold_tokens=guards.evict_pressure_tokens,
                window_s=guards.evict_pressure_window_s, tokens=tokens, complete=complete, would_block=would,
                blocked=blocked, release=release, candidate=_gain_summary(selected) if removes else None,
                selected=None if kept is None else _gain_summary(kept), reason=reason,
                observation=observation), (kept if blocked else None)


def _decide(workers, roles, work, *, minutes, experts, top_k, decode_population,
            onboarding, min_relative_gain=0., kv_fraction=None):
    """The standard rule of `decide`; a positive min_relative_gain adds G3; kv_fraction is the measured D cap."""
    if len(workers) < 2:
        raise ValueError('P/D role control requires at least two model nodes')
    if len(minutes) < 2:
        raise ValueError('At least two workload slices are required')
    kwargs = dict(experts=experts, top_k=top_k, onboarding=onboarding)
    if kv_fraction is not None:
        kwargs['kv_fraction'] = kv_fraction
    options=[dict(changed=[], roles=dict(roles)),
             *_role_maps(workers, roles, work, decode_population=decode_population, **kwargs)]
    for option in options:
        if 'error' in option:
            continue
        try:
            option['estimate']=capacities(workers,option['roles'],work,
                                          decode_population=decode_population,**kwargs)
            option['minute_rps']=[capacities(workers,option['roles'],m['work'],
                                             decode_population=m['decode_population'],**kwargs)['system_rps']
                                  for m in minutes]
        except ValueError as exc:
            if not option['changed']:
                raise
            option.pop('estimate',None)
            option['error']=str(exc)
    current=options[0]
    base=fmean(current['minute_rps'])
    eligible=[]
    for option in options[1:]:
        if 'error' in option:
            continue
        # Absolute rps: the mean is proportional to requests gained over the interval.
        gains=[new-old for new,old in zip(option['minute_rps'],current['minute_rps'])]
        mean=fmean(gains)
        error=stdev(gains)/math.sqrt(len(gains))
        option['minute_gain']=dict(rps=gains, mean_rps=mean, standard_error_rps=error,
            mean_relative=mean/base if base else None,
            eligible=mean>REQUIRED_STANDARD_ERRORS*error)
        selectable=option['minute_gain']['eligible']
        if min_relative_gain:
            relative=option['minute_gain']['mean_relative']
            passes=relative is not None and relative>=min_relative_gain
            option['minute_gain']['passes_min_relative_gain']=passes
            selectable=selectable and passes
        if selectable:
            eligible.append(option)
    best=max(eligible, key=_preference, default=None)
    return dict(action='switch' if best else 'hold', selected=best or current,
                candidates=options, rule='minute_standard_error',
                required_standard_errors=REQUIRED_STANDARD_ERRORS, slices=len(minutes))
