"""The decision rule: move to the P count whose minute-by-minute gain is consistent and largest."""
import math
from statistics import fmean, stdev

from .capacity import capacities

REQUIRED_STANDARD_ERRORS = 2.


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


def decide(workers, roles, work, *, minutes, experts, top_k, decode_population=None, onboarding=None):
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
    Each worker's capacity term is computed once for the whole interval and once
    per slice (`capacities` terms), whatever the number of candidates.
    """
    if len(workers) < 2:
        raise ValueError('P/D role control requires at least two model nodes')
    if len(minutes) < 2:
        raise ValueError('At least two workload slices are required')
    kwargs = dict(experts=experts, top_k=top_k, onboarding=onboarding)
    interval, by_minute = {}, [{} for _ in minutes]
    options=[dict(changed=[], roles=dict(roles)),
             *_role_maps(workers, roles, work, decode_population=decode_population, terms=interval, **kwargs)]
    for option in options:
        if 'error' in option:
            continue
        try:
            option['estimate']=capacities(workers,option['roles'],work,
                                          decode_population=decode_population,terms=interval,**kwargs)
            option['minute_rps']=[capacities(workers,option['roles'],m['work'],decode_population=m['decode_population'],
                                             terms=t,numbers_only=True,**kwargs)['system_rps']
                                  for m,t in zip(minutes,by_minute)]
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
        if option['minute_gain']['eligible']:
            eligible.append(option)
    best=max(eligible, key=_preference, default=None)
    return dict(action='switch' if best else 'hold', selected=best or current,
                candidates=options, rule='minute_standard_error',
                required_standard_errors=REQUIRED_STANDARD_ERRORS, slices=len(minutes))


def _preference(option):
    """The order of `decide`: largest mean minute gain, then fewest changes."""
    return option['minute_gain']['mean_rps'], -len(option['changed'])
