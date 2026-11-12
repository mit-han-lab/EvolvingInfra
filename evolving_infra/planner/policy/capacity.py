# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Saturated service capacity of a P/D role map from the frozen per-worker fits.

This predicts aggregate service capacity, not request finish times. Measured roles retain their own coefficients;
unobserved roles use the median prediction of the explicitly listed same-run donors. No output-length oracle is
consumed.
"""
import math
from statistics import median

from evolving_infra.planner.utils import positive

# Share of a D worker's KV capacity that its running batch may fill.
KV_FRACTION = .95


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


def decode_capacity(profile, worker, work, experts, top_k, population_per_d=None):
    cap = min(int(worker['max_running_requests']),
              math.floor(KV_FRACTION*worker['kv_capacity_tokens']/positive(work['context'], 'context')))
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
    # The fitted feature, fit.steps.expert_fraction, uses np.expm1, which runs Intel SVML on AVX-512 hosts and can
    # differ from math.expm1 in the last bit; sharing one function would change either the fits or the decisions there.
    fraction = 1. if top_k == experts else -math.expm1(
        cap*profile['draft_tokens']*math.log1p(-top_k/experts))
    step = positive(coeff[0]+coeff[1]*cap+coeff[2]*fraction, 'D step seconds')
    acceptance = positive(work['acceptance'], 'observed acceptance')
    return dict(batch=cap, step_s=step, duty_fraction=duty,
                rps=duty*cap*acceptance/(step*positive(work['output'], 'output')),
                supported_batch=cap in profile['batch_support'])


def capacities(workers, roles, work, *, experts, top_k, decode_population=None,
               onboarding=None, terms=None, numbers_only=False):
    """P sum assumes balancing; RR D is constrained by the slowest D.

    An optional measured decode population bounds average batch size. This is a
    quasi-steady estimate, not an inference that every engine runs full batches.
    Both bound and no-bound predictions should be recorded in decisions.

    A worker's term (P seconds or D rps, and its evidence) depends on the role
    map only through its role and, for D, the D count. `terms` (a dict) keeps
    the terms across calls with the same workers, work, decode_population,
    experts, top_k and numbers_only, keyed for D by the D count, for P by the
    identity of its onboarding observation, which must outlive `terms`. With numbers_only the result has no evidence ('workers' is None).
    """
    if set(workers) != set(roles) or set(roles.values()) != {'prefill', 'decode'}:
        raise ValueError('Complete split role map required')
    terms = {} if terms is None else terms
    p_capacity, d_rates, evidence = 0., [], {}
    d_count = sum(r == 'decode' for r in roles.values())
    for key, worker in workers.items():
        role = roles[key]
        restore = (onboarding or {}).get(key) if role == 'prefill' else None
        term = (key, role, id(restore)) if role == 'prefill' else (key, role, d_count)
        if term not in terms:
            profiles = worker[role]['models']
            if not profiles:
                raise ValueError('Missing measured model or declared role prior')
            if role == 'prefill':
                seconds = median([prefill_seconds(model, work, restore) for model in profiles])
                terms[term] = seconds, None if numbers_only else dict(
                    role=role, seconds=seconds, origin=worker[role]['origin'], onboarding=restore)
            else:
                points = [decode_capacity(model, worker, work, experts, top_k,
                            None if decode_population is None else decode_population/d_count)
                          for model in profiles]
                rate = median(p['rps'] for p in points)
                terms[term] = rate, None if numbers_only else dict(
                    role=role, rps=rate, operating_points=points, origin=worker[role]['origin'])
        value, evidence[key] = terms[term]
        if role == 'prefill':
            p_capacity += 1/value
        else:
            d_rates.append(value)
    d_capacity = d_count*min(d_rates)
    return dict(prefill_rps=p_capacity, decode_rr_rps=d_capacity,
                system_rps=min(p_capacity,d_capacity), workers=None if numbers_only else evidence,
                population_bound=decode_population)
