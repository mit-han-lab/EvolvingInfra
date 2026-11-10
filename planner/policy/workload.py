"""The workload a decision scores: completed usage between two router snapshots, the decode population, and the
slices of an interval (``--slice-secs``, 60 s by default). Inputs are completed usage and a sampled execution population, never trace future
outputs.
"""
from bisect import bisect_left
import math

from evolving_infra.planner.utils import positive


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


def population_mean(samples, *, max_gap_s=15):
    """Time-weighted potential D population over the samples' interval, keeping not-yet-ready P separate.

    Each sample applies until the next observation. Missing or unknown records
    disable a decision; they are not zero-load evidence. Pending P supply is
    included as potential work, never called a GPU batch.
    """
    start, end = samples[0]['at'], samples[-1]['at']
    if end <= start or len(samples) < 2:
        raise ValueError('Insufficient population interval')
    counts_by_sample=[]
    for sample in samples:
        pop=sample['snapshot']['decode_population']
        if (pop['fields'] != ['p_completed', 'p_pending', 'unknown']
                or set(pop['workers'])-set(sample['snapshot']['assignment']['roles'])):
            raise ValueError('Unexpected decode population schema/membership')
        values=[0,0,0]
        for counts in pop['workers'].values():
            if len(counts)!=3 or any(type(x) is not int or x<0 for x in counts):
                raise ValueError('Invalid population count')
            values=[x+y for x,y in zip(values,counts)]
        # Include the right endpoint: a newly unknown request must not permit
        # a role change just because it has zero weight in the left-hold mean.
        if values[2]:raise ValueError('Unknown terminal state in decode population')
        counts_by_sample.append(values)
    result = [0., 0., 0.]
    covered = 0.
    for left, right, values in zip(samples, samples[1:],counts_by_sample):
        a, b = left['at'], right['at']
        if not all(math.isfinite(t) for t in (a, b)) or b <= a:
            raise ValueError('Population observation clock is not increasing')
        if a >= end:  # after the last sample's time: a later pair goes back in time and raises above
            continue
        if b-a > max_gap_s:
            raise ValueError('Population observation gap')
        result = [x+(b-a)*y for x,y in zip(result, values)]
        covered += b-a
    return dict(p_completed=result[0]/covered, p_pending=result[1]/covered,
                potential=(result[0]+result[1])/covered, covered_s=covered)


def minute_slices(samples, seconds=60.):
    """Split one interval into floor(duration/seconds) nearly equal slices.

    Boundaries are actual observations shared by adjacent slices, so each
    slice's counters and population cover the same time.
    """
    at = [sample['at'] for sample in samples]
    count = int((at[-1]-at[0])//seconds)
    if count < 2:
        raise ValueError('Interval shorter than two workload slices')
    bounds = [0, *(bisect_left(at, at[0]+k*(at[-1]-at[0])/count) for k in range(1, count)),
              len(at)-1]
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Workload slice without observations')
    return [samples[a:b+1] for a, b in zip(bounds, bounds[1:])]
