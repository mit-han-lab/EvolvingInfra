"""Scaffolding shared by the planner tests.

tests/planner has no __init__.py, so pytest's default (prepend) import mode puts this directory on sys.path and the
test modules import this one as `planner_support`.
"""
import copy
import gzip
import json

# A measured P model (four terms, warm-up packing and restore ratio), a measured D model, and an interval's workload
# with two minute slices.
PREFILL_MODEL = dict(coefficients=[.01, .0001, 1e-9, .00001], tokens_per_step=4096., restored_per_cached_token=.5)
DECODE_MODEL = dict(coefficients=[.02, .001, .01], draft_tokens=6, batch_support=list(range(1, 129)))
WORK = dict(uncached=8192., cached=40000., pairs=8192*44096., output=800., context=50000., acceptance=3.)
MINUTES = [dict(work=WORK, decode_population=30), dict(work=dict(WORK, output=900.), decode_population=30)]


def measured_workers(nodes=('0', '1', '2', '3')):
    """A fit report's workers: PREFILL_MODEL and DECODE_MODEL (one copy, shared by all nodes), 128 running requests
    and 2 M KV tokens on every node."""
    p, d = copy.deepcopy(PREFILL_MODEL), copy.deepcopy(DECODE_MODEL)
    return {n: dict(prefill=dict(origin='measured', models=[p]), decode=dict(origin='measured', models=[d]),
                    max_running_requests=128, kv_capacity_tokens=2000000) for n in nodes}


# Completed-workload counters after a 600-s window in which 100 requests completed.
WINDOW_WORKLOAD = dict(requests=100, input_tokens=1200, uncached_tokens=500, causal_attention_pairs=5000,
                       output_tokens=400, verified_output_tokens=400, verify_steps=100)


def router_snapshot(roles, population, *, epoch='e', revision=0, completed=None):
    """A router snapshot: role per url, completed-workload counters (zero unless given) and the decode population
    [p_completed, p_pending, unknown] per D url."""
    if completed is None:
        completed = dict(requests=0, input_tokens=0, uncached_tokens=0, causal_attention_pairs=0, output_tokens=0,
                         verified_output_tokens=0, verify_steps=0)
    return dict(assignment=dict(epoch=epoch, revision=revision, roles=roles), completed_workload=completed,
                decode_population=dict(fields=['p_completed', 'p_pending', 'unknown'], workers=population))


def router_samples(first, last, start=100., seconds=600, step=3, wall=None):
    """Router observations every `step` seconds: snapshot `last` with completed counters growing linearly from those
    of `first`; `wall`, if given, is the wall-clock time of the first observation."""
    a, b = first['completed_workload'], last['completed_workload']
    out = []
    for i in range(seconds//step+1):
        s = copy.deepcopy(last)
        s['completed_workload'] = {k: a[k]+(b[k]-a[k])*i*step//seconds for k in b}
        out.append(dict(at=start+i*step, snapshot=s) if wall is None else
                   dict(at=start+i*step, wall=wall+i*step, snapshot=s))
    return out


def restore_record(at, tokens, node=0):
    """A restore-observations record as observe.Telemetry writes it: P node `node` scraped from `at` to `at` + 0.01 s
    with its kv-pool restore counter at `tokens` (None: not exported yet, as before the first restore)."""
    metrics = '# HELP sglang:hicache_backup_bytes_total Bytes.\n'
    if tokens is not None:
        metrics += f'sglang:load_back_tokens_total{{model_name="m",pool="kv"}} {tokens}\n'
    return dict(at=at+.01, nodes=[dict(node=node, role='prefill', at=at, errors={}, metrics=metrics)])


def write_telemetry(path, records):
    """Write restore-observations records as gzip JSON lines, the snapshot telemetry the counter readers take."""
    with gzip.open(path, 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in records)
