"""Fixed per-node P/D plan from a completed learning capture; writes new evidence, never old results.

Input: a learning directory with ``configuration.json`` (``nodes[].physical_node``/``host``, optional
``allocation``), ``calibration/final-manifest.json`` and ``calibration/report.json`` from
``python -m evolving_infra.planner calibrate``. Each physical worker gets its own P model (steps,
uncached tokens, causal attention pairs, restored tokens) and D model (step overhead, batch size,
active-expert fraction); training/validation split chronologically at 70 %. Only roles with actual
measurements are eligible: no coefficients are invented for an unobserved worker role. The decision is
an approximate capacity model, not a continuous-batching replay.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    from evolving_infra.planner.calibration.nodes import fit_prefill, fit_decode, select_fixed_plan
    from evolving_infra.planner.calibration.trace import load_steps, json_lines

    root = Path(root)
    manifest_path = root/'calibration/final-manifest.json'
    report_path = root/'calibration/report.json'
    manifest = json.loads(manifest_path.read_text())
    original = json.loads(report_path.read_text())
    cfg = json.loads((root/'configuration.json').read_text())
    params = manifest['configuration']
    start, end = manifest['start_unix'], manifest['end_unix']
    if end-start != 1800 or original['elapsed_s'] != 1800:
        raise ValueError('Preserve the completed 1800s learning requirement')
    models = {str(n['physical_node']):dict(host=n['host'], allocation=cfg.get('allocation'),
              max_running_requests=params['max_running_requests'],
              missing_roles=['prefill','decode']) for n in cfg['nodes']}
    p = original['prefill_measured_restore']
    inputs = {str(p):digest(p) for p in [manifest_path,report_path,root/'configuration.json']}
    for node in sorted({w['node'] for w in p['windows']}):
        logical = int(str(node).removeprefix('node'))
        key = str(cfg['nodes'][logical]['physical_node'])
        models[key]['prefill'] = fit_prefill([w for w in p['windows'] if w['node']==node],
                                            split_time=start+.7*(end-start))
        models[key]['missing_roles'].remove('prefill')
    for path in manifest['decode_steps']:
        logical = int(Path(path).parents[1].name.removeprefix('node'))
        key = str(cfg['nodes'][logical]['physical_node'])
        models[key]['decode'] = fit_decode(load_steps([path],start,end), start=start,end=end,
                                            experts=params['experts'],top_k=params['top_k'])
        models[key]['kv_capacity_tokens'] = manifest['kv_capacity_tokens']
        models[key]['kv_capacity_scope'] = 'minimum observed across the original D role workers'
        models[key]['missing_roles'].remove('decode')
        inputs[path] = digest(path)
    outputs=[]
    for path in manifest['prefill_requests']:
        inputs[path] = digest(path)
        for row in json_lines(path):
            if start <= row['ts'] <= end and (row.get('finish_reason') or {}).get('type')=='length':
                n=row['requested_output_len']
                if type(n) is not int or n<=0:raise ValueError('Invalid original output length')
                outputs.append(n)
    workload = dict(request_features=p['aggregate_request_features'],
                    output_tokens=sum(outputs)/len(outputs),
                    mean_context=original['measurements']['mean_decode_context'],
                    experts=params['experts'],top_k=params['top_k'],min_user_tps=params['min_user_tps'])
    decision=select_fixed_plan(models,**workload)
    return dict(schema_version=1,status='calibrated_roles_only_not_frozen',models=models,
                learning_root=str(root.resolve()),learning_input_sha256=inputs,
                legacy_p_validation_passed=p['passed'],workload=workload,decision=decision,
                calibration_source_sha256={f.name:digest(f) for f in
                    sorted((Path(__file__).resolve().parent/'calibration').glob('*.py'))},
                limitations=decision['limitations']+[
                    'No parameters invented for an unobserved worker role.',
                    'Original failed pooled fit is unchanged. This is a practical exploratory model.',
                    'Models are bound to physical hosts/allocation; new hosts require explicit calibration.'])


def add_arguments(ap):
    ap.add_argument('--learning',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)


def run(args):
    result=build(args.learning)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(selected=result['decision']['selected'],
                         models={k:dict(missing=v['missing_roles'],
                           errors={r:v[r]['errors'] for r in ['prefill','decode'] if r in v})
                                 for k,v in result['models'].items()}),indent=2))
