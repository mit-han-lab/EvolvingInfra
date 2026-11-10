"""Freeze the complete trace records of one interval (token-free), with exact physical worker identity."""
import gzip
import hashlib
import json
import math
from pathlib import Path


def records_before(path, deadline, *, key='ts', worker_offset_s=None):
    """Read the current complete file prefix once, never tail for later records."""
    path=Path(path)
    limit=path.stat().st_size
    consumed=0;digest=hashlib.sha256()
    with path.open('rb') as f:
        while f.tell()<limit:
            line=f.readline(limit-f.tell())
            if not line.endswith(b'\n'):
                break
            consumed+=len(line);digest.update(line)
            row=json.loads(line)
            if worker_offset_s is not None:
                if key!='ts' or 'worker_ts' in row:
                    raise ValueError('Clock correction requires raw worker timestamps')
                row['worker_ts']=row['ts']
                row['ts']-=worker_offset_s
            if row[key]<=deadline:
                yield row
    return dict(path=str(path),complete_bytes=consumed,sha256=digest.hexdigest())


def capture(destination, worker_inputs, telemetry, *, start, end, clock_pad_s,
            worker_clock_offsets_s=None):
    """Copy cutoff-bounded complete records and preserve exact physical identity.

    An optional fixed offset translates worker ts into controller time before
    cutoff selection. Original worker_ts and source prefix hashes are retained;
    durations and controller telemetry are unchanged. The verified residual
    bound trims edges and widens restore uncertainty. The caller must provide
    periodic evidence covering the interval; this is not a clock verifier.
    """
    if (not all(math.isfinite(v) for v in (start,end,clock_pad_s))
            or clock_pad_s < 0 or end-start <= 2*clock_pad_s):
        raise ValueError('Invalid learning interval/clock bound')
    if worker_clock_offsets_s is not None:
        if (set(worker_clock_offsets_s)!={w['node'] for w in worker_inputs}
                or any(type(v) not in (int,float) or not math.isfinite(v)
                       for v in worker_clock_offsets_s.values())):
            raise ValueError('Invalid worker clock offsets')
    target=Path(destination)
    target.mkdir(parents=True,exist_ok=False)
    if len({w['node'] for w in worker_inputs})!=len(worker_inputs):
        raise ValueError('Duplicate physical worker')
    manifest=dict(start=start,end=end,clock_pad_s=clock_pad_s,workers={},files={},
                  worker_clock_offsets_s=worker_clock_offsets_s)
    for worker in worker_inputs:
        node=worker['node']
        if not node.startswith('node') or not node[4:].isdigit():
            raise ValueError('Canonical physical identity must be nodeN')
        folder=target/'workers'/node/'scheduler';folder.mkdir(parents=True)
        item=dict(worker)
        for field in ('steps','requests'):
            source=Path(worker[field]);out=folder/(field+'.jsonl')
            offset=None if worker_clock_offsets_s is None else worker_clock_offsets_s[node]
            iterator=records_before(source,end-clock_pad_s,worker_offset_s=offset)
            count=0
            with out.open('w') as f:
                while True:
                    try:row=next(iterator)
                    except StopIteration as done:
                        proof=done.value;break
                    if row['ts']<start+clock_pad_s:continue
                    f.write(json.dumps(row,separators=(',',':'))+'\n');count+=1
            item[field]=str(out);item[field+'_source']=proof;item[field+'_count']=count
            manifest['files'][str(out.relative_to(target))]=hashlib.sha256(out.read_bytes()).hexdigest()
        manifest['workers'][node]=item
    out=target/'restore.jsonl.gz';iterator=records_before(telemetry,end,key='at')
    with gzip.open(out,'wt') as f:
        while True:
            try:row=next(iterator)
            except StopIteration as done:
                proof=done.value;break
            # Keep a short pre-start prefix to bracket the first admitted window.
            if row['at']>=start-10:f.write(json.dumps(row)+'\n')
    manifest['telemetry_source']=proof;manifest['telemetry']=str(out)
    manifest['files'][str(out.relative_to(target))]=hashlib.sha256(out.read_bytes()).hexdigest()
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
