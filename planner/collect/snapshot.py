"""Freeze the complete trace records of one interval (token-free), with exact physical worker identity."""
import gzip
import hashlib
import json
import math
from pathlib import Path

from ..common import complete_lines


def records_before(path, deadline, *, since=-math.inf, key='ts', worker_offset_s=None, cursor=None):
    """Rows with since <= row[key] <= deadline and the proof of the file's complete prefix.

    The prefix is fixed when the read starts; later records are never tailed. A fixed worker
    offset moves ts to controller time and keeps the raw value as worker_ts. `cursor` is a dict
    the caller keeps for this file across reads: it holds the hashed prefix and the leading lines
    whose key was below `since`, so a later read with a bound no lower than before parses only
    the rest. A cursor requires a file that only grows by appends, as the trace mirror's copies
    and restore-observations.jsonl do. A lower bound, a replaced file or a last hashed line no
    longer in place restarts the read from byte 0; other changes to bytes already read go
    unnoticed (the proof keeps their first hash).
    """
    path=Path(path)
    status=path.stat()
    state={} if cursor is None else cursor
    identity=(status.st_dev,status.st_ino,key,worker_offset_s)
    rows=[]
    with path.open('rb') as f:
        if state.get('identity')!=identity or since<state['since'] or not _same_tail(f,state):
            state.update(identity=identity,since=since,skip=0,hashed=0,digest=hashlib.sha256(),tail=b'')
        skip,hashed,digest,tail=state['skip'],state['hashed'],state['digest'].copy(),state['tail']
        f.seek(skip)
        position,leading=skip,True
        for line in complete_lines(f,status.st_size):
            if position>=hashed:
                digest.update(line);hashed+=len(line);tail=line
            position+=len(line)
            row=json.loads(line)
            if worker_offset_s is not None:
                if key!='ts' or 'worker_ts' in row:
                    raise ValueError('Clock correction requires raw worker timestamps')
                row['worker_ts']=row['ts']
                row['ts']-=worker_offset_s
            if row[key]<=deadline and since<=row[key]:  # deadline first: a non-number fails on it
                rows.append(row)
            leading=leading and row[key]<since
            if leading:
                skip=position
    state.update(since=since,skip=skip,hashed=hashed,digest=digest,tail=tail)
    return rows,dict(path=str(path),complete_bytes=hashed,sha256=digest.hexdigest())


def _same_tail(f, state):
    """The last line a cursor hashed is still in place."""
    f.seek(state['hashed']-len(state['tail']))
    return f.read(len(state['tail']))==state['tail']


def capture(destination, worker_inputs, telemetry, *, start, end, clock_pad_s,
            worker_clock_offsets_s=None, cursors=None):
    """Copy cutoff-bounded complete records and preserve exact physical identity.

    An optional fixed offset translates worker ts into controller time before
    cutoff selection. Original worker_ts and source prefix hashes are retained;
    durations and controller telemetry are unchanged. The verified residual
    bound trims edges and widens restore uncertainty. The caller must provide
    periodic evidence covering the interval; this is not a clock verifier.
    Returns the manifest and the rows of each written trace file (path -> rows).
    `cursors` (source path -> records_before cursor), kept across captures of
    append-only files, lets a capture read only what earlier ones have not.
    """
    if (not all(math.isfinite(v) for v in (start,end,clock_pad_s))
            or clock_pad_s < 0 or end-start <= 2*clock_pad_s):
        raise ValueError('Invalid learning interval/clock bound')
    if worker_clock_offsets_s is not None:
        if (set(worker_clock_offsets_s)!={w['node'] for w in worker_inputs}
                or any(type(v) not in (int,float) or not math.isfinite(v)
                       for v in worker_clock_offsets_s.values())):
            raise ValueError('Invalid worker clock offsets')
    cursors={} if cursors is None else cursors
    target=Path(destination)
    target.mkdir(parents=True,exist_ok=False)
    if len({w['node'] for w in worker_inputs})!=len(worker_inputs):
        raise ValueError('Duplicate physical worker')
    manifest=dict(start=start,end=end,clock_pad_s=clock_pad_s,workers={},files={},
                  worker_clock_offsets_s=worker_clock_offsets_s)
    records={}
    for worker in worker_inputs:
        node=worker['node']
        if not node.startswith('node') or not node[4:].isdigit():
            raise ValueError('Canonical physical identity must be nodeN')
        folder=target/'workers'/node/'scheduler';folder.mkdir(parents=True)
        item=dict(worker)
        offset=None if worker_clock_offsets_s is None else worker_clock_offsets_s[node]
        for field in ('steps','requests'):
            source=str(Path(worker[field]));out=folder/(field+'.jsonl')
            rows,proof=records_before(source,end-clock_pad_s,since=start+clock_pad_s,
                                      worker_offset_s=offset,cursor=cursors.setdefault(source,{}))
            data=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows).encode()
            out.write_bytes(data)
            records[str(out)]=rows
            item[field]=str(out);item[field+'_source']=proof;item[field+'_count']=len(rows)
            manifest['files'][str(out.relative_to(target))]=hashlib.sha256(data).hexdigest()
        manifest['workers'][node]=item
    # Keep a short pre-start prefix to bracket the first admitted window.
    rows,proof=records_before(telemetry,end,since=start-10,key='at',
                              cursor=cursors.setdefault(str(Path(telemetry)),{}))
    out=target/'restore.jsonl.gz'
    with gzip.open(out,'wt') as f:
        f.writelines(json.dumps(row)+'\n' for row in rows)
    manifest['telemetry_source']=proof;manifest['telemetry']=str(out)
    manifest['files'][str(out.relative_to(target))]=file_sha256(out)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest,records


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
