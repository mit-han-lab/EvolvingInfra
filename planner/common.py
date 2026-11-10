"""Helpers shared across the planner: JSON files, HTTP without proxies, and a number check."""
import json
import math
from pathlib import Path
import urllib.request

# Every planner request goes to a cluster-internal endpoint (router, engines, trace servers), so none of them is
# sent through a proxy configured in the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def open_url(request, timeout):
    """Open a URL string or a ``urllib.request.Request`` without any proxy."""
    return _OPENER.open(request, timeout=timeout)


def http_json(url: str, timeout: float = 10, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with open_url(request, timeout) as response:
        return json.loads(response.read())


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def json_lines(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            # A live writer may not have finished its final record yet.
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('Nonpositive/nonfinite ' + name)
    return value
