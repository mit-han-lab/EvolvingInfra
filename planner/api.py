"""HTTP endpoints of the router and the engines that the planner reads or writes.

Role changes use the router's ``GET``/``PUT /symmetric_pd/roles`` with epoch/revision compare-and-set (the Router
must run with ``--symmetric-pd-external-role-control``); a PUT is never retried.
"""
import json
from pathlib import Path
import sys
import urllib.error
from urllib.request import Request

from .common import http_json, open_url


class RoleClient:
    def __init__(self, url, credential, timeout=5):
        self.url = url.rstrip('/') + '/symmetric_pd/roles'
        self.credential, self.timeout = credential, timeout

    def request(self, body=None):
        req = Request(self.url,
            data=None if body is None else json.dumps(body).encode(),
            method='GET' if body is None else 'PUT',
            headers={'Authorization': 'Bearer ' + self.credential,
                     'Content-Type': 'application/json'})
        with open_url(req, self.timeout) as response:
            return json.load(response)


class JournaledRoleClient:
    """RoleClient whose HTTP errors are journaled (and printed to controller.log) with the
    router's status and response body before they propagate; Controller.apply records only the type."""

    def __init__(self, client, journal):
        self.client, self.journal, self.timeout = client, journal, client.timeout

    def request(self, body=None):
        try:
            return self.client.request(body)
        except urllib.error.HTTPError as error:
            try:
                text = error.read().decode(errors="replace")[:2000]
            except Exception as read_error:  # noqa: BLE001 - the status alone still helps
                text = f"<unreadable: {read_error!r}>"
            event = dict(event="role_request_http_error", method="GET" if body is None else "PUT",
                         status=error.code, reason=str(error.reason), response_body=text)
            self.journal(event)
            print(f"planner controller: {json.dumps(event)}", file=sys.stderr, flush=True)
            raise


def served_experts(urls, timeout=5):
    """Routed experts and experts per token of the one model all engines serve.

    Each engine's `/model_info` names its model path; the counts come from that
    model's config.json, which must be readable at the same path on this host.
    """
    counts = set()
    for url in urls:
        with open_url(url.rstrip('/') + '/model_info', timeout) as response:
            path = Path(json.load(response)['model_path'])
        config = json.loads((path / 'config.json').read_text())
        counts.add((config['n_routed_experts'], config['num_experts_per_tok']))
    if len(counts) != 1:
        raise ValueError('Engines do not serve one expert configuration: ' + repr(counts))
    return counts.pop()


def worker_capacities(workers):
    """{node: kv_capacity_tokens, max_running_requests, tp_size} from every engine's ``/server_info``.

    The request cap is the launch flag when the engine was given one; otherwise the engine computed it at
    startup, and it is the sum of the value each data-parallel scheduler reports.
    """
    capacities = {}
    for w in workers:
        info = http_json(w["url"] + "/server_info", timeout=30)
        cap = info.get("max_running_requests")
        if cap is None:
            cap = sum(int(s["effective_max_running_requests_per_dp"]) for s in info["internal_states"])
        capacities[w["node"]] = dict(kv_capacity_tokens=int(info["max_total_num_tokens"]),
                                     max_running_requests=int(cap), tp_size=int(info["tp_size"]))
    return capacities


def backend_loads(router_url):
    """The router's ``/symmetric_pd/backend/loads`` reply (per worker active and unknown P reservations)."""
    return http_json(router_url + "/symmetric_pd/backend/loads", timeout=5)
