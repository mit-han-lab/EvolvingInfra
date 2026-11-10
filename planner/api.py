"""HTTP endpoints of the router and the engines that the planner reads or writes.

Role changes use the router's ``GET``/``PUT /symmetric_pd/roles`` with epoch/revision compare-and-set (the Router
must run with ``--symmetric-pd-external-role-control``); a PUT is never retried.
"""
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request

from .common import http_json, open_url


class RoleClient:
    """The Router's role registry. An HTTP error is journaled, and printed to stderr (controller.log),
    with the router's status and response body before it propagates. Requests never use a proxy."""

    def __init__(self, url, credential, journal, timeout=5):
        self.url = url.rstrip('/') + '/symmetric_pd/roles'
        self.credential, self.journal, self.timeout = credential, journal, timeout

    def request(self, body=None):
        method = 'GET' if body is None else 'PUT'
        req = Request(self.url,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={'Authorization': 'Bearer ' + self.credential,
                     'Content-Type': 'application/json'})
        try:
            with open_url(req, self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                text = error.read().decode(errors='replace')[:2000]
            except Exception as read_error:  # noqa: BLE001 - the status alone still helps
                text = f'<unreadable: {read_error!r}>'
            event = dict(event='role_request_http_error', method=method, status=error.code,
                         reason=str(error.reason), response_body=text)
            self.journal(event)
            print(f'planner controller: {json.dumps(event)}', file=sys.stderr, flush=True)
            raise


def cas_update(client, assignment, desired, journal, intent, failure):
    """One epoch/revision compare-and-set role update from `assignment` to the `desired` roles.

    The intent is journaled before the PUT; ``intent(body)`` gives its fields. An
    old assignment after an ambiguous PUT cannot prove the PUT will never arrive,
    so only a GET follows, never a second PUT, and only the exact next revision
    with the desired roles confirms the update; anything else raises
    ``RuntimeError(failure)``. Returns the router's confirming response.
    """
    body = dict(epoch=assignment['epoch'], revision=assignment['revision'],
                expected=assignment['roles'], desired=desired)
    journal(dict(event='role_update_intent', **intent(body)))
    try:
        response = client.request(body)
    except Exception as error:  # noqa: BLE001 - read back, never retry the mutation
        journal(dict(event='role_update_response_unknown', error=type(error).__name__))
        response = client.request()
    if response['assignment'] != dict(epoch=assignment['epoch'], revision=assignment['revision']+1,
                                      roles=desired):
        raise RuntimeError(failure)
    return response


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
