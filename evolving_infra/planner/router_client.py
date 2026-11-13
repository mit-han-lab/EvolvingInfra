# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The router endpoints the planner reads or writes.

Role changes use the router's ``GET``/``PUT /symmetric_pd/roles`` with epoch/revision compare-and-set (the Router
must run with ``--symmetric-pd-external-role-control``); a PUT is never retried.
"""
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request

from evolving_infra.planner import envs
from evolving_infra.planner.utils import http_json, open_url


class RoleClient:
    """The Router's role registry. An HTTP error is journaled, and printed to stderr (controller.log),
    with the router's status and response body before it propagates. Requests never use a proxy."""

    def __init__(self, url, credential, journal, timeout=None):
        self.url = url.rstrip('/') + '/symmetric_pd/roles'
        self.credential, self.journal = credential, journal
        self.timeout = envs.EVOLVING_INFRA_PLANNER_ROLE_API_TIMEOUT_SECS if timeout is None else timeout

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


def backend_loads(router_url):
    """The router's ``/symmetric_pd/backend/loads`` reply (per worker active and unknown P reservations)."""
    return http_json(router_url + "/symmetric_pd/backend/loads", timeout=envs.EVOLVING_INFRA_PLANNER_BACKEND_LOADS_TIMEOUT_SECS)
