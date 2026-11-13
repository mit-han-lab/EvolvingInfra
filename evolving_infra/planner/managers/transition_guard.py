# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Holds before a decision while an earlier role change is still visible on the workers."""
import time


def transition_residual(snapshot, backend_loads):
    """Old plans stay on their original engines after weights change.

    Do not stack further changes while an old role is visibly executing there
    or the router is still finishing its drain and session-reference cleanup.
    This is an observed transition guard, not proof about every GPU kernel.
    """
    roles = snapshot['assignment']['roles']
    population = snapshot['decode_population']['workers']
    draining = snapshot.get('draining', {})
    if (set(backend_loads['workers']) != set(roles) or set(population)-set(roles)
            or set(draining)-set(roles)):
        raise ValueError('Transition observation membership differs')
    residual = {}
    for url, role in roles.items():
        count = (sum(population.get(url, [0,0,0])) if role == 'prefill'
                 else backend_loads['workers'][url]['active_prefill_reqs'])
        if count or url in draining:
            residual[url] = dict(assigned_role=role, old_role_requests=count)
            if url in draining:
                residual[url]['draining_age_s'] = draining[url]
    return residual


def adjusted_loads(loads):
    """A /symmetric_pd/backend/loads reply with unknown P reservations left out of active_prefill_reqs."""
    workers = {}
    for url, view in loads["workers"].items():
        view = dict(view)
        active, unknown = int(view.get("active_prefill_reqs") or 0), int(view.get("unknown_prefill_reqs") or 0)
        view["active_prefill_reqs"] = max(0, active - unknown)
        view["unknown_prefill_reqs_excluded"] = min(active, unknown)
        workers[url] = view
    return {**loads, "workers": workers}


class TransitionGuard:
    """A hold on one condition blocks decisions for at most max_wait_s (--transition-hold-max-secs; then logged);
    drains never."""

    def __init__(self, max_wait_s, clock=time.monotonic):
        self.max_wait_s, self.clock, self.since = float(max_wait_s), clock, {}

    def hold(self, kind, blocked, detail, journal, overridable=True):
        now = self.clock()
        if not blocked:
            self.since.pop(kind, None)
            return False
        waited = now - self.since.setdefault(kind, now)
        if overridable and waited >= self.max_wait_s:
            journal(dict(event="transition_guard_max_wait", kind=kind, waited_s=round(waited, 3),
                         max_wait_s=self.max_wait_s, detail=detail))
            return False
        return True


def residual_drains(residual):
    return any("draining_age_s" in row for row in residual.values())
