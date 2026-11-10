"""Low-frequency role control; no per-request or engine scheduling changes.

Inputs are completed usage and a sampled execution population, not trace future
outputs. Fitted hardware parameters stay fixed. Role PUTs use the Rust registry's
epoch/revision CAS and are never retried after an ambiguous response.
"""
import math
import time

from evolving_infra.planner.policy.capacity import capacities
from evolving_infra.planner.policy.split_policy import decide
from evolving_infra.planner.policy.workload import minute_slices, population_mean, workload_delta
from evolving_infra.planner.router_client import cas_update
from evolving_infra.planner.utils import positive


def node_assignment(snapshot, node_urls):
    """A complete bijection; never infer a physical node from directory names."""
    assignment = snapshot['assignment']
    if len(set(node_urls.values())) != len(node_urls):
        raise ValueError('Physical nodes share a worker URL')
    if set(assignment['roles']) != set(node_urls.values()):
        raise ValueError('Worker registry membership changed')
    roles = {node: assignment['roles'][url] for node, url in node_urls.items()}
    if set(roles.values()) != {'prefill', 'decode'}:
        raise ValueError('Both exclusive roles are required')
    return roles


class SplitController:
    """The decision state between two role changes: proposes a split from the samples, applies it with one
    compare-and-set update, and keeps the post-switch exclusion and the current P observations."""

    def __init__(self, fit, node_urls, initial, *, experts, top_k, fit_digest=None, exclusion_s=120.,
                 slice_secs=60., switch_standard_errors=2., observation_gap_max_secs=15.):
        self.workers = fit['workers']
        self.urls = dict(node_urls)
        if set(self.workers) != set(self.urls):
            raise ValueError('Fit and physical worker mapping differ')
        node_assignment(initial, self.urls)
        self.assignment = initial['assignment']
        self.previous = initial['completed_workload']
        self.experts, self.top_k = experts, top_k
        self.fit_digest = fit_digest
        # Seconds after a role update finishes draining that prediction inputs skip; the
        # caller keeps it short enough to leave two slices of its decision period.
        self.exclusion_s = positive(exclusion_s, 'post-switch exclusion')
        self.slice_secs, self.switch_standard_errors = slice_secs, switch_standard_errors
        self.observation_gap_max_secs = observation_gap_max_secs
        self.clock = time.monotonic
        self.prediction_excluded_from = None
        self.prediction_resume_at = None
        self.prediction_awaiting_drain = bool(initial.get('draining'))
        self.onboarding = {}

    def update_onboarding(self, snapshot, report, journal):
        """Install current-epoch observations separately from the frozen fit.

        A failed/missing worker retains its last valid observation, or its
        calibrated fallback. Old revisions never cross role transitions.
        """
        roles = node_assignment(snapshot, self.urls)
        if (snapshot['assignment'] != self.assignment
                or report['assignment'] != self.assignment
                or self.prediction_resume_at is not None
                or self.prediction_awaiting_drain or snapshot.get('draining')):
            raise ValueError('Stale or transitional onboarding observation')
        if self.fit_digest is not None and report.get('compute_model_sha256') != self.fit_digest:
            raise ValueError('Frozen hardware calibration changed')
        updates = report['workers']
        if any(roles.get(node) != 'prefill' for node in updates):
            raise ValueError('Onboarding update includes a non-P worker')
        for node,observation in updates.items():
            if observation['node'] != node:
                raise ValueError('Onboarding physical node differs from observation')
            if node in self.onboarding and observation['start'] < self.onboarding[node]['end']:
                raise ValueError('Overlapping or old onboarding update')
        self.onboarding.update(updates)
        journal(dict(event='onboarding_updated', report=report))

    def prediction_observation(self, snapshot, at, journal):
        """Exclude post-switch samples from prediction, never from raw metrics.

        Router draining includes both terminal requests and session-reference
        cleanup. Start the exclusion clock when completion is first observed.
        The first snapshot at/after the deadline establishes a new counter and
        population boundary. It is not itself a prediction interval. Completed
        usage is selected by completion time, not by request arrival time.
        """
        node_assignment(snapshot, self.urls)
        if snapshot['assignment'] != self.assignment:
            raise ValueError('Role state changed externally')
        if not math.isfinite(at):
            raise ValueError('Invalid prediction observation clock')
        if self.prediction_excluded_from is not None and at < self.prediction_excluded_from:
            raise ValueError('Prediction observation predates role confirmation')
        if snapshot.get('draining'):
            self.prediction_awaiting_drain = True
            if self.prediction_excluded_from is None:
                self.prediction_excluded_from = at
            return 'excluded'
        if self.prediction_awaiting_drain:
            self.prediction_awaiting_drain = False
            if self.prediction_excluded_from is None:
                self.prediction_excluded_from = at
            self.prediction_resume_at = at + self.exclusion_s
            journal(dict(event='role_drain_observed_complete', assignment=self.assignment,
                         prediction_resume_after_monotonic=self.prediction_resume_at))
        if self.prediction_resume_at is None:
            return 'included'
        if at < self.prediction_resume_at:
            return 'excluded'
        self.previous = snapshot['completed_workload']
        journal(dict(event='prediction_sampling_resumed',
                     excluded_from_monotonic=self.prediction_excluded_from,
                     earliest_resume_monotonic=self.prediction_resume_at,
                     actual_resume_monotonic=at, assignment=self.assignment))
        self.prediction_excluded_from = self.prediction_resume_at = None
        return 'reset'

    def propose(self, samples):
        """Decide from the observations that start at the counter boundary.

        The whole interval is recorded; candidates are compared per slice.
        """
        if (self.prediction_resume_at is not None or self.prediction_awaiting_drain
                or any(row['snapshot'].get('draining') for row in samples)):
            raise ValueError('Post-switch prediction window has not restarted')
        snapshot = samples[-1]['snapshot']
        roles = node_assignment(snapshot, self.urls)
        if snapshot['assignment'] != self.assignment:
            raise ValueError('Router incarnation/revision/roles changed externally')
        if samples[0]['snapshot']['completed_workload'] != self.previous:
            raise ValueError('Observations do not start at the counter boundary')
        population = population_mean(samples, max_gap_s=self.observation_gap_max_secs)
        work = workload_delta(self.previous, snapshot['completed_workload'])
        minutes = []
        for number, part in enumerate(minute_slices(samples, self.slice_secs)):
            try:
                minutes.append(dict(
                    work=workload_delta(part[0]['snapshot']['completed_workload'],
                                        part[-1]['snapshot']['completed_workload']),
                    decode_population=population_mean(part, max_gap_s=self.observation_gap_max_secs)['potential']))
            except ValueError as error:
                raise ValueError(f'Workload slice {number}: {error}') from error
        kwargs = dict(experts=self.experts, top_k=self.top_k, onboarding=self.onboarding)
        decision = decide(self.workers, roles, work, **kwargs, minutes=minutes,
            decode_population=population['potential'], required_standard_errors=self.switch_standard_errors)
        for candidate in decision['candidates']:
            if 'estimate' in candidate:
                candidate['saturated_estimate'] = capacities(
                    self.workers, candidate['roles'], work, **kwargs)
        decision.update(workload=work, population=population, minutes=minutes,
                        before=self.assignment)
        return decision

    def apply(self, snapshot, decision, client, journal):
        """Hold, or switch with one compare-and-set role update (`cas_update`)."""
        if snapshot['assignment'] != self.assignment or decision['before'] != self.assignment:
            raise ValueError('Stale decision')
        if decision['action'] == 'switch':
            if snapshot.get('draining') or self.prediction_awaiting_drain:
                raise ValueError('Previous role change is still draining')
            selected = decision['selected']['roles']
            desired = {url: selected[node] for node,url in self.urls.items()}
            response = cas_update(client, self.assignment, desired, journal,
                                  lambda body: dict(body=body, decision=decision),
                                  'Role update unconfirmed or changed concurrently')
            self.assignment = response['assignment']
            self.onboarding = {}
            self.prediction_excluded_from = self.clock()
            self.prediction_resume_at = self.prediction_excluded_from + self.exclusion_s
            self.prediction_awaiting_drain = bool(response.get('draining'))
            journal(dict(event='role_update_confirmed', assignment=self.assignment,
                         prediction_excluded_from_monotonic=self.prediction_excluded_from,
                         prediction_resume_after_monotonic=self.prediction_resume_at,
                         prediction_exclusion_s=self.exclusion_s))
        else:
            journal(dict(event='role_update_hold', decision=decision))
        self.previous = snapshot['completed_workload']
        return self.assignment

    def skip(self, snapshot, reason, journal):
        node_assignment(snapshot, self.urls)
        if snapshot['assignment'] != self.assignment:
            raise ValueError('Role state changed externally')
        # Advance the counter boundary with the skipped interval, rather than
        # combining its work with the next interval's population.
        self.previous = snapshot['completed_workload']
        journal(dict(event='role_update_hold_unusable_window', reason=reason,
                     assignment=self.assignment))

    def adopt(self, snapshot, at, journal):
        """Take over an assignment changed outside this controller after the fit.

        Another client changed the roles, a PUT whose reply was lost was applied
        late, or the router restarted (new epoch, counters from zero). The new
        roles are treated like a confirmed switch of this controller: current P
        observations are dropped, the counter boundary moves to this snapshot,
        and prediction samples restart after the drain and the exclusion.
        """
        node_assignment(snapshot, self.urls)
        journal(dict(event='external_role_change_adopted', before=self.assignment,
                     after=snapshot['assignment'], draining=snapshot.get('draining')))
        self.assignment = snapshot['assignment']
        self.previous = snapshot['completed_workload']
        self.onboarding = {}
        self.prediction_excluded_from = at
        self.prediction_resume_at = at + self.exclusion_s
        self.prediction_awaiting_drain = bool(snapshot.get('draining'))
