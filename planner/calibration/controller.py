"""Low-frequency role control; no per-request or engine scheduling changes.

Inputs are completed usage and a sampled execution population, not trace future
outputs. Fitted hardware parameters stay fixed. Role PUTs use the Rust registry's
epoch/revision CAS and are never retried after an ambiguous response.
"""
from bisect import bisect_left
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .dynamic import capacities, decide, positive, workload_delta
from .guards import floor_restore_ratios

SLICE_S = 60.


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
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)


def served_experts(urls, timeout=5):
    """Routed experts and experts per token of the one model all engines serve.

    Each engine's `/model_info` names its model path; the counts come from that
    model's config.json, which must be readable at the same path on this host.
    """
    counts = set()
    for url in urls:
        with urlopen(url.rstrip('/') + '/model_info', timeout=timeout) as response:
            path = Path(json.load(response)['model_path'])
        config = json.loads((path / 'config.json').read_text())
        counts.add((config['n_routed_experts'], config['num_experts_per_tok']))
    if len(counts) != 1:
        raise ValueError('Engines do not serve one expert configuration: ' + repr(counts))
    return counts.pop()


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


def population_mean(samples, start, end, *, max_gap_s=15):
    """Time-weighted potential D population, keeping not-yet-ready P separate.

    Each sample applies until the next, bounded by the next observation. Missing
    or unknown records disable a decision; they are not zero-load evidence.
    Pending P supply is included as potential work, never called a GPU batch.
    """
    if end <= start or len(samples) < 2:
        raise ValueError('Insufficient population interval')
    counts_by_sample=[]
    for sample in samples:
        pop=sample['snapshot']['decode_population']
        if (pop['fields'] != ['p_completed', 'p_pending', 'unknown']
                or set(pop['workers'])-set(sample['snapshot']['assignment']['roles'])):
            raise ValueError('Unexpected decode population schema/membership')
        values=[0,0,0]
        for counts in pop['workers'].values():
            if len(counts)!=3 or any(type(x) is not int or x<0 for x in counts):
                raise ValueError('Invalid population count')
            values=[x+y for x,y in zip(values,counts)]
        # Include the right endpoint: a newly unknown request must not permit
        # a role change just because it has zero weight in the left-hold mean.
        if values[2]:raise ValueError('Unknown terminal state in decode population')
        counts_by_sample.append(values)
    result = [0., 0., 0.]
    covered = 0.
    for left, right, values in zip(samples, samples[1:],counts_by_sample):
        a, b = left['at'], right['at']
        if not all(math.isfinite(t) for t in (a, b)) or b <= a:
            raise ValueError('Population observation clock is not increasing')
        duration = max(0., min(b, end)-max(a, start))
        if not duration:
            continue
        if b-a > max_gap_s:
            raise ValueError('Population observation gap')
        result = [x+duration*y for x,y in zip(result, values)]
        covered += duration
    if not math.isclose(covered, end-start, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Population observations do not cover interval')
    return dict(p_completed=result[0]/covered, p_pending=result[1]/covered,
                potential=(result[0]+result[1])/covered, covered_s=covered)


def minute_slices(samples, seconds=SLICE_S):
    """Split one interval into floor(duration/seconds) nearly equal slices.

    Boundaries are actual observations shared by adjacent slices, so each
    slice's counters and population cover the same time.
    """
    at = [sample['at'] for sample in samples]
    count = int((at[-1]-at[0])//seconds)
    if count < 2:
        raise ValueError('Interval shorter than two workload slices')
    bounds = [0, *(bisect_left(at, at[0]+k*(at[-1]-at[0])/count) for k in range(1, count)),
              len(at)-1]
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Workload slice without observations')
    return [samples[a:b+1] for a, b in zip(bounds, bounds[1:])]


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


class Controller:
    def __init__(self, fit, node_urls, initial, *, experts, top_k, fit_digest=None, exclusion_s=120.,
                 guards=None):
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
        self.clock = time.monotonic
        self.prediction_excluded_from = None
        self.prediction_resume_at = None
        self.prediction_awaiting_drain = bool(initial.get('draining'))
        self.onboarding = {}
        # Opt-in planner guards (evolving_infra.planner.calibration.guards.PlannerGuards); None or
        # all off keeps every decision and journal record unchanged.
        self.guards = guards
        self.last_switch_at = None
        # Whether a switch selected by the queue add-P guard has been confirmed in this
        # run; the queue removal hold (Q-hold) acts only after one. Never reset.
        self.queue_added = False
        # D batch 'measured': the effective KV fraction learned by the last applied
        # decision whose window was memory-bound; None before the first.
        self.decode_kv_fraction = None
        # Whether a periodic decision has been applied (hold or switch); E-hold's 'first_decision' release.
        self.decided = False

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

    def since_switch(self):
        """Seconds since the last confirmed role switch; None before the first."""
        return None if self.last_switch_at is None else self.clock()-self.last_switch_at

    def queue_probe_due(self, observation):
        """Whether a live P_queue observation warrants an off-period decision.

        Only with the queue add-P guard on, while prediction sampling runs, at
        least its minimum time after the last switch, and with the observed
        p95 above its threshold. The decision itself applies every condition.
        """
        guards = self.guards
        if (guards is None or not guards.queue_p95_s or observation is None
                or observation.get('p95_s') is None):
            return False
        if self.prediction_resume_at is not None or self.prediction_awaiting_drain:
            return False
        since = self.since_switch()
        if since is not None and since < guards.queue_min_since_switch_s:
            return False
        return observation['p95_s'] > guards.queue_p95_s

    def propose(self, samples, queue=None, trigger='period', decode_running=None, evict_pressure=None):
        """Decide from the observations that start at the counter boundary.

        The whole interval is recorded; candidates are compared per slice.
        `queue` (live P_queue observation) and `trigger` matter only with
        guards on; both are recorded under the decision's 'guards'. With the
        queue removal hold on, `queue_added` (a confirmed Q' switch earlier in
        the run) is passed too. With the measured D batch, `decode_running`
        (DecodeBatchMonitor.running over the samples' wall-clock window) and
        the window's p_completed population are passed with the last learned
        KV fraction. With the eviction-pressure hold on, `evict_pressure`
        (HostEvictionMonitor.observe at the decision) is passed; None means no
        signal.
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
        population = population_mean(samples, samples[0]['at'], samples[-1]['at'])
        work = workload_delta(self.previous, snapshot['completed_workload'])
        minutes = []
        for number, part in enumerate(minute_slices(samples)):
            try:
                minutes.append(dict(
                    work=workload_delta(part[0]['snapshot']['completed_workload'],
                                        part[-1]['snapshot']['completed_workload']),
                    decode_population=population_mean(part, part[0]['at'], part[-1]['at'])['potential']))
            except ValueError as error:
                raise ValueError(f'Workload slice {number}: {error}') from error
        kwargs = dict(experts=self.experts, top_k=self.top_k, onboarding=self.onboarding)
        if self.guards is None or not self.guards.enabled:
            decision = decide(self.workers, roles, work, **kwargs, minutes=minutes,
                decode_population=population['potential'])
        else:
            observation = None
            if self.guards.queue_p95_s and queue is not None:
                observation = dict(queue, since_switch_s=self.since_switch())
            hold = dict(queue_added=self.queue_added) if self.guards.queue_removal_floor_s else {}
            if self.guards.decode_batch_source == 'measured':
                hold['decode_batch'] = dict(running=decode_running, p_completed=population['p_completed'],
                                            population=population['potential'], kv_fraction=self.decode_kv_fraction)
            if self.guards.evict_pressure_hold:
                hold['evict_pressure'] = evict_pressure
                if self.guards.evict_pressure_release == 'first_decision':
                    hold['first_decision'] = not self.decided
            decision = decide(self.workers, roles, work, **kwargs, minutes=minutes,
                decode_population=population['potential'], guards=self.guards, queue=observation, **hold)
            decision['guards']['trigger'] = trigger
            if self.guards.restore_floor:
                kwargs['onboarding'] = floor_restore_ratios(self.onboarding)[0]
            if (decision['guards'].get('decode_batch') or {}).get('applied'):
                kwargs['kv_fraction'] = decision['guards']['decode_batch']['kv_fraction']
        for candidate in decision['candidates']:
            if 'estimate' in candidate:
                candidate['saturated_estimate'] = capacities(
                    self.workers, candidate['roles'], work, **kwargs)
        decision.update(workload=work, population=population, minutes=minutes,
                        before=self.assignment)
        return decision

    def apply(self, snapshot, decision, client, journal):
        """Journal before PUT. Only an exact new revision proves application.

        An old assignment after a timeout cannot prove the original PUT will
        never arrive. Stop on that ambiguity rather than issue a second PUT.
        """
        if snapshot['assignment'] != self.assignment or decision['before'] != self.assignment:
            raise ValueError('Stale decision')
        if decision['action'] == 'switch':
            if snapshot.get('draining') or self.prediction_awaiting_drain:
                raise ValueError('Previous role change is still draining')
            selected = decision['selected']['roles']
            desired = {url: selected[node] for node,url in self.urls.items()}
            body = dict(epoch=self.assignment['epoch'], revision=self.assignment['revision'],
                        expected=self.assignment['roles'], desired=desired)
            journal(dict(event='role_update_intent', body=body, decision=decision))
            expected = dict(epoch=body['epoch'], revision=body['revision']+1, roles=desired)
            try:
                response = client.request(body)
            except Exception as error:
                journal(dict(event='role_update_response_unknown', error=type(error).__name__))
                response = client.request()  # Read only, never retry the mutation.
            if response['assignment'] != expected:
                raise RuntimeError('Role update unconfirmed or changed concurrently')
            self.assignment = response['assignment']
            self.onboarding = {}
            self.prediction_excluded_from = self.clock()
            self.last_switch_at = self.prediction_excluded_from
            if (decision.get('guards') or {}).get('selected_by') == 'queue_add_prefill':
                self.queue_added = True  # the confirmed switch, not the journaled intent
            self.prediction_resume_at = self.prediction_excluded_from + self.exclusion_s
            self.prediction_awaiting_drain = bool(response.get('draining'))
            journal(dict(event='role_update_confirmed', assignment=self.assignment,
                         prediction_excluded_from_monotonic=self.prediction_excluded_from,
                         prediction_resume_after_monotonic=self.prediction_resume_at,
                         prediction_exclusion_s=self.exclusion_s))
        else:
            journal(dict(event='role_update_hold', decision=decision))
        self.decided = True
        batch = (decision.get('guards') or {}).get('decode_batch')
        if batch and batch.get('learned'):
            self.decode_kv_fraction = batch['kv_fraction']
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
