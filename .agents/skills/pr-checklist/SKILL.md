---
name: pr-checklist
description: Check an Evolving Infra change and its pull request description against the project's contribution rules. Use before opening a pull request (drafts included, e.g. `gh pr create`), before pushing review fixes, or to review someone else's pull request.
---

# PR Checklist

**Inputs**: a branch (or an open pull request), and the person who owns it.
**Returns**: a report with one entry per section below (passes, fails, could not verify),
the follow-up work, the decisions that are the owner's, and the filled pull request
description. As the author, fix what fails and open the pull request only when every section
passes. As a reviewer, report; do not change the branch.

## 0. Scope

```bash
git fetch origin && git log --oneline origin/main..HEAD && git diff --stat origin/main...HEAD
```

- **One purpose.** If the commits serve unrelated purposes, split them into one branch and
  one pull request each before going on.
- **Validation still running.** A claim whose run has not finished is not a claim yet. Wait
  for the result, or open a draft and say in Validation which run is pending.
- Ask the owner for anything only they know: the issue this implements (none is acceptable
  for a small change), and who the owner is.

## 1. Design fit

- **Shared paths stay small.** Changes to the scheduler (`engine/decoupled_engine/srt/managers/`),
  the symmetric P/D path (`srt/disaggregation/`), the host KV tier (`srt/mem_cache/`), router
  placement (`router/core/src/scheduler/`) and the planner's decision rule (`planner/policy/`)
  reach every deployment. Keep them minimal and say in Details why each is needed.
- **Defaults are decisions.** A new flag has a default that works without it. A feature of
  the paper's configuration goes in the `paper` presets
  (`srt/arg_groups/symmetric_defaults.py`, `apply_symmetric_pd_preset` in
  `router/core/src/config/cli.rs`), with a test, not in every launch command.
- **Engine code reads resolved settings**: `resolution_result(server_args, field)`, never the
  raw field, for anything resolution can set.
- **No environment-specific values in code**: host names, cluster paths, NIC names, account
  names. They belong in a config or are detected (`launch/hardware.py`).
- **Agent files** (`AGENTS.md`, a skill, the template): follow
  [`docs/contributing/agent-instructions.md`](../../../docs/contributing/agent-instructions.md)
  and run `python tools/check_agent_files.py`.

## 2. Testing and validation

- Run the test commands of [`AGENTS.md`](../../../AGENTS.md) for every component the diff
  touches, and the directory of any test the diff adds.
- A bug fix has a test that fails on the base commit for the reported reason and passes with
  the fix. Paste both results.
- New tests assert behavior through public entry points and do not use sleeps, retries or
  loosened thresholds to pass.
- A change to the serving path is run on GPUs, at the smallest level that shows it:
  `evolving-infra serve --config examples/gb300/glm52-1node.yaml` with one request, then
  `examples/gb300/submit.py` for anything that claims throughput or latency.

## 3. Docs

- Every flag, default, environment variable, config key and endpoint the diff touches is in
  `docs/configuration.md` or the component's page, in the same pull request;
  `docs/paper.md` when a paper mechanism moves.
- Docs and comments are plain statements of what the code does: no hedging, no history of
  the change.

## 4. Pull request contents

Fill a copy of [`.github/PULL_REQUEST_TEMPLATE.md`](../../../.github/PULL_REQUEST_TEMPLATE.md),
keep its sections, and create with `gh pr create --body-file <file>`.

- **Overview**: one or two lines, with the issue link when there is one.
- **Claims**: a few concrete bullets. Each is checkable.
- **Validation**: for every claim, the command and its result.
  - Performance claims use a table: configuration (model, nodes, starting split, workload,
    concurrency), baseline, changed result, scored window, and the reproduction command.
    Compare on the same window and the same P/D split; report the planner's switch times
    when the split changes during the run. With repeated runs, give the count and the spread.
  - Benchmarks replay with `replay_context_mode=actual_output`; say so if a run did not.
  - State what was not run (hardware, model, scale).
- **Details**: root cause for a fix (what failed, how it was found, why the change removes
  it). For the P/D path, a short numbered sequence of one request through router, P, transfer
  and D, limited to the steps the change touches.
- **Transcript**: see below.
- **Checklist**: tick what is true; the owner's handle comes from the owner.

### Transcript

1. Take the session or sessions in which this change was made. When a session also holds
   unrelated work, cut the file to the turns of this change first.
2. Scrub each one. Pass every name the tool cannot know with `--redact`: user and account
   names, cluster and host names, private organization and repository names.

   ```bash
   python tools/scrub_transcript.py <session file> -o transcript.md --redact <name> --redact <host>
   ```

   It prints what it replaced. Exit code 1 means a string still looks like a secret: it
   prints the line of each one. Remove the value from `transcript.md`, or rerun with
   `--allow <the first characters it printed>` when the owner confirms it is not a secret.
3. Read through `transcript.md` once for anything private that no pattern catches, and ask
   the owner to confirm before publishing.
4. Publish it as `transcripts/<head branch>.md` on a branch named `agent-logs` (of the
   repository, or of the fork the pull request comes from) and link that file. Several
   sessions go in one file, in order.

## 5. Re-review

Read every comment before changing anything. `gh pr view --comments` omits inline comments:

```bash
gh api --paginate "repos/mit-han-lab/EvolvingInfra/issues/$pr/comments"
gh api --paginate "repos/mit-han-lab/EvolvingInfra/pulls/$pr/reviews"
gh api --paginate "repos/mit-han-lab/EvolvingInfra/pulls/$pr/comments"
```

For each concern decide with the owner: reply, or change the code and say what changed. Then
update the description so Claims and Validation match the new diff.
