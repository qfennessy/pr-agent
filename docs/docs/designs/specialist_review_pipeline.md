# Specialist review pipeline for Cocos Story

!!! note "Status: proposed, not implemented"
    This page describes a design for the `qfennessy/pr-agent` fork. The configuration keys and CLI commands below are illustrative. Current PR-Agent releases do not yet provide the file-save watcher, specialist router, or quality-escalation pipeline described here.

## Executive decision

Move useful review earlier than the pull request. Run a small, cheap model after a meaningful file save, much like a quiet pair programmer. Give it narrow specialist instructions and only the context needed for that change. Escalate to a larger open-weight model when the change needs cross-file reasoning, and reserve a frontier model for high-consequence or genuinely ambiguous cases.

The proposed order is:

1. deterministic checks;
2. a roughly 3B code model for fast classification, routing, and local specialist checks;
3. a stronger open-weight model for worktree-wide verification;
4. a frontier model for sensitive paths, high-severity findings, or unresolved disagreement;
5. an independent PR review as the final backstop.

This is a cascade, not a vote. The system should not publish the union of every model's comments. One component gathers and verifies the evidence, removes duplicates and weak claims, and presents only the best findings.

## Why review on file save

The earliest useful review point is when code first exists in the worktree. Waiting for a commit or GitHub pull request adds avoidable latency.

An initial manual assessment of the current Cocos Story corpus gives the following opportunity split. These labels still need independent adjudication and should become explicit corpus metadata before they are used as benchmark truth.

| Earliest useful checkpoint | Verified defects | High severity | Meaning |
| --- | ---: | ---: | --- |
| File save | 17 of 40 | 10 | The defect-bearing file contains enough evidence for a focused check. |
| Coherent worktree | 23 of 40 | 19 | The reviewer needs callers, contracts, tests, lifecycle state, or another changed file. |
| Commit only | 0 of 40 | 0 | A commit is a stable scheduling boundary, but it does not create new code evidence. |
| Pull request only | 0 of 40 | 0 | GitHub metadata can help, but no sampled defect inherently required PR latency. |

The implication is important: file-save review is worthwhile, but it cannot replace worktree-wide review. It may find roughly 42.5% of the current defect set at the first opportunity. The remaining 57.5% require a coherent view of the change.

The existing 16 clean controls are also not enough to validate file-save review. They freeze final PR heads, not the intentionally incomplete states developers produce while editing. A local pilot must capture clean intermediate saves, self-corrected mistakes, and partial worktrees so that interruption noise is measurable.

## Current foundation and missing pieces

PR-Agent already provides useful building blocks, but its current unit of work is a pull request, branch comparison, or supplied unified diff.

| Capability | Current state | Design gap |
| --- | --- | --- |
| Plain-diff provider | Reviews a supplied diff, enriches it from the working tree, and can emit structured JSON. | No file-save event, snapshot identity, debounce, or stale-result cancellation. |
| Local Git provider | Reviews committed branch-to-branch changes without a hosted PR. | Requires a clean worktree and therefore starts too late for pair review. |
| Dynamic context | Expands a hunk toward its enclosing function or class. | Does not select a caller, contract, test, or historical defect pattern by specialist need. |
| Agent Skills | Injects reusable review guidance into a single model call. | Enabled skills are loaded together; there is no router-driven progressive disclosure. |
| `model` and `fallback_models` | Retries another model when a call fails. | This is availability failover, not candidate verification or risk escalation. |
| Multiple CI runs | Can produce separate comments by using distinct persistent comment identities. | Results are independent; there is no shared evidence, deduplication, or final adjudicator. |

The shortest implementation path is to reuse plain-diff parsing and Agent Skills while adding a local review-snapshot protocol and an explicit orchestrator.

## Developer experience

A normal save should feel like pair coding, not a miniature pull-request ceremony:

1. The editor or coding agent reports that `services/functions/src/billing/refund.ts` was saved.
2. The watcher waits briefly for related writes, computes the semantic delta, and assigns a content hash.
3. Deterministic path rules mark the change as billing-sensitive. The small router also classifies it as TypeScript, external-provider, and tenant-scoped.
4. The same small model runs with the billing and tenant-isolation specialist prompts. At most two specialists run on a save.
5. If no concrete issue is found, the system stays silent.
6. A strong file-local finding appears as short-lived local advice with an exact trigger and line. It is not posted to GitHub.
7. A cross-file or consequential candidate is queued for the worktree verifier. Sensitive changes are guaranteed broader review at the next stable checkpoint even when the small model reports confidence; they do not launch a frontier call on every save.
8. If a later save removes the problem, the old result is discarded because its content hash is stale.

The local interface should distinguish three states:

- **Fast check:** an immediate, provisional observation from a small specialist.
- **Verified:** a finding confirmed against broader worktree context.
- **Coverage unavailable:** a model or context step failed; this must never be shown as a clean result.

During the initial rollout, fast checks remain in shadow mode. Developer-visible file-save advice should turn on only after the local precision and interruption gates are met.

## Proposed pipeline

```text
file written
    |
    v
snapshot + deterministic checks
    |
    v
small classifier/router (about 3B)
    |
    +--> selected small specialist(s) --> no supported issue --> stop silently
    |                                      |
    |                                      +--> local fast check
    |
    +--> risk/context request
               |
               v
       open-weight worktree verifier
               |
               +--> rejected or duplicate --> stop
               |
               +--> verified ordinary issue --> local finding
               |
               +--> sensitive, severe, or disputed
                              |
                              v
                       frontier reviewer
                              |
                              v
                    cleanup + ranked output
```

### Review stages

| Stage | Trigger | Primary context | Default compute | Target latency | Output |
| --- | --- | --- | --- | --- | --- |
| File-save pair review | Meaningful file write, debounced | Changed hunk, enclosing symbol, local rules, cheap retrieved context | Small model with one or two specialist prompts | p95 under 3 seconds; cancel at 5 seconds | Local, ephemeral advice only |
| Worktree review | 15 seconds idle, coding-agent stop, or manual command | Entire dirty diff plus selected callers, contracts, tests, and task intent | Stronger open-weight model | p95 under 30 seconds | Local verified findings |
| Pre-commit review | Staged snapshot | Staged diff, deterministic results, unresolved local findings | Deterministic checks plus open verifier | p95 under 60 seconds | Advisory by default; block only deterministic policy failures |
| Post-commit or pre-push review | Immutable commit/branch delta | Full coherent change and commit intent | Open verifier plus selective frontier escalation | Asynchronous, under 5 minutes | Durable local report |
| Pull-request review | Hosted PR event | PR diff, description, issue, CI, and final branch state | Independent configured reviewer | Existing PR latency budget | At most three verified comments |

Latency numbers are rollout targets, not claims about a particular model or provider. They must be measured on the selected endpoint and hardware.

## The small model's jobs

The small tier does more than scan for bugs. It is used at both ends of the pipeline: first to organize work, and later to remove noise.

| Role | What the small model does |
| --- | --- |
| Change classification | Labels the change as UI, tests, refactor, dependency update, schema change, auth, tenant isolation, and other relevant categories. |
| Risk routing | Identifies changes that deserve an expensive model, including security boundaries, migrations, concurrency, billing, and destructive operations. Deterministic policy can only raise this risk, never be overruled by the model. |
| Diff prioritization | Ranks files and hunks so a larger model reads the important 10–20% first. |
| Context selection | Requests relevant architecture decision records, coding rules, interfaces, callers, and nearby tests for each changed hunk. |
| Mechanical policy checking | Detects likely violations such as a missing `tenantId` filter, unhandled error, undocumented Python function, or forbidden Firestore pattern. Deterministic tools remain the authority where a rule can be encoded. |
| Review-comment cleanup | Classifies proposed comments as actionable, stylistic, duplicate, unsupported, stale, or low-confidence. |
| Change summarization | Produces a structured fact summary for larger reviewers, not polished prose for humans. |
| Regression matching | Compares the delta with patterns from previously fixed defects and accepted review comments, using only training-period examples. |

One small model can perform several of these roles with different prompts and context packs. “Specialist agent” means an isolated responsibility and evidence contract; it does not require a separate set of weights or a long-lived autonomous process.

## Specialist catalog

The router should select only specialists relevant to the change. Fan-out to every specialist on every save would waste time and amplify false positives.

| Specialist | Deterministic route signals | Main questions | Escalate when |
| --- | --- | --- | --- |
| Authentication and authorization | Auth middleware, session/token code, route guards, security rules | Can an unauthenticated or wrong-role user reach this behavior? | Any plausible boundary bypass or missing enforcement |
| Tenant isolation and privacy | `tenantId`, user-owned records, storage paths, logging, analytics | Can data cross tenants or expose personal information? | Ownership evidence is absent, conflicting, or distributed across services |
| Data, schema, and migration | Schemas, migrations, backfills, serialization, Firestore writes | Is the change backward-compatible, idempotent, and safe for existing data? | Destructive transform, partial-failure risk, or mixed-version contract |
| Concurrency and reliability | Queues, jobs, retries, locks, async workflows, external calls | Can retries duplicate work, race, leak resources, or leave partial state? | Cross-process reasoning, non-idempotent side effect, or unclear recovery |
| Billing and third-party effects | Payments, refunds, provider callbacks, metering, quotas | Can the change charge twice, misreport usage, or lose reconciliation evidence? | Money movement, destructive provider call, or ambiguous callback ordering |
| API and contract compatibility | Interfaces, API routes, events, types, shared packages | Do producers, consumers, and tests agree on shape and lifecycle? | More than one service or release version is involved |
| Test adequacy and regression | Behavior change without a nearby test, changed assertions, mocks | Does a focused test prove the risky behavior and fail before the fix? | A material finding lacks a reproducible test path |
| UI and client state | React state, forms, loading/error states, accessibility, client caching | Can the user reach a broken, stale, inaccessible, or misleading state? | Server contract or persistent user data is involved |
| Infrastructure and deployment | Terraform, IAM, workflows, secrets, Firebase configuration | Can this change widen access, break rollout, or make rollback unsafe? | IAM, secrets, production deployment, or stateful resource changes |

Language guidance should be small modules under these domain specialists, not separate full review agents for every language. The invariant question is usually more important than whether the file is TypeScript, Python, or Terraform.

## Snapshot and context contract

Every run receives an immutable `ReviewSnapshot` so that results can be cancelled or reproduced:

```json
{
  "snapshot_id": "sha256:<content-and-policy-hash>",
  "event": "file_save",
  "base_revision": "<merge-base-or-index>",
  "changed_paths": ["services/functions/src/billing/refund.ts"],
  "focus_path": "services/functions/src/billing/refund.ts",
  "diff": "<unified diff>",
  "task_intent": "<optional issue or coding-agent task>",
  "deterministic_results": [],
  "policy_version": "specialist-pipeline-v1"
}
```

The context builder works from explicit specialist requests. Its default bundle is:

1. changed hunk and enclosing function, class, or configuration block;
2. repository rules and the selected specialist skill;
3. task or issue intent, when available before a PR exists;
4. relevant interface, type, schema, or architecture decision;
5. one direct caller or consumer;
6. the closest related test;
7. at most one retrieved historical defect pattern during the first experiment arm.

Do not send the whole repository. Keep stable instructions at the front of the prompt for provider caching, put the changed code near the beginning or end, and enforce a measured context budget. Retrieval must exclude the target change, near-duplicate changes, and examples from calibration or holdout periods.

## Finding contract

Specialists return evidence, not prose essays or hidden reasoning. A proposed schema is:

```json
{
  "snapshot_id": "sha256:...",
  "specialist": "tenant_isolation",
  "category": "authorization",
  "severity": "high",
  "location": {"path": "src/file.ts", "line": 42},
  "trigger": "A caller supplies a record id owned by another tenant",
  "observable_impact": "The update reaches a record outside the active tenant",
  "evidence": ["The query filters by id but not tenantId"],
  "suggested_test": "Use two tenants and attempt the update with the second tenant's id",
  "missing_context": [],
  "recommended_route": "frontier",
  "confidence_label": "supported"
}
```

Numeric model confidence is telemetry, not a calibrated probability and not the sole routing signal. A finding cannot be published unless it identifies introduced behavior, an exact location, a concrete trigger, and an observable impact.

## Routing and escalation policy

### Hard routing rules

The following changes bypass a “cheap model says safe” decision and always receive broader review at the next stable checkpoint:

- authentication, authorization, tenant isolation, or personal-data boundaries;
- billing, payments, refunds, metering, or irreversible third-party effects;
- schema migrations, backfills, destructive transforms, or stateful resource deletion;
- concurrency, idempotency, queues, retries, or multi-step durable workflows;
- IAM, secrets, security rules, CI credentials, or production deployment logic;
- a failed deterministic check or a high/critical candidate;
- a cross-service contract change;
- a large or diffuse change above locally measured thresholds.

Path rules, API usage, static analysis, and repository history establish a minimum risk. The small model may raise the route but may not lower that minimum.

### Escalation decisions

| Evidence state | Next action |
| --- | --- |
| No candidate, ordinary path, context complete | Stop silently and record coverage. |
| Strong file-local candidate on an ordinary path | Emit a provisional local fast check; verify on the next worktree-idle run. |
| Candidate needs a caller, contract, or test | Do not guess. Request the context and route to the open-weight verifier. |
| Small specialists disagree | Route their evidence, not their full prose, to the open-weight verifier. |
| Open verifier confirms an ordinary medium issue | Mark it verified locally; retain it for commit review. |
| Sensitive path with no immediate candidate | Run the open verifier on worktree idle and the frontier reviewer after commit or at PR time. |
| High/critical issue or verifier disagreement | Route to the frontier reviewer immediately, subject to the current-stage timeout. |
| Frontier rejects the issue | Suppress it but retain the adjudication for evaluation. |
| Any required model or context step fails | Mark coverage unavailable and retry according to the stage budget. |

The frontier model is not merely the last item in `fallback_models`. It receives the candidate, counter-evidence, deterministic results, and focused repository context, and is asked to adjudicate a specific unresolved question.

### Aggregation and publication

Only the aggregator can turn model output into user-visible feedback. It must:

- verify that the snapshot is still current;
- reject findings without a concrete failure story;
- merge comments with the same root cause;
- suppress style, naming, formatting, generic hardening, and deterministic-tool duplicates;
- prefer a smaller number of higher-severity findings;
- publish no more than three findings at PR time;
- preserve rejected and shadow findings as benchmark telemetry, not comments.

The small model may perform the first cleanup pass, but a model must never be the sole judge of its own findings.

## Illustrative configuration

This example describes the intended configuration surface. It is not accepted by current PR-Agent code.

```toml
[specialist_pipeline]
enabled = true
mode = "shadow"
events = ["file_save", "worktree_idle", "pre_commit", "post_commit", "pull_request"]
file_save_debounce_ms = 1500
file_save_timeout_seconds = 5
worktree_idle_seconds = 15
max_save_specialists = 2
max_worktree_specialists = 5
max_published_findings = 3
save_output = "local_ephemeral"
cancel_stale_snapshots = true

[specialist_pipeline.models]
router = "<approximately-3b-code-model>"
fast_specialist = "<approximately-3b-code-model>"
open_verifier = "<strong-open-weight-code-model>"
frontier_adjudicator = "<frontier-code-review-model>"

[specialist_pipeline.context]
max_router_tokens = 4000
max_save_specialist_tokens = 8000
max_worktree_tokens = 32000
include_enclosing_symbol = true
max_callers = 1
max_related_tests = 1
max_historical_examples = 1

[specialist_pipeline.escalation]
frontier_severity = ["high", "critical"]
frontier_on_disagreement = true
frontier_on_sensitive_path = true
frontier_sensitive_path_events = ["post_commit", "pull_request"]
frontier_on_failed_checks = true

[specialist_pipeline.publication]
severity_floor = "medium"
require_trigger = true
require_observable_impact = true
suppress_style = true
suppress_static_tool_duplicates = true

[[specialist_pipeline.specialists]]
name = "tenant_isolation"
skill = "specialists/tenant-isolation/SKILL.md"
path_patterns = ["**/firestore/**", "**/storage/**", "**/services/**"]
symbols = ["tenantId", "userId", "ownerId"]
minimum_route = "open_verifier"

[[specialist_pipeline.specialists]]
name = "billing"
skill = "specialists/billing/SKILL.md"
path_patterns = ["**/billing/**", "**/payments/**"]
minimum_route = "open_verifier"
frontier_events = ["post_commit", "pull_request"]
```

The model identifiers must be pinned to an evaluated provider, model revision, precision, and context limit. Product names and prices change too quickly to make an unmeasured model the architectural decision.

## Proposed local hooks

Git does not have a file-save hook. The first prototype should support explicit editor and coding-agent integration, then add an optional watcher:

```bash
# Proposed commands; they do not exist yet.
pr-agent review-snapshot --event file-save --path services/functions/src/billing/refund.ts
pr-agent review-snapshot --event worktree-idle --base origin/develop
pr-agent watch --base origin/develop --idle-seconds 15
```

Other checkpoints use normal local hooks:

- an editor `afterSave` command or coding-agent callback for file saves;
- a coding-agent stop callback for coherent worktree review;
- `pre-commit` for the staged snapshot;
- `post-commit` or `pre-push` for an immutable branch review;
- the existing GitHub integration for final PR review.

The watcher must use latest-write-wins cancellation, per-path debounce, a small concurrency cap, and content-hash caching. Untracked files need full-file handling because they have no Git base blob.

## Safety and failure behavior

- File-save agents are advisory. They do not edit code, execute model-generated commands, approve changes, or publish to GitHub.
- Deterministic checks run in an allowlisted sandbox with fixed commands. A model cannot invent a command for the host to execute.
- Code and repository rules follow the configured inference privacy boundary. A local small model is preferred for the fastest tier when its measured quality is sufficient.
- Secrets, generated files, vendored content, binaries, and excluded paths are removed before inference.
- Every result records snapshot, prompt, specialist, model, provider, revision, token use, cache use, latency, and failure state.
- An unavailable reviewer reports partial coverage; it never returns “no issue found.”
- No LLM-only result should block a save or auto-approve a PR during the proposed rollout.
- Sensitive changes retain independent human or frontier review even if the cheap pipeline is quiet.

## Evaluation plan

### 1. Make checkpoint truth measurable

Add `earliest_opportunity` and `required_context` fields to every verified defect. Independently adjudicate the provisional 17 file-save and 23 worktree labels.

Capture one to two weeks of real development events:

- saved content hash and focused path;
- dirty and staged diffs;
- task intent available at that moment;
- whether a later save self-corrected the candidate;
- commit and PR lineage;
- no source text beyond the existing corpus privacy policy.

Add at least 15–20 matched clean **checkpoint** controls before adding many more positive examples. Match them by language, subsystem, change size, and stage. Include partial-but-correct worktrees and temporary mistakes fixed before commit. Final clean PR heads alone cannot estimate pair-review noise.

### 2. Replay the cascade in paired arms

Every arm reviews the same frozen snapshot:

1. deterministic tools only;
2. one general small-model review;
3. small router plus selected small specialists;
4. specialists plus open-weight verifier;
5. full cascade with frontier escalation.

Run file-focused replay on the adjudicated save-eligible defects and checkpoint controls. Run worktree replay on all 40 verified defects and the matched controls. Keep prompt calibration, threshold calibration, temporal backtest, and final holdout separate.

### 3. Measure the thing the developer feels

Primary metrics are:

- time from the first detectable write to the first valid finding;
- incremental recall gained at file-save, worktree, commit, and PR stages;
- verified precision and severity-weighted recall;
- false interruptions per developer-hour and per clean checkpoint;
- escalation precision and high/critical escalation recall;
- stale findings withdrawn after later saves;
- p50 and p95 latency by stage;
- tokens, cached tokens, retries, and dollars per developer-hour and per verified finding;
- deterministic-tool overlap and duplicate-comment rate;
- valid structured-output rate and unavailable-coverage rate.

Do not score a comment as correct merely because it was resolved or followed by a patch. Use a failing-before/passing-after test or independently adjudicated evidence where possible.

### 4. Roll out in gates

| Gate | Required evidence | Behavior |
| --- | --- | --- |
| Offline replay | Reproducible runs; no holdout leakage; valid structured output at least 99.5% after bounded retry | No developer output |
| Live shadow | At least one week of save/worktree telemetry; latency and cost within budget | Compute only; compare with later commits and PR findings |
| Opt-in pair review | At least 80% verified precision on local candidates and no high-severity regression in the temporal backtest | Ephemeral local advice for selected paths |
| Default pair review | At least 90% actionable precision across 100 settled candidates and an acceptable false-interruption rate | File-save advice on by default; still non-blocking |
| PR publication | Verified cascade beats the current reviewer on frozen holdout at the chosen cost ceiling | At most three medium-or-higher comments |

Thresholds are rollout decisions to test, not universal facts. If the gold set has too few high/critical defects, the system cannot earn permission to auto-approve sensitive changes.

## Implementation slices and effort

Keep the work reviewable and reversible:

| Slice | Deliverable | Rough effort for one engineer |
| --- | --- | ---: |
| 1. Snapshot protocol | `ReviewSnapshot`, uncommitted plain-diff input, JSON output, hashes, and telemetry | 1 week |
| 2. Local invocation | `review-snapshot`, editor/coding-agent hook examples, debounce, cancellation, and cache | 1–2 weeks |
| 3. Router and skills | Structured classifier, deterministic risk floor, router-selected Agent Skills | 1–2 weeks |
| 4. Specialists and context | Initial domain prompts plus symbol/caller/contract/test context builder | 2 weeks |
| 5. Verification and escalation | Open verifier, frontier adjudicator, deduplication, and failure states | 1–2 weeks |
| 6. Evaluation and rollout | Checkpoint corpus support, paired replay, dashboards, and shadow gates | 2+ weeks |

A useful local prototype is roughly four to six engineering weeks because slices can overlap. A credible production pilot is more likely eight to twelve weeks once privacy, editor integration, endpoint reliability, telemetry, and benchmark adjudication are included. These are planning ranges, not estimates validated by implementation work.

Suggested code boundaries are:

```text
pr_agent/algo/review_snapshot.py       immutable local/hosted input contract
pr_agent/algo/review_router.py         deterministic and small-model routing
pr_agent/algo/review_aggregator.py     verification, deduplication, ranking
pr_agent/specialists/                  specialist registry and prompt adapters
pr_agent/tools/local_pair_review.py    file-save/worktree orchestration
pr_agent/cli.py                        review-snapshot and watch entrypoints
tests/unittest/                        snapshot, routing, staleness, and failure tests
```

The implementation should first reuse `PlainDiffGitProvider` parsing and structured output. Extract a provider-neutral snapshot boundary only when the prototype proves what extra event metadata is required.

## Decisions and non-goals

The proposal makes these decisions:

- invoke selected specialists on a meaningful file save;
- use a small model for both routing and cleanup;
- use stronger models for verification, not automatic repetition;
- escalate sensitive paths deterministically;
- preserve GitHub PR review as an independent final audit;
- optimize for time to a valid finding, precision, and cost together.

It deliberately does not propose:

- running every specialist on every save;
- treating model self-confidence as calibrated probability;
- publishing the union of several reviewers;
- replacing linters, type checks, tests, or human review;
- blocking local work or auto-approving PRs from an LLM result;
- selecting a production model from public coding leaderboards alone.

The first build decision is therefore narrow: implement shadow-mode file-save snapshots and selective small specialists, then prove that they find real defects earlier without creating an interruption tax.
