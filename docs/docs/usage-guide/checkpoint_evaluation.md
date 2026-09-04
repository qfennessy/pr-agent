# Checkpoint evaluation planning

Checkpoint evaluation artifacts describe how to replay the same immutable local-review
snapshot through several production review arms. List and dry-run modes remain limited to
local validation and deterministic planning. The separately authorized production runner
can call the frozen arms, but it cannot publish a review or enable developer-visible behavior.

All runtime settings remain disabled:

```toml
[checkpoint_evaluation]
enabled = false
allow_paid_execution = false
shadow_journal_enabled = false
publish_output = false
paid_cost_cap_usd = 0.0
shadow_journal_path = ""
shadow_journal_max_queue_entries = 256
```

The credential-free plan reports a separate, versioned `production_binding_inventory`
beside the frozen plan. Its own schema hash and content-derived inventory ID bind the readiness
claims to the immutable manifest. Every entry remains unavailable until all its named source-free
blocker contracts are implemented. A completed adapter may remain dormant behind an unavailable
binding; unavailable status or missing hard-cost-cap enforcement still stops production preflight
before the artifact store or any model client is touched.

The general-review and verified-specialists arms now have dormant production adapters. They call the
isolated no-publish replay seam with the exact immutable snapshot/configuration pair, preserve
source-free per-stage telemetry across the subprocess protocol, and fail closed while joining trusted
verifier identity to normalized severity. Evaluation-only root-cause metadata never changes the ordinary
published finding shape, but the general-review arm remains blocked until production provides an explicit
severity contract rather than an inferred default. Both bindings also remain unavailable until an operator
supplies a current, immutable provider/gateway maximum-charge authority for every frozen route and the bound
runtime controls pass preflight. The model boundary can now consume that authority immediately before every
underlying provider request, but the repository does not ship an authority or treat a LiteLLM price estimate
as one. A quote must include a non-secret, immutable gateway route-binding identifier for its exact provider and
revision. The source-free review bundle freezes a validated, credential-free HTTPS gateway endpoint and the worker
rejects it unless its identity matches every authority quote. The model boundary sends the binding identifier only
to that endpoint; generic LiteLLM provider routes do not enforce this contract. No enforcing gateway binding is
installed, so general-review
and checkpoint-stage execution remain unavailable. The general replay contract remains limited to the
standard OpenAI route; unsupported routes continue to fail configuration capture.
Issue #27 must still complete those contracts and the source/telemetry contracts needed to integrate
the production orchestration delivered by issues #26, #12, #11, #9, and #33. A frozen
target-repository replay and one week of live shadow evidence are also required before any rollout
gate can grant permission.

## Artifact boundary

The `checkpoint-evaluation-v2` contracts use content-derived SHA-256 identities, including
a hash of the concrete field/enum schema rather than only its version label, and keep
five artifact types separate:

- `EvaluationManifest` freezes corpus, policy, configuration, arms, and source-free
  references to serialized `ReviewSnapshot` artifacts.
- `CheckpointCase` contains only a snapshot identity, event lineage, cohort assignment,
  and explicitly model-visible metadata.
- `TruthArtifact` and `CheckpointTruth` contain clean-control labels, adjudicated findings,
  severity, earliest opportunity, and required context. They are loaded separately and
  never appear in an execution plan or `model_visible_payload()`.
- `EvaluationRunRecord` binds one retained attempt to a manifest, case, arm, and snapshot.
  It maps shipped `ReviewSnapshotResult` and `RunDetails` telemetry without turning missing
  tokens, cost, latency, or coverage into zero. Optional route, cache, stage-latency,
  overlap, and lifecycle fields remain structured and source-free. Model-backed specialist
  arms retain one `EvaluationStageRun` per required role, including its pinned
  model/provider/revision, one-way deployment identity hash, configuration and prompt
  hashes, prompt and schema versions, coverage and failure state, latency, tokens, priced
  cost, cache/fallback state, and confidence. Full specialist output and deployment names
  are excluded because they can contain source or environment-specific details.
- `RolloutGateDecision` records `passed`, `failed`, or `not_evaluable` for explicit metrics,
  thresholds, and minimum support.

Unknown fields are rejected when artifacts are loaded. Answer-only field names are also
rejected recursively inside model-visible metadata. Hash metadata must be a full SHA-256
identity, string metadata is a bounded lowercase identifier, and stage and change-size values
use closed enums/ranges. Before execution, the runner independently derives snapshot-backed
metadata and rejects a mismatch. A manifest cannot reuse snapshot
identities, mix one lineage across cohorts, or accept a supplied content id that does not
match its canonical JSON.

Before the production runner uses a source-bearing snapshot, it must call
`load_review_snapshot_artifact()`. The loader reads one bounded regular file without
following any supplied path component (using descriptor-relative component walking), rejects
duplicate keys and non-finite JSON, verifies the exact file-byte hash against the checkpoint,
reconstructs the `ReviewSnapshot`, and then verifies its content-derived snapshot id and
event. Unknown snapshot fields and answer-only keys in deterministic evidence fail closed.
Nested deterministic evidence is copied into immutable mappings and tuples before the
validated snapshot is returned, while `to_dict()` still emits ordinary JSON-compatible
copies. Loading performs no network or model call.

The evaluation artifacts above contain no source text, full diff, secret, hidden reasoning,
provider request identifier, or credential. The separate serialized `ReviewSnapshot` is
source-bearing, stays in local source storage outside published artifact directories, and is
referenced by its exact byte hash.

The separate `checkpoint-review-configuration-v1` bundle is also local and model-visible. It
stores the exact allowlisted general-review settings, resolved skill text, resolved repository
context in prompt order with its rendering line budget, pinned prompt date, package and installed
dependency versions, plus a hash of the executable review code and default settings. It accounts
explicitly for every supported setting as present or missing, rejects unknown or case-colliding
data, and revalidates its content hash and runtime identity before use.
An optional private `checkpoint-stage-sources-v1` extension carries the exact specialist,
candidate-verification, and frontier configurations, prompts, routes, and static-analysis evidence.
The extension is included in the bundle content hash, is never rendered in object representations,
and is not copied into manifests, reports, journals, or other source-free artifacts. Legacy bundles
without the extension keep their original canonical shape and identity. Extended bundles must match
every planned stage hash, version, model route, and deployment identity before the artifact store or
worker process is touched. The worker receives the validated stage plan and exposes only the sources
selected for that arm; production review code falls back to ambient configuration only outside this
checkpoint execution context. Verifier stage-plan hashes cover controls shared across the arm, while
each review-configuration bundle separately content-addresses checkpoint-specific static-analysis
evidence so evolving evidence does not invalidate later cases in the same paired run.
Credentials, callback configuration, telemetry headers, and output sinks have no bundle fields. A checkpoint
bundle may carry one credential-free HTTPS gateway base URL: it is part of the content-derived configuration
identity, never read from the worker environment, and must not contain a token or other secret.
The isolated worker disables publication, callback and OpenTelemetry delivery, push outputs, and
run-detail printing before importing the reviewer. It also passes only the standard OpenAI runtime
credential and a minimal process environment; proxy, ambient endpoint, Dynaconf, and other provider
controls are not inherited. If the bundle supplies the checkpoint gateway endpoint, the worker hash-matches it
to every authority quote before reviewer construction and then replays that exact endpoint. A non-empty snapshot
without this frozen endpoint fails preflight before reviewer construction. Checkpoint
materialization writes the canonical bundle beside its
source-bearing snapshot in the same atomic, owner-only local directory. Shared bundles are stored
once, while the private snapshot index records each pair's deterministic relative paths, bundle
identity, and exact artifact-byte hash. Loading rejects non-canonical bytes, unsafe links or
permissions, snapshot/bundle hash mismatches, and incompatible runtime identity before any adapter
or model can run. Bundle content is never copied into a source-free manifest, journal, report, or
log. Existing provider-neutral local-pair snapshots retain their legacy hash for local workflows,
but they cannot enter this paired evaluation path without the exact immutable bundle.

Production bindings declare their telemetry shape. The deterministic arm must be model-free,
the general-review arm must name one selected pinned model, and specialist, verified-specialist,
and full-cascade arms freeze an exact, nonempty stage plan. Each required stage fixes its
name, primary/fallback route, hashed deployment identity, configuration and prompt hashes,
and prompt/input/output schema versions. Runtime telemetry must contain that exact set once
each; missing, duplicate, unexpected, mismatched, or unpinned stages fail closed. A stage
identity and every model named in its priced-cost map must belong to that stage's frozen
route. Missing, partial, ambiguous, or unpinned identities fail before an artifact is written.
A specialist-only record may omit
the top-level model triple only when it contains complete nonempty per-stage identities. Role
telemetry is also folded into the arm's token and cost totals because production `RunDetails`
intentionally keeps specialist usage separate from the main review footer.

The role collector cannot prove how many calls reported token usage, so any observed role-token
subtotal is `partial`; an absent subtotal is `unavailable`. Cost is `complete` only when every
successful role call was priced, `partial` when only some were priced, and `unavailable` when
none were. A default numeric zero is never persisted as measured latency, tokens, or cost. A
failed role cannot turn a `no_findings` result into apparently clean coverage.

## Credential-free list and dry run

Validate a local manifest and list its cases and enabled arms:

```bash
python -m pr_agent.cli evaluation-plan \
  --manifest artifacts/evaluation-manifest.json \
  --list
```

Expand the deterministic paired plan:

```bash
python -m pr_agent.cli evaluation-plan \
  --manifest artifacts/evaluation-manifest.json \
  --dry-run
```

Optionally validate the separately stored answers at the same time:

```bash
python -m pr_agent.cli evaluation-plan \
  --manifest artifacts/evaluation-manifest.json \
  --truth private/adjudicated-truth.json \
  --dry-run
```

Both modes are local-only. Their machine-readable output states `network_calls: 0` and
`model_calls: 0`. The plan includes the same `snapshot_id` for every enabled arm paired
with a checkpoint, freezes its serialized-artifact hash and every arm's prompt,
configuration, provider, model, revision, and source-free per-stage contract, and never
includes truth. The separately versioned readiness inventory reports each arm's current
production-binding availability and bounded blocker codes without importing a model handler
or probing credentials.

## Paid execution is fail closed

`evaluate_paid_execution()` is the mandatory boundary for a future production-backed
runner. It returns `denied` unless all of these facts are true at the same time:

- evaluation and paid execution were explicitly enabled;
- publication is still disabled;
- every model-backed case/arm pair has a positive hard per-attempt cost cap and a bounded
  maximum attempt count whose full reservation fits a positive cap for this exact manifest;
- every enabled model arm pins a model revision; and
- the process has a credential for every named provider.

The request records only credential presence booleans. It never serializes an environment
variable name, token, key, credential value, or provider request identifier. The request is
immutably bound beside the manifest. Before every adapter call, the runner reloads retained
attempts, requires complete cost telemetry, and checks cumulative spend plus all remaining
reserved attempts against the cap. Every model-backed binding must also declare hard-cap
support and receives the immutable per-attempt cap in its adapter context; a binding that
cannot enforce that cap before provider calls is unavailable. A final failed attempt is
terminalized at its immutable limit, so restarting cannot create unbounded retries. The default cost cap is zero and
therefore cannot authorize spending. A caller must invoke
`PaidExecutionDecision.require_authorized()` immediately before entering production
orchestration; planning alone is not authorization.

Before a paid adapter starts, the artifact store exclusively persists its attempt number
and hard-cap reservation. A crash, raised adapter exception, or rejected result leaves an
unreconciled reservation that blocks resume instead of repeating a possibly charged call.

### Provider/gateway maximum-charge authority

`checkpoint-cost-authority-v1` is a separate source-free contract for one manifest, paid request,
case, arm, snapshot, review configuration, and per-attempt cap. It names an immutable authority
revision and hashes the external provider or enforcing-gateway guarantee. Each quote pins one exact
stage, model, provider, revision, deployment identity, hashed HTTPS gateway endpoint, non-secret immutable
gateway route-binding identifier, maximum output-token cap, and worst-case charge. The binding identifier is
safe to persist and contains no credential, source, prompt, or endpoint URL. It also expires. Unknown fields,
mutable revision aliases, ambiguous routes, incomplete route coverage,
and any quote larger than the attempt cap fail validation locally.

Every authority amount is bounded before it is used: at most 32 significant digits, a magnitude below
10^16 USD, and no more than 18 fractional digits. Values outside that range are rejected when the
authority is loaded, so a compact but enormous exponent cannot force an unbounded fixed-point rendering
while an identity is derived. Cumulative reservation arithmetic then runs in a local decimal context wide
enough to hold any sum of accepted values exactly, with inexact results trapped, so a rounded total can
never compare below the hard cap.

The raw gateway base URL is stored only in the private review-configuration bundle, where it is validated as
bounded, HTTPS, and free of URL credentials, query parameters, and fragments. It is not stored in the authority,
manifest, journal, report, or environment. Operators must treat the endpoint as non-secret configuration and
must never embed credentials in its path.

The isolated single-review worker installs one consume-only ledger process-wide and in its coroutine
context, so both worker threads and concurrent specialist, verifier, and frontier tasks share it.
Immediately before every LiteLLM `acompletion` call, including same-model
retries, fallback routes, streaming calls, and the Bedrock credential fallback, the ledger atomically
reserves the quote's full worst-case charge. It does not refund reservations based on estimated actual
usage. A request is denied before the provider client when its route is unquoted, the output cap is
missing or larger than quoted, provider SDK retries are not exactly zero, the authority is expired or
mismatched, the explicit gateway endpoint does not match the quote, the route binding conflicts with an existing
header, or the next reservation would exceed the cap. After reservation, the handler attaches the exact binding as
`X-PR-Agent-Checkpoint-Route` on the actual `acompletion` request. The external gateway contract must guarantee
that it rejects an unknown or mismatched identifier and routes a recognized identifier only to the quoted provider
and immutable revision. This header is not a credential and must never be replaced with a secret token. Dynamic
per-finding frontier telemetry attribution is normalized to
the fixed `frontier_adjudication` stage only for quote lookup; full attribution remains in telemetry.
Post-response cost and identity telemetry
still must be complete and agree with the frozen arm; the authority does not turn missing telemetry into
zero.

This boundary depends on an external provider/gateway guarantee that the quoted maximum is actually
enforced for the bound request regardless of its eventual input or output usage. Operator guesses and
the ordinary LiteLLM pricing table are observational estimates, not
spending authority. Consequently the default production binding inventory keeps
`hard_cost_cap_enforcement_unavailable` until a real authority is supplied and its frozen route controls
are proven. A direct or generic LiteLLM provider endpoint that ignores the route header does not qualify. This
repository change alone does not authorize or execute a paid request.

## Resumable raw attempt artifacts

`EvaluationArtifactStore` writes one content-addressed JSON file per attempt with private
permissions and exclusive creation. The store retains successful, malformed, timed-out,
stale, cancelled, provider-failed, and unavailable-coverage records. A later retry gets a
new attempt number and cannot overwrite an earlier failure. The manifest is written once,
and a changed manifest, corrupt record, wrong snapshot, wrong model/provider/revision,
skipped attempt, second terminal record, mixed manifest, non-private path, symlink, or
unexpected file fails resume.

`resume_plan()` omits only case/arm pairs that already have one terminal record. For every
other pair it returns the next attempt number and the identities of all retained attempts.
`inventory()` produces the hashes needed for a pilot report without copying source or
model-visible snapshot bytes into the report. Resume loading and scoring both revalidate each
aggregate or stage identity, stage name/cardinality, stage contract version, deployment
identity hash, and each priced-cost model key against the same frozen arm.

## Privacy-safe live shadow journal

`ShadowJournalWriter` is disabled unless explicitly opted in. When enabled, checkpoint
code calls only bounded `put_nowait`; a daemon worker appends source-free NDJSON in the
background. A full queue or write failure drops telemetry rather than delaying or failing
a save. Explicit shutdown may flush the queue, but the save path never waits for disk.
Enabling a writer first durably creates a private, source-free session-open boundary tied to
the prior journal tail. Until clean shutdown removes that boundary, report loading rejects the
journal, so a crash before the first queued record reaches disk cannot leave an older sealed
inventory looking current. Shutdown marks submission closed under the submission lock, releases
the lock, and only then waits for queue capacity; concurrent checkpoints return `CLOSED` immediately.
Each accepted entry is wrapped by the writer with a contiguous sequence, UTC ingestion time,
and writer-owned monotonic developer-time basis. Reports derive duration and cost-hour evidence
only from parsed, identity-checked journal records; caller-supplied timestamps or elapsed-time
wrappers cannot satisfy a gate.
At explicit shutdown, the writer seals the last retained record in each session with immutable
submitted, queued, and dropped counts plus writer-failure status. A missing seal, any dropped
submission, or any writer failure makes the raw inventory unavailable and keeps retained latency
and cost samples partial, so a biased subset cannot satisfy the live-shadow gate.

The schema allowlists snapshot and lineage hashes, event and configuration versions,
selected depth and machine reason codes, model/provider identities, hashed finding
fingerprints and lifecycle state, coverage/failure/cache/retry status, tokens, cost, and
latency. Specialist-only entries keep the same source-free per-stage identity, version,
coverage, failure, latency, token, and priced-cost records as their immutable raw attempt.
Creating a journal entry requires the exact frozen arm and revalidates aggregate identities,
the exact required stage set, every stage identity and contract version, and every per-model
cost key. It has no field for source, a diff, task
intent, prompt text, model output, deployment name, secret, credential, provider request id,
or hidden reasoning. Fingerprints are re-hashed before persistence. Only a completed review
whose complete required stage plan is present and covered marks aggregate coverage complete;
an empty or partial stage set can never pass by vacuous truth. Timeout, malformed, stale,
cancelled, provider-failed, stage-failed, and explicit coverage-unavailable attempts
keep coverage unavailable.

## Scoring and gate interpretation

`score_matched_arms()` requires one truth entry for every manifest case. Missing terminal
records, provider failures, malformed output, timeouts, stale runs, and unavailable
coverage remain in the denominator. A later successful retry does not delete its earlier
failed attempt. The scorecard reports:

- structured-output, verified-precision, verified-recall, and severity-weighted-recall
  rates, broken out by cohort with support;
- time to first valid finding and incremental recall at each checkpoint event;
- false interruptions per independently adjudicated clean checkpoint and developer-hour;
- escalation precision and high/critical escalation recall;
- stale-finding withdrawal, deterministic overlap, and duplicate-finding counts;
- p50/p95 latency overall, by event, and by recorded production stage;
- tokens, retries, dollar totals, cost per developer-hour, and cost per verified finding;
- unavailable-coverage, failed/missing-case, and failed-attempt rates; and
- matched per-snapshot quality, interruption, structured-output, latency, token, and cost
  deltas with a two-sided 95% paired Student-t interval and exact support.

Lineage timing is explicit evaluation metadata and never model-visible. A truth finding may
name an exact earliest checkpoint and a later withdrawal checkpoint. Descendant checkpoints
may be clean after a temporary defect self-corrects. Missing timing, escalation, pricing,
or duration makes the affected metric partial or unavailable, so it cannot pass a gate.

`evaluate_rollout_gate()` accepts explicit `GateRule` thresholds. A known threshold miss
returns `failed`. An absent metric, partial/unavailable measurement, or insufficient sample
returns `not_evaluable`; it never becomes permission to roll out. Loaded rule-result and
top-level decision statuses are recomputed from their frozen evidence, so a serialized
artifact cannot claim `passed` over failed or missing measurements.

### Canonical pilot report and rollout gates

`build_checkpoint_pilot_report()` is the source-free report boundary for the target-repository
pilot. It loads every retained attempt through `EvaluationArtifactStore`, calls
`score_matched_arms()` with the frozen incumbent, and evaluates the canonical rules through
`evaluate_rollout_gate()`. The report records the manifest, corpus, schema, policy,
configuration, arm-configuration, prompt, and hashed model identities; the raw replay, shadow,
and settled-candidate artifact hash inventories; terminal and incomplete pair counts; the exact scorecard id;
the full source-free scorecard with cohort support and paired uncertainty; accepted budgets;
source-free observed measurements; and all five gate decisions. It does not
serialize source, diffs, truth or adjudication identifiers, model/provider identifiers,
provider request ids, or credentials.

The canonical rules are:

| Gate | Rules that must have complete evidence |
| --- | --- |
| Offline replay | Structured-output rate at least 99.5%, complete immutable replay inventory, an independently pinned no-holdout-leakage artifact bound to the exact model-visible holdout inventory, and retry limits derived from the immutable paid request |
| Live shadow | At least seven real elapsed days with both file-save and worktree events, a raw shadow-artifact inventory, and complete p95-latency and cost-per-developer-hour measurements within explicitly accepted positive budgets |
| Opt-in pair review | Verified precision at least 80%, no negative high-severity recall delta on the temporal cohort, and complete replay artifacts |
| Default pair review | Actionable precision at least 90% derived from the exact accepted inventory of at least 100 settled candidates, observed false interruptions no greater than an explicitly accepted non-negative threshold, and complete shadow artifacts |
| PR publication | A strictly positive lower 95% bound for the cascade's quality advantage across all 18 cases in the locked Cocos Story holdout, complete holdout cost within an explicitly accepted positive ceiling, and complete replay artifacts |

Create `PilotRolloutBudgets` only from thresholds a maintainer has actually accepted.
`PilotRolloutEvidence` carries only separately bound replay declarations and a content-bound
leakage-check artifact; a bare leakage boolean cannot satisfy a gate. The checker revision and
exact model-visible holdout inventory are hashed, and a separate maintainer change must pin the
check id before it counts. Evidence cannot accept shadow
timestamps, counts, metrics, artifact hashes, or a settled-candidate precision claim. The report
derives settled-candidate precision, temporal high-severity
recall, frozen-holdout paired recall uncertainty, and holdout cost directly from accepted records,
immutable attempts, and the separate answer ledger. Callers cannot supply those rollout measurements.
The publication comparison also requires the validated `CocosCorpusInventory` and a separately
reviewed `CocosPilotAcceptance`. `build_cocos_pilot_acceptance()` generates a source-free artifact
containing the canonical adapter lock, manifest schema version/hash, corpus hash, and the full
ordered case inventory. Every assignment records only its case id, snapshot identity/artifact hash,
cohort, event, and parent case. It therefore commits to every calibration, holdout, temporal, and
control assignment without copying source. Generation does not approve it. A maintainer must
independently review that artifact and pin its
content-derived `acceptance_id` as `CANONICAL_COCOS_PILOT_ACCEPTANCE_ID` in a separate change before
publication can pass. This prevents one report call from supplying both the evidence and the values
that supposedly accept it.

The inventory must use the checked-in canonical Cocos lock and its exact cohort counts, the manifest
and inventory must match the pinned artifact's schema and corpus identities, and every manifest case
must match the accepted ordered assignment tuple exactly. Substitution, reordering, extra or missing
cases, a changed cohort, and a changed snapshot hash are rejected even when the aggregate count stays
18. The report records only the acceptance id, a reproducible source-free inventory hash, and those
accepted identities; it omits the assignment details, corpus path, and source revision. Missing or
generated-but-unpinned acceptance produces `not_evaluable`, while a conflicting acceptance id, lock,
schema, corpus, assignment, cohort count, or holdout hash is invalid input. Eighteen arbitrary cases,
19 cases, or a one-case point estimate cannot be represented as a passing confidence bound.

Live shadow evidence follows the same two-step boundary. `ShadowJournalRecord` binds one actual
`ShadowJournalEntry` content id to the writer-stamped ingestion time in UTC, contiguous sequence,
and writer-owned monotonic developer-time
denominator. The last record in each writer session also binds the writer-owned submission,
retention, drop, and failure summary. An outstanding durable session-open boundary rejects parsing
before acceptance can be generated. `build_shadow_pilot_acceptance()` verifies the journal entries use the exact manifest
policy, configuration, target arm, aggregate model identity, required stage plan, primary/fallback
route, deployment identity, prompt/configuration/schema versions, cost model identities, and journal
schema. It also rejects negative aggregate or stage latency, token, cost, or developer-time
measurements, preserves the records' observed order, and emits the complete source-free record
inventory and journal hash. Generation alone cannot pass a gate; a maintainer must independently pin
its content id as `CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID`.

Report construction then matches every parsed accepted record id and entry id in order. Truncated, extra,
substituted, duplicated, or reordered journal records fail validation. Duration is recomputed from
the first and last UTC ingestion records, p95 latency from every entry's recorded latency, event counts
from the recorded event enum, and cost per developer-hour from recorded costs and accepted developer
elapsed denominators. Missing, partial, or unpriced entries remain partial or unavailable. The report
also refuses to call the raw inventory complete, or the retained latency and cost samples complete,
when a writer session lacks its final seal or reports any drop or write failure. The report
contains only the accepted journal and record hashes, UTC span, derived counts, and derived metrics;
it does not expose journal model/provider identities or entry payloads.

Default pair-review evidence has the same independent boundary. Each `SettledCandidateRecord`
contains only a hashed candidate identity, its independent adjudication hash, and the boolean
actionable result. `build_settled_pilot_acceptance()` commits to the full ordered source-free record
inventory and binds it to the exact manifest, policy, configuration, and target arm. A maintainer must
review that artifact and separately pin its content id as
`CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID`. Report construction rejects truncated, extra, substituted,
duplicated, or reordered records, then derives both the settled denominator and actionable precision
from the accepted records. A caller-supplied point estimate and generated-but-unpinned records remain
`not_evaluable`; they can never satisfy the 100-candidate or 90% thresholds.

A missing budget is represented as `None`, not zero. Missing or empty duration, event
denominator, raw inventory, cap, accepted false-interruption threshold, or partial/unavailable
measurement produces `not_evaluable`. A complete measurement that misses its threshold
produces `failed`.

The report's model identity values are one-way hashes over the frozen manifest identities.
The separately stored answer ledger is used for scoring but its id and contents never enter
the report. Keep the report JSON publishable and keep the manifest, source-bearing snapshots,
truth ledger, raw attempt files, and shadow journals in their existing private stores.

`evaluate_output_permission()` separately binds opt-in advice, default advice, or PR
publication to its exact passed gate plus every preceding offline-replay and rollout gate
for the same arm and scorecard id. Missing decisions, stale
scorecards, duplicate decisions, failed gates, and `not_evaluable` gates deny output.
Gate rules may target a cohort explicitly with names such as
`cohort.temporal.verified_recall`, so a healthy aggregate cannot hide a temporal regression.
They may also gate a target arm's matched advantage over the incumbent with names such as
`paired.case_recall.lower_95` or `paired.cost_usd.upper_95`. The general-review arm is the
default incumbent when present; callers may freeze another explicit baseline. Output
permission also requires the exact maintainer-approved gate-spec hash, so a weaker ad hoc
threshold cannot enable advice or publication.

## External Cocos Story corpus adapter

The Cocos corpus stays in its own checkout. PR-Agent accepts a source path plus a lock that
pins the accepted source commit; exact hashes for the primary, temporal, control, specialist,
sealed-confirmation, and confirmation-annotation ledgers; the original cohort counts; and
hashes of every `id/split/target_sha` assignment. The adapter rejects symlinks (including a
symlinked confirmation directory), changed bytes, changed assignments, wrong repositories,
answer-visible annotations, an unsealed confirmation policy, mismatched confirmation
ledger/annotation ids or target SHAs, or changed 12/18/10/16/16 and 55-snapshot counts. Its
output contains only hashes, counts, and a one-way local-root identity. The 16 confirmation
cases remain a sealed measurement cohort: their defect targets cannot be used for prompt or
architecture selection and are never copied into a model-visible manifest.

The checked-in `cocos_story_corpus_lock.json` pins the corpus accepted in Cocos Story PR
#9425 at merge commit `6b98bae67bae4056c4567187454e24cca78b9467`. It contains hashes and counts,
not corpus examples, source, diffs, context, or answers.

Validate the external corpus while expanding the no-call plan:

```bash
python -m pr_agent.cli evaluation-plan \
  --manifest artifacts/evaluation-manifest.json \
  --cocos-corpus-root /path/to/cocos-story/docs/testing/code-review-agent-corpus-2025-12-29_2026-08-29 \
  --cocos-lock docs/docs/usage-guide/cocos_story_corpus_lock.json \
  --checkpoint-controls private/checkpoint-controls.json \
  --dry-run
```

The separate answer-only checkpoint-control ledger must declare exact schema version
`cocos-story-checkpoint-controls-v1` and contain 15–20 unique,
hash-addressed, independently adjudicated clean intermediate checkpoints. It must cover
partial-but-correct saves, coherent clean worktrees, temporary mistakes corrected before
commit, staged checkpoints, and stale-candidate withdrawal lineages. The adapter rejects
source-bearing or unknown fields and final clean PR heads as substitutes, and includes the
control artifact hash in the pilot corpus identity. Its file, corpus root, and every supplied
parent component must be real directories/files rather than symlinks. The evaluation manifest
cannot receive a canonical Cocos binding when this control inventory is absent, incomplete,
outside the 15–20 range, or missing its validated hash, and must bind that exact resulting
corpus identity. Until the ledger is supplied, its status is `not_evaluable`.

## Reproduction, privacy, and rollback

To reproduce a run, keep the immutable manifest, Cocos lock, serialized snapshot hashes,
every attempt file, truth artifact, scorecard, gate decisions, explicit budgets, and raw
artifact inventory. Re-run list/dry-run first, confirm its manifest id, then resume the same
artifact store. Never edit or replace a failed attempt.

Keep snapshot/source storage and answer-only truth outside model-visible directories and
outside published reports. Shadow journals use private permissions and should live outside
the source tree, normally below the Git common directory. Delete or archive them according
to the target repository's approved retention policy; do not publish them as PR artifacts.

Rollback is configuration-only while evaluation remains disabled: stop the shadow writer,
set all four behavior switches to false, clear the untracked journal/artifact path, and
continue using the incumbent reviewer. Historical evidence stays immutable. Rollback never
changes a `failed` or `not_evaluable` gate to `passed`.

Still required to complete issue #27: resolve the blocker contracts reported by
`production_binding_inventory` and make each production arm available; create the independent
checkpoint controls and serialized snapshots; run the
frozen paid replay within an authorized cap; collect at least one week of opt-in live shadow
telemetry; publish the pilot inventory, scorecard, budgets, and gate decisions; and prove
that the cascade beats the incumbent before enabling local advice or GitHub publication.

Production review orchestration now has an internal, single-use
`PRReviewer._run_structured_no_publish_once()` execution primitive. It returns the same enriched
structured review payload and request-local run telemetry, with elapsed time frozen at completion,
while forcing provider publication, GitHub Action output, external output sinks, and process-local
rendered artifacts off. It requires an outer isolation boundary and a fresh reviewer instance.

`checkpoint_review_subprocess` wraps construction and execution in a dedicated child process.
Its versioned pipe protocol is disabled unless model execution is explicitly allowed, validates
the complete request and source-free cost authority before importing model handlers, disables
working-tree enrichment and every
output sink, and returns only structured review data plus bounded run telemetry. Process-wide
settings, callbacks, credentials, and environment mutations die with the child process.

This is still not an authorized runnable production evaluation. Preflight loads each immutable
snapshot/configuration pair before artifact-store access and supplies the exact bundle in the frozen
adapter context. Provider-neutral normalizers derive versioned general-review fingerprints, preserve
trusted verified-finding keys, and compute active/withdrawn lifecycle transitions from parent
checkpoints rather than model fields. They fail closed when production output omits durable root-cause
identity or trusted severity instead of accepting model-controlled fingerprints or location-derived
identity. The subprocess protocol excludes source-bearing specialist output while retaining strict
aggregate, specialist, verifier, and frontier telemetry. Parent checkpoints execute before children,
and the runner derives withdrawals only from completed terminal parent records; children with unavailable
parents are not reserved or executed. Clean empty-diff outcomes persist an unambiguous zero-call record
with observed zero tokens and cost; missing telemetry after a model call remains unavailable. Paid
calls remain unavailable because no authoritative provider/gateway maximum-charge contract is supplied
by the repository, although the worker now enforces such a contract at every underlying call.
An enforcing gateway route binding, general-review severity,
deterministic/specialist finding contracts, and full-cascade frontier decision and aggregate-stage semantics
also remain fail-closed.
