# Checkpoint evaluation planning

Checkpoint evaluation artifacts describe how to replay the same immutable local-review
snapshot through several production review arms. The first implementation is deliberately
limited to local validation, deterministic planning, pure scoring, and rollout-gate
decisions. It does not call a model, contact a provider, publish a review, or enable any
developer-visible behavior.

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

Issue #27 must still integrate the production orchestration, specialist, router, and verifier delivered
by issues #26, #12, #11, and #9. A frozen target-repository replay and one week of live shadow
evidence are also required before any rollout gate can grant permission.

## Artifact boundary

The `checkpoint-evaluation-v1` contracts use content-derived SHA-256 identities, including
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
  overlap, and lifecycle fields remain structured and source-free.
- `RolloutGateDecision` records `passed`, `failed`, or `not_evaluable` for explicit metrics,
  thresholds, and minimum support.

Unknown fields are rejected when artifacts are loaded. Answer-only field names are also
rejected recursively inside model-visible metadata. A manifest cannot reuse snapshot
identities, mix one lineage across cohorts, or accept a supplied content id that does not
match its canonical JSON.

These artifacts contain no source text, full diff, secret, hidden reasoning, provider
request identifier, or credential. Snapshot content remains in the existing serialized
snapshot store and is referenced by hash.

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
configuration, provider, model, and revision identity, and never includes truth.

## Paid execution is fail closed

`evaluate_paid_execution()` is the mandatory boundary for a future production-backed
runner. It returns `denied` unless all of these facts are true at the same time:

- evaluation and paid execution were explicitly enabled;
- publication is still disabled;
- projected cost is complete, not partial or unavailable, and is no greater than a
  positive cap supplied for this exact manifest;
- every enabled model arm pins a model revision; and
- the process has a credential for every named provider.

The request records only credential presence booleans. It never serializes an environment
variable name, token, key, credential value, or provider request identifier. The default
cost cap is zero and therefore cannot authorize spending. A caller must invoke
`PaidExecutionDecision.require_authorized()` immediately before entering production
orchestration; planning alone is not authorization.

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
model-visible snapshot bytes into the report.

## Privacy-safe live shadow journal

`ShadowJournalWriter` is disabled unless explicitly opted in. When enabled, checkpoint
code calls only bounded `put_nowait`; a daemon worker appends source-free NDJSON in the
background. A full queue or write failure drops telemetry rather than delaying or failing
a save. Explicit shutdown may flush the queue, but the save path never waits for disk.

The schema allowlists snapshot and lineage hashes, event and configuration versions,
selected depth and machine reason codes, model/provider identities, hashed finding
fingerprints and lifecycle state, coverage/failure/cache/retry status, tokens, cost, and
latency. It has no field for source, a diff, task intent, prompt, secret, credential,
provider request id, or hidden reasoning. Fingerprints are re-hashed before persistence.
Only a completed review marks coverage complete; timeout, malformed, stale, cancelled,
provider-failed, and explicit coverage-unavailable attempts keep coverage unavailable.

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
  deltas with a 95% paired interval and exact support.

Lineage timing is explicit evaluation metadata and never model-visible. A truth finding may
name an exact earliest checkpoint and a later withdrawal checkpoint. Descendant checkpoints
may be clean after a temporary defect self-corrects. Missing timing, escalation, pricing,
or duration makes the affected metric partial or unavailable, so it cannot pass a gate.

`evaluate_rollout_gate()` accepts explicit `GateRule` thresholds. A known threshold miss
returns `failed`. An absent metric, partial/unavailable measurement, or insufficient sample
returns `not_evaluable`; it never becomes permission to roll out. Loaded rule-result and
top-level decision statuses are recomputed from their frozen evidence, so a serialized
artifact cannot claim `passed` over failed or missing measurements.

`evaluate_output_permission()` separately binds opt-in advice, default advice, or PR
publication to one exact passed gate, arm, and scorecard id. Missing decisions, stale
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
pins the accepted source commit, exact hashes for the primary, temporal, control, and
answer-only specialist ledgers, the original cohort counts, and a hash of every
`id/split/target_sha` assignment. The adapter rejects symlinks, changed bytes, changed
assignments, wrong repositories, answer-visible annotations, or changed 12/18/10/16 and
55-snapshot counts. Its output contains only hashes, counts, and a one-way local-root
identity.

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
control artifact hash in the pilot corpus identity. The evaluation manifest must bind that
exact resulting corpus identity. Until the ledger is supplied, its status is
`not_evaluable`.

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

Still required to complete issue #27: merge and bind the production arms from #26, #12,
#11, and #9; create the independent checkpoint controls; run the frozen paid replay within an
authorized cap; collect at least one week of opt-in live shadow telemetry; publish the
pilot inventory, scorecard, budgets, and gate decisions; and prove that the cascade beats
the incumbent before enabling local advice or GitHub publication.
