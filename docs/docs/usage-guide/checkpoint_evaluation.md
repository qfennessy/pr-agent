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
```

Issue #27 must still integrate the production specialist, router, and verifier delivered
by issues #12, #11, and #9. Paid replay, a target-repository pilot, and live shadow evidence
are also required before any rollout gate can grant permission.

## Artifact boundary

The `checkpoint-evaluation-v1` contracts use content-derived SHA-256 identities and keep
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
  tokens, cost, latency, or coverage into zero.
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
with a checkpoint, and never includes truth.

## Scoring and gate interpretation

`score_matched_arms()` requires one truth entry for every manifest case. Missing terminal
records, provider failures, malformed output, timeouts, stale runs, and unavailable
coverage remain in the denominator. A later successful retry does not delete its earlier
failed attempt. The initial scorecard reports:

- structured-output, verified-precision, verified-recall, and severity-weighted-recall
  rates;
- false interruptions per independently adjudicated clean checkpoint;
- unavailable-coverage and failed-attempt rates;
- p50 and p95 latency; and
- token and cost totals with complete, partial, or unavailable status.

`evaluate_rollout_gate()` accepts explicit `GateRule` thresholds. A known threshold miss
returns `failed`. An absent metric, partial/unavailable measurement, or insufficient sample
returns `not_evaluable`; it never becomes permission to roll out.

The current scorer is a dependency-independent base. Issue #27 still owns temporal
earliest-detection and withdrawal scoring, developer-hour denominators, uncertainty,
production-arm execution, spending controls, privacy-safe shadow journaling, the Cocos
Story adapter, frozen pilot evidence, and the final gate specifications.
