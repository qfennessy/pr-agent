import json
import threading
import tomllib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    CheckpointTruth,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationModelIdentity,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationValidationError,
    FindingLifecycleState,
    FindingSeverity,
    FindingTruth,
    GateStatus,
    MeasurementStatus,
    NumericMeasurement,
    ObservedFinding,
    TruthArtifact,
    content_hash,
)
from pr_agent.algo.checkpoint_evaluation_cocos import (
    CHECKPOINT_CONTROL_SCHEMA_VERSION,
    CocosCorpusLock,
    validate_cocos_story_corpus,
)
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactStore,
    OutputCapability,
    PaidExecutionRequest,
    PaidExecutionStatus,
    evaluate_output_permission,
    evaluate_paid_execution,
)
from pr_agent.algo.checkpoint_evaluation_scoring import (
    GateComparator,
    GateRule,
    GateRuleResult,
    RolloutGateDecision,
    ScoreMetric,
    evaluate_rollout_gate,
    score_matched_arms,
)
from pr_agent.algo.checkpoint_shadow_journal import (
    ShadowJournalEntry,
    ShadowJournalWriter,
    ShadowSubmitStatus,
)
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewResultState, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails
from pr_agent.cli import run


def _hash(value: str) -> str:
    return content_hash({"value": value})


def _arm(
    arm_id: str,
    kind: EvaluationArmKind,
    *,
    revision: str | None = "2026-08-30.1",
) -> EvaluationArm:
    deterministic = kind is EvaluationArmKind.DETERMINISTIC
    return EvaluationArm(
        arm_id=arm_id,
        kind=kind,
        configuration_hash=_hash(f"configuration-{arm_id}"),
        prompt_hash=_hash(f"prompt-{arm_id}"),
        model_id=None if deterministic else "small-reviewer",
        provider_id=None if deterministic else "provider-v1",
        model_revision=None if deterministic else revision,
    )


def _case(
    case_id: str,
    cohort: EvaluationCohort,
    event: ReviewEvent,
    elapsed: float,
    *,
    parent: str | None = None,
    developer_seconds: float = 60,
) -> CheckpointCase:
    return CheckpointCase(
        case_id=case_id,
        snapshot_id=_hash(f"snapshot-{case_id}"),
        snapshot_artifact_hash=_hash(f"artifact-{case_id}"),
        event=event,
        cohort=cohort,
        parent_case_id=parent,
        lineage_elapsed_seconds=elapsed,
        developer_elapsed_seconds=developer_seconds,
    )


def _manifest(cases, arms, *, corpus_hash: str | None = None) -> EvaluationManifest:
    return EvaluationManifest(
        name="evaluation",
        corpus_hash=corpus_hash or _hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        cases=tuple(cases),
        arms=tuple(arms),
    )


def _record(
    manifest: EvaluationManifest,
    case: CheckpointCase,
    arm_id: str,
    *,
    attempt: int = 1,
    state: EvaluationRunState = EvaluationRunState.COMPLETED,
    terminal: bool = True,
    findings=(),
    escalated: bool | None = None,
    latency: float = 1,
) -> EvaluationRunRecord:
    arm = next(item for item in manifest.arms if item.arm_id == arm_id)
    return EvaluationRunRecord(
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        arm_id=arm_id,
        snapshot_id=case.snapshot_id,
        attempt=attempt,
        state=state,
        terminal=terminal,
        findings=tuple(findings),
        latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, latency),
        tokens=NumericMeasurement(MeasurementStatus.COMPLETE, 10),
        cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.01),
        retry_count=max(0, attempt - 1),
        escalated=escalated,
        stage_latencies_seconds={
            "router": NumericMeasurement(MeasurementStatus.COMPLETE, latency / 4),
            "review": NumericMeasurement(MeasurementStatus.COMPLETE, latency * 3 / 4),
        },
        model_id=arm.model_id,
        provider_id=arm.provider_id,
        model_revision=arm.model_revision,
    )


def test_paid_execution_requires_every_gate_and_never_records_credentials():
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    unpinned = _arm("small", EvaluationArmKind.GENERAL_REVIEW, revision=None)
    manifest = _manifest((case,), (unpinned,))
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=2.0,
        projected_cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 1.0),
        credential_present_by_provider={"provider-v1": True},
    )

    decision = evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    assert decision.status is PaidExecutionStatus.DENIED
    assert "immutable model revision" in decision.reasons[0]
    assert "credential" not in json.dumps(request.to_dict()).lower().replace("credential_present", "")
    with pytest.raises(EvaluationValidationError, match="paid execution denied"):
        decision.require_authorized()

    pinned_manifest = _manifest((case,), (_arm("small", EvaluationArmKind.GENERAL_REVIEW),))
    authorized_request = PaidExecutionRequest(
        manifest_id=pinned_manifest.manifest_id,
        cost_cap_usd=2.0,
        projected_cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 1.0),
        credential_present_by_provider={"provider-v1": True},
    )
    authorized = evaluate_paid_execution(
        pinned_manifest,
        authorized_request,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )
    assert authorized.status is PaidExecutionStatus.AUTHORIZED
    authorized.require_authorized()


@pytest.mark.parametrize(
    ("enabled", "allow_paid", "publish", "cost_status", "cost", "credentials"),
    [
        (False, True, False, MeasurementStatus.COMPLETE, 1.0, True),
        (True, False, False, MeasurementStatus.COMPLETE, 1.0, True),
        (True, True, True, MeasurementStatus.COMPLETE, 1.0, True),
        (True, True, False, MeasurementStatus.PARTIAL, 1.0, True),
        (True, True, False, MeasurementStatus.COMPLETE, 0.0, True),
        (True, True, False, MeasurementStatus.COMPLETE, 3.0, True),
        (True, True, False, MeasurementStatus.COMPLETE, 1.0, False),
    ],
)
def test_paid_execution_fails_closed_for_each_missing_proof(
    enabled, allow_paid, publish, cost_status, cost, credentials
):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("small", EvaluationArmKind.GENERAL_REVIEW),))
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=2.0,
        projected_cost_usd=NumericMeasurement(cost_status, cost),
        credential_present_by_provider={"provider-v1": credentials},
    )
    assert evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=enabled,
        allow_paid_execution=allow_paid,
        publish_output=publish,
    ).status is PaidExecutionStatus.DENIED


def test_frozen_production_fallback_identity_is_authorized_retained_and_scored(tmp_path):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    primary = _arm("small", EvaluationArmKind.GENERAL_REVIEW)
    arm = replace(
        primary,
        fallback_models=(
            EvaluationModelIdentity("fallback-reviewer", "fallback-provider", "fallback-2026-08-30"),
        ),
    )
    manifest = _manifest((case,), (arm,))
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=2.0,
        projected_cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 1.0),
        credential_present_by_provider={"provider-v1": True, "fallback-provider": True},
    )
    assert evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    ).status is PaidExecutionStatus.AUTHORIZED
    result = ReviewSnapshotResult(
        snapshot_id=case.snapshot_id,
        state=ReviewResultState.NO_FINDINGS,
        current_snapshot_id=case.snapshot_id,
        review={"review": {"key_issues_to_review": []}},
        coverage_issues=(),
        latency_seconds=0.5,
    )
    details = RunDetails(
        model_used="fallback-reviewer",
        fallback_used=True,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        num_ai_calls=2,
        total_cost_usd=Decimal("0.02"),
        known_cost_call_count=2,
    )
    record = EvaluationRunRecord.from_snapshot_result(
        manifest,
        case,
        arm,
        result,
        details,
        attempt=1,
        terminal=True,
    )
    assert (record.model_id, record.provider_id, record.model_revision) == (
        "fallback-reviewer",
        "fallback-provider",
        "fallback-2026-08-30",
    )
    store = EvaluationArtifactStore(tmp_path / "fallback")
    assert store.append_record(manifest, record) is True
    assert store.load_records(manifest) == (record,)
    finding = FindingTruth(
        finding_id="finding",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        required_context=("changed_hunk",),
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(CheckpointTruth("case", False, _hash("fallback-truth"), (finding,)),),
    )
    assert score_matched_arms(manifest, truth, (record,)).arms[0].completed_case_count == 1


def test_artifact_store_retains_failed_attempts_and_resumes_without_replacement(tmp_path):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    store = EvaluationArtifactStore(tmp_path / "artifacts")
    failed = _record(
        manifest,
        case,
        "deterministic",
        state=EvaluationRunState.MALFORMED,
        terminal=False,
        findings=(),
    )
    assert store.append_record(manifest, failed) is True
    assert store.append_record(manifest, failed) is False

    resume = store.resume_plan(manifest)
    assert resume[0].next_attempt == 2
    assert resume[0].retained_attempt_ids == (failed.record_id,)

    success = _record(manifest, case, "deterministic", attempt=2)
    store.append_record(manifest, success)
    assert store.resume_plan(manifest) == ()
    assert [record.state for record in store.load_records(manifest)] == [
        EvaluationRunState.MALFORMED,
        EvaluationRunState.COMPLETED,
    ]
    inventory = store.inventory(manifest)
    assert inventory.terminal_pair_count == 1
    assert inventory.incomplete_pair_count == 0
    assert len(inventory.record_artifact_hashes) == 2

    record_path = next((tmp_path / "artifacts" / "records").glob("*.json"))
    record_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationValidationError, match="invalid immutable run artifact"):
        store.load_records(manifest)


def test_artifact_store_rejects_attempt_gaps_and_attempts_after_terminal(tmp_path):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))

    gap_store = EvaluationArtifactStore(tmp_path / "gap")
    with pytest.raises(EvaluationValidationError, match="omits a retained attempt"):
        gap_store.append_record(manifest, _record(manifest, case, "deterministic", attempt=2))
    assert not tuple((tmp_path / "gap" / "records").glob("*.json"))

    terminal_store = EvaluationArtifactStore(tmp_path / "terminal")
    terminal_store.append_record(manifest, _record(manifest, case, "deterministic"))
    with pytest.raises(EvaluationValidationError, match="attempts after a terminal record"):
        terminal_store.append_record(
            manifest,
            _record(
                manifest,
                case,
                "deterministic",
                attempt=2,
                state=EvaluationRunState.MALFORMED,
                terminal=False,
            ),
        )
    assert len(tuple((tmp_path / "terminal" / "records").glob("*.json"))) == 1


def test_artifact_store_rejects_a_record_for_the_wrong_snapshot(tmp_path):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    store = EvaluationArtifactStore(tmp_path / "wrong-snapshot")

    with pytest.raises(EvaluationValidationError, match="snapshot does not match"):
        store.append_record(
            manifest,
            replace(_record(manifest, case, "deterministic"), snapshot_id=_hash("other-snapshot")),
        )
    assert not tuple((tmp_path / "wrong-snapshot" / "records").glob("*.json"))


def test_shadow_journal_is_opt_in_source_free_and_non_blocking(tmp_path):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    record = _record(
        manifest,
        case,
        "deterministic",
        findings=(ObservedFinding("raw-sensitive-fingerprint", FindingSeverity.HIGH),),
    )
    entry = ShadowJournalEntry.from_run_record(
        record,
        event=case.event,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
        reason_codes=("sensitive_path",),
        selected_depth="deep",
    )

    disabled_path = tmp_path / "disabled.ndjson"
    disabled = ShadowJournalWriter(disabled_path, enabled=False)
    assert disabled.submit(entry) is ShadowSubmitStatus.DISABLED
    assert not disabled_path.exists()

    path = tmp_path / "shadow.ndjson"
    writer = ShadowJournalWriter(path, enabled=True, max_queue_entries=4)
    assert writer.submit(entry) is ShadowSubmitStatus.QUEUED
    assert writer.close()
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "raw-sensitive-fingerprint" not in serialized
    for forbidden in ("source", "diff", "secret", "reasoning", "request_id", "credential", "task_intent"):
        assert forbidden not in payload

    failed_record = _record(
        manifest,
        case,
        "deterministic",
        state=EvaluationRunState.PROVIDER_FAILURE,
        findings=(),
    )
    failed_entry = ShadowJournalEntry.from_run_record(
        failed_record,
        event=case.event,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
    )
    assert failed_entry.coverage_status is MeasurementStatus.UNAVAILABLE


def test_shadow_journal_close_retries_when_the_bounded_queue_was_full(tmp_path, monkeypatch):
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    entry = ShadowJournalEntry.from_run_record(
        _record(manifest, case, "deterministic"),
        event=case.event,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
    )
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocked_append(path, payload):
        worker_started.set()
        assert release_worker.wait(2)

    monkeypatch.setattr(
        "pr_agent.algo.checkpoint_shadow_journal._append_private_line",
        blocked_append,
    )
    writer = ShadowJournalWriter(tmp_path / "shadow.ndjson", enabled=True, max_queue_entries=1)
    assert writer.submit(entry) is ShadowSubmitStatus.QUEUED
    assert worker_started.wait(1)
    assert writer.submit(entry) is ShadowSubmitStatus.QUEUED
    assert writer.close(timeout_seconds=0.01) is False
    assert writer.submit(entry) is ShadowSubmitStatus.CLOSED

    release_worker.set()
    assert writer.close(timeout_seconds=1) is True
    assert writer._thread is not None and not writer._thread.is_alive()


def test_scorer_reports_lineage_lifecycle_events_stages_cohorts_and_paired_uncertainty():
    root = _case("root", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    child = _case("child", EvaluationCohort.HOLDOUT, ReviewEvent.WORKTREE_IDLE, 30, parent="root")
    fixed = _case("fixed", EvaluationCohort.HOLDOUT, ReviewEvent.PRE_COMMIT, 60, parent="child")
    control = _case("control", EvaluationCohort.CLEAN_CONTROL, ReviewEvent.FILE_SAVE, 0)
    arms = (
        _arm("deterministic", EvaluationArmKind.DETERMINISTIC),
        _arm("cascade", EvaluationArmKind.FULL_CASCADE),
    )
    manifest = _manifest((root, child, fixed, control), arms)
    truth_finding_root = FindingTruth(
        finding_id="finding-root",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        earliest_case_id="root",
        required_context=("changed_hunk",),
        withdrawn_at_case_id="fixed",
    )
    truth_finding_child = FindingTruth(
        finding_id="finding-child",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        earliest_case_id="root",
        required_context=("changed_hunk",),
        withdrawn_at_case_id="fixed",
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(
            CheckpointTruth("root", False, _hash("truth-root"), (truth_finding_root,)),
            CheckpointTruth("child", False, _hash("truth-child"), (truth_finding_child,)),
            CheckpointTruth("fixed", True, _hash("truth-fixed")),
            CheckpointTruth("control", True, _hash("truth-control")),
        ),
    )
    deterministic_records = (
        _record(manifest, root, "deterministic", escalated=True),
        _record(
            manifest,
            child,
            "deterministic",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH, deterministic_overlap=True),),
            escalated=True,
            latency=2,
        ),
        _record(
            manifest,
            fixed,
            "deterministic",
            findings=(
                ObservedFinding("bug", FindingSeverity.HIGH, FindingLifecycleState.WITHDRAWN),
            ),
            escalated=False,
        ),
        _record(manifest, control, "deterministic", escalated=False),
    )
    cascade_records = (
        _record(
            manifest,
            root,
            "cascade",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH, deterministic_overlap=False),),
            escalated=True,
        ),
        _record(
            manifest,
            child,
            "cascade",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH, deterministic_overlap=False),),
            escalated=True,
        ),
        _record(
            manifest,
            fixed,
            "cascade",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH),),
            escalated=False,
        ),
        _record(
            manifest,
            control,
            "cascade",
            findings=(ObservedFinding("noise", FindingSeverity.MEDIUM),),
            escalated=False,
        ),
    )

    scorecard = score_matched_arms(manifest, truth, deterministic_records + cascade_records)
    by_arm = {arm.arm_id: arm for arm in scorecard.arms}
    deterministic = by_arm["deterministic"]
    cascade = by_arm["cascade"]
    assert deterministic.metrics["incremental_recall_worktree_idle"].value == 1
    assert deterministic.metrics["time_to_first_valid_finding_p50_seconds"].value == 32
    assert deterministic.metrics["stale_findings_withdrawn_rate"].value == 1
    assert cascade.metrics["incremental_recall_file_save"].value == 1
    assert cascade.metrics["stale_findings_withdrawn_rate"].value == 0
    assert cascade.stale_finding_count == 1
    assert deterministic.metrics["latency_stage_router_p50_seconds"].status is MeasurementStatus.COMPLETE
    assert deterministic.metrics["false_interruptions_per_developer_hour"].status is MeasurementStatus.COMPLETE
    assert deterministic.metrics["cost_per_verified_finding"].value == pytest.approx(0.04)
    assert deterministic.metrics["high_critical_escalation_recall"].value == 1
    assert deterministic.cohort_metrics["holdout"]["case_support"].value == 3
    assert deterministic.cohort_metrics["clean_control"]["case_support"].value == 1
    assert {comparison.metric for comparison in scorecard.paired_comparisons} == {
        "case_recall",
        "cost_usd",
        "false_interruption_rate",
        "latency_seconds",
        "structured_output_rate",
        "tokens",
    }
    assert {comparison.metric: comparison.support for comparison in scorecard.paired_comparisons} == {
        "case_recall": 2,
        "cost_usd": 4,
        "false_interruption_rate": 2,
        "latency_seconds": 4,
        "structured_output_rate": 4,
        "tokens": 4,
    }
    publication_gate = evaluate_rollout_gate(
        "pr-publication",
        scorecard,
        "deterministic",
        (GateRule("verified_precision", GateComparator.AT_LEAST, 0.9),),
    )
    permission = evaluate_output_permission(
        OutputCapability.PR_PUBLICATION,
        (publication_gate,),
        arm_id="deterministic",
        scorecard_id=scorecard.scorecard_id,
        required_gate_spec_hash=publication_gate.gate_spec_hash,
    )
    permission.require_permitted()
    stale_permission = evaluate_output_permission(
        OutputCapability.PR_PUBLICATION,
        (publication_gate,),
        arm_id="deterministic",
        scorecard_id=_hash("stale-scorecard"),
        required_gate_spec_hash=publication_gate.gate_spec_hash,
    )
    with pytest.raises(EvaluationValidationError, match="not permitted"):
        stale_permission.require_permitted()
    weak_policy_permission = evaluate_output_permission(
        OutputCapability.PR_PUBLICATION,
        (publication_gate,),
        arm_id="deterministic",
        scorecard_id=scorecard.scorecard_id,
        required_gate_spec_hash=_hash("approved-stronger-policy"),
    )
    with pytest.raises(EvaluationValidationError, match="not permitted"):
        weak_policy_permission.require_permitted()

    paired_gate = evaluate_rollout_gate(
        "pr-publication",
        scorecard,
        "cascade",
        (GateRule("paired.case_recall.lower_95", GateComparator.AT_LEAST, -1.0),),
    )
    assert paired_gate.status is GateStatus.PASSED
    assert paired_gate.rule_results[0].observed.support == 2


def test_gate_decisions_cannot_claim_passed_when_their_evidence_failed():
    rule = GateRule("verified_precision", GateComparator.AT_LEAST, 0.9)
    observed = ScoreMetric(MeasurementStatus.COMPLETE, 0.5, 10)
    with pytest.raises(EvaluationValidationError, match="status does not match"):
        GateRuleResult(rule, GateStatus.PASSED, observed, "forged pass")

    failed = GateRuleResult(rule, GateStatus.FAILED, observed, "threshold not satisfied")
    with pytest.raises(EvaluationValidationError, match="status does not match"):
        RolloutGateDecision(
            gate_name="pr-publication",
            arm_id="cascade",
            scorecard_id=_hash("scorecard"),
            status=GateStatus.PASSED,
            rule_results=(failed,),
        )


def test_missing_runs_cannot_look_like_clean_zero_cost_evidence():
    defect = _case("defect", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    control = _case("control", EvaluationCohort.CLEAN_CONTROL, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest(
        (defect, control),
        (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),),
    )
    finding = FindingTruth(
        finding_id="finding",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        required_context=("changed_hunk",),
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(
            CheckpointTruth("defect", False, _hash("truth-defect"), (finding,)),
            CheckpointTruth("control", True, _hash("truth-control")),
        ),
    )

    arm = score_matched_arms(manifest, truth, ()).arms[0]
    assert arm.metrics["unavailable_coverage_rate"].value == 1
    assert arm.metrics["failure_or_missing_case_rate"].value == 1
    assert arm.metrics["false_interruptions_per_clean_checkpoint"].status is MeasurementStatus.PARTIAL
    assert arm.metrics["total_retries"].status is MeasurementStatus.UNAVAILABLE


def test_stale_withdrawal_uses_lineage_ancestry_when_timing_is_missing():
    root = _case("root", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, None)
    fixed = _case(
        "fixed",
        EvaluationCohort.HOLDOUT,
        ReviewEvent.WORKTREE_IDLE,
        None,
        parent="root",
    )
    later = _case(
        "later",
        EvaluationCohort.HOLDOUT,
        ReviewEvent.PRE_COMMIT,
        None,
        parent="fixed",
    )
    manifest = _manifest(
        (root, fixed, later),
        (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),),
    )
    finding = FindingTruth(
        finding_id="finding",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        earliest_case_id="root",
        required_context=("changed_hunk",),
        withdrawn_at_case_id="fixed",
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(
            CheckpointTruth("root", False, _hash("truth-root"), (finding,)),
            CheckpointTruth("fixed", True, _hash("truth-fixed")),
            CheckpointTruth("later", True, _hash("truth-later")),
        ),
    )
    records = (
        _record(manifest, root, "deterministic"),
        _record(manifest, fixed, "deterministic"),
        _record(
            manifest,
            later,
            "deterministic",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH),),
        ),
    )

    arm = score_matched_arms(manifest, truth, records).arms[0]
    assert arm.stale_finding_count == 1
    assert arm.metrics["stale_findings_withdrawn_rate"].value == 0


def test_earliest_lineage_opportunity_is_independent_of_manifest_order_without_timing():
    root = _case("root", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, None)
    child = _case(
        "child",
        EvaluationCohort.HOLDOUT,
        ReviewEvent.FILE_SAVE,
        None,
        parent="root",
    )
    manifest = _manifest(
        (child, root),
        (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),),
    )
    root_finding = FindingTruth(
        finding_id="root-finding",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        required_context=("changed_hunk",),
    )
    child_finding = replace(root_finding, finding_id="child-finding")
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(
            CheckpointTruth("child", False, _hash("truth-child"), (child_finding,)),
            CheckpointTruth("root", False, _hash("truth-root"), (root_finding,)),
        ),
    )
    records = (
        _record(manifest, child, "deterministic"),
        _record(
            manifest,
            root,
            "deterministic",
            findings=(ObservedFinding("bug", FindingSeverity.HIGH),),
        ),
    )

    arm = score_matched_arms(manifest, truth, records).arms[0]
    assert arm.metrics["incremental_recall_file_save"].value == 1
    assert arm.metrics["cost_per_verified_finding"].status is MeasurementStatus.COMPLETE


def test_truth_requires_earliest_event_to_match_when_case_id_is_implicit():
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.PRE_COMMIT, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    mismatched = FindingTruth(
        finding_id="finding",
        fingerprint="bug",
        severity=FindingSeverity.HIGH,
        earliest_opportunity=ReviewEvent.FILE_SAVE,
        required_context=("changed_hunk",),
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(CheckpointTruth("case", False, _hash("truth"), (mismatched,)),),
    )

    with pytest.raises(EvaluationValidationError, match="earliest_opportunity"):
        truth.validate_for_manifest(manifest)


def test_run_state_cannot_contradict_the_shipped_snapshot_result_state():
    case = _case("case", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest((case,), (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),))
    record = _record(manifest, case, "deterministic")

    with pytest.raises(EvaluationValidationError, match="contradicts"):
        replace(record, snapshot_result_state=ReviewResultState.COVERAGE_UNAVAILABLE)
    with pytest.raises(EvaluationValidationError, match="schema_hash"):
        replace(manifest, schema_hash=_hash("different-schema"))


def _write_json(path: Path, value) -> str:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return f"sha256:{__import__('hashlib').sha256(payload).hexdigest()}"


def test_cocos_adapter_reads_external_locked_corpus_without_copying(tmp_path, capsys):
    corpus = tmp_path / "cocos-corpus"
    corpus.mkdir()
    primary = {
        "repo": "sagacious-heritage/cocos-story",
        "entries": [
            {"id": "A01", "split": "calibration", "target_sha": "a" * 40},
            {"id": "B01", "split": "holdout", "target_sha": "b" * 40},
        ],
    }
    temporal = {
        "repo": "sagacious-heritage/cocos-story",
        "entries": [{"id": "T01", "target_sha": "c" * 40}],
    }
    controls = {
        "repo": "sagacious-heritage/cocos-story",
        "entries": [{"id": "C01", "target_sha": "d" * 40}],
    }
    annotations = {
        "repo": "sagacious-heritage/cocos-story",
        "answer_only": True,
        "snapshots": [{"snapshot_id": "A01"}, {"snapshot_id": "B01"}],
    }
    values = {
        "ledger.json": primary,
        "temporal-backtest-ledger.json": temporal,
        "controls-ledger.json": controls,
        "specialist-annotations.json": annotations,
    }
    artifact_hashes = {name: _write_json(corpus / name, value) for name, value in values.items()}
    assignment_payload = [
        {"id": "A01", "split": "calibration", "target_sha": "a" * 40},
        {"id": "C01", "split": "control", "target_sha": "d" * 40},
        {"id": "B01", "split": "holdout", "target_sha": "b" * 40},
        {"id": "T01", "split": "temporal", "target_sha": "c" * 40},
    ]
    lock = CocosCorpusLock(
        source_revision="e" * 40,
        artifact_hashes=artifact_hashes,
        assignment_hash=content_hash(assignment_payload),
        expected_cohort_counts={
            "calibration": 1,
            "holdout": 1,
            "temporal": 1,
            "control": 1,
            "unique_snapshots": 2,
        },
    )
    checkpoint_controls = tmp_path / "checkpoint-controls.json"
    scenarios = (
        "partial_correct_save",
        "coherent_clean_worktree",
        "temporary_mistake_corrected",
        "staged_checkpoint",
        "stale_candidate_withdrawn",
    )
    checkpoint_payload = {
        "schema_version": CHECKPOINT_CONTROL_SCHEMA_VERSION,
        "answer_only": True,
        "entries": [
            {
                "id": f"checkpoint-{index}",
                "snapshot_id": _hash(f"checkpoint-snapshot-{index}"),
                "snapshot_artifact_hash": _hash(f"checkpoint-artifact-{index}"),
                "stage": "file_save",
                "scenario": scenarios[index % len(scenarios)],
                "independently_adjudicated": True,
                "adjudication_hash": _hash(f"checkpoint-adjudication-{index}"),
                "is_clean": True,
                **(
                    {"parent_id": f"checkpoint-{index - 1}"}
                    if index % len(scenarios) in {2, 4}
                    else {}
                ),
                **(
                    {"expected_withdrawn_fingerprints": [f"finding-{index}"]}
                    if index % len(scenarios) == 4
                    else {}
                ),
            }
            for index in range(15)
        ]
    }
    _write_json(checkpoint_controls, checkpoint_payload)

    inventory = validate_cocos_story_corpus(
        corpus,
        lock,
        checkpoint_controls_path=checkpoint_controls,
    )
    assert inventory.checkpoint_control_count == 15
    assert inventory.checkpoint_controls_status == "complete"
    assert inventory.checkpoint_controls_hash is not None
    assert str(corpus) not in json.dumps(inventory.to_dict())

    for invalid_version in (None, "future-unknown-v99"):
        invalid_payload = dict(checkpoint_payload)
        if invalid_version is None:
            invalid_payload.pop("schema_version")
        else:
            invalid_payload["schema_version"] = invalid_version
        _write_json(checkpoint_controls, invalid_payload)
        with pytest.raises(EvaluationValidationError, match="checkpoint control schema_version"):
            validate_cocos_story_corpus(
                corpus,
                lock,
                checkpoint_controls_path=checkpoint_controls,
            )
    _write_json(checkpoint_controls, checkpoint_payload)

    case = _case("adapter", EvaluationCohort.HOLDOUT, ReviewEvent.FILE_SAVE, 0)
    manifest = _manifest(
        (case,),
        (_arm("deterministic", EvaluationArmKind.DETERMINISTIC),),
        corpus_hash=inventory.corpus_hash,
    )
    manifest_path = tmp_path / "manifest.json"
    lock_path = tmp_path / "cocos-lock.json"
    _write_json(manifest_path, manifest.to_dict())
    _write_json(lock_path, lock.to_dict())
    payload = run(inargs=[
        "evaluation-plan",
        "--manifest",
        str(manifest_path),
        "--cocos-corpus-root",
        str(corpus),
        "--cocos-lock",
        str(lock_path),
        "--checkpoint-controls",
        str(checkpoint_controls),
        "--dry-run",
    ])
    assert json.loads(capsys.readouterr().out) == payload
    assert payload["external_corpus"]["checkpoint_control_count"] == 15

    changed_controls = json.loads(checkpoint_controls.read_text(encoding="utf-8"))
    changed_controls["entries"][0]["id"] = "checkpoint-changed"
    _write_json(checkpoint_controls, changed_controls)
    changed_inventory = validate_cocos_story_corpus(
        corpus,
        lock,
        checkpoint_controls_path=checkpoint_controls,
    )
    assert changed_inventory.checkpoint_controls_hash != inventory.checkpoint_controls_hash
    assert changed_inventory.corpus_hash != inventory.corpus_hash

    _write_json(corpus / "ledger.json", {**primary, "entries": []})
    with pytest.raises(EvaluationValidationError, match="changed after the approved lock"):
        validate_cocos_story_corpus(corpus, lock)


def test_checked_in_cocos_lock_is_schema_valid():
    lock_path = Path("docs/docs/usage-guide/cocos_story_corpus_lock.json")
    lock = CocosCorpusLock.from_dict(json.loads(lock_path.read_text(encoding="utf-8")))

    assert lock.source_revision == "6b98bae67bae4056c4567187454e24cca78b9467"
    assert lock.expected_cohort_counts == {
        "calibration": 12,
        "control": 16,
        "holdout": 18,
        "temporal": 10,
        "unique_snapshots": 55,
    }


def test_checkpoint_evaluation_safety_defaults_are_disabled_and_mirrored():
    repository = tomllib.loads(Path(".pr_agent.toml").read_text(encoding="utf-8"))["checkpoint_evaluation"]
    defaults = tomllib.loads(
        Path("pr_agent/settings/configuration.toml").read_text(encoding="utf-8")
    )["checkpoint_evaluation"]

    assert repository == defaults
    assert repository == {
        "enabled": False,
        "allow_paid_execution": False,
        "shadow_journal_enabled": False,
        "publish_output": False,
        "paid_cost_cap_usd": 0.0,
        "shadow_journal_path": "",
        "shadow_journal_max_queue_entries": 256,
    }
