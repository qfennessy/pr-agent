import json
from decimal import Decimal

import pytest

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    CheckpointTruth,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationValidationError,
    FindingSeverity,
    FindingTruth,
    MeasurementStatus,
    NumericMeasurement,
    ObservedFinding,
    TruthArtifact,
    build_evaluation_plan,
    content_hash,
)
from pr_agent.algo.checkpoint_evaluation_scoring import (
    GateComparator,
    GateRule,
    MatchedArmScorecard,
    RolloutGateDecision,
    evaluate_rollout_gate,
    score_matched_arms,
)
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewResultState, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails
from pr_agent.cli import run


def _hash(value: str) -> str:
    return content_hash({"value": value})


def _case(
    case_id: str,
    cohort: EvaluationCohort,
    *,
    event: ReviewEvent = ReviewEvent.FILE_SAVE,
    parent_case_id: str | None = None,
    metadata=None,
) -> CheckpointCase:
    return CheckpointCase(
        case_id=case_id,
        snapshot_id=_hash(f"snapshot-{case_id}"),
        snapshot_artifact_hash=_hash(f"artifact-{case_id}"),
        event=event,
        cohort=cohort,
        parent_case_id=parent_case_id,
        model_visible_metadata=metadata or {"language": "python"},
    )


def _arm(arm_id: str = "deterministic") -> EvaluationArm:
    return EvaluationArm(
        arm_id=arm_id,
        kind=EvaluationArmKind.DETERMINISTIC,
        configuration_hash=_hash(f"configuration-{arm_id}"),
        prompt_hash=_hash(f"prompt-{arm_id}"),
    )


def _manifest(*cases: CheckpointCase, arms=None) -> EvaluationManifest:
    return EvaluationManifest(
        name="frozen-checkpoints",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        cases=tuple(cases),
        arms=tuple(arms or (_arm(),)),
    )


def _truth(case: CheckpointCase, *, clean: bool) -> CheckpointTruth:
    findings = () if clean else (
        FindingTruth(
            finding_id=f"finding-{case.case_id}",
            fingerprint=f"fingerprint-{case.case_id}",
            severity=FindingSeverity.HIGH,
            earliest_opportunity=case.event,
            required_context=("changed_hunk", "test_contract"),
        ),
    )
    return CheckpointTruth(
        case_id=case.case_id,
        is_clean=clean,
        adjudication_hash=_hash(f"adjudication-{case.case_id}"),
        findings=findings,
    )


def _record(
    manifest: EvaluationManifest,
    case: CheckpointCase,
    *,
    attempt: int = 1,
    state: EvaluationRunState = EvaluationRunState.COMPLETED,
    terminal: bool = True,
    findings=(),
    latency: float | None = 0.25,
    tokens: float | None = 10,
    cost: float | None = 0.01,
) -> EvaluationRunRecord:
    def measurement(value):
        if value is None:
            return NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        return NumericMeasurement(MeasurementStatus.COMPLETE, value)

    return EvaluationRunRecord(
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        arm_id="deterministic",
        snapshot_id=case.snapshot_id,
        attempt=attempt,
        state=state,
        terminal=terminal,
        findings=tuple(findings),
        latency_seconds=measurement(latency),
        tokens=measurement(tokens),
        cost_usd=measurement(cost),
    )


def test_manifest_round_trip_has_content_identity_and_no_answers():
    case = _case("holdout-1", EvaluationCohort.HOLDOUT)
    manifest = _manifest(case)

    reloaded = EvaluationManifest.from_dict(manifest.to_dict())

    assert reloaded == manifest
    assert reloaded.manifest_id == manifest.manifest_id
    assert "cohort" not in case.model_visible_payload()
    assert "case_id" not in case.model_visible_payload()
    assert "parent_case_id" not in case.model_visible_payload()
    serialized_input = json.dumps(case.model_visible_payload())
    assert "earliest_opportunity" not in serialized_input
    assert "required_context" not in serialized_input
    with pytest.raises(TypeError):
        case.model_visible_metadata["language"] = "ruby"


@pytest.mark.parametrize(
    "metadata",
    [
        {"ground_truth": "bug"},
        {"nested": {"severity": "critical"}},
        {"items": [{"required-context": ["caller"]}]},
    ],
)
def test_checkpoint_rejects_answer_only_model_metadata(metadata):
    with pytest.raises(EvaluationValidationError, match="answer-only fields"):
        _case("leaky", EvaluationCohort.HOLDOUT, metadata=metadata)


def test_manifest_rejects_unknown_fields_and_cross_cohort_lineage():
    case = _case("holdout-1", EvaluationCohort.HOLDOUT)
    manifest_value = _manifest(case).to_dict()
    manifest_value["truth"] = {"answer": "hidden"}
    with pytest.raises(EvaluationValidationError, match="unknown fields"):
        EvaluationManifest.from_dict(manifest_value)

    parent = _case("parent", EvaluationCohort.CALIBRATION)
    child = _case("child", EvaluationCohort.HOLDOUT, parent_case_id=parent.case_id)
    with pytest.raises(EvaluationValidationError, match="cannot cross"):
        _manifest(parent, child)


def test_truth_is_separate_and_must_match_manifest_cohorts():
    defect = _case("defect", EvaluationCohort.HOLDOUT)
    control = _case("control", EvaluationCohort.CLEAN_CONTROL)
    manifest = _manifest(defect, control)
    artifact = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(_truth(defect, clean=False), _truth(control, clean=True)),
    )

    artifact.validate_for_manifest(manifest)
    assert artifact.truth_artifact_id.startswith("sha256:")

    wrong = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(_truth(defect, clean=True), _truth(control, clean=True)),
    )
    with pytest.raises(EvaluationValidationError, match="clean truth"):
        wrong.validate_for_manifest(manifest)


def test_plan_is_stable_and_pairs_every_enabled_arm_with_same_snapshot():
    case = _case("case-b", EvaluationCohort.HOLDOUT)
    deterministic = _arm("deterministic")
    model_arm = EvaluationArm(
        arm_id="general",
        kind=EvaluationArmKind.GENERAL_REVIEW,
        configuration_hash=_hash("general-configuration"),
        prompt_hash=_hash("general-prompt"),
        model_id="openai/model-version-2026-08-30",
        provider_id="openai/responses-v1",
        model_revision="2026-08-30.1",
    )
    disabled = EvaluationArm(
        arm_id="disabled",
        kind=EvaluationArmKind.GENERAL_REVIEW,
        configuration_hash=_hash("disabled-configuration"),
        prompt_hash=_hash("disabled-prompt"),
        model_id="provider/model-version",
        provider_id="provider/api-v1",
        enabled=False,
    )
    manifest = _manifest(case, arms=(model_arm, disabled, deterministic))

    plan = build_evaluation_plan(manifest).to_dict()

    assert [item["arm_id"] for item in plan["items"]] == ["deterministic", "general"]
    assert {item["snapshot_id"] for item in plan["items"]} == {case.snapshot_id}
    assert plan["network_calls"] == 0
    assert plan["model_calls"] == 0
    assert next(item for item in plan["items"] if item["arm_id"] == "general")["model_revision"] == "2026-08-30.1"


def test_snapshot_result_adapter_preserves_partial_and_unavailable_telemetry():
    case = _case("case", EvaluationCohort.HOLDOUT)
    arm = _arm()
    manifest = _manifest(case, arms=(arm,))
    result = ReviewSnapshotResult(
        snapshot_id=case.snapshot_id,
        state=ReviewResultState.NO_FINDINGS,
        current_snapshot_id=case.snapshot_id,
        review={"review": {"key_issues_to_review": []}},
        coverage_issues=(),
        latency_seconds=0.5,
    )
    details = RunDetails(
        model_used="deterministic",
        prompt_tokens=7,
        completion_tokens=3,
        total_tokens=10,
        num_ai_calls=2,
        total_cost_usd=Decimal("0.02"),
        known_cost_call_count=1,
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

    assert record.state is EvaluationRunState.COMPLETED
    assert record.tokens.status is MeasurementStatus.PARTIAL
    assert record.tokens.value == 10
    assert record.cost_usd.status is MeasurementStatus.PARTIAL
    assert record.cost_usd.value == 0.02
    assert EvaluationRunRecord.from_dict(record.to_dict()) == record


def test_score_keeps_failures_and_missing_pairs_in_denominators():
    defect = _case("defect", EvaluationCohort.HOLDOUT)
    control = _case("control", EvaluationCohort.CLEAN_CONTROL)
    missing = _case("missing", EvaluationCohort.HOLDOUT)
    manifest = _manifest(defect, control, missing)
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(
            _truth(defect, clean=False),
            _truth(control, clean=True),
            _truth(missing, clean=False),
        ),
    )
    failed_attempt = _record(
        manifest,
        defect,
        attempt=1,
        state=EvaluationRunState.PROVIDER_FAILURE,
        terminal=False,
        latency=None,
        tokens=None,
        cost=None,
    )
    defect_success = _record(
        manifest,
        defect,
        attempt=2,
        findings=(ObservedFinding("fingerprint-defect", FindingSeverity.HIGH),),
    )
    control_false_positive = _record(
        manifest,
        control,
        findings=(ObservedFinding("unexpected", FindingSeverity.MEDIUM),),
    )

    scorecard = score_matched_arms(
        manifest,
        truth,
        (failed_attempt, defect_success, control_false_positive),
    )
    arm = scorecard.arms[0]

    assert arm.case_count == 3
    assert arm.attempt_count == 3
    assert arm.failed_attempt_count == 1
    assert arm.true_positive_count == 1
    assert arm.false_positive_count == 1
    assert arm.false_negative_count == 1
    assert arm.false_interruption_count == 1
    assert arm.metrics["structured_output_rate"].value == pytest.approx(2 / 3)
    assert arm.metrics["verified_precision"].value == 0.5
    assert arm.metrics["verified_recall"].value == 0.5
    assert arm.metrics["failure_attempt_rate"].value == pytest.approx(1 / 3)
    assert arm.metrics["total_tokens"].status is MeasurementStatus.PARTIAL


def test_gate_fails_known_bad_evidence_and_never_passes_missing_evidence():
    defect = _case("defect", EvaluationCohort.HOLDOUT)
    control = _case("control", EvaluationCohort.CLEAN_CONTROL)
    manifest = _manifest(defect, control)
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=(_truth(defect, clean=False), _truth(control, clean=True)),
    )
    records = (
        _record(
            manifest,
            defect,
            findings=(ObservedFinding("fingerprint-defect", FindingSeverity.HIGH),),
        ),
        _record(manifest, control),
    )
    scorecard = score_matched_arms(manifest, truth, records)

    passed = evaluate_rollout_gate(
        "offline-replay",
        scorecard,
        "deterministic",
        (
            GateRule("verified_precision", GateComparator.AT_LEAST, 0.8),
            GateRule("structured_output_rate", GateComparator.AT_LEAST, 0.995),
        ),
    )
    failed = evaluate_rollout_gate(
        "impossible",
        scorecard,
        "deterministic",
        (GateRule("verified_precision", GateComparator.AT_LEAST, 1.1),),
    )
    not_evaluable = evaluate_rollout_gate(
        "missing",
        scorecard,
        "deterministic",
        (GateRule("developer_hours", GateComparator.AT_MOST, 1.0),),
    )

    assert passed.status.value == "passed"
    assert failed.status.value == "failed"
    assert not_evaluable.status.value == "not_evaluable"
    assert not_evaluable.to_dict()["status"] == "not_evaluable"
    assert MatchedArmScorecard.from_dict(scorecard.to_dict()) == scorecard
    assert RolloutGateDecision.from_dict(passed.to_dict()) == passed


def test_cli_dry_run_does_not_enter_pr_agent_or_network_paths(tmp_path, monkeypatch, capsys):
    case = _case("holdout", EvaluationCohort.HOLDOUT)
    manifest = _manifest(case)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation planning must not invoke a model, provider, or network")

    monkeypatch.setattr("pr_agent.cli.PRAgent", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)

    payload = run(inargs=["evaluation-plan", "--manifest", str(manifest_path), "--dry-run"])

    stdout = json.loads(capsys.readouterr().out)
    assert stdout == payload
    assert stdout["network_calls"] == 0
    assert stdout["model_calls"] == 0
    assert stdout["items"][0]["snapshot_id"] == case.snapshot_id
