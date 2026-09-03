import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import pr_agent.algo.checkpoint_evaluation_report as report_module
from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    CheckpointTruth,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationStageModelIdentity,
    EvaluationStagePlan,
    EvaluationStageRun,
    EvaluationValidationError,
    FindingSeverity,
    FindingTruth,
    GateStatus,
    MeasurementStatus,
    NumericMeasurement,
    ObservedFinding,
    TruthArtifact,
    content_hash,
)
from pr_agent.algo.checkpoint_evaluation_cocos import CocosCorpusInventory
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactStore,
    PaidExecutionRequest,
    PaidPlanItemBudget,
)
from pr_agent.algo.checkpoint_evaluation_scoring import ScoreMetric, score_matched_arms
from pr_agent.algo.checkpoint_shadow_journal import DeveloperTimeBasis, ShadowJournalEntry, ShadowJournalSessionSummary
from pr_agent.algo.review_snapshot import ReviewEvent

DEFAULT_PAIR_REVIEW_GATE = report_module.DEFAULT_PAIR_REVIEW_GATE
LIVE_SHADOW_GATE = report_module.LIVE_SHADOW_GATE
OFFLINE_REPLAY_GATE = report_module.OFFLINE_REPLAY_GATE
OPT_IN_PAIR_REVIEW_GATE = report_module.OPT_IN_PAIR_REVIEW_GATE
PR_PUBLICATION_GATE = report_module.PR_PUBLICATION_GATE
PilotRolloutBudgets = report_module.PilotRolloutBudgets
PilotRolloutEvidence = report_module.PilotRolloutEvidence
SettledCandidateRecord = report_module.SettledCandidateRecord
ShadowJournalRecord = report_module.ShadowJournalRecord
build_checkpoint_pilot_report = report_module.build_checkpoint_pilot_report
build_cocos_pilot_acceptance = report_module.build_cocos_pilot_acceptance
build_replay_evidence_binding = report_module.build_replay_evidence_binding
build_holdout_leakage_check = report_module.build_holdout_leakage_check
build_settled_pilot_acceptance = report_module.build_settled_pilot_acceptance
build_shadow_pilot_acceptance = report_module.build_shadow_pilot_acceptance
canonical_rollout_gate_specs = report_module.canonical_rollout_gate_specs

_PINNED_TEST_COCOS_ACCEPTANCE_ID = (
    "sha256:d934504cb16b48f81b661fae4962c07fe26d63adef67668e100845a99b2b2459"
)
_PINNED_TEST_SHADOW_ACCEPTANCE_ID = (
    "sha256:509975310b252bc83d165ca343c7374888377e3a527ef913cf8daa1cd8822cc8"
)
_PINNED_TEST_SETTLED_ACCEPTANCE_ID = (
    "sha256:fccce140bdb6b685f43e30b49bc38be04101bb6b09b91375c0a8a5312ce8693b"
)


def _hash(value: str) -> str:
    return content_hash({"value": value})


def _manifest() -> EvaluationManifest:
    calibration = tuple(
        CheckpointCase(
            case_id=f"calibration-{index}",
            snapshot_id=_hash(f"snapshot-calibration-{index}"),
            snapshot_artifact_hash=_hash(f"artifact-calibration-{index}"),
            event=ReviewEvent.FILE_SAVE,
            cohort=EvaluationCohort.CALIBRATION,
            lineage_elapsed_seconds=0,
            developer_elapsed_seconds=60,
        )
        for index in range(12)
    )
    holdouts = tuple(
        CheckpointCase(
            case_id=f"holdout-{index}",
            snapshot_id=_hash(f"snapshot-holdout-{index}"),
            snapshot_artifact_hash=_hash(f"artifact-holdout-{index}"),
            event=ReviewEvent.FILE_SAVE,
            cohort=EvaluationCohort.HOLDOUT,
            lineage_elapsed_seconds=0,
            developer_elapsed_seconds=60,
        )
        for index in range(18)
    )
    temporal = tuple(
        CheckpointCase(
            case_id=f"temporal-{index}",
            snapshot_id=_hash(f"snapshot-temporal-{index}"),
            snapshot_artifact_hash=_hash(f"artifact-temporal-{index}"),
            event=ReviewEvent.WORKTREE_IDLE,
            cohort=EvaluationCohort.TEMPORAL,
            lineage_elapsed_seconds=0,
            developer_elapsed_seconds=60,
        )
        for index in range(10)
    )
    controls = tuple(
        CheckpointCase(
            case_id=f"control-{index}",
            snapshot_id=_hash(f"snapshot-control-{index}"),
            snapshot_artifact_hash=_hash(f"artifact-control-{index}"),
            event=ReviewEvent.PRE_COMMIT,
            cohort=EvaluationCohort.CLEAN_CONTROL,
            lineage_elapsed_seconds=0,
            developer_elapsed_seconds=60,
        )
        for index in range(16)
    )
    incumbent = EvaluationArm(
        arm_id="incumbent",
        kind=EvaluationArmKind.GENERAL_REVIEW,
        configuration_hash=_hash("configuration-incumbent"),
        prompt_hash=_hash("prompt-incumbent"),
        model_id="incumbent-model",
        provider_id="incumbent-service",
        model_revision="2026-08-30.1",
    )
    cascade = EvaluationArm(
        arm_id="cascade",
        kind=EvaluationArmKind.FULL_CASCADE,
        configuration_hash=_hash("configuration-cascade"),
        prompt_hash=_hash("prompt-cascade"),
        model_id="cascade-model",
        provider_id="cascade-service",
        model_revision="2026-08-30.2",
        stage_plan=(EvaluationStagePlan(
            stage="frontier_adjudication",
            model_route=(EvaluationStageModelIdentity(
                model_id="cascade-model",
                provider_id="cascade-service",
                model_revision="2026-08-30.2",
            ),),
            configuration_hash=_hash("cascade-stage-configuration"),
            prompt_hash=_hash("cascade-stage-prompt"),
            prompt_version="cascade-stage-prompt-v1",
            input_schema_version="cascade-stage-input-v1",
            output_schema_version="cascade-stage-output-v1",
        ),),
    )
    return EvaluationManifest(
        name="pilot",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        cases=(*calibration, *holdouts, *temporal, *controls),
        arms=(incumbent, cascade),
    )


def _truth(manifest: EvaluationManifest) -> TruthArtifact:
    return TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=tuple(
            CheckpointTruth(case.case_id, True, _hash(f"truth-{case.case_id}"))
            if case.cohort is EvaluationCohort.CLEAN_CONTROL
            else CheckpointTruth(
                case.case_id,
                False,
                _hash(f"truth-{case.case_id}"),
                (FindingTruth(
                    finding_id=f"finding-{case.case_id}",
                    fingerprint="fingerprint",
                    severity=FindingSeverity.HIGH,
                    earliest_opportunity=case.event,
                    required_context=("changed_hunk",),
                ),),
            )
            for case in manifest.cases
        ),
    )


def _record(manifest: EvaluationManifest, case: CheckpointCase, arm: EvaluationArm) -> EvaluationRunRecord:
    findings = (
        ()
        if case.cohort is EvaluationCohort.CLEAN_CONTROL
        or (case.case_id.startswith("holdout-") and arm.arm_id == "incumbent")
        else (ObservedFinding("fingerprint", FindingSeverity.HIGH),)
    )
    stage_runs = ()
    if arm.stage_plan:
        stage_plan = arm.stage_plan[0]
        stage_runs = (EvaluationStageRun(
            stage=stage_plan.stage,
            state="success",
            coverage_status=MeasurementStatus.COMPLETE,
            model_id=arm.model_id,
            provider_id=arm.provider_id,
            model_revision=arm.model_revision,
            deployment_id_hash=None,
            configuration_hash=stage_plan.configuration_hash,
            prompt_hash=stage_plan.prompt_hash,
            prompt_version=stage_plan.prompt_version,
            input_schema_version=stage_plan.input_schema_version,
            output_schema_version=stage_plan.output_schema_version,
            latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.5),
            tokens=NumericMeasurement(MeasurementStatus.COMPLETE, 10),
            cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.01),
            cost_by_model_usd={arm.model_id: 0.01},
            ai_call_count=1,
            confidence=0.9,
        ),)
    return EvaluationRunRecord(
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        arm_id=arm.arm_id,
        snapshot_id=case.snapshot_id,
        attempt=1,
        state=EvaluationRunState.COMPLETED,
        terminal=True,
        findings=findings,
        latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.5),
        tokens=NumericMeasurement(MeasurementStatus.COMPLETE, 10),
        cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.01),
        retry_count=0,
        model_id=arm.model_id,
        provider_id=arm.provider_id,
        model_revision=arm.model_revision,
        stage_runs=stage_runs,
    )


def _store(tmp_path, manifest: EvaluationManifest) -> EvaluationArtifactStore:
    store = EvaluationArtifactStore(tmp_path / "artifacts")
    paid_arms = tuple(
        arm for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    )
    store.bind_paid_request(
        manifest,
        PaidExecutionRequest(
            manifest_id=manifest.manifest_id,
            cost_cap_usd=10.0,
            plan_item_budgets=tuple(
                PaidPlanItemBudget(case.case_id, arm.arm_id, 0.02, 2)
                for case in manifest.cases
                for arm in paid_arms
            ),
            credential_present_by_provider={
                arm.provider_id: True for arm in paid_arms if arm.provider_id is not None
            },
        ),
    )
    for arm in manifest.arms:
        for case in manifest.cases:
            store.append_record(manifest, _record(manifest, case, arm))
    return store


def _complete_evidence() -> PilotRolloutEvidence:
    return PilotRolloutEvidence()


def _budgets() -> PilotRolloutBudgets:
    return PilotRolloutBudgets(
        shadow_latency_p95_seconds=1.0,
        shadow_cost_per_developer_hour_usd=0.20,
        accepted_false_interruptions_per_clean_checkpoint=0.05,
        publication_cost_ceiling_usd=2.0,
    )


def _cocos_inventory(manifest: EvaluationManifest) -> CocosCorpusInventory:
    return CocosCorpusInventory(
        lock_id="sha256:4db5b13a4f6240204350274d5147103f591a4216bfadc2560bca4a6d9ce7df13",
        source_revision="6b98bae67bae4056c4567187454e24cca78b9467",
        corpus_hash=manifest.corpus_hash,
        cohort_counts={
            "calibration": 12,
            "holdout": 18,
            "temporal": 10,
            "control": 16,
            "confirmation": 16,
            "unique_snapshots": 55,
        },
        checkpoint_control_count=16,
        checkpoint_controls_status="complete",
        checkpoint_controls_hash=_hash("checkpoint-controls"),
        root_identity=_hash("private-root"),
    )


def _shadow_records(
    manifest: EvaluationManifest,
    *,
    started_at: datetime | None = None,
) -> tuple[ShadowJournalRecord, ...]:
    started_at = started_at or datetime(2026, 8, 1, tzinfo=timezone.utc)
    arm = next(arm for arm in manifest.arms if arm.arm_id == "cascade")

    def record(index: int, event: ReviewEvent, observed_at: datetime) -> ShadowJournalRecord:
        stage_plan = arm.stage_plan[0]
        latency = 0.5 + index / 10
        stage_run = EvaluationStageRun(
            stage=stage_plan.stage,
            state="success",
            coverage_status=MeasurementStatus.COMPLETE,
            model_id=arm.model_id,
            provider_id=arm.provider_id,
            model_revision=arm.model_revision,
            deployment_id_hash=None,
            configuration_hash=stage_plan.configuration_hash,
            prompt_hash=stage_plan.prompt_hash,
            prompt_version=stage_plan.prompt_version,
            input_schema_version=stage_plan.input_schema_version,
            output_schema_version=stage_plan.output_schema_version,
            latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, latency),
            tokens=NumericMeasurement(MeasurementStatus.COMPLETE, 10),
            cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.001),
            cost_by_model_usd={arm.model_id: 0.001},
            ai_call_count=1,
            confidence=0.9,
        )
        entry = ShadowJournalEntry(
            snapshot_id=_hash(f"shadow-snapshot-{index}"),
            event=event,
            policy_hash=manifest.policy_hash,
            configuration_hash=manifest.configuration_hash,
            arm_id=arm.arm_id,
            model_id=arm.model_id,
            provider_id=arm.provider_id,
            model_revision=arm.model_revision,
            result_state=EvaluationRunState.COMPLETED,
            coverage_status=MeasurementStatus.COMPLETE,
            stage_runs=(stage_run,),
            latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, latency),
            tokens=NumericMeasurement(MeasurementStatus.COMPLETE, 10),
            cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.001),
        )
        return ShadowJournalRecord(
            sequence=index + 1,
            ingested_at_utc=observed_at.astimezone(timezone.utc),
            developer_time_basis=(
                DeveloperTimeBasis.WRITER_START
                if index == 0
                else DeveloperTimeBasis.WRITER_MONOTONIC
            ),
            entry=entry,
            developer_elapsed_seconds=None if index == 0 else 60,
            session_summary=(
                ShadowJournalSessionSummary(
                    submitted_entry_count=2,
                    queued_entry_count=2,
                    dropped_entry_count=0,
                    writer_failed=False,
                )
                if index == 1
                else None
            ),
        )

    return (
        record(0, ReviewEvent.FILE_SAVE, started_at),
        record(1, ReviewEvent.WORKTREE_IDLE, started_at + timedelta(days=8)),
    )


def _persist_shadow_records(tmp_path, records):
    payload = "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records)
    path = tmp_path / f"shadow-{content_hash({'payload': payload}).removeprefix('sha256:')}.ndjson"
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return path


def _settled_candidates(count: int = 100) -> tuple[SettledCandidateRecord, ...]:
    return tuple(
        SettledCandidateRecord(
            candidate_id=_hash(f"settled-candidate-{index}"),
            adjudication_hash=_hash(f"settled-adjudication-{index}"),
            actionable=index < 95,
        )
        for index in range(count)
    )


def _report(
    tmp_path,
    evidence=None,
    budgets=None,
    *,
    bind_corpus=True,
    pin_corpus_acceptance=False,
    pin_leakage_acceptance=False,
    pin_shadow_acceptance=False,
    pin_settled_acceptance=False,
    monkeypatch=None,
    shadow_records=None,
    settled_candidate_records=None,
    pinned_shadow_acceptance_id=_PINNED_TEST_SHADOW_ACCEPTANCE_ID,
    pinned_settled_acceptance_id=_PINNED_TEST_SETTLED_ACCEPTANCE_ID,
):
    manifest = _manifest()
    if evidence is not None:
        leakage_check = build_holdout_leakage_check(
            manifest,
            checker_revision_hash=_hash("leakage-checker-revision"),
            leakage_free=True,
        )
        evidence = replace(evidence, holdout_leakage_check=leakage_check)
        if pin_leakage_acceptance:
            if monkeypatch is None:
                raise AssertionError("pinning the leakage check requires monkeypatch")
            monkeypatch.setattr(
                report_module,
                "CANONICAL_HOLDOUT_LEAKAGE_CHECK_ID",
                leakage_check.check_id,
            )
    truth = _truth(manifest)
    store = _store(tmp_path, manifest)
    scorecard = score_matched_arms(
        manifest,
        truth,
        store.load_records(manifest),
        baseline_arm_id="incumbent",
    )
    replay_binding = build_replay_evidence_binding(
        manifest,
        scorecard,
        store.inventory(manifest),
        target_arm_id="cascade",
        incumbent_arm_id="incumbent",
    )
    if evidence is not None:
        evidence = replace(evidence, replay_binding=replay_binding)
    cocos_inventory = _cocos_inventory(manifest) if bind_corpus else None
    cocos_acceptance = (
        build_cocos_pilot_acceptance(manifest, cocos_inventory)
        if cocos_inventory is not None
        else None
    )
    if pin_corpus_acceptance:
        if monkeypatch is None or cocos_acceptance is None:
            raise AssertionError("pinning the acceptance requires monkeypatch and a corpus")
        assert cocos_acceptance.acceptance_id == _PINNED_TEST_COCOS_ACCEPTANCE_ID
        monkeypatch.setattr(
            report_module,
            "CANONICAL_COCOS_PILOT_ACCEPTANCE_ID",
            _PINNED_TEST_COCOS_ACCEPTANCE_ID,
        )
    shadow_records = tuple(shadow_records or _shadow_records(manifest))
    shadow_acceptance = build_shadow_pilot_acceptance(
        shadow_records,
        manifest=manifest,
        target_arm_id="cascade",
    )
    if pin_shadow_acceptance:
        if monkeypatch is None:
            raise AssertionError("pinning the shadow acceptance requires monkeypatch")
        assert shadow_acceptance.acceptance_id == pinned_shadow_acceptance_id
        monkeypatch.setattr(
            report_module,
            "CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID",
            pinned_shadow_acceptance_id,
        )
    settled_candidate_records = tuple(
        settled_candidate_records or _settled_candidates()
    )
    settled_candidate_acceptance = build_settled_pilot_acceptance(
        settled_candidate_records,
        manifest=manifest,
        target_arm_id="cascade",
    )
    if pin_settled_acceptance:
        if monkeypatch is None:
            raise AssertionError("pinning the settled acceptance requires monkeypatch")
        assert settled_candidate_acceptance.acceptance_id == pinned_settled_acceptance_id
        monkeypatch.setattr(
            report_module,
            "CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID",
            pinned_settled_acceptance_id,
        )
    return build_checkpoint_pilot_report(
        manifest,
        truth,
        store,
        target_arm_id="cascade",
        incumbent_arm_id="incumbent",
        budgets=budgets or PilotRolloutBudgets(),
        evidence=evidence or PilotRolloutEvidence(),
        cocos_inventory=cocos_inventory,
        cocos_acceptance=cocos_acceptance,
        shadow_journal_path=_persist_shadow_records(tmp_path, shadow_records),
        shadow_acceptance=shadow_acceptance,
        settled_candidate_records=settled_candidate_records,
        settled_candidate_acceptance=settled_candidate_acceptance,
    )


def test_canonical_gate_specs_freeze_issue_27_thresholds():
    specs = {spec.gate_name: spec for spec in canonical_rollout_gate_specs(_budgets())}

    assert tuple(specs) == (
        OFFLINE_REPLAY_GATE,
        LIVE_SHADOW_GATE,
        OPT_IN_PAIR_REVIEW_GATE,
        DEFAULT_PAIR_REVIEW_GATE,
        PR_PUBLICATION_GATE,
    )
    by_gate = {
        gate: {rule.metric: (rule.comparator.value, rule.threshold, rule.minimum_support) for rule in spec.rules}
        for gate, spec in specs.items()
    }
    assert by_gate[OFFLINE_REPLAY_GATE]["structured_output_rate"] == ("at_least", 0.995, 1)
    assert by_gate[LIVE_SHADOW_GATE]["evidence.shadow_elapsed_days"] == ("at_least", 7.0, 1)
    assert by_gate[OPT_IN_PAIR_REVIEW_GATE]["verified_precision"] == ("at_least", 0.80, 1)
    assert by_gate[DEFAULT_PAIR_REVIEW_GATE]["evidence.settled_actionable_precision"] == (
        "at_least",
        0.90,
        100,
    )
    assert by_gate[DEFAULT_PAIR_REVIEW_GATE]["evidence.settled_candidate_inventory_complete"] == (
        "at_least",
        1.0,
        1,
    )
    assert by_gate[PR_PUBLICATION_GATE]["evidence.holdout_cost_usd"] == ("at_most", 2.0, 1)
    assert by_gate[PR_PUBLICATION_GATE]["evidence.holdout_quality_advantage_lower_95"] == (
        "at_least",
        1e-12,
        18,
    )


def test_genuine_pinned_canonical_assignments_journal_and_settled_candidates_pass_all_five_gates(
    tmp_path,
    monkeypatch,
):
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_corpus_acceptance=True,
        pin_leakage_acceptance=True,
        pin_shadow_acceptance=True,
        pin_settled_acceptance=True,
        monkeypatch=monkeypatch,
    )

    assert [decision.gate_name for decision in report.gate_decisions] == [
        OFFLINE_REPLAY_GATE,
        LIVE_SHADOW_GATE,
        OPT_IN_PAIR_REVIEW_GATE,
        DEFAULT_PAIR_REVIEW_GATE,
        PR_PUBLICATION_GATE,
    ]
    assert {decision.status for decision in report.gate_decisions} == {GateStatus.PASSED}, [
        (
            decision.gate_name,
            decision.status,
            [result.to_dict() for result in decision.rule_results if result.status is not GateStatus.PASSED],
        )
        for decision in report.gate_decisions
        if decision.status is not GateStatus.PASSED
    ]
    assert report.incomplete_pair_count == 0
    assert report.terminal_pair_count == 112
    assert len(report.raw_record_artifact_hashes) == 112
    assert report.shadow_binding is not None
    assert report.shadow_binding.elapsed_days().value == 8
    assert report.shadow_binding.latency_p95_seconds.value == 0.6
    assert report.shadow_binding.cost_per_developer_hour_usd.status is MeasurementStatus.COMPLETE
    assert report.shadow_binding.cost_per_developer_hour_usd.value == pytest.approx(0.12)
    assert report.settled_candidate_binding is not None
    assert report.settled_candidate_binding.settled_count == 100
    assert report.settled_candidate_binding.actionable_count == 95
    assert report.settled_candidate_binding.actionable_precision.value == 0.95
    assert report.to_dict()["report_id"] == report.report_id


def test_temporal_and_publication_decisions_are_derived_from_frozen_attempts(
    tmp_path,
    monkeypatch,
):
    manifest = _manifest()
    truth = _truth(manifest)
    store = EvaluationArtifactStore(tmp_path / "derived")
    for arm in manifest.arms:
        for case in manifest.cases:
            record = _record(manifest, case, arm)
            if case.case_id.startswith("holdout-") and arm.arm_id == "incumbent":
                record = replace(record, findings=(ObservedFinding("fingerprint", FindingSeverity.HIGH),))
            if case.case_id == "temporal-0" and arm.arm_id == "cascade":
                record = replace(record, findings=())
            store.append_record(manifest, record)
    scorecard = score_matched_arms(
        manifest,
        truth,
        store.load_records(manifest),
        baseline_arm_id="incumbent",
    )
    evidence = replace(
        _complete_evidence(),
        replay_binding=build_replay_evidence_binding(
            manifest,
            scorecard,
            store.inventory(manifest),
            target_arm_id="cascade",
            incumbent_arm_id="incumbent",
        ),
    )

    cocos_inventory = _cocos_inventory(manifest)
    cocos_acceptance = build_cocos_pilot_acceptance(manifest, cocos_inventory)
    assert cocos_acceptance.acceptance_id == _PINNED_TEST_COCOS_ACCEPTANCE_ID
    monkeypatch.setattr(
        report_module,
        "CANONICAL_COCOS_PILOT_ACCEPTANCE_ID",
        _PINNED_TEST_COCOS_ACCEPTANCE_ID,
    )
    report = build_checkpoint_pilot_report(
        manifest,
        truth,
        store,
        target_arm_id="cascade",
        incumbent_arm_id="incumbent",
        budgets=_budgets(),
        evidence=evidence,
        cocos_inventory=cocos_inventory,
        cocos_acceptance=cocos_acceptance,
    )
    decisions = {decision.gate_name: decision for decision in report.gate_decisions}

    assert decisions[OPT_IN_PAIR_REVIEW_GATE].status is GateStatus.FAILED
    assert decisions[PR_PUBLICATION_GATE].status is GateStatus.FAILED
    publication_quality = next(
        result
        for result in decisions[PR_PUBLICATION_GATE].rule_results
        if result.rule.metric == "evidence.holdout_quality_advantage_lower_95"
    )
    assert publication_quality.observed is not None
    assert publication_quality.observed.value == 0


@pytest.mark.parametrize("attribute", ["latency_seconds", "tokens", "cost_usd"])
def test_negative_replay_aggregate_measurements_are_rejected_before_rollout(
    tmp_path,
    attribute,
):
    manifest = _manifest()
    truth = _truth(manifest)
    store = EvaluationArtifactStore(tmp_path / f"negative-{attribute}")
    for arm in manifest.arms:
        for case in manifest.cases:
            record = _record(manifest, case, arm)
            if arm.arm_id == "cascade" and case.cohort is EvaluationCohort.HOLDOUT:
                record = replace(
                    record,
                    **{attribute: NumericMeasurement(MeasurementStatus.COMPLETE, -100)},
                )
            store.append_record(manifest, record)

    with pytest.raises(EvaluationValidationError, match=f"replay {attribute} cannot be negative"):
        build_checkpoint_pilot_report(
            manifest,
            truth,
            store,
            target_arm_id="cascade",
            incumbent_arm_id="incumbent",
            budgets=_budgets(),
            evidence=_complete_evidence(),
        )


def test_missing_gate_evidence_is_not_evaluable_and_never_becomes_zero(tmp_path):
    report = _report(tmp_path)

    decisions = {decision.gate_name: decision for decision in report.gate_decisions}
    assert decisions[OFFLINE_REPLAY_GATE].status is GateStatus.NOT_EVALUABLE
    assert decisions[LIVE_SHADOW_GATE].status is GateStatus.NOT_EVALUABLE
    assert decisions[OPT_IN_PAIR_REVIEW_GATE].status is GateStatus.PASSED
    assert decisions[DEFAULT_PAIR_REVIEW_GATE].status is GateStatus.NOT_EVALUABLE
    assert decisions[PR_PUBLICATION_GATE].status is GateStatus.NOT_EVALUABLE
    missing = next(
        result
        for result in decisions[LIVE_SHADOW_GATE].rule_results
        if result.rule.metric == "evidence.shadow_elapsed_days"
    )
    assert missing.observed is not None
    assert missing.observed.status is MeasurementStatus.UNAVAILABLE
    assert missing.observed.value is None


def test_generated_but_unpinned_corpus_acceptance_cannot_publish(tmp_path):
    report = _report(tmp_path, _complete_evidence(), _budgets())
    publication = next(
        decision for decision in report.gate_decisions if decision.gate_name == PR_PUBLICATION_GATE
    )

    assert publication.status is GateStatus.NOT_EVALUABLE
    binding = next(
        result
        for result in publication.rule_results
        if result.rule.metric == "evidence.frozen_holdout_binding_complete"
    )
    assert binding.observed is not None
    assert binding.observed.status is MeasurementStatus.UNAVAILABLE


@pytest.mark.parametrize("mutation", ["substituted", "reordered", "missing", "extra"])
def test_manifest_must_exactly_match_the_pinned_cocos_assignment_inventory(
    tmp_path,
    monkeypatch,
    mutation,
):
    manifest = _manifest()
    truth = _truth(manifest)
    store = _store(tmp_path, manifest)
    inventory = _cocos_inventory(manifest)
    acceptance = build_cocos_pilot_acceptance(manifest, inventory)
    assert acceptance.acceptance_id == _PINNED_TEST_COCOS_ACCEPTANCE_ID
    if mutation == "substituted":
        changed_assignments = (
            replace(
                acceptance.assignments[0],
                snapshot_artifact_hash=_hash("substituted-snapshot"),
            ),
            *acceptance.assignments[1:],
        )
    elif mutation == "reordered":
        changed_assignments = (
            acceptance.assignments[1],
            acceptance.assignments[0],
            *acceptance.assignments[2:],
        )
    elif mutation == "missing":
        changed_assignments = acceptance.assignments[:-1]
    else:
        changed_assignments = (*acceptance.assignments, replace(
            acceptance.assignments[-1],
            case_id="extra-control",
            snapshot_id=_hash("extra-control-snapshot"),
            snapshot_artifact_hash=_hash("extra-control-artifact"),
        ))
    if mutation in {"missing", "extra"}:
        with pytest.raises(EvaluationValidationError, match="cohort counts"):
            replace(acceptance, assignments=changed_assignments)
        return
    changed_acceptance = replace(acceptance, assignments=changed_assignments)
    monkeypatch.setattr(
        report_module,
        "CANONICAL_COCOS_PILOT_ACCEPTANCE_ID",
        changed_acceptance.acceptance_id,
    )

    with pytest.raises(EvaluationValidationError, match="assignments do not exactly match"):
        build_checkpoint_pilot_report(
            manifest,
            truth,
            store,
            target_arm_id="cascade",
            incumbent_arm_id="incumbent",
            budgets=_budgets(),
            evidence=_complete_evidence(),
            cocos_inventory=inventory,
            cocos_acceptance=changed_acceptance,
        )


def test_more_than_the_exact_18_locked_holdout_cases_is_rejected(tmp_path):
    base_manifest = _manifest()
    extra_case = CheckpointCase(
        case_id="holdout-18",
        snapshot_id=_hash("snapshot-holdout-18"),
        snapshot_artifact_hash=_hash("artifact-holdout-18"),
        event=ReviewEvent.FILE_SAVE,
        cohort=EvaluationCohort.HOLDOUT,
        lineage_elapsed_seconds=0,
        developer_elapsed_seconds=60,
    )
    manifest = EvaluationManifest(
        name=base_manifest.name,
        corpus_hash=base_manifest.corpus_hash,
        policy_hash=base_manifest.policy_hash,
        configuration_hash=base_manifest.configuration_hash,
        cases=(*base_manifest.cases, extra_case),
        arms=base_manifest.arms,
    )
    with pytest.raises(EvaluationValidationError, match="cohort counts"):
        build_cocos_pilot_acceptance(manifest, _cocos_inventory(manifest))


def test_canonical_corpus_requires_complete_intermediate_controls():
    manifest = _manifest()
    inventory = replace(
        _cocos_inventory(manifest),
        checkpoint_control_count=None,
        checkpoint_controls_status="not_evaluable",
        checkpoint_controls_hash=None,
    )

    with pytest.raises(EvaluationValidationError, match="15 to 20 complete checkpoint controls"):
        build_cocos_pilot_acceptance(manifest, inventory)


def test_partial_or_small_denominator_stays_not_evaluable(tmp_path, monkeypatch):
    records = _settled_candidates(99)
    manifest = _manifest()
    acceptance_id = build_settled_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        pin_settled_acceptance=True,
        monkeypatch=monkeypatch,
        settled_candidate_records=records,
        pinned_settled_acceptance_id=acceptance_id,
    )
    decisions = {decision.gate_name: decision for decision in report.gate_decisions}

    assert decisions[OPT_IN_PAIR_REVIEW_GATE].status is GateStatus.PASSED
    assert decisions[DEFAULT_PAIR_REVIEW_GATE].status is GateStatus.NOT_EVALUABLE
    default_precision = next(
        result
        for result in decisions[DEFAULT_PAIR_REVIEW_GATE].rule_results
        if result.rule.metric == "evidence.settled_actionable_precision"
    )
    assert default_precision.reason == "metric support is below the required minimum"


def test_caller_asserted_settled_precision_is_not_accepted():
    with pytest.raises(TypeError):
        PilotRolloutEvidence(
            settled_actionable_precision=ScoreMetric(MeasurementStatus.COMPLETE, 2.0, 100),
        )


def test_generated_but_unpinned_settled_acceptance_cannot_pass(tmp_path):
    report = _report(tmp_path, _complete_evidence(), _budgets())
    default_pair = next(
        decision
        for decision in report.gate_decisions
        if decision.gate_name == DEFAULT_PAIR_REVIEW_GATE
    )
    inventory = next(
        result
        for result in default_pair.rule_results
        if result.rule.metric == "evidence.settled_candidate_inventory_complete"
    )

    assert default_pair.status is GateStatus.NOT_EVALUABLE
    assert inventory.observed is not None
    assert inventory.observed.status is MeasurementStatus.UNAVAILABLE


@pytest.mark.parametrize("mutation", ["truncated", "extra", "substituted", "reordered"])
def test_settled_inventory_rejects_truncated_extra_substituted_or_reordered_records(
    tmp_path,
    monkeypatch,
    mutation,
):
    manifest = _manifest()
    truth = _truth(manifest)
    store = _store(tmp_path, manifest)
    records = _settled_candidates()
    acceptance = build_settled_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    )
    monkeypatch.setattr(
        report_module,
        "CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID",
        acceptance.acceptance_id,
    )
    if mutation == "truncated":
        changed_records = records[:-1]
    elif mutation == "extra":
        changed_records = (*records, SettledCandidateRecord(
            candidate_id=_hash("settled-extra-candidate"),
            adjudication_hash=_hash("settled-extra-adjudication"),
            actionable=True,
        ))
    elif mutation == "substituted":
        changed_records = (
            replace(records[0], adjudication_hash=_hash("settled-substitution")),
            *records[1:],
        )
    else:
        changed_records = tuple(reversed(records))

    with pytest.raises(EvaluationValidationError, match="do not exactly match"):
        build_checkpoint_pilot_report(
            manifest,
            truth,
            store,
            target_arm_id="cascade",
            incumbent_arm_id="incumbent",
            budgets=_budgets(),
            evidence=_complete_evidence(),
            settled_candidate_records=changed_records,
            settled_candidate_acceptance=acceptance,
        )


def test_caller_asserted_shadow_summary_and_hash_are_not_accepted():
    with pytest.raises(TypeError):
        PilotRolloutEvidence(
            shadow_started_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            shadow_ended_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
            shadow_event_count=2,
            shadow_latency_p95_seconds=ScoreMetric(MeasurementStatus.COMPLETE, 0.1, 2),
            raw_shadow_artifact_hashes=(_hash("fabricated-journal"),),
        )


def test_shadow_journal_rejects_unpinned_aggregate_model_identity():
    manifest = _manifest()
    records = _shadow_records(manifest)
    changed = (
        replace(records[0], entry=replace(
            records[0].entry,
            model_id="unapproved-model",
            provider_id="unapproved-service",
            model_revision="unapproved-revision",
        )),
        records[1],
    )

    with pytest.raises(EvaluationValidationError, match="model identity does not match"):
        build_shadow_pilot_acceptance(
            changed,
            manifest=manifest,
            target_arm_id="cascade",
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("prompt_hash", _hash("wrong-shadow-prompt"), "stage versions do not match"),
        ("input_schema_version", "wrong-input-v1", "stage versions do not match"),
        ("fallback_used", True, "fallback telemetry contradicts"),
    ],
)
def test_shadow_journal_rejects_stage_contract_or_fallback_substitution(
    field_name,
    value,
    message,
):
    manifest = _manifest()
    records = _shadow_records(manifest)
    changed_stage = replace(records[0].entry.stage_runs[0], **{field_name: value})
    changed = (
        replace(records[0], entry=replace(records[0].entry, stage_runs=(changed_stage,))),
        records[1],
    )

    with pytest.raises(EvaluationValidationError, match=message):
        build_shadow_pilot_acceptance(
            changed,
            manifest=manifest,
            target_arm_id="cascade",
        )


def test_shadow_journal_rejects_missing_required_stage():
    manifest = _manifest()
    records = _shadow_records(manifest)
    changed = (
        replace(records[0], entry=replace(records[0].entry, stage_runs=())),
        records[1],
    )

    with pytest.raises(EvaluationValidationError, match="stages do not match"):
        build_shadow_pilot_acceptance(
            changed,
            manifest=manifest,
            target_arm_id="cascade",
        )


@pytest.mark.parametrize("attribute", ["latency_seconds", "tokens", "cost_usd"])
def test_shadow_journal_rejects_negative_aggregate_measurements(attribute):
    manifest = _manifest()
    records = _shadow_records(manifest)
    changed_entry = replace(
        records[0].entry,
        **{attribute: NumericMeasurement(MeasurementStatus.COMPLETE, -1)},
    )

    with pytest.raises(EvaluationValidationError, match="cannot be negative"):
        build_shadow_pilot_acceptance(
            (replace(records[0], entry=changed_entry), records[1]),
            manifest=manifest,
            target_arm_id="cascade",
        )


def test_shadow_journal_record_rejects_negative_developer_elapsed_time():
    record = _shadow_records(_manifest())[0]

    with pytest.raises(EvaluationValidationError, match="non-negative"):
        replace(record, developer_elapsed_seconds=-1)


def test_generated_but_unpinned_shadow_acceptance_cannot_pass(tmp_path):
    report = _report(tmp_path, _complete_evidence(), _budgets())
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )

    assert live_shadow.status is GateStatus.NOT_EVALUABLE
    inventory = next(
        result
        for result in live_shadow.rule_results
        if result.rule.metric == "evidence.raw_shadow_inventory_complete"
    )
    assert inventory.observed is not None
    assert inventory.observed.status is MeasurementStatus.UNAVAILABLE


def test_partial_journal_measurements_are_recomputed_and_cannot_pass(tmp_path, monkeypatch):
    manifest = _manifest()
    records = _shadow_records(manifest)
    partial_records = (
        records[0],
        replace(records[1], entry=replace(
            records[1].entry,
            cost_usd=NumericMeasurement(MeasurementStatus.UNAVAILABLE, None),
        )),
    )
    acceptance_id = build_shadow_pilot_acceptance(
        partial_records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
        shadow_records=partial_records,
        pinned_shadow_acceptance_id=acceptance_id,
    )
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )
    cost = next(
        result.observed
        for result in live_shadow.rule_results
        if result.rule.metric == "evidence.shadow_cost_per_developer_hour_usd"
    )

    assert live_shadow.status is GateStatus.NOT_EVALUABLE
    assert cost is not None
    assert cost.status is MeasurementStatus.PARTIAL
    assert cost.value == pytest.approx(0.06)


def test_dropped_shadow_entries_make_inventory_latency_and_cost_incomplete(tmp_path, monkeypatch):
    manifest = _manifest()
    records = _shadow_records(manifest)
    incomplete_records = (
        records[0],
        replace(
            records[1],
            session_summary=ShadowJournalSessionSummary(
                submitted_entry_count=3,
                queued_entry_count=2,
                dropped_entry_count=1,
                writer_failed=False,
            ),
        ),
    )
    acceptance_id = build_shadow_pilot_acceptance(
        incomplete_records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
        shadow_records=incomplete_records,
        pinned_shadow_acceptance_id=acceptance_id,
    )
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )
    results = {result.rule.metric: result.observed for result in live_shadow.rule_results}

    assert live_shadow.status is GateStatus.NOT_EVALUABLE
    assert report.shadow_binding is not None
    assert report.shadow_binding.inventory_complete is False
    assert results["evidence.raw_shadow_inventory_complete"].status is MeasurementStatus.PARTIAL
    assert results["evidence.shadow_latency_p95_seconds"].status is MeasurementStatus.PARTIAL
    assert results["evidence.shadow_cost_per_developer_hour_usd"].status is MeasurementStatus.PARTIAL


def test_single_journal_record_has_no_duration_and_is_not_evaluable(tmp_path, monkeypatch):
    manifest = _manifest()
    records = _shadow_records(manifest)[:1]
    acceptance_id = build_shadow_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
        shadow_records=records,
        pinned_shadow_acceptance_id=acceptance_id,
    )
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )
    duration = next(
        result.observed
        for result in live_shadow.rule_results
        if result.rule.metric == "evidence.shadow_elapsed_days"
    )

    assert live_shadow.status is GateStatus.NOT_EVALUABLE
    assert duration is not None
    assert duration.status is MeasurementStatus.UNAVAILABLE
    assert duration.value is None


def test_zero_elapsed_shadow_span_is_unavailable_not_a_zero_day_failure(tmp_path, monkeypatch):
    manifest = _manifest()
    records = _shadow_records(manifest)
    records = (records[0], replace(records[1], ingested_at_utc=records[0].ingested_at_utc))
    acceptance_id = build_shadow_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
        shadow_records=records,
        pinned_shadow_acceptance_id=acceptance_id,
    )
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )
    duration = next(
        result.observed
        for result in live_shadow.rule_results
        if result.rule.metric == "evidence.shadow_elapsed_days"
    )

    assert live_shadow.status is GateStatus.NOT_EVALUABLE
    assert duration is not None
    assert duration.status is MeasurementStatus.UNAVAILABLE
    assert duration.value is None


@pytest.mark.parametrize("mutation", ["truncated", "extra", "substituted", "reordered"])
def test_shadow_inventory_rejects_truncated_extra_substituted_or_reordered_records(
    tmp_path,
    monkeypatch,
    mutation,
):
    manifest = _manifest()
    truth = _truth(manifest)
    store = _store(tmp_path, manifest)
    records = _shadow_records(manifest)
    acceptance = build_shadow_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    )
    monkeypatch.setattr(
        report_module,
        "CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID",
        acceptance.acceptance_id,
    )
    if mutation == "truncated":
        changed_records = records[:-1]
    elif mutation == "extra":
        extra_entry = replace(records[-1].entry, snapshot_id=_hash("extra-shadow-snapshot"))
        changed_records = (*records, replace(
            records[-1],
            sequence=records[-1].sequence + 1,
            ingested_at_utc=records[-1].ingested_at_utc + timedelta(days=1),
            entry=extra_entry,
        ))
    elif mutation == "substituted":
        changed_records = (
            replace(
                records[0],
                entry=replace(records[0].entry, snapshot_id=_hash("substituted-shadow-snapshot")),
            ),
            records[1],
        )
    else:
        changed_records = tuple(reversed(records))

    with pytest.raises(EvaluationValidationError):
        build_checkpoint_pilot_report(
            manifest,
            truth,
            store,
            target_arm_id="cascade",
            incumbent_arm_id="incumbent",
            budgets=_budgets(),
            evidence=_complete_evidence(),
            shadow_journal_path=_persist_shadow_records(tmp_path, changed_records),
            shadow_acceptance=acceptance,
        )


def test_shadow_duration_uses_real_utc_elapsed_time_across_dst(tmp_path, monkeypatch):
    eastern = ZoneInfo("America/New_York")
    manifest = _manifest()
    records = _shadow_records(
        manifest,
        started_at=datetime(2026, 3, 2, 0, tzinfo=eastern),
    )
    records = (records[0], replace(
        records[1],
        ingested_at_utc=datetime(2026, 3, 9, 0, tzinfo=eastern).astimezone(timezone.utc),
    ))
    dst_acceptance_id = build_shadow_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id="cascade",
    ).acceptance_id
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
        shadow_records=records,
        pinned_shadow_acceptance_id=dst_acceptance_id,
    )
    live_shadow = next(
        decision for decision in report.gate_decisions if decision.gate_name == LIVE_SHADOW_GATE
    )
    elapsed = next(
        result.observed
        for result in live_shadow.rule_results
        if result.rule.metric == "evidence.shadow_elapsed_days"
    )

    assert elapsed is not None
    assert elapsed.value == 167 / 24
    assert elapsed.value < 7
    assert live_shadow.status is GateStatus.FAILED


def test_report_contains_hashes_and_decisions_but_no_sensitive_identifiers(tmp_path, monkeypatch):
    report = _report(
        tmp_path,
        _complete_evidence(),
        _budgets(),
        pin_corpus_acceptance=True,
        pin_shadow_acceptance=True,
        monkeypatch=monkeypatch,
    )
    first = report.to_dict()
    second = report.to_dict()
    serialized = json.dumps(first, sort_keys=True)

    assert first == second
    assert first["manifest_artifact_hash"].startswith("sha256:")
    assert first["model_identity_hashes"]
    assert first["corpus_binding"]["holdout_case_count"] == 18
    assert first["shadow_binding"]["event_count"] == 2
    assert first["scorecard"]["scorecard_id"] == first["scorecard_id"]
    assert first["evidence"]["replay_binding"]["scorecard_id"] == first["scorecard_id"]
    assert "incumbent-model" not in serialized
    assert "cascade-model" not in serialized
    assert "incumbent-service" not in serialized
    assert "cascade-service" not in serialized
    assert "truth_artifact_id" not in serialized
    assert "holdout-finding-0" not in serialized
    assert "temporal-finding" not in serialized
    assert '"fingerprint"' not in serialized
    assert "source" not in serialized
    assert "diff" not in serialized
