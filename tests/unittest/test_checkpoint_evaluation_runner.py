import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from functools import lru_cache

import pytest

from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.candidate_verification import (
    candidate_verification_provider_controls_hash,
    parse_candidate_verification_config,
)
from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
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
    FindingLifecycleState,
    FindingSeverity,
    MeasurementStatus,
    NumericMeasurement,
    ObservedFinding,
    deployment_identity_hash,
)
from pr_agent.algo.checkpoint_evaluation_adapters import adapt_checkpoint_review_outcome
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactStore,
    PaidExecutionRequest,
    PaidPlanItemBudget,
    evaluate_paid_execution,
)
from pr_agent.algo.checkpoint_evaluation_runner import (
    ModelTelemetryShape,
    ProductionArmBinding,
    ProductionArmResult,
    ProductionDependencyUnavailable,
    ProductionEvaluationRunner,
    _has_complete_lifecycle_coverage,
    failed_production_arm_result,
)
from pr_agent.algo.checkpoint_review_subprocess import (
    CheckpointReviewSubprocessOutcome,
    CheckpointReviewSubprocessState,
)
from pr_agent.algo.checkpoint_stage_sources import CheckpointStageSources
from pr_agent.algo.frontier_adjudication import FrontierAdjudicationConfig, FrontierModelIdentity
from pr_agent.algo.review_configuration import (
    materialize_review_configuration,
    review_configuration_artifact_name,
    review_configuration_canonical_bytes,
)
from pr_agent.algo.review_snapshot import (
    CoverageIssue,
    ReviewEvent,
    ReviewResultState,
    ReviewSnapshot,
    ReviewSnapshotResult,
)
from pr_agent.algo.review_specialists import (
    SpecialistPipelineConfig,
    SpecialistPrompt,
    SpecialistRole,
    SpecialistRoleConfig,
)
from pr_agent.algo.run_details import RunDetails, SpecialistRunDetails
from pr_agent.config_loader import get_settings


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _artifact_hash(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_hash(value) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _hash(payload)


@lru_cache(maxsize=1)
def _stage_sources() -> CheckpointStageSources:
    cascade_models = tuple(f"model-{kind.value}" for kind in (
        EvaluationArmKind.SPECIALISTS,
        EvaluationArmKind.VERIFIED_SPECIALISTS,
        EvaluationArmKind.FULL_CASCADE,
    ))
    roles = tuple(
        SpecialistRoleConfig(
            role=role,
            enabled=role is SpecialistRole.CHANGE_CLASSIFICATION,
            model=cascade_models[0] if role is SpecialistRole.CHANGE_CLASSIFICATION else f"model-{role.value}",
            deployment="private-deployment-name",
            fallback_models=cascade_models[1:] if role is SpecialistRole.CHANGE_CLASSIFICATION else (),
            fallback_deployments=("private-deployment-name", "private-deployment-name")
            if role is SpecialistRole.CHANGE_CLASSIFICATION
            else (),
            timeout_seconds=5.0,
            model_retries=1,
            provider_retries=0,
            input_token_budget=4000,
            output_token_budget=600,
            minimum_confidence=0.6,
        )
        for role in SpecialistRole
    )
    prompts = tuple(
        SpecialistPrompt(
            role=role,
            prompt_version=f"{role.value.replace('_', '-')}-prompt-v2",
            input_schema_version=f"{role.value.replace('_', '-')}-input-v2",
            schema_version=f"{role.value.replace('_', '-')}-output-v2",
            system=f"private system prompt for {role.value}",
            user=f"private user prompt for {role.value}",
        )
        for role in SpecialistRole
    )
    def verifier(strict_output_policy: bool):
        return parse_candidate_verification_config(
            {},
            {"system": "private verifier system", "user": "private verifier user"},
            primary_model="model-verifier",
            static_analysis_evidence_hash=_hash("[]"),
            static_analysis_evidence=(),
            provider_controls_hash=candidate_verification_provider_controls_hash(get_settings()),
            strict_output_policy=strict_output_policy,
        )
    frontier = FrontierAdjudicationConfig(
        enabled=True,
        route=AIModelRoute(
            models=("model-frontier",),
            deployments=("private-frontier-deployment",),
            timeout_seconds=60.0,
            model_retries=1,
            provider_retries=0,
            max_output_tokens=2048,
            collect_cost=True,
        ),
        model_identities=(
            FrontierModelIdentity(
                model="model-frontier",
                provider="provider-v1",
                revision="revision-frontier-2026-08-30",
                deployment="private-frontier-deployment",
            ),
        ),
        system_prompt="private frontier system",
        user_prompt="private frontier user",
    )
    pipeline = SpecialistPipelineConfig(
            enabled=True,
            mode="shadow",
            aggregate_timeout_seconds=8.0,
            aggregate_token_budget=12000,
            max_concurrency=1,
            cache_enabled=True,
            cache_max_entries=128,
            cancel_stale_inputs=True,
            allowed_change_labels=("tests",),
            roles=roles,
            prompts=prompts,
        )
    specialist_identities = {
        role.role.value: tuple(
            EvaluationStageModelIdentity(
                model_id=model,
                provider_id="provider-v1",
                model_revision=f"revision-{model.removeprefix('model-')}-2026-08-30",
                deployment_id_hash=deployment_identity_hash(deployment),
            )
            for model, deployment in zip(role.model_route().models, role.model_route().deployments, strict=True)
        )
        for role in pipeline.roles
        if role.enabled
    }
    verifier_identities = (
        EvaluationStageModelIdentity(
            model_id="model-verifier",
            provider_id="provider-v1",
            model_revision="revision-verifier-2026-08-30",
            deployment_id_hash=deployment_identity_hash(None),
        ),
    )
    return CheckpointStageSources(
        specialist_pipeline=pipeline,
        specialist_model_identities=specialist_identities,
        candidate_verification=verifier(False),
        candidate_verification_model_identities=verifier_identities,
        full_cascade_candidate_verification=verifier(True),
        full_cascade_candidate_verification_model_identities=verifier_identities,
        frontier_adjudication=frontier,
    ).for_checkpoint_replay(get_settings())


def _stage_plan(kind: EvaluationArmKind) -> tuple[EvaluationStagePlan, ...]:
    sources = _stage_sources()
    pipeline = sources.specialist_pipeline
    assert pipeline is not None
    role = next(item for item in pipeline.roles if item.role is SpecialistRole.CHANGE_CLASSIFICATION)
    prompt = pipeline.prompt(role.role)
    plans = [
        EvaluationStagePlan(
            stage=role.role.value,
            model_route=tuple(
                EvaluationStageModelIdentity(
                    model_id=model,
                    provider_id="provider-v1",
                    model_revision=f"revision-{model.removeprefix('model-')}-2026-08-30",
                    deployment_id_hash=deployment_identity_hash(deployment),
                )
                for model, deployment in zip(
                    role.model_route().models,
                    role.model_route().deployments,
                    strict=True,
                )
            ),
            configuration_hash=pipeline.configuration_hash,
            prompt_hash=prompt.content_hash,
            prompt_version=prompt.prompt_version,
            input_schema_version=prompt.input_schema_version,
            output_schema_version=prompt.schema_version,
        ),
    ]
    if kind in {EvaluationArmKind.VERIFIED_SPECIALISTS, EvaluationArmKind.FULL_CASCADE}:
        verifier = (
            sources.full_cascade_candidate_verification
            if kind is EvaluationArmKind.FULL_CASCADE
            else sources.candidate_verification
        )
        assert verifier is not None
        plans.append(
            EvaluationStagePlan(
                stage="candidate_verification",
                model_route=tuple(
                    EvaluationStageModelIdentity(
                        model_id=model,
                        provider_id="provider-v1",
                        model_revision="revision-verifier-2026-08-30",
                        deployment_id_hash=deployment_identity_hash(deployment),
                    )
                    for model, deployment in zip(verifier.route.models, verifier.route.deployments, strict=True)
                ),
                configuration_hash=verifier.stage_plan_configuration_hash,
                prompt_hash=verifier.prompt_hash,
                prompt_version=verifier.prompt_version,
                input_schema_version=verifier.input_schema_version,
                output_schema_version=verifier.output_schema_version,
            )
        )
    if kind is EvaluationArmKind.FULL_CASCADE:
        frontier = sources.frontier_adjudication
        assert frontier is not None
        plans.append(
            EvaluationStagePlan(
                stage="frontier_adjudication",
                model_route=tuple(
                    EvaluationStageModelIdentity(
                        model_id=identity.model,
                        provider_id=identity.provider,
                        model_revision=identity.revision,
                        deployment_id_hash=deployment_identity_hash(identity.deployment),
                    )
                    for identity in frontier.model_identities
                ),
                configuration_hash=frontier.configuration_hash,
                prompt_hash=frontier.prompt_hash,
                prompt_version=frontier.prompt_version,
                input_schema_version=frontier.input_schema_version,
                output_schema_version=frontier.output_schema_version,
            )
        )
    return tuple(plans)


@lru_cache(maxsize=1)
def _review_configuration():
    return materialize_review_configuration(
        skills_context="frozen specialist instructions",
        repo_context_files={"AGENTS.md": "frozen repository instructions"},
        repo_context_max_lines=100,
        prompt_date="2026-09-03",
        stage_sources=_stage_sources(),
    )


def _write_snapshot(
    tmp_path,
    name: str = "snapshot",
    *,
    parent_snapshot_id=None,
    changed_path="src/example.py",
    diff=None,
    review_configuration=None,
) -> tuple[ReviewSnapshot, object, str]:
    review_configuration = review_configuration or _review_configuration()
    diff = diff if diff is not None else "diff --git a/src/example.py b/src/example.py\n+value = 1\n"
    snapshot = ReviewSnapshot(
        event=ReviewEvent.FILE_SAVE,
        repository_root=str(tmp_path / "source-repository"),
        base_revision="base-revision",
        changed_paths=(changed_path,) if changed_path else (),
        diff=diff,
        policy_version="policy-v1",
        created_at="2026-08-30T12:00:00Z",
        review_configuration_hash=review_configuration.configuration_hash,
        parent_snapshot_id=parent_snapshot_id,
    )
    payload = (json.dumps(snapshot.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = tmp_path / f"{name}.json"
    tmp_path.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    configuration_path = tmp_path / review_configuration_artifact_name(review_configuration.configuration_hash)
    configuration_path.write_bytes(review_configuration_canonical_bytes(review_configuration))
    configuration_path.chmod(0o600)
    artifact_hash = _artifact_hash(payload)
    return snapshot, path, artifact_hash


def _configuration_artifact_hash(snapshot_path) -> str:
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    configuration_path = snapshot_path.with_name(
        review_configuration_artifact_name(snapshot_payload["review_configuration_hash"])
    )
    return _artifact_hash(configuration_path.read_bytes())


def _arm(kind: EvaluationArmKind) -> EvaluationArm:
    kwargs = {}
    if kind is not EvaluationArmKind.DETERMINISTIC:
        kwargs = {
            "model_id": f"model-{kind.value}",
            "provider_id": "provider-v1",
            "model_revision": f"revision-{kind.value}-2026-08-30",
        }
        if kind is not EvaluationArmKind.GENERAL_REVIEW:
            kwargs["stage_plan"] = _stage_plan(kind)
    return EvaluationArm(
        arm_id=f"arm-{kind.value}",
        kind=kind,
        configuration_hash=_hash(f"configuration-{kind.value}"),
        prompt_hash=_hash(f"prompt-{kind.value}"),
        **kwargs,
    )


def _manifest(snapshot: ReviewSnapshot, artifact_hash: str, *, arms=None) -> EvaluationManifest:
    case = CheckpointCase(
        case_id="case-one",
        snapshot_id=snapshot.snapshot_id,
        snapshot_artifact_hash=artifact_hash,
        event=snapshot.event,
        cohort=EvaluationCohort.HOLDOUT,
        model_visible_metadata={"language": "python"},
    )
    return EvaluationManifest(
        name="production-replay",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("manifest-configuration"),
        cases=(case,),
        arms=tuple(arms or (_arm(kind) for kind in EvaluationArmKind)),
    )


def test_zero_call_empty_snapshot_has_complete_lifecycle_coverage():
    arm = _arm(EvaluationArmKind.VERIFIED_SPECIALISTS)
    configuration = materialize_review_configuration(repo_context_files={})
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        changed_paths=(),
        diff="",
        policy_version="policy-v1",
        created_at="2026-09-04T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
    )
    outcome = adapt_checkpoint_review_outcome(
        snapshot,
        arm.kind,
        CheckpointReviewSubprocessOutcome(
            state=CheckpointReviewSubprocessState.COMPLETED,
            snapshot_id=snapshot.snapshot_id,
            review={"review": {"key_issues_to_review": []}},
            latency_seconds=0.0,
        ),
    )

    assert outcome.no_model_execution is True
    assert _has_complete_lifecycle_coverage(arm, outcome) is True


@pytest.mark.asyncio
async def test_zero_call_empty_snapshot_with_coverage_is_persistable_without_model_calls(tmp_path):
    configuration = _review_configuration()
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root=str(tmp_path / "source-repository"),
        base_revision="a" * 40,
        changed_paths=(),
        diff="",
        policy_version="policy-v1",
        created_at="2026-09-04T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
        coverage_issues=(CoverageIssue(reason="no_reviewable_diff"),),
    )
    payload = (json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    snapshot_path = tmp_path / "empty-covered.json"
    snapshot_path.write_bytes(payload)
    snapshot_path.chmod(0o600)
    configuration_path = tmp_path / review_configuration_artifact_name(configuration.configuration_hash)
    configuration_path.write_bytes(review_configuration_canonical_bytes(configuration))
    configuration_path.chmod(0o600)
    manifest = _manifest(snapshot, _artifact_hash(payload))
    arm = next(item for item in manifest.arms if item.kind is EvaluationArmKind.VERIFIED_SPECIALISTS)
    request, decision = _paid_authorization(manifest)

    def factory(bound_arm):
        async def adapter(loaded_snapshot, _context):
            if bound_arm.kind is not EvaluationArmKind.VERIFIED_SPECIALISTS:
                return _success(bound_arm, loaded_snapshot)
            return adapt_checkpoint_review_outcome(
                loaded_snapshot,
                bound_arm.kind,
                CheckpointReviewSubprocessOutcome(
                    state=CheckpointReviewSubprocessState.COMPLETED,
                    snapshot_id=loaded_snapshot.snapshot_id,
                    review={"review": {"key_issues_to_review": []}},
                    latency_seconds=0.0,
                ),
            )

        return adapter

    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"case-one": snapshot_path},
        review_configuration_artifact_hashes={
            "case-one": _artifact_hash(configuration_path.read_bytes()),
        },
        bindings=_bindings(manifest, factory),
        artifact_store=EvaluationArtifactStore(tmp_path / "covered-empty-artifacts"),
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    result = await runner.run()

    record = next(item for item in result.records if item.arm_id == arm.arm_id)
    assert record.state is EvaluationRunState.COVERAGE_UNAVAILABLE
    assert record.terminal is False
    assert record.tokens == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
    assert record.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
    assert record.stage_runs == ()


@pytest.mark.asyncio
async def test_runner_executes_parent_first_and_derives_withdrawal_only_for_complete_coverage(tmp_path):
    parent_snapshot, parent_path, parent_hash = _write_snapshot(tmp_path, "parent")
    child_snapshot, child_path, child_hash = _write_snapshot(
        tmp_path,
        "child",
        parent_snapshot_id=parent_snapshot.snapshot_id,
        changed_path=None,
        diff="",
    )
    arms = tuple(_arm(kind) for kind in EvaluationArmKind)
    manifest = EvaluationManifest(
        name="production-lineage",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("manifest-configuration"),
        cases=(
            CheckpointCase(
                case_id="z-parent",
                snapshot_id=parent_snapshot.snapshot_id,
                snapshot_artifact_hash=parent_hash,
                event=parent_snapshot.event,
                cohort=EvaluationCohort.HOLDOUT,
            ),
            CheckpointCase(
                case_id="a-child",
                snapshot_id=child_snapshot.snapshot_id,
                snapshot_artifact_hash=child_hash,
                event=child_snapshot.event,
                cohort=EvaluationCohort.HOLDOUT,
                parent_case_id="z-parent",
            ),
        ),
        arms=arms,
    )
    request, decision = _paid_authorization(manifest)
    finding = ObservedFinding(
        fingerprint=_hash("lineage-finding"),
        severity=FindingSeverity.HIGH,
        stage="general_review",
    )
    missing_finding = replace(finding, fingerprint=_hash("partial-lineage-finding"))
    calls = []

    def factory(arm):
        async def adapter(snapshot, _context):
            calls.append(snapshot.snapshot_id)
            result = _success(arm, snapshot)
            if (
                arm.kind is EvaluationArmKind.GENERAL_REVIEW
                and snapshot.snapshot_id == parent_snapshot.snapshot_id
            ):
                return replace(
                    result,
                    snapshot_result=replace(result.snapshot_result, state=ReviewResultState.FINDINGS),
                    findings=(finding,),
                )
            if (
                arm.kind is EvaluationArmKind.VERIFIED_SPECIALISTS
                and snapshot.snapshot_id == parent_snapshot.snapshot_id
            ):
                return replace(
                    result,
                    snapshot_result=replace(result.snapshot_result, state=ReviewResultState.FINDINGS),
                    findings=(finding, missing_finding),
                )
            if (
                arm.kind is EvaluationArmKind.GENERAL_REVIEW
                and snapshot.snapshot_id == child_snapshot.snapshot_id
            ):
                return adapt_checkpoint_review_outcome(
                    snapshot,
                    EvaluationArmKind.GENERAL_REVIEW,
                    CheckpointReviewSubprocessOutcome(
                        state=CheckpointReviewSubprocessState.COMPLETED,
                        snapshot_id=snapshot.snapshot_id,
                        review={"review": {"key_issues_to_review": []}},
                        latency_seconds=0.0,
                    ),
                )
            if (
                arm.kind is EvaluationArmKind.VERIFIED_SPECIALISTS
                and snapshot.snapshot_id == child_snapshot.snapshot_id
            ):
                result.run_details.specialist_runs["candidate_verification"] = _specialist_run(
                    arm,
                    role="candidate_verification",
                    state="partial",
                    failure_reason="verification_coverage_partial",
                )
                return replace(
                    result,
                    snapshot_result=replace(result.snapshot_result, state=ReviewResultState.FINDINGS),
                    findings=(finding,),
                )
            return result

        return adapter

    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"z-parent": parent_path, "a-child": child_path},
        review_configuration_artifact_hashes={
            "z-parent": _configuration_artifact_hash(parent_path),
            "a-child": _configuration_artifact_hash(child_path),
        },
        bindings=_bindings(manifest, factory),
        artifact_store=EvaluationArtifactStore(tmp_path / "lineage-artifacts"),
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    result = await runner.run()

    assert calls[:len(arms)] == [parent_snapshot.snapshot_id] * len(arms)
    child = next(
        record
        for record in result.records
        if record.case_id == "a-child" and record.arm_id == "arm-general_review"
    )
    assert child.findings == (replace(finding, lifecycle_state=FindingLifecycleState.WITHDRAWN),)
    assert child.tokens == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
    assert child.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
    partial_child = next(
        record
        for record in result.records
        if record.case_id == "a-child" and record.arm_id == "arm-verified_specialists"
    )
    assert set(partial_child.findings) == {
        finding,
        replace(missing_finding, lifecycle_state=FindingLifecycleState.CARRIED_FORWARD),
    }
    assert next(
        stage for stage in partial_child.stage_runs if stage.stage == "candidate_verification"
    ).coverage_status is MeasurementStatus.PARTIAL


@pytest.mark.asyncio
async def test_runner_does_not_reserve_or_execute_child_when_parent_failed(tmp_path):
    parent_snapshot, parent_path, parent_hash = _write_snapshot(tmp_path, "failed-parent")
    child_snapshot, child_path, child_hash = _write_snapshot(
        tmp_path,
        "blocked-child",
        parent_snapshot_id=parent_snapshot.snapshot_id,
        changed_path="src/blocked_child.py",
    )
    arms = tuple(_arm(kind) for kind in EvaluationArmKind)
    manifest = EvaluationManifest(
        name="failed-parent-lineage",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("manifest-configuration"),
        cases=(
            CheckpointCase(
                case_id="parent",
                snapshot_id=parent_snapshot.snapshot_id,
                snapshot_artifact_hash=parent_hash,
                event=parent_snapshot.event,
                cohort=EvaluationCohort.HOLDOUT,
            ),
            CheckpointCase(
                case_id="child",
                snapshot_id=child_snapshot.snapshot_id,
                snapshot_artifact_hash=child_hash,
                event=child_snapshot.event,
                cohort=EvaluationCohort.HOLDOUT,
                parent_case_id="parent",
            ),
        ),
        arms=arms,
    )
    request, decision = _paid_authorization(manifest)
    calls = []

    def factory(bound_arm):
        async def adapter(snapshot, _context):
            calls.append((snapshot.snapshot_id, bound_arm.kind))
            if (
                snapshot.snapshot_id == parent_snapshot.snapshot_id
                and bound_arm.kind is EvaluationArmKind.GENERAL_REVIEW
            ):
                return failed_production_arm_result(
                    snapshot,
                    state=EvaluationRunState.PROVIDER_FAILURE,
                    reason_code="provider_failure",
                    latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.1),
                    retry_count=0,
                    run_details=RunDetails(
                        model_used=bound_arm.model_id,
                        num_ai_calls=1,
                        total_cost_usd=Decimal("0.001"),
                        known_cost_call_count=1,
                        model_costs_usd={bound_arm.model_id: Decimal("0.001")},
                        start_time=0.0,
                        finish_time=0.1,
                    ),
                )
            return _success(bound_arm, snapshot)

        return adapter

    artifact_store = EvaluationArtifactStore(tmp_path / "failed-parent-artifacts")
    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"parent": parent_path, "child": child_path},
        review_configuration_artifact_hashes={
            "parent": _configuration_artifact_hash(parent_path),
            "child": _configuration_artifact_hash(child_path),
        },
        bindings=_bindings(manifest, factory),
        artifact_store=artifact_store,
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    result = await runner.run()
    records = result.records

    assert (child_snapshot.snapshot_id, EvaluationArmKind.GENERAL_REVIEW) not in calls
    assert any(
        record.case_id == "parent"
        and record.arm_id == "arm-general_review"
        and record.state is EvaluationRunState.PROVIDER_FAILURE
        for record in records
    )
    assert not any(
        record.case_id == "child" and record.arm_id == "arm-general_review"
        for record in records
    )


@pytest.mark.asyncio
async def test_runner_records_empty_snapshot_without_model_or_stage_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(
        tmp_path,
        "empty-snapshot",
        changed_path=None,
        diff="",
    )
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(current_snapshot, _context):
            if arm.kind is EvaluationArmKind.DETERMINISTIC:
                return _success(arm, current_snapshot, details=False)
            return adapt_checkpoint_review_outcome(
                current_snapshot,
                EvaluationArmKind.GENERAL_REVIEW,
                CheckpointReviewSubprocessOutcome(
                    state=CheckpointReviewSubprocessState.COMPLETED,
                    snapshot_id=current_snapshot.snapshot_id,
                    review={"review": {"key_issues_to_review": []}},
                    latency_seconds=0.0,
                ),
            )

        return adapter

    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"case-one": path},
        review_configuration_artifact_hashes={"case-one": _configuration_artifact_hash(path)},
        bindings=_bindings(manifest, factory),
        artifact_store=EvaluationArtifactStore(tmp_path / "empty-snapshot-artifacts"),
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    result = await runner.run()

    assert len(result.records) == len(EvaluationArmKind)
    paid_records = tuple(
        record for record in result.records if record.arm_id != "arm-deterministic"
    )
    assert all(record.state is EvaluationRunState.COMPLETED for record in paid_records)
    assert all(record.tokens == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0) for record in paid_records)
    assert all(record.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0) for record in paid_records)
    assert all(record.model_id is not None and not record.stage_runs for record in paid_records)
    assert all(EvaluationRunRecord.from_dict(record.to_dict()) == record for record in paid_records)


def _paid_authorization(manifest: EvaluationManifest):
    budgets = tuple(
        PaidPlanItemBudget(case.case_id, arm.arm_id, 0.02, 2)
        for case in manifest.cases
        for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    )
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=2.0,
        plan_item_budgets=budgets,
        credential_present_by_provider={"provider-v1": True},
    )
    decision = evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )
    decision.require_authorized()
    return request, decision


def _success(arm: EvaluationArm, snapshot: ReviewSnapshot, *, details=True) -> ProductionArmResult:
    run_details = None
    if details and arm.kind is not EvaluationArmKind.DETERMINISTIC:
        if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
            run_details = RunDetails(
                model_used=arm.model_id,
                prompt_tokens=6,
                completion_tokens=4,
                total_tokens=10,
                num_ai_calls=1,
                total_cost_usd=Decimal("0.01"),
                known_cost_call_count=1,
            )
        else:
            run_details = RunDetails()
            for stage in arm.stage_plan:
                run_details.specialist_runs[stage.stage] = _specialist_run(arm, role=stage.stage)
    return ProductionArmResult(
        snapshot_result=ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=ReviewResultState.NO_FINDINGS,
            current_snapshot_id=snapshot.snapshot_id,
            review={"review": {"key_issues_to_review": []}},
            coverage_issues=(),
            latency_seconds=0.25,
        ),
        run_details=run_details,
    )


def test_no_findings_result_retains_withdrawn_lifecycle_observations(tmp_path):
    snapshot, _path, _artifact_hash = _write_snapshot(tmp_path)
    arm = _arm(EvaluationArmKind.GENERAL_REVIEW)
    withdrawn = ObservedFinding(
        fingerprint=_hash("resolved-finding"),
        severity=FindingSeverity.HIGH,
        lifecycle_state=FindingLifecycleState.WITHDRAWN,
        stage="general_review",
    )

    result = replace(_success(arm, snapshot), findings=(withdrawn,))

    assert result.findings == (withdrawn,)


def test_no_findings_result_rejects_active_lifecycle_observations(tmp_path):
    snapshot, _path, _artifact_hash = _write_snapshot(tmp_path)
    arm = _arm(EvaluationArmKind.GENERAL_REVIEW)
    active = ObservedFinding(
        fingerprint=_hash("active-finding"),
        severity=FindingSeverity.HIGH,
        lifecycle_state=FindingLifecycleState.ACTIVE,
        stage="general_review",
    )

    with pytest.raises(EvaluationValidationError, match="only withdrawn"):
        replace(_success(arm, snapshot), findings=(active,))


def _specialist_run(arm: EvaluationArm, *, role="change_classification", **overrides):
    plan = arm.required_stage(role)
    selected_index = next(
        (index for index, identity in enumerate(plan.model_route) if identity.model_id == arm.model_id),
        0,
    )
    selected_identity = plan.model_route[selected_index]
    deployment_id = {
        "change_classification": "private-deployment-name",
        "candidate_verification": None,
        "frontier_adjudication": "private-frontier-deployment",
    }[role]
    values = {
        "role": role,
        "model_used": selected_identity.model_id,
        "deployment_id": deployment_id,
        "prompt_version": plan.prompt_version,
        "input_schema_version": plan.input_schema_version,
        "schema_version": plan.output_schema_version,
        "state": "success",
        "latency_seconds": 0.2,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "num_ai_calls": 1,
        "total_cost_usd": Decimal("0.001"),
        "known_cost_call_count": 1,
        "model_costs_usd": {selected_identity.model_id: Decimal("0.001")},
        "confidence": 0.9,
        "fallback_used": selected_index > 0,
    }
    values.update(overrides)
    return SpecialistRunDetails(**values)


def _bindings(manifest: EvaluationManifest, adapter_factory=None) -> list[ProductionArmBinding]:
    bindings = []
    for arm in manifest.arms:
        if adapter_factory is None:

            async def adapter(snapshot, _context, bound_arm=arm):
                return _success(bound_arm, snapshot)
        else:
            adapter = adapter_factory(arm)
        bindings.append(
            ProductionArmBinding(
                kind=arm.kind,
                configuration_hash=arm.configuration_hash,
                prompt_hash=arm.prompt_hash,
                model_identities=arm.model_identities(),
                stage_plan=arm.stage_plan,
                telemetry_shape=(
                    ModelTelemetryShape.NONE
                    if arm.kind is EvaluationArmKind.DETERMINISTIC
                    else (
                        ModelTelemetryShape.SINGLE_SELECTED
                        if arm.kind is EvaluationArmKind.GENERAL_REVIEW
                        else ModelTelemetryShape.PER_STAGE
                    )
                ),
                adapter=adapter,
                available=True,
                enforces_hard_cost_cap=arm.kind is not EvaluationArmKind.DETERMINISTIC,
            )
        )
    return bindings


def _runner(
    manifest,
    snapshot_path,
    bindings,
    store,
    request,
    decision,
    *,
    review_configuration_artifact_hashes=None,
):
    return ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"case-one": snapshot_path},
        review_configuration_artifact_hashes=(
            {"case-one": _configuration_artifact_hash(snapshot_path)}
            if review_configuration_artifact_hashes is None
            else review_configuration_artifact_hashes
        ),
        bindings=bindings,
        artifact_store=store,
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )


class _NoCallStore:
    def __init__(self):
        self.calls = []

    def resume_plan(self, _manifest):
        self.calls.append("resume_plan")
        raise AssertionError("artifact store must not be touched")

    def bind_paid_request(self, _manifest, _request):
        self.calls.append("bind_paid_request")
        raise AssertionError("artifact store must not be touched")

    def append_record(self, _manifest, _record):
        self.calls.append("append_record")
        raise AssertionError("artifact store must not be touched")


def test_stage_plan_without_private_sources_stops_before_store_calls(tmp_path):
    legacy_configuration = materialize_review_configuration(
        skills_context="frozen specialist instructions",
        repo_context_files={"AGENTS.md": "frozen repository instructions"},
        repo_context_max_lines=100,
        prompt_date="2026-09-03",
    )
    snapshot, path, artifact_hash = _write_snapshot(
        tmp_path,
        review_configuration=legacy_configuration,
    )
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="stage sources are unavailable"):
        _runner(manifest, path, _bindings(manifest), store, request, decision).preflight()

    assert store.calls == []


def test_mismatched_stage_sources_stop_before_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    arms = tuple(
        replace(
            arm,
            stage_plan=(replace(arm.stage_plan[0], prompt_hash=_hash("tampered-stage-prompt")),),
        )
        if arm.kind is EvaluationArmKind.SPECIALISTS
        else arm
        for arm in manifest.arms
    )
    manifest = replace(manifest, arms=arms)
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="stage sources do not match"):
        _runner(manifest, path, _bindings(manifest), store, request, decision).preflight()

    assert store.calls == []


def test_arm_stage_plan_accepts_snapshot_specific_verifier_evidence(tmp_path):
    first_sources = _stage_sources()
    first_configuration = _review_configuration()
    first_snapshot, first_path, first_artifact_hash = _write_snapshot(
        tmp_path,
        name="first-snapshot",
        review_configuration=first_configuration,
    )
    evidence = ({"candidate_id": "candidate-two", "content": "later checkpoint evidence"},)

    def with_evidence(verifier):
        assert verifier is not None
        return replace(
            verifier,
            static_analysis_evidence=evidence,
            static_analysis_evidence_hash=_json_hash(list(evidence)),
        )

    second_sources = replace(
        first_sources,
        candidate_verification=with_evidence(first_sources.candidate_verification),
        full_cascade_candidate_verification=with_evidence(
            first_sources.full_cascade_candidate_verification
        ),
    )
    second_configuration = materialize_review_configuration(
        skills_context="frozen specialist instructions",
        repo_context_files={"AGENTS.md": "frozen repository instructions"},
        repo_context_max_lines=100,
        prompt_date="2026-09-03",
        stage_sources=second_sources,
    )
    second_snapshot, second_path, second_artifact_hash = _write_snapshot(
        tmp_path,
        name="second-snapshot",
        changed_path="src/later.py",
        review_configuration=second_configuration,
    )
    first_manifest = _manifest(first_snapshot, first_artifact_hash)
    first_case = first_manifest.cases[0]
    second_case = replace(
        first_case,
        case_id="case-two",
        snapshot_id=second_snapshot.snapshot_id,
        snapshot_artifact_hash=second_artifact_hash,
    )
    manifest = replace(first_manifest, cases=(first_case, second_case))
    request, decision = _paid_authorization(manifest)

    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"case-one": first_path, "case-two": second_path},
        review_configuration_artifact_hashes={
            "case-one": _configuration_artifact_hash(first_path),
            "case-two": _configuration_artifact_hash(second_path),
        },
        bindings=_bindings(manifest),
        artifact_store=EvaluationArtifactStore(tmp_path / "evolving-evidence"),
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    preflight = runner.preflight()

    assert len(preflight.snapshots_by_case_id) == 2
    assert len(preflight.review_configurations_by_case_id) == 2
    assert first_sources.sources_hash != second_sources.sources_hash
    assert (
        first_sources.candidate_verification.configuration_hash
        != second_sources.candidate_verification.configuration_hash
    )
    assert (
        first_sources.candidate_verification.stage_plan_configuration_hash
        == second_sources.candidate_verification.stage_plan_configuration_hash
    )


@pytest.mark.parametrize("mutation", ("missing_verifier", "extra_verifier", "out_of_order"))
def test_stage_plan_kind_semantics_stop_before_store_calls(tmp_path, mutation):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    arms_by_kind = {arm.kind: arm for arm in manifest.arms}
    if mutation == "missing_verifier":
        target = arms_by_kind[EvaluationArmKind.VERIFIED_SPECIALISTS]
        changed = replace(target, stage_plan=arms_by_kind[EvaluationArmKind.SPECIALISTS].stage_plan)
    elif mutation == "extra_verifier":
        target = arms_by_kind[EvaluationArmKind.SPECIALISTS]
        changed = replace(target, stage_plan=arms_by_kind[EvaluationArmKind.VERIFIED_SPECIALISTS].stage_plan)
    else:
        target = arms_by_kind[EvaluationArmKind.FULL_CASCADE]
        changed = replace(target, stage_plan=(target.stage_plan[0], target.stage_plan[2], target.stage_plan[1]))
    manifest = replace(
        manifest,
        arms=tuple(changed if arm.kind is target.kind else arm for arm in manifest.arms),
    )
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="required cascade order"):
        _runner(manifest, path, _bindings(manifest), store, request, decision).preflight()

    assert store.calls == []


def test_swapped_full_cascade_verifier_source_stops_before_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    arms_by_kind = {arm.kind: arm for arm in manifest.arms}
    verified_plan = arms_by_kind[EvaluationArmKind.VERIFIED_SPECIALISTS].stage_plan
    full = arms_by_kind[EvaluationArmKind.FULL_CASCADE]
    changed_full = replace(full, stage_plan=(full.stage_plan[0], verified_plan[1], full.stage_plan[2]))
    manifest = replace(
        manifest,
        arms=tuple(changed_full if arm.kind is EvaluationArmKind.FULL_CASCADE else arm for arm in manifest.arms),
    )
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="stage sources do not match"):
        _runner(manifest, path, _bindings(manifest), store, request, decision).preflight()

    assert store.calls == []


def test_specialist_provider_revision_mismatch_stops_before_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    specialist = next(arm for arm in manifest.arms if arm.kind is EvaluationArmKind.SPECIALISTS)
    stage = specialist.stage_plan[0]
    forged_identity = replace(stage.model_route[0], model_revision="forged-revision")
    changed_specialist = replace(
        specialist,
        stage_plan=(replace(stage, model_route=(forged_identity, *stage.model_route[1:])),),
    )
    manifest = replace(
        manifest,
        arms=tuple(
            changed_specialist if arm.kind is EvaluationArmKind.SPECIALISTS else arm
            for arm in manifest.arms
        ),
    )
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="stage sources do not match"):
        _runner(manifest, path, _bindings(manifest), store, request, decision).preflight()

    assert store.calls == []


@pytest.mark.asyncio
async def test_unavailable_cascade_dependencies_stop_before_adapter_or_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    adapter_calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            adapter_calls.append((arm.kind, loaded_snapshot))
            return _success(arm, loaded_snapshot)

        return adapter

    bindings = _bindings(manifest, factory)
    missing = {
        EvaluationArmKind.SPECIALISTS: "specialist outputs are not normalized findings",
        EvaluationArmKind.VERIFIED_SPECIALISTS: "issue-9 verification integration is unavailable",
        EvaluationArmKind.FULL_CASCADE: "issue-11 routing and issue-9 verification are unavailable",
    }
    bindings = [
        replace(
            binding,
            adapter=None,
            available=False,
            unavailable_reason=missing[binding.kind],
        )
        if binding.kind in missing
        else binding
        for binding in bindings
    ]
    store = _NoCallStore()

    with pytest.raises(ProductionDependencyUnavailable, match="specialist outputs are not normalized"):
        await _runner(manifest, path, bindings, store, request, decision).run()

    assert adapter_calls == []
    assert store.calls == []


@pytest.mark.asyncio
async def test_every_arm_receives_the_same_loaded_snapshot_object(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    received = []

    def factory(arm):
        async def adapter(loaded_snapshot, context):
            received.append((loaded_snapshot, context))
            return _success(arm, loaded_snapshot)

        return adapter

    store = EvaluationArtifactStore(tmp_path / "artifacts")
    runner = _runner(manifest, path, _bindings(manifest, factory), store, request, decision)
    preflight = runner.preflight()
    rendered_preflight = repr(preflight)
    assert "frozen specialist instructions" not in rendered_preflight
    assert "frozen repository instructions" not in rendered_preflight
    assert "review_configurations_by_case_id=" not in rendered_preflight

    result = await runner.run()

    assert len(result.records) == len(EvaluationArmKind)
    assert len(received) == len(EvaluationArmKind)
    assert all(item[0] is received[0][0] for item in received)
    assert received[0][0] is not snapshot
    assert all(
        context.review_configuration is received[0][1].review_configuration
        for _, context in received
    )
    assert (
        received[0][1].review_configuration.configuration_hash
        == received[0][0].review_configuration_hash
    )
    assert all(context.publish_output is False for _, context in received)
    assert received[0][1].hard_cost_cap_usd is None
    assert all(context.hard_cost_cap_usd == 0.02 for _, context in received[1:])
    rendered_context = repr(received[0][1])
    assert "pinned skills" not in rendered_context
    assert "pinned repository rules" not in rendered_context
    assert "review_configuration=" not in rendered_context
    assert all(record.terminal for record in store.load_records(manifest))


@pytest.mark.asyncio
async def test_snapshot_artifact_hash_mismatch_stops_before_adapter_or_store_calls(tmp_path):
    snapshot, path, _artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, _hash("wrong-artifact-bytes"))
    request, decision = _paid_authorization(manifest)
    adapter_calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            adapter_calls.append(arm.kind)
            return _success(arm, loaded_snapshot)

        return adapter

    store = _NoCallStore()
    with pytest.raises(EvaluationValidationError, match="artifact bytes do not match"):
        await _runner(manifest, path, _bindings(manifest, factory), store, request, decision).run()
    assert adapter_calls == []
    assert store.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration_hashes",
    ({}, {"case-one": _hash("configuration"), "extra": _hash("extra")}),
)
async def test_configuration_artifact_hash_keys_must_match_cases_before_side_effects(
    tmp_path,
    configuration_hashes,
):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    adapter_calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            adapter_calls.append(arm.kind)
            return _success(arm, loaded_snapshot)

        return adapter

    store = _NoCallStore()
    with pytest.raises(EvaluationValidationError, match="hashes must match manifest cases exactly"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, factory),
            store,
            request,
            decision,
            review_configuration_artifact_hashes=configuration_hashes,
        ).run()
    assert adapter_calls == []
    assert store.calls == []


@pytest.mark.asyncio
async def test_configuration_artifact_tamper_stops_before_adapter_or_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    configuration_path = path.with_name(review_configuration_artifact_name(snapshot.review_configuration_hash))
    original_hash = _artifact_hash(configuration_path.read_bytes())
    configuration_path.write_bytes(configuration_path.read_bytes() + b" ")
    configuration_path.chmod(0o600)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="bytes do not match expected artifact hash"):
        await _runner(
            manifest,
            path,
            _bindings(manifest),
            store,
            request,
            decision,
            review_configuration_artifact_hashes={"case-one": original_hash},
        ).run()
    assert store.calls == []


@pytest.mark.asyncio
async def test_configuration_bundle_mismatch_stops_before_adapter_or_store_calls(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    mismatched_configuration = replace(_review_configuration(), prompt_date="2026-09-04")
    mismatched_payload = review_configuration_canonical_bytes(mismatched_configuration)
    configuration_path = path.with_name(review_configuration_artifact_name(snapshot.review_configuration_hash))
    configuration_path.write_bytes(mismatched_payload)
    configuration_path.chmod(0o600)
    store = _NoCallStore()

    with pytest.raises(EvaluationValidationError, match="does not match ReviewSnapshot"):
        await _runner(
            manifest,
            path,
            _bindings(manifest),
            store,
            request,
            decision,
            review_configuration_artifact_hashes={"case-one": _artifact_hash(mismatched_payload)},
        ).run()
    assert store.calls == []


def test_snapshot_parent_lineage_must_match_the_manifest_parent(tmp_path):
    parent, parent_path, parent_hash = _write_snapshot(tmp_path, "parent", changed_path="src/parent.py")
    child, child_path, child_hash = _write_snapshot(
        tmp_path,
        "child",
        parent_snapshot_id=None,
        changed_path="src/child.py",
    )
    parent_case = CheckpointCase(
        case_id="parent",
        snapshot_id=parent.snapshot_id,
        snapshot_artifact_hash=parent_hash,
        event=parent.event,
        cohort=EvaluationCohort.HOLDOUT,
    )
    child_case = CheckpointCase(
        case_id="child",
        snapshot_id=child.snapshot_id,
        snapshot_artifact_hash=child_hash,
        event=child.event,
        cohort=EvaluationCohort.HOLDOUT,
        parent_case_id=parent_case.case_id,
        lineage_elapsed_seconds=1,
    )
    manifest = EvaluationManifest(
        name="lineage-replay",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        cases=(parent_case, child_case),
        arms=tuple(_arm(kind) for kind in EvaluationArmKind),
    )
    request, decision = _paid_authorization(manifest)
    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={"parent": parent_path, "child": child_path},
        review_configuration_artifact_hashes={
            "parent": _configuration_artifact_hash(parent_path),
            "child": _configuration_artifact_hash(child_path),
        },
        bindings=_bindings(manifest),
        artifact_store=EvaluationArtifactStore(tmp_path / "lineage-artifacts"),
        paid_request=request,
        paid_decision=decision,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    with pytest.raises(EvaluationValidationError, match="does not match its manifest parent"):
        runner.preflight()


def test_model_visible_hash_metadata_must_match_the_loaded_snapshot(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    manifest = replace(
        manifest,
        cases=(replace(manifest.cases[0], model_visible_metadata={"task_intent_hash": _hash("forged")}),),
    )
    request, decision = _paid_authorization(manifest)

    with pytest.raises(EvaluationValidationError, match="task_intent_hash does not match"):
        _runner(
            manifest,
            path,
            _bindings(manifest),
            EvaluationArtifactStore(tmp_path / "forged-metadata"),
            request,
            decision,
        ).preflight()


def test_binding_rejects_wrong_model_identity_and_telemetry_shape(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    store = _NoCallStore()
    bindings = _bindings(manifest)
    general_index = next(
        index for index, binding in enumerate(bindings) if binding.kind is EvaluationArmKind.GENERAL_REVIEW
    )
    bindings[general_index] = replace(
        bindings[general_index],
        model_identities=(("unfrozen-model", "provider-v1", "unfrozen-revision"),),
    )
    with pytest.raises(EvaluationValidationError, match="model identities do not match"):
        _runner(manifest, path, bindings, store, request, decision).preflight()
    assert store.calls == []

    bindings = _bindings(manifest)
    specialist_index = next(
        index for index, binding in enumerate(bindings) if binding.kind is EvaluationArmKind.SPECIALISTS
    )
    bindings[specialist_index] = replace(
        bindings[specialist_index],
        telemetry_shape=ModelTelemetryShape.SINGLE_SELECTED,
    )
    with pytest.raises(EvaluationValidationError, match="wrong model telemetry shape"):
        _runner(manifest, path, bindings, store, request, decision).preflight()
    assert store.calls == []

    bindings = _bindings(manifest)
    specialist_binding = bindings[specialist_index]
    bindings[specialist_index] = replace(
        specialist_binding,
        stage_plan=(
            replace(
                specialist_binding.stage_plan[0],
                prompt_hash=_hash("unfrozen-stage-prompt"),
            ),
        ),
    )
    with pytest.raises(EvaluationValidationError, match="stage plan does not match"):
        _runner(manifest, path, bindings, store, request, decision).preflight()
    assert store.calls == []


@pytest.mark.asyncio
async def test_per_stage_binding_rejects_top_level_only_model_telemetry(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is EvaluationArmKind.SPECIALISTS:
                outcome = _success(arm, loaded_snapshot, details=False)
                return replace(
                    outcome,
                    run_details=RunDetails(model_used=arm.model_id, num_ai_calls=1),
                )
            return _success(arm, loaded_snapshot)

        return adapter

    with pytest.raises(EvaluationValidationError, match="stages do not match its frozen plan"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, factory),
            EvaluationArtifactStore(tmp_path / "top-level-only"),
            request,
            decision,
        ).run()


@pytest.mark.asyncio
async def test_result_preserves_source_free_specialist_role_telemetry(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is not EvaluationArmKind.SPECIALISTS:
                return _success(arm, loaded_snapshot)
            details = RunDetails()
            details.specialist_runs["change_classification"] = SpecialistRunDetails(
                role="change_classification",
                model_used=arm.model_id,
                deployment_id="private-deployment-name",
                prompt_version="change-classification-prompt-v2",
                input_schema_version="change-classification-input-v2",
                schema_version="change-classification-output-v2",
                state="success",
                latency_seconds=0.2,
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
                num_ai_calls=1,
                total_cost_usd=Decimal("0.001"),
                known_cost_call_count=1,
                model_costs_usd={arm.model_id: Decimal("0.001")},
                confidence=0.9,
                output={"reason": "source text must not enter the run artifact"},
            )
            return ProductionArmResult(
                snapshot_result=ReviewSnapshotResult(
                    snapshot_id=loaded_snapshot.snapshot_id,
                    state=ReviewResultState.NO_FINDINGS,
                    current_snapshot_id=loaded_snapshot.snapshot_id,
                    review={"review": {"key_issues_to_review": []}},
                    coverage_issues=(),
                    latency_seconds=0.25,
                ),
                run_details=details,
            )

        return adapter

    result = await _runner(
        manifest,
        path,
        _bindings(manifest, factory),
        EvaluationArtifactStore(tmp_path / "stage-telemetry"),
        request,
        decision,
    ).run()

    record = next(item for item in result.records if item.arm_id == "arm-specialists")
    assert (record.model_id, record.provider_id, record.model_revision) == (None, None, None)
    assert len(record.stage_runs) == 1
    stage = record.stage_runs[0]
    assert (stage.model_id, stage.provider_id, stage.model_revision) == (
        "model-specialists",
        "provider-v1",
        "revision-specialists-2026-08-30",
    )
    assert stage.tokens == NumericMeasurement(MeasurementStatus.PARTIAL, 10.0)
    assert stage.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.001)
    assert stage.coverage_status is MeasurementStatus.COMPLETE
    assert record.tokens == NumericMeasurement(MeasurementStatus.PARTIAL, 10.0)
    assert record.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.001)
    payload = json.dumps(record.to_dict())
    assert "source text must not enter" not in payload
    assert "private-deployment-name" not in payload
    assert record.stage_latencies_seconds["change_classification"].value == 0.2
    assert EvaluationRunRecord.from_dict(record.to_dict()) == record


@pytest.mark.asyncio
async def test_missing_specialist_usage_remains_unavailable_in_stage_and_aggregate(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is not EvaluationArmKind.SPECIALISTS:
                return _success(arm, loaded_snapshot)
            details = RunDetails()
            details.specialist_runs["change_classification"] = _specialist_run(
                arm,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                total_cost_usd=Decimal("0"),
                known_cost_call_count=0,
                model_costs_usd={},
                latency_seconds=0,
            )
            return replace(_success(arm, loaded_snapshot, details=False), run_details=details)

        return adapter

    store = EvaluationArtifactStore(tmp_path / "missing-stage-telemetry")
    with pytest.raises(EvaluationValidationError, match="complete cost telemetry"):
        await _runner(manifest, path, _bindings(manifest, factory), store, request, decision).run()

    record = next(item for item in store.load_records(manifest) if item.arm_id == "arm-specialists")
    stage = record.stage_runs[0]
    assert stage.tokens == NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    assert stage.cost_usd == NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    assert stage.latency_seconds == NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    assert record.tokens == NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    assert record.cost_usd == NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_cost_model", [False, True])
async def test_unpinned_specialist_model_or_cost_identity_fails_closed(tmp_path, bad_cost_model):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is not EvaluationArmKind.SPECIALISTS:
                return _success(arm, loaded_snapshot)
            details = RunDetails()
            overrides = (
                {"model_costs_usd": {"unfrozen-cost-model": Decimal("0.001")}}
                if bad_cost_model
                else {"model_used": "unfrozen-stage-model"}
            )
            details.specialist_runs["change_classification"] = _specialist_run(arm, **overrides)
            return replace(_success(arm, loaded_snapshot, details=False), run_details=details)

        return adapter

    with pytest.raises(EvaluationValidationError, match="unpinned model identity"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, factory),
            EvaluationArtifactStore(tmp_path / f"unpinned-{bad_cost_model}"),
            request,
            decision,
        ).run()


@pytest.mark.asyncio
async def test_uncovered_specialist_stage_cannot_be_persisted_as_clean_no_findings(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is not EvaluationArmKind.SPECIALISTS:
                return _success(arm, loaded_snapshot)
            details = RunDetails()
            details.specialist_runs["change_classification"] = _specialist_run(
                arm,
                state="provider_failure",
                failure_reason="RuntimeError",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                num_ai_calls=0,
                total_cost_usd=Decimal("0"),
                known_cost_call_count=0,
                model_costs_usd={},
                confidence=None,
            )
            return replace(_success(arm, loaded_snapshot, details=False), run_details=details)

        return adapter

    with pytest.raises(EvaluationValidationError, match="cannot claim clean coverage"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, factory),
            EvaluationArtifactStore(tmp_path / "false-clean"),
            request,
            decision,
        ).run()


def test_stage_run_rejects_malformed_failure_and_artifact_store_rechecks_identity(tmp_path):
    snapshot, _path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    arm = next(item for item in manifest.arms if item.kind is EvaluationArmKind.SPECIALISTS)
    details = _specialist_run(
        arm,
        state="provider_failure",
        failure_reason="source/path.py failed",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        num_ai_calls=0,
        total_cost_usd=Decimal("0"),
        known_cost_call_count=0,
        model_costs_usd={},
        confidence=None,
    )
    with pytest.raises(EvaluationValidationError, match="bounded machine-readable reason"):
        EvaluationStageRun.from_specialist_details(arm, "change_classification", details)

    stage = EvaluationStageRun.from_specialist_details(arm, "change_classification", _specialist_run(arm))
    forged_stage = replace(stage, model_id="unfrozen-stage-model")
    case = manifest.cases[0]
    record = EvaluationRunRecord(
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        arm_id=arm.arm_id,
        snapshot_id=case.snapshot_id,
        attempt=1,
        state=EvaluationRunState.COMPLETED,
        terminal=True,
        snapshot_result_state=ReviewResultState.NO_FINDINGS,
        latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.25),
        tokens=NumericMeasurement(MeasurementStatus.PARTIAL, 10),
        cost_usd=NumericMeasurement(MeasurementStatus.COMPLETE, 0.001),
        stage_latencies_seconds={forged_stage.stage: forged_stage.latency_seconds},
        stage_runs=(forged_stage,),
    )
    with pytest.raises(EvaluationValidationError, match="unpinned model identity"):
        EvaluationArtifactStore(tmp_path / "forged-stage").append_record(manifest, record)

    forged_cost_stage = replace(stage, cost_by_model_usd={"unfrozen-cost-model": 0.001})
    forged_cost_record = replace(
        record,
        stage_latencies_seconds={forged_cost_stage.stage: forged_cost_stage.latency_seconds},
        stage_runs=(forged_cost_stage,),
    )
    with pytest.raises(EvaluationValidationError, match="unpinned model identity"):
        EvaluationArtifactStore(tmp_path / "forged-cost-model").append_record(manifest, forged_cost_record)


def test_preflight_requires_all_five_kinds_and_the_exact_paid_decision(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    incomplete_manifest = _manifest(
        snapshot,
        artifact_hash,
        arms=tuple(_arm(kind) for kind in EvaluationArmKind if kind is not EvaluationArmKind.FULL_CASCADE),
    )
    request, decision = _paid_authorization(incomplete_manifest)
    store = _NoCallStore()
    with pytest.raises(EvaluationValidationError, match="exactly one enabled arm"):
        _runner(
            incomplete_manifest,
            path,
            _bindings(incomplete_manifest),
            store,
            request,
            decision,
        ).preflight()
    assert store.calls == []

    manifest = _manifest(snapshot, artifact_hash)
    request, _decision = _paid_authorization(manifest)
    stale_decision = evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=False,
        allow_paid_execution=True,
        publish_output=False,
    )
    with pytest.raises(EvaluationValidationError, match="decision does not match"):
        _runner(
            manifest,
            path,
            _bindings(manifest),
            store,
            request,
            stale_decision,
        ).preflight()
    assert store.calls == []

    bindings = _bindings(manifest)
    general_index = next(
        index for index, binding in enumerate(bindings)
        if binding.kind is EvaluationArmKind.GENERAL_REVIEW
    )
    bindings[general_index] = replace(
        bindings[general_index],
        enforces_hard_cost_cap=False,
    )
    with pytest.raises(ProductionDependencyUnavailable, match="hard per-call cost cap"):
        _runner(manifest, path, bindings, store, request, _decision).preflight()
    assert store.calls == []


@pytest.mark.asyncio
async def test_result_with_wrong_snapshot_or_selected_model_is_rejected(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def wrong_snapshot_factory(arm):
        async def adapter(loaded_snapshot, _context):
            outcome = _success(arm, loaded_snapshot)
            if arm.kind is EvaluationArmKind.DETERMINISTIC:
                return replace(
                    outcome,
                    snapshot_result=replace(outcome.snapshot_result, snapshot_id=_hash("other-snapshot")),
                )
            return outcome

        return adapter

    with pytest.raises(EvaluationValidationError, match="different immutable snapshot"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, wrong_snapshot_factory),
            EvaluationArtifactStore(tmp_path / "wrong-snapshot"),
            request,
            decision,
        ).run()

    def wrong_model_factory(arm):
        async def adapter(loaded_snapshot, _context):
            outcome = _success(arm, loaded_snapshot)
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                return replace(outcome, run_details=RunDetails(model_used="unfrozen-model"))
            return outcome

        return adapter

    with pytest.raises(EvaluationValidationError, match="unpinned model identity"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, wrong_model_factory),
            EvaluationArtifactStore(tmp_path / "wrong-model"),
            request,
            decision,
        ).run()


@pytest.mark.asyncio
async def test_nonterminal_failure_is_retained_and_only_that_pair_resumes(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    store = EvaluationArtifactStore(tmp_path / "resume")
    first_calls = []

    def failing_factory(arm):
        async def adapter(loaded_snapshot, _context):
            first_calls.append(arm.kind)
            if arm.kind is EvaluationArmKind.DETERMINISTIC:
                return failed_production_arm_result(
                    loaded_snapshot,
                    state=EvaluationRunState.PROVIDER_FAILURE,
                    reason_code="deterministic_tool_unavailable",
                    latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.1),
                    retry_count=1,
                )
            return _success(arm, loaded_snapshot)

        return adapter

    first = await _runner(manifest, path, _bindings(manifest, failing_factory), store, request, decision).run()
    assert len(first.records) == len(EvaluationArmKind)
    assert set(first_calls) == set(EvaluationArmKind)
    retained = store.load_records(manifest)
    failure = next(record for record in retained if record.arm_id == "arm-deterministic")
    assert failure.state is EvaluationRunState.PROVIDER_FAILURE
    assert failure.terminal is False
    assert failure.retry_count == 1
    assert failure.failure_reason_code == "deterministic_tool_unavailable"
    assert replace(failure, failure_reason_code="different_failure").record_id != failure.record_id
    assert failure.tokens.status is MeasurementStatus.UNAVAILABLE
    assert failure.cost_usd.status is MeasurementStatus.UNAVAILABLE
    assert failure.from_dict(failure.to_dict()) == failure

    resumed_calls = []

    def success_factory(arm):
        async def adapter(loaded_snapshot, _context):
            resumed_calls.append(arm.kind)
            return _success(arm, loaded_snapshot)

        return adapter

    second = await _runner(manifest, path, _bindings(manifest, success_factory), store, request, decision).run()
    assert resumed_calls == [EvaluationArmKind.DETERMINISTIC]
    assert len(second.records) == 1
    assert second.records[0].attempt == 2
    assert second.records[0].state is EvaluationRunState.COMPLETED
    deterministic_attempts = [record for record in store.load_records(manifest) if record.arm_id == "arm-deterministic"]
    assert [record.attempt for record in deterministic_attempts] == [1, 2]


@pytest.mark.asyncio
async def test_paid_failure_becomes_terminal_at_its_immutable_attempt_limit(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    store = EvaluationArtifactStore(tmp_path / "paid-attempt-limit")
    calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            calls.append(arm.kind)
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                attempt = sum(item is EvaluationArmKind.GENERAL_REVIEW for item in calls)
                return failed_production_arm_result(
                    loaded_snapshot,
                    state=EvaluationRunState.PROVIDER_FAILURE,
                    reason_code="provider_unavailable",
                    latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 0.1),
                    retry_count=attempt,
                    run_details=RunDetails(
                        model_used=arm.model_id,
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                        num_ai_calls=1,
                        total_cost_usd=Decimal("0.01"),
                        known_cost_call_count=1,
                    ),
                    model_identity=arm.model_identities()[0],
                )
            return _success(arm, loaded_snapshot)

        return adapter

    runner = _runner(manifest, path, _bindings(manifest, factory), store, request, decision)
    await runner.run()
    await runner.run()
    third = await runner.run()

    attempts = [
        record for record in store.load_records(manifest)
        if record.arm_id == "arm-general_review"
    ]
    assert [record.attempt for record in attempts] == [1, 2]
    assert [record.terminal for record in attempts] == [False, True]
    assert third.records == ()
    assert sum(item is EvaluationArmKind.GENERAL_REVIEW for item in calls) == 2


@pytest.mark.asyncio
async def test_adapter_cost_above_its_hard_cap_is_rejected_and_left_unreconciled(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    budgets = tuple(
        PaidPlanItemBudget("case-one", arm.arm_id, 0.02, 2)
        for arm in manifest.arms
        if arm.kind is not EvaluationArmKind.DETERMINISTIC
    )
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=0.16,
        plan_item_budgets=budgets,
        credential_present_by_provider={"provider-v1": True},
    )
    decision = evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )
    decision.require_authorized()
    calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            calls.append(arm.kind)
            if arm.kind is EvaluationArmKind.FULL_CASCADE:
                success = _success(arm, loaded_snapshot)
                details = success.run_details
                assert details is not None
                details.specialist_runs["change_classification"] = _specialist_run(
                    arm,
                    total_cost_usd=Decimal("0.10"),
                    model_costs_usd={arm.model_id: Decimal("0.10")},
                )
                return replace(success, run_details=details)
            return _success(arm, loaded_snapshot)

        return adapter

    store = EvaluationArtifactStore(tmp_path / "cumulative-cap")
    with pytest.raises(EvaluationValidationError, match="adapter exceeded its hard cost cap"):
        await _runner(manifest, path, _bindings(manifest, factory), store, request, decision).run()

    assert calls == [EvaluationArmKind.DETERMINISTIC, EvaluationArmKind.FULL_CASCADE]
    assert {record.arm_id for record in store.load_records(manifest)} == {"arm-deterministic"}
    reservations = store.load_paid_attempt_reservations(manifest, request)
    assert {(item.arm_id, item.attempt) for item in reservations} == {("arm-full_cascade", 1)}


@pytest.mark.asyncio
async def test_paid_adapter_exception_consumes_attempt_before_spending_and_blocks_resume(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)
    store = EvaluationArtifactStore(tmp_path / "pre-call-reservation")
    calls = []

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            calls.append(arm.kind)
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                raise RuntimeError("provider failed after charging")
            return _success(arm, loaded_snapshot)

        return adapter

    runner = _runner(manifest, path, _bindings(manifest, factory), store, request, decision)
    with pytest.raises(RuntimeError, match="provider failed after charging"):
        await runner.run()

    reservations = store.load_paid_attempt_reservations(manifest, request)
    recorded_attempts = {
        (record.arm_id, record.attempt) for record in store.load_records(manifest)
    }
    orphaned = [
        (item.arm_id, item.attempt)
        for item in reservations
        if (item.arm_id, item.attempt) not in recorded_attempts
    ]
    assert orphaned == [("arm-general_review", 1)]
    calls_before_resume = tuple(calls)

    with pytest.raises(EvaluationValidationError, match="unreconciled attempt reservation"):
        await runner.run()
    assert tuple(calls) == calls_before_resume
    assert calls.count(EvaluationArmKind.GENERAL_REVIEW) == 1


@pytest.mark.asyncio
async def test_missing_token_and_cost_telemetry_never_becomes_zero(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                outcome = _success(arm, loaded_snapshot, details=False)
                return replace(
                    outcome,
                    run_details=RunDetails(model_used=arm.model_id, num_ai_calls=1),
                )
            return _success(arm, loaded_snapshot)

        return adapter

    store = EvaluationArtifactStore(tmp_path / "telemetry")
    with pytest.raises(EvaluationValidationError, match="complete cost telemetry"):
        await _runner(manifest, path, _bindings(manifest, factory), store, request, decision).run()
    general = next(record for record in store.load_records(manifest) if record.arm_id == "arm-general_review")
    assert general.tokens.status is MeasurementStatus.UNAVAILABLE
    assert general.tokens.value is None
    assert general.cost_usd.status is MeasurementStatus.UNAVAILABLE
    assert general.cost_usd.value is None


@pytest.mark.asyncio
async def test_failed_model_arm_preserves_absent_identity_and_unavailable_latency(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def missing_identity_factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                return failed_production_arm_result(
                    loaded_snapshot,
                    state=EvaluationRunState.PROVIDER_FAILURE,
                    reason_code="provider_unavailable",
                    latency_seconds=NumericMeasurement(MeasurementStatus.UNAVAILABLE, None),
                    retry_count=2,
                    run_details=RunDetails(),
                )
            return _success(arm, loaded_snapshot)

        return adapter

    inferred_store = EvaluationArtifactStore(tmp_path / "inferred-failure-model")
    with pytest.raises(EvaluationValidationError, match="complete cost telemetry"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, missing_identity_factory),
            inferred_store,
            request,
            decision,
        ).run()
    inferred = next(
        record for record in inferred_store.load_records(manifest)
        if record.arm_id == "arm-general_review"
    )
    assert inferred.state is EvaluationRunState.PROVIDER_FAILURE
    assert inferred.failure_reason_code == "provider_unavailable"
    assert inferred.latency_seconds.status is MeasurementStatus.UNAVAILABLE
    assert inferred.latency_seconds.value is None
    assert (inferred.model_id, inferred.provider_id, inferred.model_revision) == (None, None, None)

    def explicit_identity_factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW:
                return failed_production_arm_result(
                    loaded_snapshot,
                    state=EvaluationRunState.PROVIDER_FAILURE,
                    reason_code="provider_unavailable",
                    latency_seconds=NumericMeasurement(MeasurementStatus.UNAVAILABLE, None),
                    retry_count=2,
                    run_details=RunDetails(),
                    model_identity=arm.model_identities()[0],
                )
            return _success(arm, loaded_snapshot)

        return adapter

    store = EvaluationArtifactStore(tmp_path / "explicit-failure-model")
    with pytest.raises(EvaluationValidationError, match="complete cost telemetry"):
        await _runner(manifest, path, _bindings(manifest, explicit_identity_factory), store, request, decision).run()
    failed = next(record for record in store.load_records(manifest) if record.arm_id == "arm-general_review")
    general_arm = next(arm for arm in manifest.arms if arm.kind is EvaluationArmKind.GENERAL_REVIEW)
    assert failed.state is EvaluationRunState.PROVIDER_FAILURE
    assert failed.failure_reason_code == "provider_unavailable"
    assert failed.latency_seconds.status is MeasurementStatus.UNAVAILABLE
    assert failed.latency_seconds.value is None
    assert (failed.model_id, failed.provider_id, failed.model_revision) == general_arm.model_identities()[0]


@pytest.mark.asyncio
async def test_failed_staged_arm_persists_without_fabricated_stage_runs(tmp_path):
    snapshot, path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    request, decision = _paid_authorization(manifest)

    def factory(arm):
        async def adapter(loaded_snapshot, _context):
            if arm.kind is EvaluationArmKind.VERIFIED_SPECIALISTS:
                return failed_production_arm_result(
                    loaded_snapshot,
                    state=EvaluationRunState.TIMEOUT,
                    reason_code="worker_timeout",
                    latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, 1.0),
                    retry_count=0,
                )
            return _success(arm, loaded_snapshot)

        return adapter

    result = await _runner(
        manifest,
        path,
        _bindings(manifest, factory),
        EvaluationArtifactStore(tmp_path / "failed-staged-model"),
        request,
        decision,
    ).run()

    failed = next(record for record in result.records if record.arm_id == "arm-verified_specialists")
    assert failed.state is EvaluationRunState.TIMEOUT
    assert failed.failure_reason_code == "worker_timeout"
    assert failed.stage_runs == ()
    assert (failed.model_id, failed.provider_id, failed.model_revision) == (None, None, None)


def test_failure_latency_and_reason_are_bounded_and_completed_ids_remain_compatible(tmp_path):
    snapshot, _path, artifact_hash = _write_snapshot(tmp_path)
    manifest = _manifest(snapshot, artifact_hash)
    deterministic = next(arm for arm in manifest.arms if arm.kind is EvaluationArmKind.DETERMINISTIC)
    completed = _success(deterministic, snapshot)
    completed_record = completed.snapshot_result
    legacy_compatible = EvaluationRunRecord.from_snapshot_result(
        manifest,
        manifest.cases[0],
        deterministic,
        completed_record,
        None,
        attempt=1,
        terminal=True,
    )
    legacy_payload = legacy_compatible.to_dict()
    assert "failure_reason_code" not in legacy_payload
    assert EvaluationRunRecord.from_dict(legacy_payload) == legacy_compatible
    assert "failure_reason_code" not in json.dumps(manifest.cases[0].model_visible_payload())

    with pytest.raises(EvaluationValidationError, match="finite and non-negative"):
        replace(completed, snapshot_result=replace(completed_record, latency_seconds=-1.0))
    with pytest.raises(EvaluationValidationError, match="cannot be negative"):
        failed_production_arm_result(
            snapshot,
            state=EvaluationRunState.TIMEOUT,
            reason_code="timeout",
            latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, -1.0),
            retry_count=0,
        )
    with pytest.raises(EvaluationValidationError, match="bounded machine-readable"):
        failed_production_arm_result(
            snapshot,
            state=EvaluationRunState.TIMEOUT,
            reason_code="provider said source/path.py failed",
            latency_seconds=NumericMeasurement(MeasurementStatus.UNAVAILABLE, None),
            retry_count=0,
        )
