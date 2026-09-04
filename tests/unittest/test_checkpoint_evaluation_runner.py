import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from functools import lru_cache

import pytest

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
    MeasurementStatus,
    NumericMeasurement,
    deployment_identity_hash,
)
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
    failed_production_arm_result,
)
from pr_agent.algo.review_configuration import (
    materialize_review_configuration,
    review_configuration_artifact_name,
    review_configuration_canonical_bytes,
)
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewResultState, ReviewSnapshot, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails, SpecialistRunDetails


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _artifact_hash(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _review_configuration():
    return materialize_review_configuration(
        skills_context="frozen specialist instructions",
        repo_context_files={"AGENTS.md": "frozen repository instructions"},
        repo_context_max_lines=100,
        prompt_date="2026-09-03",
    )


def _write_snapshot(
    tmp_path,
    name: str = "snapshot",
    *,
    parent_snapshot_id=None,
    changed_path="src/example.py",
) -> tuple[ReviewSnapshot, object, str]:
    review_configuration = _review_configuration()
    snapshot = ReviewSnapshot(
        event=ReviewEvent.FILE_SAVE,
        repository_root=str(tmp_path / "source-repository"),
        base_revision="base-revision",
        changed_paths=(changed_path,),
        diff="diff --git a/src/example.py b/src/example.py\n+value = 1\n",
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
            kwargs["stage_plan"] = (
                EvaluationStagePlan(
                    stage="change_classification",
                    model_route=(
                        EvaluationStageModelIdentity(
                            model_id=f"model-{kind.value}",
                            provider_id="provider-v1",
                            model_revision=f"revision-{kind.value}-2026-08-30",
                            deployment_id_hash=deployment_identity_hash("private-deployment-name"),
                        ),
                    ),
                    configuration_hash=_hash(f"change-classification-config-{kind.value}"),
                    prompt_hash=_hash(f"change-classification-prompt-{kind.value}"),
                    prompt_version="change-classification-prompt-v2",
                    input_schema_version="change-classification-input-v2",
                    output_schema_version="change-classification-output-v2",
                ),
            )
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
            run_details.specialist_runs["change_classification"] = _specialist_run(arm)
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


def _specialist_run(arm: EvaluationArm, *, role="change_classification", **overrides):
    version_role = role.replace("_", "-")
    values = {
        "role": role,
        "model_used": arm.model_id,
        "deployment_id": "private-deployment-name",
        "prompt_version": f"{version_role}-prompt-v2",
        "input_schema_version": f"{version_role}-input-v2",
        "schema_version": f"{version_role}-output-v2",
        "state": "success",
        "latency_seconds": 0.2,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "num_ai_calls": 1,
        "total_cost_usd": Decimal("0.001"),
        "known_cost_call_count": 1,
        "model_costs_usd": {arm.model_id: Decimal("0.001")},
        "confidence": 0.9,
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
                details = RunDetails()
                details.specialist_runs["change_classification"] = _specialist_run(
                    arm,
                    total_cost_usd=Decimal("0.10"),
                    model_costs_usd={arm.model_id: Decimal("0.10")},
                )
                return replace(_success(arm, loaded_snapshot, details=False), run_details=details)
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
async def test_failed_model_arm_requires_explicit_identity_and_preserves_unavailable_latency(tmp_path):
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

    with pytest.raises(EvaluationValidationError, match="observed model identity"):
        await _runner(
            manifest,
            path,
            _bindings(manifest, missing_identity_factory),
            EvaluationArtifactStore(tmp_path / "missing-failure-model"),
            request,
            decision,
        ).run()

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
