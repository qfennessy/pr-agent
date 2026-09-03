import hashlib
import json
import os
import subprocess
import sys

import pytest

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationStageModelIdentity,
    EvaluationStagePlan,
)
from pr_agent.algo.checkpoint_evaluation_bindings import (
    PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION,
    build_production_arm_bindings,
    production_binding_inventory,
)
from pr_agent.algo.checkpoint_evaluation_cli import run_evaluation_plan
from pr_agent.algo.checkpoint_evaluation_runner import (
    ModelTelemetryShape,
    ProductionDependencyUnavailable,
    ProductionEvaluationRunner,
)
from pr_agent.algo.review_snapshot import ReviewEvent


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _arm(kind: EvaluationArmKind) -> EvaluationArm:
    if kind is EvaluationArmKind.DETERMINISTIC:
        return EvaluationArm(
            arm_id=kind.value,
            kind=kind,
            configuration_hash=_hash(f"{kind.value}-configuration"),
            prompt_hash=_hash(f"{kind.value}-prompt"),
        )
    stage_plan = ()
    if kind not in {EvaluationArmKind.GENERAL_REVIEW}:
        stage_plan = (
            EvaluationStagePlan(
                stage="change_classification",
                model_route=(
                    EvaluationStageModelIdentity(
                        model_id=f"{kind.value}-stage-model",
                        provider_id="provider",
                        model_revision="revision",
                    ),
                ),
                configuration_hash=_hash(f"{kind.value}-stage-configuration"),
                prompt_hash=_hash(f"{kind.value}-stage-prompt"),
                prompt_version="prompt-v1",
                input_schema_version="input-v1",
                output_schema_version="output-v1",
            ),
        )
    return EvaluationArm(
        arm_id=kind.value,
        kind=kind,
        configuration_hash=_hash(f"{kind.value}-configuration"),
        prompt_hash=_hash(f"{kind.value}-prompt"),
        model_id=f"{kind.value}-model",
        provider_id="provider",
        model_revision="revision",
        stage_plan=stage_plan,
    )


def _manifest() -> EvaluationManifest:
    return EvaluationManifest(
        name="production-readiness",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        cases=(
            CheckpointCase(
                case_id="case-one",
                snapshot_id=_hash("snapshot"),
                snapshot_artifact_hash=_hash("snapshot-artifact"),
                event=ReviewEvent.FILE_SAVE,
                cohort=EvaluationCohort.HOLDOUT,
            ),
        ),
        arms=tuple(_arm(kind) for kind in EvaluationArmKind),
    )


def test_production_binding_inventory_is_exact_and_fail_closed():
    manifest = _manifest()

    inventory = production_binding_inventory(manifest)
    bindings = build_production_arm_bindings(manifest)

    assert inventory.schema_version == PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION
    assert inventory.manifest_id == manifest.manifest_id
    assert inventory.schema_hash.startswith("sha256:")
    assert inventory.inventory_id.startswith("sha256:")
    assert {item.kind for item in inventory.bindings} == set(EvaluationArmKind)
    assert all(not item.available and item.blocker_codes for item in inventory.bindings)
    assert {binding.kind for binding in bindings} == set(EvaluationArmKind)
    assert all(not binding.available for binding in bindings)
    assert all(binding.adapter is None for binding in bindings)
    assert all(binding.publish_output is False for binding in bindings)
    assert all(binding.enforces_hard_cost_cap is False for binding in bindings)

    full_cascade = next(
        item for item in inventory.bindings if item.kind is EvaluationArmKind.FULL_CASCADE
    )
    assert set(full_cascade.blocker_codes) == {
        "immutable_review_configuration_replay_unavailable",
        "finding_fingerprint_contract_unavailable",
        "verified_candidate_source_contract_unavailable",
        "frontier_stage_telemetry_contract_unavailable",
        "frontier_decision_semantics_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    }
    general_review = next(
        item for item in inventory.bindings if item.kind is EvaluationArmKind.GENERAL_REVIEW
    )
    assert set(general_review.blocker_codes) == {
        "immutable_review_configuration_replay_unavailable",
        "finding_fingerprint_contract_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    }


def test_production_bindings_preserve_frozen_manifest_contracts():
    manifest = _manifest()
    arms = {arm.kind: arm for arm in manifest.arms}
    bindings = {binding.kind: binding for binding in build_production_arm_bindings(manifest)}

    for kind, arm in arms.items():
        binding = bindings[kind]
        assert binding.configuration_hash == arm.configuration_hash
        assert binding.prompt_hash == arm.prompt_hash
        assert binding.model_identities == arm.model_identities()
        assert binding.stage_plan == arm.stage_plan

    assert bindings[EvaluationArmKind.DETERMINISTIC].telemetry_shape is ModelTelemetryShape.NONE
    assert bindings[EvaluationArmKind.GENERAL_REVIEW].telemetry_shape is ModelTelemetryShape.SINGLE_SELECTED
    for kind in (
        EvaluationArmKind.SPECIALISTS,
        EvaluationArmKind.VERIFIED_SPECIALISTS,
        EvaluationArmKind.FULL_CASCADE,
    ):
        assert bindings[kind].telemetry_shape is ModelTelemetryShape.PER_STAGE


def test_production_binding_inventory_does_not_import_model_handlers():
    script = (
        "import sys\n"
        "import pr_agent.algo.checkpoint_evaluation_bindings\n"
        "assert not any(name.startswith('pr_agent.algo.ai_handlers') for name in sys.modules)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_generated_bindings_stop_runner_preflight_before_artifact_store_access():
    manifest = _manifest()
    runner = ProductionEvaluationRunner(
        manifest,
        snapshot_paths={},
        bindings=build_production_arm_bindings(manifest),
        artifact_store=object(),
        paid_request=object(),
        paid_decision=object(),
        evaluation_enabled=True,
        allow_paid_execution=True,
        publish_output=False,
    )

    with pytest.raises(ProductionDependencyUnavailable, match="production evaluation dependencies unavailable"):
        runner.preflight()


@pytest.mark.parametrize("mode", ("--list", "--dry-run"))
def test_credential_free_plan_reports_production_binding_readiness(tmp_path, capsys, mode):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest().to_dict()), encoding="utf-8")

    payload = run_evaluation_plan(("--manifest", str(manifest_path), mode))

    assert json.loads(capsys.readouterr().out) == payload
    assert payload["network_calls"] == 0
    assert payload["model_calls"] == 0
    inventory = payload["production_binding_inventory"]
    assert inventory["schema_version"] == PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION
    assert inventory["manifest_id"] == _manifest().manifest_id
    assert len(inventory["bindings"]) == len(EvaluationArmKind)
    assert all(not item["available"] for item in inventory["bindings"])
    assert all(item["blocker_codes"] for item in inventory["bindings"])
