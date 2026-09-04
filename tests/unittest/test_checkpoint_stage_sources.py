import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from pr_agent.algo import checkpoint_review_subprocess as review_subprocess
from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.candidate_verification import (
    candidate_verification_provider_controls_hash,
    parse_candidate_verification_config,
)
from pr_agent.algo.checkpoint_cost_authority import FrozenCostAuthority, ProviderMaximumCharge
from pr_agent.algo.checkpoint_evaluation import (
    EvaluationArmKind,
    EvaluationStageModelIdentity,
    EvaluationStagePlan,
    EvaluationValidationError,
    deployment_identity_hash,
)
from pr_agent.algo.checkpoint_stage_sources import (
    CheckpointStageSources,
    checkpoint_candidate_verification_config,
    checkpoint_candidate_verification_enabled,
    checkpoint_frontier_adjudication_config,
    checkpoint_frontier_adjudication_enabled,
    checkpoint_specialist_pipeline,
    checkpoint_specialists_enabled,
    use_checkpoint_stage_sources,
)
from pr_agent.algo.frontier_adjudication import FrontierAdjudicationConfig, FrontierModelIdentity
from pr_agent.algo.review_configuration import (
    ReviewConfigurationBundle,
    materialize_review_configuration,
    review_configuration_canonical_bytes,
)
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewSnapshot
from pr_agent.algo.review_specialists import (
    SpecialistPipelineConfig,
    SpecialistPrompt,
    SpecialistRole,
    SpecialistRoleConfig,
)
from pr_agent.config_loader import get_settings


def _hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _cost_authority(snapshot: ReviewSnapshot, configuration: ReviewConfigurationBundle) -> FrozenCostAuthority:
    return FrozenCostAuthority(
        manifest_id=_hash_json("manifest"),
        paid_request_id=_hash_json("paid-request"),
        case_id="case-one",
        arm_id="arm-full-cascade",
        snapshot_id=snapshot.snapshot_id,
        arm_configuration_hash=_hash_json("arm-configuration"),
        review_configuration_hash=configuration.configuration_hash,
        hard_cost_cap_usd=Decimal("0.01"),
        authority_name="test-gateway",
        authority_revision="test-rate-card-v1",
        authority_reference_hash=_hash_json("test-authority"),
        expires_at="2099-01-01T00:00:00Z",
        quotes=(
            ProviderMaximumCharge(
                stage="specialists",
                model_id="model-security",
                provider_id="provider",
                model_revision="model-revision-v1",
                deployment_id_hash=None,
                max_output_tokens=128,
                maximum_charge_usd=Decimal("0.01"),
            ),
        ),
    )


def _specialists() -> SpecialistPipelineConfig:
    roles = tuple(
        SpecialistRoleConfig(
            role=role,
            enabled=True,
            model=f"model-{role.value}",
            deployment=f"deployment-{role.value}",
            fallback_models=(),
            fallback_deployments=(),
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
            prompt_version=f"{role.value}-prompt-v1",
            input_schema_version=f"{role.value}-input-v1",
            schema_version=f"{role.value}-output-v1",
            system=f"private system prompt for {role.value}",
            user=f"private user prompt for {role.value}",
        )
        for role in SpecialistRole
    )
    return SpecialistPipelineConfig(
        enabled=True,
        mode="shadow",
        aggregate_timeout_seconds=8.0,
        aggregate_token_budget=12000,
        max_concurrency=3,
        cache_enabled=True,
        cache_max_entries=128,
        cancel_stale_inputs=True,
        allowed_change_labels=("docs", "tests"),
        roles=roles,
        prompts=prompts,
    )


def _verifier(*, strict_output_policy: bool):
    evidence = ({"candidate_id": "candidate-1", "content": "private analyzer evidence"},)
    return parse_candidate_verification_config(
        {},
        {"system": "private verifier system", "user": "private verifier user"},
        primary_model="model-verifier",
        static_analysis_evidence_hash=_hash_json(list(evidence)),
        static_analysis_evidence=evidence,
        provider_controls_hash=candidate_verification_provider_controls_hash(
            get_settings(),
            checkpoint_replay=True,
        ),
        strict_output_policy=strict_output_policy,
    )


def _frontier() -> FrontierAdjudicationConfig:
    return FrontierAdjudicationConfig(
        enabled=True,
        route=AIModelRoute(
            models=("model-frontier",),
            deployments=("deployment-frontier",),
            timeout_seconds=60.0,
            model_retries=1,
            provider_retries=0,
            max_output_tokens=2048,
            collect_cost=True,
        ),
        model_identities=(
            FrontierModelIdentity(
                model="model-frontier",
                provider="provider-frontier",
                revision="revision-frontier-v1",
                deployment="deployment-frontier",
            ),
        ),
        system_prompt="private frontier system",
        user_prompt="private frontier user",
    )


def _sources() -> CheckpointStageSources:
    specialists = _specialists()
    specialist_identities = {
        role.role.value: tuple(
            EvaluationStageModelIdentity(
                model_id=model,
                provider_id="provider-specialist",
                model_revision="revision-specialist-v1",
                deployment_id_hash=deployment_identity_hash(deployment),
            )
            for model, deployment in zip(role.model_route().models, role.model_route().deployments, strict=True)
        )
        for role in specialists.roles
        if role.enabled
    }
    verifier_identities = (
        EvaluationStageModelIdentity(
            model_id="model-verifier",
            provider_id="provider-verifier",
            model_revision="revision-verifier-v1",
            deployment_id_hash=deployment_identity_hash(None),
        ),
    )
    return CheckpointStageSources(
        specialist_pipeline=specialists,
        specialist_model_identities=specialist_identities,
        candidate_verification=_verifier(strict_output_policy=False),
        candidate_verification_model_identities=verifier_identities,
        full_cascade_candidate_verification=_verifier(strict_output_policy=True),
        full_cascade_candidate_verification_model_identities=verifier_identities,
        frontier_adjudication=_frontier(),
    ).for_checkpoint_replay(get_settings())


def _stage_plan(sources: CheckpointStageSources) -> tuple[EvaluationStagePlan, ...]:
    assert sources.specialist_pipeline is not None
    assert sources.full_cascade_candidate_verification is not None
    assert sources.frontier_adjudication is not None
    plans = []
    for role in sources.specialist_pipeline.roles:
        prompt = sources.specialist_pipeline.prompt(role.role)
        plans.append(
            EvaluationStagePlan(
                stage=role.role.value,
                model_route=(
                    EvaluationStageModelIdentity(
                        model_id=role.model,
                        provider_id="provider-specialist",
                        model_revision="revision-specialist-v1",
                        deployment_id_hash=deployment_identity_hash(role.deployment),
                    ),
                ),
                configuration_hash=sources.specialist_pipeline.configuration_hash,
                prompt_hash=prompt.content_hash,
                prompt_version=prompt.prompt_version,
                input_schema_version=prompt.input_schema_version,
                output_schema_version=prompt.schema_version,
            )
        )
    verifier = sources.full_cascade_candidate_verification
    plans.append(
        EvaluationStagePlan(
            stage="candidate_verification",
            model_route=(
                EvaluationStageModelIdentity(
                    model_id=verifier.route.models[0],
                    provider_id="provider-verifier",
                    model_revision="revision-verifier-v1",
                    deployment_id_hash=deployment_identity_hash(verifier.route.deployments[0]),
                ),
            ),
            configuration_hash=verifier.stage_plan_configuration_hash,
            prompt_hash=verifier.prompt_hash,
            prompt_version=verifier.prompt_version,
            input_schema_version=verifier.input_schema_version,
            output_schema_version=verifier.output_schema_version,
        )
    )
    frontier = sources.frontier_adjudication
    plans.append(
        EvaluationStagePlan(
            stage="frontier_adjudication",
            model_route=(
                EvaluationStageModelIdentity(
                    model_id=frontier.route.models[0],
                    provider_id=frontier.model_identities[0].provider,
                    model_revision=frontier.model_identities[0].revision,
                    deployment_id_hash=deployment_identity_hash(frontier.route.deployments[0]),
                ),
            ),
            configuration_hash=frontier.configuration_hash,
            prompt_hash=frontier.prompt_hash,
            prompt_version=frontier.prompt_version,
            input_schema_version=frontier.input_schema_version,
            output_schema_version=frontier.output_schema_version,
        )
    )
    return tuple(plans)


def _verified_stage_plan(sources: CheckpointStageSources) -> tuple[EvaluationStagePlan, ...]:
    full_plan = _stage_plan(sources)
    verifier = sources.candidate_verification
    assert verifier is not None
    identity = sources.candidate_verification_model_identities
    candidate_stage = EvaluationStagePlan(
        stage="candidate_verification",
        model_route=identity,
        configuration_hash=verifier.stage_plan_configuration_hash,
        prompt_hash=verifier.prompt_hash,
        prompt_version=verifier.prompt_version,
        input_schema_version=verifier.input_schema_version,
        output_schema_version=verifier.output_schema_version,
    )
    return (*full_plan[:-2], candidate_stage)


def test_stage_sources_round_trip_is_content_addressed_and_repr_private():
    sources = _sources()

    restored = CheckpointStageSources.from_dict(sources.to_dict())

    assert restored.to_dict() == sources.to_dict()
    assert restored.sources_hash == sources.sources_hash
    assert "private" not in repr(restored)
    assert "private analyzer evidence" in json.dumps(restored.to_dict())


def test_review_configuration_preserves_legacy_shape_and_binds_extended_sources():
    legacy = materialize_review_configuration("", {})
    assert "stage_sources" not in legacy.to_dict()
    assert ReviewConfigurationBundle.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()

    extended = materialize_review_configuration("", {}, stage_sources=_sources())

    assert extended.configuration_hash != legacy.configuration_hash
    assert ReviewConfigurationBundle.from_dict(extended.to_dict()).to_dict() == extended.to_dict()
    assert b"private verifier system" in review_configuration_canonical_bytes(extended)
    assert "private verifier system" not in repr(extended)

    noncanonical = legacy.to_dict()
    noncanonical["stage_sources"] = None
    with pytest.raises(ValueError, match="invalid review configuration stage_sources"):
        ReviewConfigurationBundle.from_dict(noncanonical)


@pytest.mark.parametrize(
    "divergence",
    ("field", "prompt", "model", "evidence", "budget", "identity"),
)
def test_verifier_sources_may_differ_only_in_strict_output_policy(divergence):
    sources = _sources()
    verified = sources.candidate_verification
    full = sources.full_cascade_candidate_verification
    assert verified is not None
    assert full is not None
    full_identities = sources.full_cascade_candidate_verification_model_identities

    if divergence == "field":
        full = replace(full, temperature=full.temperature + 0.1)
    elif divergence == "prompt":
        full = replace(full, system_prompt=full.system_prompt + " divergent")
    elif divergence == "model":
        full = replace(full, route=replace(full.route, models=("model-verifier-divergent",)))
        full_identities = (replace(full_identities[0], model_id="model-verifier-divergent"),)
    elif divergence == "evidence":
        evidence = ({"candidate_id": "candidate-2", "content": "divergent analyzer evidence"},)
        full = replace(
            full,
            static_analysis_evidence=evidence,
            static_analysis_evidence_hash=_hash_json(list(evidence)),
        )
    elif divergence == "budget":
        full = replace(full, budgets=replace(full.budgets, max_files=full.budgets.max_files + 1))
    elif divergence == "identity":
        full_identities = (replace(full_identities[0], provider_id="provider-divergent"),)

    with pytest.raises(ValueError, match="may differ only in strict output policy"):
        CheckpointStageSources(
            specialist_pipeline=sources.specialist_pipeline,
            specialist_model_identities=sources.specialist_model_identities,
            candidate_verification=verified,
            candidate_verification_model_identities=sources.candidate_verification_model_identities,
            full_cascade_candidate_verification=full,
            full_cascade_candidate_verification_model_identities=full_identities,
            frontier_adjudication=sources.frontier_adjudication,
        )


def test_stage_plan_requires_exact_hash_versions_route_and_dependencies():
    sources = _sources()
    plan = _stage_plan(sources)
    sources.validate_stage_plan(plan)
    sources.validate_stage_plan(plan, arm_kind=EvaluationArmKind.FULL_CASCADE)
    sources.validate_stage_plan(
        _verified_stage_plan(sources),
        arm_kind=EvaluationArmKind.VERIFIED_SPECIALISTS,
    )

    with pytest.raises(EvaluationValidationError, match="prompt hash"):
        sources.validate_stage_plan((replace(plan[0], prompt_hash=_hash_json("tampered")), *plan[1:]))
    with pytest.raises(EvaluationValidationError, match="requires verification"):
        sources.validate_stage_plan(tuple(stage for stage in plan if stage.stage != "candidate_verification"))
    forged_identity = replace(plan[0].model_route[0], provider_id="forged-provider")
    with pytest.raises(EvaluationValidationError, match="immutable model identity"):
        sources.validate_stage_plan(
            (replace(plan[0], model_route=(forged_identity,)), *plan[1:]),
            arm_kind=EvaluationArmKind.FULL_CASCADE,
        )
    swapped_verifier = _verified_stage_plan(sources)[-1]
    with pytest.raises(EvaluationValidationError, match="unavailable or ambiguous"):
        sources.validate_stage_plan(
            (*plan[:-2], swapped_verifier, plan[-1]),
            arm_kind=EvaluationArmKind.FULL_CASCADE,
        )


def test_verifier_stage_plan_hash_excludes_per_checkpoint_evidence():
    sources = _sources()
    verifier = sources.candidate_verification
    assert verifier is not None
    evidence = ({"candidate_id": "candidate-2", "content": "later checkpoint evidence"},)
    changed = replace(
        verifier,
        static_analysis_evidence=evidence,
        static_analysis_evidence_hash=_hash_json(list(evidence)),
    )

    assert changed.configuration_hash != verifier.configuration_hash
    assert changed.stage_plan_configuration_hash == verifier.stage_plan_configuration_hash


def test_execution_context_exposes_only_sources_selected_by_validated_plan():
    sources = _sources()
    plan = _stage_plan(sources)
    specialist_plan = tuple(stage for stage in plan if stage.stage in {role.value for role in SpecialistRole})
    selected = sources.for_stage_plan(specialist_plan)

    with use_checkpoint_stage_sources(selected):
        assert checkpoint_specialists_enabled() is True
        assert checkpoint_specialist_pipeline() is sources.specialist_pipeline
        assert checkpoint_candidate_verification_enabled() is False
        assert checkpoint_frontier_adjudication_enabled() is False
        with pytest.raises(ValueError, match="verification stage source is unavailable"):
            checkpoint_candidate_verification_config()
        with pytest.raises(ValueError, match="frontier stage source is unavailable"):
            checkpoint_frontier_adjudication_config({}, {})


def test_candidate_and_frontier_sources_are_injected_without_ambient_reparse():
    sources = _sources()
    plan = _stage_plan(sources)
    selected = sources.for_stage_plan(plan)

    with use_checkpoint_stage_sources(selected):
        assert checkpoint_candidate_verification_enabled() is True
        assert checkpoint_candidate_verification_config(
            primary_model="ambient",
            strict_output_policy=True,
        ) is sources.full_cascade_candidate_verification
        assert checkpoint_frontier_adjudication_enabled() is True
        assert checkpoint_frontier_adjudication_config({}, {}) is sources.frontier_adjudication


@pytest.mark.asyncio
async def test_subprocess_request_revalidates_and_injects_selected_stage_sources():
    sources = _sources()
    plan = _stage_plan(sources)
    configuration = materialize_review_configuration("", {}, stage_sources=sources)
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        base_selector="main",
        changed_paths=("example.py",),
        diff="diff --git a/example.py b/example.py\n+new = True\n",
        policy_version="policy-v1",
        created_at="2026-09-03T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
    )

    async def execute(received_snapshot, received_configuration, received_cost_authority):
        assert received_snapshot is not snapshot
        assert received_configuration.stage_sources is not sources
        assert received_cost_authority.snapshot_id == snapshot.snapshot_id
        assert checkpoint_specialist_pipeline().configuration_hash == sources.specialist_pipeline.configuration_hash
        assert checkpoint_candidate_verification_config(strict_output_policy=True).configuration_hash == (
            sources.full_cascade_candidate_verification.configuration_hash
        )
        assert checkpoint_frontier_adjudication_config({}, {}).configuration_hash == (
            sources.frontier_adjudication.configuration_hash
        )
        return review_subprocess.CheckpointReviewSubprocessOutcome(
            state=review_subprocess.CheckpointReviewSubprocessState.COMPLETED,
            snapshot_id=received_snapshot.snapshot_id,
            review={"review": {"key_issues_to_review": []}},
            latency_seconds=0.1,
        )

    request = review_subprocess._CheckpointReviewSubprocessRequest(
        snapshot=snapshot,
        review_configuration=configuration,
        evaluation_stage_plan=plan,
        allow_model_execution=True,
        cost_authority=_cost_authority(snapshot, configuration),
    )
    outcome = await review_subprocess._handle_worker_request(
        json.dumps(request.to_dict()).encode(),
        executor=execute,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED


@pytest.mark.asyncio
async def test_subprocess_refuses_stage_plan_without_private_sources_before_worker_start(monkeypatch):
    sources = _sources()
    configuration = materialize_review_configuration("", {})
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        base_selector="main",
        changed_paths=("example.py",),
        diff="diff --git a/example.py b/example.py\n+new = True\n",
        policy_version="policy-v1",
        created_at="2026-09-03T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
    )
    start_worker = AsyncMock()
    monkeypatch.setattr(review_subprocess.asyncio, "create_subprocess_exec", start_worker)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        review_configuration=configuration,
        evaluation_stage_plan=_stage_plan(sources),
        allow_model_execution=True,
    )

    assert outcome.failure_reason_code == "stage_sources_unverified"
    start_worker.assert_not_awaited()


@pytest.mark.asyncio
async def test_v3_worker_rejects_extended_bundle_when_stage_plan_field_is_omitted():
    sources = _sources()
    configuration = materialize_review_configuration("", {}, stage_sources=sources)
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        base_selector="main",
        changed_paths=("example.py",),
        diff="diff --git a/example.py b/example.py\n+new = True\n",
        policy_version="policy-v1",
        created_at="2026-09-03T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
    )
    request = review_subprocess._CheckpointReviewSubprocessRequest(
        snapshot=snapshot,
        review_configuration=configuration,
        evaluation_stage_plan=_stage_plan(sources),
        allow_model_execution=True,
    ).to_dict()
    request.pop("evaluation_stage_plan")
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        json.dumps(request).encode(),
        executor=executor,
    )

    assert outcome.failure_reason_code == "invalid_request"
    executor.assert_not_awaited()
