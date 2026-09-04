"""Private, immutable production-stage inputs for checkpoint replay."""

from __future__ import annotations

import hashlib
import hmac
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence

from pr_agent.algo import CLAUDE_EXTENDED_THINKING_MODELS
from pr_agent.algo.candidate_verification import (
    CandidateVerificationConfig,
    candidate_verification_provider_controls_hash,
)
from pr_agent.algo.checkpoint_evaluation import (
    EvaluationArmKind,
    EvaluationStageModelIdentity,
    EvaluationStagePlan,
    EvaluationValidationError,
    deployment_identity_hash,
)
from pr_agent.algo.frontier_adjudication import FrontierAdjudicationConfig
from pr_agent.algo.review_specialists import (
    SpecialistPipelineConfig,
    SpecialistRole,
)

CHECKPOINT_STAGE_SOURCES_SCHEMA_VERSION = "checkpoint-stage-sources-v1"
_HASH_PREFIX = "sha256:"


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint stage sources must contain only JSON values") from exc
    return rendered.encode("utf-8")


def _sha256(value: Any) -> str:
    return _HASH_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(child) for child in value]
    return value


def _require_mapping_or_none(value: Any, label: str) -> Optional[Mapping[str, Any]]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"checkpoint stage {label} must be an object or null")
    return value


@dataclass(frozen=True)
class CheckpointStageSources:
    """Source-bearing stage contracts kept only in the private replay bundle."""

    specialist_pipeline: Optional[SpecialistPipelineConfig] = field(default=None, repr=False)
    specialist_model_identities: Mapping[str, tuple[EvaluationStageModelIdentity, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    candidate_verification: Optional[CandidateVerificationConfig] = field(default=None, repr=False)
    candidate_verification_model_identities: tuple[EvaluationStageModelIdentity, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    full_cascade_candidate_verification: Optional[CandidateVerificationConfig] = field(
        default=None,
        repr=False,
    )
    full_cascade_candidate_verification_model_identities: tuple[EvaluationStageModelIdentity, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    frontier_adjudication: Optional[FrontierAdjudicationConfig] = field(default=None, repr=False)
    schema_version: str = CHECKPOINT_STAGE_SOURCES_SCHEMA_VERSION
    sources_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_STAGE_SOURCES_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint stage sources schema version")
        if self.specialist_pipeline is not None and not isinstance(
            self.specialist_pipeline, SpecialistPipelineConfig
        ):
            raise TypeError("specialist stage source must use SpecialistPipelineConfig")
        specialist_identities = {
            stage: tuple(identities)
            for stage, identities in dict(self.specialist_model_identities).items()
        }
        object.__setattr__(self, "specialist_model_identities", MappingProxyType(specialist_identities))
        enabled_roles = ()
        if self.specialist_pipeline is not None:
            enabled_roles = tuple(role.role.value for role in self.specialist_pipeline.roles if role.enabled)
        if set(specialist_identities) != set(enabled_roles):
            raise ValueError("specialist model identities must match enabled specialist sources")
        for role in self.specialist_pipeline.roles if self.specialist_pipeline is not None else ():
            if not role.enabled:
                continue
            self._validate_identity_route(
                f"specialist {role.role.value}",
                specialist_identities[role.role.value],
                role.model_route().models,
                role.model_route().deployments,
            )
        if self.candidate_verification is not None and not isinstance(
            self.candidate_verification, CandidateVerificationConfig
        ):
            raise TypeError("verification stage source must use CandidateVerificationConfig")
        object.__setattr__(
            self,
            "candidate_verification_model_identities",
            tuple(self.candidate_verification_model_identities),
        )
        if self.candidate_verification is None:
            if self.candidate_verification_model_identities:
                raise ValueError("verification identities require a verification source")
        else:
            self._validate_identity_route(
                "verification",
                self.candidate_verification_model_identities,
                self.candidate_verification.route.models,
                self.candidate_verification.route.deployments,
            )
        if self.full_cascade_candidate_verification is not None and not isinstance(
            self.full_cascade_candidate_verification, CandidateVerificationConfig
        ):
            raise TypeError("full-cascade verification source must use CandidateVerificationConfig")
        object.__setattr__(
            self,
            "full_cascade_candidate_verification_model_identities",
            tuple(self.full_cascade_candidate_verification_model_identities),
        )
        if self.full_cascade_candidate_verification is None:
            if self.full_cascade_candidate_verification_model_identities:
                raise ValueError("full-cascade verification identities require a verification source")
        else:
            self._validate_identity_route(
                "full-cascade verification",
                self.full_cascade_candidate_verification_model_identities,
                self.full_cascade_candidate_verification.route.models,
                self.full_cascade_candidate_verification.route.deployments,
            )
        if self.candidate_verification is not None and self.full_cascade_candidate_verification is not None:
            expected_verified = replace(
                self.full_cascade_candidate_verification,
                strict_output_policy=False,
            )
            if (
                self.candidate_verification.to_dict() != expected_verified.to_dict()
                or _canonical_bytes(_thaw(self.candidate_verification.static_analysis_evidence))
                != _canonical_bytes(_thaw(expected_verified.static_analysis_evidence))
                or self.candidate_verification_model_identities
                != self.full_cascade_candidate_verification_model_identities
            ):
                raise ValueError(
                    "verified and full-cascade verification sources may differ only in strict output policy"
                )
        if self.frontier_adjudication is not None and not isinstance(
            self.frontier_adjudication, FrontierAdjudicationConfig
        ):
            raise TypeError("frontier stage source must use FrontierAdjudicationConfig")
        if self.frontier_adjudication is not None and self.full_cascade_candidate_verification is None:
            raise ValueError("frontier stage source requires full-cascade verification source")
        object.__setattr__(self, "sources_hash", _sha256(self._identity_payload()))

    @staticmethod
    def _validate_identity_route(
        label: str,
        identities: Sequence[EvaluationStageModelIdentity],
        models: Sequence[str],
        deployments: Sequence[Optional[str]],
    ) -> None:
        if len(identities) != len(models) or any(
            not isinstance(identity, EvaluationStageModelIdentity) for identity in identities
        ):
            raise ValueError(f"checkpoint {label} model identities do not match its route")
        for identity, model, deployment in zip(identities, models, deployments, strict=True):
            if identity.model_id != model or identity.deployment_id_hash != deployment_identity_hash(deployment):
                raise ValueError(f"checkpoint {label} model identities do not match its route")

    def _identity_payload(self) -> dict[str, Any]:
        verifier = self.candidate_verification
        full_cascade_verifier = self.full_cascade_candidate_verification
        return {
            "schema_version": self.schema_version,
            "specialist_pipeline": (
                None if self.specialist_pipeline is None else self.specialist_pipeline.to_dict()
            ),
            "specialist_model_identities": {
                stage: [identity.to_dict() for identity in identities]
                for stage, identities in self.specialist_model_identities.items()
            },
            "candidate_verification": (
                None
                if verifier is None
                else {
                    "configuration": verifier.to_dict(),
                    "static_analysis_evidence": _thaw(verifier.static_analysis_evidence),
                }
            ),
            "candidate_verification_model_identities": [
                identity.to_dict() for identity in self.candidate_verification_model_identities
            ],
            "full_cascade_candidate_verification": (
                None
                if full_cascade_verifier is None
                else {
                    "configuration": full_cascade_verifier.to_dict(),
                    "static_analysis_evidence": _thaw(full_cascade_verifier.static_analysis_evidence),
                }
            ),
            "full_cascade_candidate_verification_model_identities": [
                identity.to_dict()
                for identity in self.full_cascade_candidate_verification_model_identities
            ],
            "frontier_adjudication": (
                None if self.frontier_adjudication is None else self.frontier_adjudication.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "sources_hash": self.sources_hash}

    def for_checkpoint_replay(self, settings: Any) -> "CheckpointStageSources":
        """Bind verifier contracts to the worker's authorized sanitized controls."""

        override = settings.config.get("claude_extended_thinking_models_override", []) or []
        claude_extended_thinking_models = (
            tuple(model.strip() for model in override)
            if isinstance(override, list)
            and override
            and all(isinstance(model, str) and model.strip() for model in override)
            else tuple(CLAUDE_EXTENDED_THINKING_MODELS)
        )
        provider_controls_hash = candidate_verification_provider_controls_hash(
            settings,
            claude_extended_thinking_models=claude_extended_thinking_models,
            checkpoint_replay=True,
        )
        return replace(
            self,
            candidate_verification=(
                None
                if self.candidate_verification is None
                else replace(self.candidate_verification, provider_controls_hash=provider_controls_hash)
            ),
            full_cascade_candidate_verification=(
                None
                if self.full_cascade_candidate_verification is None
                else replace(
                    self.full_cascade_candidate_verification,
                    provider_controls_hash=provider_controls_hash,
                )
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointStageSources":
        expected = {
            "schema_version",
            "specialist_pipeline",
            "specialist_model_identities",
            "candidate_verification",
            "candidate_verification_model_identities",
            "full_cascade_candidate_verification",
            "full_cascade_candidate_verification_model_identities",
            "frontier_adjudication",
            "sources_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("invalid checkpoint stage sources fields")
        specialist_value = _require_mapping_or_none(value["specialist_pipeline"], "specialist source")
        specialist_identity_value = value["specialist_model_identities"]
        if not isinstance(specialist_identity_value, Mapping) or any(
            not isinstance(stage, str) or not isinstance(identities, list)
            for stage, identities in specialist_identity_value.items()
        ):
            raise ValueError("invalid checkpoint specialist model identities")
        verifier_value = _require_mapping_or_none(value["candidate_verification"], "verification source")
        full_cascade_verifier_value = _require_mapping_or_none(
            value["full_cascade_candidate_verification"],
            "full-cascade verification source",
        )
        verifier_identity_value = value["candidate_verification_model_identities"]
        full_verifier_identity_value = value["full_cascade_candidate_verification_model_identities"]
        if not isinstance(verifier_identity_value, list) or not isinstance(full_verifier_identity_value, list):
            raise ValueError("invalid checkpoint verification model identities")
        frontier_value = _require_mapping_or_none(value["frontier_adjudication"], "frontier source")
        def load_verifier(
            raw: Optional[Mapping[str, Any]],
            label: str,
        ) -> Optional[CandidateVerificationConfig]:
            if raw is None:
                return None
            if set(raw) != {"configuration", "static_analysis_evidence"}:
                raise ValueError(f"invalid checkpoint {label} source fields")
            configuration = raw["configuration"]
            evidence = raw["static_analysis_evidence"]
            if not isinstance(configuration, Mapping) or not isinstance(evidence, list):
                raise ValueError(f"invalid checkpoint {label} source values")
            return CandidateVerificationConfig.from_dict(
                configuration,
                static_analysis_evidence=evidence,
            )
        verifier = load_verifier(verifier_value, "verification")
        full_cascade_verifier = load_verifier(full_cascade_verifier_value, "full-cascade verification")
        sources = cls(
            specialist_pipeline=(
                None if specialist_value is None else SpecialistPipelineConfig.from_dict(specialist_value)
            ),
            specialist_model_identities={
                stage: tuple(EvaluationStageModelIdentity.from_dict(identity) for identity in identities)
                for stage, identities in specialist_identity_value.items()
            },
            candidate_verification=verifier,
            candidate_verification_model_identities=tuple(
                EvaluationStageModelIdentity.from_dict(identity) for identity in verifier_identity_value
            ),
            full_cascade_candidate_verification=full_cascade_verifier,
            full_cascade_candidate_verification_model_identities=tuple(
                EvaluationStageModelIdentity.from_dict(identity) for identity in full_verifier_identity_value
            ),
            frontier_adjudication=(
                None if frontier_value is None else FrontierAdjudicationConfig.from_dict(frontier_value)
            ),
            schema_version=value.get("schema_version"),
        )
        if not isinstance(value.get("sources_hash"), str) or not hmac.compare_digest(
            value["sources_hash"], sources.sources_hash
        ):
            raise ValueError("checkpoint stage sources hash mismatch")
        return sources

    def _candidate_source_for_plan(
        self,
        plan: EvaluationStagePlan,
        arm_kind: Optional[EvaluationArmKind],
    ) -> CandidateVerificationConfig:
        if arm_kind is EvaluationArmKind.VERIFIED_SPECIALISTS:
            candidates = (self.candidate_verification,)
        elif arm_kind is EvaluationArmKind.FULL_CASCADE:
            candidates = (self.full_cascade_candidate_verification,)
        else:
            candidates = (
                self.candidate_verification,
                self.full_cascade_candidate_verification,
            )
        matches = [
            candidate
            for candidate in candidates
            if candidate is not None and candidate.configuration_hash == plan.configuration_hash
        ]
        if len(matches) != 1:
            raise EvaluationValidationError("checkpoint verification stage source is unavailable or ambiguous")
        return matches[0]

    def validate_stage_plan(
        self,
        stage_plan: Sequence[EvaluationStagePlan],
        *,
        arm_kind: Optional[EvaluationArmKind] = None,
    ) -> None:
        """Match every executable stage to its exact private configuration source."""

        if not isinstance(stage_plan, (list, tuple)) or any(
            not isinstance(stage, EvaluationStagePlan) for stage in stage_plan
        ):
            raise EvaluationValidationError("checkpoint stage plan must use EvaluationStagePlan")
        plan_by_name = {stage.stage: stage for stage in stage_plan}
        effective_arm_kind = arm_kind or infer_checkpoint_arm_kind(stage_plan)
        if len(plan_by_name) != len(stage_plan):
            raise EvaluationValidationError("checkpoint stage plan contains duplicate stages")
        specialist_names = {role.value for role in SpecialistRole}
        unknown = sorted(set(plan_by_name) - specialist_names - {"candidate_verification", "frontier_adjudication"})
        if unknown:
            raise EvaluationValidationError(f"checkpoint stage plan has no private source: {unknown}")

        planned_specialists = set(plan_by_name) & specialist_names
        if planned_specialists:
            pipeline = self.specialist_pipeline
            if pipeline is None or not pipeline.enabled:
                raise EvaluationValidationError("checkpoint specialist stage source is unavailable")
            enabled_roles = {role.role.value for role in pipeline.roles if role.enabled}
            if planned_specialists != enabled_roles:
                raise EvaluationValidationError("checkpoint specialist stage plan does not match enabled roles")
            for role in pipeline.roles:
                if role.role.value not in planned_specialists:
                    continue
                prompt = pipeline.prompt(role.role)
                self._validate_plan_contract(
                    plan_by_name[role.role.value],
                    configuration_hash=pipeline.configuration_hash,
                    prompt_hash=prompt.content_hash,
                    prompt_version=prompt.prompt_version,
                    input_schema_version=prompt.input_schema_version,
                    output_schema_version=prompt.schema_version,
                    models=role.model_route().models,
                    deployments=role.model_route().deployments,
                    model_identities=self.specialist_model_identities[role.role.value],
                )
        if "candidate_verification" in plan_by_name:
            verifier = self._candidate_source_for_plan(
                plan_by_name["candidate_verification"],
                effective_arm_kind,
            )
            verifier_identities = (
                self.full_cascade_candidate_verification_model_identities
                if verifier.strict_output_policy
                else self.candidate_verification_model_identities
            )
            expected_strict_policy = effective_arm_kind is EvaluationArmKind.FULL_CASCADE
            if effective_arm_kind in {
                EvaluationArmKind.VERIFIED_SPECIALISTS,
                EvaluationArmKind.FULL_CASCADE,
            } and (
                verifier.strict_output_policy is not expected_strict_policy
            ):
                raise EvaluationValidationError(
                    "checkpoint verification strict output policy does not match its arm"
                )
            self._validate_plan_contract(
                plan_by_name["candidate_verification"],
                configuration_hash=verifier.configuration_hash,
                prompt_hash=verifier.prompt_hash,
                prompt_version=verifier.prompt_version,
                input_schema_version=verifier.input_schema_version,
                output_schema_version=verifier.output_schema_version,
                models=verifier.route.models,
                deployments=verifier.route.deployments,
                model_identities=verifier_identities,
            )
        if "frontier_adjudication" in plan_by_name:
            if "candidate_verification" not in plan_by_name:
                raise EvaluationValidationError("checkpoint frontier stage requires verification in the stage plan")
            frontier = self.frontier_adjudication
            if frontier is None or not frontier.enabled:
                raise EvaluationValidationError("checkpoint frontier stage source is unavailable")
            self._validate_plan_contract(
                plan_by_name["frontier_adjudication"],
                configuration_hash=frontier.configuration_hash,
                prompt_hash=frontier.prompt_hash,
                prompt_version=frontier.prompt_version,
                input_schema_version=frontier.input_schema_version,
                output_schema_version=frontier.output_schema_version,
                models=frontier.route.models,
                deployments=frontier.route.deployments,
                providers=tuple(identity.provider for identity in frontier.model_identities),
                revisions=tuple(identity.revision for identity in frontier.model_identities),
                model_identities=tuple(
                    EvaluationStageModelIdentity(
                        model_id=identity.model,
                        provider_id=identity.provider,
                        model_revision=identity.revision,
                        deployment_id_hash=deployment_identity_hash(identity.deployment),
                    )
                    for identity in frontier.model_identities
                ),
            )
        expected_stages = self.required_stage_names(effective_arm_kind)
        if expected_stages is not None and tuple(stage.stage for stage in stage_plan) != expected_stages:
            raise EvaluationValidationError("checkpoint stage plan does not match its required cascade order")

    def required_stage_names(self, arm_kind: EvaluationArmKind) -> Optional[tuple[str, ...]]:
        """Return the exact ordered stages authorized for one cascade arm."""

        enabled_specialists = tuple(
            role.value
            for role in SpecialistRole
            if self.specialist_pipeline is not None
            and any(item.role is role and item.enabled for item in self.specialist_pipeline.roles)
        )
        return {
            EvaluationArmKind.SPECIALISTS: enabled_specialists,
            EvaluationArmKind.VERIFIED_SPECIALISTS: (*enabled_specialists, "candidate_verification"),
            EvaluationArmKind.FULL_CASCADE: (
                *enabled_specialists,
                "candidate_verification",
                "frontier_adjudication",
            ),
        }.get(arm_kind)

    @staticmethod
    def _validate_plan_contract(
        plan: EvaluationStagePlan,
        *,
        configuration_hash: str,
        prompt_hash: str,
        prompt_version: str,
        input_schema_version: str,
        output_schema_version: str,
        models: Sequence[str],
        deployments: Sequence[Optional[str]],
        providers: Optional[Sequence[str]] = None,
        revisions: Optional[Sequence[str]] = None,
        model_identities: Optional[Sequence[EvaluationStageModelIdentity]] = None,
    ) -> None:
        scalar_pairs = (
            ("configuration hash", plan.configuration_hash, configuration_hash),
            ("prompt hash", plan.prompt_hash, prompt_hash),
            ("prompt version", plan.prompt_version, prompt_version),
            ("input schema", plan.input_schema_version, input_schema_version),
            ("output schema", plan.output_schema_version, output_schema_version),
        )
        for label, actual, expected in scalar_pairs:
            if actual != expected:
                raise EvaluationValidationError(f"checkpoint stage {plan.stage} {label} does not match its source")
        if len(plan.model_route) != len(models) or len(deployments) != len(models):
            raise EvaluationValidationError(f"checkpoint stage {plan.stage} model route does not match its source")
        for index, (identity, model, deployment) in enumerate(
            zip(plan.model_route, models, deployments, strict=True)
        ):
            if identity.model_id != model or identity.deployment_id_hash != deployment_identity_hash(deployment):
                raise EvaluationValidationError(f"checkpoint stage {plan.stage} model route does not match its source")
            if providers is not None and identity.provider_id != providers[index]:
                raise EvaluationValidationError(f"checkpoint stage {plan.stage} provider does not match its source")
            if revisions is not None and identity.model_revision != revisions[index]:
                raise EvaluationValidationError(f"checkpoint stage {plan.stage} revision does not match its source")
        if model_identities is not None and tuple(plan.model_route) != tuple(model_identities):
            raise EvaluationValidationError(
                f"checkpoint stage {plan.stage} immutable model identity does not match its source"
            )

    def for_stage_plan(self, stage_plan: Sequence[EvaluationStagePlan]) -> "CheckpointStageSources":
        """Return only the sources authorized by one arm's validated plan."""

        self.validate_stage_plan(stage_plan, arm_kind=infer_checkpoint_arm_kind(stage_plan))
        names = {stage.stage for stage in stage_plan}
        verifier = (
            self._candidate_source_for_plan(
                next(stage for stage in stage_plan if stage.stage == "candidate_verification"),
                None,
            )
            if "candidate_verification" in names
            else None
        )
        return CheckpointStageSources(
            specialist_pipeline=(
                self.specialist_pipeline if names & {role.value for role in SpecialistRole} else None
            ),
            specialist_model_identities=(
                {
                    stage: identities
                    for stage, identities in self.specialist_model_identities.items()
                    if stage in names
                }
            ),
            candidate_verification=(verifier if verifier is not None and not verifier.strict_output_policy else None),
            candidate_verification_model_identities=(
                self.candidate_verification_model_identities
                if verifier is not None and not verifier.strict_output_policy
                else ()
            ),
            full_cascade_candidate_verification=(
                verifier if verifier is not None and verifier.strict_output_policy else None
            ),
            full_cascade_candidate_verification_model_identities=(
                self.full_cascade_candidate_verification_model_identities
                if verifier is not None and verifier.strict_output_policy
                else ()
            ),
            frontier_adjudication=(
                self.frontier_adjudication if "frontier_adjudication" in names else None
            ),
        )


_ACTIVE_STAGE_SOURCES: ContextVar[Optional[CheckpointStageSources]] = ContextVar(
    "checkpoint_stage_sources",
    default=None,
)


def get_checkpoint_stage_sources() -> Optional[CheckpointStageSources]:
    return _ACTIVE_STAGE_SOURCES.get()


def infer_checkpoint_arm_kind(
    stage_plan: Sequence[EvaluationStagePlan],
) -> Optional[EvaluationArmKind]:
    names = {stage.stage for stage in stage_plan}
    if "frontier_adjudication" in names:
        return EvaluationArmKind.FULL_CASCADE
    if "candidate_verification" in names:
        return EvaluationArmKind.VERIFIED_SPECIALISTS
    if names:
        return EvaluationArmKind.SPECIALISTS
    return None


@contextmanager
def use_checkpoint_stage_sources(sources: CheckpointStageSources) -> Iterator[None]:
    if not isinstance(sources, CheckpointStageSources):
        raise TypeError("checkpoint execution requires CheckpointStageSources")
    token = _ACTIVE_STAGE_SOURCES.set(sources)
    try:
        yield
    finally:
        _ACTIVE_STAGE_SOURCES.reset(token)


def checkpoint_specialists_enabled() -> bool:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        return sources.specialist_pipeline is not None and sources.specialist_pipeline.enabled
    from pr_agent.algo.review_specialists import specialists_enabled

    return specialists_enabled()


def checkpoint_candidate_verification_enabled() -> bool:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        return (
            sources.candidate_verification is not None
            or sources.full_cascade_candidate_verification is not None
        )
    from pr_agent.config_loader import get_settings

    value = get_settings().pr_reviewer.get("enable_candidate_verification", False)
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def checkpoint_frontier_adjudication_enabled() -> bool:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        return sources.frontier_adjudication is not None and sources.frontier_adjudication.enabled
    from pr_agent.config_loader import get_settings

    value = get_settings().pr_reviewer.get("enable_frontier_adjudication", False)
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def checkpoint_specialist_pipeline() -> SpecialistPipelineConfig:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        if sources.specialist_pipeline is None:
            raise ValueError("checkpoint specialist stage source is unavailable")
        return sources.specialist_pipeline
    from pr_agent.algo.review_specialists import load_specialist_pipeline_config

    return load_specialist_pipeline_config()


def checkpoint_candidate_verification_config(**kwargs: Any) -> CandidateVerificationConfig:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        candidates = tuple(
            source
            for source in (
                sources.candidate_verification,
                sources.full_cascade_candidate_verification,
            )
            if source is not None
        )
        if len(candidates) != 1:
            raise ValueError("checkpoint verification stage source is unavailable")
        strict_output_policy = kwargs.get("strict_output_policy", False)
        if not isinstance(strict_output_policy, bool) or candidates[0].strict_output_policy is not strict_output_policy:
            raise ValueError("checkpoint verification strict output policy does not match its source")
        return candidates[0]
    from pr_agent.algo.candidate_verification import load_production_candidate_verification_config

    return load_production_candidate_verification_config(**kwargs)


def checkpoint_frontier_adjudication_config(
    section: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    azure: bool = False,
) -> FrontierAdjudicationConfig:
    sources = get_checkpoint_stage_sources()
    if sources is not None:
        if sources.frontier_adjudication is None:
            raise ValueError("checkpoint frontier stage source is unavailable")
        return sources.frontier_adjudication
    from pr_agent.algo.frontier_adjudication import load_frontier_adjudication_config

    return load_frontier_adjudication_config(section, prompt, azure=azure)
