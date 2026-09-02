"""Provider-neutral frontier adjudication for already verified review findings."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.run_details import (
    adjudication_runs_to_dict,
    get_run_details,
    record_adjudication_result,
)
from pr_agent.algo.token_handler import TokenHandler

FRONTIER_INPUT_SCHEMA_VERSION = "frontier-adjudication-input-v1"
FRONTIER_OUTPUT_SCHEMA_VERSION = "frontier-adjudication-output-v1"
FRONTIER_PROMPT_VERSION = "frontier-adjudication-prompt-v1"
FRONTIER_POLICY_VERSION = "frontier-adjudication-policy-v1"


class FrontierDecision(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    UNAVAILABLE = "unavailable"


class FrontierState(str, Enum):
    NOT_REQUIRED = "not_required"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    MALFORMED_OUTPUT = "malformed_output"
    TIMEOUT = "timeout"
    PROVIDER_FAILURE = "provider_failure"


class NormalizedSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    NormalizedSeverity.LOW: 0,
    NormalizedSeverity.MEDIUM: 1,
    NormalizedSeverity.HIGH: 2,
    NormalizedSeverity.CRITICAL: 3,
}


class FrontierContractError(ValueError):
    """Raised when an adjudication contract is incomplete or malformed."""


@dataclass(frozen=True, slots=True)
class FrontierModelIdentity:
    """One exact provider/model/revision identity in an availability route."""

    model: str
    provider: str
    revision: str
    deployment: Optional[str] = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.model, self.provider, self.revision)):
            raise FrontierContractError("frontier model, provider, and revision must be non-blank")


@dataclass(frozen=True, slots=True)
class FrontierEvidence:
    """One bounded excerpt in a candidate-verification citation group."""

    evidence_id: str
    source: str
    path: str
    side: str
    start_line: int
    end_line: int
    content: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.evidence_id, self.source, self.path, self.content)
        ):
            raise FrontierContractError("frontier evidence is incomplete")
        if self.side not in {"old", "new"}:
            raise FrontierContractError("frontier evidence side must be old or new")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise FrontierContractError("frontier evidence range is invalid")
        object.__setattr__(
            self,
            "content_sha256",
            f"sha256:{hashlib.sha256(self.content.encode('utf-8')).hexdigest()}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "path": self.path,
            "side": self.side,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrontierCandidate:
    """A verified candidate carrying the production identity issued by verification."""

    stable_finding_id: str
    root_cause_id: str
    path: str
    side: str
    start_line: int
    end_line: int
    title: str
    explanation: str
    trigger: str
    impact: str
    verified_severity: NormalizedSeverity

    def __post_init__(self) -> None:
        if not self.stable_finding_id.startswith("sha256:") or not self.root_cause_id.startswith("sha256:"):
            raise FrontierContractError("frontier candidate requires trusted production identities")
        if not all((self.path, self.title, self.explanation, self.trigger, self.impact)):
            raise FrontierContractError("frontier candidate is incomplete")
        if self.side not in {"old", "new"} or self.start_line < 1 or self.end_line < self.start_line:
            raise FrontierContractError("frontier candidate location is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_finding_id": self.stable_finding_id,
            "root_cause_id": self.root_cause_id,
            "location": {
                "path": self.path,
                "side": self.side,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
            "title": self.title,
            "explanation": self.explanation,
            "trigger": self.trigger,
            "impact": self.impact,
            "verified_severity": self.verified_severity.value,
        }


@dataclass(frozen=True, slots=True)
class FrontierSignals:
    """Upward-only deterministic and verification signals controlling escalation."""

    sensitive: bool = False
    severe: bool = False
    disputed: bool = False
    insufficient_evidence: bool = False
    deterministic_forced: bool = False
    deterministic_severity_floor: NormalizedSeverity = NormalizedSeverity.LOW
    reasons: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    @property
    def requires_escalation(self) -> bool:
        return any((
            self.sensitive,
            self.severe,
            self.disputed,
            self.insufficient_evidence,
            self.deterministic_forced,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitive": self.sensitive,
            "severe": self.severe,
            "disputed": self.disputed,
            "insufficient_evidence": self.insufficient_evidence,
            "deterministic_forced": self.deterministic_forced,
            "deterministic_severity_floor": self.deterministic_severity_floor.value,
            "reasons": list(self.reasons),
            "unresolved_questions": list(self.unresolved_questions),
        }


@dataclass(frozen=True, slots=True)
class FrontierAdjudicationRequest:
    """One immutable adjudication input bound to exact code and policy identities."""

    candidate: FrontierCandidate
    evidence: tuple[FrontierEvidence, ...]
    signals: FrontierSignals
    snapshot_id: str
    configuration_hash: str
    prompt_hash: str
    policy_version: str
    risk_policy_version: str
    schema_version: str = FRONTIER_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FRONTIER_INPUT_SCHEMA_VERSION:
            raise FrontierContractError("unsupported frontier input schema")
        if not self.evidence:
            raise FrontierContractError("frontier adjudication requires evidence")
        if not all((
            self.snapshot_id,
            self.configuration_hash,
            self.prompt_hash,
            self.policy_version,
            self.risk_policy_version,
        )):
            raise FrontierContractError("frontier adjudication identity is incomplete")
        groups: dict[str, list[FrontierEvidence]] = {}
        group_identities: dict[str, tuple[str, str, str]] = {}
        for item in self.evidence:
            identity = (item.source, item.path, item.side)
            existing_identity = group_identities.setdefault(item.evidence_id, identity)
            if existing_identity != identity:
                raise FrontierContractError(
                    "frontier evidence citation groups must share source, path, and side"
                )
            groups.setdefault(item.evidence_id, []).append(item)
        for excerpts in groups.values():
            ordered = sorted(
                excerpts,
                key=lambda item: (item.start_line, item.end_line, item.content_sha256),
            )
            if any(
                current.start_line <= previous.end_line
                for previous, current in zip(ordered, ordered[1:], strict=False)
            ):
                raise FrontierContractError(
                    "frontier evidence citation group excerpts must not overlap"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "signals": self.signals.to_dict(),
            "identity": {
                "snapshot_id": self.snapshot_id,
                "configuration_hash": self.configuration_hash,
                "prompt_hash": self.prompt_hash,
                "policy_version": self.policy_version,
                "risk_policy_version": self.risk_policy_version,
            },
        }


@dataclass(frozen=True, slots=True)
class FrontierAdjudicationConfig:
    """Explicit production route and versioned prompt for frontier adjudication."""

    enabled: bool
    route: AIModelRoute
    model_identities: tuple[FrontierModelIdentity, ...]
    system_prompt: str
    user_prompt: str
    prompt_version: str = FRONTIER_PROMPT_VERSION
    input_schema_version: str = FRONTIER_INPUT_SCHEMA_VERSION
    output_schema_version: str = FRONTIER_OUTPUT_SCHEMA_VERSION
    policy_version: str = FRONTIER_POLICY_VERSION
    minimum_confidence: float = 0.0
    stage_timeout_seconds: float = 120.0
    max_calls: int = 3

    def __post_init__(self) -> None:
        if len(self.model_identities) != len(self.route.models):
            raise FrontierContractError("frontier model identities must match the availability route")
        if self.route.collect_cost is not True:
            raise FrontierContractError("frontier route must collect attributed cost telemetry")
        if self.enabled and (
            not isinstance(self.route.model_retries, int)
            or isinstance(self.route.model_retries, bool)
            or self.route.model_retries < 1
        ):
            raise FrontierContractError(
                "enabled frontier route requires an explicit positive model-attempt limit"
            )
        if self.enabled and (
            not isinstance(self.route.provider_retries, int)
            or isinstance(self.route.provider_retries, bool)
            or self.route.provider_retries != 0
        ):
            raise FrontierContractError(
                "enabled frontier route requires provider retries to be disabled for exact telemetry"
            )
        route_keys = [(identity.model, identity.deployment) for identity in self.model_identities]
        if len(route_keys) != len(set(route_keys)):
            raise FrontierContractError("frontier model and deployment route identities must be unique")
        for index, identity in enumerate(self.model_identities):
            if identity.model != self.route.models[index] or identity.deployment != self.route.deployments[index]:
                raise FrontierContractError("frontier model identity does not match its route entry")
        if isinstance(self.minimum_confidence, bool) or not 0 <= self.minimum_confidence <= 1:
            raise FrontierContractError("frontier minimum confidence must be between 0 and 1")
        if (
            not isinstance(self.stage_timeout_seconds, (int, float))
            or isinstance(self.stage_timeout_seconds, bool)
            or not isfinite(self.stage_timeout_seconds)
            or self.stage_timeout_seconds <= 0
        ):
            raise FrontierContractError("frontier stage timeout must be finite and positive")
        if (
            not isinstance(self.route.timeout_seconds, (int, float))
            or isinstance(self.route.timeout_seconds, bool)
            or not isfinite(self.route.timeout_seconds)
            or self.route.timeout_seconds <= 0
        ):
            raise FrontierContractError("frontier model timeout must be finite and positive")
        if isinstance(self.max_calls, bool) or self.max_calls < 0:
            raise FrontierContractError("frontier max calls cannot be negative")
        if not self.system_prompt.strip() or not self.user_prompt.strip() or not self.prompt_version.strip():
            raise FrontierContractError("frontier prompt identity is incomplete")
        if self.input_schema_version != FRONTIER_INPUT_SCHEMA_VERSION:
            raise FrontierContractError("unsupported frontier input schema")
        if self.output_schema_version != FRONTIER_OUTPUT_SCHEMA_VERSION:
            raise FrontierContractError("unsupported frontier output schema")

    @property
    def configuration_hash(self) -> str:
        payload = {
            "enabled": self.enabled,
            "models": [
                {
                    "model": item.model,
                    "provider": item.provider,
                    "revision": item.revision,
                    "deployment": item.deployment,
                }
                for item in self.model_identities
            ],
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "policy_version": self.policy_version,
            "minimum_confidence": self.minimum_confidence,
            "stage_timeout_seconds": self.stage_timeout_seconds,
            "max_calls": self.max_calls,
            "route_timeout_seconds": self.route.timeout_seconds,
            "model_retries": self.route.model_retries,
            "provider_retries": self.route.provider_retries,
            "max_output_tokens": self.route.max_output_tokens,
            "collect_cost": self.route.collect_cost,
        }
        return _hash_json(payload)

    @property
    def prompt_hash(self) -> str:
        return _hash_json({"system": self.system_prompt, "user": self.user_prompt})


@dataclass(frozen=True, slots=True)
class NormalizedFrontierFinding:
    stable_finding_id: str
    root_cause_id: str
    severity: NormalizedSeverity
    path: str
    side: str
    start_line: int
    end_line: int
    evidence_citations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "normalized-review-finding-v1",
            "stable_finding_id": self.stable_finding_id,
            "root_cause_id": self.root_cause_id,
            "severity": self.severity.value,
            "location": {
                "path": self.path,
                "side": self.side,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
            "evidence_citations": list(self.evidence_citations),
        }


@dataclass(frozen=True, slots=True)
class FrontierAdjudicationResult:
    schema_version: str
    stable_finding_id: str
    decision: FrontierDecision
    state: FrontierState
    confidence: Optional[float]
    evidence_citations: tuple[str, ...]
    normalized_finding: Optional[NormalizedFrontierFinding] = None
    failure_reason: Optional[str] = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Return source-free lifecycle telemetry suitable for benchmark consumers."""

        return {
            "schema_version": self.schema_version,
            "decision": self.decision.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "evidence_citation_count": len(self.evidence_citations),
            "stable_finding_id": self.stable_finding_id,
            "normalized_severity": (
                self.normalized_finding.severity.value if self.normalized_finding else None
            ),
            "failure_reason": self.failure_reason,
            "telemetry": dict(self.telemetry),
            "publication_safe": False,
        }


def normalize_severity(value: Any, default: NormalizedSeverity = NormalizedSeverity.MEDIUM) -> NormalizedSeverity:
    """Normalize only the production severity vocabulary; unknown values keep the caller's floor."""

    try:
        return NormalizedSeverity(str(value).strip().lower())
    except (TypeError, ValueError):
        return default


def severity_at_least(value: NormalizedSeverity, floor: NormalizedSeverity) -> NormalizedSeverity:
    """Apply a deterministic severity floor that model output cannot lower."""

    return value if _SEVERITY_RANK[value] >= _SEVERITY_RANK[floor] else floor


def load_frontier_adjudication_config(
    section: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    azure: bool = False,
) -> FrontierAdjudicationConfig:
    """Build one immutable route from explicit frontier-only configuration."""

    def enabled_flag(key: str) -> bool:
        value = section.get(key, False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    enabled = enabled_flag("enable_frontier_adjudication")
    if enabled and not enabled_flag("enable_candidate_verification"):
        raise FrontierContractError(
            "enabled frontier adjudication requires candidate verification to be enabled"
        )

    def strings(key: str, *, required: bool = False) -> tuple[str, ...]:
        raw = section.get(key, [])
        if isinstance(raw, str):
            values = (raw.strip(),) if raw.strip() else ()
        elif isinstance(raw, (list, tuple)):
            if any(not isinstance(item, str) for item in raw):
                raise FrontierContractError(f"{key} must be a string list")
            values = tuple(item.strip() for item in raw)
        else:
            raise FrontierContractError(f"{key} must be a string list")
        if any(not item for item in values) or (required and not values):
            raise FrontierContractError(f"{key} requires non-blank values")
        return values

    def deployments(key: str, count: int) -> tuple[Optional[str], ...]:
        raw = section.get(key, [])
        if isinstance(raw, str):
            values = (raw.strip() or None,) if raw.strip() else ()
        elif isinstance(raw, (list, tuple)):
            if any(not isinstance(item, str) for item in raw):
                raise FrontierContractError(f"{key} must be a string list")
            values = tuple(item.strip() or None for item in raw)
        else:
            raise FrontierContractError(f"{key} must be a string list")
        if not values:
            values = (None,) * count
        if len(values) != count:
            raise FrontierContractError(f"{key} must match its model list")
        if azure and any(value is None for value in values):
            raise FrontierContractError("every Azure frontier model requires a deployment")
        return values

    def integer(key: str, default: int) -> int:
        raw = section.get(key, default)
        if isinstance(raw, bool):
            raise FrontierContractError(f"{key} must be a non-boolean integer")
        if isinstance(raw, Integral):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip(), 10)
            except ValueError as exc:
                raise FrontierContractError(
                    f"{key} must be a non-boolean integer"
                ) from exc
        raise FrontierContractError(f"{key} must be a non-boolean integer")

    def number(key: str, default: float) -> float:
        raw = section.get(key, default)
        if isinstance(raw, bool):
            raise FrontierContractError(f"{key} must be a non-boolean number")
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise FrontierContractError(
                f"{key} must be a non-boolean number"
            ) from exc

    primary_model = strings("frontier_adjudication_model", required=True)
    if len(primary_model) != 1:
        raise FrontierContractError("frontier_adjudication_model requires one model")
    fallback_models = strings("frontier_adjudication_fallback_models")
    primary_provider = strings("frontier_adjudication_provider", required=True)
    primary_revision = strings("frontier_adjudication_revision", required=True)
    if len(primary_provider) != 1 or len(primary_revision) != 1:
        raise FrontierContractError("frontier provider and revision require one value")
    fallback_providers = strings("frontier_adjudication_fallback_providers")
    fallback_revisions = strings("frontier_adjudication_fallback_revisions")
    if len(fallback_providers) != len(fallback_models) or len(fallback_revisions) != len(fallback_models):
        raise FrontierContractError("frontier fallback providers and revisions must match fallback models")

    primary_deployment = deployments("frontier_adjudication_deployment", 1)
    fallback_deployments = deployments("frontier_adjudication_fallback_deployments", len(fallback_models))
    models = (*primary_model, *fallback_models)
    route_deployments = (*primary_deployment, *fallback_deployments)
    identities = tuple(
        FrontierModelIdentity(model=model, provider=provider, revision=revision, deployment=deployment)
        for model, provider, revision, deployment in zip(
            models,
            (*primary_provider, *fallback_providers),
            (*primary_revision, *fallback_revisions),
            route_deployments,
            strict=True,
        )
    )
    model_timeout = number("frontier_adjudication_model_timeout_seconds", 60)
    stage_timeout = number("frontier_adjudication_timeout_seconds", 120)
    minimum_confidence = number("frontier_adjudication_minimum_confidence", 0.0)
    model_retries = integer("frontier_adjudication_model_retries", 1)
    provider_retries = integer("frontier_adjudication_provider_retries", 0)
    max_output_tokens = integer("frontier_adjudication_max_output_tokens", 2048)
    max_calls = integer("frontier_adjudication_max_calls", 3)
    route = AIModelRoute(
        models=models,
        deployments=route_deployments,
        timeout_seconds=model_timeout,
        model_retries=model_retries,
        provider_retries=provider_retries,
        max_output_tokens=max_output_tokens,
        collect_cost=True,
    )
    return FrontierAdjudicationConfig(
        enabled=enabled,
        route=route,
        model_identities=identities,
        system_prompt=str(prompt.get("system") or ""),
        user_prompt=str(prompt.get("user") or ""),
        prompt_version=str(prompt.get("prompt_version") or ""),
        input_schema_version=str(prompt.get("input_schema_version") or ""),
        output_schema_version=str(prompt.get("schema_version") or ""),
        stage_timeout_seconds=stage_timeout,
        minimum_confidence=minimum_confidence,
        max_calls=max_calls,
    )


def build_frontier_evidence(candidate_id: str, evidence: Sequence[Mapping[str, Any]]) -> tuple[FrontierEvidence, ...]:
    """Reuse only prompt-visible candidate-verification evidence for one candidate."""

    selected = []
    for index, item in enumerate(evidence):
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        if candidate_id not in candidate_ids:
            continue
        content = str(item.get("content") or "")
        path = str(item.get("path") or "")
        try:
            start_line = int(item.get("start_line"))
            end_line = int(item.get("end_line"))
        except (TypeError, ValueError):
            continue
        evidence_id = str(item.get("evidence_id") or f"candidate-evidence-{index + 1}")
        try:
            selected.append(FrontierEvidence(
                evidence_id=evidence_id,
                source=str(item.get("source") or ""),
                path=path,
                side=str(item.get("side") or "new"),
                start_line=start_line,
                end_line=end_line,
                content=content,
            ))
        except FrontierContractError:
            continue
    return tuple(selected)


def _hash_json(value: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


class _FrontierIdentityRefreshError(RuntimeError):
    """The identity callback failed independently of the stage deadline."""


async def _refresh_identity(current_identity: Callable[[], Any]) -> Optional[str]:
    try:
        if inspect.iscoroutinefunction(current_identity):
            value = await current_identity()
        else:
            value = await asyncio.to_thread(current_identity)
        if inspect.isawaitable(value):
            value = await value
        return str(value) if value else None
    except Exception as exc:
        raise _FrontierIdentityRefreshError from exc


async def _refresh_identity_before(
    current_identity: Callable[[], Any],
    deadline_monotonic: float,
) -> Optional[str]:
    remaining_seconds = deadline_monotonic - time.monotonic()
    if remaining_seconds <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(
        _refresh_identity(current_identity),
        timeout=remaining_seconds,
    )


def _unavailable(
    request: FrontierAdjudicationRequest,
    state: FrontierState,
    failure_reason: str,
) -> FrontierAdjudicationResult:
    telemetry = adjudication_runs_to_dict().get(request.candidate.stable_finding_id, {})
    return FrontierAdjudicationResult(
        schema_version=FRONTIER_OUTPUT_SCHEMA_VERSION,
        stable_finding_id=request.candidate.stable_finding_id,
        decision=FrontierDecision.UNAVAILABLE,
        state=state,
        confidence=None,
        evidence_citations=(),
        failure_reason=failure_reason,
        telemetry=telemetry,
    )


def _validate_output(
    raw: str,
    request: FrontierAdjudicationRequest,
    config: FrontierAdjudicationConfig,
) -> dict[str, Any]:
    try:
        output = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FrontierContractError("frontier response is not valid JSON") from exc
    if not isinstance(output, Mapping):
        raise FrontierContractError("frontier response must be an object")
    expected_keys = {
        "schema_version", "decision", "normalized_severity", "confidence",
        "evidence_citations", "unresolved_questions",
    }
    if set(output) != expected_keys or output.get("schema_version") != config.output_schema_version:
        raise FrontierContractError("frontier response does not match the output schema")
    try:
        decision = FrontierDecision(str(output["decision"]))
    except ValueError as exc:
        raise FrontierContractError("frontier decision is unsupported") from exc
    confidence = output["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise FrontierContractError("frontier confidence must be between 0 and 1")
    citations = output["evidence_citations"]
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise FrontierContractError("frontier evidence citations must be a string list")
    allowed_citations = {item.evidence_id for item in request.evidence}
    if len(citations) != len(set(citations)) or any(item not in allowed_citations for item in citations):
        raise FrontierContractError("frontier response cites evidence outside the immutable input")
    if decision is not FrontierDecision.UNAVAILABLE and not citations:
        raise FrontierContractError("confirm and reject decisions require evidence citations")
    questions = output["unresolved_questions"]
    if not isinstance(questions, list) or any(not isinstance(item, str) or not item.strip() for item in questions):
        raise FrontierContractError("frontier unresolved questions must be non-blank strings")
    severity = None
    if decision is FrontierDecision.CONFIRM:
        try:
            severity = NormalizedSeverity(str(output["normalized_severity"]).strip().lower())
        except ValueError as exc:
            raise FrontierContractError("confirmed frontier response requires normalized severity") from exc
    elif output["normalized_severity"] is not None:
        raise FrontierContractError("non-confirmed frontier response cannot assign severity")
    original_decision = decision
    if decision is not FrontierDecision.UNAVAILABLE and confidence < config.minimum_confidence:
        decision = FrontierDecision.UNAVAILABLE
    return {
        "decision": decision,
        "severity": severity,
        "confidence": float(confidence),
        "citations": tuple(citations),
        "questions": tuple(questions),
        "failure_reason": (
            "below_confidence_threshold"
            if decision is FrontierDecision.UNAVAILABLE and original_decision is not FrontierDecision.UNAVAILABLE
            else "model_unavailable" if decision is FrontierDecision.UNAVAILABLE else None
        ),
    }


def _model_identity(
    config: FrontierAdjudicationConfig,
    model: Optional[str],
    deployment: Optional[str],
) -> Optional[FrontierModelIdentity]:
    return next(
        (
            item for item in config.model_identities
            if item.model == model and item.deployment == deployment
        ),
        None,
    )


def _telemetry_complete(telemetry: Mapping[str, Any]) -> bool:
    usage = telemetry.get("usage") if isinstance(telemetry, Mapping) else None
    cost = telemetry.get("cost") if isinstance(telemetry, Mapping) else None
    retries = telemetry.get("retries") if isinstance(telemetry, Mapping) else None
    model_retries = retries.get("model") if isinstance(retries, Mapping) else None
    provider_retries = retries.get("provider") if isinstance(retries, Mapping) else None
    route_attempts = telemetry.get("route_attempts")

    route_attempts_complete = (
        isinstance(route_attempts, int)
        and not isinstance(route_attempts, bool)
        and route_attempts >= 1
    )
    model_attempts = (
        model_retries.get("attempts") if isinstance(model_retries, Mapping) else None
    )
    model_retry_attempts = (
        model_retries.get("retry_attempts")
        if isinstance(model_retries, Mapping)
        else None
    )
    configured_model_attempts = (
        model_retries.get("configured_attempts_per_model")
        if isinstance(model_retries, Mapping)
        else None
    )
    model_retry_telemetry_complete = bool(
        isinstance(model_retries, Mapping)
        and model_retries.get("status") == "complete"
        and isinstance(configured_model_attempts, int)
        and not isinstance(configured_model_attempts, bool)
        and configured_model_attempts >= 1
        and isinstance(model_attempts, int)
        and not isinstance(model_attempts, bool)
        and isinstance(model_retry_attempts, int)
        and not isinstance(model_retry_attempts, bool)
        and route_attempts_complete
        and model_attempts >= route_attempts
        and model_attempts <= route_attempts * configured_model_attempts
        and model_retry_attempts == model_attempts - route_attempts
    )
    configured_provider_retries = (
        provider_retries.get("configured_retries_per_model_attempt")
        if isinstance(provider_retries, Mapping)
        else None
    )
    provider_attempts = (
        provider_retries.get("attempts")
        if isinstance(provider_retries, Mapping)
        else None
    )
    provider_retry_attempts = (
        provider_retries.get("retry_attempts")
        if isinstance(provider_retries, Mapping)
        else None
    )
    successful_ai_calls = usage.get("ai_calls") if isinstance(usage, Mapping) else None
    provider_retry_telemetry_complete = bool(
        isinstance(provider_retries, Mapping)
        and provider_retries.get("status") == "complete"
        and isinstance(configured_provider_retries, int)
        and not isinstance(configured_provider_retries, bool)
        and configured_provider_retries == 0
        and isinstance(provider_attempts, int)
        and not isinstance(provider_attempts, bool)
        and isinstance(provider_retry_attempts, int)
        and not isinstance(provider_retry_attempts, bool)
        and isinstance(model_attempts, int)
        and provider_attempts >= model_attempts
        and provider_retry_attempts == provider_attempts - model_attempts
        and provider_retries.get("unavailable_reason") is None
    )
    attempt_accounting_complete = bool(
        isinstance(successful_ai_calls, int)
        and not isinstance(successful_ai_calls, bool)
        and isinstance(provider_attempts, int)
        and successful_ai_calls == provider_attempts
    )
    return bool(
        telemetry.get("model")
        and telemetry.get("provider")
        and telemetry.get("model_revision")
        and route_attempts_complete
        and model_retry_telemetry_complete
        and provider_retry_telemetry_complete
        and attempt_accounting_complete
        and isinstance(usage, Mapping)
        and usage.get("status") == "complete"
        and isinstance(cost, Mapping)
        and cost.get("status") == "complete"
    )


def _completion_identity_verified(
    configured: Optional[FrontierModelIdentity],
    telemetry: Mapping[str, Any],
) -> bool:
    """Require provider-issued completion identity to match the pinned route entry."""

    return bool(
        configured is not None
        and telemetry.get("provider")
        and str(telemetry["provider"]).casefold() == configured.provider.casefold()
        and telemetry.get("model_revision") == configured.revision
    )


async def run_frontier_adjudication(
    request: FrontierAdjudicationRequest,
    config: FrontierAdjudicationConfig,
    ai_handler: BaseAiHandler,
    *,
    current_identity: Callable[[], Any],
    deadline_monotonic: Optional[float] = None,
) -> FrontierAdjudicationResult:
    """Adjudicate one verified finding without publishing or mutating review output."""

    if not config.enabled or not request.signals.requires_escalation:
        return _unavailable(request, FrontierState.NOT_REQUIRED, "escalation_not_required")
    if request.configuration_hash != config.configuration_hash or request.prompt_hash != config.prompt_hash:
        return _unavailable(request, FrontierState.UNAVAILABLE, "configuration_identity_mismatch")
    if request.policy_version != config.policy_version:
        return _unavailable(request, FrontierState.UNAVAILABLE, "policy_identity_mismatch")
    finding_id = request.candidate.stable_finding_id
    started_at = time.monotonic()
    per_call_deadline = started_at + config.stage_timeout_seconds
    if deadline_monotonic is None:
        deadline_monotonic = per_call_deadline
    else:
        deadline_monotonic = min(deadline_monotonic, per_call_deadline)
    try:
        current_snapshot_id = await _refresh_identity_before(
            current_identity,
            deadline_monotonic,
        )
    except _FrontierIdentityRefreshError:
        record_adjudication_result(
            finding_id,
            provider=None,
            model_revision=None,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="identity_refresh_failed",
        )
        return _unavailable(
            request,
            FrontierState.UNAVAILABLE,
            "identity_refresh_failed",
        )
    except asyncio.TimeoutError:
        record_adjudication_result(
            finding_id,
            provider=None,
            model_revision=None,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.TIMEOUT.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="timeout",
        )
        return _unavailable(request, FrontierState.TIMEOUT, "timeout")
    if current_snapshot_id is None:
        record_adjudication_result(
            finding_id,
            provider=None,
            model_revision=None,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="identity_refresh_unavailable",
        )
        return _unavailable(request, FrontierState.UNAVAILABLE, "identity_refresh_unavailable")
    if current_snapshot_id != request.snapshot_id:
        record_adjudication_result(
            finding_id,
            provider=None,
            model_revision=None,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.STALE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="stale_snapshot",
        )
        return _unavailable(request, FrontierState.STALE, "stale_snapshot")
    if getattr(ai_handler, "supports_frontier_adjudication_telemetry", False) is not True:
        return _unavailable(
            request,
            FrontierState.UNAVAILABLE,
            "handler_telemetry_unsupported",
        )

    attribution = f"frontier_adjudication:{finding_id}"
    route = replace(config.route, attribution=attribution)
    variables = {
        "input_schema_version": request.schema_version,
        "output_schema_version": config.output_schema_version,
        "adjudication_input_json": json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":")),
    }
    try:
        system = TokenHandler.render_plain_text_prompt(config.system_prompt, variables)
        user = TokenHandler.render_plain_text_prompt(config.user_prompt, variables)

        async def attempt(model: str) -> str:
            response, _ = await ai_handler.chat_completion(
                model=model,
                system=system,
                user=user,
                temperature=0,
            )
            if not isinstance(response, str):
                raise FrontierContractError("frontier response must be text")
            return response

        raw = await asyncio.wait_for(
            retry_with_fallback_models(attempt, model_route=route),
            timeout=max(0, deadline_monotonic - time.monotonic()),
        )
        parsed = _validate_output(raw, request, config)
    except asyncio.TimeoutError:
        state = FrontierState.TIMEOUT
        failure_reason = "timeout"
        parsed = None
    except FrontierContractError:
        state = FrontierState.MALFORMED_OUTPUT
        failure_reason = "malformed_output"
        parsed = None
    except Exception:
        state = FrontierState.PROVIDER_FAILURE
        failure_reason = "provider_failure"
        parsed = None
    else:
        state = {
            FrontierDecision.CONFIRM: FrontierState.CONFIRMED,
            FrontierDecision.REJECT: FrontierState.REJECTED,
            FrontierDecision.UNAVAILABLE: FrontierState.UNAVAILABLE,
        }[parsed["decision"]]
        failure_reason = parsed["failure_reason"]

    details = get_run_details()
    existing = details.adjudication_runs.get(finding_id) if details is not None else None
    configured_identity = _model_identity(
        config,
        existing.model_used if existing is not None else None,
        existing.deployment_id if existing is not None else None,
    )
    actual_provider = existing.provider if existing is not None else None
    actual_revision = existing.model_revision if existing is not None else None
    try:
        current_snapshot_id = await _refresh_identity_before(
            current_identity,
            deadline_monotonic,
        )
    except asyncio.TimeoutError:
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.TIMEOUT.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="timeout",
        )
        return _unavailable(request, FrontierState.TIMEOUT, "timeout")
    except _FrontierIdentityRefreshError:
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="identity_refresh_failed",
        )
        return _unavailable(
            request,
            FrontierState.UNAVAILABLE,
            "identity_refresh_failed",
        )
    if current_snapshot_id is None:
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="identity_refresh_unavailable",
        )
        return _unavailable(request, FrontierState.UNAVAILABLE, "identity_refresh_unavailable")
    if current_snapshot_id != request.snapshot_id:
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.STALE.value,
            latency_seconds=time.monotonic() - started_at,
            failure_reason="stale_snapshot",
        )
        return _unavailable(request, FrontierState.STALE, "stale_snapshot")
    elapsed = time.monotonic() - started_at
    record_adjudication_result(
        finding_id,
        provider=actual_provider,
        model_revision=actual_revision,
        model_attempts_configured=config.route.model_retries,
        provider_retries_configured=config.route.provider_retries,
        prompt_version=config.prompt_version,
        input_schema_version=config.input_schema_version,
        schema_version=config.output_schema_version,
        state=state.value,
        latency_seconds=elapsed,
        confidence=parsed["confidence"] if parsed is not None else None,
        failure_reason=failure_reason,
    )
    if parsed is None:
        return _unavailable(request, state, failure_reason or "unavailable")

    telemetry = adjudication_runs_to_dict().get(finding_id, {})
    if not _completion_identity_verified(configured_identity, telemetry):
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=elapsed,
            failure_reason="completion_identity_unverified",
        )
        return _unavailable(request, FrontierState.UNAVAILABLE, "completion_identity_unverified")
    if not _telemetry_complete(telemetry):
        record_adjudication_result(
            finding_id,
            provider=actual_provider,
            model_revision=actual_revision,
            model_attempts_configured=config.route.model_retries,
            provider_retries_configured=config.route.provider_retries,
            prompt_version=config.prompt_version,
            input_schema_version=config.input_schema_version,
            schema_version=config.output_schema_version,
            state=FrontierState.UNAVAILABLE.value,
            latency_seconds=elapsed,
            failure_reason="telemetry_incomplete",
        )
        return _unavailable(request, FrontierState.UNAVAILABLE, "telemetry_incomplete")
    if parsed["decision"] is FrontierDecision.UNAVAILABLE:
        return _unavailable(request, FrontierState.UNAVAILABLE, failure_reason or "model_unavailable")
    if parsed["decision"] is FrontierDecision.REJECT:
        return FrontierAdjudicationResult(
            schema_version=config.output_schema_version,
            stable_finding_id=request.candidate.stable_finding_id,
            decision=FrontierDecision.REJECT,
            state=FrontierState.REJECTED,
            confidence=parsed["confidence"],
            evidence_citations=parsed["citations"],
            telemetry=telemetry,
        )

    severity = severity_at_least(parsed["severity"], request.signals.deterministic_severity_floor)
    normalized = NormalizedFrontierFinding(
        stable_finding_id=request.candidate.stable_finding_id,
        root_cause_id=request.candidate.root_cause_id,
        severity=severity,
        path=request.candidate.path,
        side=request.candidate.side,
        start_line=request.candidate.start_line,
        end_line=request.candidate.end_line,
        evidence_citations=parsed["citations"],
    )
    return FrontierAdjudicationResult(
        schema_version=config.output_schema_version,
        stable_finding_id=request.candidate.stable_finding_id,
        decision=FrontierDecision.CONFIRM,
        state=FrontierState.CONFIRMED,
        confidence=parsed["confidence"],
        evidence_citations=parsed["citations"],
        normalized_finding=normalized,
        telemetry=telemetry,
    )
