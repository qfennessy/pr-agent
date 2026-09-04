"""Bounded repository-evidence retrieval and result validation for review candidates."""

import asyncio
import copy
import fnmatch
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.git_patch_processing import (
    RE_HUNK_HEADER,
    iter_git_patch_lines,
    split_git_file_lines,
    strip_git_line_ending,
)
from pr_agent.algo.review_specialists import (
    SpecialistBatchResult,
    SpecialistInput,
    SpecialistRole,
    SpecialistState,
)
from pr_agent.algo.types import EDIT_TYPE
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

CANDIDATE_VERIFICATION_CONFIG_SCHEMA_VERSION = "candidate-verification-config-v1"
CANDIDATE_VERIFICATION_PROMPT_VERSION = "candidate-verification-prompt-v1"
CANDIDATE_VERIFICATION_INPUT_SCHEMA_VERSION = "candidate-verification-input-v1"
CANDIDATE_VERIFICATION_OUTPUT_SCHEMA_VERSION = "candidate-verification-output-v1"

_TELEMETRY_SAFE_DECISION_REASONS = frozenset({
    "duplicate_decision",
    "duplicate_verified_finding",
    "location_not_in_changed_lines",
    "missing_decision",
    "required_context_unavailable",
    "changed_code_evidence_unavailable",
    "trusted_identity_collision",
    "trusted_identity_unavailable",
    "unknown_candidate",
    "unverified_or_incomplete_evidence",
})
_REPO_FETCH_MAX_WORKERS = 4
_REPO_FETCH_SLOTS = threading.BoundedSemaphore(_REPO_FETCH_MAX_WORKERS)
_ATOMIC_CHANGED_EVIDENCE_SOURCES = frozenset({
    "changed_head",
    "changed_patch",
    "changed_context_patch",
})
_PATCH_CHANGED_EVIDENCE_SOURCES = frozenset({"changed_patch", "changed_context_patch"})
_CHANGED_EVIDENCE_SOURCES = _ATOMIC_CHANGED_EVIDENCE_SOURCES | {"changed_context_head"}
_MAX_CONTEXT_SYMBOLS_PER_CANDIDATE = 32
_MAX_CONTEXT_SYMBOL_CHARACTERS = 256
_SHA256_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class CandidateVerificationOutputBudgetError(ValueError):
    """Raised when no safe bounded verifier completion budget can be resolved."""


def _is_atomic_prompt_evidence(item: Mapping) -> bool:
    return bool(
        item.get("source") in _ATOMIC_CHANGED_EVIDENCE_SOURCES
        or item.get("required_evidence")
    )


class _RepositoryFetchCapacityExhausted(RuntimeError):
    """Raised when timed-out provider work already occupies every bounded fetch slot."""


class _ChangedContextCollectionStopped(RuntimeError):
    """Raised when a retrieval budget stops lazy changed-context collection."""


async def _bounded_repo_file_fetch(git_provider, path: str, timeout_seconds: float,
                                   from_pr_head: bool = False):
    """Run a synchronous provider fetch without allowing abandoned work to grow unboundedly."""
    if timeout_seconds <= 0:
        raise asyncio.TimeoutError
    if not _REPO_FETCH_SLOTS.acquire(blocking=False):
        raise _RepositoryFetchCapacityExhausted

    loop = asyncio.get_running_loop()
    result = loop.create_future()

    def deliver(value=None, error=None):
        if result.done():
            return
        if error is not None:
            result.set_exception(error)
        else:
            result.set_result(value)

    def fetch():
        try:
            if from_pr_head:
                value = git_provider.get_pr_head_file_content(path)
            else:
                value = git_provider.get_repo_file_content(path, False)
            outcome = (value, None)
        except Exception as exc:
            outcome = (None, exc)
        finally:
            _REPO_FETCH_SLOTS.release()
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:
            # The timed-out caller may have closed its event loop. The bounded
            # slot is already released and no result remains to deliver.
            pass

    try:
        threading.Thread(
            target=fetch,
            name="pr-agent-repo-context-fetch",
            daemon=True,
        ).start()
    except Exception:
        _REPO_FETCH_SLOTS.release()
        raise
    return await asyncio.wait_for(result, timeout=timeout_seconds)


@dataclass(frozen=True)
class VerificationBudgets:
    max_candidates: int = 3
    max_files: int = 6
    max_lines_per_file: int = 160
    max_total_lines: int = 600
    max_context_tokens: int = 6000
    timeout_seconds: float = 10.0


def _candidate_verification_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _freeze_static_evidence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_static_evidence_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_static_evidence_value(item) for item in value)
    return value


def _thaw_static_evidence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_static_evidence_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_static_evidence_value(item) for item in value]
    return value


def _provider_control_value(value: Any) -> Any:
    """Return a detached JSON value suitable for provider-control identity."""
    if isinstance(value, Mapping):
        return {str(key): _provider_control_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_provider_control_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def candidate_verification_provider_controls_hash(
    settings: Any,
    *,
    claude_extended_thinking_models: Optional[tuple[str, ...] | list[str]] = None,
    checkpoint_replay: bool = False,
) -> str:
    """Bind ambient LiteLLM controls that can change verifier request semantics."""

    def section_value(section_name: str, key: str, default: Any = None) -> Any:
        section = getattr(settings, section_name, None)
        if isinstance(section, Mapping):
            return section.get(key, default)
        getter = getattr(section, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(section, key, default) if section is not None else default

    openrouter = settings.get("openrouter", {}) or {}
    if not isinstance(openrouter, Mapping):
        openrouter = {}
    payload = {
        "config": {
            key: _provider_control_value(section_value("config", key, default))
            for key, default in (
                ("reasoning_effort", None),
                ("custom_reasoning_model", False),
                ("max_output_tokens", 0),
                ("custom_model_max_tokens", 0),
                ("max_model_tokens", 0),
                ("enable_claude_adaptive_thinking", False),
                ("enable_claude_extended_thinking", False),
                ("extended_thinking_budget_tokens", 2048),
                ("extended_thinking_max_output_tokens", 4096),
                ("claude_extended_thinking_models_override", []),
                ("seed", -1),
                ("add_user_to_requests", False),
                ("git_provider", ""),
            )
        },
        "openrouter": {
            key: _provider_control_value(openrouter.get(key, default))
            for key, default in (
                ("provider_only", []),
                ("provider_order", []),
                ("allow_fallbacks", True),
                ("reasoning_effort", ""),
                ("reasoning_max_tokens", 0),
                ("max_tokens", 0),
            )
        },
        "litellm": {
            key: _provider_control_value(section_value("litellm", key, default))
            for key, default in (
                ("cache_control_injection_points", None),
                ("custom_llm_provider", ""),
                ("extra_headers", None),
                ("extra_body", None),
                ("model_id", None),
                ("enable_callbacks", False),
            )
        },
        "claude_extended_thinking_models": _provider_control_value(
            claude_extended_thinking_models
        ),
    }
    if checkpoint_replay:
        payload["config"]["add_user_to_requests"] = False
        payload["config"]["git_provider"] = "plain-diff"
        payload["litellm"]["extra_headers"] = {}
        payload["litellm"]["enable_callbacks"] = False
    return _candidate_verification_hash(payload)


@dataclass(frozen=True, slots=True)
class CandidateVerificationConfig:
    """One immutable candidate-verification route, budget, and prompt contract."""

    route: AIModelRoute
    budgets: VerificationBudgets
    max_calls: int
    max_sensitive_candidates: int
    sensitive_path_globs: tuple[str, ...]
    consume_specialist_prioritization: bool
    temperature: float
    strict_output_policy: bool
    max_findings: int
    static_analysis_evidence_hash: str
    provider_controls_hash: str
    effective_output_token_caps: tuple[int, ...]
    system_prompt: str = field(repr=False)
    user_prompt: str = field(repr=False)
    static_analysis_evidence: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    prompt_version: str = CANDIDATE_VERIFICATION_PROMPT_VERSION
    input_schema_version: str = CANDIDATE_VERIFICATION_INPUT_SCHEMA_VERSION
    output_schema_version: str = CANDIDATE_VERIFICATION_OUTPUT_SCHEMA_VERSION
    schema_version: str = CANDIDATE_VERIFICATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_VERIFICATION_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported candidate verification configuration schema")
        if self.prompt_version != CANDIDATE_VERIFICATION_PROMPT_VERSION:
            raise ValueError("unsupported candidate verification prompt version")
        if self.input_schema_version != CANDIDATE_VERIFICATION_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported candidate verification input schema")
        if self.output_schema_version != CANDIDATE_VERIFICATION_OUTPUT_SCHEMA_VERSION:
            raise ValueError("unsupported candidate verification output schema")
        for name, value in (
            ("max_calls", self.max_calls),
            ("max_sensitive_candidates", self.max_sensitive_candidates),
            ("max_findings", self.max_findings),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"candidate verification {name} must be a non-negative integer")
        for name, value in (
            ("max_candidates", self.budgets.max_candidates),
            ("max_files", self.budgets.max_files),
            ("max_lines_per_file", self.budgets.max_lines_per_file),
            ("max_total_lines", self.budgets.max_total_lines),
            ("max_context_tokens", self.budgets.max_context_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"candidate verification {name} must be a non-negative integer")
        if (
            not isinstance(self.budgets.timeout_seconds, (int, float))
            or isinstance(self.budgets.timeout_seconds, bool)
            or not isfinite(self.budgets.timeout_seconds)
            or self.budgets.timeout_seconds < 0
        ):
            raise ValueError("candidate verification timeout_seconds must be finite and non-negative")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not isfinite(self.temperature)
        ):
            raise ValueError("candidate verification temperature must be finite")
        if not isinstance(self.consume_specialist_prioritization, bool):
            raise ValueError("candidate verification specialist prioritization flag must be boolean")
        if not isinstance(self.strict_output_policy, bool):
            raise ValueError("candidate verification strict output policy flag must be boolean")
        if not isinstance(self.static_analysis_evidence_hash, str) or not _SHA256_ID_PATTERN.fullmatch(
            self.static_analysis_evidence_hash
        ):
            raise ValueError("candidate verification static analysis evidence hash is invalid")
        if not isinstance(self.provider_controls_hash, str) or not _SHA256_ID_PATTERN.fullmatch(
            self.provider_controls_hash
        ):
            raise ValueError("candidate verification provider controls hash is invalid")
        if (
            not isinstance(self.effective_output_token_caps, tuple)
            or len(self.effective_output_token_caps) != len(self.route.models)
            or any(
                isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
                for cap in self.effective_output_token_caps
            )
        ):
            raise ValueError("candidate verification effective output token caps are invalid")
        if not isinstance(self.static_analysis_evidence, tuple):
            raise ValueError("candidate verification static analysis evidence must be an immutable tuple")
        if _candidate_verification_hash(
            _thaw_static_evidence_value(self.static_analysis_evidence)
        ) != self.static_analysis_evidence_hash:
            raise ValueError("candidate verification static analysis evidence hash mismatch")
        if not isinstance(self.sensitive_path_globs, tuple):
            raise ValueError("candidate verification sensitive path globs must be an immutable tuple")
        if any(not isinstance(glob, str) or not glob.strip() for glob in self.sensitive_path_globs):
            raise ValueError("candidate verification sensitive path globs must be non-blank strings")
        if (
            not isinstance(self.system_prompt, str)
            or not self.system_prompt.strip()
            or not isinstance(self.user_prompt, str)
            or not self.user_prompt.strip()
        ):
            raise ValueError("candidate verification prompt must be non-blank")

    def _configuration_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route": {
                "models": list(self.route.models),
                "deployments": list(self.route.deployments),
                "timeout_seconds": self.route.timeout_seconds,
                "model_retries": self.route.model_retries,
                "provider_retries": self.route.provider_retries,
                "max_output_tokens": self.route.max_output_tokens,
                "attribution": self.route.attribution,
                "collect_cost": self.route.collect_cost,
            },
            "budgets": {
                "max_candidates": self.budgets.max_candidates,
                "max_files": self.budgets.max_files,
                "max_lines_per_file": self.budgets.max_lines_per_file,
                "max_total_lines": self.budgets.max_total_lines,
                "max_context_tokens": self.budgets.max_context_tokens,
                "timeout_seconds": self.budgets.timeout_seconds,
            },
            "max_calls": self.max_calls,
            "max_sensitive_candidates": self.max_sensitive_candidates,
            "sensitive_path_globs": list(self.sensitive_path_globs),
            "consume_specialist_prioritization": self.consume_specialist_prioritization,
            "temperature": self.temperature,
            "strict_output_policy": self.strict_output_policy,
            "max_findings": self.max_findings,
            "static_analysis_evidence_hash": self.static_analysis_evidence_hash,
            "provider_controls_hash": self.provider_controls_hash,
            "effective_output_token_caps": list(self.effective_output_token_caps),
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
        }

    @property
    def configuration_hash(self) -> str:
        return _candidate_verification_hash(self._configuration_payload())

    @property
    def stage_plan_configuration_hash(self) -> str:
        """Hash only verifier controls shared by every case in one evaluation arm."""

        payload = self._configuration_payload()
        payload.pop("static_analysis_evidence_hash")
        return _candidate_verification_hash(payload)

    @property
    def prompt_hash(self) -> str:
        return _candidate_verification_hash({
            "prompt_version": self.prompt_version,
            "system": self.system_prompt,
            "user": self.user_prompt,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._configuration_payload(),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "configuration_hash": self.configuration_hash,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        static_analysis_evidence: Optional[tuple[Any, ...] | list[Any]] = None,
    ) -> "CandidateVerificationConfig":
        expected_fields = {
            "schema_version", "route", "budgets", "max_calls", "max_sensitive_candidates",
            "sensitive_path_globs", "consume_specialist_prioritization", "temperature",
            "strict_output_policy", "max_findings", "prompt_version", "input_schema_version",
            "output_schema_version", "static_analysis_evidence_hash", "system_prompt", "user_prompt",
            "provider_controls_hash", "effective_output_token_caps", "configuration_hash", "prompt_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ValueError("invalid candidate verification configuration fields")
        route = value.get("route")
        budgets = value.get("budgets")
        if not isinstance(route, Mapping) or not isinstance(budgets, Mapping):
            raise ValueError("invalid candidate verification route or budgets")
        if set(route) != {
            "models", "deployments", "timeout_seconds", "model_retries", "provider_retries",
            "max_output_tokens", "attribution", "collect_cost",
        } or set(budgets) != {
            "max_candidates", "max_files", "max_lines_per_file", "max_total_lines",
            "max_context_tokens", "timeout_seconds",
        }:
            raise ValueError("invalid candidate verification route or budget fields")
        if not isinstance(route.get("models"), list) or not isinstance(route.get("deployments"), list):
            raise ValueError("invalid candidate verification route values")
        models = route["models"]
        deployments = route["deployments"]
        if not models or any(not isinstance(model, str) or not model.strip() for model in models):
            raise ValueError("invalid candidate verification route models")
        if len(deployments) != len(models) or any(
            deployment is not None
            and (not isinstance(deployment, str) or not deployment.strip())
            for deployment in deployments
        ):
            raise ValueError("invalid candidate verification route deployments")
        timeout_seconds = route["timeout_seconds"]
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("invalid candidate verification route timeout")
        model_retries = route["model_retries"]
        provider_retries = route["provider_retries"]
        max_output_tokens = route["max_output_tokens"]
        if isinstance(model_retries, bool) or not isinstance(model_retries, int) or model_retries < 1:
            raise ValueError("invalid candidate verification route model retries")
        if isinstance(provider_retries, bool) or not isinstance(provider_retries, int) or provider_retries < 0:
            raise ValueError("invalid candidate verification route provider retries")
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens < 1
        ):
            raise ValueError("invalid candidate verification route output cap")
        if route["attribution"] != "candidate_verification" or not isinstance(route["collect_cost"], bool):
            raise ValueError("invalid candidate verification route attribution")
        if not isinstance(value.get("sensitive_path_globs"), list):
            raise ValueError("invalid candidate verification sensitive path globs")
        effective_output_token_caps = value.get("effective_output_token_caps")
        if (
            not isinstance(effective_output_token_caps, list)
            or len(effective_output_token_caps) != len(models)
            or any(
                isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
                for cap in effective_output_token_caps
            )
        ):
            raise ValueError("invalid candidate verification effective output token caps")
        serialized_evidence_hash = value.get("static_analysis_evidence_hash")
        if static_analysis_evidence is None:
            if serialized_evidence_hash != _candidate_verification_hash([]):
                raise ValueError("candidate verification replay requires static analysis evidence reattachment")
            frozen_static_evidence: tuple[Any, ...] = ()
        else:
            if not isinstance(static_analysis_evidence, (list, tuple)):
                raise ValueError("candidate verification static analysis evidence must be a list or tuple")
            captured_static_evidence = copy.deepcopy(_thaw_static_evidence_value(static_analysis_evidence))
            frozen_static_evidence = tuple(
                _freeze_static_evidence_value(item) for item in captured_static_evidence
            )
        candidate = cls(
            route=AIModelRoute(
                models=tuple(models),
                deployments=tuple(deployments),
                timeout_seconds=timeout_seconds,
                model_retries=model_retries,
                provider_retries=provider_retries,
                max_output_tokens=max_output_tokens,
                attribution=route["attribution"],
                collect_cost=route["collect_cost"],
            ),
            budgets=VerificationBudgets(
                max_candidates=budgets.get("max_candidates"),
                max_files=budgets.get("max_files"),
                max_lines_per_file=budgets.get("max_lines_per_file"),
                max_total_lines=budgets.get("max_total_lines"),
                max_context_tokens=budgets.get("max_context_tokens"),
                timeout_seconds=budgets.get("timeout_seconds"),
            ),
            max_calls=value.get("max_calls"),
            max_sensitive_candidates=value.get("max_sensitive_candidates"),
            sensitive_path_globs=tuple(value.get("sensitive_path_globs", ())),
            consume_specialist_prioritization=value.get("consume_specialist_prioritization"),
            temperature=value.get("temperature"),
            strict_output_policy=value.get("strict_output_policy"),
            max_findings=value.get("max_findings"),
            static_analysis_evidence_hash=serialized_evidence_hash,
            provider_controls_hash=value.get("provider_controls_hash"),
            effective_output_token_caps=tuple(effective_output_token_caps),
            system_prompt=value.get("system_prompt"),
            user_prompt=value.get("user_prompt"),
            static_analysis_evidence=frozen_static_evidence,
            prompt_version=value.get("prompt_version"),
            input_schema_version=value.get("input_schema_version"),
            output_schema_version=value.get("output_schema_version"),
            schema_version=value.get("schema_version"),
        )
        if value.get("configuration_hash") != candidate.configuration_hash:
            raise ValueError("candidate verification configuration hash mismatch")
        if value.get("prompt_hash") != candidate.prompt_hash:
            raise ValueError("candidate verification prompt hash mismatch")
        return candidate


def parse_candidate_verification_config(
    section: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    primary_model: str,
    global_deployment: Optional[str] = None,
    azure: bool = False,
    temperature: float = 0.0,
    inherited_max_output_tokens: int = 0,
    default_output_token_cap: int = 1500,
    max_candidates_override: Optional[int] = None,
    strict_output_policy: bool = False,
    static_analysis_evidence_hash: Optional[str] = None,
    static_analysis_evidence: tuple[Any, ...] = (),
    provider_controls_hash: Optional[str] = None,
    output_token_cap_resolver: Optional[Callable[[str, Optional[int]], Optional[int]]] = None,
) -> CandidateVerificationConfig:
    """Parse a verifier contract from explicit mappings without ambient settings reads."""

    if not isinstance(section, Mapping) or not isinstance(prompt, Mapping):
        raise ValueError("candidate verification settings and prompt must be mappings")

    def integer(key: str, default: int) -> int:
        raw = section.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be a non-boolean integer")
        if isinstance(raw, Integral):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip(), 10)
            except ValueError as exc:
                raise ValueError(f"{key} must be a non-boolean integer") from exc
        raise ValueError(f"{key} must be a non-boolean integer")

    def number(key: str, default: float) -> float:
        raw = section.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be a non-boolean number")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a non-boolean number") from exc
        if not isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    def string_tuple(key: str) -> tuple[str, ...]:
        raw = section.get(key, [])
        if raw in (None, "", []):
            return ()
        if isinstance(raw, str):
            values = tuple(part.strip() for part in raw.split(","))
        elif isinstance(raw, (list, tuple)) and all(isinstance(part, str) for part in raw):
            values = tuple(part.strip() for part in raw)
        else:
            raise ValueError(f"{key} must be a string list")
        if any(not value for value in values):
            raise ValueError(f"{key} cannot contain blank entries")
        return values

    def deployment_tuple(key: str) -> tuple[Optional[str], ...]:
        raw = section.get(key, [])
        if raw in (None, "", []):
            return ()
        if isinstance(raw, str):
            values = raw.split(",")
        elif isinstance(raw, (list, tuple)) and all(isinstance(part, str) for part in raw):
            values = raw
        else:
            raise ValueError(f"{key} must be a string list")
        return tuple(str(part).strip() or None for part in values)

    configured_model = str(section.get("candidate_verification_model", "") or "").strip()
    primary_model = str(primary_model or "").strip()
    model = configured_model or primary_model
    if not model:
        raise ValueError("candidate verifier model cannot be blank")
    fallback_models = string_tuple("candidate_verification_fallback_models")
    global_deployment = str(global_deployment or "").strip() or None
    explicit_deployment = str(section.get("candidate_verification_deployment", "") or "").strip() or None
    azure_route = azure is True or global_deployment is not None
    if explicit_deployment is not None:
        deployment = explicit_deployment
    elif not configured_model or model == primary_model:
        deployment = global_deployment
    elif azure_route:
        raise ValueError("candidate_verification_deployment is required when the Azure verifier model differs")
    else:
        deployment = None
    fallback_deployments = deployment_tuple("candidate_verification_fallback_deployments")
    if fallback_deployments and len(fallback_deployments) != len(fallback_models):
        raise ValueError("candidate_verification_fallback_deployments must match fallback models")
    if not fallback_deployments:
        if azure_route and fallback_models:
            raise ValueError("Azure verifier fallback models require matching fallback deployments")
        fallback_deployments = (None,) * len(fallback_models)
    deployments = (deployment, *fallback_deployments)
    if azure_route and any(item is None for item in deployments):
        raise ValueError("every Azure verifier model requires a deployment")

    configured_output_cap = integer("candidate_verification_max_output_tokens", 0)
    if configured_output_cap < 0:
        raise ValueError("candidate_verification_max_output_tokens cannot be negative")
    inherited_output_cap = 0
    if configured_output_cap == 0:
        if isinstance(inherited_max_output_tokens, bool):
            raise ValueError("max_output_tokens must be a non-boolean integer")
        try:
            inherited_output_cap = int(inherited_max_output_tokens)
        except (TypeError, ValueError):
            inherited_output_cap = 0
    route_output_cap = configured_output_cap or inherited_output_cap or None
    models = (model, *fallback_models)

    def resolve_output_caps(request_cap: Optional[int]) -> tuple[Optional[int], ...]:
        if output_token_cap_resolver is None:
            return tuple(request_cap for _ in models)
        return tuple(output_token_cap_resolver(route_model, request_cap) for route_model in models)

    try:
        effective_output_caps = resolve_output_caps(route_output_cap)
    except ValueError as exc:
        raise CandidateVerificationOutputBudgetError(str(exc)) from exc
    if any(cap is None for cap in effective_output_caps):
        if (
            isinstance(default_output_token_cap, bool)
            or not isinstance(default_output_token_cap, int)
            or default_output_token_cap < 1
        ):
            raise ValueError("default_output_token_cap must be a positive integer")
        route_output_cap = default_output_token_cap
        try:
            effective_output_caps = resolve_output_caps(route_output_cap)
        except ValueError as exc:
            raise CandidateVerificationOutputBudgetError(str(exc)) from exc
    if any(
        cap is None or isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
        for cap in effective_output_caps
    ):
        raise CandidateVerificationOutputBudgetError("candidate verifier output cap could not be resolved")
    if max_candidates_override is not None and (
        isinstance(max_candidates_override, bool)
        or not isinstance(max_candidates_override, int)
        or max_candidates_override <= 0
    ):
        raise ValueError("routed max_verification_candidates must be a positive integer")
    max_candidates = (
        max_candidates_override
        if max_candidates_override is not None
        else max(0, integer("candidate_verification_max_candidates", 3))
    )
    timeout_seconds = max(0.0, number("candidate_verification_timeout_seconds", 10))
    globs = string_tuple("candidate_verification_sensitive_path_globs")
    specialist_value = section.get("candidate_verification_consume_specialist_prioritization", False)
    if isinstance(specialist_value, str):
        specialist_value = specialist_value.strip().lower() in {"1", "true", "yes", "on"}
    else:
        specialist_value = bool(specialist_value)

    def prompt_string(key: str, default: Optional[str] = None) -> str:
        raw = prompt.get(key, default)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"candidate verification prompt {key} must be a non-blank string")
        return raw

    return CandidateVerificationConfig(
        route=AIModelRoute(
            models=models,
            deployments=deployments,
            timeout_seconds=max(0.001, timeout_seconds),
            model_retries=1,
            provider_retries=0,
            max_output_tokens=route_output_cap,
            attribution="candidate_verification",
        ),
        budgets=VerificationBudgets(
            max_candidates=max_candidates,
            max_files=max(0, integer("candidate_verification_max_files", 6)),
            max_lines_per_file=max(0, integer("candidate_verification_max_lines_per_file", 160)),
            max_total_lines=max(0, integer("candidate_verification_max_total_lines", 600)),
            max_context_tokens=max(0, integer("candidate_verification_max_context_tokens", 6000)),
            timeout_seconds=timeout_seconds,
        ),
        max_calls=max(0, integer("candidate_verification_max_model_calls", 1)),
        max_sensitive_candidates=max(0, integer("candidate_verification_max_sensitive_candidates", 6)),
        sensitive_path_globs=globs,
        consume_specialist_prioritization=specialist_value,
        temperature=float(temperature),
        strict_output_policy=strict_output_policy,
        max_findings=max(0, integer("num_max_findings", 3)),
        static_analysis_evidence_hash=(
            static_analysis_evidence_hash
            if static_analysis_evidence_hash is not None
            else _candidate_verification_hash([])
        ),
        provider_controls_hash=(
            provider_controls_hash
            if provider_controls_hash is not None
            else _candidate_verification_hash({})
        ),
        effective_output_token_caps=tuple(effective_output_caps),
        system_prompt=prompt_string("system"),
        user_prompt=prompt_string("user"),
        static_analysis_evidence=static_analysis_evidence,
        prompt_version=prompt_string("prompt_version", CANDIDATE_VERIFICATION_PROMPT_VERSION),
        input_schema_version=prompt_string("input_schema_version", CANDIDATE_VERIFICATION_INPUT_SCHEMA_VERSION),
        output_schema_version=prompt_string("schema_version", CANDIDATE_VERIFICATION_OUTPUT_SCHEMA_VERSION),
    )


def load_production_candidate_verification_config(
    *,
    settings: Any = None,
    azure: bool = False,
    max_candidates_override: Optional[int] = None,
    strict_output_policy: bool = False,
    default_output_token_cap: int = 1500,
    claude_extended_thinking_models: Optional[tuple[str, ...] | list[str]] = None,
    output_token_cap_resolver: Optional[Callable[[str, Optional[int]], Optional[int]]] = None,
) -> CandidateVerificationConfig:
    """Capture the production verifier settings and prompt once for one run."""

    settings = settings if settings is not None else get_settings()
    prompt_section = settings.pr_review_verification_prompt
    prompt = {
        key: value
        for key in ("system", "user", "prompt_version", "input_schema_version", "schema_version")
        if (value := getattr(prompt_section, key, None)) is not None
    }
    runtime_data = settings.get("data", {}) or {}
    raw_static_evidence = (
        runtime_data.get("static_analysis_evidence", [])
        if isinstance(runtime_data, Mapping)
        else []
    )
    if not isinstance(raw_static_evidence, list):
        raw_static_evidence = []
    captured_static_evidence = copy.deepcopy(raw_static_evidence)
    static_analysis_evidence_hash = _candidate_verification_hash(captured_static_evidence)
    frozen_static_evidence = tuple(
        _freeze_static_evidence_value(item)
        for item in captured_static_evidence
    )
    provider_controls_hash = candidate_verification_provider_controls_hash(
        settings,
        claude_extended_thinking_models=claude_extended_thinking_models,
    )
    config = parse_candidate_verification_config(
        dict(settings.pr_reviewer),
        prompt,
        primary_model=settings.config.model,
        global_deployment=settings.get("openai.deployment_id", ""),
        azure=azure,
        temperature=settings.config.temperature,
        inherited_max_output_tokens=settings.config.get("max_output_tokens", 0),
        default_output_token_cap=default_output_token_cap,
        max_candidates_override=max_candidates_override,
        strict_output_policy=strict_output_policy,
        static_analysis_evidence_hash=static_analysis_evidence_hash,
        static_analysis_evidence=frozen_static_evidence,
        provider_controls_hash=provider_controls_hash,
        output_token_cap_resolver=output_token_cap_resolver,
    )
    if candidate_verification_provider_controls_hash(
        settings,
        claude_extended_thinking_models=claude_extended_thinking_models,
    ) != provider_controls_hash:
        raise ValueError("candidate verification provider controls changed during capture")
    return config


_COMPLETE_RETRIEVAL_REQUEST_STATUSES = frozenset({
    "retrieved",
    "satisfied_by_changed_head",
    "satisfied_by_changed_patch",
})


def retrieval_request_is_complete(request: dict) -> bool:
    """Return whether a context request is complete for verification status."""
    return (
        not bool(request.get("required"))
        or request.get("status") in _COMPLETE_RETRIEVAL_REQUEST_STATUSES
    )


def _request_prompt_evidence(request: dict, evidence: list[dict]) -> list[dict]:
    """Return prompt-visible evidence bound to one candidate context request."""
    candidate_id = request.get("candidate_id")
    evidence_id = request.get("evidence_id")
    path = request.get("path")
    if not evidence_id:
        return []
    matches = []
    for item in evidence:
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        if (
            candidate_id in candidate_ids
            and item.get("evidence_id") == evidence_id
            and item.get("path") == path
            and str(item.get("content") or "").strip()
        ):
            matches.append(item)
    return matches


def _request_has_prompt_visible_symbols(request: dict, evidence: list[dict]) -> bool:
    """Require every symbol assigned to a request to survive final prompt clipping."""
    required_symbols = request.get("_required_context_symbols") or []
    if not required_symbols:
        return True
    visible_content = [
        str(item.get("content") or "")
        for item in _request_prompt_evidence(request, evidence)
    ]
    return all(
        any(symbol in content for content in visible_content)
        for symbol in required_symbols
    )


def telemetry_safe_artifact(artifact: dict) -> dict:
    """Return provider-neutral telemetry without repository or model-generated text."""
    safe = {}
    scalar_keys = {
        "enabled", "status", "model_calls", "verifier_attempts", "candidate_count",
        "verified_count", "model", "proposal_source", "proposal_shape",
        "proposed_candidate_count", "accepted_model_candidate_count",
        "sensitive_candidate_count", "candidate_rejection_count",
        "verifier_verified_count", "finding_limit_dropped", "rejected_count", "failure",
        "verifier_latency_seconds", "publication_safe", "first_pass_finish_reason",
        "first_pass_generation_complete",
    }
    mapping_keys = {
        "specialist_prioritization", "sensitive_audit_coverage", "prompt_budget",
        "verifier_usage", "verifier_cost", "model_candidate_coverage",
        "prompt_evidence_coverage",
    }
    for key in scalar_keys:
        if key in artifact and isinstance(artifact[key], (str, int, float, bool, type(None))):
            safe[key] = artifact[key]
    for key in (
        "configuration_hash", "prompt_hash", "static_analysis_evidence_hash", "provider_controls_hash",
    ):
        value = artifact.get(key)
        if isinstance(value, str) and _SHA256_ID_PATTERN.fullmatch(value):
            safe[key] = value
    for key, expected in (
        ("config_schema_version", CANDIDATE_VERIFICATION_CONFIG_SCHEMA_VERSION),
        ("prompt_version", CANDIDATE_VERIFICATION_PROMPT_VERSION),
        ("input_schema_version", CANDIDATE_VERIFICATION_INPUT_SCHEMA_VERSION),
        ("output_schema_version", CANDIDATE_VERIFICATION_OUTPUT_SCHEMA_VERSION),
    ):
        if artifact.get(key) == expected:
            safe[key] = expected
    for key in mapping_keys:
        value = artifact.get(key)
        if isinstance(value, dict):
            safe[key] = {
                item_key: item_value for item_key, item_value in value.items()
                if isinstance(item_key, str) and isinstance(item_value, (str, int, float, bool, type(None)))
            }

    rejections = artifact.get("candidate_rejections")
    if isinstance(rejections, list):
        safe["candidate_rejections"] = [
            {
                key: item[key]
                for key in (
                    "candidate_id", "reason", "sensitive_path", "total_count",
                    "selected_count", "omitted_count",
                )
                if key in item
            }
            for item in rejections if isinstance(item, dict)
        ]

    decisions = artifact.get("decisions")
    if isinstance(decisions, list):
        safe["decisions"] = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            decision = {
                key: item[key]
                for key in (
                    "candidate_id", "verdict", "evidence_paths", "normalized_severity",
                    "confidence", "disputed", "evidence_status",
                )
                if key in item
            }
            trusted_stable_key = item.get("trusted_stable_key")
            if isinstance(trusted_stable_key, str) and _SHA256_ID_PATTERN.fullmatch(trusted_stable_key):
                decision["trusted_stable_key"] = trusted_stable_key
            if "reason" in item:
                reason = str(item.get("reason") or "")
                decision["reason"] = (
                    reason if reason in _TELEMETRY_SAFE_DECISION_REASONS else "rejected_by_verifier"
                )
            safe["decisions"].append(decision)

    retrieval = artifact.get("retrieval")
    if isinstance(retrieval, dict):
        safe_retrieval = {
            key: retrieval[key]
            for key in (
                "budget_exhausted", "files_read", "changed_evidence_count", "lines_retrieved",
                "context_tokens", "duration_seconds",
            )
            if key in retrieval and isinstance(retrieval[key], (str, int, float, bool, type(None)))
        }
        requests = retrieval.get("requests")
        if isinstance(requests, list):
            safe_retrieval["requests"] = [
                {
                    key: item[key]
                    for key in (
                        "candidate_id", "path", "status", "error", "required", "source",
                        "start_line", "end_line", "excerpt_count",
                    )
                    if key in item and isinstance(item[key], (str, int, float, bool, type(None)))
                }
                for item in requests if isinstance(item, dict)
            ]
        retrieved = retrieval.get("retrieved_evidence")
        if isinstance(retrieved, list):
            safe_retrieval["retrieved_evidence"] = []
            for item in retrieved:
                if not isinstance(item, dict):
                    continue
                summary = {
                    key: item[key]
                    for key in (
                        "candidate_id", "candidate_ids", "source", "path", "start_line", "end_line",
                    )
                    if key in item and isinstance(item[key], (str, int, float, bool, type(None)))
                }
                candidate_ids = item.get("candidate_ids")
                if isinstance(candidate_ids, list):
                    summary["candidate_ids"] = [
                        candidate_id for candidate_id in candidate_ids
                        if isinstance(candidate_id, str)
                    ]
                summary["content_characters"] = len(str(item.get("content") or ""))
                safe_retrieval["retrieved_evidence"].append(summary)
        safe["retrieval"] = safe_retrieval
    return safe


def safe_repo_path(value) -> Optional[str]:
    """Return a normalized relative repository path, or None for unsafe input."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 512 or "\x00" in value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    normalized = str(path)
    return normalized if normalized == value else None


def _iter_changed_line_ranges(
    patch: str,
    side: str,
) -> Iterable[tuple[int, int]]:
    """Yield contiguous changed ranges without retaining one integer per line."""
    if side not in {"new", "old"}:
        return
    old_line = None
    new_line = None
    range_start = None
    range_end = None
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            new_line = int(header.group(3))
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        changed = line.startswith("+") if side == "new" else line.startswith("-")
        changed_line = new_line if side == "new" else old_line
        if changed:
            if range_end is not None and changed_line == range_end + 1:
                range_end = changed_line
            else:
                if range_start is not None:
                    yield range_start, range_end
                range_start = changed_line
                range_end = changed_line
        if not line.startswith("-"):
            new_line += 1
        if not line.startswith("+"):
            old_line += 1
    if range_start is not None:
        yield range_start, range_end


def _changed_range_containing_line(
    patch: str,
    side: str,
    line_number: int,
) -> list[tuple[int, int]]:
    """Return only the changed range needed to validate one candidate location."""
    for start, end in _iter_changed_line_ranges(patch, side):
        if start <= line_number <= end:
            return [(start, end)]
        if start > line_number:
            break
    return []


def _sensitive_change_anchors(
    patch: str,
) -> Iterable[tuple[str, int, int, list[tuple[int, int]]]]:
    """Yield each visible changed range without constructing an unbounded line list."""
    for side in ("new", "old"):
        for start, end in _iter_changed_line_ranges(patch, side):
            yield side, start, end, [(start, end)]


def _line_is_changed(line: int, ranges: list) -> bool:
    return any(start <= line <= end for start, end in ranges)


def _candidate_key(candidate: dict) -> tuple:
    root_cause = " ".join(str(candidate.get("root_cause") or "").casefold().split())
    if root_cause:
        return ("root_cause", root_cause)
    return (
        candidate.get("relevant_file"),
        candidate.get("start_line"),
        " ".join(str(candidate.get("issue_content") or "").casefold().split()),
    )


_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER_TOKEN = re.compile(r"\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)\b")
_STRUCTURAL_KEYWORDS = frozenset({
    "and", "async", "await", "break", "case", "catch", "class", "continue", "def", "do",
    "else", "except", "false", "finally", "for", "foreach", "function", "if", "in", "is",
    "lambda", "match", "new", "none", "not", "null", "or", "raise", "return", "switch",
    "throw", "true", "try", "while", "with", "yield",
})


def _replace_string_literals(value: str) -> str:
    """Replace quoted content in linear time without parsing repository code as a language."""
    shaped = []
    quote = None
    escaped = False
    for character in str(value or ""):
        if quote is None:
            if character in {"'", '"'}:
                shaped.append("<literal>")
                quote = character
            else:
                shaped.append(character)
            continue
        if character in {"\n", "\r"}:
            quote = None
            escaped = False
            shaped.append(character)
        elif escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
    return "".join(shaped)


def _normalized_evidence_shape(content: str) -> str:
    """Normalize names, literals, paths, whitespace, and line numbers while preserving control flow."""
    shaped = _replace_string_literals(content)
    shaped = _NUMBER_TOKEN.sub("<number>", shaped)
    shaped = _IDENTIFIER_TOKEN.sub(
        lambda match: match.group(0).casefold()
        if match.group(0).casefold() in _STRUCTURAL_KEYWORDS else "<identifier>",
        shaped,
    )
    return "".join(shaped.split())


def _changed_anchor_identity_details(
    patch: str,
    start_line: int,
    end_line: int,
    side: str = "new",
) -> tuple[str, int, int]:
    """Derive a range shape, its ordinal, and its total patch occurrence count."""
    old_line = None
    new_line = None
    segment = 0
    last_changed_line = None
    changed_records = []
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            new_line = int(header.group(3))
            segment += 1
            last_changed_line = None
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        is_changed = (
            side == "new" and line.startswith("+")
        ) or (
            side == "old" and line.startswith("-")
        )
        changed_line = new_line if side == "new" else old_line
        if is_changed:
            content = line[1:]
            line_shape = _normalized_evidence_shape(content)
            if last_changed_line is not None and changed_line != last_changed_line + 1:
                segment += 1
            changed_records.append((changed_line, segment, line_shape))
            last_changed_line = changed_line
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1

    target_record_indexes = [
        index
        for index, (line_number, _, _) in enumerate(changed_records)
        if start_line <= line_number <= end_line
    ]
    if not target_record_indexes:
        return "", 0, 0

    def tokenized(records: list[tuple[int, int, str]]) -> list[Optional[str]]:
        tokens = []
        previous_segment = None
        for _, record_segment, line_shape in records:
            if previous_segment is not None and record_segment != previous_segment:
                tokens.append(None)
            tokens.append(line_shape)
            previous_segment = record_segment
        return tokens

    changed_tokens = tokenized(changed_records)
    target_records = [changed_records[index] for index in target_record_indexes]
    target_shapes = tokenized(target_records)
    if not any(isinstance(token, str) and token for token in target_shapes):
        return "", 0, 0
    anchor_shape = json.dumps(target_shapes, separators=(",", ":"))
    if target_records[0][0] != start_line:
        return anchor_shape, 0, 0

    target_record_index = target_record_indexes[0]
    target_start_index = target_record_index
    for index in range(1, target_record_index + 1):
        if changed_records[index][1] != changed_records[index - 1][1]:
            target_start_index += 1

    # Count exact normalized range-shape occurrences using KMP so a multi-line
    # candidate is not renumbered by an earlier range that only shares its first
    # line or crosses a hunk/context boundary. This stays linear even for
    # generated patches with repeated anchors.
    prefix = [0] * len(target_shapes)
    matched = 0
    for index in range(1, len(target_shapes)):
        while matched and target_shapes[index] != target_shapes[matched]:
            matched = prefix[matched - 1]
        if target_shapes[index] == target_shapes[matched]:
            matched += 1
            prefix[index] = matched

    anchor_ordinal = 0
    anchor_occurrence_count = 0
    matched = 0
    for index, line_shape in enumerate(changed_tokens):
        while matched and line_shape != target_shapes[matched]:
            matched = prefix[matched - 1]
        if line_shape == target_shapes[matched]:
            matched += 1
        if matched == len(target_shapes):
            occurrence_start = index - len(target_shapes) + 1
            anchor_occurrence_count += 1
            if occurrence_start <= target_start_index:
                anchor_ordinal += 1
            matched = prefix[matched - 1]
    return anchor_shape, anchor_ordinal, anchor_occurrence_count


def _changed_anchor_identity(
    patch: str,
    start_line: int,
    end_line: int,
    side: str = "new",
) -> tuple[str, int]:
    """Derive the range shape and equal-shape start ordinal in one patch traversal."""
    anchor_shape, anchor_ordinal, _ = _changed_anchor_identity_details(
        patch, start_line, end_line, side
    )
    return anchor_shape, anchor_ordinal


def _changed_anchor_shape(patch: str, start_line: int, end_line: int, side: str = "new") -> str:
    return _changed_anchor_identity(patch, start_line, end_line, side)[0]


def _changed_anchor_ordinal(patch: str, start_line: int, side: str = "new") -> int:
    """Return the ordinal among equal-shaped anchors within one file lineage."""
    return _changed_anchor_identity(patch, start_line, start_line, side)[1]


def _trusted_file_lineage(diff_file) -> Optional[str]:
    """Use the base-side path as stable rename lineage when the provider supplies it."""
    old_path = safe_repo_path(getattr(diff_file, "old_filename", None))
    current_path = safe_repo_path(getattr(diff_file, "filename", None))
    path = old_path or current_path
    return f"file:{path}" if path else None


def _base_fetch_path(path: str, diff_file) -> str:
    """Return the base-side path while keeping evidence keyed to the current path."""
    if getattr(diff_file, "edit_type", None) == EDIT_TYPE.RENAMED:
        return safe_repo_path(getattr(diff_file, "old_filename", None)) or path
    return path


def _trusted_side_line_count(diff_file, side: str) -> Optional[int]:
    """Return a complete provider-supplied side's line count, if available."""
    if diff_file is None:
        return None
    content_attribute = "head_file" if side == "new" else "base_file"
    complete_attribute = f"{content_attribute}_is_complete"
    if not hasattr(diff_file, complete_attribute) or not getattr(diff_file, complete_attribute):
        return None
    content = getattr(diff_file, content_attribute, "") or ""
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return len(split_git_file_lines(str(content)))


def _bounded_sensitive_audit_specs(
    diff_files: list,
    normalized_globs: list[str],
    max_sensitive_candidates: Optional[int],
) -> tuple[list[tuple], int]:
    """Select sensitive ranges deterministically without constructing an unbounded payload."""
    limit = None if max_sensitive_candidates is None else max(0, int(max_sensitive_candidates))
    selected = []
    total_count = 0
    for diff_file in diff_files or []:
        relevant_file = safe_repo_path(getattr(diff_file, "filename", ""))
        old_file = safe_repo_path(getattr(diff_file, "old_filename", ""))
        sensitive_paths = {path for path in (relevant_file, old_file) if path}
        matched_glob_index = next((
            index
            for index, pattern in enumerate(normalized_globs)
            if any(fnmatch.fnmatch(path, pattern) for path in sensitive_paths)
        ), None)
        if not relevant_file or matched_glob_index is None:
            continue
        patch = getattr(diff_file, "patch", "")
        patch_digest = hashlib.sha256(str(patch or "").encode("utf-8")).hexdigest()
        lineage_path = old_file or relevant_file
        for anchor_index, (side, start_line, end_line, changed_line_ranges) in enumerate(
            _sensitive_change_anchors(patch), start=1
        ):
            total_count += 1
            # Earlier configured globs are the explicit risk order. A removed
            # guard is prioritized before added code within the same risk tier.
            # All remaining tie-breakers come from repository evidence, not
            # provider iteration order.
            priority = (
                matched_glob_index,
                0 if side == "old" else 1,
                lineage_path.casefold(),
                lineage_path,
                start_line,
                end_line,
                relevant_file.casefold(),
                relevant_file,
                patch_digest,
            )
            if limit == 0:
                continue
            selected.append((
                priority,
                diff_file,
                relevant_file,
                patch,
                anchor_index,
                side,
                start_line,
                end_line,
                changed_line_ranges,
            ))
            selected.sort(key=lambda item: item[0])
            if limit is not None and len(selected) > limit:
                selected.pop()
    return selected, total_count


def prepare_candidates(review_data: dict, diff_files: list, sensitive_globs: list,
                       max_candidates: int,
                       max_sensitive_candidates: Optional[int] = None) -> tuple[list[dict], list[dict]]:
    """Normalize model candidates, deduplicate them, and add deterministic sensitive-file audits."""
    raw_candidates = (review_data.get("review") or {}).get("key_issues_to_review") or []
    candidates = []
    rejected = []
    seen = set()
    diff_by_file = {
        path: diff_file
        for diff_file in (diff_files or [])
        if (path := safe_repo_path(getattr(diff_file, "filename", "")))
    }
    normalized_globs = [pattern.strip() for pattern in sensitive_globs if isinstance(pattern, str) and pattern.strip()]
    sensitive_specs, sensitive_total_count = _bounded_sensitive_audit_specs(
        diff_files,
        normalized_globs,
        max_sensitive_candidates,
    )
    for sensitive_count, spec in enumerate(sensitive_specs, start=1):
        (
            _, diff_file, relevant_file, patch, anchor_index, side,
            start_line, end_line, changed_line_ranges,
        ) = spec
        trusted_side_line_count = _trusted_side_line_count(diff_file, side)
        if trusted_side_line_count is not None and not (
            1 <= start_line <= end_line <= trusted_side_line_count
        ):
            rejected.append({
                "candidate_id": f"sensitive-{sensitive_count}",
                "reason": "invalid_candidate",
                "sensitive_path": True,
            })
            continue
        anchor_shape, anchor_ordinal, anchor_occurrence_count = _changed_anchor_identity_details(
            patch, start_line, end_line, side
        )
        candidates.append({
            "candidate_id": f"sensitive-{sensitive_count}",
            "candidate_type": "sensitive_path_audit",
            "relevant_file": relevant_file,
            "issue_header": "Sensitive path audit",
            "issue_content": "Independently inspect this configured sensitive path change for introduced defects.",
            "trigger": "A configured sensitive path range changed.",
            "impact": "A high-risk regression could otherwise be suppressed before verification.",
            "root_cause": f"sensitive audit for {relevant_file} change {anchor_index}",
            "start_line": start_line,
            "end_line": end_line,
            "side": side,
            "context_files": [relevant_file],
            "context_symbols": [],
            "sensitive_path": True,
            "_changed_line_ranges": changed_line_ranges,
            "_changed_anchor_shape": anchor_shape,
            "_changed_anchor_ordinal": anchor_ordinal,
            "_changed_anchor_occurrence_count": anchor_occurrence_count,
            "_trusted_lineage_key": _trusted_file_lineage(diff_file),
            "_trusted_patch_is_complete": getattr(diff_file, "patch_is_complete", False) is True,
            "_trusted_side_line_count": trusted_side_line_count,
            "_display_file": (
                safe_repo_path(getattr(diff_file, "old_filename", None))
                if side == "old" else relevant_file
            ) or relevant_file,
        })
    sensitive_omitted_count = sensitive_total_count - len(sensitive_specs)
    if sensitive_omitted_count:
        rejected.append({
            "candidate_id": "sensitive-overflow",
            "reason": "sensitive_audit_budget_exhausted",
            "sensitive_path": True,
            "total_count": sensitive_total_count,
            "selected_count": len(sensitive_specs),
            "omitted_count": sensitive_omitted_count,
        })

    model_candidate_count = 0
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            rejected.append({"candidate_id": f"candidate-{index + 1}", "reason": "invalid_candidate"})
            continue
        candidate = dict(raw_candidate)
        for key in list(candidate):
            if str(key).startswith("_"):
                candidate.pop(key, None)
        candidate["candidate_id"] = f"candidate-{index + 1}"
        candidate["candidate_type"] = "model_finding"
        candidate["sensitive_path"] = False
        proposed_side_value = candidate.get("side", "new")
        proposed_side = (
            proposed_side_value.strip().lower()
            if isinstance(proposed_side_value, str) else ""
        )
        start_line = candidate.get("start_line")
        end_line = candidate.get("end_line")
        required_text = ("issue_header", "issue_content", "trigger", "impact", "root_cause")
        required_text_valid = all(
            isinstance(candidate.get(key), str) and candidate[key].strip()
            for key in required_text
        )
        context_files_present = "context_files" in candidate
        context_symbols_present = "context_symbols" in candidate
        context_files = candidate.get("context_files")
        context_symbols = candidate.get("context_symbols")
        normalized_context_files = (
            [safe_repo_path(value) for value in context_files]
            if isinstance(context_files, list) else []
        )
        context_files_valid = (
            context_files_present
            and isinstance(context_files, list)
            and all(normalized_context_files)
        )
        context_symbols_valid = (
            context_symbols_present
            and isinstance(context_symbols, list)
            and len(context_symbols) <= _MAX_CONTEXT_SYMBOLS_PER_CANDIDATE
            and all(
                isinstance(value, str)
                and value.strip()
                and len(value.strip()) <= _MAX_CONTEXT_SYMBOL_CHARACTERS
                for value in context_symbols
            )
        )
        candidate["side"] = "new"
        candidate["relevant_file"] = safe_repo_path(candidate.get("relevant_file"))
        candidate["start_line"] = start_line
        candidate["end_line"] = end_line
        candidate["context_files"] = normalized_context_files
        candidate["context_symbols"] = [
            value.strip() for value in context_symbols
        ] if context_symbols_valid else []
        diff_file = diff_by_file.get(candidate["relevant_file"])
        changed_line_ranges = (
            _changed_range_containing_line(
                getattr(diff_file, "patch", ""),
                "new",
                start_line,
            )
            if diff_file and isinstance(start_line, int) and not isinstance(start_line, bool)
            else []
        )
        trusted_side_line_count = _trusted_side_line_count(diff_file, "new")
        if (proposed_side != "new" or not isinstance(start_line, int) or
                isinstance(start_line, bool) or not isinstance(end_line, int) or
                isinstance(end_line, bool) or not required_text_valid or
                not context_files_valid or not context_symbols_valid or
                not candidate["relevant_file"] or not diff_file or
                not _line_is_changed(candidate["start_line"], changed_line_ranges) or
                candidate["end_line"] < candidate["start_line"] or
                (trusted_side_line_count is not None
                 and candidate["end_line"] > trusted_side_line_count)):
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "invalid_candidate"})
            continue
        key = _candidate_key(candidate)
        if key in seen:
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "duplicate_candidate"})
            continue
        if model_candidate_count >= max_candidates:
            rejected.append({
                "candidate_id": candidate["candidate_id"],
                "reason": "candidate_budget_exhausted",
            })
            continue
        seen.add(key)
        candidate["_changed_line_ranges"] = changed_line_ranges
        (
            candidate["_changed_anchor_shape"],
            candidate["_changed_anchor_ordinal"],
            candidate["_changed_anchor_occurrence_count"],
        ) = _changed_anchor_identity_details(
            getattr(diff_file, "patch", ""),
            candidate["start_line"],
            candidate["end_line"],
        )
        candidate["_trusted_lineage_key"] = _trusted_file_lineage(diff_file)
        candidate["_trusted_patch_is_complete"] = getattr(diff_file, "patch_is_complete", False) is True
        candidate["_trusted_side_line_count"] = trusted_side_line_count
        candidates.append(candidate)
        model_candidate_count += 1

    # A changed location is not itself a root cause: one operation can expose
    # multiple independently verified defects. Assign a per-run occurrence
    # within the exact trusted anchor so those findings do not collide. The
    # occurrence is deliberately independent of rewordable model fields, which
    # are never part of either identity digest. Stable semantic reconciliation
    # across arbitrary candidate reordering requires persisted prior-finding
    # state and belongs to the downstream thread-publication layer.
    candidates_by_anchor = {}
    for candidate in candidates:
        anchor_key = (
            candidate.get("_trusted_lineage_key"),
            candidate.get("side", "new"),
            candidate.get("_changed_anchor_shape"),
            candidate.get("_changed_anchor_ordinal"),
        )
        candidates_by_anchor.setdefault(anchor_key, []).append(candidate)
    for anchor_candidates in candidates_by_anchor.values():
        same_anchor_candidate_count = len(anchor_candidates)
        for defect_ordinal, candidate in enumerate(anchor_candidates, start=1):
            candidate["_trusted_defect_ordinal"] = defect_ordinal
            candidate["_trusted_same_anchor_candidate_count"] = same_anchor_candidate_count
    return candidates, rejected


def validated_specialist_prioritization(
    batch_result: Optional[SpecialistBatchResult],
    specialist_input: Optional[SpecialistInput],
) -> Optional[Mapping]:
    """Return only the validated diff-prioritization output for the exact shared input.

    Successful and cached role records have already crossed the specialist output
    validator. Other states can contain rejected or incomplete output and must not
    influence candidate verification.
    """
    if (
        batch_result is None
        or specialist_input is None
        or batch_result.stale
        or batch_result.input_hash != specialist_input.input_hash
        or batch_result.snapshot_id != specialist_input.snapshot_id
        or batch_result.head_sha != specialist_input.head_sha
    ):
        return None
    for record in batch_result.records:
        if (
            record.role is SpecialistRole.DIFF_PRIORITIZATION
            and record.state in (SpecialistState.SUCCESS, SpecialistState.CACHED)
            and isinstance(record.output, Mapping)
            and isinstance(record.output.get("ranked_hunks"), (list, tuple))
            and isinstance(record.output.get("context_requests"), (list, tuple))
        ):
            return record.output
    return None


def _candidate_hunk_id(candidate: dict, specialist_input: SpecialistInput) -> Optional[str]:
    try:
        line = int(candidate.get("start_line"))
    except (TypeError, ValueError):
        return None
    path = candidate.get("relevant_file")
    for hunk in specialist_input.hunks:
        if hunk.path == path and hunk.start_line <= line <= hunk.end_line:
            return hunk.hunk_id
    return None


def _append_context_hint(candidate: dict, kind: str, target: str) -> bool:
    target = str(target or "").strip()
    if not target or len(target) > 256 or "\x00" in target or "\n" in target or "\r" in target:
        return False
    path = safe_repo_path(target)
    path_name = PurePosixPath(path).name if path else ""
    if kind != "symbol" and path and not any(character.isspace() for character in path) and (
        "/" in path or "." in path_name
    ):
        key = "context_files"
        value = path
    else:
        key = "context_symbols"
        value = target
    current = candidate.get(key) or []
    if isinstance(current, str):
        current = [current]
    elif not isinstance(current, list):
        current = []
    if value in current:
        return False
    if len(current) >= _MAX_CONTEXT_SYMBOLS_PER_CANDIDATE:
        return False
    candidate[key] = [*current, value]
    if key == "context_files":
        optional = candidate.get("_specialist_optional_context_files") or []
        candidate["_specialist_optional_context_files"] = [*optional, value]
    else:
        optional = candidate.get("_specialist_optional_context_symbols") or []
        candidate["_specialist_optional_context_symbols"] = [*optional, value]
    return True


def apply_specialist_prioritization(
    candidates: list[dict],
    prioritization: Mapping,
    specialist_input: SpecialistInput,
) -> tuple[list[dict], dict]:
    """Apply validated hunk ranks and anchored context requests without dropping candidates."""
    prepared = [dict(candidate) for candidate in candidates]
    candidate_hunks = {
        candidate["candidate_id"]: _candidate_hunk_id(candidate, specialist_input)
        for candidate in prepared
    }
    ranks = {
        str(item.get("hunk_id")): int(item["rank"])
        for item in prioritization.get("ranked_hunks", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("rank"), int)
        and not isinstance(item.get("rank"), bool)
    }
    original_order = {candidate["candidate_id"]: index for index, candidate in enumerate(prepared)}
    prepared.sort(key=lambda candidate: (
        0 if candidate.get("sensitive_path") else 1,
        ranks.get(candidate_hunks.get(candidate["candidate_id"]), float("inf")),
        original_order[candidate["candidate_id"]],
    ))

    context_hints_added = 0
    matched_requests = 0
    for request in prioritization.get("context_requests", ()):
        if not isinstance(request, Mapping):
            continue
        anchor_hunk_id = str(request.get("anchor_hunk_id") or "")
        anchor_path = request.get("anchor_path")
        matched = False
        for candidate in prepared:
            if (
                candidate.get("relevant_file") != anchor_path
                or candidate_hunks.get(candidate["candidate_id"]) != anchor_hunk_id
            ):
                continue
            matched = True
            if _append_context_hint(candidate, str(request.get("kind") or ""), request.get("target")):
                context_hints_added += 1
        if matched:
            matched_requests += 1

    return prepared, {
        "status": "applied",
        "ranked_candidate_count": sum(
            1 for hunk_id in candidate_hunks.values() if hunk_id in ranks
        ),
        "context_request_count": len(prioritization.get("context_requests", ())),
        "matched_context_request_count": matched_requests,
        "context_hints_added": context_hints_added,
    }


def _requested_path_specs(candidate: dict) -> list[dict]:
    relevant_file = candidate.get("relevant_file")
    paths = [(relevant_file, False)]
    optional_context = {
        path for value in (candidate.get("_specialist_optional_context_files") or [])
        if (path := safe_repo_path(value))
    }
    context_files = candidate.get("context_files") or []
    if isinstance(context_files, str):
        context_files = [context_files]
    if isinstance(context_files, list):
        paths.extend(
            (value, value != relevant_file and safe_repo_path(value) not in optional_context)
            for value in context_files
        )
    normalized = []
    seen = set()
    for value, required in paths:
        path = safe_repo_path(value)
        if path:
            if path not in seen:
                normalized.append({"path": path, "required": required})
                seen.add(path)
        elif required:
            normalized.append({"path": None, "required": True, "status": "unsafe_path"})
    return normalized


def _excerpt_anchor_line(content: str, candidate: dict, path: str) -> int:
    lines = split_git_file_lines(content)
    if not lines:
        return 0
    center = None
    if path == candidate.get("relevant_file"):
        center = candidate.get("start_line")
    if center is None:
        symbols = candidate.get("context_symbols") or []
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols if isinstance(symbols, list) else []:
            symbol = str(symbol).strip()
            if not symbol:
                continue
            for index, line in enumerate(lines, start=1):
                if symbol in line:
                    center = index
                    break
            if center is not None:
                break
    return max(1, min(len(lines), int(center or 1)))


def _requested_context_symbols(candidate: dict) -> list[str]:
    """Return unique, ordered symbol hints from a validated candidate."""
    symbols = candidate.get("context_symbols") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    normalized = []
    seen = set()
    for value in symbols if isinstance(symbols, list) else []:
        symbol = str(value).strip()
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def _required_context_symbols(candidate: dict) -> list[str]:
    """Return model-requested symbols, excluding optional specialist hints."""
    optional = {
        str(value).strip()
        for value in (candidate.get("_specialist_optional_context_symbols") or [])
        if isinstance(value, str) and value.strip()
    }
    return [
        symbol for symbol in _requested_context_symbols(candidate)
        if symbol not in optional
    ]


def _context_symbol_anchor_groups(
    content: str,
    candidate: dict,
    path: str,
    *,
    symbols: Optional[list[str]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    content_lines: Optional[list[str]] = None,
) -> list[tuple[int, list[str]]]:
    """Return one ordered anchor per line while scanning source at most once."""
    lines = content_lines if content_lines is not None else split_git_file_lines(content)
    if not lines:
        return []
    requested = list(symbols if symbols is not None else _requested_context_symbols(candidate))
    if len(requested) > _MAX_CONTEXT_SYMBOLS_PER_CANDIDATE:
        return []
    unmatched = dict.fromkeys(requested)
    groups: dict[int, list[str]] = {}
    if path == candidate.get("relevant_file"):
        try:
            candidate_anchor = int(candidate.get("start_line"))
        except (TypeError, ValueError):
            candidate_anchor = 1
        groups[max(1, min(len(lines), candidate_anchor))] = []
    for index, line in enumerate(lines, start=1):
        if stop_requested is not None and stop_requested():
            return []
        for symbol in tuple(unmatched):
            if symbol in line:
                groups.setdefault(index, []).append(symbol)
                unmatched.pop(symbol, None)
        if not unmatched:
            break
    if not groups:
        return [(1, [])]
    return [(line, groups[line]) for line in sorted(groups)]


def _coalesce_context_anchor_groups(
    anchor_groups: list[tuple[int, list[str]]],
    line_budget: int,
    line_count: int,
) -> list[tuple[int, int, list[str]]]:
    """Merge nearby anchors until their actual bounded excerpts cannot overlap."""
    groups = [(line, line, list(symbols)) for line, symbols in anchor_groups]
    while len(groups) > 1:
        line_limits = _balanced_context_line_limits(groups, line_budget)
        if not line_limits:
            break
        selected_ranges = [
            _excerpt_bounds(
                line_count,
                (start_line + end_line + 1) // 2,
                line_limit,
            )
            for (start_line, end_line, _), line_limit in zip(
                groups, line_limits, strict=True
            )
        ]
        merged = []
        changed = False
        for index, (start_line, end_line, symbols) in enumerate(groups):
            overlaps_previous = bool(
                merged
                and selected_ranges[index][0] <= selected_ranges[index - 1][1]
            )
            if overlaps_previous:
                previous_start, _, previous_symbols = merged[-1]
                merged[-1] = (
                    previous_start,
                    end_line,
                    [*previous_symbols, *symbols],
                )
                changed = True
            else:
                merged.append((start_line, end_line, symbols))
        groups = merged
        if not changed:
            break
    return groups


def _preserve_relevant_candidate_range(
    groups: list[tuple[int, int, list[str]]],
    candidate: dict,
    path: str,
    line_count: int,
) -> list[tuple[int, int, list[str]]]:
    """Make the full changed candidate range atomic for clipping and reuse."""
    if path != candidate.get("relevant_file"):
        return groups
    candidate_start = candidate.get("start_line")
    candidate_end = candidate.get("end_line")
    if (
        not isinstance(candidate_start, int)
        or isinstance(candidate_start, bool)
        or not isinstance(candidate_end, int)
        or isinstance(candidate_end, bool)
        or candidate_start < 1
        or candidate_end < candidate_start
        or candidate_end > line_count
    ):
        return groups

    ordered = sorted([
        *groups,
        (candidate_start, candidate_end, []),
    ])
    merged: list[tuple[int, int, list[str]]] = []
    for start_line, end_line, symbols in ordered:
        if merged and start_line <= merged[-1][1]:
            previous_start, previous_end, previous_symbols = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_end, end_line),
                [*previous_symbols, *symbols],
            )
        else:
            merged.append((start_line, end_line, list(symbols)))
    return merged


def _balanced_context_line_limits(
    groups: list[tuple[int, int, list[str]]],
    line_budget: int,
) -> list[int]:
    minimum_lines = [end_line - start_line + 1 for start_line, end_line, _ in groups]
    if sum(minimum_lines) > line_budget:
        return []
    remaining = line_budget
    limits = []
    for index, minimum in enumerate(minimum_lines):
        groups_left = len(groups) - index
        later_minimum = sum(minimum_lines[index + 1:])
        line_limit = max(minimum, remaining // groups_left)
        line_limit = min(line_limit, remaining - later_minimum)
        limits.append(line_limit)
        remaining -= line_limit
    return limits


def _excerpt_bounds(line_count: int, center: int, max_lines: int) -> tuple[int, int]:
    before = max_lines // 2
    start = max(1, center - before)
    end = min(line_count, start + max_lines - 1)
    start = max(1, end - max_lines + 1)
    return start, end


def _select_excerpt(
    content: str,
    candidate: dict,
    path: str,
    max_lines: int,
) -> tuple[str, int, int]:
    lines = split_git_file_lines(content)
    if not lines or max_lines < 1:
        return "", 0, 0
    center = _excerpt_anchor_line(content, candidate, path)
    start, end = _excerpt_bounds(len(lines), center, max_lines)
    return "\n".join(lines[start - 1:end]), start, end


def _matching_static_evidence(candidate: dict, static_evidence: Any) -> list[dict]:
    matches = []
    for item in static_evidence if isinstance(static_evidence, (list, tuple)) else []:
        if not isinstance(item, Mapping):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        path = safe_repo_path(item.get("path"))
        content = str(item.get("content") or "").strip()
        if not content or (candidate_id and candidate_id != candidate["candidate_id"]):
            continue
        if not candidate_id and path != candidate.get("relevant_file"):
            continue
        match = dict(item)
        match.update({"path": path or "", "content": content})
        match.setdefault("source", "static_analyzer")
        matches.append(match)
    return matches


def _changed_patch_range_evidence(
    diff_file,
    side: str,
    start_line: int,
    end_line: int,
    source: str,
    *,
    max_lines: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_budget_exhausted: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Return a complete side-specific patch range with at least one changed line."""
    if diff_file is None:
        return None
    range_line_count = end_line - start_line + 1
    if start_line < 1 or range_line_count < 1:
        return None
    if max_lines is not None and range_line_count > max(0, int(max_lines)):
        # This exact range cannot fit the prompt-visible evidence budget.  Fail
        # before walking the patch or materializing any range-sized lists.
        if on_budget_exhausted is not None:
            on_budget_exhausted()
        return None
    if max_tokens is not None and int(max_tokens) < 1:
        if on_budget_exhausted is not None:
            on_budget_exhausted()
        return None
    count_tokens = token_counter or (lambda value: len(str(value).encode("utf-8")))
    newline_tokens = count_tokens("\n") if max_tokens is not None else 0
    selected_tokens = 0
    old_line = None
    new_line = None
    selected = []
    selected_lines = []
    start_line_is_changed = False
    for record in iter_git_patch_lines(getattr(diff_file, "patch", "") or ""):
        if stop_requested is not None and stop_requested():
            return None
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            new_line = int(header.group(3))
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if side == "new" and not line.startswith("-") and start_line <= new_line <= end_line:
            content_line = line[1:] if line.startswith(("+", " ")) else line
            if max_tokens is not None:
                next_tokens = count_tokens(content_line) + (newline_tokens if selected else 0)
                if selected_tokens + next_tokens > int(max_tokens):
                    if on_budget_exhausted is not None:
                        on_budget_exhausted()
                    return None
                selected_tokens += next_tokens
            selected.append(content_line)
            selected_lines.append(new_line)
            if line.startswith("+") and new_line == start_line:
                start_line_is_changed = True
        elif side == "old" and not line.startswith("+") and start_line <= old_line <= end_line:
            content_line = line[1:] if line.startswith(("-", " ")) else line
            if max_tokens is not None:
                next_tokens = count_tokens(content_line) + (newline_tokens if selected else 0)
                if selected_tokens + next_tokens > int(max_tokens):
                    if on_budget_exhausted is not None:
                        on_budget_exhausted()
                    return None
                selected_tokens += next_tokens
            selected.append(content_line)
            selected_lines.append(old_line)
            if line.startswith("-") and old_line == start_line:
                start_line_is_changed = True
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1
    has_complete_range = (
        len(selected_lines) == range_line_count
        and all(line_number == start_line + offset
                for offset, line_number in enumerate(selected_lines))
    )
    if not has_complete_range or not start_line_is_changed or not selected:
        return None
    return {
        "source": source,
        "side": side,
        "content": "\n".join(selected),
        "start_line": min(selected_lines),
        "end_line": max(selected_lines),
        "anchor_start_line": start_line,
        "anchor_end_line": end_line,
    }


def _candidate_changed_patch_evidence(
    diff_file,
    candidate: dict,
    *,
    max_lines: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_budget_exhausted: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Return exact candidate-scoped changed lines from one trusted provider patch."""
    try:
        start_line = int(candidate.get("start_line"))
        end_line = int(candidate.get("end_line"))
    except (TypeError, ValueError):
        return None
    return _changed_patch_range_evidence(
        diff_file,
        candidate.get("side", "new"),
        start_line,
        end_line,
        "changed_patch",
        max_lines=max_lines,
        max_tokens=max_tokens,
        token_counter=token_counter,
        stop_requested=stop_requested,
        on_budget_exhausted=on_budget_exhausted,
    )


def _changed_context_patch_evidence(
    diff_file,
    stop_requested: Optional[Callable[[], bool]] = None,
    remaining_line_budget: Optional[Callable[[], int]] = None,
    remaining_token_budget: Optional[Callable[[], int]] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    collection_status: Optional[dict] = None,
) -> Iterable[dict]:
    """Yield changed ranges from one patch traversal so retrieval budgets can stop work early."""
    if diff_file is None:
        return

    old_line = None
    new_line = None
    pending = {"new": None, "old": None}
    added_patch = bool(
        getattr(diff_file, "edit_type", None) is EDIT_TYPE.ADDED
        and getattr(diff_file, "patch_is_complete", False)
    )
    added_patch_complete = added_patch
    added_expected_start = 1
    added_expected_total = 0
    added_observed_total = 0
    added_saw_hunk = False
    if collection_status is not None:
        collection_status["complete_added_file"] = False

    def completed(side: str) -> Optional[dict]:
        current = pending[side]
        if current is None:
            return None
        pending[side] = None
        return {
            "source": "changed_context_patch",
            "side": side,
            "content": "\n".join(current["content"]),
            "start_line": current["start_line"],
            "end_line": current["end_line"],
            "anchor_start_line": current["start_line"],
            "anchor_end_line": current["end_line"],
        }

    def append_changed(side: str, line_number: int, content: str) -> Optional[dict]:
        current = pending[side]
        finished = None
        if current is not None and line_number != current["end_line"] + 1:
            finished = completed(side)
            current = None
        line_budget = remaining_line_budget() if remaining_line_budget is not None else None
        token_budget = remaining_token_budget() if remaining_token_budget is not None else None
        added_tokens = 0
        if token_counter is not None:
            added_tokens = max(1, int(token_counter(content)))
            if current is not None:
                added_tokens += max(1, int(token_counter("\n")))
        if line_budget is not None and (
            line_budget < 1
            or (current is not None and len(current["content"]) >= line_budget)
        ):
            raise _ChangedContextCollectionStopped
        if token_budget is not None and (
            token_budget < 1
            or (current["token_count"] if current is not None else 0) + added_tokens > token_budget
        ):
            raise _ChangedContextCollectionStopped
        if current is None:
            pending[side] = {
                "start_line": line_number,
                "end_line": line_number,
                "content": [content],
                "token_count": added_tokens,
            }
        else:
            current["end_line"] = line_number
            current["content"].append(content)
            current["token_count"] += added_tokens
        return finished

    for record in iter_git_patch_lines(getattr(diff_file, "patch", "") or ""):
        if stop_requested is not None and stop_requested():
            raise _ChangedContextCollectionStopped
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            if added_patch:
                if added_observed_total != added_expected_total:
                    added_patch_complete = False
                old_start = int(header.group(1))
                old_count = int(header.group(2) or 1)
                next_new_start = int(header.group(3))
                new_count = int(header.group(4) or 1)
                if (
                    old_start != 0
                    or old_count != 0
                    or next_new_start != added_expected_start
                ):
                    added_patch_complete = False
                added_saw_hunk = True
                added_expected_start += new_count
                added_expected_total += new_count
                if (
                    remaining_line_budget is not None
                    and added_expected_total > remaining_line_budget()
                ):
                    raise _ChangedContextCollectionStopped
            next_old_line = int(header.group(1))
            next_new_line = int(header.group(3))
            for side, next_line in (("new", next_new_line), ("old", next_old_line)):
                current = pending[side]
                if current is not None and next_line != current["end_line"] + 1:
                    item = completed(side)
                    if item is not None:
                        yield item
            old_line = next_old_line
            new_line = next_new_line
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if added_patch:
            if line.startswith("+"):
                added_observed_total += 1
            else:
                added_patch_complete = False
        if line.startswith("+"):
            item = append_changed("new", new_line, line[1:])
            if item is not None:
                yield item
        elif line.startswith("-"):
            item = append_changed("old", old_line, line[1:])
            if item is not None:
                yield item
        else:
            for side in ("new", "old"):
                item = completed(side)
                if item is not None:
                    yield item
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1

    for side in ("new", "old"):
        item = completed(side)
        if item is not None:
            yield item
    if collection_status is not None:
        complete_added_file = bool(
            added_patch_complete
            and added_saw_hunk
            and added_observed_total == added_expected_total
            and added_observed_total > 0
        )
        collection_status["complete_added_file"] = complete_added_file
        collection_status["added_line_count"] = (
            added_observed_total if complete_added_file else None
        )


def _retrieval_evidence_id(candidate_id: str, path: str, source: str) -> str:
    payload = json.dumps(
        {"candidate_id": candidate_id, "path": path, "source": source},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


async def retrieve_evidence(git_provider, candidates: list[dict], budgets: VerificationBudgets,
                            static_evidence: Any, diff_files: Optional[list] = None,
                            token_counter: Optional[Callable[[str], int]] = None,
                            prefer_pr_head: bool = False) -> tuple[list[dict], dict]:
    """Fetch bounded repository excerpts and return them with an auditable retrieval record."""
    started = time.monotonic()
    deadline = started + max(0.0, budgets.timeout_seconds)
    evidence = []
    requests = []
    cache = {}
    unique_files = set()
    lines_by_path = {}
    total_lines = 0
    total_tokens = 0
    count_tokens = token_counter or (lambda value: len(str(value).encode("utf-8")))
    budget_exhausted = False
    time_budget_exhausted = False
    changed_evidence_count = 0
    changed_context_head_available = {}
    changed_context_patch_available = {}
    changed_context_failure_status = {}
    shared_repo_evidence = {}
    claimed_required_symbols: dict[tuple[str, str], set[str]] = {}
    diff_by_file = {
        path: diff_file
        for diff_file in (diff_files or [])
        if (path := safe_repo_path(getattr(diff_file, "filename", "")))
    }

    def remaining_lines(path: str) -> int:
        return max(0, min(
            budgets.max_lines_per_file - lines_by_path.get(path, 0),
            budgets.max_total_lines - total_lines,
        ))

    def append_evidence(item: dict, candidate: dict, path: str,
                        max_lines: Optional[int] = None,
                        max_tokens: Optional[int] = None,
                        prepared_excerpt: Optional[tuple[str, int, int]] = None) -> bool:
        nonlocal budget_exhausted, changed_evidence_count, time_budget_exhausted, total_lines, total_tokens
        if time.monotonic() >= deadline:
            budget_exhausted = True
            time_budget_exhausted = True
            return False
        content = str(item.get("content") or "")
        line_budget = remaining_lines(path)
        if max_lines is not None:
            line_budget = min(line_budget, max(0, int(max_lines)))
        token_budget = budgets.max_context_tokens - total_tokens
        if max_tokens is not None:
            token_budget = min(token_budget, max(0, int(max_tokens)))
        if line_budget < 1 or token_budget < 1:
            budget_exhausted = True
            return False
        if prepared_excerpt is not None:
            excerpt, start_line, end_line = prepared_excerpt
            if excerpt.count("\n") + 1 > line_budget:
                budget_exhausted = True
                return False
        elif item.get("source") in _PATCH_CHANGED_EVIDENCE_SOURCES:
            if content.count("\n") + 1 > line_budget:
                budget_exhausted = True
                return False
            excerpt = content
            start_line = item.get("start_line")
            end_line = item.get("end_line")
        else:
            excerpt, start_line, end_line = _select_excerpt(content, candidate, path, line_budget)
        if not excerpt:
            budget_exhausted = True
            return False
        prepared = dict(item)
        prepared.update({"path": path, "content": excerpt})
        prepared.setdefault("start_line", start_line)
        prepared.setdefault("end_line", end_line)
        if _is_atomic_prompt_evidence(prepared):
            prepared.setdefault("anchor_start_line", candidate.get("start_line"))
            prepared.setdefault("anchor_end_line", candidate.get("end_line"))
        evidence_tokens = count_tokens(excerpt)
        if evidence_tokens > token_budget:
            budget_exhausted = True
            if _is_atomic_prompt_evidence(prepared):
                best = None
                lower = 0.0
                upper = 1.0
                for _ in range(24):
                    midpoint = (lower + upper) / 2
                    bounded = bounded_verification_evidence([prepared], midpoint)
                    if bounded and count_tokens(bounded[0]["content"]) <= token_budget:
                        best = bounded[0]
                        lower = midpoint
                    else:
                        upper = midpoint
                if best is None:
                    return False
                prepared = best
            else:
                lower = 0
                upper = len(excerpt)
                while lower < upper:
                    midpoint = (lower + upper + 1) // 2
                    if count_tokens(excerpt[:midpoint]) <= token_budget:
                        lower = midpoint
                    else:
                        upper = midpoint - 1
                if lower < 1:
                    return False
                prepared["content"] = excerpt[:lower]
                prepared["content_truncated"] = True
                prepared["end_line"] = prepared["start_line"] + prepared["content"].count("\n")
            evidence_tokens = count_tokens(prepared["content"])
        if time.monotonic() > deadline:
            budget_exhausted = True
            time_budget_exhausted = True
            return False
        line_count = prepared["content"].count("\n") + 1
        evidence.append(prepared)
        lines_by_path[path] = lines_by_path.get(path, 0) + line_count
        total_lines += line_count
        total_tokens += evidence_tokens
        if prepared.get("source") in _CHANGED_EVIDENCE_SOURCES:
            changed_evidence_count += 1
        return True

    def context_symbols_for_request(
        candidate: dict,
        candidate_id: str,
        request_spec: dict,
        path: str,
    ) -> list[str]:
        symbols = (
            _required_context_symbols(candidate)
            if request_spec.get("required")
            else _requested_context_symbols(candidate)
        )
        claimed = claimed_required_symbols.setdefault((candidate_id, path), set())
        return [symbol for symbol in symbols if symbol not in claimed]

    def find_shared_evidence_group(
        candidate: dict,
        candidate_id: str,
        request_spec: dict,
        path: str,
        source: str,
        content: str,
    ) -> tuple[Optional[list[dict]], list[str], list[str], bool]:
        symbols = context_symbols_for_request(
            candidate, candidate_id, request_spec, path
        )
        best_group = None
        best_matches: list[str] = []
        best_anchor_compatible = False
        best_score = (-1, -1, -1, -1)
        for existing_group in shared_repo_evidence.get((source, path), ()):
            combined_content = "\n".join(
                str(existing.get("content") or "")
                for existing in existing_group
            )
            if not combined_content.strip():
                continue
            anchor_content = []
            reclaimable_lines = 0
            for existing in existing_group:
                existing_content = str(existing.get("content") or "")
                existing_lines = split_git_file_lines(existing_content)
                try:
                    excerpt_start = int(existing.get("start_line"))
                    anchor_start = int(existing.get("anchor_start_line"))
                    anchor_end = int(existing.get("anchor_end_line"))
                except (TypeError, ValueError):
                    continue
                relative_start = anchor_start - excerpt_start
                relative_end = anchor_end - excerpt_start
                if (
                    relative_start < 0
                    or relative_end < relative_start
                    or relative_end >= len(existing_lines)
                ):
                    continue
                durable_lines = existing_lines[relative_start:relative_end + 1]
                anchor_content.append("\n".join(durable_lines))
                reclaimable_lines += len(existing_lines) - len(durable_lines)
            durable_content = "\n".join(anchor_content)
            discoverable_symbols = [
                symbol for symbol in symbols if symbol in combined_content
            ]
            matched_symbols = [
                symbol for symbol in symbols if symbol in durable_content
            ]
            anchor_compatible = True
            if path == candidate.get("relevant_file"):
                source_lines = split_git_file_lines(content)
                candidate_start = candidate.get("start_line")
                candidate_end = candidate.get("end_line")
                anchor_compatible = False
                for existing in existing_group:
                    try:
                        visible_start = int(existing.get("start_line"))
                        visible_end = int(existing.get("end_line"))
                        preserved_start = int(existing.get("anchor_start_line"))
                        preserved_end = int(existing.get("anchor_end_line"))
                    except (TypeError, ValueError):
                        continue
                    visible_lines = split_git_file_lines(str(existing.get("content") or ""))
                    if (
                        not isinstance(candidate_start, int)
                        or isinstance(candidate_start, bool)
                        or not isinstance(candidate_end, int)
                        or isinstance(candidate_end, bool)
                        or candidate_start < 1
                        or candidate_end < candidate_start
                        or candidate_end > len(source_lines)
                        or visible_start > candidate_start
                        or candidate_end > visible_end
                        or preserved_start > candidate_start
                        or candidate_end > preserved_end
                    ):
                        continue
                    relative_start = candidate_start - visible_start
                    relative_end = candidate_end - visible_start
                    if (
                        relative_start < 0
                        or relative_end >= len(visible_lines)
                    ):
                        continue
                    if (
                        visible_lines[relative_start:relative_end + 1]
                        == source_lines[candidate_start - 1:candidate_end]
                    ):
                        anchor_compatible = True
                        break
            score = (
                int(anchor_compatible),
                len(matched_symbols),
                len(discoverable_symbols),
                reclaimable_lines,
            )
            if best_group is None or score > best_score:
                best_group = existing_group
                best_matches = matched_symbols
                best_anchor_compatible = anchor_compatible
                best_score = score
            if anchor_compatible and len(matched_symbols) == len(symbols):
                break
        # A shared excerpt may already prove every symbol that exists on this
        # path. Only expand it for additional symbols discoverable in the full
        # path content; symbols belonging to another required path must not
        # force an unrelated line-1 fallback or duplicate shared evidence.
        remaining_symbols = [
            symbol
            for symbol in symbols
            if symbol not in best_matches and symbol in content
        ]
        return best_group, best_matches, remaining_symbols, best_anchor_compatible

    def bind_shared_evidence_group(
        shared_items: list[dict],
        candidate_id: str,
        request_spec: dict,
        path: str,
        symbols: list[str],
    ) -> None:
        for shared_item in shared_items:
            candidate_ids = shared_item.get("candidate_ids")
            if not isinstance(candidate_ids, list):
                candidate_ids = [shared_item.pop("candidate_id")]
                shared_item["candidate_ids"] = candidate_ids
            if candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
            shared_item["required_evidence"] = bool(
                shared_item.get("required_evidence") or request_spec.get("required")
            )
        claimed_required_symbols.setdefault((candidate_id, path), set()).update(symbols)

    def set_request_excerpt_metadata(request: dict, shared_items: list[dict]) -> None:
        for key in ("start_line", "end_line", "excerpt_ranges"):
            request.pop(key, None)
        request["excerpt_count"] = len(shared_items)
        if len(shared_items) == 1:
            request.update({
                "start_line": shared_items[0]["start_line"],
                "end_line": shared_items[0]["end_line"],
            })
        else:
            request["excerpt_ranges"] = [
                {
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                }
                for item in shared_items
            ]

    def compact_shared_evidence_group(shared_items: list[dict], path: str) -> None:
        """Reclaim surrounding lines while retaining every shared anchor range."""
        nonlocal total_lines, total_tokens
        for item in shared_items:
            content = str(item.get("content") or "")
            lines = split_git_file_lines(content)
            try:
                excerpt_start = int(item.get("start_line"))
                anchor_start = int(item.get("anchor_start_line"))
                anchor_end = int(item.get("anchor_end_line"))
            except (TypeError, ValueError):
                continue
            relative_start = anchor_start - excerpt_start
            relative_end = anchor_end - excerpt_start
            if relative_start < 0 or relative_end < relative_start or relative_end >= len(lines):
                continue
            compacted = "\n".join(lines[relative_start:relative_end + 1])
            if not compacted or compacted == content:
                continue
            old_line_count = content.count("\n") + 1
            new_line_count = compacted.count("\n") + 1
            lines_by_path[path] = max(
                0,
                lines_by_path.get(path, 0) - (old_line_count - new_line_count),
            )
            total_lines -= old_line_count - new_line_count
            total_tokens -= count_tokens(content) - count_tokens(compacted)
            item.update({
                "content": compacted,
                "start_line": anchor_start,
                "end_line": anchor_end,
                "content_truncated": True,
            })
        evidence_id = shared_items[0].get("evidence_id") if shared_items else None
        for request in requests:
            if request.get("path") == path and request.get("evidence_id") == evidence_id:
                candidate_id = request.get("candidate_id")
                request_items = [
                    item
                    for item in shared_items
                    if candidate_id in (
                        item.get("candidate_ids")
                        if isinstance(item.get("candidate_ids"), list)
                        else [item.get("candidate_id")]
                    )
                ]
                if request_items:
                    set_request_excerpt_metadata(request, request_items)

    def append_context_excerpts(candidate: dict, candidate_id: str, context_path: str,
                                request_spec: dict, content: str, source: str,
                                evidence_id: str) -> bool:
        """Append bounded, prompt-visible excerpts around every symbol found in a file."""
        nonlocal budget_exhausted, changed_evidence_count, time_budget_exhausted
        nonlocal total_lines, total_tokens
        already_claimed = claimed_required_symbols.setdefault(
            (candidate_id, context_path), set()
        )
        symbols = context_symbols_for_request(
            candidate, candidate_id, request_spec, context_path
        )
        content_lines = split_git_file_lines(content)
        anchor_groups = _context_symbol_anchor_groups(
            content,
            candidate,
            context_path,
            symbols=symbols,
            stop_requested=lambda: time.monotonic() >= deadline,
            content_lines=content_lines,
        )
        if not anchor_groups:
            budget_exhausted = True
            if time.monotonic() >= deadline:
                time_budget_exhausted = True
            return False
        anchor_ranges = _coalesce_context_anchor_groups(
            anchor_groups,
            remaining_lines(context_path),
            len(content_lines),
        )
        anchor_ranges = _preserve_relevant_candidate_range(
            anchor_ranges,
            candidate,
            context_path,
            len(content_lines),
        )
        line_limits = _balanced_context_line_limits(
            anchor_ranges,
            remaining_lines(context_path),
        )
        if not line_limits:
            budget_exhausted = True
            return False
        evidence_start = len(evidence)
        previous_path_lines = lines_by_path.get(context_path)
        previous_total_lines = total_lines
        previous_total_tokens = total_tokens
        previous_changed_count = changed_evidence_count

        def rollback() -> None:
            nonlocal changed_evidence_count, total_lines, total_tokens
            del evidence[evidence_start:]
            if previous_path_lines is None:
                lines_by_path.pop(context_path, None)
            else:
                lines_by_path[context_path] = previous_path_lines
            total_lines = previous_total_lines
            total_tokens = previous_total_tokens
            changed_evidence_count = previous_changed_count

        for index, (anchor_start_line, anchor_end_line, _) in enumerate(anchor_ranges):
            groups_left = len(anchor_ranges) - index
            line_share = line_limits[index]
            token_share = (budgets.max_context_tokens - total_tokens) // groups_left
            if line_share < 1 or token_share < 1:
                budget_exhausted = True
                rollback()
                return False
            resolved_anchor_line = (anchor_start_line + anchor_end_line + 1) // 2
            excerpt_start, excerpt_end = _excerpt_bounds(
                len(content_lines), resolved_anchor_line, line_share
            )
            prepared_excerpt = (
                "\n".join(content_lines[excerpt_start - 1:excerpt_end]),
                excerpt_start,
                excerpt_end,
            )
            if not append_evidence({
                "candidate_id": candidate_id,
                "source": source,
                "content": str(content),
                "evidence_id": evidence_id,
                "required_evidence": bool(request_spec.get("required")),
                "anchor_start_line": anchor_start_line,
                "anchor_end_line": anchor_end_line,
            }, candidate, context_path, max_lines=line_share, max_tokens=token_share,
                prepared_excerpt=prepared_excerpt):
                rollback()
                return False
        already_claimed.update(
            symbol
            for _, _, anchor_symbols in anchor_ranges
            for symbol in anchor_symbols
        )
        return True

    def append_complete_context_head(candidate: dict, candidate_id: str, context_path: str,
                                     request_spec: dict, context_diff) -> bool:
        head_file = getattr(context_diff, "head_file", "")
        if not head_file or not getattr(context_diff, "head_file_is_complete", True):
            return False
        source = "changed_context_head"
        (
            shared_items,
            matched_symbols,
            remaining_symbols,
            anchor_compatible,
        ) = find_shared_evidence_group(
            candidate,
            candidate_id,
            request_spec,
            context_path,
            source,
            str(head_file),
        )
        if shared_items is not None:
            if matched_symbols or not remaining_symbols:
                bind_shared_evidence_group(
                    shared_items,
                    candidate_id,
                    request_spec,
                    context_path,
                    matched_symbols,
                )
            if not remaining_symbols and anchor_compatible:
                changed_context_head_available[(candidate_id, context_path)] = shared_items[0][
                    "evidence_id"
                ]
                return True
            compact_shared_evidence_group(shared_items, context_path)
        evidence_id = (
            shared_items[0]["evidence_id"]
            if shared_items is not None
            else _retrieval_evidence_id(candidate_id, context_path, source)
        )
        before = len(evidence)
        if not append_context_excerpts(
            candidate,
            candidate_id,
            context_path,
            request_spec,
            str(head_file),
            source,
            evidence_id,
        ):
            return False
        if shared_items is not None:
            shared_items.extend(evidence[before:])
        else:
            shared_repo_evidence.setdefault((source, context_path), []).append(evidence[before:])
        changed_context_head_available[(candidate_id, context_path)] = evidence_id
        return True

    # Reserve each exact candidate anchor before any larger same-file or
    # repository excerpt can consume its path budget. This keeps one candidate
    # from starving another candidate in the same file of changed-code proof.
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        candidate_line_budget = remaining_lines(relevant_file)
        try:
            candidate_range_lines = (
                int(candidate.get("end_line")) - int(candidate.get("start_line")) + 1
            )
        except (TypeError, ValueError):
            candidate_range_lines = 0
        changed_patch = None
        if candidate_range_lines < 1 or candidate_range_lines > candidate_line_budget:
            budget_exhausted = True
        elif time.monotonic() >= deadline or total_tokens >= budgets.max_context_tokens:
            budget_exhausted = True
            time_budget_exhausted = time.monotonic() >= deadline
        else:
            candidate_budget_stop = []
            changed_patch = _candidate_changed_patch_evidence(
                diff_by_file.get(relevant_file),
                candidate,
                max_lines=candidate_line_budget,
                max_tokens=budgets.max_context_tokens - total_tokens,
                token_counter=count_tokens,
                stop_requested=lambda: time.monotonic() >= deadline,
                on_budget_exhausted=(
                    lambda stops=candidate_budget_stop: stops.append(True)
                ),
            )
            if candidate_budget_stop:
                budget_exhausted = True
            if changed_patch is None and time.monotonic() >= deadline:
                budget_exhausted = True
                time_budget_exhausted = True
        if changed_patch is not None:
            append_evidence(
                {"candidate_id": candidate_id, **changed_patch},
                candidate,
                relevant_file,
            )

    # Supplemental patches for requested changed files come only after every
    # candidate has had its mandatory own-change anchor reserved.
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        for request_spec in _requested_path_specs(candidate):
            context_path = request_spec.get("path")
            if not context_path or context_path == relevant_file:
                continue
            context_diff = diff_by_file.get(context_path)
            context_key = (candidate_id, context_path)
            if context_diff is None:
                continue
            if (
                getattr(context_diff, "head_file", "")
                and getattr(context_diff, "head_file_is_complete", True)
            ):
                if not append_complete_context_head(
                    candidate, candidate_id, context_path, request_spec, context_diff
                ):
                    changed_context_failure_status[context_key] = (
                        "time_budget_exhausted"
                        if time_budget_exhausted
                        else "context_budget_exhausted"
                    )
                continue
            context_item_count = 0
            context_collection_status = {}
            complete_added_patch_evidence_id = _retrieval_evidence_id(
                candidate_id, context_path, "changed_context_patch"
            )
            complete_added_patch_seen = False
            complete_added_patch_end_line = None
            try:
                context_items = _changed_context_patch_evidence(
                    context_diff,
                    stop_requested=lambda path=context_path: (
                        time.monotonic() >= deadline
                        or remaining_lines(path) < 1
                        or total_tokens >= budgets.max_context_tokens
                    ),
                    remaining_line_budget=lambda path=context_path: remaining_lines(path),
                    remaining_token_budget=lambda: budgets.max_context_tokens - total_tokens,
                    token_counter=count_tokens,
                    collection_status=context_collection_status,
                )
                appended_all = True
                for context_item in context_items:
                    context_item_count += 1
                    context_item["evidence_id"] = complete_added_patch_evidence_id
                    context_item["required_evidence"] = bool(request_spec.get("required"))
                    if not append_evidence(
                        {"candidate_id": candidate_id, **context_item},
                        candidate,
                        context_path,
                    ):
                        appended_all = False
                        break
                    complete_added_patch_seen = bool(
                        context_item_count == 1
                        and context_item.get("side") == "new"
                        and context_item.get("start_line") == 1
                    )
                    if complete_added_patch_seen:
                        complete_added_patch_end_line = context_item.get("end_line")
            except _ChangedContextCollectionStopped:
                budget_exhausted = True
                if time.monotonic() >= deadline:
                    time_budget_exhausted = True
                appended_all = False
            if context_item_count == 0 and appended_all:
                appended_all = append_complete_context_head(
                    candidate, candidate_id, context_path, request_spec, context_diff
                )
            elif (
                appended_all
                and context_item_count == 1
                and complete_added_patch_seen
                and context_collection_status.get("complete_added_file")
                and complete_added_patch_end_line
                == context_collection_status.get("added_line_count")
            ):
                changed_context_patch_available[context_key] = complete_added_patch_evidence_id
            if not appended_all:
                changed_context_failure_status[context_key] = (
                    "time_budget_exhausted"
                    if time_budget_exhausted
                    else "context_budget_exhausted"
                )

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        diff_file = diff_by_file.get(candidate.get("relevant_file"))
        head_file = getattr(diff_file, "head_file", "") if diff_file is not None else ""
        changed_head_available = False
        changed_head_failure_status = None
        changed_head_evidence_id = _retrieval_evidence_id(
            candidate_id, relevant_file, "changed_head"
        )
        for static_index, static_item in enumerate(_matching_static_evidence(candidate, static_evidence)):
            evidence_item = {"candidate_id": candidate_id, **static_item}
            static_path = static_item.get("path") or f"@static/{candidate_id}/{static_index + 1}"
            append_evidence(evidence_item, candidate, static_path)
        if (candidate.get("side", "new") == "new" and head_file
                and getattr(diff_file, "head_file_is_complete", True)):
            head_request = {"required": False}
            (
                shared_items,
                matched_symbols,
                remaining_symbols,
                anchor_compatible,
            ) = find_shared_evidence_group(
                candidate,
                candidate_id,
                head_request,
                relevant_file,
                "changed_head",
                str(head_file),
            )
            if shared_items is not None:
                if matched_symbols or not remaining_symbols:
                    bind_shared_evidence_group(
                        shared_items,
                        candidate_id,
                        head_request,
                        relevant_file,
                        matched_symbols,
                    )
                changed_head_evidence_id = shared_items[0]["evidence_id"]
                changed_head_available = not remaining_symbols and anchor_compatible
                if remaining_symbols or not anchor_compatible:
                    compact_shared_evidence_group(shared_items, relevant_file)
            if not changed_head_available:
                if shared_items is None:
                    changed_head_evidence_id = _retrieval_evidence_id(
                        candidate_id, relevant_file, "changed_head"
                    )
                before = len(evidence)
                changed_head_available = append_context_excerpts(
                    candidate,
                    candidate_id,
                    relevant_file,
                    head_request,
                    str(head_file),
                    "changed_head",
                    changed_head_evidence_id,
                )
                if changed_head_available:
                    if shared_items is not None:
                        shared_items.extend(evidence[before:])
                    else:
                        shared_repo_evidence.setdefault(
                            ("changed_head", relevant_file), []
                        ).append(evidence[before:])
                else:
                    changed_head_failure_status = (
                        "time_budget_exhausted"
                        if time_budget_exhausted
                        else "context_budget_exhausted"
                    )
        for request_spec in _requested_path_specs(candidate):
            request = {"candidate_id": candidate_id, **request_spec}
            if request.get("status") == "unsafe_path":
                requests.append(request)
                continue
            path = request["path"]
            request["status"] = "pending"
            requests.append(request)
            if path == relevant_file and changed_head_available:
                request["status"] = "satisfied_by_changed_head"
                request["source"] = "changed_head"
                request["evidence_id"] = changed_head_evidence_id
                continue
            if path == relevant_file and changed_head_failure_status:
                # A complete diff head is the authoritative version of a
                # modified candidate file. If its required evidence cannot fit,
                # substituting the base repository file can make stale symbols
                # appear current and incorrectly complete the request.
                request["status"] = changed_head_failure_status
                budget_exhausted = True
                continue
            context_key = (candidate_id, path)
            if context_key in changed_context_head_available:
                request["status"] = "satisfied_by_changed_head"
                request["source"] = "changed_context_head"
                request["evidence_id"] = changed_context_head_available[context_key]
                continue
            if context_key in changed_context_patch_available:
                request["status"] = "satisfied_by_changed_patch"
                request["source"] = "changed_context_patch"
                request["evidence_id"] = changed_context_patch_available[context_key]
                continue
            if context_key in changed_context_failure_status:
                request["status"] = changed_context_failure_status[context_key]
                budget_exhausted = True
                continue
            source = "pr_head_file" if prefer_pr_head else "repository_file"
            fetch_path = path if prefer_pr_head else _base_fetch_path(path, diff_by_file.get(path))
            cache_key = ("pr_head" if prefer_pr_head else "base", fetch_path)
            cached_content = cache.get(cache_key)
            if isinstance(cached_content, bytes):
                cached_content = cached_content.decode("utf-8", errors="replace")
            (
                shared_items,
                matched_symbols,
                remaining_symbols,
                anchor_compatible,
            ) = find_shared_evidence_group(
                candidate,
                candidate_id,
                request,
                path,
                source,
                str(cached_content or ""),
            )
            if shared_items is not None:
                if matched_symbols or not remaining_symbols:
                    bind_shared_evidence_group(
                        shared_items,
                        candidate_id,
                        request,
                        path,
                        matched_symbols,
                    )
                if not remaining_symbols and anchor_compatible:
                    request.update({
                        "status": "retrieved",
                        "source": source,
                        "evidence_id": shared_items[0]["evidence_id"],
                    })
                    set_request_excerpt_metadata(request, shared_items)
                    continue
                if remaining_symbols or not anchor_compatible:
                    compact_shared_evidence_group(shared_items, path)
            if path not in unique_files and len(unique_files) >= budgets.max_files:
                request["status"] = "file_budget_exhausted"
                budget_exhausted = True
                continue
            if remaining_lines(path) < 1 or total_tokens >= budgets.max_context_tokens:
                request["status"] = "context_budget_exhausted"
                budget_exhausted = True
                continue
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                request["status"] = "time_budget_exhausted"
                budget_exhausted = True
                continue
            unique_files.add(path)
            if cache_key not in cache:
                try:
                    cache[cache_key] = await _bounded_repo_file_fetch(
                        git_provider, fetch_path, remaining_time, from_pr_head=prefer_pr_head
                    )
                except asyncio.TimeoutError:
                    request["status"] = "time_budget_exhausted"
                    budget_exhausted = True
                    continue
                except _RepositoryFetchCapacityExhausted:
                    request["status"] = "fetch_capacity_exhausted"
                    budget_exhausted = True
                    continue
                except Exception as exc:
                    cache[cache_key] = None
                    request["status"] = "fetch_failed"
                    request["error"] = type(exc).__name__
                    continue
            content = cache[cache_key]
            if not content:
                request["status"] = "missing"
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            evidence_id = (
                shared_items[0]["evidence_id"]
                if shared_items is not None
                else _retrieval_evidence_id(candidate_id, path, source)
            )
            before = len(evidence)
            if not append_context_excerpts(
                candidate,
                candidate_id,
                path,
                request,
                str(content),
                source,
                evidence_id,
            ):
                request["status"] = (
                    "time_budget_exhausted" if time_budget_exhausted else "context_budget_exhausted"
                )
                continue
            appended_items = evidence[before:]
            if shared_items is not None:
                shared_items.extend(appended_items)
                request_items = [
                    item
                    for item in shared_items
                    if candidate_id in (
                        item.get("candidate_ids")
                        if isinstance(item.get("candidate_ids"), list)
                        else [item.get("candidate_id")]
                    )
                ]
            else:
                shared_repo_evidence.setdefault((source, path), []).append(appended_items)
                request_items = appended_items
            request.update({
                "status": "retrieved",
                "source": source,
                "evidence_id": evidence_id,
            })
            set_request_excerpt_metadata(request, request_items)

    # Bind each original model-requested symbol to every required request whose
    # path-specific prompt evidence contains it. Missing symbols fall back to a
    # required external request, or to the candidate file request when no
    # external context was requested, so every original symbol remains
    # fail-closed after whole-prompt clipping.
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        symbols = _required_context_symbols(candidate)
        candidate_requests = [
            request
            for request in requests
            if request.get("candidate_id") == candidate_id
        ]
        if not symbols or not candidate_requests:
            continue
        required_requests = [
            request for request in candidate_requests if request.get("required")
        ]
        optional_context_paths = {
            path
            for value in (candidate.get("_specialist_optional_context_files") or [])
            if (path := safe_repo_path(value))
        }
        # When external context is required, bind every original symbol only
        # to evidence from that required path set. Candidate-file evidence is
        # independently required for the changed anchor, but cannot satisfy a
        # same-named helper/caller/interface request. Specialist-added paths
        # remain optional even when they happen to contain the same symbol;
        # without required external context, retain the candidate-file proof.
        symbol_requests = [
            request
            for request in (required_requests or candidate_requests)
            if (
                request.get("path") == candidate.get("relevant_file")
                or request.get("path") not in optional_context_paths
            )
        ]
        fallback_request = symbol_requests[0]
        for symbol in symbols:
            matched_requests = [
                request
                for request in symbol_requests
                if any(
                    symbol in str(item.get("content") or "")
                    for item in _request_prompt_evidence(request, evidence)
                )
            ]
            for assigned_request in matched_requests or [fallback_request]:
                assigned_request.setdefault("_required_context_symbols", []).append(symbol)
        for request in candidate_requests:
            if (
                request.get("_required_context_symbols")
                and request.get("status") in _COMPLETE_RETRIEVAL_REQUEST_STATUSES
                and not _request_has_prompt_visible_symbols(request, evidence)
            ):
                request["status"] = "context_symbol_missing"

    artifact = {
        "requests": requests,
        "retrieved_evidence": evidence,
        "budget_exhausted": budget_exhausted,
        "files_read": len(unique_files),
        "changed_evidence_count": changed_evidence_count,
        "lines_retrieved": total_lines,
        "context_tokens": total_tokens,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    return evidence, artifact


def render_verification_payload(candidates: list[dict], changed_diff: str, evidence: list[dict],
                                content_fraction: float = 1.0,
                                changed_diff_fraction: Optional[float] = None) -> str:
    """Serialize all model-controlled material as JSON data for the verifier prompt."""
    content_fraction = min(1.0, max(0.0, float(content_fraction)))
    if changed_diff_fraction is None:
        changed_diff_fraction = content_fraction
    changed_diff_fraction = min(1.0, max(0.0, float(changed_diff_fraction)))

    bounded_evidence = bounded_verification_evidence(evidence, content_fraction)
    return json.dumps({"candidates": candidates,
                       "changed_diff": _bounded_content(changed_diff, changed_diff_fraction),
                       "evidence": bounded_evidence},
                      ensure_ascii=False, indent=2)


def _bounded_content(value, fraction: float) -> str:
    value = str(value or "")
    if fraction >= 1.0 or not value:
        return value
    kept_characters = int(len(value) * fraction)
    if kept_characters < 1:
        return ""
    if kept_characters >= len(value):
        return value
    return f"{value[:kept_characters]}\n...(truncated)"


def bounded_verification_evidence(evidence: list[dict], content_fraction: float) -> list[dict]:
    """Return the exact evidence material made visible in a bounded verifier prompt."""
    fraction = min(1.0, max(0.0, float(content_fraction)))
    bounded = []
    for item in evidence:
        # Static evidence remains deeply immutable while bound to the run
        # configuration. Materialize a detached JSON-shaped copy only at the
        # prompt boundary so nested mapping proxies and tuples cannot reach
        # json.dumps or alias later ambient mutations.
        bounded_item = _thaw_static_evidence_value(item)
        if not isinstance(bounded_item, dict):
            continue
        if "content" in bounded_item:
            content = str(bounded_item["content"] or "")
            allowed_characters = int(len(content) * fraction)
            anchor_start = bounded_item.get("anchor_start_line")
            anchor_end = bounded_item.get("anchor_end_line")
            excerpt_start = bounded_item.get("start_line")
            if (fraction < 1.0 and _is_atomic_prompt_evidence(bounded_item)
                    and all(isinstance(value, int) for value in (anchor_start, anchor_end, excerpt_start))):
                lines = split_git_file_lines(content)
                relative_start = anchor_start - excerpt_start
                relative_end = anchor_end - excerpt_start
                if (relative_start < 0 or relative_end >= len(lines) or relative_end < relative_start):
                    bounded_item["content"] = ""
                else:
                    selected_start = relative_start
                    selected_end = relative_end
                    selected = "\n".join(lines[selected_start:selected_end + 1])
                    if not selected or len(selected) > allowed_characters:
                        bounded_item["content"] = ""
                    else:
                        while True:
                            before = selected_start - 1
                            after = selected_end + 1
                            choices = []
                            if before >= 0:
                                choices.append((before, selected_end))
                            if after < len(lines):
                                choices.append((selected_start, after))
                            expanded = None
                            for candidate_start, candidate_end in choices:
                                candidate_content = "\n".join(lines[candidate_start:candidate_end + 1])
                                if len(candidate_content) <= allowed_characters:
                                    expanded = candidate_start, candidate_end, candidate_content
                                    break
                            if expanded is None:
                                break
                            selected_start, selected_end, selected = expanded
                        bounded_item["content"] = selected
                        bounded_item["start_line"] = excerpt_start + selected_start
                        bounded_item["end_line"] = excerpt_start + selected_end
                        bounded_item["content_truncated"] = True
            else:
                bounded_item["content"] = _bounded_content(content, fraction)
        if str(bounded_item.get("content") or "").strip():
            bounded.append(bounded_item)
    return bounded


def _candidate_prompt_evidence_gaps(
    candidate: dict,
    available_evidence: list[dict],
    retrieval_requests: list[dict],
) -> tuple[bool, int]:
    """Report changed-anchor and required-context gaps in prompt-visible evidence."""
    missing_required_request_count = sum(
        1 for request in retrieval_requests
        if (
            not retrieval_request_is_complete(request)
            or (
                (request.get("required") or request.get("_required_context_symbols"))
                and (
                    not _request_prompt_evidence(request, available_evidence)
                    or not _request_has_prompt_visible_symbols(request, available_evidence)
                )
            )
        )
    )
    candidate_side = candidate.get("side", "new")
    candidate_start = candidate.get("start_line")
    candidate_end = candidate.get("end_line")
    missing_changed_anchor = not any(
        item.get("source") in {"changed_head", "changed_patch"}
        and str(item.get("content") or "").strip()
        and item.get("path") == candidate.get("relevant_file")
        and item.get("side", "new") == candidate_side
        and isinstance(candidate_start, int)
        and isinstance(candidate_end, int)
        and isinstance(item.get("start_line"), int)
        and isinstance(item.get("end_line"), int)
        and item["start_line"] <= candidate_start
        and candidate_end <= item["end_line"]
        for item in available_evidence
    )
    return missing_changed_anchor, missing_required_request_count


def prompt_evidence_coverage(
    candidates: list[dict],
    evidence: list[dict],
    retrieval_requests: Optional[list[dict]] = None,
) -> dict:
    """Summarize whether every candidate retained its required prompt evidence."""
    evidence_by_candidate = {}
    for item in evidence:
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        for candidate_id in candidate_ids:
            evidence_by_candidate.setdefault(candidate_id, []).append(item)
    requests_by_candidate = {}
    for request in retrieval_requests if isinstance(retrieval_requests, list) else []:
        if isinstance(request, dict):
            requests_by_candidate.setdefault(request.get("candidate_id"), []).append(request)

    missing_changed_candidate_count = 0
    missing_required_request_count = 0
    complete_candidate_count = 0
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        missing_changed, missing_required = _candidate_prompt_evidence_gaps(
            candidate,
            evidence_by_candidate.get(candidate_id, []),
            requests_by_candidate.get(candidate_id, []),
        )
        missing_changed_candidate_count += int(missing_changed)
        missing_required_request_count += missing_required
        complete_candidate_count += int(not missing_changed and not missing_required)

    status = (
        "complete"
        if complete_candidate_count == len(candidates)
        else "incomplete"
    )
    return {
        "status": status,
        "candidate_count": len(candidates),
        "complete_candidate_count": complete_candidate_count,
        "missing_changed_candidate_count": missing_changed_candidate_count,
        "missing_required_request_count": missing_required_request_count,
    }


def verified_finding_identity(candidate: dict) -> Optional[tuple[str, str]]:
    """Derive identities internally from a verified assertion and its cited proof.

    The verifier cannot supply either identifier. The stable key ignores paths,
    line numbers, and identifier spelling so a file or symbol rename preserves it,
    while code shape keeps distinct verified defects from collapsing together.
    """
    anchor_shape = str(candidate.get("_changed_anchor_shape") or "")
    anchor_ordinal = candidate.get("_changed_anchor_ordinal")
    defect_ordinal = candidate.get("_trusted_defect_ordinal")
    lineage_key = str(candidate.get("_trusted_lineage_key") or "")
    if (not anchor_shape or not lineage_key
            or not isinstance(anchor_ordinal, int) or anchor_ordinal < 1
            or not isinstance(defect_ordinal, int) or defect_ordinal < 1):
        return None
    root_digest = hashlib.sha256(
        json.dumps({
            "schema": "verified-root-cause-v2",
            "changed_anchor_shape": anchor_shape,
            "changed_anchor_ordinal": anchor_ordinal,
            "trusted_defect_ordinal": defect_ordinal,
        },
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root_cause_id = f"sha256:{root_digest}"
    stable_digest = hashlib.sha256(
        json.dumps({
            "schema": "verified-finding-stable-key-v1",
            "root_cause_id": root_cause_id,
            "trusted_lineage_key": lineage_key,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return root_cause_id, f"sha256:{stable_digest}"


def _verified_anchor_shape_id(candidate: dict) -> Optional[str]:
    """Expose a non-reversible trusted shape discriminator to downstream lifecycle code."""
    anchor_shape = str(candidate.get("_changed_anchor_shape") or "")
    if not anchor_shape:
        return None
    digest = hashlib.sha256(
        json.dumps({
            "schema": "verified-anchor-shape-v1",
            "changed_anchor_shape": anchor_shape,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def apply_verification_decisions(
    candidates: list[dict],
    evidence: list[dict],
    verification_data: dict,
    retrieval_requests: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Accept only complete verified decisions backed by retrieved evidence."""
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    evidence_by_candidate = {}
    for item in evidence:
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        for candidate_id in candidate_ids:
            evidence_by_candidate.setdefault(candidate_id, []).append(item)
    requests_by_candidate = {}
    for request in retrieval_requests if isinstance(retrieval_requests, list) else []:
        if isinstance(request, dict):
            requests_by_candidate.setdefault(request.get("candidate_id"), []).append(request)
    decisions = (verification_data.get("verification") or {}).get("decisions") or []
    verified_findings = []
    result_records = []
    seen_findings = set()
    seen_identities = set()
    decision_ids = [
        str(decision.get("candidate_id") or "").strip()
        for decision in decisions if isinstance(decision, dict)
    ] if isinstance(decisions, list) else []
    duplicate_decision_ids = {
        candidate_id for candidate_id in decision_ids if candidate_id and decision_ids.count(candidate_id) > 1
    }
    recorded_duplicate_ids = set()
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict):
            continue
        candidate_id = str(decision.get("candidate_id") or "").strip()
        if candidate_id in duplicate_decision_ids:
            if candidate_id not in recorded_duplicate_ids:
                result_records.append({
                    "candidate_id": candidate_id,
                    "verdict": "rejected",
                    "reason": "duplicate_decision",
                })
                recorded_duplicate_ids.add(candidate_id)
            continue
        candidate = candidates_by_id.get(candidate_id)
        verdict = str(decision.get("verdict") or "").strip().lower()
        record = {"candidate_id": candidate_id, "verdict": verdict or "invalid"}
        normalized_severity = str(decision.get("normalized_severity") or "").strip().lower()
        if normalized_severity in {"low", "medium", "high", "critical"}:
            record["normalized_severity"] = normalized_severity
        confidence = decision.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1:
            record["confidence"] = float(confidence)
        if isinstance(decision.get("disputed"), bool):
            record["disputed"] = decision["disputed"]
        evidence_status = str(decision.get("evidence_status") or "").strip().lower()
        if evidence_status in {"complete", "insufficient"}:
            record["evidence_status"] = evidence_status
        unresolved_questions = decision.get("unresolved_questions")
        if isinstance(unresolved_questions, list):
            record["_unresolved_questions"] = tuple(
                question.strip() for question in unresolved_questions
                if isinstance(question, str) and question.strip()
            )
        if candidate is None:
            record["reason"] = "unknown_candidate"
            result_records.append(record)
            continue
        if verdict != "verified":
            record["reason"] = str(decision.get("reason") or "rejected_by_verifier").strip()
            result_records.append(record)
            continue
        cited_paths = decision.get("evidence_paths") or []
        if isinstance(cited_paths, str):
            cited_paths = [cited_paths]
        available_evidence = evidence_by_candidate.get(candidate_id, [])
        _, missing_required_request_count = _candidate_prompt_evidence_gaps(
            candidate,
            available_evidence,
            requests_by_candidate.get(candidate_id, []),
        )
        if missing_required_request_count:
            record["verdict"] = "rejected"
            record["reason"] = "required_context_unavailable"
            result_records.append(record)
            continue
        candidate_side = candidate.get("side", "new")
        available_paths = {item.get("path") for item in available_evidence}
        normalized_citations = [safe_repo_path(path) for path in cited_paths]
        normalized_citations = [path for path in normalized_citations if path in available_paths]
        trigger = str(decision.get("trigger") or candidate.get("trigger") or "").strip()
        impact = str(decision.get("impact") or candidate.get("impact") or "").strip()
        explanation = str(decision.get("issue_content") or candidate.get("issue_content") or "").strip()
        relevant_file = safe_repo_path(decision.get("relevant_file") or candidate.get("relevant_file"))
        try:
            start_line = int(decision.get("start_line") or candidate.get("start_line"))
            end_line = int(decision.get("end_line") or candidate.get("end_line"))
        except (TypeError, ValueError):
            start_line = 0
            end_line = 0
        if not any(
            item.get("source") in {"changed_head", "changed_patch"}
            and str(item.get("content") or "").strip()
            and item.get("path") == candidate.get("relevant_file")
            and item.get("side", "new") == candidate_side
            and isinstance(item.get("start_line"), int)
            and isinstance(item.get("end_line"), int)
            and item["start_line"] <= start_line
            and end_line <= item["end_line"]
            for item in available_evidence
        ):
            record["verdict"] = "rejected"
            record["reason"] = "changed_code_evidence_unavailable"
            result_records.append(record)
            continue
        changed_line_ranges = candidate.get("_changed_line_ranges") or []
        trusted_side_line_count = candidate.get("_trusted_side_line_count")
        if (not normalized_citations or not trigger or not impact or not explanation or
                relevant_file != candidate.get("relevant_file") or start_line < 1 or end_line < start_line or
                (isinstance(trusted_side_line_count, int)
                 and end_line > trusted_side_line_count)):
            record["verdict"] = "rejected"
            record["reason"] = "unverified_or_incomplete_evidence"
            result_records.append(record)
            continue
        if not _line_is_changed(start_line, changed_line_ranges):
            record["verdict"] = "rejected"
            record["reason"] = "location_not_in_changed_lines"
            result_records.append(record)
            continue
        display_file = safe_repo_path(candidate.get("_display_file")) or relevant_file
        finding = {
            "_candidate_id": candidate_id,
            "relevant_file": display_file,
            "issue_header": str(decision.get("issue_header") or candidate.get("issue_header") or "Issue").strip(),
            "issue_content": (
                f"{explanation}\n\n**Trigger:** {trigger}\n\n**Impact:** {impact}\n\n"
                f"**Verified with:** {', '.join(f'`{path}`' for path in normalized_citations)}"
            ),
            "start_line": start_line,
            "end_line": end_line,
            "side": candidate_side,
            "trigger": trigger,
            "impact": impact,
            "verification_evidence": normalized_citations,
        }
        identity = verified_finding_identity(candidate)
        if identity is None:
            record["verdict"] = "rejected"
            record["reason"] = "trusted_identity_unavailable"
            result_records.append(record)
            continue
        anchor_shape_id = _verified_anchor_shape_id(candidate)
        anchor_occurrence_count = candidate.get("_changed_anchor_occurrence_count")
        same_anchor_candidate_count = candidate.get("_trusted_same_anchor_candidate_count")
        if (
            anchor_shape_id is None
            or isinstance(anchor_occurrence_count, bool)
            or not isinstance(anchor_occurrence_count, int)
            or anchor_occurrence_count < 1
            or isinstance(same_anchor_candidate_count, bool)
            or not isinstance(same_anchor_candidate_count, int)
            or same_anchor_candidate_count < 1
        ):
            record["verdict"] = "rejected"
            record["reason"] = "trusted_identity_unavailable"
            result_records.append(record)
            continue
        finding["root_cause_id"], finding["trusted_stable_key"] = identity
        if "normalized_severity" in record:
            finding["normalized_severity"] = record["normalized_severity"]
        finding["_trusted_anchor_shape_id"] = anchor_shape_id
        finding["_trusted_anchor_shape_occurrence_count"] = anchor_occurrence_count
        finding["_trusted_same_anchor_candidate_count"] = same_anchor_candidate_count
        finding["_trusted_patch_is_complete"] = candidate.get("_trusted_patch_is_complete") is True
        if identity in seen_identities:
            record["verdict"] = "rejected"
            record["reason"] = "trusted_identity_collision"
            result_records.append(record)
            continue
        finding_key = _candidate_key(finding)
        if finding_key in seen_findings:
            record["verdict"] = "rejected"
            record["reason"] = "duplicate_verified_finding"
            result_records.append(record)
            continue
        seen_findings.add(finding_key)
        seen_identities.add(identity)
        verified_findings.append(finding)
        record["trusted_stable_key"] = identity[1]
        record["evidence_paths"] = normalized_citations
        result_records.append(record)

    decided_ids = {record["candidate_id"] for record in result_records}
    for candidate_id in candidates_by_id.keys() - decided_ids:
        result_records.append({"candidate_id": candidate_id, "verdict": "rejected", "reason": "missing_decision"})
    candidate_order = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    verified_findings.sort(
        key=lambda finding: candidate_order.get(finding["_candidate_id"], len(candidate_order))
    )
    for finding in verified_findings:
        finding.pop("_candidate_id", None)
    get_logger().debug(
        "Candidate verification decisions validated",
        artifact=telemetry_safe_artifact({"decisions": result_records}),
    )
    return verified_findings, result_records
