"""Versioned, shadow-only specialist contracts and bounded orchestration."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from threading import Lock
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.git_patch_processing import RE_HUNK_HEADER, iter_git_patch_lines
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.review_snapshot import ReviewSnapshot
from pr_agent.algo.run_details import get_run_details, record_specialist_result, specialist_runs_to_dict
from pr_agent.algo.token_handler import TokenHandler
from pr_agent.algo.types import FilePatchInfo
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

SPECIALIST_INPUT_SCHEMA_VERSION = "review-specialist-input-v2"
SPECIALIST_BATCH_SCHEMA_VERSION = "review-specialist-shadow-batch-v1"

_ROLE_ORDER: tuple["SpecialistRole", ...]


class SpecialistRole(str, Enum):
    CHANGE_CLASSIFICATION = "change_classification"
    RISK_RECOMMENDATION = "risk_recommendation"
    DIFF_PRIORITIZATION = "diff_prioritization"


_ROLE_ORDER = tuple(SpecialistRole)


class SpecialistState(str, Enum):
    SUCCESS = "success"
    DISABLED = "disabled"
    CACHED = "cached"
    INPUT_BUDGET_EXHAUSTED = "input_budget_exhausted"
    AGGREGATE_BUDGET_EXHAUSTED = "aggregate_budget_exhausted"
    TIMEOUT = "timeout"
    PROVIDER_FAILURE = "provider_failure"
    UNAVAILABLE = "unavailable"
    MALFORMED_OUTPUT = "malformed_output"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"
    CANCELLED = "cancelled"


class SpecialistConfigurationError(ValueError):
    pass


class SpecialistOutputError(ValueError):
    pass


class SpecialistLowConfidenceError(SpecialistOutputError):
    def __init__(self, output: Mapping[str, Any]):
        super().__init__("specialist response confidence is below the role threshold")
        self.output = copy.deepcopy(dict(output))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()


def _normalized_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/")
    candidate = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"unsafe specialist path: {path!r}")
    return candidate.as_posix()


@dataclass(frozen=True)
class SpecialistHunk:
    hunk_id: str
    path: str
    start_line: int
    end_line: int
    added_lines: tuple[int, ...]
    deleted_lines: tuple[int, ...]
    patch_hash: str


@dataclass(frozen=True)
class SpecialistInput:
    """One immutable input shared by every enabled role in a batch."""

    snapshot_id: str
    head_sha: str
    title: str
    description: str
    changed_paths: tuple[str, ...]
    diff: str
    hunks: tuple[SpecialistHunk, ...]
    allowed_change_labels: tuple[str, ...] = field(default_factory=tuple)
    deterministic_results: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    event: str = "pull_request"
    policy_version: str = "specialist-shadow-v1"
    review_configuration_hash: Optional[str] = None
    schema_version: str = SPECIALIST_INPUT_SCHEMA_VERSION
    input_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_paths", tuple(sorted(set(self.changed_paths))))
        labels = tuple(sorted({str(label).strip() for label in self.allowed_change_labels}))
        if any(not label for label in labels):
            raise ValueError("allowed_change_labels cannot contain blank entries")
        object.__setattr__(self, "allowed_change_labels", labels)
        object.__setattr__(
            self,
            "hunks",
            tuple(sorted(self.hunks, key=lambda hunk: (hunk.path, hunk.start_line, hunk.hunk_id))),
        )
        object.__setattr__(
            self,
            "deterministic_results",
            tuple(_freeze_json(result) for result in self.deterministic_results),
        )
        identity = self.to_dict(include_hash=False)
        object.__setattr__(self, "input_hash", _sha256(_canonical_json(identity)))

    def to_dict(self, *, include_hash: bool = True, include_diff: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "head_sha": self.head_sha,
            "title": self.title,
            "description": self.description,
            "changed_paths": list(self.changed_paths),
            "diff": self.diff,
            "hunks": [asdict(hunk) for hunk in self.hunks],
            "allowed_change_labels": list(self.allowed_change_labels),
            "deterministic_results": [_thaw_json(result) for result in self.deterministic_results],
            "event": self.event,
            "policy_version": self.policy_version,
            "review_configuration_hash": self.review_configuration_hash,
        }
        if include_hash:
            data["input_hash"] = self.input_hash
        if not include_diff:
            data.pop("diff", None)
        return data


@dataclass(frozen=True)
class SpecialistSnapshotContext:
    snapshot: ReviewSnapshot
    current_snapshot_id: Callable[[], Optional[str]]


_snapshot_context: ContextVar[Optional[SpecialistSnapshotContext]] = ContextVar(
    "pr_agent_specialist_snapshot_context", default=None
)


def get_specialist_snapshot_context() -> Optional[SpecialistSnapshotContext]:
    return _snapshot_context.get()


@contextmanager
def use_specialist_snapshot_context(
    snapshot: ReviewSnapshot,
    current_snapshot_id: Callable[[], Optional[str]],
) -> Iterator[None]:
    """Bind local specialists to the already captured immutable snapshot."""

    token = _snapshot_context.set(
        SpecialistSnapshotContext(snapshot=snapshot, current_snapshot_id=current_snapshot_id)
    )
    try:
        yield
    finally:
        _snapshot_context.reset(token)


@dataclass(frozen=True)
class SpecialistRoleConfig:
    role: SpecialistRole
    enabled: bool
    model: str
    deployment: Optional[str]
    fallback_models: tuple[str, ...]
    fallback_deployments: tuple[Optional[str], ...]
    timeout_seconds: float
    model_retries: int
    provider_retries: int
    input_token_budget: int
    output_token_budget: int
    minimum_confidence: float

    def model_route(self) -> AIModelRoute:
        return AIModelRoute(
            models=(self.model, *self.fallback_models),
            deployments=(self.deployment, *self.fallback_deployments),
            timeout_seconds=self.timeout_seconds,
            model_retries=self.model_retries,
            provider_retries=self.provider_retries,
            max_output_tokens=self.output_token_budget,
            attribution=self.role.value,
        )

    @property
    def worst_case_provider_calls(self) -> int:
        return len(self.model_route().models) * self.model_retries * (1 + self.provider_retries)


@dataclass(frozen=True)
class SpecialistPrompt:
    role: SpecialistRole
    prompt_version: str
    input_schema_version: str
    schema_version: str
    system: str
    user: str

    @property
    def content_hash(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "role": self.role.value,
                    "prompt_version": self.prompt_version,
                    "input_schema_version": self.input_schema_version,
                    "schema_version": self.schema_version,
                    "system": self.system,
                    "user": self.user,
                }
            )
        )


@dataclass(frozen=True)
class SpecialistPipelineConfig:
    enabled: bool
    mode: str
    aggregate_timeout_seconds: float
    aggregate_token_budget: int
    max_concurrency: int
    cache_enabled: bool
    cache_max_entries: int
    cancel_stale_inputs: bool
    allowed_change_labels: tuple[str, ...]
    roles: tuple[SpecialistRoleConfig, ...]
    prompts: tuple[SpecialistPrompt, ...]
    configuration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.enabled and self.mode != "shadow":
            raise SpecialistConfigurationError("specialist_pipeline.mode must remain 'shadow'")
        labels = tuple(sorted({str(label).strip() for label in self.allowed_change_labels}))
        if not labels or any(not label for label in labels):
            raise SpecialistConfigurationError("allowed_change_labels cannot contain blank entries")
        object.__setattr__(self, "allowed_change_labels", labels)
        identity = {
            "mode": self.mode,
            "aggregate_timeout_seconds": self.aggregate_timeout_seconds,
            "aggregate_token_budget": self.aggregate_token_budget,
            "max_concurrency": self.max_concurrency,
            "cache_enabled": self.cache_enabled,
            "cache_max_entries": self.cache_max_entries,
            "cancel_stale_inputs": self.cancel_stale_inputs,
            "allowed_change_labels": self.allowed_change_labels,
            "roles": [
                {**asdict(role), "role": role.role.value}
                for role in self.roles
            ],
            "prompts": [
                {
                    "role": prompt.role.value,
                    "prompt_version": prompt.prompt_version,
                    "input_schema_version": prompt.input_schema_version,
                    "schema_version": prompt.schema_version,
                    "content_hash": prompt.content_hash,
                }
                for prompt in self.prompts
            ],
        }
        object.__setattr__(self, "configuration_hash", _sha256(_canonical_json(identity)))

    def role_config(self, role: SpecialistRole) -> SpecialistRoleConfig:
        return next(config for config in self.roles if config.role is role)

    def prompt(self, role: SpecialistRole) -> SpecialistPrompt:
        return next(prompt for prompt in self.prompts if prompt.role is role)


@dataclass(frozen=True)
class RoleExecution:
    role: SpecialistRole
    state: SpecialistState
    output: Optional[Mapping[str, Any]] = None
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
    latency_seconds: float = 0.0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_key: Optional[str] = None
    model: Optional[str] = None
    deployment: Optional[str] = None
    fallback_used: bool = False


@dataclass(frozen=True)
class SpecialistBatchResult:
    snapshot_id: str
    head_sha: str
    input_hash: str
    configuration_hash: str
    records: tuple[RoleExecution, ...]
    role_records: Mapping[str, Mapping[str, Any]]
    changed_path_count: int
    hunk_count: int
    stale: bool = False
    schema_version: str = SPECIALIST_BATCH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "head_sha": self.head_sha,
            "input_hash": self.input_hash,
            "configuration_hash": self.configuration_hash,
            "stale": self.stale,
            "input_summary": {
                "changed_paths": self.changed_path_count,
                "hunks": self.hunk_count,
            },
            "roles": copy.deepcopy(dict(self.role_records)),
        }


def unavailable_specialist_batch(
    pipeline: SpecialistPipelineConfig,
    *,
    failure_reason: str,
) -> SpecialistBatchResult:
    """Return explicit role evidence when a provider cannot supply a stable head identity."""

    records = tuple(
        RoleExecution(
            role=config.role,
            state=SpecialistState.UNAVAILABLE,
            failure_reason=failure_reason,
        )
        for config in pipeline.roles
        if config.enabled
    )
    _record_role_results(records, pipeline)
    return SpecialistBatchResult(
        snapshot_id="unavailable",
        head_sha="",
        input_hash="",
        configuration_hash=pipeline.configuration_hash,
        records=records,
        role_records=_role_records(records, pipeline),
        changed_path_count=0,
        hunk_count=0,
    )


@dataclass(frozen=True)
class _CachedRoleOutput:
    output: Mapping[str, Any]
    confidence: float
    model: Optional[str]
    deployment: Optional[str]
    fallback_used: bool


class _SpecialistCache:
    """Bounded in-process cache for validated, source-free shadow outputs."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, _CachedRoleOutput] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[_CachedRoleOutput]:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, key: str, value: _CachedRoleOutput, max_entries: int) -> None:
        with self._lock:
            self._entries[key] = copy.deepcopy(value)
            self._entries.move_to_end(key)
            while len(self._entries) > max(1, max_entries):
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_SPECIALIST_CACHE = _SpecialistCache()


def clear_specialist_cache() -> None:
    """Clear the process cache; intended for tests and explicit lifecycle resets."""

    _SPECIALIST_CACHE.clear()


def specialists_enabled() -> bool:
    """Cheap gate used before constructing any specialist object."""

    section = get_settings().get("specialist_pipeline", {}) or {}
    return section.get("enabled", False) is True


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise SpecialistConfigurationError(f"{key} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SpecialistConfigurationError(f"{key} must be a positive integer") from exc
    if parsed < 1:
        raise SpecialistConfigurationError(f"{key} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise SpecialistConfigurationError(f"{key} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SpecialistConfigurationError(f"{key} must be a non-negative integer") from exc
    if parsed < 0:
        raise SpecialistConfigurationError(f"{key} must be a non-negative integer")
    return parsed


def _positive_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise SpecialistConfigurationError(f"{key} must be positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecialistConfigurationError(f"{key} must be positive") from exc
    if parsed <= 0:
        raise SpecialistConfigurationError(f"{key} must be positive")
    return parsed


def _confidence(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise SpecialistConfigurationError(f"{key} must be between 0 and 1")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecialistConfigurationError(f"{key} must be between 0 and 1") from exc
    if not 0 <= parsed <= 1:
        raise SpecialistConfigurationError(f"{key} must be between 0 and 1")
    return parsed


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        values = [str(part).strip() for part in value]
    else:
        raise SpecialistConfigurationError(f"{key} must be a string list")
    if any(not part for part in values):
        raise SpecialistConfigurationError(f"{key} cannot contain blank entries")
    return tuple(values)


def _deployment_tuple(value: Any, key: str) -> tuple[Optional[str], ...]:
    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise SpecialistConfigurationError(f"{key} must be a string list")
    return tuple(str(part).strip() or None for part in values)


def load_specialist_pipeline_config() -> SpecialistPipelineConfig:
    """Snapshot repository settings and prompts into immutable role contracts."""

    settings = get_settings()
    section = settings.get("specialist_pipeline", {}) or {}
    if section.get("enabled", False) is not True:
        raise SpecialistConfigurationError("specialist_pipeline is disabled")
    allowed_labels = _string_tuple(section.get("allowed_change_labels", []), "allowed_change_labels")
    if not allowed_labels or len(set(allowed_labels)) != len(allowed_labels):
        raise SpecialistConfigurationError("allowed_change_labels must contain unique labels")

    roles: list[SpecialistRoleConfig] = []
    prompts: list[SpecialistPrompt] = []
    default_model = str(settings.config.model).strip()
    for role in _ROLE_ORDER:
        role_section = section.get(role.value, {}) or {}
        enabled = role_section.get("enabled", True)
        if not isinstance(enabled, bool):
            raise SpecialistConfigurationError(f"{role.value}.enabled must be a boolean")
        model = str(role_section.get("model", "") or default_model).strip()
        if not model:
            raise SpecialistConfigurationError(f"{role.value}.model cannot be blank")
        deployment = str(role_section.get("deployment", "") or "").strip() or None
        fallback_models = _string_tuple(role_section.get("fallback_models", []), f"{role.value}.fallback_models")
        fallback_deployments_raw = _deployment_tuple(
            role_section.get("fallback_deployments", []), f"{role.value}.fallback_deployments"
        )
        if fallback_deployments_raw and len(fallback_deployments_raw) != len(fallback_models):
            raise SpecialistConfigurationError(
                f"{role.value}.fallback_deployments must match fallback_models"
            )
        fallback_deployments = (
            fallback_deployments_raw
            if fallback_deployments_raw
            else (None,) * len(fallback_models)
        )
        roles.append(
            SpecialistRoleConfig(
                role=role,
                enabled=enabled,
                model=model,
                deployment=deployment,
                fallback_models=fallback_models,
                fallback_deployments=fallback_deployments,
                timeout_seconds=_positive_float(role_section.get("timeout_seconds", 5), f"{role.value}.timeout"),
                model_retries=_positive_int(role_section.get("model_retries", 1), f"{role.value}.retries"),
                provider_retries=_nonnegative_int(
                    role_section.get("provider_retries", 0), f"{role.value}.provider_retries"
                ),
                input_token_budget=_positive_int(
                    role_section.get("input_token_budget", 4000), f"{role.value}.input_token_budget"
                ),
                output_token_budget=_positive_int(
                    role_section.get("output_token_budget", 600), f"{role.value}.output_token_budget"
                ),
                minimum_confidence=_confidence(
                    role_section.get("minimum_confidence", 0.6), f"{role.value}.minimum_confidence"
                ),
            )
        )

        prompt_section = settings.get(f"review_specialist_{role.value}_prompt", {}) or {}
        prompt = SpecialistPrompt(
            role=role,
            prompt_version=str(prompt_section.get("prompt_version", "") or "").strip(),
            input_schema_version=str(
                prompt_section.get("input_schema_version", "") or ""
            ).strip(),
            schema_version=str(prompt_section.get("schema_version", "") or "").strip(),
            system=str(prompt_section.get("system", "") or ""),
            user=str(prompt_section.get("user", "") or ""),
        )
        if not all(
            (
                prompt.prompt_version,
                prompt.input_schema_version,
                prompt.schema_version,
                prompt.system.strip(),
                prompt.user.strip(),
            )
        ):
            raise SpecialistConfigurationError(f"missing versioned prompt for {role.value}")
        prompts.append(prompt)

    cache_enabled = section.get("cache_enabled", True)
    cancel_stale_inputs = section.get("cancel_stale_inputs", True)
    if not isinstance(cache_enabled, bool) or not isinstance(cancel_stale_inputs, bool):
        raise SpecialistConfigurationError("cache_enabled and cancel_stale_inputs must be booleans")
    return SpecialistPipelineConfig(
        enabled=True,
        mode=str(section.get("mode", "shadow")).strip().lower(),
        aggregate_timeout_seconds=_positive_float(
            section.get("aggregate_timeout_seconds", 8), "aggregate_timeout_seconds"
        ),
        aggregate_token_budget=_positive_int(
            section.get("aggregate_token_budget", 12000), "aggregate_token_budget"
        ),
        max_concurrency=_positive_int(section.get("max_concurrency", 3), "max_concurrency"),
        cache_enabled=cache_enabled,
        cache_max_entries=_positive_int(section.get("cache_max_entries", 128), "cache_max_entries"),
        cancel_stale_inputs=cancel_stale_inputs,
        allowed_change_labels=allowed_labels,
        roles=tuple(roles),
        prompts=tuple(prompts),
    )


def _parse_hunks(path: str, patch: str) -> tuple[SpecialistHunk, ...]:
    hunks: list[SpecialistHunk] = []
    current_header: Optional[str] = None
    current_lines: list[str] = []
    old_start_line = 0
    old_count = 0
    new_start_line = 0
    new_count = 0

    def finish() -> None:
        if current_header is None:
            return
        patch_text = "".join(current_lines)
        old_line = old_start_line
        new_line = new_start_line
        consumed_old = 0
        consumed_new = 0
        added_lines: list[int] = []
        deleted_lines: list[int] = []
        valid = True
        for line in current_lines[1:]:
            if line.startswith("+"):
                added_lines.append(new_line)
                new_line += 1
                consumed_new += 1
            elif line.startswith("-"):
                deleted_lines.append(old_line)
                old_line += 1
                consumed_old += 1
            elif line.startswith(" "):
                old_line += 1
                new_line += 1
                consumed_old += 1
                consumed_new += 1
            elif line.startswith("\\ No newline at end of file"):
                continue
            else:
                valid = False
        if not valid or consumed_old != old_count or consumed_new != new_count:
            return
        end_line = new_start_line + new_count - 1
        hunk_hash = _sha256(f"{path}\n{current_header}\n{patch_text}")
        hunks.append(
            SpecialistHunk(
                hunk_id=hunk_hash,
                path=path,
                start_line=new_start_line,
                end_line=end_line,
                added_lines=tuple(added_lines),
                deleted_lines=tuple(deleted_lines),
                patch_hash=_sha256(patch_text),
            )
        )

    for line in iter_git_patch_lines(patch):
        if line.startswith("@@"):
            finish()
            current_header = None
            current_lines = []
            match = RE_HUNK_HEADER.match(line.rstrip("\r\n"))
            if match is None:
                continue
            current_header = line.rstrip("\r\n")
            current_lines = [line]
            old_start_line = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start_line = int(match.group(3))
            new_count = int(match.group(4) or "1")
        elif current_header is not None:
            current_lines.append(line)
    finish()
    return tuple(hunks)


def build_specialist_input(
    *,
    title: str,
    description: str,
    diff_files: Sequence[FilePatchInfo],
    head_sha: str,
    allowed_change_labels: Sequence[str],
    snapshot: Optional[ReviewSnapshot] = None,
) -> SpecialistInput:
    """Build one input from provider-normalized files or an existing local snapshot."""

    normalized_files: list[tuple[str, str]] = []
    hunks: list[SpecialistHunk] = []
    for diff_file in diff_files:
        path = _normalized_path(diff_file.filename)
        patch = str(diff_file.patch or "")
        normalized_files.append((path, patch))
        hunks.extend(_parse_hunks(path, patch))
    normalized_files.sort(key=lambda item: item[0])
    provider_diff = "\n".join(f"## File: {path}\n{patch}" for path, patch in normalized_files)

    if snapshot is not None:
        snapshot_id = snapshot.snapshot_id
        input_head = snapshot.snapshot_id
        changed_paths = snapshot.changed_paths
        diff = snapshot.diff
        deterministic_results = snapshot.deterministic_results
        event = snapshot.event.value
        policy_version = snapshot.policy_version
        review_configuration_hash = snapshot.review_configuration_hash
    else:
        if not head_sha:
            raise ValueError("specialist input requires a stable head SHA")
        snapshot_identity = {
            "schema_version": SPECIALIST_INPUT_SCHEMA_VERSION,
            "head_sha": head_sha,
            "changed_paths": [path for path, _ in normalized_files],
            "diff": provider_diff,
        }
        snapshot_id = _sha256(_canonical_json(snapshot_identity))
        input_head = head_sha
        changed_paths = tuple(path for path, _ in normalized_files)
        diff = provider_diff
        deterministic_results = ()
        event = "pull_request"
        policy_version = "specialist-shadow-v1"
        review_configuration_hash = None

    return SpecialistInput(
        snapshot_id=snapshot_id,
        head_sha=input_head,
        title=str(title or ""),
        description=str(description or ""),
        changed_paths=tuple(changed_paths),
        diff=diff,
        hunks=tuple(hunks),
        allowed_change_labels=tuple(allowed_change_labels),
        deterministic_results=tuple(deterministic_results),
        event=event,
        policy_version=policy_version,
        review_configuration_hash=review_configuration_hash,
    )


def _render_prompt(prompt: SpecialistPrompt, specialist_input: SpecialistInput) -> tuple[str, str]:
    role_input = specialist_input.to_dict()
    role_input["schema_version"] = prompt.input_schema_version
    variables = {
        "specialist_input_json": _canonical_json(role_input),
        "input_schema_version": prompt.input_schema_version,
        "output_schema_version": prompt.schema_version,
    }
    system = TokenHandler.render_plain_text_prompt(prompt.system, variables)
    user = TokenHandler.render_plain_text_prompt(prompt.user, variables)
    return system, user


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    if actual != keys:
        raise SpecialistOutputError(
            f"{where} keys must be {sorted(keys)}; received {sorted(actual)}"
        )


def _output_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise SpecialistOutputError("confidence must be between 0 and 1")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecialistOutputError("confidence must be between 0 and 1") from exc
    if not 0 <= confidence <= 1:
        raise SpecialistOutputError("confidence must be between 0 and 1")
    return confidence


def _deterministic_rule_ids(specialist_input: SpecialistInput) -> set[str]:
    ids: set[str] = set()
    for index, result in enumerate(specialist_input.deterministic_results):
        candidate = result.get("id") or result.get("name") or result.get("check")
        ids.add(str(candidate or f"deterministic-{index}"))
    return ids


def _validate_evidence(value: Any, specialist_input: SpecialistInput) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecialistOutputError("evidence must be an object")
    source = value.get("source")
    if source == "diff_hunk":
        _require_exact_keys(
            value,
            {"source", "path", "hunk_id", "side", "line"},
            "diff_hunk evidence",
        )
        try:
            path = _normalized_path(str(value["path"]))
        except (TypeError, ValueError) as exc:
            raise SpecialistOutputError("diff_hunk evidence is malformed") from exc
        line = value["line"]
        if isinstance(line, bool) or not isinstance(line, int):
            raise SpecialistOutputError("diff_hunk evidence line must be an integer")
        side = value["side"]
        if not isinstance(side, str) or side not in {"old", "new"}:
            raise SpecialistOutputError("diff_hunk evidence side must be 'old' or 'new'")
        hunk = next(
            (
                item
                for item in specialist_input.hunks
                if item.path == path and item.hunk_id == value["hunk_id"]
            ),
            None,
        )
        changed_lines = () if hunk is None else (
            hunk.deleted_lines if side == "old" else hunk.added_lines
        )
        if hunk is None or line not in changed_lines:
            raise SpecialistOutputError(
                f"diff_hunk evidence does not identify a changed {side}-side line"
            )
        return {
            "source": source,
            "path": path,
            "hunk_id": hunk.hunk_id,
            "side": side,
            "line": line,
        }
    if source == "pull_request":
        _require_exact_keys(value, {"source", "field"}, "pull_request evidence")
        if value["field"] not in {"title", "description"}:
            raise SpecialistOutputError("pull_request evidence field is unsupported")
        return {"source": source, "field": value["field"]}
    if source == "deterministic_result":
        _require_exact_keys(value, {"source", "rule_id"}, "deterministic_result evidence")
        if str(value["rule_id"]) not in _deterministic_rule_ids(specialist_input):
            raise SpecialistOutputError("deterministic_result evidence is unsupported")
        return {"source": source, "rule_id": str(value["rule_id"])}
    raise SpecialistOutputError("evidence source is unsupported")


def _validate_evidence_list(value: Any, specialist_input: SpecialistInput) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecialistOutputError("every specialist item requires evidence")
    return [_validate_evidence(item, specialist_input) for item in value]


def _validate_change_classification(
    data: Mapping[str, Any],
    specialist_input: SpecialistInput,
    pipeline: SpecialistPipelineConfig,
) -> dict[str, Any]:
    if specialist_input.allowed_change_labels != pipeline.allowed_change_labels:
        raise SpecialistOutputError(
            "classifier input policy does not match the pipeline configuration"
        )
    _require_exact_keys(data, {"schema_version", "confidence", "labels"}, "classification output")
    labels = data["labels"]
    if not isinstance(labels, list):
        raise SpecialistOutputError("labels must be a list")
    validated = []
    seen: set[str] = set()
    for item in labels:
        if not isinstance(item, Mapping):
            raise SpecialistOutputError("classification labels must be objects")
        _require_exact_keys(item, {"label", "evidence"}, "classification label")
        label = str(item["label"])
        if label not in pipeline.allowed_change_labels or label in seen:
            raise SpecialistOutputError("classification label is unsupported or duplicated")
        seen.add(label)
        validated.append({"label": label, "evidence": _validate_evidence_list(item["evidence"], specialist_input)})
    return {
        "schema_version": data["schema_version"],
        "confidence": _output_confidence(data["confidence"]),
        "labels": validated,
    }


def _validate_risk_recommendation(
    data: Mapping[str, Any], specialist_input: SpecialistInput
) -> dict[str, Any]:
    _require_exact_keys(data, {"schema_version", "confidence", "recommendation", "reasons"}, "risk output")
    recommendation = str(data["recommendation"])
    if recommendation not in {"escalate", "none"}:
        raise SpecialistOutputError("risk recommendation may only escalate or report none")
    reasons = data["reasons"]
    if not isinstance(reasons, list) or not reasons:
        raise SpecialistOutputError("risk recommendation requires at least one evidenced reason")
    validated = []
    for item in reasons:
        if not isinstance(item, Mapping):
            raise SpecialistOutputError("risk reasons must be objects")
        _require_exact_keys(item, {"reason", "evidence"}, "risk reason")
        reason = str(item["reason"] or "").strip()
        if not reason:
            raise SpecialistOutputError("risk reason cannot be blank")
        validated.append({"reason": reason, "evidence": _validate_evidence_list(item["evidence"], specialist_input)})
    return {
        "schema_version": data["schema_version"],
        "confidence": _output_confidence(data["confidence"]),
        "recommendation": recommendation,
        "reasons": validated,
    }


def _validate_diff_prioritization(
    data: Mapping[str, Any], specialist_input: SpecialistInput
) -> dict[str, Any]:
    _require_exact_keys(
        data,
        {"schema_version", "confidence", "ranked_hunks", "context_requests"},
        "prioritization output",
    )
    ranked = data["ranked_hunks"]
    requests = data["context_requests"]
    if not isinstance(ranked, list) or not isinstance(requests, list):
        raise SpecialistOutputError("ranked_hunks and context_requests must be lists")
    validated_ranked = []
    seen_hunks: set[str] = set()
    seen_ranks: set[int] = set()
    for item in ranked:
        if not isinstance(item, Mapping):
            raise SpecialistOutputError("ranked hunks must be objects")
        _require_exact_keys(item, {"rank", "path", "hunk_id", "reason", "evidence"}, "ranked hunk")
        try:
            path = _normalized_path(str(item["path"]))
        except (TypeError, ValueError) as exc:
            raise SpecialistOutputError("ranked hunk is malformed") from exc
        rank = item["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise SpecialistOutputError("ranked hunk rank must be an integer")
        hunk_id = str(item["hunk_id"])
        if rank < 1 or rank in seen_ranks or hunk_id in seen_hunks:
            raise SpecialistOutputError("ranked hunk rank or identity is duplicated")
        if not any(hunk.path == path and hunk.hunk_id == hunk_id for hunk in specialist_input.hunks):
            raise SpecialistOutputError("ranked hunk is not in the immutable input")
        reason = str(item["reason"] or "").strip()
        if not reason:
            raise SpecialistOutputError("ranked hunk reason cannot be blank")
        evidence = _validate_evidence_list(item["evidence"], specialist_input)
        if not any(ref.get("path") == path and ref.get("hunk_id") == hunk_id for ref in evidence):
            raise SpecialistOutputError("ranked hunk requires evidence from that hunk")
        seen_hunks.add(hunk_id)
        seen_ranks.add(rank)
        validated_ranked.append(
            {"rank": rank, "path": path, "hunk_id": hunk_id, "reason": reason, "evidence": evidence}
        )
    if seen_ranks and seen_ranks != set(range(1, len(seen_ranks) + 1)):
        raise SpecialistOutputError("ranked hunk ranks must be contiguous")
    validated_ranked.sort(key=lambda item: item["rank"])

    validated_requests = []
    for item in requests:
        if not isinstance(item, Mapping):
            raise SpecialistOutputError("context requests must be objects")
        _require_exact_keys(
            item,
            {"kind", "target", "anchor_path", "anchor_hunk_id", "reason", "evidence"},
            "context request",
        )
        kind = str(item["kind"])
        if kind not in {"caller", "contract", "test", "symbol", "repository_rule"}:
            raise SpecialistOutputError("context request kind is unsupported")
        target = str(item["target"] or "").strip()
        reason = str(item["reason"] or "").strip()
        anchor_path = _normalized_path(str(item["anchor_path"]))
        anchor_hunk_id = str(item["anchor_hunk_id"])
        if not target or not reason:
            raise SpecialistOutputError("context request target and reason cannot be blank")
        if not any(
            hunk.path == anchor_path and hunk.hunk_id == anchor_hunk_id for hunk in specialist_input.hunks
        ):
            raise SpecialistOutputError("context request anchor is not in the immutable input")
        evidence = _validate_evidence_list(item["evidence"], specialist_input)
        validated_requests.append(
            {
                "kind": kind,
                "target": target,
                "anchor_path": anchor_path,
                "anchor_hunk_id": anchor_hunk_id,
                "reason": reason,
                "evidence": evidence,
            }
        )
    return {
        "schema_version": data["schema_version"],
        "confidence": _output_confidence(data["confidence"]),
        "ranked_hunks": validated_ranked,
        "context_requests": validated_requests,
    }


def validate_specialist_output(
    role: SpecialistRole,
    response: str,
    specialist_input: SpecialistInput,
    pipeline: SpecialistPipelineConfig,
) -> dict[str, Any]:
    try:
        data = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SpecialistOutputError("specialist response is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise SpecialistOutputError("specialist response must be a JSON object")
    prompt = pipeline.prompt(role)
    if data.get("schema_version") != prompt.schema_version:
        raise SpecialistOutputError("specialist response schema_version does not match the role contract")
    if role is SpecialistRole.CHANGE_CLASSIFICATION:
        return _validate_change_classification(data, specialist_input, pipeline)
    if role is SpecialistRole.RISK_RECOMMENDATION:
        return _validate_risk_recommendation(data, specialist_input)
    return _validate_diff_prioritization(data, specialist_input)


def _cache_key(
    specialist_input: SpecialistInput,
    pipeline: SpecialistPipelineConfig,
    config: SpecialistRoleConfig,
    prompt: SpecialistPrompt,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "snapshot_id": specialist_input.snapshot_id,
                "head_sha": specialist_input.head_sha,
                "input_hash": specialist_input.input_hash,
                "review_configuration_hash": specialist_input.review_configuration_hash,
                "pipeline_configuration_hash": pipeline.configuration_hash,
                "role": config.role.value,
                "models": config.model_route().models,
                "deployments": config.model_route().deployments,
                "prompt_version": prompt.prompt_version,
                "prompt_hash": prompt.content_hash,
                "schema_version": prompt.schema_version,
            }
        )
    )


async def _current_identity(
    current_identity: Optional[Callable[[], Any]],
) -> tuple[bool, Optional[str]]:
    if current_identity is None:
        return False, None
    try:
        value = current_identity()
        if inspect.isawaitable(value):
            value = await value
        return (True, str(value)) if value else (False, None)
    except Exception as exc:
        get_logger().warning(
            "Could not refresh specialist input identity; treating the batch as unavailable",
            artifact={"error_class": type(exc).__name__},
        )
        return False, None


def _failure_state(exc: BaseException) -> tuple[SpecialistState, str]:
    cause = exc.__cause__ or exc
    if isinstance(cause, SpecialistLowConfidenceError):
        return SpecialistState.LOW_CONFIDENCE, type(cause).__name__
    if isinstance(cause, SpecialistOutputError):
        return SpecialistState.MALFORMED_OUTPUT, type(cause).__name__
    return SpecialistState.PROVIDER_FAILURE, type(cause).__name__


async def _execute_role(
    *,
    prepared_system: str,
    prepared_user: str,
    specialist_input: SpecialistInput,
    pipeline: SpecialistPipelineConfig,
    config: SpecialistRoleConfig,
    prompt: SpecialistPrompt,
    ai_handler: BaseAiHandler,
    semaphore: asyncio.Semaphore,
    input_token_reservation: int,
    output_token_reservation: int,
    cache_key: str,
) -> RoleExecution:
    started_at = time.monotonic()

    async def attempt(model: str) -> Mapping[str, Any]:
        response, _ = await ai_handler.chat_completion(
            model=model,
            system=prepared_system,
            user=prepared_user,
            temperature=0,
        )
        if not isinstance(response, str):
            raise SpecialistOutputError("specialist response must be text")
        response_tokens = TokenHandler(model=model).count_tokens(response)
        if response_tokens > config.output_token_budget:
            raise SpecialistOutputError("specialist response exceeded its output token budget")
        output = validate_specialist_output(config.role, response, specialist_input, pipeline)
        if output["confidence"] < config.minimum_confidence:
            raise SpecialistLowConfidenceError(output)
        return output

    try:
        async with semaphore:
            output = await asyncio.wait_for(
                retry_with_fallback_models(attempt, model_route=config.model_route()),
                timeout=config.timeout_seconds,
            )
    except asyncio.TimeoutError:
        return RoleExecution(
            role=config.role,
            state=SpecialistState.TIMEOUT,
            failure_reason="TimeoutError",
            latency_seconds=time.monotonic() - started_at,
            input_tokens=input_token_reservation,
            output_tokens=output_token_reservation,
            cache_key=cache_key,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        state, failure_reason = _failure_state(exc)
        cause = exc.__cause__ or exc
        rejected_output = cause.output if isinstance(cause, SpecialistLowConfidenceError) else None
        return RoleExecution(
            role=config.role,
            state=state,
            output=rejected_output,
            confidence=(float(rejected_output["confidence"]) if rejected_output is not None else None),
            failure_reason=failure_reason,
            latency_seconds=time.monotonic() - started_at,
            input_tokens=input_token_reservation,
            output_tokens=output_token_reservation,
            cache_key=cache_key,
        )
    return RoleExecution(
        role=config.role,
        state=SpecialistState.SUCCESS,
        output=output,
        confidence=float(output["confidence"]),
        latency_seconds=time.monotonic() - started_at,
        input_tokens=input_token_reservation,
        output_tokens=output_token_reservation,
        cache_key=cache_key,
    )


async def run_shadow_specialists(
    specialist_input: SpecialistInput,
    pipeline: SpecialistPipelineConfig,
    ai_handler: BaseAiHandler,
    *,
    current_identity: Optional[Callable[[], Any]] = None,
) -> SpecialistBatchResult:
    """Run enabled roles once, concurrently, without consuming or publishing outputs."""

    expected_identity = (
        specialist_input.snapshot_id
        if specialist_input.event != "pull_request"
        else specialist_input.head_sha
    )
    if pipeline.cancel_stale_inputs and current_identity is not None:
        identity_available, refreshed_identity = await _current_identity(current_identity)
        if not identity_available or refreshed_identity != expected_identity:
            state = SpecialistState.STALE if identity_available else SpecialistState.UNAVAILABLE
            failure_reason = "stale_input" if identity_available else "stable_head_identity_unavailable"
            stale_records = tuple(
                RoleExecution(role=config.role, state=state, failure_reason=failure_reason)
                for config in pipeline.roles
                if config.enabled
            )
            _record_role_results(stale_records, pipeline)
            return SpecialistBatchResult(
                snapshot_id=specialist_input.snapshot_id,
                head_sha=specialist_input.head_sha,
                input_hash=specialist_input.input_hash,
                configuration_hash=pipeline.configuration_hash,
                records=stale_records,
                role_records=_role_records(stale_records, pipeline),
                changed_path_count=len(specialist_input.changed_paths),
                hunk_count=len(specialist_input.hunks),
                stale=identity_available,
            )

    semaphore = asyncio.Semaphore(min(pipeline.max_concurrency, len(_ROLE_ORDER)))
    results: dict[SpecialistRole, RoleExecution] = {}
    tasks: dict[asyncio.Task, SpecialistRole] = {}
    task_reservations: dict[asyncio.Task, tuple[float, int, int]] = {}
    reserved_tokens = 0
    for role in _ROLE_ORDER:
        config = pipeline.role_config(role)
        prompt = pipeline.prompt(role)
        if not config.enabled:
            continue
        system, user = _render_prompt(prompt, specialist_input)
        input_tokens = max(
            TokenHandler(model=model).count_tokens(system + "\n" + user)
            for model in config.model_route().models
        )
        key = _cache_key(specialist_input, pipeline, config, prompt)
        if pipeline.cache_enabled:
            cached = _SPECIALIST_CACHE.get(key)
            if cached is not None:
                results[role] = RoleExecution(
                    role=role,
                    state=SpecialistState.CACHED,
                    output=cached.output,
                    confidence=cached.confidence,
                    cached=True,
                    cache_key=key,
                    model=cached.model,
                    deployment=cached.deployment,
                    fallback_used=cached.fallback_used,
                )
                continue
        if input_tokens > config.input_token_budget:
            results[role] = RoleExecution(
                role=role,
                state=SpecialistState.INPUT_BUDGET_EXHAUSTED,
                failure_reason="input_token_budget",
                input_tokens=input_tokens,
                output_tokens=config.output_token_budget,
                cache_key=key,
            )
            continue
        provider_calls = config.worst_case_provider_calls
        input_token_reservation = input_tokens * provider_calls
        output_token_reservation = config.output_token_budget * provider_calls
        reservation = input_token_reservation + output_token_reservation
        if reserved_tokens + reservation > pipeline.aggregate_token_budget:
            results[role] = RoleExecution(
                role=role,
                state=SpecialistState.AGGREGATE_BUDGET_EXHAUSTED,
                failure_reason="aggregate_token_budget",
                input_tokens=input_token_reservation,
                output_tokens=output_token_reservation,
                cache_key=key,
            )
            continue
        reserved_tokens += reservation
        task = asyncio.create_task(
            _execute_role(
                prepared_system=system,
                prepared_user=user,
                specialist_input=specialist_input,
                pipeline=pipeline,
                config=config,
                prompt=prompt,
                ai_handler=ai_handler,
                semaphore=semaphore,
                input_token_reservation=input_token_reservation,
                output_token_reservation=output_token_reservation,
                cache_key=key,
            )
        )
        tasks[task] = role
        task_reservations[task] = (
            time.monotonic(),
            input_token_reservation,
            output_token_reservation,
        )

    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=pipeline.aggregate_timeout_seconds)
        for task in done:
            try:
                results[tasks[task]] = task.result()
            except asyncio.CancelledError:
                results[tasks[task]] = RoleExecution(
                    role=tasks[task], state=SpecialistState.CANCELLED, failure_reason="cancelled"
                )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            for task in pending:
                role = tasks[task]
                started_at, input_token_reservation, output_token_reservation = task_reservations[task]
                details = get_run_details()
                telemetry = details.specialist_runs.get(role.value) if details is not None else None
                results[role] = RoleExecution(
                    role=role,
                    state=SpecialistState.TIMEOUT,
                    failure_reason="aggregate_timeout",
                    latency_seconds=time.monotonic() - started_at,
                    input_tokens=input_token_reservation,
                    output_tokens=output_token_reservation,
                    model=telemetry.model_used if telemetry is not None else None,
                    deployment=telemetry.deployment_id if telemetry is not None else None,
                    fallback_used=telemetry.fallback_used if telemetry is not None else False,
                )

    stale = False
    identity_unavailable = False
    if pipeline.cancel_stale_inputs and current_identity is not None:
        identity_available, refreshed_identity = await _current_identity(current_identity)
        identity_unavailable = not identity_available
        stale = identity_available and refreshed_identity != expected_identity
    ordered = tuple(results[role] for role in _ROLE_ORDER if role in results)
    if stale or identity_unavailable:
        state = SpecialistState.STALE if stale else SpecialistState.UNAVAILABLE
        failure_reason = "stale_input" if stale else "stable_head_identity_unavailable"
        ordered = tuple(
            RoleExecution(
                role=result.role,
                state=state,
                failure_reason=failure_reason,
                latency_seconds=result.latency_seconds,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_key=result.cache_key,
                model=result.model,
                deployment=result.deployment,
                fallback_used=result.fallback_used,
            )
            for result in ordered
        )
    else:
        for result in ordered:
            if result.state is not SpecialistState.SUCCESS or not pipeline.cache_enabled or not result.cache_key:
                continue
            details = get_run_details()
            telemetry = details.specialist_runs.get(result.role.value) if details is not None else None
            _SPECIALIST_CACHE.put(
                result.cache_key,
                _CachedRoleOutput(
                    output=result.output or {},
                    confidence=result.confidence or 0,
                    model=telemetry.model_used if telemetry is not None else None,
                    deployment=telemetry.deployment_id if telemetry is not None else None,
                    fallback_used=telemetry.fallback_used if telemetry is not None else False,
                ),
                pipeline.cache_max_entries,
            )
    _record_role_results(ordered, pipeline)
    get_logger().info(
        "Specialist shadow batch completed",
        artifact={
            "snapshot_id": specialist_input.snapshot_id,
            "input_hash": specialist_input.input_hash,
            "configuration_hash": pipeline.configuration_hash,
            "states": {result.role.value: result.state.value for result in ordered},
        },
    )
    return SpecialistBatchResult(
        snapshot_id=specialist_input.snapshot_id,
        head_sha=specialist_input.head_sha,
        input_hash=specialist_input.input_hash,
        configuration_hash=pipeline.configuration_hash,
        records=ordered,
        role_records=_role_records(ordered, pipeline),
        changed_path_count=len(specialist_input.changed_paths),
        hunk_count=len(specialist_input.hunks),
        stale=stale,
    )


def _record_role_results(
    results: Sequence[RoleExecution], pipeline: SpecialistPipelineConfig
) -> None:
    for result in results:
        prompt = pipeline.prompt(result.role)
        record_specialist_result(
            result.role.value,
            prompt_version=prompt.prompt_version,
            input_schema_version=prompt.input_schema_version,
            schema_version=prompt.schema_version,
            state=result.state.value,
            latency_seconds=result.latency_seconds,
            confidence=result.confidence,
            failure_reason=result.failure_reason,
            cached=result.cached,
            input_token_reservation=result.input_tokens,
            output_token_reservation=result.output_tokens,
            output=result.output,
            model=result.model,
            deployment_id=result.deployment,
            fallback_used=result.fallback_used if result.cached else None,
        )


def _role_records(
    results: Sequence[RoleExecution], pipeline: SpecialistPipelineConfig
) -> dict[str, dict[str, Any]]:
    """Freeze complete per-role records for #27 without later context lookups."""

    telemetry = specialist_runs_to_dict()
    records: dict[str, dict[str, Any]] = {}
    for result in results:
        role = result.role.value
        if role in telemetry:
            records[role] = copy.deepcopy(telemetry[role])
            continue
        prompt = pipeline.prompt(result.role)
        records[role] = {
            "role": role,
            "model": result.model,
            "deployment": result.deployment,
            "fallback_used": result.fallback_used,
            "prompt_version": prompt.prompt_version,
            "input_schema_version": prompt.input_schema_version,
            "schema_version": prompt.schema_version,
            "state": result.state.value,
            "latency_seconds": result.latency_seconds,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "ai_calls": 0,
            },
            "cost": {"status": "unavailable", "total_usd": None, "by_model_usd": {}},
            "confidence": result.confidence,
            "failure_reason": result.failure_reason,
            "cached": result.cached,
            "reservation": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            "output": copy.deepcopy(result.output),
        }
    return records
