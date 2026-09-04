"""Immutable, credential-free configuration for checkpoint review replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional
from urllib.parse import urlsplit

from pr_agent.algo.skills_loader import get_skills_context
from pr_agent.config_loader import get_settings

if TYPE_CHECKING:
    from pr_agent.algo.checkpoint_stage_sources import CheckpointStageSources

REVIEW_CONFIGURATION_SCHEMA_VERSION = "checkpoint-review-configuration-v1"
MAX_REVIEW_CONFIGURATION_BYTES = 2_000_000
MAX_SKILLS_CONTEXT_BYTES = 500_000
MAX_REPO_CONTEXT_BYTES = 1_000_000
MAX_CONFIGURATION_DEPTH = 20
MAX_CONFIGURATION_ITEMS = 20_000
MAX_CONFIGURATION_STRING_BYTES = 1_000_000

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_HASH_PATTERN_PREFIX = "sha256:"
_MISSING = object()

_STRING_SETTINGS = frozenset({
    "config.large_patch_policy",
    "config.model",
    "config.model_reasoning",
    "config.model_weak",
    "config.reasoning_effort",
    "litellm.custom_llm_provider",
    "litellm.extra_body",
    "litellm.force_streaming_custom_llm_provider",
    "litellm.model_id",
    "openai.api_base",
    "openai.api_type",
    "openai.api_version",
    "openai.deployment_id",
    "openai.deployment_id_reasoning",
    "openai.deployment_id_weak",
    "openai.org",
    "openrouter.api_base",
    "openrouter.reasoning_effort",
    "pr_review_prompt.system",
    "pr_review_prompt.user",
    "pr_reviewer.extra_instructions",
    "pr_reviewer.review_heading",
    "pr_reviewer.review_profile",
    "config.response_language",
})
_NULLABLE_STRING_SETTINGS = frozenset({
    "config.model_reasoning",
    "config.model_weak",
    "openai.api_base",
    "openai.deployment_id",
    "openai.deployment_id_reasoning",
    "openai.deployment_id_weak",
})
_BOOLEAN_SETTINGS = frozenset({
    "config.allow_dynamic_context",
    "config.custom_reasoning_model",
    "config.duplicate_prompt_examples",
    "config.enable_claude_adaptive_thinking",
    "config.enable_claude_extended_thinking",
    "config.enable_custom_labels",
    "config.use_extra_bad_extensions",
    "litellm.drop_params",
    "litellm.disable_aiohttp",
    "openrouter.allow_fallbacks",
    "pr_reviewer.enable_candidate_verification",
    "pr_reviewer.enable_help_text",
    "pr_reviewer.enable_intro_text",
    "pr_reviewer.enable_frontier_adjudication",
    "pr_reviewer.enable_review_coverage_footer",
    "pr_reviewer.enable_review_labels_effort",
    "pr_reviewer.enable_review_labels_security",
    "pr_reviewer.final_update_message",
    "pr_reviewer.inline_key_issues",
    "pr_reviewer.publish_output_no_suggestions",
    "pr_reviewer.require_can_be_split_review",
    "pr_reviewer.require_estimate_contribution_time_cost",
    "pr_reviewer.require_estimate_effort_to_review",
    "pr_reviewer.require_merge_recommendation",
    "pr_reviewer.require_priority_files",
    "pr_reviewer.require_risk_assessment",
    "pr_reviewer.require_score_review",
    "pr_reviewer.require_security_review",
    "pr_reviewer.require_tests_review",
    "pr_reviewer.require_ticket_analysis_review",
    "pr_reviewer.require_todo_scan",
    "specialist_pipeline.enabled",
})
_INTEGER_SETTINGS = frozenset({
    "config.ai_provider_max_retries",
    "config.ai_timeout",
    "config.custom_model_max_tokens",
    "config.extended_thinking_budget_tokens",
    "config.extended_thinking_max_output_tokens",
    "config.max_extra_lines_before_dynamic_context",
    "config.max_model_tokens",
    "config.max_output_tokens",
    "config.model_retries",
    "config.patch_extra_lines_after",
    "config.patch_extra_lines_before",
    "config.seed",
    "openrouter.max_tokens",
    "openrouter.reasoning_max_tokens",
    "pr_reviewer.num_max_findings",
    "config.max_commits_tokens",
    "config.max_description_tokens",
})
_NUMBER_SETTINGS = frozenset({
    "config.model_token_count_estimate_factor",
    "config.temperature",
})
_STRING_LIST_SETTINGS = frozenset({
    "config.claude_extended_thinking_models_override",
    "config.fallback_models",
    "config.files_to_review",
    "config.ignore_language_framework",
    "config.patch_extension_skip_types",
    "litellm.force_streaming_api_base_substrings",
    "openrouter.provider_only",
    "openrouter.provider_order",
    "config.skip_keys",
})
_JSON_LIST_SETTINGS = frozenset({
    "litellm.cache_control_injection_points",
    "openai.fallback_deployments",
})
_JSON_MAPPING_SETTINGS = frozenset({
    "bad_extensions",
    "custom_labels",
    "generated_code",
    "ignore",
    "language_extension_map_org",
    "review_depth",
})
_ALLOWED_SETTING_PATHS = frozenset().union(
    _STRING_SETTINGS,
    _BOOLEAN_SETTINGS,
    _INTEGER_SETTINGS,
    _NUMBER_SETTINGS,
    _STRING_LIST_SETTINGS,
    _JSON_LIST_SETTINGS,
    _JSON_MAPPING_SETTINGS,
)
_UNSUPPORTED_ENABLED_PATHS = (
    "pr_reviewer.enable_candidate_verification",
    "pr_reviewer.enable_frontier_adjudication",
    "specialist_pipeline.enabled",
)
_UNSUPPORTED_NESTED_FLAGS = (
    ("review_depth", "consume_specialist_escalation"),
)
_SAFE_EXTRA_BODY_KEYS = frozenset({"processing_mode", "service_tier"})
_ENDPOINT_PATHS = frozenset({"openai.api_base", "openrouter.api_base"})
_OPENAI_MODEL_PREFIXES = ("chatgpt-", "gpt-", "o1", "o3", "o4", "openai/")
_UNSUPPORTED_NONEMPTY_PATHS = ("litellm.extra_headers",)


def _runtime_version() -> str:
    """Resolve the PR-Agent version without consulting the caller's cwd."""

    try:
        return version("pr-agent")
    except PackageNotFoundError:
        try:
            with (_PACKAGE_ROOT / "pyproject.toml").open("rb") as file:
                project = tomllib.load(file).get("project", {})
            package_version = project.get("version")
        except (OSError, tomllib.TOMLDecodeError):
            package_version = None
        return package_version if isinstance(package_version, str) else "unknown"


def _runtime_artifact_hash() -> str:
    """Identify executable review code even when two commits share a package version."""

    candidates = [
        _PACKAGE_ROOT / "pyproject.toml",
        _PACKAGE_ROOT / "uv.lock",
        *(_PACKAGE_ROOT / "pr_agent").rglob("*.py"),
        *(_PACKAGE_ROOT / "pr_agent").rglob("*.toml"),
    ]
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        if path.name.startswith(".secrets") or not path.is_file():
            continue
        relative = path.relative_to(_PACKAGE_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError("review runtime identity is unavailable") from exc
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    lock_path = _PACKAGE_ROOT / "uv.lock"
    if not lock_path.is_file():
        raise ValueError("review dependency identity is unavailable")
    try:
        with lock_path.open("rb") as file:
            locked = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("review dependency identity is unavailable") from exc
    package_names = sorted({
        package["name"]
        for package in locked.get("package", ())
        if isinstance(package, Mapping) and isinstance(package.get("name"), str)
    })
    for package_name in package_names:
        try:
            installed_version = version(package_name)
        except PackageNotFoundError:
            installed_version = "missing"
        identity = f"{package_name.casefold()}=={installed_version}".encode("utf-8")
        digest.update(len(identity).to_bytes(4, "big"))
        digest.update(identity)
    return _HASH_PATTERN_PREFIX + digest.hexdigest()


def _is_hash(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_HASH_PATTERN_PREFIX):
        return False
    suffix = value[len(_HASH_PATTERN_PREFIX):]
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)


def _validate_json(value: Any, label: str, *, depth: int = 0, counter: list[int] | None = None) -> Any:
    if depth > MAX_CONFIGURATION_DEPTH:
        raise ValueError(f"{label} exceeds the maximum nesting depth")
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_CONFIGURATION_ITEMS:
        raise ValueError(f"{label} contains too many values")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_CONFIGURATION_STRING_BYTES:
            raise ValueError(f"{label} contains an oversized string")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        seen: set[str] = set()
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} contains an invalid key")
            folded = key.casefold()
            if folded in seen:
                raise ValueError(f"{label} contains case-colliding keys")
            seen.add(folded)
            result[key] = _validate_json(child, label, depth=depth + 1, counter=counter)
        return result
    if isinstance(value, (list, tuple)):
        return [_validate_json(child, label, depth=depth + 1, counter=counter) for child in value]
    raise ValueError(f"{label} contains a non-JSON value")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_endpoint(value: str, label: str) -> None:
    if not value:
        return
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a credential-free HTTP endpoint")


def _validate_setting(path: str, value: Any) -> Any:
    if path in _BOOLEAN_SETTINGS and isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            value = True
        elif normalized in {"0", "false", "no", "off"}:
            value = False
    elif path in _INTEGER_SETTINGS and isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            # Preserve invalid text so strict type validation below rejects it.
            pass
    elif path in _NUMBER_SETTINGS and isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            # Preserve invalid text so strict type validation below rejects it.
            pass
    elif path in _STRING_LIST_SETTINGS and isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    elif path in _JSON_LIST_SETTINGS and isinstance(value, str):
        stripped = value.strip()
        try:
            decoded = json.loads(stripped) if stripped.startswith("[") else None
        except json.JSONDecodeError:
            decoded = None
        value = decoded if isinstance(decoded, list) else [item.strip() for item in value.split(",") if item.strip()]
    value = _validate_json(value, path)
    if path in _STRING_SETTINGS:
        if not isinstance(value, str) and not (path in _NULLABLE_STRING_SETTINGS and value is None):
            raise ValueError(f"{path} must be a string")
    elif path in _BOOLEAN_SETTINGS:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
    elif path in _INTEGER_SETTINGS:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
    elif path in _NUMBER_SETTINGS:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path} must be numeric")
    elif path in _STRING_LIST_SETTINGS:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{path} must be a list of strings")
    elif path in _JSON_LIST_SETTINGS:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
    elif path in _JSON_MAPPING_SETTINGS:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
    else:
        raise ValueError(f"unsupported review configuration path: {path}")
    if path in _ENDPOINT_PATHS:
        _validate_endpoint(value, path)
    if path == "litellm.extra_body" and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("litellm.extra_body must be valid JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) - _SAFE_EXTRA_BODY_KEYS:
            raise ValueError("litellm.extra_body contains unsupported keys")
    return value


def _validate_supported_route(settings: Mapping[str, Any]) -> None:
    models = [
        settings.get("config.model", ""),
        settings.get("config.model_weak", ""),
        settings.get("config.model_reasoning", ""),
        *settings.get("config.fallback_models", ()),
    ]
    if any(model and not model.casefold().startswith(_OPENAI_MODEL_PREFIXES) for model in models):
        raise ValueError("review configuration uses an unsupported model provider")
    if settings.get("litellm.custom_llm_provider", "") or settings.get(
        "litellm.force_streaming_custom_llm_provider", ""
    ):
        raise ValueError("review configuration uses an unsupported custom provider")
    if settings.get("litellm.model_id", ""):
        raise ValueError("review configuration uses an unsupported provider model id")
    api_type = settings.get("openai.api_type", "")
    if api_type and str(api_type).casefold() not in {"openai", "open_ai"}:
        raise ValueError("review configuration uses an unsupported OpenAI API type")
    deployment_paths = (
        "openai.deployment_id",
        "openai.deployment_id_reasoning",
        "openai.deployment_id_weak",
    )
    if any(settings.get(path, "") for path in deployment_paths) or settings.get("openai.fallback_deployments", ()):
        raise ValueError("review configuration uses an unsupported deployment route")
    api_base = settings.get("openai.api_base", "")
    if api_base:
        parsed = urlsplit(api_base)
        if parsed.scheme != "https" or parsed.hostname != "api.openai.com" or parsed.port is not None:
            raise ValueError("review configuration uses an unsupported OpenAI endpoint")
    if settings.get("openrouter.api_base", ""):
        raise ValueError("review configuration uses an unsupported OpenRouter endpoint")


def _repo_context_payload(repo_context_files: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"path": path, "content": content}
        for path, content in repo_context_files.items()
    ]


def _repo_context_from_payload(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        raise ValueError("invalid review configuration repository context")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "content"}:
            raise ValueError("invalid review configuration repository context")
        path = entry.get("path")
        content = entry.get("content")
        if not isinstance(path, str) or not path or not isinstance(content, str) or path.casefold() in seen:
            raise ValueError("invalid review configuration repository context")
        seen.add(path.casefold())
        result[path] = content
    return result


def _setting_value(settings: Any, path: str) -> Any:
    return settings.get(path, _MISSING)


def _unset_setting(settings: Any, path: str) -> None:
    if "." not in path:
        unset = getattr(settings, "unset", None)
        if unset is None:
            return
        try:
            unset(path, force=True)
        except KeyError:
            # A missing setting already has the required replay state.
            pass
        return
    section_name, leaf_name = path.split(".", 1)
    section = settings.get(section_name, None)
    if section is None or "." in leaf_name:
        return
    for stored_key in list(section.keys()):
        if str(stored_key).casefold() == leaf_name.casefold():
            section.pop(stored_key, None)
            return


def _load_checkpoint_stage_sources(value: Mapping[str, Any]) -> CheckpointStageSources:
    from pr_agent.algo.checkpoint_stage_sources import CheckpointStageSources

    return CheckpointStageSources.from_dict(value)


@dataclass(frozen=True)
class ReviewConfigurationBundle:
    """Exact model-visible general-review settings without credentials or sinks."""

    runtime_version: str
    runtime_artifact_hash: str
    settings: Mapping[str, Any]
    missing_settings: tuple[str, ...]
    skills_context: str
    repo_context_files: Mapping[str, str]
    repo_context_max_lines: int
    prompt_date: str = ""
    stage_sources: Optional[CheckpointStageSources] = field(default=None, repr=False)
    schema_version: str = REVIEW_CONFIGURATION_SCHEMA_VERSION
    configuration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_CONFIGURATION_SCHEMA_VERSION:
            raise ValueError("unsupported review configuration schema version")
        if not isinstance(self.runtime_version, str) or not self.runtime_version:
            raise ValueError("review configuration runtime version is invalid")
        if not _is_hash(self.runtime_artifact_hash):
            raise ValueError("review configuration runtime artifact hash is invalid")
        if not isinstance(self.prompt_date, str) or len(self.prompt_date) > 32:
            raise ValueError("review configuration prompt date is invalid")
        if not isinstance(self.skills_context, str):
            raise ValueError("review configuration skills context is invalid")
        if len(self.skills_context.encode("utf-8")) > MAX_SKILLS_CONTEXT_BYTES:
            raise ValueError("review configuration skills context is too large")
        if self.stage_sources is not None:
            from pr_agent.algo.checkpoint_stage_sources import CheckpointStageSources

            if not isinstance(self.stage_sources, CheckpointStageSources):
                raise TypeError("review configuration stage_sources must use CheckpointStageSources")

        raw_settings = dict(self.settings)
        if set(raw_settings) - _ALLOWED_SETTING_PATHS:
            raise ValueError("review configuration contains unsupported settings")
        validated_settings = {
            path: _validate_setting(path, value)
            for path, value in raw_settings.items()
        }
        missing = tuple(self.missing_settings)
        if missing != tuple(sorted(set(missing))):
            raise ValueError("review configuration missing_settings must be sorted and unique")
        if set(missing) - _ALLOWED_SETTING_PATHS or set(missing) & set(validated_settings):
            raise ValueError("review configuration missing_settings is invalid")
        if set(missing) | set(validated_settings) != _ALLOWED_SETTING_PATHS:
            raise ValueError("review configuration must account for every supported setting")
        _validate_supported_route(validated_settings)

        raw_repo_context = dict(self.repo_context_files)
        if any(
            not isinstance(path, str)
            or not path
            or not isinstance(content, str)
            for path, content in raw_repo_context.items()
        ):
            raise ValueError("review configuration repository context is invalid")
        repo_context = _validate_json(raw_repo_context, "repo_context_files")
        if (
            not isinstance(self.repo_context_max_lines, int)
            or isinstance(self.repo_context_max_lines, bool)
            or not 0 <= self.repo_context_max_lines <= 1_000_000
        ):
            raise ValueError("review configuration repository context line budget is invalid")
        if len(_canonical_json_bytes(_repo_context_payload(repo_context))) > MAX_REPO_CONTEXT_BYTES:
            raise ValueError("review configuration repository context is too large")

        if any(validated_settings.get(path, False) is not False for path in _UNSUPPORTED_ENABLED_PATHS):
            raise ValueError("review configuration enables an unsupported production stage")
        for section_path, flag_name in _UNSUPPORTED_NESTED_FLAGS:
            section = validated_settings.get(section_path)
            if isinstance(section, Mapping) and section.get(flag_name, False) is not False:
                raise ValueError("review configuration enables unsupported specialist routing")

        object.__setattr__(self, "settings", _freeze_json(validated_settings))
        object.__setattr__(self, "missing_settings", missing)
        object.__setattr__(self, "repo_context_files", _freeze_json(repo_context))
        configuration_hash = _HASH_PATTERN_PREFIX + hashlib.sha256(
            _canonical_json_bytes(self._identity_payload())
        ).hexdigest()
        object.__setattr__(self, "configuration_hash", configuration_hash)
        if len(review_configuration_canonical_bytes(self)) > MAX_REVIEW_CONFIGURATION_BYTES:
            raise ValueError("review configuration bundle is too large")

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "runtime_version": self.runtime_version,
            "runtime_artifact_hash": self.runtime_artifact_hash,
            "settings": _thaw_json(self.settings),
            "missing_settings": list(self.missing_settings),
            "skills_context": self.skills_context,
            "repo_context_files": _repo_context_payload(self.repo_context_files),
            "repo_context_max_lines": self.repo_context_max_lines,
            "prompt_date": self.prompt_date,
        }
        if self.stage_sources is not None:
            payload["stage_sources"] = self.stage_sources.to_dict()
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "configuration_hash": self.configuration_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewConfigurationBundle":
        legacy_fields = {
            "schema_version",
            "runtime_version",
            "runtime_artifact_hash",
            "settings",
            "missing_settings",
            "skills_context",
            "repo_context_files",
            "repo_context_max_lines",
            "prompt_date",
            "configuration_hash",
        }
        extended_fields = legacy_fields | {"stage_sources"}
        if not isinstance(value, Mapping) or (
            set(value) != legacy_fields and set(value) != extended_fields
        ):
            raise ValueError("invalid review configuration fields")
        if not isinstance(value.get("settings"), Mapping):
            raise ValueError("invalid review configuration settings")
        if not isinstance(value.get("missing_settings"), list):
            raise ValueError("invalid review configuration missing_settings")
        repo_context_files = _repo_context_from_payload(value.get("repo_context_files"))
        has_stage_sources = set(value) == extended_fields
        raw_stage_sources = value.get("stage_sources")
        if has_stage_sources and not isinstance(raw_stage_sources, Mapping):
            raise ValueError("invalid review configuration stage_sources")
        bundle = cls(
            runtime_version=value.get("runtime_version"),
            runtime_artifact_hash=value.get("runtime_artifact_hash"),
            settings=value["settings"],
            missing_settings=tuple(value["missing_settings"]),
            skills_context=value.get("skills_context"),
            repo_context_files=repo_context_files,
            repo_context_max_lines=value.get("repo_context_max_lines"),
            prompt_date=value.get("prompt_date"),
            stage_sources=_load_checkpoint_stage_sources(raw_stage_sources) if has_stage_sources else None,
            schema_version=value.get("schema_version"),
        )
        if value.get("configuration_hash") != bundle.configuration_hash:
            raise ValueError("review configuration hash mismatch")
        return bundle

    def require_compatible_runtime(self) -> None:
        if self.runtime_version != _runtime_version() or self.runtime_artifact_hash != _runtime_artifact_hash():
            raise ValueError("review configuration runtime mismatch")


def materialize_review_configuration(
    skills_context: Optional[str] = None,
    repo_context_files: Optional[Mapping[str, str]] = None,
    *,
    repo_context_max_lines: Optional[int] = None,
    prompt_date: str = "",
    stage_sources: Optional[CheckpointStageSources] = None,
) -> ReviewConfigurationBundle:
    """Capture the allowlisted review inputs once without reading credential fields."""

    settings = get_settings()
    if stage_sources is not None:
        stage_sources = stage_sources.for_checkpoint_replay(settings)
    for path in _UNSUPPORTED_NONEMPTY_PATHS:
        if settings.get(path, None):
            raise ValueError(f"review configuration contains unsupported setting: {path}")
    captured: dict[str, Any] = {}
    missing: list[str] = []
    for path in sorted(_ALLOWED_SETTING_PATHS):
        value = _setting_value(settings, path)
        if value is _MISSING:
            missing.append(path)
            continue
        captured[path] = copy.deepcopy(value)
    if repo_context_max_lines is None:
        try:
            repo_context_max_lines = max(0, int(settings.get("config.repo_context_max_lines", 500)))
        except (TypeError, ValueError):
            repo_context_max_lines = 500
    return ReviewConfigurationBundle(
        runtime_version=_runtime_version(),
        runtime_artifact_hash=_runtime_artifact_hash(),
        settings=captured,
        missing_settings=tuple(missing),
        skills_context=get_skills_context() if skills_context is None else skills_context,
        repo_context_files={} if repo_context_files is None else repo_context_files,
        repo_context_max_lines=repo_context_max_lines,
        prompt_date=prompt_date,
        stage_sources=stage_sources,
    )


def review_configuration_canonical_bytes(bundle: ReviewConfigurationBundle) -> bytes:
    if not isinstance(bundle, ReviewConfigurationBundle):
        raise TypeError("review configuration must be a ReviewConfigurationBundle")
    return _canonical_json_bytes(bundle.to_dict())


def review_configuration_artifact_name(configuration_hash: str) -> str:
    """Return the deterministic local filename for one configuration bundle."""

    if not _is_hash(configuration_hash):
        raise ValueError("review configuration artifact hash is invalid")
    return f"review-configuration-{configuration_hash.removeprefix(_HASH_PATTERN_PREFIX)}.json"


@contextmanager
def replay_review_configuration(bundle: ReviewConfigurationBundle) -> Iterator[None]:
    """Install only the captured review settings for one isolated worker execution."""

    if not isinstance(bundle, ReviewConfigurationBundle):
        raise TypeError("review configuration must be a ReviewConfigurationBundle")
    bundle.require_compatible_runtime()
    settings = get_settings()
    originals = {}
    for path in _ALLOWED_SETTING_PATHS:
        value = _setting_value(settings, path)
        originals[path] = _MISSING if value is _MISSING else copy.deepcopy(value)
    try:
        for path in bundle.missing_settings:
            _unset_setting(settings, path)
        for path, value in bundle.settings.items():
            settings.set(path, copy.deepcopy(_thaw_json(value)), merge=False)
        yield
    finally:
        for path, value in originals.items():
            if value is _MISSING:
                _unset_setting(settings, path)
            else:
                settings.set(path, value, merge=False)


def snapshot_review_configuration_hash(
    skills_context: Optional[str] = None,
    repo_context_files: Optional[Mapping[str, str]] = None,
) -> str:
    """Preserve the provider-neutral configuration identity used by local snapshot review."""

    settings = get_settings()
    credential_names = {"key", "token", "secret", "password", "credential", "credentials", "private"}
    credential_suffixes = (
        "_api_key",
        "_token",
        "_access_token",
        "_private_token",
        "_client_secret",
        "_webhook_secret",
        "_password",
        "_private_key",
        "_secret_access_key",
        "_auth_header",
        "_authorization",
        "_credential",
        "_credentials",
    )
    transient_config_keys = {"cli_mode", "git_provider", "publish_output", "propagate_tool_errors"}

    def sanitized(value: Any, *, section: str = "") -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in credential_names or normalized.endswith(credential_suffixes):
                    continue
                if section == "config" and normalized in transient_config_keys:
                    continue
                cleaned[str(key)] = sanitized(child, section=section)
            return cleaned
        if isinstance(value, (list, tuple)):
            return [sanitized(item, section=section) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    all_settings = settings.as_dict()
    all_settings.pop("PLAIN_DIFF", None)
    effective = {
        "runtime_version": _runtime_version(),
        "skills_context_sha256": hashlib.sha256(
            (get_skills_context() if skills_context is None else skills_context).encode("utf-8")
        ).hexdigest(),
        "repo_context_files": repo_context_files or {},
        "settings": {
            str(section): sanitized(contents, section=str(section).lower())
            for section, contents in all_settings.items()
        },
    }
    payload = json.dumps(effective, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return _HASH_PATTERN_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()
