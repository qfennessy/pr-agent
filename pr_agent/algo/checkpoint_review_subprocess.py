"""Process-isolated execution for immutable checkpoint review snapshots."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Optional

from pr_agent.algo.checkpoint_cost_authority import (
    CheckpointCostAuthorityError,
    FrozenCostAuthority,
    gateway_api_base_identity_hash,
    use_checkpoint_cost_authority,
)
from pr_agent.algo.checkpoint_evaluation import EvaluationStagePlan
from pr_agent.algo.checkpoint_stage_sources import (
    CheckpointStageSources,
    infer_checkpoint_arm_kind,
    use_checkpoint_stage_sources,
)
from pr_agent.algo.review_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    CoverageIssue,
    ReviewEvent,
    ReviewSnapshot,
    snapshot_review_instructions,
)
from pr_agent.algo.run_details import (
    deserialize_run_details_for_evaluation,
    serialize_run_details_for_evaluation,
)

if TYPE_CHECKING:
    from pr_agent.algo.review_configuration import ReviewConfigurationBundle

CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION = "checkpoint-review-subprocess-v5"
DEFAULT_REVIEW_SUBPROCESS_TIMEOUT_SECONDS = 180.0
MAX_REVIEW_SUBPROCESS_TIMEOUT_SECONDS = 900.0
MAX_REVIEW_SUBPROCESS_REQUEST_BYTES = 12_250_000
MAX_REVIEW_SUBPROCESS_OUTPUT_BYTES = 2_000_000
_CALLBACK_DRAIN_TIMEOUT_SECONDS = 5.0
_TRUSTED_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])
_WORKER_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('pr_agent.algo.checkpoint_review_subprocess',run_name='__main__')"
)
_PYTHON_IMPORT_ENVIRONMENT_KEYS = frozenset({
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
})
_WORKER_PROCESS_ENVIRONMENT_KEYS = frozenset({
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
})
_WORKER_CREDENTIAL_SECTIONS = (
    "anthropic",
    "aws",
    "azure_ad",
    "codestral",
    "cohere",
    "dashscope",
    "databricks",
    "deepinfra",
    "deepseek",
    "google_ai_studio",
    "groq",
    "huggingface",
    "mistral",
    "moonshot",
    "ollama",
    "openai",
    "openrouter",
    "replicate",
    "sambanova",
    "vertexai",
    "xai",
    "xiaomi_mimo",
    "zai",
)
_FAILURE_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SNAPSHOT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ANSWER_ONLY_KEYS = frozenset({
    "adjudication",
    "adjudication_hash",
    "answer",
    "clean_control",
    "earliest_opportunity",
    "expected_withdrawn_fingerprints",
    "expected_finding",
    "expected_findings",
    "finding_truth",
    "ground_truth",
    "is_clean",
    "label",
    "required_context",
    "severity",
    "truth",
    "verdict",
})
_REQUEST_FIELDS = {
    "allow_model_execution",
    "cost_authority",
    "evaluation_stage_plan",
    "review_configuration",
    "schema_version",
    "snapshot",
}
_OUTCOME_FIELDS = {
    "failure_reason_code",
    "latency_seconds",
    "review",
    "run_details",
    "schema_version",
    "snapshot_id",
    "state",
}
_SNAPSHOT_FIELDS = {
    "base_revision",
    "base_selector",
    "changed_paths",
    "coverage_issues",
    "created_at",
    "deterministic_results",
    "diff",
    "event",
    "focus_path",
    "parent_snapshot_id",
    "policy_version",
    "repository_root",
    "review_configuration_hash",
    "schema_version",
    "snapshot_id",
    "task_intent",
}
_COVERAGE_ISSUE_FIELDS = {"fingerprint", "path", "reason"}


class CheckpointReviewSubprocessState(str, Enum):
    """Terminal state returned by the subprocess boundary."""

    COMPLETED = "completed"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True)
class CheckpointReviewSubprocessOutcome:
    """Strict, bounded outcome from one isolated review attempt."""

    state: CheckpointReviewSubprocessState
    snapshot_id: Optional[str]
    review: Optional[Mapping[str, Any]] = field(default=None, repr=False)
    run_details: Optional[Mapping[str, Any]] = None
    latency_seconds: Optional[float] = None
    failure_reason_code: Optional[str] = None
    schema_version: str = CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint review subprocess outcome version")
        if self.snapshot_id is not None and not _SNAPSHOT_ID_PATTERN.fullmatch(self.snapshot_id):
            raise ValueError("checkpoint review subprocess outcome has an invalid snapshot id")
        if self.latency_seconds is not None and not _is_non_negative_finite_number(self.latency_seconds):
            raise ValueError("checkpoint review subprocess latency must be finite and non-negative")
        if self.state is CheckpointReviewSubprocessState.COMPLETED:
            if self.snapshot_id is None or self.failure_reason_code is not None:
                raise ValueError("completed checkpoint review subprocess outcomes require an id and no failure")
        else:
            if self.review is not None or self.run_details is not None:
                raise ValueError("failed checkpoint review subprocess outcomes cannot carry review data")
            if not isinstance(self.failure_reason_code, str) or not _FAILURE_REASON_PATTERN.fullmatch(
                self.failure_reason_code
            ):
                raise ValueError("failed checkpoint review subprocess outcomes require a bounded reason code")
        if self.review is not None:
            review = _validated_json_object(self.review, "review")
            object.__setattr__(self, "review", MappingProxyType(review))
        if self.run_details is not None:
            details = _validated_run_details(self.run_details)
            object.__setattr__(self, "run_details", MappingProxyType(details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "snapshot_id": self.snapshot_id,
            "review": _thaw_json(self.review),
            "run_details": _thaw_json(self.run_details),
            "latency_seconds": self.latency_seconds,
            "failure_reason_code": self.failure_reason_code,
        }


@dataclass(frozen=True)
class _CheckpointReviewSubprocessRequest:
    snapshot: ReviewSnapshot
    review_configuration: "ReviewConfigurationBundle"
    evaluation_stage_plan: tuple[EvaluationStagePlan, ...]
    allow_model_execution: bool
    cost_authority: Optional[FrozenCostAuthority] = None
    schema_version: str = CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint review subprocess request version")
        if not isinstance(self.snapshot, ReviewSnapshot):
            raise TypeError("checkpoint review subprocess request requires a ReviewSnapshot")
        from pr_agent.algo.review_configuration import ReviewConfigurationBundle

        if not isinstance(self.review_configuration, ReviewConfigurationBundle):
            raise TypeError("checkpoint review subprocess request requires a ReviewConfigurationBundle")
        object.__setattr__(self, "evaluation_stage_plan", tuple(self.evaluation_stage_plan))
        if any(not isinstance(stage, EvaluationStagePlan) for stage in self.evaluation_stage_plan):
            raise TypeError("checkpoint review subprocess request requires an EvaluationStagePlan tuple")
        if self.cost_authority is not None and not isinstance(self.cost_authority, FrozenCostAuthority):
            raise TypeError("checkpoint review subprocess request requires a FrozenCostAuthority")
        if not isinstance(self.allow_model_execution, bool):
            raise TypeError("allow_model_execution must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allow_model_execution": self.allow_model_execution,
            "snapshot": self.snapshot.to_dict(),
            "review_configuration": self.review_configuration.to_dict(),
            "evaluation_stage_plan": [stage.to_dict() for stage in self.evaluation_stage_plan],
            "cost_authority": None if self.cost_authority is None else self.cost_authority.to_dict(),
        }


def _is_non_negative_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError("non_finite_json_number")


def _decode_json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_is_not_strict_json") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}_must_be_an_object")
    return value


def _validated_json(value: Any, label: str, *, reject_answer_keys: bool = False) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label}_contains_non_finite_number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label}_contains_non_string_key")
            if reject_answer_keys and key.casefold() in _ANSWER_ONLY_KEYS:
                raise ValueError(f"{label}_contains_answer_only_key")
            result[key] = _validated_json(
                child,
                label,
                reject_answer_keys=reject_answer_keys,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _validated_json(child, label, reject_answer_keys=reject_answer_keys)
            for child in value
        ]
    raise ValueError(f"{label}_contains_non_json_value")


def _validated_json_object(value: Any, label: str, *, reject_answer_keys: bool = False) -> dict[str, Any]:
    result = _validated_json(value, label, reject_answer_keys=reject_answer_keys)
    if not isinstance(result, dict):
        raise ValueError(f"{label}_must_be_an_object")
    return result


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid_snapshot_{field_name}")
    return value


def _optional_string(payload: Mapping[str, Any], field_name: str) -> Optional[str]:
    value = payload.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"invalid_snapshot_{field_name}")
    return value


def _snapshot_from_dict(payload: Any) -> ReviewSnapshot:
    if not isinstance(payload, Mapping) or set(payload) != _SNAPSHOT_FIELDS:
        raise ValueError("invalid_snapshot_fields")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported_snapshot_version")
    changed_paths = payload.get("changed_paths")
    if (
        not isinstance(changed_paths, list)
        or any(not isinstance(path, str) or not path for path in changed_paths)
        or changed_paths != sorted(set(changed_paths))
    ):
        raise ValueError("invalid_snapshot_changed_paths")
    if not isinstance(payload.get("diff"), str):
        raise ValueError("invalid_snapshot_diff")
    try:
        event = ReviewEvent(payload.get("event"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_snapshot_event") from exc

    coverage_payload = payload.get("coverage_issues")
    if not isinstance(coverage_payload, list):
        raise ValueError("invalid_snapshot_coverage_issues")
    coverage_issues = []
    for issue in coverage_payload:
        if not isinstance(issue, Mapping) or set(issue) != _COVERAGE_ISSUE_FIELDS:
            raise ValueError("invalid_snapshot_coverage_issue")
        coverage_issues.append(
            CoverageIssue(
                reason=_required_string(issue, "reason"),
                path=_optional_string(issue, "path"),
                fingerprint=_optional_string(issue, "fingerprint"),
            )
        )

    deterministic_payload = payload.get("deterministic_results")
    if not isinstance(deterministic_payload, list):
        raise ValueError("invalid_snapshot_deterministic_results")
    deterministic_results = tuple(
        _validated_json_object(result, "deterministic_results", reject_answer_keys=True)
        for result in deterministic_payload
    )
    snapshot = ReviewSnapshot(
        event=event,
        repository_root=_required_string(payload, "repository_root"),
        base_revision=_required_string(payload, "base_revision"),
        base_selector=_required_string(payload, "base_selector"),
        changed_paths=tuple(changed_paths),
        diff=payload["diff"],
        policy_version=_required_string(payload, "policy_version"),
        created_at=_required_string(payload, "created_at"),
        focus_path=_optional_string(payload, "focus_path"),
        task_intent=_optional_string(payload, "task_intent"),
        deterministic_results=deterministic_results,
        review_configuration_hash=_optional_string(payload, "review_configuration_hash"),
        parent_snapshot_id=_optional_string(payload, "parent_snapshot_id"),
        schema_version=payload["schema_version"],
        coverage_issues=tuple(coverage_issues),
    )
    if payload.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("snapshot_id_mismatch")
    from pr_agent.algo.checkpoint_evaluation_materialize import review_snapshot_canonical_bytes

    review_snapshot_canonical_bytes(snapshot)
    return snapshot


def _request_from_dict(payload: Mapping[str, Any]) -> _CheckpointReviewSubprocessRequest:
    if set(payload) != _REQUEST_FIELDS:
        raise ValueError("invalid_request_fields")
    if payload.get("schema_version") != CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported_request_version")
    if not isinstance(payload.get("allow_model_execution"), bool):
        raise ValueError("invalid_model_execution_permission")
    from pr_agent.algo.review_configuration import ReviewConfigurationBundle

    raw_stage_plan = payload.get("evaluation_stage_plan")
    if not isinstance(raw_stage_plan, list) or any(not isinstance(stage, Mapping) for stage in raw_stage_plan):
        raise ValueError("invalid_evaluation_stage_plan")

    raw_cost_authority = payload.get("cost_authority")
    if raw_cost_authority is not None and not isinstance(raw_cost_authority, Mapping):
        raise ValueError("invalid_cost_authority")
    return _CheckpointReviewSubprocessRequest(
        snapshot=_snapshot_from_dict(payload.get("snapshot")),
        review_configuration=ReviewConfigurationBundle.from_dict(payload.get("review_configuration")),
        evaluation_stage_plan=tuple(EvaluationStagePlan.from_dict(stage) for stage in raw_stage_plan),
        cost_authority=(
            None if raw_cost_authority is None else FrozenCostAuthority.from_dict(raw_cost_authority)
        ),
        allow_model_execution=payload["allow_model_execution"],
        schema_version=payload["schema_version"],
    )


def _validated_run_details(value: Any) -> dict[str, Any]:
    return serialize_run_details_for_evaluation(
        deserialize_run_details_for_evaluation(value)
    )


def _serialize_run_details(run_details: Any) -> Optional[dict[str, Any]]:
    if run_details is None:
        return None
    return serialize_run_details_for_evaluation(run_details)


def _outcome_from_dict(payload: Mapping[str, Any]) -> CheckpointReviewSubprocessOutcome:
    if set(payload) != _OUTCOME_FIELDS:
        raise ValueError("invalid_outcome_fields")
    if payload.get("schema_version") != CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported_outcome_version")
    try:
        state = CheckpointReviewSubprocessState(payload.get("state"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_outcome_state") from exc
    return CheckpointReviewSubprocessOutcome(
        state=state,
        snapshot_id=payload.get("snapshot_id"),
        review=payload.get("review"),
        run_details=payload.get("run_details"),
        latency_seconds=payload.get("latency_seconds"),
        failure_reason_code=payload.get("failure_reason_code"),
        schema_version=payload["schema_version"],
    )


def _failure_outcome(
    state: CheckpointReviewSubprocessState,
    reason: str,
    *,
    snapshot_id: Optional[str] = None,
    latency_seconds: Optional[float] = None,
) -> CheckpointReviewSubprocessOutcome:
    return CheckpointReviewSubprocessOutcome(
        state=state,
        snapshot_id=snapshot_id,
        latency_seconds=latency_seconds,
        failure_reason_code=reason,
    )


def _current_review_configuration() -> "ReviewConfigurationBundle":
    """Materialize the current allowlisted general-review configuration once."""

    from pr_agent.algo.review_configuration import materialize_review_configuration

    return materialize_review_configuration(repo_context_files={})


def _worker_environment() -> dict[str, str]:
    """Pass only process plumbing and the supported provider credential."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _WORKER_PROCESS_ENVIRONMENT_KEYS and key not in _PYTHON_IMPORT_ENVIRONMENT_KEYS
    }
    from pr_agent.config_loader import get_settings

    api_key = get_settings().get("openai.key", None) or os.environ.get("OPENAI_API_KEY")
    if isinstance(api_key, str) and api_key:
        environment["OPENAI_API_KEY"] = api_key
    return environment


async def _execute_review(
    snapshot: ReviewSnapshot,
    review_configuration: "ReviewConfigurationBundle",
    cost_authority: Optional[FrozenCostAuthority],
) -> CheckpointReviewSubprocessOutcome:
    """Import and run production review code only after request validation."""

    expected_configuration_hash = snapshot.review_configuration_hash
    if (
        not isinstance(expected_configuration_hash, str)
        or not _SNAPSHOT_ID_PATTERN.fullmatch(expected_configuration_hash)
    ):
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "review_configuration_unverified",
            snapshot_id=snapshot.snapshot_id,
        )
    try:
        review_configuration.require_compatible_runtime()
    except Exception:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "review_configuration_unverified",
            snapshot_id=snapshot.snapshot_id,
        )
    if not hmac.compare_digest(review_configuration.configuration_hash, expected_configuration_hash):
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "review_configuration_mismatch",
            snapshot_id=snapshot.snapshot_id,
        )
    if not snapshot.diff.strip():
        return CheckpointReviewSubprocessOutcome(
            state=CheckpointReviewSubprocessState.COMPLETED,
            snapshot_id=snapshot.snapshot_id,
            review={"review": {"key_issues_to_review": []}},
            latency_seconds=0.0,
        )
    if cost_authority is None:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "cost_authority_unverified",
            snapshot_id=snapshot.snapshot_id,
        )
    try:
        cost_authority.require_active()
        if (
            not hmac.compare_digest(cost_authority.snapshot_id, snapshot.snapshot_id)
            or not hmac.compare_digest(
                cost_authority.review_configuration_hash,
                review_configuration.configuration_hash,
            )
        ):
            raise CheckpointCostAuthorityError("cost authority belongs to a different snapshot")
        checkpoint_gateway_api_base = review_configuration.checkpoint_gateway_api_base
        if checkpoint_gateway_api_base is None:
            raise CheckpointCostAuthorityError(
                "checkpoint execution requires a frozen enforcing gateway api_base"
            )
        gateway_api_base_hash = gateway_api_base_identity_hash(checkpoint_gateway_api_base)
        if any(
            not hmac.compare_digest(quote.gateway_api_base_hash, gateway_api_base_hash)
            for quote in cost_authority.quotes
        ):
            raise CheckpointCostAuthorityError(
                "checkpoint gateway api_base does not match every cost quote"
            )
    except CheckpointCostAuthorityError:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "cost_authority_unverified",
            snapshot_id=snapshot.snapshot_id,
        )

    from pr_agent.algo.review_configuration import replay_review_configuration
    from pr_agent.algo.review_execution_context import isolate_review_execution, pin_review_prompt_date
    from pr_agent.algo.review_specialists import use_specialist_snapshot_context
    from pr_agent.algo.skills_loader import pin_skills_context
    from pr_agent.config_loader import get_settings

    started = time.monotonic()
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        settings = get_settings()
        for section in _WORKER_CREDENTIAL_SECTIONS:
            try:
                settings.unset(section, force=True)
            except KeyError:
                # A missing provider section is already credential-free.
                pass
        with replay_review_configuration(review_configuration):
            # The endpoint is frozen in the source-free protocol and matched to
            # the authority above; never inherit it from the worker environment.
            settings.set("openai.api_base", checkpoint_gateway_api_base, merge=False)
            settings.set("config.git_provider", "plain-diff")
            settings.set("config.cli_mode", False)
            settings.set("config.use_repo_settings_file", False)
            settings.set("config.use_global_settings_file", False)
            settings.set("config.extra_config_url", "")
            settings.set("config.publish_output", False)
            settings.set("config.publish_output_progress", False)
            settings.set("config.enable_ai_metadata", False)
            settings.set("config.add_user_to_requests", False)
            settings.set("config.output_relevant_configurations", False)
            # This flag also enables response-cost collection. Keep it on for
            # evaluation telemetry; output_run_details and every publication sink
            # remain disabled below, so no cost footer is rendered or published.
            settings.set("config.output_run_cost", True)
            settings.set("config.output_run_details", False)
            settings.set("config.propagate_tool_errors", True)
            settings.set("litellm.enable_callbacks", False)
            settings.set("litellm.success_callback", [], merge=False)
            settings.set("litellm.failure_callback", [], merge=False)
            settings.set("litellm.service_callback", [], merge=False)
            settings.set("litellm.extra_headers", {}, merge=False)
            settings.set("litellm.turn_off_message_logging", True)
            settings.set("otel.is_enabled", False)
            settings.set("push_outputs", {}, merge=False)
            settings.set("plain_diff.content", snapshot.diff)
            settings.set("plain_diff.output_path", None)
            settings.set("plain_diff.json_output_path", None)
            settings.set("plain_diff.suppress_stdout", True)
            settings.set("plain_diff.disable_working_tree_enrichment", True)
            settings.set("plain_diff.repo_context_files", dict(review_configuration.repo_context_files))
            settings.set("config.repo_context_files", list(review_configuration.repo_context_files))
            settings.set("config.repo_context_max_lines", review_configuration.repo_context_max_lines)
            settings.set("related_tickets", [])
            existing_instructions = str(settings.get("pr_reviewer.extra_instructions", "") or "")
            settings.set(
                "pr_reviewer.extra_instructions",
                snapshot_review_instructions(snapshot, existing_instructions),
            )

            from pr_agent.algo.ai_handlers.litellm_helpers import drain_litellm_callbacks
            from pr_agent.tools.pr_reviewer import PRReviewer

            with (
                isolate_review_execution(),
                pin_review_prompt_date(review_configuration.prompt_date),
                pin_skills_context(review_configuration.skills_context),
                use_specialist_snapshot_context(snapshot, lambda: snapshot.snapshot_id),
                use_checkpoint_cost_authority(cost_authority),
            ):
                reviewer = PRReviewer("checkpoint-review-subprocess")
                # PlainDiffGitProvider enables its normal stdout publication setting
                # during construction. Re-close it inside this isolated worker.
                settings.set("config.publish_output", False)
                try:
                    execution = await reviewer._run_structured_no_publish_once()
                finally:
                    await drain_litellm_callbacks(timeout=_CALLBACK_DRAIN_TIMEOUT_SECONDS)
    latency_seconds = max(0.0, time.monotonic() - started)
    review = None if execution.review is None else _validated_json_object(execution.review, "review")
    return CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review=review,
        run_details=_serialize_run_details(execution.run_details),
        latency_seconds=latency_seconds,
    )


async def _handle_worker_request(
    raw: bytes,
    *,
    executor: Callable[
        [ReviewSnapshot, "ReviewConfigurationBundle", Optional[FrozenCostAuthority]],
        Awaitable[CheckpointReviewSubprocessOutcome],
    ] = _execute_review,
) -> CheckpointReviewSubprocessOutcome:
    """Validate a worker request completely before entering the execution seam."""

    if len(raw) > MAX_REVIEW_SUBPROCESS_REQUEST_BYTES:
        return _failure_outcome(CheckpointReviewSubprocessState.FAILED, "request_too_large")
    try:
        request = _request_from_dict(_decode_json_object(raw, "request"))
    except (KeyError, TypeError, ValueError, RecursionError):
        return _failure_outcome(CheckpointReviewSubprocessState.FAILED, "invalid_request")
    if not request.allow_model_execution:
        return _failure_outcome(
            CheckpointReviewSubprocessState.REFUSED,
            "model_execution_not_authorized",
            snapshot_id=request.snapshot.snapshot_id,
        )
    if request.snapshot.diff.strip() and request.cost_authority is None:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "cost_authority_unverified",
            snapshot_id=request.snapshot.snapshot_id,
        )
    try:
        sources = request.review_configuration.stage_sources
        if request.evaluation_stage_plan and sources is None:
            raise ValueError("checkpoint stage sources are unavailable")
        selected_sources = (
            CheckpointStageSources()
            if sources is None
            else sources.for_stage_plan(request.evaluation_stage_plan)
        )
        with use_checkpoint_stage_sources(selected_sources):
            outcome = await executor(
                request.snapshot,
                request.review_configuration,
                request.cost_authority,
            )
    except CheckpointCostAuthorityError:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "cost_authority_rejected",
            snapshot_id=request.snapshot.snapshot_id,
        )
    except Exception:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "review_execution_failed",
            snapshot_id=request.snapshot.snapshot_id,
        )
    if outcome.snapshot_id != request.snapshot.snapshot_id:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "review_snapshot_mismatch",
            snapshot_id=request.snapshot.snapshot_id,
        )
    return outcome


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            # The worker exited after the timeout; the wait below still reaps it.
            pass
        await process.wait()


async def _exchange_with_worker(
    process: asyncio.subprocess.Process,
    request: bytes,
    max_output_bytes: int,
) -> tuple[bytes, bool]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("worker pipes unavailable")
    process.stdin.write(request)
    await process.stdin.drain()
    process.stdin.close()
    wait_closed = getattr(process.stdin, "wait_closed", None)
    if wait_closed is not None:
        await wait_closed()
    chunks: list[bytes] = []
    output_size = 0
    while True:
        chunk = await process.stdout.read(min(65_536, max_output_bytes + 1 - output_size))
        if not chunk:
            break
        chunks.append(chunk)
        output_size += len(chunk)
        if output_size > max_output_bytes:
            await _stop_process(process)
            return b"", True
    await process.wait()
    return b"".join(chunks), False


async def run_checkpoint_review_subprocess(
    snapshot: ReviewSnapshot,
    *,
    review_configuration: Optional["ReviewConfigurationBundle"] = None,
    evaluation_stage_plan: tuple[EvaluationStagePlan, ...] = (),
    cost_authority: Optional[FrozenCostAuthority] = None,
    allow_model_execution: bool = False,
    timeout_seconds: float = DEFAULT_REVIEW_SUBPROCESS_TIMEOUT_SECONDS,
) -> CheckpointReviewSubprocessOutcome:
    """Run one review in a fresh interpreter, requiring explicit model permission."""

    if not isinstance(snapshot, ReviewSnapshot):
        raise TypeError("snapshot must be a ReviewSnapshot")
    if not isinstance(allow_model_execution, bool):
        raise TypeError("allow_model_execution must be a boolean")
    if not allow_model_execution:
        return _failure_outcome(
            CheckpointReviewSubprocessState.REFUSED,
            "model_execution_not_authorized",
            snapshot_id=snapshot.snapshot_id,
        )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= MAX_REVIEW_SUBPROCESS_TIMEOUT_SECONDS
    ):
        raise ValueError("timeout_seconds must be finite and within the supported bound")
    if not isinstance(evaluation_stage_plan, tuple) or any(
        not isinstance(stage, EvaluationStagePlan) for stage in evaluation_stage_plan
    ):
        raise TypeError("evaluation_stage_plan must be an EvaluationStagePlan tuple")
    if cost_authority is not None and not isinstance(cost_authority, FrozenCostAuthority):
        raise TypeError("cost_authority must be a FrozenCostAuthority")

    from pr_agent.algo.checkpoint_evaluation_materialize import review_snapshot_canonical_bytes
    from pr_agent.algo.review_configuration import (
        ReviewConfigurationBundle,
        review_configuration_canonical_bytes,
    )

    try:
        if review_configuration is None:
            review_configuration = _current_review_configuration()
        if not isinstance(review_configuration, ReviewConfigurationBundle):
            raise TypeError("review_configuration must be a ReviewConfigurationBundle")
        if not hmac.compare_digest(
            review_configuration.configuration_hash, snapshot.review_configuration_hash or ""
        ):
            return _failure_outcome(
                CheckpointReviewSubprocessState.FAILED,
                "review_configuration_mismatch",
                snapshot_id=snapshot.snapshot_id,
            )
        stage_sources = review_configuration.stage_sources
        if evaluation_stage_plan and stage_sources is None:
            return _failure_outcome(
                CheckpointReviewSubprocessState.FAILED,
                "stage_sources_unverified",
                snapshot_id=snapshot.snapshot_id,
            )
        if stage_sources is not None:
            stage_sources.validate_stage_plan(
                evaluation_stage_plan,
                arm_kind=infer_checkpoint_arm_kind(evaluation_stage_plan),
            )
        if snapshot.diff.strip():
            if cost_authority is None:
                return _failure_outcome(
                    CheckpointReviewSubprocessState.FAILED,
                    "cost_authority_unverified",
                    snapshot_id=snapshot.snapshot_id,
                )
            cost_authority.require_active()
            if (
                not hmac.compare_digest(cost_authority.snapshot_id, snapshot.snapshot_id)
                or not hmac.compare_digest(
                    cost_authority.review_configuration_hash,
                    review_configuration.configuration_hash,
                )
            ):
                raise CheckpointCostAuthorityError("cost authority belongs to a different snapshot")
        canonical_snapshot = _decode_json_object(
            review_snapshot_canonical_bytes(snapshot),
            "snapshot",
        )
        canonical_review_configuration = _decode_json_object(
            review_configuration_canonical_bytes(review_configuration),
            "review_configuration",
        )
    except CheckpointCostAuthorityError:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "cost_authority_unverified",
            snapshot_id=snapshot.snapshot_id,
        )
    except (TypeError, ValueError, RecursionError):
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "invalid_snapshot",
            snapshot_id=snapshot.snapshot_id,
        )
    request = json.dumps(
        {
            "schema_version": CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION,
            "allow_model_execution": True,
            "snapshot": canonical_snapshot,
            "review_configuration": canonical_review_configuration,
            "evaluation_stage_plan": [stage.to_dict() for stage in evaluation_stage_plan],
            "cost_authority": None if cost_authority is None else cost_authority.to_dict(),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(request) > MAX_REVIEW_SUBPROCESS_REQUEST_BYTES:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "request_too_large",
            snapshot_id=snapshot.snapshot_id,
        )
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _WORKER_BOOTSTRAP,
            _TRUSTED_PACKAGE_ROOT,
            "--worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=_TRUSTED_PACKAGE_ROOT,
            env=_worker_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_start_failed",
            snapshot_id=snapshot.snapshot_id,
        )
    started = time.monotonic()
    try:
        output, output_too_large = await asyncio.wait_for(
            _exchange_with_worker(process, request, MAX_REVIEW_SUBPROCESS_OUTPUT_BYTES),
            timeout=float(timeout_seconds),
        )
    except asyncio.TimeoutError:
        await _stop_process(process)
        return _failure_outcome(
            CheckpointReviewSubprocessState.TIMEOUT,
            "worker_timeout",
            snapshot_id=snapshot.snapshot_id,
            latency_seconds=max(0.0, time.monotonic() - started),
        )
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    except (BrokenPipeError, ConnectionError, OSError, RuntimeError):
        await _stop_process(process)
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_io_failed",
            snapshot_id=snapshot.snapshot_id,
        )
    if output_too_large:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_output_too_large",
            snapshot_id=snapshot.snapshot_id,
        )
    if process.returncode != 0:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_process_failed",
            snapshot_id=snapshot.snapshot_id,
        )
    try:
        outcome = _outcome_from_dict(_decode_json_object(output, "outcome"))
    except (KeyError, TypeError, ValueError, RecursionError):
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_protocol_failed",
            snapshot_id=snapshot.snapshot_id,
        )
    if outcome.snapshot_id != snapshot.snapshot_id:
        return _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "worker_snapshot_mismatch",
            snapshot_id=snapshot.snapshot_id,
        )
    return outcome


def _encode_worker_outcome(outcome: CheckpointReviewSubprocessOutcome) -> bytes:
    try:
        raw = json.dumps(
            outcome.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raw = json.dumps(
            _failure_outcome(CheckpointReviewSubprocessState.FAILED, "result_serialization_failed").to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    if len(raw) <= MAX_REVIEW_SUBPROCESS_OUTPUT_BYTES:
        return raw
    return json.dumps(
        _failure_outcome(
            CheckpointReviewSubprocessState.FAILED,
            "result_too_large",
            snapshot_id=outcome.snapshot_id,
        ).to_dict(),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _worker_main() -> int:
    raw = sys.stdin.buffer.read(MAX_REVIEW_SUBPROCESS_REQUEST_BYTES + 1)
    outcome = asyncio.run(_handle_worker_request(raw))
    sys.stdout.buffer.write(_encode_worker_outcome(outcome))
    sys.stdout.buffer.flush()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown or not args.worker:
        return 2
    return _worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
