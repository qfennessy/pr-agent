"""Provider-neutral finding identities for checkpoint evaluation.

Production review output is model-controlled data.  This module converts that
data into the source-free :class:`ObservedFinding` contract without accepting a
model-supplied fingerprint, stage, or lifecycle state.  Lifecycle transitions
are derived separately from the previous checkpoint for the same evaluation arm.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from pr_agent.algo.checkpoint_evaluation import (
    EvaluationValidationError,
    FindingLifecycleState,
    FindingSeverity,
    ObservedFinding,
)

CHECKPOINT_FINDING_NORMALIZATION_SCHEMA_VERSION = "checkpoint-finding-normalization-v1"
GENERAL_REVIEW_FINDING_STAGE = "general_review"
VERIFIED_FINDING_STAGE = "candidate_verification"
FRONTIER_FINDING_STAGE = "frontier_adjudication"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MODEL_CONTROLLED_FIELDS = frozenset({"deterministic_overlap", "fingerprint", "lifecycle_state", "stage"})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(payload: Mapping[str, Any]) -> str:
    identity = {
        "schema_version": CHECKPOINT_FINDING_NORMALIZATION_SCHEMA_VERSION,
        **payload,
    }
    return f"sha256:{hashlib.sha256(_canonical_json(identity)).hexdigest()}"


def _normalize_text(value: Any, field_name: str, *, casefold: bool = False) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError(f"finding {field_name} must be a non-empty string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if casefold:
        normalized = normalized.casefold()
    if not normalized:
        raise EvaluationValidationError(f"finding {field_name} must be a non-empty string")
    return normalized


def _normalize_optional_symbol(value: Any) -> str | None:
    if value is None:
        return None
    return _normalize_text(value, "symbol")


def _normalize_path(value: Any) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError("finding relevant_file must be a safe repository-relative path")
    candidate = value.strip()
    if not candidate or len(candidate) > 512 or "\x00" in candidate or "\\" in candidate:
        raise EvaluationValidationError("finding relevant_file must be a safe repository-relative path")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvaluationValidationError("finding relevant_file must be a safe repository-relative path")
    normalized = str(path)
    if normalized != candidate:
        raise EvaluationValidationError("finding relevant_file must be a safe repository-relative path")
    return normalized


def _normalize_side(value: Any) -> str:
    if value is None:
        return "new"
    if not isinstance(value, str) or value.strip().casefold() not in {"new", "old"}:
        raise EvaluationValidationError("finding side must be 'new' or 'old'")
    return value.strip().casefold()


def _normalize_line(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationValidationError(f"finding {field_name} must be a positive integer")
    return value


def _normalize_scope(finding: Mapping[str, Any]) -> dict[str, Any]:
    start_line = _normalize_line(finding.get("start_line"), "start_line")
    end_line = _normalize_line(finding.get("end_line"), "end_line")
    if end_line < start_line:
        raise EvaluationValidationError("finding end_line cannot precede start_line")
    return {
        "path": _normalize_path(finding.get("relevant_file")),
        "side": _normalize_side(finding.get("side")),
        "start_line": start_line,
        "end_line": end_line,
        "symbol": _normalize_optional_symbol(finding.get("symbol")),
    }


def _normalize_severity(finding: Mapping[str, Any], caller_value: Any = None) -> FindingSeverity:
    if caller_value is not None:
        return _severity_value(caller_value)
    normalized_value = finding.get("normalized_severity")
    severity_value = finding.get("severity")
    if normalized_value is not None and severity_value is not None:
        normalized = _severity_value(normalized_value)
        if normalized is not _severity_value(severity_value):
            raise EvaluationValidationError("finding severity fields cannot conflict")
        return normalized
    return _severity_value(normalized_value if normalized_value is not None else severity_value)


def _severity_value(value: Any) -> FindingSeverity:
    if isinstance(value, FindingSeverity):
        return value
    if not isinstance(value, str):
        raise EvaluationValidationError("finding requires an explicit normalized severity")
    try:
        return FindingSeverity(value.strip().casefold())
    except ValueError as exc:
        raise EvaluationValidationError("finding severity must be low, medium, high, or critical") from exc


def _validate_stage(stage: str) -> str:
    if not isinstance(stage, str) or not _IDENTIFIER_PATTERN.fullmatch(stage):
        raise EvaluationValidationError("finding stage must be a lowercase identifier")
    return stage


def _validate_arm_id(arm_id: str, field_name: str) -> str:
    if not isinstance(arm_id, str) or not arm_id.strip():
        raise EvaluationValidationError(f"{field_name} must be a non-empty string")
    return arm_id.strip()


def _validate_input_finding(finding: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(finding, Mapping):
        raise EvaluationValidationError(f"finding at index {index} must be a mapping")
    forbidden = sorted(_MODEL_CONTROLLED_FIELDS.intersection(finding))
    if forbidden:
        raise EvaluationValidationError(
            f"model finding cannot control derived fields: {', '.join(forbidden)}"
        )
    return finding


def _validate_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EvaluationValidationError(f"{field_name} must be a sequence")
    return value


def _caller_values_by_index(values: Mapping[int, Any] | None, count: int, field_name: str) -> Mapping[int, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count
        for index in values
    ):
        raise EvaluationValidationError(f"{field_name} must map valid finding indexes")
    return values


def normalize_general_review_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    stage: str = GENERAL_REVIEW_FINDING_STAGE,
    severity_by_index: Mapping[int, FindingSeverity | str] | None = None,
    stable_root_cause_by_index: Mapping[int, str] | None = None,
) -> tuple[ObservedFinding, ...]:
    """Normalize production general-review findings into stable observations.

    A supplied production ``root_cause`` or caller-owned stable root cause is
    location-independent.  The legacy general-review shape lacks durable root
    cause and severity fields, so callers must supply them explicitly; mutable
    titles, explanations, headers, and locations are never substitute identity.
    """

    normalized_stage = _validate_stage(stage)
    finding_batch = _validate_sequence(findings, "general-review findings")
    caller_severity = _caller_values_by_index(severity_by_index, len(finding_batch), "severity_by_index")
    caller_root_cause = _caller_values_by_index(
        stable_root_cause_by_index, len(finding_batch), "stable_root_cause_by_index"
    )
    observations = []
    for index, raw_finding in enumerate(finding_batch):
        finding = _validate_input_finding(raw_finding, index)
        _normalize_scope(finding)
        supplied_root_cause = caller_root_cause.get(index, finding.get("root_cause"))
        identity = {
            "source": "general_review",
            "root_cause": _normalize_text(supplied_root_cause, "root_cause", casefold=True),
        }
        observations.append(ObservedFinding(
            fingerprint=_fingerprint(identity),
            severity=_normalize_severity(finding, caller_severity.get(index)),
            lifecycle_state=FindingLifecycleState.ACTIVE,
            stage=normalized_stage,
        ))
    return tuple(sorted(observations, key=lambda item: item.fingerprint))


def normalize_verified_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    stage: str = VERIFIED_FINDING_STAGE,
    severity_by_fingerprint: Mapping[str, FindingSeverity | str] | None = None,
) -> tuple[ObservedFinding, ...]:
    """Normalize post-verification findings, preserving trusted stable keys.

    The caller must pass the post-verification objects produced by issue #9's
    trusted identity pipeline.  A present ``trusted_stable_key`` is reused
    exactly.  Older verified output without that key falls back to a versioned
    hash of its trusted root-cause identity.  Current production verified
    findings do not retain normalized severity beside each finding, so their
    caller must build ``severity_by_fingerprint`` or this function fails closed.
    """

    normalized_stage = _validate_stage(stage)
    if severity_by_fingerprint is None:
        severity_by_fingerprint = {}
    if not isinstance(severity_by_fingerprint, Mapping) or any(
        not isinstance(key, str) or not _SHA256_PATTERN.fullmatch(key)
        for key in severity_by_fingerprint
    ):
        raise EvaluationValidationError("severity_by_fingerprint must use sha256 identity keys")
    observations = []
    for index, raw_finding in enumerate(_validate_sequence(findings, "verified findings")):
        finding = _validate_input_finding(raw_finding, index)
        _normalize_scope(finding)
        root_cause_id = finding.get("root_cause_id")
        if not isinstance(root_cause_id, str) or not _SHA256_PATTERN.fullmatch(root_cause_id.strip()):
            raise EvaluationValidationError("verified finding root_cause_id must be a sha256 identity")
        root_cause_id = root_cause_id.strip()
        trusted_stable_key = finding.get("trusted_stable_key")
        if trusted_stable_key is None:
            fingerprint = _fingerprint({
                "source": "verified_finding",
                "root_cause_id": root_cause_id,
            })
        elif isinstance(trusted_stable_key, str) and _SHA256_PATTERN.fullmatch(trusted_stable_key.strip()):
            fingerprint = trusted_stable_key.strip()
        else:
            raise EvaluationValidationError("verified finding trusted_stable_key must be a sha256 identity")
        observations.append(ObservedFinding(
            fingerprint=fingerprint,
            severity=_normalize_severity(finding, severity_by_fingerprint.get(fingerprint)),
            lifecycle_state=FindingLifecycleState.ACTIVE,
            stage=normalized_stage,
        ))
    return tuple(sorted(observations, key=lambda item: item.fingerprint))


def normalize_frontier_findings(
    results: Sequence[Mapping[str, Any]],
    *,
    stage: str = FRONTIER_FINDING_STAGE,
) -> tuple[ObservedFinding, ...]:
    """Convert confirmed frontier telemetry or normalized findings into observations."""

    normalized_stage = _validate_stage(stage)
    observations = []
    for index, raw_result in enumerate(_validate_sequence(results, "frontier results")):
        result = _validate_input_finding(raw_result, index)
        state = result.get("state")
        if state is not None:
            schema_version = result.get("schema_version")
            if schema_version is None:
                if (
                    state not in {"unavailable", "timeout"}
                    or "decision" in result
                    or "normalized_severity" in result
                    or result.get("publication_safe") is not False
                    or not isinstance(result.get("failure_reason"), str)
                    or not result["failure_reason"].strip()
                ):
                    raise EvaluationValidationError("synthetic frontier result is invalid")
                stable_finding_id = result.get("stable_finding_id")
                if stable_finding_id is not None and (
                    not isinstance(stable_finding_id, str) or not _SHA256_PATTERN.fullmatch(stable_finding_id)
                ):
                    raise EvaluationValidationError("frontier stable_finding_id must be a sha256 identity or null")
                continue
            if schema_version != "frontier-adjudication-output-v1":
                raise EvaluationValidationError("frontier result schema is unsupported")
            expected_decision = {
                "confirmed": "confirm",
                "rejected": "reject",
                "unavailable": "unavailable",
                "stale": "unavailable",
                "malformed_output": "unavailable",
                "timeout": "unavailable",
                "provider_failure": "unavailable",
                "not_required": "unavailable",
            }.get(state)
            if expected_decision is None or result.get("decision") != expected_decision:
                raise EvaluationValidationError("frontier result state and decision are inconsistent")
            stable_finding_id = result.get("stable_finding_id")
            if not isinstance(stable_finding_id, str) or not _SHA256_PATTERN.fullmatch(stable_finding_id):
                raise EvaluationValidationError("frontier stable_finding_id must be a sha256 identity")
            severity_value = result.get("normalized_severity")
            if state != "confirmed":
                if severity_value is not None:
                    raise EvaluationValidationError("non-confirmed frontier result cannot assign severity")
                failure_reason = result.get("failure_reason")
                if state == "rejected" and failure_reason is not None:
                    raise EvaluationValidationError("rejected frontier result cannot assign a failure reason")
                if state != "rejected" and (
                    not isinstance(failure_reason, str) or not failure_reason.strip()
                ):
                    raise EvaluationValidationError("unavailable frontier result requires a failure reason")
                continue
            if result.get("failure_reason") is not None:
                raise EvaluationValidationError("confirmed frontier result cannot assign a failure reason")
        else:
            if result.get("schema_version") != "normalized-review-finding-v1":
                raise EvaluationValidationError("frontier finding schema is unsupported")
            root_cause_id = result.get("root_cause_id")
            if not isinstance(root_cause_id, str) or not _SHA256_PATTERN.fullmatch(root_cause_id):
                raise EvaluationValidationError("frontier root_cause_id must be a sha256 identity")
            location = result.get("location")
            if not isinstance(location, Mapping):
                raise EvaluationValidationError("frontier finding location must be a mapping")
            _normalize_scope({
                "relevant_file": location.get("path"),
                "side": location.get("side"),
                "start_line": location.get("start_line"),
                "end_line": location.get("end_line"),
            })
            severity_value = result.get("severity")
        stable_finding_id = result.get("stable_finding_id")
        if not isinstance(stable_finding_id, str) or not _SHA256_PATTERN.fullmatch(stable_finding_id):
            raise EvaluationValidationError("frontier stable_finding_id must be a sha256 identity")
        observations.append(ObservedFinding(
            fingerprint=stable_finding_id,
            severity=_severity_value(severity_value),
            lifecycle_state=FindingLifecycleState.ACTIVE,
            stage=normalized_stage,
        ))
    return tuple(sorted(observations, key=lambda item: item.fingerprint))


def derive_finding_lifecycle(
    current_findings: Sequence[ObservedFinding],
    parent_findings: Sequence[ObservedFinding],
    *,
    arm_id: str,
    parent_arm_id: str,
) -> tuple[ObservedFinding, ...]:
    """Derive active and newly withdrawn findings for one checkpoint lineage.

    The caller supplies the immediately preceding checkpoint record.  Previously
    withdrawn findings are historical and are not repeated; each missing prior
    active finding is emitted once as withdrawn with its trusted metadata.
    """

    normalized_arm_id = _validate_arm_id(arm_id, "arm_id")
    if normalized_arm_id != _validate_arm_id(parent_arm_id, "parent_arm_id"):
        raise EvaluationValidationError("finding lifecycle requires the same evaluation arm")
    current = tuple(_validate_sequence(current_findings, "current findings"))
    parent = tuple(_validate_sequence(parent_findings, "parent findings"))
    if any(not isinstance(finding, ObservedFinding) for finding in (*current, *parent)):
        raise EvaluationValidationError("finding lifecycle inputs must use ObservedFinding")
    if any(finding.lifecycle_state is not FindingLifecycleState.ACTIVE for finding in current):
        raise EvaluationValidationError("current normalized findings must be active")
    current_by_fingerprint = {finding.fingerprint: finding for finding in current}
    withdrawn = tuple(
        replace(finding, lifecycle_state=FindingLifecycleState.WITHDRAWN)
        for finding in parent
        if finding.lifecycle_state is FindingLifecycleState.ACTIVE
        and finding.fingerprint not in current_by_fingerprint
    )
    result = (*current, *withdrawn)
    return tuple(sorted(result, key=lambda item: item.fingerprint))
