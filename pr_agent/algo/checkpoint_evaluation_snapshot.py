"""Fail-closed loading for serialized checkpoint ``ReviewSnapshot`` artifacts.

This module is deliberately local-only.  It validates an exact, immutable snapshot
artifact before a future evaluation runner can expose its source-bearing content to
any review arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationValidationError,
    _answer_only_paths,
)
from pr_agent.algo.review_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    CoverageIssue,
    ReviewEvent,
    ReviewSnapshot,
)

MAX_REVIEW_SNAPSHOT_ARTIFACT_BYTES = 10_000_000
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


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _read_bounded_regular_file(path: Path, max_bytes: int) -> bytes:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise EvaluationValidationError("snapshot artifact byte limit must be a positive integer")
    try:
        lexical_metadata = path.lstat()
    except OSError as exc:
        raise EvaluationValidationError(f"cannot inspect ReviewSnapshot artifact: {path}") from exc
    if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISREG(lexical_metadata.st_mode):
        raise EvaluationValidationError("ReviewSnapshot artifact must be a regular file, not a symlink")
    if lexical_metadata.st_size > max_bytes:
        raise EvaluationValidationError(
            f"ReviewSnapshot artifact exceeds the {max_bytes}-byte validation limit"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationValidationError(f"cannot open ReviewSnapshot artifact: {path}") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise EvaluationValidationError("ReviewSnapshot artifact must remain a regular file")
        if (
            lexical_metadata.st_dev,
            lexical_metadata.st_ino,
        ) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise EvaluationValidationError("ReviewSnapshot artifact changed before it was opened")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise EvaluationValidationError(
                f"ReviewSnapshot artifact exceeds the {max_bytes}-byte validation limit"
            )
        final_metadata = os.fstat(descriptor)
        if (
            opened_metadata.st_size,
            opened_metadata.st_mtime_ns,
            opened_metadata.st_ctime_ns,
        ) != (
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ):
            raise EvaluationValidationError("ReviewSnapshot artifact changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise EvaluationValidationError(
                f"ReviewSnapshot artifact contains a duplicate JSON key: {key}"
            )
        value[key] = child
    return value


def _reject_non_finite_constant(value: str) -> None:
    raise EvaluationValidationError(
        f"ReviewSnapshot artifact contains a non-finite JSON number: {value}"
    )


def _load_json_object(raw: bytes) -> Mapping[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except EvaluationValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError("ReviewSnapshot artifact is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise EvaluationValidationError("ReviewSnapshot artifact must contain one JSON object")
    return value


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"ReviewSnapshot {field_name} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is not None and not isinstance(value, str):
        raise EvaluationValidationError(f"ReviewSnapshot {field_name} must be a string or null")
    return value


def _load_coverage_issues(payload: Mapping[str, Any]) -> tuple[CoverageIssue, ...]:
    values = payload.get("coverage_issues")
    if not isinstance(values, list):
        raise EvaluationValidationError("ReviewSnapshot coverage_issues must be a JSON array")
    issues: list[CoverageIssue] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise EvaluationValidationError(
                f"ReviewSnapshot coverage_issues[{index}] must be a JSON object"
            )
        unknown = sorted(set(value) - _COVERAGE_ISSUE_FIELDS)
        if unknown:
            raise EvaluationValidationError(
                f"ReviewSnapshot coverage_issues[{index}] contains unknown fields: {unknown}"
            )
        reason = _required_string(value, "reason")
        issues.append(
            CoverageIssue(
                reason=reason,
                path=_optional_string(value, "path"),
                fingerprint=_optional_string(value, "fingerprint"),
            )
        )
    return tuple(issues)


def _load_deterministic_results(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values = payload.get("deterministic_results")
    if not isinstance(values, list) or any(not isinstance(value, Mapping) for value in values):
        raise EvaluationValidationError(
            "ReviewSnapshot deterministic_results must be a JSON array of objects"
        )
    leaked_paths = _answer_only_paths(values, "ReviewSnapshot.deterministic_results")
    if leaked_paths:
        raise EvaluationValidationError(
            "answer-only fields are forbidden in a serialized ReviewSnapshot: "
            + ", ".join(leaked_paths)
        )
    return tuple(values)


def load_review_snapshot_artifact(
    path: str | Path,
    checkpoint_case: CheckpointCase,
    *,
    max_bytes: int = MAX_REVIEW_SNAPSHOT_ARTIFACT_BYTES,
) -> ReviewSnapshot:
    """Load one source-bearing snapshot only after every frozen identity matches.

    The byte hash is checked before JSON parsing.  The reconstructed snapshot id and
    event are then checked against the source-free ``CheckpointCase``.  Any ambiguity,
    unknown field, answer-only key, or schema mismatch stops before a model call.
    """

    if not isinstance(checkpoint_case, CheckpointCase):
        raise EvaluationValidationError("checkpoint_case must be a validated CheckpointCase")
    raw = _read_bounded_regular_file(Path(path), max_bytes)
    actual_artifact_hash = _sha256_bytes(raw)
    if actual_artifact_hash != checkpoint_case.snapshot_artifact_hash:
        raise EvaluationValidationError(
            "ReviewSnapshot artifact bytes do not match checkpoint snapshot_artifact_hash"
        )

    payload = _load_json_object(raw)
    unknown = sorted(set(payload) - _SNAPSHOT_FIELDS)
    missing = sorted(_SNAPSHOT_FIELDS - set(payload))
    if unknown:
        raise EvaluationValidationError(f"ReviewSnapshot artifact contains unknown fields: {unknown}")
    if missing:
        raise EvaluationValidationError(f"ReviewSnapshot artifact is missing fields: {missing}")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise EvaluationValidationError(
            f"unsupported ReviewSnapshot schema_version: {payload.get('schema_version')!r}"
        )
    changed_paths = payload.get("changed_paths")
    if not isinstance(changed_paths, list) or any(
        not isinstance(changed_path, str) or not changed_path
        for changed_path in changed_paths
    ):
        raise EvaluationValidationError("ReviewSnapshot changed_paths must be a string array")
    diff = payload.get("diff")
    if not isinstance(diff, str):
        raise EvaluationValidationError("ReviewSnapshot diff must be a string")

    try:
        event = ReviewEvent.parse(_required_string(payload, "event"))
    except ValueError as exc:
        raise EvaluationValidationError("ReviewSnapshot event is not a shipped review event") from exc
    snapshot = ReviewSnapshot(
        event=event,
        repository_root=_required_string(payload, "repository_root"),
        base_revision=_required_string(payload, "base_revision"),
        base_selector=_required_string(payload, "base_selector"),
        changed_paths=tuple(changed_paths),
        diff=diff,
        policy_version=_required_string(payload, "policy_version"),
        created_at=_required_string(payload, "created_at"),
        focus_path=_optional_string(payload, "focus_path"),
        task_intent=_optional_string(payload, "task_intent"),
        deterministic_results=_load_deterministic_results(payload),
        review_configuration_hash=_optional_string(payload, "review_configuration_hash"),
        parent_snapshot_id=_optional_string(payload, "parent_snapshot_id"),
        schema_version=payload["schema_version"],
        coverage_issues=_load_coverage_issues(payload),
    )
    supplied_snapshot_id = payload.get("snapshot_id")
    if not isinstance(supplied_snapshot_id, str) or supplied_snapshot_id != snapshot.snapshot_id:
        raise EvaluationValidationError(
            "ReviewSnapshot snapshot_id does not match its reconstructed content"
        )
    if snapshot.snapshot_id != checkpoint_case.snapshot_id:
        raise EvaluationValidationError("ReviewSnapshot snapshot_id does not match checkpoint case")
    if snapshot.event is not checkpoint_case.event:
        raise EvaluationValidationError("ReviewSnapshot event does not match checkpoint case")
    return snapshot
