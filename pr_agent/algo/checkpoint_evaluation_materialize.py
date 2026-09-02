"""Credential-free materialization of clean checkpoint-control artifacts.

The caller supplies already captured :class:`ReviewSnapshot` objects.  This module
never captures repository state, calls a model, or performs network I/O.  It only
validates those snapshots through the shared plain-diff parser and writes immutable,
private artifacts for the evaluation runner.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pr_agent.algo.checkpoint_evaluation import (CheckpointCase,
                                                 CheckpointTruth,
                                                 EvaluationArm,
                                                 EvaluationCohort,
                                                 EvaluationManifest,
                                                 EvaluationValidationError,
                                                 TruthArtifact,
                                                 _answer_only_paths,
                                                 _freeze_json, _thaw_json,
                                                 content_hash)
from pr_agent.algo.checkpoint_evaluation_cocos import \
    CHECKPOINT_CONTROL_SCHEMA_VERSION
from pr_agent.algo.local_artifact_io import (
    _open_parent_directory, read_regular_file_without_symlinks)
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewSnapshot
from pr_agent.git_providers.plain_diff_provider import parse_plain_diff

CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION = "checkpoint-control-materialization-v1"
CHECKPOINT_CONTROL_ARTIFACT_INDEX_SCHEMA_VERSION = "checkpoint-control-artifact-index-v1"
MAX_CHECKPOINT_CONTROL_SPEC_BYTES = 1_000_000
MAX_REVIEW_SNAPSHOT_ARTIFACT_BYTES = 10_000_000
_DARWIN_RENAME_EXCL = 0x00000004
_LINUX_RENAME_NOREPLACE = 0x00000001

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED_SCENARIOS = frozenset({
    "partial_correct_save",
    "coherent_clean_worktree",
    "temporary_mistake_corrected",
    "staged_checkpoint",
    "stale_candidate_withdrawn",
})
_LINEAGE_SCENARIOS = frozenset({"temporary_mistake_corrected", "stale_candidate_withdrawn"})
_BASE_ENTRY_FIELDS = frozenset({
    "id",
    "parent_id",
    "stage",
    "scenario",
    "independently_adjudicated",
    "adjudication_hash",
    "is_clean",
    "expected_withdrawn_fingerprints",
})
_MATERIALIZATION_ENTRY_FIELDS = _BASE_ENTRY_FIELDS | {"is_final_pr_head"}
_LEDGER_ENTRY_FIELDS = _BASE_ENTRY_FIELDS | {"snapshot_id", "snapshot_artifact_hash"}
_TOP_LEVEL_FIELDS = frozenset({"schema_version", "answer_only", "entries"})


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
        raise EvaluationValidationError("checkpoint materialization input must be JSON-compatible") from exc
    return (rendered + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise EvaluationValidationError(f"checkpoint control spec contains a duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_non_finite_constant(value: str) -> None:
    raise EvaluationValidationError(f"checkpoint control spec contains a non-finite JSON number: {value}")


def _load_spec(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        try:
            loaded = json.loads(_canonical_bytes(_thaw_json(value)))
        except json.JSONDecodeError as exc:  # pragma: no cover - canonical bytes are always valid JSON
            raise EvaluationValidationError("checkpoint control spec is not valid JSON") from exc
        return loaded
    raw = read_regular_file_without_symlinks(
        value,
        label="checkpoint control materialization spec",
        max_bytes=MAX_CHECKPOINT_CONTROL_SPEC_BYTES,
    )
    try:
        loaded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except EvaluationValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError("checkpoint control spec must be strict UTF-8 JSON") from exc
    if not isinstance(loaded, Mapping):
        raise EvaluationValidationError("checkpoint control spec must contain one JSON object")
    return loaded


def _require_sha256(field_name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvaluationValidationError(f"checkpoint control {field_name} must be a sha256 identity")
    return value


@dataclass(frozen=True)
class _ControlSpecEntry:
    case_id: str
    parent_id: str | None
    event: ReviewEvent
    scenario: str
    adjudication_hash: str
    expected_withdrawn_fingerprints: tuple[str, ...]
    expected_snapshot_id: str | None = None
    expected_snapshot_artifact_hash: str | None = None


def _parse_entries(payload: Mapping[str, Any]) -> tuple[_ControlSpecEntry, ...]:
    unknown_top_level = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown_top_level:
        raise EvaluationValidationError(
            f"checkpoint control materialization spec contains unknown fields: {unknown_top_level}"
        )
    schema_version = payload.get("schema_version")
    if schema_version not in {
        CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION,
        CHECKPOINT_CONTROL_SCHEMA_VERSION,
    }:
        raise EvaluationValidationError(f"unsupported checkpoint control materialization schema: {schema_version!r}")
    if payload.get("answer_only") is not True:
        raise EvaluationValidationError("checkpoint control materialization spec must be answer-only")
    leaked_paths = _answer_only_paths(
        {key: child for key, child in payload.items() if key != "entries"},
        "checkpoint_control_spec",
    )
    if leaked_paths:
        raise EvaluationValidationError("checkpoint control spec metadata leaks answer-only fields")
    entries = payload.get("entries")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise EvaluationValidationError("checkpoint control materialization spec requires an object list named entries")
    if not 15 <= len(entries) <= 20:
        raise EvaluationValidationError("checkpoint controls must contain 15 to 20 independently adjudicated cases")

    allowed_fields = (
        _MATERIALIZATION_ENTRY_FIELDS
        if schema_version == CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION
        else _LEDGER_ENTRY_FIELDS
    )
    parsed: list[_ControlSpecEntry] = []
    case_ids: set[str] = set()
    adjudication_hashes: set[str] = set()
    declared_snapshot_ids: set[str] = set()
    declared_artifact_hashes: set[str] = set()
    scenarios: set[str] = set()
    for entry in entries:
        unknown = sorted(set(entry) - allowed_fields)
        if unknown:
            raise EvaluationValidationError(f"checkpoint control contains unknown fields: {unknown}")
        case_id = entry.get("id")
        if not isinstance(case_id, str) or not _SAFE_CASE_ID.fullmatch(case_id) or case_id in case_ids:
            raise EvaluationValidationError("checkpoint control ids must be unique safe non-empty strings")
        case_ids.add(case_id)
        stage = entry.get("stage")
        if stage == "final_pr_head" or entry.get("is_final_pr_head") is True:
            raise EvaluationValidationError("final clean PR heads cannot stand in for checkpoint controls")
        if (
            schema_version == CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION
            and entry.get("is_final_pr_head") is not False
        ):
            raise EvaluationValidationError(
                "every materialization entry must explicitly declare is_final_pr_head false"
            )
        try:
            event = ReviewEvent.parse(stage) if isinstance(stage, str) else None
        except ValueError as exc:
            raise EvaluationValidationError("checkpoint control stage must be a shipped review event") from exc
        if event is None:
            raise EvaluationValidationError("checkpoint control stage must be a shipped review event")
        scenario = entry.get("scenario")
        if not isinstance(scenario, str) or scenario not in _REQUIRED_SCENARIOS:
            raise EvaluationValidationError("checkpoint controls contain an unknown or missing scenario")
        scenarios.add(scenario)
        parent_id = entry.get("parent_id")
        if parent_id is not None and (
            not isinstance(parent_id, str)
            or not _SAFE_CASE_ID.fullmatch(parent_id)
            or parent_id == case_id
        ):
            raise EvaluationValidationError("checkpoint control parent_id must name a different safe checkpoint")
        if scenario in _LINEAGE_SCENARIOS and parent_id is None:
            raise EvaluationValidationError(f"checkpoint scenario {scenario} requires lineage parent_id")
        if entry.get("independently_adjudicated") is not True or entry.get("is_clean") is not True:
            raise EvaluationValidationError(
                "checkpoint controls require an independent, affirmative clean adjudication"
            )
        adjudication_hash = _require_sha256("adjudication_hash", entry.get("adjudication_hash"))
        if adjudication_hash in adjudication_hashes:
            raise EvaluationValidationError("checkpoint control adjudication hashes must be unique")
        adjudication_hashes.add(adjudication_hash)
        withdrawn = entry.get("expected_withdrawn_fingerprints", [])
        if not isinstance(withdrawn, list) or any(
            not isinstance(fingerprint, str) or not fingerprint.strip()
            for fingerprint in withdrawn
        ):
            raise EvaluationValidationError("checkpoint withdrawn fingerprints must be a string list")
        if len(withdrawn) != len(set(withdrawn)):
            raise EvaluationValidationError("checkpoint withdrawn fingerprints must be unique within a checkpoint")
        if scenario == "stale_candidate_withdrawn" and not withdrawn:
            raise EvaluationValidationError("stale-candidate controls require a withdrawn finding fingerprint")
        expected_snapshot_id = None
        expected_artifact_hash = None
        if schema_version == CHECKPOINT_CONTROL_SCHEMA_VERSION:
            expected_snapshot_id = _require_sha256("snapshot_id", entry.get("snapshot_id"))
            expected_artifact_hash = _require_sha256(
                "snapshot_artifact_hash", entry.get("snapshot_artifact_hash")
            )
            if expected_snapshot_id in declared_snapshot_ids:
                raise EvaluationValidationError("checkpoint control snapshot ids must be unique")
            if expected_artifact_hash in declared_artifact_hashes:
                raise EvaluationValidationError("checkpoint control snapshot artifact hashes must be unique")
            declared_snapshot_ids.add(expected_snapshot_id)
            declared_artifact_hashes.add(expected_artifact_hash)
        parsed.append(
            _ControlSpecEntry(
                case_id=case_id,
                parent_id=parent_id,
                event=event,
                scenario=scenario,
                adjudication_hash=adjudication_hash,
                expected_withdrawn_fingerprints=tuple(withdrawn),
                expected_snapshot_id=expected_snapshot_id,
                expected_snapshot_artifact_hash=expected_artifact_hash,
            )
        )

    missing_scenarios = sorted(_REQUIRED_SCENARIOS - scenarios)
    if missing_scenarios:
        raise EvaluationValidationError(
            f"checkpoint controls do not cover required intermediate scenarios: {missing_scenarios}"
        )
    parents = {entry.case_id: entry.parent_id for entry in parsed}
    for case_id, parent_id in parents.items():
        if parent_id is not None and parent_id not in parents:
            raise EvaluationValidationError(f"checkpoint {case_id} names an unknown parent_id")
        visited: set[str] = set()
        current = case_id
        while parents[current] is not None:
            if current in visited:
                raise EvaluationValidationError("checkpoint control parent relationships contain a cycle")
            visited.add(current)
            current = parents[current]  # type: ignore[assignment]
    return tuple(sorted(parsed, key=lambda entry: entry.case_id))


def _validate_captured_snapshot(snapshot: ReviewSnapshot) -> None:
    if not isinstance(snapshot, ReviewSnapshot):
        raise EvaluationValidationError("checkpoint snapshots must be captured ReviewSnapshot objects")
    leaked_paths = _answer_only_paths(snapshot.deterministic_results, "ReviewSnapshot.deterministic_results")
    if leaked_paths:
        raise EvaluationValidationError(
            "answer-only fields are forbidden in a serialized ReviewSnapshot: " + ", ".join(leaked_paths)
        )
    if snapshot.coverage_issues:
        raise EvaluationValidationError("a clean checkpoint control cannot have unavailable snapshot coverage")
    try:
        parsed_files = parse_plain_diff(snapshot.diff) if snapshot.diff.strip() else []
    except ValueError as exc:
        raise EvaluationValidationError("ReviewSnapshot diff does not pass PlainDiffGitProvider parsing") from exc
    parsed_paths = {
        path
        for item in parsed_files
        for path in (getattr(item, "filename", None), getattr(item, "old_filename", None))
        if path
    }
    if parsed_paths != set(snapshot.changed_paths):
        raise EvaluationValidationError("ReviewSnapshot changed_paths do not match PlainDiffGitProvider parsing")
    for changed_path in snapshot.changed_paths:
        path = Path(changed_path)
        if path.is_absolute() or ".." in path.parts:
            raise EvaluationValidationError("ReviewSnapshot changed_paths must remain repository-relative")


def review_snapshot_canonical_bytes(snapshot: ReviewSnapshot) -> bytes:
    """Return the exact canonical bytes used by both hashing and persistence."""

    _validate_captured_snapshot(snapshot)
    payload = _canonical_bytes(snapshot.to_dict())
    if len(payload) > MAX_REVIEW_SNAPSHOT_ARTIFACT_BYTES:
        raise EvaluationValidationError(
            f"ReviewSnapshot artifact exceeds the {MAX_REVIEW_SNAPSHOT_ARTIFACT_BYTES}-byte materialization limit"
        )
    return payload


def _validate_artifact_name(name: str, label: str) -> None:
    if (
        not name
        or len(name.encode("utf-8", errors="surrogateescape")) > 128
        or "/" in name
        or name in {".", ".."}
    ):
        raise EvaluationValidationError(f"{label} must be one safe path component")


def _open_private_parent(path: Path, label: str) -> tuple[int, str]:
    directory_fd, filename = _open_parent_directory(path, label)
    try:
        _validate_artifact_name(filename, label)
        metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise EvaluationValidationError(f"{label} parent must be owner-only")
        return directory_fd, filename
    except BaseException:
        os.close(directory_fd)
        raise


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_private_directory_at(parent_fd: int, name: str, label: str) -> int:
    try:
        lexical_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise EvaluationValidationError(f"cannot inspect {label}") from exc
    if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(lexical_metadata.st_mode):
        raise EvaluationValidationError(f"{label} must be a real directory, not a symlink")
    try:
        directory_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise EvaluationValidationError(f"cannot open {label}") from exc
    opened_metadata = os.fstat(directory_fd)
    if (
        (lexical_metadata.st_dev, lexical_metadata.st_ino)
        != (opened_metadata.st_dev, opened_metadata.st_ino)
        or opened_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(opened_metadata.st_mode) != 0o700
    ):
        os.close(directory_fd)
        raise EvaluationValidationError(f"{label} must remain an owner-only directory")
    return directory_fd


def _read_descriptor(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) > max_bytes:
        raise EvaluationValidationError("existing checkpoint artifact exceeds its byte limit")
    return value


def _read_private_file_at(
    directory_fd: int,
    filename: str,
    payload: bytes,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    _validate_artifact_name(filename, "checkpoint artifact filename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise EvaluationValidationError(
            f"existing checkpoint artifact is not a safe regular file: {filename}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (
                expected_identity is not None
                and (metadata.st_dev, metadata.st_ino) != expected_identity
            )
        ):
            raise EvaluationValidationError(
                f"existing checkpoint artifact is not private and single-link: {filename}"
            )
        existing = _read_descriptor(descriptor, max(len(payload), 1))
        final_metadata = os.fstat(descriptor)
        if (
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            != (
                final_metadata.st_dev,
                final_metadata.st_ino,
                final_metadata.st_size,
                final_metadata.st_mtime_ns,
            )
        ):
            raise EvaluationValidationError(f"checkpoint artifact changed while it was read: {filename}")
        try:
            bound_metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise EvaluationValidationError(
                f"checkpoint artifact path changed while it was read: {filename}"
            ) from exc
        if (
            not stat.S_ISREG(bound_metadata.st_mode)
            or (bound_metadata.st_dev, bound_metadata.st_ino)
            != (final_metadata.st_dev, final_metadata.st_ino)
        ):
            raise EvaluationValidationError(
                f"checkpoint artifact path changed while it was read: {filename}"
            )
    finally:
        os.close(descriptor)
    if existing != payload:
        raise EvaluationValidationError(f"immutable checkpoint artifact content changed: {filename}")


def _write_staged_file(directory_fd: int, filename: str, payload: bytes) -> tuple[int, int]:
    _validate_artifact_name(filename, "checkpoint artifact filename")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise EvaluationValidationError(f"cannot create staged checkpoint artifact: {filename}") from exc
    identity: tuple[int, int] | None = None
    complete = False
    try:
        created_metadata = os.fstat(descriptor)
        identity = created_metadata.st_dev, created_metadata.st_ino
        if (
            not stat.S_ISREG(created_metadata.st_mode)
            or created_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(created_metadata.st_mode) != 0o600
            or created_metadata.st_nlink != 1
        ):
            raise EvaluationValidationError(
                f"staged checkpoint artifact is not private and single-link: {filename}"
            )
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count == 0:
                raise EvaluationValidationError(f"could not complete staged checkpoint artifact: {filename}")
            written += count
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        persisted = _read_descriptor(descriptor, max(len(payload), 1))
        final_metadata = os.fstat(descriptor)
        if (
            persisted != payload
            or final_metadata.st_size != len(payload)
            or (created_metadata.st_dev, created_metadata.st_ino)
            != (final_metadata.st_dev, final_metadata.st_ino)
            or final_metadata.st_nlink != 1
            or stat.S_IMODE(final_metadata.st_mode) != 0o600
        ):
            raise EvaluationValidationError(f"staged checkpoint artifact failed validation: {filename}")
        complete = True
        return final_metadata.st_dev, final_metadata.st_ino
    finally:
        os.close(descriptor)
        if not complete and identity is not None:
            _unlink_known_file(directory_fd, filename, identity)


def _rename_no_replace(directory_fd: int, source: str, destination: str) -> None:
    """Atomically rename one sibling path without ever replacing a destination."""

    _validate_artifact_name(source, "staged artifact name")
    _validate_artifact_name(destination, "final artifact name")
    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        rename = library.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            directory_fd,
            encoded_source,
            directory_fd,
            encoded_destination,
            _DARWIN_RENAME_EXCL,
        )
    elif hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            directory_fd,
            encoded_source,
            directory_fd,
            encoded_destination,
            _LINUX_RENAME_NOREPLACE,
        )
    else:
        raise EvaluationValidationError("this platform cannot atomically publish without replacement")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _random_staging_name(final_name: str) -> str:
    prefix = final_name[:64]
    return f".{prefix}.{secrets.token_hex(16)}.tmp"


def _unlink_known_file(directory_fd: int, filename: str, identity: tuple[int, int]) -> None:
    try:
        metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    ):
        os.unlink(filename, dir_fd=directory_fd)


def _validate_directory_binding(parent_fd: int, name: str, directory_fd: int, label: str) -> None:
    try:
        lexical_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise EvaluationValidationError(f"{label} path changed during publication") from exc
    opened_metadata = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(lexical_metadata.st_mode)
        or (lexical_metadata.st_dev, lexical_metadata.st_ino)
        != (opened_metadata.st_dev, opened_metadata.st_ino)
    ):
        raise EvaluationValidationError(f"{label} path changed during publication")


def _cleanup_staging_directory(parent_fd: int, staging_name: str, staging_fd: int) -> None:
    for filename in os.listdir(staging_fd):
        try:
            metadata = os.stat(filename, dir_fd=staging_fd, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                os.unlink(filename, dir_fd=staging_fd)
        except FileNotFoundError:
            continue
    try:
        lexical_metadata = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    opened_metadata = os.fstat(staging_fd)
    if (
        not stat.S_ISDIR(lexical_metadata.st_mode)
        or (lexical_metadata.st_dev, lexical_metadata.st_ino)
        != (opened_metadata.st_dev, opened_metadata.st_ino)
    ):
        raise EvaluationValidationError("staging directory identity changed before cleanup")
    os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_single_file(path: Path, payload: bytes) -> None:
    parent_fd, final_name = _open_private_parent(path, "checkpoint artifact")
    try:
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _read_private_file_at(parent_fd, final_name, payload)
            return
        staging_name = _random_staging_name(final_name)
        identity: tuple[int, int] | None = None
        published = False
        renamed = False
        try:
            identity = _write_staged_file(parent_fd, staging_name, payload)
            os.fsync(parent_fd)
            _rename_no_replace(parent_fd, staging_name, final_name)
            renamed = True
            _read_private_file_at(
                parent_fd,
                final_name,
                payload,
                expected_identity=identity,
            )
            os.fsync(parent_fd)
            published = True
        except FileExistsError as exc:
            raise EvaluationValidationError(
                f"checkpoint artifact appeared during atomic publication: {final_name}"
            ) from exc
        finally:
            if not published and identity is not None:
                _unlink_known_file(
                    parent_fd,
                    final_name if renamed else staging_name,
                    identity,
                )
                try:
                    os.fsync(parent_fd)
                except OSError:
                    pass
    finally:
        os.close(parent_fd)


def _verify_existing_bundle(directory_fd: int, payloads: Mapping[str, bytes]) -> None:
    actual_names = set(os.listdir(directory_fd))
    expected_names = set(payloads)
    if actual_names != expected_names:
        raise EvaluationValidationError(
            "existing checkpoint bundle is incomplete or contains unknown artifacts; "
            f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )
    for filename, payload in payloads.items():
        _read_private_file_at(directory_fd, filename, payload)


def _publish_bundle(path: Path, payloads: Mapping[str, bytes]) -> None:
    parent_fd, final_name = _open_private_parent(path, "checkpoint bundle")
    try:
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            existing_fd = _open_private_directory_at(parent_fd, final_name, "checkpoint bundle")
            try:
                _verify_existing_bundle(existing_fd, payloads)
                _validate_directory_binding(parent_fd, final_name, existing_fd, "checkpoint bundle")
            finally:
                os.close(existing_fd)
            return

        staging_name = _random_staging_name(final_name)
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise EvaluationValidationError("cannot create private checkpoint staging directory") from exc
        try:
            staging_fd = _open_private_directory_at(parent_fd, staging_name, "checkpoint staging directory")
        except BaseException:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        published = False
        renamed = False
        try:
            for filename, payload in payloads.items():
                _write_staged_file(staging_fd, filename, payload)
            _verify_existing_bundle(staging_fd, payloads)
            os.fsync(staging_fd)
            os.fsync(parent_fd)
            try:
                _rename_no_replace(parent_fd, staging_name, final_name)
            except FileExistsError as exc:
                raise EvaluationValidationError(
                    f"checkpoint bundle appeared during atomic publication: {final_name}"
                ) from exc
            renamed = True
            _verify_existing_bundle(staging_fd, payloads)
            _validate_directory_binding(parent_fd, final_name, staging_fd, "checkpoint bundle")
            os.fsync(parent_fd)
            published = True
        finally:
            if not published:
                _cleanup_staging_directory(
                    parent_fd,
                    final_name if renamed else staging_name,
                    staging_fd,
                )
                try:
                    os.fsync(parent_fd)
                except OSError:
                    pass
            os.close(staging_fd)
    finally:
        os.close(parent_fd)


def write_review_snapshot_artifact(snapshot: ReviewSnapshot, output_path: str | Path) -> str:
    """Write one exact snapshot as an immutable owner-only file and return its byte hash."""

    path = Path(output_path)
    payload = review_snapshot_canonical_bytes(snapshot)
    _publish_single_file(path, payload)
    return _sha256_bytes(payload)


@dataclass(frozen=True)
class SnapshotArtifactReference:
    case_id: str
    relative_path: str
    snapshot_id: str
    snapshot_artifact_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "relative_path": self.relative_path,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
        }


@dataclass(frozen=True)
class MaterializedCheckpointControls:
    manifest: EvaluationManifest
    truth: TruthArtifact
    checkpoint_control_ledger: Mapping[str, Any]
    artifact_index: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_control_ledger", _freeze_json(self.checkpoint_control_ledger))
        object.__setattr__(self, "artifact_index", _freeze_json(self.artifact_index))
        object.__setattr__(self, "artifact_hashes", MappingProxyType(dict(self.artifact_hashes)))

    def checkpoint_control_ledger_dict(self) -> dict[str, Any]:
        return _thaw_json(self.checkpoint_control_ledger)

    def artifact_index_dict(self) -> dict[str, Any]:
        return _thaw_json(self.artifact_index)


def materialize_checkpoint_controls(
    spec: str | Path | Mapping[str, Any],
    snapshots: Mapping[str, ReviewSnapshot],
    output_root: str | Path,
    *,
    name: str,
    policy_hash: str,
    configuration_hash: str,
    arms: Sequence[EvaluationArm],
) -> MaterializedCheckpointControls:
    """Validate and persist one clean checkpoint-control evaluation inventory.

    ``snapshots`` is an in-memory mapping populated by the production
    ``LocalPairReview`` capture path.  No repository path, provider, model, or
    credential is accepted by this API.  The caller must choose an owner-only
    output directory outside model-visible and version-controlled artifact paths.
    """

    entries = _parse_entries(_load_spec(spec))
    if not isinstance(snapshots, Mapping):
        raise EvaluationValidationError("snapshots must map checkpoint ids to ReviewSnapshot objects")
    expected_case_ids = {entry.case_id for entry in entries}
    supplied_case_ids = set(snapshots)
    if supplied_case_ids != expected_case_ids:
        raise EvaluationValidationError(
            "supplied snapshots do not match checkpoint entries; "
            f"missing={sorted(expected_case_ids - supplied_case_ids)}, "
            f"extra={sorted(supplied_case_ids - expected_case_ids)}"
        )

    references: list[SnapshotArtifactReference] = []
    snapshot_payloads: list[tuple[str, bytes]] = []
    snapshot_ids: set[str] = set()
    artifact_hashes: set[str] = set()
    all_identity_hashes: set[str] = {entry.adjudication_hash for entry in entries}
    for entry in entries:
        snapshot = snapshots[entry.case_id]
        _validate_captured_snapshot(snapshot)
        if snapshot.event is not entry.event:
            raise EvaluationValidationError(
                f"checkpoint {entry.case_id} stage does not match its ReviewSnapshot event"
            )
        if entry.parent_id is not None:
            parent_snapshot = snapshots[entry.parent_id]
            if snapshot.parent_snapshot_id != parent_snapshot.snapshot_id:
                raise EvaluationValidationError(
                    f"checkpoint {entry.case_id} ReviewSnapshot is not bound to its declared parent"
                )
        payload = review_snapshot_canonical_bytes(snapshot)
        artifact_hash = _sha256_bytes(payload)
        if snapshot.snapshot_id in snapshot_ids or snapshot.snapshot_id in all_identity_hashes:
            raise EvaluationValidationError("checkpoint snapshot identities must be unique and role-separated")
        if artifact_hash in artifact_hashes or artifact_hash in all_identity_hashes:
            raise EvaluationValidationError("checkpoint artifact hashes must be unique and role-separated")
        snapshot_ids.add(snapshot.snapshot_id)
        artifact_hashes.add(artifact_hash)
        all_identity_hashes.update({snapshot.snapshot_id, artifact_hash})
        if entry.expected_snapshot_id is not None and entry.expected_snapshot_id != snapshot.snapshot_id:
            raise EvaluationValidationError(f"checkpoint {entry.case_id} snapshot_id does not match its ledger")
        if (
            entry.expected_snapshot_artifact_hash is not None
            and entry.expected_snapshot_artifact_hash != artifact_hash
        ):
            raise EvaluationValidationError(
                f"checkpoint {entry.case_id} snapshot_artifact_hash does not match its ledger"
            )
        filename = f"review-snapshot-{snapshot.snapshot_id.removeprefix('sha256:')}.json"
        snapshot_payloads.append((filename, payload))
        references.append(
            SnapshotArtifactReference(
                case_id=entry.case_id,
                relative_path=filename,
                snapshot_id=snapshot.snapshot_id,
                snapshot_artifact_hash=artifact_hash,
            )
        )

    reference_by_id = {reference.case_id: reference for reference in references}
    ledger = {
        "schema_version": CHECKPOINT_CONTROL_SCHEMA_VERSION,
        "answer_only": True,
        "entries": [
            {
                "id": entry.case_id,
                "snapshot_id": reference_by_id[entry.case_id].snapshot_id,
                "snapshot_artifact_hash": reference_by_id[entry.case_id].snapshot_artifact_hash,
                "parent_id": entry.parent_id,
                "stage": entry.event.value,
                "scenario": entry.scenario,
                "independently_adjudicated": True,
                "adjudication_hash": entry.adjudication_hash,
                "is_clean": True,
                "expected_withdrawn_fingerprints": list(entry.expected_withdrawn_fingerprints),
            }
            for entry in entries
        ],
    }
    ledger_bytes = _canonical_bytes(ledger)
    ledger_hash = _sha256_bytes(ledger_bytes)
    corpus_hash = content_hash({
        "checkpoint_control_ledger_hash": ledger_hash,
        "snapshot_artifact_hashes": sorted(reference.snapshot_artifact_hash for reference in references),
    })
    cases = tuple(
        CheckpointCase(
            case_id=entry.case_id,
            snapshot_id=reference_by_id[entry.case_id].snapshot_id,
            snapshot_artifact_hash=reference_by_id[entry.case_id].snapshot_artifact_hash,
            event=entry.event,
            cohort=EvaluationCohort.CLEAN_CONTROL,
            parent_case_id=entry.parent_id,
            model_visible_metadata={"stage": entry.event.value},
        )
        for entry in entries
    )
    manifest = EvaluationManifest(
        name=name,
        corpus_hash=corpus_hash,
        policy_hash=policy_hash,
        configuration_hash=configuration_hash,
        cases=cases,
        arms=tuple(arms),
    )
    truth = TruthArtifact(
        manifest_id=manifest.manifest_id,
        truths=tuple(
            CheckpointTruth(
                case_id=entry.case_id,
                is_clean=True,
                adjudication_hash=entry.adjudication_hash,
            )
            for entry in entries
        ),
    )
    truth.validate_for_manifest(manifest)
    index = {
        "schema_version": CHECKPOINT_CONTROL_ARTIFACT_INDEX_SCHEMA_VERSION,
        "manifest_id": manifest.manifest_id,
        "truth_artifact_id": truth.truth_artifact_id,
        "checkpoint_control_ledger_hash": ledger_hash,
        "artifacts": [reference.to_dict() for reference in references],
    }
    named_payloads = {
        "checkpoint-controls.json": ledger_bytes,
        "evaluation-manifest.json": _canonical_bytes(manifest.to_dict()),
        "evaluation-truth.json": _canonical_bytes(truth.to_dict()),
        "snapshot-index.json": _canonical_bytes(index),
    }
    named_hashes = {
        filename: _sha256_bytes(payload)
        for filename, payload in named_payloads.items()
    }
    bundle_payloads = dict(snapshot_payloads)
    bundle_payloads.update(named_payloads)
    _publish_bundle(Path(output_root), bundle_payloads)
    return MaterializedCheckpointControls(
        manifest=manifest,
        truth=truth,
        checkpoint_control_ledger=ledger,
        artifact_index=index,
        artifact_hashes=named_hashes,
    )
