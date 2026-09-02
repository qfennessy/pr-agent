"""Privacy-safe, opt-in shadow telemetry that never blocks a review checkpoint."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import stat
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from pr_agent.algo.checkpoint_evaluation import (EVALUATION_SCHEMA_VERSION,
                                                 EvaluationArm,
                                                 EvaluationRunRecord,
                                                 EvaluationRunState,
                                                 EvaluationStageRun,
                                                 EvaluationValidationError,
                                                 FindingLifecycleState,
                                                 MeasurementStatus,
                                                 NumericMeasurement,
                                                 content_hash,
                                                 validate_run_model_telemetry)
from pr_agent.algo.review_snapshot import ReviewEvent

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVIEW_DEPTHS = {"quick", "standard", "deep"}
SHADOW_JOURNAL_RECORD_SCHEMA_VERSION = "checkpoint-shadow-journal-record-v2"


def _reject_unknown_fields(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvaluationValidationError(f"{name} contains unknown fields: {unknown}")


class ShadowSubmitStatus(str, Enum):
    DISABLED = "disabled"
    QUEUED = "queued"
    DROPPED = "dropped"
    CLOSED = "closed"


@dataclass(frozen=True)
class ShadowFinding:
    fingerprint_hash: str
    lifecycle_state: FindingLifecycleState

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.fingerprint_hash):
            raise EvaluationValidationError("shadow finding fingerprint_hash must be a sha256 identity")
        if not isinstance(self.lifecycle_state, FindingLifecycleState):
            raise EvaluationValidationError("shadow finding lifecycle_state must be a FindingLifecycleState")

    @classmethod
    def from_fingerprint(cls, fingerprint: str, lifecycle_state: FindingLifecycleState) -> "ShadowFinding":
        return cls(content_hash({"fingerprint": fingerprint}), lifecycle_state)

    def to_dict(self) -> dict[str, str]:
        return {
            "fingerprint_hash": self.fingerprint_hash,
            "lifecycle_state": self.lifecycle_state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowFinding":
        _reject_unknown_fields("shadow finding", value, {"fingerprint_hash", "lifecycle_state"})
        return cls(
            fingerprint_hash=value["fingerprint_hash"],
            lifecycle_state=FindingLifecycleState(value["lifecycle_state"]),
        )


@dataclass(frozen=True)
class ShadowJournalEntry:
    """Allowlisted telemetry only; source, diffs, prompts, and provider request ids cannot fit."""

    snapshot_id: str
    event: ReviewEvent
    policy_hash: str
    configuration_hash: str
    result_state: EvaluationRunState
    parent_snapshot_id: Optional[str] = None
    arm_id: Optional[str] = None
    selected_depth: Optional[str] = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    model_id: Optional[str] = None
    provider_id: Optional[str] = None
    model_revision: Optional[str] = None
    stage_runs: tuple[EvaluationStageRun, ...] = field(default_factory=tuple)
    findings: tuple[ShadowFinding, ...] = field(default_factory=tuple)
    coverage_status: MeasurementStatus = MeasurementStatus.UNAVAILABLE
    latency_seconds: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    tokens: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    cost_usd: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    retry_count: int = 0
    cached: bool = False
    schema_version: str = EVALUATION_SCHEMA_VERSION
    entry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported shadow journal schema_version: {self.schema_version}")
        for name, value in (
            ("snapshot_id", self.snapshot_id),
            ("policy_hash", self.policy_hash),
            ("configuration_hash", self.configuration_hash),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise EvaluationValidationError(f"shadow {name} must be a sha256 identity")
        if self.parent_snapshot_id is not None and (
            not _SHA256.fullmatch(self.parent_snapshot_id)
        ):
            raise EvaluationValidationError("shadow parent_snapshot_id must be a sha256 identity")
        if not isinstance(self.event, ReviewEvent):
            raise EvaluationValidationError("shadow event must be a ReviewEvent")
        if not isinstance(self.result_state, EvaluationRunState):
            raise EvaluationValidationError("shadow result_state must be an EvaluationRunState")
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(self.reason_codes)))
        if any(not isinstance(reason, str) or not _REASON_CODE.fullmatch(reason) for reason in self.reason_codes):
            raise EvaluationValidationError("shadow reasons must be machine-readable reason codes")
        object.__setattr__(self, "findings", tuple(self.findings))
        if any(not isinstance(finding, ShadowFinding) for finding in self.findings):
            raise EvaluationValidationError("shadow findings must use ShadowFinding")
        if not isinstance(self.coverage_status, MeasurementStatus):
            raise EvaluationValidationError("shadow coverage_status must be a MeasurementStatus")
        if (
            self.result_state is not EvaluationRunState.COMPLETED
            and self.coverage_status is MeasurementStatus.COMPLETE
        ):
            raise EvaluationValidationError("failed shadow runs cannot claim complete coverage")
        if not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise EvaluationValidationError("shadow retry_count must be a non-negative integer")
        if not isinstance(self.cached, bool):
            raise EvaluationValidationError("shadow cached must be a boolean")
        for name, measurement in (
            ("latency_seconds", self.latency_seconds),
            ("tokens", self.tokens),
            ("cost_usd", self.cost_usd),
        ):
            if not isinstance(measurement, NumericMeasurement):
                raise EvaluationValidationError(f"shadow {name} must use NumericMeasurement")
        for name in ("arm_id", "selected_depth", "model_id", "provider_id", "model_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise EvaluationValidationError(f"shadow {name} must be a non-empty string or null")
        aggregate_identity = (self.model_id, self.provider_id, self.model_revision)
        if any(value is not None for value in aggregate_identity) and not all(
            value is not None for value in aggregate_identity
        ):
            raise EvaluationValidationError("shadow model identity must be a complete triple or null")
        object.__setattr__(self, "stage_runs", tuple(self.stage_runs))
        if any(not isinstance(stage_run, EvaluationStageRun) for stage_run in self.stage_runs):
            raise EvaluationValidationError("shadow stage_runs must use EvaluationStageRun")
        stage_names = [stage_run.stage for stage_run in self.stage_runs]
        if len(stage_names) != len(set(stage_names)):
            raise EvaluationValidationError("shadow stage_runs must contain unique stage identities")
        if (
            self.coverage_status is MeasurementStatus.COMPLETE
            and any(stage.coverage_status is not MeasurementStatus.COMPLETE for stage in self.stage_runs)
        ):
            raise EvaluationValidationError("shadow coverage cannot be complete with an uncovered stage")
        if self.selected_depth is not None and self.selected_depth not in _REVIEW_DEPTHS:
            raise EvaluationValidationError("shadow selected_depth must be quick, standard, or deep")
        object.__setattr__(self, "entry_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "event": self.event.value,
            "policy_hash": self.policy_hash,
            "configuration_hash": self.configuration_hash,
            "arm_id": self.arm_id,
            "selected_depth": self.selected_depth,
            "reason_codes": list(self.reason_codes),
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "stage_runs": [stage_run.to_dict() for stage_run in self.stage_runs],
            "result_state": self.result_state.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "coverage_status": self.coverage_status.value,
            "latency_seconds": self.latency_seconds.to_dict(),
            "tokens": self.tokens.to_dict(),
            "cost_usd": self.cost_usd.to_dict(),
            "retry_count": self.retry_count,
            "cached": self.cached,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "entry_id": self.entry_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowJournalEntry":
        _reject_unknown_fields("shadow journal entry", value, {
            "schema_version", "snapshot_id", "parent_snapshot_id", "event", "policy_hash",
            "configuration_hash", "arm_id", "selected_depth", "reason_codes", "model_id", "provider_id",
            "model_revision", "stage_runs", "result_state", "findings", "coverage_status", "latency_seconds",
            "tokens", "cost_usd", "retry_count", "cached", "entry_id",
        })
        entry = cls(
            snapshot_id=value["snapshot_id"],
            parent_snapshot_id=value.get("parent_snapshot_id"),
            event=ReviewEvent.parse(value["event"]),
            policy_hash=value["policy_hash"],
            configuration_hash=value["configuration_hash"],
            arm_id=value.get("arm_id"),
            selected_depth=value.get("selected_depth"),
            reason_codes=tuple(value.get("reason_codes", ())),
            model_id=value.get("model_id"),
            provider_id=value.get("provider_id"),
            model_revision=value.get("model_revision"),
            stage_runs=tuple(EvaluationStageRun.from_dict(item) for item in value.get("stage_runs", ())),
            result_state=EvaluationRunState(value["result_state"]),
            findings=tuple(ShadowFinding.from_dict(item) for item in value.get("findings", ())),
            coverage_status=MeasurementStatus(value["coverage_status"]),
            latency_seconds=NumericMeasurement.from_dict(value["latency_seconds"]),
            tokens=NumericMeasurement.from_dict(value["tokens"]),
            cost_usd=NumericMeasurement.from_dict(value["cost_usd"]),
            retry_count=value.get("retry_count", 0),
            cached=value.get("cached", False),
            schema_version=value["schema_version"],
        )
        if value.get("entry_id") != entry.entry_id:
            raise EvaluationValidationError("shadow journal entry identity does not match its content")
        return entry

    @classmethod
    def from_run_record(
        cls,
        record: EvaluationRunRecord,
        *,
        arm: EvaluationArm,
        event: ReviewEvent,
        policy_hash: str,
        configuration_hash: str,
        parent_snapshot_id: Optional[str] = None,
        selected_depth: Optional[str] = None,
        reason_codes: tuple[str, ...] = (),
    ) -> "ShadowJournalEntry":
        if not isinstance(arm, EvaluationArm) or arm.arm_id != record.arm_id:
            raise EvaluationValidationError("shadow run record requires its exact frozen arm")
        validate_run_model_telemetry(arm, record, context="shadow run")
        coverage_status = (
            MeasurementStatus.COMPLETE
            if record.state is EvaluationRunState.COMPLETED
            and len(record.stage_runs) == len(arm.stage_plan)
            and all(stage.coverage_status is MeasurementStatus.COMPLETE for stage in record.stage_runs)
            else MeasurementStatus.UNAVAILABLE
        )
        return cls(
            snapshot_id=record.snapshot_id,
            parent_snapshot_id=parent_snapshot_id,
            event=event,
            policy_hash=policy_hash,
            configuration_hash=configuration_hash,
            arm_id=record.arm_id,
            selected_depth=selected_depth,
            reason_codes=reason_codes,
            model_id=record.model_id,
            provider_id=record.provider_id,
            model_revision=record.model_revision,
            stage_runs=record.stage_runs,
            result_state=record.state,
            findings=tuple(
                ShadowFinding.from_fingerprint(finding.fingerprint, finding.lifecycle_state)
                for finding in record.findings
            ),
            coverage_status=coverage_status,
            latency_seconds=record.latency_seconds,
            tokens=record.tokens,
            cost_usd=record.cost_usd,
            retry_count=record.retry_count,
            cached=record.cached,
        )


class DeveloperTimeBasis(str, Enum):
    WRITER_START = "writer_start"
    WRITER_MONOTONIC = "writer_monotonic"


@dataclass(frozen=True)
class ShadowJournalSessionSummary:
    """Writer-owned completeness evidence attached to the last retained event in a session."""

    submitted_entry_count: int
    queued_entry_count: int
    dropped_entry_count: int
    writer_failed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("submitted_entry_count", self.submitted_entry_count),
            ("queued_entry_count", self.queued_entry_count),
            ("dropped_entry_count", self.dropped_entry_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EvaluationValidationError(f"shadow journal {name} must be a non-negative integer")
        if self.queued_entry_count < 1:
            raise EvaluationValidationError("shadow journal session summary requires a retained entry")
        if self.submitted_entry_count != self.queued_entry_count + self.dropped_entry_count:
            raise EvaluationValidationError("shadow journal session counts do not reconcile")
        if not isinstance(self.writer_failed, bool):
            raise EvaluationValidationError("shadow journal writer_failed must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted_entry_count": self.submitted_entry_count,
            "queued_entry_count": self.queued_entry_count,
            "dropped_entry_count": self.dropped_entry_count,
            "writer_failed": self.writer_failed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowJournalSessionSummary":
        _reject_unknown_fields("shadow journal session summary", value, {
            "submitted_entry_count", "queued_entry_count", "dropped_entry_count", "writer_failed",
        })
        return cls(
            submitted_entry_count=value["submitted_entry_count"],
            queued_entry_count=value["queued_entry_count"],
            dropped_entry_count=value["dropped_entry_count"],
            writer_failed=value["writer_failed"],
        )


@dataclass(frozen=True)
class ShadowJournalRecord:
    """Writer-stamped immutable envelope around one source-free entry."""

    sequence: int
    ingested_at_utc: datetime
    developer_time_basis: DeveloperTimeBasis
    developer_elapsed_seconds: Optional[float]
    entry: ShadowJournalEntry
    session_summary: Optional[ShadowJournalSessionSummary] = None
    schema_version: str = SHADOW_JOURNAL_RECORD_SCHEMA_VERSION
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SHADOW_JOURNAL_RECORD_SCHEMA_VERSION:
            raise EvaluationValidationError("unsupported shadow journal record schema_version")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise EvaluationValidationError("shadow journal sequence must be a positive integer")
        if (
            not isinstance(self.ingested_at_utc, datetime)
            or self.ingested_at_utc.tzinfo is None
            or self.ingested_at_utc.utcoffset() != timezone.utc.utcoffset(self.ingested_at_utc)
        ):
            raise EvaluationValidationError("shadow journal ingestion time must be normalized to UTC")
        if not isinstance(self.developer_time_basis, DeveloperTimeBasis):
            raise EvaluationValidationError("shadow journal developer time basis is invalid")
        if self.developer_elapsed_seconds is not None and (
            not isinstance(self.developer_elapsed_seconds, (int, float))
            or isinstance(self.developer_elapsed_seconds, bool)
            or not math.isfinite(self.developer_elapsed_seconds)
            or self.developer_elapsed_seconds < 0
        ):
            raise EvaluationValidationError("shadow journal developer elapsed time must be finite and non-negative")
        if self.developer_time_basis is DeveloperTimeBasis.WRITER_START and self.developer_elapsed_seconds is not None:
            raise EvaluationValidationError("writer-start records cannot claim developer elapsed time")
        if self.developer_time_basis is DeveloperTimeBasis.WRITER_MONOTONIC and self.developer_elapsed_seconds is None:
            raise EvaluationValidationError("writer-monotonic records require developer elapsed time")
        if not isinstance(self.entry, ShadowJournalEntry):
            raise EvaluationValidationError("shadow journal record requires a validated entry")
        if self.session_summary is not None and not isinstance(
            self.session_summary,
            ShadowJournalSessionSummary,
        ):
            raise EvaluationValidationError("shadow journal session_summary is invalid")
        object.__setattr__(self, "record_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "ingested_at_utc": self.ingested_at_utc.isoformat(),
            "developer_time_basis": self.developer_time_basis.value,
            "developer_elapsed_seconds": self.developer_elapsed_seconds,
            "entry": self.entry.to_dict(),
            "session_summary": self.session_summary.to_dict() if self.session_summary is not None else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "record_id": self.record_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowJournalRecord":
        _reject_unknown_fields("shadow journal record", value, {
            "schema_version", "sequence", "ingested_at_utc", "developer_time_basis",
            "developer_elapsed_seconds", "entry", "session_summary", "record_id",
        })
        record = cls(
            sequence=value["sequence"],
            ingested_at_utc=datetime.fromisoformat(value["ingested_at_utc"]),
            developer_time_basis=DeveloperTimeBasis(value["developer_time_basis"]),
            developer_elapsed_seconds=value.get("developer_elapsed_seconds"),
            entry=ShadowJournalEntry.from_dict(value["entry"]),
            session_summary=(
                ShadowJournalSessionSummary.from_dict(value["session_summary"])
                if value.get("session_summary") is not None
                else None
            ),
            schema_version=value["schema_version"],
        )
        if value.get("record_id") != record.record_id:
            raise EvaluationValidationError("shadow journal record identity does not match its content")
        return record


def shadow_journal_inventory_complete(records: tuple[ShadowJournalRecord, ...]) -> bool:
    """Validate writer-session boundaries and report whether every submission was retained."""
    session_entry_count = 0
    inventory_complete = bool(records)
    for record in records:
        if record.developer_time_basis is DeveloperTimeBasis.WRITER_START:
            if session_entry_count:
                inventory_complete = False
            session_entry_count = 1
        else:
            if session_entry_count == 0:
                raise EvaluationValidationError("shadow writer-monotonic record has no writer-start record")
            session_entry_count += 1
        if record.session_summary is None:
            continue
        if record.session_summary.queued_entry_count != session_entry_count:
            raise EvaluationValidationError("shadow journal session summary does not match retained entries")
        if record.session_summary.dropped_entry_count or record.session_summary.writer_failed:
            inventory_complete = False
        session_entry_count = 0
    return inventory_complete and session_entry_count == 0


def _append_private_line(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = path.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise EvaluationValidationError("shadow journal parent must be a real directory")
    if parent_metadata.st_uid != os.geteuid() or parent_metadata.st_mode & 0o077:
        raise EvaluationValidationError("shadow journal parent must be private to the current user")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise EvaluationValidationError("shadow journal must be a private regular file")
        written = 0
        while written < len(payload):
            chunk_size = os.write(descriptor, payload[written:])
            if chunk_size == 0:
                raise EvaluationValidationError("could not complete shadow journal append")
            written += chunk_size
    finally:
        os.close(descriptor)


def load_shadow_journal(path: str | Path) -> tuple[ShadowJournalRecord, ...]:
    """Parse a private NDJSON journal and validate its immutable sequence."""
    path = Path(path)
    if not path.exists():
        return ()
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise EvaluationValidationError("shadow journal must be a private regular file")
    records: list[ShadowJournalRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                raise EvaluationValidationError("shadow journal cannot contain blank records")
            records.append(ShadowJournalRecord.from_dict(json.loads(line)))
    except EvaluationValidationError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EvaluationValidationError("shadow journal contains an invalid record") from exc
    if [record.sequence for record in records] != list(range(1, len(records) + 1)):
        raise EvaluationValidationError("shadow journal sequence must be contiguous")
    if any(
        later.ingested_at_utc < earlier.ingested_at_utc
        for earlier, later in zip(records, records[1:], strict=False)
    ):
        raise EvaluationValidationError("shadow journal ingestion time cannot move backwards")
    if len({record.record_id for record in records}) != len(records):
        raise EvaluationValidationError("shadow journal records must be unique")
    shadow_journal_inventory_complete(tuple(records))
    return tuple(records)


class ShadowJournalWriter:
    """A bounded background sink; checkpoint code performs only ``put_nowait``."""

    _STOP = object()

    def __init__(self, path: str | Path, *, enabled: bool, max_queue_entries: int = 256):
        if not isinstance(enabled, bool):
            raise EvaluationValidationError("shadow journal enabled must be a boolean")
        if not isinstance(max_queue_entries, int) or max_queue_entries < 1:
            raise EvaluationValidationError("shadow journal max_queue_entries must be a positive integer")
        self.path = Path(path)
        self.enabled = enabled
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_entries)
        self._closed = threading.Event()
        self._failed = threading.Event()
        self._stop_enqueued = threading.Event()
        self._close_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        existing = load_shadow_journal(self.path) if self.enabled else ()
        self._next_sequence = len(existing) + 1
        self._last_submitted_monotonic: Optional[float] = None
        self._submitted_entry_count = 0
        self._queued_entry_count = 0
        self._dropped_entry_count = 0
        if self.enabled:
            self._thread = threading.Thread(target=self._drain, name="pr-agent-shadow-journal", daemon=True)
            self._thread.start()

    def submit(self, entry: ShadowJournalEntry) -> ShadowSubmitStatus:
        if not isinstance(entry, ShadowJournalEntry):
            raise EvaluationValidationError("shadow journal accepts only validated ShadowJournalEntry values")
        if not self.enabled:
            return ShadowSubmitStatus.DISABLED
        with self._submit_lock:
            if self._closed.is_set():
                return ShadowSubmitStatus.CLOSED
            self._submitted_entry_count += 1
            if self._failed.is_set():
                self._dropped_entry_count += 1
                return ShadowSubmitStatus.DROPPED
            submitted_monotonic = time.monotonic()
            if self._last_submitted_monotonic is None:
                basis = DeveloperTimeBasis.WRITER_START
                developer_elapsed_seconds = None
            else:
                basis = DeveloperTimeBasis.WRITER_MONOTONIC
                developer_elapsed_seconds = submitted_monotonic - self._last_submitted_monotonic
            record = ShadowJournalRecord(
                sequence=self._next_sequence,
                ingested_at_utc=datetime.now(timezone.utc),
                developer_time_basis=basis,
                developer_elapsed_seconds=developer_elapsed_seconds,
                entry=entry,
            )
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped_entry_count += 1
                return ShadowSubmitStatus.DROPPED
            self._next_sequence += 1
            self._queued_entry_count += 1
            self._last_submitted_monotonic = submitted_monotonic
        return ShadowSubmitStatus.QUEUED

    def _drain(self) -> None:
        pending: Optional[ShadowJournalRecord] = None
        while True:
            item = self._queue.get()
            if item is self._STOP:
                try:
                    if pending is not None:
                        with self._submit_lock:
                            summary = ShadowJournalSessionSummary(
                                submitted_entry_count=self._submitted_entry_count,
                                queued_entry_count=self._queued_entry_count,
                                dropped_entry_count=self._dropped_entry_count,
                                writer_failed=self._failed.is_set(),
                            )
                        self._append_record(replace(pending, session_summary=summary))
                except (OSError, EvaluationValidationError):
                    self._failed.set()
                finally:
                    self._queue.task_done()
                return
            try:
                if not isinstance(item, ShadowJournalRecord):
                    raise EvaluationValidationError("shadow queue contained an invalid record")
                previous = pending
                pending = item
                if previous is not None:
                    self._append_record(previous)
            except (OSError, EvaluationValidationError):
                self._failed.set()
            finally:
                self._queue.task_done()

    def _append_record(self, record: ShadowJournalRecord) -> None:
        payload = (
            json.dumps(
                record.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _append_private_line(self.path, payload)

    def close(self, timeout_seconds: float = 5.0) -> bool:
        """Flush only during explicit shutdown; ``submit`` remains non-blocking."""
        if not self.enabled:
            self._closed.set()
            return not self._failed.is_set()
        assert self._thread is not None
        with self._close_lock:
            with self._submit_lock:
                self._closed.set()
                if self._thread.is_alive() and not self._stop_enqueued.is_set():
                    try:
                        self._queue.put(self._STOP, timeout=timeout_seconds)
                    except queue.Full:
                        return False
                    self._stop_enqueued.set()
            if not self._thread.is_alive():
                return not self._failed.is_set() and self._dropped_entry_count == 0
            self._thread.join(timeout_seconds)
            return (
                not self._thread.is_alive()
                and not self._failed.is_set()
                and self._dropped_entry_count == 0
            )
