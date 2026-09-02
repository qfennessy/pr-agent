"""Fail-closed authorization and immutable local artifacts for checkpoint replay.

This module does not call a provider or model.  It is the boundary a future
production-backed runner must cross before it can spend money, and the durable
store it must use so failed attempts are never replaced by a later success.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pr_agent.algo.checkpoint_evaluation import (EVALUATION_SCHEMA_VERSION,
                                                 EvaluationArmKind,
                                                 EvaluationManifest,
                                                 EvaluationPlanItem,
                                                 EvaluationRunRecord,
                                                 EvaluationRunState,
                                                 EvaluationValidationError,
                                                 GateStatus, MeasurementStatus,
                                                 NumericMeasurement,
                                                 build_evaluation_plan,
                                                 content_hash,
                                                 validate_run_model_telemetry)
from pr_agent.algo.checkpoint_evaluation_scoring import RolloutGateDecision


class PaidExecutionStatus(str, Enum):
    AUTHORIZED = "authorized"
    DENIED = "denied"


class OutputCapability(str, Enum):
    OPT_IN_PAIR_REVIEW = "opt_in_pair_review"
    DEFAULT_PAIR_REVIEW = "default_pair_review"
    PR_PUBLICATION = "pr_publication"


_OUTPUT_GATE_BY_CAPABILITY = {
    OutputCapability.OPT_IN_PAIR_REVIEW: "opt-in-pair-review",
    OutputCapability.DEFAULT_PAIR_REVIEW: "default-pair-review",
    OutputCapability.PR_PUBLICATION: "pr-publication",
}
_MUTABLE_MODEL_REVISIONS = {"default", "latest", "main", "stable"}


@dataclass(frozen=True)
class OutputPermissionDecision:
    capability: OutputCapability
    arm_id: str
    scorecard_id: str
    status: GateStatus
    reason: str

    def require_permitted(self) -> None:
        if self.status is not GateStatus.PASSED:
            raise EvaluationValidationError(
                f"{self.capability.value} is not permitted: {self.reason}"
            )


def evaluate_output_permission(
    capability: OutputCapability,
    decisions: Sequence[RolloutGateDecision],
    *,
    arm_id: str,
    scorecard_id: str,
    required_gate_spec_hash: str,
) -> OutputPermissionDecision:
    """Require one exact passed gate; missing or stale evidence never enables output."""
    if not isinstance(required_gate_spec_hash, str) or not required_gate_spec_hash.startswith("sha256:"):
        raise EvaluationValidationError("required_gate_spec_hash must be an approved sha256 identity")
    required_gate = _OUTPUT_GATE_BY_CAPABILITY[capability]
    matching = [
        decision for decision in decisions
        if decision.gate_name == required_gate
        and decision.arm_id == arm_id
        and decision.scorecard_id == scorecard_id
        and decision.gate_spec_hash == required_gate_spec_hash
    ]
    if not matching:
        status = GateStatus.NOT_EVALUABLE
        reason = "the exact current scorecard has no matching rollout decision"
    elif len(matching) > 1:
        status = GateStatus.NOT_EVALUABLE
        reason = "multiple conflicting rollout decisions match the current scorecard"
    else:
        status = matching[0].status
        reason = (
            "the exact rollout gate passed"
            if status is GateStatus.PASSED
            else f"the exact rollout gate is {status.value}"
        )
    return OutputPermissionDecision(
        capability=capability,
        arm_id=arm_id,
        scorecard_id=scorecard_id,
        status=status,
        reason=reason,
    )


@dataclass(frozen=True)
class PaidExecutionRequest:
    """Secret-free proof that one manifest has an explicit spending boundary."""

    manifest_id: str
    cost_cap_usd: float
    projected_cost_usd: NumericMeasurement
    credential_present_by_provider: Mapping[str, bool]
    schema_version: str = EVALUATION_SCHEMA_VERSION
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported paid execution schema_version: {self.schema_version}")
        if not isinstance(self.manifest_id, str) or not self.manifest_id.startswith("sha256:"):
            raise EvaluationValidationError("paid execution manifest_id must be a sha256 identity")
        if not isinstance(self.cost_cap_usd, (int, float)) or isinstance(self.cost_cap_usd, bool):
            raise EvaluationValidationError("paid execution cost_cap_usd must be numeric")
        if not math.isfinite(self.cost_cap_usd) or self.cost_cap_usd <= 0:
            raise EvaluationValidationError("paid execution cost_cap_usd must be finite and greater than zero")
        if not isinstance(self.projected_cost_usd, NumericMeasurement):
            raise EvaluationValidationError("projected_cost_usd must use NumericMeasurement")
        if not isinstance(self.credential_present_by_provider, Mapping) or any(
            not isinstance(provider, str)
            or not provider.strip()
            or not isinstance(present, bool)
            for provider, present in self.credential_present_by_provider.items()
        ):
            raise EvaluationValidationError("credential presence must map provider ids to booleans")
        credentials = MappingProxyType(dict(self.credential_present_by_provider))
        object.__setattr__(self, "credential_present_by_provider", credentials)
        object.__setattr__(self, "request_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "cost_cap_usd": self.cost_cap_usd,
            "projected_cost_usd": self.projected_cost_usd.to_dict(),
            "credential_present_by_provider": dict(sorted(self.credential_present_by_provider.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "request_id": self.request_id}


@dataclass(frozen=True)
class PaidExecutionDecision:
    request_id: str
    manifest_id: str
    status: PaidExecutionStatus
    reasons: tuple[str, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PaidExecutionStatus):
            raise EvaluationValidationError("paid execution status must be a PaidExecutionStatus")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not self.reasons or any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons):
            raise EvaluationValidationError("paid execution decision requires non-empty reasons")
        object.__setattr__(self, "decision_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "manifest_id": self.manifest_id,
            "status": self.status.value,
            "reasons": list(self.reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "decision_id": self.decision_id}

    def require_authorized(self) -> None:
        if self.status is not PaidExecutionStatus.AUTHORIZED:
            raise EvaluationValidationError("paid execution denied: " + "; ".join(self.reasons))


def evaluate_paid_execution(
    manifest: EvaluationManifest,
    request: PaidExecutionRequest,
    *,
    evaluation_enabled: bool,
    allow_paid_execution: bool,
    publish_output: bool,
) -> PaidExecutionDecision:
    """Authorize spending only when every required proof is explicit and complete."""
    reasons: list[str] = []
    if request.manifest_id != manifest.manifest_id:
        reasons.append("request belongs to a different immutable manifest")
    if not evaluation_enabled:
        reasons.append("checkpoint evaluation is disabled")
    if not allow_paid_execution:
        reasons.append("paid execution is disabled")
    if publish_output:
        reasons.append("evaluation replay cannot publish developer-visible output")
    if request.projected_cost_usd.status is not MeasurementStatus.COMPLETE:
        reasons.append("projected cost is partial or unavailable")
    elif (
        request.projected_cost_usd.value is None
        or request.projected_cost_usd.value <= 0
        or request.projected_cost_usd.value > request.cost_cap_usd
    ):
        reasons.append("projected cost must be positive and within the explicit cap")

    for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id):
        if arm.kind is EvaluationArmKind.DETERMINISTIC:
            continue
        for model_id, provider_id, model_revision in arm.model_identities():
            identity_name = f"arm {arm.arm_id} model {model_id}"
            if not model_revision:
                reasons.append(f"{identity_name} does not pin an immutable model revision")
            elif (
                model_revision.strip().lower() in _MUTABLE_MODEL_REVISIONS
                or model_revision.strip().lower().endswith("/latest")
            ):
                reasons.append(f"{identity_name} uses a mutable model revision alias")
            if not request.credential_present_by_provider.get(provider_id or "", False):
                reasons.append(f"{identity_name} has no credential for provider {provider_id}")

    if reasons:
        status = PaidExecutionStatus.DENIED
    else:
        status = PaidExecutionStatus.AUTHORIZED
        reasons.append("all paid execution gates are satisfied")
    return PaidExecutionDecision(
        request_id=request.request_id,
        manifest_id=manifest.manifest_id,
        status=status,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class ResumePlanItem:
    plan_item: EvaluationPlanItem
    next_attempt: int
    retained_attempt_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_item": self.plan_item.to_dict(),
            "next_attempt": self.next_attempt,
            "retained_attempt_ids": list(self.retained_attempt_ids),
        }


@dataclass(frozen=True)
class EvaluationArtifactInventory:
    manifest_id: str
    manifest_artifact_hash: str
    record_artifact_hashes: tuple[str, ...]
    terminal_pair_count: int
    incomplete_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_artifact_hash": self.manifest_artifact_hash,
            "record_artifact_hashes": list(self.record_artifact_hashes),
            "terminal_pair_count": self.terminal_pair_count,
            "incomplete_pair_count": self.incomplete_pair_count,
        }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _ensure_private_directory(path: Path) -> None:
    if path.exists():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EvaluationValidationError(f"artifact path is not a real directory: {path}")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise EvaluationValidationError(f"artifact directory is not private to the current user: {path}")
        return
    path.mkdir(mode=0o700, parents=True)


def _write_exclusive(path: Path, payload: bytes) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise EvaluationValidationError(f"immutable artifact is not a regular file: {path.name}")
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise EvaluationValidationError(f"immutable artifact is not private: {path.name}")
            existing = path.read_bytes()
        except OSError as exc:
            raise EvaluationValidationError(f"cannot read existing immutable artifact: {path.name}") from exc
        if existing != payload:
            raise EvaluationValidationError(f"immutable artifact content changed: {path.name}") from None
        return False
    try:
        written = 0
        while written < len(payload):
            chunk_size = os.write(descriptor, payload[written:])
            if chunk_size == 0:
                raise EvaluationValidationError(f"could not complete immutable artifact write: {path.name}")
            written += chunk_size
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


class EvaluationArtifactStore:
    """Append-only, content-addressed run records with fail-closed resume."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        _ensure_private_directory(self.root)
        self.records_path = self.root / "records"
        _ensure_private_directory(self.records_path)

    def bind_manifest(self, manifest: EvaluationManifest) -> bool:
        return _write_exclusive(self.root / "manifest.json", _canonical_bytes(manifest.to_dict()))

    def append_record(self, manifest: EvaluationManifest, record: EvaluationRunRecord) -> bool:
        if record.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("run record belongs to a different manifest")
        self.bind_manifest(manifest)
        existing = self.load_records(manifest)
        if any(item.record_id == record.record_id for item in existing):
            return False
        self._validate_attempts(manifest, (*existing, record))
        return _write_exclusive(
            self.records_path / f"{record.record_id.removeprefix('sha256:')}.json",
            _canonical_bytes(record.to_dict()),
        )

    def load_records(self, manifest: EvaluationManifest) -> tuple[EvaluationRunRecord, ...]:
        self.bind_manifest(manifest)
        records: list[EvaluationRunRecord] = []
        for path in sorted(self.records_path.iterdir()):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or path.suffix != ".json":
                raise EvaluationValidationError(f"unexpected artifact store entry: {path.name}")
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise EvaluationValidationError(f"run artifact is not private: {path.name}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                record = EvaluationRunRecord.from_dict(value)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EvaluationValidationError(f"invalid immutable run artifact: {path.name}") from exc
            expected_name = f"{record.record_id.removeprefix('sha256:')}.json"
            if path.name != expected_name:
                raise EvaluationValidationError(f"run artifact filename does not match content: {path.name}")
            if record.manifest_id != manifest.manifest_id:
                raise EvaluationValidationError("artifact store mixes run records from different manifests")
            records.append(record)
        self._validate_attempts(manifest, records)
        return tuple(sorted(records, key=lambda item: (item.case_id, item.arm_id, item.attempt, item.record_id)))

    @staticmethod
    def _validate_attempts(manifest: EvaluationManifest, records: Sequence[EvaluationRunRecord]) -> None:
        cases = {case.case_id: case for case in manifest.cases}
        arms = {arm.arm_id: arm for arm in manifest.arms if arm.enabled}
        pairs = {(item.case_id, item.arm_id) for item in build_evaluation_plan(manifest).items}
        attempts: set[tuple[str, str, int]] = set()
        terminal_pairs: set[tuple[str, str]] = set()
        by_pair: dict[tuple[str, str], list[EvaluationRunRecord]] = {}
        for record in records:
            pair = (record.case_id, record.arm_id)
            if pair not in pairs:
                raise EvaluationValidationError("artifact store contains a disabled or unknown case/arm pair")
            if record.snapshot_id != cases[record.case_id].snapshot_id:
                raise EvaluationValidationError("artifact store run snapshot does not match its checkpoint case")
            arm = arms[record.arm_id]
            validate_run_model_telemetry(arm, record, context="artifact store run")
            if record.state is EvaluationRunState.COMPLETED and not record.terminal:
                raise EvaluationValidationError("artifact store cannot retain a non-terminal completed result")
            attempt_key = (*pair, record.attempt)
            if attempt_key in attempts:
                raise EvaluationValidationError("artifact store contains a duplicate attempt number")
            attempts.add(attempt_key)
            by_pair.setdefault(pair, []).append(record)
            if record.terminal:
                if pair in terminal_pairs:
                    raise EvaluationValidationError("artifact store contains multiple terminal records for one pair")
                terminal_pairs.add(pair)
        for pair, pair_records in by_pair.items():
            ordered = sorted(pair_records, key=lambda item: item.attempt)
            if [record.attempt for record in ordered] != list(range(1, len(ordered) + 1)):
                raise EvaluationValidationError(f"artifact store omits a retained attempt for pair {pair}")
            terminal_positions = [index for index, record in enumerate(ordered) if record.terminal]
            if terminal_positions and terminal_positions[0] != len(ordered) - 1:
                raise EvaluationValidationError(f"artifact store has attempts after a terminal record for pair {pair}")

    def resume_plan(self, manifest: EvaluationManifest) -> tuple[ResumePlanItem, ...]:
        records = self.load_records(manifest)
        by_pair: dict[tuple[str, str], list[EvaluationRunRecord]] = {}
        for record in records:
            by_pair.setdefault((record.case_id, record.arm_id), []).append(record)
        items: list[ResumePlanItem] = []
        for plan_item in build_evaluation_plan(manifest).items:
            retained = sorted(by_pair.get((plan_item.case_id, plan_item.arm_id), ()), key=lambda item: item.attempt)
            if any(record.terminal for record in retained):
                continue
            items.append(ResumePlanItem(
                plan_item=plan_item,
                next_attempt=max((record.attempt for record in retained), default=0) + 1,
                retained_attempt_ids=tuple(record.record_id for record in retained),
            ))
        return tuple(items)

    def inventory(self, manifest: EvaluationManifest) -> EvaluationArtifactInventory:
        records = self.load_records(manifest)
        terminal_pairs = {(record.case_id, record.arm_id) for record in records if record.terminal}
        total_pairs = len(build_evaluation_plan(manifest).items)
        manifest_hash = content_hash(manifest.to_dict())
        record_hashes = tuple(sorted(content_hash(record.to_dict()) for record in records))
        return EvaluationArtifactInventory(
            manifest_id=manifest.manifest_id,
            manifest_artifact_hash=manifest_hash,
            record_artifact_hashes=record_hashes,
            terminal_pair_count=len(terminal_pairs),
            incomplete_pair_count=total_pairs - len(terminal_pairs),
        )
