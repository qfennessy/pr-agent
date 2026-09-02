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
from typing import Any, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (EVALUATION_SCHEMA_VERSION,
                                                 EvaluationArmKind,
                                                 EvaluationManifest,
                                                 EvaluationPlanItem,
                                                 EvaluationRunRecord,
                                                 EvaluationRunState,
                                                 EvaluationValidationError,
                                                 GateStatus,
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


_OUTPUT_GATES_BY_CAPABILITY = {
    OutputCapability.OPT_IN_PAIR_REVIEW: (
        "offline-replay", "live-shadow", "opt-in-pair-review",
    ),
    OutputCapability.DEFAULT_PAIR_REVIEW: (
        "offline-replay", "live-shadow", "opt-in-pair-review", "default-pair-review",
    ),
    OutputCapability.PR_PUBLICATION: (
        "offline-replay", "live-shadow", "opt-in-pair-review", "default-pair-review", "pr-publication",
    ),
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
    required_gate_spec_hashes: Mapping[str, str],
) -> OutputPermissionDecision:
    """Require the exact passed prerequisite chain; stale evidence never enables output."""
    required_gates = _OUTPUT_GATES_BY_CAPABILITY[capability]
    if (
        not isinstance(required_gate_spec_hashes, Mapping)
        or set(required_gate_spec_hashes) != set(required_gates)
        or any(
            not isinstance(value, str) or not value.startswith("sha256:")
            for value in required_gate_spec_hashes.values()
        )
    ):
        raise EvaluationValidationError(
            "required_gate_spec_hashes must pin every approved prerequisite gate"
        )
    status = GateStatus.PASSED
    reason = "the exact rollout gate and every prerequisite passed"
    for required_gate in required_gates:
        matching = [
            decision for decision in decisions
            if decision.gate_name == required_gate
            and decision.arm_id == arm_id
            and decision.scorecard_id == scorecard_id
            and decision.gate_spec_hash == required_gate_spec_hashes[required_gate]
        ]
        if not matching:
            status = GateStatus.NOT_EVALUABLE
            reason = f"the exact current scorecard has no matching {required_gate} decision"
            break
        if len(matching) > 1:
            status = GateStatus.NOT_EVALUABLE
            reason = f"multiple conflicting {required_gate} decisions match the current scorecard"
            break
        if matching[0].status is not GateStatus.PASSED:
            status = matching[0].status
            reason = f"the exact {required_gate} gate is {status.value}"
            break
    return OutputPermissionDecision(
        capability=capability,
        arm_id=arm_id,
        scorecard_id=scorecard_id,
        status=status,
        reason=reason,
    )


@dataclass(frozen=True)
class PaidPlanItemBudget:
    """Maximum paid attempts and hard per-call cost for one immutable case/arm pair."""

    case_id: str
    arm_id: str
    hard_cost_cap_per_attempt_usd: float
    max_attempts: int

    def __post_init__(self) -> None:
        for name, value in (("case_id", self.case_id), ("arm_id", self.arm_id)):
            if not isinstance(value, str) or not value.strip():
                raise EvaluationValidationError(f"paid budget {name} must be non-empty")
        if (
            not isinstance(self.hard_cost_cap_per_attempt_usd, (int, float))
            or isinstance(self.hard_cost_cap_per_attempt_usd, bool)
            or not math.isfinite(self.hard_cost_cap_per_attempt_usd)
            or self.hard_cost_cap_per_attempt_usd <= 0
        ):
            raise EvaluationValidationError("paid budget hard cost cap must be finite and positive")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= 100
        ):
            raise EvaluationValidationError("paid budget max_attempts must be an integer from 1 to 100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "hard_cost_cap_per_attempt_usd": self.hard_cost_cap_per_attempt_usd,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PaidPlanItemBudget":
        allowed = {"case_id", "arm_id", "hard_cost_cap_per_attempt_usd", "max_attempts"}
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise EvaluationValidationError("paid plan-item budget contains unknown fields")
        return cls(
            case_id=value["case_id"],
            arm_id=value["arm_id"],
            hard_cost_cap_per_attempt_usd=value["hard_cost_cap_per_attempt_usd"],
            max_attempts=value["max_attempts"],
        )


@dataclass(frozen=True)
class PaidExecutionRequest:
    """Secret-free proof that one manifest has an explicit spending boundary."""

    manifest_id: str
    cost_cap_usd: float
    plan_item_budgets: tuple[PaidPlanItemBudget, ...]
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
        object.__setattr__(self, "plan_item_budgets", tuple(self.plan_item_budgets))
        if not self.plan_item_budgets or any(
            not isinstance(budget, PaidPlanItemBudget) for budget in self.plan_item_budgets
        ):
            raise EvaluationValidationError("paid execution requires per-plan-item budgets")
        pairs = tuple((budget.case_id, budget.arm_id) for budget in self.plan_item_budgets)
        if len(pairs) != len(set(pairs)):
            raise EvaluationValidationError("paid execution plan-item budgets must be unique")
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
            "plan_item_budgets": [budget.to_dict() for budget in self.plan_item_budgets],
            "credential_present_by_provider": dict(sorted(self.credential_present_by_provider.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PaidExecutionRequest":
        allowed = {
            "schema_version", "manifest_id", "cost_cap_usd", "plan_item_budgets",
            "credential_present_by_provider", "request_id",
        }
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise EvaluationValidationError("paid execution request contains unknown fields")
        request = cls(
            manifest_id=value["manifest_id"],
            cost_cap_usd=value["cost_cap_usd"],
            plan_item_budgets=tuple(PaidPlanItemBudget.from_dict(item) for item in value["plan_item_budgets"]),
            credential_present_by_provider=value["credential_present_by_provider"],
            schema_version=value["schema_version"],
        )
        if value.get("request_id") != request.request_id:
            raise EvaluationValidationError("paid execution request identity does not match its content")
        return request


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
    paid_pairs = {
        (case.case_id, arm.arm_id)
        for case in manifest.cases
        for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    }
    budget_pairs = {(budget.case_id, budget.arm_id) for budget in request.plan_item_budgets}
    if budget_pairs != paid_pairs:
        reasons.append("paid budgets must match every model-backed plan item exactly")
    reserved_cost = sum(
        budget.hard_cost_cap_per_attempt_usd * budget.max_attempts
        for budget in request.plan_item_budgets
    )
    if reserved_cost > request.cost_cap_usd:
        reasons.append("reserved paid plan-item cost exceeds the explicit cap")

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
class PaidAttemptReservation:
    """Durable proof that one paid attempt number was consumed before its adapter ran."""

    manifest_id: str
    request_id: str
    case_id: str
    arm_id: str
    snapshot_id: str
    attempt: int
    hard_cost_cap_usd: float
    schema_version: str = EVALUATION_SCHEMA_VERSION
    reservation_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"unsupported paid reservation schema_version: {self.schema_version}"
            )
        for name, value in (
            ("manifest_id", self.manifest_id),
            ("request_id", self.request_id),
            ("snapshot_id", self.snapshot_id),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise EvaluationValidationError(f"paid reservation {name} must be a sha256 identity")
        for name, value in (("case_id", self.case_id), ("arm_id", self.arm_id)):
            if not isinstance(value, str) or not value.strip():
                raise EvaluationValidationError(f"paid reservation {name} must be non-empty")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise EvaluationValidationError("paid reservation attempt must be a positive integer")
        if (
            not isinstance(self.hard_cost_cap_usd, (int, float))
            or isinstance(self.hard_cost_cap_usd, bool)
            or not math.isfinite(self.hard_cost_cap_usd)
            or self.hard_cost_cap_usd <= 0
        ):
            raise EvaluationValidationError("paid reservation hard cost cap must be finite and positive")
        object.__setattr__(self, "reservation_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "request_id": self.request_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "snapshot_id": self.snapshot_id,
            "attempt": self.attempt,
            "hard_cost_cap_usd": self.hard_cost_cap_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "reservation_id": self.reservation_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PaidAttemptReservation":
        allowed = {
            "schema_version", "manifest_id", "request_id", "case_id", "arm_id", "snapshot_id",
            "attempt", "hard_cost_cap_usd", "reservation_id",
        }
        if not isinstance(value, Mapping) or set(value) != allowed:
            raise EvaluationValidationError("paid reservation fields are incomplete or unknown")
        reservation = cls(
            manifest_id=value["manifest_id"],
            request_id=value["request_id"],
            case_id=value["case_id"],
            arm_id=value["arm_id"],
            snapshot_id=value["snapshot_id"],
            attempt=value["attempt"],
            hard_cost_cap_usd=value["hard_cost_cap_usd"],
            schema_version=value["schema_version"],
        )
        if value["reservation_id"] != reservation.reservation_id:
            raise EvaluationValidationError("paid reservation identity does not match its content")
        return reservation


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
        self.reservations_path = self.root / "paid-attempt-reservations"
        _ensure_private_directory(self.reservations_path)

    def bind_manifest(self, manifest: EvaluationManifest) -> bool:
        return _write_exclusive(self.root / "manifest.json", _canonical_bytes(manifest.to_dict()))

    def bind_paid_request(self, manifest: EvaluationManifest, request: PaidExecutionRequest) -> bool:
        if request.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("paid execution request belongs to a different manifest")
        self.bind_manifest(manifest)
        return _write_exclusive(
            self.root / "paid-execution-request.json",
            _canonical_bytes(request.to_dict()),
        )

    def load_paid_request(self, manifest: EvaluationManifest) -> Optional[PaidExecutionRequest]:
        self.bind_manifest(manifest)
        path = self.root / "paid-execution-request.json"
        if not path.exists():
            return None
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise EvaluationValidationError("bound paid execution request must be a private regular file")
        try:
            request = PaidExecutionRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EvaluationValidationError("invalid bound paid execution request") from exc
        if request.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("bound paid execution request belongs to a different manifest")
        return request

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

    def reserve_paid_attempt(
        self,
        manifest: EvaluationManifest,
        request: PaidExecutionRequest,
        plan_item: EvaluationPlanItem,
        attempt: int,
    ) -> bool:
        """Consume one paid attempt number durably before entering its adapter."""
        budget = next(
            (
                item for item in request.plan_item_budgets
                if item.case_id == plan_item.case_id and item.arm_id == plan_item.arm_id
            ),
            None,
        )
        if budget is None:
            raise EvaluationValidationError("paid attempt has no immutable hard cost cap")
        if attempt > budget.max_attempts:
            raise EvaluationValidationError("paid attempt exceeds its immutable attempt limit")
        cases = {case.case_id: case for case in manifest.cases}
        if plan_item.case_id not in cases or request.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("paid attempt does not belong to the immutable manifest")
        reservation = PaidAttemptReservation(
            manifest_id=manifest.manifest_id,
            request_id=request.request_id,
            case_id=plan_item.case_id,
            arm_id=plan_item.arm_id,
            snapshot_id=cases[plan_item.case_id].snapshot_id,
            attempt=attempt,
            hard_cost_cap_usd=budget.hard_cost_cap_per_attempt_usd,
        )
        return _write_exclusive(
            self.reservations_path / f"{reservation.reservation_id.removeprefix('sha256:')}.json",
            _canonical_bytes(reservation.to_dict()),
        )

    def load_paid_attempt_reservations(
        self,
        manifest: EvaluationManifest,
        request: PaidExecutionRequest,
    ) -> tuple[PaidAttemptReservation, ...]:
        reservations: list[PaidAttemptReservation] = []
        case_by_id = {case.case_id: case for case in manifest.cases}
        budget_by_pair = {
            (budget.case_id, budget.arm_id): budget for budget in request.plan_item_budgets
        }
        for path in sorted(self.reservations_path.iterdir()):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or path.suffix != ".json":
                raise EvaluationValidationError(f"unexpected paid reservation artifact: {path.name}")
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise EvaluationValidationError(f"paid reservation artifact is not private: {path.name}")
            try:
                reservation = PaidAttemptReservation.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EvaluationValidationError(f"invalid paid reservation artifact: {path.name}") from exc
            expected_name = f"{reservation.reservation_id.removeprefix('sha256:')}.json"
            if path.name != expected_name:
                raise EvaluationValidationError("paid reservation filename does not match content")
            budget = budget_by_pair.get((reservation.case_id, reservation.arm_id))
            if (
                reservation.manifest_id != manifest.manifest_id
                or reservation.request_id != request.request_id
                or budget is None
                or reservation.case_id not in case_by_id
                or reservation.snapshot_id != case_by_id[reservation.case_id].snapshot_id
                or reservation.hard_cost_cap_usd != budget.hard_cost_cap_per_attempt_usd
                or reservation.attempt > budget.max_attempts
            ):
                raise EvaluationValidationError("paid reservation does not match its immutable authorization")
            reservations.append(reservation)
        keys = [(item.case_id, item.arm_id, item.attempt) for item in reservations]
        if len(keys) != len(set(keys)):
            raise EvaluationValidationError("paid reservation store contains duplicate attempt numbers")
        return tuple(sorted(reservations, key=lambda item: (item.case_id, item.arm_id, item.attempt)))

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

    def resume_plan(
        self,
        manifest: EvaluationManifest,
        paid_request: Optional[PaidExecutionRequest] = None,
    ) -> tuple[ResumePlanItem, ...]:
        records = self.load_records(manifest)
        reservations = (
            self.load_paid_attempt_reservations(manifest, paid_request)
            if paid_request is not None
            else ()
        )
        if paid_request is None and any(self.reservations_path.iterdir()):
            raise EvaluationValidationError("paid reservations require their immutable authorization")
        by_pair: dict[tuple[str, str], list[EvaluationRunRecord]] = {}
        for record in records:
            by_pair.setdefault((record.case_id, record.arm_id), []).append(record)
        reservations_by_pair: dict[tuple[str, str], list[PaidAttemptReservation]] = {}
        for reservation in reservations:
            reservations_by_pair.setdefault((reservation.case_id, reservation.arm_id), []).append(reservation)
        items: list[ResumePlanItem] = []
        for plan_item in build_evaluation_plan(manifest).items:
            retained = sorted(by_pair.get((plan_item.case_id, plan_item.arm_id), ()), key=lambda item: item.attempt)
            reserved = reservations_by_pair.get((plan_item.case_id, plan_item.arm_id), ())
            retained_attempts = {record.attempt for record in retained}
            if any(reservation.attempt not in retained_attempts for reservation in reserved):
                raise EvaluationValidationError(
                    f"paid pair {plan_item.case_id}/{plan_item.arm_id} has an unreconciled attempt reservation"
                )
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
