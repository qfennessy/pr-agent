"""Canonical rollout gates and privacy-safe checkpoint pilot reports.

This module intentionally contains no provider, model, or publication calls.  It
turns the immutable artifact store and independently supplied, source-free pilot
measurements into the five rollout decisions required by issue #27.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (EvaluationArmKind,
                                                 EvaluationCohort,
                                                 EvaluationManifest,
                                                 EvaluationRunRecord,
                                                 EvaluationRunState,
                                                 EvaluationValidationError,
                                                 FindingLifecycleState,
                                                 FindingSeverity,
                                                 MeasurementStatus,
                                                 TruthArtifact, content_hash,
                                                 validate_run_model_telemetry)
from pr_agent.algo.checkpoint_evaluation_cocos import (
    COCOS_ADAPTER_SCHEMA_VERSION, CocosCorpusInventory)
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactInventory, EvaluationArtifactStore)
from pr_agent.algo.checkpoint_evaluation_scoring import (GateComparator,
                                                         GateRule,
                                                         MatchedArmScorecard,
                                                         RolloutGateDecision,
                                                         ScoreMetric,
                                                         evaluate_rollout_gate,
                                                         score_matched_arms)
from pr_agent.algo.checkpoint_shadow_journal import (DeveloperTimeBasis,
                                                     ShadowJournalRecord,
                                                     load_shadow_journal)
from pr_agent.algo.review_snapshot import ReviewEvent

PILOT_REPORT_SCHEMA_VERSION = "checkpoint-evaluation-pilot-report-v1"
COCOS_PILOT_ACCEPTANCE_SCHEMA_VERSION = "cocos-story-pilot-acceptance-v1"
SHADOW_PILOT_ACCEPTANCE_SCHEMA_VERSION = "checkpoint-shadow-pilot-acceptance-v1"
SETTLED_PILOT_ACCEPTANCE_SCHEMA_VERSION = "checkpoint-settled-pilot-acceptance-v1"

# A maintainer must pin the id of a separately reviewed, source-free acceptance artifact
# before the Cocos publication gate can pass. Generating an artifact does not approve it.
CANONICAL_COCOS_PILOT_ACCEPTANCE_ID: Optional[str] = None
CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID: Optional[str] = None
CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID: Optional[str] = None
CANONICAL_HOLDOUT_LEAKAGE_CHECK_ID: Optional[str] = None

OFFLINE_REPLAY_GATE = "offline-replay"
LIVE_SHADOW_GATE = "live-shadow"
OPT_IN_PAIR_REVIEW_GATE = "opt-in-pair-review"
DEFAULT_PAIR_REVIEW_GATE = "default-pair-review"
PR_PUBLICATION_GATE = "pr-publication"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MINIMUM_POSITIVE_ADVANTAGE = 1e-12
_FROZEN_HOLDOUT_MINIMUM_SUPPORT = 18
_CANONICAL_COCOS_LOCK_ID = (
    "sha256:4db5b13a4f6240204350274d5147103f591a4216bfadc2560bca4a6d9ce7df13"
)
_CANONICAL_COCOS_COHORT_COUNTS = {
    "calibration": 12,
    "holdout": 18,
    "temporal": 10,
    "control": 16,
    "confirmation": 16,
    "unique_snapshots": 55,
}
_MANIFEST_COHORT_TO_COCOS_COUNT = {
    EvaluationCohort.CALIBRATION: "calibration",
    EvaluationCohort.HOLDOUT: "holdout",
    EvaluationCohort.TEMPORAL: "temporal",
    EvaluationCohort.CLEAN_CONTROL: "control",
}


def _unavailable_metric() -> ScoreMetric:
    return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)


def _validate_optional_positive(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise EvaluationValidationError(f"{name} must be a finite positive number when supplied")


def _validate_optional_non_negative(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise EvaluationValidationError(f"{name} must be a finite non-negative number when supplied")


def _validate_measurement_non_negative(name: str, measurement: Any) -> None:
    value = measurement.value
    if value is not None and value < 0:
        raise EvaluationValidationError(f"{name} cannot be negative")


def _validate_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvaluationValidationError(f"{name} must be a sha256:<64 lowercase hex> identity")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True)
class PilotRolloutBudgets:
    """Maintainer-accepted thresholds whose absence must keep a gate closed."""

    shadow_latency_p95_seconds: Optional[float] = None
    shadow_cost_per_developer_hour_usd: Optional[float] = None
    accepted_false_interruptions_per_clean_checkpoint: Optional[float] = None
    publication_cost_ceiling_usd: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_optional_positive("shadow_latency_p95_seconds", self.shadow_latency_p95_seconds)
        _validate_optional_positive(
            "shadow_cost_per_developer_hour_usd",
            self.shadow_cost_per_developer_hour_usd,
        )
        _validate_optional_non_negative(
            "accepted_false_interruptions_per_clean_checkpoint",
            self.accepted_false_interruptions_per_clean_checkpoint,
        )
        _validate_optional_positive("publication_cost_ceiling_usd", self.publication_cost_ceiling_usd)

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "shadow_latency_p95_seconds": self.shadow_latency_p95_seconds,
            "shadow_cost_per_developer_hour_usd": self.shadow_cost_per_developer_hour_usd,
            "accepted_false_interruptions_per_clean_checkpoint": (
                self.accepted_false_interruptions_per_clean_checkpoint
            ),
            "publication_cost_ceiling_usd": self.publication_cost_ceiling_usd,
        }


@dataclass(frozen=True)
class ReplayEvidenceBinding:
    """Bind separately computed replay measurements to one exact frozen comparison."""

    scorecard_id: str
    artifact_inventory_hash: str
    temporal_cohort_hash: str
    holdout_cohort_hash: str
    target_arm_id: str
    incumbent_arm_id: str
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "scorecard_id",
            "artifact_inventory_hash",
            "temporal_cohort_hash",
            "holdout_cohort_hash",
        ):
            _validate_hash(name, getattr(self, name))
        if not self.target_arm_id.strip() or not self.incumbent_arm_id.strip():
            raise EvaluationValidationError("replay evidence arm identifiers must be non-empty")
        object.__setattr__(self, "binding_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, str]:
        return {
            "scorecard_id": self.scorecard_id,
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "temporal_cohort_hash": self.temporal_cohort_hash,
            "holdout_cohort_hash": self.holdout_cohort_hash,
            "target_arm_id": self.target_arm_id,
            "incumbent_arm_id": self.incumbent_arm_id,
        }

    def to_dict(self) -> dict[str, str]:
        return {**self._identity_payload(), "binding_id": self.binding_id}


def _cohort_identity_hash(manifest: EvaluationManifest, cohort: EvaluationCohort) -> str:
    """Hash the exact frozen cohort without coupling it to arm or policy configuration."""
    cases = [
        {
            "case_id": case.case_id,
            "snapshot_id": case.snapshot_id,
            "snapshot_artifact_hash": case.snapshot_artifact_hash,
        }
        for case in manifest.cases
        if case.cohort is cohort
    ]
    return content_hash({"cohort": cohort.value, "cases": cases})


def build_replay_evidence_binding(
    manifest: EvaluationManifest,
    scorecard: MatchedArmScorecard,
    inventory: EvaluationArtifactInventory,
    *,
    target_arm_id: str,
    incumbent_arm_id: str,
) -> ReplayEvidenceBinding:
    """Create the hashes a separate cohort comparator must include with its measurements."""
    if scorecard.manifest_id != manifest.manifest_id or inventory.manifest_id != manifest.manifest_id:
        raise EvaluationValidationError("replay evidence inputs must bind the same manifest")

    return ReplayEvidenceBinding(
        scorecard_id=scorecard.scorecard_id,
        artifact_inventory_hash=content_hash(inventory.to_dict()),
        temporal_cohort_hash=_cohort_identity_hash(manifest, EvaluationCohort.TEMPORAL),
        holdout_cohort_hash=_cohort_identity_hash(manifest, EvaluationCohort.HOLDOUT),
        target_arm_id=target_arm_id,
        incumbent_arm_id=incumbent_arm_id,
    )


@dataclass(frozen=True)
class CocosPilotCaseAssignment:
    """One source-free, order-sensitive assignment in the accepted Cocos manifest."""

    case_id: str
    snapshot_id: str
    snapshot_artifact_hash: str
    cohort: str
    event: str
    parent_case_id: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise EvaluationValidationError("Cocos pilot case_id must be non-empty")
        _validate_hash("Cocos pilot snapshot_id", self.snapshot_id)
        _validate_hash("Cocos pilot snapshot_artifact_hash", self.snapshot_artifact_hash)
        if self.cohort not in {cohort.value for cohort in EvaluationCohort}:
            raise EvaluationValidationError("Cocos pilot cohort is invalid")
        if not isinstance(self.event, str) or not self.event.strip():
            raise EvaluationValidationError("Cocos pilot event must be non-empty")
        if self.parent_case_id is not None and (
            not isinstance(self.parent_case_id, str) or not self.parent_case_id.strip()
        ):
            raise EvaluationValidationError("Cocos pilot parent_case_id must be non-empty or null")

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "case_id": self.case_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
            "cohort": self.cohort,
            "event": self.event,
            "parent_case_id": self.parent_case_id,
        }


def _cocos_manifest_assignments(
    manifest: EvaluationManifest,
) -> tuple[CocosPilotCaseAssignment, ...]:
    return tuple(
        CocosPilotCaseAssignment(
            case_id=case.case_id,
            snapshot_id=case.snapshot_id,
            snapshot_artifact_hash=case.snapshot_artifact_hash,
            cohort=case.cohort.value,
            event=case.event.value,
            parent_case_id=case.parent_case_id,
        )
        for case in manifest.cases
    )


@dataclass(frozen=True)
class CocosPilotAcceptance:
    """Source-free corpus identities awaiting independent maintainer acceptance."""

    lock_id: str
    corpus_hash: str
    manifest_schema_version: str
    manifest_schema_hash: str
    assignments: tuple[CocosPilotCaseAssignment, ...]
    schema_version: str = COCOS_PILOT_ACCEPTANCE_SCHEMA_VERSION
    cohort_counts: Mapping[str, int] = field(init=False)
    holdout_cohort_hash: str = field(init=False)
    holdout_case_count: int = field(init=False)
    acceptance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != COCOS_PILOT_ACCEPTANCE_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"unsupported Cocos pilot acceptance schema_version: {self.schema_version}"
            )
        for name in ("lock_id", "corpus_hash", "manifest_schema_hash"):
            _validate_hash(name, getattr(self, name))
        if self.lock_id != _CANONICAL_COCOS_LOCK_ID:
            raise EvaluationValidationError("pilot acceptance does not use the canonical Cocos lock")
        if not isinstance(self.manifest_schema_version, str) or not self.manifest_schema_version.strip():
            raise EvaluationValidationError("pilot acceptance manifest schema version must be non-empty")
        object.__setattr__(self, "assignments", tuple(self.assignments))
        if not self.assignments or any(
            not isinstance(assignment, CocosPilotCaseAssignment) for assignment in self.assignments
        ):
            raise EvaluationValidationError("pilot acceptance requires Cocos case assignments")
        case_ids = tuple(assignment.case_id for assignment in self.assignments)
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("pilot acceptance case assignments must be unique")
        cohort_counts = {
            cohort.value: sum(assignment.cohort == cohort.value for assignment in self.assignments)
            for cohort in EvaluationCohort
        }
        expected_counts = {
            cohort.value: _CANONICAL_COCOS_COHORT_COUNTS[inventory_key]
            for cohort, inventory_key in _MANIFEST_COHORT_TO_COCOS_COUNT.items()
        }
        expected_counts[EvaluationCohort.THRESHOLD.value] = 0
        if cohort_counts != expected_counts:
            raise EvaluationValidationError(
                "pilot acceptance cohort counts do not match the canonical Cocos inventory"
            )
        object.__setattr__(self, "cohort_counts", MappingProxyType(cohort_counts))
        holdouts = tuple(
            assignment for assignment in self.assignments
            if assignment.cohort == EvaluationCohort.HOLDOUT.value
        )
        object.__setattr__(self, "holdout_case_count", len(holdouts))
        object.__setattr__(self, "holdout_cohort_hash", content_hash({
            "cohort": EvaluationCohort.HOLDOUT.value,
            "cases": [assignment.to_dict() for assignment in holdouts],
        }))
        if len(holdouts) != _FROZEN_HOLDOUT_MINIMUM_SUPPORT:
            raise EvaluationValidationError("pilot acceptance must contain exactly 18 holdout cases")
        object.__setattr__(self, "acceptance_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lock_id": self.lock_id,
            "corpus_hash": self.corpus_hash,
            "manifest_schema_version": self.manifest_schema_version,
            "manifest_schema_hash": self.manifest_schema_hash,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "cohort_counts": dict(self.cohort_counts),
            "holdout_cohort_hash": self.holdout_cohort_hash,
            "holdout_case_count": self.holdout_case_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "acceptance_id": self.acceptance_id}


def _validate_canonical_cocos_inventory(inventory: CocosCorpusInventory) -> None:
    if not isinstance(inventory, CocosCorpusInventory):
        raise EvaluationValidationError("pilot corpus inventory must be a CocosCorpusInventory")
    _validate_hash("Cocos inventory lock_id", inventory.lock_id)
    _validate_hash("Cocos inventory corpus_hash", inventory.corpus_hash)
    if inventory.lock_id != _CANONICAL_COCOS_LOCK_ID:
        raise EvaluationValidationError("pilot corpus inventory does not use the canonical Cocos lock")
    if dict(inventory.cohort_counts) != _CANONICAL_COCOS_COHORT_COUNTS:
        raise EvaluationValidationError("pilot corpus inventory does not match the canonical Cocos cohort counts")
    if (
        inventory.checkpoint_controls_status != "complete"
        or inventory.checkpoint_control_count is None
        or not 15 <= inventory.checkpoint_control_count <= 20
        or inventory.checkpoint_controls_hash is None
    ):
        raise EvaluationValidationError(
            "pilot corpus inventory requires 15 to 20 complete checkpoint controls"
        )
    _validate_hash("Cocos checkpoint_controls_hash", inventory.checkpoint_controls_hash)


def build_cocos_pilot_acceptance(
    manifest: EvaluationManifest,
    inventory: CocosCorpusInventory,
) -> CocosPilotAcceptance:
    """Generate source-free identities for separate review; this does not accept them."""
    if not isinstance(manifest, EvaluationManifest):
        raise EvaluationValidationError("pilot acceptance requires an EvaluationManifest")
    _validate_canonical_cocos_inventory(inventory)
    if inventory.corpus_hash != manifest.corpus_hash:
        raise EvaluationValidationError("pilot manifest and inventory do not bind the same corpus")
    manifest_counts = {
        cohort: sum(case.cohort is cohort for case in manifest.cases)
        for cohort in EvaluationCohort
    }
    expected_counts = {
        cohort: inventory.cohort_counts[inventory_key]
        for cohort, inventory_key in _MANIFEST_COHORT_TO_COCOS_COUNT.items()
    }
    expected_counts[EvaluationCohort.THRESHOLD] = 0
    if manifest_counts != expected_counts:
        raise EvaluationValidationError(
            "pilot manifest cohort counts do not match the canonical Cocos inventory"
        )
    return CocosPilotAcceptance(
        lock_id=inventory.lock_id,
        corpus_hash=inventory.corpus_hash,
        manifest_schema_version=manifest.schema_version,
        manifest_schema_hash=manifest.schema_hash,
        assignments=_cocos_manifest_assignments(manifest),
    )


@dataclass(frozen=True)
class CocosPilotCorpusBinding:
    """Publishable proof that the pilot uses the independently accepted Cocos corpus."""

    acceptance_id: str
    lock_id: str
    inventory_hash: str
    corpus_hash: str
    holdout_cohort_hash: str
    holdout_case_count: int

    def __post_init__(self) -> None:
        for name in (
            "acceptance_id", "lock_id", "inventory_hash", "corpus_hash", "holdout_cohort_hash",
        ):
            _validate_hash(name, getattr(self, name))
        if self.acceptance_id != CANONICAL_COCOS_PILOT_ACCEPTANCE_ID:
            raise EvaluationValidationError("pilot corpus binding is not independently accepted")
        if self.lock_id != _CANONICAL_COCOS_LOCK_ID:
            raise EvaluationValidationError("pilot corpus binding does not use the canonical Cocos lock")
        if self.holdout_case_count != _FROZEN_HOLDOUT_MINIMUM_SUPPORT:
            raise EvaluationValidationError("pilot corpus binding must contain exactly 18 holdout cases")

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_id": self.acceptance_id,
            "lock_id": self.lock_id,
            "inventory_hash": self.inventory_hash,
            "corpus_hash": self.corpus_hash,
            "holdout_cohort_hash": self.holdout_cohort_hash,
            "holdout_case_count": self.holdout_case_count,
        }


def _build_cocos_pilot_corpus_binding(
    manifest: EvaluationManifest,
    inventory: Optional[CocosCorpusInventory],
    acceptance: Optional[CocosPilotAcceptance],
) -> Optional[CocosPilotCorpusBinding]:
    """Bind the manifest only when a separately pinned acceptance artifact matches it."""
    if (
        inventory is None
        or acceptance is None
        or CANONICAL_COCOS_PILOT_ACCEPTANCE_ID is None
    ):
        return None
    _validate_canonical_cocos_inventory(inventory)
    if not isinstance(acceptance, CocosPilotAcceptance):
        raise EvaluationValidationError("pilot acceptance must be a CocosPilotAcceptance")
    if acceptance.acceptance_id != CANONICAL_COCOS_PILOT_ACCEPTANCE_ID:
        raise EvaluationValidationError("pilot acceptance id is not independently accepted")
    if (
        inventory.lock_id != acceptance.lock_id
        or inventory.corpus_hash != acceptance.corpus_hash
        or manifest.corpus_hash != acceptance.corpus_hash
        or manifest.schema_version != acceptance.manifest_schema_version
        or manifest.schema_hash != acceptance.manifest_schema_hash
    ):
        raise EvaluationValidationError("pilot manifest and inventory do not match the accepted corpus")
    actual_assignments = _cocos_manifest_assignments(manifest)
    if actual_assignments != acceptance.assignments:
        raise EvaluationValidationError("pilot manifest assignments do not exactly match the accepted Cocos inventory")
    holdout_case_count = acceptance.holdout_case_count
    actual_holdout_hash = acceptance.holdout_cohort_hash
    inventory_hash = content_hash({
        "adapter_schema_version": COCOS_ADAPTER_SCHEMA_VERSION,
        "lock_id": inventory.lock_id,
        "corpus_hash": inventory.corpus_hash,
        "cohort_counts": dict(sorted(inventory.cohort_counts.items())),
        "checkpoint_control_count": inventory.checkpoint_control_count,
        "checkpoint_controls_status": inventory.checkpoint_controls_status,
        "checkpoint_controls_hash": inventory.checkpoint_controls_hash,
    })
    return CocosPilotCorpusBinding(
        acceptance_id=acceptance.acceptance_id,
        lock_id=inventory.lock_id,
        inventory_hash=inventory_hash,
        corpus_hash=inventory.corpus_hash,
        holdout_cohort_hash=actual_holdout_hash,
        holdout_case_count=holdout_case_count,
    )


@dataclass(frozen=True)
class ShadowPilotAcceptance:
    """Exact source-free journal inventory awaiting independent maintainer acceptance."""

    manifest_id: str
    policy_hash: str
    configuration_hash: str
    arm_id: str
    record_inventory: tuple[Mapping[str, Any], ...]
    schema_version: str = SHADOW_PILOT_ACCEPTANCE_SCHEMA_VERSION
    journal_hash: str = field(init=False)
    acceptance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SHADOW_PILOT_ACCEPTANCE_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"unsupported shadow pilot acceptance schema_version: {self.schema_version}"
            )
        for name in ("manifest_id", "policy_hash", "configuration_hash"):
            _validate_hash(name, getattr(self, name))
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise EvaluationValidationError("shadow pilot acceptance arm_id must be non-empty")
        try:
            inventory = json.loads(json.dumps(
                [_thaw_json(item) for item in self.record_inventory],
                allow_nan=False,
                sort_keys=True,
            ))
        except (TypeError, ValueError) as exc:
            raise EvaluationValidationError("shadow pilot inventory must contain finite JSON values") from exc
        if not inventory or any(not isinstance(item, dict) for item in inventory):
            raise EvaluationValidationError("shadow pilot acceptance requires a record inventory")
        record_ids = tuple(item.get("record_id") for item in inventory)
        entry_ids = tuple(
            item.get("entry", {}).get("entry_id")
            if isinstance(item.get("entry"), dict)
            else None
            for item in inventory
        )
        for identity in (*record_ids, *entry_ids):
            _validate_hash("shadow pilot record identity", identity)
        if len(record_ids) != len(set(record_ids)) or len(entry_ids) != len(set(entry_ids)):
            raise EvaluationValidationError("shadow pilot record and entry identities must be unique")
        object.__setattr__(self, "record_inventory", tuple(_freeze_json(item) for item in inventory))
        object.__setattr__(self, "journal_hash", content_hash({
            "record_inventory": inventory,
        }))
        object.__setattr__(self, "acceptance_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "policy_hash": self.policy_hash,
            "configuration_hash": self.configuration_hash,
            "arm_id": self.arm_id,
            "record_inventory": [_thaw_json(item) for item in self.record_inventory],
            "journal_hash": self.journal_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "acceptance_id": self.acceptance_id}


def build_shadow_pilot_acceptance(
    records: Sequence[ShadowJournalRecord],
    *,
    manifest: EvaluationManifest,
    target_arm_id: str,
) -> ShadowPilotAcceptance:
    """Generate exact journal identities for separate review; this does not accept them."""
    if not isinstance(manifest, EvaluationManifest):
        raise EvaluationValidationError("shadow pilot acceptance requires an EvaluationManifest")
    records = tuple(records)
    if not records or any(not isinstance(record, ShadowJournalRecord) for record in records):
        raise EvaluationValidationError("shadow pilot acceptance requires journal records")
    observed_at_utc = tuple(record.ingested_at_utc for record in records)
    if any(
        later < earlier
        for earlier, later in zip(observed_at_utc, observed_at_utc[1:], strict=False)
    ):
        raise EvaluationValidationError("shadow journal records must remain in observed UTC order")
    record_ids = tuple(record.record_id for record in records)
    entry_ids = tuple(record.entry.entry_id for record in records)
    if len(record_ids) != len(set(record_ids)) or len(entry_ids) != len(set(entry_ids)):
        raise EvaluationValidationError("shadow journal records and entries must be unique")
    target_arm = next(
        (arm for arm in manifest.arms if arm.arm_id == target_arm_id and arm.enabled),
        None,
    )
    if target_arm is None:
        raise EvaluationValidationError("shadow pilot target arm must be enabled in the manifest")
    for index, record in enumerate(records):
        if (
            record.entry.policy_hash != manifest.policy_hash
            or record.entry.configuration_hash != manifest.configuration_hash
            or record.entry.arm_id != target_arm_id
        ):
            raise EvaluationValidationError(
                "shadow journal entry does not match the pilot manifest and target arm"
            )
        for name, measurement in (
            ("shadow latency_seconds", record.entry.latency_seconds),
            ("shadow tokens", record.entry.tokens),
            ("shadow cost_usd", record.entry.cost_usd),
        ):
            _validate_measurement_non_negative(name, measurement)
        for stage in record.entry.stage_runs:
            for name, measurement in (
                (f"shadow stage {stage.stage} latency_seconds", stage.latency_seconds),
                (f"shadow stage {stage.stage} tokens", stage.tokens),
                (f"shadow stage {stage.stage} cost_usd", stage.cost_usd),
            ):
                _validate_measurement_non_negative(name, measurement)
            for model_id, cost_usd in stage.cost_by_model_usd.items():
                _validate_optional_non_negative(
                    f"shadow stage {stage.stage} cost for {model_id}",
                    cost_usd,
                )
        synthetic_record = EvaluationRunRecord(
            manifest_id=manifest.manifest_id,
            case_id=f"shadow-record-{index}",
            arm_id=record.entry.arm_id,
            snapshot_id=record.entry.snapshot_id,
            attempt=1,
            state=record.entry.result_state,
            terminal=True,
            latency_seconds=record.entry.latency_seconds,
            tokens=record.entry.tokens,
            cost_usd=record.entry.cost_usd,
            retry_count=record.entry.retry_count,
            cached=record.entry.cached,
            model_id=record.entry.model_id,
            provider_id=record.entry.provider_id,
            model_revision=record.entry.model_revision,
            stage_runs=record.entry.stage_runs,
        )
        validate_run_model_telemetry(
            target_arm,
            synthetic_record,
            context="shadow journal entry",
        )
    return ShadowPilotAcceptance(
        manifest_id=manifest.manifest_id,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
        arm_id=target_arm_id,
        record_inventory=tuple(record.to_dict() for record in records),
    )


@dataclass(frozen=True)
class ShadowPilotBinding:
    """Publishable journal inventory and metrics recomputed from independently accepted records."""

    acceptance_id: str
    journal_hash: str
    record_artifact_hashes: tuple[str, ...]
    started_at_utc: datetime
    ended_at_utc: datetime
    event_count: int
    file_save_count: int
    worktree_count: int
    developer_elapsed_seconds: Optional[float]
    latency_p95_seconds: ScoreMetric
    cost_per_developer_hour_usd: ScoreMetric

    def __post_init__(self) -> None:
        for name in ("acceptance_id", "journal_hash"):
            _validate_hash(name, getattr(self, name))
        if self.acceptance_id != CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID:
            raise EvaluationValidationError("shadow pilot binding is not independently accepted")
        object.__setattr__(self, "record_artifact_hashes", tuple(self.record_artifact_hashes))
        if not self.record_artifact_hashes or len(self.record_artifact_hashes) != len(
            set(self.record_artifact_hashes)
        ):
            raise EvaluationValidationError("shadow pilot record inventory must be non-empty and unique")
        for record_hash in self.record_artifact_hashes:
            _validate_hash("shadow pilot record hash", record_hash)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.event_count, self.file_save_count, self.worktree_count)
        ):
            raise EvaluationValidationError("shadow pilot event counts must be non-negative integers")
        if self.event_count != len(self.record_artifact_hashes):
            raise EvaluationValidationError("shadow pilot event count must equal its exact inventory")
        if self.file_save_count + self.worktree_count > self.event_count:
            raise EvaluationValidationError("shadow pilot event subtype counts exceed the inventory")
        if self.started_at_utc.tzinfo is None or self.ended_at_utc.tzinfo is None:
            raise EvaluationValidationError("shadow pilot timestamps must be timezone-aware")
        if (
            self.started_at_utc.utcoffset() != timezone.utc.utcoffset(self.started_at_utc)
            or self.ended_at_utc.utcoffset() != timezone.utc.utcoffset(self.ended_at_utc)
        ):
            raise EvaluationValidationError("shadow pilot timestamps must be normalized to UTC")
        if self.ended_at_utc < self.started_at_utc:
            raise EvaluationValidationError("shadow pilot duration cannot move backwards")
        _validate_optional_non_negative(
            "shadow pilot developer_elapsed_seconds",
            self.developer_elapsed_seconds,
        )
        if not isinstance(self.latency_p95_seconds, ScoreMetric) or not isinstance(
            self.cost_per_developer_hour_usd,
            ScoreMetric,
        ):
            raise EvaluationValidationError("shadow pilot metrics must use ScoreMetric")

    def elapsed_days(self) -> ScoreMetric:
        if self.event_count < 2:
            return _unavailable_metric()
        elapsed = (self.ended_at_utc - self.started_at_utc).total_seconds() / 86400.0
        if elapsed <= 0:
            return _unavailable_metric()
        return ScoreMetric(MeasurementStatus.COMPLETE, elapsed, self.event_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_id": self.acceptance_id,
            "journal_hash": self.journal_hash,
            "record_artifact_hashes": list(self.record_artifact_hashes),
            "started_at_utc": self.started_at_utc.isoformat(),
            "ended_at_utc": self.ended_at_utc.isoformat(),
            "event_count": self.event_count,
            "file_save_count": self.file_save_count,
            "worktree_count": self.worktree_count,
            "developer_elapsed_seconds": self.developer_elapsed_seconds,
            "latency_p95_seconds": self.latency_p95_seconds.to_dict(),
            "cost_per_developer_hour_usd": self.cost_per_developer_hour_usd.to_dict(),
        }


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _journal_metric(
    records: Sequence[ShadowJournalRecord],
    attribute: str,
) -> ScoreMetric:
    measurements = tuple(getattr(record.entry, attribute) for record in records)
    known = tuple(
        float(measurement.value)
        for measurement in measurements
        if measurement.status is MeasurementStatus.COMPLETE and measurement.value is not None
    )
    if not known:
        return _unavailable_metric()
    status = (
        MeasurementStatus.COMPLETE
        if len(known) == len(measurements)
        else MeasurementStatus.PARTIAL
    )
    return ScoreMetric(status, _p95(known), len(known))


def _journal_cost_per_developer_hour(
    records: Sequence[ShadowJournalRecord],
) -> ScoreMetric:
    interval_records = tuple(
        record
        for record in records
        if record.developer_time_basis is DeveloperTimeBasis.WRITER_MONOTONIC
    )
    known_intervals = tuple(
        record
        for record in interval_records
        if record.developer_elapsed_seconds is not None
    )
    known_costs = tuple(
        record
        for record in records
        if record.entry.cost_usd.status is MeasurementStatus.COMPLETE
        and record.entry.cost_usd.value is not None
    )
    denominator_seconds = sum(record.developer_elapsed_seconds or 0.0 for record in known_intervals)
    if not known_costs or denominator_seconds <= 0:
        return _unavailable_metric()
    total_cost = sum(float(record.entry.cost_usd.value) for record in known_costs)
    status = (
        MeasurementStatus.COMPLETE
        if len(known_costs) == len(records) and len(known_intervals) == len(interval_records)
        else MeasurementStatus.PARTIAL
    )
    return ScoreMetric(status, total_cost / (denominator_seconds / 3600.0), len(known_costs))


def _build_shadow_pilot_binding(
    records: Sequence[ShadowJournalRecord],
    acceptance: Optional[ShadowPilotAcceptance],
    *,
    manifest: EvaluationManifest,
    target_arm_id: str,
) -> Optional[ShadowPilotBinding]:
    records = tuple(records)
    if not records or acceptance is None or CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID is None:
        return None
    if not isinstance(acceptance, ShadowPilotAcceptance):
        raise EvaluationValidationError("shadow acceptance must be a ShadowPilotAcceptance")
    if acceptance.acceptance_id != CANONICAL_SHADOW_PILOT_ACCEPTANCE_ID:
        raise EvaluationValidationError("shadow acceptance id is not independently accepted")
    actual_acceptance = build_shadow_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id=target_arm_id,
    )
    if actual_acceptance != acceptance:
        raise EvaluationValidationError("shadow records do not exactly match the accepted journal inventory")
    observed_at_utc = tuple(record.ingested_at_utc for record in records)
    developer_seconds = tuple(
        record.developer_elapsed_seconds
        for record in records
        if record.developer_time_basis is DeveloperTimeBasis.WRITER_MONOTONIC
    )
    return ShadowPilotBinding(
        acceptance_id=acceptance.acceptance_id,
        journal_hash=acceptance.journal_hash,
        record_artifact_hashes=tuple(record.record_id for record in records),
        started_at_utc=observed_at_utc[0],
        ended_at_utc=observed_at_utc[-1],
        event_count=len(records),
        file_save_count=sum(record.entry.event is ReviewEvent.FILE_SAVE for record in records),
        worktree_count=sum(record.entry.event is ReviewEvent.WORKTREE_IDLE for record in records),
        developer_elapsed_seconds=(
            sum(float(value) for value in developer_seconds if value is not None)
            if developer_seconds and all(value is not None for value in developer_seconds)
            else None
        ),
        latency_p95_seconds=_journal_metric(records, "latency_seconds"),
        cost_per_developer_hour_usd=_journal_cost_per_developer_hour(records),
    )


@dataclass(frozen=True)
class SettledCandidateRecord:
    """One source-free, independently adjudicated settled candidate outcome."""

    candidate_id: str
    adjudication_hash: str
    actionable: bool
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_hash("settled candidate_id", self.candidate_id)
        _validate_hash("settled adjudication_hash", self.adjudication_hash)
        if not isinstance(self.actionable, bool):
            raise EvaluationValidationError("settled candidate actionable must be a boolean")
        object.__setattr__(self, "record_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "adjudication_hash": self.adjudication_hash,
            "actionable": self.actionable,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "record_id": self.record_id}


@dataclass(frozen=True)
class SettledPilotAcceptance:
    """Exact settled-candidate inventory awaiting independent maintainer acceptance."""

    manifest_id: str
    policy_hash: str
    configuration_hash: str
    arm_id: str
    record_inventory: tuple[Mapping[str, Any], ...]
    schema_version: str = SETTLED_PILOT_ACCEPTANCE_SCHEMA_VERSION
    inventory_hash: str = field(init=False)
    acceptance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SETTLED_PILOT_ACCEPTANCE_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"unsupported settled pilot acceptance schema_version: {self.schema_version}"
            )
        for name in ("manifest_id", "policy_hash", "configuration_hash"):
            _validate_hash(name, getattr(self, name))
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise EvaluationValidationError("settled pilot acceptance arm_id must be non-empty")
        try:
            inventory = json.loads(json.dumps(
                [_thaw_json(item) for item in self.record_inventory],
                allow_nan=False,
                sort_keys=True,
            ))
        except (TypeError, ValueError) as exc:
            raise EvaluationValidationError(
                "settled pilot inventory must contain finite JSON values"
            ) from exc
        if not inventory or any(not isinstance(item, dict) for item in inventory):
            raise EvaluationValidationError("settled pilot acceptance requires a record inventory")
        expected_fields = {"candidate_id", "adjudication_hash", "actionable", "record_id"}
        if any(set(item) != expected_fields for item in inventory):
            raise EvaluationValidationError("settled pilot inventory contains unsupported fields")
        candidate_ids = tuple(item["candidate_id"] for item in inventory)
        record_ids = tuple(item["record_id"] for item in inventory)
        for identity in (*candidate_ids, *record_ids):
            _validate_hash("settled pilot record identity", identity)
        if len(candidate_ids) != len(set(candidate_ids)) or len(record_ids) != len(set(record_ids)):
            raise EvaluationValidationError("settled pilot candidate and record identities must be unique")
        for item in inventory:
            _validate_hash("settled pilot adjudication_hash", item["adjudication_hash"])
            if not isinstance(item["actionable"], bool):
                raise EvaluationValidationError("settled pilot actionable values must be booleans")
            expected_record_id = content_hash({
                "candidate_id": item["candidate_id"],
                "adjudication_hash": item["adjudication_hash"],
                "actionable": item["actionable"],
            })
            if item["record_id"] != expected_record_id:
                raise EvaluationValidationError("settled pilot record id does not match its content")
        object.__setattr__(self, "record_inventory", tuple(_freeze_json(item) for item in inventory))
        object.__setattr__(self, "inventory_hash", content_hash({"record_inventory": inventory}))
        object.__setattr__(self, "acceptance_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "policy_hash": self.policy_hash,
            "configuration_hash": self.configuration_hash,
            "arm_id": self.arm_id,
            "record_inventory": [_thaw_json(item) for item in self.record_inventory],
            "inventory_hash": self.inventory_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "acceptance_id": self.acceptance_id}


def build_settled_pilot_acceptance(
    records: Sequence[SettledCandidateRecord],
    *,
    manifest: EvaluationManifest,
    target_arm_id: str,
) -> SettledPilotAcceptance:
    """Generate settled-candidate identities for separate review; this does not accept them."""
    if not isinstance(manifest, EvaluationManifest):
        raise EvaluationValidationError("settled pilot acceptance requires an EvaluationManifest")
    target_arm = next(
        (arm for arm in manifest.arms if arm.arm_id == target_arm_id and arm.enabled),
        None,
    )
    if target_arm is None:
        raise EvaluationValidationError("settled pilot target arm must be enabled in the manifest")
    records = tuple(records)
    if not records or any(not isinstance(record, SettledCandidateRecord) for record in records):
        raise EvaluationValidationError("settled pilot acceptance requires candidate records")
    candidate_ids = tuple(record.candidate_id for record in records)
    record_ids = tuple(record.record_id for record in records)
    if len(candidate_ids) != len(set(candidate_ids)) or len(record_ids) != len(set(record_ids)):
        raise EvaluationValidationError("settled pilot candidate and record identities must be unique")
    return SettledPilotAcceptance(
        manifest_id=manifest.manifest_id,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
        arm_id=target_arm_id,
        record_inventory=tuple(record.to_dict() for record in records),
    )


@dataclass(frozen=True)
class SettledPilotBinding:
    """Publishable proof and precision derived from independently accepted candidates."""

    acceptance_id: str
    inventory_hash: str
    record_artifact_hashes: tuple[str, ...]
    settled_count: int
    actionable_count: int
    actionable_precision: ScoreMetric

    def __post_init__(self) -> None:
        for name in ("acceptance_id", "inventory_hash"):
            _validate_hash(name, getattr(self, name))
        if self.acceptance_id != CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID:
            raise EvaluationValidationError("settled pilot binding is not independently accepted")
        object.__setattr__(self, "record_artifact_hashes", tuple(self.record_artifact_hashes))
        if not self.record_artifact_hashes or len(self.record_artifact_hashes) != len(
            set(self.record_artifact_hashes)
        ):
            raise EvaluationValidationError("settled pilot record inventory must be non-empty and unique")
        for record_hash in self.record_artifact_hashes:
            _validate_hash("settled pilot record hash", record_hash)
        if (
            not isinstance(self.settled_count, int)
            or isinstance(self.settled_count, bool)
            or self.settled_count != len(self.record_artifact_hashes)
        ):
            raise EvaluationValidationError("settled candidate count must equal its exact inventory")
        if (
            not isinstance(self.actionable_count, int)
            or isinstance(self.actionable_count, bool)
            or not 0 <= self.actionable_count <= self.settled_count
        ):
            raise EvaluationValidationError("settled actionable count must be within the inventory")
        expected_precision = self.actionable_count / self.settled_count
        if (
            not isinstance(self.actionable_precision, ScoreMetric)
            or self.actionable_precision.status is not MeasurementStatus.COMPLETE
            or self.actionable_precision.support != self.settled_count
            or self.actionable_precision.value != expected_precision
        ):
            raise EvaluationValidationError("settled actionable precision must be derived from the inventory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_id": self.acceptance_id,
            "inventory_hash": self.inventory_hash,
            "record_artifact_hashes": list(self.record_artifact_hashes),
            "settled_count": self.settled_count,
            "actionable_count": self.actionable_count,
            "actionable_precision": self.actionable_precision.to_dict(),
        }


def _build_settled_pilot_binding(
    records: Sequence[SettledCandidateRecord],
    acceptance: Optional[SettledPilotAcceptance],
    *,
    manifest: EvaluationManifest,
    target_arm_id: str,
) -> Optional[SettledPilotBinding]:
    records = tuple(records)
    if not records or acceptance is None or CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID is None:
        return None
    if not isinstance(acceptance, SettledPilotAcceptance):
        raise EvaluationValidationError("settled acceptance must be a SettledPilotAcceptance")
    if acceptance.acceptance_id != CANONICAL_SETTLED_PILOT_ACCEPTANCE_ID:
        raise EvaluationValidationError("settled acceptance id is not independently accepted")
    actual_acceptance = build_settled_pilot_acceptance(
        records,
        manifest=manifest,
        target_arm_id=target_arm_id,
    )
    if actual_acceptance != acceptance:
        raise EvaluationValidationError(
            "settled candidates do not exactly match the accepted record inventory"
        )
    actionable_count = sum(record.actionable for record in records)
    return SettledPilotBinding(
        acceptance_id=acceptance.acceptance_id,
        inventory_hash=acceptance.inventory_hash,
        record_artifact_hashes=tuple(record.record_id for record in records),
        settled_count=len(records),
        actionable_count=actionable_count,
        actionable_precision=ScoreMetric(
            MeasurementStatus.COMPLETE,
            actionable_count / len(records),
            len(records),
        ),
    )


@dataclass(frozen=True)
class HoldoutLeakageCheck:
    """Content-bound result awaiting independent pinning before it can satisfy a gate."""

    manifest_id: str
    holdout_cohort_hash: str
    model_visible_inventory_hash: str
    checker_revision_hash: str
    leakage_free: bool
    check_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "manifest_id", "holdout_cohort_hash", "model_visible_inventory_hash", "checker_revision_hash",
        ):
            _validate_hash(name, getattr(self, name))
        if not isinstance(self.leakage_free, bool):
            raise EvaluationValidationError("holdout leakage result must be a boolean")
        object.__setattr__(self, "check_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "holdout_cohort_hash": self.holdout_cohort_hash,
            "model_visible_inventory_hash": self.model_visible_inventory_hash,
            "checker_revision_hash": self.checker_revision_hash,
            "leakage_free": self.leakage_free,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "check_id": self.check_id}


def build_holdout_leakage_check(
    manifest: EvaluationManifest,
    *,
    checker_revision_hash: str,
    leakage_free: bool,
) -> HoldoutLeakageCheck:
    """Bind an external checker verdict to the exact model-visible holdout inventory."""
    if not isinstance(manifest, EvaluationManifest):
        raise EvaluationValidationError("holdout leakage check requires an EvaluationManifest")
    inventory = tuple(
        {
            "case_id": case.case_id,
            "snapshot_id": case.snapshot_id,
            "snapshot_artifact_hash": case.snapshot_artifact_hash,
            "model_visible_payload": case.model_visible_payload(),
        }
        for case in manifest.cases
        if case.cohort is EvaluationCohort.HOLDOUT
    )
    if not inventory:
        raise EvaluationValidationError("holdout leakage check requires holdout cases")
    return HoldoutLeakageCheck(
        manifest_id=manifest.manifest_id,
        holdout_cohort_hash=_cohort_identity_hash(manifest, EvaluationCohort.HOLDOUT),
        model_visible_inventory_hash=content_hash({"holdouts": inventory}),
        checker_revision_hash=checker_revision_hash,
        leakage_free=leakage_free,
    )


def _accepted_holdout_leakage(
    manifest: EvaluationManifest,
    check: Optional[HoldoutLeakageCheck],
) -> Optional[bool]:
    if check is None or CANONICAL_HOLDOUT_LEAKAGE_CHECK_ID is None:
        return None
    if not isinstance(check, HoldoutLeakageCheck):
        raise EvaluationValidationError("holdout leakage evidence must use HoldoutLeakageCheck")
    actual = build_holdout_leakage_check(
        manifest,
        checker_revision_hash=check.checker_revision_hash,
        leakage_free=check.leakage_free,
    )
    if actual != check or check.check_id != CANONICAL_HOLDOUT_LEAKAGE_CHECK_ID:
        raise EvaluationValidationError("holdout leakage evidence is not independently accepted for this manifest")
    return check.leakage_free


@dataclass(frozen=True)
class PilotRolloutEvidence:
    """Source-free facts that are not derivable from the frozen replay scorecard."""

    holdout_leakage_check: Optional[HoldoutLeakageCheck] = None
    replay_binding: Optional[ReplayEvidenceBinding] = None

    def __post_init__(self) -> None:
        if self.holdout_leakage_check is not None and not isinstance(
            self.holdout_leakage_check, HoldoutLeakageCheck
        ):
            raise EvaluationValidationError("holdout_leakage_check must use HoldoutLeakageCheck when supplied")
        if self.replay_binding is not None and not isinstance(self.replay_binding, ReplayEvidenceBinding):
            raise EvaluationValidationError("replay_binding must use ReplayEvidenceBinding when supplied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_leakage_check": (
                self.holdout_leakage_check.to_dict() if self.holdout_leakage_check else None
            ),
            "replay_binding": self.replay_binding.to_dict() if self.replay_binding else None,
        }


@dataclass(frozen=True)
class CanonicalRolloutGateSpec:
    gate_name: str
    rules: tuple[GateRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.gate_name, str) or not self.gate_name.strip():
            raise EvaluationValidationError("canonical gate name must be a non-empty string")
        object.__setattr__(self, "rules", tuple(self.rules))
        if not self.rules or any(not isinstance(rule, GateRule) for rule in self.rules):
            raise EvaluationValidationError("canonical gates require GateRule values")

    def to_dict(self) -> dict[str, Any]:
        return {"gate_name": self.gate_name, "rules": [rule.to_dict() for rule in self.rules]}


def canonical_rollout_gate_specs(budgets: PilotRolloutBudgets) -> tuple[CanonicalRolloutGateSpec, ...]:
    """Return the exact five issue #27 gate specifications for the accepted budgets."""
    if not isinstance(budgets, PilotRolloutBudgets):
        raise EvaluationValidationError("canonical rollout gates require PilotRolloutBudgets")
    live_shadow_rules = [
        GateRule("evidence.shadow_elapsed_days", GateComparator.AT_LEAST, 7.0),
        GateRule("evidence.shadow_file_save_count", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.shadow_worktree_count", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.raw_shadow_inventory_complete", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.shadow_latency_budget_accepted", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.shadow_cost_budget_accepted", GateComparator.AT_LEAST, 1.0),
    ]
    if budgets.shadow_latency_p95_seconds is not None:
        live_shadow_rules.append(GateRule(
            "evidence.shadow_latency_p95_seconds",
            GateComparator.AT_MOST,
            budgets.shadow_latency_p95_seconds,
        ))
    if budgets.shadow_cost_per_developer_hour_usd is not None:
        live_shadow_rules.append(GateRule(
            "evidence.shadow_cost_per_developer_hour_usd",
            GateComparator.AT_MOST,
            budgets.shadow_cost_per_developer_hour_usd,
        ))
    default_pair_rules = [
        GateRule("evidence.settled_actionable_precision", GateComparator.AT_LEAST, 0.90, 100),
        GateRule("evidence.settled_candidate_inventory_complete", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.false_interruption_threshold_accepted", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.raw_shadow_inventory_complete", GateComparator.AT_LEAST, 1.0),
    ]
    if budgets.accepted_false_interruptions_per_clean_checkpoint is not None:
        default_pair_rules.append(GateRule(
            "false_interruptions_per_clean_checkpoint",
            GateComparator.AT_MOST,
            budgets.accepted_false_interruptions_per_clean_checkpoint,
        ))
    publication_rules = [
        GateRule("evidence.frozen_holdout_binding_complete", GateComparator.AT_LEAST, 1.0),
        GateRule(
            "evidence.holdout_quality_advantage_lower_95",
            GateComparator.AT_LEAST,
            _MINIMUM_POSITIVE_ADVANTAGE,
            _FROZEN_HOLDOUT_MINIMUM_SUPPORT,
        ),
        GateRule("evidence.publication_cost_ceiling_accepted", GateComparator.AT_LEAST, 1.0),
        GateRule("evidence.raw_replay_inventory_complete", GateComparator.AT_LEAST, 1.0),
    ]
    if budgets.publication_cost_ceiling_usd is not None:
        publication_rules.append(GateRule(
            "evidence.holdout_cost_usd",
            GateComparator.AT_MOST,
            budgets.publication_cost_ceiling_usd,
        ))
    return (
        CanonicalRolloutGateSpec(OFFLINE_REPLAY_GATE, (
            GateRule("structured_output_rate", GateComparator.AT_LEAST, 0.995),
            GateRule("evidence.raw_replay_inventory_complete", GateComparator.AT_LEAST, 1.0),
            GateRule("evidence.holdout_leakage_free", GateComparator.AT_LEAST, 1.0),
            GateRule("evidence.bounded_retry_configured", GateComparator.AT_LEAST, 1.0),
        )),
        CanonicalRolloutGateSpec(LIVE_SHADOW_GATE, tuple(live_shadow_rules)),
        CanonicalRolloutGateSpec(OPT_IN_PAIR_REVIEW_GATE, (
            GateRule("verified_precision", GateComparator.AT_LEAST, 0.80),
            GateRule("evidence.temporal_high_severity_recall_delta", GateComparator.AT_LEAST, 0.0),
            GateRule("evidence.raw_replay_inventory_complete", GateComparator.AT_LEAST, 1.0),
        )),
        CanonicalRolloutGateSpec(DEFAULT_PAIR_REVIEW_GATE, tuple(default_pair_rules)),
        CanonicalRolloutGateSpec(PR_PUBLICATION_GATE, tuple(publication_rules)),
    )


def _presence_metric(value: Optional[bool]) -> ScoreMetric:
    if value is None:
        return _unavailable_metric()
    return ScoreMetric(MeasurementStatus.COMPLETE, 1.0 if value else 0.0, 1)


def _count_metric(value: Optional[int]) -> ScoreMetric:
    if value is None:
        return _unavailable_metric()
    return ScoreMetric(MeasurementStatus.COMPLETE, float(value), value)


def _terminal_by_pair(
    records: Sequence[EvaluationRunRecord],
) -> Mapping[tuple[str, str], EvaluationRunRecord]:
    return {
        (record.case_id, record.arm_id): record
        for record in records
        if record.terminal
    }


def _temporal_high_severity_recall_delta(
    manifest: EvaluationManifest,
    truth: TruthArtifact,
    records: Sequence[EvaluationRunRecord],
    *,
    target_arm_id: str,
    incumbent_arm_id: str,
) -> ScoreMetric:
    """Derive the temporal high/critical recall delta from retained raw attempts."""
    truth_by_case = {item.case_id: item for item in truth.truths}
    terminal_by_pair = _terminal_by_pair(records)
    target_hits = 0
    incumbent_hits = 0
    expected_count = 0
    complete = True
    for case in manifest.cases:
        if case.cohort is not EvaluationCohort.TEMPORAL:
            continue
        expected = {
            finding.fingerprint
            for finding in truth_by_case[case.case_id].findings
            if finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
        }
        if not expected:
            continue
        expected_count += len(expected)
        for arm_id, accumulator in (
            (target_arm_id, "target"),
            (incumbent_arm_id, "incumbent"),
        ):
            terminal = terminal_by_pair.get((case.case_id, arm_id))
            if terminal is None or terminal.state is not EvaluationRunState.COMPLETED:
                complete = False
                observed: set[str] = set()
            else:
                observed = {
                    finding.fingerprint
                    for finding in terminal.findings
                    if finding.lifecycle_state is FindingLifecycleState.ACTIVE
                }
            matched = len(expected & observed)
            if accumulator == "target":
                target_hits += matched
            else:
                incumbent_hits += matched
    if expected_count == 0:
        return _unavailable_metric()
    status = MeasurementStatus.COMPLETE if complete else MeasurementStatus.PARTIAL
    return ScoreMetric(status, (target_hits - incumbent_hits) / expected_count, expected_count)


def _score_cohort(
    manifest: EvaluationManifest,
    truth: TruthArtifact,
    records: Sequence[EvaluationRunRecord],
    cohort: EvaluationCohort,
    *,
    incumbent_arm_id: str,
) -> Optional[MatchedArmScorecard]:
    cases = tuple(case for case in manifest.cases if case.cohort is cohort)
    if not cases:
        return None
    case_ids = {case.case_id for case in cases}
    projected_manifest = EvaluationManifest(
        name=f"{manifest.name}:{cohort.value}",
        corpus_hash=manifest.corpus_hash,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
        cases=cases,
        arms=manifest.arms,
        schema_version=manifest.schema_version,
        schema_hash=manifest.schema_hash,
    )
    projected_truth = TruthArtifact(
        manifest_id=projected_manifest.manifest_id,
        truths=tuple(item for item in truth.truths if item.case_id in case_ids),
        schema_version=truth.schema_version,
    )
    projected_records = tuple(
        replace(record, manifest_id=projected_manifest.manifest_id)
        for record in records
        if record.case_id in case_ids
    )
    return score_matched_arms(
        projected_manifest,
        projected_truth,
        projected_records,
        baseline_arm_id=incumbent_arm_id,
    )


def _holdout_replay_metrics(
    manifest: EvaluationManifest,
    truth: TruthArtifact,
    records: Sequence[EvaluationRunRecord],
    *,
    target_arm_id: str,
    incumbent_arm_id: str,
) -> tuple[ScoreMetric, ScoreMetric]:
    """Derive frozen-holdout paired quality and target cost through the shared scorer."""
    scorecard = _score_cohort(
        manifest,
        truth,
        records,
        EvaluationCohort.HOLDOUT,
        incumbent_arm_id=incumbent_arm_id,
    )
    if scorecard is None:
        return _unavailable_metric(), _unavailable_metric()
    arms = {arm.arm_id: arm for arm in scorecard.arms}
    target = arms[target_arm_id]
    incumbent = arms[incumbent_arm_id]
    comparison = next(
        (
            item
            for item in scorecard.paired_comparisons
            if item.arm_id == target_arm_id
            and item.baseline_arm_id == incumbent_arm_id
            and item.metric == "case_recall"
        ),
        None,
    )
    target_structured = target.metrics["structured_output_rate"]
    incumbent_structured = incumbent.metrics["structured_output_rate"]
    coverage_complete = all(
        metric.status is MeasurementStatus.COMPLETE and metric.value == 1.0
        for metric in (target_structured, incumbent_structured)
    )
    if comparison is None:
        quality = _unavailable_metric()
    else:
        quality = ScoreMetric(
            MeasurementStatus.COMPLETE if coverage_complete else MeasurementStatus.PARTIAL,
            comparison.lower_95,
            comparison.support,
        )
    return quality, target.metrics["total_cost_usd"]


def _external_metrics(
    inventory: EvaluationArtifactInventory,
    evidence: PilotRolloutEvidence,
    budgets: PilotRolloutBudgets,
    expected_replay_binding: ReplayEvidenceBinding,
    holdout_leakage_free: Optional[bool],
    bounded_retry_configured: bool,
    temporal_high_severity_recall_delta: ScoreMetric,
    holdout_quality_advantage_lower_95: ScoreMetric,
    holdout_cost_usd: ScoreMetric,
    corpus_binding: Optional[CocosPilotCorpusBinding],
    shadow_binding: Optional[ShadowPilotBinding],
    settled_binding: Optional[SettledPilotBinding],
) -> Mapping[str, ScoreMetric]:
    if evidence.replay_binding is not None and evidence.replay_binding != expected_replay_binding:
        raise EvaluationValidationError("replay measurements do not bind the exact scorecard and artifact inventory")
    replay_evidence_bound = evidence.replay_binding is not None

    raw_replay_complete = (
        inventory.incomplete_pair_count == 0
        and inventory.terminal_pair_count > 0
        and bool(inventory.record_artifact_hashes)
    )
    return {
        "evidence.raw_replay_inventory_complete": _presence_metric(True if raw_replay_complete else None),
        "evidence.holdout_leakage_free": _presence_metric(
            holdout_leakage_free if replay_evidence_bound else None
        ),
        "evidence.bounded_retry_configured": _presence_metric(
            True if replay_evidence_bound and bounded_retry_configured else None
        ),
        "evidence.shadow_elapsed_days": (
            shadow_binding.elapsed_days() if shadow_binding is not None else _unavailable_metric()
        ),
        "evidence.shadow_file_save_count": _count_metric(
            shadow_binding.file_save_count if shadow_binding is not None else None
        ),
        "evidence.shadow_worktree_count": _count_metric(
            shadow_binding.worktree_count if shadow_binding is not None else None
        ),
        "evidence.raw_shadow_inventory_complete": _presence_metric(
            True if shadow_binding is not None else None
        ),
        "evidence.shadow_latency_budget_accepted": _presence_metric(
            True if budgets.shadow_latency_p95_seconds is not None else None
        ),
        "evidence.shadow_latency_p95_seconds": (
            shadow_binding.latency_p95_seconds
            if budgets.shadow_latency_p95_seconds is not None and shadow_binding is not None
            else _unavailable_metric()
        ),
        "evidence.shadow_cost_budget_accepted": _presence_metric(
            True if budgets.shadow_cost_per_developer_hour_usd is not None else None
        ),
        "evidence.shadow_cost_per_developer_hour_usd": (
            shadow_binding.cost_per_developer_hour_usd
            if budgets.shadow_cost_per_developer_hour_usd is not None and shadow_binding is not None
            else _unavailable_metric()
        ),
        "evidence.temporal_high_severity_recall_delta": temporal_high_severity_recall_delta,
        "evidence.settled_actionable_precision": (
            settled_binding.actionable_precision
            if settled_binding is not None
            else _unavailable_metric()
        ),
        "evidence.settled_candidate_inventory_complete": _presence_metric(
            True if settled_binding is not None else None
        ),
        "evidence.false_interruption_threshold_accepted": _presence_metric(
            True if budgets.accepted_false_interruptions_per_clean_checkpoint is not None else None
        ),
        "evidence.frozen_holdout_binding_complete": _presence_metric(
            True if corpus_binding is not None else None
        ),
        "evidence.holdout_quality_advantage_lower_95": (
            holdout_quality_advantage_lower_95
            if corpus_binding is not None
            else _unavailable_metric()
        ),
        "evidence.publication_cost_ceiling_accepted": _presence_metric(
            True if budgets.publication_cost_ceiling_usd is not None else None
        ),
        "evidence.holdout_cost_usd": (
            holdout_cost_usd
            if budgets.publication_cost_ceiling_usd is not None and corpus_binding is not None
            else _unavailable_metric()
        ),
    }


def _evaluate_canonical_gate(
    spec: CanonicalRolloutGateSpec,
    scorecard: MatchedArmScorecard,
    arm_id: str,
    external_metrics: Mapping[str, ScoreMetric],
) -> RolloutGateDecision:
    arm = next((item for item in scorecard.arms if item.arm_id == arm_id), None)
    if arm is None:
        raise EvaluationValidationError(f"scorecard does not contain arm {arm_id}")
    collisions = sorted(set(arm.metrics) & set(external_metrics))
    if collisions:
        raise EvaluationValidationError(f"pilot evidence collides with scorecard metrics: {collisions}")
    augmented_arm = replace(arm, metrics={**arm.metrics, **external_metrics})
    augmented_scorecard = MatchedArmScorecard(
        manifest_id=scorecard.manifest_id,
        truth_artifact_id=scorecard.truth_artifact_id,
        arms=tuple(augmented_arm if item.arm_id == arm_id else item for item in scorecard.arms),
        paired_comparisons=scorecard.paired_comparisons,
    )
    evaluated = evaluate_rollout_gate(spec.gate_name, augmented_scorecard, arm_id, spec.rules)
    return RolloutGateDecision(
        gate_name=evaluated.gate_name,
        arm_id=evaluated.arm_id,
        scorecard_id=scorecard.scorecard_id,
        status=evaluated.status,
        rule_results=evaluated.rule_results,
    )


@dataclass(frozen=True)
class CheckpointPilotReport:
    """Publishable source-free identity, inventory, measurements, and decisions."""

    manifest_id: str
    manifest_artifact_hash: str
    schema_hash: str
    corpus_hash: str
    policy_hash: str
    configuration_hash: str
    arm_configuration_hashes: tuple[str, ...]
    prompt_hashes: tuple[str, ...]
    model_identity_hashes: tuple[str, ...]
    corpus_binding: Optional[CocosPilotCorpusBinding]
    shadow_binding: Optional[ShadowPilotBinding]
    settled_candidate_binding: Optional[SettledPilotBinding]
    target_arm_id: str
    incumbent_arm_id: str
    raw_record_artifact_hashes: tuple[str, ...]
    raw_shadow_artifact_hashes: tuple[str, ...]
    raw_settled_candidate_artifact_hashes: tuple[str, ...]
    terminal_pair_count: int
    incomplete_pair_count: int
    scorecard_id: str
    scorecard: Mapping[str, Any]
    budgets: PilotRolloutBudgets
    evidence: PilotRolloutEvidence
    gate_decisions: tuple[RolloutGateDecision, ...]
    schema_version: str = PILOT_REPORT_SCHEMA_VERSION
    report_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PILOT_REPORT_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported pilot report schema_version: {self.schema_version}")
        for name in (
            "manifest_id",
            "manifest_artifact_hash",
            "schema_hash",
            "corpus_hash",
            "policy_hash",
            "configuration_hash",
            "scorecard_id",
        ):
            _validate_hash(name, getattr(self, name))
        for field_name in (
            "arm_configuration_hashes",
            "prompt_hashes",
            "model_identity_hashes",
            "raw_record_artifact_hashes",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))):
                raise EvaluationValidationError(f"{field_name} must be sorted and unique")
            for value in values:
                _validate_hash(field_name, value)
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "raw_shadow_artifact_hashes", tuple(self.raw_shadow_artifact_hashes))
        if len(self.raw_shadow_artifact_hashes) != len(set(self.raw_shadow_artifact_hashes)):
            raise EvaluationValidationError("raw_shadow_artifact_hashes must be unique")
        for value in self.raw_shadow_artifact_hashes:
            _validate_hash("raw_shadow_artifact_hashes", value)
        object.__setattr__(
            self,
            "raw_settled_candidate_artifact_hashes",
            tuple(self.raw_settled_candidate_artifact_hashes),
        )
        if len(self.raw_settled_candidate_artifact_hashes) != len(
            set(self.raw_settled_candidate_artifact_hashes)
        ):
            raise EvaluationValidationError(
                "raw_settled_candidate_artifact_hashes must be unique"
            )
        for value in self.raw_settled_candidate_artifact_hashes:
            _validate_hash("raw_settled_candidate_artifact_hashes", value)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.terminal_pair_count, self.incomplete_pair_count)
        ):
            raise EvaluationValidationError("artifact inventory counts must be non-negative integers")
        if not isinstance(self.scorecard, Mapping):
            raise EvaluationValidationError("pilot report scorecard must be a JSON object")
        try:
            scorecard = json.loads(json.dumps(dict(self.scorecard), allow_nan=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise EvaluationValidationError("pilot report scorecard must contain only finite JSON values") from exc
        if "truth_artifact_id" in scorecard:
            raise EvaluationValidationError("pilot report scorecard cannot expose the truth artifact id")
        if scorecard.get("manifest_id") != self.manifest_id or scorecard.get("scorecard_id") != self.scorecard_id:
            raise EvaluationValidationError("pilot report scorecard identities do not match the report")
        allowed_scorecard_fields = {
            "schema_version", "manifest_id", "arms", "paired_comparisons", "scorecard_id",
        }
        if set(scorecard) != allowed_scorecard_fields:
            raise EvaluationValidationError("pilot report scorecard contains unsupported fields")
        object.__setattr__(self, "scorecard", _freeze_json(scorecard))
        if not isinstance(self.budgets, PilotRolloutBudgets) or not isinstance(self.evidence, PilotRolloutEvidence):
            raise EvaluationValidationError("pilot report requires validated budgets and evidence")
        if self.corpus_binding is not None:
            if not isinstance(self.corpus_binding, CocosPilotCorpusBinding):
                raise EvaluationValidationError("pilot report corpus binding is invalid")
            if self.corpus_binding.corpus_hash != self.corpus_hash:
                raise EvaluationValidationError("pilot report corpus binding does not match the report")
        if self.shadow_binding is not None and not isinstance(self.shadow_binding, ShadowPilotBinding):
            raise EvaluationValidationError("pilot report shadow binding is invalid")
        if self.settled_candidate_binding is not None and not isinstance(
            self.settled_candidate_binding,
            SettledPilotBinding,
        ):
            raise EvaluationValidationError("pilot report settled candidate binding is invalid")
        if not self.target_arm_id.strip() or not self.incumbent_arm_id.strip():
            raise EvaluationValidationError("pilot report arm identifiers must be non-empty")
        expected_shadow_hashes = (
            self.shadow_binding.record_artifact_hashes if self.shadow_binding is not None else ()
        )
        if self.raw_shadow_artifact_hashes != expected_shadow_hashes:
            raise EvaluationValidationError("pilot report shadow inventory must match its accepted journal")
        expected_settled_hashes = (
            self.settled_candidate_binding.record_artifact_hashes
            if self.settled_candidate_binding is not None
            else ()
        )
        if self.raw_settled_candidate_artifact_hashes != expected_settled_hashes:
            raise EvaluationValidationError(
                "pilot report settled candidate inventory must match its accepted records"
            )
        object.__setattr__(self, "gate_decisions", tuple(self.gate_decisions))
        if len(self.gate_decisions) != 5 or any(
            not isinstance(decision, RolloutGateDecision) for decision in self.gate_decisions
        ):
            raise EvaluationValidationError("pilot report requires all five rollout gate decisions")
        if any(decision.scorecard_id != self.scorecard_id for decision in self.gate_decisions):
            raise EvaluationValidationError("pilot report gate decisions must bind the exact scorecard")
        expected_gate_names = (
            OFFLINE_REPLAY_GATE,
            LIVE_SHADOW_GATE,
            OPT_IN_PAIR_REVIEW_GATE,
            DEFAULT_PAIR_REVIEW_GATE,
            PR_PUBLICATION_GATE,
        )
        if tuple(decision.gate_name for decision in self.gate_decisions) != expected_gate_names:
            raise EvaluationValidationError("pilot report gate decisions must use the canonical order and names")
        if any(decision.arm_id != self.target_arm_id for decision in self.gate_decisions):
            raise EvaluationValidationError("pilot report gate decisions must bind the target arm")
        object.__setattr__(self, "report_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "manifest_artifact_hash": self.manifest_artifact_hash,
            "schema_hash": self.schema_hash,
            "corpus_hash": self.corpus_hash,
            "policy_hash": self.policy_hash,
            "configuration_hash": self.configuration_hash,
            "arm_configuration_hashes": list(self.arm_configuration_hashes),
            "prompt_hashes": list(self.prompt_hashes),
            "model_identity_hashes": list(self.model_identity_hashes),
            "corpus_binding": self.corpus_binding.to_dict() if self.corpus_binding else None,
            "shadow_binding": self.shadow_binding.to_dict() if self.shadow_binding else None,
            "settled_candidate_binding": (
                self.settled_candidate_binding.to_dict()
                if self.settled_candidate_binding
                else None
            ),
            "target_arm_id": self.target_arm_id,
            "incumbent_arm_id": self.incumbent_arm_id,
            "raw_artifact_inventory": {
                "record_artifact_hashes": list(self.raw_record_artifact_hashes),
                "shadow_artifact_hashes": list(self.raw_shadow_artifact_hashes),
                "settled_candidate_artifact_hashes": list(
                    self.raw_settled_candidate_artifact_hashes
                ),
                "terminal_pair_count": self.terminal_pair_count,
                "incomplete_pair_count": self.incomplete_pair_count,
            },
            "scorecard_id": self.scorecard_id,
            "scorecard": _thaw_json(self.scorecard),
            "budgets": self.budgets.to_dict(),
            "evidence": self.evidence.to_dict(),
            "gate_decisions": [decision.to_dict() for decision in self.gate_decisions],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "report_id": self.report_id}


def build_checkpoint_pilot_report(
    manifest: EvaluationManifest,
    truth: TruthArtifact,
    artifact_store: EvaluationArtifactStore,
    *,
    target_arm_id: str,
    incumbent_arm_id: str,
    budgets: PilotRolloutBudgets,
    evidence: PilotRolloutEvidence,
    cocos_inventory: Optional[CocosCorpusInventory] = None,
    cocos_acceptance: Optional[CocosPilotAcceptance] = None,
    shadow_journal_path: Optional[str | Path] = None,
    shadow_acceptance: Optional[ShadowPilotAcceptance] = None,
    settled_candidate_records: Sequence[SettledCandidateRecord] = (),
    settled_candidate_acceptance: Optional[SettledPilotAcceptance] = None,
) -> CheckpointPilotReport:
    """Score immutable attempts and generate the canonical source-free pilot report."""
    if not isinstance(manifest, EvaluationManifest) or not isinstance(truth, TruthArtifact):
        raise EvaluationValidationError("pilot reports require an evaluation manifest and truth artifact")
    if not isinstance(artifact_store, EvaluationArtifactStore):
        raise EvaluationValidationError("pilot reports require an EvaluationArtifactStore")
    target = next((arm for arm in manifest.arms if arm.arm_id == target_arm_id and arm.enabled), None)
    incumbent = next((arm for arm in manifest.arms if arm.arm_id == incumbent_arm_id and arm.enabled), None)
    if target is None or target.kind is not EvaluationArmKind.FULL_CASCADE:
        raise EvaluationValidationError("pilot target must be the enabled full-cascade arm")
    if incumbent is None or incumbent.kind is not EvaluationArmKind.GENERAL_REVIEW:
        raise EvaluationValidationError("pilot incumbent must be the enabled general-review arm")
    records = artifact_store.load_records(manifest)
    shadow_records = load_shadow_journal(shadow_journal_path) if shadow_journal_path is not None else ()
    inventory = artifact_store.inventory(manifest)
    for record in records:
        for name, measurement in (
            ("replay latency_seconds", record.latency_seconds),
            ("replay tokens", record.tokens),
            ("replay cost_usd", record.cost_usd),
        ):
            _validate_measurement_non_negative(name, measurement)
        for stage_name, measurement in record.stage_latencies_seconds.items():
            _validate_measurement_non_negative(
                f"replay stage {stage_name} latency_seconds",
                measurement,
            )
    loaded_record_hashes = tuple(sorted(content_hash(record.to_dict()) for record in records))
    if loaded_record_hashes != inventory.record_artifact_hashes:
        raise EvaluationValidationError("artifact store changed while the pilot report was being generated")
    scorecard = score_matched_arms(manifest, truth, records, baseline_arm_id=incumbent_arm_id)
    expected_replay_binding = build_replay_evidence_binding(
        manifest,
        scorecard,
        inventory,
        target_arm_id=target_arm_id,
        incumbent_arm_id=incumbent_arm_id,
    )
    corpus_binding = _build_cocos_pilot_corpus_binding(
        manifest,
        cocos_inventory,
        cocos_acceptance,
    )
    shadow_binding = _build_shadow_pilot_binding(
        shadow_records,
        shadow_acceptance,
        manifest=manifest,
        target_arm_id=target_arm_id,
    )
    settled_binding = _build_settled_pilot_binding(
        settled_candidate_records,
        settled_candidate_acceptance,
        manifest=manifest,
        target_arm_id=target_arm_id,
    )
    temporal_delta = _temporal_high_severity_recall_delta(
        manifest,
        truth,
        records,
        target_arm_id=target_arm_id,
        incumbent_arm_id=incumbent_arm_id,
    )
    holdout_quality, holdout_cost = _holdout_replay_metrics(
        manifest,
        truth,
        records,
        target_arm_id=target_arm_id,
        incumbent_arm_id=incumbent_arm_id,
    )
    holdout_leakage_free = _accepted_holdout_leakage(manifest, evidence.holdout_leakage_check)
    bound_paid_request = artifact_store.load_paid_request(manifest)
    paid_budget_by_pair = (
        {
            (budget.case_id, budget.arm_id): budget
            for budget in bound_paid_request.plan_item_budgets
        }
        if bound_paid_request is not None
        else {}
    )
    expected_paid_pairs = {
        (case.case_id, arm.arm_id)
        for case in manifest.cases
        for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    }
    bounded_retry_configured = (
        set(paid_budget_by_pair) == expected_paid_pairs
        and all(
            record.attempt <= paid_budget_by_pair[(record.case_id, record.arm_id)].max_attempts
            for record in records
            if (record.case_id, record.arm_id) in expected_paid_pairs
        )
    )
    external_metrics = _external_metrics(
        inventory,
        evidence,
        budgets,
        expected_replay_binding,
        holdout_leakage_free,
        bounded_retry_configured,
        temporal_delta,
        holdout_quality,
        holdout_cost,
        corpus_binding,
        shadow_binding,
        settled_binding,
    )
    decisions = tuple(
        _evaluate_canonical_gate(spec, scorecard, target_arm_id, external_metrics)
        for spec in canonical_rollout_gate_specs(budgets)
    )
    model_identity_hashes = {
        content_hash({
            "arm_kind": arm.kind.value,
            "identities": [
                {"model": model_id, "service": provider_id, "revision": revision}
                for model_id, provider_id, revision in arm.model_identities()
            ],
        })
        for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    }
    public_scorecard = scorecard.to_dict()
    public_scorecard.pop("truth_artifact_id")
    return CheckpointPilotReport(
        manifest_id=manifest.manifest_id,
        manifest_artifact_hash=inventory.manifest_artifact_hash,
        schema_hash=manifest.schema_hash,
        corpus_hash=manifest.corpus_hash,
        policy_hash=manifest.policy_hash,
        configuration_hash=manifest.configuration_hash,
        arm_configuration_hashes=tuple(sorted({arm.configuration_hash for arm in manifest.arms if arm.enabled})),
        prompt_hashes=tuple(sorted({arm.prompt_hash for arm in manifest.arms if arm.enabled})),
        model_identity_hashes=tuple(sorted(model_identity_hashes)),
        corpus_binding=corpus_binding,
        shadow_binding=shadow_binding,
        settled_candidate_binding=settled_binding,
        target_arm_id=target_arm_id,
        incumbent_arm_id=incumbent_arm_id,
        raw_record_artifact_hashes=tuple(sorted(inventory.record_artifact_hashes)),
        raw_shadow_artifact_hashes=(
            shadow_binding.record_artifact_hashes if shadow_binding is not None else ()
        ),
        raw_settled_candidate_artifact_hashes=(
            settled_binding.record_artifact_hashes if settled_binding is not None else ()
        ),
        terminal_pair_count=inventory.terminal_pair_count,
        incomplete_pair_count=inventory.incomplete_pair_count,
        scorecard_id=scorecard.scorecard_id,
        scorecard=public_scorecard,
        budgets=budgets,
        evidence=evidence,
        gate_decisions=decisions,
    )
