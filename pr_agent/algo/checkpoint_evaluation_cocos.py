"""Read-only adapter for the external Cocos Story review corpus.

The corpus stays in its owning repository.  PR-Agent receives only a lock with
approved hashes and a source path, then emits a source-free inventory suitable
for an evaluation manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from pr_agent.algo.checkpoint_evaluation import EvaluationValidationError, content_hash
from pr_agent.algo.local_artifact_io import read_regular_file_without_symlinks
from pr_agent.algo.review_snapshot import ReviewEvent

COCOS_ADAPTER_SCHEMA_VERSION = "cocos-story-checkpoint-corpus-v2"
CHECKPOINT_CONTROL_SCHEMA_VERSION = "cocos-story-checkpoint-controls-v1"
COCOS_REPOSITORY = "sagacious-heritage/cocos-story"
_DEFAULT_COHORT_COUNTS = {
    "calibration": 12,
    "holdout": 18,
    "temporal": 10,
    "control": 16,
    "confirmation": 16,
    "unique_snapshots": 55,
}
_REQUIRED_FILES = {
    "primary": "ledger.json",
    "temporal": "temporal-backtest-ledger.json",
    "controls": "controls-ledger.json",
    "annotations": "specialist-annotations.json",
    "confirmation": "confirmation-ledger.json",
    "confirmation_annotations": "confirmation/specialist-annotations.json",
}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_CHECKPOINT_SCENARIOS = {
    "partial_correct_save",
    "coherent_clean_worktree",
    "temporary_mistake_corrected",
    "staged_checkpoint",
    "stale_candidate_withdrawn",
}
_FORBIDDEN_CONTROL_KEYS = {
    "source", "source_text", "diff", "patch", "prompt", "secret", "credential",
    "hidden_reasoning", "reasoning", "task_intent", "provider_request_id",
}
_CHECKPOINT_CONTROL_FIELDS = {
    "id", "snapshot_id", "snapshot_artifact_hash", "parent_id", "stage", "scenario",
    "independently_adjudicated", "adjudication_hash", "is_clean",
    "expected_withdrawn_fingerprints",
}
_CHECKPOINT_CONTROL_TOP_LEVEL_FIELDS = {"schema_version", "answer_only", "entries"}
_CHECKPOINT_STAGES = {event.value for event in ReviewEvent}
_MAX_COCOS_ARTIFACT_BYTES = 25_000_000


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _read_regular_file(root: Path, relative_path: str) -> bytes:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationValidationError("Cocos corpus lock paths must stay below the corpus root")
    return read_regular_file_without_symlinks(
        root / relative,
        label=f"Cocos corpus artifact {relative_path}",
        max_bytes=_MAX_COCOS_ARTIFACT_BYTES,
    )


@dataclass(frozen=True)
class CocosCorpusLock:
    source_revision: str
    artifact_hashes: Mapping[str, str]
    assignment_hash: str
    confirmation_assignment_hash: str
    expected_cohort_counts: Mapping[str, int] = field(default_factory=lambda: dict(_DEFAULT_COHORT_COUNTS))
    repository: str = COCOS_REPOSITORY
    schema_version: str = COCOS_ADAPTER_SCHEMA_VERSION
    lock_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != COCOS_ADAPTER_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported Cocos adapter schema_version: {self.schema_version}")
        if self.repository != COCOS_REPOSITORY:
            raise EvaluationValidationError("the Cocos adapter may only read sagacious-heritage/cocos-story")
        if not isinstance(self.source_revision, str) or not _GIT_COMMIT.fullmatch(self.source_revision):
            raise EvaluationValidationError("Cocos source_revision must be a full immutable Git commit")
        if set(self.artifact_hashes) != set(_REQUIRED_FILES.values()):
            raise EvaluationValidationError("Cocos artifact hashes must name every required external ledger")
        for name, digest in self.artifact_hashes.items():
            if not isinstance(name, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise EvaluationValidationError("Cocos artifact hashes must be sha256 identities")
        required_counts = {
            "calibration", "holdout", "temporal", "control", "confirmation", "unique_snapshots",
        }
        if set(self.expected_cohort_counts) != required_counts or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in self.expected_cohort_counts.values()
        ):
            raise EvaluationValidationError("Cocos cohort lock must contain positive exact counts")
        if not isinstance(self.assignment_hash, str) or not _SHA256.fullmatch(self.assignment_hash):
            raise EvaluationValidationError("Cocos assignment_hash must be a sha256 identity")
        if (
            not isinstance(self.confirmation_assignment_hash, str)
            or not _SHA256.fullmatch(self.confirmation_assignment_hash)
        ):
            raise EvaluationValidationError(
                "Cocos confirmation_assignment_hash must be a sha256 identity"
            )
        object.__setattr__(self, "artifact_hashes", MappingProxyType(dict(self.artifact_hashes)))
        object.__setattr__(self, "expected_cohort_counts", MappingProxyType(dict(self.expected_cohort_counts)))
        object.__setattr__(self, "lock_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "source_revision": self.source_revision,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "assignment_hash": self.assignment_hash,
            "confirmation_assignment_hash": self.confirmation_assignment_hash,
            "expected_cohort_counts": dict(sorted(self.expected_cohort_counts.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "lock_id": self.lock_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CocosCorpusLock":
        allowed = {
            "schema_version", "repository", "source_revision", "artifact_hashes", "assignment_hash",
            "confirmation_assignment_hash", "expected_cohort_counts", "lock_id",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise EvaluationValidationError(f"Cocos corpus lock contains unknown fields: {unknown}")
        lock = cls(
            source_revision=value["source_revision"],
            artifact_hashes=value["artifact_hashes"],
            assignment_hash=value["assignment_hash"],
            confirmation_assignment_hash=value["confirmation_assignment_hash"],
            expected_cohort_counts=value.get("expected_cohort_counts", _DEFAULT_COHORT_COUNTS),
            repository=value.get("repository", COCOS_REPOSITORY),
            schema_version=value.get("schema_version", COCOS_ADAPTER_SCHEMA_VERSION),
        )
        if value.get("lock_id") not in (None, lock.lock_id):
            raise EvaluationValidationError("Cocos lock_id does not match lock content")
        return lock


@dataclass(frozen=True)
class CocosCorpusInventory:
    lock_id: str
    source_revision: str
    corpus_hash: str
    cohort_counts: Mapping[str, int]
    checkpoint_control_count: Optional[int]
    checkpoint_controls_status: str
    checkpoint_controls_hash: Optional[str]
    root_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_schema_version": COCOS_ADAPTER_SCHEMA_VERSION,
            "lock_id": self.lock_id,
            "source_revision": self.source_revision,
            "corpus_hash": self.corpus_hash,
            "cohort_counts": dict(sorted(self.cohort_counts.items())),
            "checkpoint_control_count": self.checkpoint_control_count,
            "checkpoint_controls_status": self.checkpoint_controls_status,
            "checkpoint_controls_hash": self.checkpoint_controls_hash,
            "root_identity": self.root_identity,
        }


def _load_json(value: bytes, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"Cocos corpus artifact is not valid JSON: {name}") from exc
    if not isinstance(parsed, Mapping):
        raise EvaluationValidationError(f"Cocos corpus artifact must be an object: {name}")
    return parsed


def _object_entries(payload: Mapping[str, Any], key: str, name: str) -> list[Mapping[str, Any]]:
    entries = payload.get(key)
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise EvaluationValidationError(f"Cocos {name} must contain an object list named {key}")
    return entries


def _assignment_payload(
    primary: Mapping[str, Any], temporal: Mapping[str, Any], controls: Mapping[str, Any]
) -> list[dict]:
    assignments: list[dict] = []
    for entry in _object_entries(primary, "entries", "primary ledger"):
        assignments.append({
            "id": entry.get("id"),
            "split": entry.get("split"),
            "target_sha": entry.get("target_sha"),
        })
    for entry in _object_entries(temporal, "entries", "temporal ledger"):
        assignments.append({
            "id": entry.get("id"),
            "split": "temporal",
            "target_sha": entry.get("target_sha"),
        })
    for entry in _object_entries(controls, "entries", "control ledger"):
        assignments.append({
            "id": entry.get("id"),
            "split": "control",
            "target_sha": entry.get("target_sha"),
        })
    return sorted(assignments, key=lambda item: (str(item["split"]), str(item["id"])))


def _confirmation_assignment_payload(
    confirmation: Mapping[str, Any],
    annotations: Mapping[str, Any],
) -> list[dict[str, str]]:
    selection_policy = confirmation.get("selection_policy")
    if (
        confirmation.get("schema_version") != 1
        or confirmation.get("cohort") != "sealed_confirmation"
        or not isinstance(selection_policy, Mapping)
        or selection_policy.get("prompt_development_allowed") is not False
        or selection_policy.get("architecture_selection_allowed") is not False
    ):
        raise EvaluationValidationError("Cocos confirmation ledger must remain sealed")
    entries = _object_entries(confirmation, "entries", "confirmation ledger")
    snapshots = _object_entries(annotations, "snapshots", "confirmation annotation ledger")
    annotation_policy = annotations.get("annotation_policy")
    if (
        annotations.get("schema_version") != 1
        or annotations.get("answer_only") is not True
        or not isinstance(annotation_policy, Mapping)
        or annotation_policy.get("defect_targets_are_never_prompt_inputs") is not True
    ):
        raise EvaluationValidationError("Cocos confirmation annotations must remain answer-only")

    ledger_assignments: dict[str, str] = {}
    for entry in entries:
        identifier = entry.get("id")
        target_sha = entry.get("target_sha")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in ledger_assignments
            or entry.get("split") != "confirmation"
            or not isinstance(target_sha, str)
            or not _GIT_COMMIT.fullmatch(target_sha)
        ):
            raise EvaluationValidationError(
                "Cocos confirmation entries require unique ids, confirmation split, and immutable target SHAs"
            )
        ledger_assignments[identifier] = target_sha

    annotation_assignments: dict[str, str] = {}
    for snapshot in snapshots:
        identifier = snapshot.get("snapshot_id")
        target_sha = snapshot.get("target_sha")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in annotation_assignments
            or snapshot.get("split") != "confirmation"
            or not isinstance(target_sha, str)
            or not _GIT_COMMIT.fullmatch(target_sha)
        ):
            raise EvaluationValidationError(
                "Cocos confirmation annotations require unique ids, confirmation split, and immutable target SHAs"
            )
        annotation_assignments[identifier] = target_sha
    if ledger_assignments != annotation_assignments:
        raise EvaluationValidationError(
            "Cocos sealed confirmation ledger and annotations name different assignments"
        )
    return [
        {"id": identifier, "split": "confirmation", "target_sha": ledger_assignments[identifier]}
        for identifier in sorted(ledger_assignments)
    ]


def _contains_forbidden_control_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _FORBIDDEN_CONTROL_KEYS or _contains_forbidden_control_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_control_key(item) for item in value)
    return False


def _validate_checkpoint_controls(path: Optional[str | Path]) -> tuple[Optional[int], str, Optional[str]]:
    if path is None:
        return None, "not_evaluable", None
    raw = _read_regular_file(Path(path).parent, Path(path).name)
    payload = _load_json(raw, "checkpoint controls")
    unknown_top_level = sorted(set(payload) - _CHECKPOINT_CONTROL_TOP_LEVEL_FIELDS)
    if unknown_top_level:
        raise EvaluationValidationError(
            f"checkpoint control artifact contains unknown fields: {unknown_top_level}"
        )
    if payload.get("schema_version") != CHECKPOINT_CONTROL_SCHEMA_VERSION:
        raise EvaluationValidationError(
            "unsupported checkpoint control schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("answer_only") is not True or _contains_forbidden_control_key(payload):
        raise EvaluationValidationError("checkpoint controls must be answer-only and source-free")
    entries = _object_entries(payload, "entries", "checkpoint control artifact")
    if not 15 <= len(entries) <= 20:
        raise EvaluationValidationError("checkpoint controls must contain 15 to 20 independently adjudicated cases")
    ids: set[str] = set()
    snapshot_ids: set[str] = set()
    scenarios: set[str] = set()
    parents: dict[str, Optional[str]] = {}
    for entry in entries:
        unknown = sorted(set(entry) - _CHECKPOINT_CONTROL_FIELDS)
        if unknown:
            raise EvaluationValidationError(f"checkpoint control contains unknown fields: {unknown}")
        if entry.get("stage") == "final_pr_head":
            raise EvaluationValidationError("final clean PR heads cannot stand in for checkpoint controls")
        if entry.get("independently_adjudicated") is not True:
            raise EvaluationValidationError("every checkpoint control requires independent adjudication")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise EvaluationValidationError("checkpoint control ids must be unique non-empty strings")
        ids.add(identifier)
        stage = entry.get("stage")
        if not isinstance(stage, str) or stage not in _CHECKPOINT_STAGES:
            raise EvaluationValidationError("checkpoint control stage must be a shipped review event")
        parent_id = entry.get("parent_id")
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id.strip() or parent_id == identifier
        ):
            raise EvaluationValidationError("checkpoint control parent_id must name a different checkpoint")
        parents[identifier] = parent_id
        snapshot_id = entry.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not _SHA256.fullmatch(snapshot_id) or snapshot_id in snapshot_ids:
            raise EvaluationValidationError("checkpoint controls require unique immutable snapshot ids")
        snapshot_ids.add(snapshot_id)
        for field_name in ("snapshot_artifact_hash", "adjudication_hash"):
            value = entry.get(field_name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise EvaluationValidationError(f"checkpoint control {field_name} must be a sha256 identity")
        if entry.get("is_clean") is not True:
            raise EvaluationValidationError("checkpoint controls must be independently adjudicated as clean")
        withdrawn = entry.get("expected_withdrawn_fingerprints", [])
        if not isinstance(withdrawn, list) or any(
            not isinstance(fingerprint, str) or not fingerprint.strip()
            for fingerprint in withdrawn
        ):
            raise EvaluationValidationError("checkpoint withdrawn fingerprints must be a string list")
        scenario = entry.get("scenario")
        if not isinstance(scenario, str) or scenario not in _REQUIRED_CHECKPOINT_SCENARIOS:
            raise EvaluationValidationError("checkpoint controls contain an unknown or missing scenario")
        scenarios.add(scenario)
        if scenario in {"temporary_mistake_corrected", "stale_candidate_withdrawn"} and parent_id is None:
            raise EvaluationValidationError(f"checkpoint scenario {scenario} requires lineage parent_id")
        if scenario == "stale_candidate_withdrawn" and not withdrawn:
            raise EvaluationValidationError("stale-candidate controls require a withdrawn finding fingerprint")
    missing_scenarios = sorted(_REQUIRED_CHECKPOINT_SCENARIOS - scenarios)
    if missing_scenarios:
        raise EvaluationValidationError(
            f"checkpoint controls do not cover required intermediate scenarios: {missing_scenarios}"
        )
    for identifier, parent_id in parents.items():
        if parent_id is not None and parent_id not in ids:
            raise EvaluationValidationError(f"checkpoint {identifier} names an unknown parent_id")
        visited: set[str] = set()
        current = identifier
        while True:
            next_parent = parents[current]
            if next_parent is None:
                break
            if current in visited:
                raise EvaluationValidationError("checkpoint control parent relationships contain a cycle")
            visited.add(current)
            current = next_parent
    return len(entries), "complete", _sha256_bytes(raw)


def validate_cocos_story_corpus(
    corpus_root: str | Path,
    lock: CocosCorpusLock,
    *,
    checkpoint_controls_path: Optional[str | Path] = None,
) -> CocosCorpusInventory:
    root = Path(corpus_root)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise EvaluationValidationError("Cocos corpus root does not exist") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise EvaluationValidationError("Cocos corpus root must be a real directory")

    parsed: dict[str, Mapping[str, Any]] = {}
    actual_hashes: dict[str, str] = {}
    for role, name in _REQUIRED_FILES.items():
        raw = _read_regular_file(root, name)
        actual_hashes[name] = _sha256_bytes(raw)
        if actual_hashes[name] != lock.artifact_hashes[name]:
            raise EvaluationValidationError(f"Cocos corpus artifact changed after the approved lock: {name}")
        parsed[role] = _load_json(raw, name)
        if parsed[role].get("repo") != COCOS_REPOSITORY:
            raise EvaluationValidationError(f"Cocos corpus artifact names the wrong repository: {name}")

    primary_entries = _object_entries(parsed["primary"], "entries", "primary ledger")
    temporal_entries = _object_entries(parsed["temporal"], "entries", "temporal ledger")
    control_entries = _object_entries(parsed["controls"], "entries", "control ledger")
    annotation_snapshots = _object_entries(parsed["annotations"], "snapshots", "annotation ledger")
    confirmation_assignments = _confirmation_assignment_payload(
        parsed["confirmation"],
        parsed["confirmation_annotations"],
    )
    snapshot_ids = [entry.get("snapshot_id") for entry in annotation_snapshots]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in snapshot_ids):
        raise EvaluationValidationError("Cocos annotation snapshot ids must be non-empty strings")
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise EvaluationValidationError("Cocos annotation snapshot ids must be unique")
    counts = {
        "calibration": sum(entry.get("split") == "calibration" for entry in primary_entries),
        "holdout": sum(entry.get("split") == "holdout" for entry in primary_entries),
        "temporal": len(temporal_entries),
        "control": len(control_entries),
        "confirmation": len(confirmation_assignments),
        "unique_snapshots": len(annotation_snapshots),
    }
    if counts != dict(lock.expected_cohort_counts):
        raise EvaluationValidationError(f"Cocos frozen cohort counts changed: {counts}")
    if parsed["annotations"].get("answer_only") is not True:
        raise EvaluationValidationError("Cocos specialist annotations must remain answer-only")
    assignments = _assignment_payload(parsed["primary"], parsed["temporal"], parsed["controls"])
    if content_hash(assignments) != lock.assignment_hash:
        raise EvaluationValidationError("Cocos calibration, holdout, temporal, or control assignments changed")
    if content_hash(confirmation_assignments) != lock.confirmation_assignment_hash:
        raise EvaluationValidationError("Cocos sealed confirmation assignments changed")
    confirmation_source_hashes = parsed["confirmation_annotations"].get("source_hashes")
    if not isinstance(confirmation_source_hashes, Mapping) or (
        confirmation_source_hashes.get("confirmation-ledger.json")
        != actual_hashes["confirmation-ledger.json"].removeprefix("sha256:")
    ):
        raise EvaluationValidationError(
            "Cocos confirmation annotations are not bound to the locked confirmation ledger"
        )

    checkpoint_count, checkpoint_status, checkpoint_hash = _validate_checkpoint_controls(checkpoint_controls_path)
    corpus_hash = content_hash({
        "source_revision": lock.source_revision,
        "artifact_hashes": actual_hashes,
        "assignment_hash": lock.assignment_hash,
        "confirmation_assignment_hash": lock.confirmation_assignment_hash,
        "checkpoint_control_count": checkpoint_count,
        "checkpoint_controls_hash": checkpoint_hash,
    })
    root_identity = content_hash({"real_path": os.path.realpath(root)})
    return CocosCorpusInventory(
        lock_id=lock.lock_id,
        source_revision=lock.source_revision,
        corpus_hash=corpus_hash,
        cohort_counts=MappingProxyType(counts),
        checkpoint_control_count=checkpoint_count,
        checkpoint_controls_status=checkpoint_status,
        checkpoint_controls_hash=checkpoint_hash,
        root_identity=root_identity,
    )
