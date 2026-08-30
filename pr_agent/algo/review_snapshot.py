"""Immutable contracts for local, pre-commit review snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

SNAPSHOT_SCHEMA_VERSION = "review-snapshot-v1"


class ReviewEvent(str, Enum):
    FILE_SAVE = "file_save"
    WORKTREE_IDLE = "worktree_idle"
    PRE_COMMIT = "pre_commit"

    @classmethod
    def parse(cls, value: str) -> "ReviewEvent":
        return cls(value.strip().lower().replace("-", "_"))


class ReviewResultState(str, Enum):
    FINDINGS = "findings"
    NO_FINDINGS = "no_findings"
    COVERAGE_UNAVAILABLE = "coverage_unavailable"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass(frozen=True)
class CoverageIssue:
    """One path or stage that could not be included in the snapshot."""

    reason: str
    path: Optional[str] = None
    fingerprint: Optional[str] = field(default=None, compare=False)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _repository_identity(repository_root: str) -> str:
    """Hash the local root so ids and caches cannot cross repositories.

    The root itself remains in the local envelope, while the identity input uses a
    one-way digest so snapshot ids do not disclose an absolute path when copied.
    """

    digest = hashlib.sha256(repository_root.encode("utf-8", errors="surrogateescape")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class ReviewSnapshot:
    """Exact local input reviewed by PR-Agent.

    ``created_at`` and lineage are intentionally excluded from ``snapshot_id``.
    Re-capturing the same repository content under the same policy must produce the
    same id, while any review-relevant content or policy change must produce a new one.
    """

    event: ReviewEvent
    repository_root: str
    base_revision: str
    changed_paths: tuple[str, ...]
    diff: str
    policy_version: str
    created_at: str
    base_selector: Optional[str] = None
    focus_path: Optional[str] = None
    task_intent: Optional[str] = None
    deterministic_results: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    review_configuration_hash: Optional[str] = None
    parent_snapshot_id: Optional[str] = None
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    coverage_issues: tuple[CoverageIssue, ...] = field(default_factory=tuple)
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_selector", self.base_selector or self.base_revision)
        object.__setattr__(self, "changed_paths", tuple(sorted(set(self.changed_paths))))
        object.__setattr__(self, "deterministic_results", tuple(self.deterministic_results))
        object.__setattr__(self, "coverage_issues", tuple(self.coverage_issues))
        identity = {
            "schema_version": self.schema_version,
            "repository": _repository_identity(self.repository_root),
            "event": self.event.value,
            "base_selector": self.base_selector,
            "base_revision": self.base_revision,
            "changed_paths": self.changed_paths,
            "focus_path": self.focus_path,
            "diff": self.diff,
            "task_intent": self.task_intent,
            "deterministic_results": self.deterministic_results,
            "review_configuration_hash": self.review_configuration_hash,
            "coverage_issues": [asdict(issue) for issue in self.coverage_issues],
            "policy_version": self.policy_version,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        object.__setattr__(self, "snapshot_id", f"sha256:{digest}")

    @property
    def has_partial_coverage(self) -> bool:
        return bool(self.coverage_issues)

    def to_dict(self, include_diff: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["event"] = self.event.value
        data["snapshot_id"] = self.snapshot_id
        if not include_diff:
            data.pop("diff", None)
        return data


@dataclass(frozen=True)
class ReviewSnapshotResult:
    """Structured output bound to one immutable snapshot."""

    snapshot_id: str
    state: ReviewResultState
    current_snapshot_id: Optional[str]
    review: Optional[Mapping[str, Any]]
    coverage_issues: tuple[CoverageIssue, ...]
    latency_seconds: float
    usage: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    cached: bool = False
    advisory: bool = True
    shadow_capable: bool = True
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "coverage_issues", tuple(self.coverage_issues))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


def finding_count(structured_review: Optional[Mapping[str, Any]]) -> int:
    if not structured_review:
        return 0
    review = structured_review.get("review")
    if not isinstance(review, Mapping):
        return 0
    findings = review.get("key_issues_to_review", [])
    return len(findings) if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)) else 0
