"""Pure contracts and planning for stable review-thread lifecycle management.

This module deliberately has no provider calls. It turns verified findings and a
snapshot of existing review threads into an explicit action plan; a later, gated
integration can execute that plan after re-checking the pull-request head.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from pr_agent.algo.inline_comment_dedup import (
    body_with_finding_identity_marker,
    strip_inline_comment_markers,
)

REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION = "review-thread-lifecycle-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalise_text(value: str) -> str:
    return " ".join((value or "").split())


@dataclass(frozen=True)
class FindingIdentity:
    """Stable logical identity supplied by the verified-finding pipeline.

    ``root_cause_id`` is intentionally an upstream contract rather than prose
    derived here. A model may reword a finding without changing its root cause,
    and the reconciler must not guess semantic equivalence from mutable wording.
    """

    repository: str
    pull_request_number: int
    root_cause_id: str
    path: str
    symbol: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION
    finding_id: str = field(init=False)

    def __post_init__(self) -> None:
        repository = (self.repository or "").strip().strip("/").casefold()
        path = (self.path or "").strip().lstrip("/")
        root_cause_id = _normalise_text(self.root_cause_id)
        symbol = _normalise_text(self.symbol) if self.symbol else None
        if not repository or "/" not in repository:
            raise ValueError("repository must be in owner/name form")
        if self.pull_request_number < 1:
            raise ValueError("pull_request_number must be positive")
        if not root_cause_id:
            raise ValueError("root_cause_id is required")
        if not path:
            raise ValueError("path is required")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "root_cause_id", root_cause_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "symbol", symbol)
        identity = {
            "schema_version": self.schema_version,
            "repository": repository,
            "pull_request_number": self.pull_request_number,
            "root_cause_id": root_cause_id,
            "path": path,
            "symbol": symbol,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        object.__setattr__(self, "finding_id", f"sha256:{digest}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewThreadAnchor:
    path: str
    line: int
    start_line: Optional[int] = None
    side: str = "RIGHT"
    start_side: Optional[str] = None

    def __post_init__(self) -> None:
        path = (self.path or "").strip().lstrip("/")
        if not path:
            raise ValueError("path is required")
        if self.line < 1:
            raise ValueError("line must be positive")
        if self.start_line is not None and (self.start_line < 1 or self.start_line > self.line):
            raise ValueError("start_line must be positive and no greater than line")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "side", self.side.upper())
        object.__setattr__(self, "start_side", self.start_side.upper() if self.start_side else None)

    def to_github_comment(self, body: str) -> dict[str, Any]:
        comment = {"body": body, "path": self.path, "line": self.line, "side": self.side}
        if self.start_line is not None and self.start_line != self.line:
            comment.pop("side")
            comment["start_line"] = self.start_line
            comment["start_side"] = self.start_side or self.side
        return comment


@dataclass(frozen=True)
class ReviewThreadCommentSnapshot:
    node_id: str
    database_id: Optional[int]
    author_login: Optional[str]
    body: str
    created_at: Optional[str] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class ReviewThreadSnapshot:
    thread_id: str
    finding_id: Optional[str]
    anchor: Optional[ReviewThreadAnchor]
    is_resolved: bool
    is_outdated: bool
    bot_owned: bool
    has_replies: bool
    reviewed_head_sha: Optional[str]
    comments: tuple[ReviewThreadCommentSnapshot, ...]
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def root_comment(self) -> Optional[ReviewThreadCommentSnapshot]:
        return self.comments[0] if self.comments else None


@dataclass(frozen=True)
class DesiredReviewThread:
    identity: FindingIdentity
    anchor: ReviewThreadAnchor
    body: str

    @property
    def marked_body(self) -> str:
        return body_with_finding_identity_marker(self.body, self.identity.finding_id)


class ReviewThreadActionKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    RESOLVE = "resolve"
    UNCHANGED = "unchanged"
    SKIP = "skip"


@dataclass(frozen=True)
class ReviewThreadAction:
    action_id: str
    kind: ReviewThreadActionKind
    finding_id: str
    expected_head_sha: str
    reason: str
    thread_id: Optional[str] = None
    root_comment_id: Optional[int] = None
    anchor: Optional[ReviewThreadAnchor] = None
    body: Optional[str] = None
    depends_on_action_id: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION


@dataclass(frozen=True)
class ReviewThreadActionPlan:
    expected_head_sha: str
    actions: tuple[ReviewThreadAction, ...]
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.expected_head_sha:
            raise ValueError("expected_head_sha is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for action in data["actions"]:
            action["kind"] = action["kind"].value
        return data


class ReviewThreadActionState(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    STALE_HEAD = "stale_head"
    FAILED = "failed"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True)
class ReviewThreadActionOutcome:
    kind: ReviewThreadActionKind
    state: ReviewThreadActionState
    expected_head_sha: str
    current_head_sha: Optional[str]
    thread_id: Optional[str] = None
    comment_id: Optional[int] = None
    comment_node_id: Optional[str] = None
    reason: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        return self.state in {ReviewThreadActionState.APPLIED, ReviewThreadActionState.ALREADY_APPLIED}


@dataclass(frozen=True)
class ReviewThreadReconciliationOutcome:
    expected_head_sha: str
    current_head_sha: Optional[str]
    action_outcomes: tuple[ReviewThreadActionOutcome, ...]
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in ReviewThreadActionState}
        for outcome in self.action_outcomes:
            counts[outcome.state.value] += 1
        return counts

    @property
    def complete(self) -> bool:
        return all(outcome.succeeded for outcome in self.action_outcomes)


def _action_id(kind: ReviewThreadActionKind, finding_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{kind.value}|{finding_id}|{ordinal}".encode("utf-8")).hexdigest()[:16]
    return f"thread-action:{digest}"


def plan_review_thread_actions(
    desired_threads: tuple[DesiredReviewThread, ...],
    existing_threads: tuple[ReviewThreadSnapshot, ...],
    expected_head_sha: str,
    *,
    obsolete_policy: str = "keep",
) -> ReviewThreadActionPlan:
    """Plan lifecycle actions without calling a provider.

    Existing resolved threads and any thread with replies are never mutated. For
    moved findings, creation is planned before resolution; the resolve action is
    explicitly dependent on successful creation so execution cannot silently
    discard the finding.
    """
    if obsolete_policy not in {"keep", "resolve"}:
        raise ValueError("obsolete_policy must be 'keep' or 'resolve'")
    desired_by_id = {desired.identity.finding_id: desired for desired in desired_threads}
    if len(desired_by_id) != len(desired_threads):
        raise ValueError("desired findings must have unique identities")
    existing_by_id: dict[str, list[ReviewThreadSnapshot]] = {}
    for thread in existing_threads:
        if thread.finding_id:
            existing_by_id.setdefault(thread.finding_id, []).append(thread)

    actions: list[ReviewThreadAction] = []

    def add(kind: ReviewThreadActionKind, finding_id: str, reason: str, **kwargs) -> ReviewThreadAction:
        action = ReviewThreadAction(
            action_id=_action_id(kind, finding_id, len(actions)),
            kind=kind,
            finding_id=finding_id,
            expected_head_sha=expected_head_sha,
            reason=reason,
            **kwargs,
        )
        actions.append(action)
        return action

    for finding_id, desired in desired_by_id.items():
        matches = existing_by_id.get(finding_id, [])
        if not matches:
            add(ReviewThreadActionKind.CREATE, finding_id, "new_finding", anchor=desired.anchor,
                body=desired.marked_body)
            continue
        active_matches = [thread for thread in matches if not thread.is_resolved]
        if not active_matches:
            add(ReviewThreadActionKind.SKIP, finding_id, "existing_thread_resolved", thread_id=matches[-1].thread_id)
            continue
        if len(active_matches) > 1:
            for duplicate in active_matches:
                add(ReviewThreadActionKind.SKIP, finding_id, "duplicate_existing_identity_requires_manual_audit",
                    thread_id=duplicate.thread_id)
            continue
        current = active_matches[0]
        if not current.bot_owned or current.has_replies:
            add(ReviewThreadActionKind.SKIP, finding_id, "thread_not_safe_to_mutate", thread_id=current.thread_id)
            continue
        root = current.root_comment
        if current.anchor == desired.anchor:
            old_body = strip_inline_comment_markers(root.body if root else "").strip()
            new_body = strip_inline_comment_markers(desired.body).strip()
            if old_body == new_body:
                add(ReviewThreadActionKind.UNCHANGED, finding_id, "same_anchor_and_body",
                    thread_id=current.thread_id, root_comment_id=root.database_id if root else None)
            else:
                add(ReviewThreadActionKind.UPDATE, finding_id, "same_anchor_changed_body",
                    thread_id=current.thread_id, root_comment_id=root.database_id if root else None,
                    body=desired.marked_body)
        else:
            create = add(ReviewThreadActionKind.CREATE, finding_id, "finding_moved", anchor=desired.anchor,
                         body=desired.marked_body)
            add(ReviewThreadActionKind.RESOLVE, finding_id, "superseded_by_moved_finding",
                thread_id=current.thread_id, root_comment_id=root.database_id if root else None,
                depends_on_action_id=create.action_id)
    for finding_id, matches in existing_by_id.items():
        if finding_id in desired_by_id:
            continue
        for thread in matches:
            root = thread.root_comment
            if (obsolete_policy == "resolve" and thread.bot_owned and not thread.has_replies and
                    not thread.is_resolved):
                add(ReviewThreadActionKind.RESOLVE, finding_id, "finding_no_longer_present",
                    thread_id=thread.thread_id, root_comment_id=root.database_id if root else None)
            else:
                add(ReviewThreadActionKind.SKIP, finding_id, "obsolete_thread_preserved",
                    thread_id=thread.thread_id)

    return ReviewThreadActionPlan(expected_head_sha=expected_head_sha, actions=tuple(actions))
