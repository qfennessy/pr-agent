"""Capture and validate immutable local review snapshots."""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping, Optional, Sequence

from pr_agent.algo.review_snapshot import (CoverageIssue, ReviewEvent,
                                           ReviewResultState, ReviewSnapshot,
                                           ReviewSnapshotResult, finding_count)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.plain_diff_provider import parse_plain_diff


class SnapshotCaptureError(ValueError):
    pass


def _run_git(repository_root: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode not in allowed_returncodes:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise SnapshotCaptureError(message or f"git {' '.join(args)} failed with exit code {process.returncode}")
    return process.stdout


def find_repository_root(start: Optional[str] = None) -> Path:
    start_path = Path(start or os.getcwd()).resolve()
    process = subprocess.run(
        ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise SnapshotCaptureError("review-snapshot must run inside a Git worktree")
    return Path(process.stdout.decode("utf-8").strip()).resolve()


def _decode_z_paths(output: bytes) -> list[str]:
    return [part.decode("utf-8", errors="surrogateescape") for part in output.split(b"\0") if part]


class LocalPairReview:
    """Build exact local diffs without requiring a clean worktree or hosted PR."""

    def __init__(
        self,
        repository_root: Optional[str] = None,
        *,
        excluded_paths: Optional[Sequence[str]] = None,
        ignored_paths: Optional[Sequence[str]] = None,
        max_file_bytes: Optional[int] = None,
    ) -> None:
        self.repository_root = find_repository_root(repository_root)
        settings = get_settings().get("local_pair_review", {}) or {}
        configured_exclusions = settings.get("excluded_paths", []) if hasattr(settings, "get") else []
        self.excluded_paths = tuple(excluded_paths if excluded_paths is not None else configured_exclusions)
        self.ignored_paths = tuple(ignored_paths or ())
        configured_limit = settings.get("max_file_bytes", 1_000_000) if hasattr(settings, "get") else 1_000_000
        self.max_file_bytes = int(max_file_bytes if max_file_bytes is not None else configured_limit)

    def _relative_path(self, supplied_path: str) -> str:
        path = Path(supplied_path)
        candidate = path if path.is_absolute() else self.repository_root / path
        # Resolve existing symlinks as well as lexical ``..``. A symlink that points
        # outside the worktree is not safe input even when its link lives inside it.
        lexical = Path(os.path.abspath(candidate))
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.repository_root)
            relative = lexical.relative_to(self.repository_root)
        except ValueError as exc:
            raise SnapshotCaptureError(f"path is outside repository root: {supplied_path}") from exc
        if relative == Path("."):
            raise SnapshotCaptureError("a repository root is not a reviewable file path")
        return relative.as_posix()

    def _is_excluded(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.excluded_paths)

    def _is_ignored(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.ignored_paths)

    def _inspect_content(self, content: bytes) -> Optional[str]:
        if len(content) > self.max_file_bytes:
            return "file_too_large"
        if b"\0" in content:
            return "binary"
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return "binary"
        return None

    def _git_object_mode(self, revision: str, path: str) -> Optional[str]:
        if revision == ":":
            output = _run_git(
                self.repository_root, "--literal-pathspecs", "ls-files", "--stage", "--", path
            )
        else:
            output = _run_git(
                self.repository_root, "--literal-pathspecs", "ls-tree", revision, "--", path
            )
        line = output.decode("utf-8", errors="replace").strip()
        return line.split(maxsplit=1)[0] if line else None

    def _inspect_revision_file(self, revision: str, path: str) -> Optional[str]:
        mode = self._git_object_mode(revision, path)
        if mode is None:
            return None
        if mode == "120000":
            return "symlink"
        process = subprocess.run(
            ["git", "-C", str(self.repository_root), "show", f"{revision}:./{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            return "unreadable"
        return self._inspect_content(process.stdout)

    def _inspect_current_file(self, path: str, base_revision: str) -> Optional[str]:
        base_issue = self._inspect_revision_file(base_revision, path)
        if base_issue:
            return base_issue
        candidate = self.repository_root / path
        if candidate.is_symlink():
            return "symlink"
        if not candidate.exists():
            return self._inspect_revision_file(base_revision, path)
        if not candidate.is_file():
            return "not_a_regular_file"
        try:
            content = candidate.read_bytes()
        except OSError:
            return "unreadable"
        return self._inspect_content(content)

    def _inspect_index_file(self, path: str, base_revision: str) -> Optional[str]:
        """Inspect the exact staged blob used by a pre-commit snapshot."""
        base_issue = self._inspect_revision_file(base_revision, path)
        if base_issue:
            return base_issue
        if self._git_object_mode(":", path) == "120000":
            return "symlink"
        process = subprocess.run(
            ["git", "-C", str(self.repository_root), "show", f":./{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            return self._inspect_revision_file(base_revision, path)
        return self._inspect_content(process.stdout)

    def _resolve_base(self, base: str) -> str:
        resolved = _run_git(self.repository_root, "rev-parse", "--verify", f"{base}^{{commit}}")
        return resolved.decode("ascii").strip()

    def _tracked_path_groups(self, event: ReviewEvent, base_revision: str) -> list[tuple[str, ...]]:
        args = ["diff"]
        if event is ReviewEvent.PRE_COMMIT:
            args.append("--cached")
        args.extend(["--name-status", "-z", "--find-renames", base_revision, "--"])
        fields = _decode_z_paths(_run_git(self.repository_root, *args))
        path_groups = []
        index = 0
        while index < len(fields):
            status = fields[index]
            index += 1
            path_count = 2 if status.startswith(("R", "C")) else 1
            path_groups.append(tuple(fields[index:index + path_count]))
            index += path_count
        return path_groups

    def _untracked_paths(self) -> list[str]:
        return _decode_z_paths(
            _run_git(self.repository_root, "ls-files", "--others", "--exclude-standard", "-z", "--")
        )

    def _capture_diff(self, event: ReviewEvent, base_revision: str, paths: Sequence[str]) -> str:
        if not paths:
            return ""
        args = ["--literal-pathspecs", "diff", "--no-ext-diff", "--find-renames"]
        if event is ReviewEvent.PRE_COMMIT:
            args.append("--cached")
        args.extend([base_revision, "--", *paths])
        return _run_git(self.repository_root, *args).decode("utf-8", errors="strict")

    def _capture_untracked_addition(self, path: str) -> str:
        output = _run_git(
            self.repository_root,
            "diff",
            "--no-index",
            "--no-ext-diff",
            "--",
            "/dev/null",
            path,
            allowed_returncodes=(0, 1),
        )
        return output.decode("utf-8", errors="strict")

    def capture(
        self,
        *,
        event: ReviewEvent | str,
        base: str = "HEAD",
        focus_path: Optional[str] = None,
        task_intent: Optional[str] = None,
        deterministic_results: Iterable[Mapping[str, Any]] = (),
        review_configuration_hash: Optional[str] = None,
        policy_version: str = "local-pair-review-v1",
        parent_snapshot_id: Optional[str] = None,
    ) -> ReviewSnapshot:
        event = ReviewEvent.parse(event) if isinstance(event, str) else event
        if event is ReviewEvent.FILE_SAVE and not focus_path:
            raise SnapshotCaptureError("file-save snapshots require --path")

        base_revision = self._resolve_base(base)
        normalized_focus = self._relative_path(focus_path) if focus_path else None
        tracked_groups = self._tracked_path_groups(event, base_revision)
        untracked = [] if event is ReviewEvent.PRE_COMMIT else self._untracked_paths()

        if normalized_focus:
            tracked_groups = [group for group in tracked_groups if normalized_focus in group]
            untracked = [path for path in untracked if path == normalized_focus]
        selected_tracked: list[str] = []
        selected_untracked: list[str] = []
        coverage: list[CoverageIssue] = []

        def validate_path(path: str) -> Optional[str]:
            try:
                normalized = self._relative_path(path)
            except SnapshotCaptureError:
                coverage.append(CoverageIssue(path=path, reason="outside_repository_root"))
                return None
            if self._is_ignored(normalized):
                return None
            if self._is_excluded(normalized):
                coverage.append(CoverageIssue(path=normalized, reason="excluded"))
                return None
            file_issue = (
                self._inspect_index_file(normalized, base_revision)
                if event is ReviewEvent.PRE_COMMIT
                else self._inspect_current_file(normalized, base_revision)
            )
            if file_issue:
                coverage.append(CoverageIssue(path=normalized, reason=file_issue))
                return None
            return normalized

        # A rename/copy is one security unit. If either side is unavailable or
        # excluded, selecting the other side alone can expose the full source.
        for group in tracked_groups:
            normalized_group = [validate_path(path) for path in group]
            if all(normalized_group):
                selected_tracked.extend(path for path in normalized_group if path is not None)
        for path in untracked:
            normalized = validate_path(path)
            if normalized is not None:
                selected_untracked.append(normalized)

        diff_parts = []
        if selected_tracked:
            diff_parts.append(self._capture_diff(event, base_revision, selected_tracked))
        for path in selected_untracked:
            diff_parts.append(self._capture_untracked_addition(path))
        captured_diff = "".join(diff_parts)

        parsed_files = parse_plain_diff(captured_diff) if captured_diff.strip() else []
        parsed_paths = {
            path
            for item in parsed_files
            for path in (getattr(item, "filename", None), getattr(item, "old_filename", None))
            if path
        }
        expected_paths = set(selected_tracked) | set(selected_untracked)
        for missing_path in sorted(expected_paths - parsed_paths):
            coverage.append(CoverageIssue(path=missing_path, reason="binary_or_unparseable_diff"))

        # Serializing the already parsed objects is the narrow reuse seam: the
        # snapshot and PlainDiffGitProvider validate through the same parser.
        safe_diff = "".join(item.patch for item in parsed_files)
        changed_paths = tuple(sorted(parsed_paths))
        if not changed_paths and not coverage:
            coverage.append(CoverageIssue(reason="no_changes"))

        return ReviewSnapshot(
            event=event,
            repository_root=str(self.repository_root),
            base_revision=base_revision,
            changed_paths=changed_paths,
            focus_path=normalized_focus,
            diff=safe_diff,
            task_intent=task_intent,
            deterministic_results=tuple(deterministic_results),
            review_configuration_hash=review_configuration_hash,
            policy_version=policy_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_snapshot_id=parent_snapshot_id,
            coverage_issues=tuple(coverage),
        )

    def recapture(self, snapshot: ReviewSnapshot) -> ReviewSnapshot:
        return self.capture(
            event=snapshot.event,
            base=snapshot.base_revision,
            focus_path=snapshot.focus_path,
            task_intent=snapshot.task_intent,
            deterministic_results=snapshot.deterministic_results,
            review_configuration_hash=snapshot.review_configuration_hash,
            policy_version=snapshot.policy_version,
            parent_snapshot_id=snapshot.parent_snapshot_id,
        )


class SnapshotCache:
    """Small repository-local cache keyed by snapshot and policy identity."""

    def __init__(self, repository_root: Path, max_entries: int = 50) -> None:
        common_dir = _run_git(repository_root, "rev-parse", "--git-common-dir").decode("utf-8").strip()
        git_dir = Path(common_dir)
        if not git_dir.is_absolute():
            git_dir = repository_root / git_dir
        self.cache_dir = git_dir.resolve() / "pr-agent" / "snapshot-cache"
        self.max_entries = max(1, int(max_entries))

    def _path(self, snapshot_id: str) -> Path:
        digest = snapshot_id.removeprefix("sha256:")
        return self.cache_dir / f"{digest}.json"

    def read(self, snapshot_id: str) -> Optional[ReviewSnapshotResult]:
        path = self._path(snapshot_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if data.get("snapshot_id") != snapshot_id:
            return None
        return ReviewSnapshotResult(
            snapshot_id=data["snapshot_id"],
            state=ReviewResultState(data["state"]),
            current_snapshot_id=data.get("current_snapshot_id"),
            review=data.get("review"),
            coverage_issues=tuple(CoverageIssue(**issue) for issue in data.get("coverage_issues", [])),
            latency_seconds=float(data.get("latency_seconds", 0)),
            usage=data.get("usage", {}),
            cost=data.get("cost", {}),
            cached=True,
            advisory=True,
            shadow_capable=True,
        )

    def write(self, result: ReviewSnapshotResult) -> None:
        unavailable_states = {
            ReviewResultState.STALE,
            ReviewResultState.CANCELLED,
            ReviewResultState.COVERAGE_UNAVAILABLE,
        }
        if result.state in unavailable_states:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.cache_dir, delete=False) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, self._path(result.snapshot_id))
        cached_paths = sorted(self.cache_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for old_path in cached_paths[self.max_entries:]:
            try:
                old_path.unlink()
            except OSError:
                # Eviction is best effort; an in-use or concurrently removed cache
                # entry must not turn a completed review into unavailable coverage.
                pass


def build_snapshot_result(
    snapshot: ReviewSnapshot,
    *,
    current_snapshot: ReviewSnapshot,
    structured_review: Optional[Mapping[str, Any]],
    started_at: float,
    error: Optional[str] = None,
) -> ReviewSnapshotResult:
    coverage = list(snapshot.coverage_issues)
    if current_snapshot.snapshot_id != snapshot.snapshot_id:
        return ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=ReviewResultState.STALE,
            current_snapshot_id=current_snapshot.snapshot_id,
            review=None,
            coverage_issues=tuple(coverage),
            latency_seconds=monotonic() - started_at,
        )
    if error or structured_review is None or not snapshot.diff.strip():
        if error:
            coverage.append(CoverageIssue(reason=f"review_failed:{error}"))
        elif not snapshot.diff.strip() and not coverage:
            coverage.append(CoverageIssue(reason="no_reviewable_diff"))
        return ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=ReviewResultState.COVERAGE_UNAVAILABLE,
            current_snapshot_id=current_snapshot.snapshot_id,
            review=None,
            coverage_issues=tuple(coverage),
            latency_seconds=monotonic() - started_at,
        )

    findings = finding_count(structured_review)
    state = ReviewResultState.FINDINGS if findings else ReviewResultState.NO_FINDINGS
    review = structured_review.get("review") if isinstance(structured_review, Mapping) else None
    if not findings and coverage:
        state = ReviewResultState.COVERAGE_UNAVAILABLE
        review = None
    metadata = structured_review.get("metadata", {}) if isinstance(structured_review, Mapping) else {}
    return ReviewSnapshotResult(
        snapshot_id=snapshot.snapshot_id,
        state=state,
        current_snapshot_id=current_snapshot.snapshot_id,
        review=review,
        coverage_issues=tuple(coverage),
        latency_seconds=monotonic() - started_at,
        usage=structured_review.get("usage", {}) if isinstance(structured_review, Mapping) else {},
        cost=metadata.get("cost", {}) if isinstance(metadata, Mapping) else {},
    )
