"""Capture and validate immutable local review snapshots."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from pr_agent.algo.git_patch_processing import RE_HUNK_HEADER
from pr_agent.algo.review_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    CoverageIssue,
    ReviewEvent,
    ReviewResultState,
    ReviewSnapshot,
    ReviewSnapshotResult,
    finding_count,
)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.diff_parsing import to_hunk_only_patch
from pr_agent.git_providers.plain_diff_provider import parse_plain_diff
from pr_agent.log import get_logger


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


def _run_git_bounded(
    repository_root: Path,
    *args: str,
    max_output_bytes: int,
    allowed_returncodes: tuple[int, ...] = (0,),
    stdin_bytes: Optional[bytes] = None,
) -> Optional[bytes]:
    """Read at most one byte beyond a caller's Git-output budget."""
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stderr:
        if stdin_bytes:
            stdin.write(stdin_bytes)
            stdin.seek(0)
        process = subprocess.Popen(
            ["git", "-C", str(repository_root), *args],
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise SnapshotCaptureError("failed to capture git output")
        output = process.stdout.read(max(0, max_output_bytes) + 1)
        exceeded = len(output) > max_output_bytes
        if exceeded:
            process.terminate()
            process.stdout.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.stdout.close()
            process.wait()
        stderr.seek(0)
        error_output = stderr.read()
    if exceeded:
        return None
    if process.returncode not in allowed_returncodes:
        message = error_output.decode("utf-8", errors="replace").strip()
        raise SnapshotCaptureError(message or f"git {' '.join(args)} failed with exit code {process.returncode}")
    return output


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
    return Path(process.stdout.decode("utf-8", errors="surrogateescape").rstrip("\r\n")).resolve()


def _decode_z_paths(output: bytes) -> list[str]:
    return [part.decode("utf-8", errors="surrogateescape") for part in output.split(b"\0") if part]


def _normalize_patterns(value: Optional[Sequence[str] | str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(pattern for pattern in value if isinstance(pattern, str) and pattern)


_LOCAL_PAIR_REVIEW_LIMIT_DEFAULTS = {
    "max_file_bytes": 1_000_000,
    "max_snapshot_bytes": 5_000_000,
    "max_path_discovery_bytes": 1_000_000,
    "cache_max_entries": 50,
}


def validate_local_pair_review_limits(settings: Mapping[str, Any]) -> dict[str, int]:
    limits = {}
    for key, default in _LOCAL_PAIR_REVIEW_LIMIT_DEFAULTS.items():
        value = settings.get(key, default) if hasattr(settings, "get") else default
        try:
            limits[key] = max(0, int(value))
        except (TypeError, ValueError) as exc:
            raise SnapshotCaptureError(
                f"local_pair_review.{key} must be an integer"
            ) from exc
    return limits


class LocalPairReview:
    """Build exact local diffs without requiring a clean worktree or hosted PR."""

    def __init__(
        self,
        repository_root: Optional[str] = None,
        *,
        excluded_paths: Optional[Sequence[str]] = None,
        ignored_paths: Optional[Sequence[str]] = None,
        max_file_bytes: Optional[int] = None,
        max_snapshot_bytes: Optional[int] = None,
        max_path_discovery_bytes: Optional[int] = None,
    ) -> None:
        self.repository_root = find_repository_root(repository_root)
        settings = get_settings().get("local_pair_review", {}) or {}
        configured_exclusions = settings.get("excluded_paths", []) if hasattr(settings, "get") else []
        self.excluded_paths = _normalize_patterns(
            excluded_paths if excluded_paths is not None else configured_exclusions
        )
        self.ignored_paths = _normalize_patterns(ignored_paths)
        configured_limits = validate_local_pair_review_limits(settings)
        explicit_limits = {
            "max_file_bytes": max_file_bytes,
            "max_snapshot_bytes": max_snapshot_bytes,
            "max_path_discovery_bytes": max_path_discovery_bytes,
        }
        effective_limits = validate_local_pair_review_limits({
            key: configured_limits[key] if value is None else value
            for key, value in explicit_limits.items()
        })
        self.max_file_bytes = effective_limits["max_file_bytes"]
        self.max_snapshot_bytes = effective_limits["max_snapshot_bytes"]
        self.max_path_discovery_bytes = effective_limits["max_path_discovery_bytes"]

    def _relative_path(self, supplied_path: str) -> str:
        path = Path(supplied_path)
        candidate = path if path.is_absolute() else self.repository_root / path
        # Resolve existing symlinks as well as lexical ``..``. A symlink that points
        # outside the worktree is not safe input even when its link lives inside it.
        lexical = Path(os.path.abspath(candidate))
        try:
            resolved = candidate.resolve(strict=False)
        except RuntimeError:
            # A looped symlink is still a repository entry that must appear in
            # coverage. Keep its lexical path so inspection can classify and
            # fingerprint the link itself without following it.
            try:
                relative = lexical.relative_to(self.repository_root)
            except ValueError as exc:
                raise SnapshotCaptureError(f"path is outside repository root: {supplied_path}") from exc
            if relative == Path("."):
                raise SnapshotCaptureError("a repository root is not a reviewable file path")
            return relative.as_posix()
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
        return path in self.ignored_paths

    def _has_content_filter(self, event: ReviewEvent, path: str) -> bool:
        args = ["check-attr"]
        if event is ReviewEvent.PRE_COMMIT:
            args.append("--cached")
        args.extend(["-z", "filter", "--", path])
        fields = _decode_z_paths(
            _run_git(self.repository_root, "--literal-pathspecs", *args)
        )
        if len(fields) < 3:
            return False
        value = fields[2].lower()
        return value not in {"unspecified", "unset"}

    def _diff_filter_overrides(self) -> list[str]:
        """Replace repository-configured content filters with raw pass-throughs."""
        output = _run_git(
            self.repository_root,
            "config",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|process|required)$",
            allowed_returncodes=(0, 1),
        )
        drivers = set()
        for raw_key in output.decode("utf-8", errors="replace").splitlines():
            match = re.match(r"^filter\.(.+)\.(?:clean|process|required)$", raw_key, re.IGNORECASE)
            if match:
                drivers.add(match.group(1))
        overrides = []
        for driver in sorted(drivers):
            overrides.extend([
                "-c", f"filter.{driver}.clean=cat",
                "-c", f"filter.{driver}.process=",
                "-c", f"filter.{driver}.required=false",
            ])
        return overrides

    def _git_object_identity(self, revision: str, path: str) -> tuple[Optional[str], Optional[str]]:
        if revision == ":":
            output = _run_git(
                self.repository_root, "--literal-pathspecs", "ls-files", "--stage", "--", path
            )
            fields = output.decode("utf-8", errors="replace").strip().split(maxsplit=3)
            return (fields[0], fields[1]) if len(fields) >= 2 else (None, None)
        output = _run_git(
            self.repository_root, "--literal-pathspecs", "ls-tree", revision, "--", path
        )
        fields = output.decode("utf-8", errors="replace").strip().split(maxsplit=3)
        return (fields[0], fields[2]) if len(fields) >= 3 else (None, None)

    def _path_fingerprint(self, event: ReviewEvent, path: str, base_revision: str) -> Optional[str]:
        if event is ReviewEvent.PRE_COMMIT:
            mode, object_id = self._git_object_identity(":", path)
            if object_id is None:
                mode, object_id = self._git_object_identity(base_revision, path)
            return f"git:{mode}:{object_id}" if object_id else None

        candidate = self.repository_root / path
        try:
            digest = hashlib.sha256()
            if candidate.is_symlink():
                digest.update(b"mode:120000\0")
                digest.update(os.readlink(candidate).encode("utf-8", errors="surrogateescape"))
                return "sha256:" + digest.hexdigest()
            if candidate.is_file():
                file_stat = candidate.stat()
                worktree_mode = "100755" if file_stat.st_mode & stat.S_IXUSR else "100644"
                digest.update(f"mode:{worktree_mode}\0".encode("ascii"))
                digest.update(
                    f"size:{file_stat.st_size}\0mtime:{file_stat.st_mtime_ns}\0"
                    f"ctime:{file_stat.st_ctime_ns}\0".encode("ascii")
                )
                sample_size = max(1, min(self.max_file_bytes + 1, 64 * 1024))
                with candidate.open("rb") as handle:
                    digest.update(handle.read(sample_size))
                    if file_stat.st_size > sample_size:
                        handle.seek(max(0, file_stat.st_size - sample_size))
                        digest.update(handle.read(sample_size))
                return "sha256:" + digest.hexdigest()
        except OSError:
            return None
        mode, object_id = self._git_object_identity(base_revision, path)
        if mode == "160000" and candidate.is_dir():
            # The parent repository's base gitlink cannot describe the live
            # submodule HEAD or dirty worktree. Returning no fingerprint makes
            # result publication fail closed until submodules are supported.
            return None
        return f"git:{mode}:{object_id}" if object_id else None

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

    def _inspect_git_object(self, mode: Optional[str], object_id: Optional[str]) -> Optional[str]:
        """Inspect an immutable Git blob without materializing an oversized object."""
        if mode is None or object_id is None:
            return None
        if mode == "120000":
            return "symlink"
        try:
            size = int(_run_git(self.repository_root, "cat-file", "-s", object_id).strip())
        except (SnapshotCaptureError, ValueError):
            return "unreadable"
        if size > self.max_file_bytes:
            return "file_too_large"
        try:
            content = _run_git(self.repository_root, "cat-file", "blob", object_id)
        except SnapshotCaptureError:
            return "unreadable"
        return self._inspect_content(content)

    def _inspect_revision_file(self, revision: str, path: str) -> Optional[str]:
        return self._inspect_git_object(*self._git_object_identity(revision, path))

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
            with candidate.open("rb") as handle:
                content = handle.read(self.max_file_bytes + 1)
        except OSError:
            return "unreadable"
        return self._inspect_content(content)

    def _inspect_index_file(self, path: str, base_revision: str) -> Optional[str]:
        """Inspect the exact staged blob used by a pre-commit snapshot."""
        base_issue = self._inspect_revision_file(base_revision, path)
        if base_issue:
            return base_issue
        mode, object_id = self._git_object_identity(":", path)
        if object_id is None:
            return self._inspect_revision_file(base_revision, path)
        return self._inspect_git_object(mode, object_id)

    def _resolve_base(self, base: str) -> str:
        try:
            resolved = _run_git(
                self.repository_root, "rev-parse", "--verify", f"{base}^{{commit}}"
            )
            return resolved.decode("ascii").strip()
        except SnapshotCaptureError as exc:
            if base != "HEAD":
                raise
            symbolic_head = subprocess.run(
                ["git", "-C", str(self.repository_root), "symbolic-ref", "-q", "HEAD"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if symbolic_head.returncode != 0:
                raise exc
            empty_tree = subprocess.run(
                ["git", "-C", str(self.repository_root), "hash-object", "-t", "tree", "--stdin"],
                input=b"",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if empty_tree.returncode != 0:
                message = empty_tree.stderr.decode("utf-8", errors="replace").strip()
                raise SnapshotCaptureError(
                    message or "could not resolve the empty Git tree"
                ) from exc
            return empty_tree.stdout.decode("ascii").strip()

    def _tracked_path_groups(
        self,
        event: ReviewEvent,
        base_revision: str,
        focus_path: Optional[str] = None,
    ) -> Optional[list[tuple[str, ...]]]:
        args = [
            *self._diff_filter_overrides(),
            "--literal-pathspecs",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
        ]
        if event is ReviewEvent.PRE_COMMIT:
            args.append("--cached")
        args.extend(
            [
                "--name-status",
                "-z",
                "--find-copies",
                "--find-copies-harder",
                base_revision,
                "--",
            ]
        )
        output = _run_git_bounded(
            self.repository_root,
            *args,
            max_output_bytes=self.max_path_discovery_bytes,
        )
        if output is None:
            return None
        fields = _decode_z_paths(output)
        path_groups = []
        index = 0
        while index < len(fields):
            status = fields[index]
            index += 1
            path_count = 2 if status.startswith(("R", "C")) else 1
            path_groups.append(tuple(fields[index:index + path_count]))
            index += path_count
        if focus_path:
            return [group for group in path_groups if focus_path in group]
        return path_groups

    def _tracked_filtered_paths(
        self,
        event: ReviewEvent,
        focus_path: Optional[str] = None,
    ) -> Optional[set[str]]:
        tracked_args = ["--literal-pathspecs", "ls-files", "-z", "--"]
        if focus_path:
            tracked_args.append(focus_path)
        tracked = _run_git_bounded(
            self.repository_root,
            *tracked_args,
            max_output_bytes=self.max_path_discovery_bytes,
        )
        if tracked is None:
            return None
        if not tracked:
            return set()
        attribute_args = ["--literal-pathspecs", "check-attr"]
        if event is ReviewEvent.PRE_COMMIT:
            attribute_args.append("--cached")
        attribute_args.extend(["-z", "--stdin", "filter"])
        attributes = _run_git_bounded(
            self.repository_root,
            *attribute_args,
            max_output_bytes=self.max_path_discovery_bytes,
            stdin_bytes=tracked,
        )
        if attributes is None:
            return None
        fields = _decode_z_paths(attributes)
        if len(fields) % 3:
            raise SnapshotCaptureError("git check-attr returned malformed path data")
        return {
            fields[index]
            for index in range(0, len(fields), 3)
            if fields[index + 2].lower() not in {"unspecified", "unset"}
        }

    def _untracked_paths(self, focus_path: Optional[str] = None) -> Optional[list[str]]:
        args = [
            "--literal-pathspecs",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
        ]
        if focus_path:
            args.append(focus_path)
        output = _run_git_bounded(
            self.repository_root,
            *args,
            max_output_bytes=self.max_path_discovery_bytes,
        )
        return None if output is None else _decode_z_paths(output)

    def _capture_diff(
        self,
        event: ReviewEvent,
        base_revision: str,
        paths: Sequence[str],
        *,
        max_output_bytes: Optional[int] = None,
    ) -> Optional[str]:
        if not paths:
            return ""
        args = [
            *self._diff_filter_overrides(),
            "-c", "core.quotePath=false",
            "--literal-pathspecs", "diff", "--no-color", "--src-prefix=a/", "--dst-prefix=b/",
            "--no-ext-diff", "--no-textconv", "--find-renames",
        ]
        if event is ReviewEvent.PRE_COMMIT:
            args.append("--cached")
        args.extend([base_revision, "--", *paths])
        output = (
            _run_git(self.repository_root, *args)
            if max_output_bytes is None
            else _run_git_bounded(
                self.repository_root,
                *args,
                max_output_bytes=max_output_bytes,
            )
        )
        return None if output is None else output.decode("utf-8", errors="surrogateescape")

    def _capture_untracked_addition(
        self,
        path: str,
        *,
        max_output_bytes: Optional[int] = None,
    ) -> Optional[str]:
        args = (
            *self._diff_filter_overrides(),
            "-c", "core.quotePath=false",
            "diff", "--no-color", "--src-prefix=a/", "--dst-prefix=b/", "--no-index",
            "--no-ext-diff", "--no-textconv",
            "--", "/dev/null", path,
        )
        output = (
            _run_git(self.repository_root, *args, allowed_returncodes=(0, 1))
            if max_output_bytes is None
            else _run_git_bounded(
                self.repository_root,
                *args,
                max_output_bytes=max_output_bytes,
                allowed_returncodes=(0, 1),
            )
        )
        return None if output is None else output.decode("utf-8", errors="surrogateescape")

    def capture(
        self,
        *,
        event: ReviewEvent | str,
        base: str = "HEAD",
        base_selector: Optional[str] = None,
        focus_path: Optional[str] = None,
        task_intent: Optional[str] = None,
        deterministic_results: Iterable[Mapping[str, Any]] = (),
        review_configuration_hash: Optional[str] = None,
        review_configuration_hash_factory: Optional[Callable[[str], Optional[str]]] = None,
        policy_version: str = "local-pair-review-v1",
        parent_snapshot_id: Optional[str] = None,
    ) -> ReviewSnapshot:
        event = ReviewEvent.parse(event) if isinstance(event, str) else event
        if event is ReviewEvent.FILE_SAVE and not focus_path:
            raise SnapshotCaptureError("file-save snapshots require --path")

        base_revision = self._resolve_base(base)
        if review_configuration_hash_factory is not None:
            review_configuration_hash = review_configuration_hash_factory(base_revision)
        normalized_focus = self._relative_path(focus_path) if focus_path else None
        if (
            event is ReviewEvent.FILE_SAVE
            and normalized_focus
            and (self.repository_root / normalized_focus).is_dir()
        ):
            raise SnapshotCaptureError("file-save --path must identify a file, not a directory")
        selected_tracked: list[str] = []
        selected_tracked_groups: list[tuple[str, ...]] = []
        selected_untracked: list[str] = []
        coverage: list[CoverageIssue] = []

        def add_coverage(path: Optional[str], reason: str) -> None:
            fingerprint = self._path_fingerprint(event, path, base_revision) if path else None
            coverage.append(CoverageIssue(path=path, reason=reason, fingerprint=fingerprint))

        filtered_paths = self._tracked_filtered_paths(event, normalized_focus)
        discovery_overflow = filtered_paths is None
        if filtered_paths is None:
            add_coverage(None, "filtered_path_discovery_budget")
            filtered_paths = set()
        else:
            for filtered_path in sorted(filtered_paths):
                add_coverage(filtered_path, "content_filter_unsupported")

        tracked_groups = self._tracked_path_groups(
            event, base_revision, normalized_focus
        )
        discovery_overflow = discovery_overflow or tracked_groups is None
        if tracked_groups is None:
            add_coverage(None, "tracked_path_discovery_budget")
            tracked_groups = []
        untracked = (
            []
            if event is ReviewEvent.PRE_COMMIT
            else self._untracked_paths(normalized_focus)
        )
        if untracked is None:
            add_coverage(None, "untracked_path_discovery_budget")
            untracked = []
            discovery_overflow = True
        if discovery_overflow:
            # An incomplete path set cannot safely produce feedback. Discard the
            # other discovery stage as well so omitted repository state can never
            # be hidden behind a current-looking partial review.
            return ReviewSnapshot(
                event=event,
                repository_root=str(self.repository_root),
                base_selector=base_selector or base,
                base_revision=base_revision,
                changed_paths=(),
                focus_path=normalized_focus,
                diff="",
                task_intent=task_intent,
                deterministic_results=tuple(deterministic_results),
                review_configuration_hash=review_configuration_hash,
                policy_version=policy_version,
                created_at=datetime.now(timezone.utc).isoformat(),
                parent_snapshot_id=parent_snapshot_id,
                coverage_issues=tuple(coverage),
            )

        def validate_path(path: str) -> Optional[str]:
            try:
                normalized = self._relative_path(path)
            except SnapshotCaptureError:
                add_coverage(path, "outside_repository_root")
                return None
            if self._is_ignored(normalized):
                return None
            if self._is_excluded(normalized):
                add_coverage(normalized, "excluded")
                return None
            if self._has_content_filter(event, normalized):
                add_coverage(normalized, "content_filter_unsupported")
                return None
            file_issue = (
                self._inspect_index_file(normalized, base_revision)
                if event is ReviewEvent.PRE_COMMIT
                else self._inspect_current_file(normalized, base_revision)
            )
            if file_issue:
                add_coverage(normalized, file_issue)
                return None
            return normalized

        # A rename/copy is one security unit. If either side is unavailable or
        # excluded, selecting the other side alone can expose the full source.
        for group in tracked_groups:
            if any(path in filtered_paths for path in group):
                covered_paths = {issue.path for issue in coverage}
                for raw_path in group:
                    normalized = self._relative_path(raw_path)
                    if normalized not in covered_paths:
                        add_coverage(normalized, "rename_group_omitted")
                        covered_paths.add(normalized)
                continue
            normalized_group = [validate_path(path) for path in group]
            if all(normalized_group):
                selected_group = tuple(
                    path for path in normalized_group if path is not None
                )
                selected_tracked_groups.append(selected_group)
                selected_tracked.extend(selected_group)
            else:
                covered_paths = {issue.path for issue in coverage}
                for raw_path in group:
                    try:
                        normalized = self._relative_path(raw_path)
                    except SnapshotCaptureError:
                        continue
                    if normalized not in covered_paths:
                        add_coverage(normalized, "rename_group_omitted")
                        covered_paths.add(normalized)
        for path in untracked:
            normalized = validate_path(path)
            if normalized is not None:
                selected_untracked.append(normalized)

        def capture_selected() -> tuple[str, set[str]]:
            diff_parts = []
            omitted_paths: set[str] = set()
            remaining_bytes = self.max_snapshot_bytes
            if selected_tracked:
                part = None if remaining_bytes <= 0 else self._capture_diff(
                    event,
                    base_revision,
                    selected_tracked,
                    max_output_bytes=remaining_bytes,
                )
                if part is None:
                    omitted_paths.update(selected_tracked)
                else:
                    part_size = len(part.encode("utf-8", errors="surrogateescape"))
                    if part_size > remaining_bytes:
                        omitted_paths.update(selected_tracked)
                    else:
                        diff_parts.append(part)
                        remaining_bytes -= part_size
            for selected_path in selected_untracked:
                if remaining_bytes <= 0:
                    omitted_paths.add(selected_path)
                    continue
                part = self._capture_untracked_addition(
                    selected_path,
                    max_output_bytes=remaining_bytes,
                )
                if part is None:
                    omitted_paths.add(selected_path)
                    continue
                part_size = len(part.encode("utf-8", errors="surrogateescape"))
                if part_size > remaining_bytes:
                    omitted_paths.add(selected_path)
                    continue
                diff_parts.append(part)
                remaining_bytes -= part_size
            return "".join(diff_parts), omitted_paths

        try:
            captured_diff, captured_omissions = capture_selected()
            revalidated = all(
                validate_path(selected_path) is not None
                for selected_path in (*selected_tracked, *selected_untracked)
            )
            if revalidated:
                verified_diff, verified_omissions = capture_selected()
            else:
                verified_diff, verified_omissions = "", set()
        except UnicodeDecodeError:
            add_coverage(None, "content_changed_during_capture")
            captured_diff = verified_diff = ""
            captured_omissions = verified_omissions = set()
            revalidated = False
        if (
            not revalidated
            or verified_diff != captured_diff
            or verified_omissions != captured_omissions
        ):
            if revalidated:
                add_coverage(None, "content_changed_during_capture")
            captured_diff = ""
            selected_tracked.clear()
            selected_tracked_groups.clear()
            selected_untracked.clear()
            budget_omitted_paths: set[str] = set()
        else:
            captured_diff = verified_diff
            budget_omitted_paths = verified_omissions
            for omitted_path in sorted(budget_omitted_paths):
                add_coverage(omitted_path, "snapshot_byte_budget")

        parsed_files = parse_plain_diff(captured_diff) if captured_diff.strip() else []
        reviewable_files = []
        metadata_only_paths = set()
        for item in parsed_files:
            if to_hunk_only_patch(item.patch).strip():
                reviewable_files.append(item)
                continue
            for path in (getattr(item, "filename", None), getattr(item, "old_filename", None)):
                if path:
                    metadata_only_paths.add(path)
                    add_coverage(path, "metadata_only_diff")
        parsed_files = reviewable_files
        parsed_paths = {
            path
            for item in parsed_files
            for path in (getattr(item, "filename", None), getattr(item, "old_filename", None))
            if path
        }
        expected_paths = (
            set(selected_tracked) | set(selected_untracked)
        ) - budget_omitted_paths
        for missing_path in sorted(expected_paths - parsed_paths - metadata_only_paths):
            add_coverage(missing_path, "binary_or_unparseable_diff")

        # Serializing the already parsed objects is the narrow reuse seam: the
        # snapshot and PlainDiffGitProvider validate through the same parser.
        safe_diff = "".join(item.patch for item in parsed_files)
        changed_paths = tuple(sorted(parsed_paths))
        if not changed_paths and not coverage:
            coverage.append(CoverageIssue(reason="no_changes"))

        return ReviewSnapshot(
            event=event,
            repository_root=str(self.repository_root),
            base_selector=base_selector or base,
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

    def recapture(
        self,
        snapshot: ReviewSnapshot,
        *,
        review_configuration_hash_factory: Optional[
            Callable[[str], Optional[str]]
        ] = None,
    ) -> ReviewSnapshot:
        return self.capture(
            event=snapshot.event,
            base=snapshot.base_selector or snapshot.base_revision,
            base_selector=snapshot.base_selector,
            focus_path=snapshot.focus_path,
            task_intent=snapshot.task_intent,
            deterministic_results=snapshot.deterministic_results,
            review_configuration_hash=snapshot.review_configuration_hash,
            review_configuration_hash_factory=review_configuration_hash_factory,
            policy_version=snapshot.policy_version,
            parent_snapshot_id=snapshot.parent_snapshot_id,
        )


def _findings_match_snapshot(
    snapshot: ReviewSnapshot,
    structured_review: Mapping[str, Any],
) -> bool:
    review = structured_review.get("review")
    findings = review.get("key_issues_to_review") if isinstance(review, Mapping) else None
    if not isinstance(findings, list):
        return False

    reviewable_lines: dict[str, set[int]] = {}
    for item in parse_plain_diff(snapshot.diff):
        filename = getattr(item, "filename", None)
        if not filename or filename not in snapshot.changed_paths:
            continue
        lines = reviewable_lines.setdefault(filename, set())
        new_line = None
        for patch_line in item.patch.splitlines():
            hunk_match = RE_HUNK_HEADER.match(patch_line)
            if hunk_match:
                new_line = int(hunk_match.group(3))
                continue
            if new_line is None or patch_line.startswith("\\"):
                continue
            if patch_line.startswith("-"):
                continue
            if patch_line.startswith(("+", " ")):
                lines.add(new_line)
                new_line += 1

    for finding in findings:
        if not isinstance(finding, Mapping):
            return False
        filename = finding.get("relevant_file")
        try:
            start_line = int(str(finding.get("start_line", "")).strip())
            end_line = int(str(finding.get("end_line", "")).strip())
        except ValueError:
            return False
        allowed_lines = reviewable_lines.get(filename, set())
        if (
            not allowed_lines
            or start_line < min(allowed_lines)
            or end_line > max(allowed_lines)
            or any(line not in allowed_lines for line in range(start_line, end_line + 1))
        ):
            return False
    return True


class SnapshotCache:
    """Small repository-local cache keyed by snapshot and policy identity."""

    def __init__(self, repository_root: Path, max_entries: int = 50) -> None:
        common_dir = _run_git(repository_root, "rev-parse", "--git-common-dir").decode(
            "utf-8", errors="surrogateescape"
        ).rstrip("\r\n")
        git_dir = Path(common_dir)
        if not git_dir.is_absolute():
            git_dir = repository_root / git_dir
        self.git_dir = git_dir.resolve()
        self.cache_dir = self.git_dir / "pr-agent" / "snapshot-cache"
        self.max_entries = max(1, int(max_entries))

    def _path(self, snapshot_id: str) -> Path:
        digest = snapshot_id.removeprefix("sha256:")
        return self.cache_dir / f"{digest}.json"

    def _open_cache_dir(self, *, create: bool) -> Optional[int]:
        required_dir_fd_functions = (os.open, os.mkdir, os.stat, os.unlink)
        if any(function not in os.supports_dir_fd for function in required_dir_fd_functions):
            raise OSError("descriptor-relative snapshot cache access is unavailable")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        current_fd = os.open(self.git_dir, flags)
        try:
            for name in ("pr-agent", "snapshot-cache"):
                try:
                    next_fd = os.open(name, flags, dir_fd=current_fd)
                except FileNotFoundError:
                    if not create:
                        os.close(current_fd)
                        return None
                    try:
                        os.mkdir(name, mode=0o700, dir_fd=current_fd)
                    except FileExistsError:
                        # A concurrent hook may have created the directory. The
                        # no-follow open below decides whether it is safe.
                        pass
                    next_fd = os.open(name, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            try:
                os.close(current_fd)
            except OSError:
                # The traversal failure may already have closed the descriptor.
                pass
            raise

    def read(
        self,
        snapshot_id: str,
        *,
        snapshot: Optional[ReviewSnapshot] = None,
    ) -> Optional[ReviewSnapshotResult]:
        cache_fd = None
        try:
            cache_fd = self._open_cache_dir(create=False)
            if cache_fd is None:
                return None
            path_name = self._path(snapshot_id).name
            path_fd = os.open(
                path_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=cache_fd,
            )
            path_stat = os.fstat(path_fd)
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
                os.close(path_fd)
                return None
            with os.fdopen(path_fd, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or data.get("snapshot_id") != snapshot_id:
                return None
            state = ReviewResultState(data["state"])
            current_snapshot_id = data.get("current_snapshot_id")
            review = data.get("review")
            coverage_issues = tuple(
                CoverageIssue(**issue) for issue in data.get("coverage_issues", [])
            )
            latency_seconds = float(data.get("latency_seconds", 0))
            usage = data.get("usage", {})
            cost = data.get("cost", {})
            findings = finding_count({"review": review})
            if (
                data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
                or current_snapshot_id != snapshot_id
                or state not in {ReviewResultState.FINDINGS, ReviewResultState.NO_FINDINGS}
                or not isinstance(review, Mapping)
                or findings is None
                or not isinstance(usage, Mapping)
                or not isinstance(cost, Mapping)
                or not math.isfinite(latency_seconds)
                or latency_seconds < 0
                or data.get("advisory") is not True
                or data.get("shadow_capable") is not True
                or (state is ReviewResultState.FINDINGS and findings == 0)
                or (state is ReviewResultState.NO_FINDINGS and (findings != 0 or coverage_issues))
                or (
                    snapshot is not None
                    and not _findings_match_snapshot(snapshot, {"review": review})
                )
            ):
                return None
            return ReviewSnapshotResult(
                snapshot_id=data["snapshot_id"],
                state=state,
                current_snapshot_id=current_snapshot_id,
                review=review,
                coverage_issues=coverage_issues,
                latency_seconds=latency_seconds,
                usage=usage,
                cost=cost,
                cached=True,
                advisory=True,
                shadow_capable=True,
            )
        except (OSError, KeyError, ValueError, TypeError, AttributeError):
            return None
        finally:
            if cache_fd is not None:
                os.close(cache_fd)

    def write(self, result: ReviewSnapshotResult) -> None:
        unavailable_states = {
            ReviewResultState.STALE,
            ReviewResultState.CANCELLED,
            ReviewResultState.COVERAGE_UNAVAILABLE,
        }
        if result.state in unavailable_states:
            return
        try:
            self._write(result)
        except OSError as exc:
            get_logger().warning(
                "Could not persist local snapshot cache; continuing without cache",
                artifact={"error": type(exc).__name__},
            )

    def _write(self, result: ReviewSnapshotResult) -> None:
        cache_fd = self._open_cache_dir(create=True)
        if cache_fd is None:
            raise OSError("could not create snapshot cache directory")
        payload = json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        temporary_name = f".pr-agent-{secrets.token_hex(16)}.tmp"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=cache_fd,
            )
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            try:
                os.replace(
                    temporary_name,
                    self._path(result.snapshot_id).name,
                    src_dir_fd=cache_fd,
                    dst_dir_fd=cache_fd,
                )
            except (NotImplementedError, TypeError) as exc:
                raise OSError("descriptor-relative snapshot cache replacement is unavailable") from exc
            temporary_name = ""
            cached_paths = []
            for cached_name in os.listdir(cache_fd):
                if not cached_name.endswith(".json"):
                    continue
                try:
                    cached_stat = os.stat(
                        cached_name,
                        dir_fd=cache_fd,
                        follow_symlinks=False,
                    )
                    if stat.S_ISREG(cached_stat.st_mode):
                        cached_paths.append((cached_stat.st_mtime, cached_name))
                except OSError:
                    # Another hook may evict the entry between listing and stat.
                    continue
            cached_paths.sort(key=lambda item: item[0], reverse=True)
            for _, old_name in cached_paths[self.max_entries:]:
                try:
                    os.unlink(old_name, dir_fd=cache_fd)
                except OSError:
                    # Eviction is best effort; an in-use or concurrently removed cache
                    # entry must not turn a completed review into unavailable coverage.
                    pass
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name, dir_fd=cache_fd)
                except OSError:
                    # Best-effort cleanup must not mask the original cache failure.
                    pass
            os.close(cache_fd)


def build_snapshot_result(
    snapshot: ReviewSnapshot,
    *,
    current_snapshot: Optional[ReviewSnapshot],
    structured_review: Optional[Mapping[str, Any]],
    started_at: float,
    error: Optional[str] = None,
) -> ReviewSnapshotResult:
    coverage = list(snapshot.coverage_issues)
    metadata = structured_review.get("metadata", {}) if isinstance(structured_review, Mapping) else {}
    omitted_files = metadata.get("omitted_files", []) if isinstance(metadata, Mapping) else []
    if isinstance(omitted_files, Sequence) and not isinstance(omitted_files, (str, bytes)):
        coverage.extend(
            CoverageIssue(path=str(path), reason="token_budget_omitted")
            for path in omitted_files
            if isinstance(path, str) and path
        )
    deleted_files = metadata.get("deleted_files", []) if isinstance(metadata, Mapping) else []
    if isinstance(deleted_files, Sequence) and not isinstance(deleted_files, (str, bytes)):
        coverage.extend(
            CoverageIssue(path=str(path), reason="deleted_file_unsupported")
            for path in deleted_files
            if isinstance(path, str) and path
        )
    findings = finding_count(structured_review)
    if (
        structured_review is not None
        and (
            findings is None
            or not _findings_match_snapshot(snapshot, structured_review)
        )
    ):
        error = error or "InvalidStructuredReview"
    if current_snapshot is None:
        coverage.append(CoverageIssue(reason="current_snapshot_unavailable"))
        return ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=ReviewResultState.STALE,
            current_snapshot_id=None,
            review=None,
            coverage_issues=tuple(coverage),
            latency_seconds=monotonic() - started_at,
        )
    if current_snapshot.snapshot_id != snapshot.snapshot_id:
        return ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=ReviewResultState.STALE,
            current_snapshot_id=current_snapshot.snapshot_id,
            review=None,
            coverage_issues=tuple(coverage),
            latency_seconds=monotonic() - started_at,
        )
    unfingerprintable_coverage = any(
        issue.path is not None and issue.fingerprint is None
        for issue in snapshot.coverage_issues
    )
    if error or structured_review is None or not snapshot.diff.strip() or unfingerprintable_coverage:
        if error:
            coverage.append(CoverageIssue(reason=f"review_failed:{error}"))
        elif unfingerprintable_coverage:
            coverage.append(CoverageIssue(reason="unfingerprintable_coverage"))
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

    assert findings is not None
    state = ReviewResultState.FINDINGS if findings else ReviewResultState.NO_FINDINGS
    review = structured_review.get("review") if isinstance(structured_review, Mapping) else None
    if not findings and coverage:
        state = ReviewResultState.COVERAGE_UNAVAILABLE
        review = None
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
