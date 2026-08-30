"""Capture and validate immutable local review snapshots."""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import json
import math
import os
import re
import secrets
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from pr_agent.algo.git_patch_processing import RE_HUNK_HEADER, iter_git_patch_lines
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

_DEFAULT_SECRET_EXCLUSIONS = (
    ".secrets.toml",
    "**/.secrets.toml",
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "**/*.pem",
    "*.key",
    "**/*.key",
    "credentials.json",
    "**/credentials.json",
    "service-account*.json",
    "**/service-account*.json",
    ".npmrc",
    "**/.npmrc",
    ".pypirc",
    "**/.pypirc",
    ".netrc",
    "**/.netrc",
    ".git-credentials",
    "**/.git-credentials",
    ".aws/credentials",
    "**/.aws/credentials",
    ".kube/config",
    "**/.kube/config",
    ".docker/config.json",
    "**/.docker/config.json",
    "id_rsa",
    "**/id_rsa",
    "id_ed25519",
    "**/id_ed25519",
    "id_ed25519_sk",
    "**/id_ed25519_sk",
    "id_ecdsa",
    "**/id_ecdsa",
    "id_ecdsa_sk",
    "**/id_ecdsa_sk",
    "id_dsa",
    "**/id_dsa",
    "id_xmss",
    "**/id_xmss",
)

_UNSAFE_COPY_SIMILARITY_THRESHOLD = 0.5
_UNSAFE_COPY_SMALL_SIMILARITY_THRESHOLD = 0.8
_UNSAFE_COPY_PROBE_WINDOW = 16
_UNSAFE_COPY_MAX_PROBES = 128
_UNSAFE_COPY_MAX_TOTAL_PROBES = 65_536
_UNSAFE_COPY_MAX_TOTAL_EXACT_LINES = 65_536
_UNSAFE_COPY_MAX_SMALL_COMPARISONS = 10_000
_MAX_FINDING_LINE = 2_147_483_647


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


def _read_stable_regular_file(path: Path, max_bytes: int) -> Optional[bytes]:
    """Read a regular file without following links or accepting an in-flight rewrite."""
    file_descriptor = None
    try:
        file_descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            return None
        with os.fdopen(file_descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
        after = os.fstat(file_descriptor)
    except OSError:
        return None
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                # The read is already complete or failing closed; a close error
                # cannot make the captured bytes acceptable.
                pass
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(content) != before.st_size or before_identity != after_identity:
        return None
    return content


def _rolling_copy_hashes(content: bytes) -> Iterable[int]:
    """Yield fixed-window fingerprints that survive small insertions and edits."""
    window = _UNSAFE_COPY_PROBE_WINDOW
    if len(content) < window:
        return
    mask = (1 << 64) - 1
    base = 257
    outgoing_factor = pow(base, window - 1, 1 << 64)
    fingerprint = 0
    for value in content[:window]:
        fingerprint = ((fingerprint * base) + value + 1) & mask
    yield fingerprint
    for outgoing, incoming in zip(content, content[window:], strict=False):
        fingerprint = (fingerprint - ((outgoing + 1) * outgoing_factor)) & mask
        fingerprint = ((fingerprint * base) + incoming + 1) & mask
        yield fingerprint


def _copy_similarity_probes(content: bytes) -> frozenset[int]:
    """Select a bounded, deterministic min-hash signature for copy containment."""
    selected: set[int] = set()
    largest_first: list[int] = []
    for fingerprint in _rolling_copy_hashes(content):
        if fingerprint in selected:
            continue
        if len(largest_first) < _UNSAFE_COPY_MAX_PROBES:
            heapq.heappush(largest_first, -fingerprint)
            selected.add(fingerprint)
            continue
        largest = -largest_first[0]
        if fingerprint >= largest:
            continue
        removed = -heapq.heapreplace(largest_first, -fingerprint)
        selected.remove(removed)
        selected.add(fingerprint)
    return frozenset(selected)


def _patch_has_only_deletions(patch: str) -> bool:
    inside_hunk = False
    has_addition = False
    has_deletion = False
    for line in iter_git_patch_lines(patch):
        if RE_HUNK_HEADER.match(line):
            inside_hunk = True
            continue
        if not inside_hunk or line.startswith("\\"):
            continue
        if line.startswith("+"):
            has_addition = True
        elif line.startswith("-"):
            has_deletion = True
    return has_deletion and not has_addition


def _patch_model_visible_regions(patch: str) -> tuple[bytes, ...]:
    """Return each hunk's old and new payload bytes as separate regions."""
    regions: list[bytes] = []
    old_region: list[str] = []
    new_region: list[str] = []
    inside_hunk = False

    def flush_hunk() -> None:
        for region in (old_region, new_region):
            if region:
                regions.append(
                    "".join(region).encode("utf-8", errors="surrogateescape")
                )

    for line in iter_git_patch_lines(patch):
        hunk_match = RE_HUNK_HEADER.match(line)
        if hunk_match:
            flush_hunk()
            header_context = hunk_match.group(5)
            if header_context:
                regions.append(
                    header_context.encode("utf-8", errors="surrogateescape")
                )
            old_region = []
            new_region = []
            inside_hunk = True
            continue
        if not inside_hunk or line.startswith("\\"):
            continue
        if line.startswith(" "):
            old_region.append(line[1:])
            new_region.append(line[1:])
        elif line.startswith("-"):
            old_region.append(line[1:])
        elif line.startswith("+"):
            new_region.append(line[1:])
    flush_hunk()
    return tuple(dict.fromkeys(regions))


def _status_uses_grouped_patch(status: str) -> bool:
    """Return whether provenance must use the captured path group's patch."""
    return status.startswith(("M", "R", "C"))


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


def is_snapshot_path_excluded(
    path: str,
    additional_patterns: Optional[Sequence[str] | str] = None,
) -> bool:
    """Apply the mandatory and caller-configured snapshot exclusion policy."""
    patterns = (*_DEFAULT_SECRET_EXCLUSIONS, *_normalize_patterns(additional_patterns))
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


_LOCAL_PAIR_REVIEW_LIMIT_DEFAULTS = {
    "max_file_bytes": 1_000_000,
    "max_snapshot_bytes": 5_000_000,
    "max_path_discovery_bytes": 1_000_000,
    "cache_max_entries": 50,
}

# Keep pathspec arguments comfortably below Windows' CreateProcess command-line
# limit while also leaving room for Git options and repository/config paths.
_MAX_GIT_DIFF_PATH_BYTES = 16_384


def _batch_path_groups(
    groups: Sequence[Sequence[str]],
    max_path_bytes: int = _MAX_GIT_DIFF_PATH_BYTES,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Batch atomic pathspec groups and return paths whose group cannot fit."""
    batches: list[tuple[str, ...]] = []
    omitted_paths: list[str] = []
    batch: list[str] = []
    batch_bytes = 0
    for group in groups:
        group_paths = tuple(group)
        group_bytes = sum(
            len(path.encode("utf-8", errors="surrogateescape")) + 1
            for path in group_paths
        )
        if group_bytes > max_path_bytes:
            if batch:
                batches.append(tuple(batch))
                batch = []
                batch_bytes = 0
            omitted_paths.extend(group_paths)
            continue
        if batch and batch_bytes + group_bytes > max_path_bytes:
            batches.append(tuple(batch))
            batch = []
            batch_bytes = 0
        batch.extend(group_paths)
        batch_bytes += group_bytes
    if batch:
        batches.append(tuple(batch))
    return tuple(batches), tuple(omitted_paths)


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
                raise SnapshotCaptureError(
                    "a repository root is not a reviewable file path"
                ) from None
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
        return is_snapshot_path_excluded(path, self.excluded_paths)

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
        digest = hashlib.sha256()
        index_identity = (None, None)
        if event is ReviewEvent.WORKTREE_IDLE:
            index_identity = self._git_object_identity(":", path)
            digest.update(
                f"index:{index_identity[0]}:{index_identity[1]}\0".encode("ascii")
            )
        try:
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
        if event is ReviewEvent.WORKTREE_IDLE:
            if object_id is None and index_identity[1] is None:
                return None
            digest.update(f"base:{mode}:{object_id}\0".encode("ascii"))
            return "sha256:" + digest.hexdigest()
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
    ) -> Optional[list[tuple[str, str, tuple[str, ...]]]]:
        def discover_stage(
            stage: str, *, cached: bool, compare_base: bool
        ) -> Optional[list[tuple[str, str, tuple[str, ...]]]]:
            args = [
                *self._diff_filter_overrides(),
                "--literal-pathspecs",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
            ]
            if cached:
                args.append("--cached")
            args.extend(
                [
                    "--name-status",
                    "-z",
                    "--find-copies",
                    "--find-copies-harder",
                    "-l0",
                ]
            )
            if compare_base:
                args.append(base_revision)
            args.append("--")
            output = _run_git_bounded(
                self.repository_root,
                *args,
                max_output_bytes=self.max_path_discovery_bytes,
            )
            if output is None:
                return None
            return parse_stage(stage, _decode_z_paths(output))

        def parse_stage(
            stage: str, fields: list[str]
        ) -> list[tuple[str, str, tuple[str, ...]]]:
            path_groups = []
            index = 0
            while index < len(fields):
                status = fields[index]
                index += 1
                path_count = 2 if status.startswith(("R", "C")) else 1
                paths = tuple(fields[index:index + path_count])
                path_groups.append((stage, status, paths))
                index += path_count
            return path_groups

        path_groups = []
        stages = (
            (("index", True, True), ("worktree", False, False))
            if event is ReviewEvent.WORKTREE_IDLE
            else (("index", True, True),)
            if event is ReviewEvent.PRE_COMMIT
            else (("combined", False, True),)
        )
        for stage, cached, compare_base in stages:
            discovered = discover_stage(
                stage, cached=cached, compare_base=compare_base
            )
            if discovered is None:
                return None
            path_groups.extend(discovered)
        if focus_path:
            return [
                group
                for group in path_groups
                if (
                    focus_path == group[2][-1]
                    if group[1].startswith("C")
                    else focus_path in group[2]
                )
            ]
        return path_groups

    def _tracked_filtered_paths(
        self,
        event: ReviewEvent,
        base_revision: str,
        paths: Sequence[str],
    ) -> Optional[set[str]]:
        if not paths:
            return set()
        tracked = b"".join(
            path.encode("utf-8", errors="surrogateescape") + b"\0"
            for path in paths
        )
        attribute_args = ["--literal-pathspecs", "check-attr"]
        if event is ReviewEvent.PRE_COMMIT:
            attribute_args.append("--cached")
        attribute_args.extend(["-z", "--stdin", "filter"])
        current_attributes = _run_git_bounded(
            self.repository_root,
            *attribute_args,
            max_output_bytes=self.max_path_discovery_bytes,
            stdin_bytes=tracked,
        )
        index_attributes = (
            _run_git_bounded(
                self.repository_root,
                "--literal-pathspecs",
                "check-attr",
                "--cached",
                "-z",
                "--stdin",
                "filter",
                max_output_bytes=self.max_path_discovery_bytes,
                stdin_bytes=tracked,
            )
            if event is not ReviewEvent.PRE_COMMIT
            else b""
        )
        base_attributes = _run_git_bounded(
            self.repository_root,
            "--literal-pathspecs",
            "check-attr",
            f"--source={base_revision}",
            "-z",
            "--stdin",
            "filter",
            max_output_bytes=self.max_path_discovery_bytes,
            stdin_bytes=tracked,
        )
        if (
            current_attributes is None
            or index_attributes is None
            or base_attributes is None
        ):
            return None
        filtered_paths = set()
        for attributes in (current_attributes, index_attributes, base_attributes):
            fields = _decode_z_paths(attributes)
            if len(fields) % 3:
                raise SnapshotCaptureError("git check-attr returned malformed path data")
            filtered_paths.update(
                fields[index]
                for index in range(0, len(fields), 3)
                if fields[index + 2].lower() not in {"unspecified", "unset"}
            )
        return filtered_paths

    def _tracked_attribute_candidate_paths(
        self, focus_path: Optional[str] = None
    ) -> Optional[list[str]]:
        args = ["--literal-pathspecs", "ls-files", "-z", "--"]
        if focus_path:
            args.append(focus_path)
        output = _run_git_bounded(
            self.repository_root,
            *args,
            max_output_bytes=self.max_path_discovery_bytes,
        )
        return None if output is None else _decode_z_paths(output)

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

    def _non_regular_source_paths(self, max_output_bytes: int) -> Optional[list[str]]:
        """Discover non-regular entries within separate entry and byte budgets."""
        pending = [self.repository_root]
        paths = []
        output_bytes = 0
        visited_entries = 0
        max_entries = self.max_path_discovery_bytes
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = []
                    for entry in iterator:
                        visited_entries += 1
                        if visited_entries > max_entries:
                            return None
                        entries.append(entry)
                    entries.sort(key=lambda entry: entry.name)
            except OSError:
                return None
            for entry in entries:
                lexical_path = Path(entry.path)
                try:
                    relative = lexical_path.relative_to(self.repository_root).as_posix()
                except ValueError:
                    return None
                if directory == self.repository_root and entry.name == ".git":
                    continue
                try:
                    is_junction = getattr(os.path, "isjunction", None)
                    if is_junction is not None and is_junction(lexical_path):
                        # NTFS junctions can report as directories even without
                        # symlink following. Their targets may escape or cycle
                        # back into the repository, so an inventory containing
                        # one is incomplete unless junctions are supported.
                        return None
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError:
                    return None
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(lexical_path)
                    continue
                if stat.S_ISREG(entry_stat.st_mode):
                    continue
                path_bytes = len(
                    relative.encode("utf-8", errors="surrogateescape")
                ) + 1
                if output_bytes + path_bytes > max_output_bytes:
                    return None
                output_bytes += path_bytes
                paths.append(relative)
        return paths

    def _untracked_source_paths(self) -> Optional[list[str]]:
        """Discover bounded untracked inputs, including Git-ignored sources."""
        outputs = []
        remaining_bytes = self.max_path_discovery_bytes
        for ignored_args in (
            ("--exclude-standard",),
            ("--ignored", "--exclude-standard"),
        ):
            output = _run_git_bounded(
                self.repository_root,
                "--literal-pathspecs",
                "ls-files",
                "--others",
                *ignored_args,
                "-z",
                "--",
                max_output_bytes=remaining_bytes,
            )
            if output is None:
                return None
            outputs.append(output)
            remaining_bytes -= len(output)
        non_regular_paths = self._non_regular_source_paths(remaining_bytes)
        if non_regular_paths is None:
            return None
        return list(
            dict.fromkeys(
                (*_decode_z_paths(b"".join(outputs)), *non_regular_paths)
            )
        )

    def _unsafe_copy_sources(
        self,
        event: ReviewEvent,
        base_revision: str,
        *,
        current_candidate_paths: Sequence[str] = (),
        index_candidate_paths: Sequence[str] = (),
        current_patch_groups: Optional[Mapping[str, Sequence[str]]] = None,
        index_patch_groups: Optional[Mapping[str, Sequence[str]]] = None,
        current_source_paths: Sequence[str] = (),
    ) -> Optional[dict[str, tuple[tuple[str, str], ...]]]:
        """Find candidate changes copied from unsafe tracked or untracked files."""
        if not current_candidate_paths and not index_candidate_paths:
            return {}
        base_entries = _run_git_bounded(
            self.repository_root,
            "--literal-pathspecs",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            base_revision,
            max_output_bytes=self.max_path_discovery_bytes,
        )
        index_entries = _run_git_bounded(
            self.repository_root,
            "--literal-pathspecs",
            "ls-files",
            "--stage",
            "-z",
            max_output_bytes=self.max_path_discovery_bytes,
        )
        if base_entries is None or index_entries is None:
            return None

        source_oids: dict[str, set[str]] = {}

        def add_entries(output: bytes, *, index: bool) -> None:
            for entry in _decode_z_paths(output):
                metadata, separator, path = entry.partition("\t")
                fields = metadata.split()
                object_index = 1 if index else 2
                if not separator or len(fields) <= object_index:
                    raise SnapshotCaptureError(
                        "git object discovery returned malformed path data"
                    )
                source_oids.setdefault(path, set()).add(fields[object_index])

        add_entries(base_entries, index=False)
        add_entries(index_entries, index=True)
        source_paths = tuple(dict.fromkeys((*source_oids, *current_source_paths)))
        source_filtered = self._tracked_filtered_paths(
            event, base_revision, source_paths
        )
        if source_filtered is None:
            return None
        unsafe_reasons = {
            path: (
                "content_filter_unsupported"
                if path in source_filtered
                else "excluded"
            )
            for path in source_paths
            if (
                path in source_filtered
                or self._is_excluded(path)
                or self._is_ignored(path)
            )
        }
        if not unsafe_reasons:
            return {}

        candidate_variants: list[tuple[str, tuple[bytes, ...], str, bool]] = []
        candidate_keys: set[tuple[str, str]] = set()
        current_patch_groups = current_patch_groups or {}
        index_patch_groups = index_patch_groups or {}
        remaining_similarity_bytes = self.max_snapshot_bytes

        def add_candidate(
            path: str,
            *,
            current: bool,
            patch_paths: Optional[Sequence[str]],
            diff_stage: str,
        ) -> bool:
            nonlocal remaining_similarity_bytes
            if self._is_excluded(path) or self._is_ignored(path):
                return True
            candidate = self.repository_root / path
            candidate_oid = None
            if current:
                try:
                    candidate_stat = candidate.lstat()
                except OSError:
                    return False
                if (
                    stat.S_ISLNK(candidate_stat.st_mode)
                    or not stat.S_ISREG(candidate_stat.st_mode)
                    or candidate_stat.st_size > self.max_file_bytes
                ):
                    return True
            else:
                mode, candidate_oid = self._git_object_identity(":", path)
                if mode not in {"100644", "100755"} or candidate_oid is None:
                    return True
            patch_scoped = patch_paths is not None
            if patch_scoped:
                patch = self._capture_diff(
                    event,
                    base_revision,
                    patch_paths,
                    max_output_bytes=remaining_similarity_bytes,
                    diff_stage=diff_stage,
                )
                if patch is None:
                    return False
                regions = _patch_model_visible_regions(patch)
                candidate_size = sum(len(region) for region in regions)
                candidate_digest = hashlib.sha256()
                for region in regions:
                    candidate_digest.update(len(region).to_bytes(8, "big"))
                    candidate_digest.update(region)
                candidate_oid = f"patch:{candidate_digest.hexdigest()}"
            elif current:
                content = _read_stable_regular_file(candidate, self.max_file_bytes)
                if content is None:
                    return False
                try:
                    object_id = _run_git_bounded(
                        self.repository_root,
                        "hash-object",
                        "--stdin",
                        max_output_bytes=128,
                        stdin_bytes=content,
                    )
                    if object_id is None:
                        return False
                    candidate_oid = object_id.decode("ascii").strip()
                except (SnapshotCaptureError, UnicodeDecodeError):
                    return False
                regions = (content,)
                candidate_size = len(content)
            else:
                try:
                    candidate_size = int(
                        _run_git(
                            self.repository_root, "cat-file", "-s", candidate_oid
                        ).strip()
                    )
                except (SnapshotCaptureError, ValueError):
                    return False
                if candidate_size > self.max_file_bytes:
                    return True
                content = _run_git_bounded(
                    self.repository_root,
                    "cat-file",
                    "blob",
                    candidate_oid,
                    max_output_bytes=candidate_size,
                )
                if content is None or len(content) != candidate_size:
                    return False
                regions = (content,)
            if not regions:
                return True
            if patch_scoped and candidate_size > self.max_file_bytes:
                return False
            if candidate_size > remaining_similarity_bytes:
                return False
            assert candidate_oid is not None
            if (path, candidate_oid) in candidate_keys:
                return True
            remaining_similarity_bytes -= candidate_size
            candidate_keys.add((path, candidate_oid))
            candidate_variants.append(
                (path, regions, candidate_oid, patch_scoped)
            )
            return True

        for path in dict.fromkeys(index_candidate_paths):
            if not add_candidate(
                path,
                current=False,
                patch_paths=index_patch_groups.get(path),
                diff_stage="index",
            ):
                return None

        for path in dict.fromkeys(current_candidate_paths):
            if not add_candidate(
                path,
                current=True,
                patch_paths=current_patch_groups.get(path),
                diff_stage=(
                    "worktree"
                    if event is ReviewEvent.WORKTREE_IDLE
                    else "combined"
                ),
            ):
                return None

        if not candidate_variants:
            return {}

        object_contents: dict[str, bytes] = {}
        source_contents: dict[str, dict[str, bytes]] = {
            path: {} for path in unsafe_reasons
        }
        for path in unsafe_reasons:
            for object_id in source_oids.get(path, ()):
                try:
                    object_size = int(
                        _run_git(
                            self.repository_root, "cat-file", "-s", object_id
                        ).strip()
                    )
                except (SnapshotCaptureError, ValueError):
                    return None
                if object_size > self.max_file_bytes:
                    return None
                if object_id not in object_contents:
                    if object_size > remaining_similarity_bytes:
                        return None
                    content = _run_git_bounded(
                        self.repository_root,
                        "cat-file",
                        "blob",
                        object_id,
                        max_output_bytes=object_size,
                    )
                    if content is None or len(content) != object_size:
                        return None
                    remaining_similarity_bytes -= len(content)
                    object_contents[object_id] = content
                source_contents[path][object_id] = object_contents[object_id]

        deferred_operational_aliases: list[tuple[str, os.stat_result]] = []
        for path in unsafe_reasons:
            candidate = self.repository_root / path
            try:
                source_stat = candidate.lstat()
            except FileNotFoundError:
                if path in current_source_paths:
                    return None
                continue
            except OSError:
                return None
            if stat.S_ISLNK(source_stat.st_mode):
                if path in source_filtered or self._is_excluded(path):
                    return None
                deferred_operational_aliases.append((path, source_stat))
                continue
            if not stat.S_ISREG(source_stat.st_mode):
                # Do not follow an unsafe source to recover comparison bytes.
                # Without a complete current inventory, accepting any candidate
                # could expose content copied from the skipped object.
                return None
            if (
                source_stat.st_size > self.max_file_bytes
                or source_stat.st_size > remaining_similarity_bytes
            ):
                return None
            content = _read_stable_regular_file(candidate, self.max_file_bytes)
            if content is None:
                return None
            remaining_similarity_bytes -= len(content)
            try:
                current_oid_output = _run_git_bounded(
                    self.repository_root,
                    "hash-object",
                    "--stdin",
                    max_output_bytes=128,
                    stdin_bytes=content,
                )
                if current_oid_output is None:
                    return None
                current_oid = current_oid_output.decode("ascii").strip()
            except (SnapshotCaptureError, UnicodeDecodeError):
                return None
            source_oids.setdefault(path, set()).add(current_oid)
            source_contents[path][current_oid] = content

        for path, source_stat in deferred_operational_aliases:
            candidate = self.repository_root / path
            try:
                target = os.readlink(candidate)
                current_stat = candidate.lstat()
            except OSError:
                return None
            source_identity = (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_mode,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_ctime_ns,
            )
            current_identity = (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_mode,
                current_stat.st_size,
                current_stat.st_mtime_ns,
                current_stat.st_ctime_ns,
            )
            if source_identity != current_identity:
                return None
            lexical_target = Path(
                os.path.abspath(candidate.parent / target)
            )
            try:
                target_path = lexical_target.relative_to(
                    self.repository_root
                ).as_posix()
            except ValueError:
                return None
            if target_path not in unsafe_reasons or not source_contents.get(target_path):
                return None

        large_signatures: list[tuple[str, str, frozenset[int]]] = []
        small_sources: list[tuple[str, str, bytes]] = []
        unsafe_lines: dict[bytes, set[tuple[str, str]]] = {}
        total_exact_lines = 0
        for source, variants in source_contents.items():
            reason = unsafe_reasons[source]
            for content in variants.values():
                for line in content.split(b"\n"):
                    normalized_line = line.strip()
                    if not normalized_line:
                        continue
                    line_sources = unsafe_lines.setdefault(normalized_line, set())
                    source_identity = (source, reason)
                    if source_identity not in line_sources:
                        total_exact_lines += 1
                        if total_exact_lines > _UNSAFE_COPY_MAX_TOTAL_EXACT_LINES:
                            return None
                        line_sources.add(source_identity)
                if len(content) < 4 * _UNSAFE_COPY_PROBE_WINDOW:
                    small_sources.append((source, reason, content))
                if len(content) < _UNSAFE_COPY_PROBE_WINDOW:
                    continue
                probes = _copy_similarity_probes(content)
                if probes:
                    large_signatures.append((source, reason, probes))

        probe_index: dict[int, set[int]] = {}
        total_probes = 0
        for signature_index, (_, _, probes) in enumerate(large_signatures):
            total_probes += len(probes)
            if total_probes > _UNSAFE_COPY_MAX_TOTAL_PROBES:
                return None
            for probe in probes:
                probe_index.setdefault(probe, set()).add(signature_index)

        matched_sources: dict[str, set[tuple[str, str]]] = {}
        small_comparisons = 0
        for destination, regions, object_id, patch_scoped in candidate_variants:
            sources = set()
            if not patch_scoped:
                sources.update(
                    (source, unsafe_reasons[source])
                    for source, object_ids in source_oids.items()
                    if source in unsafe_reasons and object_id in object_ids
                )
            matched_probes: dict[int, set[int]] = {}
            for region in regions:
                if patch_scoped:
                    for line in region.split(b"\n"):
                        normalized_line = line.strip()
                        if normalized_line:
                            sources.update(unsafe_lines.get(normalized_line, ()))
                for probe in _rolling_copy_hashes(region):
                    for signature_index in probe_index.get(probe, ()):
                        matched_probes.setdefault(signature_index, set()).add(probe)
            for signature_index, probes in matched_probes.items():
                source, reason, expected = large_signatures[signature_index]
                if len(probes) / len(expected) >= _UNSAFE_COPY_SIMILARITY_THRESHOLD:
                    sources.add((source, reason))
            for source, reason, source_content in small_sources:
                if not source_content:
                    continue
                for region in regions:
                    small_comparisons += 1
                    if small_comparisons > _UNSAFE_COPY_MAX_SMALL_COMPARISONS:
                        return None
                    if source_content in region or (
                        len(region) <= 2 * len(source_content)
                        and SequenceMatcher(
                            None, source_content, region, autojunk=False
                        ).ratio() >= _UNSAFE_COPY_SMALL_SIMILARITY_THRESHOLD
                    ):
                        sources.add((source, reason))
                        break
            if sources:
                matched_sources.setdefault(destination, set()).update(sources)
        return {
            destination: tuple(sorted(sources))
            for destination, sources in matched_sources.items()
        }

    def _capture_diff(
        self,
        event: ReviewEvent,
        base_revision: str,
        paths: Sequence[str],
        *,
        max_output_bytes: Optional[int] = None,
        diff_stage: str = "combined",
    ) -> Optional[str]:
        if not paths:
            return ""
        args = [
            *self._diff_filter_overrides(),
            "-c", "core.quotePath=false",
            "--literal-pathspecs", "diff", "--no-color", "--src-prefix=a/", "--dst-prefix=b/",
            "--no-ext-diff", "--no-textconv", "--find-renames", "--find-copies",
            "--find-copies-harder", "-l0",
        ]
        if event is ReviewEvent.PRE_COMMIT or diff_stage == "index":
            args.append("--cached")
        if event is ReviewEvent.WORKTREE_IDLE and diff_stage == "worktree":
            args.extend(["--", *paths])
        else:
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
        validation_tracked: list[str] = []
        selected_tracked_groups: list[tuple[str, tuple[str, ...]]] = []
        selected_untracked: list[str] = []
        coverage: list[CoverageIssue] = []

        def add_coverage(path: Optional[str], reason: str) -> None:
            fingerprint = self._path_fingerprint(event, path, base_revision) if path else None
            coverage.append(CoverageIssue(path=path, reason=reason, fingerprint=fingerprint))

        tracked_groups = self._tracked_path_groups(
            event, base_revision, normalized_focus
        )
        discovery_overflow = tracked_groups is None
        if tracked_groups is None:
            add_coverage(None, "tracked_path_discovery_budget")
            tracked_groups = []
        changed_group_paths = tuple(
            dict.fromkeys(path for _, _, group in tracked_groups for path in group)
        )
        attribute_candidate_paths = list(changed_group_paths)
        untracked = (
            []
            if event is ReviewEvent.PRE_COMMIT
            else self._untracked_paths(normalized_focus)
        )
        if untracked is None:
            add_coverage(None, "untracked_path_discovery_budget")
            untracked = []
            discovery_overflow = True
        untracked_sources = self._untracked_source_paths()
        if untracked_sources is None:
            add_coverage(None, "untracked_source_discovery_budget")
            untracked_sources = []
            discovery_overflow = True
        index_copy_candidates = tuple(
            dict.fromkeys(
                group[-1]
                for stage, status, group in tracked_groups
                if stage == "index" and not status.startswith("D")
            )
        )
        index_patch_groups = {
            group[-1]: group
            for stage, status, group in tracked_groups
            if stage == "index" and _status_uses_grouped_patch(status)
        }
        current_copy_candidates = tuple(
            dict.fromkeys(
                (
                    *untracked,
                    *(
                        group[-1]
                        for stage, status, group in tracked_groups
                        if stage in {"combined", "worktree"}
                        and not status.startswith("D")
                    ),
                )
            )
        )
        current_patch_groups = {
            group[-1]: group
            for stage, status, group in tracked_groups
            if stage in {"combined", "worktree"}
            and _status_uses_grouped_patch(status)
        }
        unsafe_copy_sources = self._unsafe_copy_sources(
            event,
            base_revision,
            current_candidate_paths=current_copy_candidates,
            index_candidate_paths=index_copy_candidates,
            current_patch_groups=current_patch_groups,
            index_patch_groups=index_patch_groups,
            current_source_paths=untracked_sources,
        )
        if unsafe_copy_sources is None:
            add_coverage(None, "copy_source_discovery_budget")
            unsafe_copy_sources = {}
            discovery_overflow = True
        else:
            covered = {(issue.path, issue.reason) for issue in coverage}
            for destination, sources in sorted(unsafe_copy_sources.items()):
                for source, reason in sources:
                    if (source, reason) not in covered:
                        add_coverage(source, reason)
                        covered.add((source, reason))
                add_coverage(destination, "rename_group_omitted")
        if event is not ReviewEvent.PRE_COMMIT:
            active_attribute_paths = self._tracked_attribute_candidate_paths(
                normalized_focus
            )
            discovery_overflow = discovery_overflow or active_attribute_paths is None
            if active_attribute_paths is None:
                add_coverage(None, "filtered_path_discovery_budget")
            else:
                attribute_candidate_paths = list(
                    dict.fromkeys(
                        (*active_attribute_paths, *changed_group_paths, *untracked)
                    )
                )
        filtered_paths = self._tracked_filtered_paths(
            event, base_revision, attribute_candidate_paths
        )
        discovery_overflow = discovery_overflow or filtered_paths is None
        if filtered_paths is None:
            if not any(
                issue.reason == "filtered_path_discovery_budget"
                for issue in coverage
            ):
                add_coverage(None, "filtered_path_discovery_budget")
            filtered_paths = set()
        for filtered_path in sorted(filtered_paths):
            add_coverage(filtered_path, "content_filter_unsupported")
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

        initial_discovery_state = (
            tuple(tracked_groups),
            tuple(sorted(filtered_paths)),
            tuple(untracked),
            tuple(untracked_sources),
            tuple(sorted(unsafe_copy_sources.items())),
        )

        def current_discovery_state():
            current_groups = self._tracked_path_groups(
                event, base_revision, normalized_focus
            )
            if current_groups is None:
                return None
            current_group_paths = tuple(
                dict.fromkeys(
                    path for _, _, group in current_groups for path in group
                )
            )
            current_attribute_paths = list(current_group_paths)
            current_untracked = (
                []
                if event is ReviewEvent.PRE_COMMIT
                else self._untracked_paths(normalized_focus)
            )
            if current_untracked is None:
                return None
            current_untracked_sources = self._untracked_source_paths()
            if current_untracked_sources is None:
                return None
            current_index_candidates = tuple(
                dict.fromkeys(
                    group[-1]
                    for stage, status, group in current_groups
                    if stage == "index" and not status.startswith("D")
                )
            )
            current_index_patch_groups = {
                group[-1]: group
                for stage, status, group in current_groups
                if stage == "index" and _status_uses_grouped_patch(status)
            }
            current_candidates = tuple(
                dict.fromkeys(
                    (
                        *current_untracked,
                        *(
                            group[-1]
                            for stage, status, group in current_groups
                            if stage in {"combined", "worktree"}
                            and not status.startswith("D")
                        ),
                    )
                )
            )
            current_patch_groups = {
                group[-1]: group
                for stage, status, group in current_groups
                if stage in {"combined", "worktree"}
                and _status_uses_grouped_patch(status)
            }
            current_unsafe_sources = self._unsafe_copy_sources(
                event,
                base_revision,
                current_candidate_paths=current_candidates,
                index_candidate_paths=current_index_candidates,
                current_patch_groups=current_patch_groups,
                index_patch_groups=current_index_patch_groups,
                current_source_paths=current_untracked_sources,
            )
            if current_unsafe_sources is None:
                return None
            if event is not ReviewEvent.PRE_COMMIT:
                active_paths = self._tracked_attribute_candidate_paths(
                    normalized_focus
                )
                if active_paths is None:
                    return None
                current_attribute_paths = list(
                    dict.fromkeys(
                        (*active_paths, *current_group_paths, *current_untracked)
                    )
                )
            current_filtered = self._tracked_filtered_paths(
                event, base_revision, current_attribute_paths
            )
            if current_filtered is None:
                return None
            return (
                tuple(current_groups),
                tuple(sorted(current_filtered)),
                tuple(current_untracked),
                tuple(current_untracked_sources),
                tuple(sorted(current_unsafe_sources.items())),
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
            if normalized in filtered_paths:
                return None
            if normalized in unsafe_copy_sources:
                return None
            if self._has_content_filter(event, normalized):
                add_coverage(normalized, "content_filter_unsupported")
                return None
            file_issue = (
                self._inspect_index_file(normalized, base_revision)
                if event is ReviewEvent.PRE_COMMIT
                else next(
                    (
                        issue
                        for issue in (
                            self._inspect_index_file(normalized, base_revision),
                            self._inspect_current_file(normalized, base_revision),
                        )
                        if issue
                    ),
                    None,
                )
                if event is ReviewEvent.WORKTREE_IDLE
                else self._inspect_current_file(normalized, base_revision)
            )
            if file_issue:
                add_coverage(normalized, file_issue)
                return None
            return normalized

        # A rename/copy is one security unit. If either side is unavailable or
        # excluded, selecting the other side alone can expose the full source.
        for stage, _status, group in tracked_groups:
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
                validation_tracked.extend(selected_group)
                captured_group = selected_group
                selected_tracked_groups.append((stage, captured_group))
                selected_tracked.extend(captured_group)
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
            for stage in dict.fromkeys(
                stage_name for stage_name, _ in selected_tracked_groups
            ):
                stage_groups = [
                    group
                    for stage_name, group in selected_tracked_groups
                    if stage_name == stage
                ]
                tracked_batches, oversized_paths = _batch_path_groups(stage_groups)
                omitted_paths.update(oversized_paths)
                for tracked_batch in tracked_batches:
                    part = None if remaining_bytes <= 0 else self._capture_diff(
                        event,
                        base_revision,
                        tracked_batch,
                        max_output_bytes=remaining_bytes,
                        diff_stage=stage,
                    )
                    if part is None:
                        omitted_paths.update(tracked_batch)
                    else:
                        part_size = len(part.encode("utf-8", errors="surrogateescape"))
                        if part_size > remaining_bytes:
                            omitted_paths.update(tracked_batch)
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
                for selected_path in (*validation_tracked, *selected_untracked)
            )
            if revalidated:
                verified_diff, verified_omissions = capture_selected()
                verified_discovery_state = current_discovery_state()
                if verified_discovery_state != initial_discovery_state:
                    add_coverage(None, "content_changed_during_capture")
                    revalidated = False
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
            validation_tracked.clear()
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
        deletion_only_paths = set()
        for item in parsed_files:
            if to_hunk_only_patch(item.patch).strip():
                if _patch_has_only_deletions(item.patch):
                    for path in (
                        getattr(item, "filename", None),
                        getattr(item, "old_filename", None),
                    ):
                        if path and path not in deletion_only_paths:
                            deletion_only_paths.add(path)
                            add_coverage(path, "deleted_file_unsupported")
                    continue
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
        for missing_path in sorted(
            expected_paths - parsed_paths - metadata_only_paths - deletion_only_paths
        ):
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

    changed_lines: dict[str, set[int]] = {}
    for item in parse_plain_diff(snapshot.diff):
        filename = getattr(item, "filename", None)
        if not filename or filename not in snapshot.changed_paths:
            continue
        lines = changed_lines.setdefault(filename, set())
        new_line = None
        for patch_line in iter_git_patch_lines(item.patch):
            hunk_match = RE_HUNK_HEADER.match(patch_line)
            if hunk_match:
                new_line = int(hunk_match.group(3))
                continue
            if new_line is None or patch_line.startswith("\\"):
                continue
            if patch_line.startswith("-"):
                continue
            if patch_line.startswith("+"):
                lines.add(new_line)
                new_line += 1
            elif patch_line.startswith(" "):
                new_line += 1

    for finding in findings:
        if not isinstance(finding, Mapping):
            return False
        filename = str(finding.get("relevant_file") or "").strip()
        try:
            start_line = int(str(finding.get("start_line", "")).strip())
            end_line = int(str(finding.get("end_line", "")).strip())
        except ValueError:
            return False
        allowed_lines = changed_lines.get(filename) or changed_lines.get(
            filename.lstrip("/"), set()
        )
        if (
            not allowed_lines
            or start_line < 1
            or end_line < start_line
            or end_line > _MAX_FINDING_LINE
            or not any(start_line <= line <= end_line for line in allowed_lines)
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
