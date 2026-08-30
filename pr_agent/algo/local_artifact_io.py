"""Descriptor-relative reads for local artifacts that must not traverse symlinks."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from pr_agent.algo.checkpoint_evaluation import EvaluationValidationError


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _require_safe_openat_support() -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise EvaluationValidationError(
            "this platform cannot safely validate symlink-free local artifact paths"
        )


def _open_parent_directory(path: Path, label: str) -> tuple[int, str]:
    _require_safe_openat_support()
    if not path.name:
        raise EvaluationValidationError(f"{label} must name a file")
    if ".." in path.parts:
        raise EvaluationValidationError(f"{label} path cannot contain parent traversal")

    if path.is_absolute():
        start = path.anchor
        parts = path.parts[1:]
    else:
        start = "."
        parts = path.parts
    if not parts:
        raise EvaluationValidationError(f"{label} must name a file")

    try:
        directory_fd = os.open(start, _directory_open_flags())
    except OSError as exc:
        raise EvaluationValidationError(f"cannot open {label} path root") from exc
    try:
        for component in parts[:-1]:
            child_fd: int | None = None
            try:
                child_fd = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=directory_fd,
                )
                child_metadata = os.fstat(child_fd)
            except OSError as exc:
                if child_fd is not None:
                    os.close(child_fd)
                raise EvaluationValidationError(
                    f"{label} parent components must be real directories, not symlinks"
                ) from exc
            if not stat.S_ISDIR(child_metadata.st_mode):
                os.close(child_fd)
                raise EvaluationValidationError(
                    f"{label} parent components must be real directories"
                )
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd, parts[-1]
    except BaseException:
        os.close(directory_fd)
        raise


def read_regular_file_without_symlinks(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read a stable bounded file without following any supplied path component."""

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise EvaluationValidationError(f"{label} byte limit must be a positive integer")
    directory_fd, filename = _open_parent_directory(Path(path), label)
    descriptor: int | None = None
    try:
        try:
            lexical_metadata = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise EvaluationValidationError(f"cannot inspect {label}") from exc
        if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISREG(lexical_metadata.st_mode):
            raise EvaluationValidationError(f"{label} must be a regular file, not a symlink")
        if lexical_metadata.st_size > max_bytes:
            raise EvaluationValidationError(
                f"{label} exceeds the {max_bytes}-byte validation limit"
            )

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise EvaluationValidationError(f"cannot open {label}") from exc
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise EvaluationValidationError(f"{label} must remain a regular file")
        if (
            lexical_metadata.st_dev,
            lexical_metadata.st_ino,
        ) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise EvaluationValidationError(f"{label} changed before it was opened")

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise EvaluationValidationError(
                f"{label} exceeds the {max_bytes}-byte validation limit"
            )
        final_metadata = os.fstat(descriptor)
        if (
            opened_metadata.st_size,
            opened_metadata.st_mtime_ns,
            opened_metadata.st_ctime_ns,
        ) != (
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ):
            raise EvaluationValidationError(f"{label} changed while it was read")
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
