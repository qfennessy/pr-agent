"""Verify that an upstream-sync pull request has an allowed immutable topology."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BRANCH_PATTERN = re.compile(r"^sync/upstream-[0-9]{8}-([0-9a-f]{8})$")
PIN_PATTERN = re.compile(r"^Pinned upstream sync: `The-PR-Agent/pr-agent@([0-9a-f]{40})`$", re.MULTILINE)


class ProvenanceError(ValueError):
    """Raised when a sync candidate does not satisfy the provenance contract."""


@dataclass(frozen=True)
class PullRequestMetadata:
    head_ref: str
    head_repo: str
    head_sha: str
    base_repo: str
    base_sha: str
    title: str
    body: str
    upstream_main_sha: str
    expected_repository: str


def _require_sha(value: str, label: str) -> None:
    if not SHA_PATTERN.fullmatch(value):
        raise ProvenanceError(f"{label} is not a canonical lowercase 40-character commit SHA")


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ProvenanceError(detail)
    return result


def _require_commit(repository: Path, sha: str, label: str) -> None:
    result = _git(repository, "rev-parse", "--verify", f"{sha}^{{commit}}")
    if result.stdout.strip() != sha:
        raise ProvenanceError(f"{label} did not resolve to its declared commit")


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return _git(repository, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0


def verify_upstream_provenance(metadata: PullRequestMetadata, repository: Path) -> str:
    """Validate PR metadata and return ``raw-pin`` or ``resolved-merge``."""

    _require_sha(metadata.head_sha, "Pull request head")
    _require_sha(metadata.base_sha, "Pull request base")
    _require_sha(metadata.upstream_main_sha, "Upstream main")

    if metadata.head_repo != metadata.expected_repository:
        raise ProvenanceError(f"Sync branches must be owned by {metadata.expected_repository}")
    if metadata.base_repo != metadata.expected_repository:
        raise ProvenanceError(f"Sync pull requests must target {metadata.expected_repository}")

    branch_match = BRANCH_PATTERN.fullmatch(metadata.head_ref)
    if not branch_match:
        raise ProvenanceError("Sync branch must be named sync/upstream-YYYYMMDD-<8-char-upstream-sha>")

    pins = PIN_PATTERN.findall(metadata.body)
    if len(pins) != 1:
        raise ProvenanceError("Pull request body must contain exactly one canonical upstream pin")
    pin = pins[0]
    short_pin = pin[:8]
    if branch_match.group(1) != short_pin:
        raise ProvenanceError("Sync branch suffix does not match the pinned upstream commit")
    if metadata.title != f"sync: upstream @ {short_pin}":
        raise ProvenanceError("Sync pull request title does not match the pinned upstream commit")

    baseline_pattern = re.compile(
        rf"^Fork integration baseline: `{re.escape(metadata.expected_repository)}@([0-9a-f]{{40}})`$",
        re.MULTILINE,
    )
    baselines = baseline_pattern.findall(metadata.body)
    if len(baselines) != 1:
        raise ProvenanceError("Pull request body must contain exactly one canonical fork integration baseline")
    baseline = baselines[0]

    for sha, label in (
        (metadata.head_sha, "Pull request head"),
        (metadata.base_sha, "Pull request base"),
        (metadata.upstream_main_sha, "Upstream main"),
        (pin, "Pinned upstream commit"),
        (baseline, "Fork integration baseline"),
    ):
        _require_commit(repository, sha, label)

    if not _is_ancestor(repository, pin, metadata.upstream_main_sha):
        raise ProvenanceError("Pinned commit is not part of upstream main")
    if not _is_ancestor(repository, baseline, metadata.base_sha):
        raise ProvenanceError("Fork integration baseline is not an ancestor of the pull request base")
    if _is_ancestor(repository, pin, metadata.base_sha):
        raise ProvenanceError("Pinned upstream commit is already part of the pull request base")

    if metadata.head_sha == pin:
        return "raw-pin"

    parents = _git(repository, "show", "-s", "--format=%P", metadata.head_sha).stdout.strip().split()
    if len(parents) != 2:
        raise ProvenanceError("Resolved sync candidate must have exactly two parents")
    if len(set(parents)) != 2 or set(parents) != {pin, baseline}:
        raise ProvenanceError("Resolved sync parents must be exactly the pinned upstream commit and fork baseline")
    return "resolved-merge"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--head-repo", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--upstream-main-sha", required=True)
    parser.add_argument("--expected-repository", default="qfennessy/pr-agent")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata = PullRequestMetadata(
        head_ref=args.head_ref,
        head_repo=args.head_repo,
        head_sha=args.head_sha,
        base_repo=args.base_repo,
        base_sha=args.base_sha,
        title=args.title,
        body=args.body,
        upstream_main_sha=args.upstream_main_sha,
        expected_repository=args.expected_repository,
    )
    try:
        topology = verify_upstream_provenance(metadata, args.repository)
    except ProvenanceError as error:
        print(f"::error::{error}")
        return 1
    print(f"Verified immutable upstream provenance ({topology}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
