"""Verify that an upstream-sync pull request has an allowed immutable topology.

Three shapes are accepted, from a fixed upstream ``pin`` that the branch name,
title and body all identify:

* ``raw-pin``: the PR head is the pin itself.
* ``resolved-merge``: one merge whose parents are exactly the pin and the
  declared fork integration baseline, i.e. a reviewed conflict resolution.
* ``base-refreshed-merge``: that resolution followed by one or more merges of a
  newer fork ``main`` commit, each with exactly two parents -- the previous
  candidate and the fork commit being brought in. This is what "update branch"
  produces when ``main`` advances after the resolution, and it keeps the pin
  immutable while letting the PR stay current with its base.

Anything else -- a plain commit on top, a merge with a parent that is not on
the fork base, a refresh that brings in an older base than the last one -- is
rejected. Topology is all this can check; the content of a resolution is what
the human review is for.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BRANCH_PATTERN = re.compile(r"^sync/upstream-[0-9]{8}-([0-9a-f]{8})$")
PIN_PATTERN = re.compile(r"^Pinned upstream sync: `The-PR-Agent/pr-agent@([0-9a-f]{40})`$", re.MULTILINE)
# Each refresh merge costs a few git calls. A sync PR that has been updated more
# times than this is not a legitimate shape and is refused before walking further.
MAX_BASE_REFRESH_MERGES = 32


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


def _parents(repository: Path, sha: str) -> list[str]:
    return _git(repository, "show", "-s", "--format=%P", sha).stdout.strip().split()


def _fork_parents_from_head_to_pin(
    repository: Path, head: str, pin: str, base_sha: str
) -> list[str]:
    """Walk merge links from ``head`` down to ``pin``; return each link's fork-side parent.

    Every link must be a two-parent merge. Exactly one parent must be a fork commit
    (an ancestor of the PR base); the other is the link below. A chain link can
    never itself be an ancestor of the base, because it contains the pin and the
    pin is verified not to be in the base, so the two parents are distinguishable.
    Returned head-most first.
    """
    fork_parents: list[str] = []
    current = head
    while current != pin:
        if len(fork_parents) >= MAX_BASE_REFRESH_MERGES:
            raise ProvenanceError("Sync candidate has more base refresh merges than are allowed")
        parents = _parents(repository, current)
        if len(parents) != 2 or len(set(parents)) != 2:
            raise ProvenanceError("Resolved sync candidate must have exactly two parents")
        on_fork_base = [parent for parent in parents if _is_ancestor(repository, parent, base_sha)]
        if len(on_fork_base) != 1:
            raise ProvenanceError("Sync candidate merge must have exactly one parent on the fork base")
        fork_parents.append(on_fork_base[0])
        (current,) = (parent for parent in parents if parent != on_fork_base[0])
    return fork_parents


def verify_upstream_provenance(metadata: PullRequestMetadata, repository: Path) -> str:
    """Validate PR metadata and return ``raw-pin``, ``resolved-merge`` or ``base-refreshed-merge``."""

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

    # Chronological order: the resolution merge first, then each base refresh.
    fork_parents = list(reversed(_fork_parents_from_head_to_pin(repository, metadata.head_sha, pin, metadata.base_sha)))
    if fork_parents[0] != baseline:
        raise ProvenanceError("Resolved sync parents must be exactly the pinned upstream commit and fork baseline")
    for previous, refreshed in zip(fork_parents, fork_parents[1:], strict=False):
        # Each refresh must move forward along the fork base. The same commit
        # again, or an older one, is not a base update and has no reason to exist.
        if refreshed == previous or not _is_ancestor(repository, previous, refreshed):
            raise ProvenanceError("Base refresh merge must bring in a newer fork base than the previous merge")
    return "resolved-merge" if len(fork_parents) == 1 else "base-refreshed-merge"


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
