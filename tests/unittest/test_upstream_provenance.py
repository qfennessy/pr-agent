from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pr_agent.upstream_provenance import ProvenanceError, PullRequestMetadata, verify_upstream_provenance

EXPECTED_REPOSITORY = "qfennessy/pr-agent"


def _read_source_contract(path: str) -> str:
    source = Path(path)
    if not source.exists():
        pytest.skip("source-only workflow contract is not copied into the Docker test target")
    return source.read_text()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repository: Path, tree: str, message: str, *parents: str) -> str:
    args = ["commit-tree", tree]
    for parent in parents:
        args.extend(("-p", parent))
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        input=f"{message}\n",
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def provenance_graph(tmp_path: Path) -> dict[str, str | Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Provenance Test")
    (repository / "root.txt").write_text("root\n")
    _git(repository, "add", "root.txt")
    _git(repository, "commit", "-m", "root")
    root = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", f"{root}^{{tree}}")

    baseline = _commit(repository, tree, "fork baseline", root)
    base = _commit(repository, tree, "current fork base", baseline)
    pin = _commit(repository, tree, "upstream pin", root)
    upstream_main = _commit(repository, tree, "current upstream main", pin)
    foreign = _commit(repository, tree, "foreign history", root)
    stale_baseline = _commit(repository, tree, "stale fork-like history", root)
    resolved = _commit(repository, tree, "resolved merge", baseline, pin)
    reversed_resolved = _commit(repository, tree, "reversed resolved merge", pin, baseline)

    return {
        "repository": repository,
        "root": root,
        "tree": tree,
        "baseline": baseline,
        "base": base,
        "pin": pin,
        "upstream_main": upstream_main,
        "foreign": foreign,
        "stale_baseline": stale_baseline,
        "resolved": resolved,
        "reversed_resolved": reversed_resolved,
    }


def _body(pin: str, baseline: str) -> str:
    return (
        f"Pinned upstream sync: `The-PR-Agent/pr-agent@{pin}`\n"
        f"Fork integration baseline: `{EXPECTED_REPOSITORY}@{baseline}`\n"
    )


def _metadata(graph: dict[str, str | Path], *, head: str | None = None) -> PullRequestMetadata:
    pin = str(graph["pin"])
    baseline = str(graph["baseline"])
    return PullRequestMetadata(
        head_ref=f"sync/upstream-20260902-{pin[:8]}",
        head_repo=EXPECTED_REPOSITORY,
        head_sha=head or pin,
        base_repo=EXPECTED_REPOSITORY,
        base_sha=str(graph["base"]),
        title=f"sync: upstream @ {pin[:8]}",
        body=_body(pin, baseline),
        upstream_main_sha=str(graph["upstream_main"]),
        expected_repository=EXPECTED_REPOSITORY,
    )


def test_accepts_valid_raw_upstream_pin(provenance_graph: dict[str, str | Path]) -> None:
    assert verify_upstream_provenance(
        _metadata(provenance_graph), Path(provenance_graph["repository"])
    ) == "raw-pin"


@pytest.mark.parametrize("head_key", ["resolved", "reversed_resolved"])
def test_accepts_resolved_merge_with_either_parent_order(
    provenance_graph: dict[str, str | Path], head_key: str
) -> None:
    assert verify_upstream_provenance(
        _metadata(provenance_graph, head=str(provenance_graph[head_key])), Path(provenance_graph["repository"])
    ) == "resolved-merge"


def test_rejects_pin_outside_upstream_main(provenance_graph: dict[str, str | Path]) -> None:
    foreign = str(provenance_graph["foreign"])
    metadata = replace(
        _metadata(provenance_graph, head=foreign),
        head_ref=f"sync/upstream-20260902-{foreign[:8]}",
        title=f"sync: upstream @ {foreign[:8]}",
        body=_body(foreign, str(provenance_graph["baseline"])),
    )
    with pytest.raises(ProvenanceError, match="not part of upstream main"):
        verify_upstream_provenance(metadata, Path(provenance_graph["repository"]))


@pytest.mark.parametrize("parent_shape", ["one", "wrong", "three", "extra-commit"])
def test_rejects_wrong_or_extra_candidate_history(
    provenance_graph: dict[str, str | Path], parent_shape: str
) -> None:
    repository = Path(provenance_graph["repository"])
    tree = str(provenance_graph["tree"])
    baseline = str(provenance_graph["baseline"])
    pin = str(provenance_graph["pin"])
    foreign = str(provenance_graph["foreign"])
    root = str(provenance_graph["root"])
    resolved = str(provenance_graph["resolved"])
    parents = {
        "one": (baseline,),
        "wrong": (baseline, foreign),
        "three": (baseline, pin, root),
        "extra-commit": (resolved,),
    }[parent_shape]
    candidate = _commit(repository, tree, parent_shape, *parents)

    with pytest.raises(ProvenanceError, match="exactly two parents|parents must be exactly"):
        verify_upstream_provenance(_metadata(provenance_graph, head=candidate), repository)


def test_rejects_stale_or_foreign_baseline(provenance_graph: dict[str, str | Path]) -> None:
    stale_baseline = str(provenance_graph["stale_baseline"])
    metadata = replace(
        _metadata(provenance_graph),
        body=_body(str(provenance_graph["pin"]), stale_baseline),
    )
    with pytest.raises(ProvenanceError, match="not an ancestor of the pull request base"):
        verify_upstream_provenance(metadata, Path(provenance_graph["repository"]))


@pytest.mark.parametrize("duplicate", [False, True])
def test_rejects_missing_or_duplicate_baseline_metadata(
    provenance_graph: dict[str, str | Path], duplicate: bool
) -> None:
    pin = str(provenance_graph["pin"])
    baseline_line = f"Fork integration baseline: `{EXPECTED_REPOSITORY}@{provenance_graph['baseline']}`\n"
    body = f"Pinned upstream sync: `The-PR-Agent/pr-agent@{pin}`\n"
    if duplicate:
        body += baseline_line * 2
    metadata = replace(_metadata(provenance_graph), body=body)
    with pytest.raises(ProvenanceError, match="exactly one canonical fork integration baseline"):
        verify_upstream_provenance(metadata, Path(provenance_graph["repository"]))


def test_rejects_pin_already_in_current_base(provenance_graph: dict[str, str | Path]) -> None:
    repository = Path(provenance_graph["repository"])
    integrated_base = _commit(
        repository,
        str(provenance_graph["tree"]),
        "already integrated base",
        str(provenance_graph["baseline"]),
        str(provenance_graph["pin"]),
    )
    metadata = replace(_metadata(provenance_graph), base_sha=integrated_base)
    with pytest.raises(ProvenanceError, match="already part of the pull request base"):
        verify_upstream_provenance(metadata, Path(provenance_graph["repository"]))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"head_ref": "sync/upstream-20260902-deadbeef"}, "branch suffix"),
        ({"title": "sync: upstream @ deadbeef"}, "title"),
        ({"head_repo": "attacker/pr-agent"}, "branches must be owned"),
        ({"base_repo": "attacker/pr-agent"}, "must target"),
        ({"body": "missing metadata"}, "exactly one canonical upstream pin"),
    ],
)
def test_rejects_metadata_mismatches(
    provenance_graph: dict[str, str | Path], changes: dict[str, str], message: str
) -> None:
    with pytest.raises(ProvenanceError, match=message):
        verify_upstream_provenance(
            replace(_metadata(provenance_graph), **changes), Path(provenance_graph["repository"])
        )


def test_workflow_runs_only_the_protected_base_verifier() -> None:
    workflow = _read_source_contract(".github/workflows/upstream-provenance.yml")
    wrapper = _read_source_contract("scripts/verify_upstream_provenance.py")
    assert "pull_request_target:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "ref: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "python scripts/verify_upstream_provenance.py" in workflow
    assert "persist-credentials: false" in workflow
    assert wrapper == (
        "#!/usr/bin/env python3\n"
        '"""Run the packaged upstream-provenance verifier from a protected base checkout."""\n'
        "\n"
        "from pr_agent.upstream_provenance import main\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


def test_importer_conflict_recipe_creates_and_safely_retires_a_worktree() -> None:
    workflow = _read_source_contract(".github/workflows/upstream-sync.yml")
    assert 'WORKTREE_PATH=/absolute/path/you/choose/pr-agent-upstream-${SHORT}' in workflow
    assert 'git worktree add --detach \\"\\$WORKTREE_PATH\\" \\"$BASELINE\\"' in workflow
    assert "GitHub shows the PR merged" in workflow
    assert 'git -C \\"\\$WORKTREE_PATH\\" status --short' in workflow
    assert "set -euo pipefail" in workflow
    assert "TRUSTED_CHECKOUT=/absolute/path/to/your/trusted/pr-agent-checkout" in workflow
    assert 'gh pr view \\"$BRANCH\\" --json state --jq .state' in workflow
    assert 'test -z "$(git -C "$WORKTREE_PATH" status --short)"' in workflow
    assert 'git worktree remove "$WORKTREE_PATH"' in workflow
    assert "git worktree remove --force" not in workflow


def test_importer_conflict_recipe_leases_the_exact_current_remote_head() -> None:
    workflow = _read_source_contract(".github/workflows/upstream-sync.yml")
    assert 'EXPECTED_REMOTE_HEAD=$(git ls-remote --exit-code origin "refs/heads/%s" | cut -f1)' in workflow
    assert "--force-with-lease=refs/heads/%s:$EXPECTED_REMOTE_HEAD" in workflow
    assert "--force-with-lease=refs/heads/$BRANCH:$SHA" not in workflow
