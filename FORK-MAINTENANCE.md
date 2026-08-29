# Fork Maintenance

Created: 2026-08-07. Last edited: 2026-08-29.

This repository (`qfennessy/pr-agent`) is a deliberately conservative fork of
[`The-PR-Agent/pr-agent`](https://github.com/The-PR-Agent/pr-agent). PR-Agent
runs with repository write tokens and LLM API keys, executes upstream-authored
prompts, and produces review output that humans trust — so upstream changes are
treated as untrusted input until reviewed here.

## Syncing from upstream

Upstream changes enter this fork only through a reviewed pull request pinned to
a fixed upstream SHA. Never merge a PR whose head branch is upstream `main`
itself: upstream pushes silently join such a PR after review starts (this
happened on PR #1, 2026-08-07).

The `Upstream sync PR` workflow (`.github/workflows/upstream-sync.yml`, weekly
plus manual dispatch) automates this: it pins the current upstream `main` SHA,
pushes a `sync/upstream-<date>-<sha>` branch, and opens a PR whose body lists
the new commits, a diffstat, and any high-scrutiny paths touched.

### Review gates for every sync PR

1. Run a full review (`/review`) against the PR before merging.
2. Give extra scrutiny to these paths — they are how an agent framework gets
   corrupted:
   - `.github/workflows/**` — especially `pull_request_target` triggers, which
     run with this repo's secrets against PR-supplied content
   - `requirements*.txt`, `pyproject.toml`, `docker/**` — supply chain
   - `pr_agent/settings/*.toml` — prompts are executable in the LLM sense
   - `pr_agent/config_loader.py`, `pr_agent/secret_providers/**`, git-provider
     auth code — secret and token handling
3. Wait for the required `Upstream provenance` check. It verifies that the PR
   head, branch name, title, and body all identify the same canonical upstream
   SHA and that the commit belongs to upstream `main`.
4. Merge with a merge commit. The `main` ruleset rejects squash and rebase
   merges so the reviewed upstream commit remains an immutable parent.
5. After merging, confirm the merge commit's upstream parent equals the pinned
   SHA that was reviewed (`git log -1 --format=%P <merge-commit>`).

## CI posture

- Fork CI holds no real provider secrets. E2E stays `workflow_dispatch` only;
  publishing runs only from a manual dispatch or a published release.
- Unit tests, coverage, CodeQL, and upstream provenance gate PRs. The `main`
  ruleset requires an up-to-date PR, resolved review conversations, and a merge
  commit; direct and force pushes are blocked.
- This single-maintainer fork currently requires zero approving reviews because
  GitHub does not let a PR author approve their own change. Increase the count
  when a second eligible reviewer is added; the `/review` gate above still
  applies to every upstream sync.
- Keep third-party actions pinned by commit SHA, as upstream does.

## Contributing back to upstream

Follow upstream's `CONTRIBUTING.md`: focused single-topic PRs, conventional
commits, pytest coverage, `fix/<issue>` or `feature/<name>` branches.

Security-relevant issues are never reported through public issues or PRs.
Upstream has GitHub private vulnerability reporting enabled — use
<https://github.com/The-PR-Agent/pr-agent/security/advisories/new> first, and
only open a public PR once upstream agrees or a fix is released.
