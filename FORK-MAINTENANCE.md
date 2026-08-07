# Fork Maintenance

Created: 2026-08-07. Last edited: 2026-08-07.

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
3. After merging, confirm the merge commit's upstream parent equals the pinned
   SHA that was reviewed (`git log -1 --format=%P <merge-commit>`).

## CI posture

- Fork CI holds no real provider secrets. E2E and publish workflows stay
  `workflow_dispatch` (manual) only.
- Build-and-test, CodeQL, and coverage gate PRs; keep branch protection on
  `main` (PR required, checks required, no direct pushes).
- Keep third-party actions pinned by commit SHA, as upstream does.

## Contributing back to upstream

Follow upstream's `CONTRIBUTING.md`: focused single-topic PRs, conventional
commits, pytest coverage, `fix/<issue>` or `feature/<name>` branches.

Security-relevant issues are never reported through public issues or PRs.
Upstream has GitHub private vulnerability reporting enabled — use
<https://github.com/The-PR-Agent/pr-agent/security/advisories/new> first, and
only open a public PR once upstream agrees or a fix is released.
