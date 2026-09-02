# Fork Maintenance

Created: 2026-08-07. Last edited: 2026-09-02.

This repository (`qfennessy/pr-agent`) is a deliberately conservative fork of
[`The-PR-Agent/pr-agent`](https://github.com/The-PR-Agent/pr-agent). PR-Agent
runs with repository write tokens and LLM API keys, executes upstream-authored
prompts, and produces review output that humans trust — so upstream changes are
treated as untrusted input until reviewed here.

## Syncing from upstream

Upstream changes enter this fork only through a reviewed pull request bound to
a fixed upstream SHA and a fixed fork baseline. Never merge a PR whose head
branch is upstream `main` itself: upstream pushes silently join such a PR after
review starts (this happened on PR #1, 2026-08-07).

The `Upstream sync PR` workflow (`.github/workflows/upstream-sync.yml`, weekly
plus manual dispatch) automates this. It pins the current upstream `main` SHA
and current fork `main` baseline, then tries to create an integration commit
whose only parents are those two commits. If Git reports conflicts, the
workflow publishes the raw upstream pin as a temporary PR head and includes
the commands for replacing it with the required two-parent integration commit.
The branch is named `sync/upstream-<date>-<upstream-sha>`; its suffix always
identifies the upstream pin, not the integration commit.

### Resolving an upstream conflict

Use a dedicated worktree. Start from the exact `Fork integration baseline`
listed in the PR body, merge the exact `Pinned upstream sync` commit, resolve
the conflicts, and create one merge commit. Do not add preparation or cleanup
commits to the sync branch: the resulting candidate must have exactly two
parents, the fork baseline and the upstream pin (either parent order is valid).

The generated PR body provides copyable commands with an exact
`--force-with-lease` expectation for replacing the raw pin. Existing raw-pin
PRs, including PR #35, can use the same protocol: add exactly one canonical
`Fork integration baseline` line to the body, create the two-parent merge on
that baseline, and replace the PR head. If `main` advances and the candidate
then conflicts, rebuild the candidate directly on the new baseline and update
that body line; do not merge `main` into the existing candidate because that
would add an unverified commit layer.

### Automation credential

The workflow uses the protected `upstream-sync` Actions environment. That
environment allows only the `main` branch and contains a secret named
`UPSTREAM_SYNC_DEPLOY_KEY`. Its value is the private half of a write-enabled
deploy key attached only to `qfennessy/pr-agent`. Git uses that key to push the
pinned sync branch; the built-in `GITHUB_TOKEN` handles pull-request API calls.

The deploy key is essential: GitHub's built-in Actions token cannot push an
upstream commit that changes `.github/workflows/**`. If the secret is absent,
the workflow stops before checkout with a setup error instead of failing later
while pushing the pinned branch. Never allow `sync/upstream-*` branches to use
the environment: imported workflow code must not receive this credential.
Rotate the key according to the fork maintainer's normal credential schedule
and update only the deploy key and environment secret, never this workflow.

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
3. Wait for the required `Upstream provenance` check. It runs the verifier from
   the protected base branch, never the PR, and verifies that:
   - branch, title, and body identify one canonical upstream pin;
   - that pin belongs to upstream `main`;
   - the declared fork baseline is an ancestor of the current PR base; and
   - the PR head is either the raw pin or one merge commit whose two parents
     are exactly the pin and baseline.
4. Merge with a merge commit. The `main` ruleset rejects squash and rebase
   merges so the reviewed PR candidate remains an immutable parent.
5. After merging, inspect the repository merge commit's second parent (the PR
   candidate) with `git log -1 --format=%P <merge-commit>`. For a raw pin it
   equals the reviewed upstream SHA. For a resolved candidate, inspect that
   commit with `git log -1 --format=%P <candidate>` and confirm its only
   parents are the reviewed fork baseline and upstream pin.

## CI posture

- Fork CI holds no real provider secrets. E2E stays `workflow_dispatch` only;
  publishing runs only from a manual dispatch or a published release.
- Unit tests and CodeQL validate non-documentation changes, focused MkDocs
  validation covers documentation changes, and upstream provenance classifies
  every PR. The active `main` ruleset currently has no required status checks;
  it requires a PR, resolved review conversations, and a merge commit. Direct
  and force pushes are blocked.
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
