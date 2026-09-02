# CI configuration

This repository uses GitHub Actions for pull-request validation, security scanning, documentation deployment,
upstream synchronization, and releases. Workflow definitions live in [`.github/workflows`](https://github.com/qfennessy/pr-agent/tree/main/.github/workflows).

This page documents the configuration in the `qfennessy/pr-agent` fork. Repository rulesets, environments, and
secret values are configured in GitHub rather than in the repository, so maintainers must verify those settings in
GitHub when changing required checks or release permissions.

As reviewed on August 30, 2026, the active `main` ruleset does not require named status checks. The event-level path
filters below rely on that setting: if unit tests or CodeQL become required checks, replace the filters with fast
no-op jobs or update the ruleset so documentation-only PRs do not wait forever for workflows that never started.

## Workflow inventory

| Workflow | Trigger | Purpose | Credentials or write access |
| --- | --- | --- | --- |
| `Build-and-test` | Non-documentation push or pull request targeting `main` | Builds the `test` target in `docker/Dockerfile` and runs `tests/unittest` | Read-only checkout |
| `CodeQL` | Non-documentation push or pull request targeting `main`; Mondays at 06:00 UTC | Scans Python with the extended security and quality query suites | `security-events: write` |
| `docs-ci` | Pull requests changing `docs/**` or Markdown; documentation pushes to `main` or `add-docs-portal` | Validates documentation on pull requests and deploys site changes to GitHub Pages | Read access for validation; `contents: write` for deployment |
| `PR-Agent E2E tests` | Manual dispatch | Builds the test image and tests the GitHub, GitLab, and Bitbucket integrations | Provider test tokens listed below |
| `pre-commit` | Manual dispatch | Runs the hooks in `.pre-commit-config.yaml` | Read-only checkout |
| `PR-Agent` | A non-draft pull request is opened, reopened, or marked ready | Runs describe, review, and improve through the fork's own action | Model and Pinecone secrets; PR and issue write access |
| `Upstream provenance` | `pull_request_target` activity targeting `main` | On `sync/upstream-*` branches, runs the base-owned verifier and validates the immutable pin, fork baseline, and candidate topology | Read-only checkout of the protected base; network read from the PR ref and upstream |
| `Upstream sync PR` | Mondays at 12:17 UTC, or manual dispatch | Pins upstream `main`, creates a local sync branch, and opens a review PR | `main`-restricted environment containing a repository-only deploy key; pull-request write access |
| `Publish` | Published GitHub release, or manual dispatch | Publishes PyPI distributions and a multi-platform Docker image matrix, records provenance, and finalizes repository release state | Release environment and publishing secrets |

## Pull-request validation

### Unit tests

`build-and-test.yaml` builds the `test` stage from `docker/Dockerfile`. The image installs `requirements.txt` through
the package build, installs `requirements-dev.txt`, copies the application and tests, and sets `PYTHONPATH=/app`.
The workflow then runs:

```bash
docker build -f docker/Dockerfile --target test -t pragent/pr-agent:test .
docker run --rm pragent/pr-agent:test pytest -v tests/unittest
```

BuildKit's GitHub Actions cache uses the shared `dev` scope. Changes limited to `docs/**` or Markdown files skip this
workflow. The former coverage workflow was removed because it reran the complete unit suite without enforcing a
coverage threshold.

For a faster local run, use the repository virtual environment and preserve the import path:

```bash
PYTHONPATH=. ./.venv/bin/pytest tests/unittest -v
```

### Static and security checks

The `pre-commit` workflow is intentionally manual. Its active hooks check large added files, TOML, YAML, final
newlines, trailing whitespace, and import order. Ruff, Bandit, and actionlint hooks are present only as commented
configuration and therefore do not run through this workflow.

CodeQL runs on non-documentation pushes and pull requests targeting `main`, and is also scheduled weekly so newly
published queries can scan an unchanged repository. It analyzes Python with `build-mode: none`; it does not build or
execute the project.

Run the configured formatting checks locally with:

```bash
pre-commit run --all-files
```

### Automated PR-Agent review

`pr-agent-review.yaml` uses `pull_request_target` so pull requests from forks can be reviewed with repository
secrets. The job deliberately does not check out or execute pull-request code. Keep that property intact: adding a
checkout, script, action, or configuration read from the untrusted pull-request head would expose privileged tokens.

The action reference is `qfennessy/pr-agent@main`, intentionally following this protected fork rather than a fixed
commit. Changes to the protection of `main` therefore change the trust boundary of this workflow. The job skips bot
events and draft pull requests, and enables automatic describe, review, and improve operations.

The workflow receives these secrets and tokens:

| Secret or token | Purpose |
| --- | --- |
| `OPENAI_KEY` | Authenticate model requests |
| `OPENAI_ORG` | Optional OpenAI organization selection |
| `PINECONE_API_KEY` | Exported as `PINECONE.API_KEY` to authenticate Pinecone requests |
| `PINECONE_ENVIRONMENT` | Exported as `PINECONE.ENVIRONMENT` to select the Pinecone environment |
| `GITHUB_TOKEN` | Read pull-request data and publish descriptions, reviews, suggestions, and issue updates |

## Fork synchronization

`upstream-sync.yml` runs only in `qfennessy/pr-agent`. It fetches `The-PR-Agent/pr-agent` and accepts either the
current upstream `main` or a manually supplied commit. The resolved value must be a canonical commit on upstream
`main`. If the fork does not already contain it and no sync PR is open, the workflow records both that upstream pin
and the current fork-main baseline, then creates:

```text
sync/upstream-YYYYMMDD-<8-character-sha>
```

The branch suffix names the upstream pin, not necessarily the PR head. The workflow first attempts a no-fast-forward
merge of the pin into the recorded fork baseline. A clean merge produces the final, two-parent review candidate. A
conflict produces a temporary raw-pin candidate and PR instructions for resolving the conflict in a dedicated
worktree. The maintainer replaces that raw head, with an exact force-with-lease, using one merge commit whose parents
are exactly the declared baseline and pin. Commits before or after that merge are rejected; if fork `main` advances
and conflicts, rebuild the candidate on the new baseline rather than merging `main` into the old candidate.

The generated PR names high-scrutiny paths such as workflows, dependencies, prompts, secret handling, and Docker
files. The workflow uses the protected `upstream-sync` Actions environment, which permits only `main`. Its
`UPSTREAM_SYNC_DEPLOY_KEY` secret contains a private write-enabled deploy key attached only to
`qfennessy/pr-agent`; the built-in `GITHUB_TOKEN` remains limited to pull-request API calls. GitHub's built-in token
cannot push upstream commits that modify `.github/workflows/**`, so the job fails immediately when the deploy-key
secret is missing instead of falling back. Never permit `sync/upstream-*` branches to use this environment: they
contain unreviewed upstream workflow code.

`upstream-provenance.yml` uses `pull_request_target` with read-only contents permission so GitHub loads the workflow
from the protected base branch. It checks out the event's exact base SHA, fetches the PR head as a Git object without
checking it out, and runs only `scripts/verify_upstream_provenance.py` from that protected base. It never executes a
script, action, hook, or configuration from the PR candidate.

The verifier binds the branch suffix, title, and exactly one PR-body pin to the immutable upstream SHA; verifies the
pin is on upstream `main`; and requires exactly one declared fork baseline that remains an ancestor of the PR base.
The PR head may be the raw upstream pin or a conflict-resolved merge with exactly two parents: the pin and baseline,
in either order. This permits reviewed conflict resolutions without allowing an arbitrary commit chain to acquire
upstream provenance. Raw-pin candidates remain useful for clean legacy syncs and as temporary conflict placeholders,
but a conflicting raw pin must be converted to the two-parent form before GitHub can merge it.

After the repository merge, inspect its PR-candidate parent. A raw candidate must equal the pin; a resolved candidate
must have only the recorded pin and baseline as parents. Exact commands and the procedure for converting existing
raw-pin PRs such as PR #35 are in
[`FORK-MAINTENANCE.md`](https://github.com/qfennessy/pr-agent/blob/main/FORK-MAINTENANCE.md).

The merge policy and post-merge verification steps are documented in the repository's
[`FORK-MAINTENANCE.md`](https://github.com/qfennessy/pr-agent/blob/main/FORK-MAINTENANCE.md).

## Documentation deployment

`docs-ci.yaml` validates changes under `docs/**` and all Markdown changes on pull requests targeting `main`. It
deploys only changes under `docs/**` after a push to `main` or `add-docs-portal`, so a root README change does not
republish the site. Both paths install MkDocs Material, its imaging extras, and `mkdocs-glightbox`. Pull requests run
`mkdocs build`; pushes run `mkdocs gh-deploy --force`. Only the deployment job receives `contents: write`. Its cache
is keyed by the UTC week number.

Preview changes locally with:

```bash
pip install mkdocs-material "mkdocs-material[imaging]" mkdocs-glightbox
mkdocs build -f docs/mkdocs.yml
```

## End-to-end tests

`e2e_tests.yaml` is manual and runs three provider tests sequentially in one job. It requires test accounts and the
following repository secrets:

| Secret | Used by |
| --- | --- |
| `TOKEN_GITHUB` | GitHub App E2E test |
| `TOKEN_GITLAB` | GitLab webhook E2E test |
| `BITBUCKET_USERNAME` | Bitbucket App E2E test |
| `BITBUCKET_PASSWORD` | Bitbucket App E2E test |

Do not run this workflow without confirming that the credentials point to isolated test resources. A failure in an
earlier provider step prevents the later provider steps from running.

## Releases and publishing

`publish.yml` supports two entry points:

- publishing a GitHub release; or
- manually dispatching the workflow from `main` with a SemVer 2.0.0 version.

The `prepare` job normalizes both paths to a version, tag, and immutable source SHA. The concurrency group prevents a
release event tagged `vX.Y.Z` and a manual dispatch for `X.Y.Z` from overlapping. Release tags must use the `v` prefix
for that guarantee; an unprefixed `X.Y.Z` release uses a different group and can overlap the manual path. The two
publishing jobs then:

- build a Python wheel and source distribution for PyPI; and
- build twelve Docker targets for `linux/amd64` and `linux/arm64`, push versioned and selected rolling tags, and
  attach build-provenance attestations.

Both jobs run in the protected `release` environment. The workflow uses:

| Secret | Purpose |
| --- | --- |
| `PYPI_API_TOKEN` | Upload Python distributions |
| `DOCKERHUB_USERNAME` | Authenticate to Docker Hub |
| `DOCKERHUB_TOKEN` | Push Docker images |
| `RELEASE_TOKEN` | Optional token for pushing the version bump and creating the release; falls back to `GITHUB_TOKEN` |

The finalizer runs when Docker publishing succeeds even if PyPI publishing fails. In that partial-success case it
emits a warning, continues the version bump, and requires a later PyPI rerun. The push is fast-forward only: if
`main` moved after publishing began, the workflow skips the automated bump instead of attaching published artifacts
to unbuilt code.

The two release entry points have different tag behavior:

- Manual dispatch creates the GitHub release after publishing and targets the version-bump commit when one was made.
- A published-release event already has a tag. If the workflow then creates a separate version-bump commit on
  `main`, the original tag continues to identify the source tree that was built, whose `pyproject.toml` still carries
  the previous version. The distributions and images receive the requested version during their build jobs.

## Configuration review notes

The current workflows are deliberately conservative around untrusted pull requests and upstream synchronization.
Before treating all CI as comprehensive, account for these boundaries:

- A strict MkDocs build currently stops on the existing missing-`site_url` warning, so deployment uses the default
  non-strict mode.
- Pre-commit and provider E2E tests are manual, so their absence from a PR is not evidence that they passed.
- The active `check-yaml` pre-commit hook cannot construct the Python-specific emoji tags in `docs/mkdocs.yml`, so
  the manual pre-commit workflow currently fails when it checks that existing file.
- Documentation-only changes skip Docker unit tests and CodeQL and run the focused MkDocs validation instead.
- The self-review workflow trusts the moving `qfennessy/pr-agent@main` ref. That is consistent with the fork policy
  only while `main` remains protected and reviewed.
- The upstream provenance workflow deliberately uses `pull_request_target`. It has read-only permission and runs
  only the verifier checked out from the exact protected base SHA; never add a PR-head checkout or execution step.
- The publish workflow intentionally permits Docker/repository finalization when PyPI fails. Release operators must
  treat the warning as a partial release and complete the missing registry publication.
- `.github/release-drafter.yml` still contains categories and version-resolution rules, but no active workflow invokes
  Release Drafter. Maintainers must not rely on automated draft updates or title-based labeling from that file.
- Required-check selection, resolved-conversation rules, merge strategy, environment approvals, and secret scoping
  live in GitHub settings. They cannot be confirmed from these YAML files alone.

When a workflow changes, update this page in the same pull request and review permissions, triggers, expression
interpolation, third-party action pins, cache scopes, and fork behavior explicitly.
