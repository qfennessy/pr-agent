# Local snapshot review

Use `review-snapshot` to review a saved file, the current dirty worktree, or the
staged pre-commit change before a pull request exists. Each run captures one immutable
diff and binds the result to a content-derived snapshot id. If the worktree changes while
the model is running, PR-Agent returns `stale` and suppresses the old review.

Local snapshot reviews are advisory. They cannot block a save, edit code, execute a
model-generated command, or publish to GitHub.

## Explicit commands

Review one saved file:

```bash
python -m pr_agent.cli review-snapshot --event file-save --path src/file.py
```

Review all staged and unstaged changes plus untracked text files:

```bash
python -m pr_agent.cli review-snapshot --event worktree-idle --base origin/main
```

Review only the Git index before committing:

```bash
python -m pr_agent.cli review-snapshot --event pre-commit
```

Use `--intent` to pass the current task, `--policy-version` to identify routing
policy, and repeat `--deterministic-check` with a JSON object when the caller already
has lint, type-check, or test evidence. `--output` writes the rendered Markdown review;
`--json-output` writes the snapshot-bound JSON result. The JSON result is always printed
to stdout.

## Result contract

Every result includes the reviewed `snapshot_id`, the `current_snapshot_id`, coverage
issues, latency, available token and cost data, and exactly one state:

- `findings`: the review returned one or more candidate defects.
- `no_findings`: the review completed without candidate defects.
- `coverage_unavailable`: no trustworthy review completed, or no reviewable diff remained.
- `cancelled`: reserved for callers or the optional watcher that cancel active work.
- `stale`: repository content changed after capture; the old review is omitted.

Skipped binary, unreadable, oversized, excluded, or unsafe paths appear in
`coverage_issues`. They never turn into a false clean result. Default logs and the result
envelope do not contain the full diff; the immutable input envelope retains the exact diff
in memory only for the review path. `local_pair_review.max_snapshot_bytes` bounds the
cumulative captured diff; paths beyond that budget are reported with
`snapshot_byte_budget` coverage.

Successful identical snapshots can use a bounded cache under the repository's Git common
directory. The key includes repository identity, exact diff content, event, base revision,
task intent, deterministic results, and policy version, so results cannot cross repositories
or configurations. Pass `--no-cache` for a forced model run.

## Editor integration

An editor `afterSave` task can invoke the command with the saved repository-relative path:

```json
{
  "command": "python -m pr_agent.cli review-snapshot --event file-save --path ${relativeFile} --json-output \"$(git rev-parse --git-path pr-agent/latest-save.json)\"",
  "runOn": "afterSave"
}
```

Keep the integration asynchronous and non-blocking. Editors use different task schemas;
the stable integration point is the command and JSON result, not this illustrative wrapper.

## Coding-agent stop callback

Run a coherent worktree review when a coding agent reaches a stop boundary:

```bash
python -m pr_agent.cli review-snapshot \
  --event worktree-idle \
  --base origin/main \
  --intent "${TASK_SUMMARY}" \
  --json-output "$(git rev-parse --git-path pr-agent/agent-stop.json)"
```

Resolving the artifact path through Git works in both a primary checkout and a linked
worktree, where `.git` is a pointer file rather than a directory.

The callback should consume output only when `snapshot_id` equals
`current_snapshot_id`. `stale`, `cancelled`, and `coverage_unavailable` are incomplete
coverage, not approval to continue or merge.

## Optional watcher

No filesystem watcher is required. Explicit editor and coding-agent invocations are the
supported Phase 1 boundary. A future opt-in watcher may reuse the same snapshot, cache,
and stale-result contracts with per-path debounce, bounded concurrency, and
latest-write-wins cancellation.
