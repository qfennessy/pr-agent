# What this fork changes

Created: 2026-09-04. Last edited: 2026-09-04.

This file explains what `qfennessy/pr-agent` does that
[`The-PR-Agent/pr-agent`](https://github.com/The-PR-Agent/pr-agent) does not, and
how work gets into it.

For the mechanics of pulling upstream changes in, see `FORK-MAINTENANCE.md`.
This file is about the code, not the plumbing.

## What is different when you run it today

Five things behave differently out of the box. Everything else added by this fork
is switched off, and the next section explains why.

### Retries are capped

A stalled model used to be able to burn a lot of time. Three retry settings
multiply together: attempts per model, times the number of fallback models, times
the retries the provider's own client does on its own. That last one is invisible
in the logs, so a single 120-second timeout could show up as six minutes of real
time.

The fork sets `model_retries=2` and adds `ai_provider_max_retries` so the hidden
one can be pinned. Set all three and the worst case is a number you chose.

### Issue references must say what they mean

Upstream treated almost any `#123` in a description as a link to issue 123. It
skipped code blocks, but nothing else, so ordinary prose containing `#123` was
read as an issue reference.

The fork requires a word like "Fixes", "References" or "Related to" in front of
the shorthand. Full URLs are always accepted. On by default
(`require_explicit_issue_reference`); set it false for the old behavior.

### Reviews say which model wrote them

Every published review carries a line like ``> Reviewed by `deepseek-v4-flash` ``,
and says so when a fallback model answered instead of the primary.

Upstream has no attribution. Without it there is no way to tell which model
produced a review, or whether the one you configured is the one that ran.

### Several reviewers can share one pull request

Upstream keeps one persistent review comment per pull request, so a second
reviewer overwrites the first. The fork adds `persistent_comment_id`: give each
run a different value and each updates only its own comment. This is what lets a
repository run two models side by side and keep both opinions.

### You can force specific files to be reviewed

PR-Agent skips files it does not consider source code, so a pull request that
only touches a lockfile gets no review at all. The fork adds `files_to_review`,
an exact-filename allowlist that overrides the skip. Empty unless set.

## How this fork tests itself

This is the largest difference, and it is not visible in the product.

| | Upstream | This fork |
| --- | ---: | ---: |
| Unit test files | 175 | 204 |
| Test functions | 1,028 | 2,217 |

The fork roughly doubled the test count. Most of that is not testing the five
changes above; it is testing the review pipeline described in the next section,
which nobody is running yet.

The reason is the shape of the problem. A code reviewer that is wrong is worse
than no reviewer, because people stop reading it. So the new code is written to
fail closed, and the tests mostly pin the ways it must refuse to act:

- A missing measurement is recorded as unavailable, never as zero. A run with no
  cost telemetry does not report a cost of nothing.
- A review that ran out of output tokens partway cannot count as "found nothing".
- A clean result cannot close someone's review thread unless four separate things
  are all true: generation finished normally, every patch it read was complete, no
  changed file was skipped, and nothing was left unreviewed.
- Spending denials propagate rather than being swallowed by a broad `except`.

Several of those exist because a review found the opposite behavior first. They
are regression pins, not decoration.

### Running the tests

```
PYTHONPATH=. ./.venv/bin/pytest tests/unittest -q      # the whole suite
PYTHONPATH=. ./.venv/bin/pytest tests/unittest/test_pr_reviewer_thread_lifecycle.py -q
```

`PYTHONPATH=.` is required or imports fail. Python 3.12 or newer.

One trap worth knowing: the suite pins `litellm==1.99.0`, and two tests assert
against the installed version. Running with a stale virtualenv makes them fail in
a way that looks like a code regression and is not.

## How this fork changes PR-Agent

The rule is that nothing ships enabled.

Seven substantial review features are merged and every one defaults to off:

| Feature | What it would do |
| --- | --- |
| Bugs-only review mode | Report only real defects, not general commentary |
| Risk-based review depths | Spend more effort on riskier changes |
| Low-cost specialists | Cheap models that classify and rank a change |
| Candidate verification | Check a suspected bug against surrounding code before reporting it |
| Frontier adjudication | Ask a stronger model to settle disputed or sensitive findings |
| Stable review threads | Keep comments anchored across pushes instead of duplicating them |
| Checkpoint evaluation | Measure all of the above against known defects |

Together they are meant to give a developer useful feedback while they are still
writing the change, instead of waiting for a pull request. Cheap models look
first, a second pass checks whether each suspected bug is real, and a stronger
model only gets involved when the cheap ones disagree or the code is sensitive.

They are off because a feature is not finished when it works. It is finished when
there is evidence it helps.

### Evidence is code, not a promise

The fork ships a benchmark harness that replays the whole pipeline against a
frozen set of real defects and decides whether it is good enough. Five rollout
gates are written as machine-checkable rules with thresholds and minimum sample
sizes:

| Gate | What it requires |
| --- | --- |
| Offline replay | Reproducible runs, no answer leakage, 99.5% valid structured output |
| Live shadow | Seven days of real usage telemetry within a stated latency and cost budget |
| Opt-in pair review | 80% verified precision, no regression on older code |
| Default pair review | 90% actionable precision over 100 settled findings |
| Publication | Beats the current reviewer on a held-out set at an agreed cost ceiling |

A missing denominator, an insufficient sample, or a failed run makes a gate
`not_evaluable`. That is never permission to roll out.

Two deliberate obstacles are worth knowing about, because they look like bugs:

**The evaluation refuses to run.** All five replay arms are hardcoded
unavailable. Turning the settings on is not enough — contracts the repository
does not ship yet, including a spending authority for the model gateway, must be
supplied first.

**Generating evidence does not approve it.** Four acceptance identifiers are
`None` in the source. A maintainer has to pin each one after reviewing the
artifact separately. A run cannot sign off on itself.

### Where the work is tracked

Issue #21 owns the pipeline and its delivery order. Nine of its ten workstreams
are closed; the remaining one is issue #27, the benchmark and rollout gates
above. The design lives in
`docs/docs/designs/specialist_review_pipeline.md`, and the operational detail in
`docs/docs/usage-guide/checkpoint_evaluation.md`.

## Two new commands

Both are local and neither publishes anything.

- `pr-agent review-snapshot` reviews a saved file or a working tree directly,
  with no pull request involved.
- `pr-agent evaluation-plan` validates a benchmark plan without calling a model.

The slash commands (`/review`, `/describe`, `/improve`, `/ask`) are unchanged.

## How much code this is

Against the last upstream sync: about 36,500 lines across 81 files under
`pr_agent/`, including 32 new files, plus 29 new test files.

That is a lot, and worth being honest about: almost all of it is the switched-off
pipeline and its tests. The five active differences are small by comparison.
