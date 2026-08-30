## Overview

The `review` tool scans the PR code changes, and generates feedback about the PR, aiming to aid the reviewing process.
<br>
The tool can be triggered automatically every time a new PR is [opened](../usage-guide/automations_and_usage.md#github-app-automatic-tools-when-a-new-pr-is-opened), or can be invoked manually by commenting on any PR:

```
/review
```

Note that the main purpose of the `review` tool is to provide the **PR reviewer** with useful feedback and insights. The PR author, in contrast, may prefer to save time and focus on the output of the [improve](./improve.md) tool, which provides actionable code suggestions.

(Read more about the different personas in the PR process and how PR-Agent aims to assist them in our [blog](https://www.codium.ai/blog/understanding-the-challenges-and-pain-points-of-the-pull-request-cycle/))

## Example usage

### Manual triggering

Invoke the tool manually by commenting `/review` on any PR:

![review comment](https://codium.ai/images/pr_agent/review_comment.png){width=512}

After ~30 seconds, the tool will generate a review for the PR:

![review](https://codium.ai/images/pr_agent/review3.png){width=512}

#### Bugs-only reviews

The default `full` profile keeps the existing review: effort, tests, security summary, ticket analysis, and key issues
can all appear when their settings are enabled. Use the `bugs_only` profile when the review should contain only
introduced functional bugs, security vulnerabilities, or demonstrated material performance regressions:

```
/review --pr_reviewer.review_profile=bugs_only
```

Or make it persistent in `.pr_agent.toml`:

```toml
[pr_reviewer]
review_profile = "bugs_only"
```

Before, a style-only PR could still receive the normal review sections or a "No major issues detected" comment.
After selecting `bugs_only`, non-defect feedback is omitted and findings with the same root cause are collapsed.
GitHub check-run failures are supplied as bounded evidence; a finding is suppressed as a CI duplicate only when it
names an observed failed check. If CI evidence is unavailable, PR-Agent keeps the finding instead of trusting a model
guess. PR-Agent publishes no summary when no qualifying finding remains, and a clean rerun removes its previous
profile-specific persistent defect comment without replacing a full-profile review. When GitHub check-run publication
is enabled, bugs-only reviews use their own check name and a clean rerun updates that check to a clean result. Structured
output still contains `review.key_issues_to_review`, using an empty list for that case. Set `inline_key_issues = true`
to keep supported inline publication behavior.

If you want to edit [configurations](#configuration-options), add the relevant ones to the command:

```
/review --pr_reviewer.some_config1=... --pr_reviewer.some_config2=...
```

#### Risk-based review depth

Review depth is separate from the `full` or `bugs_only` output profile. When `[review_depth]` is absent or disabled,
PR-Agent keeps today's standard review and inherited model, diff, timeout, retry, finding, and publication settings.
When enabled, `quick`, `standard`, and `deep` are immutable budget profiles. Select one explicitly with
`requested_depth`, or use `auto` to route from changed paths, old and new rename paths, diff size, dependency files,
generated/docs/tests-only evidence, labels, and repository-defined sensitive categories.

This is a complete repository example:

```toml
[review_depth]
enabled = true
requested_depth = "auto"
version = "acme-review-router-v1"
large_change_files = 25
large_change_lines = 1000
consume_specialist_escalation = false
specialist_escalation_depth = "deep"

[review_depth.profiles.quick]
context_tokens = 8000
max_findings = 2
max_verification_candidates = 1
model_route = "weak"
timeout_seconds = 30
max_retries = 0
max_output_tokens = 2048
max_published_findings = 2
publication_threshold = "high"
shadow_only = false

[review_depth.profiles.standard]
context_tokens = 24000
max_findings = 3
max_verification_candidates = 3
model_route = "regular"
timeout_seconds = 120
max_retries = 1
max_output_tokens = 4096
max_published_findings = 3
publication_threshold = "medium"
shadow_only = false

[review_depth.profiles.deep]
context_tokens = 32000
max_findings = 6
max_verification_candidates = 6
model_route = "reasoning"
timeout_seconds = 240
max_retries = 2
max_output_tokens = 8192
max_published_findings = 6
publication_threshold = "low"
shadow_only = false

[[review_depth.sensitive_categories]]
name = "security"
path_patterns = ["**/security/**", "**/crypto/**", "**/secrets/**"]
labels = ["security"]

[[review_depth.sensitive_categories]]
name = "authorization"
path_patterns = ["**/auth/**", "**/authorization/**", "**/permissions/**"]
labels = ["authorization"]

[[review_depth.sensitive_categories]]
name = "tenant_isolation"
path_patterns = ["**/tenant/**", "**/tenancy/**"]
labels = ["tenant-isolation"]

[[review_depth.sensitive_categories]]
name = "billing"
path_patterns = ["**/billing/**", "**/payments/**"]
labels = ["billing"]

[[review_depth.sensitive_categories]]
name = "migration"
path_patterns = ["**/migrations/**", "**/schema/**"]
labels = ["migration"]

[[review_depth.sensitive_categories]]
name = "concurrency"
path_patterns = ["**/concurrency/**", "**/locks/**", "**/queues/**"]
labels = ["concurrency"]

[[review_depth.sensitive_categories]]
name = "destructive_operation"
path_patterns = ["**/delete/**", "**/cleanup/**", "**/purge/**"]
labels = ["destructive-operation"]

[[review_depth.sensitive_categories]]
name = "deployment_credentials"
path_patterns = ["**/.github/workflows/**", "**/deploy/**", "**/credentials/**"]
labels = ["deployment-credentials"]
```

`model_route` is provider-neutral: `inherit` keeps the ordinary configured model route, while `regular`, `weak`, and
`reasoning` select the existing `config.model`, `config.model_weak`, or `config.model_reasoning` settings and the normal
fallback/deployment configuration. `max_retries` counts retries after the first attempt, so `0` means one attempt.
`max_findings` bounds generated candidates, and `max_published_findings` is enforced before structured or Markdown
publication. `max_verification_candidates` is a request-local budget for guarded verifier stages and is recorded even
when no verifier is enabled; it does not launch an independent model by itself. `publication_threshold` is supplied to
the reviewer prompt and recorded in structured metadata; it does not invent a severity score after the model answers.
`shadow_only = true` keeps the bounded result in the run artifact and GitHub Actions output without mutating the review
provider. It does not publish structured or Markdown reviews, inline findings, progress comments, review labels, or
delete an earlier persistent review.

Forced-deep rules always win, including when a sensitive file is renamed or deleted. A dependency change requires at
least `standard`; a large or malformed change selects `deep`. Missing paths, line counts, labels, or optional escalation
evidence prevent `quick` and select at least `standard`. Unknown profiles, model-route names, budgets, thresholds, and
rule shapes fail closed to `deep` before the main review model runs. Profile inheritance is deliberately unsupported, so
there are no inheritance cycles to resolve.

The small risk specialist is not the final correctness judge. Its validated output can say only `escalate` or `none`;
`none` never lowers deterministic depth, and model prose is never fed back into the router. Guarded consumption has its
own default-off `consume_specialist_escalation` switch and should remain off until the benchmark and rollout gate in
issue #27 approves it. Disabled, unavailable, timed-out, or low-confidence specialist evidence raises a would-be quick
route to `standard`; stale, malformed, or identity-mismatched evidence fails closed to `deep`. Neither outcome can
lower a deterministic forced-deep route.

#### Shadow review specialists

PR-Agent includes a disabled-by-default shadow stage for three narrow tasks: change classification, an upward-only
risk recommendation, and diff-hunk prioritization. Shadow output is structured telemetry only. It cannot change the
main review's depth, prompt diff, findings, labels, comments, or approval state.

Enable the pipeline and only the roles you want to measure in `.pr_agent.toml`:

```toml
[specialist_pipeline]
enabled = true
mode = "shadow"
aggregate_timeout_seconds = 8
aggregate_token_budget = 12000

[specialist_pipeline.change_classification]
enabled = true
model = "openai/compatible-small-model"
deployment = "classification-deployment"
fallback_models = []
timeout_seconds = 5
model_retries = 1
input_token_budget = 4000
output_token_budget = 600
minimum_confidence = 0.6

[specialist_pipeline.risk_recommendation]
enabled = false

[specialist_pipeline.diff_prioritization]
enabled = false
```

Each role has independent input/output schema versions, model/deployment, fallback, timeout, retry, input/output
budget, confidence, and enablement settings. Compatible tuned endpoints use the same versioned role contracts as
prompted models. Calls share one
immutable input and reserve their worst-case token budget before concurrent execution. A failed, timed-out, malformed,
low-confidence, or stale role is recorded separately and never blocks the ordinary review.

Worst-case reservation includes every configured fallback model, model attempt, and provider retry. GitHub and GitLab
reviews use a refreshable head commit for stale-run cancellation, while local file-save reviews use their immutable
snapshot. Other hosted providers record `unavailable` role evidence and make no specialist calls until their adapter can
provide a stable, refreshable head identity; they are never silently treated as successful shadow runs.

Issue #11 is the only consumer of upward risk recommendations. Issue #9 is the only consumer of ranked hunks and
context requests. Those guarded consumers remain disabled until issue #27 completes the frozen benchmark, target-repo
pilot, live-shadow evidence window, and rollout decisions.

### Automatic triggering

To run the `review` automatically when a PR is opened, define in a [configuration file](../usage-guide/configuration_options.md#local-configuration-file):

```
[github_app]
pr_commands = [
    "/review",
    ...
]

[pr_reviewer]
extra_instructions = "..."
...
```

- The `pr_commands` lists commands that will be executed automatically when a PR is opened.
- The `[pr_reviewer]` section contains the configurations for the `review` tool you want to edit (if any).

## Configuration options

???+ example "General options"

    <table>
      <tr>
        <td><b>review_profile</b></td>
        <td>
          Review output contract. <code>full</code> preserves the standard review; <code>bugs_only</code> emits only
          qualifying introduced defects and stays silent when none remain. Default is <code>full</code>.
        </td>
      </tr>
      <tr>
        <td><b>persistent_comment</b></td>
        <td>If set to true, the review comment will be persistent, meaning that every new review request will edit the previous one. Default is true.</td>
      </tr>
      <tr>
        <td><b>review_heading</b></td>
        <td>
          Visible base heading for review comments, without the Markdown prefix or incremental label.
          For example, <code>review_heading = "Guideline Compliance Check"</code> renders
          <code>## Guideline Compliance Check 🔍</code> for a full review and
          <code>## Incremental Guideline Compliance Check 🔍</code> for an incremental review.
          On GitHub, GitLab, Azure DevOps, and Bitbucket Cloud, changing this value updates the same
          persistent review comment; it does not create a separate review channel.
          Default is <code>PR Reviewer Guide</code>.
        </td>
      </tr>
      <tr>
      <td><b>final_update_message</b></td>
      <td>When set to true, updating a persistent review comment during online commenting will automatically add a short comment with a link to the updated review in the pull request .Default is true.</td>
      </tr>
      <tr>
        <td><b>extra_instructions</b></td>
        <td>Optional extra instructions to the tool. For example: "focus on the changes in the file X. Ignore change in ...".</td>
      </tr>
      <tr>
        <td><b>enable_help_text</b></td>
        <td>If set to true, the tool will display a help text in the comment. Default is false.</td>
      </tr>
      <tr>
        <td><b>enable_review_coverage_footer</b></td>
        <td>If set to true, the tool will display a review coverage footer when the token budget leaves files out of the review. Default is true.</td>
      </tr>
      <tr>
        <td><b>num_max_findings</b></td>
        <td>Number of maximum returned findings. Default is 3.</td>
      </tr>
      <tr>
        <td><b>inline_key_issues</b></td>
        <td>Azure DevOps only. If set to true, each key issue is published as an inline thread. A finding leaves the review summary when a matching thread exists or Azure accepts the new thread. Findings that cannot be anchored or published stay in the summary. Default is false.</td>
      </tr>
    </table>

???+ example "Enable\\disable specific sub-sections"

    <table>
      <tr>
        <td><b>require_score_review</b></td>
        <td>If set to true, the tool will add a section that scores the PR. Default is false.</td>
      </tr>
      <tr>
        <td><b>require_tests_review</b></td>
        <td>If set to true, the tool will add a section that checks if the PR contains tests. Default is true.</td>
      </tr>
      <tr>
        <td><b>require_estimate_effort_to_review</b></td>
        <td>If set to true, the tool will add a section that estimates the effort needed to review the PR. Default is true.</td>
      </tr>
      <tr>
        <td><b>require_estimate_contribution_time_cost</b></td>
        <td>If set to true, the tool will add a section that estimates the time required for a senior developer to create and submit such changes. Default is false.</td>
      </tr>
      <tr>
        <td><b>require_can_be_split_review</b></td>
        <td>If set to true, the tool will add a section that checks if the PR contains several themes, and can be split into smaller PRs. Default is false.</td>
      </tr>
      <tr>
        <td><b>require_security_review</b></td>
        <td>If set to true, the tool will add a section that checks if the PR contains a possible security or vulnerability issue. Default is true.</td>
      </tr>
        <tr>
        <td><b>require_todo_scan</b></td>
        <td>If set to true, the tool will add a section that lists TODO comments found in the PR code changes. Default is false.
        </td>
      </tr>
      <tr>
        <td><b>require_ticket_analysis_review</b></td>
        <td>If set to true, and the PR contains a GitHub or Jira ticket link, the tool will add a section that checks if the PR in fact fulfilled the ticket requirements. Default is true.</td>
      </tr>
    </table>

???+ example "Adding PR labels"

    You can enable\disable the `review` tool to add specific labels to the PR:

    <table>
      <tr>
        <td><b>enable_review_labels_security</b></td>
        <td>If set to true, the tool will publish a 'possible security issue' label if it detects a security issue. Default is true.</td>
      </tr>
      <tr>
        <td><b>enable_review_labels_effort</b></td>
        <td>If set to true, the tool will publish a 'Review effort x/5' label (1–5 scale). Default is true.</td>
      </tr>
    </table>

## Usage Tips

### General guidelines

!!! tip ""

    The `review` tool provides a collection of configurable feedbacks about a PR.
    It is recommended to review the [Configuration options](#configuration-options) section, and choose the relevant options for your use case.

    Some of the features that are disabled by default are quite useful, and should be considered for enabling. For example:
    `require_score_review`, and more.

    On the other hand, if you find one of the enabled features to be irrelevant for your use case, disable it. No default configuration can fit all use cases.

### Automation

!!! tip ""
    When you first install PR-Agent app, the [default mode](../usage-guide/automations_and_usage.md#github-app-automatic-tools-when-a-new-pr-is-opened) for the `review` tool is:
    ```
    pr_commands = ["/review", ...]
    ```
    Meaning the `review` tool will run automatically on every PR, without any additional configurations.
    Edit this field to enable/disable the tool, or to change the configurations used.

### Auto-generated PR labels by the Review Tool

!!! tip ""

    The `review` can tool automatically add labels to your Pull Requests:

    - **`possible security issue`**: This label is applied if the tool detects a potential [security vulnerability](https://github.com/the-pr-agent/pr-agent/blob/main/pr_agent/settings/pr_reviewer_prompts.toml#L134) in the PR's code. This feedback is controlled by the 'enable_review_labels_security' flag (default is true).
    - **`review effort [x/5]`**: This label estimates the [effort](https://github.com/the-pr-agent/pr-agent/blob/main/pr_agent/settings/pr_reviewer_prompts.toml#L118) required to review the PR on a relative scale of 1 to 5, where 'x' represents the assessed effort. This feedback is controlled by the 'enable_review_labels_effort' flag (default is true).
    - **`ticket compliance`**: Adds a label indicating code compliance level ("Fully compliant" | "PR Code Verified" | "Partially compliant" | "Not compliant") to any GitHub/Jira/Linea ticket linked in the PR. Controlled by the 'require_ticket_labels' flag (default: false). If 'require_no_ticket_labels' is also enabled, PRs without ticket links will receive a "No ticket found" label.


### Auto-blocking PRs from being merged based on the generated labels

!!! tip ""

    You can configure a CI/CD Action to prevent merging PRs with specific labels. For example, implement a dedicated [GitHub Action](https://medium.com/sequra-tech/quick-tip-block-pull-request-merge-using-labels-6cc326936221).

    This approach helps ensure PRs with potential security issues or ticket compliance problems will not be merged without further review.

    Since AI may make mistakes or lack complete context, use this feature judiciously. For flexibility, users with appropriate permissions can remove generated labels when necessary. When a label is removed, this action will be automatically documented in the PR discussion, clearly indicating it was a deliberate override by an authorized user to allow the merge.

### Extra instructions

!!! tip ""

    Extra instructions are important.
    The `review` tool can be configured with extra instructions, which can be used to guide the model to a feedback tailored to the needs of your project.

    Be specific, clear, and concise in the instructions. With extra instructions, you are the prompter. Specify the relevant sub-tool, and the relevant aspects of the PR that you want to emphasize.

    Examples of extra instructions:
    ```
    [pr_reviewer]
    extra_instructions="""\
    In the code feedback section, emphasize the following:
    - Does the code logic cover relevant edge cases?
    - Is the code logic clear and easy to understand?
    - Is the code logic efficient?
    ...
    """
    ```
    Use triple quotes to write multi-line instructions. Use bullet points to make the instructions more readable.
