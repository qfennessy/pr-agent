import copy
import datetime
import json
import re
import time
from functools import partial
from typing import List, Optional, Tuple

from jinja2 import Environment, StrictUndefined, select_autoescape

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import (
    LiteLLMAIHandler,
    get_effective_litellm_output_token_cap,
)
from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.candidate_verification import (
    VerificationBudgets,
    apply_specialist_prioritization,
    apply_verification_decisions,
    bounded_verification_evidence,
    prepare_candidates,
    render_verification_payload,
    retrieve_evidence,
    safe_repo_path,
    telemetry_safe_artifact,
    validated_specialist_prioritization,
)
from pr_agent.algo.git_patch_processing import iter_git_patch_lines, split_git_file_lines
from pr_agent.algo.inline_comment_dedup import (
    InlineCommentStore,
    can_verify_inline_comment_publication,
    get_inline_comment_store,
    key_issue_body_with_markers,
    key_issue_fingerprint,
    key_issue_location_fingerprint,
)
from pr_agent.algo.pr_processing import (
    OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD,
    PRDiffCoverage,
    add_ai_metadata_to_diff_files,
    get_pr_diff,
    retry_with_fallback_models,
)
from pr_agent.algo.repo_context import build_repo_context
from pr_agent.algo.review_specialists import (
    build_specialist_input,
    get_specialist_snapshot_context,
    load_specialist_pipeline_config,
    run_shadow_specialists,
    specialists_enabled,
    unavailable_specialist_batch,
)
from pr_agent.algo.run_details import get_run_details, init_run_details, record_review_profile
from pr_agent.algo.skills_loader import get_skills_context
from pr_agent.algo.token_handler import TokenEncoder, TokenHandler
from pr_agent.algo.utils import (
    ModelType,
    PRReviewHeader,
    PRReviewIdentity,
    add_pr_review_identity,
    convert_to_markdown_v2,
    get_max_tokens,
    github_action_output,
    load_yaml,
    show_relevant_configurations,
    show_run_details,
)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider_with_context
from pr_agent.git_providers.git_provider import IncrementalPR, get_main_pr_language
from pr_agent.log import get_logger
from pr_agent.servers.help import HelpMessage
from pr_agent.tools.ticket_pr_compliance_check import extract_and_cache_pr_tickets

MAX_REVIEW_COVERAGE_FILES = 50
_SUGGESTION_FENCE_RE = re.compile(r"```[ \t]*suggestion\b", re.IGNORECASE)
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_VALID_REVIEW_PROFILES = {"full", "bugs_only"}
_BUG_FINDING_HEADERS = {
    "bug": "Bug",
    "security": "Security vulnerability",
    "performance": "Performance regression",
}
_GENERIC_CI_EVIDENCE_TERMS = {
    "assert", "assertion", "build", "check", "error", "errors", "fail", "failed", "failure", "failures",
    "job", "test", "tests", "unit",
}


class PRReviewer:
    """
    The PRReviewer class is responsible for reviewing a pull request and generating feedback using an AI model.
    """

    def __init__(self, pr_url: str, is_answer: bool = False, is_auto: bool = False, args: list = None,
                 ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        """
        Initialize the PRReviewer object with the necessary attributes and objects to review a pull request.

        Args:
            pr_url (str): The URL of the pull request to be reviewed.
            is_answer (bool, optional): Indicates whether the review is being done in answer mode. Defaults to False.
            is_auto (bool, optional): Indicates whether the review is being done in automatic mode. Defaults to False.
            ai_handler (BaseAiHandler): The AI handler to be used for the review. Defaults to None.
            args (list, optional): List of arguments passed to the PRReviewer class. Defaults to None.
        """
        self.git_provider = get_git_provider_with_context(pr_url)
        self.args = args
        configured_profile = str(get_settings().pr_reviewer.get("review_profile", "full")).strip().lower()
        if configured_profile not in _VALID_REVIEW_PROFILES:
            get_logger().warning(
                f"Unknown pr_reviewer.review_profile '{configured_profile}'; falling back to 'full'"
            )
            configured_profile = "full"
        self.review_profile = configured_profile
        self.incremental = self.parse_incremental(args)  # -i command
        self.incremental.review_profile = self.review_profile
        if self.incremental and self.incremental.is_incremental:
            self.git_provider.get_incremental_commits(self.incremental)

        self.main_language = get_main_pr_language(
            self.git_provider.get_languages(), self.git_provider.get_files()
        )
        self.pr_url = pr_url
        self.is_answer = is_answer
        self.is_auto = is_auto

        if self.is_answer and not self.git_provider.is_supported("get_issue_comments"):
            raise Exception(f"Answer mode is not supported for {get_settings().config.git_provider} for now")
        self.ai_handler = ai_handler()
        self.ai_handler.main_pr_language = self.main_language
        self.patches_diff = None
        self.remaining_files_list = []
        self.deleted_files_list = []
        self.prediction = None
        self.candidate_verification_artifact = None
        self.verified_review_data = None
        self.specialist_shadow_input = None
        self.specialist_shadow_result = None
        self._specialists_started = False
        question_str, answer_str = self._get_user_answers()
        self.pr_description, self.pr_description_files = (
            self.git_provider.get_pr_description(split_changes_walkthrough=True))
        if (self.pr_description_files and get_settings().get("config.is_auto_command", False) and
                get_settings().get("config.enable_ai_metadata", False)):
            add_ai_metadata_to_diff_files(self.git_provider, self.pr_description_files)
            get_logger().debug(f"AI metadata added to the this command")
        else:
            get_settings().set("config.enable_ai_metadata", False)
            get_logger().debug(f"AI metadata is disabled for this command")

        bugs_only = self.review_profile == "bugs_only"
        self.ci_failure_context = (
            self.git_provider.get_ci_failure_context()
            if bugs_only
            else {"status": "not_requested", "failures": []}
        )
        if not isinstance(self.ci_failure_context, dict):
            self.ci_failure_context = {"status": "unavailable", "failures": []}
        ci_failures = self.ci_failure_context.get("failures")
        if not isinstance(ci_failures, list):
            ci_failures = []
            self.ci_failure_context["failures"] = ci_failures
        self.ci_failure_evidence_by_name = {}
        for failure in ci_failures:
            if not isinstance(failure, dict):
                continue
            name = str(failure.get("name") or "").strip().casefold()
            if not name:
                continue
            evidence = " ".join((str(failure.get("title") or ""), str(failure.get("summary") or ""))).strip()
            self.ci_failure_evidence_by_name.setdefault(name, []).append(evidence)
        self.vars = {
            "title": self.git_provider.pr.title,
            "branch": self.git_provider.get_pr_branch(),
            "description": self.pr_description,
            "language": self.main_language,
            "diff": "",  # empty diff for initial calculation
            "num_pr_files": self.git_provider.get_num_of_files(),
            "num_max_findings": get_settings().pr_reviewer.num_max_findings,
            "bugs_only": bugs_only,
            "require_score": not bugs_only and get_settings().pr_reviewer.require_score_review,
            "require_tests": not bugs_only and get_settings().pr_reviewer.require_tests_review,
            "require_estimate_effort_to_review": (
                not bugs_only and get_settings().pr_reviewer.require_estimate_effort_to_review
            ),
            "require_estimate_contribution_time_cost": (
                not bugs_only and get_settings().pr_reviewer.require_estimate_contribution_time_cost
            ),
            'require_can_be_split_review': not bugs_only and get_settings().pr_reviewer.require_can_be_split_review,
            'require_security_review': not bugs_only and get_settings().pr_reviewer.require_security_review,
            'require_todo_scan': not bugs_only and get_settings().pr_reviewer.get("require_todo_scan", False),
            'question_str': question_str,
            'answer_str': answer_str,
            "extra_instructions": get_settings().pr_reviewer.extra_instructions,
            "skills_context": get_skills_context(),
            "repo_context": build_repo_context(self.git_provider),
            "commit_messages_str": self.git_provider.get_commit_messages(),
            "custom_labels": "",
            "enable_custom_labels": not bugs_only and get_settings().config.enable_custom_labels,
            "is_ai_metadata":  get_settings().get("config.enable_ai_metadata", False),
            "related_tickets": [] if bugs_only else get_settings().get('related_tickets', []),
            'duplicate_prompt_examples': get_settings().config.get('duplicate_prompt_examples', False),
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "enable_candidate_verification": self._candidate_verification_enabled(),
            "ci_failure_context": json.dumps(self.ci_failure_context, ensure_ascii=False),
        }

        self.token_handler = TokenHandler(
            self.git_provider.pr,
            self.vars,
            get_settings().pr_review_prompt.system,
            get_settings().pr_review_prompt.user
        )

    def parse_incremental(self, args: List[str]):
        is_incremental = False
        if args and len(args) >= 1:
            arg = args[0]
            if arg == "-i":
                is_incremental = True
        incremental = IncrementalPR(is_incremental)
        return incremental

    async def run(self) -> None:
        init_run_details()
        record_review_profile(self._review_profile())
        progress_response = None
        review_failed = False
        try:
            if not self.git_provider.get_files():
                get_logger().info(f"PR has no files: {self.pr_url}, skipping review")
                return None

            if self.incremental.is_incremental:
                can_run = self._can_run_incremental_review()
                # If the gate disabled incremental (e.g., commits_range is None), fall through to full review.
                if not can_run and self.incremental.is_incremental:
                    return None

            # if isinstance(self.args, list) and self.args and self.args[0] == 'auto_approve':
            #     get_logger().info(f'Auto approve flow PR: {self.pr_url} ...')
            #     self.auto_approve_logic()
            #     return None

            get_logger().info(f'Reviewing PR: {self.pr_url} ...')
            relevant_configs = {'pr_reviewer': dict(get_settings().pr_reviewer),
                                'config': dict(get_settings().config)}
            get_logger().debug("Relevant configs", artifacts=relevant_configs)

            # ticket extraction if exists
            if self._review_profile() != "bugs_only":
                await extract_and_cache_pr_tickets(self.git_provider, self.vars)

            if (
                self.incremental.is_incremental
                and hasattr(self.git_provider, "unreviewed_files_map")
                and not self.git_provider.unreviewed_files_map
            ):
                get_logger().info(f"Incremental review is enabled for {self.pr_url} but there are no new files")
                previous_review_url = ""
                if hasattr(self.git_provider, "previous_review") and self.git_provider.previous_review is not None:
                    previous_review_url = getattr(self.git_provider.previous_review, "html_url", "") or ""
                if get_settings().config.publish_output:
                    self.git_provider.publish_comment(f"Incremental Review Skipped\n"
                                    f"No files were changed since the [previous PR Review]({previous_review_url})")
                return None

            if get_settings().config.publish_output and not get_settings().config.get('is_auto_command', False):
                progress_response = self.git_provider.publish_comment("Preparing review...", is_temporary=True)

            await retry_with_fallback_models(self._prepare_prediction, model_type=ModelType.REGULAR)
            if not self.prediction:
                return None
            if self._candidate_verification_enabled():
                await self._run_candidate_verification()

            pr_review = self._prepare_pr_review()
            get_logger().debug(f"PR output", artifact=pr_review)

            should_publish = get_settings().config.publish_output and self._should_publish_review_no_suggestions(pr_review)
            if not should_publish:
                self._clear_stale_persistent_bugs_only_review()
                reason = "Review output is not published"
                if self._candidate_verification_blocks_publication():
                    reason += ": candidate verification did not complete successfully."
                elif get_settings().config.publish_output:
                    reason += ": no major issues detected."
                get_logger().info(reason)
                get_settings().data = {"artifact": pr_review}
                return

            # publish the review
            # Providers that support it (GitLab) can post the review's final comment as a resolvable thread.
            # This intent applies to the review only - never to status comments or the output of other tools.
            review_thread_kwargs = {"as_thread": True} if self.git_provider.should_publish_review_as_thread() else {}
            if get_settings().pr_reviewer.persistent_comment and not self.incremental.is_incremental:
                final_update_message = get_settings().pr_reviewer.final_update_message
                identity_marker = (
                    PRReviewIdentity.BUGS_ONLY.value
                    if self._review_profile() == "bugs_only"
                    else PRReviewIdentity.REGULAR.value
                )
                self.git_provider.publish_persistent_comment(
                    pr_review,
                    initial_header=pr_review.split("\n", 1)[0],
                    update_header=True,
                    final_update_message=final_update_message,
                    name="bugs-only review" if self._review_profile() == "bugs_only" else "review",
                    identity_marker=identity_marker,
                    legacy_initial_header=(
                        None
                        if self._review_profile() == "bugs_only"
                        else f"{PRReviewHeader.REGULAR.value} 🔍"
                    ),
                    **review_thread_kwargs,
                )
            else:
                if self.git_provider.supports_review_comment_identity() is True:
                    identity_marker = (
                        (
                            PRReviewIdentity.BUGS_ONLY_INCREMENTAL.value
                            if self._review_profile() == "bugs_only"
                            else PRReviewIdentity.FULL_INCREMENTAL.value
                        )
                        if self.incremental.is_incremental
                        else (
                            PRReviewIdentity.BUGS_ONLY.value
                            if self._review_profile() == "bugs_only"
                            else PRReviewIdentity.REGULAR.value
                        )
                    )
                    pr_review = add_pr_review_identity(pr_review, identity_marker)
                self.git_provider.publish_comment(pr_review, **review_thread_kwargs)
        except Exception as e:
            review_failed = True
            get_logger().error(f"Failed to review PR: {e}")
            if get_settings().config.get("propagate_tool_errors", False):
                raise
        finally:
            if progress_response is not None:
                try:
                    self.git_provider.remove_comment(progress_response)
                except Exception as e:
                    get_logger().exception(f"Failed to remove review progress comment, error: {e}")
            if (review_failed and get_settings().config.publish_output and
                    not get_settings().config.get("is_auto_command", False)):
                try:
                    self.git_provider.publish_comment("Failed to review PR")
                except Exception as e:
                    get_logger().exception(f"Failed to publish review failure result, error: {e}")

    def _should_publish_review_no_suggestions(self, pr_review: str) -> bool:
        if self._candidate_verification_blocks_publication():
            return False
        if self._review_profile() == "bugs_only":
            return bool(pr_review.strip())
        return get_settings().pr_reviewer.get('publish_output_no_suggestions', True) or "No major issues detected" not in pr_review

    def _review_profile(self) -> str:
        """Return the selected profile, defaulting legacy/test instances to full review."""
        return getattr(self, "review_profile", "full")

    def _clear_stale_persistent_bugs_only_review(self) -> None:
        """Remove a prior persistent defect summary after a clean bugs-only rerun."""
        if (self._candidate_verification_blocks_publication() or
                self._review_profile() != "bugs_only" or not get_settings().config.publish_output or
                not get_settings().pr_reviewer.persistent_comment or self.incremental.is_incremental):
            return
        self.git_provider.clear_persistent_review(
            identity_marker=PRReviewIdentity.BUGS_ONLY.value,
            name="bugs-only review",
        )

    async def _prepare_prediction(self, model: str) -> None:
        output = get_pr_diff(self.git_provider,
                             self.token_handler,
                             model,
                             add_line_numbers_to_hunks=True,
                             disable_extra_lines=False,
                             return_remaining_files=True,
                             return_deleted_files=True,)
        if isinstance(output, PRDiffCoverage):
            self.patches_diff = output.diff
            self.remaining_files_list = output.remaining_files
            self.deleted_files_list = output.deleted_files
        else:
            self.patches_diff = output
            self.remaining_files_list = []
            self.deleted_files_list = []

        if self.patches_diff:
            get_logger().debug(f"PR diff", diff=self.patches_diff)
            if specialists_enabled() and not getattr(self, "_specialists_started", False):
                await self._run_shadow_specialists_once()
            self.prediction = await self._get_prediction(model)
            self._reject_unparsable_prediction(model)
        else:
            get_logger().warning(f"Empty diff for PR: {self.pr_url}")
            self.prediction = None

    async def _run_shadow_specialists_once(self) -> None:
        """Run the configured shadow batch once without changing review inputs or output."""

        self._specialists_started = True
        try:
            pipeline = load_specialist_pipeline_config()
            snapshot_context = get_specialist_snapshot_context()
            if snapshot_context is not None:
                snapshot = snapshot_context.snapshot
                head_sha = snapshot.snapshot_id
                current_identity = snapshot_context.current_snapshot_id
            else:
                snapshot = None
                try:
                    head_sha = self.git_provider.get_pr_head_sha(refresh=False)
                except Exception as exc:
                    head_sha = None
                    get_logger().warning(
                        "Could not read a stable provider head identity for specialist shadow mode",
                        artifact={"error_class": type(exc).__name__},
                    )
                if not head_sha:
                    self.specialist_shadow_result = unavailable_specialist_batch(
                        pipeline,
                        failure_reason="stable_head_identity_unavailable",
                    )
                    get_logger().info(
                        "Specialist shadow telemetry",
                        artifact=self.specialist_shadow_result.to_dict(),
                    )
                    get_logger().warning(
                        "Specialist shadow batch is unavailable because the provider has no stable head identity"
                    )
                    return
                def current_identity():
                    return self.git_provider.get_pr_head_sha(refresh=True)
            specialist_input = build_specialist_input(
                title=self.vars["title"],
                description=self.pr_description,
                diff_files=self.git_provider.get_diff_files() or [],
                head_sha=head_sha,
                snapshot=snapshot,
                allowed_change_labels=pipeline.allowed_change_labels,
            )
            self.specialist_shadow_input = specialist_input
            self.specialist_shadow_result = await run_shadow_specialists(
                specialist_input,
                pipeline,
                self.ai_handler,
                current_identity=current_identity,
            )
            get_logger().info(
                "Specialist shadow telemetry",
                artifact=self.specialist_shadow_result.to_dict(),
            )
        except Exception as exc:
            # Shadow infrastructure is observational. Configuration/provider failures
            # remain telemetry and can never block or alter the ordinary review.
            get_logger().warning(
                "Specialist shadow batch failed; continuing the ordinary review",
                artifact={"error_class": type(exc).__name__},
            )

    def _reject_unparsable_prediction(self, model: str) -> None:
        """Treat a prediction that will not parse as a failure of this model.

        A model can answer promptly and still emit YAML the parser cannot read (an
        unquoted colon inside a summary is enough). Raising here, while still inside
        retry_with_fallback_models, lets the next model answer instead - previously the
        parse happened after all retries, so an unparsable answer skipped the fallback
        entirely and the run published nothing.

        Args:
            model: The model that produced self.prediction, for the log line.

        Raises:
            ValueError: When the prediction is missing or does not yield review data.
        """
        if not self.prediction or not self.prediction.strip():
            raise ValueError(f"Model {model} returned an empty prediction")
        try:
            data = load_yaml(
                self.prediction.strip(),
                keys_fix_yaml=["ticket_compliance_check", "estimated_effort_to_review_[1-5]:",
                               "security_concerns:", "key_issues_to_review:",
                               "relevant_file:", "relevant_line:", "suggestion:"],
                first_key='review', last_key='security_concerns',
            )
        except Exception as e:
            raise ValueError(f"Model {model} returned unparsable output: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get('review'), dict) or not data['review']:
            raise ValueError(f"Model {model} returned output without a non-empty 'review' mapping")

    async def _get_prediction(self, model: str) -> str:
        """
        Generate an AI prediction for the pull request review.

        Args:
            model: A string representing the AI model to be used for the prediction.

        Returns:
            A string representing the AI prediction for the pull request review.
        """
        variables = copy.deepcopy(self.vars)
        variables["diff"] = self.patches_diff  # update diff

        environment = Environment(undefined=StrictUndefined)
        system_prompt = environment.from_string(get_settings().pr_review_prompt.system).render(variables)
        user_prompt = environment.from_string(get_settings().pr_review_prompt.user).render(variables)

        response, finish_reason = await self.ai_handler.chat_completion(
            model=model,
            temperature=get_settings().config.temperature,
            system=system_prompt,
            user=user_prompt
        )

        return response

    def _changed_lines_by_file(self) -> dict[str, set[int]]:
        """Return added line numbers for each diff file, using unified-diff hunks."""
        changed_lines = {}
        for file in self.git_provider.get_diff_files() or []:
            filename = (getattr(file, "filename", "") or "").strip()
            patch = getattr(file, "patch", "") or ""
            if not filename or not patch:
                continue
            file_lines = set()
            new_line = None
            for patch_line in iter_git_patch_lines(patch):
                header = _HUNK_HEADER_RE.match(patch_line)
                if header:
                    new_line = int(header.group(1))
                    continue
                if new_line is None or patch_line.startswith("\\ No newline at end of file"):
                    continue
                if patch_line.startswith("+") and not patch_line.startswith("+++"):
                    file_lines.add(new_line)
                    new_line += 1
                elif patch_line.startswith("-") and not patch_line.startswith("---"):
                    continue
                else:
                    new_line += 1
            changed_lines[filename] = file_lines
            changed_lines.setdefault(filename.lstrip("/"), file_lines)
        return changed_lines

    @staticmethod
    def _strict_bool(value) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        return None

    @staticmethod
    def _specific_ci_terms(value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", value.casefold())
            if len(term) >= 4 and term not in _GENERIC_CI_EVIDENCE_TERMS
        }

    def _ci_failure_evidences_same_defect(self, issue: dict, matching_ci_failure: str) -> bool:
        issue_text = " ".join(
            str(issue.get(field) or "")
            for field in ("issue_content", "trigger", "impact", "root_cause")
        )
        issue_terms = self._specific_ci_terms(issue_text)
        if len(issue_terms) < 2:
            return False
        evidence_by_name = getattr(self, "ci_failure_evidence_by_name", {})
        for evidence in evidence_by_name.get(matching_ci_failure, []):
            if len(issue_terms & self._specific_ci_terms(evidence)) >= 2:
                return True
        return False

    def _normalize_bugs_only_review(self, data: dict) -> dict:
        """Keep only complete, changed-line defect reports and collapse shared root causes."""
        if self._review_profile() != "bugs_only":
            return data
        if getattr(self, "candidate_verification_artifact", None) is not None:
            issues = (data.get("review") or {}).get("key_issues_to_review")
            if not isinstance(issues, list):
                issues = []
            return {"review": {"key_issues_to_review": issues[:get_settings().pr_reviewer.num_max_findings]}}

        issues = (data.get("review") or {}).get("key_issues_to_review")
        if not isinstance(issues, list):
            issues = []
        changed_lines = self._changed_lines_by_file()
        normalized_issues = []
        seen_root_causes = set()
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            finding_type = str(issue.get("finding_type") or "").strip().lower()
            if finding_type not in _BUG_FINDING_HEADERS:
                continue
            matching_ci_failure = str(issue.get("matching_ci_failure") or "").strip().casefold()
            if (self._strict_bool(issue.get("duplicates_ci_failure")) is True and
                    self._ci_failure_evidences_same_defect(issue, matching_ci_failure)):
                continue

            relevant_file = str(issue.get("relevant_file") or "").strip()
            issue_content = str(issue.get("issue_content") or "").strip()
            trigger = str(issue.get("trigger") or "").strip()
            impact = str(issue.get("impact") or "").strip()
            root_cause = " ".join(str(issue.get("root_cause") or "").split())
            try:
                start_line = int(str(issue.get("start_line", "")).strip())
                end_line = int(str(issue.get("end_line", "")).strip())
            except ValueError:
                continue
            file_changed_lines = changed_lines.get(relevant_file) or changed_lines.get(relevant_file.lstrip("/"))
            if (not relevant_file or not issue_content or not trigger or not impact or not root_cause or
                    start_line < 1 or end_line < start_line or not file_changed_lines or
                    not any(line in file_changed_lines for line in range(start_line, end_line + 1))):
                continue

            root_cause_key = root_cause.casefold()
            if root_cause_key in seen_root_causes:
                continue
            seen_root_causes.add(root_cause_key)
            normalized_issues.append({
                "relevant_file": relevant_file,
                "issue_header": _BUG_FINDING_HEADERS[finding_type],
                "issue_content": f"{issue_content}\n\n**Trigger:** {trigger}\n\n**Impact:** {impact}",
                "start_line": start_line,
                "end_line": end_line,
            })
            if len(normalized_issues) >= get_settings().pr_reviewer.num_max_findings:
                break
        return {"review": {"key_issues_to_review": normalized_issues}}

    @staticmethod
    def _candidate_verification_enabled() -> bool:
        value = get_settings().pr_reviewer.get("enable_candidate_verification", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _candidate_verification_blocks_publication(self) -> bool:
        """Fail closed when enabled candidate verification did not reach a publishable result."""
        artifact = getattr(self, "candidate_verification_artifact", None)
        if not isinstance(artifact, dict):
            return False
        if "publication_safe" in artifact:
            return artifact.get("publication_safe") is not True
        status = artifact.get("status")
        if status in {"complete", "no_candidates"}:
            return False
        return not (status == "partial" and int(artifact.get("verified_count") or 0) > 0)

    @staticmethod
    def _verification_response_contract_error(
        candidates: list[dict], verification_data: dict
    ) -> Optional[str]:
        """Return a source-free error when a verifier response is not complete and unambiguous."""
        if not isinstance(verification_data, dict):
            return "invalid_response"
        verification = verification_data.get("verification")
        if not isinstance(verification, dict):
            return "invalid_verification"
        decisions = verification.get("decisions")
        if not isinstance(decisions, list):
            return "invalid_decisions"
        expected_ids = {candidate["candidate_id"] for candidate in candidates}
        seen_ids = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                return "invalid_decision"
            candidate_id = str(decision.get("candidate_id") or "").strip()
            if candidate_id not in expected_ids:
                return "unknown_candidate"
            if candidate_id in seen_ids:
                return "duplicate_decision"
            if str(decision.get("verdict") or "").strip().lower() not in {"verified", "rejected"}:
                return "invalid_verdict"
            if str(decision.get("verdict") or "").strip().lower() == "verified":
                required_text = (
                    "issue_header", "issue_content", "trigger", "impact",
                )
                if (
                    safe_repo_path(decision.get("relevant_file")) is None
                    or any(
                        not isinstance(decision.get(key), str)
                        or not decision[key].strip()
                        for key in required_text
                    )
                ):
                    return "invalid_verified_decision"
                start_line = decision.get("start_line")
                end_line = decision.get("end_line")
                if (
                    not isinstance(start_line, int)
                    or isinstance(start_line, bool)
                    or not isinstance(end_line, int)
                    or isinstance(end_line, bool)
                    or start_line < 1
                    or end_line < start_line
                ):
                    return "invalid_verified_decision"
                evidence_paths = decision.get("evidence_paths")
                if (
                    not isinstance(evidence_paths, list)
                    or not evidence_paths
                    or any(safe_repo_path(path) is None for path in evidence_paths)
                ):
                    return "invalid_verified_decision"
            seen_ids.add(candidate_id)
        if seen_ids != expected_ids:
            return "missing_decision"
        return None

    @staticmethod
    def _candidate_specialist_prioritization_enabled() -> bool:
        value = get_settings().pr_reviewer.get(
            "candidate_verification_consume_specialist_prioritization", False
        )
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _candidate_verifier_model_route(self, config) -> AIModelRoute:
        """Build an immutable verifier route without inheriting an incompatible Azure deployment."""

        def string_tuple(value, key: str) -> tuple[str, ...]:
            if value in (None, "", []):
                return ()
            if isinstance(value, str):
                values = [part.strip() for part in value.split(",")]
            elif isinstance(value, (list, tuple)):
                values = [str(part).strip() for part in value]
            else:
                raise ValueError(f"{key} must be a string list")
            if any(not part for part in values):
                raise ValueError(f"{key} cannot contain blank entries")
            return tuple(values)

        def deployment_tuple(value, key: str) -> tuple[Optional[str], ...]:
            if value in (None, "", []):
                return ()
            if isinstance(value, str):
                values = value.split(",")
            elif isinstance(value, (list, tuple)):
                values = value
            else:
                raise ValueError(f"{key} must be a string list")
            return tuple(str(part).strip() or None for part in values)

        settings = get_settings()
        primary_model = str(settings.config.model).strip()
        configured_model = str(config.get("candidate_verification_model", "") or "").strip()
        model = configured_model or primary_model
        if not model:
            raise ValueError("candidate verifier model cannot be blank")
        fallback_models = string_tuple(
            config.get("candidate_verification_fallback_models", []),
            "candidate_verification_fallback_models",
        )

        global_deployment = str(
            settings.get("openai.deployment_id", "") or ""
        ).strip() or None
        explicit_deployment = str(
            config.get("candidate_verification_deployment", "") or ""
        ).strip() or None
        azure_route = getattr(self.ai_handler, "azure", False) is True or global_deployment is not None
        if explicit_deployment is not None:
            deployment = explicit_deployment
        elif not configured_model or model == primary_model:
            deployment = global_deployment
        elif azure_route:
            raise ValueError(
                "candidate_verification_deployment is required when the Azure verifier model differs"
            )
        else:
            deployment = None

        fallback_deployments = deployment_tuple(
            config.get("candidate_verification_fallback_deployments", []),
            "candidate_verification_fallback_deployments",
        )
        if fallback_deployments and len(fallback_deployments) != len(fallback_models):
            raise ValueError(
                "candidate_verification_fallback_deployments must match fallback models"
            )
        if not fallback_deployments:
            if azure_route and fallback_models:
                raise ValueError(
                    "Azure verifier fallback models require matching fallback deployments"
                )
            fallback_deployments = (None,) * len(fallback_models)
        deployments = (deployment, *fallback_deployments)
        if azure_route and any(item is None for item in deployments):
            raise ValueError("every Azure verifier model requires a deployment")

        configured_output_cap = config.get("candidate_verification_max_output_tokens", 0)
        try:
            output_cap = int(configured_output_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_verification_max_output_tokens must be an integer") from exc
        if output_cap < 0:
            raise ValueError("candidate_verification_max_output_tokens cannot be negative")
        if output_cap == 0:
            try:
                output_cap = int(settings.config.get("max_output_tokens", 0))
            except (TypeError, ValueError):
                output_cap = 0

        return AIModelRoute(
            models=(model, *fallback_models),
            deployments=deployments,
            timeout_seconds=max(
                0.001, float(config.get("candidate_verification_timeout_seconds", 10))
            ),
            model_retries=1,
            provider_retries=0,
            max_output_tokens=output_cap or None,
            attribution="candidate_verification",
        )

    def _parse_review_prediction(self) -> dict:
        return load_yaml(
            self.prediction.strip(),
            keys_fix_yaml=["ticket_compliance_check", "estimated_effort_to_review_[1-5]:",
                           "security_concerns:", "key_issues_to_review:", "relevant_file:",
                           "relevant_line:", "suggestion:"],
            first_key="review",
            last_key="security_concerns",
        )

    async def _run_candidate_verification(self) -> None:
        """Verify review candidates against bounded base-branch repository evidence."""
        config = get_settings().pr_reviewer
        consume_specialist_prioritization = self._candidate_specialist_prioritization_enabled()
        artifact = {
            "enabled": True,
            "status": "initializing",
            "model_calls": 0,
            "publication_safe": False,
            "specialist_prioritization": {
                "status": "pending" if consume_specialist_prioritization else "disabled"
            },
        }
        verifier_started = None
        details_before = None
        self.candidate_verification_artifact = artifact
        try:
            review_data = self._parse_review_prediction()
            if not isinstance(review_data, dict) or not isinstance(review_data.get("review"), dict):
                artifact.update({"status": "candidate_parse_failed", "verified_count": 0})
                return
            self.verified_review_data = copy.deepcopy(review_data)
            self.verified_review_data["review"]["key_issues_to_review"] = []

            capability = getattr(self.git_provider, "supports_repo_file_fetching", None)
            if not callable(capability) or not capability():
                artifact.update({"status": "unsupported_provider", "verified_count": 0})
                return

            budgets = VerificationBudgets(
                max_candidates=max(0, int(config.get("candidate_verification_max_candidates", 3))),
                max_files=max(0, int(config.get("candidate_verification_max_files", 6))),
                max_lines_per_file=max(0, int(config.get("candidate_verification_max_lines_per_file", 160))),
                max_total_lines=max(0, int(config.get("candidate_verification_max_total_lines", 600))),
                max_context_tokens=max(0, int(config.get("candidate_verification_max_context_tokens", 6000))),
                timeout_seconds=max(0.0, float(config.get("candidate_verification_timeout_seconds", 10))),
            )
            sensitive_globs = config.get("candidate_verification_sensitive_path_globs", []) or []
            if isinstance(sensitive_globs, str):
                sensitive_globs = [sensitive_globs]
            diff_files = self.git_provider.get_diff_files()
            candidates, candidate_rejections = prepare_candidates(
                review_data,
                diff_files,
                sensitive_globs,
                budgets.max_candidates,
                max_sensitive_candidates=max(
                    0, int(config.get("candidate_verification_max_sensitive_candidates", 6))
                ),
            )
            sensitive_candidates = [
                candidate for candidate in candidates if candidate.get("sensitive_path")
            ]
            sensitive_rejections = [
                rejection for rejection in candidate_rejections
                if rejection.get("sensitive_path")
            ]
            sensitive_overflow = next((
                rejection for rejection in sensitive_rejections
                if rejection.get("reason") == "sensitive_audit_budget_exhausted"
            ), {})
            sensitive_invalid_count = sum(
                1 for rejection in sensitive_rejections
                if rejection.get("reason") != "sensitive_audit_budget_exhausted"
            )
            sensitive_selected_count = int(
                sensitive_overflow.get(
                    "selected_count", len(sensitive_candidates) + sensitive_invalid_count
                )
            )
            sensitive_omitted_count = int(sensitive_overflow.get("omitted_count", 0))
            sensitive_total_count = int(
                sensitive_overflow.get(
                    "total_count", sensitive_selected_count + sensitive_omitted_count
                )
            )
            sensitive_coverage_incomplete = bool(sensitive_rejections)
            artifact["sensitive_audit_coverage"] = {
                "status": "incomplete" if sensitive_coverage_incomplete else "complete",
                "budget": max(
                    0, int(config.get("candidate_verification_max_sensitive_candidates", 6))
                ),
                "total_count": sensitive_total_count,
                "selected_count": sensitive_selected_count,
                "candidate_count": len(sensitive_candidates),
                "omitted_count": sensitive_omitted_count,
                "unavailable_count": sensitive_invalid_count,
            }
            if consume_specialist_prioritization:
                specialist_input = getattr(self, "specialist_shadow_input", None)
                prioritization = validated_specialist_prioritization(
                    getattr(self, "specialist_shadow_result", None), specialist_input
                )
                if prioritization is None:
                    artifact["specialist_prioritization"] = {"status": "validated_output_unavailable"}
                else:
                    candidates, prioritization_artifact = apply_specialist_prioritization(
                        candidates, prioritization, specialist_input
                    )
                    artifact["specialist_prioritization"] = prioritization_artifact
            artifact.update({
                "candidate_count": len(candidates),
                "candidate_rejections": candidate_rejections,
                "verified_count": 0,
            })
            if not candidates:
                if sensitive_coverage_incomplete:
                    artifact.update({
                        "status": "sensitive_audit_coverage_incomplete",
                        "publication_safe": False,
                    })
                else:
                    artifact.update({"status": "no_candidates", "publication_safe": True})
                return

            try:
                verifier_route = self._candidate_verifier_model_route(config)
            except (TypeError, ValueError):
                artifact.update({
                    "status": "verifier_route_invalid",
                    "failure": "invalid_model_route",
                    "verified_count": 0,
                })
                return
            model = verifier_route.models[0]
            artifact["model"] = model
            encoder = TokenEncoder.get_token_encoder(model)
            runtime_data = get_settings().get("data", {}) or {}
            static_evidence = runtime_data.get("static_analysis_evidence", []) if isinstance(runtime_data, dict) else []
            evidence, retrieval_artifact = await retrieve_evidence(
                self.git_provider,
                candidates,
                budgets,
                static_evidence,
                diff_files=diff_files,
                token_counter=lambda value: len(encoder.encode(value, disallowed_special=())),
                prefer_pr_head=bool(
                    getattr(getattr(self, "incremental", None), "is_incremental", False)
                ),
            )
            artifact["retrieval"] = retrieval_artifact
            if int(config.get("candidate_verification_max_model_calls", 1)) < 1:
                artifact["status"] = "model_call_budget_exhausted"
                return

            environment = Environment(
                autoescape=select_autoescape(default_for_string=False),
                undefined=StrictUndefined,
            )
            verification_prompt = get_settings().pr_review_verification_prompt
            route_encoders = {
                route_model: TokenEncoder.get_token_encoder(route_model)
                for route_model in verifier_route.models
            }
            try:
                route_completion_reserves = {
                    route_model: get_effective_litellm_output_token_cap(
                        route_model,
                        verifier_route.max_output_tokens,
                        require_bounded_reasoning=True,
                    )
                    for route_model in verifier_route.models
                }
                if any(cap is None for cap in route_completion_reserves.values()):
                    verifier_route = AIModelRoute(
                        models=verifier_route.models,
                        deployments=verifier_route.deployments,
                        timeout_seconds=verifier_route.timeout_seconds,
                        model_retries=verifier_route.model_retries,
                        provider_retries=verifier_route.provider_retries,
                        max_output_tokens=OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD,
                        attribution=verifier_route.attribution,
                    )
                    route_completion_reserves = {
                        route_model: get_effective_litellm_output_token_cap(
                            route_model,
                            verifier_route.max_output_tokens,
                            require_bounded_reasoning=True,
                        )
                        for route_model in verifier_route.models
                    }
            except ValueError:
                artifact.update({
                    "status": "verifier_route_invalid",
                    "failure": "invalid_output_budget",
                    "verified_count": 0,
                })
                return
            route_model_max_tokens = {
                route_model: get_max_tokens(route_model)
                for route_model in verifier_route.models
            }
            route_prompt_limits = {
                route_model: max(
                    0,
                    route_model_max_tokens[route_model]
                    - route_completion_reserves[route_model],
                )
                for route_model in verifier_route.models
            }
            model_max_tokens = min(route_model_max_tokens.values())
            reserved_completion_tokens = max(route_completion_reserves.values())
            max_prompt_tokens = min(route_prompt_limits.values())

            def render_prompts(evidence_fraction: float, changed_diff_fraction: float) -> tuple[str, str, int]:
                variables = {
                    "verification_payload": render_verification_payload(
                        candidates,
                        self.patches_diff or "",
                        evidence,
                        content_fraction=evidence_fraction,
                        changed_diff_fraction=changed_diff_fraction,
                    ),
                }
                rendered_system = environment.from_string(verification_prompt.system).render(variables)
                rendered_user = environment.from_string(verification_prompt.user).render(variables)
                prompt_tokens = max(
                    len(route_encoder.encode(rendered_system, disallowed_special=()))
                    + len(route_encoder.encode(rendered_user, disallowed_special=()))
                    for route_encoder in route_encoders.values()
                )
                return rendered_system, rendered_user, prompt_tokens

            full_system_prompt, full_user_prompt, full_prompt_tokens = render_prompts(1.0, 1.0)
            evidence_fraction = 1.0
            changed_diff_fraction = 1.0
            system_prompt = full_system_prompt
            user_prompt = full_user_prompt
            prompt_tokens = full_prompt_tokens
            if full_prompt_tokens > max_prompt_tokens:
                diff_free_system, diff_free_user, diff_free_tokens = render_prompts(1.0, 0.0)
                if diff_free_tokens <= max_prompt_tokens:
                    system_prompt = diff_free_system
                    user_prompt = diff_free_user
                    prompt_tokens = diff_free_tokens
                    changed_diff_fraction = 0.0
                    lower = 0.0
                    upper = 1.0
                    for _ in range(18):
                        midpoint = (lower + upper) / 2
                        candidate_system, candidate_user, candidate_tokens = render_prompts(1.0, midpoint)
                        if candidate_tokens <= max_prompt_tokens:
                            lower = midpoint
                            changed_diff_fraction = midpoint
                            system_prompt = candidate_system
                            user_prompt = candidate_user
                            prompt_tokens = candidate_tokens
                        else:
                            upper = midpoint
                else:
                    system_prompt, user_prompt, prompt_tokens = render_prompts(0.0, 0.0)
                    evidence_fraction = 0.0
                    changed_diff_fraction = 0.0
                    if prompt_tokens > max_prompt_tokens:
                        artifact.update({
                            "status": "prompt_budget_exhausted",
                            "prompt_budget": {
                                "model_max_tokens": model_max_tokens,
                                "reserved_completion_tokens": reserved_completion_tokens,
                                "max_prompt_tokens": max_prompt_tokens,
                                "prompt_tokens": prompt_tokens,
                                "truncated": True,
                            },
                        })
                        return
                    lower = 0.0
                    upper = 1.0
                    for _ in range(18):
                        midpoint = (lower + upper) / 2
                        candidate_system, candidate_user, candidate_tokens = render_prompts(midpoint, 0.0)
                        if candidate_tokens <= max_prompt_tokens:
                            lower = midpoint
                            evidence_fraction = midpoint
                            system_prompt = candidate_system
                            user_prompt = candidate_user
                            prompt_tokens = candidate_tokens
                        else:
                            upper = midpoint
            artifact["prompt_budget"] = {
                "model_max_tokens": model_max_tokens,
                "reserved_completion_tokens": reserved_completion_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "prompt_tokens": prompt_tokens,
                "route_count": len(verifier_route.models),
                "truncated": evidence_fraction < 1.0 or changed_diff_fraction < 1.0,
                "evidence_content_fraction": round(evidence_fraction, 4),
                "changed_diff_fraction": round(changed_diff_fraction, 4),
            }
            artifact["model_calls"] = 1
            verifier_started = time.monotonic()
            details = get_run_details()
            if details is not None:
                details_before = {
                    "prompt_tokens": details.prompt_tokens,
                    "completion_tokens": details.completion_tokens,
                    "total_tokens": details.total_tokens,
                    "total_cost_usd": details.total_cost_usd,
                    "known_cost_call_count": details.known_cost_call_count,
                }
            artifact["verifier_attempts"] = 0

            async def call_verifier(attempt_model: str):
                artifact["verifier_attempts"] += 1
                prediction_result = await self.ai_handler.chat_completion(
                    model=attempt_model,
                    temperature=get_settings().config.temperature,
                    system=system_prompt,
                    user=user_prompt,
                )
                artifact["model"] = attempt_model
                return prediction_result

            prediction, _ = await retry_with_fallback_models(
                call_verifier,
                model_route=verifier_route,
            )
            verification_data = load_yaml(
                prediction.strip(),
                keys_fix_yaml=["verification:", "decisions:", "candidate_id:", "relevant_file:",
                               "start_line:", "end_line:", "evidence_paths:"],
                first_key="verification",
                last_key="decisions",
            )
            response_error = self._verification_response_contract_error(
                candidates, verification_data
            )
            if response_error is not None:
                artifact.update({
                    "status": "verifier_response_invalid",
                    "failure": response_error,
                    "verified_count": 0,
                })
                return
            prompt_evidence = bounded_verification_evidence(evidence, evidence_fraction)
            verified_findings, decisions = apply_verification_decisions(
                candidates,
                prompt_evidence,
                verification_data,
                retrieval_requests=retrieval_artifact["requests"],
            )
            max_findings = max(0, int(config.get("num_max_findings", 3)))
            published_findings = verified_findings[:max_findings]
            self.verified_review_data["review"]["key_issues_to_review"] = published_findings
            verifier_claimed_ids = {
                str(decision.get("candidate_id") or "").strip()
                for decision in verification_data["verification"]["decisions"]
                if str(decision.get("verdict") or "").strip().lower() == "verified"
            }
            rejected_verified_claim = any(
                decision.get("candidate_id") in verifier_claimed_ids
                and decision.get("verdict") != "verified"
                for decision in decisions
            )
            verification_status = "partial" if (
                sensitive_coverage_incomplete
                or rejected_verified_claim
                or retrieval_artifact["budget_exhausted"]
                or any(
                    request["status"] not in {"retrieved", "satisfied_by_changed_head"}
                    for request in retrieval_artifact["requests"]
                )
            ) else "complete"
            artifact.update({
                "status": verification_status,
                "publication_safe": (
                    not sensitive_coverage_incomplete
                    and (verification_status == "complete" or bool(published_findings))
                ),
                "decisions": decisions,
                "verified_count": len(published_findings),
                "verifier_verified_count": len(verified_findings),
                "finding_limit_dropped": len(verified_findings) - len(published_findings),
                "rejected_count": len(candidates) - len(published_findings),
            })
        except Exception as exc:
            failure = exc.__cause__ if exc.__cause__ is not None else exc
            artifact.update({
                "status": "verifier_failed",
                "failure": type(failure).__name__,
                "verified_count": 0,
            })
            get_logger().error(
                "Candidate verification failed", artifact=telemetry_safe_artifact(artifact)
            )
        finally:
            if verifier_started is not None:
                artifact["verifier_latency_seconds"] = round(time.monotonic() - verifier_started, 3)
                details = get_run_details()
                if details is not None and details_before is not None:
                    artifact["verifier_usage"] = {
                        "prompt_tokens": details.prompt_tokens - details_before["prompt_tokens"],
                        "completion_tokens": details.completion_tokens - details_before["completion_tokens"],
                        "total_tokens": details.total_tokens - details_before["total_tokens"],
                    }
                    known_cost_delta = details.known_cost_call_count - details_before["known_cost_call_count"]
                    artifact["verifier_cost"] = {
                        "status": "complete" if known_cost_delta > 0 else "unavailable",
                        "usd": str(details.total_cost_usd - details_before["total_cost_usd"])
                        if known_cost_delta > 0 else None,
                    }
            get_logger().info("Candidate verification finished", artifact=telemetry_safe_artifact(artifact))

    def _prepare_pr_review(self) -> str:
        """
        Prepare the PR review by processing the AI prediction and generating a markdown-formatted text that summarizes
        the feedback.
        """
        data = copy.deepcopy(getattr(self, "verified_review_data", None)) or self._parse_review_prediction()

        if not isinstance(data, dict) or 'review' not in data:
            get_logger().exception("Failed to parse review data", artifact={"data": data})
            return ""
        data = self._normalize_bugs_only_review(data)

        structured_publisher = getattr(self.git_provider, "publish_structured_review", None)
        if callable(structured_publisher):
            # Deep-copy the data: dict(data) is shallow, so structured_data["review"]
            # would alias data["review"], which is mutated right below (key reordering).
            # Hand implementers an isolated snapshot, since the hook is provider-neutral
            # and a provider that defers serialization would observe the mutation.
            structured_data = copy.deepcopy(data)
            details = get_run_details()
            usage = {}
            if details is not None and details.has_token_usage:
                usage = {
                    "prompt_tokens": details.prompt_tokens,
                    "completion_tokens": details.completion_tokens,
                    "total_tokens": details.total_tokens,
                }
            structured_data["usage"] = usage
            structured_data["metadata"] = {
                "review_profile": self._review_profile(),
                "omitted_files": sorted(set(self.remaining_files_list)),
                "deleted_files": sorted(set(getattr(self, "deleted_files_list", []))),
            }
            if getattr(self, "candidate_verification_artifact", None) is not None:
                structured_data["candidate_verification"] = telemetry_safe_artifact(
                    self.candidate_verification_artifact
                )
            specialist_shadow_result = getattr(self, "specialist_shadow_result", None)
            if specialist_shadow_result is not None:
                structured_data["metadata"]["specialist_shadow"] = specialist_shadow_result.to_dict()
            structured_publisher(structured_data)

        if self._candidate_verification_blocks_publication():
            return ""

        github_action_output(data, 'review')

        # move data['review'] 'key_issues_to_review' key to the end of the dictionary
        if 'key_issues_to_review' in data['review']:
            key_issues_to_review = data['review'].pop('key_issues_to_review')
            data['review']['key_issues_to_review'] = key_issues_to_review

        if get_settings().config.publish_output and get_settings().pr_reviewer.get('inline_key_issues', False):
            data = self._publish_key_issues_as_inline_comments(data)

        if self._review_profile() == "bugs_only" and not (
                (data.get("review") or {}).get("key_issues_to_review")):
            return ""

        incremental_review_markdown_text = None
        # Add incremental review section
        if self.incremental.is_incremental:
            last_commit_url = f"{self.git_provider.get_pr_url()}/commits/" \
                              f"{self.git_provider.incremental.first_new_commit_sha}"
            incremental_review_markdown_text = f"Starting from commit {last_commit_url}"

        markdown_text = convert_to_markdown_v2(data, self.git_provider.is_supported("gfm_markdown"),
                                            incremental_review_markdown_text,
                                               git_provider=self.git_provider,
                                               files=self.git_provider.get_diff_files(),
                                               review_profile=self._review_profile())

        if (self._review_profile() != "bugs_only" and self.remaining_files_list and
                get_settings().pr_reviewer.enable_review_coverage_footer):
            displayed_files = self.remaining_files_list[:MAX_REVIEW_COVERAGE_FILES]
            markdown_text += (
                "\n\n<hr>\n\n"
                "⚠️ **Review coverage:** The following files were not included in this review "
                "because of the token budget:\n"
                + "\n".join(f"- `{file}`" for file in displayed_files)
            )
            remaining_count = len(self.remaining_files_list) - len(displayed_files)
            if remaining_count:
                markdown_text += f"\n... and {remaining_count} more"

        # Add help text if gfm_markdown is supported
        if (self._review_profile() != "bugs_only" and self.git_provider.is_supported("gfm_markdown") and
                get_settings().pr_reviewer.enable_help_text):
            markdown_text += "<hr>\n\n<details> <summary><strong>💡 Tool usage guide:</strong></summary><hr> \n\n"
            markdown_text += HelpMessage.get_review_usage_guide()
            markdown_text += "\n</details>\n"

        # Output the relevant configurations if enabled
        if (self._review_profile() != "bugs_only" and
                get_settings().get('config', {}).get('output_relevant_configurations', False)):
            markdown_text += show_relevant_configurations(relevant_section='pr_reviewer')

        # Output the agent run details (model, tokens, time cost) if enabled
        if (self._review_profile() != "bugs_only" and
                get_settings().get('config', {}).get('output_run_details', False)):
            markdown_text += show_run_details(self.git_provider.is_supported("gfm_markdown"))

        # Add custom labels from the review prediction (effort, security)
        if self._review_profile() != "bugs_only":
            self.set_review_labels(data)

        if markdown_text == None or len(markdown_text) == 0:
            markdown_text = ""

        return markdown_text

    def _build_key_issue_comment(self, issue, diff_files: dict) -> Optional[dict]:
        if not isinstance(issue, dict):
            return None
        relevant_file = (issue.get("relevant_file") or "").strip()
        issue_content = _SUGGESTION_FENCE_RE.sub("```text", (issue.get("issue_content") or "").strip())
        issue_header = (issue.get("issue_header") or "").strip()
        if issue_header.lower() == "possible bug":
            issue_header = "Possible Issue"
        try:
            start_line = int(str(issue.get("start_line", 0)).strip())
            end_line = int(str(issue.get("end_line", 0)).strip())
        except ValueError:
            start_line, end_line = 0, 0

        if not relevant_file or not issue_content or start_line < 1 or end_line < start_line:
            get_logger().warning("Review finding has no usable location, keeping it in the summary",
                                 artifact={"relevant_file": relevant_file, "start_line": start_line,
                                           "end_line": end_line})
            return None
        if issue.get("side") == "old":
            get_logger().info(
                "Review finding targets deleted code, keeping its old-side location in the summary",
                artifact={"relevant_file": relevant_file, "start_line": start_line, "end_line": end_line},
            )
            return None

        file = diff_files.get(relevant_file) or diff_files.get(relevant_file.lstrip("/"))
        if file is None:
            get_logger().warning("Review finding points at a file that is not in the diff, "
                                 "keeping it in the summary", artifact={"relevant_file": relevant_file})
            return None
        if not file.head_file or end_line > len(split_git_file_lines(file.head_file)):
            get_logger().warning("Review finding points past the end of the file, keeping it in the summary",
                                 artifact={"relevant_file": relevant_file, "start_line": start_line,
                                           "end_line": end_line})
            return None

        relevant_file = file.filename.strip()
        body = f"**{issue_header}**\n\n{issue_content}" if issue_header else issue_content
        return {"body": body,
                "relevant_file": relevant_file,
                "relevant_lines_start": start_line,
                "relevant_lines_end": end_line,
                "fallback_to_pr_comment": False}

    def _can_verify_inline_key_issue_publication(self) -> bool:
        return can_verify_inline_comment_publication(self.git_provider)

    def _published_inline_key_issue_fingerprints(self, store: InlineCommentStore,
                                                 fingerprints: set[str]) -> set[str]:
        try:
            for body in self.git_provider.get_recent_inline_comment_bodies():
                store.add_body(body)
        except Exception as e:
            get_logger().warning(
                f"Inline key-issue publishing cannot verify new Azure DevOps threads, error: {e}; "
                "keeping findings in the review summary")
            return set()
        return {fingerprint for fingerprint in fingerprints if store.seen(fingerprint)}

    def _publish_key_issues_as_inline_comments(self, data: dict) -> dict:
        issues = (data.get("review") or {}).get("key_issues_to_review")
        if not isinstance(issues, list) or not issues:
            return data
        if not self._can_verify_inline_key_issue_publication():
            get_logger().info("Inline key-issue publishing is not verifiable for this provider; "
                              "keeping findings in the review summary")
            return data

        diff_files = {}
        for file in self.git_provider.get_diff_files() or []:
            if not file.filename:
                continue
            path = file.filename.strip()
            diff_files[path] = file
            diff_files.setdefault(path.lstrip("/"), file)
        store = get_inline_comment_store(self.git_provider)
        store.load()
        if store.load_failed:
            get_logger().warning("Inline key-issue publishing cannot verify existing Azure DevOps threads; "
                                 "keeping findings in the review summary")
            return data
        remaining_issues = []
        candidate_comments = {}
        candidate_issues = {}
        candidate_fingerprints = {}
        published = 0
        for issue in issues:
            try:
                comment = self._build_key_issue_comment(issue, diff_files)
                if comment is None:
                    remaining_issues.append(issue)
                    continue
                fingerprint = key_issue_fingerprint(comment["relevant_file"], comment["body"])
                if store.seen(fingerprint):
                    published += 1
                    continue
                location_fingerprint = key_issue_location_fingerprint(
                    fingerprint, comment["relevant_lines_start"], comment["relevant_lines_end"])
                if location_fingerprint in candidate_comments:
                    candidate_issues[location_fingerprint].append(issue)
                    continue
                comment["body"] = key_issue_body_with_markers(
                    comment["body"], fingerprint, location_fingerprint,
                    getattr(self.git_provider, "max_comment_chars", None))
                candidate_comments[location_fingerprint] = comment
                candidate_issues[location_fingerprint] = [issue]
                candidate_fingerprints[location_fingerprint] = fingerprint
            except Exception as e:
                get_logger().warning(f"Failed to prepare a review finding for inline publication, error: {e}",
                                     artifact={"issue": issue})
                remaining_issues.append(issue)

        if candidate_comments:
            try:
                self.git_provider.publish_code_suggestions(list(candidate_comments.values()))
            except Exception as e:
                locations = [{"relevant_file": comment["relevant_file"],
                              "start_line": comment["relevant_lines_start"],
                              "end_line": comment["relevant_lines_end"]}
                             for comment in candidate_comments.values()]
                get_logger().warning(
                    f"Failed to publish review findings as Azure DevOps threads, error: {e}",
                    artifact={"locations": locations})
            verified_locations = self._published_inline_key_issue_fingerprints(store, set(candidate_comments))
            for location_fingerprint, comment in candidate_comments.items():
                issues_for_location = candidate_issues[location_fingerprint]
                if location_fingerprint in verified_locations:
                    store.add(candidate_fingerprints[location_fingerprint])
                    store.add(location_fingerprint)
                    published += len(issues_for_location)
                    continue
                get_logger().warning("Failed to publish a review finding as an Azure DevOps inline comment, "
                                     "keeping it in the summary",
                                     artifact={"relevant_file": comment["relevant_file"],
                                               "start_line": comment["relevant_lines_start"],
                                               "end_line": comment["relevant_lines_end"]})
                remaining_issues.extend(issues_for_location)

        if not published:
            return data
        get_logger().info(f"Published {published} review finding(s) as inline comments")

        data = copy.deepcopy(data)
        if remaining_issues:
            data["review"]["key_issues_to_review"] = remaining_issues
        else:
            data["review"].pop("key_issues_to_review", None)
        return data

    def _get_user_answers(self) -> Tuple[str, str]:
        """
        Retrieves the question and answer strings from the discussion messages related to a pull request.

        Returns:
            A tuple containing the question and answer strings.
        """
        question_str = ""
        answer_str = ""

        if self.is_answer:
            discussion_messages = self.git_provider.get_issue_comments()

            # providers return the comments oldest-first. PyGithub's PaginatedList reverses lazily,
            # so prefer it and only materialise the plain lists other providers return.
            newest_first = getattr(discussion_messages, "reversed", None)
            if newest_first is None:
                newest_first = reversed(list(discussion_messages))

            for message in newest_first:
                if "Questions to better understand the PR:" in message.body:
                    question_str = message.body
                elif '/answer' in message.body:
                    answer_str = message.body

                if answer_str and question_str:
                    break

        return question_str, answer_str

    def _get_previous_review_comment(self):
        """
        Get the previous review comment if it exists.
        """
        try:
            if hasattr(self.git_provider, "get_previous_review"):
                return self.git_provider.get_previous_review(
                    full=not self.incremental.is_incremental,
                    incremental=self.incremental.is_incremental,
                )
        except Exception as e:
            get_logger().exception(f"Failed to get previous review comment, error: {e}")

    def _remove_previous_review_comment(self, comment):
        """
        Remove the previous review comment if it exists.
        """
        try:
            if comment:
                self.git_provider.remove_comment(comment)
        except Exception as e:
            get_logger().exception(f"Failed to remove previous review comment, error: {e}")

    def _can_run_incremental_review(self) -> bool:
        """
        Checks if we can run incremental review according the various configurations and previous review.
        """
        # checking if running is auto mode but there are no new commits
        if self.is_auto and not self.incremental.first_new_commit_sha:
            get_logger().info(f"Incremental review is enabled for {self.pr_url} but there are no new commits")
            return False

        if not hasattr(self.git_provider, "get_incremental_commits"):
            get_logger().info(f"Incremental review is not supported for {get_settings().config.git_provider}")
            return False
        if self.incremental.commits_range is None:
            get_logger().info(
                f"Incremental review not initialized for {get_settings().config.git_provider}; "
                f"falling back to full review."
            )
            self.incremental.is_incremental = False
            return False
        # checking if there are enough commits to start the review
        num_new_commits = len(self.incremental.commits_range)
        num_commits_threshold = get_settings().pr_reviewer.minimal_commits_for_incremental_review
        not_enough_commits = num_new_commits < num_commits_threshold
        # checking if the commits are not too recent to start the review
        recent_commits_threshold = datetime.datetime.now() - datetime.timedelta(
            minutes=get_settings().pr_reviewer.minimal_minutes_for_incremental_review
        )
        last_seen_commit_date = (
            self.incremental.last_seen_commit.commit.author.date if self.incremental.last_seen_commit else None
        )
        all_commits_too_recent = (
            last_seen_commit_date > recent_commits_threshold if self.incremental.last_seen_commit else False
        )
        # check all the thresholds or just one to start the review
        condition = any if get_settings().pr_reviewer.require_all_thresholds_for_incremental_review else all
        if condition((not_enough_commits, all_commits_too_recent)):
            get_logger().info(
                f"Incremental review is enabled for {self.pr_url} but didn't pass the threshold check to run:"
                f"\n* Number of new commits = {num_new_commits} (threshold is {num_commits_threshold})"
                f"\n* Last seen commit date = {last_seen_commit_date} (threshold is {recent_commits_threshold})"
            )
            return False
        return True

    def set_review_labels(self, data):
        if not get_settings().config.publish_output:
            return

        if not get_settings().pr_reviewer.require_estimate_effort_to_review:
            get_settings().pr_reviewer.enable_review_labels_effort = False # we did not generate this output
        if not get_settings().pr_reviewer.require_security_review:
            get_settings().pr_reviewer.enable_review_labels_security = False # we did not generate this output

        if (get_settings().pr_reviewer.enable_review_labels_security or
                get_settings().pr_reviewer.enable_review_labels_effort):
            try:
                review_labels = []
                if get_settings().pr_reviewer.enable_review_labels_effort:
                    estimated_effort = data['review']['estimated_effort_to_review_[1-5]']
                    estimated_effort_number = None
                    if isinstance(estimated_effort, str):
                        try:
                            estimated_effort_number = int(estimated_effort.split(',')[0])
                        except ValueError:
                            get_logger().warning(f"Invalid estimated_effort value: {estimated_effort}")
                    elif isinstance(estimated_effort, int):
                        estimated_effort_number = estimated_effort
                    else:
                        get_logger().warning(f"Unexpected type for estimated_effort: {type(estimated_effort)}")
                    if estimated_effort_number is not None:
                        estimated_effort_number = max(1, min(5, int(estimated_effort_number)))
                        review_labels.append(f'Review effort {estimated_effort_number}/5')
                if get_settings().pr_reviewer.enable_review_labels_security and get_settings().pr_reviewer.require_security_review:
                    security_concerns = data['review']['security_concerns']  # yes, because ...
                    security_concerns_bool = 'yes' in security_concerns.lower() or 'true' in security_concerns.lower()
                    if security_concerns_bool:
                        review_labels.append('Possible security concern')

                current_labels = self.git_provider.get_pr_labels(update=True)
                if not current_labels:
                    current_labels = []
                get_logger().debug(f"Current labels:\n{current_labels}")
                if current_labels:
                    current_labels_filtered = [label for label in current_labels if
                                               not label.lower().startswith('review effort') and not label.lower().startswith(
                                                   'possible security concern')]
                else:
                    current_labels_filtered = []
                new_labels = review_labels + current_labels_filtered
                if (current_labels or review_labels) and sorted(new_labels) != sorted(current_labels):
                    get_logger().info(f"Setting review labels:\n{review_labels + current_labels_filtered}")
                    self.git_provider.publish_labels(new_labels)
                else:
                    get_logger().info(f"Review labels are already set:\n{review_labels + current_labels_filtered}")
            except Exception as e:
                get_logger().error(f"Failed to set review labels, error: {e}")

    def auto_approve_logic(self):
        """
        Auto-approve a pull request if it meets the conditions for auto-approval.
        """
        if get_settings().config.enable_auto_approval:
            is_auto_approved = self.git_provider.auto_approve()
            if is_auto_approved:
                get_logger().info("Auto-approved PR")
                self.git_provider.publish_comment("Auto-approved PR")
        else:
            get_logger().info("Auto-approval option is disabled")
            self.git_provider.publish_comment("Auto-approval option for PR-Agent is disabled. "
                                              "You can enable it via a [configuration file](https://github.com/Codium-ai/pr-agent/blob/main/docs/REVIEW.md#auto-approval-1)")
