import asyncio
import copy
import datetime
import json
import re
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple

from jinja2 import Environment, StrictUndefined, select_autoescape

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import (
    LiteLLMAIHandler,
    get_effective_litellm_output_token_cap,
)
from pr_agent.algo.ai_request_context import AIModelRoute, get_ai_request_options
from pr_agent.algo.candidate_verification import (
    CandidateVerificationOutputBudgetError,
    apply_specialist_prioritization,
    apply_verification_decisions,
    bounded_verification_evidence,
    candidate_verification_provider_controls_hash,
    prepare_candidates,
    prompt_evidence_coverage,
    render_verification_payload,
    retrieval_request_is_complete,
    retrieve_evidence,
    safe_repo_path,
    telemetry_safe_artifact,
    validated_specialist_prioritization,
    verified_finding_identity,
)
from pr_agent.algo.checkpoint_stage_sources import (
    checkpoint_candidate_verification_config,
    checkpoint_candidate_verification_enabled,
    checkpoint_frontier_adjudication_config,
    checkpoint_frontier_adjudication_enabled,
    checkpoint_specialist_pipeline,
    checkpoint_specialists_enabled,
    get_checkpoint_stage_sources,
)
from pr_agent.algo.config_utils import parse_env_bool
from pr_agent.algo.frontier_adjudication import (
    FrontierAdjudicationRequest,
    FrontierCandidate,
    FrontierContractError,
    FrontierSignals,
    NormalizedSeverity,
    build_frontier_evidence,
    normalize_severity,
    run_frontier_adjudication,
)
from pr_agent.algo.git_patch_processing import iter_git_patch_lines, split_git_file_lines
from pr_agent.algo.inline_comment_dedup import (
    SUMMARY_FALLBACK_MARKER_VERSION,
    InlineCommentStore,
    can_verify_inline_comment_publication,
    get_inline_comment_store,
    key_issue_body_with_markers,
    key_issue_fingerprint,
    key_issue_location_fingerprint,
    summary_fallback_markers,
)
from pr_agent.algo.pr_processing import (
    OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD,
    PRDiffCoverage,
    add_ai_metadata_to_diff_files,
    get_pr_diff,
    retry_with_fallback_models,
)
from pr_agent.algo.repo_context import build_repo_context
from pr_agent.algo.review_execution_context import get_review_prompt_date, review_execution_is_isolated
from pr_agent.algo.review_router import (
    ChangedFile,
    ChangeKind,
    ReviewDepthEscalation,
    ReviewRouteDecision,
    ReviewRouteRequest,
    ReviewRoutingConfiguration,
    load_review_routing_configuration,
    review_route_decision_to_dict,
    route_review,
)
from pr_agent.algo.review_specialists import (
    SPECIALIST_BATCH_SCHEMA_VERSION,
    RoleExecution,
    SpecialistBatchResult,
    SpecialistRole,
    SpecialistState,
    build_specialist_input,
    get_specialist_snapshot_context,
    run_shadow_specialists,
    unavailable_specialist_batch,
    validate_specialist_output,
)
from pr_agent.algo.review_thread_reconciler import (
    DesiredReviewThread,
    ReviewThreadActionKind,
    ReviewThreadActionState,
    ReviewThreadAnchor,
    ReviewThreadFailureKind,
    SummaryFallbackEntry,
    SummaryFallbackReason,
    execute_review_thread_action_plan,
    finding_identities_from_verified_findings,
    plan_review_thread_actions,
)
from pr_agent.algo.run_details import (
    RunDetails,
    adjudication_runs_to_dict,
    get_run_details,
    init_run_details,
    isolate_run_details,
    record_review_profile,
    record_review_route,
    record_specialist_result,
)
from pr_agent.algo.skills_loader import get_skills_context
from pr_agent.algo.token_handler import TokenEncoder, TokenHandler
from pr_agent.algo.types import EDIT_TYPE
from pr_agent.algo.utils import (
    DuplicateYamlKeyError,
    ModelType,
    PRReviewHeader,
    PRReviewIdentity,
    add_pr_review_identity,
    convert_to_markdown_v2,
    get_max_tokens,
    get_model,
    github_action_output,
    load_yaml,
    push_outputs,
    show_relevant_configurations,
    show_run_details,
)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider_with_context
from pr_agent.git_providers.git_provider import (
    GitProvider,
    IncrementalPR,
    get_main_pr_language,
)
from pr_agent.log import get_logger
from pr_agent.servers.help import HelpMessage
from pr_agent.tools.ticket_pr_compliance_check import extract_and_cache_pr_tickets

load_production_candidate_verification_config = checkpoint_candidate_verification_config
load_frontier_adjudication_config = checkpoint_frontier_adjudication_config
load_specialist_pipeline_config = checkpoint_specialist_pipeline
specialists_enabled = checkpoint_specialists_enabled

MAX_REVIEW_COVERAGE_FILES = 50
_SUGGESTION_FENCE_RE = re.compile(r"```[ \t]*suggestion\b", re.IGNORECASE)
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_MACHINE_FAILURE_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_VALID_REVIEW_PROFILES = {"full", "bugs_only"}
_ROUTING_INVENTORY_UNSET = object()
_REVIEW_THREAD_ACTION_KINDS = (
    ReviewThreadActionKind.CREATE,
    ReviewThreadActionKind.UPDATE,
    ReviewThreadActionKind.RESOLVE,
    ReviewThreadActionKind.UNCHANGED,
    ReviewThreadActionKind.SKIP,
    ReviewThreadActionKind.SUMMARY_FALLBACK,
)
_REVIEW_THREAD_ACTION_STATES = (
    ReviewThreadActionState.APPLIED,
    ReviewThreadActionState.ALREADY_APPLIED,
    ReviewThreadActionState.STALE_HEAD,
    ReviewThreadActionState.STALE_INVENTORY,
    ReviewThreadActionState.FAILED,
    ReviewThreadActionState.NOT_EXECUTED,
    ReviewThreadActionState.SKIPPED,
    ReviewThreadActionState.FALLBACK_REQUIRED,
    ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
)
_REVIEW_THREAD_FALLBACK_REASONS = (
    SummaryFallbackReason.INVALID_INLINE_LOCATION,
    SummaryFallbackReason.INLINE_REJECTED,
    SummaryFallbackReason.PERMISSION_DENIED,
    SummaryFallbackReason.RATE_LIMITED,
    SummaryFallbackReason.PROVIDER_FAILURE,
)
_BUG_FINDING_HEADERS = {
    "bug": "Bug",
    "security": "Security vulnerability",
    "performance": "Performance regression",
}
_GENERIC_CI_EVIDENCE_TERMS = {
    "assert", "assertion", "build", "check", "error", "errors", "fail", "failed", "failure", "failures",
    "job", "test", "tests", "unit",
}


@dataclass(frozen=True)
class StructuredReviewExecution:
    """Provider-neutral output from one forced no-publish review run."""

    review: Optional[Mapping[str, Any]] = field(repr=False)
    run_details: Optional[RunDetails] = field(repr=False)


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
        self._review_prediction_finish_reason = None
        self.candidate_verification_artifact = None
        self.frontier_adjudication_artifact = None
        self._frontier_adjudication_config = None
        self._candidate_verification_published_finding_count = None
        self.verified_review_data = None
        self.specialist_shadow_input = None
        self.specialist_shadow_result = None
        self._specialists_started = False
        self.review_routing_configuration = None
        self.review_route_request = None
        self.review_route_decision = None
        self._review_context_tokens = None
        self._review_max_findings = None
        self._review_max_verification_candidates = None
        self._review_max_published_findings = None
        self._review_shadow_only = False
        self._force_no_publish = False
        self._structured_review_result = None
        self._review_execution_started = False
        self.review_thread_reconciliation_artifact = None
        self._review_thread_summary_fallbacks = ()
        self._review_thread_lifecycle_notice = None
        self._review_thread_lifecycle_blocks_summary = False
        question_str, answer_str = self._get_user_answers()
        self.pr_description, self.pr_description_files = (
            self.git_provider.get_pr_description(split_changes_walkthrough=True))
        self._enable_ai_metadata = bool(
            self.pr_description_files
            and get_settings().get("config.is_auto_command", False)
            and get_settings().get("config.enable_ai_metadata", False)
        )
        if self._enable_ai_metadata:
            add_ai_metadata_to_diff_files(self.git_provider, self.pr_description_files)
            get_logger().debug("AI metadata added to the this command")
        else:
            get_logger().debug("AI metadata is disabled for this command")

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
            "publication_threshold": "none",
            "bugs_only": bugs_only,
            "require_score": not bugs_only and get_settings().pr_reviewer.require_score_review,
            "require_tests": not bugs_only and get_settings().pr_reviewer.require_tests_review,
            "require_estimate_effort_to_review": (
                not bugs_only and get_settings().pr_reviewer.require_estimate_effort_to_review
            ),
            "require_estimate_contribution_time_cost": (
                not bugs_only and get_settings().pr_reviewer.require_estimate_contribution_time_cost
            ),
            "require_risk_assessment": (
                not bugs_only
                and parse_env_bool(get_settings().pr_reviewer.get("require_risk_assessment", False)) is True
            ),
            "require_merge_recommendation": (
                not bugs_only
                and parse_env_bool(get_settings().pr_reviewer.get("require_merge_recommendation", False)) is True
            ),
            "require_priority_files": (
                not bugs_only
                and parse_env_bool(get_settings().pr_reviewer.get("require_priority_files", False)) is True
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
            "is_ai_metadata": self._enable_ai_metadata,
            "related_tickets": [] if bugs_only else get_settings().get('related_tickets', []),
            'duplicate_prompt_examples': get_settings().config.get('duplicate_prompt_examples', False),
            "date": get_review_prompt_date(),
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
        self._review_execution_started = True
        init_run_details()
        record_review_profile(self._review_profile())
        progress_response = None
        review_failed = False
        route_prepared = False
        try:
            if self.incremental.is_incremental:
                can_run = self._can_run_incremental_review()
                # If the gate disabled incremental (e.g., commits_range is None), fall through to full review.
                if not can_run and self.incremental.is_incremental:
                    scope_empty = self.git_provider.is_incremental_scope_empty()
                    if scope_empty is True:
                        self._prepare_known_empty_route()
                    else:
                        # A missing commit marker is not authoritative emptiness after
                        # rebases or force-pushes. Preserve real/unknown routing evidence
                        # before the threshold/no-new-commit early return.
                        self._prepare_review_route()
                    if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                        get_settings().data = {"artifact": ""}
                    return None
                if (
                    self.incremental.is_incremental
                    and self.git_provider.is_incremental_scope_empty() is True
                ):
                    self._prepare_known_empty_route()
                    get_logger().info(
                        f"Incremental review is enabled for {self.pr_url} but there are no new files"
                    )
                    previous_review_url = ""
                    if (
                        hasattr(self.git_provider, "previous_review")
                        and self.git_provider.previous_review is not None
                    ):
                        previous_review_url = (
                            getattr(self.git_provider.previous_review, "html_url", "") or ""
                        )
                    if get_settings().config.publish_output and self._provider_mutations_allowed():
                        self.git_provider.publish_comment(
                            "Incremental Review Skipped\n"
                            f"No files were changed since the [previous PR Review]({previous_review_url})"
                        )
                    if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                        get_settings().data = {"artifact": ""}
                    return None

                # A provider can have a non-empty unfiltered safety inventory but
                # no reviewable files after ignore/extension filtering. Route that
                # evidence before the ordinary empty-file gate so a sensitive path
                # cannot disappear without recording the required safety depth.
                self._prepare_review_route()
                route_prepared = True

            if not self.git_provider.get_files():
                if not route_prepared:
                    self._prepare_route_after_empty_review_inventory()
                if getattr(self, "_force_no_publish", False):
                    self._publish_structured_review_data(
                        {"review": {"key_issues_to_review": []}},
                        source_free=True,
                    )
                if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                    get_settings().data = {"artifact": ""}
                get_logger().info(f"PR has no files: {self.pr_url}, skipping review")
                return None

            if not route_prepared:
                self._prepare_review_route()
            if self._frontier_adjudication_enabled():
                if not self._candidate_verification_enabled():
                    self.frontier_adjudication_artifact = {
                        "enabled": True,
                        "status": "configuration_invalid",
                        "failure": "candidate_verification_required",
                        "results": [],
                        "publication_safe": False,
                    }
                    self._publish_structured_review_data({
                        "review": {"key_issues_to_review": []},
                    }, source_free=True)
                    if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                        get_settings().data = {"artifact": ""}
                    return None
                if not self._prepare_frontier_adjudication_config():
                    self._publish_structured_review_data({
                        "review": {"key_issues_to_review": []},
                    }, source_free=True)
                    if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                        get_settings().data = {"artifact": ""}
                    get_logger().warning(
                        "Frontier adjudication preflight is unavailable",
                        artifact=self.frontier_adjudication_artifact,
                    )
                    return None
            if self._specialist_escalation_consumption_enabled():
                await self._run_guarded_specialist_escalation()
            if self._local_artifact_mutations_allowed() and getattr(self, "_review_shadow_only", False):
                # Shadow output is consumed through the same request-local artifact
                # channel as local, health, and MOSAICO runs. Clear any value left by
                # an earlier command so a failed/unavailable shadow attempt cannot be
                # mistaken for a fresh review.
                get_settings().data = {"artifact": ""}

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

            if (get_settings().config.publish_output and self._provider_mutations_allowed() and
                    not get_settings().config.get('is_auto_command', False)):
                progress_response = self.git_provider.publish_comment("Preparing review...", is_temporary=True)

            model_route = self._review_model_route()
            if model_route is None:
                await retry_with_fallback_models(self._prepare_prediction, model_type=ModelType.REGULAR)
            else:
                await retry_with_fallback_models(
                    self._prepare_prediction,
                    model_type=ModelType.REGULAR,
                    model_route=model_route,
                )
            if not self.prediction:
                if getattr(self, "_force_no_publish", False):
                    self._publish_structured_review_data({
                        "review": {"key_issues_to_review": []},
                    })
                return None
            candidate_verification_enabled = self._candidate_verification_enabled()
            if candidate_verification_enabled:
                await self._run_candidate_verification()
            elif self._frontier_adjudication_enabled():
                self.frontier_adjudication_artifact = {
                    "enabled": True,
                    "status": "configuration_invalid",
                    "failure": "candidate_verification_required",
                    "results": [],
                    "publication_safe": False,
                }

            pr_review = self._prepare_pr_review()
            await self._push_prepared_review_output(pr_review)
            if self._local_artifact_mutations_allowed():
                get_logger().debug("PR output", artifact=pr_review)
            else:
                get_logger().debug("Structured no-publish review prepared")

            should_publish = (
                get_settings().config.publish_output
                and self._provider_mutations_allowed()
                and self._should_publish_review_no_suggestions(pr_review)
            )
            if not should_publish:
                self._clear_stale_persistent_bugs_only_review()
                reason = "Review output is not published"
                if self._candidate_verification_blocks_publication():
                    reason += ": candidate verification did not complete successfully."
                elif get_settings().config.publish_output:
                    reason += ": no major issues detected."
                get_logger().info(reason)
                if self._local_artifact_mutations_allowed():
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
                    self._provider_mutations_allowed() and
                    not get_settings().config.get("is_auto_command", False)):
                try:
                    self.git_provider.publish_comment("Failed to review PR")
                except Exception as e:
                    get_logger().exception(f"Failed to publish review failure result, error: {e}")

    async def _run_structured_no_publish_once(self) -> StructuredReviewExecution:
        """Run one already-isolated reviewer while forcing review output sinks closed."""

        if not review_execution_is_isolated():
            raise RuntimeError("structured review execution requires an outer isolation boundary")
        if getattr(self, "_review_execution_started", False):
            raise RuntimeError("structured review execution requires a fresh reviewer instance")
        self._review_execution_started = True

        previous_force_no_publish = getattr(self, "_force_no_publish", False)
        related_tickets = copy.deepcopy(self.vars.get("related_tickets"))
        had_related_tickets = "related_tickets" in self.vars
        self._force_no_publish = True
        self._structured_review_result = None
        try:
            self.vars["related_tickets"] = []
            with isolate_run_details():
                await self.run()
                run_details = copy.deepcopy(get_run_details())
                if run_details is not None:
                    run_details.freeze_duration()
                return StructuredReviewExecution(
                    review=copy.deepcopy(self._structured_review_result),
                    run_details=run_details,
                )
        finally:
            self._force_no_publish = previous_force_no_publish
            self._structured_review_result = None
            if had_related_tickets:
                self.vars["related_tickets"] = related_tickets
            else:
                self.vars.pop("related_tickets", None)

    def _should_publish_review_no_suggestions(self, pr_review: str) -> bool:
        if (
            self._candidate_verification_blocks_publication()
            or getattr(self, "_review_shadow_only", False)
        ):
            return False
        if self._review_profile() == "bugs_only":
            return bool(pr_review.strip())
        return get_settings().pr_reviewer.get('publish_output_no_suggestions', True) or "No major issues detected" not in pr_review

    def _provider_mutations_allowed(self) -> bool:
        """Keep a shadow-only route observational, including cleanup and progress updates."""
        return not (
            getattr(self, "_review_shadow_only", False)
            or getattr(self, "_force_no_publish", False)
        )

    def _local_artifact_mutations_allowed(self) -> bool:
        """Keep forced structured execution from mutating process-local output state."""
        return not getattr(self, "_force_no_publish", False)

    def _review_profile(self) -> str:
        """Return the selected profile, defaulting legacy/test instances to full review."""
        return getattr(self, "review_profile", "full")

    def _routing_configuration(self) -> ReviewRoutingConfiguration:
        configuration = getattr(self, "review_routing_configuration", None)
        if configuration is None:
            configuration = load_review_routing_configuration(
                get_settings().get("review_depth", None)
            )
            self.review_routing_configuration = configuration
        return configuration

    def _prepare_review_route(
        self,
        *,
        raw_inventory: Any = _ROUTING_INVENTORY_UNSET,
    ) -> ReviewRouteDecision:
        """Select and apply deterministic depth before any review model call."""

        configuration = self._routing_configuration()
        if configuration.enabled:
            try:
                files = self._changed_files_for_routing(raw_inventory=raw_inventory)
            except Exception as exc:
                # Provider metadata is routing evidence, not a reason to abort the
                # review. An empty immutable snapshot records the missing input and
                # prevents the router from selecting quick.
                files = ()
                get_logger().warning(
                    "Review-depth routing could not read changed-file metadata",
                    artifact={"error_class": type(exc).__name__},
                )
            labels = self._labels_for_routing()
        else:
            # Preserve legacy provider behavior: disabled routing performs no new
            # label or diff metadata calls and inherits the ordinary review budget.
            files = ()
            labels = None
        request = ReviewRouteRequest(
            files=files,
            requested_depth=configuration.requested_depth,
            review_profile=self._review_profile(),
            labels=labels,
        )
        self.review_route_request = request
        decision = route_review(request, configuration.policy)
        self._apply_review_route(decision)
        return decision

    def _prepare_route_after_empty_review_inventory(self) -> ReviewRouteDecision:
        """Route a filtered-empty full PR without losing provider safety evidence."""

        configuration = self._routing_configuration()
        routing_getter = getattr(type(self.git_provider), "get_files_for_routing", None)
        has_richer_inventory = (
            configuration.enabled
            and callable(routing_getter)
            and routing_getter is not GitProvider.get_files_for_routing
        )
        if not has_richer_inventory:
            # With routing disabled, preserve legacy behavior and make no metadata
            # calls. With the base capability, get_files() is the routing inventory
            # and its empty result is already authoritative.
            return self._prepare_known_empty_route()

        try:
            raw_inventory = routing_getter(self.git_provider)
        except Exception as exc:
            get_logger().warning(
                "Review-depth routing could not read the unfiltered changed-file inventory",
                artifact={"error_class": type(exc).__name__},
            )
            # Preserve an explicit evidence gap and any available detailed inventory.
            return self._prepare_review_route(raw_inventory=None)
        if raw_inventory is None:
            return self._prepare_review_route(raw_inventory=None)
        if not raw_inventory:
            return self._prepare_known_empty_route()
        return self._prepare_review_route(raw_inventory=raw_inventory)

    def _prepare_known_empty_route(self) -> ReviewRouteDecision:
        """Apply routing to an authoritative empty scope without file lookups."""

        configuration = self._routing_configuration()
        labels = self._labels_for_routing() if configuration.enabled else None
        request = ReviewRouteRequest(
            files=(),
            requested_depth=configuration.requested_depth,
            review_profile=self._review_profile(),
            labels=labels,
            changed_files_complete=True,
        )
        self.review_route_request = request
        decision = route_review(request, configuration.policy)
        self._apply_review_route(decision)
        return decision

    def _changed_files_for_routing(
        self,
        *,
        raw_inventory: Any = _ROUTING_INVENTORY_UNSET,
    ) -> tuple[ChangedFile, ...]:
        """Keep ignored or unsupported files visible to the safety router.

        The ordinary review diff is intentionally filtered for context quality. Risk
        routing has a different contract: every changed path must remain available so
        an ignore rule cannot hide a forced-deep path. Reconcile the provider's raw
        file inventory with the richer review diff through provider-owned routing
        capabilities rather than branching on provider type.
        """

        evidence_incomplete = False
        # Snapshot the raw inventory before asking a provider to materialize
        # detailed patches. Some incremental providers reuse mutable inventory
        # containers while loading diffs; retaining an immutable tuple here keeps
        # authoritative paths stable without provider-specific branching.
        try:
            if raw_inventory is _ROUTING_INVENTORY_UNSET:
                routing_getter = getattr(type(self.git_provider), "get_files_for_routing", None)
                if callable(routing_getter):
                    raw_inventory = routing_getter(self.git_provider)
                else:
                    raw_inventory = self.git_provider.get_files()
            raw_files = (
                tuple(self._provider_changed_file_for_routing(file) for file in raw_inventory)
                if raw_inventory is not None
                else ()
            )
        except Exception as exc:
            get_logger().warning(
                "Review-depth routing could not read the unfiltered changed-file inventory",
                artifact={"error_class": type(exc).__name__},
            )
            raw_files = None
            evidence_incomplete = True

        try:
            detailed = tuple(
                self._provider_changed_file_for_routing(file)
                for file in (self.git_provider.get_diff_files() or [])
            )
        except Exception as exc:
            get_logger().warning(
                "Review-depth routing could not read the detailed changed-file inventory",
                artifact={"error_class": type(exc).__name__},
            )
            detailed = ()
            evidence_incomplete = True
        detailed_by_path: dict[str, int] = {}
        for index, changed_file in enumerate(detailed):
            for path in (changed_file.old_path, changed_file.new_path):
                if path:
                    detailed_by_path.setdefault(path, index)

        if raw_files is None:
            # Preserve the detailed evidence, but add one unknown record so missing
            # inventory can never be mistaken for a complete low-risk-only change.
            return (*detailed, ChangedFile(new_path=None, kind=ChangeKind.UNKNOWN))

        if not raw_files:
            # A pull request with no raw paths cannot prove that a filtered detailed
            # diff is complete. Record the gap even when the detailed inventory exists.
            return (*detailed, ChangedFile(new_path=None, kind=ChangeKind.UNKNOWN))

        reconciled = []
        matched_detailed = set()
        for raw_changed_file in raw_files:
            detailed_index = next(
                (
                    detailed_by_path[path]
                    for path in (raw_changed_file.old_path, raw_changed_file.new_path)
                    if path in detailed_by_path
                ),
                None,
            )
            if detailed_index is None:
                reconciled.append(raw_changed_file)
                continue
            merged_files, merge_incomplete = self._merge_changed_file_routing_evidence(
                raw_changed_file,
                detailed[detailed_index],
            )
            reconciled.extend(merged_files)
            evidence_incomplete = evidence_incomplete or merge_incomplete
            matched_detailed.add(detailed_index)

        unmatched_detailed = [
            changed_file
            for index, changed_file in enumerate(detailed)
            if index not in matched_detailed
        ]
        if unmatched_detailed:
            evidence_incomplete = True
            reconciled.extend(unmatched_detailed)
        if evidence_incomplete:
            reconciled.append(ChangedFile(new_path=None, kind=ChangeKind.UNKNOWN))
        return tuple(reconciled)

    @staticmethod
    def _merge_changed_file_routing_evidence(
        raw: ChangedFile,
        detailed: ChangedFile,
    ) -> tuple[tuple[ChangedFile, ...], bool]:
        """Retain authoritative rename provenance while preferring detailed counts.

        Some providers expose the old path only in their raw file inventory while
        their patch adapter reports a rename with only the new filename. Conflicting
        rename provenance remains as separate evidence plus an incomplete marker so
        routing fails safe instead of choosing one path silently.
        """
        if raw.kind is not ChangeKind.RENAMED and detailed.kind is not ChangeKind.RENAMED:
            return (detailed,), False

        incompatible_kinds = {
            kind
            for kind in (raw.kind, detailed.kind)
            if kind not in {ChangeKind.RENAMED, ChangeKind.MODIFIED, ChangeKind.UNKNOWN}
        }
        new_paths = {path for path in (raw.new_path, detailed.new_path) if path}
        old_paths = {path for path in (raw.old_path, detailed.old_path) if path}
        if incompatible_kinds or len(new_paths) > 1 or len(old_paths) > 1:
            return (raw, detailed), True

        merged = replace(
            detailed,
            kind=ChangeKind.RENAMED,
            old_path=next(iter(old_paths), None),
            new_path=next(iter(new_paths), None),
            additions=(detailed.additions if detailed.additions is not None else raw.additions),
            deletions=(detailed.deletions if detailed.deletions is not None else raw.deletions),
            generated=(detailed.generated if detailed.generated is not None else raw.generated),
        )
        incomplete = (
            not merged.old_path
            or not merged.new_path
            or merged.old_path == merged.new_path
        )
        return (merged,), incomplete

    def _provider_changed_file_for_routing(self, file: Any) -> ChangedFile:
        """Convert routing evidence and apply only the provider's path adapter."""

        changed_file = self._changed_file_for_routing(file)
        path_adapter = getattr(type(self.git_provider), "normalize_file_path_for_routing", None)
        if not callable(path_adapter):
            return changed_file
        return replace(
            changed_file,
            old_path=path_adapter(self.git_provider, changed_file.old_path),
            new_path=path_adapter(self.git_provider, changed_file.new_path),
        )

    @staticmethod
    def _changed_file_for_routing(file: Any) -> ChangedFile:
        def value(*names: str) -> Any:
            if isinstance(file, Mapping):
                return next((file[name] for name in names if name in file), None)
            return next((getattr(file, name) for name in names if hasattr(file, name)), None)

        def path_value(*names: str) -> str | None:
            for name in names:
                candidate = file.get(name) if isinstance(file, Mapping) else getattr(file, name, None)
                if isinstance(candidate, str) and candidate:
                    return candidate
            return None

        if isinstance(file, str):
            return ChangedFile(new_path=file, kind=ChangeKind.UNKNOWN)

        edit_type = value("edit_type")
        status = value("status")
        kind = {
            EDIT_TYPE.ADDED: ChangeKind.ADDED,
            EDIT_TYPE.DELETED: ChangeKind.DELETED,
            EDIT_TYPE.MODIFIED: ChangeKind.MODIFIED,
            EDIT_TYPE.RENAMED: ChangeKind.RENAMED,
            EDIT_TYPE.UNKNOWN: ChangeKind.UNKNOWN,
        }.get(edit_type, ChangeKind.UNKNOWN)
        if kind is ChangeKind.UNKNOWN and isinstance(status, str):
            kind = {
                "added": ChangeKind.ADDED,
                "removed": ChangeKind.DELETED,
                "deleted": ChangeKind.DELETED,
                "renamed": ChangeKind.RENAMED,
                "modified": ChangeKind.MODIFIED,
            }.get(status.casefold(), ChangeKind.UNKNOWN)
        elif kind is ChangeKind.UNKNOWN:
            if value("new_file") is True:
                kind = ChangeKind.ADDED
            elif value("deleted_file") is True:
                kind = ChangeKind.DELETED
            elif value("renamed_file") is True:
                kind = ChangeKind.RENAMED

        filename = path_value("filename", "new_path", "path", "b_path", "a_path")
        old_filename = path_value("old_filename", "previous_filename", "old_path", "a_path")
        if kind is ChangeKind.DELETED:
            old_path, new_path = old_filename or filename, None
        elif kind is ChangeKind.RENAMED:
            old_path, new_path = old_filename, filename
        else:
            old_path, new_path = None, filename

        def line_count(*names: str) -> int | None:
            count = value(*names)
            return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None

        return ChangedFile(
            new_path=new_path,
            old_path=old_path,
            kind=kind,
            additions=line_count("num_plus_lines", "additions"),
            deletions=line_count("num_minus_lines", "deletions"),
            generated=value("generated") if isinstance(value("generated"), bool) else None,
        )

    def _labels_for_routing(self) -> tuple[str, ...] | None:
        try:
            if not self.git_provider.is_supported("get_labels"):
                return None
            routing_getter = getattr(type(self.git_provider), "get_pr_labels_for_routing", None)
            if callable(routing_getter):
                labels = routing_getter(self.git_provider)
            else:
                labels = self.git_provider.get_pr_labels()
        except Exception as exc:
            get_logger().warning(
                "Review-depth routing could not read pull-request labels",
                artifact={"error_class": type(exc).__name__},
            )
            return None
        if labels is None:
            return None
        if not isinstance(labels, (list, tuple)):
            return (labels,)
        return tuple(labels)

    def _apply_review_route(self, decision: ReviewRouteDecision) -> None:
        self.review_route_decision = decision
        budget = decision.applied_budget
        self._review_context_tokens = budget.context_tokens if decision.routing_enabled else None
        self._review_max_findings = budget.max_findings if decision.routing_enabled else None
        self._review_max_verification_candidates = (
            budget.max_verification_candidates if decision.routing_enabled else None
        )
        self._review_max_published_findings = (
            budget.max_published_findings if decision.routing_enabled else None
        )
        self._review_shadow_only = bool(decision.routing_enabled and budget.shadow_only is True)
        if hasattr(self, "vars"):
            self.vars["num_max_findings"] = (
                budget.max_findings
                if decision.routing_enabled and budget.max_findings is not None
                else get_settings().pr_reviewer.num_max_findings
            )
            self.vars["publication_threshold"] = (
                budget.publication_threshold
                if decision.routing_enabled and budget.publication_threshold is not None
                else "none"
            )
            self.vars["max_verification_candidates"] = self._review_max_verification_candidates
        if decision.routing_enabled:
            serialized = review_route_decision_to_dict(decision)
            record_review_route(serialized)
            get_logger().info(
                "Review depth selected",
                artifact={
                    "requested_depth": decision.requested_depth,
                    "applied_depth": decision.applied_depth.value,
                    "reason_codes": [reason.code for reason in decision.reasons],
                    "policy_version": decision.policy_version,
                },
            )

    def _specialist_escalation_consumption_enabled(self) -> bool:
        configuration = self._routing_configuration()
        return bool(configuration.enabled and configuration.consume_specialist_escalation)

    async def _run_guarded_specialist_escalation(self) -> None:
        """Consume only a validated upward-only risk record behind an explicit gate."""

        if specialists_enabled():
            await self._run_shadow_specialists_once()
            escalation = self._validated_specialist_escalation()
        else:
            escalation = ReviewDepthEscalation(
                source="specialist:risk_recommendation",
                minimum_depth=None,
                reasons=(),
                available=False,
            )
        if escalation is None:
            return
        request = replace(self.review_route_request, escalation=escalation)
        self.review_route_request = request
        self._apply_review_route(route_review(request, self._routing_configuration().policy))

    @staticmethod
    def _is_expected_unavailable_specialist_batch(
        result: SpecialistBatchResult,
        pipeline: Any,
    ) -> bool:
        """Recognize the batch emitted when no stable provider identity exists."""
        if (
            result.schema_version != SPECIALIST_BATCH_SCHEMA_VERSION
            or result.stale
            or result.snapshot_id != "unavailable"
            or result.head_sha != ""
            or result.input_hash != ""
            or result.configuration_hash != getattr(pipeline, "configuration_hash", None)
            or result.changed_path_count != 0
            or result.hunk_count != 0
            or not isinstance(result.records, tuple)
            or not isinstance(result.role_records, Mapping)
        ):
            return False

        role_configs = getattr(pipeline, "roles", None)
        if not isinstance(role_configs, tuple):
            return False
        enabled_roles = []
        enabled_role_configs = {}
        for config in role_configs:
            role = getattr(config, "role", None)
            enabled = getattr(config, "enabled", None)
            model = getattr(config, "model", None)
            deployment = getattr(config, "deployment", None)
            if (
                not isinstance(role, SpecialistRole)
                or not isinstance(enabled, bool)
                or role in enabled_roles
                or not isinstance(model, str)
                or not model.strip()
                or (deployment is not None and not isinstance(deployment, str))
            ):
                return False
            if enabled:
                enabled_roles.append(role)
                enabled_role_configs[role] = config
        if len(result.records) != len(enabled_roles):
            return False

        record_roles = []
        for record in result.records:
            expected_record = RoleExecution(
                role=record.role,
                state=SpecialistState.UNAVAILABLE,
                failure_reason="stable_head_identity_unavailable",
            ) if isinstance(record, RoleExecution) else None
            if (
                not isinstance(record, RoleExecution)
                or record.role not in enabled_roles
                or record.role in record_roles
                or record != expected_record
            ):
                return False
            record_roles.append(record.role)
        if set(record_roles) != set(enabled_roles):
            return False

        expected_role_names = {role.value for role in enabled_roles}
        if set(result.role_records) != expected_role_names:
            return False
        for role in enabled_roles:
            try:
                prompt = pipeline.prompt(role)
            except Exception:
                return False
            expected_role_record = {
                "role": role.value,
                "model": enabled_role_configs[role].model,
                "deployment": enabled_role_configs[role].deployment,
                "fallback_used": False,
                "route_attempts": 0,
                "model_retry_attempts": 0,
                "prompt_version": prompt.prompt_version,
                "input_schema_version": prompt.input_schema_version,
                "schema_version": prompt.schema_version,
                "state": SpecialistState.UNAVAILABLE.value,
                "latency_seconds": 0.0,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "ai_calls": 0,
                },
                "cost": {
                    "status": "unavailable",
                    "total_usd": None,
                    "by_model_usd": {},
                },
                "confidence": None,
                "failure_reason": "stable_head_identity_unavailable",
                "cached": False,
                "reservation": {"input_tokens": 0, "output_tokens": 0},
                "output": None,
            }
            role_record = result.role_records[role.value]
            if not isinstance(role_record, Mapping) or dict(role_record) != expected_role_record:
                return False
        return True

    def _validated_specialist_escalation(self) -> ReviewDepthEscalation | None:
        result = getattr(self, "specialist_shadow_result", None)
        pipeline = getattr(self, "_specialist_pipeline", None)
        specialist_input = getattr(self, "_specialist_input", None)
        source = "specialist:risk_recommendation"
        unavailable = ReviewDepthEscalation(
            source=source,
            minimum_depth=None,
            reasons=(),
            available=False,
        )
        invalid = ReviewDepthEscalation(
            source=source,
            minimum_depth=None,
            reasons=("validated_contract_invalid",),
            available=True,
        )
        if result is None:
            return unavailable
        if not isinstance(result, SpecialistBatchResult) or pipeline is None:
            return invalid
        if specialist_input is None:
            return (
                unavailable
                if self._is_expected_unavailable_specialist_batch(result, pipeline)
                else invalid
            )
        if (
            result.schema_version != SPECIALIST_BATCH_SCHEMA_VERSION
            or result.stale
            or result.snapshot_id != specialist_input.snapshot_id
            or result.head_sha != specialist_input.head_sha
            or result.input_hash != specialist_input.input_hash
            or result.configuration_hash != pipeline.configuration_hash
        ):
            return invalid

        if not isinstance(result.records, tuple) or any(
            not isinstance(record, RoleExecution) for record in result.records
        ):
            return invalid

        role_configs = getattr(pipeline, "roles", None)
        if not isinstance(role_configs, tuple):
            return invalid
        configurations = {}
        for config in role_configs:
            role = getattr(config, "role", None)
            enabled = getattr(config, "enabled", None)
            if not isinstance(role, SpecialistRole) or not isinstance(enabled, bool) or role in configurations:
                return invalid
            configurations[role] = config
        record_roles = set()
        for record in result.records:
            config = configurations.get(record.role)
            if config is None or config.enabled is not True or record.role in record_roles:
                return invalid
            record_roles.add(record.role)

        risk_config = configurations.get(SpecialistRole.RISK_RECOMMENDATION)
        records = [record for record in result.records if record.role is SpecialistRole.RISK_RECOMMENDATION]
        if risk_config is None:
            return invalid
        if risk_config.enabled is not True:
            return unavailable if not records else invalid
        if len(records) != 1:
            return invalid
        record = records[0]
        if record.state is SpecialistState.MALFORMED_OUTPUT or record.state is SpecialistState.DISABLED:
            return invalid
        if record.state not in {SpecialistState.SUCCESS, SpecialistState.CACHED}:
            return unavailable
        output = record.output
        if not isinstance(output, Mapping):
            return invalid
        try:
            # Re-run #12's versioned validator at the consumer boundary. This keeps
            # raw, rejected, or hand-constructed model output from becoming a route
            # input even if a malformed batch object reaches this method.
            validated_output = validate_specialist_output(
                SpecialistRole.RISK_RECOMMENDATION,
                json.dumps(dict(output)),
                specialist_input,
                pipeline,
            )
        except Exception:
            return invalid
        confidence = validated_output.get("confidence")
        if record.confidence != confidence:
            return invalid
        recommendation = validated_output.get("recommendation")
        if recommendation == "none":
            return None
        evidence = self._canonical_specialist_evidence(validated_output.get("reasons"))
        if not evidence:
            return invalid
        return ReviewDepthEscalation(
            source=source,
            minimum_depth=self._routing_configuration().specialist_escalation_depth,
            reasons=("validated_recommendation:escalate", *evidence),
        )

    @staticmethod
    def _canonical_specialist_evidence(reasons: Any) -> tuple[str, ...]:
        """Keep validated anchors only; never feed model prose back into routing."""

        if not isinstance(reasons, list):
            return ()
        anchors = []
        for reason in reasons:
            if not isinstance(reason, Mapping) or set(reason) != {"reason", "evidence"}:
                return ()
            evidence_items = reason.get("evidence")
            if not isinstance(evidence_items, list) or not evidence_items:
                return ()
            for evidence in evidence_items:
                if not isinstance(evidence, Mapping):
                    return ()
                source = evidence.get("source")
                if source == "diff_hunk" and set(evidence) == {"source", "path", "hunk_id", "line"}:
                    anchors.append(
                        f"diff_hunk:{evidence['path']}:{evidence['hunk_id']}:{evidence['line']}"
                    )
                elif source == "pull_request" and set(evidence) == {"source", "field"}:
                    anchors.append(f"pull_request:{evidence['field']}")
                elif source == "deterministic_result" and set(evidence) == {"source", "rule_id"}:
                    anchors.append(f"deterministic_result:{evidence['rule_id']}")
                else:
                    return ()
        return tuple(dict.fromkeys(anchors))

    def _review_model_route(self) -> AIModelRoute | None:
        decision = getattr(self, "review_route_decision", None)
        if decision is None or not decision.routing_enabled:
            return None
        budget = decision.applied_budget
        route_name = (budget.model_route or "inherit").casefold()
        has_request_controls = any(
            value is not None
            for value in (budget.timeout_seconds, budget.max_retries, budget.max_output_tokens)
        )
        if route_name == "inherit" and not has_request_controls:
            return None
        if route_name == "weak":
            primary = get_model("model_weak")
        elif route_name == "reasoning":
            primary = get_model("model_reasoning")
        else:
            primary = get_settings().config.model
        fallback_models = get_settings().config.fallback_models
        if isinstance(fallback_models, str):
            fallback_models = [model.strip() for model in fallback_models.split(",") if model.strip()]
        models = (primary, *tuple(fallback_models or ()))
        def deployment_setting(key: str) -> str | None:
            value = get_settings().get(key, None)
            if value is None or value == "":
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-blank string when configured")
            return value.strip()

        regular_deployment = deployment_setting("openai.deployment_id")
        deployment_key = {
            "weak": "openai.deployment_id_weak",
            "reasoning": "openai.deployment_id_reasoning",
        }.get(route_name)
        dedicated_model_configured = bool(
            deployment_key and get_settings().get(f"config.model_{route_name}", None)
        )
        if deployment_key is None or not dedicated_model_configured:
            deployment = regular_deployment
        else:
            deployment = deployment_setting(deployment_key)
            if regular_deployment is not None and deployment is None:
                raise ValueError(
                    f"review depth {route_name} model has no matching {deployment_key}"
                )
        fallback_deployments = get_settings().get("openai.fallback_deployments", [])
        if isinstance(fallback_deployments, str):
            fallback_deployments = (
                [item.strip() for item in fallback_deployments.split(",")]
                if fallback_deployments.strip()
                else []
            )
        if fallback_deployments is None:
            fallback_deployments = []
        if not isinstance(fallback_deployments, (list, tuple)):
            raise ValueError("openai.fallback_deployments must be a list or comma-separated string")
        if fallback_deployments:
            if len(fallback_deployments) != len(fallback_models or ()):
                raise ValueError(
                    "review depth model route fallback_deployments must match fallback_models"
                )
            normalized_fallback_deployments = []
            for fallback_deployment in fallback_deployments:
                if fallback_deployment is None:
                    normalized_fallback_deployments.append(None)
                elif isinstance(fallback_deployment, str):
                    normalized_fallback_deployments.append(
                        fallback_deployment.strip() or None
                    )
                else:
                    raise ValueError(
                        "openai.fallback_deployments entries must be strings or null"
                    )
            deployments = (deployment, *normalized_fallback_deployments)
        else:
            if fallback_models and deployment is not None:
                raise ValueError(
                    "review depth model route requires one fallback_deployments entry "
                    "per fallback model when deployments are configured"
                )
            deployments = (deployment,) * len(models)
        return AIModelRoute(
            models=models,
            deployments=deployments,
            timeout_seconds=budget.timeout_seconds,
            model_retries=(budget.max_retries + 1 if budget.max_retries is not None else None),
            max_output_tokens=budget.max_output_tokens,
        )

    def _clear_stale_persistent_bugs_only_review(self) -> None:
        """Remove a prior persistent defect summary after a clean bugs-only rerun."""
        if (not self._provider_mutations_allowed() or
                getattr(self, "_review_thread_lifecycle_blocks_summary", False) or
                self._candidate_verification_blocks_publication() or
                self._review_profile() != "bugs_only" or
                not get_settings().config.publish_output or
                not get_settings().pr_reviewer.persistent_comment or self.incremental.is_incremental):
            return
        threaded_findings = getattr(self, "_review_thread_lifecycle_threaded_findings", False)
        if (
            self._review_thread_lifecycle_enabled()
            and self._review_thread_lifecycle_provider_supported()
            and not threaded_findings
            and (
                not isinstance(getattr(self, "review_thread_reconciliation_artifact", None), Mapping)
                or self.review_thread_reconciliation_artifact.get("authoritative_absence") is not True
            )
        ):
            return
        clear_review = (
            self.git_provider.clear_persistent_review_comment
            if threaded_findings
            else self.git_provider.clear_persistent_review
        )
        clear_review(
            identity_marker=PRReviewIdentity.BUGS_ONLY.value,
            name="bugs-only review",
        )

    async def _prepare_prediction(self, model: str) -> None:
        decision = getattr(self, "review_route_decision", None)
        if (decision is not None and decision.routing_enabled) or review_execution_is_isolated():
            # Model-specific tokenization matters when the selected profile uses a
            # weak or reasoning route. Isolated reviews also rebuild after ticket
            # extraction so diff pruning accounts for the prompt that will be sent.
            self.token_handler = TokenHandler(
                self.git_provider.pr,
                self.vars,
                get_settings().pr_review_prompt.system,
                get_settings().pr_review_prompt.user,
                model=model,
            )
        diff_kwargs = {
            "add_line_numbers_to_hunks": True,
            "disable_extra_lines": False,
            "return_remaining_files": True,
            "return_deleted_files": True,
        }
        context_tokens = getattr(self, "_review_context_tokens", None)
        if context_tokens is not None:
            diff_kwargs["max_context_tokens"] = context_tokens
        if decision is not None and decision.routing_enabled:
            request_options = get_ai_request_options()
            requested_max_output_tokens = (
                request_options.max_output_tokens
                if request_options is not None and request_options.max_output_tokens is not None
                else decision.applied_budget.max_output_tokens
            )
            max_output_tokens = get_effective_litellm_output_token_cap(
                model,
                requested_max_output_tokens,
                claude_extended_thinking_models=getattr(
                    getattr(self, "ai_handler", None),
                    "claude_extended_thinking_models",
                    None,
                ),
                require_bounded_reasoning=True,
            )
            if max_output_tokens is not None:
                # This is the same request-local cap sent to the model. Passing it
                # through diff construction keeps each fallback model's input plus
                # completion within that model's own effective context window.
                diff_kwargs["max_output_tokens"] = max_output_tokens
        output = get_pr_diff(
            self.git_provider,
            self.token_handler,
            model,
            **diff_kwargs,
        )
        if isinstance(output, PRDiffCoverage):
            self.patches_diff = output.diff
            self.remaining_files_list = output.remaining_files
            self.deleted_files_list = output.deleted_files
        else:
            self.patches_diff = output
            self.remaining_files_list = []
            self.deleted_files_list = []

        if self.patches_diff:
            get_logger().debug("PR diff", diff=self.patches_diff)
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
        pipeline = None
        try:
            pipeline = load_specialist_pipeline_config()
            self._specialist_pipeline = pipeline
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
                additional_deterministic_results=self._specialist_deterministic_results(),
                allowed_change_labels=pipeline.allowed_change_labels,
            )
            self.specialist_shadow_input = specialist_input
            self._specialist_input = specialist_input
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
            if pipeline is not None:
                try:
                    self.specialist_shadow_result = unavailable_specialist_batch(
                        pipeline,
                        failure_reason="specialist_batch_failed",
                    )
                except Exception as telemetry_exc:
                    get_logger().debug(
                        "Could not materialize failed specialist telemetry",
                        artifact={"error_class": type(telemetry_exc).__name__},
                    )
            get_logger().warning(
                "Specialist shadow batch failed; continuing the ordinary review",
                artifact={"error_class": type(exc).__name__},
            )

    def _specialist_deterministic_results(self) -> tuple[dict[str, Any], ...]:
        decision = getattr(self, "review_route_decision", None)
        if decision is None or not decision.routing_enabled:
            return ()
        return tuple(
            {
                "id": reason.code,
                "result": {
                    "minimum_depth": reason.minimum_depth.value,
                    "evidence": list(reason.evidence),
                },
            }
            for reason in decision.reasons
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

        self._review_prediction_finish_reason = None
        response, finish_reason = await self.ai_handler.chat_completion(
            model=model,
            temperature=get_settings().config.temperature,
            system=system_prompt,
            user=user_prompt
        )
        normalized_finish_reason = str(finish_reason or "").strip().casefold()
        self._review_prediction_finish_reason = normalized_finish_reason or None

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
            normalized_issue = {
                "relevant_file": relevant_file,
                "issue_header": _BUG_FINDING_HEADERS[finding_type],
                "issue_content": f"{issue_content}\n\n**Trigger:** {trigger}\n\n**Impact:** {impact}",
                "start_line": start_line,
                "end_line": end_line,
            }
            if getattr(self, "_force_no_publish", False):
                normalized_issue.update({
                    "root_cause": root_cause,
                })
            normalized_issues.append(normalized_issue)
            if len(normalized_issues) >= self._maximum_generated_findings():
                break
        return {"review": {"key_issues_to_review": normalized_issues}}

    @staticmethod
    def _candidate_verification_enabled() -> bool:
        if get_checkpoint_stage_sources() is not None:
            return checkpoint_candidate_verification_enabled()
        value = get_settings().pr_reviewer.get("enable_candidate_verification", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _candidate_verification_blocks_publication(
        self,
        review_data: Mapping[str, Any] | None = None,
    ) -> bool:
        """Fail closed unless verification and the final publication agree."""
        artifact = getattr(self, "candidate_verification_artifact", None)
        if not isinstance(artifact, dict):
            return False

        published_finding_count = getattr(
            self, "_candidate_verification_published_finding_count", None
        )
        if review_data is not None:
            issues = (review_data.get("review") or {}).get("key_issues_to_review")
            published_finding_count = len(issues) if isinstance(issues, list) else 0
            self._candidate_verification_published_finding_count = published_finding_count

        if artifact.get("publication_safe") is False:
            return True
        status = artifact.get("status")
        if status == "complete":
            return False
        if status == "no_candidates":
            return published_finding_count not in (None, 0)
        if status == "partial":
            effective_finding_count = (
                published_finding_count
                if published_finding_count is not None
                else int(artifact.get("verified_count") or 0)
            )
            return effective_finding_count <= 0
        return True

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
            verdict = str(decision.get("verdict") or "").strip().lower()
            if verdict not in {"verified", "rejected"}:
                return "invalid_verdict"
            severity = decision.get("normalized_severity")
            normalized_severity = str(severity).strip().lower() if severity is not None else None
            if verdict == "verified" and normalized_severity not in {"low", "medium", "high", "critical"}:
                return "invalid_severity"
            if severity is not None and normalized_severity not in {"low", "medium", "high", "critical"}:
                return "invalid_severity"
            confidence = decision.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                return "invalid_confidence"
            disputed = decision.get("disputed")
            if (
                verdict == "verified" and not isinstance(disputed, bool)
            ) or (disputed is not None and not isinstance(disputed, bool)):
                return "invalid_disputed_signal"
            evidence_status = decision.get("evidence_status")
            if (
                verdict == "verified"
                and str(evidence_status or "").strip().lower()
                not in {"complete", "insufficient"}
            ) or (
                evidence_status is not None
                and str(evidence_status).strip().lower()
                not in {"complete", "insufficient"}
            ):
                return "invalid_evidence_status"
            unresolved_questions = decision.get("unresolved_questions")
            if (verdict == "verified" and not isinstance(unresolved_questions, list)) or (
                unresolved_questions is not None and (
                not isinstance(unresolved_questions, list)
                or any(not isinstance(question, str) or not question.strip() for question in unresolved_questions)
                )
            ):
                return "invalid_unresolved_questions"
            if verdict == "verified":
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

    def _parse_review_prediction(self) -> dict:
        return load_yaml(
            self.prediction.strip(),
            keys_fix_yaml=["ticket_compliance_check", "estimated_effort_to_review_[1-5]:",
                           "risk_level:", "merge_recommendation:", "security_concerns:",
                           "key_issues_to_review:", "relevant_file:", "relevant_line:", "suggestion:"],
            first_key="review",
            last_key="security_concerns",
        )

    @staticmethod
    def _frontier_adjudication_enabled() -> bool:
        if get_checkpoint_stage_sources() is not None:
            return checkpoint_frontier_adjudication_enabled()
        value = get_settings().pr_reviewer.get("enable_frontier_adjudication", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _prepare_frontier_adjudication_config(self) -> bool:
        """Validate and freeze frontier configuration before any model dispatch."""

        try:
            config = load_frontier_adjudication_config(
                get_settings().pr_reviewer,
                get_settings().frontier_adjudication_prompt,
                azure=getattr(getattr(self, "ai_handler", None), "azure", False) is True,
            )
        except (FrontierContractError, TypeError, ValueError):
            self._frontier_adjudication_config = None
            self.frontier_adjudication_artifact = {
                "enabled": True,
                "status": "configuration_invalid",
                "failure": "invalid_configuration",
                "results": [],
                "publication_safe": False,
            }
            return False

        snapshot_context = get_specialist_snapshot_context()
        provider_head_method = getattr(type(self.git_provider), "get_pr_head_sha", None)
        if snapshot_context is None and provider_head_method is GitProvider.get_pr_head_sha:
            # A provider that inherits the base implementation has no refreshable
            # identity to bind pre/post-call checks. Fail before the review and
            # verifier model calls instead of paying for an unusable adjudication.
            preflight_snapshot_id = None
        elif snapshot_context is not None:
            preflight_snapshot_id = snapshot_context.snapshot.snapshot_id
        else:
            try:
                preflight_snapshot_id = self.git_provider.get_pr_head_sha(refresh=False)
            except Exception:
                preflight_snapshot_id = None
        if not isinstance(preflight_snapshot_id, str) or not preflight_snapshot_id.strip():
            self._frontier_adjudication_config = None
            self.frontier_adjudication_artifact = {
                "enabled": True,
                "status": "unavailable",
                "failure": "stable_identity_unavailable",
                "results": [],
                "publication_safe": False,
            }
            return False
        self._frontier_adjudication_config = config
        return True

    @staticmethod
    def _frontier_candidate_severity(candidate: Mapping[str, Any], decision: Mapping[str, Any]):
        explicit = decision.get("normalized_severity") or candidate.get("normalized_severity")
        if explicit:
            return normalize_severity(explicit)
        header = str(candidate.get("issue_header") or "").casefold()
        if "critical" in header or "p0" in header:
            return NormalizedSeverity.CRITICAL
        if "high" in header or "p1" in header:
            return NormalizedSeverity.HIGH
        if "low" in header or "p3" in header:
            return NormalizedSeverity.LOW
        return NormalizedSeverity.MEDIUM

    @staticmethod
    def _frontier_sensitive_categories(
        candidate: Mapping[str, Any],
        route_decision: Optional[ReviewRouteDecision],
    ) -> tuple[str, ...]:
        """Return deterministic candidate-local and PR-wide sensitive categories."""

        candidate_path = safe_repo_path(candidate.get("relevant_file"))
        candidate_paths = {candidate_path} if candidate_path else set()
        lineage_key = candidate.get("_trusted_lineage_key")
        if isinstance(lineage_key, str) and lineage_key.startswith("file:"):
            lineage_path = safe_repo_path(lineage_key.removeprefix("file:"))
            if lineage_path:
                candidate_paths.add(lineage_path)
        categories = ["candidate_verification"] if candidate.get("sensitive_path") is True else []
        if route_decision is not None:
            for reason in route_decision.reasons:
                if not reason.code.startswith("sensitive_category:"):
                    continue
                label_matched = any(
                    isinstance(item, str) and item.startswith("label:")
                    for item in reason.evidence
                )
                matched_paths = {
                    safe_repo_path(item.removeprefix("path:"))
                    for item in reason.evidence
                    if isinstance(item, str) and item.startswith("path:")
                }
                if label_matched or not candidate_paths.isdisjoint(matched_paths):
                    categories.append(reason.code.removeprefix("sensitive_category:"))
        return tuple(dict.fromkeys(categories))

    @staticmethod
    def _frontier_verification_dependency_artifact() -> dict:
        return {
            "enabled": True,
            "status": "unavailable",
            "failure": "candidate_verification_incomplete",
            "results": [],
            "publication_safe": False,
        }

    @staticmethod
    def _finalize_frontier_verification_dependency(
        frontier_artifact: dict,
        verification_artifact: Mapping[str, Any],
    ) -> None:
        verification_status = str(
            verification_artifact.get("status") or "incomplete"
        ).strip() or "incomplete"
        if (
            verification_status == "no_candidates"
            and verification_artifact.get("publication_safe") is True
        ):
            frontier_artifact["status"] = "not_required"
            frontier_artifact.pop("failure", None)
            return
        frontier_artifact.update({
            "status": "unavailable",
            "failure": f"candidate_verification_{verification_status}",
        })

    @classmethod
    def _frontier_signals(
        cls,
        candidate: Mapping[str, Any],
        decision: Mapping[str, Any],
        route_decision: Optional[ReviewRouteDecision],
    ) -> tuple[NormalizedSeverity, FrontierSignals]:
        """Build per-finding escalation signals from verifier and deterministic routing evidence."""

        severity = cls._frontier_candidate_severity(candidate, decision)
        sensitive_categories = cls._frontier_sensitive_categories(candidate, route_decision)
        questions = tuple(decision.get("_unresolved_questions") or ())
        sensitive = bool(sensitive_categories)
        return severity, FrontierSignals(
            sensitive=sensitive,
            severe=severity in {NormalizedSeverity.HIGH, NormalizedSeverity.CRITICAL},
            disputed=decision.get("disputed") is True,
            insufficient_evidence=(
                decision.get("evidence_status") == "insufficient" or bool(questions)
            ),
            deterministic_forced=sensitive,
            deterministic_severity_floor=(
                NormalizedSeverity.HIGH if sensitive else NormalizedSeverity.LOW
            ),
            reasons=tuple(
                f"deterministic_sensitive_route:{category}"
                for category in sensitive_categories
            ),
            unresolved_questions=questions,
        )

    async def _run_frontier_adjudications(
        self,
        verified_findings: list[dict],
        candidates: list[dict],
        evidence: list[dict],
        decisions: list[dict],
    ) -> None:
        """Run selected frontier calls as telemetry without changing verified findings."""

        artifact = {
            "enabled": True,
            "status": "initializing",
            "results": [],
            "publication_safe": False,
        }
        self.frontier_adjudication_artifact = artifact
        config = getattr(self, "_frontier_adjudication_config", None)
        if config is None and not self._prepare_frontier_adjudication_config():
            return
        config = self._frontier_adjudication_config
        batch_deadline = time.monotonic() + config.stage_timeout_seconds

        snapshot_context = get_specialist_snapshot_context()
        if snapshot_context is not None:
            snapshot_id = snapshot_context.snapshot.snapshot_id
            current_identity = snapshot_context.current_snapshot_id
        else:
            try:
                snapshot_id = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.git_provider.get_pr_head_sha,
                        refresh=False,
                    ),
                    timeout=max(0, batch_deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                artifact.update({"status": "unavailable", "failure": "stage_timeout"})
                return
            except Exception:
                snapshot_id = None

            def current_identity():
                return self.git_provider.get_pr_head_sha(refresh=True)

        if not snapshot_id:
            artifact.update({"status": "unavailable", "failure": "stable_identity_unavailable"})
            return

        accepted_candidate_ids = {
            item.get("candidate_id")
            for item in decisions
            if (
                isinstance(item, dict)
                and item.get("candidate_id")
                and item.get("verdict") == "verified"
                and not item.get("reason")
            )
        }
        candidates_by_identity = {}
        for candidate in candidates:
            if candidate.get("candidate_id") not in accepted_candidate_ids:
                continue
            identity = verified_finding_identity(candidate)
            if identity is not None:
                candidates_by_identity.setdefault(identity, []).append(candidate)
        decision_by_candidate = {
            item.get("candidate_id"): item
            for item in decisions
            if isinstance(item, dict) and item.get("candidate_id")
        }
        deterministic_sensitive_identities = set()
        for candidate in candidates:
            if candidate.get("sensitive_path") is not True:
                continue
            identity = verified_finding_identity(candidate)
            if identity is not None:
                deterministic_sensitive_identities.add(identity)
        route_decision = getattr(self, "review_route_decision", None)
        risk_policy_version = (
            route_decision.policy_version if route_decision is not None else "review-router-unavailable"
        )
        max_calls = config.max_calls
        eligible_count = 0
        call_count = 0
        eligible_findings = []
        for finding in verified_findings:
            identity = (finding.get("root_cause_id"), finding.get("trusted_stable_key"))
            matching_candidates = candidates_by_identity.get(identity, [])
            if len(matching_candidates) != 1:
                artifact["results"].append({
                    "stable_finding_id": finding.get("trusted_stable_key"),
                    "state": "unavailable",
                    "failure_reason": "trusted_candidate_identity_unavailable",
                    "publication_safe": False,
                })
                continue
            candidate = dict(matching_candidates[0])
            if identity in deterministic_sensitive_identities:
                candidate["sensitive_path"] = True
            decision = decision_by_candidate.get(candidate.get("candidate_id"), {})
            severity, signals = self._frontier_signals(
                candidate,
                decision,
                route_decision,
            )
            if not signals.requires_escalation:
                continue
            eligible_count += 1
            eligible_findings.append((finding, candidate, severity, signals))

        deterministic_severity_rank = {
            NormalizedSeverity.LOW: 0,
            NormalizedSeverity.MEDIUM: 1,
            NormalizedSeverity.HIGH: 2,
            NormalizedSeverity.CRITICAL: 3,
        }
        # Verifier severity, dispute signals, and response order are not trusted budget
        # inputs. The stable sort preserves the existing order for deterministic ties.
        eligible_findings.sort(key=lambda item: (
            not item[3].deterministic_forced,
            -deterministic_severity_rank.get(
                item[3].deterministic_severity_floor, len(deterministic_severity_rank)
            ),
        ))
        for finding, candidate, severity, signals in eligible_findings:
            if call_count >= max_calls:
                artifact["results"].append({
                    "stable_finding_id": finding.get("trusted_stable_key"),
                    "state": "unavailable",
                    "failure_reason": "call_budget_exhausted",
                    "publication_safe": False,
                })
                continue
            if time.monotonic() >= batch_deadline:
                artifact["results"].append({
                    "stable_finding_id": finding.get("trusted_stable_key"),
                    "state": "timeout",
                    "failure_reason": "stage_timeout_exhausted",
                    "publication_safe": False,
                })
                continue
            frontier_evidence = build_frontier_evidence(
                str(candidate.get("candidate_id") or ""), evidence
            )
            try:
                frontier_candidate = FrontierCandidate(
                    stable_finding_id=str(finding.get("trusted_stable_key") or ""),
                    root_cause_id=str(finding.get("root_cause_id") or ""),
                    path=str(finding.get("relevant_file") or ""),
                    side=str(finding.get("side") or "new"),
                    start_line=int(finding.get("start_line")),
                    end_line=int(finding.get("end_line")),
                    title=str(finding.get("issue_header") or ""),
                    explanation=str(finding.get("issue_content") or ""),
                    trigger=str(finding.get("trigger") or ""),
                    impact=str(finding.get("impact") or ""),
                    verified_severity=severity,
                )
                request = FrontierAdjudicationRequest(
                    candidate=frontier_candidate,
                    evidence=frontier_evidence,
                    signals=signals,
                    snapshot_id=str(snapshot_id),
                    configuration_hash=config.configuration_hash,
                    prompt_hash=config.prompt_hash,
                    policy_version=config.policy_version,
                    risk_policy_version=risk_policy_version,
                )
            except (FrontierContractError, TypeError, ValueError):
                artifact["results"].append({
                    "stable_finding_id": finding.get("trusted_stable_key"),
                    "state": "unavailable",
                    "failure_reason": "adjudication_input_incomplete",
                    "publication_safe": False,
                })
                continue
            result = await run_frontier_adjudication(
                request,
                config,
                self.ai_handler,
                current_identity=current_identity,
                deadline_monotonic=batch_deadline,
            )
            call_count += 1
            artifact["results"].append(result.to_telemetry_dict())
        if any(
            result.get("state") not in {"confirmed", "rejected"}
            for result in artifact["results"]
        ):
            artifact["status"] = "partial"
        elif not eligible_count:
            artifact["status"] = "not_required"
        else:
            artifact["status"] = "complete"

    async def _run_candidate_verification(self) -> None:
        """Verify review candidates against bounded base-branch repository evidence."""
        frontier_enabled = self._frontier_adjudication_enabled()
        frontier_dependency_artifact = None
        if frontier_enabled:
            frontier_dependency_artifact = self._frontier_verification_dependency_artifact()
        self.frontier_adjudication_artifact = frontier_dependency_artifact
        artifact = {
            "enabled": True,
            "status": "initializing",
            "model_calls": 0,
            "publication_safe": False,
            "first_pass_finish_reason": getattr(self, "_review_prediction_finish_reason", None),
            "first_pass_generation_complete": (
                getattr(self, "_review_prediction_finish_reason", None) == "stop"
            ),
            "specialist_prioritization": {"status": "initializing"},
        }
        verifier_started = None
        verification_config = None
        details_before = None
        self.candidate_verification_artifact = artifact
        try:
            runtime_settings = get_settings()
            route_candidate_cap = getattr(self, "_review_max_verification_candidates", None)

            def current_claude_extended_thinking_models() -> Optional[tuple[str, ...]]:
                raw_models = getattr(self.ai_handler, "claude_extended_thinking_models", None)
                if (
                    isinstance(raw_models, (list, tuple))
                    and all(isinstance(item, str) and item.strip() for item in raw_models)
                ):
                    return tuple(raw_models)
                return None

            claude_extended_thinking_models = current_claude_extended_thinking_models()

            def provider_controls_unchanged() -> bool:
                current_claude_models = current_claude_extended_thinking_models()
                return (
                    current_claude_models == claude_extended_thinking_models
                    and candidate_verification_provider_controls_hash(
                        get_settings(),
                        claude_extended_thinking_models=current_claude_models,
                    ) == verification_config.provider_controls_hash
                )

            def resolve_verifier_output_cap(route_model: str, request_cap: Optional[int]) -> Optional[int]:
                kwargs = {"require_bounded_reasoning": True}
                if claude_extended_thinking_models is not None:
                    kwargs["claude_extended_thinking_models"] = list(claude_extended_thinking_models)
                return get_effective_litellm_output_token_cap(route_model, request_cap, **kwargs)

            try:
                verification_config = load_production_candidate_verification_config(
                    settings=runtime_settings,
                    azure=getattr(self.ai_handler, "azure", False) is True,
                    max_candidates_override=route_candidate_cap,
                    strict_output_policy=frontier_enabled,
                    default_output_token_cap=OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD,
                    claude_extended_thinking_models=claude_extended_thinking_models,
                    output_token_cap_resolver=resolve_verifier_output_cap,
                )
            except CandidateVerificationOutputBudgetError:
                artifact.update({
                    "status": "verifier_route_invalid",
                    "failure": "invalid_output_budget",
                    "verified_count": 0,
                })
                return
            except (TypeError, ValueError):
                artifact.update({
                    "status": "verifier_route_invalid",
                    "failure": "invalid_model_route",
                    "verified_count": 0,
                })
                return
            review_data = self._parse_review_prediction()
            if not isinstance(review_data, dict) or not isinstance(review_data.get("review"), dict):
                artifact.update({"status": "candidate_parse_failed", "verified_count": 0})
                return
            raw_candidate_input = review_data["review"].get("key_issues_to_review")
            raw_candidate_list_valid = isinstance(raw_candidate_input, list)
            proposed_candidate_count = (
                len(raw_candidate_input) if raw_candidate_list_valid else 0
            )
            artifact.update({
                "proposal_source": "first_pass_review",
                "proposal_shape": "list" if raw_candidate_list_valid else "invalid",
                "proposed_candidate_count": proposed_candidate_count,
            })
            self.verified_review_data = copy.deepcopy(review_data)
            self.verified_review_data["review"]["key_issues_to_review"] = []

            if not raw_candidate_list_valid:
                artifact.update({
                    "status": "candidate_input_invalid",
                    "candidate_count": 0,
                    "accepted_model_candidate_count": 0,
                    "candidate_rejection_count": 0,
                    "verified_count": 0,
                    "publication_safe": False,
                })
                return

            capability = getattr(self.git_provider, "supports_repo_file_fetching", None)
            if not callable(capability) or not capability():
                artifact.update({"status": "unsupported_provider", "verified_count": 0})
                return
            consume_specialist_prioritization = verification_config.consume_specialist_prioritization
            artifact.update({
                "configuration_hash": verification_config.configuration_hash,
                "prompt_hash": verification_config.prompt_hash,
                "config_schema_version": verification_config.schema_version,
                "prompt_version": verification_config.prompt_version,
                "input_schema_version": verification_config.input_schema_version,
                "output_schema_version": verification_config.output_schema_version,
                "static_analysis_evidence_hash": verification_config.static_analysis_evidence_hash,
                "provider_controls_hash": verification_config.provider_controls_hash,
                "specialist_prioritization": {
                    "status": "pending" if consume_specialist_prioritization else "disabled"
                },
            })
            budgets = verification_config.budgets
            sensitive_globs = verification_config.sensitive_path_globs
            diff_files = self.git_provider.get_diff_files()
            # A truncated provider patch hides code the first pass never saw, so a
            # clean run over it can never be authoritative evidence of absence.
            artifact["reviewed_patches_complete"] = bool(diff_files) and all(
                getattr(diff_file, "patch_is_complete", False) is True
                for diff_file in diff_files
            )
            # [ignore] patterns drop changed files before diff_files is built, so
            # they never appear in remaining_files_list either. A provider that does
            # not report its exclusions leaves this unknown, which fails closed.
            excluded_paths = getattr(self.git_provider, "excluded_diff_file_paths", None)
            artifact["reviewed_file_inventory_complete"] = (
                isinstance(excluded_paths, tuple) and not excluded_paths
            )
            candidates, candidate_rejections = prepare_candidates(
                review_data,
                diff_files,
                sensitive_globs,
                budgets.max_candidates,
                max_sensitive_candidates=verification_config.max_sensitive_candidates,
            )
            sensitive_candidates = [
                candidate for candidate in candidates if candidate.get("sensitive_path")
            ]
            accepted_model_candidate_count = sum(
                1 for candidate in candidates if not candidate.get("sensitive_path")
            )
            model_candidate_rejection_count = sum(
                1 for rejection in candidate_rejections
                if not rejection.get("sensitive_path")
            )
            model_candidate_validation_rejection_count = sum(
                1 for rejection in candidate_rejections
                if (
                    not rejection.get("sensitive_path")
                    and rejection.get("reason") != "duplicate_candidate"
                )
            )
            model_candidate_validation_incomplete = bool(
                model_candidate_validation_rejection_count
            )
            model_candidate_budget_exhausted = any(
                rejection.get("reason") == "candidate_budget_exhausted"
                for rejection in candidate_rejections
                if not rejection.get("sensitive_path")
            )
            model_candidate_coverage_incomplete = bool(
                proposed_candidate_count
                and (
                    not accepted_model_candidate_count
                    or model_candidate_validation_incomplete
                    or model_candidate_budget_exhausted
                )
            )
            if proposed_candidate_count == 0:
                model_candidate_coverage_status = "complete"
            elif model_candidate_coverage_incomplete:
                model_candidate_coverage_status = "incomplete"
            elif model_candidate_rejection_count:
                model_candidate_coverage_status = "partial"
            else:
                model_candidate_coverage_status = "complete"
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
                "budget": verification_config.max_sensitive_candidates,
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
                "accepted_model_candidate_count": accepted_model_candidate_count,
                "sensitive_candidate_count": len(sensitive_candidates),
                "candidate_rejection_count": len(candidate_rejections),
                "model_candidate_coverage": {
                    "status": model_candidate_coverage_status,
                    "proposed_count": proposed_candidate_count,
                    "accepted_count": accepted_model_candidate_count,
                    "rejected_count": model_candidate_rejection_count,
                },
                "candidate_rejections": candidate_rejections,
                "verified_count": 0,
            })
            if not candidates:
                if sensitive_coverage_incomplete:
                    artifact.update({
                        "status": "sensitive_audit_coverage_incomplete",
                        "publication_safe": False,
                    })
                elif model_candidate_coverage_incomplete:
                    artifact.update({
                        "status": "candidate_validation_incomplete",
                        "publication_safe": False,
                    })
                else:
                    artifact.update({
                        "status": "no_candidates",
                        "publication_safe": True,
                        "finding_limit_dropped": 0,
                    })
                return

            verifier_route = verification_config.route
            model = verifier_route.models[0]
            artifact["model"] = model
            encoder = TokenEncoder.get_token_encoder(model)
            evidence, retrieval_artifact = await retrieve_evidence(
                self.git_provider,
                candidates,
                budgets,
                verification_config.static_analysis_evidence,
                diff_files=diff_files,
                token_counter=lambda value: len(encoder.encode(value, disallowed_special=())),
                prefer_pr_head=bool(
                    getattr(getattr(self, "incremental", None), "is_incremental", False)
                ),
            )
            artifact["retrieval"] = retrieval_artifact
            if verification_config.max_calls < 1:
                artifact["status"] = "model_call_budget_exhausted"
                return

            environment = Environment(
                autoescape=select_autoescape(default_for_string=False),
                undefined=StrictUndefined,
            )
            route_encoders = {
                route_model: TokenEncoder.get_token_encoder(route_model)
                for route_model in verifier_route.models
            }
            if not provider_controls_unchanged():
                artifact.update({
                    "status": "verifier_request_context_changed",
                    "failure": "provider_controls_changed",
                    "verified_count": 0,
                })
                return
            route_completion_reserves = dict(zip(
                verifier_route.models,
                verification_config.effective_output_token_caps,
                strict=True,
            ))
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
                rendered_system = environment.from_string(verification_config.system_prompt).render(variables)
                rendered_user = environment.from_string(verification_config.user_prompt).render(variables)
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
                "evidence_content_fraction": (
                    1.0 if evidence_fraction >= 1.0
                    else min(round(evidence_fraction, 4), 0.9999)
                ),
                "changed_diff_fraction": (
                    1.0 if changed_diff_fraction >= 1.0
                    else min(round(changed_diff_fraction, 4), 0.9999)
                ),
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
                if not provider_controls_unchanged():
                    raise RuntimeError("candidate verifier provider controls changed after capture")
                prediction_result = await self.ai_handler.chat_completion(
                    model=attempt_model,
                    temperature=verification_config.temperature,
                    system=system_prompt,
                    user=user_prompt,
                )
                if not provider_controls_unchanged():
                    raise RuntimeError("candidate verifier provider controls changed during request")
                artifact["model"] = attempt_model
                return prediction_result

            prediction, _ = await retry_with_fallback_models(
                call_verifier,
                model_route=verifier_route,
            )
            try:
                verification_data = load_yaml(
                    prediction.strip(),
                    keys_fix_yaml=["verification:", "decisions:", "candidate_id:", "relevant_file:",
                                   "start_line:", "end_line:", "evidence_paths:"],
                    first_key="verification",
                    last_key="decisions",
                    reject_duplicate_keys=verification_config.strict_output_policy,
                )
            except DuplicateYamlKeyError:
                artifact.update({
                    "status": "verifier_response_invalid",
                    "failure": "duplicate_mapping_key",
                    "verified_count": 0,
                })
                return
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
            visible_coverage = prompt_evidence_coverage(
                candidates,
                prompt_evidence,
                retrieval_artifact["requests"],
            )
            prompt_evidence_incomplete = visible_coverage["status"] != "complete"
            verified_findings, decisions = apply_verification_decisions(
                candidates,
                prompt_evidence,
                verification_data,
                retrieval_requests=retrieval_artifact["requests"],
            )
            if frontier_enabled:
                await self._run_frontier_adjudications(
                    verified_findings,
                    candidates,
                    prompt_evidence,
                    decisions,
                )
            max_findings = verification_config.max_findings
            published_findings = (
                [] if model_candidate_coverage_incomplete
                else verified_findings[:max_findings]
            )
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
                model_candidate_coverage_incomplete
                or sensitive_coverage_incomplete
                or prompt_evidence_incomplete
                or rejected_verified_claim
                or any(
                    not retrieval_request_is_complete(request)
                    for request in retrieval_artifact["requests"]
                )
            ) else "complete"
            artifact.update({
                "status": (
                    "candidate_validation_incomplete"
                    if model_candidate_coverage_incomplete else verification_status
                ),
                "publication_safe": (
                    not model_candidate_coverage_incomplete
                    and not sensitive_coverage_incomplete
                    and (verification_status == "complete" or bool(published_findings))
                ),
                "decisions": decisions,
                "prompt_evidence_coverage": visible_coverage,
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
            if (
                frontier_dependency_artifact is not None
                and self.frontier_adjudication_artifact is frontier_dependency_artifact
            ):
                self._finalize_frontier_verification_dependency(
                    frontier_dependency_artifact,
                    artifact,
                )
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
            if verification_config is not None:
                self._record_candidate_verification_stage(verification_config, artifact)
            get_logger().info("Candidate verification finished", artifact=telemetry_safe_artifact(artifact))

    @staticmethod
    def _record_candidate_verification_stage(verification_config, artifact: Mapping[str, Any]) -> None:
        """Finalize source-free verifier telemetry for checkpoint evaluation."""

        status = str(artifact.get("status") or "unavailable").strip().lower()
        state, failure_reason = {
            "complete": ("success", None),
            "no_candidates": ("not_required", None),
            "partial": ("partial", "verification_coverage_partial"),
            "candidate_validation_incomplete": ("partial", "candidate_validation_incomplete"),
            "prompt_budget_exhausted": ("input_budget_exhausted", "prompt_budget_exhausted"),
            "verifier_response_invalid": ("malformed_output", "verifier_response_invalid"),
            "verifier_failed": ("provider_failure", "verifier_failed"),
        }.get(status, ("unavailable", status or "verification_unavailable"))
        explicit_failure = str(artifact.get("failure") or "").strip()
        if failure_reason is not None and explicit_failure:
            normalized_failure = re.sub(r"(?<!^)(?=[A-Z])", "_", explicit_failure).lower()
            normalized_failure = re.sub(r"[^a-z0-9_]+", "_", normalized_failure).strip("_")[:128]
            if _MACHINE_FAILURE_REASON_RE.fullmatch(normalized_failure):
                failure_reason = normalized_failure
        details = get_run_details()
        existing = (
            details.specialist_runs.get("candidate_verification")
            if details is not None
            else None
        )
        has_observed_model = existing is not None and existing.model_used is not None
        # No-call exits still need the frozen route identity to materialize the
        # planned stage. ai_call_count remains zero, so this does not claim that
        # the configured primary actually executed.
        record_specialist_result(
            "candidate_verification",
            prompt_version=verification_config.prompt_version,
            input_schema_version=verification_config.input_schema_version,
            schema_version=verification_config.output_schema_version,
            state=state,
            latency_seconds=float(artifact.get("verifier_latency_seconds") or 0.0),
            failure_reason=failure_reason,
            model=None if has_observed_model else verification_config.route.models[0],
            deployment_id=None if has_observed_model else verification_config.route.deployments[0],
            fallback_used=None if has_observed_model else False,
        )

    def _publish_structured_review_data(
        self,
        data: Mapping[str, Any],
        *,
        source_free: bool = False,
    ) -> None:
        """Publish one isolated, provider-neutral review snapshot."""

        capture_structured_review = getattr(self, "_force_no_publish", False)
        structured_publisher = getattr(self.git_provider, "publish_structured_review", None)
        if (
            not capture_structured_review
            and (not self._provider_mutations_allowed() or not callable(structured_publisher))
        ):
            return
        # Deep-copy the data: dict(data) is shallow, so structured_data["review"]
        # would alias data["review"], which _prepare_pr_review mutates later.
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
            "omitted_files": (
                [] if source_free else sorted(set(self.remaining_files_list))
            ),
            "deleted_files": (
                []
                if source_free
                else sorted(set(getattr(self, "deleted_files_list", [])))
            ),
        }
        if getattr(self, "candidate_verification_artifact", None) is not None:
            structured_data["candidate_verification"] = telemetry_safe_artifact(
                self.candidate_verification_artifact
            )
        review_thread_artifact = getattr(self, "review_thread_reconciliation_artifact", None)
        if not source_free and isinstance(review_thread_artifact, Mapping):
            structured_data["review_thread_lifecycle"] = copy.deepcopy(review_thread_artifact)
        frontier_artifact = getattr(self, "frontier_adjudication_artifact", None)
        if frontier_artifact is not None:
            structured_data["frontier_adjudication"] = copy.deepcopy(frontier_artifact)
        adjudication_runs = adjudication_runs_to_dict(details)
        if adjudication_runs:
            structured_data["adjudication_runs"] = adjudication_runs
        review_route_decision = getattr(self, "review_route_decision", None)
        if (
            not source_free
            and review_route_decision is not None
            and review_route_decision.routing_enabled
        ):
            structured_data["metadata"]["review_route"] = review_route_decision_to_dict(
                review_route_decision
            )
        specialist_shadow_result = getattr(self, "specialist_shadow_result", None)
        if not source_free and specialist_shadow_result is not None:
            structured_data["metadata"]["specialist_shadow"] = specialist_shadow_result.to_dict()
        if capture_structured_review:
            self._structured_review_result = copy.deepcopy(structured_data)

        if not self._provider_mutations_allowed() or not callable(structured_publisher):
            return
        structured_publisher(structured_data)

    def _prepare_pr_review(self) -> str:
        """
        Prepare the PR review by processing the AI prediction and generating a markdown-formatted text that summarizes
        the feedback.
        """
        self._prepared_push_output_payload = None
        self._review_thread_lifecycle_threaded_findings = False
        data = copy.deepcopy(getattr(self, "verified_review_data", None)) or self._parse_review_prediction()

        if not isinstance(data, dict) or 'review' not in data:
            get_logger().exception("Failed to parse review data", artifact={"data": data})
            return ""
        data = self._normalize_bugs_only_review(data)
        if not getattr(self, "_force_no_publish", False):
            issues = (data.get("review") or {}).get("key_issues_to_review")
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        issue.pop("normalized_severity", None)
        data = self._apply_finding_budget(data)
        data = self._apply_publication_budget(data)
        output_data = data
        candidate_verification_blocked = self._candidate_verification_blocks_publication(
            data
        )

        lifecycle_enabled = self._review_thread_lifecycle_enabled()
        lifecycle_owns_inline_publication = (
            lifecycle_enabled
            and self._review_thread_lifecycle_provider_supported()
        )
        lifecycle_data = output_data
        if (
            lifecycle_owns_inline_publication
            and not candidate_verification_blocked
            and self._provider_mutations_allowed()
            and get_settings().config.publish_output
        ):
            lifecycle_data = self._apply_review_thread_lifecycle(output_data)

        self._publish_structured_review_data(output_data)

        if candidate_verification_blocked or getattr(self, "_review_thread_lifecycle_blocks_summary", False):
            return ""

        if self._provider_mutations_allowed():
            github_action_output(output_data, 'review')

        data = lifecycle_data

        # move data['review'] 'key_issues_to_review' key to the end of the dictionary
        if 'key_issues_to_review' in data['review']:
            key_issues_to_review = data['review'].pop('key_issues_to_review')
            data['review']['key_issues_to_review'] = key_issues_to_review

        if (not lifecycle_owns_inline_publication and self._provider_mutations_allowed() and
                get_settings().config.publish_output and
                get_settings().pr_reviewer.get('inline_key_issues', False)):
            data = self._publish_key_issues_as_inline_comments(data)

        if self._review_profile() == "bugs_only" and not (
                (data.get("review") or {}).get("key_issues_to_review")) and not (
                getattr(self, "_review_thread_summary_fallbacks", ())
                or getattr(self, "_review_thread_lifecycle_notice", None)
        ):
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
        markdown_text = self._append_review_thread_lifecycle_summary(markdown_text)

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

        # Snapshot sink data while rendering, then let the async run path await the synchronous
        # channel implementation in a worker thread. This keeps request event loops responsive
        # without making synchronous push_outputs callers fire-and-forget.
        self._prepared_push_output_payload = copy.deepcopy(output_data.get('review', {}))

        # Add custom labels from the review prediction (effort, security)
        if self._provider_mutations_allowed() and self._review_profile() != "bugs_only":
            self.set_review_labels(data)

        if markdown_text is None or len(markdown_text) == 0:
            markdown_text = ""

        return markdown_text

    async def _push_prepared_review_output(self, markdown_text: str) -> None:
        payload = copy.deepcopy(getattr(self, "_prepared_push_output_payload", None))
        if payload is None or not self._provider_mutations_allowed() or not get_settings().config.publish_output:
            return
        try:
            await asyncio.to_thread(push_outputs, "review", payload=payload, markdown=markdown_text)
        except Exception as e:
            # push_outputs is defensive, but keep an executor/runtime failure non-fatal too.
            get_logger().warning(f"push_outputs dispatch failed: {type(e).__name__}")

    def _maximum_generated_findings(self) -> int:
        limit = getattr(self, "_review_max_findings", None)
        if isinstance(limit, int) and not isinstance(limit, bool):
            return limit
        return get_settings().pr_reviewer.num_max_findings

    def _apply_finding_budget(self, data: dict) -> dict:
        return self._limit_findings(data, getattr(self, "_review_max_findings", None))

    def _apply_publication_budget(self, data: dict) -> dict:
        return self._limit_findings(data, getattr(self, "_review_max_published_findings", None))

    @staticmethod
    def _review_thread_lifecycle_enabled() -> bool:
        value = get_settings().get("review_thread_lifecycle.enabled", False)
        return parse_env_bool(value) is True

    def _review_thread_lifecycle_provider_supported(self) -> bool:
        capability = getattr(self.git_provider, "supports_review_thread_lifecycle", None)
        return callable(capability) and capability() is True

    def _review_thread_absence_is_authoritative(self, published_finding_count: int) -> bool:
        """Allow obsolete cleanup only after a coverage-complete full verification run."""
        artifact = getattr(self, "candidate_verification_artifact", None)
        incremental = getattr(self, "incremental", None)
        route_decision = getattr(self, "review_route_decision", None)
        publication_threshold = (
            getattr(getattr(route_decision, "applied_budget", None), "publication_threshold", None)
            if route_decision is not None and getattr(route_decision, "routing_enabled", False)
            else None
        )
        finding_limit_dropped = artifact.get("finding_limit_dropped") if isinstance(artifact, Mapping) else None
        proposed_candidate_count = (
            artifact.get("proposed_candidate_count") if isinstance(artifact, Mapping) else None
        )
        generation_cap = self._maximum_generated_findings()
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("publication_safe") is not True
            or artifact.get("status") not in {"complete", "no_candidates"}
            or artifact.get("first_pass_generation_complete") is not True
            or artifact.get("reviewed_patches_complete") is not True
            or artifact.get("reviewed_file_inventory_complete") is not True
            or bool(getattr(incremental, "is_incremental", False))
            or bool(getattr(self, "_review_shadow_only", False))
            or bool(getattr(self, "remaining_files_list", []))
            or not isinstance(finding_limit_dropped, int)
            or isinstance(finding_limit_dropped, bool)
            or finding_limit_dropped != 0
            or not isinstance(generation_cap, int)
            or isinstance(generation_cap, bool)
            or generation_cap < 1
            or not isinstance(proposed_candidate_count, int)
            or isinstance(proposed_candidate_count, bool)
            or proposed_candidate_count < 0
            or proposed_candidate_count >= generation_cap
            or (
                publication_threshold is not None
                and str(publication_threshold).strip().casefold() not in {"none", "low"}
            )
        ):
            return False
        verified_count = artifact.get("verified_count")
        return (
            isinstance(verified_count, int)
            and not isinstance(verified_count, bool)
            and verified_count == published_finding_count
        )

    @staticmethod
    def _review_thread_finding_body(issue: Mapping[str, Any]) -> str:
        issue_content = _SUGGESTION_FENCE_RE.sub(
            "```text", str(issue.get("issue_content") or "").strip()
        )
        issue_header = str(issue.get("issue_header") or "").strip()
        if issue_header.lower() == "possible bug":
            issue_header = "Possible Issue"
        if not issue_content:
            raise ValueError("verified finding requires issue_content")
        return f"**{issue_header}**\n\n{issue_content}" if issue_header else issue_content

    def _desired_review_threads(
        self,
        issues: list[Mapping[str, Any]],
        *,
        repository: str,
        pull_request_number: int,
    ) -> tuple[DesiredReviewThread, ...]:
        identities = finding_identities_from_verified_findings(
            issues,
            repository=repository,
            pull_request_number=pull_request_number,
        )
        diff_files = {}
        for file in self.git_provider.get_diff_files() or []:
            filename = str(getattr(file, "filename", "") or "").strip()
            if not filename:
                continue
            diff_files[filename] = file
            diff_files.setdefault(filename.lstrip("/"), file)
            old_filename = str(getattr(file, "old_filename", "") or "").strip()
            if old_filename:
                diff_files.setdefault(old_filename, file)
                diff_files.setdefault(old_filename.lstrip("/"), file)

        desired = []
        for issue, identity in zip(issues, identities, strict=True):
            body = self._review_thread_finding_body(issue)
            comment = self._build_key_issue_comment(issue, diff_files, allow_old_side=True)
            anchor = None
            if comment is not None:
                start_line = int(comment["relevant_lines_start"])
                end_line = int(comment["relevant_lines_end"])
                anchor = ReviewThreadAnchor(
                    path=comment["relevant_file"],
                    line=end_line,
                    start_line=start_line if start_line != end_line else None,
                    side="LEFT" if comment["relevant_side"] == "old" else "RIGHT",
                )
                body = comment["body"]
            desired.append(DesiredReviewThread(identity=identity, anchor=anchor, body=body))
        return tuple(desired)

    def _set_review_thread_lifecycle_unavailable(
        self,
        status: str,
        notice_reason: str,
        *,
        failure: Optional[str] = None,
    ) -> None:
        artifact = {
            "enabled": True,
            "status": status,
            "head_match": None,
            "authoritative_absence": False,
            "mutations_attempted": 0,
            "requires_fresh_inventory": False,
            "summary_fallback_count": 0,
            "reused_summary_fallback_count": 0,
            "failure_kinds": {status: 1},
            "results": {
                "created": 0,
                "updated": 0,
                "unchanged": 0,
                "resolved": 0,
                "failed": 1,
            },
            "metrics": {
                "actions": {
                    kind.value: 0
                    for kind in _REVIEW_THREAD_ACTION_KINDS
                },
                "action_states": {
                    f"{kind.value}.{state.value}": 0
                    for kind in _REVIEW_THREAD_ACTION_KINDS
                    for state in _REVIEW_THREAD_ACTION_STATES
                },
                "states": {
                    state.value: 0
                    for state in _REVIEW_THREAD_ACTION_STATES
                },
                "summary_fallbacks": {
                    reason.value: 0
                    for reason in _REVIEW_THREAD_FALLBACK_REASONS
                },
                "unavailable": {status: 1},
            },
        }
        if failure:
            artifact["failure"] = failure
        self.review_thread_reconciliation_artifact = artifact
        self._review_thread_lifecycle_blocks_summary = status in {"head_unavailable", "stale_head"}
        self._review_thread_lifecycle_notice = (
            "> ⚠️ **PR-Agent inline lifecycle:** "
            f"{notice_reason}; verified findings remain in this review summary."
        )
        get_logger().warning(
            "Review-thread lifecycle did not run",
            artifact=copy.deepcopy(artifact),
        )

    def _apply_review_thread_lifecycle(self, data: dict) -> dict:
        """Reconcile verified GitHub findings while preserving a visible summary on failure."""
        self._review_thread_lifecycle_threaded_findings = False
        self._review_thread_summary_fallbacks = ()
        self._review_thread_lifecycle_notice = None
        self._review_thread_lifecycle_blocks_summary = False
        self.review_thread_reconciliation_artifact = {
            "enabled": True,
            "status": "initializing",
            "mutations_attempted": 0,
            "summary_fallback_count": 0,
        }
        issues = (data.get("review") or {}).get("key_issues_to_review")
        if not isinstance(issues, list):
            self._set_review_thread_lifecycle_unavailable(
                "configuration_invalid", "verified finding output was unavailable"
            )
            return data
        if not isinstance(getattr(self, "candidate_verification_artifact", None), Mapping):
            self._set_review_thread_lifecycle_unavailable(
                "configuration_invalid", "candidate verification did not provide a trusted result"
            )
            return data
        if not self._review_thread_lifecycle_provider_supported():
            self._set_review_thread_lifecycle_unavailable(
                "unsupported_provider", "stable thread reconciliation is GitHub-only"
            )
            return data

        repository = str(getattr(self.git_provider, "repo", "") or "").strip()
        pull_request_number = getattr(self.git_provider, "pr_num", None)
        if (
            not repository
            or isinstance(pull_request_number, bool)
            or not isinstance(pull_request_number, int)
            or pull_request_number < 1
        ):
            self._set_review_thread_lifecycle_unavailable(
                "provider_identity_unavailable", "the pull-request identity could not be verified"
            )
            return data
        try:
            desired_threads = self._desired_review_threads(
                issues,
                repository=repository,
                pull_request_number=pull_request_number,
            )
        except (TypeError, ValueError) as error:
            self._set_review_thread_lifecycle_unavailable(
                "finding_identity_invalid",
                "verified finding identity was incomplete or ambiguous",
                failure=type(error).__name__,
            )
            return data

        try:
            expected_head_sha = self.git_provider.get_pr_head_sha()
            current_head_sha = self.git_provider.get_pr_head_sha(refresh=True)
        except Exception as error:
            self._set_review_thread_lifecycle_unavailable(
                "head_unavailable",
                "the reviewed pull-request head could not be confirmed",
                failure=type(error).__name__,
            )
            return data
        if not expected_head_sha or current_head_sha != expected_head_sha:
            self._set_review_thread_lifecycle_unavailable(
                "stale_head", "the pull-request head changed before reconciliation"
            )
            return data

        try:
            existing_threads = self.git_provider.get_review_thread_snapshots(require_viewer_bot=True)
            existing_summary_bodies = ()
            summary_is_replaced = bool(
                get_settings().pr_reviewer.persistent_comment
                and not self.incremental.is_incremental
            )
            if not summary_is_replaced:
                existing_summary_bodies = self.git_provider.get_bot_owned_review_summary_bodies()
            inventory_head_sha = self.git_provider.get_pr_head_sha(refresh=True)
        except Exception as error:
            self._set_review_thread_lifecycle_unavailable(
                "inventory_unavailable",
                "the complete review-thread inventory could not be read",
                failure=type(error).__name__,
            )
            return data
        if inventory_head_sha != expected_head_sha:
            self._set_review_thread_lifecycle_unavailable(
                "stale_head", "the pull-request head changed while threads were inventoried"
            )
            return data

        obsolete_policy = str(
            get_settings().get("review_thread_lifecycle.obsolete_thread_policy", "keep") or ""
        ).strip().casefold()
        if obsolete_policy not in {"keep", "resolve", "mark_fixed"}:
            self._set_review_thread_lifecycle_unavailable(
                "configuration_invalid", "the obsolete-thread policy is invalid"
            )
            return data
        authoritative_absence = self._review_thread_absence_is_authoritative(len(issues))
        try:
            plan = plan_review_thread_actions(
                desired_threads,
                existing_threads,
                expected_head_sha,
                obsolete_policy=obsolete_policy,
                authoritative_absence=authoritative_absence,
            )
            outcome = execute_review_thread_action_plan(
                plan,
                self.git_provider,
                existing_summary_bodies=existing_summary_bodies,
            )
        except Exception as error:
            self._set_review_thread_lifecycle_unavailable(
                "reconciliation_failed",
                "the reconciliation plan could not be completed",
                failure=type(error).__name__,
            )
            return data

        action_pairs = tuple(zip(plan.actions, outcome.action_outcomes, strict=True))
        hard_failure_states = {
            ReviewThreadActionState.STALE_HEAD,
            ReviewThreadActionState.STALE_INVENTORY,
            ReviewThreadActionState.FAILED,
            ReviewThreadActionState.NOT_EXECUTED,
            ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
        }
        hard_failures = [
            result for _, result in action_pairs if result.state in hard_failure_states
        ]
        fallback_finding_ids = {entry.finding_id for entry in outcome.summary_fallbacks}
        existing_fallback_finding_ids = set()
        for body in existing_summary_bodies:
            markers = summary_fallback_markers(body)
            if markers and all(
                version == SUMMARY_FALLBACK_MARKER_VERSION for version, _ in markers
            ):
                existing_fallback_finding_ids.update(finding_id for _, finding_id in markers)
        fallback_eligible_finding_ids = {
            action.finding_id
            for action, result in action_pairs
            if result.state == ReviewThreadActionState.FALLBACK_REQUIRED
            or (
                result.state == ReviewThreadActionState.FAILED
                and result.failure_kind != ReviewThreadFailureKind.RATE_LIMITED
                and not result.requires_fresh_inventory
            )
        }
        reused_fallback_finding_ids = (
            existing_fallback_finding_ids & fallback_eligible_finding_ids
        )
        threaded_finding_ids = {
            action.finding_id
            for action, result in action_pairs
            if action.kind in {
                ReviewThreadActionKind.CREATE,
                ReviewThreadActionKind.UPDATE,
                ReviewThreadActionKind.UNCHANGED,
            }
            and result.succeeded
        }
        self._review_thread_lifecycle_threaded_findings = bool(threaded_finding_ids)
        finding_ids = [desired.identity.finding_id for desired in desired_threads]
        handled_finding_ids = (
            threaded_finding_ids | fallback_finding_ids | reused_fallback_finding_ids
        )
        remaining_issues = [
            issue for issue, finding_id in zip(issues, finding_ids, strict=True)
            if finding_id not in handled_finding_ids
        ]
        updated = copy.deepcopy(data)
        if remaining_issues:
            updated["review"]["key_issues_to_review"] = remaining_issues
        else:
            updated["review"].pop("key_issues_to_review", None)

        self._review_thread_summary_fallbacks = outcome.summary_fallbacks
        self._review_thread_lifecycle_blocks_summary = outcome.requires_fresh_inventory
        protected_current_findings = {
            action.finding_id
            for action, result in action_pairs
            if action.finding_id in finding_ids and result.state == ReviewThreadActionState.SKIPPED
        }
        if hard_failures:
            status = "partial"
            self._review_thread_lifecycle_notice = (
                "> ⚠️ **PR-Agent inline lifecycle:** reconciliation was incomplete; "
                "affected findings remain visible in a review summary."
            )
        elif outcome.summary_fallbacks or reused_fallback_finding_ids:
            status = "complete_with_fallback"
        elif protected_current_findings:
            status = "protected_discussion"
        else:
            status = "complete"
        failure_kinds = {}
        for result in hard_failures:
            key = result.failure_kind.value if result.failure_kind else result.state.value
            failure_kinds[key] = failure_kinds.get(key, 0) + 1
        self.review_thread_reconciliation_artifact = {
            "enabled": True,
            "status": status,
            "head_match": outcome.current_head_sha == expected_head_sha,
            "authoritative_absence": authoritative_absence,
            "mutations_attempted": sum(
                1 for _, result in action_pairs if result.mutation_attempted
            ),
            "requires_fresh_inventory": outcome.requires_fresh_inventory,
            "summary_fallback_count": len(outcome.summary_fallbacks),
            "reused_summary_fallback_count": len(reused_fallback_finding_ids),
            "failure_kinds": failure_kinds,
            "results": {
                "created": sum(
                    1 for action, result in action_pairs
                    if action.kind == ReviewThreadActionKind.CREATE and result.succeeded
                ),
                "updated": sum(
                    1 for action, result in action_pairs
                    if action.kind == ReviewThreadActionKind.UPDATE and result.succeeded
                ),
                "unchanged": sum(
                    1 for action, result in action_pairs
                    if action.kind == ReviewThreadActionKind.UNCHANGED and result.succeeded
                ),
                "resolved": sum(
                    1 for action, result in action_pairs
                    if action.kind == ReviewThreadActionKind.RESOLVE and result.succeeded
                ),
                "failed": len(hard_failures),
            },
            "metrics": {
                **outcome.metrics,
                "unavailable": {},
            },
        }
        get_logger().info(
            "Review-thread lifecycle finished",
            artifact=copy.deepcopy(self.review_thread_reconciliation_artifact),
        )
        return updated

    def _append_review_thread_lifecycle_summary(self, markdown_text: str) -> str:
        markdown_text = markdown_text or ""
        sections = []
        fallbacks: tuple[SummaryFallbackEntry, ...] = getattr(
            self, "_review_thread_summary_fallbacks", ()
        )
        if fallbacks:
            sections.append(
                "### Inline review fallbacks\n\n"
                + "\n\n".join(entry.rendered_body for entry in fallbacks)
            )
        notice = getattr(self, "_review_thread_lifecycle_notice", None)
        if notice:
            sections.append(notice)
        if not sections:
            return markdown_text
        suffix = "\n\n".join(sections)
        return f"{markdown_text.rstrip()}\n\n{suffix}" if markdown_text else suffix

    @staticmethod
    def _limit_findings(data: dict, limit: int | None) -> dict:
        if limit is None:
            return data
        issues = (data.get("review") or {}).get("key_issues_to_review")
        if not isinstance(issues, list) or len(issues) <= limit:
            return data
        limited = copy.deepcopy(data)
        limited["review"]["key_issues_to_review"] = issues[:limit]
        return limited

    def _build_key_issue_comment(
        self,
        issue,
        diff_files: dict,
        *,
        allow_old_side: bool = False,
    ) -> Optional[dict]:
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
        relevant_side = "old" if issue.get("side") == "old" else "new"
        if relevant_side == "old" and not allow_old_side:
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
        target_file = file.base_file if relevant_side == "old" else file.head_file
        if not target_file or end_line > len(split_git_file_lines(target_file)):
            get_logger().warning("Review finding points past the end of the file, keeping it in the summary",
                                 artifact={"relevant_file": relevant_file, "start_line": start_line,
                                           "end_line": end_line})
            return None

        relevant_file = file.filename.strip()
        body = f"**{issue_header}**\n\n{issue_content}" if issue_header else issue_content
        comment = {"body": body,
                   "relevant_file": relevant_file,
                   "relevant_lines_start": start_line,
                   "relevant_lines_end": end_line,
                   "fallback_to_pr_comment": False}
        if allow_old_side:
            comment["relevant_side"] = relevant_side
        return comment

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

        if ((get_settings().pr_reviewer.enable_review_labels_security or
                get_settings().pr_reviewer.enable_review_labels_effort) and
                self.git_provider.is_supported("get_labels")):
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
