import copy
import difflib
import hashlib
import itertools
import json
import os
import re
import stat
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse
from weakref import WeakValueDictionary

try:
    import fcntl
except ImportError:  # pragma: no cover - the GitHub service runs on Unix.
    fcntl = None

from github import AppAuthentication, Auth, Github, GithubException
from github.Issue import Issue
from retry.api import retry_call
from starlette_context import context

from ..algo.file_filter import filter_ignored
from ..algo.git_patch_processing import (
    RE_HUNK_HEADER,
    extract_hunk_headers,
    iter_git_patch_lines,
    strip_git_line_ending,
)
from ..algo.inline_comment_dedup import (
    FINDING_IDENTITY_MARKER_VERSION,
    SUMMARY_FALLBACK_MARKER_VERSION,
    body_fingerprint,
    body_with_markers,
    code_fingerprint,
    finding_identity_markers,
    get_inline_comment_store,
    has_marker,
    is_agent_inline_comment,
    summary_fallback_markers,
)
from ..algo.language_handler import is_valid_file
from ..algo.review_thread_reconciler import (
    ReviewThreadActionKind,
    ReviewThreadActionOutcome,
    ReviewThreadActionState,
    ReviewThreadAnchor,
    ReviewThreadCommentSnapshot,
    ReviewThreadFailureKind,
    ReviewThreadSnapshot,
)
from ..algo.types import EDIT_TYPE
from ..algo.utils import (
    Range,
    clip_tokens,
    comment_matches_pr_review_identity,
    find_line_number_of_relevant_line_in_file,
    get_pr_review_comment_identifiers,
    load_large_diff,
    set_file_languages,
)
from ..config_loader import get_settings
from ..log import get_logger
from ..servers.utils import RateLimitExceeded
from .git_provider import (
    MAX_FILES_ALLOWED_FULL,
    FilePatchInfo,
    GitProvider,
    IncrementalPR,
    get_cached_global_settings,
    is_own_persistent_comment_for_identities,
    redact_credentials,
)

_REVIEW_THREAD_MUTATION_LOCKS_GUARD = threading.Lock()
_REVIEW_THREAD_MUTATION_LOCKS = WeakValueDictionary()
_REVIEW_THREAD_MUTATION_PROCESS_LOCK_PREFIX = os.path.join(tempfile.gettempdir(), "pr-agent-review-thread")


def _github_patch_is_complete(patch: object, additions: object, deletions: object) -> bool:
    """Prove that one original GitHub patch contains every declared changed record."""
    if (
        not isinstance(patch, str)
        or not patch.strip()
        or isinstance(additions, bool)
        or not isinstance(additions, int)
        or additions < 0
        or isinstance(deletions, bool)
        or not isinstance(deletions, int)
        or deletions < 0
    ):
        return False

    saw_hunk = False
    expected_old = expected_new = observed_old = observed_new = 0
    observed_additions = observed_deletions = 0
    for record in iter_git_patch_lines(patch):
        line = strip_git_line_ending(record)
        match = RE_HUNK_HEADER.match(line)
        if match:
            if saw_hunk and (observed_old != expected_old or observed_new != expected_new):
                return False
            _, expected_old, expected_new, _, _ = extract_hunk_headers(match)
            observed_old = observed_new = 0
            saw_hunk = True
            continue
        if not saw_hunk:
            continue
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            observed_new += 1
            observed_additions += 1
        elif line.startswith("-"):
            observed_old += 1
            observed_deletions += 1
        elif line.startswith(" "):
            observed_old += 1
            observed_new += 1
        else:
            return False
        if observed_old > expected_old or observed_new > expected_new:
            return False

    return (
        saw_hunk
        and observed_old == expected_old
        and observed_new == expected_new
        and observed_additions == additions
        and observed_deletions == deletions
    )


class _ReviewThreadMutationLockError(RuntimeError):
    pass


@contextmanager
def _review_thread_mutation_lock(repository: str, pull_request_number: int, finding_id: str):
    """Serialize same-finding mutations across threads and service workers."""
    key = f"{repository.casefold()}#{pull_request_number}:{finding_id}"
    lock_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    lock_path = f"{_REVIEW_THREAD_MUTATION_PROCESS_LOCK_PREFIX}-{lock_digest}.lock"
    with _REVIEW_THREAD_MUTATION_LOCKS_GUARD:
        lock = _REVIEW_THREAD_MUTATION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _REVIEW_THREAD_MUTATION_LOCKS[key] = lock
    with lock:
        if fcntl is None:
            raise _ReviewThreadMutationLockError("cross-process file locking is unavailable")
        descriptor = None
        try:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(lock_path, flags, 0o600)
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise OSError("create coordination path is not a regular file")
            if hasattr(os, "geteuid") and lock_stat.st_uid != os.geteuid():
                raise OSError("create coordination file has an unexpected owner")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except (OSError, ValueError) as error:
            if descriptor is not None:
                os.close(descriptor)
            raise _ReviewThreadMutationLockError(f"cross-process mutation coordination failed: {error}") from error
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _next_page_url(headers: dict) -> str:
    link = headers.get("Link", "")
    if not link:
        return ""
    for part in link.split(","):
        match = re.search(r'<([^>]+)>\s*;\s*rel="next"', part.strip())
        if match:
            return match.group(1)
    return ""


class _ReviewThreadGraphQLError(RuntimeError):
    def __init__(self, message: str, *, status=None, headers=None, data=None):
        super().__init__(message)
        self.status = status
        self.headers = headers or {}
        self.data = data


def _github_actor_is_bot_identity(actor: object) -> bool:
    """Recognize GitHub App bot actors without trusting a login suffix alone."""
    return bool(
        isinstance(actor, dict)
        and actor.get("id")
        and str(actor.get("login") or "").casefold().endswith("[bot]")
        and actor.get("__typename") in {"Bot", "User"}
    )


def _review_thread_failure_details(error: Exception) -> dict:
    """Classify GitHub failures and retain actionable retry evidence."""
    status = getattr(error, "status", None)
    raw_headers = getattr(error, "headers", None) or {}
    try:
        headers = {str(key).casefold(): str(value) for key, value in raw_headers.items()}
    except (AttributeError, TypeError, ValueError):
        headers = {}
    details = f"{error} {getattr(error, 'data', '')}".casefold()
    remaining = headers.get("x-ratelimit-remaining")
    rate_limited = bool(
        isinstance(error, RateLimitExceeded)
        or status == 429
        or remaining == "0"
        or any(
            token in details
            for token in ("rate limit", "secondary rate", "abuse detection", "too many requests")
        )
    )
    if rate_limited:
        retry_after_seconds = None
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                retry_after_seconds = max(0.0, float(retry_after))
            except (TypeError, ValueError, OverflowError):
                # Ignore malformed optional retry metadata from the provider.
                pass
        rate_limit_reset_at = None
        reset_at = headers.get("x-ratelimit-reset")
        if reset_at is not None:
            try:
                rate_limit_reset_at = int(reset_at)
            except (TypeError, ValueError, OverflowError):
                # Ignore malformed optional reset metadata from the provider.
                pass
        retry_source = (
            "retry-after"
            if retry_after_seconds is not None
            else "x-ratelimit-reset"
            if rate_limit_reset_at is not None
            else "provider-signal"
        )
        return {
            "failure_kind": ReviewThreadFailureKind.RATE_LIMITED,
            "retry_after_seconds": retry_after_seconds,
            "rate_limit_reset_at": rate_limit_reset_at,
            "retry_source": retry_source,
        }
    if status == 422:
        return {"failure_kind": ReviewThreadFailureKind.INVALID_INLINE_LOCATION}
    if status in {401, 403}:
        return {"failure_kind": ReviewThreadFailureKind.PERMISSION_DENIED}
    if any(token in details for token in ("permission denied", "forbidden", "resource not accessible")):
        return {"failure_kind": ReviewThreadFailureKind.PERMISSION_DENIED}
    return {"failure_kind": ReviewThreadFailureKind.PROVIDER_FAILURE}


def _bounded_ci_text(value, limit: int = 1000) -> str:
    return " ".join(str(value or "").split())[:limit]


MAX_CI_FAILURES = 20
MAX_CI_CHECK_RUNS = 100


class GithubProvider(GitProvider):
    # None until a filtering pass records the result, so callers that need a
    # complete inventory of changed files fail closed rather than assume none.
    excluded_diff_file_paths: Optional[tuple[str, ...]] = None

    def __init__(self, pr_url: Optional[str] = None):
        self.repo_obj = None
        try:
            self.installation_id = context.get("installation_id", None)
        except Exception:
            self.installation_id = None
        self.max_comment_chars = 65000
        self.base_url = get_settings().get("GITHUB.BASE_URL", "https://api.github.com").rstrip("/") # "https://api.github.com"
        self.base_url_html = self.base_url.split("api/")[0].rstrip("/") if "api/" in self.base_url else "https://github.com"
        self.github_client = self._get_github_client()
        self.repo = None
        self.pr_num = None
        self.pr = None
        self.issue_main = None
        self.github_user_id = None
        self.diff_files = None
        self.excluded_diff_file_paths = None
        self.git_files = None
        self.incremental = IncrementalPR(False)
        self._routing_incremental_files = None
        self._check_run_ids: dict = {}
        if pr_url and 'pull' in pr_url:
            self.set_pr(pr_url)
            self.pr_commits = list(self.pr.get_commits())
            if self.pr_commits:
                self.last_commit_id = self.pr_commits[-1]
            else:
                self.last_commit_id = self._get_repo().get_commit(self.pr.head.sha)
            self.pr_url = self.get_pr_url() # pr_url for github actions can be as api.github.com, so we need to get the url from the pr object
        elif pr_url and 'issue' in pr_url: #url is an issue
            self.issue_main = self._get_issue_handle(pr_url)
        else: #Instantiated the provider without a PR / Issue
            self.pr_commits = None

    def _get_issue_handle(self, issue_url) -> Optional[Issue]:
        repo_name, issue_number = self._parse_issue_url(issue_url)
        if not repo_name or not issue_number:
            get_logger().error(f"Given url: {issue_url} is not a valid issue.")
            return None
        # else: Check if can get a valid Repo handle:
        try:
            repo_obj = self.github_client.get_repo(repo_name)
            if not repo_obj:
                get_logger().error(f"Given url: {issue_url}, belonging to owner/repo: {repo_name} does "
                                   f"not have a valid repository: {self.get_git_repo_url(issue_url)}")
                return None
            # else: Valid repo handle:
            return repo_obj.get_issue(issue_number)
        except Exception as e:
            get_logger().exception(f"Failed to get an issue object for issue: {issue_url}, belonging to owner/repo: {repo_name}")
            return None

    def get_incremental_commits(self, incremental=IncrementalPR(False)):
        self.incremental = incremental
        self._routing_incremental_files = None
        if self.incremental.is_incremental:
            self.unreviewed_files_map = dict()
            self._get_incremental_commits()

    def is_supported(self, capability: str) -> bool:
        if capability == "push_code" and get_settings().config.restricted_mode:
            return False
        return True

    def supports_line_question_history(self) -> bool:
        return True

    def _get_owner_and_repo_path(self, given_url: str) -> str:
        try:
            repo_path = None
            if 'issues' in given_url:
                repo_path, _ = self._parse_issue_url(given_url)
            elif 'pull' in given_url:
                repo_path, _ = self._parse_pr_url(given_url)
            elif given_url.endswith('.git'):
                parsed_url = urlparse(given_url)
                repo_path = (parsed_url.path.split('.git')[0])[1:] # /<owner>/<repo>.git -> <owner>/<repo>
            if not repo_path:
                get_logger().error(f"url is neither an issues url nor a PR url nor a valid git url: {given_url}. Returning empty result.")
                return ""
            return repo_path
        except Exception as e:
            get_logger().exception(f"unable to parse url: {given_url}. Returning empty result.")
            return ""

    def get_git_repo_url(self, issues_or_pr_url: str) -> str:
        repo_path = self._get_owner_and_repo_path(issues_or_pr_url) #Return: <OWNER>/<REPO>
        if not repo_path or repo_path not in issues_or_pr_url:
            get_logger().error(f"Unable to retrieve owner/path from url: {issues_or_pr_url}")
            return ""
        return f"{self.base_url_html}/{repo_path}.git" #https://github.com / <OWNER>/<REPO>.git

    # Given a git repo url, return prefix and suffix of the provider in order to view a given file belonging to that repo.
    # Example: https://github.com/the-pr-agent/pr-agent.git and branch: v0.8 -> prefix: "https://github.com/the-pr-agent/pr-agent/blob/v0.8", suffix: ""
    # In case git url is not provided, provider will use PR context (which includes branch) to determine the prefix and suffix.
    def get_canonical_url_parts(self, repo_git_url:str, desired_branch:str) -> Tuple[str, str]:
        owner = None
        repo = None
        scheme_and_netloc = None

        if repo_git_url or self.issue_main: #Either user provided an external git url, which may be different than what this provider was initialized with, or an issue:
            desired_branch = desired_branch if repo_git_url else self.issue_main.repository.default_branch
            html_url = repo_git_url if repo_git_url else self.issue_main.html_url
            parsed_git_url = urlparse(html_url)
            scheme_and_netloc = parsed_git_url.scheme + "://" + parsed_git_url.netloc
            repo_path = self._get_owner_and_repo_path(html_url)
            if repo_path.count('/') == 1: #Has to have the form <owner>/<repo>
                owner, repo = repo_path.split('/')
            else:
                get_logger().error(f"Invalid repo_path: {repo_path} from url: {html_url}")
                return ("", "")

        if (not owner or not repo) and self.repo: #"else" - User did not provide an external git url, or not an issue, use self.repo object
            owner, repo = self.repo.split('/')
            scheme_and_netloc = self.base_url_html
            desired_branch = self.repo_obj.default_branch
        if not all([scheme_and_netloc, owner, repo]): #"else": Not invoked from a PR context,but no provided git url for context
            get_logger().error("Unable to get canonical url parts since missing context (PR or explicit git url)")
            return ("", "")

        prefix = f"{scheme_and_netloc}/{owner}/{repo}/blob/{desired_branch}"
        suffix = ""  # github does not add a suffix
        return (prefix, suffix)

    def get_pr_url(self) -> str:
        return self.pr.html_url

    def set_pr(self, pr_url: str):
        self.repo, self.pr_num = self._parse_pr_url(pr_url)
        self.pr = self._get_pr()

    def _get_incremental_commits(self):
        if not self.pr_commits:
            self.pr_commits = list(self.pr.get_commits())

        self.previous_review = self.get_previous_review(
            full=True,
            incremental=True,
            review_profile=self.incremental.review_profile,
        )
        if self.previous_review:
            self.incremental.commits_range = self.get_commit_range()
            historical_files = {}
            for commit in self.incremental.commits_range:
                if commit.commit.message.startswith(f"Merge branch '{self._get_repo().default_branch}'"):
                    get_logger().info(f"Skipping merge commit {commit.commit.message}")
                    continue
                historical_files.update({file.filename: file for file in commit.files})

            net_files, incomplete = self._get_incremental_net_files()
            self._routing_incremental_files = tuple(net_files)
            if incomplete:
                self._routing_incremental_files += (FilePatchInfo(
                    base_file="",
                    head_file="",
                    patch="",
                    filename="",
                    edit_type=EDIT_TYPE.UNKNOWN,
                ),)
                # A partial/unavailable compare must not drop review input. Retain
                # the historical union as a conservative fallback while routing
                # consumes the known net evidence plus an UNKNOWN sentinel.
                self.unreviewed_files_map.update(historical_files)
                self.unreviewed_files_map.update({file.filename: file for file in net_files})
            else:
                self.unreviewed_files_map.update({file.filename: file for file in net_files})
        else:
            get_logger().info("No previous review found, will review the entire PR")
            self.incremental.is_incremental = False

    @staticmethod
    def _github_file_paths(file) -> set[str]:
        """Return the trustworthy path endpoints exposed by one GitHub file record."""

        filename = getattr(file, "filename", None)
        if not isinstance(filename, str) or not filename.strip():
            raise TypeError("GitHub changed file is missing filename")
        paths = {filename.strip()}
        previous_filename = getattr(file, "previous_filename", None)
        if previous_filename is not None:
            if not isinstance(previous_filename, str) or not previous_filename.strip():
                raise TypeError("GitHub changed file has an invalid previous_filename")
            paths.add(previous_filename.strip())
        return paths

    def _get_incremental_net_files(self) -> tuple[list, bool]:
        """Return the unfiltered, PR-scoped baseline-to-head inventory."""

        base_sha = self.incremental.last_seen_commit_sha
        head_sha = getattr(getattr(self.pr, "head", None), "sha", None)
        if not base_sha or not head_sha:
            get_logger().warning(
                "Cannot fetch GitHub incremental net diff without baseline and head commits."
            )
            return [], True

        files = []
        compare_incomplete = False
        try:
            comparison = self._get_repo().compare(base_sha, head_sha)
            raw_files = getattr(comparison, "files", None)
            if raw_files is None or isinstance(raw_files, (str, bytes, dict)):
                raise TypeError("GitHub incremental compare files must be iterable")
            for file in raw_files:
                self._github_file_paths(file)
                files.append(file)
        except Exception as error:
            get_logger().warning(
                "Failed to fetch the complete GitHub incremental net diff; preserving "
                f"known paths and marking routing evidence incomplete: {error}"
            )
            compare_incomplete = True

        # GitHub's compare endpoint exposes at most 300 changed files. Reaching
        # the documented cap cannot prove completeness, even when iteration ends.
        compare_incomplete = compare_incomplete or len(files) >= 300

        # A baseline-to-head comparison can include changes brought in by merging
        # the target branch. Scope it to the pull request's own current inventory,
        # retaining both sides of renames. When that inventory is incomplete, keep
        # all known compare evidence and add the caller's UNKNOWN safety sentinel.
        current_pr_files = []
        current_pr_incomplete = False
        try:
            raw_current_pr_files = self.pr.get_files()
            if raw_current_pr_files is None or isinstance(
                raw_current_pr_files, (str, bytes, dict)
            ):
                raise TypeError("GitHub pull-request files must be iterable")
            for file in raw_current_pr_files:
                self._github_file_paths(file)
                current_pr_files.append(file)
        except Exception as error:
            get_logger().warning(
                "Failed to fetch the complete GitHub pull-request file inventory; "
                f"preserving compare evidence and marking it incomplete: {error}"
            )
            current_pr_incomplete = True

        try:
            changed_files = getattr(self.pr, "changed_files", None)
        except Exception as error:
            get_logger().warning(
                "Failed to read the GitHub pull-request changed-file count; "
                f"marking routing evidence incomplete: {error}"
            )
            changed_files = None
        if (
            not isinstance(changed_files, int)
            or isinstance(changed_files, bool)
            or changed_files < 0
            or changed_files != len(current_pr_files)
        ):
            current_pr_incomplete = True
        current_pr_incomplete = current_pr_incomplete or len(current_pr_files) >= 3000
        if current_pr_incomplete:
            return files, True

        current_pr_paths = set()
        for file in current_pr_files:
            current_pr_paths.update(self._github_file_paths(file))
        files = [
            file for file in files
            if self._github_file_paths(file) & current_pr_paths
        ]

        return files, compare_incomplete

    def get_files_for_routing(self):
        """Return the unfiltered net incremental inventory instead of commit history."""

        if self.incremental.is_incremental:
            routing_files = getattr(self, "_routing_incremental_files", None)
            if routing_files is None:
                return [FilePatchInfo(
                    base_file="",
                    head_file="",
                    patch="",
                    filename="",
                    edit_type=EDIT_TYPE.UNKNOWN,
                )]
            return list(routing_files)
        return self.get_files()

    def is_incremental_scope_empty(self) -> Optional[bool]:
        empty = super().is_incremental_scope_empty()
        if empty is not True:
            return empty
        routing_files = getattr(self, "_routing_incremental_files", None)
        if routing_files is None or any(
            getattr(file, "edit_type", None) is EDIT_TYPE.UNKNOWN
            for file in routing_files
        ):
            return None
        return not routing_files

    def get_commit_range(self):
        last_review_time = self.previous_review.created_at
        first_new_commit_index = None
        for index in range(len(self.pr_commits) - 1, -1, -1):
            if self.pr_commits[index].commit.author.date > last_review_time:
                self.incremental.first_new_commit = self.pr_commits[index]
                first_new_commit_index = index
            else:
                self.incremental.last_seen_commit = self.pr_commits[index]
                break
        return self.pr_commits[first_new_commit_index:] if first_new_commit_index is not None else []

    def get_previous_review(self, *, full: bool, incremental: bool, review_profile: str = "full"):
        if not (full or incremental):
            raise ValueError("At least one of full or incremental must be True")
        if not getattr(self, "comments", None):
            self.comments = list(self.pr.get_issue_comments())
        identifiers = get_pr_review_comment_identifiers(
            full=full,
            incremental=incremental,
            review_profile=review_profile,
        )
        for index in range(len(self.comments) - 1, -1, -1):
            body = self.comments[index].body
            if not comment_matches_pr_review_identity(body, identifiers, review_profile):
                continue
            if is_own_persistent_comment_for_identities(body, identifiers):
                return self.comments[index]
        return None

    def get_files(self):
        if self.incremental.is_incremental and self.unreviewed_files_map:
            return self.unreviewed_files_map.values()
        try:
            git_files = context.get("git_files", None)
            if git_files:
                return git_files
            self.git_files = list(self.pr.get_files()) # 'list' to handle pagination
            context["git_files"] = self.git_files
            return self.git_files
        except Exception:
            if not self.git_files:
                self.git_files = list(self.pr.get_files())
            return self.git_files

    def get_num_of_files(self):
        if hasattr(self.git_files, "totalCount"):
            return self.git_files.totalCount
        else:
            try:
                return len(self.git_files)
            except Exception as e:
                return -1

    def get_diff_files(self) -> list[FilePatchInfo]:
        """
        Retrieves the list of files that have been modified, added, deleted, or renamed in a pull request in GitHub,
        along with their content and patch information.

        Returns:
            diff_files (List[FilePatchInfo]): List of FilePatchInfo objects representing the modified, added, deleted,
            or renamed files in the merge request.
        """
        # the retry settings are read at call time rather than in a decorator, so that importing this module
        # does not require a [github] settings section (issue #2427)
        return retry_call(self._get_diff_files, exceptions=RateLimitExceeded,
                          tries=get_settings().get("GITHUB.RATELIMIT_RETRIES", 5), delay=2, backoff=2, jitter=(1, 3))

    def _get_diff_files(self) -> list[FilePatchInfo]:
        try:
            try:
                diff_files = context.get("diff_files", None)
                if diff_files:
                    return diff_files
            except Exception:
                pass

            if self.diff_files:
                return self.diff_files

            # filter files using [ignore] patterns
            files_original = self.get_files()
            files = filter_ignored(files_original)
            # Ignored paths never reach the model, so record them for callers that
            # must know whether the reviewed inventory was complete.
            kept_names = {file.filename for file in files}
            self.excluded_diff_file_paths = tuple(sorted(
                file.filename for file in files_original if file.filename not in kept_names
            ))
            if files_original != files:
                try:
                    names_original = [file.filename for file in files_original]
                    names_new = [file.filename for file in files]
                    get_logger().info("Filtered out [ignore] files for pull request:", extra=
                    {"files": names_original,
                     "filtered_files": names_new})
                except Exception:
                    pass

            diff_files = []
            invalid_files_names = []
            is_close_to_rate_limit = False

            # The base.sha will point to the current state of the base branch (including parallel merges), not the original base commit when the PR was created
            # We can fix this by finding the merge base commit between the PR head and base branches
            # Note that The pr.head.sha is actually correct as is - it points to the latest commit in your PR branch.
            # This SHA isn't affected by parallel merges to the base branch since it's specific to your PR's branch.
            repo = self.repo_obj
            pr = self.pr
            try:
                compare = repo.compare(pr.base.sha, pr.head.sha) # communication with GitHub
                merge_base_commit = compare.merge_base_commit
            except Exception as e:
                get_logger().error(f"Failed to get merge base commit: {e}")
                merge_base_commit = pr.base
            if merge_base_commit.sha != pr.base.sha:
                get_logger().info(
                    f"Using merge base commit {merge_base_commit.sha} instead of base commit ")

            counter_valid = 0
            for file in files:
                if not is_valid_file(file.filename):
                    invalid_files_names.append(file.filename)
                    continue

                github_patch = file.patch
                patch = github_patch
                is_renamed = file.status == "renamed" and getattr(file, "previous_filename", None)
                old_filename = file.previous_filename if is_renamed else None
                if is_close_to_rate_limit:
                    new_file_content_str = ""
                    original_file_content_str = ""
                else:
                    # allow only a limited number of files to be fully loaded. We can manage the rest with diffs only
                    counter_valid += 1
                    avoid_load = False
                    if counter_valid >= MAX_FILES_ALLOWED_FULL and patch and not self.incremental.is_incremental:
                        avoid_load = True
                        if counter_valid == MAX_FILES_ALLOWED_FULL:
                            get_logger().info("Too many files in PR, will avoid loading full content for rest of files")

                    if avoid_load:
                        new_file_content_str = ""
                    else:
                        new_file_content_str = self._get_pr_file_content(file, self.pr.head.sha)  # communication with GitHub

                    if self.incremental.is_incremental and self.unreviewed_files_map:
                        original_file_content_str = self._get_pr_file_content(
                            file, self.incremental.last_seen_commit_sha, path=old_filename)
                        patch = load_large_diff(file.filename, new_file_content_str, original_file_content_str)
                        self.unreviewed_files_map[file.filename] = patch
                    else:
                        if avoid_load:
                            original_file_content_str = ""
                        else:
                            original_file_content_str = self._get_pr_file_content(file, merge_base_commit.sha, path=old_filename)
                            # original_file_content_str = self._get_pr_file_content(file, self.pr.base.sha)
                        if not patch:
                            patch = load_large_diff(file.filename, new_file_content_str, original_file_content_str)


                if file.status == 'added':
                    edit_type = EDIT_TYPE.ADDED
                elif file.status == 'removed':
                    edit_type = EDIT_TYPE.DELETED
                elif file.status == 'renamed':
                    edit_type = EDIT_TYPE.RENAMED
                elif file.status == 'modified':
                    edit_type = EDIT_TYPE.MODIFIED
                else:
                    get_logger().error(f"Unknown edit type: {file.status}")
                    edit_type = EDIT_TYPE.UNKNOWN

                # count number of lines added and removed
                github_additions = getattr(file, 'additions', None)
                github_deletions = getattr(file, 'deletions', None)
                if hasattr(file, 'additions') and hasattr(file, 'deletions'):
                    num_plus_lines = github_additions
                    num_minus_lines = github_deletions
                else:
                    patch_lines = list(iter_git_patch_lines(patch))
                    num_plus_lines = len([line for line in patch_lines if line.startswith('+')])
                    num_minus_lines = len([line for line in patch_lines if line.startswith('-')])
                patch_is_complete = (
                    not self.incremental.is_incremental
                    and _github_patch_is_complete(github_patch, github_additions, github_deletions)
                )

                file_patch_canonical_structure = FilePatchInfo(original_file_content_str, new_file_content_str, patch,
                                                               file.filename, edit_type=edit_type,
                                                               old_filename=old_filename,
                                                               num_plus_lines=num_plus_lines,
                                                               num_minus_lines=num_minus_lines,
                                                               patch_is_complete=patch_is_complete,)
                diff_files.append(file_patch_canonical_structure)
            if invalid_files_names:
                get_logger().info(f"Filtered out files with invalid extensions: {invalid_files_names}")

            self.diff_files = diff_files
            try:
                context["diff_files"] = diff_files
            except Exception:
                pass

            return diff_files

        except Exception as e:
            get_logger().error(f"Failing to get diff files: {e}",
                               artifact={"traceback": traceback.format_exc()})
            raise RateLimitExceeded("Rate limit exceeded for GitHub API.") from e

    def publish_description(self, pr_title: str, pr_body: str):
        if pr_title is None:
            self.pr.edit(body=pr_body)
        else:
            self.pr.edit(title=pr_title, body=pr_body)

    def get_latest_commit_url(self) -> str:
        return self.last_commit_id.html_url

    def get_pr_head_sha(self, refresh: bool = False) -> Optional[str]:
        if refresh:
            return self._get_repo().get_pull(self.pr_num).head.sha
        return self.last_commit_id.sha if getattr(self, "last_commit_id", None) else None

    def get_comment_url(self, comment) -> str:
        return comment.html_url

    def publish_persistent_comment(self, pr_comment: str,
                                   initial_header: str,
                                   update_header: bool = True,
                                   name='review',
                                   final_update_message=True,
                                   identity_marker: str | None = None,
                                   legacy_initial_header: str | None = None):
        if get_settings().github.publish_as_check_run:
            if self._publish_check_run(pr_comment, name):
                return
        self.publish_persistent_comment_full(
            pr_comment,
            initial_header,
            update_header,
            name,
            final_update_message,
            identity_marker=identity_marker,
            legacy_initial_header=legacy_initial_header,
        )

    def supports_review_comment_identity(self) -> bool:
        return True

    def supports_review_thread_lifecycle(self) -> bool:
        return True

    def get_ci_failure_context(self) -> dict:
        """Return bounded failed check-run details for the current PR head."""
        if not getattr(self, "last_commit_id", None) or not getattr(self, "pr", None):
            return {"status": "unavailable", "failures": []}
        failure_conclusions = {"action_required", "cancelled", "failure", "startup_failure", "timed_out"}
        failures = []
        examined_runs = 0
        try:
            url = f"{self.base_url}/repos/{self.repo}/commits/{self.last_commit_id.sha}/check-runs"
            while url and len(failures) < MAX_CI_FAILURES and examined_runs < MAX_CI_CHECK_RUNS:
                headers, data = self.pr._requester.requestJsonAndCheck("GET", url)
                for run in data.get("check_runs", []):
                    examined_runs += 1
                    conclusion = str(run.get("conclusion") or "").strip().lower()
                    if conclusion in failure_conclusions:
                        output = run.get("output") or {}
                        failures.append({
                            "name": _bounded_ci_text(run.get("name"), 200),
                            "conclusion": conclusion,
                            "title": _bounded_ci_text(output.get("title")),
                            "summary": _bounded_ci_text(output.get("summary")),
                        })
                    if len(failures) >= MAX_CI_FAILURES or examined_runs >= MAX_CI_CHECK_RUNS:
                        break
                url = _next_page_url(headers)
        except Exception:
            get_logger().warning("Failed to load CI failure context")
            return {"status": "unavailable", "failures": []}
        return {"status": "available", "failures": failures}

    def clear_persistent_review(self, identity_marker: str, name: str = "review") -> bool:
        """Clear the matching comment, or update an existing GitHub check to a clean result."""
        if not get_settings().github.publish_as_check_run:
            return super().clear_persistent_review(identity_marker, name)
        if not getattr(self, "last_commit_id", None):
            return False
        check_run_name = f"PR Agent - {name.capitalize()}"
        existing_id = self._check_run_ids.get(name) or self._find_existing_check_run(
            check_run_name, self.last_commit_id.sha
        )
        if not existing_id:
            return super().clear_persistent_review(identity_marker, name)
        if self._update_check_run(
            existing_id,
            "No qualifying defects found in the latest bugs-only review.",
            name,
        ):
            super().clear_persistent_review(identity_marker, name)
            return True
        return super().clear_persistent_review(identity_marker, name)

    def _check_run_output(self, text: str, name: str) -> tuple[str, dict]:
        check_run_name = f"PR Agent - {name.capitalize()}"
        summary = text.split("\n\n")[0] if "\n\n" in text else text[:200]
        summary = summary.strip(" #")
        text = text[:65535]
        return check_run_name, {
            "title": check_run_name,
            "summary": summary[:300],
            "text": text,
        }

    def _update_check_run(self, check_run_id: int, text: str, name: str) -> bool:
        check_run_name, output = self._check_run_output(text, name)
        try:
            self.pr._requester.requestJsonAndCheck(
                "PATCH",
                f"{self.base_url}/repos/{self.repo}/check-runs/{check_run_id}",
                input={"status": "completed", "conclusion": "neutral", "output": output},
            )
            self._check_run_ids[name] = check_run_id
            return True
        except Exception:
            get_logger().warning(f"Failed to update check run {check_run_id}")
            return False

    def _publish_check_run(self, text: str, name: str) -> bool:
        if not getattr(self, 'last_commit_id', None):
            get_logger().error("Cannot publish check run without a commit SHA")
            return False
        check_run_name, output = self._check_run_output(text, name)
        create_body = {
            "name": check_run_name,
            "head_sha": self.last_commit_id.sha,
            "status": "completed",
            "conclusion": "neutral",
            "output": output,
        }
        existing_id = self._check_run_ids.get(name)
        if not existing_id:
            existing_id = self._find_existing_check_run(check_run_name, self.last_commit_id.sha)
        if existing_id:
            if self._update_check_run(existing_id, text, name):
                return True
            get_logger().warning(f"Creating a new check run after update failed for {existing_id}")
        try:
            headers, data = self.pr._requester.requestJsonAndCheck(
                "POST",
                f"{self.base_url}/repos/{self.repo}/check-runs",
                input=create_body,
            )
            self._check_run_ids[name] = data["id"]
            return True
        except Exception:
            get_logger().warning("Failed to create check run, falling back to comment")
            return False

    def _find_existing_check_run(self, check_run_name: str, head_sha: str) -> Optional[int]:
        pr = getattr(self, 'pr', None)
        if not pr:
            return None
        try:
            url = f"{self.base_url}/repos/{self.repo}/commits/{head_sha}/check-runs"
            while url:
                headers, data = pr._requester.requestJsonAndCheck("GET", url)
                for run in data.get("check_runs", []):
                    if run.get("name") == check_run_name:
                        return run["id"]
                url = _next_page_url(headers)
        except Exception:
            get_logger().warning("Failed to look up existing check runs")
        return None

    def publish_comment(self, pr_comment: str, is_temporary: bool = False):
        if not self.pr and not self.issue_main:
            get_logger().error("Cannot publish a comment if missing PR/Issue context")
            return None

        if is_temporary and not get_settings().config.publish_output_progress:
            get_logger().debug(f"Skipping publish_comment for temporary comment: {pr_comment}")
            return None
        pr_comment = self.limit_output_characters(pr_comment, self.max_comment_chars)

        # In case this is an issue, can publish the comment on the issue.
        if self.issue_main:
            return self.issue_main.create_comment(pr_comment)

        response = self.pr.create_issue_comment(pr_comment)
        if hasattr(response, "user") and hasattr(response.user, "login"):
            self.github_user_id = response.user.login
        response.is_temporary = is_temporary
        if not hasattr(self.pr, 'comments_list'):
            self.pr.comments_list = []
        self.pr.comments_list.append(response)
        return response

    def publish_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str, original_suggestion=None):
        body = self.limit_output_characters(body, self.max_comment_chars)
        self.publish_inline_comments([self.create_inline_comment(body, relevant_file, relevant_line_in_file)])


    def create_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str,
                              absolute_position: int = None):
        body = self.limit_output_characters(body, self.max_comment_chars)
        position, absolute_position = find_line_number_of_relevant_line_in_file(self.diff_files,
                                                                                relevant_file.strip('`'),
                                                                                relevant_line_in_file,
                                                                                absolute_position)
        if position == -1:
            get_logger().info(f"Could not find position for {relevant_file} {relevant_line_in_file}")
            subject_type = "FILE"
        else:
            subject_type = "LINE"
        path = relevant_file.strip()
        return dict(body=body, path=path, position=position) if subject_type == "LINE" else {}

    def publish_inline_comments(self, comments: list[dict], disable_fallback: bool = False):
        store = None
        pending_fingerprints = []
        dedup_code_fp_key = "_dedup_code_fp"
        if get_settings().get("config.persistent_inline_comments", False):
            store = get_inline_comment_store(self)
            local_seen = set()
            deduped = []
            skipped = 0
            for comment in comments:
                if not comment:
                    deduped.append(comment)
                    continue
                path = comment.get("path", "")
                body = comment.get("body", "")
                # GitHub committable comments are anchored by diff position, which
                # shifts as the PR gains commits; anchor the fingerprint on the file
                # path and comment content instead so it stays stable across runs.
                body_fp = body_fingerprint(path, None, body)
                pre_transform_code_fp = comment.get(dedup_code_fp_key)
                code_fp = pre_transform_code_fp or code_fingerprint(path, None, body)
                # A fallback re-publish (disable_fallback=True) is for a comment
                # that has not been posted yet, so do not filter it; only the
                # top-level call drops duplicates. The fallback still gets marked
                # and recorded below so it dedups on later runs.
                if not disable_fallback and (
                        store.seen(body_fp) or store.seen(code_fp)
                        or body_fp in local_seen or (code_fp and code_fp in local_seen)):
                    skipped += 1
                    continue
                marked = dict(comment)
                marked.pop(dedup_code_fp_key, None)
                if has_marker(body):
                    pass  # already carries a marker from the first pass
                else:
                    marked["body"] = body_with_markers(
                        body, body_fp, code_fp, getattr(self, "max_comment_chars", None))
                deduped.append(marked)
                local_seen.add(body_fp)
                if code_fp:
                    local_seen.add(code_fp)
                pending_fingerprints.append((body_fp, code_fp))
            if skipped and not any(deduped):
                get_logger().info(
                    f"Persistent inline comments: all {skipped} suggestion(s) "
                    f"already posted; nothing to publish")
                return
            comments = deduped
        else:
            comments = [
                {key: value for key, value in comment.items() if key != dedup_code_fp_key}
                if comment else comment
                for comment in comments
            ]
        try:
            # publish all comments in a single message
            self.pr.create_review(commit=self.last_commit_id, comments=comments)
            # The whole batch posted; record its fingerprints so the rest of this
            # run dedups against them. Cross-run dedup relies on the markers in the
            # posted bodies, so comments the fallback below drops stay unrecorded
            # and can be retried on a later run.
            if store is not None:
                for body_fp, code_fp in pending_fingerprints:
                    store.add(body_fp)
                    store.add(code_fp)
        except Exception as e:
            get_logger().info("Initially failed to publish inline comments as committable")

            if (getattr(e, "status", None) == 422 and not disable_fallback):
                pass  # continue to try _publish_inline_comments_fallback_with_verification
            else:
                raise e # will end up with publishing the comments one by one

            try:
                self._publish_inline_comments_fallback_with_verification(comments)
            except Exception as e:
                get_logger().error(f"Failed to publish inline code comments fallback, error: {e}")
                raise

    def get_review_thread_comments(self, comment_id: int) -> list[dict]:
        """
        Retrieves all comments in the same thread as the given comment.

        Args:
            comment_id: Review comment ID

        Returns:
            List of comments in the same thread
        """
        try:
            # Fetch all comments with a single API call
            all_comments = list(self.pr.get_comments())

            # Find the target comment by ID
            target_comment = next((c for c in all_comments if c.id == comment_id), None)
            if not target_comment:
                return []

            # Get root comment id
            root_comment_id = target_comment.raw_data.get("in_reply_to_id", target_comment.id)
            # Build the thread - include the root comment and all replies to it
            thread_comments = [
                c for c in all_comments if
                c.id == root_comment_id or c.raw_data.get("in_reply_to_id") == root_comment_id
            ]


            return thread_comments

        except Exception as e:
            get_logger().exception("Failed to get review comments for an inline ask command", artifact={"comment_id": comment_id, "error": e})
            return []

    def supports_thread_resolution(self) -> bool:
        return True

    def _request_review_thread_graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = self.github_client._Github__requester.requestJson(
            "POST", "/graphql", input=payload
        )
        if not isinstance(response, tuple) or len(response) != 3:
            raise RuntimeError("unexpected GitHub GraphQL response format")
        status, headers, raw_body = response
        try:
            body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
        except (TypeError, ValueError) as e:
            raise _ReviewThreadGraphQLError(
                f"GitHub GraphQL response is not valid JSON: {e}",
                status=status,
                headers=headers,
                data=raw_body,
            ) from e
        if not isinstance(body, dict):
            raise _ReviewThreadGraphQLError(
                "GitHub GraphQL response body is not an object",
                status=status,
                headers=headers,
                data=body,
            )
        if isinstance(status, int) and status >= 400:
            raise _ReviewThreadGraphQLError(
                str(body.get("message") or f"GitHub GraphQL HTTP {status}"),
                status=status,
                headers=headers,
                data=body,
            )
        if body.get("errors"):
            raise _ReviewThreadGraphQLError(
                f"GitHub GraphQL errors: {body['errors']}",
                status=status,
                headers=headers,
                data=body["errors"],
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise _ReviewThreadGraphQLError(
                "GitHub GraphQL response has no data object",
                status=status,
                headers=headers,
                data=body,
            )
        return data

    def _get_additional_review_thread_comments(self, thread_id: str, cursor: str) -> tuple[list[dict], str]:
        query = """
        query($threadId: ID!, $after: String) {
          node(id: $threadId) {
            ... on PullRequestReviewThread {
              comments(first: 100, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  id databaseId body createdAt url
                  author { id login __typename }
                  pullRequestReview { commit { oid } }
                }
              }
            }
          }
        }
        """
        data = self._request_review_thread_graphql(query, {"threadId": thread_id, "after": cursor})
        node = data.get("node")
        if not isinstance(node, dict) or not isinstance(node.get("comments"), dict):
            raise RuntimeError(f"GitHub returned incomplete comments for review thread {thread_id}")
        connection = node["comments"]
        comments = connection.get("nodes")
        page_info = connection.get("pageInfo")
        if not isinstance(comments, list) or not isinstance(page_info, dict):
            raise RuntimeError(f"GitHub returned malformed comments for review thread {thread_id}")
        next_cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else ""
        if page_info.get("hasNextPage") and not next_cursor:
            raise RuntimeError(f"GitHub omitted the next comment cursor for review thread {thread_id}")
        return comments, next_cursor or ""

    def get_review_thread_snapshots(
        self,
        *,
        require_viewer_bot: bool = False,
    ) -> tuple[ReviewThreadSnapshot, ...]:
        """Return an authoritative, paginated snapshot of GitHub review threads.

        API and pagination failures raise instead of returning an empty inventory;
        callers must not confuse an unreadable PR with a PR that has no threads.
        """
        owner, repo_name = self.repo.split("/", 1)
        query = """
        query($owner: String!, $name: String!, $number: Int!, $after: String) {
          viewer { id login __typename }
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              reviewThreads(first: 100, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  id isResolved isOutdated path line startLine diffSide startDiffSide
                  originalLine originalStartLine
                  subjectType viewerCanResolve
                  resolvedBy { id login __typename }
                  comments(first: 100) {
                    pageInfo { hasNextPage endCursor }
                    nodes {
                      id databaseId body createdAt url
                      author { id login __typename }
                      pullRequestReview { commit { oid } }
                    }
                  }
                }
              }
            }
          }
        }
        """
        cursor = None
        seen_thread_cursors = set()
        viewer_identity = None
        raw_threads = []
        while True:
            data = self._request_review_thread_graphql(
                query,
                {"owner": owner, "name": repo_name, "number": self.pr_num, "after": cursor},
            )
            viewer = data.get("viewer")
            if (
                not isinstance(viewer, dict)
                or not viewer.get("id")
                or not viewer.get("login")
                or not viewer.get("__typename")
            ):
                raise RuntimeError("GitHub review-thread inventory has no authenticated viewer")
            if viewer_identity is None:
                viewer_identity = viewer
            elif viewer_identity != viewer:
                raise RuntimeError("GitHub review-thread inventory viewer changed during pagination")
            repository = data.get("repository")
            pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
            connection = pull_request.get("reviewThreads") if isinstance(pull_request, dict) else None
            if not isinstance(connection, dict):
                raise RuntimeError("GitHub review-thread inventory has no pull request connection")
            nodes = connection.get("nodes")
            page_info = connection.get("pageInfo")
            if not isinstance(nodes, list) or not isinstance(page_info, dict):
                raise RuntimeError("GitHub review-thread inventory is malformed")
            raw_threads.extend(nodes)
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                raise RuntimeError("GitHub omitted the next review-thread cursor")
            if cursor in seen_thread_cursors:
                raise RuntimeError("GitHub repeated a review-thread cursor")
            seen_thread_cursors.add(cursor)

        if require_viewer_bot and not _github_actor_is_bot_identity(viewer_identity):
            raise RuntimeError("review-thread lifecycle requires an authenticated GitHub Bot identity")

        snapshots = []
        for thread in raw_threads:
            if not isinstance(thread, dict) or not thread.get("id"):
                raise RuntimeError("GitHub returned a review thread without an id")
            comment_connection = thread.get("comments")
            if not isinstance(comment_connection, dict):
                raise RuntimeError(f"GitHub returned review thread {thread['id']} without comments")
            raw_comments = comment_connection.get("nodes")
            comment_page_info = comment_connection.get("pageInfo")
            if not isinstance(raw_comments, list) or not isinstance(comment_page_info, dict):
                raise RuntimeError(f"GitHub returned malformed comments for review thread {thread['id']}")
            raw_comments = list(raw_comments)
            comment_cursor = comment_page_info.get("endCursor") if comment_page_info.get("hasNextPage") else ""
            if comment_page_info.get("hasNextPage") and not comment_cursor:
                raise RuntimeError(f"GitHub omitted the next comment cursor for review thread {thread['id']}")
            seen_comment_cursors = set()
            while comment_cursor:
                if comment_cursor in seen_comment_cursors:
                    raise RuntimeError(f"GitHub repeated a comment cursor for review thread {thread['id']}")
                seen_comment_cursors.add(comment_cursor)
                page, comment_cursor = self._get_additional_review_thread_comments(thread["id"], comment_cursor)
                raw_comments.extend(page)

            comments = []
            for comment in raw_comments:
                if not isinstance(comment, dict) or not comment.get("id"):
                    raise RuntimeError(f"GitHub returned an invalid comment in review thread {thread['id']}")
                author = comment.get("author")
                comments.append(ReviewThreadCommentSnapshot(
                    node_id=comment["id"],
                    database_id=comment.get("databaseId"),
                    author_id=author.get("id") if isinstance(author, dict) else None,
                    author_login=author.get("login") if isinstance(author, dict) else None,
                    author_type=author.get("__typename") if isinstance(author, dict) else None,
                    body=comment.get("body") or "",
                    created_at=comment.get("createdAt"),
                    url=comment.get("url"),
                ))

            root = comments[0] if comments else None
            marker_values = finding_identity_markers(root.body if root else "")
            marker_ids = (
                {finding_id for _, finding_id in marker_values}
                if marker_values and all(
                    marker_version == FINDING_IDENTITY_MARKER_VERSION
                    for marker_version, _ in marker_values
                )
                else set()
            )
            finding_id = next(iter(marker_ids)) if len(marker_ids) == 1 else None
            root_actor = {
                "id": root.author_id if root else None,
                "login": root.author_login if root else None,
                "__typename": root.author_type if root else None,
            }
            root_author_is_viewer_bot = bool(
                root
                and viewer_identity
                and _github_actor_is_bot_identity(viewer_identity)
                and _github_actor_is_bot_identity(root_actor)
                and root.author_id == viewer_identity.get("id")
                and root.author_login == viewer_identity.get("login")
            )
            bot_owned = bool(
                finding_id and root_author_is_viewer_bot and is_agent_inline_comment(root.body)
            )
            resolved_by = thread.get("resolvedBy")
            resolved_by_viewer_bot = bool(
                thread.get("isResolved")
                and isinstance(resolved_by, dict)
                and viewer_identity
                and _github_actor_is_bot_identity(viewer_identity)
                and _github_actor_is_bot_identity(resolved_by)
                and resolved_by.get("id") == viewer_identity.get("id")
                and resolved_by.get("login") == viewer_identity.get("login")
            )
            resolved_by_other_actor = bool(
                thread.get("isResolved")
                and isinstance(resolved_by, dict)
                and resolved_by.get("id")
                and not resolved_by_viewer_bot
            )
            anchor = None
            if not thread.get("isOutdated") and thread.get("subjectType") != "FILE":
                anchor = ReviewThreadAnchor.from_github(
                    thread.get("path"),
                    thread.get("line"),
                    thread.get("startLine"),
                    thread.get("diffSide"),
                    thread.get("startDiffSide"),
                )
            original_anchor = ReviewThreadAnchor.from_github(
                thread.get("path"),
                thread.get("originalLine"),
                thread.get("originalStartLine"),
                thread.get("diffSide"),
                thread.get("startDiffSide"),
            )
            review = raw_comments[0].get("pullRequestReview") if raw_comments else None
            commit = review.get("commit") if isinstance(review, dict) else None
            snapshots.append(ReviewThreadSnapshot(
                thread_id=thread["id"],
                finding_id=finding_id,
                anchor=anchor,
                original_anchor=original_anchor,
                is_resolved=bool(thread.get("isResolved")),
                is_outdated=bool(thread.get("isOutdated")),
                bot_owned=bot_owned,
                has_replies=len(comments) > 1,
                reviewed_head_sha=commit.get("oid") if isinstance(commit, dict) else None,
                comments=tuple(comments),
                subject_type=thread.get("subjectType"),
                viewer_can_resolve=bool(thread.get("viewerCanResolve")),
                resolved_by_viewer_bot=resolved_by_viewer_bot,
                resolved_by_other_actor=resolved_by_other_actor,
            ))
        return tuple(snapshots)

    def get_bot_owned_review_summary_bodies(self) -> tuple[str, ...]:
        """Return fallback-bearing summary bodies owned by the authenticated GitHub Bot.

        Append-only review modes need this inventory to suppress a fallback that
        is already visible from an earlier run. Human-authored comments are never
        trusted even when they copy a well-formed lifecycle marker.
        """
        owner, repo_name = self.repo.split("/", 1)
        query = """
        query($owner: String!, $name: String!, $number: Int!, $after: String) {
          viewer { id login __typename }
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              comments(first: 100, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes { body author { id login __typename } }
              }
            }
          }
        }
        """
        cursor = None
        seen_cursors = set()
        viewer_identity = None
        bodies = []
        while True:
            data = self._request_review_thread_graphql(
                query,
                {"owner": owner, "name": repo_name, "number": self.pr_num, "after": cursor},
            )
            viewer = data.get("viewer")
            if not _github_actor_is_bot_identity(viewer):
                raise RuntimeError("review-summary inventory requires an authenticated GitHub Bot identity")
            if viewer_identity is None:
                viewer_identity = viewer
            elif viewer_identity != viewer:
                raise RuntimeError("GitHub review-summary inventory viewer changed during pagination")
            repository = data.get("repository")
            pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
            connection = pull_request.get("comments") if isinstance(pull_request, dict) else None
            if not isinstance(connection, dict):
                raise RuntimeError("GitHub review-summary inventory has no pull request connection")
            nodes = connection.get("nodes")
            page_info = connection.get("pageInfo")
            if not isinstance(nodes, list) or not isinstance(page_info, dict):
                raise RuntimeError("GitHub review-summary inventory is malformed")
            for comment in nodes:
                if not isinstance(comment, dict):
                    raise RuntimeError("GitHub returned an invalid review-summary comment")
                body = comment.get("body")
                author = comment.get("author")
                if not isinstance(body, str) or not isinstance(author, dict):
                    continue
                markers = summary_fallback_markers(body)
                if (
                    markers
                    and all(version == SUMMARY_FALLBACK_MARKER_VERSION for version, _ in markers)
                    and _github_actor_is_bot_identity(author)
                    and author.get("id") == viewer_identity.get("id")
                    and author.get("login") == viewer_identity.get("login")
                ):
                    bodies.append(body)
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                raise RuntimeError("GitHub omitted the next review-summary cursor")
            if cursor in seen_cursors:
                raise RuntimeError("GitHub repeated a review-summary cursor")
            seen_cursors.add(cursor)
        return tuple(bodies)

    def _live_review_head_sha(self) -> str:
        _, data = self.pr._requester.requestJsonAndCheck(
            "GET", f"{self.base_url}/repos/{self.repo}/pulls/{self.pr_num}"
        )
        head = data.get("head") if isinstance(data, dict) else None
        sha = head.get("sha") if isinstance(head, dict) else None
        if not sha:
            raise RuntimeError("GitHub pull request response has no head SHA")
        return sha

    def _check_review_thread_head(self, kind: ReviewThreadActionKind,
                                  expected_head_sha: str) -> tuple[Optional[str], Optional[ReviewThreadActionOutcome]]:
        try:
            current_head_sha = self._live_review_head_sha()
        except Exception as e:
            return None, ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                reason=f"head_check_failed: {e}",
                **_review_thread_failure_details(e),
            )
        if current_head_sha != expected_head_sha:
            return current_head_sha, ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.STALE_HEAD,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason="pull_request_head_changed",
            )
        return current_head_sha, None

    def _revalidate_review_thread_mutation(
        self,
        kind: ReviewThreadActionKind,
        expected_head_sha: str,
        expected_thread: ReviewThreadSnapshot,
        *,
        comment_id: Optional[int] = None,
        expected_finding_threads: Optional[tuple[ReviewThreadSnapshot, ...]] = None,
    ) -> tuple[Optional[str], Optional[ReviewThreadActionOutcome]]:
        """Fail closed unless the exact planned thread inventory is still current."""
        current_head_sha, blocked = self._check_review_thread_head(kind, expected_head_sha)
        if blocked:
            return current_head_sha, blocked
        root = expected_thread.root_comment
        expected_is_safe = bool(
            expected_thread.finding_id
            and expected_thread.bot_owned
            and not expected_thread.has_replies
            and not expected_thread.is_resolved
            and root
            and (comment_id is None or root.database_id == comment_id)
            and (kind != ReviewThreadActionKind.RESOLVE or expected_thread.viewer_can_resolve)
        )
        if not expected_is_safe:
            return current_head_sha, ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                thread_id=expected_thread.thread_id,
                comment_id=comment_id,
                reason="planned_thread_precondition_is_not_safe",
            )
        try:
            snapshots = self.get_review_thread_snapshots(require_viewer_bot=True)
            matches = [
                thread
                for thread in snapshots
                if thread.thread_id == expected_thread.thread_id
            ]
        except Exception as e:
            return current_head_sha, ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                thread_id=expected_thread.thread_id,
                comment_id=comment_id,
                reason=f"thread_revalidation_failed: {e}",
                **_review_thread_failure_details(e),
            )
        if expected_finding_threads is not None:
            expected_by_id = {thread.thread_id: thread for thread in expected_finding_threads}
            current_finding_threads = [
                thread for thread in snapshots
                if thread.finding_id == expected_thread.finding_id
            ]
            current_by_id = {thread.thread_id: thread for thread in current_finding_threads}
            if (
                len(expected_by_id) != len(expected_finding_threads)
                or len(current_by_id) != len(current_finding_threads)
                or current_by_id != expected_by_id
            ):
                return current_head_sha, ReviewThreadActionOutcome(
                    kind=kind,
                    state=ReviewThreadActionState.STALE_INVENTORY,
                    expected_head_sha=expected_head_sha,
                    current_head_sha=current_head_sha,
                    thread_id=expected_thread.thread_id,
                    comment_id=comment_id,
                    reason="finding_thread_set_changed_since_inventory",
                )
        if len(matches) != 1 or matches[0] != expected_thread:
            return current_head_sha, ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                thread_id=expected_thread.thread_id,
                comment_id=comment_id,
                reason="review_thread_changed_since_inventory",
            )
        # Inventory retrieval is paginated and can race with a force-push. Check
        # the head once more immediately before the destructive mutation.
        return self._check_review_thread_head(kind, expected_head_sha)

    def _review_thread_post_mutation_outcome(
        self,
        kind: ReviewThreadActionKind,
        expected_head_sha: str,
        *,
        thread_id: Optional[str] = None,
        comment_id: Optional[int] = None,
        comment_node_id: Optional[str] = None,
        force_refresh_reason: Optional[str] = None,
    ) -> ReviewThreadActionOutcome:
        """Confirm head stability after a side effect before dependent cleanup."""
        try:
            current_head_sha = self._live_review_head_sha()
        except Exception as e:
            return ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                thread_id=thread_id,
                comment_id=comment_id,
                comment_node_id=comment_node_id,
                reason=f"post_mutation_head_check_failed: {e}",
                mutation_attempted=True,
                mutation_result_ambiguous=True,
                **_review_thread_failure_details(e),
            )
        if current_head_sha != expected_head_sha:
            return ReviewThreadActionOutcome(
                kind=kind,
                state=ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                thread_id=thread_id,
                comment_id=comment_id,
                comment_node_id=comment_node_id,
                reason="pull_request_head_changed_after_mutation",
                mutation_attempted=True,
                mutation_result_ambiguous=True,
            )
        return ReviewThreadActionOutcome(
            kind=kind,
            state=(
                ReviewThreadActionState.APPLIED_REQUIRES_REFRESH
                if force_refresh_reason
                else ReviewThreadActionState.APPLIED
            ),
            expected_head_sha=expected_head_sha,
            current_head_sha=current_head_sha,
            thread_id=thread_id,
            comment_id=comment_id,
            comment_node_id=comment_node_id,
            reason=force_refresh_reason,
            mutation_attempted=True,
            mutation_result_ambiguous=bool(force_refresh_reason),
        )

    def _create_review_thread_locked(
        self,
        comment: dict,
        expected_head_sha: str,
        finding_id: str,
        anchor: ReviewThreadAnchor,
        expected_threads: tuple[ReviewThreadSnapshot, ...],
    ) -> ReviewThreadActionOutcome:
        current_head_sha, blocked = self._check_review_thread_head(
            ReviewThreadActionKind.CREATE, expected_head_sha
        )
        if blocked:
            return blocked
        try:
            same_finding = [
                thread
                for thread in self.get_review_thread_snapshots(require_viewer_bot=True)
                if thread.finding_id == finding_id
            ]
        except Exception as e:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason=f"create_inventory_failed: {e}",
                **_review_thread_failure_details(e),
            )
        expected_by_id = {thread.thread_id: thread for thread in expected_threads}
        current_by_id = {thread.thread_id: thread for thread in same_finding}
        if (
            len(expected_by_id) != len(expected_threads)
            or len(current_by_id) != len(same_finding)
            or any(thread.finding_id != finding_id for thread in expected_threads)
        ):
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason="create_inventory_identity_set_is_invalid",
            )
        if any(
            current_by_id.get(thread_id) != expected_thread
            for thread_id, expected_thread in expected_by_id.items()
        ):
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason="finding_thread_changed_since_planning",
            )
        if any(
            thread.is_resolved
            and (
                not thread.bot_owned
                or thread.has_replies
                or not thread.resolved_by_viewer_bot
            )
            for thread in same_finding
        ):
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason="finding_thread_authoritatively_resolved_since_planning",
            )
        unexpected = [
            thread for thread in same_finding
            if thread.thread_id not in expected_by_id
        ]
        if unexpected:
            root = unexpected[0].root_comment if len(unexpected) == 1 else None
            if (
                len(unexpected) == 1
                and not unexpected[0].is_resolved
                and unexpected[0].anchor == anchor
                and unexpected[0].bot_owned
                and not unexpected[0].has_replies
                and root
                and root.body == comment.get("body")
            ):
                return ReviewThreadActionOutcome(
                    kind=ReviewThreadActionKind.CREATE,
                    state=ReviewThreadActionState.ALREADY_APPLIED,
                    expected_head_sha=expected_head_sha,
                    current_head_sha=current_head_sha,
                    thread_id=unexpected[0].thread_id,
                    comment_id=root.database_id,
                    comment_node_id=root.node_id,
                    reason="finding_thread_already_created",
                )
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason="finding_thread_appeared_since_planning",
            )
        current_head_sha, blocked = self._check_review_thread_head(
            ReviewThreadActionKind.CREATE, expected_head_sha
        )
        if blocked:
            return blocked
        payload = dict(comment)
        payload["commit_id"] = expected_head_sha
        try:
            _, data = self.pr._requester.requestJsonAndCheck(
                "POST", f"{self.base_url}/repos/{self.repo}/pulls/{self.pr_num}/comments", input=payload
            )
            comment_id = data.get("id") if isinstance(data, dict) else None
            comment_node_id = data.get("node_id") if isinstance(data, dict) else None
            if not comment_id or not comment_node_id:
                return self._review_thread_post_mutation_outcome(
                    ReviewThreadActionKind.CREATE,
                    expected_head_sha,
                    comment_id=comment_id,
                    comment_node_id=comment_node_id,
                    force_refresh_reason="created_comment_identity_incomplete",
                )
            try:
                active = [
                    thread
                    for thread in self.get_review_thread_snapshots(require_viewer_bot=True)
                    if not thread.is_resolved and thread.finding_id == finding_id and thread.anchor == anchor
                ]
            except Exception as e:
                return self._review_thread_post_mutation_outcome(
                    ReviewThreadActionKind.CREATE,
                    expected_head_sha,
                    comment_id=comment_id,
                    comment_node_id=comment_node_id,
                    force_refresh_reason=f"post_create_inventory_failed: {e}",
                )
            if not active:
                return self._review_thread_post_mutation_outcome(
                    ReviewThreadActionKind.CREATE,
                    expected_head_sha,
                    comment_id=comment_id,
                    comment_node_id=comment_node_id,
                    force_refresh_reason="created_finding_thread_not_observed",
                )
            canonical = min(
                active,
                key=lambda thread: (
                    thread.root_comment.database_id
                    if thread.root_comment and thread.root_comment.database_id
                    else 2**63,
                    thread.thread_id,
                ),
            )
            if len(active) > 1:
                if any(
                    not thread.bot_owned
                    or thread.has_replies
                    or not thread.viewer_can_resolve
                    or not thread.root_comment
                    or thread.root_comment.body != comment.get("body")
                    for thread in active
                ):
                    return self._review_thread_post_mutation_outcome(
                        ReviewThreadActionKind.CREATE,
                        expected_head_sha,
                        comment_id=comment_id,
                        comment_node_id=comment_node_id,
                        force_refresh_reason="concurrent_finding_thread_not_safe_to_converge",
                    )
                for duplicate in active:
                    if duplicate.thread_id == canonical.thread_id:
                        continue
                    resolved = self._resolve_review_thread_locked(
                        duplicate.thread_id,
                        expected_head_sha,
                        duplicate,
                    )
                    if not resolved.succeeded:
                        return self._review_thread_post_mutation_outcome(
                            ReviewThreadActionKind.CREATE,
                            expected_head_sha,
                            comment_id=comment_id,
                            comment_node_id=comment_node_id,
                            force_refresh_reason="concurrent_finding_thread_convergence_failed",
                        )
            canonical_root = canonical.root_comment
            canonical_comment_id = canonical_root.database_id if canonical_root else None
            canonical_comment_node_id = canonical_root.node_id if canonical_root else None
            return self._review_thread_post_mutation_outcome(
                ReviewThreadActionKind.CREATE,
                expected_head_sha,
                thread_id=canonical.thread_id,
                comment_id=canonical_comment_id or comment_id,
                comment_node_id=canonical_comment_node_id or comment_node_id,
                force_refresh_reason=(
                    "canonical_comment_identity_incomplete"
                    if not canonical_comment_id or not canonical_comment_node_id
                    else None
                ),
            )
        except Exception as e:
            failure_details = _review_thread_failure_details(e)
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                reason=f"create_failed: {e}",
                mutation_attempted=True,
                mutation_result_ambiguous=failure_details["failure_kind"] in {
                    ReviewThreadFailureKind.RATE_LIMITED,
                    ReviewThreadFailureKind.PROVIDER_FAILURE,
                },
                **failure_details,
            )

    def create_review_thread(
        self,
        comment: dict,
        expected_head_sha: str,
        expected_threads: tuple[ReviewThreadSnapshot, ...] = (),
    ) -> ReviewThreadActionOutcome:
        marker_values = finding_identity_markers(comment.get("body") if isinstance(comment, dict) else "")
        marker_ids = {
            finding_id
            for marker_version, finding_id in marker_values
            if marker_version == FINDING_IDENTITY_MARKER_VERSION
        }
        anchor = ReviewThreadAnchor.from_github(
            comment.get("path") if isinstance(comment, dict) else None,
            comment.get("line") if isinstance(comment, dict) else None,
            comment.get("start_line") if isinstance(comment, dict) else None,
            comment.get("side", "RIGHT") if isinstance(comment, dict) else "RIGHT",
            comment.get("start_side") if isinstance(comment, dict) else None,
        )
        if (
            len(marker_values) != 1
            or len(marker_ids) != 1
            or anchor is None
        ):
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                reason="create_requires_one_supported_finding_marker_and_anchor",
                failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
            )
        finding_id = next(iter(marker_ids))
        try:
            with _review_thread_mutation_lock(self.repo, self.pr_num, finding_id):
                return self._create_review_thread_locked(
                    comment,
                    expected_head_sha,
                    finding_id,
                    anchor,
                    expected_threads,
                )
        except _ReviewThreadMutationLockError as error:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.CREATE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                reason=f"create_coordination_failed: {error}",
                failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
            )

    def update_review_thread(
        self,
        comment_id: int,
        body: str,
        expected_head_sha: str,
        expected_thread: ReviewThreadSnapshot,
        expected_finding_threads: Optional[tuple[ReviewThreadSnapshot, ...]] = None,
    ) -> ReviewThreadActionOutcome:
        if not expected_thread.finding_id:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.UPDATE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                comment_id=comment_id,
                reason="planned_thread_finding_id_is_missing",
            )
        if expected_finding_threads is None:
            expected_finding_threads = (expected_thread,)
        try:
            with _review_thread_mutation_lock(self.repo, self.pr_num, expected_thread.finding_id):
                return self._update_review_thread_locked(
                    comment_id,
                    body,
                    expected_head_sha,
                    expected_thread,
                    expected_finding_threads,
                )
        except _ReviewThreadMutationLockError as error:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.UPDATE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                comment_id=comment_id,
                reason=f"update_coordination_failed: {error}",
                failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
            )

    def _update_review_thread_locked(
        self,
        comment_id: int,
        body: str,
        expected_head_sha: str,
        expected_thread: ReviewThreadSnapshot,
        expected_finding_threads: Optional[tuple[ReviewThreadSnapshot, ...]],
    ) -> ReviewThreadActionOutcome:
        current_head_sha, blocked = self._revalidate_review_thread_mutation(
            ReviewThreadActionKind.UPDATE,
            expected_head_sha,
            expected_thread,
            comment_id=comment_id,
            expected_finding_threads=expected_finding_threads,
        )
        if blocked:
            return blocked
        try:
            _, data = self.pr._requester.requestJsonAndCheck(
                "PATCH", f"{self.base_url}/repos/{self.repo}/pulls/comments/{comment_id}", input={"body": body}
            )
            return self._review_thread_post_mutation_outcome(
                ReviewThreadActionKind.UPDATE,
                expected_head_sha,
                comment_id=data.get("id", comment_id) if isinstance(data, dict) else comment_id,
                comment_node_id=data.get("node_id") if isinstance(data, dict) else None,
            )
        except Exception as e:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.UPDATE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                comment_id=comment_id,
                reason=f"update_failed: {e}",
                mutation_attempted=True,
                **_review_thread_failure_details(e),
            )

    def resolve_review_thread(
        self,
        thread_id: str,
        expected_head_sha: str,
        expected_thread: ReviewThreadSnapshot,
    ) -> ReviewThreadActionOutcome:
        if not expected_thread.finding_id:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.RESOLVE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                thread_id=thread_id,
                reason="planned_thread_finding_id_is_missing",
            )
        try:
            with _review_thread_mutation_lock(self.repo, self.pr_num, expected_thread.finding_id):
                return self._resolve_review_thread_locked(
                    thread_id,
                    expected_head_sha,
                    expected_thread,
                )
        except _ReviewThreadMutationLockError as error:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.RESOLVE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                thread_id=thread_id,
                reason=f"resolve_coordination_failed: {error}",
                failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
            )

    def _resolve_review_thread_locked(
        self,
        thread_id: str,
        expected_head_sha: str,
        expected_thread: ReviewThreadSnapshot,
    ) -> ReviewThreadActionOutcome:
        if thread_id != expected_thread.thread_id:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.RESOLVE,
                state=ReviewThreadActionState.STALE_INVENTORY,
                expected_head_sha=expected_head_sha,
                current_head_sha=None,
                thread_id=thread_id,
                reason="planned_thread_id_mismatch",
            )
        current_head_sha, blocked = self._revalidate_review_thread_mutation(
            ReviewThreadActionKind.RESOLVE,
            expected_head_sha,
            expected_thread,
        )
        if blocked:
            return blocked
        mutation = """
        mutation($threadId: ID!) {
          resolveReviewThread(input: {threadId: $threadId}) {
            thread { id isResolved }
          }
        }
        """
        try:
            data = self._request_review_thread_graphql(mutation, {"threadId": thread_id})
            thread = data.get("resolveReviewThread", {}).get("thread")
            if not isinstance(thread, dict) or not thread.get("isResolved"):
                raise RuntimeError("resolveReviewThread did not return a resolved thread")
            return self._review_thread_post_mutation_outcome(
                ReviewThreadActionKind.RESOLVE,
                expected_head_sha,
                thread_id=thread.get("id") or thread_id,
            )
        except Exception as e:
            return ReviewThreadActionOutcome(
                kind=ReviewThreadActionKind.RESOLVE,
                state=ReviewThreadActionState.FAILED,
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
                thread_id=thread_id,
                reason=f"resolve_failed: {e}",
                mutation_attempted=True,
                **_review_thread_failure_details(e),
            )

    def resolve_comment_thread(self, comment_id: int) -> bool:
        """Resolve the review thread containing the given comment via GitHub GraphQL API."""
        try:
            owner, repo_name = self.repo.split("/")

            # Get the comment's node_id via REST
            headers, data = self.pr._requester.requestJsonAndCheck(
                "GET", f"{self.base_url}/repos/{self.repo}/pulls/comments/{comment_id}"
            )
            comment_node_id = data.get("node_id", "")
            if not comment_node_id:
                get_logger().warning(f"No node_id found for comment {comment_id}")
                return False

            # Find the review thread containing this comment (paginated)
            thread_id = None
            is_already_resolved = False
            cursor = None
            while True:
                after_clause = f', after: "{cursor}"' if cursor else ""
                query = f"""
                query {{
                    repository(owner: "{owner}", name: "{repo_name}") {{
                        pullRequest(number: {self.pr_num}) {{
                            reviewThreads(first: 100{after_clause}) {{
                                pageInfo {{ hasNextPage endCursor }}
                                nodes {{
                                    id
                                    isResolved
                                    comments(first: 100) {{
                                        nodes {{
                                            id
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                """
                response_tuple = self.github_client._Github__requester.requestJson(
                    "POST", "/graphql", input={"query": query}
                )
                if not (isinstance(response_tuple, tuple) and len(response_tuple) == 3):
                    get_logger().error("Unexpected GraphQL response format")
                    return False

                response_json = json.loads(response_tuple[2])
                errors = response_json.get("errors")
                if errors:
                    get_logger().error(
                        f"GraphQL errors querying review threads: {errors}"
                    )
                    return False
                review_threads = (response_json.get("data", {}).get("repository", {})
                                  .get("pullRequest", {}).get("reviewThreads", {}))
                threads = review_threads.get("nodes", [])

                for thread in threads:
                    comment_ids = [c["id"] for c in thread.get("comments", {}).get("nodes", [])]
                    if comment_node_id in comment_ids:
                        if thread.get("isResolved"):
                            is_already_resolved = True
                        else:
                            thread_id = thread["id"]
                        break

                if thread_id or is_already_resolved:
                    break
                page_info = review_threads.get("pageInfo", {})
                if not page_info.get("hasNextPage"):
                    break
                cursor = page_info.get("endCursor")

            if is_already_resolved:
                get_logger().info(f"Thread for comment {comment_id} is already resolved")
                return True

            if not thread_id:
                get_logger().warning(f"No thread found for comment {comment_id}")
                return False

            # Resolve the thread
            mutation = f"""
            mutation {{
                resolveReviewThread(input: {{threadId: "{thread_id}"}}) {{
                    thread {{
                        isResolved
                    }}
                }}
            }}
            """
            resolve_tuple = self.github_client._Github__requester.requestJson(
                "POST", "/graphql", input={"query": mutation}
            )
            if not isinstance(resolve_tuple, tuple) or len(resolve_tuple) != 3:
                get_logger().error(f"Unexpected mutation response format for thread {thread_id}: {type(resolve_tuple)}")
                return False
            resolve_json = json.loads(resolve_tuple[2])
            errors = resolve_json.get("errors")
            if errors:
                get_logger().error(f"GraphQL errors resolving thread {thread_id}: {errors}")
                return False
            is_resolved = (resolve_json.get("data", {}).get("resolveReviewThread", {})
                           .get("thread", {}).get("isResolved", False))
            if not is_resolved:
                get_logger().warning(
                    f"Resolve mutation returned isResolved=false for thread "
                    f"{thread_id} — possible permission issue"
                )
                return False
            get_logger().info(f"Resolved review thread {thread_id}")
            return True
        except Exception as e:
            get_logger().exception(f"Failed to resolve comment thread: {e}")
            return False

    def _publish_inline_comments_fallback_with_verification(self, comments: list[dict]):
        """
        Check each inline comment separately against the GitHub API and discard of invalid comments,
        then publish all the remaining valid comments in a single review.
        For invalid comments, also try removing the suggestion part and posting the comment just on the first line.
        """
        verified_comments, invalid_comments = self._verify_code_comments(comments)

        # publish as a group the verified comments
        if verified_comments:
            self.pr.create_review(commit=self.last_commit_id, comments=verified_comments)

        # try to publish one by one the invalid comments as a one-line code comment
        if invalid_comments and get_settings().github.try_fix_invalid_inline_comments:
            invalid_comments_list = [comment for comment, _ in invalid_comments]
            fixed_comments_as_one_liner = self._try_fix_invalid_inline_comments(invalid_comments_list)
            for comment in fixed_comments_as_one_liner:
                try:
                    self.publish_inline_comments([comment], disable_fallback=True)
                    get_logger().info(f"Published invalid comment as a single line comment: {comment}")
                except:
                    get_logger().error(f"Failed to publish invalid comment as a single line comment: {comment}")

            dropped_count = len(invalid_comments) - len(fixed_comments_as_one_liner)
            if dropped_count > 0:
                dropped_paths = [c.get("path") for c, _ in invalid_comments]
                for fixed_c in fixed_comments_as_one_liner:
                    fixed_path = fixed_c.get("path")
                    if fixed_path in dropped_paths:
                        dropped_paths.remove(fixed_path)
                get_logger().warning(
                    f"Dropped {dropped_count} invalid comments that could not be fixed. Paths: {dropped_paths}"
                )
        elif invalid_comments:
            dropped_paths = [c.get("path") for c, _ in invalid_comments]
            get_logger().warning(
                f"Dropped {len(invalid_comments)} invalid comments "
                f"(try_fix_invalid_inline_comments is off). Paths: {dropped_paths}"
            )

    def _verify_code_comment(self, comment: dict):
        is_verified = False
        e = None
        try:
            # event ="" # By leaving this blank, you set the review action state to PENDING
            input = dict(commit_id=self.last_commit_id.sha, comments=[comment])
            headers, data = self.pr._requester.requestJsonAndCheck(
                "POST", f"{self.pr.url}/reviews", input=input)
            pending_review_id = data["id"]
            is_verified = True
        except Exception as err:
            is_verified = False
            pending_review_id = None
            e = err
        if pending_review_id is not None:
            try:
                self.pr._requester.requestJsonAndCheck("DELETE", f"{self.pr.url}/reviews/{pending_review_id}")
            except Exception:
                pass
        return is_verified, e

    def _verify_code_comments(self, comments: list[dict]) -> tuple[list[dict], list[tuple[dict, Exception]]]:
        """Very each comment against the GitHub API and return 2 lists: 1 of verified and 1 of invalid comments"""
        verified_comments = []
        invalid_comments = []
        for comment in comments:
            time.sleep(1)  # for avoiding secondary rate limit
            is_verified, e = self._verify_code_comment(comment)
            if is_verified:
                verified_comments.append(comment)
            else:
                invalid_comments.append((comment, e))
        return verified_comments, invalid_comments

    def _try_fix_invalid_inline_comments(self, invalid_comments: list[dict]) -> list[dict]:
        """
        Try fixing invalid comments by removing the suggestion part and setting the comment just on the first line.
        Return only comments that have been modified in some way.
        This is a best-effort attempt to fix invalid comments, and should be verified accordingly.
        """
        import copy
        fixed_comments = []
        for comment in invalid_comments:
            try:
                fixed_comment = copy.deepcopy(comment)  # avoid modifying the original comment dict for later logging
                if "```suggestion" in comment["body"]:
                    fixed_comment["body"] = comment["body"].split("```suggestion")[0]
                if "start_line" in comment:
                    fixed_comment["line"] = comment["start_line"]
                    del fixed_comment["start_line"]
                if "start_side" in comment:
                    fixed_comment["side"] = comment["start_side"]
                    del fixed_comment["start_side"]
                if fixed_comment != comment:
                    fixed_comments.append(fixed_comment)
            except Exception as e:
                get_logger().error(f"Failed to fix inline comment, error: {e}")
        return fixed_comments

    def publish_code_suggestions(self, code_suggestions: list) -> bool:
        """
        Publishes code suggestions as comments on the PR.
        """
        post_parameters_list = []

        code_suggestions_with_fingerprints = copy.deepcopy(code_suggestions)
        for suggestion in code_suggestions_with_fingerprints:
            suggestion["_dedup_code_fp"] = code_fingerprint(
                suggestion.get("relevant_file", ""), None, suggestion.get("body", ""))
        code_suggestions_validated = self.validate_comments_inside_hunks(code_suggestions_with_fingerprints)

        for suggestion in code_suggestions_validated:
            body = suggestion['body']
            relevant_file = suggestion['relevant_file']
            relevant_lines_start = suggestion['relevant_lines_start']
            relevant_lines_end = suggestion['relevant_lines_end']

            if not relevant_lines_start or relevant_lines_start == -1:
                get_logger().exception(
                    f"Failed to publish code suggestion, relevant_lines_start is {relevant_lines_start}")
                continue

            if relevant_lines_end < relevant_lines_start:
                get_logger().exception(f"Failed to publish code suggestion, "
                                  f"relevant_lines_end is {relevant_lines_end} and "
                                  f"relevant_lines_start is {relevant_lines_start}")
                continue

            if relevant_lines_end > relevant_lines_start:
                post_parameters = {
                    "body": body,
                    "path": relevant_file,
                    "line": relevant_lines_end,
                    "start_line": relevant_lines_start,
                    "start_side": "RIGHT",
                    "_dedup_code_fp": suggestion.get("_dedup_code_fp"),
                }
            else:  # API is different for single line comments
                post_parameters = {
                    "body": body,
                    "path": relevant_file,
                    "line": relevant_lines_start,
                    "side": "RIGHT",
                    "_dedup_code_fp": suggestion.get("_dedup_code_fp"),
                }
            post_parameters_list.append(post_parameters)

        try:
            self.publish_inline_comments(post_parameters_list)
            return True
        except Exception as e:
            get_logger().error(f"Failed to publish code suggestion, error: {e}")
            return False

    def edit_comment(self, comment, body: str):
        try:
            body = self.limit_output_characters(body, self.max_comment_chars)
            comment.edit(body=body)
        except GithubException as e:
            if hasattr(e, "status") and e.status == 403:
                # Log as warning for permission-related issues (usually due to polling)
                get_logger().warning(
                    "Failed to edit github comment due to permission restrictions",
                    artifact={"error": e})
            else:
                get_logger().exception("Failed to edit github comment", artifact={"error": e})

    def edit_comment_from_comment_id(self, comment_id: int, body: str):
        try:
            # self.pr.get_issue_comment(comment_id).edit(body)
            body = self.limit_output_characters(body, self.max_comment_chars)
            headers, data_patch = self.pr._requester.requestJsonAndCheck(
                "PATCH", f"{self.base_url}/repos/{self.repo}/issues/comments/{comment_id}",
                input={"body": body}
            )
        except Exception as e:
            get_logger().exception(f"Failed to edit comment, error: {e}")

    def reply_to_comment_from_comment_id(self, comment_id: int, body: str):
        try:
            # self.pr.get_issue_comment(comment_id).edit(body)
            body = self.limit_output_characters(body, self.max_comment_chars)
            headers, data_patch = self.pr._requester.requestJsonAndCheck(
                "POST", f"{self.base_url}/repos/{self.repo}/pulls/{self.pr_num}/comments/{comment_id}/replies",
                input={"body": body}
            )
        except Exception as e:
            get_logger().exception(f"Failed to reply comment, error: {e}")

    def get_comment_body_from_comment_id(self, comment_id: int):
        try:
            # self.pr.get_issue_comment(comment_id).edit(body)
            headers, data_patch = self.pr._requester.requestJsonAndCheck(
                "GET", f"{self.base_url}/repos/{self.repo}/issues/comments/{comment_id}"
            )
            return data_patch.get("body","")
        except Exception as e:
            get_logger().exception(f"Failed to edit comment, error: {e}")
            return None

    def publish_file_comments(self, file_comments: list) -> bool:
        try:
            headers, existing_comments = self.pr._requester.requestJsonAndCheck(
                "GET", f"{self.pr.url}/comments"
            )
            for comment in file_comments:
                comment['commit_id'] = self.last_commit_id.sha
                comment['body'] = self.limit_output_characters(comment['body'], self.max_comment_chars)

                found = False
                for existing_comment in existing_comments:
                    comment['commit_id'] = self.last_commit_id.sha
                    our_app_name = get_settings().get("GITHUB.APP_NAME", "")
                    same_comment_creator = False
                    if self.deployment_type == 'app':
                        same_comment_creator = our_app_name.lower() in existing_comment['user']['login'].lower()
                    elif self.deployment_type == 'user':
                        same_comment_creator = self.github_user_id == existing_comment['user']['login']
                    if existing_comment['subject_type'] == 'file' and comment['path'] == existing_comment['path'] and same_comment_creator:

                        headers, data_patch = self.pr._requester.requestJsonAndCheck(
                            "PATCH", f"{self.base_url}/repos/{self.repo}/pulls/comments/{existing_comment['id']}", input={"body":comment['body']}
                        )
                        found = True
                        break
                if not found:
                    headers, data_post = self.pr._requester.requestJsonAndCheck(
                        "POST", f"{self.pr.url}/comments", input=comment
                    )
            return True
        except Exception as e:
            get_logger().error(f"Failed to publish diffview file summary, error: {e}")
            return False

    def remove_initial_comment(self):
        try:
            for comment in getattr(self.pr, 'comments_list', []):
                if comment.is_temporary:
                    self.remove_comment(comment)
        except Exception as e:
            get_logger().exception(f"Failed to remove initial comment, error: {e}")

    def remove_comment(self, comment):
        try:
            comment.delete()
        except Exception as e:
            get_logger().exception(f"Failed to remove comment, error: {e}")

    def get_title(self):
        return self.pr.title

    def get_languages(self):
        languages = self._get_repo().get_languages()
        return languages

    def get_pr_branch(self):
        return self.pr.head.ref

    def get_pr_owner_id(self) -> str | None:
        if not self.repo:
            return None
        return self.repo.split('/')[0]

    def get_pr_description_full(self):
        return self.pr.body

    def get_user_id(self):
        if not self.github_user_id:
            try:
                self.github_user_id = self.github_client.get_user().raw_data['login']
            except Exception as e:
                self.github_user_id = ""
                # logging.exception(f"Failed to get user id, error: {e}")
        return self.github_user_id

    def get_notifications(self, since: datetime):
        deployment_type = get_settings().get("GITHUB.DEPLOYMENT_TYPE", "user")

        if deployment_type != 'user':
            raise ValueError("Deployment mode must be set to 'user' to get notifications")

        notifications = self.github_client.get_user().get_notifications(since=since)
        return notifications

    def get_issue_comments(self):
        return self.pr.get_issue_comments()

    def get_repo_settings(self):
        settings_files = []
        global_settings = self._get_global_repo_settings()
        if global_settings:
            settings_files.append(("global", global_settings))

        # Normalize each candidate before applying precedence so a whitespace-only
        # settings value doesn't short-circuit the PR_AGENT_CONFIG_BRANCH fallback.
        settings_branch = get_settings().get("CONFIG.CONFIG_BRANCH", None)
        settings_branch = settings_branch.strip() if isinstance(settings_branch, str) else ""
        env_branch = (os.environ.get("PR_AGENT_CONFIG_BRANCH") or "").strip()
        config_branch = settings_branch or env_branch
        if config_branch:
            # Only treat a missing branch/file (GithubException) as an expected
            # reason to fall back to the default branch. Unexpected errors are
            # left to propagate so they aren't masked by a silent fallback.
            try:
                contents = self.repo_obj.get_contents(".pr_agent.toml", ref=config_branch).decoded_content
                if settings_files:
                    settings_files.append(("local", contents))
                    return settings_files
                return contents
            except GithubException as e:
                # Only a missing branch/file (404) is an expected reason to fall back to the default
                # branch. Other errors (e.g. 403/5xx) are surfaced rather than silently masked by a
                # fallback that could apply unintended settings.
                if e.status != 404:
                    raise
                get_logger().debug(
                    f"No .pr_agent.toml on branch '{config_branch}', falling back to default branch")
        try:
            # more logical to take 'pr_agent.toml' from the default branch
            contents = self.repo_obj.get_contents(".pr_agent.toml").decoded_content
            if config_branch and not settings_files:
                return contents
            settings_files.append(("local", contents))
        except GithubException as e:
            # A missing local .pr_agent.toml (404) is expected for most repos; log it quietly to
            # avoid warning noise, and surface only unexpected errors as warnings.
            if e.status == 404:
                get_logger().debug("No local .pr_agent.toml found; using existing settings")
            else:
                get_logger().warning(f"Failed to load .pr_agent.toml file, error: {e}")
        except Exception as e:
            get_logger().warning(f"Failed to load .pr_agent.toml file, error: {e}")

        return settings_files if settings_files else ""

    def _get_global_repo_settings(self):
        if not get_settings().config.use_global_settings_file:
            return ""

        # Be robust to providers built without full __init__ (e.g. __new__ in tests/helpers):
        # without a repo/client there is no org to resolve, so skip global settings quietly.
        if not getattr(self, "repo", None) or getattr(self, "github_client", None) is None:
            return ""

        repo_owner = self.get_pr_owner_id()
        if not repo_owner:
            return ""
        # Cache per org: global settings change rarely, so avoid a lookup (and repeated 403/404
        # fallbacks) on every webhook event.
        return get_cached_global_settings(
            f"github:{getattr(self, 'base_url', '')}:{repo_owner}",
            lambda: self._fetch_global_repo_settings(repo_owner))

    def _fetch_global_repo_settings(self, repo_owner):
        try:
            global_settings_repo = self.github_client.get_repo(f"{repo_owner}/pr-agent-settings")
            return global_settings_repo.get_contents(".pr_agent.toml").decoded_content
        except GithubException as e:
            # A missing pr-agent-settings repo/file (404) or lack of access (403) is an expected,
            # stable fallback (skip global settings, continue with local) — return "" so it's cached.
            if e.status in (403, 404):
                get_logger().debug(
                    "No accessible organization global .pr_agent.toml; using local settings only",
                    artifact={"status": e.status})
                return ""
            # Transient/unexpected errors propagate so the caller does not cache the failure.
            raise

    def get_repo_file_content(self, file_path: str, from_default_branch: bool = False):
        try:
            # Prefer the PR target (base) ref so repo-context instruction files match the branch
            # the PR is merging into. Fall back to the repo default branch when no PR base is
            # available, or always when from_default_branch is requested.
            if from_default_branch:
                ref = None
            else:
                base = getattr(getattr(self, "pr", None), "base", None)
                ref = getattr(base, "sha", None) or getattr(base, "ref", None)
            if ref:
                contents = self.repo_obj.get_contents(file_path, ref=ref).decoded_content
            else:
                contents = self.repo_obj.get_contents(file_path).decoded_content
            if isinstance(contents, bytes):
                return contents.decode("utf-8", errors="replace")
            return contents
        except GithubException as e:
            # A missing file is an expected "no context" outcome. Let transient/unexpected
            # errors propagate so build_repo_context() treats them as a fetch error and does
            # not cache an empty result until the TTL expires.
            if e.status == 404:
                return ""
            raise

    def get_workspace_name(self):
        return self.repo.split('/')[0]

    def add_eyes_reaction(self, issue_comment_id: int, disable_eyes: bool = False) -> Optional[int]:
        if disable_eyes:
            return None
        try:
            headers, data_patch = self.pr._requester.requestJsonAndCheck(
                "POST", f"{self.base_url}/repos/{self.repo}/issues/comments/{issue_comment_id}/reactions",
                input={"content": "eyes"}
            )
            return data_patch.get("id", None)
        except Exception as e:
            get_logger().warning(f"Failed to add eyes reaction, error: {e}")
            return None

    def remove_reaction(self, issue_comment_id: int, reaction_id: str) -> bool:
        try:
            # self.pr.get_issue_comment(issue_comment_id).delete_reaction(reaction_id)
            headers, data_patch = self.pr._requester.requestJsonAndCheck(
                "DELETE",
                f"{self.base_url}/repos/{self.repo}/issues/comments/{issue_comment_id}/reactions/{reaction_id}"
            )
            return True
        except Exception as e:
            get_logger().exception(f"Failed to remove eyes reaction, error: {e}")
            return False

    def _parse_pr_url(self, pr_url: str) -> Tuple[str, int]:
        parsed_url = urlparse(pr_url)

        if parsed_url.path.startswith('/api/v3'):
            parsed_url = urlparse(pr_url.replace("/api/v3", ""))

        path_parts = parsed_url.path.strip('/').split('/')
        if 'api.github.com' in parsed_url.netloc or '/api/v3' in pr_url:
            if len(path_parts) < 5 or path_parts[3] != 'pulls':
                raise ValueError("The provided URL does not appear to be a GitHub PR URL")
            repo_name = '/'.join(path_parts[1:3])
            try:
                pr_number = int(path_parts[4])
            except ValueError as e:
                raise ValueError("Unable to convert PR number to integer") from e
            return repo_name, pr_number

        if len(path_parts) < 4 or path_parts[2] != 'pull':
            raise ValueError("The provided URL does not appear to be a GitHub PR URL")

        repo_name = '/'.join(path_parts[:2])
        try:
            pr_number = int(path_parts[3])
        except ValueError as e:
            raise ValueError("Unable to convert PR number to integer") from e

        return repo_name, pr_number

    def _parse_issue_url(self, issue_url: str) -> Tuple[str, int]:
        parsed_url = urlparse(issue_url)

        if parsed_url.path.startswith('/api/v3'): #Check if came from github app
            parsed_url = urlparse(issue_url.replace("/api/v3", ""))

        path_parts = parsed_url.path.strip('/').split('/')
        if 'api.github.com' in parsed_url.netloc or '/api/v3' in issue_url: #Check if came from github app
            if len(path_parts) < 5 or path_parts[3] != 'issues':
                raise ValueError("The provided URL does not appear to be a GitHub ISSUE URL")
            repo_name = '/'.join(path_parts[1:3])
            try:
                issue_number = int(path_parts[4])
            except ValueError as e:
                raise ValueError("Unable to convert issue number to integer") from e
            return repo_name, issue_number

        if len(path_parts) < 4 or path_parts[2] != 'issues':
            raise ValueError("The provided URL does not appear to be a GitHub PR issue")

        repo_name = '/'.join(path_parts[:2])
        try:
            issue_number = int(path_parts[3])
        except ValueError as e:
            raise ValueError("Unable to convert issue number to integer") from e

        return repo_name, issue_number

    def _get_github_client(self):
        self.deployment_type = get_settings().get("GITHUB.DEPLOYMENT_TYPE", "user")
        self.auth = None
        if self.deployment_type == 'app':
            try:
                private_key = get_settings().github.private_key
                # The app id is an integer in the settings toml, but PyJWT >=2.11 requires a
                # string `iss` claim, and PyGithub 1.59 passes it through raw (#2955).
                app_id = str(get_settings().github.app_id)
            except AttributeError as e:
                raise ValueError("GitHub app ID and private key are required when using GitHub app deployment") from e
            if not self.installation_id:
                raise ValueError("GitHub app installation ID is required when using GitHub app deployment")
            auth = AppAuthentication(app_id=app_id, private_key=private_key,
                                     installation_id=self.installation_id)
            self.auth = auth
        elif self.deployment_type == 'user':
            try:
                token = get_settings().github.user_token
            except AttributeError as e:
                raise ValueError(
                    "GitHub token is required when using user deployment. See: "
                    "https://github.com/Codium-ai/pr-agent#method-2-run-from-source") from e
            self.auth = Auth.Token(token)
        if self.auth:
            return Github(auth=self.auth, base_url=self.base_url)
        else:
            raise ValueError("Could not authenticate to GitHub")

    def _get_repo(self):
        if hasattr(self, 'repo_obj') and \
                hasattr(self.repo_obj, 'full_name') and \
                self.repo_obj.full_name == self.repo:
            return self.repo_obj
        else:
            self.repo_obj = self.github_client.get_repo(self.repo)
            return self.repo_obj


    def _get_pr(self):
        return self._get_repo().get_pull(self.pr_num)

    def get_pr_file_content(self, file_path: str, branch: str) -> str:
        try:
            file_content_str = str(
                self._get_repo()
                .get_contents(file_path, ref=branch)
                .decoded_content.decode()
            )
        except Exception:
            file_content_str = ""
        return file_content_str

    def create_or_update_pr_file(
        self, file_path: str, branch: str, contents="", message=""
    ) -> None:
        try:
            file_obj = self._get_repo().get_contents(file_path, ref=branch)
            sha1=file_obj.sha
        except Exception:
            sha1=""
        self.repo_obj.update_file(
            path=file_path,
            message=message,
            content=contents,
            sha=sha1,
            branch=branch,
        )

    def _get_pr_file_content(self, file: FilePatchInfo, sha: str, path: str = None) -> str:
        return self.get_pr_file_content(path or file.filename, sha)

    def publish_labels(self, pr_types):
        try:
            label_color_map = {"Bug fix": "1d76db", "Tests": "e99695", "Bug fix with tests": "c5def5",
                               "Enhancement": "bfd4f2", "Documentation": "d4c5f9",
                               "Other": "d1bcf9"}
            post_parameters = []
            for p in pr_types:
                color = label_color_map.get(p, "d1bcf9")  # default to "Other" color
                post_parameters.append({"name": p, "color": color})
            headers, data = self.pr._requester.requestJsonAndCheck(
                "PUT", f"{self.pr.issue_url}/labels", input=post_parameters
            )
        except Exception as e:
            get_logger().warning(f"Failed to publish labels, error: {e}")

    def _read_pr_labels(self, update=False):
        if not update:
            return [label.name for label in self.pr.labels]

        _, labels = self.pr._requester.requestJsonAndCheck(
            "GET", f"{self.pr.issue_url}/labels"
        )
        return [label["name"] for label in labels]

    def get_pr_labels(self, update=False):
        try:
            return self._read_pr_labels(update=update)
        except Exception as e:
            get_logger().exception(f"Failed to get labels, error: {e}")
            # Preserve the provider's historical best-effort contract for callers
            # that do not need to distinguish failure from a confirmed empty set.
            return []

    def get_pr_labels_for_routing(self, update=False):
        try:
            return self._read_pr_labels(update=update)
        except Exception as e:
            get_logger().exception(f"Failed to get labels for review routing, error: {e}")
            # Routing must distinguish unavailable evidence from a confirmed empty
            # set so a metadata outage cannot select the quick profile.
            return None

    def get_repo_labels(self):
        labels = self.repo_obj.get_labels()
        return [label for label in itertools.islice(labels, 50)]

    def get_commit_messages(self):
        """
        Retrieves the commit messages of a pull request.

        Returns:
            str: A string containing the commit messages of the pull request.
        """
        max_tokens = get_settings().get("CONFIG.MAX_COMMITS_TOKENS", None)
        try:
            commit_list = self.pr.get_commits()
            commit_messages = [commit.commit.message for commit in commit_list]
            commit_messages_str = "\n".join([f"{i + 1}. {message}" for i, message in enumerate(commit_messages)])
        except Exception:
            commit_messages_str = ""
        if max_tokens:
            commit_messages_str = clip_tokens(commit_messages_str, max_tokens)
        return commit_messages_str

    def generate_link_to_relevant_line_number(self, suggestion) -> str:
        try:
            relevant_file = suggestion['relevant_file'].strip('`').strip("'").strip('\n')
            relevant_line_str = suggestion['relevant_line'].strip('\n')
            if not relevant_line_str:
                return ""

            position, absolute_position = find_line_number_of_relevant_line_in_file \
                (self.diff_files, relevant_file, relevant_line_str)

            if absolute_position != -1:
                # # link to right file only
                # link = f"https://github.com/{self.repo}/blob/{self.pr.head.sha}/{relevant_file}" \
                #        + "#" + f"L{absolute_position}"

                # link to diff
                sha_file = hashlib.sha256(relevant_file.encode('utf-8')).hexdigest()
                link = f"{self.base_url_html}/{self.repo}/pull/{self.pr_num}/files#diff-{sha_file}R{absolute_position}"
                return link
        except Exception as e:
            get_logger().info(f"Failed adding line link, error: {e}")

        return ""

    def get_line_link(self, relevant_file: str, relevant_line_start: int, relevant_line_end: int = None) -> str:
        sha_file = hashlib.sha256(relevant_file.encode('utf-8')).hexdigest()
        relevant_line_start, relevant_line_end = self._normalize_line_range(
            relevant_line_start, relevant_line_end
        )
        if relevant_line_start == -1:
            link = f"{self.base_url_html}/{self.repo}/pull/{self.pr_num}/files#diff-{sha_file}"
        elif relevant_line_end:
            link = f"{self.base_url_html}/{self.repo}/pull/{self.pr_num}/files#diff-{sha_file}R{relevant_line_start}-R{relevant_line_end}"
        else:
            link = f"{self.base_url_html}/{self.repo}/pull/{self.pr_num}/files#diff-{sha_file}R{relevant_line_start}"
        return link

    def get_lines_link_original_file(self, filepath: str, component_range: Range) -> str:
        """
        Returns the link to the original file on GitHub that corresponds to the given filepath and component range.

        Args:
            filepath (str): The path of the file.
            component_range (Range): The range of lines that represent the component.

        Returns:
            str: The link to the original file on GitHub.

        Example:
            >>> filepath = "path/to/file.py"
            >>> component_range = Range(line_start=10, line_end=20)
            >>> link = get_lines_link_original_file(filepath, component_range)
            >>> print(link)
            "https://github.com/{repo}/blob/{commit_sha}/{filepath}/#L11-L21"
        """
        line_start = component_range.line_start + 1
        line_end = component_range.line_end + 1
        # link = (f"https://github.com/{self.repo}/blob/{self.last_commit_id.sha}/{filepath}/"
        #         f"#L{line_start}-L{line_end}")
        link = (f"{self.base_url_html}/{self.repo}/blob/{self.last_commit_id.sha}/{filepath}/"
                f"#L{line_start}-L{line_end}")

        return link

    def get_pr_id(self):
        try:
            pr_id = f"{self.repo}/{self.pr_num}"
            return pr_id
        except:
            return ""

    def fetch_sub_issues(self, issue_url):
        """
        Fetch sub-issues linked to the given GitHub issue URL using GraphQL via PyGitHub.
        """
        sub_issues = set()

        # Extract owner, repo, and issue number from URL
        parts = issue_url.rstrip("/").split("/")
        owner, repo, issue_number = parts[-4], parts[-3], parts[-1]

        try:
            # Gets Issue ID from Issue Number
            query = f"""
            query {{
                repository(owner: "{owner}", name: "{repo}") {{
                    issue(number: {issue_number}) {{
                        id
                    }}
                }}
            }}
            """
            response_tuple = self.github_client._Github__requester.requestJson("POST", "/graphql",
                                                                               input={"query": query})

            # Extract the JSON response from the tuple and parses it
            if isinstance(response_tuple, tuple) and len(response_tuple) == 3:
                response_json = json.loads(response_tuple[2])
            else:
                get_logger().error(f"Unexpected response format: {response_tuple}")
                return sub_issues


            issue_id = (((response_json.get("data") or {})
                        .get("repository") or {})
                        .get("issue") or {}).get("id")

            if not issue_id:
                get_logger().warning(f"Issue ID not found for {issue_url}")
                return sub_issues

            # Fetch Sub-Issues
            sub_issues_query = f"""
            query {{
                node(id: "{issue_id}") {{
                    ... on Issue {{
                        subIssues(first: 10) {{
                            nodes {{
                                url
                            }}
                        }}
                    }}
                }}
            }}
            """
            sub_issues_response_tuple = self.github_client._Github__requester.requestJson("POST", "/graphql", input={
                "query": sub_issues_query})

            # Extract the JSON response from the tuple and parses it
            if isinstance(sub_issues_response_tuple, tuple) and len(sub_issues_response_tuple) == 3:
                sub_issues_response_json = json.loads(sub_issues_response_tuple[2])
            else:
                get_logger().error("Unexpected sub-issues response format", artifact={"response": sub_issues_response_tuple})
                return sub_issues

            sub_issues_data = (((sub_issues_response_json.get("data") or {})
                                .get("node") or {})
                                .get("subIssues") or {})
            if not sub_issues_data:
                get_logger().error("Invalid sub-issues response structure")
                return sub_issues

            nodes = sub_issues_data.get("nodes") or []
            get_logger().info(f"Github Sub-issues fetched: {len(nodes)}", artifact={"nodes": nodes})

            for sub_issue in nodes:
                if not sub_issue:
                    continue
                if "url" in sub_issue:
                    sub_issues.add(sub_issue["url"])

        except Exception as e:
            get_logger().exception(f"Failed to fetch sub-issues. Error: {e}")

        return sub_issues

    def auto_approve(self) -> bool:
        try:
            res = self.pr.create_review(event="APPROVE")
            if res.state == "APPROVED":
                return True
            return False
        except Exception as e:
            get_logger().exception(f"Failed to auto-approve, error: {e}")
            return False

    def calc_pr_statistics(self, pull_request_data: dict):
            return {}

    def validate_comments_inside_hunks(self, code_suggestions):
        """
        validate that all committable comments are inside PR hunks - this is a must for committable comments in GitHub
        """
        code_suggestions_copy = copy.deepcopy(code_suggestions)
        diff_files = self.get_diff_files()
        RE_HUNK_HEADER = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")

        diff_files = set_file_languages(diff_files)

        for suggestion in code_suggestions_copy:
            try:
                relevant_file_path = suggestion['relevant_file']
                for file in diff_files:
                    if file.filename == relevant_file_path:

                        # generate on-demand the patches range for the relevant file
                        patch_str = file.patch
                        if not hasattr(file, 'patches_range'):
                            file.patches_range = []
                            patch_lines = [
                                strip_git_line_ending(line)
                                for line in iter_git_patch_lines(patch_str)
                            ]
                            for i, line in enumerate(patch_lines):
                                if line.startswith('@@'):
                                    match = RE_HUNK_HEADER.match(line)
                                    # identify hunk header
                                    if match:
                                        section_header, size1, size2, start1, start2 = extract_hunk_headers(match)
                                        file.patches_range.append({'start': start2, 'end': start2 + size2 - 1})

                        patches_range = file.patches_range
                        comment_start_line = suggestion.get('relevant_lines_start', None)
                        comment_end_line = suggestion.get('relevant_lines_end', None)
                        original_suggestion = suggestion.get('original_suggestion', None) # needed for diff code
                        if not comment_start_line or not comment_end_line or not original_suggestion:
                            continue

                        # check if the comment is inside a valid hunk
                        is_valid_hunk = False
                        min_distance = float('inf')
                        patch_range_min = None
                        # find the hunk that contains the comment, or the closest one
                        for i, patch_range in enumerate(patches_range):
                            d1 = comment_start_line - patch_range['start']
                            d2 = patch_range['end'] - comment_end_line
                            if d1 >= 0 and d2 >= 0:  # found a valid hunk
                                is_valid_hunk = True
                                min_distance = 0
                                patch_range_min = patch_range
                                break
                            elif d1 * d2 <= 0:  # comment is possibly inside the hunk
                                d1_clip = abs(min(0, d1))
                                d2_clip = abs(min(0, d2))
                                d = max(d1_clip, d2_clip)
                                if d < min_distance:
                                    patch_range_min = patch_range
                                    min_distance = min(min_distance, d)
                        if not is_valid_hunk:
                            if min_distance < 10:  # 10 lines - a reasonable distance to consider the comment inside the hunk
                                # make the suggestion non-committable, yet multi line
                                suggestion['relevant_lines_start'] = max(suggestion['relevant_lines_start'], patch_range_min['start'])
                                suggestion['relevant_lines_end'] = min(suggestion['relevant_lines_end'], patch_range_min['end'])
                                body = suggestion['body'].strip()

                                # present new diff code in collapsible
                                existing_code = original_suggestion['existing_code'].rstrip() + "\n"
                                improved_code = original_suggestion['improved_code'].rstrip() + "\n"
                                diff = difflib.unified_diff(existing_code.split('\n'),
                                                            improved_code.split('\n'), n=999)
                                patch_orig = "\n".join(diff)
                                patch = "\n".join(
                                    strip_git_line_ending(line)
                                    for line in list(iter_git_patch_lines(patch_orig))[5:]
                                ).strip('\n')
                                diff_code = f"\n\n<details><summary>New proposed code:</summary>\n\n```diff\n{patch.rstrip()}\n```"
                                # replace ```suggestion ... ``` with diff_code, using regex:
                                body = re.sub(r'```suggestion.*?```', diff_code, body, flags=re.DOTALL)
                                body += "\n\n</details>"
                                suggestion['body'] = body
                                get_logger().info(f"Comment was moved to a valid hunk, "
                                                  f"start_line={suggestion['relevant_lines_start']}, end_line={suggestion['relevant_lines_end']}, file={file.filename}")
                            else:
                                get_logger().error(f"Comment is not inside a valid hunk, "
                                                   f"start_line={suggestion['relevant_lines_start']}, end_line={suggestion['relevant_lines_end']}, file={file.filename}")
            except Exception as e:
                get_logger().error(f"Failed to process patch for committable comment, error: {e}")
        return code_suggestions_copy

    #Clone related
    def _prepare_clone_url_with_token(self, repo_url_to_clone: str) -> str | None:
        scheme = "https://"

        #For example, to clone:
        #https://github.com/Codium-ai/pr-agent-pro.git
        #Need to embed inside the github token:
        #https://<token>@github.com/Codium-ai/pr-agent-pro.git

        github_token = self.auth.token
        github_base_url = self.base_url_html
        if not all([github_token, github_base_url]):
            get_logger().error("Either missing auth token or missing base url")
            return None
        if scheme not in github_base_url:
            get_logger().error(f"Base url: {redact_credentials(github_base_url)} is missing prefix: {scheme}")
            return None
        github_com = github_base_url.split(scheme)[1]  # e.g. 'github.com' or github.<org>.com
        if not github_com:
            get_logger().error(f"Base url: {redact_credentials(github_base_url)} has an empty base url")
            return None
        if github_com not in repo_url_to_clone:
            get_logger().error(f"url to clone: {redact_credentials(repo_url_to_clone)} "
                               f"does not contain {redact_credentials(github_base_url)}")
            return None
        repo_full_name = repo_url_to_clone.split(github_com)[-1]
        if not repo_full_name:
            get_logger().error(f"url to clone: {redact_credentials(repo_url_to_clone)} is malformed")
            return None

        clone_url = scheme
        if self.deployment_type == 'app':
            clone_url += "git:"
        clone_url += f"{github_token}@{github_com}{repo_full_name}"
        return clone_url
