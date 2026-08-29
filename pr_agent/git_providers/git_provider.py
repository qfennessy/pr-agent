# enum EDIT_TYPE (ADDED, DELETED, MODIFIED, RENAMED)
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

from pr_agent.algo.types import FilePatchInfo
from pr_agent.algo.utils import (Range, add_pr_review_identity,
                                 comment_matches_identity, process_description)
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

MAX_FILES_ALLOWED_FULL = 50

# Hidden marker that lets several PR-Agent runs keep separate persistent comments on the
# same PR - for example one review per model. Persistent comments are otherwise found by
# their visible header, which is identical for every run, so a second run would overwrite
# the first run's comment instead of publishing its own.
PERSISTENT_COMMENT_ID_MARKER = "<!-- pr-agent-persistent-id:"
# Opens the visible attribution line placed under the comment's heading.
PERSISTENT_COMMENT_ATTRIBUTION_PREFIX = "> Reviewed by"

_GLOBAL_SETTINGS_CACHE: dict = {}
_GLOBAL_SETTINGS_CACHE_TTL_SECONDS = 15 * 60
_GLOBAL_SETTINGS_CACHE_MAX_SIZE = 256
# Only cache reasonably-sized settings blobs; a valid .pr_agent.toml is tiny. This bounds the
# process-wide cache memory (256 entries x this) regardless of the much larger apply-time size cap.
_GLOBAL_SETTINGS_CACHE_MAX_VALUE_BYTES = 1024 * 1024


def get_cached_global_settings(cache_key, fetch_fn):
    """Return the org/group/workspace global .pr_agent.toml via a bounded TTL cache.

    Global settings change rarely, so caching avoids a provider API lookup (and repeated
    403/404 fallbacks) on every webhook event. Empty/"not found" results are cached too, to
    prevent repeated failed lookups. Oversized values are returned but not cached (to bound
    memory). Pass a falsy cache_key to bypass the cache.
    """
    def _fetch_safely():
        # A transient/unexpected fetch failure must NOT be cached, so it is retried instead of
        # disabling global settings for the whole TTL. fetch_fn returns "" for expected "not found".
        try:
            return fetch_fn(), True
        except Exception as e:
            get_logger().warning(f"Failed to load global settings, error: {e}")
            return "", False

    if not cache_key:
        return _fetch_safely()[0]
    now = time.monotonic()
    entry = _GLOBAL_SETTINGS_CACHE.get(cache_key)
    if entry is not None and entry[1] > now:
        return entry[0]
    value, cacheable = _fetch_safely()
    if not cacheable:
        return value
    value_size = len(value) if isinstance(value, (bytes, str)) else 0
    if value_size <= _GLOBAL_SETTINGS_CACHE_MAX_VALUE_BYTES:
        _GLOBAL_SETTINGS_CACHE[cache_key] = (value, now + _GLOBAL_SETTINGS_CACHE_TTL_SECONDS)
        while len(_GLOBAL_SETTINGS_CACHE) > _GLOBAL_SETTINGS_CACHE_MAX_SIZE:
            oldest_key = min(_GLOBAL_SETTINGS_CACHE, key=lambda k: _GLOBAL_SETTINGS_CACHE[k][1])
            _GLOBAL_SETTINGS_CACHE.pop(oldest_key, None)
    return value

def get_git_ssl_env() -> dict[str, str]:
    """
    Get git SSL configuration arguments for per-command use.
    This fixes SSL certificate issues when cloning repos with self-signed certificates.
    Returns the current environment with the addition of SSL config changes if any such SSL certificates exist.
    """
    ssl_cert_file = os.environ.get('SSL_CERT_FILE')
    requests_ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
    git_ssl_ca_info = os.environ.get('GIT_SSL_CAINFO')

    chosen_cert_file = ""

    # Try SSL_CERT_FILE first
    if ssl_cert_file:
        if os.path.exists(ssl_cert_file):
            if ((requests_ca_bundle and requests_ca_bundle != ssl_cert_file)
                    or (git_ssl_ca_info and git_ssl_ca_info != ssl_cert_file)):
                get_logger().warning(f"Found mismatch among: SSL_CERT_FILE, REQUESTS_CA_BUNDLE, GIT_SSL_CAINFO. "
                                     f"Using the SSL_CERT_FILE to resolve ambiguity.",
                                  artifact={"ssl_cert_file": ssl_cert_file, "requests_ca_bundle": requests_ca_bundle,
                                            'git_ssl_ca_info': git_ssl_ca_info})
            else:
                get_logger().info(f"Using SSL certificate bundle for git operations", artifact={"ssl_cert_file": ssl_cert_file})
            chosen_cert_file = ssl_cert_file
        else:
            get_logger().warning("SSL certificate bundle not found for git operations", artifact={"ssl_cert_file": ssl_cert_file})

    # Fallback to REQUESTS_CA_BUNDLE
    elif requests_ca_bundle:
        if os.path.exists(requests_ca_bundle):
            if (git_ssl_ca_info and git_ssl_ca_info != requests_ca_bundle):
                get_logger().warning(f"Found mismatch between: REQUESTS_CA_BUNDLE, GIT_SSL_CAINFO. "
                                     f"Using the REQUESTS_CA_BUNDLE to resolve ambiguity.",
                artifact = {"requests_ca_bundle": requests_ca_bundle, 'git_ssl_ca_info': git_ssl_ca_info})
            else:
                get_logger().info("Using SSL certificate bundle from REQUESTS_CA_BUNDLE for git operations",
                                  artifact={"requests_ca_bundle": requests_ca_bundle})
            chosen_cert_file = requests_ca_bundle
        else:
            get_logger().warning("requests CA bundle not found for git operations", artifact={"requests_ca_bundle": requests_ca_bundle})

    #Fallback to GIT CA:
    elif git_ssl_ca_info:
        if os.path.exists(git_ssl_ca_info):
            get_logger().info("Using git SSL CA info from GIT_SSL_CAINFO for git operations",
                              artifact={"git_ssl_ca_info": git_ssl_ca_info})
            chosen_cert_file = git_ssl_ca_info
        else:
            get_logger().warning("git SSL CA info not found for git operations", artifact={"git_ssl_ca_info": git_ssl_ca_info})

    else:
        get_logger().warning("Neither SSL_CERT_FILE nor REQUESTS_CA_BUNDLE nor GIT_SSL_CAINFO are defined, or they are defined but not found. Returning environment without SSL configuration")

    returned_env = os.environ.copy()
    if chosen_cert_file:
        returned_env.update({"GIT_SSL_CAINFO": chosen_cert_file, "REQUESTS_CA_BUNDLE": chosen_cert_file})
    return returned_env


def get_persistent_comment_id() -> str:
    """Return the configured identity for this run's persistent comments, or "".

    Set config.persistent_comment_id when more than one PR-Agent run comments on the same
    PR (for example a review per model, run as separate CI jobs). Each run then finds and
    updates only its own comment. Leave it unset for the single-run default.

    Returns:
        The trimmed id, or "" when unset.

    Example:
        >>> get_settings().set("config.persistent_comment_id", "kimi-k3")
        >>> get_persistent_comment_id()
        'kimi-k3'
    """
    try:
        value = get_settings().config.get("persistent_comment_id", "")
    except AttributeError:
        return ""
    return str(value).strip() if value else ""


def _persistent_comment_marker(comment_id: str) -> str:
    """Return the exact marker line published for the given id."""
    return f"{PERSISTENT_COMMENT_ID_MARKER} {comment_id} -->"


def _last_line(text: str) -> str:
    """Return the final non-empty line of a body, or "" - where the marker is written.

    Ownership is decided on the LAST line, never on a substring of the whole body. A
    review can quote a marker in its own text (a PR that edits the marker template will
    quote it verbatim in the diff), and a substring test then reads that quotation as
    proof of ownership - which drops the real marker and hands the comment to the wrong
    run. Anchoring to the final line makes quoted text inert.
    """
    stripped = (text or "").rstrip()
    if not stripped:
        return ""
    return stripped.rsplit("\n", 1)[-1].strip()


def _persistent_comment_attribution(comment_id: str) -> str:
    """Render the visible "who reviewed this" line for the current run.

    Names the model that actually answered, not just the configured id: with
    fallback_models a different model may have produced the text, and crediting the
    wrong one is worse than not crediting at all. When the answering model is the
    reviewer's own, the redundant half is dropped - "reviewer x, answered by x" is
    noise that makes the interesting case (a fallback) harder to spot.

    Args:
        comment_id: The configured reviewer identity.

    Returns:
        A single markdown line.

    Example:
        >>> _persistent_comment_attribution("kimi-k3")  # doctest: +SKIP
        '> Reviewed by `kimi-k3` - fallback model `openai/kimi-k2.7-code` answered'
    """
    try:
        model_used = get_settings().config.get("last_used_model", "") or comment_id
    except AttributeError:
        model_used = comment_id
    # Model ids carry a provider prefix ("openai/kimi-k3") that reviewer ids do not.
    is_own_model = str(model_used).rsplit("/", 1)[-1] == comment_id
    if is_own_model:
        return f"{PERSISTENT_COMMENT_ATTRIBUTION_PREFIX} `{comment_id}`"
    return (
        f"{PERSISTENT_COMMENT_ATTRIBUTION_PREFIX} `{comment_id}` - "
        f"fallback model `{model_used}` answered"
    )


def attach_persistent_comment_id(pr_comment: str) -> str:
    """Label a comment with this run's reviewer identity, visibly and invisibly.

    The visible attribution goes directly under the heading, because every reviewer
    publishes under the same "PR Reviewer Guide" heading: with three reviewers on one
    PR, an attribution at the bottom of a long comment means the reader cannot tell the
    opinions apart while scanning. The hidden marker stays the last line, where comment
    ownership and the CI publish guard both look for it.

    Args:
        pr_comment: The rendered comment body.

    Returns:
        The body unchanged when no id is configured; otherwise the body with an
        attribution line after its heading and the id marker as its final line.

    Example:
        >>> attach_persistent_comment_id("## PR Reviewer Guide")  # doctest: +SKIP
        '## PR Reviewer Guide\\n\\n> Reviewed by `kimi-k3`\\n<!-- pr-agent-persistent-id: kimi-k3 -->'
    """
    comment_id = get_persistent_comment_id()
    if not comment_id:
        return pr_comment
    marker = _persistent_comment_marker(comment_id)
    if _last_line(pr_comment) == marker:  # already attached; do not double-append
        return pr_comment

    attribution = _persistent_comment_attribution(comment_id)
    body = (pr_comment or "").rstrip()
    heading, _, rest = body.partition("\n")
    if heading.strip():
        rest = rest.lstrip("\n")
        # Replace a stale attribution rather than stacking a second one.
        first_rest_line, _, remainder = rest.partition("\n")
        if first_rest_line.startswith(PERSISTENT_COMMENT_ATTRIBUTION_PREFIX):
            rest = remainder.lstrip("\n")
        body = f"{heading}\n\n{attribution}\n\n{rest}".rstrip() if rest else f"{heading}\n\n{attribution}"
    else:
        body = attribution
    return f"{body}\n{marker}"


def is_own_persistent_comment(comment_body: str, initial_header: str) -> bool:
    """Decide whether an existing PR comment is the one this run should update.

    With an id configured, both this tool's header and this run's marker line must match,
    so parallel tools and reviewers never edit each other's comments. Without one, the
    historical header match applies, except that a comment belonging to an identified run
    is skipped: an un-identified run must not adopt another reviewer's comment as its own.

    Args:
        comment_body: The existing comment's body.
        initial_header: The header this run publishes under.

    Returns:
        True when this run owns the comment.

    Example:
        >>> is_own_persistent_comment("## PR Reviewer Guide 🔍\\nbody", "## PR Reviewer Guide 🔍")
        True
    """
    return is_own_persistent_comment_for_identities(comment_body, (initial_header,))


def is_own_persistent_comment_for_identities(comment_body: str, identities: Iterable[str]) -> bool:
    """Match a tool identity and, when configured, this run's persistent reviewer id."""
    body = comment_body or ""
    if not any(comment_matches_identity(body, identity) for identity in identities if identity):
        return False
    last_line = _last_line(body)
    comment_id = get_persistent_comment_id()
    if comment_id:
        return last_line == _persistent_comment_marker(comment_id)
    owned_by_an_identified_run = (
        last_line.startswith(PERSISTENT_COMMENT_ID_MARKER) and last_line.endswith("-->")
    )
    return not owned_by_an_identified_run


def _comment_body(comment) -> str:
    """Read comment text from provider objects and dictionary-shaped payloads."""
    if isinstance(comment, dict):
        return str(comment.get("body") or comment.get("comment") or "")
    return str(getattr(comment, "body", "") or "")


class GitProvider(ABC):
    @abstractmethod
    def is_supported(self, capability: str) -> bool:
        pass

    def supports_incremental_kind(self, kind: str) -> bool:
        """Whether `get_incremental_commits()` can scope an incremental run to `kind`
        (e.g. "suggestions" for `/improve -i`). Providers implementing kind-aware
        incremental anchoring override this; the default is no support, so tools
        fall back to a full run."""
        return False

    def supports_code_suggestions_artifact(self) -> bool:
        """Return whether `publish_code_suggestions()` writes a standalone output artifact."""
        return False

    def publish_code_suggestions_artifact(
            self, code_suggestions: list, artifact_footer: str = "",
            no_suggestions_message: str = "No code suggestions found for the PR.") -> bool:
        """Publish suggestions to a standalone artifact, optionally with additional context.

        Providers that return True from `supports_code_suggestions_artifact()` should override
        this method when they can preserve the footer in the same artifact. The default keeps
        backward compatibility for providers that only implement `publish_code_suggestions()`.
        """
        return self.publish_code_suggestions(code_suggestions)

    #Given a url (issues or PR/MR) - get the .git repo url to which they belong. Needs to be implemented by the provider.
    def get_git_repo_url(self, issues_or_pr_url: str) -> str:
        get_logger().warning("Not implemented! Returning empty url")
        return ""

    # Given a git repo url, return prefix and suffix of the provider in order to view a given file belonging to that repo. Needs to be implemented by the provider.
    # For example: For a git: https://git_provider.com/MY_PROJECT/MY_REPO.git and desired branch: <MY_BRANCH> then it should return ('https://git_provider.com/projects/MY_PROJECT/repos/MY_REPO/.../<MY_BRANCH>', '?=<SOME HEADER>')
    # so that to properly view the file: docs/readme.md -> <PREFIX>/docs/readme.md<SUFFIX> -> https://git_provider.com/projects/MY_PROJECT/repos/MY_REPO/<MY_BRANCH>/docs/readme.md?=<SOME HEADER>)
    def get_canonical_url_parts(self, repo_git_url:str, desired_branch:str) -> Tuple[str, str]:
        get_logger().warning("Not implemented! Returning empty prefix and suffix")
        return ("", "")


    #Clone related API
    #An object which ensures deletion of a cloned repo, once it becomes out of scope.
    # Example usage:
    #    with TemporaryDirectory() as tmp_dir:
    #            returned_obj: GitProvider.ScopedClonedRepo = self.git_provider.clone(self.repo_url, tmp_dir, remove_dest_folder=False)
    #            print(returned_obj.path) #Use returned_obj.path.
    #    #From this point, returned_obj.path may be deleted at any point and therefore must not be used.
    class ScopedClonedRepo(object):
        def __init__(self, dest_folder):
            self.path = dest_folder

        def __del__(self):
            if self.path and os.path.exists(self.path):
                shutil.rmtree(self.path, ignore_errors=True)

    #Method to allow implementors to manipulate the repo url to clone (such as embedding tokens in the url string). Needs to be implemented by the provider.
    def _prepare_clone_url_with_token(self, repo_url_to_clone: str) -> str | None:
        get_logger().warning("Not implemented! Returning None")
        return None

    # Does a shallow clone, using a forked process to support a timeout guard.
    # In case operation has failed, it is expected to throw an exception as this method does not return a value.
    def _clone_inner(self, repo_url: str, dest_folder: str, operation_timeout_in_seconds: int=None) -> None:
        #The following ought to be equivalent to:
        # #Repo.clone_from(repo_url, dest_folder)
        # , but with throwing an exception upon timeout.
        # Note: This can only be used in context that supports using pipes.
        try:
            ssl_env = get_git_ssl_env()
        except Exception as e:
            get_logger().exception(
                "Failed to prepare SSL environment for git operations, falling back to default env",
                artifact={"error": e}
            )
            ssl_env = os.environ.copy()

        subprocess.run([
            "git", "clone",
            "--filter=blob:none",
            "--depth", "1",
            repo_url, dest_folder
        ], env=ssl_env, check=True,  # check=True will raise an exception if the command fails
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=operation_timeout_in_seconds)

    CLONE_TIMEOUT_SEC = 20
    # Clone a given url to a destination folder. If successful, returns an object that wraps the destination folder,
    # deleting it once it is garbage collected. See: GitProvider.ScopedClonedRepo for more details.
    def clone(self, repo_url_to_clone: str, dest_folder: str, remove_dest_folder: bool = True,
              operation_timeout_in_seconds: int=CLONE_TIMEOUT_SEC) -> ScopedClonedRepo|None:
        returned_obj = None
        clone_url = self._prepare_clone_url_with_token(repo_url_to_clone)
        if not clone_url:
            get_logger().error("Clone failed: Unable to obtain url to clone.")
            return returned_obj
        try:
            if remove_dest_folder and os.path.exists(dest_folder) and os.path.isdir(dest_folder):
                shutil.rmtree(dest_folder)
            self._clone_inner(clone_url, dest_folder, operation_timeout_in_seconds)
            returned_obj = GitProvider.ScopedClonedRepo(dest_folder)
        except Exception as e:
            get_logger().exception(f"Clone failed: Could not clone url.",
                artifact={"error": str(e), "url": clone_url, "dest_folder": dest_folder})
        finally:
            return returned_obj

    @abstractmethod
    def get_files(self) -> list:
        pass

    @abstractmethod
    def get_diff_files(self) -> list[FilePatchInfo]:
        pass

    def get_incremental_commits(self, is_incremental):
        pass

    @abstractmethod
    def publish_description(self, pr_title: str, pr_body: str):
        # pr_title may be None, which means "leave the existing title unchanged"
        # and update only the description. Implementations must not write the
        # title in that case.
        pass

    @abstractmethod
    def publish_code_suggestions(self, code_suggestions: list) -> bool:
        pass

    @abstractmethod
    def get_languages(self):
        pass

    @abstractmethod
    def get_pr_branch(self):
        pass

    @abstractmethod
    def get_user_id(self):
        pass

    @abstractmethod
    def get_pr_description_full(self) -> str:
        pass

    def edit_comment(self, comment, body: str):
        pass

    def edit_comment_from_comment_id(self, comment_id: int, body: str):
        pass

    def get_comment_body_from_comment_id(self, comment_id: int) -> str:
        pass

    def reply_to_comment_from_comment_id(self, comment_id: int, body: str):
        pass

    def get_pr_description(self, full: bool = True, split_changes_walkthrough=False) -> str | tuple:
        from pr_agent.algo.utils import clip_tokens
        from pr_agent.config_loader import get_settings
        max_tokens_description = get_settings().get("CONFIG.MAX_DESCRIPTION_TOKENS", None)
        description = self.get_pr_description_full() if full else self.get_user_description()
        if split_changes_walkthrough:
            description, files = process_description(description)
            if max_tokens_description:
                description = clip_tokens(description, max_tokens_description)
            return description, files
        else:
            if max_tokens_description:
                description = clip_tokens(description, max_tokens_description)
            return description

    def get_user_description(self) -> str:
        if hasattr(self, 'user_description') and not (self.user_description is None):
            return self.user_description

        description = (self.get_pr_description_full() or "").strip()
        description_lowercase = description.lower()
        get_logger().debug(f"Existing description", description=description_lowercase)

        # if the existing description wasn't generated by the pr-agent, just return it as-is
        if not self._is_generated_by_pr_agent(description_lowercase):
            get_logger().info(f"Existing description was not generated by the pr-agent")
            self.user_description = description
            return description

        # if the existing description was generated by the pr-agent, but it doesn't contain a user description,
        # return nothing (empty string) because it means there is no user description
        user_description_header = "### **user description**"
        if user_description_header not in description_lowercase:
            get_logger().info(f"Existing description was generated by the pr-agent, but it doesn't contain a user description")
            return ""

        # otherwise, extract the original user description from the existing pr-agent description and return it
        # user_description_start_position = description_lowercase.find(user_description_header) + len(user_description_header)
        # return description[user_description_start_position:].split("\n", 1)[-1].strip()

        # the 'user description' is in the beginning. extract and return it
        possible_headers = self._possible_headers()
        start_position = description_lowercase.find(user_description_header) + len(user_description_header)
        end_position = len(description)
        for header in possible_headers: # try to clip at the next header
            if header != user_description_header and header in description_lowercase:
                end_position = min(end_position, description_lowercase.find(header))
        if end_position != len(description) and end_position > start_position:
            original_user_description = description[start_position:end_position].strip()
            if original_user_description.endswith("___"):
                original_user_description = original_user_description[:-3].strip()
        else:
            original_user_description = description.split("___")[0].strip()
            if original_user_description.lower().startswith(user_description_header):
                original_user_description = original_user_description[len(user_description_header):].strip()

        get_logger().info(f"Extracted user description from existing description",
                          description=original_user_description)
        self.user_description = original_user_description
        return original_user_description

    def _possible_headers(self):
        return ("### **user description**", "### **pr type**", "### **pr description**", "### **pr labels**", "### **type**", "### **description**",
                "### **labels**", "### 🤖 generated by pr agent")

    # Headers that are unique to pr-agent output; humans never write these
    # naturally, so they can safely be matched anywhere in the body.
    _UNIQUE_PR_AGENT_HEADERS = frozenset((
        "### **user description**",
        "### **pr type**",
        "### **pr description**",
        "### **pr labels**",
        "### \U0001f916 generated by pr agent",
    ))

    def _is_generated_by_pr_agent(self, description_lowercase: str) -> bool:
        # Fast path: hidden HTML comment injected at write time.
        if '<!-- pr-agent-generated -->' in description_lowercase:
            return True
        possible_headers = self._possible_headers()
        # For headers unique to pr-agent, tolerate leading content (the ratchet
        # case from #2633: a description like "Fixes #42\n### **PR Type**" should
        # still be recognized as pr-agent output).
        for header in possible_headers:
            # the header must start its own line: a quoted copy, e.g. inside a
            # blockquote or backticks, is somebody else's text, not ours
            if header in self._UNIQUE_PR_AGENT_HEADERS and (
                    description_lowercase.startswith(header)
                    or ("\n" + header) in description_lowercase):
                return True
        # For generic headers (type, description, labels), require startswith
        # to avoid misclassifying human descriptions that happen to contain
        # one of these common markdown headings.
        return any(description_lowercase.startswith(header) for header in possible_headers if header not in self._UNIQUE_PR_AGENT_HEADERS)

    @abstractmethod
    def get_repo_settings(self):
        pass

    def get_repo_file_content(self, file_path: str, from_default_branch: bool = False):
        return ""

    def get_workspace_name(self):
        return ""

    def get_pr_id(self):
        return ""

    def get_line_link(self, relevant_file: str, relevant_line_start: int, relevant_line_end: int = None) -> str:
        return ""

    def get_lines_link_original_file(self, filepath:str, component_range: Range) -> str:
        return ""

    #### comments operations ####
    @abstractmethod
    def publish_comment(self, pr_comment: str, is_temporary: bool = False):
        pass

    def should_publish_review_as_thread(self) -> bool:
        return False

    def supports_review_comment_identity(self) -> bool:
        return False

    def get_ci_failure_context(self) -> dict:
        """Return bounded failed-check metadata when the provider can supply it."""
        return {"status": "unavailable", "failures": []}

    def clear_persistent_review(self, identity_marker: str, name: str = "review") -> bool:
        """Remove the newest persistent review matching an exact tool identity."""
        try:
            comments = list(self.get_issue_comments())
            for comment in reversed(comments):
                if is_own_persistent_comment_for_identities(_comment_body(comment), (identity_marker,)):
                    self.remove_comment(comment)
                    return True
        except Exception as e:
            get_logger().exception(f"Failed to clear persistent {name}, error: {e}")
        return False

    def unresolve_comment_thread(self, comment):  # noqa: B027 - intentional no-op
        pass

    def resolve_comment_thread(self, comment_id) -> bool:  # noqa: B027 - intentional no-op
        return False

    def supports_thread_resolution(self) -> bool:
        """Providers that implement resolve_comment_thread override this."""
        return False

    def resolve_outdated_inline_threads(self):  # noqa: B027 - intentional no-op
        pass

    def publish_persistent_comment(self, pr_comment: str,
                                   initial_header: str,
                                   update_header: bool = True,
                                   name='review',
                                   final_update_message=True,
                                   as_thread: bool = False,
                                   identity_marker: str | None = None,
                                   legacy_initial_header: str | None = None):
        return self.publish_comment(pr_comment, **({'as_thread': True} if as_thread else {}))

    def publish_persistent_comment_full(self, pr_comment: str,
                                   initial_header: str,
                                   update_header: bool = True,
                                   name='review',
                                   final_update_message=True,
                                   as_thread: bool = False,
                                   identity_marker: str | None = None,
                                   legacy_initial_header: str | None = None):
        pr_comment = attach_persistent_comment_id(pr_comment)
        try:
            pr_comment = add_pr_review_identity(pr_comment, identity_marker)
            prev_comments = list(self.get_issue_comments())
            identifiers = (
                [identity_marker, legacy_initial_header]
                if identity_marker
                else [initial_header]
            )
            comment_to_update = next(
                (
                    comment
                    for identifier in identifiers
                    if identifier
                    for comment in prev_comments
                    if is_own_persistent_comment_for_identities(_comment_body(comment), (identifier,))
                ),
                None,
            )
            if comment_to_update is not None:
                comment = comment_to_update
                latest_commit_url = self.get_latest_commit_url()
                comment_url = self.get_comment_url(comment)
                if update_header:
                    update_message = f"#### ({name.capitalize()} updated until commit {latest_commit_url})\n"
                    update_anchor = identity_marker or initial_header
                    updated_anchor = f"{update_anchor}\n\n{update_message}"
                    pr_comment_updated = pr_comment.replace(update_anchor, updated_anchor, 1)
                else:
                    pr_comment_updated = pr_comment
                get_logger().info(f"Persistent mode - updating comment {comment_url} to latest {name} message")
                # response = self.mr.notes.update(comment.id, {'body': pr_comment_updated})
                self.edit_comment(comment, pr_comment_updated)
                if as_thread:
                    try:
                        # Reopen the thread if it was resolved, so the developer revisits the updated review.
                        self.unresolve_comment_thread(comment)
                    except Exception as e:
                        # The review was already updated in place; a reopen failure must not reach the
                        # outer except, whose fallback publish would duplicate the review.
                        get_logger().warning(f"Failed to reopen review thread: {e}")
                if final_update_message:
                    try:
                        return self.publish_comment(
                            f"**[Persistent {name}]({comment_url})** updated to latest commit {latest_commit_url}")
                    except Exception:
                        # The review was already updated in place; a notification failure must not reach
                        # the outer except, whose fallback publish would duplicate the review.
                        get_logger().opt(exception=True).warning(
                            "Failed to publish persistent review update message; review was already updated")
                        return comment
                return comment
        except Exception as e:
            get_logger().exception(f"Failed to update persistent review, error: {e}")
            pass
        return self.publish_comment(pr_comment, **({'as_thread': True} if as_thread else {}))

    @abstractmethod
    def publish_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str, original_suggestion=None):
        pass

    def create_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str,
                              absolute_position: int = None):
        raise NotImplementedError("This git provider does not support creating inline comments yet")

    @abstractmethod
    def publish_inline_comments(self, comments: list[dict]):
        pass

    @abstractmethod
    def remove_initial_comment(self):
        pass

    @abstractmethod
    def remove_comment(self, comment):
        pass

    @abstractmethod
    def get_issue_comments(self):
        pass

    def get_comment_url(self, comment) -> str:
        return ""

    def get_review_thread_comments(self, comment_id: int) -> list[dict]:
        pass

    #### labels operations ####
    @abstractmethod
    def publish_labels(self, labels):
        pass

    @abstractmethod
    def get_pr_labels(self, update=False):
        pass

    def get_repo_labels(self):
        pass

    @abstractmethod
    def add_eyes_reaction(self, issue_comment_id: int, disable_eyes: bool = False) -> Optional[int]:
        pass

    @abstractmethod
    def remove_reaction(self, issue_comment_id: int, reaction_id: int) -> bool:
        pass

    #### commits operations ####
    @abstractmethod
    def get_commit_messages(self):
        pass

    def get_pr_url(self) -> str:
        if hasattr(self, 'pr_url'):
            return self.pr_url
        return ""

    def get_latest_commit_url(self) -> str:
        return ""

    def auto_approve(self) -> bool:
        return False

    def calc_pr_statistics(self, pull_request_data: dict):
        return {}

    def get_num_of_files(self):
        try:
            return len(self.get_diff_files())
        except Exception as e:
            return -1

    def limit_output_characters(self, output: str, max_chars: int):
        return output[:max_chars] + '...' if len(output) > max_chars else output


def get_main_pr_language(languages, files) -> str:
    """
    Get the main language of the commit. Return an empty string if cannot determine.
    """
    main_language_str = ""
    if not languages:
        get_logger().info("No languages detected")
        return main_language_str
    if not files:
        get_logger().info("No files in diff")
        return main_language_str

    try:
        top_language = max(languages, key=languages.get).lower()

        # validate that the specific commit uses the main language
        extension_list = []
        for file in files:
            if not file:
                continue
            if isinstance(file, str):
                file = FilePatchInfo(base_file=None, head_file=None, patch=None, filename=file)
            extension_list.append(file.filename.rsplit('.')[-1])

        # get the most common extension
        most_common_extension = '.' + max(set(extension_list), key=extension_list.count)
        try:
            language_extension_map_org = get_settings().language_extension_map_org
            language_extension_map = {k.lower(): v for k, v in language_extension_map_org.items()}

            if top_language in language_extension_map and most_common_extension in language_extension_map[top_language]:
                main_language_str = top_language
            else:
                for language, extensions in language_extension_map.items():
                    if most_common_extension in extensions:
                        main_language_str = language
                        break
        except Exception as e:
            get_logger().exception(f"Failed to get main language: {e}")

        ## old approach:
        # most_common_extension = max(set(extension_list), key=extension_list.count)
        # if most_common_extension == 'py' and top_language == 'python' or \
        #         most_common_extension == 'js' and top_language == 'javascript' or \
        #         most_common_extension == 'ts' and top_language == 'typescript' or \
        #         most_common_extension == 'tsx' and top_language == 'typescript' or \
        #         most_common_extension == 'go' and top_language == 'go' or \
        #         most_common_extension == 'java' and top_language == 'java' or \
        #         most_common_extension == 'c' and top_language == 'c' or \
        #         most_common_extension == 'cpp' and top_language == 'c++' or \
        #         most_common_extension == 'cs' and top_language == 'c#' or \
        #         most_common_extension == 'swift' and top_language == 'swift' or \
        #         most_common_extension == 'php' and top_language == 'php' or \
        #         most_common_extension == 'rb' and top_language == 'ruby' or \
        #         most_common_extension == 'rs' and top_language == 'rust' or \
        #         most_common_extension == 'scala' and top_language == 'scala' or \
        #         most_common_extension == 'kt' and top_language == 'kotlin' or \
        #         most_common_extension == 'pl' and top_language == 'perl' or \
        #         most_common_extension == top_language:
        #     main_language_str = top_language

    except Exception as e:
        get_logger().exception(e)

    return main_language_str




class IncrementalPR:
    def __init__(self, is_incremental: bool = False):
        self.is_incremental = is_incremental
        self.commits_range = None
        self.first_new_commit = None
        self.last_seen_commit = None

    @property
    def first_new_commit_sha(self):
        return None if self.first_new_commit is None else self.first_new_commit.sha

    @property
    def last_seen_commit_sha(self):
        return None if self.last_seen_commit is None else self.last_seen_commit.sha
