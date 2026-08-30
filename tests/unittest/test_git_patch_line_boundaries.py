import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pr_agent.algo.git_patch_processing import (
    decouple_and_convert_to_hunks_with_lines_numbers,
    extend_patch,
    extract_hunk_lines_from_patch,
    iter_git_patch_lines,
    split_git_file_lines,
    strip_git_line_ending,
)
from pr_agent.algo.types import FilePatchInfo
from pr_agent.algo.utils import find_line_number_of_relevant_line_in_file, load_large_diff
from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
from pr_agent.git_providers.bitbucket_provider import BitbucketProvider, _split_git_diff_sections
from pr_agent.git_providers.gitea_provider import GiteaProvider
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.git_providers.gitlab_provider import GitLabProvider
from pr_agent.mosaico.diff_provider import parse_unified_diff
from pr_agent.tools.pr_code_suggestions import PRCodeSuggestions

UNICODE_SEPARATORS = ("\u0085", "\u2028", "\u2029")


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_shared_git_line_helpers_preserve_unicode_separator_content(separator):
    payload = f"+first{separator}+still-first\r\r\n-second{separator}-still-second\n"

    records = list(iter_git_patch_lines(payload))

    assert records == [f"+first{separator}+still-first\r\r\n", f"-second{separator}-still-second\n"]
    assert [strip_git_line_ending(record) for record in records] == [
        f"+first{separator}+still-first\r",
        f"-second{separator}-still-second",
    ]
    assert split_git_file_lines(payload) == [
        f"+first{separator}+still-first\r",
        f"-second{separator}-still-second",
    ]


def test_shared_git_line_helpers_preserve_bare_terminal_carriage_return():
    assert split_git_file_lines("content\r") == ["content\r"]


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_hunk_renderers_preserve_trailing_separator_and_meaningful_cr(separator):
    patch_text = f"@@ -0,0 +1 @@\r\n+payload{separator}\r\r\n"
    expected_content = f"+payload{separator}\r"

    rendered = decouple_and_convert_to_hunks_with_lines_numbers(
        patch_text,
        SimpleNamespace(filename="example.py"),
    )
    full_hunk, selected = extract_hunk_lines_from_patch(
        patch_text,
        "example.py",
        1,
        1,
        "right",
    )

    assert rendered.endswith(f"1 {expected_content}")
    assert full_hunk.endswith(expected_content)
    assert selected == expected_content


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_dynamic_context_indexes_source_with_git_line_boundaries(separator):
    original = f"header{separator}not-another-line\nold\n"
    patch_text = "@@ -2 +2 @@\n-old\n+new\n"

    extended = extend_patch(
        original,
        patch_text,
        patch_extra_lines_before=1,
        patch_extra_lines_after=0,
        filename="example.py",
    )

    assert f" header{separator}not-another-line\n-old\n+new" in extended


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_hosted_inline_locator_does_not_invent_hunk_or_line(separator):
    patch_text = (
        "@@ -0,0 +1,2 @@\n"
        f"+first{separator}@@ -90,0 +90,1 @@\n"
        "+second\n"
    )
    diff_file = FilePatchInfo("", "", patch_text, "example.py")

    position, line_number = find_line_number_of_relevant_line_in_file(
        [diff_file], "example.py", "+second"
    )

    assert (position, line_number) == (2, 2)


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_suggestion_range_does_not_accept_invented_line(separator):
    patch_text = f"@@ -0,0 +1 @@\n+first{separator}+not-a-second-line\n"

    assert PRCodeSuggestions._get_patch_range_lines(patch_text, 2, 2) is None
    assert PRCodeSuggestions._get_patch_range_lines(patch_text, 1, 1) == [
        f"first{separator}+not-a-second-line"
    ]


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_complete_head_suggestion_validation_uses_git_line_numbers(separator):
    tool = PRCodeSuggestions.__new__(PRCodeSuggestions)
    tool.git_provider = SimpleNamespace(
        diff_files=[
            FilePatchInfo("", f"first{separator}second\n", "", "example.py")
        ]
    )

    valid, reason, _ = tool._validate_suggestion("example.py", 2, 2, "second")

    assert valid is False
    assert reason == "the anchored range is outside the file"


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_mosaico_does_not_invent_file_or_hunk(separator):
    raw = (
        f"diff --git a/real{separator}name.py b/real{separator}name.py\r\n"
        "--- a/real.py\r\n"
        "+++ b/real.py\r\n"
        "@@ -0,0 +1 @@\r\n"
        f"+payload{separator}diff --git a/fake.py b/fake.py{separator}@@ -0,0 +1 @@\r\r\n"
    )

    files = parse_unified_diff(raw)

    assert len(files) == 1
    assert files[0].filename == f"real{separator}name.py"
    assert files[0].patch == raw
    assert files[0].head_file.endswith(f"fake.py{separator}@@ -0,0 +1 @@\r\r\n")


def _github_provider(raw_patch):
    raw_file = SimpleNamespace(
        filename="example.py",
        status="modified",
        patch=raw_patch,
        previous_filename=None,
    )
    provider = GithubProvider.__new__(GithubProvider)
    provider.diff_files = None
    provider.git_files = None
    provider.incremental = SimpleNamespace(is_incremental=False)
    provider.unreviewed_files_map = {}
    provider.pr = SimpleNamespace(
        base=SimpleNamespace(sha="base"),
        head=SimpleNamespace(sha="head"),
        get_files=lambda: [raw_file],
    )
    provider.repo_obj = SimpleNamespace(
        compare=lambda _base, _head: SimpleNamespace(
            merge_base_commit=SimpleNamespace(sha="base")
        )
    )
    provider._get_pr_file_content = lambda *_args, **_kwargs: "content"
    return provider


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_github_raw_counter_uses_only_lf_records(separator):
    raw_patch = f"@@ -1 +1 @@\n-old{separator}-fake\n+new{separator}+fake\n"
    provider = _github_provider(raw_patch)

    with patch("pr_agent.git_providers.github_provider.filter_ignored", side_effect=lambda files: files), \
         patch("pr_agent.git_providers.github_provider.is_valid_file", return_value=True):
        diff_file = provider.get_diff_files()[0]

    assert diff_file.num_plus_lines == 1
    assert diff_file.num_minus_lines == 1


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_github_hunk_ranges_ignore_embedded_header(separator):
    diff_file = FilePatchInfo(
        "", "", f"@@ -0,0 +1 @@\n+payload{separator}@@ -0,0 +90 @@\n", "example.py"
    )
    provider = GithubProvider.__new__(GithubProvider)
    provider.get_diff_files = lambda: [diff_file]

    provider.validate_comments_inside_hunks([{"relevant_file": "example.py"}])

    assert diff_file.patches_range == [{"start": 1, "end": 1}]


def _gitlab_provider(raw_patch):
    raw_change = {
        "new_path": "example.py",
        "old_path": "example.py",
        "diff": raw_patch,
        "new_file": False,
        "deleted_file": False,
        "renamed_file": False,
    }
    provider = GitLabProvider.__new__(GitLabProvider)
    provider.diff_files = None
    provider.incremental = None
    provider.unreviewed_files_map = {}
    provider.mr = SimpleNamespace(diff_refs={"base_sha": "base", "head_sha": "head"})
    provider.id_mr = 1
    provider._get_merge_request_changes = lambda: {"changes": [raw_change]}
    provider._expand_submodule_changes = lambda changes: changes
    provider.get_pr_file_content = lambda *_args, **_kwargs: "content"
    return provider


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_gitlab_raw_counter_and_anchor_use_only_lf_records(separator):
    raw_patch = f"@@ -1 +1 @@\n-old{separator}-fake\n+new{separator}+fake\n"
    provider = _gitlab_provider(raw_patch)

    with patch("pr_agent.git_providers.gitlab_provider.filter_ignored", side_effect=lambda files, _kind: files), \
         patch("pr_agent.git_providers.gitlab_provider.is_valid_file", return_value=True):
        diff_file = provider.get_diff_files()[0]

    assert diff_file.num_plus_lines == 1
    assert diff_file.num_minus_lines == 1

    provider.RE_HUNK_HEADER = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)"
    )
    _, found, _, _, target_line = provider.find_in_file(
        diff_file, f"+new{separator}+fake"
    )
    assert found is True
    assert target_line == 2


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_azure_raw_counter_uses_only_lf_records(separator):
    provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
    provider.diff_files = None
    provider.repo_slug = "repo"
    provider.workspace_slug = "workspace"
    provider.pr_num = 1
    provider.incremental = None
    provider.unreviewed_files_map = {}
    provider.pr = SimpleNamespace(
        last_merge_commit=SimpleNamespace(commit_id="head"),
        last_merge_target_commit=SimpleNamespace(commit_id="base"),
    )
    change = SimpleNamespace(
        additional_properties={"item": {"path": "example.py"}, "changeType": "edit"}
    )
    provider.azure_devops_client = MagicMock()
    provider.azure_devops_client.get_pull_request_iterations.return_value = [SimpleNamespace(id=1)]
    provider.azure_devops_client.get_pull_request_iteration_changes.return_value = SimpleNamespace(
        change_entries=[change]
    )
    provider.azure_devops_client.get_item.side_effect = [
        SimpleNamespace(content="new"),
        SimpleNamespace(content="old"),
    ]
    raw_patch = f"@@ -1 +1 @@\n-old{separator}-fake\n+new{separator}+fake\n"

    with patch("pr_agent.git_providers.azuredevops_provider.filter_ignored", side_effect=lambda files, _kind: files), \
         patch("pr_agent.git_providers.azuredevops_provider.is_valid_file", return_value=True), \
         patch("pr_agent.git_providers.azuredevops_provider.load_large_diff", return_value=raw_patch):
        diff_file = provider.get_diff_files()[0]

    assert diff_file.num_plus_lines == 1
    assert diff_file.num_minus_lines == 1


def _gitea_patch_map(raw_patch):
    provider = GiteaProvider.__new__(GiteaProvider)
    provider.logger = MagicMock()
    provider.owner = "owner"
    provider.repo = "repo"
    provider.pr_number = 1
    provider.file_diffs = {}
    provider.repo_api = MagicMock()
    provider.repo_api.get_pull_request_diff.return_value = raw_patch
    provider._GiteaProvider__add_file_diff()
    return provider.file_diffs


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_gitea_does_not_invent_file_or_hunk(separator):
    raw = (
        "diff --git a/real.py b/real.py\r\n"
        "--- a/real.py\r\n"
        "+++ b/real.py\r\n"
        "@@ -0,0 +1 @@\r\n"
        f"+payload{separator}diff --git a/fake.py b/fake.py{separator}@@ -0,0 +1 @@\r\r\n"
    )

    file_diffs = _gitea_patch_map(raw)

    assert list(file_diffs) == ["real.py"]
    assert file_diffs["real.py"] == raw[raw.index("@@"):]


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_bitbucket_file_split_requires_real_lf_header(separator):
    raw = (
        "diff --git a/real.py b/real.py\r\n"
        "@@ -0,0 +1 @@\r\n"
        f"+payload{separator}diff --git a/fake.py b/fake.py\r\r\n"
    )

    assert _split_git_diff_sections(raw) == [raw]


@pytest.mark.parametrize("separator", UNICODE_SEPARATORS)
def test_bitbucket_provider_keeps_embedded_file_header_in_one_patch(separator):
    class _Path:
        path = "real.py"

        @staticmethod
        def get_data(_key):
            return {}

    raw = (
        "diff --git a/real.py b/real.py\r\n"
        "index 1111111..2222222 100644\r\n"
        "--- a/real.py\r\n"
        "+++ b/real.py\r\n"
        "@@ -0,0 +1 @@\r\n"
        f"+payload{separator}diff --git a/fake.py b/fake.py\r\r\n"
    )
    raw_diff = SimpleNamespace(
        old=_Path(),
        new=_Path(),
        data={"status": "modified", "lines_added": 1, "lines_removed": 0},
    )
    provider = BitbucketProvider.__new__(BitbucketProvider)
    provider.diff_files = None
    provider.pr = SimpleNamespace(diffstat=lambda: [raw_diff], diff=lambda: raw)

    with patch("pr_agent.git_providers.bitbucket_provider.filter_ignored", side_effect=lambda files, _kind: files), \
         patch("pr_agent.git_providers.bitbucket_provider.is_valid_file", return_value=True):
        diff_files = provider.get_diff_files()

    assert len(diff_files) == 1
    assert diff_files[0].patch == raw[raw.index("@@"):]


def test_synthetic_diff_preserves_unicode_and_meaningful_carriage_returns():
    original = "old\u2028content\r\r\n"
    new = "new\u2028content\r\r\n"

    patch_text = load_large_diff("example.py", new, original)

    assert "-old\u2028content\r\r\n" in patch_text
    assert "+new\u2028content\r\r\n" in patch_text
