from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
from pr_agent.git_providers.git_provider import IncrementalPR


class TestAzureDevopsProviderRepoContext:
    def test_get_repo_file_content_reads_from_target_commit(self):
        # Repo-context files must be read from the PR target (base) commit, matching
        # the other providers.
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr = MagicMock()
        provider.pr.last_merge_target_commit.commit_id = "base-sha"
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_item.return_value = MagicMock(content="repo context")

        content = provider.get_repo_file_content("AGENTS.md")

        assert content == "repo context"
        _, kwargs = provider.azure_devops_client.get_item.call_args
        assert kwargs["path"] == "AGENTS.md"
        assert kwargs["repository_id"] == "my-repo"
        assert kwargs["project"] == "my-project"
        assert kwargs["version_descriptor"].version == "base-sha"
        assert kwargs["version_descriptor"].version_type == "commit"

    def test_get_repo_file_content_from_default_branch_omits_version(self):
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr = MagicMock()
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_item.return_value = MagicMock(content="repo context")

        content = provider.get_repo_file_content("AGENTS.md", from_default_branch=True)

        assert content == "repo context"
        _, kwargs = provider.azure_devops_client.get_item.call_args
        assert kwargs["version_descriptor"] is None  # no version -> default branch

    def test_get_repo_file_content_treats_missing_file_as_empty(self):
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr = MagicMock()
        provider.pr.last_merge_target_commit.commit_id = "base-sha"
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_item.side_effect = Exception("Operation returned a 404 status code.")

        assert provider.get_repo_file_content("MISSING.md") == ""

    def test_get_repo_file_content_propagates_non_404_errors(self):
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr = MagicMock()
        provider.pr.last_merge_target_commit.commit_id = "base-sha"
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_item.side_effect = Exception("Operation returned a 500 status code.")

        with pytest.raises(Exception, match="500 status code"):
            provider.get_repo_file_content("AGENTS.md")


class TestAzureDevopsProviderFiles:
    @staticmethod
    def _provider():
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr_num = 1
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_pull_request_commits.return_value = [SimpleNamespace(commit_id="m1")]
        return provider

    def test_get_files_full_skips_commits_without_changes(self):
        provider = self._provider()
        provider.azure_devops_client.get_pull_request_commits.return_value = [
            SimpleNamespace(commit_id="m1"),
            SimpleNamespace(commit_id="m2"),
        ]
        provider.azure_devops_client.get_changes.side_effect = [
            SimpleNamespace(changes=None),
            SimpleNamespace(changes=[{"item": {"path": "/src/app.py"}}]),
        ]

        assert provider._get_files_full() == ["/src/app.py"]

    def test_get_files_full_skips_changes_without_paths(self):
        provider = self._provider()
        provider.azure_devops_client.get_changes.return_value = SimpleNamespace(changes=[
            {},
            {"item": None},
            {"item": {"path": ""}},
            {"item": {"path": "/src/app.py"}},
        ])

        assert provider._get_files_full() == ["/src/app.py"]

    def test_get_files_full_supports_sdk_change_objects(self):
        provider = self._provider()
        provider.azure_devops_client.get_changes.return_value = SimpleNamespace(changes=[
            SimpleNamespace(item=SimpleNamespace(path="/src/sdk.py")),
        ])

        assert provider._get_files_full() == ["/src/sdk.py"]

    def test_get_files_full_skips_tree_entries(self):
        provider = self._provider()
        provider.azure_devops_client.get_changes.return_value = SimpleNamespace(changes=[
            {"item": {"path": "/src", "gitObjectType": "tree"}},
            {"item": {"path": "/src/app.py", "gitObjectType": "blob"}},
        ])

        assert provider._get_files_full() == ["/src/app.py"]

    def test_routing_inventory_uses_net_iteration_and_preserves_change_shapes(self):
        provider = self._provider()
        provider.incremental = None
        provider.unreviewed_files_map = {}
        provider._latest_pr_iteration_changes = None
        provider.azure_devops_client.get_pull_request_iterations.return_value = [
            SimpleNamespace(id=4)
        ]
        provider.azure_devops_client.get_pull_request_iteration_changes.return_value = (
            SimpleNamespace(
                change_entries=[
                    SimpleNamespace(additional_properties={
                        "item": {"path": "/docs/new.md"},
                        "changeType": "edit, rename",
                        "originalPath": "/services/auth/old.py",
                    }),
                    SimpleNamespace(additional_properties={
                        "item": {"path": "/docs/deleted.md"},
                        "changeType": "delete",
                    }),
                ],
                next_skip=0,
            )
        )

        files = provider.get_files_for_routing()

        assert [(file.filename, file.old_filename, file.edit_type) for file in files] == [
            ("/docs/new.md", "/services/auth/old.py", EDIT_TYPE.RENAMED),
            ("/docs/deleted.md", None, EDIT_TYPE.DELETED),
        ]
        assert provider.normalize_file_path_for_routing(files[0].filename) == "docs/new.md"
        provider.azure_devops_client.get_pull_request_commits.assert_not_called()
        provider.azure_devops_client.get_pull_request_iteration_changes.assert_called_once_with(
            repository_id="my-repo",
            pull_request_id=1,
            iteration_id=4,
            project="my-project",
            top=2000,
            skip=0,
            compare_to=0,
        )

    def test_incremental_routing_inventory_precedes_ignore_and_extension_filters(self):
        provider = self._provider()
        commit = SimpleNamespace(
            commit_id="head",
            sha="head",
            parents=[SimpleNamespace(commit_id="base")],
        )
        provider.pr_commits = [commit]
        provider.previous_review = None
        provider.get_previous_review = MagicMock(return_value=SimpleNamespace())
        provider._get_commit_range = MagicMock(return_value=[commit])
        provider.incremental = IncrementalPR(True)
        provider.unreviewed_files_map = {}
        provider._routing_incremental_files = None
        provider.azure_devops_client.get_changes.return_value = SimpleNamespace(changes=[
            {"item": {"path": "/docs/guide.md"}, "changeType": "edit"},
            {
                "item": {"path": "/generated/guard.md"},
                "changeType": "rename",
                "originalPath": "/services/auth/guard.py",
            },
            SimpleNamespace(
                item=SimpleNamespace(path="/services/auth/key.pem"),
                change_type="edit",
            ),
            {"item": {"path": "/docs/deleted.md"}, "changeType": "delete"},
            {"item": {"path": "/docs", "gitObjectType": "tree"}, "changeType": "edit"},
        ])

        with (
            patch(
                "pr_agent.git_providers.azuredevops_provider.filter_ignored",
                return_value=["/docs/guide.md", "/services/auth/key.pem", "/docs/deleted.md"],
            ),
            patch(
                "pr_agent.git_providers.azuredevops_provider.is_valid_file",
                side_effect=lambda path: path != "/services/auth/key.pem",
            ),
        ):
            provider._get_incremental_commits()

        assert provider.unreviewed_files_map == {
            "/docs/guide.md": "/docs/guide.md",
            "/docs/deleted.md": "/docs/deleted.md",
        }
        routing_files = provider.get_files_for_routing()
        assert [
            (file.filename, file.old_filename, file.edit_type)
            for file in routing_files
        ] == [
            ("/docs/guide.md", None, EDIT_TYPE.MODIFIED),
            ("/generated/guard.md", "/services/auth/guard.py", EDIT_TYPE.RENAMED),
            ("/services/auth/key.pem", None, EDIT_TYPE.MODIFIED),
            ("/docs/deleted.md", None, EDIT_TYPE.DELETED),
        ]

    @pytest.mark.parametrize("filtered_by", ["ignore", "extension"])
    def test_filtered_only_incremental_scope_is_known_empty(self, filtered_by):
        provider = self._provider()
        commit = SimpleNamespace(commit_id="head", sha="head", parents=[SimpleNamespace()])
        provider.pr_commits = [commit]
        provider.get_previous_review = MagicMock(return_value=SimpleNamespace())
        provider._get_commit_range = MagicMock(return_value=[commit])
        provider.incremental = IncrementalPR(True)
        provider.unreviewed_files_map = {}
        provider._routing_incremental_files = None
        provider.azure_devops_client.get_changes.return_value = SimpleNamespace(changes=[{
            "item": {"path": "/services/auth/key.pem"},
            "changeType": "edit",
        }])
        filtered = [] if filtered_by == "ignore" else ["/services/auth/key.pem"]

        with (
            patch(
                "pr_agent.git_providers.azuredevops_provider.filter_ignored",
                return_value=filtered,
            ),
            patch(
                "pr_agent.git_providers.azuredevops_provider.is_valid_file",
                return_value=False,
            ),
        ):
            provider._get_incremental_commits()

        assert provider.unreviewed_files_map == {}
        assert provider.is_incremental_scope_empty() is True
        assert [file.filename for file in provider.get_files_for_routing()] == [
            "/services/auth/key.pem"
        ]

    def test_incremental_routing_inventory_marks_partial_change_fetch_as_unknown(self):
        provider = self._provider()
        commits = [
            SimpleNamespace(commit_id="one", sha="one", parents=[SimpleNamespace()]),
            SimpleNamespace(commit_id="two", sha="two", parents=[SimpleNamespace()]),
        ]
        provider.pr_commits = commits
        provider.get_previous_review = MagicMock(return_value=SimpleNamespace())
        provider._get_commit_range = MagicMock(return_value=commits)
        provider.incremental = IncrementalPR(True)
        provider.unreviewed_files_map = {}
        provider._routing_incremental_files = None
        provider.azure_devops_client.get_changes.side_effect = [
            SimpleNamespace(changes=[{
                "item": {"path": "/docs/guide.md"},
                "changeType": "edit",
            }]),
            RuntimeError("page unavailable"),
        ]

        provider._get_incremental_commits()

        routing_files = provider.get_files_for_routing()
        assert routing_files[0].filename == "/docs/guide.md"
        assert routing_files[-1].filename == ""
        assert routing_files[-1].edit_type is EDIT_TYPE.UNKNOWN

    def test_incremental_routing_inventory_reads_all_mapping_and_sdk_pages(self):
        provider = self._provider()
        commit = SimpleNamespace(commit_id="head", sha="head", parents=[SimpleNamespace()])
        provider.pr_commits = [commit]
        provider.get_previous_review = MagicMock(return_value=SimpleNamespace())
        provider._get_commit_range = MagicMock(return_value=[commit])
        provider.incremental = IncrementalPR(True)
        provider.unreviewed_files_map = {}
        provider._routing_incremental_files = None
        docs_change = {"item": {"path": "/docs/guide.md"}, "changeType": "edit"}
        provider.azure_devops_client.get_changes.side_effect = [
            {"changes": [docs_change] * 2000},
            SimpleNamespace(changes=[{
                "item": {"path": "/services/auth/guard.py"},
                "changeType": "edit",
            }]),
        ]

        provider._get_incremental_commits()

        assert [file.filename for file in provider.get_files_for_routing()] == [
            "/docs/guide.md",
            "/services/auth/guard.py",
        ]
        assert provider.unreviewed_files_map == {
            "/docs/guide.md": "/docs/guide.md",
            "/services/auth/guard.py": "/services/auth/guard.py",
        }
        assert provider.azure_devops_client.get_changes.call_args_list == [
            call(
                project="my-project",
                repository_id="my-repo",
                commit_id="head",
                top=2000,
                skip=0,
            ),
            call(
                project="my-project",
                repository_id="my-repo",
                commit_id="head",
                top=2000,
                skip=2000,
            ),
        ]

    def test_incremental_routing_inventory_retains_first_page_on_later_failure(self):
        provider = self._provider()
        commits = [
            SimpleNamespace(commit_id="docs", sha="docs", parents=[SimpleNamespace()]),
            SimpleNamespace(commit_id="head", sha="head", parents=[SimpleNamespace()]),
        ]
        provider.pr_commits = commits
        provider.get_previous_review = MagicMock(return_value=SimpleNamespace())
        provider._get_commit_range = MagicMock(return_value=commits)
        provider.incremental = IncrementalPR(True)
        provider.unreviewed_files_map = {}
        provider._routing_incremental_files = None
        rename_change = {
            "item": {"path": "/generated/guard.md"},
            "changeType": "rename",
            "originalPath": "/services/auth/guard.py",
        }
        delete_change = {
            "item": {"path": "/services/auth/deleted.py"},
            "changeType": "delete",
        }
        filler_change = {"item": {"path": "/docs/filler.md"}, "changeType": "edit"}
        provider.azure_devops_client.get_changes.side_effect = [
            {"changes": [{"item": {"path": "/docs/guide.md"}, "changeType": "edit"}]},
            {"changes": [rename_change, delete_change, *([filler_change] * 1998)]},
            RuntimeError("later page unavailable"),
        ]

        with patch(
            "pr_agent.git_providers.azuredevops_provider.filter_ignored",
            return_value=[],
        ):
            provider._get_incremental_commits()

        routing_files = provider.get_files_for_routing()
        assert [
            (file.filename, file.old_filename, file.edit_type)
            for file in routing_files
        ] == [
            ("/docs/guide.md", None, EDIT_TYPE.MODIFIED),
            ("/generated/guard.md", "/services/auth/guard.py", EDIT_TYPE.RENAMED),
            ("/services/auth/deleted.py", None, EDIT_TYPE.DELETED),
            ("/docs/filler.md", None, EDIT_TYPE.MODIFIED),
            ("", None, EDIT_TYPE.UNKNOWN),
        ]
        assert provider.get_files_for_routing() == routing_files
        assert provider.azure_devops_client.get_changes.call_args_list == [
            call(
                project="my-project",
                repository_id="my-repo",
                commit_id="docs",
                top=2000,
                skip=0,
            ),
            call(
                project="my-project",
                repository_id="my-repo",
                commit_id="head",
                top=2000,
                skip=0,
            ),
            call(
                project="my-project",
                repository_id="my-repo",
                commit_id="head",
                top=2000,
                skip=2000,
            ),
        ]
        assert provider.unreviewed_files_map == {}

    def test_routing_normalization_only_strips_one_documented_provider_root(self):
        provider = self._provider()

        assert provider.normalize_file_path_for_routing("/docs/guide.md") == "docs/guide.md"
        assert provider.normalize_file_path_for_routing("docs/guide.md") == "docs/guide.md"
        assert provider.normalize_file_path_for_routing("//outside") == "/outside"

    def test_routing_inventory_retains_malformed_blob_but_ignores_tree(self):
        provider = self._provider()
        provider.incremental = None
        provider.unreviewed_files_map = {}
        provider._latest_pr_iteration_changes = (
            SimpleNamespace(additional_properties={
                "item": {"path": "/docs/guide.md", "gitObjectType": "blob"},
                "changeType": "edit",
            }),
            SimpleNamespace(additional_properties={
                "item": {"gitObjectType": "blob"},
                "changeType": "edit",
            }),
            SimpleNamespace(additional_properties={
                "item": {"path": "/docs", "gitObjectType": "tree"},
                "changeType": "edit",
            }),
        )

        files = provider.get_files_for_routing()

        assert [(file.filename, file.edit_type) for file in files] == [
            ("/docs/guide.md", EDIT_TYPE.MODIFIED),
            ("", EDIT_TYPE.UNKNOWN),
        ]

    def test_detailed_net_rename_keeps_original_path_for_routing_reconciliation(self):
        provider = self._provider()
        provider.diff_files = None
        provider._diff_path_map = None
        provider.incremental = None
        provider.unreviewed_files_map = {}
        provider.pr = SimpleNamespace(
            last_merge_commit=SimpleNamespace(commit_id="head"),
            last_merge_target_commit=SimpleNamespace(commit_id="base"),
        )
        provider._latest_pr_iteration_changes = (
            SimpleNamespace(additional_properties={
                "item": {"path": "/docs/new.md"},
                "changeType": "rename",
                "originalPath": "/services/auth/old.py",
            }),
        )
        provider.azure_devops_client.get_item.return_value = SimpleNamespace(content="new")

        with patch(
            "pr_agent.git_providers.azuredevops_provider.filter_ignored",
            side_effect=lambda files, _kind: files,
        ), patch(
            "pr_agent.git_providers.azuredevops_provider.is_valid_file",
            return_value=True,
        ), patch(
            "pr_agent.git_providers.azuredevops_provider.load_large_diff",
            return_value="@@ -0,0 +1 @@\n+new\n",
        ):
            diff_file = provider.get_diff_files()[0]

        assert diff_file.edit_type is EDIT_TYPE.RENAMED
        assert diff_file.filename == "/docs/new.md"
        assert diff_file.old_filename == "/services/auth/old.py"


def _provider_with_diff(*filenames):
    provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
    provider.repo_slug = "my-repo"
    provider.workspace_slug = "my-project"
    provider.pr_num = 1
    provider.temp_comments = []
    provider.azure_devops_client = MagicMock()
    provider.diff_files = [
        FilePatchInfo(
            base_file="",
            head_file="\n".join(f"line {line}" for line in range(1, 13)),
            patch="",
            filename=filename,
        )
        for filename in filenames
    ]
    return provider


def _created_threads(provider):
    return [kwargs["comment_thread"] for _, kwargs in provider.azure_devops_client.create_thread.call_args_list]


def _suggestion(relevant_file):
    return {
        "body": "```suggestion\nfixed\n```",
        "relevant_file": relevant_file,
        "relevant_lines_start": 10,
        "relevant_lines_end": 12,
    }


class TestAzureDevopsProviderSuggestionAnchoring:
    def test_suggestion_without_leading_slash_is_published_with_the_diff_path(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        assert provider.publish_code_suggestions([_suggestion("src/Api/Controllers/SomeController.cs")]) is True

        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context.file_path == "/src/Api/Controllers/SomeController.cs"
        assert threads[0].comments[0].content == _suggestion("/src/Api/Controllers/SomeController.cs")["body"]
        assert threads[0].thread_context.right_file_start.line == 10
        assert threads[0].thread_context.right_file_end.line == 12

    def test_suggestion_span_covers_the_complete_final_line(self):
        provider = _provider_with_diff("/src/app.py")
        provider.diff_files[0].head_file = "\n".join([
            *(f"line {line}" for line in range(1, 10)),
            "    if ready:",
            "        run()",
            "    }",
        ])

        provider.publish_code_suggestions([_suggestion("/src/app.py")])

        context = _created_threads(provider)[0].thread_context
        assert context.right_file_start.offset == 1
        assert context.right_file_end.offset == 6

    def test_suggestion_end_offset_uses_utf16_code_units(self):
        provider = _provider_with_diff("/src/app.py")
        provider.diff_files[0].head_file = "\n".join([
            *(f"line {line}" for line in range(1, 12)),
            "return '😀'",
        ])

        provider.publish_code_suggestions([_suggestion("/src/app.py")])

        context = _created_threads(provider)[0].thread_context
        assert context.right_file_end.offset == 12

    def test_suggestion_with_unavailable_final_line_becomes_a_pr_level_comment(self):
        provider = _provider_with_diff("/src/app.py")
        provider.diff_files[0].head_file = "line 1"

        provider.publish_code_suggestions([_suggestion("/src/app.py")])

        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context is None
        assert "could not resolve the complete line range" in threads[0].comments[0].content

    def test_unavailable_final_line_does_not_stop_the_batch(self):
        provider = _provider_with_diff("/src/short.py", "/src/complete.py")
        provider.diff_files[0].head_file = "line 1"

        provider.publish_code_suggestions([
            _suggestion("/src/short.py"),
            _suggestion("/src/complete.py"),
        ])

        threads = _created_threads(provider)
        anchored = [thread for thread in threads if thread.thread_context is not None]
        assert len(anchored) == 1
        assert anchored[0].thread_context.file_path == "/src/complete.py"

    def test_unavailable_final_line_respects_disabled_fallback(self):
        provider = _provider_with_diff("/src/app.py")
        provider.diff_files[0].head_file = "line 1"
        suggestion = _suggestion("/src/app.py")
        suggestion["fallback_to_pr_comment"] = False

        assert provider.publish_code_suggestions([suggestion]) is False
        provider.azure_devops_client.create_thread.assert_not_called()

    def test_regular_inline_finding_keeps_its_existing_character_anchor(self):
        provider = _provider_with_diff("/src/app.py")
        finding = _suggestion("/src/app.py")
        finding["body"] = "Review finding"

        provider.publish_code_suggestions([finding])

        context = _created_threads(provider)[0].thread_context
        assert context.right_file_start.offset == 1
        assert context.right_file_end.offset == 1

    def test_suggestion_with_matching_path_is_published_unchanged(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/SomeController.cs")])

        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context.file_path == "/src/Api/Controllers/SomeController.cs"

    def test_suggestion_with_extra_leading_slash_is_published_with_the_diff_path(self):
        provider = _provider_with_diff("src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/SomeController.cs")])

        assert _created_threads(provider)[0].thread_context.file_path == "src/Api/Controllers/SomeController.cs"

    def test_suggestion_with_padded_backticks_is_published_with_the_diff_path(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([_suggestion("` src/Api/Controllers/SomeController.cs `")])

        assert _created_threads(provider)[0].thread_context.file_path == "/src/Api/Controllers/SomeController.cs"

    def test_unmatched_suggestion_becomes_a_pr_level_comment_instead_of_an_orphaned_thread(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/Removed.cs")])

        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context is None
        body = threads[0].comments[0].content
        assert "/src/Api/Controllers/Removed.cs" in body
        assert "fixed" in body

    def test_unmatched_suggestions_are_published_in_one_comment(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/First.cs"),
            _suggestion("/src/Api/Controllers/Second.cs"),
        ])

        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context is None
        body = threads[0].comments[0].content
        assert "/src/Api/Controllers/First.cs" in body
        assert "/src/Api/Controllers/Second.cs" in body

    def test_diff_path_index_is_reused_for_a_batch(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.get_diff_files = MagicMock(return_value=provider.diff_files)

        provider.publish_code_suggestions([
            _suggestion("src/Api/Controllers/SomeController.cs"),
            _suggestion("/src/Api/Controllers/SomeController.cs"),
        ])

        provider.get_diff_files.assert_called_once_with()

    def test_transient_diff_failure_does_not_cache_an_empty_path_index(self):
        provider = _provider_with_diff()
        provider.diff_files = None
        diff_file = FilePatchInfo(
            base_file="",
            head_file="",
            patch="",
            filename="/src/Api/Controllers/SomeController.cs",
        )
        responses = iter([None, [diff_file]])

        def load_diff_files():
            provider.diff_files = next(responses)
            return provider.diff_files or []

        provider.get_diff_files = MagicMock(side_effect=load_diff_files)

        assert provider._resolve_diff_file_path("src/Api/Controllers/SomeController.cs") is None
        assert provider._resolve_diff_file_path("src/Api/Controllers/SomeController.cs") == diff_file.filename
        assert provider.get_diff_files.call_count == 2

    def test_incremental_mode_invalidates_the_diff_path_index(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider._diff_path_map = {"stale.cs": "/stale.cs"}
        provider._get_incremental_commits = MagicMock()
        incremental = MagicMock()
        incremental.is_incremental = True

        provider.get_incremental_commits(incremental)

        assert provider.diff_files is None
        assert provider._diff_path_map is None

    def test_set_pr_invalidates_the_diff_path_index(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider._diff_path_map = {"stale.cs": "/stale.cs"}
        provider._parse_pr_url = MagicMock(return_value=("project", "repo", 2))
        provider._get_pr = MagicMock(return_value=MagicMock())

        provider.set_pr("https://dev.azure.com/example/project/_git/repo/pullrequest/2")

        assert provider.diff_files is None
        assert provider._diff_path_map is None

    def test_unmatched_suggestion_path_does_not_break_markdown(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([_suggestion("` /src/Api/Controllers/Removed.cs `")])

        body = _created_threads(provider)[-1].comments[0].content
        assert body.startswith("`/src/Api/Controllers/Removed.cs` (lines 10-12)")

    def test_aggregate_fallback_retries_suggestions_individually(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = [RuntimeError("request failed"),
                                                                  MagicMock(), MagicMock()]

        result = provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/First.cs"),
            _suggestion("/src/Api/Controllers/Second.cs"),
        ])

        assert result is True
        assert provider.azure_devops_client.create_thread.call_count == 3

    def test_unanchored_publish_failure_is_reported(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = RuntimeError("request failed")

        assert provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/Removed.cs")]) is False

    def test_anchored_publish_failure_is_reported(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = RuntimeError("request failed")

        assert provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/SomeController.cs")]) is False

    def test_disabled_fallback_does_not_retry_a_failed_suggestion(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = RuntimeError("request failed")
        suggestion = _suggestion("/src/Api/Controllers/SomeController.cs")
        suggestion["fallback_to_pr_comment"] = False

        assert provider.publish_code_suggestions([suggestion]) is False
        assert provider.azure_devops_client.create_thread.call_count == 1

    def test_anchored_publish_failure_uses_the_publish_failure_reason(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = [RuntimeError("request failed"), MagicMock()]

        provider.publish_code_suggestions([_suggestion("/src/Api/Controllers/SomeController.cs")])

        fallback_body = _created_threads(provider)[-1].comments[0].content
        assert "could not be published as an inline comment" in fallback_body

    def test_malformed_suggestion_does_not_stop_the_batch(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([
            {"body": "missing location"},
            _suggestion("/src/Api/Controllers/SomeController.cs"),
        ])

        assert len(_created_threads(provider)) == 1

    @pytest.mark.parametrize("overrides", [
        {"relevant_file": 123},
        {"relevant_file": " "},
        {"relevant_file": "``"},
        {"body": None},
        {"relevant_lines_start": "10"},
        {"relevant_lines_start": True},
        {"relevant_lines_start": -2},
        {"relevant_lines_end": None},
    ])
    def test_invalid_values_do_not_stop_the_batch(self, overrides):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        malformed = _suggestion("/src/Api/Controllers/SomeController.cs")
        malformed.update(overrides)

        result = provider.publish_code_suggestions([
            malformed,
            _suggestion("/src/Api/Controllers/SomeController.cs"),
        ])

        assert result is True
        threads = _created_threads(provider)
        assert len(threads) == 1
        assert threads[0].thread_context.file_path == "/src/Api/Controllers/SomeController.cs"

    def test_diff_path_resolver_rejects_non_string_paths(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        assert provider._resolve_diff_file_path(123) is None

    def test_invalid_range_does_not_retry_successful_suggestions(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        invalid = _suggestion("/src/Api/Controllers/SomeController.cs")
        invalid["relevant_lines_start"] = -1

        result = provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/SomeController.cs"),
            invalid,
        ])

        assert result is True
        assert len(_created_threads(provider)) == 1

    def test_reversed_range_does_not_retry_successful_suggestions(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        invalid = _suggestion("/src/Api/Controllers/SomeController.cs")
        invalid["relevant_lines_start"] = 12
        invalid["relevant_lines_end"] = 10

        result = provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/SomeController.cs"),
            invalid,
        ])

        assert result is True
        assert len(_created_threads(provider)) == 1

    def test_partial_publish_failure_does_not_retry_successful_suggestions(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.azure_devops_client.create_thread.side_effect = [MagicMock(), RuntimeError("request failed"),
                                                                  RuntimeError("request failed"),
                                                                  RuntimeError("request failed")]

        result = provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/SomeController.cs"),
            _suggestion("src/Api/Controllers/SomeController.cs"),
        ])

        assert result is True

    def test_unmatched_suggestion_does_not_stop_the_remaining_suggestions(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_code_suggestions([
            _suggestion("/src/Api/Controllers/Removed.cs"),
            _suggestion("src/Api/Controllers/SomeController.cs"),
        ])

        anchored = [t for t in _created_threads(provider) if t.thread_context is not None]
        assert len(anchored) == 1
        assert anchored[0].thread_context.file_path == "/src/Api/Controllers/SomeController.cs"


class TestAzureDevopsProviderCreateInlineComment:
    def test_resolved_line_comment_uses_the_diff_path(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")
        provider.diff_files[0].patch = "@@ -1,3 +1,4 @@\n context\n+    var x = 1;\n"
        provider.diff_files[0].head_file = " context\n    var x = 1;\n"

        comment = provider.create_inline_comment("body", "src/Api/Controllers/SomeController.cs", "    var x = 1;")

        assert comment["path"] == "/src/Api/Controllers/SomeController.cs"
        assert comment["subject_type"] == "LINE"

    def test_unresolved_line_returns_a_file_level_comment_instead_of_an_empty_dict(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        comment = provider.create_inline_comment("body", "src/Api/Controllers/SomeController.cs", "no such line")

        assert comment
        assert comment["subject_type"] == "FILE"
        assert comment["path"] == "/src/Api/Controllers/SomeController.cs"

    def test_file_level_comment_is_published_without_a_line_anchor(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_inline_comment("body", "src/Api/Controllers/SomeController.cs", "no such line")

        thread_context = _created_threads(provider)[0].thread_context
        assert thread_context == {"filePath": "/src/Api/Controllers/SomeController.cs"}

    def test_comment_on_a_file_outside_the_diff_becomes_a_pr_level_comment(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_inline_comment("body", "src/Api/Controllers/Removed.cs", "no such line")

        thread = _created_threads(provider)[0]
        assert thread.thread_context is None
        assert "src/Api/Controllers/Removed.cs" in thread.comments[0].content
        assert "body" in thread.comments[0].content

    def test_pr_level_fallback_removes_backticks_from_the_display_path(self):
        provider = _provider_with_diff("/src/Api/Controllers/SomeController.cs")

        provider.publish_inline_comment("body", "src/Api/Controllers`Removed.cs", "no such line")

        body = _created_threads(provider)[0].comments[0].content
        assert body.startswith("`src/Api/ControllersRemoved.cs`")


class TestAzureDevopsProviderInlineComments:
    @staticmethod
    def _provider(threads):
        provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
        provider.repo_slug = "my-repo"
        provider.workspace_slug = "my-project"
        provider.pr_num = 42
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_threads.return_value = threads
        provider.diff_files = [FilePatchInfo(base_file="", head_file="", patch="", filename="/app.py")]
        return provider

    def test_get_inline_comment_bodies_only_returns_line_threads(self):
        line_thread = SimpleNamespace(
            thread_context=SimpleNamespace(file_path="/app.py", right_file_start=SimpleNamespace(line=3)),
            comments=[SimpleNamespace(content="line finding")],
        )
        file_thread = SimpleNamespace(
            thread_context=SimpleNamespace(file_path="/app.py", right_file_start=None),
            comments=[SimpleNamespace(content="file finding")],
        )
        pr_thread = SimpleNamespace(
            thread_context=None,
            comments=[SimpleNamespace(content="PR finding")],
        )
        provider = self._provider([line_thread, file_thread, pr_thread])

        assert provider.get_inline_comment_bodies() == ["line finding"]
        provider.azure_devops_client.get_threads.assert_called_once_with(
            repository_id="my-repo",
            pull_request_id=42,
            project="my-project",
        )

    def test_get_inline_comment_bodies_supports_serialized_context(self):
        thread = SimpleNamespace(
            thread_context={"filePath": "/app.py", "rightFileStart": {"line": 3, "offset": 1}},
            comments=[SimpleNamespace(content="line finding"), SimpleNamespace(content="")],
        )

        assert self._provider([thread]).get_inline_comment_bodies() == ["line finding"]

    def test_get_inline_comment_bodies_includes_recent_successful_posts(self):
        provider = self._provider([])
        provider.publish_code_suggestions([{
            "body": "line finding",
            "relevant_file": "/app.py",
            "relevant_lines_start": 3,
            "relevant_lines_end": 3,
        }])

        assert provider.get_inline_comment_bodies() == ["line finding"]

    def test_set_pr_clears_inline_comment_state(self):
        provider = self._provider([])
        provider._published_inline_comment_bodies = ["old finding"]
        provider._inline_comment_store = MagicMock()
        provider._parse_pr_url = MagicMock(return_value=("new-project", "new-repo", 43))
        provider._get_pr = MagicMock(return_value=MagicMock())

        provider.set_pr("https://dev.azure.com/example/new-project/_git/new-repo/pullrequest/43")

        assert provider._published_inline_comment_bodies == []
        assert provider._inline_comment_store is None

    def test_recent_inline_comment_bodies_returns_a_copy(self):
        provider = self._provider([])
        provider._published_inline_comment_bodies = ["line finding"]

        bodies = provider.get_recent_inline_comment_bodies()
        bodies.append("other finding")

        assert provider.get_recent_inline_comment_bodies() == ["line finding"]
