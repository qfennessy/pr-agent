"""The local review records source-free telemetry when an operator enables it.

Issue #55. The journal, its reader, and the rollout gate that consumes it all
existed; nothing ever opened a writer, so the live-shadow gate could never start
collecting the seven days it needs.
"""

from types import SimpleNamespace

import pytest

from pr_agent.algo.checkpoint_evaluation import EvaluationRunState, MeasurementStatus
from pr_agent.algo.checkpoint_shadow_journal import (
    load_shadow_journal,
    shadow_entry_from_snapshot_result,
    shadow_journal_inventory_complete,
    shadow_journal_writer_from_settings,
)
from pr_agent.algo.review_snapshot import (
    ReviewEvent,
    ReviewResultState,
    ReviewSnapshot,
    ReviewSnapshotResult,
)


def _snapshot(**overrides) -> ReviewSnapshot:
    fields = dict(
        event=ReviewEvent.FILE_SAVE,
        repository_root="/tmp/repository",
        base_revision="base-revision",
        changed_paths=("src/example.py",),
        diff="diff --git a/src/example.py b/src/example.py\n+value = 1\n",
        policy_version="local-pair-review-v1",
        created_at="2026-09-05T12:00:00Z",
    )
    fields.update(overrides)
    return ReviewSnapshot(**fields)


def _result(snapshot, **overrides) -> ReviewSnapshotResult:
    fields = dict(
        snapshot_id=snapshot.snapshot_id,
        state=ReviewResultState.NO_FINDINGS,
        current_snapshot_id=snapshot.snapshot_id,
        review={"review": {"key_issues_to_review": []}},
        coverage_issues=(),
        latency_seconds=1.25,
        usage={"total_tokens": 1200},
        cost={"total_usd": 0.004},
    )
    fields.update(overrides)
    return ReviewSnapshotResult(**fields)


def _settings(enabled=True, path="", max_queue=256):
    return SimpleNamespace(
        checkpoint_evaluation=SimpleNamespace(
            shadow_journal_enabled=enabled,
            shadow_journal_path=path,
            shadow_journal_max_queue_entries=max_queue,
        )
    )


class TestWriterIsOffByDefault:
    def test_disabled_settings_open_nothing(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        writer = shadow_journal_writer_from_settings(
            _settings(enabled=False, path=str(target))
        )
        assert writer is None
        assert not target.exists()

    def test_an_empty_path_opens_nothing(self):
        assert shadow_journal_writer_from_settings(_settings(path="")) is None

    def test_a_settings_object_without_the_section_opens_nothing(self):
        assert shadow_journal_writer_from_settings(SimpleNamespace()) is None


class TestEntryContents:
    def test_a_clean_run_records_complete_coverage(self):
        snapshot = _snapshot()
        entry = shadow_entry_from_snapshot_result(snapshot, _result(snapshot))

        assert entry.snapshot_id == snapshot.snapshot_id
        assert entry.event is ReviewEvent.FILE_SAVE
        assert entry.result_state is EvaluationRunState.COMPLETED
        assert entry.coverage_status is MeasurementStatus.COMPLETE
        assert entry.latency_seconds.value == pytest.approx(1.25)
        assert entry.tokens.value == pytest.approx(1200)
        assert entry.cost_usd.value == pytest.approx(0.004)

    def test_a_cost_serialized_as_a_string_is_still_a_measurement(self):
        """The CLI serializes cost as an exact decimal string, not a float.

        Rejecting strings journals every priced run as having unknown cost, which
        makes the live-shadow cost metric unobtainable.
        """
        snapshot = _snapshot()
        result = _result(snapshot, cost={"total_usd": "0.0042", "status": "complete"})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.cost_usd.status is MeasurementStatus.COMPLETE
        assert entry.cost_usd.value == pytest.approx(0.0042)

    def test_an_unparseable_cost_stays_unavailable(self):
        snapshot = _snapshot()
        result = _result(snapshot, cost={"total_usd": "not-a-number"})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.cost_usd.status is MeasurementStatus.UNAVAILABLE
        assert entry.cost_usd.value is None

    def test_missing_telemetry_stays_unavailable_and_never_becomes_zero(self):
        snapshot = _snapshot()
        result = _result(snapshot, usage={}, cost={})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.tokens.status is MeasurementStatus.UNAVAILABLE
        assert entry.tokens.value is None
        assert entry.cost_usd.status is MeasurementStatus.UNAVAILABLE
        assert entry.cost_usd.value is None

    def test_a_stale_run_cannot_claim_complete_coverage(self):
        snapshot = _snapshot()
        result = _result(snapshot, state=ReviewResultState.STALE, review=None)
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.result_state is EvaluationRunState.STALE
        assert entry.coverage_status is not MeasurementStatus.COMPLETE

    def test_no_source_text_or_diff_can_reach_the_record(self):
        snapshot = _snapshot()
        entry = shadow_entry_from_snapshot_result(snapshot, _result(snapshot))
        rendered = repr(entry.to_dict())

        assert "value = 1" not in rendered
        assert "diff --git" not in rendered
        assert "/tmp/repository" not in rendered
        assert "src/example.py" not in rendered


class TestRoundTrip:
    def test_an_enabled_run_writes_a_record_the_reader_accepts(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        snapshot = _snapshot()
        writer = shadow_journal_writer_from_settings(_settings(path=str(target)))
        assert writer is not None

        writer.submit(shadow_entry_from_snapshot_result(snapshot, _result(snapshot)))
        # Never inside an assert: python -O strips those, and the writer would
        # then never flush or close.
        closed = writer.close()
        assert closed is True

        records = load_shadow_journal(target)
        assert len(records) == 1
        assert records[0].entry.snapshot_id == snapshot.snapshot_id
        assert shadow_journal_inventory_complete(records) is True

    def test_a_lineage_records_its_parent(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        parent = _snapshot()
        child = _snapshot(
            changed_paths=("src/example.py", "src/other.py"),
            parent_snapshot_id=parent.snapshot_id,
        )
        writer = shadow_journal_writer_from_settings(_settings(path=str(target)))
        writer.submit(shadow_entry_from_snapshot_result(parent, _result(parent)))
        writer.submit(shadow_entry_from_snapshot_result(child, _result(child)))
        closed = writer.close()
        assert closed is True

        records = load_shadow_journal(target)
        assert [record.entry.parent_snapshot_id for record in records] == [
            None,
            parent.snapshot_id,
        ]


class TestRecordingNeverBreaksTheReview:
    def test_a_failing_writer_does_not_raise_into_the_review(self, monkeypatch, tmp_path):
        from pr_agent import cli

        monkeypatch.setattr(
            cli, "_shadow_journal_writer", cli._SHADOW_JOURNAL_UNOPENED, raising=False
        )
        monkeypatch.setattr(
            cli,
            "shadow_journal_writer_from_settings",
            lambda _settings: (_ for _ in ()).throw(RuntimeError("disk on fire")),
        )
        snapshot = _snapshot()

        # The review must survive a recording failure without noticing it.
        cli._record_shadow_journal_entry(snapshot, _result(snapshot))
        cli._close_shadow_journal()

    def test_a_malformed_result_does_not_raise_into_the_review(self, monkeypatch):
        from pr_agent import cli

        monkeypatch.setattr(
            cli, "_shadow_journal_writer", cli._SHADOW_JOURNAL_UNOPENED, raising=False
        )
        cli._record_shadow_journal_entry(object(), object())
        cli._close_shadow_journal()
