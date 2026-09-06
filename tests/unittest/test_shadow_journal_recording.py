"""The local review records source-free telemetry when an operator enables it.

Issue #55. The journal, its reader, and the rollout gate that consumes it all
existed; nothing ever opened a writer, so the live-shadow gate could never start
collecting the seven days it needs.
"""

from types import SimpleNamespace

import pytest

from pr_agent.algo.checkpoint_evaluation import (
    EvaluationRunState,
    EvaluationValidationError,
    MeasurementStatus,
)
from pr_agent.algo.checkpoint_shadow_journal import (
    load_shadow_journal,
    record_shadow_journal_drop,
    shadow_entry_from_snapshot_result,
    shadow_journal_drop_marker_path,
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


class TestPartialCostIsNotCalledComplete:
    """An underestimate that claims completeness makes a cost gate easier to pass."""

    def test_a_partial_cost_stays_partial(self):
        snapshot = _snapshot()
        result = _result(snapshot, cost={"total_usd": "0.0042", "status": "partial"})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.cost_usd.status is MeasurementStatus.PARTIAL
        assert entry.cost_usd.value == pytest.approx(0.0042)

    def test_a_cost_reported_unavailable_keeps_no_value(self):
        snapshot = _snapshot()
        result = _result(snapshot, cost={"total_usd": "0.0042", "status": "unavailable"})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.cost_usd.status is MeasurementStatus.UNAVAILABLE
        assert entry.cost_usd.value is None

    def test_an_unrecognised_verdict_is_not_a_reason_for_confidence(self):
        snapshot = _snapshot()
        result = _result(snapshot, cost={"total_usd": "0.0042", "status": "who knows"})
        entry = shadow_entry_from_snapshot_result(snapshot, result)

        assert entry.cost_usd.status is MeasurementStatus.PARTIAL


class TestALostReviewIsVisible:
    """A review that never reached the journal must not leave it looking complete."""

    def test_a_concurrent_writer_cannot_open_the_same_journal(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        first = shadow_journal_writer_from_settings(_settings(path=str(target)))
        assert first is not None

        with pytest.raises(Exception):
            shadow_journal_writer_from_settings(_settings(path=str(target)))

        first.close()

    def test_a_drop_marker_makes_the_inventory_incomplete(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        snapshot = _snapshot()
        writer = shadow_journal_writer_from_settings(_settings(path=str(target)))
        writer.submit(shadow_entry_from_snapshot_result(snapshot, _result(snapshot)))
        closed = writer.close()
        assert closed is True

        records = load_shadow_journal(target)
        # The surviving session seals cleanly and looks complete on its own.
        assert shadow_journal_inventory_complete(records) is True

        record_shadow_journal_drop(target, "SessionBoundaryHeld")

        # Once the lost review is known, the same records are no longer a
        # complete inventory.
        assert shadow_journal_inventory_complete(records, target) is False
        assert shadow_journal_drop_marker_path(target).exists()

    def test_the_marker_is_private_and_records_a_reason(self, tmp_path):
        target = tmp_path / "shadow.ndjson"
        record_shadow_journal_drop(target, "SessionBoundaryHeld")
        marker = shadow_journal_drop_marker_path(target)

        assert oct(marker.stat().st_mode)[-3:] == "600"
        assert "SessionBoundaryHeld" in marker.read_text(encoding="utf-8")


class TestTheMarkerCannotBeUsedToWriteElsewhere:
    """A repository-local .pr_agent.toml can choose shadow_journal_path."""

    def test_a_symlinked_marker_is_refused_rather_than_followed(self, tmp_path):
        private = tmp_path / "journal-dir"
        private.mkdir(mode=0o700)
        victim = tmp_path / "victim.txt"
        victim.write_text("untouched", encoding="utf-8")
        target = private / "shadow.ndjson"
        marker = shadow_journal_drop_marker_path(target)
        marker.symlink_to(victim)

        with pytest.raises(Exception):
            record_shadow_journal_drop(target, "SessionBoundaryHeld")

        # The planted link must not have been written through or re-permissioned.
        assert victim.read_text(encoding="utf-8") == "untouched"
        assert oct(victim.stat().st_mode)[-3:] != "600"

    def test_a_world_readable_parent_is_refused(self, tmp_path):
        loose = tmp_path / "loose"
        loose.mkdir(mode=0o755)

        with pytest.raises(Exception):
            record_shadow_journal_drop(loose / "shadow.ndjson", "SessionBoundaryHeld")


class TestTheReportHonoursLostReviews:
    """The marker is worthless if the production report path never looks at it."""

    def test_acceptance_accepts_a_journal_path(self):
        import inspect

        from pr_agent.algo.checkpoint_evaluation_report import (
            _build_shadow_pilot_binding,
            build_shadow_pilot_acceptance,
        )

        for function in (build_shadow_pilot_acceptance, _build_shadow_pilot_binding):
            assert "journal_path" in inspect.signature(function).parameters

    def test_the_report_passes_its_path_through(self):
        import inspect

        from pr_agent.algo import checkpoint_evaluation_report as report

        source = inspect.getsource(report.build_checkpoint_pilot_report)
        assert "journal_path=shadow_journal_path" in source


class TestCacheHitsAreRecorded:
    def test_a_cached_review_still_produces_an_entry(self):
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._run_review_snapshot_impl)
        cached_return = source.index("return cached_result")
        recorded = source.rindex("_record_shadow_journal_entry", 0, cached_return)
        # The recorder must run before the cache-hit early return, or normal
        # cache-hit activity never reaches the journal at all.
        assert recorded < cached_return

    def test_a_cached_entry_reports_itself_as_cached(self):
        snapshot = _snapshot()
        entry = shadow_entry_from_snapshot_result(
            snapshot, _result(snapshot, cached=True)
        )

        assert entry.cached is True


class TestCacheHitsReportTheirOwnEvent:
    """A cache hit made no model request, and is a distinct event from the last."""

    def test_a_cache_hit_reports_zero_spend_not_the_cached_cost(self):
        snapshot = _snapshot()
        result = _result(
            snapshot,
            cached=True,
            usage={"total_tokens": 1200},
            cost={"total_usd": "0.0042", "status": "complete"},
            latency_seconds=42.0,
        )
        entry = shadow_entry_from_snapshot_result(snapshot, result, lookup_seconds=0.01)

        # Zero is measured truth here, not missing data: this invocation spent
        # nothing. Repeating the cached cost would charge a historical call again
        # on every hit and inflate cost per developer hour.
        assert entry.tokens.status is MeasurementStatus.COMPLETE
        assert entry.tokens.value == 0.0
        assert entry.cost_usd.status is MeasurementStatus.COMPLETE
        assert entry.cost_usd.value == 0.0

    def test_a_cache_hit_reports_this_lookup_not_the_original_latency(self):
        snapshot = _snapshot()
        result = _result(snapshot, cached=True, latency_seconds=42.0)
        entry = shadow_entry_from_snapshot_result(snapshot, result, lookup_seconds=0.01)

        assert entry.latency_seconds.value == pytest.approx(0.01)

    def test_two_identical_cache_hits_are_both_retained(self, tmp_path):
        """Two hits can measure the same lookup time and hash alike.

        entry_id is a content hash, so identical events share one. The writer's
        record_id carries the sequence number and stays distinct, which is what
        the reader and the acceptance check key on. Keying on entry_id instead
        would throw away an ordinary second cache hit.
        """
        target = tmp_path / "shadow.ndjson"
        snapshot = _snapshot()
        result = _result(snapshot, cached=True)
        writer = shadow_journal_writer_from_settings(_settings(path=str(target)))
        writer.submit(shadow_entry_from_snapshot_result(snapshot, result, lookup_seconds=0.01))
        writer.submit(shadow_entry_from_snapshot_result(snapshot, result, lookup_seconds=0.01))
        closed = writer.close()
        assert closed is True

        records = load_shadow_journal(target)
        assert len(records) == 2
        assert records[0].entry.entry_id == records[1].entry.entry_id
        assert records[0].record_id != records[1].record_id

    def test_a_fresh_review_still_reports_its_real_telemetry(self):
        snapshot = _snapshot()
        entry = shadow_entry_from_snapshot_result(snapshot, _result(snapshot))

        assert entry.tokens.value == pytest.approx(1200)
        assert entry.latency_seconds.value == pytest.approx(1.25)


class TestARepositoryCannotChooseTheJournalDestination:
    """shadow_journal_path names a host file the recorder creates and appends to."""

    def test_the_section_is_host_only(self):
        from pr_agent.config_security import REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION

        assert "checkpoint_evaluation" in REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION
        # Empty allowlist means every key in the section is dropped from
        # repository settings, the same treatment push_outputs and otel get.
        assert REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION["checkpoint_evaluation"] == frozenset()

    def test_repo_settings_cannot_enable_recording_or_pick_a_path(self, tmp_path):
        from pr_agent.config_security import REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION

        allowed = REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION["checkpoint_evaluation"]
        for key in (
            "shadow_journal_enabled",
            "shadow_journal_path",
            "shadow_journal_max_queue_entries",
            "allow_paid_execution",
            "paid_cost_cap_usd",
        ):
            assert key not in allowed


class TestRecordingDoesNotDelayTheResult:
    """Opening the writer reads the whole journal and fsyncs a boundary."""

    def test_every_record_call_follows_its_emit(self):
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._run_review_snapshot_impl)
        emits = [i for i in range(len(source)) if source.startswith("_emit_snapshot_result(", i)]
        records = [
            i for i in range(len(source)) if source.startswith("_record_shadow_journal_entry(", i)
        ]
        assert emits and records
        # Each recording happens after some emit, so a week-old journal never
        # sits between the review finishing and the developer seeing it.
        for record_at in records:
            assert any(emit_at < record_at for emit_at in emits)

    def test_the_result_is_flushed_before_the_recorder_can_run(self):
        """Ordering alone is not enough when stdout is a pipe.

        A redirected stdout is block-buffered, so an unflushed print would still
        be sitting in the buffer while the recorder reads and fsyncs the journal.
        """
        import contextlib

        from pr_agent import cli

        flushed = []

        class _BlockBufferedStdout:
            def write(self, text):
                return len(text)

            def flush(self):
                flushed.append(True)

        result = SimpleNamespace(to_dict=lambda: {"state": "no_findings"})
        with contextlib.redirect_stdout(_BlockBufferedStdout()):
            cli._emit_snapshot_result(result, None)

        assert flushed, "the snapshot result must be flushed, not left in the buffer"


class TestAReviewThatSpentNothingSaysSoExactly:
    """The gate needs a complete cost on every record to compute cost per hour."""

    def test_a_model_free_review_records_a_complete_zero(self):
        """A stale snapshot or an empty diff makes no request, so its cost is zero.

        Recording that as unavailable would make the gate's cost metric partial
        for an ordinary invocation whose cost is precisely known.
        """
        snapshot = _snapshot()
        result = _result(snapshot, usage={}, cost={})
        entry = shadow_entry_from_snapshot_result(snapshot, result, model_calls=0)

        assert entry.cost_usd.status is MeasurementStatus.COMPLETE
        assert entry.cost_usd.value == 0.0
        assert entry.tokens.status is MeasurementStatus.COMPLETE
        assert entry.tokens.value == 0.0

    def test_an_unknown_call_count_still_reports_what_was_measured(self):
        """None means the caller cannot say, which is not the same as zero."""
        snapshot = _snapshot()
        entry = shadow_entry_from_snapshot_result(snapshot, _result(snapshot))

        assert entry.tokens.value == pytest.approx(1200)

    def test_a_cache_hit_reports_zero_even_without_the_count(self):
        """result.cached is authoritative, so a caller cannot reintroduce the bug."""
        snapshot = _snapshot()
        result = _result(
            snapshot,
            cached=True,
            usage={"total_tokens": 1200},
            cost={"total_usd": "0.0042", "status": "complete"},
        )
        entry = shadow_entry_from_snapshot_result(snapshot, result, lookup_seconds=0.01)

        assert entry.cost_usd.value == 0.0
        assert entry.tokens.value == 0.0

    def test_a_nonsense_call_count_is_refused(self):
        snapshot = _snapshot()
        for bad in (-1, True, 1.5, "0"):
            with pytest.raises(EvaluationValidationError):
                shadow_entry_from_snapshot_result(snapshot, _result(snapshot), model_calls=bad)

    def test_the_review_path_reports_the_count_it_measured(self):
        """num_ai_calls is the only thing that knows an empty diff made no request."""
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._run_review_snapshot_impl)
        assert "model_calls=None if details is None else details.num_ai_calls" in source
        # The two short-circuits never reach a provider at all.
        assert source.count("model_calls=0") == 2


class TestNothingBetweenTheModelAndTheRecorderIsUnguarded:
    """Three rounds of review found three separate steps in this span exposed.

    Publication, then cache.write(), then markdown staging. Guarding them one at
    a time left the next one up still exposed, so the span is guarded as a whole.
    """

    def test_one_guard_covers_the_whole_span(self):
        import inspect

        from pr_agent import cli

        lines = inspect.getsource(cli._run_review_snapshot_impl).splitlines()
        guard = next(i for i, line in enumerate(lines) if line.strip() == "except BaseException as exc:")
        opens = [i for i, line in enumerate(lines) if line == "    try:" and i < guard]
        assert opens, "the guard has no function-level try"
        protected = "\n".join(lines[opens[0]:guard])

        # Every step that can fail after the review has started must be inside it.
        for statement in (
            "asyncio.run(inner())",
            "tempfile.TemporaryDirectory(",
            "reviewer.recapture(",
            "markdown_path.read_bytes()",
            "structured_review.get(",
            "build_snapshot_result(",
            "cache.write(",
            "_atomic_replace_bytes(",
            "_emit_snapshot_result(",
            "_record_shadow_journal_entry(",
        ):
            assert statement in protected, f"{statement} sits outside the recording guard"

        handler = "\n".join(lines[guard:guard + 6])
        assert "_mark_shadow_journal_drop(exc)" in handler
        assert "raise" in handler

    def test_the_guard_only_marks_when_a_review_was_attempted(self):
        """Marking a drop for an argument error would falsely spoil the inventory."""
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._run_review_snapshot_impl)
        assert "review_attempted = False" in source
        assert "review_attempted = True" in source
        assert "if review_attempted:" in source

    def test_marking_a_drop_cannot_replace_the_real_failure(self):
        import inspect

        from pr_agent import cli

        assert "except Exception" in inspect.getsource(cli._mark_shadow_journal_drop)


class TestAPaidReviewIsNeverLostToAFailedPublication:
    """A review that ran and cost money must reach the journal or a drop marker."""

    def test_every_emit_records_through_a_finally(self):
        """Publication can raise between the review finishing and the recorder.

        --output writes to a caller-supplied path and _emit_snapshot_result()
        writes --json-output, so either can fail after the model was paid. Without
        a finally the event has neither an entry nor a drop marker, and the
        remaining journal still reports a complete inventory.
        """
        import inspect

        from pr_agent import cli

        lines = inspect.getsource(cli._run_review_snapshot_impl).splitlines()
        # Comments name these functions too, and a mention is not a call site.
        code = [line for line in lines if not line.strip().startswith("#")]
        emits = [i for i, line in enumerate(code) if "_emit_snapshot_result(" in line]
        records = [i for i, line in enumerate(code) if "_record_shadow_journal_entry(" in line]
        assert len(emits) == len(records) == 3

        for emit_at, record_at in zip(emits, records, strict=True):
            assert emit_at < record_at
            between = code[emit_at:record_at]
            assert any(line.strip() == "finally:" for line in between), (
                f"the emit at line {emit_at} does not record through a finally"
            )

    def test_everything_after_the_model_call_is_inside_the_protected_region(self):
        """The paid review exists from build_snapshot_result() onward.

        Three things between there and the recorder write to disk and can fail:
        cache.write() into .git/pr-agent, --output through _atomic_replace_bytes(),
        and --json-output through _emit_snapshot_result(). Any one of them raising
        outside the try loses the event and its cost while an older surviving
        journal still reads as a complete inventory.
        """
        import inspect

        from pr_agent import cli

        lines = inspect.getsource(cli._run_review_snapshot_impl).splitlines()
        record_at = max(i for i, line in enumerate(lines) if "_record_shadow_journal_entry(" in line)
        # Walk back to the finally that reaches the recorder, then to its try.
        close = max(i for i in range(record_at) if lines[i].strip() == "finally:")
        depth = len(lines[close]) - len(lines[close].lstrip())
        open_at = max(
            i for i in range(close)
            if lines[i].strip() == "try:" and len(lines[i]) - len(lines[i].lstrip()) == depth
        )
        protected = "\n".join(lines[open_at:close])

        for statement in ("cache.write(", "_atomic_replace_bytes(", "_emit_snapshot_result("):
            assert statement in protected, f"{statement} can lose a paid review"

    def test_the_recorder_cannot_mask_the_publication_error(self):
        """A finally that raised would replace the real failure with its own."""
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._record_shadow_journal_entry)
        assert "except Exception" in source
        # Belt and braces: the behaviour itself is covered by
        # TestRecordingNeverBreaksTheReview.


class TestRecordingImpliesPricing:
    """The live-shadow gate reads cost per developer hour, so entries need a cost."""

    def test_recording_turns_on_cost_collection(self, tmp_path):
        from pr_agent import cli
        from pr_agent.config_loader import get_settings

        settings = get_settings()
        original = settings.get("config.output_run_cost", False)
        original_section = dict(settings.get("checkpoint_evaluation", {}) or {})
        try:
            settings.set("config.output_run_cost", False)
            settings.set("checkpoint_evaluation.shadow_journal_enabled", True)
            settings.set("checkpoint_evaluation.shadow_journal_path", str(tmp_path / "shadow.ndjson"))

            cli._collect_cost_for_shadow_recording()

            # Without this the journal records seven days of entries whose cost is
            # permanently unavailable, and the gate can never read a cost per hour.
            assert settings.get("config.output_run_cost") is True
        finally:
            settings.set("config.output_run_cost", original)
            settings.set("checkpoint_evaluation", original_section, merge=False)

    def test_pricing_stays_off_when_recording_is_off(self, tmp_path):
        from pr_agent import cli
        from pr_agent.config_loader import get_settings

        settings = get_settings()
        original = settings.get("config.output_run_cost", False)
        original_section = dict(settings.get("checkpoint_evaluation", {}) or {})
        try:
            settings.set("config.output_run_cost", False)
            settings.set("checkpoint_evaluation.shadow_journal_enabled", False)
            settings.set("checkpoint_evaluation.shadow_journal_path", str(tmp_path / "shadow.ndjson"))
            cli._collect_cost_for_shadow_recording()
            assert settings.get("config.output_run_cost") is False

            # Enabled but with no path opens no writer, so it must not price either.
            settings.set("checkpoint_evaluation.shadow_journal_enabled", True)
            settings.set("checkpoint_evaluation.shadow_journal_path", "")
            cli._collect_cost_for_shadow_recording()
            assert settings.get("config.output_run_cost") is False
        finally:
            settings.set("config.output_run_cost", original)
            settings.set("checkpoint_evaluation", original_section, merge=False)

    def test_the_predicate_matches_the_writer_it_stands_in_for(self, tmp_path):
        """Two copies of this condition would eventually disagree."""

        from pr_agent import cli

        for enabled, path, expected in (
            (True, str(tmp_path / "shadow.ndjson"), True),
            (True, "", False),
            (False, str(tmp_path / "shadow.ndjson"), False),
            ("true", str(tmp_path / "shadow.ndjson"), False),
        ):
            settings = _settings(enabled=enabled, path=path)
            assert cli._shadow_recording_requested(settings) is expected
            assert (shadow_journal_writer_from_settings(settings) is not None) is expected

    def test_the_pricing_flag_is_part_of_the_configuration_identity(self, tmp_path):
        """This is why the flag has to be reapplied, not just applied once.

        output_run_cost is not among the hash's transient config keys, so turning
        it on changes the snapshot's configuration identity.
        """
        from pr_agent import cli
        from pr_agent.algo.review_configuration import snapshot_review_configuration_hash
        from pr_agent.config_loader import get_settings

        entry_state = cli._snapshot_all_settings()
        try:
            get_settings().set("config.output_run_cost", False)
            without = snapshot_review_configuration_hash("skills", {})
            get_settings().set("config.output_run_cost", True)
            with_pricing = snapshot_review_configuration_hash("skills", {})

            assert without != with_pricing
        finally:
            cli._restore_all_settings(entry_state)

    def test_every_repository_settings_rebuild_reapplies_it(self):
        """A recapture restores the invocation baseline, which predates the flag.

        current_configuration_hash() rebuilds the repository layer on every
        staleness check. If that path calls apply_local_repo_settings() directly,
        pricing is not reapplied, the recapture hashes a different configuration
        than the initial capture, and the review is judged stale and returns
        without ever calling the model.
        """
        import inspect

        from pr_agent import cli

        source = inspect.getsource(cli._run_review_snapshot_impl)
        assert "_apply_snapshot_repository_settings(" in source
        bare_calls = [
            line.strip()
            for line in source.splitlines()
            if "apply_local_repo_settings(" in line
            and "_apply_snapshot_repository_settings(" not in line
        ]
        assert bare_calls == [], f"these bypass the shared helper: {bare_calls}"

        # The helper is the only thing allowed to rebuild that layer, so it must
        # be the thing that reapplies what recording implies.
        helper = inspect.getsource(cli._apply_snapshot_repository_settings)
        assert "apply_local_repo_settings(" in helper
        assert "_collect_cost_for_shadow_recording()" in helper
