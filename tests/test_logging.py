"""Event log: append-only behaviour, round-tripping and the determinism digest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpg_economy.config import config_hash, load_config
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from rpg_economy.logging_ import (
    SUPPORTED_PYTHON_SERIES,
    Environment,
    EventLogWriter,
    EventType,
    LogError,
    RunMetadata,
    code_digest,
    current_python_version,
    log_digest,
    read_events,
    read_metadata,
)
from tests.conftest import BASE_CONFIG

BASE = BASE_CONFIG


def make_metadata(seed: int = 0, condition: Condition = Condition.A) -> RunMetadata:
    config = load_config(BASE)
    digest = config_hash(config)
    return RunMetadata(
        run_id=RunMetadata.make_run_id(
            config_hash=digest,
            condition=condition,
            scenario=Scenario.RESOURCE_SCARCITY,
            experiment=ExperimentRegime.EQUAL_ACTIVITY,
            seed=seed,
        ),
        design_version=config.design_version,
        schema_version=config.schema_version,
        condition=condition,
        scenario=Scenario.RESOURCE_SCARCITY,
        experiment=ExperimentRegime.EQUAL_ACTIVITY,
        seed=seed,
        config_hash=digest,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        llm_params={"temperature": config.llm.temperature},
        llm_mock=config.llm.mock,
        environment=Environment.capture(),
    )


def write_sample_log(path: Path, *, seed: int = 0) -> None:
    with EventLogWriter(path, make_metadata(seed=seed)) as log:
        log.write(EventType.CYCLE_START, tick=0, cycle=0)
        log.write(
            EventType.TRIGGER_FIRED,
            tick=1,
            cycle=0,
            agent_id="woodworking_1",
            payload={"trigger": "T3_price_move", "good": "iron"},
        )
        log.write(EventType.CYCLE_END, tick=1, cycle=0, payload={"effective_activations": 1})


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
def test_metadata_is_the_first_event(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    assert next(read_events(path)).type is EventType.RUN_METADATA


def test_events_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    events = list(read_events(path))

    assert [e.type for e in events] == [
        EventType.RUN_METADATA,
        EventType.CYCLE_START,
        EventType.TRIGGER_FIRED,
        EventType.CYCLE_END,
    ]
    assert events[2].agent_id == "woodworking_1"
    assert events[2].payload["good"] == "iron"


def test_sequence_numbers_are_gapless(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    assert [e.seq for e in read_events(path)] == [0, 1, 2, 3]


def test_metadata_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    meta = read_metadata(path)
    assert meta.condition is Condition.A
    assert meta.config_hash == config_hash(load_config(BASE))


def test_every_line_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


# ---------------------------------------------------------------------------
# Append-only enforcement
# ---------------------------------------------------------------------------
def test_writer_refuses_a_non_empty_log(tmp_path: Path) -> None:
    """Run logs are write-once: analysis reads them and never mutates them."""
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    with pytest.raises(LogError, match="write-once"), EventLogWriter(path, make_metadata()):
        pass


def test_writing_after_close_fails(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = EventLogWriter(path, make_metadata())
    with writer:
        pass
    with pytest.raises(LogError, match="not open"):
        writer.write(EventType.TICK, tick=1, cycle=0)


def test_malformed_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(LogError, match=":5:"):
        list(read_events(path))


# ---------------------------------------------------------------------------
# Determinism digest
# ---------------------------------------------------------------------------
def test_digest_is_equal_for_identical_runs(tmp_path: Path) -> None:
    """Same seed and configuration must produce equal logs, wall clock aside."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_sample_log(a)
    write_sample_log(b)

    assert a.read_bytes() != b.read_bytes(), "environment block should differ"
    assert log_digest(a) == log_digest(b)


def test_digest_differs_when_a_seed_differs(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_sample_log(a, seed=1)
    write_sample_log(b, seed=2)
    assert log_digest(a) != log_digest(b)


def test_digest_differs_when_an_event_differs(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_sample_log(a)
    with EventLogWriter(b, make_metadata()) as log:
        log.write(EventType.CYCLE_START, tick=0, cycle=0)
    assert log_digest(a) != log_digest(b)


def test_no_wall_clock_outside_metadata_environment(tmp_path: Path) -> None:
    """A timestamp in an event would break the byte-identical-log requirement."""
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    for event in read_events(path):
        if event.type is EventType.RUN_METADATA:
            continue
        rendered = json.dumps(event.model_dump(mode="json"))
        assert "started_at" not in rendered
        assert "timestamp" not in rendered


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
def test_run_id_is_deterministic_and_identity_sensitive() -> None:
    digest = config_hash(load_config(BASE))
    identity = {
        "config_hash": digest,
        "scenario": Scenario.RESOURCE_SCARCITY,
        "experiment": ExperimentRegime.EQUAL_ACTIVITY,
        "seed": 7,
    }
    first = RunMetadata.make_run_id(condition=Condition.A, **identity)
    again = RunMetadata.make_run_id(condition=Condition.A, **identity)
    other = RunMetadata.make_run_id(condition=Condition.B, **identity)

    assert first == again
    assert first != other


# ---------------------------------------------------------------------------
# Fixed interpreter version (IMPLEMENTATION.md section 3)
# ---------------------------------------------------------------------------
def test_metadata_records_the_exact_interpreter_version(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    recorded = read_metadata(path).python_version

    assert recorded == current_python_version()
    assert recorded.count(".") == 2, "the patch release must be recorded, not just 3.14"
    assert recorded.startswith(f"{SUPPORTED_PYTHON_SERIES}.")


def test_interpreter_version_is_not_in_the_excluded_environment_block() -> None:
    """It is part of the environment, not incidental provenance, so the digest sees it."""
    assert "python_version" not in Environment.model_fields
    assert "python_version" in RunMetadata.model_fields


def test_digest_covers_the_interpreter_version(tmp_path: Path) -> None:
    """A rerun under a different patch release is not the same environment."""
    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    baseline = log_digest(path)

    other = tmp_path / "other.jsonl"
    metadata = make_metadata().model_copy(update={"python_version": "3.14.999"})
    with EventLogWriter(other, metadata) as log:
        log.write(EventType.CYCLE_START, tick=0, cycle=0)
        log.write(
            EventType.TRIGGER_FIRED,
            tick=1,
            cycle=0,
            agent_id="woodworking_1",
            payload={"trigger": "T3_price_move", "good": "iron"},
        )
        log.write(EventType.CYCLE_END, tick=1, cycle=0, payload={"effective_activations": 1})

    assert log_digest(other) != baseline


# ---------------------------------------------------------------------------
# A run that stops early still leaves its record
# ---------------------------------------------------------------------------
def test_every_event_reaches_the_file_as_it_is_written(tmp_path: Path) -> None:
    """A run killed mid-way must still show every call it already made."""
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata()) as log:
        log.write(EventType.CYCLE_START, tick=0, cycle=0)
        # Read while the writer is still open, as a crash would leave it.
        lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "events were held in a buffer instead of written"


def test_only_a_log_that_reached_run_end_is_finished(tmp_path: Path) -> None:
    from rpg_economy.logging_ import last_event, log_is_finished

    path = tmp_path / "run.jsonl"
    write_sample_log(path)
    assert not log_is_finished(path)
    assert last_event(path)["type"] == EventType.CYCLE_END.value

    finished = tmp_path / "finished.jsonl"
    with EventLogWriter(finished, make_metadata()) as log:
        log.write(EventType.RUN_END, tick=1, cycle=1, payload={"ticks": 1})
    assert log_is_finished(finished)

    finished.write_bytes(finished.read_bytes()[:-5])
    assert not log_is_finished(finished)
    assert not log_is_finished(tmp_path / "missing.jsonl")


class TestCodeDigest:
    """The run records which code produced it, so resume and derive-budget can tell."""

    def test_is_a_stable_sha256_hex_digest(self) -> None:
        first = code_digest()
        assert len(first) == 64
        int(first, 16)
        assert code_digest() == first

    @pytest.fixture
    def package_copies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Two copies of the package; point ``rpg_economy.__file__`` at one with ``use``."""
        import shutil

        import rpg_economy

        source = Path(rpg_economy.__file__).resolve().parent
        copies = []
        for name in ("one", "two"):
            target = tmp_path / name / "rpg_economy"
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
            copies.append(target)

        def use(copy: Path) -> str:
            monkeypatch.setattr(rpg_economy, "__file__", str(copy / "__init__.py"))
            code_digest.cache_clear()
            return code_digest()

        yield copies, use
        monkeypatch.undo()
        code_digest.cache_clear()

    def test_changes_with_any_simulation_module_and_ignores_install_location(
        self, package_copies
    ) -> None:
        copies, use = package_copies
        original = use(copies[0])
        assert use(copies[1]) == original

        (copies[1] / "domain.py").write_text((copies[1] / "domain.py").read_text() + "\n# edited\n")
        edited = use(copies[1])
        assert edited != original

        (copies[1] / "domain.py").write_bytes((copies[0] / "domain.py").read_bytes())
        (copies[1] / "engine" / "extra_module.py").write_text("")
        assert use(copies[1]) not in {original, edited}

        # Non-Python files are not code.
        (copies[1] / "engine" / "extra_module.py").unlink()
        (copies[1] / "engine" / "notes.txt").write_text("irrelevant")
        assert use(copies[1]) == original

    @pytest.mark.parametrize(
        "relative",
        [
            "runner.py",
            "cli_support.py",
            "engine/market.py",
            "agents/prompts.py",
            "orchestration/director.py",
            "llm/client.py",
            "messaging/bus.py",
            "logging_/writer.py",
            "config/models.py",
        ],
    )
    def test_covers_the_modules_that_determine_a_run(self, package_copies, relative) -> None:
        copies, use = package_copies
        original = use(copies[0])
        path = copies[0] / relative
        path.write_text(path.read_text() + "\n# edited\n")
        assert use(copies[0]) != original

    @pytest.mark.parametrize(
        "relative",
        [
            "analysis/report.py",
            "metrics/run_metrics.py",
            "dashboard/replay.py",
            "i18n/catalog.py",
            "cli.py",
            "campaign/runner.py",
            "integrity.py",
        ],
    )
    def test_ignores_code_that_does_not_change_a_run(self, package_copies, relative) -> None:
        copies, use = package_copies
        original = use(copies[0])
        path = copies[0] / relative
        path.write_text(path.read_text() + "\n# edited\n")
        assert use(copies[0]) == original

    def test_is_computed_once_per_process(self, package_copies) -> None:
        """A file edited on disk mid-campaign is not recorded as the running code."""
        copies, use = package_copies
        original = use(copies[0])
        (copies[0] / "domain.py").write_text((copies[0] / "domain.py").read_text() + "\n# x\n")
        # Cached: the edit is not seen until the cache is cleared.
        assert code_digest() == original
        code_digest.cache_clear()
        assert code_digest() != original

    def test_metadata_field_defaults_to_none_for_older_logs(self) -> None:
        assert make_metadata().code_digest is None
        payload = make_metadata().model_dump(mode="json")
        payload.pop("code_digest")
        assert RunMetadata.model_validate(payload).code_digest is None

    def test_runs_record_the_current_code_digest(self) -> None:
        from rpg_economy.runner import _metadata

        assert _metadata(load_config(BASE)).code_digest == code_digest()
