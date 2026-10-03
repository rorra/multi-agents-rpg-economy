"""The read-only observation dashboard (IMPLEMENTATION.md phase 7).

The dashboard is never part of the experiment. That is enforced structurally --
by what it is allowed to import and by opening logs read-only -- rather than by
convention, because a tool that could perturb a run would invalidate every run
it was pointed at.
"""

from __future__ import annotations

import ast
import json
import os
import stat
from pathlib import Path

import pytest

from rpg_economy.dashboard import build_dashboard, render_dashboard
from rpg_economy.domain import Condition, Scenario
from rpg_economy.runner import run_condition, run_trio
from tests.conftest import make_config

SRC = Path(__file__).resolve().parents[1] / "src" / "rpg_economy"
DASHBOARD = SRC / "dashboard"

#: Reaching any of these would give the dashboard a path to influence a run.
FORBIDDEN_PACKAGES = ("rpg_economy.engine", "rpg_economy.llm", "rpg_economy.orchestration")


def short_config(**overrides):
    base = {
        "phases": {"base_ticks": 6, "shock_ticks": 6, "recovery_ticks": 8},
        "time": {"horizon_ticks": 20},
        "metrics": {"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    }
    base.update(overrides)
    return make_config(**base)


# ---------------------------------------------------------------------------
# Structural separation
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_dashboard_cannot_reach_the_engine() -> None:
    """An import-graph guarantee, not a promise about behaviour.

    A module with no path to the engine, the orchestration layer or the LLM
    client cannot perturb a run whatever it does with what it reads.
    """
    violations: list[str] = []
    for path in sorted(DASHBOARD.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith(FORBIDDEN_PACKAGES):
                    violations.append(f"{path.name}: imports {name}")
    assert not violations, "dashboard reaches into the experiment:\n" + "\n".join(violations)


@pytest.mark.invariant
def test_the_dashboard_never_writes(tmp_path: Path) -> None:
    """Reading a log must not modify it -- content or mtime."""
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    log = result.log_path

    before = log.read_bytes()
    before_stat = log.stat()
    log.chmod(before_stat.st_mode & ~stat.S_IWUSR)  # read-only on disk
    try:
        build_dashboard(log)
        assert log.read_bytes() == before
        assert log.stat().st_mtime == before_stat.st_mtime
    finally:
        os.chmod(log, before_stat.st_mode)


@pytest.mark.invariant
def test_the_dashboard_reads_a_log_it_did_not_produce(tmp_path: Path) -> None:
    """Fed exclusively from logs: no live state, no config, no run object."""
    config = short_config(run={"condition": "A"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    moved = tmp_path / "elsewhere" / "copied.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(result.log_path.read_bytes())

    data = build_dashboard(moved)
    assert data.condition == Condition.A.value
    assert data.seed == config.run.seed


# ---------------------------------------------------------------------------
# What it shows
# ---------------------------------------------------------------------------
def test_the_feed_collapses_repeats_by_agent(tmp_path: Path) -> None:
    """Six stalled artisans interleave, so adjacency-based collapsing collapses
    nothing; grouping by (event, agent) is what makes the feed readable."""
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    data = build_dashboard(result.log_path)

    pairs = [(entry.event_type, entry.agent_id) for entry in data.feed]
    assert len(pairs) == len(set(pairs)), "feed holds duplicate (event, agent) entries"
    for entry in data.feed:
        assert entry.first_tick <= entry.last_tick
        assert entry.count >= 1


def test_the_dashboard_surfaces_the_shock(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    data = build_dashboard(result.log_path)

    assert "shock_applied" in data.event_counts
    assert any(entry.event_type == "shock_applied" for entry in data.feed)


def test_the_dashboard_shows_final_holdings_from_the_log(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    data = build_dashboard(result.log_path)

    assert data.holdings
    assert set(data.holdings) == set(result.state.accounts)


def test_condition_c_shows_bulletin_activity(tmp_path: Path) -> None:
    trio = run_trio(short_config(), tmp_path, seed=7)

    counts = {
        condition: build_dashboard(result.log_path).event_counts
        for condition, result in trio.items()
    }

    assert counts[Condition.C].get("bulletin_published", 0) > 0
    assert counts[Condition.B].get("bulletin_published", 0) == 0
    assert counts[Condition.A].get("director_call", 0) > 0


# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_view_holds_no_language(tmp_path: Path) -> None:
    """Language is applied at render time; the data carries enums and numbers."""
    config = short_config(run={"condition": "A", "scenario": "resource_scarcity"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    payload = build_dashboard(result.log_path).to_payload()
    text = repr(payload)

    assert payload["scenario"] == Scenario.RESOURCE_SCARCITY.value
    for rendered in ("Escasez", "Resource scarcity", "Centralized", "Centralizada"):
        assert rendered not in text


def test_both_languages_render_from_one_view(tmp_path: Path) -> None:
    config = short_config(run={"condition": "C"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    data = build_dashboard(result.log_path)

    english = render_dashboard(data, locale="en")
    spanish = render_dashboard(data, locale="es")

    assert "Read-only view" in english
    assert "solo lectura" in spanish
    assert english != spanish


@pytest.mark.invariant
def test_every_event_type_and_bucket_has_a_label_in_both_languages() -> None:
    """A new event type must not surface as a missing-message crash.

    The dashboard renders whatever an event log contains, so the catalogs have
    to cover the enums exhaustively rather than the subset in use today.
    """
    import json

    from rpg_economy.domain import TokenBucket
    from rpg_economy.i18n import SUPPORTED_LOCALES, locales_dir
    from rpg_economy.logging_ import EventType

    required = {f"event.{e.value}" for e in EventType} | {f"bucket.{b.value}" for b in TokenBucket}

    for locale in SUPPORTED_LOCALES:
        catalog = json.loads((locales_dir() / f"{locale}.json").read_text(encoding="utf-8"))
        missing = sorted(required - set(catalog))
        assert not missing, f"{locale} is missing: {missing}"


# ---------------------------------------------------------------------------
# Replay (IMPLEMENTATION.md phase 7)
# ---------------------------------------------------------------------------
def _trio(tmp_path: Path, seed: int = 7):
    return run_trio(short_config(), tmp_path, seed=seed)


@pytest.mark.invariant
def test_replay_stays_outside_the_experiment(tmp_path: Path) -> None:
    """The isolation invariant covers the replay too, not only the text view.

    Enforced by the same import-graph rule: the replay lives inside the
    dashboard package, so a module here that reached the engine would fail
    test_the_dashboard_cannot_reach_the_engine. This asserts the file is
    actually in that package rather than beside it.
    """
    from rpg_economy.dashboard import replay as replay_module

    assert Path(replay_module.__file__).parent == DASHBOARD


@pytest.mark.invariant
def test_a_replay_reads_logs_and_writes_only_its_page(tmp_path: Path) -> None:
    import os
    import stat

    from rpg_economy.dashboard.html import write_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    trio = _trio(tmp_path / "runs")
    paths = trio_logs(tmp_path / "runs")
    before = {p: (p.read_bytes(), p.stat()) for p in paths}
    for path in paths:
        path.chmod(before[path][1].st_mode & ~stat.S_IWUSR)
    try:
        bundle = build_bundle(paths)
        page = write_replay(bundle, tmp_path / "out" / "replay.html", locale="es")
        for path in paths:
            assert path.read_bytes() == before[path][0]
            assert path.stat().st_mtime == before[path][1].st_mtime
    finally:
        for path in paths:
            os.chmod(path, before[path][1].st_mode)

    assert page.exists()
    assert len(trio) == 3


@pytest.mark.invariant
def test_a_bundle_refuses_runs_from_different_worlds(tmp_path: Path) -> None:
    """The trio view claims 'same world, three orchestrations'.

    Rendering mismatched runs side by side would illustrate the opposite while
    looking exactly the same, which is the failure the trio integrity gate
    (design s12) prevents in the runner and this prevents in the replay.
    """
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    _trio(tmp_path / "a", seed=7)
    _trio(tmp_path / "b", seed=8)

    with pytest.raises(ValueError, match="one seed"):
        build_bundle([trio_logs(tmp_path / "a")[0], trio_logs(tmp_path / "b")[0]])


def test_a_track_carries_one_frame_per_tick(tmp_path: Path) -> None:
    """Frames are complete states, so a seek is an index and never a re-fold."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path)
    track = build_track(trio[Condition.A].log_path)

    assert len(track.frames) == track.horizon + 1
    for i, frame in enumerate(track.frames):
        assert frame.tick == i
    # Cumulative counters never go backwards.
    totals = [f.trades_total for f in track.frames]
    assert totals == sorted(totals)


def test_the_shock_window_is_recovered_from_the_log(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import build_track

    config = short_config()
    trio = _trio(tmp_path)
    track = build_track(trio[Condition.B].log_path)

    assert track.shock_tick == config.phases.shock_tick
    assert track.reversion_tick == config.phases.reversion_tick


@pytest.mark.invariant
def test_an_agent_is_unobserved_rather_than_broke_before_activation(tmp_path: Path) -> None:
    """Zero gold and unknown gold are different claims.

    At tick 0 no agent has been activated, so nothing in the log says what any
    of them holds. Rendering that as 0 would show the whole roster as bankrupt.
    """
    from rpg_economy.dashboard.replay import AGENT_FIELDS, build_track

    trio = _trio(tmp_path)
    track = build_track(trio[Condition.B].log_path)
    observed_index = AGENT_FIELDS.index("observed")

    # K activations happen at tick 0, so some agents are observed immediately and
    # the rest are not observed at all yet -- which is the case that must not
    # render as zero gold.
    first = track.frames[0]
    unobserved = [row for row in first.agents.values() if row[observed_index] == 0]
    assert len(first.agents) < len(track.agent_ids) or unobserved, (
        "expected at least one agent with no logged observation at tick 0"
    )

    last = track.frames[-1]
    assert all(row[observed_index] == 1 for row in last.agents.values()), (
        "every agent should have been observed by the end of a run"
    )


def test_the_replay_page_is_self_contained(tmp_path: Path) -> None:
    """No server, no CDN: it must open from disk."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    _trio(tmp_path / "runs")
    html = render_replay(build_bundle(trio_logs(tmp_path / "runs")), locale="es")

    assert "<script src=" not in html
    assert 'link rel="stylesheet"' not in html
    # The SVG namespace is an identifier, never fetched; anything else is a URL.
    html = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in html and "https://" not in html
    assert "window.DATA=" in html


@pytest.mark.invariant
def test_the_replay_renders_in_both_languages_from_one_timeline(tmp_path: Path) -> None:
    """Pages render in both English and Spanish (IMPLEMENTATION.md section 2)."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))

    spanish = render_replay(bundle, locale="es")
    english = render_replay(bundle, locale="en")

    assert "Reproducción" in spanish and "solo lectura" in spanish
    assert "Replay" in english and "Read-only" in english
    # One timeline, two pages: the embedded data must be language-neutral.
    payload = json.dumps(bundle.to_payload(), ensure_ascii=False)
    for rendered in ("Escasez", "Resource scarcity", "Centralizada", "Centralized"):
        assert rendered not in payload


@pytest.mark.invariant
def test_model_text_cannot_break_out_of_the_page_script(tmp_path: Path) -> None:
    """A short_reason is model-written: it must stay data, whatever it contains."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    trio = _trio(tmp_path / "runs")
    poison = "<!-- <script>alert(1)</script> </script><b>&amp;"
    log = trio[Condition.B].log_path
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        event = json.loads(line)
        if event.get("type") == "agent_output":
            event["payload"]["short_reason"] = poison
            lines[index] = json.dumps(event, ensure_ascii=False) + "\n"
            break
    else:
        pytest.fail("the run logged no agent output")
    log.write_text("".join(lines), encoding="utf-8")

    html = render_replay(build_bundle(trio_logs(tmp_path / "runs")), locale="en")

    # The page keeps exactly its own two script elements, and the data opens
    # no comment and closes no tag.
    assert html.count("<script") == 2 and html.count("</script>") == 2
    start = html.index("window.DATA=")
    blob = html[start : html.index("</script>", start)]
    assert not any(char in blob for char in "<>&")
    data = json.loads(blob.removeprefix("window.DATA=").split(";window.STR=")[0])
    reasons = [
        step.get("d", {}).get("why")
        for track in data["tracks"]
        for frame in track["frames"]
        for step in frame.get("st", [])
    ]
    assert poison in reasons
    assert html.rstrip().endswith("</html>")


def test_the_tail_consumes_only_complete_lines(tmp_path: Path) -> None:
    """A log is being appended to while a pilot runs; a half-written line is not
    an error to recover from, it is a line that has not happened yet."""
    from rpg_economy.dashboard import tail_events

    trio = _trio(tmp_path / "runs")
    source = trio[Condition.B].log_path
    lines = source.read_bytes().split(b"\n")

    partial = tmp_path / "partial.jsonl"
    complete = b"\n".join(lines[:-2]) + b"\n"
    partial.write_bytes(complete + lines[-2][:20])  # last line cut mid-way

    events, offset = tail_events(partial)
    assert offset == len(complete)

    partial.write_bytes(complete + lines[-2] + b"\n")
    more, offset2 = tail_events(partial, from_offset=offset)
    assert offset2 > offset
    assert len(events) >= 0 and isinstance(more, list)


def test_the_tail_returns_nothing_for_a_missing_log(tmp_path: Path) -> None:
    from rpg_economy.dashboard import tail_events

    events, offset = tail_events(tmp_path / "not-yet.jsonl")
    assert events == [] and offset == 0


@pytest.mark.invariant
def test_effective_activations_are_counted_not_differenced(tmp_path: Path) -> None:
    """Executed actions and wasted activations are not paired one-to-one.

    An activation can execute a WAIT and still be wasted (design s11.2), and a
    rejected action executes nothing, so subtracting wasted from *executed* can
    go negative and clamp to zero.
    """
    from rpg_economy.dashboard.replay import AGENT_FIELDS, build_track

    trio = _trio(tmp_path / "runs")
    track = build_track(trio[Condition.A].log_path)
    eff = AGENT_FIELDS.index("effective")
    acts = AGENT_FIELDS.index("activations")

    final = track.frames[-1]
    assert final.agents, "no agents observed"
    for agent, row in final.agents.items():
        assert row[acts] > 0, f"{agent} recorded no activations"
        assert 0 <= row[eff] <= row[acts], f"{agent}: effective {row[eff]} of {row[acts]}"
    assert any(row[eff] > 0 for row in final.agents.values()), (
        "no agent had a single effective activation, which the checker contradicts"
    )


def test_a_replay_carries_its_provenance(tmp_path: Path) -> None:
    """A page outlives the context it was made in; it has to say what it is."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"), caveat="Instrument validation only.")
    html = render_replay(bundle, locale="en")

    assert "Instrument validation only." in html
    assert "INSTRUMENT VALIDATION" in html
    for track in bundle.tracks:
        assert track.prompt_version in html
        assert track.config_hash[:12] in html


@pytest.mark.invariant
def test_replay_bucket_totals_reconcile_with_the_ledger(tmp_path: Path) -> None:
    """A page whose bars contradict the log must not render.

    The Director's calls are logged under their own event type, not `llm_call`,
    so an aggregation reading only `llm_call` would show a Director token bar at
    zero beside a panel counting calls. A viewer has no way to tell which number
    is wrong, so the contradiction is made unrenderable.
    """
    import json

    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    for condition, result in trio.items():
        track = build_track(result.log_path)
        final = track.frames[-1]
        consumed = next(
            json.loads(line)
            for line in result.log_path.read_text().splitlines()
            if '"tokens_consumed"' in line
        )["payload"]

        ledger = {bucket: total for bucket, total in consumed["total"].items() if total}
        assert final.tokens == ledger, f"{condition.value}: {final.tokens} vs ledger {ledger}"
        assert final.tokens_total == consumed["authoritative_total"]

    # Condition A must show Director consumption, not a zero bar beside a
    # Director panel counting calls.
    from rpg_economy.domain import TokenBucket

    a_final = build_track(trio[Condition.A].log_path).frames[-1]
    assert a_final.tokens.get(TokenBucket.DIRECTOR.value, 0) > 0
    assert a_final.director_calls > 0


@pytest.mark.invariant
def test_a_replay_refuses_to_build_when_bars_would_contradict_the_log(tmp_path: Path) -> None:
    """Guards the guard: the reconciliation must actually reject a mismatch."""
    import json

    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    source = trio[Condition.A].log_path
    lines = source.read_text().splitlines()

    tampered = tmp_path / "tampered.jsonl"
    out = []
    for line in lines:
        event = json.loads(line)
        if event["type"] == "tokens_consumed":
            # Inflate one bucket: the displayed total can no longer match.
            event["payload"]["total"]["director"] += 12345
        out.append(json.dumps(event))
    tampered.write_text("\n".join(out) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="disagree with the ledger"):
        build_track(tampered)


# ---------------------------------------------------------------------------
# Step-by-step view
# ---------------------------------------------------------------------------
def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()[1:]]


def test_every_activation_becomes_one_step_in_order(tmp_path: Path) -> None:
    """The step list shows each activation, in the order the log recorded it."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    for result in trio.values():
        track = build_track(result.log_path)
        expected: dict[int, list[str]] = {}
        for event in _events(result.log_path):
            if event["type"] == "observation_delivered":
                expected.setdefault(event["tick"], []).append(event["agent_id"])
        for frame in track.frames:
            assert [s["a"] for s in frame.steps] == expected.get(frame.tick, [])
            assert [s["n"] for s in frame.steps] == list(range(1, len(frame.steps) + 1))
            for step in frame.steps:
                assert step["l"], "every activation made a model call"
                assert "d" in step or step["x"], "a step has a decision or an invalid output"


def test_steps_say_why_the_agent_acted(tmp_path: Path) -> None:
    """A: the Director's TASK. B and C: the triggers the scheduler served."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    for condition, result in trio.items():
        steps = [s for f in build_track(result.log_path).frames for s in f.steps]
        assert steps
        for step in steps:
            if condition is Condition.A:
                assert step["w"]["task"]["ty"], step
                assert step["r"]["st"] in {"SUCCESS", "PARTIAL", "BLOCKED", "FAILED"}
            else:
                assert step["w"]["tr"], step
                assert "r" not in step


def _unpack(packed: str) -> list[str]:
    import base64
    import gzip

    return json.loads(gzip.decompress(base64.b64decode(packed)))


def test_every_model_call_carries_its_prompt_and_reply(tmp_path: Path) -> None:
    """Each agent call and each Director call points at the exact texts in the log."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    for condition, result in trio.items():
        events = _events(result.log_path)
        systems = {
            e["payload"]["sha"]: e["payload"]["text"]
            for e in events
            if e["type"] == "prompt_system"
        }
        assert systems, "every run logs the system prompts it sent"
        track = build_track(result.log_path)
        texts = _unpack(track.prompts.packed())
        calls = [e["payload"] for e in events if e["type"] == "llm_call"]
        shown = [c for f in track.frames for s in f.steps for c in s["l"]]
        assert len(shown) == len(calls)
        for payload, entry in zip(calls, shown, strict=True):
            system, user, raw = entry[5]
            assert texts[system] == systems[payload["system_sha"]]
            assert texts[user] == payload["user"]
            assert texts[raw] == payload["raw"]
        director = [e["payload"] for e in events if e["type"] == "director_call"]
        shown_director = [c for f in track.frames for c in (f.sched or {}).get("calls", [])]
        assert len(shown_director) == len(director)
        assert bool(director) is (condition is Condition.A)
        for payload, call in zip(director, shown_director, strict=True):
            assert texts[call["ps"]] == systems[payload["system_sha"]]
            assert texts[call["pu"]] == payload["user"]
            assert texts[call["pr"]] == payload["raw"]
