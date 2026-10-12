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


def test_each_system_prompt_is_logged_once_per_run(tmp_path: Path) -> None:
    trio = _trio(tmp_path / "runs")
    for result in trio.values():
        shas = [
            e["payload"]["sha"] for e in _events(result.log_path) if e["type"] == "prompt_system"
        ]
        assert shas and len(shas) == len(set(shas))


def test_the_prompt_texts_are_packed_reproducibly() -> None:
    from rpg_economy.dashboard.replay import PromptTexts

    store = PromptTexts()
    assert store.packed() is None
    assert store.add(None) is None
    first = store.add("áé prompt")
    assert store.add("reply") == first + 1
    assert store.add("áé prompt") == first, "a repeated text is stored once"
    assert _unpack(store.packed()) == ["áé prompt", "reply"]
    again = PromptTexts()
    again.add("áé prompt")
    again.add("reply")
    assert again.packed() == store.packed()


def test_a_failed_call_shows_its_prompt_and_no_reply(tmp_path: Path) -> None:
    """An attempt that got no reply keeps its prompt; the page shows no reply for it."""
    from rpg_economy.dashboard.replay import build_track

    log = _trio(tmp_path / "runs")[Condition.B].log_path
    lines = log.read_text().splitlines()
    out, injected = [lines[0]], None
    for line in lines[1:]:
        event = json.loads(line)
        if injected is None and event["type"] == "llm_call":
            failed = {
                **event,
                "type": "agent_output_invalid",
                "payload": {
                    "activation": event["payload"].get("activation"),
                    "attempt": 1,
                    "reason": "llm error: no completion",
                    "system_sha": event["payload"]["system_sha"],
                    "user": "the attempted prompt",
                },
            }
            out.append(json.dumps(failed))
            injected = event
        out.append(line)
    assert injected is not None
    edited = tmp_path / "failed.jsonl"
    edited.write_text("\n".join(out) + "\n")
    track = build_track(edited)
    texts = _unpack(track.prompts.packed())
    entries = [x for f in track.frames for s in f.steps for x in s["x"]]
    (entry,) = [x for x in entries if x[2] is not None]
    system, user = entry[2]
    assert texts[user] == "the attempted prompt"
    events = _events(log)
    shas = {
        e["payload"]["sha"]: e["payload"]["text"] for e in events if e["type"] == "prompt_system"
    }
    assert texts[system] == shas[injected["payload"]["system_sha"]]


def test_a_log_without_prompt_text_still_replays(tmp_path: Path) -> None:
    """Runs recorded before prompts were logged: no texts, no buttons."""
    from rpg_economy.dashboard.replay import build_track

    log = _trio(tmp_path / "runs")[Condition.B].log_path
    lines = log.read_text().splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        event = json.loads(line)
        if event["type"] == "prompt_system":
            continue
        for key in ("system_sha", "user"):
            event["payload"].pop(key, None)
        kept.append(json.dumps(event))
    old = tmp_path / "old.jsonl"
    old.write_text("\n".join(kept) + "\n")
    track = build_track(old)
    assert all(
        c[5][0] is None and c[5][1] is None for f in track.frames for s in f.steps for c in s["l"]
    )


def test_only_condition_c_steps_saw_the_bulletin(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    for condition, result in trio.items():
        track = build_track(result.log_path)
        flags = {s["o"]["bul"] for f in track.frames for s in f.steps}
        published = [f for f in track.frames if f.bulletin is not None]
        if condition is Condition.C:
            assert flags == {1} and published
        else:
            assert flags == {0} and not published


def test_messages_carry_their_sender_and_type(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    track = build_track(trio[Condition.B].log_path)
    sent = [e for e in _events(trio[Condition.B].log_path) if e["type"] == "message_sent"]
    shown = [m for f in track.frames for m in f.messages]
    assert len(shown) == len(sent)
    for message, event in zip(shown, sent, strict=True):
        assert message["from"] == event["agent_id"]
        assert message["to"] == event["payload"]["recipient_id"]
        assert message["ty"] == event["payload"]["type"]


def test_expedition_returns_carry_their_outcome_and_haul(tmp_path: Path) -> None:
    """A return shows whether it succeeded, what it brought back and what broke."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    log = trio[Condition.B].log_path
    resolved = [e for e in _events(log) if e["type"] == "expedition_resolved"]
    track = build_track(log)
    shown = [(f.tick, r) for f in track.frames for r in f.returns]

    assert len(shown) == len(resolved)
    for (tick, entry), event in zip(shown, resolved, strict=True):
        assert tick == event["tick"]
        assert entry["a"] == event["agent_id"]
        assert entry["ok"] == (1 if event["payload"]["outcome"] == "success" else 0)
        assert entry["y"] == event["payload"]["yielded"]
        # The loot and the rest that follows every return, not only a failed one.
        assert entry["gd"] == str(event["payload"].get("gold") or "0")
        assert entry["rec"] == (event["payload"].get("recovering_until_tick") or 0)


def test_a_departing_adventurer_shows_its_gear(tmp_path: Path) -> None:
    """The replay shows what each carried item added, as the engine fixed it."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    log = trio[Condition.B].log_path
    started = [e for e in _events(log) if e["type"] == "expedition_started"]
    track = build_track(log)
    departures = [e for f in track.frames for s in f.steps for e in s["e"] if e[0] == "expedition"]
    assert departures
    assert [d[5] for d in departures] == [e["payload"]["gear_bonus"] for e in started]
    for frame in track.frames:
        for step in frame.steps:
            if any(e[0] == "expedition" for e in step["e"]):
                assert step["o"]["gear"], f"{step['a']} left with no gear in its observation"


def test_a_track_carries_what_its_shock_did(tmp_path: Path) -> None:
    """The replay describes the shock from the log, not from the config."""
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    applied = next(e for e in _events(trio[Condition.B].log_path) if e["type"] == "shock_applied")
    track = build_track(trio[Condition.B].log_path)

    assert track.shock is not None
    assert track.shock["scenario"] == applied["payload"]["scenario"]
    assert track.shock["target_resource"] == applied["payload"]["target_resource"]
    assert track.shock["spawn_multiplier"] == applied["payload"]["spawn_multiplier"]


def test_world_stock_is_reconstructed_back_to_the_endowment(tmp_path: Path) -> None:
    """Walking the log back from the final holdings lands on the starting stock."""
    from collections import Counter

    from rpg_economy.dashboard.replay import build_track
    from rpg_economy.engine import build_world

    config = short_config()
    trio = _trio(tmp_path / "runs")
    track = build_track(trio[Condition.B].log_path)

    world = build_world(config)
    endowment: Counter[str] = Counter()
    for agent_id in world.agent_ids:
        for good, units in world.account(agent_id).inventory.items():
            endowment[good.value] += units

    # Undo tick 0 as well: the reconstruction reports end-of-tick stock.
    first = track.frames[0].world_stock
    tick0 = Counter(first)
    for event in _events(trio[Condition.B].log_path):
        if event["tick"] != 0:
            continue
        payload = event["payload"]
        if event["type"] == "production_completed":
            for good, units in payload["consumed"].items():
                tick0[good] += units
            tick0[payload["product"]] -= payload["quantity"]
        elif event["type"] == "consumption":
            tick0[payload["product"]] += payload["quantity"]
        elif event["type"] == "equipment_equipped":
            tick0[payload["product"]] += 1
        elif event["type"] == "expedition_resolved":
            for good, units in payload["yielded"].items():
                tick0[good] -= units
    assert +tick0 == +endowment


def test_zone_stock_reaches_the_replay(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path / "runs")
    track = build_track(trio[Condition.A].log_path)
    assert all(f.zone_stock for f in track.frames)
    assert all(
        units >= 0
        for f in track.frames
        for pool in f.zone_stock.values()
        for units in pool.values()
    )


@pytest.mark.invariant
def test_the_crafting_guide_comes_only_from_the_runs_own_config(tmp_path: Path) -> None:
    """The logs hold the config only by hash, so the guide proves it has the right one."""
    from rpg_economy.dashboard.replay import build_bundle, trio_logs, world_rules

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))

    rules = world_rules(bundle, short_config())
    assert rules is not None
    assert rules["recipes"]["armor"]["in"] == {"iron": 2, "leather": 2}
    assert set(rules["gear"]["warrior"]) == {"melee_weapon", "armor", "health_potion"}

    # A different configuration (the full horizon) describes a different run.
    assert world_rules(bundle, make_config()) is None


def test_the_page_carries_the_guide_and_its_icons(tmp_path: Path) -> None:
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs, world_rules

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    bundle.rules = world_rules(bundle, short_config())
    html = render_replay(bundle, locale="es")

    assert 'id="guide"' in html
    for good in ("iron", "wood", "leather", "herbs", "arcane_crystal", "melee_weapon", "armor"):
        assert f'id="g-{good}"' in html
    assert "Guía de crafting" in json.dumps(
        json.loads(html.split("window.STR=")[1].split(";</script>")[0]), ensure_ascii=False
    )


def test_an_experiment_2_replay_shows_its_budget(tmp_path: Path) -> None:
    """The budget bar reads the cap the log records, or it never appears."""
    from rpg_economy.dashboard.replay import build_track

    config = short_config(
        run={"experiment": "experiment_2", "condition": "B"},
        budget={"experiment_2_total_tokens": 40_000},
    )
    result = run_condition(config, tmp_path / "run.jsonl")
    assert build_track(result.log_path).budget_total == 40_000


def test_the_replay_has_no_frame_past_the_last_tick_that_ran(tmp_path: Path) -> None:
    """run_end is stamped one past the last tick; it must not add an empty tick."""
    from rpg_economy.dashboard.replay import build_track

    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl")
    track = build_track(result.log_path)
    assert track.horizon == config.time.horizon_ticks - 1
    assert len(track.frames) == config.time.horizon_ticks
    assert track.frames[-1].cycle == config.time.total_cycles - 1


def test_role_gold_ends_at_the_logged_final_holdings(tmp_path: Path) -> None:
    """Walked back from the run_end holdings, so the last frame must equal them."""
    from collections import defaultdict
    from decimal import Decimal

    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path)
    for condition, result in trio.items():
        run_end = next(e for e in reversed(_events(result.log_path)) if e["type"] == "run_end")
        expected: dict[str, Decimal] = defaultdict(Decimal)
        for holding in run_end["payload"]["holdings"].values():
            expected[holding["role"]] += Decimal(holding["gold"])

        frames = build_track(result.log_path).frames
        final = frames[-1]
        assert final.role_gold == {
            role: float(total.quantize(Decimal("0.01"))) for role, total in sorted(expected.items())
        }, condition.value
        assert final.to_payload()["rg"] == final.role_gold
        assert all(set(frame.role_gold) == set(expected) for frame in frames)


def test_role_gold_moves_only_with_loot_taxes_and_storage_in_total(tmp_path: Path) -> None:
    """Trades move gold between roles; only loot, confiscation pay, taxes and storage change the total."""
    import itertools

    from rpg_economy.dashboard.replay import build_track

    result = run_condition(short_config(run={"condition": "B"}), tmp_path / "B.jsonl")
    events = _events(result.log_path)
    frames = build_track(result.log_path).frames

    for previous, frame in itertools.pairwise(frames):
        flow = sum(
            float(e["payload"].get("gold", 0))
            for e in events
            if e["type"] in ("expedition_resolved", "goods_confiscated")
            and e["tick"] == frame.tick
        ) - sum(
            float(e["payload"]["amount"])
            for e in events
            if e["type"] in ("tax_paid", "storage_paid") and e["tick"] == frame.tick
        )
        change = sum(frame.role_gold.values()) - sum(previous.role_gold.values())
        assert change == pytest.approx(flow, abs=0.03), frame.tick


def test_a_step_lists_the_orders_of_a_place_orders(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import build_track

    runs = _trio(tmp_path)
    steps = [
        s
        for result in runs.values()
        for f in build_track(result.log_path).frames
        for s in f.steps
        if s.get("d") and s["d"].get("ac") == "place_orders"
    ]
    assert steps, "the rule-based agents place several orders at once"
    for step in steps:
        orders = step["d"]["os"]
        assert 2 <= len(orders) <= short_config().market.max_orders_per_activation
        assert all(order[0] in ("buy", "sell", "cancel_order") for order in orders)


def test_the_replay_shows_the_taxes(tmp_path: Path) -> None:
    """Taxes collected per kind, per agent, on the feed and on the timeline."""
    from rpg_economy.dashboard.replay import AGENT_FIELDS, build_track

    config = short_config(
        run={"condition": "B"},
        economy={"taxes": {"wealth_every_ticks": 5, "wealth_exemption": 50}},
    )
    result = run_condition(config, tmp_path / "B.jsonl")
    events = _events(result.log_path)
    track = build_track(result.log_path)
    paid = [e for e in events if e["type"] == "tax_paid"]
    assert paid, "a run with taxes configured collected none"

    last = track.frames[-1]
    for kind in ("sales", "wealth"):
        expected = sum(float(e["payload"]["amount"]) for e in paid if e["payload"]["tax"] == kind)
        assert last.charges.get(kind, 0) == pytest.approx(expected, abs=0.02)

    taxes = AGENT_FIELDS.index("taxes")
    for agent_id in {e["agent_id"] for e in paid}:
        expected = sum(float(e["payload"]["amount"]) for e in paid if e["agent_id"] == agent_id)
        assert last.agents[agent_id][taxes] == pytest.approx(expected, abs=0.02)

    wealth = [e for e in paid if e["payload"]["tax"] == "wealth"]
    # Arrears settled between collections are payments, not collections.
    collections = sorted({e["tick"] for e in wealth if not e["payload"].get("arrears")})
    assert track.wealth_tax_ticks == collections
    paying_ticks = sorted({e["tick"] for e in wealth})
    for tick in paying_ticks:
        (line,) = [f for f in track.frames[tick].feed if f["e"] == "tax_paid"]
        payers = [e for e in wealth if e["tick"] == tick]
        # The collection and the arrears settled at the same tick stay apart.
        current = [e for e in payers if not e["payload"].get("arrears")]
        owed = [e for e in payers if e["payload"].get("arrears")]
        assert line["c"] == (1 if current else 0)
        assert line["q"] == sum(1 for e in current if float(e["payload"]["amount"]) > 0)
        assert line["aq"] == sum(1 for e in owed if float(e["payload"]["amount"]) > 0)
        assert line["v"] == pytest.approx(
            sum(float(e["payload"]["amount"]) for e in current), abs=0.02
        )
        assert line["ar"] == pytest.approx(
            sum(float(e["payload"]["amount"]) for e in owed), abs=0.02
        )
        # The collection card: who paid, on what, how much, and whether it was arrears.
        rows = list(track.frames[tick].wealth_tax)
        assert sorted(row[0] for row in rows) == sorted(e["agent_id"] for e in payers)
        for e in payers:
            arrears = 1 if e["payload"].get("arrears") else 0
            (row,) = [r for r in rows if r[0] == e["agent_id"] and r[3] == arrears]
            if "taxable" in e["payload"]:
                assert row[1] == pytest.approx(float(e["payload"]["taxable"]))
            else:
                assert row[1] is None
            assert row[2] == pytest.approx(float(e["payload"]["amount"]))
    assert not any(f.wealth_tax for f in track.frames if f.tick not in paying_ticks)

    # A taxed trade carries its tax on the feed, unless the per-tick cap
    # dropped its line (the tick then says "more").
    for tick in {e["tick"] for e in paid if e["payload"]["tax"] == "sales"}:
        feed = track.frames[tick].feed
        assert any(f.get("tx") for f in feed) or any(f["e"] == "more" for f in feed)


def test_the_replay_carries_the_best_bid_and_ask(tmp_path: Path) -> None:
    """Quotes are rebuilt from the order events: the book at the end of each tick."""
    from rpg_economy.dashboard.replay import build_track

    result = run_condition(short_config(run={"condition": "B"}), tmp_path / "B.jsonl")
    events = _events(result.log_path)
    frames = build_track(result.log_path).frames

    closing = {"order_filled", "order_expired", "order_cancelled"}
    for frame in frames:
        book: dict = {}
        for e in events:
            if e["tick"] > frame.tick:
                break
            p = e["payload"]
            if e["type"] == "order_placed":
                book[p["order_id"]] = [
                    p["good"],
                    p["side"],
                    float(p["limit_price"]),
                    p["remaining"],
                ]
            elif e["type"] == "order_partially_filled" and p["order_id"] in book:
                book[p["order_id"]][3] = p["remaining"]
            elif e["type"] in closing:
                book.pop(p["order_id"], None)
        for good, (bid, ask) in frame.quotes.items():
            bids = [o[2] for o in book.values() if o[0] == good and o[1] == "buy" and o[3]]
            asks = [o[2] for o in book.values() if o[0] == good and o[1] == "sell" and o[3]]
            assert bid == (max(bids) if bids else None)
            assert ask == (min(asks) if asks else None)
            if bid is not None and ask is not None:
                assert bid < ask, "a crossed book would have traded"
    assert any(frame.quotes for frame in frames)


# ---------------------------------------------------------------------------
# Wealth-tax schedule, per-agent gold, the mock caveat, locale-aware numbers
# ---------------------------------------------------------------------------
def _taxed_trio(tmp_path: Path, **taxes):
    config = short_config(economy={"taxes": {"wealth_every_ticks": 5, **taxes}})
    run_trio(config, tmp_path, seed=7)
    return config


def test_the_wealth_tax_schedule_covers_every_collection_that_paid(tmp_path: Path) -> None:
    """The schedule is the engine's rule; every tick at which somebody paid is on it."""
    from rpg_economy.dashboard.replay import build_bundle, trio_logs, world_rules

    config = _taxed_trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    rules = world_rules(bundle, config)
    assert rules is not None
    wealth = rules["wealth"]
    assert wealth["every"] == 5
    assert wealth["exemption"] == str(config.economy.taxes.wealth_exemption)
    assert wealth["ticks"] and all(0 <= t < config.time.horizon_ticks for t in wealth["ticks"])
    for track in bundle.tracks:
        assert set(track.wealth_tax_ticks) <= set(wealth["ticks"]), track.condition


def test_a_collection_nobody_pays_is_still_scheduled(tmp_path: Path) -> None:
    """With nobody above the minimum there are no payments, but still a schedule."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs, world_rules

    config = _taxed_trio(tmp_path / "runs", wealth_exemption=10_000_000)
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    assert all(not track.wealth_tax_ticks for track in bundle.tracks)
    bundle.rules = world_rules(bundle, config)
    assert bundle.rules is not None and bundle.rules["wealth"]["ticks"]

    html = render_replay(bundle, locale="en")
    data = json.loads(html.split("window.DATA=")[1].split(";window.STR=")[0])
    assert data["rules"]["wealth"]["ticks"] == bundle.rules["wealth"]["ticks"]
    # Nothing to jump to until the page has worked out the next collection.
    assert '<button id="taxjump" type="button" disabled>' in html


def test_no_schedule_without_a_wealth_tax(tmp_path: Path) -> None:
    from rpg_economy.dashboard.replay import wealth_tax_schedule

    assert wealth_tax_schedule(short_config(economy={"taxes": {"wealth_rate": 0}})) == []


def test_token_buckets_render_in_a_fixed_order(tmp_path: Path) -> None:
    """Agent, Director, P2P content, bulletin: never the alphabetical order of the keys."""
    config = short_config(run={"condition": "C"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    data = build_dashboard(result.log_path)
    data.tokens = {
        "total": {"bulletin": 1, "p2p_content": 2, "director": 3, "agent": 4},
        "authoritative_total": 10,
    }

    text = render_dashboard(data, locale="en")
    section = text[text.index("Tokens by bucket") :]
    labels = ("  Agents: 4", "  Director: 3", "  P2P content: 2", "  Bulletin: 1")
    positions = [section.index(label) for label in labels]
    assert positions == sorted(positions)


def test_the_tax_card_counts_only_agents_below_the_exemption() -> None:
    """Agents that could not pay or only settled arrears are not below the exempt amount."""
    script = (DASHBOARD / "assets" / "replay.js").read_text(encoding="utf-8")
    assert "tr.agentIds.length - current.length" not in script
    assert "others: below" in script
    # A collection nobody paid is worded as such in the feed, not "collected 0".
    assert "if (e.c && !e.q)" in script
    # The budget share does not shadow the pct() helper.
    assert "var pct =" not in script and "var share =" in script


def test_the_wealth_summary_takes_the_exemption_from_the_config() -> None:
    """The exempt minimum is configuration, so the page never states it as a literal."""
    from rpg_economy.i18n import Catalog

    for locale in ("en", "es"):
        catalog = Catalog.load(locale)
        for key in ("replay.wealth.summary", "replay.wealth.nobody"):
            assert "{exemption}" in catalog.raw(key)
            assert "100" not in catalog.raw(key)


def test_agent_gold_is_reconstructed_for_every_agent_and_tick(tmp_path: Path) -> None:
    """Per-agent gold ends at the logged holdings and adds up to the role totals."""
    from decimal import Decimal

    from rpg_economy.dashboard.replay import build_track

    trio = _trio(tmp_path)
    for condition, result in trio.items():
        run_end = next(e for e in reversed(_events(result.log_path)) if e["type"] == "run_end")
        holdings = run_end["payload"]["holdings"]
        track = build_track(result.log_path)
        assert set(holdings) <= set(track.agent_ids), condition.value

        final = track.frames[-1]
        by_id = dict(zip(track.agent_ids, final.agent_gold, strict=True))
        for agent, holding in holdings.items():
            assert by_id[agent] == float(Decimal(holding["gold"])), (condition.value, agent)

        for frame in track.frames:
            assert len(frame.agent_gold) == len(track.agent_ids)
            assert frame.to_payload()["ag"] == frame.agent_gold
            totals: dict[str, float] = {}
            for agent, gold in zip(track.agent_ids, frame.agent_gold, strict=True):
                role = holdings[agent]["role"]
                totals[role] = totals.get(role, 0.0) + gold
            for role, total in totals.items():
                assert total == pytest.approx(frame.role_gold[role], abs=0.01)


def test_a_mock_answered_replay_says_so(tmp_path: Path) -> None:
    """A replay of mock runs carries the caveat banner even when nobody gave one."""
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs
    from rpg_economy.i18n import t

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    assert all(track.model_observed.startswith("mock:") for track in bundle.tracks)
    assert bundle.mock

    for locale in ("en", "es"):
        html = render_replay(bundle, locale=locale)
        assert 'class="banner"' in html
        assert t("replay.banner.mock", locale=locale) in html

    for track in bundle.tracks:
        track.model_observed = "gpt-5-nano"
    assert not bundle.mock
    assert 'class="banner"' not in render_replay(bundle, locale="en")


def test_the_text_view_formats_numbers_for_its_language(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    data = build_dashboard(result.log_path)
    total = data.tokens.get("authoritative_total", 0)
    assert total >= 1000, "the fixture needs a total with a thousands separator"

    english = render_dashboard(data, locale="en")
    spanish = render_dashboard(data, locale="es")
    assert f"{total:,}" in english
    assert f"{total:,}".replace(",", ".") in spanish
    gold = next(iter(data.holdings.values()))["gold"]
    whole, _, cents = f"{float(gold):.2f}".partition(".")
    assert f"{int(whole):,}.{cents}" in english
    assert f"{int(whole):,}".replace(",", ".") + f",{cents}" in spanish


def test_the_page_script_handles_the_reported_interactions() -> None:
    """Static checks on the player script for behaviour a browser test would need."""
    script = (DASHBOARD / "assets" / "replay.js").read_text(encoding="utf-8")
    # Tax lines drawn in the tax colour.
    assert '"panel", "tax"]' in script
    # Space on a summary, link or tab is left to that control.
    assert 'closest("button,summary,a,[role=button],[role=tab]")' in script
    # Home and End do not also scroll the page.
    assert 'e.key === "Home") { e.preventDefault();' in script
    assert 'e.key === "End") { e.preventDefault();' in script
    # The tax jump is disabled after the last collection.
    assert '$("#taxjump").disabled = nextTaxTick() === null' in script
    # A crossed book falls back to the last trade.
    assert "bid >= ask) return m" in script
    # Tabs name their panel.
    assert 'setAttribute("aria-controls"' in script and '"tabpanel"' in script


def test_agent_items_are_reconstructed_for_every_agent_and_tick(tmp_path: Path) -> None:
    """Per-agent units end at the logged holdings, add up to the world stock, and
    match what each agent was shown whenever nothing moved its stock earlier that tick."""
    from rpg_economy.dashboard.replay import build_track

    moving = {
        "trade_executed",
        "expedition_resolved",
        "production_completed",
        "consumption",
        "equipment_equipped",
        "goods_confiscated",
    }
    trio = _trio(tmp_path)
    for condition, result in trio.items():
        events = _events(result.log_path)
        holdings = next(e for e in reversed(events) if e["type"] == "run_end")["payload"][
            "holdings"
        ]
        track = build_track(result.log_path)
        final = dict(zip(track.agent_ids, track.frames[-1].agent_items, strict=True))
        for agent, holding in holdings.items():
            assert final[agent] == sum(holding["inventory"].values()), (condition.value, agent)
        for frame in track.frames:
            assert frame.to_payload()["ai"] == frame.agent_items
            assert sum(frame.agent_items) == sum(frame.world_stock.values())
            assert min(frame.agent_items) >= 0

        index = {agent: k for k, agent in enumerate(track.agent_ids)}
        touched: set[str] = set()
        tick = -1
        checked = 0
        for event in events:
            if event["tick"] != tick:
                tick, touched = event["tick"], set()
            payload = event["payload"]
            if event["type"] in moving:
                touched |= {
                    event.get("agent_id"),
                    payload.get("buyer_id"),
                    payload.get("seller_id"),
                }
            elif event["type"] == "observation_delivered" and tick > 0:
                agent = event["agent_id"]
                if agent in touched:
                    continue
                shown = sum((payload["observation"].get("inventory") or {}).values())
                assert track.frames[tick - 1].agent_items[index[agent]] == shown, (tick, agent)
                checked += 1
        assert checked, condition.value


def test_rejections_are_labelled_by_code(tmp_path: Path) -> None:
    """A rejected action carries the engine's code, which the page translates."""
    from rpg_economy.dashboard.html import REJECTION_KEYS
    from rpg_economy.dashboard.replay import build_track
    from rpg_economy.domain import RejectionCode
    from rpg_economy.i18n import Catalog

    assert {code.value for code in RejectionCode} <= set(REJECTION_KEYS)
    assert "resting" in REJECTION_KEYS
    for locale in ("en", "es"):
        catalog = Catalog.load(locale)
        for code in REJECTION_KEYS:
            assert catalog.raw(f"rejection.{code}")

    trio = _trio(tmp_path)
    for result in trio.values():
        rejected = [e for e in _events(result.log_path) if e["type"] == "action_rejected"]
        shown = [
            entry[1]
            for frame in build_track(result.log_path).frames
            for step in frame.steps
            for entry in step["e"]
            if entry[0] == "rejected"
        ]
        assert shown == [e["payload"].get("code") or e["payload"]["reason"] for e in rejected]


def test_the_replay_page_is_byte_reproducible(tmp_path: Path) -> None:
    """The template table is emitted in key order, whatever the hash seed."""
    import re

    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    html = render_replay(bundle, locale="es")
    strings = json.loads(re.search(r"window\.STR=(\{.*?\});</script>", html).group(1))
    assert list(strings["tpl"]) == sorted(strings["tpl"])
    assert render_replay(bundle, locale="es") == html


def test_a_two_model_header_joins_the_models_in_the_page_language(tmp_path: Path) -> None:
    from rpg_economy.dashboard.html import render_replay
    from rpg_economy.dashboard.replay import build_bundle, trio_logs
    from rpg_economy.i18n import t

    _trio(tmp_path / "runs")
    bundle = build_bundle(trio_logs(tmp_path / "runs"))
    bundle.tracks[0].model_observed = "model-one"
    bundle.tracks[1].model_observed = "model-two"
    for locale in ("en", "es"):
        joined = t("replay.meta.versus", locale=locale).join(["model-one", "model-two"])
        assert joined in render_replay(bundle, locale=locale)


def test_the_text_view_names_the_triggers(tmp_path: Path) -> None:
    from rpg_economy.i18n import t

    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    data = build_dashboard(result.log_path)
    assert data.trigger_counts
    for locale in ("en", "es"):
        text = render_dashboard(data, locale=locale)
        for trigger in data.trigger_counts:
            assert t(f"trigger.{trigger}", locale=locale) in text


def test_the_text_view_sorts_prices_by_the_localized_label() -> None:
    """Each language lists the goods in its own alphabetical order."""
    from decimal import Decimal

    from rpg_economy.dashboard.view import DashboardData, PricePoint
    from rpg_economy.i18n import t

    goods = ("iron", "wood", "leather", "herbs", "arcane_crystal", "mana_potion", "armor")
    data = DashboardData(
        run_id="r",
        condition="B",
        scenario="resource_scarcity",
        experiment="experiment_1",
        seed=1,
        ticks=1,
    )
    for good in goods:
        data.prices[good] = [PricePoint(tick=0, good=good, price=Decimal("1"), quantity=1)]
    for locale in ("en", "es"):
        text = render_dashboard(data, locale=locale)
        section = text.split(t("dashboard.section.prices", locale=locale), 1)[1]
        section = section.split("\n\n", 1)[0]
        rows = [line.strip().split(":", 1)[0] for line in section.splitlines() if line.strip()]
        assert len(rows) == len(goods)
        assert rows == sorted(rows, key=lambda label: label.casefold().replace("ó", "o")), locale


def test_the_page_script_keeps_focus_and_links(tmp_path: Path) -> None:
    """Static checks for keyboard focus, table semantics and deep links."""
    script = (DASHBOARD / "assets" / "replay.js").read_text(encoding="utf-8")
    # Re-rendered controls carry a stable key and get focus back.
    assert 'setAttribute("data-key", "step:"' in script
    assert 'setAttribute("data-key", "agent:"' in script
    assert "again.focus()" in script
    # Agent rows keep table semantics: a real button, not a role=button row.
    assert "activatable(row" not in script
    assert 'el("button", "agentbtn"' in script
    # Guide cards point at their text.
    assert "aria-describedby" in script
    # Axis labels are formatted for the page's language.
    assert "toFixed(opts.integer" not in script and "num(v, decimals)" in script
    # Gridlines sit on round 1/2/5 x 10^n steps.
    assert "function niceStep(raw)" in script and "niceStep((hi - lo) / 4)" in script
    # The current-value dot does not depend on the line style.
    assert "!l.dash" not in script and "l.dot" in script
    # The link carries the tab and the followed agent, and a bad tab is ignored.
    assert '"tab=" +' in script and '"agent=" +' in script
    assert "TABS.some(" in script
    # A link with no agent clears whatever was being followed.
    assert "tr.ui.focus = q.agent && tr.agentIds.indexOf(q.agent) >= 0 ? q.agent : null" in script
    # The protocol label goes through the catalog, not a literal.
    assert '"RESULT ' not in script and "replay.result_status" in script
    # The wealth card names who could not pay, in the singular too.
    assert "replay.wealth.summary_unpaid" in script
    assert "replay.wealth.nobody_unpaid_one" in script
    # Codes are translated; a shock on every agent says so.
    assert "reasonL(e[1])" in script and "replay.shock.liquidity_loss_all" in script


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"reason": "no JSON object found in completion: ''", "truncated": True}, "truncated"),
        ({"reason": "llm error: no completion"}, "llm_error"),
        ({"reason": "no JSON object found in completion: 'hi'"}, "invalid_json"),
        ({"reason": "invalid RESULT fields: bad"}, "result_fields"),
        ({"reason": "action failed schema validation: x"}, "schema"),
    ],
)
def test_invalid_outputs_get_a_kind_the_page_can_label(payload: dict, kind: str) -> None:
    from rpg_economy.dashboard.replay import _invalid_kind
    from rpg_economy.i18n import Catalog

    assert _invalid_kind(payload) == kind
    for locale in ("en", "es"):
        assert Catalog.load(locale).raw(f"replay.invalid.{kind}")


def test_the_check_command_summarizes_a_trio(tmp_path: Path, capsys) -> None:
    from rpg_economy.cli import main

    _trio(tmp_path / "runs")
    assert main(["check", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "condition A" in out and "condition B" in out and "condition C" in out
    assert "Director tasks to" in out
    assert main(["check", str(tmp_path / "empty")]) == 2
