"""The paper figures: one pass over each log, every figure in every language."""

from __future__ import annotations

import json
from pathlib import Path

from rpg_economy.analysis.paper_figures import FORMATS, extract, generate, gini
from rpg_economy.i18n import SUPPORTED_LOCALES, get_catalog, set_locale
from rpg_economy.runner import run_trio
from tests.conftest import make_config


def _trio(tmp_path: Path):
    config = make_config(
        phases={"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 30},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    run_trio(config, tmp_path / "trio", seed=1001)
    logs = sorted((tmp_path / "trio").rglob("*.jsonl"))
    assert len(logs) == 3
    return config, logs


def test_gini_is_zero_when_equal_and_grows_with_concentration() -> None:
    assert gini([5.0, 5.0, 5.0]) == 0
    assert gini([0.0, 0.0, 9.0]) > gini([2.0, 3.0, 4.0])
    assert gini([1.0]) is None


def test_extract_reads_condition_seed_and_series(tmp_path: Path) -> None:
    config, logs = _trio(tmp_path)
    runs = [
        extract(
            path, config.phases.shock_tick, config.phases.reversion_tick, config.time.horizon_ticks
        )
        for path in logs
    ]
    assert sorted(r.condition for r in runs) == ["A", "B", "C"]
    for run in runs:
        assert run.seed == 1001
        assert len(run.units) == run.windows
        assert all(len(series) == run.windows for series in run.price.values())
        assert run.activations >= run.wasted


def test_generate_writes_every_figure_in_every_language(tmp_path: Path) -> None:
    config, logs = _trio(tmp_path)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"runs": []}), encoding="utf-8")
    set_locale("en")

    written = generate(logs, results, config, tmp_path / "paper", workers=1)

    assert get_catalog().locale == "en"  # rendering must not change the caller's language
    assert set(written) == set(SUPPORTED_LOCALES)
    counts = {locale: len(paths) for locale, paths in written.items()}
    assert len(set(counts.values())) == 1
    for paths in written.values():
        assert paths and all(p.is_file() and p.stat().st_size > 0 for p in paths)
        assert {p.suffix.lstrip(".") for p in paths} == set(FORMATS)
