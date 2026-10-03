"""Read-only run observation, fed exclusively from logs (IMPLEMENTATION.md phase 7).

The dashboard is an observation tool and is never part of the experiment. It
imports nothing from the engine, the orchestration layer or the LLM client, and
it opens every log file read-only. Its structural guarantee is the import graph:
a module that cannot reach the engine cannot perturb a run, whatever it does.
"""

from rpg_economy.dashboard.html import render_replay, write_replay
from rpg_economy.dashboard.replay import ReplayBundle, build_bundle, build_track, trio_logs
from rpg_economy.dashboard.view import (
    DashboardData,
    build_dashboard,
    render_dashboard,
    tail_events,
)

__all__ = [
    "DashboardData",
    "ReplayBundle",
    "build_bundle",
    "build_dashboard",
    "build_track",
    "render_dashboard",
    "render_replay",
    "tail_events",
    "trio_logs",
    "write_replay",
]
