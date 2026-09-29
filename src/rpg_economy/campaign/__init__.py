"""Pilot and final run orchestration (design v1.0 s10, s11.4; IMPLEMENTATION.md phase 7).

The campaign layer decides *which* runs happen. It never decides anything about
how a run behaves: it composes conditions, scenarios, experiments and seeds into
a plan, prices that plan, and executes it. Nothing here may influence a run's
outcome.
"""
