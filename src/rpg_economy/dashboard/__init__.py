"""Read-only run observation, fed exclusively from logs (IMPLEMENTATION.md phase 7).

The dashboard is an observation tool and is never part of the experiment. It
imports nothing from the engine, the orchestration layer or the LLM client, and
it opens every log file read-only. Its structural guarantee is the import graph:
a module that cannot reach the engine cannot perturb a run, whatever it does.
"""
