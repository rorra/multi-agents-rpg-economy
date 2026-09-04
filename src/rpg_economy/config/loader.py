"""Configuration loading, layering and hashing.

A run's configuration is assembled from a base YAML file plus optional scenario
and per-run override layers, validated once, and then hashed. The hash is what
run metadata records (design v1.0 s15), so every run says exactly which
configuration produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from rpg_economy.config.hashing import canonical_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ExperimentRegime


class ConfigError(Exception):
    """Raised when a configuration cannot be loaded, merged or validated."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new mapping.

    Nested mappings merge key by key; every other value, lists included, is
    replaced wholesale. Replacing rather than concatenating lists is deliberate:
    an override that means to shorten a list must be able to.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    base_path: Path | str,
    *,
    layers: list[Path | str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> SimulationConfig:
    """Load, layer and validate a configuration.

    Args:
        base_path: the base YAML file, normally ``configs/base.yaml``.
        layers: further YAML files merged in order, e.g. a scenario file.
        overrides: a final in-memory mapping, used by the trio runner to set
            condition, seed and experiment without writing a file per run.
    """
    payload = _read_yaml(Path(base_path))
    for layer in layers or []:
        payload = deep_merge(payload, _read_yaml(Path(layer)))
    if overrides:
        payload = deep_merge(payload, overrides)

    try:
        return SimulationConfig.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError, re-raised with context
        sources = ", ".join(str(p) for p in [base_path, *(layers or [])])
        raise ConfigError(f"invalid configuration assembled from {sources}: {exc}") from exc


#: ``llm`` fields that say how a run is executed or priced, never what it runs.
#: None of them can change a run's behaviour, so none of them enters the hash.
EXECUTION_ONLY_LLM_FIELDS = (
    # Real provider or offline mock. Run metadata records it separately
    # (``llm_mock``), and the campaign's resume check compares it on its own.
    "mock",
    # The list price and the date it was quoted. Cost projection only; a price
    # update between a crash and its resume must not re-pay finished trios.
    "pricing",
    # Transport: how long to wait for, and how often to retry, one request. A
    # retried call is the same call, so these never reach what the model sees.
    "request_timeout_s",
    "max_transient_retries",
    "retry_base_delay_s",
    "retry_max_delay_s",
)


#: Top-level sections that choose *which* runs a campaign makes, never how any
#: one of them behaves. A run is identified by ``run`` (its seed included), so
#: extending the seed list or the campaign matrix must not re-pay finished trios.
CAMPAIGN_ONLY_SECTIONS = ("seeds", "campaign")


#: ``metrics`` fields read only by the analysis of finished logs, never while a
#: run executes. ``volatility_window_w`` is not among them: the global view the
#: Director and the bulletin receive uses it at run time.
ANALYSIS_ONLY_METRIC_FIELDS = (
    "recovery_band_pct_x",
    "recovery_hold_ticks_m",
    "useful_production_ticks_t",
    "unanswered_request_ticks_d",
)


def config_payload(config: SimulationConfig) -> dict[str, Any]:
    """Return the plain-data form of ``config`` used for hashing.

    Left out, because they say how or which runs are executed, not what any
    one run is:

    * the execution-only ``llm`` fields (``EXECUTION_ONLY_LLM_FIELDS``):
      whether a run calls the real provider or the offline mock, what the
      tokens are quoted at, and how the transport waits and retries;
    * ``seeds`` and ``campaign`` (``CAMPAIGN_ONLY_SECTIONS``): the run's own
      seed is in ``run``, and the lists around it only pick other runs;
    * the analysis-only ``metrics`` fields (``ANALYSIS_ONLY_METRIC_FIELDS``):
      the recovery band and hold, and the useful-production and
      unanswered-request windows, which only the analysis of a finished log
      reads, so re-tuning them re-analyses runs instead of re-running them;
    * ``run.phase`` (pilot, final or ad hoc): which batch a run belongs to,
      recorded in its metadata, not what the run is;
    * ``budget.experiment_2_total_tokens`` in an Experiment 1 run, which has
      no token cap and never reads it; and ``budget.derived_code_digest``,
      which records which code the budget came from. ``budget.derived_from_mock``
      stays: a mock-derived budget marks a different setup.

    Keeping any of them in would give one run several identities, and a
    resumed campaign would re-run (and re-pay for) trios that already finished
    because a price, a seed list or an unused budget was edited.
    """
    payload = config.model_dump(mode="json")
    for name in EXECUTION_ONLY_LLM_FIELDS:
        payload["llm"].pop(name, None)
    for section in CAMPAIGN_ONLY_SECTIONS:
        payload.pop(section, None)
    for name in ANALYSIS_ONLY_METRIC_FIELDS:
        payload["metrics"].pop(name, None)
    payload["run"].pop("phase", None)
    # Provenance of the budget, not part of what a run does.
    payload["budget"].pop("derived_code_digest", None)
    # ``!=``, not ``is not``: a StrEnum equals its value, should one arrive bare.
    if config.run.experiment != ExperimentRegime.EQUAL_COST:
        payload["budget"].pop("experiment_2_total_tokens", None)
    return payload


def config_hash(config: SimulationConfig) -> str:
    """Return the canonical SHA-256 hash of a validated configuration.

    The hash covers validated values, so it is invariant to YAML key order,
    comments and formatting, and it changes if and only if a parameter changes.
    """
    return canonical_hash(config_payload(config))


def setup_hash(config: SimulationConfig) -> str:
    """The hash of everything but the ``run`` section: the setup's identity.

    A configuration is one setup; each run derives from it by choosing a
    condition, scenario, experiment and seed. Those choices live in ``run`` and
    nowhere else, so leaving that section out gives one hash shared by every run
    of a setup in one experiment, and different for any run whose world, agents
    or budget differ. It follows ``config_payload``: an Experiment 1 run's
    setup does not include the Experiment 2 budget it never reads, so the
    three runs of an Experiment 1 trio agree on it whether or not a budget was
    derived between them.
    """
    payload = config_payload(config)
    payload.pop("run", None)
    return canonical_hash(payload)
