"""Run metadata construction shared by the CLI and the run harness."""

from __future__ import annotations

from rpg_economy.config import config_hash, setup_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.logging_ import Environment, RunMetadata, code_digest


def build_metadata(config: SimulationConfig) -> RunMetadata:
    """Assemble the reproducibility record for a run of ``config``."""
    digest = config_hash(config)
    # The batch the run belongs to (pilot, final or ad hoc). Outside the config
    # hash, so the metadata is where it is recorded.
    phase = getattr(config.run, "phase", None)
    return RunMetadata(
        run_id=RunMetadata.make_run_id(
            config_hash=digest,
            condition=config.run.condition,
            scenario=config.run.scenario,
            experiment=config.run.experiment,
            seed=config.run.seed,
        ),
        design_version=config.design_version,
        schema_version=config.schema_version,
        condition=config.run.condition,
        scenario=config.run.scenario,
        experiment=config.run.experiment,
        seed=config.run.seed,
        config_hash=digest,
        setup_hash=setup_hash(config),
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        llm_params={
            "temperature": str(config.llm.temperature),
            # Fixed and load-bearing: at the provider default this model can
            # spend its whole completion budget reasoning and answer nothing, so
            # a log that omits it cannot explain its own run (design v1.0 s13).
            "reasoning_effort": str(config.llm.reasoning_effort),
            "max_output_tokens": str(config.llm.max_output_tokens),
            "seed": str(config.llm.seed),
            "model_snapshot": str(config.llm.model_snapshot),
        },
        llm_mock=config.llm.mock,
        phase=None if phase is None else str(getattr(phase, "value", phase)),
        environment=Environment.capture(),
        # The package source that wrote the run, so a resume or a budget
        # derivation can tell logs written by other code.
        code_digest=code_digest(),
    )
