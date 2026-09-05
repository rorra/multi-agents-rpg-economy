"""Construction of the initial world from configuration.

Agent identity is fixed and reproducible: the same configuration always produces
the same agents, with the same ids, roles and endowments, in the same order. This
is a control the design requires -- "mismos agentes, IDs, rasgos y dotaciones en
tríos emparejados" (design v1.0 s12) -- so conditions A, B and C differ only in
how those identical agents are coordinated.
"""

from __future__ import annotations

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Role, parse_good
from rpg_economy.engine.state import AgentAccount, WorldState


def build_world(config: SimulationConfig) -> WorldState:
    """Create the initial world state described by ``config``."""
    state = WorldState()
    population = config.population
    gold = config.economy.initial_gold
    inventory = config.economy.initial_inventory
    starting = {subclass.value: item for subclass, item in config.expeditions.starting_gear.items()}

    def add(agent_id: str, role: Role, specialization: str | None) -> None:
        try:
            starting_gold = gold[role.value]
        except KeyError:
            raise KeyError(f"economy.initial_gold has no entry for role {role.value!r}") from None
        state.accounts[agent_id] = AgentAccount(
            agent_id=agent_id,
            role=role,
            specialization=specialization,
            gold=starting_gold,
            # A specialization's own entry, when there is one, replaces its
            # role's: an artisan starts with the inputs of its own recipes.
            inventory={
                parse_good(good): units
                for good, units in sorted(
                    inventory.get(specialization or "", inventory.get(role.value, {})).items()
                )
            },
            # The item an adventurer needs to set out, worn from the start.
            equipment=(
                {starting[specialization]: config.economy.durability[starting[specialization]]}
                if role is Role.ADVENTURER and specialization in starting
                else {}
            ),
        )

    # Ids and order come from the population itself, the same source config
    # validation checks liquidity_loss.affected_agents against.
    for agent_id, role, specialization in population.members():
        add(agent_id, role, specialization)

    return state
