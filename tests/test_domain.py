"""Domain vocabulary, and its agreement with the glossary and the design document."""

from __future__ import annotations

import re

from rpg_economy.domain import (
    RESOURCE_TO_MAIN_PRODUCT,
    Condition,
    Product,
    Resource,
    Scenario,
    TaskType,
    Trigger,
)
from tests.conftest import ROOT

GLOSSARY = ROOT / "docs" / "glossary.md"


def test_design_vocabulary_is_complete() -> None:
    """The design fixes 5 resources, 6 products, 6 TASK types, 5 triggers."""
    assert len(Resource) == 5
    assert len(Product) == 6
    assert len(TaskType) == 6
    assert len(Trigger) == 5
    assert len(Condition) == 3


def test_resource_to_product_mapping_matches_the_design_table() -> None:
    """Fixed mapping from design v1.0 s11.3. Immutable once final runs begin."""
    assert RESOURCE_TO_MAIN_PRODUCT == {
        Resource.IRON: Product.MELEE_WEAPON,
        Resource.WOOD: Product.BOW_GEAR,
        Resource.LEATHER: Product.ARMOR,
        Resource.HERBS: Product.HEALTH_POTION,
        Resource.ARCANE_CRYSTAL: Product.STAFF_FOCUS,
    }


def test_every_resource_maps_to_a_main_product() -> None:
    assert set(RESOURCE_TO_MAIN_PRODUCT) == set(Resource)


def test_glossary_covers_every_enum_value() -> None:
    """The glossary must list every enum value used in logs and metrics."""
    text = GLOSSARY.read_text(encoding="utf-8")
    coded = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text))

    import enum

    import rpg_economy.domain as domain

    # Every enum the domain defines, so a new one cannot go unlisted.
    missing = []
    for name in dir(domain):
        kind = getattr(domain, name)
        if not (isinstance(kind, type) and issubclass(kind, enum.Enum)):
            continue
        if kind.__module__ != domain.__name__:
            continue
        for member in kind:
            if member.value not in coded and f"{kind.__name__}.{member.value}" not in coded:
                missing.append(f"{kind.__name__}.{member.value}")
    assert not missing, f"glossary is missing: {sorted(missing)}"


def test_enum_values_are_language_neutral_ascii() -> None:
    """Stored data must never carry a translated or accented value."""
    for member in [*Resource, *Product, *TaskType, *Scenario, *Trigger, *Condition]:
        assert member.value.isascii(), member
