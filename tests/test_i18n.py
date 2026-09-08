"""Message catalogs, the ``t()`` helper, and the no-hardcoded-strings rule."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rpg_economy.i18n import (
    DEFAULT_LOCALE,
    LOCALE_ENV_VAR,
    SUPPORTED_LOCALES,
    Catalog,
    MissingMessageError,
    UnsupportedLocaleError,
    clear_cache,
    get_catalog,
    locales_dir,
    resolve_locale,
    set_locale,
    t,
)
from rpg_economy.i18n.catalog import PLACEHOLDER_RE

#: Anchored to this file so the scan works from any working directory.
SRC = Path(__file__).resolve().parents[1] / "src" / "rpg_economy"


@pytest.fixture(autouse=True)
def _reset_locale() -> None:
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Catalog parity -- the property that makes bilingual reports possible
# ---------------------------------------------------------------------------
def test_every_locale_has_a_catalog() -> None:
    for locale in SUPPORTED_LOCALES:
        assert (locales_dir() / f"{locale}.json").is_file()


def test_catalogs_have_identical_key_sets() -> None:
    """en and es must render from the same results file (IMPLEMENTATION.md section 2)."""
    keys = {loc: Catalog.load(loc).message_keys() for loc in SUPPORTED_LOCALES}
    reference = keys[DEFAULT_LOCALE]
    for locale, actual in keys.items():
        assert actual == reference, (
            f"{locale} catalog differs: missing {sorted(reference - actual)}, "
            f"extra {sorted(actual - reference)}"
        )


def test_catalogs_have_identical_placeholders() -> None:
    """A dropped placeholder would raise only when that message is first rendered."""
    catalogs = {loc: Catalog.load(loc) for loc in SUPPORTED_LOCALES}
    reference = catalogs[DEFAULT_LOCALE]
    for key in reference.message_keys():
        expected = set(PLACEHOLDER_RE.findall(reference.raw(key)))
        for locale, catalog in catalogs.items():
            assert set(PLACEHOLDER_RE.findall(catalog.raw(key))) == expected, (
                f"placeholder mismatch for {key!r} in {locale}"
            )


def test_no_empty_messages() -> None:
    for locale in SUPPORTED_LOCALES:
        catalog = Catalog.load(locale)
        for key in catalog.message_keys():
            assert catalog.raw(key).strip(), f"{locale}:{key} is empty"


def test_catalogs_are_valid_json_objects_of_strings() -> None:
    for locale in SUPPORTED_LOCALES:
        raw = json.loads((locales_dir() / f"{locale}.json").read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items())


def test_every_domain_enum_value_has_a_label() -> None:
    """Reports label conditions, scenarios, goods and roles by their enum value."""
    from rpg_economy.domain import (
        AdventurerSubclass,
        ArtisanSpecialization,
        Condition,
        ExperimentRegime,
        Product,
        Resource,
        Role,
        Scenario,
        TaskType,
        Trigger,
    )

    expected = {
        **{f"condition.{m.value}": m for m in Condition},
        **{f"experiment.{m.value}": m for m in ExperimentRegime},
        **{f"scenario.{m.value}": m for m in Scenario},
        **{f"resource.{m.value}": m for m in Resource},
        **{f"product.{m.value}": m for m in Product},
        **{f"role.{m.value}": m for m in Role},
        **{f"subclass.{m.value}": m for m in AdventurerSubclass},
        **{f"specialization.{m.value}": m for m in ArtisanSpecialization},
        **{f"task.{m.value}": m for m in TaskType},
        **{f"trigger.{m.value}": m for m in Trigger},
    }
    available = Catalog.load(DEFAULT_LOCALE).message_keys()
    assert not (set(expected) - available), sorted(set(expected) - available)


# ---------------------------------------------------------------------------
# t()
# ---------------------------------------------------------------------------
def test_translation_differs_between_locales() -> None:
    assert t("condition.A", locale="en") != t("condition.A", locale="es")


def test_placeholders_are_substituted() -> None:
    rendered = t("error.locale.unsupported", locale="en", lang="de", supported="en, es")
    assert "de" in rendered and "{lang}" not in rendered


def test_missing_key_raises() -> None:
    """Silent fallback would let untranslated text reach a report."""
    with pytest.raises(MissingMessageError):
        t("no.such.key", locale="en")


def test_missing_placeholder_raises() -> None:
    with pytest.raises(MissingMessageError, match="placeholders"):
        t("error.locale.unsupported", locale="en")


def test_unsupported_locale_raises() -> None:
    with pytest.raises(UnsupportedLocaleError):
        t("condition.A", locale="fr")


def test_unsupported_locale_error_is_translated() -> None:
    """The error comes from the catalog, in the language already in use."""
    set_locale("es")
    expected = Catalog.load("es").t(
        "error.locale.unsupported", lang="'fr'", supported=", ".join(SUPPORTED_LOCALES)
    )
    with pytest.raises(UnsupportedLocaleError) as raised:
        resolve_locale("fr")
    assert str(raised.value) == expected


# ---------------------------------------------------------------------------
# Locale resolution
# ---------------------------------------------------------------------------
def test_default_locale_is_english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOCALE_ENV_VAR, raising=False)
    assert resolve_locale() == "en"


def test_env_var_selects_the_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCALE_ENV_VAR, "es")
    assert resolve_locale() == "es"


def test_explicit_argument_beats_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCALE_ENV_VAR, "es")
    assert resolve_locale("en") == "en"


def test_set_locale_changes_the_active_catalog() -> None:
    set_locale("es")
    assert get_catalog().locale == "es"
    assert t("condition.B") == Catalog.load("es").raw("condition.B")


def test_unsupported_env_locale_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCALE_ENV_VAR, "de")
    with pytest.raises(UnsupportedLocaleError):
        resolve_locale()


# ---------------------------------------------------------------------------
# No hardcoded user-facing strings (IMPLEMENTATION.md sections 2 and 6)
# ---------------------------------------------------------------------------
#: Calls whose text reaches a user and must therefore be translated.
USER_FACING_CALLS = frozenset(
    {
        "print",
        "set_title",
        "set_xlabel",
        "set_ylabel",
        "set_label",
        "suptitle",
        "title",
        "xlabel",
        "ylabel",
        "legend",
        "annotate",
        "set_suptitle",
    }
)


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _string_literal_args(node: ast.Call) -> list[str]:
    literals = [
        arg.value
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    literals += [
        kw.value.value
        for kw in node.keywords
        if kw.arg in {"label", "title", "help", "description"}
        and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, str)
    ]
    return literals


def test_no_hardcoded_user_facing_strings() -> None:
    """User-facing text must resolve through ``t()``, never appear as a literal.

    Catches the realistic regression: a ``print`` or a matplotlib label written
    directly in English, which would silently bypass the Spanish report.
    """
    violations: list[str] = []
    scanned = sorted(SRC.rglob("*.py"))
    # A wrong root would find no files and pass vacuously.
    assert scanned, f"no source files found under {SRC}"
    for path in scanned:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in USER_FACING_CALLS:
                continue
            for literal in _string_literal_args(node):
                if literal.strip():
                    violations.append(f"{path}:{node.lineno}: {name}({literal!r})")

    assert not violations, "hardcoded user-facing strings:\n" + "\n".join(violations)


def test_the_hardcoded_string_check_actually_detects_a_violation(tmp_path: Path) -> None:
    """Guards the guard: an always-passing lint would be worse than none."""
    offender = tmp_path / "bad.py"
    offender.write_text("print('Run finished')\n", encoding="utf-8")

    tree = ast.parse(offender.read_text(encoding="utf-8"))
    found = [
        literal
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) in USER_FACING_CALLS
        for literal in _string_literal_args(node)
    ]
    assert found == ["Run finished"]


@pytest.mark.invariant
def test_no_message_uses_locale_as_a_placeholder() -> None:
    """``locale`` is reserved by ``t()`` for its own override.

    ``t(key, locale=...)`` selects the catalog; it never reaches the message. A
    message with a ``{locale}`` placeholder is therefore unfillable, and would
    only fail at render time.
    """
    offenders: list[str] = []
    for locale in SUPPORTED_LOCALES:
        catalog = json.loads((locales_dir() / f"{locale}.json").read_text(encoding="utf-8"))
        for key, message in catalog.items():
            if "locale" in PLACEHOLDER_RE.findall(message):
                offenders.append(f"{locale}: {key}")
    assert not offenders, "messages with the reserved 'locale' placeholder:\n" + "\n".join(
        offenders
    )


# ---------------------------------------------------------------------------
# Locale-aware numbers
# ---------------------------------------------------------------------------
def test_numbers_use_the_separators_of_their_language() -> None:
    from decimal import Decimal

    from rpg_economy.i18n import format_number

    assert format_number(1035.456, 2, locale="en") == "1,035.46"
    assert format_number(1035.456, 2, locale="es") == "1.035,46"
    assert format_number(Decimal("1234567.5"), 1, locale="es") == "1.234.567,5"
    assert format_number("12.50", 2, locale="es") == "12,50"
    assert format_number(-2100, locale="en") == "-2,100"
    assert format_number(-2100, locale="es") == "-2.100"
    assert format_number(7, locale="es") == "7"


def test_numbers_follow_the_active_locale() -> None:
    from rpg_economy.i18n import format_number

    set_locale("es")
    assert format_number(1500.5, 2) == "1.500,50"
    set_locale("en")
    assert format_number(1500.5, 2) == "1,500.50"


def test_every_locale_has_number_separators() -> None:
    from rpg_economy.i18n.numbers import SEPARATORS

    assert set(SEPARATORS) == set(SUPPORTED_LOCALES)


def test_negative_decimals_are_refused() -> None:
    from rpg_economy.i18n import format_number

    with pytest.raises(ValueError):
        format_number(1, -1, locale="en")
