"""Locale-aware number formatting for user-facing text.

Stored data holds plain numbers; the separators a reader expects are applied at
render time, like every other piece of language. English writes ``1,035.46``,
Spanish ``1.035,46``. Thousands are always grouped, four-digit numbers
included, so columns of figures line up the same in both languages (the replay
page does the same).
"""

from __future__ import annotations

from decimal import Decimal

from rpg_economy.i18n.catalog import get_catalog

#: (thousands separator, decimal separator) per locale.
SEPARATORS: dict[str, tuple[str, str]] = {"en": (",", "."), "es": (".", ",")}


def format_number(
    value: int | float | Decimal | str,
    decimals: int = 0,
    *,
    locale: str | None = None,
) -> str:
    """Format ``value`` with ``decimals`` places in the active (or given) locale.

    ``locale`` follows the same resolution as :func:`t`: an explicit value, else
    the active locale, else ``SIM_LOCALE``, else English. Strings are parsed as
    decimals, so a logged ``"12.50"`` formats without a float round trip.
    """
    if decimals < 0:
        raise ValueError("decimals must be zero or more")
    resolved = get_catalog(locale).locale
    thousands, point = SEPARATORS[resolved]
    number = Decimal(value) if isinstance(value, str) else value
    text = f"{number:,.{decimals}f}"
    # Through a placeholder, so the two swaps cannot collide.
    return text.replace(",", "\0").replace(".", point).replace("\0", thousands)
