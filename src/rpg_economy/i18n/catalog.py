"""Message catalogs and the ``t()`` helper.

Every user-facing string -- CLI output, report text, figure titles and axis
labels, dashboard text -- resolves through this module (IMPLEMENTATION.md section 2).
Stored data never contains a translated string: logs and metrics files hold
language-neutral keys and enum values, and language is applied at render time.

Locale is chosen by, in order: an explicit argument, ``SIM_LOCALE``, then the
default ``en``.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "es")
LOCALE_ENV_VAR = "SIM_LOCALE"

#: ``{placeholder}`` occurrences, ignoring ``{{`` escapes.
PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def locales_dir() -> Path:
    """Return the repository's ``locales/`` directory."""
    return Path(__file__).resolve().parents[3] / "locales"


class MissingMessageError(KeyError):
    """Raised when a message key is absent from the catalog.

    Deliberately fatal rather than falling back to the key: untranslated text
    must never end up in a report or a figure silently.
    """


class UnsupportedLocaleError(ValueError):
    """Raised for a locale with no catalog."""


class Catalog:
    """An immutable message catalog for one locale."""

    def __init__(self, locale: str, messages: dict[str, str]) -> None:
        self.locale = locale
        self._messages = dict(messages)

    @classmethod
    def load(cls, locale: str, directory: Path | None = None) -> Catalog:
        if locale not in SUPPORTED_LOCALES:
            raise UnsupportedLocaleError(_unsupported_message(locale))
        path = (directory or locales_dir()) / f"{locale}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise UnsupportedLocaleError(f"cannot read catalog {path}: {exc}") from exc
        if not isinstance(raw, dict) or not all(isinstance(v, str) for v in raw.values()):
            raise UnsupportedLocaleError(f"catalog {path} must map string keys to strings")
        return cls(locale, raw)

    def message_keys(self) -> frozenset[str]:
        return frozenset(self._messages)

    def raw(self, key: str) -> str:
        try:
            return self._messages[key]
        except KeyError:
            raise MissingMessageError(
                f"message key {key!r} is missing from the {self.locale!r} catalog"
            ) from None

    def t(self, key: str, **kwargs: Any) -> str:
        """Render ``key`` in this locale, substituting ``kwargs``."""
        template = self.raw(key)
        required = set(PLACEHOLDER_RE.findall(template))
        missing = required - set(kwargs)
        if missing:
            raise MissingMessageError(
                f"message {key!r} requires placeholders {sorted(missing)} which were not supplied"
            )
        return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Process-wide active locale
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict[str, Catalog] = {}
_active_locale: str | None = None


def resolve_locale(explicit: str | None = None) -> str:
    """Resolve the locale from an explicit value, the environment, or the default."""
    candidate = explicit or os.environ.get(LOCALE_ENV_VAR) or DEFAULT_LOCALE
    if candidate not in SUPPORTED_LOCALES:
        raise UnsupportedLocaleError(_unsupported_message(candidate))
    return candidate


def _unsupported_message(requested: str) -> str:
    """The unsupported-locale error, in the active language (or the default).

    The requested locale has no catalog by definition, so the message comes from
    the one already in use; it is read straight from disk to stay independent of
    the cache and its lock.
    """
    fallback = _active_locale if _active_locale in SUPPORTED_LOCALES else DEFAULT_LOCALE
    catalog = _cache.get(fallback) or Catalog.load(fallback)
    return catalog.t(
        "error.locale.unsupported", lang=repr(requested), supported=", ".join(SUPPORTED_LOCALES)
    )


def get_catalog(locale: str | None = None) -> Catalog:
    """Return the catalog for ``locale``, loading and caching it on first use."""
    resolved = resolve_locale(locale if locale is not None else _active_locale)
    with _lock:
        if resolved not in _cache:
            _cache[resolved] = Catalog.load(resolved)
        return _cache[resolved]


def set_locale(locale: str | None) -> str:
    """Set the process-wide active locale and return it."""
    global _active_locale
    resolved = resolve_locale(locale)
    _active_locale = resolved
    get_catalog(resolved)
    return resolved


def t(key: str, /, *, locale: str | None = None, **kwargs: Any) -> str:
    """Translate ``key`` into the active (or given) locale."""
    return get_catalog(locale).t(key, **kwargs)


def clear_cache() -> None:
    """Drop cached catalogs. Intended for tests."""
    global _active_locale
    with _lock:
        _cache.clear()
        _active_locale = None
