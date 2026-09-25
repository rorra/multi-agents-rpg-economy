"""Errors the analysis raises, each naming the catalog message that explains it.

The English text stays the exception's message, for library callers and logs;
``key`` and ``params`` let the CLI print the same refusal in the reader's
language through ``t(err.key, **err.params)``.
"""

from __future__ import annotations

from typing import Any


class AnalysisError(Exception):
    """A log, or a set of logs, that cannot enter the analysis.

    ``key`` names the catalog message and ``params`` fill it; the English
    message is the ``str`` of the exception.
    """

    def __init__(self, message: str, *, key: str | None = None, **params: Any) -> None:
        super().__init__(message)
        self.key = key
        self.params = params
