"""Canonical hashing of configuration and other structured payloads.

The config hash identifies the exact parameter set a run executed under. Design
v1.0 section 15 requires it in every run's metadata, so that any run can be
traced to the configuration that produced it.

The hash must depend on *values only*, never on incidental serialization detail:
YAML mapping order, indentation, or comments must not change it, or a
cosmetically reformatted config would look like a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CANONICAL_ENCODING = "utf-8"


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` to a canonical JSON string.

    Mapping keys are sorted, separators are fixed, and non-ASCII characters are
    escaped, so that two structurally equal payloads always produce identical
    bytes regardless of how they were constructed or parsed.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    """Return the SHA-256 hex digest of ``payload``'s canonical serialization."""
    return hashlib.sha256(canonical_json(payload).encode(CANONICAL_ENCODING)).hexdigest()
