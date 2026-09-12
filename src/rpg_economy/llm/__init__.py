"""LLM client interface, mock client, token accounting and retry policy.

The only package permitted to touch the network or read credentials. Everything
runs offline against ``MockLLMClient``.
"""

from rpg_economy.llm.client import (
    LLMClient,
    LLMError,
    LLMResponse,
    MockLLMClient,
    OpenAIClient,
    ProviderError,
    Responder,
    always,
    build_client,
    scripted,
)
from rpg_economy.llm.tokens import TokenLedger, apportion, estimate_tokens

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MockLLMClient",
    "OpenAIClient",
    "ProviderError",
    "Responder",
    "TokenLedger",
    "always",
    "apportion",
    "build_client",
    "estimate_tokens",
    "scripted",
]
