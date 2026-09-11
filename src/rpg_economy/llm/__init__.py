"""LLM client interface, mock client, token accounting and retry policy.

The only package permitted to touch the network or read credentials. Everything
runs offline against ``MockLLMClient``.
"""
