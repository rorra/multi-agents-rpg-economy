"""Deterministic world: state, market, expeditions, shocks, tick loop.

Contains no network access and no LLM dependency of any kind. LLMs propose
actions; this package validates and executes them (design v1.0 s2).
"""
