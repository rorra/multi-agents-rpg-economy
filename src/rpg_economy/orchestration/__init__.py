"""Orchestration: Director (condition A), triggers and scheduler (B/C), bulletin (C).

Nothing in the trigger or scheduling path may consult a language model. That is
what keeps "who acts next" a mechanical decision in Conditions B and C, and it is
enforced by test, not by convention.
"""
