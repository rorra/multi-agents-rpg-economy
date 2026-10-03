"""Behaviour check: the real model, judged by the rules each role must follow.

``obligations`` states those rules once, as checks over an observation and an
answer. ``cases`` builds situations in the engine, each in a clean and a noisy
variant; ``replay`` adds real moments from a run log, rebuilt exactly with
today's prompts; ``runner`` asks the model a few times per situation and judges
every answer by every rule that applies. ``audit`` applies the same rules to
every activation of a real run, so a run is judged by what the check tests.
Run before any costly batch, after any change to prompts, the world or the
model; the results file is evidence of how the agents behave, read by nothing
in the experiment itself.
"""
