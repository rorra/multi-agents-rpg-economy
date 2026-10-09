# RPG Economy — Multi-Agent Orchestration Experiment

Comparing **centralized (Director) vs. decentralized orchestration** of LLM-based
multi-agent systems in a simulated RPG economy.

## Overview

The economy (adventurers and artisans trading through a central order book) is
the same in every run; only the coordination changes:

| Condition | Coordination |
|---|---|
| A — Centralized | An LLM Director picks which agent acts and assigns it a task from a closed list. |
| B — Decentralized | No Director. Mechanical triggers and a neutral scheduler decide who acts; agents talk peer-to-peer. |
| C — Decentralized + bulletin | Same as B, plus a passive global summary (the same one the Director sees). |

Runs are compared seed by seed, under equal activations per cycle (Experiment 1)
and under an equal token budget (Experiment 2).

## How the economy works

- Adventurers (warrior, mage, ranger) gather raw materials on expeditions and
  sell them. Gear and potions add to the chance of success; gear wears out on
  every trip. A successful trip also brings back some gold, more for mages,
  who use up a mana potion on every trip. After each trip an adventurer rests
  a few ticks: it can trade meanwhile, and its orders stay on the book while
  it is away.
- Artisans (smithing, woodworking, alchemy) turn raw materials into weapons,
  armor, bows, staves and potions.
- Trading happens only on a central limit-order book: prices come from trades.
  Orders expire after a few ticks. One action can place several orders at once
  (`place_orders`), up to a configured limit.
- Taxes: the seller of a finished product pays a profit tax, a share of what
  the sale brings in above the cost of the product's inputs at their last
  traded prices; a sale at or below that cost pays nothing. Every few ticks each
  agent pays a wealth tax on its gold minus its frozen gold, minus the tax it
  already owes, minus an exempt minimum (and, during a liquidity freeze, minus
  the frozen share of the gold held in bids placed before the freeze): below
  the minimum nothing is paid, and whoever holds more pays more. It is
  paid from available gold; what cannot be paid becomes arrears. Arrears are a
  lien: gold owed cannot back a bid or be spent, it is not taxed again, and it
  is paid from free gold at the start of each cycle. Artisans also pay
  storage per unit held, never charged below a gold floor. The replay
  shows what each tax has collected and what each agent has paid; each
  wealth-tax collection is marked on the timeline and the chart, and on its
  tick every condition shows who paid how much and who settled arrears.
- Agents can send each other structured messages (REQUEST, OFFER, INFORM, ACK,
  REJECT) in B and C. An ACK accepts a proposal and places the matching order
  in the same action; ACK and REJECT are never answered.
- Each run has a base phase, one shock (resource scarcity, demand surge,
  productivity drop or liquidity loss) and a recovery phase.

Every number above lives in `configs/base.yaml`.

## Documents

| Document | Role |
|---|---|
| [`docs/design/DESIGN_v1.0.md`](docs/design/DESIGN_v1.0.md) | Experimental design — the source of truth. In Spanish. |
| [`docs/design/IMPLEMENTATION.md`](docs/design/IMPLEMENTATION.md) | How the design maps to code: stack, invariants, phases. |
| [`docs/glossary.md`](docs/glossary.md) | Spanish design terms and their English names in code. |

## Setup

Requires Python 3.14.

```bash
poetry install --extras analysis --extras llm
cp .env.example .env    # only needed for real-provider runs
```

Everything runs offline by default (`llm.mock: true`), including full A/B/C runs
and the analysis. The exceptions are `smoke` and `derisk`, which always call the
real provider (see Usage).

## Usage

Run one seed-paired trio offline and open its replay:

```bash
rpg-economy trio --seed 1004 --out runs/demo --good herbs
xdg-open runs/demo/replay/en/replay.html
```

The replay shows the three conditions side by side, tick by tick: the price of
any good in all three on one chart, a comparison table, and for each condition
every activation of the current tick in order: why the agent acted (Director
task or trigger), what it saw, the model call, what it decided and why, what the
engine did, and the RESULT sent back to the Director. Each model call, the
Director's included, has a "Prompt and reply" button that opens the exact
system prompt, prompt and reply sent and received. Click an agent to follow
all its activations. Keys: Space play/pause, arrows step, Home/End first/last
tick, S shock, R recovery, T next wealth-tax collection.

Other commands:

```bash
rpg-economy config-show
rpg-economy campaign-plan --phase pilot --profile-logs runs/mock
rpg-economy campaign-run --phase pilot --out runs/mock
rpg-economy economy-check --shock-efficacy
rpg-economy economy-check --shock-efficacy --pilot-logs runs/pilot/calibration
rpg-economy report runs/final --layer configs/budget.yaml --out reports
rpg-economy dashboard runs/pilot/.../A.jsonl
rpg-economy replay runs/pilot/.../seed_1004 --out reports/replay --all-locales
```

Add `--locale es` before the command (`rpg-economy --locale es <command>`), or
set `SIM_LOCALE=es`, for Spanish output; every command,
`replay`, `dashboard` and `trio` included, honours `SIM_LOCALE`. Every command
runs offline with the mock provider unless `--real` is given, except `smoke` and
`derisk`: those two exist to check the real provider and always call it, with
no `--real` flag. Any real-provider batch, those two included, is refused until
it has a cost estimate, the projected total is confirmed with `--confirm-cost`,
and `--spend-limit-set` acknowledges a spend limit on the provider account.

Sample output is in [`reports/samples/`](reports/samples/): a bilingual report and
a trio replay page that opens in any browser.

## Pilot and experiment

First the pilot, then the experiment. The pilot checks that everything behaves
with real agents and derives the Experiment 2 token budget; the experiment then
runs with that budget added as a configuration layer.

```bash
# 1. Check offline that each shock moves its target
rpg-economy economy-check --shock-efficacy
# 2. Profile the pilot offline, to project its cost
rpg-economy campaign-run --phase pilot --out runs/mock
# 3. Run both pilot stages with the real provider; each writes to its own folder
#    (runs/pilot/calibration, runs/pilot/budget-derivation). Without
#    --confirm-cost the command prints the projected total and stops.
#    Calibration runs a compressed horizon (campaign.pilot_calibration_horizon,
#    phases scaled in proportion); budget derivation runs the full horizon.
rpg-economy pilot --stage calibration --out runs/pilot \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
rpg-economy pilot --stage budget-derivation --out runs/pilot \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
# 4. Check the shocks bite with LLM agents, and derive the Experiment 2 budget
rpg-economy economy-check --shock-efficacy --pilot-logs runs/pilot/calibration
rpg-economy derive-budget --pilot-logs runs/pilot/budget-derivation \
    --out runs/pilot/budget.json --layer-out configs/budget.yaml
# 5. Run the experiment and report
rpg-economy campaign-run --layer configs/budget.yaml --phase final --out runs/final \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
rpg-economy report runs/final --layer configs/budget.yaml --out reports
# 6. Draw the figures for the paper (both languages, PNG and PDF)
rpg-economy paper-figures runs/final --layer configs/budget.yaml \
    --results reports/resource_scarcity-experiment_1/en/results.json --out reports/paper
```

Notes on running it:

- `paper-figures` reads the run logs and the report's `results.json`; it
  writes every figure to `reports/paper/en/` and `reports/paper/es/`.
- `economy-check --shock-efficacy` checks the primary scenario and the
  secondary ones in `campaign.secondary_scenarios`; `--scenario` checks any one.
- `trio --real`, `smoke` and `derisk` refuse a final seed unless
  `--allow-final-seed` is given: running one before the experiment looks at
  final data.
- A trio counts as done once its three logs pass the integrity check
  (`integrity.json` in the trio folder). Resuming skips done trios and prices
  only the ones still to run. A provider failure re-runs only the condition
  that failed; the logs set aside stay in `previous-N/`, and their spend is
  counted in `billing.json`.
- `pilot --verify-only` fails if any planned run is missing or unfinished.
- Resuming also requires the logs to come from the same prompts, global view
  format, Python release and source code (`code_digest` in the run metadata);
  any other log is set aside and its trio re-run. `derive-budget` likewise
  refuses logs written by other code.
- A budget derived from mock runs is marked `budget.derived_from_mock`, and
  real `campaign-run`, `pilot` and `trio` refuse it.
- Ctrl-C stops a batch in two steps. The first Ctrl-C starts nothing new: each
  running trio finishes the condition it is on and does not start the next,
  what was paid for is kept, and resuming runs only the missing conditions. A
  second Ctrl-C aborts at the next macro-cycle. The folder lock is held until
  everything has stopped, and the command exits with code 130. A third Ctrl-C
  exits at once: the lock is released when the process ends, and the
  unfinished runs keep their logged spend and are re-run on resume.
- `campaign-run`, `pilot`, `trio`, `smoke` and `derisk` lock their output
  folder (`.campaign.lock`) while they run; a second run on the same folder is
  refused with exit code 10. A lock left on this host whose `flock` is free is
  taken over, even if its recorded pid now belongs to another process (a note
  says so). `--break-lock` is needed only for a lock left on another host or
  one that cannot be read; a lock held by a running process is never replaced.
- A real `campaign-run` or `pilot` with nothing left to run skips the cost
  confirmation but still verifies and writes `billing.json`. If what is pending
  changes between the confirmation and the start, the command refuses with exit
  code 5 and runs nothing.
- `billing.json` is written even when the batch fails. It also counts the spend
  of unfinished runs (`unfinished_spend`), and `billed_total_all` adds finished,
  unfinished and set-aside logs together.
- `report --allow-mock` and `economy-check --shock-efficacy --pilot-logs ...
  --allow-mock` read mock runs too, only to exercise the workflow offline.
- `derive-budget --layer-out` refuses to overwrite an existing file unless
  `--force` is given.
- Exit code 5 means the command refused before spending anything: a missing
  cost confirmation or spend limit, a real run asked for without `--real`, a
  final seed, a mock-derived budget, a budget not written by `derive-budget` or
  derived under other code, or pending work that changed.
- Exit code 6 means a run log that cannot be used: missing, unreadable or
  malformed.
- Exit code 7 means that a run, stage or verification did not complete
  cleanly: an integrity failure, a failed trio, or a failed billing check.
- Adding seeds to a campaign keeps the trios already finished: the
  configuration hash does not cover the seed list or the campaign section, nor
  the budget for Experiment 1 runs.
- Re-running `smoke` sets the earlier logs aside in `previous-N/`; re-running
  `derisk` keeps each model's trio that finished and passed the check, and
  prices and confirms only the rest. `smoke` asks to confirm its typical cost
  and also prints the worst case, to size the spend limit.
- An unsupported `--locale` or `SIM_LOCALE` exits with code 2.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

CI runs the same checks on Python 3.14, with the `analysis` and `llm` extras
installed.

## Notes

- All tunables live in `configs/base.yaml`. Every run records the hash of the
  configuration it ran with, its prompt hashes, model, Python version and a
  digest of the source modules that determine a run's behaviour.
- The engine is deterministic given a seed, and every run is written as an
  append-only JSONL log that is enough to reconstruct it.
- Logs and results store language-neutral values; English or Spanish is applied
  when rendering.
