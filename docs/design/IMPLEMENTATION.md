# Implementation notes

How the experimental design (`DESIGN_v1.0.md`) is turned into code. The design
document is the source of truth; when something here disagrees with it, the
design wins. Gaps found while implementing are resolved in the design first,
then coded.

## 1. Overview

Three conditions run on the same economy and are compared seed by seed:

- **A — Centralized**: an LLM Director picks which agent acts and gives it a
  TASK from a closed list.
- **B — Decentralized**: no Director. Mechanical triggers decide who asks to
  act, a neutral scheduler serves the requests, and agents coordinate through
  P2P messages and the market.
- **C — Decentralized + bulletin**: same as B, plus a passive global summary
  (the same one the Director sees in A).

Two budget regimes: Experiment 1 fixes the number of activations per
macro-cycle (K); Experiment 2 fixes the total token budget per run.

## 2. Language and i18n

- Code, comments, commits and repo docs are in English.
- Anything a user reads (CLI, reports, figures, dashboard) goes through
  `i18n.t(key)` with catalogs in `locales/en.json` and `locales/es.json`.
  Locale comes from `--locale` or `SIM_LOCALE` (default `en`).
- Stored results never contain translated text: logs and metrics use neutral
  keys, and the language is applied when rendering. The same results produce
  `reports/en/` and `reports/es/`.
- The design uses Spanish domain terms; the code uses English enums. The
  mapping is kept in `docs/glossary.md`.

## 3. Stack

- Python 3.14 only. The exact patch version is recorded in every run's
  metadata and is part of the determinism digest: a run under a different
  interpreter is a different environment.
- pydantic v2 for schemas, pytest, numpy/pandas/scipy for analysis,
  matplotlib for figures.
- A single provider-agnostic LLM client in `llm/client.py` (OpenAI). It is the
  only module that loads credentials (`OPENAI_API_KEY` from `.env`) or touches
  the network. The engine works fully offline with rule-based or mock agents.
- Mock or real provider is an execution mode, not part of the setup: the
  configs keep `llm.mock: true`, commands that may call the provider take
  `--real`, and `llm.mock` is left out of the config hash, so one
  configuration has the same identity whether it runs mocked or real.
- Every tunable from the design (K, thresholds, durations, seeds, model,
  temperature...) lives in YAML under `configs/`, never as a literal in code.
- Logs are append-only JSONL. Analysis reads them and never writes to them.

## 4. Invariants

These are what keep the A/B/C comparison valid. Each one has a test.

### 4.1 Engine vs. decisions
The engine is deterministic given a seed (inventories, gold, recipes, order
matching, expeditions, durability, shocks). LLMs only propose actions; the
engine validates and executes them. All randomness comes from the run seed, so
same seed + same decisions gives the same run.

### 4.2 Director (A)
Sits outside the economy and sees only the global summary and recent
TASK/RESULTs. At most K + 1 calls per macro-cycle, retries included. Each call
either assigns one TASK or closes the cycle. An invalid output gets one retry
if the cap allows it, otherwise it is logged as `director_error`. Director
calls don't use K but do count against the token budget. It can never change
the economy directly. Its agent list marks who is away on an expedition (can
trade, cannot equip or set out) and who is resting in town (can trade and
equip, cannot set out yet). A TASK whose own Director call spends the last of
the budget is logged as `director_task_dropped`, never as assigned. A's
`cycle_end` carries `post_exhaustion`, as B and C do, so cycles after the
budget ran out are left out of the K-binding rate in every condition.
A TASK whose context references a RESULT outside the recent window (stale or
never issued) is still assigned: that reference is dropped and listed in
`dropped_context` on the assignment event. The window size is
`observability.director_recent_results`; with 0 the Director sees no past
RESULTs.

### 4.3 TASK / RESULT (A)
Both are structured. TASK = `{type, params, priority, context}`, with `type`
from the six-type vocabulary and `context` limited to references to recent
RESULTs. RESULT = `{status, action_taken, outcome, needs, offers,
observations}`. An invalid RESULT gets one retry and is then recorded as
`FAILED`. The needs, offers and observations are a report beside the decision:
a malformed entry is dropped on its own, and the decision stands.

The Director sees full RESULTs. The agent that receives a TASK sees only the
public part of each RESULT its context references: id, agent, action and
status. Rejection codes and any other private state never reach it. A TASK
parameter that is optional and null or empty is ignored, not treated as
invalid.

### 4.4 Agent action
`{action, target, quantity, price, recipient, priority, short_reason,
message_type, orders}`. One retry on invalid output, then `WAIT`. A retry resends
the same prompt with the provider seed moved on by one, in every condition, so
it is a fresh draw rather than a replay of the failed one.
`short_reason` is kept for qualitative analysis only. `place_orders` carries up
to `market.max_orders_per_activation` buy, sell and cancel_order orders in
`orders`, each parsed and executed as the action it is: accepted orders go on
the book, refused ones are logged with their code (`order_index` in the event),
and the activation is wasted only if none was accepted. The RESULT in A lists
each order's outcome.

### 4.5 Triggers and scheduler (B, C)
No LLM involved. Triggers T1–T5 (message received, own order event, price
move, own-state threshold, inactivity) are evaluated by the engine. The
bulletin never fires a trigger. The T4 gold thresholds read available gold
(not reserved in bids, not frozen and not owed as tax), identically in B and C,
and the prompt states that available gold leaves out reserved gold, frozen gold
and tax owed. Every trigger firing is logged. The scheduler serves up to K
requests per cycle, fewest activations first, ties broken by a seeded permutation, and it
never looks at state or message content. A test with a mock LLM that raises
if called proves this. Modules on the mechanical path may import the token
budget and token counting (`llm/budget.py`, `llm/tokens.py`), never the client:
checking whether a call is affordable is arithmetic, not a decision.

### 4.6 Bulletin (C)
Built by the same function as the Director's global view, so the format is
identical. It is passive: included only when an agent is activated for some
other reason, and its tokens count only when included.

The global view is built once per macro-cycle, at the start of the decision
phase, at the same moment as the bulletin, and the Director reuses it for every
call of that cycle. Its shock line names only the shock type and the affected
good or specialization, never private-gold aggregates or the exact multipliers
(global view format 1.2). The Director's view is logged in full once per cycle
(`director_view`), as the bulletin is.

### 4.7 Budgets
K is the same in A, B and C and in both experiments. Experiment 2 counts
tokens only, split into `agent`, `director`, `p2p_content` and `bulletin`
buckets. The budget value is the median total tokens of the primary scenario under
Experiment 1, computed by a script. A budget derived from
mock runs is marked (`budget.derived_from_mock`) and refused by real campaign
runs. `derive-budget` admits only pilot runs whose every call was billed by the
provider, and records the code digest they ran under
(`budget.derived_code_digest`, left out of the config hash). A real run refuses
a budget without that record, or one derived under other code. When it runs out, LLM
calls stop and the engine keeps ticking mechanically to the end of the run.

Billing provenance comes from the provider response, never from config: a call
counts as billed spend only when its response carried a usage object. Mock
calls still consume the Experiment 2 budget, but never appear in spend totals
or reconciliations.

### 4.8 Market
Central limit order book with price-time priority, order expiry and no
official price. Production capacity counts recipe runs (batches) per
macro-cycle, not output units. Capacity credit keeps the fraction left after
crafting (its ceiling is the rate plus just under one run), so a rate of 0.3
yields 3 runs in 10 cycles. An adventurer away on an expedition can trade
(place orders and keep the ones it has) but cannot equip until it is back in
town. Back in town it rests for a fixed number of ticks (after a failure, plus the
recovery for the zone's risk): while resting it can trade and equip, but it cannot set out again
until the rest is over. Expeditions that return on the same tick draw from the
zone pools in an order keyed by the run seed, the tick and the agent id, so two
agents returning together draw in the same order in A, B and C. Agents only see
their own orders/trades, best bid/ask and
recent trades of the goods they care about. Aggregates are only in the global
view.

Wealth tax: at each collection the base is the agent's gold minus its frozen
gold, minus what it already owes, minus the exemption. While a liquidity freeze
is active, the frozen share of the gold reserved in bids placed before the
shock is also left out of the base. The tax is paid from available gold;
whatever cannot be paid becomes arrears. Arrears are a lien: gold owed is not
available (it cannot back a bid or be spent), it is not taxed again, and
it is paid from free gold at the start of each cycle, before any new
assessment. Arrears are
never forgiven. The agent sees the amount it owes in its state, and at the end
of the run the amount still owed is subtracted from its wealth (Gini,
bankruptcy). Otherwise gold kept in bids is part of the base, so nobody avoids
the tax that way.

Storage is never charged below `economy.charge_floor`, and the floor is checked
against gold net of frozen gold, of tax owed and of the gold that a pre-freeze
bid would leave frozen when released: none of that gold is the agent's to
spend, so it does not count toward the floor.

The config loader rejects a recipe input below 1, a recipe with no inputs, a
durability below 1, a `demand_surge` target that is not a consumable some
subclass carries in its gear, a `resource_scarcity` target that spawns in no
zone, a `productivity_drop` target with no artisans, an expedition table keyed
by risk that misses a risk level some zone uses, and interest lists that do not
parse.

The engine raises on a broken invariant (for example, releasing more gold than
is reserved) and the run stops. Such an error is never logged as a rejected
action: it is a bug, not an agent's mistake.

### 4.8.1 Prompts
Prompts are versioned (`PROMPT_VERSION`, currently 2.61) and hashed into run
metadata. They never give tactical advice: they state the world rules, what the
agent sees, and plain market facts.

World rules. Rendered by role, and identical for a role in A, B and C:
artisans get recipes, crafting and selling; adventurers and the Director also
get the zones, the rest after a trip and the wear of equipment. Every valid
action documents its parameters (`cancel_order` takes the order id in
`order_id`; `send_message` takes a `recipient`, a `message_type` and the good in
`target`; `place_orders` runs its orders in the order listed, each judged on its
own, up to the per-activation limit). The rules include a price guide: each
good's configured reference price, stated as where prices started and as a
reference for what a good is worth, not an official price or a limit; prices
follow supply and demand from it. They also state the taxes, the storage cost
and its floor, that available gold leaves out reserved gold, frozen gold and tax
owed, that an order crossing the agent's own opposite order is refused, that
adventurers only sell raw materials and only buy gear and potions (the market
refuses the rest), and that each step (buy, craft, sell) takes an activation,
so artisans work in batches.

In B and C the rules also state what each message means: a message moves
nothing and only orders trade, an ACK means the other agent placed its matching
order and a REJECT that it declined, a new REQUEST or OFFER replaces the open
one on the same good, and a proposal can be answered only while it is open. The
open proposals addressed to the agent are shown, marked open or closed. How an
agent tells others its needs (messages in B and C, RESULT fields in A) is
stated in each condition's action schema.

Director. Its roster lists each agent with its tasks so far and the tick of
the last (`[n,tT]`), fewest tasks first, and marks who is away or resting. It is
told that an agent never tasked does nothing, that a task ends in the action
that serves it, and the purpose of each TASK type (design §3.1.1), which the
agent receiving the task is told too. Each agent gets at most one TASK per
macro-cycle, as B and C activate an agent at most once per cycle.

Agent state. The agent sees its gold, inventory, equipment, open orders, its
own recent trades (`observability.own_history`), the tax it owes, and for an
adventurer its success chance per zone if it set out now
(`your_success_chance`, computed by the engine from the gear it carries) and
the table of what each item of its subclass adds (`gear_if_carried`, which is
not what it holds).

Headlines. Above the state, short lines state what stands out, using only what
the agent already sees:

- "Setting out" (adventurers): whether it can set out now and, if not, why
  (resting until a tick, away), and the gear that pays for itself that it does
  not carry: to buy if offered and affordable, to bid for at a stated price if
  not. A bid stays on the book while the adventurer is away.
- "Broken": its main item broke and it carries none, with the price to bid.
- "Gear": for each item it lacks, its price, what it costs a trip and what it
  adds to a trip in each zone, computed with the engine's own rules
  (`ExpeditionSystem.expected_take`) at market prices.
- "Zones now": per zone, how many trips set out over the last 30 ticks (public,
  like adventurers seen leaving town), the crowding shift, the haul and gold a
  success is likely to find there, and what a trip is worth to this adventurer a
  tick with its rest. What is left in a zone is never shown.
- "Ready to sell", "Wanted", "Unsold", "Buying", "Not offered" and "Too dear"
  (artisans): a held product a buyer bids for at or above its
  input cost; a product buyers want, with the inputs per run and the runs the
  capacity allows; a product held unsold with no bid, with no more to be crafted
  until some sell; an input whose bids expired unfilled, with the price that
  buys now; a product nobody offers or bids for, to offer at its guide price;
  an input whose price comes within one step of what it is worth to the
  artisan.
- "Wanted" (adventurers): a raw material it holds that buyers bid for, stated
  as "you hold 2 and nobody else offers it; buyers bid 10.00 so far, and you set
  the price".
- "Confiscated": while the scarcity shock lasts, what the shock bought from the
  agent and what it paid (see section 4.10).

Market signals. Above the market view, per followed good: whether it is short
(a bid rests and nothing is offered) or unsold (an offer rests and no bid), and
which of the agent's own orders expired unfilled. For an artisan, each recipe
with the runs its inputs allow, the input cost of one unit (each input at its
last trade, or its guide price before any), and how many of the product it
already holds, own sell orders included. Price suggestions use a step of
`market.price_step_pct` (10%): a seller with no buyer is told to ask about a
step less (never below an artisan's input cost), a buyer with no seller to bid a
step above its failed bid, but only once the good has not sold for
`market.recent_sale_ticks`; until then the last sale stands. A raw material
moving back toward its typical price goes halfway there when that is more than
a step. When the spread of a raw material is within two steps, the line gives
the midpoint. An artisan's input has a ceiling: what the product is worth less
its other inputs, leaving the product `market.input_margin_pct` over its
inputs. A good's typical price, the median of its trades over
`market.usual_price_ticks`, is shown whenever the latest trade is
`market.usual_price_gap_pct` or more away from it, with the guide standing in
below `market.usual_price_min_trades` trades. All of it is drawn from public
trades and shown alike in A, B and C.

Engine rules that go with the prompts: an artisan holding
`economy.craft_stock_limit` units of a product cannot craft more of it; setting
out equips the durable gear of the adventurer's subclass that it holds; and
gear is optional, each item carried adding success and a share of a
successful trip's gold (`expeditions.gear_gold_bonus`).

Parsing never charges a formatting habit that carries no decision, in any
condition: a Director TASK parameter that is null or blank is dropped; in A a
stray `recipient` on a non-message action is ignored and an offer price of 0
reads as null; B and C ignore the A-only RESULT fields; a stray `orders` list
or `message_type` beside another action is ignored. Action and target values
are normalized (case folded, spaces and hyphens become `_`) on both the
Director and agent sides, and a blank, "none" or "null" `message_type` reads as
null. Every model completion is logged in full (`raw` on `llm_call` and
`director_call`), and so is the prompt: `user` holds the prompt text and
`system_sha` names the system prompt, whose text is logged once per run in a
`prompt_system` event. A call that fails outright logs the same two fields on
its `agent_output_invalid` event; nothing is charged for it and no delivery
event (`message_delivered`, `bulletin_included_in_prompt`) is logged, so
inclusions are counted from those events, not from the prompt text. The replay
packs these texts once per condition (gzip, base64) and opens them on demand.
Each call records what it carried: `message_delivered` with the inbox ids, the
ids of the open proposals shown again (`open_proposal_ids`) and their tokens,
and `bulletin_included_in_prompt` with the bulletin's tokens. Run metadata
records the global view version, and the global view format hash covers the
version, the public shock details and the cell formatting.

Serialization is compact without changing content: the directory is one
`id:role/specialization` record per agent, and the global view is a columnar
table rendered by the same function for the Director and the bulletin.

