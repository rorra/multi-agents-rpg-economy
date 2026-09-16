"""An offline responder that makes mock runs economically meaningful.

The mock client exercises the whole pipeline, but something has to decide what
the completion says. This responder reads the activation's prompt -- exactly as a
model would -- recovers which agent is being asked, and answers with the
rule-based policy's action rendered as the design's JSON action schema.

Nothing about prompt assembly, parsing, validation, the retry policy or token
accounting is special-cased for mocks, and a complete Condition B run offline
produces the same economics as the rule-based harness.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

from rpg_economy.agents.action import PRICE_QUANTUM, AgentAction
from rpg_economy.agents.prompts import OPEN_PROPOSALS_HEADER
from rpg_economy.agents.rule_based import RuleBasedPolicy, policy_for, reference_price
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import (
    ActionType,
    AdventurerSubclass,
    Condition,
    MessageType,
    Priority,
    Resource,
    Role,
    TaskType,
    parse_good,
)
from rpg_economy.engine import Engine
from rpg_economy.engine.rng import agent_stream_name
from rpg_economy.llm.client import Responder
from rpg_economy.messaging import MessageBus

#: How far a counterparty's price may sit from the recent one and still be taken.
ACCEPT_TOLERANCE = Decimal("0.05")
#: How much better for the other side a proposal is re-posted after a REJECT.
REJECTED_STEP = Decimal("0.05")

_AGENT_ID_RE = re.compile(r'"agent_id"\s*:\s*"([^"]+)"')

#: The mock's own peer-to-peer draws, kept off the agent's decision stream.
#: Drawing them from ``engine.random.agent(agent_id)`` would shift every later
#: draw of the rule-based policy in B and C relative to A, where the mock sends
#: no messages, and the trio would no longer share its agents' decisions.
MOCK_P2P_STREAM_SUFFIX = ":mock_p2p"


def mock_p2p_random(engine: Engine, agent_id: str):
    """The stream the mock uses for its messaging choices, one per agent."""
    return engine.random.stream(agent_stream_name(agent_id) + MOCK_P2P_STREAM_SUFFIX)


def action_to_json(action: AgentAction) -> str:
    """Render an action as the completion text the schema asks for."""
    payload = {
        "action": action.action.value,
        "target": action.target,
        "quantity": action.quantity,
        "price": None if action.price is None else float(action.price),
        "recipient": action.recipient,
        "priority": action.priority.value,
        "short_reason": action.short_reason,
    }
    if action.message_type is not None:
        payload["message_type"] = action.message_type.value
    if action.orders:
        payload["orders"] = [
            {
                "action": order.action.value,
                "target": order.target,
                "quantity": order.quantity,
                "price": None if order.price is None else float(order.price),
            }
            for order in action.orders
        ]
    return json.dumps(payload, sort_keys=True)


def agent_id_from_prompt(user_prompt: str) -> str:
    """Recover the acting agent from its own state block."""
    match = _AGENT_ID_RE.search(user_prompt)
    if not match:
        raise ValueError("prompt carries no agent_id; cannot answer for an unknown agent")
    return match.group(1)


def rule_based_responder(
    config: SimulationConfig,
    engine: Engine,
    *,
    reply_to_messages: bool = True,
    messages: MessageBus | None = None,
) -> Responder:
    """A responder that answers with the rule-based policy for the acting agent.

    ``reply_to_messages`` lets an agent answer a peer instead of trading when it
    has unread mail. That is what gives T1 end-to-end coverage: a message is
    sent, the recipient's next activation is triggered by T1, and the exchange
    continues rather than dying at the first message. A REQUEST or OFFER gets an
    ACK when the deal suits the recipient and a REJECT when it does not; an ACK
    back makes the proposer place its order, a REJECT makes it place one at a
    better price. ``messages`` is where the proposer finds what it proposed.
    """
    policies: dict[Role, RuleBasedPolicy] = {}

    def respond(system: str, user: str) -> str:
        agent_id = agent_id_from_prompt(user)
        account = engine.state.account(agent_id)

        if reply_to_messages and "Messages addressed to you" in user:
            inbox = _inbox(user)
            follow_up = _follow_up(engine, messages, agent_id, inbox)
            if follow_up is not None:
                return action_to_json(follow_up)
            reply = _reply_to_sender(
                engine, agent_id, inbox, expiry_ticks=config.market.order_expiry_ticks
            )
            if reply is not None:
                return action_to_json(reply)
        if reply_to_messages and OPEN_PROPOSALS_HEADER in user:
            # A proposal received earlier and still open is answered the same
            # way, so offline B and C exercise the open-proposals path too.
            reply = _reply_to_sender(engine, agent_id, _open_proposals_block(user))
            if reply is not None:
                return action_to_json(reply)

        policy = policies.setdefault(account.role, policy_for(account.role, config))
        action = policy.decide(engine, account, engine.random.agent(agent_id))

        # An artisan that just quoted also tells a plausible counterparty about
        # it, so peer-to-peer traffic exists at all in an offline run.
        # Never in Condition A, where agents send no peer messages (design s3.1).
        # Only a single quote is replaced: a place_orders batch is never traded
        # for one message, which would drop the rest of the batch.
        quote = _single_quote(action)
        if (
            config.run.condition is not Condition.A
            and account.role is Role.ARTISAN
            and quote is not None
            and mock_p2p_random(engine, agent_id).random() < 0.25
        ):
            selling = quote.action is ActionType.SELL
            peer = _pick_peer(config, engine, agent_id, quote.target, selling=selling)
            if peer is not None:
                return action_to_json(
                    AgentAction(
                        action=ActionType.SEND_MESSAGE,
                        message_type=MessageType.OFFER if selling else MessageType.REQUEST,
                        target=quote.target,
                        quantity=quote.quantity,
                        price=quote.price,
                        recipient=peer,
                        priority=action.priority,
                        short_reason="tell a counterparty about my quote",
                    )
                )
        return action_to_json(action)

    return respond


def _single_quote(action: AgentAction) -> AgentAction | None:
    """The buy or sell the action consists of, when it is exactly one order."""
    if action.action in {ActionType.BUY, ActionType.SELL}:
        return action
    if action.action is ActionType.PLACE_ORDERS and len(action.orders) == 1:
        only = action.orders[0]
        return only if only.action in {ActionType.BUY, ActionType.SELL} else None
    return None


def _json_list_after(user_prompt: str, header: str) -> list:
    """The JSON list that follows ``header`` in the prompt, or [] if none does."""
    start = user_prompt.find(header)
    if start < 0:
        return []
    list_start = user_prompt.find("\n[", start)
    if list_start < 0:
        return []
    try:
        items, _ = json.JSONDecoder().raw_decode(user_prompt[list_start + 1 :])
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []


def _inbox(user_prompt: str) -> list[dict]:
    """The messages in the prompt's inbox section, as the model would read them."""
    return _json_list_after(user_prompt, "Messages addressed to you")


def _open_proposals_block(user_prompt: str) -> list[dict]:
    """The open-proposals block, in the inbox's message shape.

    Each entry is flat (sender_id, type, good, quantity, price, ticks_left);
    the good, quantity and price move under ``payload`` as in an inbox message,
    and every one is open by construction.
    """
    return [
        {
            "message_id": entry.get("message_id"),
            "sender_id": entry.get("sender_id"),
            "type": entry.get("type"),
            "payload": {
                "good": entry.get("good"),
                "quantity": entry.get("quantity"),
                "price": entry.get("price"),
            },
            "open": True,
            "ticks_left": entry.get("ticks_left"),
        }
        for entry in _json_list_after(user_prompt, OPEN_PROPOSALS_HEADER)
        if isinstance(entry, dict)
    ]


def _price(value: object) -> Decimal | None:
    try:
        return Decimal(str(value)).quantize(PRICE_QUANTUM) if value is not None else None
    except ArithmeticError:
        return None


def _reply_to_sender(
    engine: Engine, agent_id: str, inbox: list[dict], *, expiry_ticks: int | None = None
) -> AgentAction | None:
    """Answer the first open REQUEST or OFFER in the inbox: ACK if it suits, else REJECT.

    A proposal the prompt marks closed (``open`` false: expired, replaced or
    answered) is skipped, and so is one a later one from the same sender about
    the same good replaced: neither can be answered.

    An ACK or a REJECT closes an exchange and is never answered; the engine
    refuses one that answers nothing, and one to a proposal that has expired.
    """
    account = engine.state.account(agent_id)
    opening = {MessageType.REQUEST.value, MessageType.OFFER.value}
    # Only the latest proposal per sender and good is open: a newer one
    # replaced the others, which can no longer be answered.
    latest = {
        (m.get("sender_id"), (m.get("payload") or {}).get("good")): index
        for index, m in enumerate(inbox)
        if m.get("type") in opening
    }
    for index, message in enumerate(inbox):
        sender = message.get("sender_id")
        payload = message.get("payload") or {}
        good_name = payload.get("good")
        kind = message.get("type")
        if kind not in opening or latest.get((sender, good_name)) != index:
            continue
        if message.get("open") is False:
            continue  # the prompt says it can no longer be answered
        sent_tick = message.get("sent_tick")
        if (
            expiry_ticks is not None
            and isinstance(sent_tick, int)
            and engine.tick >= sent_tick + expiry_ticks
        ):
            continue  # expired: there is nothing left to answer
        if not sender or sender == agent_id or not good_name:
            continue
        try:
            good = parse_good(good_name)
        except ValueError:
            continue
        price = _price(payload.get("price"))
        reference = reference_price(engine, good)
        if kind == MessageType.OFFER.value:
            # They sell: worth it if the price is near the recent one and affordable.
            suits = price is not None and price <= reference * (1 + ACCEPT_TOLERANCE)
            backed = int(account.available_gold / price) if suits and price else 0
        else:
            # They buy: worth it if there is stock to sell and the price is fair.
            suits = price is not None and price >= reference * (1 - ACCEPT_TOLERANCE)
            backed = account.available_quantity(good)
        wanted = int(payload.get("quantity") or 1)
        quantity = min(wanted, backed)
        if suits and quantity > 0:
            # Accepting places the matching order in the same action.
            return AgentAction(
                action=ActionType.SEND_MESSAGE,
                message_type=MessageType.ACK,
                target=good_name,
                quantity=quantity,
                price=price,
                recipient=sender,
                short_reason="accept the proposal and place the order",
            )
        return AgentAction(
            action=ActionType.SEND_MESSAGE,
            message_type=MessageType.REJECT,
            target=good_name,
            recipient=sender,
            short_reason="decline the proposal",
        )
    return None


def _follow_up(
    engine: Engine, messages: MessageBus | None, agent_id: str, inbox: list[dict]
) -> AgentAction | None:
    """Act on an answer to one of this agent's proposals.

    An ACK means the other side is in: place the order proposed. A REJECT means
    it is not: place the order anyway, at a price better for the other side.
    """
    if messages is None:
        return None
    account = engine.state.account(agent_id)
    for message in inbox:
        kind = message.get("type")
        if kind not in {MessageType.ACK.value, MessageType.REJECT.value}:
            continue
        peer = message.get("sender_id")
        good_name = (message.get("payload") or {}).get("good")
        proposal = next(
            (
                sent
                for sent in reversed(messages.sent)
                if sent.sender_id == agent_id
                and sent.recipient_id == peer
                and sent.type in {MessageType.REQUEST, MessageType.OFFER}
                and sent.payload.get("good") == good_name
            ),
            None,
        )
        if proposal is None:
            continue
        price = _price(proposal.payload.get("price"))
        quantity = proposal.payload.get("quantity") or 1
        if price is None:
            continue
        good = parse_good(good_name)
        accepted = kind == MessageType.ACK.value
        if proposal.type is MessageType.OFFER:
            if not accepted:
                price = max(PRICE_QUANTUM, (price * (1 - REJECTED_STEP)).quantize(PRICE_QUANTUM))
            quantity = min(int(quantity), account.available_quantity(good))
            side = ActionType.SELL
        else:
            if not accepted:
                price = (price * (1 + REJECTED_STEP)).quantize(PRICE_QUANTUM)
            quantity = min(int(quantity), int(account.available_gold / price) if price else 0)
            side = ActionType.BUY
        if quantity <= 0:
            continue
        return AgentAction(
            action=side,
            target=good_name,
            quantity=quantity,
            price=price,
            priority=Priority.HIGH,
            short_reason=(
                "place the order the peer accepted"
                if accepted
                else "re-post the declined proposal at a better price"
            ),
        )
    return None


def _pick_peer(
    config: SimulationConfig,
    engine: Engine,
    agent_id: str,
    good: str | None,
    *,
    selling: bool = True,
) -> str | None:
    """A counterparty for ``good``, chosen from the agent's mock messaging stream.

    Selling, only an agent that uses the good: a raw material goes to the
    artisans whose recipes take it, a product to the adventurers whose gear
    includes it. Buying, only one that makes it: a raw material comes from
    the adventurers who gather it best, a product from its specialization.
    Offering herbs to a warrior, or a weapon to an artisan, is noise.
    """
    candidates = [
        other
        for other in engine.state.agent_ids
        if other != agent_id
        and (good is None or _deals_in(config, engine.state.account(other), good, selling))
    ]
    if not candidates:
        return None
    index = mock_p2p_random(engine, agent_id).randrange(len(candidates))
    return candidates[index]


def _deals_in(config: SimulationConfig, account: object, good: str, selling: bool) -> bool:
    """Whether ``account`` would buy ``good`` (``selling``) or supplies it."""
    parsed = parse_good(good)
    recipes = config.economy.recipes
    if account.role is Role.ARTISAN:
        if isinstance(parsed, Resource):
            return selling and any(
                recipe.specialization.value == account.specialization and parsed in recipe.inputs
                for recipe in recipes.values()
            )
        recipe = recipes.get(parsed)
        return (
            not selling
            and recipe is not None
            and recipe.specialization.value == (account.specialization)
        )
    if account.role is Role.ADVENTURER:
        subclass = AdventurerSubclass(account.specialization)
        if isinstance(parsed, Resource):
            settings = config.expeditions.subclasses.get(subclass)
            affinity = settings.resource_affinity.get(parsed, 1) if settings else 1
            return not selling and affinity > 1
        return selling and parsed in config.expeditions.gear.get(subclass, {})
    return False


def director_responder(config: SimulationConfig, engine: Engine) -> Responder:
    """An offline Director that assigns tasks by a fixed, mechanical rule.

    It stands in for a coordinating model the way the rule-based responder stands
    in for an agent: the Director's *loop* -- the cap, the retry policy, the
    error handling -- is what is under test, and that loop must be exercised
    by something whose choices are reproducible.

    It rotates through agents that have acted least, picks a task their role can
    perform, and closes the cycle once K activations are used.
    """
    served: dict[str, int] = {}

    def respond(system: str, user: str) -> str:
        used, limit = _activations_from_prompt(user)
        if used >= limit:
            return json.dumps({"decision": "close"})

        # An adventurer away on an expedition may trade (not equip or set
        # out), but the rule-based policy these mock agents follow only waits
        # while away, so the mock Director does not spend a TASK on one. One
        # resting in town can trade and equip.
        candidates = sorted(
            (a for a in engine.state.agent_ids if engine.state.account(a).expedition is None),
            key=lambda a: (served.get(a, 0), a),
        )
        if not candidates:
            return json.dumps({"decision": "close"})
        agent_id = candidates[0]
        served[agent_id] = served.get(agent_id, 0) + 1

        account = engine.state.account(agent_id)
        task_type = _task_for_role(account.role)
        return json.dumps(
            {
                "decision": "assign",
                "task": {
                    "type": task_type.value,
                    "agent_id": agent_id,
                    "params": {},
                    "priority": "medium",
                },
            },
            sort_keys=True,
        )

    return respond


def _task_for_role(role: Role) -> TaskType:
    """A task the role can actually perform (design v1.0 s3.1.1)."""
    if role is Role.ADVENTURER:
        return TaskType.EVALUATE_EXPEDITION
    return TaskType.EVALUATE_PRODUCTION


_USED_RE = re.compile(r"Activations used this macro-cycle: (\d+) of (\d+)")


def _activations_from_prompt(user_prompt: str) -> tuple[int, int]:
    match = _USED_RE.search(user_prompt)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def json_action(**fields: object) -> str:
    """Build a completion body for tests."""
    payload = {
        "action": ActionType.WAIT.value,
        "target": None,
        "quantity": None,
        "price": None,
        "recipient": None,
        "priority": "medium",
        "short_reason": "",
    }
    payload.update(fields)
    if isinstance(payload.get("price"), Decimal):
        payload["price"] = float(payload["price"])
    return json.dumps(payload, sort_keys=True)
