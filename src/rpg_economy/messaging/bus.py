"""Message delivery and the static public directory (design v1.0 s6, s6.1)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rpg_economy.domain import MessageType, Priority, Role
from rpg_economy.messaging.types import P2P_MESSAGE_TYPES, DirectoryEntry, Message


class MessageRejected(Exception):
    """Raised when a message cannot be delivered as specified."""


class NotAReply(MessageRejected):
    """An ACK or REJECT that answers no open REQUEST or OFFER.

    Also what answering an expired proposal raises: once expired, there is
    nothing open to answer.
    """


#: Messages that open an exchange, and those that close one.
OPENING_TYPES = frozenset({MessageType.REQUEST, MessageType.OFFER})
CLOSING_TYPES = frozenset({MessageType.ACK, MessageType.REJECT})


class Directory:
    """The full static public directory. Every agent knows every agent."""

    def __init__(self, entries: Iterable[DirectoryEntry]) -> None:
        self._entries = {entry.agent_id: entry for entry in entries}

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._entries

    def entry(self, agent_id: str) -> DirectoryEntry:
        try:
            return self._entries[agent_id]
        except KeyError:
            raise MessageRejected(f"unknown agent {agent_id!r}") from None

    def agent_ids(self) -> list[str]:
        return sorted(self._entries)

    def with_role(self, role: Role) -> list[str]:
        return [a for a in self.agent_ids() if self._entries[a].role is role]


class MessageBus:
    """Delivers peer-to-peer messages and holds each agent's inbox.

    Only agent-to-agent traffic passes through here. The bulletin does not,
    which is what makes it structurally unable to fire T1 (design v1.0 s3.3).

    At most one proposal is open per (sender, recipient, good): a new REQUEST
    or OFFER from X to Y about G replaces the one X left open with Y about G,
    so an ACK or REJECT always answers the latest.

    An unanswered REQUEST or OFFER stays open for ``proposal_expiry_ticks``, the
    same number of ticks an order rests on the book, and then expires: an ACK
    or REJECT to it is refused like one that answers nothing. ``clock`` gives
    the current tick to lookups that are not told one. Without an expiry,
    proposals stay open until answered.
    """

    def __init__(
        self,
        directory: Directory,
        *,
        proposal_expiry_ticks: int | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._directory = directory
        self._inboxes: dict[str, list[Message]] = {}
        self._sent: list[Message] = []
        self._next_id = 1
        #: The one REQUEST or OFFER not yet answered per (recipient, sender,
        #: good): the message itself, which carries its type and sent tick.
        self._open: dict[tuple[str, str, str | None], Message] = {}
        self._expiry_ticks = proposal_expiry_ticks
        self._clock = clock

    def set_proposal_expiry(self, ticks: int, clock: Callable[[], int]) -> None:
        """Make open proposals expire after ``ticks``, read against ``clock``."""
        if ticks <= 0:
            raise ValueError(f"proposal expiry must be positive, got {ticks}")
        self._expiry_ticks = ticks
        self._clock = clock

    @property
    def proposal_expiry_ticks(self) -> int | None:
        return self._expiry_ticks

    @property
    def directory(self) -> Directory:
        return self._directory

    @property
    def sent(self) -> list[Message]:
        return list(self._sent)

    def send(
        self,
        *,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        tick: int,
        priority: Priority = Priority.MEDIUM,
        payload: dict | None = None,
    ) -> Message:
        """Deliver one message to one recipient and return it.

        The returned Message carries the bus-assigned ``message_id``, which the
        executor records in its message_sent event.
        """
        if message_type not in P2P_MESSAGE_TYPES:
            raise MessageRejected(
                f"{message_type} is not peer-to-peer traffic; TASK and RESULT are "
                f"Director traffic in Condition A only"
            )
        if sender_id not in self._directory:
            raise MessageRejected(f"unknown sender {sender_id!r}")
        if recipient_id not in self._directory:
            raise MessageRejected(f"unknown recipient {recipient_id!r}")
        if sender_id == recipient_id:
            raise MessageRejected(f"{sender_id} cannot message itself")
        good = (payload or {}).get("good")
        if message_type in CLOSING_TYPES:
            # ACK and REJECT answer a REQUEST or OFFER, once, and are never
            # answered themselves. Answering an acknowledgment starts a loop in
            # which two agents spend every activation acknowledging each other.
            if self.open_proposal(sender_id, recipient_id, good, tick=tick) is None:
                raise NotAReply(
                    f"{message_type.value} answers a REQUEST or OFFER received from "
                    f"{recipient_id} about {good}; there is none open"
                )
            del self._open[(sender_id, recipient_id, good)]

        message = Message(
            message_id=self._next_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            type=message_type,
            sent_tick=tick,
            priority=priority,
            payload=payload or {},
        )
        self._next_id += 1
        self._inboxes.setdefault(recipient_id, []).append(message)
        self._sent.append(message)
        if message_type in OPENING_TYPES:
            # Replaces whatever the sender left open with this recipient about
            # this good: only the latest proposal can be answered.
            self._open[(recipient_id, sender_id, good)] = message
        return message

    def open_proposal(
        self, answerer: str, proposer: str, good: str | None, *, tick: int | None = None
    ) -> MessageType | None:
        """The open REQUEST or OFFER ``proposer`` sent ``answerer`` about ``good``.

        The latest one, the only one that can be open, as of ``tick`` (the
        clock's tick when not given). An expired proposal is discarded here: it
        can no longer be answered.
        """
        key = (answerer, proposer, good)
        pending = self._open.get(key)
        if pending is None:
            return None
        if self._expired(pending, self._now(tick)):
            del self._open[key]
            return None
        return pending.type

    def open_proposals(
        self, agent_id: str, *, tick: int | None = None
    ) -> list[tuple[Message, int | None]]:
        """Every REQUEST or OFFER addressed to ``agent_id`` still open at ``tick``.

        Each with the ticks it has left before it expires (None without an
        expiry), oldest first. What an agent can still answer, whether or not
        the message is still in its inbox: the inbox is drained on activation,
        a proposal stays open until answered or expired. Expired ones are
        discarded here, as :meth:`open_proposal` discards them.
        """
        now = self._now(tick)
        found: list[tuple[Message, int | None]] = []
        for key, message in list(self._open.items()):
            if key[0] != agent_id:
                continue
            if self._expired(message, now):
                del self._open[key]
                continue
            left = (
                None
                if self._expiry_ticks is None or now is None
                else message.sent_tick + self._expiry_ticks - now
            )
            found.append((message, left))
        return sorted(found, key=lambda pair: pair[0].message_id)

    def _now(self, tick: int | None) -> int | None:
        if tick is not None:
            return tick
        return self._clock() if self._clock is not None else None

    def _expired(self, message: Message, now: int | None) -> bool:
        # A proposal sent at tick t is open while now < t + expiry, as an order rests.
        return (
            self._expiry_ticks is not None
            and now is not None
            and now >= message.sent_tick + self._expiry_ticks
        )

    def inbox(self, agent_id: str) -> list[Message]:
        """Messages waiting for ``agent_id``, oldest first. Non-destructive."""
        return list(self._inboxes.get(agent_id, ()))

    def unread_count(self, agent_id: str) -> int:
        return len(self._inboxes.get(agent_id, ()))

    def drain(self, agent_id: str) -> list[Message]:
        """Take and clear an agent's inbox. Called when the agent is activated."""
        return self._inboxes.pop(agent_id, [])
