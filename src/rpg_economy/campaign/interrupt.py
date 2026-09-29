"""Three-level Ctrl-C for real batches.

A real batch is paid for as it runs, so the first Ctrl-C must not throw away
the conditions already running. It only stops new work: nothing new starts,
and each running trio finishes the condition it is on and does not begin the
next. Paid work is kept, and a resume runs only the missing conditions. A
second Ctrl-C aborts: each running condition stops at its next macro-cycle and
leaves an unfinished log, which a resume re-runs. A third Ctrl-C exits at
once, without waiting for anything: the process ends with status 130, the
kernel releases the output directory's flock, and the unfinished logs keep the
spend they recorded and are re-run on resume.

The first two levels never raise. The SIGINT handler only sets flags (and writes one line
to stderr with ``os.write``, which cannot re-enter a buffered stream), so the
main thread never leaves the wait on its workers early, and the output
directory's lock is released only after every worker has joined.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager

#: The three levels, as ``escalate`` reports them.
STOP = "stop"
ABORT = "abort"
EXIT = "exit"

#: The exit status of a process ended by the third Ctrl-C (128 + SIGINT).
EXIT_STATUS = 130


class Interrupt:
    """The soft (stop) and hard (abort) flags of one batch."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._abort = threading.Event()

    def should_stop(self) -> bool:
        """Start nothing new: no further condition, attempt or trio."""
        return self._stop.is_set()

    def should_abort(self) -> bool:
        """Stop the running conditions at their next macro-cycle."""
        return self._abort.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    def request_abort(self) -> None:
        # An abort is also a stop: nothing new starts after it either.
        self._stop.set()
        self._abort.set()

    def escalate(self) -> str:
        """One more Ctrl-C: stop first, then abort, then exit. The level reached.

        ``EXIT`` is only reported; the SIGINT handler is what acts on it.
        """
        if not self._stop.is_set():
            self.request_stop()
            return STOP
        if not self._abort.is_set():
            self.request_abort()
            return ABORT
        return EXIT

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, woken at once by a stop. True if stopped."""
        return self._stop.wait(seconds)

    @contextmanager
    def handling_sigint(
        self, *, stopping: str = "", aborting: str = "", exiting: str = ""
    ) -> Iterator[Interrupt]:
        """Route SIGINT to ``escalate`` for the block, then restore the previous handler.

        ``stopping``, ``aborting`` and ``exiting`` are the lines written to
        stderr when each level is reached. They are rendered before the
        handler is installed: translating inside a signal handler could wait on
        a lock the interrupted main thread holds. On the third level the
        handler exits the process with ``os._exit``, which runs no cleanup and
        so cannot block on anything. Outside the main thread (where signal
        handlers cannot be installed) the block runs with the flags alone.
        """
        if threading.current_thread() is not threading.main_thread():
            yield self
            return

        lines = {STOP: stopping, ABORT: aborting, EXIT: exiting}

        def handler(signum: int, frame: object) -> None:
            level = self.escalate()
            text = lines.get(level)
            if text:
                with contextlib.suppress(OSError):
                    os.write(2, (text + "\n").encode("utf-8"))
            if level == EXIT:
                os._exit(EXIT_STATUS)

        previous = signal.signal(signal.SIGINT, handler)
        try:
            yield self
        finally:
            signal.signal(signal.SIGINT, previous)


__all__ = ["ABORT", "EXIT", "EXIT_STATUS", "STOP", "Interrupt"]
