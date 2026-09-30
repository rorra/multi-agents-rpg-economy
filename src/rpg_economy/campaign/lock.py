"""An exclusive lock on a campaign's output directory.

Two campaign or pilot processes pointed at the same ``--out`` would each find the
same trios pending and each pay for them. The lock makes the second one refuse
instead. It is an ``fcntl.flock`` on a lock file in the directory: the kernel
lets exactly one open file hold it, and releases it when its holder exits,
however it exits. The file records the owner's pid, host and start time, for
the refusal message, and is removed when the run ends normally.

A process killed outright leaves its file behind, but not its flock. On the
host that wrote it, a free flock is proof that its owner is gone, so a
leftover file recorded by this host is taken over, even when its recorded pid
now belongs to another process (a reused pid; the takeover then says so). A
leftover file from another host, or one that cannot be read, cannot be proven
released from here -- a network filesystem may not carry flocks between
hosts -- and is taken over only when asked explicitly (``break_lock``). A
flock that is held is never taken over: its holder is alive.

When the lock file cannot be opened or locked at all (permissions, or a
filesystem without flock support), ``CampaignLockUnavailable`` says why.

Nothing is renamed or unlinked to take a lock over, so no two processes can
both believe they won. The one unlink, on release, happens while the flock is
still held; a process that opened the file just before it is removed finds,
once it gets the flock, that the path no longer names what it locked, and
starts again.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The lock file's name inside the output directory. Not a ``.jsonl``, so no
#: log reader ever takes it for a run.
LOCK_NAME = ".campaign.lock"


class CampaignLockError(RuntimeError):
    """The output directory is locked by another run.

    ``held`` is True when another process holds the flock (its owner is
    alive), and False for a leftover that cannot be proven released from this
    host, which only ``break_lock`` takes over.
    """

    def __init__(self, path: Path, holder: dict[str, Any], *, held: bool = True) -> None:
        super().__init__(
            f"{path} is {'held' if held else 'left'} by pid {holder.get('pid', '?')} on "
            f"{holder.get('host', '?')} since {holder.get('started_at', '?')}"
        )
        self.path = path
        self.holder = holder
        self.held = held


class CampaignLockUnavailable(RuntimeError):
    """The lock file could not be created, opened or locked at all.

    Permissions, a read-only directory, or a filesystem that does not support
    ``flock`` (``ENOLCK``, ``EOPNOTSUPP``). ``detail`` is the system's reason.
    """

    def __init__(self, path: Path, error: OSError) -> None:
        detail = error.strerror or str(error)
        super().__init__(f"cannot lock {path}: {detail}")
        self.path = path
        self.detail = detail
        self.errno = error.errno


@dataclass(frozen=True)
class CampaignLock:
    """A held lock, and the leftover one it replaced, if any.

    ``pid_reused`` is True when the replaced lock's recorded pid is alive on
    this host: the flock was free, so that process is not its owner.
    """

    path: Path
    replaced: dict[str, Any] | None = None
    pid_reused: bool = False


def read_lock(path: Path) -> dict[str, Any]:
    """The holder recorded in a lock file; empty if it cannot be read."""
    try:
        holder = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    return holder if isinstance(holder, dict) else {}


def _pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists on this host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, owned by someone else
        return True
    except OSError:
        return False
    return True


def is_stale(holder: dict[str, Any]) -> bool:
    """Whether a lock's recorded owner is gone: same host, pid no longer alive.

    Informational only. Taking a leftover over rests on the free flock
    (``is_local``), which a reused pid cannot fake.
    """
    pid = holder.get("pid")
    if not isinstance(pid, int) or not is_local(holder):
        return False
    return not _pid_alive(pid)


def is_local(holder: dict[str, Any]) -> bool:
    """Whether a lock file was written on this host.

    For such a file a free flock proves its owner is gone: the kernel that
    would hold the flock for it is this one.
    """
    return bool(holder.get("host")) and holder.get("host") == platform.node()


#: How many times to reopen the lock file when it was replaced or removed
#: between opening and locking it. Each retry needs another process to finish
#: a whole run in between, so a handful is plenty.
_ATTEMPTS = 8


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(descriptor, 1 << 16):
        chunks.append(chunk)
    return b"".join(chunks)


def _parse(raw: bytes) -> dict[str, Any]:
    try:
        holder = json.loads(raw.decode("utf-8"))
    except ValueError:
        return {}
    return holder if isinstance(holder, dict) else {}


def _names(path: Path, descriptor: int) -> bool:
    """Whether ``path`` still names the file open on ``descriptor``."""
    try:
        on_disk = os.stat(path)
    except FileNotFoundError:
        return False
    held = os.fstat(descriptor)
    return (on_disk.st_dev, on_disk.st_ino) == (held.st_dev, held.st_ino)


def _write(descriptor: int, holder: dict[str, Any]) -> None:
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, (json.dumps(holder, sort_keys=True) + "\n").encode("utf-8"))
    os.fsync(descriptor)


def _acquire(
    path: Path, holder: dict[str, Any], *, break_lock: bool
) -> tuple[int, dict | None, bool]:
    """Take the flock on ``path`` and record ``holder`` in it.

    Returns the open descriptor, the leftover holder it replaced, if any, and
    whether that holder's pid was alive (reused by another process). Raises
    ``CampaignLockError`` while another process holds the flock, and for a
    leftover from another host or an unreadable one unless ``break_lock``;
    ``CampaignLockUnavailable`` when the file cannot be opened or locked.
    """
    for _ in range(_ATTEMPTS):
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        except OSError as exc:
            raise CampaignLockUnavailable(path, exc) from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Held: its owner is alive, whatever the file says.
                raise CampaignLockError(path, _parse(_read_descriptor(descriptor))) from None
            except OSError as exc:
                # ENOLCK, EOPNOTSUPP, ...: no flock here, so no lock at all.
                raise CampaignLockUnavailable(path, exc) from exc
            if not _names(path, descriptor):
                # Released and removed between our open and our flock.
                os.close(descriptor)
                continue
            raw = _read_descriptor(descriptor)
            replaced: dict[str, Any] | None = None
            pid_reused = False
            if raw.strip():
                # Left by a run that never released it. The free flock proves
                # its owner gone when the file was written on this host,
                # whatever became of its pid; otherwise only when asked to.
                existing = _parse(raw)
                if not (break_lock or is_local(existing)):
                    raise CampaignLockError(path, existing, held=False)
                replaced = existing
                pid = existing.get("pid")
                pid_reused = is_local(existing) and isinstance(pid, int) and _pid_alive(pid)
            if not _names(path, descriptor):
                os.close(descriptor)
                continue
            _write(descriptor, holder)
            return descriptor, replaced, pid_reused
        except BaseException:
            os.close(descriptor)
            raise
    raise CampaignLockError(path, read_lock(path))


@contextmanager
def campaign_lock(directory: Path | str, *, break_lock: bool = False) -> Iterator[CampaignLock]:
    """Hold the output directory's lock for the duration of the block.

    Raises ``CampaignLockError`` if another run holds it. A leftover written on
    this host is replaced (see ``is_local``); any other leftover only with
    ``break_lock``; a held flock never. Raises ``CampaignLockUnavailable`` when
    the directory or the lock file cannot be created, opened or locked. On
    exit, exceptions included, the file is removed while it still records this
    run, and the flock is released.
    """
    root = Path(directory)
    path = root / LOCK_NAME
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CampaignLockUnavailable(path, exc) from exc
    holder = {
        "pid": os.getpid(),
        "host": platform.node(),
        "started_at": datetime.now(UTC).isoformat(),
    }

    descriptor, replaced, pid_reused = _acquire(path, holder, break_lock=break_lock)
    try:
        yield CampaignLock(path=path, replaced=replaced, pid_reused=pid_reused)
    finally:
        try:
            # Removed before the flock is released, and only while it is still
            # this run's own record at this path.
            if _names(path, descriptor) and _parse(_read_descriptor(descriptor)) == holder:
                path.unlink(missing_ok=True)
        finally:
            os.close(descriptor)  # releases the flock


__all__ = [
    "LOCK_NAME",
    "CampaignLock",
    "CampaignLockError",
    "CampaignLockUnavailable",
    "campaign_lock",
    "is_local",
    "is_stale",
    "read_lock",
]
