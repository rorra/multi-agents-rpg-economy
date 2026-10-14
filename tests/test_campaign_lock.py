"""The exclusive lock on a campaign's output directory."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from rpg_economy.campaign.lock import (
    LOCK_NAME,
    CampaignLockError,
    campaign_lock,
    is_stale,
    read_lock,
)
from rpg_economy.cli import main
from rpg_economy.i18n import clear_cache


@pytest.fixture(autouse=True)
def _reset_locale() -> None:
    clear_cache()
    yield
    clear_cache()


def _dead_pid() -> int:
    """The pid of a process that has already exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _plant(directory: Path, **holder: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_NAME
    path.write_text(json.dumps(holder), encoding="utf-8")
    return path


def test_the_lock_records_its_owner_and_is_removed_on_exit(tmp_path: Path) -> None:
    with campaign_lock(tmp_path / "runs") as lock:
        holder = read_lock(lock.path)
        assert holder["pid"] == os.getpid()
        assert holder["host"] == platform.node()
        assert holder["started_at"]
        assert lock.replaced is None
    assert not (tmp_path / "runs" / LOCK_NAME).exists()


def test_the_lock_is_removed_when_the_run_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), campaign_lock(tmp_path):
        raise RuntimeError("boom")
    assert not (tmp_path / LOCK_NAME).exists()


def test_a_held_lock_refuses_a_second_run(tmp_path: Path) -> None:
    with campaign_lock(tmp_path):
        with pytest.raises(CampaignLockError) as info, campaign_lock(tmp_path):
            pytest.fail("a second holder got the lock")
        assert info.value.holder["pid"] == os.getpid()
        # The refused attempt left the holder's lock in place.
        assert (tmp_path / LOCK_NAME).exists()


def test_a_lock_left_by_a_dead_process_on_this_host_is_replaced(tmp_path: Path) -> None:
    pid = _dead_pid()
    _plant(tmp_path, pid=pid, host=platform.node(), started_at="then")

    with campaign_lock(tmp_path) as lock:
        assert lock.replaced == {"pid": pid, "host": platform.node(), "started_at": "then"}
        assert read_lock(lock.path)["pid"] == os.getpid()
    assert not (tmp_path / LOCK_NAME).exists()


@pytest.mark.parametrize(
    "holder",
    [
        {"pid": 1, "host": "another-host", "started_at": "then"},
        {},  # unreadable: an owner still writing it, or a damaged file
    ],
)
def test_a_lock_that_cannot_be_proven_stale_needs_break_lock(
    tmp_path: Path, holder: dict[str, object]
) -> None:
    _plant(tmp_path, **holder)
    assert not is_stale(read_lock(tmp_path / LOCK_NAME))

    with pytest.raises(CampaignLockError), campaign_lock(tmp_path):
        pass
    assert (tmp_path / LOCK_NAME).exists()

    with campaign_lock(tmp_path, break_lock=True) as lock:
        assert read_lock(lock.path)["pid"] == os.getpid()
    assert not (tmp_path / LOCK_NAME).exists()


def test_a_live_process_on_this_host_is_never_stale() -> None:
    assert not is_stale({"pid": os.getpid(), "host": platform.node()})
    assert is_stale({"pid": _dead_pid(), "host": platform.node()})


def test_a_lock_that_is_no_longer_ours_is_left_alone(tmp_path: Path) -> None:
    """Released only while it still holds this run: a broken-and-retaken lock is someone else's."""
    with campaign_lock(tmp_path) as lock:
        lock.path.write_text(json.dumps({"pid": 1, "host": "elsewhere"}), encoding="utf-8")
    assert read_lock(tmp_path / LOCK_NAME) == {"pid": 1, "host": "elsewhere"}


# ---------------------------------------------------------------------------
# Through the CLI
# ---------------------------------------------------------------------------
def _config_file(tmp_path: Path) -> Path:
    import yaml

    from tests.conftest import make_config

    config = make_config(
        phases={"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
        time={"horizon_ticks": 12},
        metrics={"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
    )
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())), encoding="utf-8")
    return path


@pytest.mark.parametrize("locale", ["en", "es"])
def test_campaign_run_refuses_a_directory_another_run_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    locale: str,
) -> None:
    from rpg_economy.campaign import parallel

    def never(*args: object, **kwargs: object) -> None:
        pytest.fail("a locked directory was run")

    from rpg_economy.i18n import t

    monkeypatch.setattr(parallel, "run_campaign_parallel", never)
    out = tmp_path / "runs"

    argv = ["--locale", locale, "campaign-run", "--config", str(_config_file(tmp_path))]
    # Held for real: another open file description holds the flock.
    with campaign_lock(out) as held:
        code = main([*argv, "--phase", "pilot", "--out", str(out)])
        assert held.path.exists(), "the refused run removed another run's lock"
        holder = read_lock(held.path)

    assert code == 10
    err = capsys.readouterr().err
    expected = t(
        "error.campaign.locked",
        locale=locale,
        path=str(out / LOCK_NAME),
        pid=holder["pid"],
        host=holder["host"],
        since=holder["started_at"],
    )
    assert expected in err
    # A held lock is never broken, so the refusal does not offer to.
    assert "--break-lock" not in err
    assert "Traceback" not in err


def test_a_stale_lock_retaken_meanwhile_is_left_to_its_new_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between judging a leftover and writing over it, the file may have been replaced.

    The replacement is live -- its owner holds the flock on it -- so it must
    survive, and this run must refuse rather than record itself in a file no
    longer at the path.
    """
    import fcntl

    from rpg_economy.campaign import lock as lock_module

    stale = {"pid": _dead_pid(), "host": platform.node(), "started_at": "then"}
    path = _plant(tmp_path, **stale)
    live = {"pid": os.getpid(), "host": platform.node(), "started_at": "now"}
    held: list[int] = []

    def retake() -> None:
        path.unlink()
        path.write_text(json.dumps(live), encoding="utf-8")
        descriptor = os.open(path, os.O_RDWR)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(descriptor)

    judged = lock_module.is_local

    def judged_then_retaken(holder: dict[str, object]) -> bool:
        verdict = judged(holder)
        if not held:
            retake()
        return verdict

    monkeypatch.setattr(lock_module, "is_local", judged_then_retaken)

    try:
        with pytest.raises(CampaignLockError) as info, campaign_lock(tmp_path):
            pytest.fail("took a lock another run had just taken")
    finally:
        for descriptor in held:
            os.close(descriptor)

    assert info.value.held
    assert info.value.holder == live
    assert read_lock(path) == live
    assert [p.name for p in tmp_path.iterdir()] == [LOCK_NAME], "a set-aside copy was left"


_HOLD = """
import fcntl, json, os, platform, sys, time
path = sys.argv[1]
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.write(fd, json.dumps({"pid": int(sys.argv[2]), "host": platform.node(),
                         "started_at": "then"}).encode())
print("held", flush=True)
time.sleep(60)
"""


def test_a_held_flock_is_never_taken_over_and_dies_with_its_holder(tmp_path: Path) -> None:
    """The kernel, not the file, says who holds the lock.

    Another process holds the flock; its file even records a dead pid, and
    --break-lock is given. It is refused all the same: a held flock has a live
    holder. Once that process is killed outright, its flock goes with it, and
    the leftover file is replaced as stale.
    """
    path = tmp_path / LOCK_NAME
    dead = _dead_pid()
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(path), str(dead)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        for break_lock in (False, True):
            with pytest.raises(CampaignLockError), campaign_lock(tmp_path, break_lock=break_lock):
                pytest.fail("took a flock another process holds")
    finally:
        holder.kill()
        holder.wait()

    with campaign_lock(tmp_path) as lock:
        assert lock.replaced is not None and lock.replaced["pid"] == dead
        assert read_lock(lock.path)["pid"] == os.getpid()
    assert not path.exists()


def test_racing_runs_on_a_stale_lock_never_hold_it_together(tmp_path: Path) -> None:
    """Many runs racing to replace one stale lock: at most one holds it at any moment."""
    import threading
    import time

    _plant(tmp_path, pid=_dead_pid(), host=platform.node(), started_at="then")
    start = threading.Barrier(8)
    guard = threading.Lock()
    holding = 0
    most = 0
    won = 0

    def contend() -> None:
        nonlocal holding, most, won
        start.wait()
        for _ in range(20):
            try:
                with campaign_lock(tmp_path):
                    with guard:
                        holding += 1
                        most = max(most, holding)
                        won += 1
                    time.sleep(0.001)
                    with guard:
                        holding -= 1
            except CampaignLockError:
                pass

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert won >= 1
    assert most == 1, "two runs held the lock at once"
    assert not (tmp_path / LOCK_NAME).exists()


def test_break_lock_through_the_cli_replaces_the_lock_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rpg_economy.campaign import parallel
    from rpg_economy.campaign.parallel import ParallelResult
    from rpg_economy.i18n import t

    seen: list[dict[str, object]] = []

    def run(config, plan, directory, **kwargs):
        seen.append(read_lock(Path(directory) / LOCK_NAME))
        return ParallelResult(directory=Path(directory), workers=1)

    monkeypatch.setattr(parallel, "run_campaign_parallel", run)
    out = tmp_path / "runs"
    foreign = {"pid": 4242, "host": "another-host", "started_at": "then"}
    _plant(out, **foreign)
    argv = [
        "--locale",
        "es",
        "campaign-run",
        "--config",
        str(_config_file(tmp_path)),
        "--phase",
        "pilot",
        "--out",
        str(out),
    ]

    assert main(argv) == 10, "a lock from another host was replaced without --break-lock"
    capsys.readouterr()
    assert main([*argv, "--break-lock"]) == 0

    err = capsys.readouterr().err
    assert (
        t(
            "cli.campaign.lock_replaced",
            locale="es",
            path=str(out / LOCK_NAME),
            pid=4242,
            host="another-host",
            since="then",
        )
        in err
    )
    assert seen and seen[0]["pid"] == os.getpid(), "the run did not hold the lock"
    assert not (out / LOCK_NAME).exists()


def test_a_stale_lock_is_replaced_without_break_lock_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rpg_economy.campaign import parallel
    from rpg_economy.campaign.parallel import ParallelResult

    monkeypatch.setattr(
        parallel,
        "run_campaign_parallel",
        lambda config, plan, directory, **kwargs: ParallelResult(Path(directory), 1),
    )
    out = tmp_path / "runs"
    pid = _dead_pid()
    _plant(out, pid=pid, host=platform.node(), started_at="then")

    config = str(_config_file(tmp_path))
    code = main(["campaign-run", "--config", config, "--phase", "pilot", "--out", str(out)])

    assert code == 0
    err = capsys.readouterr().err
    assert f"pid {pid}" in err
    assert "Replaced the lock" in err


def test_a_free_flock_takes_over_a_local_lock_whose_pid_was_reused(tmp_path: Path) -> None:
    """The recorded pid is alive, but it does not hold the flock, so it is not the owner."""
    reused = os.getppid()  # alive, and not this process
    _plant(tmp_path, pid=reused, host=platform.node(), started_at="then")
    assert not is_stale(read_lock(tmp_path / LOCK_NAME))

    with campaign_lock(tmp_path) as lock:
        assert lock.replaced == {"pid": reused, "host": platform.node(), "started_at": "then"}
        assert lock.pid_reused
        assert read_lock(lock.path)["pid"] == os.getpid()
    assert not (tmp_path / LOCK_NAME).exists()


def test_a_dead_owner_is_not_reported_as_a_reused_pid(tmp_path: Path) -> None:
    _plant(tmp_path, pid=_dead_pid(), host=platform.node(), started_at="then")
    with campaign_lock(tmp_path) as lock:
        assert lock.replaced is not None and not lock.pid_reused


def test_a_reused_pid_is_taken_over_through_the_cli_with_a_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rpg_economy.campaign import parallel
    from rpg_economy.campaign.parallel import ParallelResult
    from rpg_economy.i18n import t

    monkeypatch.setattr(
        parallel,
        "run_campaign_parallel",
        lambda config, plan, directory, **kwargs: ParallelResult(Path(directory), 1),
    )
    out = tmp_path / "runs"
    reused = os.getppid()
    _plant(out, pid=reused, host=platform.node(), started_at="then")

    config = str(_config_file(tmp_path))
    argv = ["--locale", "es", "campaign-run", "--config", config, "--phase", "pilot"]
    assert main([*argv, "--out", str(out)]) == 0, "a free local flock needed --break-lock"

    err = capsys.readouterr().err
    params = {
        "path": str(out / LOCK_NAME),
        "pid": reused,
        "host": platform.node(),
        "since": "then",
    }
    assert t("cli.campaign.lock_replaced", locale="es", **params) in err
    assert t("cli.campaign.lock_pid_reused", locale="es", **params) in err


def test_a_filesystem_without_flock_is_refused_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import errno

    from rpg_economy.campaign import lock as lock_module
    from rpg_economy.campaign.lock import CampaignLockUnavailable
    from rpg_economy.i18n import t

    def no_locks(descriptor: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

    monkeypatch.setattr(lock_module.fcntl, "flock", no_locks)
    with pytest.raises(CampaignLockUnavailable) as info, campaign_lock(tmp_path / "a"):
        pytest.fail("locked without a flock")
    assert info.value.errno == errno.ENOLCK

    out = tmp_path / "runs"
    config = str(_config_file(tmp_path))
    code = main(["campaign-run", "--config", config, "--phase", "pilot", "--out", str(out)])
    assert code == 10
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert (
        t(
            "error.campaign.lock_unavailable",
            locale="en",
            path=str(out / LOCK_NAME),
            detail=os.strerror(errno.ENOLCK),
        )
        in err
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_directory_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy.campaign.lock import CampaignLockUnavailable

    out = tmp_path / "runs"
    out.mkdir()
    config = str(_config_file(tmp_path))
    out.chmod(0o500)
    try:
        with pytest.raises(CampaignLockUnavailable), campaign_lock(out):
            pytest.fail("locked an unwritable directory")
        code = main(["campaign-run", "--config", config, "--phase", "pilot", "--out", str(out)])
    finally:
        out.chmod(0o700)
    assert code == 10
    err = capsys.readouterr().err
    assert "Traceback" not in err and str(out / LOCK_NAME) in err


def test_a_lock_that_cannot_be_proven_released_says_break_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy.i18n import t

    out = tmp_path / "runs"
    _plant(out, pid=4242, host="another-host", started_at="then")
    config = str(_config_file(tmp_path))
    code = main(["campaign-run", "--config", config, "--phase", "pilot", "--out", str(out)])
    assert code == 10
    err = capsys.readouterr().err
    assert (
        t(
            "error.campaign.lock_unproven",
            locale="en",
            path=str(out / LOCK_NAME),
            pid=4242,
            host="another-host",
            since="then",
        )
        in err
    )
