"""Command-line entry point.

Every user-facing string this module emits -- argparse help text included --
resolves through ``i18n.t`` (IMPLEMENTATION.md section 2). Because argparse renders its
help while the parser is being built, the locale is resolved from ``argv``
before the parser exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

from rpg_economy.campaign.interrupt import Interrupt
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime
from rpg_economy.i18n import SUPPORTED_LOCALES, format_number, t

DEFAULT_CONFIG = Path("configs/base.yaml")


def peek_locale(argv: Sequence[str]) -> str | None:
    """Extract ``--locale`` from ``argv`` before argparse runs.

    argparse builds its help text eagerly, so the catalog has to be selected
    first. Returns None when the flag is absent, letting the usual
    ``SIM_LOCALE`` then ``en`` fallback apply.
    """
    for index, token in enumerate(argv):
        if token == "--locale" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--locale="):
            return token.split("=", 1)[1]
    return None


class _HelpFormatter(argparse.HelpFormatter):
    """argparse's formatter, with its ``usage:`` prefix from the catalog."""

    def add_usage(self, usage, actions, groups, prefix=None) -> None:  # type: ignore[no-untyped-def]
        super().add_usage(
            usage, actions, groups, t("cli.argparse.usage") if prefix is None else prefix
        )


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser whose own chrome is translated too.

    argparse words its section titles, its ``-h`` help and the ``usage:``
    prefix in English, through ``gettext``; here they come from the catalog,
    so ``--locale es --help`` is Spanish throughout. Subcommand parsers are
    built with this class as well (``add_subparsers`` uses the parent's).
    """

    def __init__(self, *args, add_help: bool = True, **kwargs) -> None:  # type: ignore[no-untyped-def]
        kwargs.setdefault("formatter_class", _HelpFormatter)
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = t("cli.argparse.positionals")
        self._optionals.title = t("cli.argparse.options")
        if add_help:
            self.add_argument(
                "-h",
                "--help",
                action="help",
                default=argparse.SUPPRESS,
                help=t("cli.argparse.help"),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="rpg-economy", description=t("app.description"))
    parser.add_argument(
        "--locale",
        choices=SUPPORTED_LOCALES,
        default=None,
        help=t("cli.help.locale"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("config-show", help=t("cli.help.config_show"))
    show.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    show.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))

    check = sub.add_parser("economy-check", help=t("cli.help.economy_check"))
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    check.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    check.add_argument("--seed", type=int, default=None, help=t("cli.help.seed"))
    check.add_argument("--scenario", default=None, help=t("cli.help.scenario"))
    check.add_argument("--log", type=Path, default=None, help=t("cli.help.log"))
    check.add_argument("--shock-efficacy", action="store_true", help=t("cli.help.shock_efficacy"))
    check.add_argument(
        "--pilot-logs", type=Path, default=None, help=t("cli.help.efficacy_pilot_logs")
    )
    check.add_argument("--allow-mock", action="store_true", help=t("cli.help.efficacy_allow_mock"))

    budget = sub.add_parser("derive-budget", help=t("cli.help.derive_budget"))
    budget.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    budget.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    budget.add_argument("--pilot-logs", type=Path, required=True, help=t("cli.help.pilot_logs"))
    budget.add_argument("--out", type=Path, default=None, help=t("cli.help.budget_out"))
    budget.add_argument("--layer-out", type=Path, default=None, help=t("cli.help.budget_layer_out"))
    budget.add_argument("--allow-mock", action="store_true", help=t("cli.help.allow_mock"))
    budget.add_argument("--force", action="store_true", help=t("cli.help.budget_force"))

    plan = sub.add_parser("campaign-plan", help=t("cli.help.campaign_plan"))
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    plan.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    plan.add_argument(
        "--phase", choices=("pilot", "final"), default="pilot", help=t("cli.help.phase")
    )
    plan.add_argument("--profile-logs", type=Path, default=None, help=t("cli.help.profile_logs"))
    plan.add_argument("--out", type=Path, default=None, help=t("cli.help.plan_out"))

    campaign = sub.add_parser("campaign-run", help=t("cli.help.campaign_run"))
    campaign.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    campaign.add_argument("--real", action="store_true", help=t("cli.help.real"))
    campaign.add_argument(
        "--layer", type=Path, action="append", default=[], help=t("cli.help.layer")
    )
    # Required: a default phase would let a forgotten flag run the wrong seed set.
    campaign.add_argument(
        "--phase", choices=("pilot", "final"), required=True, help=t("cli.help.phase")
    )
    campaign.add_argument("--out", type=Path, required=True, help=t("cli.help.runs_out"))
    campaign.add_argument(
        "--profile-logs", type=Path, default=None, help=t("cli.help.profile_logs")
    )
    campaign.add_argument(
        "--confirm-cost",
        default=None,
        help=t("cli.help.confirm_cost"),
    )
    campaign.add_argument("--no-resume", action="store_true", help=t("cli.help.no_resume"))
    campaign.add_argument(
        "--condition", choices=["A", "B", "C"], default=None, help=t("cli.help.condition")
    )
    campaign.add_argument("--workers", type=int, default=8, help=t("cli.help.workers"))
    campaign.add_argument(
        "--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set")
    )
    campaign.add_argument("--break-lock", action="store_true", help=t("cli.help.break_lock"))

    derisk = sub.add_parser("derisk", help=t("cli.help.derisk"))
    derisk.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    derisk.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    derisk.add_argument("--out", type=Path, required=True, help=t("cli.help.runs_out"))
    derisk.add_argument("--seed", type=int, default=1001, help=t("cli.help.seed"))
    derisk.add_argument("--horizon", type=int, default=120, help=t("cli.help.horizon"))
    derisk.add_argument(
        "--models", default="gpt-5-nano,gpt-5-mini", help=t("cli.help.derisk_models")
    )
    derisk.add_argument("--profile-logs", type=Path, default=None, help=t("cli.help.profile_logs"))
    derisk.add_argument("--confirm-cost", default=None, help=t("cli.help.confirm_cost"))
    derisk.add_argument("--model-pricing", default=None, help=t("cli.help.model_pricing"))
    derisk.add_argument(
        "--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set")
    )
    derisk.add_argument("--report-only", action="store_true", help=t("cli.help.report_only"))
    derisk.add_argument(
        "--allow-final-seed", action="store_true", help=t("cli.help.allow_final_seed")
    )
    derisk.add_argument("--break-lock", action="store_true", help=t("cli.help.break_lock"))

    smoke = sub.add_parser("smoke", help=t("cli.help.smoke"))
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    smoke.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    smoke.add_argument("--out", type=Path, required=True, help=t("cli.help.runs_out"))
    smoke.add_argument("--seed", type=int, default=1004, help=t("cli.help.seed"))
    smoke.add_argument(
        "--budget-fraction", type=float, default=0.2, help=t("cli.help.budget_fraction")
    )
    smoke.add_argument("--confirm-cost", default=None, help=t("cli.help.confirm_cost"))
    smoke.add_argument("--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set"))
    smoke.add_argument("--report-only", action="store_true", help=t("cli.help.smoke_report_only"))
    smoke.add_argument(
        "--allow-final-seed", action="store_true", help=t("cli.help.allow_final_seed")
    )
    smoke.add_argument("--break-lock", action="store_true", help=t("cli.help.break_lock"))

    pilot = sub.add_parser("pilot", help=t("cli.help.pilot"))
    pilot.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    pilot.add_argument("--real", action="store_true", help=t("cli.help.real"))
    pilot.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    pilot.add_argument(
        "--stage",
        choices=("calibration", "budget-derivation"),
        required=True,
        help=t("cli.help.pilot_stage"),
    )
    pilot.add_argument("--out", type=Path, required=True, help=t("cli.help.runs_out"))
    pilot.add_argument("--profile-logs", type=Path, default=None, help=t("cli.help.profile_logs"))
    pilot.add_argument(
        "--condition", choices=["A", "B", "C"], default=None, help=t("cli.help.condition")
    )
    pilot.add_argument("--workers", type=int, default=8, help=t("cli.help.workers"))
    pilot.add_argument("--confirm-cost", default=None, help=t("cli.help.confirm_cost"))
    pilot.add_argument("--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set"))
    pilot.add_argument("--no-resume", action="store_true", help=t("cli.help.no_resume"))
    pilot.add_argument("--verify-only", action="store_true", help=t("cli.help.verify_only"))
    pilot.add_argument("--break-lock", action="store_true", help=t("cli.help.break_lock"))
    pilot.add_argument("--seed", type=int, default=None, help=t("cli.help.pilot_seed"))
    pilot.add_argument("--scenario", default=None, help=t("cli.help.pilot_scenario"))

    trio = sub.add_parser("trio", help=t("cli.help.trio"))
    trio.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    trio.add_argument("--real", action="store_true", help=t("cli.help.real"))
    trio.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    trio.add_argument("--seed", type=int, default=None, help=t("cli.help.seed"))
    trio.add_argument("--scenario", default=None, help=t("cli.help.scenario"))
    trio.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentRegime],
        default=None,
        help=t("cli.help.experiment"),
    )
    trio.add_argument("--out", type=Path, required=True, help=t("cli.help.trio_out"))
    trio.add_argument("--good", default=None, help=t("cli.help.replay_good"))
    trio.add_argument("--no-replay", action="store_true", help=t("cli.help.no_replay"))
    trio.add_argument("--profile-logs", type=Path, default=None, help=t("cli.help.profile_logs"))
    trio.add_argument("--confirm-cost", default=None, help=t("cli.help.confirm_cost"))
    trio.add_argument("--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set"))
    trio.add_argument(
        "--allow-final-seed", action="store_true", help=t("cli.help.allow_final_seed")
    )
    trio.add_argument("--break-lock", action="store_true", help=t("cli.help.break_lock"))

    behavior = sub.add_parser("behavior-check", help=t("cli.help.behavior_check"))
    behavior.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    behavior.add_argument(
        "--layer", type=Path, action="append", default=[], help=t("cli.help.layer")
    )
    behavior.add_argument("--real", action="store_true", help=t("cli.help.real"))
    behavior.add_argument("--reps", type=int, default=8, help=t("cli.help.behavior_reps"))
    behavior.add_argument("--only", default=None, help=t("cli.help.behavior_only"))
    behavior.add_argument("--workers", type=int, default=8, help=t("cli.help.workers"))
    behavior.add_argument("--out", type=Path, default=None, help=t("cli.help.behavior_out"))
    behavior.add_argument("--confirm-cost", default=None, help=t("cli.help.confirm_cost"))
    behavior.add_argument(
        "--spend-limit-set", action="store_true", help=t("cli.help.spend_limit_set")
    )
    behavior.add_argument(
        "--from-log",
        type=Path,
        action="append",
        default=[],
        help=t("cli.help.behavior_from_log"),
    )
    behavior.add_argument("--moments", type=int, default=3, help=t("cli.help.behavior_moments"))

    balance = sub.add_parser("balance", help=t("cli.help.balance"))
    balance.add_argument("logs", type=Path, nargs="+", help=t("cli.help.balance_logs"))

    audit = sub.add_parser("behavior-audit", help=t("cli.help.behavior_audit"))
    audit.add_argument("logs", type=Path, nargs="+", help=t("cli.help.behavior_audit_logs"))
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    audit.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    audit.add_argument("--out", type=Path, default=None, help=t("cli.help.behavior_audit_out"))

    report = sub.add_parser("report", help=t("cli.help.report"))
    report.add_argument("logs", type=Path, help=t("cli.help.report_logs"))
    report.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    report.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    report.add_argument("--out", type=Path, default=Path("reports"), help=t("cli.help.report_out"))
    report.add_argument("--include-pilot", action="store_true", help=t("cli.help.include_pilot"))
    report.add_argument("--allow-mock", action="store_true", help=t("cli.help.report_allow_mock"))

    paper = sub.add_parser("paper-figures", help=t("cli.help.paper_figures"))
    paper.add_argument("logs", type=Path, help=t("cli.help.report_logs"))
    paper.add_argument("--results", type=Path, required=True, help=t("cli.help.paper_results"))
    paper.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.config"))
    paper.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))
    paper.add_argument(
        "--out", type=Path, default=Path("reports/paper"), help=t("cli.help.paper_out")
    )
    paper.add_argument("--workers", type=int, default=8, help=t("cli.help.paper_workers"))

    dashboard = sub.add_parser("dashboard", help=t("cli.help.dashboard"))
    dashboard.add_argument("log", type=Path, help=t("cli.help.log_path"))

    check = sub.add_parser("check", help=t("cli.help.check"))
    check.add_argument("logs", type=Path, help=t("cli.help.check_logs"))

    replay = sub.add_parser("replay", help=t("cli.help.replay"))
    replay.add_argument("source", type=Path, help=t("cli.help.replay_source"))
    replay.add_argument("--out", type=Path, default=None, help=t("cli.help.replay_out"))
    replay.add_argument("--good", default=None, help=t("cli.help.replay_good"))
    replay.add_argument("--all-locales", action="store_true", help=t("cli.help.replay_all_locales"))
    replay.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=t("cli.help.replay_config")
    )
    replay.add_argument("--layer", type=Path, action="append", default=[], help=t("cli.help.layer"))

    return parser


def with_real_provider(config: SimulationConfig) -> SimulationConfig:
    """The same setup, executed against the real provider instead of the mock."""
    return config.model_copy(update={"llm": config.llm.model_copy(update={"mock": False})})


def _localized(message: object) -> str:
    """A campaign or analysis error, or a plan notice, in the reader's language.

    ``PlanError``, ``CostError``, ``AnalysisError`` and plan ``Notice`` carry a
    catalog key and its parameters; their English text is for library callers
    and payloads only. Anything without a key is printed as it is.
    """
    key = getattr(message, "key", None)
    if key is None:
        return str(message)
    params = {
        name: ", ".join(map(str, value)) if isinstance(value, list | tuple) else value
        for name, value in dict(getattr(message, "params", {})).items()
    }
    for name, prefix in (
        ("scenario", "scenario"),
        ("experiment", "experiment"),
        ("condition", "condition"),
        ("tier", "campaign.tier"),
    ):
        if name in params:
            params[name] = t(f"{prefix}.{params[name]}")
    return t(key, **params)


def _yes_no(value: bool) -> str:
    return t("common.yes" if value else "common.no")


def _stopped(directory: Path) -> int:
    """Report a batch ended by Ctrl-C: finished runs kept, a rerun resumes."""
    print(t("cli.interrupt.stopped", path=str(directory)), file=sys.stderr)
    return 130


def _locked(
    directory: Path,
    body: Callable[[], int],
    *,
    break_lock: bool,
    interrupt: Interrupt | None = None,
) -> int:
    """Run ``body`` holding the lock on ``directory``; refuse if another run holds it.

    Two runs on one output directory would each pay for the same pending trios.
    With ``interrupt``, Ctrl-C only raises its flags while ``body`` runs (a
    first one stops new work, a second aborts the running conditions), so
    ``body`` returns -- and the lock is released -- only once every worker has
    joined. A third exits the process at once (status 130); the kernel
    releases the flock.
    """
    from contextlib import nullcontext

    from rpg_economy.campaign.lock import (
        CampaignLockError,
        CampaignLockUnavailable,
        campaign_lock,
    )

    try:
        with campaign_lock(directory, break_lock=break_lock) as lock:
            if lock.replaced is not None:
                holder = {
                    "path": str(lock.path),
                    "pid": lock.replaced.get("pid", "?"),
                    "host": lock.replaced.get("host", "?"),
                    "since": lock.replaced.get("started_at", "?"),
                }
                print(t("cli.campaign.lock_replaced", **holder), file=sys.stderr)
                if lock.pid_reused:
                    print(t("cli.campaign.lock_pid_reused", **holder), file=sys.stderr)
            handling = (
                interrupt.handling_sigint(
                    stopping=t("cli.interrupt.stopping"),
                    aborting=t("cli.interrupt.aborting"),
                    exiting=t("cli.interrupt.exiting"),
                )
                if interrupt is not None
                else nullcontext()
            )
            with handling:
                return body()
    except CampaignLockUnavailable as exc:
        print(
            t("error.campaign.lock_unavailable", path=str(exc.path), detail=exc.detail),
            file=sys.stderr,
        )
        return 10
    except CampaignLockError as exc:
        print(
            t(
                "error.campaign.locked" if exc.held else "error.campaign.lock_unproven",
                path=str(exc.path),
                pid=exc.holder.get("pid", "?"),
                host=exc.holder.get("host", "?"),
                since=exc.holder.get("started_at", "?"),
            ),
            file=sys.stderr,
        )
        return 10


def _refuse_final_seed(config: SimulationConfig, seed: int, *, allowed: bool) -> bool:
    """Whether a one-off real run on ``seed`` must be refused, printing why.

    A final seed run before the campaign is a look at final data: its outcome
    could inform choices the final runs are meant to be blind to (design v1.0
    s11.4). Allowed only when asked for explicitly.
    """
    if allowed or seed not in config.seeds.final:
        return False
    print(t("error.final_seed.refused", seed=seed), file=sys.stderr)
    return True


def _refuse_mock_budget(config: SimulationConfig) -> bool:
    """Whether a real batch must be refused for its Experiment 2 budget, printing why.

    ``derive-budget --allow-mock`` exercises the pipeline offline; the number it
    writes measures the mock, not the model, so no real run may spend against it
    (design v1.0 s8.4: the budget comes from real pilot consumption). Nor may a
    budget derive-budget did not write, or one derived under other code.
    """
    if config.budget.derived_from_mock:
        print(t("error.budget.mock_derived"), file=sys.stderr)
        return True
    if config.budget.experiment_2_total_tokens is None:
        return False
    # The budget must come from derive-budget, under the code that will spend
    # it: a hand-written number, or one derived before the code changed, would
    # not be the pilot's real consumption (design v1.0 s8.4).
    derived = config.budget.derived_code_digest
    if derived is None:
        print(t("error.budget.not_derived"), file=sys.stderr)
        return True
    from rpg_economy.logging_.metadata import code_digest

    if derived != code_digest():
        print(t("error.budget.code_changed"), file=sys.stderr)
        return True
    return False


def _unusable_run(exc: BaseException, directory: Path) -> int:
    """Report a one-off run whose logs broke or whose trio failed the gate.

    Localized, and without a traceback: the logs are kept where they are, and
    the message says so.
    """
    from rpg_economy.integrity import TrioIntegrityError

    if isinstance(exc, TrioIntegrityError):
        print(t("error.trio.integrity", path=str(directory), detail=str(exc)), file=sys.stderr)
        return 7
    print(t("error.run.log_unusable", path=str(directory)), file=sys.stderr)
    return 6


def _paper_figures(config: object, logs: Path, results: Path, out: Path, *, workers: int) -> int:
    """Draw the paper's figures, in every language, from run logs and the report's results."""
    from rpg_economy.analysis.paper_figures import generate

    if not results.is_file():
        print(t("error.paper.no_results", path=str(results)), file=sys.stderr)
        return 2
    paths = sorted(p for c in ("A", "B", "C") for p in logs.rglob(f"{c}.jsonl"))
    if not paths:
        print(t("error.report.no_logs", path=str(logs)), file=sys.stderr)
        return 2
    written = generate(paths, results, config, out, workers=workers)
    for locale, files in written.items():
        print(t("cli.paper.written", count=len(files), language=locale, path=str(out / locale)))
    return 0


def _report(
    config: object,
    logs: Path,
    out: Path,
    *,
    include_pilot: bool,
    allow_mock: bool,
) -> int:
    """Analyse a directory of run logs and write the report in every language.

    One report per scenario and experiment: contrasts are per scenario (design
    s11.4) and the two experiments are never pooled.
    """
    import warnings

    from rpg_economy.analysis.pipeline import AnalysisError, IncompleteTrioWarning, analyse
    from rpg_economy.analysis.report import generate_all_locales

    try:
        # Reported below in the user's language instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncompleteTrioWarning)
            analysis = analyse(logs, config, include_pilot=include_pilot, allow_mock=allow_mock)
    except (AnalysisError, ValueError) as exc:
        # An AnalysisError names its catalog message; anything else is shown as is.
        print(t("error.report.invalid", detail=_localized(exc)), file=sys.stderr)
        return 2

    if analysis.skipped_seeds:
        print(
            t("cli.report.skipped_seeds", seeds=", ".join(map(str, analysis.skipped_seeds))),
            file=sys.stderr,
        )
    for (scenario, experiment), seeds in analysis.incomplete_trios.items():
        print(
            t(
                "cli.report.incomplete_trios",
                scenario=t(f"scenario.{scenario}"),
                experiment=t(f"experiment.{experiment}"),
                seeds=", ".join(map(str, seeds)),
            ),
            file=sys.stderr,
        )
    _print_analysis_exclusions(analysis)
    for (scenario, experiment), conditions in analysis.skipped_cells.items():
        print(
            t(
                "cli.report.skipped_cell",
                scenario=t(f"scenario.{scenario}"),
                experiment=t(f"experiment.{experiment}"),
                conditions=", ".join(conditions),
            ),
            file=sys.stderr,
        )
    for (scenario, experiment), results in analysis.results.items():
        paths = generate_all_locales(
            results,
            out / f"{scenario}-{experiment}",
            price_series=analysis.price_series[(scenario, experiment)],
            shock_tick=config.phases.shock_tick,
            reversion_tick=config.phases.reversion_tick,
        )
        for path in paths.values():
            print(t("report.written", path=str(path)))
    return 0


def _print_analysis_exclusions(analysis: object) -> None:
    """One stderr line per kind of exclusion: skipped logs, rejected trios, refused cells.

    The report states each in full; this only says, in the reader's language,
    that something was left out and why.
    """

    def cell(key: tuple[str, str]) -> str:
        return f"{t(f'scenario.{key[0]}')} / {t(f'experiment.{key[1]}')}"

    skipped = getattr(analysis, "skipped_logs", None) or []
    if skipped:
        reasons: dict[str, int] = {}
        for entry in skipped:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
        summary = "; ".join(
            f"{t(f'report.skip_reason.{reason}')}: {format_number(count)}"
            for reason, count in sorted(reasons.items())
        )
        print(
            t("report.note.skipped_logs", count=format_number(len(skipped)), reasons=summary),
            file=sys.stderr,
        )
    rejected = [
        (key, entry)
        for key, entries in sorted((getattr(analysis, "rejected_trios", None) or {}).items())
        for entry in entries
    ]
    if rejected:
        trios = "; ".join(
            f"{cell(key)} {entry['seed']} ({t('report.reject_reason.' + entry['reason'])})"
            for key, entry in rejected
        )
        print(
            t("report.note.rejected_trios", count=format_number(len(rejected)), trios=trios),
            file=sys.stderr,
        )
    refused = getattr(analysis, "refused_cells", None) or {}
    if refused:
        cells = "; ".join(
            f"{cell(key)} ({', '.join(fields)})" for key, fields in sorted(refused.items())
        )
        print(
            t("report.note.refused_cells", count=format_number(len(refused)), cells=cells),
            file=sys.stderr,
        )


def _derive_budget(
    config: object,
    pilot_logs: Path,
    out: Path | None,
    *,
    layer_out: Path | None = None,
    allow_mock: bool = False,
    force: bool = False,
) -> int:
    """Derive the Experiment 2 budget from pilot logs and record it.

    The logs are checked against the loaded configuration (pilot seeds, full
    horizon, config hash), so the calibration stage or a stray final run cannot
    set the budget. An existing ``--layer-out`` is overwritten only with
    ``--force``: it may be the budget the final runs are loaded with.
    """
    import json

    from rpg_economy.analysis.budget_derivation import derive_budget, find_logs
    from rpg_economy.analysis.errors import AnalysisError
    from rpg_economy.logging_.metadata import code_digest

    if layer_out is not None and layer_out.exists() and not force:
        print(t("error.budget.layer_exists", path=str(layer_out)), file=sys.stderr)
        return 4

    try:
        derivation = derive_budget(find_logs(pilot_logs), config=config, allow_mock=allow_mock)
    except AnalysisError as exc:  # BudgetDerivationError included
        # Named in the catalog (``key``, ``params``) like every analysis error.
        print(t("error.budget.underivable", detail=_localized(exc)), file=sys.stderr)
        return 4

    print(
        t(
            "cli.budget.derived",
            budget=derivation.budget,
            runs=len(derivation.runs),
            scenario=t("scenario.resource_scarcity"),
        )
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Marked as the layer is: a number measured on the mock must not pass
        # for the real budget once the JSON travels without the layer.
        payload = {
            **derivation.to_payload(),
            "derived_from_mock": allow_mock,
            "derived_code_digest": code_digest(),
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(t("cli.budget.written", path=str(out)))
    if layer_out is not None:
        layer_out.parent.mkdir(parents=True, exist_ok=True)
        # A budget from mock runs only exercises the pipeline. The layer says
        # so in a field, not only a comment, so it cannot pass for the real one
        # once it is copied around: a real campaign or pilot refuses it.
        marker = (
            "# From mock runs (--allow-mock): tests the pipeline, not a real budget.\n"
            if allow_mock
            else ""
        )
        layer_out.write_text(
            "# Derived by derive-budget from the pilot logs (design v1.0 s8.4).\n"
            + marker
            + f"budget:\n  experiment_2_total_tokens: {derivation.budget}\n"
            + ("  derived_from_mock: true\n" if allow_mock else "")
            + f"  derived_code_digest: {code_digest()}\n",
            encoding="utf-8",
        )
        print(t("cli.budget.layer_written", path=str(layer_out)))
    return 0


def _economy_check(config: object, log_path: Path | None) -> int:
    """Run a rule-based simulation and print its economic sanity check."""
    import tempfile

    from rpg_economy.agents.decision_phase import RuleBasedActivation
    from rpg_economy.analysis import analyse, render, target_good_for
    from rpg_economy.cli_support import build_metadata
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.orchestration import DecentralizedOrchestrator

    destination = log_path or Path(tempfile.mkdtemp()) / "economy_check.jsonl"
    initial = build_world(config)
    state = build_world(config)

    # The experiment's activation regime (triggers, scheduler, K per cycle), so
    # the economy checked is the one the conditions will actually run.
    with EventLogWriter(destination, build_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run(DecentralizedOrchestrator(config, engine, RuleBasedActivation(config)))

    check = analyse(destination, config, state, initial, target_good_for(config))
    print(render(check, config))
    print()
    from rpg_economy.analysis.balance import balance_from_log, render_balance

    print(render_balance(balance_from_log(destination)))
    print()
    with destination.open(encoding="utf-8") as handle:
        events = sum(1 for _ in handle)
    print(t("cli.run.log_written", path=str(destination), events=events))
    return 0


def _shock_efficacy(
    config: object,
    pilot_logs: Path | None,
    *,
    scenario_given: bool = False,
    allow_mock: bool = False,
) -> int:
    """Report whether each shock moves its target.

    Without logs: each pilot seed runs offline with and without its shock, one
    block per scenario (``efficacy_scenarios``).
    With pilot logs: each finished run's shock phase against its own base phase;
    mock runs are left out unless ``allow_mock``.
    """
    from rpg_economy.analysis.shock_efficacy import (
        efficacy_scenarios,
        from_logs,
        measure_scenario,
    )
    from rpg_economy.logging_ import run_logs

    if pilot_logs is None:
        seeds = list(config.seeds.pilot)
        # The primary scenario and every configured secondary one: a secondary
        # shock that does not bite would otherwise reach the final campaign
        # unchecked. --scenario narrows the check to that one.
        scenarios = [config.run.scenario] if scenario_given else efficacy_scenarios(config)
        all_effective = True
        for index, scenario in enumerate(scenarios):
            if index:
                print()
            print(
                t("cli.efficacy.title", scenario=t(f"scenario.{scenario.value}"), seeds=len(seeds))
            )
            # Every block is printed before the verdict, so one ineffective
            # scenario does not hide the others.
            all_effective &= _render_efficacy(measure_scenario(config, scenario, seeds=seeds))
        return 0 if all_effective else 9

    groups, unmatched = from_logs(run_logs(pilot_logs), config, allow_mock=allow_mock)
    if unmatched:
        print(t("cli.efficacy.unmatched", count=len(unmatched)), file=sys.stderr)
    if not groups:
        print(t("cli.efficacy.no_logs", path=str(pilot_logs)), file=sys.stderr)
        return 2
    all_effective = True
    for (scenario, condition), efficacy in groups.items():
        print(
            t(
                "cli.efficacy.title_logs",
                scenario=t(f"scenario.{scenario}"),
                condition=t(f"condition.{condition}"),
                runs=len(efficacy.effects),
            )
        )
        all_effective &= _render_efficacy(efficacy)
        print()
    return 0 if all_effective else 9


def _render_efficacy(efficacy: object) -> bool:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:+.0f}%"

    def count(value: float | None) -> str:
        return "-" if value is None else f"{value:+g}"

    for effect in efficacy.effects:
        print(
            t(
                "cli.efficacy.seed",
                seed=effect.seed,
                price=pct(effect.price_change),
                stockouts=count(effect.stockout_change),
                supply=pct(effect.supply_change),
                value=pct(effect.value_change),
            )
        )
    print(
        t(
            "cli.efficacy.median",
            price=pct(efficacy.median("price_change")),
            stockouts=count(efficacy.median("stockout_change")),
            supply=pct(efficacy.median("supply_change")),
            value=pct(efficacy.median("value_change")),
        )
    )
    effective = efficacy.effective()
    print(t("cli.efficacy.effective" if effective else "cli.efficacy.ineffective"))
    return effective


def _confirm_real_batch(
    config: SimulationConfig,
    estimate: object | None,
    *,
    confirm_cost: str | None,
    spend_limit_set: bool,
    runs: int,
) -> int:
    """The gate a real batch passes before it spends; 0 to go ahead, else 5.

    A real budget, a priced batch, an acknowledged provider-side spend limit,
    and the printed total confirmed by value.
    """
    if _refuse_mock_budget(config):
        return 5
    if estimate is None:
        print(t("error.campaign.unpriced"), file=sys.stderr)
        return 5
    if not spend_limit_set:
        print(t("error.derisk.no_spend_limit"), file=sys.stderr)
        return 5
    if confirm_cost != str(estimate.total_cost):
        print(
            t(
                "error.campaign.unconfirmed",
                total=str(estimate.total_cost),
                currency=estimate.currency,
                runs=runs,
            ),
            file=sys.stderr,
        )
        return 5
    return 0


def _print_failed_trios(result: object) -> None:
    """Why each failed trio failed, one line each, on stderr."""
    for outcome in result.failed:
        print(
            t(
                "error.campaign.trio_failed",
                scenario=t(f"scenario.{outcome.trio.scenario.value}"),
                seed=outcome.trio.seed,
                detail=outcome.detail,
            ),
            file=sys.stderr,
        )


def _print_billing(report: dict) -> None:
    """The billing verification line, and the spend of unfinished and set-aside logs."""
    print(
        t(
            "cli.pilot.billing",
            runs=report["runs"],
            real=_yes_no(report["all_real"]),
            billed=_yes_no(report["all_fully_billed"]),
            total=format_number(report["billed_total"]),
        )
    )
    unfinished = report.get("unfinished_spend") or {}
    if unfinished.get("logs"):
        print(
            t(
                "cli.pilot.billing_unfinished",
                logs=unfinished["logs"],
                total=format_number(unfinished["billed_total"]),
            )
        )
    set_aside = report.get("set_aside") or {}
    if set_aside.get("logs"):
        print(
            t(
                "cli.pilot.billing_set_aside",
                logs=set_aside["logs"],
                total=format_number(set_aside["billed_total"]),
            )
        )
    if unfinished.get("logs") or set_aside.get("logs"):
        total = report["billed_total"] + unfinished.get("billed_total", 0)
        total += set_aside.get("billed_total", 0)
        print(
            t("cli.pilot.billing_total", total=format_number(report.get("billed_total_all", total)))
        )


def _write_billing(directory: Path, *, expected_runs: int) -> dict:
    """Verify a batch's billing, print it, and write it to ``billing.json``."""
    from rpg_economy.campaign.pilot import verify_stage_billing

    report = verify_stage_billing(directory, expected_runs=expected_runs)
    _print_billing(report)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "billing.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _pending_changed(directory: Path, confirmed: object, current: object) -> bool:
    """Whether what is pending now, under the lock, left the confirmed set; prints why.

    The batch was priced and confirmed on ``confirmed`` before the lock was
    taken. Trios that finished meanwhile only shrink it; anything pending now
    that was not confirmed would be spent without having been shown.
    """
    allowed = set(confirmed.trios)
    unconfirmed = [trio for trio in current.trios if trio not in allowed]
    if not unconfirmed:
        return False
    print(
        t(
            "error.campaign.pending_changed",
            path=str(directory),
            confirmed=len(confirmed.trios),
            pending=len(current.trios),
            unconfirmed=len(unconfirmed),
        ),
        file=sys.stderr,
    )
    return True


def _stage_profile(profile_logs: Path):
    from rpg_economy.campaign import measure_profile
    from rpg_economy.campaign.cost import CostError

    profile = measure_profile(profile_logs)
    if profile.runs_measured == 0:
        raise CostError(
            f"no measured runs under {profile_logs}",
            key="error.cost.no_runs_under",
            path=str(profile_logs),
        )
    return profile


def _smoke(
    config: object,
    out: Path,
    *,
    seed: int,
    budget_fraction: float,
    confirm_cost: str | None,
    spend_limit_set: bool,
    report_only: bool,
    allow_final_seed: bool = False,
    break_lock: bool = False,
) -> int:
    """The two real-provider smoke runs.

    SMOKE-1 is an unconstrained trio: it establishes that the provider's usage
    fields reach the ledger and that bucket attribution reconciles with them on
    every call. SMOKE-2 replays condition A against a budget set from SMOKE-1's
    measured spend, so exhaustion is provoked by real consumption rather than a
    guess -- the point is that the budget binds on the provider's total, and a
    hand-picked number could bind on ours by accident.
    """
    from decimal import ROUND_HALF_UP, Decimal

    from rpg_economy.campaign.runner import set_aside_previous
    from rpg_economy.campaign.smoke import smoke_config, verify_smoke
    from rpg_economy.integrity import TrioIntegrityError
    from rpg_economy.llm import ProviderError
    from rpg_economy.logging_ import LogError
    from rpg_economy.runner import RunInterrupted, run_condition, run_trio

    root = Path(out)
    one, two = root / "smoke1", root / "smoke2"
    planned = smoke_config(config, experiment="experiment_1").time.horizon_ticks

    if not report_only:
        if _refuse_final_seed(config, seed, allowed=allow_final_seed):
            return 5
        if config.llm.pricing is None:
            print(t("error.campaign.invalid", detail=t("error.cost.no_pricing")), file=sys.stderr)
            return 4
        estimate = _smoke_estimate(config)
        print(t("cli.smoke.cost", total=estimate, currency=config.llm.pricing.currency))
        print(
            t(
                "cli.smoke.cost_worst",
                worst=_smoke_worst_case(config),
                currency=config.llm.pricing.currency,
            )
        )
        print(t("cli.smoke.cost_note"))
        if not spend_limit_set:
            print(t("error.derisk.no_spend_limit"), file=sys.stderr)
            return 5
        if confirm_cost != str(estimate):
            print(
                t(
                    "error.campaign.unconfirmed",
                    total=str(estimate),
                    currency=config.llm.pricing.currency,
                    runs=4,
                ),
                file=sys.stderr,
            )
            return 5

        # Where the step being run writes, for the message if its log fails.
        current = one
        interrupt = Interrupt()

        def execute() -> int:
            nonlocal current
            try:
                print(t("cli.smoke.running", label="SMOKE-1"))
                # A rerun keeps the earlier runs, set aside: they were billed,
                # and run logs are write-once.
                set_aside_previous(one)
                run_trio(
                    smoke_config(config, experiment="experiment_1"),
                    one,
                    seed=seed,
                    should_stop=interrupt.should_stop,
                    should_abort=interrupt.should_abort,
                )

                measured = verify_smoke(
                    one / "A.jsonl",
                    label="SMOKE-1",
                    planned_horizon=planned,
                    expect_exhaustion=False,
                )
                budget = max(1, int(measured.provider_total * budget_fraction))
                print(
                    t(
                        "cli.smoke.budget",
                        budget=format_number(budget),
                        measured=format_number(measured.provider_total),
                        fraction=f"{budget_fraction:.0%}",
                    )
                )

                # SMOKE-1 is kept, and verified again by a rerun.
                if interrupt.should_stop():
                    return _stopped(root)
                print(t("cli.smoke.running", label="SMOKE-2"))
                current = two
                two.mkdir(parents=True, exist_ok=True)
                set_aside_previous(two, (Condition.A,))
                run_condition(
                    smoke_config(config, experiment="experiment_2", budget=budget).for_condition(
                        Condition.A, seed=seed
                    ),
                    two / "A.jsonl",
                    should_abort=interrupt.should_abort,
                )
            except RunInterrupted:
                return _stopped(root)
            except ProviderError as exc:
                print(t("error.provider.failed", detail=str(exc)), file=sys.stderr)
                return 8
            except (LogError, OSError, TrioIntegrityError) as exc:
                return _unusable_run(exc, current)
            return 0

        # Held for both runs: a second smoke on the same directory would set
        # this one's logs aside mid-run and pay for its own.
        code = _locked(root, execute, break_lock=break_lock, interrupt=interrupt)
        if code:
            return code

    reports = []
    for path, label, exhaustion in (
        (one / "A.jsonl", "SMOKE-1", False),
        (two / "A.jsonl", "SMOKE-2", True),
    ):
        if not path.exists():
            print(t("error.smoke.no_run", label=label, path=str(path)), file=sys.stderr)
            return 6
        try:
            reports.append(
                verify_smoke(
                    path, label=label, planned_horizon=planned, expect_exhaustion=exhaustion
                )
            )
        except LogError, OSError:
            print(t("error.run.log_unusable", path=str(path)), file=sys.stderr)
            return 6

    ok = True
    for report in reports:
        print()
        print(t("cli.smoke.header", label=report.label, model=report.model))
        print(
            t(
                "cli.smoke.usage",
                calls=format_number(report.llm_calls),
                input=format_number(report.provider_input),
                output=format_number(report.provider_output),
                ledger=format_number(report.ledger_total),
            )
        )
        if config.llm.pricing is not None:
            actual = (
                Decimal(report.provider_input) / 1_000_000 * config.llm.pricing.input_per_1m_tokens
                + Decimal(report.provider_output)
                / 1_000_000
                * config.llm.pricing.output_per_1m_tokens
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            print(t("cli.smoke.actual_cost", total=actual, currency=config.llm.pricing.currency))
        for check in report.checks:
            print(
                t(
                    "cli.smoke.check",
                    mark=t("cli.smoke.pass" if check.passed else "cli.smoke.fail"),
                    name=check.name,
                    detail=check.detail,
                )
            )
        ok = ok and report.passed

    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "smoke.json").write_text(
        json.dumps([r.to_payload() for r in reports], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print(t("cli.smoke.written", path=str(root / "smoke.json")))
    if not ok:
        print(t("error.smoke.failed"), file=sys.stderr)
        return 7
    return 0


def _smoke_estimate(config: object):
    """Typical price of the two smoke runs, the figure confirmed with --confirm-cost.

    Output at the completion length measured on the real provider, not at the
    output cap; ``_smoke_worst_case`` is the cap.
    """
    from decimal import ROUND_HALF_UP, Decimal

    from rpg_economy.campaign.cost import MEASURED_OUTPUT_PER_CALL
    from rpg_economy.campaign.smoke import SMOKE_CALLS, SMOKE_INPUT_TOKENS

    pricing = config.llm.pricing
    # One measured constant, not two literals drifting apart.
    output = SMOKE_CALLS * MEASURED_OUTPUT_PER_CALL
    return (
        Decimal(SMOKE_INPUT_TOKENS) / 1_000_000 * pricing.input_per_1m_tokens
        + Decimal(output) / 1_000_000 * pricing.output_per_1m_tokens
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _smoke_worst_case(config: SimulationConfig) -> Decimal:
    """Price of the two smoke runs at every cap: what a spend limit must cover.

    Each run as ``_worst_case_run`` prices a campaign run: every activation the
    K cap allows, each agent call retried once, every Director call the K + 1
    cap allows, and every reply at ``max_output_tokens``. SMOKE-1 is a trio;
    SMOKE-2 is condition A again, bounded here by A without a budget, since its
    budget is only known once SMOKE-1 has run.
    """
    from decimal import ROUND_HALF_UP

    from rpg_economy.campaign.cost import _worst_case_run
    from rpg_economy.campaign.smoke import SMOKE_CALLS, SMOKE_INPUT_TOKENS, smoke_config

    pricing = config.llm.pricing
    sized = smoke_config(config, experiment="experiment_1")
    input_per_call = SMOKE_INPUT_TOKENS / SMOKE_CALLS
    worst_input = worst_output = 0
    for condition in (Condition.A, Condition.B, Condition.C, Condition.A):
        run_input, run_output = _worst_case_run(sized, condition, input_per_call, None)
        worst_input += run_input
        worst_output += run_output
    return (
        Decimal(worst_input) / 1_000_000 * pricing.input_per_1m_tokens
        + Decimal(worst_output) / 1_000_000 * pricing.output_per_1m_tokens
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
