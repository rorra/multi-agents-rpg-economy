"""Command-line entry point.

Every user-facing string this module emits -- argparse help text included --
resolves through ``i18n.t`` (IMPLEMENTATION.md section 2). Because argparse renders its
help while the parser is being built, the locale is resolved from ``argv``
before the parser exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rpg_economy.domain import ExperimentRegime
from rpg_economy.i18n import SUPPORTED_LOCALES, t

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
