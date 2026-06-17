"""Command-line entry points for CAOS_SEISMIC (the `caos-seismic` console script).

This module is the *thin* command surface the `scripts/*.ps1` and `scripts/*.sh` wrappers call. It owns
no science: every heavy stage is delegated to a stage subpackage (`caos_seismic.data`, `.catalog`,
`.model`, `.inference`, `.eval`), which are **imported lazily inside each command** so that:

  * the package stays importable with ONLY the core deps (numpy, pandas, scipy, requests, pyyaml,
    pydantic, h3, scikit-learn) — heavy science deps (obspy, pycsep, geopandas, pygtide) are pulled in
    only by the stage that needs them, and
  * a stage subpackage that a parallel build has not landed yet produces a single, clear, *actionable*
    error ("run `pip install -e .[science]`" / "stage not yet available") instead of an import-time crash.

Subcommands (kept 1:1 with the scripts):

    fetch            pull the recent + historical catalog (ComCat spine + regional/anchor sources)
    build-features   Mc + b-value, magnitude homogenization, dual-catalog declustering, features
    train            fit the smoothed-seismicity null + space-time ETAS (+ R-J fallback)
    infer            run the daily forecast clock -> compact artifact under results/
    daily            the production job: fetch -> infer -> scoped publish (commit + push)
    backanalysis     pseudo-prospective CSEP back-analysis over a date range
    check            environment + repo + config sanity checks (no network, no science deps required)

Framing (non-negotiable): this is a *forecaster*, never a *predictor*. See `contracts.py`.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import REPO_ROOT, config_hash, load, load_region

app = typer.Typer(
    name="caos-seismic",
    help="Conditional probabilistic seismic forecasting (forecasts, never predictions; CSEP-scored).",
    add_completion=False,
    no_args_is_help=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Small console helpers (rich if present, plain otherwise — core dep, but degrade safely)
# ─────────────────────────────────────────────────────────────────────────────


def _echo(msg: str, *, err: bool = False) -> None:
    typer.echo(msg, err=err)


def _fail(msg: str, *, code: int = 1) -> "typer.Exit":
    """Print an actionable error to stderr and raise typer.Exit(code)."""
    _echo(f"error: {msg}", err=True)
    return typer.Exit(code)


def _require_stage(module: str, *, extra: str | None = None):
    """Import a stage subpackage lazily, with a clear, actionable error if it (or its heavy deps) are absent.

    `module` is dotted relative to this package, e.g. "data.fetch". `extra` names the pip extra that
    supplies the missing heavy dependency (so the message tells the user exactly what to install).
    """
    full = f"{__package__}.{module}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as exc:
        missing = exc.name or full
        if missing == full or missing.startswith(f"{__package__}.{module.split('.')[0]}"):
            raise _fail(
                f"stage '{module}' is not available yet in this build "
                f"({full!r} could not be imported). This command depends on a pipeline stage that has "
                f"not landed in your checkout. Update the repo, then retry."
            ) from exc
        hint = f"pip install -e .[{extra}]" if extra else "pip install -e .[science]"
        raise _fail(
            f"stage '{module}' needs an optional dependency that is not installed "
            f"(missing module {missing!r}). Install the science stack:  {hint}"
        ) from exc


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repo root, capturing text output."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages — thin delegations to the stage subpackages
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def fetch(
    region: str = typer.Option("chile", "--region", "-r", help="Region id (configs/region.<id>.yaml)."),
    days: Optional[int] = typer.Option(
        None, "--days", help="Only pull the last N days (default: the configured fetch window)."
    ),
    focus: Optional[str] = typer.Option(
        None, "--focus", help="Optional sub-region focus key (e.g. 'north' for Chile)."
    ),
) -> None:
    """Pull the catalog (USGS ComCat spine + regional/anchor sources) and write a provenance manifest.

    The ComCat spine works with `requests` + `pandas` alone; regional FDSN / ISC-GEM / GCMT enrichers may
    pull in optional science deps lazily inside the stage.
    """
    reg = load_region(region)
    mod = _require_stage("data.fetch")
    _echo(f"fetch · region={reg.id} ({reg.name_en})")
    result = mod.run_fetch(region=reg, days=days, focus=focus)
    _echo(f"fetch · done: {result}")


@app.command(name="build-features")
def build_features(
    region: str = typer.Option("chile", "--region", "-r", help="Region id."),
) -> None:
    """Mc + b-value estimation, Mw homogenization, dual-catalog declustering, and feature extraction.

    Implements the dual-catalog rule (configs/declustering.yaml): the declustered catalog feeds the
    stationary background; the FULL un-declustered catalog feeds the conditional/ETAS model.
    """
    reg = load_region(region)
    mod = _require_stage("catalog.features")
    _echo(f"build-features · region={reg.id}")
    result = mod.run_build_features(region=reg)
    _echo(f"build-features · done: {result}")


@app.command()
def train(
    region: str = typer.Option("chile", "--region", "-r", help="Region id."),
) -> None:
    """Fit the stationary smoothed-seismicity null and the space-time ETAS model (+ R-J fallback).

    Rejects any ETAS fit that violates either stability gate (alpha < beta; branching ratio n < 1;
    see configs/etas.yaml).
    """
    reg = load_region(region)
    mod = _require_stage("model.train")
    _echo(f"train · region={reg.id}")
    result = mod.run_train(region=reg)
    _echo(f"train · done: {result}")


@app.command()
def infer(
    region: str = typer.Option("chile", "--region", "-r", help="Region id."),
    issue: Optional[str] = typer.Option(
        None, "--issue", help="Issue date (YYYY-MM-DD, UTC). Default: today (UTC)."
    ),
) -> None:
    """Run the forecast clock for the issue date and write a compact artifact under results/.

    The forecast clock hands the model only the catalog slice strictly before the issue time (no
    leakage), then writes `results/forecast-<region>-<date>.json.gz` + updates `results/index.json` and a
    provenance manifest. Matches the ForecastArtifact schema in contracts.py.
    """
    reg = load_region(region)
    issue_dt = _parse_issue(issue)
    mod = _require_stage("inference.daily")
    _echo(f"infer · region={reg.id} · issue={issue_dt.isoformat()}")
    result = mod.run_infer(region=reg, issue=issue_dt)
    _echo(f"infer · done: {result}")


@app.command()
def backanalysis(
    region: str = typer.Option("chile", "--region", "-r", help="Region id."),
    start: str = typer.Option(..., "--start", help="First issue date (YYYY-MM-DD, UTC)."),
    end: str = typer.Option(..., "--end", help="Last issue date (YYYY-MM-DD, UTC)."),
) -> None:
    """Pseudo-prospective CSEP back-analysis over [start, end] (the forecast clock advances day by day).

    For each issue date the model sees only the catalog slice (-inf, t); the forecast is sealed and
    scored against the catalog *as it was at issue time*. Emits per-region/period CSEP test outcomes.
    """
    reg = load_region(region)
    start_dt = _parse_issue(start)
    end_dt = _parse_issue(end)
    if end_dt < start_dt:
        raise _fail("--end is before --start.")
    mod = _require_stage("eval.backanalysis", extra="science")
    _echo(f"backanalysis · region={reg.id} · {start_dt.isoformat()} -> {end_dt.isoformat()}")
    result = mod.run_backanalysis(region=reg, start=start_dt, end=end_dt)
    _echo(f"backanalysis · done: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# daily — the production job (fetch -> infer -> scoped publish). Owned here (ops layer).
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def daily(
    region: str = typer.Option("chile", "--region", "-r", help="Region id."),
    no_publish: bool = typer.Option(
        False, "--no-publish", help="Run fetch + infer but skip the git commit/push (local dry run)."
    ),
    catch_up: bool = typer.Option(
        True, "--catch-up/--no-catch-up", help="Also produce any missed prior days (configs/publish.yaml)."
    ),
) -> None:
    """The production daily job: fetch -> infer -> scoped publish.

    Publishing is **scoped**: only the configs/publish.yaml `git.add_allowlist` paths (results/,
    manifests/) are staged — never `git add -A`/`.`. The commit aborts if anything outside the allowlist
    is staged. The wrapper scripts (scripts/daily.*) perform the actual commit + push; this command runs
    the pipeline and (unless --no-publish) the scoped staging + commit + push itself so the job works
    even when invoked directly (e.g. from a systemd unit).
    """
    reg = load_region(region)
    publish_cfg = load("publish")

    # 1) Determine the set of issue dates to run (today + any missed days if catch-up is on).
    today = datetime.now(timezone.utc).date()
    issue_dates = _missed_issue_dates(reg.id, today) if catch_up else []
    if today not in issue_dates:
        issue_dates.append(today)
    issue_dates = sorted(set(issue_dates))

    fetch_mod = _require_stage("data.fetch")
    infer_mod = _require_stage("inference.daily")

    _echo(f"daily · region={reg.id} · issue_dates={[d.isoformat() for d in issue_dates]}")

    # 2) Fetch once (the freshest catalog covers every issue date in this batch).
    fetch_mod.run_fetch(region=reg, days=None, focus=None)

    # 3) Infer for each issue date (oldest first, so index.json ends current).
    produced: list[str] = []
    for d in issue_dates:
        issue_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        res = infer_mod.run_infer(region=reg, issue=issue_dt)
        produced.append(str(res))

    if no_publish:
        _echo(f"daily · --no-publish set; produced {len(produced)} artifact(s), not committing.")
        return

    # 4) Scoped publish.
    _publish_scoped(publish_cfg, region=reg.id, n_dates=len(issue_dates))
    _echo("daily · published.")


# ─────────────────────────────────────────────────────────────────────────────
# check — environment + repo + config sanity (no network, no science deps)
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def check(
    region: str = typer.Option("chile", "--region", "-r", help="Region id to validate."),
    smoke: bool = typer.Option(
        True,
        "--smoke/--no-smoke",
        help="Also run a tiny end-to-end smoke pipeline (network-tolerant; skips gracefully offline).",
    ),
) -> None:
    """Environment + repo + config sanity checks, then an optional tiny end-to-end smoke run.

    Verifies: Python >= 3.12 (warn otherwise), core imports, configs load + parse, the region loads, the
    publish allowlist is sane, results/ + manifests/ exist, and git is available. With ``--smoke`` (the
    default) it additionally exercises fetch -> clean -> Mc/b -> a trivial ETAS condition -> artifact
    write on a *tiny* Chile fixture (small bbox, last ~120 days, M>=4) using ONLY the core deps. The
    smoke run is **network-tolerant**: if ComCat is unreachable (offline / CI) it is skipped with a
    warning, never failing the check. Exits non-zero only if a hard sanity check fails.
    """
    problems: list[str] = []
    warnings: list[str] = []

    # Python version (target 3.12).
    pyver = sys.version_info
    _echo(f"python   · {platform.python_version()} ({sys.executable})")
    if (pyver.major, pyver.minor) < (3, 12):
        warnings.append(f"Python {platform.python_version()} < 3.12 (target is 3.12).")

    # Core imports.
    for mod in ("numpy", "pandas", "scipy", "requests", "yaml", "pydantic", "h3", "sklearn"):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            problems.append(f"core dependency '{mod}' not importable: {exc}")
    _echo(f"core deps · {'ok' if not problems else 'MISSING (see below)'}")

    # Configs load + parse.
    for name in ("grid", "completeness", "declustering", "etas", "forecast", "publish"):
        try:
            cfg = load(name)
            if not isinstance(cfg, dict) or not cfg:
                problems.append(f"config '{name}.yaml' loaded empty.")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"config '{name}.yaml' failed to load: {exc}")
    _echo("configs   · loaded")

    # Region loads.
    try:
        reg = load_region(region)
        _echo(f"region    · {reg.id} '{reg.name_en}' m_max={reg.m_max}")
        if not reg.attribution:
            warnings.append(f"region '{reg.id}' has no attribution list (required on public surfaces).")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"region '{region}' failed to load: {exc}")

    # Publish allowlist sanity (must be explicit, must NOT be '.' or '-A').
    try:
        pub = load("publish")
        allowlist = list(pub.get("git", {}).get("add_allowlist", []))
        if not allowlist:
            problems.append("publish.yaml git.add_allowlist is empty (scoped publish would have nothing to stage).")
        for entry in allowlist:
            if entry.strip() in {".", "-A", "--all", "*"}:
                problems.append(f"publish.yaml allowlist contains a non-scoped entry {entry!r}.")
        _echo(f"publish   · allowlist={allowlist}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"publish.yaml failed to load: {exc}")

    # Output dirs.
    for sub in ("results", "manifests"):
        p = REPO_ROOT / sub
        if not p.is_dir():
            problems.append(f"output directory '{sub}/' is missing.")
    _echo("dirs      · results/ manifests/")

    # git available.
    try:
        out = _git("rev-parse", "--is-inside-work-tree", check=True)
        _echo(f"git       · {out.stdout.strip()} (branch: {_git('branch', '--show-current', check=False).stdout.strip() or '?'})")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"git not available / not a work tree: {exc}")

    # Provenance hash (proves config.py + configs are coherent).
    try:
        h = config_hash("region.chile" if region == "chile" else f"region.{region}", "etas", "forecast")
    except Exception:  # region config hashing is best-effort
        h = config_hash("etas", "forecast")
    _echo(f"cfg hash  · {h}")

    # Optional tiny end-to-end smoke run (network-tolerant; core deps only).
    if smoke and not problems:
        try:
            ok, msg = _smoke_pipeline(region)
            _echo(f"smoke     · {msg}")
            if not ok:
                warnings.append(f"smoke run reported a soft issue: {msg}")
        except _SmokeSkip as skip:
            _echo(f"smoke     · skipped ({skip})")
        except Exception as exc:  # noqa: BLE001 - a smoke failure is a warning, not a hard fail
            warnings.append(f"smoke run raised {type(exc).__name__}: {exc}")

    for w in warnings:
        _echo(f"warn · {w}", err=True)
    if problems:
        for p in problems:
            _echo(f"FAIL · {p}", err=True)
        raise typer.Exit(1)
    _echo("check · OK")


class _SmokeSkip(RuntimeError):
    """Raised inside the smoke run when the network is unavailable (skip, don't fail)."""


def _smoke_pipeline(region_id: str) -> tuple[bool, str]:
    """Tiny fetch -> clean -> Mc/b -> trivial ETAS condition -> artifact write, on core deps only.

    Pulls a *small* recent slice of the Chile catalog (a tight bbox, the last ~120 days, M>=4) over
    the ComCat spine (requests + pandas, no science deps), homogenizes + estimates Mc/b, builds the
    fit cells, conditions the model family (ETAS if it fits, else the R-J fallback + smoothed null),
    assembles a :class:`ForecastArtifact`, and writes it to a throwaway temp directory (never the
    committed ``results/``). Returns ``(ok, message)``. Raises :class:`_SmokeSkip` when ComCat is
    unreachable so an offline machine / CI passes the check.
    """
    import tempfile

    reg = load_region(region_id)

    # 1) Fetch a tiny slice over the spine (network-tolerant).
    fetch_mod = importlib.import_module(f"{__package__}.data.fetch")
    contracts = importlib.import_module(f"{__package__}.contracts")
    bb = reg.bbox
    # A small bbox around the most active northern Chile margin (or the region centre).
    small_bbox = contracts.BBox(
        lat_min=max(bb.lat_min, -24.0),
        lat_max=min(bb.lat_max, -20.0),
        lon_min=max(bb.lon_min, -71.5),
        lon_max=min(bb.lon_max, -69.5),
    )
    now = datetime.now(timezone.utc)
    start = now - __import__("datetime").timedelta(days=120)
    try:
        raw = fetch_mod.fetch_comcat(
            starttime=start, endtime=now, bbox=small_bbox, minmagnitude=4.0, source="usgs_comcat"
        )
    except Exception as exc:  # network blip / offline / over-large → skip, don't fail the check
        raise _SmokeSkip(f"ComCat unreachable: {type(exc).__name__}") from exc

    if raw.empty or len(raw) < 1:
        return True, "fetch ok but window empty (no M>=4 events in the last 120d here) — pipeline not exercised"

    # 2) Clean / homogenize to Mw (core deps only).
    clean_mod = importlib.import_module(f"{__package__}.data.clean")
    clean = clean_mod.clean_catalog(raw).catalog
    mw = clean["mw"].dropna()
    if mw.empty:
        return True, f"fetched {len(raw)} events but none had a usable Mw (no conversion anchor) — Mc/ETAS skipped"

    # 3) Mc + b (estimated).
    completeness = importlib.import_module(f"{__package__}.catalog.completeness")
    mc_est = completeness.mc_estimate(mw.to_numpy(), min_events=10, regional_default=4.0)
    mc = float(mc_est.mc)

    # 4) Trivial condition: build cells, fit the model family at "now", assemble an artifact.
    daily = importlib.import_module(f"{__package__}.inference.daily")
    res = daily.run_infer(region=reg, issue=now, catalog=clean, publish=False)
    if res.artifact is None:
        return False, f"conditioned on {len(clean)} events (Mc={mc:.2f}) but no artifact assembled"

    # 5) Write the artifact to a throwaway dir (never the committed results/).
    artifact_mod = importlib.import_module(f"{__package__}.inference.artifact")
    with tempfile.TemporaryDirectory() as tmp:
        paths = artifact_mod.write_artifact(res.artifact, results_dir=Path(tmp))
        reloaded = artifact_mod.load_artifact(paths["artifact"])
    n_cells = len(reloaded.forecast)
    return True, (
        f"fetch {len(raw)} -> clean {len(clean)} (Mc={mc:.2f}) -> conditioned -> "
        f"artifact {n_cells} H3 cell(s), QA={'pass' if res.qa_passed else 'block'}"
    )


@app.command()
def version() -> None:
    """Print the package version (used by the scripts' `--version` smoke test)."""
    _echo(f"caos-seismic {__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_issue(value: str | None) -> datetime:
    """Parse a YYYY-MM-DD issue date (UTC midnight). Default: today UTC."""
    if value is None:
        d = datetime.now(timezone.utc).date()
    else:
        try:
            d = date.fromisoformat(value)
        except ValueError as exc:
            raise _fail(f"invalid date {value!r}; expected YYYY-MM-DD.") from exc
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _existing_issue_dates(region_id: str) -> set[date]:
    """Issue dates already present as committed artifacts under results/."""
    out: set[date] = set()
    results_dir = REPO_ROOT / "results"
    if not results_dir.is_dir():
        return out
    # forecast-<region>-YYYY-MM-DD.json[.gz]
    for p in results_dir.glob(f"forecast-{region_id}-*.json*"):
        stem = p.name
        for ext in (".json.gz", ".json"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        token = stem.split("-", 2)[-1] if stem.count("-") >= 2 else ""
        try:
            out.add(date.fromisoformat(token))
        except ValueError:
            continue
    return out


def _missed_issue_dates(region_id: str, today: date, max_back: int = 7) -> list[date]:
    """Catch-up: issue dates in the last `max_back` days that have no committed artifact yet.

    Bounded to a week so a long-dormant laptop does not attempt to backfill months at once (those would
    not be honest pseudo-prospective forecasts anyway — the catalog has since been revised).
    """
    have = _existing_issue_dates(region_id)
    missed: list[date] = []
    for back in range(max_back, -1, -1):
        d = date.fromordinal(today.toordinal() - back)
        if d not in have:
            missed.append(d)
    return missed


def _publish_scoped(publish_cfg: dict, *, region: str, n_dates: int) -> None:
    """Stage ONLY the allowlist paths, abort if anything else is staged, commit with the configured prefix, push.

    This is the same scoped-publish discipline the wrapper scripts enforce; implemented here so the
    `daily` command is self-sufficient when invoked directly (systemd / Task Scheduler).
    """
    git_cfg = publish_cfg.get("git", {})
    allowlist = [str(p) for p in git_cfg.get("add_allowlist", [])]
    if not allowlist:
        raise _fail("publish.yaml git.add_allowlist is empty; refusing to publish.")
    for entry in allowlist:
        if entry.strip() in {".", "-A", "--all", "*"}:
            raise _fail(f"publish.yaml allowlist has a non-scoped entry {entry!r}; refusing to publish.")

    prefix = str(git_cfg.get("commit_message_prefix", "data: daily forecast"))
    remote = str(git_cfg.get("remote", "origin"))
    branch = str(git_cfg.get("publish_branch", "main"))

    # Reset the index so a pre-existing staged change cannot ride along.
    _git("reset", "-q", check=False)
    # Stage only the allowlist.
    for entry in allowlist:
        _git("add", "--", entry, check=True)

    # Verify nothing outside the allowlist got staged.
    staged = _git("diff", "--cached", "--name-only", check=True).stdout.splitlines()
    staged = [s for s in staged if s.strip()]
    if not staged:
        _echo("publish · nothing to commit (no new artifacts).")
        return
    allowed_prefixes = tuple(e.rstrip("/") for e in allowlist)
    offenders = [s for s in staged if not any(s == p or s.startswith(p + "/") for p in allowed_prefixes)]
    if offenders:
        _git("reset", "-q", check=False)
        raise _fail(
            "scoped-publish guard tripped: paths outside the allowlist were staged "
            f"({offenders}). Index reset; nothing committed."
        )

    today_iso = date.today().isoformat()
    suffix = f"{region} {today_iso}" + (f" (+{n_dates - 1} catch-up)" if n_dates > 1 else "")
    message = f"{prefix}: {suffix}"
    _git("commit", "-q", "-m", message, check=True)
    _echo(f"publish · committed: {message}")

    # Push (best effort; a missing remote is a clear, non-fatal message for local dev).
    push = _git("push", remote, f"HEAD:{branch}", check=False)
    if push.returncode != 0:
        _echo(
            f"publish · commit made locally but push failed (remote '{remote}', branch '{branch}'):\n"
            f"{push.stderr.strip()}",
            err=True,
        )
        raise typer.Exit(1)
    _echo(f"publish · pushed to {remote} {branch}.")


if __name__ == "__main__":  # pragma: no cover
    app()
