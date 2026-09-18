"""The git-as-data publish path: the dedicated job checkout is the single writer to the publish branch.

These tests drive the real git plumbing of ``caos_seismic.cli`` against throwaway local repositories (a
bare ``origin``, a developer clone on ``develop``, and the job checkout: a detached worktree carrying the
marker file). No network, no science deps. They pin the properties the daily job relies on:

* in the job checkout one run publishes ONE commit, as a fast-forward of the publish branch, and no
  developer branch receives a data commit;
* a push that fails leaves the commit in the job checkout and the next ``job-sync`` publishes it (the
  failure that lost the 2026-09-08 forecast cannot recur);
* the publish branch moving during a run (a PR merge) is absorbed by a rebase of the data commits;
* ``job-sync`` fast-forwards, drops the leftovers of an interrupted run (keeping ignored files), refuses
  local code changes and non-data commits, and drops a backlog that conflicts upstream;
* outside the job checkout the deprecated legacy path still publishes (the scheduled tasks use it until
  they are re-registered) but no longer pushes the developer branch when the fetch fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from caos_seismic import cli

PREFIX = "data: daily forecast"
CFG = {
    "git": {
        "publish_branch": "main",
        "remote": "origin",
        "add_allowlist": ["results/", "manifests/"],
        "commit_message_prefix": PREFIX,
    }
}


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True)


def remote_tip(r: SimpleNamespace, branch: str = "main") -> str:
    out = git(r.seed, "ls-remote", str(r.origin), f"refs/heads/{branch}").stdout.split()
    return out[0] if out else ""


def head(cwd: Path) -> str:
    return git(cwd, "rev-parse", "HEAD").stdout.strip()


def changed_files(cwd: Path, sha: str) -> list[str]:
    return [f for f in git(cwd, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).stdout.splitlines() if f]


def parents(cwd: Path, sha: str) -> list[str]:
    return git(cwd, "rev-list", "--parents", "-n", "1", sha).stdout.split()[1:]


def write_run_outputs(root: Path, day: str, index_text: str | None = None) -> None:
    """What one daily run writes: a forecast artifact, the index, and an inference manifest."""
    (root / "results" / f"forecast-global-{day}.json.gz").write_bytes(f"forecast {day}".encode())
    (root / "results" / "index.json").write_text(index_text or f'{{"latest": "{day}"}}\n', encoding="utf-8")
    manifest = root / "manifests" / "global" / day / "inference.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(f'{{"issue": "{day}"}}\n', encoding="utf-8")


def reject_pushes(origin: Path, reject: bool) -> None:
    hook = origin / "hooks" / "pre-receive"
    if reject:
        hook.write_text("#!/bin/sh\necho 'push rejected by test hook' >&2\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
    elif hook.exists():
        hook.unlink()


def upstream_commit(r: SimpleNamespace, rel: str, text: str, message: str) -> str:
    """A commit pushed to origin/main by someone else (a PR merge, another writer)."""
    git(r.seed, "pull", "-q", "--ff-only", "origin", "main")
    path = r.seed / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    git(r.seed, "add", "--", rel)
    git(r.seed, "commit", "-q", "-m", message)
    git(r.seed, "push", "-q", "origin", "HEAD:refs/heads/main")
    return head(r.seed)


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    empty_global = tmp_path / "empty.gitconfig"
    empty_global.write_text("", encoding="utf-8")
    for key, value in {
        "GIT_CONFIG_GLOBAL": str(empty_global),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "job",
        "GIT_AUTHOR_EMAIL": "job@example.invalid",
        "GIT_COMMITTER_NAME": "job",
        "GIT_COMMITTER_EMAIL": "job@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(cli.ENV_PUBLISH_BRANCH, raising=False)
    monkeypatch.setattr(cli, "_GIT_ATTEMPTS", 2)
    monkeypatch.setattr(cli, "_GIT_RETRY_DELAY_S", 0.0)

    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    git(seed, "remote", "add", "origin", str(origin))
    (seed / "src").mkdir()
    (seed / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (seed / "results").mkdir()
    (seed / "results" / "index.json").write_text('{"latest": null}\n', encoding="utf-8")
    (seed / "manifests").mkdir()
    (seed / "manifests" / "global_fetch_manifest.json").write_text("{}\n", encoding="utf-8")
    (seed / ".gitignore").write_text(f"checkpoints/\nlogs/\n{cli.JOB_MARKER}\n", encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "init")
    git(seed, "push", "-q", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "-q", "origin", "HEAD:refs/heads/develop")

    dev = tmp_path / "dev"
    git(tmp_path, "clone", "-q", "-b", "develop", str(origin), str(dev))
    job = tmp_path / "dev.job"
    git(dev, "worktree", "add", "-q", "--detach", str(job), "origin/main")
    (job / cli.JOB_MARKER).write_text("dedicated job checkout\n", encoding="utf-8")

    monkeypatch.setattr(cli, "REPO_ROOT", job)
    return SimpleNamespace(origin=origin, seed=seed, dev=dev, job=job)


# ─────────────────────────────────────────────────────────────────────────────
# Job checkout detection
# ─────────────────────────────────────────────────────────────────────────────


def test_job_checkout_needs_marker_and_detached_head(repos, monkeypatch):
    assert cli._is_job_checkout()

    monkeypatch.setattr(cli, "REPO_ROOT", repos.dev)  # developer checkout: on a branch, no marker
    assert not cli._is_job_checkout()
    with pytest.raises(typer.Exit):
        cli._require_job_checkout("job-sync")

    monkeypatch.setattr(cli, "REPO_ROOT", repos.job)
    git(repos.job, "checkout", "-q", "-b", "oops")  # a branch checked out in the job checkout
    assert not cli._is_job_checkout()
    with pytest.raises(typer.Exit):
        cli._require_job_checkout("job-sync")
    with pytest.raises(typer.Exit):  # the publish step refuses too, instead of taking the legacy path
        cli._publish_scoped(CFG, region="global", n_dates=1)


# ─────────────────────────────────────────────────────────────────────────────
# The production path: one fast-forward commit per run
# ─────────────────────────────────────────────────────────────────────────────


def test_publish_is_one_fast_forward_commit_and_no_developer_branch_moves(repos):
    base = remote_tip(repos)
    develop_before = remote_tip(repos, "develop")
    write_run_outputs(repos.job, "2026-09-19")
    (repos.job / "notes.txt").write_text("not an artifact\n", encoding="utf-8")  # outside the allowlist

    cli._publish_scoped(CFG, region="global", n_dates=1)

    tip = remote_tip(repos)
    assert tip == head(repos.job)
    assert parents(repos.job, tip) == [base]  # exactly one new commit, a fast-forward
    assert git(repos.job, "log", "-1", "--format=%s", tip).stdout.startswith(f"{PREFIX}: global ")
    assert sorted(changed_files(repos.job, tip)) == [
        "manifests/global/2026-09-19/inference.json",
        "results/forecast-global-2026-09-19.json.gz",
        "results/index.json",
    ]
    assert remote_tip(repos, "develop") == develop_before
    assert cli._head_branch() is None  # still detached


def test_nothing_new_publishes_nothing(repos):
    base = remote_tip(repos)
    cli._publish_scoped(CFG, region="global", n_dates=1)
    assert remote_tip(repos) == base == head(repos.job)


def test_failed_push_is_published_by_the_next_job_sync(repos):
    base = remote_tip(repos)
    reject_pushes(repos.origin, True)
    write_run_outputs(repos.job, "2026-09-08")

    with pytest.raises(typer.Exit):
        cli._publish_scoped(CFG, region="global", n_dates=1)
    data_commit = head(repos.job)
    assert remote_tip(repos) == base  # nothing reached the remote
    assert parents(repos.job, data_commit) == [base]  # the commit is kept in the job checkout

    reject_pushes(repos.origin, False)
    cli._job_sync(CFG)

    assert remote_tip(repos) == data_commit
    assert "results/forecast-global-2026-09-08.json.gz" in changed_files(repos.job, data_commit)


def test_branch_moving_during_the_run_is_absorbed_by_a_rebase(repos):
    write_run_outputs(repos.job, "2026-09-19")
    merged = upstream_commit(repos, "src/app.py", "x = 2\n", "Merge pull request #99")

    cli._publish_scoped(CFG, region="global", n_dates=1)

    tip = remote_tip(repos)
    assert tip == head(repos.job)
    assert parents(repos.job, tip) == [merged]  # linear: the data commit sits on the merged code
    assert (repos.job / "src" / "app.py").read_text(encoding="utf-8") == "x = 2\n"


# ─────────────────────────────────────────────────────────────────────────────
# job-sync
# ─────────────────────────────────────────────────────────────────────────────


def test_job_sync_fast_forwards_and_drops_leftovers_but_keeps_ignored_files(repos):
    new_main = upstream_commit(repos, "src/app.py", "x = 3\n", "Merge pull request #100")
    # Leftovers of an interrupted run under the allowlist, plus an ignored checkpoint that must survive.
    (repos.job / "results" / "index.json").write_text('{"latest": "half-written"}\n', encoding="utf-8")
    (repos.job / "results" / "forecast-global-2026-09-19.json.gz").write_bytes(b"partial")
    ckpt = repos.job / "results" / "checkpoints" / "outlook.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")

    cli._job_sync(CFG)

    assert head(repos.job) == new_main
    assert cli._head_branch() is None
    assert (repos.job / "results" / "index.json").read_text(encoding="utf-8") == '{"latest": null}\n'
    assert not (repos.job / "results" / "forecast-global-2026-09-19.json.gz").exists()
    assert ckpt.read_bytes() == b"weights"
    assert (repos.job / "src" / "app.py").read_text(encoding="utf-8") == "x = 3\n"


def test_job_sync_refuses_local_code_changes(repos):
    (repos.job / "src" / "app.py").write_text("x = 'hot patch'\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli._job_sync(CFG)


def test_non_data_commits_are_never_pushed(repos):
    base = remote_tip(repos)
    (repos.job / "src" / "app.py").write_text("x = 'unreviewed'\n", encoding="utf-8")
    git(repos.job, "commit", "-q", "-am", f"{PREFIX}: disguised code change")

    with pytest.raises(typer.Exit):
        cli._job_sync(CFG)
    with pytest.raises(typer.Exit):
        cli._publish_scoped(CFG, region="global", n_dates=1)
    assert remote_tip(repos) == base


def test_job_sync_refuses_outside_the_job_checkout(repos, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", repos.dev)
    with pytest.raises(typer.Exit):
        cli._job_sync(CFG)


def test_job_sync_drops_a_backlog_that_conflicts_upstream(repos):
    reject_pushes(repos.origin, True)
    write_run_outputs(repos.job, "2026-09-19", index_text='{"latest": "ours"}\n')
    with pytest.raises(typer.Exit):
        cli._publish_scoped(CFG, region="global", n_dates=1)
    reject_pushes(repos.origin, False)
    theirs = upstream_commit(repos, "results/index.json", '{"latest": "theirs"}\n', f"{PREFIX}: global other")

    cli._job_sync(CFG)

    assert head(repos.job) == theirs == remote_tip(repos)
    assert (repos.job / "results" / "index.json").read_text(encoding="utf-8") == '{"latest": "theirs"}\n'
    assert git(repos.job, "status", "--porcelain").stdout.strip() == ""


def test_publish_branch_override_targets_a_scratch_branch(repos, monkeypatch):
    main_before = remote_tip(repos)
    git(repos.seed, "push", "-q", "origin", f"{main_before}:refs/heads/scratch/e2e")
    monkeypatch.setenv(cli.ENV_PUBLISH_BRANCH, "scratch/e2e")

    cli._job_sync(CFG)
    write_run_outputs(repos.job, "2026-09-19")
    cli._publish_scoped(CFG, region="global", n_dates=1)

    assert remote_tip(repos, "scratch/e2e") == head(repos.job)
    assert remote_tip(repos) == main_before


# ─────────────────────────────────────────────────────────────────────────────
# The deprecated legacy path (developer checkout), kept until the tasks are re-registered
# ─────────────────────────────────────────────────────────────────────────────


def test_legacy_path_still_publishes_and_shows_why_it_is_deprecated(repos, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", repos.dev)
    base = remote_tip(repos)
    develop_head_before = head(repos.dev)
    write_run_outputs(repos.dev, "2026-09-19")

    cli._publish_scoped(CFG, region="global", n_dates=1)

    local_copy = head(repos.dev)
    published = remote_tip(repos)
    # Two commit objects for one run: one on the developer branch, one on main. This is the divergence.
    assert parents(repos.dev, local_copy) == [develop_head_before]
    assert parents(repos.dev, published) == [base]
    assert local_copy != published
    assert git(repos.dev, "rev-parse", f"{published}^{{tree}}").stdout == git(
        repos.dev, "rev-parse", f"{local_copy}^{{tree}}"
    ).stdout


def test_legacy_path_never_pushes_the_developer_branch_when_the_fetch_fails(repos, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", repos.dev)
    base = remote_tip(repos)
    # Fetch fails, push would work: the old fallback pushed `HEAD:main` from the developer branch here.
    git(repos.dev, "remote", "set-url", "origin", str(repos.dev.parent / "missing.git"))
    git(repos.dev, "remote", "set-url", "--push", "origin", str(repos.origin))
    write_run_outputs(repos.dev, "2026-09-19")

    with pytest.raises(typer.Exit):
        cli._publish_scoped(CFG, region="global", n_dates=1)
    assert remote_tip(repos) == base
