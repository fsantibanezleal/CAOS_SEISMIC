"""Provenance helpers — write the versioned manifests that make every stage reproducible.

A manifest (see :class:`caos_seismic.contracts.Manifest`) is the small JSON record committed to
``manifests/`` for every pipeline stage. Raw data, features and weights are NEVER versioned; the
manifest is the durable, byte-level provenance of *how* they were produced (source URLs, query
params, row counts, config hash, code git SHA, issue timestamp). It is what makes a past forecast
auditable months later (see ``data-and-pipelines.md`` §4, the versioned pipeline DAG).

This module depends only on the standard library plus the package's own ``config``/``contracts`` —
no heavy or optional dependencies — so the manifest record can always be written.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, config_hash
from .contracts import Manifest

MANIFEST_DIR = REPO_ROOT / "manifests"


def utc_now_iso() -> str:
    """Current UTC instant as an ISO-8601 string with a trailing ``Z`` (second resolution)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha(short: bool = True) -> str | None:
    """Return the current code git SHA, or ``None`` if not in a git checkout / git is absent.

    Recorded in every manifest so a forecast can be tied back to the exact code that produced it.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


def build_manifest(
    stage: str,
    region_id: str,
    *,
    config_names: tuple[str, ...] = (),
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> Manifest:
    """Assemble a typed :class:`Manifest` for ``stage`` with provenance stamped in.

    ``config_names`` are the ``configs/<name>.yaml`` files this stage read; their combined hash is
    stored so a config change invalidates downstream reproductions.
    """
    return Manifest(
        stage=stage,  # type: ignore[arg-type]  # Literal validated by pydantic
        created_at=utc_now_iso(),
        region_id=region_id,
        code_git_sha=git_sha(),
        config_hash=config_hash(*config_names) if config_names else None,
        inputs=inputs or {},
        outputs=outputs or {},
        stats=stats or {},
    )


def write_manifest(manifest: Manifest, *, manifest_dir: Path | None = None) -> Path:
    """Write ``manifest`` to ``manifests/<region>_<stage>_<created_at>.json`` and return the path.

    The directory is created if absent. JSON is pretty-printed and UTF-8 (no BOM).
    """
    out_dir = manifest_dir or MANIFEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = manifest.created_at.replace(":", "").replace("-", "")
    path = out_dir / f"{manifest.region_id}_{manifest.stage}_{stamp}.json"
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
