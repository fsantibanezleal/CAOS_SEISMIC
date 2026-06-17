"""Provenance — make every daily forecast byte-reproducible months later.

Without an immutable record of *exactly* what produced a forecast, honest pseudo-prospective and
true-prospective CSEP scoring is impossible: you would end up scoring today's model against a
retroactively revised ComCat catalog, which is optimistic leakage (web-app-spec.md §8.2,
"Reproducibility / data-versioning"; model-design.md §9 step 1).

This module reads/writes :class:`caos_seismic.contracts.Manifest` objects and assembles the
``provenance`` block embedded in each :class:`ForecastArtifact`. A manifest pins:

* **config hash** — :func:`caos_seismic.config.config_hash` over the configs that govern the run;
* **code git SHA** — the exact commit, so the code is recoverable;
* **input catalog snapshot id** — the immutable catalog state used (ComCat revisions/retractions are
  snapshotted, never silently overwritten);
* **Mc grid version** — the magnitude-of-completeness map is a first-class versioned artifact;
* **declustering choice**, **model + params**, and the **issue timestamp**.

Manifests live under ``manifests/`` as one JSON per stage; nothing here imports heavy deps.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, config_hash
from ..contracts import Manifest

MANIFEST_DIR = REPO_ROOT / "manifests"

#: The configs whose contents define a daily run; hashed together for the provenance manifest.
GOVERNING_CONFIGS: tuple[str, ...] = (
    "grid",
    "completeness",
    "declustering",
    "etas",
    "forecast",
    "publish",
)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_git_sha(short: bool = False) -> str | None:
    """Return the current repo commit SHA, or ``None`` if git is unavailable / not a repo.

    Resolved by shelling out to ``git rev-parse`` in :data:`REPO_ROOT`. Never raises — provenance
    must degrade gracefully (a missing SHA is recorded as ``None``, not a crash) so a forecast can
    still be produced from a source checkout that is not a git working tree.
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    sha = out.stdout.strip()
    return sha or None


def snapshot_id(catalog, region_id: str, t_issue: str) -> str:
    """Deterministic id for an immutable input-catalog snapshot.

    The id is a content hash of the conditioning catalog (event ids + times + magnitudes) plus the
    region and issue time. Two runs over the identical catalog state get the identical id; a single
    revised/retracted event changes it — which is exactly the audit signal we want (model-design.md
    §9 step 1: "Handle ComCat revisions/retractions by snapshotting, never by silently overwriting").

    Accepts a DataFrame (hashes the contract columns) or anything JSON-serializable.
    """
    try:
        import pandas as pd  # local import keeps this usable without a DataFrame

        if isinstance(catalog, pd.DataFrame):
            cols = [c for c in ("event_id", "time", "mw", "mag", "mag_type") if c in catalog.columns]
            sub = catalog[cols].copy()
            if "time" in sub:
                sub["time"] = pd.to_datetime(sub["time"], utc=True).astype("int64")
            payload = sub.sort_values(cols[0] if cols else sub.columns[0]).to_csv(index=False)
        else:
            payload = json.dumps(catalog, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive; provenance must not crash a run
        payload = json.dumps(str(catalog))
    blob = f"{region_id}|{t_issue}|{payload}".encode("utf-8")
    return "cat-" + hashlib.sha256(blob).hexdigest()[:16]


def build_manifest(
    *,
    stage: str,
    region_id: str,
    t_issue: str,
    input_snapshot_id: str,
    mc_grid_version: str,
    declustering: str,
    model_name: str,
    model_version: str,
    model_params: dict[str, Any] | None = None,
    configs: tuple[str, ...] = GOVERNING_CONFIGS,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> Manifest:
    """Assemble a :class:`Manifest` for a daily run, hashing the governing configs + code SHA.

    Parameters mirror the provenance requirement in model-design.md §9 / web-app-spec.md §8.2: the
    config hash, code SHA, input snapshot, Mc grid version, declustering choice, model + params, and
    issue timestamp are all pinned. ``inputs``/``outputs``/``stats`` carry stage-specific extras.
    """
    base_inputs: dict[str, Any] = {
        "catalog_snapshot_id": input_snapshot_id,
        "mc_grid_version": mc_grid_version,
        "declustering": declustering,
        "t_issue": t_issue,
    }
    base_inputs.update(inputs or {})
    base_outputs: dict[str, Any] = {
        "model": {
            "name": model_name,
            "version": model_version,
            "params": model_params or {},
        },
    }
    base_outputs.update(outputs or {})
    return Manifest(
        stage=stage,  # type: ignore[arg-type]  # validated by the Literal in the model
        created_at=_utc_now_iso(),
        region_id=region_id,
        code_git_sha=code_git_sha(),
        config_hash=config_hash(*configs),
        inputs=base_inputs,
        outputs=base_outputs,
        stats=stats or {},
    )


def provenance_block(manifest: Manifest) -> dict[str, Any]:
    """Project a :class:`Manifest` into the compact ``provenance`` dict embedded in the artifact.

    The artifact carries a *slim* provenance view (the full manifest lives in ``manifests/``). This
    is what the SPA shows on the "data lineage" panel.
    """
    model = manifest.outputs.get("model", {})
    return {
        "code_git_sha": manifest.code_git_sha,
        "config_hash": manifest.config_hash,
        "catalog_snapshot_id": manifest.inputs.get("catalog_snapshot_id"),
        "mc_grid_version": manifest.inputs.get("mc_grid_version"),
        "declustering": manifest.inputs.get("declustering"),
        "model": {"name": model.get("name"), "version": model.get("version")},
        "issued_at": manifest.inputs.get("t_issue"),
        "created_at": manifest.created_at,
    }


def manifest_path(region_id: str, t_issue: str, stage: str, base_dir: Path | None = None) -> Path:
    """Path for a manifest JSON: ``manifests/<region>/<YYYY-MM-DD>/<stage>.json``."""
    base = base_dir or MANIFEST_DIR
    day = t_issue[:10]
    return base / region_id / day / f"{stage}.json"


def write_manifest(manifest: Manifest, base_dir: Path | None = None) -> Path:
    """Serialize a :class:`Manifest` to ``manifests/<region>/<day>/<stage>.json`` and return its path.

    Pretty-printed (sorted keys) so a git diff of two daily manifests is human-readable.
    """
    t_issue = str(manifest.inputs.get("t_issue", manifest.created_at))
    path = manifest_path(manifest.region_id, t_issue, manifest.stage, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_manifest(path: Path) -> Manifest:
    """Load a :class:`Manifest` from a JSON file written by :func:`write_manifest`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Manifest.model_validate(data)
