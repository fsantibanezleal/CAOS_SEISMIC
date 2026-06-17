"""Stage A — catalog fetch.

The **spine** is USGS ComCat over the raw FDSN ``event`` web service, accessed with ``requests``
and ``pandas`` *only* — ObsPy is **not** required for the spine (the public daily job must run on
the core deps alone). The implementation follows ``docs/data-and-pipelines.md`` §1–§2:

* ``GET /count`` first (cheap) to decide whether to tile (USGS, 2024, *FDSN event web service*).
* The service returns **HTTP 400** when a single request would exceed the **20,000-event cap**;
  we tile the time window (bisection) until each tile is under the cap and stitch the results.
* ``updatedafter`` produces daily incremental deltas — only events whose origin/magnitude was
  updated since the last successful run (ComCat continuously revises and retracts events).
* A polite ``User-Agent`` (a contact string) is sent on every request, read from the environment.
* Retry with exponential backoff on transient/over-large responses: 204 (no data → empty),
  400/413 (too large → tile smaller), 429/503 (slow down).

GeoJSON is parsed into a DataFrame matching :data:`caos_seismic.contracts.CATALOG_COLUMNS`,
**keeping** ``mag_type`` (``magType``) — mixing mb/Ms/Mw silently distorts the Gutenberg–Richter
tail, and the Mw homogenization in :mod:`caos_seismic.data.clean` depends on it.

Optional helpers (``fetch_fdsn_obspy``, ``download_isc_gem``, ``download_gcmt_ndk``) cover the
regional networks (CSN via EarthScope/IRIS, ISC, EMSC) and the long-term anchors (ISC-GEM, GCMT).
They lazily import ObsPy and raise an actionable error if it is missing — they are *not* on the
daily critical path.

References
----------
* International FDSN, *FDSN event web service specification* (fdsnws-event 1.0).
* USGS Earthquake Hazards Program, *ANSS Comprehensive Catalog (ComCat) Documentation*.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from ..config import REPO_ROOT, config_hash, load_region
from ..contracts import CATALOG_COLUMNS, BBox, Manifest, Region, validate_catalog

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Default FDSN ``event`` base URL for the USGS ComCat spine (overridable via env).
DEFAULT_COMCAT_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/"

#: Hard per-request cap of the FDSN ``event`` service. Exceeding it returns HTTP 400; we tile.
FDSN_EVENT_CAP = 20_000

#: Stay comfortably below the hard cap so catalog revisions between ``/count`` and ``/query``
#: (ComCat updates continuously) cannot tip a tile over the limit mid-pull.
DEFAULT_TILE_TARGET = 15_000

#: Whole-Earth bbox — the GLOBAL field the model trains on (any country is a VIEW into this).
GLOBAL_BBOX = BBox(lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0)

#: Default worldwide completeness floor for the global pull. The historical global catalog only gets
#: homogeneous and complete around M≳4.5 (below it ComCat's worldwide Mc drifts hard with network era
#: and region); M>=4.5 keeps a multi-decade global volume tractable. Override with ``--min-magnitude``.
DEFAULT_GLOBAL_MIN_MAGNITUDE = 4.5

#: Latitude bands for the first cut of the worldwide tiling. Splitting by latitude before time keeps
#: each band's seismicity density (the circum-Pacific belt dominates) from forcing a single global
#: time window into pathologically deep bisection. Time tiling then refines each band under the cap.
DEFAULT_LAT_BANDS: tuple[tuple[float, float], ...] = (
    (-90.0, -60.0),
    (-60.0, -30.0),
    (-30.0, 0.0),
    (0.0, 30.0),
    (30.0, 60.0),
    (60.0, 90.0),
)

#: HTTP statuses we retry (transient) vs. treat as "tile smaller".
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_TOO_LARGE_STATUSES = frozenset({400, 413})

#: Env var names (see ``.env.example``). The spine needs no credentials.
ENV_USER_AGENT = "CAOS_SEISMIC_USER_AGENT"
ENV_COMCAT_BASE = "COMCAT_FDSN_BASE"

_FALLBACK_USER_AGENT = "CAOS_SEISMIC/0.1 (+https://github.com/fsantibanezleal/CAOS_SEISMIC)"


class ComCatError(RuntimeError):
    """Raised when ComCat cannot satisfy a request after retries/tiling."""


# ─────────────────────────────────────────────────────────────────────────────
# Low-level HTTP with polite retry/backoff
# ─────────────────────────────────────────────────────────────────────────────


def _user_agent() -> str:
    """Polite contact ``User-Agent`` from the env (FDSN services request one). Never a secret."""
    return os.environ.get(ENV_USER_AGENT, "").strip() or _FALLBACK_USER_AGENT


def _comcat_base() -> str:
    base = os.environ.get(ENV_COMCAT_BASE, "").strip() or DEFAULT_COMCAT_BASE
    return base if base.endswith("/") else base + "/"


def _iso(t: str | datetime | pd.Timestamp) -> str:
    """Normalize a time to the ISO-8601 form the FDSN spec accepts (UTC, no offset suffix)."""
    if isinstance(t, str):
        return t
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class _Backoff:
    """Exponential backoff with a cap. Deterministic (no jitter) so tests are reproducible."""

    base_s: float = 1.0
    factor: float = 2.0
    cap_s: float = 60.0

    def delay(self, attempt: int) -> float:
        return min(self.cap_s, self.base_s * (self.factor ** max(0, attempt)))


def _request(
    endpoint: str,
    params: dict[str, Any],
    *,
    session: requests.Session | None = None,
    max_retries: int = 5,
    timeout_s: float = 120.0,
    backoff: _Backoff | None = None,
    _sleep=time.sleep,
) -> requests.Response | None:
    """Issue one FDSN GET with retry/backoff.

    Returns the :class:`requests.Response` on 200, ``None`` on 204 (no data). Raises
    :class:`ComCatError` for over-large requests (caller must tile) and on exhausted retries.
    """
    backoff = backoff or _Backoff()
    sess = session or requests
    url = _comcat_base() + endpoint
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = sess.get(url, params=params, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:  # network blip → retry
            last_exc = exc
            logger.warning("FDSN request error (%s); retry %d/%d", exc, attempt + 1, max_retries)
            if attempt < max_retries:
                _sleep(backoff.delay(attempt))
                continue
            raise ComCatError(f"network error contacting {url}: {exc}") from exc

        status = resp.status_code
        if status == 200:
            return resp
        if status == 204:  # FDSN: valid request, no events in window
            return None
        if status in _TOO_LARGE_STATUSES:
            # 400/413 here means the window exceeds the 20k cap → caller bisects the window.
            raise ComCatError(f"FDSN {status} (over-large request — tile smaller): {resp.url}")
        if status in _RETRY_STATUSES:
            logger.warning("FDSN %d; backing off (attempt %d/%d)", status, attempt + 1, max_retries)
            if attempt < max_retries:
                _sleep(backoff.delay(attempt))
                continue
            raise ComCatError(f"FDSN {status} after {max_retries} retries: {resp.url}")
        # Any other status is a hard error.
        raise ComCatError(f"FDSN {status}: {resp.url}\n{resp.text[:500]}")

    raise ComCatError(f"unreachable: exhausted retries for {url}: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Count + query (the spine)
# ─────────────────────────────────────────────────────────────────────────────


def _base_params(
    *,
    starttime: str | datetime | pd.Timestamp | None,
    endtime: str | datetime | pd.Timestamp | None,
    bbox: BBox | None,
    minmagnitude: float | None,
    updatedafter: str | datetime | pd.Timestamp | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    p: dict[str, Any] = {}
    if starttime is not None:
        p["starttime"] = _iso(starttime)
    if endtime is not None:
        p["endtime"] = _iso(endtime)
    if updatedafter is not None:
        p["updatedafter"] = _iso(updatedafter)
    if minmagnitude is not None:
        p["minmagnitude"] = float(minmagnitude)
    if bbox is not None:
        p.update(
            minlatitude=bbox.lat_min,
            maxlatitude=bbox.lat_max,
            minlongitude=bbox.lon_min,
            maxlongitude=bbox.lon_max,
        )
    if extra:
        p.update(extra)
    return p


def fetch_comcat_count(
    *,
    starttime: str | datetime | pd.Timestamp | None = None,
    endtime: str | datetime | pd.Timestamp | None = None,
    bbox: BBox | None = None,
    minmagnitude: float | None = None,
    updatedafter: str | datetime | pd.Timestamp | None = None,
    session: requests.Session | None = None,
    extra: dict[str, Any] | None = None,
) -> int:
    """Return the event count for a query via the cheap ``/count`` endpoint.

    Used to decide whether a single ``/query`` would exceed the 20,000-event cap before paying
    for the full download (USGS FDSN ``event`` ``/count``).
    """
    params = _base_params(
        starttime=starttime,
        endtime=endtime,
        bbox=bbox,
        minmagnitude=minmagnitude,
        updatedafter=updatedafter,
        extra=extra,
    )
    params["format"] = "text"  # /count returns a bare integer body
    resp = _request("count", params, session=session)
    if resp is None:
        return 0
    body = resp.text.strip()
    try:
        return int(body)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ComCatError(f"unexpected /count body: {body!r}") from exc


def _query_geojson(
    params: dict[str, Any], *, session: requests.Session | None = None
) -> list[dict[str, Any]]:
    """Run one ``/query`` returning GeoJSON; return its ``features`` list ([] on 204)."""
    q = dict(params)
    q["format"] = "geojson"
    q.setdefault("orderby", "time-asc")
    resp = _request("query", q, session=session)
    if resp is None:
        return []
    try:
        payload = resp.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ComCatError(f"non-JSON GeoJSON response: {resp.text[:300]}") from exc
    return list(payload.get("features", []))


def _time_tiles(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bbox: BBox | None,
    minmagnitude: float | None,
    updatedafter: str | datetime | pd.Timestamp | None,
    extra: dict[str, Any] | None,
    target: int,
    session: requests.Session | None,
    _depth: int = 0,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield time sub-windows each estimated to be ``<= target`` events, by recursive bisection.

    ``/count`` drives the split decision; if a window is still flagged over-large at query time
    (the catalog grew between count and query), :func:`fetch_comcat` bisects again as a fallback.
    """
    if _depth > 40:  # ~1e12 splits — pathological; bail rather than spin
        raise ComCatError("time tiling exceeded max depth; window too dense to split")

    n = fetch_comcat_count(
        starttime=start,
        endtime=end,
        bbox=bbox,
        minmagnitude=minmagnitude,
        updatedafter=updatedafter,
        session=session,
        extra=extra,
    )
    # A single-instant window we still can't satisfy: yield it and let the query layer try.
    span = end - start
    if n <= target or span <= pd.Timedelta(seconds=1):
        yield (start, end)
        return
    mid = start + span / 2
    yield from _time_tiles(
        start, mid, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
        extra=extra, target=target, session=session, _depth=_depth + 1,
    )
    yield from _time_tiles(
        mid, end, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
        extra=extra, target=target, session=session, _depth=_depth + 1,
    )


def fetch_comcat(
    *,
    starttime: str | datetime | pd.Timestamp,
    endtime: str | datetime | pd.Timestamp,
    bbox: BBox | None = None,
    minmagnitude: float | None = None,
    updatedafter: str | datetime | pd.Timestamp | None = None,
    source: str = "usgs_comcat",
    tile_target: int = DEFAULT_TILE_TARGET,
    session: requests.Session | None = None,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch a ComCat catalog as a clean DataFrame, tiling around the 20k-event cap.

    The workflow is: ``/count`` → if over ``tile_target`` recursively bisect the time window →
    ``/query`` each tile (with a defensive re-bisection on an unexpected over-large response) →
    concatenate → parse GeoJSON → de-duplicate by ``event_id`` (overlapping tile edges) → sort
    by time. The result matches :data:`caos_seismic.contracts.CATALOG_COLUMNS`.

    Parameters mirror the FDSN ``event`` service. ``updatedafter`` enables daily incremental
    deltas. ``extra`` passes through any additional FDSN params (e.g. ``maxdepth``, ``catalog``).
    """
    start = _to_ts(starttime)
    end = _to_ts(endtime)
    if end <= start:
        raise ValueError(f"endtime ({end}) must be after starttime ({start})")

    features: list[dict[str, Any]] = []
    tiles = list(
        _time_tiles(
            start, end, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
            extra=extra, target=tile_target, session=session,
        )
    )
    logger.info("ComCat fetch: %d tile(s) over [%s, %s)", len(tiles), start, end)

    for t0, t1 in tiles:
        features.extend(
            _query_tile(
                t0, t1, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
                extra=extra, session=session, target=tile_target,
            )
        )

    df = _features_to_frame(features, source=source)
    if not df.empty:
        df = df.drop_duplicates(subset="event_id", keep="last").reset_index(drop=True)
        df = df.sort_values("time").reset_index(drop=True)
    return validate_catalog(df)


def _query_tile(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bbox: BBox | None,
    minmagnitude: float | None,
    updatedafter: str | datetime | pd.Timestamp | None,
    extra: dict[str, Any] | None,
    session: requests.Session | None,
    target: int,
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """Query one tile; if the service still says "too large", bisect and retry the halves."""
    params = _base_params(
        starttime=start, endtime=end, bbox=bbox, minmagnitude=minmagnitude,
        updatedafter=updatedafter, extra=extra,
    )
    try:
        return _query_geojson(params, session=session)
    except ComCatError as exc:
        span = end - start
        if "over-large" not in str(exc) or span <= pd.Timedelta(seconds=1) or _depth > 40:
            raise
        mid = start + span / 2
        logger.info("tile over-large at query time; bisecting [%s, %s)", start, end)
        left = _query_tile(
            start, mid, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
            extra=extra, session=session, target=target, _depth=_depth + 1,
        )
        right = _query_tile(
            mid, end, bbox=bbox, minmagnitude=minmagnitude, updatedafter=updatedafter,
            extra=extra, session=session, target=target, _depth=_depth + 1,
        )
        return left + right


# ─────────────────────────────────────────────────────────────────────────────
# GeoJSON → catalog DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def _features_to_frame(features: Sequence[dict[str, Any]], *, source: str) -> pd.DataFrame:
    """Parse FDSN GeoJSON ``features`` into a DataFrame matching ``CATALOG_COLUMNS``.

    GeoJSON layout (USGS FDSN): ``feature['id']`` is the stable ComCat id; ``properties.time`` is
    epoch **milliseconds** UTC; ``geometry.coordinates`` is ``[lon, lat, depth_km]``;
    ``properties.mag`` / ``properties.magType`` carry the native magnitude and its type — both are
    kept (``magType`` is never dropped).
    """
    cols = list(CATALOG_COLUMNS)
    if not features:
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
        empty["time"] = pd.to_datetime(empty["time"], utc=True)
        for c in ("latitude", "longitude", "depth_km", "mag", "mw"):
            empty[c] = pd.to_numeric(empty[c], errors="coerce")
        return empty[cols]

    rows: list[dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        lon, lat = coords[0], coords[1]
        depth = coords[2] if len(coords) > 2 else None
        t_ms = props.get("time")
        mag = props.get("mag")
        mag_type = props.get("magType")
        rows.append(
            {
                "event_id": feat.get("id"),
                "time": t_ms,
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "mag": mag,
                "mag_type": mag_type,
                # mw is filled by the clean stage (TLS conversion); == mag where already Mw.
                "mw": mag if _is_mw(mag_type) else None,
                "source": source,
            }
        )

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for c in ("latitude", "longitude", "depth_km", "mag", "mw"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["event_id"] = df["event_id"].astype("string")
    df["mag_type"] = df["mag_type"].astype("string")
    df["source"] = df["source"].astype("string")
    # Drop rows with no usable hypocenter or origin time (defensive; FDSN occasionally emits these).
    df = df.dropna(subset=["time", "latitude", "longitude"]).reset_index(drop=True)
    return df[list(CATALOG_COLUMNS)]


def _is_mw(mag_type: Any) -> bool:
    """True if the native magnitude type is already a moment magnitude (no conversion needed)."""
    if not isinstance(mag_type, str):
        return False
    mt = mag_type.strip().lower()
    return mt in {"mw", "mww", "mwc", "mwb", "mwr", "mwp", "mw(mb)", "mwmwc"} or mt.startswith("mw")


def _to_ts(t: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — the thin `caos-seismic fetch` delegation
# ─────────────────────────────────────────────────────────────────────────────


def run_fetch(
    *,
    region: Region | str = "chile",
    days: int | None = None,
    focus: str | None = None,
    min_magnitude: float | None = None,
    updatedafter: str | None = None,
    write_raw: bool = True,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Stage (A) entry point the ``caos-seismic fetch`` command calls (a single REGION view).

    Resolves the region, picks the fetch window (``--days N`` ⇒ the last ``N`` days; otherwise the
    region default ``starttime`` so the whole historical span is pulled), optionally narrows the
    bbox to a configured ``--focus`` sub-region (e.g. ``north`` for Chile, read from
    ``configs/region.<id>.yaml: focus_<key>``), runs the tiled ComCat spine pull, writes the raw
    Parquet store + a ``stage="fetch"`` provenance manifest, and returns a small JSON-able summary
    (the CLI prints it). The ComCat spine needs only ``requests`` + ``pandas`` — no science deps.

    ``min_magnitude`` overrides the configured completeness floor; ``updatedafter`` (ISO-8601 UTC)
    switches to the incremental delta path (only revised/new events) for the daily job.

    Returns ``{"region", "n_events", "time_min", "time_max", "raw_path"}``.
    """
    region_obj = region if isinstance(region, Region) else load_region(region)

    # Window: last N days, or the region/default historical span.
    endtime = pd.Timestamp.now(tz="UTC")
    if days is not None and days > 0:
        starttime: str | pd.Timestamp = endtime - pd.Timedelta(days=int(days))
    else:
        starttime = "2010-01-01"

    # Optional focus sub-region: a tighter bbox declared as `focus_<key>` in the region config.
    bbox_region = region_obj
    if focus:
        focus_bbox = _focus_bbox(region_obj.id, focus)
        if focus_bbox is not None:
            bbox_region = Region(
                id=region_obj.id,
                name_en=region_obj.name_en,
                name_es=region_obj.name_es,
                bbox=focus_bbox,
                m_max=region_obj.m_max,
                attribution=region_obj.attribution,
            )

    df, manifest = fetch_region_comcat(
        bbox_region,
        starttime=starttime,
        endtime=endtime,
        minmagnitude=min_magnitude,
        updatedafter=updatedafter,
        write_raw=write_raw,
        write_manifest=True,
        session=session,
    )
    return {
        "region": region_obj.id,
        "focus": focus,
        "n_events": int(len(df)),
        "time_min": df["time"].min().isoformat() if not df.empty else None,
        "time_max": df["time"].max().isoformat() if not df.empty else None,
        "raw_path": manifest.outputs.get("raw_path"),
    }


def _focus_bbox(region_id: str, focus: str) -> BBox | None:
    """Read a ``focus_<key>`` bbox from ``configs/region.<id>.yaml`` (returns ``None`` if absent)."""
    try:
        import yaml

        from ..config import CONFIG_DIR

        raw = yaml.safe_load((CONFIG_DIR / f"region.{region_id}.yaml").read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - config optional
        return None
    block = (raw or {}).get(f"focus_{focus.strip().lower()}")
    if not isinstance(block, dict):
        return None
    try:
        return BBox(**block)
    except Exception:  # pragma: no cover - malformed focus block
        return None


# ─────────────────────────────────────────────────────────────────────────────
# High-level region pull (writes raw store + fetch manifest)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_region_comcat(
    region: Region | str = "chile",
    *,
    starttime: str | datetime | pd.Timestamp = "2010-01-01",
    endtime: str | datetime | pd.Timestamp | None = None,
    minmagnitude: float | None = None,
    updatedafter: str | datetime | pd.Timestamp | None = None,
    write_raw: bool = True,
    write_manifest: bool = True,
    raw_dir: Path | None = None,
    manifest_dir: Path | None = None,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, Manifest]:
    """End-to-end runnable ComCat pull for a region, with raw store + provenance manifest.

    This is the function ``scripts/fetch`` calls for stage (A). It resolves the region bbox and a
    sensible ``minmagnitude`` (the completeness target floor) from ``configs/``, runs the tiled
    ComCat fetch, writes the raw catalog to ``data/raw/`` as Parquet (gitignored), and writes a
    :class:`~caos_seismic.contracts.Manifest` (``stage="fetch"``) to ``manifests/`` recording the
    query params, retrieved-at timestamp, row counts, and config/code provenance.

    Returns ``(catalog_df, manifest)``. With ``updatedafter`` set, performs an incremental delta
    pull (only revised/new events) — the caller merges it into the existing raw store.
    """
    region_obj = region if isinstance(region, Region) else load_region(region)
    if endtime is None:
        endtime = pd.Timestamp.now(tz="UTC")
    if minmagnitude is None:
        # Default to the region's completeness target floor so we don't pull below Mc.
        try:
            from ..config import load as _load_cfg

            minmagnitude = float(_load_cfg("completeness").get("target", {}).get("m_min", 3.5))
        except Exception:  # pragma: no cover - config optional
            minmagnitude = 3.5

    retrieved_at = datetime.now(timezone.utc).isoformat()
    df = fetch_comcat(
        starttime=starttime,
        endtime=endtime,
        bbox=region_obj.bbox,
        minmagnitude=minmagnitude,
        updatedafter=updatedafter,
        source="usgs_comcat",
        session=session,
    )

    raw_dir = raw_dir or (REPO_ROOT / "data" / "raw")
    manifest_dir = manifest_dir or (REPO_ROOT / "manifests")
    raw_path = raw_dir / f"comcat_{region_obj.id}.parquet"
    if updatedafter is not None:
        # Keep deltas separate so the merge into the base store is explicit + auditable.
        stamp = pd.Timestamp(updatedafter).strftime("%Y%m%dT%H%M%S")
        raw_path = raw_dir / f"comcat_{region_obj.id}.delta.{stamp}.parquet"

    if write_raw and not df.empty:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_path, index=False)
        logger.info("wrote raw store: %s (%d events)", raw_path, len(df))

    manifest = Manifest(
        stage="fetch",
        created_at=retrieved_at,
        region_id=region_obj.id,
        config_hash=_safe_config_hash("region.chile", "completeness"),
        inputs={
            "source": "usgs_comcat",
            "fdsn_base": _comcat_base(),
            "bbox": region_obj.bbox.model_dump(),
            "starttime": _iso(starttime),
            "endtime": _iso(endtime),
            "minmagnitude": minmagnitude,
            "updatedafter": _iso(updatedafter) if updatedafter is not None else None,
            "incremental": updatedafter is not None,
        },
        outputs={
            "raw_path": str(raw_path.relative_to(REPO_ROOT)) if write_raw and not df.empty else None,
            "format": "parquet",
        },
        stats={
            "n_events": int(len(df)),
            "time_min": df["time"].min().isoformat() if not df.empty else None,
            "time_max": df["time"].max().isoformat() if not df.empty else None,
            "mag_types": _value_counts(df, "mag_type"),
        },
    )
    if write_manifest:
        write_manifest_json(manifest, manifest_dir, region_obj.id)
    return df, manifest


def write_manifest_json(manifest: Manifest, manifest_dir: Path, region_id: str) -> Path:
    """Write a manifest to ``manifests/<region>_<stage>_manifest.json`` (versioned provenance)."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{region_id}_{manifest.stage}_manifest.json"
    path.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote manifest: %s", path)
    return path


def _safe_config_hash(*names: str) -> str | None:
    try:
        return config_hash(*names)
    except Exception:  # pragma: no cover - config optional in tests
        return None


def _value_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ComCat fetch — worldwide spine, tiled by latitude band × time around the 20k cap
# ─────────────────────────────────────────────────────────────────────────────
#
# Core thesis of the product: GLOBAL context conditions short-term LOCAL forecasts. The model trains
# on worldwide seismicity; any country is a *view* into one global field. So the spine must be a real,
# multi-decade, whole-Earth pull — not a region box. The worldwide event density (the circum-Pacific
# belt) makes a single global window blow past the 20,000-event/request cap by orders of magnitude, so
# we tile in TWO axes: first by latitude band (coarse, density-aware), then by time within each band
# (the recursive ``/count``-driven bisection already used for the region spine). The same
# ``updatedafter`` mechanism gives cheap daily deltas over the whole globe.


def fetch_comcat_global(
    *,
    starttime: str | datetime | pd.Timestamp,
    endtime: str | datetime | pd.Timestamp,
    minmagnitude: float | None = DEFAULT_GLOBAL_MIN_MAGNITUDE,
    updatedafter: str | datetime | pd.Timestamp | None = None,
    lat_bands: Sequence[tuple[float, float]] = DEFAULT_LAT_BANDS,
    source: str = "usgs_comcat",
    tile_target: int = DEFAULT_TILE_TARGET,
    session: requests.Session | None = None,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch a **worldwide** ComCat catalog, tiled by latitude band × time around the 20k cap.

    This is the global spine the conditional model trains on. The workflow per latitude band is the
    same proven recipe as :func:`fetch_comcat` (``/count`` → recursive time bisection → ``/query``
    each tile with a defensive re-bisection), but the bbox is the band's lat slice over the full
    longitude range. Bands are fetched independently and concatenated; the final frame is de-duplicated
    by ``event_id`` (band edges are open intervals so this is mostly defensive) and time-sorted. The
    result matches :data:`caos_seismic.contracts.CATALOG_COLUMNS`.

    Parameters
    ----------
    starttime, endtime:
        The global pull window (e.g. ``"1990-01-01"`` → now for a multi-decade spine).
    minmagnitude:
        Worldwide completeness floor (default :data:`DEFAULT_GLOBAL_MIN_MAGNITUDE` = 4.5 — below it the
        historical global catalog is neither complete nor homogeneous and the volume explodes). Pass a
        lower value for a recent, smaller window where you specifically want the small events.
    updatedafter:
        Daily incremental delta: only events revised/added since this time (whole globe).
    lat_bands:
        The coarse latitude split (default :data:`DEFAULT_LAT_BANDS`, six 30° bands). Each band is
        time-tiled independently so dense belts do not force the sparse poles into deep bisection.
    tile_target:
        Per-tile soft cap (default :data:`DEFAULT_TILE_TARGET` = 15,000), kept below the hard 20k so a
        catalog revision between ``/count`` and ``/query`` cannot tip a tile over.

    Notes
    -----
    For a true multi-decade M≥4.5 worldwide pull this issues many requests; respect ComCat fair-use
    (the caller throttles the daily job). For daily inference you call this with ``updatedafter`` set,
    which returns only the (small) global delta.
    """
    start = _to_ts(starttime)
    end = _to_ts(endtime)
    if end <= start:
        raise ValueError(f"endtime ({end}) must be after starttime ({start})")

    frames: list[pd.DataFrame] = []
    for lat_min, lat_max in lat_bands:
        band = BBox(lat_min=lat_min, lat_max=lat_max, lon_min=-180.0, lon_max=180.0)
        logger.info("global fetch · latitude band [%.0f, %.0f)", lat_min, lat_max)
        band_df = fetch_comcat(
            starttime=start,
            endtime=end,
            bbox=band,
            minmagnitude=minmagnitude,
            updatedafter=updatedafter,
            source=source,
            tile_target=tile_target,
            session=session,
            extra=extra,
        )
        if not band_df.empty:
            frames.append(band_df)

    if not frames:
        return _features_to_frame([], source=source)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="event_id", keep="last").reset_index(drop=True)
    df = df.sort_values("time").reset_index(drop=True)
    return validate_catalog(df)


def fetch_global_comcat(
    *,
    starttime: str | datetime | pd.Timestamp = "1990-01-01",
    endtime: str | datetime | pd.Timestamp | None = None,
    minmagnitude: float | None = DEFAULT_GLOBAL_MIN_MAGNITUDE,
    updatedafter: str | datetime | pd.Timestamp | None = None,
    lat_bands: Sequence[tuple[float, float]] = DEFAULT_LAT_BANDS,
    write_raw: bool = True,
    write_manifest: bool = True,
    raw_dir: Path | None = None,
    manifest_dir: Path | None = None,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, Manifest]:
    """End-to-end runnable **global** ComCat pull, with raw store + ``stage="fetch"`` manifest.

    This is the function ``scripts/fetch --global`` calls. It runs the worldwide tiled pull
    (:func:`fetch_comcat_global`), writes the raw global catalog to ``data/raw/comcat_global.parquet``
    (gitignored), and writes a provenance :class:`~caos_seismic.contracts.Manifest` (``region_id =
    "global"``) recording the window, bands, minmagnitude, retrieved-at, and row counts.

    With ``updatedafter`` set it performs an **incremental global delta** and writes it to a separate
    ``comcat_global.delta.<stamp>.parquet`` so the merge into the base store stays explicit and
    auditable (the caller merges delta → base, then re-runs clean). Returns ``(catalog_df, manifest)``.
    """
    if endtime is None:
        endtime = pd.Timestamp.now(tz="UTC")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    df = fetch_comcat_global(
        starttime=starttime,
        endtime=endtime,
        minmagnitude=minmagnitude,
        updatedafter=updatedafter,
        lat_bands=lat_bands,
        source="usgs_comcat",
        session=session,
    )

    raw_dir = raw_dir or (REPO_ROOT / "data" / "raw")
    manifest_dir = manifest_dir or (REPO_ROOT / "manifests")
    raw_path = raw_dir / "comcat_global.parquet"
    if updatedafter is not None:
        stamp = pd.Timestamp(updatedafter).strftime("%Y%m%dT%H%M%S")
        raw_path = raw_dir / f"comcat_global.delta.{stamp}.parquet"

    if write_raw and not df.empty:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_path, index=False)
        logger.info("wrote raw global store: %s (%d events)", raw_path, len(df))

    manifest = Manifest(
        stage="fetch",
        created_at=retrieved_at,
        region_id="global",
        config_hash=_safe_config_hash("completeness"),
        inputs={
            "source": "usgs_comcat",
            "fdsn_base": _comcat_base(),
            "scope": "global",
            "bbox": GLOBAL_BBOX.model_dump(),
            "lat_bands": [list(b) for b in lat_bands],
            "starttime": _iso(starttime),
            "endtime": _iso(endtime),
            "minmagnitude": minmagnitude,
            "updatedafter": _iso(updatedafter) if updatedafter is not None else None,
            "incremental": updatedafter is not None,
        },
        outputs={
            "raw_path": str(raw_path.relative_to(REPO_ROOT)) if write_raw and not df.empty else None,
            "format": "parquet",
        },
        stats={
            "n_events": int(len(df)),
            "time_min": df["time"].min().isoformat() if not df.empty else None,
            "time_max": df["time"].max().isoformat() if not df.empty else None,
            "mag_types": _value_counts(df, "mag_type"),
            "lat_band_counts": _lat_band_counts(df, lat_bands),
        },
    )
    if write_manifest:
        write_manifest_json(manifest, manifest_dir, "global")
    return df, manifest


def _lat_band_counts(
    df: pd.DataFrame, lat_bands: Sequence[tuple[float, float]]
) -> dict[str, int]:
    """Event count per latitude band (a quick density sanity stat for the global fetch manifest)."""
    if df.empty:
        return {}
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    out: dict[str, int] = {}
    for lat_min, lat_max in lat_bands:
        sel = (lat >= lat_min) & (lat < lat_max)
        out[f"[{lat_min:.0f},{lat_max:.0f})"] = int(sel.sum())
    return out


def run_fetch_global(
    *,
    days: int | None = None,
    min_magnitude: float | None = DEFAULT_GLOBAL_MIN_MAGNITUDE,
    start: str | None = None,
    updatedafter: str | None = None,
    write_raw: bool = True,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Stage (A) GLOBAL entry point — what ``caos-seismic fetch --global`` calls.

    Picks the window (``--days N`` ⇒ last N days; else ``--start`` or the multi-decade default
    ``"1990-01-01"``), runs the worldwide tiled ComCat pull, writes the raw global store + a
    ``region_id="global"`` fetch manifest, and returns a small JSON-able summary the CLI prints.

    ``--updatedafter`` (ISO-8601 UTC) takes precedence for the **daily incremental** global delta — it
    is the cheap path the production job uses every day; pair it with the existing base global store.
    """
    endtime = pd.Timestamp.now(tz="UTC")
    if days is not None and days > 0:
        starttime: str | pd.Timestamp = endtime - pd.Timedelta(days=int(days))
    elif start:
        starttime = start
    else:
        starttime = "1990-01-01"

    df, manifest = fetch_global_comcat(
        starttime=starttime,
        endtime=endtime,
        minmagnitude=min_magnitude,
        updatedafter=updatedafter,
        write_raw=write_raw,
        write_manifest=True,
        session=session,
    )
    return {
        "scope": "global",
        "min_magnitude": min_magnitude,
        "incremental": updatedafter is not None,
        "n_events": int(len(df)),
        "time_min": df["time"].min().isoformat() if not df.empty else None,
        "time_max": df["time"].max().isoformat() if not df.empty else None,
        "raw_path": manifest.outputs.get("raw_path"),
        "lat_band_counts": manifest.stats.get("lat_band_counts"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional helpers (regional networks + long-term anchors) — lazy heavy imports
# ─────────────────────────────────────────────────────────────────────────────


def _require_obspy():
    """Import ObsPy lazily, raising an actionable error if the science extra isn't installed."""
    try:
        import obspy  # noqa: F401
        from obspy.clients.fdsn import Client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when obspy absent
        raise ImportError(
            "ObsPy is required for regional FDSN clients (CSN/EarthScope, ISC, EMSC) and for "
            "parsing GCMT .ndk / ISC-GEM. The ComCat spine does NOT need it. Install the science "
            "extra:\n    pip install 'caos-seismic[science]'   (or: pip install obspy)"
        ) from exc
    from obspy.clients.fdsn import Client

    return Client


#: ObsPy FDSN short names for the regional providers we support.
_FDSN_CLIENTS: dict[str, str] = {
    "csn": "EARTHSCOPE",   # Chile CSN surfaced via EarthScope/IRIS FDSN (net C/C1)
    "earthscope": "EARTHSCOPE",
    "iris": "EARTHSCOPE",
    "isc": "ISC",
    "emsc": "EMSC",
    "ingv": "INGV",
    "geonet": "GEONET",
    "scedc": "SCEDC",
    "ncedc": "NCEDC",
}


def fetch_fdsn_obspy(
    provider: str,
    *,
    starttime: str | datetime | pd.Timestamp,
    endtime: str | datetime | pd.Timestamp,
    bbox: BBox | None = None,
    minmagnitude: float | None = None,
    network: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Fetch a regional/global catalog via an ObsPy FDSN client (CSN, ISC, EMSC, INGV, …).

    Used for the **regional driver** (Chile → CSN via EarthScope/IRIS, net ``C``/``C1``) and the
    independent **cross-check** (EMSC), per ``docs/data-and-pipelines.md`` §1.2. ObsPy is imported
    lazily; an actionable error is raised if it is missing. The result matches
    :data:`caos_seismic.contracts.CATALOG_COLUMNS` so it can be deduped against the ComCat spine.

    Note: ``get_events()`` has no bulk analogue — loop time windows and respect each provider's
    20k cap. This helper does a single window; callers tile if needed.
    """
    Client = _require_obspy()
    key = provider.strip().lower()
    short = _FDSN_CLIENTS.get(key)
    if short is None:
        raise ValueError(
            f"unknown FDSN provider {provider!r}; known: {sorted(_FDSN_CLIENTS)}"
        )
    src = source or key
    client = Client(short)
    kwargs: dict[str, Any] = dict(starttime=_obspy_utc(starttime), endtime=_obspy_utc(endtime))
    if minmagnitude is not None:
        kwargs["minmagnitude"] = float(minmagnitude)
    if bbox is not None:
        kwargs.update(
            minlatitude=bbox.lat_min, maxlatitude=bbox.lat_max,
            minlongitude=bbox.lon_min, maxlongitude=bbox.lon_max,
        )
    if network:
        kwargs["network"] = network

    cat = client.get_events(**kwargs)
    return _obspy_catalog_to_frame(cat, source=src)


def fetch_region_fdsn(
    region: Region | str,
    provider: str | None = None,
    *,
    starttime: str | datetime | pd.Timestamp = "2000-01-01",
    endtime: str | datetime | pd.Timestamp | None = None,
    minmagnitude: float | None = None,
    network: str | None = None,
    window_days: float = 90.0,
    source: str | None = None,
) -> pd.DataFrame:
    """Per-VIEW regional FDSN pull for a country view — the local-network catalog (low, stable Mc).

    The product trains globally but each country is a *view*; the short-horizon skill of a view scales
    with how low and stable its local Mc is, so each view is driven by its **regional network**
    (Chile → CSN via EarthScope/IRIS net ``C``/``C1``; California → SCEDC/NCEDC; NZ → GeoNet; Italy →
    INGV). This helper resolves the provider (explicit ``provider`` or the region config's
    ``catalogs.regional`` key), clips to the region bbox, and pulls the window in ``window_days`` slices
    (``get_events`` has no bulk analogue and each provider enforces the 20k cap), concatenating into one
    CATALOG_COLUMNS frame deduped against the ComCat spine downstream. ObsPy is imported lazily.

    Parameters
    ----------
    region:
        A :class:`~caos_seismic.contracts.Region` or region id (its bbox + ``catalogs.regional`` are read).
    provider:
        FDSN provider key (``csn``/``earthscope``/``scedc``/``geonet``/``ingv``/…). If ``None``, read
        ``catalogs.regional`` from ``configs/region.<id>.yaml``.
    window_days:
        Time-slice width for the windowed pull (kept well under each provider's per-request cap).
    """
    from ..config import load_region as _load_region

    region_obj = region if isinstance(region, Region) else _load_region(region)
    prov = provider or _region_regional_provider(region_obj.id)
    if not prov:
        raise ValueError(
            f"no regional FDSN provider for region {region_obj.id!r}: pass `provider=` or set "
            f"`catalogs.regional` in configs/region.{region_obj.id}.yaml"
        )
    end = _to_ts(endtime) if endtime is not None else pd.Timestamp.now(tz="UTC")
    start = _to_ts(starttime)
    if end <= start:
        raise ValueError(f"endtime ({end}) must be after starttime ({start})")

    src = source or prov.strip().lower()
    step = pd.Timedelta(days=float(window_days))
    frames: list[pd.DataFrame] = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        try:
            chunk = fetch_fdsn_obspy(
                prov,
                starttime=t0,
                endtime=t1,
                bbox=region_obj.bbox,
                minmagnitude=minmagnitude,
                network=network,
                source=src,
            )
        except Exception as exc:  # one empty/oversized window must not abort the whole pull
            logger.warning("regional FDSN window [%s, %s) failed for %s: %s", t0, t1, prov, exc)
            chunk = _obspy_catalog_to_frame_empty(src)
        if not chunk.empty:
            frames.append(chunk)
        t0 = t1

    if not frames:
        return _obspy_catalog_to_frame_empty(src)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="event_id", keep="last").reset_index(drop=True)
    df = df.sort_values("time").reset_index(drop=True)
    return validate_catalog(df)


def _region_regional_provider(region_id: str) -> str | None:
    """Read ``catalogs.regional`` (the view's local network) from ``configs/region.<id>.yaml``."""
    try:
        import yaml

        from ..config import CONFIG_DIR

        raw = yaml.safe_load((CONFIG_DIR / f"region.{region_id}.yaml").read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - config optional
        return None
    cat = (raw or {}).get("catalogs") or {}
    prov = cat.get("regional")
    return str(prov) if prov else None


def _obspy_catalog_to_frame_empty(source: str) -> pd.DataFrame:
    """Empty CATALOG_COLUMNS frame (typed ``time`` column) for skipped/failed regional windows."""
    return _features_to_frame([], source=source)


def fetch_emsc_crosscheck(
    *,
    starttime: str | datetime | pd.Timestamp,
    endtime: str | datetime | pd.Timestamp,
    bbox: BBox | None = None,
    minmagnitude: float | None = None,
    window_days: float = 30.0,
) -> pd.DataFrame:
    """Independent EMSC SeismicPortal cross-check pull (deduped against the ComCat spine downstream).

    EMSC is the *independent* provider that lets the clean stage catch ComCat false/retracted events
    and disagreements (it is the lowest-authority source in the dedupe priority, used only to confirm).
    Pulled in ``window_days`` slices via the ObsPy ``EMSC`` FDSN client (lazy import). Returns a
    CATALOG_COLUMNS frame tagged ``source="emsc"``.
    """
    end = _to_ts(endtime)
    start = _to_ts(starttime)
    if end <= start:
        raise ValueError(f"endtime ({end}) must be after starttime ({start})")
    step = pd.Timedelta(days=float(window_days))
    frames: list[pd.DataFrame] = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        try:
            chunk = fetch_fdsn_obspy(
                "emsc", starttime=t0, endtime=t1, bbox=bbox,
                minmagnitude=minmagnitude, source="emsc",
            )
        except Exception as exc:
            logger.warning("EMSC cross-check window [%s, %s) failed: %s", t0, t1, exc)
            chunk = _obspy_catalog_to_frame_empty("emsc")
        if not chunk.empty:
            frames.append(chunk)
        t0 = t1
    if not frames:
        return _obspy_catalog_to_frame_empty("emsc")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="event_id", keep="last").reset_index(drop=True)
    return validate_catalog(df.sort_values("time").reset_index(drop=True))


def _obspy_utc(t: str | datetime | pd.Timestamp):
    from obspy import UTCDateTime

    return UTCDateTime(_iso(t))


def _obspy_catalog_to_frame(cat, *, source: str) -> pd.DataFrame:
    """Convert an ObsPy ``Catalog`` to the CATALOG_COLUMNS DataFrame, preserving the magnitude type."""
    rows: list[dict[str, Any]] = []
    for ev in cat:
        origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        magnitude = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        if origin is None:
            continue
        mag = float(magnitude.mag) if magnitude and magnitude.mag is not None else None
        mag_type = magnitude.magnitude_type if magnitude else None
        rid = str(ev.resource_id)
        eid = rid.rsplit("/", 1)[-1] if "/" in rid else rid
        rows.append(
            {
                "event_id": eid,
                "time": pd.Timestamp(origin.time.datetime, tz="UTC"),
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "depth_km": (origin.depth / 1000.0) if origin.depth is not None else None,
                "mag": mag,
                "mag_type": mag_type,
                "mw": mag if _is_mw(mag_type) else None,
                "source": source,
            }
        )
    df = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    if df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for c in ("latitude", "longitude", "depth_km", "mag", "mw"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return validate_catalog(df)


def download_isc_gem(dest: Path, *, url: str | None = None, session: requests.Session | None = None) -> Path:
    """Download the ISC-GEM Global Instrumental Catalogue CSV to ``dest`` (gitignored raw store).

    ISC-GEM v12.1 (DOI ``10.31905/d808b825``) is the Mw-homogenized long-term anchor for the
    b-value and the ML/mb→Mw conversion overlap. **License: CC-BY-SA 3.0** — a *redistributed*
    derived catalog must keep the license and attribution. This helper only downloads the raw file
    to the gitignored store; it does not redistribute.

    Pass an explicit ``url`` to the current CSV (the ISC download page issues versioned URLs); no
    hard-coded mirror is assumed.
    """
    if url is None:
        raise ValueError(
            "download_isc_gem requires an explicit `url` to the current ISC-GEM CSV "
            "(obtain it from https://www.isc.ac.uk/iscgem/ download.php — versioned per release). "
            "ISC-GEM is CC-BY-SA 3.0: keep license + attribution on any redistributed derivative."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _request_raw(url, session=session)
    dest.write_bytes(resp.content)
    logger.info("downloaded ISC-GEM CSV → %s (%d bytes)", dest, len(resp.content))
    return dest


#: ISC-GEM CSV column → catalog column. The ISC-GEM main CSV is a ``#``-commented, comma-separated file
#: whose data header row is ``date, lat, lon, smajax, sminax, ..., depth, ..., mw, ...`` with columns
#: identified by name in the last comment line. We map by name to stay robust to column re-ordering
#: across versions. All ISC-GEM magnitudes are already Mw-homogenized (the catalog's whole point).
_ISC_GEM_COLMAP: dict[str, str] = {
    "eventid": "event_id",
    "date": "time",
    "lat": "latitude",
    "lon": "longitude",
    "depth": "depth_km",
    "mw": "mag",
}


def read_isc_gem_csv(path: Path, *, source: str = "isc_gem") -> pd.DataFrame:
    """Parse the ISC-GEM Global Instrumental Catalogue main CSV into the CATALOG_COLUMNS frame.

    ISC-GEM (1904–present, **Mw-homogenized**, relocated) is the long-term homogeneous anchor for the
    global b-value, large-event recurrence, and — crucially — the ``native → Mw`` TLS conversion overlap
    in :mod:`caos_seismic.data.clean`. Every magnitude here is already Mw, so ``mag_type = "Mw"`` and
    ``mw == mag``. Core deps only (``pandas``); the file is a ``#``-commented CSV whose last comment line
    is the column header. **License: CC-BY-SA 3.0** — keep attribution on any redistributed derivative.

    Robust to ISC-GEM's quirks: leading ``#`` comment block, a header carried as the final comment line,
    columns identified by *name* (order varies across versions), and whitespace padding.
    """
    header: list[str] | None = None
    data_lines: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                # The data header is the last comment line that names the known columns.
                candidate = [c.strip().lower() for c in line.lstrip("#").split(",")]
                if "lat" in candidate and "lon" in candidate and "mw" in candidate:
                    header = candidate
                continue
            if line.strip():
                data_lines.append(line)

    if header is None or not data_lines:
        raise ComCatError(
            f"could not locate an ISC-GEM data header (lat/lon/mw) in {path}; the file format may have "
            "changed — inspect the CSV header comment block."
        )

    from io import StringIO

    # ISC-GEM repeats quality columns ("unc", "q") several times, so the header has duplicate names.
    # Parse positionally (uniquified names) and resolve the columns we want by their FIRST occurrence.
    seen: dict[str, int] = {}
    uniq_header: list[str] = []
    first_index: dict[str, int] = {}
    for idx, name in enumerate(header):
        if name not in first_index:
            first_index[name] = idx
        if name in seen:
            seen[name] += 1
            uniq_header.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            uniq_header.append(name)

    frame = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None,
        names=uniq_header,
        skipinitialspace=True,
    )
    # Map the named ISC-GEM columns we use (by first occurrence) into the catalog schema.
    out = pd.DataFrame(index=frame.index)
    for src, dst in _ISC_GEM_COLMAP.items():
        if src in first_index and first_index[src] < frame.shape[1]:
            out[dst] = frame.iloc[:, first_index[src]]
    for col in ("event_id", "time", "latitude", "longitude", "depth_km", "mag"):
        if col not in out.columns:
            out[col] = pd.NA

    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ("latitude", "longitude", "depth_km", "mag"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["event_id"] = out["event_id"].astype("string")
    if out["event_id"].isna().all():
        # Fall back to a stable synthetic id from origin time + hypocenter when none is provided.
        out["event_id"] = (
            "iscgem_"
            + out["time"].astype("string").fillna("NaT")
            + "_"
            + out["latitude"].round(3).astype("string")
            + "_"
            + out["longitude"].round(3).astype("string")
        )
    out["mag_type"] = "Mw"
    out["mw"] = out["mag"]
    out["source"] = source
    out = out.dropna(subset=["time", "latitude", "longitude"]).reset_index(drop=True)
    return validate_catalog(out[list(CATALOG_COLUMNS)])


def download_gcmt_ndk(dest: Path, *, url: str | None = None, session: requests.Session | None = None) -> Path:
    """Download a GCMT ``.ndk`` moment-tensor file to ``dest`` (gitignored raw store).

    GCMT centroid moment tensors (Mw, nodal planes, P/T axes) for M≳5 since 1976 — the mechanism
    enricher and a Mw anchor for homogenization. Parse with ObsPy (``obspy.read_events``, module
    ``obspy.io.ndk``) via :func:`read_gcmt_ndk`. Free for research with citation (Dziewonski et
    al. 1981; Ekström et al. 2012). Pass the exact monthly/aggregate ``.ndk`` URL from
    ``globalcmt.org/CMTfiles.html``.
    """
    if url is None:
        raise ValueError(
            "download_gcmt_ndk requires an explicit `url` to a GCMT .ndk file "
            "(from https://www.globalcmt.org/CMTfiles.html). Cite Ekström et al. (2012)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _request_raw(url, session=session)
    dest.write_bytes(resp.content)
    logger.info("downloaded GCMT ndk → %s (%d bytes)", dest, len(resp.content))
    return dest


def read_gcmt_ndk(path: Path, *, source: str = "gcmt") -> pd.DataFrame:
    """Parse a GCMT ``.ndk`` file into the CATALOG_COLUMNS DataFrame (Mw, lazy ObsPy import).

    Used to assemble the ISC-GEM/GCMT Mw overlap that anchors the TLS magnitude conversion in
    :mod:`caos_seismic.data.clean`. The reported magnitude is moment magnitude, so ``mag_type`` is
    ``"Mw"`` and ``mw == mag``.
    """
    try:
        from obspy import read_events
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ObsPy is required to parse GCMT .ndk files (obspy.io.ndk). "
            "Install: pip install 'caos-seismic[science]'"
        ) from exc
    cat = read_events(str(path))
    df = _obspy_catalog_to_frame(cat, source=source)
    # GCMT magnitudes are moment magnitudes regardless of how ObsPy labels them.
    df["mag_type"] = "Mw"
    df["mw"] = df["mag"]
    return validate_catalog(df)


def _request_raw(url: str, *, session: requests.Session | None = None, timeout_s: float = 300.0) -> requests.Response:
    """GET an arbitrary file URL with the polite User-Agent (for ISC-GEM/GCMT downloads)."""
    sess = session or requests
    resp = sess.get(url, headers={"User-Agent": _user_agent()}, timeout=timeout_s)
    if resp.status_code != 200:
        raise ComCatError(f"download failed ({resp.status_code}) for {url}")
    return resp
