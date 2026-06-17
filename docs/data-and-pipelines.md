<!-- markdownlint-disable MD013 MD033 -->
# Data & Pipelines

> The dominant signal in a short-horizon seismic forecast is the **earthquake catalog itself**
> (ETAS-class clustering). Every enricher (geodesy, fault geometry, slab, tides) is *upside, not
> foundation*, and ships only when it shows positive, significant information gain over a
> catalog-only ETAS on held-out data. This page documents the data sources and their licenses, the
> versioned pipeline DAG, and the hard line between what is committed to git and what is rebuildable.

The system runs **one inference per day**. The heavy work is offline; the served surface is a single
compact artifact. The contract for that artifact is in
[`contracts.py`](../src/caos_seismic/contracts.py); the access layer is
[`data/fetch.py`](../src/caos_seismic/data/fetch.py).

---

## 1. Data source register

Every row below is scriptable. License, stability, and the exact access path are stated.
Attribution-required sources are flagged; the web app's credits page renders the attribution strings
carried in each region config (`region.<id>.yaml: attribution`).

### 1.1 Catalog spine and global anchors

| Source | Role | Access | License | Cadence |
|---|---|---|---|---|
| **USGS ComCat** | the daily spine | FDSN `event` web service (`requests` only — no ObsPy) | US Government work, public domain | real-time |
| **ISC-GEM v12.1** | Mw-homogenized long-term anchor (1904–2021, $M \ge 5.5$) | CSV download from ISC | **CC-BY-SA 3.0** (share-alike) | versioned (DOI [10.31905/d808b825](https://doi.org/10.31905/d808b825)) |
| **ISC Bulletin (REVIEWED)** | cleanest retrospective hypocenters (~24-month lag) | ISC catalogue web service / `ObsPy Client("ISC")` | open for research with attribution | ~24-month lag |
| **GCMT** | moment-tensor / mechanism enricher + Mw anchor | `.ndk` from globalcmt.org, parsed via ObsPy | free for research with citation | monthly |
| **EMSC SeismicPortal** | independent dedup cross-check | FDSN `event` / `ObsPy Client("EMSC")` | open for research with attribution | real-time |

**ComCat specifics (encoded in the spine).** The FDSN `event` service caps a single request at
**20,000 events** (returns HTTP 400 over the cap), so the spine queries `/count` first and recursively
bisects the time window until each tile is under a safe target before stitching the results
(`fetch_comcat()` in [`data/fetch.py`](../src/caos_seismic/data/fetch.py)). The `updatedafter`
parameter drives daily incremental deltas (only events whose origin/magnitude was revised since the
last run — ComCat continuously revises *and retracts* events). `magType` is a **first-class field**:
it is read and kept (mixing mb/Ms/Mw silently distorts the Gutenberg–Richter tail and the Mw
homogenization depends on it). Every call carries a polite `User-Agent` and retries with exponential
backoff on transient (429/503) and over-large (400/413 → "tile smaller") responses.

> **ISC-GEM is CC-BY-SA 3.0.** Internal training is unaffected, but any *redistributed* ISC-GEM-derived
> catalog must keep the license and attribution. `download_isc_gem()` only fetches the raw file to the
> gitignored store and requires an explicit, versioned URL — it never assumes a mirror and never
> redistributes.

### 1.2 Regional networks (the source of short-horizon skill)

Short-term skill scales with how low and stable $M_c$ is, so the **target region** is driven by its
local network; ComCat is global context and EMSC is an independent cross-check.

| Region | Network | Access | License / attribution |
|---|---|---|---|
| **Chile** (v0) | CSN — Centro Sismológico Nacional | EarthScope/IRIS FDSN (`Client("EARTHSCOPE")`, net `C`/`C1`) | public use with **mandatory CSN attribution** |
| California | SCEDC (Caltech) + NCEDC (Berkeley) | `Client("SCEDC")` / `Client("NCEDC")`, AWS open data `s3://scedc-pds` (`--no-sign-request`) | open research use with attribution |
| New Zealand | GeoNet | `service.geonet.org.nz/fdsnws` / `Client("GEONET")` | **CC-BY 3.0 NZ** ("Earth Sciences New Zealand") |
| Italy | INGV ISIDe | `webservices.ingv.it/fdsnws` / `Client("INGV")` | CC-BY (confirm version per dataset) |
| Japan | JMA / NIED Hi-net | registration + agreement gated | **NOT redistributable — internal use only** |

All regional access goes through `fetch_fdsn_obspy()` in
[`data/fetch.py`](../src/caos_seismic/data/fetch.py): one `obspy.clients.fdsn.Client` API across every
FDSN provider, imported **lazily** with an actionable error if ObsPy is absent — the ComCat spine does
not need it. `get_events()` has no bulk analogue, so callers loop time windows and respect each
provider's 20k cap.

> **Hard flag — Japan.** JMA / NIED Hi-net raw files, credentials, and unvetted derived products are
> **internal-only and never shipped** in the public repo or app. Check the agreement before exposing
> any derivative.

### 1.3 Enrichers (geophysical context covariates)

Ranked by *expected* lift for a conditional short-term forecast (the ranking is inferred, not
measured — each enricher's marginal information gain over a catalog-only ETAS is quantified on
held-out data *before* it touches a public number). None substitutes for the catalog.

| Enricher | What it gives | License | Cadence |
|---|---|---|---|
| **Slab2** (highest, subduction) | depth-to-slab, dip, interface distance | USGS public domain | static |
| **GEM Active Faults** | distance-to-fault, fault style | CC-BY-SA 4.0 (verify repo LICENSE) | versioned |
| **Bird (2003) PB2002** plate model | distance-to-boundary, boundary type | open research | static |
| **NGL GNSS / MIDAS** | strain-rate field (feeds the *background* term) | open + attribution | daily–weekly |
| **Focal-mechanism stress** | rake, P/T axes, $\Delta\mathrm{CFS}$ triggering | open + cite | event-driven |
| **InSAR — COMET LiCSAR** *(deferred)* | surface deformation | products free + attribution | per-acquisition |
| **Heat flow** *(skip v1)* | crustal background | open | static |

For Chile the relevant enrichers are `enrichers: [slab2, gem_faults, bird_pb2002, ngl_gnss]`
(`region.chile.yaml`), and the region notes record that subduction megathrusts violate the
isotropic-kernel / point-source assumptions of generic ETAS for great earthquakes — Slab2 geometry
and anisotropic/finite-fault triggering are needed, and California generic parameters must **not** be
reused.

### 1.4 Tidal stress (a computed feature, not a downloaded catalog)

Tides are useless as a standalone predictor — tidal stresses on faults (~0.1–10 kPa) are ~$10^{-3}$–
$10^{-4}$ of earthquake stress drops (~1–10 MPa), so they can only *advance/retard* a rupture already
near failure, never cause one. The effect is real but small and regime-dependent (~0.5–1 % global
rate excess; up to a factor ~3 only for shallow ocean-loaded thrusts). It is defensible solely as a
**physically-motivated, regularized covariate** that may shrink to ~0, encoded as a rate-and-state
multiplier

$$\frac{R}{r} = \exp\!\left(\frac{\Delta\mathrm{CFS}(t)}{A\,\sigma}\right),
\qquad \Delta\mathrm{CFS}(t) = \Delta\tau(t) + \mu_f\,\Delta\sigma_n(t),$$

with a learnable, regularizable coefficient allowed to go to ~0. Body tide via `pygtide`
(ETERNA PREDICT), ocean tidal loading via SPOTL with a global ocean-tide model (TPXO/GOT/FES). **For
Chilean / subduction / coastal targets ocean loading dominates — skipping it is the single biggest
tidal-modeling error.** Honest expectation: for most regions the gain is negligible and the near-null
is reported openly; only shallow ocean-loaded thrust/ridge regions and a separate tremor/slow-slip
channel show measurable (still few-percent) lift, validated with-vs-without in the CSEP harness on
declustered catalogs out-of-sample — never from in-sample Schuster p-values (huge $N$ makes tiny
effects "significant").

---

## 2. Catalog hygiene (mandatory, ordered)

A model trained on a dirty catalog learns the network's detection changes, not the Earth. The order
is load-bearing.

1. **$M_c(x, y, t)$** — estimate the magnitude of completeness **per spatial cell and time epoch**,
   never globally (a single $M_c$ injects fake non-stationarity). Primary estimator: maximum-curvature
   (MAXC) with a configurable correction, cross-checked with the goodness-of-fit test and EMR for
   uncertainty (Wiemer & Wyss 2000, *BSSA* 90(4), 859–869,
   doi:[10.1785/0119990114](https://doi.org/10.1785/0119990114); Woessner & Wiemer 2005, *BSSA* 95(2),
   684–698, doi:[10.1785/0120040007](https://doi.org/10.1785/0120040007)). The Aki–Utsu
   binning-corrected $b$-value is $\hat b = \log_{10} e / (\bar m - (M_c - \Delta M/2))$. Implemented
   in [`catalog/completeness.py`](../src/caos_seismic/catalog/completeness.py) (`mc_estimate`,
   `rolling_mc`, `aki_utsu_b_value`); config in `completeness.yaml`.

   > **The +0.2 MAXC correction is California-tuned, not universal.** Re-validate it per region
   > (GFT/EMR cross-check + FMD inspection) and take the conservative value
   > (`completeness.yaml: mc.maxc_correction`). $M_c$ and $b$ are re-estimated on a rolling window
   > (`mc.rolling_window_days`) and exposed as monitored quantities; drift in either flags a
   > catalog/network breakage. The $M_c$ grid is a **first-class versioned artifact** stored next to
   > each catalog snapshot.

2. **Magnitude homogenization to Mw** — catalogs mix ML/mb/Ms/Md/Mw (different saturation, different
   physics). Where Mw is missing for small events, a regional **total-least-squares** conversion
   (both axes have error — not OLS) is fit and anchored on the ISC-GEM / GCMT overlap. Both the native
   value+type and the Mw-homogenized value are stored (the `mag` / `mag_type` / `mw` columns of
   `CATALOG_COLUMNS`). The conversion is versioned — a wrong conversion shifts the whole GR tail and
   every rate forecast.

3. **Declustering — the dual-catalog rule** (the most common pipeline mistake, made explicit in
   `declustering.yaml`):
   - **Declustered catalog** (Gardner–Knopoff windows, OpenQuake hmtk coefficients
     $L(M) = 10^{0.1238M + 0.983}$ km, $T(M) = 10^{0.032M + 2.7389}$ d for $M \ge 6.5$ else
     $10^{0.5409M - 0.547}$ d) → **only** for the stationary Poisson background $\mu(x,y)$ and the
     Poisson-baseline calibration.
   - **Full, un-declustered catalog** → fed to the conditional/ETAS model, because
     aftershock/foreshock triggering **is** the predictable signal. *Declustering the conditional
     input is the single most common pipeline mistake.*
   - **Zaliapin–Ben-Zion** nearest-neighbor proximity is computed as ML *features* (not just keep/drop
     labels): $\eta_{ij} = t_{ij}\,(r_{ij})^{d_f}\,10^{-b\,m_i}$, decomposed
     $T_j = t_{ij}\,10^{-q b m_i}$, $R_j = (r_{ij})^{d_f}\,10^{-(1-q) b m_i}$, $q \approx 0.5$
     (`declustering.yaml: features`).

> **Target-side consequence.** Because the forecast deliberately includes clustering, the *scored*
> target events are mostly aftershocks — so a trivial "aftershocks follow mainshocks" model already
> passes *consistency* tests. Scoring is therefore on the **non-declustered** catalog, and skill is
> established **only** by winning comparison tests against a real ETAS baseline (both capture Omori).
> See [`evaluation.md`](evaluation.md).

---

## 3. The versioned pipeline DAG

The pipeline is a deterministic DAG: each stage reads a versioned input manifest, writes an output
manifest, and is fully re-runnable from manifests + code + configs. Raw data is rebuildable and never
committed. Every stage stamps provenance (source catalog versions, $M_c$ grid version, declustering
choice, config hash, code git SHA, issue timestamp) — see
[`inference/provenance.py`](../src/caos_seismic/inference/provenance.py) and the `Manifest` schema in
[`contracts.py`](../src/caos_seismic/contracts.py). A rendered version of this DAG is in
[`diagrams/pipeline-flow.svg`](diagrams/pipeline-flow.svg).

```
            configs/ (VERSIONED): region · grid · completeness · declustering · etas · forecast · publish
                                                  │
 (A) FETCH ───────────────────────────────────────▼──────────────────────────────────────────────────────
   ComCat updatedafter delta + regional FDSN (CSN/SCEDC/GeoNet/INGV) + EMSC cross-check
   ISC-GEM / GCMT (periodic refresh)         Slab2 / faults / plate model / NGL strain (static)
        → raw event store (Parquet)                              [NEVER versioned]
        → fetch manifest (per-source URL, params, retrieved-at, row counts, checksums)   [VERSIONED]
                                                  │
 (B) CLEAN / HOMOGENIZE ───────────────────────────▼──────────────────────────────────────────────────────
   dedupe across providers by preferred-origin id · keep magType · homogenize → Mw (TLS, ISC-GEM/GCMT anchor)
        → clean catalog (Parquet, native + Mw)                   [NEVER versioned]
        → clean manifest (dedupe stats, conversion coeffs + version)   [VERSIONED]
                                                  │
 (C) Mc + DECLUSTERING ────────────────────────────▼──────────────────────────────────────────────────────
   Mc(x,y,t) grid (MAXC / GFT / EMR) · cut < Mc · GK-declustered (background) · ZBZ η/T/R (features+labels)
        → Mc grid + declustered + ZBZ-labeled catalogs            [NEVER versioned]
        → mc_decluster manifest (Mc method, grid hash, b(t), decluster params)   [VERSIONED]
                                                  │
 (D) FEATURE BUILD ────────────────────────────────▼──────────────────────────────────────────────────────
   recent-window counts · ETAS intensities · η/T/R · slab/fault/strain joins · tidal dCFS / Mf envelope
        → feature store (Parquet on the forecast grid)            [NEVER versioned]
        → feature manifest (feature list, grid spec, enricher versions)   [VERSIONED]
                                                  │
 (E) TRAIN / FIT ──────────────────────────────────▼──────────────────────────────────────────────────────
   smoothed-seismicity Poisson (null) · ML space-time ETAS (primary + reference) · gated NTPP/covariates
        → model weights / fitted params                          [NEVER versioned]
        → model manifest (model id, fit window, params, CSEP scores, config hash)   [VERSIONED]
                                                  │
 (F) DAILY INFERENCE ──────────────────────────────▼──────────────────────────────────────────────────────
   forecast clock: hand the model ONLY the catalog slice (-inf, t) · cut < Mc · per-cell rate per horizon
   1d/2d/7d · expected + bounds (P10/median/P90) · calibrated public probability vs baseline
   gridded-rate (Poisson CSEP) + catalog-based (>=10k Monte-Carlo synthetic catalogs)
        → raw daily forecast object                              [NEVER versioned]
                                                  │
 (G) COMPACT ARTIFACT ─────────────────────────────▼──────────────────────────────────────────────────────
   sparsity floor + H3 binning + quantize + gzip → ONE artifact (few hundred KB – few MB)
   per-cell rates + baseline + bounds + calibration summary + provenance + coverage mask + timestamp
        → results/forecast-<region>-YYYY-MM-DD.json.gz           [VERSIONED — compact only]
        → results/index.json (latest pointer + rolling CSEP calibration)   [VERSIONED]
```

The CLI maps 1:1 onto these stages (`caos-seismic fetch / build-features / train / infer`, plus the
production `daily` job; see [`cli.py`](../src/caos_seismic/cli.py)).

**Three cross-cutting properties** make the back-analysis honest:

- **Forecast clock.** At each issue time $t$ the model is handed only the catalog slice
  $(-\infty, t)$, the forecast is sealed, then the clock advances. This makes temporal leakage
  *structurally impossible*, not a matter of discipline (`conditioning_slice` / `ForecastClock` in
  [`inference/clock.py`](../src/caos_seismic/inference/clock.py)).
- **Cold-start floor.** The conditional rate floors to the long-term smoothed-seismicity background
  (not a hard floor); the UI distinguishes "low but poorly-constrained," "genuinely quiescent," and
  "no data / out-of-coverage" (the `coverage_mask` — blank ≠ safe).
- **Input-state snapshots.** ComCat continuously revises magnitudes/locations and retracts events, so
  the fetch manifest snapshots the *exact* catalog state per issue (`snapshot_id()` content-hashes the
  conditioning catalog). A past forecast must be byte-reproducible months later; otherwise
  pseudo-prospective CSEP scoring is scored against a retroactively-improved catalog (optimistic
  leakage). Revisions are snapshotted, never silently overwritten.

---

## 4. What is versioned vs NEVER versioned

**VERSIONED (committed to git):**

- Pipeline code (`src/caos_seismic/`, `scripts/`, the static web app under `app/`).
- Configs (`configs/*.yaml`: region, grid/H3 resolution, $M_c$ method, declustering params, ETAS,
  horizons, magnitude thresholds, $M_{\max}$, publish allowlist).
- **Manifests** (`manifests/`) — the provenance / reproducibility record: source URLs, query params,
  retrieved-at timestamps, row counts, checksums, conversion coefficients, model params, CSEP scores,
  code SHA.
- **Compact daily results** (`results/forecast-<region>-YYYY-MM-DD.json.gz`) + `results/index.json`
  (latest pointer + rolling CSEP calibration). Small, and the static app's only data.
- `requirements.txt` / `pyproject.toml`, `.env.example`, README, license/credits.

**NEVER versioned (`.gitignore`, rebuildable from manifests + code):**

- Raw downloaded catalogs / waveforms / enricher grids (ComCat JSON, ISC-GEM CSV, GCMT `.ndk`, Slab2
  `.grd`, NGL `.tenv3`, InSAR rasters, TPXO model files).
- The clean / declustered / $M_c$-grid / feature Parquet stores.
- Model weights / fitted-parameter binaries.
- The `.venv/`, caches, the working `.env`, and all secrets.
- JMA / NIED Hi-net raw files and any agreement-gated derived products.

The git repo stays **small** — only configs, manifests, code, and compact gzipped results are
committed, growing by a few-hundred-KB-to-few-MB artifact per day. The working set on the build host
is ~1–5 GB for a focused region (mostly raw + features, all rebuildable). No GPU is needed for the
ETAS baseline; a single modest host runs the daily job (ETAS fitting is seconds-to-minutes of CPU).

---

## 5. Public-repo secret hygiene

This repo is **public**. The following is non-negotiable.

- **No secrets in the repo.** The spine needs no credentials; the only environment variable it reads is
  a polite contact `User-Agent` (`.env.example`), which is *not* a secret. Restricted-network tokens,
  AWS keys (use `--no-sign-request` for open buckets), and any API token are never committed; the real
  `.env` is gitignored.
- **No local machine paths** and no reference to any private vault, host, or internal infrastructure in
  any committed artifact. Public surfaces use only `https://github.com/fsantibanezleal/…` URLs.
- **License / attribution obligations are tracked** so the credits page is not forgotten: ISC-GEM
  **CC-BY-SA 3.0** share-alike (keep the license + provenance on any redistributed derivative);
  attribution-required sources **CSN-Chile, GeoNet / Earth Sciences NZ, EMSC, GCMT, NGL, Slab2/USGS,
  GEM faults**; TPXO academic-use terms. These are not redistribution blockers but the app **must**
  display the credits (`region.<id>.yaml: attribution`).
- **The product stands alone** — it cites only canonical literature and never positions against, tears
  down, or copies the implementation fingerprints of any specific third-party project.

---

## Reference index

Kagan 2017 (doi:10.1093/gji/ggx300) · Savran et al. 2020 (doi:10.1785/0120200026) · Wiemer & Wyss
2000 (doi:10.1785/0119990114) · Woessner & Wiemer 2005 (doi:10.1785/0120040007) · ISC-GEM v12.1
(doi:10.31905/d808b825) · Bird 2003 (doi:10.1029/2001GC000252) · Hayes et al. 2018 (Slab2,
doi:10.1126/science.aat4723) · Zaliapin & Ben-Zion 2020 (doi:10.1029/2018JB017120).
