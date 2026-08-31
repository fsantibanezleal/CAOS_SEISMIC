<!-- markdownlint-disable MD013 -->
# CAOS_SEISMIC — Conditional Probabilistic Seismic Forecasting

[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_SEISMIC)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live-2ea44f)](https://seismic.fasl-work.com)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21508362-blue)](https://doi.org/10.5281/zenodo.21508362)

> **Earthquakes cannot be predicted, but their probability can be forecast — reported honestly, with
> uncertainty, evaluated against reality, never as an alarm and never as a promise of safety.**

CAOS_SEISMIC ingests decades of seismic (and complementary geophysical) data and produces **bounded,
calibrated, conditional probability forecasts** of seismic events over short horizons — **1 day, 2 days,
7 days** — for a region and magnitude band. It runs **one inference per day** and publishes a compact
forecast artifact that a static web app visualizes.

This is **Operational Earthquake Forecasting (OEF)**, the mainstream scientific framing — *not*
earthquake prediction. The output is a *conditional estimator*: given the recent state of seismicity, it
bounds the probability of events in the near term. Every number is a probability strictly in (0, 1),
scoped to a region × magnitude band × horizon, shown **next to its long-term baseline**, with an
uncertainty band, and **scored prospectively** against what actually happens.

⚠️ **Not an alarm. Not life-safety.** This is an independent research/education tool that *complements*
official agencies (in Chile, the **Centro Sismológico Nacional, CSN**; civil protection: **SENAPRED**).
It never issues an alarm, a countdown, or a "safe" state. See [The honest limits](#the-honest-limits).

- **Project page:** https://github.com/fsantibanezleal/CAOS_SEISMIC
- **Author:** Felipe Santibáñez-Leal — https://fsantibanezleal.github.io

---

## What makes a forecast honest here

1. **A defensible baseline.** The core is a maximum-likelihood **space–time ETAS** model (Ogata 1998) —
   the de-facto operational benchmark — plus the mandatory **stationary smoothed-seismicity Poisson
   null** that any time-dependent model must beat. A transparent **Reasenberg–Jones** aftershock model
   is the sanity-check fallback.
2. **Skill is *proven*, not asserted.** Every model is scored with the community-standard **CSEP**
   framework via **pyCSEP**: consistency tests (N / M / S / L / conditional-L) and comparison tests
   (paired-t / Wilcoxon on **information gain per earthquake**, in *nats*) against **both** the Poisson
   null **and** ETAS. ROC/AUC is **banned** as a skill metric (it is calibration-blind). A live
   **reliability diagram** ("when we said 5 %, it happened ~5 % of the time") is the headline artifact.
3. **Calibration is a release blocker.** An uncalibrated probability does not ship.
4. **Leakage is structurally impossible.** A strict **forecast clock** hands the model only the catalog
   slice up to issue time; every forecast is logged immutably with the exact input state, so any past
   forecast is byte-reproducible and scored against the catalog *as it was at issue time*.
5. **A stronger model must earn the map.** A GPU-trained neural temporal point process (a conditional,
   Hawkes-biased model) is a gated challenger: it reaches the public map **only** if it beats ETAS in our
   own prospective CSEP harness **and** is calibrated. Until then, ETAS is what ships.

## The honest limits

- **No deterministic prediction.** Whether a small rupture cascades into a great earthquake depends on
  unmeasurably fine details of the crust (Geller et al., 1997). Deterministic prediction is effectively
  impossible; the physical picture is self-organized criticality (a leading framework, not settled
  physics).
- **Absolute probabilities stay small.** Even during an active sequence, the absolute probability of a
  large event in the next day is usually well under a few percent. The *relative* gain over background can
  be large (1–3 orders of magnitude); the *absolute* number stays low — so we always show it next to the
  baseline.
- **A single outcome neither validates nor invalidates a probabilistic forecast.** During the 2019
  Ridgecrest sequence an operational model gave ≈3 % chance of a larger event in the first week; the
  M 7.1 struck ~34 h later. A 3 % forecast is **not wrong** when the 3 % outcome occurs.
- **The dangerous failure mode is communication.** The lesson of L'Aquila (2009) is that *false
  reassurance* causes harm. This product never over-reassures and never issues an alarm.

## Architecture

Heavy compute is **offline**; the web app is a **pure static viewer with no processing backend**.

```
  Local GPU workstation (scheduled daily ~03:00)         Public repo (git-as-data)      Static web
  ──────────────────────────────────────────────         ────────────────────────       ──────────
  fetch (ComCat + CSN delta) → hygiene (Mc, Mw,           results/forecast-              GitHub Pages /
  decluster) → condition/fit ETAS (+ GPU challenger)  ──▶ YYYY-MM-DD.json.gz     ──────▶ Netlify SPA
  → simulate ensemble → calibrate → ONE compact           results/index.json             reads the
  artifact → scoped `git add results/` → commit → push    (+ manifests, CSEP)            committed JSON
```

- **Compute:** a local always-on GPU workstation runs the daily job (a Windows Task Scheduler task /
  `cron`, ~03:00 local). ETAS fitting is CPU seconds-to-minutes; the GPU accelerates the challenger and
  large Monte-Carlo ensembles.
- **Publish:** the job commits the compact artifact (a few hundred KB to a few MB) to this repo. Content
  updates once per day as one small commit. The commit is **scoped** to `results/` + `manifests/`.
- **Web:** a Vite + React + TypeScript SPA (i18n EN→ES, light/dark, dark-technical palette) renders the
  committed artifact. **No server computes anything** — the "API" is static JSON assets.

## Repository layout

```
CAOS_SEISMIC/
├── src/caos_seismic/        # the forecasting library (real implementation)
│   ├── config.py            # typed loading of configs/*.yaml
│   ├── contracts.py         # the public interfaces + the artifact/manifest schemas (the seams)
│   ├── data/                # fetch (ComCat/CSN/ISC-GEM/GCMT/enrichers) + clean/homogenize
│   ├── catalog/             # completeness (Mc, b-value) + declustering (GK, Zaliapin–Ben-Zion)
│   ├── model/               # ETAS, Reasenberg–Jones, smoothed-seismicity, the Forecaster port
│   ├── inference/           # forecast clock, daily run, ensemble, compact artifact writer
│   ├── eval/                # pyCSEP wrappers (consistency + comparison tests, reliability)
│   └── cli.py               # entry points used by scripts/
├── configs/                 # region (Chile), grid, completeness, declustering, ETAS, horizons
├── scripts/                 # parallel *.ps1 + *.sh: setup, fetch, build-features, train, infer, daily, dev, check
├── app/                     # the static web app (Vite + React + TS)
├── docs/                    # deep technical docs: methodology (equations), model, data, evaluation, web
├── manifests/               # provenance (VERSIONED): what was fetched, Mc grid, decluster, model, params
├── results/                 # compact daily artifacts (VERSIONED): forecast-*.json.gz + index.json
├── data/  models/           # raw data, features, weights — NEVER versioned (.gitignore; rebuildable)
├── requirements.txt  pyproject.toml  .env.example  .gitignore  LICENSE
```

**Versioned:** code, configs, manifests, the compact daily results.
**Never versioned:** raw catalogs, processed features, model weights — rebuildable from manifests + code.

## Data sources & attribution

Catalog spine: **USGS ComCat** (FDSN event service; public domain). Regional driver (short-horizon
skill): **CSN — Centro Sismológico Nacional, Chile** (attribution required). Long-term homogeneous
anchor: **ISC-GEM** (CC-BY-SA 3.0). Mechanisms: **Global CMT**. Cross-check: **EMSC**. Enrichers:
**Slab2** (USGS), **GEM Global Active Faults**, **Bird (2003) PB2002** plate model, **Nevada Geodetic
Lab** GNSS, and a physically-motivated **tidal** stress covariate. Each source's license and required
attribution are tracked in [`docs/data-and-pipelines.md`](docs/data-and-pipelines.md) and surfaced on the
web app's credits page.

## Quick start

```bash
# 1) Environment (Python 3.12 recommended)
scripts/setup.sh            # or scripts\setup.ps1 on Windows  → creates .venv, installs requirements

# 2) Fetch a region's catalog (Chile by default), build features, fit, infer
scripts/fetch.sh            # ComCat + CSN → raw store + provenance manifest
scripts/build-features.sh   # Mc + magnitude homogenization + declustering + features
scripts/train.sh            # fit smoothed-seismicity null + space–time ETAS; run CSEP tests
scripts/infer.sh            # daily forecast clock → compact artifact in results/

# 3) Preview the web app locally (static)
cd app && npm install && npm run dev
```

The daily production job is `scripts/daily.{ps1,sh}` (scheduled ~03:00): `fetch → infer → scoped commit
→ push`. See [`docs/deploy.md`](docs/deploy.md).

## Preprint

The methodology and the prospective results are written up as a preprint (CC-BY-4.0):
**"Horizon-Dependent Value of Geodetic Context in Operational Earthquake Forecasting:
A Leakage-Free, Multi-Region Study with Pre-Registered Negative Results"**,
concept DOI [10.5281/zenodo.21508362](https://doi.org/10.5281/zenodo.21508362)
(source in [`manuscripts/oef-geodetic-horizon/`](manuscripts/oef-geodetic-horizon/)).
Every number in it is backed by a committed artifact in `results/`.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — the models and their equations (Gutenberg–Richter,
  Omori–Utsu, ETAS, Reasenberg–Jones, smoothed seismicity, …) and the CSEP evaluation framework.
- [`docs/model.md`](docs/model.md) — the conditional estimator, target definition, features, calibration.
- [`docs/data-and-pipelines.md`](docs/data-and-pipelines.md) — sources, licenses, the pipeline DAG.
- [`docs/evaluation.md`](docs/evaluation.md) — the back-analysis protocol and how skill is established.
- [`docs/deploy.md`](docs/deploy.md) — the local-compute + git-as-data + static-web deployment.

## Disclaimer

CAOS_SEISMIC provides probabilistic *forecasts*, not *predictions*, for research and educational
purposes. It is provided "as is", without warranty, and **must not be used for life-safety or
emergency decisions**. It does not represent or speak for any official agency. Always follow your
national seismological and civil-protection authorities.

## License

[MIT](LICENSE) © 2026 Felipe Santibáñez-Leal.
