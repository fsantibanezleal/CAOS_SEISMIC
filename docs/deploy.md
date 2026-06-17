<!-- markdownlint-disable MD013 MD033 -->
# Deployment

> **Heavy compute runs once per day, offline; the runtime is stateless and read-only.** There is no
> processing backend. A scheduled job fits the model, writes one compact artifact, commits it to this
> repo, and a static web app renders the committed JSON. This is the cheapest, most reproducible, and
> most honest deployment for a daily forecast: cost scales with *bytes served*, not compute, and every
> published number is a versioned git object anyone can audit.

The architecture has three boxes — **compute → git → web** — rendered in
[`diagrams/architecture.svg`](diagrams/architecture.svg):

```
  Local GPU workstation (scheduled daily ~03:00)          Public repo (git-as-data)        Static web
  ──────────────────────────────────────────────          ─────────────────────────        ──────────
  fetch (ComCat + CSN delta) → hygiene (Mc, Mw,            results/forecast-                 GitHub Pages /
  decluster) → condition/fit ETAS (+ GPU challenger)  ──▶  <region>-YYYY-MM-DD.json.gz  ──▶  Netlify SPA
  → simulate ensemble → calibrate → QA-gate →              results/index.json                reads the
  ONE compact artifact → scoped `git add` → commit/push    (+ manifests, CSEP)               committed JSON
```

---

## 1. Compute — a local GPU workstation

A single modest host is sufficient because **the heavy compute is offline and the ETAS fit is
seconds-to-minutes of CPU, no GPU** (INLAbru-class daily fits land well under a minute for hundreds of
events). The GPU is there only to accelerate two optional things:

- the **gated neural challenger** (when/if one is trained — it produces frozen weights occasionally,
  and even then the daily job only runs *forward inference*), and
- **large Monte-Carlo ensembles** ($\ge 10{,}000$ synthetic catalogs/day; `forecast.yaml:
  ensemble.n_synthetic_catalogs`).

Requirements: Python 3.12, the core deps (numpy / pandas / scipy / requests / pyyaml / pydantic / h3 /
scikit-learn) plus the science extra (`obspy`, `pycsep`, …) installed into a repo-local `.venv` by
`scripts/setup`. The daily job needs outbound HTTPS to the FDSN services and push access to this repo.
The package is importable on the **core deps alone**; every heavy dependency (obspy, pycsep,
geopandas, pygtide) is imported *lazily* inside the stage that needs it, with a clear error if missing
— so a partial environment degrades to an actionable message, never an import-time crash.

---

## 2. The daily job

The production job is `caos-seismic daily` (CLI in [`cli.py`](../src/caos_seismic/cli.py); wrapped by
`scripts/daily.{ps1,sh}`). It runs **fetch → infer → scoped publish**:

1. **Determine the issue dates.** Today (UTC) plus — when `publish.yaml: schedule.catch_up_missed` is
   on — any days in the last week with no committed artifact yet. The catch-up is *bounded to a week*
   so a long-dormant laptop never tries to backfill months (those would not be honest
   pseudo-prospective forecasts: the catalog has since been revised).
2. **Fetch once.** A ComCat `updatedafter` incremental delta plus the regional network, merged into the
   raw store; the freshest catalog covers every issue date in the batch.
3. **Infer for each issue date** (oldest first, so `index.json` ends current). The forecast clock hands
   the model only the catalog slice strictly before the issue time, simulates the ensemble for
   {1d, 2d, 7d} × {P10, median, P90}, calibrates, runs the rolling CSEP consistency checks, and writes
   one compact gzipped artifact under `results/`.
4. **Scoped publish** (unless `--no-publish`) — commit and push (§3).

A local dry run is `caos-seismic daily --no-publish` (or `scripts/infer` for a single issue date).

### 2.1 Operational QA gate

A single bad / duplicated / retracted event near $M^*$ can swing a public probability, so the job
**must not auto-publish a stale or anomalous artifact** (`forecast.yaml: qa_gate`). It gates on:
input-catalog sanity (no event-count z-score above `max_event_count_zscore`, no duplicate/retracted
event near the threshold via `forbid_duplicate_near_threshold`), a rolling-window **N-test drift
monitor on the forecaster itself** (`ntest_drift_window_days`) as an early warning of model/catalog
breakage, and artifact integrity. On an FDSN outage / rate-limit or a failed run, the product serves
**"unavailable" with a staleness banner — never silently a stale or corrupted artifact.** The artifact
carries a `staleness: {generated, next_run, ok}` block (see
[`contracts.py: Staleness`](../src/caos_seismic/contracts.py)); when `ok` is false the SPA degrades
visibly (banner + desaturation/hatch). All FDSN access is wrapped in retry/backoff (HTTP 413 is treated
as "tile smaller").

---

## 3. Publish — git-as-data, scoped commits only

The artifact is committed to this repo as the single source of the web app's data. The commit is
**strictly scoped** — this is the load-bearing safety rule, because the build host also has `data/`,
`models/`, `.venv/`, and a working `.env` present, and a `git add -A` would leak raw data or secrets.

`publish.yaml: git` defines an explicit `add_allowlist` (`results/`, `manifests/`) and the scoped
publish in [`cli.py`](../src/caos_seismic/cli.py) (`_publish_scoped`):

- resets the index, stages **only** the allowlist paths (never `git add .` / `-A`),
- verifies nothing outside the allowlist got staged and **aborts** (resetting the index) if anything
  did,
- commits with the configured `commit_message_prefix` and pushes `HEAD:main` to `origin`.

`caos-seismic check` refuses any allowlist entry that is `.`, `-A`, `--all`, or `*`, so a non-scoped
publish cannot be configured by accident. A **pre-push hook** hard-fails on any out-of-allowlist path
as defence in depth. The push credential is a dedicated **least-privilege deploy key / fine-grained
PAT scoped to this repo only**, held in the local git credential store — **never committed**.

Content updates once per day as one small commit (a few hundred KB to a few MB). Over years this grows
the repo by tens to low-hundreds of MB — modest, because *only* the compact gzipped results and the
manifests are versioned; raw data, features, and weights never are (see
[`data-and-pipelines.md`](data-and-pipelines.md) §4).

---

## 4. The daily schedule (~03:00)

The job fires **daily, early morning (~03:00 local)** — `publish.yaml: schedule.time_local = "03:00"`,
`cadence: daily`. Pick the platform-native scheduler:

- **Windows (the GPU workstation):** a Task Scheduler task running `scripts\daily.ps1` daily at 03:00,
  with "run task as soon as possible after a scheduled start is missed" enabled (the laptop may
  sleep/reboot — `catch_up_missed` handles the backfill on next wake).
- **Linux host / VPS:** a `cron` entry or a `systemd` timer:

  ```cron
  # /etc/cron.d/caos-seismic  — daily forecast at 03:00 local
  0 3 * * *  caos  cd /opt/CAOS_SEISMIC && scripts/daily.sh >> var/daily.log 2>&1
  ```

  or, equivalently, a `systemd` `OnCalendar=*-*-* 03:00:00` timer wrapping `scripts/daily.sh`.

A **full re-fit / re-training** (including any GPU challenger) runs on a slower cadence or when a large
event occurs — `publish.yaml: train_cadence` (`full_refit: weekly`, `event_triggered_magnitude: 6.5`).
The daily job only conditions on / forward-infers from the latest fit; it does not re-train every day.

---

## 5. Web — a pure static viewer

The web app under `app/` is a **Vite + React + TypeScript SPA** (i18n EN→ES, light/dark, the
dark-technical palette) that renders the committed artifact. **No server computes anything** — the
"API" is static JSON assets:

- the client fetches `results/index.json` first (latest pointer + rolling CSEP calibration + a
  staleness mirror so the banner can render before the artifact loads),
- then the chosen `forecast-<region>-YYYY-MM-DD.json(.gz)`, decoding the sparse, H3-keyed forecast tree
  (`forecast[cell][horizon][threshold] → {p, lo, hi, rate, baseline}`),
- and switches horizon / bound / region among arrays already present in the single artifact (no extra
  round-trips). The no-map summary renders from the same artifact's baseline/calibration fields with
  zero map-library cost.

The TypeScript types in [`app/src/data/types.ts`](../app/src/data/types.ts) mirror the Python
`ForecastArtifact` byte-for-byte (same field names, the `expected → p` rename aside), so the contract
cannot drift. Host the built SPA on any static host — **GitHub Pages** (served straight from this repo)
or **Netlify** — fronted by a CDN cache. Because every request reads a precomputed artifact, the
runtime is stateless and read-only, and the deployment has no database, no secrets, and no compute to
scale.

---

## 6. Reproducibility and audit

Every daily run is byte-reproducible from `manifests/` + `configs/` + code: the manifest pins the
config hash, the code git SHA, the immutable input-catalog snapshot id, the $M_c$ grid version, the
declustering choice, and the model + params (`build_manifest()` in
[`inference/provenance.py`](../src/caos_seismic/inference/provenance.py)). A reviewer can check out the
recorded SHA, re-run the pipeline, and reproduce the committed artifact — or audit a months-old
forecast against the catalog *as it was at issue time*. That auditability is the reason the deployment
is git-as-data rather than an opaque service.
