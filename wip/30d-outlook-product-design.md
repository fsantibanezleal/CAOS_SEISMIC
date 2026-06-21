# 30-day outlook product — design (geodetic background, cached-weekly)

**Status:** design (2026-06-18). Gated on **E14** (the neural-30d multi-region validation, running) confirming
the 30-day geodetic win is robust enough to ship. Grounded in E10/E11.

## Why this is buildable despite the 30-min neural fit

The naive objection: the geodetic neural costs ~30 min to fit, so it cannot run in the daily pipeline. But
**E11 proved the calibrated neural is a *time-flat geodetic BACKGROUND***, not a triggering model: its total
forecast is constant across windows (18.1 @ 7d, 77.3 @ 30d, ratio = 30/7), and the GNSS strain field that
drives it changes on a **multi-week/seasonal** timescale, not daily. So the neural background does **not**
need daily refitting — it needs refitting only when the strain field (or the catalog's background
structure) has meaningfully moved.

## Architecture (cached-weekly background + daily serve)

1. **Weekly (or monthly) neural refit** — a separate scheduled job (NOT the 03:00 daily) fits the
   strain-conditioned neural and writes a **cached 30-day background field artifact**: per-cell expected
   `M≥mc` counts at the 30-day horizon (a sparse/H3/quantized grid like the daily artifact). ~30 min, once
   per week. Tagged with its fit date + the strain-data vintage.
2. **Daily serve** — the daily job (already fixed + self-publishing) reads the cached 30-day background
   field and publishes it as the **30-day outlook** artifact. Because the background is time-flat, the
   cached field IS the outlook; optionally the daily ETAS 30-day *triggering* is added on top (the hybrid
   `λ = μ_neural + ETAS_triggering`) for the cells near recent large events — a small, cheap correction.
3. **Web** — a new **"30-day outlook"** view (its own route/tab) rendering the geodetic-informed 30-day
   field, clearly labelled as a *longer-horizon outlook* distinct from the 1-7 day operational forecast,
   with the honest framing: "the geodetic context measurably improves the 30-day background where the 1-7
   day triggering forecast cannot use it (E11: +0.078 nats/eq vs ETAS at 30 d)."

## Honest framing (mandatory)

- The 1-7 day product stays **ETAS** (the geodetic context does NOT help there — E11: −0.053 at 7 d). No
  regression, no overclaim.
- The 30-day outlook is where the one **measured** geodetic advantage lives. Ship it there, gated to E14's
  multi-region verdict. If E14 shows the 30-day win is global-only (not regionally robust), ship it as a
  GLOBAL outlook and say so; if regionally robust, enable the regional views.
- The strain channel is real (`gnss_strain_rate`); the other geophysical channels remain honestly
  zero-filled until wired.

## Build steps (when E14 clears the gate)

1. `inference/outlook.py` — fit-and-cache the 30-day neural background field artifact (reuse the neural fit
   + `expected_counts` at H=30; write a compact artifact + an `outlook-index.json`).
2. A `caos-seismic outlook` CLI command + a weekly schedule (Task Scheduler / cron) — separate from the
   daily job; publishes via the same robust `_publish_push_to_branch` (commit-tree to main).
3. Daily job optionally grafts ETAS 30-day triggering onto the cached background (the hybrid).
4. Web: a `/outlook` route + a deck.gl 30-day field view; nav entry; i18n; the honest framing copy.
5. Back-analysis: extend the published evidence with E14's per-view 30-day IGPE so the outlook's skill
   claim is backed by the live numbers, not asserted.

## What this does NOT do

It does not claim a 1-7 day improvement (there is none — the ETAS ceiling stands). It productizes the
single real, measured, validated win at the horizon where it exists. That is the honest deliverable.
