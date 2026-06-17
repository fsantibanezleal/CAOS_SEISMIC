# data/ — provenance only (the data itself is NEVER committed)

The actual catalogs, features, and grids under `data/raw/`, `data/clean/`, `data/mc/`, and
`data/features/` are **gitignored** and **rebuildable** from `configs/` + `manifests/` + code via
`scripts/fetch` and `scripts/build-features`. This file documents *where the data comes from* so the
corpus is reproducible without storing it.

| Source | What | Access | License | Attribution |
|---|---|---|---|---|
| USGS ComCat | global spine, real-time | FDSN `event` (`earthquake.usgs.gov/fdsnws/event/1/`) | public domain | USGS/ANSS |
| CSN (Chile) | regional driver (low Mc) | EarthScope/IRIS FDSN (net `C`/`C1`) or ISC | public, **attribution required** | Centro Sismológico Nacional, U. de Chile |
| ISC-GEM v12.1 | Mw-homogenized 1904–2021 | `isc.ac.uk/iscgem/` | **CC-BY-SA 3.0** | ISC-GEM |
| Global CMT | moment tensors M≥5 | `globalcmt.org` (`.ndk`) | research + citation | GCMT Project |
| EMSC | dedup cross-check | `seismicportal.eu/fdsnws/event/1/query` | research + attribution | EMSC-CSEM |
| Slab2 | subduction geometry | ScienceBase / `github.com/usgs/slab2` | USGS public domain | Hayes et al. 2018 |
| GEM Active Faults | fault geometry | `github.com/GEMScienceTools/gem-global-active-faults` | CC-BY-SA 4.0 | GEM |
| Bird PB2002 | plate boundaries | `peterbird.name/publications/2003_pb2002/` | open research | Bird 2003 |
| NGL GNSS | crustal strain | `geodesy.unr.edu` | open + attribution | Nevada Geodetic Lab |
| Tidal stress | computed feature | `pygtide` (+ SPOTL/TPXO ocean loading) | open tools | — |

**Never** place restricted, registration-gated data (e.g. JMA / NIED Hi-net) or any credential in this
public repo. ISC-GEM is share-alike: a redistributed derived catalog must keep CC-BY-SA + provenance.
