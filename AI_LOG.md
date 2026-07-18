# AI-assisted development log

Per Denolle-group norms, we record AI usage and the decisions it informed. Keep
entries short: what was asked, what was accepted/rejected, and why.

## 2026-07-17 — Repo restructure & roadmap (Claude, Opus 4.8)
- **Audited** the prototype `notebooks/01_thunderquakes_in_the_pnw.ipynb`. Key findings:
  the ASOS co-location approach is infeasible (airport-centric) — *retired*; the
  `snr>=3` "detection" label is circular — *retired*; no infrasound / no ML / no
  Oklahoma coverage yet.
- **Restructured** the notebook prototype into a standard Python package
  (`src/thunderquakes/`) with pixi + conda environments. Original notebook preserved
  under `notebooks/archive/` for provenance.
- **Decisions locked (by MD):** open-only lightning catalogs (GLM/ALDN/WWLLN);
  infrasound is an optional second branch (seismic-only must stand alone).
- Roadmap encoded as GitHub issues (WS0–WS5). See ROADMAP.md.

## 2026-07-18 — WS2 #4 Oklahoma station inventory (Claude, Opus 4.8)
- Ran `build_inventory("OK")`; found 282 permanent seismic stations, 9 seismoacoustic.
- **Fixed a real bug:** initial `is_infrasound` conflated acoustic-rate mics (`BDF`/`BDO`)
  with 1-sps meteorological pressure (`LDF`/`LDM`/`LDO`). Now classified by SEED
  band+instrument code (`is_acoustic_channel`).
- Added the 2016 Oklahoma nodal experiments from the **IRIS PH5 archive** (not in
  standard fdsnws): `2A` = LASSO (~1829 nodes, no infrasound), `YW` = Community
  Wavefield (adds 9 `HDF` seismoacoustic sites near Enid, Jun–Nov 2016). Network
  codes discovered by querying IRISPH5, not guessed.
- Added actual archived-data availability via the fdsnws-availability *extent*
  service (`data_start`/`data_end`/`n_timespans` gappiness proxy) — reveals
  metadata epochs overstate real data (e.g. GS.ADOK meta 2010 vs data 2013).
- Result: 18 seismoacoustic stations total. Verification: 13 tests pass, ruff clean,
  outputs regenerated and inspected (map + CSVs).

## 2026-07-18 — WS3 #6 GLM loader + WWLLN loader (Claude, Opus 4.8)
- Implemented `lightning.load_glm_strikes`: lists GOES-GLM L2 granules on AWS S3
  (anon), reads flash centroids via xarray/h5netcdf, filters to bbox, returns the
  common schema. Threaded granule reads (`max_workers`) for practical speed.
  GLM East satellite auto-picked by date (goes16 → goes19 at 2025-04-04).
  Verified on real data: 5,120 OK flashes in 20 min during the 2019-05-20 outbreak.
- Added `h5netcdf` dep (read netCDF from S3 file objects).
- Implemented `lightning.load_wwlln_strikes` / `parse_wwlln_file` against the WWLLN
  located-stroke (A-file) format. **Access note:** WWLLN needs a data agreement but
  is operated by UW (Holzworth/ESS) — internal path available. Loader is ready for
  the files (esp. Alaska / pre-2018 where GLM fails).
- GLM/WWLLN report energy, not peak current → `peak_current` left NaN by design.
- `scripts/fetch_glm.py` caches a region+window catalog to outputs/ (CSV).
- Verification: 18 tests pass (no-network parse/filter tests), ruff clean.

## Template
### YYYY-MM-DD — <topic> (<model>)
- Prompt/intent:
- Accepted:
- Rejected / modified:
- Verification performed:
