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

## 2026-07-18 — WS1 #3 PNWML signal characterization (Claude, Opus 4.8)
- Local PNWML waveforms were incomplete (.part files); metadata CSVs complete.
  **Decision:** reconstruct labelled waveforms from FDSN using the metadata
  (station/time/channel) rather than depend on the multi-GB HDF5 — fully
  reproducible, reuses `data.waveforms.fetch_window`. Verified 15000 samp @ 100 Hz
  matches the PNWML 150 s window; 596/596 traces fetched successfully.
- Added pure, tested signal-feature functions to `features/characterize.py`
  (dominant freq, spectral centroid/bandwidth/flatness, envelope duration,
  band-energy ratios, kurtosis) + `scripts/characterize_pnwml.py`.
- Classes characterized: thunder, sonic boom, surface event, noise (comcat
  earthquake/explosion deferred — metadata not local).
- Result: thunder has the highest spectral centroid (~10 Hz) and uniquely peaks
  energy in 10-20 Hz; surface events are lower-frequency; noise is flat/long/
  low-kurtosis. Nearest confuser is sonic boom (also acoustic-coupled) →
  motivates the CNN + infrasound branch. Noted a ~22 Hz sensor/decimation notch
  on CC/BH channels (artifact, not source physics).
- Verification: 22 tests pass, ruff clean, figures + summary reviewed.

## 2026-07-19 — WS1 #3 cache + duration + seismo-acoustic (Claude, Opus 4.8)
- Added an on-disk waveform cache (`data/cache.py`, `.npz` per window + `.miss`
  markers) so figures/analyses never re-fetch from FDSN. Populated 596 traces.
- **Window question, answered from data:** individual claps ~few s; median *active*
  energy ~39 s; episode span median 89 s (often filling/exceeding the 150 s window).
  → recommend **50 s window @ 100 Hz, stride 25 s** (captures the discriminative
  burst; stride covers long tails). 150 s is unnecessary.
- **Caught a data trap:** PNWML `trace_P_arrival_sample`=7000 is a *placeholder*
  for thunder, not the true onset — the real energy sits ~30-40 s earlier. Reworked
  the duration metric to be onset-free (span above 3×robust-noise).
- **Seismo-acoustic delay** at CC.CPCO/KWBU/SVIC: 8/9 events show seismic vs
  infrasound envelopes correlated at ~0 s lag (xcorr 0.65-0.94, median 0.85) —
  strong coupling signature; feeds WS4 Model B. Constrained lag search to ±30 s to
  drop a spurious 104 s match. Small N (infrasound sparse at those event times).
- Dropped comcat earthquake/explosion (QuakeXNet already covers those).
- Enforced ruff on scripts/ too; 25 tests pass.

## 2026-07-19 — WS4 #8 seismic-only CNN baseline (Claude, Opus 4.8)
- Built the cache-backed dataset pipeline (`models/dataset.py`): 50 s @ 100 Hz
  window cropped around the envelope peak (onset-free) → log-spectrogram; classes
  thunder / sonic boom / surface event / noise (eq/explosion left to QuakeXNet).
- `models/cnn.py` (2D CNN, dropout for MC-dropout later) + `models/train.py`
  (station-disjoint GroupShuffleSplit, class-weighted loss, PR-AUC headline).
- First baseline (station-disjoint, ~600 traces): **thunder PR-AUC 0.61** (base
  rate 0.37). Confusion is exactly the WS1 prediction — thunder ↔ sonic boom and
  thunder ↔ surface event; noise clean. Clear mandate for Model B (infrasound).
- Ran under the `ml` pixi env (PyTorch). 28 tests pass; ruff clean.
- Baseline only: needs more data/augmentation, 3-C input, and the infrasound
  branch — tracked on #8/#9.

## 2026-07-21 — WS4 #8/#9/#11 CNN hardening + first OK generalization test (Claude, Opus 4.8)
- **BlurPool anti-aliasing** (Zhang 2019 MaxBlurPool: MaxPool stride-1 + binomial
  low-pass, stride-2) in every downsampling block — plain strided pooling aliases
  spectrogram structure.
- **Architecture exploration**: compared short-fat (w64/d3), long-skinny (w16/d5),
  and a w32/d3 baseline. Short-fat won on both PNW (PR-AUC 0.65) and OK-targeted
  (PR-AUC 0.73) training.
- **Augmentation** (`models/augment.py`): real-noise injection sampled from the
  TRAIN-split noise windows (no leakage), amplitude scaling, polarity flip, white-
  noise dithering — applied only after the station-disjoint split.
- **Region-aware classes** (`REGION_CLASSES`): OK model trains on {thunder, sonic
  boom, noise} — surface events are a real PNW/AK confuser but not an OK source
  type, so they're dropped rather than wasting model capacity / distorting the
  deployment-distribution match.
- **Waveform verification galleries** (`scripts/plot_waveforms.py`): per-class
  waveform+spectrogram grids for human review — generated for PNW (4 classes) and
  OK's class set (3 classes). Confirms thunder and sonic-boom are visually near-
  identical multi-clap bursts — the confusion is physically real, not a labelling bug.
- **OK-targeted model**: PR-AUC 0.73, thunder/noise separation now clean (0 cross-
  errors); remaining confusion is purely thunder<->sonic-boom (15/60 test events).
  IMPORTANT CAVEAT: PNWML has ZERO Oklahoma thunder examples (all 146 are PNW
  networks CC/UW/PB) — this model is trained entirely on PNW data, curated to OK's
  class mix. It has not yet seen a single real OK thunderquake.
- **First synchronous-lightning generalization test** (`scripts/detect_ok_thunder
  quakes.py`, issue #11): ran the OK model over continuous data at OK.CROK during
  the 2019-05-20 severe-weather outbreak, sliding 50s/25s-stride window, matched
  candidate detections against GOES-GLM strikes.
  - First attempt used radius=50km/window=90s and got a "100% match" — caught this
    as methodologically weak: with ~21,350 GLM flashes statewide in 90 min (1 every
    0.25s), almost any window matches something nearby by chance regardless of the
    seismic signal's validity.
  - Fixed: tightened to radius=15km (thunder's approximate audible/coupling range)
    and window=±20s, and added a NULL-RATE BASELINE (same test on a random sample
    of all windows, not just candidates) so the result is interpretable.
  - Honest result: candidate match rate 40% vs null baseline 20% (n=5 candidates —
    too small to claim significance, but directionally consistent with the model
    finding something real). This is a legitimate first pass, not a validated
    detector — next iteration needs more candidate events (longer time span,
    multiple stations) before the PNW->OK generalization claim can be trusted.
- Verification: 31 tests pass (dataset/augment/cnn all covered where feasible
  without GPU), ruff clean across src/tests/scripts.

## 2026-07-22 — WS4 #11 scaled-up OK generalization test (Claude, Opus 4.8)
- Scaled the single-station pilot to `scripts/detect_ok_thunderquakes_batch.py`:
  5 geographically-spread OK stations (CROK, FNO, NOKA, N4.T35B, RLOK), one
  6-hour window (2019-05-20 20:00-2019-05-21 02:00 UTC, the validated outbreak
  day), GLM strikes fetched ONCE and reused across all stations (station-
  independent). Pools candidate-vs-null match counts and runs a two-proportion
  z-test — the honest way to interpret a match rate against a catalog with
  ~1 flash every 0.23s statewide.
- **Found and fixed a real bug while scaling up:** the GLM S3 *listing* call
  had no retry (only individual granule reads did), so one transient connection
  reset during a ~1000+-granule fetch crashed the whole run. Added a shared
  retry-with-backoff helper in `lightning/glm.py` around both listing and reads.
- Extracted `two_proportion_ztest` into `evaluation/stats.py` (torch-free) so it
  is unit-testable from the default env; the batch script imports it.
- **Result (pooled, n=76 candidates across 3 active stations):** candidate match
  rate 54% vs null baseline 42%, z=1.46, p=0.072 — suggestive but short of
  conventional significance. Direction is consistent everywhere there were
  candidates (CROK, NOKA, N4.T35B all candidate-rate >= null-rate; no station
  went the other way), which is itself informative — a pure-noise result would
  scatter both directions. FNO and RLOK had ZERO candidates in 6 hours (plausible:
  the outbreak was concentrated further north/east; not yet independently
  verified via local strike density near those two stations — flagged as a
  follow-up, not asserted as fact).
- This is a legitimate, appropriately-hedged first quantitative generalization
  result, not a validated detector. Left issue #11 open with next steps: more
  days, verify the zero-candidate stations, use high-confidence matches as
  auto-label candidates for an actual OK-labeled training set.
- 34 tests pass; ruff clean.

## 2026-07-22 — WS4 #11 GLM-triggered extraction pipeline (Claude, Opus 4.8)
- User steered away from self-labelling (training on the model's own flagged
  windows risks reinforcing its existing thunder<->sonic-boom bias) toward
  **GLM-triggered extraction**: pull seismic windows at times/stations where an
  independent lightning strike landed very close, regardless of what the model
  predicts, then human-verify before trusting as a label.
- `scripts/survey_ok_close_strikes.py`: sized the opportunity FIRST, before
  building the pipeline. For the single already-validated 2019-05-20 6-hour
  storm window: 12,650 pairs within 5km, 48,151 within 10km, across up to
  198/282 permanent OK stations. Several strikes landed within 50-170m of a
  seismometer. This is model-independent (uses only GLM + station coordinates).
- `scripts/extract_ok_thunder_candidates.py`: clusters raw (station,strike) pairs
  into distinct time-gap-separated EVENTS (median 1 strike/event; a 90s gap
  threshold matches the WS1 episode-duration finding), filters out the ~3.5%
  tail with >100 strikes/cluster (a storm sitting continuously overhead is
  sustained noise, not a discrete thunderclap episode like the PNWML labels),
  fetches the real waveform independent of the model, and produces a
  waveform+spectrogram gallery for human QC.
- **Found and fixed a real methodological bug via the gallery**: window
  centering used a fixed quarter-window offset, but GLM (optical) and the
  coupled seismic arrival are NOT simultaneous -- the acoustic wave lags by
  distance/340 m/s (up to ~29s at the 10km radius used). Far events were
  landing near/past the window edge. Fixed by centering on the physically
  expected arrival time instead of a fixed offset.
- **Found a real data-availability limitation**: many small OK/GS local-network
  stations return HTTP 204 (no data) for specific candidate times even within
  their nominal deployment epoch -- consistent with the WS2 finding that
  metadata epochs overstate real archived data. Professionally-run TA/N4
  stations are far more reliable. Fetch success rate on a random candidate
  sample was ~15% for this reason (not a code bug -- verified directly against
  FDSN).
- **Honest QC read (n=15 fetched, human-reviewed)**: roughly 5-6 events show
  clean, discrete, broadband thunderclap-like bursts consistent with the WS1
  characterization (OK.CROK@2229m is an excellent multi-clap example); several
  are ambiguous/continuous, likely real misses (strike not audibly/seismically
  coupled at that distance/station) rather than pipeline errors.
- Sizing conclusion: **3,244 distinct candidate events from ONE storm** (10km
  radius, deduplicated) -- more than enough raw material to build a real
  OK-native training set once auto-scored/filtered further; this is a viable
  path, not just a one-off proof of concept.
- Corrected a repo-hygiene issue found in passing: the raw survey CSV (15MB,
  fully regenerable from GLM+station coords) belongs in `outputs/` (gitignored),
  not `catalogs/` (committed derived products) -- moved and fixed both scripts.
- 34 tests pass; ruff clean.

## Template
### YYYY-MM-DD — <topic> (<model>)
- Prompt/intent:
- Accepted:
- Rejected / modified:
- Verification performed:
