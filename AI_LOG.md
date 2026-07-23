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

## 2026-07-22 (cont.) — WS4 #9/#11 how much can infrasound help verify OK candidates (Claude, Opus 4.8)
- Promoted `seismo_acoustic_lag`/`smooth_env`/`resample_to` out of
  `scripts/analyze_thunder.py` into `features/seismoacoustic.py` (shared, unit-
  tested: synthetic known-shift + uncorrelated-noise cases) so both the WS1
  PNW analysis and this new OK check use the same code.
- `scripts/verify_seismoacoustic_candidates.py`: of the 3,244 GLM-triggered OK
  candidate events (#11), only **59 (1.8%) sit at a permanent OK seismoacoustic
  station** (T35B/TUL1/TUL3 — the WS2 inventory found just 9 such stations
  statewide). Fetched real seismic+infrasound pairs for 47/59 and computed
  envelope cross-correlation.
- **Result: median xcorr 0.35, only 6/47 (13%) exceed the "strong coupling"
  threshold (>=0.5)** — much lower than the WS1 PNW baseline (median 0.85).
  This is NOT an apples-to-apples comparison though: the PNW number came from
  PNWML's analyst-CURATED thunder labels (only well-recorded events made the
  cut), whereas this is a raw, unfiltered proximity population that necessarily
  includes real misses (strikes that weren't actually audible/coupled at that
  station/distance). A weak-but-present negative correlation between xcorr and
  distance (r ~ -0.17 to -0.20) is physically sensible and confirms the metric
  is capturing real signal, not noise — visually confirmed too: the top-ranked
  panels in the gallery show genuinely co-varying seismic/infrasound envelope
  shapes, not spurious matches.
- **Answer to "how much can infrasound help build the dataset":** infrasound
  helps QUALITY, not COVERAGE. Only 1.8% of candidates are even eligible (sparse
  OK infrasound infrastructure), but among those, coupling identifies a smaller,
  physically-verified high-confidence subset (13%) suitable as gold-standard
  labels / Model B ablation data. The bulk of the OK training set will have to
  rely on seismic-only verification (human QC, feature-based auto-scoring) —
  infrasound cannot scale to most of the state given current station coverage.
- 38 tests pass; ruff clean.

## 2026-07-22 (cont.) — WS5 progress report + axis-label audit (Claude, Opus 4.8)
- Wrote a full progress report (`report/manuscript.qmd`, Quarto) covering intro,
  data, methods, results/QC, and follow-up work across WS0-WS4, embedding all
  15 figures generated this session. Rendered successfully (`quarto render`);
  every image reference verified to resolve to a real file on disk.
- Mid-task: user asked that every plot have x/y axis labels with physical units.
  Audited every plotting script and fixed gaps: station maps lacked degree units
  ("Longitude"->"Longitude (°E)"); PNWML distribution boxplots used raw variable
  names as labels instead of units ("kurtosis"->"Kurtosis (dimensionless)",
  "duration_80pct_s"->"Duration containing 80% of energy (s)"); waveform/
  spectrogram galleries (plot_waveforms.py, extract_ok_thunder_candidates.py,
  verify_seismoacoustic_candidates.py) had titles but no axis labels at all on
  the actual waveform/spectrogram subplots -- added "Time (s)" / "Amplitude
  (counts)" / "Frequency (Hz)" throughout; detection scripts' P(thunder) and
  match-rate axes clarified as explicit probabilities/fractions (0-1).
- Regenerated all 15 figures. Cache-backed ones (PNWML characterization,
  galleries) reproduced bit-for-bit identical numbers instantly. GLM-dependent
  ones (close-strike map, single-station and batch detection) were refetched
  and reproduced identical statistics (5,120/76 candidates, 40%/20%, 54%/42%
  z=1.46 p=0.072, etc.) -- confirms the whole pipeline is deterministic.
- **Caught and fixed a near-miss**: an attempt to regenerate the station maps
  via `build_inventory.py --no-availability` (to skip the slow fdsnws-
  availability query) actually re-ran the full FDSN station query and
  overwrote `catalogs/stations_OK_summary.csv` / `stations_OK_infrasound.csv`,
  stripping the committed `data_start`/`data_end`/`n_timespans` availability
  columns before other regenerations could run. Caught via `git status` before
  committing, reverted with `git checkout --`, and redid the map regeneration
  the safe way: call `plot_map()` directly on the existing committed CSV, with
  no FDSN query at all. A reminder that regenerating a committed derived
  artifact should default to the narrowest operation that reproduces it, not
  the full pipeline.
- 38 tests pass; ruff clean.

## 2026-07-22 (cont.) — Map quality, gallery whitespace, moveout check (Claude, Opus 4.8)
- Added `thunderquakes.plotting`: shared print-style rcParams (>=10pt floor) and
  `gray_relief_background()` -- PyGMT-rendered pure hillshade (not elevation-
  colored) through a manual near-white 2-stop CPT so texture never competes with
  data markers, plus coastline/state-border overlay (`fig.coast`), on a linear
  (Cartesian) PyGMT projection so it aligns pixel-for-pixel with matplotlib's
  plain lat/lon `imshow` (a Mercator projection would misalign at AK's latitude
  span). PyGMT added as a pixi dependency; installed cleanly since GMT itself
  was already present on the machine.
- Map fixes across `build_inventory.py` / `survey_ok_close_strikes.py`, all from
  user feedback on the rendered maps:
  - **Equal physical aspect** via `ax.set_aspect(1/cos(mean_lat))` -- naive
    equal-degree axes visibly squeezed Alaska (20 deg of latitude vs 40 deg of
    longitude at high latitude, where a longitude degree is much shorter).
  - **Legend moved below the plot** (was overlapping/hiding markers in the
    upper-right corner).
  - **"since <year>" annotation**: `first_year_at_least()` finds the first
    calendar year >=10 stations of a category were simultaneously active, using
    real archived-data extent (data_start/data_end) over nominal metadata epoch
    -- unit-tested.
  - **Padding beyond the nominal bbox** to the union with actual station
    coordinates, so markers at/near the region boundary are never half-clipped.
  - **Close-strike survey map now also overlays station markers** (triangle
    seismic / star seismoacoustic, same color scale as strike count) on user
    request, not just the strike-count heatmap.
- **Removed all on-figure titles/suptitles** across every plotting script (user
  request): descriptive framing moved to Quarto figure captions + `fig-alt=`
  instead, avoiding the overlap/clipping several titles were causing at print
  width. Per-panel identifying labels in galleries (station/time/xcorr) were
  kept since they're necessary data labels, not decorative headlines.
- **Gallery whitespace fix**: `plot_waveforms.py` / `extract_ok_thunder_
  candidates.py` / `verify_seismoacoustic_candidates.py` now use `sharex="col"`
  (all panels share the same fixed time window) and link spectrogram y-axes via
  `ax.sharey()` (same frequency band everywhere), so tick labels/axis labels are
  only drawn once per row/column instead of repeated on every panel. Abbreviated
  "Frequency"->"Freq", "Amplitude"->"Amp" per user request. Fixed one real
  overlap bug this surfaced: long per-panel titles bled into the neighbouring
  column at 2-column width; reordered/shortened the title text.
- **New: seismoacoustic moveout check** (`scripts/plot_seismoacoustic_moveout.py`,
  prompted by user asking for concrete propagation verification). Computes the
  predicted acoustic travel time (distance/340 m/s) for every OK candidate with
  a fetched seismic+infrasound pair and compares it to the observed envelope
  lag, plus two raw-waveform (not envelope) record-section examples time-aligned
  to the predicted arrival.
  - **Result complicates the earlier "13% strong coupling" finding**: median
    absolute deviation from the predicted line is 9.5 s, and the scatter is wide
    -- coupling strength does not cleanly track distance the way a real
    propagating wave should.
  - User asked directly whether this could be wind/anthropogenic-noise
    contamination rather than genuine acoustic coupling -- a real, currently
    unaddressed gap. A single whole-window cross-correlation cannot distinguish
    a real time-localized thunderclap from diffuse common-mode noise (wind can
    excite both a seismometer and a co-located microbarometer simultaneously,
    with no distance-dependent moveout). Documented as an explicit prerequisite
    (report §4.6, follow-up #1) before any "strong coupling" event is trusted as
    a gold-standard label: sliding-window (localized) cross-correlation and
    spectral-profile matching against the WS1 thunder signature are the two
    concrete next checks, neither yet implemented.
- Rewrote the report (`report/manuscript.qmd`) with richer captions/alt-text for
  all 17 figures (up from the earlier draft) reflecting the no-titles convention,
  the new moveout section, and the wind-contamination caveat as an explicit
  limitation and top-priority follow-up item.
- 45 tests pass (+4 for `thunderquakes.plotting`, +3 for `first_year_at_least`);
  ruff clean. Report renders cleanly; all 17 figure references verified against
  disk.

## 2026-07-23 — Nighttime rerun to test the wind-contamination hypothesis (Claude, Opus 4.8)
- **Prompt/intent:** the prior "evening" window (2019-05-20 20:00-02:00 UTC =
  15:00-21:00 CDT) was still afternoon/early-evening by local clock — not a real
  test of whether daytime cultural/traffic noise explains the near-zero-lag
  seismoacoustic scatter flagged as the top follow-up item. User asked to select
  a genuinely nighttime slice of the same storm and redo the full pipeline.
- Selected **2019-05-21 04:00-10:00 UTC (23:00-05:00 CDT)** — confirmed still
  actively electrified overnight (17,985 GLM flashes in one representative hour)
  with a comparable close-strike yield to the evening window (6,064/24,746/
  101,416 pairs within 5/10/20 km vs. evening's 12,650/48,151/188,883). Backed up
  all 10 evening-window output artifacts with an `_evening` suffix before rerunning
  survey -> extraction -> seismoacoustic verification -> moveout -> single-station
  detect -> 5-station batch detect on the nighttime window.
- **Found and fixed a real bug while rerunning `detect_ok_thunderquakes_batch.py`:**
  three consecutive runs crashed on what looked like transient GLM S3 network
  errors, but the actual traceback showed `s3fs`'s own timeout-retry path has a
  bug — on `FSTimeoutError`/`EndpointConnectionError` it inspects `ex.args[0]` to
  decide whether to suppress the error, but those exceptions carry no `.args`, so
  the check itself raises `IndexError` instead of the `OSError` our retry wrapper
  expected. Fixed in `src/thunderquakes/lightning/glm.py` by broadening the caught
  exception set (`S3_TRANSIENT_ERRORS = (OSError, IndexError)`) in both `_retry`
  and the granule-read fallback, so a single flaky granule is skipped/retried
  instead of aborting the whole fetch. Ruff clean; no unit test exists for this
  path (it requires live S3 flakiness to trigger), so verification was
  observational — the rerun completed cleanly afterward.
- **Result — mixed, and reported honestly rather than cherry-picked** (full
  numbers and station breakdown in report §4.7):
  - The CNN classifier's pooled candidate-vs-null discrimination got much
    *stronger* at night: p=0.072 (evening, z=1.46, 54% vs 42%, n=76) ->
    **p=0.0001** (night, z=3.83, 33% vs 12%, n=123), driven mainly by OK.FNO
    (0 candidates evening -> 50% vs 25% match rate at night) and OK.NOKA growing
    from n=3 to n=10 while staying strong (67%->70% vs 0%->10% null). OK.CROK is
    the one station with **zero discrimination in both windows** (53%/50% evening,
    21%/21% night) — flagged as likely a station-specific (site/noise-floor) issue
    rather than a day/night artifact, new follow-up item.
  - The seismoacoustic **moveout fit did not improve at night — it got worse**
    (median deviation from the 340 m/s line: 9.5 s evening, n=47 -> 16.4 s night,
    n=28), and the near-zero-lag population spanning the full distance range
    (the signature consistent with common-mode/wind contamination) persisted, with
    median |lag| actually *dropping* from 2.0 s to 0.3 s. This argues *against*
    daytime cultural noise as the main contaminant and *toward* wind or another
    always-present source — it sharpens rather than resolves the open question
    from the previous entry.
  - Net: strengthens confidence the classifier is tracking a real lightning-
    coincident signal; weakens the case that the raw seismic-infrasound
    cross-correlation metric specifically reflects genuine acoustic propagation.
    Follow-up #1 (sliding-window localized cross-correlation + spectral-profile
    check) is now the clear next step, specifically targeted at the near-zero-lag
    population's spectral signature (wind: low-frequency/non-impulsive vs.
    thunder: ~10 Hz, high flatness, per §4.1).
- Added report §4.7 "Night vs. evening: does less anthropogenic noise sharpen the
  signal?" with the full comparison table and both moveout-scatter figures
  side-by-side-referenced; updated the Follow-Up Work list (new items on
  investigating OK.CROK specifically, and partial resolution of the FNO/RLOK
  zero-candidate question — FNO now has candidates at night, RLOK still has none
  in either window, suggesting genuinely low local strike density there).
- Report renders cleanly (`quarto render manuscript.qmd`, `report/` dir).

## Template
### YYYY-MM-DD — <topic> (<model>)
- Prompt/intent:
- Accepted:
- Rejected / modified:
- Verification performed:
