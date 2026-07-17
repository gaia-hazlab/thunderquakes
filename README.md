# thunderquakes

Seismoacoustic detection of **thunderquakes** — thunder-induced ground motion — to
increase the completeness of lightning-strike catalogs. Target regions, in priority
order: **Oklahoma → Pacific Northwest → Alaska**.

We detect thunderquakes on seismometers (everywhere) and, where co-located
infrasound microphones exist (TA / N4 / AK-TA `BDF` channels), on the coupled
**seismo-acoustic** signal, then validate against independent open lightning
catalogs (GOES-GLM, ALDN, WWLLN).

## Why
Lightning-detection networks miss strikes, especially in remote terrain and at high
latitude. Thunder couples into the ground locally; a seismic (and infrasound) detector
co-located with sparse lightning coverage can recover missed strikes and fill catalog gaps.

## Approach at a glance
- **Labels:** the PNWML analyst-labeled `thunder` class (gold positives) vs
  `earthquake / explosion / surface-event / noise` (gold negatives).
- **Physics:** thunderquakes are broadband (~1–20+ Hz), emergent, ~10–30 s per clap,
  with an air-coupled acoustic arrival lagging the direct seismic arrival — the
  seismo-acoustic delay is the key discriminator.
- **Model:** a spectrogram CNN. *Model A* is seismic-only (deployable everywhere);
  *Model B* adds an infrasound branch on the co-located subset. Uncertainty via
  MC-dropout + a small deep ensemble, with calibration checks.
- **Ground truth:** open lightning catalogs only, treated as a lower bound; results
  are reported as **completeness gain**, not naive precision.

See [ROADMAP.md](ROADMAP.md) and the GitHub issues for the workstream plan.

## Setup (pixi — preferred)
```bash
pixi install                 # default env (data + analysis)
pixi run -e all test         # run tests
pixi run -e all thunderquakes stations --region OK
```
Environments: `default` (analysis), `ml` (adds PyTorch), `dev` (pytest/ruff),
`report` (Quarto), `all`.

## Setup (conda fallback)
```bash
conda env create -f environment.yml
conda activate thunderquakes
```

## Data
Raw data is **not** committed. Point the package at your data root via the
`THUNDERQUAKES_DATA` env var or `config.yaml` (`data_root: ...`). See
[data/README.md](data/README.md) for provenance and expected layout.

## Package layout
```
src/thunderquakes/
  config.py        paths, regions, signal defaults (replaces hardcoded paths)
  data/            PNWML loaders + FDSN waveform access
  stations/        seismic + infrasound inventory (WS2)
  lightning/       GLM / ALDN / WWLLN loaders (WS3)
  features/        spectrograms + per-class characterization (WS1)
  models/          CNN, uncertainty, training (WS4)
  evaluation/      ground-truthing vs lightning catalogs (WS4)
report/            Quarto manuscript (WS5)
notebooks/archive/ original prototype notebook, preserved for provenance
```

## AI-assisted development
Development decisions and AI usage are logged in [AI_LOG.md](AI_LOG.md), per group norms.
