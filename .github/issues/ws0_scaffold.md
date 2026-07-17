## Goal
Standard Python package + reproducible environments so all later work is portable off the UW host.

## Tasks
- [x] `src/thunderquakes/` package layout
- [x] `pixi.toml` (default/ml/dev/report/all envs) + `environment.yml` conda fallback
- [x] `config.py` — config-driven data paths (replaces hardcoded `/data/whd02/...`)
- [x] Preserve original prototype under `notebooks/archive/`
- [x] `AI_LOG.md`, `data/README.md` provenance
- [x] pytest suite for config/geo/evaluation/features
- [ ] GitHub Actions CI: `pixi run -e all test` + `ruff check`
- [ ] Pin/lock the pixi environment (`pixi.lock`) and commit it

## Notes
Retires the prototype's dead ends: ASOS co-location (airport-centric, 0% match) and
`snr>=3` "detection" labels (circular). See ROADMAP.md.
