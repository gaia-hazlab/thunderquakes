# Roadmap

Workstreams (WS) map 1:1 to GitHub issues/milestones. Priority region order:
**Oklahoma → PNW → Alaska**. Locked decisions: open-only lightning catalogs;
infrasound is an optional second branch (seismic-only must stand alone).

| WS | Title | Depends on | Key deliverable |
|----|-------|-----------|-----------------|
| WS0 | Foundations & repo scaffold | — | Package + pixi env + AI log + lit/signal model |
| WS1 | Labeled data & signal characterization | WS0 | Per-class frequency/duration/flatness table + plots |
| WS2 | Oklahoma station + infrasound inventory | WS0 | Table of usable seismic/BDF stations & epochs (OK, then PNW/AK) |
| WS3 | Open lightning ground-truth catalogs | WS0 | Unified strike loader (GLM/ALDN/WWLLN), common schema |
| WS4 | CNN model, train/val/test, ground-truthing | WS1,WS2,WS3 | Trained classifier + PNW→OK generalization + completeness gain |
| WS5 | Reproducible Quarto report + AI logs | WS1–WS4 | IMRaD `.qmd`, figures-from-raw, `AI_LOG.md` |

## Milestones (indicative)
- **M1 (wk 1–3):** WS0 done; WS1 + WS2(OK) + WS3(GLM) in progress.
- **M2 (wk 4–6):** WS1/WS2/WS3 complete; first seismic-only CNN trained on PNWML.
- **M3 (wk 7–10):** dual-branch + uncertainty/calibration; PNW→OK generalization test.
- **M4 (wk 10–12):** OK continuous run + completeness gain; Quarto report drafted.

## Cross-cutting risks
- **Label contamination** — never label positives by SNR alone (the prototype's trap);
  require lightning + (where available) seismo-acoustic confirmation.
- **Domain shift PNW→OK→AK** — expect degradation; budget for fine-tuning.
- **Alaska lightning truth is weakest** (GLM poor at high latitude) — hence AK is 3rd.

## What the prototype already established (do not redo)
- The PNWML `thunder` class exists and is analyst-labeled.
- **ASOS weather observations are airport-centric and do NOT co-locate** with remote
  seismic thunder stations → ASOS is not a viable label/validation source. Retired.
