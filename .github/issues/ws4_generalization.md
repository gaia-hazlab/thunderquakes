## Goal (WS4, the headline result)
Generalize to the deployment target (OK) and measure catalog completeness gain.

## Tasks
- [ ] **Train PNW → test OK** domain-shift experiment; report degradation + optional fine-tune.
- [ ] Run Model A over **continuous OK data** for selected convective days.
- [ ] `evaluation.match_detections_to_strikes` vs GLM (implemented) → TP if strike within (km, s).
- [ ] Report as **completeness gain vs the open catalog** (catalog = lower bound), not naive precision.

## Deliverable
The core scientific result: new thunderquake detections + estimated completeness gain.
