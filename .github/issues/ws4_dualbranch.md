## Goal (WS4, Model B)
Add an infrasound branch on the co-located subset; prove the seismo-acoustic delay helps.

## Tasks
- [ ] `build_dual_branch_cnn`: shared encoder, seismic + infrasound branches, late fusion.
- [ ] Model A == Model B with infrasound branch zeroed (single codebase).
- [ ] **Ablation**: A vs B on the co-located subset (TA/N4/AK-TA `BDF`).
- [ ] Use Model B high-confidence outputs to auto-label/clean Model A's training set.

## Deliverable
Ablation table quantifying the infrasound contribution.
