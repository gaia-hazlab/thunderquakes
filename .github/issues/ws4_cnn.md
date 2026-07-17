## Goal (WS4, Model A)
Seismic-only spectrogram CNN — the deployable workhorse (all stations/epochs).

## Tasks
- [ ] `models/cnn.py::build_seismic_cnn` (3-C spectrogram input).
- [ ] `models/train.py`: **event/station/region-disjoint** splits (no leakage).
- [ ] Train on PNWML {thunder, earthquake, explosion, surface, noise}.
- [ ] Metrics: **PR-AUC / precision@fixed-recall** (class imbalance), per-class confusion.
- [ ] TensorBoard logging; config in `configs/seismic_only.yaml`.

## Deliverable
Trained Model A + evaluation report on held-out PNW.
