## Goal (WS1)
Characterize each PNWML class to (a) validate the signal model and (b) fix preprocessing.

## Tasks
- [ ] Finish `data/pnwml.py` loaders (done for metadata) + waveform reader from HDF5.
- [ ] Per-class distributions: dominant band, duration, **spectral flatness**, kurtosis, SNR.
- [ ] Where infrasound co-located: measure the seismo-acoustic delay for `thunder`.
- [ ] Decide window length + sampling rate (proposed: 30 s @ 100 Hz seismic).
- [ ] Emit figures + summary table for the report.

## Classes
`thunder` (positive) vs `earthquake / explosion / surface event / noise` (negatives).

## Deliverable
`thunderquakes characterize` CLI + notebook-free figures; per-class table in the report.
