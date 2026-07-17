## Goal (WS2, priority region)
Table of Oklahoma seismic + infrasound stations usable for continuous monitoring, including retrospectively.

## Tasks
- [ ] `stations.build_inventory("OK")` over networks OK, GS, N4, TA (implemented — verify against FDSN).
- [ ] Flag co-located infrasound (`BDF`/`HDF`/`BDH`/`HDH`) — TA/N4 carry microphones across central US.
- [ ] Add operational epochs + continuous-data availability (FDSN availability service).
- [ ] Note dense historical experiments (LASSO / IRIS Community Wavefield near Enid, 2016).
- [ ] Export `outputs/stations_OK.csv`.

## Deliverable
CSV + map of OK stations with seismic + infrasound channels and epochs.
