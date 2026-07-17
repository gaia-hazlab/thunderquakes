# Data provenance

Raw data is **not** committed to git. Configure the data root via the
`THUNDERQUAKES_DATA` environment variable or `config.yaml` (`data_root:`), then
place datasets in the layout below.

## PNWML (labeled seismic waveforms — gold labels)
Curated Pacific Northwest ML dataset (Ni et al.). Expected layout under
`$THUNDERQUAKES_DATA/PNWML/`:
```
comcat/{waveforms.hdf5,metadata.csv}   # earthquake, explosion
exotic/{waveforms.hdf5,metadata.csv}   # thunder, surface event, ...
noise/{waveforms.hdf5,metadata.csv}
```
The prototype read these from `/data/whd02/niyiyu_data/PNWML/` on a UW host;
that path is now config-driven.

## Lightning catalogs (open-only — independent ground truth)
- **GOES-GLM** — `s3://noaa-goes16/GLM-L2-LCFA/` (public, no credentials). OK & PNW, 2018+.
- **ALDN** — Alaska Lightning Detection Network (BLM/AICC). Alaska.
- **WWLLN** — research data agreement; global, high-latitude, pre-2018 backstop.

Treat every catalog as a **lower bound** on true lightning activity (detection
efficiency varies by network and region).

## Seismic / infrasound waveforms
Fetched on demand from FDSN (IRIS) — see `thunderquakes.data.waveforms` and
`thunderquakes.stations`. Networks: OK/GS/N4/TA (Oklahoma), UW/UO/CC (PNW),
AK/TA/AV (Alaska). Infrasound = `BDF`/`HDF`/`BDH`/`HDH` channels.
