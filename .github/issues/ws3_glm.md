## Goal (WS3, primary truth for OK & PNW)
Load GOES-GLM lightning flashes as independent ground truth.

## Tasks
- [ ] Implement `lightning/glm.py` `load_glm_strikes()`: list `s3://noaa-goes16/GLM-L2-LCFA/`
      granules for a time range via `s3fs`, read flash centroids with `xarray`, filter to bbox.
- [ ] Normalise to COMMON_SCHEMA (`time_utc, latitude, longitude, peak_current, source`).
- [ ] Add GOES-18 (West) option; document poor Alaska coverage.
- [ ] Cache granules under the data root.

## Deliverable
`load_glm_strikes(t0, t1, bbox)` returning the common schema; used by WS4 evaluation.
