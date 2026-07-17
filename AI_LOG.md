# AI-assisted development log

Per Denolle-group norms, we record AI usage and the decisions it informed. Keep
entries short: what was asked, what was accepted/rejected, and why.

## 2026-07-17 — Repo restructure & roadmap (Claude, Opus 4.8)
- **Audited** the prototype `notebooks/01_thunderquakes_in_the_pnw.ipynb`. Key findings:
  the ASOS co-location approach is infeasible (airport-centric) — *retired*; the
  `snr>=3` "detection" label is circular — *retired*; no infrasound / no ML / no
  Oklahoma coverage yet.
- **Restructured** the notebook prototype into a standard Python package
  (`src/thunderquakes/`) with pixi + conda environments. Original notebook preserved
  under `notebooks/archive/` for provenance.
- **Decisions locked (by MD):** open-only lightning catalogs (GLM/ALDN/WWLLN);
  infrasound is an optional second branch (seismic-only must stand alone).
- Roadmap encoded as GitHub issues (WS0–WS5). See ROADMAP.md.

## Template
### YYYY-MM-DD — <topic> (<model>)
- Prompt/intent:
- Accepted:
- Rejected / modified:
- Verification performed:
