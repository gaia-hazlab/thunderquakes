#!/usr/bin/env bash
# Create labels, milestones, and roadmap issues on the GitHub repo.
# Usage: REPO=mdenolle/thunderquakes bash .github/create_roadmap.sh
set -euo pipefail
: "${REPO:?set REPO=owner/name}"

echo "== labels =="
mklabel() { gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force >/dev/null; }
mklabel "WS0-foundations"   "5319e7" "Scaffold, envs, lit/signal model"
mklabel "WS1-characterize"  "1d76db" "Labeled data & signal characterization"
mklabel "WS2-stations"      "0e8a16" "Seismic + infrasound inventory"
mklabel "WS3-lightning"     "fbca04" "Open lightning ground-truth catalogs"
mklabel "WS4-model"         "d93f0b" "CNN, training, uncertainty, ground-truthing"
mklabel "WS5-report"        "b60205" "Quarto report + reproducibility + AI logs"
mklabel "region:OK"         "0052cc" "Oklahoma (priority 1)"
mklabel "region:PNW"        "0052cc" "Pacific Northwest (priority 2)"
mklabel "region:AK"         "0052cc" "Alaska (priority 3)"

echo "== milestones =="
mkms() { gh api "repos/$REPO/milestones" -f title="$1" -f description="$2" >/dev/null 2>&1 || true; }
mkms "M1 — foundations & data plumbing" "WS0 done; WS1/WS2(OK)/WS3(GLM) underway"
mkms "M2 — first classifier"            "WS1/WS2/WS3 complete; seismic-only CNN trained on PNWML"
mkms "M3 — seismoacoustic + generalization" "Dual-branch + uncertainty; PNW->OK generalization"
mkms "M4 — completeness & report"       "OK continuous run + completeness gain; Quarto report"

# map milestone title -> number
ms_num() { gh api "repos/$REPO/milestones?state=all" --jq ".[] | select(.title|startswith(\"$1\")) | .number"; }
M1=$(ms_num "M1"); M2=$(ms_num "M2"); M3=$(ms_num "M3"); M4=$(ms_num "M4")

echo "== issues =="
mkissue() { # title labels milestone bodyfile
  gh issue create --repo "$REPO" --title "$1" --label "$2" \
    ${3:+--milestone "$3"} --body-file "$4" >/dev/null
  echo "  + $1"
}
D=.github/issues
mkissue "[WS0] Repo scaffold, pixi/conda envs, CI"                 "WS0-foundations" "$M1" "$D/ws0_scaffold.md"
mkissue "[WS0] Literature review & thunderquake signal model"      "WS0-foundations" "$M1" "$D/ws0_litreview.md"
mkissue "[WS1] PNWML loaders + per-class signal characterization"  "WS1-characterize,region:PNW" "$M1" "$D/ws1_characterize.md"
mkissue "[WS2] Oklahoma seismic + infrasound station inventory"    "WS2-stations,region:OK" "$M1" "$D/ws2_ok_inventory.md"
mkissue "[WS2] Extend inventory to PNW and Alaska"                 "WS2-stations,region:PNW,region:AK" "$M2" "$D/ws2_pnw_ak_inventory.md"
mkissue "[WS3] GOES-GLM lightning loader (OK & PNW truth)"        "WS3-lightning,region:OK,region:PNW" "$M1" "$D/ws3_glm.md"
mkissue "[WS3] ALDN + WWLLN loaders (Alaska + backstop)"          "WS3-lightning,region:AK" "$M2" "$D/ws3_aldn_wwlln.md"
mkissue "[WS4] Seismic-only spectrogram CNN + training harness"   "WS4-model" "$M2" "$D/ws4_cnn.md"
mkissue "[WS4] Dual-branch seismoacoustic model + ablation"       "WS4-model" "$M3" "$D/ws4_dualbranch.md"
mkissue "[WS4] Uncertainty (MC-dropout + ensemble) + calibration" "WS4-model" "$M3" "$D/ws4_uncertainty.md"
mkissue "[WS4] PNW->OK generalization + completeness ground-truthing" "WS4-model,region:OK" "$M4" "$D/ws4_generalization.md"
mkissue "[WS5] Quarto report, reproducibility, AI logs"          "WS5-report" "$M4" "$D/ws5_report.md"
echo "done."
