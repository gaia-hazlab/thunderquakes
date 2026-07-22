"""Shared plotting style: print-readable fonts and a US-Letter figure width.

Two problems this fixes across the project's figure-generating scripts:
  1. Font sizes were often set as small absolute point values (5-8pt) chosen to
     fit dense multi-panel galleries on screen. When such a figure is placed at
     print width in a report, that text shrinks below readability. Call
     ``set_paper_style()`` once at the top of a plotting script; it sets
     matplotlib rcParams so that even unstyled text defaults to a readable size,
     and exposes ``MIN_FONT`` as the floor any explicit fontsize= call should use.
  2. Figures need to fit a US Letter page (8.5 x 11 in) with ~1 in margins, i.e.
     a printable width of ~6.5 in. ``LETTER_WIDTH_IN`` / ``LETTER_HEIGHT_IN`` are
     the safe values to size figsize against, and ``letter_figsize`` computes a
     figsize that fits the full page width for a target row/column aspect.
"""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt

MIN_FONT = 10  # points; the floor requested for every axis/tick/title/legend label
LETTER_WIDTH_IN = 6.5  # printable width, US Letter, 1in margins each side
LETTER_HEIGHT_IN = 9.0  # printable height, US Letter, 1in margins each side


def set_paper_style() -> None:
    """Set matplotlib rcParams so default (unstyled) text is print-readable."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "savefig.dpi": 200,
    })


def letter_figsize(aspect: float = 0.6, width_in: float = LETTER_WIDTH_IN) -> tuple:
    """A (width, height) in inches spanning the full Letter print width.

    ``aspect`` is height/width; e.g. 0.6 gives a typical wide single-panel figure.
    """
    return (width_in, width_in * aspect)


def gray_relief_background(bbox, ax, alpha: float = 1.0) -> bool:
    """Draw a very light-grey shaded-relief basemap on ``ax`` for ``(lat0, lat1, lon0, lon1)``.

    Renders pure hillshade (gradient of elevation, not elevation itself) through
    a manual 2-stop CPT confined to near-white greys (RGB 235-252) so the relief
    texture is visible but never competes with station/strike markers plotted on
    top, plus thin coastline and state/political border lines for geographic
    context. Tiles are cached to ``data/cache/relief_<hash>.png`` (gitignored)
    since the same region is rendered on every script re-run. Resolution is
    chosen from the bbox size to keep download/render time reasonable for large
    regions (e.g. Alaska). Returns True on success; on any failure (no network,
    no pygmt), returns False and leaves ``ax`` untouched so callers degrade
    gracefully to a plain white background.
    """
    lat0, lat1, lon0, lon1 = bbox
    span = max(lat1 - lat0, lon1 - lon0)
    resolution = "10m" if span > 15 else ("05m" if span > 5 else "03m")

    from thunderquakes.config import DATA_ROOT

    cache_dir = DATA_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # v2: cache key bumped when coastline/border overlay was added, so old
    # (relief-only) cached tiles are not reused silently.
    key = hashlib.sha1(f"{bbox}-{resolution}-v2".encode()).hexdigest()[:16]
    png_path = cache_dir / f"relief_{key}.png"

    if not png_path.exists():
        try:
            import numpy as np
            import pygmt

            region = [lon0, lon1, lat0, lat1]
            grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
            shade = pygmt.grdgradient(grid=grid, azimuth=300, normalize="t0.35")
            lo, hi = float(shade.min()), float(shade.max())
            cpt_path = cache_dir / f"relief_{key}.cpt"
            with open(cpt_path, "w") as f:
                f.write(f"{lo}\t235\t235\t235\t{hi}\t252\t252\t252\n")
                f.write("B\t235\t235\t235\nF\t252\t252\t252\nN\t245\t245\t245\n")
            # Linear (Cartesian) projection: degrees map to inches with no
            # distortion, matching matplotlib's plain lat/lon imshow exactly
            # (a Mercator "M" projection is nonlinear in latitude and would
            # misalign the background for large-latitude-span regions like AK).
            # Height also corrected for cos(latitude) so coastline/border
            # linework isn't warped before matplotlib re-projects it.
            lon_span, lat_span = lon1 - lon0, lat1 - lat0
            mean_lat_rad = np.radians((lat0 + lat1) / 2)
            w = 8.0
            h = max(0.5, w * lat_span / (lon_span * max(np.cos(mean_lat_rad), 0.05)))
            fig = pygmt.Figure()
            fig.grdimage(grid=shade, region=region, projection=f"X{w}i/{h}i",
                        cmap=str(cpt_path), frame=False)
            fig.coast(region=region, projection=f"X{w}i/{h}i",
                     shorelines="0.4p,gray30", borders=["1/0.5p,gray20", "2/0.3p,gray50"])
            fig.savefig(png_path, transparent=False, dpi=200)
        except Exception:
            return False

    try:
        img = plt.imread(png_path)
    except Exception:
        return False
    ax.imshow(img, extent=(lon0, lon1, lat0, lat1), origin="upper",
             aspect="auto", alpha=alpha, zorder=0)
    return True
