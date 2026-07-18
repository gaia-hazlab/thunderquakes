"""Build a seismic + infrasound station inventory per region (WS2).

Deliverable: a table of network.station, available seismic + infrasound channels,
operational epochs, and actual archived-data availability, so we can pick stations
usable for continuous monitoring — including retrospectively. Oklahoma is the
priority target.

Data sources:
  - standard fdsnws-station (IRIS) for permanent/temporary catalogued networks
    (OK, GS, N4, TA);
  - the IRIS PH5 archive (IRISPH5) for dense nodal / assembled datasets
    (2016 LASSO = 2A, Community Wavefield = YW).

Infrasound classification (see :mod:`thunderquakes.config`): a channel is a
pressure sensor if its SEED instrument code (2nd char) is ``D``; it is *usable*
infrasound only if its band code resolves ~1-20 Hz acoustics (``BDF``/``HDF``/…),
excluding 1-sps meteorological pressure (``LDF``/``LDM``/``LDO``).
"""

from __future__ import annotations

import io

import pandas as pd
import requests
from obspy.clients.fdsn import Client

from thunderquakes.config import (
    AVAILABILITY_EXTENT_URL,
    REGIONS,
    is_acoustic_channel,
    is_pressure_channel,
)

# SEED seismometer/geophone instrument codes (2nd char): H/L high/low-gain seismo,
# N accelerometer, G gravimeter, P geophone (nodal DP*/GP* channels).
_SEISMOMETER_INSTRUMENT_CODES = frozenset("HLNGP")


def _query_stations(client: Client, networks, bbox) -> list[dict]:
    lat0, lat1, lon0, lon1 = bbox
    inv = client.get_stations(
        network=",".join(networks),
        minlatitude=lat0,
        maxlatitude=lat1,
        minlongitude=lon0,
        maxlongitude=lon1,
        level="channel",
    )
    rows = []
    for net in inv:
        for sta in net:
            for cha in sta:
                rows.append(
                    {
                        "network": net.code,
                        "station": sta.code,
                        "channel": cha.code,
                        "sampling_rate": cha.sample_rate,
                        "latitude": cha.latitude,
                        "longitude": cha.longitude,
                        "start": pd.Timestamp(str(cha.start_date)) if cha.start_date else pd.NaT,
                        "end": pd.Timestamp(str(cha.end_date)) if cha.end_date else pd.NaT,
                        "is_infrasound": is_acoustic_channel(cha.code),
                        "is_pressure": is_pressure_channel(cha.code),
                    }
                )
    return rows


def build_inventory(
    region: str,
    client: str | Client = "IRIS",
    include_nodal: bool = True,
) -> pd.DataFrame:
    """Query fdsnws (+ PH5 nodal) for a region's channels and classify them.

    Returns one row per (network, station, channel, epoch) with lat/lon, epoch,
    a ``datacenter`` tag, and boolean flags ``is_infrasound`` (acoustic-rate
    pressure) and ``is_pressure`` (any pressure sensor).
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(REGIONS)}")
    spec = REGIONS[region]
    cl = client if isinstance(client, Client) else Client(client)

    rows = _query_stations(cl, spec["seismic_networks"], spec["bbox"])
    for r in rows:
        r["datacenter"] = "IRIS"

    ph5_nets = spec.get("ph5_networks", []) if include_nodal else []
    if ph5_nets:
        ph5 = Client("IRISPH5")
        ph5_rows = _query_stations(ph5, ph5_nets, spec["bbox"])
        for r in ph5_rows:
            r["datacenter"] = "IRISPH5"
        rows += ph5_rows

    df = pd.DataFrame(rows)
    df.insert(0, "region", region)
    return df


def add_availability(
    inventory: pd.DataFrame,
    url: str = AVAILABILITY_EXTENT_URL,
    chunk: int = 150,
    timeout: int = 120,
) -> pd.DataFrame:
    """Merge actual archived-data extents onto the inventory (fdsnws networks only).

    Queries the fdsnws-availability *extent* service for every (network, station)
    served by fdsnws (PH5 nodal networks are not covered — their data extent
    equals the experiment window, so we fall back to their metadata epochs).

    Adds columns: ``data_start``, ``data_end``, ``n_timespans`` (gappiness proxy),
    ``restriction``.
    """
    out = inventory.copy()
    fdsn = out[out.get("datacenter", "IRIS") == "IRIS"]
    pairs = fdsn[["network", "station"]].drop_duplicates().itertuples(index=False)
    pairs = list(pairs)

    frames = []
    for i in range(0, len(pairs), chunk):
        lines = ["format=text"]
        for net, sta in pairs[i : i + chunk]:
            lines.append(f"{net} {sta} * * 1990-01-01T00:00:00 2027-01-01T00:00:00")
        try:
            r = requests.post(url, data="\n".join(lines) + "\n", timeout=timeout)
        except requests.RequestException:
            continue
        if r.status_code != 200 or not r.text.strip():
            continue
        frames.append(
            pd.read_csv(
                io.StringIO(r.text),
                sep=r"\s+",
                comment=None,
                names=[
                    "network", "station", "location", "channel", "quality",
                    "sampling_rate", "earliest", "latest", "updated",
                    "n_timespans", "restriction",
                ],
                skiprows=1,
            )
        )

    if not frames:
        for c in ("data_start", "data_end", "n_timespans", "restriction"):
            out[c] = pd.NaT if "data" in c else pd.NA
        return out

    av = pd.concat(frames, ignore_index=True)
    av["earliest"] = pd.to_datetime(av["earliest"], utc=True, errors="coerce")
    av["latest"] = pd.to_datetime(av["latest"], utc=True, errors="coerce")
    agg = (
        av.groupby(["network", "station", "channel"])
        .agg(
            data_start=("earliest", "min"),
            data_end=("latest", "max"),
            n_timespans=("n_timespans", "sum"),
            restriction=("restriction", "first"),
        )
        .reset_index()
    )
    return out.merge(agg, on=["network", "station", "channel"], how="left")


def find_infrasound_stations(inventory: pd.DataFrame) -> pd.DataFrame:
    """Rows for stations that carry at least one acoustic-rate infrasound channel.

    These are the seismoacoustic sites usable for WS4 Model B (dual-branch).
    """
    infra = inventory.loc[inventory["is_infrasound"], ["network", "station"]].drop_duplicates()
    return inventory.merge(infra, on=["network", "station"], how="inner")


def station_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Collapse the channel-level inventory to one row per station.

    Columns: network, station, datacenter, lat/lon, metadata epoch span, seismic
    channel codes, infrasound (acoustic) channel codes, a ``seismoacoustic`` flag,
    and — if availability was added — the actual archived-data span + gappiness.
    """
    has_avail = "data_start" in inventory.columns

    def _agg(g: pd.DataFrame) -> pd.Series:
        seis = sorted(
            {c for c in g["channel"] if len(c) >= 2 and c[1] in _SEISMOMETER_INSTRUMENT_CODES}
        )
        infra = sorted(set(g.loc[g["is_infrasound"], "channel"]))
        pressure = sorted(set(g.loc[g["is_pressure"], "channel"]))
        data = {
            "datacenter": g["datacenter"].iloc[0] if "datacenter" in g else "IRIS",
            "latitude": g["latitude"].iloc[0],
            "longitude": g["longitude"].iloc[0],
            "start": g["start"].min(),
            "end": g["end"].max(),
            "seismic_channels": ",".join(seis),
            "infrasound_channels": ",".join(infra),
            "pressure_channels": ",".join(pressure),
            "seismoacoustic": bool(infra),
        }
        if has_avail:
            data["data_start"] = g["data_start"].min()
            data["data_end"] = g["data_end"].max()
            data["n_timespans"] = g["n_timespans"].sum(min_count=1)
        return pd.Series(data)

    out = (
        inventory.groupby(["network", "station"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    return out.sort_values(["seismoacoustic", "network", "station"], ascending=[False, True, True])
