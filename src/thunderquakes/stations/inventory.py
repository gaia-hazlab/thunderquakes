"""Build a seismic + infrasound station inventory per region (WS2).

Deliverable: a table of network.station, available seismic + infrasound (BDF)
channels, operational epochs, so we can pick stations usable for continuous
monitoring — including retrospectively. Oklahoma is the priority target; the
TA/N4 networks carry co-located BDF infrasound microphones across the central US.
"""

from __future__ import annotations

import pandas as pd
from obspy.clients.fdsn import Client

from thunderquakes.config import INFRASOUND_CHANNELS, REGIONS


def build_inventory(region: str, client: str | Client = "IRIS") -> pd.DataFrame:
    """Query FDSN for all channels of the region's networks and flag infrasound.

    Returns one row per (network, station, channel) with lat/lon and epoch.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(REGIONS)}")
    spec = REGIONS[region]
    lat0, lat1, lon0, lon1 = spec["bbox"]
    cl = client if isinstance(client, Client) else Client(client)

    inv = cl.get_stations(
        network=",".join(spec["seismic_networks"]),
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
                        "region": region,
                        "network": net.code,
                        "station": sta.code,
                        "channel": cha.code,
                        "latitude": cha.latitude,
                        "longitude": cha.longitude,
                        "start": pd.Timestamp(str(cha.start_date)) if cha.start_date else pd.NaT,
                        "end": pd.Timestamp(str(cha.end_date)) if cha.end_date else pd.NaT,
                        "is_infrasound": cha.code in INFRASOUND_CHANNELS,
                    }
                )
    return pd.DataFrame(rows)


def find_infrasound_stations(inventory: pd.DataFrame) -> pd.DataFrame:
    """Stations that carry at least one infrasound channel (seismoacoustic sites)."""
    infra = inventory.loc[inventory["is_infrasound"], ["network", "station"]].drop_duplicates()
    return inventory.merge(infra, on=["network", "station"], how="inner")
