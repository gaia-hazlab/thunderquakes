"""Build a seismic + infrasound station inventory per region (WS2).

Deliverable: a table of network.station, available seismic + infrasound channels,
operational epochs, so we can pick stations usable for continuous monitoring —
including retrospectively. Oklahoma is the priority target; the TA/N4 networks
carry co-located infrasound microphones (``BDF``) across the central US.

Infrasound classification (see :mod:`thunderquakes.config`): a channel is a
pressure sensor if its SEED instrument code (2nd char) is ``D``; it is *usable*
infrasound only if its band code resolves ~1-20 Hz acoustics (``BDF``/``BDO``/…),
excluding 1-sps meteorological pressure (``LDF``/``LDM``/``LDO``).
"""

from __future__ import annotations

import pandas as pd
from obspy.clients.fdsn import Client

from thunderquakes.config import REGIONS, is_acoustic_channel, is_pressure_channel

# SEED seismometer instrument codes (2nd char): H=high-gain, L=low-gain, N=accel, P/G=gravimeter.
_SEISMOMETER_INSTRUMENT_CODES = frozenset("HLNG")


def build_inventory(region: str, client: str | Client = "IRIS") -> pd.DataFrame:
    """Query FDSN for all channels of the region's networks and classify them.

    Returns one row per (network, station, channel, epoch) with lat/lon, epoch,
    and boolean flags ``is_infrasound`` (acoustic-rate pressure) and
    ``is_pressure`` (any pressure sensor, incl. low-rate met pressure).
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
                        "sampling_rate": cha.sample_rate,
                        "latitude": cha.latitude,
                        "longitude": cha.longitude,
                        "start": pd.Timestamp(str(cha.start_date)) if cha.start_date else pd.NaT,
                        "end": pd.Timestamp(str(cha.end_date)) if cha.end_date else pd.NaT,
                        "is_infrasound": is_acoustic_channel(cha.code),
                        "is_pressure": is_pressure_channel(cha.code),
                    }
                )
    return pd.DataFrame(rows)


def find_infrasound_stations(inventory: pd.DataFrame) -> pd.DataFrame:
    """Rows for stations that carry at least one acoustic-rate infrasound channel.

    These are the seismoacoustic sites usable for WS4 Model B (dual-branch).
    """
    infra = inventory.loc[inventory["is_infrasound"], ["network", "station"]].drop_duplicates()
    return inventory.merge(infra, on=["network", "station"], how="inner")


def station_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Collapse the channel-level inventory to one row per station.

    Columns: network, station, lat/lon, operational epoch span, seismic channel
    codes, infrasound (acoustic) channel codes, and a ``seismoacoustic`` flag.
    """

    def _agg(g: pd.DataFrame) -> pd.Series:
        seis = sorted(
            {c for c in g["channel"] if len(c) >= 2 and c[1] in _SEISMOMETER_INSTRUMENT_CODES}
        )
        infra = sorted(set(g.loc[g["is_infrasound"], "channel"]))
        pressure = sorted(set(g.loc[g["is_pressure"], "channel"]))
        return pd.Series(
            {
                "latitude": g["latitude"].iloc[0],
                "longitude": g["longitude"].iloc[0],
                "start": g["start"].min(),
                "end": g["end"].max(),
                "seismic_channels": ",".join(seis),
                "infrasound_channels": ",".join(infra),
                "pressure_channels": ",".join(pressure),
                "seismoacoustic": bool(infra),
            }
        )

    out = (
        inventory.groupby(["network", "station"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    return out.sort_values(["seismoacoustic", "network", "station"], ascending=[False, True, True])
