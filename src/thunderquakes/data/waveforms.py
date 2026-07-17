"""FDSN waveform access with proper preprocessing.

Improves on the prototype's ``fetch_and_score`` (cell 28): merges gappy
streams, optionally removes instrument response so characterization is done in
physical units, and never silently takes ``st[0]``.

WS1/WS4 use this. Kept dependency-light (obspy only).
"""

from __future__ import annotations

import warnings

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

from thunderquakes.config import SEISMIC_BAND

# Preference order for a vertical seismic channel.
SEISMIC_CHANNELS = ("HHZ", "BHZ", "EHZ", "HNZ")


def fetch_window(
    net: str,
    sta: str,
    t_center,
    pre_s: float = 300.0,
    post_s: float = 600.0,
    channels=SEISMIC_CHANNELS,
    remove_response: bool = False,
    band=SEISMIC_BAND,
    client: str | Client = "IRIS",
) -> tuple[Stream | None, str | None]:
    """Fetch a preprocessed window; return ``(stream, channel)`` or ``(None, None)``.

    Tries each channel in ``channels`` until data is returned. Detrends, tapers,
    merges gaps, optionally removes response, then band-passes.
    """
    cl = client if isinstance(client, Client) else Client(client)
    t0 = UTCDateTime(t_center.isoformat() if hasattr(t_center, "isoformat") else t_center)
    for chan in channels:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = cl.get_waveforms(net, sta, "*", chan, t0 - pre_s, t0 + post_s)
        except Exception:
            continue
        if not st:
            continue
        st.merge(method=1, fill_value="interpolate")
        st.detrend("demean")
        st.taper(0.05)
        if remove_response:
            try:
                st.remove_response(output="VEL")
            except Exception:
                pass  # fall back to counts; caller decides
        fmin, fmax = band
        st.filter("bandpass", freqmin=fmin, freqmax=min(fmax, 0.45 * st[0].stats.sampling_rate))
        return st, chan
    return None, None
