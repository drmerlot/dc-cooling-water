"""Hourly weather for the model: ERA5 reanalysis temperature and humidity from the Open-Meteo archive, as xarray."""
import json
import os
import urllib.request

import numpy as np
import xarray as xr

OPEN_METEO = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
              "&hourly=temperature_2m,relative_humidity_2m&temperature_unit=celsius&timezone={tz}")


def from_open_meteo_json(payload):
    """An Open-Meteo archive response (dict or path to a saved JSON file) as a Dataset with 'air_temperature' (C)
    and 'relative_humidity' (%) on local hourly time."""
    if isinstance(payload, (str, os.PathLike)):
        with open(payload) as f:
            payload = json.load(f)
    h = payload["hourly"]
    time = np.array(h["time"], dtype="datetime64[m]")
    return xr.Dataset({"air_temperature": ("time", np.array(h["temperature_2m"], dtype=float), {"units": "degC"}),
                       "relative_humidity": ("time", np.array(h["relative_humidity_2m"], dtype=float), {"units": "%"})},
                      coords={"time": time},
                      attrs={"source": "ERA5 reanalysis via the Open-Meteo archive API", "latitude": payload.get("latitude"),
                             "longitude": payload.get("longitude")})


def fetch_open_meteo(lat, lon, start="1995-01-01", end="2024-12-31", tz="America/Detroit", cache=None):
    """Download hourly ERA5 temperature and humidity for one point (about 7 MB for 30 years). With cache, a path
    to a JSON file, the response is saved there and reused."""
    if cache and os.path.exists(cache):
        return from_open_meteo_json(cache)
    url = OPEN_METEO.format(lat=lat, lon=lon, start=start, end=end, tz=tz.replace("/", "%2F"))
    with urllib.request.urlopen(url, timeout=300) as r:
        payload = json.load(r)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(payload, f)
    return from_open_meteo_json(payload)
