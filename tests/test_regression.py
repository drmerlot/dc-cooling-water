"""Reproduces Water Ledger's published West Michigan estimates (per 100 MW of IT capacity) from the same inputs:
30 years of hourly ERA5 weather at 42.88 N, 85.52 W (Grand Rapids area), 200 runs, seed 7.

Needs the weather file (about 7 MB). Set DCW_WEST_MICHIGAN_WEATHER to a saved Open-Meteo JSON, or set
DCW_FETCH=1 to download it; otherwise the test is skipped.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from dc_cooling_water import fetch_open_meteo, from_open_meteo_json, run_ensemble, summarize

REF = json.loads((Path(__file__).parent / "water_ledger_west_michigan.json").read_text())


def weather():
    path = os.environ.get("DCW_WEST_MICHIGAN_WEATHER")
    if path and os.path.exists(path):
        return from_open_meteo_json(path)
    if os.environ.get("DCW_FETCH"):
        return fetch_open_meteo(42.88, -85.52, cache=str(Path(__file__).parent / ".cache" / "west_michigan.json"))
    pytest.skip("set DCW_WEST_MICHIGAN_WEATHER or DCW_FETCH=1 to run the reproduction test")


def test_reproduces_water_ledger_west_michigan():
    w = weather()
    s = summarize(run_ensemble(w.time.values, w.air_temperature.values, w.relative_humidity.values, mw=100, n=200, seed=7))
    for key, quantiles in REF.items():
        design, var = key.split("/")
        flow, stat = var.split("_", 1)
        for q, expected in quantiles.items():
            got = float(s[flow].sel(design=design, stat=stat, quantile=q))
            np.testing.assert_allclose(got, expected, rtol=1e-12, err_msg=f"{key} {q}")
