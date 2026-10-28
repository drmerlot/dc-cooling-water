"""Reproduces Water Ledger's published West Michigan estimates (model 1.2) from the same inputs: 30 years of hourly
ERA5 weather at 42.88 N, 85.52 W (Grand Rapids area), 200 runs, seed 7, and staff water from 200 draws, seed 13.

The weather tests need the weather file (about 7 MB). Set DCW_WEST_MICHIGAN_WEATHER to a saved Open-Meteo JSON, or
set DCW_FETCH=1 to download it; otherwise they are skipped.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from dc_cooling_water import fetch_open_meteo, from_open_meteo_json, run_ensemble, sample_staff, summarize
from dc_cooling_water.model import STAFF_PARAMETERS

REF = json.loads((Path(__file__).parent / "water_ledger_west_michigan.json").read_text())
PUBLISHED = 0.5e-5 + 1e-12          # published values are rounded to 5 decimal places
FLOW_KEYS = [(d, k) for d, stats in REF["per_100mw"].items() for k in stats if k.split("_")[0] in ("withdrawn", "consumed", "returned")]


@pytest.fixture(scope="module")
def weather():
    path = os.environ.get("DCW_WEST_MICHIGAN_WEATHER")
    if path and os.path.exists(path):
        return from_open_meteo_json(path)
    if os.environ.get("DCW_FETCH"):
        return fetch_open_meteo(42.88, -85.52, cache=str(Path(__file__).parent / ".cache" / "west_michigan.json"))
    pytest.skip("set DCW_WEST_MICHIGAN_WEATHER or DCW_FETCH=1 to run the reproduction test")


def ensemble(w, mw):
    return run_ensemble(w.time.values, w.air_temperature.values, w.relative_humidity.values, mw=mw, n=200, seed=7)


def test_staff_draws_match_water_ledger():
    s = sample_staff()
    np.testing.assert_allclose(np.stack([s[k] for k in STAFF_PARAMETERS], axis=1), REF["staff_runs"], rtol=1e-15)


def test_reproduces_water_ledger_west_michigan(weather):
    ds = ensemble(weather, 100)
    s = summarize(ds, staff=False)
    for design, key in FLOW_KEYS:
        flow, stat = key.split("_", 1)
        for q, expected in REF["per_100mw"][design][key].items():
            got = float(s[flow].sel(design=design, stat=stat, quantile=q))
            assert abs(got - expected) <= PUBLISHED, f"{design}/{key} {q}: {got} vs {expected}"
    for design, stats in REF["per_100mw"].items():
        for var, key in (("wue", "wue_l_per_kwh"), ("wue_makeup", "wue_makeup_l_per_kwh"), ("param_utilization", "utilization")):
            v = ds[var].sel(design=design) if "design" in ds[var].dims else ds[var]
            for q in ("p10", "p50", "p90"):
                got = float(v.quantile(int(q[1:]) / 100, dim="run"))
                assert abs(got - stats[key][q]) <= PUBLISHED, f"{design}/{key} {q}"


def test_campus_at_300_mw_matches_water_ledger(weather):
    """At 300 MW, Water Ledger shows the per-100 MW figures scaled to the size plus staff water at that size, the
    two ranges added end to end (scenario.flow / WQ.flow). summarize() does the same."""
    s = summarize(ensemble(weather, 300))
    runs = np.array(REF["staff_runs"])
    g = (runs[:, 0] + runs[:, 1] * 300) * runs[:, 2] / 1e6
    staff = {"withdrawn": g, "consumed": g * runs[:, 3], "returned": g * (1 - runs[:, 3])}
    for design, key in FLOW_KEYS:
        flow, stat = key.split("_", 1)
        k = 365.25 if key.endswith("_mg") else 1.0
        for q in ("p10", "p50", "p90"):
            expected = REF["per_100mw"][design][key][q] * 3 + np.quantile(staff[flow], int(q[1:]) / 100) * k
            got = float(s[flow].sel(design=design, stat=stat, quantile=q))
            assert abs(got - expected) <= 3 * PUBLISHED, f"{design}/{key} {q}: {got} vs {expected}"
    # the figures on the page: a 300 MW closed-loop campus takes about 2,700 gallons a day and loses about 340
    dry = s.sel(design="dry", stat="avg", quantile="p50")
    assert round(float(dry["withdrawn"]) * 1e6, -2) == 2700
    assert round(float(dry["consumed"]) * 1e6, -1) == 340
