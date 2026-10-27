"""Lake-source (once-through) cooling: water drawn from a lake, run once through heat exchangers and returned warmer.

Withdrawal follows from the heat load and the temperature rise: flow = heat / (density x specific heat x rise). Nearly
all of it returns, but the added heat drives extra ("forced") evaporation from the lake surface, estimated with the
linearized heat-balance method the USGS uses for once-through power plants (Diehl et al. 2013, after Harbeck 1964 and
Ward 1980): the share of added heat leaving as evaporation is dE / (dE + dC + dR), where evaporation and conduction
share the same wind function and radiation does not. Natural surface temperatures: NOAA GLERL daily long-term averages.
"""
import io
import urllib.request

import numpy as np
import xarray as xr

from . import model

RHO, CP, LV = 1000.0, 4186.0, 2.45e6       # kg m-3, J kg-1 K-1, J kg-1
P_MB = 1013.0
BOWEN = P_MB * 0.24 / (0.622 * 585.0)      # psychrometric constant, mb per C (cp of air 0.24 cal/g/K, L 585 cal/g)
SIGMA = 1.17e-7 / 86400.0                  # Stefan-Boltzmann, cal cm-2 s-1 K-4
EMIS = 0.97
MPH_PER_MS = 2.23694

# Uniform ranges for the Monte Carlo; utilization and heat factor reuse the cooling-tower model's ranges.
RANGES = {
    "utilization": model.RANGES["utilization"],
    "heat_factor": (model.RANGES["hf_econ"][0], model.RANGES["hf_chiller"][1]),
    "rise_c": (6.0, 10.0),                 # temperature rise across the exchangers, C [estimate]
    "wind_mph": (5.0, 15.0),               # wind over the lake [estimate]
}
GLERL = "https://apps.glerl.noaa.gov/erddap/tabledap/glerlLTAvgTemp.csv?lake,dayOfYear,temperature"
GLERL_CODES = {"Lake Superior": "s", "Lake Michigan": "m", "Lake Huron": "h", "Lake Erie": "e", "Lake Ontario": "o"}


def _es(t):
    """Saturation vapor pressure over water, mb (Magnus)."""
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def evaporation_ratio(t, wind_mph):
    """Share of added heat lost by evaporation at natural water temperature t (C); linearized heat balance
    (Ward 1980; USGS SIR 2013-5188)."""
    f = 7.0e-8 * wind_mph                  # Ward wind function, cal cm-2 s-1 mb-1
    de = _es(t + 0.5) - _es(t - 0.5)       # mb per C
    d_e = 585.0 * f * de
    d_c = 585.0 * f * BOWEN
    d_r = 4 * EMIS * SIGMA * (t + 273.15) ** 3
    return d_e / (d_e + d_c + d_r)


def withdrawal_m3s(it_mw, utilization, heat_factor, rise_c):
    """Water drawn through the exchangers, m3 s-1."""
    return it_mw * 1e6 * utilization * heat_factor / (RHO * CP * rise_c)


def forced_evaporation_m3s(it_mw, utilization, heat_factor, water_temp_c, wind_mph):
    """Extra evaporation from the lake caused by the returned heat, m3 s-1."""
    heat_w = it_mw * 1e6 * utilization * heat_factor
    return evaporation_ratio(water_temp_c, wind_mph) * heat_w / LV / RHO


def sample_parameters(n=200, seed=11):
    lo = np.array([v[0] for v in RANGES.values()])
    hi = np.array([v[1] for v in RANGES.values()])
    draws = lo + (hi - lo) * np.random.default_rng(seed).random((n, len(RANGES)))
    return {k: draws[:, i] for i, k in enumerate(RANGES)}


def run_lake(water_temps, n=200, seed=11, mw=100.0):
    """Monte Carlo for one lake, vectorized over runs. water_temps: daily long-term average surface temperatures (C).
    Returns an xarray Dataset in million gallons a day per `mw` of IT capacity: withdrawn (steady), forced evaporation
    by day of year, its average and its highest 30-day mean, and the evaporative share of the added heat."""
    t = np.asarray(water_temps, dtype=float)
    p = {k: v[:, None] for k, v in sample_parameters(n, seed).items()}
    heat_w = mw * 1e6 * p["utilization"] * p["heat_factor"]
    mgd = 86400 * model.GAL_M3 / 1e6
    withdrawn = (heat_w / (RHO * CP * p["rise_c"]))[:, 0] * mgd
    er = evaporation_ratio(t[None, :], p["wind_mph"])
    evap = er * heat_w / LV / RHO * mgd
    month = np.stack([np.convolve(e, np.ones(30) / 30, mode="valid") for e in evap])
    ds = xr.Dataset(
        {"withdrawn": (("run",), withdrawn), "evap_daily": (("run", "dayofyear"), evap),
         "evap_avg": (("run",), evap.mean(axis=1)), "evap_peak_month": (("run",), month.max(axis=1)),
         "er_avg": (("run",), er.mean(axis=1))},
        coords={"run": np.arange(n), "dayofyear": np.arange(1, t.size + 1)},
        attrs={"units": "million gallons per day", "it_capacity_mw": mw, "n_runs": n, "seed": seed})
    for k, v in p.items():
        ds[f"param_{k}"] = (("run",), v[:, 0])
    return ds


def glerl_temperatures(lake, source=GLERL):
    """NOAA GLERL long-term average daily surface temperature for one Great Lake, as a DataArray on day of year."""
    code = GLERL_CODES[lake]
    with urllib.request.urlopen(source, timeout=120) as r:
        text = r.read().decode()
    rows = [line.split(",") for line in text.splitlines()[2:] if line]
    days = sorted((int(d), float(v)) for c, d, v in rows if c == code)
    return xr.DataArray([v for _, v in days], coords={"dayofyear": [d for d, _ in days]}, dims="dayofyear",
                        name="water_temperature", attrs={"units": "degC", "lake": lake, "source": "NOAA GLERL"})
