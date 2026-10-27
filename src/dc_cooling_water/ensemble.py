"""Monte Carlo ensembles of the cooling water model, returned as xarray Datasets, with dask across locations.

Runs are vectorized: a chunk of parameter draws is evaluated against the whole hourly record at once as
(runs, hours) arrays, then reduced to daily totals and per-run statistics before the next chunk, so memory stays
bounded for 30-year hourly records.
"""
from collections.abc import Mapping

import dask
import numpy as np
import xarray as xr

from . import model

STATS = ("avg", "peak_1in10", "record", "annual_mg")
FLOWS = ("withdrawn", "consumed", "returned")
L_PER_GAL = 3.78541


def _day_layout(time):
    """Day starts for np.add.reduceat, the day labels and the year of each day (local calendar days)."""
    days = np.asarray(time).astype("datetime64[D]")
    if days.size and np.any(np.diff(days).astype(int) < 0):
        raise ValueError("time must be sorted")
    starts = np.r_[0, np.flatnonzero(np.diff(days).astype(int)) + 1]
    labels = days[starts]
    years = labels.astype("datetime64[Y]").astype(int) + 1970
    return starts, labels, years


def _stats(daily, years):
    """Per-run statistics from daily volumes (runs, days) in million gallons: average day, 1-in-10-year peak day
    (90th percentile of annual maxima), record day and annual total."""
    uyears = np.unique(years)
    ann_max = np.stack([daily[:, years == y].max(axis=1) for y in uyears], axis=1)
    avg = daily.mean(axis=1)
    return np.stack([avg, np.quantile(ann_max, 0.9, axis=1), daily.max(axis=1), avg * 365.25], axis=1)


def _evaluate(tdb, twb, starts, years, design, params, mw):
    """Stats (runs, flows, stats), water use effectiveness (runs, 2) and daily volumes (runs, flows, days)."""
    p = {k: np.asarray(v, dtype=float).reshape(-1, 1) for k, v in params.items()}
    kw = mw * 1000.0 * p["utilization"]
    rates = model.hourly_rates(tdb, twb, design, p)
    daily = [np.add.reduceat(r * kw, starts, axis=1) / 1e6 for r in rates]
    stats = np.stack([_stats(d, years) for d in daily], axis=1)
    wue = np.stack([rates[1].mean(axis=1), rates[0].mean(axis=1)], axis=1) * L_PER_GAL
    return stats, wue, np.stack(daily, axis=1)


def run_ensemble(time, air_temperature, relative_humidity, mw=100.0, n=200, seed=7, designs=model.DESIGNS,
                 chunk=25, keep_daily=False):
    """Run the Monte Carlo ensemble for one location.

    time: hourly timestamps (local), air_temperature in C, relative_humidity in percent. mw is IT capacity in MW.
    Returns an xarray Dataset with per-run statistics (design, run, stat) for each flow, the central (mid-range)
    run, water use effectiveness, the sampled parameters, and optionally daily volumes (design, run, day).
    """
    tdb = np.asarray(air_temperature, dtype=float)
    rh = np.asarray(relative_humidity, dtype=float)
    twb = model.wet_bulb(tdb, rh)
    starts, labels, years = _day_layout(time)
    params = model.sample_parameters(n, seed)
    central = {k: np.array([v]) for k, v in model.central_parameters().items()}

    st = np.empty((len(designs), n, len(FLOWS), len(STATS)))
    wue = np.empty((len(designs), n, 2))
    cst = np.empty((len(designs), len(FLOWS), len(STATS)))
    daily = np.empty((len(designs), n, len(FLOWS), len(labels)), dtype=np.float32) if keep_daily else None
    for d, design in enumerate(designs):
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            s, w, dd = _evaluate(tdb, twb, starts, years, design, {k: v[sl] for k, v in params.items()}, mw)
            st[d, sl], wue[d, sl] = s, w
            if keep_daily:
                daily[d, sl] = dd
        cst[d] = _evaluate(tdb, twb, starts, years, design, central, mw)[0][0]

    coords = {"design": list(designs), "run": np.arange(n), "stat": list(STATS)}
    ds = xr.Dataset(coords=coords, attrs={"it_capacity_mw": mw, "n_runs": n, "seed": seed,
                                          "units": "avg, peak_1in10 and record in million gallons per day; annual_mg in million gallons"})
    for f, flow in enumerate(FLOWS):
        ds[flow] = (("design", "run", "stat"), st[:, :, f, :])
        ds[f"{flow}_central"] = (("design", "stat"), cst[:, f, :])
    ds["wue"] = (("design", "run"), wue[:, :, 0], {"units": "L per IT kWh", "basis": "consumed"})
    ds["wue_makeup"] = (("design", "run"), wue[:, :, 1], {"units": "L per IT kWh", "basis": "withdrawn (makeup)"})
    for k, v in params.items():
        ds[f"param_{k}"] = (("run",), v)
    if keep_daily:
        ds = ds.assign_coords(day=labels)
        for f, flow in enumerate(FLOWS):
            ds[f"{flow}_daily"] = (("design", "run", "day"), daily[:, :, f, :], {"units": "million gallons per day"})
    return ds


def summarize(ds, quantiles=(0.1, 0.5, 0.9)):
    """Quantiles across runs (and the central run) for each flow: (design, stat, quantile)."""
    out = xr.Dataset(attrs=ds.attrs)
    names = [f"p{int(q * 100)}" for q in quantiles] + ["central"]
    for flow in FLOWS:
        q = ds[flow].quantile(list(quantiles), dim="run").assign_coords(quantile=names[:-1])
        out[flow] = xr.concat([q, ds[f"{flow}_central"].expand_dims(quantile=["central"])], dim="quantile").transpose("design", "stat", "quantile")
    return out


def run_locations(weather: Mapping[str, xr.Dataset], scheduler="threads", **kwargs):
    """Run the ensemble for several locations in parallel with dask and stack them along a 'location' dimension.

    weather: {name: Dataset with a 'time' coordinate and 'air_temperature' (C) and 'relative_humidity' (%)}.
    Extra keyword arguments go to run_ensemble. numpy releases the GIL in the heavy array work, so threads scale.
    """
    tasks = [dask.delayed(run_ensemble)(w["time"].values, w["air_temperature"].values, w["relative_humidity"].values, **kwargs)
             for w in weather.values()]
    results = dask.compute(*tasks, scheduler=scheduler)
    return xr.concat(results, dim=xr.DataArray(list(weather), dims="location", name="location"))
