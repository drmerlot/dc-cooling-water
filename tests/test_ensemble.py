import numpy as np
import xarray as xr

from dc_cooling_water import ensemble, model
from test_model import synthetic_weather


def loop_reference(time, t, rh, design, n, seed, mw):
    """Straightforward per-run, per-day loop: the shape of the original Water Ledger engine."""
    days = np.asarray(time).astype("datetime64[D]")
    uniq, idx = np.unique(days, return_inverse=True)
    years = uniq.astype("datetime64[Y]").astype(int)
    twb = model.wet_bulb(t, rh)
    params = model.sample_parameters(n, seed)
    out = []
    for i in range(n):
        p = {k: v[i] for k, v in params.items()}
        w = model.hourly_rates(t, twb, design, p)[0]
        daily = np.bincount(idx, weights=w * mw * 1000 * p["utilization"]) / 1e6
        ann = [daily[years == y].max() for y in np.unique(years)]
        out.append([daily.mean(), np.quantile(ann, 0.9), daily.max(), daily.mean() * 365.25])
    return np.array(out)


def test_vectorized_matches_loop():
    time, t, rh = synthetic_weather(3 * 365)
    ds = ensemble.run_ensemble(time, t, rh, mw=50, n=12, seed=3, designs=("hybrid", "towers"), chunk=5)
    for design in ("hybrid", "towers"):
        ref = loop_reference(time, t, rh, design, 12, 3, 50)
        np.testing.assert_allclose(ds["withdrawn"].sel(design=design).values, ref, rtol=1e-12)


def test_dataset_layout_and_summary():
    time, t, rh = synthetic_weather(400)
    ds = ensemble.run_ensemble(time, t, rh, n=8, keep_daily=True)
    assert set(ds["withdrawn"].dims) == {"design", "run", "stat"}
    assert ds.sizes["day"] == 400 and ds["withdrawn_daily"].dtype == np.float32
    # averages and annual totals add up; peak and record days need not (each flow peaks on its own day)
    add = ["avg", "annual_mg"]
    np.testing.assert_allclose(ds["withdrawn"].sel(stat=add), (ds["consumed"] + ds["returned"]).sel(stat=add), rtol=1e-9)
    s = ensemble.summarize(ds)
    assert list(s["withdrawn"].coords["quantile"].values) == ["p10", "p50", "p90", "central"]
    q = s["withdrawn"].sel(stat="avg")
    assert bool((q.sel(quantile="p10") <= q.sel(quantile="p90")).all())
    # scales linearly with capacity
    ds2 = ensemble.run_ensemble(time, t, rh, n=8, mw=200)
    np.testing.assert_allclose(ds2["withdrawn"], 2 * ds["withdrawn"], rtol=1e-12)


def test_run_locations_with_dask_matches_serial():
    weather = {}
    for i, name in enumerate(("north", "south")):
        time, t, rh = synthetic_weather(365, seed=i)
        weather[name] = xr.Dataset({"air_temperature": ("time", t + 3 * i), "relative_humidity": ("time", rh)}, coords={"time": time})
    a = ensemble.run_locations(weather, n=6, scheduler="synchronous")
    b = ensemble.run_locations(weather, n=6, scheduler="threads")
    assert list(b["location"].values) == ["north", "south"]
    xr.testing.assert_identical(a, b)
    # the warmer location uses more water with towers
    avg = b["withdrawn"].sel(design="towers", stat="avg").mean("run")
    assert float(avg.sel(location="south")) > float(avg.sel(location="north"))
