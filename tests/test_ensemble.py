import numpy as np
import xarray as xr

from dc_cooling_water import ensemble, model
from test_model import synthetic_weather


def loop_reference(time, t, rh, design, n, seed, mw):
    """Straightforward per-run, per-day loop: the shape of the original Water Ledger engine."""
    days = np.asarray(time).astype("datetime64[D]")
    uniq, idx = np.unique(days, return_inverse=True)
    years = uniq.astype("datetime64[Y]").astype(int)
    months = uniq.astype("datetime64[M]")
    twb = model.wet_bulb(t, rh)
    params = model.sample_parameters(n, seed)
    out = []
    for i in range(n):
        p = {k: v[i] for k, v in params.items()}
        w = model.hourly_rates(t, twb, design, p)[0]
        daily = np.bincount(idx, weights=w * mw * 1000 * p["utilization"]) / 1e6
        ann = [daily[years == y].max() for y in np.unique(years)]
        out.append([daily.mean(), np.quantile(ann, 0.9), daily.max(), daily.mean() * 365.25,
                    max(daily[months == m].mean() for m in np.unique(months)), max(daily[years == y].sum() for y in np.unique(years))])
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
    # cooling water scales linearly with capacity
    ds2 = ensemble.run_ensemble(time, t, rh, n=8, mw=200)
    np.testing.assert_allclose(ds2["withdrawn"], 2 * ds["withdrawn"], rtol=1e-12)


def test_summary_adds_staff_water_at_size():
    time, t, rh = synthetic_weather(400)
    ds = ensemble.run_ensemble(time, t, rh, n=8, mw=300, designs=("dry", "towers"))
    cooling, total = ensemble.summarize(ds, staff=False), ensemble.summarize(ds)
    w, c, r = model.staff_water(model.sample_staff(), 300)
    central = model.staff_water(model.central_staff(), 300)
    for flow, g, gc in (("withdrawn", w, central[0]), ("consumed", c, central[1]), ("returned", r, central[2])):
        added = total[flow] - cooling[flow]
        for q, v in (("p10", np.quantile(g, 0.1)), ("p50", np.quantile(g, 0.5)), ("p90", np.quantile(g, 0.9)), ("central", gc)):
            np.testing.assert_allclose(added.sel(quantile=q, stat=["avg", "peak_1in10", "record", "max_month"]), v, rtol=1e-9)
            np.testing.assert_allclose(added.sel(quantile=q, stat=["annual_mg", "worst_year_mg"]), v * 365.25, rtol=1e-9)
    # for closed loop, staff water is most of what is taken
    dry = total["withdrawn"].sel(design="dry", stat="avg", quantile="p50")
    assert float(cooling["withdrawn"].sel(design="dry", stat="avg", quantile="p50")) < 0.1 * float(dry)


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
