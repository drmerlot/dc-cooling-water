import numpy as np
import pytest

from dc_cooling_water import model


def synthetic_weather(days=730, seed=0):
    """Two years of hourly weather with a seasonal and daily cycle, Michigan-like."""
    rng = np.random.default_rng(seed)
    h = np.arange(days * 24)
    t = 9 + 14 * np.sin(2 * np.pi * (h / 24 - 110) / 365.25) + 5 * np.sin(2 * np.pi * (h % 24 - 9) / 24) + rng.normal(0, 2, h.size)
    rh = np.clip(70 - 15 * np.sin(2 * np.pi * (h % 24 - 9) / 24) + rng.normal(0, 8, h.size), 15, 100)
    time = np.datetime64("2001-01-01T00:00") + h.astype("timedelta64[h]")
    return time, t, rh


def test_wet_bulb_not_above_dry_bulb():
    # Stull (2011) is least accurate for cold, very dry air (about +0.9 C at -20 C and 10% RH); test its good range
    t = np.linspace(-10, 38, 49)
    for rh in (20, 50, 90):
        assert np.all(model.wet_bulb(t, rh) <= t + 0.3)


def test_evaporative_ratio_bounds_and_trend():
    t = np.linspace(-20, 40, 61)
    er = model.evaporative_ratio(t)
    assert er.min() >= 0.55 and er.max() <= 1.0
    assert er[-1] > er[0]          # warmer air: more of the heat leaves as evaporation


@pytest.mark.parametrize("design", model.DESIGNS)
def test_water_balance(design):
    _, t, rh = synthetic_weather(30)
    p = model.central_parameters()
    w, c, r = model.hourly_rates(t, model.wet_bulb(t, rh), design, p)
    np.testing.assert_allclose(w, c + r, rtol=1e-12)
    assert np.all(w > 0) and np.all(c >= 0) and np.all(r >= 0)


def test_dry_design_uses_base_only():
    _, t, rh = synthetic_weather(30)
    p = model.central_parameters()
    w, c, r = model.hourly_rates(t, model.wet_bulb(t, rh), "dry", p)
    np.testing.assert_allclose(w, p["base_gpkwh"])
    np.testing.assert_allclose(c, p["base_gpkwh"] * p["base_used"])


def test_designs_order_by_water_use():
    _, t, rh = synthetic_weather(365)
    p = model.central_parameters()
    tw = model.wet_bulb(t, rh)
    use = {d: model.hourly_rates(t, tw, d, p)[0].mean() for d in model.DESIGNS}
    assert use["dry"] < use["assist"] < use["hybrid"] < use["towers"]


def test_sampling_matches_per_run_loop():
    """Vectorized draws equal, to floating-point rounding, the per-run loop of rng.uniform calls the original model used."""
    n, seed = 50, 7
    rng = np.random.default_rng(seed)
    loop = [{k: rng.uniform(*v) for k, v in model.RANGES.items()} for _ in range(n)]
    vec = model.sample_parameters(n, seed)
    for k in model.RANGES:
        np.testing.assert_allclose(vec[k], [r[k] for r in loop], rtol=1e-14)
    for k, (lo, hi) in model.RANGES.items():
        assert vec[k].min() >= lo and vec[k].max() <= hi


def test_unknown_design():
    with pytest.raises(ValueError):
        model.hourly_rates(np.zeros(3), np.zeros(3), "spray-pond", model.central_parameters())
