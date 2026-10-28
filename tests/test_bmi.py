import json

import numpy as np
import pytest
import yaml

from dc_cooling_water import CoolingWaterBmi, model
from dc_cooling_water.bmi import INPUTS, OUTPUTS
from test_model import synthetic_weather

TEMP = "atmosphere_bottom_air__temperature"
RH = "atmosphere_bottom_air_water_vapor__relative_saturation"
WITHDRAWN = "data_center_cooling_water__withdrawal_volume_flux"
CONSUMED = "data_center_cooling_water__consumption_volume_flux"
RETURNED = "data_center_cooling_water__return_volume_flux"


def write_config(tmp_path, **cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_information_and_variables(tmp_path):
    m = CoolingWaterBmi()
    m.initialize(write_config(tmp_path, design="towers", it_capacity_mw=100))
    assert m.get_component_name()
    assert m.get_input_item_count() == len(m.get_input_var_names()) == 2
    assert m.get_output_item_count() == len(m.get_output_var_names()) == 4
    for name in list(INPUTS) + list(OUTPUTS):
        assert m.get_var_grid(name) == 0
        assert m.get_var_type(name) == "float64"
        assert m.get_var_itemsize(name) == 8 and m.get_var_nbytes(name) == 8
        assert m.get_var_location(name) == "node"
        assert m.get_var_units(name)
    assert m.get_grid_type(0) == "scalar" and m.get_grid_rank(0) == 0 and m.get_grid_size(0) == 1
    assert m.get_time_units() == "s" and m.get_time_step() == 3600 and m.get_start_time() == 0
    with pytest.raises(NotImplementedError):
        m.get_grid_shape(0, np.empty(0, dtype=int))
    m.finalize()


def test_set_value_drives_the_model(tmp_path):
    m = CoolingWaterBmi()
    m.initialize(write_config(tmp_path, design="towers", it_capacity_mw=100))
    out = np.empty(1)
    m.set_value(TEMP, np.array([30.0]))
    m.set_value(RH, np.array([50.0]))
    hot = m.get_value(WITHDRAWN, out).copy()
    m.set_value(TEMP, np.array([0.0]))
    cold = m.get_value(WITHDRAWN, out).copy()
    assert hot > cold > 0
    w, c, r = (m.get_value(n, np.empty(1))[0] for n in (WITHDRAWN, CONSUMED, RETURNED))
    assert w == pytest.approx(c + r, rel=1e-12)
    # matches the model directly: gal per kWh x kWh per second / gal per m3, plus staff water at 100 MW
    p = model.central_parameters()
    gal = model.hourly_rates(np.array([0.0]), model.wet_bulb(np.array([0.0]), 50.0), "towers", p)[0][0]
    staff = model.staff_water(model.central_staff(), 100)[0] * 1e6 / 86400 / model.GAL_M3
    assert w == pytest.approx(gal * 100 * 1000 * p["utilization"] / 3600 / model.GAL_M3 + staff, rel=1e-12)
    ptr = m.get_value_ptr(WITHDRAWN)
    m.set_value(TEMP, np.array([25.0]))
    assert ptr[0] == m.get_value(WITHDRAWN, np.empty(1))[0]     # the pointer stays live
    idx = m.get_value_at_indices(TEMP, np.empty(1), np.array([0]))
    assert idx[0] == 25.0
    m.set_value_at_indices(TEMP, np.array([0]), np.array([10.0]))
    assert m.get_value(TEMP, np.empty(1))[0] == 10.0
    with pytest.raises(KeyError):
        m.set_value(WITHDRAWN, np.array([1.0]))


def test_forcing_file_and_update_until(tmp_path):
    time, t, rh = synthetic_weather(10)
    forcing = tmp_path / "weather.json"
    forcing.write_text(json.dumps({"hourly": {"time": [str(x) for x in time.astype("datetime64[m]")],
                                              "temperature_2m": t.tolist(), "relative_humidity_2m": rh.tolist()}}))
    m = CoolingWaterBmi()
    m.initialize(write_config(tmp_path, design="hybrid", it_capacity_mw=50, forcing=str(forcing), parameters={"coc": 4}))
    assert m.get_end_time() == len(t) * 3600
    total = 0.0
    while m.get_current_time() < m.get_end_time():
        total += m.get_value(WITHDRAWN, np.empty(1))[0] * 3600
        m.update()
    p = {**model.central_parameters(), "coc": 4.0}
    gal = model.hourly_rates(t, model.wet_bulb(t, rh), "hybrid", p)[0]
    staff = model.staff_water(model.central_staff(), 50)[0] * 1e6 / 24     # gallons an hour
    expected = (gal * 50 * 1000 * p["utilization"] + staff).sum() / model.GAL_M3
    assert total == pytest.approx(expected, rel=1e-9)
    m.initialize(write_config(tmp_path, design="hybrid", forcing=str(forcing)))
    m.update_until(5 * 3600)
    assert m.get_current_time() == 5 * 3600
    assert m.get_value(TEMP, np.empty(1))[0] == pytest.approx(t[5])


def test_closed_loop_includes_staff_water_at_capacity(tmp_path):
    """A 300 MW closed-loop campus: loop top-ups plus staff water for the people on site, as Water Ledger 1.2."""
    m = CoolingWaterBmi()
    m.initialize(write_config(tmp_path, design="dry", it_capacity_mw=300))
    m.set_value(TEMP, np.array([20.0]))
    m.set_value(RH, np.array([60.0]))
    w, c, r = (m.get_value(n, np.empty(1))[0] * 86400 * model.GAL_M3 for n in (WITHDRAWN, CONSUMED, RETURNED))  # gal/day
    topup = 300 * 4000 * 0.03 / 365
    staff = (40 + 0.3 * 300) * 22.5
    assert w == pytest.approx(topup + staff, rel=1e-9)
    assert c == pytest.approx(topup + 0.1 * staff, rel=1e-9) and w == pytest.approx(c + r, rel=1e-12)
    m.initialize(write_config(tmp_path, design="dry", it_capacity_mw=300, parameters={"staff_gpcd": 30, "topup": 0.01}))
    w2 = m.get_value(WITHDRAWN, np.empty(1))[0] * 86400 * model.GAL_M3
    assert w2 == pytest.approx(300 * 4000 * 0.01 / 365 + 130 * 30, rel=1e-9)


def test_bad_configuration(tmp_path):
    m = CoolingWaterBmi()
    with pytest.raises(ValueError):
        m.initialize(write_config(tmp_path, design="spray-pond"))
    with pytest.raises(ValueError):
        m.initialize(write_config(tmp_path, parameters={"not_a_parameter": 1}))
