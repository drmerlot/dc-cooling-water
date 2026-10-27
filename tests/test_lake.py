import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dc_cooling_water import LakeCoolingBmi, lake

REF = json.loads((Path(__file__).parent / "water_ledger_lakes.json").read_text())
WITHDRAWN = "data_center_cooling_water__withdrawal_volume_flux"
RETURNED = "data_center_cooling_water__return_volume_flux"
EVAP = "lake_water__forced_evaporation_volume_flux"
TEMP = "lake_water_surface__temperature"


@pytest.mark.parametrize("name", sorted(REF))
def test_reproduces_water_ledger_lakes(name):
    """Same NOAA GLERL temperatures, same seed: Water Ledger's published lake-source estimates."""
    ds = lake.run_lake(REF[name]["temps"], n=200)
    for var in ("withdrawn", "evap_avg", "evap_peak_month", "er_avg"):
        for q, pct in (("p10", 10), ("p50", 50), ("p90", 90)):
            np.testing.assert_allclose(np.percentile(ds[var].values, pct), REF[name]["res"][var][q], rtol=1e-12)


def test_evaporation_share_rises_with_temperature():
    t = np.linspace(0, 25, 26)
    er = lake.evaporation_ratio(t, 10.0)
    assert np.all(np.diff(er) > 0) and er.min() > 0.2 and er.max() < 1


def test_bmi_daily_run(tmp_path):
    temps = REF["Lake Michigan"]["temps"]
    (tmp_path / "t.csv").write_text("\n".join(str(t) for t in temps))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"it_capacity_mw": 100, "forcing": str(tmp_path / "t.csv"), "wind_speed": 4.47}))
    m = LakeCoolingBmi()
    m.initialize(str(cfg))
    assert m.get_time_step() == 86400 and m.get_end_time() == len(temps) * 86400
    evap = []
    while m.get_current_time() < m.get_end_time():
        w, r, e = (m.get_value(n, np.empty(1))[0] for n in (WITHDRAWN, RETURNED, EVAP))
        assert w == r and 0 < e < w
        evap.append(e)
        m.update()
    p = {k: (a + b) / 2 for k, (a, b) in lake.RANGES.items()}
    expected = lake.forced_evaporation_m3s(100, p["utilization"], p["heat_factor"], np.array(temps), 4.47 * lake.MPH_PER_MS)
    np.testing.assert_allclose(evap, expected, rtol=1e-12)
    assert np.argmax(evap) > 150        # more evaporation from warm summer water
    m.set_value(TEMP, np.array([20.0]))
    warm = m.get_value(EVAP, np.empty(1))[0]
    m.set_value(TEMP, np.array([4.0]))
    assert warm > m.get_value(EVAP, np.empty(1))[0]
    m.finalize()


def test_bmi_rejects_unknown_parameters(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"parameters": {"coc": 4}}))
    with pytest.raises(ValueError):
        LakeCoolingBmi().initialize(str(cfg))
