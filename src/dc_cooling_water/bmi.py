"""CSDMS Basic Model Interface (BMI 2.0) for the cooling water model, so a coupling framework can drive it hour by
hour: set air temperature and humidity, call update(), read the water withdrawn, consumed and returned.

One data center is one scalar grid point. A run uses a single parameter set (the central values by default, any of
them overridable in the configuration file); the Monte Carlo ensemble lives in dc_cooling_water.ensemble.

Configuration (YAML):
    design: dry | assist | hybrid | towers      (default towers)
    it_capacity_mw: 100                          (IT capacity, MW; sets the cooling load and the staff on site)
    time_step: 3600                              (seconds; the model is calibrated hourly)
    end_time: 31536000                           (seconds; default one year)
    forcing: path/to/open_meteo.json             (optional; otherwise inputs come from set_value each step)
    parameters: {coc: 5, utilization: 0.8, staff_gpcd: 20, ...}
                                                 (optional overrides of the central values: model.PARAMETERS and
                                                  model.STAFF_PARAMETERS)

Variables (all on the one scalar grid point, float64):
    input   atmosphere_bottom_air__temperature                         degC     air (dry-bulb) temperature
    input   atmosphere_bottom_air_water_vapor__relative_saturation     percent  relative humidity
    output  data_center_cooling_water__withdrawal_volume_flux          m3 s-1   water taken: cooling makeup (evaporation,
                                                                                drift, blowdown, loop top-ups) plus staff water
    output  data_center_cooling_water__consumption_volume_flux         m3 s-1   water lost: evaporation, drift, loop top-ups and
                                                                                the used-up share of staff water
    output  data_center_cooling_water__return_volume_flux              m3 s-1   water sent out: blowdown and the rest of staff water
    output  atmosphere_bottom_air__wet_bulb_temperature                degC     wet-bulb temperature (Stull 2011)
Staff water is a constant flow at the configured capacity (it follows the people on site, not the weather or the
IT load); withdrawal always equals consumption plus return.
"""
import numpy as np
import yaml
from bmipy import Bmi

from . import lake, model
from .weather import from_open_meteo_json

GAL_PER_M3 = model.GAL_M3

INPUTS = {
    "atmosphere_bottom_air__temperature": "degC",
    "atmosphere_bottom_air_water_vapor__relative_saturation": "percent",
}
OUTPUTS = {
    "data_center_cooling_water__withdrawal_volume_flux": "m3 s-1",
    "data_center_cooling_water__consumption_volume_flux": "m3 s-1",
    "data_center_cooling_water__return_volume_flux": "m3 s-1",
    "atmosphere_bottom_air__wet_bulb_temperature": "degC",
}


class _ScalarPointBmi(Bmi):
    """BMI plumbing shared by the components here: one point (a 'scalar' grid), float64 values held in 1-element
    arrays so get_value_ptr stays live, inputs set with set_value, and a fixed time step. Subclasses define INPUTS,
    OUTPUTS, _name, initialize() and _compute()."""

    INPUTS: dict = {}
    OUTPUTS: dict = {}

    def __init__(self):
        self._values = {name: np.zeros(1) for name in list(self.INPUTS) + list(self.OUTPUTS)}
        self._time = 0.0
        self._dt = 3600.0
        self._end = 365 * 86400.0

    def update(self):
        self._time += self._dt
        self._advance()
        self._compute()

    def _advance(self):
        """Load the next forcing step, if the component reads forcing itself."""

    def update_until(self, time):
        while self._time + self._dt <= time + 1e-9:
            self.update()

    def finalize(self):
        pass

    # ---------- information ----------
    def get_component_name(self):
        return self._name

    def get_input_item_count(self):
        return len(self.INPUTS)

    def get_output_item_count(self):
        return len(self.OUTPUTS)

    def get_input_var_names(self):
        return tuple(self.INPUTS)

    def get_output_var_names(self):
        return tuple(self.OUTPUTS)

    # ---------- variables ----------
    def _units(self, name):
        if name in self.INPUTS:
            return self.INPUTS[name]
        if name in self.OUTPUTS:
            return self.OUTPUTS[name]
        raise KeyError(name)

    def get_var_grid(self, name):
        self._units(name)
        return 0

    def get_var_type(self, name):
        return str(self._values[name].dtype)

    def get_var_units(self, name):
        return self._units(name)

    def get_var_itemsize(self, name):
        return self._values[name].itemsize

    def get_var_nbytes(self, name):
        return self._values[name].nbytes

    def get_var_location(self, name):
        self._units(name)
        return "node"

    # ---------- time ----------
    def get_current_time(self):
        return self._time

    def get_start_time(self):
        return 0.0

    def get_end_time(self):
        return self._end

    def get_time_units(self):
        return "s"

    def get_time_step(self):
        return self._dt

    # ---------- values ----------
    def get_value(self, name, dest):
        dest[:] = self._values[name]
        return dest

    def get_value_ptr(self, name):
        return self._values[name]

    def get_value_at_indices(self, name, dest, inds):
        dest[:] = self._values[name].take(inds)
        return dest

    def set_value(self, name, src):
        if name not in self.INPUTS:
            raise KeyError(f"{name} is not an input variable")
        self._values[name][:] = src
        self._compute()

    def set_value_at_indices(self, name, inds, src):
        if name not in self.INPUTS:
            raise KeyError(f"{name} is not an input variable")
        self._values[name].put(inds, src)
        self._compute()

    # ---------- grid: one scalar point ----------
    def get_grid_rank(self, grid):
        return 0

    def get_grid_size(self, grid):
        return 1

    def get_grid_type(self, grid):
        return "scalar"

    def get_grid_shape(self, grid, shape):
        raise NotImplementedError("scalar grid has no shape")

    def get_grid_spacing(self, grid, spacing):
        raise NotImplementedError("scalar grid has no spacing")

    def get_grid_origin(self, grid, origin):
        raise NotImplementedError("scalar grid has no origin")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("scalar grid has no coordinates")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("scalar grid has no coordinates")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("scalar grid has no coordinates")

    def get_grid_node_count(self, grid):
        return 1

    def get_grid_edge_count(self, grid):
        return 0

    def get_grid_face_count(self, grid):
        return 0

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("scalar grid has no edges")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("scalar grid has no faces")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("scalar grid has no faces")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("scalar grid has no faces")
class CoolingWaterBmi(_ScalarPointBmi):
    """BMI 2.0 wrapper around dc_cooling_water.model: hourly cooling-tower, hybrid and closed-loop water use."""

    INPUTS = INPUTS
    OUTPUTS = OUTPUTS

    _name = "Data center cooling water model"

    def __init__(self):
        super().__init__()
        self._design = "towers"
        self._mw = 100.0
        self._params = {**model.central_parameters(), **model.central_staff()}
        self._forcing = None

    # ---------- control ----------
    def initialize(self, config_file=None):
        cfg = {}
        if config_file:
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
        self._design = cfg.get("design", "towers")
        if self._design not in model.DESIGNS:
            raise ValueError(f"design must be one of {model.DESIGNS}")
        self._mw = float(cfg.get("it_capacity_mw", 100.0))
        self._dt = float(cfg.get("time_step", 3600.0))
        self._params = {**model.central_parameters(), **model.central_staff(),
                        **{k: float(v) for k, v in (cfg.get("parameters") or {}).items()}}
        unknown = set(self._params) - set(model.PARAMETERS) - set(model.STAFF_PARAMETERS)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        self._time = 0.0
        if cfg.get("forcing"):
            w = from_open_meteo_json(cfg["forcing"])
            self._forcing = (w["air_temperature"].values, w["relative_humidity"].values)
            self._end = float(cfg.get("end_time", len(self._forcing[0]) * self._dt))
            self._load_forcing()
        else:
            self._forcing = None
            self._end = float(cfg.get("end_time", 365 * 86400.0))
        self._compute()

    def _load_forcing(self):
        i = min(int(self._time // self._dt), len(self._forcing[0]) - 1)
        self._values["atmosphere_bottom_air__temperature"][0] = self._forcing[0][i]
        self._values["atmosphere_bottom_air_water_vapor__relative_saturation"][0] = self._forcing[1][i]

    def _compute(self):
        tdb = self._values["atmosphere_bottom_air__temperature"]
        rh = self._values["atmosphere_bottom_air_water_vapor__relative_saturation"]
        twb = model.wet_bulb(tdb, rh)
        w, c, r = model.hourly_rates(tdb, twb, self._design, self._params)
        kwh_per_s = self._mw * 1000.0 * self._params["utilization"] / 3600.0     # IT energy per second of model time
        to_m3s = kwh_per_s / GAL_PER_M3
        mgd_to_m3s = 1e6 / 86400.0 / GAL_PER_M3
        sw, sc, sr = model.staff_water(self._params, self._mw)
        self._values["data_center_cooling_water__withdrawal_volume_flux"][0] = w[0] * to_m3s + sw * mgd_to_m3s
        self._values["data_center_cooling_water__consumption_volume_flux"][0] = c[0] * to_m3s + sc * mgd_to_m3s
        self._values["data_center_cooling_water__return_volume_flux"][0] = r[0] * to_m3s + sr * mgd_to_m3s
        self._values["atmosphere_bottom_air__wet_bulb_temperature"][0] = twb[0]

    def _advance(self):
        if self._forcing is not None:
            self._load_forcing()

    def finalize(self):
        self._forcing = None




LAKE_INPUTS = {
    "lake_water_surface__temperature": "degC",
    "atmosphere_bottom_air_flow__speed": "m s-1",
}
LAKE_OUTPUTS = {
    "data_center_cooling_water__withdrawal_volume_flux": "m3 s-1",
    "data_center_cooling_water__return_volume_flux": "m3 s-1",
    "lake_water__forced_evaporation_volume_flux": "m3 s-1",
    "data_center_cooling_water__return_temperature_rise": "degC",
}


class LakeCoolingBmi(_ScalarPointBmi):
    """BMI 2.0 wrapper around dc_cooling_water.lake: once-through lake cooling, daily by default. Inputs are the
    lake's natural surface temperature and wind speed; outputs the water drawn and returned and the extra
    evaporation the returned heat causes.

    Configuration (YAML):
        it_capacity_mw: 100
        time_step: 86400                         (seconds)
        end_time: 31536000
        forcing: path/to/temps.csv               (optional; one water temperature per line, one per step)
        wind_speed: 4.5                          (m s-1; used when no wind is set; default the middle of the range)
        parameters: {utilization: 0.75, heat_factor: 1.14, rise_c: 8}
    """

    INPUTS = LAKE_INPUTS
    OUTPUTS = LAKE_OUTPUTS
    _name = "Once-through lake cooling model"

    def __init__(self):
        super().__init__()
        self._dt = 86400.0
        self._mw = 100.0
        self._params = {k: (a + b) / 2 for k, (a, b) in lake.RANGES.items()}
        self._forcing = None

    def initialize(self, config_file=None):
        cfg = {}
        if config_file:
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
        self._mw = float(cfg.get("it_capacity_mw", 100.0))
        self._dt = float(cfg.get("time_step", 86400.0))
        self._params = {**{k: (a + b) / 2 for k, (a, b) in lake.RANGES.items()},
                        **{k: float(v) for k, v in (cfg.get("parameters") or {}).items()}}
        unknown = set(self._params) - set(lake.RANGES)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        wind = cfg.get("wind_speed")
        self._values["atmosphere_bottom_air_flow__speed"][0] = float(wind) if wind is not None else self._params["wind_mph"] / lake.MPH_PER_MS
        self._time = 0.0
        self._forcing = None
        if cfg.get("forcing"):
            with open(cfg["forcing"]) as f:
                self._forcing = np.array([float(line) for line in f if line.strip()])
            self._end = float(cfg.get("end_time", len(self._forcing) * self._dt))
            self._advance()
        else:
            self._end = float(cfg.get("end_time", 365 * 86400.0))
        self._compute()

    def _advance(self):
        if self._forcing is not None:
            i = min(int(self._time // self._dt), len(self._forcing) - 1)
            self._values["lake_water_surface__temperature"][0] = self._forcing[i]

    def _compute(self):
        p = self._params
        t = self._values["lake_water_surface__temperature"]
        wind_mph = self._values["atmosphere_bottom_air_flow__speed"] * lake.MPH_PER_MS
        q = lake.withdrawal_m3s(self._mw, p["utilization"], p["heat_factor"], p["rise_c"])
        self._values["data_center_cooling_water__withdrawal_volume_flux"][0] = q
        self._values["data_center_cooling_water__return_volume_flux"][0] = q
        self._values["lake_water__forced_evaporation_volume_flux"][0] = lake.forced_evaporation_m3s(
            self._mw, p["utilization"], p["heat_factor"], t, wind_mph)[0]
        self._values["data_center_cooling_water__return_temperature_rise"][0] = p["rise_c"]

    def finalize(self):
        self._forcing = None
