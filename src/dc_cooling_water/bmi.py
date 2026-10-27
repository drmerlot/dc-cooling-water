"""CSDMS Basic Model Interface (BMI 2.0) for the cooling water model, so a coupling framework can drive it hour by
hour: set air temperature and humidity, call update(), read the water withdrawn, consumed and returned.

One data center is one scalar grid point. A run uses a single parameter set (the central values by default, any of
them overridable in the configuration file); the Monte Carlo ensemble lives in dc_cooling_water.ensemble.

Configuration (YAML):
    design: dry | assist | hybrid | towers      (default towers)
    it_capacity_mw: 100                          (IT capacity, MW)
    time_step: 3600                              (seconds; the model is calibrated hourly)
    end_time: 31536000                           (seconds; default one year)
    forcing: path/to/open_meteo.json             (optional; otherwise inputs come from set_value each step)
    parameters: {coc: 5, utilization: 0.8, ...}  (optional overrides of the central values)
"""
import numpy as np
import yaml
from bmipy import Bmi

from . import model
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


class CoolingWaterBmi(Bmi):
    """BMI 2.0 wrapper around dc_cooling_water.model."""

    _name = "Data center cooling water model"

    def __init__(self):
        self._values = {name: np.zeros(1) for name in list(INPUTS) + list(OUTPUTS)}
        self._time = 0.0
        self._dt = 3600.0
        self._end = 365 * 86400.0
        self._design = "towers"
        self._mw = 100.0
        self._params = model.central_parameters()
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
        self._params = {**model.central_parameters(), **{k: float(v) for k, v in (cfg.get("parameters") or {}).items()}}
        unknown = set(self._params) - set(model.PARAMETERS)
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
        self._values["data_center_cooling_water__withdrawal_volume_flux"][0] = w[0] * to_m3s
        self._values["data_center_cooling_water__consumption_volume_flux"][0] = c[0] * to_m3s
        self._values["data_center_cooling_water__return_volume_flux"][0] = r[0] * to_m3s
        self._values["atmosphere_bottom_air__wet_bulb_temperature"][0] = twb[0]

    def update(self):
        """Advance one time step: read the next forcing hour (if any) and recompute the water fluxes."""
        self._time += self._dt
        if self._forcing is not None:
            self._load_forcing()
        self._compute()

    def update_until(self, time):
        while self._time + self._dt <= time + 1e-9:
            self.update()

    def finalize(self):
        self._forcing = None

    # ---------- information ----------
    def get_component_name(self):
        return self._name

    def get_input_item_count(self):
        return len(INPUTS)

    def get_output_item_count(self):
        return len(OUTPUTS)

    def get_input_var_names(self):
        return tuple(INPUTS)

    def get_output_var_names(self):
        return tuple(OUTPUTS)

    # ---------- variables ----------
    def _units(self, name):
        if name in INPUTS:
            return INPUTS[name]
        if name in OUTPUTS:
            return OUTPUTS[name]
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
        if name not in INPUTS:
            raise KeyError(f"{name} is not an input variable")
        self._values[name][:] = src
        self._compute()

    def set_value_at_indices(self, name, inds, src):
        if name not in INPUTS:
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
