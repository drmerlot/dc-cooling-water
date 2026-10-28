"""Weather-driven data center cooling water model with Monte Carlo uncertainty, xarray output, dask ensembles and a
CSDMS Basic Model Interface. Extracted from Water Ledger (waterledger.org/mi)."""
from . import lake
from .bmi import CoolingWaterBmi, LakeCoolingBmi
from .ensemble import run_ensemble, run_locations, summarize
from .model import (DESIGNS, RANGES, STAFF, central_parameters, central_staff, evaporative_ratio, hourly_rates, sample_parameters,
                    sample_staff, staff_water, wet_bulb)
from .weather import fetch_open_meteo, from_open_meteo_json

__version__ = "0.2.0"
__all__ = ["CoolingWaterBmi", "LakeCoolingBmi", "lake", "run_ensemble", "run_locations", "summarize", "DESIGNS", "RANGES", "STAFF", "central_parameters",
           "central_staff", "evaporative_ratio", "hourly_rates", "sample_parameters", "sample_staff", "staff_water", "wet_bulb", "fetch_open_meteo", "from_open_meteo_json"]
