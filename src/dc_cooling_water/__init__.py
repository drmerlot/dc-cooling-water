"""Weather-driven data center cooling water model with Monte Carlo uncertainty, xarray output, dask ensembles and a
CSDMS Basic Model Interface. Extracted from Water Ledger (waterledger.org/mi)."""
from .bmi import CoolingWaterBmi
from .ensemble import run_ensemble, run_locations, summarize
from .model import DESIGNS, RANGES, central_parameters, evaporative_ratio, hourly_rates, sample_parameters, wet_bulb
from .weather import fetch_open_meteo, from_open_meteo_json

__version__ = "0.1.0"
__all__ = ["CoolingWaterBmi", "run_ensemble", "run_locations", "summarize", "DESIGNS", "RANGES", "central_parameters",
           "evaporative_ratio", "hourly_rates", "sample_parameters", "wet_bulb", "fetch_open_meteo", "from_open_meteo_json"]
