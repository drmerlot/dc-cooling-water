# dc-cooling-water

[![tests](https://github.com/drmerlot/dc-cooling-water/actions/workflows/tests.yml/badge.svg)](https://github.com/drmerlot/dc-cooling-water/actions/workflows/tests.yml)

A weather-driven model of the water a data center's cooling uses: withdrawn, consumed (evaporated) and returned,
hour by hour over a multi-decade weather record, for four cooling designs, with Monte Carlo uncertainty.

- **Vectorized ensembles** in numpy: hundreds of parameter draws evaluated against 30 years of hourly weather as
  `(runs, hours)` arrays, reduced to daily volumes and statistics chunk by chunk.
- **xarray output**: per-run statistics on `(design, run, stat)`, optional daily volumes on `(design, run, day)`, and
  quantile summaries.
- **dask** to run many locations in parallel.
- **CSDMS Basic Model Interfaces (BMI 2.0)** through `bmipy` for two components, so a coupling framework can drive
  them step by step: `CoolingWaterBmi` (hourly; air temperature and humidity in, water fluxes out) and
  `LakeCoolingBmi` (daily; lake temperature and wind in, withdrawal and forced lake evaporation out).
- **A STAC catalog of zarr results** for eight Michigan locations ([`catalog/`](catalog/catalog.json)), with the
  datacube extension describing every variable, so the outputs are discoverable and open straight into xarray.

It is the model behind [Water Ledger](https://waterledger.org/mi/), a public tool that estimates data center water
use for every Michigan city and township, extracted as a standalone package. It reproduces Water Ledger's published
estimates under its current model, 1.2 (see Validation).

## What changed in 0.2.0 (Water Ledger model 1.2)

Closed loop's water that is not for cooling was a base use per IT kWh (0.001–0.008 gal, 10–50% of it lost), which
grew with megawatts and counted humidification: about 26,000 gal/day taken and 6,200 lost for a 300 MW closed-loop
campus in West Michigan, four to seven times what Michigan applicants have projected. It is now two terms, as in Water Ledger 1.2:

- **Loop top-ups**, per kWh like the cooling water: 1–5% a year of a loop holding 2,000–6,000 gal per MW, all lost.
- **Staff water** for the people on site, added at the campus's size because it follows the staff, not the megawatts.

Humidification is no longer counted, since these designs recirculate indoor air. The same campus now takes about
2,700 gal/day (80% range about 1,750–4,400) and loses about 340. Other changes: `RANGES["base_gpkwh"]` and `base_used` are
replaced by `fill_gal_per_mw` and `topup` (the other parameters' draws are unchanged); new `sample_staff`,
`central_staff` and `staff_water`; `summarize()` adds staff water by default (`staff=False` gives cooling water
alone); two new statistics, `max_month` and `worst_year_mg`; `CoolingWaterBmi`'s fluxes include staff water at
`it_capacity_mw`; the STAC catalog is rebuilt.

## Install

```bash
pip install git+https://github.com/drmerlot/dc-cooling-water
```

Python 3.10 or newer; depends on numpy, xarray, dask, bmipy and pyyaml.

## Quickstart

```python
import dc_cooling_water as dcw

# 30 years of hourly ERA5 temperature and humidity for Grand Rapids, Michigan (cached after the first download)
w = dcw.fetch_open_meteo(42.88, -85.52, cache=".cache/grand_rapids.json")

ds = dcw.run_ensemble(w.time.values, w.air_temperature.values, w.relative_humidity.values, mw=300, n=200)
s = dcw.summarize(ds)       # (design, stat, quantile): p10, p50, p90 and the central run, staff water included
s["withdrawn"].sel(design="towers", stat="peak_1in10")      # million gallons on a 1-in-10-year peak day
s["withdrawn"].sel(design="dry", stat="avg", quantile="p50") * 1e6    # closed loop: about 2,700 gallons a day
```

`run_ensemble` returns per-run cooling water (with loop top-ups), which scales with capacity; `summarize` adds staff
water at the campus's size.

Several locations in parallel with dask, stacked on a `location` dimension:

```python
ds = dcw.run_locations({"Grand Rapids": w_gr, "Marquette": w_mq}, mw=300, n=200)
```

Coupled through the BMI:

```python
import numpy as np
from dc_cooling_water import CoolingWaterBmi

m = CoolingWaterBmi()
m.initialize("config.yaml")     # design, it_capacity_mw, time_step, end_time, optional forcing file and parameters
m.set_value("atmosphere_bottom_air__temperature", np.array([29.0]))
m.set_value("atmosphere_bottom_air_water_vapor__relative_saturation", np.array([55.0]))
m.update()
m.get_value("data_center_cooling_water__withdrawal_volume_flux", np.empty(1))      # m3 s-1
```

`CoolingWaterBmi` variables (one scalar grid point; staff water is a constant flow at `it_capacity_mw`, central
values unless overridden in `parameters`):

| | Name | Units | |
|---|---|---|---|
| in | `atmosphere_bottom_air__temperature` | degC | air temperature |
| in | `atmosphere_bottom_air_water_vapor__relative_saturation` | percent | relative humidity |
| out | `data_center_cooling_water__withdrawal_volume_flux` | m3 s-1 | evaporation, drift, blowdown and loop top-ups, plus staff water |
| out | `data_center_cooling_water__consumption_volume_flux` | m3 s-1 | evaporation, drift, loop top-ups and the used-up share of staff water |
| out | `data_center_cooling_water__return_volume_flux` | m3 s-1 | blowdown and the rest of staff water |
| out | `atmosphere_bottom_air__wet_bulb_temperature` | degC | wet-bulb temperature (Stull 2011) |

Lake-source (once-through) cooling, from NOAA GLERL lake temperatures:

```python
from dc_cooling_water import lake
temps = lake.glerl_temperatures("Lake Michigan")           # long-term daily average surface temperature
ds = lake.run_lake(temps.values, n=200, mw=300)             # withdrawn, forced evaporation by day of year, ...
```

Open the published results through STAC:

```python
import pystac
from dc_cooling_water.stac import open_item

catalog = pystac.Catalog.from_file("https://raw.githubusercontent.com/drmerlot/dc-cooling-water/main/catalog/catalog.json")
item = next(catalog.get_items("west-grand-rapids", recursive=True))
ds = open_item(item)        # per-run statistics, parameters and median daily volumes, as xarray
```

Browse it in [STAC Browser](https://radiantearth.github.io/stac-browser/#/external/raw.githubusercontent.com/drmerlot/dc-cooling-water/main/catalog/catalog.json).
See `examples/` for an eight-location Michigan run, the catalog build and a BMI coupling loop.

## Method

Each hour, for each parameter draw:

- **Heat rejected** = IT energy × heat factor. The heat factor ramps with wet-bulb temperature from an economizer
  value (1.03–1.08: pumps, fans, losses) to a chiller value (1.15–1.25; compressor heat adds up to about 20% to
  tower load, DOE FEMP). Wet-bulb from Stull (2011).
- **Evaporation** = heat × evaporative ratio ÷ latent heat (1,050 BTU/lb). The evaporative ratio is the Rutberg et
  al. (2011) dry-bulb relation, as used by the USGS for cooling towers (Diehl et al. 2013).
- **Drift** = a share of circulating flow (EPA WaterSense at Work); circulating flow = heat ÷ (500 × range °F).
- **Blowdown** = evaporation ÷ (cycles of concentration − 1) (DOE FEMP Best Management Practice #10).
- **Loop top-up**: the closed loop's small leaks, topped up and all of it lost: 1–5% a year (NIH 2023: closed loops
  typically lose under 5%) of a loop holding 2,000–6,000 gallons per MW of IT capacity (an estimate; a one-time
  1.2 million gallon fill was reported locally for Microsoft's Lowell Township proposal), spread over the IT energy.

Withdrawn = evaporation + drift + blowdown + top-up; consumed = evaporation + drift + top-up; returned = blowdown.
The designs differ only in when evaporation runs: `dry` (closed loop, designed to use no water for cooling: top-ups
only), `assist` (closed loop with evaporative assist above a dry-bulb setpoint; Microsoft publishes 85 °F),
`hybrid` (dry coolers, evaporative in warm weather, ramping between two dry-bulb setpoints) and `towers`
(evaporative all year).

**Staff water** (restrooms, kitchens, cleaning) is not per kWh: it follows the people on site, who grow slowly with
size. People on a typical day = 20–60 plus 0.1–0.5 per MW of IT capacity (Michigan's data center sales tax
exemption, MCL 205.54ee, needs at least 30 new jobs; JLARC 2024 puts a typical Virginia building at about 50 workers), times 15–30 gallons a person a day; 5–15% of it is used up and the rest goes
down the drain (USGS consumptive-use coefficients for the Great Lakes Basin). All four are estimates. It is added
at the campus's size: `summarize()` adds its 10th/50th/90th percentiles over 200 draws to the cooling water's, as
Water Ledger does, so the two ranges add end to end and the combined range is a little wide. No humidification is
counted: the designs modeled recirculate indoor air, so outside air comes in only for ventilation; a direct
outside-air design would need some in Michigan's dry winters and is not modeled.

**Uncertainty.** 14 parameters are drawn independently from uniform ranges (`RANGES` in `model.py`, each with its
source), plus the yearly top-up share on its own stream; 200 runs by default, seeded and reproducible, plus a
central run at mid-range. Staff water has its own 200 draws (`STAFF`, seed 13). Per run: average day, highest
month's average day, 1-in-10-year peak day (90th percentile of annual maximum days), record day, average year and
highest calendar year. The reported range is the 10th to 90th percentile across runs. It reflects parameter
uncertainty only: every run sees the same weather record, so it is not a confidence interval and leaves out climate
change and designs outside the ranges.

## Lake-source cooling

Water drawn through the heat exchangers is heat ÷ (density × specific heat × temperature rise) and nearly all of it
returns, warmer. The returned heat drives extra evaporation from the lake, estimated with the linearized
heat-balance method the USGS uses for once-through power plants (Diehl et al. 2013, after Harbeck 1964 and Ward
1980): the share of added heat leaving as evaporation is dE ÷ (dE + dC + dR), where evaporation and conduction share
Ward's wind function and radiation does not. Natural surface temperatures are NOAA GLERL long-term daily averages.
Temperature rise (6–10 °C) and wind (5–15 mph) are estimates.

## Validation

- **Reproduces Water Ledger model 1.2.** Run on the same 30 years of ERA5 weather (Grand Rapids area, 1995–2024)
  with the same seeds, all 224 published West Michigan cooling statistics (4 designs × 14 statistics × p10, p50, p90
  and central, per 100 MW) match Water Ledger's
  [published estimates](https://waterledger.org/mi/data/estimates.json) at their published precision (5 decimal
  places), as do the water use effectiveness and utilization ranges; the 200 staff water draws match the published
  ones; and a 300 MW campus, staff water added, matches what Water Ledger shows: closed loop takes about 2,700
  gallons a day and loses about 340 (`tests/test_regression.py`; also run in CI, which downloads the record itself).
  Against the engine's unrounded output the 224 statistics agree to a relative difference under 1e-14.
- **Lake-source cooling reproduces Water Ledger** for Lakes Michigan, Superior, Huron and Erie, to a relative
  difference under 1e-15 (`tests/test_lake.py`, data included).
- **Against published values** (West Michigan, 80% range):
  - Cooling towers: 1.49–1.81 L of makeup water per IT kWh. LBNL (2024) reports a median of about 2.0–2.2 for large
    tower-based facilities across U.S. climates; lower is expected in a cool climate.
  - Hybrid: 0.03–0.09 L consumed per kWh, at or below the efficient end of Lei et al. (2025) for climate zone 5A
    (about 0.1–0.7).
  - Closed loop: about 0.002 L taken per IT kWh at 300 MW, staff water included, inside LBNL's 0–0.1 for designs
    without water cooling and near the bottom, since humidification is not counted.
  - One Michigan proposal (Van Buren Township, a 1 GW DTE electricity contract, three buildings, one with cooling
    towers): the developer estimated 2–3.6 million gallons a day. Sized as Water Ledger now sizes it (about 287 MW of
    computing on towers and 538 MW closed loop, Southeast Michigan), the model gives about 2.25 on an average day and
    2.8 on a 1-in-10-year peak day.
- **Not done:** calibration against metered data center water use. No metered Michigan data is public, so these are
  checks against published ranges, not a calibrated model.

## Performance

On an Apple M3 Max laptop, one location (200 runs × 4 designs × 262,992 hours, six statistics per flow) takes about
5.2 s, against 15.7 s for Water Ledger's per-run loop. Eight Michigan locations take about 42 s serially and 23 s
with dask threads.

## Limitations

- The heat-factor and setpoint ranges are estimates; chillers are a ramp on wet-bulb, not a coefficient-of-performance model.
- Average utilization is drawn from 60–90% of IT capacity; AI training loads can run higher.
- Staff numbers, water per person and loop volume are estimates; no Michigan campus has published metered use.
- Landscape watering and humidification for designs that cool with outside air are not counted.
- The evaporative ratio comes from power-plant cooling towers.
- No build-out phasing; capacity is constant over the record.
- Stull's wet-bulb formula is least accurate in cold, very dry air, where it only nudges the heat factor.
- Weather is ERA5 reanalysis at one grid point (about 0.25°), which smooths local extremes slightly.

## Tests

```bash
pip install -e ".[test]"
pytest                                                   # model, ensemble, lake, BMI and STAC tests
DCW_FETCH=1 pytest tests/test_regression.py              # downloads the West Michigan record and checks reproduction
```

## References

- Diehl, T. H., Harris, M. A., Murphy, J. C., Hutson, S. S., and Ladd, D. E. (2013). Methods for estimating water
  consumption for thermoelectric power plants in the United States. USGS Scientific Investigations Report 2013-5188.
  (Source of the cooling-tower evaporative ratio and of the forced-evaporation heat balance after Harbeck 1964 and
  Ward 1980.)
- Rutberg, M. J., Delgado, A., Herzog, H. J., and Ghoniem, A. F. (2011). A system-level generic model of water use
  at power plants and its application to regional water use estimation. ASME International Mechanical Engineering
  Congress and Exposition.
- Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. Journal of Applied Meteorology
  and Climatology 50: 2267–2269.
- U.S. DOE Federal Energy Management Program. Best Management Practice #10: Cooling Tower Management.
- U.S. EPA (2012). WaterSense at Work: Best Management Practices for Commercial and Institutional Facilities.
- Shehabi, A., et al. (2024). 2024 United States Data Center Energy Usage Report. Lawrence Berkeley National Laboratory.
- Lei, N., Lu, J., Shehabi, A., and Masanet, E. (2025). The water use of data center workloads: a review and
  assessment of key determinants. Resources, Conservation and Recycling 219.
- NIH Office of Research Facilities (August 2023). Closed Loop Water Treatment. Technical Bulletin (closed systems
  typically need makeup water of less than 5% of their volume a year).
- Joint Legislative Audit and Review Commission, Virginia (2024). Data Centers in Virginia 2024. Report 598.
- Shaffer, K. H., and Runkle, D. L. (2007). Consumptive water-use coefficients for the Great Lakes Basin and
  climatically similar areas. USGS Scientific Investigations Report 2007-5197.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society
  146: 1999–2049. Accessed through the Open-Meteo archive API.
- NOAA Great Lakes Environmental Research Laboratory. Great Lakes long-term average surface water temperature (GLSEA).
- Hutton, E. W. H., Piper, M. D., and Tucker, G. E. (2020). The Basic Model Interface 2.0. Journal of Open Source
  Software 5(51): 2317.

## License

BSD-3-Clause.
