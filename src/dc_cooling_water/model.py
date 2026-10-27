"""Physics of the cooling water model: hourly heat rejected by a data center turned into water withdrawn, consumed
(evaporated) and returned, for four cooling designs.

Hour by hour:
  heat rejected (BTU) = IT energy x heat factor. The heat factor ramps with wet-bulb temperature from an economizer
      value (pumps, fans, losses) to a chiller value (compressor heat adds up to ~20% to tower load; DOE FEMP).
  evaporation = heat x evaporative ratio / latent heat, with the evaporative ratio from the Rutberg et al. (2011)
      dry-bulb relation used by USGS SIR 2013-5188 for cooling towers.
  drift = a fraction of circulating flow (EPA WaterSense at Work 6.3); circulating flow = heat / (500 x range in F).
  blowdown = evaporation / (cycles of concentration - 1)  (DOE FEMP BMP #10 water balance).
  base use = a small non-cooling use (staff sanitary use, some humidification, loop top-ups), part of it consumed.

Designs differ only in when they evaporate:
  dry     closed loop designed to use no water for cooling: base use only
  assist  closed loop with evaporative assist on the hottest hours (dry-bulb setpoint)
  hybrid  dry coolers most of the year, evaporative in warm weather (ramp between two dry-bulb setpoints)
  towers  open cooling towers, evaporative all year

Every function here is vectorized: parameters may be scalars or arrays shaped (runs, 1) against hourly weather
shaped (hours,), giving (runs, hours).
"""
import numpy as np

KWH_BTU = 3412.14
HFG = 1050.0           # latent heat near tower temperatures, BTU/lb
LB_GAL = 8.34
GAL_M3 = 264.172
DESIGNS = ("dry", "assist", "hybrid", "towers")

# (low, high) uniform ranges sampled in the Monte Carlo; the central run uses the midpoints. Sources in the README.
RANGES = {
    "utilization": (0.60, 0.90),       # average IT load / IT capacity [estimate]
    "hf_econ": (1.03, 1.08),           # heat factor with economizer cooling [estimate]
    "hf_chiller": (1.15, 1.25),        # heat factor with chillers running; compressor heat up to ~20% (DOE FEMP)
    "wb_econ": (8.0, 12.0),            # wet-bulb (C) below which the economizer carries the load [estimate]
    "wb_chiller": (16.0, 20.0),        # wet-bulb (C) above which chillers carry the load [estimate]
    "er_adj": (-0.05, 0.05),           # uncertainty on the Rutberg evaporative ratio (USGS +/-5-15% at baseload)
    "coc": (3.0, 7.0),                 # cycles of concentration (DOE FEMP: 2-4 typical, 6+ possible)
    "range_f": (8.0, 15.0),            # tower range, F (USGS SIR 2013-5188)
    "drift": (0.00005, 0.0005),        # drift share of circulating flow (EPA)
    "base_gpkwh": (0.001, 0.008),      # base use, gal per IT kWh (LBNL 2024: ~0-0.1 L/kWh for designs without water cooling)
    "assist_on": (27.5, 31.0),         # dry-bulb C where assist starts (Microsoft: 85 F = 29.4 C)
    "assist_share": (0.10, 0.30),      # assist water as a share of the full evaporative rate [estimate]
    "hybrid_on": (21.0, 27.0),         # dry-bulb C where hybrid starts evaporating [estimate]
    "hybrid_full": (30.0, 33.0),       # dry-bulb C where hybrid is fully evaporative [estimate]
}
# Share of base use that is consumed rather than returned [estimate]: sanitary use is ~10% consumed (USGS commercial
# coefficient, Great Lakes Basin), humidification and watering are fully consumed. Drawn on its own random stream.
BASE_USED = (0.10, 0.50)
PARAMETERS = tuple(RANGES) + ("base_used",)


def wet_bulb(t_c, rh):
    """Stull (2011) wet-bulb approximation, C. Stated for RH 5-99% and -20 to 50 C; least accurate in cold, very dry
    air (it can exceed the dry-bulb by up to ~0.9 C at -20 C and 10% RH), where it only nudges the heat factor."""
    rh = np.clip(rh, 5, 99)
    return (t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t_c + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def evaporative_ratio(t_c, adj=0.0):
    """Share of heat rejected by evaporation (Rutberg et al. 2011 dry-bulb relation, as in USGS SIR 2013-5188)."""
    er = 1 - (-0.000279 * t_c ** 3 + 0.00109 * t_c ** 2 - 0.345 * t_c + 26.7) / 100
    return np.clip(er + adj, 0.55, 1.0)


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def hourly_rates(tdb, twb, design, p):
    """Gallons per IT kWh for each hour: (withdrawn, consumed, returned).

    tdb, twb: dry- and wet-bulb temperature (C), shape (hours,). p: parameter dict whose values are scalars or arrays
    shaped (runs, 1). Withdrawn always equals consumed plus returned.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, not {design!r}")
    hf = p["hf_econ"] + (p["hf_chiller"] - p["hf_econ"]) * _ramp(twb, p["wb_econ"], p["wb_chiller"])
    btu = KWH_BTU * hf
    evap_full = btu * evaporative_ratio(tdb, p["er_adj"]) / HFG / LB_GAL
    circ = btu / (LB_GAL * p["range_f"])                    # gal circulated per kWh
    if design == "towers":
        w = np.ones_like(evap_full)
    elif design == "hybrid":
        w = _ramp(tdb, p["hybrid_on"], p["hybrid_full"]) * np.ones_like(evap_full)
    elif design == "assist":
        w = _ramp(tdb, p["assist_on"], p["assist_on"] + 4.0) * p["assist_share"] * np.ones_like(evap_full)
    else:
        w = np.zeros_like(evap_full)
    evap = evap_full * w
    drift = circ * p["drift"] * (w > 0)
    blow = evap / (p["coc"] - 1.0)
    base = p["base_gpkwh"] * np.ones_like(evap_full)
    withdrawn = evap + drift + blow + base
    consumed = evap + drift + base * p["base_used"]
    returned = blow + base * (1 - p["base_used"])
    return withdrawn, consumed, returned


def sample_parameters(n=200, seed=7):
    """Monte Carlo draws: independent uniforms over RANGES (run-major, so the draws match a per-run loop of
    rng.uniform calls) and base_used from its own stream. Returns {name: array (n,)}."""
    lo = np.array([v[0] for v in RANGES.values()])
    hi = np.array([v[1] for v in RANGES.values()])
    u = np.random.default_rng(seed).random((n, len(RANGES)))
    draws = lo + (hi - lo) * u
    out = {k: draws[:, i] for i, k in enumerate(RANGES)}
    out["base_used"] = BASE_USED[0] + (BASE_USED[1] - BASE_USED[0]) * np.random.default_rng(11).random(n)
    return out


def central_parameters():
    """Every parameter at the middle of its range."""
    return {**{k: (a + b) / 2 for k, (a, b) in RANGES.items()}, "base_used": sum(BASE_USED) / 2}
