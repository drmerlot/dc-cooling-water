"""Physics of the cooling water model: hourly heat rejected by a data center turned into water withdrawn, consumed
(evaporated) and returned, for four cooling designs.

Hour by hour:
  heat rejected (BTU) = IT energy x heat factor. The heat factor ramps with wet-bulb temperature from an economizer
      value (pumps, fans, losses) to a chiller value (compressor heat adds up to ~20% to tower load; DOE FEMP).
  evaporation = heat x evaporative ratio / latent heat, with the evaporative ratio from the Rutberg et al. (2011)
      dry-bulb relation used by USGS SIR 2013-5188 for cooling towers.
  drift = a fraction of circulating flow (EPA WaterSense at Work 6.3); circulating flow = heat / (500 x range in F).
  blowdown = evaporation / (cycles of concentration - 1)  (DOE FEMP BMP #10 water balance).
  loop top-up = the closed loop's small leaks, topped up and all of it lost: 1-5% a year of a loop holding
      2,000-6,000 gal per MW of IT capacity, spread over the IT energy.
  staff water = restrooms, kitchens and cleaning for the people on site, about 10% of it lost. It follows the
      people, who grow slowly with size, so it is not per kWh: staff_water() gives it at the campus's size, to be
      added to the cooling water (Water Ledger model 1.2).

Designs differ only in when they evaporate:
  dry     closed loop designed to use no water for cooling: loop top-ups only (plus the staff water every design has)
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
    "fill_gal_per_mw": (2000.0, 6000.0),  # closed loop volume, gal per MW of IT capacity [estimate; Lowell reported a 1.2 million gallon fill]
    "assist_on": (27.5, 31.0),         # dry-bulb C where assist starts (Microsoft: 85 F = 29.4 C)
    "assist_share": (0.10, 0.30),      # assist water as a share of the full evaporative rate [estimate]
    "hybrid_on": (21.0, 27.0),         # dry-bulb C where hybrid starts evaporating [estimate]
    "hybrid_full": (30.0, 33.0),       # dry-bulb C where hybrid is fully evaporative [estimate]
}
# Share of the loop volume lost to small leaks and topped up each year (NIH 2023: closed loops typically lose under
# 5% a year), all of it consumed. Drawn on its own random stream (where model 1.1 drew its base-use share), so the
# other parameters' draws are unchanged.
TOPUP = (0.01, 0.05)
PARAMETERS = tuple(RANGES) + ("topup",)

# Staff water [estimates]. People on site on a typical day = a fixed core plus a little per MW of IT capacity
# (Michigan's sales tax exemption needs at least 30 new jobs; hyperscale campuses employ tens to a few hundred
# people), times gallons per person a day (office buildings use about 10-20; more with a cafeteria). About 10% is
# used up and the rest goes down the drain (USGS consumptive-use coefficients for commercial use, Great Lakes Basin).
STAFF = {"core": (20.0, 60.0), "per_mw": (0.1, 0.5), "gpcd": (15.0, 30.0), "used": (0.05, 0.15)}
STAFF_PARAMETERS = tuple(f"staff_{k}" for k in STAFF)


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
    # loop top-up, gal/day per MW of capacity, as gal per IT kWh
    topup = p["fill_gal_per_mw"] * p["topup"] / 365.0 / (24 * 1000 * p["utilization"]) * np.ones_like(evap_full)
    withdrawn = evap + drift + blow + topup
    consumed = evap + drift + topup
    returned = blow
    return withdrawn, consumed, returned


def sample_parameters(n=200, seed=7):
    """Monte Carlo draws: independent uniforms over RANGES (run-major, so the draws match a per-run loop of
    rng.uniform calls) and topup from its own stream. Returns {name: array (n,)}."""
    lo = np.array([v[0] for v in RANGES.values()])
    hi = np.array([v[1] for v in RANGES.values()])
    u = np.random.default_rng(seed).random((n, len(RANGES)))
    draws = lo + (hi - lo) * u
    out = {k: draws[:, i] for i, k in enumerate(RANGES)}
    out["topup"] = TOPUP[0] + (TOPUP[1] - TOPUP[0]) * np.random.default_rng(11).random(n)
    return out


def central_parameters():
    """Every parameter at the middle of its range."""
    return {**{k: (a + b) / 2 for k, (a, b) in RANGES.items()}, "topup": sum(TOPUP) / 2}


def sample_staff(n=200, seed=13):
    """Monte Carlo draws of the staff water parameters on their own stream (run-major, as Water Ledger's
    staff_runs): {"staff_core", "staff_per_mw", "staff_gpcd", "staff_used": array (n,)}."""
    lo = np.array([v[0] for v in STAFF.values()])
    hi = np.array([v[1] for v in STAFF.values()])
    draws = lo + (hi - lo) * np.random.default_rng(seed).random((n, len(STAFF)))
    return {k: draws[:, i] for i, k in enumerate(STAFF_PARAMETERS)}


def central_staff():
    """Staff water parameters at the middle of their ranges."""
    return {f"staff_{k}": (a + b) / 2 for k, (a, b) in STAFF.items()}


def staff_water(s, mw):
    """Staff water at `mw` MW of IT capacity, million gallons per day: (withdrawn, consumed, returned). s holds
    the staff parameters (scalars or arrays, as from sample_staff or central_staff)."""
    g = (s["staff_core"] + s["staff_per_mw"] * mw) * s["staff_gpcd"] / 1e6
    return g, g * s["staff_used"], g * (1 - s["staff_used"])
