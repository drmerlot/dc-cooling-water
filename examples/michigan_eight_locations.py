"""Run the ensemble for eight Michigan locations in parallel with dask and print the middle estimate and 80% range
for a 300 MW campus. Downloads 30 years of hourly ERA5 weather per location on first run (cached in .cache/)."""
import dc_cooling_water as dcw

LOCATIONS = {  # the eight climate regions Water Ledger uses
    "Southeast (Ann Arbor)": (42.25, -83.60), "Southwest (Kalamazoo)": (42.29, -85.59),
    "West (Grand Rapids)": (42.88, -85.52), "Central (Lansing)": (42.73, -84.55),
    "Thumb (Saginaw)": (43.42, -83.95), "Northwest (Traverse City)": (44.76, -85.62),
    "Northeast (Alpena)": (45.06, -83.43), "Upper Peninsula (Marquette)": (46.54, -87.40),
}

if __name__ == "__main__":
    weather = {name: dcw.fetch_open_meteo(lat, lon, cache=f".cache/{lat}_{lon}.json") for name, (lat, lon) in LOCATIONS.items()}
    ds = dcw.run_locations(weather, mw=300, n=200)
    for name in LOCATIONS:
        s = dcw.summarize(ds.sel(location=name))
        print(name)
        for design in dcw.DESIGNS:
            avg = s["withdrawn"].sel(design=design, stat="avg") * 1e6
            peak = s["withdrawn"].sel(design=design, stat="peak_1in10") * 1e6
            print(f"  {design:7s} average {float(avg.sel(quantile='p50')):>12,.0f} gal/day "
                  f"(80%: {float(avg.sel(quantile='p10')):,.0f} to {float(avg.sel(quantile='p90')):,.0f}); "
                  f"1-in-10 peak day {float(peak.sel(quantile='p50')):,.0f}")
