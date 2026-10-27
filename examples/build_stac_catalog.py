"""Run the eight Michigan locations and publish the results as zarr stores in a static STAC catalog (catalog/).
Downloads 30 years of hourly ERA5 weather per location on first run (cached in .cache/)."""
import shutil

import dc_cooling_water as dcw
from dc_cooling_water.stac import write_catalog

LOCATIONS = {  # Water Ledger's eight Michigan climate regions
    "southeast-ann-arbor": (42.25, -83.60), "southwest-kalamazoo": (42.29, -85.59),
    "west-grand-rapids": (42.88, -85.52), "central-lansing": (42.73, -84.55),
    "thumb-saginaw": (43.42, -83.95), "northwest-traverse-city": (44.76, -85.62),
    "northeast-alpena": (45.06, -83.43), "upper-peninsula-marquette": (46.54, -87.40),
}

if __name__ == "__main__":
    weather = {k: dcw.fetch_open_meteo(lat, lon, cache=f".cache/{lat}_{lon}.json") for k, (lat, lon) in LOCATIONS.items()}
    ens = dcw.run_locations(weather, mw=100, n=200, keep_daily=True)
    results = {k: (lat, lon, ens.sel(location=k).drop_vars("location")) for k, (lat, lon) in LOCATIONS.items()}
    shutil.rmtree("catalog", ignore_errors=True)
    cat = write_catalog(results, "catalog",
                        title="Data center cooling water, eight Michigan locations",
                        description=("Ensemble statistics from dc-cooling-water for a 100 MW (IT) campus under four cooling designs, "
                                     "200 Monte Carlo runs on 1995-2024 hourly ERA5 weather at each of eight Michigan locations."),
                        start="1995-01-01", end="2024-12-31T23:00:00")
    cat.validate_all()
    print("catalog written and validated:", sum(1 for _ in cat.get_items(recursive=True)), "items")
