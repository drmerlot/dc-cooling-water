"""Publish ensemble results as zarr stores described by a static STAC catalog.

Each location becomes a STAC Item: a point geometry, the weather record's time span, the run settings, and one zarr
asset described with the STAC datacube extension (dimensions and variables), so tools like STAC Browser or
pystac-client can find the data and xarray can open it directly from the asset href.
"""
import datetime as dt
import os

import numpy as np
import pystac
import xarray as xr
from pystac.extensions.datacube import DatacubeExtension, Dimension, Variable

from .ensemble import FLOWS

ZARR_MEDIA_TYPE = "application/vnd+zarr"


def results_dataset(ens):
    """What gets published for one location: per-run statistics, the central run, water use effectiveness,
    sampled parameters, and the median daily volume across runs when daily values were kept."""
    keep = [v for v in ens.data_vars if not v.endswith("_daily")]
    out = ens[keep]
    if "withdrawn_daily" in ens:
        for flow in FLOWS:
            out[f"{flow}_daily_p50"] = ens[f"{flow}_daily"].astype("float64").median("run").astype("float32")
            out[f"{flow}_daily_p50"].attrs = {"units": "million gallons per day", "description": f"median {flow} volume across runs, by day"}
    return out


def _datacube(item, ds):
    cube = DatacubeExtension.ext(item.assets["data"], add_if_missing=True)
    dims = {}
    for name, size in ds.sizes.items():
        values = ds[name].values if name in ds.coords else None
        if name == "day":
            dims[name] = Dimension({"type": "temporal", "extent": [str(values[0]), str(values[-1])], "step": "P1D"})
        elif values is not None and values.dtype.kind in "UO":
            dims[name] = Dimension({"type": "other", "values": [str(v) for v in values]})
        else:
            dims[name] = Dimension({"type": "other", "extent": [0, int(size) - 1]})
    cube.dimensions = dims
    cube.variables = {name: Variable({"type": "data", "dimensions": list(var.dims), "unit": var.attrs.get("units", ""),
                                      "description": var.attrs.get("description", name)})
                      for name, var in ds.data_vars.items()}


def write_catalog(results, out_dir, title, description, start, end, href_base=None):
    """results: {location id: (lat, lon, ensemble Dataset)}. Writes a self-contained static STAC catalog
    (catalog.json, a collection, one item per location) with each location's zarr store beside its item. href_base, if given, is the public URL that
    out_dir will be served from (for absolute asset links); otherwise links are relative."""
    os.makedirs(out_dir, exist_ok=True)
    start_dt = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone.utc)
    lats = [r[0] for r in results.values()]
    lons = [r[1] for r in results.values()]
    collection = pystac.Collection(
        id="dc-cooling-water-results", title=title, description=description, license="BSD-3-Clause",
        extent=pystac.Extent(pystac.SpatialExtent([[min(lons), min(lats), max(lons), max(lats)]]),
                             pystac.TemporalExtent([[start_dt, end_dt]])))
    datasets = {}
    for loc, (lat, lon, ens) in results.items():
        ds = datasets[loc] = results_dataset(ens)
        item = pystac.Item(id=loc, geometry={"type": "Point", "coordinates": [lon, lat]}, bbox=[lon, lat, lon, lat],
                           datetime=None, start_datetime=start_dt, end_datetime=end_dt,
                           properties={"title": loc.replace("-", " ").title(), "it_capacity_mw": float(ens.attrs["it_capacity_mw"]),
                                       "n_runs": int(ens.attrs["n_runs"]), "seed": int(ens.attrs["seed"]),
                                       "designs": [str(d) for d in ens["design"].values]})
        href = f"{href_base.rstrip('/')}/{collection.id}/{loc}/{loc}.zarr" if href_base else f"./{loc}.zarr"
        item.add_asset("data", pystac.Asset(href=href, media_type=ZARR_MEDIA_TYPE, roles=["data"],
                                            title="Ensemble statistics (zarr v2, consolidated)"))
        _datacube(item, ds)
        collection.add_item(item)
    catalog = pystac.Catalog(id="dc-cooling-water", title=title, description=description)
    catalog.add_child(collection)
    catalog.normalize_hrefs(out_dir)
    # each zarr store sits next to its item's JSON, matching the asset's relative href
    for item in collection.get_items():
        datasets[item.id].to_zarr(os.path.join(os.path.dirname(item.get_self_href()), f"{item.id}.zarr"),
                                  mode="w", consolidated=True, zarr_format=2)
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return catalog


def open_item(item, chunks=None):
    """Open a STAC item's zarr asset as an xarray Dataset. The stores are small, so by default the arrays load as
    numpy (chunks=None); pass chunks={} or "auto" for lazy dask arrays."""
    return xr.open_zarr(item.assets["data"].get_absolute_href(), consolidated=True, chunks=chunks)
