import numpy as np
import pytest

pystac = pytest.importorskip("pystac")
pytest.importorskip("zarr")

from dc_cooling_water import run_ensemble  # noqa: E402
from dc_cooling_water.stac import open_item, write_catalog  # noqa: E402
from test_model import synthetic_weather  # noqa: E402


def test_catalog_round_trip(tmp_path):
    results = {}
    for i, (name, lat, lon) in enumerate((("north-site", 46.5, -87.4), ("south-site", 42.3, -83.6))):
        time, t, rh = synthetic_weather(400, seed=i)
        results[name] = (lat, lon, run_ensemble(time, t + 3 * i, rh, n=6, keep_daily=True))
    cat = write_catalog(results, str(tmp_path / "catalog"), "Test", "Synthetic", "2001-01-01", "2002-02-04")
    reopened = pystac.Catalog.from_file(str(tmp_path / "catalog" / "catalog.json"))
    items = {i.id: i for i in reopened.get_items(recursive=True)}
    assert set(items) == {"north-site", "south-site"}
    item = items["south-site"]
    assert item.geometry["coordinates"] == [-83.6, 42.3]
    assert item.ext.has("cube") and "withdrawn" in item.assets["data"].to_dict()["cube:variables"]
    ds = open_item(item)
    np.testing.assert_allclose(ds["withdrawn"].values, results["south-site"][2]["withdrawn"].values)
    assert ds["withdrawn_daily_p50"].sizes["day"] == 400
