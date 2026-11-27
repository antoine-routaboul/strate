from datetime import date

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from strate import entities, fusion, geo
from strate.modules.optical import Optical, reflectance


# -- graphe d'entités ------------------------------------------------------------------
def test_graph_is_consistent():
    g = entities.load()
    assert len(g.sites) == 10
    for s in g.sites.values():
        assert -90 <= s.lat <= 90 and -180 <= s.lon <= 180
        assert s.coord_source.startswith("https://")
        assert abs(sum(o.share for o in s.owners) - 1) < 1e-9
        assert g.owners_of(s.id)


# -- géométrie ----------------------------------------------------------------------------
def test_site_grid():
    g = geo.site_grid(-86.928214, 49.677409, 4000, 20)
    assert g.crs == "EPSG:32616"  # Ontario ouest : UTM 16N
    assert g.n == 400
    assert g.area_km2 == pytest.approx(64.0)


def test_read_returns_full_window(tmp_path, monkeypatch):
    """Régression : la lecture d'UNE bande doit rendre tout le carré, pas une seule ligne."""
    monkeypatch.setattr(geo.pc, "sign", lambda h: h)
    grid = geo.site_grid(-86.928214, 49.677409, 1000, 20)  # carré de 100 x 100 px
    path = tmp_path / "b.tif"
    data = np.arange(1, 301 * 301 + 1, dtype="uint16").reshape(301, 301)
    with rasterio.open(path, "w", driver="GTiff", width=301, height=301, count=1, dtype="uint16",
                       crs=grid.crs, transform=from_origin(grid.x0 - 1000, grid.y1 + 1000, 20, 20)) as dst:
        dst.write(data, 1)
    one = geo.read(str(path), grid, Resampling.nearest)
    assert one.shape == (1, 100, 100)
    # le coin nord-ouest du site est à 50 px du bord de l'image source
    assert one[0, 0, 0] == data[50, 50]
    assert one[0, 99, 99] == data[149, 149]
    assert geo.read(str(path), grid, Resampling.nearest, [1]).shape == (1, 100, 100)


# -- optique ----------------------------------------------------------------------------
def test_reflectance_baseline_offset():
    dn = np.array([0, 1000, 3000], dtype="uint16")
    old = reflectance(dn, "03.01")
    new = reflectance(dn, "04.00")
    assert np.isnan(old[0]) and np.isnan(new[0])  # 0 = hors image
    assert old[2] == pytest.approx(0.3)
    assert new[2] == pytest.approx(0.2)  # baseline ≥ 04.00 : DN décalés de +1000


def _comp(n=10, bare_cols=0, water_cols=0):
    red = np.full((n, n), 0.04, "float32")
    nir = np.full((n, n), 0.30, "float32")  # végétation : NDVI ≈ 0.76
    swir = np.full((n, n), 0.12, "float32")
    red[:, :bare_cols], nir[:, :bare_cols], swir[:, :bare_cols] = 0.25, 0.28, 0.35  # sol nu
    water = np.zeros((n, n), "float32")
    if water_cols:
        water[:, n - water_cols:] = 1.0
        red[:, n - water_cols:], nir[:, n - water_cols:], swir[:, n - water_cols:] = 0.03, 0.02, 0.01
    return {"valid": np.ones((n, n), "uint8"), "red": red, "nir": nir, "swir": swir, "water": water}


def test_measures_areas():
    prev = _comp(bare_cols=2, water_cols=1)
    cur = _comp(bare_cols=5, water_cols=1)
    m = Optical.measures(cur, prev, pixel_km2=1.0)
    assert m["quality"] == 1.0
    assert m["bare_km2"] == pytest.approx(50)
    assert m["water_km2"] == pytest.approx(10)
    assert m["new_bare_km2"] == pytest.approx(30)  # 3 colonnes défrichées
    assert m["change_km2"] == pytest.approx(30)


def test_scl_water_without_water_spectrum_is_not_water():
    c = _comp()
    c["water"][:] = 1.0  # le SCL dit « eau » partout, mais le spectre est celui de la végétation
    assert Optical.measures(c, None, 1.0)["water_km2"] == 0


def test_measures_skips_cloudy_month():
    c = _comp(bare_cols=5)
    c["valid"][:6] = 0  # 60 % du carré sous les nuages
    assert Optical.measures(c, None, 1.0) == {"quality": pytest.approx(0.4)}


# -- convergence --------------------------------------------------------------------------
def test_robust_z_flags_spike_only():
    s = pd.Series([10, 11, 9, 10, 12, 10, 11, 9, 10, 30], dtype=float)
    z = fusion.robust_z(s)
    assert z.iloc[:6].isna().all()  # pas assez d'historique
    assert abs(z.iloc[8]) < 2
    assert z.iloc[9] > 5


def test_dossier_levels(monkeypatch):
    dates = pd.date_range("2022-01-01", periods=20, freq="MS")
    rows = []
    for i, d in enumerate(dates):
        sat_spike = i >= 18  # satellite : anomalie sur 2 observations de suite (persistance)
        att_spike = i == 19
        rows += [
            dict(entity="greenstone", date=d, signal="bare_km2", value=5 + (3 if sat_spike else i % 2 * 0.1),
                 unit="km2", quality=1, source="t"),
            dict(entity="greenstone", date=d, signal="site_views",
                 value=100 * (10 if att_spike else 1) + i % 3, unit="vues", quality=1, source="t"),
        ]
    monkeypatch.setattr(fusion, "signals", lambda sid: pd.DataFrame(rows))
    d = fusion.dossier("greenstone")
    assert d["level"].iloc[-1] == 2  # satellite ET attention s'écartent ensemble
    assert (d["level"].iloc[:-1] == 0).all()
    assert "surface nue en hausse" in d["explication"].iloc[-1]


# -- API ------------------------------------------------------------------------------------
def test_api_widgets_and_sites():
    from fastapi.testclient import TestClient

    from strate.api import app

    c = TestClient(app)
    w = c.get("/widgets.json").json()
    assert {"strate_radar", "strate_dossier"} <= set(w)
    for spec in w.values():
        assert spec["endpoint"].startswith("api/")
        assert c.get("/" + spec["endpoint"], params={"site": "greenstone"}).status_code == 200
    assert len(c.get("/api/sites").json()) == 10
    assert c.get("/api/dossier", params={"site": "nope"}).status_code == 404


def test_months_helper():
    from strate.core import months

    assert months(date(2025, 11, 5), date(2026, 2, 1)) == [
        date(2025, 11, 1), date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)]


# -- catalogue mondial ------------------------------------------------------------------------
def test_match_by_distance_and_by_name():
    from strate.catalog import match

    base = pd.DataFrame({"name": ["Escondida", "Spence", "Zaldívar"], "aliases": [None, None, None],
                         "lat": [-24.27, -22.80, -24.21], "lon": [-69.07, -69.27, -69.08]})
    other = pd.DataFrame({
        "name": ["Minera Escondida", "Unrelated pit", "Spence mine", "Far away"],
        "lat": [-24.30, -22.801, -22.86, 10.0],  # Escondida à ~3,5 km mais même nom
        "lon": [-69.08, -69.271, -69.27, 10.0],
    })
    m = match(base, other)
    assert m.iloc[0] == 0  # nom ressemblant à < 10 km
    assert m.iloc[1] == 1 or m.iloc[2] == 1  # Spence n'est pris qu'une fois
    assert not (m.iloc[1] == 1 and m.iloc[2] == 1)
    assert m.iloc[2] == 1  # le nom passe devant la distance
    assert m.iloc[3] == -1


def test_footprints_attach_and_orphans():
    import geopandas as gpd
    from shapely.geometry import box

    from strate.catalog import footprints

    sites = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    poly = gpd.GeoDataFrame(
        {"AREA": [1.0, 0.5, 2.0, 0.3], "COUNTRY_NAME": ["X"] * 4},
        geometry=[box(0.001, 0.001, 0.005, 0.005),  # collé au site
                  box(0.006, 0.0, 0.009, 0.004),    # collé au site
                  box(1.0, 1.0, 1.01, 1.01),        # orphelin A
                  box(1.012, 1.0, 1.015, 1.01)],    # orphelin A (à < 1 km)
        crs=4326)
    fp, unnamed = footprints(sites, poly)
    assert fp.loc[0, "footprint_km2"] == pytest.approx(1.5)
    assert fp.loc[0, "footprint_polygons"] == 2
    assert len(unnamed) == 1
    assert unnamed.loc[0, "footprint_km2"] == pytest.approx(2.3)


def test_seasonal_z_ignores_unseen_seasons():
    # 3 étés d'historique, puis un mois d'avril jamais vu : pas de z (pas de fausse alerte de fonte)
    idx = pd.to_datetime([f"{y}-{m:02d}-01" for y in (2021, 2022, 2023) for m in (7, 8, 9)] + ["2024-04-01",
                                                                                                  "2024-08-01"])
    s = pd.Series([5.0, 5.1, 4.9] * 3 + [9.0, 8.0], index=idx)
    z = fusion.seasonal_z(s)
    assert np.isnan(z.loc["2024-04-01"])
    assert z.loc["2024-08-01"] > 5


def test_persistence_drops_isolated_satellite_month():
    z = pd.Series([0.1, 3.0, 0.2, 2.5, 2.8, -2.5, 2.6])
    a = fusion.persistent(z)
    assert a.tolist() == [0.0, 0.0, 0.0, 0.0, 2.8, 0.0, 0.0]


def test_pack_roundtrip_keeps_precision(tmp_path):
    from strate.modules.optical import pack, unpack

    rng = np.random.default_rng(0)
    def r():
        return rng.random((20, 20)).astype("float32")

    comp = {"valid": np.ones((20, 20), "uint8"), "water": r(), "red": r() * 0.5, "nir": r(), "swir": r(),
            "scenes": np.array(["a"])}
    comp["red"][0, 0] = np.nan
    comp["swir"][1, 1] = -0.05  # réflectance légèrement négative (offset ESA) : bornée à 0, pas « non vu »
    np.savez_compressed(tmp_path / "c.npz", **pack(comp))
    back = unpack(np.load(tmp_path / "c.npz"))
    assert np.isnan(back["red"][0, 0])
    assert np.nanmax(np.abs(back["nir"] - comp["nir"])) <= 6e-5
    assert np.abs(back["water"] - comp["water"]).max() <= 0.006
    assert back["swir"][1, 1] == 0.0


# -- calibration ----------------------------------------------------------------------------
def _fake_panel(signal_strength: float, n_mines: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_mines):
        for y in (2017, 2018, 2019):
            dp = rng.normal(0, 0.3)
            rows.append({"id": f"m{i}", "year": y, "dP": dp,
                         "d_bare": signal_strength * dp + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_calibration_stats_see_signal_and_no_signal():
    from strate.calibrate import _boot, _direction, _rho

    strong = _fake_panel(1.0)
    rho, lo, hi = _boot(strong, _rho("d_bare"), n=200)
    assert rho > 0.8 and lo > 0.6
    assert _direction("d_bare")(strong) > 0.85

    none = _fake_panel(0.0)
    rho, lo, hi = _boot(none, _rho("d_bare"), n=200)
    assert lo < 0 < hi  # pas de lien : l'intervalle contient 0


def test_wikipedia_list_parser():
    from strate.catalog import parse_coord, parse_list

    wt = """{| class="wikitable"
! Mine !! Owner !! Coordinates
|-
| nowrap="" |[[Mount Whaleback mine|Mt. Whaleback mine]]
|[[BHP]]
|
|-
| nowrap |Unlinked Pit<ref>x</ref>
|Someone
|{{coord|29|14|26|S|124|32|19|E|display=inline}}
|-
|'''[[Kalgoorlie]]''' || town || {{coord|-30.75|121.47}}
|}"""
    rows = parse_list(wt)
    assert rows[0] == {"name": "Mt. Whaleback mine", "article": "Mount Whaleback mine", "coord": None}
    assert rows[1]["name"] == "Unlinked Pit"
    assert rows[1]["coord"] == pytest.approx((-29.2406, 124.5386), abs=1e-3)
    assert rows[2]["article"] == "Kalgoorlie"  # 1re colonne seulement : la ville vient d'ici, pas d'ailleurs
    assert parse_coord("bad|data") is None


# -- radar -------------------------------------------------------------------------------------------
def test_radar_blocks_average_power_and_ignore_nan():
    from strate.modules.radar import _blocks, _db

    a = np.ones((11, 11), "float32")
    a[0, 0] = np.nan
    b = _blocks(a, 5)
    assert b.shape == (2, 2)  # la ligne/colonne en trop est coupée
    assert np.allclose(b, 1.0)
    assert np.isnan(_db(np.array([0.0, -1.0], "float32"))).all()


def test_radar_masks_fallback_square(monkeypatch):
    import geopandas as gpd

    from strate.modules import radar

    monkeypatch.setattr(radar.catalog, "mine_polygons", lambda *a, **k: gpd.GeoDataFrame(geometry=[]))
    site = entities.load().site("greenstone")
    grid = geo.site_grid(site.lon, site.lat, site.radius_m + radar.RING_M, 20)
    mine, ring, src = radar.masks(site, grid)
    assert src == "carré du site"
    assert not (mine & ring).any()  # jamais un pixel dans les deux
    assert mine.sum() == pytest.approx((2 * site.radius_m / 20 + 1) ** 2, rel=0.05)


# -- screener -------------------------------------------------------------------------------------
def test_screener_normalisation():
    from strate.screener import commodity_groups, norm_country

    assert norm_country("Russia") == norm_country("Russian Federation") == "Russian Federation"
    assert norm_country("CÃ´te d'Ivoire") == "Côte d'Ivoire"
    assert norm_country("Democratic Republic of The Congo") == "DR Congo"
    assert norm_country(None) is None
    assert commodity_groups("Gold ore, copper; silver") == ["or", "cuivre", "argent"]
    assert commodity_groups("lignite, bituminous coal") == ["charbon"]
    assert commodity_groups("PGM, palladium") == ["platinoïdes"]
    assert commodity_groups("tin ore") == ["étain"]
    assert commodity_groups("stone, limestone") == ["granulats / pierre"]


def test_main_production_prefers_ore_then_main_metal():
    from strate.screener import main_production

    s = {"argent (métal)": (2018, 267.0), "cuivre (métal)": (2018, 1_167_900.0)}
    assert main_production(s, ["cuivre", "or", "argent"])[0] == "cuivre (métal)"
    s["minerai extrait"] = (2019, 2e7)
    assert main_production(s, ["cuivre"]) == ("minerai extrait", 2019, 2e7)
    assert main_production(None, ["or"])[0] is None


def test_screener_query_filters():
    from strate.screener import query

    df = pd.DataFrame({
        "nom": ["A", "B", "C"], "pays": ["Chile", "Peru", "Chile"], "region": [None] * 3,
        "proprietaire": ["X", None, "Y"], "categorie": ["mine active"] * 3,
        "commodites": ["cuivre, or", "or", None], "commodite_principale": ["cuivre", "or", None],
        "emprise_km2": [10.0, 1.0, np.nan], "prod_estimee_t": [1e7, 1e6, np.nan],
        "prod_reelle_t": [np.nan, 5.0, np.nan], "analyse": [True, False, False], "niveau": [2, None, None],
        "lat": [-24.0, -10.0, -30.0], "lon": [-69.0, -76.0, -70.0], "confiance": [None] * 3,
        "prod_reelle_annee": [None] * 3,
    })
    assert list(query(df, pays=["Chile"])["nom"]) == ["A", "C"]
    assert list(query(df, commodite=["or"])["nom"]) == ["A", "B"]
    assert list(query(df, avec_emprise=True, sort="emprise_km2", desc=False)["nom"]) == ["B", "A"]
    assert list(query(df, niveau_min=1)["nom"]) == ["A"]
    assert list(query(df, prod_min=5e6)["nom"]) == ["A"]
    assert list(query(df, bbox=(-72, -26, -68, -20))["nom"]) == ["A"]
    assert list(query(df, sort="pas_une_colonne")["nom"])[0] == "A"  # tri inconnu → défaut, pas d'erreur


def test_screener_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from strate import screener
    from strate.api import app

    fake = pd.DataFrame([{c: None for c in
                          ["id", "nom", "pays", "region", "categorie", "commodites", "commodite_principale",
                           "proprietaire", "prod_estimee_type", "prod_reelle_serie", "alerte", "mois_signal",
                           "confiance", "wikipedia", "sources", "avertissement"]}
                         | {"id": "x1", "nom": "Mine X", "pays": "Chile", "categorie": "mine active",
                            "commodites": "cuivre", "emprise_km2": 3.0, "prod_estimee_t": 2e6,
                            "prod_estimee_bas90": 2e5, "prod_estimee_haut90": 2e7, "prod_reelle_t": np.nan,
                            "prod_reelle_annee": None, "analyse": False, "niveau": None, "suivi": False,
                            "lat": -20.0, "lon": -69.0}])
    monkeypatch.setattr(screener, "table", lambda refresh=False: fake)
    c = TestClient(app)
    r = c.get("/api/screener", params={"pays": "Chile"}).json()
    assert r["total"] == 1 and r["rows"][0]["nom"] == "Mine X"
    assert c.get("/api/screener", params={"pays": "Peru"}).json()["total"] == 0
    assert c.get("/api/screener", params={"bbox": "1,2"}).status_code == 422
    csv = c.get("/api/screener.csv").text
    assert csv.splitlines()[0].startswith("id,nom,pays") and "Mine X" in csv
    assert c.get("/api/screener/facets").json()["total"] == 1
    assert c.get("/api/screener/rows").json()[0]["nom"] == "Mine X"
    assert "strate_screener" in c.get("/widgets.json").json()
