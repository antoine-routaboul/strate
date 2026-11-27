"""Seconde revue, API et screener : jumelles renvoyées vers la mine suivie, analyse « récente » (les
signaux de la seule calibration 2016–2019 ne comptent pas), lot d'analyses qui ne relance pas une
mine terminée, liste suivie relue après une modification de sites.toml, recherche par les noms de la
jumelle, production réelle annoncée seulement quand elle existe."""

from __future__ import annotations

import json
import os
import queue
import shutil
import time
from datetime import date, timedelta
from importlib.resources import files

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from strate import api, catalog, entities, fusion, screener

CAL = {"modele_emprise": {
    "minerai": {"intercept": 5.85, "slope": 0.795, "n": 218, "facteur_median": 2.46, "facteur_90": 8.35,
                "emprise_min_km2": 0.1, "emprise_max_km2": 300.0},
}}
RECENT = (pd.Timestamp(date.today()) - pd.Timedelta(days=60)).to_period("M").to_timestamp()


def _reset_caches():
    catalog.load.cache_clear()
    entities.load.cache_clear()
    entities.from_catalog.cache_clear()
    screener.twins.cache_clear()
    api._haystack.cache_clear()
    api._production_ids_cached.cache_clear()
    screener._seen_inputs = None
    screener._mem = None
    screener._cal = None
    screener._last_cache.clear()


def _catalog(od_lat: float, od_lon: float, has_production: bool | None = None) -> pd.DataFrame:
    rows = [
        # jumelle d'Olympic Dam : même mine, autre id, porte l'alias « Roxby Downs Mine »
        dict(id="icmm06349", icmm="X1", name="Olympic Dam", aliases="Olympic Dam;Roxby Downs Mine",
             lat=od_lat + 0.004, lon=od_lon, country="Australia", asset_type="Mine",
             commodities="copper, uranium, gold", footprint_km2=42.4, jasansky="COM1"),
        # lié à Jasansky mais sans aucune ligne de production (cas de Coral Bay HPAL)
        dict(id="icmm_coral", icmm="X2", name="Coral Bay HPAL", aliases=None, lat=8.6, lon=117.4,
             country="Philippines", asset_type="Plant", commodities="nickel", footprint_km2=None,
             jasansky="COM2"),
        # analysée seulement pour la calibration 2016–2019
        dict(id="icmm_old", icmm="X3", name="Spring Creek", aliases=None, lat=45.0, lon=-106.9,
             country="USA", asset_type="Mine", commodities="coal", footprint_km2=20.0, jasansky=None),
    ]
    cat = pd.DataFrame(rows)
    for c in ("region", "confidence", "owner", "operator", "wikipedia", "production_status", "wikidata",
              "minx", "miny", "maxx", "maxy"):
        cat[c] = None
    cat["sources"] = "icmm"
    if has_production is not None:
        cat["has_production"] = [has_production, False, False]
    return cat


def _signals(root, site: str, months: list[pd.Timestamp]) -> None:
    d = root / "signals" / "optical"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"entity": site, "date": months, "signal": "bare", "value": 0.1, "unit": "",
                  "quality": 1.0, "source": "test"}).to_parquet(d / f"{site}.parquet", index=False)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """API coupée des vraies données : catalogue, signaux, calibration et sites.toml dans tmp_path."""
    (tmp_path / "catalog").mkdir()
    sites = tmp_path / "sites.toml"
    shutil.copy(str(files("strate.data").joinpath("sites.toml")), sites)
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps(CAL))
    old = time.time() - 3600
    for p in (sites, cal):
        os.utime(p, (old, old))
    monkeypatch.setattr(entities, "files", lambda pkg: tmp_path)  # entities.load lit ce sites.toml
    monkeypatch.setattr(screener, "_sites_path", lambda: sites)
    monkeypatch.setattr(screener, "_cal_path", lambda: cal)
    monkeypatch.setattr(screener, "DATA_DIR", tmp_path)
    monkeypatch.setattr(screener, "OUT", tmp_path / "catalog" / "screener.parquet")
    monkeypatch.setattr(catalog, "OUT", tmp_path / "catalog")
    monkeypatch.setattr(catalog, "all_production", lambda: pd.DataFrame(
        {"facility_id": ["COM1"], "serie": ["minerai extrait"], "year": [2019], "value_tonnes": [9.0e6]}))
    monkeypatch.setattr(fusion, "radar", lambda month=None: pd.DataFrame())
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "_queue", queue.Queue())  # le vrai travailleur reste sur l'ancienne file
    _reset_caches()
    od = entities.load().sites["olympic_dam"]
    _catalog(od.lat, od.lon).to_parquet(tmp_path / "catalog" / "mines.parquet", index=False)
    _signals(tmp_path, "olympic_dam", [RECENT - pd.DateOffset(months=1), RECENT])
    _signals(tmp_path, "icmm_old", list(pd.date_range("2016-06-01", "2019-10-01", freq="MS")))
    _reset_caches()
    yield tmp_path
    _reset_caches()


# -- #2 : jumelle d'une mine suivie --------------------------------------------------------------------
def test_twin_id_resolves_to_watched_site(world):
    c = TestClient(api.app)
    assert screener.twins()["olympic_dam"] == "icmm06349"
    s = c.get("/api/site", params={"site": "icmm06349"}).json()
    assert s["id"] == "olympic_dam" and s["suivi"] is True
    assert s["production_emprise"] is not None and s["analyse"] is True
    r = c.post("/api/analyze", params={"site": "icmm06349"}).json()
    assert r == {"site": "olympic_dam", "statut": "en file"}
    assert "icmm06349" not in api.JOBS and api._queue.qsize() == 1
    # second clic sur l'une ou l'autre fiche : toujours une seule analyse en file
    c.post("/api/analyze", params={"site": "olympic_dam"})
    assert api._queue.qsize() == 1
    assert c.get("/api/dossier", params={"site": "icmm06349"}).status_code == 200


# -- #14 : analyse récente / données anciennes ---------------------------------------------------------
def test_site_analysis_state_recent_vs_calibration_only(world):
    c = TestClient(api.app)
    od = c.get("/api/site", params={"site": "olympic_dam"}).json()
    assert od["analyse"] is True and od["donnees_anciennes"] is False
    assert od["derniere_date"] == f"{RECENT:%Y-%m}"
    old = c.get("/api/site", params={"site": "icmm_old"}).json()
    assert old["analyse"] is False and old["donnees_anciennes"] is True and old["derniere_date"] == "2019-10"
    none = c.get("/api/site", params={"site": "icmm_coral"}).json()
    assert none["analyse"] is False and none["donnees_anciennes"] is False and none["derniere_date"] is None


def test_screener_analyse_column_uses_recent_rule_and_ages_with_the_day(world, monkeypatch):
    tab = screener.table().set_index("id")
    assert bool(tab.loc["olympic_dam", "analyse"]) and not bool(tab.loc["icmm_old", "analyse"])
    assert screener.facets(screener.table())["analysees"] == 1
    # un an plus tard, sans qu'aucun fichier ne change : l'analyse d'Olympic Dam n'est plus récente
    later = date.today() + timedelta(days=365)

    class Later(date):
        @classmethod
        def today(cls):
            return later

    monkeypatch.setattr(screener, "date", Later)
    tab = screener.table().set_index("id")
    assert not bool(tab.loc["olympic_dam", "analyse"])
    assert screener.analysis_state("olympic_dam")["donnees_anciennes"] is True


def test_batch_skips_finished_mines_but_keeps_old_data_eligible(world, monkeypatch):
    tab = pd.DataFrame({"id": ["m0", "m1", "m2", "icmm_old"], "nom": list("ABCD"),
                        "prod_estimee_t": [4.0, 3.0, 2.0, 1.0], "analyse": [True, False, False, False]})
    monkeypatch.setattr(screener, "table", lambda refresh=False: tab)
    api.JOBS["m1"] = "terminé"  # aucune image exploitable : terminé sans signaux récents
    c = TestClient(api.app)
    r = c.post("/api/screener/analyze", params={"n": 5}).json()
    assert [x["site"] for x in r["mises_en_file"]] == ["m2", "icmm_old"]
    assert r["deja_terminees"] == ["m1"] and r["deja_en_cours"] == []
    r2 = c.post("/api/screener/analyze", params={"n": 5}).json()
    assert r2["mises_en_file"] == []  # ni m1 terminé, ni les deux en file ne sont remis
    # demande explicite : une mine terminée peut être relancée
    api.JOBS["olympic_dam"] = "terminé"
    assert c.post("/api/analyze", params={"site": "olympic_dam"}).json()["statut"] == "en file"


# -- #4 : liste suivie relue après modification de sites.toml --------------------------------------------
def test_sites_and_widgets_follow_sites_toml(world):
    c = TestClient(api.app)
    assert "xtest" not in {s["id"] for s in c.get("/api/sites").json()}
    sites = world / "sites.toml"
    with sites.open("a", encoding="utf-8") as fh:
        fh.write('\n[sites.xtest]\nname = "Xtest"\ncountry = "CA"\nregion = "Ontario"\n'
                 'commodity = ["gold"]\nlat = 50.0\nlon = -80.0\nradius_m = 3000\n'
                 'owners = [{ company = "equinox", share = 1.0 }]\nstage = "production"\n'
                 'coord_source = "https://example.org"\n')
    os.utime(sites, None)
    assert "xtest" in {s["id"] for s in c.get("/api/sites").json()}
    opts = c.get("/widgets.json").json()["strate_dossier"]["params"][0]["options"]
    assert "xtest" in {o["value"] for o in opts}


# -- #16 : recherche par les noms de la jumelle -----------------------------------------------------------
def test_search_and_screener_find_watched_mine_by_twin_alias(world):
    c = TestClient(api.app)
    hits = [h["id"] for h in c.get("/api/search", params={"q": "Roxby Downs"}).json()]
    assert hits == ["olympic_dam"]
    rows = c.get("/api/screener", params={"q": "roxby"}).json()["rows"]
    assert [r["id"] for r in rows] == ["olympic_dam"]
    assert "recherche" not in rows[0] and "autres_noms" not in rows[0]  # colonnes internes


# -- #7 (API) : production réelle seulement si elle existe ---------------------------------------------------
def test_real_production_flag_without_and_with_catalog_column(world):
    c = TestClient(api.app)
    assert c.get("/api/site", params={"site": "icmm_coral"}).json()["production_reelle"] is None
    assert c.get("/api/site", params={"site": "olympic_dam"}).json()["production_reelle"] == "COM1"
    assert c.get("/api/catalog/stats").json()["avec_production_reelle"] == 1
    # mines.parquet récent : la colonne has_production écrite par catalog.build fait foi
    od = entities.load().sites["olympic_dam"]
    p = world / "catalog" / "mines.parquet"
    _catalog(od.lat, od.lon, has_production=False).to_parquet(p, index=False)
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert c.get("/api/catalog/stats").json()["avec_production_reelle"] == 0
    assert c.get("/api/site", params={"site": "olympic_dam"}).json()["production_reelle"] is None
