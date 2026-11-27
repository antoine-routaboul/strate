"""Corrections du screener et de l'API : fraîcheur de la table, jumelles de la liste suivie, domaine du
modèle d'emprise, filtres pays / zone de carte, file d'analyses, paramètres invalides."""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from strate import api, catalog, entities, fusion, screener

CAL = {"modele_emprise": {
    "minerai": {"intercept": 5.85, "slope": 0.795, "n": 218, "facteur_median": 2.46, "facteur_90": 8.35,
                "emprise_min_km2": 0.1, "emprise_max_km2": 300.0},
    "charbon": {"intercept": 6.70, "slope": 0.243, "n": 161, "facteur_median": 2.28, "facteur_90": 10.4,
                "emprise_min_km2": 0.1, "emprise_max_km2": 190.0},
}}


def _reset_caches():
    catalog.load.cache_clear()
    entities.load.cache_clear()
    entities.from_catalog.cache_clear()
    screener.twins.cache_clear()
    api._haystack.cache_clear()
    screener._seen_inputs = None
    screener._mem = None
    screener._cal = None


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Screener coupé des vraies données : dossiers, catalogue et calibration dans tmp_path."""
    (tmp_path / "catalog").mkdir()
    (tmp_path / "signals").mkdir()
    sites = tmp_path / "sites.toml"
    sites.write_text("")
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps(CAL))
    old = time.time() - 3600
    for p in (sites, cal):
        os.utime(p, (old, old))
    monkeypatch.setattr(screener, "DATA_DIR", tmp_path)
    monkeypatch.setattr(screener, "OUT", tmp_path / "catalog" / "screener.parquet")
    monkeypatch.setattr(catalog, "OUT", tmp_path / "catalog")
    monkeypatch.setattr(screener, "_sites_path", lambda: sites)
    monkeypatch.setattr(screener, "_cal_path", lambda: cal)
    _reset_caches()
    yield tmp_path
    _reset_caches()  # les caches ne doivent pas garder le faux catalogue pour les tests suivants


def _counting_build(monkeypatch, delay=0.0):
    calls = []

    def fake():
        stamp = screener._inputs_mtime_ns()
        calls.append(stamp)
        time.sleep(delay)
        df = pd.DataFrame({"id": ["a"], "n": [len(calls)]})
        df.attrs["version"] = screener.TABLE_VERSION
        screener._write(df, stamp)
        return df

    monkeypatch.setattr(screener, "build", fake)
    return calls


def _touch(p, t):
    os.utime(p, (t, t))


# -- #1 #9 : fraîcheur ---------------------------------------------------------------------------
def test_table_rebuilds_on_any_signal_family_and_tolerates_empty_dirs(iso, monkeypatch):
    calls = _counting_build(monkeypatch)
    (iso / "signals" / "optical").mkdir()  # dossier vide : plus de max() d'une liste vide
    screener.table()
    screener.table()
    assert len(calls) == 1

    att = iso / "signals" / "attention"
    att.mkdir()
    f = att / "icmm01433.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(f)
    now = time.time() + 10
    _touch(f, now)
    _touch(att, now)
    screener.table()
    assert len(calls) == 2  # l'attention (et le radar) changent le niveau d'alerte

    f.unlink()  # supprimer un fichier ne laisse aucun fichier plus récent : c'est le dossier qui change
    _touch(att, now + 10)
    screener.table()
    assert len(calls) == 3
    screener.table()
    assert len(calls) == 3


def test_table_rebuilds_old_format_and_on_calibration_change(iso, monkeypatch):
    calls = _counting_build(monkeypatch)
    old = pd.DataFrame({"id": ["a"]})
    old.attrs["version"] = screener.TABLE_VERSION - 1
    old.to_parquet(screener.OUT)  # écrite par l'ancien code, plus récente que toutes les entrées
    screener.table()
    assert len(calls) == 1
    _touch(iso / "calibration.json", time.time() + 10)
    screener.table()
    assert len(calls) == 2


def test_signal_written_during_build_triggers_rebuild(iso, monkeypatch):
    sig = iso / "signals" / "radar"
    sig.mkdir()
    calls = _counting_build(monkeypatch)
    real = screener.build

    def slow():
        df = real()
        f = sig / "x.parquet"  # écrit après la lecture des entrées, avant la fin de la construction
        pd.DataFrame({"x": [1]}).to_parquet(f)
        _touch(f, time.time() + 5)
        return df

    monkeypatch.setattr(screener, "build", slow)
    screener.table()
    monkeypatch.setattr(screener, "build", real)
    screener.table()
    assert len(calls) == 2


# -- #6 : une seule reconstruction à la fois ---------------------------------------------------------
def test_concurrent_calls_build_once(iso, monkeypatch):
    calls = _counting_build(monkeypatch, delay=0.3)
    out = []
    ts = [threading.Thread(target=lambda: out.append(screener.table())) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and len(out) == 4
    assert not list((iso / "catalog").glob(".screener-*"))  # pas de fichier temporaire oublié


# -- #2 : catalogue reconstruit par un autre processus ------------------------------------------------
def test_sync_clears_catalog_caches(iso, monkeypatch):
    hooks = []
    monkeypatch.setattr(screener, "_catalog_hooks", [lambda: hooks.append(1)])
    mines = iso / "catalog" / "mines.parquet"
    pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]}).to_parquet(mines)
    screener.sync()
    assert len(catalog.load()) == 2
    pd.DataFrame({"id": ["a"], "name": ["A"]}).to_parquet(mines)  # la CLI réécrit le catalogue
    _touch(mines, time.time() + 10)
    assert len(catalog.load()) == 2  # cache du processus
    screener.sync()
    assert len(catalog.load()) == 1 and len(hooks) == 2
    screener.sync()
    assert len(hooks) == 2  # rien n'a changé : pas de vidage


# -- #11 : une seule calibration, relue quand le fichier change -----------------------------------------
def test_calibration_reloaded_on_change(iso):
    cal = iso / "calibration.json"
    assert screener.calibration()["modele_emprise"]["minerai"]["n"] == 218
    assert api.calibration is screener.calibration
    new = json.loads(json.dumps(CAL))
    new["modele_emprise"]["minerai"]["n"] = 300
    cal.write_text(json.dumps(new))
    _touch(cal, time.time() + 10)
    assert api.calibration()["modele_emprise"]["minerai"]["n"] == 300
    cal.write_text('{"modele_emprise": ')  # en cours d'écriture par la calibration
    _touch(cal, time.time() + 20)
    assert screener.calibration()["modele_emprise"]["minerai"]["n"] == 300


# -- #25 : domaine du modèle d'emprise ------------------------------------------------------------------
def test_footprint_estimate_only_inside_fit_population(monkeypatch):
    monkeypatch.setattr(screener, "calibration", lambda: CAL)
    fe = screener.footprint_estimate
    kind, est, lo, hi = fe(10.0, ["cuivre", "or"], "mine active")
    assert kind == "minerai" and lo < est < hi and hi / est == pytest.approx(8.35)
    assert fe(10.0, ["charbon"], "mine active")[0] == "charbon"
    assert fe(10.0, ["or", "charbon"], "mine active")[0] == "minerai"  # commodité principale
    assert fe(10.0, ["cuivre"], "usine")[0] is None  # une fonderie n'extrait pas de minerai
    assert fe(10.0, [], "emprise sans nom")[0] is None  # charbon ou métal ? on ne sait pas
    assert fe(10.0, ["granulats / pierre"], "mine active")[0] is None  # hors population
    assert fe(10.0, ["diamant"], "mine active")[0] is None
    assert fe(250.0, ["charbon"], "mine active")[0] is None  # hors plage d'emprise apprise
    assert fe(np.nan, ["or"], "mine active")[0] is None
    assert fe(np.float64(10.0), ["or"], "mine active")[0] == "minerai"


def _fake_catalog(sites):
    od = sites["olympic_dam"]
    rows = [
        dict(id="icmm06349", icmm="X1", name="Olympic Dam", aliases=None, lat=od.lat + 0.004, lon=od.lon,
             country="Australia", asset_type="Mine", commodities="copper, uranium, gold",
             footprint_km2=42.4, jasansky="COM00937.00", wikidata=None),
        dict(id="icmm_smelter", icmm="X2", name="Hayden smelter", aliases=None, lat=33.0, lon=-110.8,
             country="USA", asset_type="Smelter;Refinery", commodities="copper", footprint_km2=5.5,
             jasansky=None, wikidata=None),
        dict(id="fp_1.000_2.000", icmm=None, name=None, aliases=None, lat=1.0, lon=2.0, country="Russia",
             asset_type="Mine (emprise satellite)", commodities=None, footprint_km2=100.0, jasansky=None,
             wikidata=None),
        dict(id="q1", icmm=None, name="Quarry", aliases=None, lat=40.0, lon=5.0, country="France",
             asset_type="quarry", commodities="limestone", footprint_km2=3.0, jasansky=None, wikidata="Q1"),
        dict(id="icmm_coal", icmm="X3", name="Coal Pit", aliases=None, lat=-22.0, lon=148.0,
             country="Australia ", asset_type="Mine", commodities="thermal coal", footprint_km2=20.0,
             jasansky=None, wikidata=None),
        dict(id="icmm_ru", icmm="X4", name="Nornik", aliases=None, lat=69.3, lon=88.2,
             country="Russian Federation", asset_type="Mine", commodities="nickel", footprint_km2=50.0,
             jasansky=None, wikidata=None),
    ]
    cat = pd.DataFrame(rows)
    for c in ("region", "confidence", "owner", "operator", "wikipedia", "production_status", "minx",
              "miny", "maxx", "maxy"):
        cat[c] = None
    cat["sources"] = "icmm"
    return cat.set_index("id", drop=False)


# -- #3 #21 #25 : construction de la table ------------------------------------------------------------
def test_build_merges_watchlist_twin_and_respects_model_domain(iso, monkeypatch):
    sites = entities.load().sites
    cat = _fake_catalog(sites)
    cat.reset_index(drop=True).to_parquet(iso / "catalog" / "mines.parquet")  # relu par catalog.load
    monkeypatch.setattr(screener, "calibration", lambda: CAL)
    monkeypatch.setattr(catalog, "all_production", lambda: pd.DataFrame(
        {"facility_id": ["COM00937.00"], "serie": ["minerai extrait"], "year": [2019],
         "value_tonnes": [9.0e6]}))
    monkeypatch.setattr(fusion, "radar", lambda month=None: pd.DataFrame())
    df = screener.table().set_index("id")
    assert screener.twins()["olympic_dam"] == "icmm06349"
    assert "icmm06349" not in df.index  # une seule ligne par mine
    od = df.loc["olympic_dam"]
    assert od["suivi"] and od["emprise_km2"] == pytest.approx(42.4)
    assert od["prod_estimee_type"] == "minerai" and od["prod_estimee_t"] > 0
    assert od["prod_reelle_t"] == pytest.approx(9.0e6) and od["prod_reelle_annee"] == 2019
    assert od["pays"] == "Australia"
    assert pd.isna(df.loc["icmm_smelter", "prod_estimee_t"])
    assert df.loc["icmm_smelter", "categorie"] == "usine"
    assert pd.isna(df.loc["fp_1.000_2.000", "prod_estimee_t"])
    assert pd.isna(df.loc["fp_1.000_2.000", "prod_estimee_type"])  # null dans le JSON
    assert pd.isna(df.loc["q1", "prod_estimee_t"])
    assert df.loc["icmm_coal", "prod_estimee_type"] == "charbon"
    assert df.loc["icmm_coal", "pays"] == "Australia"
    assert df.loc["fp_1.000_2.000", "pays"] == df.loc["icmm_ru", "pays"] == "Russian Federation"
    assert df.loc["icmm_smelter", "pays"] == "United States"
    assert len(df) == len(sites) + len(cat) - 1


# -- #5 #45 : zone visible de la carte -----------------------------------------------------------------
def _pts():
    return pd.DataFrame({
        "nom": ["Red Dog", "Chukotka", "Tokyo", "Lima"], "lat": [68.07, 66.0, 35.7, -12.0],
        "lon": [-162.87, 175.0, 139.7, -77.0], "prod_estimee_t": [4.0, 3.0, 2.0, 1.0],
    })


def test_bbox_wraps_antimeridian():
    q = screener.query
    df = _pts()
    assert set(q(df, bbox=(170, 50, 200, 72))["nom"]) == {"Red Dog", "Chukotka"}  # Leaflet : est > 180
    assert set(q(df, bbox=(170, 50, -160, 72))["nom"]) == {"Red Dog", "Chukotka"}  # déjà ramené
    assert set(q(df, bbox=(-200, 50, -150, 72))["nom"]) == {"Red Dog", "Chukotka"}  # ouest < -180
    assert len(q(df, bbox=(-250, -90, 250, 90))) == 4  # vue de plus de 360°
    assert set(q(df, bbox=(-80, -20, -70, 0))["nom"]) == {"Lima"}
    assert set(q(df, bbox=(-80, 0, -70, -20))["nom"]) == {"Lima"}  # sud et nord inversés
    assert set(q(df, bbox=(130, 30, 180, 40))["nom"]) == {"Tokyo"}


def test_bbox_rejects_non_finite(monkeypatch):
    monkeypatch.setattr(screener, "table", lambda refresh=False: _pts())
    r = TestClient(api.app).get("/api/screener", params={"bbox": "nan,1,2,3"})
    assert r.status_code == 422


# -- #10 #21 #48 : pays -------------------------------------------------------------------------------
def test_country_normalisation_and_filter():
    nc = entities.norm_country
    assert nc("Côte D’Ivoire") == nc("CÃ´te d'Ivoire") == nc("Cote d'Ivoire") == "Côte d'Ivoire"
    assert nc("People's Republic of China") == "China"
    assert nc("Australia ") == "Australia" and nc("CA") == "Canada" and nc("Wales") == "United Kingdom"
    assert nc("Myanmar/Burma") == "Myanmar" and nc("United Republic of Tanzania") == "Tanzania"
    assert screener.norm_country is nc  # un seul vocabulaire pour le screener et l'API
    df = pd.DataFrame({"nom": ["A", "B", "C", "D"], "prod_estimee_t": [1.0] * 4,
                       "pays": ["Chile", "Russian Federation", "Côte d'Ivoire", None]})
    q = screener.query
    assert list(q(df, pays=["chile"])["nom"]) == ["A"]
    assert list(q(df, pays=[" CHILE "])["nom"]) == ["A"]
    assert list(q(df, pays=["Chili"])["nom"]) == ["A"]
    assert list(q(df, pays=["russia"])["nom"]) == ["B"]
    assert list(q(df, pays=["Cote d’ivoire", "Russia"])["nom"]) == ["B", "C"]
    assert q(df, pays=["Peru"]).empty


def test_stats_count_normalised_countries(monkeypatch):
    cat = pd.DataFrame({"id": ["a", "b", "c", "d"], "icmm": ["1", None, None, None],
                        "asset_type": ["Mine"] * 4, "footprint_km2": [1.0, None, None, None],
                        "jasansky": [None] * 4, "wikipedia": [None] * 4,
                        "country": ["Russia", "Russian Federation", "Czechia", "Czech Republic"]})
    monkeypatch.setattr(api, "_cat", lambda: cat)
    assert TestClient(api.app).get("/api/catalog/stats").json()["pays"] == 2


# -- #47 : source des coordonnées ------------------------------------------------------------------------
def test_coord_source_of_wikipedia_list_rows():
    cs = entities.catalog_source
    assert "Wikipedia" in cs({"sources": "wikipedia"}) and "Maus" not in cs({"sources": "wikipedia"})
    assert "Maus" in cs({"sources": "maus", "minx": 1.0})
    assert cs({"icmm": "X1", "sources": "icmm"}).startswith("ICMM")
    assert "wikidata.org" in cs({"wikidata": "Q1", "sources": "wikidata"})


# -- #4 #8 : file d'analyses --------------------------------------------------------------------------
@pytest.fixture
def jobs(monkeypatch):
    """File isolée : le vrai travailleur reste bloqué sur l'ancienne file, rien n'est téléchargé."""
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "_queue", queue.Queue())
    return api


def test_screener_analyze_skips_queued_mines(jobs, monkeypatch):
    tab = pd.DataFrame({"id": [f"m{i}" for i in range(6)], "nom": list("ABCDEF"),
                        "prod_estimee_t": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        "analyse": [True, False, False, False, False, False]})
    monkeypatch.setattr(screener, "table", lambda refresh=False: tab)
    c = TestClient(api.app)
    r1 = c.post("/api/screener/analyze", params={"n": 2}).json()
    assert [x["site"] for x in r1["mises_en_file"]] == ["m1", "m2"] and r1["deja_en_cours"] == []
    jobs.JOBS["m1"] = "en cours (1/3) : images"
    r2 = c.post("/api/screener/analyze", params={"n": 2}).json()
    assert [x["site"] for x in r2["mises_en_file"]] == ["m3", "m4"]
    assert r2["deja_en_cours"] == ["m1", "m2"] and r2["en_file"] == 4
    r3 = c.post("/api/screener/analyze", params={"n": 5}).json()
    assert [x["site"] for x in r3["mises_en_file"]] == ["m5"]
    r4 = c.post("/api/screener/analyze", params={"n": 5}).json()
    assert r4["mises_en_file"] == [] and len(r4["deja_en_cours"]) == 5


def test_jobs_listing_survives_concurrent_enqueue(jobs):
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            jobs.enqueue(f"s{i}")
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        deadline = time.time() + 0.5
        while time.time() < deadline:
            jobs.jobs()  # ne doit jamais lever « dictionary changed size during iteration »
    finally:
        stop.set()
        t.join()


# -- #7 : paramètres invalides → 4xx, pas 500 -------------------------------------------------------------
@pytest.mark.parametrize(("url", "code"), [
    ("/api/thumbnail?site=greenstone&month=2023-13", 422),
    ("/api/thumbnail?site=greenstone&month=2020", 422),
    ("/api/radar?month=abc", 422),
    ("/api/radar?month=2023-13", 422),
    ("/api/signals?site=greenstone&module=nope", 404),
    ("/api/events?site=nope", 404),
])
def test_bad_parameters_are_client_errors(url, code):
    assert TestClient(api.app, raise_server_exceptions=False).get(url).status_code == code


# -- #26 #3 : fiche d'une mine suivie --------------------------------------------------------------------
def test_site_detail_contract_and_twin_production(monkeypatch):
    if catalog.load().empty:
        pytest.skip("catalogue absent")
    twin = screener.twins().get("olympic_dam")
    if twin is None:
        pytest.skip("pas de jumelle pour Olympic Dam dans ce catalogue")
    tab = pd.DataFrame([{"id": "olympic_dam", "prod_estimee_type": "minerai", "prod_estimee_t": 1.4e7,
                         "prod_estimee_bas90": 1.7e6, "prod_estimee_haut90": 1.2e8, "emprise_km2": 42.4}])
    monkeypatch.setattr(screener, "table", lambda refresh=False: tab)
    c = TestClient(api.app)
    s = c.get("/api/site", params={"site": "olympic_dam"}).json()
    assert "production_estimee" not in s
    pe = s["production_emprise"]
    assert {"type", "tonnes_an", "bas_90", "haut_90", "facteur_typique", "facteur_90", "emprise_km2",
            "mines_calibration"} <= set(pe)
    assert s["pays"] == "Australia" and s["sources"] == "liste suivie"
    assert s["production_reelle"] == catalog.load().loc[twin, "jasansky"]
    hits = c.get("/api/search", params={"q": "olympic dam"}).json()
    assert [h["id"] for h in hits].count("olympic_dam") == 1 and twin not in [h["id"] for h in hits]
    assert c.get("/api/site", params={"site": "nope"}).status_code == 404
