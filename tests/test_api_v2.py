"""Routes prévisions / sociétés / corrélations (docs/previsions-societes.md § 3) et CLI forecast / companies.

Modules forecast et companies remplacés par des faux (sys.modules) : aucun réseau, aucune donnée réelle.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from strate import api, cli, core, screener
from strate.veille import store as vstore


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FORECAST_DIR", tmp_path / "forecast")
    monkeypatch.setattr(vstore, "DB_PATH", tmp_path / "veille" / "veille.db")
    monkeypatch.setattr(screener, "twins", lambda: {})
    monkeypatch.setattr(api, "calibration", lambda: {})
    return tmp_path


@pytest.fixture
def client(env):
    return TestClient(api.app)


def fake(monkeypatch, name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(f"strate.{name}")
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, f"strate.{name}", mod)
    return mod


def no_module(monkeypatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, f"strate.{name}", None)  # import → ImportError


BT = {
    "run_at": "2026-11-27T10:00:00Z",
    "periode": {"reglage": "≤2014", "test": "2015–2021"},
    "rows": [
        {
            "model": "naif",
            "horizon": 1,
            "n_mines": 120,
            "n_points": 700,
            "r": 0.91,
            "r_lo": 0.88,
            "r_hi": 0.93,
            "rho": 0.9,
            "rho_lo": 0.87,
            "rho_hi": 0.92,
            "skill": 0.0,
            "facteur_median": 1.2,
            "bon_sens": float("nan"),
            "couverture80": 0.79,
            "couverture90": 0.9,
        },
        {
            "modele": "tendance",
            "horizon": 2,
            "n_mines": 110,
            "n_points": 600,
            "r": 0.4,
            "r_lo": -0.05,
            "r_hi": 0.6,
            "rho": 0.35,
            "rho_lo": 0.1,
            "rho_hi": 0.5,
            "skill": 0.05,
            "facteur_median": 1.4,
            "bon_sens": 0.6,
            "couverture80": 0.81,
            "couverture90": 0.89,
        },
    ],
    "retenu": {1: "naif", 2: "tendance"},
    "methode": "walk-forward strict",
}


def write_bt(env, d=BT) -> None:
    p = env / "forecast" / "backtest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), "utf-8")  # NaN écrit tel quel (JSON non strict)


# -- outils ---------------------------------------------------------------------------------------------
def test_finite_cleans_everything():
    out = api._finite(
        {
            "a": float("nan"),
            "b": np.float64(np.inf),
            "c": np.int64(3),
            "d": [np.nan, 1.5],
            "e": pd.NA,
            "f": pd.Timestamp("2024-01-02"),
            "g": np.bool_(True),
            4: (1, None),
            "h": pd.NaT,
            "i": np.array([1.0, np.nan]),
        }
    )
    assert out == {
        "a": None,
        "b": None,
        "c": 3,
        "d": [None, 1.5],
        "e": None,
        "f": "2024-01-02T00:00:00",
        "g": True,
        "4": [1, None],
        "h": None,
        "i": [1.0, None],
    }
    json.dumps(out, allow_nan=False)


def test_routes_registered():
    paths = {r.path for r in api.app.routes}
    for p in (
        "/api/forecast/backtest",
        "/api/forecast/site",
        "/api/companies",
        "/api/company",
        "/api/companies/correlations",
        "/api/correlations/summary",
    ):
        assert p in paths


# -- /api/forecast/backtest ---------------------------------------------------------------------------
def test_backtest_absent_gives_empty_with_message(client):
    d = client.get("/api/forecast/backtest").json()
    assert d["rows"] == [] and d["retenu"] == {} and d["run_at"] is None
    assert d["periode"] == {"reglage": "≤2014", "test": "2015–2021"}
    assert "strate forecast backtest" in d["message"] and d["methode"]


def test_backtest_reads_file_nan_to_null(client, env):
    write_bt(env)
    r = client.get("/api/forecast/backtest")
    assert r.status_code == 200
    d = r.json()
    assert "message" not in d and d["run_at"] == BT["run_at"] and d["methode"] == "walk-forward strict"
    assert d["retenu"] == {"1": "naif", "2": "tendance"}
    a, b = d["rows"]
    assert set(a) == set(api.BT_ROW)
    assert a["bon_sens"] is None and a["r_lo"] == 0.88
    assert b["model"] == "tendance"  # alias « modele »


def test_backtest_insufficient_reason(client, env):
    write_bt(env, {"statut": "insuffisant", "raison": "trop peu d'années pour un rétro-test"})
    d = client.get("/api/forecast/backtest").json()
    assert d["rows"] == [] and d["message"] == "trop peu d'années pour un rétro-test"


def test_backtest_unreadable_file(client, env):
    p = env / "forecast" / "backtest.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ en cours d'écri", "utf-8")
    d = client.get("/api/forecast/backtest").json()
    assert d["rows"] == [] and "illisibles" in d["message"]


# -- /api/forecast/site ---------------------------------------------------------------------------------
QUAL = {
    "model": "tendance",
    "horizon": 1,
    "r": 0.5,
    "r_lo": 0.2,
    "r_hi": 0.7,
    "rho": 0.4,
    "rho_lo": -0.01,
    "rho_hi": 0.6,
    "skill": np.float64(0.1),
    "skill_lo": 0.02,
    "skill_hi": 0.15,
    "facteur_median": 1.3,
    "couverture80": 0.8,
    "couverture90": 0.9,
    "n": np.int64(40),
    "n_mines": 12,
}
SITE = {
    "site_id": "kalgoorlie",
    "serie": "minerai extrait",
    "derniere_annee": 2024,
    "historique": [{"year": 2019, "value": 1e6}, {"year": 2020, "value": np.nan}],
    "prevision": [
        {
            "year": 2026,
            "model": "tendance",
            "value": 1.1e6,
            "lo80": 0.9e6,
            "hi80": 1.3e6,
            "lo90": 0.8e6,
            "hi90": float("inf"),
        },
        {"year": 2029, "model": "naif", "value": 1e6, "lo80": 1, "hi80": 2, "lo90": 1, "hi90": 3},
    ],
    "qualite": QUAL,
    "qualite_par_horizon": [
        QUAL,
        {**QUAL, "model": "naif", "horizon": 2, "r": None, "r_lo": None, "r_hi": None, "rho": None,
         "r_niveau": 0.98, "r_niveau_lo": 0.97, "r_niveau_hi": 0.99},
    ],
    "suivi_site": {"n": 6, "facteur_median": 1.2, "modele": "tendance", "horizon": 1},
}


def test_forecast_site_ok(client, env, monkeypatch):
    write_bt(env)  # horizons 1 et 2 mesurés
    calls = []
    fake(monkeypatch, "forecast", forecast_site=lambda s: calls.append(s) or SITE)
    d = client.get("/api/forecast/site", params={"site": "kalgoorlie"}).json()
    assert calls == ["kalgoorlie"] and "message" not in d
    assert d["serie"] == "minerai extrait" and d["historique"][1] == {"year": 2020, "value": None}
    p0, p1 = d["prevision"]
    assert p0["hi90"] is None and p0["model"] == "tendance"
    assert (p0["horizon"], p0["mesure"]) == (2, True)  # 2026 − 2024 ; h = 2 mesuré dans le rétro-test
    assert (p1["horizon"], p1["mesure"]) == (5, False)  # au-delà des horizons mesurés
    q = d["qualite"]
    assert set(q) == set(api.QUALITE)
    assert (q["r"], q["r_lo"], q["r_hi"], q["rho_lo"], q["rho_hi"]) == (0.5, 0.2, 0.7, -0.01, 0.6)
    assert (q["skill"], q["skill_lo"], q["skill_hi"], q["n"], q["n_mines"]) == (0.1, 0.02, 0.15, 40, 12)
    assert (q["model"], q["horizon"], q["portee"]) == ("tendance", 1, "global")
    assert "global" in q["note"].lower()
    naif = d["qualite_par_horizon"][1]
    assert naif["r"] is None and naif["r_niveau_lo"] == 0.97 and "aucune variation" in naif["note"]
    s = d["suivi_site"]
    assert set(s) == {*api.SUIVI, "portee"}
    assert (s["model"], s["horizon"], s["n"], s["facteur_median"]) == ("tendance", 1, 6, 1.2)
    assert s["portee"] == "mine"
    assert s["note"] == api.NOTE_SUIVI  # le module n'en donne pas : la note par défaut dit pourquoi pas d'IC
    json.dumps(d, allow_nan=False)


def test_forecast_site_keeps_module_site_caveat(client, env, monkeypatch):
    """Le suivi propre à la mine n'a pas d'IC (n ≤ 7, années dépendantes) : la note du module, la période
    et le facteur maximal arrivent jusqu'au client, comme l'unité de n et la période de la qualité globale."""
    suivi = {"modele": "naif", "horizon": 1, "periode": "2015–2021", "n": 6, "facteur_median": 1.08,
             "facteur_max": 1.25, "note": "Description de cette mine : pas d'IC, pas un test."}
    qual = {**QUAL, "unite_n": "points (mine × année cible)", "periode": "2015–2021"}
    fake(monkeypatch, "forecast", forecast_site=lambda s: {**SITE, "qualite": qual, "suivi_site": suivi})
    write_bt(env)
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert d["suivi_site"] == {"model": "naif", "horizon": 1, "periode": "2015–2021", "n": 6,
                               "unite_n": "années cibles de cette mine", "facteur_median": 1.08,
                               "facteur_max": 1.25,
                               "note": "Description de cette mine : pas d'IC, pas un test.", "portee": "mine"}
    q = d["qualite"]
    assert (q["unite_n"], q["periode"], q["n"]) == ("points (mine × année cible)", "2015–2021", 40)


def test_forecast_site_module_horizon_and_mesure_win(client, env, monkeypatch):
    row = {**SITE["prevision"][0], "horizon": 7, "mesure": False}
    fake(monkeypatch, "forecast", forecast_site=lambda s: {**SITE, "prevision": [row]})
    write_bt(env)
    p = client.get("/api/forecast/site", params={"site": "x"}).json()["prevision"][0]
    assert (p["horizon"], p["mesure"]) == (7, False)


def test_forecast_site_tries_twin(client, monkeypatch):
    monkeypatch.setattr(screener, "twins", lambda: {"greenstone": "icmm-42"})
    fake(monkeypatch, "forecast", forecast_site=lambda s: SITE if s == "icmm-42" else None)
    assert (
        client.get("/api/forecast/site", params={"site": "greenstone"}).json()["serie"] == "minerai extrait"
    )


def test_forecast_site_error_on_first_id_still_tries_twin(client, monkeypatch):
    """Une exception sur l'id demandé ne doit pas empêcher d'essayer la jumelle."""
    monkeypatch.setattr(screener, "twins", lambda: {"greenstone": "icmm-42"})

    def fn(s):
        if s == "greenstone":
            raise KeyError("pas dans le panel")
        return SITE

    fake(monkeypatch, "forecast", forecast_site=fn)
    d = client.get("/api/forecast/site", params={"site": "greenstone"}).json()
    assert d["serie"] == "minerai extrait" and "indisponible" not in d.get("message", "")


def test_forecast_site_no_series_is_null(client, monkeypatch):
    fake(monkeypatch, "forecast", forecast_site=lambda s: None)
    r = client.get("/api/forecast/site", params={"site": "x"})
    assert r.status_code == 200 and r.json() is None


def test_forecast_site_module_missing_or_failing(client, monkeypatch):
    fake(monkeypatch, "forecast")  # module sans forecast_site (en cours d'écriture)
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert d["serie"] is None and d["prevision"] == [] and d["message"]

    def boom(s):
        raise FileNotFoundError("production absente")

    fake(monkeypatch, "forecast", forecast_site=boom)
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert d["historique"] == [] and "production absente" in d["message"]
    no_module(monkeypatch, "forecast")
    assert client.get("/api/forecast/site", params={"site": "x"}).json()["message"]


def test_forecast_site_without_backtest_keeps_history(client, env, monkeypatch):
    """forecast_site est en lecture seule : sans backtest.json il rend l'historique et un message ; l'API le
    transmet au lieu d'une structure vide."""
    calls = []
    ro = {"serie": "minerai extrait", "historique": SITE["historique"], "prevision": [], "qualite": None,
          "message": "Rétro-test pas encore lancé"}
    fake(monkeypatch, "forecast", forecast_site=lambda s: calls.append(s) or ro)
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert calls == ["x"] and len(d["historique"]) == 2 and d["prevision"] == []
    assert d["qualite"] is None and d["message"] == "Rétro-test pas encore lancé"
    fake(monkeypatch, "forecast", forecast_site=lambda s: {**ro, "message": None})
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert "strate forecast backtest" in d["message"]


def test_forecast_site_without_backtest_quality_is_null(client, env, monkeypatch):
    """Rétro-test absent : le module rend qualite = {"n": 0} et suivi_site = {"n": 0}. Sans aucun score,
    l'API répond null (et non une qualité vide annotée « Score global… ») ; le message dit pourquoi."""
    ro = {"serie": "minerai extrait", "historique": SITE["historique"], "prevision": [], "qualite": {"n": 0},
          "qualite_par_horizon": [{"n": 0}], "suivi_site": {"n": 0}, "message": api.MSG_BACKTEST}
    fake(monkeypatch, "forecast", forecast_site=lambda s: ro)
    d = client.get("/api/forecast/site", params={"site": "x"}).json()
    assert d["qualite"] is None and d["qualite_par_horizon"] == [] and d["suivi_site"] is None
    assert d["message"] == api.MSG_BACKTEST
    # une mine sans année de test garde son suivi (n = 0, note du module) : le modèle est connu
    fake(monkeypatch, "forecast", forecast_site=lambda s: {**SITE, "suivi_site": {"modele": "naif", "n": 0}})
    write_bt(env)
    s = client.get("/api/forecast/site", params={"site": "x"}).json()["suivi_site"]
    assert (s["model"], s["n"], s["note"]) == ("naif", 0, api.NOTE_SUIVI)


def test_forecast_site_note_as_message(client, env, monkeypatch):
    fake(
        monkeypatch,
        "forecast",
        forecast_site=lambda s: {**SITE, "prevision": [], "note": "série trop ancienne"},
    )
    write_bt(env)
    assert client.get("/api/forecast/site", params={"site": "x"}).json()["message"] == "série trop ancienne"


def test_backtest_naive_rows_get_note(client, env):
    row = {**BT["rows"][0], "r": None, "rho": None, "r_niveau": 0.98, "r_niveau_lo": 0.97,
           "r_niveau_hi": 0.99, "skill_lo": 0.0, "skill_hi": 0.0}
    write_bt(env, {**BT, "rows": [row]})
    r = client.get("/api/forecast/backtest").json()["rows"][0]
    assert r["r"] is None and r["r_niveau_lo"] == 0.97 and r["skill_hi"] == 0.0
    assert "aucune variation" in r["note"]


# -- /api/companies ---------------------------------------------------------------------------------------
TABLE = pd.DataFrame(
    [
        {
            "id": "nem",
            "nom": "Newmont",
            "pays": "United States",
            "tickers": "NYSE:NEM, TSX:NGT",
            "cik": "0001164727",
            "n_mines": 14,
            "capex_dernier": 3.0e9,
            "capex_annee": 2025,
            "effectifs": np.nan,
            "effectifs_annee": None,
            "ca_dernier": 2.2e10,
            "devise": "USD",
            "investissements_12m": 2,
        },
        {
            "id": "vale",
            "nom": "Vale",
            "pays": "Brazil",
            "tickers": ["NYSE:VALE"],
            "cik": None,
            "n_mines": 9,
            "capex_dernier": 6.0e9,
            "capex_annee": 2024,
            "effectifs": 64000,
            "effectifs_annee": 2024,
            "ca_dernier": 4.0e10,
            "devise": "USD",
            "investissements_12m": 0,
        },
        {
            "id": "codelco",
            "nom": "Corporación Nacional del Cobre",
            "pays": "Chile",
            "tickers": None,
            "cik": None,
            "n_mines": 8,
            "capex_dernier": np.nan,
            "capex_annee": None,
            "effectifs": None,
            "effectifs_annee": None,
            "ca_dernier": None,
            "devise": None,
            "investissements_12m": None,
        },
    ]
)


def test_companies_table_filters(client, monkeypatch):
    fake(monkeypatch, "companies", table=lambda: TABLE)
    d = client.get("/api/companies").json()
    assert d["total"] == 3 and "message" not in d
    nem = d["rows"][0]
    assert set(nem) == set(api.CO_ROW)
    assert nem["tickers"] == ["NYSE:NEM", "TSX:NGT"] and nem["effectifs"] is None
    assert d["rows"][2]["tickers"] == [] and d["rows"][2]["capex_dernier"] is None
    # recherche sans accent, par ticker, par pays (normalisé), limite
    assert [r["id"] for r in client.get("/api/companies", params={"q": "corporacion"}).json()["rows"]] == [
        "codelco"
    ]
    assert client.get("/api/companies", params={"q": "ngt"}).json()["rows"][0]["id"] == "nem"
    assert [
        r["id"] for r in client.get("/api/companies", params={"pays": "Brazil,Chile"}).json()["rows"]
    ] == ["vale", "codelco"]
    d = client.get("/api/companies", params={"limit": 1}).json()
    assert d["total"] == 3 and len(d["rows"]) == 1


def test_companies_pays_aliases_and_investment_count(client, monkeypatch):
    """Filtre pays par nom canonique (français, ISO2, casse) ; investissements_12m entier, 0 si aucun."""
    fake(monkeypatch, "companies", table=lambda: TABLE.assign(investissements_12m=[2.0, 0, None]))
    ids = [r["id"] for r in client.get("/api/companies", params={"pays": "brésil, CL"}).json()["rows"]]
    assert ids == ["vale", "codelco"]
    assert client.get("/api/companies", params={"pays": "united states"}).json()["total"] == 1
    assert client.get("/api/companies", params={"pays": "Atlantide"}).json()["total"] == 0
    counts = [r["investissements_12m"] for r in client.get("/api/companies").json()["rows"]]
    assert counts == [2, 0, 0] and all(type(c) is int for c in counts)
    fake(monkeypatch, "companies", table=lambda: TABLE.drop(columns=["investissements_12m"]))
    assert client.get("/api/companies").json()["rows"][0]["investissements_12m"] is None  # inconnu


def test_companies_legacy_names_and_list(client, monkeypatch):
    fake(
        monkeypatch,
        "companies",
        table=lambda: [
            {
                "id": "bhp",
                "name": "BHP",
                "capex": 9.4e9,
                "year": 2025,
                "employees": 9e4,
                "employees_year": 2024,
                "revenue": 5e10,
                "currency": "USD",
                "tickers": ["ASX:BHP"],
            }
        ],
    )
    r = client.get("/api/companies").json()["rows"][0]
    assert (r["nom"], r["capex_dernier"], r["capex_annee"], r["effectifs"], r["ca_dernier"], r["devise"]) == (
        "BHP",
        9.4e9,
        2025,
        9e4,
        5e10,
        "USD",
    )


def test_companies_empty_or_broken(client, monkeypatch):
    fake(monkeypatch, "companies", table=lambda: pd.DataFrame())
    d = client.get("/api/companies").json()
    assert d == {"total": 0, "rows": [], "message": api.MSG_SOCIETES}

    def boom():
        raise FileNotFoundError("data/companies")

    fake(monkeypatch, "companies", table=boom)
    assert client.get("/api/companies").json()["rows"] == []
    no_module(monkeypatch, "companies")
    d = client.get("/api/companies").json()
    assert d["total"] == 0 and d["message"]


# -- /api/company ----------------------------------------------------------------------------------------
DOSSIER = {
    "profil": {"id": "nem", "nom": "Newmont", "qid": "Q1132", "cik": "0001164727", "x": float("nan")},
    "mines": [
        {"site_id": "boddington", "nom": "Boddington", "part": 1.0},
        {"id": "penasquito", "name": "Peñasquito", "share": np.nan},
    ],
    "faits": [
        {
            "indicateur": "capex",
            "annee": 2024,
            "valeur": 3.0e9,
            "unite": "USD",
            "devise": "USD",
            "source": "https://data.sec.gov/...",
            "formulaire": "10-K",
        }
    ],
    "investissements": [
        {
            "date": "2026-05-01",
            "montant": 1.2e9,
            "devise": "USD",
            "categorie": "expansion",
            "titre": "Newmont investit 1,2 Md$",
            "url": "https://ex.com/a",
            "site_id": None,
        }
    ],
    "correlations": [
        {
            "x": "capex",
            "y": "production",
            "decalage": 1,
            "r": 0.3,
            "r_lo": -0.1,
            "r_hi": 0.6,
            "rho": 0.25,
            "n": 12,
        }
    ],
}


def test_company_ok(client, monkeypatch):
    fake(monkeypatch, "companies", company=lambda i: DOSSIER if i == "nem" else None, table=lambda: TABLE)
    d = client.get("/api/company", params={"id": "nem"}).json()
    assert d["profil"]["qid"] == "Q1132" and d["profil"]["x"] is None
    assert d["mines"][1] == {"site_id": "penasquito", "nom": "Peñasquito", "part": None}
    assert d["faits"][0]["formulaire"] == "10-K" and d["investissements"][0]["categorie"] == "expansion"
    assert d["correlations"][0] == {
        "x": "capex",
        "y": "production",
        "decalage": 1,
        "r": 0.3,
        "r_lo": -0.1,
        "r_hi": 0.6,
        "rho": 0.25,
        "rho_lo": None,
        "rho_hi": None,
        "n": 12,
        "verdict_r": None,
        "verdict_rho": None,
    }
    assert client.get("/api/company", params={"id": "inconnue"}).status_code == 404


def test_company_contract_correlations_and_announcements(client, monkeypatch):
    """Format du contrat : lignes par société (lag, n, IC de r et de ρ, un verdict par coefficient) et
    annonces avec categorie / montant / devise ; l'ancien event_type sert de catégorie."""
    dossier = {
        "profil": {"id": "bhp"},
        "mines": [],
        "correlations": [
            {"x": "CAPEX", "y": "production", "lag": 0, "n": 9, "r": 0.4, "r_lo": 0.1, "r_hi": 0.7,
             "rho": 0.3, "rho_lo": -0.2, "rho_hi": 0.6, "verdict_r": "non démontré : q ≥ 0,10",
             "verdict_rho": "non démontré : l'IC 90 % de ρ contient 0"},
            {"x": "CAPEX", "y": "emprise", "decalage": 1, "n_points": 6, "rho": 0.1, "verdict": "trop peu"},
        ],
        "announcements": [
            {"date": "2026-03-01", "title": "BHP lance l'extension", "event_type": "operations",
             "source_url": "u"},
            {"date": "2026-04-01", "titre": "t", "categorie": "exploration", "montant": 2e8, "devise": "AUD"},
        ],
    }
    fake(monkeypatch, "companies", company=lambda i: dossier)
    d = client.get("/api/company", params={"id": "bhp"}).json()
    a, b = d["correlations"]
    assert (a["decalage"], a["n"], a["rho_lo"], a["rho_hi"]) == (0, 9, -0.2, 0.6)
    assert a["verdict_r"].startswith("non démontré") and "ρ" in a["verdict_rho"]
    assert (b["decalage"], b["n"], b["verdict_rho"], b["verdict_r"]) == (1, 6, "trop peu", None)
    i0, i1 = d["investissements"]
    assert (i0["categorie"], i0["montant"], i0["devise"], i0["titre"]) == ("operations", None, None,
                                                                          "BHP lance l'extension")
    assert (i1["categorie"], i1["montant"], i1["devise"]) == ("exploration", 2e8, "AUD")


def test_company_without_data_or_module(client, monkeypatch):
    fake(monkeypatch, "companies", company=lambda i: None, table=lambda: [])
    r = client.get("/api/company", params={"id": "nem"})
    assert r.status_code == 200 and r.json()["mines"] == [] and r.json()["message"] == api.MSG_SOCIETES
    no_module(monkeypatch, "companies")
    d = client.get("/api/company", params={"id": "nem"}).json()
    assert d["profil"] == {} and d["message"]


def test_company_legacy_dossier(client, monkeypatch):
    fake(
        monkeypatch,
        "companies",
        dossier=lambda i: {
            "company": {"id": i, "name": "Vale"},
            "mines": [],
            "financials": [
                {
                    "metric": "capex",
                    "year": 2024,
                    "value": 6e9,
                    "unit": "USD",
                    "source_url": "u",
                    "form": "20-F",
                }
            ],
            "announcements": [{"date": "2026-01-01", "title": "t", "source_url": "u2", "site_id": "s"}],
        },
    )
    d = client.get("/api/company", params={"id": "vale"}).json()
    assert d["profil"]["name"] == "Vale"
    assert d["faits"][0] == {
        "indicateur": "capex",
        "annee": 2024,
        "valeur": 6e9,
        "unite": "USD",
        "devise": None,
        "source": "u",
        "formulaire": "20-F",
    }
    assert d["investissements"][0]["titre"] == "t" and d["investissements"][0]["url"] == "u2"


def test_company_wide_financials_employees_and_mine_names(client, monkeypatch):
    monkeypatch.setattr(api, "_site_names", lambda ids: {"icmm1": "Mine Une"})
    src = {"url": "https://www.sec.gov/a", "form": "20-F", "concept": "ifrs-full:Revenue"}
    fake(
        monkeypatch,
        "companies",
        dossier=lambda i: {
            "company": {"id": i, "name": "BHP"},
            "mines": [{"site": "icmm1", "roles": ["exploitant"], "share": 0.5}, {"site": "icmm2"}],
            "financials": [
                {
                    "year": 2024,
                    "revenue": 5e10,
                    "capex": np.nan,
                    "capex_revenue": 0.2,
                    "capex_growth": -0.1,
                    "capex_per_employee": 1e5,
                    "currency": "USD",
                    "sources": {"revenue": src},
                }
            ],
            "employees": [
                {
                    "year": 2024,
                    "employees": 9e4,
                    "source_kind": "sec_dei",
                    "source_url": "u",
                    "accepted": True,
                },
                {"year": 2023, "employees": 9e9, "source_kind": "wikidata", "accepted": False},
            ],
        },
    )
    d = client.get("/api/company", params={"id": "bhp"}).json()
    assert d["mines"] == [
        {"site_id": "icmm1", "nom": "Mine Une", "part": 0.5},
        {"site_id": "icmm2", "nom": None, "part": None},
    ]
    f = {x["indicateur"]: x for x in d["faits"]}
    assert f["revenue"] == {
        "indicateur": "revenue",
        "annee": 2024,
        "valeur": 5e10,
        "unite": "USD",
        "devise": "USD",
        "source": "https://www.sec.gov/a",
        "formulaire": "20-F",
    }
    assert "capex" not in f  # NaN : pas de fait
    assert (f["capex_revenue"]["unite"], f["capex_growth"]["unite"], f["capex_per_employee"]["unite"]) == (
        "ratio",
        "variation",
        "USD/salarié",
    )
    assert f["effectifs"]["valeur"] == 9e4 and f["effectifs"]["formulaire"] == "sec_dei"
    assert sum(x["indicateur"] == "effectifs" for x in d["faits"]) == 1  # valeur refusée écartée


# -- /api/companies/correlations ---------------------------------------------------------------------------
CORR = [
    {
        "x": "capex",
        "y": "production",
        "decalage": 1,
        "r": 0.31,
        "r_lo": 0.05,
        "r_hi": 0.52,
        "rho": 0.28,
        "rho_lo": 0.02,
        "rho_hi": 0.5,
        "n_societes": 14,
        "n_points": 120,
        "verdict": "lien positif",
    },
    {
        "x": "effectifs",
        "y": "production",
        "lag": 0,
        "pearson_r": np.nan,
        "pearson_lo": None,
        "pearson_hi": None,
        "spearman_rho": 0.1,
        "spearman_lo": -0.2,
        "spearman_hi": 0.35,
        "n_companies": 9,
        "n": 40,
        "verdict": "non démontré",
    },
]


def test_companies_correlations(client, monkeypatch):
    fake(monkeypatch, "companies", correlations=lambda: pd.DataFrame(CORR), METHODE="bootstrap sociétés")
    d = client.get("/api/companies/correlations").json()
    assert d["methode"] == "bootstrap sociétés" and "message" not in d
    a, b = d["rows"]
    assert set(a) == set(api.CO_CORR) and a["n_points"] == 120
    assert (b["decalage"], b["r"], b["rho"], b["rho_lo"], b["n_societes"], b["n_points"]) == (
        0,
        None,
        0.1,
        -0.2,
        9,
        40,
    )


def test_companies_correlations_contract_columns(client, monkeypatch):
    """Un verdict par coefficient, la q-valeur BH et n sociétés ; lignes par société exclues de la table."""
    rows = [
        {**CORR[0], "portee": "ensemble", "q_value": 0.04, "verdict_r": "lien positif démontré",
         "verdict_rho": "non démontré : q ≥ 0,10", "verdict": "non démontré : q ≥ 0,10"},
        {**CORR[0], "portee": "societe", "company_id": "bhp", "verdict": "x"},
        {**CORR[1], "verdict": "non démontré (ρ)"},  # ancien format : verdict = verdict de ρ
    ]
    fake(monkeypatch, "companies", correlations=lambda: {"rows": rows, "methode": "m"})
    a, b = client.get("/api/companies/correlations").json()["rows"]
    assert (a["q_value"], a["verdict_r"], a["verdict_rho"], a["n_societes"]) == (
        0.04, "lien positif démontré", "non démontré : q ≥ 0,10", 14)
    assert (b["verdict_r"], b["verdict_rho"], b["verdict"], b["q_value"]) == (
        None, "non démontré (ρ)", "non démontré (ρ)", None)


def test_companies_legacy_demontre_without_q_is_downgraded(client, monkeypatch):
    """Ancien format : un « démontré » sans q-valeur (ou avec q ≥ 0,10) ne passe pas tel quel ; un
    « démontré » avec q < 0,10 est transmis. Même garde pour la table globale, le résumé et la fiche."""
    legacy = {**CORR[0], "verdict": "lien positif démontré (IC 90 % de ρ au-dessus de 0)", "rho_lo": -0.1,
              "rho_hi": 0.3}
    hors_ic = {**CORR[0], "x": "effectifs", "verdict_r": "lien positif démontré", "q_r": 0.3,
               "verdict_rho": "lien positif démontré", "q_rho": 0.04}
    fake(monkeypatch, "companies", correlations=lambda: {"rows": [legacy, hors_ic], "methode": "m"},
         company=lambda i: {"profil": {"id": i}, "correlations": [{**legacy, "q_value": None}]})
    a, b = client.get("/api/companies/correlations").json()["rows"]
    assert a["q_value"] is None and not api._demontre(a["verdict_rho"]) and not api._demontre(a["verdict"])
    assert a["verdict_rho"] == "non démontré : l'IC 90 % de ρ contient 0"
    assert b["verdict_r"].startswith("non démontré : IC 90 % de r hors de 0") and "q = 0.30" in b["verdict_r"]
    assert b["verdict_rho"] == "lien positif démontré"  # q_rho = 0,04 : le verdict du module tient
    summ = client.get("/api/correlations/summary").json()["rows"]
    soc = [r for r in summ if r["domaine"] == "sociétés"]
    assert [api._demontre(r["verdict"]) for r in soc] == [False, False, False, True]
    c = client.get("/api/company", params={"id": "bhp"}).json()["correlations"][0]
    assert not api._demontre(c["verdict_rho"]) and "q-valeur absente" not in c["verdict_rho"]
    assert c["verdict_rho"] == "non démontré : l'IC 90 % de ρ contient 0"
    assert c["verdict_r"] is None


def test_companies_correlations_empty(client, monkeypatch):
    fake(monkeypatch, "companies", correlations=lambda: [])
    d = client.get("/api/companies/correlations").json()
    assert d["rows"] == [] and d["methode"] == api.METHODE_SOCIETES and d["message"]
    no_module(monkeypatch, "companies")
    assert client.get("/api/companies/correlations").json()["rows"] == []


# -- /api/correlations/summary ------------------------------------------------------------------------
CAL = {
    "paires_annees": 326,
    "taille": {
        "rho": 0.48,
        "ic90": [0.35, 0.6],
        "mines": 107,
        "fenetre_seule": {"rho": 0.51, "ic90_bas": 0.37, "ic90_haut": 0.63, "mines": 107},
    },
    "variation": {"rho": 0.11, "ic90": [-0.02, 0.18]},
    "modele_emprise": {"minerai": {"rho": 0.58, "n": 245}},
    "radar": {"mines_test": 73, "variation": [0.11, -0.03, 0.25]},
}


def veille_db(env) -> None:
    p = env / "veille" / "veille.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE correlations (run_at TEXT, scope TEXT, event_type TEXT, signal TEXT, "
        "lag_months INTEGER, n_mines INTEGER, n_events INTEGER, lift REAL, lift_lo REAL, lift_hi REAL, "
        "rho REAL, rho_lo REAL, rho_hi REAL, p_value REAL, q_value REAL, verdict TEXT, details TEXT)"
    )
    rows = [
        (
            "2026-11-01T00:00:00Z",
            "global",
            "greve",
            "bare_km2",
            0,
            12,
            30,
            2.1,
            1.3,
            3.4,
            None,
            None,
            None,
            0.01,
            0.05,
            "lien démontré",
            None,
        ),
        (
            "2026-11-01T00:00:00Z",
            "global",
            "volume",
            "activity_db",
            1,
            20,
            300,
            None,
            None,
            None,
            0.12,
            -0.01,
            0.25,
            0.2,
            0.4,
            "non démontré",
            None,
        ),
        (
            "2026-10-01T00:00:00Z",
            "global",
            "ancien",
            "bare_km2",
            0,
            1,
            1,
            9.0,
            1.0,
            9.0,
            None,
            None,
            None,
            0.5,
            0.5,
            "vieux calcul",
            None,
        ),
    ]
    con.executemany("INSERT INTO correlations VALUES (" + ",".join("?" * 17) + ")", rows)
    con.commit()
    con.close()


def test_summary_all_sources(client, env, monkeypatch):
    monkeypatch.setattr(api, "calibration", lambda: CAL)
    veille_db(env)
    write_bt(env)
    fake(monkeypatch, "companies", correlations=lambda: CORR)
    d = client.get("/api/correlations/summary").json()
    rows = d["rows"]
    assert {x["domaine"]: x["n"] for x in d["domaines"]} == {
        "calibration": 5,
        "veille": 2,
        "sociétés": 3,
        "prévisions": 4,
    }
    keys = {"domaine", "x", "y", "coefficient", "type", "lo", "hi", "n", "unite_n", "n_bootstrap_unit",
            "unite_bootstrap", "verdict", "source"}
    assert all(set(r) == keys for r in rows)
    assert all(r["coefficient"] is not None and r["type"] in ("r", "rho", "phi", "lift") for r in rows)
    assert all(r["unite_n"] and r["verdict"] for r in rows)
    json.dumps(d, allow_nan=False)

    cal = [r for r in rows if r["domaine"] == "calibration"]
    taille = cal[0]
    assert (taille["coefficient"], taille["lo"], taille["hi"], taille["n"]) == (0.48, 0.35, 0.6, 107)
    assert taille["verdict"].startswith("lien positif")
    var = next(r for r in cal if r["x"].startswith("variation annuelle"))
    assert var["n"] == 326 and var["verdict"].startswith("non démontré")
    emp = next(r for r in cal if r["x"].startswith("emprise"))
    assert emp["lo"] < 0.58 < emp["hi"] and "Fisher" in emp["source"]
    se = math.sqrt((1 + 0.58**2 / 2) / 242)
    assert emp["lo"] == pytest.approx(math.tanh(math.atanh(0.58) - 1.6449 * se), abs=1e-3)
    radar = next(r for r in cal if "radar" in r["x"])
    assert radar["n"] == 73 and radar["lo"] == -0.03

    ve = [r for r in rows if r["domaine"] == "veille"]  # seul le dernier calcul compte
    lift = next(r for r in ve if r["type"] == "lift")
    assert (lift["coefficient"], lift["lo"], lift["hi"], lift["n"], lift["verdict"]) == (
        2.1,
        1.3,
        3.4,
        30,
        "lien démontré",
    )
    rho = next(r for r in ve if r["type"] == "rho")
    assert "décalage +1 mois" in rho["y"] and rho["x"] == "volume d'actualité"

    assert (lift["unite_n"], lift["n_bootstrap_unit"], lift["unite_bootstrap"]) == ("mois-événements", 12,
                                                                                    "mines")
    assert rho["verdict"] == "non démontré" and rho["n_bootstrap_unit"] == 20  # volume : verdict de ρ

    so = [r for r in rows if r["domaine"] == "sociétés"]
    assert [r["type"] for r in so] == ["r", "rho", "rho"]  # r NaN de la 2e ligne retiré
    assert "décalage +1 an" in so[0]["y"] and so[0]["n"] == 120
    assert (so[0]["unite_n"], so[0]["n_bootstrap_unit"], so[0]["unite_bootstrap"]) == (
        "paires société-année", 14, "sociétés")
    # le verdict « lien positif » du module porte sur ρ : la ligne r a le sien, jamais « démontré » sans q
    assert so[1]["verdict"] == "lien positif" and so[0]["verdict"].startswith("non démontré")

    pr = [r for r in rows if r["domaine"] == "prévisions"]
    assert {r["type"] for r in pr} == {"r", "rho"}
    t2 = next(r for r in pr if "tendance" in r["x"] and r["type"] == "r")
    assert t2["verdict"].startswith("non démontré") and t2["n"] == 600 and "2 ans" in t2["x"]
    assert t2["x"].startswith("variation prévue") and "variation réelle du log de la production" in t2["y"]
    assert (t2["unite_n"], t2["n_bootstrap_unit"]) == ("points mine-année", 110)
    assert t2["unite_bootstrap"] == "mines"


def test_summary_companies_label(client, monkeypatch):
    row = {**CORR[0], "label": "Croissance du CAPEX ↔ croissance de la production (année suivante)"}
    fake(monkeypatch, "companies", correlations=lambda: [row])
    so = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "sociétés"]
    assert (
        so[0]["x"] == "Croissance du CAPEX" and so[0]["y"] == "croissance de la production (année suivante)"
    )


def test_summary_robust_without_data(client, monkeypatch):
    def boom():
        raise RuntimeError("calibration cassée")

    monkeypatch.setattr(api, "calibration", boom)
    no_module(monkeypatch, "companies")
    d = client.get("/api/correlations/summary").json()
    assert d["rows"] == []
    msgs = {x["domaine"]: x["message"] for x in d["domaines"]}
    assert "calibration cassée" in msgs["calibration"] and msgs["veille"] and msgs["prévisions"]


def test_approx_ci_edges():
    assert api._approx_ci(None, 50) == (None, None)
    assert api._approx_ci(0.5, 3) == (None, None)
    assert api._approx_ci(1.0, 50) == (None, None)
    lo, hi = api._approx_ci(0.0, 103)
    assert lo == pytest.approx(-hi) and hi == pytest.approx(math.tanh(1.6449 * math.sqrt(1 / 100)), abs=1e-4)


def test_verdicts():
    assert api._verdict(0.1, 0.3).startswith("lien positif")
    assert api._verdict(-0.3, -0.1).startswith("lien négatif")
    assert api._verdict(-0.1, 0.3).startswith("non démontré")
    assert api._verdict(1.2, 3.0, 1.0, "lift").startswith("risque accru")
    assert "contient 1" in api._verdict(0.8, 3.0, 1.0, "lift")
    assert api._verdict(None, 0.2, sym="r") == "IC 90 % de r indisponible"
    assert api._verdict_sans_test(0.1, 0.3, 0.0, "r", "q absente").startswith("non démontré")
    assert api._verdict_sans_test(-0.1, 0.3, 0.0, "r", "q absente").startswith("non démontré")


def test_summary_verdict_from_each_row_own_ci(client, env, monkeypatch):
    """Régression : le verdict du module calculé sur un coefficient n'est plus recopié sur l'autre."""
    row = {**CORR[0], "r_lo": 0.01, "r_hi": 0.25, "rho_lo": -0.1, "rho_hi": 0.3,
           "verdict": "lien positif démontré (IC 90 % de ρ au-dessus de 0)"}
    fake(monkeypatch, "companies", correlations=lambda: [row])
    so = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "sociétés"]
    r_row = next(r for r in so if r["type"] == "r")
    assert r_row["verdict"].startswith("non démontré") and "ρ" not in r_row["verdict"]
    row = {**row, "verdict_r": "non démontré : q = 0,3 ≥ 0,10", "verdict_rho": "non démontré : IC contient 0"}
    fake(monkeypatch, "companies", correlations=lambda: [row])
    so = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "sociétés"]
    assert [r["verdict"] for r in so] == ["non démontré : q = 0,3 ≥ 0,10", "non démontré : IC contient 0"]

    # veille : le verdict du lift reste sur la ligne lift ; la ligne φ a verdict_rho ou un repli prudent
    p = env / "veille" / "veille.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE correlations (run_at TEXT, scope TEXT, event_type TEXT, signal TEXT, "
                "lag_months INTEGER, n_mines INTEGER, n_events INTEGER, lift REAL, lift_lo REAL, "
                "lift_hi REAL, rho REAL, rho_lo REAL, rho_hi REAL, p_value REAL, q_value REAL, verdict TEXT, "
                "details TEXT)")
    con.execute("INSERT INTO correlations VALUES ('t', 'global', 'greve', 'bare_km2', 0, 12, 30, 2.1, 1.3, "
                "3.4, 0.08, 0.02, 0.15, 0.01, 0.05, 'lien démontré', NULL)")
    con.commit()
    ve = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "veille"]
    lift, phi = ve
    assert lift["type"] == "lift" and lift["verdict"] == "lien démontré"
    assert phi["type"] == "phi" and phi["verdict"].startswith("non démontré") and "φ" in phi["verdict"]
    con.execute("ALTER TABLE correlations ADD COLUMN verdict_rho TEXT")
    con.execute("UPDATE correlations SET verdict_rho = 'lien démontré (φ)'")
    con.commit()
    con.close()
    phi = client.get("/api/correlations/summary").json()["rows"]
    assert next(r for r in phi if r["type"] == "phi")["verdict"] == "lien démontré (φ)"
    assert client.get("/api/veille/correlations").json()["rows"][0]["verdict_rho"] == "lien démontré (φ)"


def test_summary_forecast_naive_rows_are_levels(client, env):
    row = {**BT["rows"][0], "r": None, "r_lo": None, "r_hi": None, "rho": None, "rho_lo": None,
           "rho_hi": None, "r_niveau": 0.98, "r_niveau_lo": 0.97, "r_niveau_hi": 0.99, "rho_niveau": 0.97,
           "rho_niveau_lo": 0.96, "rho_niveau_hi": 0.98}
    write_bt(env, {**BT, "rows": [row, {**row, "model": "tendance"}]})  # tendance réglée sur φ = 0
    pr = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "prévisions"]
    assert [(r["type"], r["coefficient"], r["lo"]) for r in pr] == [("r", 0.98, 0.97), ("rho", 0.97, 0.96)]
    assert all(r["x"].startswith("niveau prévu (log)") and "niveau réel (log)" in r["y"] for r in pr)
    assert "« naif » = « tendance »" in pr[0]["x"]
    assert all("taille des mines" in r["source"] for r in pr)


def test_summary_forecast_identical_models_share_rows(client, env):
    """Deux modèles aux prévisions identiques (même r, ρ, IC, n) : une paire de lignes, pas un doublon."""
    t = BT["rows"][1]
    write_bt(env, {**BT, "rows": [t, {**t, "modele": "tendance+taille"}, {**t, "r": 0.41}]})
    pr = [r for r in client.get("/api/correlations/summary").json()["rows"] if r["domaine"] == "prévisions"]
    assert len(pr) == 4 and "« tendance » = « tendance+taille »" in pr[0]["x"]
    assert pr[2]["coefficient"] == 0.41 and "=" not in pr[2]["x"]


# -- CLI ----------------------------------------------------------------------------------------------
def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["strate", *argv])
    cli.main()


def test_cli_forecast_backtest_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    fake(monkeypatch, "forecast", backtest=lambda: {**BT, "rows": [{**BT["rows"][0], "r": np.float64(0.91)}]})
    run_cli(monkeypatch, "forecast", "backtest")
    p = tmp_path / "forecast" / "backtest.json"
    saved = json.loads(p.read_text("utf-8"))
    assert saved["rows"][0]["bon_sens"] is None and saved["retenu"] == {"1": "naif", "2": "tendance"}
    out = capsys.readouterr().out
    assert "naif" in out and "0.91 [0.88, 0.93]" in out and "h=1 → naif" in out


def test_cli_forecast_backtest_keeps_module_file(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    p = tmp_path / "forecast" / "backtest.json"

    def backtest():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"run_at": "module"}', "utf-8")
        return {"run_at": "retour"}

    fake(monkeypatch, "forecast", backtest=backtest)
    run_cli(monkeypatch, "forecast", "backtest")
    assert json.loads(p.read_text("utf-8")) == {"run_at": "module"}


def test_cli_forecast_ledger_and_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    fake(monkeypatch, "forecast", write_ledger=lambda: 42)
    run_cli(monkeypatch, "forecast", "ledger")
    assert "42 prévisions" in capsys.readouterr().out
    fake(monkeypatch, "forecast")
    with pytest.raises(SystemExit, match="write_ledger"):
        run_cli(monkeypatch, "forecast", "ledger")


def test_cli_forecast_backtest_table_with_ci(tmp_path, monkeypatch, capsys):
    row = {**BT["rows"][0], "r": None, "r_lo": None, "r_hi": None, "skill_lo": -0.01, "skill_hi": 0.02,
           "r_niveau": 0.98, "r_niveau_lo": 0.97, "r_niveau_hi": 0.99}
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    fake(monkeypatch, "forecast", backtest=lambda: {**BT, "rows": [row]})
    run_cli(monkeypatch, "forecast", "backtest")
    out = capsys.readouterr().out
    assert "- [-, -]" in out  # naïf : pas de r sur les variations, affiché comme tel
    assert "0.00 [-0.01, 0.02]" in out and "0.98 [0.97, 0.99]" in out
    assert "variation réelle du log de la production" in out and "700" in out and "120" in out


def test_cli_companies_refresh(monkeypatch, capsys):
    calls = []
    fake(monkeypatch, "companies", refresh=lambda **kw: calls.append(kw) or {"traitees": 3})
    run_cli(monkeypatch, "companies", "refresh", "--max-companies", "3")
    run_cli(monkeypatch, "companies", "refresh")
    assert calls == [{"max_companies": 3}, {}]
    assert '"traitees": 3' in capsys.readouterr().out
    no_module(monkeypatch, "companies")
    with pytest.raises(SystemExit, match="indisponible"):
        run_cli(monkeypatch, "companies", "refresh")


def test_cli_companies_refresh_falls_back_to_update(monkeypatch, capsys):
    """Ancien module sans refresh : update(limit=…) avec la même borne, au lieu d'un refus."""
    calls = []
    fake(monkeypatch, "companies", update=lambda **kw: calls.append(kw) or {"ok": 1})
    run_cli(monkeypatch, "companies", "refresh", "--max-companies", "7")
    run_cli(monkeypatch, "companies", "refresh")
    assert calls == [{"limit": 7}, {}] and '"ok": 1' in capsys.readouterr().out
    fake(monkeypatch, "companies")
    with pytest.raises(SystemExit, match="ni refresh ni update"):
        run_cli(monkeypatch, "companies", "refresh")


# -- modules réels (fumée) -----------------------------------------------------------------------------
def test_smoke_real_modules():
    """Modules et données réels, en lecture seule : saute proprement si les données locales manquent. Ne
    lance ni rétro-test, ni réseau (forecast_site est en lecture seule ; backtest.json doit exister)."""
    from strate.core import DATA_DIR

    if not (DATA_DIR / "forecast" / "backtest.json").is_file():
        pytest.skip("data/forecast/backtest.json absent : uv run strate forecast backtest")
    c = TestClient(api.app)
    bt = c.get("/api/forecast/backtest").json()
    if not bt["rows"]:
        pytest.skip(f"rétro-test sans mesure : {bt.get('message')}")
    assert all(set(r) == set(api.BT_ROW) for r in bt["rows"])
    assert all(r["note"] for r in bt["rows"] if r["r"] is None)  # naïf : pourquoi pas de r
    s = c.get("/api/correlations/summary").json()
    json.dumps(s, allow_nan=False)
    assert all(r["verdict"] and r["unite_n"] and r["type"] in api.SYM for r in s["rows"])
    pr = [r for r in s["rows"] if r["domaine"] == "prévisions"]
    assert pr and all(r["n_bootstrap_unit"] <= r["n"] for r in pr)
    try:
        from strate import forecast

        sid = str(forecast._panel()["site_id"].iloc[0])
    except Exception as e:  # panel de production absent ou module en cours d'écriture
        pytest.skip(f"panel de production indisponible : {e}")
    d = c.get("/api/forecast/site", params={"site": sid}).json()
    assert d is not None and d["historique"] and json.dumps(d, allow_nan=False)
    if d["qualite"]:
        assert d["qualite"]["portee"] == "global" and set(d["qualite"]) == set(api.QUALITE)
    assert all(isinstance(p["horizon"], int) and isinstance(p["mesure"], bool) for p in d["prevision"])
    co = c.get("/api/companies", params={"limit": 3}).json()
    assert co["total"] >= len(co["rows"]) and all(set(r) == set(api.CO_ROW) for r in co["rows"])
