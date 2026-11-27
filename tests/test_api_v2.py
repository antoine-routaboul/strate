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
SITE = {
    "serie": "minerai extrait",
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
        }
    ],
    "qualite": {
        "r": 0.5,
        "rho": 0.4,
        "skill": np.float64(0.1),
        "facteur_median": 1.3,
        "couverture90": 0.9,
        "n": np.int64(40),
    },
}


def test_forecast_site_ok(client, monkeypatch):
    calls = []
    fake(monkeypatch, "forecast", forecast_site=lambda s: calls.append(s) or SITE)
    d = client.get("/api/forecast/site", params={"site": "kalgoorlie"}).json()
    assert calls == ["kalgoorlie"]
    assert d["serie"] == "minerai extrait" and d["historique"][1] == {"year": 2020, "value": None}
    assert d["prevision"][0]["hi90"] is None and d["prevision"][0]["model"] == "tendance"
    assert d["qualite"] == {
        "r": 0.5,
        "rho": 0.4,
        "skill": 0.1,
        "facteur_median": 1.3,
        "couverture90": 0.9,
        "n": 40,
    }


def test_forecast_site_tries_twin(client, monkeypatch):
    monkeypatch.setattr(screener, "twins", lambda: {"greenstone": "icmm-42"})
    fake(monkeypatch, "forecast", forecast_site=lambda s: SITE if s == "icmm-42" else None)
    assert (
        client.get("/api/forecast/site", params={"site": "greenstone"}).json()["serie"] == "minerai extrait"
    )


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
        "n": 12,
    }
    assert client.get("/api/company", params={"id": "inconnue"}).status_code == 404


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
    keys = {"domaine", "x", "y", "coefficient", "type", "lo", "hi", "n", "verdict", "source"}
    assert all(set(r) == keys for r in rows)
    assert all(r["coefficient"] is not None and r["type"] in ("r", "rho", "lift") for r in rows)
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

    so = [r for r in rows if r["domaine"] == "sociétés"]
    assert [r["type"] for r in so] == ["r", "rho", "rho"]  # r NaN de la 2e ligne retiré
    assert "décalage +1 an" in so[0]["y"] and so[0]["n"] == 120

    pr = [r for r in rows if r["domaine"] == "prévisions"]
    assert {r["type"] for r in pr} == {"r", "rho"}
    t2 = next(r for r in pr if "tendance" in r["x"] and r["type"] == "r")
    assert t2["verdict"].startswith("non démontré") and t2["n"] == 600 and "2 ans" in t2["x"]


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
    assert api._verdict(1.2, 3.0, 1.0).startswith("lien démontré")
    assert "inclut 1" in api._verdict(0.8, 3.0, 1.0)
    assert api._verdict(None, 0.2) == "IC 90 % indisponible"


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
