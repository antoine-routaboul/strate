"""Calibration : sélection recalée sur le catalogue actuel, radar reproductible depuis le dépôt."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strate import calibrate as K


def _cat() -> pd.DataFrame:
    rows = [
        ("A", "Mine A", "JA", 10.0), ("B", "Mine B", "JB", 5.0), ("C", "Mine C", "JC", 3.0),
        ("D", "Mine D (doublon)", "JA", 2.0), ("E", "Mine E", "JE", 4.0), ("F", "Mine F", "JF", None),
        ("G", "Mine G", None, 1.0), ("H", "Mine H", "JX", 7.0),
    ]
    df = pd.DataFrame(rows, columns=["id", "name", "jasansky", "footprint_km2"])
    return df.assign(country="Chile", commodities="copper").set_index("id", drop=False)


def _prod() -> pd.DataFrame:
    rows = []
    for jid, years in (("JA", range(2015, 2020)), ("JB", range(2015, 2020)), ("JE", (2016, 2017)),
                       ("JF", range(2015, 2020)), ("JX", range(2015, 2020))):
        rows += [{"jasansky": jid, "year": y, "serie": "minerai extrait", "value_tonnes": 1e6 * (1 + y % 3)}
                 for y in years]
    return pd.DataFrame(rows)


def _equip() -> pd.Series:
    return pd.Series({"JA": "open pit", "JB": "underground", "JE": "open pit", "JF": "open pit",
                      "JX": "open pit, underground"}, dtype="string")


def _elig() -> pd.DataFrame:
    return K._eligibility_table(_cat(), _prod(), _equip())


def test_eligibility_reasons():
    e = _elig()["exclusion"]
    assert pd.isna(e["A"]) and pd.isna(e["H"])
    assert e["B"] == "mine purement souterraine"
    assert e["C"].startswith("aucune production réelle")
    assert e["D"] == "production JA déjà portée par A"
    assert e["E"].startswith("moins de 3 ans")
    assert e["F"] == "emprise inconnue"
    assert e["G"].startswith("aucun id Jasansky")
    assert int(e.isna().sum()) == 2


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "OUT", tmp_path)
    # ancien tirage : A lisait la production JX (désormais sur H), B était lié à JB, Z a quitté le catalogue
    pd.DataFrame([
        {"id": "A", "jasansky": "JX", "nom": "Mine A", "pays": "Chile", "serie": "minerai extrait",
         "annees": 5, "equipement": "open pit", "emprise_km2": 0.2, "groupe": "réglage"},
        {"id": "B", "jasansky": "JX", "nom": "Mine B", "pays": "Chile", "serie": "minerai extrait",
         "annees": 5, "equipement": "open pit", "emprise_km2": 5.0, "groupe": "test"},
        {"id": "Z", "jasansky": "JZ", "nom": "Mine Z", "pays": "Peru", "serie": "minerai extrait",
         "annees": 4, "equipement": "open pit", "emprise_km2": 1.0, "groupe": "test"},
    ]).to_csv(tmp_path / "selection.csv", index=False)
    return tmp_path


def test_refresh_selection_rederives_and_flags(out):
    sel = K.refresh_selection(_elig()).set_index("id")
    # A : production et emprise relues dans le catalogue actuel, provenance du tirage conservée
    assert sel.at["A", "jasansky"] == "JA" and sel.at["A", "jasansky_tirage"] == "JX"
    assert sel.at["A", "emprise_km2"] == 10.0 and sel.at["A", "emprise_tirage_km2"] == 0.2
    assert pd.isna(sel.at["A", "exclusion"])
    # B : souterraine ; l'ancienne production pointe désormais vers H
    assert sel.at["B", "exclusion"].startswith("mine purement souterraine")
    assert "JX désormais rattaché à H" in sel.at["B", "exclusion"]
    assert sel.at["Z", "exclusion"] == "absente du catalogue actuel" and sel.at["Z", "nom"] == "Mine Z"
    # groupes réglage/test inchangés
    assert sel["groupe"].to_dict() == {"A": "réglage", "B": "test", "Z": "test"}
    info = json.loads((out / "selection_info.json").read_text())
    assert (info["eligibles"], info["tirees"], info["retenues"]) == (2, 3, 1)
    assert set(info["exclues"]) == {"B", "Z"} and info["production_reliee"] == {"A": ["JX", "JA"]}
    assert info["emprise_modifiee_km2"] == {"A": [0.2, 10.0]}
    # relancer ne perd pas la provenance du tirage
    again = K.refresh_selection(_elig()).set_index("id")
    assert again.at["A", "jasansky_tirage"] == "JX" and again.at["A", "emprise_tirage_km2"] == 0.2


def test_panels_use_refreshed_selection(out, monkeypatch):
    sel = K.refresh_selection(_elig())
    monkeypatch.setattr(K, "_all_production", _prod)
    monkeypatch.setattr(K, "_window_km2", lambda sid: 100.0)
    years = range(2015, 2020)
    sat = pd.DataFrame({"bare_km2": [1.0, 1.1, 1.2, 1.3, 1.4], "water_km2": 0.1, "change_km2": 0.2,
                        "new_bare_km2": 0.1, "months": 6}, index=pd.Index(years, name="year"))
    monkeypatch.setattr(K, "annual", lambda sid, grid: {0: sat})
    df = K.panel(grid=[K.GRID[0]], only_done=False, sel=sel)
    assert set(df["id"]) == {"A"}
    # production de JA (catalogue actuel), pas celle de JX du tirage
    ja = _prod().query("jasansky == 'JA'").set_index("year")["value_tonnes"]
    assert df.set_index("year")["P"].to_dict() == {y: ja[y] for y in range(2016, 2020)}

    class FakeRadar:
        def load_pairs(self, sid):
            dates = pd.date_range("2016-01-01", "2019-12-31", freq="12D")
            return pd.DataFrame({"date": dates, **{m: 0.1 for m in K.RADAR_METRICS}})

    monkeypatch.setitem(sys.modules, "strate.modules.radar", SimpleNamespace(Radar=FakeRadar))
    rp = K.radar_panel(sel=sel)
    assert set(rp["id"]) == {"A"} and (rp["emprise"] == 10.0).all()


def _radar_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(40):
        groupe = "réglage" if i % 2 else "test"
        years = [2016] if i == 0 else range(2016, 2020)  # mine 0 : une seule année, aucune variation
        for y in years:
            lp = 14 + rng.normal()
            rows.append({"id": f"m{i}", "groupe": groupe, "serie": "minerai extrait", "year": y,
                         "P": np.exp(lp), "lP": lp, "emprise": 1 + i, "paires": 20,
                         **{m: abs(rng.normal(0.2, 0.1)) for m in K.RADAR_METRICS}})
    return K._year_diffs(pd.DataFrame(rows), ["lP", *K.RADAR_METRICS])


def test_radar_report_writes_test_group(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "OUT", tmp_path)
    rep = K.radar_report(_radar_df(), n_boot=30)
    t = rep["groupe_test"]
    assert t["metrique"] == rep["choix_sur_reglage"]["metrique"] and t["n_boot"] == 30 == rep["n_boot"]
    # un seul jeu de mines : la mine de test sans variation exploitable n'est comptée nulle part
    assert t["mines"] == 19
    for k in ("variation", "bon_sens", "bon_sens_grands_mouvements", "intra_mine_niveau"):
        assert len(t[k]) == 3
    saved = json.loads((tmp_path / "radar_report.json").read_text())
    assert saved["groupe_test"]["mines"] == 19


def _hold(lo: float, hi: float) -> dict:
    return {"gain_erreur_mediane": {"facteur": (lo + hi) / 2, "ic90": [lo, hi]}, "mines_validation": 50}


def _test_block(**over) -> dict:
    t = {"metrique": "activity_db", "mines": 80, "n_boot": 1000, "variation": [0.11, -0.03, 0.26],
         "bon_sens": [0.55, 0.47, 0.64], "bon_sens_grands_mouvements": [0.6, 0.49, 0.7],
         "intra_mine_niveau": [0.14, -0.002, 0.27]}
    return t | over


def test_save_radar_summary_derives_verdict_and_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "OUT", tmp_path)
    summary = tmp_path / "calibration.json"
    summary.write_text(json.dumps({"modele_emprise": {"minerai": {"n": 3}}, "mines": 142}))
    monkeypatch.setattr(K, "_summary_path", lambda: summary)
    (tmp_path / "radar_holdout.json").write_text(json.dumps(_hold(0.9, 1.1)))
    (tmp_path / "radar_report.json").write_text(json.dumps({"groupe_test": _test_block()}))
    r = K.save_radar_summary()
    assert r["verdict"] == "non démontré : aucun intervalle à 90 % n'exclut le hasard"
    assert r["taille"]["verdict"].startswith("rejeté : n'améliore pas") and r["mines_test"] == 80
    saved = json.loads(summary.read_text())
    assert saved["modele_emprise"] == {"minerai": {"n": 3}} and saved["mines"] == 142
    assert saved["radar"]["verdict"] == r["verdict"]
    # un intervalle qui exclut le hasard rend le verdict « expérimental », jamais codé en dur
    r2 = K.save_radar_summary({"groupe_test": _test_block(variation=[0.2, 0.05, 0.3])})
    assert r2["verdict"].startswith("expérimental") and "variation" in r2["verdict"]
    assert K.radar_verdict(_test_block(bon_sens=[0.3, 0.2, 0.4])).startswith("non démontré : bon_sens")
    assert K.radar_verdict(_test_block(variation=[0.1, float("nan"), float("nan")])).startswith(
        "non démontré : aucun")
    assert K.radar_size_verdict([1.05, 1.3]).startswith("utile")


def test_save_summary_keeps_other_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "OUT", tmp_path)
    summary = tmp_path / "calibration.json"
    summary.write_text(json.dumps({"modele_emprise": {"x": 1}, "radar": {"verdict": "v"}}))
    monkeypatch.setattr(K, "_summary_path", lambda: summary)
    monkeypatch.setattr(K, "fit_size_model", lambda: {})
    t1 = {"rho": 0.5, "ic90_bas": 0.4, "ic90_haut": 0.6, "mines": 100}
    boot = [0.1, 0.0, 0.2]
    rep = {"mines": 100, "paires_annees": 300, "fenetre": [2016, 2019], "T1_coupe": t1,
           "T2_defaut": {"d_bare": {"rho": boot, "bon_sens_si_prod_bouge_10pct": boot}},
           "T3_reglage": {"rho_reglage": 0.2, "rho_test": boot, "seuils_choisis": {}},
           "alertes_vs_production": {"emprise_bouge_10pct": {"precision": boot},
                                     "taux_de_base_prod_bouge_20pct": 0.3},
           "echantillon": {"eligibles": 428, "tirees": 150, "retenues": 135, "exclues": 15}}
    (tmp_path / "report.json").write_text(json.dumps(rep))
    K.save_summary()
    saved = json.loads(summary.read_text())
    assert saved["modele_emprise"] == {"x": 1} and saved["radar"] == {"verdict": "v"}
    assert saved["mines"] == 100 and saved["echantillon"]["eligibles"] == 428


def test_report_all_runs_every_step_in_order(monkeypatch):
    calls = []
    sel = pd.DataFrame({"id": ["A"], "exclusion": [None]})
    df = pd.DataFrame({"id": ["A"]})

    def rec(name, ret=None):
        def f(*a, **k):
            calls.append(name)
            return ret
        return f

    monkeypatch.setattr(K, "refresh_selection", rec("refresh", sel))
    monkeypatch.setattr(K, "panel", rec("panel", df))
    monkeypatch.setattr(K, "report", rec("report", {}))
    monkeypatch.setattr(K, "save_summary", rec("save_summary"))
    monkeypatch.setattr(K, "save_footprint_model", rec("save_footprint_model"))
    monkeypatch.setattr(K, "radar_panel", rec("radar_panel", df))
    monkeypatch.setattr(K, "radar_report", rec("radar_report", {}))
    monkeypatch.setattr(K, "radar_holdout", rec("radar_holdout", {}))
    monkeypatch.setattr(K, "save_radar_summary", rec("save_radar_summary", {"verdict": "v"}))
    K.report_all(say=lambda *_: None)
    assert calls == ["refresh", "panel", "report", "save_summary", "save_footprint_model", "radar_panel",
                     "radar_report", "radar_holdout", "save_radar_summary"]

    from strate import cli

    seen = []
    monkeypatch.setattr(K, "report_all", lambda: seen.append(1) or {"optique": {}})
    cli.calibrate_cmd(SimpleNamespace(step="report", n=150, parallel=1))
    assert seen == [1]
