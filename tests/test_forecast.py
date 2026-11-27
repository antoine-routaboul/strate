"""Prévisions de production (docs/previsions-societes.md § 1) : données synthétiques, aucun réseau."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from strate import forecast as fc

TODAY = date(2026, 11, 27)


def _panel(kind: str = "trend", n: int = 40, seed: int = 0, y0: int = 2000, y1: int = 2020,
           noise: float = 0.05) -> pd.DataFrame:
    """Mines synthétiques ; production ≈ 10^(6 + 0,8·log10 emprise) au départ.

    trend : croissance log-linéaire propre à chaque mine (± bruit) ; rw : marche aléatoire pure ;
    revert : écarts AR(1) autour de la taille donnée par l'emprise (rappel de 30 %/an)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fp = 10 ** rng.uniform(-0.5, 2.0)
        base = math.log(10 ** (6 + 0.8 * math.log10(fp)))
        g = rng.uniform(-0.15, 0.15)
        lv = base + (rng.normal(0, 0.8) if kind == "revert" else 0.0)
        for y in range(y0, y1 + 1):
            if kind == "trend":
                lv = base + g * (y - y0) + rng.normal(0, noise)
            elif kind == "revert":
                lv = base + 0.7 * (lv - base) + rng.normal(0, 0.2)
            else:
                lv = lv + rng.normal(0, 0.2)
            rows.append({"site_id": f"m{i:03d}", "jasansky": f"J{i:03d}", "nom": f"Mine {i}",
                         "serie": "minerai extrait", "emprise_type": "minerai", "footprint_km2": fp,
                         "year": y, "value": math.exp(lv)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trend_bt():
    return fc.run_backtest(_panel("trend"), n_boot=100, today=TODAY)


def _row(bt: dict, model: str, h: int) -> dict:
    return next(r for r in bt["rows"] if r["model"] == model and r["horizon"] == h)


# -- briques ------------------------------------------------------------------------------------
def test_damping():
    assert fc.damping(0.0, 3) == 0
    assert fc.damping(1.0, 3) == 3
    assert fc.damping(0.5, 2) == pytest.approx(0.75)


def test_cases_never_read_data_after_origin():
    p = _panel("trend", n=5)
    c1 = fc.make_cases(p)
    p2 = p.copy()
    p2.loc[p2["year"] > 2010, "value"] *= 7.0  # tout ce qui suit 2010 change
    c2 = fc.make_cases(p2)
    k = ["site_id", "origin", "h"]
    a = c1[c1["origin"] <= 2010].set_index(k).sort_index()
    b = c2[c2["origin"] <= 2010].set_index(k).sort_index()
    pd.testing.assert_series_equal(a["l0"], b["l0"])
    pd.testing.assert_series_equal(a["slope"], b["slope"])
    assert (c1["target"] == c1["origin"] + c1["h"]).all()
    assert c1["target"].between(fc.FIRST_TARGET, fc.LAST_TARGET).all()


def test_slope_needs_three_points_in_window():
    years = np.array([2000, 2004, 2008, 2009, 2010])
    lv = np.log([1.0, 2, 3, 4, 5])
    assert math.isnan(fc._origin_state(years, lv, 2009)[1])  # [2005, 2009] : 2 années seulement
    assert math.isnan(fc._origin_state(years, lv, 2007)[0])  # pas de valeur à l'origine
    l0, sl = fc._origin_state(years, lv, 2010)
    assert l0 == pytest.approx(math.log(5)) and sl > 0


def test_anchor_coefs_recover_relation_and_domain():
    p = _panel("trend", n=30, noise=0.0)
    p = p[p["year"] == 2000]
    coefs = fc.anchor_coefs(p)
    a, b, n, lo, hi = coefs[("minerai", 2000)]
    assert (a, b, n) == (pytest.approx(6.0), pytest.approx(0.8), 30)
    assert math.isnan(fc._anchor_ln("minerai", hi + 1, coefs[("minerai", 2000)]))
    assert fc._anchor_ln("minerai", 1.0, coefs[("minerai", 2000)]) == pytest.approx(math.log(10) * 6.8)
    assert math.isnan(fc._anchor_ln(None, 1.0, coefs[("minerai", 2000)]))


def test_metrics_perfect_and_naive():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.normal(10, 1, n)
    l0 = y + rng.normal(0, 0.3, n)
    base = {"site_id": [f"s{i % 20}" for i in range(n)], "target": np.arange(n), "y": y, "l0": l0,
            "p_naif": l0, "lo80": y - 1, "hi80": y + 1, "lo90": y - 2, "hi90": y + 2}
    perfect = fc.metrics(pd.DataFrame(base | {"p": y}), n_boot=100)
    assert perfect["r"] == pytest.approx(1.0) and perfect["skill"] == pytest.approx(1.0)
    assert perfect["r_lo"] == pytest.approx(1.0) and perfect["bon_sens"] == 1.0
    assert perfect["couverture80"] == 1.0 and perfect["n_mines"] == 20 and perfect["n_points"] == n
    naive = fc.metrics(pd.DataFrame(base | {"p": l0}), n_boot=100)
    assert math.isnan(naive["r"]) and math.isnan(naive["r_lo"])  # aucune variation prévue : pas de r
    assert naive["skill"] == 0 and naive["bon_sens"] == 0.5
    assert naive["r_niveau"] > 0.9  # le niveau est trivialement corrélé
    assert naive["r_niveau_lo"] > 0.8 and naive["note"] == fc.NOTE_IMMOBILE and "note" not in perfect


def test_choose_requires_ci_above_zero():
    path = {"naif": {"skill_moyen": 0.0, "skill_moyen_lo": 0.0},
            "tendance": {"skill_moyen": 0.10, "skill_moyen_lo": -0.01},
            "tendance+taille": {"skill_moyen": 0.05, "skill_moyen_lo": 0.01}}
    assert fc.choose(path) == "tendance+taille"
    assert fc.choose(path | {"tendance+taille": {"skill_moyen": 0.05, "skill_moyen_lo": None}}) == "naif"


def test_path_skill_is_mean_of_horizon_skills():
    rng = np.random.default_rng(0)
    c = pd.DataFrame({"site_id": np.repeat([f"s{i}" for i in range(20)], 2), "h": np.tile([1, 2], 20)})
    c["y"] = rng.normal(0, 1, len(c))
    y = c["y"].to_numpy()
    preds = {"naif": np.zeros(len(c)), "tendance": y * 0.5, "tendance+taille": y}
    ps = fc.path_skill(c, preds, (1, 2), n_boot=200)
    assert ps["tendance+taille"]["skill_moyen"] == pytest.approx(1.0)
    assert ps["tendance"]["skill_moyen"] == pytest.approx(0.75)  # 1 − 0,25 à chaque horizon
    assert ps["naif"]["skill_moyen"] == 0 and ps["tendance"]["n_mines"] == 20
    assert ps["tendance"]["skill_moyen_lo"] <= 0.75 <= ps["tendance"]["skill_moyen_hi"]


def test_single_model_for_whole_path(tmp_path):
    """Un seul modèle sur tout le chemin : pas de saut de la valeur prévue d'un horizon à l'autre."""
    p = _panel("revert", n=80, seed=2, y1=2019)
    bt = fc.backtest(out=tmp_path, n_boot=50, panel=p, today=TODAY)
    assert len(set(bt["retenu"].values())) == 1 and bt["retenu_modele"] in fc.MODELS
    ch = bt["skill_chemin"][bt["retenu_modele"]]
    assert ch["skill_moyen_lo"] > 0 and ch["horizons"] == bt["horizons_mesures"]
    r = fc.forecast_site("m005", out=tmp_path, panel=p, today=TODAY)
    assert {x["model"] for x in r["prevision"]} == {bt["retenu_modele"]}


def test_error_quantiles_extrapolate_with_sqrt():
    rng = np.random.default_rng(0)
    err = pd.DataFrame({"model": "naif", "h": 1, "target": 2010, "err": rng.normal(0, 1, 500)})
    q = fc.error_quantiles(err, h_max=4)
    assert not q["naif"][1]["extrapole"] and q["naif"][4]["extrapole"]
    assert q["naif"][4]["q"][3] == pytest.approx(q["naif"][1]["q"][3] * 2)
    assert q["naif"][1]["q"][0] < q["naif"][1]["q"][1] < 0 < q["naif"][1]["q"][2] < q["naif"][1]["q"][3]
    assert fc._q(q, "naif", 9) == pytest.approx([x * 3 for x in q["naif"][1]["q"]])  # au-delà : √(9/1)


# -- rétro-test ----------------------------------------------------------------------------------
def test_backtest_structure(trend_bt):
    bt = trend_bt
    assert bt["statut"] == "ok" and bt["periode"] == {"reglage": "≤2014", "test": "2015–2021"}
    assert bt["horizons_mesures"] == list(fc.HORIZONS) and bt["horizon_max_publie"] == fc.H_MAX
    assert {(r["model"], r["horizon"]) for r in bt["rows"]} == {(m, h) for m in fc.MODELS
                                                                 for h in fc.HORIZONS}
    keys = {"model", "horizon", "n_mines", "n_points", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi",
            "skill", "facteur_median", "bon_sens", "couverture80", "couverture90"}
    assert all(keys <= set(r) for r in bt["rows"])
    assert set(bt["retenu"]) == {str(h) for h in fc.HORIZONS} and isinstance(bt["methode"], str)
    assert [d["horizon"] for d in bt["diagnostic_tendance_brute"]] == list(fc.HORIZONS)
    assert bt["diagnostic_tendance_brute"][0]["skill"] > 0.3  # séries en tendance : φ = 1 aide aussi


def test_trend_beats_naive_on_trending_mines(trend_bt):
    t = _row(trend_bt, "tendance", 1)
    assert trend_bt["reglages"]["phi"] >= 0.8 and isinstance(trend_bt["reglages"]["phi"], float)
    assert t["skill_lo"] > 0.3 and t["r"] > 0.5 and t["r_lo"] > 0.3 and t["bon_sens"] > 0.7
    assert trend_bt["retenu"]["1"] in ("tendance", "tendance+taille")


def test_intervals_have_honest_coverage(trend_bt):
    for r in trend_bt["rows"]:  # couverture mesurée à chaque horizon publié
        if r["horizon"] <= 3:
            assert 0.65 <= r["couverture80"] <= 0.92
            assert 0.78 <= r["couverture90"] <= 0.98
        else:  # peu d'erreurs de réglage aux longs horizons (cibles 2012–2014 pour h = 10)
            assert 0.65 <= r["couverture80"] <= 1 and 0.75 <= r["couverture90"] <= 1
        assert r["couverture90"] >= r["couverture80"]


def test_random_walk_keeps_naive():
    bt = fc.run_backtest(_panel("rw", n=60, seed=3), n_boot=100, today=TODAY)
    assert set(bt["retenu"].values()) == {"naif"}
    assert any("aucun gain n'est affiché" in a for a in bt["avertissements"])
    n = _row(bt, "naif", 1)
    assert n["skill"] == 0 and n["r"] is None and n["note"] == fc.NOTE_IMMOBILE
    assert n["r_niveau_lo"] is not None and n["rho_niveau_hi"] is not None  # mesure de repli, avec IC


def test_pull_and_single_settings():
    assert fc.pull(0.0, 5) == 0 and fc.pull(1.0, 1) == 1
    assert fc.pull(0.5, np.array([1, 2])) == pytest.approx([0.5, 0.75])
    c = pd.DataFrame({"h": [1, 2], "l0": [0.0, 0.0], "slope": [1.0, 1.0], "anchor": [10.0, np.nan]})
    p = fc.predict(c, phi=0.5, kappa=0.1)
    assert p["naif"].tolist() == [0, 0] and p["tendance"] == pytest.approx([0.5, 0.75])
    assert p["tendance+taille"] == pytest.approx([0.5 + 0.1 * 9.5, 0.75])  # sans emprise : la tendance


def test_size_model_wins_when_mines_revert_to_footprint():
    bt = fc.run_backtest(_panel("revert", n=80, seed=2), n_boot=100, today=TODAY)
    assert 0.15 <= bt["reglages"]["kappa"] <= 0.5
    assert bt["reglages"]["rappel_par_horizon"]["2"] > bt["reglages"]["rappel_par_horizon"]["1"]
    s = _row(bt, "tendance+taille", 1)
    assert s["skill_lo"] > 0 and s["r_lo"] > 0 and s["rho"] > 0
    assert bt["retenu"]["1"] == bt["retenu_modele"] == "tendance+taille"


def test_selection_optimism_is_measured():
    bt = fc.run_backtest(_panel("revert", n=80, seed=2), n_boot=50, today=TODAY)
    ctl = bt["controle_selection"]
    assert [c["choix_sur"] for c in ctl] == ["A", "B"]
    assert all(c["n_mines_choix"] + c["n_mines_autre"] == _row(bt, "naif", 1)["n_mines"] for c in ctl)
    assert all(c["modele"] == "tendance+taille" and c["skill_moyen_autre_moitie"] > 0 for c in ctl)
    assert any("controle_selection" in a for a in bt["avertissements"])
    assert bt["choix_serie"] == {"mines_avec_annees_reglage": 0, "serie_differente": 0}  # pas de colonne


def test_series_choice_lookahead_is_counted():
    p = _panel("trend", n=20)
    p["serie_reglage"] = np.where(p["site_id"].isin(["m000", "m001"]), "or (métal)", p["serie"])
    bt = fc.run_backtest(p, n_boot=30, today=TODAY)
    assert bt["choix_serie"] == {"mines_avec_annees_reglage": 20, "serie_differente": 2}
    assert any("2 mine(s) sur 20" in a for a in bt["avertissements"])


def test_tuning_never_sees_test_period():
    p = _panel("trend", n=30, seed=5)
    a = fc.run_backtest(p, n_boot=50, today=TODAY)
    p2 = p.copy()
    p2.loc[p2["year"] > fc.TUNE_END, "value"] *= np.exp(np.random.default_rng(9).normal(0, 1, (p2["year"] >
                                                                                            fc.TUNE_END).sum()))
    b = fc.run_backtest(p2, n_boot=50, today=TODAY)
    assert a["reglages"] == b["reglages"]
    assert a["intervalles"]["test"] == b["intervalles"]["test"]
    assert a["rows_reglage"] == b["rows_reglage"]
    assert a["rows"] != b["rows"]


def test_backtest_absent_and_written(tmp_path):
    empty = fc.run_backtest(pd.DataFrame(), today=TODAY)
    assert empty["statut"] == "absent" and empty["retenu"]["1"] == "naif"
    assert fc.load_backtest(tmp_path) is None
    res = fc.backtest(out=tmp_path, n_boot=30, panel=_panel("trend", n=15), today=TODAY)
    assert (tmp_path / "backtest.json").exists()
    assert fc.load_backtest(tmp_path)["rows"] == res["rows"]


# -- prévisions d'avenir et registre ------------------------------------------------------------
def test_forecast_site(tmp_path):
    p = _panel("trend", n=20, y1=2019)
    fc.backtest(out=tmp_path, n_boot=30, panel=p, today=TODAY)
    assert fc.forecast_site("inconnu", out=tmp_path, panel=p, today=TODAY) is None
    r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
    assert r["serie"] == "minerai extrait" and len(r["historique"]) == 20
    years = [x["year"] for x in r["prevision"]]
    assert years == list(range(2020, TODAY.year + 2))
    w = [x["hi90"] / x["lo90"] for x in r["prevision"]]
    assert all(b >= a - 1e-9 for a, b in zip(w, w[1:], strict=False))  # s'élargit avec l'horizon
    assert all(x["lo90"] <= x["lo80"] <= x["value"] <= x["hi80"] <= x["hi90"] for x in r["prevision"])
    keys = {"model", "horizon", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi", "skill", "skill_lo",
            "skill_hi", "couverture80", "couverture90", "facteur_median", "n", "n_mines", "r_niveau",
            "r_niveau_lo", "rho_niveau_hi", "note"}
    assert keys <= set(r["qualite"]) and r["qualite"]["portee"] == "global"
    st = r["suivi_site"]  # période de test seulement : cibles 2015–2019 à h = 1
    assert st["n"] == 5 and st["periode"] == "2015–2021" and st["facteur_max"] >= st["facteur_median"] >= 1
    assert fc.forecast_site("J001", out=tmp_path, panel=p, today=TODAY)["site_id"] == "m001"


def test_future_anchor_uses_same_family_as_tuning():
    """Avenir : modèle d'emprise de la coupe de l'année d'origine, comme dans le rétro-test (pas un autre)."""
    p = _panel("revert", n=40, seed=4, y1=2019)
    bt = fc.run_backtest(p, n_boot=30, today=TODAY)
    assert "avenir" not in bt["emprise"]
    coef = next(tuple(x[2:]) for x in bt["emprise"]["par_origine"] if x[:2] == ["minerai", 2019])
    g = p[p["site_id"] == "m003"]
    lfp = math.log10(g["footprint_km2"].iat[0])
    assert fc._future_anchor("minerai", lfp, bt, 2019) == pytest.approx(fc._anchor_ln("minerai", lfp, coef))
    assert math.isnan(fc._future_anchor("minerai", lfp, bt, 1990))  # pas d'ajustement : pas de rappel
    f = fc.site_forecast_frame(g, bt, TODAY)
    s, t = f[f["model"] == "tendance+taille"], f[f["model"] == "tendance"]
    anc = fc._anchor_ln("minerai", lfp, coef)
    lam = fc.pull(bt["reglages"]["kappa"], s["h"].to_numpy(float))
    lt = np.log(t["value"].to_numpy(float))
    assert np.log(s["value"].to_numpy(float)) == pytest.approx(lt + lam * (anc - lt))


def test_forecast_site_is_read_only(tmp_path):
    p = _panel("trend", n=12, y1=2019)
    r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
    assert not (tmp_path / "backtest.json").exists() and list(tmp_path.iterdir()) == []
    assert r["prevision"] == [] and r["message"] and len(r["historique"]) == 20
    assert r["serie"] == "minerai extrait" and r["qualite"] == {"n": 0}


def test_published_interval_is_the_measured_one_even_if_biased(tmp_path):
    """Quantiles d'erreur biaisés (réel toujours au-dessus du prévu) : les bornes publiées sont celles dont
    le rétro-test mesure la couverture (prévision + quantiles), sans être ramenées à la valeur prévue."""
    p = _panel("trend", n=20, y1=2019)
    bt = fc.run_backtest(p, n_boot=30, today=TODAY)
    bt["retenu_modele"] = "naif"
    for x in bt["intervalles"]["avenir"]["naif"].values():
        x["q"] = [0.05, 0.1, 0.2, 0.3]
    (tmp_path / "backtest.json").write_text(json.dumps(bt), "utf-8")
    r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
    assert r["prevision"]
    for x in r["prevision"]:
        off = [math.log(x[k] / x["value"]) for k in ("lo90", "lo80", "hi80", "hi90")]
        assert off == pytest.approx([0.05, 0.1, 0.2, 0.3], abs=1e-4)


def test_forecast_site_blames_backtest_not_series(tmp_path):
    """backtest.json présent mais sans résultat : le message nomme le rétro-test, pas la série de la mine."""
    p = _panel("trend", n=12, y1=2019)
    for statut in ("insuffisant", "absent"):
        bt = {"statut": statut, "raison": "trop peu d'années pour un rétro-test", "rows": [],
              "run_at": "2026-11-27T00:00:00Z"}
        (tmp_path / "backtest.json").write_text(json.dumps(bt), "utf-8")
        r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
        assert r["prevision"] == [] and r["qualite"] == {"n": 0} and r["suivi_site"] == {"n": 0}
        assert statut in r["message"] and "trop peu d'années" in r["message"]
        assert "série insuffisante" not in r["message"] and r["note"] == r["message"]
        assert r["backtest_run_at"] == bt["run_at"] and len(r["historique"]) == 20


def test_measured_needs_rows_and_unextrapolated_quantiles():
    rows = [{"model": m, "horizon": h, "n_points": 50} for m in fc.MODELS for h in (1, 2, 3)]
    rows += [{"model": m, "horizon": 4, "n_points": 3} for m in fc.MODELS]  # trop peu de cas de test
    q = {m: {h: {"q": [-1, -0.5, 0.5, 1], "n": 40, "extrapole": h == 3} for h in (1, 2, 3, 4)}
         for m in fc.MODELS}
    assert fc._measured(rows, q) == [1, 2]


def test_future_capped_at_longest_measured_horizon(tmp_path):
    p = _panel("trend", n=20, y1=2019)
    bt = fc.backtest(out=tmp_path, n_boot=30, panel=p, today=TODAY)
    r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
    assert [x["horizon"] for x in r["prevision"]] == list(range(1, TODAY.year + 2 - 2019))
    assert all(x["mesure"] is True for x in r["prevision"])
    assert max(q["horizon"] for q in r["qualite_par_horizon"]) == bt["horizon_max_publie"]
    bt["horizon_max_publie"], bt["horizons_mesures"] = 7, list(range(1, 8))  # 2026 = h 7 : dernier publié
    (tmp_path / "backtest.json").write_text(json.dumps(bt), "utf-8")
    r = fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)
    assert [x["year"] for x in r["prevision"]] == list(range(2020, 2027))
    bt["horizon_max_publie"], bt["horizons_mesures"] = 6, list(range(1, 7))  # année en cours non mesurée
    (tmp_path / "backtest.json").write_text(json.dumps(bt), "utf-8")
    assert fc.forecast_site("m001", out=tmp_path, panel=p, today=TODAY)["prevision"] == []


def test_stale_series_gets_no_forecast(tmp_path):
    p = _panel("trend", n=20, y1=2019)
    old = _panel("trend", n=1, y0=2000, y1=2008).assign(site_id="vieux", jasansky="JV")
    p = pd.concat([p, old], ignore_index=True)
    fc.backtest(out=tmp_path, n_boot=30, panel=p, today=TODAY)
    r = fc.forecast_site("vieux", out=tmp_path, panel=p, today=TODAY)
    assert r["prevision"] == [] and r["note"]


def test_ledger_is_append_only(tmp_path):
    p = _panel("trend", n=10, y1=2019)
    fc.backtest(out=tmp_path, n_boot=30, panel=p, today=TODAY)
    t1 = datetime(2026, 11, 27, 8, tzinfo=UTC)
    n1 = fc.write_ledger(out=tmp_path, panel=p, today=TODAY, now=t1)
    assert n1 == 10 * 3 * (TODAY.year + 1 - 2019)
    first = fc.read_ledger(tmp_path)
    assert list(first.columns) == fc.LEDGER_COLS and (first["made_at"] == "2026-11-27T08:00:00Z").all()
    assert fc.write_ledger(out=tmp_path, panel=p, today=TODAY, now=datetime(2026, 11, 28, tzinfo=UTC)) == 0
    p2 = p.copy()
    p2.loc[(p2["site_id"] == "m000") & (p2["year"] == 2019), "value"] *= 2  # donnée révisée
    n2 = fc.write_ledger(out=tmp_path, panel=p2, today=TODAY, now=datetime(2026, 11, 29, tzinfo=UTC))
    assert n2 == 3 * (TODAY.year + 1 - 2019)
    after = fc.read_ledger(tmp_path)
    assert len(after) == n1 + n2
    pd.testing.assert_frame_equal(after.iloc[:n1].reset_index(drop=True), first)  # rien de réécrit
    bt = fc.load_backtest(tmp_path)  # intervalles élargis (nouveau rétro-test) : nouvelles lignes datées
    for qs in bt["intervalles"]["avenir"].values():
        for v in qs.values():
            v["q"] = [x * 1.5 for x in v["q"]]
    (tmp_path / "backtest.json").write_text(json.dumps(bt), "utf-8")
    n3 = fc.write_ledger(out=tmp_path, panel=p2, today=TODAY, now=datetime(2026, 11, 30, tzinfo=UTC))
    assert n3 == n1 and len(fc.read_ledger(tmp_path)) == n1 + n2 + n3


def test_ledger_dedup_compares_with_true_last_row(tmp_path, monkeypatch):
    """Dernière ligne aux bornes NaN : la comparaison se fait avec elle, pas avec un mélange de lignes."""
    key = {"site_id": "m000", "serie": "minerai extrait", "target_year": 2027, "model": "naif"}
    old = pd.DataFrame([key | {"made_at": "2026-01-01T00:00:00Z", "value": 10.0, "lo80": 5.0, "hi80": 20.0,
                               "lo90": 4.0, "hi90": 25.0},
                        key | {"made_at": "2026-02-01T00:00:00Z", "value": 10.0, "lo80": np.nan,
                               "hi80": np.nan, "lo90": np.nan, "hi90": np.nan}])[fc.LEDGER_COLS]
    tmp_path.mkdir(exist_ok=True)
    old.to_parquet(tmp_path / fc.LEDGER_NAME, index=False)
    (tmp_path / fc.BACKTEST_NAME).write_text("{}", "utf-8")
    same_as_first = pd.DataFrame([key | {"h": 1, "value": 10.0, "lo80": 5.0, "hi80": 20.0, "lo90": 4.0,
                                         "hi90": 25.0}])
    monkeypatch.setattr(fc, "all_forecasts", lambda *a, **k: same_as_first)
    # identique à la 1re ligne mais pas à la dernière (bornes NaN) : c'est une nouvelle prévision
    assert fc.write_ledger(out=tmp_path, panel=pd.DataFrame(), now=datetime(2026, 3, 1, tzinfo=UTC)) == 1
    assert fc.write_ledger(out=tmp_path, panel=pd.DataFrame(), now=datetime(2026, 4, 1, tzinfo=UTC)) == 0


# -- panel réel (structure) -------------------------------------------------------------------------
def test_production_panel_main_series_dedup_and_emprise():
    cat = pd.DataFrame({
        "id": ["a", "b", "a_bis", "d"], "jasansky": ["JA", "JB", "JA", "JD"],
        "name": ["A", "B", "A bis", "D"], "commodities": ["copper", "gold, silver", "copper", "coal"],
        "footprint_km2": [5.0, 2.0, 5.0, 10.0]})
    rows = []
    for y in range(2010, 2015):
        rows += [("JA", y, "minerai extrait", 1e6), ("JA", y, "cuivre (métal)", 1e4),
                 ("JB", y, "argent (métal)", 50.0), ("JB", y, "or (métal)", 5.0),
                 ("JD", y, "charbon", 2e6)]
    prod = pd.DataFrame(rows, columns=["facility_id", "year", "serie", "value_tonnes"])
    p = fc.production_panel(prod, cat, underground={"JD"})
    s = p.drop_duplicates("site_id").set_index("site_id")
    assert set(s.index) == {"a", "b", "d"}  # « a_bis » partage l'id Jasansky de « a »
    assert s.loc["a", "serie"] == "minerai extrait" and s.loc["a", "emprise_type"] == "minerai"
    assert s.loc["b", "serie"] == "or (métal)" and pd.isna(s.loc["b", "emprise_type"])
    assert s.loc["d", "serie"] == "charbon" and pd.isna(s.loc["d", "emprise_type"])  # souterraine
    assert (p.groupby("site_id").size() == 5).all()
    assert s.loc["a", "serie_reglage"] == "minerai extrait"
    assert fc.production_panel(prod.iloc[:0], cat, underground=set()).empty
