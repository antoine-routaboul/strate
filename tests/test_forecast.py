"""Prévisions de production (docs/previsions-societes.md § 1) : données synthétiques, aucun réseau."""

from __future__ import annotations

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

    trend : croissance log-linéaire propre à chaque mine (± bruit) ; rw : marche aléatoire pure."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fp = 10 ** rng.uniform(-0.5, 2.0)
        base = math.log(10 ** (6 + 0.8 * math.log10(fp)))
        g = rng.uniform(-0.15, 0.15)
        lv = base
        for y in range(y0, y1 + 1):
            if kind == "trend":
                lv = base + g * (y - y0) + rng.normal(0, noise)
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


def test_choose_requires_ci_above_zero():
    rows = [{"model": "naif", "horizon": 1, "skill": 0.0, "skill_lo": 0.0},
            {"model": "tendance", "horizon": 1, "skill": 0.10, "skill_lo": -0.01},
            {"model": "tendance+taille", "horizon": 1, "skill": 0.05, "skill_lo": 0.01},
            {"model": "tendance", "horizon": 2, "skill": 0.10, "skill_lo": None}]
    assert fc.choose(rows, horizons=(1, 2)) == {"1": "tendance+taille", "2": "naif"}


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
    assert {(r["model"], r["horizon"]) for r in bt["rows"]} == {(m, h) for m in fc.MODELS for h in (1, 2, 3)}
    keys = {"model", "horizon", "n_mines", "n_points", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi",
            "skill", "facteur_median", "bon_sens", "couverture80", "couverture90"}
    assert all(keys <= set(r) for r in bt["rows"])
    assert set(bt["retenu"]) == {"1", "2", "3"} and isinstance(bt["methode"], str)


def test_trend_beats_naive_on_trending_mines(trend_bt):
    t = _row(trend_bt, "tendance", 1)
    assert trend_bt["reglages"]["phi"]["1"] >= 0.8
    assert t["skill_lo"] > 0.3 and t["r"] > 0.5 and t["r_lo"] > 0.3 and t["bon_sens"] > 0.7
    assert trend_bt["retenu"]["1"] in ("tendance", "tendance+taille")


def test_intervals_have_honest_coverage(trend_bt):
    for r in trend_bt["rows"]:
        assert 0.65 <= r["couverture80"] <= 0.92
        assert 0.78 <= r["couverture90"] <= 0.98
        assert r["couverture90"] >= r["couverture80"]


def test_random_walk_keeps_naive():
    bt = fc.run_backtest(_panel("rw", n=60, seed=3), n_boot=100, today=TODAY)
    assert bt["retenu"] == {"1": "naif", "2": "naif", "3": "naif"}
    assert _row(bt, "naif", 1)["skill"] == 0


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
    assert {"r", "r_lo", "r_hi", "rho", "skill", "facteur_median", "couverture90", "n"} <= set(r["qualite"])
    assert r["suivi_site"]["n"] > 0
    assert fc.forecast_site("J001", out=tmp_path, panel=p, today=TODAY)["site_id"] == "m001"


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
    assert fc.production_panel(prod.iloc[:0], cat, underground=set()).empty
