"""Correctifs satellite et calibration : trames radar d'un même passage, caches optiques, panel annuel."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strate import calibrate
from strate.modules import optical, radar
from strate.modules.optical import Optical


# -- radar : un passage = une acquisition ---------------------------------------------------------------
def _item(t: datetime, href: str = "x"):
    return SimpleNamespace(datetime=t, id=f"S1_{t:%Y%m%dT%H%M%S}_{href}",
                           assets={"vv": SimpleNamespace(href=href)}, properties={})


def test_acquisitions_group_frames_of_same_pass():
    t0 = datetime(2018, 3, 1, 6, 0, 0)
    items = [_item(t0 + timedelta(days=12)), _item(t0 + timedelta(seconds=25)), _item(t0)]
    groups = radar.acquisitions(items)
    assert [len(g) for g in groups] == [2, 1]
    assert groups[0][0].datetime == t0  # trié dans le temps


def test_consecutive_requires_distinct_close_passes():
    t = datetime(2018, 3, 1, 6)
    assert not radar.consecutive(t, t + timedelta(seconds=20))  # même passage : pas une paire
    assert radar.consecutive(t, t + timedelta(days=12))
    assert radar.consecutive(t, t + timedelta(days=6))
    assert not radar.consecutive(t, t + timedelta(days=radar.MAX_GAP_DAYS + 1))


def test_mosaic_fills_from_every_readable_frame():
    a = np.array([[1.0, np.nan], [np.nan, np.nan]], "float32")
    b = np.array([[3.0, 2.0], [np.nan, np.nan]], "float32")
    m = radar.mosaic([a, None, b])
    assert m[0, 0] == 2.0 and m[0, 1] == 2.0 and np.isnan(m[1, 1])
    assert radar.mosaic([None, None]) is None


def _pair(d, dp, orbit=1, v=0.1):
    return {"date": date.fromisoformat(d), "date_prev": date.fromisoformat(dp), "orbit": orbit,
            "activity_db": v}


def test_clean_pairs_drops_zero_day_rows_and_keeps_shortest_gap():
    df = pd.DataFrame([
        _pair("2018-01-13", "2018-01-01"),
        _pair("2018-01-13", "2018-01-13", v=9.0),  # « paire » d'une trame avec une autre du même passage
        _pair("2018-01-25", "2018-01-01", v=5.0),  # calcul où le 13 était illisible
        _pair("2018-01-25", "2018-01-13", v=0.2),
    ])
    out = radar.clean_pairs(df)
    assert list(zip(out["date"].astype(str), out["date_prev"].astype(str), strict=True)) == [
        ("2018-01-13", "2018-01-01"), ("2018-01-25", "2018-01-13")]
    assert 9.0 not in out["activity_db"].tolist()


def test_merge_pairs_key_includes_previous_date():
    old = pd.DataFrame([_pair("2018-01-13", "2018-01-01", v=0.1), _pair("2018-01-13", "2018-01-13", v=9.0)])
    new = pd.DataFrame([_pair("2018-01-13", "2018-01-01", v=0.3)])
    out = radar.merge_pairs(old, new)
    assert len(out) == 1 and out["activity_db"].iloc[0] == 0.3  # recalcul gagne, ligne corrompue retirée


def test_radar_pairs_never_compare_a_pass_with_itself(monkeypatch, tmp_path):
    from strate import geo

    site = SimpleNamespace(id="x", lon=121.5, lat=-30.8, radius_m=600)
    monkeypatch.setattr(radar.entities, "load", lambda: SimpleNamespace(site=lambda sid: site))
    grid = geo.site_grid(site.lon, site.lat, site.radius_m + radar.RING_M, radar.RES)
    n = grid.n
    yy = np.arange(n)[:, None] * np.ones(n)
    mine = yy < n // 2
    monkeypatch.setattr(radar, "masks", lambda s, g: (mine, ~mine, "test"))
    t0 = datetime(2018, 3, 1, 6)
    # passage 1 : deux trames qui couvrent chacune une moitié du carré ; passage 2, 12 jours après
    items = [_item(t0, "haut"), _item(t0 + timedelta(seconds=20), "bas"),
             _item(t0 + timedelta(days=12), "p2")]
    monkeypatch.setattr(radar.Radar, "_items", lambda self, s, a, b: (items, 7))
    rng = np.random.default_rng(0)

    def read(href, g, resampling):
        a = rng.uniform(0.05, 0.2, (n, n)).astype("float32")
        if href == "haut":
            a[n // 2:] = 0
        elif href == "bas":
            a[:n // 2] = 0
        return a[None]

    monkeypatch.setattr(radar.geo, "read", read)
    monkeypatch.setattr(radar.Radar, "pairs_path", lambda self, sid: tmp_path / f"{sid}.parquet")
    df = radar.Radar(workers=1).pairs("x", date(2018, 3, 1), date(2018, 3, 31))
    assert len(df) == 1
    assert df["date_prev"].iloc[0] == date(2018, 3, 1) and df["date"].iloc[0] == date(2018, 3, 13)
    assert (pd.read_parquet(tmp_path / "x.parquet")["date"] != pd.read_parquet(tmp_path / "x.parquet")[
        "date_prev"]).all()


def test_load_pairs_ignores_corrupt_cached_rows(monkeypatch, tmp_path):
    rows = [_pair(f"2018-{m:02d}-13", f"2018-{m:02d}-01") for m in range(1, 11)]
    rows += [_pair(f"2018-{m:02d}-01", f"2018-{m:02d}-01", v=50.0) for m in range(1, 11)]
    df = pd.DataFrame(rows)
    for c in calibrate.RADAR_METRICS:
        df[c] = df["activity_db"]
    df.to_parquet(tmp_path / "m1.parquet", index=False)
    monkeypatch.setattr(radar.Radar, "pairs_path", lambda self, sid: tmp_path / f"{sid}.parquet")
    loaded = radar.Radar().load_pairs("m1")
    assert len(loaded) == 10 and loaded["activity_db"].max() == pytest.approx(0.1)


# -- calibration : variations entre années vraiment consécutives ------------------------------------
def test_year_diffs_only_between_consecutive_years():
    df = pd.DataFrame({"id": ["a"] * 3 + ["b"] * 2, "year": [2016, 2018, 2019, 2016, 2017],
                       "lP": [1.0, 2.0, 2.5, 3.0, 3.5], "x": [0.0, 1.0, 3.0, 1.0, 1.0]})
    out = calibrate._year_diffs(df, ["lP", "x"]).set_index(["id", "year"])
    assert np.isnan(out.loc[("a", 2018), "dP"])  # 2016 → 2018 n'est pas une variation annuelle
    assert out.loc[("a", 2019), "dP"] == pytest.approx(0.5)
    assert out.loc[("a", 2019), "d_x"] == pytest.approx(2.0)
    assert out.loc[("b", 2017), "dP"] == pytest.approx(0.5)


# -- optique : ne figer dans le cache que des composites définitifs ---------------------------------
def test_persist_rules():
    m = date(2026, 10, 1)
    assert not optical.persist(m, 0, 3, 3, today=date(2026, 10, 20))  # mois en cours
    assert not optical.persist(m, 0, 3, 3, today=date(2026, 11, 1))  # catalogue pas encore complet
    assert optical.persist(m, 0, 0, 3, today=date(2026, 11, 20))  # mois clos, nuageux : définitif
    assert not optical.persist(m, 2, 0, 3, today=date(2026, 11, 20))  # lectures en échec
    assert optical.persist(m, 1, 3, 3, today=date(2026, 11, 20))  # un échec mais assez de scènes


@pytest.fixture
def opt_env(monkeypatch, tmp_path):
    monkeypatch.setattr(optical, "DATA_DIR", tmp_path)
    site = SimpleNamespace(id="s", lon=121.5, lat=-30.8, radius_m=200)
    return site, tmp_path


def _scene_ok(self, item, grid):
    n = grid.n
    one = np.full((n, n), 0.2, "float32")
    return np.ones((n, n), bool), np.zeros((n, n), bool), one, one * 2, one, item.id


def test_composite_not_cached_when_all_reads_fail(monkeypatch, opt_env):
    site, root = opt_env

    def boom(self, item, grid):
        raise OSError("réseau")

    monkeypatch.setattr(Optical, "_scene", boom)
    o = Optical(workers=1)
    comp = o.composite(site, date(2020, 5, 1), [SimpleNamespace(id="a"), SimpleNamespace(id="b")])
    assert comp["valid"].sum() == 0
    assert not o._cache("s", date(2020, 5, 1)).exists()


def test_composite_of_open_month_kept_in_memory_only(monkeypatch, opt_env):
    site, _ = opt_env
    monkeypatch.setattr(Optical, "_scene", _scene_ok)
    o = Optical(workers=1)
    m = date(2026, 11, 1)
    comp = o.composite(site, m, [SimpleNamespace(id="a")], today=date(2026, 11, 23))
    assert "red" in comp and not o._cache("s", m).exists()
    o.composite(site, date(2026, 9, 1), [SimpleNamespace(id="a")], today=date(2026, 11, 23))
    assert o._cache("s", date(2026, 9, 1)).exists()


def test_composite_written_before_month_end_is_recomputed(monkeypatch, opt_env):
    site, _ = opt_env
    monkeypatch.setattr(Optical, "_scene", _scene_ok)
    o = Optical(workers=1)
    m = date(2020, 5, 1)
    o.composite(site, m, [SimpleNamespace(id="a")])
    path = o._cache("s", m)
    early = datetime(2020, 5, 20).timestamp()
    os.utime(path, (early, early))  # comme un fichier écrit en cours de mois par l'ancien code
    assert o._cached("s", m) is None


def test_cache_path_depends_on_settings(opt_env):
    _, root = opt_env
    m = date(2017, 4, 1)
    assert Optical()._cache("s", m) == root / "cache" / "optical" / "s" / "2017-04.npz"  # inchangé
    assert Optical(scenes_per_month=1, scl_checks=4)._cache("s", m) == \
        root / "cache" / "optical" / "s" / "s1c4" / "2017-04.npz"


def _write(path, reglage=None, n=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = {"valid": np.ones((n, n), "uint8"), "red": np.full((n, n), 0.1, "float32"),
            "nir": np.full((n, n), 0.3, "float32"), "swir": np.full((n, n), 0.2, "float32"),
            "water": np.zeros((n, n), "float32"), "scenes": np.array(["a"])}
    extra = {} if reglage is None else {"reglage": np.array(reglage, "uint8")}
    np.savez_compressed(path, **optical.pack(comp), **extra)


def test_calibration_reads_its_own_cache_then_unlabelled_legacy(opt_env):
    _, root = opt_env
    d = root / "cache" / "optical" / "s"
    _write(d / "2016-01.npz")  # écrit par le 1er run de calibration (1 scène, sans étiquette)
    _write(d / "2016-02.npz", reglage=[3, 6])  # écrit depuis par le suivi (3 scènes) : refusé
    _write(d / "s1c4" / "2016-03.npz", reglage=[1, 4])
    comps = calibrate._comps("s")
    assert sorted(comps) == [date(2016, 1, 1), date(2016, 3, 1)]
    # le suivi (3 scènes) ne relit pas un composite étiqueté 1 scène, ni l'inverse
    assert Optical()._cached("s", date(2016, 2, 1)) is not None
    assert Optical(scenes_per_month=1, scl_checks=4)._cached("s", date(2016, 1, 1)) is None  # sans repli
    assert calibrate._optical()._cached("s", date(2016, 1, 1)) == d / "2016-01.npz"
    calibrate._window_km2.cache_clear()
    assert calibrate._window_km2("s") == pytest.approx(100 * optical.RES ** 2 / 1e6)
    calibrate._window_km2.cache_clear()


# -- modèle « surface nue » : référence fenêtre seule et fenêtres saturées -----------------------------
def test_size_model_reports_window_baseline_and_drops_saturated():
    rng = np.random.default_rng(1)
    base = calibrate.GRID.index({"ndvi_bare": 0.2, "swir_change": 0.08})
    rows = []
    for i in range(30):
        win = float(10 ** rng.uniform(0.5, 2.3))
        frac = 0.95 if i < 4 else float(rng.uniform(0.05, 0.6))
        P = 10 ** (6 + 0.5 * np.log10(win) + rng.normal(0, 0.2))
        rows.append({"id": f"m{i}", "serie": "minerai extrait", "grid": base, "P": P,
                     "bare": frac * win, "fenetre_km2": win})
    out = calibrate.fit_size_model(pd.DataFrame(rows))["minerai"]
    assert out["exclues_saturees"] == 4 and out["n"] == 26
    assert {"fenetre_seule", "fraction_nue_seule", "rho"} <= set(out)
    assert out["rho"]["fenetre"] > out["rho"]["fraction_nue"]  # ici, seule la fenêtre porte la taille


def test_persistence_needs_neighbouring_months():
    import pandas as pd

    from strate import fusion

    z = pd.Series([3.0, 3.0, 3.0, 3.0], index=pd.to_datetime(["2024-09-01", "2024-10-01", "2025-04-01",
                                                                "2025-05-01"]))
    a = fusion.persistent(z)
    assert a.tolist() == [0.0, 3.0, 0.0, 3.0]  # octobre confirme septembre ; avril ne confirme pas octobre
