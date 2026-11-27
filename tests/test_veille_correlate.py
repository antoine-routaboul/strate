"""Corrélateur de la veille : suivi, analyses (exécutant simulé), étude d'événements sur données synthétiques.

Aucun accès réseau : les analyses satellite et les dossiers Mine sont injectés.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from strate.veille import correlate, store

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = store.init(tmp_path / "veille.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def no_catalog(monkeypatch):
    """Les tests ne lisent jamais le vrai catalogue pour les jumelles."""
    monkeypatch.setattr(correlate, "_twin_map", dict)


def gkg_done(con, stamp="20200101000000"):
    """Flux GKG traité depuis `stamp` : c'est ce qui ouvre la couverture d'actualité de toutes les mines."""
    con.execute("INSERT OR IGNORE INTO feed_files(name, feed, stamp, status) VALUES (?, 'gkg', ?, 'done')",
                (f"{stamp}.gkg.csv.zip", stamp))


def add_event(con, site, when, etype="accident", conf=0.8, name=None):
    iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = f"d{site}{iso}{etype}{conf}"
    con.execute("INSERT OR IGNORE INTO raw_docs(id, source, fetched_at, url, status) "
                "VALUES (?,?,?,?, 'done')", (doc, "gdelt_gkg", iso, f"https://ex.org/{doc}"))
    con.execute(
        """INSERT INTO events(id, doc_id, site_id, site_name, event_type, published_at, month, url,
                              confidence, match_method, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (doc + "e", doc, site, name or site.upper(), etype, iso, iso[:7], f"https://ex.org/{doc}", conf,
         "nom", iso))


def month(i: int) -> datetime:
    """i-ième mois depuis janvier 2020 (le 15)."""
    return datetime(2020 + i // 12, i % 12 + 1, 15, tzinfo=UTC)


# -- suivi automatique ------------------------------------------------------------------------------
def test_track_ranks_caps_and_filters(con):
    for k in range(3):
        add_event(con, "a", NOW - timedelta(days=k + 1), conf=0.9)
    add_event(con, "b", NOW - timedelta(days=2), conf=0.9)
    add_event(con, "c", NOW - timedelta(days=2), conf=0.3)  # trop incertain
    add_event(con, "d", NOW - timedelta(days=60), conf=0.9)  # trop ancien
    add_event(con, "e", NOW - timedelta(days=3), conf=0.6)
    con.commit()
    assert correlate.track(con, new_per_day=2, now=NOW) == 2
    rows = {r["site_id"]: dict(r) for r in con.execute("SELECT * FROM tracked")}
    assert set(rows) == {"a", "b"}  # classées par nombre × confiance, plafond respecté
    assert rows["a"]["events_count"] == 3 and rows["a"]["analysis_status"] == "en file"
    assert "3 événements en 30 jours" in rows["a"]["reason"]
    # même jour : plafond déjà atteint
    assert correlate.track(con, new_per_day=2, now=NOW + timedelta(hours=1)) == 0
    # lendemain : la place se libère, « e » entre ; « c » et « d » jamais
    assert correlate.track(con, new_per_day=2, now=NOW + timedelta(days=1)) == 1
    assert {r[0] for r in con.execute("SELECT site_id FROM tracked")} == {"a", "b", "e"}
    # les compteurs des mines suivies sont rafraîchis
    add_event(con, "b", NOW + timedelta(days=1), conf=0.9)
    correlate.track(con, new_per_day=0, now=NOW + timedelta(days=2))
    b = con.execute("SELECT events_count, last_event_at FROM tracked WHERE site_id='b'").fetchone()
    assert b[0] == 2 and b[1].startswith("2026-11-27")


# -- analyses satellite -----------------------------------------------------------------------------
def test_analyses_bounded_status_and_errors(con):
    for sid, n in (("x", 5), ("y", 3), ("z", 1)):
        con.execute("INSERT INTO tracked(site_id, reason, since, events_count, analysis_status) "
                    "VALUES (?, 'r', ?, ?, 'en file')", (sid, "2026-11-26T00:00:00Z", n))
    con.commit()
    calls = []

    def runner(sid):
        calls.append(sid)
        if sid == "y":
            raise RuntimeError("serveur saturé")

    r = correlate.analyse_pending(con, 2, runner, NOW)
    assert calls == ["x", "y"]  # les plus citées d'abord, au plus 2
    assert r == {"analyses": 1, "analyses_erreurs": 1, "analyses_en_attente": 1}
    st = dict(con.execute("SELECT site_id, analysis_status FROM tracked").fetchall())
    assert st["x"] == "terminé" and st["y"] == "erreur : serveur saturé" and st["z"] == "en file"
    # cycle suivant : z (jamais analysée) passe ; x est récente, y attend 24 h avant de réessayer
    calls.clear()
    correlate.analyse_pending(con, 2, runner, NOW + timedelta(minutes=15))
    assert calls == ["z"]
    calls.clear()
    correlate.analyse_pending(con, 5, runner, NOW + timedelta(days=2))
    assert calls == ["y"]
    calls.clear()
    correlate.analyse_pending(con, 5, lambda s: calls.append(s), NOW + timedelta(days=40))
    assert sorted(calls) == ["x", "y", "z"]  # analyses de plus de 30 jours : rafraîchies


# -- statistiques -----------------------------------------------------------------------------------
def test_benjamini_hochberg():
    q = correlate.bh([0.01, 0.04, 0.03, None, 0.5])
    assert q[3] is None
    assert q[0] == pytest.approx(0.04)  # 0.01 * 4 / 1
    assert q[2] == pytest.approx(0.04 * 4 / 3)  # monotonie : min(0.03*4/2, 0.04*4/3)
    assert q[1] == pytest.approx(0.04 * 4 / 3)
    assert q[4] == pytest.approx(0.5)


def test_cmh_exact_single_stratum_is_fisher():
    """Avec une seule mine, le test CMH exact est le test exact de Fisher."""
    from scipy.stats import fisher_exact

    t, k, ne, a = 30, 7, 6, 4
    r = correlate._cmh_exact(np.array([[t, k, ne, a]]))
    _, p = fisher_exact([[a, ne - a], [k - a, t - ne - k + a]])
    assert r["p_value"] == pytest.approx(p, rel=1e-6)
    assert r["expected"] == pytest.approx(ne * k / t)


def test_cmh_exact_has_no_floor_and_skips_degenerate_strata():
    strong = np.array([[60, 3, 3, 3]] * 8 + [[60, 0, 0, 0], [60, 60, 2, 2]])  # 2 mines dégénérées
    r = correlate._cmh_exact(strong)
    assert r["p_value"] < 1e-15  # bien sous l'ancien plancher 1/501 de la permutation
    assert r["expected"] == pytest.approx(8 * 3 * 3 / 60)
    assert correlate._cmh_exact(np.array([[60, 0, 5, 0]] * 6))["p_value"] is None  # aucune anomalie


def test_mantel_haenszel_risk_ratio():
    counts = np.array([[20, 5, 4, 2], [40, 6, 10, 3]], dtype=float)
    # Σ a·n0/T / Σ c·n1/T = (2·16/20 + 3·30/40) / (3·4/20 + 3·10/40)
    rr = (2 * 16 / 20 + 3 * 30 / 40) / (3 * 4 / 20 + 3 * 10 / 40)
    r = correlate._lift_test(counts, np.random.default_rng(0), 200)
    assert r["lift"] == pytest.approx(rr)
    assert r["corrected"] is False


def test_perfect_association_gives_finite_lift_and_verdict():
    r = correlate._lift_test(np.array([[60, 3, 3, 3]] * 8, dtype=float), np.random.default_rng(0), 300)
    assert r["corrected"] is True
    assert r["lift"] == pytest.approx(3.5 * 58 / (0.5 * 4))  # +0,5 par case : fini et élevé
    assert r["lift_lo"] > 1 and r["lift_hi"] >= r["lift_lo"]
    row = dict(event_type="accident", lag_months=0, q_value=0.001, lift_lo=r["lift_lo"], lift_hi=r["lift_hi"])
    assert correlate._verdict(row) == "lien démontré"
    assert correlate._verdict(row | {"lag_months": 2}) == "exploratoire"
    assert correlate._verdict(row | {"q_value": None}) == "trop peu de données"


def test_no_coincidence_is_not_a_positive_lift():
    """Aucune anomalie sur un mois d'événement : lift 0 (exact), jamais poussé au-dessus de 1 par la
    correction de continuité ; aucune anomalie du tout : rien à estimer."""
    rng = np.random.default_rng(0)
    r = correlate._lift_test(np.array([[60, 6, 3, 0]] * 8, dtype=float), rng, 200)
    assert r["lift"] == 0.0 and r["lift_hi"] < 1 and r["corrected"] is False
    r = correlate._lift_test(np.array([[60, 0, 3, 0]] * 8, dtype=float), rng, 200)
    assert r["lift"] is None and r["lift_lo"] is None and r["p_value"] is None


# -- étude d'événements -----------------------------------------------------------------------------
N_MONTHS = 60  # 2020-01 → 2024-12


def make_dossiers(event_months: dict[str, list[int]], lag: int, noise_seed: int = 0, z_fn=None):
    """Dossiers synthétiques : anomalie persistante bare_km2 au mois d'événement + lag, plus du bruit."""
    rng = np.random.default_rng(noise_seed)
    out = {}
    for sid, ems in event_months.items():
        a = np.zeros(N_MONTHS)
        for m in ems:
            if 0 <= m + lag < N_MONTHS:
                a[m + lag] = 2.5
        noise = rng.random(N_MONTHS) < 0.04
        a[noise] = -2.2
        z = z_fn(sid) if z_fn else a + rng.normal(0, 0.3, N_MONTHS)
        out[sid] = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=N_MONTHS, freq="MS"),
                                 "a_bare_km2": a, "z_bare_km2": z})
    return out


def seed_events(con, event_months, etype="accident"):
    gkg_done(con)  # collecte d'actualité depuis 2020-01
    for sid, ems in event_months.items():
        con.execute("INSERT INTO tracked(site_id, reason, since) VALUES (?, 'r', '2026-01-01T00:00:00Z')",
                    (sid,))
        add_event(con, sid, month(0), etype="autre", conf=0.9)
        for m in ems:
            add_event(con, sid, month(m), etype=etype)
    con.commit()


def test_study_finds_planted_link_at_lag_zero(con):
    ev = {f"m{i}": [5 + i, 20 + 2 * i, 40 + i] for i in range(8)}  # 8 mines, 24 mois-événements
    seed_events(con, ev)
    dos = make_dossiers(ev, lag=0)
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=300, signals=["bare_km2"])
    acc = {r["lag_months"]: r for r in rows if r["event_type"] == "accident"}
    assert set(acc) == set(range(-3, 4))
    hit = acc[0]
    assert hit["n_mines"] == 8 and hit["n_events"] == 24
    assert hit["lift"] > 5 and hit["lift_lo"] > 1
    assert hit["p_value"] < 1e-6 and hit["q_value"] < 0.10
    assert hit["verdict"] == "lien démontré" and hit["details"]["famille"] == "principale"
    assert hit["details"]["p_anomalie_evenement"] == pytest.approx(1.0)
    # les autres décalages : calculés, jamais affirmés
    assert all(acc[k]["verdict"] == "exploratoire" for k in acc if k)
    assert acc[2]["details"]["famille"] == "exploratoire"
    # type jamais vu : aucune ligne ; « autre » (8 mois-événements < 10) : pas testé, pas de q-valeur
    assert not [r for r in rows if r["event_type"] == "social"]
    autre = [r for r in rows if r["event_type"] == "autre"]
    assert autre and all(r["verdict"] == "trop peu de données" and r["q_value"] is None for r in autre)


def test_study_lagged_link_is_only_exploratory(con):
    ev = {f"m{i}": [5 + i, 20 + 2 * i, 40 + i] for i in range(8)}
    seed_events(con, ev)
    rows = correlate.study(con, make_dossiers(ev, lag=1).__getitem__, now=NOW, n_boot=300,
                           signals=["bare_km2"])
    acc = {r["lag_months"]: r for r in rows if r["event_type"] == "accident"}
    assert acc[1]["q_value"] < 0.10 and acc[1]["lift_lo"] > 1  # repéré dans sa propre famille…
    assert acc[1]["verdict"] == "exploratoire"  # …mais rien n'est déclaré hors du décalage 0
    assert acc[0]["verdict"] == "non démontré"
    assert not [r for r in rows if r["verdict"] == "lien démontré"]


def test_strong_link_survives_full_production_family(con):
    """15 mines × 5 signaux × 8 types × 7 décalages : un vrai lien n'est plus noyé par un plancher de p."""
    rng = np.random.default_rng(11)
    n = 72
    sigs = [f"s{j}" for j in range(5)]
    types = list(store.EVENT_TYPES)
    gkg_done(con)
    dos = {}
    for i in range(15):
        sid = f"p{i}"
        con.execute("INSERT INTO tracked(site_id, reason, since) VALUES (?, 'r', '2026-01-01T00:00:00Z')",
                    (sid,))
        ems = {t: sorted(rng.choice(n, 2, replace=False).tolist()) for t in types}
        for t, ms in ems.items():
            for m in ms:
                add_event(con, sid, month(m), etype=t)
        df = {"date": pd.date_range("2020-01-01", periods=n, freq="MS")}
        for s in sigs:
            a = np.where(rng.random(n) < 0.05, 2.3, 0.0)
            if s == "s0":
                a[ems["accident"]] = 2.5
            df[f"a_{s}"], df[f"z_{s}"] = a, a
        dos[sid] = pd.DataFrame(df)
    con.commit()
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=200, signals=sigs)
    primary = [r for r in rows if r["lag_months"] == 0 and r["q_value"] is not None]
    assert len(primary) >= 40  # toute la famille principale (types × signaux) a été testée
    hit = next(r for r in rows if (r["event_type"], r["signal"], r["lag_months"]) == ("accident", "s0", 0))
    assert hit["n_mines"] == 15 and hit["q_value"] < 0.01
    assert hit["verdict"] == "lien démontré"
    false = [r for r in rows if r["verdict"] == "lien démontré" and r is not hit]
    assert len(false) <= 1


def test_study_too_few_mines(con):
    ev = {f"m{i}": [5 + i, 20 + i, 30 + i, 40 + i, 50 + i] for i in range(3)}  # 15 événements, 3 mines
    seed_events(con, ev)
    rows = correlate.study(con, make_dossiers(ev, lag=0).__getitem__, now=NOW, n_boot=100,
                           signals=["bare_km2"])
    r = next(r for r in rows if r["event_type"] == "accident" and r["lag_months"] == 0)
    assert r["lift"] > 1  # calculé et affiché…
    assert r["verdict"] == "trop peu de données" and r["q_value"] is None  # …mais jamais affirmé


def test_study_no_link_on_independent_data(con):
    rng = np.random.default_rng(7)
    ev = {f"m{i}": sorted(rng.choice(N_MONTHS, 4, replace=False).tolist()) for i in range(10)}
    seed_events(con, ev)
    # anomalies placées ailleurs, indépendamment des événements
    other = {k: sorted(rng.choice(N_MONTHS, 4, replace=False).tolist()) for k in ev}
    dos = make_dossiers(other, lag=0, noise_seed=3)
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=200, signals=["bare_km2"])
    acc = [r for r in rows if r["event_type"] == "accident"]
    assert all(r["verdict"] != "lien démontré" for r in acc)


# -- couverture d'actualité -------------------------------------------------------------------------
def test_coverage_from_gkg_feed_and_doc_history(con):
    assert correlate.coverage(con) == (None, {})
    gkg_done(con, "20260512091500")
    con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES ('x', 'gkg', '20250101000000', "
                "'missing')")  # jamais lu : ne compte pas
    for sid, status in (("fait1", "fait"), ("pasfait", None)):
        con.execute("INSERT INTO tracked(site_id, reason, since, backfill_status) VALUES (?, 'r', ?, ?)",
                    (sid, "2026-11-01T00:00:00Z", status))
        for d in ("2024-03-02T10:00:00Z", "2026-10-01T10:00:00Z"):
            con.execute("INSERT INTO raw_docs(id, source, fetched_at, published_at, url, hint_site, status) "
                        "VALUES (?, 'gdelt_doc', ?, ?, ?, ?, 'rejected')", (sid + d, d, d, "u" + d, sid))
    con.commit()
    gkg, doc = correlate.coverage(con)
    assert gkg == correlate._midx(2026, 5)
    assert doc == {"fait1": correlate._midx(2024, 3)}  # historique non fait : rien ne prouve la collecte
    # jumelle renommée dans `tracked` : ses articles gardent l'ancien id dans hint_site
    con.execute("INSERT INTO tracked(site_id, reason, since, backfill_status) VALUES "
                "('real9', 'r', '2026-11-01T00:00:00Z', 'fait')")
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, published_at, url, hint_site, status) "
                "VALUES ('j9', 'gdelt_doc', 'x', '2023-07-01T00:00:00Z', 'u9', 'icmm9', 'done')")
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, published_at, url, hint_site, status) "
                "VALUES ('g9', 'gdelt_gkg', 'x', '2019-01-01T00:00:00Z', 'g9', 'real9', 'done')")
    con.commit()
    _, doc = correlate.coverage(con, lambda s: {"icmm9": "real9"}.get(s, s))
    assert doc["real9"] == correlate._midx(2023, 7)  # seuls les articles de l'API documentaire comptent


def test_volume_timeline_does_not_extend_coverage(con):
    """Frise de volume depuis 2017, événements récents seulement, alertes plus fréquentes depuis un an :
    la tendance des alertes ne doit pas passer pour un lien (défaut d'origine : couverture dès 2017)."""
    rng = np.random.default_rng(5)
    n = 82  # 2020-01 → 2026-10
    dos = {}
    for i in range(12):
        sid = f"t{i}"
        con.execute("INSERT INTO tracked(site_id, reason, since) VALUES (?, 'r', '2026-06-01T00:00:00Z')",
                    (sid,))
        for y in range(2017, 2027):
            for mo in range(1, 13):
                if (y, mo) <= (2026, 11):
                    con.execute("INSERT INTO news_volume VALUES (?,?,?, 'gdelt_doc_timeline', ?)",
                                (sid, f"{y}-{mo:02d}", float(rng.poisson(4)), "2026-11-01T00:00:00Z"))
        for m in sorted(rng.choice(range(77, 83), 3, replace=False)):  # 2026-06 → 2026-11
            add_event(con, sid, month(int(m)), etype="operations")
        p = np.where(np.arange(n) < 70, 0.04, 0.25)  # plus d'alertes depuis 2025-11, sans lien
        a = np.where(rng.random(n) < p, 2.4, 0.0)
        dos[sid] = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="MS"),
                                 "a_bare_km2": a, "z_bare_km2": a})
    con.commit()
    # sans flux GKG traité ni historique documentaire : aucune mine n'a de couverture
    stats: dict = {}
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=200, signals=["bare_km2"], stats=stats)
    assert not [r for r in rows if r["event_type"] == "operations"] and stats["sans_couverture"] == 12
    gkg_done(con, "20260501000000")
    con.commit()
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=200, signals=["bare_km2"])
    op = {r["lag_months"]: r for r in rows if r["event_type"] == "operations"}
    assert op[0]["details"]["mois_apparies"] == 12 * 6  # 2026-05 → 2026-10 seulement, pas depuis 2017
    assert op[0]["verdict"] != "lien démontré"
    assert not [r for r in rows if r["verdict"] == "lien démontré"]


def test_volume_rank_correlation(con):
    rng = np.random.default_rng(2)
    sites = [f"v{i}" for i in range(6)]
    zs = {}
    gkg_done(con)
    for sid in sites:
        con.execute("INSERT INTO tracked(site_id, reason, since) VALUES (?, 'r', '2026-01-01T00:00:00Z')",
                    (sid,))
        vol = rng.poisson(5, N_MONTHS).astype(float) * (1 + sites.index(sid))  # échelle propre à la mine
        zs[sid] = np.sign(rng.normal(size=N_MONTHS)) * (vol / vol.max() * 3 + rng.normal(0, 0.2, N_MONTHS))
        for i, n in enumerate(vol):
            con.execute("INSERT INTO news_volume VALUES (?,?,?, 'gdelt_doc_timeline', ?)",
                        (sid, month(i).strftime("%Y-%m"), n, "2026-11-01T00:00:00Z"))
            con.execute("INSERT INTO news_volume VALUES (?,?,?, 'events', ?)",  # ignorée : frise préférée
                        (sid, month(i).strftime("%Y-%m"), 999.0, "2026-11-01T00:00:00Z"))
    con.commit()
    dos = make_dossiers({s: [] for s in sites}, lag=0, z_fn=zs.__getitem__)
    rows = correlate.study(con, dos.__getitem__, now=NOW, n_boot=200, signals=["bare_km2"])
    vol = {r["lag_months"]: r for r in rows if r["event_type"] == "volume"}
    assert vol[0]["n_mines"] == 6
    assert vol[0]["rho"] > 0.8 and vol[0]["rho_lo"] > 0
    assert vol[0]["p_value"] < 1e-10  # loi normale de la permutation : pas de plancher
    assert vol[0]["verdict"] == "lien démontré"
    assert abs(vol[2]["rho"]) < 0.3 and vol[2]["verdict"] == "exploratoire"


def test_rho_normal_p_matches_permutation():
    """Moyenne et variance de la permutation intra-mine : la p-valeur normale suit la permutation."""
    rng = np.random.default_rng(3)
    pairs = []
    for _ in range(6):
        v = rng.poisson(5, 30).astype(float)
        pairs.append((v, v * 0.06 + rng.normal(0, 1, 30)))
    r = correlate._rho_test(pairs, rng, 100)
    obs, perm = 0.0, np.zeros(4000)
    for v, s in pairs:
        cv, cs = correlate._ranks(v), correlate._ranks(s)
        obs += cv @ cs
        perm += rng.permuted(np.tile(cv, (4000, 1)), axis=1) @ cs
    p_perm = (np.abs(perm) >= abs(obs)).mean()
    assert 0.001 < r["p_value"] < 0.5
    assert r["p_value"] == pytest.approx(p_perm, abs=0.02)


# -- cycle complet ----------------------------------------------------------------------------------
def test_run_once_writes_rows_and_respects_recompute(con):
    ev = {f"m{i}": [5 + i, 20 + 2 * i, 40 + i] for i in range(6)}
    seed_events(con, ev)
    add_event(con, "fresh", NOW - timedelta(days=1))
    con.commit()
    dos = make_dossiers(ev, lag=0)
    calls = []
    kw = dict(runner=calls.append, dossier_fn=lambda s: dos.get(s, pd.DataFrame()), n_boot=100, twins={})
    c = correlate.run_once(con, new_per_day=5, analyses_per_run=1, now=NOW, **kw)
    assert c["nouvelles_suivies"] == 1 and c["recalcul"] and c["lignes"] > 0
    assert len(calls) == 1
    st = con.execute("SELECT ok, counters FROM pipeline_status WHERE stage='correlation'").fetchone()
    assert st[0] == 1 and json.loads(st[1])["lignes"] == c["lignes"]
    n = con.execute("SELECT COUNT(*) FROM correlations WHERE scope='global'").fetchone()[0]
    assert n == c["lignes"]
    row = con.execute("SELECT details, verdict FROM correlations WHERE event_type='accident' "
                      "AND lag_months=0 AND signal='bare_km2'").fetchone()
    assert json.loads(row[0])["mines_panel"] == 6 and row[1] == "lien démontré"
    assert c["liens_demontres"] >= 1 and c["exploratoires"] >= 1
    # 15 min plus tard : pas de recalcul (toutes les 6 h au plus)
    c2 = correlate.run_once(con, now=NOW + timedelta(minutes=15), **kw)
    assert not c2["recalcul"]
    c3 = correlate.run_once(con, now=NOW + timedelta(hours=7), **kw)
    assert c3["recalcul"]
    assert con.execute("SELECT COUNT(DISTINCT run_at) FROM correlations").fetchone()[0] == 2


def test_empty_recompute_clears_stale_verdicts(con):
    ev = {f"m{i}": [5 + i, 20 + 2 * i, 40 + i] for i in range(6)}
    seed_events(con, ev)
    dos = make_dossiers(ev, lag=0)
    kw = dict(runner=lambda s: None, dossier_fn=lambda s: dos.get(s, pd.DataFrame()), n_boot=50, twins={})
    assert correlate.run_once(con, now=NOW, **kw)["lignes"] > 0
    con.execute("DELETE FROM events")
    con.commit()
    c = correlate.run_once(con, now=NOW + timedelta(hours=7), **kw)
    assert c["recalcul"] and c["lignes"] == 0
    # l'API lit max(run_at) : plus aucun ancien « lien démontré » présenté comme actuel
    assert con.execute("SELECT MAX(run_at) FROM correlations WHERE scope='global'").fetchone()[0] is None
    msg = con.execute("SELECT message FROM pipeline_status WHERE stage='correlation'").fetchone()[0]
    assert "aucune combinaison calculable" in msg


def test_run_once_isolates_errors(con):
    def boom(_sid):
        raise ValueError("dossier illisible")

    gkg_done(con)
    add_event(con, "a", NOW - timedelta(days=1))
    con.commit()
    c = correlate.run_once(con, now=NOW, runner=lambda s: None, dossier_fn=boom, n_boot=10, twins={})
    assert c["nouvelles_suivies"] == 1 and c["analyses"] == 1  # le reste du cycle a tourné
    assert c["lignes"] == 0  # mine au dossier illisible : sortie du panel, sans faire planter l'étude
    assert c["dossiers_illisibles"] == 1
    st = con.execute("SELECT ok, message FROM pipeline_status WHERE stage='correlation'").fetchone()
    assert st[0] == 1 and "1 dossier(s) illisible(s)" in st[1]


# -- jumelles du catalogue --------------------------------------------------------------------------
def test_twin_ids_are_tracked_and_analysed_once(con):
    twins = {"icmm1": "real1", "icmm2": "real2"}
    # suivie avant le renvoi : la jumelle icmm2 devient real2 ; icmm1 sera fusionnée avec real1
    for sid in ("icmm1", "real1", "icmm2"):
        con.execute("INSERT INTO tracked(site_id, reason, since) VALUES (?, 'r', '2026-11-01T00:00:00Z')",
                    (sid,))
    add_event(con, "icmm3", NOW - timedelta(days=1))
    add_event(con, "real3", NOW - timedelta(days=2))
    add_event(con, "icmm1", NOW - timedelta(days=3))
    con.commit()
    twins["icmm3"] = "real3"
    assert correlate.track(con, new_per_day=5, now=NOW, twins=twins) == 1
    rows = {r["site_id"]: dict(r) for r in con.execute("SELECT * FROM tracked")}
    assert set(rows) == {"real1", "real2", "real3"}
    assert rows["real3"]["events_count"] == 2 and rows["real1"]["events_count"] == 1
    assert "2 événements" in rows["real3"]["reason"]
    calls = []
    correlate.analyse_pending(con, 10, calls.append, NOW)
    assert sorted(calls) == ["real1", "real2", "real3"]  # jamais une seconde analyse sous l'id jumeau
    # étude : les événements de la jumelle comptent pour la mine suivie
    gkg_done(con)
    con.commit()
    seen = []

    def dossier(sid):
        seen.append(sid)
        return make_dossiers({sid: []}, lag=0)[sid]

    correlate.study(con, dossier, now=NOW, n_boot=10, signals=["bare_km2"], twins=twins)
    assert sorted(seen) == ["real1", "real2", "real3"]


def test_methode_describes_the_computation():
    m = correlate.METHODE
    for part in ("Cochran–Mantel–Haenszel", "Mantel–Haenszel", "0,5", "bootstrap", "décalage 0",
                 "exploratoires", "GKG", "article le plus ancien", "q < 0,10", "5 mines", "10 mois"):
        assert part in m
    assert "permutation (mois d'événement mélangés" not in m  # l'ancien test n'existe plus


def test_coverage_reads_doc_oldest_meta_even_after_purge(con):
    # la collecte note « doc_oldest:<mine> » (plus ancien article de l'historique, y compris ceux déjà vus
    # par le flux GKG) : la couverture la lit même si raw_docs a été purgé
    con.execute("INSERT INTO tracked(site_id, reason, since, backfill_status) VALUES "
                "('m1', 'r', '2026-11-01T00:00:00Z', 'fait'), ('m2', 'r', '2026-11-01T00:00:00Z', NULL)")
    con.execute("INSERT INTO meta(key, value) VALUES ('doc_oldest:m1', '2022-04-17T08:00:00Z'), "
                "('doc_oldest:m2', '2021-01-01T00:00:00Z'), ('doc_oldest:m3', 'abîmé')")
    con.commit()
    _, doc = correlate.coverage(con)
    assert doc == {"m1": correlate._midx(2022, 4)}       # m2 : historique non fait ; m3 : valeur illisible
    # une ligne raw_docs plus récente ne remplace pas la date notée
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, published_at, url, hint_site, status) "
                "VALUES ('r1', 'gdelt_doc', 'x', '2025-01-01T00:00:00Z', 'u1', 'm1', 'done')")
    con.commit()
    assert correlate.coverage(con)[1] == {"m1": correlate._midx(2022, 4)}


def test_continuity_wording_matches_docs():
    # même règle dans METHODE et docs/veille.md : 0,5 seulement quand aucune anomalie hors des mois
    # d'événement (dénominateur nul) ; un numérateur nul reste un lift exact de 0
    from pathlib import Path
    doc = " ".join((Path(__file__).parents[1] / "docs" / "veille.md").read_text(encoding="utf-8").split())
    phrase = "quand aucune anomalie ne tombe hors des mois d'événement"
    assert phrase in correlate.METHODE and phrase in doc
    assert "quand une case est vide" not in doc
