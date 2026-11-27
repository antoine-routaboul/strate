"""Veille, étape 1 (collecteur) : lecture GKG sur un extrait réel réduit, planification des quarts d'heure,
404 → missing → abandon, API documentaire sous métronome (429 / Retry-After), noms multilingues.
Aucun accès réseau : client httpx sur MockTransport, horloge factice pour le métronome."""

import io
import itertools
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from strate import entities
from strate.veille import collect, store

FIX = Path(__file__).parent / "fixtures"
EN = FIX / "gdelt_sample.gkg.csv"
TR = FIX / "gdelt_sample.translation.gkg.csv"
NOW = datetime(2026, 11, 26, 12, 7, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = store.init(tmp_path / "veille.db")
    yield c
    c.close()


def _zip(path: Path, inner: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner, path.read_bytes())
    return buf.getvalue()


class FakeTime:
    def __init__(self, t0: float = 1_000_000.0):
        self.t = t0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, d: float) -> None:
        self.slept.append(d)
        self.t += d


def _lastupdate(stamp: str, suffix: str) -> str:
    return (f"1 aa {collect.GKG_BASE}{stamp}.export.CSV.zip\n"
            f"2 bb {collect.GKG_BASE}{stamp}.mentions.CSV.zip\n"
            f"3 cc {collect.GKG_BASE}{stamp}{suffix}\n")


# -- lecture GKG -----------------------------------------------------------------------------------------
def test_parse_real_sample_english():
    stats: dict = {}
    docs = list(collect.iter_gkg(EN.open(encoding="utf-8", newline="\n"), "gkg", "2026-11-26T00:00:00Z",
                                 stats))
    assert stats == {"read": 6, "kept": 3}
    d = docs[0]
    assert d["source"] == "gdelt_gkg" and d["lang"] == "en" and d["hint_site"] is None
    assert d["id"] == collect.doc_id(d["url"]) and len(d["id"]) == 40
    assert d["published_at"] == "2026-11-25T11:41:00Z"  # PAGE_PRECISEPUBTIMESTAMP, pas la date de collecte
    assert "ENV_MINING" in d["themes"].split(";") and "," not in d["themes"]  # offsets retirés
    assert d["organizations"].startswith("Zoomlion Co")  # colonne V2 : casse d'origine
    p = json.loads(d["payload"])
    assert p["kept"] == "thème:ENV_MINING" and p["crawl"] == "2026-11-25T12:00:00Z"
    assert isinstance(d["tone"], float)


def test_parse_real_sample_translation():
    stats: dict = {}
    docs = list(collect.iter_gkg(TR.open(encoding="utf-8", newline="\n"), "gkg_tr", "t", stats))
    assert stats == {"read": 8, "kept": 5}
    by_lang = {d["lang"]: d for d in docs}
    assert set(by_lang) == {"es", "fr", "zh", "de"}  # srclc ISO 639-3 → 639-1
    zh = by_lang["zh"]
    assert zh["title"].startswith("業者申請宜蘭員山增3公頃礦區")  # entités HTML décodées
    assert json.loads(zh["payload"])["kept"] == "titre:礦"
    assert json.loads(zh["payload"])["srclc"] == "zho"
    assert json.loads(by_lang["de"]["payload"])["kept"] == "titre:bergwerk"
    mali = next(d for d in docs if "Mali" in d["title"])
    locs = json.loads(mali["locations"])
    assert any(loc["country"] == "Mali" for loc in locs)
    assert all(-90 <= loc["lat"] <= 90 for loc in locs if loc["lat"] is not None)


def test_parse_is_streaming():
    # le lecteur rend le 1er document sans lire la suite : un fichier de 50 Mo n'est jamais chargé entier
    def boom():
        raise AssertionError("lu trop loin")
        yield  # pragma: no cover

    lines = itertools.chain(EN.open(encoding="utf-8", newline="\n"), boom())
    first = next(collect.iter_gkg(lines, "gkg", "t", {}))
    assert first["title"].startswith("Zoomlion")


def test_bad_rows_counted_not_fatal():
    stats: dict = {}
    good = EN.read_text(encoding="utf-8").splitlines(keepends=True)
    docs = list(collect.iter_gkg(["court\tligne\n", "\n"] + good, "gkg", "t", stats))
    assert len(docs) == 3 and stats["bad"] == 1 and stats["read"] == 7


def test_locations_v1_v2_and_fips():
    v2 = ("1#Russia#RS#RS##60#100#RS#276;4#Berlin, Berlin, Germany#GM#GM16#16538#52.5167#13.4#-1746443#243;"
          "1#Russia#RS#RS##60#100#RS#300")
    locs = collect.parse_locations(v2)
    # doublon retiré ; pays lu dans le nom (RS est le code FIPS de la Russie, pas la Serbie)
    assert [loc["country"] for loc in locs] == [entities.norm_country("Russia"), "Germany"]
    assert locs[1]["cc"] == "GM" and locs[1]["lat"] == 52.5167 and locs[1]["type"] == 4
    v1 = "4#Kiruna, Norrbottens Lan, Sweden#SW#SW14#67.85#20.2167#-2530"
    assert collect.parse_locations("", v1)[0]["country"] == "Sweden"
    assert collect.parse_locations("1#X#XX#XX##nan#abc#X#1")[0]["lat"] is None


@pytest.mark.parametrize("title", [
    "Wichtiges Bergwerk vor Neustart", "Авария на шахте в Кузбассе", "Kopalnia Wujek wstrzymuje wydobycie",
    "La minería chilena crece", "Mineração: Vale retoma produção", "山西煤矿事故", "佐渡金山の鉱山",
    "광산 붕괴 사고", "Tambang emas ilegal ditutup", "Maden ocağında grev", "Mina Escondida en huelga",
    "LKAB:s gruva i Kiruna", "Kaivosyhtiö Terrafame", "انهيار منجم ذهب", "קריסה במכרה", "ορυχείο λιγνίτη",
    "Gold mine workers strike", "Rescue of trapped miners", "Coal mining output falls",
    "Gouvernance minière au Mali", "Khai thác mỏ than Quảng Ninh", "Bányászati baleset", "Рудник Удачный",
])
def test_title_mine_keyword_multilingual(title):
    assert collect.title_hit(title)


@pytest.mark.parametrize("title", [
    "Nuovo record di Minerva, il chatbot", "Kecelakaan di Banyumas", "Determined to win",
    "Examiner says", "Black Friday deals", "Mineral water prices", "",
])
def test_title_no_false_keyword(title):
    assert collect.title_hit(title) is None


def test_mining_reason_theme_first():
    assert collect.mining_reason(["LEADER", "ENV_MINING"], "Anything") == "thème:ENV_MINING"
    assert collect.mining_reason(["LEADER"], "Gold mine reopens") == "titre:mine"
    assert collect.mining_reason(["LEADER"], "Election results") is None


def test_norm_url_same_article_same_id():
    a = "https://WWW.Example.com/news/mine-collapse/?utm_source=x&id=3#top"
    b = "https://www.example.com/news/mine-collapse?id=3"
    assert collect.norm_url(a) == collect.norm_url(b)
    assert collect.doc_id(a) == collect.doc_id(b)
    assert collect.doc_id("https://example.com/a?id=3") != collect.doc_id("https://example.com/a?id=4")


def test_insert_docs_dedup(con):
    docs = list(collect.iter_gkg(EN.open(encoding="utf-8", newline="\n"), "gkg", "t", {}))
    assert collect.insert_docs(con, docs) == 3
    assert collect.insert_docs(con, docs) == 0
    assert con.execute("SELECT COUNT(*) FROM raw_docs WHERE status='new'").fetchone()[0] == 3


# -- planification et flux ---------------------------------------------------------------------------------
def test_plan_backfill_then_incremental(con):
    latest = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    assert collect.plan(con, "gkg", latest, backfill_hours=2) == 8
    stamps = [r[0] for r in con.execute("SELECT stamp FROM feed_files ORDER BY stamp")]
    assert stamps[0] == "20261126101500" and stamps[-1] == "20261126120000"
    assert collect.plan(con, "gkg", latest, backfill_hours=2) == 0  # idempotent
    assert collect.plan(con, "gkg", latest + timedelta(minutes=45), backfill_hours=2) == 3
    name = con.execute("SELECT name FROM feed_files WHERE stamp='20261126124500'").fetchone()[0]
    assert name == "20261126124500.gkg.csv.zip"
    assert collect.plan(con, "gkg_tr", latest, 0.25) == 1
    assert con.execute("SELECT name FROM feed_files WHERE feed='gkg_tr'").fetchone()[0] == \
        "20261126120000.translation.gkg.csv.zip"


def test_plan_catchup_is_capped(con):
    latest = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    collect.plan(con, "gkg", latest - timedelta(days=30), backfill_hours=0.25)
    n = collect.plan(con, "gkg", latest, backfill_hours=0.25)
    assert n == collect.MAX_CATCHUP // collect.STEP + 1


def _feed_transport(routes: dict, calls: list):
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        calls.append(url)
        for k, v in routes.items():
            if url.endswith(k):
                return v() if callable(v) else v
        return httpx.Response(404, text="not found")
    return httpx.MockTransport(handler)


def test_run_once_end_to_end(con):
    calls: list[str] = []
    en_zip, tr_zip = _zip(EN, "20261126114500.gkg.csv"), _zip(TR, "20261126114500.translation.gkg.csv")
    routes = {
        "lastupdate.txt": httpx.Response(200, text=_lastupdate("20261126120000", ".gkg.csv.zip")),
        "lastupdate-translation.txt": httpx.Response(
            200, text=_lastupdate("20261126114500", ".translation.gkg.csv.zip")),
        "20261126114500.gkg.csv.zip": httpx.Response(200, content=en_zip),
        "20261126114500.translation.gkg.csv.zip": httpx.Response(200, content=tr_zip),
        # 20261126120000.gkg.csv.zip : absent (404) comme souvent pour les fichiers récents
    }
    c = httpx.Client(transport=_feed_transport(routes, calls))
    out = collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c, now=NOW,
                           names=False)
    assert out["files_done"] == 2 and out["files_missing"] == 2
    assert out["docs_read"] == 14 and out["docs_kept"] == 8 and out["docs_new"] == 8
    st = {r["name"]: r["status"] for r in con.execute("SELECT name, status FROM feed_files")}
    assert st == {"20261126113000.translation.gkg.csv.zip": "missing",
                  "20261126114500.translation.gkg.csv.zip": "done",
                  "20261126114500.gkg.csv.zip": "done", "20261126120000.gkg.csv.zip": "missing"}
    row = con.execute("SELECT n_docs, n_kept, tries FROM feed_files WHERE name='20261126114500.gkg.csv.zip'")
    assert tuple(row.fetchone()) == (6, 3, 1)
    assert con.execute("SELECT COUNT(*) FROM raw_docs").fetchone()[0] == 8
    ps = con.execute("SELECT * FROM pipeline_status WHERE stage='collecte'").fetchone()
    assert ps["ok"] == 1 and json.loads(ps["counters"])["docs_new"] == 8
    assert all(u.startswith("http://data.gdeltproject.org/gdeltv2/") for u in calls)

    # 10 min plus tard : l'absent n'est pas réessayé (attente de 15 min après le 1er essai)
    calls.clear()
    collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c,
                     now=NOW + timedelta(minutes=10), names=False)
    assert not any(u.endswith("20261126120000.gkg.csv.zip") for u in calls)
    # 16 min plus tard : réessayé, et cette fois publié
    routes["20261126120000.gkg.csv.zip"] = httpx.Response(200, content=_zip(EN, "x.gkg.csv"))
    collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c,
                     now=NOW + timedelta(minutes=16), names=False)
    r = con.execute("SELECT status, tries FROM feed_files WHERE name='20261126120000.gkg.csv.zip'").fetchone()
    assert tuple(r) == ("done", 2)
    assert con.execute("SELECT COUNT(*) FROM raw_docs").fetchone()[0] == 8  # mêmes URL : pas de doublon


def test_missing_given_up_after_72h(con):
    con.executemany(
        "INSERT INTO feed_files(name, feed, stamp, status, tries, last_try) VALUES (?,?,?,'missing',?,?)",
        [(s + ".gkg.csv.zip", "gkg", s, k, "2026-11-26T10:00:00Z")
         for s, k in (("20261122110000", 9), ("20261125110000", 3))])
    assert collect.give_up_missing(con, NOW) == 1
    st = dict(con.execute("SELECT stamp, status FROM feed_files").fetchall())
    assert st == {"20261122110000": "error", "20261125110000": "missing"}


def test_old_pending_404_is_abandoned(con):
    calls: list[str] = []
    c = httpx.Client(transport=_feed_transport({}, calls))
    con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES "
                "('20261120000000.gkg.csv.zip', 'gkg', '20261120000000', 'pending')")
    row = con.execute("SELECT * FROM feed_files").fetchone()
    assert collect.fetch_file(con, c, row, NOW)["status"] == "error"
    assert "abandonné" in con.execute("SELECT message FROM feed_files").fetchone()[0]


def test_network_error_never_consumes_tries(con):
    # une coupure, même longue, ne rend jamais un fichier « error » : rien n'est perdu au retour du réseau
    def handler(req):
        raise httpx.ConnectError("coupure")
    c = httpx.Client(transport=httpx.MockTransport(handler))
    con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES "
                "('20261126110000.gkg.csv.zip', 'gkg', '20261126110000', 'pending')")
    for _ in range(collect.MAX_TRIES + 3):
        row = con.execute("SELECT * FROM feed_files").fetchone()
        res = collect.fetch_file(con, c, row, NOW)
        assert res["status"] == "pending" and res["net"] and res["connect"]
    r = con.execute("SELECT status, tries, message FROM feed_files").fetchone()
    assert (r["status"], r["tries"]) == ("pending", 0) and r["message"].startswith("réseau")


def test_unreadable_file_consumes_tries_then_error(con):
    c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"PK\x03\x04trunc")))
    con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES "
                "('20261126110000.gkg.csv.zip', 'gkg', '20261126110000', 'pending')")
    for k in range(1, collect.MAX_TRIES + 1):
        row = con.execute("SELECT * FROM feed_files").fetchone()
        res = collect.fetch_file(con, c, row, NOW)
        assert res["status"] == ("error" if k == collect.MAX_TRIES else "pending")
    assert con.execute("SELECT tries FROM feed_files").fetchone()[0] == collect.MAX_TRIES


def test_corrupt_zip_is_not_fatal(con):
    c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"PK\x03\x04trunc")))
    con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES "
                "('20261126110000.gkg.csv.zip', 'gkg', '20261126110000', 'pending')")
    res = collect.fetch_file(con, c, con.execute("SELECT * FROM feed_files").fetchone(), NOW)
    assert res["status"] == "pending" and res.get("error")


def test_lastupdate_down_still_plans_and_reports(con):
    def handler(req):
        if "lastupdate" in str(req.url):
            return httpx.Response(503)
        return httpx.Response(404)
    c = httpx.Client(transport=httpx.MockTransport(handler))
    out = collect.run_once(con, backfill_hours=0.25, max_files=5, doc_queries=0, client=c, now=NOW,
                           names=False)
    assert out["files_missing"] == 2
    ps = con.execute("SELECT ok, message FROM pipeline_status WHERE stage='collecte'").fetchone()
    assert ps["ok"] == 0 and "lastupdate" in ps["message"]


def test_due_reserves_slots_for_missing(con):
    rows = [(f"2026112610{m:02d}00.gkg.csv.zip", "gkg", f"2026112610{m:02d}00", "pending")
            for m in (0, 15, 30, 45)]
    rows += [(f"2026112609{m:02d}00.gkg.csv.zip", "gkg", f"2026112609{m:02d}00", "missing")
             for m in (0, 15)]
    con.executemany("INSERT INTO feed_files(name, feed, stamp, status) VALUES (?,?,?,?)", rows)
    got = [r["stamp"] for r in collect.due(con, NOW, 4)]
    assert got[0] == "20261126090000" and "20261126104500" in got and len(got) == 4


# -- API documentaire ---------------------------------------------------------------------------------------
def test_throttle_min_interval_and_defer():
    ft = FakeTime()
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    th.wait()
    assert ft.slept == []
    th.wait()
    assert ft.slept == [6.0]
    ft.t += 10  # long traitement : pas d'attente inutile
    th.wait()
    assert ft.slept == [6.0]
    th.defer(30)
    th.wait()
    assert ft.slept[-1] == pytest.approx(30)


def _doc_client(responses: list, seen: list, ft: FakeTime):
    it = iter(responses)

    def handler(req):
        seen.append((ft.t, dict(req.url.params)))
        return next(it)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_doc_get_429_retry_after_then_ok():
    ft, seen = FakeTime(), []
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(429, headers={"Retry-After": "20"}, text="slow down"),
                     httpx.Response(200, json={"timeline": []})], seen, ft)
    assert collect.doc_get(c, {"query": "x"}, th) == {"timeline": []}
    assert seen[1][0] - seen[0][0] >= 20


def test_doc_get_rate_limit_text_and_give_up():
    ft, seen = FakeTime(), []
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    limit = "Please limit requests to one every 5 seconds or contact us"
    c = _doc_client([httpx.Response(200, text=limit), httpx.Response(429)], seen, ft)
    with pytest.raises(collect.RateLimited):
        collect.doc_get(c, {"query": "x"}, th)
    assert seen[1][0] - seen[0][0] >= 12  # sans Retry-After : 12 s


def test_doc_get_query_refused():
    ft = FakeTime()
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, text="The specified phrase is too short.")], [], ft)
    with pytest.raises(collect.QueryRefused, match="too short"):
        collect.doc_get(c, {"query": "x"}, th)
    c = _doc_client([httpx.Response(200, text="")], [], ft)
    assert collect.doc_get(c, {"query": "x"}, th) == {}


def test_mine_query_syntax():
    assert collect.mine_query(["Grasberg"]) == '"Grasberg" mine'
    assert collect.mine_query(["Escondida", "Minera Escondida"]) == '("Escondida" OR "Minera Escondida") mine'
    assert collect.mine_query([]) is None


def test_timeline_months_sums_any_resolution():
    js = {"timeline": [{"series": "Article Count", "data": [
        {"date": "20240101T000000Z", "value": 2}, {"date": "20240115T000000Z", "value": 3},
        {"date": "20240201000000", "value": 0}, {"date": "bad", "value": 9}]}]}
    assert collect.timeline_months(js) == {"2024-01": 5.0, "2024-02": 0.0}


@pytest.fixture
def tiny_catalog(monkeypatch):
    cat = pd.DataFrame([
        {"id": "m1", "name": "Grasberg", "aliases": "Grasberg mine;Ertsberg", "wikidata": "Q1"},
        {"id": "m2", "name": "Escondida", "aliases": None, "wikidata": "Q2"},
        {"id": "m3", "name": "Mine", "aliases": None, "wikidata": None},
    ]).set_index("id", drop=False)
    monkeypatch.setattr(collect.catalog, "load", lambda: cat)
    return cat


def test_mine_names_skips_generic_and_short(tiny_catalog):
    # « Grasberg mine » ne diffère de « Grasberg » que par « mine » : il ne prend pas une place
    assert collect.mine_names("m1", "Grasberg") == ["Grasberg", "Ertsberg"]
    assert collect.mine_names("m3", "Mine") == []
    assert collect.mine_names("liste:x", "Kamoa-Kakula") == ["Kamoa-Kakula"]


def _track(con, *sites):
    for s, name in sites:
        con.execute("INSERT INTO tracked(site_id, site_name, reason, since) VALUES (?,?,?,?)",
                    (s, name, "test", "2026-11-01T00:00:00Z"))


TIMELINE = {"timeline": [{"series": "Article Count", "data": [
    {"date": "20250301T000000Z", "value": 4}, {"date": "20250302T000000Z", "value": 1},
    {"date": "20250401T000000Z", "value": 7}]}]}
ARTLIST = {"articles": [
    {"url": "https://mining.com/grasberg-mudslide/", "title": "Grasberg mine mudslide &amp; halt",
     "seendate": "20250915T081500Z", "domain": "mining.com", "language": "English",
     "sourcecountry": "Canada"},
    {"url": "https://kompas.id/tambang-grasberg", "title": "Tambang Grasberg", "seendate": "20250916T000000Z",
     "domain": "kompas.id", "language": "Indonesian", "sourcecountry": "Indonesia"}]}


def test_doc_backfill_writes_volume_and_hinted_docs(con, tiny_catalog):
    _track(con, ("m1", "Grasberg"), ("m2", "Escondida"))
    # article déjà reçu par le flux, pas encore normalisé : il reçoit la mine visée
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, url, status) VALUES (?,?,?,?,'new')",
                (collect.doc_id("https://kompas.id/tambang-grasberg"), "gdelt_gkg_tr", "t",
                 "https://kompas.id/tambang-grasberg"))
    ft, seen = FakeTime(), []
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, json=TIMELINE), httpx.Response(200, json=ARTLIST)], seen, ft)
    out = collect.doc_backfill(con, c, NOW, max_queries=3, throttle=th)  # budget : une seule mine
    assert out["doc_queries"] == 2 and out["doc_mines"] == 1 and out["doc_articles"] == 1
    assert seen[0][1]["mode"] == "timelinevolraw" and seen[1][1]["mode"] == "artlist"
    assert seen[0][1]["query"] == '("Grasberg" OR "Ertsberg") mine'
    assert seen[1][1]["maxrecords"] == "250" and seen[1][1]["sort"] == "datedesc"
    assert seen[0][1]["startdatetime"] == "20170101000000"
    assert seen[1][0] - seen[0][0] >= 6  # métronome
    vol = dict(con.execute("SELECT month, n_articles FROM news_volume WHERE site_id='m1' "
                           "AND source='gdelt_doc_timeline'").fetchall())
    assert vol == {"2025-03": 5.0, "2025-04": 7.0}
    docs = {r["url"]: r for r in con.execute("SELECT * FROM raw_docs")}
    new = docs["https://mining.com/grasberg-mudslide/"]
    assert new["source"] == "gdelt_doc" and new["hint_site"] == "m1" and new["lang"] == "en"
    assert new["title"] == "Grasberg mine mudslide & halt" and new["published_at"] == "2025-09-15T08:15:00Z"
    assert docs["https://kompas.id/tambang-grasberg"]["hint_site"] == "m1"
    st = dict(con.execute("SELECT site_id, backfill_status FROM tracked").fetchall())
    assert st == {"m1": "fait", "m2": None}
    assert float(con.execute("SELECT value FROM meta WHERE key='doc_api_next_at'").fetchone()[0]) > 0
    # début de couverture : article le plus ancien renvoyé par artlist (même celui déjà reçu par le flux)
    oldest = con.execute("SELECT value FROM meta WHERE key='doc_oldest:m1'").fetchone()[0]
    assert oldest == "2025-09-15T08:15:00Z"


def test_doc_backfill_refused_and_rate_limited(con, tiny_catalog):
    _track(con, ("m1", "Grasberg"), ("m2", "Escondida"), ("m3", "Mine"))
    ft = FakeTime()
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, text="Your search contained a keyword that is too common."),
                     httpx.Response(429), httpx.Response(429)], [], ft)
    out = collect.doc_backfill(con, c, NOW, max_queries=10, throttle=th)
    st = dict(con.execute("SELECT site_id, backfill_status FROM tracked").fetchall())
    assert st["m1"].startswith("erreur : requête refusée")
    assert st["m2"] is None  # 429 persistant : retenté au prochain cycle
    assert st["m3"] == "erreur : aucun nom exploitable pour l'API"
    assert out["errors"] and out["doc_mines"] == 0


def test_doc_backfill_artlist_falls_back_to_default_window(con, tiny_catalog):
    _track(con, ("m2", "Escondida"))
    ft, seen = FakeTime(), []
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, json=TIMELINE), httpx.Response(200, text="Invalid date range."),
                     httpx.Response(200, json=ARTLIST)], seen, ft)
    out = collect.doc_backfill(con, c, NOW, max_queries=2, throttle=th)
    assert out["doc_queries"] == 3 and out["doc_mines"] == 1
    assert "startdatetime" in seen[1][1] and "startdatetime" not in seen[2][1]
    assert con.execute("SELECT backfill_status FROM tracked").fetchone()[0] == "fait"


def test_doc_backfill_respects_cross_process_deadline(con, tiny_catalog):
    _track(con, ("m1", "Grasberg"))
    ft, seen = FakeTime(), []
    con.execute("INSERT INTO meta(key, value) VALUES ('doc_api_next_at', ?)", (str(ft.t + 4),))
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, json=TIMELINE), httpx.Response(200, json={})], seen, ft)
    collect.doc_backfill(con, c, NOW, max_queries=2, throttle=th)
    assert ft.slept[0] == pytest.approx(4)


# -- noms multilingues -------------------------------------------------------------------------------------
def _sparql_handler(calls: list):
    def handler(req):
        calls.append(req)
        body = {"results": {"bindings": [
            {"x": {"value": "http://www.wikidata.org/entity/Q1"},
             "l": {"value": "Grasberg", "xml:lang": "fr"}},
            {"x": {"value": "http://www.wikidata.org/entity/Q1"},
             "l": {"value": "Grasberg", "xml:lang": "en"}},
            {"x": {"value": "http://www.wikidata.org/entity/Q1"},
             "l": {"value": "グラスベルグ鉱山", "xml:lang": "ja"}},
            {"x": {"value": "http://www.wikidata.org/entity/Q2"},
             "l": {"value": "Эскондида", "xml:lang": "ru"}},
            {"x": {"value": "http://www.wikidata.org/entity/Q2"}, "l": {"value": "12", "xml:lang": "en"}},
        ]}}
        return httpx.Response(200, json=body)
    return handler


def test_fetch_names_multilingual_weekly(con, tiny_catalog, tmp_path):
    calls: list = []
    c = httpx.Client(transport=httpx.MockTransport(_sparql_handler(calls)))
    path = tmp_path / "names.parquet"
    n = collect.fetch_names(con, c, path=path, now=NOW)
    assert n == 3 and len(calls) == 1 and calls[0].method == "POST"
    assert b"skos%3AaltLabel" in calls[0].content or b"skos:altLabel" in calls[0].content
    df = pd.read_parquet(path)
    assert list(df.columns) == ["site_id", "name", "lang"]
    rows = {(r.site_id, r.name): r.lang for r in df.itertuples()}
    assert rows == {("m1", "Grasberg"): "en", ("m1", "グラスベルグ鉱山"): "ja", ("m2", "Эскондида"): "ru"}
    # moins d'une semaine après : rien n'est redemandé
    assert collect.fetch_names(con, c, path=path, now=NOW + timedelta(days=6)) == 0 and len(calls) == 1
    assert collect.fetch_names(con, c, path=path, now=NOW + timedelta(days=8)) == 3 and len(calls) == 2
    assert collect.fetch_names(con, c, path=path, now=NOW + timedelta(days=8), force=True) == 3


def test_run_once_never_raises(con, monkeypatch):
    def handler(req):
        raise httpx.ConnectError("hors ligne")
    c = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(collect, "fetch_names", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("wd")))
    out = collect.run_once(con, backfill_hours=0.25, max_files=2, doc_queries=0, client=c, now=NOW)
    assert out["files_done"] == 0
    ps = con.execute("SELECT ok, message FROM pipeline_status WHERE stage='collecte'").fetchone()
    assert ps["ok"] == 0 and "noms" in ps["message"]


# -- corrections après revue ---------------------------------------------------------------------------------
def _bad_deflate_zip() -> bytes:
    """Zip au répertoire central intact mais au flux deflate abîmé : la lecture lève zlib.error."""
    raw = _zip(EN, "x.gkg.csv")
    z = zipfile.ZipFile(io.BytesIO(raw))
    info = z.infolist()[0]
    start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    buf = bytearray(raw)
    for k in range(start + 10, start + min(info.compress_size, 400)):
        buf[k] ^= 0x5A
    return bytes(buf)


def _empty_zip() -> bytes:
    buf = io.BytesIO()
    zipfile.ZipFile(buf, "w").close()
    return buf.getvalue()


@pytest.mark.parametrize("content", [_bad_deflate_zip(), _empty_zip()], ids=["deflate", "vide"])
def test_unreadable_zip_isolated_and_counted(con, content):
    # un fichier illisible (zlib.error, zip sans fichier) consomme un essai et n'arrête pas le lot
    routes = {
        "lastupdate.txt": httpx.Response(200, text=_lastupdate("20261126120000", ".gkg.csv.zip")),
        "lastupdate-translation.txt": httpx.Response(
            200, text=_lastupdate("20261126114500", ".translation.gkg.csv.zip")),
        "20261126120000.gkg.csv.zip": httpx.Response(200, content=content),
        "20261126114500.gkg.csv.zip": httpx.Response(200, content=_zip(EN, "a.gkg.csv")),
    }
    c = httpx.Client(transport=_feed_transport(routes, []))
    out = collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c, now=NOW,
                           names=False)
    bad = con.execute("SELECT status, tries, message FROM feed_files "
                      "WHERE name='20261126120000.gkg.csv.zip'").fetchone()
    assert (bad["status"], bad["tries"]) == ("pending", 1) and "illisible" in bad["message"]
    assert out["files_done"] == 1 and out["docs_new"] == 3
    # aucune insertion partielle du fichier abîmé : seuls les 3 documents du bon fichier
    assert con.execute("SELECT COUNT(*) FROM raw_docs").fetchone()[0] == 3


def test_offline_cycles_lose_nothing(con):
    # 6 cycles hors ligne (1 h 30) : aucun fichier abandonné, un seul essai de fichier par cycle,
    # puis tout est rattrapé au retour du réseau
    online = {"up": False}
    calls: list[str] = []

    def handler(req):
        url = str(req.url)
        if not online["up"]:
            calls.append(url)
            raise httpx.ConnectError("no route to host")
        if url.endswith("lastupdate.txt"):
            return httpx.Response(200, text=_lastupdate("20261126120000", ".gkg.csv.zip"))
        if url.endswith("lastupdate-translation.txt"):
            return httpx.Response(200, text=_lastupdate("20261126120000", ".translation.gkg.csv.zip"))
        return httpx.Response(200, content=_zip(EN, "a.gkg.csv"))

    c = httpx.Client(transport=httpx.MockTransport(handler))
    for k in range(6):
        calls.clear()
        out = collect.run_once(con, backfill_hours=1, max_files=10, doc_queries=0, client=c,
                               now=NOW + k * collect.STEP, names=False)
        assert out["files_net"] == 1 and sum("lastupdate" not in u for u in calls) == 1
    st = dict(con.execute("SELECT status, COUNT(*) FROM feed_files GROUP BY status").fetchall())
    assert set(st) == {"pending"}
    assert con.execute("SELECT MAX(tries) FROM feed_files").fetchone()[0] in (0, None)
    ps = con.execute("SELECT ok, message FROM pipeline_status WHERE stage='collecte'").fetchone()
    assert ps["ok"] == 0 and "réseau" in ps["message"]
    online["up"] = True
    out = collect.run_once(con, backfill_hours=1, max_files=100, doc_queries=0, client=c,
                           now=NOW + 6 * collect.STEP, names=False)
    assert out["files_net"] == 0
    assert set(dict(con.execute("SELECT status, 1 FROM feed_files").fetchall())) == {"done"}


def test_server_busy_respects_retry_after(con):
    calls: list[str] = []
    routes = {
        "lastupdate.txt": httpx.Response(200, text=_lastupdate("20261126120000", ".gkg.csv.zip")),
        "lastupdate-translation.txt": httpx.Response(
            200, text=_lastupdate("20261126120000", ".translation.gkg.csv.zip")),
    }

    def busy():
        return httpx.Response(503, headers={"Retry-After": "1200"})
    routes[".zip"] = busy
    c = httpx.Client(transport=_feed_transport(routes, calls))
    out = collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c, now=NOW,
                           names=False)
    assert out["files_net"] == 1 and sum(u.endswith(".zip") for u in calls) == 1
    assert con.execute("SELECT MAX(tries) FROM feed_files").fetchone()[0] in (0, None)
    hold = con.execute("SELECT value FROM meta WHERE key='feed_retry_at'").fetchone()[0]
    assert hold == "2026-11-26T12:27:00Z"  # 12:07 + Retry-After de 20 min
    calls.clear()  # 10 min plus tard : on attend encore
    collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c,
                     now=NOW + timedelta(minutes=10), names=False)
    assert not any(u.endswith(".zip") for u in calls)
    calls.clear()  # 25 min plus tard : reprise
    routes[".zip"] = httpx.Response(200, content=_zip(EN, "a.gkg.csv"))
    out = collect.run_once(con, backfill_hours=0.5, max_files=10, doc_queries=0, client=c,
                           now=NOW + timedelta(minutes=25), names=False)
    assert out["files_done"] >= 1


def test_plan_without_backfill_takes_latest(con):
    latest = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    assert collect.plan(con, "gkg", latest, backfill_hours=0) == 1
    assert con.execute("SELECT stamp FROM feed_files").fetchone()[0] == "20261126120000"
    assert collect.plan(con, "gkg", latest + collect.STEP, backfill_hours=0) == 1


def test_mine_names_rejects_countries_generic_numbers(monkeypatch):
    cat = pd.DataFrame([
        {"id": "a", "name": "El Callao", "aliases": "Laguna Mine Santa Rita;Peru", "wikidata": None},
        {"id": "b", "name": "Gold", "aliases": "Southern Mine;Mine 33;Mina Carbón;Central North Open Pit;"
                                              "Copper Mountain;Chinese Mine;Papua New Guinea Gold",
         "wikidata": None},
    ]).set_index("id", drop=False)
    monkeypatch.setattr(collect.catalog, "load", lambda: cat)
    assert collect.mine_names("a", "El Callao") == ["El Callao", "Laguna Mine Santa Rita"]
    assert collect.mine_names("b", "North") == []
    assert collect.mine_names("liste:x", "Mine No 3") == []
    assert collect.mine_names("liste:x", "South Africa") == []
    assert collect.mine_names("liste:x", "Oyu Tolgoi") == ["Oyu Tolgoi"]


@pytest.mark.parametrize(("name", "code"), [
    ("English", "en"), ("Maltese", "mt"), ("Kinyarwanda", "rw"), ("Sinhalese", "si"), ("Tibetan", "bo"),
    ("Chinese (Simplified)", "zh"), ("Cebuano", "ceb"), ("Klingon", None), (None, None), ("", None),
])
def test_doc_lang_iso_or_none(name, code):
    assert collect.doc_lang(name) == code
    docs = collect.artlist_docs({"articles": [{"url": "https://x.mt/a", "language": name}]}, "m", "q", "t")
    assert docs[0]["lang"] == code


def test_gkg_row_without_url_skipped():
    line = next(iter(EN.open(encoding="utf-8", newline="\n")))
    cols = line.rstrip("\r\n").split("\t")
    assert collect.parse_record(cols, "gkg", "t") is not None
    cols[4] = "  "
    assert collect.parse_record(cols, "gkg", "t") is None


def test_mine_names_same_core_takes_one_slot(monkeypatch):
    # « Inden coal Mine », « Inden surface mine » : même cœur que « Inden » → une seule place sur trois
    cat = pd.DataFrame([{"id": "i", "name": "Inden", "aliases": "Inden coal Mine;Inden surface mine;"
                         "Garzweiler;Hambach", "wikidata": None}]).set_index("id", drop=False)
    monkeypatch.setattr(collect.catalog, "load", lambda: cat)
    assert collect.mine_names("i", "Inden") == ["Inden", "Garzweiler", "Hambach"]


def test_doc_api_odd_json_is_no_result_not_a_crash():
    # une réponse JSON inattendue (liste, entrées non objets) ne fait pas tomber l'historique entier
    ft = FakeTime()
    th = collect.Throttle(6.0, clock=ft.clock, sleep=ft.sleep)
    c = _doc_client([httpx.Response(200, json=[1, 2])], [], ft)
    assert collect.doc_get(c, {"query": "x"}, th) == {}
    assert collect.timeline_months({"timeline": ["x", {"data": [3, {"date": "20240101", "value": 2}]}]}) \
        == {"2024-01": 2.0}
    docs = collect.artlist_docs({"articles": ["x", None, {"url": "https://a.b/c"}]}, "m", "q", "t")
    assert [d["url"] for d in docs] == ["https://a.b/c"]


def test_mine_names_keeps_full_place_name_of_major_mine(monkeypatch):
    # « Red Lake » : aucun mot distinctif, mais nom complet d'une grande mine → phrase exacte gardée ;
    # le même genre de nom en ALIAS (« Copper Mountain » d'une autre mine) reste refusé
    cat = pd.DataFrame([
        {"id": "rl", "name": "Red Lake", "aliases": "Red Lake Mine;Campbell", "wikidata": None},
        {"id": "o", "name": "Other", "aliases": "Copper Mountain;Southern Mine", "wikidata": None},
    ]).set_index("id", drop=False)
    monkeypatch.setattr(collect.catalog, "load", lambda: cat)
    assert collect.mine_names("rl", "Red Lake") == ["Red Lake", "Campbell"]
    assert collect.mine_query(["Red Lake"]) == '"Red Lake" mine'
    assert collect.mine_names("o", "Other") == ["Other"]
    assert collect.mine_names("liste:x", "South Africa") == []
    assert collect.mine_names("liste:x", "Big Hill") == ["Big Hill"]


def _one_file(con, stamp="20261126110000", tries=0):
    con.execute("INSERT INTO feed_files(name, feed, stamp, status, tries) VALUES (?, 'gkg', ?, 'pending', ?)",
                (stamp + ".gkg.csv.zip", stamp, tries))


def test_non_zip_200_is_network_not_file_fault(con):
    # un portail captif répond 200 + HTML à tout : aucun essai consommé, lot arrêté après MAX_NET_FAILS
    for k in range(5):
        _one_file(con, f"2026112611{k}000")
    c = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>Login to Wi-Fi</html>")))
    out = collect.collect_feeds(con, c, NOW, backfill_hours=0, max_files=10)
    assert out["files_net"] == collect.MAX_NET_FAILS and out["files_notzip"] == collect.MAX_NET_FAILS
    assert {r[0] for r in con.execute("SELECT tries FROM feed_files")} == {0}
    assert any("pas des zips" in e for e in out["errors"])


def test_old_non_zip_file_eventually_consumes_tries(con):
    _one_file(con, "20261120110000")                      # plus de 72 h : le fichier lui-même est en cause
    c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html></html>")))
    res = collect.fetch_file(con, c, con.execute("SELECT * FROM feed_files").fetchone(), NOW)
    assert res.get("unreadable") and not res.get("net")
    assert con.execute("SELECT tries FROM feed_files").fetchone()[0] == 1


def test_abandoned_unreadable_file_visible_in_status(con):
    _one_file(con, "20261126110000", tries=collect.MAX_TRIES - 1)
    routes = {"20261126110000.gkg.csv.zip": httpx.Response(200, content=b"PK\x03\x04trunc")}
    c = httpx.Client(transport=_feed_transport(routes, []))
    collect.run_once(con, backfill_hours=0, max_files=10, doc_queries=0, client=c, now=NOW, names=False)
    ps = con.execute("SELECT ok, message FROM pipeline_status WHERE stage='collecte'").fetchone()
    assert ps["ok"] == 0 and "20261126110000.gkg.csv.zip" in ps["message"] and "abandonné" in ps["message"]


def test_region_only_names_are_not_query_names(monkeypatch):
    monkeypatch.setattr(collect.catalog, "load", lambda: pd.DataFrame())
    assert collect.mine_names("liste:x", "Nevada") == []
    assert collect.mine_names("liste:x", "Katanga Mine") == []
    assert collect.mine_names("liste:x", "Norilsk Nickel") == ["Norilsk Nickel"]


def test_mine_names_keeps_country_word_name_of_major_mine(monkeypatch):
    # « Cobre Panama » (substance + pays) : nom complet gardé et cité en premier ; « Chile Copper »,
    # « Nevada Gold Mines » (tournures de titres) et les régions (« New Mexico », « Western Australia »)
    # restent refusés
    cat = pd.DataFrame([
        {"id": "cp", "name": "Cobre Panama", "aliases": "Cerro Petaquilla;Molejon;Cobre mine",
         "wikidata": None},
        {"id": "ch", "name": "Chile Copper", "aliases": "Panama Copper", "wikidata": None},
    ]).set_index("id", drop=False)
    monkeypatch.setattr(collect.catalog, "load", lambda: cat)
    assert collect.mine_names("cp", "Cobre Panama") == ["Cobre Panama", "Cerro Petaquilla", "Molejon"]
    assert collect.mine_query(collect.mine_names("cp", None)).startswith('("Cobre Panama" OR ')
    assert collect.mine_names("ch", "Chile Copper") == []
    for bad in ("Nevada Gold Mines", "New Mexico", "Western Australia", "New South Wales", "South Africa",
                "Papua New Guinea Gold", "San Salvador"):
        assert collect.mine_names("liste:x", bad) == [], bad
    assert collect.mine_names("liste:x", "New Denmark Colliery") == ["New Denmark Colliery"]


@pytest.mark.parametrize("name", ["Ural Mountains", "Lake Victoria", "Atacama Desert", "Kivu Lake",
                                  "Copper Chile", "Gold Peru", "Gold Canada", "Copper Arizona",
                                  "Coal Victoria", "Peru Copper"])
def test_region_plus_place_or_sector_word_is_not_a_query_name(monkeypatch, name):
    # pays ou région + relief (« Ural Mountains ») : un lieu ; + substance anglaise (« Copper Chile ») :
    # une tournure de titre. Aucun des deux ne désigne une mine, quel que soit l'ordre des mots.
    monkeypatch.setattr(collect.catalog, "load", lambda: pd.DataFrame())
    assert collect._place_name_key(name) is None
    assert collect.mine_names("liste:x", name) == []


def test_place_names_without_region_still_query_names(monkeypatch):
    monkeypatch.setattr(collect.catalog, "load", lambda: pd.DataFrame())
    # un mot d'ouvrage dans le nom rend la phrase exacte précise, même avec un pays ou une région
    for name in ("Red Lake", "Copper Mountain", "Big Hill", "Cobre Panama", "New Denmark Colliery",
                 "Turkey Creek Mine", "New Jersey Zinc Pit", "Atacama Desert Mine"):
        assert collect.mine_names("liste:x", name) == [name], name
