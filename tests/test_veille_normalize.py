"""Veille, étape 2 (normaliseur) : formats, doublons, rattachement aux mines, classement multilingue.

Petit catalogue synthétique + un échantillon réel GDELT réduit (tests/fixtures/veille_normalize_docs.json,
fichier du 25/11/2026). Aucun accès réseau.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from strate import catalog, entities
from strate.veille import normalize as N
from strate.veille import store

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parent / "fixtures" / "veille_normalize_docs.json"

# id, nom, alias, lat, lon, pays, emprise km², production publiée, ICMM
ROWS = [
    ("c_esc", "Escondida", "Escondida Mine;La Escondida", -24.26, -69.08, "Chile", 30.0, True, "x"),
    ("c_esc2", "La Escondida", None, -27.94, -70.22, "Chile", None, False, None),
    ("c_esc3", "La Escondida", None, 27.67, -101.44, "Mexico", None, False, "x"),
    ("c_lb", "Los Bronces", None, -33.15, -70.28, "Chile", 25.0, True, "x"),
    ("c_lb2", "Los Bronces", None, -27.00, -70.00, "Chile", None, False, None),
    ("c_vic_au", "Victoria Mine", None, -37.50, 144.00, "Australia", None, False, "x"),
    ("c_vic_de", "Victoria Mine", None, 51.50, 7.40, "Germany", None, False, "x"),
    ("c_isa", "Mount Isa Mines", "Mt Isa Copper", -20.73, 139.48, "Australia", 40.0, True, "x"),
    ("c_eh", "Ernest Henry and Mount Isa", None, -19.34, 146.85, "Australia", None, False, "x"),
    ("c_gold", "Gold Mine", None, 40.0, -110.0, "United States", None, False, "x"),
    ("c_no2", "Mine No 2", None, 41.0, -111.0, "United States", None, False, "x"),
    ("c_felix", "Felix", None, 3.88, -53.37, "French Guiana", None, False, "x"),
    ("c_states", "States Mine", None, 39.0, -105.0, "United States", None, False, None),
    ("c_fek", "Fekola", None, 12.60, -11.35, "Mali", 8.0, True, "x"),
    ("c_kol", "Kolubara", None, 44.47, 20.23, "Serbia", 20.0, False, "x"),
    ("c_bf", "Black Friday Mines", None, 38.0, -107.0, "United States", None, False, None),
    ("c_rasp", "Распадская", "Raspadskaya", 53.80, 87.90, "Russia", 12.0, True, "x"),
    ("c_zijin", "Zijinshan", None, 25.18, 116.40, "China", 9.0, True, "x"),
    ("c_pong", "Pongkor and Cibaliung", None, -6.65, 106.55, "Indonesia", 2.0, False, "x"),
    ("c_timm", "Timmins West", None, 48.47, -81.33, "Canada", 3.0, False, "x"),
    ("c_peru", "Mina Peru", "Peru", -12.0, -76.0, "Peru", None, False, "x"),
    ("c_dup1", "Kittila", None, 67.90, 25.39, "Finland", 5.0, True, "x"),
    ("c_dup2", "Kittila", None, 67.92, 25.40, "Finland", None, False, None),    # même mine, 2 km
    ("c_big", "Ballarat East", ";".join([f"Telegraph{i} Co." for i in range(30)] + ["Ballarat"]),
     -37.58, 143.86, "Australia", 1.0, False, "x"),
    ("c_gs", "Greenstone Mine", None, 49.68, -86.93, "Canada", 15.0, True, "x"),
]


def _cat() -> pd.DataFrame:
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame(ROWS, columns=cols)
    df["wikidata"] = None
    df["wikipedia"] = None
    return df


def _watch():
    return [entities.Site(id="greenstone", name="Greenstone", country="Canada", region="Ontario",
                          commodity=["gold"], lat=49.677, lon=-86.928, radius_m=4000, owners=[],
                          stage="production", coord_source="test")]


@pytest.fixture(scope="module")
def idx():
    names = pd.DataFrame({"site_id": ["c_zijin", "c_rasp", "c_gold"],
                          "name": ["紫金山金铜矿", "Распадская шахта", "金矿"], "lang": ["zh", "ru", "zh"]})
    return N.build_index(_cat(), _watch(), names, twins={"greenstone": "c_gs"})


def _doc(title, locs=(), themes=(), orgs=(), persons=(), lang="en", hint=None):
    return {"id": "d", "title": title, "locations": list(locs), "themes": list(themes),
            "organizations": list(orgs), "persons": list(persons), "lang": lang, "hint_site": hint}


def _loc(name, lat, lon, typ=4, country=None, cc=None):
    return {"name": name, "lat": lat, "lon": lon, "type": typ,
            "country": country or name.rsplit(",", 1)[-1].strip(), "cc": cc}


def _sites(matches):
    return {m["site_id"]: m for m in matches}


# -- clés de noms -----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Gold Mine", "Mine No 2", "North Pit", "Peru", "Mina Colombia", "Ok Mine"])
def test_generic_names_are_not_indexed(name):
    assert N.word_key(name) is None


def test_name_keys_drop_mine_words_and_keep_numbers():
    assert N.word_key("Mount Isa Mines") == ("mount", "isa")
    assert N.word_key("Kopalnia Bogdanka") == ("bogdanka",)
    assert N.word_key("Neyveli II Mine") == ("neyveli", "2")
    assert N.word_key("Шахта Распадская") == ("распадская",)


def test_cjk_keys():
    assert N.cjk_key("金矿") is None              # « mine d'or » : rien de distinctif
    assert N.cjk_key("第一煤矿") is None
    key, weak = N.cjk_key("紫金山金铜矿")
    assert key.startswith("紫金山") and not weak


# -- index ------------------------------------------------------------------------------------------
def test_index_twin_merge_clusters_and_filters(idx):
    assert idx.words[("greenstone",)] == ("greenstone",)        # jumelle du catalogue → id suivi
    assert "c_gs" not in idx.sites
    assert idx.words[("kittila",)] == ("c_dup1",)               # deux fiches à 2 km : une seule mine
    assert ("states",) not in idx.words                         # mot seul, fiche non étayée
    assert ("gold",) not in idx.words and ("peru",) not in idx.words
    assert ("telegraph0",) not in idx.words                     # inventaire d'alias ignoré
    assert ("ballarat",) in idx.words                           # … sauf les alias proches du nom
    assert idx.words[("mount", "isa")] == ("c_isa",)            # nom complet prioritaire sur un morceau
    assert idx.words[("pongkor",)] == ("c_pong",)               # morceau de « Pongkor and Cibaliung »
    assert set(idx.words[("escondida",)]) == {"c_esc", "c_esc3"}   # c_esc2 : fiche non étayée


def test_index_is_fast_and_matching_does_not_scan_sites():
    n = 20_000
    cat = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "name": [f"Zorbat{i:05d}x Mine" for i in range(n)],
                        "aliases": None, "lat": 10.0, "lon": [i / 100 for i in range(n)], "country": "Chile",
                        "footprint_km2": 1.0, "has_production": False, "icmm": "x"})
    t = time.perf_counter()
    big = N.build_index(cat)
    assert time.perf_counter() - t < 10
    doc = _doc("Strike at Zorbat12345x copper mine", [_loc("Chile", -30, -71, typ=1)])
    t = time.perf_counter()
    for _ in range(200):
        m, _ = N.match(doc, big)
    assert (time.perf_counter() - t) / 200 < 0.005
    assert [x["site_id"] for x in m] == ["s12345"]


# -- rattachement -----------------------------------------------------------------------------------
def test_homonyms_resolved_by_place_then_country_only(idx):
    m, _ = N.match(_doc("Strike at BHP's Escondida copper mine", [_loc("Chile", -30, -71, typ=1)]), idx)
    assert list(_sites(m)) == ["c_esc"] and m[0]["method"] == "mot+pays"
    m, _ = N.match(_doc("Huelga en la mina La Escondida", [_loc("Coahuila, Mexico", 27.7, -101.4)],
                        lang="es"), idx)
    assert list(_sites(m)) == ["c_esc3"] and m[0]["method"] == "mot+lieu"
    # deux « Los Bronces » au Chili : ni lieu proche ni pays ne tranchent → aucun rattachement
    # (spécification), même si l'une des deux est la grande mine (pas de préférence « à la notoriété »)
    chile = [_loc("Chile", -30, -71, typ=1)]
    m, st = N.match(_doc("Anglo American halts Los Bronces copper mine", chile), idx)
    assert m == [] and st["ambigus"] == 1
    m, st = N.match(_doc("Los Bronces mine expansion approved"), idx)       # article sans lieu
    assert m == [] and st["ambigus"] == 1
    near = [_loc("Lo Barnechea, Chile", -33.2, -70.3)]
    m, _ = N.match(_doc("Anglo American halts Los Bronces copper mine", near), idx)
    assert list(_sites(m)) == ["c_lb"] and m[0]["method"] == "nom+lieu"


def test_common_single_word_needs_nearby_place(idx):
    near = _loc("Ballarat, Victoria, Australia", -37.55, 143.85)
    m, _ = N.match(_doc("Victoria gold mine reopens", [_loc("Australia", -25, 135, typ=1)]), idx)
    assert m == []                                         # « Victoria » + pays seulement : refusé
    m, _ = N.match(_doc("Victoria gold mine reopens", [_loc("Bendigo, Victoria, Australia", -37.3, 144.2)]),
                   idx)
    assert list(_sites(m)) == ["c_vic_au"]
    m, _ = N.match(_doc("Ballarat gold mine expands", [near]), idx)
    assert list(_sites(m)) == ["c_big"]


def test_full_name_in_title_without_place(idx):
    m, _ = N.match(_doc("Glencore to close Mount Isa copper mine"), idx)
    assert list(_sites(m)) == ["c_isa"] and m[0]["method"] == "nom"


def test_person_and_generic_words_do_not_match(idx):
    m, st = N.match(_doc("Felix visits gold mine", persons=["Felix Tshisekedi"]), idx)
    assert m == [] and st["ecartes"] == 1
    m, _ = N.match(_doc("United States gold mine output rises", [_loc("United States", 39, -98, typ=1)],
                        orgs=["United States"]), idx)
    assert m == []
    m, _ = N.match(_doc("Best Black Friday deals on TVs", [_loc("United States", 39, -98, typ=1)]), idx)
    assert m == []                                         # nom obscur, article sans rapport avec les mines
    m, _ = N.match(_doc("Black Friday mine reopens after flood"), idx)
    assert m == []                                         # « Friday » : mot courant, lieu exigé
    m, _ = N.match(_doc("Black Friday mine reopens after flood", [_loc("Gunnison, Colorado, United States",
                                                                       38.1, -107.1)]), idx)
    assert list(_sites(m)) == ["c_bf"]


def test_single_word_needs_mining_title(idx):
    mali = [_loc("Kenieba, Mali", 12.84, -11.24)]
    m, _ = N.match(_doc("Fekola festival opens", mali), idx)
    assert m == []
    m, _ = N.match(_doc("Gouvernance minière : le contrôle parlementaire à Fekola", mali, lang="fr"), idx)
    assert list(_sites(m)) == ["c_fek"] and m[0]["method"] == "mot+lieu"


def test_unit_number_separates(idx):
    m, _ = N.match(_doc("Kolubara No 2 mine accident", [_loc("Serbia", 44, 21, typ=1)]), idx)
    assert m == []
    m, _ = N.match(_doc("Kolubara mine accident", [_loc("Serbia", 44, 21, typ=1)]), idx)
    assert list(_sites(m)) == ["c_kol"]


def test_place_name_far_from_mine_is_not_the_mine(idx):
    far = _loc("Timmins West, Somewhere", 10.0, 10.0)
    m, _ = N.match(_doc("Timmins West gold mine news", [far]), idx)
    assert m == []
    m, _ = N.match(_doc("Timmins West gold mine news", [_loc("Timmins West, Ontario, Canada", 48.48, -81.3)]),
                   idx)
    assert list(_sites(m)) == ["c_timm"]


def test_cyrillic_and_cjk_titles(idx):
    m, _ = N.match(_doc("Авария на шахте Распадская", [_loc("Russia", 60, 100, typ=1, country="Russian",
                                                             cc="RS")], lang="ru"), idx)
    assert list(_sites(m)) == ["c_rasp"] and m[0]["method"] == "mot+pays"
    m, _ = N.match(_doc("紫金山金铜矿发生透水事故", [_loc("China", 35, 105, typ=1)], lang="zh"), idx)
    assert list(_sites(m)) == ["c_zijin"]


def test_hint_site(idx):
    m, _ = N.match(_doc("Quarterly report", hint="c_fek"), idx)
    assert m[0]["site_id"] == "c_fek" and m[0]["method"] == "requête" and m[0]["strength"] == 0.5
    # nom présent mais homonymes non tranchés : la requête par mine arbitre
    m, _ = N.match(_doc("Los Bronces mine", hint="c_lb2"), idx)
    assert _sites(m)["c_lb2"]["method"] == "requête+nom"
    m, _ = N.match(_doc("Greenstone gold mine update", hint="c_gs"), idx)   # id de la jumelle
    assert list(_sites(m)) == ["greenstone"]


def test_fips_country_codes():
    assert N.loc_country({"country": "Russian", "cc": "RS"}) == "Russian Federation"
    assert N.loc_country({"country": "Australia", "cc": "AS"}) == "Australia"
    assert N.loc_country({"country": "Dutch", "cc": "NL"}) == "Netherlands"
    assert N.loc_country({"country": "South African", "cc": "SF"}) == "South Africa"


# -- classement -------------------------------------------------------------------------------------
@pytest.mark.parametrize("title,lang,expected", [
    ("Grève illimitée à la mine de Fekola", "fr", "social"),
    ("Derrumbe en mina deja tres muertos", "es", "accident"),
    ("Rompimento de barragem deixa mortos", "pt", "accident"),
    ("Vazamento de rejeitos contamina rio", "pt", "environnement"),
    ("Übernahme des Bergwerks durch Glencore", "de", "finance"),
    ("Grubenunglück: zwei Bergleute verschüttet", "de", "accident"),
    ("Забастовка шахтёров на руднике", "ru", "social"),
    ("紫金矿业收购一座金矿", "zh", "finance"),
    ("鉱山で落盤事故", "ja", "accident"),
    ("광산 노조 파업 돌입", "ko", "social"),
    ("Longsor di tambang emas, lima penambang tewas", "id", "accident"),
    ("Maden ocağında grev", "tr", "social"),
    ("Strajk w kopalni Bogdanka", "pl", "social"),
    ("انفجار في منجم للفحم", "ar", "accident"),
    ("Sciopero alla miniera", "it", "social"),
    ("Mine output rises as expansion completes", "en", "operations"),
    ("Gunmen attack convoy near gold mine", "en", "securite"),
    ("Court suspends mining permit", "en", "juridique"),
])
def test_multilingual_keywords(title, lang, expected):
    typ, strength, detail = N.classify(title, [], lang)
    assert typ == expected, detail
    assert strength == 0.9


# les articles de l'API documentaire n'ont pas de thèmes : « mourir » doit suffire dans chaque langue
@pytest.mark.parametrize("title,lang", [
    ("Two workers die at Kibali mine", "en"), ("Miner dies in Kibali mine", "en"),
    ("Worker died at Kibali mine", "en"), ("Un mineur meurt dans la mine de Fekola", "fr"),
    ("Dos mineros mueren en la mina", "es"), ("Minero murió en derrumbe", "es"),
    ("Trabalhador morre em mina", "pt"), ("Bergmann stirbt in Grube", "de"),
    ("Zwei Kumpel kamen ums Leben", "de"), ("Minatore muore in miniera", "it"),
    ("Górnik zmarł w kopalni", "pl"), ("Шахтёр умер на руднике", "ru"),
    ("Madenci hayatını kaybetti", "tr"), ("Penambang meninggal di tambang emas", "id"),
    ("توفي عامل في منجم", "ar"), ("煤矿工人丧生", "zh"), ("鉱山で作業員が亡くなる", "ja"),
    ("광산 노동자 숨진 채 발견", "ko"),
])
def test_dying_verbs_are_accidents(title, lang):
    assert N.classify(title, [], lang)[0] == "accident", title


def test_dying_keywords_avoid_false_senses():
    assert N.classify("Murder trial over mine dispute: meurtre", [], "fr")[0] != "accident"
    assert N.classify("Maden şirketi halka açık oldu", [], "tr")[0] != "accident"   # « oldu » : devenu


def test_classify_themes_and_false_friends():
    assert N.classify("Mine news", ["STRIKE", "PROTEST"], "en")[0] == "social"
    typ, s, _ = N.classify("Mine news", ["LEGISLATION"], "en")
    assert (typ, s) == ("autre", 0.7)                         # un thème fréquent ne suffit pas
    assert N.classify("M2.5 earthquake strikes near the mine", [], "en")[0] != "social"
    assert N.classify("Lực bán gia tăng", [], "vi")[0] == "autre"   # pas de dictionnaire : thèmes seuls
    assert N.classify("Dünya düştü", [], "tr")[0] == "autre"
    assert N.classify("Grève à la mine", ["STRIKE"], "fr")[1] == 1.0  # mot-clé + thème


# -- formats ----------------------------------------------------------------------------------------
def _raw(**kw):
    r = {"id": "x1", "source": "gdelt_gkg_tr", "url": "https://news.example.com/a?id=1",
         "domain": "news.example.com", "lang": "fr", "title": "Grève &amp; mine",
         "published_at": "2026-11-25T10:00:00Z", "themes": "STRIKE,12;ENV_MINING,40",
         "locations": json.dumps([{"name": "Mali", "lat": 17, "lon": -4, "type": 1}]),
         "organizations": "a;b", "persons": None, "hint_site": None}
    r.update(kw)
    return r


@pytest.mark.parametrize("kw,reason", [
    ({"url": "ftp://x.com/a"}, "URL invalide"), ({"url": "https://localhost/a"}, "URL invalide"),
    ({"url": "not a url"}, "URL invalide"), ({"published_at": None}, "date absente"),
    ({"published_at": "hier"}, "date illisible"), ({"published_at": "2027-06-01T00:00:00Z"}, "date future"),
    ({"published_at": "2014-12-31T00:00:00Z"}, "date avant 2015"), ({"title": "   "}, "titre vide"),
    ({"lang": "français"}, "langue non ISO"), ({"lang": None}, "langue absente"),
    ({"locations": "[{"}, "lieux illisibles"),
    ({"locations": json.dumps([{"lat": 95, "lon": 0}])}, "lat/lon hors bornes"),
    ({"locations": json.dumps([{"lat": 10, "lon": None}])}, "lat/lon hors bornes"),
])
def test_validation_rejects(kw, reason):
    doc, why = N.validate(_raw(**kw), NOW)
    assert doc is None and why == reason


def test_validation_normalises():
    raw = _raw(published_at="20261125101500", lang="fra", url="https://www.Example.com/x")
    doc, why = N.validate(raw, NOW)
    assert why is None
    assert doc["published_at"] == "2026-11-25T10:15:00Z" and doc["lang"] == "fr"
    assert doc["title"] == "Grève & mine" and doc["themes"] == ["STRIKE", "ENV_MINING"]
    assert N.validate(_raw(lang=None, source="gdelt_gkg"), NOW)[0]["lang"] == "en"   # flux anglais
    assert N.canon_url("https://www.x.com/a/?utm_source=t&id=2#top") == N.canon_url("http://x.com/a?id=2")


# -- étape complète ---------------------------------------------------------------------------------
@pytest.fixture
def con(tmp_path):
    c = store.init(tmp_path / "veille.db")
    yield c
    c.close()


def _insert(con, **kw):
    r = _raw(**kw)
    r.setdefault("fetched_at", "2026-11-25T12:00:00Z")
    cols = ["id", "source", "fetched_at", "published_at", "url", "domain", "lang", "title", "themes",
            "locations", "organizations", "persons", "hint_site"]
    con.execute(f"INSERT INTO raw_docs({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [r.get(c) for c in cols])


def test_run_once_end_to_end(con, idx):
    chile = json.dumps([{"name": "Chile", "lat": -30, "lon": -71, "type": 1, "country": "Chile"}])
    _insert(con, id="a", title="Grève à la mine Escondida et à Mount Isa", locations=chile,
            url="https://a.com/1")
    _insert(con, id="b", title="Grève à la mine Escondida et à Mount Isa", locations=chile,
            url="https://a.com/2", published_at="2026-11-26T09:00:00Z")          # même titre, 1 jour après
    _insert(con, id="c", url="https://a.com/1?utm_source=x", title="Autre titre sur une mine quelconque")
    _insert(con, id="d", url="ftp://bad", title="x")
    _insert(con, id="e", title="Marchés : le cuivre recule encore cette semaine", url="https://b.com/9")
    out = N.run_once(con, index=idx, now=NOW)
    st = dict(con.execute("SELECT id, status FROM raw_docs").fetchall())
    assert st == {"a": "done", "b": "rejected", "c": "rejected", "d": "rejected", "e": "done"}
    reasons = dict(con.execute("SELECT id, reject_reason FROM raw_docs WHERE status='rejected'").fetchall())
    assert reasons == {"b": "doublon de a", "c": "doublon de a", "d": "URL invalide"}
    ev = con.execute("SELECT * FROM events ORDER BY site_id").fetchall()
    assert [e["site_id"] for e in ev] == ["c_esc", "c_isa"]                    # une ligne par mine
    e = dict(ev[0])
    assert e["event_type"] == "social" and e["month"] == "2026-11" and e["country"] == "Chile"
    proof = json.loads(e["evidence"])
    assert proof["type"]["retenu"] == "social" and "greve" in proof["type"]["mots"]
    assert e["confidence"] == pytest.approx(proof["force_rattachement"] * proof["force_classement"], abs=1e-3)
    assert out["lus"] == 5 and out["doublons"] == 2 and out["rejetes"] == 1 and out["sans_mine"] == 1
    assert out["evenements"] == 2
    row = con.execute("SELECT ok, counters FROM pipeline_status WHERE stage='normalisation'").fetchone()
    assert row["ok"] == 1 and json.loads(row["counters"])["evenements"] == 2
    # idempotent : rien de neuf, rien de refait
    assert N.run_once(con, index=idx, now=NOW)["lus"] == 0
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2


def test_dedup_against_already_processed_docs(con, idx):
    _insert(con, id="old", title="Fekola : la production d'or progresse fortement", url="https://m.ml/1")
    N.run_once(con, index=idx, now=NOW)
    _insert(con, id="new", title="Fekola : la production d'or progresse fortement", url="https://m.ml/2",
            published_at="2026-11-25T18:00:00Z")
    _insert(con, id="later", title="Fekola : la production d'or progresse fortement", url="https://m.ml/3",
            published_at="2026-11-20T10:00:00Z")   # 5 jours avant : hors fenêtre, ce n'est pas un doublon
    out = N.run_once(con, index=idx, now=NOW)
    assert out["doublons"] == 1
    assert con.execute("SELECT status FROM raw_docs WHERE id='later'").fetchone()[0] == "done"


def test_hint_transferred_from_duplicate_in_same_batch(con, idx):
    _insert(con, id="g", title="Quarterly update from the operator team", url="https://q.com/1")
    _insert(con, id="h", title="Quarterly update from the operator team", url="https://q.com/2",
            source="gdelt_doc", hint_site="c_fek", published_at="2026-11-25T11:00:00Z")
    N.run_once(con, index=idx, now=NOW)
    ev = con.execute("SELECT doc_id, site_id, match_method FROM events").fetchall()
    assert [tuple(e) for e in ev] == [("g", "c_fek", "requête")]


def test_empty_catalogue_leaves_docs_pending(con):
    _insert(con, id="z")
    out = N.run_once(con, index=N.NameIndex(), now=NOW)
    assert out["lus"] == 0
    assert con.execute("SELECT status FROM raw_docs WHERE id='z'").fetchone()[0] == "new"
    assert con.execute("SELECT ok FROM pipeline_status WHERE stage='normalisation'").fetchone()[0] == 0


def test_real_gdelt_sample(con, idx):
    """Échantillon réel : Pongkor (indonésien) et Fekola (français) rattachés, bruit anglais ignoré."""
    docs = json.loads(FIXTURE.read_text("utf-8"))
    cols = ["id", "source", "fetched_at", "published_at", "url", "domain", "lang", "title", "themes",
            "locations", "organizations", "persons", "tone", "hint_site", "payload"]
    for d in docs:
        con.execute(f"INSERT INTO raw_docs({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [d.get(c) for c in cols])
    out = N.run_once(con, index=idx, now=NOW)
    ev = con.execute("SELECT site_id, event_type, lang FROM events ORDER BY lang, title").fetchall()
    got = {(e["site_id"], e["event_type"], e["lang"]) for e in ev}
    assert ("c_pong", "accident", "id") in got
    assert ("c_fek", "juridique", "fr") in got
    assert {e["site_id"] for e in ev} <= {"c_pong", "c_fek"}
    assert out["lus"] == len(docs)


# -- cache de l'index -------------------------------------------------------------------------------
def test_get_index_cached_and_rebuilt_on_change(tmp_path, monkeypatch):
    out_dir = tmp_path / "catalog"
    out_dir.mkdir()
    _cat().to_parquet(out_dir / "mines.parquet")
    names = tmp_path / "names.parquet"
    monkeypatch.setattr(catalog, "OUT", out_dir)
    monkeypatch.setattr(N, "NAMES_PATH", names)
    N._cache.clear()
    try:
        a = N.get_index()
        assert N.get_index() is a and ("fekola",) in a.words
        pd.DataFrame({"site_id": ["c_kol"], "name": ["Колубара"], "lang": ["sr"]}).to_parquet(names)
        b = N.get_index()
        assert b is not a and ("колубара",) in b.words
    finally:
        N._cache.clear()
        catalog.load.cache_clear()
        entities.load.cache_clear()


# -- corrections après revue ------------------------------------------------------------------------
EXTRA = [
    ("x_amg", "American Gold Mine", None, 40.5, -112.0, "United States", 1.0, True, "x"),
    ("x_icc", "Indian Copper Complex", None, 22.6, 86.4, "India", 2.0, True, "x"),
    ("x_cc", "Canadian Coal Company Mine", None, 49.0, -114.0, "Canada", 1.0, False, "x"),
    ("x_cm", "Canadian Malartic", None, 48.13, -78.13, "Canada", 10.0, True, "x"),
    ("x_state", "State Coal Mine", None, -38.6, 145.6, "Australia", 1.0, False, "x"),
    ("x_bank", "Bank", None, -26.2, 27.9, "South Africa", 1.0, True, "x"),
    ("x_june", "June Mine", None, 39.5, -105.5, "United States", 1.0, False, "x"),
]


@pytest.fixture(scope="module")
def idx2():
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame(ROWS + EXTRA, columns=cols)
    df["wikidata"] = None
    df["wikipedia"] = None
    return N.build_index(df, _watch(), None, twins={"greenstone": "c_gs"})


@pytest.mark.parametrize("name", ["American Gold Mine", "Indian Copper Complex", "Canadian Coal Company Mine",
                                  "Chinese Silver", "Mina Mexicana de Cobre", "Australisches Gold"])
def test_demonyms_and_commodities_are_not_distinctive(name):
    assert N.word_key(name) is None


def test_demonym_with_distinctive_word_still_indexed(idx2):
    assert N.word_key("Canadian Malartic") == ("canadian", "malartic")
    m, _ = N.match(_doc("Canadian Malartic gold mine output rises", [_loc("Canada", 60, -95, typ=1)]), idx2)
    # un seul mot distinctif (+ nationalité) : traité comme un nom d'un mot, pays seul → sous 0,5
    assert list(_sites(m)) == ["x_cm"] and m[0]["method"] == "mot+pays" and m[0]["strength"] < 0.5
    m, _ = N.match(_doc("Canadian Malartic gold mine output rises", [_loc("Malartic, Quebec, Canada", 48.14,
                                                                          -78.13)]), idx2)
    assert m[0]["method"] == "mot+lieu" and m[0]["strength"] * 0.7 >= 0.5


@pytest.mark.parametrize("title,locs", [
    ("American gold coin sales surge in November", [_loc("United States", 39.8, -98.5, typ=1)]),
    ("American silver eagle demand slows", []),
    ("Indian copper demand to double by 2030", []),
    ("Canadian coal exports to Asia fall", [_loc("Canada", 60, -95, typ=1)]),
    ("June gold output falls", [_loc("United States", 39.8, -98.5, typ=1)]),
    ("State coal mining agency announces audit", [_loc("Victoria, Australia", -37, 144, typ=2)]),
    ("South African bank raises gold mining forecast", [_loc("Johannesburg, South Africa", -26.2, 28.0)]),
])
def test_headline_false_positives_do_not_match(idx2, title, locs):
    m, _ = N.match(_doc(title, locs, themes=["ENV_MINING"]), idx2)
    assert m == [], m


def test_common_multiword_name_needs_nearby_place(idx2):
    m, _ = N.match(_doc("State Coal Mine reopens", [_loc("Australia", -25, 135, typ=1)]), idx2)
    assert m == []
    near = [_loc("Wonthaggi, Victoria, Australia", -38.6, 145.6)]
    m, _ = N.match(_doc("State Coal Mine reopens", near), idx2)
    assert list(_sites(m)) == ["x_state"] and m[0]["method"] == "mot+lieu"   # « State » + substance


def test_lowercase_word_is_not_a_proper_name(idx2):
    joburg = [_loc("Johannesburg, South Africa", -26.2, 28.0)]
    m, st = N.match(_doc("Strike at the bank mine near Johannesburg", joburg), idx2)
    assert m == [] and st["ecartes"] == 1
    m, _ = N.match(_doc("Strike at Bank mine near Johannesburg", joburg), idx2)
    assert list(_sites(m)) == ["x_bank"]
    m, _ = N.match(_doc("strike at bank mine near johannesburg", joburg), idx2)   # titre tout en minuscules
    assert list(_sites(m)) == ["x_bank"]


def test_single_word_needs_nearby_place_to_reach_threshold(idx):
    best = 1.0                                                    # classement le plus fort possible
    for locs in ([], [_loc("Mali", 17, -4, typ=1)]):
        m, _ = N.match(_doc("Fekola gold mine strike", locs), idx)
        assert m and m[0]["strength"] * best < 0.5, m
    m, _ = N.match(_doc("Fekola gold mine strike", [_loc("Kenieba, Mali", 12.84, -11.24)]), idx)
    assert m[0]["method"] == "mot+lieu" and m[0]["strength"] * 0.7 >= 0.5
    m, _ = N.match(_doc("Glencore to close Mount Isa copper mine"), idx)
    assert m[0]["strength"] * 0.7 >= 0.5                          # nom complet de plusieurs mots


@pytest.mark.parametrize("title", ["Kolubara 2 mine flooded", "Kolubara II mine flooded",
                                   "Kolubara #2 mine flooded", "Kolubara Unit 2 mine flooded",
                                   "Kolubara-B mine flooded", "Kolubara No. 2 mine flooded",
                                   "Kolubara shaft 3 flooded"])
def test_unit_notations_separate(idx, title):
    m, _ = N.match(_doc(title, [_loc("Serbia", 44, 21, typ=1)]), idx)
    assert m == []


@pytest.mark.parametrize("title", ["Kolubara 2026 coal output guidance", "Kolubara 10% coal output rise",
                                   "Kolubara 2.5 Mt coal output"])
def test_years_percentages_are_not_units(idx, title):
    m, _ = N.match(_doc(title, [_loc("Serbia", 44, 21, typ=1)]), idx)
    assert list(_sites(m)) == ["c_kol"]


def test_classify_title_keyword_beats_theme_only_tie():
    typ, _, detail = N.classify("Escondida copper production rises 10%",
                                ["KILL", "CRISISLEX_T03_DEAD", "WOUND", "MANMADE_DISASTER_IMPLIED"], "en")
    assert typ == "operations" and detail["scores"]["accident"] == detail["scores"]["operations"]


@pytest.mark.parametrize("url", ["http://a..com/x", "https://.com/x", "https://a.com:99999/x",
                                 "https://1.2.3.4/x", "https://a-.com/x"])
def test_validation_rejects_malformed_hosts(url):
    assert N.validate(_raw(url=url), NOW) == (None, "URL invalide")


def test_language_codes():
    assert N.norm_lang("xyz") is None
    assert N.norm_lang("Vietnamese") == "vi" and N.norm_lang("tib") == "bo" and N.norm_lang("ceb") == "ceb"
    assert N.validate(_raw(lang="xyz"), NOW) == (None, "langue non ISO")
    assert N.validate(_raw(url="https://xn--bcher-kva.de/a"), NOW)[1] is None


def test_watchlist_only_index_leaves_docs_pending(con):
    _insert(con, id="z")
    out = N.run_once(con, index=N.build_index(None, _watch()), now=NOW)
    assert out["lus"] == 0
    assert con.execute("SELECT status FROM raw_docs WHERE id='z'").fetchone()[0] == "new"


def test_hint_kept_for_url_duplicate_in_same_batch(con, idx):
    _insert(con, id="a", url="https://www.mining.com/kol-news/", domain="www.mining.com",
            title="Quarterly coal update from the operator", source="gdelt_gkg")
    _insert(con, id="b", url="https://mining.com/kol-news?utm_source=x", domain="mining.com",
            title="Quarterly coal update from the operator", source="gdelt_doc", hint_site="c_kol",
            published_at="2026-11-25T11:00:00Z")
    _insert(con, id="c", url="http://mining.com/kol-news", domain="mining.com",
            title="Quarterly coal update from the operator", source="gdelt_doc", hint_site="c_fek",
            published_at="2026-11-25T12:00:00Z")
    N.run_once(con, index=idx, now=NOW)
    ev = {tuple(e) for e in con.execute("SELECT doc_id, site_id, match_method FROM events")}
    assert ev == {("a", "c_kol", "requête"), ("a", "c_fek", "requête")}     # deux requêtes, aucune perdue


def test_hint_applied_to_already_processed_original(con, idx):
    _insert(con, id="a", url="https://www.mining.com/fekola-news/", domain="www.mining.com",
            title="Fekola : la mine d'or accélère sa production", source="gdelt_gkg")
    N.run_once(con, index=idx, now=NOW)
    assert {r[0] for r in con.execute("SELECT site_id FROM events")} == {"c_fek"}
    _insert(con, id="b", url="https://mining.com/fekola-news", domain="mining.com",
            title="Fekola : la mine d'or accélère sa production", source="gdelt_doc", hint_site="c_kol",
            published_at="2026-11-25T11:00:00Z")
    out = N.run_once(con, index=idx, now=NOW)
    assert out["doublons"] == 1 and out["requetes_heritees"] == 1
    st = dict(con.execute("SELECT id, reject_reason FROM raw_docs").fetchall())
    assert st["b"] == "doublon de a"
    ev = {tuple(e) for e in con.execute("SELECT doc_id, site_id, match_method FROM events")}
    assert ev == {("a", "c_fek", "mot"), ("a", "c_kol", "requête")}
    assert N.run_once(con, index=idx, now=NOW)["lus"] == 0


def test_one_bad_document_does_not_block_the_batch(con, idx, monkeypatch):
    _insert(con, id="bad", url="https://a.com/1", title="Kolubara coal mine news")
    _insert(con, id="odd", url="https://a.com/2", title="Kolubara coal mine flooded",
            locations=json.dumps([{"name": 5, "lat": 1, "lon": 2, "type": "4"}]))
    _insert(con, id="ok", url="https://a.com/3", title="Fekola gold mine strike")
    real = N.match

    def flaky(doc, index):
        if doc["id"] == "bad":
            raise RuntimeError("boum")
        return real(doc, index)

    monkeypatch.setattr(N, "match", flaky)
    out = N.run_once(con, index=idx, now=NOW)
    st = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, status, reject_reason FROM raw_docs")}
    assert st["bad"] == ("rejected", "erreur de traitement (RuntimeError)")
    assert st["odd"][0] == "done" and st["ok"][0] == "done"
    assert out["erreurs"] == 1 and out["evenements"] == 2
    assert N.run_once(con, index=idx, now=NOW)["lus"] == 0              # plus rien de bloqué


@pytest.mark.parametrize("title", ["Kolubara mine: 12 workers trapped",
                                   "Kolubara mine, 2 dead after collapse", "Kolubara mine (3 injured)"])
def test_casualty_count_after_punctuation_is_not_a_unit(idx, title):
    # une ponctuation entre le nom et le nombre : bilan humain, pas numéro d'unité → la mine reste rattachée
    m, _ = N.match(_doc(title, [_loc("Serbia", 44, 21, typ=1)]), idx)
    assert m, title


# -- un mot distinctif + substance / nationalité = nom d'un mot ; bilans humains ≠ numéros d'unité --------
ROWS3 = [
    ("k_kib", "Kibali", "Kibali Gold", 3.13, 29.58, "Democratic Republic of the Congo", 15.0, True, "x"),
    ("k_od", "Olympic Dam", None, -30.44, 136.88, "Australia", 20.0, True, "x"),
    ("k_gra", "Grasberg", None, -4.05, 137.11, "Indonesia", 30.0, True, "x"),
    ("k_lai", "Laizhou Gold Operations", None, 37.18, 119.94, "China", 3.0, True, "x"),
    ("k_ros", "Rossing Uranium", None, -22.48, 15.04, "Namibia", 10.0, True, "x"),
    ("k_fek", "Fekola", None, 12.60, -11.35, "Mali", 8.0, True, "x"),
    ("k_small", "Pervomayskaya Coal", None, 54.0, 86.0, "Russia", None, False, None),
]


@pytest.fixture(scope="module")
def idx3():
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame(ROWS3, columns=cols)
    df["wikidata"] = None
    df["wikipedia"] = None
    return N.build_index(df)


def test_weak_key_rule():
    assert N._weak_key(("kibali", "gold")) and N._weak_key(("canadian", "malartic"))
    assert N._weak_key(N.word_key("Laizhou Gold Operations"))
    assert not N._weak_key(("mount", "isa")) and not N._weak_key(("olympic", "dam"))
    assert not N._weak_key(("los", "bronces")) and not N._weak_key(("escondida", "norte", "zaldivar"))


@pytest.mark.parametrize("title,locs", [
    ("Kibali gold output rises", []),
    ("Kibali gold output rises", [_loc("Congo", -2, 23, typ=1, country="Democratic Republic of the Congo")]),
    ("Laizhou gold output rises", [_loc("China", 35, 105, typ=1)]),
    ("Rossing uranium production falls", []),
])
def test_one_distinctive_word_plus_commodity_stays_below_threshold(idx3, title, locs):
    m, _ = N.match(_doc(title, locs), idx3)
    assert all(x["strength"] < 0.5 for x in m), m
    assert all(x["method"].startswith("mot") for x in m), m


def test_one_distinctive_word_plus_commodity_with_nearby_place(idx3):
    m, _ = N.match(_doc("Kibali gold output rises", [_loc("Durba, Congo", 3.1, 29.6)]), idx3)
    assert list(_sites(m)) == ["k_kib"] and m[0]["method"] == "mot+lieu" and m[0]["strength"] * 0.7 >= 0.5


def test_toponym_name_stays_multiword(idx3):
    m, _ = N.match(_doc("Olympic Dam copper output rises"), idx3)
    assert list(_sites(m)) == ["k_od"] and m[0]["method"] == "nom" and m[0]["strength"] * 0.7 >= 0.5


def test_weak_multiword_key_needs_supported_record(idx3):
    # « Pervomayskaya Coal » : un mot distinctif, fiche non étayée → non indexée (comme un mot seul)
    assert ("pervomayskaya", "coal") not in idx3.words and ("pervomayskaya",) not in idx3.words


@pytest.mark.parametrize("title,site", [
    ("Grasberg 10 miners trapped after collapse", "k_gra"),
    ("Kibali gold mine: 2 dead after collapse", "k_kib"),
    ("Olympic Dam 3 workers injured", "k_od"),
    ("Fekola: 3 killed in pit wall collapse", "k_fek"),
    ("Kibali, 2 workers injured at gold mine", "k_kib"),
    ("Fekola 12 mineurs bloqués dans la mine", "k_fek"),
    ("Grasberg 5 muertos en la mina", "k_gra"),
])
def test_casualty_counts_are_not_units(idx3, title, site):
    near = {"k_gra": _loc("Tembagapura, Indonesia", -4.1, 137.1), "k_kib": _loc("Durba, Congo", 3.1, 29.6),
            "k_od": _loc("Roxby Downs, Australia", -30.5, 136.9),
            "k_fek": _loc("Kenieba, Mali", 12.84, -11.24)}
    m, _ = N.match(_doc(title, [near[site]]), idx3)
    assert site in _sites(m), title


@pytest.mark.parametrize("title", ["Grasberg 2 mine flooded", "Grasberg mine No 2 flooded",
                                   "Grasberg N° 2 flooded", "Grasberg Pit 4 flooded"])
def test_attached_unit_designators_still_separate(idx3, title):
    m, st = N.match(_doc(title, [_loc("Tembagapura, Indonesia", -4.1, 137.1)]), idx3)
    assert m == [] and st["ecartes"] == 1, title                    # compté, pas perdu en silence


def test_duplicate_takes_over_when_original_fails(con, idx, monkeypatch):
    _insert(con, id="q1", url="https://mining.com/q", domain="mining.com", title="Kolubara coal mine news")
    _insert(con, id="q2", url="https://www.mining.com/q/", domain="www.mining.com",
            title="Kolubara coal mine news", source="gdelt_doc", hint_site="c_kol",
            published_at="2026-11-25T11:00:00Z")
    real = N.match

    def flaky(doc, index):
        if doc["id"] == "q1":
            raise RuntimeError("boum")
        return real(doc, index)

    monkeypatch.setattr(N, "match", flaky)
    out = N.run_once(con, index=idx, now=NOW)
    st = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, status, reject_reason FROM raw_docs")}
    assert st["q1"] == ("rejected", "erreur de traitement (RuntimeError)")
    assert st["q2"] == ("done", None)
    assert out["doublons"] == 0 and out["erreurs"] == 1
    ev = {tuple(e) for e in con.execute("SELECT doc_id, site_id FROM events")}
    assert ev == {("q2", "c_kol")}


# -- noms faits de mots de lieu (« Red Lake ») : indexés comme le collecteur les cherche ---------------
ROWS4 = [
    ("p_rl", "Red Lake", "Red Lake Mine", 51.03, -93.83, "Canada", 7.6, True, "x"),
    ("p_cm", "Copper Mountain", None, 49.33, -120.53, "Canada", 10.0, True, "x"),
    ("p_alias", "Other Mine", "Copper Hill", 10.0, 10.0, "Chile", None, False, None),
]


@pytest.fixture(scope="module")
def idx4():
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame(ROWS4, columns=cols)
    df["wikidata"] = None
    df["wikipedia"] = None
    return N.build_index(df)


def test_place_name_keys_match_collect_query_names(idx4):
    from strate.veille import collect
    for name in ("Red Lake", "Copper Mountain"):
        assert collect._place_name_key(name) is not None and N.place_key(name) in idx4.words, name
    assert ("copper", "hill") not in idx4.words           # alias : refusé ici comme par le collecteur
    assert N.place_key("Escondida") is None


def test_region_plus_relief_name_needs_nearby_place():
    # « Ural Mountains » (fiche Wikidata) : jamais un nom de plusieurs mots cherché seul, sinon toute
    # actualité minière de l'Oural lui serait rattachée (0,675 avant correction)
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame([("u", "Ural Mountains", "Lake Victoria", 60.0, 59.0, "Russia", None, False, None)],
                      columns=cols).assign(wikidata=None, wikipedia=None)
    ix = N.build_index(df)
    assert N.word_key("Ural Mountains") is None and N.word_key("Lake Victoria") is None
    assert ix.place_keys == {("ural", "mountains")}           # nom principal : lieu exigé ; alias : rien
    assert N.match(_doc("Gold mine flooded in the Ural Mountains"), ix)[0] == []
    assert N.match(_doc("Gold mine flooded in the Ural Mountains", [_loc("Russia", 60, 100, typ=1)]),
                   ix)[0] == []
    m, _ = N.match(_doc("Ural Mountains mine flooded", [_loc("Karpinsk, Russia", 59.9, 59.2)]), ix)
    assert list(_sites(m)) == ["u"]


def test_place_name_with_mine_query_reaches_threshold(idx4):
    # article de l'API documentaire (sans thèmes ni lieux) renvoyé par la requête « "Red Lake" mine »
    doc = _doc("Red Lake mine strike ends", hint="p_rl")
    m, _ = N.match(doc, idx4)
    assert _sites(m)["p_rl"]["method"] == "requête+nom"
    rows = N._event_rows({**doc, "id": "x", "published_at": "2026-11-01T00:00:00Z", "url": "https://a.b",
                          "domain": "a.b"}, m, idx4, "t")
    assert rows[0][14] >= 0.5                                # confiance : compté par le corrélateur


def test_place_name_needs_nearby_place_without_query(idx4):
    near = [_loc("Red Lake, Ontario, Canada", 51.03, -93.80)]
    m, _ = N.match(_doc("Red Lake mine output rises", near), idx4)
    assert _sites(m)["p_rl"]["method"] == "mot+lieu" and m[0]["strength"] * 0.9 >= 0.5
    for locs in ([], [_loc("Canada", 60, -95, typ=1)], [_loc("Red Lake, Minnesota", 47.88, -95.02)]):
        m, _ = N.match(_doc("Red Lake mine output rises", locs), idx4)
        assert m == [], locs                                  # ville homonyme, pays seul, aucun lieu
    m, _ = N.match(_doc("Copper Mountain mine strike", [_loc("Princeton, Canada", 49.46, -120.51)]), idx4)
    assert list(_sites(m)) == ["p_cm"]


def test_hint_confirmed_by_name_of_unindexed_or_duplicate_record():
    # fiches en double (< 10 km) et fiche non étayée : leur nom n'est pas dans l'index de recherche, mais la
    # requête de CETTE fiche est confirmée par son nom dans le titre (sinon : 0,5 × classement < 0,5)
    cols = ["id", "name", "aliases", "lat", "lon", "country", "footprint_km2", "has_production", "icmm"]
    df = pd.DataFrame([("h1", "Huaron Mines", None, -11.01, -76.45, "Peru", None, False, "x"),
                       ("h2", "Huaron", None, -11.02, -76.44, "Peru", 3.0, True, "x"),
                       ("w1", "Whitney Mine", None, 38.28, -108.72, "United States", None, False, None)],
                      columns=cols).assign(wikidata=None, wikipedia=None)
    ix = N.build_index(df)
    assert ix.words[("huaron",)] == ("h2",) and ix.same_as["h1"] == "h2"
    m, _ = N.match(_doc("Huaron mine strike", hint="h1"), ix)
    assert [(x["site_id"], x["method"]) for x in m] == [("h1", "requête+nom")]   # pas deux fois la mine
    m, _ = N.match(_doc("Whitney Mine flooded", hint="w1"), ix)
    assert _sites(m)["w1"]["method"] == "requête+nom"
    m, _ = N.match(_doc("Whitney Mine No 2 flooded", hint="w1"), ix)
    assert _sites(m)["w1"]["method"] == "requête"                                # autre unité


def test_doc_history_flags_follow_recognised_query_names(con, idx4, monkeypatch):
    from strate.veille import collect
    monkeypatch.setattr(collect.catalog, "load", lambda: pd.DataFrame())
    con.executemany("INSERT INTO tracked(site_id, site_name, reason, since, backfill_status) "
                    "VALUES (?, ?, 'r', '2026-11-01T00:00:00Z', ?)",
                    [("p_rl", "Red Lake", "fait"), ("p_cm", "Ural Mountains", "fait"),
                     ("p_alias", "Other Mine", None)])
    assert N.flag_doc_names(con, idx4) == (1, 1)
    flags = dict(con.execute("SELECT key, value FROM meta WHERE key LIKE 'doc_reconnu:%'").fetchall())
    assert flags == {"doc_reconnu:p_rl": "1", "doc_reconnu:p_cm": "0"}
