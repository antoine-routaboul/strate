"""Couche sociétés : univers, identifiants, comptes SEC, effectifs, segments, ratios, corrélations.

Aucun accès réseau : la SEC et Wikidata sont remplacées par un faux client qui sert les fichiers de
tests/fixtures/companies/ (un extrait réel de companyfacts Newmont, le reste synthétique).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strate import companies as C
from strate import entities

FIX = Path(__file__).parent / "fixtures" / "companies"


def fx(name: str) -> str:
    return (FIX / name).read_text("utf-8")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Sorties dans un dossier temporaire ; ni vraie veille, ni contact SEC de la machine."""
    out = tmp_path / "companies"
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "STATUS", out / "status.json")
    monkeypatch.setattr(C, "STATE", out / "state.json")
    monkeypatch.setattr(C, "CONFIG", tmp_path / "config.local.toml")
    monkeypatch.setattr(C, "_veille_db", lambda: tmp_path / "veille.db")
    monkeypatch.delenv("STRATE_SEC_UA", raising=False)
    return tmp_path


def fake_catalog() -> pd.DataFrame:
    rows = [
        dict(id="icmm1", name="Boddington", aliases=None, owner=None, operator="Newmont Mining Corporation",
             icmm="ICMM1", jasansky="J1"),
        dict(id="icmm2", name="Tanami", aliases="Tanami Operations", owner="Newmont Corporation",
             operator=None,
             icmm="ICMM2", jasansky="J2"),
        dict(id="icmm3", name="Goldstrike", aliases=None, owner="Barrick Gold Corporation;Newmont",
             operator="Barrick Mining Corporation", icmm="ICMM3", jasansky=None),
        dict(id="icmm4", name="Omai", aliases="Omai;Omai Gold Mines Ltd.;National Lead Co. Mine", owner=None,
             operator=None, icmm="ICMM4", jasansky=None),
        dict(id="icmm5", name="Peñasquito", aliases="Penasquito", owner=None, operator="Q123456",
             icmm="ICMM5", jasansky=None),
        dict(id="wd1", name="Ahafo", aliases=None, owner=None, operator="New Gold Inc.", icmm=None,
             jasansky=None),
    ]
    return pd.DataFrame(rows).set_index("id", drop=False)


class FakeHttp:
    """Sert des réponses selon l'URL ; compte les appels ; Wikidata selon le texte de la requête."""

    def __init__(self, routes: dict[str, object] | None = None, sparql_rows=None):
        self.routes = routes or {}
        self.calls: list[str] = []
        self.sparql_rows = sparql_rows or {}
        self.requests = 0

    def _find(self, url):
        self.calls.append(url)
        self.requests += 1
        for k, v in self.routes.items():
            if k in url:
                return v
        return None

    def get_json(self, url):
        v = self._find(url)
        return json.loads(v) if isinstance(v, str) else v

    def get_text(self, url):
        v = self._find(url)
        return v if isinstance(v, str) else None

    def sparql(self, q):
        if "P1128" in q and "p:P1128" in q:
            return self.sparql_rows.get("employees", [])
        return self.sparql_rows.get("lookup", [])


def _b(**kw):
    out = {}
    for k, v in kw.items():
        out[k] = {"value": f"http://www.wikidata.org/entity/{v}" if k == "item" else v}
    return out


# -- noms ----------------------------------------------------------------------------------------------------
def test_norm_key_merges_legal_forms_but_keeps_distinctive_words():
    k = C.norm_key
    assert k("Newmont Mining Corporation") == k("Newmont") == k("NEWMONT Corp /DE/") == "newmont"
    assert k("Barrick Gold Corporation") == "barrick"
    assert k("New Gold Inc.") == "new gold"  # « new » seul serait générique
    assert k("Mineral Resources Limited") == "mineral resources"
    assert k("KG Mining (Bald Mountain), Inc") == "kg mining"  # parenthèse retirée, « kg » trop court
    assert k("Vale S.A.") == "vale" and k("BHP Group") == "bhp"
    assert k("Q123456") is None and k(None) is None
    assert C.id_base("Royal Gold Inc") == "royal_gold"


def test_company_alias_only_when_name_ends_with_legal_form():
    assert C._company_alias("Omai Gold Mines Ltd.")
    assert C._company_alias("Pato Consolidated Dredging Co.")
    assert not C._company_alias("National Lead Co. Mine")
    assert not C._company_alias("Omai")


def test_name_variants_cover_wikidata_spellings():
    v = C.name_variants("Newmont Mining Corporation")
    assert "Newmont Mining Corporation" in v and "Newmont Mining" in v and "Newmont Mining Inc." in v


# -- univers -------------------------------------------------------------------------------------------------
def test_universe_dedupes_links_mines_and_keeps_watchlist_ids():
    uni = C.build_universe(fake_catalog(), entities.load(), veille_sites=["icmm5"])
    by = uni.set_index("id")
    # Newmont : identifiant de la liste suivie, trois écritures fusionnées, CIK de la graine
    nm = by.loc["newmont"]
    assert nm["watchlist"] and nm["cik"] == "0001164727" and "graine SEC" in nm["sources"]
    sites = {m["site"]: m["roles"] for m in C.mines_of(nm)}
    assert sites["icmm1"] == ["exploitant"] and sites["icmm2"] == ["propriétaire"]
    assert "icmm3" in sites and "boddington" in sites  # liste suivie (sites.toml)
    # Barrick Gold et Barrick Mining : une seule société grâce aux alias de la graine
    bar = [i for i in uni["id"] if "barrick" in i]
    assert len(bar) == 1 and by.loc[bar[0], "cik"] == "0000756894"
    # alias ICMM société gardé, alias « … Co. Mine » ignoré, QID non résolu ignoré
    assert any(uni["name"] == "Omai Gold Mines Ltd.")
    assert not any(uni["name"].str.contains("National Lead"))
    assert not any(uni["name"] == "Q123456")
    # cotations jumelles : un seul CIK principal, l'autre en complément
    assert by.loc["bhp", "cik"] == "0000811809" and by.loc["bhp", "cik_alt"] == "0001171264"
    assert len(set(uni["id"])) == len(uni)


def test_assign_ids_priority_seed_then_tickers_then_wikidata():
    uni = pd.DataFrame([
        dict(id="a", name="Alpha Mining", key="alpha", aliases="Alpha Mining", tickers="NYSE:ALP", cik=None,
             cik_source=None, qid=None),
        dict(id="b", name="Beta Corp", key="beta", aliases="Beta Corp", tickers="", cik=None, cik_source=None,
             qid=None),
        dict(id="c", name="Gamma", key="gamma", aliases="Gamma", tickers="", cik="0000000003",
             cik_source="graine", qid=None),
    ])
    wd = {"Beta Corp": dict(qid="Q2", cik="0000000002"), "Gamma": dict(qid="Q3", cik="0000009999"),
          "Alpha Mining": dict(qid="Q1", cik="0000008888")}
    tick = {"T:ALP": dict(cik="0000000001", title="ALPHA MINING")}
    out = C.assign_ids(uni, wd, tick).set_index("id")
    assert out.loc["a", "cik"] == "0000000001" and "ticker" in out.loc["a", "cik_source"]
    assert out.loc["b", "cik"] == "0000000002" and "P5531" in out.loc["b", "cik_source"]
    assert out.loc["c", "cik"] == "0000000003"  # la graine vérifiée n'est jamais écrasée
    assert out.loc["c", "qid"] == "Q3"


def test_wikidata_lookup_prefers_item_with_cik_and_employees_need_date():
    http = FakeHttp(sparql_rows={
        "lookup": [_b(name="Acme", item="Q900"), _b(name="Acme", item="Q901", cik="1234")],
        "employees": [_b(item="Q901", n="5000", date="2023-12-31T00:00:00Z"), _b(item="Q901", n="7000")],
    })
    wd = C.wikidata_lookup(http, ["Acme"])
    assert wd["Acme"] == dict(qid="Q901", cik="0000001234")
    e = C.wikidata_employees(http, ["Q901"])
    assert e.to_dict("records") == [dict(qid="Q901", year=2023, employees=5000.0,
                                         source_url="https://www.wikidata.org/wiki/Q901#P1128")]


# -- comptes annuels -----------------------------------------------------------------------------------------
def test_parse_facts_annual_only_latest_filing_and_concept_fallback():
    f = json.loads(fx("newmont_companyfacts.json"))
    d = C.parse_facts(f, "newmont", "0001164727")
    w = d.pivot_table(index="year", columns="metric", values="value")
    assert 2026 not in w.index  # 2026 n'existe qu'en 10-Q (trimestres) : jamais pris pour un exercice
    assert w.loc[2025, "capex"] == 3.035e9 and w.loc[2025, "revenue"] == 22.669e9
    # retraitement : 2016 publié en 2019 et en 2020 → le dépôt le plus récent gagne
    r16 = d[(d["metric"] == "revenue") & (d["year"] == 2016)].iloc[0]
    rev = f["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
    allv = [x for x in rev
            if x["end"] == "2016-12-31" and x["form"] == "10-K"]
    assert r16["filed"] == max(x["filed"] for x in allv)
    # immobilisations : concept principal, puis « avec droits d'utilisation » pour les années manquantes
    ppe = d[d["metric"] == "ppe_net"].set_index("year")
    assert ppe["concept"].nunique() == 2 and ppe.index.is_unique
    assert (d["value"][d["metric"] == "capex"] > 0).all()
    assert d["source_url"].str.match(r"https://www\.sec\.gov/Archives/edgar/data/1164727/\d{18}/").all()
    assert set(d["unit"]) == {"USD"}


def test_parse_facts_ifrs_and_instant_only_at_fiscal_close():
    facts = {"facts": {"ifrs-full": {
        "Revenue": {"units": {"CAD": [
            dict(start="2023-01-01", end="2023-12-31", val=100, accn="A1", form="40-F", filed="2024-03-01"),
            dict(start="2024-01-01", end="2024-12-31", val=120, accn="A2", form="40-F", filed="2025-03-01"),
            dict(start="2024-01-01", end="2024-12-31", val=121, accn="A3", form="6-K", filed="2025-04-01")]}},
        "PropertyPlantAndEquipment": {"units": {"CAD": [
            dict(end="2024-12-31", val=500, accn="A2", form="40-F", filed="2025-03-01"),
            dict(end="2024-06-30", val=480, accn="A2", form="40-F", filed="2025-03-01")]}}}}}
    d = C.parse_facts(facts, "x", "0000000001")
    rev = d[d["metric"] == "revenue"].set_index("year")["value"]
    assert rev.to_dict() == {2023: 100.0, 2024: 120.0}  # le 6-K n'est pas un rapport annuel
    ppe = d[d["metric"] == "ppe_net"]
    assert ppe["value"].tolist() == [500.0] and set(d["unit"]) == {"CAD"}


def test_merge_alt_fills_missing_years_same_currency_only():
    main = pd.DataFrame([dict(metric="capex", year=2024, unit="USD", value=1.0)]).reindex(columns=C.FIN_COLS)
    alt = pd.DataFrame([dict(metric="capex", year=2024, unit="USD", value=9.0),
                        dict(metric="capex", year=2020, unit="USD", value=2.0),
                        dict(metric="capex", year=2019, unit="ZAR", value=30.0)]).reindex(columns=C.FIN_COLS)
    out = C.merge_alt(main, alt)
    assert out.set_index("year")["value"].to_dict() == {2020: 2.0, 2024: 1.0}


def test_dei_employees_latest_annual_filing():
    facts = {"facts": {"dei": {"EntityNumberOfEmployees": {"units": {"pure": [
        dict(end="2024-12-31", val=900, accn="0000000001-25-000001", form="10-K", filed="2025-02-01"),
        dict(end="2024-12-31", val=950, accn="0000000001-25-000009", form="10-K/A", filed="2025-05-01"),
        dict(end="2025-06-30", val=1000, accn="0000000001-25-000010", form="10-Q", filed="2025-08-01")]}}}}}
    rows = C.dei_employees(facts, "x", "0000000001")
    assert [(r["year"], r["employees"]) for r in rows] == [(2024, 950.0)]


# -- effectifs dans le texte ---------------------------------------------------------------------------------
def test_extract_employees_english_french_and_rejections():
    n, snip = C.extract_employees(C.html_to_text(fx("annual_report_en.htm")))
    assert n == 22_200 and "December 31, 2025" in snip  # « In 2024 employees » n'est pas un effectif
    n, _ = C.extract_employees(C.html_to_text(fx("aif_fr.htm")), lang="fr")
    assert n == 3_400
    assert C.extract_employees("the workforce was approximately 31,000 employees and contractors.") is None
    assert C.extract_employees("We had 3 employees.") is None  # hors plage
    assert C.extract_employees("No headcount disclosed here.") is None


def test_validate_employees_rejects_outlier_against_nearest_year():
    assert C.validate_employees(22_000, 2025, {2024: 21_000})[0]
    ok, why = C.validate_employees(220, 2025, {2024: 21_000, 2019: 50})
    assert not ok and "écart" in why
    assert C.validate_employees(5_000, 2025, {})[0]
    assert not C.validate_employees(2_000_000, 2025, {})[0]


def test_annual_text_url_uses_primary_or_aif_exhibit():
    http = FakeHttp({"-index.htm": fx("filing_index_40f.htm")})
    f40 = dict(form="40-F", accn="0001104659-26-000001", primary="tm40f.htm")
    assert C.annual_text_url(http, "0000002809", f40).endswith("/aif.htm")
    f10 = dict(form="10-K", accn="0001164727-26-000010", primary="nem-20251231.htm")
    assert C.annual_text_url(http, "0001164727", f10) == (
        "https://www.sec.gov/Archives/edgar/data/1164727/000116472726000010/nem-20251231.htm")


def test_annual_filings_sorted_and_filtered():
    sub = {"filings": {"recent": {"form": ["10-Q", "10-K", "8-K", "10-K"],
                                  "accessionNumber": ["a", "b", "c", "d"],
                                  "filingDate": ["2026-05-01", "2026-02-19", "2026-01-01", "2025-02-20"],
                                  "reportDate": ["", "2025-12-31", "", "2024-12-31"],
                                  "primaryDocument": ["q.htm", "k25.htm", "e.htm", "k24.htm"]}}}
    fs = C.annual_filings(sub)
    assert [f["accn"] for f in fs] == ["b", "d"] and fs[0]["report"] == "2025-12-31"


# -- segments ------------------------------------------------------------------------------------------------
def test_parse_segment_capex_keeps_annual_single_segment_facts():
    xs = C.parse_segment_capex(fx("segments_instance.xml"))
    got = {(x["label"], x["year"], x["concept"]): x["value"] for x in xs}
    assert got[("Boddington", 2025, "us-gaap:PaymentsToAcquireProductiveAssets")] == 310e6
    assert got[("Boddington", 2024, "us-gaap:PaymentsToAcquireProductiveAssets")] == 250e6
    assert got[("Tanami", 2025, "us-gaap:PaymentsToAcquireProductiveAssets")] == 420e6  # axe neutre toléré
    assert ("Boddington", 2025, "extension:SustainingCapitalExpenditures") in got
    labels = {x["label"] for x in xs}
    assert "Penasquito" not in labels  # segment × pays : deux dimensions, ambigu
    assert all(x["value"] != 80e6 for x in xs)  # trimestre ignoré
    assert all(x["value"] != 3035e6 for x in xs)  # total sans dimension ignoré
    assert not any(x["concept"].endswith("Revenues") for x in xs)


def test_mine_matcher_company_mines_then_unique_catalogue_name():
    cat = fake_catalog()
    m = C.MineMatcher(cat, entities.load())
    assert m.match("Tanami", ["icmm2", "icmm1"]) == ("icmm2", "Tanami", "mine de la société")
    assert m.match("Goldstrike", []) == ("icmm3", "Goldstrike", "catalogue (nom unique)")
    assert m.match("Africa", ["icmm1"]) == (None, None, None)
    assert m.match("Corporate And Other", ["icmm1"]) == (None, None, None)
    assert m.match("Australia", []) == (None, None, None)
    assert m.match("Boddington", ["boddington", "icmm1"]) == (None, None, None)  # deux fiches : ambigu


def test_announced_investments_from_veille(isolated):
    db = isolated / "veille.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events (site_id TEXT, site_name TEXT, event_type TEXT, published_at TEXT, "
                "title TEXT, url TEXT)")
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", [
        ("icmm1", "Boddington", "operations", "2026-03-01T00:00:00Z", "Newmont approves $1bn expansion",
         "u1"),
        ("icmm1", "Boddington", "finance", "2026-04-01T00:00:00Z", "Inversión de US$ 300 millones", "u2"),
        ("icmm1", "Boddington", "accident", "2026-05-01T00:00:00Z", "Expansion halted after accident", "u3"),
        ("icmm1", "Boddington", "operations", "2026-06-01T00:00:00Z", "Quarterly gold output falls", "u4"),
        ("icmm9", "Autre", "operations", "2026-06-01T00:00:00Z", "Big investment", "u5")])
    con.commit()
    con.close()
    rows = C.announced_investments("newmont", ["icmm1"])
    assert sorted(r["source_url"] for r in rows) == ["u1", "u2"]
    assert all(r["kind"] == "annonce_veille" and np.isnan(r["value"]) for r in rows)


# -- ratios et statistiques ----------------------------------------------------------------------------------
def test_ratios_growth_only_consecutive_years_same_concept():
    w = pd.DataFrame([
        dict(company_id="a", year=2020, capex=100.0, revenue=1000.0, da=50.0, ppe_net=1.0, employees=10.0,
             capex_concept="c1", ppe_net_concept="p1"),
        dict(company_id="a", year=2021, capex=150.0, revenue=1000.0, da=50.0, ppe_net=1.0, employees=10.0,
             capex_concept="c1", ppe_net_concept="p1"),
        dict(company_id="a", year=2023, capex=300.0, revenue=1000.0, da=50.0, ppe_net=1.0, employees=10.0,
             capex_concept="c1", ppe_net_concept="p1"),
        dict(company_id="a", year=2024, capex=330.0, revenue=0.0, da=50.0, ppe_net=1.0, employees=np.nan,
             capex_concept="c2", ppe_net_concept="p1"),
    ])
    r = C.ratios(w).set_index("year")
    assert r.loc[2021, "capex_growth"] == pytest.approx(0.5)
    assert np.isnan(r.loc[2023, "capex_growth"])  # année manquante
    assert np.isnan(r.loc[2024, "capex_growth"])  # concept changé : croissance non comparable
    assert r.loc[2020, "capex_revenue"] == pytest.approx(0.1) and np.isnan(r.loc[2024, "capex_revenue"])
    assert r.loc[2020, "capex_per_employee"] == 10 and r.loc[2020, "capex_da"] == 2


def _panel(signal: float, k: int = 12, years: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(k):
        for y in range(years):
            x = rng.normal()
            rows.append(dict(company_id=f"c{c}", year=2015 + y, x=x, y=signal * x + rng.normal()))
    return pd.DataFrame(rows)


def test_corr_stats_coefficients_interval_and_verdict():
    s = C.corr_stats(_panel(2.0), "x", "y", n_boot=300)
    assert s["n"] == 72 and s["n_companies"] == 12
    assert s["pearson_r"] > 0.7 and s["pearson_lo"] <= s["pearson_r"] <= s["pearson_hi"]
    assert s["spearman_lo"] > 0 and s["verdict"].startswith("lien positif démontré")
    s = C.corr_stats(_panel(0.0, seed=3), "x", "y", n_boot=300)
    assert s["spearman_lo"] < 0 < s["spearman_hi"] and s["verdict"].startswith("non démontré")
    s = C.corr_stats(_panel(2.0, k=2, years=3), "x", "y", n_boot=100)
    assert s["verdict"].startswith("trop peu de données")


def test_company_satellite_same_mines_both_years():
    uni = pd.DataFrame([dict(id="a", mines_json=json.dumps([dict(site="m1", roles=[], share=None),
                                                             dict(site="m2", roles=[], share=0.5)]))])
    def reader(site):
        rows = []
        for y, b in ({"m1": {2021: 10.0, 2022: 11.0}, "m2": {2022: 4.0}}[site]).items():
            for mo in (1, 2, 3, 4):
                rows.append(dict(date=pd.Timestamp(y, mo, 1), signal="bare_km2", value=b, quality=0.9))
                rows.append(dict(date=pd.Timestamp(y, mo, 1), signal="new_bare_km2", value=0.5, quality=0.9))
            rows.append(dict(date=pd.Timestamp(y, 5, 1), signal="bare_km2", value=999.0, quality=0.1))
        return pd.DataFrame(rows)
    s = C.company_satellite(uni, {"m1", "m2"}, reader=reader)
    assert len(s) == 1 and s.iloc[0]["year"] == 2022 and s.iloc[0]["n_mines"] == 1  # m2 n'a pas 2021
    assert s.iloc[0]["sat_growth"] == pytest.approx(np.log(11 / 10))
    assert s.iloc[0]["new_share"] == pytest.approx(0.05)


def test_company_production_mean_log_growth():
    uni = pd.DataFrame([dict(id="a", mines_json=json.dumps([dict(site="m1", roles=[], share=None)]))])
    cat = pd.DataFrame([dict(id="m1", jasansky="F1")]).set_index("id", drop=False)
    ore = [dict(facility_id="F1", year=y, serie="minerai extrait", value_tonnes=v, source_id="s")
           for y, v in ((2015, 100.0), (2016, 110.0), (2018, 121.0))]
    metal = [dict(facility_id="F1", year=2016, serie="or (métal)", value_tonnes=5.0, source_id="s")]
    prod = pd.DataFrame(ore + metal)
    p = C.company_production(uni, cat, prod).set_index("year")
    assert list(p.index) == [2016] and p.loc[2016, "prod_growth"] == pytest.approx(np.log(1.1))


def test_forecast_backtest_never_uses_the_future():
    rows = []
    for y in range(2010, 2020):
        ocf = 100.0 * (1.1 ** (y - 2010))
        capex = 0.5 * ocf if y < 2017 else 5.0 * ocf  # rupture APRÈS 2016
        rows.append(dict(company_id="a", year=y, capex=capex, ocf=ocf, unit="USD"))
    w = pd.DataFrame(rows)
    bt = C.forecast_backtest(w)
    bt = bt[bt["model"] == "cash_flow"].set_index("year")  # un modèle par ligne : ratio CAPEX/flux
    # 2016 : prévision faite fin 2015 → ratio passé 0,5×1,1 → aucun indice de la rupture de 2017
    assert bt.loc[2016, "pred"] == pytest.approx(0.55 * w.set_index("year").loc[2015, "ocf"])
    assert bt.loc[2017, "pred"] < bt.loc[2017, "actual"] / 5
    fc = C.forecast_next(w, "cash_flow", np.zeros(30))
    assert fc.iloc[0]["year"] == 2020 and fc.iloc[0]["lo90"] == pytest.approx(fc.iloc[0]["value"])


# -- accès SEC -----------------------------------------------------------------------------------------------
def test_sec_user_agent_from_env_or_config_never_default_contact(isolated, monkeypatch):
    assert C.sec_user_agent() == (C.GENERIC_UA, False)
    (isolated / "config.local.toml").write_text('[sec]\nuser_agent = "Lab research-desk"\n', "utf-8")
    assert C.sec_user_agent() == ("Lab research-desk", True)
    monkeypatch.setenv("STRATE_SEC_UA", "Env agent")
    assert C.sec_user_agent() == ("Env agent", True)


def test_http_respects_retry_after_and_rate(monkeypatch):
    sleeps = []
    monkeypatch.setattr(C.time, "sleep", lambda s: sleeps.append(s))

    class R:
        def __init__(self, code, headers=None, body=None):
            self.status_code, self.headers, self._b = code, headers or {}, body

        def json(self):
            return self._b

    seq = [R(429, {"retry-after": "7"}), R(200, body={"ok": 1})]
    h = C.Http("test agent", min_interval=0.0)
    h.c = type("Fake", (), {"get": lambda self, url, **kw: seq.pop(0)})()
    assert h.get_json("https://data.sec.gov/x") == {"ok": 1}
    assert 7.0 in sleeps and h.requests == 2


# -- de bout en bout -----------------------------------------------------------------------------------------
def _routes(with_archives: bool) -> dict:
    r = {"companyfacts/CIK0001164727": fx("newmont_companyfacts.json")}
    if with_archives:
        sub = {"filings": {"recent": {"form": ["10-K"], "accessionNumber": ["0001164727-26-000010"],
                                      "filingDate": ["2026-02-19"], "reportDate": ["2025-12-31"],
                                      "primaryDocument": ["nem-20251231.htm"]}}}
        r.update({
            "submissions/CIK0001164727": sub,
            "000116472726000010/index.json": {"directory": {"item": [{"name": "nem-20251231_htm.xml"},
                                                                     {"name": "nem-20251231_lab.xml"}]}},
            "nem-20251231_htm.xml": fx("segments_instance.xml"),
            "nem-20251231.htm": fx("annual_report_en.htm"),
            "company_tickers.json": {"0": {"cik_str": 1164727, "ticker": "NEM", "title": "NEWMONT Corp"}},
        })
    return r


def _only_newmont(uni_fn):
    def f(*a, **k):
        u = uni_fn(*a, **k)
        keep = u["id"].eq("newmont") | u["cik"].isna()
        return u[keep].reset_index(drop=True)
    return f


def test_update_without_contact_reads_financials_only_and_resumes(monkeypatch):
    monkeypatch.setattr(C, "build_universe", _only_newmont(C.build_universe))
    http = FakeHttp(_routes(False), sparql_rows={"lookup": [_b(name="Newmont", item="Q1785405",
                                                                cik="0001164727")]})
    st = C.update(http=http, cat=fake_catalog())
    assert st["sec_contact"] is False and "pas de contact SEC" in st["message"]
    assert st["n_with_financials"] == 1 and st["n_segment_rows"] == 0
    assert not any("www.sec.gov" in u for u in http.calls)  # archives jamais appelées sans contact
    fin = C.load("financials")
    assert set(fin["company_id"]) == {"newmont"} and fin["year"].max() == 2025
    n = len(http.calls)
    C.update(http=http, cat=fake_catalog())  # relancé tout de suite : rien n'est retéléchargé
    assert len(http.calls) == n
    for t in ("companies", "financials", "employees", "segments"):
        assert C._path(t).exists()
    assert "Lab" not in json.dumps(C.status())


def test_update_with_contact_reads_text_and_segments(monkeypatch):
    monkeypatch.setenv("STRATE_SEC_UA", "Lab research-desk")
    monkeypatch.setattr(C, "build_universe", _only_newmont(C.build_universe))
    st = C.update(http=FakeHttp(_routes(True)), cat=fake_catalog(), wikidata=False)
    assert st["sec_contact"] is True and "Lab research-desk" not in json.dumps(st)
    emp = C.load("employees")
    row = emp[emp["source_kind"] == "rapport_annuel"].iloc[0]
    assert row["employees"] == 22_200 and row["year"] == 2025 and bool(row["accepted"])
    seg = C.load("segments")
    seg = seg[seg["kind"] == "xbrl_segment"]
    tan = seg[seg["label"] == "Tanami"].iloc[0]
    assert tan["site_id"] == "icmm2" and tan["match"] == "mine de la société"
    assert seg[seg["label"] == "Africa"]["site_id"].isna().all()


def test_update_limit_and_failure_isolated(monkeypatch):
    calls = []

    class Boom(FakeHttp):
        def get_json(self, url):
            calls.append(url)
            raise RuntimeError("réseau coupé")

    st = C.update(limit=2, http=Boom(), cat=fake_catalog(), wikidata=False)
    assert st["processed_this_run"] == 2 and st["ok_this_run"] == 0 and len(st["errors"]) == 2
    state = json.loads(C.STATE.read_text("utf-8"))
    assert sum(not v["ok"] for v in state["companies"].values()) == 2


def test_report_and_api_shapes(monkeypatch):
    monkeypatch.setattr(C, "build_universe", _only_newmont(C.build_universe))
    C.update(http=FakeHttp(_routes(False)), cat=fake_catalog(), wikidata=False)

    def reader(site):
        if site not in ("icmm1", "icmm2"):
            return pd.DataFrame()
        rows = []
        for y in range(2020, 2026):
            for mo in range(1, 7):
                rows.append(dict(date=pd.Timestamp(y, mo, 1), signal="bare_km2",
                                 value=10 + y - 2020 + mo / 10, quality=0.95))
                rows.append(dict(date=pd.Timestamp(y, mo, 1), signal="new_bare_km2", value=0.3, quality=0.95))
        return pd.DataFrame(rows)

    rep = C.report(cat=fake_catalog(), prod=pd.DataFrame(), analysed={"icmm1", "icmm2"}, reader=reader,
                   n_boot=50)
    corr = C.correlations()
    tests = {c["test"] for c in corr}
    assert {"capex_vs_satellite", "capex_vs_new_bare", "capex_vs_production", "capex_forecast"} <= tests
    for c in corr:
        assert set(C.CORR_COLS) <= set(c) and c["verdict"]
    sat = [c for c in corr if c["test"] == "capex_vs_satellite" and c["lag"] == 0][0]
    assert sat["n"] >= 1 and sat["verdict"].startswith("trop peu")  # une seule société : jamais « démontré »
    assert rep["n_correlations"] == len(corr)
    d = C.dossier("newmont")
    json.dumps(d, allow_nan=False)
    y25 = [y for y in d["financials"] if y["year"] == 2025][0]
    assert y25["capex"] == 3.035e9 and y25["sources"]["capex"]["url"].startswith("https://www.sec.gov/")
    assert "capex_revenue" in y25 and y25["currency"] == "USD"
    assert d["company"]["cik"] == "0001164727" and d["mines"]
    t = C.table()
    json.dumps(t, allow_nan=False)
    nm = [x for x in t if x["id"] == "newmont"][0]
    assert nm["year"] == 2025 and nm["capex"] == 3.035e9
    assert C.dossier("inconnue") is None
    json.dumps(C.status(), allow_nan=False)
