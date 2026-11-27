"""Sociétés (docs/previsions-societes.md § 2) : entités, identifiants, faits SEC/Wikidata, investissements
annoncés, corrélations, mise à jour bornée et reprenable, lecture pour l'API.

Aucun accès réseau : SEC et Wikidata sont remplacées par un faux client. Les fichiers
tests/fixtures/sec_*.json sont des extraits réels (companyfacts Newmont us-gaap et BHP ifrs-full,
company_tickers_exchange).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strate import companies as C
from strate import entities

FIX = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIX / name).read_text("utf-8"))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Sorties dans un dossier temporaire ; ni vraie veille, ni User-Agent de la machine."""
    out = tmp_path / "companies"
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "STATUS", out / "status.json")
    monkeypatch.setattr(C, "STATE", out / "state.json")
    monkeypatch.setattr(C, "TICKERS_CACHE", out / "sec_tickers.json")
    monkeypatch.setattr(C, "_veille_db", lambda: tmp_path / "absente.db")
    monkeypatch.delenv("STRATE_SEC_UA", raising=False)
    return tmp_path


def fake_catalog() -> pd.DataFrame:
    rows = [
        dict(
            id="icmm1",
            name="Boddington",
            owner=None,
            operator="Newmont Mining Corporation",
            jasansky="J1",
            commodities="gold",
        ),
        dict(
            id="icmm2",
            name="Tanami",
            owner="Newmont Corporation",
            operator="Newmont",
            jasansky="J2",
            commodities="gold",
        ),
        dict(
            id="icmm3",
            name="Escondida",
            owner="BHP Group; Rio Tinto Group",
            operator="BHP Group",
            jasansky="J3",
            commodities="copper",
        ),
        dict(
            id="icmm4", name="Penasquito", owner=None, operator="Q123456", jasansky=None, commodities="gold"
        ),
        dict(
            id="icmm5", name="Ahafo", owner=None, operator="New Gold Inc.", jasansky=None, commodities="gold"
        ),
        dict(
            id="icmm6",
            name="Cerro",
            owner="Corporación Minera X SA",
            operator=None,
            jasansky=None,
            commodities="silver",
        ),
    ]
    return pd.DataFrame(rows).set_index("id", drop=False)


class FakeHttp:
    """Répond selon l'URL ou le texte SPARQL ; lève sur toute URL inconnue (aucun vrai réseau)."""

    def __init__(self, routes: dict[str, object] | None = None, sparql=None, none_status: int | None = None):
        self.routes = routes or {}
        self.none_status = none_status  # code HTTP simulé quand une route répond None
        self.last_status = None
        self.calls: list[str] = []
        self.sparql_calls: list[str] = []
        self.sparql_fn = sparql or (lambda q: [])
        self.requests = 0
        self.sparql_requests = 0

    def get_json(self, url):
        self.calls.append(url)
        self.requests += 1
        for k, v in self.routes.items():
            if k in url:
                if isinstance(v, Exception):
                    raise v
                self.last_status = 200 if v is not None else self.none_status
                return v
        raise AssertionError(f"URL inattendue : {url}")

    def sparql(self, q):
        self.sparql_calls.append(q)
        self.sparql_requests += 1
        return self.sparql_fn(q)


def _b(**kw) -> dict:
    return {k: {"value": f"http://www.wikidata.org/entity/{v}" if k == "item" else v} for k, v in kw.items()}


# -- noms et entités ------------------------------------------------------------------------------------
def test_norm_key_merges_legal_forms_and_keeps_distinctive_words():
    assert C.norm_key("Newmont Mining Corporation") == C.norm_key("Newmont") == "newmont"
    assert C.norm_key("NEWMONT Corp /DE/") == "newmont"
    assert C.norm_key("The Barrick Gold Corp.") == "barrick"
    assert C.norm_key("New Gold Inc.") == "new gold"  # « new » seul ne distingue rien
    assert C.norm_key("Q123456") is None and C.norm_key(None) is None
    assert "Newmont Corporation" in C.name_variants(
        "Newmont Mining Corporation"
    ) or "Newmont Mining Corporation" in C.name_variants("Newmont Mining Corporation")


def test_norm_key_never_reduces_to_a_country_or_common_word():
    assert C.norm_key("China Coal Energy") == "china coal energy"
    assert C.norm_key("China Resources") == "china resources"
    assert C.norm_key("Korea Coal Corporation") == "korea coal"
    assert C.norm_key("Japan Gold") == "japan gold" and C.norm_key("Japan Energy") == "japan energy"
    assert C.norm_key("Mitsubishi Coal Mining") == "mitsubishi coal" != C.norm_key("Mitsubishi Corporation")
    assert C.norm_key("Royal Gold Inc") == "royal gold"


def test_build_entities_does_not_merge_distinct_companies():
    """Les cinq fusions erronées trouvées sur le vrai catalogue."""
    names = ["Equinox Resources", "China Coal Energy", "China Resources", "Korea Coal Corporation",
             "Korea Resources Corporation", "Mitsubishi Coal Mining", "Mitsubishi Corporation", "Japan Gold",
             "Japan Energy"]
    cat = pd.DataFrame([dict(id=f"m{i}", name=f"Mine {i}", owner=n, operator=None, jasansky=None,
                             commodities="coal") for i, n in enumerate(names)]).set_index("id", drop=False)
    co, lk = C.build_entities(cat, entities.load(), twins={})
    owner = dict(zip(lk["site_id"], lk["company_id"], strict=True))
    assert len({owner[f"m{i}"] for i in range(len(names))}) == len(names)  # neuf sociétés distinctes
    assert owner["m0"] != "equinox"  # Mata Da Corda (Equinox Resources) n'est pas à Equinox Gold
    assert set(lk.loc[lk["company_id"] == "equinox", "site_id"]) == {"greenstone"}


def test_build_entities_dedupes_links_and_shares():
    graph = entities.load()
    co, lk = C.build_entities(fake_catalog(), graph, twins={"boddington": "icmm1"})
    ids = set(co["id"])
    assert {"newmont", "bhp"} <= ids
    # « Newmont Mining Corporation », « Newmont Corporation » et « Newmont » : une seule société
    assert not any(i.startswith("newmont_") for i in ids)
    nm = co.set_index("id").loc["newmont"]
    assert nm["suivie"] and nm["cik"] == "0001164727" and "NYSE:NEM" in nm["tickers"]
    # la jumelle du catalogue de Boddington est rattachée à la mine de la liste suivie (pas comptée deux fois)
    m = lk[lk["company_id"] == "newmont"].set_index("site_id")
    assert "boddington" in m.index and "icmm1" not in m.index
    assert m.at["boddington", "part"] == pytest.approx(1.0)
    assert m.at["icmm2", "role"] == "propriétaire + exploitant" and np.isnan(m.at["icmm2", "part"])
    # liste « ; » éclatée, QID non résolu gardé comme identifiant
    assert set(lk.loc[lk["site_id"] == "icmm3", "company_id"]) >= {"bhp", "rio_tinto"}
    q = co[co["qid"] == "Q123456"]
    assert len(q) == 1 and q.iloc[0]["n_mines"] == 1
    assert int(nm["n_mines"]) == lk[lk["company_id"] == "newmont"]["site_id"].nunique()


def test_merge_by_qid_folds_entities_and_links():
    co = pd.DataFrame(
        [
            dict(
                id="barrick_mining",
                nom="Barrick Mining",
                cle="barrick mining",
                alias="Barrick Mining",
                tickers="NYSE:B",
                cik="0000756894",
                cik_source="graine",
                qid="Q1",
                pays=None,
                sources="graine SEC",
                suivie=False,
                n_mines=1,
            ),
            dict(
                id="barrick",
                nom="Barrick",
                cle="barrick",
                alias="Barrick",
                tickers="",
                cik=None,
                cik_source=None,
                qid="Q1",
                pays="Canada",
                sources="catalogue",
                suivie=False,
                n_mines=2,
            ),
        ],
        columns=C.COMPANY_COLS,
    )
    lk = pd.DataFrame(
        [
            dict(
                company_id="barrick_mining", site_id="s1", nom="A", role="propriétaire", part=0.6, source="x"
            ),
            dict(company_id="barrick", site_id="s1", nom="A", role="exploitant", part=np.nan, source="x"),
            dict(company_id="barrick", site_id="s2", nom="B", role="exploitant", part=np.nan, source="x"),
        ],
        columns=C.LINK_COLS,
    )
    co2, lk2 = C.merge_by_qid(co, lk)
    assert list(co2["id"]) == ["barrick_mining"]
    r = co2.iloc[0]
    assert r["pays"] == "Canada" and "Barrick" in r["alias"] and r["n_mines"] == 2
    assert sorted(lk2["site_id"]) == ["s1", "s2"]
    assert lk2.set_index("site_id").at["s1", "part"] == pytest.approx(0.6)


# -- identifiants ---------------------------------------------------------------------------------------
def test_parse_sec_tickers_both_formats_and_ambiguous_names():
    rows = C.parse_sec_tickers(fixture("sec_tickers_exchange.json"))
    nem = next(r for r in rows if r["ticker"] == "NEM")
    assert nem == dict(cik="0001164727", nom="NEWMONT Corp /DE/", ticker="NEM", bourse="NYSE")
    old = C.parse_sec_tickers({"0": {"cik_str": 2809, "ticker": "AEM", "title": "AGNICO EAGLE MINES LTD"}})
    assert old == [dict(cik="0000002809", nom="AGNICO EAGLE MINES LTD", ticker="AEM", bourse=None)]
    extra = [
        dict(cik="0000000001", nom="Newmont Inc", ticker="NMX", bourse="OTC"),
        dict(cik="0000716643", nom="REGIS CORP", ticker="RGS", bourse="NASDAQ"),
    ]
    by_t, by_k = C.ticker_index(rows + extra)
    assert by_t["BHP"]["cik"] == "0000811809"
    assert "newmont" not in by_k  # deux CIK pour le même nom : correspondance par nom refusée
    assert by_k["agnico eagle mines"]["cik"] == "0000002809"
    assert C.strict_key("Regis Resources") not in by_k  # clé stricte : pas de « Regis Corp »


def test_assign_ids_priority_seed_ticker_name_wikidata():
    co = pd.DataFrame(
        [
            dict(
                id="a",
                nom="Agnico Eagle Mines Limited",
                cle=C.norm_key("Agnico Eagle Mines Limited"),
                alias="",
                tickers="",
                cik=None,
                cik_source=None,
                qid=None,
                pays=None,
                sources="",
                suivie=False,
                n_mines=1,
            ),
            dict(
                id="t",
                nom="Teck",
                cle="teck",
                alias="",
                tickers="TSX:TECK.B;NYSE:TECK",
                cik=None,
                cik_source=None,
                qid=None,
                pays=None,
                sources="",
                suivie=False,
                n_mines=1,
            ),
            dict(
                id="s",
                nom="Seeded",
                cle="seeded",
                alias="",
                tickers="NYSE:BHP",
                cik="0000000042",
                cik_source="graine",
                qid=None,
                pays=None,
                sources="",
                suivie=True,
                n_mines=1,
            ),
            dict(
                id="w",
                nom="Regis Resources",
                cle="regis",
                alias="",
                tickers="ASX:RRL",
                cik=None,
                cik_source=None,
                qid=None,
                pays=None,
                sources="",
                suivie=False,
                n_mines=1,
            ),
            dict(
                id="q",
                nom="Q99",
                cle=None,
                alias="",
                tickers="",
                cik=None,
                cik_source=None,
                qid="Q99",
                pays=None,
                sources="",
                suivie=False,
                n_mines=1,
            ),
        ],
        columns=C.COMPANY_COLS,
    )
    wd = {
        "Regis Resources": dict(qid="Q109627116", cik=None, pays="Australia", label=None),
        "Q99": dict(qid="Q99", cik="0000000099", pays="Peru", label="Minera Q"),
    }
    out = C.assign_ids(co, wd, C.parse_sec_tickers(fixture("sec_tickers_exchange.json"))).set_index("id")
    assert out.at["a", "cik"] == "0000002809" and "nom" in out.at["a", "cik_source"]
    assert out.at["t", "cik"] == "0000886986" and "ticker TECK" in out.at["t", "cik_source"]
    assert out.at["s", "cik"] == "0000000042"  # la graine n'est jamais écrasée
    assert out.at["w", "qid"] == "Q109627116" and out.at["w", "pays"] == "Australia"
    assert pd.isna(out.at["w", "cik"])  # ASX : pas de ticker américain, pas de CIK inventé
    assert out.at["q", "nom"] == "Minera Q" and out.at["q", "cik"] == "0000000099"
    assert "Wikidata P5531" in out.at["q", "cik_source"]


def _co_row(cid, nom, alias="", tickers="", cik=None, src=None):
    return dict(id=cid, nom=nom, cle=C.norm_key(nom), alias=alias, tickers=tickers, cik=cik, cik_source=src,
                qid=None, pays=None, sources="", suivie=False, n_mines=1)


def test_assign_ids_rejects_truncated_single_word_and_subsidiary_names():
    tick = C.parse_sec_tickers(fixture("sec_tickers_exchange.json")) + [
        dict(cik="0000745308", nom="ST JOE Co", ticker="JOE", bourse="NYSE"),
        dict(cik="0001076378", nom="CEMEX SAB DE CV", ticker="CX", bourse="NYSE"),
        dict(cik="0000000777", nom="Equinox Gold Corp.", ticker="EQX", bourse="NYSE")]
    co = pd.DataFrame([_co_row("st_joe_minerals", "St. Joe Minerals", alias="St. Joe;St. Joe Minerals"),
                       _co_row("cemex_uk", "Cemex (United Kingdom)", alias="Cemex (United Kingdom)"),
                       _co_row("eqx", "Equinox Gold", alias="Equinox Gold"),
                       _co_row("a", "Agnico Eagle Mines Limited")], columns=C.COMPANY_COLS)
    out = C.assign_ids(co, {}, tick).set_index("id")
    assert pd.isna(out.at["st_joe_minerals", "cik"])  # « St. Joe » n'est que le début du nom
    assert pd.isna(out.at["cemex_uk", "cik"])  # filiale : pas le CIK de la maison mère
    assert out.at["eqx", "cik"] == "0000000777"  # deux mots, nom complet : accepté
    assert out.at["a", "tickers"] == "NYSE:AEM"  # ticker de la SEC recopié


def test_sic_filter_rejects_non_mining_filers():
    assert C.sic_ok("1040") and C.sic_ok(1221) and C.sic_ok("3334") and C.sic_ok("6795")
    assert not any(map(C.sic_ok, ["6500", "3241", "1389", "6798", "2621", "4931", "2911", "", None]))
    co = pd.DataFrame([_co_row("x", "X Holdings", cik="0000000001", src="SEC company_tickers (nom)"),
                       _co_row("nem", "Newmont", cik="0001164727", src="graine vérifiée (SEC)")],
                      columns=C.COMPANY_COLS)
    sic = {"0000000001": dict(sic="6500", description="Real Estate"),
           "0001164727": dict(sic="6500", description="test")}
    out = C.reject_non_mining(co, sic).set_index("id")
    assert pd.isna(out.at["x", "cik"]) and "SIC 6500" in out.at["x", "cik_source"]
    assert out.at["nem", "cik"] == "0001164727"  # la graine vérifiée n'est pas filtrée


def test_wikidata_lookup_keeps_company_with_cik_then_oldest_qid():
    def sparql(q):
        assert "wdt:P5531|wdt:P1128" in q  # seuls les éléments « société »
        return [
            _b(name="BHP", item="Q900"),
            _b(name="BHP", item="Q625713", cik="0001171264", iso="AU"),
            _b(name="Regis Resources", item="Q20"),
            _b(name="Regis Resources", item="Q10", iso="AU"),
        ]

    got = C.wikidata_lookup(FakeHttp(sparql=sparql), ["BHP", "Regis Resources"])
    assert {k: got["BHP"][k] for k in ("qid", "cik", "pays")} == dict(
        qid="Q625713", cik="0001171264", pays="Australia"
    )
    assert got["Regis Resources"]["qid"] == "Q10" and got["Regis Resources"]["pays"] == "Australia"


# -- faits SEC ------------------------------------------------------------------------------------------
def test_parse_companyfacts_newmont_us_gaap():
    df = C.parse_companyfacts(fixture("sec_newmont_companyfacts.json"), "newmont", "0001164727")
    assert set(df["indicateur"]) >= {"capex", "chiffre_affaires", "exploration", "actifs_miniers"}
    capex = df[df["indicateur"] == "capex"].set_index("annee")
    assert list(capex.index) == [2016, 2017, 2018, 2019]
    assert capex.at[2018, "valeur"] == pytest.approx(1_032_000_000)
    assert capex.at[2018, "concept"] == "us-gaap:PaymentsToAcquireProductiveAssets"
    assert set(capex["formulaire"]) <= {"10-K", "10-K/A"} and set(capex["devise"]) == {"USD"}
    assert capex.at[2018, "source"].startswith("https://www.sec.gov/Archives/edgar/data/1164727/")
    assert set(df["origine"]) == {"sec"} and (df["unite"] == "monnaie").all()
    ca = df[df["indicateur"] == "chiffre_affaires"].set_index("annee")
    assert ca.at[2019, "valeur"] == pytest.approx(9_740_000_000)  # même valeur que Wikidata P2139
    assert ca.at[2017, "valeur"] == pytest.approx(7_379_000_000)  # chiffre retraité du dépôt le plus récent
    assert ca["concept"].isin(C.CONCEPTS["chiffre_affaires"]).all()
    assets = df[df["indicateur"] == "actifs_miniers"]
    assert (assets["fin_exercice"].str[5:] == "12-31").all()  # bilans aux seules clôtures annuelles


def test_parse_companyfacts_bhp_ifrs_june_year_end_and_employees():
    df = C.parse_companyfacts(fixture("sec_bhp_companyfacts.json"), "bhp", "0000811809")
    capex = df[df["indicateur"] == "capex"].set_index("annee")
    assert capex.at[2018, "valeur"] == pytest.approx(4_979_000_000)  # exercice clos le 30 juin, pas le 6-K
    assert capex.at[2018, "fin_exercice"] == "2018-06-30"
    assert capex.at[2018, "concept"].startswith("ifrs-full:PurchaseOfPropertyPlantAndEquipment")
    assert set(capex["formulaire"]) == {"20-F"}
    emp = df[df["indicateur"] == "effectifs"]
    assert len(emp) and (emp["unite"] == "personnes").all() and emp["devise"].isna().all()
    assert emp["valeur"].between(10_000, 200_000).all()


def _fact(end, val, filed, form="10-K", start=None, accn=None):
    x = dict(end=end, val=val, filed=filed, form=form, accn=accn or f"0000000000-{filed[:4]}-{end[:4]}")
    if start:
        x["start"] = start
    return x


def test_parse_companyfacts_rules_restatement_quarter_currency_and_dei():
    facts = {
        "facts": {
            "us-gaap": {
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            _fact("2020-12-31", -100, "2021-02-01", start="2020-01-01", accn="a-2020"),
                            _fact(
                                "2020-12-31", -120, "2022-02-01", start="2020-01-01", accn="a-2021"
                            ),  # retraité
                            _fact("2021-12-31", 130, "2022-02-01", start="2021-01-01", accn="a-2021"),
                            _fact("2021-06-30", 60, "2021-08-01", form="10-Q", start="2021-01-01"),
                            _fact(
                                "2021-09-30", 90, "2022-02-01", start="2021-07-01", accn="a-2021"
                            ),  # trimestre dans un 10-K
                        ]
                    }
                },
                "PaymentsToAcquireProductiveAssets": {
                    "units": {
                        "USD": [_fact("2019-12-31", 80, "2020-02-01", start="2019-01-01", accn="a-2019")]
                    }
                },
                "PaymentsToAcquireMiningAssets": {
                    "units": {
                        "CAD": [_fact("2018-12-31", 70, "2019-02-01", start="2018-01-01", accn="a-2018")]
                    }
                },
                "PropertyPlantAndEquipmentNet": {
                    "units": {
                        "USD": [
                            _fact("2021-12-31", 1000, "2022-02-01", accn="a-2021"),
                            _fact("2021-09-30", 990, "2022-02-01", accn="a-2021"),
                        ]
                    }
                },  # bilan trimestriel repris
            },
            "dei": {
                "EntityNumberOfEmployees": {
                    "units": {"employee": [_fact("2022-02-15", 3200, "2022-02-20", accn="a-2021")]}
                }
            },  # page de garde après la clôture
        }
    }
    df = C.parse_companyfacts(facts, "x", "0000000001")
    cap = df[df["indicateur"] == "capex"].set_index("annee")
    assert cap.at[2020, "valeur"] == 120 and cap.at[2021, "valeur"] == 130  # signe retiré, dernier dépôt
    assert cap.at[2019, "concept"] == "us-gaap:PaymentsToAcquireProductiveAssets"  # bouche le trou
    assert 2018 not in cap.index  # autre devise : jamais mélangée
    ppe = df[df["indicateur"] == "actifs_miniers"]
    assert list(ppe["valeur"]) == [1000]
    emp = df[df["indicateur"] == "effectifs"].iloc[0]
    assert emp["annee"] == 2021 and emp["valeur"] == 3200 and emp["unite"] == "personnes"


# -- faits Wikidata -------------------------------------------------------------------------------------
def test_wikidata_facts_dated_with_currency_and_best_facts():
    def sparql(q):
        assert "pq:P585" in q
        return [
            _b(item="Q1", prop="P2139", amount="9740000000", date="2019-01-01T00:00:00Z", code="USD"),
            _b(item="Q1", prop="P2139", amount="12000000000", date="2019-06-01T00:00:00Z", code="USD"),
            _b(item="Q1", prop="P2139", amount="15000000000", date="2020-01-01T00:00:00Z", code="AUD"),
            _b(item="Q1", prop="P2139", amount="1", date="2021-01-01T00:00:00Z"),  # sans devise
            _b(item="Q1", prop="P1128", amount="+14300", date="+2020-01-01T00:00:00Z"),
            _b(item="Q2", prop="P1128", amount="500", date="2020-01-01T00:00:00Z"),  # société inconnue
        ]

    wf = C.wikidata_facts(FakeHttp(sparql=sparql), {"Q1": "regis"})
    ca = wf[wf["indicateur"] == "chiffre_affaires"]
    assert list(ca["annee"]) == [2019] and ca.iloc[0]["valeur"] == 12e9  # même année : la plus récente
    assert set(ca["devise"]) == {"USD"}  # une seule devise par série
    emp = wf[wf["indicateur"] == "effectifs"].iloc[0]
    assert emp["valeur"] == 14300 and emp["source"] == "https://www.wikidata.org/wiki/Q1#P1128"
    sec = pd.DataFrame(
        [
            dict(
                company_id="regis",
                indicateur="chiffre_affaires",
                annee=2018,
                valeur=1.0,
                unite="monnaie",
                devise="USD",
                source="s",
                formulaire="10-K",
                concept="c",
                fin_exercice="2018-12-31",
                depose=None,
                accn=None,
                cik=None,
                origine="sec",
            )
        ]
    )
    best = C.best_facts(pd.concat([sec, wf], ignore_index=True))
    assert set(best.loc[best["indicateur"] == "chiffre_affaires", "origine"]) == {"sec"}
    assert set(best.loc[best["indicateur"] == "effectifs", "origine"]) == {"wikidata"}


# -- investissements annoncés ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "title,domain,montant,devise,cat",
    [
        ("Newmont to invest $1.2 billion in Boddington expansion", None, 1.2e9, "USD", "expansion"),
        ("Agnico Eagle invests C$100 million in exploration at Detour Lake", None, 1e8, "CAD", "exploration"),
        ("Rio Tinto buys Arcadium Lithium in $6.7 billion deal", None, 6.7e9, "USD", "rachat/fusion"),
        (
            "Vale investirá R$ 2,5 bilhões em nova usina de pelotização",
            None,
            2.5e9,
            "BRL",
            "usine/traitement",
        ),
        ("Codelco invertirá 1.500 millones de dólares en un acueducto", None, 1.5e9, "USD", "infrastructure"),
        ("Orano va investir 1,7 milliard d'euros dans une usine", None, 1.7e9, "EUR", "usine/traitement"),
        ("Rio Tinto investiert 2 Milliarden Euro in Solarpark", None, 2e9, "EUR", "décarbonation/énergie"),
        ("Northern Star approves $1.5bn KCGM mill expansion", "abc.net.au", 1.5e9, "AUD", "usine/traitement"),
        ("Teck to spend $500m on new railway link", "globeandmail.ca", 5e8, "CAD", "infrastructure"),
        ("BHP commits US$4.9bn to Jansen potash", None, 4.9e9, "USD", "autre"),
    ],
)
def test_extract_investment_multilingual(title, domain, montant, devise, cat):
    got = C.extract_investment(title, domain)
    assert got is not None
    assert got["montant"] == pytest.approx(montant) and got["devise"] == devise and got["categorie"] == cat


@pytest.mark.parametrize(
    "title",
    [
        "Newmont reports revenue of $4.6 billion for the quarter",  # résultat, pas investissement
        "Gold price hits $2,400 an ounce as miners expand",  # prix unitaire
        "Barrick fined $10 million over spill",  # pas de verbe d'investissement
        "Mine expansion plan approved",  # pas de montant
        "Company plans 2025 expansion investment",  # une année n'est pas un montant
        "Miner to invest $50,000 in community garden",  # sous le seuil
        "",
        # préfixes de devise collés à un mot (« yea-rs », « inv-estors », « amat-eur »)
        "Barrick approves 5 years 2.5 billion investment",
        # indemnisation, rachat d'actions, dépréciation, offre rejetée
        "Vale approves $1.4 billion settlement for Mariana dam victims",
        "BHP board approves $10bn share buyback",
        "Glencore commits $1bn to share buyback",
        "Newmont develops plan after $5 billion impairment",
        "Anglo American rejects BHP's $39 billion bid",
        "Vale aprova acordo de reparação de R$ 170 bilhões",
        "Barrick annonce une dépréciation de 1 milliard de dollars",
        "Codelco rechaza oferta de US$ 2.000 millones",
        "Rio Tinto investiert trotz Abschreibung von 1 Milliarde Euro",
    ],
)
def test_extract_investment_rejections(title):
    assert C.extract_investment(title) is None


@pytest.mark.parametrize(
    "title,devise",
    [
        ("Over the next five years 2 billion dollars will be invested in the mine", "USD"),
        ("Lithium miner attracts investors 200 million dollars for expansion", "USD"),
        ("Amateur 5 million dollar expansion approved", "USD"),
        ("Hindalco to invest Rs 2,000 crore in smelter", "INR"),
        ("Glencore invests EUR 300 million in recycling plant", "EUR"),
    ],
)
def test_extract_investment_currency_prefix_needs_word_boundary(title, devise):
    got = C.extract_investment(title)
    assert got is not None and got["devise"] == devise


def test_parse_number_locales():
    assert C.parse_number("1,200") == 1200 and C.parse_number("1,5") == 1.5
    assert C.parse_number("1.500") == 1500 and C.parse_number("2.75") == 2.75
    assert C.parse_number("1 200") == 1200 and C.parse_number("1.234,5") == 1234.5


def test_match_companies_longest_and_no_common_words():
    co = pd.DataFrame(
        dict(
            id=["rio", "tinto_x", "gold", "nem"],
            nom=["Rio Tinto", "Tinto", "Gold Corp", "Newmont"],
            alias=["Rio Tinto Group", "", "", "Newmont Mining"],
        )
    )
    idx = C.name_index(co)
    assert "gold" not in idx  # mot courant seul : jamais une société
    assert C.match_companies("Rio Tinto and Newmont Mining strike gold deal", idx) == ["rio", "nem"]


def _veille(tmp_path) -> Path:
    db = tmp_path / "veille.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE raw_docs (id TEXT, published_at TEXT, fetched_at TEXT, url TEXT, domain TEXT, lang TEXT,
                               title TEXT, organizations TEXT, hint_site TEXT, status TEXT);
        CREATE TABLE events (doc_id TEXT, site_id TEXT);""")
    con.executemany(
        "INSERT INTO raw_docs VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "d1",
                "2026-10-01T00:00:00Z",
                None,
                "https://x/1",
                "mining.com",
                "en",
                "Newmont to invest $1.2 billion in mine expansion",
                None,
                None,
                "done",
            ),
            (
                "d2",
                "2026-09-01T00:00:00Z",
                None,
                "https://x/2",
                "abc.net.au",
                "en",
                "Owner approves $300m processing plant at the mine",
                "",
                None,
                "done",
            ),
            (
                "d3",
                "2026-08-01T00:00:00Z",
                None,
                "https://x/3",
                "x.com",
                "en",
                "BHP to buy stake for $2bn",
                "bhp group",
                None,
                "rejected",
            ),
            (
                "d4",
                "2026-07-01T00:00:00Z",
                None,
                "https://x/4",
                "x.com",
                "en",
                "Gold rallies again",
                None,
                None,
                "new",
            ),
        ],
    )
    con.executemany("INSERT INTO events VALUES (?,?)", [("d1", "boddington"), ("d2", "boddington")])
    con.commit()
    con.close()
    return db


def test_investments_from_veille_attach_by_name_and_mine(tmp_path):
    co, lk = C.build_entities(fake_catalog(), entities.load(), twins={})
    inv = C.investments(co, lk, _veille(tmp_path))
    assert list(inv["doc_id"]) == ["d1", "d2"]  # rejeté et sans montant : écartés
    a = inv.set_index("doc_id")
    assert a.at["d1", "company_id"] == "newmont" and a.at["d1", "rattachement"] == "nom+mine"
    assert a.at["d1", "site_id"] == "boddington" and a.at["d1", "categorie"] == "expansion"
    assert a.at["d2", "company_id"] == "newmont" and a.at["d2", "rattachement"] == "mine"
    assert a.at["d2", "devise"] == "AUD" and bool(a.at["d2", "devise_deduite"])


# -- corrélations ---------------------------------------------------------------------------------------
def _panel(signal: float, k: int = 10, years: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(k):
        x = rng.normal(size=years)
        y = signal * x + rng.normal(scale=0.5, size=years)
        rows += [dict(company_id=f"c{c}", annee=2010 + t, x=x[t], y=y[t]) for t in range(years)]
    return pd.DataFrame(rows)


def test_corr_stats_coefficients_interval_n_and_verdict():
    s = C.corr_stats(_panel(1.0), n_boot=300)
    assert s["r"] > 0.6 and 0 < s["rho_lo"] < s["rho"] < s["rho_hi"] and s["r_lo"] < s["r"] < s["r_hi"]
    assert s["n_points"] == 80 and s["n_societes"] == 10
    assert C.verdict(s, 8, 3).startswith("lien positif démontré")
    z = C.corr_stats(_panel(0.0, seed=3), n_boot=300)
    assert z["rho_lo"] < 0 < z["rho_hi"] and C.verdict(z, 8, 3).startswith("non démontré")
    tiny = C.corr_stats(_panel(1.0, k=1, years=4), n_boot=100)
    assert C.verdict(tiny, 8, 3).startswith("trop peu de données")


def test_correlate_capex_production_lag_weighted_by_share():
    years = list(range(2008, 2020))
    rng = np.random.default_rng(0)
    facts, prod, links, cat = [], [], [], []
    for c in range(6):
        capex = np.exp(np.cumsum(rng.normal(0, 0.3, len(years))) + 20)
        g = np.diff(np.log(capex))
        # la production suit le CAPEX avec un an de retard
        p_growth = np.r_[0.0, 0.0, g[:-1]] * 0.9 + rng.normal(0, 0.03, len(years))
        tonnes = np.exp(np.cumsum(p_growth) + 15)
        for y, v in zip(years, capex, strict=True):
            facts.append(
                dict(
                    company_id=f"c{c}",
                    indicateur="capex",
                    annee=y,
                    valeur=v,
                    unite="monnaie",
                    devise="USD",
                    source="s",
                    formulaire="10-K",
                    concept="us-gaap:X",
                    fin_exercice=f"{y}-12-31",
                    depose=None,
                    accn=None,
                    cik=None,
                    origine="sec",
                )
            )
        for site, part, scale in ((f"m{c}", 1.0, 1.0), (f"n{c}", 0.01, -1.0)):
            links.append(
                dict(company_id=f"c{c}", site_id=site, nom=site, role="propriétaire", part=part, source="x")
            )
            cat.append(dict(id=site, jasansky=f"J{site}", commodities="gold"))
            vals = tonnes if scale > 0 else tonnes[::-1]  # mine minoritaire en sens inverse : poids 1 %
            prod += [
                dict(facility_id=f"J{site}", year=y, serie="minerai extrait", value_tonnes=v, source_id="j")
                for y, v in zip(years, vals, strict=True)
            ]
    facts, links = pd.DataFrame(facts, columns=C.FACT_COLS), pd.DataFrame(links, columns=C.LINK_COLS)
    cat = pd.DataFrame(cat).set_index("id", drop=False)
    out = C.correlate(
        facts, links, cat=cat, prod=pd.DataFrame(prod), twins={}, reader=lambda s: pd.DataFrame(), n_boot=200
    )
    assert set(out.columns) == set(C.CORR_COLS)
    ens = out[out["portee"] == "ensemble"].set_index(["x", "y", "decalage"])
    assert len(ens) == len(C.PAIRS)
    lag1 = ens.loc[("CAPEX", "production (mines détenues)", 1)]
    lag0 = ens.loc[("CAPEX", "production (mines détenues)", 0)]
    assert lag1["rho"] > 0.7 and lag1["rho_lo"] > 0 and lag1["verdict"].startswith("lien positif")
    assert abs(lag0["rho"]) < lag1["rho"]
    assert lag1["n_societes"] == 6 and lag1["n_points"] == 6 * 10
    sat = ens.loc[("CAPEX", "emprise satellite (surface nue)", 0)]
    assert sat["n_points"] == 0 and sat["verdict"].startswith("trop peu")
    per = out[(out["portee"] == "societe") & (out["decalage"] == 1) & (out["y"].str.startswith("production"))]
    assert len(per) == 6 and per["r"].notna().all()


def test_mine_footprint_growth_needs_three_clear_months():
    d = pd.DataFrame(
        dict(
            date=pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2023-01-01",
                ]
            ),
            signal="bare_km2",
            value=[10, 10, 10, 20, 20, 20, 40],
            quality=1.0,
        )
    )
    g = C.mine_footprint_growth(["s"], reader=lambda s: d)
    assert list(g["annee"]) == [2022] and g.iloc[0]["g"] == pytest.approx(np.log(2))


# -- réseau ---------------------------------------------------------------------------------------------
def test_sec_user_agent_env_or_generic(monkeypatch):
    assert C.sec_user_agent() == "Strate/0.1 open-source research tool"
    monkeypatch.setenv("STRATE_SEC_UA", "Labo X recherche")
    assert C.sec_user_agent() == "Labo X recherche"


def test_http_rate_limit_and_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(C.time, "sleep", lambda s: sleeps.append(s))

    class R:
        def __init__(self, code, headers=None, data=None):
            self.status_code, self.headers, self._d = code, headers or {}, data

        def json(self):
            return self._d

    answers = [R(429, {"retry-after": "3"}), R(200, data={"ok": 1}), R(404)]

    class Client:
        def get(self, url):
            return answers.pop(0)

    h = C.Http(ua="test")
    h.c = Client()
    assert h.min_interval >= 0.1  # ≤ 10 requêtes/s
    assert h.get_json("u") == {"ok": 1}
    assert 3.0 in sleeps and h.requests == 2
    assert h.get_json("u") is None


# -- mise à jour de bout en bout ------------------------------------------------------------------------
def _routes():
    return {
        "company_tickers_exchange.json": fixture("sec_tickers_exchange.json"),
        "CIK0001164727.json": fixture("sec_newmont_companyfacts.json"),
        "CIK0000811809.json": fixture("sec_bhp_companyfacts.json"),
        "companyfacts/CIK": None,
    }  # toute autre société : pas de companyfacts


def _sparql(q):
    if "pq:P585" in q:
        return [_b(item="Q106238955", prop="P1128", amount="4000", date="2021-01-01T00:00:00Z")]
    if "Northern Star Resources" in q:
        return [_b(name="Northern Star Resources", item="Q106238955", iso="AU")]
    return []


def test_refresh_bounded_resumable_and_api_shapes(monkeypatch, tmp_path):
    from strate import api

    cat = fake_catalog()
    http = FakeHttp(_routes(), _sparql)
    kw = dict(
        http=http,
        cat=cat,
        twins={},
        veille_db=_veille(tmp_path),
        prod=pd.DataFrame(),
        reader=lambda s: pd.DataFrame(),
        n_boot=50,
    )
    st = C.refresh(max_companies=2, **kw)
    first = [u for u in http.calls if "companyfacts" in u]
    assert len(first) == 2 and st["sec_tentees_ce_lancement"] == 2 and st["sec_restantes"] > 0
    assert st["requetes_sec"] == 3  # tickers + 2 companyfacts
    st2 = C.refresh(max_companies=500, **kw)
    second = [u for u in http.calls if "companyfacts" in u][2:]
    assert not set(first) & set(second)  # reprise : rien n'est relu
    assert st2["sec_restantes"] == 0 and st2["message"] == "à jour"
    assert sum("company_tickers" in u for u in http.calls) == 1  # table SEC gardée en cache
    assert sum("pq:P585" in q for q in http.sparql_calls) == 1  # Wikidata : une fois par mois
    C.refresh(max_companies=500, **kw)
    assert len([u for u in http.calls if "companyfacts" in u]) == len(first) + len(second)

    t = C.table()
    assert list(t.columns) == C.TABLE_COLS
    nem = t.set_index("id").loc["newmont"]
    assert nem["capex_dernier"] == pytest.approx(1_463_000_000) and nem["capex_annee"] == 2019
    assert nem["devise"] == "USD" and nem["investissements_12m"] >= 1 and nem["cik"] == "0001164727"
    ns = t.set_index("id").loc["northern_star"]
    assert ns["pays"] == "Australia" and ns["effectifs"] == 4000 and ns["effectifs_annee"] == 2021

    d = C.company("newmont")
    assert set(d) == {"profil", "mines", "faits", "investissements", "correlations"}
    assert d["profil"]["cik"] == "0001164727" and any(m["site_id"] == "boddington" for m in d["mines"])
    f = d["faits"][0]
    assert {"indicateur", "annee", "valeur", "unite", "devise", "source", "formulaire"} <= set(f)
    assert d["investissements"][0]["montant"] == pytest.approx(1.2e9)
    assert C.company("inconnue") is None
    json.dumps(d, allow_nan=False)  # JSON strict

    c = C.correlations()
    assert c["methode"] and len(c["rows"]) == len(C.PAIRS)
    assert {
        "x",
        "y",
        "decalage",
        "r",
        "r_lo",
        "r_hi",
        "rho",
        "rho_lo",
        "rho_hi",
        "n_societes",
        "n_points",
        "verdict",
    } <= set(c["rows"][0])

    # les routes de l'API lisent ces champs tels quels
    monkeypatch.setattr(api, "_mod", lambda name: C if name == "companies" else None)
    rows = api.companies_api(q="newmont", pays=None, limit=10)["rows"]
    assert rows[0]["capex_dernier"] == pytest.approx(1_463_000_000) and rows[0]["tickers"]
    payload = api.company_api("newmont")
    assert payload["faits"] and payload["investissements"][0]["categorie"] == "expansion"
    assert api.companies_correlations()["rows"][0]["n_points"] is not None


def test_refresh_failure_of_one_company_is_isolated(tmp_path):
    routes = _routes()
    routes["CIK0001164727.json"] = RuntimeError("panne")
    http = FakeHttp(routes, _sparql)
    st = C.refresh(
        max_companies=500,
        http=http,
        cat=fake_catalog(),
        twins={},
        veille_db=tmp_path / "no.db",
        prod=pd.DataFrame(),
        reader=lambda s: pd.DataFrame(),
        n_boot=20,
        wikidata=False,
    )
    assert any(e.startswith("newmont") for e in st["erreurs"])
    state = json.loads(C.STATE.read_text("utf-8"))
    assert state["sec"]["newmont"]["ok"] is False and state["sec"]["bhp"]["ok"] is True
    assert "bhp" in set(C.load("facts")["company_id"])


def _refresh_kw(http, tmp_path, **kw):
    return dict(http=http, cat=fake_catalog(), twins={}, veille_db=tmp_path / "no.db", prod=pd.DataFrame(),
                reader=lambda s: pd.DataFrame(), n_boot=20, wikidata=False) | kw


def test_failed_sec_read_keeps_stored_facts_and_reports_error(tmp_path):
    C.refresh(max_companies=500, **_refresh_kw(FakeHttp(_routes(), _sparql), tmp_path))
    before = C.load("facts")
    n_nem = int((before["company_id"] == "newmont").sum())
    assert n_nem > 0
    routes = _routes()
    routes["CIK0001164727.json"] = None  # 403 (User-Agent refusé), 404 ou 5 échecs : get_json rend None
    st = C.refresh(max_companies=500, force=True, **_refresh_kw(FakeHttp(routes, _sparql, 403), tmp_path))
    after = C.load("facts")
    assert int((after["company_id"] == "newmont").sum()) == n_nem  # rien d'effacé
    assert any(e.startswith("newmont : companyfacts illisible (HTTP 403)") for e in st["erreurs"])
    state = json.loads(C.STATE.read_text("utf-8"))
    assert state["sec"]["newmont"]["ok"] is False and state["sec"]["newmont"]["n_faits"] == n_nem


def test_refresh_checks_sic_before_reading_non_seed_filer(tmp_path):
    cat = pd.concat([fake_catalog(), pd.DataFrame([dict(id="icmm9", name="Vault", owner="Joe Holdings Ltd",
                                                        operator=None, jasansky=None, commodities="gold")]
                                                  ).set_index("id", drop=False)])
    routes = _routes()
    # réponses fabriquées pour le test : un déposant au nom identique, secteur immobilier
    tick = fixture("sec_tickers_exchange.json")
    tick["data"].append([9999999, "JOE HOLDINGS LTD", "JOEH", "NYSE"])
    routes["company_tickers_exchange.json"] = tick
    routes["submissions/CIK0009999999.json"] = {"sic": "6500", "sicDescription": "Real Estate"}
    http = FakeHttp(routes, _sparql)
    C.refresh(max_companies=500, **_refresh_kw(http, tmp_path, cat=cat))
    assert any("submissions/CIK0009999999" in u for u in http.calls)
    assert not any("companyfacts/CIK0009999999" in u for u in http.calls)
    co = C.load("companies").set_index("id")
    assert pd.isna(co.at["joe_holdings", "cik"]) and "SIC 6500" in co.at["joe_holdings", "cik_source"]
    n = len(http.calls)
    C.refresh(max_companies=500, force=True, **_refresh_kw(http, tmp_path, cat=cat))
    assert not any("CIK0009999999" in u for u in http.calls[n:])  # rejet mémorisé : plus de requête


def test_permanent_404_is_not_retried_every_day():
    st = {"sec": {"a": dict(at=C._now(), cik="1", ok=False, http=404),
                  "b": dict(at="2000-01-01T00:00:00Z", cik="2", ok=False, http=404),
                  "c": dict(at=C._now(), cik="3", ok=False, http=None)}}
    co = pd.DataFrame(dict(id=["a", "b", "c"], cik=["1", "2", "3"]))
    assert C.sec_queue(co, st) == ["b"]
    st["sec"]["c"]["at"] = "2026-01-01T00:00:00Z"  # échec réseau ancien : retenté après RETRY_DAYS
    assert set(C.sec_queue(co, st)) == {"b", "c"}


def test_readers_empty_before_first_refresh():
    assert C.table().empty and list(C.table().columns) == C.TABLE_COLS
    assert C.company("newmont") is None
    assert C.correlations()["rows"] == []
