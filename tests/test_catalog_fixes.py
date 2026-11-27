"""Catalogue : production Jasansky (charbon, base du métal, id mère), rapprochement des noms, doublons
Wikidata / listes Wikipedia, mojibake ICMM, titres encodés. Petits jeux synthétiques, aucun téléchargement."""

import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPoint, Point

from strate import catalog


# -- production Jasansky ----------------------------------------------------------------------------
@pytest.fixture
def jas_dir(tmp_path, monkeypatch):
    """Arborescence data/sources/jasansky2023 minimale, branchée sur catalog.SRC."""
    d = tmp_path / "jasansky2023" / "x" / "data"
    d.mkdir(parents=True)
    monkeypatch.setattr(catalog, "SRC", tmp_path)
    catalog.all_production.cache_clear()
    yield d
    catalog.all_production.cache_clear()


def _write_production(d, minerals, commodities, coal):
    pd.DataFrame(minerals, columns=["id", "facility_id", "year", "type", "material", "value_tonnes",
                                    "source_id"]).to_csv(d / "minerals.csv", index=False)
    pd.DataFrame(commodities, columns=["id", "facility_id", "year", "material", "commodity", "value_tonnes",
                                       "source_id", "id_minerals"]).to_csv(d / "commodities.csv", index=False)
    pd.DataFrame(coal, columns=["id", "facility_id", "year", "type", "material", "value_tonnes",
                                "source_id"]).to_csv(d / "coal.csv", index=False)


def _val(p, fid, year, serie):
    r = p[(p["facility_id"] == fid) & (p["year"] == year) & (p["serie"] == serie)]["value_tonnes"]
    return r.tolist()


def test_coal_counted_once_coal_mined_first(jas_dir):
    # finding 12 : « Clean coal » et « Coal mined » décrivent le même charbon, jamais additionnés
    _write_production(jas_dir, [], [], [
        (1, "COM00003.00", 2001, "Clean coal", "F.sbc", 5_000_000, "s1"),
        (2, "COM00003.00", 2001, "Coal mined", "F.sbc", 5_356_109, "s1"),
        (3, "COM00003.00", 2002, "Clean coal", "F.sbc", 4_000_000, "s2"),  # seul : repli
        (4, "COM00004.00", 2001, "Coal mined", "F.sbc", 100, "s3"),
        (5, "COM00004.00", 2001, "Coal mined", "F.lig", 50, "s3"),  # deux charbons extraits : somme
    ])
    p = catalog.all_production()
    assert _val(p, "COM00003.00", 2001, "charbon") == [5_356_109]
    assert _val(p, "COM00003.00", 2002, "charbon") == [4_000_000]
    assert _val(p, "COM00004.00", 2001, "charbon") == [150]


def test_metal_keeps_one_basis(jas_dir):
    # finding 23 : métal du concentré OU du minerai traité, pas les deux
    _write_production(
        jas_dir,
        [(3060, "COM00402.00", 2013, "Concentrate", "Con.Fe", 3_900_000, "s"),
         (3062, "COM00402.00", 2013, "Ore processed", "O.Fe", 10_200_000, "s"),
         (10, "COM00500.00", 2013, "Ore processed", "O.Cu", 1_000_000, "s")],
        [(1, "COM00402.00", 2013, "Con.Fe", "Me.Fe", 2_402_400, "s", 3060),
         (2, "COM00402.00", 2013, "O.Fe", "Me.Fe", 3_151_800, "s", 3062),
         # base inconnue : lue dans le préfixe du matériau (« Con. » → concentré)
         (3, "COM00500.00", 2013, "Con.Cu", "Me.Cu", 9_000, "s", None),
         (4, "COM00500.00", 2013, "O.Cu", "Me.Cu", 10_000, "s", 10),
         # même base, deux matériaux : Cu du concentré de cuivre + Cu du concentré de zinc
         (5, "COM00600.00", 2013, "Con.Cu", "Me.Cu", 700, "s", None),
         (6, "COM00600.00", 2013, "Con.Zn", "Me.Cu", 30, "s", None)],
        [])
    p = catalog.all_production()
    assert _val(p, "COM00402.00", 2013, "fer (métal)") == [2_402_400]
    assert _val(p, "COM00402.00", 2013, "minerai traité") == [10_200_000]
    assert _val(p, "COM00500.00", 2013, "cuivre (métal)") == [9_000]
    assert _val(p, "COM00600.00", 2013, "cuivre (métal)") == [730]


# -- installations Jasansky : id mère -----------------------------------------------------------------
def _jas_zip(tmp_path, monkeypatch):
    f = gpd.GeoDataFrame({
        "facility_id": ["COM00272.00", "COM00272.01", "COM00272.02", "COM00900.00"],
        "facility_name": ["Codelco Norte", "Codelco Norte", "Codelco Norte", "Lonely"],
        "facility_other_names": [None, None, None, "Lone Pit"],
        "sub_site_name": [None, "Chuquicamata", "Radomiro Tomic", None],
        "sub_site_other_names": [None, "División Chuquicamata", None, None],
        "facility_type": ["Mine", None, None, "Mine"],
        "activity_status": ["Active", None, None, None],
    }, geometry=[MultiPoint([(-68.9159, -22.3074), (-68.8679, -22.1988)]), Point(-68.9159, -22.3074),
                 Point(-68.8679, -22.1988), Point(10.0, 10.0)], crs=4326)
    src = tmp_path / "build"
    (src / "data").mkdir(parents=True)
    f.to_file(src / "data" / "facilities.gpkg", driver="GPKG")
    z = tmp_path / "jasansky2023" / "open_database_mine_production.zip"
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(src / "data" / "facilities.gpkg", "data/facilities.gpkg")
    monkeypatch.setattr(catalog, "SRC", tmp_path)


def test_load_jasansky_collapses_sub_facilities_to_parent(tmp_path, monkeypatch):
    _jas_zip(tmp_path, monkeypatch)
    j = catalog.load_jasansky()
    assert set(j["jasansky"]) == {"COM00272.00", "COM00900.00"}  # jamais un sous-id
    codelco = j[j["jasansky"] == "COM00272.00"]
    assert sorted(codelco["name"]) == ["Chuquicamata", "Radomiro Tomic"]  # le point mère doublon disparaît
    assert "Codelco Norte" in codelco.iloc[0]["aliases"]
    assert set(codelco["status"]) == {"Active"}


def test_link_jasansky_gives_parent_id_and_prefers_icmm(tmp_path, monkeypatch):
    # finding 13 : Chuquicamata (ICMM) reçoit l'id mère qui porte la production, pas le sous-id .01
    _jas_zip(tmp_path, monkeypatch)
    jas = catalog.load_jasansky()
    cat = pd.DataFrame({
        "id": ["icmm01372", "q1", "q2"], "icmm": ["ICMM01372", None, None],
        "name": ["Chuquicamata", "Chuquicamata pit", "Radomiro Tomic"], "aliases": [None, None, None],
        "lat": [-22.2968, -22.3070, -22.1990], "lon": [-68.9031, -68.9155, -68.8680]})
    out = catalog.link_jasansky(cat, jas, {"COM00272.00"})
    assert out.loc[0, "jasansky"] == "COM00272.00"  # fiche ICMM d'abord, même un peu plus loin
    assert out["jasansky"].tolist().count("COM00272.00") == 1  # une installation → un seul site
    assert out.loc[0, "production_status"] == "Active"


# -- noms -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("Mount Keith", "Mount Margaret"), ("Gold Creek", "Silver Creek"), ("Las Tortolas", "Las Pintadas"),
    ("San Luis", "San Antonio"), ("Mine No. 6 (WV)", "Mine No 3"),
    ("German Creek Grasstree", "Oaky Creek Mine"), ("Chatterley Whitfield Colliery", "Silverdale Colliery"),
    ("Grube Silberkrone", "Grube Lautenthals Glück"),
    ("Mulia/Jumbang", "Asam Asam")])
def test_similar_rejects_generic_overlap(a, b):
    # finding 15 : un mot générique partagé (Mount, Creek, San, Grube…) ou des numéros différents ≠ même mine
    assert not catalog._similar(a, b)


@pytest.mark.parametrize("a,b", [
    ("Escondida", "Minera Escondida"), ("Rössing", "Rössing uranium mine"),
    ("Kayenta coal Mine", "Kayenta Mine"),
    ("Donlin Creek", "Donlin Gold mine"), ("Lidell", "Liddell"), ("Gold Creek", "Gold Creek Mine"),
    ("Mt. Whaleback mine", "Mount Whaleback mine"), ("Mine No 3", "Mine No. 3 (WV)"),
    ("Asam-Asam", "Asam Asam;Mulia/Jumbang")])
def test_similar_keeps_real_matches(a, b):
    assert catalog._similar(a, b)


def test_match_primary_name_beats_alias_hit():
    # finding 14 : « Mulia/Jumbang » (alias du groupe, 200 m) ne passe plus devant « Asam-Asam » (460 m)
    base = pd.DataFrame({"name": ["Asam Asam"], "aliases": ["Asam Asam;Mulia/Jumbang;Mulia coal Mine"],
                         "lat": [-3.900], "lon": [115.200]})
    other = pd.DataFrame({"name": ["Mulia/Jumbang", "Asam-Asam"],
                          "lat": [-3.9018, -3.9041], "lon": [115.200, 115.200]})
    m = catalog.match(base, other)
    assert m.tolist() == [-1, 0]


# -- doublons Wikidata ------------------------------------------------------------------------------------
def _icmm(**kw):
    d = {"icmm": ["ICMM12517"], "id": ["icmm12517"], "name": ["Kayenta coal Mine"], "aliases": [None],
         "lat": [36.526667], "lon": [-110.431111], "country": ["United States"], "sources": ["icmm"]}
    d.update(kw)
    return pd.DataFrame(d)


def _wd(rows):
    return pd.DataFrame(rows, columns=["wikidata", "name", "lat", "lon", "country", "wd_type", "owner",
                                       "operator", "product", "wikipedia"])


def test_merge_wikidata_folds_duplicate_item_and_unquotes_title():
    # finding 16 : la 2e fiche Wikidata (celle qui a l'article) n'est plus un 2e site à 10 m
    # finding 22 : titre d'article stocké en clair
    wd = _wd([("Q137861312", "Kayenta coal Mine", 36.526667, -110.431111, "US", "mine", None, None, None,
               None),
              ("Q6380556", "Kayenta Mine", 36.5267, -110.431, "US", "mine", "Peabody", None, None,
               "Kayenta_Mine"),
              ("Q1504704", "Rössing uranium mine", -22.48, 15.06, "NA", "mine", None, None, None,
               "R%C3%B6ssing_uranium_mine")])
    cat, folded = catalog.merge_wikidata(_icmm(), wd)
    assert "q6380556" not in set(cat["id"]) and "q137861312" not in set(cat["id"])
    row = cat[cat["id"] == "icmm12517"].iloc[0]
    assert row["wikipedia"] == "Kayenta_Mine" and row["wikidata"] == "Q6380556"  # article + son QID
    assert row["owner"] == "Peabody"
    assert "Kayenta Mine" in row["aliases"]
    assert folded == {"Q6380556": 0}
    assert cat[cat["id"] == "q1504704"].iloc[0]["wikipedia"] == "Rössing_uranium_mine"


# -- listes Wikipedia --------------------------------------------------------------------------------------
def _wpl(rows):
    return pd.DataFrame(rows, columns=["name", "lat", "lon", "wikipedia", "wikidata", "list_page",
                                       "commodity_hint", "where_hint"])


SELBY = [(n, 53.78004, -1.20709, "Selby_Coalfield", "Q7447556", "Yorkshire", "coal", None)
         for n in ("Gascoigne Wood", "North Selby", "Riccall", "Stillingfleet")]


def test_clean_wplists_parent_article_duplicates_and_bad_names():
    w = _wpl(SELBY + [
        # même mine sur deux listes (finding 17)
        ("Meliadine Gold Mine", 63.032025, -92.221067, "Meliadine_Gold_Mine", "Q107535328", "a", "gold",
         None),
        ("Meliadine Mine", 63.032778, -92.221389, "Meliadine_Gold_Mine", "Q107535328", "b", None, None),
        # même point, sans QID pour l'une
        ("Taio gold mine", 33.139, 130.894, "Taio_gold_mine", "Q11674272", "c", "gold", None),
        ("Taio mine", 33.139, 130.894, None, None, "c", None, None),
        # un district cité à 3 endroits éloignés : lien retiré, lignes gardées
        ("Sundergarh", 22.02, 84.50, "Sundergarh", "Q2352156", "d", None, None),
        ("Sundergarh", 21.92, 85.30, "Sundergarh", "Q2352156", "d", None, None),
        # rang pris pour un nom, restes de modèle et tabulations (finding 19)
        ("8", 33.149, -110.987, None, None, "e", "copper", None),
        ("Rössing\t", -22.48, 15.06, None, None, "f", None, None),
        ("Mounana{{cite web", -1.40, 13.16, None, None, "g", None, None),
    ])
    out = catalog.clean_wplists(w)
    names = set(out["name"])
    assert not names & {"Gascoigne Wood", "North Selby", "Riccall", "Stillingfleet"}  # finding 18
    assert "8" not in names and "Rössing" in names and "Mounana" in names
    mel = out[out["wikidata"] == "Q107535328"]
    assert len(mel) == 1 and mel.iloc[0]["aliases"] == "Meliadine Mine"
    assert len(out[out["name"].str.startswith("Taio")]) == 1
    sund = out[out["name"] == "Sundergarh"]
    assert len(sund) == 2 and sund["wikidata"].isna().all()
    assert out["wikidata"].dropna().is_unique


def test_merge_wplists_unique_ids_and_fold():
    cat = pd.DataFrame({
        "id": ["icmm08883"], "icmm": ["ICMM08883"], "name": ["Moose River"], "aliases": [None],
        "lat": [44.98], "lon": [-62.94], "wikidata": [None], "wikipedia": [None], "sources": ["icmm"]})
    w = _wpl([
        ("Moose River Consolidated", 44.981, -62.941, None, None, "a", "gold", None),
        ("Moose River Gold Mine", 44.984167, -62.942778, "Moose_River_Gold_Mines", "Q14875728", "b", None,
         None),
        # deux mines distinctes au même point, sans QID : deux ids distincts, aucune perdue
        ("Sala silvergruva", 64.8009, 20.7423, None, None, "c", "silver", None),
        ("Vildmarksgruvan", 64.8009, 20.7423, None, None, "c", None, None),
    ])
    out = catalog.merge_wplists(cat, w)
    assert out["id"].is_unique
    assert {"Sala silvergruva", "Vildmarksgruvan"} <= set(out["name"])
    moose = out[out["id"] == "icmm08883"].iloc[0]
    assert len(out[out["name"].fillna("").str.startswith("Moose")]) == 1  # finding 17 : pas de 2e site
    assert moose["wikidata"] == "Q14875728" and moose["wikipedia"] == "Moose_River_Gold_Mines"
    assert "Moose River Gold Mine" in moose["aliases"]


def test_parse_list_skips_rank_column():
    wt = """{| class="wikitable"
! Rank !! Mine !! Coordinates
|-
| 1 || [[Morenci mine|Morenci]] || {{coord|33|05|N|109|20|W}}
|-
| 2 || [[Bingham Canyon Mine]]\t || {{coord|40.52|-112.15}}
|}"""
    rows = catalog.parse_list(wt)
    assert [r["name"] for r in rows] == ["Morenci", "Bingham Canyon Mine"]
    assert rows[0]["article"] == "Morenci mine"


# -- texte --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad,good", [
    ("RÃ¶ssing", "Rössing"), ("PyhÃƒÂ¤salmi", "Pyhäsalmi"), ("GuaÃƒÂ\xadba coal Mine", "Guaíba coal Mine"),
    ("Ã„Å’SA coal Mine", "ČSA coal Mine"), ("Al Baâ€™itha", "Al Ba’itha"),
    ("CÃ´te d'Ivoire", "Côte d'Ivoire"),
    ("Rönnskär; RÃ¶nnskÃ¤r", "Rönnskär; Rönnskär"),  # forme juste et forme abîmée mêlées
    ("Äänekoski", "Äänekoski"), ("Escondida", "Escondida")])
def test_fix_text_repairs_mojibake_only(bad, good):
    # finding 20
    assert catalog.fix_text(bad) == good


def test_title_unquotes():
    assert catalog._title("R%C3%B6ssing_uranium_mine") == "Rössing_uranium_mine"
    assert catalog._title("Mount Whaleback") == "Mount_Whaleback"
    assert catalog._title(None) is None


def test_mine_numbers_never_make_names_similar():
    from strate.catalog import _similar

    pairs = [("Prosperity Mine No. 1", "Rockhouse 1 Mine"),
             ("Pratt No.1 coal Mine", "Dolomite Number 1 Slope Mine"),
             ("Lick Branch Mine No 2", "Big Creek No 2 Surface Mine"),
             ("#1 (KY ECP)", "Booneville #1")]
    for a, b in pairs:
        assert not _similar(a, b), (a, b)
    assert _similar("Mine No 6", "Mine No. 6")
    assert not _similar("Mine No 6", "Mine No 3")


def test_fold_requires_same_distinctive_name():
    from strate.catalog import _sim_tokens, _tokens

    assert _sim_tokens(_tokens("Kayenta Mine"), _tokens("Kayenta"), exact=True)
    assert not _sim_tokens(_tokens("Doctor Divide Mine"), _tokens("Tonopah Divide"), exact=True)
    assert not _sim_tokens(_tokens("Golden Mine"), _tokens("Golden Sunlight"), exact=True)
    assert not _sim_tokens(_tokens("Burgin Number 1"), _tokens("Burgin Kennecott"), exact=True)
