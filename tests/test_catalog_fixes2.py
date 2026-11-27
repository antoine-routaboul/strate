"""Catalogue, 2e revue : pays (français, tirets, États disparus), numéros romains et lettres, 3e passe et
échange des liens Jasansky, colonne has_production, doublons Wikidata au même point. Jeux synthétiques."""

import logging

import pandas as pd
import pytest

from strate import catalog, entities, screener


# -- #1 pays : noms français, tirets, points ------------------------------------------------------------
@pytest.mark.parametrize("typed,canon", [
    ("Tunisie", "Tunisia"), ("Estonie", "Estonia"), ("Chypre", "Cyprus"), ("Jordanie", "Jordan"),
    ("Slovénie", "Slovenia"), ("Cambodge", "Cambodia"), ("Lettonie", "Latvia"), ("Irak", "Iraq"),
    ("Syrie", "Syria"), ("Libye", "Libya"), ("Liban", "Lebanon"), ("Fidji", "Fiji"),
    ("Biélorussie", "Belarus"), ("Émirats arabes unis", "United Arab Emirates"),
    ("Écosse", "United Kingdom"), ("Angleterre", "United Kingdom"), ("Pays de Galles", "United Kingdom"),
    ("Kirghizstan", "Kyrgyzstan"), ("RD Congo", "DR Congo"), ("Congo-Kinshasa", "DR Congo"),
    ("République du Congo", "Republic of the Congo"), ("Royaume-Uni", "United Kingdom"),
    ("Royaume Uni", "United Kingdom"), ("Pays Bas", "Netherlands"), ("Pays-Bas", "Netherlands"),
    ("Nouvelle Calédonie", "New Caledonia"), ("Papouasie Nouvelle Guinée", "Papua New Guinea"),
    ("Côte-d'Ivoire", "Côte d'Ivoire"), ("Côte d’Ivoire", "Côte d'Ivoire"), ("U.S.A.", "United States"),
    ("guinea bissau", "Guinea-Bissau"), ("états unis", "United States")])
def test_country_french_and_hyphen_variants(typed, canon):
    assert entities.norm_country(typed) == canon
    col = pd.Series([canon, "Chile"])
    assert screener.country_mask(col, [typed]).tolist() == [True, False]


def test_every_french_name_maps_to_its_country():
    # le vocabulaire est complet : chaque nom français retrouve son pays, aucun ne pointe ailleurs
    for canon, names in entities.FRENCH_NAMES.items():
        for n in names:
            assert entities.norm_country(n) == canon, n
    assert entities.norm_country("Niger") == "Niger" and entities.norm_country("Nigéria") == "Nigeria"
    assert entities.norm_country("Guinée") == "Guinea"


def test_catalog_countries_are_in_modern_vocabulary():
    # les pays réellement présents dans le catalogue construit (lecture seule) : tous connus, sauf
    # les États disparus, que build() remplace désormais par le pays actuel
    cat = catalog.load()
    if cat.empty:
        pytest.skip("catalogue non construit")
    odd = {c for c in cat["country"].dropna().map(entities.norm_country).unique()
           if c not in entities.COUNTRIES}
    assert all(entities.is_historical(c) for c in odd), odd


# -- #8 États disparus → pays actuel du voisin ---------------------------------------------------------------
def test_historical_states_not_renamed_by_name():
    for c in ("Austria–Hungary", "Soviet Union", "Czechoslovakia", "Russian Empire", "Sweden-Finland",
              "Socialist Federal Republic of Yugoslavia", "German Reich", "Holy Roman Empire"):
        assert entities.is_historical(c), c
        assert entities.norm_country(c) not in entities.COUNTRIES  # jamais un pays actuel deviné au nom
    assert not entities.is_historical("Germany") and not entities.is_historical(None)
    assert entities.norm_country("Zaire") == "DR Congo"  # ancien nom d'un seul pays : renommé


def test_modern_country_uses_nearest_site_and_keeps_name_in_region():
    cat = pd.DataFrame({
        "id": ["kladno", "ref_cz", "ref_sk", "slovak_old", "nowhere"],
        "lat": [50.15, 50.16, 48.70, 48.71, -40.0], "lon": [14.10, 14.12, 21.26, 21.25, -120.0],
        "country": ["Austria–Hungary", "Czech Republic", "Slovakia", "Czechoslovakia", "Soviet Union"],
        "region": [None, "Bohême", "Košice", "Spiš", None]})
    out = catalog._modern_country(cat)
    assert out["country"].tolist()[:4] == ["Czech Republic", "Czech Republic", "Slovakia", "Slovakia"]
    assert pd.isna(out.loc[4, "country"])  # pas de voisin à < 50 km : vide plutôt qu'inventé
    assert out.loc[0, "region"] == "Austria–Hungary" and out.loc[4, "region"] == "Soviet Union"
    assert out.loc[3, "region"] == "Spiš"  # une région déjà connue n'est pas écrasée


# -- #9 numéros romains et lettres ----------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("Neyveli I", "Neyveli II"), ("Neyveli IA", "Neyveli II"), ("Kolar A", "Kolar B"),
    ("Field a", "Field D"),
    ("Ramagundam I", "Ramagundam III"), ("King II coal Mine", "Coal King Mine"),
    ("Kolubara D and E coal Mine", "Kolubara B coal Mine")])
def test_roman_numerals_and_letters_separate_in_exact_mode(a, b):
    assert not catalog._sim_tokens(catalog._tokens(a), catalog._tokens(b), exact=True)


@pytest.mark.parametrize("a,b", [
    ("Neyveli II", "Neyveli 2"), ("Kolar A", "Kolar A Mine"), ("J R Simplot", "Simplot"),
    ("Mine d'Or", "Mine d'Or project"), ("King's Mine", "King Mine"), ("Mine No 3", "Mine No. 3 (WV)")])
def test_numbering_keeps_real_twins(a, b):
    # lettre en tête, initiales, élision, possessif : pas des numéros, comme avant
    assert catalog._sim_tokens(catalog._tokens(a), catalog._tokens(b), exact=True)


def test_fold_no_longer_merges_numbered_units():
    base = pd.DataFrame({"name": ["King II coal Mine"], "aliases": [None], "lat": [10.0], "lon": [10.0]})
    other = pd.DataFrame({"name": ["King Coal Mine", "King II"], "lat": [10.001, 10.002],
                          "lon": [10.0, 10.0]})
    m = pd.Series([-1, -1], index=other.index)
    m.iloc[1] = 0  # « King II » a pris la fiche ; « King Coal Mine » ne doit pas y être fondu
    assert catalog.fold(base, other, m).tolist() == [-1, -1]


# -- #5 / #6 liens Jasansky -------------------------------------------------------------------------------
def _jas(rows):
    return pd.DataFrame(rows, columns=["jasansky", "name", "aliases", "lat", "lon", "facility_type",
                                       "status"])


def _cat(rows):
    return pd.DataFrame(rows, columns=["id", "icmm", "name", "aliases", "lat", "lon"])


def test_third_pass_links_far_producer_with_same_name(caplog):
    # Ok Tedi : point Jasansky à 81 km de la fiche ICMM du même nom ; Cerrejon à 12 km
    cat = _cat([("icmm06145", "ICMM06145", "Ok Tedi", None, -5.206, 141.137),
                ("icmm09270", "ICMM09270", "Cerrejon", None, 11.033, -72.666),
                ("icmm1", "ICMM1", "Gold Creek", None, 0.0, 0.0)])
    jas = _jas([("COM00934.00", "Ok Tedi", None, -5.934, 141.138, "Mine", "Active"),
                ("COM00225.00", "Cerrejon", None, 11.133, -72.624, "Mine", "Active"),
                ("COM00999.00", "Gold Creek", None, 0.5, 0.0, "Mine", "Active"),  # nom générique : jamais
                ("COM00500.00", "Ok Tedi", None, -5.9, 141.2, "Refinery", "Active")])  # sans production
    with caplog.at_level(logging.WARNING, logger="strate.catalog"):
        out = catalog.link_jasansky(cat, jas, {"COM00934.00", "COM00225.00", "COM00999.00"})
    assert out["jasansky"].tolist() == ["COM00934.00", "COM00225.00", None]
    assert "3e passe" in caplog.text and "Ok Tedi" in caplog.text


def test_third_pass_requires_unique_candidate_or_single_icmm():
    jas = _jas([("COM1.00", "Morro Velho", None, -19.86, -43.74, "Mine", "Active"),
                ("COM2.00", "Twin Peaks", None, 40.0, 10.0, "Mine", "Active")])
    cat = _cat([("icmm01256", "ICMM01256", "Morro Velho Mine", None, -19.99, -43.74),  # 15 km
                ("q6914173", None, "Morro Velho", None, -20.20, -43.74),  # 38 km, Wikidata
                ("q1", None, "Twin Peaks", None, 40.2, 10.0),
                ("q2", None, "Twin Peaks", None, 39.8, 10.0)])  # deux candidats Wikidata : on s'abstient
    out = catalog.link_jasansky(cat, jas, {"COM1.00", "COM2.00"})
    assert out["jasansky"].tolist() == ["COM1.00", None, None, None]


def test_producer_takes_row_from_non_producer_holder(caplog):
    # Rio Tuba (mine, 5,45 Mt) est un alias de la fiche « Coral Bay HPAL » que la raffinerie sans production
    # prenait par le nom principal
    cat = _cat([("icmm06116", "ICMM06116", "Coral Bay HPAL",
                 "Rio Tuba Mine;Coral Bay Nickel Corporation;Rio Tuba", 8.5600, 117.4200)])
    jas = _jas([("COM00291.00", "Coral Bay Nickel Corporation", None, 8.5573, 117.4247, "Refinery",
                 "Active"),
                ("COM01086.00", "Rio Tuba", None, 8.5700, 117.4182, "Mine", "Active")])
    with caplog.at_level(logging.WARNING, logger="strate.catalog"):
        out = catalog.link_jasansky(cat, jas, {"COM01086.00"})
    assert out.loc[0, "jasansky"] == "COM01086.00"
    assert "remplace COM00291.00" in caplog.text


def test_linked_producer_does_not_displace_anyone():
    # une productrice déjà liée à sa propre fiche ne prend pas celle d'une installation sans production
    cat = _cat([("a", "A", "Dariba Smelter", "Rajpura Dariba", 24.957, 74.124),
                ("b", "B", "Rajpura Dariba", None, 24.947, 74.131)])
    jas = _jas([("COM00313.00", "Dariba Smelter", None, 24.9628, 74.1216, "Smelter", "Active"),
                ("COM01061.00", "Rajpura Dariba", None, 24.9473, 74.1313, "Mine", "Active")])
    out = catalog.link_jasansky(cat, jas, {"COM01061.00"})
    assert out["jasansky"].tolist() == ["COM00313.00", "COM01061.00"]


# -- #7 has_production -------------------------------------------------------------------------------------
def test_load_adds_has_production_for_old_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "OUT", tmp_path)
    monkeypatch.setattr(catalog, "all_production",
                        lambda: pd.DataFrame({"facility_id": ["COM01086.00"], "year": [2019]}))
    pd.DataFrame({"id": ["a", "b", "c"], "name": ["Rio Tuba", "Coral Bay", "x"],
                  "jasansky": ["COM01086.00", "COM00291.00", None]}).to_parquet(tmp_path / "mines.parquet")
    catalog.load.cache_clear()
    try:
        cat = catalog.load()
        assert cat["has_production"].tolist() == [True, False, False]
        assert cat["has_production"].dtype == bool
    finally:
        catalog.load.cache_clear()


def test_load_keeps_built_column_and_survives_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "OUT", tmp_path)

    def boom():
        raise FileNotFoundError("open_database_mine_production.zip")

    monkeypatch.setattr(catalog, "all_production", boom)
    pd.DataFrame({"id": ["a"], "jasansky": ["COM1.00"]}).to_parquet(tmp_path / "mines.parquet")
    catalog.load.cache_clear()
    try:
        assert catalog.load()["has_production"].tolist() == [False]  # source absente : pas de graphique vide
        catalog.load.cache_clear()
        pd.DataFrame({"id": ["a"], "jasansky": ["COM1.00"], "has_production": [True]}).to_parquet(
            tmp_path / "mines.parquet")
        assert catalog.load()["has_production"].tolist() == [True]  # colonne de build() : lue telle quelle
    finally:
        catalog.load.cache_clear()


# -- #10 doublons Wikidata au même point -------------------------------------------------------------------
def _wd(rows):
    return pd.DataFrame(rows, columns=["wikidata", "name", "lat", "lon", "country", "wd_type", "owner",
                                       "operator", "product", "wikipedia"])


def test_wikidata_only_duplicates_at_same_point_are_folded():
    # fiche ICMM du même nom à > 10 km : aucune des deux fiches Wikidata ne la rejoint
    icmm = pd.DataFrame({"icmm": ["ICMM05776"], "id": ["icmm05776"], "name": ["Majdanpek"],
                         "aliases": [None], "lat": [44.60], "lon": [21.94], "country": ["Serbia"],
                         "sources": ["icmm"]})
    wd = _wd([("Q62593429", "Majdanpek", 44.38, 21.944, "Serbia", "mine", None, None, "silver", None),
              ("Q62575710", "Majdanpek", 44.38, 21.944, "Serbia", "mine", None, None, "gold", None),
              ("Q1", "El Dorado North Mine", 40.0, -110.0, "US", "mine", None, None, None, None),
              ("Q2", "El Dorado South Mine", 40.0, -110.0, "US", "mine", None, None, None, None),
              ("Q3", "A Tunnel Mine", 41.0, -110.0, "US", "mine", None, None, None, None),
              ("Q4", "B Tunnel Mine", 41.0, -110.0, "US", "mine", None, None, None, None),
              ("Q5", "Quarry", 42.0, -110.0, "US", "mine", None, None, None, None),
              ("Q6", "Quarry", 42.0, -110.0, "US", "mine", None, None, None, None),
              ("Q7", "Ida", 43.0, -110.0, "US", "mine", None, None, None, "Ida_Mine"),
              ("Q8", "Ida Mine", 43.0003, -110.0, "US", "mine", "Owner Co", None, None, None)])
    cat, folded = catalog.merge_wikidata(icmm, wd)
    maj = cat[cat["name"] == "Majdanpek"]
    assert len(maj) == 2  # la fiche ICMM + UNE fiche Wikidata
    kept = maj[maj["sources"] == "wikidata"].iloc[0]
    assert kept["id"] == "q62575710"  # le QID le plus ancien
    assert set(kept["commodities"].split("; ")) == {"gold", "silver"}
    assert folded["Q62593429"] == kept.name
    assert {"q1", "q2", "q3", "q4", "q5", "q6"} <= set(cat["id"])  # noms différents ou sans mot distinctif
    ida = cat[cat["id"] == "q7"].iloc[0]  # celle qui a l'article est gardée, l'autre comble ses manques
    assert "q8" not in set(cat["id"]) and ida["owner"] == "Owner Co"
    assert folded["Q8"] == ida.name
