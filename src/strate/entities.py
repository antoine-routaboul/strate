"""Graphe d'entités : sociétés ↔ sites. Chargé depuis data/sites.toml."""

from __future__ import annotations

import math
import tomllib
import unicodedata
from functools import cache
from importlib.resources import files

from pydantic import BaseModel

# -- pays : un seul nom par pays ---------------------------------------------------------------
# Les sources écrivent le même pays de plusieurs façons (« Russia » / « Russian Federation »,
# « Côte D’Ivoire » / « CÃ´te d'Ivoire »…) et l'utilisateur tape en français ou en minuscules.
# Tout ce que l'API affiche ou filtre passe par norm_country, sinon un pays compte double.
ISO2 = {"CA": "Canada", "AU": "Australia", "US": "United States", "CL": "Chile", "PE": "Peru",
        "ZA": "South Africa", "CN": "China", "RU": "Russian Federation", "BR": "Brazil", "MX": "Mexico",
        "GB": "United Kingdom", "IN": "India", "ID": "Indonesia", "KZ": "Kazakhstan", "MN": "Mongolia"}

# clés en forme « repliée » (fold) : minuscules, sans accents, apostrophes droites, tirets → espaces
_ALIASES = {
    "russia": "Russian Federation", "russie": "Russian Federation",
    "people's republic of china": "China", "prc": "China", "chine": "China",
    "czechia": "Czech Republic", "tchequie": "Czech Republic", "republique tcheque": "Czech Republic",
    "cote d'ivoire": "Côte d'Ivoire", "ivory coast": "Côte d'Ivoire",
    "democratic republic of the congo": "DR Congo", "democratic republic of congo": "DR Congo",
    "drc": "DR Congo", "rdc": "DR Congo", "congo, the democratic republic of the": "DR Congo",
    "zaire": "DR Congo",
    "republic of the congo": "Republic of the Congo", "congo": "Republic of the Congo",
    "congo-brazzaville": "Republic of the Congo",
    "united republic of tanzania": "Tanzania",
    "myanmar/burma": "Myanmar", "burma": "Myanmar",
    "lao people's democratic republic": "Laos", "viet nam": "Vietnam",
    "korea, republic of": "South Korea", "republic of korea": "South Korea",
    "democratic people's republic of korea": "North Korea",
    "korea, democratic people's republic of": "North Korea",
    "usa": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "wales": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "northern ireland": "United Kingdom", "great britain": "United Kingdom",
    "macedonia": "North Macedonia", "swaziland": "Eswatini",
    "turkiye": "Turkey", "iran (islamic republic of)": "Iran",
    "bolivia (plurinational state of)": "Bolivia", "kyrgyz republic": "Kyrgyzstan",
    # anciens noms d'UN SEUL pays actuel : renommage sans ambiguïté
    "rhodesia": "Zimbabwe", "southern rhodesia": "Zimbabwe", "east germany": "Germany",
    "west germany": "Germany", "german democratic republic": "Germany",
    "federal republic of germany": "Germany",
}

# Nom français de chaque pays du catalogue (l'interface est en français et le filtre « pays » est du texte
# libre) ; les clés de ce tableau forment aussi le vocabulaire des pays actuels que build() vérifie.
FRENCH_NAMES: dict[str, tuple[str, ...]] = {
    "Afghanistan": ("Afghanistan",), "Albania": ("Albanie",), "Algeria": ("Algérie",),
    "Andorra": ("Andorre",), "Angola": ("Angola",), "Argentina": ("Argentine",), "Armenia": ("Arménie",),
    "Aruba": ("Aruba",), "Australia": ("Australie",), "Austria": ("Autriche",),
    "Azerbaijan": ("Azerbaïdjan",), "Bahrain": ("Bahreïn",), "Bangladesh": ("Bangladesh",),
    "Belarus": ("Biélorussie", "Bélarus"), "Belgium": ("Belgique",), "Belize": ("Belize",),
    "Benin": ("Bénin",), "Bhutan": ("Bhoutan",), "Bolivia": ("Bolivie",),
    "Bosnia and Herzegovina": ("Bosnie-Herzégovine",), "Botswana": ("Botswana",), "Brazil": ("Brésil",),
    "Brunei": ("Brunei", "Brunéi"), "Bulgaria": ("Bulgarie",), "Burkina Faso": ("Burkina Faso",),
    "Burundi": ("Burundi",), "Cambodia": ("Cambodge",), "Cameroon": ("Cameroun",), "Canada": ("Canada",),
    "Central African Republic": ("République centrafricaine", "Centrafrique"), "Chad": ("Tchad",),
    "Chile": ("Chili",), "China": ("Chine",), "Colombia": ("Colombie",), "Costa Rica": ("Costa Rica",),
    "Croatia": ("Croatie",), "Cuba": ("Cuba",), "Cyprus": ("Chypre",),
    "Czech Republic": ("Tchéquie", "République tchèque"), "Côte d'Ivoire": ("Côte d'Ivoire",),
    "DR Congo": ("RD Congo", "RDC", "République démocratique du Congo", "Congo-Kinshasa"),
    "Denmark": ("Danemark",), "Dominican Republic": ("République dominicaine",),
    "Ecuador": ("Équateur",), "Egypt": ("Égypte",), "El Salvador": ("Salvador",),
    "Equatorial Guinea": ("Guinée équatoriale",), "Eritrea": ("Érythrée",), "Estonia": ("Estonie",),
    "Eswatini": ("Eswatini",), "Ethiopia": ("Éthiopie",),
    "Federated States of Micronesia": ("Micronésie", "États fédérés de Micronésie"), "Fiji": ("Fidji",),
    "Finland": ("Finlande",), "France": ("France",), "French Guiana": ("Guyane", "Guyane française"),
    "Gabon": ("Gabon",), "Georgia": ("Géorgie",), "Germany": ("Allemagne",), "Ghana": ("Ghana",),
    "Greece": ("Grèce",), "Greenland": ("Groenland",), "Guatemala": ("Guatemala",),
    "Guernsey": ("Guernesey",), "Guinea": ("Guinée",), "Guinea-Bissau": ("Guinée-Bissau",),
    "Guyana": ("Guyana",), "Haiti": ("Haïti",), "Honduras": ("Honduras",), "Hungary": ("Hongrie",),
    "Iceland": ("Islande",), "India": ("Inde",), "Indonesia": ("Indonésie",), "Iran": ("Iran",),
    "Iraq": ("Irak",), "Ireland": ("Irlande",), "Isle of Man": ("Île de Man",), "Israel": ("Israël",),
    "Italy": ("Italie",), "Jamaica": ("Jamaïque",), "Japan": ("Japon",), "Jersey": ("Jersey",),
    "Jordan": ("Jordanie",), "Kazakhstan": ("Kazakhstan",), "Kenya": ("Kenya",), "Kiribati": ("Kiribati",),
    "Kosovo": ("Kosovo",), "Kyrgyzstan": ("Kirghizstan", "Kirghizistan"), "Laos": ("Laos",),
    "Latvia": ("Lettonie",), "Lebanon": ("Liban",), "Lesotho": ("Lesotho",),
    "Liberia": ("Libéria", "Liberia"), "Libya": ("Libye",), "Lithuania": ("Lituanie",),
    "Luxembourg": ("Luxembourg",), "Madagascar": ("Madagascar",), "Malawi": ("Malawi",),
    "Malaysia": ("Malaisie",), "Mali": ("Mali",), "Malta": ("Malte",), "Mauritania": ("Mauritanie",),
    "Mexico": ("Mexique",), "Moldova": ("Moldavie",), "Mongolia": ("Mongolie",),
    "Montenegro": ("Monténégro",), "Morocco": ("Maroc",), "Mozambique": ("Mozambique",),
    "Myanmar": ("Birmanie",), "Namibia": ("Namibie",), "Nauru": ("Nauru",), "Nepal": ("Népal",),
    "Netherlands": ("Pays-Bas",), "New Caledonia": ("Nouvelle-Calédonie",),
    "New Zealand": ("Nouvelle-Zélande",), "Nicaragua": ("Nicaragua",), "Niger": ("Niger",),
    "Nigeria": ("Nigeria", "Nigéria"), "North Korea": ("Corée du Nord",),
    "North Macedonia": ("Macédoine du Nord",), "Norway": ("Norvège",), "Oman": ("Oman",),
    "Pakistan": ("Pakistan",), "Palau": ("Palaos",), "Panama": ("Panama",),
    "Papua New Guinea": ("Papouasie-Nouvelle-Guinée",), "Paraguay": ("Paraguay",), "Peru": ("Pérou",),
    "Philippines": ("Philippines",), "Poland": ("Pologne",), "Portugal": ("Portugal",), "Qatar": ("Qatar",),
    "Republic of the Congo": ("République du Congo", "Congo-Brazzaville"), "Romania": ("Roumanie",),
    "Russian Federation": ("Russie",), "Rwanda": ("Rwanda",), "Saudi Arabia": ("Arabie saoudite",),
    "Senegal": ("Sénégal",), "Serbia": ("Serbie",), "Sierra Leone": ("Sierra Leone",),
    "Singapore": ("Singapour",), "Slovakia": ("Slovaquie",), "Slovenia": ("Slovénie",),
    "Solomon Islands": ("Îles Salomon",), "Somalia": ("Somalie",), "South Africa": ("Afrique du Sud",),
    "South Korea": ("Corée du Sud",), "Spain": ("Espagne",), "Sri Lanka": ("Sri Lanka",),
    "Sudan": ("Soudan",), "Suriname": ("Suriname",), "Sweden": ("Suède",), "Switzerland": ("Suisse",),
    "Syria": ("Syrie",), "Taiwan": ("Taïwan",), "Tajikistan": ("Tadjikistan",), "Tanzania": ("Tanzanie",),
    "Thailand": ("Thaïlande",), "Togo": ("Togo",), "Trinidad and Tobago": ("Trinité-et-Tobago",),
    "Tunisia": ("Tunisie",), "Turkey": ("Turquie",), "Turkmenistan": ("Turkménistan",),
    "Uganda": ("Ouganda",), "Ukraine": ("Ukraine",), "United Arab Emirates": ("Émirats arabes unis",),
    "United Kingdom": ("Royaume-Uni", "Angleterre", "Écosse", "Pays de Galles", "Irlande du Nord",
                       "Grande-Bretagne"),
    "United States": ("États-Unis",), "Uruguay": ("Uruguay",), "Uzbekistan": ("Ouzbékistan",),
    "Venezuela": ("Venezuela", "Vénézuéla"), "Vietnam": ("Viêt Nam",),
    "Western Sahara": ("Sahara occidental",), "Yemen": ("Yémen",), "Zambia": ("Zambie",),
    "Zimbabwe": ("Zimbabwe",),
}
COUNTRIES = frozenset(FRENCH_NAMES)

# États disparus (Wikidata P17 des mines anciennes). Pas de renommage par le nom : l'URSS, l'Autriche-Hongrie
# ou la Yougoslavie couvrent plusieurs pays actuels. build() leur donne le pays du site voisin le plus
# proche (catalog._modern_country) et garde le nom historique dans « region ».
HISTORICAL_STATES = frozenset({
    "soviet union", "ussr", "union of soviet socialist republics", "russian empire", "austria hungary",
    "austrian empire", "austro hungarian empire", "kingdom of hungary", "german reich", "german empire",
    "nazi germany", "weimar republic", "kingdom of prussia", "prussia", "holy roman empire",
    "kingdom of bohemia", "kingdom of saxony", "czechoslovakia", "sweden finland", "yugoslavia",
    "socialist federal republic of yugoslavia", "kingdom of yugoslavia", "federal republic of yugoslavia",
    "serbia and montenegro", "ottoman empire", "british raj", "british india", "kingdom of italy",
    "kingdom of sardinia", "polish lithuanian commonwealth", "grand duchy of finland", "denmark norway",
    "united kingdom of great britain and ireland", "spanish empire", "portuguese empire",
})

_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'",
                         # tirets : « Royaume-Uni » = « Royaume Uni », « Austria–Hungary » idem
                         "-": " ", "–": " ", "—": " ", "‑": " ", "−": " ",
                         ".": None})  # « St. Lucia » = « St Lucia », « U.S.A. » = « USA »


def fold(s: str) -> str:
    """Minuscules, sans accents, apostrophes droites, tirets → espaces, sans points, espaces simples.

    « Côte D’Ivoire », « Côte-d'Ivoire » et « cote d'ivoire » donnent la même clé (l'apostrophe courbe
    n'est pas de l'ASCII : sans la convertir avant, elle disparaît et la clé ne correspond plus à aucun
    alias). Même règle pour les noms de mines dans la recherche : « Coal-Mac » se trouve par « coal mac ».
    """
    s = unicodedata.normalize("NFKD", s.translate(_QUOTES)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


# Toutes les écritures connues → nom canonique, en clés repliées par fold (qui décide seul de la forme)
COUNTRY_ALIASES: dict[str, str] = {fold(k): v for k, v in _ALIASES.items()}
for _canon, _names in FRENCH_NAMES.items():
    for _n in (_canon, *_names):
        COUNTRY_ALIASES.setdefault(fold(_n), _canon)


def is_historical(c) -> bool:
    """Vrai pour un État disparu (« Soviet Union », « Austria–Hungary ») : à remplacer par le pays actuel."""
    return isinstance(c, str) and fold(c) in HISTORICAL_STATES


def norm_country(c) -> str | None:
    """Nom canonique d'un pays (code ISO2, alias, nom français, mojibake) ; sinon le nom tel quel."""
    if not isinstance(c, str) or not c.strip():
        return None
    c = " ".join(c.split())
    if "Ã" in c:  # texte UTF-8 lu comme Latin-1 (« CÃ´te d'Ivoire »)
        try:
            c = c.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if len(c) == 2 and c.isalpha():
        c = ISO2.get(c.upper(), c)
    return COUNTRY_ALIASES.get(fold(c), c)


class Company(BaseModel):
    id: str
    name: str
    tickers: list[str] = []
    wikipedia: str | None = None


class Ownership(BaseModel):
    company: str
    share: float


class KnownEvent(BaseModel):
    date: str  # "YYYY-MM"
    event: str


class Site(BaseModel):
    id: str
    name: str
    country: str
    region: str
    commodity: list[str]
    wikipedia: str | None = None
    lat: float
    lon: float
    radius_m: int
    owners: list[Ownership]
    stage: str
    coord_source: str
    known_events: list[KnownEvent] = []


class Graph(BaseModel):
    companies: dict[str, Company]
    sites: dict[str, Site]

    def site(self, site_id: str) -> Site:
        """Un site de la liste suivie, sinon n'importe quelle mine du catalogue mondial."""
        if site_id in self.sites:
            return self.sites[site_id]
        s = from_catalog(site_id)
        if s is None:
            raise KeyError(f"site inconnu : {site_id!r}")
        return s

    def owners_of(self, site_id: str) -> list[tuple[Company, float]]:
        return [(self.companies[o.company], o.share) for o in self.site(site_id).owners]

    def sites_of(self, company_id: str) -> list[Site]:
        return [s for s in self.sites.values() if any(o.company == company_id for o in s.owners)]


MAUS_CITE = "Maus et al. (2022), contour tracé sur Sentinel-2 — https://doi.org/10.1594/PANGAEA.942325"


def catalog_source(row) -> str:
    """Source citée pour les coordonnées d'une ligne du catalogue : celle qui a fourni le point.

    Les lignes des listes Wikipedia (wp_…) n'ont ni ICMM ni Wikidata : les attribuer à Maus par défaut
    citerait un contour satellite qui n'existe pas pour elles.
    """
    if row.get("icmm"):
        return f"ICMM Global Mining Dataset v1.5 ({row['icmm']})"
    if row.get("wikidata"):
        return f"https://www.wikidata.org/wiki/{row['wikidata']}"
    src = row.get("sources") or ""
    if "wikipedia" in src:
        return "Wikipedia, listes de mines — https://en.wikipedia.org/wiki/Lists_of_mines"
    if "maus" in src or row.get("minx") is not None:
        return MAUS_CITE
    return "source non renseignée dans le catalogue"


@cache
def from_catalog(site_id: str) -> Site | None:
    """Fiche d'une mine du catalogue. Zone d'analyse = emprise réelle (contours satellite) + 500 m."""
    from strate import catalog

    cat = catalog.load()
    if cat.empty or site_id not in cat.index:
        return None
    r = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in cat.loc[site_id].items()}
    lat, lon, radius = r["lat"], r["lon"], 3000
    if r.get("minx") is not None:
        lat, lon = (r["miny"] + r["maxy"]) / 2, (r["minx"] + r["maxx"]) / 2
        dx = (r["maxx"] - r["minx"]) * 111_320 * math.cos(math.radians(lat))
        dy = (r["maxy"] - r["miny"]) * 110_570
        radius = int(min(max(max(dx, dy) / 2 + 500, 1500), 8000))
    comm = [c.strip() for c in (r.get("commodities") or "").replace(";", ",").split(",") if c.strip()]
    country = norm_country(r.get("country")) or "?"
    name = r.get("name") or f"Site minier non nommé ({country})"
    return Site(id=site_id, name=name, country=country, region=r.get("region") or "",
                commodity=comm, wikipedia=r.get("wikipedia"), lat=lat, lon=lon, radius_m=radius,
                owners=[], stage=r.get("production_status") or r.get("asset_type") or "inconnu",
                coord_source=catalog_source(r))


@cache
def load() -> Graph:
    raw = tomllib.loads(files("strate.data").joinpath("sites.toml").read_text("utf-8"))
    companies = {k: Company(id=k, **v) for k, v in raw["companies"].items()}
    sites = {k: Site(id=k, **v) for k, v in raw["sites"].items()}
    for s in sites.values():  # sites.toml écrit le pays en code ISO2 (« CA ») : même nom partout dans l'API
        s.country = norm_country(s.country) or s.country
    for s in sites.values():
        for o in s.owners:
            if o.company not in companies:
                raise ValueError(f"{s.id}: propriétaire inconnu {o.company!r}")
    return Graph(companies=companies, sites=sites)
