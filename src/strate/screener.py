"""Screener : une ligne par site du catalogue mondial, filtrable et triable.

Colonnes calculées sans aucun téléchargement satellite (donc pour TOUTES les mines) :
  - pays et commodités normalisés (les sources écrivent « Russia » / « Russian Federation »,
    « gold ore » / « gold »…) ;
  - emprise réelle (contours Maus 2022) et production estimée d'après l'emprise : modèle calibré
    (calibrate.footprint_model), toujours avec son intervalle à 90 %, et seulement dans la population
    où il a été ajusté (mine de métal ou de charbon à commodité connue ; ni usine, ni emprise anonyme) ;
  - dernière production réelle publiée (Jasansky et al. 2023) quand elle existe.
Colonnes ajoutées pour les mines déjà analysées : niveau d'alerte du dernier mois, explication.
Les mines de la liste suivie n'apparaissent qu'une fois, sous leur identifiant, avec les données de
leur « jumelle » du catalogue (voir twins).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable
from datetime import date
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from strate import catalog, entities, fusion
from strate.core import DATA_DIR
from strate.entities import COUNTRY_ALIASES, ISO2, norm_country  # noqa: F401  (réexportés)
from strate.entities import fold as _fold

OUT = DATA_DIR / "catalog" / "screener.parquet"
BIG_FOOTPRINT_KM2 = 100  # au-delà, l'emprise mélange souvent plusieurs exploitations
TWIN_KM = 10.0  # rayon de recherche du « jumeau » catalogue d'une mine de la liste suivie
# Version du contenu de la table : à incrémenter quand la construction change, sinon un screener.parquet
# écrit par l'ancien code reste servi tant qu'aucune donnée d'entrée n'est plus récente que lui.
TABLE_VERSION = 3  # 3 : « analyse » = analyse récente, colonne de recherche (noms de la jumelle compris)
# Une analyse compte comme « récente » si son dernier mois de signaux optiques date de moins de ~6 mois :
# même règle que fusion.radar. Les mines passées au satellite pour la seule calibration (2016–2019)
# ont des signaux, mais rien d'actuel : elles doivent rester proposées à l'analyse.
RECENT_DAYS = 183

# commodité brute → groupe (français), dans l'ordre de priorité d'affichage
COMMODITY_GROUPS = [
    ("charbon", ("coal", "lignite", "anthracite", "bituminous", "coking", "thermal coal")),
    ("or", ("gold",)),
    ("cuivre", ("copper",)),
    ("fer", ("iron", "magnetite", "hematite", "taconite")),
    ("argent", ("silver",)),
    ("zinc", ("zinc",)),
    ("plomb", ("lead",)),
    ("nickel", ("nickel",)),
    ("lithium", ("lithium", "spodumene")),
    ("cobalt", ("cobalt",)),
    ("platinoïdes", ("platinum", "palladium", "pgm", "pge", "rhodium", "iridium", "ruthenium", "osmium")),
    ("uranium", ("uranium",)),
    ("aluminium / bauxite", ("bauxite", "alumin")),
    ("manganèse", ("manganese",)),
    ("chrome", ("chrom",)),
    ("étain", ("tin",)),
    ("tungstène", ("tungsten", "wolfram")),
    ("molybdène", ("molybden",)),
    ("terres rares", ("rare earth", "ree", "neodymium", "lanthan", "cerium")),
    ("diamant", ("diamond",)),
    ("potasse / sel", ("potash", "salt", "halite", "sylvite")),
    ("phosphate", ("phosph",)),
    ("graphite", ("graphite",)),
    ("vanadium", ("vanadium",)),
    ("antimoine", ("antimony",)),
    ("granulats / pierre", ("stone", "limestone", "granite", "gravel", "sand", "slate", "marble", "clay",
                            "chalk", "gypsum", "aggregate", "dolomite", "basalt", "kaolin")),
]


def commodity_groups(raw) -> list[str]:
    """« gold ore, copper; Silver » → ['or', 'cuivre', 'argent'] (ordre d'apparition, sans doublon)."""
    if not isinstance(raw, str):
        return []
    out = []
    for tok in re.split(r"[;,/]", raw):
        t = _fold(tok)
        if not t:
            continue
        for group, keys in COMMODITY_GROUPS:
            if any(re.search(rf"\b{re.escape(k)}", t) for k in keys):
                if group not in out:
                    out.append(group)
                break
    return out


# Groupes de commodités présents dans la population d'ajustement du modèle « emprise → production »
# (mines à production de minerai extrait connue, Jasansky et al. 2023). Hors de cette population
# (carrières, potasse, diamant, graphite…) le modèle n'a jamais été vérifié : pas d'estimation.
ORE_GROUPS = frozenset({
    "or", "cuivre", "fer", "argent", "zinc", "plomb", "nickel", "cobalt", "platinoïdes", "uranium",
    "molybdène", "étain", "manganèse", "chrome", "aluminium / bauxite", "lithium", "tungstène",
    "antimoine", "vanadium", "terres rares",
})

_cal: tuple[int, dict] | None = None
_cal_lock = threading.Lock()


def _mtime_ns(p) -> int:
    try:
        return Path(str(p)).stat().st_mtime_ns
    except OSError:
        return 0


def _cal_path():
    return files("strate.data").joinpath("calibration.json")


def calibration() -> dict:
    """calibration.json, relu dès que le fichier change.

    Seul chargeur du projet (l'API l'utilise aussi) : la calibration tourne dans un autre processus,
    et un cache « à vie » servirait l'ancien modèle à côté des estimations du nouveau.
    """
    global _cal
    p = _cal_path()
    mt = _mtime_ns(p)
    with _cal_lock:
        if _cal is None or _cal[0] != mt:
            try:
                data = json.loads(p.read_text("utf-8")) if p.is_file() else {}
            except (OSError, json.JSONDecodeError):
                # fichier en cours d'écriture : on garde l'ancien et on relira au prochain appel
                return _cal[1] if _cal else {}
            _cal = (mt, data)
        return _cal[1]


def estimate_kind(groups: list[str], categorie: str | None = None) -> str | None:
    """Modèle applicable (« minerai », « charbon ») ou None hors de la population d'ajustement.

    Le modèle a été ajusté sur des mines nommées à production connue : une usine (fonderie,
    raffinerie) n'extrait rien, et une emprise sans commodité connue peut être du charbon comme
    du métal — les deux modèles divergent d'un facteur 3 à 100 km², l'intervalle ne le couvre pas.
    """
    if categorie == "usine" or not groups:
        return None
    if groups[0] == "charbon":
        return "charbon"
    return "minerai" if groups[0] in ORE_GROUPS else None


def footprint_estimate(emprise_km2, groups: list[str], categorie: str | None = None
                       ) -> tuple[str | None, float, float, float]:
    """(type, tonnes/an, bas 90 %, haut 90 %) d'après l'emprise ; NaN hors domaine du modèle."""
    none = (None, np.nan, np.nan, np.nan)
    try:
        emprise_km2 = float(emprise_km2)
    except (TypeError, ValueError):
        return none
    if not math.isfinite(emprise_km2) or emprise_km2 <= 0:
        return none
    kind = estimate_kind(groups, categorie)
    m = calibration().get("modele_emprise", {}).get(kind) if kind else None
    if not m or not (m["emprise_min_km2"] <= emprise_km2 <= m["emprise_max_km2"]):
        return none
    est = 10 ** (m["intercept"] + m["slope"] * math.log10(emprise_km2))
    return kind, est, est / m["facteur_90"], est * m["facteur_90"]


GROUP_TO_SERIES = {"or": "or", "cuivre": "cuivre", "fer": "fer", "argent": "argent", "zinc": "zinc",
                   "plomb": "plomb", "nickel": "nickel", "uranium": "uranium", "platinoïdes": "platine",
                   "cobalt": "cobalt", "lithium": "lithium", "étain": "étain", "manganèse": "manganèse",
                   "molybdène": "molybdène", "aluminium / bauxite": "aluminium"}


def _last_by_series(prod: pd.DataFrame) -> dict[str, dict[str, tuple[int, float]]]:
    """installation → {série : (dernière année, tonnes)}."""
    p = prod[prod["value_tonnes"] > 0].sort_values("year", ascending=False)
    last = p.groupby(["facility_id", "serie"]).first()
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for (fid, serie), r in last.iterrows():
        out.setdefault(fid, {})[serie] = (int(r["year"]), float(r["value_tonnes"]))
    return out


def main_production(series: dict[str, tuple[int, float]] | None, groups: list[str]):
    """Série principale d'une mine : minerai extrait > minerai traité > charbon > métal de la commodité
    principale > métal le plus lourd. Retourne (série, année, tonnes) ou (None, None, NaN)."""
    if not series:
        return None, None, np.nan
    order = ["minerai extrait", "minerai traité", "charbon"]
    order += [f"{GROUP_TO_SERIES[g]} (métal)" for g in groups if g in GROUP_TO_SERIES]
    for s in order:
        if s in series:
            return s, *series[s]
    s = max(series, key=lambda k: series[k][1])
    return s, *series[s]


# -- caches liés au catalogue ---------------------------------------------------------------------
# catalog.load(), entities.from_catalog() et entities.load() sont mis en cache pour la vie du processus.
# Quand le catalogue ou sites.toml est réécrit par un autre processus (la CLI), il faut les vider,
# sinon le screener se reconstruit avec l'ancien catalogue et l'écrit comme « à jour » sur disque.
_sync_lock = threading.Lock()
_seen_inputs: tuple[int, int] | None = None
_catalog_hooks: list[Callable[[], None]] = []


def _sites_path():
    return files("strate.data").joinpath("sites.toml")


def on_catalog_change(fn: Callable[[], None]) -> Callable[[], None]:
    """Enregistre un vidage de cache à appeler quand le catalogue change (ex. index de recherche de l'API)."""
    _catalog_hooks.append(fn)
    return fn


def sync() -> None:
    """Vide les caches tirés du catalogue / de sites.toml si l'un des deux fichiers a changé.

    Au premier appel on vide aussi : impossible de savoir quand le cache déjà rempli a été lu.
    """
    global _seen_inputs
    key = (_mtime_ns(catalog.OUT / "mines.parquet"), _mtime_ns(_sites_path()))
    with _sync_lock:
        if key == _seen_inputs:
            return
        catalog.load.cache_clear()
        entities.load.cache_clear()
        entities.from_catalog.cache_clear()
        twins.cache_clear()
        for fn in _catalog_hooks:
            fn()
        _seen_inputs = key


def _km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Distance approchée (km), suffisante à l'échelle d'une dizaine de km."""
    dlon = (lons - lon + 180) % 360 - 180
    return np.hypot((lats - lat) * 111.2, dlon * 111.2 * math.cos(math.radians(lat)))


@cache
def twins() -> dict[str, str]:
    """Mine de la liste suivie → sa ligne « jumelle » du catalogue (même mine, identifiant ICMM/Wikidata).

    Candidats : la ligne au QID Wikidata cité comme source des coordonnées, et les lignes nommées à moins
    de TWIN_KM dont le nom (ou un alias) ressemble. Parmi eux, celle qui porte le plus de données
    (emprise, production réelle, fiche ICMM), puis la plus proche. Sans ce lien, la même mine apparaît
    deux fois dans le screener et sa jumelle « non analysée » est remise en file d'analyse satellite.
    """
    cat = catalog.load()
    if cat.empty:
        return {}
    named = cat[cat["name"].notna()]
    lats, lons = named["lat"].to_numpy(float), named["lon"].to_numpy(float)
    out: dict[str, str] = {}
    for s in entities.load().sites.values():
        d = _km(s.lat, s.lon, lats, lons)
        cand = {j for j in np.flatnonzero(d <= TWIN_KM)
                if catalog._similar(s.name, named["name"].iat[j])
                or catalog._similar(s.name, named["aliases"].iat[j])}
        q = re.search(r"wikidata\.org/wiki/(Q\d+)", s.coord_source)
        if q:
            cand |= set(np.flatnonzero((named["wikidata"] == q.group(1)).to_numpy()))
        cand -= {named.index.get_loc(t) for t in out.values()}  # une ligne n'a qu'un jumeau
        if not cand:
            continue
        best = max(cand, key=lambda j: (pd.notna(named["footprint_km2"].iat[j]),
                                        pd.notna(named["jasansky"].iat[j]),
                                        pd.notna(named["icmm"].iat[j]), -d[j]))
        out[s.id] = named.index[best]
    return out


# -- construction --------------------------------------------------------------------------------
EST_COLS = ["prod_estimee_type", "prod_estimee_t", "prod_estimee_bas90", "prod_estimee_haut90"]
TWIN_COLS = ["confiance", "emprise_km2", "avertissement", "prod_reelle_serie", "prod_reelle_annee",
             "prod_reelle_t"]


CATALOG_COLUMNS = ["id", "icmm", "name", "aliases", "lat", "lon", "country", "region", "asset_type",
                   "commodities", "confidence", "owner", "operator", "wikidata", "wikipedia", "jasansky",
                   "has_production", "production_status", "footprint_km2", "sources"]


def build() -> pd.DataFrame:
    stamp = _inputs_mtime_ns()  # avant de lire : une analyse écrite pendant la construction relancera
    sync()
    cat = catalog.load().copy()
    if cat.empty:  # catalogue pas encore construit (installation neuve) : seule la liste suivie apparaît
        cat = pd.DataFrame(columns=CATALOG_COLUMNS).astype(
            {"lat": float, "lon": float, "footprint_km2": float})
    g = entities.load()
    groups = cat["commodities"].map(commodity_groups)
    t = cat["asset_type"].fillna("").str.lower()
    processing = ~t.str.contains("mine") & t.str.contains("smelter|refinery|plant")
    categorie = np.select([cat["id"].str.startswith("fp_").values, processing.values,
                           cat["icmm"].notna().values], ["emprise sans nom", "usine", "mine active"],
                          "mine (Wikidata / Wikipedia)")
    est = [footprint_estimate(e, gr, c)
           for e, gr, c in zip(cat["footprint_km2"], groups, categorie, strict=True)]
    last = _last_by_series(catalog.all_production())
    mp = [main_production(last.get(j) if isinstance(j, str) else None, gr)
          for j, gr in zip(cat["jasansky"], groups, strict=True)]
    df = pd.DataFrame({
        "id": cat["id"].values,
        "nom": cat["name"].where(cat["name"].notna(), None).values,
        "pays": cat["country"].map(norm_country).values,
        "region": cat["region"].values,
        "type": cat["asset_type"].values,
        "categorie": categorie,
        "commodites": groups.map(", ".join).replace("", None).values,
        "commodite_principale": groups.map(lambda x: x[0] if x else None).values,
        "confiance": cat["confidence"].values,
        "proprietaire": cat["owner"].where(cat["owner"].notna(), cat["operator"]).values,
        "emprise_km2": cat["footprint_km2"].values,
        "avertissement": np.where(cat["footprint_km2"].fillna(0) > BIG_FOOTPRINT_KM2,
                                  "emprise très étendue : peut regrouper plusieurs mines ou de l'orpaillage",
                                  None),
        **{c: [e[i] for e in est] for i, c in enumerate(EST_COLS)},
        "prod_reelle_serie": [x[0] for x in mp],
        "prod_reelle_annee": pd.array([x[1] for x in mp], dtype="Int64"),
        "prod_reelle_t": [x[2] for x in mp],
        "wikipedia": cat["wikipedia"].values,
        "sources": cat["sources"].values,
        "autres_noms": cat["aliases"].values if "aliases" in cat else None,
        "lat": cat["lat"].values, "lon": cat["lon"].values,
    })
    # la liste suivie en tête de table, sous ses propres identifiants (ceux des analyses), avec
    # l'emprise, l'estimation et la production réelle de sa jumelle du catalogue, qui disparaît.
    tw = twins()
    by_id = df.set_index("id")
    groups_by_id = dict(zip(cat["id"], groups, strict=True))
    rows = []
    for s in g.sites.values():
        gr = commodity_groups(", ".join(s.commodity))
        r = {"id": s.id, "nom": s.name, "pays": norm_country(s.country), "region": s.region,
             "type": "Mine (liste suivie)", "categorie": "mine suivie",
             "commodites": ", ".join(gr) or None, "commodite_principale": (gr or [None])[0],
             "proprietaire": ", ".join(f"{g.companies[o.company].name} ({o.share:.0%})" for o in s.owners),
             "wikipedia": s.wikipedia, "sources": "liste suivie", "lat": s.lat, "lon": s.lon}
        twin = tw.get(s.id)
        if twin in by_id.index:
            tr = by_id.loc[twin]
            r |= {c: tr[c] for c in TWIN_COLS}
            # la jumelle disparaît de la table : ses noms et alias (« Mt Charlotte », « Cadia East »…)
            # restent cherchables sur la ligne suivie, sinon la mine est introuvable sous ces noms
            r["autres_noms"] = "; ".join(x for x in (cat.at[twin, "name"], cat.at[twin, "aliases"])
                                         if isinstance(x, str))
            gr = gr or groups_by_id[twin]
            r |= dict(zip(EST_COLS, footprint_estimate(tr["emprise_km2"], gr, "mine suivie"), strict=True))
        rows.append(r)
    wl = pd.DataFrame(rows).reindex(columns=df.columns)
    df = pd.concat([wl, df[~df["id"].isin(set(tw.values()))]], ignore_index=True)
    df["prod_reelle_annee"] = df["prod_reelle_annee"].astype("Int64")
    df["suivi"] = df["id"].isin(list(g.sites))
    # texte de recherche replié une fois ici, pas sur 80 000 lignes à chaque requête du screener
    df["recherche"] = (df["nom"].fillna("") + " " + df["proprietaire"].fillna("") + " "
                       + df["region"].fillna("") + " " + df["autres_noms"].fillna("")).map(_fold)
    df = add_analysis(df)
    df.attrs["version"] = TABLE_VERSION
    _write(df, stamp)
    return df


def _write(df: pd.DataFrame, stamp: int) -> None:
    """Écriture atomique (un lecteur ne voit jamais un parquet à moitié écrit), datée des entrées lues.

    Le fichier porte la date de la plus récente entrée vue AU DÉBUT de la construction : tout signal
    écrit pendant qu'on construisait est donc plus récent que la table, qui sera reconstruite.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=OUT.parent, prefix=".screener-", suffix=".parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.chmod(tmp, 0o644)  # mkstemp crée en 0600
        os.utime(tmp, ns=(stamp, stamp))
        os.replace(tmp, OUT)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- état des analyses ------------------------------------------------------------------------------
_last_cache: dict[str, tuple[int, pd.Timestamp]] = {}
_last_lock = threading.Lock()


def last_signal_dates(module_id: str = "optical") -> dict[str, pd.Timestamp]:
    """Mine → dernier mois de signaux d'un module. Chaque fichier n'est relu que s'il a changé.

    Un fichier illisible (en cours d'écriture) est ignoré, sans être mis en cache : relu au prochain appel.
    """
    d = DATA_DIR / "signals" / module_id
    out: dict[str, pd.Timestamp] = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.parquet"):
        mt, key = _mtime_ns(p), str(p)
        with _last_lock:
            hit = _last_cache.get(key)
        if hit is None or hit[0] != mt:
            try:
                last = pd.to_datetime(pd.read_parquet(p, columns=["date"])["date"]).max()
            except (OSError, ValueError, KeyError):
                continue
            hit = (mt, last)
            with _last_lock:
                _last_cache[key] = hit
        if pd.notna(hit[1]):
            out[p.stem] = hit[1]
    return out


def is_recent(last, today: date | None = None) -> bool:
    """Vrai si ce dernier mois de signaux date de moins de RECENT_DAYS (règle de fusion.radar)."""
    ref = pd.Timestamp(today or date.today())
    return pd.notna(last) and pd.Timestamp(last) >= ref - pd.Timedelta(days=RECENT_DAYS)


def analysis_state(site_id: str, today: date | None = None) -> dict:
    """État de l'analyse optique d'une mine, pour l'API.

    analyse : signaux récents · donnees_anciennes : des signaux, mais seulement anciens (calibration
    2016–2019) · derniere_date : « AAAA-MM » du dernier mois mesuré, ou None.
    """
    last = last_signal_dates().get(site_id)
    if last is None:
        return {"analyse": False, "donnees_anciennes": False, "derniere_date": None}
    rec = is_recent(last, today)
    return {"analyse": rec, "donnees_anciennes": not rec, "derniere_date": f"{last:%Y-%m}"}


def add_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes d'analyse pour les mines passées récemment au satellite (dernier mois, < 6 mois).

    Datée du jour (attrs « jour ») : la récence change avec le temps sans qu'aucun fichier ne change.
    """
    df = df.copy()
    for c in ("analyse", "niveau", "alerte", "mois_signal"):
        df[c] = None
    recent = {i for i, t in last_signal_dates().items() if is_recent(t)}
    df["analyse"] = df["id"].isin(recent)
    rad = fusion.radar()
    if len(rad):
        rad = rad.set_index("site")
        hit = df["id"].isin(rad.index)
        df.loc[hit, "niveau"] = df.loc[hit, "id"].map(rad["niveau"]).astype("Int64")
        df.loc[hit, "alerte"] = df.loc[hit, "id"].map(rad["explication"])
        df.loc[hit, "mois_signal"] = df.loc[hit, "id"].map(lambda i: f"{rad.loc[i, 'date']:%Y-%m}")
    df.attrs["jour"] = date.today().isoformat()
    return df


def _inputs_mtime_ns() -> int:
    """Date de la plus récente entrée de la table : catalogue, liste suivie, calibration, et TOUTES les
    familles de signaux (le niveau d'alerte vient de fusion.radar, qui lit optique, attention et radar).
    Les dossiers comptent aussi : supprimer un fichier de signaux change la date de son dossier."""
    paths = [catalog.OUT / "mines.parquet", _sites_path(), _cal_path()]
    sig = DATA_DIR / "signals"
    if sig.is_dir():
        paths += [sig, *(d for d in sig.iterdir() if d.is_dir()), *sig.glob("*/*.parquet")]
    return max((_mtime_ns(p) for p in paths), default=0)


_build_lock = threading.Lock()
_mem: tuple[tuple[int, int, int], pd.DataFrame] | None = None


def _fresh() -> pd.DataFrame | None:
    """La table sur disque si elle est à jour (lue une fois puis gardée en mémoire), sinon None."""
    global _mem
    sync()
    try:
        st = OUT.stat()
    except OSError:
        return None
    if st.st_mtime_ns < _inputs_mtime_ns():
        return None
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    if _mem is None or _mem[0] != key:
        df = pd.read_parquet(OUT)
        if df.attrs.get("version") != TABLE_VERSION:
            return None
        _mem = (key, df)
    df, today = _mem[1], date.today().isoformat()
    if df.attrs.get("jour", today) != today:
        # un nouveau jour : une analyse peut avoir cessé d'être récente sans qu'aucun fichier ne change.
        # Seules les colonnes d'analyse sont recalculées (quelques secondes), pas toute la table.
        df = add_analysis(df)
        _mem = (key, df)
    return df


def table(refresh: bool = False) -> pd.DataFrame:
    """Table du screener, reconstruite si le catalogue, la calibration ou les analyses ont changé.

    Partagée entre les appels : ne pas la modifier en place. Une seule reconstruction à la fois :
    l'interface interroge plusieurs routes en parallèle, et chaque reconstruction prend 10 à 20 s.
    """
    global _mem
    if not refresh and (df := _fresh()) is not None:
        return df
    with _build_lock:
        if not refresh and (df := _fresh()) is not None:  # un autre appel vient de la reconstruire
            return df
        df = build()
        st = OUT.stat()
        _mem = ((st.st_mtime_ns, st.st_size, st.st_ino), df)
        return df


def country_key(c) -> str:
    """Clé de comparaison d'un pays : nom canonique replié (casse, accents, alias, français)."""
    n = norm_country(c)
    return _fold(n) if n else ""


def country_mask(col: pd.Series, pays: list[str]) -> pd.Series:
    """Lignes dont le pays correspond à l'un des pays demandés, quelle que soit leur écriture."""
    wanted = {country_key(p) for p in pays} - {""}
    keys = {v: country_key(v) for v in col.dropna().unique()}  # ~200 pays : pas 80 000 appels
    return col.map(keys).isin(wanted)


def lon_mask(lon: pd.Series, w: float, e: float) -> pd.Series:
    """Filtre de longitude d'une vue de carte, antiméridien compris.

    Leaflet (worldCopyJump) ne ramène pas les bornes dans [-180, 180] : une vue centrée sur le détroit
    de Béring donne ouest=170, est=200. On ramène chaque borne dans [-180, 180) ; si l'ouest dépasse
    alors l'est, la vue traverse l'antiméridien. Une vue de 360° ou plus couvre toutes les longitudes.
    """
    if e - w >= 360:
        return pd.Series(True, index=lon.index)
    w, e = (w + 180) % 360 - 180, (e + 180) % 360 - 180
    lon = (lon + 180) % 360 - 180  # 180 et -180 sont la même longitude
    return lon.between(w, e) if w <= e else (lon >= w) | (lon <= e)


SORTABLE = {"nom", "pays", "commodite_principale", "emprise_km2", "prod_estimee_t", "prod_reelle_t",
            "prod_reelle_annee", "niveau", "categorie", "confiance"}


def query(df: pd.DataFrame, *, q: str | None = None, pays: list[str] | None = None,
          commodite: list[str] | None = None, categorie: list[str] | None = None,
          emprise_min: float | None = None, emprise_max: float | None = None,
          prod_min: float | None = None, prod_max: float | None = None,
          avec_emprise: bool = False, avec_prod_reelle: bool = False, analyse: bool | None = None,
          niveau_min: int | None = None, bbox: tuple[float, float, float, float] | None = None,
          sort: str = "prod_estimee_t", desc: bool = True) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if q:
        qf = _fold(q)
        if "recherche" in df.columns:  # nom, propriétaire, région et autres noms (jumelle comprise)
            hay = df["recherche"].fillna("")
        else:
            hay = (df["nom"].fillna("") + " " + df["proprietaire"].fillna("") + " "
                   + df["region"].fillna("")).map(_fold)
        m &= hay.str.contains(qf, regex=False)
    if pays:
        m &= country_mask(df["pays"], pays)
    if commodite:
        m &= df["commodites"].fillna("").map(lambda s: any(c in s.split(", ") for c in commodite))
    if categorie:
        m &= df["categorie"].isin(categorie)
    if emprise_min is not None:
        m &= df["emprise_km2"] >= emprise_min
    if emprise_max is not None:
        m &= df["emprise_km2"] <= emprise_max
    if prod_min is not None:
        m &= df["prod_estimee_t"] >= prod_min
    if prod_max is not None:
        m &= df["prod_estimee_t"] <= prod_max
    if avec_emprise:
        m &= df["emprise_km2"].notna()
    if avec_prod_reelle:
        m &= df["prod_reelle_t"].notna()
    if analyse is not None:
        m &= df["analyse"] == analyse
    if niveau_min is not None:
        m &= pd.to_numeric(df["niveau"], errors="coerce").fillna(-1) >= niveau_min
    if bbox:
        w, s, e, n = bbox
        m &= df["lat"].between(min(s, n), max(s, n)) & lon_mask(df["lon"], w, e)
    out = df[m]
    if sort not in SORTABLE:
        sort = "prod_estimee_t"
    return out.sort_values(sort, ascending=not desc, na_position="last", kind="stable")


def facets(df: pd.DataFrame) -> dict:
    com = df["commodites"].dropna().str.split(", ").explode().value_counts()
    return {
        "pays": df["pays"].value_counts().to_dict(),
        "commodites": com.to_dict(),
        "categories": df["categorie"].value_counts().to_dict(),
        "total": int(len(df)),
        "avec_emprise": int(df["emprise_km2"].notna().sum()),
        "avec_prod_estimee": int(df["prod_estimee_t"].notna().sum()),
        "avec_prod_reelle": int(df["prod_reelle_t"].notna().sum()),
        "analysees": int(df["analyse"].sum()),
    }
