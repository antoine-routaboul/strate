"""Sociétés minières : qui détient quelles mines, ce qu'elles déclarent, ce qu'elles annoncent (spec § 2).

Règles explicites seulement, aucun modèle de langage :

    entités    liste suivie (sites.toml) + propriétaires/exploitants du catalogue (Wikidata P127/P137, ICMM)
               dédoublonnées par nom normalisé puis par QID ; identifiants QID, CIK SEC, tickers
    faits      SEC companyfacts (us-gaap, ifrs-full, dei) + Wikidata P1128/P2139 datés (P585)
    annonces   titres de la veille (raw_docs) : montant + devise + verbe d'investissement → catégorie
    liens      corrélations r, ρ, IC 90 % bootstrap, n — par société et sur l'ensemble des sociétés

SEC : ≤ 8 requêtes/s (limite officielle 10), User-Agent lu dans STRATE_SEC_UA, sinon générique (jamais
d'adresse personnelle). Mise à jour bornée (`max_companies`) et reprenable (state.json).

Sorties (data/companies/) : companies, mines, facts, investments, correlations (.parquet), sec_tickers.json,
state.json, status.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from strate import entities
from strate.core import DATA_DIR

log = logging.getLogger(__name__)

OUT = DATA_DIR / "companies"
STATE = OUT / "state.json"
STATUS = OUT / "status.json"
TICKERS_CACHE = OUT / "sec_tickers.json"

DEFAULT_UA = "Strate/0.1 open-source research tool"
SEC_MIN_INTERVAL = 1 / 8  # ≤ 10 requêtes/s exigé par la SEC : marge de sécurité
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_TICKERS_URLS = ("https://www.sec.gov/files/company_tickers_exchange.json",
                    "https://www.sec.gov/files/company_tickers.json")
REFRESH_DAYS = 14  # comptes annuels : relire une société au plus toutes les deux semaines
RETRY_DAYS = 1
ABSENT_DAYS = 30  # companyfacts en 404 : la société ne publie pas de XBRL, inutile de redemander chaque jour
WIKIDATA_DAYS = 30
TICKERS_DAYS = 7
MAX_COMPANIES = 50
FIRST_YEAR = 2000
ANNUAL_FORMS = {"10-K", "10-K405", "10-KT", "20-F", "40-F"}
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSEAMERICAN", "AMEX", "NYSEARCA", "NYSEMKT", "OTC", "CBOE"}

METHODE_CORRELATIONS = (
    "Variations annuelles en log (année t vs t−1, exercices consécutifs, même concept comptable et même "
    "devise). Production : moyenne, pondérée par la part détenue (1 si inconnue), des variations de la série "
    "principale de chaque mine (Jasansky et al. 2023). Emprise : surface nue Sentinel-2 (médiane des mois "
    "clairs) sur les mêmes mines les deux années. Décalage k : CAPEX(t) ↔ y(t+k). r de Pearson, ρ de "
    "Spearman ; IC 90 % par bootstrap sur les sociétés (ensemble) ou sur les années (une société) ; "
    "n sociétés, n points. « Démontré » seulement si l'IC 90 % de ρ exclut 0.")

# -- graine de CIK vérifiés ---------------------------------------------------------------------------
# Grands miniers déposant à la SEC, CIK contrôlés sur data.sec.gov/submissions et company_tickers_exchange.
# Ils passent avant toute correspondance automatique (cotations jumelles : BHP Ltd / BHP plc).
SEED: list[dict] = [
    dict(name="Newmont", cik="0001164727", tickers=["NYSE:NEM"],
         aliases=["Newmont Mining", "Newmont Goldcorp", "Newmont Corporation"]),
    dict(name="Freeport-McMoRan", cik="0000831259", tickers=["NYSE:FCX"], aliases=["Freeport McMoRan Inc"]),
    dict(name="Barrick Mining", cik="0000756894", tickers=["NYSE:B"],
         aliases=["Barrick Gold", "Barrick Gold Corporation", "Barrick"]),
    dict(name="Agnico Eagle Mines", cik="0000002809", tickers=["NYSE:AEM"], aliases=["Agnico Eagle"]),
    dict(name="Kinross Gold", cik="0000701818", tickers=["NYSE:KGC"], aliases=[]),
    dict(name="BHP Group", cik="0000811809", tickers=["NYSE:BHP"],
         aliases=["BHP", "BHP Billiton", "BHP Billiton Limited", "BHP Group Ltd"]),
    dict(name="Rio Tinto plc", cik="0000863064", tickers=["NYSE:RIO"],
         aliases=["Rio Tinto", "Rio Tinto Group"]),
    dict(name="Vale", cik="0000917851", tickers=["NYSE:VALE"], aliases=["Vale S.A.", "CVRD"]),
    dict(name="Teck Resources", cik="0000886986", tickers=["NYSE:TECK"], aliases=["Teck Cominco", "Teck"]),
    dict(name="Southern Copper", cik="0001001838", tickers=["NYSE:SCCO"], aliases=[]),
    dict(name="Hecla Mining", cik="0000719413", tickers=["NYSE:HL"], aliases=[]),
    dict(name="Coeur Mining", cik="0000215466", tickers=["NYSE:CDE"], aliases=["Coeur d'Alene Mines"]),
    dict(name="Royal Gold", cik="0000085535", tickers=["NASDAQ:RGLD"], aliases=[]),
    dict(name="Wheaton Precious Metals", cik="0001323404", tickers=["NYSE:WPM"], aliases=["Silver Wheaton"]),
    dict(name="Franco-Nevada", cik="0001456346", tickers=["NYSE:FNV"], aliases=[]),
    dict(name="Cameco", cik="0001009001", tickers=["NYSE:CCJ"], aliases=[]),
    dict(name="Peabody Energy", cik="0001064728", tickers=["NYSE:BTU"], aliases=["Peabody"]),
    dict(name="Alcoa", cik="0001675149", tickers=["NYSE:AA"], aliases=[]),
    dict(name="Albemarle", cik="0000915913", tickers=["NYSE:ALB"], aliases=[]),
    dict(name="MP Materials", cik="0001801368", tickers=["NYSE:MP"], aliases=[]),
    dict(name="Gold Fields", cik="0001172724", tickers=["NYSE:GFI"], aliases=[]),
    dict(name="AngloGold Ashanti", cik="0001973832", tickers=["NYSE:AU"], aliases=["AngloGold Ashanti plc"]),
    dict(name="Harmony Gold", cik="0001023514", tickers=["NYSE:HMY"], aliases=["Harmony Gold Mining"]),
    dict(name="Sibanye-Stillwater", cik="0001786909", tickers=["NYSE:SBSW"], aliases=["Sibanye Gold"]),
    dict(name="Pan American Silver", cik="0000771992", tickers=["NASDAQ:PAAS"], aliases=[]),
    dict(name="Hudbay Minerals", cik="0001322422", tickers=["NYSE:HBM"], aliases=["Hudbay"]),
    dict(name="Cleveland-Cliffs", cik="0000764065", tickers=["NYSE:CLF"],
         aliases=["Cliffs Natural Resources"]),
    dict(name="Buenaventura", cik="0001013131", tickers=["NYSE:BVN"],
         aliases=["Compañía de Minas Buenaventura"]),
]

# -- noms de sociétés -----------------------------------------------------------------------------------
# Retirés EN FIN de nom : « Newmont Mining Corporation » et « Newmont » donnent la même clé. Les formes
# juridiques partent toujours (sauf le premier mot) ; UN SEUL mot de remplissage part ensuite, et seulement
# s'il reste un mot distinctif (ni pays, ni mot courant, ni terme géographique) : « New Gold » ne devient pas
# « new », « China Coal Energy » ne devient pas « china », « Mitsubishi Coal Mining » reste
# « mitsubishi coal » (distinct de « Mitsubishi Corporation »).
_LEGAL = {
    "inc", "incorporated", "ltd", "limited", "corp", "corporation", "co", "company", "plc", "sa", "sab", "cv",
    "de", "ag", "nv", "bv", "llc", "lp", "pty", "pt", "tbk", "gmbh", "spa", "sarl", "sas", "ab", "asa", "oy",
    "oyj", "jsc", "pjsc", "ojsc", "oao", "ao", "zao", "ooo", "sl", "srl", "kk", "bhd", "berhad", "nl", "fi",
    "cie", "cia", "the",
}
_FILLER = {"group", "holdings", "holding", "mining", "mines", "minerals", "resources", "energy", "gold",
           "silver", "copper", "coal", "metals", "international", "compania"}
_QID = re.compile(r"\s*(Q\d+)\s*")


def _split_words(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", s)  # « KG Mining (Bald Mountain), Inc » : la parenthèse n'est pas le nom
    s = entities.fold(s).replace("&", " and ")
    return re.findall(r"[a-z0-9]+", s)


_COUNTRY_WORDS: set[str] = set()


def _country_words() -> set[str]:
    """Mots des noms de pays connus (« korea », « japan », « china »…) : jamais distinctifs seuls."""
    if not _COUNTRY_WORDS:
        names = [*entities.COUNTRY_ALIASES, *map(entities.fold, entities.COUNTRY_ALIASES.values())]
        _COUNTRY_WORDS.update(w for n in names for w in re.findall(r"[a-z0-9]+", n))
    return _COUNTRY_WORDS


def _distinctive(t: str) -> bool:
    from strate import catalog

    return (len(t) >= 3 and not t.isdigit() and t not in _LEGAL | _FILLER and t not in _COMMON
            and t not in catalog._GENERIC and t not in _country_words())


def key_parts(name) -> tuple[str | None, str]:
    """(clé de dédoublonnage, mot de remplissage retiré ou « »)."""
    if not isinstance(name, str) or _QID.fullmatch(name):
        return None, ""
    w = _split_words(name)
    if w and w[0] == "the":
        w = w[1:]
    while len(w) > 1 and w[-1] in _LEGAL:
        w = w[:-1]
    filler = ""
    if len(w) > 1 and w[-1] in _FILLER and any(_distinctive(t) for t in w[:-1]):
        filler, w = w[-1], w[:-1]
    return " ".join(w) or None, filler


def norm_key(name) -> str | None:
    """Clé de dédoublonnage d'un nom de société (formes juridiques et un remplissage final retirés)."""
    return key_parts(name)[0]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60]


def id_base(name: str) -> str:
    """Identifiant lisible : le nom sans forme juridique finale (« Royal Gold Inc » → royal_gold)."""
    w = _split_words(name)
    while len(w) > 1 and w[-1] in _LEGAL:
        w = w[:-1]
    return slug(" ".join(w)) or "societe"


def _split_list(v) -> list[str]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if isinstance(x, str) and x]
    if not isinstance(v, str):
        return []
    return [x.strip() for x in v.split(";") if x.strip()]


def _unique(base: str, taken: set[str]) -> str:
    out, k = base, 2
    while out in taken:
        out, k = f"{base}_{k}", k + 1
    taken.add(out)
    return out


# -- entités ----------------------------------------------------------------------------------------------
COMPANY_COLS = ["id", "nom", "cle", "alias", "tickers", "cik", "cik_source", "qid", "pays", "sources",
                "suivie", "n_mines"]
LINK_COLS = ["company_id", "site_id", "nom", "role", "part", "source"]


def _twins() -> dict[str, str]:
    """Mine de la liste suivie → sa jumelle du catalogue (évite de compter deux fois la même mine)."""
    try:
        from strate import screener
        return screener.twins()
    except Exception as e:  # catalogue absent ou module en cours de modification
        log.warning("jumelles indisponibles (%s)", e)
        return {}


def build_entities(cat: pd.DataFrame | None = None, graph=None, twins: dict[str, str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sociétés, liens société ↔ mine). Ordre d'autorité : liste suivie (identifiants fixes, parts
    connues), graine SEC (alias de changement de nom), propriétaires/exploitants du catalogue."""
    from strate import catalog

    graph = graph or entities.load()
    if cat is None:
        cat = catalog.load()
    if twins is None:
        twins = _twins()
    canon = {c: w for w, c in twins.items()}  # ligne du catalogue → mine de la liste suivie
    comp: dict[str, dict] = {}
    by_key: dict[str, str] = {}
    by_strict: dict[str, str] = {}
    fillers: dict[str, set[str]] = {}  # société → mots de remplissage retirés de ses noms (« » : nom nu)
    by_qid: dict[str, str] = {}
    taken: set[str] = set()
    links: dict[tuple[str, str], dict] = {}

    def new(cid: str, name: str, **kw) -> dict:
        taken.add(cid)
        return comp.setdefault(cid, dict(id=cid, nom=name, alias=set(), tickers=[], cik=None, cik_source=None,
                                         qid=None, pays=None, sources=set(), suivie=False) | kw)

    def register(cid: str, name: str) -> None:
        k, f = key_parts(name)
        sk = strict_key(name)
        if sk:
            by_strict.setdefault(sk, cid)
        if k and by_key.setdefault(k, cid) == cid:
            fillers.setdefault(cid, set()).add(f)
        comp[cid]["alias"].add(name)

    def find(name: str) -> str | None:
        """Même nom strict, sinon même clé si les remplissages sont compatibles : « Equinox Resources »
        ne rejoint pas « Equinox Gold » (deux sociétés), « Newmont Mining » rejoint « Newmont »."""
        sk = strict_key(name)
        if sk in by_strict:
            return by_strict[sk]
        k, f = key_parts(name)
        cid = by_key.get(k) if k else None
        if cid is None:
            return None
        fs = fillers.get(cid, set())
        return cid if f == "" or f in fs or fs <= {""} else None

    def link(cid: str, site: str, role: str, part: float | None, source: str) -> None:
        site_name = None
        if site in graph.sites:
            site_name = graph.sites[site].name
        elif not cat.empty and site in cat.index:
            site_name = cat.at[site, "name"] if isinstance(cat.at[site, "name"], str) else None
        cur = links.get((cid, site))
        if cur is None:
            links[(cid, site)] = dict(company_id=cid, site_id=site, nom=site_name, role=role,
                                      part=np.nan if part is None else float(part), source=source)
        else:
            if role not in cur["role"].split(" + "):
                cur["role"] += f" + {role}"
            if part is not None and not np.isfinite(cur["part"]):
                cur["part"] = float(part)

    for c in graph.companies.values():
        e = new(c.id, c.name, suivie=True, tickers=list(c.tickers))
        e["sources"].add("liste suivie")
        register(c.id, c.name)
    for s in graph.sites.values():
        for o in s.owners:
            link(o.company, s.id, "propriétaire", o.share, "sites.toml")

    for sd in SEED:
        names = [sd["name"], *sd["aliases"]]
        cid = next((x for x in map(find, names) if x), None) or _unique(id_base(sd["name"]), taken)
        e = comp.get(cid) or new(cid, sd["name"])
        e["sources"].add("graine SEC")
        e["cik"], e["cik_source"] = e["cik"] or sd["cik"], e["cik_source"] or "graine vérifiée (SEC)"
        e["tickers"] = list(dict.fromkeys([*e["tickers"], *sd["tickers"]]))
        for n in names:
            register(cid, n)

    def named(raw: str, site: str, role: str) -> None:
        q = _QID.fullmatch(raw)
        if q:  # libellé Wikidata non résolu : l'entité est connue par son QID
            cid = by_qid.get(q.group(1))
            if cid is None:
                cid = _unique(q.group(1).lower(), taken)
                new(cid, q.group(1), qid=q.group(1))
                by_qid[q.group(1)] = cid
        else:
            cid = find(raw)
            if cid is None:
                if not norm_key(raw):
                    return
                cid = _unique(id_base(raw), taken)
                new(cid, raw)
            register(cid, raw)
        comp[cid]["sources"].add("catalogue")
        link(cid, canon.get(site, site), role, None, "catalogue (Wikidata P127/P137, ICMM)")

    if not cat.empty:
        sub = cat[cat["owner"].notna() | cat["operator"].notna()]
        for sid, own, opr in zip(sub.index, sub["owner"], sub["operator"], strict=True):
            for n in _split_list(own):
                named(n, sid, "propriétaire")
            for n in _split_list(opr):
                named(n, sid, "exploitant")

    lk = pd.DataFrame(list(links.values()), columns=LINK_COLS)
    n_mines = lk.groupby("company_id")["site_id"].nunique() if len(lk) else pd.Series(dtype=int)
    rows = [dict(id=e["id"], nom=e["nom"], cle=norm_key(e["nom"]), alias=";".join(sorted(e["alias"])),
                 tickers=";".join(e["tickers"]), cik=e["cik"], cik_source=e["cik_source"], qid=e["qid"],
                 pays=e["pays"], sources=";".join(sorted(e["sources"])), suivie=e["suivie"],
                 n_mines=int(n_mines.get(e["id"], 0))) for e in comp.values()]
    co = pd.DataFrame(rows, columns=COMPANY_COLS)
    return _sorted(co), lk


def _sorted(co: pd.DataFrame) -> pd.DataFrame:
    if co.empty:
        return co
    co = co.assign(_c=co["cik"].notna())
    co = co.sort_values(["suivie", "_c", "n_mines", "id"], ascending=[False, False, False, True])
    return co.drop(columns="_c").reset_index(drop=True)


def merge_by_qid(co: pd.DataFrame, lk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deux entités au même QID n'en font qu'une (« Barrick » du catalogue et « Barrick Mining ») : on garde
    la plus autoritaire (liste suivie, CIK, nombre de mines) et on lui rattache alias, tickers et mines."""
    if co.empty or co["qid"].notna().sum() == 0:
        return co, lk
    co = _sorted(co).copy()
    remap: dict[str, str] = {}
    for _, g in co[co["qid"].notna()].groupby("qid"):
        if len(g) < 2:
            continue
        keep = g.index[0]
        for i in g.index[1:]:
            remap[co.at[i, "id"]] = co.at[keep, "id"]
            for col in ("alias", "tickers", "sources"):
                co.at[keep, col] = ";".join(dict.fromkeys(
                    [*_split_list(co.at[keep, col]), *_split_list(co.at[i, col])]))
            for col in ("cik", "cik_source", "pays"):
                if not isinstance(co.at[keep, col], str) and isinstance(co.at[i, col], str):
                    co.at[keep, col] = co.at[i, col]
    if not remap:
        return co, lk
    co = co[~co["id"].isin(remap)].copy()
    lk = lk.assign(company_id=lk["company_id"].replace(remap))
    lk = lk.sort_values("part", na_position="last").drop_duplicates(["company_id", "site_id"])
    co["n_mines"] = co["id"].map(lk.groupby("company_id")["site_id"].nunique()).fillna(0).astype(int)
    return _sorted(co), lk.reset_index(drop=True)


# -- accès réseau ------------------------------------------------------------------------------------------
def sec_user_agent() -> str:
    """STRATE_SEC_UA si défini (contact choisi par l'utilisateur), sinon un en-tête générique."""
    return os.environ.get("STRATE_SEC_UA", "").strip() or DEFAULT_UA


class Http:
    """Client HTTP à débit limité (≤ 8 requêtes/s) qui respecte 429/Retry-After.

    Les tests passent un faux client qui expose get_json et sparql."""

    def __init__(self, ua: str | None = None, min_interval: float = SEC_MIN_INTERVAL):
        import httpx

        headers = {"User-Agent": ua or sec_user_agent(), "Accept-Encoding": "gzip, deflate"}
        self.c = httpx.Client(headers=headers, timeout=60, follow_redirects=True)
        self.min_interval, self._last, self._lock = min_interval, 0.0, threading.Lock()
        self.requests = 0
        self.sparql_requests = 0
        self.last_status: int | None = None  # code HTTP de la dernière réponse (None : aucune réponse)

    def _get(self, url: str):
        import httpx

        for attempt in range(5):
            with self._lock:
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            self.requests += 1
            try:
                r = self.c.get(url)
            except httpx.TransportError as e:
                log.warning("%s : %s, nouvel essai", url, type(e).__name__)
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("retry-after", "")
                time.sleep(min(float(ra) if ra.replace(".", "", 1).isdigit() else 5 * (attempt + 1), 120))
                continue
            return r
        return None

    def get_json(self, url: str) -> dict | None:
        r = self._get(url)
        self.last_status = None if r is None else r.status_code
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def sparql(self, query: str) -> list[dict]:
        from strate import catalog

        self.sparql_requests += 1
        return catalog._sparql(self.c, query)


# -- identifiants ----------------------------------------------------------------------------------------
def parse_sec_tickers(js) -> list[dict]:
    """company_tickers_exchange.json ({fields, data}) ou company_tickers.json ({"0": {cik_str…}})."""
    out = []
    if isinstance(js, dict) and "fields" in js and "data" in js:
        f = {k: i for i, k in enumerate(js["fields"])}
        for r in js["data"]:
            ex = str(r[f["exchange"]] or "").upper() if "exchange" in f else None
            out.append(dict(cik=str(r[f["cik"]]).zfill(10), nom=r[f["name"]],
                            ticker=str(r[f["ticker"]]).upper(), bourse=ex))
    elif isinstance(js, dict):
        for v in js.values():
            if isinstance(v, dict) and "cik_str" in v:
                out.append(dict(cik=str(v["cik_str"]).zfill(10), nom=v.get("title"),
                                ticker=str(v.get("ticker", "")).upper(), bourse=None))
    return out


def sec_tickers(http, force: bool = False) -> list[dict]:
    """Table ticker ↔ CIK de la SEC, gardée TICKERS_DAYS jours dans data/companies/sec_tickers.json."""
    cached = _read_json(TICKERS_CACHE)
    if cached.get("rows") and not force and _age_days(cached.get("at")) < TICKERS_DAYS:
        return cached["rows"]
    for url in SEC_TICKERS_URLS:
        rows = parse_sec_tickers(http.get_json(url))
        if rows:
            _write_json(TICKERS_CACHE, dict(at=_now(), url=url, rows=rows))
            return rows
    return cached.get("rows", [])


def strict_key(name) -> str | None:
    """Clé stricte (formes juridiques seules retirées, jamais « Resources », « Mining »…) : pour relier un
    nom au registre SEC, « Regis Resources » ne doit pas devenir « Regis Corp » (salons de coiffure)."""
    if not isinstance(name, str) or _QID.fullmatch(name):
        return None
    w = _split_words(re.sub(r"(\s*/[A-Za-z]{2,5}/)+\s*$", "", name))  # « NEWMONT Corp /DE/ »
    if w and w[0] == "the":
        w = w[1:]
    while len(w) > 1 and w[-1] in _LEGAL:
        w = w[:-1]
    return " ".join(w) or None


def ticker_index(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """(ticker → ligne, clé stricte de nom → ligne) ; un nom porté par plusieurs CIK n'est pas utilisé."""
    by_t, by_k, dup = {}, {}, set()
    for r in rows:
        by_t.setdefault(r["ticker"], r)
        k = strict_key(r.get("nom"))
        if not k:
            continue
        if k in by_k and by_k[k]["cik"] != r["cik"]:
            dup.add(k)
        by_k.setdefault(k, r)
    return by_t, {k: v for k, v in by_k.items() if k not in dup}


_SUFFIXES = ("", " Corporation", " Corp.", " Inc.", " Limited", " Ltd", " plc", " Group", " S.A.", " AG")
_LEGAL_TAIL = re.compile(r"[\s,]+(inc|incorporated|ltd|limited|corp|corporation|co|company|plc|s\.?a\.?|ag|"
                         r"n\.?v\.?|llc|pty|group|holdings)\.?$", re.I)


def name_variants(name: str) -> list[str]:
    """Écritures probables sur Wikidata (recherche exacte) : « Newmont Mining Corporation », « Newmont »…"""
    base = name.strip()
    while (b := _LEGAL_TAIL.sub("", base).strip(" ,")) and b != base:
        base = b
    return list(dict.fromkeys([name.strip(), *(base + s for s in _SUFFIXES)]))


def _qid_of(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _qnum(q: str) -> int:
    return int(q[1:]) if q[1:].isdigit() else 10**12


def wikidata_lookup(http, names: Iterable[str] = (), qids: Iterable[str] = (), batch: int = 300
                    ) -> dict[str, dict]:
    """Nom exact (libellé ou alias en/mul) ou QID → {qid, cik, pays, label}. Seuls comptent les éléments
    « société » : CIK, effectif, place de cotation, ISIN ou LEI (sinon la mine ou la rivière répondraient)."""
    out: dict[str, dict] = {}
    names = sorted({n for n in names if n and len(n) < 120})
    tail = """OPTIONAL { ?item wdt:P5531 ?cik } OPTIONAL { ?item wdt:P17 ?pays . ?pays wdt:P297 ?iso }"""
    for i in range(0, len(names), batch):
        vals = " ".join(f"{json.dumps(n, ensure_ascii=False)}@{lang}" for n in names[i:i + batch]
                        for lang in ("en", "mul"))
        q = (f"SELECT ?name ?item ?cik ?iso WHERE {{ VALUES ?name {{ {vals} }} "
             f"?item rdfs:label|skos:altLabel ?name . "
             f"FILTER EXISTS {{ ?item wdt:P5531|wdt:P1128|wdt:P414|wdt:P946|wdt:P1278 [] }} {tail} }}")
        for b in http.sparql(q):
            _keep_best(out, b["name"]["value"], b)
    qids = sorted({q for q in qids if q})
    for i in range(0, len(qids), batch):
        vals = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        q = (f"SELECT ?item ?cik ?iso ?name WHERE {{ VALUES ?item {{ {vals} }} {tail} "
             f'OPTIONAL {{ ?item rdfs:label ?name FILTER(LANG(?name)="en") }} }}')
        for b in http.sparql(q):
            _keep_best(out, _qid_of(b["item"]["value"]), b)
    return out


def _keep_best(out: dict, key: str, b: dict) -> None:
    """Plusieurs éléments pour un nom : celui qui a un CIK, puis le plus ancien (QID le plus petit)."""
    qid = _qid_of(b["item"]["value"])
    cik = b.get("cik", {}).get("value")
    iso = b.get("iso", {}).get("value")
    cand = dict(qid=qid, cik=cik.zfill(10) if cik and cik.isdigit() else None,
                pays=entities.norm_country(iso) if iso else None, label=b.get("name", {}).get("value"))
    cur = out.get(key)
    if cur is None or (cand["cik"] and not cur["cik"]) or (
            bool(cand["cik"]) == bool(cur["cik"]) and _qnum(qid) < _qnum(cur["qid"])):
        out[key] = cand
    elif cur["qid"] == qid and not cur["pays"]:
        cur["pays"] = cand["pays"]


def assign_ids(co: pd.DataFrame, wd: dict[str, dict], tickers: list[dict]) -> pd.DataFrame:
    """QID, pays et CIK : graine vérifiée > ticker US exact (SEC) > nom normalisé (SEC) > Wikidata P5531."""
    co = co.copy()
    by_t, by_k = ticker_index(tickers)
    for i, r in co.iterrows():
        hit = wd.get(r["qid"]) if isinstance(r["qid"], str) else None
        if hit is None:
            names = [v for n in [r["nom"], *_split_list(r["alias"])] for v in name_variants(n)]
            hit = next((wd[n] for n in names if n in wd), None)
        if hit:
            if not isinstance(r["qid"], str):
                co.at[i, "qid"] = hit["qid"]
            if not isinstance(r["pays"], str) and hit.get("pays"):
                co.at[i, "pays"] = hit["pays"]
            if _QID.fullmatch(str(r["nom"])) and hit.get("label"):
                co.at[i, "nom"], co.at[i, "cle"] = hit["label"], norm_key(hit["label"])
        if isinstance(r["cik"], str) and r["cik"]:
            continue
        found = None
        for t in _split_list(r["tickers"]):
            ex, _, sym = t.partition(":")
            row = by_t.get(sym.upper())
            us_row = row and (not row.get("bourse") or row["bourse"] in US_EXCHANGES)
            if ex.upper() in US_EXCHANGES and us_row:
                found = (row["cik"], f"SEC company_tickers (ticker {sym.upper()})")
                break
        keys = [k for k in map(strict_key, [r["nom"], *_split_list(r["alias"])]) if k in by_k]
        if found is None and keys:
            row = by_k[keys[0]]
            found = (row["cik"], f"SEC company_tickers (nom « {row['nom']} »)")
        if found is None and hit and hit.get("cik"):
            found = (hit["cik"], f"Wikidata P5531 ({hit['qid']})")
        if found:
            co.at[i, "cik"], co.at[i, "cik_source"] = found
    return co


# -- faits SEC (companyfacts) -------------------------------------------------------------------------------
# Concepts candidats par indicateur, par ordre de préférence. Par société, le concept couvrant le plus
# d'exercices gagne ; les autres ne bouchent que les années manquantes, dans la même unité.
CONCEPTS: dict[str, list[str]] = {
    "capex": [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets",
        "us-gaap:PaymentsToAcquireMiningAssets",
        "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "ifrs-full:PurchaseOfPropertyPlantAndEquipment",
        "ifrs-full:PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentProperty"
        "AndOtherNoncurrentAssets"],
    "chiffre_affaires": [
        "us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:RevenueMineralSales",
        "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet", "ifrs-full:Revenue",
        "ifrs-full:RevenueFromContractsWithCustomers"],
    "exploration": [
        "us-gaap:ExplorationExpenseMining", "us-gaap:ExplorationExpense",
        "us-gaap:ResultsOfOperationsExplorationExpense",
        "ifrs-full:ExpenseArisingFromExplorationForAndEvaluationOfMineralResources",
        "ifrs-full:CashFlowsFromUsedInExplorationForAndEvaluationOfMineralResourcesClassifiedAsInvesting"
        "Activities",
        "ifrs-full:PurchaseOfExplorationAndEvaluationAssets"],
    "actifs_miniers": [
        "us-gaap:PropertyPlantAndEquipmentNet",
        "us-gaap:PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        "us-gaap:MineralPropertiesNet",
        "ifrs-full:PropertyPlantAndEquipment",
        "ifrs-full:PropertyPlantAndEquipmentIncludingRightofuseAssets"],
    "effectifs": ["dei:EntityNumberOfEmployees", "ifrs-full:NumberOfEmployees",
                  "ifrs-full:AverageNumberOfEmployees"],
}
INSTANT = {"actifs_miniers"}  # postes de bilan : valeur à la clôture
BOTH = {"effectifs"}  # effectif à une date ou moyenne de l'exercice
ABS = {"capex", "exploration"}  # sorties de trésorerie parfois signées négativement
LABELS = {
    "capex": "CAPEX (investissements corporels payés)", "chiffre_affaires": "Chiffre d'affaires",
    "exploration": "Dépenses d'exploration", "actifs_miniers": "Actifs miniers (immobilisations nettes)",
    "effectifs": "Effectifs",
}
FACT_COLS = ["company_id", "indicateur", "annee", "valeur", "unite", "devise", "source", "formulaire",
             "concept", "fin_exercice", "depose", "accn", "cik", "origine"]


def filing_url(cik: str, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index.htm"


def _annual(x: dict) -> bool:
    return (x.get("form") or "").removesuffix("/A") in ANNUAL_FORMS


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _full_year(x: dict) -> bool:
    return "start" in x and 330 <= _days(x["start"], x["end"]) <= 400


def _units(facts: dict, concept: str, people: bool = False) -> tuple[str | None, list[dict]]:
    """(unité retenue, faits annuels). Monnaie : USD sinon la plus fournie ; effectifs : toutes les écritures
    (« Employee », « employees »… ) fusionnées."""
    ns, local = concept.split(":")
    units = facts.get("facts", {}).get(ns, {}).get(local, {}).get("units", {})
    if not units:
        return None, []
    if people:
        return "personnes", [x for xs in units.values() for x in xs if _annual(x)]
    unit = "USD" if "USD" in units else max(units, key=lambda u: len(units[u]))
    return unit, [x for x in units[unit] if _annual(x)]


def _closes(facts: dict) -> dict[str, list[str]]:
    """Dates de clôture déclarées par chaque dépôt annuel (fins d'exercices complets)."""
    out: dict[str, set[str]] = {}
    for ns in ("us-gaap", "ifrs-full"):
        for c in facts.get("facts", {}).get(ns, {}).values():
            for xs in c.get("units", {}).values():
                for x in xs:
                    if _annual(x) and _full_year(x):
                        out.setdefault(x["accn"], set()).add(x["end"])
    return {k: sorted(v) for k, v in out.items()}


def _year_of(x: dict, closes: dict[str, list[str]], instant_ok: bool) -> int | None:
    if "start" in x:
        return int(x["end"][:4]) if _full_year(x) else None
    ends = closes.get(x["accn"], [])
    if x["end"] in ends:
        return int(x["end"][:4])
    if not instant_ok:
        return None  # bilan trimestriel repris dans un rapport annuel : pas une clôture
    prior = [e for e in ends if e <= x["end"]]  # effectif daté après la clôture (page de garde)
    return int(prior[-1][:4]) if prior else int(x["end"][:4])


def parse_companyfacts(facts: dict, company_id: str, cik: str) -> pd.DataFrame:
    """Une valeur par (indicateur, exercice) : rapports annuels seulement, exercice complet (330–400 j) ou
    clôture, retraitement → dernier dépôt. Chaque ligne garde concept, unité, formulaire, dépôt."""
    closes = _closes(facts)
    rows = []
    for ind, cands in CONCEPTS.items():
        series = []
        for prio, concept in enumerate(cands):
            unit, xs = _units(facts, concept, people=ind == "effectifs")
            byyear: dict[int, dict] = {}
            for x in xs:
                if ind in INSTANT and "start" in x:
                    continue
                if ind not in INSTANT | BOTH and "start" not in x:
                    continue
                y = _year_of(x, closes, instant_ok=ind in BOTH)
                if y is None or y < FIRST_YEAR:
                    continue
                cur = byyear.get(y)
                if cur is None or (x["filed"], x["end"]) > (cur["filed"], cur["end"]):
                    byyear[y] = x
            if byyear:
                series.append((len(byyear), -prio, concept, unit, byyear))
        if not series:
            continue
        series.sort(reverse=True)
        main_unit, done = series[0][3], set()
        for _, _, concept, unit, byyear in series:
            if unit != main_unit:
                continue  # jamais deux devises dans une même série
            for y, x in sorted(byyear.items()):
                if y in done:
                    continue
                done.add(y)
                v = float(x["val"])
                money = unit != "personnes"
                rows.append(dict(company_id=company_id, indicateur=ind, annee=y,
                                 valeur=abs(v) if ind in ABS else v,
                                 unite="monnaie" if money else "personnes",
                                 devise=unit if money else None, source=filing_url(cik, x["accn"]),
                                 formulaire=x.get("form"), concept=concept, fin_exercice=x["end"],
                                 depose=x.get("filed"), accn=x["accn"], cik=cik, origine="sec"))
    df = pd.DataFrame(rows, columns=FACT_COLS)
    return df.sort_values(["indicateur", "annee"]).reset_index(drop=True)


# -- faits Wikidata -----------------------------------------------------------------------------------------
WD_PROPS = {"P1128": "effectifs", "P2139": "chiffre_affaires"}


def wikidata_facts(http, qid_to_company: dict[str, str], batch: int = 200) -> pd.DataFrame:
    """Effectifs (P1128) et chiffre d'affaires (P2139) datés par le qualificatif P585 ; sans date : ignoré.
    Devise : code ISO 4217 (P498) de l'unité. Plusieurs déclarations la même année : la plus récente."""
    rows = []
    qids = sorted(qid_to_company)
    for i in range(0, len(qids), batch):
        vals = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        q = (f"SELECT ?item ?prop ?amount ?date ?code WHERE {{ VALUES ?item {{ {vals} }} "
             f'VALUES (?prop ?p ?psv) {{ ("P1128" p:P1128 psv:P1128) ("P2139" p:P2139 psv:P2139) }} '
             f"?item ?p ?st . ?st ?psv ?vn . ?vn wikibase:quantityAmount ?amount . ?st pq:P585 ?date . "
             f"OPTIONAL {{ ?vn wikibase:quantityUnit ?u . ?u wdt:P498 ?code }} }}")
        for b in http.sparql(q):
            try:
                qid, prop = _qid_of(b["item"]["value"]), b["prop"]["value"]
                v, d = float(b["amount"]["value"]), b["date"]["value"].lstrip("+")
                y = int(d[:4])
            except (KeyError, ValueError):
                continue
            ind = WD_PROPS.get(prop)
            if ind is None or qid not in qid_to_company or y < FIRST_YEAR:
                continue
            money = ind != "effectifs"
            cur = b.get("code", {}).get("value") if money else None
            if money and not cur:
                continue  # montant sans devise connue : inutilisable
            rows.append(dict(company_id=qid_to_company[qid], indicateur=ind, annee=y, valeur=v,
                             unite="monnaie" if money else "personnes", devise=cur,
                             source=f"https://www.wikidata.org/wiki/{qid}#{prop}",
                             formulaire=f"Wikidata {prop}", concept=f"wikidata:{prop}", fin_exercice=d[:10],
                             depose=None, accn=None, cik=None, origine="wikidata"))
    df = pd.DataFrame(rows, columns=FACT_COLS)
    if df.empty:
        return df
    # une seule devise par série (la plus fréquente), une valeur par année (date la plus récente)
    main = df.groupby(["company_id", "indicateur"])["devise"].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else None)
    keep = [pd.isna(main.get((c, i))) or d == main.get((c, i))
            for c, i, d in zip(df["company_id"], df["indicateur"], df["devise"], strict=True)]
    df = df[keep].sort_values("fin_exercice")
    df = df.drop_duplicates(["company_id", "indicateur", "annee"], keep="last")
    return df.sort_values(["company_id", "indicateur", "annee"]).reset_index(drop=True)


def best_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Série retenue par (société, indicateur) : SEC si la société en déclare, sinon Wikidata (jamais un
    mélange des deux, qui juxtaposerait des périmètres et des devises différents)."""
    if facts.empty:
        return facts
    has_sec = set(map(tuple, facts.loc[facts["origine"] == "sec", ["company_id", "indicateur"]].to_numpy()))
    keep = [o == "sec" or (c, i) not in has_sec
            for c, i, o in zip(facts["company_id"], facts["indicateur"], facts["origine"], strict=True)]
    return facts[keep].reset_index(drop=True)


# -- investissements annoncés (veille) ---------------------------------------------------------------------
_MULT = {
    "thousand": 1e3, "k": 1e3, "mil": 1e3,
    "million": 1e6, "millions": 1e6, "mln": 1e6, "mn": 1e6, "m": 1e6, "mio": 1e6, "mio.": 1e6,
    "millon": 1e6, "millones": 1e6, "milhao": 1e6, "milhoes": 1e6, "me": 1e6, "m€": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "bln": 1e9, "milliard": 1e9, "milliards": 1e9,
    "md": 1e9, "mds": 1e9, "mrd": 1e9, "mrd.": 1e9, "milliarde": 1e9, "milliarden": 1e9,
    "mil millones": 1e9, "bilhao": 1e9, "bilhoes": 1e9, "trillion": 1e12, "crore": 1e7, "crores": 1e7,
    "lakh": 1e5,
}
_PREFIX_CUR = [
    (r"US\s?\$|U\.S\.\s?\$|USD\s?\$?", "USD"), (r"A\$|AU\$|AUD\s?\$?", "AUD"), (r"C\$|CA\$|CAD\s?\$?", "CAD"),
    (r"NZ\$|NZD", "NZD"), (r"R\$|BRL", "BRL"), (r"HK\$|HKD", "HKD"), (r"S\$|SGD", "SGD"),
    (r"MX\$|MXN", "MXN"), (r"€|EUR", "EUR"), (r"£|GBP", "GBP"), (r"¥|JPY", "JPY"), (r"₹|INR|Rs\.?", "INR"),
    (r"ZAR", "ZAR"), (r"CLP", "CLP"), (r"PEN|S/\.?", "PEN"), (r"CNY|RMB", "CNY"), (r"CHF", "CHF"),
    (r"\$", "$"),
]
_SUFFIX_CUR = [
    (r"(?:us|u\.s\.|american)\s+dollars?|d[oó]lares\s+(?:estadounidenses|americanos)|"
     r"dollars?\s+am[eé]ricains?|usd|us\$", "USD"),
    (r"australian\s+dollars?|aud|dollars?\s+australiens?", "AUD"),
    (r"canadian\s+dollars?|cad|dollars?\s+canadiens?", "CAD"),
    (r"dollars?|d[oó]lares|d[oó]lar", "$"), (r"euros?|eur|€", "EUR"),
    (r"pounds?|livres?\s+sterling|gbp|£", "GBP"),
    (r"rand|zar", "ZAR"), (r"reais|brl", "BRL"), (r"yuans?|renminbi|cny|rmb", "CNY"),
    (r"rupees?|roupies?|inr", "INR"), (r"soles|pen", "PEN"), (r"pesos\s+chilenos|clp", "CLP"),
    (r"pesos\s+mexicanos|mxn", "MXN"), (r"yens?|jpy", "JPY"), (r"roubles?|rubles?|rub", "RUB"),
    (r"tenge|kzt", "KZT"),
    (r"francs?\s+cfa|fcfa|xof|xaf", "XOF"),
]
_NUM = r"\d{1,3}(?:[.,   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_MULT_RE = (r"mil\s+millones|thousand|millions?|millones|mill[oó]n|milh[oõ]es|milh[aã]o|billions?|milliards?|"
            r"milliarden|milliarde|bilh[oõ]es|bilh[aã]o|trillion|crores?|lakh|mln|bln|mio\.?|mrd\.?|mds|"
            r"bn|mn|md|m|b|k")
_PRE = "|".join(f"(?:{p})" for p, _ in _PREFIX_CUR)
_SUF = "|".join(f"(?:{p})" for p, _ in _SUFFIX_CUR)
_AMOUNT = re.compile(
    rf"(?P<pre>{_PRE})\s?(?P<n1>{_NUM})[\s-]?(?P<m1>{_MULT_RE})?(?![a-z])"
    rf"|(?P<n2>{_NUM})[\s-]?(?P<m2>{_MULT_RE})\.?\s?(?:(?:de|d'|of)\s?)?(?P<suf>{_SUF})(?![a-z])"
    rf"|(?P<n3>{_NUM})\s?(?P<m3>{_MULT_RE})?\s?(?P<sym>€|£|\$)", re.I)
_PER_UNIT = re.compile(r"^\s*(?:/|per\b|a share|an? (?:ounce|tonne|ton|pound)|par\b|por\b|la (?:once|tonne)|"
                       r"pro\b|each)", re.I)
# verbes et noms d'investissement (texte replié : minuscules, sans accents)
INVEST_VERB = re.compile(
    r"\b(invest\w*|spend\w*|capex|capital (?:expenditure|spending|budget|program\w*)|allocat\w*|commit\w*|"
    r"earmark\w*|plough\w*|approv\w*|green[- ]?light\w*|to build|build\w*|construct\w*|expan\w*|"
    r"develop\w*|upgrad\w*|acqui\w*|buy\w*|"
    r"bought|purchas\w*|takeover|take over|merg\w*|bid\w*|"
    r"investi\w*|consacr\w*|debloqu\w*|inject\w*|rachat|rachet\w*|acquiert|fusion\w*|opa|construi\w*|"
    r"agrandi\w*|invers\w*|inviert\w*|invertir\w*|destinar\w*|destinara\w*|adquier\w*|adquisici\w*|compra\w*|"
    r"constru\w*|ampli\w*|aporta\w*|adquir\w*|aquisic\w*|fusao|investier\w*|investition\w*|ubernimm\w*|"
    r"ubernahme\w*|kauf\w*|baut|ausbau\w*)\b")
# chiffres de résultat, de dette ou de sanction près du montant : ce n'est pas un investissement
NOT_INVEST = re.compile(
    r"\b(revenues?|sales|profits?|earnings|net income|loss(?:es)?|ebitda|dividends?|fined?|fines|penalt\w*|"
    r"lawsuits?|sues?|damages|market (?:cap\w*|value)|valued|worth|debt|bonds?|loans?|royalt\w*|tax(?:es)?|"
    r"wages?|salar\w*|benefices?|chiffre d'affaires|pertes?|amendes?|dette|ganancias?|utilidad\w*|ingresos|"
    r"perdidas?|multas?|deuda|lucros?|receitas?|prejuizos?|dividas?|gewinn\w*|umsatz\w*|verlust\w*|strafe\w*|"
    r"schulden)\b")
# catégories, par ordre de priorité (la première qui correspond l'emporte)
CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("rachat/fusion", re.compile(
        r"\b(acqui\w*|takeover|take over|merg\w*|buyout|buys?|bought|purchas\w* (?:of )?(?:a )?(?:stake|"
        r"interest|shares)|stake|bid\w*|rachat|rachet\w*|acquiert|fusion\w*|opa|adquisici\w*|adquier\w*|"
        r"compra\w*|fusion|aquisic\w*|adquir\w*|fusao|ubernahme\w*|ubernimm\w*|kauf\w*)\b")),
    ("exploration", re.compile(
        r"\b(explor\w*|drill\w*|forage\w*|perforac\w*|sondag\w*|bohr\w*|prospect\w*)\b")),
    ("décarbonation/énergie", re.compile(
        r"\b(solar\w*|wind\w*|renewabl\w*|decarboni\w*|electrific\w*|batter\w*|green|hydrogen|emissions?|"
        r"power (?:plant|station|project)|low[- ]carbon|net[- ]zero|energie|solaire|eolien\w*|"
        r"renouvelable\w*|"
        r"decarbon\w*|energia|eolic\w*|renovable\w*|solares|erneuerbar\w*|windpark|photovolta\w*)\b")),
    ("usine/traitement", re.compile(
        r"\b(plant|mill|processing|concentrator|smelter|refiner\w*|leach\w*|flotation|tailings|usine\w*|"
        r"traitement|fonderie|raffinerie|planta|procesamiento|fundicion|refineria|usina|processamento|"
        r"beneficiamento|anlage|hutte|raffinerie|aufbereitung)\b")),
    ("infrastructure", re.compile(
        r"\b(rail\w*|port|ports|road\w*|pipeline\w*|water|desalinat\w*|transmission|camp|infrastructur\w*|"
        r"chemin de fer|route|ferrocarril|puerto|carretera|acueducto|desaladora|infraestructura|ferrovia|"
        r"porto|"
        r"estrada|infraestrutura|infrastruktur|hafen|bahn)\b")),
    ("expansion", re.compile(
        r"\b(expan\w*|extension|ramp[- ]?up|new mine|(?:stage|phase) (?:2|ii|two|3|iii|three)|develop\w*|"
        r"build\w*|construct\w*|underground|capacity|"
        r"life of mine|agrandi\w*|nouvelle mine|ampliac\w*|desarrollo|construc\w*|nueva mina|"
        r"expansao|desenvolvimento|nova mina|erweiterung|ausbau\w*)\b")),
]
_TLD_DOLLAR = {"ca": "CAD", "au": "AUD", "nz": "NZD", "hk": "HKD", "sg": "SGD", "mx": "MXN", "cl": "CLP",
               "ar": "ARS", "co": "COP"}
MIN_AMOUNT = 1e5  # sous 100 000 : prix unitaire, salaire, action…


def parse_number(s: str) -> float | None:
    """« 1,200 » → 1200 ; « 1,5 » → 1.5 ; « 2.75 » → 2.75 ; « 1 200 » → 1200 ; « 1.234.567 » → 1234567.
    Un séparateur unique suivi d'exactement trois chiffres est un séparateur de milliers."""
    s = s.replace(" ", " ").replace(" ", " ").strip()
    if re.fullmatch(r"\d{1,3}(?: \d{3})+", s):
        return float(s.replace(" ", ""))
    s = s.replace(" ", "")
    seps = [c for c in s if c in ".,"]
    try:
        if not seps:
            return float(s)
        if len(set(seps)) == 2:  # les deux : le dernier est le séparateur décimal
            dec = "," if s.rfind(",") > s.rfind(".") else "."
            th = "." if dec == "," else ","
            return float(s.replace(th, "").replace(dec, "."))
        sep = seps[0]
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:
            return float(s.replace(sep, ""))
        return float(s.replace(sep, "."))
    except ValueError:
        return None


def _currency(m: re.Match, domain: str | None) -> tuple[str, bool]:
    """(devise ISO, déduite ?) : « $ » seul prend la devise du domaine (.ca → CAD), sinon USD."""
    raw = m.group("pre") or m.group("suf") or m.group("sym") or ""
    code = None
    table = _PREFIX_CUR if m.group("pre") else _SUFFIX_CUR
    for p, c in table:
        if re.fullmatch(p, raw.strip(), re.I):
            code = c
            break
    if code is None:
        code = {"€": "EUR", "£": "GBP", "m€": "EUR", "md€": "EUR"}.get(raw.lower(), "$")
    if code != "$":
        return code, False
    tld = (domain or "").rsplit(".", 1)[-1].lower()
    return _TLD_DOLLAR.get(tld, "USD"), True


def extract_investment(title: str, domain: str | None = None) -> dict | None:
    """Montant + devise + verbe d'investissement dans un titre, et sa catégorie. None si l'une des trois
    pièces manque, si le montant est un prix unitaire ou < 100 000, ou si un mot de résultat / dette /
    sanction est collé au montant (« revenue of $2bn »)."""
    if not isinstance(title, str) or not title.strip():
        return None
    folded = entities.fold(title)
    if not INVEST_VERB.search(folded):
        return None
    for m in _AMOUNT.finditer(title):
        num = m.group("n1") or m.group("n2") or m.group("n3")
        mult_raw = (m.group("m1") or m.group("m2") or m.group("m3") or "").lower()
        v = parse_number(num)
        if v is None:
            continue
        key = entities.fold(mult_raw)
        mult = _MULT.get(key, _MULT.get(key.rstrip("."), 1.0)) if mult_raw else 1.0
        amount = v * mult
        if amount < MIN_AMOUNT or _PER_UNIT.search(title[m.end():m.end() + 16]):
            continue
        if not mult_raw and 1900 <= v <= 2100:
            continue  # une année
        near = entities.fold(title[max(0, m.start() - 30):m.end() + 30])
        if NOT_INVEST.search(near):
            continue
        verb = INVEST_VERB.search(folded)
        cat = next((c for c, rx in CATEGORIES if rx.search(folded)), "autre")
        devise, deduite = _currency(m, domain)
        return dict(montant=float(amount), devise=devise, devise_deduite=deduite, categorie=cat,
                    verbe=verb.group(1), extrait=m.group(0).strip())
    return None


# mots trop communs pour désigner seuls une société dans un titre
_COMMON = {
    "gold", "silver", "copper", "mining", "mines", "mine", "metals", "energy", "resources", "group", "first",
    "royal", "global", "national", "general", "united", "american", "anglo", "pacific", "atlantic", "golden",
    "premier", "capital", "new", "north", "south", "east", "west", "central", "international", "minerals",
    "iron", "coal", "nickel", "zinc", "lithium", "uranium", "state", "government", "city", "china", "india",
    "canada", "australia", "africa", "peru", "chile", "brazil", "mexico", "news", "world", "best", "vale",
    "orion", "apollo", "summit", "liberty", "victory", "phoenix", "eagle", "star", "sun", "sierra",
    "mountain",
}


def name_index(co: pd.DataFrame) -> dict[str, str]:
    """Clé de nom (nom + alias) → société. Clé d'un seul mot : ≥ 4 lettres et pas un mot courant."""
    from strate import catalog

    idx: dict[str, str] = {}
    for cid, nom, ali in zip(co["id"], co["nom"], co["alias"], strict=True):
        for n in [nom, *_split_list(ali)]:
            k = norm_key(n)
            if not k:
                continue
            w = k.split()
            if len(w) == 1 and (len(k) < 4 or k in _COMMON or k in catalog._GENERIC or k.isdigit()):
                continue
            idx.setdefault(k, cid)
    return idx


def match_companies(text: str, idx: dict[str, str], max_words: int = 5) -> list[str]:
    """Sociétés nommées dans un texte : n-grammes de mots repliés, le plus long d'abord, sans recouvrement."""
    w = _split_words(text or "")
    out, i = [], 0
    while i < len(w):
        for L in range(min(max_words, len(w) - i), 0, -1):
            cid = idx.get(" ".join(w[i:i + L]))
            if cid:
                if cid not in out:
                    out.append(cid)
                i += L
                break
        else:
            i += 1
    return out


INV_COLS = ["doc_id", "date", "montant", "devise", "devise_deduite", "categorie", "titre", "url", "domaine",
            "langue", "site_id", "company_id", "rattachement", "verbe", "extrait"]


def _veille_db() -> Path:
    try:
        from strate.veille import store
        return Path(store.DB_PATH)
    except Exception:  # module en cours de modification ailleurs : chemin par défaut
        return DATA_DIR / "veille" / "veille.db"


def _veille_query(sql: str, params: Iterable = (), db: Path | None = None) -> list[tuple]:
    p = db or _veille_db()
    if not p.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
        try:
            return con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()
    except sqlite3.Error as e:
        log.warning("veille illisible (%s)", e)
        return []


def investments(co: pd.DataFrame, lk: pd.DataFrame, db: Path | None = None) -> pd.DataFrame:
    """Annonces d'investissement extraites des titres de la veille, rattachées à la société nommée (titre ou
    organisations GDELT) et/ou à la mine (rattachement de la veille, puis ses propriétaires/exploitants)."""
    docs = _veille_query("SELECT id, published_at, fetched_at, url, domain, lang, title, organizations, "
                         "hint_site FROM raw_docs WHERE title IS NOT NULL AND status != 'rejected'", db=db)
    if not docs:
        return pd.DataFrame(columns=INV_COLS)
    ev: dict[str, list[str]] = {}
    for doc_id, site in _veille_query("SELECT doc_id, site_id FROM events", db=db):
        ev.setdefault(doc_id, []).append(site)
    idx = name_index(co) if len(co) else {}
    owners: dict[str, list[str]] = {}
    for cid, sid in (zip(lk["company_id"], lk["site_id"], strict=True) if len(lk) else []):
        owners.setdefault(sid, []).append(cid)
    rows = []
    for doc_id, pub, fetched, url, domain, lang, title, orgs, hint in docs:
        got = extract_investment(title, domain)
        if got is None:
            continue
        named = match_companies(title, idx)
        for o in _split_list(orgs):
            k = norm_key(o)
            if k in idx and idx[k] not in named:
                named.append(idx[k])
        sites = list(dict.fromkeys([*ev.get(doc_id, []), *([hint] if hint else [])]))
        targets: list[tuple[str | None, str | None, str]] = []
        for cid in named:
            s = next((x for x in sites if cid in owners.get(x, [])), None)
            targets.append((cid, s, "nom+mine" if s else "nom"))
        if not named:
            for s in sites:
                cs = owners.get(s, [])
                targets += [(c, s, "mine") for c in cs] or [(None, s, "mine")]
        for cid, sid, how in targets:
            rows.append(dict(doc_id=doc_id, date=(pub or fetched or "")[:10] or None, titre=title, url=url,
                             domaine=domain, langue=lang, site_id=sid, company_id=cid, rattachement=how,
                             **got))
    df = pd.DataFrame(rows, columns=INV_COLS)
    return df.drop_duplicates(["doc_id", "company_id", "site_id"]).sort_values("date", ascending=False) \
        .reset_index(drop=True)


# -- production et satellite par société -------------------------------------------------------------------
def _weights(lk: pd.DataFrame) -> pd.DataFrame:
    w = lk[["company_id", "site_id", "part"]].copy()
    w["w"] = w["part"].where(w["part"] > 0, 1.0).fillna(1.0)
    return w


def mine_production_growth(cat: pd.DataFrame, prod: pd.DataFrame, sites: Iterable[str],
                           twins: dict[str, str] | None = None) -> pd.DataFrame:
    """Variation log annuelle de la série principale de chaque mine (choix de screener.main_production,
    fixé une fois par mine pour ne jamais comparer minerai et métal), années consécutives seulement."""
    from strate import screener

    cols = ["site_id", "annee", "g"]
    if prod.empty or cat.empty or "jasansky" not in cat:
        return pd.DataFrame(columns=cols)
    twins = twins or {}
    by_fac = {k: g for k, g in prod[prod["value_tonnes"] > 0].groupby("facility_id")}
    rows = []
    for sid in set(sites):
        cid = twins.get(sid, sid)
        if cid not in cat.index or not isinstance(cat.at[cid, "jasansky"], str):
            continue
        p = by_fac.get(cat.at[cid, "jasansky"])
        if p is None:
            continue
        last = {s: (int(g["year"].iloc[-1]), float(g["value_tonnes"].iloc[-1]))
                for s, g in p.sort_values("year").groupby("serie")}
        serie, _, _ = screener.main_production(last, screener.commodity_groups(cat.at[cid, "commodities"]))
        v = p[p["serie"] == serie].groupby("year")["value_tonnes"].sum().sort_index()
        lg = np.log(v)
        for y in v.index:
            if y - 1 in v.index:
                rows.append(dict(site_id=sid, annee=int(y), g=float(lg[y] - lg[y - 1])))
    return pd.DataFrame(rows, columns=cols)


def company_growth(lk: pd.DataFrame, mine_g: pd.DataFrame, col: str = "g") -> pd.DataFrame:
    """Moyenne des variations des mines d'une société, pondérée par la part détenue (1 si inconnue)."""
    cols = ["company_id", "annee", "valeur", "n_mines"]
    if mine_g.empty or lk.empty:
        return pd.DataFrame(columns=cols)
    d = _weights(lk).merge(mine_g, on="site_id")
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["wg"] = d["w"] * d[col]
    g = d.groupby(["company_id", "annee"]).agg(s=("wg", "sum"), w=("w", "sum"),
                                                n_mines=("site_id", "nunique"))
    g["valeur"] = g["s"] / g["w"]
    return g.reset_index()[cols]


def _optical(site_id: str) -> pd.DataFrame:
    p = DATA_DIR / "signals" / "optical" / f"{site_id}.parquet"
    try:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except (OSError, ValueError):
        return pd.DataFrame()


def mine_footprint_growth(sites: Iterable[str], reader: Callable[[str], pd.DataFrame] = _optical,
                          min_quality: float = 0.5, min_months: int = 3) -> pd.DataFrame:
    """Variation log annuelle de la surface nue (médiane des mois clairs, ≥ 3 mois par an)."""
    rows = []
    for sid in set(sites):
        d = reader(sid)
        if d.empty or "signal" not in d:
            continue
        d = d[(d["signal"] == "bare_km2") & (d["quality"] >= min_quality) & (d["value"] > 0)]
        if d.empty:
            continue
        yr = pd.to_datetime(d["date"]).dt.year
        a = d.groupby(yr)["value"].agg(["median", "size"])
        a = a[a["size"] >= min_months]["median"]
        for y in a.index:
            if y - 1 in a.index:
                rows.append(dict(site_id=sid, annee=int(y), g=float(math.log(a[y] / a[y - 1]))))
    return pd.DataFrame(rows, columns=["site_id", "annee", "g"])


def fact_growth(facts: pd.DataFrame, indicateur: str) -> pd.DataFrame:
    """Variation log d'un indicateur (exercices consécutifs, même concept, valeurs > 0)."""
    cols = ["company_id", "annee", "valeur"]
    f = best_facts(facts)
    f = f[(f["indicateur"] == indicateur) & (f["valeur"] > 0)] if len(f) else f
    if f.empty:
        return pd.DataFrame(columns=cols)
    f = f.sort_values(["company_id", "annee"])
    g = f.groupby("company_id")
    ok = (f["annee"] - g["annee"].shift() == 1) & (f["concept"] == g["concept"].shift())
    out = f.assign(valeur=np.log(f["valeur"] / g["valeur"].shift()))[ok]
    return out[cols].reset_index(drop=True)


# -- corrélations ----------------------------------------------------------------------------------------
CORR_COLS = ["portee", "company_id", "x", "y", "decalage", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi",
             "n_societes", "n_points", "verdict", "calcule_le"]
MIN_POINTS_COMPANY = 5  # points d'une société pour sa propre corrélation
MIN_YEARS_POOLED = 3  # une société n'entre dans l'ensemble qu'avec au moins 3 années appariées


def _ci(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < 20:
        return np.nan, np.nan
    lo, hi = np.percentile(v, [5, 95])
    return float(lo), float(hi)


def _coefs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan, np.nan
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def corr_stats(d: pd.DataFrame, x: str = "x", y: str = "y", group: str | None = "company_id",
               n_boot: int = 1000, seed: int = 0) -> dict:
    """r de Pearson, ρ de Spearman et leurs IC 90 % par bootstrap : sur les groupes (sociétés) si `group`,
    sinon sur les points. Toujours avec n points et n groupes."""
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    n, k = len(d), (d[group].nunique() if group else 1)
    r, rho = _coefs(d[x].to_numpy(), d[y].to_numpy())
    res = dict(r=r, r_lo=np.nan, r_hi=np.nan, rho=rho, rho_lo=np.nan, rho_hi=np.nan, n_points=int(n),
               n_societes=int(k))
    if not np.isfinite(r):
        return res
    rng = np.random.default_rng(seed)
    rs, rhos = [], []
    if group and k >= 2:
        parts = [g[[x, y]].to_numpy() for _, g in d.groupby(group)]
        for _ in range(n_boot):
            b = np.concatenate([parts[i] for i in rng.integers(0, k, k)])
            a, c = _coefs(b[:, 0], b[:, 1])
            rs.append(a)
            rhos.append(c)
    elif not group:
        arr = d[[x, y]].to_numpy()
        for _ in range(n_boot):
            b = arr[rng.integers(0, n, n)]
            a, c = _coefs(b[:, 0], b[:, 1])
            rs.append(a)
            rhos.append(c)
    res["r_lo"], res["r_hi"] = _ci(rs)
    res["rho_lo"], res["rho_hi"] = _ci(rhos)
    return res


def verdict(s: dict, min_points: int, min_groups: int = 1) -> str:
    if s["n_points"] < min_points or s["n_societes"] < min_groups or not np.isfinite(s["rho"]):
        return f"trop peu de données (n = {s['n_points']}, sociétés = {s['n_societes']})"
    if not np.isfinite(s["rho_lo"]):
        return "intervalle incalculable : chiffres indicatifs"
    if s["rho_lo"] > 0:
        return "lien positif démontré (IC 90 % de ρ au-dessus de 0)"
    if s["rho_hi"] < 0:
        return "lien négatif démontré (IC 90 % de ρ sous 0)"
    return "non démontré : l'IC 90 % de ρ contient 0"


PAIRS = [  # (x, y, décalage) ; x et y sont des variations annuelles en log
    ("CAPEX", "production (mines détenues)", 0), ("CAPEX", "production (mines détenues)", 1),
    ("CAPEX", "production (mines détenues)", 2), ("effectifs", "production (mines détenues)", 0),
    ("CAPEX", "emprise satellite (surface nue)", 0), ("CAPEX", "emprise satellite (surface nue)", 1),
]


def correlate(facts: pd.DataFrame, lk: pd.DataFrame, *, cat: pd.DataFrame | None = None,
              prod: pd.DataFrame | None = None, twins: dict[str, str] | None = None,
              reader: Callable[[str], pd.DataFrame] = _optical, n_boot: int = 1000) -> pd.DataFrame:
    """Toutes les corrélations du § 2 : sur l'ensemble (portee = « ensemble ») et par société."""
    from strate import catalog

    run_at = _now()
    sites = set(lk["site_id"]) if len(lk) else set()
    if cat is None:
        cat = catalog.load()
    if prod is None:
        try:
            prod = catalog.all_production()
        except (OSError, KeyError, ValueError) as e:
            log.warning("production réelle indisponible (%s)", e)
            prod = pd.DataFrame()
    if twins is None:
        twins = _twins()
    series = {
        "CAPEX": fact_growth(facts, "capex"), "effectifs": fact_growth(facts, "effectifs"),
        "production (mines détenues)": company_growth(lk, mine_production_growth(cat, prod, sites, twins)),
        "emprise satellite (surface nue)": company_growth(lk, mine_footprint_growth(sites, reader)),
    }
    rows = []
    for xn, yn, lag in PAIRS:
        xs, ys = series[xn], series[yn]
        if xs.empty or ys.empty:
            d = pd.DataFrame(columns=["company_id", "annee", "x", "y"])
        else:
            d = xs.rename(columns={"valeur": "x"}).merge(
                ys.assign(annee=ys["annee"] - lag).rename(columns={"valeur": "y"})[
                    ["company_id", "annee", "y"]],
                on=["company_id", "annee"])
        cnt = d.groupby("company_id").size() if len(d) else pd.Series(dtype=int)
        pooled = d[d["company_id"].isin(cnt[cnt >= MIN_YEARS_POOLED].index)]
        s = corr_stats(pooled, n_boot=n_boot)
        rows.append(dict(portee="ensemble", company_id=None, x=xn, y=yn, decalage=lag, **s,
                         verdict=verdict(s, 8, 3), calcule_le=run_at))
        for cid in cnt[cnt >= MIN_POINTS_COMPANY].index:
            s = corr_stats(d[d["company_id"] == cid], group=None, n_boot=n_boot)
            rows.append(dict(portee="societe", company_id=cid, x=xn, y=yn, decalage=lag, **s,
                             verdict=verdict(s, MIN_POINTS_COMPANY), calcule_le=run_at))
    return pd.DataFrame(rows, columns=CORR_COLS)


# -- stockage ---------------------------------------------------------------------------------------------
def _path(name: str) -> Path:
    return OUT / f"{name}.parquet"


def load(name: str) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except (OSError, ValueError) as e:
        log.warning("%s illisible (%s)", p, e)
        return pd.DataFrame()


def _write(name: str, df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = _path(name).with_suffix(".tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(_path(name))


def _replace_facts(new: pd.DataFrame, *, company_id: str | None = None, origine: str) -> None:
    """Remplace les faits d'une origine (pour une société, ou pour toutes) ; le reste est conservé."""
    old = load("facts")
    if len(old) and set(FACT_COLS) <= set(old.columns):
        drop = old["origine"] == origine
        if company_id is not None:
            drop &= old["company_id"] == company_id
        old = old[~drop]
    else:
        old = pd.DataFrame(columns=FACT_COLS)
    parts = [x for x in (old, new) if len(x)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FACT_COLS)
    _write("facts", df.reindex(columns=FACT_COLS))


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text("utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_deep(d), ensure_ascii=False, indent=1, default=str), "utf-8")
    tmp.replace(p)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_days(stamp: str | None) -> float:
    if not stamp:
        return math.inf
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return math.inf
    return (datetime.now(UTC) - t).total_seconds() / 86400


# -- mise à jour ---------------------------------------------------------------------------------------------
def _carry_ids(co: pd.DataFrame) -> pd.DataFrame:
    """QID, pays et CIK trouvés lors d'une mise à jour précédente : gardés sans relire le réseau."""
    old = load("companies")
    if old.empty or not {"id", "qid", "cik", "pays"} <= set(old.columns):
        return co
    prev = old.drop_duplicates("id").set_index("id")
    co = co.copy()
    for i, cid in zip(co.index, co["id"], strict=True):
        if cid not in prev.index:
            continue
        for col in ("qid", "pays"):
            if not isinstance(co.at[i, col], str) and isinstance(prev.at[cid, col], str):
                co.at[i, col] = prev.at[cid, col]
        # CIK : seul celui venu de Wikidata (relu une fois par mois) est repris ; graine et registre SEC
        # sont recalculés à chaque fois, une règle corrigée s'applique donc tout de suite
        src = prev.at[cid, "cik_source"]
        if not isinstance(co.at[i, "cik"], str) and isinstance(src, str) and src.startswith("Wikidata"):
            co.at[i, "cik"], co.at[i, "cik_source"] = prev.at[cid, "cik"], src
    return co


def sec_queue(co: pd.DataFrame, state: dict, force: bool = False) -> list[str]:
    """Sociétés à lire à la SEC, dans l'ordre : jamais lues, puis les plus anciennes (priorité du tri)."""
    per = state.get("sec", {})
    todo = []
    for cid, cik in zip(co["id"], co["cik"], strict=True):
        if not isinstance(cik, str):
            continue
        st = per.get(cid, {})
        # un échec est retenté plus tôt, sauf un 404 (pas de companyfacts publiés pour ce CIK)
        wait = REFRESH_DAYS if st.get("ok") else ABSENT_DAYS if st.get("http") == 404 else RETRY_DAYS
        if force or st.get("cik") != cik or _age_days(st.get("at")) >= wait:
            todo.append(cid)
    return sorted(todo, key=lambda c: per.get(c, {}).get("at") or "")


def refresh(max_companies: int = MAX_COMPANIES, *, http=None, force: bool = False, wikidata: bool = True,
            cat: pd.DataFrame | None = None, graph=None, twins: dict[str, str] | None = None,
            veille_db: Path | None = None, with_correlations: bool = True, n_boot: int = 1000,
            prod: pd.DataFrame | None = None, reader: Callable[[str], pd.DataFrame] | None = None,
            say: Callable[[str], None] = log.info) -> dict:
    """Met à jour data/companies/ : entités, identifiants, faits SEC (au plus `max_companies` sociétés par
    appel, chacune écrite dès qu'elle est lue : un arrêt ne perd rien et l'appel suivant reprend), faits
    Wikidata, investissements annoncés, corrélations. Retourne le statut écrit dans status.json."""
    from strate import catalog

    http = http or Http()
    state = _read_json(STATE)
    errors: list[str] = []
    if cat is None:
        cat = catalog.load()
    if twins is None:
        twins = _twins()
    co, lk = build_entities(cat, graph, twins)
    co = _carry_ids(co)

    tick = []
    try:
        tick = sec_tickers(http, force=force)
    except Exception as e:  # réseau : on garde les CIK déjà connus
        errors.append(f"SEC company_tickers : {type(e).__name__}")
    wd: dict[str, dict] = {}
    if wikidata and (force or _age_days(state.get("wikidata_ids_at")) >= WIKIDATA_DAYS):
        pick = co[(co["n_mines"] >= 2) | co["suivie"] | co["cik"].notna() | co["qid"].notna()]
        names = {v for n, a in zip(pick["nom"], pick["alias"], strict=True)
                 for x in [n, *_split_list(a)[:2]] if not _QID.fullmatch(x) for v in name_variants(x)}
        try:
            wd = wikidata_lookup(http, names, [q for q in co["qid"] if isinstance(q, str)])
            state["wikidata_ids_at"] = _now()
            say(f"Wikidata : {len(wd)} correspondances pour {len(names)} noms")
        except Exception as e:
            errors.append(f"Wikidata (identifiants) : {type(e).__name__}")
    co = assign_ids(co, wd, tick)
    co, lk = merge_by_qid(co, lk)
    _write("companies", co)
    _write("mines", lk)
    _write_json(STATE, state)

    queue = sec_queue(co, state, force)
    per = state.setdefault("sec", {})
    ciks = dict(zip(co["id"], co["cik"], strict=True))
    names = dict(zip(co["id"], co["nom"], strict=True))
    done = tried = 0
    for cid in queue[:max(0, max_companies)]:
        cik = ciks[cid]
        tried += 1
        try:
            js = http.get_json(SEC_FACTS_URL.format(cik=cik))
            code = getattr(http, "last_status", None)
            if not isinstance(js, dict) or "facts" not in js:
                # lecture ratée (403, 404, 5 échecs, JSON inattendu) : les faits déjà stockés restent
                # intacts, l'échec est signalé et la société sera relue plus tard
                why = f"HTTP {code}" if code else "pas de réponse exploitable"
                per[cid] = dict(at=_now(), cik=cik, ok=False, http=code,
                                n_faits=int(per.get(cid, {}).get("n_faits", 0) or 0),
                                message=f"companyfacts illisible ({why}) : faits précédents conservés")
                errors.append(f"{cid} : companyfacts illisible ({why})")
                say(f"{names[cid]} : {per[cid]['message']}")
                _write_json(STATE, state)
                continue
            fa = parse_companyfacts(js, cid, cik)
            _replace_facts(fa, company_id=cid, origine="sec")
            per[cid] = dict(at=_now(), cik=cik, ok=True, http=code, n_faits=int(len(fa)),
                            message=f"{fa['annee'].nunique()} exercices")
            say(f"{names[cid]} : {per[cid]['message']}")
            done += 1
        except Exception as e:  # une société en échec n'arrête pas les autres
            log.exception("société %s", cid)
            per[cid] = dict(at=_now(), cik=cik, ok=False, message=f"{type(e).__name__}: {e}"[:300])
            errors.append(f"{cid} : {type(e).__name__}")
        _write_json(STATE, state)

    if wikidata and (force or _age_days(state.get("wikidata_facts_at")) >= WIKIDATA_DAYS):
        q2c = {q: c for c, q in zip(co["id"], co["qid"], strict=True) if isinstance(q, str)}
        try:
            _replace_facts(wikidata_facts(http, q2c), origine="wikidata")
            state["wikidata_facts_at"] = _now()
        except Exception as e:
            errors.append(f"Wikidata (faits) : {type(e).__name__}")
        _write_json(STATE, state)

    inv = investments(co, lk, veille_db)
    _write("investments", inv)
    facts = load("facts")
    facts = facts if set(FACT_COLS) <= set(facts.columns) else pd.DataFrame(columns=FACT_COLS)
    if with_correlations:
        _write("correlations", correlate(facts, lk, cat=cat, prod=prod, twins=twins, n_boot=n_boot,
                                         reader=reader or _optical))
    rest = len(sec_queue(co, state))
    st = dict(
        mis_a_jour=_now(), user_agent_personnalise=bool(os.environ.get("STRATE_SEC_UA", "").strip()),
        n_societes=int(len(co)), n_avec_cik=int(co["cik"].notna().sum()),
        n_avec_qid=int(co["qid"].notna().sum()),
        n_avec_mines=int((co["n_mines"] > 0).sum()), n_liens_mines=int(len(lk)),
        sec_lues_ce_lancement=done, sec_tentees_ce_lancement=tried, sec_restantes=rest,
        n_societes_avec_faits=int(facts["company_id"].nunique()) if len(facts) else 0,
        n_faits=int(len(facts)), n_investissements=int(len(inv)),
        requetes_sec=getattr(http, "requests", None),
        requetes_wikidata=getattr(http, "sparql_requests", None),
        erreurs=errors[-50:],
        message=(f"{rest} société(s) SEC restent à lire : relancer « strate companies refresh »" if rest
                 else "à jour"))
    _write_json(STATUS, st)
    return _deep(st)


# -- lecture (API) ------------------------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _deep(v):
    """Nettoyage récursif pour un JSON strict (NaN → None, types numpy → Python)."""
    if isinstance(v, dict):
        return {k: _deep(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep(x) for x in v]
    return _clean(v)


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def _new_format(df: pd.DataFrame, cols: list[str]) -> bool:
    return not df.empty and set(cols) <= set(df.columns)


def _last(f: pd.DataFrame, ind: str) -> pd.DataFrame:
    """Dernière valeur par société d'un indicateur (série retenue)."""
    x = f[f["indicateur"] == ind].sort_values("annee")
    return x.groupby("company_id").tail(1).set_index("company_id") if len(x) else pd.DataFrame()


TABLE_COLS = ["id", "nom", "pays", "tickers", "cik", "qid", "n_mines", "capex_dernier", "capex_annee",
              "effectifs", "effectifs_annee", "ca_dernier", "devise", "investissements_12m"]


def table(today: date | None = None) -> pd.DataFrame:
    """Une ligne par société (champs de GET /api/companies). Vide tant que refresh() n'a pas tourné."""
    co = load("companies")
    if not _new_format(co, COMPANY_COLS):
        return pd.DataFrame(columns=TABLE_COLS)
    facts = load("facts")
    f = best_facts(facts) if _new_format(facts, FACT_COLS) else pd.DataFrame(columns=FACT_COLS)
    capex, emp, ca = _last(f, "capex"), _last(f, "effectifs"), _last(f, "chiffre_affaires")
    inv = load("investments")
    since = ((today or date.today()) - timedelta(days=365)).isoformat()
    n12 = (inv[inv["date"].fillna("") >= since].groupby("company_id").size()
           if _new_format(inv, INV_COLS) else pd.Series(dtype=int))

    def get(df, cid, col):
        return df.at[cid, col] if len(df) and cid in df.index else None

    rows = []
    for r in co.itertuples(index=False):
        cur = get(capex, r.id, "devise") or get(ca, r.id, "devise")
        rows.append(dict(id=r.id, nom=r.nom, pays=r.pays, tickers=_split_list(r.tickers), cik=r.cik,
                         qid=r.qid,
                         n_mines=int(r.n_mines), capex_dernier=get(capex, r.id, "valeur"),
                         capex_annee=get(capex, r.id, "annee"), effectifs=get(emp, r.id, "valeur"),
                         effectifs_annee=get(emp, r.id, "annee"), ca_dernier=get(ca, r.id, "valeur"),
                         devise=cur, investissements_12m=int(n12.get(r.id, 0))))
    out = pd.DataFrame(rows, columns=TABLE_COLS)
    for c in ("capex_annee", "effectifs_annee"):
        out[c] = out[c].astype("Int64")
    return out


def company(company_id: str) -> dict | None:
    """Fiche d'une société (champs de GET /api/company) ; None si inconnue."""
    co = load("companies")
    if not _new_format(co, COMPANY_COLS) or company_id not in set(co["id"]):
        return None
    r = co[co["id"] == company_id].iloc[0]
    lk, facts, inv, corr = load("mines"), load("facts"), load("investments"), load("correlations")
    m = lk[lk["company_id"] == company_id].sort_values(["part", "site_id"], ascending=[False, True]) \
        if _new_format(lk, LINK_COLS) else pd.DataFrame(columns=LINK_COLS)
    f = facts[facts["company_id"] == company_id] if _new_format(facts, FACT_COLS) \
        else pd.DataFrame(columns=FACT_COLS)
    retenu = set(map(tuple, best_facts(f)[["indicateur", "annee", "origine"]].to_numpy())) \
        if len(f) else set()
    f = f.sort_values(["indicateur", "annee", "origine"])
    i = inv[inv["company_id"] == company_id] if _new_format(inv, INV_COLS) else pd.DataFrame(columns=INV_COLS)
    c = corr[(corr["portee"] == "societe") & (corr["company_id"] == company_id)] \
        if _new_format(corr, CORR_COLS) else pd.DataFrame(columns=CORR_COLS)
    cik, qid = _clean(r["cik"]), _clean(r["qid"])
    profil = dict(
        id=r["id"], nom=r["nom"], alias=_split_list(r["alias"]), tickers=_split_list(r["tickers"]), cik=cik,
        cik_source=_clean(r["cik_source"]), qid=qid, pays=_clean(r["pays"]),
        sources=_split_list(r["sources"]),
        suivie=bool(r["suivie"]), n_mines=int(r["n_mines"]),
        sec_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}" if cik else None,
        wikidata_url=f"https://www.wikidata.org/wiki/{qid}" if qid else None)
    faits = [dict(indicateur=x.indicateur, libelle=LABELS.get(x.indicateur, x.indicateur), annee=int(x.annee),
                  valeur=_clean(x.valeur), unite=x.unite, devise=_clean(x.devise), source=x.source,
                  formulaire=_clean(x.formulaire), concept=x.concept, fin_exercice=_clean(x.fin_exercice),
                  origine=x.origine, retenu=(x.indicateur, x.annee, x.origine) in retenu)
             for x in f.itertuples(index=False)]
    return _deep(dict(
        profil=profil,
        mines=[dict(site_id=x.site_id, nom=_clean(x.nom), part=_clean(x.part), role=x.role, source=x.source)
               for x in m.itertuples(index=False)],
        faits=faits,
        investissements=[dict(date=x.date, montant=x.montant, devise=x.devise, categorie=x.categorie,
                              titre=x.titre, url=x.url, site_id=_clean(x.site_id),
                              rattachement=x.rattachement)
                         for x in i.itertuples(index=False)],
        correlations=[{k: _clean(row[k]) for k in CO_CORR_KEYS} | {"n": int(row["n_points"])}
                      for row in c.to_dict("records")],
    ))


CO_CORR_KEYS = ("x", "y", "decalage", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi", "n_points", "verdict")


def correlations() -> dict:
    """Corrélations sur l'ensemble des sociétés (champs de GET /api/companies/correlations)."""
    c = load("correlations")
    rows = []
    if _new_format(c, CORR_COLS):
        rows = _records(c[c["portee"] == "ensemble"].drop(columns=["portee", "company_id"]))
    return dict(rows=rows, methode=METHODE_CORRELATIONS)


def status() -> dict:
    return _deep(_read_json(STATUS))
