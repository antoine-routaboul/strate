"""Couche sociétés : qui exploite les mines, combien elles investissent, où, et est-ce visible du ciel.

Chaîne (règles explicites, aucun modèle de langage) :

    univers (liste suivie + propriétaires/exploitants du catalogue + veille)
      └─▶ identifiants (CIK : graine vérifiée, SEC company_tickers, Wikidata P5531)
            └─▶ comptes annuels SEC companyfacts (us-gaap / ifrs-full) + effectifs
                  └─▶ CAPEX par segment (instance XBRL) + investissements annoncés (veille)
                        └─▶ ratios, corrélations (r, ρ, n, IC 90 % bootstrap par société, verdict), prévision

Accès SEC : data.sec.gov (companyfacts, submissions) répond avec un User-Agent générique ; les archives
www.sec.gov (rapports annuels, instances XBRL, company_tickers.json) exigent un contact déclaré, lu
UNIQUEMENT dans STRATE_SEC_UA ou config.local.toml ([sec] user_agent). Sans contact, effectifs par le
texte et CAPEX par segment sont sautés, et le statut le dit. Débit : 8 requêtes/s au plus, 429 respecté.

Sorties (data/companies/, non versionné) : companies, financials, employees, segments, correlations
(.parquet) + status.json ; state.json garde l'avancement pour reprendre une mise à jour interrompue.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from strate import entities
from strate.core import DATA_DIR

log = logging.getLogger(__name__)

OUT = DATA_DIR / "companies"
TABLES = ("companies", "financials", "employees", "segments", "correlations")
STATUS = OUT / "status.json"
STATE = OUT / "state.json"
CONFIG = Path(__file__).resolve().parents[2] / "config.local.toml"

GENERIC_UA = "Strate open-source research tool"
SEC_MIN_INTERVAL = 1 / 8  # fair access SEC : 8 requêtes/s au plus (limite officielle : 10)
REFRESH_DAYS = 7  # une société relue il y a moins d'une semaine n'est pas retéléchargée
WIKIDATA_REFRESH_DAYS = 30
FIRST_YEAR = 2005
ANNUAL_FORMS = {"10-K", "10-K405", "10-KT", "20-F", "40-F"}
TEXT_FILINGS = 3  # rapports annuels lus pour les effectifs (les plus récents)
SEGMENT_FILINGS = 2  # instances XBRL lues pour le CAPEX par segment

# -- graine de CIK vérifiés ---------------------------------------------------------------------------
# Grands miniers déposant à la SEC. Chaque CIK a été vérifié une fois sur data.sec.gov/submissions
# (nom déclaré et dernier rapport annuel). Écartés faute de CIK confirmé : B2Gold, Fortuna, Centrus.
SEED_SOURCE = "CIK vérifié sur https://data.sec.gov/submissions/ le 2026-11-27"
SEED: list[dict] = [
    dict(name="Newmont", cik="0001164727", tickers=["NYSE:NEM"],
         aliases=["Newmont Mining", "Newmont Goldcorp", "Newmont Corporation"]),
    dict(name="Freeport-McMoRan", cik="0000831259", tickers=["NYSE:FCX"], aliases=["Freeport McMoRan Inc"]),
    dict(name="Barrick Mining", cik="0000756894", tickers=["NYSE:B"],
         aliases=["Barrick Gold", "Barrick Gold Corporation", "Barrick"]),
    dict(name="Agnico Eagle Mines", cik="0000002809", tickers=["NYSE:AEM"], aliases=["Agnico Eagle"]),
    dict(name="Kinross Gold", cik="0000701818", tickers=["NYSE:KGC"], aliases=[]),
    dict(name="BHP Group", cik="0000811809", tickers=["NYSE:BHP"],
         aliases=["BHP", "BHP Billiton", "BHP Billiton Limited"]),
    dict(name="BHP Group Plc", cik="0001171264", tickers=[], aliases=["BHP Billiton plc"]),
    dict(name="Rio Tinto plc", cik="0000863064", tickers=["NYSE:RIO"],
         aliases=["Rio Tinto", "Rio Tinto Group"]),
    dict(name="Rio Tinto Limited", cik="0000887028", tickers=[], aliases=[]),
    dict(name="Vale", cik="0000917851", tickers=["NYSE:VALE"], aliases=["Vale S.A.", "CVRD"]),
    dict(name="Teck Resources", cik="0000886986", tickers=["NYSE:TECK"], aliases=["Teck Cominco", "Teck"]),
    dict(name="Southern Copper", cik="0001001838", tickers=["NYSE:SCCO"],
         aliases=["Southern Copper Corporation"]),
    dict(name="Hecla Mining", cik="0000719413", tickers=["NYSE:HL"], aliases=[]),
    dict(name="Coeur Mining", cik="0000215466", tickers=["NYSE:CDE"], aliases=["Coeur d'Alene Mines"]),
    dict(name="Royal Gold", cik="0000085535", tickers=["NASDAQ:RGLD"], aliases=[]),
    dict(name="Wheaton Precious Metals", cik="0001323404", tickers=["NYSE:WPM"], aliases=["Silver Wheaton"]),
    dict(name="Franco-Nevada", cik="0001456346", tickers=["NYSE:FNV"], aliases=[]),
    dict(name="Cameco", cik="0001009001", tickers=["NYSE:CCJ"], aliases=[]),
    dict(name="Peabody Energy", cik="0001064728", tickers=["NYSE:BTU"], aliases=["Peabody"]),
    dict(name="Arch Resources", cik="0001037676", tickers=[], aliases=["Arch Coal"]),
    dict(name="Core Natural Resources", cik="0001710366", tickers=["NYSE:CNR"], aliases=["CONSOL Energy"]),
    dict(name="Alcoa", cik="0001675149", tickers=["NYSE:AA"], aliases=[]),
    dict(name="Albemarle", cik="0000915913", tickers=["NYSE:ALB"], aliases=[]),
    dict(name="MP Materials", cik="0001801368", tickers=["NYSE:MP"], aliases=[]),
    dict(name="Gold Fields", cik="0001172724", tickers=["NYSE:GFI"], aliases=[]),
    dict(name="AngloGold Ashanti", cik="0001973832", tickers=["NYSE:AU"], aliases=["AngloGold Ashanti plc"]),
    dict(name="AngloGold Ashanti (Pty)", cik="0001067428", tickers=[], aliases=[]),
    dict(name="Harmony Gold", cik="0001023514", tickers=["NYSE:HMY"], aliases=["Harmony Gold Mining"]),
    dict(name="Sibanye-Stillwater", cik="0001786909", tickers=["NYSE:SBSW"], aliases=["Sibanye Gold"]),
    dict(name="First Majestic Silver", cik="0001308648", tickers=["NYSE:AG"], aliases=[]),
    dict(name="Pan American Silver", cik="0000771992", tickers=["NASDAQ:PAAS"], aliases=[]),
    dict(name="Equinox Gold", cik="0001756607", tickers=["NYSE:EQX"], aliases=[]),
    dict(name="IAMGOLD", cik="0001203464", tickers=["NYSE:IAG"], aliases=[]),
    dict(name="New Gold", cik="0000800166", tickers=[], aliases=[]),
    dict(name="Eldorado Gold", cik="0000918608", tickers=["NYSE:EGO"], aliases=[]),
    dict(name="SSR Mining", cik="0000921638", tickers=["NASDAQ:SSRM"], aliases=["Silver Standard Resources"]),
    dict(name="Hudbay Minerals", cik="0001322422", tickers=["NYSE:HBM"], aliases=["Hudbay"]),
    dict(name="Taseko Mines", cik="0000878518", tickers=["NYSE:TGB"], aliases=["Trekor Metals"]),
    dict(name="Cleveland-Cliffs", cik="0000764065", tickers=["NYSE:CLF"],
         aliases=["Cliffs Natural Resources"]),
    dict(name="Warrior Met Coal", cik="0001691303", tickers=["NYSE:HCC"], aliases=[]),
    dict(name="Alpha Metallurgical Resources", cik="0001704715", tickers=["NYSE:AMR"],
         aliases=["Contura Energy"]),
    dict(name="Energy Fuels", cik="0001385849", tickers=["NYSE:UUUU"], aliases=[]),
    dict(name="Uranium Energy", cik="0001334933", tickers=["NYSE:UEC"], aliases=[]),
    dict(name="Buenaventura", cik="0001013131", tickers=["NYSE:BVN"],
         aliases=["Compañía de Minas Buenaventura"]),
    dict(name="Nexa Resources", cik="0001713930", tickers=["NYSE:NEXA"], aliases=["Votorantim Metais"]),
    dict(name="Endeavour Silver", cik="0001277866", tickers=["NYSE:EXK"], aliases=[]),
    dict(name="OR Royalties", cik="0001627272", tickers=["NYSE:OR"], aliases=["Osisko Gold Royalties"]),
    dict(name="Alamos Gold", cik="0001178819", tickers=["NYSE:AGI"], aliases=[]),
    dict(name="Lithium Americas", cik="0001966983", tickers=["NYSE:LAC"], aliases=[]),
    dict(name="Sigma Lithium", cik="0001848309", tickers=["NASDAQ:SGML"], aliases=[]),
    dict(name="Compass Minerals", cik="0001227654", tickers=["NYSE:CMP"], aliases=[]),
    dict(name="Mosaic", cik="0001285785", tickers=["NYSE:MOS"], aliases=["The Mosaic Company"]),
    dict(name="Intrepid Potash", cik="0001421461", tickers=["NYSE:IPI"], aliases=[]),
    dict(name="McEwen", cik="0000314203", tickers=["NYSE:MUX"], aliases=["McEwen Mining"]),
    dict(name="Seabridge Gold", cik="0001231346", tickers=["NYSE:SA"], aliases=[]),
    dict(name="NovaGold Resources", cik="0001173420", tickers=["NYSE:NG"], aliases=[]),
    dict(name="Gatos Silver", cik="0001517006", tickers=[], aliases=[]),
    dict(name="Denison Mines", cik="0001063259", tickers=["NYSE:DNN"], aliases=[]),
    dict(name="NexGen Energy", cik="0001698535", tickers=["NYSE:NXE"], aliases=[]),
    dict(name="Ur-Energy", cik="0001375205", tickers=["NYSE:URG"], aliases=[]),
    dict(name="Piedmont Lithium", cik="0001728205", tickers=[], aliases=[]),
]

# -- noms de sociétés -----------------------------------------------------------------------------------
# Retirés EN FIN de nom : « Newmont Mining Corporation » et « Newmont » donnent la même clé. Les formes
# juridiques partent toujours (sauf le premier mot) ; les mots de remplissage seulement si le reste garde
# un mot distinctif (« New Gold » ne doit pas devenir « new », ni « Mineral Resources » « mineral »).
_LEGAL = {
    "inc", "incorporated", "ltd", "limited", "corp", "corporation", "co", "company", "plc", "sa", "sab", "cv",
    "de", "ag", "nv", "bv", "llc", "lp", "pty", "pt", "tbk", "gmbh", "spa", "sarl", "sas", "ab", "asa", "oy",
    "oyj", "jsc", "pjsc", "ojsc", "oao", "ao", "zao", "ooo", "sl", "srl", "kk", "bhd", "berhad", "nl", "fi",
    "cie", "cia",
}
_FILLER = {"group", "holdings", "holding", "mining", "mines", "minerals", "resources", "energy", "gold",
           "silver", "copper", "coal", "metals", "international", "compania"}
_REGION_WORDS = {
    "africa", "african", "america", "americas", "american", "australia", "australian", "asia", "pacific",
    "europe", "european", "corporate", "other", "others", "total", "eliminations", "elimination",
    "unallocated", "reconciling", "consolidated", "segments", "segment", "region", "international",
    "royalties", "royalty", "streams", "stream", "exploration", "projects", "global", "oceania", "latin",
}


def _split_words(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", s)  # « KG Mining (Bald Mountain), Inc » : la parenthèse n'est pas le nom
    s = entities.fold(s).replace("&", " and ")
    return re.findall(r"[a-z0-9]+", s)


def norm_key(name) -> str | None:
    """Clé de dédoublonnage d'un nom de société (formes juridiques et remplissage final retirés)."""
    from strate import catalog

    if not isinstance(name, str) or re.fullmatch(r"\s*Q\d+\s*", name):  # QID Wikidata non résolu
        return None
    w = _split_words(name)
    if w and w[0] == "the":
        w = w[1:]
    while len(w) > 1:
        if w[-1] in _LEGAL:
            w = w[:-1]
        elif w[-1] in _FILLER and any(t not in catalog._GENERIC and t not in _LEGAL | _FILLER and len(t) >= 3
                                      for t in w[:-1]):
            w = w[:-1]
        else:
            break
    return " ".join(w) or None


def id_base(name: str) -> str:
    """Identifiant lisible : le nom sans forme juridique finale (« Royal Gold Inc » → royal_gold)."""
    w = _split_words(name)
    while len(w) > 1 and w[-1] in _LEGAL:
        w = w[:-1]
    return slug(" ".join(w))


def slug(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:60]


_LEGAL_END = re.compile(r"\b(inc|ltd|corp|co|company|plc|llc|limited|corporation|pty|s\.?a|gmbh|ag)\.?\s*$",
                        re.I)


def _company_alias(alias: str) -> bool:
    """Alias ICMM qui est un nom de société : il SE TERMINE par une forme juridique
    (« Omai Gold Mines Ltd. »), pas « National Lead Co. Mine » (une mine qui cite une société)."""
    return bool(alias) and bool(_LEGAL_END.search(alias.strip(" ,;")))


def _split_list(v) -> list[str]:
    if not isinstance(v, str):
        return []
    return [x.strip() for x in v.split(";") if x.strip()]


# -- univers ----------------------------------------------------------------------------------------------
def _veille_sites() -> list[str]:
    """Mines suivies par la veille (si sa base existe), lues en lecture seule."""
    rows = _veille_query("SELECT site_id FROM tracked")
    return [r[0] for r in rows]


def _veille_db() -> Path:
    try:
        from strate.veille import store
        return Path(store.DB_PATH)
    except Exception:  # module en cours de modification ailleurs : chemin par défaut
        return DATA_DIR / "veille" / "veille.db"


def _veille_query(sql: str, params: Iterable = ()) -> list[tuple]:
    p = _veille_db()
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


def build_universe(cat: pd.DataFrame | None = None, graph=None, veille_sites: list[str] | None = None
                   ) -> pd.DataFrame:
    """Toutes les sociétés connues, dédoublonnées par clé de nom, avec leurs mines.

    Ordre d'autorité : liste suivie (identifiants fixes, utilisés par l'API), graine SEC (alias de
    changement de nom : « Barrick Gold » = « Barrick Mining »), catalogue, veille.
    """
    from strate import catalog

    graph = graph or entities.load()
    if cat is None:
        cat = catalog.load()
    if veille_sites is None:
        veille_sites = _veille_sites()
    comp: dict[str, dict] = {}
    by_key: dict[str, str] = {}

    def new(cid: str, name: str) -> dict:
        return comp.setdefault(cid, dict(id=cid, name=name, aliases=set(), tickers=[], cik=None, cik_alt=[],
                                         cik_source=None, qid=None, sources=set(), mines={},
                                         watchlist=False))

    def register(cid: str, name: str) -> None:
        k = norm_key(name)
        if k and k not in by_key:
            by_key[k] = cid
        comp[cid]["aliases"].add(name)

    def find(name: str) -> str | None:
        k = norm_key(name)
        return by_key.get(k) if k else None

    def link(cid: str, site: str, role: str, share: float | None = None) -> None:
        m = comp[cid]["mines"].setdefault(site, dict(site=site, roles=[], share=None))
        if role not in m["roles"]:
            m["roles"].append(role)
        if share is not None:
            m["share"] = share

    for c in graph.companies.values():
        e = new(c.id, c.name)
        e.update(watchlist=True, tickers=list(c.tickers))
        e["sources"].add("liste suivie")
        register(c.id, c.name)
    for s in graph.sites.values():
        for o in s.owners:
            link(o.company, s.id, "liste suivie", o.share)

    taken = set(comp)
    for sd in SEED:
        names = [sd["name"], *sd["aliases"]]
        cid = next((x for x in map(find, names) if x), None)
        if cid is None:
            cid = _unique(id_base(sd["name"]), taken)
            new(cid, sd["name"])
        e = comp[cid]
        e["sources"].add("graine SEC")
        if e["cik"] is None:
            e["cik"], e["cik_source"] = sd["cik"], SEED_SOURCE
        elif sd["cik"] != e["cik"]:  # cotation jumelle ou ancienne entité : historique complémentaire
            e["cik_alt"].append(sd["cik"])
        e["tickers"] = list(dict.fromkeys([*e["tickers"], *sd["tickers"]]))
        for n in names:
            register(cid, n)

    def add_named(name: str, site: str, role: str) -> None:
        cid = find(name)
        if cid is None:
            k = norm_key(name)
            if not k:
                return
            cid = _unique(id_base(name), taken)
            new(cid, name)
            register(cid, name)
        comp[cid]["aliases"].add(name)
        comp[cid]["sources"].add({"propriétaire": "catalogue", "exploitant": "catalogue"}.get(role, role))
        link(cid, site, role)

    if not cat.empty:
        sub = cat[cat["owner"].notna() | cat["operator"].notna()]
        for sid, own, opr in zip(sub.index, sub["owner"], sub["operator"], strict=True):
            for n in _split_list(own):
                add_named(n, sid, "propriétaire")
            for n in _split_list(opr):
                add_named(n, sid, "exploitant")
        if "icmm" in cat:
            ali = cat[cat["icmm"].notna() & cat["aliases"].notna()]
            for sid, v in zip(ali.index, ali["aliases"], strict=True):
                for n in _split_list(v):
                    if _company_alias(n):
                        add_named(n, sid, "alias ICMM")
        for sid in veille_sites:
            if sid in cat.index:
                r = cat.loc[sid]
                for n in [*_split_list(r.get("owner")), *_split_list(r.get("operator"))]:
                    add_named(n, sid, "veille")

    rows = []
    for e in comp.values():
        names = Counter(a for a in e["aliases"])
        mines = sorted(e["mines"].values(), key=lambda m: m["site"])
        rows.append(dict(
            id=e["id"], name=e["name"] or names.most_common(1)[0][0], key=norm_key(e["name"]),
            aliases=";".join(sorted(e["aliases"])), tickers=";".join(e["tickers"]), cik=e["cik"],
            cik_alt=";".join(e["cik_alt"]) or None, cik_source=e["cik_source"], qid=e["qid"],
            sources=";".join(sorted(e["sources"])),
            watchlist=e["watchlist"], n_mines=len(mines), mines_json=json.dumps(mines, ensure_ascii=False)))
    df = pd.DataFrame(rows)
    df = df.sort_values(["watchlist", "n_mines", "id"], ascending=[False, False, True])
    return df.reset_index(drop=True)


def _unique(base: str, taken: set[str]) -> str:
    out, k = base or "societe", 2
    while out in taken:
        out, k = f"{base}_{k}", k + 1
    taken.add(out)
    return out


def mines_of(row) -> list[dict]:
    try:
        return json.loads(row["mines_json"]) if isinstance(row["mines_json"], str) else []
    except ValueError:
        return []


# -- accès réseau ------------------------------------------------------------------------------------------
def sec_user_agent() -> tuple[str, bool]:
    """(User-Agent, contact déclaré ?). Jamais écrit dans les journaux ni le statut."""
    ua = os.environ.get("STRATE_SEC_UA", "").strip()
    if not ua and CONFIG.exists():
        try:
            ua = str(tomllib.loads(CONFIG.read_text("utf-8")).get("sec", {}).get("user_agent", "")).strip()
        except (OSError, tomllib.TOMLDecodeError) as e:
            log.warning("config.local.toml illisible (%s)", e)
    return (ua, True) if ua else (GENERIC_UA, False)


class Http:
    """Client HTTP à débit limité (SEC fair access) avec respect de 429/Retry-After.

    Les tests passent un faux client qui expose les mêmes méthodes get_json/get_text.
    """

    def __init__(self, ua: str | None = None, min_interval: float = SEC_MIN_INTERVAL):
        import httpx

        ua = ua or sec_user_agent()[0]
        self.c = httpx.Client(headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60,
                              follow_redirects=True)
        self.min_interval, self._last, self._lock = min_interval, 0.0, threading.Lock()
        self.requests = 0

    def _get(self, url: str, **kw):
        import httpx

        for attempt in range(5):
            with self._lock:
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            self.requests += 1
            try:
                r = self.c.get(url, **kw)
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
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def get_text(self, url: str) -> str | None:
        r = self._get(url)
        return r.text if r is not None and r.status_code == 200 else None

    def status(self, url: str) -> int | None:
        r = self._get(url)
        return None if r is None else r.status_code

    def sparql(self, query: str) -> list[dict]:
        from strate import catalog

        self.sparql_requests = getattr(self, "sparql_requests", 0) + 1
        return catalog._sparql(self.c, query)


# -- identifiants ----------------------------------------------------------------------------------------
def _sparql_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def wikidata_lookup(http, names: list[str], qids: list[str] = (), batch: int = 600) -> dict[str, dict]:
    """Nom exact (libellé ou alias anglais) → {qid, cik, label}. Seuls les éléments « société » comptent :
    ils ont un CIK, un effectif, une place de cotation, un ISIN ou un LEI (sinon « Escondida » la mine
    ou « Vale » la rivière répondraient)."""
    out: dict[str, dict] = {}
    names = sorted({n for n in names if n and len(n) < 120})
    for i in range(0, len(names), batch):
        # libellés anglais ET « mul » (langue par défaut que Wikidata généralise pour les noms propres)
        vals = " ".join(f"{_sparql_str(n)}@{lang}" for n in names[i:i + batch] for lang in ("en", "mul"))
        q = f"""SELECT ?name ?item ?cik WHERE {{ VALUES ?name {{ {vals} }}
          ?item rdfs:label|skos:altLabel ?name .
          FILTER EXISTS {{ ?item wdt:P5531|wdt:P1128|wdt:P414|wdt:P946|wdt:P1278 [] }}
          OPTIONAL {{ ?item wdt:P5531 ?cik }} }}"""
        for b in http.sparql(q):
            _keep_best(out, b["name"]["value"], b)
    if qids:
        vals = " ".join(f"wd:{q}" for q in sorted(set(qids)))
        q = f"""SELECT ?item ?cik ?name WHERE {{ VALUES ?item {{ {vals} }}
          OPTIONAL {{ ?item wdt:P5531 ?cik }}
          OPTIONAL {{ ?item rdfs:label ?name FILTER(LANG(?name)="en") }} }}"""
        for b in http.sparql(q):
            _keep_best(out, b["item"]["value"].rsplit("/", 1)[-1], b)
    return out


_SUFFIXES = ("", " Corporation", " Corp.", " Inc.", " Limited", " Ltd", " plc", " Group", " S.A.", " AG")
_LEGAL_TAIL = re.compile(r"[\s,]+(inc|incorporated|ltd|limited|corp|corporation|co|company|plc|s\.?a\.?|ag|"
                         r"n\.?v\.?|llc|pty|group|holdings)\.?$", re.I)


def name_variants(name: str) -> list[str]:
    """Écritures probables d'un nom sur Wikidata : la recherche y est exacte, or l'élément s'appelle
    souvent « Newmont Corporation » quand le catalogue écrit « Newmont Mining Corporation » ou « Newmont »."""
    base = name.strip()
    while True:
        b = _LEGAL_TAIL.sub("", base).strip(" ,")
        if b == base or not b:
            break
        base = b
    return list(dict.fromkeys([name.strip(), *(base + s for s in _SUFFIXES)]))


def _qnum(q: str) -> int:
    return int(q[1:]) if q[1:].isdigit() else 10**12


def _keep_best(out: dict, key: str, b: dict) -> None:
    """Plusieurs éléments pour un nom : celui qui a un CIK, puis le plus ancien (QID le plus petit)."""
    qid = b["item"]["value"].rsplit("/", 1)[-1]
    cik = b.get("cik", {}).get("value")
    cand = dict(qid=qid, cik=cik.zfill(10) if cik and cik.isdigit() else None)
    cur = out.get(key)
    if cur is None or (cand["cik"] and not cur["cik"]) or (
            bool(cand["cik"]) == bool(cur["cik"]) and _qnum(qid) < _qnum(cur["qid"])):
        out[key] = cand


def wikidata_employees(http, qids: list[str], batch: int = 300) -> pd.DataFrame:
    """Effectifs Wikidata (P1128) datés (qualificatif P585) : une source sans contact SEC requis."""
    rows = []
    qids = sorted(set(q for q in qids if q))
    for i in range(0, len(qids), batch):
        vals = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        q = f"""SELECT ?item ?n ?date WHERE {{ VALUES ?item {{ {vals} }}
          ?item p:P1128 ?st . ?st ps:P1128 ?n . OPTIONAL {{ ?st pq:P585 ?date }} }}"""
        for b in http.sparql(q):
            d = b.get("date", {}).get("value")
            try:
                n = float(b["n"]["value"])
            except (KeyError, ValueError):
                continue
            if not d:
                continue  # effectif sans date : inutilisable pour une série annuelle
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            rows.append(dict(qid=qid, year=int(d[:4]), employees=n,
                             source_url=f"https://www.wikidata.org/wiki/{qid}#P1128"))
    return pd.DataFrame(rows, columns=["qid", "year", "employees", "source_url"])


def sec_tickers(http) -> dict[str, dict]:
    """company_tickers.json (contact SEC requis) : ticker et clé de nom → CIK."""
    js = http.get_json("https://www.sec.gov/files/company_tickers.json")
    if not js:
        return {}
    out = {}
    for v in js.values():
        cik = str(v.get("cik_str", "")).zfill(10)
        out[f"T:{str(v.get('ticker', '')).upper()}"] = dict(cik=cik, title=v.get("title"))
        k = norm_key(v.get("title"))
        if k:
            out.setdefault(f"K:{k}", dict(cik=cik, title=v.get("title")))
    return out


_US_EXCH = ("NYSE", "NASDAQ", "NYSEAMERICAN", "AMEX", "NYSEARCA", "OTC")


def assign_ids(uni: pd.DataFrame, wd: dict[str, dict], tick: dict[str, dict]) -> pd.DataFrame:
    """Complète CIK et QID : graine vérifiée > company_tickers SEC > Wikidata."""
    uni = uni.copy()
    for i, r in uni.iterrows():
        names = [v for n in [r["name"], *_split_list(r["aliases"])] for v in name_variants(n)]
        hit = next((wd[n] for n in names if n in wd), None)
        if hit and not isinstance(r["qid"], str):
            uni.at[i, "qid"] = hit["qid"]
        if isinstance(r["cik"], str) and r["cik"]:
            continue
        for t in _split_list(r["tickers"]):
            ex, _, sym = t.partition(":")
            if ex.upper() in _US_EXCH and f"T:{sym.upper()}" in tick:
                uni.at[i, "cik"] = tick[f"T:{sym.upper()}"]["cik"]
                uni.at[i, "cik_source"] = f"SEC company_tickers.json (ticker {sym.upper()})"
                break
        else:
            k = r["key"]
            if isinstance(k, str) and f"K:{k}" in tick:
                uni.at[i, "cik"] = tick[f"K:{k}"]["cik"]
                uni.at[i, "cik_source"] = f"SEC company_tickers.json (nom « {tick[f'K:{k}']['title']} »)"
            elif hit and hit["cik"]:
                uni.at[i, "cik"] = hit["cik"]
                uni.at[i, "cik_source"] = f"Wikidata P5531 — https://www.wikidata.org/wiki/{hit['qid']}"
    return uni


# -- comptes annuels (companyfacts) -------------------------------------------------------------------------
# Concepts candidats par mesure, par ordre de préférence. Par société, le concept le mieux couvert gagne,
# les autres ne bouchent que les années manquantes (chaque ligne garde le concept utilisé).
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueNet",
                "us-gaap:SalesRevenueGoodsNet", "us-gaap:RevenueMineralSales", "ifrs-full:Revenue",
                "ifrs-full:RevenueFromContractsWithCustomers"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
              "us-gaap:PaymentsToAcquireProductiveAssets",
              "us-gaap:PaymentsToAcquireMiningAssets", "us-gaap:PaymentsForCapitalImprovements",
              "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
              "ifrs-full:PurchaseOfPropertyPlantAndEquipment",
              "ifrs-full:PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets"],
    "ppe_net": ["us-gaap:PropertyPlantAndEquipmentNet",
                "us-gaap:PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
                "ifrs-full:PropertyPlantAndEquipment"],
    "ocf": ["us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "ifrs-full:CashFlowsFromUsedInOperatingActivities",
            "ifrs-full:CashFlowsFromUsedInOperatingActivitiesContinuingOperations"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss", "ifrs-full:ProfitLoss",
                   "ifrs-full:ProfitLossAttributableToOwnersOfParent"],
    "da": ["us-gaap:DepreciationDepletionAndAmortization", "us-gaap:DepreciationAndAmortization",
           "us-gaap:DepreciationAmortizationAndAccretionNet", "ifrs-full:DepreciationAndAmortisationExpense",
           "ifrs-full:AdjustmentsForDepreciationAndAmortisationExpense"],
}
INSTANT = {"ppe_net"}
METRIC_LABELS = {
    "revenue": "Chiffre d'affaires", "capex": "CAPEX (investissements corporels payés)",
    "ppe_net": "Immobilisations corporelles nettes", "ocf": "Flux de trésorerie d'exploitation",
    "net_income": "Résultat net", "da": "Amortissements et dépréciations",
    "capex_revenue": "CAPEX / chiffre d'affaires", "capex_per_employee": "CAPEX par salarié",
    "capex_growth": "Croissance du CAPEX", "ppe_growth": "Croissance des immobilisations",
    "capex_da": "CAPEX / amortissements (> 1 : investissement de croissance)",
    "capex_prevision": "CAPEX prévu (année suivante)",
}
FIN_COLS = ["company_id", "cik", "year", "metric", "value", "unit", "concept", "form", "accn", "filed",
            "period_end", "source_url", "lo90", "hi90"]


def filing_url(cik: str, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index.htm"


def _base_form(f: str) -> str:
    return (f or "").removesuffix("/A")


def _annual_facts(facts: dict, concept: str) -> tuple[str | None, list[dict]]:
    ns, local = concept.split(":")
    units = facts.get("facts", {}).get(ns, {}).get(local, {}).get("units", {})
    if not units:
        return None, []
    unit = "USD" if "USD" in units else max(units, key=lambda u: len(units[u]))
    return unit, [x for x in units[unit] if _base_form(x.get("form")) in ANNUAL_FORMS]


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def parse_facts(facts: dict, company_id: str, cik: str) -> pd.DataFrame:
    """Valeurs annuelles par mesure : exercices complets seulement, retraitements → dernier dépôt."""
    # dates de clôture déclarées par chaque dépôt annuel : les postes de bilan (instantanés) ne sont gardés
    # qu'à ces dates, sinon un bilan trimestriel repris dans un 10-K passerait pour une clôture
    closes: set[tuple[str, str]] = set()
    per_metric: dict[str, list[tuple[str, str, dict[int, dict]]]] = {}
    for metric, cands in CONCEPTS.items():
        if metric in INSTANT:
            continue
        per_metric[metric] = []
        for concept in cands:
            unit, xs = _annual_facts(facts, concept)
            byyear: dict[int, dict] = {}
            for x in xs:
                if "start" not in x or not 330 <= _days(x["start"], x["end"]) <= 400:
                    continue
                closes.add((x["accn"], x["end"]))
                y = int(x["end"][:4])
                cur = byyear.get(y)
                if cur is None or (x["filed"], x["end"]) > (cur["filed"], cur["end"]):
                    byyear[y] = x
            if byyear:
                per_metric[metric].append((concept, unit, byyear))
    for metric in INSTANT:
        per_metric[metric] = []
        for concept in CONCEPTS[metric]:
            unit, xs = _annual_facts(facts, concept)
            byyear = {}
            for x in xs:
                if "start" in x or (x["accn"], x["end"]) not in closes:
                    continue
                y = int(x["end"][:4])
                cur = byyear.get(y)
                if cur is None or (x["filed"], x["end"]) > (cur["filed"], cur["end"]):
                    byyear[y] = x
            if byyear:
                per_metric[metric].append((concept, unit, byyear))

    rows = []
    for metric, lst in per_metric.items():
        if not lst:
            continue
        prio = {c: i for i, c in enumerate(CONCEPTS[metric])}
        lst = sorted(lst, key=lambda t: (-len(t[2]), prio[t[0]]))
        main_unit = lst[0][1]
        done: set[int] = set()
        for concept, unit, byyear in lst:
            if unit != main_unit:
                continue  # jamais deux monnaies dans une même série
            for y, x in sorted(byyear.items()):
                if y in done or y < FIRST_YEAR:
                    continue
                done.add(y)
                v = float(x["val"])
                rows.append(dict(company_id=company_id, cik=cik, year=y, metric=metric,
                                 value=abs(v) if metric == "capex" else v, unit=unit, concept=concept,
                                 form=x.get("form"), accn=x["accn"], filed=x["filed"], period_end=x["end"],
                                 source_url=filing_url(cik, x["accn"]), lo90=np.nan, hi90=np.nan))
    return pd.DataFrame(rows, columns=FIN_COLS)


def merge_alt(main: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les (mesure, année) absents du CIK principal, dans la même monnaie seulement."""
    if alt.empty:
        return main
    if main.empty:
        return alt
    units = main.groupby("metric")["unit"].first()
    have = set(zip(main["metric"], main["year"], strict=True))
    keep = [(m, y) not in have and units.get(m, u) == u
            for m, y, u in zip(alt["metric"], alt["year"], alt["unit"], strict=True)]
    out = pd.concat([main, alt[keep]], ignore_index=True)
    return out.sort_values(["metric", "year"]).reset_index(drop=True)


def dei_employees(facts: dict, company_id: str, cik: str) -> list[dict]:
    """dei:EntityNumberOfEmployees quand la société le déclare (rare chez les miniers)."""
    units = facts.get("facts", {}).get("dei", {}).get("EntityNumberOfEmployees", {}).get("units", {})
    best: dict[int, dict] = {}
    for xs in units.values():
        for x in xs:
            if _base_form(x.get("form")) not in ANNUAL_FORMS:
                continue
            y = int(x["end"][:4])
            if y not in best or x["filed"] > best[y]["filed"]:
                best[y] = x
    return [dict(company_id=company_id, year=y, employees=float(x["val"]), source_kind="sec_dei",
                 source_url=filing_url(cik, x["accn"]), snippet="dei:EntityNumberOfEmployees",
                 accepted=True, reason="déclaré dans l'instance XBRL") for y, x in sorted(best.items())]


# -- effectifs dans le texte des rapports annuels ------------------------------------------------------
_NUM = r"(\d{1,3}(?:[,   ]\d{3})+|\d{2,6})"
_NUM_FR = r"(\d{1,3}(?:[   .]\d{3})+|\d{2,6})"
_QUAL_EN = r"(?:approximately|about|around|some|over|more than|nearly|roughly|a total of|an average of)?"
_EMP_EN = [
    re.compile(rf"(?:had|employ(?:ed|s)?|workforce (?:of|was|totall?ed)|total of|consisted of|comprised)\s+"
               rf"{_QUAL_EN}\s*(?:a total of\s+|approximately\s+)?{_NUM}\s+(?:full[- ]time\s+|permanent\s+|"
               rf"direct\s+|active\s+)?(?:employees|people|persons|staff|workers)", re.I),
    re.compile(rf"{_QUAL_EN}\s*{_NUM}\s+(?:full[- ]time\s+|permanent\s+)?employees\b", re.I),
    re.compile(rf"(?:total (?:number of )?employees|number of employees|total workforce)\s*(?:was|of|:)?\s*"
               rf"{_QUAL_EN}\s*{_NUM}", re.I),
]
_QUAL_FR = r"(?:environ|près de|plus de|quelque|au total)?"
_EMP_FR = [
    re.compile(rf"(?:comptait|employait|emploie|employaient|"
               rf"effectifs? (?:total )?(?:de|était de|s'élevait à)|total de)\s+{_QUAL_FR}\s*{_NUM_FR}\s+"
               rf"(?:employés|salariés|personnes|travailleurs|membres du personnel)", re.I),
    re.compile(rf"{_QUAL_FR}\s*{_NUM_FR}\s+(?:employés|salariés)\b", re.I),
]
_DATED = re.compile(r"(as of|as at|at (?:the end of|december|june|march|september)|au 31|à la fin|en date du|"
                    r"year[- ]end)", re.I)
_COMBINED = re.compile(r"^\W{0,3}(?:and|or|including|plus|et|y compris)\s+(?:\w+\s+){0,2}contractors?|"
                       r"^\W{0,3}(?:et|and)\s+(?:\w+\s+){0,2}(?:entrepreneurs|sous-traitants)", re.I)
EMP_RANGE = (10, 600_000)


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return " ".join(html.unescape(s).split())


def extract_employees(text: str, lang: str = "en") -> tuple[int, str] | None:
    """Effectif annoncé dans un rapport annuel : (nombre, extrait). Règles :
    - phrases du type « we had approximately 22,200 employees » (anglais) ou « comptait 3 400 employés » ;
    - un nombre suivi de « and contractors » est un total mêlé : rejeté ;
    - hors [10 ; 600 000] : rejeté ; une phrase datée (« as of December 31 ») passe avant les autres."""
    pats = _EMP_FR + _EMP_EN if lang == "fr" else _EMP_EN + _EMP_FR
    cands = []
    for rank, pat in enumerate(pats):
        for m in pat.finditer(text):
            raw = m.group(1)
            n = int(re.sub(r"[^\d]", "", raw))
            if not EMP_RANGE[0] <= n <= EMP_RANGE[1]:
                continue
            if 1900 <= n <= 2100 and not re.search(r"[,   .]", raw):
                continue  # une année (« in 2024 employees… »), pas un effectif
            if _COMBINED.search(text[m.end():m.end() + 60]):
                continue
            ctx = text[max(0, m.start() - 160):m.end() + 40]
            dated = bool(_DATED.search(ctx))
            cands.append((not dated, rank, m.start(), n, ctx.strip()))
    if not cands:
        return None
    _, _, _, n, ctx = min(cands)
    return n, ctx[:300]


def validate_employees(n: float, year: int, others: dict[int, float], max_ratio: float = 5.0
                       ) -> tuple[bool, str]:
    """Plausibilité : dans la plage, et pas plus de ×5 d'écart avec l'effectif connu de l'année la plus
    proche (une autre source), sinon le nombre lu est probablement autre chose (sites, contractants…)."""
    if not EMP_RANGE[0] <= n <= EMP_RANGE[1]:
        return False, "hors plage plausible"
    near = [(abs(y - year), v) for y, v in others.items() if v and abs(y - year) <= 3]
    if near:
        _, ref = min(near)
        r = max(n / ref, ref / n)
        if r > max_ratio:
            return False, f"écart ×{r:.1f} avec l'année la plus proche ({ref:,.0f})"
    return True, "plausible"


def annual_filings(sub: dict, n: int | None = None) -> list[dict]:
    """Rapports annuels (10-K, 20-F, 40-F) les plus récents d'une fiche submissions."""
    rec = sub.get("filings", {}).get("recent", {})
    out = []
    for i, f in enumerate(rec.get("form", [])):
        if f in ANNUAL_FORMS:
            out.append(dict(form=f, accn=rec["accessionNumber"][i], filed=rec["filingDate"][i],
                            report=rec.get("reportDate", [""] * (i + 1))[i] or rec["filingDate"][i],
                            primary=rec.get("primaryDocument", [""] * (i + 1))[i]))
    out.sort(key=lambda d: d["filed"], reverse=True)
    return out[:n] if n else out


def _folder(cik: str, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}"


_ROW = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_HREF = re.compile(r'(?i)href="([^"]+)"')


def annual_text_url(http, cik: str, f: dict) -> str | None:
    """Document où lire l'effectif : le document principal (10-K, 20-F) ; pour un 40-F, la notice annuelle
    (Annual Information Form, pièce EX-99.x) repérée dans l'index du dépôt."""
    folder = _folder(cik, f["accn"])
    if f["form"] != "40-F":
        return f"{folder}/{f['primary']}" if f.get("primary") else None
    idx = http.get_text(f"{folder}/{f['accn']}-index.htm")
    if not idx:
        return None
    best = None
    for row in _ROW.findall(idx):
        txt, href = html_to_text(row).lower(), _HREF.search(row)
        if not href or not re.search(r"\.(htm|html|txt)$", href.group(1), re.I):
            continue
        url = href.group(1) if href.group(1).startswith("http") else f"https://www.sec.gov{href.group(1)}"
        if "annual information form" in txt or "notice annuelle" in txt:
            return url
        if best is None and re.search(r"\bex-99\.1\b", txt):
            best = url
    return best


# -- CAPEX par segment (instance XBRL) ----------------------------------------------------------------------
SEG_AXES = {"StatementBusinessSegmentsAxis", "SegmentsAxis"}
_SEG_AXIS_EXT = re.compile(r"(Segment|Mine|Operation|Project|Site|Asset)s?Axis$")
_NEUTRAL_AXES = {"ConsolidationItemsAxis"}
SEG_CAPEX = {
    "PaymentsToAcquireProductiveAssets", "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireMiningAssets", "SegmentExpenditureAdditionToLongLivedAssets",
    "PropertyPlantAndEquipmentAdditions",
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    "AdditionsToNoncurrentAssets", "AdditionsOtherThanThroughBusinessCombinationsPropertyPlantAndEquipment",
    "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwill"
    "InvestmentPropertyAndOtherNoncurrentAssets",
}
_SEG_CAPEX_EXT = re.compile(r"(CapitalExpenditure|Capex|SustainingCapital|DevelopmentCapital|"
                            r"PaymentsForPropertyPlantAndEquipment)", re.I)
SEG_COLS = ["company_id", "kind", "year", "date", "axis", "member", "label", "concept", "value", "unit",
            "site_id", "site_name", "match", "title", "event_type", "accn", "source_url"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def member_label(qname: str) -> str:
    """« nem:PenasquitoMember » → « Penasquito » ; « x:CopperSouthAustraliaSegmentMember » → mots séparés."""
    s = qname.rsplit(":", 1)[-1]
    s = re.sub(r"(Segment)?Member$", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)", " ", s)
    return " ".join(s.split())


def parse_segment_capex(xml_text: str) -> list[dict]:
    """Faits CAPEX ventilés par segment dans une instance XBRL : (axe, membre, année, concept, valeur)."""
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    ctx: dict[str, dict] = {}
    for c in root.iter():
        if _local(c.tag) != "context":
            continue
        dims, start, end = {}, None, None
        for e in c.iter():
            t = _local(e.tag)
            if t == "explicitMember":
                dims[e.get("dimension", "")] = (e.text or "").strip()
            elif t == "startDate":
                start = (e.text or "").strip()
            elif t == "endDate":
                end = (e.text or "").strip()
        ctx[c.get("id")] = dict(dims=dims, start=start, end=end)
    out, seen = [], set()
    for e in root.iter():
        cref = e.get("contextRef")
        if not cref or cref not in ctx:
            continue
        local = _local(e.tag)
        ns = e.tag[1:].split("}")[0] if e.tag.startswith("{") else ""
        std = "fasb.org/us-gaap" in ns or "xbrl.ifrs.org" in ns
        if not ((std and local in SEG_CAPEX) or (not std and _SEG_CAPEX_EXT.search(local))):
            continue
        c = ctx[cref]
        if not c["start"] or not c["end"] or not 330 <= _days(c["start"], c["end"]) <= 400:
            continue
        seg = [(a, m) for a, m in c["dims"].items() if _local(a) not in _NEUTRAL_AXES]
        if len(seg) != 1:
            continue
        axis, member = seg[0]
        if not (_local(axis) in SEG_AXES or _SEG_AXIS_EXT.search(_local(axis))):
            continue
        try:
            v = abs(float((e.text or "").strip()))
        except ValueError:
            continue
        prefix = "us-gaap" if "fasb.org" in ns else "ifrs-full" if "ifrs.org" in ns else "extension"
        key = (member, c["end"], local)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(axis=axis, member=member, label=member_label(member), year=int(c["end"][:4]),
                        concept=f"{prefix}:{local}", value=v, unit=e.get("unitRef")))
    return out


def instance_url(http, cik: str, accn: str) -> str | None:
    """Instance XBRL d'un dépôt : celle extraite du XBRL en ligne (…_htm.xml), sinon le .xml principal."""
    folder = _folder(cik, accn)
    js = http.get_json(f"{folder}/index.json")
    if not js:
        return None
    names = [it.get("name", "") for it in js.get("directory", {}).get("item", [])]
    for n in names:
        if n.endswith("_htm.xml"):
            return f"{folder}/{n}"
    skip = re.compile(r"(_cal|_def|_lab|_pre)\.xml$|FilingSummary")
    xs = [n for n in names if n.endswith(".xml") and not skip.search(n)]
    return f"{folder}/{xs[0]}" if xs else None


class MineMatcher:
    """Rattache un libellé de segment à une mine : mêmes mots distinctifs (catalog._sim_tokens exact),
    d'abord parmi les mines de la société, sinon dans tout le catalogue si un SEUL site correspond.
    Libellés géographiques ou comptables (« Africa », « Corporate and other ») : jamais rattachés."""

    def __init__(self, cat: pd.DataFrame | None, graph=None):
        from strate import catalog

        self.catalog = catalog
        self.cat = cat if cat is not None else pd.DataFrame()
        self.graph = graph or entities.load()
        self._index: dict[str, list[str]] | None = None

    def names(self, site_id: str) -> list[str]:
        if site_id in self.graph.sites:
            return [self.graph.sites[site_id].name]
        if not self.cat.empty and site_id in self.cat.index:
            r = self.cat.loc[site_id]
            return [x for x in [r.get("name"), *_split_list(r.get("aliases"))] if isinstance(x, str)]
        return []

    def _build(self) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = defaultdict(list)
        if not self.cat.empty:
            for sid, n, a in zip(self.cat.index, self.cat["name"], self.cat.get("aliases", pd.Series(
                    index=self.cat.index, dtype=object)), strict=True):
                for name in [n, *_split_list(a)]:
                    for t in self.catalog._core(self.catalog._tokens(name)):
                        idx[t].append(sid)
        return idx

    def _aggregate(self, label: str) -> bool:
        words = set(_split_words(label))
        if words & _REGION_WORDS:
            return True
        return entities.fold(label) in entities.COUNTRY_ALIASES

    def match(self, label: str, company_sites: list[str]) -> tuple[str | None, str | None, str | None]:
        cg = self.catalog
        tl = cg._tokens(label)
        if not cg._core(tl) or self._aggregate(label):
            return None, None, None
        hits = [s for s in company_sites if any(cg._sim_tokens(tl, cg._tokens(n), exact=True)
                                                for n in self.names(s))]
        if len(set(hits)) == 1:
            return hits[0], self.names(hits[0])[0], "mine de la société"
        if hits:
            return None, None, None  # plusieurs mines de la société portent ce nom : ambigu
        if self._index is None:
            self._index = self._build()
        core = cg._core(tl)
        cand = set.intersection(*(set(self._index.get(t, ())) for t in core)) if core else set()
        ok = {s for s in cand if any(cg._sim_tokens(tl, cg._tokens(n), exact=True) for n in self.names(s))}
        if len(ok) == 1:
            s = ok.pop()
            return s, self.names(s)[0], "catalogue (nom unique)"
        return None, None, None


# -- investissements annoncés (veille) ----------------------------------------------------------------------
INVEST_WORDS = re.compile(
    r"\b(invest\w*|expansion|expand\w*|capex|capital (?:spending|expenditure|project)|construction|"
    r"build\w*|develop(?:ment)?|upgrade\w*|extension|ramp[- ]up|"
    r"investissement\w*|agrandissement|extension|expansion|construction|"
    r"inversi[oó]n\w*|ampliaci[oó]n|expansi[oó]n|construcci[oó]n|"
    r"investimento\w*|amplia[cç][aã]o|expans[aã]o|constru[cç][aã]o|"
    r"investition\w*|erweiterung|ausbau)\b", re.I)


def announced_investments(company_id: str, sites: list[str]) -> list[dict]:
    """Articles de la veille (types opérations/finance) dont le titre parle d'investissement ou d'expansion,
    sur les mines de la société. Ce sont des annonces de presse, pas des montants audités."""
    if not sites:
        return []
    rows = []
    for i in range(0, len(sites), 500):
        chunk = sites[i:i + 500]
        q = (f"SELECT site_id, site_name, event_type, published_at, title, url FROM events "
             f"WHERE event_type IN ('operations','finance') AND site_id IN ({','.join('?' * len(chunk))})")
        for sid, sname, et, pub, title, url in _veille_query(q, chunk):
            if not title or not INVEST_WORDS.search(title):
                continue
            rows.append(dict(company_id=company_id, kind="annonce_veille", year=int(pub[:4]), date=pub[:10],
                             axis=None, member=None, label=None, concept=None, value=np.nan, unit=None,
                             site_id=sid, site_name=sname, match="veille", title=title, event_type=et,
                             accn=None, source_url=url))
    return rows


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


def _replace_company(name: str, cid: str, new: pd.DataFrame, cols: list[str],
                     keep: Callable[[pd.DataFrame], pd.Series] | None = None) -> None:
    """Remplace les lignes d'une société (les autres restent) : écriture après chaque société → reprise."""
    old = load(name)
    if len(old):
        drop = old["company_id"] == cid
        if keep is not None:
            drop &= ~keep(old)
        old = old[~drop]
    parts = [x for x in (old, new) if len(x)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    _write(name, df.reindex(columns=cols))


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
        return (datetime.now(UTC) - datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                ).total_seconds() / 86400
    except ValueError:
        return math.inf


EMP_COLS = ["company_id", "year", "employees", "source_kind", "source_url", "snippet", "accepted", "reason"]


# -- mise à jour ---------------------------------------------------------------------------------------------
def update(limit: int | None = None, *, http=None, force: bool = False, refresh_days: float = REFRESH_DAYS,
           wikidata: bool = True, wikidata_min_mines: int = 2, wikidata_aliases: int = 2,
           cat: pd.DataFrame | None = None,
           say: Callable[[str], None] = log.info) -> dict:
    """Met à jour la couche sociétés, de façon incrémentale et reprenable.

    `limit` : nombre maximal de sociétés interrogées à la SEC pendant cet appel (les plus prioritaires
    d'abord : liste suivie, graine, puis par nombre de mines). Une société lue il y a moins de
    `refresh_days` jours est sautée, sauf `force`. Chaque société est écrite dès qu'elle est finie.
    """
    from strate import catalog

    ua, contact = sec_user_agent()
    http = http or Http(ua)
    state = _read_json(STATE)
    errors: list[str] = []
    if cat is None:
        cat = catalog.load()
    graph = entities.load()

    uni = build_universe(cat, graph)
    old = load("companies")
    if len(old) and "qid" in old:  # QID et CIK déjà trouvés : gardés si Wikidata n'est pas relu
        prev = old.set_index("id")
        for i, r in uni.iterrows():
            if r["id"] in prev.index:
                p = prev.loc[r["id"]]
                if not isinstance(r["qid"], str) and isinstance(p.get("qid"), str):
                    uni.at[i, "qid"] = p["qid"]
                if not isinstance(r["cik"], str) and isinstance(p.get("cik"), str):
                    uni.at[i, "cik"], uni.at[i, "cik_source"] = p["cik"], p.get("cik_source")

    wd, tick = {}, {}
    if wikidata and (force or _age_days(state.get("wikidata_at")) > WIKIDATA_REFRESH_DAYS):
        pick = uni[(uni["n_mines"] >= wikidata_min_mines) | uni["watchlist"] | uni["cik"].notna()]
        names = sorted({v for _, r in pick.iterrows()
                        for n in [r["name"], *_split_list(r["aliases"])[:wikidata_aliases]]
                        for v in name_variants(n)})
        try:
            wd = wikidata_lookup(http, names)
            state["wikidata_at"] = _now()
            say(f"Wikidata : {len(wd)} noms reconnus sur {len(names)}")
        except Exception as e:  # réseau, quota : on garde les identifiants déjà connus
            errors.append(f"Wikidata : {type(e).__name__}")
    if contact:
        tick = sec_tickers(http)
        if not tick:
            errors.append("SEC company_tickers.json : réponse vide ou refusée")
    uni = assign_ids(uni, wd, tick)
    _write("companies", uni)

    if wikidata and len(wd):
        try:
            we = wikidata_employees(http, [q for q in uni["qid"] if isinstance(q, str)])
            q2c = {q: c for c, q in zip(uni["id"], uni["qid"], strict=True) if isinstance(q, str)}
            rows = [dict(company_id=q2c[r.qid], year=r.year, employees=r.employees, source_kind="wikidata",
                         source_url=r.source_url, snippet="Wikidata P1128 (date P585)", accepted=True,
                         reason="source tierce datée") for r in we.itertuples() if r.qid in q2c]
            emp = load("employees")
            emp = emp[emp["source_kind"] != "wikidata"] if len(emp) else emp
            parts = [x for x in (emp, pd.DataFrame(rows, columns=EMP_COLS)) if len(x)]
            _write("employees", pd.concat(parts, ignore_index=True).reindex(columns=EMP_COLS) if parts
                   else pd.DataFrame(columns=EMP_COLS))
        except Exception as e:
            errors.append(f"Wikidata effectifs : {type(e).__name__}")

    todo = uni[uni["cik"].notna()]
    done_ok = 0
    processed = 0
    per = state.setdefault("companies", {})
    matcher = MineMatcher(cat, graph)
    for _, r in todo.iterrows():
        if limit is not None and processed >= limit:
            break
        st = per.get(r["id"], {})
        if not force and st.get("ok") and _age_days(st.get("fetched_at")) < refresh_days:
            continue
        processed += 1
        try:
            msg = _update_company(http, r, contact, matcher)
            per[r["id"]] = dict(fetched_at=_now(), ok=True, message=msg)
            done_ok += 1
            say(f"{r['name']} : {msg}")
        except Exception as e:  # une société en échec n'arrête pas les autres
            log.exception("société %s", r["id"])
            per[r["id"]] = dict(fetched_at=_now(), ok=False, message=f"{type(e).__name__}: {e}"[:300])
            errors.append(f"{r['id']} : {type(e).__name__}")
        _write_json(STATE, state)

    # annonces de la veille pour toutes les sociétés (aucun réseau)
    ann = []
    for _, r in uni.iterrows():
        ann += announced_investments(r["id"], [m["site"] for m in mines_of(r)])
    seg = load("segments")
    seg = seg[seg["kind"] != "annonce_veille"] if len(seg) else seg
    parts = [x for x in (seg, pd.DataFrame(ann, columns=SEG_COLS)) if len(x)]
    _write("segments", pd.concat(parts, ignore_index=True).reindex(columns=SEG_COLS) if parts
           else pd.DataFrame(columns=SEG_COLS))
    _write_json(STATE, state)

    st = _status(uni, contact, errors, processed, done_ok, getattr(http, "requests", None))
    _write_json(STATUS, {**_read_json(STATUS), **st})
    return st


def _update_company(http, r, contact: bool, matcher: MineMatcher) -> str:
    cik, cid = r["cik"], r["id"]
    facts = http.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    fin = parse_facts(facts, cid, cik) if facts else pd.DataFrame(columns=FIN_COLS)
    for alt in _split_list(r.get("cik_alt")):  # ancienne entité, cotation jumelle : années manquantes
        fa = http.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{alt}.json")
        if fa:
            fin = merge_alt(fin, parse_facts(fa, cid, alt))
    _replace_company("financials", cid, fin, FIN_COLS)
    emp_rows = dei_employees(facts, cid, cik) if facts else []
    parts = [f"{fin['year'].nunique() if len(fin) else 0} exercices" if facts else "companyfacts absent"]

    seg_rows: list[dict] = []
    if contact:
        sub = http.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        filings = annual_filings(sub) if sub else []
        known = {}
        emp_old = load("employees")
        if len(emp_old):
            e = emp_old[(emp_old["company_id"] == cid) & emp_old["accepted"].astype(bool)]
            known = dict(zip(e["year"], e["employees"], strict=True))
        known.update({x["year"]: x["employees"] for x in emp_rows})
        n_txt = 0
        for f in filings[:TEXT_FILINGS]:
            url = annual_text_url(http, cik, f)
            txt = http.get_text(url) if url else None
            if not txt:
                continue
            year = int((f["report"] or f["filed"])[:4])
            got = extract_employees(html_to_text(txt), lang="en")
            if got is None:
                got = extract_employees(html_to_text(txt), lang="fr")
            if got is None:
                continue
            n, snip = got
            ok, why = validate_employees(n, year, {y: v for y, v in known.items() if y != year})
            emp_rows.append(dict(company_id=cid, year=year, employees=float(n), source_kind="rapport_annuel",
                                 source_url=url, snippet=snip, accepted=ok, reason=why))
            if ok:
                known[year] = n
                n_txt += 1
        parts.append(f"effectifs lus dans {n_txt} rapport(s)")
        sites = [m["site"] for m in mines_of(r)]
        for f in filings[:SEGMENT_FILINGS]:
            iu = instance_url(http, cik, f["accn"])
            xml = http.get_text(iu) if iu else None
            if not xml:
                continue
            try:
                facts_seg = parse_segment_capex(xml)
            except ET.ParseError:
                continue
            for x in facts_seg:
                sid, sname, how = matcher.match(x["label"], sites)
                seg_rows.append(dict(
                    company_id=cid, kind="xbrl_segment", year=x["year"], date=None, axis=x["axis"],
                    member=x["member"], label=x["label"], concept=x["concept"], value=x["value"],
                    unit=x["unit"],
                    site_id=sid, site_name=sname, match=how, title=None, event_type=None, accn=f["accn"],
                    source_url=filing_url(cik, f["accn"])))
        seg_df = pd.DataFrame(seg_rows, columns=SEG_COLS)
        if len(seg_df):  # dédoublonnage : un même segment-année revient dans deux rapports successifs
            seg_df = seg_df.sort_values("accn", ascending=False).drop_duplicates(
                ["member", "year", "concept"]).sort_values(["year", "value"], ascending=[True, False])
        _replace_company("segments", cid, seg_df, SEG_COLS, keep=lambda d: d["kind"] != "xbrl_segment")
        parts.append(f"{len(seg_df)} lignes de CAPEX par segment")
    _replace_company("employees", cid, pd.DataFrame(emp_rows, columns=EMP_COLS), EMP_COLS,
                     keep=lambda d: d["source_kind"] == "wikidata")
    return ", ".join(parts)


def _status(uni: pd.DataFrame, contact: bool, errors: list[str], processed: int, ok: int, requests) -> dict:
    fin, emp, seg = load("financials"), load("employees"), load("segments")
    base = fin[fin["metric"].isin(list(CONCEPTS))] if len(fin) else fin
    msg = ("contact SEC déclaré : effectifs (texte des rapports) et CAPEX par segment lus"
           if contact else
           "pas de contact SEC (STRATE_SEC_UA ou config.local.toml [sec] user_agent) : comptes annuels "
           "seulement ; effectifs par le texte, CAPEX par segment et company_tickers.json sautés "
           "(www.sec.gov répond 403 sans contact)")
    return dict(
        updated_at=_now(), sec_contact=contact, message=msg, errors=errors[-50:],
        n_companies=int(len(uni)), n_with_cik=int(uni["cik"].notna().sum()),
        n_with_qid=int(uni["qid"].notna().sum()), n_with_mines=int((uni["n_mines"] > 0).sum()),
        processed_this_run=processed, ok_this_run=ok, sec_requests=requests,
        n_with_financials=int(base["company_id"].nunique()) if len(base) else 0,
        n_financial_rows=int(len(base)),
        n_with_capex=int(base.loc[base["metric"] == "capex", "company_id"].nunique()) if len(base) else 0,
        n_with_employees=(int(emp.loc[emp["accepted"].astype(bool), "company_id"].nunique()) if len(emp)
                          else 0),
        n_segment_rows=int((seg["kind"] == "xbrl_segment").sum()) if len(seg) else 0,
        n_segment_matched=int(((seg["kind"] == "xbrl_segment") & seg["site_id"].notna()).sum()) if len(seg)
        else 0,
        n_announcements=int((seg["kind"] == "annonce_veille").sum()) if len(seg) else 0,
    )


# -- ratios ----------------------------------------------------------------------------------------------
def best_employees(emp: pd.DataFrame) -> pd.DataFrame:
    """Un effectif par société-année : SEC dei > rapport annuel > Wikidata (seules les valeurs acceptées)."""
    if emp.empty:
        return pd.DataFrame(columns=["company_id", "year", "employees", "source_kind", "source_url"])
    rank = {"sec_dei": 0, "rapport_annuel": 1, "wikidata": 2}
    e = emp[emp["accepted"].astype(bool)].copy()
    e["_r"] = e["source_kind"].map(rank).fillna(9)
    e = e.sort_values(["company_id", "year", "_r"]).drop_duplicates(["company_id", "year"])
    return e.drop(columns="_r").reset_index(drop=True)


def wide(fin: pd.DataFrame, emp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Une ligne par société-année, une colonne par mesure (+ concept utilisé pour les croissances)."""
    base = fin[fin["metric"].isin(list(CONCEPTS))] if len(fin) else fin
    if base.empty:
        return pd.DataFrame()
    w = base.pivot_table(index=["company_id", "year"], columns="metric", values="value", aggfunc="first")
    conc = base.pivot_table(index=["company_id", "year"], columns="metric", values="concept", aggfunc="first")
    for m in ("capex", "ppe_net"):
        if m in conc:
            w[f"{m}_concept"] = conc[m]
    unit = base.groupby(["company_id", "year"])["unit"].first()
    w["unit"] = unit
    w = w.reset_index()
    if emp is not None and len(emp):
        be = best_employees(emp)[["company_id", "year", "employees"]]
        w = w.merge(be, on=["company_id", "year"], how="left")
    return w


def ratios(w: pd.DataFrame) -> pd.DataFrame:
    """capex/CA, capex/salarié, capex/amortissements, croissances (années consécutives, même concept)."""
    if w.empty:
        return w
    w = w.sort_values(["company_id", "year"]).copy()
    for c in ("capex", "revenue", "ppe_net", "da", "employees", "capex_concept", "ppe_net_concept"):
        if c not in w:
            w[c] = np.nan
    w["capex_revenue"] = w["capex"] / w["revenue"].where(w["revenue"] > 0)
    w["capex_per_employee"] = w["capex"] / w["employees"].where(w["employees"] > 0)
    w["capex_da"] = w["capex"] / w["da"].where(w["da"] > 0)
    g = w.groupby("company_id")
    for m, out in (("capex", "capex_growth"), ("ppe_net", "ppe_growth")):
        prev, prev_y, prev_c = g[m].shift(), g["year"].shift(), g[f"{m}_concept"].shift()
        ok = (w["year"] - prev_y == 1) & (prev > 0) & (w[m] > 0) & (w[f"{m}_concept"] == prev_c)
        w[out] = (w[m] / prev - 1).where(ok)
    return w


# -- statistiques -----------------------------------------------------------------------------------------
def corr_stats(df: pd.DataFrame, x: str, y: str, group: str = "company_id", n_boot: int = 1000,
               seed: int = 0, min_n: int = 8, min_groups: int = 4) -> dict:
    """Pearson r et Spearman ρ, n, nombre de sociétés, IC 90 % par bootstrap SUR LES SOCIÉTÉS (les années
    d'une même société ne sont pas indépendantes), et un verdict honnête."""
    d = df[[group, x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n, k = len(d), d[group].nunique()
    res = dict(n=int(n), n_companies=int(k), pearson_r=np.nan, pearson_lo=np.nan, pearson_hi=np.nan,
               spearman_rho=np.nan, spearman_lo=np.nan, spearman_hi=np.nan)
    if n < 3 or d[x].nunique() < 3 or d[y].nunique() < 3:
        res["verdict"] = f"trop peu de données (n={n}, sociétés={k})"
        return res
    res["pearson_r"] = float(pearsonr(d[x], d[y]).statistic)
    res["spearman_rho"] = float(spearmanr(d[x], d[y]).statistic)
    groups = {g: v for g, v in d.groupby(group)}
    ids = list(groups)
    rng = np.random.default_rng(seed)
    pr, sr = [], []
    for _ in range(n_boot):
        b = pd.concat([groups[i] for i in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
        if b[x].nunique() < 3 or b[y].nunique() < 3:
            continue
        pr.append(pearsonr(b[x], b[y]).statistic)
        sr.append(spearmanr(b[x], b[y]).statistic)
    if pr:
        res["pearson_lo"], res["pearson_hi"] = (float(v) for v in np.nanpercentile(pr, [5, 95]))
        res["spearman_lo"], res["spearman_hi"] = (float(v) for v in np.nanpercentile(sr, [5, 95]))
    if n < min_n or k < min_groups:
        res["verdict"] = f"trop peu de données (n={n}, sociétés={k}) : chiffres indicatifs seulement"
    elif res["spearman_lo"] > 0:
        res["verdict"] = "lien positif démontré (IC 90 % de ρ au-dessus de 0)"
    elif res["spearman_hi"] < 0:
        res["verdict"] = "lien négatif démontré (IC 90 % de ρ sous 0)"
    else:
        res["verdict"] = "non démontré : l'intervalle à 90 % contient 0"
    return res


# -- satellite et production par société ---------------------------------------------------------------------
def _optical(site_id: str) -> pd.DataFrame:
    p = DATA_DIR / "signals" / "optical" / f"{site_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def mine_satellite_annual(site_id: str, min_quality: float = 0.5, min_months: int = 3,
                          reader: Callable[[str], pd.DataFrame] = _optical) -> pd.DataFrame:
    """Par an : médiane des mois clairs de la surface nue et de la surface nouvellement mise à nu."""
    d = reader(site_id)
    if d.empty:
        return pd.DataFrame(columns=["site", "year", "bare_km2", "new_bare_km2"])
    d = d[(d["quality"] >= min_quality) & d["signal"].isin(["bare_km2", "new_bare_km2"])].copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    cnt = d[d["signal"] == "bare_km2"].groupby("year").size()
    a = d.pivot_table(index="year", columns="signal", values="value", aggfunc="median")
    a = a[a.index.isin(cnt[cnt >= min_months].index)]
    for c in ("bare_km2", "new_bare_km2"):
        if c not in a:
            a[c] = np.nan
    return a.reset_index().assign(site=site_id)[["site", "year", "bare_km2", "new_bare_km2"]]


def company_satellite(uni: pd.DataFrame, analysed: set[str],
                      reader: Callable[[str], pd.DataFrame] = _optical) -> pd.DataFrame:
    """Expansion vue du ciel par société-année, sur ses mines analysées (pondérées par la part détenue
    quand elle est connue). Croissance = log(surface nue Y / Y−1) sur les MÊMES mines les deux années."""
    rows = []
    cache: dict[str, pd.DataFrame] = {}
    for _, r in uni.iterrows():
        ms = [m for m in mines_of(r) if m["site"] in analysed]
        if not ms:
            continue
        parts = []
        for m in ms:
            if m["site"] not in cache:
                cache[m["site"]] = mine_satellite_annual(m["site"], reader=reader)
            a = cache[m["site"]]
            if len(a):
                parts.append(a.assign(w=m.get("share") if m.get("share") is not None else 1.0))
        if not parts:
            continue
        a = pd.concat(parts, ignore_index=True)
        for y in sorted(a["year"].unique()):
            cur, prev = a[a["year"] == y].set_index("site"), a[a["year"] == y - 1].set_index("site")
            both = cur.index.intersection(prev.index)
            if not len(both):
                continue
            w = cur.loc[both, "w"]
            b1, b0 = (cur.loc[both, "bare_km2"] * w).sum(), (prev.loc[both, "bare_km2"] * w).sum()
            nb = (cur.loc[both, "new_bare_km2"] * w).sum()
            if b0 > 0 and b1 > 0:
                rows.append(dict(company_id=r["id"], year=int(y), sat_growth=math.log(b1 / b0),
                                 new_share=nb / b0, n_mines=int(len(both))))
    return pd.DataFrame(rows, columns=["company_id", "year", "sat_growth", "new_share", "n_mines"])


_SERIES_PREF = ["minerai extrait", "minerai traité", "charbon"]


def company_production(uni: pd.DataFrame, cat: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Croissance de production réelle (Jasansky 2023) par société-année : moyenne des croissances log de
    la série principale de chaque mine liée (minerai, sinon charbon, sinon le métal le mieux couvert)."""
    if prod.empty or cat.empty or "jasansky" not in cat:
        return pd.DataFrame(columns=["company_id", "year", "prod_growth", "n_mines"])
    by_fac = {k: g for k, g in prod.groupby("facility_id")}
    growth_cache: dict[str, pd.Series] = {}

    def mine_growth(fid: str) -> pd.Series:
        if fid in growth_cache:
            return growth_cache[fid]
        p = by_fac.get(fid)
        s = pd.Series(dtype=float)
        if p is not None:
            series = next((x for x in _SERIES_PREF if x in set(p["serie"])), None)
            if series is None:
                series = p.groupby("serie")["year"].nunique().idxmax()
            v = p[p["serie"] == series].groupby("year")["value_tonnes"].sum().sort_index()
            v = v[v > 0]
            g = np.log(v).diff()
            s = g[(v.index.to_series().diff() == 1).to_numpy()]
        growth_cache[fid] = s
        return s

    rows = []
    for _, r in uni.iterrows():
        gs = []
        for m in mines_of(r):
            if m["site"] in cat.index:
                fid = cat.at[m["site"], "jasansky"]
                if isinstance(fid, str):
                    s = mine_growth(fid)
                    if len(s):
                        gs.append(s)
        if not gs:
            continue
        df = pd.concat(gs, axis=1)
        mean, cnt = df.mean(axis=1), df.notna().sum(axis=1)
        for y, v in mean.dropna().items():
            rows.append(dict(company_id=r["id"], year=int(y), prod_growth=float(v), n_mines=int(cnt[y])))
    return pd.DataFrame(rows, columns=["company_id", "year", "prod_growth", "n_mines"])


# -- prévision du CAPEX (rétro-test) ------------------------------------------------------------------
# Chaque modèle ne voit que l'information disponible à la fin de l'exercice précédent. La persistance
# (« l'an prochain comme cette année ») est la référence à battre : un modèle qui ne la bat pas de façon
# démontrée n'est pas utilisé pour la prévision publiée.
FORECAST_MODELS = {
    "persistance": "CAPEX(Y) = CAPEX(Y−1)",
    "tendance": "CAPEX(Y) = CAPEX(Y−1) × croissance médiane des 3 derniers exercices",
    "cash_flow": "CAPEX(Y) = médiane passée de CAPEX/OCF(Y−1) × OCF(Y−1)",
    "ajustement_partiel": "moyenne géométrique de « persistance » et « cash_flow » (le CAPEX rattrape "
                          "lentement la trésorerie)",
}


def _predict(cap: pd.Series, ocf: pd.Series, y: int, min_hist: int = 3) -> dict[str, float] | None:
    """Prévisions de CAPEX(y) par chaque modèle, avec les seuls exercices < y. None si historique court."""
    hist = [t for t in cap.index if t < y and cap[t] > 0]
    if len(hist) < min_hist or (y - 1) not in hist:
        return None
    prev = cap[y - 1]
    out = {"persistance": prev}
    g = [math.log(cap[t] / cap[t - 1]) for t in hist[-3:] if (t - 1) in cap.index and cap[t - 1] > 0]
    if len(g) >= 2:
        out["tendance"] = prev * math.exp(float(np.median(g)))
    ks = [cap[t] / ocf[t - 1] for t in hist if (t - 1) in ocf.index and ocf[t - 1] > 0]
    o1 = ocf.get(y - 1, np.nan)
    if len(ks) >= 2 and o1 > 0:
        out["cash_flow"] = float(np.median(ks)) * o1
        out["ajustement_partiel"] = math.sqrt(prev * out["cash_flow"])
    return out


def forecast_backtest(w: pd.DataFrame, min_hist: int = 3) -> pd.DataFrame:
    """Une ligne par (société, année, modèle) : CAPEX prévu fin Y−1, CAPEX réel, CAPEX de Y−1."""
    rows = []
    cols = ["company_id", "year", "model", "actual", "prev", "pred"]
    if w.empty or "capex" not in w:
        return pd.DataFrame(columns=cols)
    for cid, g in w.sort_values("year").groupby("company_id"):
        g = g.set_index("year")
        cap = g["capex"].dropna()
        ocf = g["ocf"].dropna() if "ocf" in g else pd.Series(dtype=float)
        for y in cap.index:
            p = _predict(cap, ocf, y, min_hist)
            if not p or not cap[y] > 0:
                continue
            for m, v in p.items():
                rows.append(dict(company_id=cid, year=int(y), model=m, actual=cap[y], prev=cap[y - 1],
                                 pred=v))
    return pd.DataFrame(rows, columns=cols)


def forecast_next(w: pd.DataFrame, model: str, errors: np.ndarray, min_err: int = 20) -> pd.DataFrame:
    """Prévision de l'exercice qui suit le dernier publié, par `model`, intervalle à 90 % tiré des erreurs
    (log réel/prévu) du rétro-test de ce modèle ; pas d'intervalle sous `min_err` erreurs."""
    rows = []
    if w.empty or "capex" not in w:
        return pd.DataFrame()
    q05, q95 = np.percentile(errors, [5, 95]) if len(errors) >= min_err else (np.nan, np.nan)
    for cid, g in w.sort_values("year").groupby("company_id"):
        g = g.set_index("year")
        cap = g["capex"].dropna()
        if cap.empty:
            continue
        y = int(cap.index.max()) + 1
        ocf = g["ocf"].dropna() if "ocf" in g else pd.Series(dtype=float)
        p = (_predict(cap, ocf, y) or {}).get(model)
        if p is None:
            continue
        rows.append(dict(company_id=cid, year=y, value=p, unit=g.at[y - 1, "unit"], model=model,
                         lo90=p * math.exp(q05) if np.isfinite(q05) else np.nan,
                         hi90=p * math.exp(q95) if np.isfinite(q95) else np.nan))
    return pd.DataFrame(rows)


# -- rapport ----------------------------------------------------------------------------------------------
CORR_COLS = ["test", "label", "x", "y", "lag", "n", "n_companies", "pearson_r", "pearson_lo", "pearson_hi",
             "spearman_rho", "spearman_lo", "spearman_hi", "verdict", "details", "run_at"]


def _pairs(gw: pd.DataFrame, other: pd.DataFrame, col: str, lag: int) -> pd.DataFrame:
    o = other[["company_id", "year", col]].assign(year=other["year"] - lag)
    return gw.merge(o, on=["company_id", "year"], how="inner")


def report(*, cat: pd.DataFrame | None = None, prod: pd.DataFrame | None = None,
           analysed: Iterable[str] | None = None, reader: Callable[[str], pd.DataFrame] = _optical,
           n_boot: int = 1000, write: bool = True) -> dict:
    """Ratios, corrélations (avec r, ρ, n, IC 90 %, verdict) et prévision ; écrit correlations.parquet,
    ajoute ratios et prévisions à financials.parquet et met à jour status.json. Aucun réseau."""
    from strate import catalog, core

    uni, fin, emp = load("companies"), load("financials"), load("employees")
    run_at = _now()
    if uni.empty or fin.empty:
        st = dict(report_at=run_at, report_message="rien à analyser : lancer d'abord companies.update()")
        if write:
            _write_json(STATUS, {**_read_json(STATUS), **st})
        return st
    if cat is None:
        cat = catalog.load()
    if prod is None:
        try:
            prod = catalog.all_production()
        except (OSError, KeyError, ValueError) as e:
            log.warning("production Jasansky indisponible (%s)", e)
            prod = pd.DataFrame()
    analysed = set(analysed if analysed is not None else core.analysed("optical"))

    w = ratios(wide(fin, emp))
    w["capex_glog"] = np.log1p(w["capex_growth"])
    gw = w[["company_id", "year", "capex_glog"]].dropna()

    tests = []

    def add(test, label, x, y, lag, df, details=None):
        s = corr_stats(df, x, y, n_boot=n_boot)
        tests.append(dict(test=test, label=label, x=x, y=y, lag=lag, **s,
                          details=json.dumps(details or {}, ensure_ascii=False), run_at=run_at))

    sat = company_satellite(uni, analysed, reader=reader)
    for lag in (0, 1):
        quand = "même année" if lag == 0 else "année suivante"
        add("capex_vs_satellite", f"Croissance du CAPEX ↔ croissance de la surface nue des mines ({quand})",
            "capex_glog", "sat_growth", lag, _pairs(gw, sat, "sat_growth", lag),
            dict(mines_analysees=len(analysed), societes_avec_satellite=int(sat["company_id"].nunique())))
        add("capex_vs_new_bare",
            f"Croissance du CAPEX ↔ surface nouvellement mise à nu / surface nue ({quand})",
            "capex_glog", "new_share", lag, _pairs(gw, sat, "new_share", lag))
    pr = company_production(uni, cat, prod if prod is not None else pd.DataFrame())
    for lag in (0, 1):
        quand = "même année" if lag == 0 else "année suivante"
        add("capex_vs_production", f"Croissance du CAPEX ↔ croissance de la production réelle ({quand})",
            "capex_glog", "prod_growth", lag, _pairs(gw, pr, "prod_growth", lag),
            dict(source="Jasansky et al. 2023 (production jusqu'en 2019–2021)"))

    bt = forecast_backtest(w)
    fc = pd.DataFrame()
    if len(bt):
        bt = bt.assign(pred_g=np.log(bt["pred"] / bt["prev"]), act_g=np.log(bt["actual"] / bt["prev"]),
                       err=np.log(bt["actual"] / bt["pred"]))
        base = bt[bt["model"] == "persistance"].set_index(["company_id", "year"])["err"]
        best, best_err = "persistance", float(base.abs().median())
        for m in [m for m in FORECAST_MODELS if m != "persistance"]:
            d = bt[bt["model"] == m].set_index(["company_id", "year"])
            d = d.assign(err_p=base.reindex(d.index)).dropna(subset=["err_p"]).reset_index()
            if d.empty:
                continue
            gain = _boot_ratio(d, n_boot)
            beats = gain["hi"] < 1
            add("capex_forecast", f"Prévision du CAPEX, modèle « {m} » (rétro-test) : croissance prévue ↔ "
                "croissance réelle", "pred_g", "act_g", 1, d,
                dict(modele=FORECAST_MODELS[m], erreur_mediane_log=float(d["err"].abs().median()),
                     erreur_mediane_log_persistance=float(d["err_p"].abs().median()),
                     ratio_erreur_sur_persistance=gain,
                     verdict_gain=("bat la persistance (IC 90 % du ratio d'erreur sous 1)" if beats else
                                   "fait PIRE que la persistance (IC 90 % au-dessus de 1)" if gain["lo"] > 1
                                   else "ne bat pas la persistance de façon démontrée")))
            if beats and gain["value"] * best_err < best_err and m != best:
                best, best_err = m, float(d["err"].abs().median())
        errs = bt.loc[bt["model"] == best, "err"].to_numpy()
        fc = forecast_next(w, best, errs)
    corr = pd.DataFrame(tests, columns=CORR_COLS)

    if write:
        _write("correlations", corr)
        fin2 = _with_ratio_rows(fin, w, fc)
        _write("financials", fin2)
    st = dict(report_at=run_at, n_correlations=int(len(corr)),
              n_forecasts=int(len(fc)), correlations=json.loads(corr.drop(columns=["details"]).to_json(
                  orient="records", force_ascii=False)))
    if write:
        _write_json(STATUS, {**_read_json(STATUS), **st})
    return st


def _boot_ratio(bt: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """Erreur médiane du modèle / erreur médiane de la persistance, IC 90 % par société."""
    def f(d):
        den = d["err_p"].abs().median()
        return d["err"].abs().median() / den if den > 0 else np.nan
    groups = {g: v for g, v in bt.groupby("company_id")}
    ids = list(groups)
    rng = np.random.default_rng(seed)
    vals = [f(pd.concat([groups[i] for i in rng.choice(ids, len(ids))], ignore_index=True))
            for _ in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    lo, hi = (np.percentile(vals, [5, 95]) if vals else (np.nan, np.nan))
    return dict(value=float(f(bt)), lo=float(lo), hi=float(hi), n=int(len(bt)), n_companies=len(ids))


RATIO_METRICS = ["capex_revenue", "capex_per_employee", "capex_da", "capex_growth", "ppe_growth"]


def _with_ratio_rows(fin: pd.DataFrame, w: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    base = fin[fin["metric"].isin(list(CONCEPTS))]
    rows = []
    cik = dict(zip(base["company_id"], base["cik"], strict=False))
    for m in RATIO_METRICS:
        d = w[["company_id", "year", m, "unit"]].dropna(subset=[m])
        unit = {"capex_per_employee": None, "capex_revenue": "ratio", "capex_da": "ratio"}.get(m, "variation")
        for r in d.itertuples(index=False):
            rows.append(dict(company_id=r.company_id, cik=cik.get(r.company_id), year=int(r.year), metric=m,
                             value=float(getattr(r, m)), unit=unit or f"{r.unit}/salarié", concept="calcul",
                             form=None, accn=None, filed=None, period_end=None, source_url=None,
                             lo90=np.nan, hi90=np.nan))
    for r in fc.itertuples(index=False) if len(fc) else []:
        rows.append(dict(company_id=r.company_id, cik=cik.get(r.company_id), year=int(r.year),
                         metric="capex_prevision", value=float(r.value), unit=r.unit,
                         concept=f"modèle {r.model} : {FORECAST_MODELS[r.model]}", form=None,
                         accn=None, filed=None, period_end=None, source_url=None, lo90=r.lo90, hi90=r.hi90))
    out = pd.concat([base, pd.DataFrame(rows, columns=FIN_COLS)], ignore_index=True) if rows else base
    return out.reindex(columns=FIN_COLS)


# -- lecture pour l'API --------------------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _deep(v):
    """Nettoyage récursif pour JSON strict (NaN → None, types numpy → Python)."""
    if isinstance(v, dict):
        return {k: _deep(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep(x) for x in v]
    return _clean(v)


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def table() -> list[dict]:
    """Liste des sociétés pour l'écran « Sociétés » : dernier exercice publié et ratios clés."""
    uni = load("companies")
    if uni.empty:
        return []
    fin = load("financials")
    last = pd.DataFrame()
    if len(fin):
        keep = ["revenue", "capex", "capex_revenue", "capex_growth", "capex_per_employee"]
        f = fin[fin["metric"].isin(keep)]
        if len(f):
            ly = f[f["metric"] == "capex"].groupby("company_id")["year"].max()
            f = f[f.set_index("company_id").index.map(ly).to_numpy() == f["year"].to_numpy()]
            last = f.pivot_table(index="company_id", columns="metric", values="value", aggfunc="first")
            last["year"] = ly
            last["currency"] = fin[fin["metric"] == "capex"].groupby("company_id")["unit"].last()
    emp = best_employees(load("employees"))
    le = pd.DataFrame()
    if len(emp):
        le = emp.sort_values("year").groupby("company_id").last().reset_index()
        le = le.set_index("company_id", drop=False)
    out = []
    for r in uni.itertuples(index=False):
        L = last.loc[r.id].to_dict() if r.id in last.index else {}
        E = le.loc[r.id].to_dict() if r.id in le.index else {}
        out.append({k: _clean(v) for k, v in dict(
            id=r.id, name=r.name, tickers=_split_list(r.tickers), cik=r.cik, watchlist=bool(r.watchlist),
            n_mines=int(r.n_mines), year=L.get("year"), currency=L.get("currency"), revenue=L.get("revenue"),
            capex=L.get("capex"), capex_revenue=L.get("capex_revenue"), capex_growth=L.get("capex_growth"),
            capex_per_employee=L.get("capex_per_employee"), employees=E.get("employees"),
            employees_year=int(E["year"]) if E.get("year") is not None else None,
        ).items()})
    return out


def dossier(company_id: str) -> dict | None:
    """Fiche complète d'une société : identité, mines, comptes annuels avec sources, effectifs, segments,
    annonces, prévision. None si la société est inconnue."""
    uni = load("companies")
    if uni.empty or company_id not in set(uni["id"]):
        return None
    r = uni[uni["id"] == company_id].iloc[0]
    fin, emp, seg = load("financials"), load("employees"), load("segments")
    f = fin[fin["company_id"] == company_id] if len(fin) else fin
    years = []
    if len(f):
        for y, g in f[f["metric"] != "capex_prevision"].groupby("year"):
            row = {"year": int(y)}
            src = {}
            for x in g.itertuples(index=False):
                row[x.metric] = _clean(x.value)
                if isinstance(x.source_url, str):
                    src[x.metric] = dict(url=x.source_url, concept=x.concept, form=x.form, accn=x.accn)
            units = g.loc[g["metric"].isin(list(CONCEPTS)), "unit"]
            row["currency"] = _clean(units.iloc[0]) if len(units) else None
            row["sources"] = src
            years.append(row)
    fc = f[f["metric"] == "capex_prevision"] if len(f) else f
    e = emp[emp["company_id"] == company_id].sort_values("year") if len(emp) else emp
    s = seg[seg["company_id"] == company_id] if len(seg) else seg
    segs = s[s["kind"] == "xbrl_segment"] if len(s) else s
    segs = segs.sort_values(["year", "value"], ascending=[False, False]) if len(segs) else segs
    ann = s[s["kind"] == "annonce_veille"].sort_values("date", ascending=False) if len(s) else s
    return _deep(dict(
        company={k: _clean(v) for k, v in dict(
            id=r["id"], name=r["name"], aliases=_split_list(r["aliases"]), tickers=_split_list(r["tickers"]),
            cik=r["cik"], cik_source=r["cik_source"], qid=r["qid"], sources=_split_list(r["sources"]),
            watchlist=bool(r["watchlist"]), n_mines=int(r["n_mines"])).items()},
        mines=mines_of(r),
        financials=years,
        employees=_records(e.drop(columns=["company_id"])) if len(e) else [],
        segments=(_records(segs.drop(columns=["company_id", "kind", "title", "event_type", "date"]))
                  if len(segs) else []),
        announcements=_records(ann[["date", "site_id", "site_name", "event_type", "title", "source_url"]])
        if len(ann) else [],
        forecast=_records(fc[["year", "value", "unit", "lo90", "hi90", "concept"]]) if len(fc) else [],
        labels=METRIC_LABELS,
    ))


def correlations() -> list[dict]:
    c = load("correlations")
    if c.empty:
        return []
    out = _records(c)
    for x in out:
        try:
            x["details"] = json.loads(x["details"]) if x.get("details") else {}
        except ValueError:
            x["details"] = {}
    return out


def status() -> dict:
    return _deep(_read_json(STATUS))
