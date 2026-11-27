"""Catalogue mondial des mines : fusion de sources ouvertes, téléchargées chez l'utilisateur.

Sources (rien n'est republié dans le repo, chaque utilisateur télécharge depuis l'original) :
  icmm      ICMM Global Mining Dataset v1.5 (juil. 2026) — ~7 100 mines/usines actives nommées, CC BY 4.0
  wikidata  Toutes les mines géolocalisées de Wikidata (~55 000, dont carrières et mines fermées), CC0
  maus      Maus et al. 2022, 44 929 contours de mines tracés sur Sentinel-2 2019, CC BY-SA 4.0
  jasansky  Jasansky et al. 2023, production réelle de ~1 200 mines 2000–2021, CC BY 4.0

Fusion :
  1. ICMM = base (nom, commodités, type d'actif, niveau de confiance), mojibake réparé.
  2. Wikidata rapproché à < 2 km, ou < 10 km si les noms se ressemblent (nom principal avant alias,
     alias avant simple distance) → ajoute propriétaire, exploitant, article Wikipedia. Une 2e fiche de
     la même mine (même nom, < 2 km) est fondue dans la ligne au lieu de devenir un 2e site. Les mines
     Wikidata sans correspondance sont ajoutées.
  2b. Listes Wikipedia : nettoyées (rangs, articles de bassin, doublons entre listes), puis rapprochées
     de la même façon.
  3. Jasansky rapproché de la même façon, installation par installation (id mère « .00 », qui seul porte
     la production, par le meilleur de ses points), fiches ICMM d'abord → identifiant de production.
     Une productrice reprend la fiche qu'elle nomme à une installation sans production ; une productrice
     restée seule rejoint un site du même nom exact jusqu'à 100 km. « has_production » = id + production.
  4. Contours Maus : chaque mine reçoit les contours à < 2 km (emprise réelle → zone d'analyse
     satellite exacte). Les contours restants, regroupés à 1 km, deviennent des sites non nommés.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
import zipfile
from difflib import SequenceMatcher
from functools import cache, lru_cache
from urllib.parse import unquote

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from strate import entities
from strate.core import DATA_DIR

log = logging.getLogger(__name__)

SRC = DATA_DIR / "sources"
OUT = DATA_DIR / "catalog"
UA = {"User-Agent": "strate/0.1 (open-source research tool)"}
EARTH_KM = 6371.0
METRIC = "EPSG:8857"  # Equal Earth : surfaces justes partout sur le globe

SOURCES = {
    "icmm": dict(
        url="https://www.icmm.com/website/data/2026/global-mining-dataset-1-5.xlsx",
        file="icmm/global-mining-dataset-1-5.xlsx",
        cite="ICMM Global Mining Dataset, July 2026 (V1.5) release — CC BY 4.0",
    ),
    "maus": dict(
        url="https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg",
        file="maus2022/global_mining_polygons_v2.gpkg",
        cite="Maus et al. (2022), Global-scale mining polygons v2, PANGAEA, "
        "https://doi.org/10.1594/PANGAEA.942325 — CC BY-SA 4.0",
    ),
    "jasansky": dict(
        url="https://zenodo.org/records/7369478/files/open_database_mine_production.zip?download=1",
        file="jasansky2023/open_database_mine_production.zip",
        cite="Jasansky et al. (2023), Open database on global coal and metal mine production, "
        "https://doi.org/10.5281/zenodo.7369478 — CC BY 4.0",
    ),
    "wikidata": dict(url="https://query.wikidata.org/sparql", file="wikidata/mines.parquet",
                     cite="Wikidata — CC0"),
    "wplists": dict(url="https://en.wikipedia.org/wiki/Lists_of_mines", file="wplists/mines.parquet",
                    cite="Wikipedia, « Lists of mines » et sous-listes — CC BY-SA 4.0"),
}


# -- téléchargement ---------------------------------------------------------------------
def fetch(force: bool = False) -> None:
    with httpx.Client(headers=UA, timeout=300, follow_redirects=True) as c:
        for name, s in SOURCES.items():
            path = SRC / s["file"]
            if path.exists() and not force:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "wikidata":
                wikidata_mines(c).to_parquet(path, index=False)
            elif name == "wplists":
                wikipedia_lists(c).to_parquet(path, index=False)
            else:
                r = c.get(s["url"])
                r.raise_for_status()
                path.write_bytes(r.content)
            log.warning("téléchargé %s → %s", name, path)


# -- Wikidata ---------------------------------------------------------------------------
# Le serveur coupe toute requête > 60 s. Stratégie robuste : 1 requête rapide pour la liste des mines
# (coordonnées, pays, type), puis noms / propriétés / articles par LOTS de QID (VALUES), en POST.
_WD_BASE = """SELECT ?item ?coord ?country ?type WHERE {
  ?item wdt:P31 ?type ; wdt:P625 ?coord . ?type wdt:P279* wd:Q820477 .
  OPTIONAL { ?item wdt:P17 ?country } }"""
_WD_PROPS = {"owner": "P127", "operator": "P137", "product": "P1056"}
_LANGS = ["en", "mul", "fr", "es", "de", "pt", "it", "pl", "cs", "sv", "ru", "zh", "ja"]
BATCH = 1500


def _sparql(c: httpx.Client, q: str) -> list[dict]:
    for attempt in range(6):
        try:
            r = c.post(SOURCES["wikidata"]["url"], data={"query": q},
                       headers={"Accept": "application/sparql-results+json"})
        except httpx.TransportError as e:  # délai dépassé, connexion coupée…
            log.warning("Wikidata : %s, nouvel essai", type(e).__name__)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("retry-after", 5 * (attempt + 1))))
            continue
        r.raise_for_status()
        try:
            return r.json()["results"]["bindings"]
        except ValueError:  # réponse tronquée : délai serveur dépassé en plein envoi
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikidata : requête trop lourde ou serveur indisponible")


def _qid(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def _title(v) -> str | None:
    """Titre d'article en clair (« Rössing_uranium_mine »), jamais encodé en %XX.

    Les URL d'articles de Wikidata arrivent encodées ; les titres des listes Wikipedia, non. Un titre
    encodé est refusé par MediaWiki et ré-encodé (%25C3…) par le module attention → aucune donnée.
    """
    if not isinstance(v, str) or not v:
        return None
    return unquote(v).replace(" ", "_")


def _batches(qids: list[str], size: int = BATCH):
    for i in range(0, len(qids), size):
        yield " ".join(f"wd:{q}" for q in qids[i:i + size])


def _labels(c: httpx.Client, qids) -> dict[str, str]:
    """Meilleur libellé de chaque QID, dans l'ordre de préférence _LANGS."""
    langs = ", ".join(f'"{lang}"' for lang in _LANGS)
    rank = {lang: i for i, lang in enumerate(_LANGS)}
    best: dict[str, tuple[int, str]] = {}
    for values in _batches(sorted(q for q in set(qids) if q)):
        q = f"""SELECT ?x ?l WHERE {{ VALUES ?x {{ {values} }}
          ?x rdfs:label ?l . FILTER(LANG(?l) IN ({langs})) }}"""
        for b in _sparql(c, q):
            k, lab = _qid(b["x"]["value"]), b["l"]
            r = rank.get(lab.get("xml:lang"), 99)
            if k not in best or r < best[k][0]:
                best[k] = (r, lab["value"])
    return {k: v for k, (_, v) in best.items()}


def wikidata_mines(c: httpx.Client) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for b in _sparql(c, _WD_BASE):
        qid = _qid(b["item"]["value"])
        m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", b["coord"]["value"])
        if not m or qid in rows:  # 1re coordonnée seulement ; ignore les globes non terrestres
            continue
        rows[qid] = {"wikidata": qid, "lon": float(m[1]), "lat": float(m[2]),
                     "country": _qid(b.get("country", {}).get("value")),
                     "wd_type": _qid(b["type"]["value"])}
    df = pd.DataFrame(rows.values())
    qids = df["wikidata"].tolist()
    log.warning("Wikidata : %d mines, récupération des détails par lots de %d", len(qids), BATCH)

    props: dict[str, dict[str, set[str]]] = {col: {} for col in _WD_PROPS}
    arts: dict[str, str] = {}
    pvals = " ".join(f"wdt:{p}" for p in _WD_PROPS.values())
    pcol = {p: col for col, p in _WD_PROPS.items()}
    for values in _batches(qids):
        q = f"""SELECT ?item ?p ?v WHERE {{ VALUES ?item {{ {values} }} VALUES ?p {{ {pvals} }}
          ?item ?p ?v . }}"""
        for b in _sparql(c, q):
            col = pcol[_qid(b["p"]["value"])]
            props[col].setdefault(_qid(b["item"]["value"]), set()).add(_qid(b["v"]["value"]))
        q = f"""SELECT ?item ?article WHERE {{ VALUES ?item {{ {values} }}
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}"""
        for b in _sparql(c, q):
            arts[_qid(b["item"]["value"])] = _title(_qid(b["article"]["value"]))

    needed = set(qids) | set(df["country"].dropna()) | set(df["wd_type"].dropna())
    needed |= {v for d in props.values() for s in d.values() for v in s}
    lab = _labels(c, needed)
    df["name"] = df["wikidata"].map(lab)
    df["country"] = df["country"].map(lambda q: lab.get(q, q) if q else None)
    df["wd_type"] = df["wd_type"].map(lambda q: lab.get(q, q))
    for col, vals in props.items():
        df[col] = df["wikidata"].map(
            lambda k, vals=vals: "; ".join(sorted(lab.get(x, x) for x in vals.get(k, ()))) or None)
    df["wikipedia"] = df["wikidata"].map(arts)
    return df


# -- Listes de mines de Wikipedia --------------------------------------------------------------
# « Lists of mines » et ses sous-listes : tableaux par pays / métal. Beaucoup de lignes renvoient vers
# un article dont la fiche Wikidata n'est PAS classée « mine » (ex. Mount Whaleback) : elles manquent
# à la requête Wikidata. On ne lit que la 1re colonne (le nom de la mine), jamais les autres colonnes
# (villes, sociétés…) pour ne pas importer des lieux qui ne sont pas des mines.
WP_API = "https://en.wikipedia.org/w/api.php"
WP_ROOT = "Lists of mines"
_WP_SKIP = ("lost mines", "deepest mines", "largest gold mines by production", "mining disasters",
            "mineral production", "by production")


def _wp(c: httpx.Client, **params) -> dict:
    params |= {"format": "json", "formatversion": "2", "maxlag": "5"}
    for attempt in range(6):
        try:
            r = c.get(WP_API, params=params, timeout=30)
        except httpx.TransportError as e:
            log.warning("Wikipedia : %s, nouvel essai", type(e).__name__)
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(float(r.headers.get("retry-after", 3 * (attempt + 1))), 60)
            log.warning("Wikipedia : HTTP %s, attente %.0f s", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("API Wikipedia indisponible")


def _is_list(title: str) -> bool:
    tl = title.lower()
    return tl.startswith(("list of", "lists of")) and any(w in tl for w in ("mine", "collier", "quarr")) \
        and not any(s in tl for s in _WP_SKIP)


def wp_list_pages(c: httpx.Client, depth: int = 2) -> list[str]:
    seen, frontier = {WP_ROOT}, [WP_ROOT]
    for _ in range(depth):
        nxt = []
        for page in frontier:
            res = _wp(c, action="parse", page=page, prop="links", redirects=1)
            links = res.get("parse", {}).get("links", [])
            for lk in links:
                ttl = lk["title"]
                if lk.get("ns") == 0 and lk.get("exists", True) and _is_list(ttl) and ttl not in seen:
                    seen.add(ttl)
                    nxt.append(ttl)
        frontier = nxt
    return sorted(seen - {WP_ROOT})


_COORD = re.compile(r"\{\{\s*coord\s*\|([^{}]*)\}\}", re.I)


def parse_coord(body: str) -> tuple[float, float] | None:
    """{{coord|…}} → (lat, lon). Gère décimal, d|m|N et d|m|s|N."""
    parts = [p.strip() for p in body.split("|") if "=" not in p]
    for i, p in enumerate(parts):
        if p.upper() in ("N", "S"):
            lat_p, rest = parts[:i], parts[i + 1:]
            j = next((k for k, q in enumerate(rest) if q.upper() in ("E", "W")), None)
            if j is None:
                return None
            try:
                def dms(xs):
                    xs = [float(x) for x in xs if x]
                    return sum(v / 60 ** k for k, v in enumerate(xs))
                lat = dms(lat_p) * (-1 if p.upper() == "S" else 1)
                lon = dms(rest[:j]) * (-1 if rest[j].upper() == "W" else 1)
            except ValueError:
                return None
            return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


def _clean_cell(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    while re.search(r"\{\{[^{}]*\}\}", s):  # modèles imbriqués : de l'intérieur vers l'extérieur
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\{\{.*$", "", s, flags=re.S)  # modèle ouvert sur la ligne suivante (« Mounana{{cite web »)
    s = _LINK.sub(lambda m: m.group(2) or m.group(1), s)
    s = re.sub(r"'{2,}|<[^>]+>|\}\}", "", s)
    s = re.sub(r"\s+", " ", s)  # tabulations et retours parasites (« Rössing\t »)
    return s.strip(" |!*†")


def _strip_attrs(cell: str) -> str:
    """Retire les attributs de cellule (« nowrap | », « style=… | ») : 1er « | » hors [[ ]] et {{ }}."""
    depth = 0
    for i, ch in enumerate(cell):
        if cell.startswith(("[[", "{{"), i):
            depth += 1
        elif cell.startswith(("]]", "}}"), i):
            depth -= 1
        elif ch == "|" and depth == 0:
            return cell[i + 1:]
    return cell


def _is_name(s: str) -> bool:
    """Un nom de mine contient au moins une lettre : « 8 », « – », « 14 » sont des rangs ou des vides."""
    return bool(s) and len(s) <= 80 and re.search(r"[^\W\d_]", s) is not None


_RANK_HDR = re.compile(r"(rank|rang|no\.?|n°|#|nr\.?|number|s\.? ?no\.?)", re.I)
_NAME_HDR = re.compile(r"(name|mine|mines|mine name|name of mine|colliery|collieries|quarry|project|deposit|"
                       r"operation|site|pit|mine ?/ ?project)( name)?", re.I)


def _cells(lines: list[str]) -> list[str]:
    """Cellules d'une ligne de tableau, qu'elles soient sur une ligne (« a || b ») ou une par ligne."""
    out = []
    for ln in lines:
        sep = "!!" if ln.startswith("!") else "||"
        out += [_strip_attrs(c) for c in ln[1:].split(sep)]
    return out


def _name_col(header: list[str]) -> int:
    """Colonne du nom : celle titrée « Name / Mine… », sinon la 1re, sauf si la 1re est un rang."""
    h = [_clean_cell(c).lower() for c in header]
    for k, x in enumerate(h):
        if _NAME_HDR.fullmatch(x):
            return k
    return 1 if h and _RANK_HDR.fullmatch(h[0]) and len(h) > 1 else 0


def parse_list(wikitext: str) -> list[dict]:
    """Lignes de tableaux : nom (colonne « Name/Mine », sinon la 1re), article lié, coordonnées."""
    out = []
    for table in re.findall(r"\{\|.*?\n\|\}", wikitext, flags=re.S):
        chunks = re.split(r"\n\|-[^\n]*", table)
        header: list[str] = []
        for chunk in chunks:  # en-tête : avant le 1er « |- », ou 1re ligne faite de « ! »
            hl = [ln for ln in chunk.strip().split("\n") if ln.startswith("!")]
            if hl:
                header = _cells(hl)
                break
        col = _name_col(header)
        for row in chunks[1:]:
            lines = [ln for ln in row.strip().split("\n") if ln.startswith(("|", "!"))]
            if not lines or lines[0].startswith("!"):
                continue
            cells = _cells(lines)
            if col >= len(cells):
                continue
            first = cells[col]
            link = _LINK.search(first)
            name = _clean_cell(first)
            co = _COORD.search(row)
            if not _is_name(name):
                continue
            out.append({"name": name, "article": link.group(1).strip() if link else None,
                        "coord": parse_coord(co.group(1)) if co else None})
    return out


def _hints(page: str) -> tuple[str | None, str | None]:
    tl = page.lower()
    comm = next((c for c in ("gold", "copper", "coal", "iron", "diamond", "uranium", "nickel", "lithium",
                             "silver", "zinc", "potash", "salt", "bauxite") if c in tl), None)
    if "collier" in tl:
        comm = "coal"
    m = re.search(r" in (?:the )?(.+)$", page)
    return comm, (m.group(1) if m else None)


def wikipedia_lists(c: httpx.Client) -> pd.DataFrame:
    rows = []
    pages = wp_list_pages(c)
    log.warning("listes Wikipedia : %d pages", len(pages))
    for k, page in enumerate(pages, 1):
        if k % 10 == 0:
            log.warning("  page %d/%d", k, len(pages))
        res = _wp(c, action="parse", page=page, prop="wikitext", redirects=1)
        wt = res.get("parse", {}).get("wikitext", "")
        comm, where = _hints(page)
        for r in parse_list(wt):
            rows.append({**r, "list_page": page, "commodity_hint": comm, "where_hint": where})
    df = pd.DataFrame(rows)
    # coordonnées + QID des articles liés (par 50, redirects suivis)
    arts = sorted({a for a in df["article"].dropna()})
    info: dict[str, dict] = {}
    log.warning("listes Wikipedia : coordonnées de %d articles", len(arts))
    for i in range(0, len(arts), 50):
        q = _wp(c, action="query", titles="|".join(arts[i:i + 50]), prop="coordinates|pageprops",
                ppprop="wikibase_item", redirects=1, colimit="max")["query"]
        alias = {x["from"]: x["to"] for x in q.get("redirects", []) + q.get("normalized", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for a in arts[i:i + 50]:
            tgt = alias.get(alias.get(a, a), alias.get(a, a))
            p = pages.get(tgt, {})
            co = (p.get("coordinates") or [None])[0]
            info[a] = {"art_title": tgt if p and not p.get("missing") else None,
                       "art_coord": (co["lat"], co["lon"]) if co else None,
                       "art_qid": p.get("pageprops", {}).get("wikibase_item")}
    df = df.join(df["article"].map(lambda a: info.get(a, {})).apply(pd.Series))
    # Coordonnée propre à la ligne, ou empruntée à l'article lié : si cet article est un bassin ou un
    # district (« Selby Coalfield »), elle ne situe pas la mine. _clean_wplists() en a besoin.
    df["coord_own"] = df["coord"].notna()
    coord = df["coord"].where(df["coord"].notna(), df.get("art_coord"))
    df = df[coord.notna()].copy()
    coord = coord[coord.notna()]
    df["lat"] = [x[0] for x in coord]
    df["lon"] = [x[1] for x in coord]
    df["wikipedia"] = df["art_title"].map(lambda x: x.replace(" ", "_") if isinstance(x, str) else None)
    df["wikidata"] = df["art_qid"]
    df = df.drop_duplicates(subset=["name", "lat", "lon"])
    return df[["name", "lat", "lon", "wikipedia", "wikidata", "list_page", "commodity_hint", "where_hint",
               "coord_own"]]


# -- lecteurs ---------------------------------------------------------------------------
# 1er octet d'un caractère UTF-8 relu en Latin-1 / Windows-1252 : 0xC2–0xF4 → « Â » à « ô ».
# Large exprès : une réparation n'est gardée que si les octets forment de l'UTF-8 valide (voir fix_text),
# donc un vrai « é » ou « ö » isolé reste intact.
_MOJI = re.compile("[\u00c2-\u00f4]")


def _bytes_1252(s: str) -> bytes | None:
    """Les octets d'origine d'un texte décodé à tort en Windows-1252 (repli Latin-1 pour 0x81, 0x8D…)."""
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 0xFF:
                return None
            out.append(ord(ch))
    return bytes(out)


def _fix_word(v: str) -> str:
    for _ in range(3):  # deux passes pour le double encodage, une de marge
        if not _MOJI.search(v):
            break
        raw = _bytes_1252(v)
        try:
            fixed = raw.decode("utf-8") if raw is not None else v
        except UnicodeDecodeError:
            break
        if fixed == v:
            break
        v = fixed
    return v


def fix_text(v):
    """Répare le « mojibake » : UTF-8 relu en Latin-1, parfois deux fois (« PyhÃƒÂ¤salmi » → « Pyhäsalmi »).

    Mot par mot (espaces ASCII, « ; », « , ») : certaines listes d'alias mêlent la forme juste et la forme
    abîmée (« Rönnskär; … RÃ¶nnskÃ¤r »). Une réparation n'est gardée que si les octets forment de l'UTF-8
    valide : un vrai « Ä » (Äänekoski) n'en forme pas et reste intact. Sans ça, « RÃ¶ssing » ne
    ressemble plus à « Rössing » et la mine ICMM n'est pas rapprochée de sa fiche Wikidata (emprise,
    production) : deux sites pour une mine.
    """
    if not isinstance(v, str) or not _MOJI.search(v):
        return v
    return re.sub(r"[^ \t\n;,]+", lambda m: _fix_word(m.group()), v)


def load_icmm() -> pd.DataFrame:
    d = pd.read_excel(SRC / SOURCES["icmm"]["file"], sheet_name="Global Mining Dataset v1.5")
    d.columns = [c.strip() for c in d.columns]
    d = d.dropna(subset=["Latitude", "Longitude"])
    for col in d.columns:  # noms, alias, pays : ~140 valeurs en mojibake dans la v1.5
        if col not in ("Latitude", "Longitude"):
            d[col] = d[col].map(fix_text)
    comm = d[["Primary Commodity", "Secondary Commodity", "Other Commodities"]].astype("string")
    return pd.DataFrame({
        "icmm": d["ICMMID"],
        "name": d["Mine Name"].astype("string").str.strip(),
        "aliases": d["Group Names"],
        "lat": pd.to_numeric(d["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(d["Longitude"], errors="coerce"),
        "country": d["Country"],
        "region": d["Region"],
        "asset_type": d["Asset Type"],
        "commodities": comm.apply(lambda r: _commodities(r.dropna()), axis=1),
        "confidence": d["Confidence Factor"],
    }).dropna(subset=["lat", "lon"])


def _commodities(vals) -> str | None:
    out = []
    for v in vals:
        for x in re.split(r"[;,/]", str(v)):
            x = x.strip().lower()
            if x and x not in out:
                out.append(x)
    return ", ".join(out) or None


def _parent(fid: pd.Series) -> pd.Series:
    """« COM00272.01 » → « COM00272.00 » : l'installation mère, la seule qui porte la production."""
    return fid.astype(str).str.replace(r"\.\d+$", ".00", regex=True)


def _join_names(*vals) -> str | None:
    out: list[str] = []
    for v in vals:
        if isinstance(v, str):
            for x in re.split(r"[;,]", v):
                x = x.strip()
                if x and x not in out:
                    out.append(x)
    return ";".join(out) or None


def load_jasansky() -> pd.DataFrame:
    """Installations Jasansky : une ligne par POINT, toutes rattachées à l'installation mère « .00 ».

    Toute la production est publiée sur l'id mère (aucune ligne sur .01, .02…) ; les sous-installations
    (« Chuquicamata » et « Radomiro Tomic » pour Codelco Norte) ne sont que des points. Rapprocher un
    site d'un sous-id le laissait sans production, et la mère, orpheline, perdait sa vérité terrain.
    Chaque point garde son nom de sous-site (plus précis) ; le nom de l'installation devient un alias.
    Colonnes : jasansky (id mère) · name · aliases · lat · lon · facility_type · status.
    """
    z = SRC / SOURCES["jasansky"]["file"]
    with zipfile.ZipFile(z) as zf:
        tmp = z.parent / "x"
        zf.extractall(tmp)
    f = gpd.read_file(tmp / "data" / "facilities.gpkg")
    f["jasansky"] = _parent(f["facility_id"])
    f["is_sub"] = f["facility_id"] != f["jasansky"]
    # statut et type : ceux de la mère, sinon le 1er renseigné parmi ses sous-sites
    meta = f.sort_values("is_sub", kind="stable").groupby("jasansky")[
        ["facility_type", "activity_status"]].first()
    sub_name = f["sub_site_name"].where(f["is_sub"])
    f["pt_name"] = sub_name.where(sub_name.notna(), f["facility_name"])
    f["pt_aliases"] = [
        _join_names(so, fn, fo) if sub and isinstance(sn, str) else _join_names(fo)
        for sub, sn, so, fn, fo in zip(f["is_sub"], f["sub_site_name"], f["sub_site_other_names"],
                                       f["facility_name"], f["facility_other_names"], strict=True)]
    f = f[[g is not None and not g.is_empty for g in f.geometry]]
    pts = f.explode(index_parts=False)  # le MultiPoint de la mère = les points de ses sous-sites
    rp = pts.geometry.representative_point()
    out = pd.DataFrame({
        "jasansky": pts["jasansky"].values, "name": pts["pt_name"].values,
        "aliases": pts["pt_aliases"].values,
        "lat": rp.y.values, "lon": rp.x.values, "is_sub": pts["is_sub"].values,
    })
    # le point de la mère double celui d'un sous-site : on garde le sous-site, mieux nommé
    out = out.sort_values("is_sub", ascending=False, kind="stable")
    key = pd.DataFrame({"j": out["jasansky"], "la": out["lat"].round(5), "lo": out["lon"].round(5)})
    out = out[~key.duplicated()]
    out = out.sort_index(kind="stable").drop(columns="is_sub")
    out["facility_type"] = out["jasansky"].map(meta["facility_type"])
    out["status"] = out["jasansky"].map(meta["activity_status"])
    return out.reset_index(drop=True)


_METALS = {"Cu": "cuivre", "Au": "or", "Ag": "argent", "Zn": "zinc", "Pb": "plomb", "Ni": "nickel",
           "Fe": "fer", "Mo": "molybdène", "Co": "cobalt", "U": "uranium", "Pt": "platine",
           "Pd": "palladium", "Li": "lithium", "Sn": "étain", "Mn": "manganèse", "Al": "aluminium"}
# Base du métal publié, par ordre de préférence : le métal récupéré dans le concentré (ce que la mine
# produit), sinon le métal contenu dans le minerai traité. Les deux décrivent le même métal à deux
# étapes : les additionner le compte deux fois (COM00402 2013 : 2,4 Mt de Fe concentré + 3,15 Mt de Fe
# dans le minerai = 5,55 Mt). Base lue dans minerals.type via id_minerals, sinon dans le préfixe du
# matériau (« Con. » / « O. »).
_METAL_BASIS = {"Concentrate": 0, "Ore processed": 1, "Ore mined": 2}
_MATERIAL_BASIS = {"Con": "Concentrate", "O": "Ore processed"}


def _metal_rows(c: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Lignes de métal ramenées à une seule base par installation · année · métal."""
    mtype = m.set_index("id")["type"]
    basis = c["id_minerals"].map(mtype)
    guess = c["material"].astype(str).str.split(".").str[0].map(_MATERIAL_BASIS)
    rank = basis.where(basis.notna(), guess).map(_METAL_BASIS).fillna(9)
    c = c.assign(_rank=rank)
    best = c.groupby(["facility_id", "year", "commodity"])["_rank"].transform("min")
    # la lixiviation (cathodes SX-EW) est un flux À PART du concentré : on l'ajoute, on ne la jette pas
    proc = c["mine_processing"] if "mine_processing" in c else pd.Series(index=c.index, dtype=object)
    leach = proc.astype(str).str.lower() == "leaching"
    return c[(c["_rank"] == best) | leach].drop(columns="_rank")


def _coal_rows(k: pd.DataFrame) -> pd.DataFrame:
    """Une seule mesure du charbon par installation · année : « Coal mined », sinon « Clean coal ».

    coal.csv donne les deux pour chaque année (souvent égales) : les additionner doublait tout le
    charbon (log10 2 ≈ 0,3 d'erreur sur les modèles calibrés). « Coal mined » = tout-venant extrait,
    la même base que « minerai extrait » : c'est ce que le satellite voit creuser.
    """
    pref = k["type"].map({"Coal mined": 0, "Clean coal": 1}).fillna(2)
    best = pref.groupby([k["facility_id"], k["year"]]).transform("min")
    return k[pref == best]


@cache
def all_production() -> pd.DataFrame:
    """Toute la production réelle (Jasansky et al. 2023), chargée une fois : minerai, métal, charbon.

    Colonnes : facility_id · year · serie · value_tonnes · source_id. Plusieurs lignes d'une même série
    ne s'additionnent que si elles décrivent des matériaux différents sur une même base (Cu dans le
    concentré de cuivre + Cu dans le concentré de zinc), jamais deux bases du même tonnage.
    """
    d = SRC / "jasansky2023" / "x" / "data"
    if not (d / "minerals.csv").exists():
        if not (SRC / SOURCES["jasansky"]["file"]).exists():  # source pas encore téléchargée
            log.warning("production réelle absente : lancer `strate catalog` pour la télécharger")
            return pd.DataFrame(columns=["facility_id", "year", "serie", "value_tonnes", "source_id"])
        load_jasansky()  # extrait l'archive si besoin
    m = pd.read_csv(d / "minerals.csv", low_memory=False)
    c = _metal_rows(pd.read_csv(d / "commodities.csv", low_memory=False), m)
    m = m.assign(serie=m["type"].map({"Ore mined": "minerai extrait", "Ore processed": "minerai traité",
                                      "Concentrate": "concentré"}).fillna(m["type"]))
    sym = c["commodity"].astype(str).str.removeprefix("Me.")
    c = c.assign(serie=sym.map(lambda s: f"{_METALS.get(s, s)} (métal)"))
    k = _coal_rows(pd.read_csv(d / "coal.csv", low_memory=False)).assign(serie="charbon")
    cols = ["facility_id", "year", "serie", "value_tonnes", "source_id"]
    df = pd.concat([x[cols] for x in (m, c, k)], ignore_index=True)
    return df.groupby(["facility_id", "year", "serie"], as_index=False).agg(
        value_tonnes=("value_tonnes", "sum"), source_id=("source_id", "first"))


def production(facility_id: str) -> pd.DataFrame:
    """Production réelle par an d'une installation : year · serie · value_tonnes · source_id."""
    p = all_production()
    return p[p["facility_id"] == facility_id].drop(columns="facility_id").reset_index(drop=True)


# -- rapprochement ------------------------------------------------------------------------
def _xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def _chord(km: float) -> float:
    return 2 * np.sin(km / EARTH_KM / 2)


# Mots sans valeur pour reconnaître une mine : retirés du nom (« Spence mine » = « Spence »).
_STOP = {"mine", "mines", "mining", "minera", "minero", "mineracao", "mineração", "project", "projects",
         "operation", "operations", "the", "of", "and", "de", "la", "le", "les", "mina", "minas", "pit",
         "pits", "open", "cut", "underground", "complex", "deposit", "deposits", "co", "company", "ltd",
         "inc", "corp", "corporation", "plc", "sa", "llc", "pty", "mill", "plant", "site", "idle"}
# Mots gardés mais NON distinctifs : à eux seuls, ils ne prouvent pas que deux noms désignent la même
# mine (« Mount Keith » ≠ « Mount Margaret », « Gold Creek » ≠ « Silver Creek », « Grube X » ≠ « Grube Y »).
_GENERIC = {
    # relief, eau, orientation, taille
    "mount", "mt", "mont", "monte", "montana", "cerro", "sierra", "mountain", "mountains", "hill", "hills",
    "ridge", "peak", "valley", "creek", "river", "lake", "lakes", "gulch", "canyon", "spring", "springs",
    "bay", "island", "field", "fields", "flat", "flats", "point", "rock", "rocks", "well", "wells", "dam",
    "north", "south", "east", "west", "northern", "southern", "eastern", "western", "central", "norte",
    "sur", "este", "oeste", "upper", "lower", "new", "old", "big", "little", "great", "grand", "deep",
    "main", "extension", "ext", "nueva", "nuevo", "novo", "nova", "black", "white", "red", "blue", "green",
    # articles et saints (multilingue)
    "el", "los", "las", "lo", "da", "do", "das", "dos", "di", "du", "des", "del", "der", "die", "den", "van",
    "von", "san", "santa", "santo", "sao", "são", "saint", "st", "ste", "sainte", "nuestra", "senora", "y",
    # numérotation et type d'ouvrage
    "no", "nr", "number", "colliery", "collieries", "grube", "gruva", "gruvan", "gruvor", "gruvfält",
    "zeche", "kopalnia", "kwk", "rudnik", "quarry", "quarries", "shaft", "shafts", "works", "smelter",
    "refinery", "concentrator", "steel", "district", "area", "zone", "property", "prospect", "claim",
    "claims", "placer", "lode", "vein", "reef", "coalfield", "group", "resources", "consolidated",
    "united", "joint", "venture", "division", "unit", "block", "seam", "surface",
    # substances
    "gold", "copper", "coal", "iron", "ore", "silver", "zinc", "lead", "nickel", "tin", "uranium",
    "lithium", "diamond", "diamonds", "platinum", "cobalt", "manganese", "chrome", "chromite", "bauxite",
    "potash", "salt", "phosphate", "limestone", "gypsum", "graphite", "rare", "earth", "earths", "sand",
    "gravel", "granite", "marble", "slate", "clay", "kaolin", "tungsten", "molybdenum", "vanadium",
    "mineral", "minerals", "metals", "oro", "plata", "cobre", "carbon", "hierro", "ouro", "ferro",
}


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


# comparés après repli ASCII : « mineração » → « mineracao »
_STOP = {_fold(w) for w in _STOP}
_GENERIC = {_fold(w) for w in _GENERIC}


# Chiffres romains I à XX : « Neyveli II » = « Neyveli 2 », un numéro d'unité comme un chiffre arabe
_ROMAN = {r: str(k) for k, r in enumerate(
    "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split(), 1)}


def _filler(t: str) -> bool:
    """Mot qui peut suivre un numéro sans en faire un mot du nom (« Kolubara D and E coal Mine »)."""
    return t in _STOP or t in _GENERIC or t.isdigit() or t in _ROMAN or len(t) <= 2


@lru_cache(maxsize=200_000)
def _tokens_of(s: str) -> frozenset[str]:
    """Mots utiles d'un nom. Les NUMÉROS restent, ils séparent deux unités d'un même site.

    Numéros : chiffres (« Mine No 3 » ≠ « Mine No 6 »), chiffres romains convertis (« Neyveli I » ≠
    « Neyveli II », « King II » = « King 2 ») et lettre seule en suffixe (« Kolar A » ≠ « Kolar B »,
    « Field a » ≠ « Field D »). Une lettre seule n'est PAS un numéro en tête de nom, collée à une
    apostrophe (« d'Or », « King's », « O'Brien ») ou suivie d'un mot du nom (initiales « J R Simplot ») :
    elle est ignorée comme avant. « U.S. » devient « us ».
    """
    f = _fold(s)  # l'apostrophe courbe disparaît (« Zhunge’er » → « zhungeer »), la droite sépare
    f = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), f)
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+", f)]
    out = []
    for k, (t, a, b) in enumerate(words):
        if t in _STOP:
            continue
        single = len(t) == 1 and t.isalpha()
        if single or t in _ROMAN:
            glued = (a > 0 and f[a - 1] == "'") or (b < len(f) and f[b] == "'")
            suffix = k > 0 and not glued and all(_filler(w) for w, _, _ in words[k + 1:])
            if len(t) > 1 or suffix:  # « ii », « iv » : numéros partout ; « i », « v », « x » en suffixe
                out.append(_ROMAN.get(t, t))
            continue
        out.append(t)
    return frozenset(out)


@lru_cache(maxsize=200_000)
def _seq_of(s: str) -> tuple[str, ...]:
    """Tous les mots d'un nom dans l'ordre, sauf « mine », « the »… (lettres seules et romains compris)."""
    f = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), _fold(s))
    return tuple(_ROMAN.get(t, t) for t in re.findall(r"[a-z0-9]+", f) if t not in _STOP)


def _word_seq(s) -> tuple[str, ...]:
    return _seq_of(s) if isinstance(s, str) else ()


def _tokens(s) -> frozenset[str]:
    return _tokens_of(s) if isinstance(s, str) else frozenset()


def _close(a: str, b: str) -> bool:
    """Même mot, ou faute d'orthographe sur un mot long (« Lidell » / « Liddell ») ; jamais un nombre."""
    if a == b:
        return True
    if a.isdigit() or b.isdigit() or min(len(a), len(b)) < 5 or a[0] != b[0] or abs(len(a) - len(b)) > 2:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def _is_num(t: str) -> bool:
    """Numéro d'unité : chiffres (romains déjà convertis par _tokens_of) ou lettre seule en suffixe."""
    return t.isdigit() or len(t) == 1


def _core(ta: frozenset[str]) -> set[str]:
    """Mots DISTINCTIFS d'un nom : ni génériques, ni numéros, ni codes de 2 lettres (« WV », « KY » : un
    État ou un pays, pas le nom de la mine)."""
    return {t for t in ta if t not in _GENERIC and not _is_num(t) and not (len(t) <= 2 and t.isalpha())}


def _sim_tokens(ta: frozenset[str], tb: frozenset[str], exact: bool = False) -> bool:
    """Deux noms se ressemblent s'ils partagent au moins la moitié de leurs mots DISTINCTIFS.

    Les numéros servent seulement à SÉPARER (« Mine No 3 » ≠ « Mine No 6 ») : ils ne comptent jamais
    comme mot partagé, sinon « Prosperity Mine No. 1 » et « Rockhouse 1 Mine » se ressembleraient.
    `exact` : mêmes mots distinctifs des deux côtés (pour fondre deux fiches d'une même mine).
    """
    if not ta or not tb:
        return False
    na, nb = {t for t in ta if _is_num(t)}, {t for t in tb if _is_num(t)}
    if (na or nb) and na != nb:  # numéro d'un seul côté ou numéros différents : puits différents
        if na and nb:
            return False
        if exact:
            return False
    da, db = _core(ta), _core(tb)
    if not da and not db:  # noms entièrement génériques (« Gold Creek », « No 2 ») : identiques ou rien
        def strip(x):  # sans les codes d'État ; les numéros (chiffres, lettres) restent
            return {t for t in x if not (len(t) == 2 and t.isalpha())}
        return strip(ta) == strip(tb)
    if not da or not db:
        return False
    if exact:
        return len(da) == len(db) and all(any(_close(t, u) for u in db) for t in da)
    small, big = (da, db) if len(da) <= len(db) else (db, da)
    shared = sum(any(_close(t, u) for u in big) for t in small)
    return shared / len(small) >= 0.5


def _name_set(v) -> list[frozenset[str]]:
    """Chaque nom d'une liste « a;b;c » à part : la chaîne concaténée mêlerait les mots de plusieurs mines."""
    if not isinstance(v, str):
        return []
    return [t for t in (_tokens(x) for x in v.split(";")) if t]


def _similar(a, b) -> bool:
    """Vrai si un des noms de `a` ressemble à un des noms de `b` (listes « ; » acceptées)."""
    return any(_sim_tokens(x, y) for x in _name_set(a) for y in _name_set(b))


def _pairs(base: pd.DataFrame, other: pd.DataFrame, near_km: float, far_km: float,
           name_cols: tuple[str, ...]) -> list[tuple[int, float, int, int]]:
    """Couples candidats (rang du nom, distance km, i de other, j de base).

    Rang du nom : 3 = nom principal ↔ nom principal, 2 = nom principal de base ↔ un alias de `other`
    (le nom d'installation porté par un sous-site Jasansky), 1 = un alias de base seulement, 0 = aucun
    (gardé seulement à < near_km). Les alias ICMM (« Group Names ») listent
    les autres actifs du groupe : un voisin reconnu par alias ne doit pas passer devant la mine
    dont c'est le nom (« Asam Asam » prenait le charbon de « Mulia/Jumbang » à 200 m).
    """
    if base.empty or other.empty:
        return []
    tree = cKDTree(_xyz(base["lat"], base["lon"]))
    oxyz = _xyz(other["lat"], other["lon"])
    ok = np.isfinite(oxyz).all(axis=1)
    cols = [c for c in name_cols if c in base]
    bcols = [base[c].to_numpy(object) for c in cols]
    onames = other["name"].to_numpy(object)
    oalias = other["aliases"].to_numpy(object) if "aliases" in other else np.full(len(other), None)
    bcache: dict[int, tuple[list, list]] = {}
    pairs = []
    for i in np.flatnonzero(ok):
        idxs = tree.query_ball_point(oxyz[i], _chord(far_km))
        if not idxs:
            continue
        op, oa = _name_set(onames[i]), _name_set(oalias[i])
        d = np.linalg.norm(tree.data[idxs] - oxyz[i], axis=1) / _chord(1.0)  # ≈ km
        for j, dist in zip(idxs, d, strict=True):
            if j not in bcache:
                ns = [_name_set(col[j]) for col in bcols]
                bcache[j] = (ns[0] if ns else [], [t for x in ns[1:] for t in x])
            primary, aliases = bcache[j]
            tier = 0
            if any(_sim_tokens(a, b) for a in op for b in primary):
                tier = 3
            elif any(_sim_tokens(a, b) for a in oa for b in primary):
                tier = 2
            elif any(_sim_tokens(a, b) for a in op + oa for b in aliases):
                tier = 1
            if tier or dist <= near_km:
                pairs.append((tier, float(dist), int(i), int(j)))
    return pairs


def match(base: pd.DataFrame, other: pd.DataFrame, near_km=2.0, far_km=10.0,
          name_cols=("name", "aliases"), group: pd.Series | None = None,
          prefer: pd.Series | None = None) -> pd.Series:
    """Pour chaque ligne de `other`, l'index de la ligne de `base` correspondante (ou -1).

    Règle : nom principal ressemblant à < far_km, puis alias ressemblant à < far_km, puis la plus proche à
    < near_km ; à rang égal, la plus proche. Chaque ligne de base n'est prise qu'une fois.
    `group` : les lignes d'un même groupe (points d'une même installation) ne prennent qu'UNE ligne de base,
    par leur meilleur point. `prefer` : à 100 m près, la ligne préférée gagne (installation avec production).
    """
    pairs = _pairs(base, other, near_km, far_km, tuple(name_cols))
    grp = group.to_numpy(object) if group is not None else np.arange(len(other))
    pref = prefer.to_numpy(bool) if prefer is not None else np.zeros(len(other), bool)
    if prefer is not None:
        pairs.sort(key=lambda p: (-p[0], round(p[1], 1), not pref[p[2]], p[1], p[2], p[3]))
    else:
        pairs.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
    used_o, used_b, out = set(), set(), np.full(len(other), -1)
    for _, _, i, j in pairs:
        if grp[i] in used_o or j in used_b:
            continue
        used_o.add(grp[i])
        used_b.add(j)
        out[i] = j
    return pd.Series(out, index=other.index)


def fold(base: pd.DataFrame, other: pd.DataFrame, m: pd.Series, near_km=2.0) -> pd.Series:
    """Doublons de `other` restés sans ligne après match() : la ligne de base déjà prise qu'ils doublent.

    Wikidata a souvent deux fiches pour une mine (une ancienne avec l'article, une récente sans) ; la
    correspondance un-pour-un n'en garde qu'une et l'autre devenait un 2e site à 10 m. Ici, une ligne
    restée seule rejoint la ligne de base déjà rapprochée la plus proche à < near_km dont le nom
    PRINCIPAL ressemble (les alias ICMM désignent d'autres actifs du groupe : pas de repli par alias).
    """
    left = other[m.reindex(other.index).to_numpy() < 0]
    taken = set(m[m >= 0].astype(int))
    out = pd.Series(-1, index=other.index)
    if left.empty or not taken:
        return out
    best: dict[int, tuple[float, int]] = {}
    lnames, bnames = left["name"].to_numpy(object), base["name"].to_numpy(object)
    for tier, dist, i, j in _pairs(base, left, near_km, near_km, ("name",)):
        if tier < 2 or dist > near_km or j not in taken or (i in best and dist >= best[i][0]):
            continue
        # fondre exige les MÊMES mots distinctifs : « Doctor Divide Mine » n'est pas « Tonopah Divide »
        if any(_sim_tokens(a, b, exact=True) for a in _name_set(lnames[i]) for b in _name_set(bnames[j])):
            best[i] = (dist, j)
    for i, (_, j) in best.items():
        out.loc[left.index[i]] = j
    return out


# -- emprise réelle (contours Maus) ------------------------------------------------------------
def footprints(sites: pd.DataFrame, poly: gpd.GeoDataFrame, attach_km=2.0, cluster_km=1.0,
               priority: pd.Series | None = None):
    """Attache à chaque site les contours à < attach_km ; regroupe les contours orphelins.

    `priority` (0 = d'abord) : les contours vont d'abord aux sites prioritaires (mines ICMM vérifiées),
    puis seulement aux autres. Sans ça, une petite mine Wikidata voisine d'une grande fosse peut
    « voler » toute l'emprise de la grande mine (vu au Chili : 287 km² sur une galerie historique).
    """
    pm = poly.to_crs(METRIC)
    pts = gpd.GeoDataFrame(sites[[]], geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
                           crs=4326).to_crs(METRIC)
    if priority is None:
        priority = pd.Series(0, index=sites.index)
    prio = priority.reindex(sites.index).fillna(9)
    parts_j, left = [], pm
    for tier in sorted(prio.unique()):
        cand = pts[prio == tier]
        if left.empty or cand.empty:
            continue
        jt = gpd.sjoin_nearest(left, cand, how="left", max_distance=attach_km * 1000, distance_col="d")
        jt = jt[~jt.index.duplicated(keep="first")]
        parts_j.append(jt[jt["index_right"].notna()])
        left = left.loc[jt[jt["index_right"].isna()].index]
    j = pd.concat(parts_j + [left.assign(index_right=np.nan)]) if parts_j else left.assign(index_right=np.nan)
    attached = j[j["index_right"].notna()].copy()
    attached["index_right"] = attached["index_right"].astype(int)
    area = attached.groupby("index_right")["AREA"].sum()
    bounds = attached.to_crs(4326).geometry.bounds.assign(site=attached["index_right"]).groupby("site").agg(
        minx=("minx", "min"), miny=("miny", "min"), maxx=("maxx", "max"), maxy=("maxy", "max"))
    fp = pd.DataFrame({"footprint_km2": area}).join(bounds)
    fp["footprint_polygons"] = attached.groupby("index_right").size()

    orphans = pm.loc[j[j["index_right"].isna()].index]
    groups = orphans.buffer(cluster_km * 500).union_all()
    parts = gpd.GeoDataFrame(geometry=list(getattr(groups, "geoms", [groups])), crs=METRIC)
    o = gpd.sjoin(orphans, parts, how="left", predicate="intersects")
    o = o[~o.index.duplicated(keep="first")]
    o4 = o.to_crs(4326)
    agg = o4.groupby("index_right").agg(
        footprint_km2=("AREA", "sum"), country=("COUNTRY_NAME", "first"), n=("AREA", "size"))
    b = o4.geometry.bounds.assign(g=o["index_right"]).groupby("g").agg(
        minx=("minx", "min"), miny=("miny", "min"), maxx=("maxx", "max"), maxy=("maxy", "max"))
    agg = agg.join(b)
    clat, clon = (agg["miny"] + agg["maxy"]) / 2, (agg["minx"] + agg["maxx"]) / 2
    unnamed = pd.DataFrame({
        # identifiant stable d'une reconstruction à l'autre : le centre de l'emprise (≈ 100 m)
        "id": [f"fp_{la:.3f}_{lo:.3f}" for la, lo in zip(clat, clon, strict=True)],
        "name": None,
        "lat": clat,
        "lon": clon,
        "country": agg["country"],
        "asset_type": "Mine (emprise satellite)",
        "footprint_km2": agg["footprint_km2"],
        "footprint_polygons": agg["n"],
        "minx": agg["minx"], "miny": agg["miny"], "maxx": agg["maxx"], "maxy": agg["maxy"],
        "sources": "maus",
    }).drop_duplicates("id").reset_index(drop=True)
    return fp, unnamed


# -- construction ---------------------------------------------------------------------------
WD_COLS = ["wikidata", "name", "lon", "lat", "country", "wd_type", "owner", "operator", "product",
           "wikipedia"]


def _km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _add_aliases(cat: pd.DataFrame, j, *names) -> None:
    """Ajoute des noms aux alias d'une ligne, sans doublon (casse ignorée) ni répétition du nom."""
    cur = cat.at[j, "aliases"]
    have = [x for x in (cur.split(";") if isinstance(cur, str) and cur else [])]
    seen = {x.strip().lower() for x in have} | {str(cat.at[j, "name"]).strip().lower()}
    for v in names:
        for x in (v.split(";") if isinstance(v, str) else []):
            x = x.strip()
            if x and x.lower() not in seen:
                have.append(x)
                seen.add(x.lower())
    cat.at[j, "aliases"] = ";".join(have) or None


def _absorb(cat: pd.DataFrame, j, row: pd.Series, source: str | None = None) -> None:
    """Fond une fiche doublon dans la ligne j : son nom devient un alias, son article comble un manque.

    L'article et le QID vont ensemble : si la ligne n'a pas d'article et le doublon en a un, la ligne
    prend son QID aussi (c'est la fiche « principale », celle que lit le module attention).
    """
    _add_aliases(cat, j, row.get("name"), row.get("aliases"))
    if pd.isna(cat.at[j, "wikipedia"]) and isinstance(row.get("wikipedia"), str):
        cat.at[j, "wikipedia"] = row["wikipedia"]
        if isinstance(row.get("wikidata"), str):
            cat.at[j, "wikidata"] = row["wikidata"]
    elif pd.isna(cat.at[j, "wikidata"]) and isinstance(row.get("wikidata"), str):
        cat.at[j, "wikidata"] = row["wikidata"]
    for col in ("owner", "operator"):
        if col in cat and pd.isna(cat.at[j, col]) and isinstance(row.get(col), str):
            cat.at[j, col] = row[col]
    if source and source not in str(cat.at[j, "sources"]).split("+"):
        cat.at[j, "sources"] = f"{cat.at[j, 'sources']}+{source}"


def merge_wikidata(icmm: pd.DataFrame, wd: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """ICMM + Wikidata. Rend le catalogue et les QID fondus dans une ligne (QID → index de la ligne)."""
    icmm = icmm.copy()
    wd = wd.copy()
    wd["wikipedia"] = wd["wikipedia"].map(_title)  # anciennes extractions : titres encodés en %XX
    m = match(icmm, wd)
    hit = m[m >= 0]
    for col in ("wikidata", "wikipedia", "owner", "operator", "wd_type"):
        icmm[col] = None
        icmm[col] = icmm[col].astype(object)
        icmm.loc[hit.values, col] = wd.loc[hit.index, col].values
    icmm.loc[hit.values, "sources"] = "icmm+wikidata"
    # le nom Wikidata devient un alias (« Mount Whaleback » ↔ ICMM « Mount Newman »)
    icmm.loc[hit.values, "aliases"] = [
        ";".join(x for x in (a, n) if isinstance(x, str) and x)
        for a, n in zip(icmm.loc[hit.values, "aliases"], wd.loc[hit.index, "name"], strict=True)]
    f = fold(icmm, wd, m)
    folded: dict[str, int] = {}
    for i, j in f[f >= 0].items():
        _absorb(icmm, j, wd.loc[i])
        if isinstance(wd.at[i, "wikidata"], str):
            folded[wd.at[i, "wikidata"]] = int(j)
    extra = wd[(m < 0) & (f < 0)].copy()
    extra["id"] = extra["wikidata"].str.lower()
    extra["asset_type"] = extra["wd_type"].str.capitalize()
    extra["commodities"] = extra["product"]
    extra["sources"] = "wikidata"
    if len(f[f >= 0]):
        log.warning("Wikidata : %d fiches en double fondues dans une mine ICMM", int((f >= 0).sum()))
    extra, same = _dedup_same_point(extra)
    pos = {lab: len(icmm) + k for k, lab in enumerate(extra.index)}  # index après concat (ICMM d'abord)
    folded.update({q: pos[lab] for q, lab in same.items()})
    return pd.concat([icmm, extra.drop(columns=["product"])], ignore_index=True), folded


def _dedup_same_point(rows: pd.DataFrame, km: float = 0.5) -> tuple[pd.DataFrame, dict[str, object]]:
    """Mines Wikidata sans fiche ICMM en double : même nom distinctif au même point → une seule ligne.

    fold() ne fond un doublon que dans une ligne ICMM déjà rapprochée ; sans ICMM à < 2 km, les deux
    fiches d'une même mine (« Majdanpek » or + argent, « RTB Bor ») restaient deux sites au même point,
    dont un seul recevait l'emprise. Garde-fous : < `km`, les mêmes mots dans le même ordre hors
    « mine », « project »… (génériques, lettres et numéros compris : « A Tunnel » ≠ « B Tunnel »,
    « El Dorado North » ≠ « El Dorado South », « Little Buck » ≠ « Big Buck », « Neyveli I » ≠
    « Neyveli II », sections cadastrales US « NW SE Section 30 » ≠ « SE NW Section 30 ») et au moins un
    mot distinctif (deux « Quarry » voisines ne sont pas forcément la même).
    La ligne gardée : celle qui a un article, sinon le QID le plus ancien.
    Rend les lignes restantes et, pour chaque QID fondu, l'étiquette de la ligne qui l'a absorbé.
    """
    rows = rows.copy()
    if "aliases" not in rows:
        rows["aliases"] = None
    rows["aliases"] = rows["aliases"].astype(object)
    xyz = _xyz(rows["lat"], rows["lon"])
    fin = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    if len(fin) < 2:
        return rows, {}
    toks = [_tokens(n) for n in rows["name"]]
    seq = [_word_seq(n) for n in rows["name"]]
    links = [(int(fin[a]), int(fin[b])) for a, b in cKDTree(xyz[fin]).query_pairs(_chord(km))
             if _core(toks[fin[a]]) and seq[fin[a]] == seq[fin[b]]]
    if not links:
        return rows, {}

    def qnum(q) -> int:
        return int(q[1:]) if isinstance(q, str) and q[1:].isdigit() else 1 << 62

    rank = [(not isinstance(w, str), qnum(q), k)
            for k, (w, q) in enumerate(zip(rows["wikipedia"], rows["wikidata"], strict=True))]
    labels, drop, same = rows.index, [], {}
    for comp in _components(len(rows), links):
        if len(comp) < 2:
            continue
        comp = sorted(comp, key=lambda x: rank[x])
        keep = labels[comp[0]]
        for x in comp[1:]:
            row = rows.loc[labels[x]]
            _absorb(rows, keep, row)
            cm = [c.strip() for v in (rows.at[keep, "commodities"], row.get("commodities"))
                  if isinstance(v, str) for c in v.split(";") if c.strip()]
            rows.at[keep, "commodities"] = "; ".join(dict.fromkeys(cm)) or None
            if isinstance(row.get("wikidata"), str):
                same[row["wikidata"]] = keep
            drop.append(labels[x])
    log.warning("Wikidata : %d fiches en double au même point fondues (sans fiche ICMM)", len(drop))
    return rows.drop(index=drop), same


def _components(n: int, links) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return list(groups.values())


def clean_wplists(w: pd.DataFrame, same_km: float = 25.0, dup_km: float = 1.0) -> pd.DataFrame:
    """Lignes des listes Wikipedia prêtes à fusionner : noms propres, articles vérifiés, doublons fondus.

    1. Noms : rangs (« 8 »), cellules vides et restes de modèles (« Mounana{{cite web », « Rössing\t »).
    2. Article partagé par plusieurs lignes : si aucune ne porte son nom (« Mount Sicker »), si elles
       portent trois noms distincts ou plus (« Selby Coalfield » pour six houillères) ou si elles
       s'étalent sur plus de `same_km` (« Sundergarh », un district), l'article est un bassin / district /
       relief, pas une mine : son QID et son lien sont retirés, et les lignes qui
       n'avaient que SES coordonnées (aucune à elles) sont écartées, faute de lieu. Sinon c'est l'article
       de la mine : les lignes proches sont la même mine, citée sur plusieurs listes.
    3. Doublons : même QID, ou < `dup_km` avec un nom ressemblant → une ligne, les autres noms en alias.
    """
    w = w.copy().reset_index(drop=True)
    w["name"] = w["name"].map(lambda x: _clean_cell(x) if isinstance(x, str) else None)
    ok = w["name"].map(lambda x: isinstance(x, str) and _is_name(x)).astype(bool)
    if (~ok).any():
        log.warning("listes Wikipedia : %d lignes sans nom de mine écartées (rangs, cellules vides)",
                    int((~ok).sum()))
    w = w[ok].reset_index(drop=True)
    w["wikipedia"] = w["wikipedia"].map(_title).astype(object)
    w["wikidata"] = w["wikidata"].where(w["wikidata"].notna(), None).astype(object)
    if "aliases" not in w:
        w["aliases"] = None
    own = w["coord_own"].astype(bool) if "coord_own" in w else None

    key = w["wikidata"].where(w["wikidata"].notna(), w["wikipedia"])
    drop: list[int] = []
    for _, g in w[key.notna()].groupby(key[key.notna()]):
        if len(g) < 2:
            continue
        title = next((t.replace("_", " ") for t in g["wikipedia"] if isinstance(t, str)), None)
        about = g["name"].map(lambda n, t=title: _similar(n, t)).to_numpy(bool)
        ab = g[about]
        spread = _km(ab["lat"].to_numpy()[:, None], ab["lon"].to_numpy()[:, None],
                     ab["lat"].to_numpy()[None, :], ab["lon"].to_numpy()[None, :]).max() if len(ab) else 0.0
        names = g["name"].tolist()
        n_mines = len(_components(len(names), [(x, y) for x in range(len(names))
                                               for y in range(x + 1, len(names))
                                               if _similar(names[x], names[y])]))
        # deux noms = ancien et nouveau nom d'une mine (Bernic Lake / Tanco) ; trois ou plus = un bassin
        if len(ab) and spread <= same_km and n_mines <= 2:
            rest = g[~about]
            if len(rest):
                d = _km(rest["lat"].to_numpy()[:, None], rest["lon"].to_numpy()[:, None],
                        ab["lat"].to_numpy()[None, :], ab["lon"].to_numpy()[None, :]).min(axis=1)
                w.loc[rest.index[d > same_km], ["wikipedia", "wikidata"]] = None
            continue
        w.loc[g.index, ["wikipedia", "wikidata"]] = None
        if own is not None:
            borrowed = g.index[~own.loc[g.index].to_numpy()]
        else:  # extraction sans « coord_own » : plusieurs lignes au même point = celui de l'article
            borrowed = g.index[g[["lat", "lon"]].round(4).duplicated(keep=False).to_numpy()]
        drop += list(borrowed)
    if drop:
        log.warning("listes Wikipedia : %d lignes situées seulement par un article de bassin/district "
                    "écartées : %s", len(drop), ", ".join(w.loc[drop, "name"].head(12)))
    w = w.drop(index=drop).reset_index(drop=True)

    links = []
    for _, idx in w.dropna(subset=["wikidata"]).groupby("wikidata").groups.items():
        links += list(zip(idx[:-1], idx[1:], strict=True))
    if len(w) > 1:
        tree = cKDTree(_xyz(w["lat"], w["lon"]))
        q = w["wikidata"].to_numpy(object)
        names = w["name"].to_numpy(object)
        for a, b in tree.query_pairs(_chord(dup_km)):
            if isinstance(q[a], str) and isinstance(q[b], str) and q[a] != q[b]:
                continue
            if _similar(names[a], names[b]):
                links.append((a, b))
    # la ligne gardée : celle qui a un QID, un article, qui porte le nom de l'article, ses propres coordonnées
    titled = [not _similar(n, t.replace("_", " ") if isinstance(t, str) else None)
              for n, t in zip(w["name"], w["wikipedia"], strict=True)]
    rank = list(zip(w["wikidata"].isna(), w["wikipedia"].isna(), titled,
                    ~own.to_numpy() if own is not None else np.zeros(len(w), bool), range(len(w)),
                    strict=True))
    keep, merged = [], 0
    for comp in _components(len(w), links):
        comp = sorted(comp, key=lambda x: rank[x])
        r = comp[0]
        keep.append(r)
        if len(comp) > 1:
            merged += len(comp) - 1
            w.at[r, "aliases"] = _join_alias(w.at[r, "name"], *(w.at[x, "name"] for x in comp[1:]),
                                             *(w.at[x, "aliases"] for x in comp))
            for col in ("wikipedia", "wikidata"):
                if not isinstance(w.at[r, col], str):
                    w.at[r, col] = next((w.at[x, col] for x in comp if isinstance(w.at[x, col], str)), None)
    if merged:
        log.warning("listes Wikipedia : %d doublons (même mine sur plusieurs listes) fondus", merged)
    return w.loc[sorted(keep)].reset_index(drop=True)


def _join_alias(name, *vals) -> str | None:
    seen, out = {str(name).strip().lower()}, []
    for v in vals:
        for x in (v.split(";") if isinstance(v, str) else []):
            x = x.strip()
            if x and x.lower() not in seen:
                seen.add(x.lower())
                out.append(x)
    return ";".join(out) or None


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _fold(str(s))).strip("-")[:40] or "x"


def _unique_ids(new: pd.DataFrame, taken: set[str]) -> list[str]:
    """QID si libre, sinon « wp_lat_lon », sinon « wp_lat_lon_nom » : jamais deux mines sous un même id."""
    out = []
    for q, la, lo, n in zip(new["wikidata"], new["lat"], new["lon"], new["name"], strict=True):
        i = q.lower() if isinstance(q, str) else f"wp_{la:.3f}_{lo:.3f}"
        if i in taken:
            stem = f"wp_{la:.3f}_{lo:.3f}_{_slug(n)}"
            i, k = stem, 2
            while i in taken:
                i, k = f"{stem}_{k}", k + 1
        taken.add(i)
        out.append(i)
    return out


def merge_wplists(cat: pd.DataFrame, wpl: pd.DataFrame, folded: dict[str, int] | None = None) -> pd.DataFrame:
    """Listes Wikipedia : comble les mines absentes de Wikidata « mine » (ex. Mount Whaleback)."""
    cat = cat.copy()
    for col in ("aliases", "wikipedia", "wikidata"):
        cat[col] = cat[col].astype(object) if col in cat else None
    wpl = clean_wplists(wpl)
    # déjà là via Wikidata (ou fondue dans une ligne) : le nom de la liste devient seulement un alias
    with_q = cat.reset_index().dropna(subset=["wikidata"]).drop_duplicates("wikidata")
    by_q = {**(folded or {}), **with_q.set_index("wikidata")["index"].to_dict()}
    known = wpl["wikidata"].isin(list(by_q))
    for q, n, a in zip(wpl.loc[known, "wikidata"], wpl.loc[known, "name"], wpl.loc[known, "aliases"],
                       strict=True):
        _add_aliases(cat, by_q[q], n, a)
    wpl = wpl[~known].reset_index(drop=True)
    mw = match(cat, wpl)
    hit = mw[mw >= 0]
    for col in ("wikipedia", "wikidata"):
        tgt = cat.loc[hit.values, col]
        cat.loc[hit.values, col] = tgt.where(tgt.notna(), wpl.loc[hit.index, col].values)
    cat.loc[hit.values, "sources"] = cat.loc[hit.values, "sources"] + "+wikipedia"
    # le nom de la liste devient un alias : « Whaleback » retrouve le site même s'il s'appelle autrement
    for i, j in hit.items():
        _add_aliases(cat, j, wpl.at[i, "name"], wpl.at[i, "aliases"])
    # même mine citée sous un autre nom, restée seule (« Moose River Gold Mine » à 470 m de « Moose River »)
    fw = fold(cat, wpl, mw)
    for i, j in fw[fw >= 0].items():
        _absorb(cat, j, wpl.loc[i], source="wikipedia")
    new = wpl[(mw < 0) & (fw < 0)].copy()
    new["id"] = _unique_ids(new, set(cat["id"]))
    new["commodities"] = new["commodity_hint"]
    new["asset_type"] = "Mine (liste Wikipedia)"
    new["country"] = None  # « in Oregon », « in Western Australia » : une région, pas un pays
    new["region"] = new["where_hint"]
    new["sources"] = "wikipedia"
    new = new.drop(columns=[c for c in ("commodity_hint", "where_hint", "list_page", "coord_own")
                            if c in new])
    log.warning("listes Wikipedia : %d rapprochées, %d fondues, %d ajoutées", len(hit), int((fw >= 0).sum()),
                len(new))
    return pd.concat([cat, new], ignore_index=True)


def link_jasansky(cat: pd.DataFrame, jas: pd.DataFrame, with_production: set[str]) -> pd.DataFrame:
    """Rattache chaque installation Jasansky (id mère, par le meilleur de ses points) à UN site.

    D'abord aux mines ICMM dont le nom ressemble, puis à tous les sites : comme pour les contours, la
    fiche ICMM vérifiée passe avant une fiche Wikidata voisine (« Collaguasi » à 1,4 km d'un sous-site
    volait la production de Collahuasi). Sans nom, la distance seule ne suffit pas à doubler la priorité.
    """
    cat = cat.copy()
    pref = jas["jasansky"].isin(with_production)
    first = cat[cat["icmm"].notna()] if "icmm" in cat else cat.iloc[:0]
    m1 = match(first, jas, near_km=-1.0, group=jas["jasansky"], prefer=pref)  # nom obligatoire
    m1 = m1.map(lambda k: first.index[k] if k >= 0 else -1)
    done = set(jas.loc[m1 >= 0, "jasansky"])
    rest_b = cat.drop(index=list(m1[m1 >= 0]))
    rest_o = jas[~jas["jasansky"].isin(done)]
    m2 = match(rest_b, rest_o, group=rest_o["jasansky"], prefer=pref.loc[rest_o.index])
    m2 = m2.map(lambda k: rest_b.index[k] if k >= 0 else -1)
    mj = _swap_producers(cat, jas, pd.concat([m1[m1 >= 0], m2[m2 >= 0]]), pref)
    m3 = _link_far(cat.drop(index=list(mj)), jas[jas["jasansky"].isin(set(with_production)
                                                 - set(jas.loc[mj.index, "jasansky"]))])
    mj = pd.concat([mj, m3])
    hit = mj[mj >= 0]
    cat["jasansky"] = None
    cat["jasansky"] = cat["jasansky"].astype(object)
    cat.loc[hit.values, "jasansky"] = jas.loc[hit.index, "jasansky"].values
    cat["production_status"] = None
    cat["production_status"] = cat["production_status"].astype(object)
    cat.loc[hit.values, "production_status"] = jas.loc[hit.index, "status"].values
    return cat


def _swap_producers(cat: pd.DataFrame, jas: pd.DataFrame, mj: pd.Series, pref: pd.Series,
                    far_km: float = 10.0) -> pd.Series:
    """Une installation productrice restée seule reprend la fiche tenue par une installation SANS production
    quand elle reconnaît cette fiche par son nom ou un alias (à < `far_km`).

    match() classe d'abord par rang du nom : une raffinerie sans production (« Coral Bay Nickel », nom
    principal de la fiche « Coral Bay HPAL ») passait devant la mine productrice dont la fiche porte le nom
    en alias (« Rio Tuba », 5,45 Mt/an, à 600 m) ; la fiche affichait « production réelle » sans données et
    la calibration perdait Rio Tuba. Seules les installations NON rattachées en profitent : une
    productrice déjà liée ne déplace rien (pas de chaîne d'échanges entre sites voisins).
    `mj` : point Jasansky → étiquette de la fiche ; rend `mj` corrigé, chaque échange journalisé.
    """
    held = [(lab, row) for lab, row in mj.items() if not pref.get(lab, False)]
    linked = set(jas.loc[mj.index, "jasansky"])
    free = jas[pref.reindex(jas.index, fill_value=False).to_numpy(bool) & ~jas["jasansky"].isin(linked)]
    if not held or free.empty:
        return mj
    base = cat.loc[[row for _, row in held]]
    pairs = sorted((p for p in _pairs(base, free, -1.0, far_km, ("name", "aliases")) if p[0] >= 1),
                   key=lambda p: (-p[0], p[1]))
    mj = mj.copy()
    used_f, used_b = set(), set()
    for _, dist, i, j in pairs:
        fid = free["jasansky"].iloc[i]
        if fid in used_f or j in used_b:
            continue
        used_f.add(fid)
        used_b.add(j)
        old_lab, row = held[j]
        log.warning("Jasansky : %s « %s » (production) remplace %s « %s » (sans production) sur %s à %.1f km",
                    fid, free["name"].iloc[i], jas.at[old_lab, "jasansky"], jas.at[old_lab, "name"],
                    cat.at[row, "id"] if "id" in cat else row, dist)
        mj = mj.drop(index=old_lab)
        mj.loc[free.index[i]] = row
    return mj


def _link_far(cat: pd.DataFrame, jas: pd.DataFrame, far_km: float = 100.0) -> pd.Series:
    """3e passe : installation PRODUCTRICE restée seule ↔ site du même nom exact jusqu'à `far_km`.

    Les coordonnées Jasansky sont parfois celles du bureau, du port ou de l'usine : Cerrejon est à 12 km
    de la fiche ICMM « Cerrejon », Ok Tedi à 81 km. À < 10 km, les deux premières passes les manquent et
    la vérité terrain (production) n'atteint jamais la calibration. Garde-fous : mêmes mots distinctifs
    (exact, au moins un mot distinctif : jamais « Gold Creek » à 80 km), un seul site candidat pour
    l'installation (ou une seule fiche ICMM parmi eux) et une seule installation pour le site. Chaque
    lien est journalisé pour relecture.
    `cat` : sites encore sans installation ; `jas` : points des installations productrices sans site.
    Rend, pour le meilleur point de chaque installation liée, l'étiquette du site (index de `cat`).
    """
    out = pd.Series(-1, index=jas.index[:0], dtype=object)
    if cat.empty or jas.empty:
        return out
    tree = cKDTree(_xyz(cat["lat"], cat["lon"]))
    cnames = cat["name"].to_numpy(object)
    jxyz = _xyz(jas["lat"], jas["lon"])
    cand: dict[str, dict[int, tuple[float, object]]] = {}  # installation → {position du site: (km, point)}
    for k, (lab, fid, nm, al) in enumerate(zip(jas.index, jas["jasansky"], jas["name"],
                                              jas.get("aliases", pd.Series(None, index=jas.index)),
                                              strict=True)):
        if not np.isfinite(jxyz[k]).all():
            continue
        names = [t for t in _name_set(nm) + _name_set(al) if _core(t)]
        if not names:
            continue
        idxs = tree.query_ball_point(jxyz[k], _chord(far_km))
        for j in idxs:
            cn = [t for t in _name_set(cnames[j]) if _core(t)]
            if not any(_sim_tokens(a, b, exact=True) for a in names for b in cn):
                continue
            dist = float(np.linalg.norm(tree.data[j] - jxyz[k]) / _chord(1.0))
            cur = cand.setdefault(fid, {})
            if j not in cur or dist < cur[j][0]:
                cur[j] = (dist, lab)
    # plusieurs candidats : la fiche ICMM vérifiée seule l'emporte (« Morro Velho Mine » ICMM à 15 km,
    # « Morro Velho » Wikidata à 39 km), comme dans les deux premières passes ; sinon on s'abstient
    icmm = cat["icmm"].notna().to_numpy() if "icmm" in cat else np.zeros(len(cat), bool)
    for fid, c in cand.items():
        if len(c) > 1 and sum(icmm[j] for j in c) == 1:
            cand[fid] = {j: v for j, v in c.items() if icmm[j]}
    claims: dict[int, int] = {}
    for c in cand.values():
        for j in c:
            claims[j] = claims.get(j, 0) + 1
    links = {}
    for fid, c in cand.items():
        if len(c) != 1:
            continue
        (j, (dist, lab)), = c.items()
        if claims[j] != 1:
            continue
        links[lab] = cat.index[j]
        log.warning("Jasansky (3e passe, même nom) : %s « %s » → %s « %s » à %.1f km", fid,
                    jas.at[lab, "name"], cat["id"].iloc[j] if "id" in cat else cat.index[j], cnames[j], dist)
    return pd.Series(links, dtype=object) if links else out


def _modern_country(cat: pd.DataFrame, max_km: float = 50.0) -> pd.DataFrame:
    """États disparus (« Austria–Hungary », « Soviet Union »…) → pays actuel du site voisin le plus proche.

    Choix documenté : on ne renomme jamais par le nom seul, l'URSS ou l'Autriche-Hongrie couvrent plusieurs
    pays actuels (les mines « Austria–Hungary » de Kladno sont tchèques). Le voisin vient du même catalogue
    (fiches ICMM, Wikidata, contours Maus avec leur pays) : à < `max_km`, il est quasi toujours dans le
    même pays. Sans voisin, le pays reste vide. Le nom historique n'est pas perdu : il va dans « region »
    quand elle est vide.
    """
    hist = cat["country"].map(entities.is_historical).astype(bool)
    if not hist.any():
        return cat
    cat = cat.copy()
    ref = cat[cat["country"].notna() & ~hist & cat["lat"].notna() & cat["lon"].notna()]
    old = cat.loc[hist, "country"]
    new = pd.Series(None, index=old.index, dtype=object)
    if len(ref):
        tree = cKDTree(_xyz(ref["lat"], ref["lon"]))
        q = _xyz(cat.loc[hist, "lat"], cat.loc[hist, "lon"])
        fin = np.isfinite(q).all(axis=1)
        d = np.full(len(q), np.inf)
        k = np.zeros(len(q), int)
        if fin.any():
            d[fin], k[fin] = tree.query(q[fin], distance_upper_bound=_chord(max_km))
        ok = np.isfinite(d)
        new[ok] = ref["country"].to_numpy(object)[k[ok]]
    if "region" not in cat:
        cat["region"] = None
    empty = cat.loc[hist, "region"].isna() | (cat.loc[hist, "region"].astype(str).str.strip() == "")
    cat.loc[empty[empty].index, "region"] = old[empty]
    cat.loc[hist, "country"] = new
    log.warning("pays : %d sites d'un État disparu rattachés au pays actuel du voisin (%d sans voisin)",
                int(hist.sum()), int(new.isna().sum()))
    return cat


def build(download: bool = True) -> pd.DataFrame:
    if download:
        fetch()
    icmm = load_icmm().reset_index(drop=True)
    icmm["id"] = icmm["icmm"].str.lower()
    icmm["sources"] = "icmm"

    wd_path = SRC / SOURCES["wikidata"]["file"]
    if wd_path.exists():
        wd = pd.read_parquet(wd_path).reset_index(drop=True)
    else:  # Wikidata indisponible : on construit quand même avec les autres sources
        log.warning("Wikidata absent : catalogue construit sans ses mines")
        wd = pd.DataFrame(columns=WD_COLS).astype({"lat": float, "lon": float})
    cat, folded = merge_wikidata(icmm, wd)

    wpl_path = SRC / SOURCES["wplists"]["file"]
    if wpl_path.exists():
        cat = merge_wplists(cat, pd.read_parquet(wpl_path), folded)
    dup = cat["id"].duplicated()
    if dup.any():  # ne devrait plus arriver : on le dit au lieu de supprimer en silence
        log.warning("%d identifiants en double retirés : %s", int(dup.sum()),
                    ", ".join(cat.loc[dup, "id"][:10]))
        cat = cat[~dup]
    cat = cat.reset_index(drop=True)

    prod_ids = set(all_production()["facility_id"])
    cat = link_jasansky(cat, load_jasansky().reset_index(drop=True), prod_ids)

    poly = gpd.read_file(SRC / SOURCES["maus"]["file"])
    prio = pd.Series(np.where(cat["icmm"].notna(), 0, np.where(cat["name"].notna(), 1, 2)), index=cat.index)
    fp, unnamed = footprints(cat, poly, priority=prio)
    cat = cat.join(fp)
    cat = pd.concat([cat, unnamed], ignore_index=True).reset_index(drop=True)

    cat["name"] = cat["name"].where(cat["name"].notna(), None)
    # mojibake venu aussi de Wikidata / Wikipedia (« Coalâ€‘Mac ») : réparé partout, pas seulement ICMM
    for col in ("name", "aliases", "region", "owner", "operator"):
        if col in cat:
            cat[col] = cat[col].map(fix_text)
    # un seul vocabulaire de pays (ICMM, Wikidata et Maus écrivent « Russia » / « Russian Federation »…)
    cat["country"] = cat["country"].map(entities.norm_country)
    cat = _modern_country(cat)
    known = set(cat["country"].dropna())
    wp = cat["country"].isna() & cat["region"].notna() & cat["id"].astype(str).str.startswith("wp_")
    as_country = cat.loc[wp, "region"].map(entities.norm_country)
    cat.loc[wp & as_country.reindex(cat.index).isin(known), "country"] = as_country
    odd = sorted(set(cat["country"].dropna()) - entities.COUNTRIES)
    if odd:  # un nom hors du vocabulaire actuel scinde un pays dans les filtres et le décompte « pays »
        log.warning("pays hors vocabulaire (à ajouter dans entities.FRENCH_NAMES) : %s", ", ".join(odd))
    # production réelle = id Jasansky ET lignes de production : 255 raffineries / fonderies liées n'en ont
    # aucune, l'API les comptait « avec production réelle » et la fiche montrait un graphique vide
    cat["has_production"] = cat["jasansky"].isin(prod_ids).astype(bool)
    keep = ["id", "icmm", "name", "aliases", "lat", "lon", "country", "region", "asset_type", "commodities",
            "confidence", "owner", "operator", "wikidata", "wikipedia", "jasansky", "has_production",
            "production_status",
            "footprint_km2", "footprint_polygons", "minx", "miny", "maxx", "maxy", "sources", "wd_type"]
    cat = cat[[c for c in keep if c in cat]]
    OUT.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(OUT / "mines.parquet", index=False)
    load.cache_clear()
    return cat


@cache
def _maus() -> gpd.GeoDataFrame:
    g = gpd.read_file(SRC / SOURCES["maus"]["file"])
    _ = g.sindex  # index spatial construit une fois
    return g


def mine_polygons(lat: float, lon: float, radius_m: float, attach_km: float = 2.0):
    """Contours Maus à moins de `attach_km` du carré d'analyse : le masque « mine » d'un site."""
    g = _maus()
    d = attach_km / 111.0 + radius_m / 111_000 / max(np.cos(np.radians(lat)), 0.2)
    idx = list(g.sindex.intersection((lon - d, lat - d, lon + d, lat + d)))
    return g.iloc[idx]


@cache
def load() -> pd.DataFrame:
    p = OUT / "mines.parquet"
    if not p.exists():
        return pd.DataFrame()
    cat = pd.read_parquet(p).set_index("id", drop=False)
    if "has_production" not in cat:
        cat["has_production"] = _has_production(cat)
    return cat


def _has_production(cat: pd.DataFrame) -> pd.Series:
    """Catalogue construit avant la colonne « has_production » : même règle que build(), recalculée."""
    if "jasansky" not in cat:
        return pd.Series(False, index=cat.index)
    try:
        ids = set(all_production()["facility_id"])
    except (OSError, KeyError, zipfile.BadZipFile) as e:  # source Jasansky absente : pas de production
        log.warning("production Jasansky illisible (%s) : aucun site marqué avec production", e)
        ids = set()
    return (cat["jasansky"].notna() & cat["jasansky"].isin(ids)).astype(bool)

