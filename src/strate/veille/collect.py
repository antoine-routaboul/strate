"""Étape 1 de la veille : le collecteur. Récupère l'actualité minière mondiale, sans aucune IA.

Deux sources ouvertes, toutes deux GDELT (métadonnées seulement : jamais la page de l'article) :

- le flux GKG toutes les 15 min, en anglais et traduit (~65 langues). Chaque quart d'heure est une ligne
  de `feed_files` : on sait toujours ce qui manque. Les fichiers récents répondent souvent 404 pendant des
  heures : ils passent `missing`, sont réessayés avec une attente croissante, abandonnés après 72 h ;
- l'API documentaire, pour l'historique des mines suivies (`tracked`) : volume mensuel d'articles depuis
  2017 (`news_volume`) et derniers articles (`raw_docs` avec `hint_site`). Elle bloque au-delà d'une
  requête toutes les 5 s : un seul métronome (≥ 6 s) pour tous les appels, même entre processus.

Le tri « minier » est volontairement large (thème GDELT minier OU mot « mine » dans le titre, en
~25 langues) : le normaliseur ne garde ensuite que les articles qui citent une mine du catalogue.
Le client HTTP est injectable (tests sans réseau : `httpx.MockTransport`).
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import pandas as pd

from strate import catalog, entities
from strate.core import DATA_DIR
from strate.veille import store

log = logging.getLogger(__name__)

UA = {"User-Agent": "strate/0.1 (open-source research tool)"}
STAGE = "collecte"

# -- flux GKG -----------------------------------------------------------------------------------------
GKG_BASE = "http://data.gdeltproject.org/gdeltv2/"  # http : le serveur redirige vers https
LASTUPDATE = {"gkg": GKG_BASE + "lastupdate.txt", "gkg_tr": GKG_BASE + "lastupdate-translation.txt"}
SUFFIX = {"gkg": ".gkg.csv.zip", "gkg_tr": ".translation.gkg.csv.zip"}
SOURCE = {"gkg": "gdelt_gkg", "gkg_tr": "gdelt_gkg_tr"}
STEP = timedelta(minutes=15)
GIVE_UP = timedelta(hours=72)          # un fichier encore absent 72 h après son heure ne viendra plus
MAX_CATCHUP = timedelta(days=7)        # Mac éteint un mois : on ne retélécharge pas 6 000 fichiers
MAX_TRIES = 6                          # fichiers illisibles (zip tronqué, HTTP 4xx) avant abandon ; jamais
                                       # le réseau : une coupure ou un serveur saturé ne consomme aucun essai
MAX_NET_FAILS = 3                      # erreurs réseau d'affilée dans un cycle : on arrête le lot
FEED_RETRY_MAX = timedelta(hours=1)    # Retry-After du serveur de fichiers : attente maximale respectée
RETRY_MAX = timedelta(hours=3)         # attente maximale entre deux essais d'un fichier absent
N_COLS = 27

# Thèmes GDELT miniers (liste du cahier des charges, ouverte : « … »)
MINING_THEMES = frozenset({
    "ENV_MINING", "WB_1699_METAL_ORE_MINING", "WB_1698_COAL_MINING", "WB_895_MINING_SYSTEMS",
    "WB_1700_NONMETALLIC_MINERAL_MINING_AND_QUARRYING", "TAX_FNCACT_MINER", "TAX_FNCACT_MINERS",
    "ENV_COAL", "ENV_METALS", "SLFID_MINERAL_RESOURCES",
})

# Mot « mine » dans le titre, en langue d'origine. Comparé après _plain (minuscules, sans accents) :
# « minería » = « mineria ». Trois familles : mot entier, début de mot (déclinaisons russes, « minier »
# → « minière », « miniera »), et sous-chaîne pour les écritures sans espaces (chinois, japonais, thaï…).
_WORDS = (
    "mine", "mines", "mining", "miner", "miners", "colliery", "collieries", "quarry", "quarries", "mina",
    "minas", "minera", "mineras", "minero", "mineros", "mineiro", "mineira", "mineradora", "mineradoras",
    "minerit", "minahan", "bergwerk", "grube", "gruben", "zeche", "mgodi", "migodi", "garimpo",
    "yacimiento", "yacimientos", "jazida", "bánya", "mijnbouw", "mỏ than", "mỏ vàng", "mỏ đồng",
    "mỏ quặng", "khai thác mỏ", "khai khoáng",
)
_STEMS = (
    "minería", "mineração", "minier", "bergbau", "kopaln", "górnicz", "górnictw", "hornict", "bányász",
    "bányáb", "bányáj", "gruva", "gruvan", "gruvor", "gruvbolag", "gruvdrift", "gruvarbet", "gruve",
    "kaivos", "maden", "рудник", "горнодобыва", "горноруд", "шахт", "гірнич", "копальн", "уурхай",
    "ορυχει", "μεταλλει", "rudnik", "खदान", "खनन", "சுரங்க", "მაღარო", "հանք",
)
# écritures sans espaces, ou à préfixes collés (arabe « المنجم », hébreu « במכרה ») : sous-chaîne
_SUBSTR = (
    "矿", "礦", "鉱", "炭坑", "광산", "탄광", "광업", "เหมือง", "منجم", "مناجم", "تعدين", "מכרה", "খনি",
    "tambang", "lombong",
)


def _plain(s: str) -> str:
    """Minuscules sans accents, mais sans détruire les écritures non latines (cyrillique, hangul…)."""
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def _alt(words: Iterable[str]) -> str:
    return "|".join(re.escape(_plain(w)) for w in sorted(set(words), key=len, reverse=True))


_TITLE_RE = re.compile(rf"(?<!\w)(?:{_alt(_WORDS)})(?!\w)|(?<!\w)(?:{_alt(_STEMS)})|(?:{_alt(_SUBSTR)})")

# Codes de langue GDELT (ISO 639-2/3, « srclc ») → ISO 639-1, comme le reste de Strate
LANG3 = {
    "afr": "af", "sqi": "sq", "alb": "sq", "ara": "ar", "hye": "hy", "arm": "hy", "aze": "az", "axe": "az",
    "bel": "be", "ben": "bn", "bos": "bs", "bul": "bg", "cat": "ca", "zho": "zh", "chi": "zh", "hrv": "hr",
    "ces": "cs", "cze": "cs", "dan": "da", "nld": "nl", "dut": "nl", "eng": "en", "est": "et", "fin": "fi",
    "fra": "fr", "fre": "fr", "glg": "gl", "kat": "ka", "geo": "ka", "deu": "de", "ger": "de", "ell": "el",
    "gre": "el", "guj": "gu", "heb": "he", "hin": "hi", "hun": "hu", "isl": "is", "ice": "is", "ind": "id",
    "ita": "it", "jpn": "ja", "kan": "kn", "kaz": "kk", "kor": "ko", "lav": "lv", "lit": "lt", "mkd": "mk",
    "mac": "mk", "msa": "ms", "may": "ms", "mal": "ml", "mar": "mr", "mon": "mn", "nep": "ne", "nor": "no",
    "fas": "fa", "per": "fa", "pol": "pl", "por": "pt", "pan": "pa", "ron": "ro", "rum": "ro", "rus": "ru",
    "srp": "sr", "sin": "si", "slk": "sk", "slo": "sk", "slv": "sl", "som": "so", "spa": "es", "swa": "sw",
    "swe": "sv", "tam": "ta", "tel": "te", "tha": "th", "tur": "tr", "ukr": "uk", "urd": "ur", "vie": "vi",
    "uzb": "uz", "tgl": "tl", "fil": "tl", "amh": "am", "khm": "km", "mya": "my", "bur": "my", "lao": "lo",
}
# l'API documentaire donne la langue en toutes lettres
LANG_NAMES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "portuguese": "pt", "italian": "it",
    "russian": "ru", "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar", "turkish": "tr",
    "indonesian": "id", "polish": "pl", "dutch": "nl", "swedish": "sv", "finnish": "fi", "norwegian": "no",
    "danish": "da", "czech": "cs", "slovak": "sk", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg",
    "greek": "el", "ukrainian": "uk", "hebrew": "he", "hindi": "hi", "vietnamese": "vi", "thai": "th",
    "persian": "fa", "urdu": "ur", "bengali": "bn", "tamil": "ta", "telugu": "te", "malay": "ms",
    "serbian": "sr", "croatian": "hr", "bosnian": "bs", "slovenian": "sl", "lithuanian": "lt",
    "latvian": "lv", "estonian": "et", "albanian": "sq", "macedonian": "mk", "georgian": "ka",
    "armenian": "hy", "azerbaijani": "az", "kazakh": "kk", "mongolian": "mn", "nepali": "ne",
    "swahili": "sw", "afrikaans": "af", "catalan": "ca", "galician": "gl", "icelandic": "is",
    "gujarati": "gu", "punjabi": "pa", "marathi": "mr", "malayalam": "ml", "kannada": "kn", "sinhala": "si",
    "somali": "so", "tagalog": "tl", "filipino": "tl", "belarusian": "be", "uzbek": "uz", "burmese": "my",
    "khmer": "km", "lao": "lo", "amharic": "am",
    # langues plus rares de l'API (liste GDELT « translingual » et au-delà)
    "sinhalese": "si", "tibetan": "bo", "maltese": "mt", "kinyarwanda": "rw", "irish": "ga", "welsh": "cy",
    "basque": "eu", "kurdish": "ku", "pashto": "ps", "tajik": "tg", "turkmen": "tk", "kyrgyz": "ky",
    "kirghiz": "ky", "hausa": "ha", "yoruba": "yo", "igbo": "ig", "zulu": "zu", "xhosa": "xh",
    "malagasy": "mg", "haitian": "ht", "haitian creole": "ht", "luxembourgish": "lb", "javanese": "jv",
    "sundanese": "su", "esperanto": "eo", "latin": "la", "faroese": "fo", "scottish gaelic": "gd",
    "frisian": "fy", "corsican": "co", "samoan": "sm", "maori": "mi", "shona": "sn", "sesotho": "st",
    "chichewa": "ny", "oromo": "om", "tigrinya": "ti", "uyghur": "ug", "yiddish": "yi", "dhivehi": "dv",
    "tatar": "tt", "bashkir": "ba", "chechen": "ce", "farsi": "fa", "norwegian bokmal": "no",
    "bokmal": "no", "nynorsk": "nn", "moldovan": "ro", "flemish": "nl", "castilian": "es",
    "cebuano": "ceb", "hmong": "hmn", "odia": "or", "oriya": "or", "assamese": "as", "sindhi": "sd",
    "kashmiri": "ks", "lingala": "ln", "wolof": "wo", "fula": "ff", "tswana": "tn", "setswana": "tn",
    "kirundi": "rn", "tsonga": "ts", "venda": "ve", "swati": "ss", "ndebele": "nd", "quechua": "qu",
    "guarani": "gn", "aymara": "ay", "greenlandic": "kl", "inuktitut": "iu", "dzongkha": "dz",
}

# -- API documentaire ----------------------------------------------------------------------------------
DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DOC_START = "20170101000000"           # l'historique de l'API commence en 2017
DOC_INTERVAL = 6.0                     # GDELT exige 5 s ; 1 s de marge
RETRY_AFTER_MAX = 120.0                # un Retry-After délirant ne bloque pas la chaîne
MAX_NAMES = 3                          # noms par requête : au-delà la requête devient trop longue

# -- noms multilingues --------------------------------------------------------------------------------
NAMES_PATH = DATA_DIR / "veille" / "names.parquet"
NAMES_EVERY = timedelta(days=7)
NAMES_RETRY = timedelta(hours=6)       # Wikidata en panne : on ne retente pas à chaque cycle
NAMES_BATCH = 400
_NAME_LANG_PREF = ("en", "mul", "fr", "es", "pt", "de")


# == outils ===========================================================================================
def make_client() -> httpx.Client:
    return httpx.Client(headers=UA, timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(UTC)).astimezone(UTC)


def stamp_of(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def parse_stamp(s: str) -> datetime | None:
    """AAAAMMJJHHMMSS (ou avec séparateurs « 20170101T000000Z ») → datetime UTC ; None si illisible."""
    d = re.sub(r"\D", "", s or "")
    if len(d) < 14:
        return None
    try:
        return datetime.strptime(d[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def floor15(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC)
    return dt.replace(minute=dt.minute - dt.minute % 15, second=0, microsecond=0)


_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ocid$|cmpid$)")


def norm_url(url: str) -> str:
    """Forme canonique d'une URL : même article = même empreinte, d'où qu'il vienne (GKG ou API doc)."""
    u = (url or "").strip()
    try:
        p = urlsplit(u)
    except ValueError:
        return u
    host = (p.hostname or "").lower()
    if p.port and not ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)):
        host = f"{host}:{p.port}"
    query = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                       if not _TRACKING.match(k.lower())])
    path = p.path.rstrip("/") if p.path not in ("", "/") else ""
    return urlunsplit((p.scheme.lower(), host, path, query, ""))


def doc_id(url: str) -> str:
    return hashlib.sha1(norm_url(url).encode("utf-8")).hexdigest()


def _domain(url: str) -> str | None:
    try:
        h = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    return h.removeprefix("www.") or None


def _meta_get(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


# == tri minier ========================================================================================
def title_hit(title: str | None) -> str | None:
    """Mot « mine » trouvé dans le titre (forme repliée), sinon None."""
    if not title:
        return None
    m = _TITLE_RE.search(_plain(title))
    return m.group(0) if m else None


def mining_reason(themes: Iterable[str], title: str | None) -> str | None:
    """Pourquoi l'article est gardé (« thème:ENV_MINING », « titre:mina ») ; None = pas minier."""
    hit = sorted(set(themes) & MINING_THEMES)
    if hit:
        return "thème:" + hit[0]
    w = title_hit(title)
    return "titre:" + w if w else None


# == lecture d'un fichier GKG ============================================================================
_TITLE_TAG = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_PUBTS_TAG = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")


def _names(field: str, limit: int = 50) -> list[str]:
    """« Nom,offset;Nom,offset » (V2) ou « nom;nom » (V1) → noms uniques dans l'ordre."""
    out: dict[str, None] = {}
    for part in field.split(";"):
        name = part.rsplit(",", 1)[0] if re.search(r",\d+$", part) else part
        name = name.strip()
        if name:
            out.setdefault(name, None)
            if len(out) >= limit:
                break
    return list(out)


def _float(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_locations(v2: str, v1: str = "", limit: int = 30) -> list[dict]:
    """Lieux GDELT → [{name, country, adm1, lat, lon, type, cc}].

    `cc` est le code FIPS 10-4 de GDELT, PAS l'ISO (RS = Russie, GM = Allemagne) : le pays lisible est
    donc pris dans le nom complet (« Berlin, Berlin, Germany » → Germany) et passé par norm_country.
    """
    out, seen = [], set()
    for part in (v2 or v1 or "").split(";"):
        f = part.split("#")
        if len(f) >= 9:                       # V2 : type#nom#pays#adm1#adm2#lat#lon#featureid#offset
            typ, name, cc, adm1, lat, lon = f[0], f[1], f[2], f[3], f[5], f[6]
        elif len(f) == 7:                     # V1 : type#nom#pays#adm1#lat#lon#featureid
            typ, name, cc, adm1, lat, lon = f[0], f[1], f[2], f[3], f[4], f[5]
        else:
            continue
        la, lo = _float(lat), _float(lon)
        key = (name, la, lo)
        if not name or key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "country": entities.norm_country(name.rsplit(",", 1)[-1]),
                    "adm1": adm1 or None, "lat": la, "lon": lo,
                    "type": int(typ) if typ.isdigit() else None, "cc": cc or None})
        if len(out) >= limit:
            break
    return out


def parse_record(cols: list[str], feed: str, fetched_at: str) -> dict | None:
    """Une ligne GKG 2.1 découpée → document `raw_docs`, ou None s'il n'est pas minier.

    Colonnes réelles (vérifiées sur un fichier de novembre 2026) : 7/8 thèmes V1/V2, 9/10 lieux V1/V2,
    11/12 personnes V1/V2, 13/14 organisations V1/V2, 15 ton, 25 traduction, 26 extras (titre). Les
    colonnes V2 gardent la casse d'origine (« Freeport-McMoRan ») et les V1 sont en minuscules.
    """
    if len(cols) < N_COLS:
        return None
    themes = _names(cols[8], limit=10_000) or _names(cols[7], limit=10_000)
    m = _TITLE_TAG.search(cols[26])
    title = " ".join(html.unescape(m.group(1)).split())[:500] if m else None
    reason = mining_reason(themes, title)
    url = cols[4].strip()
    if reason is None or not url:  # sans URL, pas d'empreinte : toutes ces lignes auraient le même id
        return None
    crawl = parse_stamp(cols[1])
    pm = _PUBTS_TAG.search(cols[26])
    pub = parse_stamp(pm.group(1)) if pm else None
    when = pub or crawl
    srclc, mt = None, None
    for kv in cols[25].split(";"):
        k, _, v = kv.partition(":")
        if k == "srclc":
            srclc = v.strip().lower()
        elif k == "eng":
            mt = v.strip()
    lang = "en" if feed == "gkg" else (LANG3.get(srclc, srclc) if srclc else None)
    tone = _float(cols[15].split(",", 1)[0])
    payload = {"gkg": cols[0], "crawl": _iso(crawl) if crawl else None, "kept": reason}
    if srclc:
        payload |= {"srclc": srclc, "mt": mt}
    return {
        "id": doc_id(url), "source": SOURCE[feed], "fetched_at": fetched_at,
        "published_at": _iso(when) if when else None, "url": url,
        "domain": (cols[3].strip().lower() or _domain(url)), "lang": lang, "title": title,
        "themes": ";".join(themes) or None,
        "locations": json.dumps(parse_locations(cols[10], cols[9]), ensure_ascii=False),
        "organizations": ";".join(_names(cols[14]) or _names(cols[13])) or None,
        "persons": ";".join(_names(cols[12]) or _names(cols[11])) or None,
        "tone": tone, "hint_site": None, "payload": json.dumps(payload, ensure_ascii=False),
    }


def iter_gkg(lines: Iterable[str], feed: str, fetched_at: str, stats: dict) -> Iterator[dict]:
    """Lit un flux de lignes GKG au fil de l'eau (jamais tout le fichier en mémoire)."""
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            continue
        stats["read"] = stats.get("read", 0) + 1
        cols = line.split("\t")
        if len(cols) < N_COLS:
            stats["bad"] = stats.get("bad", 0) + 1
            continue
        try:
            doc = parse_record(cols, feed, fetched_at)
        except Exception as e:  # une ligne corrompue ne doit pas faire perdre le fichier entier
            stats["bad"] = stats.get("bad", 0) + 1
            log.debug("ligne GKG illisible : %s", e)
            continue
        if doc is not None:
            stats["kept"] = stats.get("kept", 0) + 1
            yield doc


def iter_zip(data: bytes, feed: str, fetched_at: str, stats: dict) -> Iterator[dict]:
    """Zip GDELT en mémoire (compressé, ~5-15 Mo) → documents, décompressés ligne à ligne."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")] or z.namelist()
        if not names:
            raise zipfile.BadZipFile("zip vide (aucun fichier)")
        with z.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="\n")
            yield from iter_gkg(text, feed, fetched_at, stats)


_COLS = ("id", "source", "fetched_at", "published_at", "url", "domain", "lang", "title", "themes",
         "locations", "organizations", "persons", "tone", "hint_site", "payload")
_INSERT = (f"INSERT INTO raw_docs({', '.join(_COLS)}, status) VALUES ({', '.join('?' * len(_COLS))}, 'new') "
           "ON CONFLICT(id) DO NOTHING")


def insert_docs(con: sqlite3.Connection, docs: Iterable[dict], chunk: int = 500) -> int:
    """Insère par paquets ; un article déjà connu (même URL canonique) n'est pas dupliqué."""
    n, buf = 0, []
    for d in docs:
        buf.append(tuple(d.get(c) for c in _COLS))
        if len(buf) >= chunk:
            n += con.executemany(_INSERT, buf).rowcount
            buf = []
    if buf:
        n += con.executemany(_INSERT, buf).rowcount
    return n


# == planification des quarts d'heure ===================================================================
def latest_stamp(c: httpx.Client, feed: str) -> str | None:
    """Dernier quart d'heure publié d'après lastupdate*.txt (le flux traduit a souvent 15 min de retard)."""
    r = c.get(LASTUPDATE[feed])
    r.raise_for_status()
    for line in r.text.splitlines():
        url = line.split()[-1] if line.split() else ""
        if url.endswith(SUFFIX[feed]):
            name = url.rsplit("/", 1)[-1]
            if parse_stamp(name[:14]):
                return name[:14]
    return None


def file_name(feed: str, stamp: str) -> str:
    return stamp + SUFFIX[feed]


def plan(con: sqlite3.Connection, feed: str, latest: datetime, backfill_hours: float) -> int:
    """Ajoute en `pending` chaque quart d'heure pas encore connu jusqu'à `latest` inclus.

    1er lancement : `backfill_hours` d'historique. Ensuite : tout ce qui suit le dernier quart d'heure
    connu (les trous sont déjà des lignes `pending` ou `missing`), au plus MAX_CATCHUP en arrière.
    """
    latest = floor15(latest)
    last = con.execute("SELECT MAX(stamp) FROM feed_files WHERE feed=?", (feed,)).fetchone()[0]
    last_dt = parse_stamp(last) if last else None
    if last_dt:
        start = last_dt + STEP
    else:  # backfill_hours=0 : au moins le dernier quart d'heure, sinon rien ne serait jamais planifié
        start = min(latest - timedelta(hours=max(backfill_hours, 0)) + STEP, latest)
    start = max(start, latest - MAX_CATCHUP)
    rows, t = [], start
    while t <= latest:
        s = stamp_of(t)
        rows.append((file_name(feed, s), feed, s))
        t += STEP
    return con.executemany(
        "INSERT OR IGNORE INTO feed_files(name, feed, stamp, status) VALUES (?, ?, ?, 'pending')", rows
    ).rowcount if rows else 0


def _retry_delay(tries: int) -> timedelta:
    return min(STEP * (2 ** max(tries - 1, 0)), RETRY_MAX)


def give_up_missing(con: sqlite3.Connection, now: datetime) -> int:
    cutoff = stamp_of(now - GIVE_UP)
    return con.execute(
        "UPDATE feed_files SET status='error', message='absent plus de 72 h après sa date : abandonné' "
        "WHERE status='missing' AND stamp < ?", (cutoff,)).rowcount


def due(con: sqlite3.Connection, now: datetime, max_files: int) -> list[sqlite3.Row]:
    """Fichiers à tenter : les absents dont l'attente est écoulée (un quart du lot leur est réservé, sinon
    un gros rattrapage les laisserait expirer), puis les nouveaux, du plus récent au plus ancien."""
    if max_files <= 0:
        return []
    missing = [r for r in con.execute(
        "SELECT name, feed, stamp, status, tries, last_try FROM feed_files WHERE status='missing' "
        "ORDER BY stamp")
        if not r["last_try"] or (parse_stamp(r["last_try"]) or now) + _retry_delay(r["tries"]) <= now]
    reserve = missing[:max(1, max_files // 4)]
    pending = con.execute(
        "SELECT name, feed, stamp, status, tries, last_try FROM feed_files WHERE status='pending' "
        "ORDER BY stamp DESC LIMIT ?", (max_files - len(reserve),)).fetchall()
    out = reserve + pending
    if len(out) < max_files:  # peu de nouveaux : le reste du lot va aux autres absents
        out += missing[len(reserve):len(reserve) + max_files - len(out)]
    return out


def _retry_after_s(r: httpx.Response) -> float | None:
    """En-tête Retry-After (secondes ou date HTTP) → secondes ; None s'il est absent ou illisible."""
    v = r.headers.get("retry-after")
    if not v:
        return None
    try:
        sec = float(v)
    except ValueError:
        try:
            sec = (parsedate_to_datetime(v) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, IndexError):
            return None
    return sec if math.isfinite(sec) and sec > 0 else None


def fetch_file(con: sqlite3.Connection, c: httpx.Client, row, now: datetime) -> dict:
    """Télécharge, lit et range un fichier ; met à jour sa ligne `feed_files`. Ne lève jamais (sauf panne
    de la base elle-même).

    Trois sortes d'échec, traitées différemment :
    - 404 : `missing`, réessayé avec une attente croissante ; abandonné (`error`) 72 h après sa date ;
    - réseau (coupure, délai dépassé, 429, 5xx) : la faute n'est pas au fichier. Aucun essai consommé,
      statut inchangé, réessayé au cycle suivant ; `res["net"]` dit à l'appelant d'arrêter le lot ;
    - contenu illisible (zip tronqué ou vide, flux deflate abîmé, autre HTTP 4xx, toute exception de
      lecture) : un essai consommé, abandon (`error`) après MAX_TRIES. Isolé à ce fichier.
    """
    feed, stamp, name = row["feed"], row["stamp"], row["name"]
    prev = row["tries"] or 0
    t = _iso(now)
    res = {"status": row["status"], "read": 0, "kept": 0, "new": 0}

    def mark(status: str, msg: str, tries: int, n_docs=None, n_kept=None):
        res["status"] = status
        con.execute("UPDATE feed_files SET status=?, tries=?, last_try=?, n_docs=?, n_kept=?, message=? "
                    "WHERE name=?", (status, tries, t, n_docs, n_kept, msg[:300], name))

    def net(msg: str, **extra):
        mark(row["status"], f"réseau : {msg} (réessayé au cycle suivant, aucun essai compté)", prev)
        res.update(net=True, error=True, **extra)
        return res

    def unreadable(msg: str):
        tries = prev + 1
        mark("error" if tries >= MAX_TRIES else row["status"],
             msg + (" : abandonné" if tries >= MAX_TRIES else ""), tries)
        res.update(error=True, unreadable=True, reason=msg)
        return res

    try:
        r = c.get(GKG_BASE + name)
    except httpx.TransportError as e:  # hors ligne, DNS, délai dépassé, connexion coupée en cours de lecture
        return net(f"{type(e).__name__} : {e}",
                   connect=isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)))
    except httpx.HTTPError as e:       # redirections en boucle, réponse mal encodée
        return unreadable(f"{type(e).__name__} : {e}")
    if r.status_code == 404:
        old = (parse_stamp(stamp) or now) < now - GIVE_UP
        mark("error" if old else "missing",
             "absent plus de 72 h après sa date : abandonné" if old else "404 : pas encore publié", prev + 1)
        return res
    if r.status_code == 429 or r.status_code >= 500:
        return net(f"HTTP {r.status_code}", retry_after=_retry_after_s(r))
    if r.status_code >= 400:
        return unreadable(f"HTTP {r.status_code}")
    if not r.content.startswith(b"PK"):
        # 200 qui n'est pas un zip (page HTML d'un portail captif, d'un proxy) : sans doute la faute du
        # réseau, pas du fichier. Aucun essai compté, compté comme erreur réseau (le lot s'arrête après
        # MAX_NET_FAILS d'affilée) ; seulement 72 h après sa date, le fichier lui-même est mis en cause.
        if (parse_stamp(stamp) or now) >= now - GIVE_UP:
            return net("réponse 200 qui n'est pas un zip (portail captif ou proxy ?)", notzip=True)
        return unreadable("réponse qui n'est pas un zip")
    stats: dict = {}
    try:
        # documents miniers d'abord tous lus (quelques centaines au plus) puis insérés : un fichier abîmé
        # au milieu ne laisse aucune insertion partielle
        docs = list(iter_zip(r.content, feed, t, stats))
    except Exception as e:  # zip tronqué ou vide, zlib.error, EOFError… : la faute est au fichier seul
        return unreadable(f"fichier illisible : {type(e).__name__} : {e}")
    res["new"] = insert_docs(con, docs)
    res["read"], res["kept"] = stats.get("read", 0), stats.get("kept", 0)
    msg = f"{res['new']} nouveaux" + (f", {stats['bad']} lignes illisibles" if stats.get("bad") else "")
    mark("done", msg, prev + 1, res["read"], res["kept"])
    return res


def collect_feeds(con: sqlite3.Connection, c: httpx.Client, now: datetime, backfill_hours: float,
                  max_files: int) -> dict:
    out = {"files_done": 0, "files_missing": 0, "files_error": 0, "files_given_up": 0, "files_net": 0,
           "files_unreadable": 0, "files_notzip": 0, "docs_read": 0, "docs_kept": 0, "docs_new": 0,
           "errors": []}
    abandoned: list[str] = []
    for feed in LASTUPDATE:
        try:
            s = latest_stamp(c, feed)
        except httpx.HTTPError as e:
            out["errors"].append(f"{feed} : lastupdate illisible ({type(e).__name__})")
            s = None
        # sans lastupdate, on planifie quand même jusqu'au quart d'heure précédent : 404 → missing
        latest = parse_stamp(s) if s else floor15(now) - STEP
        plan(con, feed, min(latest, now), backfill_hours)
    out["files_given_up"] = give_up_missing(con, now)
    con.commit()
    hold = parse_stamp(_meta_get(con, "feed_retry_at") or "")
    if hold and now < hold:  # le serveur a demandé d'attendre (Retry-After) : aucun fichier ce cycle
        out["errors"].append(f"serveur GDELT saturé : reprise après {_iso(hold)}")
        return out
    net_fails = 0
    for row in due(con, now, max_files):
        r = fetch_file(con, c, row, now)
        con.commit()  # un fichier = une transaction : l'API voit l'avancement, un arrêt ne perd rien
        if r.get("net"):
            out["files_net"] += 1
            out["files_notzip"] += bool(r.get("notzip"))
            net_fails += 1
            if r.get("retry_after"):
                until = now + min(timedelta(seconds=r["retry_after"]), FEED_RETRY_MAX)
                _meta_set(con, "feed_retry_at", _iso(until))
                con.commit()
                break
            if r.get("connect") or net_fails >= MAX_NET_FAILS:
                break  # hors ligne : inutile d'attendre un délai dépassé par fichier
            continue
        net_fails = 0
        out["docs_read"] += r["read"]
        out["docs_kept"] += r["kept"]
        out["docs_new"] += r["new"]
        if r["status"] == "done":
            out["files_done"] += 1
        elif r["status"] == "missing":
            out["files_missing"] += 1
        elif r["status"] == "error":
            out["files_error"] += 1
            if r.get("unreadable"):
                abandoned.append(f"{row['name']} ({r.get('reason', 'illisible')})")
        out["files_unreadable"] += bool(r.get("unreadable")) and r["status"] != "error"   # essai compté
    if out["files_net"]:
        why = " dont des réponses qui ne sont pas des zips (portail captif ?)" if out["files_notzip"] else ""
        out["errors"].append(f"réseau : {out['files_net']} fichier(s) non téléchargé(s){why}, lot "
                             "interrompu, repris au cycle suivant")
    if abandoned:  # un fichier perdu pour de bon doit se voir dans pipeline_status, pas seulement en base
        out["errors"].append(f"{len(abandoned)} fichier(s) illisible(s) abandonné(s) après {MAX_TRIES} "
                             "essais : " + ", ".join(abandoned[:3]) + (" …" if len(abandoned) > 3 else ""))
    return out


# == API documentaire : historique par mine ===========================================================
class RateLimited(Exception):
    """GDELT refuse encore après attente : on arrête l'historique pour ce cycle."""


class QueryRefused(Exception):
    """Requête jugée invalide par GDELT (phrase trop courte, mot trop courant…) : erreur propre à la mine."""


class Throttle:
    """Métronome : au moins `interval` secondes entre deux appels, et jamais avant un `defer` (Retry-After).

    Horloge murale (time.time) pour pouvoir partager l'échéance entre processus via la table meta.
    """

    def __init__(self, interval: float = DOC_INTERVAL, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.next_at = 0.0

    def defer(self, seconds: float) -> None:
        self.next_at = max(self.next_at, self.clock() + seconds)

    def defer_until(self, t: float) -> None:
        self.next_at = max(self.next_at, t)

    def wait(self) -> None:
        d = self.next_at - self.clock()
        if d > 0:
            self.sleep(d)
        self.next_at = max(self.next_at, self.clock()) + self.interval


DOC_THROTTLE = Throttle()  # partagé par tous les appels du processus


def _retry_after(r: httpx.Response, attempt: int) -> float:
    sec = _retry_after_s(r) or DOC_INTERVAL * 2 ** (attempt + 1)  # sans en-tête : 12 s, 24 s…
    return min(sec, RETRY_AFTER_MAX)


def doc_get(c: httpx.Client, params: dict, throttle: Throttle, attempts: int = 2) -> dict:
    """Un appel à l'API documentaire, sous le métronome. 429 (ou message « limit requests » en 200) →
    attente (Retry-After) puis nouvel essai ; après `attempts` refus : RateLimited."""
    for k in range(attempts):
        throttle.wait()
        r = c.get(DOC_API, params=params)
        text = r.text
        if r.status_code == 429 or "limit requests" in text[:300].lower():
            throttle.defer(_retry_after(r, k))
            continue
        if r.status_code >= 500:
            r.raise_for_status()
        if r.status_code >= 400:
            raise QueryRefused(f"HTTP {r.status_code}")
        if not text.strip():
            return {}
        try:
            js = r.json()
        except ValueError:
            raise QueryRefused(" ".join(text.split())[:160]) from None
        return js if isinstance(js, dict) else {}  # liste ou valeur nue : traité comme « aucun résultat »
    raise RateLimited("GDELT : trop de requêtes (429)")


# Gentilés et régions : « Chinese », « Australian », « africain » ne désignent pas une mine. Avec les noms
# de pays (entities), ils ne rendent jamais un nom de requête distinctif (« Peru », « Chinese Mine »).
_DEMONYMS = frozenset(catalog._fold(w) for w in """
    chinese russian australian canadian american african chilean peruvian brazilian mexican indonesian
    indian mongolian kazakh kazakhstani congolese zambian ghanaian malian guinean burkinabe tanzanian
    zimbabwean botswanan namibian argentine argentinian bolivian colombian ecuadorian philippine filipino
    papuan swedish finnish norwegian polish german french british spanish portuguese turkish iranian saudi
    ukrainian kyrgyz uzbek mauritanian nigerian ivorian sudanese ethiopian kenyan malagasy mozambican
    angolan japanese korean vietnamese thai burmese pakistani afghan serbian greek irish scottish welsh
    english european asian latin caribbean arctic pacific atlantic national international global world
    chinois chinoise russe australien australienne canadien canadienne americain americaine africain
    africaine chilien chilienne peruvien peruvienne bresilien bresilienne mexicain mexicaine congolais
    congolaise malien malienne guineen guineenne ivoirien ivoirienne senegalais senegalaise marocain
    marocaine francais francaise europeen europeenne
    chino china ruso rusa australiano canadiense chileno chilena peruano peruana brasileno brasilena
    brasileiro brasileira mexicano mexicana argentino argentina boliviano boliviana colombiano colombiana
    ecuatoriano ecuatoriana africano africana americano americana
    africa asia europe america oceania""".split())
# États, provinces et grandes régions minières : un nom d'un mot égal à l'un d'eux (« Nevada », « Victoria »,
# « Katanga ») désigne la région dans les articles bien plus souvent que la mine.
_REGIONS = frozenset("""
    nevada arizona utah montana idaho alaska colorado wyoming minnesota michigan wisconsin kentucky
    virginia pennsylvania illinois indiana ohio alabama texas california oregon washington dakota
    victoria queensland tasmania pilbara goldfields ontario quebec manitoba saskatchewan alberta columbia
    labrador yukon nunavut territories katanga lualaba haut kivu ituri atacama antofagasta tarapaca
    coquimbo sonora chihuahua zacatecas durango sinaloa guerrero oaxaca cajamarca arequipa apurimac cusco
    puno junin ancash minas gerais para bahia goias xinjiang shanxi shaanxi shandong yunnan guizhou
    inner mongolia kuzbass kemerovo yakutia sakha siberia chukotka magadan kamchatka kola ural
    urals limpopo mpumalanga gauteng copperbelt kalimantan sulawesi sumatra papua""".split())
_COUNTRY_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(
    re.escape(k) for k in sorted(entities.COUNTRY_ALIASES, key=len, reverse=True) if k) + r")(?![a-z0-9])")


def _distinctive(name: str) -> frozenset[str]:
    """Mots qui désignent vraiment une mine : ni génériques (relief, substance, « North »…), ni numéros,
    ni pays (« Peru », « Papua New Guinea »), ni gentilés (« Chinese »), ni seulement des régions
    (« Nevada », « Katanga »), ni un pays ou une région suivi d'une étendue (« Atacama Desert », « Chile
    Plateau ») ; vide = nom inutilisable."""
    rest = _COUNTRY_RE.sub(" ", entities.fold(name))
    core = frozenset(t for t in catalog._core(catalog._tokens(rest)) if t not in _DEMONYMS)
    if core <= _REGIONS or (region_like(name) and not _has_mine_word(name)):
        return frozenset()
    return core


# mot d'ouvrage DANS le nom : la phrase exacte « Turkey Creek Mine » ne désigne que la mine, alors que
# « Ural Mountains » ou « Copper Chile » sans lui désignent un lieu ou une tournure de titre
_MINE_TYPE = frozenset("""
mine mines pit pits colliery collieries quarry quarries shaft shafts mina minas tajo kopalnia rudnik
""".split())


def _has_mine_word(name: str) -> bool:
    return bool(_MINE_TYPE.intersection(re.findall(r"[a-z0-9]+", entities.fold(name))))


def _geo_words(name: str) -> tuple[str, set[str], set[str]]:
    """(nom replié, pays marqués par _PAYS, mots utiles) et, parmi ces mots, les pays et régions."""
    f = _COUNTRY_RE.sub(lambda m: " " + _PAYS + m.group(0).replace(" ", "") + " ", entities.fold(name))
    words = {t for t in catalog._tokens(f) if not t.isdigit() and t not in _DEMONYMS and len(t) > 1}
    return f, words, {t for t in words if t.startswith(_PAYS) or t in _REGIONS}


def region_like(name: str) -> bool:
    """Nom fait d'un pays ou d'une région et seulement de mots d'orientation, de relief, d'étendue ou de
    substances anglaises (« Ural Mountains », « Lake Victoria », « Copper Chile », « Cerro Colorado ») :
    un lieu géographique ou une tournure de titre, pas un nom de mine. Les mots d'ouvrage (« Mine »)
    sont ignorés ; partagé avec `normalize`, qui ne cherche ces noms qu'avec un lieu à moins de 50 km."""
    _, words, geo = _geo_words(name)
    return bool(geo) and not words - geo - _REGION_MAKERS - _GEO_FEATURES - _EN_SECTOR


# Mots de lieu (relief, eau, orientation, taille, couleur) : non distinctifs seuls, mais à plusieurs ils
# forment le nom propre d'une grande mine (« Red Lake », « Copper Mountain », « Big Hill »).
_PLACE_WORDS = frozenset("""
mount mt mont monte montana cerro sierra mountain mountains hill hills ridge peak valley creek river lake
lakes gulch canyon spring springs bay island field fields flat flats point rock rocks well wells dam north
south east west northern southern eastern western central norte sur este oeste upper lower new old big
little great grand deep nueva nuevo novo nova black white red blue green""".split())
_PAYS = "zzpays"   # marque d'un nom de pays dans `_place_name_key`
# avec un pays ou une région, ces mots forment un autre nom de région, pas une mine (« New Mexico »,
# « Western Australia », « New South Wales », « San Salvador »)
_REGION_MAKERS = frozenset("""
north south east west northern southern eastern western central upper lower new old great little big
norte sur este oeste nueva nuevo novo nova san santa santo sao saint st""".split())
# avec un pays ou une région, un mot de relief ou d'étendue nomme un lieu géographique, pas une mine
# (« Ural Mountains », « Lake Victoria », « Atacama Desert », « Pilbara Coast »)
_GEO_FEATURES = _PLACE_WORDS | frozenset("""
desert plateau basin range ranges coast coastal highlands highland lowlands plain plains peninsula gulf
sea delta forest belt region province district state territory""".split())
# substances et mots du secteur en anglais : avec un pays ou une région, ils forment une tournure de titre
# de la traduction anglaise que l'API cherche (« Chile copper », « Gold Peru », « Coal Victoria »),
# quel que soit l'ordre. Une substance dans une autre langue (« Cobre Panama ») reste un nom propre.
_EN_SECTOR = frozenset("""
gold copper coal iron ore silver zinc nickel tin lead lithium uranium cobalt platinum palladium diamond
diamonds potash phosphate bauxite manganese graphite tungsten molybdenum vanadium chrome chromite rare
earth earths steel metal metals mineral minerals resources oil gas energy""".split())


def _place_name_key(name: str) -> frozenset[str] | None:
    """Clé d'un nom COMPLET fait seulement de mots génériques, de lieu ou de pays (« Red Lake », « Copper
    Mountain », « Cobre Panama ») : utilisable en phrase exacte (très précise pour GDELT), seulement pour
    le nom du catalogue ou de la liste suivie, jamais pour un alias (« Southern Mine », « Copper Mountain »
    en alias d'une autre mine). Il faut au moins deux mots hors numéros et un mot de lieu, de pays ou de
    région. Avec un pays ou une région, le reste du nom doit contenir un mot qui n'est ni un mot
    d'orientation (« South Africa », « New Mexico »), ni un relief ou une étendue (« Ural Mountains »,
    « Lake Victoria » : des lieux géographiques), ni une substance ou un mot du secteur en anglais
    (« Chile Copper », « Gold Peru » : des tournures de titres), sauf si le nom porte lui-même un mot
    d'ouvrage (« Turkey Creek Mine », « New Jersey Zinc Pit » : la phrase exacte est précise). Sans mot de
    lieu, le pays ou la région ne doit pas non plus ouvrir le nom. « Cobre Panama » passe : « cobre » n'est
    pas de l'anglais. Coût connu : « Cerro Colorado » (Chili) est refusé, « colorado » étant aussi un État
    américain ; la règle ne distingue pas l'adjectif espagnol du nom de région."""
    f, words, geo = _geo_words(name)
    toks = catalog._tokens(f)
    if len(words) < 2 or not (words & _PLACE_WORDS or geo):
        return None
    if region_like(name) and (not _has_mine_word(name) or not words - geo - _REGION_MAKERS):
        return None
    if not words & _PLACE_WORDS:
        first = next((t for t in re.findall(r"[a-z0-9]+", f) if t in toks), None)
        if first in geo:
            return None
    return frozenset({"=" + " ".join(sorted(words))})


def mine_names(site_id: str, site_name: str | None) -> list[str]:
    """Noms utilisables dans une requête : nom du catalogue d'abord (l'API cherche dans les traductions
    anglaises), puis alias et nom suivi. Sans guillemets, assez longs, avec au moins un mot distinctif
    (jamais « Peru », « Gold », « Southern Mine », « Mine 33 ») ; une variante de même cœur distinctif,
    qui n'ajoute que « mine », une substance ou un mot générique (« Grasberg mine » après « Grasberg »,
    « Inden coal Mine » après « Inden »), ne prend pas une des MAX_NAMES places. Exception : le nom du
    catalogue ou de la liste suivie fait de mots de lieu ou de pays (« Red Lake », « Cobre Panama ») reste
    cherché en phrase entière."""
    cands: list[tuple[str, bool]] = []          # (nom, nom principal : catalogue ou liste suivie)
    cat = catalog.load()
    if len(cat) and site_id in cat.index:
        row = cat.loc[site_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cands.append((row.get("name"), True))
        al = row.get("aliases")
        if isinstance(al, str):
            cands += [(a, False) for a in al.split(";")]
    cands.append((site_name, True))
    out, seen = [], set()
    for n, main in cands:
        if not isinstance(n, str):
            continue
        n = " ".join(n.replace('"', " ").split())
        k = _distinctive(n)  # même cœur distinctif (« Inden », « Inden coal Mine ») : une seule place
        if not k and main:
            k = _place_name_key(n)
        if len(n) < 4 or not k or k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out[:MAX_NAMES]


def mine_query(names: list[str]) -> str | None:
    """« "Grasberg" mine » ou « ("Escondida" OR "Minera Escondida") mine » (syntaxe GDELT : OR entre
    parenthèses, phrases entre guillemets)."""
    if not names:
        return None
    phrases = [f'"{n}"' for n in names]
    core = phrases[0] if len(phrases) == 1 else "(" + " OR ".join(phrases) + ")"
    return core + " mine"


def timeline_months(js: dict) -> dict[str, float]:
    """timelinevolraw → {AAAA-MM: articles}. Somme quelle que soit la résolution renvoyée (jour, heure)."""
    out: dict[str, float] = {}
    for series in js.get("timeline") or []:
        if not isinstance(series, dict):
            continue
        for p in series.get("data") or []:
            if not isinstance(p, dict):
                continue
            d = re.sub(r"\D", "", str(p.get("date", "")))
            v = _float(p.get("value"))
            if len(d) >= 6 and v is not None:
                m = f"{d[:4]}-{d[4:6]}"
                out[m] = out.get(m, 0.0) + v
        break  # 1re série : « Article Count »
    return out


def doc_lang(name) -> str | None:
    """Langue de l'API documentaire (« English », « Sinhalese », « Chinese (Simplified) ») → code ISO 639
    (2 lettres, sinon 3) ; None si inconnue, jamais le nom anglais tel quel (le normaliseur exige un code)."""
    if not isinstance(name, str):
        return None
    k = " ".join(re.sub(r"\(.*?\)", " ", name).casefold().split())
    if k in LANG_NAMES:
        return LANG_NAMES[k]
    if re.fullmatch(r"[a-z]{2}", k):
        return k
    if re.fullmatch(r"[a-z]{3}", k):
        return LANG3.get(k, k)
    return None


def artlist_docs(js: dict, site_id: str, query: str, fetched_at: str) -> list[dict]:
    docs = []
    for a in js.get("articles") or []:
        if not isinstance(a, dict):
            continue
        url = str(a.get("url") or "").strip()
        if not url:
            continue
        seen = parse_stamp(a.get("seendate") or "")
        lang = a.get("language")
        docs.append({
            "id": doc_id(url), "source": "gdelt_doc", "fetched_at": fetched_at,
            "published_at": _iso(seen) if seen else None, "url": url,
            "domain": (a.get("domain") or _domain(url) or "").lower() or None,
            "lang": doc_lang(lang),
            "title": " ".join(html.unescape(a.get("title") or "").split())[:500] or None,
            "themes": None, "locations": None, "organizations": None, "persons": None, "tone": None,
            "hint_site": site_id,
            "payload": json.dumps({"query": query, "sourcecountry": a.get("sourcecountry")},
                                  ensure_ascii=False),
        })
    return docs


def _insert_hinted(con: sqlite3.Connection, docs: list[dict]) -> int:
    """Insère les articles d'une mine ; un article déjà reçu par le flux et pas encore normalisé reçoit
    la mine visée (hint_site). Rend le nombre d'articles nouveaux."""
    if not docs:
        return 0
    ids = [d["id"] for d in docs]
    known = {r[0] for r in con.execute(
        f"SELECT id FROM raw_docs WHERE id IN ({','.join('?' * len(ids))})", ids)}
    con.executemany(
        f"INSERT INTO raw_docs({', '.join(_COLS)}, status) VALUES ({', '.join('?' * len(_COLS))}, 'new') "
        "ON CONFLICT(id) DO UPDATE SET hint_site=excluded.hint_site "
        "WHERE raw_docs.hint_site IS NULL AND raw_docs.status='new'",
        [tuple(d.get(c) for c in _COLS) for d in docs])
    return len(set(ids) - known)


def doc_backfill(con: sqlite3.Connection, c: httpx.Client, now: datetime, max_queries: int,
                 throttle: Throttle) -> dict:
    """Historique des mines suivies sans `backfill_status` : 2 requêtes par mine (volume, articles)."""
    out = {"doc_queries": 0, "doc_mines": 0, "doc_articles": 0, "doc_months": 0, "errors": []}
    if max_queries < 2:
        return out
    nxt = _float(_meta_get(con, "doc_api_next_at") or "")
    if nxt:
        throttle.defer_until(nxt)  # un autre processus (CLI « once ») vient peut-être d'appeler l'API
    t, end = _iso(now), stamp_of(now)
    todo = con.execute("SELECT site_id, site_name FROM tracked WHERE backfill_status IS NULL "
                       "ORDER BY since, site_id").fetchall()
    queries = [(row["site_id"], mine_query(mine_names(row["site_id"], row["site_name"]))) for row in todo]
    # mines sans nom utilisable : notées tout de suite, elles ne consomment aucune requête
    con.executemany("UPDATE tracked SET backfill_status=?, backfill_at=? WHERE site_id=?",
                    [("erreur : aucun nom exploitable pour l'API", t, s) for s, q in queries if q is None])
    con.commit()
    for site, q in queries:
        if q is None:
            continue
        if out["doc_queries"] + 2 > max_queries:
            break
        base = {"query": q, "format": "json", "startdatetime": DOC_START, "enddatetime": end}
        try:
            out["doc_queries"] += 1
            tl = doc_get(c, base | {"mode": "timelinevolraw"}, throttle)
            out["doc_queries"] += 1
            art = {"mode": "artlist", "maxrecords": "250", "sort": "datedesc"}
            try:
                al = doc_get(c, base | art, throttle)
            except QueryRefused:
                # liste d'articles refusée sur 2017 → aujourd'hui : fenêtre par défaut de l'API (récente),
                # au plus une requête de plus par mine
                out["doc_queries"] += 1
                al = doc_get(c, {"query": q, "format": "json"} | art, throttle)
        except QueryRefused as e:
            con.execute("UPDATE tracked SET backfill_status=?, backfill_at=? WHERE site_id=?",
                        (f"erreur : requête refusée par GDELT ({e})"[:300], t, site))
            con.commit()
            continue
        except RateLimited as e:
            out["errors"].append(str(e))
            break
        except httpx.HTTPError as e:  # réseau : la mine reste à faire, retentée au cycle suivant
            out["errors"].append(f"API documentaire : {type(e).__name__}")
            break
        finally:
            _meta_set(con, "doc_api_next_at", f"{throttle.next_at:.3f}")
        months = timeline_months(tl)
        con.executemany(
            "INSERT OR REPLACE INTO news_volume(site_id, month, n_articles, source, fetched_at) "
            "VALUES (?, ?, ?, 'gdelt_doc_timeline', ?)", [(site, m, v, t) for m, v in sorted(months.items())])
        docs = artlist_docs(al, site, q, t)
        out["doc_articles"] += _insert_hinted(con, docs)
        # début de couverture de la mine pour le corrélateur : article le plus ancien renvoyé. Gardé à part :
        # certains de ces articles étaient déjà en base sous la source du flux, et raw_docs peut être purgé
        oldest = min((d["published_at"] for d in docs if d["published_at"]), default=None)
        if oldest:
            _meta_set(con, f"doc_oldest:{site}", oldest)
        out["doc_months"] += len(months)
        out["doc_mines"] += 1
        con.execute("UPDATE tracked SET backfill_status='fait', backfill_at=? WHERE site_id=?", (t, site))
        con.commit()
    con.commit()
    return out


# == noms multilingues (Wikidata) ======================================================================
def _label_query(values: str) -> str:
    return f"""SELECT ?x ?l WHERE {{ VALUES ?x {{ {values} }}
      {{ ?x rdfs:label ?l }} UNION {{ ?x skos:altLabel ?l }} }}"""


def fetch_names(con: sqlite3.Connection, c: httpx.Client | None = None, force: bool = False,
                path: Path | None = None, now: datetime | None = None) -> int:
    """Noms de chaque mine du catalogue dans toutes les langues (libellés + alias Wikidata) →
    names.parquet (site_id, name, lang). Au plus une fois par semaine ; rend le nombre de noms écrits
    (0 si la liste est encore fraîche)."""
    path = path or NAMES_PATH
    now = _now(now)
    if not force:
        done = parse_stamp(_meta_get(con, "names_at") or "")
        tried = parse_stamp(_meta_get(con, "names_try_at") or "")
        if path.exists() and done and now - done < NAMES_EVERY:
            return 0
        if tried and now - tried < NAMES_RETRY:
            return 0
    _meta_set(con, "names_try_at", _iso(now))
    con.commit()
    cat = catalog.load()
    if not len(cat) or "wikidata" not in cat:
        return 0
    by_qid: dict[str, list[str]] = {}
    for sid, q in cat["wikidata"].dropna().items():
        for qq in str(q).split(";"):
            if re.fullmatch(r"Q\d+", qq.strip()):
                by_qid.setdefault(qq.strip(), []).append(str(sid))
    rank = {lang: i for i, lang in enumerate(_NAME_LANG_PREF)}
    best: dict[tuple[str, str], str] = {}  # (qid, nom) → langue préférée
    own = c is None
    c = c or make_client()
    try:
        for values in catalog._batches(sorted(by_qid), NAMES_BATCH):
            for b in catalog._sparql(c, _label_query(values)):
                qid = catalog._qid(b["x"]["value"])
                name = " ".join(b["l"]["value"].split())
                lang = b["l"].get("xml:lang") or ""
                if len(name) < 3 or len(name) > 120 or name.isdigit():
                    continue
                k = (qid, name)
                if k not in best or (rank.get(lang, 99), lang) < (rank.get(best[k], 99), best[k]):
                    best[k] = lang
    finally:
        if own:
            c.close()
    rows = [(sid, name, lang) for (qid, name), lang in best.items() for sid in by_qid.get(qid, ())]
    df = pd.DataFrame(rows, columns=["site_id", "name", "lang"]).sort_values(["site_id", "name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # écriture atomique : le normaliseur ne lit jamais un fichier à moitié écrit
    _meta_set(con, "names_at", _iso(now))
    con.commit()
    return len(df)


# == étape complète ====================================================================================
def run_once(con: sqlite3.Connection, backfill_hours: float = 48, max_files: int = 40, doc_queries: int = 6,
             *, client: httpx.Client | None = None, throttle: Throttle | None = None,
             now: datetime | None = None, names: bool = True) -> dict:
    """Un passage borné : flux GKG (au plus `max_files` fichiers), historique API documentaire (au plus
    `doc_queries` requêtes), noms multilingues si la liste a plus d'une semaine. Ne lève pas : chaque
    panne est notée dans pipeline_status et retentée au cycle suivant."""
    now = _now(now)
    throttle = throttle or DOC_THROTTLE
    own = client is None
    c = client or make_client()
    counters: dict = {}
    errors: list[str] = []
    try:
        for label, fn in (
            ("flux", lambda: collect_feeds(con, c, now, backfill_hours, max_files)),
            ("historique", lambda: doc_backfill(con, c, now, doc_queries, throttle)),
            ("noms", lambda: {"names": fetch_names(con, c, now=now)} if names else {}),
        ):
            try:
                r = fn()
            except Exception as e:  # une étape en panne n'empêche pas les suivantes
                log.exception("collecte : %s en échec", label)
                con.rollback()
                errors.append(f"{label} : {type(e).__name__} : {e}"[:200])
                continue
            errors += r.pop("errors", [])
            counters |= r
    finally:
        if own:
            c.close()
    msg = (f"{counters.get('files_done', 0)} fichiers lus, {counters.get('files_missing', 0)} absents, "
           f"{counters.get('docs_new', 0)} articles miniers nouveaux ; historique : "
           f"{counters.get('doc_mines', 0)} mines, {counters.get('doc_articles', 0)} articles")
    if counters.get("files_unreadable"):
        msg += f" ; {counters['files_unreadable']} fichier(s) illisible(s), réessayé(s) plus tard"
    if counters.get("names"):
        msg += f" ; {counters['names']} noms multilingues"
    if errors:
        msg += " — " + " ; ".join(errors)
    store.set_status(con, STAGE, not errors, msg, **counters)
    con.commit()
    return counters
