"""API Strate : sert l'interface autonome ET les widgets d'OpenBB Workspace.

Lancer :  uv run uvicorn strate.api:app --port 7779
OpenBB Workspace → « Add data » → http://127.0.0.1:7779  (lit /widgets.json)
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import math
import queue
import sqlite3
import threading
from datetime import UTC, date, datetime, timedelta
from functools import cache
from importlib.resources import files
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from strate import __version__, catalog, entities, fusion, screener
from strate.core import DATA_DIR
from strate.entities import fold as _fold
from strate.entities import norm_country
from strate.modules import REGISTRY
from strate.veille import store as vstore

app = FastAPI(title="Strate", version=__version__,
              description="Alternative data open source — dossier Mine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://127.0.0.1:7779", "http://localhost:7779"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON propre (dates ISO, NaN → null)."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    rows = out.to_dict(orient="records")
    return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
            for r in rows]


def _clean(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


MONTH = r"^\d{4}-(0[1-9]|1[0-2])$"  # AAAA-MM : un mois invalide donne 422, pas une erreur serveur


def _cat() -> pd.DataFrame:
    """Le catalogue, relu si un autre processus (la CLI) l'a reconstruit depuis notre dernière lecture."""
    screener.sync()
    return catalog.load()


def _graph() -> entities.Graph:
    """Liste suivie + catalogue, relus si sites.toml ou le catalogue ont changé (sinon /api/sites et les
    options de widgets.json servent l'ancienne liste jusqu'à ce qu'une autre route synchronise)."""
    screener.sync()
    return entities.load()


def _site(site: str):
    try:
        return _graph().site(site)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


def _resolve(site: str) -> str:
    """Id d'une jumelle du catalogue → id de la mine suivie correspondante.

    La jumelle est la même mine sous un id ICMM/Wikidata : sans ce renvoi, sa fiche paraît « non suivie,
    non analysée » et /api/analyze relancerait une seconde analyse satellite de la même mine.
    """
    screener.sync()
    return next((w for w, t in screener.twins().items() if t == site), site)


def _has_production(row: dict) -> bool:
    """La ligne a-t-elle une production réelle publiée ? Un id Jasansky seul ne suffit pas : ~250
    installations liées (fonderies, raffineries, réserves seules) n'ont aucune ligne de production.

    Colonne has_production écrite par catalog.build ; recalculée pour un mines.parquet plus ancien."""
    jid = row.get("jasansky")
    if not isinstance(jid, str):
        return False
    hp = row.get("has_production")
    if hp is not None and not pd.isna(hp):
        return bool(hp)
    return jid in _production_ids()


def _production_ids() -> frozenset[str]:
    try:
        return _production_ids_cached()
    except OSError:  # données Jasansky absentes : rien ne peut être montré, donc rien n'est annoncé
        return frozenset()


@cache
def _production_ids_cached() -> frozenset[str]:
    return frozenset(catalog.all_production()["facility_id"].dropna())


def _production_mask(cat: pd.DataFrame) -> pd.Series:
    """has_production pour tout le catalogue (même repli que _has_production)."""
    jas = cat["jasansky"]
    if "has_production" in cat:
        return jas.notna() & cat["has_production"].fillna(False).astype(bool)
    if not jas.notna().any():  # inutile de charger toute la production Jasansky
        return pd.Series(False, index=cat.index)
    return jas.isin(_production_ids())


def _catalog_row(site: str) -> dict:
    """Ligne du catalogue d'un site ; pour une mine de la liste suivie, celle de sa jumelle du catalogue
    (emprise, production réelle Jasansky…), sinon ces données resteraient introuvables sous son id."""
    cat = _cat()
    key = screener.twins().get(site, site)
    if cat.empty or key not in cat.index:
        return {}
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in cat.loc[key].to_dict().items()}


# -- données ------------------------------------------------------------------------
@app.get("/api/sites")
def sites():
    g = _graph()
    return [
        dict(id=s.id, nom=s.name, pays=s.country, region=s.region, lat=s.lat, lon=s.lon,
             metaux=", ".join(s.commodity), stade=s.stage,
             proprietaires=", ".join(f"{g.companies[o.company].name} ({o.share:.0%})" for o in s.owners),
             source_coordonnees=s.coord_source)
        for s in g.sites.values()
    ]


# -- catalogue mondial ----------------------------------------------------------------------
PROCESSING = ("smelter", "refinery", "plant")


def kinds(cat: pd.DataFrame) -> pd.Series:
    """0 mine nommée active (ICMM) · 1 usine/fonderie/raffinerie · 2 mine Wikidata · 3 emprise sans nom."""
    t = cat["asset_type"].fillna("").str.lower()
    processing = ~t.str.contains("mine") & t.str.contains("|".join(PROCESSING))
    k = np.select([cat["id"].str.startswith("fp"), cat["icmm"].isna(), processing], [3, 2, 1], 0)
    return pd.Series(k, index=cat.index)


@app.get("/api/catalog/points")
def catalog_points():
    """Toutes les mines, format compact pour la carte : [id, lat, lon, type, nom].

    Les jumelles des mines suivies sont retirées : la carte les montre déjà via /api/sites."""
    cat = _cat()
    if cat.empty:
        return []
    cat = cat[~cat.index.isin(list(screener.twins().values()))]
    k = kinds(cat)
    ok = np.isfinite(cat["lat"]) & np.isfinite(cat["lon"])
    return [[i, round(la, 4), round(lo, 4), int(kk), n if isinstance(n, str) else None]
            for i, la, lo, kk, n in zip(cat["id"][ok], cat["lat"][ok], cat["lon"][ok], k[ok],
                                         cat["name"][ok], strict=True)]


@app.get("/api/catalog/stats")
def catalog_stats():
    cat = _cat()
    if cat.empty:
        return {"total": 0}
    counts = kinds(cat).value_counts().to_dict()
    return {
        "total": len(cat),
        "mines_actives_icmm": int(counts.get(0, 0)),
        "usines_fonderies_raffineries": int(counts.get(1, 0)),
        "mines_wikidata": int(counts.get(2, 0)),
        "emprises_sans_nom": int(counts.get(3, 0)),
        "avec_emprise_satellite": int(cat["footprint_km2"].notna().sum()),
        "avec_production_reelle": int(_production_mask(cat).sum()),
        "avec_article_wikipedia": int(cat["wikipedia"].notna().sum()),
        # pays normalisés : « Russia » et « Russian Federation » ne comptent qu'une fois
        "pays": int(cat["country"].dropna().map(norm_country).nunique()),
        "sources": [s["cite"] for s in catalog.SOURCES.values()],
    }


@cache
def _haystack() -> tuple[pd.Series, pd.Series]:
    """Noms repliés (« Cerrejón » trouve « Cerrejon »), vidés avec les autres caches du catalogue."""
    cat = catalog.load()
    names = cat["name"].fillna("").map(_fold)
    return names, names + " " + cat["aliases"].fillna("").map(_fold)


screener.on_catalog_change(_haystack.cache_clear)


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), limit: int = 20):
    cat = _cat()
    g = _graph()
    qn = _fold(q)
    tw = screener.twins() if not cat.empty else {}
    names, hay = _haystack() if not cat.empty else (pd.Series(dtype=str), pd.Series(dtype=str))

    def tracked_hit(s) -> bool:
        # la jumelle est exclue des résultats du catalogue : ses noms et alias (« Mt Charlotte »,
        # « Hope Bay Mine »…) doivent donc trouver la mine suivie, sinon elle est introuvable sous ces noms
        t = tw.get(s.id)
        return qn in _fold(s.name) or (t in hay.index and qn in hay[t])

    out = [{"id": s.id, "nom": s.name, "pays": s.country, "suivi": True}
           for s in g.sites.values() if tracked_hit(s)]
    if not cat.empty:
        hit = cat[hay.str.contains(qn, regex=False) & ~cat.index.isin(list(tw.values()))].copy()
        hit["starts"] = ~names[hit.index].str.startswith(qn)
        hit = hit.sort_values(["starts", "footprint_km2"], ascending=[True, False], na_position="last")
        out += [{"id": r.id, "nom": _clean(r.name), "pays": norm_country(_clean(r.country)), "suivi": False}
                for r in hit.head(limit).itertuples()]
    return out[:limit]


@app.get("/api/site")
def site_detail(site: str):
    """Fiche d'une mine. L'id d'une jumelle du catalogue renvoie la fiche de la mine suivie (« id »)."""
    site = _resolve(site)
    s = _site(site)
    row = _catalog_row(site)
    g = _graph()
    state = screener.analysis_state(site)
    owners = ", ".join(f"{g.companies[o.company].name} ({o.share:.0%})" for o in s.owners) \
        or row.get("owner") or row.get("operator") or "inconnu"
    return {
        "id": s.id, "nom": s.name, "pays": s.country, "region": s.region, "lat": s.lat, "lon": s.lon,
        "metaux": ", ".join(s.commodity) or "inconnu", "stade": s.stage, "proprietaires": owners,
        "source_coordonnees": s.coord_source, "rayon_m": s.radius_m, "suivi": site in g.sites,
        "type": row.get("asset_type"), "confiance": row.get("confidence"),
        "emprise_km2": row.get("footprint_km2"), "wikidata": row.get("wikidata"),
        "wikipedia": s.wikipedia,
        # id Jasansky seulement s'il porte des données : sinon l'interface annonce une production vide
        "production_reelle": row.get("jasansky") if _has_production(row) else None,
        "sources": "liste suivie" if site in g.sites else row.get("sources"),
        # analyse = signaux optiques récents (< ~6 mois). Des signaux de la seule calibration 2016–2019
        # donnent donnees_anciennes : la fiche doit encore proposer une analyse à jour.
        "analyse": state["analyse"], "donnees_anciennes": state["donnees_anciennes"],
        "derniere_date": state["derniere_date"],
        # l'estimation d'après la surface nue Sentinel est retirée : elle suivait la taille du carré
        # d'analyse (choisi à la main pour la liste suivie), pas un état mesuré de la surface
        "production_emprise": _emprise_estimate(site),
        "statut_analyse": JOBS.get(site),
        "veille": _veille_summary(site),
    }


calibration = screener.calibration  # un seul chargeur, relu quand calibration.json change


@app.get("/api/calibration")
def calibration_api():
    """Résultats de la calibration sur la production réelle (lus par l'interface)."""
    return calibration()


def _emprise_estimate(site_id: str) -> dict | None:
    """Ordre de grandeur d'après l'emprise Maus (screener) : disponible même sans analyse satellite.

    None hors du domaine du modèle (usine, commodité inconnue ou hors population d'ajustement,
    emprise hors de la plage apprise). Facteurs et effectif lus dans la même calibration que l'estimation.
    """
    tab = screener.table()
    row = tab[tab["id"] == site_id]
    if row.empty or pd.isna(row["prod_estimee_t"].iloc[0]):
        return None
    r = row.iloc[0]
    m = calibration().get("modele_emprise", {}).get(r["prod_estimee_type"], {})
    return {"type": "charbon" if r["prod_estimee_type"] == "charbon" else "minerai extrait",
            "tonnes_an": float(r["prod_estimee_t"]), "bas_90": float(r["prod_estimee_bas90"]),
            "haut_90": float(r["prod_estimee_haut90"]), "facteur_typique": m.get("facteur_median"),
            "facteur_90": m.get("facteur_90"), "emprise_km2": float(r["emprise_km2"]),
            "mines_calibration": m.get("n")}


@app.get("/api/production")
def production(site: str):
    """Production réelle (Jasansky et al. 2023) : la vérité terrain pour calibrer le satellite."""
    jid = _catalog_row(site).get("jasansky")
    return records(catalog.production(jid)) if isinstance(jid, str) else []


# -- file d'analyses ----------------------------------------------------------------------------
# Un seul travailleur traite les mines l'une après l'autre : on peut empiler une sélection entière
# du screener sans saturer les serveurs d'images.
JOBS: dict[str, str] = {}
_queue: queue.Queue[str] = queue.Queue()
_lock = threading.Lock()
MAX_BATCH = 25
STEPS = (
    ("optical", date(2020, 1, 1), "images satellite optiques 2020 → aujourd'hui"),
    ("attention", date(2018, 1, 1), "attention Wikipedia"),
    ("radar", date(2020, 1, 1), "radar Sentinel-1 2020 → aujourd'hui"),
)


BUSY = ("en cours", "en file")


def _set(site: str, statut: str) -> None:
    with _lock:
        JOBS[site] = statut


def _analyse(site: str) -> None:
    try:
        for i, (mod, start, label) in enumerate(STEPS, 1):
            _set(site, f"en cours ({i}/{len(STEPS)}) : {label}")
            REGISTRY[mod].update(site, start, date.today())
        _set(site, "terminé")
    except Exception as e:  # l'erreur est montrée à l'utilisateur, le serveur continue
        _set(site, f"erreur : {e}")


def _worker() -> None:
    while True:
        site = _queue.get()
        try:
            _analyse(site)
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True, name="strate-analyses").start()


def _enqueue(site: str) -> tuple[str, bool]:
    """(statut, mise en file maintenant ?) — une mine déjà en file ou en cours n'est pas remise."""
    with _lock:
        st = JOBS.get(site, "")
        if st.startswith(BUSY):
            return st, False
        JOBS[site] = "en file"
        _queue.put(site)
        return JOBS[site], True


def enqueue(site: str) -> str:
    return _enqueue(site)[0]


@app.post("/api/analyze")
def analyze(site: str):
    """Met une mine du catalogue en file d'analyse (≈ 5 à 12 min par mine : optique + radar).

    Explicite : relance même une mine déjà « terminé » (seul le lot du screener l'écarte). L'id d'une
    jumelle est renvoyé vers la mine suivie, pour ne pas analyser deux fois la même mine."""
    site = _resolve(site)
    _site(site)
    return {"site": site, "statut": enqueue(site)}


@app.get("/api/jobs")
def jobs():
    """État des analyses : en file, en cours, terminées, en erreur."""
    with _lock:  # copie sous verrou : enqueue ajoute des clés depuis d'autres threads
        items = list(JOBS.items())
    return {"en_file": _queue.qsize(), "analyses": [{"site": k, "statut": v} for k, v in items]}


# -- screener ---------------------------------------------------------------------------------
COLS = ["id", "nom", "pays", "region", "categorie", "commodites", "commodite_principale", "proprietaire",
        "emprise_km2", "avertissement", "prod_estimee_type", "prod_estimee_t", "prod_estimee_bas90",
        "prod_estimee_haut90",
        "prod_reelle_serie", "prod_reelle_annee", "prod_reelle_t", "analyse", "niveau", "alerte",
        "mois_signal", "suivi", "confiance", "lat", "lon", "wikipedia", "sources"]


def _csv_list(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _filters(q: str | None = None, pays: str | None = None, commodite: str | None = None,
             categorie: str | None = None, emprise_min: float | None = None, emprise_max: float | None = None,
             prod_min: float | None = None, prod_max: float | None = None, avec_emprise: bool = False,
             avec_prod_reelle: bool = False, analyse: bool | None = None, niveau_min: int | None = None,
             bbox: str | None = Query(None, description="ouest,sud,est,nord"),
             sort: str = "prod_estimee_t", desc: bool = True) -> dict:
    box = None
    if bbox:
        try:
            box = tuple(float(x) for x in bbox.split(","))
            if len(box) != 4 or not all(math.isfinite(x) for x in box):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "bbox attendu : ouest,sud,est,nord") from None
    return {"q": q, "pays": _csv_list(pays), "commodite": _csv_list(commodite),
            "categorie": _csv_list(categorie), "emprise_min": emprise_min, "emprise_max": emprise_max,
            "prod_min": prod_min, "prod_max": prod_max, "avec_emprise": avec_emprise,
            "avec_prod_reelle": avec_prod_reelle, "analyse": analyse, "niveau_min": niveau_min,
            "bbox": box, "sort": sort, "desc": desc}


Filters = Annotated[dict, Depends(_filters)]


@app.get("/api/screener")
def screener_api(f: Filters, page: int = Query(1, ge=1),
                 per_page: int = Query(100, ge=1, le=500)):
    """Toutes les mines du catalogue, filtrées et triées. Production estimée = ordre de grandeur calibré."""
    sel = screener.query(screener.table(), **f)
    rows = sel.iloc[(page - 1) * per_page: page * per_page][COLS]
    return {"total": int(len(sel)), "page": page, "per_page": per_page, "rows": records(rows)}


@app.get("/api/screener/rows")
def screener_rows(f: Filters, limit: int = Query(200, ge=1, le=2000)):
    """Même filtre, réponse en liste simple (widget OpenBB Workspace)."""
    sel = screener.query(screener.table(), **f).head(limit)
    return records(sel[["nom", "pays", "categorie", "commodites", "emprise_km2", "prod_estimee_t",
                        "prod_estimee_bas90", "prod_estimee_haut90", "prod_reelle_t", "prod_reelle_annee",
                        "niveau", "alerte", "id"]])


@app.get("/api/screener/facets")
def screener_facets():
    return screener.facets(screener.table())


@app.get("/api/screener.csv")
def screener_csv(f: Filters):
    """Export CSV de toute la sélection (sources et licences : voir /api/catalog/stats)."""
    sel = screener.query(screener.table(), **f)[COLS]
    buf = io.StringIO()
    sel.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="strate-screener.csv"'})


@app.post("/api/screener/analyze")
def screener_analyze(f: Filters, n: int = Query(10, ge=1, le=MAX_BATCH)):
    """Met en file d'analyse les `n` premières mines de la sélection ni analysées ni déjà en file (max 25).

    Une mine en file ou en cours n'a pas encore de signaux (analyse = faux) : sans l'écarter, un second
    clic reprendrait les mêmes mines et n'ajouterait rien. Idem pour une mine « terminé » dans cette
    session sans signaux récents (aucune image exploitable) : la relancer à chaque clic coûterait 5 à
    12 min pour le même résultat ; /api/analyze la relance si on le demande explicitement.
    `mises_en_file` ne liste que les nouvelles, `deja_en_cours` les mines de la sélection qui attendaient
    déjà ou tournent, `deja_terminees` celles terminées dans cette session.
    """
    sel = screener.query(screener.table(), **f)
    with _lock:
        busy = {k for k, v in JOBS.items() if v.startswith(BUSY)}
        done = {k for k, v in JOBS.items() if v == "terminé"}
    deja = [i for i in sel["id"] if i in busy]
    fini = [i for i in sel["id"] if i in done]
    new = []
    for i in sel.loc[~sel["analyse"].astype(bool) & ~sel["id"].isin(busy | done), "id"]:
        if len(new) >= n:
            break
        st, queued = _enqueue(i)
        if queued:
            new.append({"site": i, "statut": st})
        else:  # mise en file entre-temps par un autre appel
            deja.append(i)
    return {"mises_en_file": new, "deja_en_cours": deja, "deja_terminees": fini, "en_file": _queue.qsize()}


@app.get("/api/radar")
def radar(month: str | None = Query(None, description="AAAA-MM ; vide = dernier mois", pattern=MONTH)):
    return records(fusion.radar(month))


@app.get("/api/dossier")
def dossier(site: str = "greenstone"):
    site = _resolve(site)  # même renvoi que /api/site : les signaux sont rangés sous l'id suivi
    _site(site)
    return records(fusion.dossier(site))


@app.get("/api/signals")
def signals(site: str = "greenstone", module: str | None = None):
    site = _resolve(site)
    _site(site)
    if module and module not in REGISTRY:
        raise HTTPException(404, f"module inconnu : {module!r} (disponibles : {', '.join(REGISTRY)})")
    mods = [REGISTRY[module]] if module else REGISTRY.values()
    df = pd.concat([m.load(site) for m in mods], ignore_index=True)
    return records(df)


@app.get("/api/events")
def events(site: str | None = None):
    if site:
        site = _resolve(site)
        _site(site)
    ids = [site] if site else list(_graph().sites)
    return [r for sid in ids for r in fusion.event_check(sid)]


@app.get("/api/thumbnail")
def thumbnail(site: str = "greenstone", month: str = Query(..., description="AAAA-MM", pattern=MONTH)):
    site = _resolve(site)
    _site(site)
    y, m = map(int, month.split("-"))
    path = DATA_DIR / "thumbs" / site / f"{month}.jpg"
    if not path.exists():
        png = REGISTRY["optical"].thumbnail(site, date(y, m, 1))
        if png is None:
            raise HTTPException(404, "aucune image sans nuages ce mois-là")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return Response(path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


# -- veille (actualité minière, docs/veille.md) ---------------------------------------------------------
# Lecture seule : la base appartient au service de veille (src/strate/veille). Une base absente (veille
# jamais lancée) ou des tables manquantes donnent des réponses vides, jamais une erreur serveur.
VEILLE_CONF = 0.5  # même seuil que la mise en suivi et l'étude d'événements
VEILLE_STAGES = ("collecte", "normalisation", "correlation", "service")
ISO_PREFIX = r"^\d{4}-(0[1-9]|1[0-2])(-\d{2}(T\d{2}:\d{2}(:\d{2})?Z?)?)?$"
EVENT_COLS = ("id, site_id, site_name, event_type, published_at, title, url, domain, lang, country, lat, "
              "lon, confidence, match_method")
METHODE_INDISPONIBLE = "Méthode indisponible : le module de corrélation ne se charge pas."


def _methode() -> str:
    """Texte de méthode du corrélateur lui-même (une seule source : correlate.METHODE)."""
    try:
        from strate.veille import correlate  # import tardif : numpy/pandas déjà chargés par l'API
    except Exception:  # module en panne : la page Veille reste consultable
        return METHODE_INDISPONIBLE
    m = getattr(correlate, "METHODE", None)
    return m if isinstance(m, str) and m else METHODE_INDISPONIBLE


def _vq(sql: str, params: tuple | list = ()) -> list[dict]:
    """Requête en lecture seule sur la base de veille ; [] si la base ou la table n'existe pas encore."""
    path = vstore.DB_PATH
    if not path.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return []
    con.row_factory = sqlite3.Row
    try:
        return [{k: _clean(r[k]) for k in r.keys()} for r in con.execute(sql, tuple(params))]
    except sqlite3.Error:  # table absente (étape jamais lancée), base verrouillée, fichier pas SQLite
        return []
    finally:
        con.close()


def _v1(sql: str, params: tuple | list = (), default=0):
    rows = _vq(sql, params)
    if not rows:
        return default
    v = next(iter(rows[0].values()))
    return default if v is None else v


def _since(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _site_ids(site: str) -> list[str]:
    """La mine et sa jumelle : un article peut être rattaché à l'id du catalogue ou à celui de la liste
    suivie. Le lien des jumelles demande le catalogue : sans lui, l'id seul."""
    ids = {site}
    try:
        tw = screener.twins()
    except Exception:  # catalogue absent ou illisible : la veille reste consultable
        tw = {}
    ids.update(t for w, t in tw.items() if w == site)
    ids.update(w for w, t in tw.items() if t == site)
    return sorted(ids)


def _in(ids: list[str]) -> str:
    return ",".join("?" * len(ids))


def _veille_summary(site: str) -> dict:
    """Pour la fiche d'une mine : {"events_30j", "dernier", "events_count"}.

    events_30j et dernier : événements de confiance ≥ 0,5 (30 derniers jours, plus récent).
    events_count : tous les événements de la mine, quelle que soit la confiance, comme les liste
    /api/veille/site par défaut ; 0 dit à l'interface qu'il n'y a rien à demander."""
    ids = _site_ids(site)
    n = _v1(f"SELECT count(*) FROM events WHERE site_id IN ({_in(ids)}) AND confidence >= ? "
            "AND published_at >= ?", [*ids, VEILLE_CONF, _since(30)])
    last = _v1(f"SELECT max(published_at) FROM events WHERE site_id IN ({_in(ids)}) AND confidence >= ?",
               [*ids, VEILLE_CONF], default=None)
    total = _v1(f"SELECT count(*) FROM events WHERE site_id IN ({_in(ids)})", ids)
    return {"events_30j": int(n), "dernier": last, "events_count": int(total)}


def _counters(v) -> dict:
    """Compteurs JSON d'une étape → objet sans NaN ni infini (le JSON strict de FastAPI les refuse)."""
    from strate.veille.daemon import finite

    try:
        c = json.loads(v) if isinstance(v, str) and v else {}
    except (ValueError, RecursionError):
        return {}
    return finite(c) if isinstance(c, dict) else {}


@app.get("/api/veille/status")
def veille_status():
    """État de chaque étape de la chaîne, totaux de la base et service (actif ? dernier cycle)."""
    from strate.veille import daemon  # import tardif : daemon n'est utile qu'ici

    rows = _vq("SELECT stage, last_run, ok, message, counters FROM pipeline_status")
    order = {s: i for i, s in enumerate(VEILLE_STAGES)}
    stages = [{**r, "ok": None if r["ok"] is None else bool(r["ok"]), "counters": _counters(r["counters"])}
              for r in sorted(rows, key=lambda r: (order.get(r["stage"], 99), r["stage"]))]
    count = {
        "docs": "SELECT count(*) FROM raw_docs",
        "docs_en_attente": "SELECT count(*) FROM raw_docs WHERE status='new'",
        "rejetes": "SELECT count(*) FROM raw_docs WHERE status='rejected'",
        "events": "SELECT count(*) FROM events",
        "mines_avec_evenements": "SELECT count(DISTINCT site_id) FROM events",
        "suivies": "SELECT count(*) FROM tracked",
        "fichiers_faits": "SELECT count(*) FROM feed_files WHERE status='done'",
        "fichiers_manquants": "SELECT count(*) FROM feed_files WHERE status='missing'",
    }
    svc = next((r for r in rows if r["stage"] == "service"), None)
    return {"stages": stages, "totaux": {k: int(_v1(q)) for k, q in count.items()},
            "service": daemon.service_state(svc)}


@app.get("/api/veille/types")
def veille_types():
    return vstore.EVENT_TYPES


def _event_filters(type: str | None, site: str | None, lang: str | None, country: str | None,
                   since: str | None, min_conf: float) -> tuple[str, list]:
    where, params = ["confidence >= ?"], [min_conf]
    if type:
        where.append("event_type = ?")
        params.append(type)
    if site:
        ids = _site_ids(site)
        where.append(f"site_id IN ({_in(ids)})")
        params += ids
    if lang:
        where.append("lower(lang) = lower(?)")
        params.append(lang)
    if country:
        where.append("lower(country) = lower(?)")
        params.append(country)
    if since:
        where.append("published_at >= ?")
        params.append(since)
    return " AND ".join(where), params


@app.get("/api/veille/events")
def veille_events(type: str | None = None, site: str | None = None, lang: str | None = None,
                  country: str | None = None,
                  since: str | None = Query(None, pattern=ISO_PREFIX, description="AAAA-MM[-JJ]"),
                  min_conf: float = Query(VEILLE_CONF, ge=0, le=1),
                  limit: int = Query(100, ge=1, le=1000)):
    """Événements rattachés à une mine, les plus récents d'abord."""
    where, params = _event_filters(type, site, lang, country, since, min_conf)
    total = _v1(f"SELECT count(*) FROM events WHERE {where}", params)
    rows = _vq(f"SELECT {EVENT_COLS} FROM events WHERE {where} ORDER BY published_at DESC, id LIMIT ?",
               [*params, limit])
    return {"total": int(total), "events": rows}


@app.get("/api/veille/stats")
def veille_stats(days: int = Query(30, ge=1, le=3650), min_conf: float = Query(VEILLE_CONF, ge=0, le=1)):
    """Répartition des événements des `days` derniers jours (jours sans événement inclus, à 0)."""
    since = _since(days)
    base, params = "FROM events WHERE confidence >= ? AND published_at >= ?", [min_conf, since]

    def by(col: str) -> dict:
        rows = _vq(f"SELECT coalesce({col}, 'inconnu') AS k, count(*) AS n {base} GROUP BY k "
                   "ORDER BY n DESC, k", params)
        return {r["k"]: r["n"] for r in rows}

    jours = {r["jour"]: r["n"] for r in _vq(
        f"SELECT substr(published_at, 1, 10) AS jour, count(*) AS n {base} GROUP BY jour", params)}
    d0 = datetime.now(UTC).date() - timedelta(days=days)
    par_jour = [{"jour": (j := (d0 + timedelta(days=i)).isoformat()), "n": jours.pop(j, 0)}
                for i in range(days + 1)]
    par_jour += [{"jour": j, "n": n} for j, n in sorted(jours.items())]  # dates hors plage (horloges)
    top = _vq(f"SELECT site_id, max(site_name) AS site_name, count(*) AS n {base} GROUP BY site_id "
              "ORDER BY n DESC, site_id LIMIT 20", params)
    return {"par_type": by("event_type"), "par_jour": sorted(par_jour, key=lambda r: r["jour"]),
            "par_langue": by("lang"), "par_pays": by("country"), "top_mines": top}


@app.get("/api/veille/correlations")
def veille_correlations(scope: str = "global"):
    """Dernier calcul des liens événements ↔ signaux satellite (risque relatif MH, rho, IC 90 %, p, q,
    verdict, famille principale/exploratoire). La méthode exacte vient de correlate.METHODE."""
    run_at = _v1("SELECT max(run_at) FROM correlations WHERE scope = ?", [scope], default=None)
    # verdict_rho (verdict de φ / ρ sur son propre IC) : colonne d'un schéma récent, sinon null
    has_vr = any(c.get("name") == "verdict_rho" for c in _vq("PRAGMA table_info(correlations)"))
    # famille principale = décalage 0 (seule à pouvoir conclure « lien démontré »), puis l'exploratoire
    rows = [] if run_at is None else _vq(
        "SELECT event_type, signal, lag_months, n_mines, n_events, lift, lift_lo, lift_hi, rho, rho_lo, "
        f"rho_hi, p_value, q_value, verdict, {'verdict_rho' if has_vr else 'NULL'} AS verdict_rho, "
        "CASE WHEN lag_months = 0 THEN 'principale' ELSE 'exploratoire' END AS famille "
        "FROM correlations WHERE scope = ? AND run_at = ? "
        "ORDER BY lag_months <> 0, q_value IS NULL, q_value, event_type, signal, lag_months", [scope, run_at])
    return {"run_at": run_at, "methode": _methode(), "rows": rows}


@app.get("/api/veille/site")
def veille_site(site: str, min_conf: float = Query(0.0, ge=0, le=1)):
    """Veille d'une mine : 200 derniers événements, volume d'actualité mensuel, suivi automatique.

    Volume : l'historique de l'API documentaire GDELT prime ; les mois qu'il ne couvre pas sont
    complétés par le compte des événements."""
    ids = _site_ids(site)
    events = _vq(f"SELECT {EVENT_COLS} FROM events WHERE site_id IN ({_in(ids)}) AND confidence >= ? "
                 "ORDER BY published_at DESC, id LIMIT 200", [*ids, min_conf])
    vol: dict[str, dict] = {}
    for r in _vq(f"SELECT month, sum(n_articles) AS n_articles, source FROM news_volume "
                 f"WHERE site_id IN ({_in(ids)}) GROUP BY month, source", ids):
        if r["month"] not in vol or r["source"] == "gdelt_doc_timeline":
            vol[r["month"]] = r
    tracked = _vq(f"SELECT * FROM tracked WHERE site_id IN ({_in(ids)}) ORDER BY site_id = ? DESC LIMIT 1",
                  [*ids, site])
    return {"events": events, "volume": [vol[m] for m in sorted(vol)],
            "tracked": tracked[0] if tracked else None}


@app.get("/api/veille/tracked")
def veille_tracked():
    """Mines mises en suivi automatiquement par la veille, activité la plus récente d'abord."""
    return _vq("SELECT site_id, site_name, reason, since, last_event_at, events_count, analysis_status "
               "FROM tracked ORDER BY last_event_at IS NULL, last_event_at DESC, site_id")


# -- prévisions et sociétés (docs/previsions-societes.md, § 3) ------------------------------------------
# Lecture seule. forecast et companies sont importés tard (et peuvent manquer ou être en panne) : chaque
# route répond alors une structure vide avec un « message », jamais une erreur serveur. NaN → null partout.
log = logging.getLogger(__name__)
FORECAST_DIR = DATA_DIR / "forecast"
PERIODE = {"reglage": "≤2014", "test": "2015–2021"}
METHODE_FORECAST = (
    "Rétro-test walk-forward : pour prédire l'année t à l'horizon h, seules les données ≤ t − h servent. "
    "Réglages choisis sur ≤ 2014, évaluation sur 2015–2021 (jamais vue). r de Pearson et ρ de Spearman "
    "prévision ↔ réel, IC 90 % par bootstrap sur les mines ; skill = 1 − MSE(modèle) / MSE(naïf) ; "
    "intervalles de prévision = quantiles empiriques des erreurs passées, couverture mesurée.")
METHODE_SOCIETES = (
    "Corrélations par société puis sur l'ensemble des sociétés ayant assez d'années : r de Pearson, "
    "ρ de Spearman, IC 90 % par bootstrap sur les sociétés, n sociétés et n points.")
MSG_BACKTEST = "Rétro-test pas encore lancé : uv run strate forecast backtest"
MSG_SOCIETES = "Aucune donnée société : uv run strate companies refresh"
Z90 = 1.6448536269514722


def _finite(o):
    """Objet quelconque → JSON strict : NaN/inf/NA → null, types numpy → Python, dates → ISO."""
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _finite(float(o))
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, np.ndarray)):
        return [_finite(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _finite(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _finite(o.to_list())
    if o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    try:
        return None if pd.isna(o) else str(o)
    except (TypeError, ValueError):
        return str(o)


def _rows(obj) -> list[dict]:
    """DataFrame, liste de dicts ou {"rows": […]} → liste de dicts propres."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return _finite(records(obj.reset_index(drop=True)))
    if isinstance(obj, dict):
        obj = obj.get("rows", [])
    return [_finite(r) for r in obj if isinstance(r, dict)] if isinstance(obj, (list, tuple)) else []


def _pick(row: dict, *names):
    """Première valeur non nulle parmi des noms de champ équivalents (le module peut évoluer)."""
    for n in names:
        v = row.get(n)
        if v is not None:
            return v
    return None


def _shape(row: dict, spec: dict[str, tuple[str, ...]]) -> dict:
    return {k: _finite(_pick(row, k, *alias)) for k, alias in spec.items()}


def _mod(name: str):
    """Import tardif d'un module de strate ; None s'il manque ou ne se charge pas (écrit en parallèle)."""
    try:
        return importlib.import_module(f"strate.{name}")
    except Exception as e:  # module absent ou en panne : la route répond vide, avec un message
        log.warning("module strate.%s indisponible : %s", name, e)
        return None


def _fn(mod, *names):
    return next((f for n in names if mod is not None and callable(f := getattr(mod, n, None))), None)


def _list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [t.strip() for t in v.replace(";", ",").replace("|", ",").split(",") if t.strip()]
    return [x for x in _finite(list(v)) if x is not None] if isinstance(v, (list, tuple, np.ndarray)) else [v]


# -- prévisions --
# r, ρ : variation prévue ↔ variation réelle du log de la production depuis la dernière valeur connue (null
# pour un modèle qui ne prévoit aucune variation, le naïf) ; *_niveau : mêmes mesures sur le log des niveaux.
BT_ROW = {"model": ("modele",), "horizon": ("h",), "n_mines": (), "n_points": ("n",), "r": (), "r_lo": (),
          "r_hi": (), "rho": (), "rho_lo": (), "rho_hi": (), "r_niveau": (), "r_niveau_lo": (),
          "r_niveau_hi": (), "rho_niveau": (), "rho_niveau_lo": (), "rho_niveau_hi": (), "skill": (),
          "skill_lo": (), "skill_hi": (), "facteur_median": (), "bon_sens": (), "couverture80": (),
          "couverture90": (), "note": ()}
QUALITE = {"model": ("modele",), "horizon": ("h",), "r": (), "r_lo": (), "r_hi": (), "rho": (), "rho_lo": (),
           "rho_hi": (), "r_niveau": (), "r_niveau_lo": (), "r_niveau_hi": (), "rho_niveau": (),
           "rho_niveau_lo": (), "rho_niveau_hi": (), "skill": (), "skill_lo": (), "skill_hi": (),
           "couverture80": (), "couverture90": (), "facteur_median": (), "n": ("n_points",), "unite_n": (),
           "n_mines": (), "periode": (), "portee": (), "note": ()}
# suivi propre à la mine : description sans IC (au plus 7 années cibles, dépendantes) ; la note le dit
SUIVI = {"model": ("modele",), "horizon": ("h",), "periode": (), "n": (), "unite_n": (), "facteur_median": (),
         "facteur_max": (), "note": ()}
NOTE_NAIF = ("Le modèle ne prévoit aucune variation : r et ρ (variation prévue ↔ variation réelle) "
             "n'existent pas ; r_niveau et ρ_niveau comparent les niveaux (log), dominés par les écarts de "
             "taille entre mines.")
NOTE_SUIVI = ("Description de cette mine sur la période de test (au plus 7 années cibles, dépendantes) : "
              "pas d'IC, pas un test.")
NOTE_PORTEE = ("Score global du modèle retenu sur la période de test 2015–2021 (toutes les mines), pas la "
               "précision propre à cette mine : voir suivi_site.")


def _bt_row(r: dict) -> dict:
    out = _shape(r, BT_ROW)
    if out["note"] is None and out["r"] is None and out["r_niveau"] is not None:
        out["note"] = NOTE_NAIF
    return out


def _backtest_file() -> tuple[dict | None, str | None]:
    """(contenu de data/forecast/backtest.json, message si absent ou illisible)."""
    p = FORECAST_DIR / "backtest.json"
    if not p.is_file():
        return None, MSG_BACKTEST
    try:
        d = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):  # fichier en cours d'écriture
        return None, "Résultats du rétro-test illisibles (écriture en cours ?) : réessayer."
    return (d, None) if isinstance(d, dict) else (None, "Résultats du rétro-test mal formés.")


def _backtest_payload() -> dict:
    d, msg = _backtest_file()
    d = d or {}
    retenu = d.get("retenu") if isinstance(d.get("retenu"), dict) else {}
    out = {"run_at": _finite(d.get("run_at")),
           "periode": _finite(d["periode"]) if isinstance(d.get("periode"), dict) else dict(PERIODE),
           "rows": [_bt_row(r) for r in _rows(d.get("rows"))],
           "retenu": {str(k): _finite(v) for k, v in retenu.items()},
           "methode": d.get("methode") if isinstance(d.get("methode"), str) and d["methode"]
           else METHODE_FORECAST}
    if not msg and not out["rows"] and isinstance(d.get("raison"), str):  # rétro-test lancé mais insuffisant
        msg = d["raison"]
    if msg:
        out["message"] = msg
    return out


@app.get("/api/forecast/backtest")
def forecast_backtest():
    """Bilan du rétro-test walk-forward (data/forecast/backtest.json) : r, ρ, IC 90 %, skill, couverture."""
    return _backtest_payload()


def _empty_forecast(msg: str) -> dict:
    return {"serie": None, "historique": [], "prevision": [], "qualite": None, "qualite_par_horizon": [],
            "suivi_site": None, "message": msg}


def _measured_horizons() -> set[int]:
    """Horizons mesurés hors échantillon dans le rétro-test publié (au moins un point de test)."""
    d, _ = _backtest_file()
    out = set()
    for r in _rows((d or {}).get("rows")):
        h, n = _pick(r, "horizon", "h"), _pick(r, "n_points", "n")
        if isinstance(h, int) and not isinstance(h, bool) and (not isinstance(n, (int, float)) or n > 0):
            out.add(h)
    return out


def _qualite(q) -> dict | None:
    if not isinstance(q, dict):
        return None
    out = _shape(q, QUALITE)
    out["portee"] = out["portee"] or "global"
    if out["note"] is None:
        naif = out["r"] is None and out["model"] is not None
        out["note"] = NOTE_NAIF if naif else NOTE_PORTEE
    return out


def _suivi(s) -> dict | None:
    """Suivi propre à la mine, avec la note du module qui dit pourquoi il n'a pas d'IC."""
    if not isinstance(s, dict):
        return None
    out = {**_shape(s, SUIVI), "portee": "mine"}
    out["note"] = out["note"] or NOTE_SUIVI
    return out


def _prevision(rows: list[dict], last_year, measured: set[int]) -> list[dict]:
    out = []
    for r in rows:
        p = _shape(r, {"year": ("annee", "target_year"), "horizon": ("h",), "mesure": (),
                       "model": ("modele",), "value": ("valeur",), "lo80": (), "hi80": (), "lo90": (),
                       "hi90": ()})
        if not isinstance(p["horizon"], int) and isinstance(p["year"], int) and isinstance(last_year, int):
            p["horizon"] = p["year"] - last_year
        if not isinstance(p["mesure"], bool):  # mesuré = cet horizon a un score hors échantillon publié
            p["mesure"] = isinstance(p["horizon"], int) and p["horizon"] in measured
        out.append(p)
    return out


@app.get("/api/forecast/site")
def forecast_site(site: str):
    """Historique, prévision avec intervalles 80/90 % (horizon, mesuré ou non) et qualité du modèle retenu
    (score global avec IC 90 %, et suivi propre à la mine) ; null sans série. forecast.forecast_site est en
    lecture seule : sans rétro-test, l'historique est rendu avec un message."""
    fn = _fn(_mod("forecast"), "forecast_site")
    if fn is None:
        return _empty_forecast("Prévision indisponible : module de prévision absent ou incomplet.")
    d, err = None, None
    for sid in dict.fromkeys([site, *_site_ids(site)]):  # la mine, puis ses jumelles
        try:
            d = fn(sid)
        except Exception as e:  # données absentes ou module en cours d'écriture : on tente la jumelle
            log.warning("forecast_site(%s) : %s", sid, e)
            err = e
            continue
        if d:
            break
    if not d or not isinstance(d, dict):
        return _empty_forecast(f"Prévision indisponible : {err}") if err is not None else None
    hist = [_shape(r, {"year": ("annee",), "value": ("valeur",)}) for r in _rows(d.get("historique"))]
    last = _pick(d, "derniere_annee")
    if not isinstance(last, int):
        years = [h["year"] for h in hist if isinstance(h["year"], int) and h["value"] is not None]
        last = max(years) if years else None
    out = {"site_id": _finite(d.get("site_id")), "nom": _finite(d.get("nom")),
           "serie": _finite(d.get("serie")), "unite": _finite(d.get("unite")),
           "derniere_annee": _finite(last), "historique": hist,
           "prevision": _prevision(_rows(d.get("prevision")), last, _measured_horizons()),
           "qualite": _qualite(d.get("qualite")),
           "qualite_par_horizon": [q for q in map(_qualite, _rows(d.get("qualite_par_horizon"))) if q],
           "suivi_site": _suivi(d.get("suivi_site")),
           "backtest_run_at": _finite(d.get("backtest_run_at"))}
    msg = next((m for m in (d.get("message"), d.get("note")) if isinstance(m, str) and m), None)
    if msg is None and _backtest_file()[0] is None:
        msg = _backtest_file()[1]
    if msg:
        out["message"] = msg
    return out


# -- sociétés --
CO_ROW = {"id": (), "nom": ("name",), "pays": ("country",), "tickers": (), "cik": (), "n_mines": (),
          "capex_dernier": ("capex",), "capex_annee": ("year",), "effectifs": ("employees",),
          "effectifs_annee": ("employees_year",), "ca_dernier": ("revenue",), "devise": ("currency",),
          "investissements_12m": ()}
# verdict = verdict_rho (ancien format, calculé sur l'IC de ρ) ; verdict_r porte sur l'IC de r ; q_value :
# Benjamini-Hochberg sur la famille de tests du même calcul ; n_societes = unité du bootstrap.
CO_CORR = {"x": (), "y": (), "decalage": ("lag",), "r": ("pearson_r",), "r_lo": ("pearson_lo",),
           "r_hi": ("pearson_hi",), "rho": ("spearman_rho", "spearman"), "rho_lo": ("spearman_lo",),
           "rho_hi": ("spearman_hi",), "n_societes": ("n_companies",), "n_points": ("n",), "q_value": (),
           "verdict_r": (), "verdict_rho": ("verdict",), "verdict": ("verdict_rho",)}


def _companies_rows() -> tuple[list[dict], str | None]:
    fn = _fn(_mod("companies"), "table")
    if fn is None:
        return [], "Module sociétés indisponible."
    try:
        rows = _rows(fn())
    except Exception as e:  # data/companies/ absent ou illisible
        log.warning("companies.table : %s", e)
        return [], MSG_SOCIETES
    out = []
    for r in rows:
        s = _shape(r, CO_ROW)
        s["tickers"] = _list(s["tickers"])
        if "investissements_12m" in r:  # compte d'annonces : 0 quand aucune ; absent du module → inconnu
            v = s["investissements_12m"]
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            s["investissements_12m"] = int(v) if ok else 0
        out.append(s)
    return out, None if out else MSG_SOCIETES


def _country_key(c) -> str | None:
    n = norm_country(c)
    return _fold(n) if n else None


@app.get("/api/companies")
def companies_api(q: str | None = None, pays: str | None = None, limit: int = Query(200, ge=1, le=5000)):
    """Sociétés propriétaires/exploitantes : dernier CAPEX, effectifs, CA, annonces des 12 derniers mois."""
    rows, msg = _companies_rows()
    if q:
        qn = _fold(q)

        def hay(r: dict) -> str:
            vals = (r["nom"], r["id"], r["cik"], *r["tickers"])
            return _fold(" ".join(str(v) for v in vals if v is not None))

        rows = [r for r in rows if qn in hay(r)]
    if pays:  # nom canonique (alias, nom français, ISO2), sans casse ni accent
        want = {k for p in _csv_list(pays) or [] if (k := _country_key(p))}
        rows = [r for r in rows if _country_key(r["pays"]) in want]
    out = {"total": len(rows), "rows": rows[:limit]}
    if msg:
        out["message"] = msg
    return out


FAIT = {"indicateur": ("metric", "concept"), "annee": ("year",), "valeur": ("value",), "unite": ("unit",),
        "devise": ("currency",), "source": ("source_url",), "formulaire": ("form",)}
RATIOS = {"capex_revenue", "capex_da"}


def _unit(metric: str, devise) -> str | None:
    """Unité d'une mesure du format large : montants dans la devise, ratios, croissances, par salarié."""
    if metric in RATIOS:
        return "ratio"
    if metric.endswith("_growth"):
        return "variation"
    if metric.endswith("_per_employee"):
        return f"{devise}/salarié" if devise else None
    return devise


def _faits(d: dict) -> list[dict]:
    """Faits par exercice, en long. Accepte aussi le format large (une ligne par exercice, une colonne par
    mesure, sources[mesure] = {url, form}) et les effectifs rangés à part (seules les valeurs acceptées)."""
    out = []
    for r in _rows(_pick(d, "faits", "financials")):
        if any(k in r for k in ("indicateur", "metric", "valeur", "value")):
            out.append(_shape(r, FAIT))
            continue
        year, dev = _pick(r, "annee", "year"), _pick(r, "devise", "currency")
        src = r.get("sources") if isinstance(r.get("sources"), dict) else {}
        for k, v in r.items():
            if k in ("annee", "year", "devise", "currency", "sources") or isinstance(v, bool) \
                    or not isinstance(v, (int, float)):
                continue
            s = src.get(k) if isinstance(src.get(k), dict) else {}
            out.append({"indicateur": k, "annee": year, "valeur": v, "unite": _unit(k, dev), "devise": dev,
                        "source": s.get("url"), "formulaire": s.get("form")})
    for e in _rows(d.get("employees")):
        if e.get("accepted") is False or e.get("employees") is None:
            continue
        out.append({"indicateur": "effectifs", "annee": e.get("year"), "valeur": e["employees"],
                    "unite": "salariés", "devise": None, "source": e.get("source_url"),
                    "formulaire": e.get("source_kind")})
    return out


def _site_names(ids: list[str]) -> dict[str, str]:
    """Nom de chaque mine : liste suivie, sinon catalogue (vide si ni l'un ni l'autre ne se lit)."""
    out: dict[str, str] = {}
    try:
        g = _graph().sites
        out.update({i: g[i].name for i in ids if i in g})
        cat = _cat()
        rest = [i for i in ids if i not in out and i in cat.index]
        if rest and "name" in cat:
            out.update({i: n for i, n in cat.loc[rest, "name"].items() if isinstance(n, str)})
    except Exception as e:  # catalogue absent : les ids restent
        log.warning("noms des mines indisponibles : %s", e)
    return out


# annonce d'une société : l'ancien dossier ne donnait que le type d'événement de la veille → categorie
INV = {"date": (), "montant": ("amount",), "devise": ("currency",), "categorie": ("category", "event_type"),
       "titre": ("title",), "url": ("source_url",), "site_id": ()}
# corrélations propres à une société (bootstrap sur ses années) ; verdict = verdict de ρ (ancien format)
CO_CORR_SOCIETE = {"x": (), "y": (), "decalage": ("lag",), "n": ("n_points",), "r": ("pearson_r",),
                   "r_lo": ("pearson_lo",), "r_hi": ("pearson_hi",), "rho": ("spearman_rho", "spearman"),
                   "rho_lo": ("spearman_lo",), "rho_hi": ("spearman_hi",), "verdict_r": (),
                   "verdict_rho": ("verdict",)}


def _company_payload(d: dict) -> dict:
    prof = _pick(d, "profil", "company")
    mines = [_shape(r, {"site_id": ("id", "site"), "nom": ("name", "site_name"), "part": ("share",)})
             for r in _rows(d.get("mines"))]
    if missing := [m["site_id"] for m in mines if m["nom"] is None and isinstance(m["site_id"], str)]:
        names = _site_names(missing)
        for m in mines:
            m["nom"] = m["nom"] or names.get(m["site_id"])
    return {
        "profil": _finite(prof) if isinstance(prof, dict) else {},
        "mines": mines,
        "faits": _faits(d),
        "investissements": [_shape(r, INV) for r in _rows(_pick(d, "investissements", "announcements"))],
        "correlations": [_shape(r, CO_CORR_SOCIETE) for r in _rows(d.get("correlations"))],
    }


@app.get("/api/company")
def company_api(id: str):
    """Fiche d'une société : profil, mines (part), faits par exercice avec source, annonces, corrélations."""
    mod = _mod("companies")
    fn = _fn(mod, "company", "dossier")
    empty = {"profil": {}, "mines": [], "faits": [], "investissements": [], "correlations": []}
    if fn is None:
        return {**empty, "message": "Module sociétés indisponible."}
    try:
        d = fn(id)
    except KeyError:
        d = None
    except Exception as e:  # données absentes ou illisibles
        log.warning("companies.company(%s) : %s", id, e)
        return {**empty, "message": MSG_SOCIETES}
    if not isinstance(d, dict) or not d:
        if not _companies_rows()[0]:  # aucune donnée : réponse vide expliquée, pas une 404
            return {**empty, "message": MSG_SOCIETES}
        raise HTTPException(404, f"société inconnue : {id!r}")
    return _company_payload(d)


def _companies_corr() -> tuple[list[dict], str | None, str, list[dict]]:
    """(lignes au format de la route, message, méthode, lignes brutes du module)."""
    mod = _mod("companies")
    fn = _fn(mod, "correlations")
    cands = (getattr(mod, n, None) for n in ("METHODE_CORRELATIONS", "METHODE"))
    methode = next((m for m in cands if isinstance(m, str) and m), METHODE_SOCIETES)
    if fn is None:
        return [], "Module sociétés indisponible.", methode, []
    try:
        res = fn()
    except Exception as e:
        log.warning("companies.correlations : %s", e)
        return [], MSG_SOCIETES, methode, []
    if isinstance(res, dict) and isinstance(res.get("methode"), str) and res["methode"]:
        methode = res["methode"]
    raw = [r for r in _rows(res) if r.get("portee") in (None, "ensemble")]  # table globale seulement
    rows = [_shape(r, CO_CORR) for r in raw]
    return rows, None if rows else MSG_SOCIETES, methode, raw


@app.get("/api/companies/correlations")
def companies_correlations():
    """Corrélations sociétés (CAPEX ↔ production, effectifs ↔ production, CAPEX ↔ emprise) : r, ρ, IC, n."""
    rows, msg, methode, _ = _companies_corr()
    out = {"rows": rows, "methode": methode}
    if msg:
        out["message"] = msg
    return out


# -- toutes les corrélations (page « Corrélations ») --
# Chaque ligne porte le verdict de SON coefficient : celui du module quand il l'a calculé sur ce coefficient
# (verdict_r, verdict_rho, verdict du lift), sinon un verdict tiré de l'IC de la ligne. Ce dernier ne dit
# jamais « démontré » là où le module exige davantage que l'IC (q-valeur, nombre d'unités indépendantes).
SYM = {"r": "r", "rho": "ρ", "phi": "φ", "lift": "lift"}


def _verdict(lo, hi, null: float = 0.0, sym: str = "ρ") -> str:
    """Verdict de l'IC 90 % seul (calibration, prévisions : un test par coefficient publié)."""
    if lo is None or hi is None:
        return f"IC 90 % de {sym} indisponible"
    if lo > null:
        return f"lien positif : IC 90 % de {sym} au-dessus de {null:g}" if null == 0 \
            else f"risque accru : IC 90 % du {sym} au-dessus de {null:g}"
    if hi < null:
        return f"lien négatif : IC 90 % de {sym} sous {null:g}" if null == 0 \
            else f"risque réduit : IC 90 % du {sym} sous {null:g}"
    return f"non démontré : l'IC 90 % de {sym} contient {null:g}"


def _verdict_sans_test(lo, hi, null: float, sym: str, manque: str) -> str:
    """Verdict de repli pour un module dont la règle exige plus que l'IC : jamais « démontré »."""
    v = _verdict(lo, hi, null, sym)
    if v.startswith(("lien", "risque")):
        return f"non démontré : IC 90 % de {sym} hors de {null:g}, mais {manque}"
    return v


def _approx_ci(rho, n) -> tuple[float | None, float | None]:
    """IC 90 % approché d'un ρ publié sans intervalle (Fisher z, erreur type de Bonett–Wright)."""
    if not isinstance(rho, (int, float)) or not isinstance(n, int) or n <= 3 or not -1 < rho < 1:
        return None, None
    se = math.sqrt((1 + rho * rho / 2) / (n - 3))
    z = math.atanh(rho)
    return math.tanh(z - Z90 * se), math.tanh(z + Z90 * se)


def _sum_row(domaine, x, y, coef, typ, lo, hi, n, unite_n, n_boot, unite_boot, source,
             verdict=None) -> dict | None:
    coef, lo, hi, n, n_boot = (_finite(v) for v in (coef, lo, hi, n, n_boot))
    if coef is None:
        return None
    if not (isinstance(verdict, str) and verdict.strip()):
        verdict = _verdict(lo, hi, 1.0 if typ == "lift" else 0.0, SYM[typ])
    return {"domaine": domaine, "x": x, "y": y, "coefficient": coef, "type": typ, "lo": lo, "hi": hi, "n": n,
            "unite_n": unite_n, "n_bootstrap_unit": n_boot, "unite_bootstrap": unite_boot, "verdict": verdict,
            "source": source}


def _triple(v) -> tuple:
    """[valeur, bas, haut] de calibration.json → tuple de trois (None si absent)."""
    return tuple(v[:3]) if isinstance(v, (list, tuple)) and len(v) >= 3 else (None, None, None)


def _calibration_summary() -> list[dict]:
    c = calibration() or {}
    src = "calibration.json"
    out = []
    t = c.get("taille") or {}
    if t:
        lo, hi = (t.get("ic90") or [None, None])[:2]
        out.append(_sum_row("calibration", "surface nue Sentinel-2 (fenêtre sur l'emprise)",
                            "production annuelle (niveau)", t.get("rho"), "rho", lo, hi, t.get("mines"),
                            "mines", t.get("mines"), "mines", src))
        for k, x in (("fenetre_seule", "taille de la fenêtre seule (sans image)"),
                     ("fraction_nue", "surface nue / fenêtre")):
            s = t.get(k) or {}
            out.append(_sum_row("calibration", x, "production annuelle (niveau)", s.get("rho"), "rho",
                                s.get("ic90_bas"), s.get("ic90_haut"), s.get("mines"), "mines",
                                s.get("mines"), "mines", src))
    v = c.get("variation") or {}
    if v:
        lo, hi = (v.get("ic90") or [None, None])[:2]
        out.append(_sum_row("calibration", "variation annuelle de la surface nue", "variation annuelle de la "
                            "production", v.get("rho"), "rho", lo, hi, c.get("paires_annees"),
                            "paires d'années", v.get("mines") or c.get("mines"), "mines", src))
    for k, m in (c.get("modele_emprise") or {}).items():
        if isinstance(m, dict) and m.get("rho") is not None:
            lo, hi = _approx_ci(m.get("rho"), m.get("n"))
            out.append(_sum_row("calibration", "emprise Maus (km²)", f"production ({k})", m["rho"], "rho",
                                lo, hi, m.get("n"), "mines", m.get("n"),
                                "mines (IC analytique, pas de bootstrap)",
                                src + " ; IC 90 % approché (Fisher z)"))
    r = c.get("radar") or {}
    for k, x, y in (("variation", "variation de l'activité radar Sentinel-1",
                     "variation annuelle de la production"),
                    ("intra_mine_niveau", "activité radar (écart à la moyenne de la mine)",
                     "production (écart à la moyenne de la mine)")):
        rho, lo, hi = _triple(r.get(k))
        out.append(_sum_row("calibration", x, y, rho, "rho", lo, hi, r.get("mines_test"), "mines",
                            r.get("mines_test"), "mines", src + " (radar)"))
    return out


def _veille_summary_rows() -> list[dict]:
    """Lignes lift (verdict du module : q < 0,10, décalage 0, IC hors de 1) et φ / ρ (verdict_rho du
    module, sinon repli prudent) ; n = mois-événements (mois avec actualité pour le volume), bootstrap sur
    les mines."""
    d = veille_correlations("global")
    src = f"veille (calcul du {d['run_at']})" if d.get("run_at") else "veille"
    out = []
    for r in d.get("rows", []):
        lag = r.get("lag_months") or 0
        y = f"{r.get('signal')} (décalage {lag:+d} mois)" if lag else str(r.get("signal"))
        et = r.get("event_type")
        volume = et == "volume"
        x = "volume d'actualité" if volume else f"événement « {et} »"
        unite = "mois avec actualité" if volume else "mois-événements"
        out.append(_sum_row("veille", x, y, r.get("lift"), "lift", r.get("lift_lo"), r.get("lift_hi"),
                            r.get("n_events"), unite, r.get("n_mines"), "mines", src,
                            r.get("verdict") or _verdict_sans_test(r.get("lift_lo"), r.get("lift_hi"), 1.0,
                                                                   "lift", "verdict du module absent")))
        typ = "rho" if volume else "phi"
        vr = r.get("verdict_rho")
        if not vr and volume:  # ligne de volume : le verdict du module porte sur ρ, seul coefficient
            vr = r.get("verdict")
        if not vr:
            vr = "exploratoire" if lag else "trop peu de données" if r.get("q_value") is None \
                else _verdict_sans_test(r.get("rho_lo"), r.get("rho_hi"), 0.0, SYM[typ], "sans test propre")
        out.append(_sum_row("veille", x, y, r.get("rho"), typ, r.get("rho_lo"), r.get("rho_hi"),
                            r.get("n_events"), unite, r.get("n_mines"), "mines", src, vr))
    return out


def _companies_summary_rows() -> list[dict]:
    """Table globale des sociétés : verdict_r pour r, verdict_rho (ou l'ancien verdict, calculé sur ρ) pour
    ρ ; sans verdict du module, repli qui ne dit jamais « démontré » (règle : IC, q < 0,10 et assez de
    sociétés)."""
    out = []
    rows, _, _, raw = _companies_corr()
    for r, b in zip(rows, raw, strict=True):
        lag = r.get("decalage")
        y = f"{r['y']} (décalage {lag:+d} an)" if isinstance(lag, int) and lag else str(r["y"])
        x = str(r["x"])
        if isinstance(b.get("label"), str) and b["label"]:  # libellé lisible du module (« CAPEX ↔ … »)
            x, _, yl = b["label"].partition(" ↔ ")
            y = yl or y
        for typ in ("r", "rho"):
            v = r.get(f"verdict_{typ}") or _verdict_sans_test(
                r[f"{typ}_lo"], r[f"{typ}_hi"], 0.0, SYM[typ], "verdict du module absent (q-valeur et nombre "
                "de sociétés non vérifiés)")
            out.append(_sum_row("sociétés", x, y, r[typ], typ, r[f"{typ}_lo"], r[f"{typ}_hi"], r["n_points"],
                                "paires société-année", r["n_societes"], "sociétés", "data/companies", v))
    return out


NOTE_NIVEAUX = " ; niveaux : dominés par la taille des mines, pas une mesure de la prévision des variations"


def _forecast_summary_rows() -> list[dict]:
    """r et ρ du rétro-test : variation prévue ↔ variation réelle du log de la production (période de test).
    Pour un modèle qui ne prévoit aucune variation (naïf, ou tendance réglée sur φ = 0), r et ρ n'existent
    pas : leurs lignes portent alors sur les niveaux (log), dominés par les écarts de taille entre mines.
    Des modèles aux mesures identiques (mêmes prévisions) partagent une paire de lignes, nommés ensemble."""
    d = _backtest_payload()
    src = "data/forecast/backtest.json" + (f" ({d['run_at']})" if d.get("run_at") else "")
    test = (d.get("periode") or {}).get("test") or PERIODE["test"]

    def a(models: list, h) -> str:
        noms = " = ".join(f"« {m} »" for m in models)
        return f"{noms} à {h} an{'s' if isinstance(h, int) and h > 1 else ''}"

    out, variations, niveaux = [], {}, {}
    for r in d["rows"]:  # mêmes prévisions → mêmes mesures : une seule paire de lignes, modèles regroupés
        niveau = r.get("r") is None and r.get("rho") is None
        k = "_niveau" if niveau else ""
        key = (r.get("horizon"), r.get("n_points"), *(r.get(f"{c}{k}{b}") for c in ("r", "rho")
                                                      for b in ("", "_lo", "_hi")))
        (niveaux if niveau else variations).setdefault(key, (r, []))[1].append(r.get("model"))
    for r, models in variations.values():
        for typ in ("r", "rho"):
            out.append(_sum_row("prévisions", f"variation prévue par {a(models, r.get('horizon'))}",
                                f"variation réelle du log de la production, {test}", r.get(typ), typ,
                                r.get(f"{typ}_lo"), r.get(f"{typ}_hi"), r.get("n_points"),
                                "points mine-année", r.get("n_mines"), "mines", src))
    for r, models in niveaux.values():
        for typ in ("r", "rho"):
            k = f"{typ}_niveau"
            out.append(_sum_row("prévisions", f"niveau prévu (log) par {a(models, r.get('horizon'))}",
                                f"niveau réel (log) de la production, {test}", r.get(k), typ,
                                r.get(f"{k}_lo"), r.get(f"{k}_hi"), r.get("n_points"), "points mine-année",
                                r.get("n_mines"), "mines", src + NOTE_NIVEAUX))
    return out


@app.get("/api/correlations/summary")
def correlations_summary():
    """Toutes les corrélations connues de l'application, une ligne par coefficient (r, ρ, φ ou lift) avec
    IC 90 %, n et ce qu'il compte (unite_n), nombre d'unités rééchantillonnées par le bootstrap, verdict de
    ce coefficient et source : calibration, veille, sociétés, prévisions."""
    rows, domaines = [], []
    for nom, fn in (("calibration", _calibration_summary), ("veille", _veille_summary_rows),
                    ("sociétés", _companies_summary_rows), ("prévisions", _forecast_summary_rows)):
        try:
            got = [r for r in fn() if r is not None]
            msg = None if got else "aucune corrélation calculée"
        except Exception as e:  # une source en panne ne masque pas les autres
            log.warning("corrélations %s : %s", nom, e)
            got, msg = [], f"source indisponible : {e}"
        rows += got
        domaines.append({"domaine": nom, "n": len(got), "message": msg})
    return {"rows": rows, "domaines": domaines}


# -- OpenBB Workspace ---------------------------------------------------------------------
def _site_param():
    return {
        "paramName": "site", "value": "greenstone", "label": "Site", "type": "text",
        "description": "Mine à afficher",
        "options": [{"label": s.name, "value": s.id} for s in _graph().sites.values()],
    }


def _pays_param():
    return {"paramName": "pays", "value": "", "label": "Pays", "type": "text",
            "description": "Un ou plusieurs pays séparés par des virgules (vide = tous)"}


def _commodite_param():
    return {"paramName": "commodite", "value": "", "label": "Commodité", "type": "text",
            "description": "Ex. or, cuivre, charbon (vide = toutes)",
            "options": [{"label": "toutes", "value": ""}]
            + [{"label": g, "value": g} for g, _ in screener.COMMODITY_GROUPS]}


_PARAMS = {"site": _site_param, "pays": _pays_param, "commodite": _commodite_param}

WIDGETS = {
    "strate_radar": dict(name="Radar des mines", endpoint="api/radar", w=24, h=10,
                         description="Niveau de convergence de chaque mine, dernier mois.", params=[]),
    "strate_dossier": dict(name="Dossier d'une mine", endpoint="api/dossier", w=40, h=14,
                           description="Signaux mensuels, scores z et explication des alertes.",
                           params=["site"]),
    "strate_signals": dict(name="Signaux bruts", endpoint="api/signals", w=24, h=12,
                           description="Toutes les mesures avec unité, qualité et source.",
                           params=["site"]),
    "strate_events": dict(name="Contrôle sur événements connus", endpoint="api/events", w=24, h=8,
                          description="Les alertes tombent-elles au moment des événements réels ?",
                          params=[]),
    "strate_screener": dict(name="Screener des mines du monde", endpoint="api/screener/rows", w=40, h=16,
                            description="~80 000 sites : emprise, production estimée (ordre de grandeur "
                                        "calibré), production réelle publiée, alertes.",
                            params=["pays", "commodite"]),
    "strate_sites": dict(name="Sites suivis", endpoint="api/sites", w=24, h=8,
                         description="Mines, propriétaires et source des coordonnées.", params=[]),
}


@app.get("/widgets.json")
def widgets_json():
    out = {}
    for wid, w in WIDGETS.items():
        out[wid] = {
            "name": w["name"],
            "description": w["description"],
            "category": "Strate",
            "subCategory": "Mine",
            "type": "table",
            "endpoint": w["endpoint"],
            "gridData": {"w": w["w"], "h": w["h"]},
            "source": ["Strate"],
            "params": [_PARAMS[p]() for p in w["params"]],
        }
    return out


# /apps.json (tableau de bord prêt à l'emploi) : à ajouter une fois le format validé
# dans Workspace — il est optionnel, et un fichier mal formé gênerait plus qu'il n'aide.


# -- interface autonome ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return files("strate.web").joinpath("index.html").read_text("utf-8")
