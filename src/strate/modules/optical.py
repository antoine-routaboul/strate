"""Activité d'un site minier vue par Sentinel-2 (optique, 10–20 m, gratuit).

Pour chaque mois : composite médian des meilleures scènes sans nuages au-dessus du site,
puis trois signaux :
  - bare_km2   : surface sans végétation (fosses, verses, routes, plateformes)
  - water_km2  : surface en eau (bassins de résidus, fosses ennoyées)
  - change_km2 : surface qui a changé par rapport au même mois un an plus tôt
                 (comparer au même mois neutralise les saisons)
quality = part du carré réellement observée sans nuage/neige ce mois-là.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from strate import entities, geo
from strate.core import DATA_DIR, Module, frame

log = logging.getLogger(__name__)

COLLECTION = "sentinel-2-l2a"
RES = 20.0  # m : résolution commune (la bande SWIR B11 est à 20 m)
MAX_TILE_CLOUD = 60  # % : on filtre large, puis on juge les nuages AU-DESSUS DU SITE
SCENES_PER_MONTH = 3
SCL_CHECKS_PER_MONTH = 6
MIN_SCENE_VALID = 0.5
MIN_MONTH_VALID = 0.6
# Un mois n'est figé dans le cache qu'une fois fini ET après ce délai : le catalogue Planetary Computer
# publie les scènes avec un à deux jours de retard ; un composite écrit trop tôt resterait incomplet à vie.
CATALOG_LAG_DAYS = 3

# Scene Classification Layer de Sentinel-2 : classes gardées comme « vues claires »
SCL_CLEAR = (4, 5, 6)  # végétation, sol nu, eau
SCL_WATER = 6

# Seuils (à calibrer dans le backtest)
NDVI_BARE = 0.2
SWIR_CHANGE = 0.08
NDVI_DROP = 0.2
WATER_NIR = 0.06
WATER_SWIR = 0.05


_BANDS = ("red", "nir", "swir")


def pack(comp: dict) -> dict:
    """Stockage compact : réflectance ×10 000 en uint16 (0 = pas vu), eau en % uint8.

    ~4× plus léger que des float32 pleins de NaN, précision 1e-4 (bien sous le bruit du capteur).
    """
    out = dict(comp)
    for b in _BANDS:
        if b in comp:
            a = comp[b]
            out[b] = np.where(np.isfinite(a), np.clip(np.round(a * 10000) + 1, 1, 65535), 0).astype("uint16")
    if "water" in comp:
        out["water"] = np.round(comp["water"] * 100).astype("uint8")
    out["packed"] = np.array(1, "uint8")
    return out


def unpack(z) -> dict:
    d = {k: z[k] for k in z.files}
    if "packed" not in d:  # ancien format float32 : déjà prêt
        return d
    for b in _BANDS:
        if b in d:
            a = d[b].astype("float32")
            d[b] = np.where(a > 0, (a - 1) / 10000, np.nan).astype("float32")
    if "water" in d:
        d["water"] = d["water"].astype("float32") / 100
    del d["packed"]
    return d


def thresholds() -> dict:
    """Seuils en vigueur (valeurs du module, éventuellement remplacées par la calibration)."""
    return {"ndvi_bare": NDVI_BARE, "swir_change": SWIR_CHANGE, "ndvi_drop": NDVI_DROP,
            "water_nir": WATER_NIR, "water_swir": WATER_SWIR}


def is_water(c: dict, p: dict | None = None) -> np.ndarray:
    """Eau = classée eau par l'ESA (SCL) ET spectre d'eau (proche IR et SWIR très bas).

    Le SCL seul confond ombres de nuages et sols sombres avec de l'eau : sur Canadian Malartic
    la combinaison divise par ~2 la variation mensuelle parasite de la surface en eau.
    """
    p = p or thresholds()
    with np.errstate(invalid="ignore"):
        return (c["water"] > 0.5) & (c["nir"] < p["water_nir"]) & (c["swir"] < p["water_swir"])


def reflectance(dn: np.ndarray, baseline: str | None) -> np.ndarray:
    """DN → réflectance. Depuis la baseline 04.00 (janv. 2022) l'ESA ajoute +1000 aux DN."""
    r = dn.astype("float32")
    offset = 1000.0 if baseline and float(baseline) >= 4.0 else 0.0
    r = (r - offset) / 10000.0
    r[dn == 0] = np.nan
    return r


def cache_dir(site_id: str, scenes_per_month: int = SCENES_PER_MONTH,
              scl_checks: int = SCL_CHECKS_PER_MONTH) -> Path:
    """Dossier des composites d'un site pour un réglage donné (voir Optical._cache)."""
    d = DATA_DIR / "cache" / "optical" / site_id
    if (scenes_per_month, scl_checks) == (SCENES_PER_MONTH, SCL_CHECKS_PER_MONTH):
        return d
    return d / f"s{scenes_per_month}c{scl_checks}"


def month_closes(month: date) -> datetime:
    """Instant à partir duquel le composite d'un mois est définitif (fin du mois + délai du catalogue)."""
    nxt = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return datetime.combine(nxt + timedelta(days=CATALOG_LAG_DAYS), datetime.min.time())


def persist(month: date, failed: int, n_scenes: int, wanted: int, today: date | None = None) -> bool:
    """Le composite peut-il être figé dans le cache ?

    Non si le mois n'est pas clos (d'autres scènes arriveront ; écrit en cours de mois, il restait
    définitivement incomplet). Non si des lectures ont échoué sans qu'on ait obtenu le nombre de scènes
    voulu : la scène illisible était peut-être la seule claire (sinon une panne réseau passagère gelait un
    mois « sans données » pour toujours).
    """
    today = today or date.today()
    if datetime.combine(today, datetime.min.time()) < month_closes(month):
        return False
    return failed == 0 or n_scenes >= wanted


class Optical(Module):
    id = "optical"
    title = "Activité satellite (Sentinel-2)"
    description = "Surfaces nues, en eau et modifiées autour d'un site minier, mois par mois."
    entity_kind = "site"
    signals = {
        "bare_km2": "Surface sans végétation (fosses, verses, routes) en km²",
        "water_km2": "Surface en eau (bassins, fosses ennoyées) en km²",
        "change_km2": "Surface de l'empreinte minière modifiée vs le même mois un an plus tôt, en km²",
        "new_bare_km2": "Surface nouvellement mise à nu sur un an (défrichement, expansion), en km²",
    }

    def __init__(self, workers: int = 8, scenes_per_month: int = SCENES_PER_MONTH,
                 scl_checks: int = SCL_CHECKS_PER_MONTH, read_legacy: bool = False):
        self.workers = workers
        self.scenes_per_month = scenes_per_month  # 1 = plus rapide (calibration), 3 = moins bruité
        self.scl_checks = scl_checks
        # relire aussi les composites non étiquetés du chemin historique (voir _cached) : réservé à la
        # calibration, qui les a écrits elle-même avec 1 scène avant la séparation des réglages
        self.read_legacy = read_legacy

    # -- catalogue ----------------------------------------------------------------
    def _items(self, site, start: date, end: date):
        search = geo.catalog().search(
            collections=[COLLECTION],
            intersects={"type": "Point", "coordinates": [site.lon, site.lat]},
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}},
        )
        for attempt in range(5):  # le catalogue STAC coupe parfois la connexion en pleine réponse
            try:
                found = list(search.items())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log.warning("%s : recherche STAC interrompue (%s), nouvel essai", site.id, e)
                time.sleep(5 * (attempt + 1))
        by_month = defaultdict(list)
        for it in found:
            d = it.datetime.date()
            by_month[date(d.year, d.month, 1)].append(it)
        for v in by_month.values():
            v.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        return by_month

    # -- composite mensuel ----------------------------------------------------------
    def _cache(self, site_id: str, month: date) -> Path:
        """Emplacement du composite, propre au réglage (scènes par mois, scènes SCL examinées).

        Réglage par défaut (3 scènes) : chemin historique, pour ne pas invalider le cache existant.
        Autre réglage : sous-dossier « s{scènes}c{contrôles} » (la calibration écrit dans « s1c4 »).
        Sans cette séparation, un composite à 1 scène écrit par la calibration était relu comme un composite
        à 3 scènes (et inversement) : même fichier, bruit différent. Les composites 1 scène écrits AVANT cette
        séparation sont restés au chemin historique : relus en repli si read_legacy (et par calibrate._comps).
        """
        return cache_dir(site_id, self.scenes_per_month, self.scl_checks) / f"{month:%Y-%m}.npz"

    def _cache_valid(self, path: Path, month: date) -> bool:
        """Un composite en cache n'est réutilisé que s'il a été écrit APRÈS la clôture du mois et avec le
        même réglage (fichiers récents : clé « reglage » ; anciens fichiers sans clé : acceptés)."""
        if path.stat().st_mtime < month_closes(month).timestamp():
            return False  # écrit pendant le mois (version antérieure du code) : à recalculer
        with np.load(path) as z:
            if "reglage" in z.files:
                return z["reglage"].tolist() == [self.scenes_per_month, self.scl_checks]
        return True

    def _cached(self, site_id: str, month: date) -> Path | None:
        """Composite réutilisable : chemin du réglage, sinon (read_legacy) l'ancien chemin non étiqueté.

        Un fichier du chemin historique écrit depuis la séparation porte reglage = [3, 6] : il est refusé
        par _cache_valid pour un autre réglage, seul un fichier ancien (sans étiquette) passe le repli.
        """
        path = self._cache(site_id, month)
        if path.exists() and self._cache_valid(path, month):
            return path
        old = cache_dir(site_id) / path.name
        if self.read_legacy and old != path and old.exists() and self._cache_valid(old, month):
            return old
        return None

    def _scene(self, item, grid):
        scl = geo.read(item.assets["SCL"].href, grid, Resampling.nearest)[0]
        clear = np.isin(scl, SCL_CLEAR)
        frac = float(clear.mean())
        if frac < MIN_SCENE_VALID:
            return None
        bl = item.properties.get("s2:processing_baseline")
        red, nir, swir = (
            reflectance(geo.read(item.assets[b].href, grid, Resampling.average)[0], bl)
            for b in ("B04", "B08", "B11")
        )
        clear &= np.isfinite(red) & np.isfinite(nir) & np.isfinite(swir)
        return clear, scl == SCL_WATER, red, nir, swir, item.id

    def composite(self, site, month: date, items, today: date | None = None) -> dict | None:
        """Composite médian du mois. Figé dans le cache seulement s'il est définitif (voir persist)."""
        cached = self._cached(site.id, month)
        if cached is not None:
            with np.load(cached) as z:
                return unpack(z)
        path = self._cache(site.id, month)
        grid = geo.site_grid(site.lon, site.lat, site.radius_m, RES)
        scenes, failed = [], 0
        for it in items[:self.scl_checks]:
            try:
                s = self._scene(it, grid)
            except Exception as e:  # une scène illisible ne bloque pas le mois
                log.warning("%s %s : %s", site.id, it.id, e)
                failed += 1
                continue
            if s:
                scenes.append(s)
            if len(scenes) >= self.scenes_per_month:
                break
        if not scenes:
            comp = {"valid": np.zeros((grid.n, grid.n), "uint8"), "scenes": np.array([], "U80")}
        else:
            clear = np.stack([s[0] for s in scenes])
            water = np.stack([s[1] for s in scenes])

            def med(i):
                a = np.stack([s[i] for s in scenes])
                a[~clear] = np.nan
                with warnings.catch_warnings():  # pixels jamais vus : NaN attendu
                    warnings.simplefilter("ignore", RuntimeWarning)
                    return np.nanmedian(a, axis=0).astype("float32")

            valid = clear.sum(0)
            with np.errstate(all="ignore"):
                water_frac = np.where(valid > 0, (water & clear).sum(0) / valid, 0).astype("float32")
            comp = {
                "valid": valid.astype("uint8"),
                "water": water_frac,
                "red": med(2),
                "nir": med(3),
                "swir": med(4),
                "scenes": np.array([s[5] for s in scenes]),
            }
        if persist(month, failed, len(scenes), self.scenes_per_month, today):
            path.parent.mkdir(parents=True, exist_ok=True)
            reglage = np.array([self.scenes_per_month, self.scl_checks], "uint8")
            np.savez_compressed(path, **pack(comp), reglage=reglage)
        else:  # rendu en mémoire seulement : recalculé au prochain passage
            log.info("%s %s : composite provisoire, non mis en cache", site.id, f"{month:%Y-%m}")
        return comp

    # -- signaux --------------------------------------------------------------------
    @staticmethod
    def measures(comp: dict, prev: dict | None, pixel_km2: float, p: dict | None = None) -> dict:
        """Signaux d'un mois. `p` remplace les seuils par défaut (utilisé par la calibration)."""
        p = {**thresholds(), **(p or {})}
        valid = comp["valid"] > 0
        n = valid.size
        vf = float(valid.mean())
        out = {"quality": vf}
        if vf < MIN_MONTH_VALID:
            return out
        total = n * pixel_km2
        with np.errstate(all="ignore"):
            ndvi = (comp["nir"] - comp["red"]) / (comp["nir"] + comp["red"])
        water = is_water(comp, p) & valid
        bare = (ndvi < p["ndvi_bare"]) & ~water & valid
        out["water_km2"] = water.sum() / valid.sum() * total
        out["bare_km2"] = bare.sum() / valid.sum() * total
        if prev is not None and "red" in prev:
            both = valid & (prev["valid"] > 0)
            if both.mean() >= MIN_MONTH_VALID:
                with np.errstate(all="ignore"):
                    pndvi = (prev["nir"] - prev["red"]) / (prev["nir"] + prev["red"])
                    pwater = is_water(prev, p)
                    pbare = (pndvi < p["ndvi_bare"]) & ~pwater
                    changed = (np.abs(comp["swir"] - prev["swir"]) > p["swir_change"]) | (
                        pndvi - ndvi > p["ndvi_drop"]
                    )
                # On ne garde que l'empreinte minière : sol nu à l'une des deux dates.
                # (Vérifié visuellement sur Greenstone 2024→2025 : la végétation seule
                #  fluctue d'une année sur l'autre, l'activité minière se lit sur le sol nu.)
                changed &= both & (bare | pbare)
                out["change_km2"] = changed.sum() / both.sum() * total
                out["new_bare_km2"] = (bare & ~pbare & both).sum() / both.sum() * total
        return out

    def run(self, entity_id: str, start: date, end: date) -> pd.DataFrame:
        site = entities.load().site(entity_id)
        # un an de plus en amont pour pouvoir comparer au même mois de l'année précédente
        lookback = date(start.year - 1, start.month, 1)
        by_month = self._items(site, lookback, end)
        months = sorted(by_month)
        with ThreadPoolExecutor(self.workers) as ex:
            built = ex.map(lambda m: self.composite(site, m, by_month[m]), months)
            comps = dict(zip(months, built, strict=True))
        px = geo.site_grid(site.lon, site.lat, site.radius_m, RES).pixel_km2
        rows = []
        src = f"Copernicus Sentinel-2 L2A via Microsoft Planetary Computer ({COLLECTION})"
        for m in months:
            if m < start:
                continue
            prev = comps.get(date(m.year - 1, m.month, 1))
            meas = self.measures(comps[m], prev, px)
            q = meas.pop("quality")
            for sig, val in meas.items():
                rows.append(
                    dict(entity=site.id, date=m, signal=sig, value=round(float(val), 4),
                         unit="km2", quality=round(q, 3), source=src)
                )
        return frame(rows)

    # -- image pour l'interface -----------------------------------------------------------
    def thumbnail(self, entity_id: str, month: date) -> bytes | None:
        """JPEG couleurs naturelles (10 m) de la scène la plus claire du mois."""
        from PIL import Image

        site = entities.load().site(entity_id)
        end = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        items = self._items(site, month, end).get(month, [])
        grid10 = geo.site_grid(site.lon, site.lat, site.radius_m, 10.0)
        grid20 = geo.site_grid(site.lon, site.lat, site.radius_m, RES)
        for it in items[:SCL_CHECKS_PER_MONTH]:
            scl = geo.read(it.assets["SCL"].href, grid20, Resampling.nearest)[0]
            if np.isin(scl, SCL_CLEAR).mean() < 0.9:
                continue
            rgb = geo.read(it.assets["visual"].href, grid10, Resampling.bilinear, [1, 2, 3])
            buf = BytesIO()
            Image.fromarray(np.moveaxis(rgb, 0, -1)).save(buf, "JPEG", quality=85)
            return buf.getvalue()
        return None
