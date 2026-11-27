"""Activité d'un site minier vue par le radar Sentinel-1 (bande C, 20 m, à travers nuages et nuit).

Idée : creuser, déplacer des stocks, faire rouler des engins change la rugosité de la surface, donc
le signal radar renvoyé. On mesure à quel point il change d'un passage au suivant (12 jours, même
orbite = même géométrie de vue), dans l'emprise de la mine (contours Maus 2022) et dans une couronne
autour (même météo).

Le chatoiement radar (grain aléatoire) domine à 20 m : on moyenne d'abord sur des blocs de 100 m.
Plusieurs métriques candidates sont calculées ; la calibration sur la production réelle choisit
(testé sur le Super Pit : la couronne réagit plus que la mine à la pluie, seuls les gros changements
> 3 dB ressortent côté mine — la couronne n'est donc pas une référence neutre).

Source : Sentinel-1 RTC (rétrodiffusion corrigée du relief), Microsoft Planetary Computer, CC BY 4.0.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.features import rasterize

from strate import catalog, entities, geo
from strate.core import DATA_DIR, Module, frame

log = logging.getLogger(__name__)

COLLECTION = "sentinel-1-rtc"
RES = 20.0
RING_M = 1500  # largeur de la couronne de référence autour du carré du site
GAP_M = 300  # zone tampon exclue entre mine et couronne
MAX_GAP_DAYS = 25  # paires consécutives seulement (12 j ; 6 j quand S1A et S1B volaient ensemble)
MIN_GAP_DAYS = 1  # en dessous, ce n'est pas un nouveau passage mais une autre trame du même
SAME_PASS_S = 600  # trames d'un même passage : quelques secondes d'écart le long de la trace
MIN_MINE_PX = 50
BLOCK = 5  # pixels de 20 m → blocs de 100 m (écrase le chatoiement)
THRESHOLDS_DB = (2.0, 3.0)
SOURCE = "Copernicus Sentinel-1 RTC via Microsoft Planetary Computer (CC BY 4.0)"


def masks(site, grid: geo.Grid) -> tuple[np.ndarray, np.ndarray, str]:
    """Masque mine (contours Maus rasterisés) et masque couronne, sur la grille du site."""
    polys = catalog.mine_polygons(site.lat, site.lon, site.radius_m)
    shape = (grid.n, grid.n)
    if len(polys):
        g = polys.to_crs(grid.crs).geometry
        mine = rasterize(((geom, 1) for geom in g), out_shape=shape, transform=grid.transform, fill=0,
                         dtype="uint8").astype(bool)
        near = rasterize(((geom.buffer(GAP_M), 1) for geom in g), out_shape=shape,
                         transform=grid.transform, fill=0, dtype="uint8").astype(bool)
        if mine.sum() >= MIN_MINE_PX:
            return mine, ~near, "contours Maus 2022"
    # pas de contour : le carré d'analyse du site = mine, la bande extérieure = couronne
    yy, xx = np.mgrid[:grid.n, :grid.n]
    c = (grid.n - 1) / 2
    r_in = site.radius_m / grid.res
    d = np.maximum(np.abs(yy - c), np.abs(xx - c))
    return d <= r_in, d > r_in + GAP_M / grid.res, "carré du site"


def acquisitions(items) -> list[list]:
    """Regroupe les trames d'un même passage (même orbite, quelques secondes d'écart) en une acquisition.

    Un site à cheval sur deux trames Sentinel-1 apparaît deux fois au même passage. Les traiter comme deux
    passages fabriquait une « paire » de 0 jour (le même instant comparé à lui-même sur deux couvertures
    différentes) qui, en plus, écrasait dans le cache la vraie paire du jour (même clé date + orbite).
    """
    groups: list[list] = []
    for it in sorted(items, key=lambda it: it.datetime):
        if groups and (it.datetime - groups[-1][-1].datetime).total_seconds() <= SAME_PASS_S:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def mosaic(frames: list) -> np.ndarray | None:
    """Mosaïque en puissance linéaire des trames lisibles d'un passage (moyenne là où elles se recouvrent :
    c'est la même acquisition, les recouvrements sont quasi identiques)."""
    ok = [f for f in frames if f is not None]
    if not ok:
        return None
    if len(ok) == 1:
        return ok[0]
    with warnings.catch_warnings():  # pixels vus par aucune trame : NaN attendu
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(ok), axis=0).astype("float32")


def consecutive(prev_t, t) -> bool:
    """Deux acquisitions forment une paire si elles sont des passages distincts et rapprochés."""
    return MIN_GAP_DAYS <= (t - prev_t).days <= MAX_GAP_DAYS


def clean_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Paires utilisables d'un cache radar.

    - retire les lignes date == date_prev : trame comparée à une autre trame du même passage, écrites avant
      le regroupement des trames (corrompues : 0 jour d'écart, mesure de couverture et non d'activité) ;
    - pour une même date et orbite, garde la paire au plus court écart (le vrai passage précédent ; une
      paire plus longue vient d'un calcul où un passage intermédiaire était illisible).
    Les fichiers de cache ne sont pas réécrits ici : le nettoyage s'applique à chaque lecture.
    """
    if df.empty:
        return df
    d = df[pd.to_datetime(df["date"]) != pd.to_datetime(df["date_prev"])]
    d = d.assign(_t=pd.to_datetime(d["date"]), _tp=pd.to_datetime(d["date_prev"]))
    d = d.sort_values(["orbit", "_t", "_tp"]).drop_duplicates(["orbit", "_t"], keep="last")
    return d.drop(columns=["_t", "_tp"]).sort_values("date").reset_index(drop=True)


def merge_pairs(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Cache + nouveau calcul : une paire = (date_prev, date, orbite) ; le calcul le plus récent gagne."""
    df = pd.concat([old, new]) if len(old) else new
    df = df.drop_duplicates(["date_prev", "date", "orbit"], keep="last")
    return clean_pairs(df)


def _blocks(a: np.ndarray, k: int = BLOCK) -> np.ndarray:
    n = (a.shape[0] // k) * k
    with np.errstate(invalid="ignore"):
        return np.nanmean(a[:n, :n].reshape(n // k, k, n // k, k), axis=(1, 3))


def _db(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10 * np.log10(a)
    out[~np.isfinite(out) | (a <= 0)] = np.nan
    return out


class Radar(Module):
    id = "radar"
    title = "Activité radar (Sentinel-1)"
    description = "Changement de la surface d'un passage radar au suivant, mine moins environnement."
    entity_kind = "site"
    signals = {
        "activity_db": "Changement radar dans la mine moins dans son environnement (dB, médiane du mois)",
    }

    def __init__(self, workers: int = 6):
        self.workers = workers

    def _items(self, site, start: date, end: date):
        for attempt in range(5):
            try:
                found = list(geo.catalog().search(
                    collections=[COLLECTION],
                    intersects={"type": "Point", "coordinates": [site.lon, site.lat]},
                    datetime=f"{start.isoformat()}/{end.isoformat()}").items())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log.warning("%s : recherche STAC interrompue (%s)", site.id, e)
                time.sleep(5 * (attempt + 1))
        if not found:
            return [], None
        # une seule orbite (la plus fréquente) : même angle de vue d'un passage à l'autre
        orbit = Counter(it.properties.get("sat:relative_orbit") for it in found).most_common(1)[0][0]
        items = sorted((it for it in found if it.properties.get("sat:relative_orbit") == orbit),
                       key=lambda it: it.datetime)
        return items, orbit

    def pairs_path(self, site_id: str):
        return DATA_DIR / "cache" / self.id / f"{site_id}.parquet"

    def load_pairs(self, site_id: str) -> pd.DataFrame:
        """Paires en cache, nettoyées (voir clean_pairs) : à utiliser pour toute lecture du cache."""
        path = self.pairs_path(site_id)
        return clean_pairs(pd.read_parquet(path)) if path.exists() else pd.DataFrame()

    def pairs(self, site_id: str, start: date, end: date) -> pd.DataFrame:
        """Une ligne par paire de passages consécutifs : changement dans la mine et dans la couronne."""
        site = entities.load().site(site_id)
        grid = geo.site_grid(site.lon, site.lat, site.radius_m + RING_M, RES)
        mine, ring, mask_src = masks(site, grid)
        items, orbit = self._items(site, start - timedelta(days=MAX_GAP_DAYS), end)
        groups = acquisitions(items)

        mine_b, ring_b = _blocks(mine.astype(float)) > 0.5, _blocks(ring.astype(float)) > 0.5

        def frame_lin(it):
            try:
                lin = geo.read(it.assets["vv"].href, grid, Resampling.average)[0].astype("float32")
                lin[lin <= 0] = np.nan
                return lin
            except Exception as e:  # trame illisible : les autres trames du passage peuvent suffire
                log.warning("%s %s : %s", site_id, it.id, e)
                return None

        def read(group):
            lin = mosaic([frame_lin(it) for it in group])
            if lin is None:  # passage entièrement illisible : on le saute
                return None
            return _db(lin), _db(_blocks(lin))  # 20 m et blocs de 100 m (moyenne en puissance)

        rows, prev, prev_t = [], None, None
        with ThreadPoolExecutor(self.workers) as ex:
            for group, img in zip(groups, ex.map(read, groups), strict=True):
                t = group[0].datetime
                if img is not None and prev is not None and consecutive(prev_t, t):
                    d = np.abs(img[0] - prev[0])
                    db = np.abs(img[1] - prev[1])
                    vm, vr = d[mine], d[ring]
                    bm, br = db[mine_b], db[ring_b]
                    if np.isfinite(vm).sum() >= MIN_MINE_PX and np.isfinite(vr).sum() >= MIN_MINE_PX \
                            and np.isfinite(bm).sum() >= 3:
                        row = {"date": t.date(), "date_prev": prev_t.date(), "orbit": orbit,
                               "mine_db": float(np.nanmean(vm)), "ring_db": float(np.nanmean(vr)),
                               "mine_px": int(np.isfinite(vm).sum()), "masque": mask_src}
                        for thr in THRESHOLDS_DB:
                            row[f"mine_frac{thr:g}"] = float(np.nanmean(bm[np.isfinite(bm)] > thr))
                            row[f"ring_frac{thr:g}"] = float(np.nanmean(br[np.isfinite(br)] > thr))
                        rows.append(row)
                if img is not None:
                    prev, prev_t = img, t
        df = pd.DataFrame(rows)
        if len(df):
            df["activity_db"] = df["mine_db"] - df["ring_db"]
            for thr in THRESHOLDS_DB:
                df[f"excess{thr:g}"] = df[f"mine_frac{thr:g}"] - df[f"ring_frac{thr:g}"]
            path = self.pairs_path(site_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            df = merge_pairs(old, df)
            df.to_parquet(path, index=False)
        return df

    def run(self, entity_id: str, start: date, end: date) -> pd.DataFrame:
        p = self.pairs(entity_id, start, end)
        if p.empty:
            return frame([])
        dt = pd.to_datetime(p["date"])
        p = p[(dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))]
        p = p.assign(month=pd.to_datetime(p["date"]).dt.to_period("M").dt.to_timestamp())
        rows = [{"entity": entity_id, "date": m, "signal": "activity_db",
                 "value": round(float(g["activity_db"].median()), 4), "unit": "dB",
                 "quality": round(min(len(g) / 2, 1.0), 2), "source": SOURCE}
                for m, g in p.groupby("month")]
        return frame(rows)
