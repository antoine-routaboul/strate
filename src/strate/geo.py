"""Outils satellite partagés : grille d'un site, lecture de COG, catalogue STAC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Lecture HTTP rapide des Cloud-Optimized GeoTIFF : on ne lit que la fenêtre utile.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}
os.environ.update({k: v for k, v in GDAL_ENV.items() if k not in os.environ})


@dataclass(frozen=True)
class Grid:
    """Carré centré sur un site, dans la projection UTM du site."""

    crs: str
    x0: float  # bord ouest
    y1: float  # bord nord
    res: float
    n: int

    @property
    def transform(self):
        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def pixel_km2(self) -> float:
        return self.res * self.res / 1e6

    @property
    def area_km2(self) -> float:
        return self.n * self.n * self.pixel_km2


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def site_grid(lon: float, lat: float, radius_m: float, res: float) -> Grid:
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    n = int(round(2 * radius_m / res))
    return Grid(crs=f"EPSG:{epsg}", x0=x - radius_m, y1=y + radius_m, res=res, n=n)


def read(href: str, grid: Grid, resampling: Resampling, bands: int | list[int] = 1) -> np.ndarray:
    """Lit uniquement la fenêtre du site, reprojetée sur la grille du site.

    Retourne TOUJOURS un tableau (bandes, n, n), même pour une seule bande.
    """
    if isinstance(bands, int):
        bands = [bands]
    with rasterio.Env(**GDAL_ENV), rasterio.open(pc.sign(href)) as src, WarpedVRT(
        src,
        crs=grid.crs,
        transform=grid.transform,
        width=grid.n,
        height=grid.n,
        resampling=resampling,
        src_nodata=0,  # 0 = hors image chez Sentinel-2 (DN et SCL)
        nodata=0,
    ) as vrt:
        return vrt.read(bands)


@cache
def catalog():
    from pystac_client import Client

    return Client.open(STAC_URL)
