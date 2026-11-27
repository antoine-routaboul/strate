"""Registre des modules. Ajouter un module = l'importer ici, tout le reste suit (API, widgets)."""

from strate.core import Module
from strate.modules.attention import Attention
from strate.modules.optical import Optical
from strate.modules.radar import Radar

REGISTRY: dict[str, Module] = {m.id: m for m in (Optical(), Attention(), Radar())}

# Famille de chaque signal : la convergence exige des familles INDÉPENDANTES.
FAMILY = {
    "bare_km2": "satellite",
    "water_km2": "satellite",
    "change_km2": "satellite",
    "new_bare_km2": "satellite",
    "site_views": "attention",
    "owner_views": "attention",
    "activity_db": "radar",  # expérimental : signal temporel non démontré (voir docs/calibration.md)
}
SEASONAL_FAMILIES = {"satellite", "radar"}  # comparés aux mêmes saisons (neige, pluie)
LOG_SIGNALS = {"site_views", "owner_views"}  # comptages : on travaille en log
