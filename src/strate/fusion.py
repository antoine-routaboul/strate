"""Moteur de convergence du dossier Mine.

Pour chaque signal d'un site : écart du mois à son propre historique (score z robuste,
médiane et MAD) :
  - signaux satellite : comparés aux MÊMES SAISONS des 3 années précédentes (mois ±1,
    au moins 3 observations) — sinon la fonte des neiges ressemble à une ouverture de mine ;
  - signaux d'attention : comparés aux 24 mois précédents (au moins 6 observations).
Pour chaque famille indépendante (satellite, attention…) on garde le signal le plus anormal.
Niveau du mois :
  0 = calme · 1 = anomalie sur une seule famille · 2+ = plusieurs familles s'écartent en même temps
Une alerte n'est jamais affirmée sans ses sources : chaque ligne liste les signaux qui la portent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strate import entities
from strate.core import analysed
from strate.modules import FAMILY, LOG_SIGNALS, REGISTRY, SEASONAL_FAMILIES

WINDOW = 24
MIN_HISTORY = 6
SEASON_YEARS = 3
SEASON_MIN = 3
Z_ALERT = 2.0

LABELS = {
    "bare_km2": "surface nue",
    "water_km2": "surface en eau",
    "change_km2": "surface modifiée sur un an",
    "new_bare_km2": "surface nouvellement défrichée",
    "site_views": "vues Wikipedia de la mine",
    "owner_views": "vues Wikipedia du propriétaire",
    "activity_db": "activité radar (expérimental)",
}


def signals(site_id: str) -> pd.DataFrame:
    frames = [m.load(site_id) for m in REGISTRY.values() if m.entity_kind == "site"]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["entity", "date", "signal", "value", "unit", "quality", "source"])
    return pd.concat(frames, ignore_index=True)


def robust_z(s: pd.Series) -> pd.Series:
    """z du point t contre les WINDOW observations précédentes (pas le point lui-même)."""
    vals = s.to_numpy(dtype=float)
    z = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        hist = vals[max(0, i - WINDOW):i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < MIN_HISTORY or not np.isfinite(vals[i]):
            continue
        z[i] = _z(vals[i], hist)
    return pd.Series(z, index=s.index)


# Effet minimal (même unité que le signal) : en dessous, un écart n'est jamais une alerte, aussi
# rare soit-il. Sans ça, 3 pixels d'eau en plein désert d'Atacama donnent z = 10.
MIN_EFFECT_KM2 = 0.25


def _z(x: float, hist: np.ndarray, floor: float = 1e-6) -> float:
    med = np.median(hist)
    mad = 1.4826 * np.median(np.abs(hist - med))
    # le dénominateur ne descend jamais sous floor/Z_ALERT : un écart < floor donne |z| < Z_ALERT
    return (x - med) / max(mad, 0.05 * abs(med), floor / Z_ALERT, 1e-6)


def seasonal_z(s: pd.Series, floor: float = MIN_EFFECT_KM2) -> pd.Series:
    """z du mois t contre les mêmes saisons (mois ±1) des SEASON_YEARS années précédentes."""
    s = s.dropna()
    vals = s.to_numpy(dtype=float)
    # année et mois en entiers une fois pour toutes : les relire sur l'index à chaque tour coûtait
    # l'essentiel du temps du radar (des centaines de milliers d'opérations pandas)
    years = np.asarray(s.index.year, dtype=int)
    months = np.asarray(s.index.month, dtype=int)
    z = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        ago = (years[i] - years) * 12 + (months[i] - months)  # mois écoulés
        moy = np.abs(((months - months[i]) + 6) % 12 - 6)  # distance circulaire entre mois
        mask = (ago >= 11) & (ago <= 12 * SEASON_YEARS + 1) & (moy <= 1)
        hist = vals[mask]
        if len(hist) >= SEASON_MIN:
            z[i] = _z(vals[i], hist, floor)
    return pd.Series(z, index=s.index)


def persistent(z: pd.Series) -> pd.Series:
    """z gardé seulement s'il dépasse le seuil deux observations de suite, dans le même sens.

    Un mois isolé (pluie, éclairage, nuage mal masqué) ne fait plus d'alerte ; sur les 7 mines
    testées, cette règle retire environ la moitié des alertes satellite.
    """
    hit = z.abs() >= Z_ALERT
    keep = hit & hit.shift(1, fill_value=False) & (np.sign(z) == np.sign(z.shift(1)))
    if isinstance(z.index, pd.DatetimeIndex):
        # « de suite » = mois voisins : une anomalie d'octobre et une d'avril (hiver sans image entre
        # les deux) ne se confirment pas l'une l'autre
        idx = z.index.to_series()
        gap = (idx.dt.year * 12 + idx.dt.month).diff()
        keep &= (gap <= 2).to_numpy()
    return z.where(keep, 0.0).where(z.notna())


def dossier(site_id: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table mensuelle : valeur et z de chaque signal, niveau de convergence, explication.

    `df` : signaux déjà lus (le radar les lit d'abord pour écarter les sites sans mois récent)."""
    df = signals(site_id) if df is None else df
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="date", columns="signal", values="value").sort_index()
    out = pd.DataFrame(index=wide.index)
    for sig in wide.columns:
        x = wide[sig]
        if sig in LOG_SIGNALS:
            x = np.log1p(x)
        out[sig] = wide[sig]
        if FAMILY.get(sig) in SEASONAL_FAMILIES:
            z = seasonal_z(x.dropna())
            out[f"z_{sig}"] = z.reindex(wide.index)
            out[f"a_{sig}"] = persistent(z).reindex(wide.index)
        else:
            z = robust_z(x.dropna())
            out[f"z_{sig}"] = z.reindex(wide.index)
            out[f"a_{sig}"] = out[f"z_{sig}"]

    fam_z = {}
    for fam in sorted(set(FAMILY.values())):
        cols = [f"a_{s}" for s, f in FAMILY.items() if f == fam and f"a_{s}" in out]
        if not cols:
            continue
        zz = out[cols]
        idx = zz.abs().fillna(-1).to_numpy().argmax(axis=1)
        fam_z[fam] = pd.Series(zz.to_numpy()[np.arange(len(zz)), idx], index=out.index)
        out[f"family_{fam}"] = fam_z[fam]
    out["level"] = sum((fz.abs() >= Z_ALERT).astype(int) for fz in fam_z.values())

    def explain(row) -> str:
        parts = []
        for s in LABELS:
            z = row.get(f"a_{s}")
            if z is not None and np.isfinite(z) and abs(z) >= Z_ALERT:
                sens = "hausse" if z > 0 else "baisse"
                parts.append(f"{LABELS[s]} en {sens} (z={z:+.1f})")
        return " · ".join(parts)

    out["explication"] = out.apply(explain, axis=1)
    site = entities.load().site(site_id)
    ev = {e.date: e.event for e in site.known_events}
    out["evenement_connu"] = [ev.get(f"{d:%Y-%m}", "") for d in out.index]
    out.insert(0, "site", site_id)
    return out.reset_index()


def event_check(site_id: str, tolerance_months: int = 3) -> list[dict]:
    """Mini-backtest : chaque événement connu est-il accompagné d'une anomalie à ±N mois ?"""
    site = entities.load().site(site_id)
    d = dossier(site_id)
    res = []
    for e in site.known_events:
        if d.empty:  # site pas encore analysé (installation neuve) : rien à confronter, pas d'erreur
            res.append(dict(site=site_id, date=e.date, evenement=e.event, detecte=False,
                            mois_en_alerte=[], details=[]))
            continue
        t = pd.Timestamp(f"{e.date}-01")
        near = d[(d["date"] - t).abs() <= pd.Timedelta(days=31 * tolerance_months)]
        hits = near[near["level"] >= 1]
        res.append(dict(
            site=site_id, date=e.date, evenement=e.event, detecte=bool(len(hits)),
            mois_en_alerte=[f"{x:%Y-%m}" for x in hits["date"]],
            details=[x for x in hits["explication"] if x],
        ))
    return res


def radar(month: str | None = None) -> pd.DataFrame:
    """Tous les sites, dernier mois disponible (ou le mois demandé), triés par niveau."""
    rows = []
    g = entities.load()
    ids = list(dict.fromkeys([*g.sites, *analysed("optical")]))
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=183)
    for sid in ids:
        try:
            site = g.site(sid)
        except KeyError:
            continue
        sig = signals(sid)
        # sans mois récent, le site sera écarté plus bas : inutile de calculer tout son dossier (les
        # ~150 mines de calibration 2016–2019 coûtaient l'essentiel du temps du radar)
        if sig.empty or (not month and pd.to_datetime(sig["date"]).max() < cutoff):
            continue
        d = dossier(sid, sig)
        if d.empty:
            continue
        if month:
            d = d[d["date"] == pd.Timestamp(f"{month}-01")]
        if d.empty:
            continue
        last = d.iloc[-1]
        if not month and last["date"] < cutoff:
            continue  # pas de données récentes (ex. mines analysées seulement pour la calibration 2016–2019)
        rows.append(dict(site=sid, nom=site.name, pays=site.country, date=last["date"],
                         niveau=int(last["level"]), explication=last["explication"],
                         suivi=sid in g.sites))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["niveau", "site"], ascending=[False, True])
