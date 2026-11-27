"""Prévisions de production par mine, qui s'évaluent elles-mêmes (docs/previsions-societes.md, § 1).

Le passé sert à prédire le futur, et l'outil rejoue l'histoire pour se noter : pour chaque mine et chaque
année cible t (2005–2021), on ne s'autorise que les données ≤ t − h (h = horizon). Découpage strict :
réglages (φ, λ, quantiles des intervalles) choisis sur les cibles ≤ 2014, évaluation sur 2015–2021,
jamais vue pendant le réglage.

Modèles, du plus simple au plus riche (tout en log de la production) :
  naif             dernière valeur connue (production de l'an t − h) ;
  tendance         + pente MCO des 5 dernières années × (φ + φ² + … + φ^h), un seul φ réglé ;
  tendance+taille  idem, rappelé vers l'ordre de grandeur donné par l'emprise Maus (modèle
                   log10 t = a + b·log10 km², par type « minerai » / « charbon ») : part du chemin
                   parcourue λ_h = 1 − (1 − κ)^h, un seul κ (rappel par an) réglé.

Un seul modèle est « retenu » pour tout le chemin de prévision (pas de saut d'un modèle à l'autre entre
deux horizons) : celui dont le skill moyen sur les horizons mesurés est le meilleur hors échantillon, à
condition que la borne basse de son IC 90 % (bootstrap sur les mines) soit > 0. Sinon le naïf reste
retenu. Aucune IA : règles et moindres carrés, entièrement reproductibles.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from strate.core import DATA_DIR

log = logging.getLogger(__name__)

OUT = DATA_DIR / "forecast"
LEDGER_NAME, BACKTEST_NAME = "ledger.parquet", "backtest.json"
MSG_BACKTEST = "Rétro-test pas encore lancé : uv run strate forecast backtest"
LEDGER_COLS = ["made_at", "site_id", "serie", "target_year", "model", "value", "lo80", "hi80", "lo90",
               "hi90"]

FIRST_TARGET, LAST_TARGET = 2005, 2021
TUNE_END = 2014  # réglages ≤ 2014, évaluation 2015–2021
H_MAX = 10  # horizon le plus long rejoué (2005–2021 et des données depuis 2000 le permettent)
HORIZONS = tuple(range(1, H_MAX + 1))  # tous rejoués hors échantillon ; seuls les mesurés sont publiés
WINDOW, MIN_WINDOW = 5, 3  # tendance : 5 dernières années, au moins 3 observées
PHI_GRID = tuple(round(float(x), 2) for x in np.arange(0.0, 1.0001, 0.05))
KAPPA_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
MODELS = ("naif", "tendance", "tendance+taille")
EMPRISE_TYPE = {"minerai extrait": "minerai", "charbon": "charbon"}  # séries où le modèle d'emprise vaut
MIN_ANCHOR = 20  # mines minimales pour ré-estimer le modèle d'emprise à une origine
MIN_N = 8  # en dessous, une corrélation ne veut rien dire
MIN_ERR = 30  # erreurs minimales pour un quantile empirique
N_BOOT = 1000
LN10 = math.log(10.0)
Q = (0.05, 0.10, 0.90, 0.95)  # quantiles des erreurs : intervalles 90 % et 80 %

METHODE = (
    "Rétro-test walk-forward : pour chaque mine et chaque année cible t de 2005 à 2021, seules les "
    "données ≤ t − h servent (h = 1 à 10 ans). Réglages (amortissement φ, rappel λ vers l'emprise, "
    "quantiles des intervalles) choisis sur les cibles ≤ 2014 ; mesures sur 2015–2021, jamais vues. "
    "r (Pearson) et ρ (Spearman) comparent variation prévue et variation réelle du log de la production "
    "depuis la dernière valeur connue (le naïf, qui ne prévoit aucune variation, n'a pas de r ; un "
    "modèle réglé sur φ = 0 non plus) ; "
    "*_niveau : mêmes mesures sur le log des niveaux. IC 90 % par bootstrap en tirant des MINES (leurs "
    "années ne sont pas indépendantes), appariés entre modèles. skill = 1 − MSE(modèle) / MSE(naïf) en "
    "log. facteur_median : erreur multiplicative médiane. bon_sens : part des variations réelles dont le "
    "sens est bien prévu (prévision immobile = ½). Couvertures : part des valeurs réelles 2015–2021 dans "
    "les intervalles 80 % / 90 % bâtis avec les erreurs ≤ 2014 seulement. Un seul φ et un seul κ, réglés "
    "sur la moyenne des MSE relatives des horizons 1 à 10 (chaque horizon pèse autant). Un horizon est "
    "« mesuré » s'il a au moins 8 cas de test pour chaque modèle et des quantiles d'erreur ≤ 2014 non "
    "extrapolés (au moins 30 erreurs) ; les prévisions d'avenir s'arrêtent au plus long horizon mesuré "
    "sans trou depuis h = 1, et leurs intervalles sont ceux dont la couverture 2015–2021 est publiée "
    "(quantiles des erreurs ≤ 2014). Un seul modèle retenu pour tout le chemin (aucun saut entre "
    "horizons) : le meilleur skill moyen sur les horizons mesurés (moyenne des skills par horizon, IC 90 % "
    "par bootstrap sur les mines), s'il a une borne basse > 0 ; sinon le naïf. Le modèle d'emprise "
    "est ré-estimé à chaque origine sur la coupe des productions de l'année d'origine (≤ t − h) ; les "
    "prévisions d'avenir utilisent ce même ajustement, à l'année de la dernière valeur connue de la mine.")


# =============================================================================================
# Données : série principale de chaque mine
# =============================================================================================
def _series_last(p: pd.DataFrame) -> dict[str, tuple[int, float]]:
    """{série : (dernière année, tonnes)} d'une installation — l'entrée de screener.main_production."""
    p = p[p["value_tonnes"] > 0].sort_values("year")
    last = p.groupby("serie").last()
    return {str(s): (int(r["year"]), float(r["value_tonnes"])) for s, r in last.iterrows()}


def _underground() -> set[str]:
    """Installations purement souterraines (leur emprise ne dit rien de la production) ; vide si absent."""
    try:
        import geopandas as gpd

        from strate import catalog
        from strate.calibrate import _equipment, _underground_only

        f = gpd.read_file(catalog.SRC / "jasansky2023" / "x" / "data" / "facilities.gpkg",
                          ignore_geometry=True)
        return {str(k) for k, v in _equipment(f).items() if _underground_only(v)}
    except Exception as e:  # noqa: BLE001 — donnée facultative
        log.warning("équipement des mines indisponible : %s", e)
        return set()


def production_panel(prod: pd.DataFrame | None = None, cat: pd.DataFrame | None = None,
                     underground: set[str] | None = None) -> pd.DataFrame:
    """Panel annuel : site_id · jasansky · nom · serie · emprise_type · footprint_km2 · year · value.

    Une seule série par mine, choisie comme screener.main_production (minerai extrait > minerai traité >
    charbon > métal de la commodité principale > série la plus lourde). Un id Jasansky porté par plusieurs
    sites du catalogue ne compte qu'une fois (premier site). emprise_type n'est renseigné que si la série
    est celle du modèle d'emprise et la mine n'est pas purement souterraine.
    """
    from strate import catalog
    from strate.screener import commodity_groups, main_production

    cat = catalog.load() if cat is None else cat
    prod = catalog.all_production() if prod is None else prod
    cols = ["site_id", "jasansky", "nom", "serie", "emprise_type", "footprint_km2", "year", "value"]
    if cat is None or cat.empty or "jasansky" not in cat or prod is None or prod.empty:
        return pd.DataFrame(columns=cols)
    underground = _underground() if underground is None else underground
    linked = cat[cat["jasansky"].notna()]
    linked = linked[~linked["jasansky"].duplicated()]
    groups = {k: g for k, g in prod[prod["value_tonnes"] > 0].groupby("facility_id")}
    frames = []
    for r in linked.to_dict("records"):
        p = groups.get(r["jasansky"])
        if p is None:
            continue
        serie = main_production(_series_last(p), commodity_groups(r.get("commodities")))[0]
        if serie is None:
            continue
        s = p[p["serie"] == serie].groupby("year", as_index=False)["value_tonnes"].sum()
        fp = r.get("footprint_km2")
        fp = float(fp) if fp is not None and pd.notna(fp) and float(fp) > 0 else np.nan
        kind = EMPRISE_TYPE.get(serie) if r["jasansky"] not in underground and np.isfinite(fp) else None
        frames.append(pd.DataFrame({
            "site_id": str(r["id"]), "jasansky": str(r["jasansky"]),
            "nom": r.get("name") if isinstance(r.get("name"), str) else None, "serie": serie,
            "emprise_type": kind, "footprint_km2": fp,
            "year": s["year"].astype(int).to_numpy(), "value": s["value_tonnes"].astype(float).to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


@lru_cache(maxsize=2)
def _panel_cached(stamp: tuple) -> pd.DataFrame:
    return production_panel()


def _panel() -> pd.DataFrame:
    from strate import catalog

    try:
        stamp = ((catalog.OUT / "mines.parquet").stat().st_mtime_ns,)
    except OSError:
        stamp = (0,)
    return _panel_cached(stamp)


# =============================================================================================
# Cas à prévoir et modèles
# =============================================================================================
def _slope(years: np.ndarray, lv: np.ndarray) -> float:
    if len(years) < MIN_WINDOW:
        return math.nan
    x = years - years.mean()
    return float((x * (lv - lv.mean())).sum() / (x ** 2).sum())


def _origin_state(years: np.ndarray, lv: np.ndarray, o: int) -> tuple[float, float]:
    """(log de la valeur à l'origine o, pente des 5 dernières années ≤ o) ; NaN si non calculable."""
    k = np.flatnonzero(years == o)
    if not len(k):
        return math.nan, math.nan
    w = (years >= o - WINDOW + 1) & (years <= o)
    return float(lv[k[0]]), _slope(years[w].astype(float), lv[w])


def make_cases(panel: pd.DataFrame, first: int = FIRST_TARGET, last: int = LAST_TARGET,
               h_max: int = H_MAX) -> pd.DataFrame:
    """Cas du rétro-test : (site, origine o, horizon h, cible t = o + h) avec y connu en t.

    Ne lit que les années ≤ o pour l'état à l'origine (valeur, pente) : aucune donnée de t n'y entre.
    Colonnes : site_id · emprise_type · lfp (log10 emprise) · origin · h · target · l0 · slope · y.
    """
    out = []
    for sid, g in panel.groupby("site_id", sort=True):
        g = g[g["value"] > 0].sort_values("year")
        years, lv = g["year"].to_numpy(int), np.log(g["value"].to_numpy(float))
        val = dict(zip(years.tolist(), lv.tolist(), strict=True))
        kind, fp = g["emprise_type"].iat[0], g["footprint_km2"].iat[0]
        lfp = math.log10(fp) if isinstance(kind, str) and np.isfinite(fp) else math.nan
        for o in years.tolist():
            l0, sl = _origin_state(years, lv, o)
            if not np.isfinite(sl):
                continue
            for h in range(1, h_max + 1):
                t = o + h
                if first <= t <= last and t in val:
                    out.append((sid, kind if isinstance(kind, str) else None, lfp, o, h, t, l0, sl, val[t]))
    cols = ["site_id", "emprise_type", "lfp", "origin", "h", "target", "l0", "slope", "y"]
    return pd.DataFrame(out, columns=cols)


def anchor_coefs(panel: pd.DataFrame, min_n: int = MIN_ANCHOR) -> dict[tuple[str, int], tuple]:
    """Modèle d'emprise ré-estimé pour chaque (type, année o) sur la coupe des productions de l'année o :
    log10 t = a + b·log10 km². Retourne {(type, o): (a, b, n, lfp_min, lfp_max)} — données ≤ o seulement."""
    d = panel[panel["emprise_type"].notna() & (panel["value"] > 0)]
    out = {}
    for (kind, o), g in d.groupby(["emprise_type", "year"]):
        if len(g) < min_n:
            continue
        x, y = np.log10(g["footprint_km2"].to_numpy(float)), np.log10(g["value"].to_numpy(float))
        if x.std() < 1e-9:
            continue
        b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        out[(str(kind), int(o))] = (float(y.mean() - b * x.mean()), b, int(len(g)), float(x.min()),
                                    float(x.max()))
    return out


def _anchor_ln(kind, lfp: float, coef: tuple | None) -> float:
    """Ordre de grandeur (log naturel des tonnes) donné par l'emprise ; NaN hors du domaine ajusté."""
    if not isinstance(kind, str) or coef is None or not np.isfinite(lfp):
        return math.nan
    a, b, _, lo, hi = coef
    return LN10 * (a + b * lfp) if lo <= lfp <= hi else math.nan


def add_anchor(cases: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    cases = cases.copy()
    cases["anchor"] = [_anchor_ln(k, f, coefs.get((k, int(o))) if isinstance(k, str) else None)
                       for k, f, o in zip(cases["emprise_type"], cases["lfp"], cases["origin"], strict=True)]
    return cases


def damping(phi: float, h) -> np.ndarray:
    """φ + φ² + … + φ^h (tendance amortie)."""
    h = np.asarray(h, float)
    if phi == 1.0:
        return h
    return phi * (1 - phi ** h) / (1 - phi)


def pull(kappa: float, h) -> np.ndarray:
    """Part du chemin vers l'emprise parcourue en h ans : 1 − (1 − κ)^h."""
    return 1.0 - (1.0 - kappa) ** np.asarray(h, float)


def predict(cases: pd.DataFrame, phi: float, kappa: float) -> dict[str, np.ndarray]:
    """Prévisions (log) des trois modèles pour un φ (amortissement) et un κ (rappel par an) uniques."""
    h = cases["h"].to_numpy(float)
    l0, sl = cases["l0"].to_numpy(float), cases["slope"].to_numpy(float)
    trend = l0 + sl * damping(phi, h)
    anc = cases["anchor"].to_numpy(float) if "anchor" in cases else np.full(len(cases), np.nan)
    size = np.where(np.isfinite(anc), trend + pull(kappa, h) * (anc - trend), trend)
    return {"naif": l0, "tendance": trend, "tendance+taille": size}


def _rel_mse(y: np.ndarray, p: np.ndarray, base: np.ndarray, h: np.ndarray) -> float:
    """Moyenne, sur les horizons, de MSE(prévision) / MSE(naïf) : chaque horizon pèse autant."""
    out = []
    for k in np.unique(h):
        m = h == k
        b = np.mean((y[m] - base[m]) ** 2)
        if b > 0:
            out.append(np.mean((y[m] - p[m]) ** 2) / b)
    return float(np.mean(out)) if out else math.inf


def tune(cases: pd.DataFrame, horizons=HORIZONS) -> tuple[float, float]:
    """φ puis κ minimisant la MSE relative (log) sur les cas fournis (période de réglage, cibles ≤ 2014)."""
    c = cases[cases["h"].isin(list(horizons))]
    if c.empty:
        return 0.0, 0.0
    y, l0, sl = c["y"].to_numpy(float), c["l0"].to_numpy(float), c["slope"].to_numpy(float)
    h = c["h"].to_numpy(float)
    phi = min(PHI_GRID, key=lambda p: (_rel_mse(y, l0 + sl * damping(p, h), l0, h), p))
    anc = c["anchor"].to_numpy(float) if "anchor" in c else np.full(len(c), np.nan)
    m = np.isfinite(anc)
    if m.sum() < MIN_ERR:
        return float(phi), 0.0
    y, l0, h, anc = y[m], l0[m], h[m], anc[m]
    tr = l0 + sl[m] * damping(phi, h)
    kappa = min(KAPPA_GRID, key=lambda k: (_rel_mse(y, tr + pull(k, h) * (anc - tr), l0, h), k))
    return float(phi), float(kappa)


def error_quantiles(err: pd.DataFrame, h_max: int = H_MAX) -> dict[str, dict[int, dict]]:
    """Quantiles empiriques des erreurs log (réel − prévu), par modèle et horizon.

    Un horizon avec moins de MIN_ERR erreurs reprend le dernier horizon fourni, élargi en √(h / h₀).
    Les quantiles sont rendus monotones : l'intervalle ne rétrécit jamais quand l'horizon s'allonge."""
    out: dict[str, dict[int, dict]] = {}
    for m in MODELS:
        e = err[err["model"] == m]
        qs: dict[int, dict] = {}
        for h in range(1, h_max + 1):
            v = e.loc[e["h"] == h, "err"].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) >= MIN_ERR:
                qs[h] = {"q": [float(x) for x in np.quantile(v, Q)], "n": int(len(v)), "extrapole": False}
            elif qs:
                h0 = max(k for k, x in qs.items() if not x["extrapole"])
                f = math.sqrt(h / h0)
                qs[h] = {"q": [x * f for x in qs[h0]["q"]], "n": qs[h0]["n"], "extrapole": True}
        prev = None
        for h in sorted(qs):  # un horizon plus lointain n'est jamais plus sûr qu'un plus proche
            if prev is not None:
                q = qs[h]["q"]
                qs[h]["q"] = [min(q[0], prev[0]), min(q[1], prev[1]), max(q[2], prev[2]), max(q[3], prev[3])]
            prev = qs[h]["q"]
        out[m] = qs
    return out


def _q(quant: dict, model: str, h: int) -> list[float] | None:
    qs = quant.get(model) or {}
    x = qs.get(h, qs.get(str(h)))
    if x is None and qs:
        keys = sorted(int(k) for k in qs)
        k = keys[-1]
        if h <= k:
            return None
        base = qs.get(k, qs.get(str(k)))
        return [v * math.sqrt(h / k) for v in base["q"]]
    return None if x is None else x["q"]


# =============================================================================================
# Mesures avec IC (bootstrap sur les mines, apparié entre modèles)
# =============================================================================================
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_N:
        return math.nan
    sx, sy = x.std(), y.std()
    if sx <= 1e-12 or sy <= 1e-12:
        return math.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_N or x.std() <= 1e-12 or y.std() <= 1e-12:
        return math.nan
    return _pearson(rankdata(x), rankdata(y))


def _point(y, p, l0, pn, lo80, hi80, lo90, hi90) -> dict[str, float]:
    dp, dy = p - l0, y - l0
    e, e0 = (p - y) ** 2, (pn - y) ** 2
    mv = np.abs(dy) > 1e-12
    sens = np.where(np.abs(dp[mv]) <= 1e-12, 0.5, np.sign(dp[mv]) == np.sign(dy[mv]))
    return {"r": _pearson(dp, dy), "rho": _spearman(dp, dy),
            "r_niveau": _pearson(p, y), "rho_niveau": _spearman(p, y),
            "skill": float(1 - e.mean() / e0.mean()) if e0.mean() > 0 else math.nan,
            "facteur_median": float(np.exp(np.median(np.abs(p - y)))) if len(y) else math.nan,
            "bon_sens": float(sens.mean()) if mv.sum() else math.nan,
            "couverture80": float(np.mean((lo80 <= y) & (y <= hi80))) if len(y) else math.nan,
            "couverture90": float(np.mean((lo90 <= y) & (y <= hi90))) if len(y) else math.nan}


CI_KEYS = ("r", "rho", "r_niveau", "rho_niveau", "skill")


def metrics(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mesures d'un (modèle, horizon) : colonnes site_id · y · p · l0 · p_naif · lo80 · hi80 · lo90 · hi90
    (log). IC 90 % percentile en rééchantillonnant les mines ; même graine = tirages appariés."""
    d = d.sort_values(["site_id", "target"], kind="stable")
    a = [d[c].to_numpy(float) for c in ("y", "p", "l0", "p_naif", "lo80", "hi80", "lo90", "hi90")]
    out = {"n_mines": int(d["site_id"].nunique()), "n_points": int(len(d))}
    pt = _point(*a)
    out |= pt
    codes, uniq = pd.factorize(d["site_id"], sort=True)
    groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    draws = {k: [] for k in CI_KEYS}
    if len(groups) >= 2 and len(d) >= MIN_N:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            b = _point(*[x[idx] for x in a])
            for k in CI_KEYS:
                draws[k].append(b[k])
    for k in CI_KEYS:
        v = np.asarray(draws[k], float)
        v = v[np.isfinite(v)]
        ok = len(v) >= max(20, n_boot // 5) and np.isfinite(pt[k])
        out[f"{k}_lo"] = float(np.percentile(v, 5)) if ok else math.nan
        out[f"{k}_hi"] = float(np.percentile(v, 95)) if ok else math.nan
    return out


def path_skill(cases: pd.DataFrame, preds: dict[str, np.ndarray], horizons, n_boot: int = N_BOOT,
               seed: int = 0) -> dict[str, dict]:
    """Skill moyen sur un chemin d'horizons : moyenne, horizon par horizon, de 1 − MSE(modèle) / MSE(naïf).

    IC 90 % percentile en rééchantillonnant les MINES (mêmes tirages pour tous les modèles). Retourne
    {modèle : {skill_moyen, skill_moyen_lo, skill_moyen_hi, n_mines, n_points, horizons}}."""
    hs = [h for h in horizons if (cases["h"] == h).any()]
    c = cases.reset_index(drop=True)
    keep = c["h"].isin(hs).to_numpy()
    out = {m: {"skill_moyen": math.nan, "skill_moyen_lo": math.nan, "skill_moyen_hi": math.nan,
               "n_mines": 0, "n_points": 0, "horizons": hs} for m in MODELS}
    if not hs:
        return out
    codes, uniq = pd.factorize(c.loc[keep, "site_id"], sort=True)
    hi = np.searchsorted(hs, c.loc[keep, "h"].to_numpy(int))
    y = c.loc[keep, "y"].to_numpy(float)
    sq = {m: np.zeros((len(uniq), len(hs))) for m in MODELS}  # erreurs² sommées par (mine, horizon)
    for m in MODELS:
        np.add.at(sq[m], (codes, hi), (y - preds[m][keep]) ** 2)

    def avg(w: np.ndarray) -> dict[str, float]:
        base = w @ sq["naif"]
        ok = base > 0
        return {m: float(np.mean(1 - (w @ sq[m])[ok] / base[ok])) if ok.any() else math.nan for m in MODELS}

    pt = avg(np.ones(len(uniq)))
    rng = np.random.default_rng(seed)
    draws = [avg(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float))
             for _ in range(n_boot if len(uniq) >= 2 else 0)]
    for m in MODELS:
        v = np.asarray([d[m] for d in draws], float)
        v = v[np.isfinite(v)]
        ok = len(v) >= max(20, n_boot // 5) and np.isfinite(pt[m])
        out[m] |= {"skill_moyen": pt[m], "n_mines": int(len(uniq)), "n_points": int(keep.sum()),
                   "skill_moyen_lo": float(np.percentile(v, 5)) if ok else math.nan,
                   "skill_moyen_hi": float(np.percentile(v, 95)) if ok else math.nan}
    return out


def choose(path: dict[str, dict]) -> str:
    """Modèle retenu pour tout le chemin : meilleur skill moyen parmi ceux dont l'IC 90 % exclut 0, sinon
    le naïf."""
    ok = [(v["skill_moyen"], m) for m, v in path.items() if m != "naif" and _finite(v.get("skill_moyen_lo"))
          and v["skill_moyen_lo"] > 0]
    return max(ok)[1] if ok else "naif"


def _measured(rows: list[dict], q_tune: dict) -> list[int]:
    """Horizons mesurés hors échantillon : une ligne de test (≥ MIN_N cas) pour chaque modèle, et des
    quantiles d'erreur de réglage non extrapolés (la couverture publiée vaut pour ces intervalles)."""
    out = []
    for h in HORIZONS:
        have = {r["model"] for r in rows if r["horizon"] == h and r.get("n_points", 0) >= MIN_N}
        q_ok = all((x := (q_tune.get(m) or {}).get(h)) is not None and not x["extrapole"] for m in MODELS)
        if have >= set(MODELS) and q_ok:
            out.append(h)
    return out


def _width(quant: dict, model: str, h: int) -> float:
    q = _q(quant, model, h)
    return float(q[3] - q[0]) if q is not None else math.nan


def _h_pub(bt: dict) -> int:
    """Plus long horizon publié ; ancien format : le plus long horizon des lignes du rétro-test."""
    v = bt.get("horizon_max_publie")
    if isinstance(v, int):
        return v
    hs = [int(r["horizon"]) for r in bt.get("rows") or [] if isinstance(r.get("horizon"), int)]
    return max(hs) if hs else 0


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# =============================================================================================
# Rétro-test complet
# =============================================================================================
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _errors(cases: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"model": m, "h": cases["h"].to_numpy(int),
                                    "target": cases["target"].to_numpy(int),
                                    "err": cases["y"].to_numpy(float) - p}) for m, p in preds.items()],
                     ignore_index=True)


def _bounds(p: np.ndarray, h: np.ndarray, quant: dict, model: str) -> list[np.ndarray]:
    """Bornes log lo90, lo80, hi80, hi90 : prévision + quantiles d'erreur de son horizon."""
    out = np.full((len(p), 4), np.nan)
    for k in np.unique(h):
        q = _q(quant, model, int(k))
        if q is not None:
            m = h == k
            out[m] = p[m, None] + np.asarray(q)[None, :]
    return [out[:, 0], out[:, 1], out[:, 2], out[:, 3]]


DIAG_KEYS = ("n_mines", "n_points", "r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi", "skill", "skill_lo",
             "skill_hi", "bon_sens")


def _eval_frame(c: pd.DataFrame, p: np.ndarray, p_naif: np.ndarray, bounds: list | None) -> pd.DataFrame:
    """Cas évalués (log) pour metrics() ; bornes lo90, lo80, hi80, hi90 ou None (pas d'intervalle)."""
    lo90, lo80, hi80, hi90 = bounds if bounds is not None else [np.full(len(c), np.nan)] * 4
    return pd.DataFrame({"site_id": c["site_id"].to_numpy(), "target": c["target"].to_numpy(),
                         "y": c["y"].to_numpy(float), "p": p, "l0": c["l0"].to_numpy(float),
                         "p_naif": p_naif, "lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90})


def run_backtest(panel: pd.DataFrame, n_boot: int = N_BOOT, today: date | None = None) -> dict:
    """Rétro-test sur un panel (sans écrire) ; voir le docstring du module."""
    today = today or date.today()
    run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {"run_at": run_at, "periode": {"reglage": f"≤{TUNE_END}", "test": f"{TUNE_END + 1}–{LAST_TARGET}"},
            "methode": METHODE, "rows": [], "retenu": {str(h): "naif" for h in HORIZONS}}
    if panel is None or panel.empty:
        return base | {"statut": "absent", "raison": "aucune production réelle liée au catalogue"}
    coefs = anchor_coefs(panel)
    cases = add_anchor(make_cases(panel), coefs)
    tune_c, test_c = cases[cases["target"] <= TUNE_END], cases[cases["target"] > TUNE_END]
    if tune_c.empty or test_c.empty:
        return base | {"statut": "insuffisant", "raison": "trop peu d'années pour un rétro-test"}
    phi, kappa = tune(tune_c)
    preds_tune, preds_test = predict(tune_c, phi, kappa), predict(test_c, phi, kappa)
    q_tune = error_quantiles(_errors(tune_c, preds_tune))  # intervalles : erreurs ≤ 2014 seulement
    rows, rows_reglage, diag = [], [], []
    raw = predict(test_c, 1.0, kappa)["tendance"]  # tendance brute (φ = 1) : ce que le réglage a écarté
    for h in HORIZONS:
        for label, c, preds, dst in (("test", test_c, preds_test, rows), ("reglage", tune_c, preds_tune,
                                                                           rows_reglage)):
            m = (c["h"] == h).to_numpy()
            if m.sum() < MIN_N:
                continue
            nb = n_boot if label == "test" else max(n_boot // 5, 100)
            for model in MODELS:
                p = preds[model][m]
                d = _eval_frame(c[m], p, preds["naif"][m], _bounds(p, c["h"].to_numpy(int)[m], q_tune, model))
                dst.append({"model": model, "horizon": h, **metrics(d, nb, seed=h)})
            if label == "test":
                mt = metrics(_eval_frame(c[m], raw[m], preds["naif"][m], None), max(n_boot // 5, 100), seed=h)
                diag.append({"model": "tendance (φ = 1)", "horizon": h, **{k: mt[k] for k in DIAG_KEYS}})
    mesures = _measured(rows, q_tune)
    h_pub = next((h - 1 for h in range(1, H_MAX + 2) if h not in mesures), H_MAX)  # sans trou depuis h = 1
    chemin = path_skill(test_c, preds_test, [h for h in mesures if h <= h_pub], n_boot, seed=0)
    modele = choose(chemin)
    retenu = {str(h): modele for h in mesures if h <= h_pub}
    petit = modele != "naif" and chemin[modele]["skill_moyen"] < 0.05
    all_c = pd.concat([tune_c, test_c])  # diagnostic : quantiles bâtis sur toutes les erreurs ≤ 2021
    q_all = error_quantiles(_errors(all_c, predict(all_c, phi, kappa)))
    largeur = {m: {h: _width(q_all, m, h) / _width(q_tune, m, h) for h in mesures
                   if _width(q_tune, m, h) > 0} for m in MODELS}
    years = panel["year"].astype(int)
    res = base | {
        "statut": "ok", "rows": rows, "retenu": retenu, "retenu_modele": modele, "skill_chemin": chemin,
        "rows_reglage": rows_reglage,
        "reglages": {"phi": phi, "kappa": kappa,
                     "rappel_par_horizon": {h: float(pull(kappa, h)) for h in range(1, H_MAX + 1)}},
        "horizons_mesures": mesures, "horizon_max_publie": h_pub,
        "intervalles": {"test": q_tune, "avenir": q_tune, "quantiles": list(Q),
                        "source_avenir": f"erreurs des cibles ≤ {TUNE_END} (celles dont la couverture "
                                         f"{TUNE_END + 1}–{LAST_TARGET} est mesurée)",
                        "largeur_toutes_erreurs_sur_reglage": largeur},
        "emprise": {"par_origine": [[k, o, *v] for (k, o), v in sorted(coefs.items())],
                    "source_avenir": "par_origine, à l'année de la dernière valeur connue de la mine (même "
                                     "famille de modèle que pour le réglage de κ et les intervalles)"},
        "donnees": {"source": "Jasansky et al. 2023 (CC BY 4.0), série principale de chaque mine",
                    "mines": int(panel["site_id"].nunique()), "annees": [int(years.min()), int(years.max())],
                    "mines_avec_emprise": int(panel.loc[panel["emprise_type"].notna(), "site_id"].nunique()),
                    "cas_reglage": int(len(tune_c)), "cas_test": int(len(test_c))},
        "avertissements": [
            "L'emprise Maus est tracée sur des images 2019 : pour les années anciennes elle vient « du "
            "futur ». Le rappel vers l'emprise n'est donc pas strictement temporel, contrairement au reste.",
            "La base de production s'arrête en 2019–2020 (2021 presque vide) : la période de test porte "
            "surtout sur 2015–2019.",
            "Signaux satellite et actualité non utilisés : trop récents (2016+) pour un rétro-test "
            "2005–2021.",
            *([] if not petit else [
                "Gain moyen du modèle retenu sur le naïf inférieur à 5 % de MSE : la prévision reste "
                "proche de « la dernière valeur connue »."])],
        "diagnostic_tendance_brute": diag,
        "today": today.isoformat()}
    return _clean(res)


def backtest(out: Path | None = None, n_boot: int = N_BOOT, panel: pd.DataFrame | None = None,
             today: date | None = None) -> dict:
    """Lance le rétro-test et écrit data/forecast/backtest.json."""
    out = OUT if out is None else out
    res = run_backtest(_panel() if panel is None else panel, n_boot, today)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{BACKTEST_NAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    os.replace(tmp, out / BACKTEST_NAME)
    return res


def load_backtest(out: Path | None = None) -> dict | None:
    """backtest.json tel qu'écrit (lecture seule, pour l'API) ; None s'il n'existe pas."""
    p = (OUT if out is None else out) / BACKTEST_NAME
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# =============================================================================================
# Prévisions d'avenir
# =============================================================================================
def _reglages(bt: dict) -> tuple[float, float]:
    """(φ, κ) du rétro-test ; 0 (naïf) si absents ou d'un ancien format."""
    reg = bt.get("reglages") or {}
    return tuple(float(v) if _finite(v) else 0.0 for v in (reg.get("phi"), reg.get("kappa")))


def _future_anchor(kind, lfp: float, bt: dict, origin: int) -> float:
    """Ordre de grandeur pour une prévision d'avenir faite depuis l'origine `origin` (dernière année connue).

    Même famille que le rétro-test : le modèle d'emprise ré-estimé sur la coupe de l'année d'origine
    (par_origine). Sans ajustement pour ce (type, année) — moins de MIN_ANCHOR mines —, pas de rappel,
    exactement comme pour les cas du rétro-test. κ et les quantiles d'erreur ont été réglés sur ce même
    ancrage ; la calibration.json (autre échantillon, 2017–2020) n'est plus utilisée ici."""
    if not isinstance(kind, str):
        return math.nan
    par = {(x[0], int(x[1])): tuple(x[2:]) for x in (bt.get("emprise") or {}).get("par_origine", [])}
    return _anchor_ln(kind, lfp, par.get((kind, int(origin))))


def site_forecast_frame(g: pd.DataFrame, bt: dict, today: date | None = None) -> pd.DataFrame:
    """Prévisions des trois modèles d'une mine (lignes du panel), de la dernière année connue + 1 à
    l'année en cours + 1, sans dépasser le plus long horizon mesuré hors échantillon. Vide si la série est
    insuffisante (pente non calculable) ou trop ancienne (l'année en cours serait hors des horizons
    mesurés). Colonne mesure : horizon mesuré dans le rétro-test (toujours vrai avec ce plafond)."""
    today = today or date.today()
    cols = ["site_id", "serie", "target_year", "h", "mesure", "model", "value", "lo80", "hi80", "lo90",
            "hi90"]
    g = g[g["value"] > 0].sort_values("year")
    if g.empty or bt.get("statut") != "ok":
        return pd.DataFrame(columns=cols)
    years, lv = g["year"].to_numpy(int), np.log(g["value"].to_numpy(float))
    o = int(years[-1])
    l0, sl = _origin_state(years, lv, o)
    h_pub = _h_pub(bt)
    if not np.isfinite(sl) or today.year - o > h_pub or today.year + 1 - o < 1:
        return pd.DataFrame(columns=cols)
    hs = np.arange(1, min(today.year + 1 - o, h_pub) + 1)
    mes = set(bt.get("horizons_mesures") or range(1, h_pub + 1))
    kind, fp = g["emprise_type"].iat[0], g["footprint_km2"].iat[0]
    lfp = math.log10(fp) if isinstance(kind, str) and np.isfinite(fp) else math.nan
    c = pd.DataFrame({"h": hs, "l0": l0, "slope": sl, "anchor": _future_anchor(kind, lfp, bt, o)})
    preds = predict(c, *_reglages(bt))
    quant = (bt.get("intervalles") or {}).get("avenir") or {}
    frames = []
    for model, p in preds.items():
        lo90, lo80, hi80, hi90 = _bounds(p, hs, quant, model)
        frames.append(pd.DataFrame({"site_id": g["site_id"].iat[0], "serie": g["serie"].iat[0],
                                    "target_year": o + hs, "h": hs,
                                    "mesure": [int(h) in mes for h in hs], "model": model, "value": np.exp(p),
                                    "lo80": np.exp(lo80), "hi80": np.exp(hi80), "lo90": np.exp(lo90),
                                    "hi90": np.exp(hi90)}))
    return pd.concat(frames, ignore_index=True)[cols]


def _retenu(bt: dict, h: int) -> str:
    if isinstance(bt.get("retenu_modele"), str):  # un seul modèle pour tout le chemin
        return bt["retenu_modele"]
    r = bt.get("retenu") or {}
    keys = sorted(int(k) for k in r) or [1]
    return r.get(str(min(h, keys[-1])), "naif")


def _bt_or_run(out: Path | None) -> dict:
    bt = load_backtest(out)
    return bt if bt is not None else backtest(out)


def _resolve(site_id: str, panel: pd.DataFrame) -> pd.DataFrame:
    g = panel[panel["site_id"] == site_id]
    if g.empty:
        g = panel[panel["jasansky"] == site_id]
    if g.empty:
        try:  # mine de la liste suivie : production rangée sous sa jumelle du catalogue
            from strate.screener import twins
            tw = twins().get(site_id)
            g = panel[panel["site_id"] == tw] if tw else g
        except Exception as e:  # noqa: BLE001
            log.warning("jumelles indisponibles : %s", e)
    return g


def _site_track(g: pd.DataFrame, bt: dict, model: str) -> dict:
    """Bilan de la mine elle-même à h = 1 (rétro-test rejoué avec les mêmes réglages)."""
    par = (bt.get("emprise") or {}).get("par_origine", [])
    coefs = {(x[0], int(x[1])): tuple(x[2:]) for x in par}
    c = add_anchor(make_cases(g, h_max=1), coefs)
    if c.empty:
        return {"n": 0}
    e = np.abs(c["y"].to_numpy(float) - predict(c, *_reglages(bt))[model])
    return {"n": int(len(c)), "facteur_median": float(np.exp(np.median(e))), "modele": model, "horizon": 1}


def forecast_site(site_id: str, out: Path | None = None, panel: pd.DataFrame | None = None,
                  today: date | None = None) -> dict | None:
    """Historique, prévision (modèle retenu par horizon, intervalles 80/90 %) et qualité mesurée d'une mine.
    None si la mine n'a pas de série de production.

    Lecture seule : lit backtest.json sans jamais lancer ni écrire le rétro-test. S'il manque, la réponse
    garde l'historique, une prévision vide et un champ « message »."""
    panel = _panel() if panel is None else panel
    g = _resolve(site_id, panel)
    if g.empty:
        return None
    g = g.sort_values("year")
    histo = [{"year": int(y), "value": float(v)} for y, v in zip(g["year"], g["value"], strict=True)]
    ident = {"site_id": str(g["site_id"].iat[0]), "nom": g["nom"].iat[0], "serie": g["serie"].iat[0],
             "unite": "t", "historique": histo, "derniere_annee": int(g["year"].max())}
    bt = load_backtest(out)  # lecture seule : jamais de rétro-test lancé ni écrit ici
    if bt is None:
        return _clean(ident | {"prevision": [], "qualite": {"n": 0}, "qualite_par_horizon": [],
                               "suivi_site": {"n": 0}, "message": MSG_BACKTEST,
                               "note": MSG_BACKTEST, "backtest_run_at": None})
    fc = site_forecast_frame(g, bt, today)
    prev = []
    env = np.zeros(4)  # écarts log (lo90, lo80, hi80, hi90) : enveloppe qui ne rétrécit pas au changement
    for h, f in fc.groupby("h", sort=True):  # de modèle retenu d'un horizon à l'autre
        r = f[f["model"] == _retenu(bt, int(h))]
        if len(r):
            r = r.iloc[0]
            v = float(r["value"])
            with np.errstate(divide="ignore", invalid="ignore"):
                off = np.log(np.array([r["lo90"], r["lo80"], r["hi80"], r["hi90"]], float) / v)
            if np.isfinite(off).all():
                env = np.r_[np.minimum(off[:2], env[:2]), np.maximum(off[2:], env[2:])]
                off = env
            b = v * np.exp(off)
            prev.append({"year": int(r["target_year"]), "horizon": int(h), "mesure": bool(r["mesure"]),
                         "model": r["model"], "value": v,
                         "lo90": b[0], "lo80": b[1], "hi80": b[2], "hi90": b[3]})
    rows = bt.get("rows") or []

    def qual(h: int) -> dict | None:
        m = _retenu(bt, h)
        r = next((x for x in rows if x["horizon"] == h and x["model"] == m), None)
        if r is None:
            return None
        keys = ("r", "r_lo", "r_hi", "rho", "rho_lo", "rho_hi", "skill", "skill_lo", "skill_hi",
                "facteur_median", "couverture80", "couverture90", "n_mines")
        return {"model": m, "horizon": h, **{k: r.get(k) for k in keys}, "n": r.get("n_points")}

    q1 = qual(1)
    return _clean(ident | {
        "prevision": prev,
        "qualite": q1 if q1 is not None else {"n": 0},
        "qualite_par_horizon": [q for h in HORIZONS if h <= _h_pub(bt) and (q := qual(h)) is not None],
        "suivi_site": _site_track(g, bt, _retenu(bt, 1)),
        "note": None if prev else "série insuffisante ou trop ancienne pour une prévision",
        "backtest_run_at": bt.get("run_at")})


# =============================================================================================
# Registre append-only
# =============================================================================================
@contextmanager
def _locked(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".ledger.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def read_ledger(out: Path | None = None) -> pd.DataFrame:
    p = (OUT if out is None else out) / LEDGER_NAME
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=LEDGER_COLS)


def all_forecasts(panel: pd.DataFrame, bt: dict, today: date | None = None) -> pd.DataFrame:
    frames = [site_forecast_frame(g, bt, today) for _, g in panel.groupby("site_id", sort=True)]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_ledger(out: Path | None = None, panel: pd.DataFrame | None = None, today: date | None = None,
                 now: datetime | None = None) -> int:
    """Ajoute au registre data/forecast/ledger.parquet les prévisions d'avenir des trois modèles.

    Append-only : une ligne écrite n'est jamais modifiée ni supprimée. Une prévision identique (valeur et
    bornes) à la dernière déjà enregistrée (même mine, série, année cible, modèle) n'est pas répétée — la
    première datée fait foi. Retourne le nombre de lignes ajoutées.
    """
    out = OUT if out is None else out
    panel = _panel() if panel is None else panel
    bt = _bt_or_run(out)
    fc = all_forecasts(panel, bt, today)
    if fc.empty:
        return 0
    made_at = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = fc.assign(made_at=made_at, target_year=fc["target_year"].astype(int))[LEDGER_COLS]
    key = ["site_id", "serie", "target_year", "model"]
    with _locked(out):
        old = read_ledger(out)
        if len(old):
            # vraie dernière ligne de chaque clé (GroupBy.last sauterait les NaN colonne par colonne et
            # mélangerait des lignes d'époques différentes)
            last = old.sort_values("made_at", kind="stable").drop_duplicates(key, keep="last")
            vals = ["value", "lo80", "hi80", "lo90", "hi90"]
            m = new.merge(last[key + vals], on=key, how="left", suffixes=("", "_old"))
            same = np.ones(len(m), bool)
            for v in vals:  # bornes NaN des deux côtés = identiques ; valeur absente avant = nouvelle
                a, b = m[v].to_numpy(float), m[f"{v}_old"].to_numpy(float)
                same &= np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=(v != "value"))
            new = new[~same]
        if new.empty:
            return 0
        allr = pd.concat([old, new], ignore_index=True) if len(old) else new
        tmp = out / f".{LEDGER_NAME}.{os.getpid()}.tmp"
        allr.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, out / LEDGER_NAME)
    return int(len(new))


# =============================================================================================
# CLI : python -m strate.forecast backtest|ledger (strate forecast … une fois branché dans cli.py)
# =============================================================================================
def _print_rows(res: dict) -> None:
    def f(v, pct=False):
        return "  n.d." if v is None else (f"{v:+.0%}" if pct else f"{v:+.2f}")
    print(f"Rétro-test (réglage {res['periode']['reglage']}, test {res['periode']['test']})")
    for r in res.get("rows", []):
        print(f"h={r['horizon']} {r['model']:<16} n={r['n_points']:>5} ({r['n_mines']} mines)  "
              f"r={f(r['r'])} [{f(r['r_lo'])};{f(r['r_hi'])}]  ρ={f(r['rho'])} [{f(r['rho_lo'])};"
              f"{f(r['rho_hi'])}]  skill={f(r['skill'], True)} [{f(r['skill_lo'], True)};"
              f"{f(r['skill_hi'], True)}]  ×{r['facteur_median']}  sens={r['bon_sens']}  "
              f"couv80={r['couverture80']} couv90={r['couverture90']}")
    ch = (res.get("skill_chemin") or {}).get(res.get("retenu_modele")) or {}
    print("retenu :", res.get("retenu_modele"), "skill moyen", ch.get("skill_moyen"),
          f"[{ch.get('skill_moyen_lo')} ; {ch.get('skill_moyen_hi')}]", "| réglages :", res.get("reglages"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="strate forecast", description="Prévisions de production auto-évaluées")
    p.add_argument("action", choices=["backtest", "ledger"])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = p.parse_args(argv)
    if a.action == "backtest":
        _print_rows(backtest(n_boot=a.n_boot))
    else:
        print(f"{write_ledger()} prévisions ajoutées au registre {OUT / LEDGER_NAME}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
