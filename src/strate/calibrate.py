"""Calibration des signaux satellite sur la production réelle (Jasansky et al. 2023).

Question : est-ce que ce que Strate mesure depuis l'espace suit ce que les mines produisent vraiment ?

Fenêtre commune : 2016–2019 (Sentinel-2 L2A disponible dès 2016 ; la base de production s'arrête
en pratique en 2019). Pour chaque mine-année on agrège les mois clairs, puis trois tests :
  T1  En coupe : les grosses mines (production) ont-elles une grosse empreinte (surface nue) ?
  T2  Dans le temps : quand la production d'une mine monte, ses signaux satellite montent-ils ?
  T3  Seuils : réglés sur la moitié des mines, vérifiés sur l'autre moitié (pas de sur-ajustement).
Tous les résultats ont un intervalle de confiance (bootstrap par mine).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from strate import catalog
from strate.core import DATA_DIR
from strate.modules import optical
from strate.modules.optical import Optical

log = logging.getLogger(__name__)

OUT = DATA_DIR / "calibration"
YEARS = (2016, 2019)
SERIES_PREF = ["minerai extrait", "minerai traité", "charbon"]
# Réglage satellite de la calibration : 1 scène (la plus claire) par mois parmi 4 examinées
CALIB_SCENES, CALIB_CHECKS = 1, 4
# Fenêtre « saturée » : presque tout le carré d'analyse est nu (mine plus grande que la fenêtre plafonnée
# à 8 km de rayon, ou désert) ; la surface nue y mesure la fenêtre, plus la mine
SATURATION = 0.9


def _optical() -> Optical:
    """Optical au réglage de la calibration, qui relit aussi les composites 1 scène écrits avant la
    séparation des caches par réglage (chemin historique, fichiers sans étiquette « reglage »)."""
    return Optical(workers=4, scenes_per_month=CALIB_SCENES, scl_checks=CALIB_CHECKS, read_legacy=True)


# -- 1. sélection -------------------------------------------------------------------------
def _all_production() -> pd.DataFrame:
    """Toutes les séries de production des installations rattachées au catalogue."""
    ids = set(catalog.load()["jasansky"].dropna())
    p = catalog.all_production()
    return p[p["facility_id"].isin(ids)].rename(columns={"facility_id": "jasansky"})


def _equipment(f: pd.DataFrame) -> pd.Series:
    """Équipement par installation MÈRE (« XXX.00 ») : réunit celui des sous-installations.

    Les sites du catalogue pointent vers l'id mère (seul porteur de production) ; pour 28 mères,
    l'équipement n'est renseigné que sur les sous-installations — sans ce regroupement, le filtre
    « mine purement souterraine » les laissait passer.
    """
    parent = f["facility_id"].astype(str).str.replace(r"\.\d+$", ".00", regex=True)
    eq = f["facility_equipment"].astype("string").str.lower()
    return eq.groupby(parent).agg(lambda s: ", ".join(sorted(set(s.dropna()))) or pd.NA).astype("string")


def _main_series(p: pd.DataFrame, commodities: str | None) -> str | None:
    have = set(p["serie"])
    for s in SERIES_PREF:
        if s in have:
            return s
    fr = {"copper": "cuivre", "gold": "or", "silver": "argent", "zinc": "zinc", "lead": "plomb",
          "nickel": "nickel", "iron ore": "fer", "iron": "fer", "molybdenum": "molybdène",
          "uranium": "uranium", "platinum": "platine", "cobalt": "cobalt"}
    for c in (commodities if isinstance(commodities, str) else "").split(","):
        s = f"{fr.get(c.strip().lower(), c.strip().lower())} (métal)"
        if s in have:
            return s
    return p.groupby("serie").size().idxmax() if len(p) else None


def _underground_only(eq) -> bool:
    """Mine purement souterraine : sa production ne se voit pas depuis l'espace."""
    return isinstance(eq, str) and "underground" in eq and "open" not in eq and "surface" not in eq


ELIG_COLS = ["id", "jasansky", "nom", "pays", "serie", "annees", "equipement", "emprise_km2", "exclusion"]


def _eligibility_table(cat: pd.DataFrame, prod: pd.DataFrame, equip: pd.Series) -> pd.DataFrame:
    """Éligibilité de CHAQUE site du catalogue à la calibration, avec la raison d'exclusion (None = éligible).

    Règles (celles du tirage) : production réelle liée (= has_production : id Jasansky présent ET ayant des
    lignes dans all_production), ≥ 3 ans de production > 0 sur la fenêtre, pas purement souterraine
    (équipement regroupé par installation mère), emprise connue. Un id Jasansky porté par plusieurs sites
    ne compte qu'une fois : pour le premier site du catalogue, comme au tirage d'origine.
    """
    groups = {k: g for k, g in prod.groupby("jasansky")}
    linked = cat[cat["jasansky"].notna()]
    owner: dict[str, str] = {}
    rows = []
    for site in linked.to_dict("records"):
        jid = site["jasansky"]
        r = {"id": site["id"], "jasansky": jid, "nom": site.get("name"), "pays": site.get("country"),
             "serie": None, "annees": 0, "equipement": None, "emprise_km2": site.get("footprint_km2"),
             "exclusion": None}
        if jid not in groups:
            r["exclusion"] = f"aucune production réelle pour {jid}"
        elif jid in owner:
            r["exclusion"] = f"production {jid} déjà portée par {owner[jid]}"
        else:
            owner[jid] = site["id"]
            p = groups[jid]
            serie = _main_series(p, site.get("commodities"))
            s = p[(p["serie"] == serie) & p["year"].between(YEARS[0] - 1, YEARS[1])]
            years = int(s[s["value_tonnes"] > 0]["year"].nunique())
            eq = equip.get(jid)
            r.update(serie=serie, annees=years, equipement=eq if isinstance(eq, str) else None)
            if _underground_only(eq):
                r["exclusion"] = "mine purement souterraine"
            elif years < 3:
                r["exclusion"] = f"moins de 3 ans de production sur {YEARS[0] - 1}–{YEARS[1]}"
            elif pd.isna(r["emprise_km2"]):
                r["exclusion"] = "emprise inconnue"
        rows.append(r)
    # sites sans id Jasansky : listés aussi, pour distinguer « délié » de « sorti du catalogue »
    rest = cat[cat["jasansky"].isna()]
    unlinked = pd.DataFrame({"id": rest["id"], "nom": rest.get("name"), "pays": rest.get("country"),
                             "emprise_km2": rest.get("footprint_km2"), "annees": 0,
                             "exclusion": "aucun id Jasansky (pas de production réelle liée)"})
    out = pd.concat([pd.DataFrame(rows, columns=ELIG_COLS), unlinked.reindex(columns=ELIG_COLS)],
                    ignore_index=True)
    return out.set_index("id", drop=False)


def _eligibility() -> pd.DataFrame:
    """Éligibilité calculée sur le catalogue ACTUEL (mines.parquet) et l'équipement Jasansky."""
    import geopandas as gpd

    f = gpd.read_file(catalog.SRC / "jasansky2023" / "x" / "data" / "facilities.gpkg")
    return _eligibility_table(catalog.load(), _all_production(), _equipment(f))


def _write_selection(sel: pd.DataFrame, elig: pd.DataFrame) -> None:
    """Écrit selection.csv et selection_info.json (population éligible, exclusions, liens changés)."""
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "selection.csv", index=False)
    kept = sel["exclusion"].isna()
    relinked = sel[kept & (sel["jasansky"] != sel["jasansky_tirage"])]
    drift = (sel["emprise_km2"] / sel["emprise_tirage_km2"] - 1).abs()
    moved = sel[kept & (drift > 0.05)]
    info = {
        "date": date.today().isoformat(),
        "eligibles": int(elig["exclusion"].isna().sum()), "tirees": int(len(sel)),
        "retenues": int(kept.sum()),
        "exclues": dict(zip(sel.loc[~kept, "id"], sel.loc[~kept, "exclusion"], strict=True)),
        "production_reliee": {r.id: [r.jasansky_tirage, r.jasansky] for r in relinked.itertuples()},
        # le carré d'analyse optique reste celui du cache (emprise au moment du calcul satellite)
        "emprise_modifiee_km2": {r.id: [r.emprise_tirage_km2, r.emprise_km2] for r in moved.itertuples()},
    }
    (OUT / "selection_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float))
    log.warning("sélection : %d retenues / %d tirées, %d mines éligibles aujourd'hui",
                info["retenues"], info["tirees"], info["eligibles"])
    for sid, why in info["exclues"].items():
        log.warning("  exclue %s : %s", sid, why)


def select(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """Tirage : mines avec ≥ 3 ans de production réelle sur 2016–2019 et une emprise satellite connue.

    Les mines purement souterraines sont exclues : leur production ne se voit pas depuis l'espace.
    """
    elig = _eligibility()
    sel = elig[elig["exclusion"].isna()].sort_values("jasansky").reset_index(drop=True)
    log.warning("%d mines éligibles", len(sel))
    if len(sel) > n:
        sel = sel.sample(n, random_state=seed)
    sel = sel.sort_values("id").reset_index(drop=True)
    sel["groupe"] = np.where(np.random.default_rng(seed).random(len(sel)) < 0.5, "réglage", "test")
    sel = sel.assign(jasansky_tirage=sel["jasansky"], emprise_tirage_km2=sel["emprise_km2"])
    _write_selection(sel, elig)
    return sel


def refresh_selection(elig: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recale le tirage figé (selection.csv) sur le catalogue et les règles d'éligibilité ACTUELS.

    Le tirage date d'un ancien catalogue : depuis, des ids Jasansky ont changé de site, des mines se sont
    révélées purement souterraines (équipement regroupé par mère), des emprises ont bougé. Sans ce recalage
    le panel associait une fenêtre satellite à la production d'une AUTRE mine. On garde les ids et les
    groupes réglage/test tirés (et donc radar_choix.json) ; jasansky, série et emprise sont relus ; les mines
    devenues inéligibles restent listées avec leur raison (colonne « exclusion ») mais sortent des panels.
    Les caches satellite sont indexés par id de site : aucun téléchargement.
    """
    sel = pd.read_csv(OUT / "selection.csv")
    elig = _eligibility() if elig is None else elig
    # provenance du tirage : figée à la première actualisation, jamais réécrite ensuite
    if "jasansky_tirage" not in sel:
        sel["jasansky_tirage"] = sel["jasansky"]
    if "emprise_tirage_km2" not in sel:
        sel["emprise_tirage_km2"] = sel["emprise_km2"]
    known = sel["id"].isin(elig.index).to_numpy()
    cur = elig.reindex(sel["id"]).reset_index(drop=True)
    for c in ("jasansky", "serie", "annees", "equipement", "emprise_km2"):
        sel[c] = cur[c]
    for c in ("nom", "pays"):  # nom lisible même pour une mine sortie du catalogue
        sel[c] = cur[c].fillna(sel[c])
    why = cur["exclusion"].astype(object).where(known, "absente du catalogue actuel")
    # où est partie la production du tirage ? (ex. COM00818 de Brucutu désormais sur Gongo Soco)
    carrier = elig[elig["jasansky"].notna()].groupby("jasansky")["id"].first()
    for i, r in sel.iterrows():
        jt = r["jasansky_tirage"]
        if pd.notna(why[i]) and jt != r["jasansky"] and jt in carrier.index and carrier[jt] != r["id"]:
            why[i] = f"{why[i]} ; {jt} désormais rattaché à {carrier[jt]} ({elig.at[carrier[jt], 'nom']})"
    sel["exclusion"] = why
    _write_selection(sel, elig)
    return sel


def _retained(sel: pd.DataFrame) -> pd.DataFrame:
    """Mines du tirage encore éligibles (colonne « exclusion » vide ; absente = ancien selection.csv)."""
    return sel[sel["exclusion"].isna()] if "exclusion" in sel else sel


# -- 2. satellite ----------------------------------------------------------------------------
def run(parallel: int = 4) -> None:
    sel = _retained(pd.read_csv(OUT / "selection.csv"))
    # 1 scène (la plus claire) par mois : ~3× plus rapide ; le bruit se moyenne dans la médiane annuelle
    opt = _optical()

    def job(sid):
        t = time.time()
        err = None
        for _ in range(3):
            try:
                opt.update(sid, date(YEARS[0], 1, 1), date(YEARS[1], 12, 31))
                print(f"{sid:20s} ok {time.time() - t:5.0f} s", flush=True)
                return
            except Exception as e:  # on note et on passe : une mine en échec ne bloque pas les autres
                err = e
                time.sleep(10)
        print(f"{sid:20s} ERREUR {err!r}", flush=True)

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(job, sel["id"]))


# -- 3. panel mine × année ---------------------------------------------------------------------
METRICS = {
    "d_bare": "variation de la surface nue (log)",
    "new_rel": "surface nouvellement défrichée / surface nue de l'an passé",
    "change_rel": "empreinte modifiée / surface nue de l'an passé",
    "d_water": "variation de la surface en eau (log)",
}
GRID = [{"ndvi_bare": nb, "swir_change": sc}
        for nb in (0.15, 0.2, 0.25, 0.3) for sc in (0.05, 0.08, 0.12, 0.16)]
MIN_MONTHS = 3


def _calib_dirs(site_id: str) -> tuple:
    """Dossiers de composites de la calibration : réglage 1 scène, puis chemin historique (repli)."""
    return optical.cache_dir(site_id, CALIB_SCENES, CALIB_CHECKS), optical.cache_dir(site_id)


def _comps(site_id: str) -> dict[date, dict]:
    """Composites 1 scène de la calibration.

    Depuis la séparation des caches par réglage (voir Optical._cache), ils s'écrivent dans
    cache/optical/<site>/s1c4/. Ceux du premier run de calibration sont restés au chemin historique
    cache/optical/<site>/ : on les relit en repli, sauf s'ils portent l'étiquette d'un autre réglage
    (un composite 3 scènes écrit depuis par le suivi des mines).
    """
    opt = _optical()
    stems = {f.stem for d in _calib_dirs(site_id) for f in (d.glob("*.npz") if d.exists() else [])}
    out = {}
    for stem in sorted(stems):
        y, m = map(int, stem.split("-"))
        if not YEARS[0] - 1 <= y <= YEARS[1]:
            continue
        f = opt._cached(site_id, date(y, m, 1))
        if f is not None:
            with np.load(f) as z:
                out[date(y, m, 1)] = optical.unpack(z)
    return out


@cache
def _window_km2(site_id: str) -> float:
    """Surface du carré d'analyse satellite d'une mine (km²), lue sur un de ses composites en cache.

    Le rayon du carré est dimensionné sur l'emprise connue de la mine (entities : demi-emprise + 500 m,
    borné à 1,5–8 km) : la taille de la fenêtre porte donc déjà l'information « grosse mine ».
    """
    for d in _calib_dirs(site_id):
        for f in sorted(d.glob("*.npz")) if d.exists() else []:
            with np.load(f) as z:
                return float(z["valid"].size * optical.RES * optical.RES / 1e6)
    return float("nan")


def _with_window(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute fenetre_km2 et frac_nue (surface nue / fenêtre) si absents (anciens panel.parquet)."""
    if "fenetre_km2" not in df:
        df = df.assign(fenetre_km2=df["id"].map(_window_km2))
    return df.assign(frac_nue=df["bare"] / df["fenetre_km2"])


def annual(site_id: str, grid: list[dict]) -> dict[int, pd.DataFrame]:
    """Pour chaque jeu de seuils : médiane annuelle des signaux sur les mois clairs."""
    comps = _comps(site_id)
    px = optical.RES * optical.RES / 1e6
    res = {}
    for gi, p in enumerate(grid):
        rows = []
        for m, c in comps.items():
            if "red" not in c:
                continue
            meas = Optical.measures(c, comps.get(date(m.year - 1, m.month, 1)), px, p)
            if meas["quality"] >= optical.MIN_MONTH_VALID and "bare_km2" in meas:
                rows.append({"year": m.year, **meas})
        df = pd.DataFrame(rows).reindex(columns=["year", "quality", "bare_km2", "water_km2", "change_km2",
                                                 "new_bare_km2"])
        if df.empty:
            res[gi] = df
            continue
        g = df.groupby("year")
        a = g[["bare_km2", "water_km2", "change_km2", "new_bare_km2"]].median()
        a["months"] = g.size()
        res[gi] = a[a["months"] >= MIN_MONTHS]
    return res


def done_ids() -> set[str]:
    """Mines dont le calcul satellite de calibration est terminé (journal du run)."""
    log_path = DATA_DIR.parent / "logs" / "calibration_run.log"
    if not log_path.exists():
        return set()
    return {ln.split()[0] for ln in log_path.read_text().splitlines() if " ok " in ln}


def panel(grid: list[dict] | None = None, only_done: bool = True,
          sel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Une ligne par mine × paire d'années consécutives × jeu de seuils.

    `sel` : sélection déjà recalée ; sinon recalage sur le catalogue actuel (refresh_selection), pour ne
    jamais apparier une fenêtre satellite à une production que le catalogue a rattachée ailleurs.
    """
    grid = grid or GRID
    sel = _retained(refresh_selection() if sel is None else sel)
    if only_done:
        sel = sel[sel["id"].isin(done_ids())]
    prod = _all_production()
    rows = []
    for r in sel.itertuples():
        mine = prod[(prod["jasansky"] == r.jasansky) & (prod["serie"] == r.serie)]
        p = mine.groupby("year")["value_tonnes"].sum()
        sat = annual(r.id, grid)
        win = _window_km2(r.id)
        for gi, a in sat.items():
            for y in range(YEARS[0], YEARS[1] + 1):
                if y not in a.index or y - 1 not in a.index or p.get(y, 0) <= 0 or p.get(y - 1, 0) <= 0:
                    continue
                cur, prev = a.loc[y], a.loc[y - 1]
                if prev["bare_km2"] <= 0.05 or cur["bare_km2"] <= 0.05:
                    continue
                rows.append({
                    "id": r.id, "nom": r.nom, "groupe": r.groupe, "serie": r.serie, "grid": gi, "year": y,
                    "P": p[y], "lP": np.log(p[y]), "dP": np.log(p[y] / p[y - 1]), "bare": cur["bare_km2"],
                    "fenetre_km2": win, "frac_nue": cur["bare_km2"] / win,
                    "d_bare": np.log(cur["bare_km2"] / prev["bare_km2"]),
                    "new_rel": cur["new_bare_km2"] / prev["bare_km2"],
                    "change_rel": cur["change_km2"] / prev["bare_km2"],
                    "d_water": np.log((cur["water_km2"] + 0.01) / (prev["water_km2"] + 0.01)),
                    "months": int(min(cur["months"], prev["months"])),
                })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "panel.parquet", index=False)
    return df


# -- 4. statistiques ------------------------------------------------------------------------------
def _boot(df: pd.DataFrame, stat, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Statistique + intervalle à 90 % par bootstrap SUR LES MINES (années d'une mine non indépendantes)."""
    ids = df["id"].unique()
    if not len(ids):  # groupe vide (ex. aucun grand mouvement) : rien à estimer
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby("id")}
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        v = stat(pd.concat([groups[k] for k in pick], ignore_index=True))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float(stat(df)), float("nan"), float("nan")
    return float(stat(df)), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def _rho(x: str, y: str = "dP"):
    def f(d):
        d = d[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 8 or d[x].nunique() < 3:
            return np.nan
        return spearmanr(d[x], d[y]).statistic
    return f


def _within(x: str, y: str = "dP"):
    """Corrélation après avoir retiré la moyenne de chaque mine (effet propre au temps)."""
    def f(d):
        d = d[["id", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        dd = d.assign(xc=d[x] - d.groupby("id")[x].transform("mean"),
                      yc=d[y] - d.groupby("id")[y].transform("mean"))
        return spearmanr(dd["xc"], dd["yc"]).statistic if len(dd) >= 8 else np.nan
    return f


def _direction(x: str, real: float = 0.10):
    def f(d):
        dd = d[d["dP"].abs() > real]
        return float((np.sign(dd[x]) == np.sign(dd["dP"])).mean()) if len(dd) >= 8 else np.nan
    return f


def _precision(x: str, sat: float, real: float = 0.20):
    """Quand le satellite voit bouger l'emprise de plus de `sat`, la production bouge-t-elle de > `real` ?"""
    def f(d):
        hit = d[x].abs() > sat
        return float((d.loc[hit, "dP"].abs() > real).mean()) if hit.sum() >= 5 else np.nan
    return f


def report(df: pd.DataFrame | None = None) -> dict:
    df = panel() if df is None else df
    base = df[df["grid"] == GRID.index({"ndvi_bare": 0.2, "swir_change": 0.08})]
    out = {"mines": int(df["id"].nunique()), "paires_annees": int(len(base)), "fenetre": list(YEARS)}
    # population et tirage tels que recalés sur le catalogue actuel (cité par la doc : « N éligibles »)
    info = OUT / "selection_info.json"
    if info.exists():
        i = json.loads(info.read_text())
        out["echantillon"] = {k: i[k] for k in ("eligibles", "tirees", "retenues")} | {
            "exclues": len(i["exclues"])}

    # T1 — en coupe : taille de l'empreinte vs tonnage (minerai et charbon : tonnes comparables)
    # La surface nue (km²) est mesurée dans un carré dimensionné sur l'emprise connue de la mine : elle vaut
    # fraction nue × taille de la fenêtre. Pour savoir ce que voit VRAIMENT le satellite, on sépare les deux :
    # fraction nue seule (ce qu'apporte l'image) et taille de fenêtre seule (aucune image, juste l'emprise).
    keys = ["rho", "ic90_bas", "ic90_haut"]
    t1 = _with_window(base[base["serie"].isin(["minerai extrait", "charbon"])]).groupby("id").agg(
        P=("P", "mean"), bare=("bare", "mean"), fenetre=("fenetre_km2", "first")).reset_index()
    t1 = t1.assign(lP=np.log(t1["P"]), lB=np.log(t1["bare"]), lF=np.log(t1["fenetre"]),
                   frac=t1["bare"] / t1["fenetre"]).assign(dP=lambda x: x["lP"])
    out["T1_coupe"] = dict(zip(keys, _boot(t1, _rho("lB")), strict=True)) | {"mines": len(t1)}
    out["T1_fraction_nue"] = dict(zip(keys, _boot(t1, _rho("frac")), strict=True)) | {
        "mines": len(t1), "lecture": "surface nue / fenêtre d'analyse : ce que l'image ajoute à la taille"}
    out["T1_fenetre_seule"] = dict(zip(keys, _boot(t1, _rho("lF")), strict=True)) | {
        "mines": len(t1), "lecture": "taille de la fenêtre (dérivée de l'emprise connue), sans aucune image"}
    sat = t1[t1["frac"] > SATURATION]
    out["T1_hors_fenetres_saturees"] = dict(zip(keys, _boot(t1[t1["frac"] <= SATURATION], _rho("lB")),
                                                strict=True)) | {"mines": int(len(t1) - len(sat)),
                                                                 "exclues": int(len(sat))}

    # T2 — dans le temps, seuils par défaut, toutes les mines
    # - signaux de variation (d_*) contre la variation de production
    # - signaux d'intensité (new_rel, change_rel) contre le NIVEAU de production de la mine cette année-là
    out["T2_defaut"] = {}
    for m in METRICS:
        if m.startswith("d_"):
            out["T2_defaut"][m] = {"rho": _boot(base, _rho(m)), "rho_intra_mine": _boot(base, _within(m)),
                                   "bon_sens_si_prod_bouge_10pct": _boot(base, _direction(m))}
        else:
            out["T2_defaut"][m] = {"rho_intra_mine_vs_niveau": _boot(base, _within(m, "lP"))}

    # T3 — réglage sur un groupe, vérification sur l'autre
    tune, test = df[df["groupe"] == "réglage"], df[df["groupe"] == "test"]
    # le lien attendu est POSITIF (plus d'activité vue ↔ plus de production) : on ne retient jamais une
    # corrélation négative trouvée par hasard sur le groupe de réglage
    scores = []
    for gi in range(len(GRID)):
        d = tune[tune["grid"] == gi]
        for m in METRICS:
            stat = _rho(m) if m.startswith("d_") else _within(m, "lP")
            v = stat(d)
            if np.isfinite(v):
                scores.append((v, gi, m))
    best_rho, best_gi, best_m = max(scores)
    dtest = test[test["grid"] == best_gi]
    out["T3_reglage"] = {
        "seuils_choisis": GRID[best_gi], "signal_choisi": best_m, "rho_reglage": best_rho,
        "rho_test": _boot(dtest, _rho(best_m) if best_m.startswith("d_") else _within(best_m, "lP")),
        "bon_sens_test": _boot(dtest, _direction(best_m)) if best_m.startswith("d_") else None,
        "mines_test": int(dtest["id"].nunique()), "paires_test": int(len(dtest)),
    }

    # Précision d'une « alerte » annuelle sur l'emprise (seuils par défaut)
    base_rate = float((base["dP"].abs() > 0.20).mean())
    prec = {}
    real = base["dP"].abs() > 0.20
    for s in (0.05, 0.10, 0.20):
        hit = base["d_bare"].abs() > s
        prec[f"emprise_bouge_{int(s * 100)}pct"] = {
            "alertes": int(hit.sum()),
            "precision": _boot(base, _precision("d_bare", s)),
            "rappel": float((hit & real).sum() / max(real.sum(), 1)),
        }
    out["alertes_vs_production"] = {"taux_de_base_prod_bouge_20pct": base_rate, **prec}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    return out


# -- 5. modèle « surface nue → production » (encode surtout la taille de la fenêtre) ----------------
def _loo_errors(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Erreurs absolues (log10) en validation croisée, une mine retirée à la fois."""
    e = np.empty(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        e[i] = abs(y[i] - X[i] @ np.linalg.lstsq(X[k], y[k], rcond=None)[0])
    return e


def _factors(e: np.ndarray) -> dict:
    return {"facteur_median": float(10 ** np.median(e)), "facteur_90": float(10 ** np.percentile(e, 90))}


def fit_size_model(df: pd.DataFrame | None = None) -> dict:
    """log10(production) ~ log10(surface nue), par type de tonnage, erreur par validation croisée LOO.

    Minerai extrait et charbon seulement : des tonnes de matière déplacée, comparables entre mines.
    Attention : la surface nue = fraction nue × taille du carré d'analyse, et ce carré est dimensionné sur
    l'emprise connue de la mine. Le modèle encode donc surtout la taille de l'emprise ; on publie à côté
    la référence « fenêtre seule » (aucune image) et le lien avec la fraction nue (ce que l'image apporte).
    Fenêtres saturées (fraction nue > SATURATION) exclues : leur surface nue mesure la fenêtre, pas la mine.
    """
    df = pd.read_parquet(OUT / "panel.parquet") if df is None else df
    b = _with_window(df[(df["grid"] == GRID.index({"ndvi_bare": 0.2, "swir_change": 0.08}))
                        & df["serie"].isin(["minerai extrait", "charbon"])])
    m = b.groupby(["id", "serie"]).agg(P=("P", "mean"), bare=("bare", "mean"),
                                        fenetre=("fenetre_km2", "first")).reset_index()
    m = m.assign(frac=m["bare"] / m["fenetre"]).dropna(subset=["P", "bare", "fenetre"])
    out = {}
    for key, serie in (("minerai", "minerai extrait"), ("charbon", "charbon")):
        all_ = m[m["serie"] == serie]
        d = all_[all_["frac"] <= SATURATION]
        if len(d) < 5:  # trop peu de mines pour une validation croisée
            continue
        y = np.log10(d["P"].to_numpy())
        xb, xf = np.log10(d["bare"].to_numpy()), np.log10(d["fenetre"].to_numpy())
        one = np.ones(len(d))
        X = np.c_[one, xb]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        errs = _loo_errors(X, y)
        out[key] = {"intercept": float(beta[0]), "slope": float(beta[1]), "n": int(len(d)),
                    **_factors(errs),
                    "surface_min_km2": float(d["bare"].min()), "surface_max_km2": float(d["bare"].max()),
                    "exclues_saturees": int(len(all_) - len(d)),
                    # références : que vaut la seule taille de fenêtre ? que dit la fraction nue seule ?
                    "fenetre_seule": _factors(_loo_errors(np.c_[one, xf], y)),
                    "fraction_nue_seule": _factors(_loo_errors(np.c_[one, d["frac"].to_numpy()], y)),
                    "rho": {"surface_nue": float(spearmanr(xb, y).statistic),
                            "fraction_nue": float(spearmanr(d["frac"], y).statistic),
                            "fenetre": float(spearmanr(xf, y).statistic)}}
    return out


def _summary_path():
    """src/strate/data/calibration.json : synthèse livrée avec le code, lue par l'API et l'interface."""
    from importlib.resources import files

    return files("strate.data").joinpath("calibration.json")


def _read_summary() -> dict:
    path = _summary_path()
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def _write_summary(summary: dict) -> None:
    _summary_path().write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


def save_summary() -> dict:
    """Partie optique de calibration.json.

    FUSIONNÉE dans le fichier existant : les blocs « modele_emprise » et « radar », écrits par
    save_footprint_model / save_radar_summary, ne doivent pas disparaître quand on relance l'optique seule.
    """
    rep = json.loads((OUT / "report.json").read_text())
    t1, t2, t3 = rep["T1_coupe"], rep["T2_defaut"]["d_bare"], rep["T3_reglage"]
    summary = {
        "date": date.today().isoformat(), "fenetre": rep["fenetre"], "mines": rep["mines"],
        "paires_annees": rep["paires_annees"], "echantillon": rep.get("echantillon"),
        "taille": {"rho": t1["rho"], "ic90": [t1["ic90_bas"], t1["ic90_haut"]], "mines": t1["mines"],
                   # à lire avec : la fenêtre seule (sans image) et la fraction nue (ce qu'ajoute l'image)
                   "fenetre_seule": rep.get("T1_fenetre_seule"), "fraction_nue": rep.get("T1_fraction_nue")},
        "variation": {"rho": t2["rho"][0], "ic90": t2["rho"][1:],
                      "bon_sens": t2["bon_sens_si_prod_bouge_10pct"]},
        "reglage": {"rho_reglage": t3["rho_reglage"], "rho_test": t3["rho_test"],
                    "seuils_testes": t3["seuils_choisis"], "retenus": "seuils par défaut"},
        "alertes": {"precision": rep["alertes_vs_production"]["emprise_bouge_10pct"]["precision"],
                    "taux_de_base": rep["alertes_vs_production"]["taux_de_base_prod_bouge_20pct"]},
        # ex-« modele_taille » : renommé car il ne mesure pas une taille de mine indépendante de l'emprise
        # (voir fit_size_model) ; l'estimation de production publiée vient du modèle d'emprise Maus
        "modele_surface_nue": {
            "nature": "surface nue Sentinel-2 dans une fenêtre dimensionnée sur l'emprise connue : encode "
                      "surtout la taille de l'emprise ; comparer à « fenetre_seule » (aucune image)",
            "usage": "diagnostic de calibration, pas d'estimation de production",
            **fit_size_model()},
    }
    _write_summary(_read_summary() | summary)
    return summary


# -- 6. radar Sentinel-1 ------------------------------------------------------------------------------
RADAR_METRICS = {
    "activity_db": "changement moyen 20 m, mine − couronne (dB)",
    "excess2": "part de la mine changeant de > 2 dB (blocs 100 m), moins la couronne",
    "excess3": "part de la mine changeant de > 3 dB (blocs 100 m), moins la couronne",
    "mine_frac3": "part de la mine changeant de > 3 dB (sans correction couronne)",
}


def run_radar(parallel: int = 8) -> None:
    from strate.modules.radar import Radar

    sel = _retained(pd.read_csv(OUT / "selection.csv"))
    radar = Radar(workers=4)

    def job(sid):
        t = time.time()
        for _ in range(3):
            try:
                p = radar.pairs(sid, date(YEARS[0], 1, 1), date(YEARS[1], 12, 31))
                print(f"{sid:20s} radar ok {len(p):4d} paires {time.time() - t:5.0f} s", flush=True)
                return
            except Exception as e:  # une mine en échec ne bloque pas les autres
                err = e
                time.sleep(10)
        print(f"{sid:20s} radar ERREUR {err!r}", flush=True)

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(job, sel["id"]))


def _year_diffs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Variations d'une année sur la suivante, par mine, SEULEMENT entre années consécutives.

    Une année manquante (trop peu de paires radar, production nulle) ferait sinon comparer 2016 à 2018
    comme si c'était une variation annuelle. « lP » donne « dP », les autres colonnes « d_<col> ».
    """
    df = df.sort_values(["id", "year"]).reset_index(drop=True)
    consec = df.groupby("id")["year"].diff() == 1
    for c in cols:
        df["dP" if c == "lP" else f"d_{c}"] = df.groupby("id")[c].diff().where(consec)
    return df


def radar_panel(sel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mine × année : médiane annuelle des métriques radar (passages de même orbite) + production.

    Même sélection recalée que panel() : jasansky, série et emprise du catalogue actuel.
    """
    from strate.modules.radar import Radar

    sel = _retained(refresh_selection() if sel is None else sel)
    prod = _all_production()
    radar = Radar()
    rows = []
    for r in sel.itertuples():
        p = radar.load_pairs(r.id)  # cache nettoyé : sans les « paires » d'un passage avec lui-même
        if p.empty:
            continue
        p["year"] = pd.to_datetime(p["date"]).dt.year
        mine = prod[(prod["jasansky"] == r.jasansky) & (prod["serie"] == r.serie)]
        P = mine.groupby("year")["value_tonnes"].sum()
        for y, g in p.groupby("year"):
            if len(g) < 8 or P.get(y, 0) <= 0:
                continue
            rows.append({"id": r.id, "nom": r.nom, "groupe": r.groupe, "serie": r.serie, "year": y,
                         "P": P[y], "lP": np.log(P[y]), "paires": len(g), "emprise": r.emprise_km2,
                         **{m: float(g[m].median()) for m in RADAR_METRICS}})
    if not rows:  # aucun passage radar en cache : rien à écrire (ne pas écraser un panel existant)
        return pd.DataFrame()
    df = _year_diffs(pd.DataFrame(rows), ["lP", *RADAR_METRICS])
    df.to_parquet(OUT / "radar_panel.parquet", index=False)
    return df


def _loo_gain(cs: pd.DataFrame, m: str) -> dict:
    """Erreur médiane (facteur) en prédisant log P : emprise seule vs emprise + intensité radar."""
    d = cs.dropna(subset=[m]).copy()
    y = d["lP"].to_numpy() / np.log(10)
    xa = np.log10(d["emprise"].to_numpy())
    xr = np.log10(d[m].clip(lower=1e-4).to_numpy())

    def loo(X):
        e = []
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            beta = np.linalg.lstsq(X[k], y[k], rcond=None)[0]
            e.append(abs(y[i] - X[i] @ beta))
        return float(10 ** np.median(e)), float(10 ** np.percentile(e, 90))

    one = np.ones(len(y))
    a, ar = loo(np.c_[one, xa]), loo(np.c_[one, xa, xr])
    return {"mines": len(y), "emprise_seule": {"facteur_median": a[0], "facteur_90": a[1]},
            "emprise_plus_radar": {"facteur_median": ar[0], "facteur_90": ar[1]}}


RADAR_GRANDS = 0.20  # « grand mouvement » de production : |variation log| > 0,20


def _radar_test(test: pd.DataFrame, m: str, n_boot: int) -> dict:
    """Chiffres du groupe de test pour la métrique choisie sur le groupe de réglage.

    Un seul jeu de mines pour tous les chiffres : celles qui ont au moins une variation annuelle
    exploitable (sinon intra-mine et variation ne parlaient pas des mêmes mines : 82 contre 80).
    """
    dm = f"d_{m}"
    pairs = test.dropna(subset=["dP", dm])
    level = test[test["id"].isin(pairs["id"].unique())]
    big = pairs[pairs["dP"].abs() > RADAR_GRANDS]
    return {
        "metrique": m, "mines": int(pairs["id"].nunique()), "paires": int(len(pairs)), "n_boot": n_boot,
        "variation": _boot(pairs, _rho(dm), n=n_boot),
        "bon_sens": _boot(pairs, _direction(dm), n=n_boot),
        "bon_sens_grands_mouvements": _boot(big, _direction(dm, real=RADAR_GRANDS), n=n_boot),
        "intra_mine_niveau": _boot(level, _within(m, "lP"), n=n_boot),
    }


def radar_report(df: pd.DataFrame | None = None, n_boot: int = 1000) -> dict:
    """Rapport radar complet, y compris le groupe de test que cite calibration.json (radar_report.json).

    Un seul nombre de tirages bootstrap pour tout le rapport : les intervalles publiés se recalculent
    à l'identique depuis le dépôt.
    """
    df = radar_panel() if df is None else df
    out = {"mines": int(df["id"].nunique()), "mine_annees": int(len(df)), "n_boot": n_boot}
    res = {}
    for m in RADAR_METRICS:
        d = df.dropna(subset=["dP", f"d_{m}"])
        res[m] = {
            # même mine, années différentes : plus d'activité radar ↔ plus de production ?
            "intra_mine_niveau": _boot(df, _within(m, "lP"), n=n_boot),
            # variation d'une année sur l'autre
            "variation": _boot(d, _rho(f"d_{m}"), n=n_boot),
            "bon_sens": _boot(d, _direction(f"d_{m}"), n=n_boot),
            # en coupe : intensité × taille de l'emprise vs tonnage
            "coupe_x_emprise": _boot(
                df[df["serie"].isin(["minerai extrait", "charbon"])].assign(
                    xm=lambda x, m=m: x[m].clip(lower=0) * x["emprise"], dP=lambda x: x["lP"]),
                _rho("xm"), n=n_boot),
        }
    out["metriques"] = res

    # Le radar apporte-t-il quelque chose AU-DELÀ de la taille de l'emprise ?
    # Erreur de prédiction en validation croisée mine par mine : emprise seule vs emprise + radar.
    cs = df[df["serie"].isin(["minerai extrait", "charbon"])].groupby("id").agg(
        lP=("lP", "mean"), emprise=("emprise", "first"), **{m: (m, "median") for m in RADAR_METRICS})
    out["emprise_seule_coupe"] = _boot(cs.reset_index().assign(dP=lambda x: x["lP"],
                                                               le=lambda x: np.log(x["emprise"])), _rho("le"),
                                        n=n_boot)
    out["gain_radar_coupe"] = {m: _loo_gain(cs, m) for m in ("excess3", "excess2", "mine_frac3")}

    tune, test = df[df["groupe"] == "réglage"], df[df["groupe"] == "test"]
    best = max(RADAR_METRICS, key=lambda m: np.nan_to_num(_within(m, "lP")(tune), nan=-9))
    out["choix_sur_reglage"] = {"metrique": best, "rho_reglage": _within(best, "lP")(tune),
                                "mines_reglage": int(tune["id"].nunique())}
    # vérification sur l'autre moitié : seule source des chiffres radar de calibration.json
    out["groupe_test"] = _radar_test(test, best, n_boot)
    (OUT / "radar_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    return out


def radar_holdout(df: pd.DataFrame | None = None, n_boot: int = 2000) -> dict:
    """Validation sur les mines qui n'ont PAS servi à choisir la métrique (radar_choix.json).

    Erreurs de prédiction du tonnage en validation croisée (une mine retirée à la fois), emprise seule
    vs emprise + radar ; intervalle à 90 % du gain par bootstrap sur les mines.
    """
    df = pd.read_parquet(OUT / "radar_panel.parquet") if df is None else df
    choice = json.loads((OUT / "radar_choix.json").read_text())
    m = choice["metrique_choisie"]
    hold = df[~df["id"].isin(choice["mines_de_choix"]) & df["serie"].isin(["minerai extrait", "charbon"])]
    cs = hold.groupby("id").agg(lP=("lP", "mean"), emprise=("emprise", "first"), x=(m, "median")).dropna()
    y = cs["lP"].to_numpy() / np.log(10)
    xa, xr = np.log10(cs["emprise"].to_numpy()), np.log10(cs["x"].clip(lower=1e-4).to_numpy())
    one = np.ones(len(y))

    def loo_err(X):
        e = np.empty(len(y))
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            beta = np.linalg.lstsq(X[k], y[k], rcond=None)[0]
            e[i] = abs(y[i] - X[i] @ beta)
        return e

    ea, ear = loo_err(np.c_[one, xa]), loo_err(np.c_[one, xa, xr])
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        gains.append(10 ** np.median(ea[i]) / 10 ** np.median(ear[i]))
    beta = np.linalg.lstsq(np.c_[one, xa, xr], y, rcond=None)[0]
    out = {
        "metrique": m, "mines_validation": int(len(y)),
        "emprise_seule": {"facteur_median": float(10 ** np.median(ea)),
                          "facteur_90": float(10 ** np.percentile(ea, 90))},
        "emprise_plus_radar": {"facteur_median": float(10 ** np.median(ear)),
                               "facteur_90": float(10 ** np.percentile(ear, 90))},
        "gain_erreur_mediane": {"facteur": float(10 ** np.median(ea) / 10 ** np.median(ear)),
                                "ic90": [float(np.percentile(gains, 5)), float(np.percentile(gains, 95))]},
        "part_mines_mieux_predites": float((ear < ea).mean()),
        "modele": {"intercept": float(beta[0]), "pente_emprise": float(beta[1]),
                   "pente_radar": float(beta[2])},
    }
    (OUT / "radar_holdout.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# -- 7. modèle « emprise Maus → production » (pour le screener, toutes les mines) --------------------
def footprint_points() -> pd.DataFrame:
    """Les mines du modèle d'emprise : id · type (minerai | charbon) · P (t/an, médiane 2017–2020) · emprise.

    Séparé du modèle pour que les graphiques (scripts/figures.py) montrent exactement ces points-là.
    """
    import geopandas as gpd

    cat = catalog.load()
    cat = cat[cat["jasansky"].notna() & cat["footprint_km2"].notna()]
    prod = catalog.all_production()
    f = gpd.read_file(catalog.SRC / "jasansky2023" / "x" / "data" / "facilities.gpkg")
    eq = _equipment(f)
    rows = []
    for r in cat.itertuples():
        if _underground_only(eq.get(r.jasansky)):
            continue
        p = prod[(prod["facility_id"] == r.jasansky) & prod["year"].between(2017, 2020)]
        for serie, key in (("minerai extrait", "minerai"), ("charbon", "charbon")):
            v = p.loc[(p["serie"] == serie) & (p["value_tonnes"] > 0), "value_tonnes"]
            if len(v):
                rows.append({"id": r.id, "type": key, "P": float(v.median()), "emprise": r.footprint_km2})
                break
    return pd.DataFrame(rows, columns=["id", "type", "P", "emprise"])


def footprint_model(seed: int = 0) -> dict:
    """log10(tonnes/an) ~ log10(emprise Maus km²), par type (minerai extrait, charbon).

    Emprise Maus = contours tracés sur Sentinel-2 2019 → production médiane 2017–2020. Mines purement
    souterraines exclues. Erreur mesurée deux fois : validation croisée (une mine retirée à la fois) et
    moitié tenue à l'écart (jamais vue par l'ajustement).
    """
    d = footprint_points()
    out = {}
    rng = np.random.default_rng(seed)
    for key, g in d.groupby("type"):
        x, y = np.log10(g["emprise"].to_numpy()), np.log10(g["P"].to_numpy())
        X = np.c_[np.ones(len(g)), x]
        loo = np.empty(len(g))
        for i in range(len(g)):
            k = np.arange(len(g)) != i
            loo[i] = abs(y[i] - X[i] @ np.linalg.lstsq(X[k], y[k], rcond=None)[0])
        half = rng.random(len(g)) < 0.5
        beta_h = np.linalg.lstsq(X[half], y[half], rcond=None)[0]
        held = np.abs(y[~half] - X[~half] @ beta_h)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        out[key] = {"intercept": float(beta[0]), "slope": float(beta[1]), "n": int(len(g)),
                    "rho": float(spearmanr(x, y).statistic),
                    "facteur_median": float(10 ** np.median(loo)),
                    "facteur_90": float(10 ** np.percentile(loo, 90)),
                    "moitie_tenue_a_l_ecart": {"n": int((~half).sum()),
                                               "facteur_median": float(10 ** np.median(held)),
                                               "facteur_90": float(10 ** np.percentile(held, 90))},
                    "emprise_min_km2": float(g["emprise"].min()),
                    "emprise_max_km2": float(g["emprise"].max())}
    return out


def save_footprint_model() -> dict:
    """Ajoute le modèle « emprise → production » à src/strate/data/calibration.json."""
    summary = _read_summary()
    summary["modele_emprise"] = footprint_model()
    _write_summary(summary)
    return summary["modele_emprise"]


# Valeur « hasard » de chaque test radar : corrélations nulles, une chance sur deux pour le sens
RADAR_CHANCE = {"variation": 0.0, "intra_mine_niveau": 0.0,
                "bon_sens": 0.5, "bon_sens_grands_mouvements": 0.5}


def radar_verdict(test: dict) -> str:
    """Verdict DÉDUIT des intervalles à 90 % du groupe de test (jamais écrit à la main).

    « non démontré » tant que chaque intervalle contient la valeur du hasard ; un intervalle manquant
    (trop peu de données) compte comme non concluant.
    """
    above, below = [], []
    for k, chance in RADAR_CHANCE.items():
        _, lo, hi = (float(v) for v in test[k])
        if lo > chance:
            above.append(k)
        elif hi < chance:
            below.append(k)
    if above and len(above) == len(RADAR_CHANCE):
        return "validé : tous les intervalles à 90 % excluent le hasard"
    if above:
        return f"expérimental : mieux que le hasard sur {', '.join(above)} seulement (IC 90 %)"
    if below:
        return f"non démontré : {', '.join(below)} en sens contraire du lien attendu (IC 90 %)"
    return "non démontré : aucun intervalle à 90 % n'exclut le hasard"


def radar_size_verdict(gain_ic90: list[float]) -> str:
    """Le radar améliore-t-il l'estimation du tonnage ? gain = erreur emprise seule / emprise + radar."""
    lo, hi = (float(v) for v in gain_ic90)
    if lo > 1:
        return "utile : améliore l'estimation de production (IC 90 % du gain > 1)"
    if hi < 1:
        return "rejeté : dégrade l'estimation de production (IC 90 % du gain < 1)"
    return "rejeté : n'améliore pas l'estimation de production (IC 90 % du gain inclut 1)"


def save_radar_summary(rep: dict | None = None) -> dict:
    """Bloc « radar » de calibration.json, lu dans radar_report.json et radar_holdout.json.

    Aucun chiffre calculé ici : tout vient des rapports écrits par radar_report() et radar_holdout(),
    et les verdicts sont déduits des intervalles.
    """
    rep = json.loads((OUT / "radar_report.json").read_text()) if rep is None else rep
    t = rep["groupe_test"]
    hold = json.loads((OUT / "radar_holdout.json").read_text())
    gain = hold["gain_erreur_mediane"]
    summary = _read_summary()
    summary["radar"] = {
        "metrique_temporelle": t["metrique"], "mines_test": t["mines"], "n_boot": t["n_boot"],
        "variation": t["variation"], "bon_sens": t["bon_sens"],
        "bon_sens_grands_mouvements": t["bon_sens_grands_mouvements"],
        "intra_mine_niveau": t["intra_mine_niveau"],
        "taille": {"gain_erreur": gain, "mines": hold["mines_validation"],
                   "verdict": radar_size_verdict(gain["ic90"])},
        "verdict": radar_verdict(t),
    }
    _write_summary(summary)
    return summary["radar"]


# -- 8. tout régénérer -----------------------------------------------------------------------------
def report_all(say=print) -> dict:
    """Régénère TOUT ce que cite docs/calibration.md, depuis les caches satellite (aucun téléchargement).

    Ordre : sélection recalée → panel optique → rapport → synthèse → modèle d'emprise → panel radar →
    rapport radar → validation radar → synthèse radar.
    """
    sel = refresh_selection()
    kept = _retained(sel)
    say(f"sélection : {len(kept)}/{len(sel)} mines retenues, {len(sel) - len(kept)} exclues")
    df = panel(sel=sel)
    say(f"panel optique : {df['id'].nunique()} mines, {len(df)} lignes")
    rep = report(df)
    save_summary()
    save_footprint_model()
    out = {"optique": rep}
    rp = radar_panel(sel=sel)
    if rp.empty:  # sans cache radar, le bloc radar de calibration.json n'est PAS à jour : le dire
        log.warning("radar : aucun passage en cache, bloc « radar » de calibration.json non régénéré")
        return out
    say(f"panel radar : {rp['id'].nunique()} mines, {len(rp)} lignes")
    out["radar"] = radar_report(rp)
    out["radar_holdout"] = radar_holdout(rp)
    out["radar_resume"] = save_radar_summary(out["radar"])
    say(f"radar : {out['radar_resume']['verdict']}")
    return out
