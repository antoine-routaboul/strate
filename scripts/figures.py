"""Graphiques du README : `uv run python scripts/figures.py` (après `strate calibrate report`).

Deux figures, chacune en version claire et sombre (GitHub affiche celle du thème du lecteur) :
  docs/img/emprise-production[-sombre].png      la taille de l'emprise donne l'échelle de production
  docs/img/variations-satellite[-sombre].png    le satellite ne suit pas les variations annuelles
Les points sont exactement ceux des modèles publiés (calibrate.footprint_points, panel de calibration).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from strate import calibrate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
CAL = json.loads((ROOT / "src" / "strate" / "data" / "calibration.json").read_text("utf-8"))

# palette de référence (validée daltonisme) : rôles, pas de couleurs en dur dans le tracé
THEMES = {
    "clair": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
              "grid": "#e6e5e1", "s1": "#2a78d6", "s2": "#eb6834", "suffix": ""},
    "sombre": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e88",
               "grid": "#2f2f2d", "s1": "#3987e5", "s2": "#d95926", "suffix": "-sombre"},
}
SOURCE_EMPRISE = "Strate · données : Maus et al. 2022 (emprises), Jasansky et al. 2023 (production)"
SOURCE_VARIATIONS = ("Strate · données : Copernicus Sentinel-2 L2A (surface nue), "
                     "Jasansky et al. 2023 (production)")


def fr(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def boot_rho(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, n: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    """IC 90 % de la corrélation de rang, en rééchantillonnant les mines (pas les points)."""
    rng = np.random.default_rng(seed)
    g = np.arange(len(x)) if groups is None else groups
    ids = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ids}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(ids, len(ids))])
        vals.append(spearmanr(x[pick], y[pick]).statistic)
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def frame(t: dict, figsize=(10, 6.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def titles(fig, t, title: str, subtitle: str, source: str):
    fig.text(0.07, 0.955, title, fontsize=14, fontweight="bold", color=t["text"], ha="left", va="top")
    fig.text(0.07, 0.905, subtitle, fontsize=9.5, color=t["text2"], ha="left", va="top", linespacing=1.5)
    fig.text(0.07, 0.02, source, fontsize=7.5, color=t["muted"], ha="left", va="bottom")


def figure_emprise(t: dict) -> Path:
    d = calibrate.footprint_points()
    models = CAL["modele_emprise"]
    fig, ax = frame(t)
    parts = []
    for key, col, label in (("minerai", t["s1"], "Minerai extrait"), ("charbon", t["s2"], "Charbon")):
        g = d[d["type"] == key]
        x, y = g["emprise"].to_numpy(), g["P"].to_numpy() / 1e6
        rho = spearmanr(x, y).statistic
        lo, hi = boot_rho(x, y)
        parts.append(f"{label} : ρ = {fr(rho)} (IC 90 % {fr(lo)} – {fr(hi)}), n = {len(g)} mines")
        ax.scatter(x, y, s=26, color=col, alpha=0.75, edgecolors=t["surface"], linewidths=0.8,
                   label=f"{label} ({len(g)} mines)", zorder=3)
        m = models[key]
        xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
        ax.plot(xs, 10 ** (m["intercept"] + m["slope"] * np.log10(xs)) / 1e6, color=col, linewidth=2,
                zorder=4)
        ax.annotate(label.split()[0].lower(), (xs[-1], 10 ** (m["intercept"] + m["slope"]
                    * np.log10(xs[-1])) / 1e6), xytext=(6, 0), textcoords="offset points",
                    color=t["text2"], fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xlabel("Emprise de la mine sur image satellite (km², échelle log)", color=t["text2"], fontsize=9.5)
    ax.set_ylabel("Production annuelle (millions de tonnes, échelle log)", color=t["text2"], fontsize=9.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=t["text"])
    for h in leg.legend_handles:
        h.set_alpha(1)
    titles(fig, t, "Plus une mine est grande vue du ciel, plus elle produit",
           " · ".join(parts[:1]) + "\n" + parts[1]
           + "  —  erreur typique d'estimation : × " + fr(models["minerai"]["facteur_median"], 1)
           + " (minerai), × " + fr(models["charbon"]["facteur_median"], 1) + " (charbon)", SOURCE_EMPRISE)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.80, bottom=0.12)
    out = IMG / f"emprise-production{t['suffix']}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def figure_variations(t: dict) -> Path:
    gi = calibrate.GRID.index({"ndvi_bare": 0.2, "swir_change": 0.08})
    p = pd.read_parquet(ROOT / "data" / "calibration" / "panel.parquet")
    p = p[p["grid"] == gi].dropna(subset=["d_bare", "dP"])
    x, y = p["d_bare"].to_numpy(), p["dP"].to_numpy()
    # chiffres officiels (calibration.json) : un seul tirage de bootstrap fait foi, celui publié
    var = CAL["variation"]
    rho, (lo, hi), same = var["rho"], var["ic90"], var["bon_sens"][0]
    lim = np.log(2.0)  # ±100 % / −50 % : la grande majorité des points ; le reste est épinglé au bord
    clipped = int(((np.abs(x) > lim) | (np.abs(y) > lim)).sum())
    fig, ax = frame(t, figsize=(10, 6.9))
    ax.axhline(0, color=t["muted"], linewidth=1, zorder=2)
    ax.axvline(0, color=t["muted"], linewidth=1, zorder=2)
    ax.scatter(np.clip(x, -lim, lim), np.clip(y, -lim, lim), s=26, color=t["s1"], alpha=0.7,
               edgecolors=t["surface"], linewidths=0.8, zorder=3)
    pad = 0.04
    quadrants = ((1, 1, "right", "top", "même sens"), (-1, -1, "left", "bottom", "même sens"),
                 (-1, 1, "left", "top", "sens opposé"), (1, -1, "right", "bottom", "sens opposé"))
    for (qx, qy, ha, va, txt) in quadrants:
        ax.text(qx * (lim - pad), qy * (lim - pad), txt, ha=ha, va=va, color=t["muted"], fontsize=9)
    ticks = np.log([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2])
    lab = [f"{(np.exp(v) - 1) * 100:+.0f} %".replace("+0 %", "0") for v in ticks]
    ax.set_xticks(ticks, lab)
    ax.set_yticks(ticks, lab)
    ax.set_xlim(-lim * 1.03, lim * 1.03)
    ax.set_ylim(-lim * 1.03, lim * 1.03)
    ax.set_xlabel("Variation annuelle de la surface nue, Sentinel-2 (échelle log)", color=t["text2"],
                  fontsize=9.5)
    ax.set_ylabel("Variation annuelle de la production réelle (échelle log)", color=t["text2"], fontsize=9.5)
    titles(fig, t, "Vu du ciel, on ne voit pas la production monter ou baisser",
           f"ρ = {fr(rho)} (IC 90 % {fr(lo)} – {fr(hi)}), n = {len(p)} paires d'années sur "
           f"{p['id'].nunique()} mines, 2016–2019\n"
           f"Quand la production bouge de plus de 10 %, le satellite va dans le même sens dans "
           f"{fr(same * 100, 0)} % des cas (le hasard donne 50 %)"
           + (f"\n{clipped} points au-delà de −50 % / +100 % sont épinglés au bord du cadre"
              if clipped else ""),
           SOURCE_VARIATIONS)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.77, bottom=0.12)
    out = IMG / f"variations-satellite{t['suffix']}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for t in THEMES.values():
        print(figure_emprise(t))
        print(figure_variations(t))


if __name__ == "__main__":
    main()
