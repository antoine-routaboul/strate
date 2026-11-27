"""Étape 3 de la veille : le corrélateur.

Trois tâches, chacune bornée pour qu'un cycle de 15 min rende toujours la main :

1. **Suivi automatique** — les mines dont on parle (événements de confiance ≥ 0,5 sur 30 jours) entrent
   dans `tracked`, au plus `new_per_day` par jour : sans plafond, une vague d'actualité (un accident
   repris par 500 journaux) remplirait la file d'analyses satellite pour des semaines.
2. **Analyses satellite** — au plus `analyses_per_run` mines suivies par cycle passent par les modules
   (optique, attention, radar), comme le bouton « Analyser » de l'interface. Une mine analysée apparaît
   d'elle-même dans le radar et le screener.
3. **Étude d'événements** — au plus toutes les `recompute_hours` heures : les événements d'un type
   coïncident-ils avec des anomalies satellite plus souvent que le hasard ? Statistique de Cochran–
   Mantel–Haenszel stratifiée par mine, p-valeur par rotation circulaire des événements dans chaque mine
   (les mois d'une mine ne sont pas indépendants), rapport de risques de Mantel–Haenszel avec IC 90 % par
   bootstrap sur les mines, q-valeurs de Benjamini-Hochberg. Seul le décalage 0 peut conclure ; les
   décalages ±1 à ±3 mois sont exploratoires. Aucune IA : que des comptages et des rangs (texte
   complet : `METHODE`).

Rien n'est affirmé sans données suffisantes (5 mines informatives, 10 mois-événements) ni sans correction des
comparaisons multiples, et seuls comptent les mois où l'actualité était réellement collectée.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from strate.veille import store

STAGE = "correlation"
MIN_CONF = 0.5  # en dessous, le rattachement à la mine ou le type est trop incertain pour compter
TRACK_WINDOW_DAYS = 30
ANALYSIS_REFRESH_DAYS = 30  # composites mensuels : réanalyser plus souvent n'apporte aucun mois nouveau
ERROR_RETRY_HOURS = 24  # une erreur (serveur d'images saturé…) est retentée le lendemain
STUCK_HOURS = 12  # « en cours » depuis plus longtemps = processus tué en route : on relance
LAGS = range(-3, 4)  # satellite au mois de l'événement + décalage
PRIMARY_LAG = 0  # seule famille qui peut conclure ; les autres décalages sont exploratoires
MIN_MINES = 5
MIN_EVENT_MONTHS = 10
MIN_MONTHS_RHO = 6  # mois appariés minimum pour qu'une mine entre dans la corrélation de rang
Q_ALERT = 0.10
N_BOOT = 500
KEEP_RUNS = 28  # ~une semaine d'historique à 4 calculs par jour : la base ne grossit pas sans fin
META_LAST = "correlation_last_run"

DEMONTRE, NON_DEMONTRE, EXPLORATOIRE, TROP_PEU = (
    "lien démontré", "non démontré", "exploratoire", "trop peu de données")

# Même chaîne que le bouton « Analyser » (api.STEPS) : on ne l'importe pas, car importer l'API démarre
# son fil de travail et charge FastAPI dans le service de veille.
STEPS = (
    ("optical", date(2020, 1, 1), "images satellite optiques"),
    ("attention", date(2018, 1, 1), "attention Wikipedia"),
    ("radar", date(2020, 1, 1), "radar Sentinel-1"),
)

METHODE = (
    "Panel mine × mois des mines suivies ou analysées (une « jumelle » du catalogue est ramenée à la mine "
    "suivie). Anomalie = alerte satellite persistante retenue par le dossier Mine (|z| ≥ 2 deux mois de "
    "suite, par rapport à la même saison des années précédentes), signaux optiques et radar seulement. "
    "Événement = au moins un article du type ce mois-là (confiance ≥ 0,5). Couverture : pour chaque mine, "
    "seuls comptent les mois où l'actualité était réellement collectée, depuis le premier mois du flux "
    "GKG traité, ou, si l'historique de la mine est fait et que le normaliseur reconnaît l'un des noms "
    "cherchés dans un titre, depuis l'article le plus ancien renvoyé par l'API documentaire (jamais depuis "
    "le début de la frise de volume). "
    "Test : statistique de Cochran–Mantel–Haenszel stratifiée par mine (un tableau 2×2 par mine : mois "
    "d'événement ou non × anomalie ou non ; somme des anomalies tombant sur un mois d'événement). Les mois "
    "d'une mine ne sont pas indépendants (une anomalie retenue dure au moins deux mois, les événements "
    "arrivent en grappes) : mélanger les mois, comme le CMH exact, rejetterait trop souvent. La p-valeur "
    "bilatérale vient donc de la rotation circulaire de la frise des événements dans chaque mine (chaque "
    "décalage modulo la durée est également probable sous l'hypothèse nulle ; l'autocorrélation des deux "
    "frises est conservée), loi exacte par convolution des mines. Simulation sous l'hypothèse nulle "
    "(frises de Markov persistantes, 5 à 20 mines informatives, 24 à 60 mois) : 6 à 8 % de faux « lien "
    "démontré » pour un test, 7 à 10 % de calculs avec au moins un faux « lien démontré » sur 24 tests ; "
    "le CMH exact en donnait 15 à 38 % pour un test. Limite : la rotation suppose les événements "
    "stationnaires dans la fenêtre couverte ; une tendance commune aux deux frises n'est pas corrigée. "
    "La p-valeur du CMH exact (mois supposés indépendants) reste dans les détails. "
    "Effet (lift) = rapport de risques de Mantel–Haenszel, correction de "
    "continuité de 0,5 sur chaque case quand aucune anomalie ne tombe hors des mois d'événement (une "
    "association parfaite donne un lift fini et élevé) ; IC 90 % par bootstrap sur les mines (500 "
    "tirages). Coefficient de corrélation de chaque type : phi intra-mine entre mois d'événement (0/1) et "
    "mois d'anomalie (0/1), sommes centrées dans chaque mine (Σ(a − n_e·K/T) / √(Σ n_e(T−n_e)/T · "
    "Σ K(T−K)/T)), IC 90 % par les mêmes tirages de mines ; son numérateur est la statistique du test, "
    "d'où la même p-valeur. Il a son propre verdict (IC hors de 0), jamais recopié de celui du lift. "
    "Volume d'actualité ↔ écart du signal (|z|) : corrélation de rang centrée dans chaque mine, p-valeur "
    "par loi normale de même moyenne et même variance que la rotation circulaire de la frise du volume "
    "dans chaque mine (pas le mélange des mois, pour la même raison), même bootstrap. "
    "Famille principale = décalage 0 (types × signaux, plus le volume) : q-valeurs de "
    "Benjamini-Hochberg sur cette famille ; lien démontré si q < 0,10 et IC hors de la valeur nulle "
    "(1 pour le lift, 0 pour la corrélation). Décalages −3 à +3 mois (satellite après l'événement si "
    "positif) : exploratoires, avec leur propre correction de Benjamini-Hochberg, jamais déclarés "
    "démontrés. Seules comptent les mines informatives (au moins un mois d'événement et au moins une "
    "anomalie, aucun des deux tous les mois) : elles seules entrent dans le test, le lift, le phi et le "
    "bootstrap, et n_mines les compte. Trop peu de données sous 5 mines informatives ou 10 "
    "mois-événements de ces mines : ni q-valeur ni verdict."
)

Runner = Callable[[str], None]
DossierFn = Callable[[str], pd.DataFrame]


def _utc(dt: datetime | None) -> datetime:
    return (dt or datetime.now(UTC)).astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _midx(y: int, m: int) -> int:
    """Mois en entier (AAAA*12 + mois-1) : les décalages deviennent de simples additions."""
    return y * 12 + m - 1


def _midx_str(s: str) -> int:
    return _midx(int(s[:4]), int(s[5:7]))


# -- jumelles du catalogue --------------------------------------------------------------------------
def _twin_map() -> dict[str, str]:
    """Jumelle du catalogue → mine suivie (la même mine sous un id ICMM/Wikidata).

    Le normaliseur peut rattacher un article à la jumelle : sans ce renvoi, la même mine serait suivie et
    analysée deux fois (comme l'évite api._resolve), et ses signaux rangés sous le mauvais id.
    """
    try:
        from strate import screener

        screener.sync()
        return {twin: followed for followed, twin in screener.twins().items() if twin != followed}
    except Exception:  # catalogue absent ou illisible : chaque id reste lui-même
        return {}


def _canon(twins: dict[str, str]) -> Callable[[str], str]:
    return lambda sid: twins.get(sid, sid)


# -- 1. suivi automatique ---------------------------------------------------------------------------
def _merge_twins(con: sqlite3.Connection, canon: Callable[[str], str]) -> None:
    """Une jumelle déjà suivie (avant ce renvoi) devient la mine suivie, ou disparaît si celle-ci l'est."""
    ids = [r[0] for r in con.execute("SELECT site_id FROM tracked ORDER BY site_id")]
    present = set(ids)
    for sid in ids:
        c = canon(sid)
        if c == sid:
            continue
        if c in present:
            con.execute("DELETE FROM tracked WHERE site_id = ?", (sid,))
        else:
            con.execute("UPDATE tracked SET site_id = ? WHERE site_id = ?", (c, sid))
            present.add(c)
        present.discard(sid)


def track(con: sqlite3.Connection, new_per_day: int = 20, now: datetime | None = None,
          twins: dict[str, str] | None = None) -> int:
    """Met en suivi les mines dont on parle ; rafraîchit les compteurs des mines déjà suivies."""
    now = _utc(now)
    canon = _canon(_twin_map() if twins is None else twins)
    _merge_twins(con, canon)
    since = _iso(now - timedelta(days=TRACK_WINDOW_DAYS))
    raw = con.execute(
        """SELECT site_id, MAX(site_name) AS site_name, COUNT(*) AS n, SUM(confidence) AS score,
                  MAX(published_at) AS last
           FROM events WHERE confidence >= ? AND published_at >= ? AND published_at <= ?
           GROUP BY site_id""",
        (MIN_CONF, since, _iso(now)),
    ).fetchall()
    agg: dict[str, dict] = {}
    for r in raw:  # une mine et sa jumelle ne font qu'une candidate
        c = canon(r["site_id"])
        a = agg.setdefault(c, dict(site_id=c, site_name=None, n=0, score=0.0, last=""))
        if r["site_id"] == c or a["site_name"] is None:
            a["site_name"] = r["site_name"] or a["site_name"]
        a["n"] += r["n"]
        a["score"] += r["score"]
        a["last"] = max(a["last"], r["last"] or "")
    cands = sorted(agg.values(), key=lambda a: (-a["score"], -a["n"], a["site_id"]))
    tracked = {r[0] for r in con.execute("SELECT site_id FROM tracked")}
    today = now.strftime("%Y-%m-%d")
    used = con.execute("SELECT COUNT(*) FROM tracked WHERE substr(since, 1, 10) = ?", (today,)).fetchone()[0]
    room = max(0, new_per_day - used)
    added = 0
    for r in cands:
        if added >= room:
            break
        if r["site_id"] in tracked:
            continue
        reason = (f"actualité : {r['n']} événement{'s' if r['n'] > 1 else ''} en {TRACK_WINDOW_DAYS} jours "
                  f"(confiance moyenne {r['score'] / r['n']:.2f})")
        con.execute(
            """INSERT INTO tracked(site_id, site_name, reason, since, last_event_at, events_count,
                                   analysis_status) VALUES (?,?,?,?,?,?, 'en file')""",
            (r["site_id"], r["site_name"], reason, _iso(now), r["last"] or None, r["n"]),
        )
        added += 1
    # compteurs sur tout l'historique (pas seulement 30 jours) : c'est ce que montre la page Veille
    totals: dict[str, list] = {}
    for sid, n, last in con.execute("SELECT site_id, COUNT(*), MAX(published_at) FROM events "
                                    "WHERE confidence >= ? GROUP BY site_id", (MIN_CONF,)):
        t = totals.setdefault(canon(sid), [0, ""])
        t[0] += n
        t[1] = max(t[1], last or "")
    con.executemany(
        "UPDATE tracked SET events_count = ?, last_event_at = COALESCE(?, last_event_at) WHERE site_id = ?",
        [(totals.get(s, [0, ""])[0], totals.get(s, [0, ""])[1] or None, s)
         for s in (r[0] for r in con.execute("SELECT site_id FROM tracked"))],
    )
    con.commit()
    return added


# -- 2. analyses satellite --------------------------------------------------------------------------
def analyse_site(site_id: str) -> None:
    """Analyse complète d'une mine par les modules, en reprenant là où les données s'arrêtent.

    Les 3 derniers mois connus sont recalculés (composite d'un mois en cours, images arrivées en retard) ;
    sans cette reprise, réanalyser une mine chaque mois retéléchargerait six ans d'images. Une jumelle du
    catalogue est analysée sous l'id de la mine suivie, comme le fait l'API.
    """
    from strate.modules import REGISTRY  # import tardif : les tests n'en ont pas besoin

    site_id = _canon(_twin_map())(site_id)
    today = date.today()
    for mod, start, _label in STEPS:
        m = REGISTRY[mod]
        old = m.load(site_id)
        s = start
        if len(old):
            last = pd.Timestamp(old["date"].max()) - pd.DateOffset(months=3)
            s = max(start, last.date().replace(day=1))
        m.update(site_id, s, today)


def _due(row: sqlite3.Row, now: datetime) -> bool:
    status, at = row["analysis_status"] or "", _parse(row["analysis_at"])
    if at is None:
        return True
    age = now - at
    if status.startswith("erreur"):
        return age >= timedelta(hours=ERROR_RETRY_HOURS)
    if status.startswith("en cours"):
        return age >= timedelta(hours=STUCK_HOURS)
    return age >= timedelta(days=ANALYSIS_REFRESH_DAYS)


def analyse_pending(con: sqlite3.Connection, analyses_per_run: int = 2, runner: Runner | None = None,
                    now: datetime | None = None) -> dict:
    """Lance au plus `analyses_per_run` analyses ; jamais analysées d'abord, puis les plus citées."""
    now = _utc(now)
    runner = runner or analyse_site
    rows = con.execute(
        "SELECT * FROM tracked ORDER BY analysis_at IS NOT NULL, events_count DESC, since, site_id"
    ).fetchall()
    due = [r["site_id"] for r in rows if _due(r, now)]
    todo = due[:max(0, analyses_per_run)]
    done = errors = 0
    for sid in todo:
        # « en cours » validé tout de suite : l'API le montre pendant les minutes que prend l'analyse
        con.execute("UPDATE tracked SET analysis_status = 'en cours', analysis_at = ? WHERE site_id = ?",
                    (_iso(now), sid))
        con.commit()
        try:
            runner(sid)
            status = "terminé"
            done += 1
        except Exception as e:  # une mine en échec ne bloque pas les autres ; retentée plus tard
            status = f"erreur : {e}"[:300]
            errors += 1
        con.execute("UPDATE tracked SET analysis_status = ? WHERE site_id = ?", (status, sid))
        con.commit()
    return {"analyses": done, "analyses_erreurs": errors, "analyses_en_attente": len(due) - len(todo)}


# -- 3. étude d'événements --------------------------------------------------------------------------
def bh(p: list[float | None]) -> list[float | None]:
    """q-valeurs de Benjamini-Hochberg (les None, non testés, restent None)."""
    idx = [i for i, v in enumerate(p) if v is not None and np.isfinite(v)]
    out: list[float | None] = [None] * len(p)
    if not idx:
        return out
    pv = np.array([p[i] for i in idx], dtype=float)
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    for k, o in enumerate(order):
        out[idx[o]] = float(q[k])
    return out


def _default_candidates() -> set[str]:
    from strate.core import analysed
    from strate.modules import REGISTRY

    return {s for mod in REGISTRY for s in analysed(mod)}


def _sat_signals() -> list[str]:
    from strate.modules import FAMILY, SEASONAL_FAMILIES

    # l'attention Wikipedia est exclue : elle suit l'actualité par construction, ce n'est pas un capteur
    return [s for s, f in FAMILY.items() if f in SEASONAL_FAMILIES]


def coverage(con: sqlite3.Connection, canon: Callable[[str], str] | None = None) -> tuple[int | None,
                                                                                       dict[str, int]]:
    """Premier mois où l'actualité était réellement collectée : (flux GKG pour toutes les mines,
    {mine : premier mois de l'historique documentaire}).

    Le flux GKG ne voit que ce qui paraît depuis que la collecte tourne. L'historique documentaire d'une
    mine (« fait ») renvoie ses 250 articles les plus récents : la collecte de cette mine est complète
    depuis le plus ancien d'entre eux. La frise de volume, elle, remonte à 2017 sans qu'aucun article de
    ces années-là n'ait été lu : la prendre pour couverture compterait des années « sans événement »
    faute de collecte, et une simple tendance des alertes passerait pour un lien.
    """
    canon = canon or (lambda s: s)
    first = con.execute("SELECT MIN(stamp) FROM feed_files WHERE status = 'done'").fetchone()[0]
    ok = bool(first) and len(first) >= 6 and first[:6].isdigit()
    gkg = _midx(int(first[:4]), int(first[4:6])) if ok else None
    # historique « fait » : sous l'id suivi ou sous sa jumelle (une jumelle suivie avant le renvoi a pu
    # être renommée dans `tracked` alors que ses articles gardent l'ancien id dans hint_site). Seulement
    # si le normaliseur reconnaît l'un des noms cherchés (meta « doc_reconnu ») : sinon aucun article de
    # la requête ne devient un événement, et ses années compteraient « sans événement » faute de nom.
    reconnu = {canon(k.split(":", 1)[1]) for k, v in con.execute(
        "SELECT key, value FROM meta WHERE key LIKE 'doc_reconnu:%'") if v == "1"}
    fait = {canon(r[0]) for r in con.execute("SELECT site_id FROM tracked WHERE backfill_status = 'fait'")}
    fait &= reconnu
    doc: dict[str, int] = {}
    # 1re source : la date du plus ancien article renvoyé, notée par la collecte (survit à une purge)
    for key, oldest in con.execute("SELECT key, value FROM meta WHERE key LIKE 'doc_oldest:%'"):
        c = canon(key.split(":", 1)[1])
        if c in fait and oldest and len(oldest) >= 7 and oldest[:4].isdigit() and oldest[5:7].isdigit():
            m = _midx_str(oldest)
            doc[c] = min(m, doc.get(c, m))
    for sid, oldest in con.execute(
            """SELECT hint_site, MIN(published_at) FROM raw_docs
               WHERE source = 'gdelt_doc' AND hint_site IS NOT NULL AND published_at IS NOT NULL
               GROUP BY hint_site"""):
        c = canon(sid)
        if c in fait and oldest and len(oldest) >= 7 and oldest[:4].isdigit() and oldest[5:7].isdigit():
            m = _midx_str(oldest)
            doc[c] = min(m, doc.get(c, m))
    return gkg, doc


def _panel(con: sqlite3.Connection, dossier_fn: DossierFn, candidates: set[str] | None,
           now: datetime, signals: list[str], canon: Callable[[str], str], stats: dict) -> dict[str, dict]:
    """Par mine : anomalies et z par signal (indexés par mois entier), mois d'événement par type,
    volume d'actualité et premier mois couvert par la collecte."""
    ev = pd.read_sql_query(
        "SELECT site_id, event_type, month FROM events WHERE confidence >= ?", con, params=(MIN_CONF,))
    vol = pd.read_sql_query("SELECT site_id, month, n_articles, source FROM news_volume", con)
    ev["site_id"] = ev["site_id"].map(canon)
    vol["site_id"] = vol["site_id"].map(canon)
    gkg, doc = coverage(con, canon)
    sites = {canon(r[0]) for r in con.execute("SELECT site_id FROM tracked")}
    sites |= set(ev["site_id"]) | set(vol["site_id"])
    if candidates is not None:
        sites &= candidates
    cur = _midx(now.year, now.month)
    ev = ev[ev["month"].str.match(r"^\d{4}-\d{2}")]
    ev = ev.assign(m=ev["month"].map(_midx_str))
    ev = ev[ev["m"] <= cur]
    vol = vol[vol["month"].str.match(r"^\d{4}-\d{2}")]
    vol = vol.assign(m=vol["month"].map(_midx_str))
    vol = vol[vol["m"] <= cur]
    out: dict[str, dict] = {}
    for sid in sorted(sites):
        starts = [s for s in (gkg, doc.get(sid)) if s is not None]
        if not starts:
            stats["sans_couverture"] = stats.get("sans_couverture", 0) + 1
            continue  # aucune collecte : impossible de dire qu'il ne s'est « rien » passé
        try:
            d = dossier_fn(sid)
        except Exception:  # mine inconnue du catalogue, fichier illisible : elle sort du panel
            stats["dossiers_illisibles"] = stats.get("dossiers_illisibles", 0) + 1
            continue
        if d is None or d.empty or "date" not in d:
            continue
        months = np.array([_midx(t.year, t.month) for t in pd.to_datetime(d["date"])])
        sat = {}
        for s in signals:
            if f"a_{s}" not in d:
                continue
            # a_ = z retenu seulement s'il persiste (sinon 0) : l'anomalie ; z_ = écart brut : le volume
            a, z = (_series(d[c], months) for c in (f"a_{s}", f"z_{s}" if f"z_{s}" in d else f"a_{s}"))
            if len(a):
                sat[s] = (a, z)
        if not sat:
            continue
        e = ev[ev["site_id"] == sid]
        v = vol[vol["site_id"] == sid]
        if len(v):  # la frise de l'API documentaire est la mesure de référence ; sinon ce qui existe
            tl = v["source"] == "gdelt_doc_timeline"
            v = v[tl] if tl.any() else v[v["source"] == v["source"].iloc[0]]
        out[sid] = dict(
            sat=sat,
            ev={t: set(g["m"].astype(int)) for t, g in e.groupby("event_type")},
            # mine et jumelle peuvent avoir chacune leur frise : la plus grande valeur du mois, pas la somme
            vol=pd.Series(v["n_articles"].to_numpy(dtype=float), index=v["m"].astype(int).to_numpy())
            .groupby(level=0).max(),
            start=min(starts), end=cur,
        )
    return out


def _series(col: pd.Series, months: np.ndarray) -> pd.Series:
    s = pd.Series(col.to_numpy(dtype=float), index=months).dropna()
    return s[~s.index.duplicated(keep="last")]


def _ci(vals: np.ndarray) -> tuple[float | None, float | None]:
    n = len(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) < max(1, n // 2):  # trop de tirages indéfinis : l'intervalle ne voudrait rien dire
        return None, None
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi)


def _mh_parts(counts: np.ndarray) -> tuple[np.ndarray, ...]:
    """Contributions de chaque mine au rapport de risques de Mantel–Haenszel, brutes et corrigées (+0,5).

    counts = (mines × 4) [T mois, K anomalies, n_e mois-événements, a anomalies avec événement].
    RR_MH = Σ a·n₀/T / Σ c·n₁/T (n₁ = n_e, n₀ = T − n_e, c = K − a). Une mine sans mois d'événement, sans
    mois sans événement ou sans aucune anomalie n'apporte rien : ni au rapport, ni à la correction.
    """
    t, k, ne, a = (counts[:, i].astype(float) for i in range(4))
    n0, c = t - ne, k - a
    inf = (ne > 0) & (n0 > 0) & (k > 0)
    tt = np.where(t > 0, t, 1.0)
    num = np.where(inf, a * n0 / tt, 0.0)
    den = np.where(inf, c * ne / tt, 0.0)
    # 0,5 ajouté à chacune des 4 cases : T + 2, n₁ + 1, n₀ + 1
    numc = np.where(inf, (a + 0.5) * (n0 + 1) / (t + 2), 0.0)
    denc = np.where(inf, (c + 0.5) * (ne + 1) / (t + 2), 0.0)
    return num, den, numc, denc


def _mh_ratio(num, den, numc, denc):
    """Rapport brut si son dénominateur est non nul, sinon corrigé (0,5) ; NaN sans mine informative.

    La correction ne sert qu'à l'association parfaite (aucune anomalie hors des mois d'événement), où le
    rapport brut serait infini. Un numérateur nul donne un rapport brut de 0, fini et exact : le corriger
    le pousserait au-dessus de 1 (0,5 / (n₁ + 1) contre c / n₀), une absence totale de coïncidence
    passerait pour un lien positif.
    """
    num, den, numc, denc = (np.asarray(x, dtype=float) for x in (num, den, numc, denc))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = num / np.where(den > 0, den, 1.0)
        cor = np.where(denc > 0, numc / np.where(denc > 0, denc, 1.0), np.nan)
    return np.where(den > 0, raw, cor)


def _cmh_exact(counts: np.ndarray) -> dict:
    """Test de Cochran–Mantel–Haenszel, p-valeur exacte bilatérale.

    À marges fixées, le nombre d'anomalies tombant sur un mois d'événement suit dans chaque mine une loi
    hypergéométrique ; leur somme sur les mines a pour loi le produit de convolution de ces lois. La
    p-valeur somme les probabilités des totaux au plus aussi probables que l'observé (comme
    `mantelhaen.test(exact=TRUE)` de R) : pas de plancher de Monte-Carlo, pas d'approximation des
    petits effectifs. La statistique χ² de CMH (correction 0,5) est rendue pour information.
    """
    from scipy.stats import hypergeom

    dist = np.ones(1)
    offset = obs = 0
    expected = var = 0.0
    for t, k, ne, a in counts.astype(np.int64):
        if not (0 < ne < t and 0 < k < t):
            continue  # tableau dégénéré : son compte est imposé par les marges, il n'apprend rien
        lo, hi = max(0, ne + k - t), min(ne, k)
        dist = np.convolve(dist, hypergeom.pmf(np.arange(lo, hi + 1), t, k, ne))
        offset += lo
        obs += int(a)
        expected += ne * k / t
        var += ne * (t - ne) * k * (t - k) / (t * t * (t - 1))
    if var <= 0:
        return dict(p_value=None, expected=expected, chi2=None)
    f = dist[obs - offset]
    p = float(dist[dist <= f * (1 + 1e-7)].sum())
    chi2 = max(0.0, abs(obs - expected) - 0.5) ** 2 / var
    return dict(p_value=min(1.0, max(p, np.finfo(float).tiny)), expected=expected, chi2=chi2)


def _phi_parts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sommes centrées de chaque mine pour le phi intra-mine (mois d'événement 0/1 × mois d'anomalie 0/1).

    Dans une mine : Sxy = a − n_e·K/T (anomalies sur un mois d'événement moins leur nombre attendu, le
    terme même du test CMH), Sxx = n_e·(T − n_e)/T, Syy = K·(T − K)/T. Les mines non informatives
    (sans événement, sans anomalie, ou l'un des deux tous les mois) sont écartées en amont
    (`_informative`), comme pour le test.
    """
    t, k, ne, a = (counts[:, i].astype(float) for i in range(4))
    tt = np.where(t > 0, t, 1.0)
    return a - ne * k / tt, ne * (t - ne) / tt, k * (t - k) / tt


def _phi_ratio(sxy, sxx, syy):
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sqrt(np.asarray(sxx, dtype=float) * np.asarray(syy, dtype=float))
        return np.where(den > 0, np.asarray(sxy, dtype=float) / np.where(den > 0, den, 1.0), np.nan)


def _informative(counts: np.ndarray) -> np.ndarray:
    """Mines qui apportent quelque chose au test : 0 < n_e < T et 0 < K < T.

    Une mine sans mois d'événement, sans mois sans événement, sans anomalie ou en anomalie tous les mois
    a un compte imposé par ses marges : elle n'entre ni dans la statistique CMH, ni dans le lift, ni dans
    le phi. La compter parmi les « mines » du seuil MIN_MINES ou du bootstrap laisserait une seule mine
    informative produire un verdict (avec un IC réduit à un point).
    """
    t, k, ne = counts[:, 0], counts[:, 1], counts[:, 2]
    return (ne > 0) & (ne < t) & (k > 0) & (k < t)


def _rotation_counts(hit: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Anomalies tombant sur un mois d'événement pour chacune des T rotations circulaires des événements.

    Rotation s : l'événement du mois t est placé au mois t + s (modulo T). La rotation 0 est l'observé.
    """
    t = len(hit)
    idx = (np.arange(t)[:, None] + np.arange(t)[None, :]) % t
    return (ev.astype(bool)[idx] & hit.astype(bool)[None, :]).sum(axis=1)


def _rotation_p(vecs: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """p-valeur bilatérale exacte du test par rotation intra-mine (somme des anomalies sur événement).

    Les mois d'une mine ne sont pas indépendants : une anomalie retenue dure au moins deux mois
    (persistance |z| ≥ 2 deux mois de suite) et les événements arrivent en grappes. Mélanger les mois
    (loi hypergéométrique du test CMH exact) suppose qu'ils sont échangeables : sous persistance, ce test
    rejette trop souvent (simulation : 12 à 16 % au lieu de 10 % pour p < 0,10). Faire tourner la frise
    des événements de chaque mine garde son autocorrélation et celle des anomalies ; sous H0 (frises
    indépendantes, événements stationnaires), chacune des T rotations est également probable, d'où la
    loi exacte de Σ a : produit de convolution des lois des rotations de chaque mine. p = P(|S − E| ≥
    |obs − E|), E = Σ n_e·K/T.
    """
    dist = np.ones(1)
    obs, mean = 0, 0.0
    for hit, ev in vecs:
        a = _rotation_counts(hit, ev)
        if np.ptp(a) == 0:
            continue  # toutes les rotations donnent le même compte : la mine n'apprend rien
        dist = np.convolve(dist, np.bincount(a) / len(a))
        obs += int(a[0])
        mean += float(a.mean())
    if len(dist) == 1:
        return None
    x = np.arange(len(dist))
    p = float(dist[np.abs(x - mean) >= abs(obs - mean) - 1e-9].sum())
    return min(1.0, max(p, np.finfo(float).tiny))


def _lift_test(counts: np.ndarray, rng: np.random.Generator, n_boot: int,
               vecs: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """counts : (mines × 4) [T, K, n_e, a_e]. Lift de Mantel–Haenszel et phi intra-mine, IC 90 % par le
    MÊME bootstrap sur les mines (mêmes tirages pour les deux), p commune (le numérateur du phi
    intra-mine est la statistique du test, Σ(a − attendu)). Avec `vecs` (frises 0/1 anomalie, événement
    de chaque mine, dans l'ordre de `counts`) : p du test par rotation (`_rotation_p`), qui tient compte
    de la persistance ; sinon p du CMH exact (mois supposés indépendants), toujours rendue en `p_cmh`."""
    tot = counts.sum(axis=0)
    parts = _mh_parts(counts)
    sums = [float(x.sum()) for x in parts]
    lift = float(_mh_ratio(*sums))
    corrected = not sums[1] > 0
    m = len(counts)
    w = rng.multinomial(m, np.full(m, 1 / m), size=n_boot)  # mines tirées avec remise
    lo, hi = _ci(_mh_ratio(*(w @ x for x in parts)))
    pp = _phi_parts(counts)
    phi = float(_phi_ratio(*(float(x.sum()) for x in pp)))
    phi_lo, phi_hi = _ci(_phi_ratio(*(w @ x for x in pp)))
    test = _cmh_exact(counts)
    p = _rotation_p(vecs) if vecs is not None else test["p_value"]
    return dict(lift=lift if np.isfinite(lift) else None, lift_lo=lo, lift_hi=hi,
                rho=phi if np.isfinite(phi) else None, rho_lo=phi_lo, rho_hi=phi_hi, p_value=p,
                p_cmh=test["p_value"],
                p_event=float(tot[3] / tot[2]) if tot[2] else None,
                p_no_event=float((tot[1] - tot[3]) / (tot[0] - tot[2])) if tot[0] > tot[2] else None,
                expected=test["expected"], chi2=test["chi2"],
                corrected=bool(corrected and np.isfinite(lift)))


def _ranks(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    return r - r.mean()


def _rho_test(pairs: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator, n_boot: int) -> dict:
    """Corrélation de rang intra-mine : rangs centrés dans chaque mine, puis corrélation commune.

    Centrer par mine retire l'effet « grande mine = beaucoup d'articles et grand écart » : on ne mesure
    que les co-variations dans le temps, mine par mine. Le volume d'actualité et l'écart |z| sont
    autocorrélés : mélanger les mois sous-estimerait la variance de Σ x·y. Sous l'hypothèse nulle, on fait
    tourner la frise du volume dans chaque mine (T rotations également probables, autocorrélation
    conservée) : Σ x·y a pour moyenne 0 (rangs centrés) et pour variance la somme des variances de
    rotation de chaque mine ; la p-valeur vient de la loi normale de ces deux moments, sans plancher de
    tirages. `var_permutation` (mois mélangés, Σ Σx²·Σy² / (n − 1)) est rendue pour comparaison.
    """
    sxy, sxx, syy, var, var_perm = [], [], [], 0.0, 0.0
    for v, s in pairs:
        cv, cs = _ranks(v), _ranks(s)
        sxy.append(float(cv @ cs))
        sxx.append(float(cv @ cv))
        syy.append(float(cs @ cs))
        t = len(cv)
        rot = cv[(np.arange(t)[:, None] + np.arange(t)[None, :]) % t] @ cs  # Σ x(t+k)·y(t), k = 0…T−1
        var += float(np.mean(rot ** 2) - np.mean(rot) ** 2)
        var_perm += sxx[-1] * syy[-1] / (t - 1)
    sxy, sxx, syy = np.array(sxy), np.array(sxx), np.array(syy)
    rho = float(sxy.sum() / np.sqrt(sxx.sum() * syy.sum()))
    m = len(pairs)
    w = rng.multinomial(m, np.full(m, 1 / m), size=n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = (w @ sxy) / np.sqrt((w @ sxx) * (w @ syy))
    lo, hi = _ci(rb)
    zstat = float(sxy.sum() / math.sqrt(var)) if var > 0 else 0.0
    p = math.erfc(abs(zstat) / math.sqrt(2))
    zp = float(sxy.sum() / math.sqrt(var_perm)) if var_perm > 0 else 0.0
    return dict(rho=rho, rho_lo=lo, rho_hi=hi, p_value=min(1.0, max(p, np.finfo(float).tiny)),
                p_permutation=min(1.0, max(math.erfc(abs(zp) / math.sqrt(2)), np.finfo(float).tiny)))


def study(con: sqlite3.Connection, dossier_fn: DossierFn | None = None, candidates: set[str] | None = None,
          now: datetime | None = None, n_boot: int = N_BOOT, seed: int = 0, signals: list[str] | None = None,
          twins: dict[str, str] | None = None, stats: dict | None = None) -> list[dict]:
    """Calcule toutes les combinaisons et rend les lignes à écrire dans `correlations`.

    `stats` (facultatif) reçoit les mines écartées : sans couverture, dossier illisible.
    """
    now = _utc(now)
    stats = {} if stats is None else stats
    if dossier_fn is None:
        from strate import fusion

        dossier_fn = fusion.dossier
        if candidates is None:  # seules les mines déjà passées par un module ont des signaux
            candidates = _default_candidates()
    from strate.fusion import Z_ALERT

    signals = signals or _sat_signals()
    cache: dict[str, pd.DataFrame] = {}

    def cached(sid: str) -> pd.DataFrame:  # un dossier = des lectures parquet + des z : une fois par calcul
        if sid not in cache:
            cache[sid] = dossier_fn(sid)
        return cache[sid]

    canon = _canon(_twin_map() if twins is None else twins)
    panel = _panel(con, cached, candidates, now, signals, canon, stats)
    stats["mines_panel"] = len(panel)
    rng = np.random.default_rng(seed)  # graine fixe : mêmes données → mêmes résultats d'un calcul à l'autre
    rows: list[dict] = []
    for etype in store.EVENT_TYPES:
        for sig in signals:
            for lag in LAGS:
                counts, vecs = [], []
                for p in panel.values():
                    if sig not in p["sat"]:
                        continue
                    a = p["sat"][sig][0].sort_index()
                    ms = a.index.to_numpy() - lag  # mois de l'événement apparié à chaque mois satellite
                    keep = (ms >= p["start"]) & (ms <= p["end"])  # mois d'actualité réellement collectés
                    if not keep.any():
                        continue
                    hit = np.abs(a.to_numpy()[keep]) >= Z_ALERT
                    e = np.isin(ms[keep], list(p["ev"].get(etype, ())))
                    counts.append([keep.sum(), hit.sum(), e.sum(), (hit & e).sum()])
                    vecs.append((hit, e))
                c = np.array(counts, dtype=float).reshape(-1, 4)
                if not c[:, 2].sum():
                    continue  # type jamais vu sur ces mines : une ligne vide n'apprendrait rien
                # seules les mines informatives (mois d'événement ET anomalies, chacun avec son contraire)
                # entrent dans le test, le lift, le phi et le bootstrap : n_mines les compte, elles seules
                inf = _informative(c)
                ci = c[inf]
                n_mines, n_events = int(inf.sum()), int(ci[:, 2].sum())
                row = dict(event_type=etype, signal=sig, lag_months=lag, n_mines=n_mines, n_events=n_events,
                           lift=None, lift_lo=None, lift_hi=None, rho=None, rho_lo=None, rho_hi=None,
                           p_value=None)
                r = _lift_test(ci, rng, n_boot, [v for v, ok in zip(vecs, inf, strict=True) if ok]) \
                    if n_mines else dict(p_event=None, p_no_event=None, expected=None, chi2=None,
                                         corrected=False, p_cmh=None)
                det = dict(mines_panel=len(c), mines_avec_evenements=int((c[:, 2] > 0).sum()),
                           mois_evenements_total=int(c[:, 2].sum()), mines_informatives=n_mines,
                           mois_apparies=int(c[:, 0].sum()), mois_apparies_informatifs=int(ci[:, 0].sum()),
                           anomalies=int(c[:, 1].sum()),
                           test="rotation circulaire intra-mine (loi exacte)", n_boot=n_boot,
                           p_cmh_mois_independants=r.pop("p_cmh"),
                           p_anomalie_evenement=r.pop("p_event"),
                           p_anomalie_sans_evenement=r.pop("p_no_event"),
                           anomalies_attendues=r.pop("expected"), chi2_cmh=r.pop("chi2"),
                           correction_continuite=r.pop("corrected"))
                row.update(r)
                row["_tested"] = n_mines >= MIN_MINES and n_events >= MIN_EVENT_MONTHS
                det.update(coefficient_rho="phi intra-mine (mois d'événement × mois d'anomalie)",
                           unite_n="mois-événements des mines informatives", unite_bootstrap="mines",
                           n_unites_bootstrap=n_mines)
                row["details"] = det
                rows.append(row)
    for sig in signals:
        for lag in LAGS:
            pairs = []
            n_months = 0
            for p in panel.values():
                if sig not in p["sat"] or not len(p["vol"]):
                    continue
                z = p["sat"][sig][1].abs()
                vm = p["vol"].index.to_numpy()
                # même règle de couverture que pour les événements : seulement les mois où l'actualité
                # était réellement collectée (la frise de volume remonte à 2017 sans lecture d'articles)
                both = np.isin(vm + lag, z.index.to_numpy()) & (vm >= p["start"]) & (vm <= p["end"])
                if both.sum() < MIN_MONTHS_RHO:
                    continue
                v = p["vol"].to_numpy()[both]
                s = z.reindex(vm[both] + lag).to_numpy()
                if np.ptp(v) == 0 or np.ptp(s) == 0:
                    continue  # aucune variation dans la mine : aucun rang à comparer
                pairs.append((v, s))
                n_months += int((v > 0).sum())
            if not pairs:
                continue
            rt = _rho_test(pairs, rng, n_boot)
            p_perm = rt.pop("p_permutation")
            row = dict(event_type="volume", signal=sig, lag_months=lag, n_mines=len(pairs), n_events=n_months,
                       lift=None, lift_lo=None, lift_hi=None, **rt)
            row["_tested"] = len(pairs) >= MIN_MINES and n_months >= MIN_EVENT_MONTHS
            row["details"] = dict(mois_apparies=int(sum(len(v) for v, _ in pairs)), mesure="|z| du signal",
                                  test="rang intra-mine, loi normale de la rotation", n_boot=n_boot,
                                  p_permutation_mois_independants=p_perm,
                                  coefficient_rho="rho de Spearman intra-mine (volume × |z|)",
                                  unite_n="mois avec articles", unite_bootstrap="mines",
                                  n_unites_bootstrap=len(pairs))
            rows.append(row)
    # deux familles corrigées séparément : le décalage 0 (seul à pouvoir conclure) et les autres
    # (exploratoires). Les lignes sous le minimum de données ne sont pas des tests.
    for primary in (True, False):
        fam = [r for r in rows if (r["lag_months"] == PRIMARY_LAG) == primary]
        qs = bh([r["p_value"] if r["_tested"] else None for r in fam])
        for r, q in zip(fam, qs, strict=True):
            r["q_value"] = q
            r["details"]["famille"] = "principale" if primary else "exploratoire"
    for r in rows:
        r["verdict"] = _verdict(r)
        r["verdict_rho"] = _verdict(r, "rho")
        del r["_tested"]
    return rows


def _verdict(r: dict, coef: str | None = None) -> str:
    """Verdict d'UN coefficient, d'après son propre IC : `coef` = 'lift' (IC hors de 1) ou 'rho' (IC hors
    de 0 : rho du volume, phi d'un type). Par défaut : le lift pour un type, le rho pour le volume (colonne
    `verdict`). Mêmes règles pour les deux : décalage 0, assez de données, q < 0,10 (q du test commun)."""
    if r["q_value"] is None:
        return TROP_PEU
    if r["lag_months"] != PRIMARY_LAG:
        return EXPLORATOIRE  # calculé, montré, jamais affirmé
    coef = coef or ("rho" if r["event_type"] == "volume" else "lift")
    lo, hi = r.get(f"{coef}_lo"), r.get(f"{coef}_hi")
    null = 1.0 if coef == "lift" else 0.0
    outside = lo is not None and hi is not None and (lo > null or hi < null)
    return DEMONTRE if r["q_value"] < Q_ALERT and outside else NON_DEMONTRE


def write(con: sqlite3.Connection, rows: list[dict], run_at: str) -> int:
    """Écrit un calcul. Un calcul sans aucune combinaison efface les précédents : l'API lit le dernier
    `run_at`, et des verdicts d'un ancien calcul ne doivent pas passer pour l'état actuel."""
    if not rows:
        con.execute("DELETE FROM correlations WHERE scope = 'global'")
        return 0
    cols = ("event_type", "signal", "lag_months", "n_mines", "n_events", "lift", "lift_lo", "lift_hi",
            "rho", "rho_lo", "rho_hi", "p_value", "q_value", "verdict", "verdict_rho")
    con.executemany(
        f"INSERT OR REPLACE INTO correlations(run_at, scope, {', '.join(cols)}, details) "
        f"VALUES (?, 'global', {', '.join('?' * len(cols))}, ?)",
        [(run_at, *(r[c] for c in cols), json.dumps(r["details"], ensure_ascii=False)) for r in rows],
    )
    runs = [x[0] for x in con.execute("SELECT DISTINCT run_at FROM correlations ORDER BY run_at DESC")]
    if len(runs) > KEEP_RUNS:
        con.execute("DELETE FROM correlations WHERE run_at < ?", (runs[KEEP_RUNS - 1],))
    return len(rows)


def _last_run(con: sqlite3.Connection) -> datetime | None:
    r = con.execute("SELECT value FROM meta WHERE key = ?", (META_LAST,)).fetchone()
    return _parse(r[0]) if r else None


# -- point d'entrée ---------------------------------------------------------------------------------
def run_once(con: sqlite3.Connection, new_per_day: int = 20, analyses_per_run: int = 2,
             recompute_hours: float = 6, *, runner: Runner | None = None, dossier_fn: DossierFn | None = None,
             candidates: set[str] | None = None, now: datetime | None = None, n_boot: int = N_BOOT,
             twins: dict[str, str] | None = None) -> dict:
    """Un cycle : suivi → analyses → (si dû) étude d'événements. Chaque partie isole ses erreurs.

    `runner` (analyse d'une mine), `dossier_fn` (table mensuelle des signaux) et `twins` (jumelle → mine
    suivie) sont injectables : les tests n'appellent jamais les serveurs d'images ni le catalogue.
    """
    now = _utc(now)
    counters: dict = {"nouvelles_suivies": 0, "analyses": 0, "analyses_erreurs": 0, "recalcul": False,
                      "lignes": 0, "liens_demontres": 0}
    errors: list[str] = []
    if twins is None:
        twins = _twin_map()
    try:
        counters["nouvelles_suivies"] = track(con, new_per_day, now, twins)
    except Exception as e:
        con.rollback()
        errors.append(f"suivi : {e}")
    try:
        counters.update(analyse_pending(con, analyses_per_run, runner, now))
    except Exception as e:
        con.rollback()
        errors.append(f"analyses : {e}")
    last = _last_run(con)
    stats: dict = {}
    if last is None or now - last >= timedelta(hours=recompute_hours):
        try:
            rows = study(con, dossier_fn, candidates, now, n_boot, twins=twins, stats=stats)
            counters["lignes"] = write(con, rows, _iso(now))
            counters["liens_demontres"] = sum(r["verdict"] == DEMONTRE for r in rows)
            counters["exploratoires"] = sum(r["verdict"] == EXPLORATOIRE for r in rows)
            counters.update({k: v for k, v in stats.items() if v})
            counters["recalcul"] = True
            # noté même sans ligne : sans actualité, pas la peine de tout relire toutes les 15 min
            con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (META_LAST, _iso(now)))
            con.commit()
        except Exception as e:
            con.rollback()
            errors.append(f"étude : {e}")
    counters["suivies"] = con.execute("SELECT COUNT(*) FROM tracked").fetchone()[0]
    if errors:
        msg = "; ".join(errors)
    else:
        msg = f"{counters['suivies']} mines suivies, {counters['analyses']} analyse(s)"
        if counters["recalcul"]:
            msg += (f", {counters['lignes']} combinaisons dont {counters['liens_demontres']} lien(s) "
                    "démontré(s)" if counters["lignes"] else ", aucune combinaison calculable")
            if stats.get("dossiers_illisibles"):
                msg += f" ({stats['dossiers_illisibles']} dossier(s) illisible(s))"
    store.set_status(con, STAGE, not errors, msg, **counters)
    con.commit()
    return counters
