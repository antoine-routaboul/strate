"""Ligne de commande.

  uv run strate update [--site greenstone] [--module optical] [--start 2020-01] [--end 2026-10]
  uv run strate catalog        # catalogue mondial des mines (sources ouvertes)
  uv run strate calibrate all  # signaux satellite vs production réelle (≈ 2 h 30)
  uv run strate calibrate report  # rapports + calibration.json depuis les caches, sans téléchargement
  uv run strate check          # contrôle des alertes sur les événements connus
  uv run strate serve          # API + interface sur http://127.0.0.1:7779
  uv run strate veille once    # un cycle de veille (collecte → normalisation → corrélation)
  uv run strate veille run     # veille en continu (toutes les 15 min) ; install/uninstall : service
                               # launchd (Mac) ou systemd (Linux)
  uv run strate veille status  # état de chaque étape et du service
  uv run strate veille backup --to copie.db  # copie cohérente de la base, même service en marche
  uv run strate forecast backtest  # rétro-test walk-forward → data/forecast/backtest.json
  uv run strate forecast ledger    # prévisions datées ajoutées au registre (jamais réécrites)
  uv run strate companies refresh [--max-companies 50]  # faits SEC/Wikidata, borné, reprend ensuite
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from strate import entities, fusion
from strate.modules import REGISTRY


def _month(s: str, end: bool = False) -> date:
    y, m = map(int, s.split("-")[:2])
    if not end:
        return date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)  # dernier jour du mois


def update(args) -> None:
    sites = [args.site] if args.site else list(entities.load().sites)
    mods = [REGISTRY[args.module]] if args.module else list(REGISTRY.values())
    start, end = _month(args.start), _month(args.end, end=True)

    def job(sid):
        for m in mods:
            t = time.time()
            try:
                df = m.update(sid, start, end)
                print(f"{sid:20s} {m.id:10s} {len(df):5d} lignes  {time.time() - t:5.0f} s", flush=True)
            except Exception as e:  # un site en échec ne bloque pas les autres
                print(f"{sid:20s} {m.id:10s} ERREUR {e!r}", flush=True)

    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(job, sites))


def check(_args) -> None:
    for sid in entities.load().sites:
        for r in fusion.event_check(sid):
            mark = "DÉTECTÉ" if r["detecte"] else "manqué "
            print(f"{mark}  {sid:18s} {r['date']}  {r['evenement']}")
            if r["mois_en_alerte"]:
                print(f"           mois en alerte : {', '.join(r['mois_en_alerte'])}")
            for d in r["details"]:
                print(f"           {d}")


def build_catalog(args) -> None:
    from strate import catalog

    if args.refresh:
        catalog.fetch(force=True)
    cat = catalog.build()
    pays = cat["country"].dropna().nunique()
    print(f"{len(cat):,} sites dans {pays} pays → {catalog.OUT / 'mines.parquet'}")
    print(cat["sources"].value_counts().to_string())


def calibrate_cmd(args) -> None:
    import json

    from strate import calibrate

    if args.step in ("select", "all"):
        s = calibrate.select(args.n)
        print(f"{len(s)} mines sélectionnées → {calibrate.OUT / 'selection.csv'}")
    if args.step in ("run", "all"):
        calibrate.run(parallel=args.parallel)
    if args.step in ("report", "all"):
        # tout ce que cite docs/calibration.md : sélection recalée sur le catalogue actuel, panels et
        # rapports optique + radar, calibration.json (verdict radar déduit des intervalles)
        r = calibrate.report_all()
        print(json.dumps(r["optique"], ensure_ascii=False, indent=2, default=float))


def serve(args) -> None:
    import uvicorn

    uvicorn.run("strate.api:app", host="127.0.0.1", port=args.port)


def veille_cmd(args) -> None:
    import json
    import sqlite3
    from pathlib import Path

    from strate.veille import daemon

    # journal horodaté : en service, launchd ou systemd le redirige vers logs/veille.log
    lg = logging.getLogger("strate")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
        lg.addHandler(h)
        lg.propagate = False  # sinon chaque ligne sortirait deux fois (gestionnaire racine de main())
    if args.action == "once":
        limits = {"collecte": {k: v for k, v in (("backfill_hours", args.backfill_hours),
                                                 ("max_files", args.max_files)) if v is not None}}
        stages = tuple(args.stage) if args.stage else None
        res = daemon.run_cycle(limits=limits, stages=stages)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    elif args.action == "run":
        daemon.run_forever(interval_s=args.interval)
    elif args.action == "status":
        veille_status()
    elif args.action == "install":
        try:
            r = daemon.install(dry_run=args.dry_run)
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
        if args.dry_run:
            print(r["texte"])
        print(f"{r['gestionnaire']} : {r['fichier']}")
        if not args.dry_run:
            print("service chargé" if r["charge"] else f"chargement refusé : {r.get('message')}")
            if r["charge"] and r.get("message"):  # service lancé, mais avertissement (linger refusé…)
                print(f"attention : {r['message']}")
            print(f"journal : {daemon.LOG_PATH}")
    elif args.action == "backup":
        from strate.veille import store

        if not args.to:
            raise SystemExit("backup : indiquer la copie avec --to (ex. --to /tmp/veille.db)")
        try:
            r = store.backup(Path(args.to))
        except (OSError, ValueError, sqlite3.Error) as e:
            raise SystemExit(f"copie impossible : {e}") from None
        print(f"{r['copie']} ({r['octets']:,} octets) copiée depuis {r['source']}")
    elif args.action == "uninstall":
        try:
            r = daemon.uninstall()
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
        print(("service arrêté" if r["arrete"] else "service non chargé")
              + (", fichier du service retiré" if r["fichier_retire"] else ", aucun fichier de service"))


def veille_status() -> None:
    """Lecture seule : ne crée pas la base si la veille n'a jamais tourné."""
    import json
    import sqlite3

    from strate.veille import daemon, store

    rows: list[dict] = []
    totals: dict[str, int] = {}
    problem = None
    if store.DB_PATH.exists():
        try:
            con = sqlite3.connect(f"file:{store.DB_PATH}?mode=ro", uri=True, timeout=10)
        except sqlite3.Error as e:
            con, problem = None, str(e)
        if con is not None:
            con.row_factory = sqlite3.Row
            try:
                rows = [dict(r) for r in con.execute("SELECT * FROM pipeline_status")]
                for k, sql in (("documents", "SELECT count(*) FROM raw_docs"),
                               ("en attente", "SELECT count(*) FROM raw_docs WHERE status='new'"),
                               ("événements", "SELECT count(*) FROM events"),
                               ("mines suivies", "SELECT count(*) FROM tracked")):
                    totals[k] = con.execute(sql).fetchone()[0]
            except sqlite3.OperationalError as e:  # tables pas encore créées, base verrouillée
                problem = None if "no such table" in str(e) else str(e)
            except sqlite3.Error as e:  # fichier abîmé ou qui n'est pas une base SQLite
                problem = str(e)
            finally:
                con.close()
    print(f"base : {store.DB_PATH}" + ("" if store.DB_PATH.exists() else " (absente : jamais lancée)"))
    if problem:
        print(f"  base illisible : {problem}")
    order = {s: i for i, s in enumerate((*daemon.STAGE_NAMES, daemon.SERVICE))}
    for r in sorted(rows, key=lambda r: (order.get(r["stage"], 99), str(r["stage"]))):
        mark = "ok    " if r["ok"] else "ERREUR"
        print(f"  {r['stage']!s:14s} {mark} {r['last_run'] or '-'!s:21s} {r['message'] or ''}")
        try:
            c = json.loads(r["counters"] or "{}")
        except (TypeError, ValueError):
            c = {}
        if not isinstance(c, dict):
            c = {"compteurs": c}
        if c:
            print("                 " + ", ".join(f"{k}={v}" for k, v in c.items()))
    if totals:
        print("totaux : " + ", ".join(f"{k} {v:,}" for k, v in totals.items()))
    svc = daemon.service_state(next((r for r in rows if r["stage"] == daemon.SERVICE), None))
    print(f"service (base) : {'actif' if svc['actif'] else 'inactif'}, dernier cycle {svc['dernier_cycle']}")
    ld = daemon.manager_state()
    name = ld.pop("gestionnaire", None) or "service système"
    print(f"{name} : " + ("non installé" if not ld["installe"] else
                          ", ".join(f"{k}={v}" for k, v in ld.items() if k != "installe")))


def _module_fn(module: str, name: str):
    """Fonction d'un module écrit à part (import tardif) ; message clair s'il manque."""
    import importlib

    try:
        mod = importlib.import_module(f"strate.{module}")
    except Exception as e:  # module absent ou en panne
        raise SystemExit(f"module strate.{module} indisponible : {e}") from None
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise SystemExit(f"strate.{module}.{name} n'existe pas encore")
    return fn


def _fmt(v, nd: int = 2) -> str:
    return "-" if not isinstance(v, (int, float)) or v != v else f"{v:.{nd}f}"


def _ci(r: dict, k: str) -> str:
    return f"{_fmt(r.get(k))} [{_fmt(r.get(k + '_lo'))}, {_fmt(r.get(k + '_hi'))}]"


def _print_backtest(res: dict) -> None:
    """Tableau du rétro-test : une ligne par (modèle, horizon), chaque coefficient avec son IC 90 % et n."""
    print(f"{'modèle':16s} h  {'r [IC 90 %]':22s} {'rho [IC 90 %]':22s} {'skill [IC 90 %]':22s} "
          f"{'r niveau [IC 90 %]':22s} {'cov80':>6s} {'cov90':>6s} {'mines':>6s} {'points':>7s}")
    for r in res.get("rows") or []:
        print(f"{r.get('model')!s:16s} {r.get('horizon')!s:2s} {_ci(r, 'r'):22s} {_ci(r, 'rho'):22s} "
              f"{_ci(r, 'skill'):22s} {_ci(r, 'r_niveau'):22s} {_fmt(r.get('couverture80')):>6s} "
              f"{_fmt(r.get('couverture90')):>6s} {r.get('n_mines')!s:>6s} {r.get('n_points')!s:>7s}")
    print("r, rho : variation prévue ↔ variation réelle du log de la production (période de test) ; « - » "
          "quand le modèle ne prévoit aucune variation (naïf). r niveau : log des niveaux, dominé par les "
          "écarts de taille entre mines. IC 90 % : bootstrap sur les mines.")
    if res.get("retenu"):
        print("retenu : " + ", ".join(f"h={h} → {m}" for h, m in res["retenu"].items()))


def forecast_cmd(args) -> None:
    import json

    from strate.core import DATA_DIR

    if args.action == "ledger":
        n = _module_fn("forecast", "write_ledger")()
        print(f"{n} prévisions ajoutées au registre → {DATA_DIR / 'forecast' / 'ledger.parquet'}")
        return
    path = DATA_DIR / "forecast" / "backtest.json"
    before = path.stat().st_mtime_ns if path.exists() else None
    res = _module_fn("forecast", "backtest")()
    if isinstance(res, dict) and (not path.exists() or path.stat().st_mtime_ns == before):
        from strate.api import _finite  # même nettoyage NaN → null que l'API

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_finite(res), ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(path)
    res = res if isinstance(res, dict) else {}
    print(f"rétro-test {res.get('run_at') or ''} → {path}")
    if not res.get("rows") and isinstance(res.get("raison"), str):
        print(f"aucune mesure : {res['raison']}")
    _print_backtest(res)


def companies_cmd(args) -> None:
    """refresh(max_companies=…) du module ; à défaut l'ancien update(limit=…), même borne."""
    import importlib
    import json

    try:
        mod = importlib.import_module("strate.companies")
    except Exception as e:  # module absent ou en panne
        raise SystemExit(f"module strate.companies indisponible : {e}") from None
    if callable(getattr(mod, "refresh", None)):
        kw = {} if args.max_companies is None else {"max_companies": args.max_companies}
        res = mod.refresh(**kw)
    elif callable(getattr(mod, "update", None)):
        kw = {} if args.max_companies is None else {"limit": args.max_companies}
        res = mod.update(**kw)
    else:
        raise SystemExit("strate.companies n'a ni refresh ni update")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))


def _interval(v: str) -> int:
    """--interval : secondes entières, au moins daemon.MIN_INTERVAL_S (0 ou négatif n'a pas de sens)."""
    from strate.veille.daemon import MIN_INTERVAL_S

    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"nombre entier de secondes attendu, pas {v!r}") from None
    if n < MIN_INTERVAL_S:
        raise argparse.ArgumentTypeError(f"au moins {MIN_INTERVAL_S} s (reçu {n})")
    return n


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="strate")
    sub = p.add_subparsers(required=True)
    u = sub.add_parser("update", help="calcule / met à jour les signaux")
    u.add_argument("--site")
    u.add_argument("--module", choices=sorted(REGISTRY))
    u.add_argument("--start", default="2020-01")
    u.add_argument("--end", default=f"{date.today():%Y-%m}")
    u.add_argument("--parallel", type=int, default=3)
    u.set_defaults(fn=update)
    sub.add_parser("check", help="alertes vs événements connus").set_defaults(fn=check)
    c = sub.add_parser("catalog", help="télécharge les sources et construit le catalogue mondial")
    c.add_argument("--refresh", action="store_true", help="re-télécharge toutes les sources")
    c.set_defaults(fn=build_catalog)
    k = sub.add_parser("calibrate", help="compare les signaux satellite à la production réelle")
    k.add_argument("step", choices=["select", "run", "report", "all"],
                   help="report : régénère sélection recalée, panels, rapports et calibration.json")
    k.add_argument("-n", type=int, default=150, help="nombre de mines")
    k.add_argument("--parallel", type=int, default=8)
    k.set_defaults(fn=calibrate_cmd)
    s = sub.add_parser("serve", help="lance l'API et l'interface")
    s.add_argument("--port", type=int, default=7779)
    s.set_defaults(fn=serve)
    v = sub.add_parser("veille", help="veille d'actualité minière en continu (GDELT, toutes langues)")
    v.add_argument("action", choices=["once", "run", "status", "install", "uninstall", "backup"])
    v.add_argument("--stage", action="append", choices=["collecte", "normalisation", "correlation"],
                   help="once : seulement cette étape (répétable)")
    v.add_argument("--backfill-hours", type=int, help="once : historique GDELT rattrapé au 1er lancement")
    v.add_argument("--max-files", type=int, help="once : fichiers GDELT au plus par cycle")
    v.add_argument("--interval", type=_interval, default=900,
                   help="run : secondes entre deux cycles (au moins 60)")
    v.add_argument("--dry-run", action="store_true",
                   help="install : affiche le plist (Mac) ou l'unité systemd (Linux) sans rien écrire")
    v.add_argument("--to", help="backup : chemin de la copie de la base")
    v.set_defaults(fn=veille_cmd)
    f = sub.add_parser("forecast", help="prévisions de production auto-évaluées (rétro-test, registre)")
    f.add_argument("action", choices=["backtest", "ledger"],
                   help="backtest : rétro-test walk-forward ; ledger : prévisions datées dans le registre")
    f.set_defaults(fn=forecast_cmd)
    co = sub.add_parser("companies", help="sociétés : faits SEC/Wikidata, investissements, corrélations")
    co.add_argument("action", choices=["refresh"])
    co.add_argument("--max-companies", type=int, help="sociétés au plus par lancement (reprend ensuite)")
    co.set_defaults(fn=companies_cmd)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
