"""Le service de veille : enchaîne collecte → normalisation → corrélation tous les quarts d'heure.

    uv run strate veille once      # un cycle, puis rend la main
    uv run strate veille run       # boucle sans fin (ce que lance launchd)
    uv run strate veille install   # service « com.strate.veille » (launchd, Mac) ou « strate-veille »
                                   # (systemd utilisateur, Linux : une machine virtuelle dans le cloud)

Pourquoi ces choix :
- chaque étape est isolée : une panne réseau de la collecte n'empêche pas de normaliser ce qui est déjà
  là, ni de recalculer les corrélations ; l'erreur est notée dans pipeline_status et retentée au cycle
  suivant ;
- le cycle démarre au quart d'heure pile, au rythme où GDELT publie ses fichiers ;
- l'attente suit l'horloge murale par petits pas : après une mise en veille du Mac, le cycle manqué part
  au réveil au lieu d'attendre encore le reste de l'intervalle ;
- SIGTERM (launchctl, systemctl, extinction) : plus aucune étape ne démarre ; l'étape en cours peut
  finir dans le délai d'arrêt du service (60 s, ExitTimeOut de launchd ou TimeoutStopSec de systemd).
  Une étape plus longue (corrélation avec analyses satellite) est tuée : SQLite annule sa transaction
  non validée (rien d'incohérent) et le battement « arrêté » manque, mais l'état du service le voit
  inactif (pid disparu). Une étape longue peut interroger `stop_requested()` entre deux unités de
  travail pour s'arrêter à temps ;
- priorité basse (os.nice) : la veille ne doit jamais ralentir le Mac ;
- un verrou de fichier empêche deux cycles simultanés (service + `veille once` lancé à la main).
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import json
import logging
import math
import os
import plistlib
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from strate.veille import store

log = logging.getLogger("strate.veille")

# (nom dans pipeline_status, module, limites par défaut du contrat docs/veille.md)
STAGES: tuple[tuple[str, str, dict], ...] = (
    ("collecte", "strate.veille.collect", {"backfill_hours": 48, "max_files": 40, "doc_queries": 6}),
    ("normalisation", "strate.veille.normalize", {"batch": 5000}),
    ("correlation", "strate.veille.correlate",
     {"new_per_day": 20, "analyses_per_run": 2, "recompute_hours": 6}),
)
STAGE_NAMES = tuple(s[0] for s in STAGES)
SERVICE = "service"
INTERVAL_S = 900
MIN_INTERVAL_S = 60  # en dessous, les cycles s'enchaîneraient sans laisser respirer GDELT ni le Mac
NICE = 10
POLL_S = 30  # pas d'attente : borne le retard après une mise en veille du Mac
LOG_MAX_BYTES = 20_000_000

LABEL = "com.strate.veille"
PROJECT_DIR = Path(__file__).resolve().parents[3]
LOG_PATH = PROJECT_DIR / "logs" / "veille.log"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
UNIT = "strate-veille.service"
UNIT_PATH = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user" / UNIT

_stop = threading.Event()
_signals = 0


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stop_requested() -> bool:
    """Vrai après SIGTERM/SIGINT : une étape longue peut s'arrêter entre deux unités de travail."""
    return _stop.is_set()


def finite(o):
    """NaN et infinis → None, récursivement : le JSON strict (navigateur, FastAPI) les refuse."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def _jsonable(d) -> dict:
    """Compteurs rendus sérialisables (types numpy, dates…, NaN → null) : set_status ne doit jamais
    échouer ni écrire un `NaN` que l'API ne pourrait pas renvoyer."""
    try:
        out = finite(json.loads(json.dumps(d if isinstance(d, dict) else {"resultat": d}, default=str)))
        return out if isinstance(out, dict) else {"resultat": out}
    except (TypeError, ValueError, RecursionError):
        return {"resultat": str(d)}


# noms pris par set_status et par le résumé du cycle : un compteur d'étape du même nom est renommé
_RESERVED = frozenset({"con", "stage", "ok", "message", "erreur", "duree_s"})


def _counters(res: dict) -> dict:
    """Compteurs d'une étape sans collision : « ok » → « ok_etape » (sinon set_status lèverait TypeError
    et une étape réussie serait notée en erreur, ou le résumé masquerait son vrai résultat)."""
    out = {}
    for k, v in res.items():
        k = str(k)
        while k in _RESERVED or k in out:
            k += "_etape"
        out[k] = v
    return out


def _valid_interval(v) -> bool:
    """Nombre fini et positif (pas un booléen, pas NaN ni l'infini : la boucle ne repartirait jamais)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def next_boundary(now: float, interval_s: int = INTERVAL_S, offset_s: int = 0) -> float:
    """Prochain multiple de l'intervalle (heure UTC) strictement après `now` : 12:00, 12:15, 12:30…

    Un cycle plus long que l'intervalle saute au quart d'heure suivant au lieu de s'empiler."""
    if not _valid_interval(interval_s):
        raise ValueError(f"intervalle invalide : {interval_s!r} s (il faut un nombre fini et positif)")
    k = (now - offset_s) // interval_s + 1
    return k * interval_s + offset_s


# -- un cycle ------------------------------------------------------------------------------------------
def _status_row(con: sqlite3.Connection, stage: str) -> tuple | None:
    """(last_run, ok, message, counters) de l'étape : sert à voir si l'étape a écrit son propre statut."""
    r = con.execute("SELECT last_run, ok, message, counters FROM pipeline_status WHERE stage=?",
                    (stage,)).fetchone()
    return tuple(r) if r else None


def _heartbeat(con: sqlite3.Connection, message: str, ok: bool = True, **counters) -> None:
    """Ligne « service » de pipeline_status : prouve à l'API que le service tourne.

    Les compteurs précédents sont gardés (dernier_cycle reste connu pendant le cycle suivant)."""
    try:
        r = con.execute("SELECT counters FROM pipeline_status WHERE stage=?", (SERVICE,)).fetchone()
        try:
            prev = json.loads(r[0]) if r and r[0] else {}
        except ValueError:
            prev = {}
        merged = _jsonable({**(prev if isinstance(prev, dict) else {}), **counters, "pid": os.getpid()})
        store.set_status(con, SERVICE, ok, message, **merged)
        con.commit()
    except sqlite3.Error as e:  # base verrouillée trop longtemps : le prochain battement réessaiera
        log.warning("battement du service non écrit : %s", e)


def _run_stage(con: sqlite3.Connection, stage: str, module: str, kwargs: dict) -> dict:
    """Lance une étape ; toute erreur (import, réseau, données) est notée et n'arrête pas la chaîne."""
    t0 = time.time()
    try:
        mod = importlib.import_module(module)  # import tardif : un module absent ne casse que son étape
        with store.watch_status() as written:
            res = _counters(_jsonable(mod.run_once(con, **kwargs)))
        after = _status_row(con, stage)
        mine = written.get(stage)
        if mine is not None and after is not None and after == mine:
            # l'étape a écrit son statut et il est toujours là (elle peut avoir noté elle-même un échec) :
            # c'est lui qui fait foi
            ok, msg = bool(after[1]), after[2] or ""
        else:
            # pas de statut écrit, ou statut annulé par l'étape elle-même (rollback après set_status :
            # ligne absente, ou ligne du cycle précédent) : on l'écrit pour elle, sans le prendre pour le sien
            msg = "terminé" if mine is None else "terminé (statut de l'étape annulé par son rollback)"
            store.set_status(con, stage, True, msg, **res)
            ok = True
        con.commit()
        dt = round(time.time() - t0, 1)
        if ok:
            log.info("%-13s ok en %.0f s : %s", stage, dt, res)
            return {**res, "ok": True, "duree_s": dt}
        log.warning("%-13s en échec (noté par l'étape) en %.0f s : %s", stage, dt, msg)
        return {**res, "ok": False, "erreur": msg, "duree_s": dt}
    except Exception as e:  # isolation voulue : l'étape suivante doit tourner
        with contextlib.suppress(sqlite3.Error):
            con.rollback()
        msg = f"erreur : {type(e).__name__}: {e}"[:500]
        log.exception("%-13s %s", stage, msg)
        try:
            store.set_status(con, stage, False, msg, duree_s=round(time.time() - t0, 1))
            con.commit()
        except sqlite3.Error as e2:
            log.warning("statut de %s non écrit : %s", stage, e2)
        return {"ok": False, "erreur": msg, "duree_s": round(time.time() - t0, 1)}


@contextlib.contextmanager
def _cycle_lock(db: Path):
    """Verrou exclusif non bloquant à côté de la base : rend False si un autre cycle tourne déjà."""
    db.parent.mkdir(parents=True, exist_ok=True)
    with open(db.parent / "cycle.lock", "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run_cycle(db_path: Path | None = None, limits: dict[str, dict] | None = None,
              stages: tuple[str, ...] | None = None, cycle: int | None = None,
              interval_s: int = INTERVAL_S) -> dict:
    """Un passage collecte → normalisation → corrélation. `limits` : {étape: kwargs de run_once}.

    Rend {étape: compteurs | erreur}. S'arrête entre deux étapes si SIGTERM est arrivé."""
    db = Path(db_path or store.DB_PATH)
    limits = limits or {}
    out: dict[str, dict] = {}
    with _cycle_lock(db) as got:
        if not got:
            log.warning("un autre cycle de veille tourne déjà : celui-ci est sauté")
            return {"saute": {"ok": False, "erreur": "un autre cycle tourne déjà"}}
        con = store.init(db)
        try:
            for name, module, defaults in STAGES:
                if stages and name not in stages:
                    continue
                if _stop.is_set():
                    out[name] = {"ok": False, "erreur": "arrêt demandé avant l'étape"}
                    continue
                if cycle is not None:  # le battement suit l'étape : un long cycle reste visible « actif »
                    _heartbeat(con, f"cycle {cycle} : {name} en cours", etat="en marche", cycles=cycle - 1,
                               interval_s=interval_s, prochain=None)
                out[name] = _run_stage(con, name, module, {**defaults, **limits.get(name, {})})
        finally:
            con.close()
    return out


# -- la boucle -------------------------------------------------------------------------------------------
def _handle_signal(signum, _frame) -> None:
    """1er signal : on finit l'étape en cours puis on s'arrête. 2e signal : arrêt immédiat."""
    global _signals
    _signals += 1
    _stop.set()
    if _signals > 1:
        raise SystemExit(128 + signum)
    log.info("signal %s reçu : arrêt après l'étape en cours", signal.Signals(signum).name)


def _sleep_until(t_wall: float) -> None:
    """Attend jusqu'à l'heure murale `t_wall` par petits pas (l'horloge monotone s'arrête pendant la
    veille du Mac ; l'horloge murale, non)."""
    while not _stop.is_set():
        left = t_wall - time.time()
        if left <= 0:
            return
        _stop.wait(min(POLL_S, left))


def _trim_log(path: Path | None = None, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Journal launchd trop gros → copié en .1 puis vidé (launchd l'écrit en ajout : la suite repart à 0)."""
    path = path or LOG_PATH
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            shutil.copyfile(path, path.with_suffix(path.suffix + ".1"))
            with open(path, "r+") as fh:
                fh.truncate(0)
    except OSError as e:
        log.warning("journal non tourné : %s", e)


def run_forever(interval_s: int = INTERVAL_S, db_path: Path | None = None,
                limits: dict[str, dict] | None = None, max_cycles: int | None = None,
                offset_s: int = 0) -> int:
    """Boucle du service. Rend le nombre de cycles faits (max_cycles : pour les essais)."""
    global _signals
    if not _valid_interval(interval_s) or interval_s < MIN_INTERVAL_S:
        raise ValueError(f"intervalle invalide : {interval_s!r} s (au moins {MIN_INTERVAL_S} s)")
    _stop.clear()
    _signals = 0
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, _handle_signal)
    with contextlib.suppress(OSError, AttributeError):
        os.nice(NICE)
    db = Path(db_path or store.DB_PATH)
    log.info("veille démarrée (pid %s, intervalle %s s, base %s)", os.getpid(), interval_s, db)
    con = store.init(db)
    _heartbeat(con, "démarré", etat="en marche", cycles=0, interval_s=interval_s, prochain=None)
    con.close()
    n, last_cycle = 0, None
    while not _stop.is_set() and (max_cycles is None or n < max_cycles):
        _trim_log()
        t0 = time.time()
        res = run_cycle(db, limits, cycle=n + 1, interval_s=interval_s)
        n += 1
        last_cycle = _iso(time.time())
        nxt = next_boundary(time.time(), interval_s, offset_s)
        ok = sum(1 for r in res.values() if r.get("ok"))
        msg = f"cycle {n} terminé : {ok}/{len(res)} étapes ok en {time.time() - t0:.0f} s"
        log.info("%s ; prochain à %s", msg, _iso(nxt))
        con = store.init(db)
        _heartbeat(con, msg, etat="en marche", cycles=n, interval_s=interval_s, dernier_cycle=last_cycle,
                   prochain=_iso(nxt), etapes_ok=ok, etapes=len(res))
        con.close()
        if max_cycles is not None and n >= max_cycles:
            break
        _sleep_until(nxt)
    con = store.init(db)
    _heartbeat(con, f"arrêté après {n} cycle(s)", ok=True, etat="arrêté", cycles=n, interval_s=interval_s,
               dernier_cycle=last_cycle)
    con.close()
    log.info("veille arrêtée après %s cycle(s)", n)
    return n


# -- état du service (lu par l'API et `strate veille status`) ----------------------------------------
def _pid_alive(pid) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # le processus existe, sous un autre utilisateur
        return True
    except OSError:
        return None
    return True


def service_state(row: dict | None, now: float | None = None) -> dict:
    """{"actif", "dernier_cycle", "prochain", "message"} d'après la ligne « service » de pipeline_status.

    Actif = ni arrêté, ni processus disparu, et un battement récent : une étape de corrélation lance des
    analyses satellite (5 à 12 min par mine), d'où une marge large (3 intervalles, au moins 1 h)."""
    if not row:
        return {"actif": False, "dernier_cycle": None, "prochain": None, "message": None}
    c = row.get("counters") or {}
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except ValueError:
            c = {}
    if not isinstance(c, dict):  # compteurs JSON valides mais pas un objet ([1, 2], 3…) : ignorés
        c = {}
    c = finite(c)
    now = time.time() if now is None else now
    actif = False
    try:
        last = datetime.strptime(str(row.get("last_run") or ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        age = now - last.timestamp()
        iv = c.get("interval_s")
        interval = iv if isinstance(iv, (int, float)) and not isinstance(iv, bool) and iv > 0 else INTERVAL_S
        actif = c.get("etat") != "arrêté" and age <= max(3 * interval, 3600) and _pid_alive(c.get("pid")) \
            is not False
    except (TypeError, ValueError):
        pass

    def txt(v):  # dates et message : du texte ou rien
        return v if isinstance(v, str) else None

    return {"actif": bool(actif), "dernier_cycle": txt(c.get("dernier_cycle")),
            "prochain": txt(c.get("prochain")), "message": txt(row.get("message"))}


# -- launchd -----------------------------------------------------------------------------------------------
def plist_dict(uv: str, project: Path | None = None, log_path: Path | None = None) -> dict:
    """Définition du service. KeepAlive : launchd le relance s'il meurt ; ThrottleInterval évite une
    boucle de redémarrages si le démarrage échoue ; ExitTimeOut (60 s) laisse finir une étape courte
    après SIGTERM, pas une corrélation avec analyses satellite (tuée, transaction annulée par SQLite)."""
    project, log_path = project or PROJECT_DIR, log_path or LOG_PATH
    env = {"PATH": f"{Path(uv).parent}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
           "PYTHONUNBUFFERED": "1"}
    if os.environ.get("STRATE_DATA"):
        env["STRATE_DATA"] = os.environ["STRATE_DATA"]
    return {
        "Label": LABEL,
        "ProgramArguments": [uv, "run", "--project", str(project), "strate", "veille", "run"],
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 60,
        "ExitTimeOut": 60,
        "ProcessType": "Background",
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "EnvironmentVariables": env,
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def _launchd_install(dry_run: bool = False) -> dict:
    """Écrit ~/Library/LaunchAgents/com.strate.veille.plist et le charge (launchctl bootstrap)."""
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv introuvable dans le PATH : installer uv avant le service")
    # chemin absolu sans résoudre le lien : /opt/homebrew/bin/uv survit à `brew upgrade`, pas le Cellar
    pl = plist_dict(os.path.abspath(uv))
    if dry_run:
        return {"gestionnaire": "launchd", "fichier": str(PLIST_PATH), "plist": str(PLIST_PATH),
                "contenu": pl, "texte": plistlib.dumps(pl).decode(), "charge": False}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # réinstallation : l'ancienne version est d'abord arrêtée
    PLIST_PATH.write_bytes(plistlib.dumps(pl))
    r = _launchctl("bootstrap", _domain(), str(PLIST_PATH))
    return {"gestionnaire": "launchd", "fichier": str(PLIST_PATH), "plist": str(PLIST_PATH), "contenu": pl,
            "texte": plistlib.dumps(pl).decode(), "charge": r.returncode == 0,
            "message": (r.stderr or r.stdout).strip(), "launchctl": (r.stderr or r.stdout).strip()}


def _launchd_uninstall() -> dict:
    """Arrête le service (launchctl bootout) et retire son plist."""
    r = _launchctl("bootout", f"{_domain()}/{LABEL}")
    existed = PLIST_PATH.exists()
    PLIST_PATH.unlink(missing_ok=True)
    return {"gestionnaire": "launchd", "arrete": r.returncode == 0, "plist_retire": existed,
            "fichier_retire": existed, "launchctl": (r.stderr or r.stdout).strip()}


def launchd_state() -> dict:
    """État vu par launchd : chargé ?, état, pid, dernier code de sortie."""
    if not PLIST_PATH.exists():
        return {"installe": False, "charge": False}
    try:
        r = _launchctl("print", f"{_domain()}/{LABEL}")
    except OSError as e:
        return {"installe": True, "charge": None, "erreur": str(e)}
    if r.returncode != 0:
        return {"installe": True, "charge": False}
    out: dict = {"installe": True, "charge": True}
    for line in r.stdout.splitlines():
        k, _, v = line.strip().partition(" = ")
        if k in ("state", "pid", "last exit code", "runs"):
            out[k.replace(" ", "_")] = v
    return out


# -- systemd (Linux : machine virtuelle dans le cloud) ----------------------------------------------------
def _systemd_escape(v: str, quote: bool, dollar: bool = False) -> str:
    """Valeur sûre pour un fichier d'unité : % (spécificateurs) doublé ; entre guillemets, \\ et " échappés.
    `dollar` : $ doublé, seulement pour ExecStart (qui développe $VAR ; Environment= ne le fait pas et
    garderait $$ tel quel). Un retour à la ligne casserait l'unité."""
    v = str(v)
    if "\n" in v or "\r" in v:
        raise ValueError(f"retour à la ligne interdit dans une unité systemd : {v!r}")
    v = v.replace("%", "%%")
    if not quote:
        return v
    v = v.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + (v.replace("$", "$$") if dollar else v) + '"'


def unit_text(uv: str, project: Path | None = None, log_path: Path | None = None) -> str:
    """Unité systemd utilisateur, équivalente au plist launchd :

    - Restart=always + RestartSec=60 + StartLimitIntervalSec=0 : relancé s'il meurt, une fois par minute
      au plus, sans jamais abandonner (KeepAlive + ThrottleInterval de launchd) ;
    - KillMode=mixed : SIGTERM au seul processus principal (uv), qui le relaie à Python. Avec le mode par
      défaut, Python recevrait aussi le SIGTERM de systemd : deux signaux, donc arrêt immédiat au lieu de
      finir l'étape en cours ; SIGKILL à tout le groupe après TimeoutStopSec (60 s, comme ExitTimeOut) ;
    - pas de Nice= : le service s'abaisse déjà lui-même (os.nice) ;
    - journal en ajout dans logs/veille.log, tourné par _trim_log comme sur le Mac."""
    project, log_path = project or PROJECT_DIR, log_path or LOG_PATH
    path = f"{Path(uv).parent}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    env = [("PATH", path), ("PYTHONUNBUFFERED", "1")]
    if os.environ.get("STRATE_DATA"):
        env.append(("STRATE_DATA", os.environ["STRATE_DATA"]))
    q = _systemd_escape
    lines = [
        "[Unit]",
        "Description=Strate : veille d'actualité minière (collecte, normalisation, corrélation)",
        "StartLimitIntervalSec=0",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={q(str(project), False)}",
        f"ExecStart={q(uv, True, True)} run --project {q(str(project), True, True)} strate veille run",
        *(f"Environment={q(f'{k}={v}', True)}" for k, v in env),
        "Restart=always",
        "RestartSec=60",
        "KillMode=mixed",
        "TimeoutStopSec=60",
        f"StandardOutput=append:{q(str(log_path), False)}",
        f"StandardError=append:{q(str(log_path), False)}",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def _user_env() -> dict:
    """Hors session de connexion (sudo -u, su sans -l, cron), XDG_RUNTIME_DIR manque et systemctl --user
    ne trouve pas le gestionnaire : on le rétablit quand le dossier existe (cas du linger)."""
    env = dict(os.environ)
    run_dir = Path(f"/run/user/{os.getuid()}")
    if not env.get("XDG_RUNTIME_DIR") and run_dir.is_dir():
        env["XDG_RUNTIME_DIR"] = str(run_dir)
    return env


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=False,
                          env=_user_env())


def _loginctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["loginctl", "--no-ask-password", *args], capture_output=True, text=True,
                          check=False)


def _sudo_loginctl(*args: str) -> subprocess.CompletedProcess:
    """sudo sans mot de passe (-n) : l'utilisateur d'une machine Azure l'a d'ordinaire ; sinon échoue net."""
    return subprocess.run(["sudo", "-n", "loginctl", *args], capture_output=True, text=True, check=False)


def _user() -> str:
    import getpass

    return getpass.getuser()


def _linger() -> dict:
    """Sans « linger », systemd arrête les services utilisateur à la déconnexion SSH et ne les relance pas
    au redémarrage : la veille ne tournerait que pendant qu'on est connecté. On l'active soi-même
    (polkit l'autorise en général), sinon par sudo sans mot de passe ; à défaut, on dit quoi lancer."""
    user = _user()
    hint = f"sudo loginctl enable-linger {user}"
    try:
        r = _loginctl("show-user", user, "-p", "Linger", "--value")
        if r.returncode == 0 and r.stdout.strip() == "yes":
            return {"linger": True, "message": ""}
        r = _loginctl("enable-linger", user)
        if r.returncode != 0:
            r2 = _sudo_loginctl("enable-linger", user)
            r = r2 if r2.returncode == 0 else r
    except OSError as e:
        return {"linger": False, "message": f"loginctl indisponible ({e}) : {hint}"}
    if r.returncode == 0:
        return {"linger": True, "message": ""}
    why = (r.stderr or r.stdout).strip()
    return {"linger": False, "message": f"linger refusé ({why}) : lancer {hint}, sinon la veille s'arrête "
                                        "à la déconnexion"}


def _systemd_install(dry_run: bool = False) -> dict:
    """Écrit ~/.config/systemd/user/strate-veille.service, l'active et le (re)démarre."""
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv introuvable dans le PATH : installer uv avant le service")
    text = unit_text(os.path.abspath(uv))
    base = {"gestionnaire": "systemd", "fichier": str(UNIT_PATH), "texte": text}
    if dry_run:
        return {**base, "charge": False}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.write_text(text)
    msgs = []
    ok = True
    # restart et pas seulement enable --now : une réinstallation doit relancer avec la nouvelle unité
    for step in (("daemon-reload",), ("enable", UNIT), ("restart", UNIT)):
        r = _systemctl(*step)
        if r.returncode != 0:
            ok = False
            msgs.append(f"systemctl --user {' '.join(step)} : {(r.stderr or r.stdout).strip()}")
            break
    lg = _linger()
    if lg["message"]:
        msgs.append(lg["message"])
    return {**base, "charge": ok, "linger": lg["linger"], "message": " ; ".join(msgs)}


def _systemd_uninstall() -> dict:
    """Arrête et désactive le service, retire l'unité."""
    r = _systemctl("disable", "--now", UNIT)
    existed = UNIT_PATH.exists()
    UNIT_PATH.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    return {"gestionnaire": "systemd", "arrete": r.returncode == 0, "fichier_retire": existed,
            "plist_retire": existed, "message": (r.stderr or r.stdout).strip()}


def systemd_state() -> dict:
    """État vu par systemd : actif ?, sous-état, pid, redémarrages, dernier code de sortie."""
    if not UNIT_PATH.exists():
        return {"installe": False, "charge": False}
    props = ("ActiveState", "SubState", "MainPID", "NRestarts", "ExecMainStatus")
    try:
        r = _systemctl("show", UNIT, *(f"--property={p}" for p in props))
    except OSError as e:
        return {"installe": True, "charge": None, "erreur": str(e)}
    if r.returncode != 0:  # `show` rend 0 même pour une unité inconnue : ici, gestionnaire injoignable
        return {"installe": True, "charge": None, "erreur": (r.stderr or r.stdout).strip()}
    kv = dict(line.partition("=")[::2] for line in r.stdout.splitlines() if "=" in line)
    active = kv.get("ActiveState", "")
    out: dict = {"installe": True, "charge": active in ("active", "activating", "reloading")}
    for k, name in (("ActiveState", "state"), ("SubState", "substate"), ("MainPID", "pid"),
                    ("NRestarts", "restarts"), ("ExecMainStatus", "last_exit_code")):
        if kv.get(k, "") != "":
            out[name] = kv[k]
    return out


# -- choix du gestionnaire de services -------------------------------------------------------------------
def service_manager(platform: str | None = None) -> str:
    """« launchd » sur macOS, « systemd » sur Linux ; ailleurs, pas de service (veille run à la main)."""
    platform = platform or sys.platform
    if platform == "darwin":
        return "launchd"
    if platform.startswith("linux"):
        return "systemd"
    raise RuntimeError(f"pas de service pour la plateforme {platform!r} : lancer `strate veille run`")


def install(dry_run: bool = False, manager: str | None = None) -> dict:
    """Installe le service de la plateforme (launchd sur Mac, systemd utilisateur sur Linux)."""
    manager = manager or service_manager()
    return _systemd_install(dry_run) if manager == "systemd" else _launchd_install(dry_run)


def uninstall(manager: str | None = None) -> dict:
    manager = manager or service_manager()
    return _systemd_uninstall() if manager == "systemd" else _launchd_uninstall()


def manager_state(manager: str | None = None) -> dict:
    """État du service vu par le gestionnaire de la plateforme ({"installe": False} s'il n'y en a pas)."""
    try:
        manager = manager or service_manager()
    except RuntimeError:
        return {"gestionnaire": None, "installe": False, "charge": False}
    st = systemd_state() if manager == "systemd" else launchd_state()
    return {"gestionnaire": manager, **st}
