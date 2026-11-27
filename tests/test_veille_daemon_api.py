"""Veille : service (cycle isolé par étape, alignement au quart d'heure, arrêt sur SIGTERM, battement),
CLI `strate veille`, services launchd et systemd (sans rien installer) et routes /api/veille/* en lecture
seule."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from strate import api, cli, screener
from strate.veille import daemon, store


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime.now(UTC)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "veille" / "veille.db"
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(screener, "twins", lambda: {})  # pas de catalogue réel dans les essais
    daemon._stop.clear()
    yield path
    daemon._stop.clear()


def _fake_stage(name: str, calls: list, *, fail: bool = False, writes_status: bool = True, counters=None):
    stage = {"strate.veille.collect": "collecte", "strate.veille.normalize": "normalisation",
             "strate.veille.correlate": "correlation"}[name]
    mod = types.ModuleType(name)

    def run_once(con, **kw):
        calls.append((stage, kw))
        if fail:
            raise ConnectionError("GDELT injoignable")
        con.execute("CREATE TABLE IF NOT EXISTS essai(x)")  # écrit quelque chose : doit être validé
        con.execute("INSERT INTO essai VALUES (?)", (stage,))
        if writes_status:
            store.set_status(con, stage, True, "ok par l'étape", **(counters or {}))
        return counters or {"lus": 3}

    mod.run_once = run_once
    return mod


def _install_stages(monkeypatch, calls, **per_stage):
    for name in ("strate.veille.collect", "strate.veille.normalize", "strate.veille.correlate"):
        monkeypatch.setitem(sys.modules, name, _fake_stage(name, calls, **per_stage.get(name, {})))


def _status(path) -> dict:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    out = {r["stage"]: dict(r) for r in con.execute("SELECT * FROM pipeline_status")}
    con.close()
    return out


# -- service ---------------------------------------------------------------------------------------------
def test_next_boundary_aligns_on_quarter_hour():
    t = datetime(2026, 11, 26, 12, 7, 30, tzinfo=UTC).timestamp()
    assert daemon.next_boundary(t) == datetime(2026, 11, 26, 12, 15, tzinfo=UTC).timestamp()
    exact = datetime(2026, 11, 26, 12, 15, tzinfo=UTC).timestamp()
    assert daemon.next_boundary(exact) == exact + 900  # jamais l'instant présent : pas de double cycle
    assert daemon.next_boundary(t, 900, 120) == datetime(2026, 11, 26, 12, 17, tzinfo=UTC).timestamp()


def test_cycle_isolates_stage_errors_and_merges_limits(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls, **{
        "strate.veille.collect": {"fail": True},
        "strate.veille.correlate": {"writes_status": False, "counters": {"suivies": 2}},
    })
    res = daemon.run_cycle(limits={"collecte": {"max_files": 5}, "correlation": {"analyses_per_run": 0}})
    assert [c[0] for c in calls] == ["collecte", "normalisation", "correlation"]  # la panne n'arrête rien
    assert calls[0][1] == {"backfill_hours": 48, "max_files": 5, "doc_queries": 6}
    assert calls[2][1] == {"new_per_day": 20, "analyses_per_run": 0, "recompute_hours": 6}
    assert res["collecte"]["ok"] is False and "GDELT injoignable" in res["collecte"]["erreur"]
    assert res["normalisation"]["ok"] and res["correlation"]["suivies"] == 2
    st = _status(db)
    assert st["collecte"]["ok"] == 0 and "ConnectionError" in st["collecte"]["message"]
    assert st["normalisation"]["message"] == "ok par l'étape"
    # l'étape n'a pas écrit son statut : le service l'écrit à sa place, avec ses compteurs
    assert st["correlation"]["ok"] == 1 and json.loads(st["correlation"]["counters"]) == {"suivies": 2}
    con = sqlite3.connect(db)
    # l'écriture de l'étape en échec est annulée, celles des autres sont validées
    assert sorted(r[0] for r in con.execute("SELECT x FROM essai")) == ["correlation", "normalisation"]


def test_missing_stage_module_is_only_that_stage_error(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls)
    monkeypatch.setitem(sys.modules, "strate.veille.normalize", None)  # module pas encore écrit
    res = daemon.run_cycle()
    assert res["normalisation"]["ok"] is False and "ModuleNotFoundError" in res["normalisation"]["erreur"]
    assert res["collecte"]["ok"] and res["correlation"]["ok"]


def test_only_selected_stages(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls)
    assert list(daemon.run_cycle(stages=("normalisation",))) == ["normalisation"]
    assert [c[0] for c in calls] == ["normalisation"]


def test_second_cycle_is_skipped_while_one_runs(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls)
    with daemon._cycle_lock(db) as got:
        assert got
        assert "saute" in daemon.run_cycle()
    assert calls == []
    daemon.run_cycle()
    assert len(calls) == 3  # verrou rendu


def test_run_forever_heartbeat_nice_and_stop(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls)
    niced, handlers = [], {}
    monkeypatch.setattr(daemon.os, "nice", lambda n: niced.append(n))
    monkeypatch.setattr(daemon.signal, "signal", lambda s, h: handlers.__setitem__(s, h))
    monkeypatch.setattr(daemon, "_trim_log", lambda: None)
    assert daemon.run_forever(interval_s=900, max_cycles=1) == 1
    assert niced == [daemon.NICE]
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}
    svc = _status(db)["service"]
    c = json.loads(svc["counters"])
    assert c["etat"] == "arrêté" and c["cycles"] == 1 and c["dernier_cycle"] and c["interval_s"] == 900
    assert daemon.service_state(svc)["actif"] is False
    assert daemon.service_state(svc)["dernier_cycle"] == c["dernier_cycle"]


def test_sigterm_finishes_current_stage_then_stops(db, monkeypatch):
    calls: list = []
    _install_stages(monkeypatch, calls)
    collect = sys.modules["strate.veille.collect"]
    real = collect.run_once

    def run_once(con, **kw):
        daemon._handle_signal(signal.SIGTERM, None)  # launchctl stop pendant la collecte
        return real(con, **kw)

    collect.run_once = run_once
    monkeypatch.setattr(daemon.os, "nice", lambda n: None)
    monkeypatch.setattr(daemon.signal, "signal", lambda s, h: None)
    monkeypatch.setattr(daemon, "_trim_log", lambda: None)
    t = time.time()
    assert daemon.run_forever(interval_s=900) == 1  # sans max_cycles : c'est le signal qui arrête
    assert time.time() - t < 10  # pas d'attente du quart d'heure suivant
    assert [c[0] for c in calls] == ["collecte"]
    st = _status(db)
    assert st["collecte"]["ok"] == 1  # l'étape en cours a fini proprement
    assert json.loads(st["service"]["counters"])["etat"] == "arrêté"
    with pytest.raises(SystemExit):  # second signal : arrêt immédiat
        daemon._handle_signal(signal.SIGTERM, None)


def test_sleep_follows_wall_clock(monkeypatch):
    daemon._stop.clear()
    clock = [1000.0]
    monkeypatch.setattr(daemon.time, "time", lambda: clock[0])
    waits = []

    def wait(s):
        waits.append(s)
        clock[0] += 400  # le Mac a dormi : l'horloge murale avance plus que l'attente demandée
        return False

    monkeypatch.setattr(daemon._stop, "wait", wait)
    daemon._sleep_until(1900.0)
    assert waits == [daemon.POLL_S] * 3  # petits pas, réveil dès que l'heure est passée


def test_service_state(monkeypatch):
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    fresh = {"last_run": _iso(NOW - timedelta(minutes=5)), "message": "cycle 3 terminé",
             "counters": json.dumps({"etat": "en marche", "interval_s": 900, "pid": 42,
                                     "dernier_cycle": "2026-11-26T12:00:00Z"})}
    assert daemon.service_state(fresh) == {"actif": True, "dernier_cycle": "2026-11-26T12:00:00Z",
                                           "prochain": None, "message": "cycle 3 terminé"}
    stale = {**fresh, "last_run": _iso(NOW - timedelta(hours=2))}
    assert daemon.service_state(stale)["actif"] is False
    stopped = {**fresh, "counters": json.dumps({"etat": "arrêté", "interval_s": 900})}
    assert daemon.service_state(stopped)["actif"] is False
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: False)  # processus mort sans battement final
    assert daemon.service_state(fresh)["actif"] is False
    assert daemon.service_state(None)["actif"] is False


def test_trim_log(tmp_path):
    log = tmp_path / "veille.log"
    log.write_text("x" * 100)
    daemon._trim_log(log, max_bytes=50)
    assert log.read_text() == "" and (tmp_path / "veille.log.1").read_text() == "x" * 100


# -- launchd (rien n'est installé : launchctl est simulé) ------------------------------------------------
def test_install_writes_plist_and_bootstraps(tmp_path, monkeypatch):
    plist = tmp_path / "LaunchAgents" / "com.strate.veille.plist"
    logp = tmp_path / "logs" / "veille.log"
    monkeypatch.setattr(daemon, "PLIST_PATH", plist)
    monkeypatch.setattr(daemon, "LOG_PATH", logp)
    monkeypatch.setattr(daemon.shutil, "which", lambda name: "/opt/homebrew/bin/uv")
    ran = []
    monkeypatch.setattr(daemon, "_launchctl",
                        lambda *a: ran.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    r = daemon.install(manager="launchd")
    assert r["charge"] and r["gestionnaire"] == "launchd"
    pl = plistlib.loads(plist.read_bytes())
    assert pl["Label"] == "com.strate.veille" and pl["KeepAlive"] is True and pl["RunAtLoad"] is True
    assert pl["ProgramArguments"][0].endswith("/uv") and pl["ProgramArguments"][0].startswith("/")
    assert pl["ProgramArguments"][1:] == ["run", "--project", str(daemon.PROJECT_DIR), "strate", "veille",
                                          "run"]
    assert (daemon.PROJECT_DIR / "pyproject.toml").exists()
    assert pl["StandardOutPath"] == pl["StandardErrorPath"] == str(logp)
    assert logp.parent.is_dir()
    dom = f"gui/{daemon.os.getuid()}"
    assert ran == [("bootout", f"{dom}/com.strate.veille"), ("bootstrap", dom, str(plist))]
    ran.clear()
    u = daemon.uninstall(manager="launchd")
    assert u["plist_retire"] and u["fichier_retire"] and not plist.exists()
    assert ran == [("bootout", f"{dom}/com.strate.veille")]


def test_install_dry_run_writes_nothing(tmp_path, monkeypatch):
    plist = tmp_path / "com.strate.veille.plist"
    monkeypatch.setattr(daemon, "PLIST_PATH", plist)
    monkeypatch.setattr(daemon, "_launchctl", lambda *a: pytest.fail("launchctl ne doit pas être appelé"))
    r = daemon.install(dry_run=True, manager="launchd")
    assert not plist.exists() and r["contenu"]["KeepAlive"] is True and "<plist" in r["texte"]


# -- systemd (Linux, machine virtuelle : systemctl et loginctl sont simulés) --------------------------------
def _fake_run(ran, linger="no", fail=()):
    def run(prog):
        def f(*a):
            ran.append((prog, *a))
            bad = any(a[:len(x)] == x for x in fail)
            out = linger if prog == "loginctl" and a[:1] == ("show-user",) else ""
            return SimpleNamespace(returncode=1 if bad else 0, stdout=out, stderr="refusé" if bad else "")
        return f
    return run


def test_systemd_install_writes_unit_enables_restarts_and_lingers(tmp_path, monkeypatch):
    unit = tmp_path / "systemd" / "user" / "strate-veille.service"
    logp = tmp_path / "logs" / "veille.log"
    monkeypatch.setattr(daemon, "UNIT_PATH", unit)
    monkeypatch.setattr(daemon, "LOG_PATH", logp)
    monkeypatch.setattr(daemon.shutil, "which", lambda name: "/home/azureuser/.local/bin/uv")
    monkeypatch.setattr(daemon, "_user", lambda: "azureuser")
    ran: list = []
    run = _fake_run(ran)
    monkeypatch.setattr(daemon, "_systemctl", run("systemctl"))
    monkeypatch.setattr(daemon, "_loginctl", run("loginctl"))
    r = daemon.install(manager="systemd")
    assert r["charge"] and r["linger"] and r["message"] == "" and r["gestionnaire"] == "systemd"
    text = unit.read_text()
    assert text == r["texte"] and logp.parent.is_dir()
    lines = text.splitlines()
    exe = f'ExecStart="/home/azureuser/.local/bin/uv" run --project "{daemon.PROJECT_DIR}" strate veille run'
    assert exe in lines
    for want in ("Restart=always", "RestartSec=60", "StartLimitIntervalSec=0", "KillMode=mixed",
                 "TimeoutStopSec=60", "WantedBy=default.target", f"StandardOutput=append:{logp}",
                 f"StandardError=append:{logp}", 'Environment="PYTHONUNBUFFERED=1"'):
        assert want in lines, want
    assert not any(line.startswith("Nice=") for line in lines)  # os.nice suffit, pas de double baisse
    assert [x for x in ran if x[0] == "systemctl"] == [
        ("systemctl", "daemon-reload"), ("systemctl", "enable", "strate-veille.service"),
        ("systemctl", "restart", "strate-veille.service")]
    assert ("loginctl", "enable-linger", "azureuser") in ran
    ran.clear()
    u = daemon.uninstall(manager="systemd")
    assert u["arrete"] and u["fichier_retire"] and not unit.exists()
    assert ran == [("systemctl", "disable", "--now", "strate-veille.service"), ("systemctl", "daemon-reload")]


def test_systemd_linger_refused_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "UNIT_PATH", tmp_path / "strate-veille.service")
    monkeypatch.setattr(daemon, "LOG_PATH", tmp_path / "logs" / "veille.log")
    monkeypatch.setattr(daemon.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(daemon, "_user", lambda: "azureuser")
    ran: list = []
    run = _fake_run(ran, fail=(("enable-linger",),))
    monkeypatch.setattr(daemon, "_systemctl", run("systemctl"))
    monkeypatch.setattr(daemon, "_loginctl", run("loginctl"))
    monkeypatch.setattr(daemon, "_sudo_loginctl", run("sudo"))
    r = daemon.install(manager="systemd")
    assert ("sudo", "enable-linger", "azureuser") in ran  # sudo -n tenté avant d'abandonner
    assert r["charge"] and r["linger"] is False
    assert "sudo loginctl enable-linger azureuser" in r["message"]


def test_systemd_linger_already_on_is_not_requested_again(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "_user", lambda: "azureuser")
    ran: list = []
    monkeypatch.setattr(daemon, "_loginctl", _fake_run(ran, linger="yes")("loginctl"))
    assert daemon._linger() == {"linger": True, "message": ""}
    assert ran == [("loginctl", "show-user", "azureuser", "-p", "Linger", "--value")]


def test_systemd_failed_enable_stops_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "UNIT_PATH", tmp_path / "strate-veille.service")
    monkeypatch.setattr(daemon, "LOG_PATH", tmp_path / "logs" / "veille.log")
    monkeypatch.setattr(daemon.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(daemon, "_user", lambda: "azureuser")
    ran: list = []
    run = _fake_run(ran, linger="yes", fail=(("enable",),))
    monkeypatch.setattr(daemon, "_systemctl", run("systemctl"))
    monkeypatch.setattr(daemon, "_loginctl", run("loginctl"))
    r = daemon.install(manager="systemd")
    assert r["charge"] is False and "systemctl --user enable" in r["message"]
    assert ("systemctl", "restart", "strate-veille.service") not in ran


def test_systemd_dry_run_writes_nothing(tmp_path, monkeypatch):
    unit = tmp_path / "strate-veille.service"
    monkeypatch.setattr(daemon, "UNIT_PATH", unit)
    monkeypatch.setattr(daemon.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(daemon, "_systemctl", lambda *a: pytest.fail("systemctl ne doit pas être appelé"))
    monkeypatch.setattr(daemon, "_loginctl", lambda *a: pytest.fail("loginctl ne doit pas être appelé"))
    r = daemon.install(dry_run=True, manager="systemd")
    assert not unit.exists() and "[Service]" in r["texte"]


def test_systemd_unit_escapes_spaces_percent_dollar_and_quotes(monkeypatch):
    monkeypatch.setenv("STRATE_DATA", "/srv/strate data/100%/$HOME")
    text = daemon.unit_text('/opt/my tools/uv', Path("/home/a/my $proj"), Path("/var/log/st%rate.log"))
    lines = text.splitlines()
    assert 'ExecStart="/opt/my tools/uv" run --project "/home/a/my $$proj" strate veille run' in lines
    assert "WorkingDirectory=/home/a/my $proj" in lines
    assert 'Environment="STRATE_DATA=/srv/strate data/100%%/$HOME"' in lines  # Environment= : $ tel quel
    assert "StandardOutput=append:/var/log/st%%rate.log" in lines
    assert 'Environment="PATH=/opt/my tools:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"' in lines
    assert daemon._systemd_escape('a"b\\c', True) == '"a\\"b\\\\c"'
    assert daemon._systemd_escape("$x", True, dollar=True) == '"$$x"'
    assert daemon._systemd_escape("$x", True) == '"$x"'
    with pytest.raises(ValueError):
        daemon.unit_text("/usr/bin/uv", Path("/home/a/bad\nname"))


def test_systemd_state_parses_systemctl_show(tmp_path, monkeypatch):
    unit = tmp_path / "strate-veille.service"
    monkeypatch.setattr(daemon, "UNIT_PATH", unit)
    assert daemon.systemd_state() == {"installe": False, "charge": False}
    unit.write_text("[Service]\n")
    out = "ActiveState=active\nSubState=running\nMainPID=4242\nNRestarts=1\nExecMainStatus=0\n"
    monkeypatch.setattr(daemon, "_systemctl", lambda *a: SimpleNamespace(returncode=0, stdout=out, stderr=""))
    st = daemon.systemd_state()
    assert st == {"installe": True, "charge": True, "state": "active", "substate": "running", "pid": "4242",
                  "restarts": "1", "last_exit_code": "0"}
    out = "ActiveState=failed\nSubState=failed\nMainPID=0\nNRestarts=3\nExecMainStatus=1\n"
    assert daemon.systemd_state()["charge"] is False
    monkeypatch.setattr(daemon, "_systemctl", lambda *a: SimpleNamespace(
        returncode=1, stdout="", stderr="Failed to connect to bus: No medium found"))
    st = daemon.systemd_state()  # gestionnaire injoignable : inconnu, pas « arrêté »
    assert st["charge"] is None and "bus" in st["erreur"]


def test_service_manager_by_platform(monkeypatch):
    assert daemon.service_manager("darwin") == "launchd"
    assert daemon.service_manager("linux") == "systemd"
    with pytest.raises(RuntimeError):
        daemon.service_manager("win32")
    monkeypatch.setattr(daemon, "service_manager", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert daemon.manager_state() == {"gestionnaire": None, "installe": False, "charge": False}


# -- CLI -----------------------------------------------------------------------------------------------
def test_cli_status_and_once(db, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "manager_state",
                        lambda: {"gestionnaire": "launchd", "installe": False, "charge": False})
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "status"])
    cli.main()
    out = capsys.readouterr().out
    assert "absente" in out and "non installé" in out
    assert not db.exists()  # status ne crée pas la base

    con = store.init(db)
    store.set_status(con, "collecte", False, "erreur : réseau", fichiers=0)
    con.commit()
    con.close()
    cli.main()
    out = capsys.readouterr().out
    assert "collecte" in out and "ERREUR" in out and "fichiers=0" in out and "documents 0" in out

    seen = {}
    monkeypatch.setattr(daemon, "run_cycle", lambda limits, stages: seen.update(l=limits, s=stages) or {})
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "once", "--stage", "collecte", "--max-files", "2"])
    cli.main()
    assert seen == {"l": {"collecte": {"max_files": 2}}, "s": ("collecte",)}


# -- API -----------------------------------------------------------------------------------------------
ROUTES = ["/api/veille/status", "/api/veille/types", "/api/veille/events", "/api/veille/stats",
          "/api/veille/correlations", "/api/veille/site?site=x", "/api/veille/tracked"]


def test_api_without_database(db):
    c = TestClient(api.app)
    for r in ROUTES:
        assert c.get(r).status_code == 200, r
    assert not db.exists()  # lecture seule : la base n'est pas créée par l'API
    s = c.get("/api/veille/status").json()
    assert s["stages"] == [] and set(s["totaux"].values()) == {0}
    assert s["service"]["actif"] is False and s["service"]["dernier_cycle"] is None
    assert set(s["totaux"]) == {"docs", "docs_en_attente", "rejetes", "events", "mines_avec_evenements",
                                "suivies", "fichiers_faits", "fichiers_manquants"}
    assert c.get("/api/veille/events").json() == {"total": 0, "events": []}
    st = c.get("/api/veille/stats?days=7").json()
    assert len(st["par_jour"]) == 8 and all(d["n"] == 0 for d in st["par_jour"])
    assert st["par_type"] == {} and st["top_mines"] == []
    co = c.get("/api/veille/correlations").json()
    assert co["run_at"] is None and co["rows"] == [] and co["methode"]
    assert c.get("/api/veille/site?site=x").json() == {"events": [], "volume": [], "tracked": None}
    assert c.get("/api/veille/tracked").json() == []
    assert c.get("/api/veille/types").json() == store.EVENT_TYPES


def test_api_with_empty_file_without_tables(db):
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).close()
    c = TestClient(api.app)
    for r in ROUTES:
        assert c.get(r).status_code == 200, r


def _populate(path):
    con = store.init(path)
    docs = [  # (id, site, type, jours avant, lang, pays, confiance)
        ("e1", "olympic", "accident", 1, "en", "AS", 0.9),
        ("e2", "olympic", "social", 3, "es", "AS", 0.8),
        ("e3", "escondida", "social", 2, "es", "CI", 0.7),
        ("e4", "escondida", "finance", 40, "es", "CI", 0.9),  # hors fenêtre de 30 jours
        ("e5", "grasberg", "autre", 5, "id", None, 0.3),  # sous le seuil de confiance
    ]
    for i, site, typ, ago, lang, country, conf in docs:
        pub = _iso(NOW - timedelta(days=ago))
        con.execute("INSERT INTO raw_docs(id, source, fetched_at, url, status) VALUES (?,?,?,?,?)",
                    (f"d{i}", "gdelt_gkg", pub, f"https://ex.com/{i}", "done"))
        con.execute("INSERT INTO events(id, doc_id, site_id, site_name, event_type, published_at, month, "
                    "title, url, domain, lang, country, lat, lon, confidence, match_method, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, f"d{i}", site, site.title(), typ, pub, pub[:7], f"titre {i}", f"https://ex.com/{i}",
                     "ex.com", lang, country, -30.4, 136.9, conf, "nom", pub))
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, url, status) "
                "VALUES ('n1','gdelt_gkg','x','u','new')")
    con.execute("INSERT INTO raw_docs(id, source, fetched_at, url, status, reject_reason) "
                "VALUES ('r1','gdelt_gkg','x','u','rejected','date future')")
    for name, st in (("a.gkg.csv.zip", "done"), ("b.gkg.csv.zip", "done"), ("c.gkg.csv.zip", "missing")):
        con.execute("INSERT INTO feed_files(name, feed, stamp, status) VALUES (?,?,?,?)",
                    (name, "gkg", "1", st))
    for m, n, src in (("2026-09", 12, "gdelt_doc_timeline"), ("2026-09", 2, "events"),
                      ("2026-10", 3, "events")):
        con.execute("INSERT INTO news_volume VALUES (?,?,?,?,?)", ("olympic", m, n, src, "x"))
    con.execute("INSERT INTO tracked(site_id, site_name, reason, since, last_event_at, events_count, "
                "analysis_status) VALUES ('olympic','Olympic','actualité : 2 événements en 30 jours','x',?,2,"
                "'terminé')", (_iso(NOW - timedelta(days=1)),))
    con.execute("INSERT INTO tracked(site_id, site_name, reason, since, last_event_at, events_count) "
                "VALUES ('escondida','Escondida','actualité','x',?,1)", (_iso(NOW - timedelta(days=2)),))
    corr = ("INSERT INTO correlations(run_at, scope, event_type, signal, lag_months, n_mines, n_events, "
            "lift, lift_lo, lift_hi, rho, rho_lo, rho_hi, p_value, q_value, verdict, details) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    con.execute(corr, ("2026-11-01T00:00:00Z", "global", "accident", "bare_km2", 0, 5, 10, 9.0, 1, 2,
                       None, None, None, 0.5, 0.5, "non démontré", None))  # ancien calcul
    con.execute(corr, ("2026-11-26T00:00:00Z", "global", "accident", "bare_km2", 0, 8, 20, float("inf"), 1.1,
                       float("inf"), None, None, None, 0.2, 0.3, "non démontré", None))
    con.execute(corr, ("2026-11-26T00:00:00Z", "global", "social", "activity_db", 1, 8, 20, 2.0, 1.4, 3.1,
                       None, None, None, 0.001, 0.02, "exploratoire", None))  # décalage ≠ 0 : exploratoire
    store.set_status(con, "service", True, "cycle 1 terminé", etat="en marche", interval_s=900,
                     dernier_cycle="2026-11-26T12:00:00Z")
    store.set_status(con, "normalisation", True, "ok", lus=5)
    store.set_status(con, "collecte", False, "erreur : 404")
    con.commit()
    con.close()


def test_api_with_data(db):
    _populate(db)
    c = TestClient(api.app)
    s = c.get("/api/veille/status").json()
    assert [x["stage"] for x in s["stages"]] == ["collecte", "normalisation", "service"]
    assert s["stages"][0]["ok"] is False and s["stages"][1]["counters"] == {"lus": 5}
    assert s["totaux"] == {"docs": 7, "docs_en_attente": 1, "rejetes": 1, "events": 5,
                           "mines_avec_evenements": 3, "suivies": 2, "fichiers_faits": 2,
                           "fichiers_manquants": 1}
    assert s["service"]["dernier_cycle"] == "2026-11-26T12:00:00Z"

    ev = c.get("/api/veille/events").json()
    assert ev["total"] == 4 and [e["id"] for e in ev["events"]] == ["e1", "e3", "e2", "e4"]  # récents d'abord
    assert set(ev["events"][0]) == {"id", "site_id", "site_name", "event_type", "published_at", "title",
                                    "url", "domain", "lang", "country", "lat", "lon", "confidence",
                                    "match_method"}
    assert c.get("/api/veille/events?min_conf=0").json()["total"] == 5
    assert [e["id"] for e in c.get("/api/veille/events?type=social&lang=ES").json()["events"]] == ["e3", "e2"]
    assert c.get("/api/veille/events?site=escondida&country=ci").json()["total"] == 2
    # horodatage complet : une date seule serait tronquée à minuit et dépendrait de l'heure du test
    since = (NOW - timedelta(days=2, hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert {e["id"] for e in c.get(f"/api/veille/events?since={since}").json()["events"]} == {"e1", "e3"}
    lim = c.get("/api/veille/events?limit=1").json()
    assert lim["total"] == 4 and len(lim["events"]) == 1
    assert c.get("/api/veille/events?since=hier").status_code == 422

    st = c.get("/api/veille/stats?days=30").json()
    assert st["par_type"] == {"social": 2, "accident": 1}
    assert st["par_langue"] == {"es": 2, "en": 1} and st["par_pays"] == {"AS": 2, "CI": 1}
    assert st["top_mines"][0] == {"site_id": "olympic", "site_name": "Olympic", "n": 2}
    assert len(st["par_jour"]) == 31 and sum(d["n"] for d in st["par_jour"]) == 3
    assert st["par_jour"] == sorted(st["par_jour"], key=lambda d: d["jour"])

    co = c.get("/api/veille/correlations").json()
    assert co["run_at"] == "2026-11-26T00:00:00Z" and len(co["rows"]) == 2  # dernier calcul seulement
    # famille principale (décalage 0) d'abord, même si l'exploratoire a un q plus petit
    assert [(r["lag_months"], r["famille"]) for r in co["rows"]] == [(0, "principale"), (1, "exploratoire")]
    assert co["rows"][1]["verdict"] == "exploratoire"
    assert co["rows"][0]["lift"] is None and co["rows"][0]["lift_hi"] is None  # infini → null
    from strate.veille import correlate

    assert co["methode"] == correlate.METHODE  # une seule source pour le texte de méthode

    site = c.get("/api/veille/site?site=olympic").json()
    assert [e["id"] for e in site["events"]] == ["e1", "e2"]
    # l'historique GDELT prime sur le compte des événements, qui complète les autres mois
    assert site["volume"] == [{"month": "2026-09", "n_articles": 12.0, "source": "gdelt_doc_timeline"},
                              {"month": "2026-10", "n_articles": 3.0, "source": "events"}]
    assert site["tracked"]["analysis_status"] == "terminé"

    tr = c.get("/api/veille/tracked").json()
    assert [t["site_id"] for t in tr] == ["olympic", "escondida"]
    assert set(tr[0]) == {"site_id", "site_name", "reason", "since", "last_event_at", "events_count",
                          "analysis_status"}


def test_api_site_follows_twin(db, monkeypatch):
    _populate(db)
    monkeypatch.setattr(screener, "twins", lambda: {"olympic-dam": "olympic"})  # suivie → jumelle catalogue
    c = TestClient(api.app)
    assert [e["id"] for e in c.get("/api/veille/site?site=olympic-dam").json()["events"]] == ["e1", "e2"]
    assert c.get("/api/veille/events?site=olympic-dam").json()["total"] == 2


def test_site_detail_carries_veille_summary(db, monkeypatch):
    s = SimpleNamespace(id="olympic", name="Olympic Dam", country="Australia", region=None, lat=-30.4,
                        lon=136.9, commodity=["copper"], stage="production", owners=[], coord_source="x",
                        radius_m=3000, wikipedia=None)
    monkeypatch.setattr(api, "_resolve", lambda site: site)
    monkeypatch.setattr(api, "_site", lambda site: s)
    monkeypatch.setattr(api, "_catalog_row", lambda site: {})
    monkeypatch.setattr(api, "_graph", lambda: SimpleNamespace(sites={"olympic": s}, companies={}))
    monkeypatch.setattr(api, "_emprise_estimate", lambda site: None)
    monkeypatch.setattr(screener, "analysis_state",
                        lambda site: {"analyse": False, "donnees_anciennes": False, "derniere_date": None})
    c = TestClient(api.app)
    assert c.get("/api/site?site=olympic").json()["veille"] == {"events_30j": 0, "dernier": None,
                                                                "events_count": 0}
    _populate(db)
    v = c.get("/api/site?site=olympic").json()["veille"]
    assert v == {"events_30j": 2, "dernier": _iso(NOW - timedelta(days=1)), "events_count": 2}
    # events_count compte tous les événements (comme /api/veille/site par défaut), confiance basse comprise
    s.id = "grasberg"
    v = c.get("/api/site?site=grasberg").json()["veille"]
    assert v == {"events_30j": 0, "dernier": None, "events_count": 1}


# -- corrections après revue -----------------------------------------------------------------------------
def test_nan_counters_never_break_status(db, monkeypatch):
    """Une étape qui rend NaN/inf : le cycle passe, la base n'a pas de `NaN`, l'API répond 200 (null)."""
    calls: list = []
    _install_stages(monkeypatch, calls, **{
        "strate.veille.correlate": {"writes_status": False,
                                    "counters": {"lift_moyen": float("nan"), "max": float("inf"),
                                                 "liste": [1.0, float("-inf")]}},
    })
    res = daemon.run_cycle()
    assert res["correlation"]["ok"] and res["correlation"]["lift_moyen"] is None
    raw = _status(db)["correlation"]["counters"]
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=lambda x: pytest.fail(f"constante non JSON : {x}"))
    c = TestClient(api.app)
    r = c.get("/api/veille/status")
    assert r.status_code == 200
    st = {x["stage"]: x for x in r.json()["stages"]}
    assert st["correlation"]["counters"] == {"lift_moyen": None, "max": None, "liste": [1.0, None]}
    # une ligne écrite par un autre programme avec un NaN littéral : nettoyée à la lecture
    con = sqlite3.connect(db)
    con.execute("INSERT OR REPLACE INTO pipeline_status VALUES ('collecte', ?, 1, 'x', ?)",
                (_iso(NOW), '{"x": NaN, "y": {"z": Infinity}}'))
    con.execute("INSERT OR REPLACE INTO pipeline_status VALUES ('service', ?, 1, 'x', ?)",
                (_iso(NOW), '{"etat": "en marche", "interval_s": NaN, "pid": 1}'))
    con.commit()
    con.close()
    r = c.get("/api/veille/status")
    assert r.status_code == 200
    stages = {x["stage"]: x for x in r.json()["stages"]}
    assert stages["collecte"]["counters"] == {"x": None, "y": {"z": None}}


def test_stage_own_failure_status_is_reported(db, monkeypatch):
    """Une étape qui note elle-même son échec (set_status ok=False) puis rend ses compteurs n'est pas
    comptée « ok » par le service, ni dans le battement, ni dans `veille once`."""
    calls: list = []
    _install_stages(monkeypatch, calls)
    corr = sys.modules["strate.veille.correlate"]

    def run_once(con, **kw):
        store.set_status(con, "correlation", False, "étude : réseau coupé", lignes=0)
        return {"lignes": 0, "ok": True}  # une clé « ok » des compteurs ne masque pas l'échec

    corr.run_once = run_once
    res = daemon.run_cycle()
    assert res["collecte"]["ok"] and res["normalisation"]["ok"]
    assert res["correlation"]["ok"] is False and res["correlation"]["erreur"] == "étude : réseau coupé"
    assert res["correlation"]["lignes"] == 0
    st = _status(db)["correlation"]
    assert st["ok"] == 0 and st["message"] == "étude : réseau coupé"  # le service ne l'écrase pas

    monkeypatch.setattr(daemon.os, "nice", lambda n: None)
    monkeypatch.setattr(daemon.signal, "signal", lambda s, h: None)
    monkeypatch.setattr(daemon, "_trim_log", lambda: None)
    daemon.run_forever(interval_s=900, max_cycles=1)
    c = json.loads(_status(db)["service"]["counters"])
    assert c["etapes_ok"] == 2 and c["etapes"] == 3


def test_corrupt_database_gives_empty_answers(db, monkeypatch, capsys):
    db.parent.mkdir(parents=True)
    db.write_bytes(b"ceci n'est pas une base SQLite" * 200)
    c = TestClient(api.app)
    for r in ROUTES:
        assert c.get(r).status_code == 200, r
    assert c.get("/api/veille/status").json()["stages"] == []
    monkeypatch.setattr(daemon, "manager_state",
                        lambda: {"gestionnaire": "launchd", "installe": False, "charge": False})
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "status"])
    cli.main()  # pas de trace d'erreur
    out = capsys.readouterr().out
    assert "base illisible" in out and "inactif" in out


def test_service_counters_not_an_object(db, monkeypatch):
    con = store.init(db)
    ins = "INSERT OR REPLACE INTO pipeline_status VALUES (?, ?, 1, 'x', ?)"
    con.execute(ins, ("service", _iso(NOW), "[1, 2]"))
    con.execute(ins, ("collecte", _iso(NOW), "3"))
    con.commit()
    con.close()
    c = TestClient(api.app)
    r = c.get("/api/veille/status")
    assert r.status_code == 200
    j = r.json()
    assert j["service"]["dernier_cycle"] is None
    assert {x["stage"]: x for x in j["stages"]}["collecte"]["counters"] == {}
    for row in ({"last_run": _iso(NOW), "counters": [1, 2]}, {"last_run": 5, "counters": "3"},
                {"last_run": None, "counters": {"interval_s": "x", "dernier_cycle": 7}}):
        st = daemon.service_state(row)
        assert st["actif"] in (True, False) and st["dernier_cycle"] is None
    monkeypatch.setattr(daemon, "manager_state",
                        lambda: {"gestionnaire": "launchd", "installe": False, "charge": False})
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "status"])
    cli.main()  # pas de trace d'erreur


def test_interval_is_validated(db, monkeypatch, capsys):
    with pytest.raises(ValueError):
        daemon.next_boundary(1000.0, 0)
    with pytest.raises(ValueError):
        daemon.next_boundary(1000.0, -900)
    with pytest.raises(ValueError):
        daemon.run_forever(interval_s=0, max_cycles=1)
    assert not db.exists()  # refusé avant d'écrire quoi que ce soit
    started = []
    monkeypatch.setattr(daemon, "run_forever", lambda interval_s: started.append(interval_s))
    for bad in ("0", "-5", "10", "abc"):
        monkeypatch.setattr(sys, "argv", ["strate", "veille", "run", "--interval", bad])
        with pytest.raises(SystemExit):
            cli.main()
    assert started == [] and "--interval" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "run", "--interval", "600"])
    cli.main()
    assert started == [600]


def test_stop_requested_and_plist_timeout_documented():
    daemon._stop.clear()
    assert daemon.stop_requested() is False
    daemon._stop.set()
    assert daemon.stop_requested() is True
    daemon._stop.clear()
    pl = daemon.plist_dict("/opt/homebrew/bin/uv")
    assert pl["ExitTimeOut"] == 60
    assert "tuée" in daemon.plist_dict.__doc__ and "stop_requested" in daemon.__doc__


def test_reserved_counter_names_never_fail_a_stage(db, monkeypatch):
    """Une étape qui rend « message » ou « ok » parmi ses compteurs reste réussie (compteur renommé)."""
    calls: list = []
    _install_stages(monkeypatch, calls, **{
        "strate.veille.collect": {"writes_status": False, "counters": {"message": "rien de neuf", "n": 1}},
        "strate.veille.normalize": {"writes_status": False, "counters": {"ok": 3, "stage": "x"}},
    })
    res = daemon.run_cycle()
    assert res["collecte"]["ok"] is True and res["collecte"]["message_etape"] == "rien de neuf"
    assert res["normalisation"]["ok"] is True and res["normalisation"]["ok_etape"] == 3
    st = _status(db)
    assert st["collecte"]["ok"] == 1 and st["normalisation"]["ok"] == 1
    assert json.loads(st["normalisation"]["counters"])["stage_etape"] == "x"
    assert daemon._counters({"ok": 1, "ok_etape": 2}) == {"ok_etape": 1, "ok_etape_etape": 2}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "900", None])
def test_interval_must_be_finite_positive(db, bad):
    with pytest.raises(ValueError):
        daemon.next_boundary(1000.0, bad)
    with pytest.raises(ValueError):
        daemon.run_forever(interval_s=bad, max_cycles=1)
    assert not db.exists()


def test_old_database_gains_verdict_rho_column(tmp_path):
    # base créée avant la colonne verdict_rho : init l'ajoute, sans perdre les lignes
    import sqlite3
    db = tmp_path / "veille.db"
    old = "\n".join(x for x in store.SCHEMA.splitlines() if "verdict_rho" not in x)
    assert "verdict_rho" not in old
    c = sqlite3.connect(db)
    c.executescript(old)
    c.execute("INSERT INTO correlations(run_at, scope, event_type, signal, lag_months, n_mines, n_events, "
              "verdict) VALUES ('r', 'global', 'accident', 'bare_km2', 0, 5, 10, 'non démontré')")
    c.commit()
    c.close()
    con = store.init(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(correlations)")]
    assert cols[-1] == "verdict_rho" and con.execute("SELECT COUNT(*) FROM correlations").fetchone()[0] == 1
    store.init(db).close()                                 # deuxième passage : rien à ajouter, pas d'erreur
    con.close()


def test_stage_status_rolled_back_by_the_stage_is_not_trusted(db, monkeypatch):
    # l'étape écrit son statut puis annule sa propre transaction : ligne absente (1er cycle) ou ligne du
    # cycle précédent. Avant correction : TypeError sur after[1] et étape notée en erreur.
    calls: list = []
    _install_stages(monkeypatch, calls)
    norm = sys.modules["strate.veille.normalize"]

    def run_once(con, **kw):
        store.set_status(con, "normalisation", False, "échec provisoire", lus=0)
        con.rollback()
        return {"lus": 7}

    norm.run_once = run_once
    res = daemon.run_cycle(stages=("normalisation",))
    assert res["normalisation"]["ok"] is True and res["normalisation"]["lus"] == 7
    st = _status(db)["normalisation"]
    assert st["ok"] == 1 and "annulé" in st["message"] and json.loads(st["counters"]) == {"lus": 7}
    # 2e cycle : la ligne du 1er cycle est toujours là, différente de celle que l'étape a écrite et annulée
    res = daemon.run_cycle(stages=("normalisation",))
    assert res["normalisation"]["ok"] is True and "annulé" in _status(db)["normalisation"]["message"]


def test_status_watch_does_not_replace_set_status_and_is_per_thread(db):
    import threading
    orig = store.set_status
    con = store.init(db)
    other: dict = {}

    def in_thread():
        c2 = store.init(db)
        with store.watch_status() as w2:
            store.set_status(c2, "collecte", True, "fil")
        other.update(w2)
        c2.commit()
        c2.close()

    with store.watch_status() as seen:
        assert store.set_status is orig                     # aucun remplacement dans le module
        th = threading.Thread(target=in_thread)
        th.start()
        th.join()
        store.set_status(con, "normalisation", True, "principal")
    con.commit()
    assert set(seen) == {"normalisation"} and set(other) == {"collecte"}
    store.set_status(con, "correlation", True, "hors relevé")    # hors du contexte : rien n'est relevé
    assert "correlation" not in seen
    con.close()


# -- copie de la base (rapatrier la veille d'une machine du cloud) -----------------------------------------
def test_backup_copies_a_consistent_database_while_open(db, tmp_path, monkeypatch, capsys):
    con = store.init(db)
    store.set_status(con, "collecte", True, "ok", fichiers=3)
    con.commit()  # la base reste ouverte, comme pendant que le service tourne
    dest = tmp_path / "rapatrie" / "veille.db"
    r = store.backup(dest, db)
    assert r["octets"] > 0 and not dest.with_name("veille.db.tmp").exists()
    c2 = sqlite3.connect(dest)
    assert c2.execute("SELECT message FROM pipeline_status WHERE stage='collecte'").fetchone() == ("ok",)
    c2.close()
    con.close()
    with pytest.raises(ValueError):
        store.backup(db, db)
    with pytest.raises(FileNotFoundError):
        store.backup(dest, tmp_path / "absente.db")
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "backup", "--to", str(tmp_path / "cli.db")])
    cli.main()
    assert "copiée depuis" in capsys.readouterr().out and (tmp_path / "cli.db").exists()
    monkeypatch.setattr(sys, "argv", ["strate", "veille", "backup"])
    with pytest.raises(SystemExit, match="--to"):
        cli.main()


@pytest.mark.skipif(not shutil.which("systemd-analyze") or not shutil.which("uv"),
                    reason="systemd-analyze absent (macOS) : vérifié sur la CI Linux")
def test_systemd_unit_passes_systemd_analyze_verify(tmp_path, monkeypatch):
    """L'unité générée est lue par le vrai systemd (CI Ubuntu) : syntaxe, ExecStart, guillemets."""
    unit = tmp_path / "strate-veille.service"
    unit.write_text(daemon.unit_text(os.path.abspath(shutil.which("uv")), log_path=tmp_path / "veille.log"))
    r = subprocess.run(["systemd-analyze", "--user", "verify", str(unit)], capture_output=True, text=True,
                       check=False)
    problems = [ln for ln in (r.stderr + r.stdout).splitlines() if "strate-veille" in ln]
    assert r.returncode == 0 and not problems, r.stderr + r.stdout


def test_linger_falls_back_to_passwordless_sudo(monkeypatch):
    monkeypatch.setattr(daemon, "_user", lambda: "azureuser")
    ran: list = []
    run = _fake_run(ran, fail=(("enable-linger",),))
    monkeypatch.setattr(daemon, "_loginctl", run("loginctl"))
    monkeypatch.setattr(daemon, "_sudo_loginctl", _fake_run(ran)("sudo"))
    assert daemon._linger() == {"linger": True, "message": ""}


def test_user_env_restores_runtime_dir_outside_a_login_session(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    real = daemon.Path

    def fake_path(p, *a):
        return real(tmp_path) if str(p).startswith("/run/user/") else real(p, *a)
    monkeypatch.setattr(daemon, "Path", fake_path)
    assert daemon._user_env()["XDG_RUNTIME_DIR"] == str(tmp_path)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/deja/la")
    assert daemon._user_env()["XDG_RUNTIME_DIR"] == "/deja/la"


def test_backup_path_with_hash_copies_the_right_file(tmp_path):
    src = tmp_path / "q#dir" / "veille.db"
    con = store.init(src)
    store.set_status(con, "collecte", True, "vrai fichier")
    con.commit()
    con.close()
    dest = tmp_path / "copie.db"
    store.backup(dest, src)
    c = sqlite3.connect(dest)
    assert c.execute("SELECT message FROM pipeline_status").fetchone() == ("vrai fichier",)
    c.close()
    assert not (tmp_path / "q").exists()
