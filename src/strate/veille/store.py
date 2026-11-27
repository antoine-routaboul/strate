"""Base de la veille : une seule base SQLite partagée par les trois étapes de la chaîne.

    collecteur ──raw_docs──▶ normaliseur ──events──▶ corrélateur ──correlations──▶ API / interface
         ▲                        │                       │
         └── feed_files           └── tracked ◀───────────┘ (mise en suivi automatique)

Chaque étape lit ce que la précédente a écrit et marque ce qu'elle a traité : on peut arrêter et
relancer n'importe quand (Mac en veille, coupure réseau) sans rien perdre ni rien traiter deux fois.
SQLite en mode WAL : l'API lit pendant que les étapes écrivent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from strate.core import DATA_DIR

DB_PATH = DATA_DIR / "veille" / "veille.db"
SCHEMA_VERSION = 1

SCHEMA = """
-- fichiers GDELT (flux 15 min) : à récupérer, récupérés, absents pour l'instant
CREATE TABLE IF NOT EXISTS feed_files (
    name        TEXT PRIMARY KEY,          -- ex. 20261125120000.translation.gkg.csv.zip
    feed        TEXT NOT NULL,             -- 'gkg' (anglais) | 'gkg_tr' (traduit, ~65 langues)
    stamp       TEXT NOT NULL,             -- AAAAMMJJHHMMSS (UTC)
    status      TEXT NOT NULL,             -- 'pending' | 'done' | 'missing' | 'error'
    tries       INTEGER NOT NULL DEFAULT 0,
    last_try    TEXT,
    n_docs      INTEGER,                   -- documents lus dans le fichier
    n_kept      INTEGER,                   -- documents miniers gardés
    message     TEXT
);

-- documents bruts (étape 1) : un article = une ligne, identifié par l'empreinte de son URL
CREATE TABLE IF NOT EXISTS raw_docs (
    id            TEXT PRIMARY KEY,        -- sha1(url normalisée)
    source        TEXT NOT NULL,           -- 'gdelt_gkg' | 'gdelt_gkg_tr' | 'gdelt_doc' | autre
    fetched_at    TEXT NOT NULL,           -- ISO 8601 UTC
    published_at  TEXT,                    -- ISO 8601 UTC (si connu)
    url           TEXT NOT NULL,
    domain        TEXT,
    lang          TEXT,                    -- code ISO 639 (srclc GDELT, ou détecté)
    title         TEXT,                    -- titre tel que publié (langue d'origine)
    themes        TEXT,                    -- thèmes GDELT séparés par ';'
    locations     TEXT,                    -- JSON : [{name, country, adm1, lat, lon, type}]
    organizations TEXT,                    -- séparées par ';'
    persons       TEXT,                    -- séparées par ';'
    tone          REAL,
    hint_site     TEXT,                    -- mine visée par la requête (source gdelt_doc), sinon NULL
    payload       TEXT,                    -- JSON : champs bruts supplémentaires
    status        TEXT NOT NULL DEFAULT 'new',   -- 'new' | 'done' | 'rejected'
    reject_reason TEXT
);
CREATE INDEX IF NOT EXISTS raw_docs_status ON raw_docs(status);

-- événements validés (étape 2) : un article peut concerner plusieurs mines (une ligne par mine)
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,        -- sha1(doc_id + site_id)
    doc_id        TEXT NOT NULL REFERENCES raw_docs(id),
    site_id       TEXT NOT NULL,           -- id du catalogue / de la liste suivie
    site_name     TEXT,
    event_type    TEXT NOT NULL,           -- voir EVENT_TYPES
    published_at  TEXT NOT NULL,           -- ISO 8601 UTC
    month         TEXT NOT NULL,           -- AAAA-MM (pour l'étude d'événements)
    title         TEXT,
    url           TEXT NOT NULL,
    domain        TEXT,
    lang          TEXT,
    country       TEXT,
    lat           REAL,
    lon           REAL,
    confidence    REAL NOT NULL,           -- 0–1 : force du rattachement à la mine × du classement
    match_method  TEXT NOT NULL,           -- ex. 'nom+lieu', 'nom', 'requête'
    evidence      TEXT,                    -- JSON : ce qui a déclenché le rattachement et le type
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_site ON events(site_id, month);
CREATE INDEX IF NOT EXISTS events_time ON events(published_at);

-- volume d'actualité mensuel par mine (historique via l'API documentaire GDELT)
CREATE TABLE IF NOT EXISTS news_volume (
    site_id   TEXT NOT NULL,
    month     TEXT NOT NULL,               -- AAAA-MM
    n_articles REAL NOT NULL,
    source    TEXT NOT NULL,               -- 'gdelt_doc_timeline' | 'events'
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (site_id, month, source)
);

-- mines suivies automatiquement (étape 3)
CREATE TABLE IF NOT EXISTS tracked (
    site_id        TEXT PRIMARY KEY,
    site_name      TEXT,
    reason         TEXT NOT NULL,          -- ex. 'actualité : 3 événements en 30 jours'
    since          TEXT NOT NULL,
    last_event_at  TEXT,
    events_count   INTEGER NOT NULL DEFAULT 0,
    analysis_status TEXT,                  -- 'en file' | 'terminé' | 'erreur : …' | NULL
    analysis_at    TEXT,
    backfill_status TEXT,                  -- historique d'actualité : NULL | 'fait' | 'erreur : …'
    backfill_at    TEXT
);

-- résultats de corrélation (étape 3) : une ligne par (type d'événement, signal, décalage)
CREATE TABLE IF NOT EXISTS correlations (
    run_at      TEXT NOT NULL,
    scope       TEXT NOT NULL,             -- 'global' ou id de mine
    event_type  TEXT NOT NULL,             -- type, ou 'volume' (volume d'actualité)
    signal      TEXT NOT NULL,             -- signal satellite (bare_km2, activity_db…)
    lag_months  INTEGER NOT NULL,          -- décalage : satellite au mois de l'événement + lag
    n_mines     INTEGER NOT NULL,
    n_events    INTEGER NOT NULL,
    lift        REAL,                      -- P(anomalie | événement) / P(anomalie | sans événement)
    lift_lo     REAL,
    lift_hi     REAL,
    rho         REAL,                      -- corrélation de rang (volume vs signal), sinon NULL
    rho_lo      REAL,
    rho_hi      REAL,
    p_value     REAL,                      -- CMH exacte (lift) ; loi normale (volume ↔ signal)
    q_value     REAL,                      -- p corrigée des comparaisons multiples (Benjamini-Hochberg)
    verdict     TEXT NOT NULL,             -- 'lien démontré' (q < 0,10) | 'non démontré' | 'trop peu…'
    details     TEXT,                      -- JSON
    PRIMARY KEY (run_at, scope, event_type, signal, lag_months)
);

-- état de chaque étape (pour la page « Veille » et le diagnostic)
CREATE TABLE IF NOT EXISTS pipeline_status (
    stage     TEXT PRIMARY KEY,            -- 'collecte' | 'normalisation' | 'correlation' | 'service'
    last_run  TEXT,
    ok        INTEGER,
    message   TEXT,
    counters  TEXT                         -- JSON
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

EVENT_TYPES = {
    "accident": "Accident, effondrement, victimes",
    "social": "Grève, conflit social, manifestation",
    "juridique": "Justice, régulation, permis, sanctions",
    "environnement": "Pollution, digue à résidus, impact environnemental",
    "operations": "Production, expansion, arrêt, redémarrage",
    "finance": "Rachat, fusion, financement, résultats, faillite",
    "securite": "Attaque, enlèvement, insécurité autour du site",
    "autre": "Actualité minière sans type précis",
}

_lock = threading.Lock()


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init(path: Path | None = None) -> sqlite3.Connection:
    con = connect(path)
    with _lock:
        con.executescript(SCHEMA)
        con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),))
        con.commit()
    return con


@contextmanager
def session(path: Path | None = None):
    """Connexion initialisée, validée à la sortie, annulée en cas d'erreur."""
    con = init(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def set_status(con: sqlite3.Connection, stage: str, ok: bool, message: str = "", **counters) -> None:
    con.execute(
        "INSERT OR REPLACE INTO pipeline_status(stage, last_run, ok, message, counters) VALUES (?,?,?,?,?)",
        (stage, now(), int(ok), message, json.dumps(counters, ensure_ascii=False)),
    )
