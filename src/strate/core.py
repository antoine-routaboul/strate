"""Contrat commun à tous les modules + stockage.

Un module produit des signaux au format « long » :
    entity · date (mois, 1er jour) · signal · value · unit · quality · source
Tout le reste (API, widgets OpenBB, fusion) se branche sur ce format.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd

COLUMNS = ["entity", "date", "signal", "value", "unit", "quality", "source"]

DATA_DIR = Path(os.environ.get("STRATE_DATA", Path(__file__).resolve().parents[2] / "data"))


class Module(ABC):
    id: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    entity_kind: ClassVar[str]  # "site" ou "company"
    signals: ClassVar[dict[str, str]]  # nom du signal -> explication

    @abstractmethod
    def run(self, entity_id: str, start: date, end: date) -> pd.DataFrame:
        """Calcule les signaux d'une entité sur [start, end]. Retourne un DataFrame COLUMNS."""

    # -- stockage -----------------------------------------------------------------
    def path(self, entity_id: str) -> Path:
        return DATA_DIR / "signals" / self.id / f"{entity_id}.parquet"

    def load(self, entity_id: str) -> pd.DataFrame:
        p = self.path(entity_id)
        if not p.exists():
            return empty()
        return pd.read_parquet(p)

    def save(self, entity_id: str, df: pd.DataFrame) -> None:
        """Fusionne avec l'existant (le plus récent calcul gagne) puis écrit."""
        old = self.load(entity_id)
        df = pd.concat([old, df]) if len(old) else df
        df = df.drop_duplicates(["entity", "date", "signal"], keep="last")
        df = df.sort_values(["signal", "date"]).reset_index(drop=True)
        p = self.path(entity_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        df[COLUMNS].to_parquet(p, index=False)

    def update(self, entity_id: str, start: date, end: date) -> pd.DataFrame:
        df = self.run(entity_id, start, end)
        if len(df):
            self.save(entity_id, df)
        return self.load(entity_id)


def analysed(module_id: str = "optical") -> list[str]:
    """Entités pour lesquelles un module a déjà produit des signaux."""
    d = DATA_DIR / "signals" / module_id
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df[COLUMNS]


def months(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
