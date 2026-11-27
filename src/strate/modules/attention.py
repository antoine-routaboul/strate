"""Attention publique : vues mensuelles des articles Wikipedia d'un site et de ses propriétaires.

Source : API officielle Wikimedia Pageviews (données depuis juillet 2015, trafic humain seulement).
Un pic d'attention sur l'article d'une mine précède ou accompagne souvent une nouvelle
(accident, rachat, mise en production) : c'est un signal de « quelque chose se passe ».
"""

from __future__ import annotations

import time
from datetime import date
from urllib.parse import quote

import httpx
import pandas as pd

from strate import entities
from strate.core import Module, frame

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "strate/0.1 (open-source research tool)"}
SOURCE = "Wikimedia Pageviews API (en.wikipedia, utilisateurs humains)"


def _get(client: httpx.Client, url: str, **kw) -> httpx.Response:
    """GET poli : respecte les limites de débit de Wikimedia (429 → on attend et on réessaie)."""
    for attempt in range(6):
        r = client.get(url, **kw)
        if r.status_code != 429:
            return r
        time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
    r.raise_for_status()
    return r


def aliases(article: str, client: httpx.Client) -> list[str]:
    """Titre actuel + tous ses redirects : après un renommage, l'historique reste sous l'ancien titre."""
    r = _get(client, WIKI_API, params=dict(
        action="query", titles=article.replace("_", " "), redirects=1, prop="redirects",
        rdlimit="max", rdnamespace=0, format="json",
    ))
    r.raise_for_status()
    pages = r.json()["query"]["pages"].values()
    titles = []
    for p in pages:
        titles.append(p["title"])
        titles += [x["title"] for x in p.get("redirects", [])]
    return [t.replace(" ", "_") for t in titles] or [article]


def pageviews(article: str, start: date, end: date, client: httpx.Client) -> dict[date, int]:
    """Vues mensuelles de l'article, redirects compris."""
    out: dict[date, int] = {}
    for title in aliases(article, client):
        url = f"{API}/{quote(title, safe='')}/monthly/{start:%Y%m}01/{end:%Y%m%d}"
        r = _get(client, url)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for it in r.json()["items"]:
            ts = it["timestamp"]
            m = date(int(ts[:4]), int(ts[4:6]), 1)
            out[m] = out.get(m, 0) + int(it["views"])
    return out


class Attention(Module):
    id = "attention"
    title = "Attention publique (Wikipedia)"
    description = "Vues mensuelles de l'article de la mine et de ceux de ses propriétaires."
    entity_kind = "site"
    signals = {
        "site_views": "Vues mensuelles de l'article Wikipedia de la mine",
        "owner_views": "Vues mensuelles des articles des propriétaires, pondérées par leur part",
    }

    def run(self, entity_id: str, start: date, end: date) -> pd.DataFrame:
        g = entities.load()
        site = g.site(entity_id)
        rows = []
        transport = httpx.HTTPTransport(retries=4)
        with httpx.Client(headers=HEADERS, timeout=60, transport=transport) as c:
            if site.wikipedia:
                for m, v in pageviews(site.wikipedia, start, end, c).items():
                    rows.append(dict(entity=site.id, date=m, signal="site_views", value=v,
                                     unit="vues", quality=1.0, source=SOURCE))
            owners: dict[date, float] = {}
            covered = 0.0
            for comp, share in g.owners_of(site.id):
                if not comp.wikipedia:
                    continue
                covered += share
                for m, v in pageviews(comp.wikipedia, start, end, c).items():
                    owners[m] = owners.get(m, 0.0) + share * v
            for m, v in owners.items():
                rows.append(dict(entity=site.id, date=m, signal="owner_views", value=round(v),
                                 unit="vues", quality=round(covered, 2), source=SOURCE))
        return frame(rows)
