import logging
import time
from typing import Iterable, Optional
from urllib.parse import quote

import requests

from .config import CONFIG
from .db import now_utc

LOGGER = logging.getLogger(__name__)


def _get_headers() -> dict:
    headers = {"User-Agent": "arxiv-local-search/0.1 (mailto:local@example.com)"}
    if CONFIG.semantic_scholar_api_key:
        headers["x-api-key"] = CONFIG.semantic_scholar_api_key
    return headers


def fetch_citations_count(arxiv_id: str) -> Optional[int]:
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return None
    paper_id = f"ARXIV:{arxiv_id}"
    encoded = quote(paper_id, safe="")
    url = f"{CONFIG.citations_base_url.rstrip('/')}/paper/{encoded}"
    params = {"fields": "citationCount"}
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_get_headers(),
            timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        return int(payload.get("citationCount", 0))
    except Exception as exc:
        LOGGER.info("citations fetch failed for %s: %s", arxiv_id, exc)
        return None


def update_citations(cur, arxiv_ids: Iterable[str], sleep_seconds: float = 1.0) -> int:
    updated = 0
    for arxiv_id in arxiv_ids:
        count = fetch_citations_count(arxiv_id)
        if count is None:
            continue
        cur.execute(
            "UPDATE papers SET citations_count=?, citations_updated_at=? WHERE arxiv_id=?",
            (count, now_utc(), arxiv_id),
        )
        updated += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return updated
