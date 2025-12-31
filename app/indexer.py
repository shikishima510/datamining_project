import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

from .config import CONFIG
from .db import now_utc

LOGGER = logging.getLogger(__name__)


def upsert_papers_batch(cur, records: List[Dict[str, Any]]) -> Tuple[int, int]:
    inserted = 0
    updated = 0
    for rec in records:
        row = cur.execute(
            "SELECT id, updated_at FROM papers WHERE arxiv_id=?", (rec["arxiv_id"],)
        ).fetchone()
        payload = (
            rec["arxiv_id"],
            rec["title"],
            rec["abstract"],
            json.dumps(rec["authors"], ensure_ascii=True),
            rec["primary_category"],
            json.dumps(rec["categories"], ensure_ascii=True),
            json.dumps(rec["cross_categories"], ensure_ascii=True),
            rec["published_at"],
            rec["updated_at"],
            rec.get("doi"),
            rec.get("journal_ref"),
            rec.get("comment"),
            rec["abs_url"],
            rec["pdf_url"],
            rec["version_count"],
            rec["cross_list_count"],
            now_utc(),
        )
        if not row:
            cur.execute(
                """
                INSERT INTO papers(
                  arxiv_id, title, abstract, authors_json, primary_category,
                  categories_json, cross_categories_json, published_at, updated_at,
                  doi, journal_ref, comment, abs_url, pdf_url,
                  version_count, cross_list_count, ingested_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                payload,
            )
            inserted += 1
            paper_id = cur.lastrowid
        else:
            cur.execute(
                """
                UPDATE papers SET
                  title=?, abstract=?, authors_json=?, primary_category=?,
                  categories_json=?, cross_categories_json=?, published_at=?, updated_at=?,
                  doi=?, journal_ref=?, comment=?, abs_url=?, pdf_url=?,
                  version_count=?, cross_list_count=?, ingested_at=?
                WHERE arxiv_id=?
                """,
                (
                    rec["title"],
                    rec["abstract"],
                    json.dumps(rec["authors"], ensure_ascii=True),
                    rec["primary_category"],
                    json.dumps(rec["categories"], ensure_ascii=True),
                    json.dumps(rec["cross_categories"], ensure_ascii=True),
                    rec["published_at"],
                    rec["updated_at"],
                    rec.get("doi"),
                    rec.get("journal_ref"),
                    rec.get("comment"),
                    rec["abs_url"],
                    rec["pdf_url"],
                    rec["version_count"],
                    rec["cross_list_count"],
                    now_utc(),
                    rec["arxiv_id"],
                ),
            )
            updated += 1
            paper_id = row["id"]

        cur.execute(
            "INSERT OR REPLACE INTO papers_fts(rowid, title, abstract) VALUES(?,?,?)",
            (paper_id, rec["title"], rec["abstract"]),
        )
    return inserted, updated


def cleanup_old_papers(cur, days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    rows = cur.execute(
        "SELECT id FROM papers WHERE updated_at < ?", (cutoff_iso,)
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    cur.executemany(
        "INSERT INTO papers_fts(papers_fts, rowid, title, abstract) VALUES('delete', ?, '', '')",
        [(pid,) for pid in ids],
    )
    cur.execute("DELETE FROM papers WHERE updated_at < ?", (cutoff_iso,))
    return len(ids)


def rebuild_fts(cur) -> int:
    LOGGER.info("Rebuilding FTS index")
    cur.execute("DELETE FROM papers_fts")
    rows = cur.execute("SELECT id, title, abstract FROM papers").fetchall()
    for row in rows:
        cur.execute(
            "INSERT INTO papers_fts(rowid, title, abstract) VALUES(?,?,?)",
            (row["id"], row["title"], row["abstract"]),
        )
    return len(rows)
