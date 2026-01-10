import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set

import requests

from app.config import CONFIG
from app.db import db_cursor, init_db, now_utc
from app.graph import pagerank

try:
    from tqdm import trange as ranger
except ImportError:
    print("tqdm not installed, progress bar disabled. Install tqdm for better experience.")
    ranger = range

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _get_headers() -> dict:
    headers = {"User-Agent": "arxiv-local-search/0.1 (mailto:local@example.com)"}
    if CONFIG.semantic_scholar_api_key:
        headers["x-api-key"] = CONFIG.semantic_scholar_api_key
    return headers


def _extract_arxiv_id(external_ids: dict) -> str | None:
    if not external_ids:
        return None
    for key in ("ArXiv", "arXiv", "arxiv"):
        if key in external_ids and external_ids[key]:
            return str(external_ids[key])
    return None


def fetch_edges_batch(arxiv_ids: List[str], max_retry: int = 5) -> dict[str, dict[str, object]]:
    url = f"{CONFIG.citations_base_url.rstrip('/')}/paper/batch"
    params = {"fields": "references.externalIds,citations.externalIds,externalIds,citationCount"}
    payload = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]}
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(
                url,
                params=params,
                headers=_get_headers(),
                json=payload,
                timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
                LOGGER.info("429 rate limited batch, sleep %ss", delay)
                time.sleep(delay)
                if attempt <= max_retry:
                    continue
            resp.raise_for_status()
            items = resp.json() or []
            break
        except Exception as exc:
            if attempt >= max_retry:
                LOGGER.info("graph batch fetch failed: %s", exc)
                return {}
            time.sleep(min(2 ** attempt, 30))

    result: dict[str, dict[str, object]] = {}
    for item in items:
        if not item:
            continue
        ext = item.get("externalIds") or {}
        arxiv_id = _extract_arxiv_id(ext)
        if not arxiv_id:
            continue
        ref_ids = []
        for ref in item.get("references") or []:
            rid = _extract_arxiv_id(ref.get("externalIds") or {})
            if rid:
                ref_ids.append(rid)
        cit_ids = []
        for cit in item.get("citations") or []:
            cid = _extract_arxiv_id(cit.get("externalIds") or {})
            if cid:
                cit_ids.append(cid)
        result[arxiv_id] = {
            "refs": ref_ids,
            "cits": cit_ids,
            "citationCount": item.get("citationCount"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Semantic Scholar 获取论文引用及其数量，构建论文引用图，并计算 PageRank 分数。")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="仅处理 updated_at >= (今天 - days) 的已下载数据库中的论文。默认: 180",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="最多处理的论文数量。设为 0 或负数表示不限制。默认: 200",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="每次调用 Semantic Scholar 批量 API 的论文数量。默认: 200",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="每批 API 请求之间的等待秒数（避免触发限流）。默认: 0.5",
    )
    parser.add_argument(
        "--pagerank-iterations",
        type=int,
        default=20,
        help="PageRank 算法的迭代次数。次数越多收敛越精确，但耗时更长。默认: 20",
    )
    args = parser.parse_args()

    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_iso = cutoff.isoformat()

    with db_cursor(commit=True) as cur:
        rows = cur.execute(
            "SELECT arxiv_id FROM papers WHERE updated_at >= ? ORDER BY updated_at DESC",
            (cutoff_iso,),
        ).fetchall()
        all_ids = [row["arxiv_id"] for row in rows]
        if args.limit and args.limit > 0:
            all_ids = all_ids[: args.limit]
        id_set: Set[str] = set(all_ids)

        offset = 0
        batch_size = max(args.batch_size, 1)
        for offset in ranger(0, len(all_ids), batch_size):
            batch = all_ids[offset : offset + batch_size]
            batch_edges = fetch_edges_batch(batch)
            for arxiv_id in batch:
                payload = batch_edges.get(arxiv_id, {})
                refs = payload.get("refs", []) if isinstance(payload, dict) else []
                cits = payload.get("cits", []) if isinstance(payload, dict) else []
                citation_count = payload.get("citationCount") if isinstance(payload, dict) else None
                for ref in refs:
                    if ref in id_set:
                        cur.execute(
                            "INSERT OR IGNORE INTO paper_edges(from_arxiv_id, to_arxiv_id, source, ingested_at) "
                            "VALUES(?,?,?,?)",
                            (arxiv_id, ref, "reference", now_utc()),
                        )
                for cit in cits:
                    if cit in id_set:
                        cur.execute(
                            "INSERT OR IGNORE INTO paper_edges(from_arxiv_id, to_arxiv_id, source, ingested_at) "
                            "VALUES(?,?,?,?)",
                            (cit, arxiv_id, "citation", now_utc()),
                        )
                if citation_count is not None:
                    cur.execute(
                        "UPDATE papers SET citations_count=?, citations_updated_at=? WHERE arxiv_id=?",
                        (int(citation_count), now_utc(), arxiv_id),
                    )
            cur.connection.commit()
            if args.sleep > 0:
                time.sleep(args.sleep)

        edge_rows = cur.execute(
            "SELECT from_arxiv_id, to_arxiv_id FROM paper_edges"
        ).fetchall()
        edges: Dict[str, List[str]] = {}
        for row in edge_rows:
            edges.setdefault(row["from_arxiv_id"], []).append(row["to_arxiv_id"])

        scores = pagerank(id_set, edges, iterations=args.pagerank_iterations)
        for arxiv_id, score in scores.items():
            cur.execute(
                "UPDATE papers SET pagerank_score=?, pagerank_updated_at=? WHERE arxiv_id=?",
                (float(score), now_utc(), arxiv_id),
            )
        cur.connection.commit()

    logging.info("graph edges: %s, pagerank computed: %s", len(edge_rows), len(scores))


if __name__ == "__main__":
    main()
