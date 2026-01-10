import argparse
import logging
from datetime import datetime, timedelta, timezone

from app.citations import update_citations
from app.db import db_cursor, init_db

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Semantic Scholar 获取论文引用数并更新本地数据库。")
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
        default=500,
        help="每次从数据库查询的论文数量。默认: 500",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每次 API 请求之间的等待秒数（避免触发限流）。默认: 1.0",
    )
    args = parser.parse_args()

    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_iso = cutoff.isoformat()

    with db_cursor(commit=True) as cur:
        limit = args.limit
        batch_size = max(args.batch_size, 1)
        offset = 0
        updated = 0
        while True:
            if limit and limit > 0:
                batch_limit = min(batch_size, max(limit - offset, 0))
                if batch_limit <= 0:
                    break
            else:
                batch_limit = batch_size
            rows = cur.execute(
                "SELECT arxiv_id FROM papers WHERE updated_at >= ? "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (cutoff_iso, batch_limit, offset),
            ).fetchall()
            if not rows:
                break
            arxiv_ids = [row["arxiv_id"] for row in rows]
            updated += update_citations(cur, arxiv_ids, sleep_seconds=args.sleep)
            cur.connection.commit()
            offset += len(rows)

    logging.info("citations updated: %s", updated)


if __name__ == "__main__":
    main()
