import argparse
import logging

from app.arxiv_oai import harvest_sets
from app.db import db_cursor, init_db, set_sync_state
from app.indexer import cleanup_old_papers

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 arXiv OAI-PMH 接口同步论文元数据到本地数据库。")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="时间窗口天数。抓取 updated_at >= (今天 - days - 10) 的论文；清理时删除 updated_at < (今天 - days) 的记录。默认: 180",
    )
    parser.add_argument(
        "--sets",
        type=str,
        default="",
        help="指定要抓取的 arXiv set（逗号分隔，如 'cs,eess'）。留空则自动选择 cs 和 eess。",
    )
    parser.add_argument(
        "--from-dt",
        type=str,
        default="",
        help="强制指定抓取起始日期（ISO 格式，如 '2024-01-01'）。设置后覆盖 --days 计算的起始时间。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传模式。从上次中断的 resumptionToken 继续抓取，而非重新开始。",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="跳过清理步骤，保留所有旧论文（默认会删除 updated_at < (今天 - days) 的记录）。",
    )
    args = parser.parse_args()

    init_db()
    sets_override = [s.strip() for s in args.sets.split(",") if s.strip()] or None

    with db_cursor(commit=True) as cur:
        if not args.resume:
            set_sync_state(cur, "active_setSpec", "")
            set_sync_state(cur, "active_from_dt", "")
            set_sync_state(cur, "active_resumption_token", "")
        if args.from_dt:
            set_sync_state(cur, "active_from_dt", args.from_dt)

        result = harvest_sets(cur, args.days, backfill=not args.resume, sets_override=sets_override)
        deleted = 0
        if not args.no_cleanup:
            deleted = cleanup_old_papers(cur, args.days)
        cur.connection.commit()

    logging.info("sync result: %s deleted=%s", result, deleted)


if __name__ == "__main__":
    main()
