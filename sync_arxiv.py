import argparse
import logging

from app.arxiv_oai import harvest_sets
from app.db import db_cursor, init_db, set_sync_state
from app.indexer import cleanup_old_papers

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--sets", type=str, default="")
    parser.add_argument("--from-dt", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
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
