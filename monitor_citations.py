import argparse
import logging
from datetime import datetime, timedelta, timezone

from app.db import db_cursor, init_db

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_iso = cutoff.isoformat()

    with db_cursor() as cur:
        total = cur.execute(
            "SELECT COUNT(*) AS c FROM papers WHERE updated_at >= ?",
            (cutoff_iso,),
        ).fetchone()[0]
        done = cur.execute(
            "SELECT COUNT(*) AS c FROM papers WHERE updated_at >= ? AND citations_count IS NOT NULL",
            (cutoff_iso,),
        ).fetchone()[0]
        pending = max(total - done, 0)
        last = cur.execute(
            "SELECT MAX(citations_updated_at) AS m FROM papers WHERE citations_updated_at IS NOT NULL",
        ).fetchone()[0]

    logging.info("citations progress: %s/%s done, pending=%s", done, total, pending)
    logging.info("last updated at: %s", last)


if __name__ == "__main__":
    main()
