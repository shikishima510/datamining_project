import logging

from app.db import db_cursor, init_db
from app.indexer import rebuild_fts

logging.basicConfig(level=logging.INFO)


def main() -> None:
    init_db()
    with db_cursor(commit=True) as cur:
        count = rebuild_fts(cur)
        cur.connection.commit()
    logging.info("rebuild complete: %s rows", count)


if __name__ == "__main__":
    main()
