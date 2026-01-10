#!/usr/bin/env python3
"""
迁移脚本：将 papers 表中的 published_at 和 updated_at 从 RFC 2822 格式转换为 ISO 8601 格式。

日期格式说明：
    - RFC 2822 (迁移前): "Thu, 19 Jun 2025 10:23:50 GMT"
        arXiv OAI-PMH 接口原始返回格式，不支持字符串字典序比较。
    - ISO 8601 无微秒 (迁移后): "2025-06-19T10:23:50+00:00"
        本项目数据库存储的标准格式，支持字符串字典序比较。

用法：
    python migrate_dates.py [--dry-run]

选项：
    --dry-run   仅预览将被修改的记录数量，不实际执行更新。
"""

import argparse
import logging
import sqlite3
from email.utils import parsedate_to_datetime
try:
    from tqdm import tqdm
except ImportError:
    print("未安装 tqdm 库，进度条功能将被禁用。可通过 'pip install tqdm' 安装。")
    tqdm = lambda x: x  # Fallback: no progress bar

from app.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def normalize_rfc2822_to_iso(raw: str) -> str | None:
    """
    将 RFC 2822 格式的日期字符串转换为 ISO 8601 格式（不含微秒）。

    Args:
        raw: RFC 2822 格式的日期字符串，例如 "Thu, 19 Jun 2025 10:23:50 GMT"

    Returns:
        ISO 8601 格式的日期字符串，例如 "2025-06-19T10:23:50+00:00"
        若已经是 ISO 8601 格式或解析失败，返回 None 表示无需转换。
    """
    if not raw:
        return None
    # ISO 8601 格式以数字开头 (如 "2025-06-19T...")
    # RFC 2822 格式以星期几开头 (如 "Thu, 19 Jun...")
    if raw[0].isdigit():
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.replace(microsecond=0).isoformat()
    except Exception:
        return None


def migrate(dry_run: bool = False) -> None:
    conn = sqlite3.connect(CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 统计需要迁移的记录
    rows = cur.execute(
        "SELECT id, arxiv_id, published_at, updated_at FROM papers"
    ).fetchall()

    to_update = []
    for row in tqdm(rows):
        new_published = normalize_rfc2822_to_iso(row["published_at"])
        new_updated = normalize_rfc2822_to_iso(row["updated_at"])
        if new_published is not None or new_updated is not None:
            to_update.append({
                "id": row["id"],
                "arxiv_id": row["arxiv_id"],
                "published_at": new_published if new_published else row["published_at"],
                "updated_at": new_updated if new_updated else row["updated_at"],
            })

    LOGGER.info("总记录数: %d", len(rows))
    LOGGER.info("需要迁移: %d", len(to_update))

    if dry_run:
        LOGGER.info("[DRY-RUN] 不执行实际更新。")
        if to_update:
            LOGGER.info("示例转换 (前 3 条):")
            for rec in to_update[:3]:
                LOGGER.info("  %s: published_at=%s, updated_at=%s",
                            rec["arxiv_id"], rec["published_at"], rec["updated_at"])
        cur.close()
        conn.close()
        return

    # 执行更新
    updated_count = 0
    for rec in to_update:
        cur.execute(
            "UPDATE papers SET published_at=?, updated_at=? WHERE id=?",
            (rec["published_at"], rec["updated_at"], rec["id"]),
        )
        updated_count += 1

    conn.commit()
    cur.close()
    conn.close()

    LOGGER.info("迁移完成，更新了 %d 条记录。", updated_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 papers 表中的日期从 RFC 2822 格式迁移到 ISO 8601 格式。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将被修改的记录数量，不实际执行更新。",
    )
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
