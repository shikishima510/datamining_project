#!/usr/bin/env python3
"""
绘制 papers 表中 updated_at 列的时间分布直方图。

对数据库的日期格式要求：
    - ISO 8601 (无微秒): "2025-06-19T10:23:50+00:00"

用法：
    python plot_dates.py [--mode relative|absolute] [--bins N] [--days N] [--output FILE]

选项：
    --mode      显示模式：
                  - relative: X 轴显示 "距今 -N 天"，标题显示截止日期 (默认)
                  - absolute: X 轴显示绝对日期
    --bins      直方图柱数，默认 50
    --days      仅绘制最近 N 天内的论文，过滤长尾数据。默认 0 (不过滤)
    --output    输出文件路径，默认 plot_dates.png
"""

import argparse
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser as dtparser

from app.config import CONFIG


def load_dates(days: int = 0) -> List[datetime]:
    """
    从数据库加载所有 updated_at 日期。

    Args:
        days: 仅加载最近 N 天内的日期。0 表示不过滤。

    Returns:
        datetime 对象列表 (UTC 时区)
    """
    conn = sqlite3.connect(CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT updated_at FROM papers WHERE updated_at IS NOT NULL").fetchall()
    cur.close()
    conn.close()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days) if days > 0 else None

    dates = []
    for row in rows:
        try:
            dt = dtparser.parse(row["updated_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if cutoff is None or dt >= cutoff:
                dates.append(dt)
        except Exception:
            continue
    return dates


def plot_relative(dates: List[datetime], bins: int, output: str) -> None:
    """
    绘制相对日期直方图。

    X 轴: 距今天数 (负数，如 -200, -100, 0)
    标题: "论文更新时间分布 (截止 YYYY-MM-DD)"
    """
    now = datetime.now(timezone.utc)
    days_ago = [(dt - now).total_seconds() / 86400 for dt in dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(days_ago, bins=bins, edgecolor="white", alpha=0.8, color="#2d7ff9")

    ax.set_xlabel("Days ago", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.set_title(f"Paper update time distribution (as of {now.strftime('%Y-%m-%d')})", fontsize=14)

    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"图表已保存至: {output}")


def plot_absolute(dates: List[datetime], bins: int, output: str) -> None:
    """
    绘制绝对日期直方图。

    X 轴: 绝对日期 (如 2024-01, 2024-07, 2025-01)
    标题: "论文更新时间分布"
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(dates, bins=bins, edgecolor="white", alpha=0.8, color="#ff6b4a")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.set_title("Paper update time distribution", fontsize=14)

    # 格式化 X 轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"图表已保存至: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制 papers 表中 updated_at 列的时间分布直方图。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
        help="显示模式: relative (距今天数) 或 absolute (绝对日期)。默认: relative",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="直方图柱数。默认: 50",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="仅绘制最近 N 天内的论文，过滤长尾数据。默认: 0 (不过滤)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot_dates.png",
        help="输出文件路径。默认: plot_dates.png",
    )
    args = parser.parse_args()

    dates = load_dates(days=args.days)
    if not dates:
        print("没有找到有效的日期数据。")
        return

    print(f"加载了 {len(dates)} 条日期记录。")
    print(f"日期范围: {min(dates).strftime('%Y-%m-%d')} ~ {max(dates).strftime('%Y-%m-%d')}")

    if args.mode == "relative":
        plot_relative(dates, args.bins, args.output)
    else:
        plot_absolute(dates, args.bins, args.output)


if __name__ == "__main__":
    main()
