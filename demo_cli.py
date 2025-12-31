import logging

from app.db import db_cursor, init_db
from app.profile import update_profile, profile_summary, build_feed_query, diversify_results
from app.search import search_papers
from app.models import SearchFilters

logging.basicConfig(level=logging.INFO)


def main() -> None:
    init_db()
    user_id = "demo_user"
    with db_cursor(commit=True) as cur:
        results, debug = search_papers(
            cur,
            raw_query="transformer model",
            filters=SearchFilters(time_range_days=180),
            user_id=user_id,
            size=5,
            page=1,
        )
        print("Search results:")
        for r in results:
            print(f"- {r['arxiv_id']} {r['title']} score={r['score']:.4f}")

        if results:
            update_profile(cur, user_id, results[0]["arxiv_id"], "like")
        if len(results) > 1:
            update_profile(cur, user_id, results[1]["arxiv_id"], "dislike")

        profile = cur.execute(
            "SELECT profile_json FROM user_profile WHERE user_id=?", (user_id,)
        ).fetchone()
        profile_json = profile and profile[0]
        print("Profile summary:", profile_summary({} if not profile_json else __import__("json").loads(profile_json)))

        raw_query, filters = build_feed_query(
            {} if not profile_json else __import__("json").loads(profile_json),
            time_range_days=180,
        )
        feed_results, _ = search_papers(
            cur,
            raw_query=raw_query,
            filters=filters,
            user_id=user_id,
            size=10,
            page=1,
        )
        feed_results = diversify_results(feed_results, title_overlap=0.8)
        print("\nFeed results:")
        for r in feed_results[:5]:
            print(f"- {r['arxiv_id']} {r['title']} score={r['score']:.4f}")


if __name__ == "__main__":
    main()
