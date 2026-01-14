"""
arXiv 本地检索系统评估脚本
===========================
评估内容:
1. 消融实验 (Ablation Study): 各排序因子的贡献度
2. 个性化有效性 (Personalization): 反馈前后排序变化
3. PageRank vs Citation: 发现"隐藏宝石"的能力
4. 检索质量 (Retrieval Quality): 语义相似度

运行方式:
    python eval_metrics.py --all
    python eval_metrics.py --ablation
    python eval_metrics.py --personalization
    python eval_metrics.py --pagerank-vs-citations
"""

import argparse
import json
import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db import db_cursor, init_db
from app.search import search_papers
from app.profile import update_profile, profile_summary, extract_terms
from app.models import SearchFilters
from app.config import CONFIG


# ============================================================================
# Test Queries (覆盖不同研究领域)
# ============================================================================
TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "graph neural network",
        "category": "cs.LG",
        "description": "图神经网络 - 机器学习热门方向",
    },
    {
        "id": "Q2",
        "query": "large language model reasoning",
        "category": "cs.CL",
        "description": "大语言模型推理 - NLP前沿",
    },
    {
        "id": "Q3",
        "query": "autonomous driving perception",
        "category": "cs.CV",
        "description": "自动驾驶感知 - 计算机视觉应用",
    },
    {
        "id": "Q4",
        "query": "federated learning privacy",
        "category": "cs.CR",
        "description": "联邦学习隐私 - 安全与机器学习交叉",
    },
    {
        "id": "Q5",
        "query": "transformer architecture efficient",
        "category": "cs.LG",
        "description": "高效Transformer - 模型优化",
    },
]


# ============================================================================
# Evaluation Metrics
# ============================================================================
def calculate_semantic_similarity(query: str, results: List[Dict[str, Any]]) -> float:
    """计算查询与结果标题+摘要的TF-IDF余弦相似度"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("Warning: sklearn not installed, skipping semantic similarity")
        return 0.0

    if not results:
        return 0.0

    documents = [query]
    for r in results:
        text = f"{r['title']} {r.get('abstract', '')[:500]}"
        documents.append(text)

    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000).fit_transform(documents)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        return float(np.mean(similarities))
    except Exception:
        return 0.0


def calculate_score_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算各评分因子的平均贡献"""
    if not results:
        return {}

    factors = ["bm25", "recency", "pref", "citations", "pagerank", "title_boost"]
    sums = {f: 0.0 for f in factors}

    for r in results:
        scores = r.get("explain", {}).get("scores", {})
        for f in factors:
            sums[f] += scores.get(f, 0.0)

    n = len(results)
    return {f: sums[f] / n for f in factors}


def calculate_rank_correlation(list_a: List[str], list_b: List[str]) -> float:
    """计算两个排序列表的Kendall tau相关系数 (简化版: 基于位置差)"""
    if not list_a or not list_b:
        return 0.0

    common = set(list_a) & set(list_b)
    if len(common) < 2:
        return 0.0

    rank_a = {item: i for i, item in enumerate(list_a) if item in common}
    rank_b = {item: i for i, item in enumerate(list_b) if item in common}

    # Spearman-like: sum of squared rank differences
    diff_sum = sum((rank_a[item] - rank_b[item]) ** 2 for item in common)
    n = len(common)
    # Normalize to [-1, 1] range (1 = identical, 0 = no correlation)
    max_diff = n * (n ** 2 - 1) / 3
    if max_diff == 0:
        return 1.0
    correlation = 1 - (6 * diff_sum) / max_diff
    return max(min(correlation, 1.0), -1.0)


# ============================================================================
# Experiment 1: Ablation Study (消融实验)
# ============================================================================
def run_ablation_study(output_file: Optional[str] = None) -> str:
    """
    消融实验: 逐个禁用排序因子, 观察对结果的影响
    """
    print("\n" + "=" * 60)
    print("实验一: 消融实验 (Ablation Study)")
    print("=" * 60)

    # 保存原始配置
    original_weights = {
        "bm25": CONFIG.score_weight_bm25,
        "recency": CONFIG.score_weight_recency,
        "pref": CONFIG.score_weight_pref,
        "citations": CONFIG.score_weight_citations,
        "pagerank": CONFIG.score_weight_pagerank,
    }

    ablation_configs = [
        ("Full Model", {}),
        ("No BM25", {"score_weight_bm25": 0.0}),
        ("No Recency", {"score_weight_recency": 0.0}),
        ("No Citations", {"score_weight_citations": 0.0}),
        ("No PageRank", {"score_weight_pagerank": 0.0}),
        ("No Personalization", {"score_weight_pref": 0.0}),
        ("BM25 Only", {
            "score_weight_recency": 0.0,
            "score_weight_pref": 0.0,
            "score_weight_citations": 0.0,
            "score_weight_pagerank": 0.0,
        }),
    ]

    results_table = []
    query = TEST_QUERIES[0]  # 使用第一个查询作为基准

    init_db()

    for config_name, overrides in ablation_configs:
        # 应用配置覆盖
        for key, val in overrides.items():
            setattr(CONFIG, key, val)

        with db_cursor() as cur:
            results, debug = search_papers(
                cur,
                raw_query=query["query"],
                filters=SearchFilters(time_range_days=180),
                user_id=None,  # 无用户偏好
                size=20,
                page=1,
                disable_pref=True,
            )

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        sem_sim = calculate_semantic_similarity(query["query"], results)
        top_ids = [r["arxiv_id"] for r in results[:10]]

        results_table.append({
            "config": config_name,
            "avg_score": avg_score,
            "semantic_sim": sem_sim,
            "top_ids": top_ids,
            "top_titles": [r["title"][:60] for r in results[:3]],
        })

        # 恢复原始配置
        for key, val in original_weights.items():
            setattr(CONFIG, f"score_weight_{key}", val)

    # 计算与Full Model的排序相关性
    full_model_ids = results_table[0]["top_ids"]
    for row in results_table:
        row["rank_corr_vs_full"] = calculate_rank_correlation(full_model_ids, row["top_ids"])

    # 生成报告
    report = []
    report.append(f"\n### 查询: \"{query['query']}\" ({query['description']})\n")
    report.append("| 配置 | 平均得分 | 语义相似度 | 与Full Model排序相关性 |")
    report.append("|------|----------|------------|------------------------|")

    for row in results_table:
        report.append(
            f"| {row['config']} | {row['avg_score']:.4f} | {row['semantic_sim']:.4f} | {row['rank_corr_vs_full']:.4f} |"
        )

    report.append("\n**Top 3 Results (Full Model):**")
    for i, title in enumerate(results_table[0]["top_titles"], 1):
        report.append(f"{i}. {title}...")

    report.append("\n**Top 3 Results (BM25 Only):**")
    for i, title in enumerate(results_table[-1]["top_titles"], 1):
        report.append(f"{i}. {title}...")

    output = "\n".join(report)
    print(output)
    return output


# ============================================================================
# Experiment 2: Personalization Effectiveness (个性化有效性)
# ============================================================================
def run_personalization_experiment(output_file: Optional[str] = None) -> str:
    """
    个性化实验: 模拟用户反馈, 观察排序变化
    """
    print("\n" + "=" * 60)
    print("实验二: 个性化有效性 (Personalization Effectiveness)")
    print("=" * 60)

    test_user_id = "eval_test_user"
    query = "neural network"

    init_db()

    # Step 1: 清空测试用户数据
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_profile WHERE user_id=?", (test_user_id,))
        cur.execute("DELETE FROM user_events WHERE user_id=?", (test_user_id,))

    # Step 2: 初始搜索 (无偏好)
    with db_cursor() as cur:
        results_before, _ = search_papers(
            cur,
            raw_query=query,
            filters=SearchFilters(time_range_days=180),
            user_id=test_user_id,
            size=20,
            page=1,
        )
    ranking_before = [(r["arxiv_id"], r["score"], r["primary_category"]) for r in results_before]

    # Step 3: 模拟用户反馈 - 喜欢 "cs.CL" (NLP) 相关论文
    with db_cursor(commit=True) as cur:
        # 找到一些 cs.CL 论文并标记为 like
        cl_papers = [r for r in results_before if "cs.CL" in r.get("categories", [])]
        if not cl_papers:
            # 如果结果中没有, 直接查询数据库
            rows = cur.execute(
                "SELECT arxiv_id FROM papers WHERE primary_category='cs.CL' LIMIT 3"
            ).fetchall()
            cl_paper_ids = [row["arxiv_id"] for row in rows]
        else:
            cl_paper_ids = [p["arxiv_id"] for p in cl_papers[:3]]

        for arxiv_id in cl_paper_ids:
            update_profile(cur, test_user_id, arxiv_id, "like")

        # 同时 dislike 一些 cs.CV 论文
        cv_papers = [r for r in results_before if r.get("primary_category") == "cs.CV"]
        for p in cv_papers[:2]:
            update_profile(cur, test_user_id, p["arxiv_id"], "dislike")

    # Step 4: 再次搜索 (有偏好)
    with db_cursor() as cur:
        results_after, _ = search_papers(
            cur,
            raw_query=query,
            filters=SearchFilters(time_range_days=180),
            user_id=test_user_id,
            size=20,
            page=1,
        )

        # 获取用户画像
        profile_row = cur.execute(
            "SELECT profile_json FROM user_profile WHERE user_id=?", (test_user_id,)
        ).fetchone()
        profile = json.loads(profile_row["profile_json"]) if profile_row else {}

    ranking_after = [(r["arxiv_id"], r["score"], r["primary_category"]) for r in results_after]

    # Step 5: 分析排序变化
    report = []
    report.append(f"\n### 查询: \"{query}\"\n")
    report.append(f"**模拟反馈:** Like 3篇 cs.CL 论文, Dislike 2篇 cs.CV 论文\n")

    # 用户画像摘要
    summary = profile_summary(profile)
    report.append("**更新后的用户画像:**")
    report.append(f"- Top Terms: {summary.get('top_terms', [])[:5]}")
    report.append(f"- Top Categories: {summary.get('top_categories', [])[:3]}")
    report.append(f"- Negative Terms Count: {summary.get('negative_terms_count', 0)}\n")

    # 类别分布变化
    def category_distribution(results: List[Tuple]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for _, _, cat in results[:10]:
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    dist_before = category_distribution(ranking_before)
    dist_after = category_distribution(ranking_after)

    report.append("**Top-10 结果类别分布变化:**")
    report.append("| 类别 | 反馈前 | 反馈后 | 变化 |")
    report.append("|------|--------|--------|------|")

    all_cats = set(dist_before.keys()) | set(dist_after.keys())
    for cat in sorted(all_cats):
        before = dist_before.get(cat, 0)
        after = dist_after.get(cat, 0)
        change = after - before
        sign = "+" if change > 0 else ""
        report.append(f"| {cat} | {before} | {after} | {sign}{change} |")

    # 具体排名变化示例
    report.append("\n**排名变化示例 (cs.CL 论文):**")
    report.append("| arxiv_id | 反馈前排名 | 反馈后排名 | 排名提升 |")
    report.append("|----------|------------|------------|----------|")

    id_to_rank_before = {r[0]: i + 1 for i, r in enumerate(ranking_before)}
    id_to_rank_after = {r[0]: i + 1 for i, r in enumerate(ranking_after)}

    # 找出提升最多的论文
    rank_changes = []
    for arxiv_id in id_to_rank_after:
        if arxiv_id in id_to_rank_before:
            change = id_to_rank_before[arxiv_id] - id_to_rank_after[arxiv_id]
            rank_changes.append((arxiv_id, id_to_rank_before[arxiv_id], id_to_rank_after[arxiv_id], change))

    rank_changes.sort(key=lambda x: x[3], reverse=True)
    for arxiv_id, rb, ra, change in rank_changes[:5]:
        sign = "+" if change > 0 else ""
        report.append(f"| {arxiv_id} | #{rb} | #{ra} | {sign}{change} |")

    # 清理测试用户
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM user_profile WHERE user_id=?", (test_user_id,))
        cur.execute("DELETE FROM user_events WHERE user_id=?", (test_user_id,))

    output = "\n".join(report)
    print(output)
    return output


# ============================================================================
# Experiment 3: PageRank vs Citations (发现隐藏宝石)
# ============================================================================
def run_pagerank_vs_citations(output_file: Optional[str] = None) -> str:
    """
    对比PageRank和引用数, 找出被低估/高估的论文
    """
    print("\n" + "=" * 60)
    print("实验三: PageRank vs Citations (发现隐藏宝石)")
    print("=" * 60)

    init_db()

    report = []

    with db_cursor() as cur:
        # 查询有PageRank和引用数的论文
        rows = cur.execute("""
            SELECT arxiv_id, title, primary_category, citations_count, pagerank_score, updated_at
            FROM papers
            WHERE citations_count IS NOT NULL
              AND citations_count > 0
              AND pagerank_score IS NOT NULL
              AND pagerank_score > 0
            ORDER BY updated_at DESC
            LIMIT 10000
        """).fetchall()

    if not rows:
        return "No papers with both citations and pagerank found."

    # 计算归一化分数
    citations = [r["citations_count"] for r in rows]
    pageranks = [r["pagerank_score"] for r in rows]

    max_cit = max(citations) if citations else 1
    max_pr = max(pageranks) if pageranks else 1

    papers_data = []
    for r in rows:
        norm_cit = r["citations_count"] / max_cit
        norm_pr = r["pagerank_score"] / max_pr
        # 差异度: PageRank高但引用少 = 隐藏宝石
        hidden_gem_score = norm_pr - norm_cit
        papers_data.append({
            "arxiv_id": r["arxiv_id"],
            "title": r["title"][:80],
            "category": r["primary_category"],
            "citations": r["citations_count"],
            "pagerank": r["pagerank_score"],
            "norm_cit": norm_cit,
            "norm_pr": norm_pr,
            "hidden_gem_score": hidden_gem_score,
        })

    # 找出 "隐藏宝石" (PageRank高, 引用相对低)
    papers_data.sort(key=lambda x: x["hidden_gem_score"], reverse=True)
    hidden_gems = papers_data[:10]

    # 找出 "过誉论文" (引用高, PageRank相对低)
    papers_data.sort(key=lambda x: x["hidden_gem_score"])
    overrated = papers_data[:10]

    report.append("\n### 隐藏宝石 (PageRank高, 引用数相对低)")
    report.append("*这些论文在引用网络中处于关键位置, 但绝对引用数不高*\n")
    report.append("| arxiv_id | 引用数 | PageRank | 类别 | 标题 |")
    report.append("|----------|--------|----------|------|------|")
    for p in hidden_gems[:7]:
        report.append(f"| {p['arxiv_id']} | {p['citations']} | {p['pagerank']:.6f} | {p['category']} | {p['title'][:50]}... |")

    report.append("\n### 高引用但PageRank较低的论文")
    report.append("*这些论文引用数高, 但在引用网络中相对孤立*\n")
    report.append("| arxiv_id | 引用数 | PageRank | 类别 | 标题 |")
    report.append("|----------|--------|----------|------|------|")
    for p in overrated[:7]:
        report.append(f"| {p['arxiv_id']} | {p['citations']} | {p['pagerank']:.6f} | {p['category']} | {p['title'][:50]}... |")

    # 统计数据
    import statistics
    cit_list = [p["citations"] for p in papers_data]
    pr_list = [p["pagerank"] for p in papers_data]

    report.append("\n### 统计摘要")
    report.append(f"- 分析论文数: {len(papers_data)}")
    report.append(f"- 引用数: 中位数={statistics.median(cit_list):.0f}, 均值={statistics.mean(cit_list):.1f}, 最大={max(cit_list)}")
    report.append(f"- PageRank: 中位数={statistics.median(pr_list):.6f}, 均值={statistics.mean(pr_list):.6f}, 最大={max(pr_list):.6f}")

    # Spearman相关性
    try:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(cit_list, pr_list)
        report.append(f"- Spearman相关系数: {corr:.4f} (p={pval:.2e})")
        report.append(f"  *解读: 相关但不完全一致, 说明PageRank提供了引用数之外的信息*")
    except ImportError:
        pass

    output = "\n".join(report)
    print(output)
    return output


# ============================================================================
# Experiment 4: Multi-Query Retrieval Quality (多查询检索质量)
# ============================================================================
def run_retrieval_quality(output_file: Optional[str] = None) -> str:
    """
    对多个查询计算检索质量指标
    """
    print("\n" + "=" * 60)
    print("实验四: 多查询检索质量评估")
    print("=" * 60)

    init_db()

    report = []
    report.append("\n### 检索质量评估\n")
    report.append("| Query ID | 查询 | 语义相似度 | Avg BM25 | Avg Recency | Avg PageRank | Top Category |")
    report.append("|----------|------|------------|----------|-------------|--------------|--------------|")

    all_results = []

    for q in TEST_QUERIES:
        with db_cursor() as cur:
            results, debug = search_papers(
                cur,
                raw_query=q["query"],
                filters=SearchFilters(time_range_days=180),
                user_id=None,
                size=20,
                page=1,
            )

        sem_sim = calculate_semantic_similarity(q["query"], results)
        score_dist = calculate_score_distribution(results)

        # Top category in results
        cat_counts: Dict[str, int] = {}
        for r in results[:10]:
            cat = r.get("primary_category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "N/A"

        report.append(
            f"| {q['id']} | {q['query'][:25]}... | {sem_sim:.4f} | "
            f"{score_dist.get('bm25', 0):.4f} | {score_dist.get('recency', 0):.4f} | "
            f"{score_dist.get('pagerank', 0):.4f} | {top_cat} |"
        )

        all_results.append({
            "query_id": q["id"],
            "query": q["query"],
            "semantic_sim": sem_sim,
            "score_dist": score_dist,
            "top_results": [{"id": r["arxiv_id"], "title": r["title"][:60]} for r in results[:3]],
        })

    # 平均值
    avg_sem = sum(r["semantic_sim"] for r in all_results) / len(all_results)
    report.append(f"| **AVG** | - | **{avg_sem:.4f}** | - | - | - | - |")

    # 展示每个查询的Top结果
    report.append("\n### 各查询Top-3结果\n")
    for ar in all_results:
        report.append(f"**{ar['query_id']}: {ar['query']}**")
        for i, r in enumerate(ar["top_results"], 1):
            report.append(f"  {i}. [{r['id']}] {r['title']}...")
        report.append("")

    output = "\n".join(report)
    print(output)
    return output


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv 检索系统评估脚本")
    parser.add_argument("--all", action="store_true", help="运行所有实验")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--personalization", action="store_true", help="运行个性化实验")
    parser.add_argument("--pagerank-vs-citations", action="store_true", help="运行PageRank对比实验")
    parser.add_argument("--retrieval", action="store_true", help="运行检索质量实验")
    parser.add_argument("--output", "-o", type=str, default="eval_report.md", help="输出报告文件")

    args = parser.parse_args()

    if not any([args.all, args.ablation, args.personalization, args.pagerank_vs_citations, args.retrieval]):
        args.all = True

    reports = []
    reports.append("# arXiv 本地检索系统评估报告")
    reports.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.all or args.ablation:
        reports.append("\n## 实验一: 消融实验")
        reports.append(run_ablation_study())

    if args.all or args.personalization:
        reports.append("\n## 实验二: 个性化有效性")
        reports.append(run_personalization_experiment())

    if args.all or args.pagerank_vs_citations:
        reports.append("\n## 实验三: PageRank vs Citations")
        reports.append(run_pagerank_vs_citations())

    if args.all or args.retrieval:
        reports.append("\n## 实验四: 检索质量评估")
        reports.append(run_retrieval_quality())

    # 写入文件
    full_report = "\n".join(reports)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n{'=' * 60}")
    print(f"报告已保存至: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
