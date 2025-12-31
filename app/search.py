import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dtparser
import requests

from .config import CONFIG
from .models import StructuredQuery, SearchFilters
from .profile import extract_terms

LOGGER = logging.getLogger(__name__)

_ALNUM_BOUNDARY = re.compile(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize_query(raw_query: str) -> str:
    text = raw_query.strip()
    if not text:
        return text
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"([A-Za-z])([0-9])", r"\\1 \\2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\\1 \\2", text)
    text = " ".join(text.split())
    return text.lower()


def _expand_query_terms(raw_query: str) -> List[str]:
    normalized = _normalize_query(raw_query)
    if not normalized:
        return []
    tokens = _TOKEN_RE.findall(normalized)
    variants = {raw_query.strip(), raw_query.strip().lower(), normalized}
    variants.update(tokens)
    if tokens and len(tokens) <= 3:
        joined = "".join(tokens)
        if 2 <= len(joined) <= 12:
            variants.add(joined)
            variants.add("-".join(tokens))
    return [v for v in variants if v]


def _merge_expanded_terms(parsed: StructuredQuery) -> StructuredQuery:
    expand_terms = parsed.expand_terms or []
    normalized_terms = parsed.normalized_terms or []
    if expand_terms or normalized_terms:
        merged = []
        merged.extend(parsed.should_terms or [])
        merged.extend(expand_terms)
        merged.extend(normalized_terms)
        parsed.should_terms = list(dict.fromkeys([t for t in merged if t]))
    return parsed


def _tokenize_query(raw_query: str) -> List[str]:
    normalized = _normalize_query(raw_query)
    if not normalized:
        return []
    tokens = _TOKEN_RE.findall(normalized)
    return [t for t in tokens if not (t.isdigit() and len(t) < 2)]


def _basic_query_terms(raw_query: str) -> List[str]:
    text = raw_query.strip()
    if not text:
        return []
    has_digit = any(ch.isdigit() for ch in text)
    if has_digit:
        tokens = _TOKEN_RE.findall(_normalize_query(text))
        joined = "".join(tokens)
        terms = []
        if joined:
            terms.append(joined)
        terms.append(text)
        return list(dict.fromkeys([t for t in terms if t]))
    tokens = _tokenize_query(text)
    return tokens if tokens else [text]


def _escape_term(term: str) -> str:
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def compile_match(structured: StructuredQuery) -> Optional[str]:
    must_terms = [_escape_term(t) for t in structured.must_terms if t]
    phrase_terms = [_escape_term(t) for t in structured.phrase_terms if t]
    must_not_terms = [_escape_term(t) for t in structured.must_not_terms if t]
    should_terms = [_escape_term(t) for t in structured.should_terms if t]

    must = must_terms + phrase_terms
    parts = []
    if must:
        parts.append(" AND ".join(must))
    if should_terms:
        should_expr = " OR ".join(should_terms)
        if must:
            parts.append(f"({should_expr})")
        else:
            parts.append(should_expr)
    if not parts:
        return None
    expr = " AND ".join(parts)
    if must_not_terms:
        expr = f"{expr} AND NOT (" + " OR ".join(must_not_terms) + ")"
    return expr


def _category_filter_sql(include: List[str], exclude: List[str]) -> Tuple[str, List[Any]]:
    clauses = []
    params: List[Any] = []
    for cat in include:
        clauses.append("categories_json LIKE ?")
        params.append(f'%"{cat}"%')
    for cat in exclude:
        clauses.append("categories_json NOT LIKE ?")
        params.append(f'%"{cat}"%')
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _recency_score(updated_at: str) -> float:
    try:
        dt = dtparser.parse(updated_at)
    except Exception:
        return 0.0
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    return math.exp(-days / max(CONFIG.recency_tau_days, 1.0))


def _bm25_to_score(bm25_value: float) -> float:
    return 1.0 / (1.0 + max(bm25_value, 0.0))


def _load_profile(cur, user_id: Optional[str]) -> Dict[str, Any]:
    if not user_id:
        return {}
    row = cur.execute(
        "SELECT profile_json FROM user_profile WHERE user_id=?", (user_id,)
    ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def _seen_penalty(cur, user_id: Optional[str], arxiv_id: str) -> float:
    if not user_id:
        return 0.0
    row = cur.execute(
        "SELECT 1 FROM user_events WHERE user_id=? AND arxiv_id=? ORDER BY ts DESC LIMIT 1",
        (user_id, arxiv_id),
    ).fetchone()
    return 1.0 if row else 0.0


def _preference_score(profile: Dict[str, Any], title: str, abstract: str, category: str) -> float:
    if not profile:
        return 0.0
    term_weights = profile.get("term_weights", {})
    cat_weights = profile.get("category_weights", {})
    neg_terms = set(profile.get("negative_terms", []))
    neg_cats = set()

    terms = extract_terms(title, abstract, top_k=15)
    score = 0.0
    for t in terms:
        if t in neg_terms:
            score -= 0.5
        score += float(term_weights.get(t, 0.0))
    if category in neg_cats:
        score -= 0.5
    score += float(cat_weights.get(category, 0.0))
    return score


class NoOpParser:
    def parse(self, raw_query: str, filters: Optional[SearchFilters]) -> StructuredQuery:
        structured = StructuredQuery()
        if raw_query:
            if '"' in raw_query:
                structured.should_terms = [raw_query]
            else:
                structured.should_terms = _basic_query_terms(raw_query)
        if filters:
            structured.time_range_days = filters.time_range_days
            structured.categories_include = filters.categories_include
            structured.categories_exclude = filters.categories_exclude
        return structured


class LLMParser:
    def __init__(self) -> None:
        self.base_url = CONFIG.llm_base_url.rstrip("/")
        self.model = CONFIG.llm_model
        self.api_key = CONFIG.llm_api_key
        self.azure_endpoint = CONFIG.azure_openai_endpoint.rstrip("/")
        self.azure_api_key = CONFIG.azure_openai_api_key
        self.azure_deployment = CONFIG.azure_openai_deployment
        self.azure_api_version = CONFIG.azure_openai_api_version

    def parse(self, raw_query: str, filters: Optional[SearchFilters]) -> StructuredQuery:
        if self.azure_endpoint and self.azure_api_key and self.azure_deployment:
            return self._parse_azure(raw_query, filters)
        if not self.api_key:
            return NoOpParser().parse(raw_query, filters)
        schema = StructuredQuery.schema()
        prompt = (
            "You are a search query parser. Output strict JSON that matches this schema. "
            "If missing values, use empty list. Schema: "
            f"{json.dumps(schema, ensure_ascii=True)}"
        )
        user_prompt = (
            "Decide whether robust normalization is needed (e.g., SAM3 vs SAM 3 vs SAM-3). "
            "If needed, populate expand_terms or normalized_terms. "
            f"Query: {raw_query}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        try:
            url = self.base_url
            if not url.endswith("/v1/chat"):
                url = f"{self.base_url}/v1/chat"
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = StructuredQuery.parse_raw(content)
        except Exception as exc:
            LOGGER.info("LLM parse failed: %s", exc)
            return NoOpParser().parse(raw_query, filters)

        parsed = _merge_expanded_terms(parsed)
        if filters:
            parsed.time_range_days = filters.time_range_days
            parsed.categories_include = filters.categories_include
            parsed.categories_exclude = filters.categories_exclude
        return parsed

    def _parse_azure(self, raw_query: str, filters: Optional[SearchFilters]) -> StructuredQuery:
        schema = StructuredQuery.schema()
        prompt = (
            "You are a search query parser. Output strict JSON that matches this schema. "
            "If missing values, use empty list. Schema: "
            f"{json.dumps(schema, ensure_ascii=True)}"
        )
        user_prompt = (
            "Decide whether robust normalization is needed (e.g., SAM3 vs SAM 3 vs SAM-3). "
            "If needed, populate expand_terms or normalized_terms. "
            f"Query: {raw_query}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions"
            f"?api-version={self.azure_api_version}"
        )
        try:
            resp = requests.post(
                url,
                headers={"api-key": self.azure_api_key},
                json=payload,
                timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = StructuredQuery.parse_raw(content)
        except Exception as exc:
            LOGGER.info("Azure LLM parse failed: %s", exc)
            return NoOpParser().parse(raw_query, filters)

        parsed = _merge_expanded_terms(parsed)
        if filters:
            parsed.time_range_days = filters.time_range_days
            parsed.categories_include = filters.categories_include
            parsed.categories_exclude = filters.categories_exclude
        return parsed


def get_parser() -> Any:
    if CONFIG.llm_enabled:
        return LLMParser()
    return NoOpParser()


def search_papers(
    cur,
    raw_query: str,
    filters: Optional[SearchFilters],
    user_id: Optional[str],
    size: int,
    page: int = 1,
    disable_pref: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start_ts = datetime.now(timezone.utc)
    parser = get_parser()
    structured = parser.parse(raw_query, filters)
    match_expr = compile_match(structured)
    query_tokens = _tokenize_query(raw_query)

    cutoff = datetime.now(timezone.utc) - timedelta(days=structured.time_range_days)
    cutoff_iso = cutoff.isoformat()

    cat_sql, cat_params = _category_filter_sql(
        structured.categories_include, structured.categories_exclude
    )
    params: List[Any] = []
    window = max(size * 5 * page, size)
    if match_expr:
        sql = (
            "SELECT p.*, bm25(papers_fts, ?, ?) as bm25_score "
            "FROM papers_fts JOIN papers p ON p.id = papers_fts.rowid "
            "WHERE papers_fts MATCH ? AND p.updated_at >= ?"
        )
        params.extend([CONFIG.fts_title_weight, CONFIG.fts_abstract_weight, match_expr, cutoff_iso])
    else:
        sql = "SELECT p.*, 0.0 as bm25_score FROM papers p WHERE p.updated_at >= ?"
        params.append(cutoff_iso)
    if cat_sql:
        sql += cat_sql
        params.extend(cat_params)
    sql += " ORDER BY p.updated_at DESC LIMIT ?"
    params.append(window)

    rows = cur.execute(sql, params).fetchall()
    profile = _load_profile(cur, user_id)

    results = []
    for row in rows:
        if match_expr:
            bm25_val = float(row["bm25_score"])
            bm25_score = _bm25_to_score(bm25_val)
        else:
            bm25_score = 0.0
        recency = _recency_score(row["updated_at"])
        pref = 0.0 if disable_pref else _preference_score(
            profile, row["title"], row["abstract"], row["primary_category"]
        )
        title_tokens = set(_TOKEN_RE.findall(row["title"].lower()))
        title_match = bool(query_tokens) and set(query_tokens).issubset(title_tokens)
        title_boost = CONFIG.title_match_boost if title_match else 0.0
        citations = row["citations_count"] or 0
        citations_score = 0.0
        if citations > 0:
            citations_score = min(math.log1p(citations), CONFIG.citations_log_cap) / max(
                CONFIG.citations_log_cap, 1.0
            )
        pagerank = row["pagerank_score"] or 0.0
        neg_terms_hit = False
        if structured.must_not_terms:
            paper_terms = set(extract_terms(row["title"], row["abstract"], top_k=30))
            neg_terms_hit = bool(paper_terms & set(structured.must_not_terms))
        seen = _seen_penalty(cur, user_id, row["arxiv_id"])
        total = (
            CONFIG.score_weight_bm25 * bm25_score
            + CONFIG.score_weight_recency * recency
            + CONFIG.score_weight_pref * pref
            + CONFIG.score_weight_citations * citations_score
            + CONFIG.score_weight_pagerank * pagerank
            + title_boost
            - CONFIG.score_weight_seen_penalty * seen
        )
        explain = {
            "match_expr": match_expr,
            "matched_terms": structured.must_terms + structured.should_terms + structured.phrase_terms,
            "neg_terms": structured.must_not_terms,
            "neg_terms_hit": neg_terms_hit,
            "scores": {
                "bm25": bm25_score,
                "recency": recency,
                "pref": pref,
                "citations": citations_score,
                "pagerank": pagerank,
                "title_boost": title_boost,
                "seen_penalty": seen,
                "total": total,
            },
            "filters": {
                "categories_include": structured.categories_include,
                "categories_exclude": structured.categories_exclude,
                "time_range_days": structured.time_range_days,
            },
        }
        results.append(
            {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "authors": json.loads(row["authors_json"]),
                "primary_category": row["primary_category"],
                "categories": json.loads(row["categories_json"]),
                "published_at": row["published_at"],
                "updated_at": row["updated_at"],
                "citations_count": row["citations_count"],
                "pagerank_score": row["pagerank_score"],
                "abs_url": row["abs_url"],
                "pdf_url": row["pdf_url"],
                "score": total,
                "explain": explain,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    offset = (page - 1) * size
    debug = {
        "structured_query": structured.dict(),
        "sql": sql,
        "timings": {
            "elapsed_ms": (datetime.now(timezone.utc) - start_ts).total_seconds() * 1000,
        },
        "page": page,
        "size": size,
        "window": window,
    }
    return results[offset : offset + size], debug
