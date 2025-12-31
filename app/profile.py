import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .db import now_utc, load_profile, save_profile
from .config import CONFIG
from .models import SearchFilters

LOGGER = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "we", "our", "their", "from", "as", "at", "it", "its", "into", "via", "using",
    "based", "new", "novel", "study", "paper", "approach", "results", "model", "models",
    "analysis", "method", "methods",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def extract_terms(title: str, abstract: str, top_k: int = 20) -> List[str]:
    text = f"{title} {abstract}".lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    top = [t for t, _ in counts.most_common(top_k)]
    return top


def _init_profile() -> Dict[str, Any]:
    return {
        "term_weights": {},
        "category_weights": {},
        "negative_terms": [],
        "negative_categories": [],
        "category_neg_counts": {},
    }


def _update_weights(profile: Dict[str, Any], terms: List[str], category: str, delta: float, cat_delta: float) -> None:
    term_weights = profile.setdefault("term_weights", {})
    cat_weights = profile.setdefault("category_weights", {})
    for t in terms:
        term_weights[t] = float(term_weights.get(t, 0.0)) + delta
    if category:
        cat_weights[category] = float(cat_weights.get(category, 0.0)) + cat_delta


def _action_deltas(action: str) -> tuple[float, float]:
    if action == "like":
        return 0.5, 0.5
    if action == "save":
        return 0.8, 0.8
    if action in {"dislike", "hide"}:
        return -0.5, -0.5
    if action == "view":
        return 0.1, 0.1
    return 0.0, 0.0


def _apply_action(profile: Dict[str, Any], terms: List[str], category: str, action: str, direction: int = 1) -> None:
    term_delta, cat_delta = _action_deltas(action)
    term_delta *= direction
    cat_delta *= direction
    _update_weights(profile, terms, category, delta=term_delta, cat_delta=cat_delta)
    if action in {"dislike", "hide"}:
        neg_terms = set(profile.get("negative_terms", []))
        if direction > 0:
            for t in terms[:5]:
                neg_terms.add(t)
        else:
            for t in terms[:5]:
                neg_terms.discard(t)
        profile["negative_terms"] = sorted(neg_terms)


def update_profile_from_text(cur, user_id: str, title: str, abstract: str, action: str) -> Dict[str, Any]:
    profile = load_profile(cur, user_id) or _init_profile()
    terms = extract_terms(title, abstract, top_k=20)
    if action == "like":
        _update_weights(profile, terms, "", delta=0.5, cat_delta=0.0)
    elif action == "save":
        _update_weights(profile, terms, "", delta=0.8, cat_delta=0.0)
    elif action in {"dislike", "hide"}:
        _update_weights(profile, terms, "", delta=-0.5, cat_delta=0.0)
        neg_terms = set(profile.get("negative_terms", []))
        for t in terms[:5]:
            neg_terms.add(t)
        profile["negative_terms"] = sorted(neg_terms)
    elif action == "view":
        _update_weights(profile, terms, "", delta=0.1, cat_delta=0.0)

    version = int(profile.get("version", 0)) + 1
    profile["version"] = version
    save_profile(cur, user_id, profile, version)
    return profile


def update_profile_from_terms(cur, user_id: str, terms: List[str], action: str) -> Dict[str, Any]:
    profile = load_profile(cur, user_id) or _init_profile()
    terms = [t for t in terms if t]
    if action == "like":
        _update_weights(profile, terms, "", delta=0.5, cat_delta=0.0)
    elif action == "save":
        _update_weights(profile, terms, "", delta=0.8, cat_delta=0.0)
    elif action in {"dislike", "hide"}:
        _update_weights(profile, terms, "", delta=-0.5, cat_delta=0.0)
        neg_terms = set(profile.get("negative_terms", []))
        for t in terms[:5]:
            neg_terms.add(t)
        profile["negative_terms"] = sorted(neg_terms)
    elif action == "view":
        _update_weights(profile, terms, "", delta=0.1, cat_delta=0.0)

    version = int(profile.get("version", 0)) + 1
    profile["version"] = version
    save_profile(cur, user_id, profile, version)
    return profile


def update_profile(cur, user_id: str, arxiv_id: str, action: str) -> Dict[str, Any]:
    profile = load_profile(cur, user_id) or _init_profile()
    row = cur.execute(
        "SELECT title, abstract, primary_category FROM papers WHERE arxiv_id=?",
        (arxiv_id,),
    ).fetchone()
    if not row:
        return profile
    terms = extract_terms(row["title"], row["abstract"], top_k=15)
    category = row["primary_category"]

    _apply_action(profile, terms, category, action, direction=1)

    version = int(profile.get("version", 0)) + 1
    profile["version"] = version
    save_profile(cur, user_id, profile, version)
    return profile


def update_profile_with_prev(cur, user_id: str, arxiv_id: str, action: str, prev_action: str | None) -> Dict[str, Any]:
    profile = load_profile(cur, user_id) or _init_profile()
    row = cur.execute(
        "SELECT title, abstract, primary_category FROM papers WHERE arxiv_id=?",
        (arxiv_id,),
    ).fetchone()
    if not row:
        return profile
    terms = extract_terms(row["title"], row["abstract"], top_k=15)
    category = row["primary_category"]
    if prev_action:
        _apply_action(profile, terms, category, prev_action, direction=-1)
    _apply_action(profile, terms, category, action, direction=1)

    version = int(profile.get("version", 0)) + 1
    profile["version"] = version
    save_profile(cur, user_id, profile, version)
    return profile


def profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    term_weights = profile.get("term_weights", {})
    cat_weights = profile.get("category_weights", {})
    top_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    top_cats = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "top_terms": top_terms,
        "top_categories": top_cats,
        "negative_terms_count": len(profile.get("negative_terms", [])),
        "negative_categories_count": len(profile.get("negative_categories", [])),
    }


def diversify_results(results: List[Dict[str, Any]], title_overlap: float) -> List[Dict[str, Any]]:
    selected = []
    seen_tokens = []
    for item in results:
        title_tokens = set(TOKEN_RE.findall(item["title"].lower()))
        if not title_tokens:
            selected.append(item)
            continue
        duplicate = False
        for tokens in seen_tokens:
            overlap = len(title_tokens & tokens) / max(len(title_tokens | tokens), 1)
            if overlap >= title_overlap:
                duplicate = True
                break
        if duplicate:
            continue
        seen_tokens.append(title_tokens)
        selected.append(item)
    return selected


def build_feed_query(profile: Dict[str, Any], time_range_days: int) -> Tuple[str, SearchFilters]:
    term_weights = profile.get("term_weights", {})
    cat_weights = profile.get("category_weights", {})
    top_terms = [t for t, _ in sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:8]]
    top_cats = [c for c, _ in sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)[:3]]
    raw_query = " ".join(top_terms)
    filters = SearchFilters(
        time_range_days=time_range_days,
        categories_include=top_cats,
        categories_exclude=[],
    )
    return raw_query, filters
