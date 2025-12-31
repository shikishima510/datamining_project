import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    db_path: str = os.environ.get("APP_DB_PATH", "data/app.db")
    saved_dir: str = os.environ.get("SAVED_PDF_DIR", "saved")
    saved_scan_max_pages: int = int(os.environ.get("SAVED_SCAN_MAX_PAGES", "5"))
    llm_enabled: bool = _get_bool("LLM_ENABLED", False)
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1/chat")
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_pdf_keywords_enabled: bool = _get_bool("LLM_PDF_KEYWORDS_ENABLED", True)
    llm_pdf_max_chars: int = int(os.environ.get("LLM_PDF_MAX_CHARS", "12000"))
    llm_pdf_max_keywords: int = int(os.environ.get("LLM_PDF_MAX_KEYWORDS", "20"))
    azure_openai_endpoint: str = os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com"
    )
    azure_openai_api_key: str = os.environ.get(
        "AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY"
    )
    azure_openai_deployment: str = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT", "YOUR_AZURE_DEPLOYMENT"
    )
    azure_openai_api_version: str = os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
    )
    request_timeout_connect: int = int(os.environ.get("HTTP_CONNECT_TIMEOUT", "10"))
    request_timeout_read: int = int(os.environ.get("HTTP_READ_TIMEOUT", "60"))
    harvest_sleep_seconds: float = float(os.environ.get("HARVEST_SLEEP", "3"))
    harvest_sleep_jitter: float = float(os.environ.get("HARVEST_SLEEP_JITTER", "0.5"))
    max_retry: int = int(os.environ.get("HARVEST_MAX_RETRY", "10"))
    fts_title_weight: float = float(os.environ.get("FTS_TITLE_WEIGHT", "3.0"))
    fts_abstract_weight: float = float(os.environ.get("FTS_ABSTRACT_WEIGHT", "1.0"))
    recency_tau_days: float = float(os.environ.get("RECENCY_TAU_DAYS", "30"))
    score_weight_bm25: float = float(os.environ.get("SCORE_WEIGHT_BM25", "0.6"))
    score_weight_recency: float = float(os.environ.get("SCORE_WEIGHT_RECENCY", "0.2"))
    score_weight_pref: float = float(os.environ.get("SCORE_WEIGHT_PREF", "0.015"))
    score_weight_seen_penalty: float = float(os.environ.get("SCORE_WEIGHT_SEEN_PENALTY", "0.0"))
    title_match_boost: float = float(os.environ.get("TITLE_MATCH_BOOST", "0.15"))
    score_weight_citations: float = float(os.environ.get("SCORE_WEIGHT_CITATIONS", "0.25"))
    citations_log_cap: float = float(os.environ.get("CITATIONS_LOG_CAP", "8.0"))
    score_weight_pagerank: float = float(os.environ.get("SCORE_WEIGHT_PAGERANK", "0.35"))
    negative_category_threshold: int = int(os.environ.get("NEGATIVE_CATEGORY_THRESHOLD", "3"))
    citations_base_url: str = os.environ.get(
        "CITATIONS_BASE_URL", "https://api.semanticscholar.org/graph/v1"
    )
    semantic_scholar_api_key: str = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    enable_tfidf_mmr: bool = _get_bool("ENABLE_TFIDF_MMR", False)
    diversify_title_overlap: float = float(os.environ.get("DIVERSIFY_TITLE_OVERLAP", "0.8"))


CONFIG = Config()
