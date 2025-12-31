from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class StructuredQuery(BaseModel):
    must_terms: List[str] = Field(default_factory=list)
    should_terms: List[str] = Field(default_factory=list)
    must_not_terms: List[str] = Field(default_factory=list)
    categories_include: List[str] = Field(default_factory=list)
    categories_exclude: List[str] = Field(default_factory=list)
    time_range_days: int = 180
    phrase_terms: List[str] = Field(default_factory=list)
    expand_terms: List[str] = Field(default_factory=list)
    normalized_terms: List[str] = Field(default_factory=list)

    @validator("time_range_days")
    def _valid_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("time_range_days must be positive")
        return v


class SearchFilters(BaseModel):
    time_range_days: int = 180
    categories_include: List[str] = Field(default_factory=list)
    categories_exclude: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    user_id: Optional[str]
    raw_query: str
    filters: Optional[SearchFilters]
    size: int = 20
    page: int = 1

    @validator("page")
    def _valid_page(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("page must be positive")
        return v


class SearchResult(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    primary_category: str
    categories: List[str]
    published_at: str
    updated_at: str
    citations_count: Optional[int] = None
    pagerank_score: Optional[float] = None
    abs_url: str
    pdf_url: str
    score: float
    explain: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    debug: Dict[str, Any]


class SyncRequest(BaseModel):
    user_id: Optional[str]
    days: int = 180
    backfill: bool = False
    cleanup: bool = True
    rebuild_fts: bool = False


class SyncResponse(BaseModel):
    inserted: int
    updated: int
    deleted: int
    elapsed_seconds: float
    checkpoint: Dict[str, Any]


class FeedRequest(BaseModel):
    user_id: str
    time_range_days: int = 180
    size: int = 20
    page: int = 1

    @validator("page")
    def _valid_feed_page(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("page must be positive")
        return v


class FeedResponse(BaseModel):
    results: List[SearchResult]


class EventRequest(BaseModel):
    user_id: str
    arxiv_id: str
    action: str
    context: Optional[Dict[str, Any]]

    @validator("action")
    def _valid_action(cls, v: str) -> str:
        if v not in {"view", "like", "dislike", "save", "hide"}:
            raise ValueError("invalid action")
        return v


class EventResponse(BaseModel):
    ok: bool
    profile_summary: Dict[str, Any]


class ScanSavedRequest(BaseModel):
    user_id: str


class ScanSavedResponse(BaseModel):
    ok: bool
    processed: int
    skipped: int
    profile_summary: Dict[str, Any]


class ProfileResponse(BaseModel):
    user_id: str
    profile: Dict[str, Any]
    updated_at: Optional[str]
    version: Optional[int]


class HealthResponse(BaseModel):
    ok: bool
    status: str
