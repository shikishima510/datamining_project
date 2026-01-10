import json
import logging
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

from .db import db_cursor, init_db
from .config import CONFIG
from .models import (
    EventRequest,
    EventResponse,
    FeedRequest,
    FeedResponse,
    HealthResponse,
    ProfileResponse,
    ScanSavedRequest,
    ScanSavedResponse,
    SearchRequest,
    SearchResponse,
    SyncRequest,
    SyncResponse,
)
from .arxiv_oai import harvest_sets
from .indexer import cleanup_old_papers, rebuild_fts
from .profile import (
    build_feed_query,
    profile_summary,
    update_profile,
    update_profile_from_text,
    update_profile_from_terms,
    update_profile_with_prev,
    diversify_results,
)
from .llm import extract_keywords
from .search import search_papers

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="arXiv Local Search")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, status="ok")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/static/{filename}")
def static_file(filename: str) -> Response:
    from fastapi.responses import FileResponse
    path = os.path.join("app/static", filename)
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=404)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>arXiv 本地检索与推荐</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/wordcloud@1.2.2/src/wordcloud2.min.js"></script>
  <style>
    :root {
      --bg: #f4efe8;
      --ink: #1c1a17;
      --muted: #6b6258;
      --accent: #ff6b4a;
      --accent-2: #2d7ff9;
      --card: #fffaf3;
      --line: #e6dccf;
      --shadow: 0 10px 30px rgba(28, 26, 23, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "PingFang SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #ffe6d8 0, transparent 40%),
                  radial-gradient(circle at 80% 10%, #d9e7ff 0, transparent 35%),
                  var(--bg);
    }
    header {
      padding: 32px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(255,107,74,0.08), rgba(45,127,249,0.08));
    }
    h1 {
      margin: 0 0 6px 0;
      font-size: 28px;
      letter-spacing: 0.5px;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
    }
    main {
      padding: 24px;
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
      gap: 20px;
    }
    .panel, .result-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: var(--shadow);
      animation: fadeIn 0.6s ease;
    }
    .panel h2 {
      margin: 0 0 12px 0;
      font-size: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fff;
      font-family: inherit;
    }
    textarea { min-height: 72px; }
    .row { display: grid; gap: 10px; margin-bottom: 12px; }
    .row.two { grid-template-columns: 1fr 1fr; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 600;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), #ff9266);
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      box-shadow: 0 8px 16px rgba(255, 107, 74, 0.25);
    }
    .btn.secondary {
      background: linear-gradient(135deg, var(--accent-2), #6aa6ff);
      box-shadow: 0 8px 16px rgba(45, 127, 249, 0.25);
    }
    .btn:active { transform: translateY(1px); }
    .results {
      display: grid;
      gap: 14px;
    }
    .result-card h3 {
      margin: 0 0 6px 0;
      font-size: 18px;
    }
    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
    }
    .action-icon {
      width: 24px;
      height: 24px;
      cursor: pointer;
      opacity: 0.6;
      transition: opacity 0.2s, transform 0.2s;
    }
    .action-icon:hover {
      opacity: 1;
      transform: scale(1.1);
    }
    .action-icon:active {
      transform: scale(0.95);
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .tag {
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #f1e7da;
    }
    .paper-link {
        text-decoration: none;
        color: inherit;
        cursor: pointer;
    }
    .paper-link:hover {
        color: var(--accent);
    }
    .mono {
      font-family: "IBM Plex Mono", "Menlo", monospace;
      font-size: 11px;
      color: var(--muted);
      white-space: pre-wrap;
      background: #faf6f0;
      padding: 8px;
      border-radius: 8px;
      border: 1px dashed var(--line);
    }
    .cat-bar-container {
      margin-bottom: 6px;
    }
    .cat-bar-label {
      font-size: 12px;
      color: var(--ink);
      margin-bottom: 2px;
    }
    .cat-bar-bg {
      background: var(--line);
      border-radius: 4px;
      height: 14px;
      overflow: hidden;
    }
    .cat-bar-fill {
      height: 100%;
      border-radius: 4px;
      background: linear-gradient(90deg, var(--accent-2), #6aa6ff);
      transition: width 0.3s ease;
    }
    .neg-tag {
      display: inline-block;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #ffdddd;
      color: #aa3333;
      margin: 2px;
    }
    .profile-section-title {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>arXiv 本地检索与个性化推荐</h1>
    <div class="subtitle">同步、检索、反馈闭环都在本机完成。默认不启用 LLM。</div>
  </header>
  <main>
    <section class="panel">
      <h2>同步数据</h2>
      <div class="row two">
        <div>
          <label>天数</label>
          <input id="syncDays" type="number" value="180" />
        </div>
        <div>
          <label>回填</label>
          <select id="syncBackfill">
            <option value="false">否</option>
            <option value="true">是</option>
          </select>
        </div>
      </div>
      <div class="row two">
        <div>
          <label>清理过期</label>
          <select id="syncCleanup">
            <option value="true">是</option>
            <option value="false">否</option>
          </select>
        </div>
        <div>
          <label>重建 FTS</label>
          <select id="syncRebuild">
            <option value="false">否</option>
            <option value="true">是</option>
          </select>
        </div>
      </div>
      <button class="btn" onclick="doSync()">开始同步</button>
      <div id="syncOutput" class="mono" style="margin-top:10px;"></div>
    </section>

    <section class="panel">
      <h2>搜索与推荐</h2>
      <div class="row">
        <label>用户 ID</label>
        <input id="userId" type="text" value="demo_user" />
      </div>
      <div class="row">
        <label>搜索（自然语言或关键词）</label>
        <input id="rawQuery" type="text" placeholder="例如: graph neural networks in bioinformatics" />
      </div>
      <div class="row two">
        <div>
          <label>分类 include（逗号分隔）</label>
          <input id="catInclude" type="text" placeholder="cs.LG,cs.AI" />
        </div>
        <div>
          <label>分类 exclude</label>
          <input id="catExclude" type="text" placeholder="cs.SE" />
        </div>
      </div>
      <div class="row two">
        <div>
          <label>时间范围天数</label>
          <input id="timeRange" type="number" value="180" />
        </div>
        <div>
          <label>返回条数</label>
          <input id="resultSize" type="number" value="10" />
        </div>
      </div>
      <div class="row two">
        <div>
          <label>页码</label>
          <input id="pageNum" type="number" value="1" min="1" />
        </div>
        <div style="display:flex; gap:8px; align-items:end;">
          <button class="btn secondary" onclick="changePage(-1)">上一页</button>
          <button class="btn" onclick="changePage(1)">下一页</button>
        </div>
      </div>
      <div class="row two">
        <button class="btn secondary" onclick="doSearch()">搜索</button>
        <button class="btn" onclick="doFeed()">猜你想看</button>
      </div>
      <div class="row">
        <button class="btn" onclick="doScanSaved()">同步该用户已保存 PDF</button>
      </div>
    </section>

    <section class="panel">
      <h2>用户反馈</h2>
      <div class="row two">
        <div>
          <label>arXiv ID</label>
          <input id="eventArxivId" type="text" placeholder="2401.00001" />
        </div>
        <div>
          <label>动作</label>
          <select id="eventAction">
            <option value="view">view</option>
            <option value="like">like</option>
            <option value="dislike">dislike</option>
            <option value="save">save</option>
            <option value="hide">hide</option>
          </select>
        </div>
      </div>
      <button class="btn" onclick="doEvent()">提交反馈</button>
      <details>
        <summary style="cursor:pointer; color: var(--muted); font-size: 13px;">反馈日志</summary>
        <div id="eventOutput" class="mono" style="margin-top:10px;"></div>
      </details>
    </section>

    <section class="panel">
      <h2>当前用户画像</h2>
      <div class="row" style="margin-top:10px;">
        <button class="btn secondary" onclick="loadProfile(true)">刷新画像</button>
      </div>
      <details open>
        <summary style="cursor:pointer; color: var(--muted); font-size: 13px;">可视化</summary>
        <div id="profileVisual">
          <div id="wordcloudContainer" style="width:100%; height:200px; margin:10px 0;"></div>
          <div id="categoryBars" style="margin:10px 0;"></div>
          <div id="negativeTags" style="margin:10px 0;"></div>
        </div>
      </details>
      <details>
        <summary style="cursor:pointer; color: var(--muted); font-size: 13px;">原始数据</summary>
        <div id="profileOutput" class="mono" style="margin-top:10px;"></div>
      </details>
    </section>

    <section>
      <div class="results" id="results"></div>
    </section>
  </main>
  <script>
    const api = (path, options) => fetch(path, options).then(r => r.json());
    const byId = (id) => document.getElementById(id);

    function splitCats(val) {
      return val.split(",").map(s => s.trim()).filter(Boolean);
    }

    async function doSync() {
      byId("syncOutput").textContent = "同步中...";
      const payload = {
        days: Number(byId("syncDays").value || 180),
        backfill: byId("syncBackfill").value === "true",
        cleanup: byId("syncCleanup").value === "true",
        rebuild_fts: byId("syncRebuild").value === "true"
      };
      const res = await api("/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      byId("syncOutput").textContent = JSON.stringify(res, null, 2);
    }

    let lastMode = "search";

    async function doSearch(resetPage = true) {
      lastMode = "search";
      if (resetPage) {
        byId("pageNum").value = 1;
      }
      const payload = {
        user_id: byId("userId").value,
        raw_query: byId("rawQuery").value,
        filters: {
          time_range_days: Number(byId("timeRange").value || 180),
          categories_include: splitCats(byId("catInclude").value),
          categories_exclude: splitCats(byId("catExclude").value)
        },
        size: Number(byId("resultSize").value || 10),
        page: Number(byId("pageNum").value || 1)
      };
      const res = await api("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      renderResults(res.results || []);
    }

    async function doFeed(resetPage = true) {
      lastMode = "feed";
      if (resetPage) {
        byId("pageNum").value = 1;
      }
      const payload = {
        user_id: byId("userId").value,
        time_range_days: Number(byId("timeRange").value || 180),
        size: Number(byId("resultSize").value || 10),
        page: Number(byId("pageNum").value || 1)
      };
      const res = await api("/feed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      renderResults(res.results || []);
    }

    function changePage(delta) {
      const input = byId("pageNum");
      let page = Number(input.value || 1) + delta;
      if (page < 1) page = 1;
      input.value = page;
      if (lastMode === "feed") {
        doFeed(false);
      } else {
        doSearch(false);
      }
    }

    async function doEvent() {
      byId("eventOutput").textContent = "提交中...";
      const payload = {
        user_id: byId("userId").value,
        arxiv_id: byId("eventArxivId").value,
        action: byId("eventAction").value
      };
      const res = await api("/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      byId("eventOutput").textContent = JSON.stringify(res, null, 2);
    }

    async function doScanSaved() {
      byId("eventOutput").textContent = "同步中...";
      const payload = { user_id: byId("userId").value };
      const res = await api("/scan_saved", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      byId("eventOutput").textContent = JSON.stringify(res, null, 2);
      loadProfile();
    }

    function renderResults(items) {
      const container = byId("results");
      container.innerHTML = "";
      if (!items.length) {
        container.innerHTML = "<div class='panel'>暂无结果</div>";
        return;
      }
      items.forEach(item => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.id = `card-${item.arxiv_id}`;
        const tags = (item.categories || []).slice(0, 6).map(c => `<span class='tag'>${c}</span>`).join("");
        const citations = item.citations_count != null ? ` · 引用 ${item.citations_count}` : "";
        const pagerank = item.pagerank_score != null ? ` · PR ${item.pagerank_score.toFixed(4)}` : "";
        const absUrl = item.abs_url || `https://arxiv.org/abs/${item.arxiv_id}`;
        
        card.innerHTML = `
          <h3><a href="${absUrl}" target="_blank" class="paper-link" onclick="submitAction('${item.arxiv_id}', 'view', false)">${item.title}</a></h3>
          <div class="meta">${item.arxiv_id} · ${item.primary_category} · ${item.updated_at}${citations}${pagerank}</div>
          <div>${item.abstract || ""}</div>
          <div class="tags">${tags}</div>
          <div class="mono">score: ${item.score.toFixed(4)}\\nexplain: ${JSON.stringify(item.explain.scores)}</div>
          <div class="actions">
             <img src="/static/thumb up.png" title="Like" class="action-icon" onclick="submitAction('${item.arxiv_id}', 'like')">
             <img src="/static/save.png" title="Save" class="action-icon" onclick="submitAction('${item.arxiv_id}', 'save')">
             <img src="/static/thumb down.png" title="Dislike" class="action-icon" onclick="submitAction('${item.arxiv_id}', 'dislike')">
             <img src="/static/hide.webp" title="Hide" class="action-icon" onclick="submitAction('${item.arxiv_id}', 'hide')">
          </div>
        `;
        container.appendChild(card);
      });
    }

    async function submitAction(arxivId, action, refreshProfile=true) {
      if (action === 'hide') {
         const card = document.getElementById(`card-${arxivId}`);
         if (card) {
             card.style.display = 'none';
         }
      }

      const payload = { user_id: byId("userId").value, arxiv_id: arxivId, action };
      const res = await api("/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      byId("eventOutput").textContent = JSON.stringify(res, null, 2);
      if (refreshProfile) {
          loadProfile();
      }
    }

    let cachedProfileHash = null;

    function hashProfile(profile) {
      // Create a hash based on actual content, not version
      const tw = profile.term_weights || {};
      const cw = profile.category_weights || {};
      const nt = profile.negative_terms || [];
      const nc = profile.negative_categories || [];
      // Simple string hash of sorted keys and rounded values
      const termStr = Object.entries(tw).sort((a,b) => a[0].localeCompare(b[0]))
        .map(([k,v]) => `${k}:${v.toFixed(4)}`).join(",");
      const catStr = Object.entries(cw).sort((a,b) => a[0].localeCompare(b[0]))
        .map(([k,v]) => `${k}:${v.toFixed(4)}`).join(",");
      const negStr = [...nt, ...nc].sort().join(",");
      return `${termStr}|${catStr}|${negStr}`;
    }

    async function loadProfile(forceRefresh = false) {
      const userId = byId("userId").value;
      if (!userId) return;
      const res = await api(`/profile/${userId}`);
      const profile = res.profile || {};
      
      // Update raw JSON always
      byId("profileOutput").textContent = JSON.stringify(profile, null, 2);
      
      // Skip visual refresh if content unchanged (unless forced)
      const newHash = hashProfile(profile);
      if (!forceRefresh && cachedProfileHash !== null && newHash === cachedProfileHash) {
        console.log("Profile unchanged, skipping visual refresh");
        return;
      }
      console.log("Profile changed, refreshing visuals. Old hash:", cachedProfileHash, "New hash:", newHash);
      cachedProfileHash = newHash;
      
      const termWeights = profile.term_weights || {};
      const catWeights = profile.category_weights || {};
      const negTerms = profile.negative_terms || [];
      const negCats = profile.negative_categories || [];
      
      // Word Cloud for term_weights
      const wcContainer = byId("wordcloudContainer");
      wcContainer.innerHTML = "";
      const canvas = document.createElement("canvas");
      canvas.width = wcContainer.offsetWidth || 300;
      canvas.height = 200;
      wcContainer.appendChild(canvas);
      
      const terms = Object.entries(termWeights);
      if (terms.length > 0) {
        const maxWeight = Math.max(...terms.map(([_,w]) => Math.abs(w)), 1);
        const wordList = terms
          .filter(([_,w]) => w > 0)
          .sort((a,b) => b[1] - a[1])
          .slice(0, 50)
          .map(([term, weight]) => [term, Math.max(8, (weight / maxWeight) * 48)]);
        
        if (wordList.length > 0 && typeof WordCloud !== 'undefined') {
          WordCloud(canvas, {
            list: wordList,
            gridSize: 8,
            weightFactor: 1,
            fontFamily: "IBM Plex Sans, sans-serif",
            color: function(word, weight) {
              const hue = 200 + Math.random() * 40;
              return `hsl(${hue}, 70%, 45%)`;
            },
            rotateRatio: 0.3,
            backgroundColor: "transparent"
          });
        }
      } else {
        wcContainer.innerHTML = "<div style='color:var(--muted);font-size:12px;text-align:center;padding:20px;'>暂无兴趣词汇</div>";
      }
      
      // Category Bars
      const catContainer = byId("categoryBars");
      const catEntries = Object.entries(catWeights).sort((a,b) => b[1] - a[1]).slice(0, 5);
      if (catEntries.length > 0) {
        const maxCat = Math.max(...catEntries.map(([_,w]) => Math.abs(w)), 1);
        catContainer.innerHTML = "<div class='profile-section-title'>类别偏好</div>" + catEntries.map(([cat, weight]) => {
          const pct = Math.min(100, Math.max(0, (weight / maxCat) * 100));
          return `<div class="cat-bar-container">
            <div class="cat-bar-label">${cat} (${weight.toFixed(2)})</div>
            <div class="cat-bar-bg"><div class="cat-bar-fill" style="width:${pct}%"></div></div>
          </div>`;
        }).join("");
      } else {
        catContainer.innerHTML = "";
      }
      
      // Negative Terms/Categories
      const negContainer = byId("negativeTags");
      const allNeg = [...negTerms, ...negCats];
      if (allNeg.length > 0) {
        negContainer.innerHTML = "<div class='profile-section-title'>负向过滤</div>" + 
          allNeg.map(t => `<span class="neg-tag">${t}</span>`).join("");
      } else {
        negContainer.innerHTML = "";
      }
    }
  </script>
</body>
</html>
"""


@app.post("/sync", response_model=SyncResponse)
def sync(req: SyncRequest) -> SyncResponse:
    started = time.time()
    with db_cursor(commit=True) as cur:
        try:
            harvest = harvest_sets(cur, req.days, req.backfill)
        except Exception as exc:
            LOGGER.exception("sync failed")
            raise HTTPException(status_code=500, detail=str(exc))

        deleted = 0
        if req.cleanup:
            deleted = cleanup_old_papers(cur, req.days)
        if req.rebuild_fts:
            rebuild_fts(cur)
        cur.connection.commit()

    elapsed = time.time() - started
    return SyncResponse(
        inserted=harvest["inserted"],
        updated=harvest["updated"],
        deleted=deleted,
        elapsed_seconds=elapsed,
        checkpoint=harvest,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    with db_cursor() as cur:
        results, debug = search_papers(
            cur,
            raw_query=req.raw_query,
            filters=req.filters,
            user_id=req.user_id,
            size=req.size,
            page=req.page,
            disable_pref=False,
        )
    return SearchResponse(results=results, debug=debug)


@app.get("/paper/{arxiv_id}")
def get_paper(arxiv_id: str) -> Dict[str, Any]:
    with db_cursor() as cur:
        row = cur.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="paper not found")
        payload = dict(row)
        payload["authors"] = json.loads(payload.pop("authors_json"))
        payload["categories"] = json.loads(payload.pop("categories_json"))
        payload["cross_categories"] = json.loads(payload.pop("cross_categories_json"))
        return payload


@app.post("/feed", response_model=FeedResponse)
def feed(req: FeedRequest) -> FeedResponse:
    with db_cursor() as cur:
        profile = cur.execute(
            "SELECT profile_json FROM user_profile WHERE user_id=?", (req.user_id,)
        ).fetchone()
        profile_json = json.loads(profile[0]) if profile else {}
        raw_query, filters = build_feed_query(profile_json, req.time_range_days)
        results, _debug = search_papers(
            cur,
            raw_query=raw_query,
            filters=filters,
            user_id=req.user_id,
            size=req.size * 3,
            page=req.page,
        )
        results = diversify_results(results, title_overlap=0.8)
    return FeedResponse(results=results[: req.size])


@app.post("/event", response_model=EventResponse)
def event(req: EventRequest) -> EventResponse:
    with db_cursor(commit=True) as cur:
        prev_row = cur.execute(
            "SELECT action FROM user_events WHERE user_id=? AND arxiv_id=? ORDER BY ts DESC LIMIT 1",
            (req.user_id, req.arxiv_id),
        ).fetchone()
        prev_action = prev_row["action"] if prev_row else None
        cur.execute(
            "DELETE FROM user_events WHERE user_id=? AND arxiv_id=?",
            (req.user_id, req.arxiv_id),
        )
        cur.execute(
            "INSERT INTO user_events(user_id, arxiv_id, action, ts, context_json) "
            "VALUES(?,?,?,?,?)",
            (req.user_id, req.arxiv_id, req.action, time.strftime("%Y-%m-%dT%H:%M:%SZ"), json.dumps(req.context or {}, ensure_ascii=True)),
        )
        profile = update_profile_with_prev(cur, req.user_id, req.arxiv_id, req.action, prev_action)
        summary = profile_summary(profile)
        if req.action == "save":
            row = cur.execute(
                "SELECT pdf_url FROM papers WHERE arxiv_id=?", (req.arxiv_id,)
            ).fetchone()
            if row and row["pdf_url"]:
                user_dir = os.path.join(CONFIG.saved_dir, req.user_id)
                os.makedirs(user_dir, exist_ok=True)
                filename = f"{req.arxiv_id}.pdf"
                path = os.path.join(user_dir, filename)
                if not os.path.exists(path):
                    try:
                        import requests

                        resp = requests.get(row["pdf_url"], timeout=60)
                        resp.raise_for_status()
                        with open(path, "wb") as f:
                            f.write(resp.content)
                        cur.execute(
                            "INSERT OR IGNORE INTO saved_files(user_id, path, file_mtime, ingested_at) "
                            "VALUES(?,?,?,?)",
                            (req.user_id, path, os.path.getmtime(path), time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                        )
                    except Exception as exc:
                        LOGGER.info("save pdf failed: %s", exc)
    return EventResponse(ok=True, profile_summary=summary)


def _extract_pdf_text(path: str, max_pages: int) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except Exception:
        return os.path.basename(path), ""
    reader = PdfReader(path)
    title = os.path.basename(path)
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)
    text_parts = []
    for page in reader.pages[:max_pages]:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    return title, " ".join(text_parts)


@app.post("/scan_saved", response_model=ScanSavedResponse)
def scan_saved(req: ScanSavedRequest) -> ScanSavedResponse:
    user_dir = os.path.join(CONFIG.saved_dir, req.user_id)
    os.makedirs(user_dir, exist_ok=True)
    processed = 0
    skipped = 0
    with db_cursor(commit=True) as cur:
        for filename in os.listdir(user_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(user_dir, filename)
            mtime = os.path.getmtime(path)
            row = cur.execute(
                "SELECT file_mtime FROM saved_files WHERE user_id=? AND path=?",
                (req.user_id, path),
            ).fetchone()
            if row and float(row["file_mtime"]) >= mtime:
                skipped += 1
                continue
            title, text = _extract_pdf_text(path, CONFIG.saved_scan_max_pages)
            llm_terms = []
            if CONFIG.llm_pdf_keywords_enabled:
                snippet = text[: CONFIG.llm_pdf_max_chars]
                llm_terms = extract_keywords(title, snippet)
            if llm_terms:
                update_profile_from_terms(cur, req.user_id, llm_terms, "save")
            else:
                update_profile_from_text(cur, req.user_id, title, text, "save")
            cur.execute(
                "INSERT INTO user_events(user_id, arxiv_id, action, ts, context_json) "
                "VALUES(?,?,?,?,?)",
                (req.user_id, path, "save", time.strftime("%Y-%m-%dT%H:%M:%SZ"), json.dumps({"source": "local_pdf"}, ensure_ascii=True)),
            )
            cur.execute(
                "INSERT INTO saved_files(user_id, path, file_mtime, ingested_at) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id, path) DO UPDATE SET file_mtime=excluded.file_mtime, ingested_at=excluded.ingested_at",
                (req.user_id, path, mtime, time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
            processed += 1
        profile_row = cur.execute(
            "SELECT profile_json FROM user_profile WHERE user_id=?",
            (req.user_id,),
        ).fetchone()
        summary = profile_summary(json.loads(profile_row["profile_json"])) if profile_row else profile_summary({})
    return ScanSavedResponse(ok=True, processed=processed, skipped=skipped, profile_summary=summary)


@app.get("/profile/{user_id}", response_model=ProfileResponse)
def profile(user_id: str) -> ProfileResponse:
    with db_cursor() as cur:
        row = cur.execute(
            "SELECT profile_json, updated_at, version FROM user_profile WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if not row:
            return ProfileResponse(user_id=user_id, profile={}, updated_at=None, version=None)
        return ProfileResponse(
            user_id=user_id,
            profile=json.loads(row["profile_json"]),
            updated_at=row["updated_at"],
            version=row["version"],
        )
