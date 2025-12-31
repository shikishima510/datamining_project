# arXiv 本地检索 + 个性化推荐系统（使用指南）

本项目提供：
- 本地 arXiv 元数据同步（cs/eess，最近 180 天可配置）
- SQLite FTS5 全文检索 + BM25
- 反馈闭环（view/like/dislike/save/hide）
- 本地 PDF 保存 + 从 PDF 生成偏好画像
- 引用数 + PageRank 图谱（可选）

---

## 1) 启动与同步

### 环境准备
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 同步元数据
```
python sync_arxiv.py --days 180
```

### 启动 API
```
python -m app
```

访问：
- 首页：`http://127.0.0.1:8000/`
- 文档：`http://127.0.0.1:8000/docs`

---

## 2) 搜索与推荐

### 搜索（带关键词）
- 前端：输入关键词后点“搜索”
- API：
```
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_user","raw_query":"sam3","filters":{"time_range_days":180,"categories_include":["cs.CV"],"categories_exclude":[]},"size":10,"page":1}'
```

### 猜你想看（无 query）
- 前端：点“猜你想看”
- API：
```
curl -X POST http://127.0.0.1:8000/feed \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_user","time_range_days":180,"size":10,"page":1}'
```

---

## 3) 用户反馈（影响画像）

### 在搜索结果卡片里选择反馈
- 每条结果可选择 `view / like / dislike / save / hide`
- 点击“提交反馈”后自动更新画像

### API 方式
```
curl -X POST http://127.0.0.1:8000/event \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_user","arxiv_id":"2401.17868","action":"like"}'
```

**说明**
- 同一篇论文只保留最后一次反馈（不会累积）
- `save` 会自动下载 PDF 到 `saved/<user_id>/<arxiv_id>.pdf`

---

## 4) 本地 PDF 作为偏好输入

把 PDF 放到：
```
saved/<user_id>/
```
然后同步：
```
curl -X POST http://127.0.0.1:8000/scan_saved \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_user"}'
```

默认会用 LLM 抽关键词（需开启 LLM）。抽不到则回退为本地关键词抽取。

---

## 5) 引用数 + PageRank 图谱（可选）

### 引用数抓取（Semantic Scholar）
```
python update_citations.py --limit 0 --batch-size 500 --sleep 1
```

### 引用关系图谱 + PageRank
```
python update_graph.py --days 180 --limit 0 --batch-size 200 --sleep 0.5
```

在搜索/推荐中可以通过权重影响排序：
- `SCORE_WEIGHT_CITATIONS`
- `SCORE_WEIGHT_PAGERANK`

---

## 6) 用户画像调试

前端首页提供“当前用户画像”折叠面板，可查看 `term_weights`。

API 方式：
```
curl http://127.0.0.1:8000/profile/demo_user
```

---

## 7) 常见问题

**Q: 搜索结果不准**
- 检查分类过滤与时间范围
- `sam3` 这类带数字关键词已做特殊处理

**Q: 猜你想看无结果**
- 用户画像为空（先 like/save）
- 类别过滤太严格

**Q: LLM 关键词抽取不生效**
- 检查 `LLM_ENABLED=true` 且 key 正确

**Q: 429 速率限制**
- 降低 `--batch-size`
- 提高 `--sleep`
- 配置 `SEMANTIC_SCHOLAR_API_KEY`

---

## 8) 常用环境变量

```
APP_DB_PATH=data/app.db
SAVED_PDF_DIR=saved
LLM_ENABLED=false
LLM_BASE_URL=https://api.openai.com/v1/chat
LLM_API_KEY=...
LLM_MODEL=gpt-4o
SCORE_WEIGHT_PREF=0.015
SCORE_WEIGHT_CITATIONS=0.25
SCORE_WEIGHT_PAGERANK=0.35
```

---

## 9) 重置数据

清空所有用户画像：
```
sqlite3 data/app.db "DELETE FROM user_profile;"
```


清空已扫描 PDF 记录：
```
sqlite3 data/app.db "DELETE FROM saved_files;"
```
