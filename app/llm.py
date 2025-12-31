import json
import logging
from typing import List

import requests

from .config import CONFIG

LOGGER = logging.getLogger(__name__)

_LLM_STOPWORDS = {
    "method", "methods", "approach", "approaches", "model", "models", "paper",
    "study", "studies", "analysis", "results", "framework", "algorithm",
    "estimation", "estimate", "estimating", "prediction", "predict", "predicting",
    "evaluation", "benchmark", "benchmarks", "dataset", "datasets", "data",
    "system", "systems", "experiment", "experiments", "performance", "result",
    "using", "use", "used", "based", "problem", "problems", "task", "tasks",
    "we", "our", "propose", "proposed", "novel", "new", "robust",
}


def _post_openai(payload: dict) -> str:
    url = CONFIG.llm_base_url
    if not url.endswith("/v1/chat"):
        url = f"{CONFIG.llm_base_url}/v1/chat"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {CONFIG.llm_api_key}"},
        json=payload,
        timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _post_azure(payload: dict) -> str:
    url = (
        f"{CONFIG.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
        f"{CONFIG.azure_openai_deployment}/chat/completions"
        f"?api-version={CONFIG.azure_openai_api_version}"
    )
    resp = requests.post(
        url,
        headers={"api-key": CONFIG.azure_openai_api_key},
        json=payload,
        timeout=(CONFIG.request_timeout_connect, CONFIG.request_timeout_read),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_keywords(title: str, text: str) -> List[str]:
    if not CONFIG.llm_enabled:
        return []
    if CONFIG.azure_openai_endpoint and CONFIG.azure_openai_api_key and CONFIG.azure_openai_deployment:
        use_azure = True
    else:
        use_azure = False
        if not CONFIG.llm_api_key:
            return []

    schema = {
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"],
    }
    prompt = (
        "Extract 8-12 core topical keywords/phrases that best describe the paper's domain "
        "and application. Avoid generic method terms (e.g., model, approach, estimation, "
        "framework, evaluation) and avoid venue/author names. Prefer concrete concepts "
        "and tasks. Return strict JSON matching: "
        f"{json.dumps(schema, ensure_ascii=True)}"
    )
    payload = {
        "model": CONFIG.llm_model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Title: {title}\nText: {text}",
            },
        ],
        "temperature": 0.0,
    }
    try:
        content = _post_azure(payload) if use_azure else _post_openai(payload)
        data = json.loads(content)
        keywords = data.get("keywords") or []
        if not isinstance(keywords, list):
            return []
        cleaned = []
        for k in keywords:
            val = str(k).strip().lower()
            if not val:
                continue
            if len(val) < 3:
                continue
            if val in _LLM_STOPWORDS:
                continue
            cleaned.append(val)
        return cleaned[: CONFIG.llm_pdf_max_keywords]
    except Exception as exc:
        LOGGER.info("LLM keyword extraction failed: %s", exc)
        return []
