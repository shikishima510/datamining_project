import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .config import CONFIG
from .db import get_sync_state, now_utc, set_sync_state
from .indexer import upsert_papers_batch

try:
    from lxml import etree as ET
except Exception:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore


LOGGER = logging.getLogger(__name__)

BASE_URL = "https://oaipmh.arxiv.org/oai"


def _sleep_with_jitter(base_seconds: float) -> None:
    jitter = random.random() * CONFIG.harvest_sleep_jitter
    time.sleep(base_seconds + jitter)


def _parse_xml(text: str) -> ET.Element:
    return ET.fromstring(text.encode("utf-8"))


def _findall(elem: ET.Element, path: str) -> List[ET.Element]:
    return elem.findall(path)


def _findtext(elem: ET.Element, path: str) -> Optional[str]:
    node = elem.find(path)
    return node.text.strip() if node is not None and node.text else None


def _request(params: Dict[str, str]) -> requests.Response:
    timeout = (CONFIG.request_timeout_connect, CONFIG.request_timeout_read)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 503:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    delay = int(retry_after) + random.randint(0, 2)
                    LOGGER.info("503 Retry-After=%s", retry_after)
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt >= CONFIG.max_retry:
                raise
            backoff = min(2 ** attempt, 60)
            LOGGER.info("request error %s, retry in %ss", exc, backoff)
            time.sleep(backoff)


def list_sets() -> List[str]:
    LOGGER.info("Listing sets")
    resp = _request({"verb": "ListSets"})
    root = _parse_xml(resp.text)
    set_specs = []
    for set_el in _findall(root, ".//{*}set"):
        spec = _findtext(set_el, "{*}setSpec")
        if spec:
            set_specs.append(spec)
    return set_specs


def select_sets(set_specs: List[str]) -> List[str]:
    if "cs" in set_specs and "eess" in set_specs:
        return ["cs", "eess"]
    selected = []
    for target in ("cs", "eess"):
        candidates = [s for s in set_specs if s.startswith(target + ":")]
        if candidates:
            selected.append(candidates[0])
    return selected


def _parse_versions(arxiv_raw: ET.Element) -> Tuple[Optional[str], Optional[str], int]:
    versions = _findall(arxiv_raw, ".//{*}version")
    if not versions:
        return None, None, 0
    dates = []
    for ver in versions:
        dt = _findtext(ver, "{*}date")
        if dt:
            dates.append(dt)
    if not dates:
        return None, None, len(versions)
    published_at = dates[0]
    updated_at = dates[-1]
    return published_at, updated_at, len(versions)


def _parse_authors(arxiv_raw: ET.Element) -> List[str]:
    authors = []
    for author in _findall(arxiv_raw, ".//{*}author"):
        keyname = _findtext(author, "{*}keyname") or ""
        forenames = _findtext(author, "{*}forenames") or ""
        name = f"{forenames} {keyname}".strip()
        if name:
            authors.append(name)
    return authors


def _parse_record(record: ET.Element) -> Optional[Dict[str, Any]]:
    header = record.find("{*}header")
    if header is None:
        return None
    if header.attrib.get("status") == "deleted":
        return None
    datestamp = _findtext(header, "{*}datestamp")

    metadata = record.find("{*}metadata")
    if metadata is None:
        return None
    arxiv_raw = metadata.find("{*}arXivRaw")
    if arxiv_raw is None:
        return None

    arxiv_id = _findtext(arxiv_raw, "{*}id")
    if not arxiv_id:
        return None
    title = _findtext(arxiv_raw, "{*}title") or ""
    abstract = _findtext(arxiv_raw, "{*}abstract") or ""
    authors = _parse_authors(arxiv_raw)

    categories_text = _findtext(arxiv_raw, "{*}categories") or ""
    categories = [c for c in categories_text.split() if c]
    primary_category = categories[0] if categories else ""
    cross_categories = categories[1:] if len(categories) > 1 else []

    doi = _findtext(arxiv_raw, "{*}doi")
    journal_ref = _findtext(arxiv_raw, "{*}journal-ref")
    comment = _findtext(arxiv_raw, "{*}comments")

    published_at, updated_at, version_count = _parse_versions(arxiv_raw)
    cross_list_count = max(len(cross_categories), 0)
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "primary_category": primary_category,
        "categories": categories,
        "cross_categories": cross_categories,
        "published_at": published_at,
        "updated_at": updated_at,
        "doi": doi,
        "journal_ref": journal_ref,
        "comment": comment,
        "version_count": version_count,
        "cross_list_count": cross_list_count,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "record_datestamp": datestamp,
    }


def _parse_list_records(text: str) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    root = _parse_xml(text)
    error = root.find(".//{*}error")
    if error is not None:
        code = error.attrib.get("code")
        raise ValueError(code or "unknown_error")

    records = []
    last_datestamp = None
    for rec in _findall(root, ".//{*}record"):
        parsed = _parse_record(rec)
        if parsed:
            records.append(parsed)
            if parsed.get("record_datestamp"):
                last_datestamp = parsed["record_datestamp"]

    token_el = root.find(".//{*}resumptionToken")
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return records, token, last_datestamp


def _normalize_oai_datetime(value: str) -> str:
    if not value:
        return value
    if value.endswith("Z"):
        return value
    if "+" in value:
        try:
            dt = datetime.fromisoformat(value)
            dt = dt.astimezone(timezone.utc)
            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except ValueError:
            return value
    if "T" in value:
        try:
            dt = datetime.fromisoformat(value)
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except ValueError:
            return value
    return value


def _clamp_from_dt(value: str) -> str:
    if not value:
        return value
    now_dt = datetime.now(timezone.utc)
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed > now_dt:
        safe = now_dt - timedelta(days=1)
        return safe.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return value


def _shift_from_dt(value: str, days: int) -> str:
    if not value:
        return value
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(value)
        shifted = parsed - timedelta(days=days)
        return shifted.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        return value


def _to_oai_day(value: str) -> str:
    if not value:
        return value
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(value)
        return parsed.date().isoformat()
    except ValueError:
        return value


def _get_from_dt(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def harvest_sets(cur, days: int, backfill: bool, sets_override: Optional[List[str]] = None) -> Dict[str, Any]:
    state = {
        "last_harvest_datestamp": get_sync_state(cur, "last_harvest_datestamp"),
        "active_setSpec": get_sync_state(cur, "active_setSpec"),
        "active_from_dt": get_sync_state(cur, "active_from_dt"),
        "active_resumption_token": get_sync_state(cur, "active_resumption_token"),
    }

    set_specs = sets_override or select_sets(list_sets())
    if not set_specs:
        raise RuntimeError("No suitable sets found for cs/eess")

    total_inserted = 0
    total_updated = 0
    max_datestamp = state["last_harvest_datestamp"]

    for set_spec in set_specs:
        if state["active_setSpec"] and state["active_setSpec"] != set_spec:
            continue

        if state["active_from_dt"]:
            from_dt = state["active_from_dt"]
        else:
            if backfill or not state["last_harvest_datestamp"]:
                from_dt = _get_from_dt(days + 10)
            else:
                from_dt = state["last_harvest_datestamp"]
        from_dt = _normalize_oai_datetime(from_dt)
        from_dt = _clamp_from_dt(from_dt)

        LOGGER.info("Harvesting set=%s from=%s", set_spec, from_dt)
        resumption_token = state["active_resumption_token"]
        retry_bad_token = 0
        retry_bad_argument = 0

        while True:
            params = {"verb": "ListRecords"}
            if resumption_token:
                params["resumptionToken"] = resumption_token
            else:
                params.update({
                    "metadataPrefix": "arXivRaw",
                    "set": set_spec,
                    "from": _to_oai_day(from_dt),
                })

            resp = _request(params)
            _sleep_with_jitter(CONFIG.harvest_sleep_seconds)

            try:
                records, next_token, last_datestamp = _parse_list_records(resp.text)
            except ValueError as exc:
                if str(exc) == "badArgument" and not resumption_token:
                    retry_bad_argument += 1
                    if retry_bad_argument > 3:
                        raise RuntimeError("badArgument persists for from_dt")
                    from_dt = _shift_from_dt(from_dt, days=365)
                    LOGGER.info("badArgument, retry with from=%s", from_dt)
                    continue
                if str(exc) == "badResumptionToken":
                    LOGGER.info("badResumptionToken, rollback to from=%s", from_dt)
                    retry_bad_token += 1
                    if retry_bad_token > 2:
                        raise RuntimeError("badResumptionToken exceeded")
                    resumption_token = None
                    continue
                raise

            inserted, updated = upsert_papers_batch(cur, records)
            total_inserted += inserted
            total_updated += updated

            if last_datestamp:
                max_datestamp = max(last_datestamp, max_datestamp or last_datestamp)

            set_sync_state(cur, "active_setSpec", set_spec)
            set_sync_state(cur, "active_from_dt", from_dt)
            set_sync_state(cur, "active_resumption_token", next_token or "")
            set_sync_state(cur, "last_sync_time", now_utc())
            if max_datestamp:
                set_sync_state(cur, "last_harvest_datestamp", max_datestamp)

            cur.connection.commit()

            if not next_token:
                break
            resumption_token = next_token

        set_sync_state(cur, "active_setSpec", "")
        set_sync_state(cur, "active_from_dt", "")
        set_sync_state(cur, "active_resumption_token", "")
        cur.connection.commit()

    return {
        "inserted": total_inserted,
        "updated": total_updated,
        "last_harvest_datestamp": max_datestamp,
        "sets": set_specs,
    }
