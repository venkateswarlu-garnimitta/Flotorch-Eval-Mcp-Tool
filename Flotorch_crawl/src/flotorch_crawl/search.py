"""
Web search module for Flotorch Crawl.

Supports DuckDuckGo (default) and optional Serper (Google) when API key is set.
"""

import logging
from typing import Any, Dict, List

from flotorch_crawl.config import get_serper_api_key

logger = logging.getLogger(__name__)


def search_web(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> List[Dict[str, Any]]:
    """
    Execute web search and return structured results.

    Uses DuckDuckGo by default. If SERPER_API_KEY is set, uses Serper for
    Google search instead.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10).
        region: Region/locale for search (e.g. wt-wt, us-en).

    Returns:
        List of dicts with keys: title, url, snippet (or body).
    """
    api_key = get_serper_api_key()
    if api_key:
        return _search_serper(query, max_results, api_key)
    return _search_duckduckgo(query, max_results, region)


def _search_duckduckgo(
    query: str,
    max_results: int,
    region: str,
) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise ImportError(
            "duckduckgo-search is required for web search. "
            "Install with: pip install duckduckgo-search"
        ) from e

    results: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("url", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
    except Exception as e:
        logger.exception("DuckDuckGo search failed")
        raise RuntimeError(f"Search failed: {e}") from e

    return results


def _search_serper(
    query: str,
    max_results: int,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Search using Serper (Google results, requires API key)."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx required for Serper. pip install httpx") from e

    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": min(max_results, 100)}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Serper API error: {e.response.text}") from e
    except Exception as e:
        logger.exception("Serper search failed")
        raise RuntimeError(f"Search failed: {e}") from e

    results: List[Dict[str, Any]] = []
    for r in data.get("organic", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        })
    return results
