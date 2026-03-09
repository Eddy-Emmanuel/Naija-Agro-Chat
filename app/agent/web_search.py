"""Lightweight web search helper (SerpAPI).

This module provides a simple wrapper for hitting the SerpAPI search REST
endpoint and formatting the results into a context string that can be used
as a retrieval fallback when the local knowledge base has no relevant docs.

To use this helper, set SERPAPI_API_KEY in your environment (e.g. in .env).

SerpAPI docs: https://serpapi.com/
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


def web_search(query: str, num_results: int = 5, api_key: Optional[str] = None) -> List[Dict]:
    """Perform a web search and return a list of result dicts.

    Returns a list of dicts with keys: title, snippet, link.
    """

    api_key = api_key or os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not set in the environment")

    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
    }

    url = f"{SERPAPI_SEARCH_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "NaijaAgroChat/1.0"})

    try:
        with urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Web search request failed: {e}")

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            }
        )

    return results


def format_search_results(results: List[Dict]) -> str:
    """Convert web search results into a chunked context string."""
    if not results:
        return ""

    chunks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or "(no title)"
        snippet = r.get("snippet") or ""
        link = r.get("link") or ""
        chunks.append(f"[Web result #{i}] {title}\n{snippet}\n{link}")

    return "\n\n---\n\n".join(chunks)
