"""Lightweight web search helper (SerpAPI).

This module provides a simple wrapper for hitting the SerpAPI search REST
endpoint and formatting the results into a context string that can be used
as a retrieval fallback when the local knowledge base has no relevant docs.

To use this helper, set SERPAPI_API_KEY in your environment (e.g. in .env).

SerpAPI docs: https://serpapi.com/
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper

def web_search(query: str, num_results: int = 5, api_key: Optional[str] = None) -> List[Dict]:
    """Perform a web search and return a list of result dicts.

    Returns a list of dicts with keys: title, snippet, link.

    Uses LangChain utilities:
      • SerpAPIWrapper when SERPAPI_API_KEY is set
      • WikipediaAPIWrapper as a fallback when no key is available.
    """

    api_key = api_key or os.getenv("SERPAPI_API_KEY")

    if api_key:
        serp = SerpAPIWrapper(serpapi_api_key=api_key)
        raw = serp.run(query)

        # SerpAPIWrapper returns a single text string; we split it into results.
        # This is a heuristic but works well for short snippets.
        items = [item.strip() for item in raw.split("\n") if item.strip()]
        results = [
            {"title": f"Result {i+1}", "snippet": item, "link": ""}
            for i, item in enumerate(items[:num_results])
        ]
        return results

    # Fallback: use Wikipedia's summary endpoint
    wiki = WikipediaAPIWrapper().run(query)
    if not wiki:
        raise ValueError(
            "No SERPAPI_API_KEY found and Wikipedia fallback failed. "
            "Set SERPAPI_API_KEY in the environment to enable web search."
        )

    return [{"title": query, "snippet": wiki, "link": "https://en.wikipedia.org/wiki/" + query.replace(' ', '_')}]


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
