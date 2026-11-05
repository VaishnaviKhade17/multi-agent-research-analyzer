#paper_tools.py

"""
Academic paper search tools: arXiv and Semantic Scholar.

Both are free, no API key required. Results are cached via Db.py so a
repeated query doesn't re-hit the live API. Query volume is also reduced
upstream in contradiction_agent.py (2 short queries instead of 5
sentence-length ones), which is the main fix for the 429s seen in your
run — this retry is a secondary safety net.

Design note: each tool returns real API fields only (title, authors, year,
abstract, url) — nothing is invented or reworded, matching strict data
fidelity for anything downstream that cites these papers.
"""

import time
import requests
import xml.etree.ElementTree as ET
from langchain.tools import tool
from Db import get_cached, set_cached

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _format_papers(papers: list) -> str:
    if not papers:
        return "No papers found."
    out = []
    for p in papers:
        authors = ", ".join(p["authors"]) if p["authors"] else "Unknown"
        out.append(
            f"Title: {p['title']}\n"
            f"Authors: {authors}\n"
            f"Year: {p['year']}\n"
            f"URL: {p['url']}\n"
            f"Abstract: {p['abstract'][:400]}\n"
        )
    return "\n----\n".join(out)


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers on a topic. Returns title, authors,
    year, abstract, and URL for each paper found. Prefer this over general
    web search for academic/research questions."""
    cached = get_cached("arxiv", query)
    if cached is not None:
        return _format_papers(cached)

    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": 5},
            timeout=10,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        papers = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")
            url = entry.find("atom:id", ARXIV_NS).text.strip()
            published = entry.find("atom:published", ARXIV_NS).text[:4]
            authors = [
                a.find("atom:name", ARXIV_NS).text
                for a in entry.findall("atom:author", ARXIV_NS)
            ]
            papers.append(
                {
                    "source": "arxiv",
                    "title": title,
                    "authors": authors,
                    "year": published,
                    "abstract": summary,
                    "url": url,
                }
            )
        set_cached("arxiv", query, papers)
        return _format_papers(papers)
    except Exception as e:
        return f"arXiv search failed ({e}). Try semantic_scholar_search instead."


@tool
def semantic_scholar_search(query: str) -> str:
    """Search Semantic Scholar for academic papers on a topic. Returns title,
    authors, year, abstract, and URL for each paper found. Prefer this over
    general web search for academic/research questions."""
    cached = get_cached("semantic_scholar", query)
    if cached is not None:
        return _format_papers(cached)

    last_error = None

    # One retry with backoff on 429 — reduced query volume upstream is
    # the primary fix, this just absorbs the occasional remaining hit.
    for attempt in range(2):
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 5, "fields": "title,authors,year,abstract,url"},
                timeout=10,
            )

            if resp.status_code == 429:
                if attempt == 0:
                    print("Semantic Scholar rate-limited, retrying in 3s...")
                    time.sleep(3)
                    continue
                return (
                    "Semantic Scholar rate-limited (429) after retry. "
                    "Relying on arxiv_search for this query instead."
                )

            resp.raise_for_status()
            data = resp.json()
            papers = []
            for p in data.get("data", []):
                papers.append(
                    {
                        "source": "semantic_scholar",
                        "title": p.get("title") or "Untitled",
                        "authors": [a.get("name", "") for a in p.get("authors", [])],
                        "year": p.get("year") or "N/A",
                        "abstract": p.get("abstract") or "No abstract available.",
                        "url": p.get("url") or "",
                    }
                )
            set_cached("semantic_scholar", query, papers)
            return _format_papers(papers)

        except Exception as e:
            last_error = e
            break

    return f"Semantic Scholar search failed ({last_error}). Try arxiv_search instead."