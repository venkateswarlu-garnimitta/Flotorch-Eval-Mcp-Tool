"""
Web scraping module for Flotorch Crawl.

Fetches pages, extracts text, and discovers links using httpx + BeautifulSoup.
"""

import logging
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from flotorch_crawl.config import (
    get_max_links_per_page,
    get_max_scrape_size,
    get_request_timeout,
    get_user_agent,
)

logger = logging.getLogger(__name__)


def scrape_url(url: str) -> dict:
    """
    Fetch a URL and extract main text content.

    Args:
        url: Full URL to fetch.

    Returns:
        Dict with keys: url, title, text, error (if any).
    """
    headers = {"User-Agent": get_user_agent()}
    timeout = get_request_timeout()
    max_size = get_max_scrape_size()

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            if len(content) > max_size:
                content = content[:max_size]
            encoding = resp.charset_encoding or "utf-8"
            try:
                html = content.decode(encoding, errors="replace")
            except Exception:
                html = content.decode("utf-8", errors="replace")

    except httpx.HTTPStatusError as e:
        return {
            "url": url,
            "title": "",
            "text": "",
            "error": f"HTTP {e.response.status_code}",
        }
    except Exception as e:
        return {
            "url": url,
            "title": "",
            "text": "",
            "error": str(e),
        }

    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script, style, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        # Get main content or body
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""
        if not text:
            text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple newlines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return {"url": url, "title": title, "text": text[:50000], "error": None}
    except Exception as e:
        return {"url": url, "title": "", "text": "", "error": str(e)}


def list_links(url: str, same_domain_only: bool = True) -> dict:
    """
    Fetch a URL and list all links found on the page.

    Args:
        url: Full URL to fetch.
        same_domain_only: If True, only return links on the same domain.

    Returns:
        Dict with keys: url, links (list of {href, text}), error (if any).
    """
    headers = {"User-Agent": get_user_agent()}
    timeout = get_request_timeout()
    max_links = get_max_links_per_page()

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            encoding = resp.charset_encoding or "utf-8"
            try:
                html = content.decode(encoding, errors="replace")
            except Exception:
                html = content.decode("utf-8", errors="replace")

    except httpx.HTTPStatusError as e:
        return {"url": url, "links": [], "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"url": url, "links": [], "error": str(e)}

    try:
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(url).netloc
        seen = set()
        links: List[dict] = []

        for a in soup.find_all("a", href=True):
            if len(links) >= max_links:
                break
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:"):
                continue
            abs_url = urljoin(url, href)
            parsed = urlparse(abs_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if same_domain_only and parsed.netloc != base_domain:
                continue
            if abs_url in seen:
                continue
            seen.add(abs_url)
            text = a.get_text(strip=True) or abs_url
            links.append({"href": abs_url, "text": text[:200]})

        return {"url": url, "links": links, "error": None}
    except Exception as e:
        return {"url": url, "links": [], "error": str(e)}


def scrape_links(
    url: str,
    same_domain_only: bool = True,
    max_pages: Optional[int] = None,
) -> dict:
    """
    List links on a page, then scrape each link (recursive crawl).

    Args:
        url: Starting URL.
        same_domain_only: Only follow links on the same domain.
        max_pages: Maximum number of pages to scrape (default from config).

    Returns:
        Dict with keys: base_url, pages (list of {url, title, text}), errors.
    """
    from flotorch_crawl.config import get_max_pages_to_scrape

    limit = max_pages if max_pages is not None else get_max_pages_to_scrape()
    link_result = list_links(url, same_domain_only=same_domain_only)
    if link_result.get("error"):
        return {
            "base_url": url,
            "pages": [],
            "errors": [link_result["error"]],
        }

    links = link_result["links"][:limit]
    pages: List[dict] = []
    errors: List[str] = []

    for item in links:
        href = item["href"]
        scraped = scrape_url(href)
        if scraped.get("error"):
            errors.append(f"{href}: {scraped['error']}")
        else:
            pages.append({
                "url": scraped["url"],
                "title": scraped["title"],
                "text": scraped["text"],
            })

    return {
        "base_url": url,
        "pages": pages,
        "errors": errors if errors else [],
    }
