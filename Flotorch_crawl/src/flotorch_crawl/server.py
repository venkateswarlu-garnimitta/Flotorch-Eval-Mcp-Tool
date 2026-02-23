#!/usr/bin/env python3
"""
Flotorch Crawl MCP Server.

Exposes web search, scraping, link discovery, and recursive crawl tools via FastMCP.
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

load_dotenv()
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from flotorch_crawl.scraper import list_links, scrape_links, scrape_url
from flotorch_crawl.search import search_web

logging.basicConfig(
    format="[%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

mcp = FastMCP(
    "Flotorch Crawl",
    instructions=(
        "Flotorch Crawl MCP Server: Web search, scraping, link discovery, "
        "and recursive crawling. Use search_web for web search, scrape_url for "
        "single-page content, list_links to discover links, and scrape_links "
        "to crawl and scrape multiple pages from a starting URL."
    ),
    json_response=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    stateless_http=True,
)


async def _run_sync(fn, *args, **kwargs):
    """Run sync blocking function in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


@mcp.custom_route("/.well-known/flotorch-crawl-mcp", methods=["GET"])
async def discovery(_request: Request) -> JSONResponse:
    """Discovery endpoint for transport detection."""
    return JSONResponse({
        "transport": "HTTP_STREAMABLE",
        "protocol": "streamable-http",
        "message": "Flotorch Crawl MCP Server - Set transport to HTTP_STREAMABLE",
    })


@mcp.tool()
async def search_web_tool(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> str:
    """
    Perform a web search and return results with titles, URLs, and snippets.

    Uses DuckDuckGo by default. For Google search, set SERPER_API_KEY.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 10).
        region: Region for search, e.g. wt-wt (global), us-en, uk-en.

    Returns:
        JSON string with list of {title, url, snippet}.
    """
    try:
        results = await _run_sync(search_web, query, max_results, region)
        return json.dumps({"results": results}, indent=2)
    except Exception as e:
        logger.exception("search_web failed")
        return json.dumps({"error": str(e), "results": []})


@mcp.tool()
async def scrape_url_tool(url: str) -> str:
    """
    Scrape a single URL and extract main text content.

    Fetches the page, removes scripts/styles, and returns cleaned text.

    Args:
        url: Full URL to scrape (e.g. https://example.com/page).

    Returns:
        JSON with url, title, text, and optional error.
    """
    try:
        result = await _run_sync(scrape_url, url)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("scrape_url failed")
        return json.dumps({"url": url, "title": "", "text": "", "error": str(e)})


@mcp.tool()
async def list_links_tool(url: str, same_domain_only: bool = True) -> str:
    """
    List all links found on a webpage.

    Fetches the URL and extracts anchor tags with href. Optionally filters
    to links on the same domain only.

    Args:
        url: Full URL to fetch and analyze.
        same_domain_only: If true, only return links on the same domain (default).

    Returns:
        JSON with url, links (list of {href, text}), and optional error.
    """
    try:
        result = await _run_sync(list_links, url, same_domain_only)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("list_links failed")
        return json.dumps({"url": url, "links": [], "error": str(e)})


@mcp.tool()
async def scrape_links_tool(
    url: str,
    same_domain_only: bool = True,
    max_pages: int = 20,
) -> str:
    """
    Crawl a page: list its links, then scrape each linked page.

    Combines list_links and scrape_url. Fetches the starting URL, discovers
    links, then scrapes each link up to max_pages.

    Args:
        url: Starting URL.
        same_domain_only: Only follow links on the same domain (default).
        max_pages: Maximum number of linked pages to scrape (default 20).

    Returns:
        JSON with base_url, pages (list of {url, title, text}), and errors.
    """
    try:
        result = await _run_sync(scrape_links, url, same_domain_only, max_pages)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("scrape_links failed")
        return json.dumps({
            "base_url": url,
            "pages": [],
            "errors": [str(e)],
        })


async def main() -> None:
    """Run the MCP server."""
    port = int(os.getenv("PORT", 8081))
    host = os.getenv("HOST", "0.0.0.0")

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.log_level = "INFO"

    logger.info(
        f"Flotorch Crawl MCP Server starting on http://{host}:{port}\n"
        "Streamable HTTP at / | Discovery at /.well-known/flotorch-crawl-mcp"
    )

    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
