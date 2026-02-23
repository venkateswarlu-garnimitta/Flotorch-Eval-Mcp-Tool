"""
Configuration for Flotorch Crawl MCP Server.

Environment variables, defaults, and scraper settings.
"""

import os
from typing import Optional


def get_serper_api_key() -> Optional[str]:
    """Return Serper API key if set (enables Google search)."""
    return os.getenv("SERPER_API_KEY")


def get_max_scrape_size() -> int:
    """Max bytes to fetch per URL (default 1MB)."""
    return int(os.getenv("CRAWL_MAX_PAGE_SIZE", "1048576"))


def get_max_links_per_page() -> int:
    """Max links to extract per page for list_links / scrape_links."""
    return int(os.getenv("CRAWL_MAX_LINKS_PER_PAGE", "100"))


def get_max_pages_to_scrape() -> int:
    """Max pages to scrape when using scrape_links (crawl)."""
    return int(os.getenv("CRAWL_MAX_PAGES", "20"))


def get_request_timeout() -> float:
    """HTTP request timeout in seconds."""
    return float(os.getenv("CRAWL_REQUEST_TIMEOUT", "30"))


def get_user_agent() -> str:
    """User-Agent header for HTTP requests."""
    return os.getenv(
        "CRAWL_USER_AGENT",
        "Mozilla/5.0 (compatible; FlotorchCrawl/1.0; +https://flotorch.ai)",
    )
