# Flotorch Crawl MCP Server

MCP server for web search, scraping, link discovery, and recursive crawling.

## Quick Start

```bash
cd Flotorch_crawl
pip install -e .
python -m flotorch_crawl
```

Server runs at `http://0.0.0.0:8081`.

## Tools

| Tool | Description |
|------|-------------|
| `search_web_tool` | Web search; returns titles, URLs, snippets |
| `scrape_url_tool` | Scrape a single URL; extract main text |
| `list_links_tool` | List all links on a page |
| `scrape_links_tool` | List links, then scrape each (crawl) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8081 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `SERPER_API_KEY` | — | Optional; enables Google search via Serper |
| `CRAWL_MAX_PAGE_SIZE` | 1048576 | Max bytes per page (1MB) |
| `CRAWL_MAX_LINKS_PER_PAGE` | 100 | Max links to extract per page |
| `CRAWL_MAX_PAGES` | 20 | Max pages for scrape_links |
| `CRAWL_REQUEST_TIMEOUT` | 30 | HTTP timeout (seconds) |
| `CRAWL_USER_AGENT` | FlotorchCrawl/1.0 | User-Agent header |

## MCP Client Config

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "http://localhost:8081",
  "timeout": 60000,
  "sse_read_timeout": 60000
}
```

## Project Structure

```
Flotorch_crawl/
├── src/flotorch_crawl/
│   ├── config.py    # Env and limits
│   ├── search.py    # Web search (DuckDuckGo / Serper)
│   ├── scraper.py   # Scrape, list links, crawl
│   ├── server.py    # FastMCP server and tools
│   ├── __init__.py
│   └── __main__.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- httpx, beautifulsoup4, duckduckgo-search, fastmcp
