"""
Entry point for running the Flotorch Evaluation MCP Server.

Usage:
    python -m flotorch_eval_mcp
"""

import asyncio

from flotorch_eval_mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())
