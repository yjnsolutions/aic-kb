import asyncio
from typing import Optional

import typer
from dotenv import load_dotenv

from aic_kb.search.search import run_rag_agent

load_dotenv()
app = typer.Typer()


@app.command()
def get_package_documentation(
    package_name: str,
    version: Optional[str] = None,
    depth: Optional[int] = typer.Option(None, help="Maximum recursion depth for crawling (None for unlimited)"),
    strategy: str = typer.Option("bfs", help="Crawling strategy: 'bfs' or 'dfs'"),
    ignore_robots: bool = typer.Option(False, help="Ignore robots.txt rules"),
    limit: Optional[int] = typer.Option(None, help="Maximum number of pages to crawl (None for unlimited)"),
    disable_caching: bool = typer.Option(False, help="Disable caching (for crawling, completion and embeddings)"),
):
    """
    Scrape documentation for a Python package and save as markdown files.

    Usage:
        uv run aic-kb get-package-documentation requests
        uv run aic-kb get-package-documentation requests --version 2.31.0 --depth 2 --strategy dfs --embedding-model text-embedding-3-small --limit 10
    """
    from aic_kb.pypi_doc_scraper.crawl import _get_package_documentation

    asyncio.run(
        _get_package_documentation(
            package_name, version, depth, strategy, ignore_robots, limit, caching_enabled=not disable_caching
        )
    )


@app.command()
def search(
    text: str,
    match_count: int = typer.Option(3, help="Number of matches to return"),
):
    """
    Search the documentation database using semantic similarity.

    Returns the most relevant documentation chunks for the given search text.
    """
    asyncio.run(run_rag_agent(text, match_count))
    # asyncio.run(_search(text, match_count))


def main():
    app()


if __name__ == "__main__":
    main()
