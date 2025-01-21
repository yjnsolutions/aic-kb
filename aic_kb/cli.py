import asyncio
import os
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"Hello {build_string(name, 1)}!")


@app.command()
def get_package_documentation(
    package_name: str,
    version: Optional[str] = None,
    depth: Optional[int] = typer.Option(None, help="Maximum recursion depth for crawling (None for unlimited)"),
    strategy: str = typer.Option("bfs", help="Crawling strategy: 'bfs' or 'dfs'"),
    ignore_robots: bool = typer.Option(False, help="Ignore robots.txt rules"),
    embedding_model: str = typer.Option("text-embedding-3-small", help="Model to use for embeddings"),
    limit: Optional[int] = typer.Option(None, help="Maximum number of pages to crawl (None for unlimited)"),
    disable_crawl_cache: bool = typer.Option(False, help="Disable crawl caching"),
):
    """
    Scrape documentation for a Python package and save as markdown files.

    Usage:
        uv run aic-kb get-package-documentation requests
        uv run aic-kb get-package-documentation requests --version 2.31.0 --depth 2 --strategy dfs --embedding-model text-embedding-3-small --limit 10
    """
    from aic_kb.pypi_doc_scraper.crawl import _get_package_documentation

    # Set embedding model in environment
    os.environ["EMBEDDING_MODEL"] = embedding_model

    asyncio.run(
        _get_package_documentation(
            package_name, version, depth, strategy, ignore_robots, limit, crawl_cache_enabled=not disable_crawl_cache
        )
    )


def build_string(name: str, repeat: int) -> str:
    return name * repeat


def main():
    app()


if __name__ == "__main__":
    main()
