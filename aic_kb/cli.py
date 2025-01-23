import asyncio
import json
import os
from typing import Optional

import typer

from aic_kb.pypi_doc_scraper.extract import get_embedding
from aic_kb.pypi_doc_scraper.store import create_connection_pool

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


@app.command()
def search(
    text: str,
    match_count: int = typer.Option(5, help="Number of matches to return"),
    connection_string: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/aic_kb", help="PostgreSQL connection string"
    ),
):
    """
    Search the documentation database using semantic similarity.

    Returns the most relevant documentation chunks for the given search text.
    """

    async def _search():
        # Get embedding for search text
        embedding = await get_embedding(text)

        # Connect to database
        conn = await create_connection_pool()
        try:
            # Search for matches using the match_site_pages function
            results = await conn.fetch(
                "SELECT * FROM match_site_pages($1, $2)", json.dumps(embedding, separators=(",", ":")), match_count
            )

            # Print results
            for i, row in enumerate(results, 1):
                typer.echo(f"\n=== Match {i} (Similarity: {row['similarity']:.3f}) ===")
                typer.echo(f"Title: {row['title']}")
                typer.echo(f"URL: {row['url']}")
                typer.echo(f"Summary: {row['summary']}")
                typer.echo(f"Chunk #: {row['chunk_number']}")
                typer.echo(f"\nContent:\n{row['content']}...")
                typer.echo("-" * 80)

        finally:
            await conn.close()

    asyncio.run(_search())


def main():
    app()


if __name__ == "__main__":
    main()
