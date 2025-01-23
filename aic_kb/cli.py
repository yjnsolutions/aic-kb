import asyncio
import json
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aic_kb.pypi_doc_scraper.extract import get_embedding
from aic_kb.pypi_doc_scraper.store import create_connection_pool

load_dotenv()
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
        console = Console()
        
        with console.status("[bold green]Getting embedding for search text..."):
            embedding = await get_embedding(text)

        with console.status("[bold green]Searching database..."):
            # Connect to database
            conn = await create_connection_pool()
            try:
                results = await conn.fetch(
                    "SELECT * FROM match_site_pages($1, $2)", 
                    json.dumps(embedding, separators=(",", ":")), 
                    match_count
                )

                # Create results table
                if not results:
                    console.print("\n[yellow]No matches found[/yellow]")
                    return

                console.print("\n[bold blue]Search Results[/bold blue]")
                
                for i, row in enumerate(results, 1):
                    # Create panel for each result
                    content = Table.grid(padding=(0, 1))
                    content.add_row("[bold cyan]Title:[/bold cyan]", row['title'])
                    content.add_row("[bold cyan]URL:[/bold cyan]", row['url'])
                    content.add_row("[bold cyan]Summary:[/bold cyan]", row['summary'])
                    content.add_row(
                        "[bold cyan]Similarity:[/bold cyan]", 
                        f"{row['similarity']:.3f}"
                    )
                    content.add_row("[bold cyan]Chunk #:[/bold cyan]", str(row['chunk_number']))
                    content.add_row("")  # Empty row as separator
                    content.add_row("[bold cyan]Content:[/bold cyan]")
                    content.add_row(row['content'])

                    panel = Panel(
                        content,
                        title=f"[bold]Match {i}[/bold]",
                        border_style="blue"
                    )
                    console.print(panel)
                    console.print("")  # Add spacing between panels
            finally:
                await conn.close()

    asyncio.run(_search())


def main():
    app()


if __name__ == "__main__":
    main()
