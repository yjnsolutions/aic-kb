import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aic_kb.pypi_doc_scraper.extract import get_embedding
from aic_kb.pypi_doc_scraper.store import create_connection_pool


async def _search(text: str, match_count: int):
    console = Console()

    with console.status("[bold green]Getting embedding for search text..."):
        embedding = await get_embedding(text)

    with console.status("[bold green]Searching database..."):
        # Connect to database
        pool = await create_connection_pool()

        async with pool.acquire() as connection:
            results = await connection.fetch(
                "SELECT * FROM match_site_pages($1, $2)", json.dumps(embedding, separators=(",", ":")), match_count
            )

            # Create results table
            if not results:
                console.print("\n[yellow]No matches found[/yellow]")
                return

            console.print("\n[bold blue]Search Results[/bold blue]")

            for i, row in enumerate(results, 1):
                # Create panel for each result
                content = Table.grid(padding=(0, 1))
                content.add_row("[bold cyan]Title:[/bold cyan]", row["title"])
                content.add_row("[bold cyan]URL:[/bold cyan]", row["url"])
                content.add_row("[bold cyan]Summary:[/bold cyan]", row["summary"])
                content.add_row("[bold cyan]Similarity:[/bold cyan]", f"{row['similarity']:.3f}")
                content.add_row("[bold cyan]Chunk #:[/bold cyan]", str(row["chunk_number"]))
                content.add_row("")  # Empty row as separator
                content.add_row("[bold cyan]Content:[/bold cyan]")
                content.add_row(row["content"])

                panel = Panel(content, title=f"[bold]Match {i}[/bold]", border_style="blue")
                console.print(panel)
                console.print("")  # Add spacing between panels
