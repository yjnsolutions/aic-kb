from dataclasses import dataclass
from typing import List

import asyncpg
import pydantic_core
from openai import BaseModel
from pydantic_ai import Agent, RunContext

from aic_kb.pypi_doc_scraper.extract import get_embedding
from aic_kb.pypi_doc_scraper.store import create_connection_pool


@dataclass
class Deps:
    pool: asyncpg.Pool


class Answer(BaseModel):
    content: str
    reference_urls: List[str]


rag_agent = Agent(model="openai:gpt-4o", result_type=Answer, deps_type=Deps)


class SearchResult(BaseModel):
    title: str
    url: str
    tool_name: str
    source_type: str
    summary: str
    similarity: float
    chunk_number: int
    content: str


@rag_agent.tool
async def retrieve_from_database(context: RunContext[Deps], search_query: str, match_count: int) -> List[SearchResult]:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
        match_count: The number of matches to return.
    """
    embedding = await get_embedding(search_query)
    embedding_json = pydantic_core.to_json(embedding).decode()

    rows = await context.deps.pool.fetch(
        "SELECT title, url, tool_name, source_type, summary, similarity, chunk_number, content FROM match_site_pages($1, $2)",
        embedding_json,
        match_count,
    )

    return [
        SearchResult(
            title=row["title"],
            url=row["url"],
            tool_name=row["tool_name"],
            source_type=row["source_type"],
            summary=row["summary"],
            similarity=row["similarity"],
            chunk_number=row["chunk_number"],
            content=row["content"],
        )
        for row in rows
    ]


async def run_rag_agent(question: str):
    """Run the RAG agent on the given question."""
    print(f"Asking question: {question}")

    pool = await create_connection_pool()
    try:
        deps = Deps(pool=pool)
        run_result = await rag_agent.run(user_prompt=question, deps=deps)
        print(run_result.data.content)
        print("======================")
        for url in run_result.data.reference_urls:
            print(url)
    finally:
        await pool.close()


# async def _search(text: str, match_count: int):
#     console = Console()
#
#     with console.status("[bold green]Getting embedding for search text..."):
#         embedding = await get_embedding(text)
#
#     with console.status("[bold green]Searching database..."):
#         # Connect to database
#         pool = await create_connection_pool()
#
#         async with pool.acquire() as connection:
#             results = await connection.fetch(
#                 "SELECT * FROM match_site_pages($1, $2)", json.dumps(embedding, separators=(",", ":")), match_count
#             )
#
#             # Create results table
#             if not results:
#                 console.print("\n[yellow]No matches found[/yellow]")
#                 return
#
#             console.print("\n[bold blue]Search Results[/bold blue]")
#
#             for i, row in enumerate(results, 1):
#                 # Create panel for each result
#                 content = Table.grid(padding=(0, 1))
#                 content.add_row("[bold cyan]Title:[/bold cyan]", row["title"])
#                 content.add_row("[bold cyan]URL:[/bold cyan]", row["url"])
#                 content.add_row("[bold cyan]Summary:[/bold cyan]", row["summary"])
#                 content.add_row("[bold cyan]Similarity:[/bold cyan]", f"{row['similarity']:.3f}")
#                 content.add_row("[bold cyan]Chunk #:[/bold cyan]", str(row["chunk_number"]))
#                 content.add_row("")  # Empty row as separator
#                 content.add_row("[bold cyan]Content:[/bold cyan]")
#                 content.add_row(row["content"])
#
#                 panel = Panel(content, title=f"[bold]Match {i}[/bold]", border_style="blue")
#                 console.print(panel)
#                 console.print("")  # Add spacing between panels
