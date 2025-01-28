from typing import List

import aiohttp
import asyncpg
import pydantic_core
from pydantic_ai import Agent, RunContext

from aic_kb.pypi_doc_scraper.extract import get_embedding
from aic_kb.pypi_doc_scraper.store import create_connection_pool
from aic_kb.search.types import (
    Answer,
    Deps,
    KnowledgeBaseSearchResult,
    StackOverflowSearchResult,
    ToolStats,
)

AGENT_MODEL = "openai:gpt-4o"

rag_agent = Agent(
    model=AGENT_MODEL,
    system_prompt="""
        You are a bot that helps automated AI coding systems to find documentation for Python packages/tools. 
        As much as possible, give code examples extracted from the tool results, along with links to the source documentation.
        If you don't find the answer, tell us, be honest. 
    """,
    result_type=Answer,
    deps_type=Deps,
)


async def load_tool_names(pool: asyncpg.Pool) -> List[ToolStats]:
    """Load the names of all tools in the database."""
    async with pool.acquire() as connection:
        result = await connection.fetch(
            """
            select tool_name, source_type, td.url, count(*) as page_count 
            from tool_docs td left outer join page p on p.tool_id = td.id 
            group by tool_name, source_type, td.url
            order by page_count desc
        """
        )
        return [
            ToolStats(
                tool_name=row["tool_name"],
                source_type=row["source_type"],
                url=row["url"],
                page_count=row["page_count"],
            )
            for row in result
        ]


@rag_agent.system_prompt
async def add_available_tool_names(context: RunContext[Deps]) -> str:
    """Add the names of all tools in the database to the context."""
    tool_names = [
        f"{tool.tool_name} ({tool.source_type}, {tool.page_count} pages from {tool.url})"
        for tool in context.deps.tool_stats
    ]
    return f"Available tools: {', '.join(tool_names)}"


@rag_agent.tool
async def retrieve_from_stackoverflow(context: RunContext[Deps], search_query: str) -> StackOverflowSearchResult | None:
    """Get an answer from Stack Overflow."""

    base_url = "https://api.stackexchange.com/2.3"

    # Common parameters for all requests
    common_params = {
        "site": "stackoverflow",
    }

    async with aiohttp.ClientSession() as session:
        # Step 1: Search for questions with accepted answers
        search_params = {
            **common_params,
            "q": search_query,
            "sort": "relevance",
            "order": "desc",
            "pagesize": "1",
            "accepted": "true",
        }

        async with session.get(f"{base_url}/search/advanced", params=search_params) as resp:
            search_data = await resp.json()

            if not search_data.get("items"):
                return None

            first_result = search_data["items"][0]
            question_id = first_result["question_id"]
            answer_id = first_result["accepted_answer_id"]

        # Step 2: Get the question details
        question_params = {**common_params, "filter": "withbody"}  # Include the question body

        async with session.get(f"{base_url}/questions/{question_id}", params=question_params) as resp:
            question_data = await resp.json()
            question_details = question_data["items"][0]

        # Step 3: Get the accepted answer
        answer_params = {**common_params, "filter": "withbody"}  # Include the answer body

        async with session.get(f"{base_url}/answers/{answer_id}", params=answer_params) as resp:
            answer_data = await resp.json()
            answer_details = answer_data["items"][0]

        return StackOverflowSearchResult(
            question=question_details["title"],
            accepted_answer=answer_details["body"],
            source_url=f"https://stackoverflow.com/q/{question_id}",
        )


@rag_agent.tool
# TODO add tool name here to filter results
async def retrieve_from_knowledge_base(
    context: RunContext[Deps], search_query: str, match_count: int
) -> List[KnowledgeBaseSearchResult]:
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
        KnowledgeBaseSearchResult(
            title=row["title"],
            source_url=row["url"],
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
        deps = Deps(pool=pool, tool_stats=await load_tool_names(pool))
        run_result = await rag_agent.run(user_prompt=question, deps=deps)
        print(run_result.data.content)
        print("=" * 80)
        for url in run_result.data.reference_urls:
            print(url)
        print("=" * 80)
        print(run_result.all_messages())
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
