import asyncio
import json
import logging
import os
import re
import urllib.parse
from asyncio import Semaphore
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
from asyncpg import Pool, create_pool

from aic_kb.pypi_doc_scraper.extract import ProcessedChunk, process_chunk

from .extract import chunk_text
from .types import Document, SourceType


async def ensure_db_initialized(connection: asyncpg.Connection):
    """Ensure database is initialized with required tables and functions"""
    # Read the init_db.sql file
    init_sql_path = Path(__file__).parent / "init_db.sql"
    init_sql = init_sql_path.read_text()

    # Execute the initialization SQL
    await connection.execute(init_sql)


async def create_connection_pool() -> asyncpg.Pool:
    pool = await create_pool(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        min_size=int(os.environ["POSTGRES_POOL_MIN_SIZE"]),
        max_size=int(os.environ["POSTGRES_POOL_MAX_SIZE"]),
    )

    async with pool.acquire() as connection:
        # Check if table exists before initializing
        table_exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'tool_docs'
            );
            """
        )

        if not table_exists:
            try:
                await ensure_db_initialized(connection)
            except Exception as e:
                logging.error(f"Error initializing database: {e}")
                raise

    return pool


async def store_tool_docs(
    connection_pool: Pool, tool_name: str, source_type: SourceType, root_url: str, metadata: Dict[str, str | None]
) -> int:
    async with connection_pool.acquire() as connection:
        tool_docs_stmt = await connection.prepare(
            """
            INSERT INTO tool_docs (tool_name, source_type, url, updated_at, metadata)
            VALUES ($1, $2, $3, timezone('utc'::text, now()), $4)
            ON CONFLICT (tool_name, source_type, url)
            DO UPDATE SET
                updated_at = timezone('utc'::text, now()),
                metadata = EXCLUDED.metadata
            RETURNING id
            """
        )

        tool_id = await tool_docs_stmt.fetchval(
            tool_name, source_type.value, root_url, json.dumps(metadata)  # Using root_url from document
        )
        return tool_id


async def process_and_store_document(
    tool_id: int,
    document: Document,
    connection_pool: asyncpg.Pool,
    logger: logging.Logger,
    max_concurrent: int = 5,
    cache_enabled: bool = True,
) -> List[Optional[ProcessedChunk]]:
    """
    Process and store document with caching support.
    Called for every document/url crawled.
    """

    # Create docs directory if it doesn't exist
    output_dir = Path("data/docs")
    output_dir.mkdir(exist_ok=True)

    # Convert URL to filename
    parsed = urllib.parse.urlparse(document.url)
    filename = re.sub(r"[^\w\-_]", "_", parsed.path + "_" + parsed.query + "_" + parsed.fragment).strip("_")

    # Remove leading and trailing underscores
    filename = filename.strip("_")
    if not filename:
        filename = "index"

    # Save original content to file
    output_path = output_dir / f"{filename}.md"
    output_path.write_text(document.content)

    # Process chunks
    chunks = chunk_text(document.content)
    # Create semaphore for controlling concurrency for OpenAI API calls
    sem = Semaphore(max_concurrent)
    processed_chunks = await asyncio.gather(
        *[process_chunk(sem, chunk, i, document.url, output_path, cache_enabled) for i, chunk in enumerate(chunks)]
    )

    async def store_chunk(
        connection_pool: asyncpg.Pool, page_id: int, processed_chunk: ProcessedChunk, logger: logging.Logger
    ) -> Optional[ProcessedChunk]:
        try:
            async with connection_pool.acquire() as connection:
                async with connection.transaction():
                    # Create a new prepared statement for each chunk
                    chunk_stmt = await connection.prepare(
                        """
                        INSERT INTO page_chunk 
                        (page_id, chunk_number, title, summary, content, embedding, metadata, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, timezone('utc'::text, now()))
                        ON CONFLICT (page_id, chunk_number)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = timezone('utc'::text, now())
                        RETURNING id
                        """
                    )

                    chunk_id = await chunk_stmt.fetchval(
                        page_id,
                        processed_chunk.chunk_number,
                        processed_chunk.title,
                        processed_chunk.summary,
                        processed_chunk.content,
                        json.dumps(processed_chunk.embedding, separators=(",", ":")),
                        json.dumps(processed_chunk.metadata, separators=(",", ":")),
                    )
                logger.info(
                    f"Stored chunk {processed_chunk.chunk_number} (id={chunk_id}) from {document.url} (page={page_id})"
                )
            return processed_chunk
        except Exception as e:
            logger.error(f"Error storing chunk {processed_chunk.chunk_number} from {document.url}: {e}")
            return None

    async with connection_pool.acquire() as connection:
        # Start a transaction
        async with connection.transaction():
            # upsert page
            page_stmt = await connection.prepare(
                """
                INSERT INTO page (tool_id, url, updated_at)
                VALUES ($1, $2, timezone('utc'::text, now()))
                ON CONFLICT (tool_id, url)
                DO UPDATE SET
                    updated_at = timezone('utc'::text, now())
                RETURNING id
                """
            )

            pid = await page_stmt.fetchval(tool_id, document.url)

        # Create tasks for storing chunks
        tasks = [store_chunk(connection_pool, pid, processed_chunk, logger) for processed_chunk in processed_chunks]
        stored_chunks = await asyncio.gather(*tasks, return_exceptions=False)
        return stored_chunks
