import asyncio
import json
import logging
import os
import re
import urllib.parse
from asyncio import Semaphore
from pathlib import Path
from typing import List, Optional

import asyncpg

from aic_kb.pypi_doc_scraper.extract import ProcessedChunk, process_chunk

from .extract import chunk_text


async def ensure_db_initialized(connection: asyncpg.Connection):
    """Ensure database is initialized with required tables and functions"""
    # Read the init_db.sql file
    init_sql_path = Path(__file__).parent / "init_db.sql"
    init_sql = init_sql_path.read_text()

    # Execute the initialization SQL
    await connection.execute(init_sql)


async def create_connection_pool() -> asyncpg.Pool:
    pool = await asyncpg.create_pool(
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
                AND table_name = 'openai_site_pages'
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


async def process_and_store_document(
    url: str,
    content: str,
    connection_pool: asyncpg.Pool,
    logger: logging.Logger,
    max_concurrent: int = 5,
    cache_enabled: bool = True,
) -> List[Optional[ProcessedChunk]]:
    """Process and store document with caching support"""
    # Create semaphore for controlling concurrency
    sem = Semaphore(max_concurrent)

    # Create docs directory if it doesn't exist
    output_dir = Path("data/docs")
    output_dir.mkdir(exist_ok=True)

    # Convert URL to filename
    parsed = urllib.parse.urlparse(url)
    filename = re.sub(r"[^\w\-_]", "_", parsed.path + "_" + parsed.query + "_" + parsed.fragment).strip("_")

    # Remove leading and trailing underscores
    filename = filename.strip("_")
    if not filename:
        filename = "index"

    # Save original content to file
    output_path = output_dir / f"{filename}.md"
    output_path.write_text(content)

    # Process chunks
    chunks = chunk_text(content)

    async def process_and_store_chunk(chunk: str, i: int) -> Optional[ProcessedChunk]:
        try:
            async with sem:
                processed_chunk = await process_chunk(chunk, i, url, cache_enabled)

                # Insert into database
                metadata = {
                    "filename": filename,
                    "original_path": str(output_path),
                }

                # Serialize metadata to JSON string
                metadata_json = json.dumps(metadata)

                async with connection_pool.acquire() as connection:
                    insert_stmt = await connection.prepare(
                        """
                        INSERT INTO openai_site_pages 
                        (url, chunk_number, title, summary, content, metadata, embedding, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, timezone('utc'::text, now()))
                        ON CONFLICT (url, chunk_number) 
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = timezone('utc'::text, now())
                        RETURNING id
                    """
                    )

                    await insert_stmt.fetchval(
                        url,
                        processed_chunk.chunk_number,
                        processed_chunk.title,
                        processed_chunk.summary,
                        processed_chunk.content,
                        metadata_json,
                        json.dumps(processed_chunk.embedding, separators=(",", ":")),
                    )

                return processed_chunk
        except Exception as e:
            logger.error(f"Error processing chunk {i} from {url}: {e}")
            return None

    # Process all chunks with controlled concurrency
    tasks = [process_and_store_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks, return_exceptions=False)

    return processed_chunks
