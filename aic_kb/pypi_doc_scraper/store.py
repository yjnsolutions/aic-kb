import asyncio
import json
import logging
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


async def create_connection():
    connection = await asyncpg.connect(
        user="postgres",  # Replace with actual credentials
        password="mysecretpassword",  # Replace with actual credentials
        database="postgres",  # Replace with actual database name
        host="localhost",  # Replace with actual host
    )

    # Check if table exists before initializing
    table_exists = await connection.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'site_pages'
        );
        """
    )

    if not table_exists:
        try:
            await ensure_db_initialized(connection)
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    return connection


async def process_and_store_document(
    url: str,
    content: str,
    connection: asyncpg.Connection,
    logger: logging.Logger,
    max_concurrent: int = 5,  # Default to 5 concurrent chunks
) -> List[Optional[ProcessedChunk]]:
    """
    Store the scraped content in markdown files and process chunks for embeddings.
    Args:
        url: The URL of the scraped page
        content: The markdown content to store
        connection: asyncpg Connection instance
        logger: logger configured to log to Rich console
        max_concurrent: Maximum number of concurrent chunk processing tasks
    """
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

    insert_stmt = await connection.prepare(
        """
        INSERT INTO site_pages 
        (url, chunk_number, title, summary, content, metadata, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (url, chunk_number) 
        DO UPDATE SET
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
        RETURNING id
    """
    )

    async def process_and_store_chunk(chunk: str, i: int) -> Optional[ProcessedChunk]:
        try:
            # Use semaphore to limit concurrent processing
            async with sem:
                processed_chunk = await process_chunk(chunk, i, url)

                # Insert into database
                metadata = {
                    "filename": filename,
                    "original_path": str(output_path),
                }

                # Serialize metadata to JSON string
                metadata_json = json.dumps(metadata)

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
