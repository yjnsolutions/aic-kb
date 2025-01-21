import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import List

import asyncpg
from rich.progress import Progress, TaskID

from aic_kb.pypi_doc_scraper.extract import ProcessedChunk, process_chunk


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
    url: str, content: str, progress: Progress, task_id: TaskID, connection: asyncpg.Connection, logger: logging.Logger
) -> List[ProcessedChunk]:
    """
    Store the scraped content in markdown files and process chunks for embeddings.
    Args:
        url: The URL of the scraped page
        content: The markdown content to store
        progress: Rich progress bar instance
        task_id: Task ID for updating progress
        connection: asyncpg Connection instance
        :param logger: logger configured to log to Rich console
    """

    from .extract import chunk_text

    # Create docs directory if it doesn't exist
    output_dir = Path("data/docs")
    output_dir.mkdir(exist_ok=True)

    # Update description while processing
    progress.update(task_id, description=f"[cyan]Processing: {url}")

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
    processed_chunks = []

    # Prepare insert statement
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

    for i, chunk in enumerate(chunks):
        try:
            processed_chunk = await process_chunk(chunk, i, url)
            processed_chunks.append(processed_chunk)

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

        except Exception as e:
            logger.error(f"Error processing chunk {i} from {url}: {e}")
            continue

    progress.update(task_id, advance=1)
    return processed_chunks
