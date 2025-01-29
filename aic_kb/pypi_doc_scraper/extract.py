import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from anyio import Semaphore
from litellm import acompletion, aembedding

from aic_kb.pypi_doc_scraper.types import ProcessedChunk, TitleAndSummary

logger = logging.getLogger(__name__)


# Add global cost tracking
class CostTracker:
    def __init__(self):
        self.completion_cost = 0.0
        self.embedding_cost = 0.0
        self.completion_tokens = 0
        self.embedding_tokens = 0

    def add_completion_cost(self, cost: float, tokens: int):
        self.completion_cost += cost
        self.completion_tokens += tokens

    def add_embedding_cost(self, cost: float, tokens: int):
        self.embedding_cost += cost
        self.embedding_tokens += tokens

    @property
    def total_cost(self) -> float:
        return self.completion_cost + self.embedding_cost

    def get_summary(self) -> str:
        return (
            f"=== Cost and Token Usage Summary ===\n"
            f"Completion Cost: ${self.completion_cost:.6f}\n"
            f"Embedding Cost: ${self.embedding_cost:.6f}\n"
            f"Total Cost: ${self.total_cost:.6f}\n"
            f"\n"
            f"Completion Tokens: {self.completion_tokens}\n"
            f"Embedding Tokens: {self.embedding_tokens}\n"
            f"Total Tokens: {self.completion_tokens + self.embedding_tokens}"
        )


# Create global cost tracker instance
cost_tracker = CostTracker()


async def get_title_and_summary(chunk: str, url: str, cache_enabled: bool = True) -> TitleAndSummary:
    """Extract title and summary using GPT-4."""
    cache_dir = ".title_and_summary_cache"
    cache_file = os.path.join(cache_dir, hashlib.sha256(f"{url}:{chunk}".encode()).hexdigest() + ".json")

    # Check cache if enabled
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached_result = json.load(f)
                    logger.info(f"Title/Summary cache HIT for {url}")
                    return TitleAndSummary(**cached_result)
            except Exception as e:
                logger.warning(f"Cache read error for title/summary {url}: {e}")

    logger.info(f"Title/Summary cache MISS for {url}")
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await acompletion(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."},
            ],
            response_format={"type": "json_object"},
        )
        # Track cost and tokens
        cost = response._hidden_params.get("response_cost", 0)
        total_tokens = response.usage.total_tokens
        cost_tracker.add_completion_cost(cost, total_tokens)

        # Log individual call stats
        logger.info(
            f"Title/Summary Generation - Tokens: {total_tokens} "
            f"(Prompt: {response.usage.prompt_tokens}, "
            f"Completion: {response.usage.completion_tokens}), "
            f"Cost: ${cost:.6f}"
        )
        result = json.loads(response.choices[0].message.content)
        title_and_summary = TitleAndSummary(**result)

        # Store in cache if enabled
        if cache_enabled:
            try:
                with open(cache_file, "w") as f:
                    json.dump(title_and_summary.model_dump(), f)
            except Exception as e:
                logger.warning(f"Cache write error for title/summary {url}: {e}")

        return title_and_summary
    except Exception as e:
        logger.warning(f"Error getting title and summary: {e}")
        return TitleAndSummary(title="Error processing title", summary="Error processing summary")


async def get_embedding(text: str, cache_enabled: bool = True) -> List[float]:
    """Get embedding vector using litellm."""
    cache_dir = ".embedding_cache"
    cache_file = os.path.join(cache_dir, hashlib.sha256(text.encode()).hexdigest() + ".json")

    # Check cache if enabled
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached_embedding = json.load(f)
                    logger.info("Embedding cache HIT")
                    return cached_embedding
            except Exception as e:
                logger.warning(f"Cache read error for embedding: {e}")

    logger.info("Embedding cache MISS")
    try:
        response = await aembedding(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), input=text)
        # Track cost and tokens
        cost = response._hidden_params.get("response_cost", 0)
        total_tokens = response.usage.total_tokens
        cost_tracker.add_embedding_cost(cost, total_tokens)

        # Log individual call stats
        logger.info(f"Embedding Generation - Tokens: {total_tokens}, " f"Cost: ${cost:.6f}")
        embedding = response.data[0]["embedding"]

        # Store in cache if enabled
        if cache_enabled:
            try:
                with open(cache_file, "w") as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Cache write error for embedding: {e}")

        return embedding
    except Exception as e:
        logger.warning(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(
    semaphore: Semaphore, chunk: str, chunk_number: int, url: str, output_path: Path, cache_enabled: bool = True
) -> ProcessedChunk:
    """Process a single chunk of text."""

    async with semaphore:
        # Get title and summary
        extracted = await get_title_and_summary(chunk, url, cache_enabled)

        # Get embedding
        embedding = await get_embedding(chunk, cache_enabled)

    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
        "file_path": str(output_path),
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted.title,
        summary=extracted.summary,
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding,
    )


# Add function to log final cost summary
def log_cost_summary():
    """Log the final cost and token usage summary."""
    logger.info(cost_tracker.get_summary())


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = min(start + chunk_size, text_length)

        # Extract the initial chunk
        chunk = text[start:end]

        # Count the number of code block markers in the chunk
        code_block_markers = chunk.count("```")

        # If there's an unclosed code block (odd count of markers)
        if code_block_markers % 2 != 0:
            # Find the position of the next closing code block marker after the current end
            next_code_block_end = text.find("```", end)

            if next_code_block_end != -1:
                # Adjust the end position to include the closing code block marker
                end = next_code_block_end + 3  # +3 to include the "```"
                chunk = text[start:end]
            else:
                # If no closing marker is found, include the rest of the text
                end = text_length
                chunk = text[start:end]
        else:
            # Try to find a paragraph break
            if "\n\n" in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind("\n\n")
                if last_break > chunk_size * 0.3:
                    end = start + last_break
                    chunk = text[start:end]
            # If no paragraph break, try to break at a sentence
            elif ". " in chunk:
                # Find the last sentence break
                last_period = chunk.rfind(". ")
                if last_period > chunk_size * 0.3:
                    end = start + last_period + 1
                    chunk = text[start:end]

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position to the end of the current chunk
        start = end

    return chunks
