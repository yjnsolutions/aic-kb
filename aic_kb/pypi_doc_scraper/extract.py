import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

from litellm import acompletion, aembedding

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


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using litellm."""
    try:
        response = await aembedding(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), input=text)
        # Track cost and tokens
        cost = response._hidden_params.get("response_cost", 0)
        total_tokens = response.usage.total_tokens
        cost_tracker.add_embedding_cost(cost, total_tokens)

        # Log individual call stats
        logger.info(f"Embedding Generation - Tokens: {total_tokens}, " f"Cost: ${cost:.6f}")
        return response.data[0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
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
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks
