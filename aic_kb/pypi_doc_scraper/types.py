from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TitleAndSummary(BaseModel):
    title: str
    summary: str


class CrawlUrlResult(BaseModel):
    content: Optional[str]
    final_url: str
    links: List[Dict] = []  # Default to empty list if no links


class ProcessedChunk(BaseModel):
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
