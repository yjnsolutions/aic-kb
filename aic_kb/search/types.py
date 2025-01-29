from dataclasses import dataclass
from typing import List

import asyncpg
from openai import BaseModel


class ToolStats(BaseModel):
    tool_name: str
    source_type: str
    url: str
    page_count: int


@dataclass
class Deps:
    pool: asyncpg.Pool
    tool_stats: List[ToolStats]


class Answer(BaseModel):
    content: str
    reference_urls: List[str]


class KnowledgeBaseSearchResult(BaseModel):
    title: str
    source_url: str
    tool_name: str
    source_type: str
    summary: str
    similarity: float
    chunk_number: int
    content: str


class StackOverflowSearchResult(BaseModel):
    question: str
    accepted_answer: str
    source_url: str
