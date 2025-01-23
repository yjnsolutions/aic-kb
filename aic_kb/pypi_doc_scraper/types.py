from typing import Dict, List, Optional

from pydantic import BaseModel


class CrawlUrlResult(BaseModel):
    content: Optional[str]
    final_url: str
    links: List[Dict] = []  # Default to empty list if no links
