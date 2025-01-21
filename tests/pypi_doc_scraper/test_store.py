import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from rich.progress import Progress

from aic_kb.pypi_doc_scraper.crawl import process_and_store_document


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing"""
    return logging.getLogger("test_logger")


@pytest.fixture
async def mock_db_connection():
    """Create a mock asyncpg connection for testing"""
    mock_conn = AsyncMock()
    mock_conn.prepare = AsyncMock()
    mock_conn.prepare.return_value.fetchval = AsyncMock(return_value=1)  # Return dummy ID
    return mock_conn


@pytest.mark.asyncio
async def test_process_and_store_document(mock_db_connection, mock_logger, tmp_path):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]
    mock_completion_response._hidden_params = {"response_cost": 0.0}
    mock_completion_response.usage = AsyncMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    mock_embedding_response._hidden_params = {"response_cost": 0.0}
    mock_embedding_response.usage = AsyncMock(total_tokens=10)

    with (
        patch("aic_kb.pypi_doc_scraper.extract.acompletion", return_value=mock_completion_response) as mock_completion,
        patch("aic_kb.pypi_doc_scraper.extract.aembedding", return_value=mock_embedding_response) as mock_embedding,
    ):
        url = "https://example.com/docs/page"
        content = "# Test Content"
        with Progress() as progress:
            task_id = progress.add_task("Testing", total=1)
            await process_and_store_document(url, content, progress, task_id, mock_db_connection, mock_logger)

        # Verify mock calls
        mock_completion.assert_called()
        mock_embedding.assert_called()

        # Verify file was created
        output_file = Path("data/docs/docs_page.md")
        assert output_file.exists()
        assert output_file.read_text() == content


@pytest.mark.asyncio
async def test_process_and_store_document_special_chars(mock_db_connection, mock_logger):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    with (
        patch("aic_kb.pypi_doc_scraper.extract.acompletion", return_value=mock_completion_response) as mock_completion,
        patch("aic_kb.pypi_doc_scraper.extract.aembedding", return_value=mock_embedding_response) as mock_embedding,
    ):
        # Test URL with special characters
        url = "https://example.com/docs/page?with=params#fragment"
        content = "# Test Content"
        with Progress() as progress:
            task_id = progress.add_task("Testing", total=1)
            await process_and_store_document(url, content, progress, task_id, mock_db_connection, mock_logger)

        # Verify mock calls
        mock_completion.assert_called()
        mock_embedding.assert_called()

        output_file = Path("data/docs/docs_page_with_params_fragment.md")
        assert output_file.exists()
        assert output_file.read_text() == content
