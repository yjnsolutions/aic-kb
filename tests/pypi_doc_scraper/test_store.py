import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aic_kb.pypi_doc_scraper.crawl import process_and_store_document


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing"""
    return logging.getLogger("test_logger")


@pytest.mark.asyncio
async def test_process_and_store_document(mock_db_connection_pool, mock_logger, tmp_path):
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
        # Pass cache_enabled=False to disable caching during test
        await process_and_store_document(url, content, mock_db_connection_pool, mock_logger, cache_enabled=False)

        # Verify mock calls
        mock_completion.assert_called()
        mock_embedding.assert_called()

        # Verify file was created
        output_file = Path("data/docs/docs_page.md")
        assert output_file.exists()
        assert output_file.read_text() == content


@pytest.mark.asyncio
async def test_process_and_store_document_special_chars(mock_db_connection_pool, mock_logger):
    # Mock litellm responses
    mock_completion_response = Mock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = Mock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    with (
        patch("aic_kb.pypi_doc_scraper.extract.acompletion", return_value=mock_completion_response) as mock_completion,
        patch("aic_kb.pypi_doc_scraper.extract.aembedding", return_value=mock_embedding_response) as mock_embedding,
    ):
        # Test URL with special characters
        url = "https://example.com/docs/page?with=params#fragment"
        content = "# Test Content"
        # Pass cache_enabled=False to disable caching during test
        await process_and_store_document(url, content, mock_db_connection_pool, mock_logger, cache_enabled=False)

        # Verify mock calls
        mock_completion.assert_called()
        mock_embedding.assert_called()

        output_file = Path("data/docs/docs_page_with_params_fragment.md")
        assert output_file.exists()
        assert output_file.read_text() == content
