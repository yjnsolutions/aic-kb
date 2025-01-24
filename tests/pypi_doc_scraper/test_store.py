import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch  # Removed 'patch.dict'

import pytest

from aic_kb.pypi_doc_scraper.crawl import process_and_store_document
from aic_kb.pypi_doc_scraper.store import create_connection_pool  # Ensure this import is present


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


@pytest.mark.asyncio
async def test_database_initialization_when_table_does_not_exist():
    mock_conn = AsyncMock()
    mock_conn.prepare = AsyncMock()
    mock_conn.prepare.return_value.fetchval = AsyncMock(return_value=1)  # Return dummy ID
    mock_conn.fetchval = AsyncMock(return_value=True)  # Mock table existence check
    mock_conn.execute = AsyncMock()  # Mock execute method
    mock_conn.close = AsyncMock()  # Mock close method

    # Create a mock pool
    mock_pool = AsyncMock()
    mock_pool.connection = mock_conn
    mock_pool.acquire = Mock()
    # Make acquire() return the mock connection as a context manager
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    # Create a mock for create_pool that returns an awaitable
    mock_create_pool = AsyncMock()
    mock_create_pool.return_value = mock_pool

    # Patch create_pool with our awaitable mock
    with patch("aic_kb.pypi_doc_scraper.store.create_pool", mock_create_pool):
        # Call the function under test
        pool = await create_connection_pool()
        
        # Verify the returned pool is our mock
        assert pool == mock_pool
        mock_create_pool.assert_called_once()
        # Fix: Assert on mock_pool.acquire instead of mock_create_pool.acquire
        mock_pool.acquire.assert_called_once()
        
        # Verify that fetchval was called to check if table exists
        mock_conn.fetchval.assert_called_once()
