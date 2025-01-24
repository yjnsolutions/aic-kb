import hashlib
import logging
import os
from unittest.mock import AsyncMock, Mock, patch
from urllib.robotparser import RobotFileParser

import pytest
import requests
from crawl4ai import CrawlerRunConfig
from pydantic import BaseModel

from aic_kb.pypi_doc_scraper.crawl import (
    CrawlStrategy,
    _get_package_documentation,
    crawl_recursive,
    crawl_url,
)
from aic_kb.pypi_doc_scraper.store import process_and_store_document


@pytest.fixture
async def mock_db_connection_pool():
    """Create a mock asyncpg connection pool for testing"""
    mock_conn = AsyncMock()
    mock_conn.prepare = AsyncMock()
    mock_conn.prepare.return_value.fetchval = AsyncMock(return_value=1)  # Return dummy ID
    mock_conn.fetchval = AsyncMock(return_value=True)  # Mock table existence check
    mock_conn.execute = AsyncMock()  # Mock execute method
    mock_conn.close = AsyncMock()  # Mock close method

    # Create mock pool
    mock_pool = AsyncMock()
    mock_pool.acquire = AsyncMock()
    # Make acquire() return the mock connection as a context manager
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    return mock_pool


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing"""
    return logging.getLogger("test_logger")


@pytest.mark.asyncio
async def test_get_package_documentation(mock_db_connection_pool, mock_logger):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    # Mock the PyPI response
    mock_pypi_data = {"info": {"project_urls": {"Documentation": "https://docs.example.com"}}}

    with (
        patch("aic_kb.pypi_doc_scraper.extract.acompletion", return_value=mock_completion_response) as mock_completion,
        patch("aic_kb.pypi_doc_scraper.extract.aembedding", return_value=mock_embedding_response) as mock_embedding,
        patch("requests.get") as mock_get,
        patch("aic_kb.pypi_doc_scraper.crawl.crawl_recursive") as mock_crawl,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        # Configure mock PyPI response
        mock_response = Mock()
        mock_response.json.return_value = mock_pypi_data
        mock_get.return_value = mock_response

        # Configure mock crawl_recursive to simulate content processing
        async def mock_crawl_with_content(*args, **kwargs):
            # Simulate processing content by calling process_and_store_document
            await process_and_store_document(
                "https://docs.example.com", "# Test Content", mock_db_connection_pool, mock_logger
            )
            return {"https://docs.example.com"}

        mock_crawl.side_effect = mock_crawl_with_content

        # Test with invalid package
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        with pytest.raises(requests.exceptions.HTTPError):
            await _get_package_documentation("this-package-definitely-does-not-exist")

        # Reset mock for valid package test
        mock_response.raise_for_status.side_effect = None

        # Test with valid package
        await _get_package_documentation("requests", caching_enabled=False)

        # Verify mock calls
        mock_get.assert_called_with("https://pypi.org/pypi/requests/json")
        mock_crawl.assert_called_once_with(
            "https://docs.example.com", None, CrawlStrategy.BFS, None, limit=None, caching_enabled=False
        )
        mock_completion.assert_called()
        mock_embedding.assert_called()


@pytest.mark.asyncio
async def test_crawl_recursive(mock_db_connection_pool):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    start_url = "https://example.com/docs"

    # Test unlimited depth
    with (
        patch("aic_kb.pypi_doc_scraper.crawl.AsyncWebCrawler") as mock_crawler,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
        patch(
            "aic_kb.pypi_doc_scraper.crawl.process_and_store_document", return_value=["chunk1"]
        ) as mock_process_and_store_document,
    ):
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)

        async def mock_aexit(*args):
            await mock_instance.close()
            return None

        mock_instance.__aexit__ = AsyncMock(side_effect=mock_aexit)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2 = type("Markdown", (object,), {"raw_markdown": "# Test"})()
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=None, strategy=CrawlStrategy.BFS, caching_enabled=False)
        assert start_url in urls

        # Add verification of crawler calls
        mock_crawler.assert_called_once()
        mock_instance = mock_crawler.return_value
        mock_instance.arun.assert_called()
        mock_instance.close.assert_called_once()

        # Verify mock calls
        mock_process_and_store_document.assert_called()

    # Test BFS strategy
    with (
        patch("aic_kb.pypi_doc_scraper.crawl.AsyncWebCrawler") as mock_crawler,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
        patch(
            "aic_kb.pypi_doc_scraper.crawl.process_and_store_document", return_value=["chunk1"]
        ) as mock_process_and_store_document,
    ):
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)

        async def mock_aexit(*args):
            await mock_instance.close()
            return None

        mock_instance.__aexit__ = AsyncMock(side_effect=mock_aexit)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2 = type("Markdown", (object,), {"raw_markdown": "# Test"})()
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.BFS, caching_enabled=False)
        assert start_url in urls

        # Verify mock calls
        mock_process_and_store_document.assert_called()

    # Test DFS strategy
    with (
        patch("aic_kb.pypi_doc_scraper.crawl.AsyncWebCrawler") as mock_crawler,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
        patch(
            "aic_kb.pypi_doc_scraper.crawl.process_and_store_document", return_value=["chunk1"]
        ) as mock_process_and_store_document,
    ):
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)

        async def mock_aexit(*args):
            await mock_instance.close()
            return None

        mock_instance.__aexit__ = AsyncMock(side_effect=mock_aexit)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2 = type("Markdown", (object,), {"raw_markdown": "# Test"})()
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.DFS, caching_enabled=False)
        assert start_url in urls

        # Verify mock calls
        mock_process_and_store_document.assert_called()


@pytest.mark.asyncio
async def test_robots_txt_handling(mock_db_connection_pool):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    # Test with robots.txt blocking and allowing
    robot_parser = RobotFileParser()
    robot_parser.parse(["User-agent: *", "Disallow: /blocked", "Allow: /docs"])

    with (
        patch("aic_kb.pypi_doc_scraper.crawl.AsyncWebCrawler") as mock_crawler,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
        patch(
            "aic_kb.pypi_doc_scraper.crawl.process_and_store_document", return_value=["chunk1"]
        ) as mock_process_and_store_document,
    ):
        # Create mock instance with async methods
        mock_instance = Mock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)

        async def mock_aexit(*args):
            await mock_instance.close()
            return None

        mock_instance.__aexit__ = AsyncMock(side_effect=mock_aexit)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2 = type("Markdown", (object,), {"raw_markdown": "# Test Content"})()
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        # Test blocked URL
        blocked_url = "https://example.com/blocked"
        urls = await crawl_recursive(
            blocked_url, depth=1, strategy=CrawlStrategy.BFS, robot_parser=robot_parser, caching_enabled=False
        )
        assert len(urls) == 0  # Should not crawl blocked URL

        # Test allowed URL
        allowed_url = "https://example.com/docs"
        urls = await crawl_recursive(
            allowed_url, depth=1, strategy=CrawlStrategy.BFS, robot_parser=robot_parser, caching_enabled=False
        )
        assert len(urls) == 1  # Should crawl allowed URL
        assert mock_process_and_store_document.call_count > 0  # Should have called completion for content processing


@pytest.mark.asyncio
async def test_crawl_url_caching(mock_db_connection_pool):
    """Test the caching functionality of crawl_url"""
    url = "https://example.com/test"
    cache_dir = ".crawl_cache"
    cache_file = os.path.join(cache_dir, hashlib.sha256(url.encode()).hexdigest() + ".json")

    # Clean up any existing cache
    if os.path.exists(cache_file):
        os.remove(cache_file)

    with (
        patch("aic_kb.pypi_doc_scraper.crawl.AsyncWebCrawler") as mock_crawler,
        patch("aic_kb.pypi_doc_scraper.crawl.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        # Setup mock crawler
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock()
        mock_instance.arun = AsyncMock()

        # Create a proper mock response class that inherits from BaseModel
        class MockMarkdownV2(BaseModel):
            raw_markdown: str = "test content"
            markdown_with_citations: str = "test content with citations"
            references_markdown: str = "test references"

        class MockResponse(BaseModel):
            success: bool = True
            status_code: int = 200
            markdown_v2: MockMarkdownV2 = MockMarkdownV2()
            links: dict = {"internal": []}
            error_message: None = None
            url: str = "https://example.com/test"  # Add required url field
            html: str = "<html>test content</html>"  # Add required html field

        def model_dump(self, mode="json"):
            """Implement model_dump method"""
            return {
                "success": self.success,
                "status_code": self.status_code,
                "markdown_v2": self.markdown_v2.model_dump(mode=mode),
                "links": self.links,
                "error_message": self.error_message,
                "url": self.url,
                "html": self.html,
            }

        mock_response = MockResponse()
        mock_instance.arun.return_value = mock_response
        mock_instance.crawler_strategy = Mock()
        mock_crawler.return_value = mock_instance

        # First request - should create cache
        os.makedirs(cache_dir, exist_ok=True)
        config = CrawlerRunConfig()
        await crawl_url(mock_instance, config, url, cache_enabled=True)

        # Verify cache was created
        assert os.path.exists(cache_file)

        # Second request - should use cache
        await crawl_url(mock_instance, config, url, cache_enabled=True)

        # Verify crawler was only called once
        assert mock_instance.arun.call_count == 1

        # Cleanup
        if os.path.exists(cache_file):
            os.remove(cache_file)
