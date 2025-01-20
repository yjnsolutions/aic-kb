from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from urllib.robotparser import RobotFileParser

import pytest
import requests
from rich.progress import Progress

from aic_kb.pypi_doc_scraper import (
    CrawlStrategy,
    _get_package_documentation,
    crawl_recursive,
    process_and_store_document,
)


@pytest.mark.asyncio
async def test_process_and_store_document(tmp_path):
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))]
    
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    
    with patch("aic_kb.pypi_doc_scraper.extract.acompletion", return_value=mock_completion_response) as mock_completion, \
         patch("aic_kb.pypi_doc_scraper.extract.aembedding", return_value=mock_embedding_response) as mock_embedding:
        
        url = "https://example.com/docs/page"
        content = "# Test Content"
        with Progress() as progress:
            task_id = progress.add_task("Testing", total=1)
            await process_and_store_document(url, content, progress, task_id)

        # Verify mock calls
        mock_completion.assert_called()
        mock_embedding.assert_called()
        
        # Verify file was created
        output_file = Path("docs/docs_page.md")
        assert output_file.exists()
        assert output_file.read_text() == content


@pytest.mark.asyncio
async def test_get_package_documentation():
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))]
    
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    
    # Mock the PyPI response
    mock_pypi_data = {"info": {"project_urls": {"Documentation": "https://docs.example.com"}}}

    with patch("litellm.acompletion", return_value=mock_completion_response), \
         patch("litellm.aembedding", return_value=mock_embedding_response), \
         patch("requests.get") as mock_get, \
         patch("aic_kb.pypi_doc_scraper.crawl_recursive") as mock_crawl:
        # Configure mock PyPI response
        mock_response = Mock()
        mock_response.json.return_value = mock_pypi_data
        mock_get.return_value = mock_response

        # Configure mock crawl_recursive
        mock_crawl.return_value = {"https://docs.example.com"}

        # Test with invalid package
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        with pytest.raises(requests.exceptions.HTTPError):
            await _get_package_documentation("this-package-definitely-does-not-exist")

        # Reset mock for valid package test
        mock_response.raise_for_status.side_effect = None

        # Test with valid package
        await _get_package_documentation("requests")
        
        # Verify mock calls
        mock_get.assert_called_with("https://pypi.org/pypi/requests/json")
        mock_crawl.assert_called_once_with(
            "https://docs.example.com",
            None,  # depth
            CrawlStrategy.BFS,
            None,  # robot_parser
            limit=None
        )


@pytest.mark.asyncio
async def test_process_and_store_document_special_chars():
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))]
    
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    
    with patch("litellm.acompletion", return_value=mock_completion_response), \
         patch("litellm.aembedding", return_value=mock_embedding_response):
        
        # Test URL with special characters
        url = "https://example.com/docs/page?with=params#fragment"
        content = "# Test Content"
        with Progress() as progress:
            task_id = progress.add_task("Testing", total=1)
            await process_and_store_document(url, content, progress, task_id)

        output_file = Path("docs/docs_page_with_params_fragment.md")
        assert output_file.exists()
        assert output_file.read_text() == content


@pytest.mark.asyncio
async def test_crawl_recursive():
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))]
    
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    
    start_url = "https://example.com/docs"

    # Test unlimited depth
    with patch("litellm.acompletion", return_value=mock_completion_response), \
         patch("litellm.aembedding", return_value=mock_embedding_response), \
         patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=None, strategy=CrawlStrategy.BFS)
        assert start_url in urls
        
        # Add verification of crawler calls
        mock_crawler.assert_called_once()
        mock_instance = mock_crawler.return_value
        mock_instance.arun.assert_called()
        mock_instance.close.assert_called_once()

    # Test BFS strategy
    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.BFS)
        assert start_url in urls

    # Test DFS strategy
    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.status_code = 200
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = {"internal": []}
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.DFS)
        assert start_url in urls


@pytest.mark.asyncio
async def test_robots_txt_handling():
    # Mock litellm responses
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))]
    
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector
    
    start_url = "https://example.com/docs"

    # Test with robots.txt blocking
    robot_parser = RobotFileParser()
    robot_parser.parse(["User-agent: *", "Disallow: /docs"])

    with patch("litellm.acompletion", return_value=mock_completion_response), \
         patch("litellm.aembedding", return_value=mock_embedding_response), \
         patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        # Create mock instance with async methods
        mock_instance = Mock()
        # Add async context manager methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=1, strategy=CrawlStrategy.BFS, robot_parser=robot_parser)
        assert len(urls) == 0  # Should not crawl blocked URL
