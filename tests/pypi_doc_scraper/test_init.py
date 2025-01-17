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
    # Test normal case
    url = "https://example.com/docs/page"
    content = "# Test Content"
    with Progress() as progress:
        task_id = progress.add_task("Testing", total=1)
        await process_and_store_document(url, content, progress, task_id)

    output_file = Path("docs/docs_page.md")
    assert output_file.exists()
    assert output_file.read_text() == content


@pytest.mark.asyncio
async def test_get_package_documentation():
    # Test with invalid package
    with pytest.raises(requests.exceptions.HTTPError):
        await _get_package_documentation("this-package-definitely-does-not-exist")

    # Test with valid package
    await _get_package_documentation("requests")
    assert Path("docs").exists()
    assert len(list(Path("docs").glob("*.md"))) > 0


@pytest.mark.asyncio
async def test_process_and_store_document_special_chars():
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
    start_url = "https://example.com/docs"

    # Test unlimited depth
    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        mock_instance = Mock()
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = []
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=None, strategy=CrawlStrategy.BFS)
        assert start_url in urls

    # Test BFS strategy
    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        # Create mock instance
        mock_instance = Mock()
        # Make start() return a coroutine
        mock_instance.start.return_value = None
        mock_instance.start = AsyncMock()
        # Make arun() return a coroutine
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = []
        # Make close() return a coroutine
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.BFS)
        assert start_url in urls

    # Test DFS strategy
    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.arun.return_value.success = True
        mock_instance.arun.return_value.markdown_v2.raw_markdown = "# Test"
        mock_instance.arun.return_value.html = "<html><body>Test content</body></html>"
        mock_instance.arun.return_value.links = []
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=2, strategy=CrawlStrategy.DFS)
        assert start_url in urls


@pytest.mark.asyncio
async def test_robots_txt_handling():
    start_url = "https://example.com/docs"

    # Test with robots.txt blocking
    robot_parser = RobotFileParser()
    robot_parser.parse(["User-agent: *", "Disallow: /docs"])

    with patch("aic_kb.pypi_doc_scraper.AsyncWebCrawler") as mock_crawler:
        # Create mock instance with async methods
        mock_instance = Mock()
        mock_instance.start = AsyncMock()
        mock_instance.arun = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_crawler.return_value = mock_instance

        urls = await crawl_recursive(start_url, depth=1, strategy=CrawlStrategy.BFS, robot_parser=robot_parser)
        assert len(urls) == 0  # Should not crawl blocked URL
