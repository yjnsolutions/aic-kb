import re
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import urllib.parse
from pathlib import Path
from typing import List, Optional
from rich.progress import Progress, TaskID
from enum import Enum
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

import requests


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        with Progress() as progress:
            # Create the main task
            overall_task = progress.add_task("[cyan]Crawling documentation...", total=len(urls))

            async def process_url(url: str):
                async with semaphore:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        # Pass progress and task_id here
                        await process_and_store_document(url, result.markdown_v2.raw_markdown, progress, overall_task)
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
                    progress.update(overall_task, advance=1)

            # Process all URLs in parallel with limited concurrency
            await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


async def process_and_store_document(url: str, content: str, progress: Progress, task_id: TaskID) -> None:
    """
    Store the scraped content in a markdown file.

    Args:
        url: The URL of the scraped page
        content: The markdown content to store
        progress: Rich progress bar instance
        task_id: Task ID for updating progress
    """
    # Create docs directory if it doesn't exist
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    # Update description while processing
    progress.update(task_id, description=f"[cyan]Processing: {url}")

    # Convert URL to filename
    parsed = urllib.parse.urlparse(url)
    filename = re.sub(r"[^\w\-_]", "_", parsed.path + "_" + parsed.query + "_" + parsed.fragment).strip("_")

    # remove leading and trailing underscores
    filename = filename.strip("_")

    if not filename:
        filename = "index"

    # Save content to file
    output_path = output_dir / f"{filename}.md"
    output_path.write_text(content)

class CrawlStrategy(Enum):
    BFS = "bfs"
    DFS = "dfs"

async def crawl_recursive(
    start_url: str,
    depth: int,
    strategy: CrawlStrategy,
    robot_parser: Optional[RobotFileParser] = None,
    max_concurrent: int = 5
) -> set[str]:
    """
    Recursively crawl a website starting from a URL.

    Args:
        start_url: Starting URL to crawl from
        depth: Maximum recursion depth
        strategy: BFS or DFS crawling strategy
        robot_parser: RobotFileParser instance for robots.txt rules
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Set of successfully crawled URLs
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        crawled_urls = set()
        to_crawl = [(start_url, 0)]  # (url, depth)
        base_domain = urlparse(start_url).netloc

        semaphore = asyncio.Semaphore(max_concurrent)

        while to_crawl:
            current_url, current_depth = to_crawl.pop(0) if strategy == CrawlStrategy.BFS else to_crawl.pop()

            if current_depth > depth:
                continue

            if current_url in crawled_urls:
                continue

            # Check if URL is allowed by robots.txt
            if robot_parser and not robot_parser.can_fetch("*", current_url):
                continue

            # Stay within same domain
            if urlparse(current_url).netloc != base_domain:
                continue

            async with semaphore:
                result = await crawler.arun(
                    url=current_url,
                    config=crawl_config,
                    session_id="session1"
                )

                if result.success:
                    crawled_urls.add(current_url)
                    with Progress() as progress:
                        task_id = progress.add_task(f"Processing {current_url}", total=1)
                        await process_and_store_document(current_url, result.markdown_v2.raw_markdown, progress, task_id)

                    # Extract and normalize new URLs from the page
                    for link in result.links:
                        normalized_url = urljoin(current_url, link)
                        if normalized_url not in crawled_urls:
                            to_crawl.append((normalized_url, current_depth + 1))

        return crawled_urls
    finally:
        await crawler.close()

async def _get_package_documentation(
    package_name: str,
    version: Optional[str] = None,
    depth: int = 3,
    strategy: str = "bfs",
    ignore_robots: bool = False
) -> None:
    """
    Get documentation for a Python package.

    Args:
        package_name: Name of the package
        version: Optional version string
        depth: Maximum recursion depth
        strategy: Crawling strategy ('bfs' or 'dfs')
        ignore_robots: Whether to ignore robots.txt rules
    """
    # Get package info from PyPI
    pypi_url = f"https://pypi.org/pypi/{package_name}/json"
    if version:
        pypi_url = f"https://pypi.org/pypi/{package_name}/{version}/json"

    response = requests.get(pypi_url)
    response.raise_for_status()

    data = response.json()
    project_urls = data["info"].get("project_urls", {})
    documentation_url = project_urls.get("Documentation") or data["info"].get("documentation_url")

    if not documentation_url:
        raise ValueError(f"No documentation URL found for {package_name}")

    # Set up robots.txt parser
    robot_parser = None
    if not ignore_robots:
        robot_parser = RobotFileParser()
        robots_url = urljoin(documentation_url, "/robots.txt")
        robot_parser.set_url(robots_url)
        try:
            robot_parser.read()
        except Exception as e:
            print(f"Warning: Could not read robots.txt: {e}")
            robot_parser = None

    # Run recursive crawler
    crawl_strat = CrawlStrategy.BFS if strategy.lower() == "bfs" else CrawlStrategy.DFS
    await crawl_recursive(documentation_url, depth, crawl_strat, robot_parser)
