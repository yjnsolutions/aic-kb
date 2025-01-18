import asyncio
import logging
import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from rich.progress import Progress, TaskID

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    # Remove leading and trailing underscores
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
    depth: Optional[int],
    strategy: CrawlStrategy,
    robot_parser: Optional[RobotFileParser] = None,
    max_concurrent: int = 5,
) -> set[str]:
    """
    Recursively crawl a website starting from a URL.

    Args:
        start_url: Starting URL to crawl from
        depth: Maximum recursion depth (None for unlimited)
        strategy: BFS or DFS crawling strategy
        robot_parser: RobotFileParser instance for robots.txt rules
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Set of successfully crawled URLs
    """
    logger.info(
        f"Starting recursive crawl of {start_url} with depth={'unlimited' if depth is None else depth}, strategy={strategy.value}"
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    logger.debug("Initializing web crawler")
    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawled_urls = set()
        to_crawl = [(start_url, 0)]  # (url, depth)
        base_domain = urlparse(start_url).netloc
        logger.info(f"Base domain: {base_domain}")

        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"Set concurrency limit to {max_concurrent}")

        with Progress() as progress:
            # Initialize progress with current known URLs
            task_id = progress.add_task("[cyan]Crawling pages...", total=len(to_crawl))
            
            while to_crawl:
                current_url, current_depth = to_crawl.pop(0) if strategy == CrawlStrategy.BFS else to_crawl.pop()

                if depth is not None and current_depth > depth:
                    logger.info(f"Skipping {current_url}: max depth reached")
                    continue

                if current_url in crawled_urls:
                    logger.info(f"Skipping {current_url}: already crawled")
                    continue

                # Check if URL is allowed by robots.txt
                if robot_parser and not robot_parser.can_fetch("*", current_url):
                    logger.info(f"Skipping {current_url}: blocked by robots.txt")
                    continue

                # Stay within same domain
                if urlparse(current_url).netloc != base_domain:
                    logger.info(f"Skipping {current_url}: outside base domain")
                    continue

                async with semaphore:
                    logger.info(f"Crawling {current_url} at depth {current_depth}")
                    result = await crawler.arun(url=current_url, config=crawl_config, session_id="session1")

                    if result.success:
                        logger.info(f"Successfully crawled: {current_url}")
                        logger.debug(f"Page content: {result.html}")
                        crawled_urls.add(current_url)

                        # Process document
                        await process_and_store_document(
                            current_url, result.markdown_v2.raw_markdown, progress, task_id
                        )

                        # Process internal links
                        if result.links and "internal" in result.links:
                            for link in result.links["internal"]:
                                normalized_url = urljoin(current_url, link.get("href", ""))
                                # Only add if not already crawled and not already in to_crawl
                                if (normalized_url not in crawled_urls and 
                                    normalized_url not in [url for url, _ in to_crawl]):
                                    to_crawl.append((normalized_url, current_depth + 1))
                                    # Update progress total with new URL
                                    progress.update(task_id, total=progress.tasks[task_id].total + 1)

                        logger.info(f"Found {len(result.links.get('internal', []))} internal links on {current_url}")

                    else:
                        logger.error(
                            f"Failed to crawl {current_url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}"
                        )
                    
                    # Update progress
                    progress.update(task_id, advance=1)

        logger.info(f"Crawl completed. Processed {len(crawled_urls)} URLs")
        return crawled_urls


async def _get_package_documentation(
    package_name: str,
    version: Optional[str] = None,
    depth: Optional[int] = None,
    strategy: str = "bfs",
    ignore_robots: bool = False,
) -> None:
    """
    Get documentation for a Python package.

    Args:
        package_name: Name of the package
        version: Optional version string
        depth: Maximum recursion depth (None for unlimited)
        strategy: Crawling strategy ('bfs' or 'dfs')
        ignore_robots: Whether to ignore robots.txt rules
    """
    logger.info(
        f"Getting documentation for package {package_name}"
        + (f" version {version}" if version else "")
        + f" with {'unlimited' if depth is None else depth} depth"
    )

    # Get package info from PyPI
    pypi_url = f"https://pypi.org/pypi/{package_name}/json"
    if version:
        pypi_url = f"https://pypi.org/pypi/{package_name}/{version}/json"

    logger.debug(f"Fetching package info from {pypi_url}")
    response = requests.get(pypi_url)
    response.raise_for_status()

    data = response.json()
    project_urls = data["info"].get("project_urls", {})
    documentation_url = project_urls.get("Documentation") or data["info"].get("documentation_url")

    if not documentation_url:
        logger.error(f"No documentation URL found for {package_name}")
        raise ValueError(f"No documentation URL found for {package_name}")

    logger.info(f"Documentation URL: {documentation_url}")

    # Set up robots.txt parser
    robot_parser = None
    if not ignore_robots:
        robot_parser = RobotFileParser()
        robots_url = urljoin(documentation_url, "/robots.txt")
        logger.debug(f"Checking robots.txt at {robots_url}")
        robot_parser.set_url(robots_url)
        try:
            robot_parser.read()
            logger.info("Successfully read robots.txt")
        except Exception as e:
            logger.warning(f"Could not read robots.txt: {e}")
            robot_parser = None

    # Run recursive crawler
    crawl_strat = CrawlStrategy.BFS if strategy.lower() == "bfs" else CrawlStrategy.DFS
    logger.info(f"Starting crawl with strategy: {crawl_strat.value}")
    await crawl_recursive(documentation_url, depth, crawl_strat, robot_parser)
