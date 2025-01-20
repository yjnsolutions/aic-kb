import asyncio
import logging
import re
import signal
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from playwright.async_api import BrowserContext, Page
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID

from aic_kb.pypi_doc_scraper.extract import ProcessedChunk, process_chunk


def setup_rich_logging(progress=None):
    """Configure all relevant loggers to use RichHandler"""
    # Get console from progress if provided, otherwise create new one
    if progress:
        console = progress.console
    else:
        console = Console()

    # Create handler
    rich_handler = RichHandler(console=console, show_time=False, show_path=False, markup=True)

    # Configure format
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [rich_handler]
    root_logger.setLevel(logging.INFO)

    # Configure specific loggers
    loggers_to_configure = ["aic_kb.pypi_doc_scraper", "httpx", "LiteLLM"]

    # Set litellm logger to WARNING level
    logging.getLogger("litellm").setLevel(logging.WARNING)

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Remove existing handlers
        logger.addHandler(rich_handler)
        logger.propagate = False


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers

# Global shutdown event
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def process_and_store_document(
    url: str, content: str, progress: Progress, task_id: TaskID
) -> List[ProcessedChunk]:
    """
    Store the scraped content in markdown files and process chunks for embeddings.

    Args:
        url: The URL of the scraped page
        content: The markdown content to store
        progress: Rich progress bar instance
        task_id: Task ID for updating progress
    """
    from .extract import chunk_text

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

    # Save original content to file
    output_path = output_dir / f"{filename}.md"
    output_path.write_text(content)

    # Process chunks
    chunks = chunk_text(content)
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            processed_chunk = await process_chunk(chunk, i, url)
            processed_chunks.append(processed_chunk)
        except Exception as e:
            logger.error(f"Error processing chunk {i} from {url}: {e}")
            continue

    # TODO: Store processed chunks in your vector database or other storage
    return processed_chunks


class CrawlStrategy(Enum):
    BFS = "bfs"
    DFS = "dfs"


class URLTracker:
    def __init__(self):
        self.final_urls: Dict[str, str] = {}

    def set_final_url(self, original_url: str, final_url: str):
        self.final_urls[original_url] = final_url


async def crawl_recursive(
    start_url: str,
    depth: Optional[int],
    strategy: CrawlStrategy,
    robot_parser: Optional[RobotFileParser] = None,
    max_concurrent: int = 5,
) -> Set[str]:
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
        url_tracker = URLTracker()

        crawled_urls = set()
        to_crawl = [(start_url, 0)]  # (url, depth)
        base_domain = urlparse(start_url).netloc
        logger.info(f"Base domain: {base_domain}")

        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"Set concurrency limit to {max_concurrent}")

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            TaskProgressColumn,
            TextColumn,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
        ) as progress:
            setup_rich_logging(progress)

            task_id = progress.add_task("[cyan]Crawling pages...", total=len(to_crawl))

            # Setup signal handlers
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            while to_crawl and not shutdown_event.is_set():
                current_url, current_depth = to_crawl.pop(0) if strategy == CrawlStrategy.BFS else to_crawl.pop()

                if depth is not None and current_depth > depth:
                    logger.info(f"Skipping {current_url}: max depth reached")
                    continue

                if current_url in crawled_urls:
                    logger.info(f"Skipping {current_url}: already crawled")
                    continue

                if robot_parser and not robot_parser.can_fetch("*", current_url):
                    logger.info(f"Skipping {current_url}: blocked by robots.txt")
                    continue

                if urlparse(current_url).netloc != base_domain:
                    logger.info(f"Skipping {current_url}: outside base domain")
                    continue

                async with semaphore:
                    if shutdown_event.is_set():
                        break

                    logger.info(f"Crawling {current_url} at depth {current_depth}")
                    try:

                        async def capture_final_url(page: Page, context: BrowserContext, **kwargs):
                            url_tracker.set_final_url(current_url, page.url)
                            return page

                        crawler.crawler_strategy.set_hook("before_return_html", capture_final_url)

                        result = await crawler.arun(url=current_url, config=crawl_config, session_id="session1")

                        # Check for successful status code (2xx range)
                        if not result.success or (result.status_code and not 200 <= result.status_code < 300):
                            logger.warning(
                                f"Skipping {current_url}: HTTP {result.status_code or 'unknown'} - "
                                f"{result.error_message or 'Unknown error'}"
                            )
                            continue

                        # Get the actual final URL from our tracker
                        actual_final_url = url_tracker.final_urls.get(current_url, current_url)

                        if actual_final_url != current_url:
                            logger.info(f"Redirect detected: {current_url} -> {actual_final_url}")
                        base_url = actual_final_url

                        # Skip if final URL is outside base domain
                        if urlparse(base_url).netloc != base_domain:
                            logger.info(f"Skipping {base_url}: outside base domain after redirect")
                            continue

                        # Only process and store successful responses
                        if result.markdown_v2 and result.markdown_v2.raw_markdown:
                            processed_chunks = await process_and_store_document(
                                base_url, result.markdown_v2.raw_markdown, progress, task_id
                            )
                            progress.update(task_id, advance=1)
                            logger.info(f"Successfully crawled and stored: {base_url} (redirect from {current_url})")
                            logger.info(f"Processed {len(processed_chunks)} chunks from {base_url}")
                            crawled_urls.add(base_url)

                            # Process internal links only for successfully stored pages
                            if result.links and "internal" in result.links:
                                new_urls = 0
                                for link in result.links["internal"]:
                                    href = link.get("href", "")
                                    if not href:
                                        continue

                                    normalized_url = href.replace(current_url, actual_final_url)
                                    if normalized_url not in crawled_urls and normalized_url not in [
                                        url for url, _ in to_crawl
                                    ]:
                                        to_crawl.append((normalized_url, current_depth + 1))
                                        new_urls += 1

                                if new_urls > 0:
                                    progress.update(task_id, total=len(to_crawl))
                                    logger.info(f"Found {new_urls} new internal links on {base_url}")
                        else:
                            logger.warning(f"No markdown content for {base_url}")
                            continue
                    except Exception as e:
                        logger.error(f"Error crawling {current_url}: {str(e)}")
                        continue

            if shutdown_event.is_set():
                logger.info("Shutdown requested, cleaning up...")
                await crawler.close()
                progress.stop()
                logger.info("Cleanup complete")
            else:
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
    setup_rich_logging()
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
    try:
        await crawl_recursive(documentation_url, depth, crawl_strat, robot_parser)
    except Exception as e:
        logger.error(f"Error during documentation crawl: {str(e)}")
        if not shutdown_event.is_set():
            raise
    finally:
        if shutdown_event.is_set():
            logger.info("Package documentation crawl interrupted by user")
