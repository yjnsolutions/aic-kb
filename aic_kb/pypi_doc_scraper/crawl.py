import asyncio
import hashlib
import json
import logging
import os
import signal
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from playwright.async_api import BrowserContext, Page
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)

from aic_kb.pypi_doc_scraper.store import (
    create_connection_pool,
    process_and_store_document,
)


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


class CrawlStrategy(Enum):
    BFS = "bfs"
    DFS = "dfs"


class URLTracker:
    def __init__(self):
        self.final_urls: Dict[str, str] = {}

    def set_final_url(self, original_url: str, final_url: str):
        self.final_urls[original_url] = final_url


class CachedResult:
    def __init__(self, data):
        self.success = data["success"]
        self.status_code = data["status_code"]
        self.markdown_v2 = (
            type("Markdown", (), {"raw_markdown": data["markdown_v2"]["raw_markdown"]})()
            if data["markdown_v2"]
            else None
        )
        self.links = data["links"]
        self.error_message = data["error_message"]


async def crawl_url(
    crawler: AsyncWebCrawler, crawl_config: CrawlerRunConfig, url: str, cache_enabled: bool
) -> Tuple[Any, str]:
    final_url = None
    cache_dir = ".crawl_cache"
    cache_file = os.path.join(cache_dir, hashlib.sha256(url.encode()).hexdigest() + ".json")

    # Check cache if enabled
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)
                    logger.info(f"Crawl cache HIT for {url}")
                    return CachedResult(cached_data["result"]), cached_data["final_url"]
            except Exception as e:
                logger.warning(f"Cache read error for {url}: {e}")

    logger.info(f"Crawl cache MISS for {url}")

    async def capture_final_url(page: Page, context: BrowserContext, **kwargs):
        nonlocal final_url
        final_url = page.url
        return page

    crawler.crawler_strategy.set_hook("before_return_html", capture_final_url)

    result = await crawler.arun(url=url, config=crawl_config, session_id="session1")

    if final_url is None:
        final_url = url

    # Store in cache if enabled
    if cache_enabled:
        try:
            # Create a simplified version of the result for caching
            cacheable_result = {
                "success": result.success if hasattr(result, "success") else True,
                "status_code": result.status_code if hasattr(result, "status_code") else 200,
                "markdown_v2": {"raw_markdown": result.markdown_v2.raw_markdown if result.markdown_v2 else None},
                "links": result.links if hasattr(result, "links") else None,
                "error_message": result.error_message if hasattr(result, "error_message") else None,
            }

            with open(cache_file, "w") as f:
                json.dump({"result": cacheable_result, "final_url": final_url}, f)
        except Exception as e:
            logger.warning(f"Cache write error for {url}: {e}")

    return result, final_url


async def crawl_recursive(
    start_url: str,
    depth: Optional[int],
    strategy: CrawlStrategy,
    robot_parser: Optional[RobotFileParser] = None,
    max_concurrent: int = 15,
    limit: Optional[int] = None,
    crawl_cache_enabled: bool = True,
) -> Set[str]:
    """
    Recursively crawl a website starting from a URL.

    Args:
        start_url: Starting URL to crawl from
        depth: Maximum recursion depth (None for unlimited)
        strategy: BFS or DFS crawling strategy
        robot_parser: RobotFileParser instance for robots.txt rules
        max_concurrent: Maximum number of concurrent requests
        limit: Maximum number of pages to crawl (None for unlimited)

    Returns:
        Set of successfully crawled URLs
    """
    # Create database connection pool
    connection_pool = await create_connection_pool()

    try:
        logger.info(
            f"Starting recursive crawl of {start_url} with "
            f"depth={'unlimited' if depth is None else depth}, "
            f"strategy={strategy.value}, "
            f"limit={'unlimited' if limit is None else limit}"
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
            crawled_urls: Set[str] = set()
            to_crawl = [(start_url, 0)]  # (url, depth)
            base_domain = urlparse(start_url).netloc
            logger.info(f"Base domain: {base_domain}")

            semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Set concurrency limit to {max_concurrent}")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
            ) as progress:
                setup_rich_logging(progress)

                task_id = progress.add_task("[cyan]Crawling pages...", total=1, completed=0)

                # Setup signal handlers
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                while to_crawl and not shutdown_event.is_set():
                    # Add limit check
                    if limit is not None and len(crawled_urls) >= limit:
                        logger.info(f"Reached crawl limit of {limit} pages")
                        break

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

                    # Only use semaphore for the crawl operation
                    result = None
                    actual_final_url = None
                    async with semaphore:
                        if shutdown_event.is_set():
                            break

                        logger.info(f"Crawling {current_url} at depth {current_depth}")
                        try:
                            result, actual_final_url = await crawl_url(
                                crawler, crawl_config, current_url, crawl_cache_enabled
                            )
                        except Exception as e:
                            logger.error(f"Error crawling {current_url}: {str(e)}")
                            continue

                    # Process results outside of semaphore
                    if result:
                        if actual_final_url != current_url:
                            logger.info(f"Redirect detected: {current_url} -> {actual_final_url}")
                        base_url = actual_final_url

                        # Check for successful status code (2xx range)
                        if not result.success or (result.status_code and not 200 <= result.status_code < 300):
                            logger.warning(
                                f"Skipping {current_url}: HTTP {result.status_code or 'unknown'} - "
                                f"{result.error_message or 'Unknown error'}"
                            )
                            continue

                        # Skip if final URL is outside base domain
                        if urlparse(base_url).netloc != base_domain:
                            logger.info(f"Skipping {base_url}: outside base domain after redirect")
                            continue

                        # Only process and store successful responses
                        if result.markdown_v2 and result.markdown_v2.raw_markdown:
                            # Process document without semaphore
                            store_task = asyncio.create_task(
                                process_and_store_document(
                                    base_url, result.markdown_v2.raw_markdown, connection_pool, logger
                                )
                            )

                            logger.info(f"Successfully crawled: {base_url} (redirect from {current_url})")
                            crawled_urls.add(base_url)

                            # Process internal links only for successfully stored pages
                            if result.links and "internal" in result.links:
                                new_urls = 0

                                async def process_link(link):
                                    nonlocal new_urls
                                    href = link.get("href", "")
                                    if not href:
                                        return

                                    # Remove anchor fragments and normalize URL
                                    normalized_url = href.split("#")[0].replace(current_url, actual_final_url)
                                    if normalized_url not in crawled_urls and normalized_url not in [
                                        url for url, _ in to_crawl
                                    ]:
                                        to_crawl.append((normalized_url, current_depth + 1))
                                        new_urls += 1

                                # Process links in parallel batches of 50
                                batch_size = 50
                                for i in range(0, len(result.links["internal"]), batch_size):
                                    batch = result.links["internal"][i : i + batch_size]
                                    await asyncio.gather(*[process_link(link) for link in batch])
                                if new_urls > 0:
                                    progress.update(task_id, total=len(crawled_urls) + len(to_crawl))
                                    logger.info(f"Found {new_urls} new internal links on {base_url}")

                            # Wait for document processing to complete
                            processed_chunks = await store_task
                            logger.info(f"Processed {len(processed_chunks)} chunks from {base_url}")
                            progress.update(task_id, advance=1)
                        else:
                            logger.warning(f"No markdown content for {base_url}")
                            continue

                if shutdown_event.is_set():
                    logger.info("Shutdown requested, cleaning up...")
                    progress.stop()
                    logger.info("Cleanup complete")
                else:
                    logger.info(f"Crawl completed. Processed {len(crawled_urls)} URLs")

                # Log final cost summary
                from .extract import log_cost_summary

                log_cost_summary()

                return crawled_urls

    finally:
        # Close the connection when done
        await connection_pool.close()


async def _get_package_documentation(
    package_name: str,
    version: Optional[str] = None,
    depth: Optional[int] = None,
    strategy: str = "bfs",
    ignore_robots: bool = False,
    limit: Optional[int] = None,
    crawl_cache_enabled: bool = True,
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
        await crawl_recursive(
            documentation_url, depth, crawl_strat, robot_parser, limit=limit, crawl_cache_enabled=crawl_cache_enabled
        )
    except Exception as e:
        logger.error(f"Error during documentation crawl: {str(e)}")
        if not shutdown_event.is_set():
            raise
    finally:
        if shutdown_event.is_set():
            logger.info("Package documentation crawl interrupted by user")
