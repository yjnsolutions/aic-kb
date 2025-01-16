import re
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import urllib.parse
from pathlib import Path
from typing import List, Optional
from rich.progress import Progress, TaskID

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


async def _get_package_documentation(package_name: str, version: Optional[str] = None) -> None:
    """
    Get documentation for a Python package.

    Args:
        package_name: Name of the package
        version: Optional version string
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

    # Try to find sitemap
    sitemap_url = f"{documentation_url.rstrip('/')}/sitemap.xml"
    response = requests.get(sitemap_url)

    urls_to_crawl = []
    if response.status_code == 200:
        # Parse sitemap XML
        from xml.etree import ElementTree

        root = ElementTree.fromstring(response.content)

        # Extract URLs from sitemap
        urls_to_crawl = [
            url.text
            for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if url.text is not None
        ]
    else:
        # Fallback to just the main documentation URL
        urls_to_crawl = [documentation_url]

    # Run the crawler
    await crawl_parallel(urls_to_crawl)
