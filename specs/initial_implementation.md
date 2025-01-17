# Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Create a CLI app that takes in a python package name and version, and scrapes official documentation to generate markdown files

## Mid-Level Objective

- Build a python Typer cli application
- Accept a "package-name" argument and an optional "version" argument
- Use Pypi to get URL of official documentation
- Use sitemap.xml in the documentation website to get all the pages
- Scrape the documentation pages to get the content

## Implementation Notes
- No need to import any external libraries, see pyproject.toml for the libraries you can use
- Comment every function and class
- For typer commands add usage examples starting with `uv run get-package-documentation <params>`
- Create unit tests using pytest for every function and class. Test corner cases and edge cases, as well as normal cases
- Carefully review each low level task for exact code changes

## Context

### Beginning context
- aic_kb/cli.py
- tests/test_cli.py

### Ending context  
- aic_kb/cli.py
- aic_kb/pypi_doc_scraper/__init__.py (new file)
- tests/test_cli.py
- tests/pypi_doc_scraper/test_init.py (new file)

## Low-Level Tasks
> Ordered from start to finish

1. Create our scraping function
```python
# CREATE aic_kb/pypi_doc_scraper/__init__.py: Use code block below no changes.

import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


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
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

```

2. Create our CLI command
```aider
UPDATE aic_kb/pypi_doc_scraper/__init__.py:
    CREATE _get_package_documentation(package_name: str, version: str = None) -> None:
        USE Pypi API to get the URL of the official documentation
        USE the sitemap.xml in the documentation website to get all the pages
        Return a list of URLs to scrape
        USE crawl_parallel to scrape the documentation pages
    CREATE def process_and_store_document(url: str, content: str) -> None:
        Store the content in a markdown file
        Use the URL to create a file name
        Save the content in the file
```

3. Use the new functions in the CLI
```aider
UPDATE aic_kb/cli.py:
    ADD a new typer command:
        @app.command()
        def get_package_documentation(package_name: str, version: str = None):
            _get_package_documentation(package_name, version)
```
