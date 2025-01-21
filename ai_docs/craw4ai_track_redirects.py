import asyncio

from crawl4ai import (AsyncWebCrawler, BrowserConfig, CacheMode,
                      CrawlerRunConfig)
from playwright.async_api import BrowserContext, Page


async def test_url(crawler: AsyncWebCrawler, url: str):
    final_url = None

    async def capture_final_url(page: Page, context: BrowserContext, **kwargs):
        nonlocal final_url
        final_url = await page.evaluate("window.location.href")
        return page

    crawler.crawler_strategy.set_hook("before_return_html", capture_final_url)

    result = await crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS))

    print(f"\nTesting URL: {url}")
    print(f"Initial URL: {url}")
    print(f"Crawl4AI result.url: {result.url}")
    print(f"Actual final URL: {final_url}")
    print(f"Status Code: {result.status_code}")
    print("-" * 80)


async def main():
    browser_config = BrowserConfig(headless=True, ignore_https_errors=True, verbose=True)

    urls_to_test = ["https://docs.pydantic.dev/", "https://crawl4ai.com/", "https://requests.readthedocs.io/"]

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls_to_test:
            await test_url(crawler, url)


if __name__ == "__main__":
    asyncio.run(main())
