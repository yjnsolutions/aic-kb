# Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Do not rely on sitemaps to fetch pages, but recursively crawl websites

## Mid-Level Objective

- Add a "recursion depth" option to the `get-package-documentation` command
- Add a "crawl-strategy" option to the `get-package-documentation` command (BFS or DFS)
- Keep track of the URLs that have been crawled to avoid duplicates
- Normalize URLs to prevent crawling the same content with different URLs
- Stay within the intended domain/scope
- Handle relative vs absolute URLs properly 
- Respect robots.txt rules by default
- Add a "ignore-robots-txt" option to the `get-package-documentation` command to ignore robots.txt rules

## Implementation Notes
- No need to import any external libraries, see pyproject.toml for the libraries you can use
- Comment every function and class
- Use crawl4ai to crawl websites, do NOT use requests
- Create unit tests using pytest for every function and class. Test corner cases and edge cases, as well as normal cases
- Carefully review each low level task for exact code changes

## Context

### Beginning context
- aic_kb/cli.py
- aic_kb/pypi_doc_scraper/__init__.py
- tests/test_cli.py
- tests/pypi_doc_scraper/test_init.py

### Ending context  
- aic_kb/cli.py
- aic_kb/pypi_doc_scraper/__init__.py (new file)
- tests/test_cli.py
- tests/pypi_doc_scraper/test_init.py (new file)

## Low-Level Tasks
> Ordered from start to finish

1. Remove sitemap.xml logic
```aider
UPDATE _get_package_documentation(package_name: str, version: str = None) -> None:
    REMOVE the sitemap.xml logic
```

2. Read robots.txt
```aider
UPDATE _get_package_documentation(package_name: str, version: str = None) -> None:
    READ the robots.txt file and save the rules
```

3. Implement recursive crawling
```aider
UPDATE `__init__.py`:
    ADD a new function:
        def crawl_recursive(...):
            Crawl the URL recursively to the specified depth using the specified strategy
            Keep track/update crawled URLs
            Use the robots.txt rules
```

