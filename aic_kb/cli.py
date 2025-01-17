import asyncio
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"Hello {build_string(name, 1)}!")


@app.command()
def get_package_documentation(
    package_name: str, 
    version: Optional[str] = None,
    depth: int = typer.Option(3, help="Maximum recursion depth for crawling"),
    strategy: str = typer.Option("bfs", help="Crawling strategy: 'bfs' or 'dfs'"),
    ignore_robots: bool = typer.Option(False, help="Ignore robots.txt rules")
):
    """
    Scrape documentation for a Python package and save as markdown files.

    Usage:
        uv run aic-kb get-package-documentation requests
        uv run aic-kb get-package-documentation requests --version 2.31.0 --depth 2 --strategy dfs
    """
    from aic_kb.pypi_doc_scraper import _get_package_documentation

    asyncio.run(_get_package_documentation(package_name, version, depth, strategy, ignore_robots))


def build_string(name: str, repeat: int) -> str:
    return name * repeat


def main():
    app()


if __name__ == "__main__":
    main()
