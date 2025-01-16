from pathlib import Path

import pytest
import requests
from rich.progress import Progress

from aic_kb.pypi_doc_scraper import (
    _get_package_documentation,
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
