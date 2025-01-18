from typer.testing import CliRunner
from unittest.mock import patch, AsyncMock
from aic_kb.cli import app, build_string

runner = CliRunner()


def test_build_string():
    assert build_string("world", 3) == "worldworldworld"


def test_get_package_documentation():
    # Mock the async function
    with patch('aic_kb.pypi_doc_scraper._get_package_documentation', new_callable=AsyncMock) as mock_get_docs:
        result = runner.invoke(app, ["get-package-documentation", "requests"])
        assert result.exit_code == 0
        # Verify the mock was called with correct arguments
        mock_get_docs.assert_called_once_with("requests", None, None, "bfs", False)
