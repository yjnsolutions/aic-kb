from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from aic_kb.cli import app, build_string

runner = CliRunner()


def test_build_string():
    assert build_string("world", 3) == "worldworldworld"


def test_get_package_documentation():
    # Mock litellm calls
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(message=AsyncMock(content='{"title": "Test Title", "summary": "Test Summary"}'))
    ]

    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [[0.1] * 1536]  # Mock embedding vector

    # Mock both litellm functions and the doc scraper
    with (
        patch("litellm.acompletion", return_value=mock_completion_response),
        patch("litellm.aembedding", return_value=mock_embedding_response),
        patch("aic_kb.pypi_doc_scraper._get_package_documentation", new_callable=AsyncMock) as mock_get_docs,
    ):

        result = runner.invoke(app, ["get-package-documentation", "requests"])
        assert result.exit_code == 0

        # Verify the mock was called with correct arguments
        mock_get_docs.assert_called_once_with("requests", None, None, "bfs", False, None)

        # Test with all optional parameters
        result = runner.invoke(
            app,
            [
                "get-package-documentation",
                "requests",
                "--version",
                "2.31.0",
                "--depth",
                "2",
                "--strategy",
                "dfs",
                "--ignore-robots",
                "--embedding-model",
                "custom-model",
                "--limit",
                "10",
            ],
        )
        assert result.exit_code == 0

        # Verify the second call with all parameters
        mock_get_docs.assert_called_with("requests", "2.31.0", 2, "dfs", True, 10)
