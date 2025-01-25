from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from aic_kb.cli import app

runner = CliRunner()


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
        patch(
            "aic_kb.pypi_doc_scraper.crawl._get_package_documentation",
            new_callable=AsyncMock,
        ) as mock_get_docs,
    ):
        result = runner.invoke(app, ["get-package-documentation", "requests"])
        assert result.exit_code == 0

        # Verify the mock was called with correct arguments
        mock_get_docs.assert_called_once_with("requests", None, None, "bfs", False, None, caching_enabled=True)

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
                "--limit",
                "10",
                "--disable-caching",
            ],
        )
        assert result.exit_code == 0

        # Verify the second call with all parameters
        mock_get_docs.assert_called_with("requests", "2.31.0", 2, "dfs", True, 10, caching_enabled=False)


def test_search_command_with_results(mock_db_connection_pool):
    # Mock embedding response
    mock_embedding = [0.1] * 1536  # Mock embedding vector

    # Create mock search results
    mock_results = [
        {
            "title": "Test Title",
            "url": "http://test.com",
            "summary": "Test Summary",
            "similarity": 0.95,
            "chunk_number": 1,
            "content": "Test Content",
        }
    ]

    # Create async mock for get_embedding
    async_mock_get_embedding = AsyncMock(return_value=mock_embedding)

    with (
        patch("aic_kb.search.search.get_embedding", async_mock_get_embedding),
        patch("aic_kb.search.search.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0

        # Verify the embedding was requested
        async_mock_get_embedding.assert_called_once()
        # Verify database query was made
        mock_db_connection_pool.acquire.assert_called_once()


def test_search_command_no_results(mock_db_connection_pool):
    # Mock embedding response
    mock_embedding = [0.1] * 1536  # Mock embedding vector

    # Create async mock for get_embedding
    async_mock_get_embedding = AsyncMock(return_value=mock_embedding)

    with (
        patch("aic_kb.search.search.get_embedding", async_mock_get_embedding),
        patch("aic_kb.search.search.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0

        # Verify the embedding was requested
        async_mock_get_embedding.assert_called_once()
        # Verify database query was made
        mock_db_connection_pool.acquire.assert_called_once()


def test_search_command_fix_mocking(mock_db_connection_pool):
    # Mock embedding response
    mock_embedding = [0.1] * 1536  # Mock embedding vector

    # Create async mock for get_embedding
    async_mock_get_embedding = AsyncMock(return_value=mock_embedding)

    with (
        patch("aic_kb.search.search.get_embedding", async_mock_get_embedding),
        patch("aic_kb.search.search.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0

        # Verify the embedding was requested
        async_mock_get_embedding.assert_called_once()
        # Verify database query was made
        mock_db_connection_pool.acquire.assert_called_once()
