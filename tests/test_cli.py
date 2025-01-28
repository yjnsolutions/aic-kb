from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from aic_kb.search.types import (
    Answer,
    KnowledgeBaseSearchResult,
    StackOverflowSearchResult,
    ToolStats,
)

runner = CliRunner()


def test_get_package_documentation(monkeypatch):
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
        from aic_kb.cli import app

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


def test_search_command_with_results(mock_db_connection_pool, monkeypatch):
    """Test search command when results are found"""
    # Mock the tool stats
    mock_tool_stats = [
        ToolStats(tool_name="test-tool", source_type="readthedocs", url="http://test.com", page_count=10)
    ]

    # Mock the knowledge base results
    mock_kb_results = [
        KnowledgeBaseSearchResult(
            title="Test Title",
            source_url="http://test.com",
            tool_name="test-tool",
            source_type="readthedocs",
            summary="Test Summary",
            similarity=0.95,
            chunk_number=1,
            content="Test Content",
        )
    ]

    # Mock Stack Overflow result
    mock_so_result = StackOverflowSearchResult(
        question="Test Question", accepted_answer="Test Answer", source_url="https://stackoverflow.com/q/123"
    )

    # Mock agent response
    mock_answer = Answer(
        content="Here's what I found...", reference_urls=["http://test.com", "https://stackoverflow.com/q/123"]
    )

    with (
        patch("aic_kb.search.search.load_tool_names", AsyncMock(return_value=mock_tool_stats)),
        patch("aic_kb.search.search.retrieve_from_knowledge_base", AsyncMock(return_value=mock_kb_results)),
        patch("aic_kb.search.search.retrieve_from_stackoverflow", AsyncMock(return_value=mock_so_result)),
        patch("aic_kb.search.search.rag_agent.run", AsyncMock(return_value=AsyncMock(data=mock_answer))),
        patch("aic_kb.search.search.create_connection_pool", return_value=mock_db_connection_pool),
    ):
        from aic_kb.cli import app

        result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0
        assert "Here's what I found..." in result.stdout


def test_search_command_no_results(mock_db_connection_pool, monkeypatch):
    """Test search command when no results are found"""
    # Mock the tool stats
    mock_tool_stats = [
        ToolStats(tool_name="test-tool", source_type="readthedocs", url="http://test.com", page_count=10)
    ]

    # Mock empty knowledge base results
    mock_kb_results = []

    # Mock no Stack Overflow result
    mock_so_result = None

    # Mock agent response for no results
    mock_answer = Answer(content="I couldn't find any relevant information.", reference_urls=[])

    with (
        patch("aic_kb.search.search.load_tool_names", AsyncMock(return_value=mock_tool_stats)),
        patch("aic_kb.search.search.retrieve_from_knowledge_base", AsyncMock(return_value=mock_kb_results)),
        patch("aic_kb.search.search.retrieve_from_stackoverflow", AsyncMock(return_value=mock_so_result)),
        patch("aic_kb.search.search.rag_agent.run", AsyncMock(return_value=AsyncMock(data=mock_answer))),
        patch("aic_kb.search.search.create_connection_pool", return_value=mock_db_connection_pool),
        # patch("aic_kb.search.search.rag_agent", AsyncMock(return_value=mock_answer)),
    ):
        from aic_kb.cli import app

        result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0
        assert "I couldn't find any relevant information." in result.stdout
