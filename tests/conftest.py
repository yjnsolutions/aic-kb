from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
async def mock_db_connection_pool():
    """Create a mock asyncpg connection pool for testing"""
    mock_conn = AsyncMock()
    mock_conn.prepare = AsyncMock()
    mock_conn.prepare.return_value.fetchval = AsyncMock(return_value=1)  # Return dummy ID
    mock_conn.fetchval = AsyncMock(return_value=True)  # Mock table existence check
    mock_conn.execute = AsyncMock()  # Mock execute method
    mock_conn.close = AsyncMock()  # Mock close method

    # Create mock pool
    mock_pool = AsyncMock()
    mock_pool.acquire = Mock()
    # Make acquire() return the mock connection as a context manager
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    return mock_pool
