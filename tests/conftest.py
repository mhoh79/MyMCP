"""
Pytest configuration and shared fixtures for MCP HTTP client tests.
"""

import os
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "auth: marks tests that require authentication"
    )
    config.addinivalue_line(
        "markers", "ratelimit: marks tests that require rate limiting"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add auth marker to authentication tests
        if "auth" in item.nodeid:
            item.add_marker(pytest.mark.auth)
        
        # Add ratelimit marker to rate limiting tests
        if "rate_limit" in item.nodeid:
            item.add_marker(pytest.mark.ratelimit)
        
        # Add integration marker to concurrent and performance tests
        if "concurrent" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that may take longer
        if any(keyword in item.nodeid for keyword in ["exceeding_limit", "large_response"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def server_url():
    """Get math server URL from environment or use default."""
    return os.getenv("MATH_SERVER_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def stats_server_url():
    """Get stats server URL from environment or use default."""
    return os.getenv("STATS_SERVER_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def api_key():
    """Get API key from environment or use default."""
    return os.getenv("MCP_API_KEY", "test-api-key-12345678")
