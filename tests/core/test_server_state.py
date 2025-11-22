"""Tests for ServerState class."""

import time
import pytest
from src.core.server_state import ServerState


class TestServerState:
    """Test ServerState class functionality."""
    
    def test_init(self):
        """Test ServerState initialization."""
        state = ServerState()
        
        assert state.total_requests == 0
        assert state.active_connections == 0
        assert state.mcp_initialized is True
        assert isinstance(state.start_time, float)
        assert state.start_time > 0
    
    def test_get_uptime_seconds(self):
        """Test uptime calculation."""
        state = ServerState()
        
        # Wait a small amount of time
        time.sleep(0.1)
        
        uptime = state.get_uptime_seconds()
        assert uptime >= 0.1
        assert uptime < 1.0  # Should be less than 1 second
    
    def test_increment_requests(self):
        """Test request counter increment."""
        state = ServerState()
        
        assert state.total_requests == 0
        
        state.increment_requests()
        assert state.total_requests == 1
        
        state.increment_requests()
        assert state.total_requests == 2
        
        # Increment many times
        for _ in range(10):
            state.increment_requests()
        assert state.total_requests == 12
    
    def test_increment_connections(self):
        """Test connection counter increment."""
        state = ServerState()
        
        assert state.active_connections == 0
        
        state.increment_connections()
        assert state.active_connections == 1
        
        state.increment_connections()
        assert state.active_connections == 2
    
    def test_decrement_connections(self):
        """Test connection counter decrement."""
        state = ServerState()
        
        # Increment a few times
        state.increment_connections()
        state.increment_connections()
        state.increment_connections()
        assert state.active_connections == 3
        
        # Decrement
        state.decrement_connections()
        assert state.active_connections == 2
        
        state.decrement_connections()
        assert state.active_connections == 1
        
        state.decrement_connections()
        assert state.active_connections == 0
    
    def test_decrement_connections_at_zero(self):
        """Test that decrement doesn't go below zero."""
        state = ServerState()
        
        assert state.active_connections == 0
        
        # Try to decrement when already at zero
        state.decrement_connections()
        assert state.active_connections == 0  # Should stay at 0
        
        state.decrement_connections()
        assert state.active_connections == 0  # Should stay at 0
    
    def test_multiple_operations(self):
        """Test multiple operations together."""
        state = ServerState()
        
        # Simulate some activity
        state.increment_requests()
        state.increment_connections()
        state.increment_requests()
        state.increment_connections()
        state.increment_requests()
        
        assert state.total_requests == 3
        assert state.active_connections == 2
        
        state.decrement_connections()
        assert state.active_connections == 1
        
        uptime = state.get_uptime_seconds()
        assert uptime >= 0
