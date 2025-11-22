"""
Security middleware for MCP servers.

This module provides authentication, rate limiting, and logging middleware
for HTTP transport in MCP servers. All middleware can be enabled/disabled
via configuration.

Features:
- Authentication via Bearer token (configurable)
- Rate limiting using token bucket algorithm per client IP (configurable)
- Request logging for debugging
- CORS support with configurable origins
"""

import logging
import time
from typing import Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import Config

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Handles X-Forwarded-For header for proxied requests.
    This is a shared utility used by multiple middleware classes.
    
    Args:
        request: HTTP request
        
    Returns:
        Client IP address as string
    """
    # Check for X-Forwarded-For header (common with proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.
    
    Each client IP gets its own bucket with tokens that refill at a constant rate.
    When a request comes in, it consumes a token. If no tokens are available,
    the request is rate limited.
    
    Attributes:
        capacity: Maximum number of tokens in the bucket
        refill_rate: Number of tokens added per second
        tokens: Current number of tokens in the bucket
        last_refill: Timestamp of last token refill
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize a token bucket.
        
        Args:
            capacity: Maximum number of tokens (burst capacity)
            refill_rate: Tokens added per second (sustained rate)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume (default: 1)
            
        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication via Bearer token.
    
    Checks for 'Authorization: Bearer <api_key>' header and validates
    against the configured API key. Returns 401 if authentication fails.
    
    Can be disabled via configuration (config.authentication.enabled).
    """
    
    def __init__(self, app: FastAPI, config: Config):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            config: Configuration object with authentication settings
        """
        super().__init__(app)
        self.config = config
        self.enabled = config.authentication.enabled
        self.api_key = config.authentication.api_key
        
        if self.enabled:
            logger.info("Authentication middleware enabled")
        else:
            logger.info("Authentication middleware disabled")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication check.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response from next handler or 401 error response
        """
        # Skip authentication if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip authentication for health/metrics endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            logger.warning(f"Request to {request.url.path} missing Authorization header")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Missing Authorization header. Expected: Authorization: Bearer <api_key>"
                }
            )
        
        # Check for Bearer token format
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(f"Request to {request.url.path} has invalid Authorization format")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid Authorization format. Expected: Authorization: Bearer <api_key>"
                }
            )
        
        # Validate API key
        provided_key = parts[1]
        if provided_key != self.api_key:
            logger.warning(f"Request to {request.url.path} with invalid API key")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid API key"
                }
            )
        
        # Authentication successful
        logger.debug(f"Request to {request.url.path} authenticated successfully")
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting using token bucket algorithm.
    
    Tracks requests per client IP address and enforces a configurable
    rate limit (requests per minute). Uses token bucket algorithm for
    smooth rate limiting with burst capacity.
    
    Can be disabled via configuration (config.rate_limiting.enabled).
    """
    
    def __init__(self, app: FastAPI, config: Config):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            config: Configuration object with rate limiting settings
        """
        super().__init__(app)
        self.config = config
        self.enabled = config.rate_limiting.enabled
        self.requests_per_minute = config.rate_limiting.requests_per_minute
        
        # Token bucket for each client IP (with cleanup of old entries)
        self.buckets: Dict[str, TokenBucket] = {}
        self.max_buckets = 10000  # Limit to prevent memory leaks
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Clean up every hour
        
        # Calculate bucket parameters
        # Capacity = requests per minute (allows bursts)
        # Refill rate = requests per second (sustained rate)
        self.bucket_capacity = self.requests_per_minute
        self.refill_rate = self.requests_per_minute / 60.0
        
        if self.enabled:
            logger.info(f"Rate limiting enabled: {self.requests_per_minute} requests/minute")
        else:
            logger.info("Rate limiting disabled")
    

    def cleanup_old_buckets(self) -> None:
        """
        Clean up inactive buckets to prevent memory leaks.
        
        Removes buckets that haven't been accessed recently or if we exceed
        the maximum bucket count.
        """
        now = time.time()
        
        # Only cleanup periodically
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        
        # If we're over the limit, remove oldest buckets
        if len(self.buckets) > self.max_buckets:
            # Sort by last refill time and keep only the most recent
            sorted_buckets = sorted(
                self.buckets.items(),
                key=lambda x: x[1].last_refill,
                reverse=True
            )
            self.buckets = dict(sorted_buckets[:self.max_buckets // 2])
            logger.info(f"Cleaned up old rate limit buckets, now tracking {len(self.buckets)} IPs")
    
    def get_bucket(self, client_ip: str) -> TokenBucket:
        """
        Get or create token bucket for client IP.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            TokenBucket instance for this client
        """
        # Periodic cleanup
        self.cleanup_old_buckets()
        
        if client_ip not in self.buckets:
            self.buckets[client_ip] = TokenBucket(
                capacity=self.bucket_capacity,
                refill_rate=self.refill_rate
            )
        
        return self.buckets[client_ip]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through rate limiting check.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response from next handler or 429 error response
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health/metrics endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Get client IP and token bucket
        client_ip = get_client_ip(request)
        bucket = self.get_bucket(client_ip)
        
        # Try to consume a token
        if not bucket.consume():
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for {client_ip} on {request.url.path}")
            
            # Calculate retry-after time
            # Time to accumulate one token (in seconds)
            retry_after = int(1.0 / self.refill_rate) if self.refill_rate > 0 else 60
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Rate limit check passed
        remaining = int(bucket.tokens)
        logger.debug(f"Rate limit OK for {client_ip}, {remaining} tokens remaining")
        
        # Process request and add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request method, path, client IP, status code, and response time.
    Useful for debugging and monitoring.
    """
    
    def __init__(self, app: FastAPI):
        """
        Initialize request logging middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        logger.info("Request logging middleware enabled")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response from next handler
        """
        # Get client IP using shared utility
        client_ip = get_client_ip(request)
        
        # Record start time
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            duration = time.time() - start_time
            
            # Log request details
            logger.info(
                f"{request.method} {request.url.path} "
                f"from {client_ip} "
                f"-> {response.status_code} "
                f"({duration:.3f}s)"
            )
            
            return response
            
        except Exception as e:
            # Log errors
            duration = time.time() - start_time
            logger.error(
                f"{request.method} {request.url.path} "
                f"from {client_ip} "
                f"-> ERROR: {str(e)} "
                f"({duration:.3f}s)",
                exc_info=True
            )
            raise


def setup_middleware(app: FastAPI, config: Config) -> None:
    """
    Configure and apply all middleware to FastAPI application.
    
    Middleware is applied in order:
    1. Request logging (first, to log everything)
    2. CORS (for cross-origin requests)
    3. Authentication (validate API key)
    4. Rate limiting (enforce request limits)
    
    Args:
        app: FastAPI application instance
        config: Configuration object with middleware settings
    """
    logger.info("Setting up middleware...")
    
    # 1. Request Logging Middleware
    # Apply first to log all requests including those that fail auth/rate limiting
    app.add_middleware(RequestLoggingMiddleware)
    
    # 2. CORS Middleware
    # Configure CORS with origins from config
    if config.server.cors_origins:
        logger.info(f"Configuring CORS with origins: {config.server.cors_origins}")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.info("CORS not configured (no origins specified)")
    
    # 3. Authentication Middleware
    # Check API key if enabled
    if config.authentication.enabled:
        app.add_middleware(AuthenticationMiddleware, config=config)
    else:
        logger.info("Authentication middleware skipped (disabled in config)")
    
    # 4. Rate Limiting Middleware
    # Enforce request rate limits if enabled
    if config.rate_limiting.enabled:
        app.add_middleware(RateLimitMiddleware, config=config)
    else:
        logger.info("Rate limiting middleware skipped (disabled in config)")
    
    logger.info("Middleware setup complete")
