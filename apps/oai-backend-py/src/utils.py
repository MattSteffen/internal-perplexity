"""Utility functions for error handling and common operations."""

from fastapi import HTTPException
from openai import APIError, APIStatusError


def map_openai_error_to_http(error: APIError) -> HTTPException:
    """Map OpenAI API errors to appropriate HTTP status codes.

    Converts OpenAI client errors into proper HTTP exceptions with appropriate
    status codes:
    - 404: Model not found
    - 400: Bad request (invalid parameters)
    - 401: Authentication errors
    - 429: Rate limit exceeded
    - 503: Ollama service errors/unavailable
    - 502: Ollama service unavailable (other server errors)
    """
    if isinstance(error, APIStatusError):
        status_code = error.status_code
        # Map common status codes
        if status_code == 404:
            # Model not found or resource not found
            return HTTPException(
                status_code=404,
                detail=f"Model not found: {error.message}",
            )
        elif status_code == 400:
            # Bad request (invalid parameters)
            return HTTPException(
                status_code=400,
                detail=f"Invalid request: {error.message}",
            )
        elif status_code == 401:
            # Unauthorized
            return HTTPException(
                status_code=401,
                detail=f"Authentication error: {error.message}",
            )
        elif status_code == 429:
            # Rate limit
            return HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {error.message}",
            )
        elif status_code == 500:
            # Server error from Ollama
            return HTTPException(
                status_code=503,
                detail=f"Ollama service error: {error.message}",
            )
        elif status_code >= 500:
            # Other server errors
            return HTTPException(
                status_code=502,
                detail=f"Ollama service unavailable: {error.message}",
            )
        else:
            # Other client errors
            return HTTPException(
                status_code=status_code,
                detail=error.message or "An error occurred",
            )
    else:
        # Generic API error (connection issues, etc.)
        error_msg = getattr(error, "message", None) or str(error) or "Failed to connect to Ollama service"
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "unreachable" in error_msg.lower():
            return HTTPException(
                status_code=503,
                detail=f"Ollama service is unavailable: {error_msg}",
            )
        return HTTPException(
            status_code=500,
            detail=f"Internal error: {error_msg}",
        )
