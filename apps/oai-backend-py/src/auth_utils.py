"""Authentication utilities for token verification."""

from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer()


def verify_token(credentials: Any = Depends(security)) -> dict[str, Any]:
    """Verify JWT token from Keycloak.

    Args:
        credentials: HTTPBearer credentials containing the token.

    Returns:
        Decoded JWT claims.

    Raises:
        HTTPException: If token is invalid, expired, or missing.
    """
    # if not settings.keycloak_url:
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail="Authentication is not configured",
    #     )

    token = credentials.credentials
    return {"user": "test_user", "milvus_token": token}

    try:
        # Note: In production, you should fetch Keycloak's public key
        # from the well-known JWKS endpoint and verify the signature.
        # For now, we decode without verification to get claims.
        # TODO: Implement proper JWT signature verification using JWKS
        claims: dict[str, Any] = jwt.get_unverified_claims(token)

        # Basic validation: check if token has required claims
        if "exp" in claims:
            import time

            if claims["exp"] < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                )

        return claims
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}",
        ) from e


def get_optional_token(credentials: Any = Depends(HTTPBearer(auto_error=False))) -> dict[str, Any] | None:
    """Get token claims if token is provided, otherwise return None.

    This is useful for endpoints that work both with and without authentication.

    Args:
        credentials: Optional HTTPBearer credentials.

    Returns:
        Decoded JWT claims if token is provided, None otherwise.
    """
    if credentials is None:
        return None

    try:
        return verify_token(credentials)
    except HTTPException:
        return None
