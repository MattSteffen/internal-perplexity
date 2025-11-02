"""OAuth2 authentication setup with Keycloak."""

from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI

from src.config import settings

oauth = OAuth()
keycloak = None


def init_oauth(app: FastAPI) -> None:
    """Initialize OAuth with FastAPI app and register Keycloak provider.

    Args:
        app: FastAPI application instance.
    """
    global keycloak

    # Register Keycloak as an OAuth provider
    # Note: This will only work if keycloak_url is configured
    # For Starlette/FastAPI, OAuth integration happens automatically via the app
    if settings.keycloak_url:
        keycloak = oauth.register(
            name="keycloak",
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            server_metadata_url=f"{settings.keycloak_url}/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
