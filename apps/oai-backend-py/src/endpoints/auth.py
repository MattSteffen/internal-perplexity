"""Authentication endpoints for Keycloak OAuth2."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import RedirectResponse

from src.auth import keycloak
from src.auth_utils import verify_token
from src.config import settings

router = APIRouter()


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    """Initiate OAuth2 login flow with Keycloak (GitLab identity provider).

    This endpoint redirects the user to Keycloak for authentication.
    The kc_idp_hint=gitlab parameter tells Keycloak to auto-redirect to GitLab.

    curl -X GET http://localhost:8000/login
    """
    if not settings.keycloak_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    if keycloak is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider is not initialized",
        )

    redirect_uri = settings.redirect_uri
    # kc_idp_hint=gitlab makes Keycloak auto-redirect to GitLab
    params = {
        "client_id": settings.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "kc_idp_hint": "gitlab",
    }

    return await keycloak.authorize_redirect(request, redirect_uri, **params)


@router.get("/auth/callback")
async def auth_callback(request: Request) -> RedirectResponse:
    """Handle OAuth2 callback from Keycloak after GitLab login.

    This endpoint exchanges the authorization code for access tokens
    and sets an HTTP-only cookie with the access token.

    curl -X GET http://localhost:8000/auth/callback?code=<authorization_code>&state=<state>
    """
    if keycloak is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider is not initialized",
        )

    try:
        token = await keycloak.authorize_access_token(request)
        await keycloak.parse_id_token(request, token)

        # Create redirect response to frontend
        response = RedirectResponse(url=settings.frontend_redirect_url)

        # Set HTTP-only cookie with access token
        response.set_cookie(
            key="access_token",
            value=token["access_token"],
            httponly=True,
            secure=True,
            samesite="lax",
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
        ) from e


@router.get("/logout")
async def logout() -> RedirectResponse:
    """Logout endpoint that clears cookies and redirects to Keycloak logout.

    curl -X GET http://localhost:8000/logout
    """
    if not settings.keycloak_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    logout_url = f"{settings.keycloak_url}/protocol/openid-connect/logout" f"?redirect_uri={settings.frontend_redirect_url}"

    response = RedirectResponse(url=logout_url)
    response.delete_cookie("access_token")

    return response


@router.get("/auth/me")
async def get_current_user(user: dict = Depends(verify_token)) -> dict:
    """Get current authenticated user information.

    Returns the decoded JWT claims for the authenticated user.

    curl -X GET http://localhost:8000/auth/me \
      -H "Authorization: Bearer $ACCESS_TOKEN"
    """
    return {"user": user}
