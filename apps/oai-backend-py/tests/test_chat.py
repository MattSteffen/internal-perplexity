"""Tests for the chat completion endpoint behavior."""

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from src.auth_utils import get_optional_token
from src.endpoints import chat as chat_endpoint
from src.main import app


def test_chat_milvuschat_requires_collection() -> None:
    """MilvusChat should require a collection parameter."""
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "milvuschat",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert "collection" in response.json()["detail"]


def test_chat_milvuschat_infers_token_from_user(monkeypatch) -> None:
    """MilvusChat should infer token from authenticated user when omitted."""
    captured: dict[str, object] = {}

    async def _fake_create_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return ChatCompletion(
            id="test",
            object="chat.completion",
            created=0,
            model="milvuschat",
            choices=[
                {
                    "index": 0,
                    "message": ChatCompletionMessage(role="assistant", content="ok"),
                    "finish_reason": "stop",
                }
            ],
        )

    app.dependency_overrides[get_optional_token] = lambda: {
        "milvus_token": "user:pass",
        "username": "user",
    }
    monkeypatch.setattr(chat_endpoint.client_router, "create_completion", _fake_create_completion)

    try:
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "milvuschat",
                "collection": "my_collection",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured.get("token") == "user:pass"
    assert captured.get("collection") == "my_collection"
