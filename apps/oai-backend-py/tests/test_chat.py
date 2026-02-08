"""Tests for the chat completion endpoint behavior."""

from fastapi.testclient import TestClient

from src.main import app


def test_chat_rejects_agent_models() -> None:
    """Agent names should not be accepted as chat models."""
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "milvuschat",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert "/v1/agents/milvuschat" in response.json()["detail"]
