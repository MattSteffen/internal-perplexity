"""Tests for agent endpoint behavior."""

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from src.agents import agent_registry
from src.clients.router import client_router
from src.main import app
from src.milvus_client import get_milvus_context


class _DummyContext:
    def __init__(self) -> None:
        self.client = None
        self.user = {"milvus_token": "user:pass", "username": "user"}


class _DummyMilvusClient:
    def has_collection(self, collection_name: str) -> bool:  # noqa: ARG002
        return True

    def load_collection(self, collection_name: str) -> None:  # noqa: ARG002
        return None

    def describe_collection(self, collection_name: str) -> dict:  # noqa: ARG002
        return {}


def test_agent_endpoint_missing_messages() -> None:
    client = TestClient(app)
    app.dependency_overrides[get_milvus_context] = lambda: _DummyContext()
    try:
        response = client.post(
            "/v1/agents/milvuschat",
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "messages" in response.json()["detail"]


def test_agent_endpoint_dispatches_to_registry(monkeypatch) -> None:
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

    app.dependency_overrides[get_milvus_context] = lambda: _DummyContext()
    monkeypatch.setattr(agent_registry, "create_completion", _fake_create_completion)

    try:
        client = TestClient(app)
        response = client.post(
            "/v1/agents/milvuschat",
            json={
                "collection": "my_collection",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured.get("agent_name") == "milvuschat"
    assert isinstance(captured.get("body"), dict)


def test_legacy_agent_endpoint_removed() -> None:
    client = TestClient(app)
    response = client.post(\"/v1/agent\", json={})
    assert response.status_code == 404


def test_agent_response_includes_history(monkeypatch) -> None:
    async def _fake_create_completion(**kwargs):  # type: ignore[no-untyped-def]
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

    class _ContextWithClient(_DummyContext):
        def __init__(self) -> None:
            super().__init__()
            self.client = _DummyMilvusClient()

    app.dependency_overrides[get_milvus_context] = lambda: _ContextWithClient()
    monkeypatch.setattr(client_router, "create_completion", _fake_create_completion)

    try:
        client = TestClient(app)
        response = client.post(
            "/v1/agents/milvuschat",
            json={
                "collection": "test_collection",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert "history" in payload
    history = payload["history"]
    assert isinstance(history, list)
    assert any(item.get("role") == "system" for item in history)
    assert any(item.get("role") == "user" for item in history)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "ok"
