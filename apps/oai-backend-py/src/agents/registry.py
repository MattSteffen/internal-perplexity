"""Agent registry for managing available agents."""

from typing import Any

from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion
from pymilvus import MilvusClient  # type: ignore

from src.agents.base import Agent
from src.agents.milvuschat import milvuschat_agent

_AVAILABLE_AGENTS: dict[str, Agent] = {
    "milvuschat": milvuschat_agent,
}


class AgentRegistry:
    """Registry for managing and executing agents."""

    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents = agents if agents is not None else _AVAILABLE_AGENTS.copy()

    def list_agents(self) -> list[str]:
        return sorted(self._agents.keys())

    def get_agent(self, agent_name: str) -> Agent:
        key = agent_name.lower()
        if key not in self._agents:
            raise KeyError(f"Agent '{agent_name}' is not registered")
        return self._agents[key]

    def register_agent(self, agent_name: str, agent: Agent) -> None:
        self._agents[agent_name.lower()] = agent

    async def create_completion(
        self,
        agent_name: str,
        body: dict[str, Any],
        user: dict[str, Any] | None = None,
        milvus_client: MilvusClient | None = None,
    ) -> StreamingResponse | ChatCompletion:
        agent = self.get_agent(agent_name)
        return await agent.create_completion(
            agent_name=agent_name.lower(),
            body=body,
            user=user,
            milvus_client=milvus_client,
        )


agent_registry = AgentRegistry()
