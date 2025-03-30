import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message  # type: ignore
import pymilvus


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(default="http://host.docker.internal:11434")
        MILVUS_HOST: str = Field(default="host.docker.internal")
        MILVUS_PORT: str = Field(default="19530")
        MILVUS_COLLECTION: str = Field(default="conference_docs")

    def __init__(self):
        self.type = "pipe"
        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv(
                    "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
                ),
                "MILVUS_HOST": os.getenv("MILVUS_HOST", "host.docker.internal"),
                "MILVUS_PORT": os.getenv("MILVUS_PORT", "19530"),
                "MILVUS_COLLECTION": os.getenv("MILVUS_COLLECTION", "conference_docs"),
            }
        )

    def get_connection_to_milvus(self):
        """Establish connection to Milvus"""
        collection = None
        try:
            pymilvus.connections.connect(
                alias="rag",
                host=self.valves.MILVUS_HOST,
                port=self.valves.MILVUS_PORT,
            )
            print(
                f"Connected to Milvus at {self.valves.MILVUS_HOST}:{self.valves.MILVUS_PORT}"
            )

            if not pymilvus.utility.has_collection(self.valves.MILVUS_COLLECTION):
                print(
                    f"Warning: Collection {self.valves.MILVUS_COLLECTION} does not exist in Milvus"
                )
                return None

            collection = pymilvus.Collection(self.valves.MILVUS_COLLECTION)
            collection.load()
            print(f"Using Milvus collection: {self.valves.MILVUS_COLLECTION}")
            return collection
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return None

    def get_embedding(self, text):
        """Get embedding from Ollama for the given text"""
        try:
            response = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": "all-minilm:v2", "prompt": text},
            )
            if response.status_code != 200:
                print(f"Failed to get embedding: HTTP {response.status_code}")
                return None

            result = response.json()
            return result.get("embedding")
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def search_milvus(self, query, limit=5):
        """Search Milvus collection with embedded query"""
        collection = self.get_connection_to_milvus()
        if collection is None:
            return []

        # Get embedding for the query
        embedding = self.get_embedding(query)
        if embedding is None:
            print("Failed to get embedding for query")
            return []

        try:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # Perform the search
            results = collection.search(
                data=[embedding],
                anns_field="embedding",  # Assuming your vector field is named "embedding"
                param=search_params,
                limit=limit,
                output_fields=["text"],  # Adjust based on your schema
            )

            documents = []
            for hits in results:
                for hit in hits:
                    documents.append(
                        {
                            "id": hit.id,
                            "content": hit.entity.get("text"),
                            "source": hit.entity.get("source"),
                            "score": hit.score,
                        }
                    )

            return documents
        except Exception as e:
            print(f"Error searching Milvus: {e}")
            return []

    def pipes(self) -> List[dict]:
        return [{"id": "gemma3:latest", "name": "gemma3-rag"}]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        # Process messages
        ollama_messages = []
        context = ""

        # Extract user query from the last user message
        user_query = ""
        for message in reversed(messages):
            if message["role"] == "user":
                if isinstance(message.get("content"), list):
                    user_query = " ".join(
                        [
                            item["text"]
                            for item in message["content"]
                            if item["type"] == "text"
                        ]
                    )
                else:
                    user_query = message.get("content", "")
                break

        # If we have a non-empty user query, perform RAG
        if user_query.strip():
            rag_results = self.search_milvus(user_query)
            if rag_results:
                context = "\nRelevant context:\n"
                for idx, result in enumerate(rag_results):
                    context += (
                        f"{idx+1}. {result['content']} (Source: {result['source']})\n"
                    )

        # Format messages for Ollama
        for message in messages:
            content = ""
            if isinstance(message.get("content"), list):
                content = " ".join(
                    [
                        item["text"]
                        for item in message["content"]
                        if item["type"] == "text"
                    ]
                )
            else:
                content = message.get("content", "")

            ollama_messages.append({"role": message["role"], "content": content})

        # Add system message with context
        if system_message:
            ollama_messages.insert(
                0, {"role": "system", "content": f"{system_message}{context}"}
            )
        elif context:
            ollama_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Please use the following context to help answer the user's question: {context}",
                },
            )

        # Prepare payload for Ollama
        payload = {
            "model": "gemma3:latest",
            "messages": ollama_messages,
            "stream": body.get("stream", False),
            "options": {
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop": body.get("stop", []),
            },
        }

        # Log if context was added
        if context:
            print(f"Added RAG context to prompt")

        url = f"{self.valves.OLLAMA_BASE_URL}/api/chat"

        # Make the request to Ollama
        try:
            if body.get("stream", False):
                return self.stream_response(url, payload)
            else:
                return self.non_stream_response(url, payload)
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, payload):
        try:
            with requests.post(
                url, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                            elif "done" in data and data["done"]:
                                break
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {line}")
        except Exception as e:
            print(f"Error in stream_response: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, payload):
        try:
            response = requests.post(url, json=payload, timeout=(3.05, 60))
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return res.get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error in non_stream_response: {e}")
            return f"Error: {e}"
