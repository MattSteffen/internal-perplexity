"""
title: Ollama REST Pipeline
author: open-webui
date: 2025-03-18
version: 1.0
license: MIT
description: A pipeline for forwarding requests to an Ollama instance running on host.docker.internal:11434
requirements: requests, pymilvus, grpcio
"""
import pymilvus
from typing import List, Union, Generator, Iterator
import json
import requests

class Pipeline:
    def __init__(self):
        self.ollama_url = "http://host.docker.internal:11434"

    def on_startup(self):
        # This function is called when the server is started
        print("pymilvus version:", pymilvus.__version__)
        pass

    def on_shutdown(self):
        # This function is called when the server is stopped
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator[str, None, None]:
        # Format the request body for Ollama
        ollama_body = {
            "model": "gemma3",
            "prompt": user_message,
            "stream": True
        }
        
        # If you need to pass the full conversation history
        if messages and len(messages) > 0:
            # Convert messages to a format Ollama can understand
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_messages.append({"role": role, "content": content})
            
            ollama_body["messages"] = formatted_messages

        # Stream the response from Ollama
        response = requests.post(
            self.ollama_url+"/api/generate",
            json=ollama_body,
            stream=True,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            yield f"Error from Ollama: {response.text}"
            return

        # Process the streaming response
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                # If the response is done, we can break
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                yield f"Error decoding JSON: {line.decode('utf-8')}"