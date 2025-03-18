"""
title: Internal Perplexity
author: matt-steffen
funding_url: https://github.com/open-webui
version: 0.0.1
license: None
"""

from typing import List, Union, Generator, Iterator
import requests


"""
TODO:
- This pipeline will call another server that will perform all the search and summarization
- The server will be a simple python server that will take in a prompt and return a response.
- This will be used until I get around to making the proper front end as openwebui will not suffice.
"""


class Pipe:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "perplexity_pipeline"
        self.name = "Perplexity Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = "http://host.docker.internal:11434"
        MODEL = "llama3.2:1b"

        for k in ["user", "chat_id", "title"]:
            if k in body:
                del body[k]
        body["stream"] = True

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={**body, "model": MODEL},
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
