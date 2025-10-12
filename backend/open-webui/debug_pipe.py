"""
Debug script for testing the Radchat pipeline outside of Open WebUI.
"""

import asyncio
import sys
from pathlib import Path

# Add the radchat directory to the path so we can import it
sys.path.insert(0, str(Path(__file__).parent / "radchat"))

from radchat import Pipe


async def mock_event_emitter(event: dict):
    """Mock event emitter that prints events."""
    event_type = event.get("type")
    data = event.get("data", {})

    if event_type == "status":
        print(f"[STATUS] {data.get('description')} (done: {data.get('done')})")
    elif event_type == "citation":
        source = data.get("source", {})
        distance = data.get("distance", "N/A")
        print(f"[CITATION] {source.get('name')} (distance: {distance})")
    else:
        print(f"[{event_type.upper()}] {data}")


# TODO:
# - Make several queries, one of 'who are you', one of 'all the files written by X' for metadata filtering, one for semantic search.
async def main():
    """Main test function."""
    print("=" * 60)
    print("Radchat Debug Script")
    print("=" * 60)

    # Create a Pipe instance
    print("\n[INFO] Creating Pipe instance...")
    pipe = Pipe()

    # Print valve configuration
    print(f"[INFO] Collection: {pipe.user_valves.COLLECTION_NAME}")
    print(f"[INFO] Milvus Username: {pipe.user_valves.MILVUS_USERNAME}")
    print(f"[INFO] Milvus Host: From CONFIG")

    # Create a test query
    query = "are there any papers that have 'ai' as a unique word in the metadata?"
    print(f"\n[INFO] Query: '{query}'")

    # Build the request body
    body = {"messages": [{"role": "user", "content": query}]}

    # Mock user (optional, can be None)
    mock_user = {"id": "test-user", "name": "Test User", "email": "test@example.com"}

    print("\n[INFO] Starting pipe execution...\n")
    print("-" * 60)

    try:
        # Execute the pipe
        response_generator = pipe.pipe(
            body=body, __event_emitter__=mock_event_emitter, __user__=mock_user
        )

        # Collect and display the streaming response
        full_response = ""
        async for chunk in response_generator:
            if isinstance(chunk, dict):
                # Handle different response types
                if "choices" in chunk:
                    # Standard OpenAI-style response
                    if chunk.get("object") == "chat.completion.chunk":
                        # Streaming chunk
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        thinking = delta.get("thinking", "")

                        if thinking:
                            print(f"{thinking}", end="", flush=True)
                        if content:
                            if full_response == "":
                                print("\n")
                            print(content, end="", flush=True)
                            full_response += content
                    elif chunk.get("object") == "chat.completion.final":
                        # Final response with citations
                        print("\n\n" + "-" * 60)
                        print("[FINAL RESPONSE]")
                        message_content = chunk["choices"][0]["message"]["content"]
                        print(message_content)

                        # Print citations
                        citations = chunk.get("citations", [])
                        if citations:
                            print("\n[CITATIONS]")
                            for i, citation in enumerate(citations, 1):
                                source = citation.get("source", {})
                                distance = citation.get("distance", "N/A")
                                print(
                                    f"  {i}. {source.get('name')} (distance: {distance})"
                                )
                elif "error" in chunk:
                    print(f"\n[ERROR] {chunk['error']}")
            else:
                print(f"[UNKNOWN CHUNK TYPE] {type(chunk)}: {chunk}")

        print("\n" + "-" * 60)
        print("\n[INFO] Pipe execution completed successfully!")

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
