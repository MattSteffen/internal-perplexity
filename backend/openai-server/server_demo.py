import openai
import asyncio

# Test client for the OpenAI proxy server
async def test_proxy_server():
    # Configure OpenAI client to use your proxy server
    client = openai.AsyncOpenAI(
        api_key="your-proxy-api-key",  # This will be forwarded to the destination
        base_url="http://localhost:8000/v1"  # Your proxy server URL
    )
    
    try:
        # Test non-streaming chat completion
        print("Testing non-streaming chat completion...")
        response = await client.chat.completions.create(
            model="gemma3",
            messages=[
                {"role": "user", "content": "Hello! How are you?"}
            ],
            max_tokens=100
        )
        print("Non-streaming response:", response.choices[0].message.content)
        print()
        
        # Test streaming chat completion
        print("Testing streaming chat completion...")
        stream = await client.chat.completions.create(
            model="gemma3",
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            max_tokens=50,
            stream=True
        )
        
        print("Streaming response: ", end="")
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\n")
        
        # Test models endpoint
        print("Testing models endpoint...")
        models = await client.models.list()
        print(f"Available models: {len(models.data)} models found")
        for model in models.data[:3]:  # Show first 3 models
            print(f"  - {model.id}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.close()

# Synchronous version for easier testing
def test_proxy_server_sync():
    client = openai.OpenAI(
        api_key="your-proxy-api-key",
        base_url="http://localhost:8000/v1"
    )
    
    try:
        # Simple test
        response = client.chat.completions.create(
            model="gemma3",
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=50
        )
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run async test
    print("Running async test...")
    asyncio.run(test_proxy_server())
    
    print("\nRunning sync test...")
    test_proxy_server_sync()