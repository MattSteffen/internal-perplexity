"""Weather tool implementation."""

import json
from typing import Any

from openai.types.chat import ChatCompletionToolParam


class WeatherTool:
    """Weather tool that returns mock weather data for a given location."""

    def get_definition(self) -> ChatCompletionToolParam:
        """Get the OpenAI tool definition for the weather tool.

        Returns:
            ChatCompletionToolParam with weather function schema.
        """
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": ("Get the current weather for a specific location. " "Returns temperature in Celsius, conditions, and humidity."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit for temperature (defaults to celsius)",
                        },
                    },
                    "required": ["location"],
                },
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the weather tool with the given arguments.

        Args:
            arguments: Dictionary with 'location' and optionally 'unit' keys.

        Returns:
            JSON string with weather data.
        """
        print("Calling weather tool", arguments)
        location = arguments.get("location", "Unknown")
        unit = arguments.get("unit", "celsius")

        # Mock weather data - in production, this would call a real weather API
        # For now, return deterministic mock data based on location
        temp_celsius = 20 + hash(location) % 15  # Temperature between 20-35°C

        if unit == "fahrenheit":
            temp = int((temp_celsius * 9 / 5) + 32)
        else:
            temp = temp_celsius

        conditions = ["sunny", "cloudy", "partly cloudy", "rainy"][hash(location) % 4]
        humidity = 40 + (hash(location) % 30)  # Humidity between 40-70%

        result = {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "conditions": conditions,
            "humidity": humidity,
        }

        return json.dumps(result, indent=2)
