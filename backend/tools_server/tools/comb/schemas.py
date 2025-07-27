from pydantic import BaseModel
from typing import List

class CombInput(BaseModel):
    perspective: str
    filters: List[str] = []

CombInputSchema = {
    "type": "function",
    "function": {
        "name": "comb",
        "description": "Iteratively reads through documents in the database, collecting tidbits that might be related to the user's desires from a given perspective.",
        "parameters": {
            "type": "object",
            "required": ["perspective"],
            "properties": {
                "perspective": {
                    "type": "string",
                    "description": "The perspective from which to analyze the documents",
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter expressions to apply to the search",
                    "default": [],
                },
            },
        },
    },
}
