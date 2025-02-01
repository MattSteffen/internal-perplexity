import os
from typing import List

def find_dirs(input_dir: str, allowed_extensions: List[str] = ['.json']) -> List[str]:
    """Finds directories containing JSON files."""
    json_dirs = set()
    for root, _, files in os.walk(input_dir):
        if any(file.endswith('.json') for file in files):
            json_dirs.add(root)
    return list(json_dirs)