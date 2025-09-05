#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

DIR="$1"
OUTPUT="concatenated_output.txt"

# Find all files (excluding directories), concatenate into one file
find "$DIR" -type f -exec cat {} + > "$OUTPUT"

# Copy the contents to clipboard
pbcopy < "$OUTPUT"

echo "All files concatenated into $OUTPUT and copied to clipboard."
