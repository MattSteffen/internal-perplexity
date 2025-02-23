"""
main.py
-------
Entry point for the Researcher Agent MVP.

Usage:
    python main.py "Your research query here"

This script loads configuration, initializes logging and the main ResearcherAgent,
and executes the user query to produce a final summarized output.
"""

import argparse
import logging
import yaml
from researcher_agent import ResearcherAgent

def load_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load agent configuration from YAML file
    config = load_config("config/agent_config.yaml")
    
    # Setup logging (this could also use utils/logging_util.py for more control)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Create the main ResearcherAgent instance
    agent = ResearcherAgent(
        num_workers=config.get("num_workers", 3),
        max_recursion_depth=config.get("max_recursion_depth", 2),
        llm_interface=None,             # Replace with a real LLM interface instance as needed.
        vectorstore_connector=None,     # Replace with a real vectorstore connector instance.
        security_context=None           # Replace with a real security context instance.
    )
    
    # Parse command-line argument for the user query
    parser = argparse.ArgumentParser(description="Researcher Agent MVP")
    parser.add_argument("query", help="User query for research")
    args = parser.parse_args()
    
    # Execute the query and print the final summary
    final_summary = agent.execute_query(args.query)
    print("Final Summary:")
    print(final_summary)

if __name__ == "__main__":
    main()
