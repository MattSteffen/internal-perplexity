"""
query_processor.py
------------------
Converts raw user queries into structured research objectives and decomposes them into sub-queries.
"""

class QueryProcessor:
    """
    Processes and decomposes queries into smaller, focused sub-queries.

    Attributes:
        llm_interface: (Optional) An interface to an LLM for prompt-based parsing.
    """
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        # Query templates could be loaded from config/prompt_templates.yaml if needed.

    def parse_query(self, user_query):
        """
        Analyzes and structures the raw user query.

        Parameters:
            user_query (str): The original research query.

        Returns:
            dict: A parsed representation of the query.
        """
        # For MVP, simply wrap the query in a dict.
        return {"original_query": user_query}

    def generate_sub_queries(self, parsed_query):
        """
        Breaks down the main query into sub-queries.

        Parameters:
            parsed_query (dict): The structured query from parse_query.

        Returns:
            list: A list of sub-query strings.
        """
        original_query = parsed_query.get("original_query", "")
        words = original_query.split()
        # If the query is short, return it as a single sub-query.
        if len(words) < 5:
            return [original_query]
        # Otherwise, split into two halves for demonstration.
        mid = len(words) // 2
        sub_query1 = " ".join(words[:mid])
        sub_query2 = " ".join(words[mid:])
        return [sub_query1, sub_query2]
