"""
worker.py
---------
Implements the ResearchWorker class, responsible for executing focused research tasks on sub-queries.
"""

import logging

class ResearchWorker:
    """
    Performs research on a specific sub-query and reports the findings.

    Key properties:
      - worker_id: Unique identifier for the worker.
      - parent_agent: Reference to the main ResearcherAgent.
      - sub_query: The specific query fragment to process.
      - local_context: Local state tracking for the worker.
      - current_depth: The current level in recursive processing.
    """
    def __init__(self, worker_id, parent_agent, sub_query, current_depth):
        self.worker_id = worker_id
        self.parent_agent = parent_agent
        self.sub_query = sub_query
        self.current_depth = current_depth
        self.local_context = {"results": []}
    
    def execute_task(self):
        """
        Processes the sub-query, simulating research and (if allowed) spawning a sub-worker.

        Returns:
            dict: A dictionary containing findings and related metadata.
        """
        logging.info(f"Worker {self.worker_id} processing sub-query: '{self.sub_query}'")
        
        # Simulate retrieval and analysis (e.g., using vectorstore_connector and llm_interface)
        result = {
            "worker_id": self.worker_id,
            "sub_query": self.sub_query,
            "findings": f"Findings for '{self.sub_query}'",
            "score": 0.8  # Example static score
        }
        
        # Optionally, if the maximum recursion depth is not reached, spawn a sub-worker for further detail.
        if self.current_depth < self.parent_agent.max_recursion_depth:
            logging.info(f"Worker {self.worker_id} spawning a sub-worker for further investigation.")
            sub_sub_query = f"Further details on {self.sub_query}"
            sub_worker = self.parent_agent.spawn_worker(sub_sub_query, self.current_depth + 1)
            sub_result = sub_worker.execute_task()
            result["sub_result"] = sub_result
        
        return result

    def evaluate_sources(self, findings):
        """
        Evaluate and score retrieved documents based on metadata and relevance.

        Parameters:
            findings (list): A list of retrieved document findings.

        Returns:
            float: A computed relevance score.
        """
        # For MVP, a dummy evaluation is returned.
        return 0.8

    def report_results(self, result):
        """
        Reports the research findings back to the parent agent.
        In this MVP, reporting is achieved by simply returning the result.

        Parameters:
            result (dict): The research findings.

        Returns:
            dict: The same research findings.
        """
        return result
