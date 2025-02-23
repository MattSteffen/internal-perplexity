"""
researcher_agent.py
-------------------
Contains the ResearcherAgent class, the main orchestrator for managing the research process.
"""

import logging
from query_processor import QueryProcessor
from worker import ResearchWorker
from result_aggregator import ResultAggregator

class ResearcherAgent:
    """
    Orchestrates the entire research process.
    
    Key properties:
      - num_workers: Maximum number of workers to spawn.
      - max_recursion_depth: Maximum allowed depth for recursive sub-query processing.
      - llm_interface: Interface to the LLM for processing/generating queries.
      - vectorstore_connector: Connector for interacting with Milvus or other vectorstores.
      - security_context: Handles user authentication and data filtering.
      - research_context: Stores query history and worker results.
    """
    def __init__(self, num_workers=3, max_recursion_depth=2, llm_interface=None,
                 vectorstore_connector=None, security_context=None):
        self.num_workers = num_workers
        self.max_recursion_depth = max_recursion_depth
        self.llm_interface = llm_interface
        self.vectorstore_connector = vectorstore_connector
        self.security_context = security_context
        self.research_context = {"query_history": [], "worker_results": []}
        self.query_processor = QueryProcessor(llm_interface)
        self.result_aggregator = ResultAggregator()
    
    def execute_query(self, user_query, current_depth=0):
        """
        Processes the user query, spawns workers for sub-queries, and aggregates results.

        Parameters:
            user_query (str): The original research query.
            current_depth (int): The current recursion depth (default is 0).

        Returns:
            str: A final formatted summary of the research findings.
        """
        logging.info(f"Executing query: {user_query} at depth {current_depth}")
        self.research_context["query_history"].append(user_query)
        
        # Parse the user query into a structured format and generate sub-queries.
        parsed_query = self.query_processor.parse_query(user_query)
        sub_queries = self.query_processor.generate_sub_queries(parsed_query)
        
        # Limit the number of workers spawned based on configuration.
        worker_count = min(len(sub_queries), self.num_workers)
        for i in range(worker_count):
            sub_query = sub_queries[i]
            worker = self.spawn_worker(sub_query, current_depth)
            result = worker.execute_task()
            self.research_context["worker_results"].append(result)
        
        # Aggregate and finalize results.
        aggregated_results = self.gather_results()
        final_summary = self.validate_and_finalize(aggregated_results)
        return final_summary
    
    def spawn_worker(self, sub_query, current_depth):
        """
        Creates and returns a new ResearchWorker instance for the given sub-query.

        Parameters:
            sub_query (str): The specific sub-query to process.
            current_depth (int): The current recursion depth.

        Returns:
            ResearchWorker: The spawned worker.
        """
        worker_id = len(self.research_context["worker_results"]) + 1
        worker = ResearchWorker(worker_id, self, sub_query, current_depth)
        return worker
    
    def gather_results(self):
        """
        Collects and aggregates results from all spawned workers.

        Returns:
            list: Aggregated list of worker results.
        """
        aggregated = self.result_aggregator.aggregate(self.research_context["worker_results"])
        return aggregated
    
    def validate_and_finalize(self, aggregated_results):
        """
        Applies any necessary validation (e.g., metadata rules, security checks)
        and finalizes the summary output.

        Parameters:
            aggregated_results (list): The aggregated research findings.

        Returns:
            str: The final, formatted summary.
        """
        final_summary = self.result_aggregator.format_summary(aggregated_results)
        return final_summary
