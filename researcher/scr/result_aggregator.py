"""
result_aggregator.py
--------------------
Aggregates results from multiple workers and formats the final summary with citations.
"""

class ResultAggregator:
    """
    Aggregates and formats worker results into a unified summary.

    Attributes:
        deduplication_strategy: (Optional) Function to remove duplicate findings.
        citation_formatter: (Optional) Function to format inline citations.
    """
    def __init__(self, deduplication_strategy=None, citation_formatter=None):
        self.deduplication_strategy = deduplication_strategy or self.default_deduplication
        self.citation_formatter = citation_formatter or self.default_citation_formatter

    def aggregate(self, worker_results):
        """
        Merges and deduplicates results from all workers.

        Parameters:
            worker_results (list): List of dictionaries containing individual worker findings.

        Returns:
            list: A deduplicated list of results.
        """
        # For this MVP, perform a simple deduplication based on the 'sub_query' field.
        unique_results = []
        seen = set()
        for res in worker_results:
            identifier = res.get("sub_query")
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(res)
        return unique_results

    def format_summary(self, aggregated_results):
        """
        Formats a human-readable summary including inline citations.

        Parameters:
            aggregated_results (list): List of aggregated research findings.

        Returns:
            str: A formatted summary string.
        """
        summary = "Aggregated Research Findings:\n"
        for result in aggregated_results:
            summary += f"- {result.get('sub_query')}: {result.get('findings')}\n"
            if "sub_result" in result:
                sub = result["sub_result"]
                summary += f"  * Further: {sub.get('findings')}\n"
            summary += f"  (Score: {result.get('score')})\n"
        
        # Append formatted citations
        citations = self.citation_formatter(aggregated_results)
        summary += "\nCitations:\n" + citations
        return summary

    def default_deduplication(self, results):
        """
        Default strategy: return results unchanged.

        Parameters:
            results (list): List of results.

        Returns:
            list: The original results list.
        """
        return results

    def default_citation_formatter(self, results):
        """
        Creates a simple citation string for each worker's result.

        Parameters:
            results (list): Aggregated research findings.

        Returns:
            str: A formatted citations string.
        """
        citations = ""
        for result in results:
            citations += f"Worker {result.get('worker_id')} cited for sub-query '{result.get('sub_query')}'.\n"
        return citations
