import copy
import logging
import math
import statistics
import time
from typing import Any

import ollama
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..llm.embeddings import EmbedderConfig
from .database_client import (
    BenchmarkResult,
    BenchmarkRunResults,
    DatabaseBenchmark,
    DatabaseClientConfig,
)

OUTPUT_FIELDS = [
    "source",
    "chunk_index",
    "text",
    "str_metadata",
    "title",
    "author",
    "date",
    "keywords",
    "unique_words",
]


class BenchmarkMetrics:
    """
    Utility class for calculating Information Retrieval metrics.
    
    Provides methods to compute standard IR evaluation metrics including
    MRR, Recall@K, Precision@K, NDCG@K, and Hit Rate@K.
    """

    @staticmethod
    def calculate_mrr(placements: list[int | None]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR is the average of 1/rank for each query where the relevant document
        was found. Queries where the document was not found contribute 0.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            
        Returns:
            Mean Reciprocal Rank (0.0 to 1.0)
        """
        if not placements:
            return 0.0
        
        reciprocal_ranks = []
        for placement in placements:
            if placement is not None and placement > 0:
                reciprocal_ranks.append(1.0 / placement)
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def calculate_recall_at_k(placements: list[int | None], k: int) -> float:
        """
        Calculate Recall@K.
        
        Recall@K is the fraction of queries where the relevant document was
        found in the top-k results.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            k: Top-k threshold
            
        Returns:
            Recall@K (0.0 to 1.0)
        """
        if not placements:
            return 0.0
        
        found_in_top_k = sum(1 for p in placements if p is not None and p <= k)
        return found_in_top_k / len(placements)

    @staticmethod
    def calculate_precision_at_k(placements: list[int | None], k: int) -> float:
        """
        Calculate Precision@K.
        
        Precision@K is the fraction of top-k results that are relevant.
        Since we have one relevant document per query, this is equivalent to
        Recall@K for our use case.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            k: Top-k threshold
            
        Returns:
            Precision@K (0.0 to 1.0)
        """
        # For single relevant document per query, Precision@K = Recall@K
        return BenchmarkMetrics.calculate_recall_at_k(placements, k)

    @staticmethod
    def calculate_ndcg_at_k(placements: list[int | None], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
        
        NDCG@K measures ranking quality by considering position of relevant items.
        For binary relevance (found/not found), this simplifies to a position-weighted metric.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            k: Top-k threshold
            
        Returns:
            NDCG@K (0.0 to 1.0)
        """
        if not placements:
            return 0.0
        
        # For binary relevance, DCG = 1 / log2(rank + 1) for found items in top-k
        # IDCG = 1 / log2(2) = 1.0 (perfect ranking has relevant doc at position 1)
        dcg = 0.0
        for placement in placements:
            if placement is not None and placement <= k:
                dcg += 1.0 / math.log2(placement + 1)
        
        # IDCG is 1.0 (perfect case: relevant doc at position 1)
        idcg = 1.0
        
        # Average NDCG across all queries
        ndcg_per_query = dcg / len(placements) if placements else 0.0
        return ndcg_per_query / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_hit_rate_at_k(placements: list[int | None], k: int) -> float:
        """
        Calculate Hit Rate@K.
        
        Hit Rate@K is the percentage of queries where at least one relevant
        document was found in the top-k results. This is equivalent to Recall@K
        for single relevant document per query.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            k: Top-k threshold
            
        Returns:
            Hit Rate@K (0.0 to 1.0)
        """
        return BenchmarkMetrics.calculate_recall_at_k(placements, k)

    @staticmethod
    def calculate_summary_stats(placements: list[int | None]) -> tuple[float, float, float]:
        """
        Calculate summary statistics for placement positions.
        
        Args:
            placements: List of placement positions (1-indexed) or None if not found
            
        Returns:
            Tuple of (mean, median, std_dev) for found placements only
        """
        found_placements = [p for p in placements if p is not None]
        
        if not found_placements:
            return (0.0, 0.0, 0.0)
        
        mean = statistics.mean(found_placements)
        median = statistics.median(found_placements)
        std_dev = statistics.stdev(found_placements) if len(found_placements) > 1 else 0.0
        
        return (mean, median, std_dev)


class MilvusBenchmark(DatabaseBenchmark):
    """
    A class to benchmark Milvus search performance.
    """

    def __init__(self, db_config: DatabaseClientConfig, embed_config: EmbedderConfig) -> None:
        """
        Initializes the MilvusBenchmark class, setting up clients for Ollama and Milvus.
        """
        self.db_config = db_config
        self.embed_config = embed_config

        self.ollama_client = ollama.Client(host=self.embed_config.base_url)

        self.milvus_client = self._connect_milvus()

    def _connect_milvus(self) -> MilvusClient | None:
        """
        Connects to the Milvus database and loads the collection.
        """
        try:
            client = MilvusClient(uri=self.db_config.uri, token=self.db_config.token)
            if not client.has_collection(collection_name=self.db_config.collection):
                return None
            client.load_collection(collection_name=self.db_config.collection)
            return client
        except Exception:
            return None

    def get_embedding(self, text: str) -> list[float] | None:
        """
        Generates an embedding for the given text using Ollama.
        """
        try:
            response = self.ollama_client.embeddings(model=self.embed_config.model, prompt=text)
            return response.get("embedding")
        except Exception:
            return None

    def search(self, queries: list[str], filters: list[str] | None = []) -> list[dict[str, Any]]:
        """
        Performs a hybrid search in Milvus using the given queries and filters.
        """
        if not self.milvus_client:
            return {"error": "Milvus client not initialized."}

        # search_start_time = time.time()
        filters = copy.deepcopy(filters)

        filters.insert(
            0,
            f"array_contains_any(security_group, {list(self.milvus_client.describe_user(self.db_config.username).get('roles', []))})",
        )
        filter_str = " and ".join(filters)

        search_requests = []
        embedding_failures = 0

        # Generate embeddings for queries
        for query in queries:
            embedding = self.get_embedding(query)
            if embedding:
                search_requests.append(
                    AnnSearchRequest(
                        data=[embedding],
                        anns_field="text_embedding",
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=filter_str,
                        limit=10,
                    )
                )
            else:
                embedding_failures += 1

            # Add sparse search requests
            search_requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field="text_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str,
                    limit=10,
                )
            )
            search_requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field="metadata_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str,
                    limit=10,
                )
            )

        if not search_requests:
            return {"error": "No valid search requests could be created."}

        # Perform hybrid search
        # api_start_time = time.time()

        try:
            ranker = RRFRanker(k=100)
            results = self.milvus_client.hybrid_search(
                collection_name=self.db_config.collection,
                reqs=search_requests,
                ranker=ranker,
                output_fields=OUTPUT_FIELDS,
                limit=100,
            )

            # Process results
            processed_results = []
            if results:
                for doc in results[0]:
                    entity = doc.entity.to_dict()
                    entity["distance"] = doc.distance
                    processed_results.append(entity)

            return processed_results

        except Exception:
            raise

    def run_benchmark(self, generate_queries: bool = False, k_values: list[int] | None = None, skip_docs_without_questions: bool = True) -> BenchmarkRunResults:
        """
        Run comprehensive benchmark with IR quality metrics.
        
        Args:
            generate_queries: If True, generate queries using LLM (not yet implemented)
            k_values: List of k values for @K metrics (default: [1, 5, 10, 25, 50, 100])
            skip_docs_without_questions: If True, skip documents without benchmark_questions (default: True)
            
        Returns:
            BenchmarkRunResults with comprehensive IR metrics
        """
        if k_values is None:
            k_values = [1, 5, 10, 25, 50, 100]

        # Load documents from collection
        # Only get chunk_index == 0 to avoid processing the same document multiple times
        logger.info(f"Loading documents from collection '{self.db_config.collection}' with filter 'chunk_index == 0'")
        try:
            all_docs = self.milvus_client.query(
                collection_name=self.db_config.collection,
                filter="chunk_index == 0",
                output_fields=["source", "text", "id", "chunk_index"],
                limit=10000,
            )
            if all_docs is None:
                all_docs = []
        except Exception as e:
            logger.error(f"Error querying collection '{self.db_config.collection}': {e}")
            raise ValueError(
                f"Failed to query collection '{self.db_config.collection}': {e}. "
                "Ensure the collection exists and is accessible."
            ) from e
        
        logger.info(f"Loaded {len(all_docs)} documents from collection")

        # Generate queries from documents using stored benchmark_questions
        queries_by_doc = {}
        query_generation_stats = {
            "total_docs": len(all_docs),
            "docs_with_queries": 0,
            "stored_questions_used": 0,
            "docs_skipped_no_source": 0,
            "docs_skipped_no_text": 0,
            "docs_skipped_no_questions": 0,
        }

        logger.info(f"Starting query generation with generate_queries={generate_queries}, skip_docs_without_questions={skip_docs_without_questions}")
        with tqdm(total=len(all_docs), desc="Processing documents", unit="doc") as pbar:
            for doc_idx, doc in enumerate(all_docs):
                source = doc.get("source")
                text = doc.get("text")

                if not source:
                    logger.debug(f"Document {doc_idx}: Skipping - no source field")
                    query_generation_stats["docs_skipped_no_source"] += 1
                    pbar.update(1)
                    continue
                
                if not text or len(text.strip()) == 0:
                    logger.debug(f"Document {doc_idx} (source={source}): Skipping - no text or empty text")
                    query_generation_stats["docs_skipped_no_text"] += 1
                    pbar.update(1)
                    continue

                logger.debug(f"Processing document {doc_idx}: source={source}, text_length={len(text)}")

                # Try to use stored benchmark questions
                stored_questions_found = False
                if not generate_queries:
                    logger.debug(f"Checking for stored questions for source: {source}")
                    try:
                        # Query for stored benchmark questions (only for chunk_index 0 to avoid duplicates)
                        stored_questions_result = self.milvus_client.query(
                            collection_name=self.db_config.collection,
                            filter=f"source == '{source}' AND chunk_index == 0",
                            output_fields=["benchmark_questions"],
                            limit=1,
                        )
                        logger.debug(f"Stored questions query returned {len(stored_questions_result) if stored_questions_result else 0} results")

                        if stored_questions_result and stored_questions_result[0].get("benchmark_questions"):
                            import json

                            try:
                                stored_questions = json.loads(stored_questions_result[0]["benchmark_questions"])
                                logger.debug(f"Parsed stored questions: {type(stored_questions)}, length={len(stored_questions) if isinstance(stored_questions, list) else 'N/A'}")
                                if isinstance(stored_questions, list) and len(stored_questions) > 0:
                                    queries_by_doc[source] = stored_questions
                                    query_generation_stats["docs_with_queries"] += 1
                                    query_generation_stats["stored_questions_used"] += len(stored_questions)
                                    pbar.set_postfix_str(f"Stored questions: {query_generation_stats['stored_questions_used']}")
                                    stored_questions_found = True
                                    logger.debug(f"Using {len(stored_questions)} stored questions for source: {source}")
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.debug(f"Failed to parse stored questions for {source}: {e}")
                        else:
                            logger.debug(f"No stored questions found for source: {source}")
                    except Exception as e:
                        logger.warning(f"Error getting stored questions for {source}: {e}")
                
                # Skip documents without questions if configured to do so
                if not stored_questions_found:
                    if skip_docs_without_questions:
                        logger.debug(f"Skipping document {source} - no benchmark questions found")
                        query_generation_stats["docs_skipped_no_questions"] += 1
                    else:
                        logger.warning(f"Document {source} has no benchmark questions but skip_docs_without_questions=False (not skipping)")

                pbar.update(1)

        # Log query generation statistics
        logger.info("Query generation statistics:")
        logger.info(f"  Total documents processed: {query_generation_stats['total_docs']}")
        logger.info(f"  Documents with queries: {query_generation_stats['docs_with_queries']}")
        logger.info(f"  Stored questions used: {query_generation_stats['stored_questions_used']}")
        logger.info(f"  Documents skipped (no source): {query_generation_stats['docs_skipped_no_source']}")
        logger.info(f"  Documents skipped (no text): {query_generation_stats['docs_skipped_no_text']}")
        logger.info(f"  Documents skipped (no questions): {query_generation_stats['docs_skipped_no_questions']}")
        logger.info(f"  Total queries_by_doc keys: {len(queries_by_doc)}")
        if queries_by_doc:
            total_queries = sum(len(queries) for queries in queries_by_doc.values())
            logger.info(f"  Total queries in queries_by_doc: {total_queries}")
            logger.info(f"  Sample sources with queries: {list(queries_by_doc.keys())[:5]}")

        # TODO: Implement LLM-based query generation

        if not queries_by_doc:
            logger.error("No queries could be generated from documents!")
            logger.error(f"Query generation stats: {query_generation_stats}")
            
            # Provide more specific error messages based on the situation
            if query_generation_stats["total_docs"] == 0:
                raise ValueError(
                    f"No documents found in collection '{self.db_config.collection}' with filter 'chunk_index == 0'. "
                    "Ensure the collection contains documents with chunk_index == 0."
                )
            elif query_generation_stats["docs_skipped_no_questions"] > 0:
                # Documents were loaded but all were skipped because they had no questions
                if skip_docs_without_questions:
                    raise ValueError(
                        f"All {query_generation_stats['total_docs']} documents were skipped because they have no benchmark_questions. "
                        f"Set skip_docs_without_questions=False to generate queries using LLM, or ensure documents have benchmark_questions."
                    )
                else:
                    raise ValueError(
                        f"No queries could be generated from {query_generation_stats['total_docs']} documents. "
                        "Documents may be missing benchmark_questions and LLM query generation is not yet implemented."
                    )
            else:
                # Other reasons (no source, no text, etc.)
                raise ValueError(
                    f"No queries could be generated from documents. "
                    f"Stats: {query_generation_stats}. "
                    "Ensure documents have valid source, text, and benchmark_questions fields."
                )

        # Run benchmark searches
        results_by_doc: dict[str, list[BenchmarkResult]] = {}
        placement_distribution: dict[int, int] = {}
        distance_distribution: list[float] = []
        search_time_distribution: list[float] = []

        benchmark_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "found_in_top_10": 0,
            "found_in_top_100": 0,
        }

        with tqdm(total=len(queries_by_doc), desc="Benchmarking documents", unit="doc") as doc_pbar:
            for source, queries in queries_by_doc.items():
                results_by_doc[source] = []
                doc_search_start = time.time()

                with tqdm(
                    total=len(queries),
                    desc=f"Queries for {source}...",
                    unit="query",
                    leave=False,
                ) as query_pbar:
                    for query in queries:
                        query_start_time = time.time()
                        benchmark_stats["total_searches"] += 1

                        try:
                            search_results = self.search(queries=[query])
                            search_time = time.time() - query_start_time
                            search_time_distribution.append(search_time)

                            found_in_results = False
                            for i, res in enumerate(search_results):
                                if res.get("source") == source:
                                    placement = i + 1
                                    distance = res["distance"]
                                    result = BenchmarkResult(
                                        query=query,
                                        expected_source=str(source),
                                        placement_order=placement,
                                        distance=distance,
                                        time_to_search=search_time,
                                        found=True,
                                    )
                                    results_by_doc[source].append(result)
                                    placement_distribution[placement] = placement_distribution.get(placement, 0) + 1
                                    distance_distribution.append(distance)

                                    # Track success metrics
                                    if placement <= 10:
                                        benchmark_stats["found_in_top_10"] += 1
                                    if placement <= 100:
                                        benchmark_stats["found_in_top_100"] += 1

                                    found_in_results = True
                                    benchmark_stats["successful_searches"] += 1
                                    break

                            if not found_in_results:
                                result = BenchmarkResult(
                                    query=query,
                                    expected_source=str(source),
                                    found=False,
                                    time_to_search=search_time,
                                )
                                results_by_doc[source].append(result)
                                benchmark_stats["failed_searches"] += 1

                            query_pbar.set_postfix_str(f"Found: {'Yes' if found_in_results else 'No'} ({search_time:.3f}s)")
                            query_pbar.update(1)

                        except Exception:
                            search_time = time.time() - query_start_time
                            result = BenchmarkResult(
                                query=query,
                                expected_source=str(source),
                                found=False,
                                time_to_search=search_time,
                            )
                            results_by_doc[source].append(result)
                            benchmark_stats["failed_searches"] += 1
                            query_pbar.update(1)
                            continue

                doc_time = time.time() - doc_search_start
                doc_pbar.set_postfix_str(f"Queries: {len(queries)}, Time: {doc_time:.2f}s")
                doc_pbar.update(1)

        # Collect all placements for metric calculations
        all_placements: list[int | None] = []
        for results in results_by_doc.values():
            for result in results:
                all_placements.append(result.placement_order)

        total_queries = len(all_placements)
        queries_found = sum(1 for p in all_placements if p is not None)
        queries_not_found = total_queries - queries_found

        # Calculate percent in top-k (fixed: count total queries found, not unique positions)
        percent_in_top_k = {}
        for k in k_values:
            count = sum(v for p, v in placement_distribution.items() if p <= k)
            percent_in_top_k[k] = (count / total_queries) * 100 if total_queries > 0 else 0

        # Calculate IR metrics using BenchmarkMetrics utility
        mrr = BenchmarkMetrics.calculate_mrr(all_placements)
        recall_at_k = {k: BenchmarkMetrics.calculate_recall_at_k(all_placements, k) for k in k_values}
        precision_at_k = {k: BenchmarkMetrics.calculate_precision_at_k(all_placements, k) for k in k_values}
        ndcg_at_k = {k: BenchmarkMetrics.calculate_ndcg_at_k(all_placements, k) for k in k_values}
        hit_rate_at_k = {k: BenchmarkMetrics.calculate_hit_rate_at_k(all_placements, k) for k in k_values}

        # Calculate summary statistics
        mean_placement, median_placement, std_placement = BenchmarkMetrics.calculate_summary_stats(all_placements)

        # Log final benchmark statistics
        logger.info("Benchmark Results Summary:")
        logger.info(f"  Total queries: {total_queries}")
        logger.info(f"  Queries found: {queries_found} ({queries_found/total_queries*100:.1f}%)")
        logger.info(f"  Queries not found: {queries_not_found} ({queries_not_found/total_queries*100:.1f}%)")
        logger.info(f"  MRR: {mrr:.4f}")
        logger.info(f"  Mean placement: {mean_placement:.2f}")
        logger.info(f"  Median placement: {median_placement:.2f}")
        logger.info(f"  Std dev placement: {std_placement:.2f}")
        for k in k_values:
            logger.info(f"  Recall@{k}: {recall_at_k[k]:.4f}")

        # Convert integer keys to strings for Pydantic validation
        results_by_doc_str_keys = {str(k): v for k, v in results_by_doc.items()}

        return BenchmarkRunResults(
            results_by_doc=results_by_doc_str_keys,
            placement_distribution=placement_distribution,
            distance_distribution=distance_distribution,
            percent_in_top_k=percent_in_top_k,
            search_time_distribution=search_time_distribution,
            mrr=mrr,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mean_placement=mean_placement,
            median_placement=median_placement,
            std_placement=std_placement,
            total_queries=total_queries,
            queries_found=queries_found,
            queries_not_found=queries_not_found,
        )


# if __name__ == "__main__":
#     db_config = DatabaseClientConfig(
#         provider="milvus",
#         host="localhost",
#         port=19530,
#         username="root",
#         password="Milvus",
#         collection="test_collection",
#     )
#     embed_config = EmbedderConfig(
#         provider="ollama",
#         model="all-minilm:v2",
#         base_url="http://localhost:11434",
#     )

#     benchmark = MilvusBenchmark(db_config=db_config, embed_config=embed_config)
#     if benchmark.milvus_client:
#         # Example with pre-defined queries
#         queries_by_doc = {
#             "doc1.txt": ["query for doc1", "another query for doc1"],
#             "doc2.txt": ["query for doc2"],
#         }
#         results = benchmark.run_benchmark(queries_by_doc=queries_by_doc)
#         benchmark.plot_results(results, "benchmark_results_predefined")
#         benchmark.save_results(results, "benchmark_results_predefined/results.json")

#         # Example with auto-generated queries
#         results_auto = benchmark.run_benchmark()
#         benchmark.plot_results(results_auto, "benchmark_results_auto")
#         benchmark.save_results(results_auto, "benchmark_results_auto/results.json")
