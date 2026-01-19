"""
Milvus benchmark implementation for evaluating search quality.

This module provides tools for benchmarking Milvus search performance
using IR (Information Retrieval) metrics like MRR, Recall@K, NDCG@K, etc.
"""

import json
import logging
import math
import statistics
import time
from typing import Any

from pymilvus import MilvusClient
from tqdm import tqdm

from ..llm.embeddings import EmbedderConfig, get_embedder
from .database_client import (
    BenchmarkResult,
    BenchmarkRunResults,
    DatabaseBenchmark,
    DatabaseClientConfig,
    SearchResult,
)
from .milvus_client import MilvusDB

logger = logging.getLogger(__name__)


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

        dcg = 0.0
        for placement in placements:
            if placement is not None and placement <= k:
                dcg += 1.0 / math.log2(placement + 1)

        idcg = 1.0
        ndcg_per_query = dcg / len(placements) if placements else 0.0
        return ndcg_per_query / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_hit_rate_at_k(placements: list[int | None], k: int) -> float:
        """
        Calculate Hit Rate@K.

        Hit Rate@K is the percentage of queries where at least one relevant
        document was found in the top-k results.

        Args:
            placements: List of placement positions (1-indexed) or None if not found
            k: Top-k threshold

        Returns:
            Hit Rate@K (0.0 to 1.0)
        """
        return BenchmarkMetrics.calculate_recall_at_k(placements, k)

    @staticmethod
    def calculate_summary_stats(
        placements: list[int | None],
    ) -> tuple[float, float, float]:
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
    Benchmark class for evaluating Milvus search quality.

    Uses MilvusDB.search() for performing searches and calculates
    IR metrics to evaluate retrieval quality.
    """

    def __init__(
        self,
        db_config: DatabaseClientConfig,
        embed_config: EmbedderConfig,
        db: MilvusDB | None = None,
    ) -> None:
        """
        Initialize the MilvusBenchmark.

        Args:
            db_config: Configuration for the database
            embed_config: Configuration for the embedder
            db: Optional pre-configured MilvusDB instance
        """
        self.db_config = db_config
        self.embed_config = embed_config

        # Create embedder
        self.embedder = get_embedder(embed_config)

        # Use provided db or create a new one
        if db is not None:
            self.db = db
            # Ensure embedder is set
            if self.db.embedder is None:
                self.db.set_embedder(self.embedder)
        else:
            # Create a minimal MilvusDB for searching
            # We need to create it with minimal config since we don't have full crawler_config
            self._create_db_client()

        # Direct Milvus client for queries not exposed by MilvusDB
        self._milvus_client: MilvusClient | None = None

    def _create_db_client(self) -> None:
        """Create a MilvusDB client for benchmarking."""
        from ..config import CrawlerConfig

        # Create a minimal crawler config for the DB client
        # This is needed because MilvusDB requires a CrawlerConfig
        minimal_config = CrawlerConfig(
            name="benchmark",
            embeddings=self.embed_config,
            llm=self.embed_config,  # Placeholder, not used for search
            vision_llm=self.embed_config,  # Placeholder, not used for search
            database=self.db_config,
            converter=None,
            extractor=None,
            chunking=None,
        )

        self.db = MilvusDB(
            config=self.db_config,
            embedding_dimension=self.embedder.get_dimension(),
            crawler_config=minimal_config,
            embedder=self.embedder,
        )

        # Connect to the database
        try:
            self.db.connect(create_if_missing=False)
        except Exception as e:
            logger.warning(f"Failed to connect MilvusDB: {e}")
            self.db = None

    def _get_milvus_client(self) -> MilvusClient | None:
        """Get a direct Milvus client for queries."""
        if self._milvus_client is None:
            try:
                self._milvus_client = MilvusClient(
                    uri=self.db_config.uri, token=self.db_config.token
                )
                self._milvus_client.load_collection(self.db_config.collection)
            except Exception:
                return None
        return self._milvus_client

    def search(
        self,
        queries: list[str],
        filters: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform a search using MilvusDB.

        Args:
            queries: List of query texts
            filters: Optional filter expressions

        Returns:
            List of SearchResult objects
        """
        if self.db is None or not self.db.is_connected():
            logger.error("MilvusDB not connected")
            return []

        return self.db.search(texts=queries, filters=filters, limit=100)

    def run_benchmark(
        self,
        generate_queries: bool = False,
        k_values: list[int] | None = None,
        skip_docs_without_questions: bool = True,
    ) -> BenchmarkRunResults:
        """
        Run comprehensive benchmark with IR quality metrics.

        Args:
            generate_queries: If True, generate queries using LLM (not yet implemented)
            k_values: List of k values for @K metrics (default: [1, 5, 10, 25, 50, 100])
            skip_docs_without_questions: If True, skip documents without benchmark_questions

        Returns:
            BenchmarkRunResults with comprehensive IR metrics
        """
        if k_values is None:
            k_values = [1, 5, 10, 25, 50, 100]

        milvus_client = self._get_milvus_client()
        if milvus_client is None:
            raise ValueError("Failed to connect to Milvus for benchmark")

        # Load documents from collection (only chunk_index == 0)
        logger.info(
            f"Loading documents from collection '{self.db_config.collection}' with filter 'chunk_index == 0'"
        )
        try:
            all_docs = milvus_client.query(
                collection_name=self.db_config.collection,
                filter="chunk_index == 0",
                output_fields=["source", "text", "id", "chunk_index"],
                limit=10000,
            )
            if all_docs is None:
                all_docs = []
        except Exception as e:
            logger.error(
                f"Error querying collection '{self.db_config.collection}': {e}"
            )
            raise ValueError(
                f"Failed to query collection '{self.db_config.collection}': {e}. "
                "Ensure the collection exists and is accessible."
            ) from e

        logger.info(f"Loaded {len(all_docs)} documents from collection")

        # Generate queries from documents using stored benchmark_questions
        queries_by_doc: dict[str, list[str]] = {}
        query_generation_stats = {
            "total_docs": len(all_docs),
            "docs_with_queries": 0,
            "stored_questions_used": 0,
            "docs_skipped_no_source": 0,
            "docs_skipped_no_text": 0,
            "docs_skipped_no_questions": 0,
        }

        logger.info(
            f"Starting query generation with generate_queries={generate_queries}, "
            f"skip_docs_without_questions={skip_docs_without_questions}"
        )

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
                    logger.debug(
                        f"Document {doc_idx} (source={source}): Skipping - no text"
                    )
                    query_generation_stats["docs_skipped_no_text"] += 1
                    pbar.update(1)
                    continue

                # Try to use stored benchmark questions
                stored_questions_found = False
                if not generate_queries:
                    logger.debug(f"Checking for stored questions for source: {source}")
                    try:
                        stored_questions_result = milvus_client.query(
                            collection_name=self.db_config.collection,
                            filter=f"source == '{source}' AND chunk_index == 0",
                            output_fields=["benchmark_questions"],
                            limit=1,
                        )

                        if stored_questions_result and stored_questions_result[0].get(
                            "benchmark_questions"
                        ):
                            try:
                                stored_questions = json.loads(
                                    stored_questions_result[0]["benchmark_questions"]
                                )
                                if (
                                    isinstance(stored_questions, list)
                                    and len(stored_questions) > 0
                                ):
                                    queries_by_doc[source] = stored_questions
                                    query_generation_stats["docs_with_queries"] += 1
                                    query_generation_stats[
                                        "stored_questions_used"
                                    ] += len(stored_questions)
                                    pbar.set_postfix_str(
                                        f"Stored questions: {query_generation_stats['stored_questions_used']}"
                                    )
                                    stored_questions_found = True
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.debug(
                                    f"Failed to parse stored questions for {source}: {e}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Error getting stored questions for {source}: {e}"
                        )

                if not stored_questions_found:
                    if skip_docs_without_questions:
                        query_generation_stats["docs_skipped_no_questions"] += 1

                pbar.update(1)

        # Log statistics
        logger.info("Query generation statistics:")
        logger.info(f"  Total documents: {query_generation_stats['total_docs']}")
        logger.info(f"  Documents with queries: {query_generation_stats['docs_with_queries']}")
        logger.info(f"  Stored questions used: {query_generation_stats['stored_questions_used']}")

        if not queries_by_doc:
            logger.error("No queries could be generated from documents!")
            raise ValueError(
                "No queries could be generated. Ensure documents have benchmark_questions."
            )

        # Run benchmark searches
        results_by_doc: dict[str, list[BenchmarkResult]] = {}
        placement_distribution: dict[int, int] = {}
        distance_distribution: list[float] = []
        search_time_distribution: list[float] = []

        with tqdm(
            total=len(queries_by_doc), desc="Benchmarking documents", unit="doc"
        ) as doc_pbar:
            for source, queries in queries_by_doc.items():
                results_by_doc[source] = []

                with tqdm(
                    total=len(queries),
                    desc=f"Queries for {source}...",
                    unit="query",
                    leave=False,
                ) as query_pbar:
                    for query in queries:
                        query_start_time = time.time()

                        try:
                            # Use MilvusDB.search()
                            search_results = self.search(queries=[query])
                            search_time = time.time() - query_start_time
                            search_time_distribution.append(search_time)

                            found_in_results = False
                            for i, res in enumerate(search_results):
                                if res.document.source == source:
                                    placement = i + 1
                                    distance = res.distance
                                    result = BenchmarkResult(
                                        query=query,
                                        expected_source=str(source),
                                        placement_order=placement,
                                        distance=distance,
                                        time_to_search=search_time,
                                        found=True,
                                    )
                                    results_by_doc[source].append(result)
                                    placement_distribution[placement] = (
                                        placement_distribution.get(placement, 0) + 1
                                    )
                                    distance_distribution.append(distance)
                                    found_in_results = True
                                    break

                            if not found_in_results:
                                result = BenchmarkResult(
                                    query=query,
                                    expected_source=str(source),
                                    found=False,
                                    time_to_search=search_time,
                                )
                                results_by_doc[source].append(result)

                            query_pbar.set_postfix_str(
                                f"Found: {'Yes' if found_in_results else 'No'}"
                            )
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
                            query_pbar.update(1)

                doc_pbar.update(1)

        # Collect all placements for metric calculations
        all_placements: list[int | None] = []
        for results in results_by_doc.values():
            for result in results:
                all_placements.append(result.placement_order)

        total_queries = len(all_placements)
        queries_found = sum(1 for p in all_placements if p is not None)
        queries_not_found = total_queries - queries_found

        # Calculate percent in top-k
        percent_in_top_k = {}
        for k in k_values:
            count = sum(v for p, v in placement_distribution.items() if p <= k)
            percent_in_top_k[k] = (count / total_queries) * 100 if total_queries > 0 else 0

        # Calculate IR metrics
        mrr = BenchmarkMetrics.calculate_mrr(all_placements)
        recall_at_k = {
            k: BenchmarkMetrics.calculate_recall_at_k(all_placements, k) for k in k_values
        }
        precision_at_k = {
            k: BenchmarkMetrics.calculate_precision_at_k(all_placements, k)
            for k in k_values
        }
        ndcg_at_k = {
            k: BenchmarkMetrics.calculate_ndcg_at_k(all_placements, k) for k in k_values
        }
        hit_rate_at_k = {
            k: BenchmarkMetrics.calculate_hit_rate_at_k(all_placements, k)
            for k in k_values
        }

        mean_placement, median_placement, std_placement = (
            BenchmarkMetrics.calculate_summary_stats(all_placements)
        )

        # Log results
        logger.info("Benchmark Results Summary:")
        logger.info(f"  Total queries: {total_queries}")
        logger.info(f"  Queries found: {queries_found} ({queries_found/total_queries*100:.1f}%)")
        logger.info(f"  MRR: {mrr:.4f}")

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
