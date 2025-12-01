import copy
import logging
import random
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

    def run_benchmark(self, generate_queries: bool = False) -> BenchmarkRunResults:
        """
        Run comprehensive benchmark.
        """
        # benchmark_start_time = time.time()
        top_k_values = list(range(1, 101))

        # Load documents from collection
        # Only get chunk_index == 0 to avoid processing the same document multiple times
        logger.info(f"Loading documents from collection '{self.db_config.collection}' with filter 'chunk_index == 0'")
        all_docs = self.milvus_client.query(
            collection_name=self.db_config.collection,
            filter="chunk_index == 0",
            output_fields=["source", "text", "id"],
            limit=10000,
        )
        logger.info(f"Loaded {len(all_docs)} documents from collection")

        # Generate queries from documents

        queries_by_doc = {}
        query_generation_stats = {
            "total_docs": len(all_docs),
            "docs_with_queries": 0,
            "total_queries_generated": 0,
            "stored_questions_used": 0,
            "docs_skipped_no_source": 0,
            "docs_skipped_no_text": 0,
            "docs_skipped_text_too_short": 0,
        }

        logger.info(f"Starting query generation with generate_queries={generate_queries}")
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

                # Try to use stored benchmark questions first (when generate_queries=False)
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
                                # Fall through to generate queries from text
                        else:
                            logger.debug(f"No stored questions found for source: {source}")
                    except Exception as e:
                        logger.warning(f"Error getting stored questions for {source}: {e}")
                        # Fall through to generate queries from text
                
                # Fall back to generating queries from text if stored questions weren't found
                if not stored_questions_found:
                    logger.debug(f"Generating queries from text for source: {source}")
                    words = text.split()
                    logger.debug(f"Document has {len(words)} words")
                    if len(words) > 30:
                        # Generate multiple queries per document for better statistics
                        num_queries = min(3, max(1, len(words) // 100))  # 1-3 queries based on document length
                        logger.debug(f"Will generate {num_queries} queries for this document")

                        for query_idx in range(num_queries):
                            start_index = random.randint(0, len(words) - 30)
                            query = " ".join(words[start_index : start_index + 30])

                            if source not in queries_by_doc:
                                queries_by_doc[source] = []
                            queries_by_doc[source].append(query)
                            query_generation_stats["total_queries_generated"] += 1
                            logger.debug(f"Generated query {query_idx + 1}/{num_queries}: {query[:50]}...")

                        query_generation_stats["docs_with_queries"] += 1
                        pbar.set_postfix_str(f"Generated queries: {query_generation_stats['total_queries_generated']}")
                        logger.debug(f"Successfully generated {num_queries} queries for source: {source}")
                    else:
                        logger.debug(f"Skipping document - text too short ({len(words)} words, need >30)")
                        query_generation_stats["docs_skipped_text_too_short"] += 1

                pbar.update(1)

        # Log query generation statistics
        logger.info("Query generation statistics:")
        logger.info(f"  Total documents processed: {query_generation_stats['total_docs']}")
        logger.info(f"  Documents with queries: {query_generation_stats['docs_with_queries']}")
        logger.info(f"  Total queries generated: {query_generation_stats['total_queries_generated']}")
        logger.info(f"  Stored questions used: {query_generation_stats['stored_questions_used']}")
        logger.info(f"  Documents skipped (no source): {query_generation_stats['docs_skipped_no_source']}")
        logger.info(f"  Documents skipped (no text): {query_generation_stats['docs_skipped_no_text']}")
        logger.info(f"  Documents skipped (text too short): {query_generation_stats['docs_skipped_text_too_short']}")
        logger.info(f"  Total queries_by_doc keys: {len(queries_by_doc)}")
        if queries_by_doc:
            total_queries = sum(len(queries) for queries in queries_by_doc.values())
            logger.info(f"  Total queries in queries_by_doc: {total_queries}")
            logger.info(f"  Sample sources with queries: {list(queries_by_doc.keys())[:5]}")

        # TODO: Implement LLM-based query generation
        # if generate_queries:

        if not queries_by_doc:
            logger.error("No queries could be generated from documents!")
            logger.error(f"Query generation stats: {query_generation_stats}")
            raise ValueError("No queries could be generated from documents")

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

        # Calculate percent in top-k
        percent_in_top_k = {}
        total_queries = sum(len(q) for q in queries_by_doc.values())

        for k in top_k_values:
            count = sum(1 for placement in placement_distribution.keys() if placement <= k)
            percent_in_top_k[k] = (count / total_queries) * 100 if total_queries > 0 else 0

        # Log final benchmark statistics
        # benchmark_time = time.time() - benchmark_start_time

        # Log top-k performance highlights
        # top_k_highlights = [(k, percent_in_top_k[k]) for k in [1, 5, 10, 50, 100]]
        # for k, percentage in top_k_highlights:

        # Convert integer keys to strings for Pydantic validation
        results_by_doc_str_keys = {str(k): v for k, v in results_by_doc.items()}

        return BenchmarkRunResults(
            results_by_doc=results_by_doc_str_keys,
            placement_distribution=placement_distribution,
            distance_distribution=distance_distribution,
            percent_in_top_k=percent_in_top_k,
            search_time_distribution=search_time_distribution,
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
