import os
import time
import json
import copy
import random
from typing import Any, Dict, List, Optional

import ollama
import matplotlib.pyplot as plt
import numpy as np
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from tqdm import tqdm

from .database_client import (
    DatabaseBenchmark,
    BenchmarkResult,
    BenchmarkRunResults,
    DatabaseClientConfig,
)


OUTPUT_FIELDS = [
    "default_source",
    "default_chunk_index",
    "default_text",
    "default_metadata",
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

    def __init__(
        self, db_config: "DatabaseClientConfig", embed_config: "EmbedderConfig"
    ) -> None:
        """
        Initializes the MilvusBenchmark class, setting up clients for Ollama and Milvus.
        """
        self.db_config = db_config
        self.embed_config = embed_config

        self.ollama_client = ollama.Client(host=self.embed_config.base_url)

        self.milvus_client = self._connect_milvus()

    def _connect_milvus(self) -> Optional[MilvusClient]:
        """
        Connects to the Milvus database and loads the collection.
        """
        try:
            client = MilvusClient(uri=self.db_config.uri, token=self.db_config.token)
            if not client.has_collection(collection_name=self.db_config.collection):
                return None
            client.load_collection(collection_name=self.db_config.collection)
            return client
        except Exception as e:
            return None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding for the given text using Ollama.
        """
        try:
            response = self.ollama_client.embeddings(
                model=self.embed_config.model, prompt=text
            )
            return response.get("embedding")
        except Exception as e:
            return None

    def search(
        self, queries: List[str], filters: Optional[List[str]] = []
    ) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search in Milvus using the given queries and filters.
        """
        if not self.milvus_client:
            return {"error": "Milvus client not initialized."}

        search_start_time = time.time()
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
                        anns_field="default_text_embedding",
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
                    anns_field="default_text_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str,
                    limit=10,
                )
            )
            search_requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field="default_metadata_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str,
                    limit=10,
                )
            )

        if not search_requests:
            return {"error": "No valid search requests could be created."}

        # Perform hybrid search
        api_start_time = time.time()

        try:
            ranker = RRFRanker(k=100)
            results = self.milvus_client.hybrid_search(
                collection_name=self.db_config.collection,
                reqs=search_requests,
                ranker=ranker,
                output_fields=OUTPUT_FIELDS,
                limit=100,
            )

            api_time = time.time() - api_start_time
            total_time = time.time() - search_start_time

            # Process results
            processed_results = []
            if results:
                for doc in results[0]:
                    entity = doc.entity.to_dict()
                    entity["distance"] = doc.distance
                    processed_results.append(entity)

            return processed_results

        except Exception as e:
            api_time = time.time() - api_start_time
            total_time = time.time() - search_start_time
            raise

    def run_benchmark(self, generate_queries: bool = False) -> BenchmarkRunResults:
        """
        Run comprehensive benchmark.
        """
        benchmark_start_time = time.time()
        top_k_values = list(range(1, 101))

        # Load documents from collection
        docs_start_time = time.time()

        all_docs = self.milvus_client.query(
            collection_name=self.db_config.collection,
            output_fields=["default_text"],
            limit=10000,
        )

        docs_load_time = time.time() - docs_start_time

        # Generate queries from documents

        queries_by_doc = {}
        query_generation_stats = {
            "total_docs": len(all_docs),
            "docs_with_queries": 0,
            "total_queries_generated": 0,
            "stored_questions_used": 0,
        }

        with tqdm(total=len(all_docs), desc="Processing documents", unit="doc") as pbar:
            for doc in all_docs:
                source = doc.get("id")
                text = doc.get("default_text")

                if not text or len(text.strip()) == 0:
                    pbar.update(1)
                    continue

                # Try to use stored benchmark questions first (when generate_queries=False)
                if not generate_queries:
                    try:
                        # Query for stored benchmark questions (only for chunk_index 0 to avoid duplicates)
                        stored_questions_result = self.milvus_client.query(
                            collection_name=self.db_config.collection,
                            filter=f"default_source == '{source}' AND default_chunk_index == 0",
                            output_fields=["benchmark_questions"],
                            limit=1,
                        )

                        if stored_questions_result and stored_questions_result[0].get(
                            "benchmark_questions"
                        ):
                            import json

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
                                    pbar.update(1)
                                    continue  # Skip to next document
                            except (json.JSONDecodeError, KeyError) as e:
                                continue
                    except Exception as e:
                        print(f"Error getting stored questions for {source}: {e}")
                        continue
                # Fall back to generating queries from text
                words = text.split()
                if len(words) > 30:
                    # Generate multiple queries per document for better statistics
                    num_queries = min(
                        3, max(1, len(words) // 100)
                    )  # 1-3 queries based on document length

                    for _ in range(num_queries):
                        start_index = random.randint(0, len(words) - 30)
                        query = " ".join(words[start_index : start_index + 30])

                        if source not in queries_by_doc:
                            queries_by_doc[source] = []
                        queries_by_doc[source].append(query)
                        query_generation_stats["total_queries_generated"] += 1

                    query_generation_stats["docs_with_queries"] += 1
                    pbar.set_postfix_str(
                        f"Generated queries: {query_generation_stats['total_queries_generated']}"
                    )

                pbar.update(1)

        # Log query generation statistics

        # TODO: Implement LLM-based query generation
        # if generate_queries:

        if not queries_by_doc:
            raise ValueError("No queries could be generated from documents")

        # Run benchmark searches
        results_by_doc: Dict[str, List[BenchmarkResult]] = {}
        placement_distribution: Dict[int, int] = {}
        distance_distribution: List[float] = []
        search_time_distribution: List[float] = []

        benchmark_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "found_in_top_10": 0,
            "found_in_top_100": 0,
        }

        with tqdm(
            total=len(queries_by_doc), desc="Benchmarking documents", unit="doc"
        ) as doc_pbar:
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
                                if res.get("id") == source:
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
                                    placement_distribution[placement] = (
                                        placement_distribution.get(placement, 0) + 1
                                    )
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

                            query_pbar.set_postfix_str(
                                f"Found: {'Yes' if found_in_results else 'No'} ({search_time:.3f}s)"
                            )
                            query_pbar.update(1)

                        except Exception as e:
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
                doc_pbar.set_postfix_str(
                    f"Queries: {len(queries)}, Time: {doc_time:.2f}s"
                )
                doc_pbar.update(1)

        # Calculate percent in top-k
        percent_in_top_k = {}
        total_queries = sum(len(q) for q in queries_by_doc.values())

        for k in top_k_values:
            count = sum(
                1 for placement in placement_distribution.keys() if placement <= k
            )
            percent_in_top_k[k] = (
                (count / total_queries) * 100 if total_queries > 0 else 0
            )

        # Log final benchmark statistics
        benchmark_time = time.time() - benchmark_start_time

        # Log top-k performance highlights
        top_k_highlights = [(k, percent_in_top_k[k]) for k in [1, 5, 10, 50, 100]]
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
