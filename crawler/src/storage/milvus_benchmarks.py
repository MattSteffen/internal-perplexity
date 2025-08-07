import os
import time
import json
import random
from typing import Any, Dict, List, Optional

import ollama
import matplotlib.pyplot as plt
import numpy as np
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from src.storage.database_client import (
    DatabaseBenchmark,
    BenchmarkResult,
    BenchmarkRunResults,
    DatabaseClientConfig,
)
from src.processing.embeddings import EmbedderConfig


OUTPUT_FIELDS = [
    "source",
    "chunk_index",
    "metadata",
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
                print(
                    f"Error: Collection '{self.db_config.collection}' does not exist."
                )
                return None
            client.load_collection(collection_name=self.db_config.collection)
            print("Successfully connected to Milvus and loaded collection.")
            return client
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
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
            print(f"Error getting embedding from Ollama: {e}")
            return None

    def search(
        self, queries: List[str], filters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search in Milvus using the given queries and filters.
        """
        if not self.milvus_client:
            return {"error": "Milvus client not initialized."}

        search_requests = []
        for query in queries:
            embedding = self.get_embedding(query)
            if embedding:
                search_requests.append(
                    AnnSearchRequest(
                        data=[embedding],
                        anns_field="text_embedding",
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=" and ".join(filters) if filters else None,
                        limit=10,
                    )
                )

            search_requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field="text_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=" and ".join(filters) if filters else None,
                    limit=10,
                )
            )
            search_requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field="metadata_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=" and ".join(filters) if filters else None,
                    limit=10,
                )
            )

        if not search_requests:
            return {"error": "No valid search requests could be created."}

        ranker = RRFRanker(k=100)
        results = self.milvus_client.hybrid_search(
            collection_name=self.db_config.collection,
            reqs=search_requests,
            ranker=ranker,
            output_fields=OUTPUT_FIELDS,
            limit=100,
        )

        processed_results = []
        if results:
            for doc in results[0]:
                entity = doc.entity.to_dict()
                entity["distance"] = doc.distance
                processed_results.append(entity)

        return processed_results

    def run_benchmark(self, generate_queries: bool = False) -> BenchmarkRunResults:
        top_k_values = list(range(1, 101))

        queries_by_doc = {}
        all_docs = self.milvus_client.query(
            collection_name=self.db_config.collection,
            output_fields=["text"],
            limit=10000,
        )
        for doc in all_docs:
            source = doc.get("id")
            text = doc.get("text")
            words = text.split()
            if len(words) > 30:
                start_index = random.randint(0, len(words) - 30)
                query = " ".join(words[start_index : start_index + 30])
                if source not in queries_by_doc:
                    queries_by_doc[source] = []
                queries_by_doc[source].append(query)

        if generate_queries:
            # for each of the documents, generate 3 queries via an LLM
            print("Generating queries...")

        results_by_doc: Dict[str, List[BenchmarkResult]] = {}
        placement_distribution: Dict[int, int] = {}
        distance_distribution: List[float] = []
        search_time_distribution: List[float] = []

        for source, queries in queries_by_doc.items():
            results_by_doc[source] = []
            for query in queries:
                start_time = time.time()
                search_results = self.search(queries=[query])
                end_time = time.time()

                time_to_search = end_time - start_time
                search_time_distribution.append(time_to_search)

                found_in_results = False
                for i, res in enumerate(search_results):
                    if res.get("id") == source:
                        placement = i + 1
                        distance = res["distance"]
                        result = BenchmarkResult(
                            query=query,
                            expected_source=source,
                            placement_order=placement,
                            distance=distance,
                            time_to_search=time_to_search,
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
                        query=query, expected_source=source, found=False
                    )
                    results_by_doc[source].append(result)

        percent_in_top_k = {}
        for k in top_k_values:
            count = sum(
                1 for placement in placement_distribution.keys() if placement <= k
            )
            total_queries = sum(len(q) for q in queries_by_doc.values())
            percent_in_top_k[k] = (
                (count / total_queries) * 100 if total_queries > 0 else 0
            )

        return BenchmarkRunResults(
            results_by_doc=results_by_doc,
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
