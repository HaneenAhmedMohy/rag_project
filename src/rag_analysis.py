import time
from typing import List, Dict, Any
import os

# Assuming RAGSystem is already implemented and imported
from rag_system import RAGSystem

class RAGAnalyzer:
    def __init__(self):
        self.results = []

    def record(self, data: Dict[str, Any]):
        self.results.append(data)

    def run_chunk_size_experiment(self, chunk_sizes: List[int], test_queries: List[str]):
        print("\n--- Running Chunk Size Experiment ---")
        for size in chunk_sizes:
            print(f"\nTesting chunk size: {size}")
            os.environ["OPENAI_EMBED_MODEL"] = "mxbai-embed-large"
            rag = RAGSystem(chunk_size=size)
            chunks = rag.load_and_chunk_text("books/chapter3_4.txt")
            rag.build_index(chunks)

            for query in test_queries:
                result = rag.query(query, k=3)
                self.record({
                    "experiment": "chunk_size",
                    "chunk_size": size,
                    "query": query,
                    "retrieved_chunks": result.get("chunks"),
                    "similarity_scores": result.get("scores"),
                    "answer": result.get("answer"),
                })
                print(f"Query: {query}\nAnswer: {result.get('answer')}\n")

    def run_k_value_experiment(self, k_values: List[int], test_queries: List[str]):
        print("\n--- Running Retrieval Size (k) Experiment ---")
        os.environ["OPENAI_EMBED_MODEL"] = "mxbai-embed-large"
        rag = RAGSystem(chunk_size=500)
        chunks = rag.load_and_chunk_text("books/chapter3_4.txt")
        rag.build_index(chunks)

        for k in k_values:
            print(f"\nTesting k={k}")
            for query in test_queries:
                result = rag.query(query, k=k)
                self.record({
                    "experiment": "k_value",
                    "k": k,
                    "query": query,
                    "retrieved_chunks": result.get("chunks"),
                    "similarity_scores": result.get("scores"),
                    "answer": result.get("answer"),
                })
                print(f"Query: {query}\nAnswer: {result.get('answer')}\n")

    def run_model_comparison(self, test_queries: List[str]):
        print("\n--- Running Embedding Model Comparison ---")

        print("Building RAG system with bge-m3...")
        os.environ["OPENAI_EMBED_MODEL"] = "bge-m3"
        rag_bge = RAGSystem(chunk_size=500)
        chunks = rag_bge.load_and_chunk_text("books/chapter3_4.txt")
        rag_bge.build_index(chunks)

        print("Building RAG system with mxbai-embed-large...")
        os.environ["OPENAI_EMBED_MODEL"] = "mxbai-embed-large"
        rag_mxbai = RAGSystem(chunk_size=500)
        rag_mxbai.build_index(chunks)

        for query in test_queries:
            print(f"\nQuery: {query}")

            start_time = time.time()
            result_bge = rag_bge.query(query, k=3)
            time_bge = time.time() - start_time

            start_time = time.time()
            result_mxbai = rag_mxbai.query(query, k=3)
            time_mxbai = time.time() - start_time

            self.record({
                "experiment": "model_compare",
                "query": query,
                "bge": {
                    "retrieved_chunks": result_bge.get("chunks"),
                    "similarity_scores": result_bge.get("scores"),
                    "answer": result_bge.get("answer"),
                    "time": time_bge,
                },
                "mxbai": {
                    "retrieved_chunks": result_mxbai.get("chunks"),
                    "similarity_scores": result_mxbai.get("scores"),
                    "answer": result_mxbai.get("answer"),
                    "time": time_mxbai,
                },
            })

            print(f"bge-m3 Answer: {result_bge.get('answer')} (time: {time_bge:.4f}s)")
            print(f"mxbai Answer: {result_mxbai.get('answer')} (time: {time_mxbai:.4f}s)\n")

    def save_results(self, filename="results/experiment_results.json"):
        import json, os
        os.makedirs("results", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

    def generate_report(self):
        with open("README.md", "w") as f:
            f.write("# RAG Experiments Report\n\n")
            f.write("Report will be filled after analyzing results.\n")


def main():
    analyzer = RAGAnalyzer()
    test_queries = [
        "What is the difference between instruct models and chat models?",
        "How does temperature affect model outputs?",
        "What is the ChatML format?",
        "Explain the RLHF training process",
        "What are system messages used for?",
    ]

    print("\n=== Experiment 1: Chunk Size Impact ===")
    analyzer.run_chunk_size_experiment([300, 500, 1000], test_queries)

    print("\n=== Experiment 2: Retrieval Size (k) Impact ===")
    analyzer.run_k_value_experiment([1, 3, 5, 10], test_queries)

    print("\n=== Experiment 3: Embedding Model Comparison ===")
    analyzer.run_model_comparison(test_queries)

    analyzer.save_results()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
