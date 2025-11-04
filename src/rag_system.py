"""
RAG System - Fully working for GPT-5 + Optomatica
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")


# ---------------- Helper functions ----------------

def get_embedding(text: str) -> List[float]:
    """Generate embedding for input text using the environment model"""
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def _normalize_vectors(np_array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(np_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np_array / norms


# ---------------- RAG Class ----------------

class RAGSystem:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.index = None
        self.chunks: List[Dict] = []
        self.index_path = Path("faiss_index.ivf")
        self.chunks_path = Path("chunks_metadata.json")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_dim = None

    def load_and_chunk_text(self, file_path: str) -> List[Dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        token_ids = self.tokenizer.encode(text)
        total_tokens = len(token_ids)

        chunks = []
        chunk_id = 0
        for i in range(0, total_tokens, self.chunk_size):
            chunk_text = self.tokenizer.decode(token_ids[i:i + self.chunk_size])
            chunks.append({
                "chunk_id": chunk_id,
                "source": str(path),
                "text": chunk_text
            })
            chunk_id += 1

        self.chunks = chunks
        return chunks

    def build_index(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("No chunks provided")

        embeddings = []
        for i, ch in enumerate(chunks):
            try:
                emb = get_embedding(ch["text"])
            except Exception as e:
                raise RuntimeError(f"Failed embedding chunk {i}: {e}")
            embeddings.append(emb)

        emb_array = np.array(embeddings, dtype="float32")
        self.embedding_dim = emb_array.shape[1]
        emb_array = _normalize_vectors(emb_array)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(emb_array)

        # Save index and chunks
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Index built and saved: {len(chunks)} vectors (dim={self.embedding_dim})")

    def load_index(self):
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError("Index or metadata missing")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedding_dim = self.index.d
        print(f"Loaded index from {self.index_path}, {len(self.chunks)} chunks")

    def get_top_k_similar(self, query: str, k: int = 3) -> List[Dict]:
        if self.index is None or not self.chunks:
            raise RuntimeError("Index not loaded")

        q_emb = get_embedding(query)
        q = _normalize_vectors(np.array(q_emb, dtype="float32").reshape(1, -1))
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "text": chunk["text"],
                "source": chunk["source"]
            })
        return results

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        context_chunks = context_chunks[:2]  # limit context to reduce tokens
        context_str = "\n\n".join([f"[Chunk {c['chunk_id']}]: {c['text']}" for c in context_chunks])

        prompt = (
            f"You are a helpful RAG assistant. Answer the user's question using only the context below.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\nAnswer:"
        )

        try:
            response = client.responses.create(
                model=CHAT_MODEL,
                input=prompt,
                max_output_tokens=1000  # increase from 300
            )

            # print("RAW GPT-5 response:", response)

            # Extract text
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()
            elif hasattr(response, "output") and len(response.output) > 0:
                parts = []
                for item in response.output:
                    if "content" in item and item.content is not None:
                        for c in item.content:
                            if c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
                answer = " ".join(parts).strip()
                if answer:
                    return answer

            return "⚠️ No response from GPT-5."

        except Exception as e:
            print("Error generating answer:", e)
            return "⚠️ Error generating answer."





    def query(self, question: str, k: int = 3) -> Dict:
        retrieved = self.get_top_k_similar(question, k)
        answer = self.generate_answer(question, retrieved)
        return {"answer": answer, "retrieved_chunks": retrieved}


# ---------------- Main ----------------

def main():
    rag = RAGSystem(chunk_size=500)

    try:
        if rag.index_path.exists() and rag.chunks_path.exists():
            rag.load_index()
        else:
            chunks = rag.load_and_chunk_text("books/chapter3_4.txt")
            print(f"Created {len(chunks)} chunks")
            rag.build_index(chunks)
    except Exception as e:
        print(f"Error preparing index: {e}")
        return

    test_queries = [
        "What is the difference between instruct models and chat models?",
        "How does temperature affect model outputs?",
        "What is the ChatML format?",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        try:
            result = rag.query(query, k=3)
        except Exception as e:
            print(f"Error during query: {e}")
            continue

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n[{i}] Similarity: {chunk['score']:.4f}")
            print(f"Text: {chunk['text'][:200]}...")
        print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
