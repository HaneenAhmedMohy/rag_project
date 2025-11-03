# RAG System Experiment Report

## 📌 Executive Summary

This project explored the performance of a Retrieval-Augmented Generation (RAG) system using embedding-based document retrieval combined with a GPT-5 chat model. The experiments revealed that RAG quality is strongly dependent on chunking strategy, embedding model selection, and prompt structure. Proper preprocessing, retrieval tuning, and evaluation of retrieved context significantly improve final answer quality — while poor retrieval directly causes hallucinations or incomplete answers.

## 🔍 Detailed Findings

### ✅ Effective Configurations

| Category        | Best Performing Choice                 | Notes                                         |
| --------------- | -------------------------------------- | --------------------------------------------- |
| Embedding Model | `text-embedding-3-large`               | Improved semantic search accuracy             |
| Chunk Size      | ~500 tokens                            | Balanced context richness & relevance         |
| Retrieval       | Top-k = 3                              | Avoids irrelevant context noise               |
| Prompt Format   | ChatML structured system prompt        | Reduces hallucinations and improves grounding |
| Evaluation      | Manual qualitative + similarity scores | Ensures retrieved chunks are meaningful       |

### ❌ Bad Configurations & Effects

| Bad Practice                     | Result                                     |
| -------------------------------- | ------------------------------------------ |
| Very small chunks (100 tokens)   | Fragmented meaning, low accuracy           |
| Very large chunks (1000+ tokens) | Model ignores context due to overload      |
| Top-k > 5                        | Retrieval noise increases, answers degrade |
| Lack of retrieval check          | Hallucination likelihood increases         |



⚠️ Severe hallucination due to irrelevant chunk retrieval.

## ⚙️ Recommended RAG Configuration

```yaml
chunk_size: 500
chunk_overlap: 50
embedding_model: text-embedding-3-large
vector_metric: cosine
retriever: top_k
k: 3
rerank: false (optional future improvement)
prompt_template: ChatML system + context + query
```

## 💡 Key Insights About RAG

* Retrieval quality matters more than model size
* Good chunking = better semantic matches
* Always test retrieved context before answering
* Prompt format affects how well the model uses context
* RAG fails when retrieval is bad — not when the model is weak


🧠 **Conclusion:** Great RAG systems come from good retrieval pipelines, not only strong LLMs. Proper preprocessing, indexing, and evaluation dramatically improve factual accuracy and reliability.
