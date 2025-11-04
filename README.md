# RAG System Analysis Report

## Executive Summary

This analysis evaluated the impact of chunk size, retrieval size (k),
and embedding choice on RAG performance. Smaller chunks (300 tokens)
delivered more accurate and complete retrievals, while very large chunks
reduced recall. Optimal retrieval performance was observed with k=3, and
results consistently showed that embedding quality and granularity
significantly influence RAG accuracy.

## Experiment 1: Chunk Size Impact

### Methodology

The same knowledge source was indexed using chunk sizes of 300, 500, and
1000 tokens. Queries related to RAG concepts were executed and outputs
compared for accuracy and completeness.

### Results

-   **300 tokens**: Highest recall and consistent full responses.
-   **500 tokens**: Some missing answers (e.g., RLHF query).
-   **1000 tokens**: Missed answers more frequently and lost detail.

### Key Findings

-   Smaller chunks retrieve more specific and relevant content.
-   Larger chunks risk missing key information.
-   Mid-size chunks moderately reduce vector count but lower answer
    quality.

### Recommendation

**Use 300--400 token chunks** to balance granularity and retrieval
precision.

## Experiment 2: Retrieval Size (k) Impact

### Methodology

Indexed data was queried while varying k = 1, 3, 5, 10.

### Results

-   **k = 1**: Decent but limited recall; missed nuance.
-   **k = 3**: Best balance; complete and accurate answers.
-   **k = 5/10**: Noisy retrieval; some irrelevant info, occasional
    failure.

### Key Findings

-   Too few results = incomplete context\
-   Too many = noise and confusion

### Recommendation

**Use k = 3** for optimal retrieval accuracy.


### Example: Good Retrieval

**Query:** What is the difference between instruct models and chat
models?\
**Retrieved:** Clear definitions + ChatML explanation\
**Answer:** Instruct models: Trained on a mix of instruction-following and plain completion samples, so a prompt can be ambiguous—should the model elaborate (completion) or answer (instruct)? They operate on plain text without explicit role markers.

- Chat models: RLHF fine-tuned to complete chat transcripts annotated with ChatML, which uses explicit roles (system, user, assistant) and message boundaries. A system message sets the assistant’s behavior/persona, and the structure makes the expected assistant reply unambiguous, enabling clear, conversational instruction-following.


### Example: Poor Retrieval

**Query:** Explain the RLHF training process\
**Chunk size:** 500 tokens\
**Issue:** No answer returned --- large chunk caused context loss.

## Overall Recommendations

-   **Chunk size:** 300 tokens --- preserves detail
-   **k:** 3 --- balanced recall vs noise

## Insights Learned

-   RAG depends heavily on granularity --- too large chunks harm
    retrieval.
-   Retrieval count (k) must balance recall and noise.
-   Missing context often causes full answer failure vs partial
    degradation.

