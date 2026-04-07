# NLP Lesson Key Takeaways

---

## 1. Model Performance

- **Transformer-based architectures** outperform traditional NLP methods (e.g., TF-IDF, basic word embeddings)
- However, we need to balance **use case requirements with computing costs** when selecting technologies

---

## 2. The Word "Embedding" — Two Very Different Meanings

This is one of the most common sources of confusion in NLP. The word **"embedding"** is used in two distinct ways:

### A) Embedding as a Model Parameter (Training Time)

When people talk about **word embeddings** in the context of word2vec, or **token embeddings** in the context of LLMs like BERT, they are referring to **weight matrices learned during training**.

- These are **parameters inside the model** — they are updated via backpropagation during training
- After training, they are saved as part of the **model's weight files** (e.g., `.bin` or `.safetensors`)
- They live **inside the model**, not in a vector database

**word2vec is a special case:** The entire purpose of training word2vec is to learn word vectors — so the trained weights *themselves* are the word embeddings. You extract them directly from the model and use them as vector representations of words.

**BERT/LLMs are different:** The token embedding layer is just the *first step* inside the model. Input tokens are converted to initial vectors by this layer, but those vectors are then passed through many transformer layers (attention, feed-forward, etc.). The token embedding weights are internal parameters, not the final output you use.

### B) Embedding as an Inference Output (Inference Time)

When people talk about embeddings in the context of **RAG systems and vector databases**, they mean something different — **vector representations produced by passing data through an already-trained model**.

- You take a trained model (e.g., BERT, sentence-transformers)
- You feed it text at **inference time**
- The model **outputs a vector** representing that text
- That output vector is what gets stored in the vector database

This is fundamentally different from training parameters — the model is frozen, and you are simply using it to *produce* outputs.

### Summary Table

| | word2vec Embeddings | BERT Token Embedding Layer | Vector DB Embeddings |
|---|---|---|---|
| **What are they?** | Trained weight vectors | Internal weight matrix | Inference-time output vectors |
| **When are they created?** | During training | During training | During inference |
| **Where do they live?** | Inside / extracted from model | Inside the model | In a vector database |
| **Are they model parameters?** | Yes | Yes | No — they are outputs |

> **Key reminder:** The word "embedding" can mean a *learned model parameter* (word2vec, token embedding layer) OR an *inference-time output vector* (stored in a vector DB). These are not the same thing, even though both are called embeddings.

---

## 3. Vector Databases & RAG Systems

### RAG Flow

A typical RAG pipeline works in two phases:

**Indexing (offline):**
1. Load documents and split them into chunks
2. Pass each chunk through an embedding model to produce a vector
3. Store the vectors (with the original text) in a vector database

**Retrieval & Generation (at query time):**
1. Embed the user's query using the same embedding model
2. Search the vector database for the most similar chunks (via cosine similarity or euclidean distance)
3. Pass the retrieved chunks + the original question to an LLM
4. The LLM generates an answer grounded in the retrieved context

### What a Vector Database Stores

A vector database can store **any kind of vector embedding** — text, sentence, document, image, audio, etc. What is always true is that **everything stored is an inference-time output** — a vector produced by a trained model. Training parameters (model weights) are never stored in a vector database.

### Chunking

- **Chunking** is the standard approach — splitting documents into smaller pieces before embedding
- The **chunking strategy and settings** (chunk size, overlap, splitting logic) matter more than which library you use; a poorly tuned chunker will hurt retrieval regardless of the tool
- Chunks are preferred over sentence-level embeddings because they provide **richer contextual information** across multiple sentences
- Document-level embeddings are less common, though some systems use them for a broad first-pass retrieval before switching to chunk-level precision

### Embedding Models

- The choice of embedding model affects retrieval quality — different models have different strengths and token limits
- **ChromaDB's default embedding model** (all-MiniLM-L6-v2) runs locally and is fast, but has a token limit (~256 tokens); chunks exceeding this will be silently truncated
- **OpenAI's embedding models** (e.g., text-embedding-3-small) run remotely — good quality, but require an API call per chunk, which raises **latency, cost, and privacy concerns** (your data leaves your environment)
- Beyond those tradeoffs, the quality difference between modern embedding models is modest for most tasks — the bigger wins usually come from chunking strategy and retrieval tuning

### Similarity Search

Once embeddings are stored, retrieval uses distance/similarity metrics to find semantically relevant chunks:

| Metric | Description | Common Use |
|---|---|---|
| **Cosine Similarity** | Measures the angle between two vectors | Most common for text — invariant to vector magnitude |
| **Dot Product** | Similar to cosine but considers magnitude | Used when magnitude carries meaning |
| **Euclidean Distance (L2)** | Straight-line distance between two vectors | Less common for text |

In practice, **cosine similarity and euclidean distance produce similar rankings** for normalized text embeddings — the difference in retrieval quality is generally small. Cosine similarity is the default choice because text embeddings are typically normalized.

### Choice of LLM

- **Any LLM can serve as the generation component** in a RAG system
- The LLM only receives a small, focused context window — the retrieved chunks plus the question — so it does not need to "know" the source documents
- This means you can swap in different LLMs (local or hosted) without changing your retrieval pipeline, as long as the context fits within the model's context window

---

## 4. Model Sharing

To share a custom trained model, you need to provide:
- **Model architecture** — the structure and layers of the model
- **Trained weights** — the learned parameters, including the token embedding layer

You do **not** need to share the vector database contents — those are inference-time outputs that can be regenerated by running the model on your data again.

---

## Summary Notes

- The word **"embedding"** has two meanings in NLP — don't confuse them:
  1. **Embedding as parameter** — e.g., word2vec word vectors, BERT's token embedding layer weights. Learned during training, stored inside the model.
  2. **Embedding as output** — vectors produced at inference time and stored in a vector database. Generated *by* the model, not *part of* the model.
- **word2vec is a special case** where the training goal is specifically to learn word vectors. BERT uses its token embedding layer only as a first internal step.
- **Vector databases store inference-time outputs only** — never training parameters.
- **Chunking strategy and settings** are more impactful than the choice of chunking library.
- **Embedding model choice** involves a tradeoff between local (fast, private, token-limited) and remote (higher quality, but latency and privacy costs).
- **Cosine similarity vs. euclidean distance** — both work; the practical difference in retrieval quality is small for normalized embeddings.
- **Any LLM can be used** in a RAG system — it only sees the retrieved chunks and the question, not the full knowledge base.
