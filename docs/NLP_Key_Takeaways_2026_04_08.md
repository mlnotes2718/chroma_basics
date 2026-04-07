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

### What a Vector Database Stores

A vector database can store **any kind of vector embedding** — text embeddings, sentence embeddings, document embeddings, image embeddings, audio embeddings, etc. The type of data is not the defining characteristic.

What *is* always true is that **everything stored in a vector database is an inference-time output** — a vector produced by passing data through a trained model. Training parameters (model weights) are never stored in a vector database; they live inside the model files themselves.

### Chunking

- **Chunking** is the standard approach in RAG systems — splitting documents into smaller pieces before generating embeddings
- Preferred over sentence-level embeddings because chunks provide **contextual information across multiple sentences**
- Document-level embeddings are less common, though some systems use them for an initial broad retrieval step before switching to chunk-level precision

### How Similarity Search Works in a Vector DB

Once embeddings are stored in a vector database, they are used for **semantic similarity search** — finding content that is semantically close to a query. This is done using distance/similarity metrics:

| Metric | Description | Common Use |
|---|---|---|
| **Cosine Similarity** | Measures the angle between two vectors | Most common for text — invariant to vector magnitude |
| **Dot Product** | Similar to cosine but considers magnitude | Used when magnitude carries meaning |
| **Euclidean Distance (L2)** | Straight-line distance between two vectors | Less common for text |

**Cosine similarity** dominates text use cases because:
- It only cares about the *direction* of vectors, not their magnitude
- Text embeddings are typically normalized, making cosine similarity very effective
- It correlates well with semantic similarity in language

The broader purpose of embeddings in a RAG system is:
1. **Similarity search** — find chunks semantically close to the query
2. **Retrieval** — return the most relevant chunks as context for the LLM
3. **Ranking / filtering** — some systems do a broad initial retrieval then re-rank results

---

## 4. Model Sharing

To share a custom trained model, you need to provide:
- **Model architecture** — the structure and layers of the model
- **Trained weights** — the learned parameters, including the token embedding layer

You do **not** need to share the vector database contents — those are inference-time outputs that can be regenerated by running the model on your data again.

---

## Summary Notes

- The word **"embedding"** has two meanings in NLP — don't confuse them:
  1. **Embedding as parameter** — e.g., word2vec word vectors, BERT's token embedding layer weights. These are learned during training and stored inside the model.
  2. **Embedding as output** — vectors produced at inference time and stored in a vector database. These are generated *by* the model, not *part of* the model.
- **word2vec is a special case** where the training goal is specifically to learn word vectors, so the weights themselves are the embeddings. In contrast, BERT uses its token embedding layer only as a first internal step.
- **Vector databases store all kinds of embeddings** (text, image, etc.), but they always store inference-time outputs — never training parameters.
- **Cosine similarity** is the most common metric used in vector databases for text similarity search.
