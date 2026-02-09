# Summary: Embeddings in NLP (Traditional → Neural → Transformer)

## 1. What is an Embedding?
An **embedding** is a numerical (dense vector) representation of a discrete object (e.g., a word) that captures useful information for machine learning models.

- Purpose: Convert symbols → numbers
- Meaning is encoded in **geometry (distances & directions)**, not individual values

---

## 2. What is a Word Embedding?
A **word embedding** maps each word to a vector such that:
- Semantically similar words are close in vector space
- Dissimilar words are far apart

Examples:
- Word2Vec
- GloVe
- FastText

---

## 3. Embedding Dimension: Minimum and Maximum

### Minimum
- **Theoretical minimum:** 1 (not useful)
- **Practical minimum:** ~10–50 (toy or simple tasks)

### Maximum
- **Theoretical maximum:** Unlimited
- **Practical range:**
  - Static embeddings: 100–300
  - Transformers: 768–12,000+

Trade-off:
- Too small → insufficient capacity
- Too large → overfitting, inefficiency

---

## 4. Meaning of Each Number in an Embedding

Key truth:
> **Individual embedding dimensions have no fixed human-interpretable meaning.**

- Embeddings are **distributed representations**
- Meaning emerges from:
  - Distances
  - Angles
  - Directions between vectors
- Dimensions are **rotation-invariant** and arbitrary

Meaning lives in **relationships**, not coordinates.

---

## 5. How Are Embeddings Trained?

### Core principle
**Distributional hypothesis**:
> Words appearing in similar contexts have similar meanings.

### Neural embeddings (most common)
- Trained via **gradient descent**
- Objective: Predict context or next word
- Examples:
  - Word2Vec (Skip-gram, CBOW)
  - FastText
  - BERT, GPT

### Non-gradient embeddings
- LSA (SVD-based)
- Spectral embeddings
- TF-IDF (not really embeddings)

---

## 6. Traditional NLP vs Transformer NLP

### Traditional NLP
- Representations:
  - Bag-of-Words
  - TF-IDF
  - N-grams
- Position:
  - N-grams
  - Sliding windows
- Interaction:
  - Hand-crafted features
  - RNN/LSTM recurrence
- Pipelines were **modular and non end-to-end**

### Transformer NLP
- Token embeddings
- Positional embeddings
- Attention mechanism
- End-to-end learned representations
- Contextual embeddings

**Key difference:**
> Traditional NLP separates meaning, position, and interaction; transformers learn them jointly.

---

## 7. Are All Embeddings Trained with Gradient Descent?

**No.**

| Embedding Type | Gradient Descent |
|---|---|
| Word2Vec | Yes |
| FastText | Yes |
| GloVe | Yes |
| BERT / GPT | Yes |
| LSA | No |
| TF-IDF | No |
| Spectral embeddings | No |
| Random embeddings | No |

Modern NLP is dominated by gradient-based methods due to scalability and flexibility.

---

## 8. Embeddings in RNN / LSTM Models

Key point:
> **Embeddings are crucial in RNN/LSTM-based NLP models.**

- Embeddings provide semantic signal
- RNN/LSTM:
  - Models order
  - Aggregates information over time
- Cannot compensate for poor embeddings (GIGO principle)

Empirical finding:
- Pre-trained embeddings often contribute more than the RNN itself

---

## 9. Pre-trained Embeddings: Frozen vs Fine-Tuned

Pre-trained embeddings are **initializations**, not fixed by default.

### Option 1: Frozen
- No updates during training
- Good for small datasets
- Faster, more stable

### Option 2: Fine-tuned
- Updated via backpropagation
- Adapts to task/domain
- Risk of overfitting or forgetting

Best practice:
- Small data → freeze
- Larger or domain-specific data → fine-tune (often with smaller LR)

---

## 10. Big Picture Takeaways

- Embeddings turn language into geometry
- Individual dimensions are meaningless; geometry is everything
- Gradient descent dominates modern embedding learning
- In RNN/LSTM models, embeddings carry most semantic power
- Transformers reduce reliance on static embeddings via attention
- Pre-trained embeddings can (and often should) be fine-tuned

---

## One-Sentence Summary

**Embeddings are learned geometric representations of language; how powerful your NLP model is largely depends on how well those vectors encode meaning and context.**
