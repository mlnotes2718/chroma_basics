# NLP Foundations Summary (2026)

This document summarizes the key NLP concepts discussed: **tokenization, NLTK, spaCy, stemming, lemmatization, n-grams, TF-IDF, and cosine similarity**, with a modern (2026) perspective.

---

## 1. Tokenization

**Tokenization** is the process of breaking raw text into smaller units called *tokens* (words, subwords, or characters).

- Example:  
  `"I love machine learning" → ["I", "love", "machine", "learning"]`

### Why it matters
- Models cannot process raw text, only numbers.
- Tokenization is the **first step** in any NLP pipeline.

### In modern LLMs
- LLMs (GPT, BERT, LLaMA) do **not** use word tokenization.
- They use **subword tokenization** (BPE, WordPiece, SentencePiece).
- Models predict the **next token**, not the next word.

---

## 2. NLTK vs spaCy (Production Reality)

### NLTK
- Educational and research-focused.
- Explicit and easy to understand.
- Rarely used in modern production systems.
- Still found in legacy pipelines and teaching.

### spaCy
- Built for speed and production.
- Widely used for:
  - Named Entity Recognition (NER)
  - Information extraction
  - Text classification
  - Pre/post-processing around LLMs
- Common in real-world NLP systems in 2026.

---

## 3. Stemming

**Stemming** is a rule-based process that removes suffixes to reduce words to a crude base form.

- Example:  
  `studies → studi`, `running → run`

### Purpose
- Reduce vocabulary size.
- Improve recall in keyword-based systems.

### 2026 usage
- ❌ Not used in LLMs or semantic systems.
- ✅ Still used in:
  - Search engines
  - Log analysis
  - Legacy ML pipelines

**Key idea:** Stemming is fast and cheap, but destroys meaning.

---

## 4. Lemmatization

**Lemmatization** converts words into their dictionary (base) form using linguistic rules.

- Example:  
  `better → good`, `running → run`, `was → be`

### Purpose
- Normalize text **without losing meaning**.
- Produce human-readable output.

### 2026 usage
- Used in:
  - High-quality search
  - Text analytics and reporting
  - Rule-based information extraction
- ❌ Not used inside LLMs.

**Key idea:** Lemmatization understands language; stemming does not.

---

## 5. N-grams

An **n-gram** is a contiguous sequence of *n tokens*.

- Unigram (1): `machine`
- Bigram (2): `machine learning`
- Trigram (3): `I love machine`

### Purpose
- Capture local word order and short phrases.

### Pipeline position
```
Raw Text → Tokenization → (Optional Normalization) → N-grams → Vectorization → Model
```

### 2026 usage
- Common in:
  - Search
  - Spam detection
  - Classical ML text classifiers
- Not explicitly used in LLMs (attention generalizes n-grams).

---

## 6. TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** measures how important a word is to a document relative to the corpus.

### Intuition
- Words frequent in one document but rare globally get higher scores.
- Common words ("the", "is") get downweighted.

### Components
- **TF:** How often a word appears in a document.
- **IDF:** How rare the word is across documents.

### What TF-IDF values mean
- Each document → a vector.
- Each number → importance of a word in that document.

### Still useful in 2026
- Search engines
- Document similarity
- Explainable ML baselines

---

## 7. Why Cosine Similarity with TF-IDF

After TF-IDF, documents are vectors in high-dimensional space.

**Cosine similarity** measures the angle between vectors:
- Focuses on **direction (topic)**, not magnitude (length).
- Ignores document length differences.

### Why not Euclidean distance?
- Penalizes longer documents even if content is similar.

### Key rule
> TF-IDF represents importance patterns; cosine similarity compares those patterns fairly.

Cosine similarity remains standard in both TF-IDF and embedding-based systems.

---

## 8. Big Picture (2026 Mental Model)

```
Tokenization → (Stem / Lemma) → N-grams → TF-IDF → Cosine Similarity
```

Modern systems often combine:
- **Classical NLP (TF-IDF, spaCy)** for speed and determinism
- **LLMs and embeddings** for semantics and reasoning

---

## One-Line Takeaway

> Classical NLP techniques are not obsolete in 2026 — they are **specialized tools** that complement LLMs when speed, cost, and interpretability matter.

