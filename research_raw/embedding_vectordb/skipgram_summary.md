# Skip-gram Word Embedding (Summary)

## Overview
Skip-gram is a word embedding model that learns vector representations of words by predicting **context words** given a **center word**. Meaning emerges from word co-occurrence, not from predefined linguistic rules.

---

## Toy Example
**Sentence:**  
“I like cats”

**Vocabulary:**  
- I  
- like  
- cats  

**Window size:** 1

**Training pairs (center → context):**
- like → I  
- like → cats  

---

## What Skip-gram Learns
Each word has **two vectors**:
- **Input vector** (used when the word is the center)
- **Output vector** (used when the word is a context)

---

## Core Computation
For a center word \(w_c\) and context word \(w_o\):

\[
\text{score}(w_c, w_o) = \mathbf{v}_{w_c} \cdot \mathbf{u}_{w_o}
\]

Scores are converted to probabilities using **softmax**:

\[
P(w_o | w_c) = \frac{e^{\text{score}}}{\sum e^{\text{scores}}}
\]

---

## Learning Intuition
- Words that appear together are **pulled closer** in embedding space
- Unrelated words are **pushed apart**
- Training repeatedly adjusts vectors to improve context prediction

---

## Result
After training:
- Words appearing together (e.g., *like*, *I*, *cats*) are **geometrically close**
- Embeddings capture semantic similarity through **distance and direction**

---

## Key Takeaways
- Word embeddings are **not semantic by design**
- Meaning **emerges from statistics**
- Skip-gram = *adjust vectors so dot products predict nearby words*

---

## References
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013  
- Goldberg & Levy, *word2vec Explained*, 2014  
- Jurafsky & Martin, *Speech and Language Processing*
