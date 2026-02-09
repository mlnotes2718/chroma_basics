# Transformer, Attention, and Embeddings --- Summary

## 1. Tokenization

-   Text is split into tokens using a fixed tokenizer
    (BPE/WordPiece/Unigram).
-   Tokenizer is **not learned during model training**.
-   Example: "I love NLP" → \["I", "love", "NLP"\]

## 2. Token IDs

-   Tokens are mapped to integer IDs using a vocabulary.
-   IDs are indices, not semantic values.

## 3. Embedding Layer

-   First **learned** layer in a Transformer.
-   Maps token IDs to dense vectors.
-   Embeddings start as random values and are learned via gradient
    descent.
-   Embeddings encode meaning only through relationships, not individual
    dimensions.

## 4. Positional Encoding

-   Added to embeddings so the model knows token order.
-   Transformers have no inherent notion of sequence order.

## 5. Self-Attention

-   Each token attends to every other token in the same sequence.
-   Uses Query (Q), Key (K), and Value (V) projections.
-   Attention computes relevance via dot products, softmax, and weighted
    sums.
-   Produces context-aware token representations.

## 6. Difference from RNN/LSTM

-   RNNs process tokens sequentially and suffer from memory bottlenecks.
-   Attention allows parallel processing and direct long-range
    dependencies.
-   This makes Transformers more scalable and effective.

## 7. Training vs Inference

-   Forward pass (attention computation) is the same in training and
    inference.
-   Training additionally includes loss computation and backpropagation.
-   Inference uses frozen, learned parameters.

## 8. Gradient Descent and Learning

-   All learned parameters start randomly:
    -   Embeddings
    -   Q/K/V matrices
    -   Feedforward layers
-   Loss gradients adjust parameters to reduce prediction error.
-   Meaning emerges from repeated gradient updates over large data.

## 9. Gradient Flow Through Attention

-   Gradients flow through:
    -   Attention output
    -   Softmax weights
    -   Q/K/V projections
    -   Token embeddings
-   This teaches the model where to attend.

## 10. Multi-Head Attention

-   Multiple attention heads operate in parallel.
-   Each head has its own Q/K/V matrices.
-   Heads specialize via gradient pressure and competition.
-   Different heads capture different relationships (syntax, semantics,
    sentiment, etc.).
-   Specialization is emergent, not rule-based.

## 11. Embeddings and Similarity

-   Embedding space geometry is learned during pre-training.
-   Your corpus does not change the space; it only places points within
    it.
-   Similarity is measured between vectors using cosine similarity or
    dot product.
-   Individual dimensions are not interpretable.

## 12. Why Fine-Tuning Embeddings Is Hard

-   Requires high-quality contrastive data.
-   Changes affect the global semantic space.
-   Risk of catastrophic forgetting.
-   Hard to evaluate and often unnecessary.
-   Better alternatives: chunking, hybrid retrieval, reranking.

## Core Mental Models

-   Embeddings = learned coordinates in semantic space.
-   Attention = learned weighted averaging.
-   Training = changing parameters; inference = using them.
-   Multi-head attention = multiple learned perspectives.

## Key References

-   Vaswani et al., 2017 --- Attention Is All You Need
-   Jurafsky & Martin --- Speech and Language Processing
-   Goodfellow et al. --- Deep Learning
