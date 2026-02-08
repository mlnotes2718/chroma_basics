# Transformer & Vector DB — Exam Flash Notes

## 1. Transformer Training
- Input: raw text (datasets, files)
- Steps:
  - Tokenization (subword / partial word)
  - Build vocab (token ↔ ID)
  - Token IDs → embeddings
  - Transformer forward pass
  - Loss → backprop → update weights
- Vector DB: **NOT used**

---

## 2. What Is Learned
- Token embedding matrix
- Attention weights: **WQ, WK, WV, WO**
- Feed-forward weights
- LayerNorm parameters (γ, β)
- (Optional) positional embeddings

All are **model parameters**, saved in model checkpoints.

---

## 3. Inference vs Fine-Tuning
- Same tokenizer + model
- Inference:
  - Forward pass only
  - No loss, no backprop
- Fine-tuning:
  - Forward pass + loss
  - Backprop updates weights

**Rule:** Training = inference + learning

---

## 4. Shapes (Example)
Assume:
- Tokens = 10
- Embedding dim = 3

- Token embeddings: **(10 × 3)**
- WQ, WK, WV, WO: **(3 × 3)**
- Q, K, V: **(10 × 3)**
- Attention matrix: **(10 × 10)**
- FFN: (3 → d_ff → 3)
- LayerNorm γ, β: **(3,)**

---

## 5. Model Sharing Requirements
To
