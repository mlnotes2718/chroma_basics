# Transformer, Embeddings, and Vector Databases — Concise Summary Notes

## 1. Transformer Training (From Raw Text)

**Pipeline**


Raw text dataset
→ Tokenization (subword / partial-word)
→ Vocabulary (token ↔ ID)
→ Token IDs
→ Token embeddings + positional encoding
→ Transformer layers
→ Loss computation
→ Backpropagation
→ Updated model weights (saved as checkpoints)


**Key points**
- Raw training text comes from datasets, not vector databases
- Token embeddings are learned **model parameters**
- Attention (Q, K, V, O), FFN, LayerNorm are all trained via gradient descent
- Model weights are saved in model files (`.pt`, `.bin`, `.safetensors`)

---

## 2. Model Parameters After Training

A trained transformer contains:
- Token embedding matrix
- Attention weights (WQ, WK, WV, WO)
- Feed-forward network weights
- LayerNorm scale (γ) and bias (β)
- (Optional) learned positional embeddings

These are **not stored in a vector DB**.

---

## 3. Inference vs Fine-Tuning

**Inference**
- Same tokenizer and model
- Forward pass only
- No loss, no backprop, no weight updates
- Dropout OFF

**Fine-tuning**
- Same tokenizer and model
- Forward pass + loss
- Backpropagation ON
- Weights updated

**Rule**
> Inference = training forward pass without learning

---

## 4. Shapes (Simple Example)

Assume:
- 10 tokens
- Embedding dimension = 3

**Token embeddings**


Shape: (10 × 3)


**Self-attention (single head)**
WQ, WK, WV : (3 × 3)
Q, K, V : (10 × 3)
Attention : (10 × 10)
WO : (3 × 3)


**Feed-forward network**
W1 : (3 × d_ff)
W2 : (d_ff × 3)


**LayerNorm**
γ, β : (3,)


---

## 5. Tokenization & Model Sharing

To reproduce exact inference, you must provide:
- Tokenization algorithm
- Vocabulary and token-ID mapping
- Model architecture
- Model weights

All four are mandatory.

---

## 6. What a Vector Database Is For

Vector DBs store:
- Semantic embeddings of text chunks
- Document or chunk IDs
- Metadata

Vector DBs do **not** store:
- Model weights
- Token embeddings
- Attention parameters
- Training text for model learning

---

## 7. Embeddings Stored in a Vector DB

Type of embeddings:
- Sentence embeddings
- Paragraph embeddings
- Document embeddings

Creation:

Text chunk
→ Embedding model
→ One fixed-length vector
→ Stored in vector DB


Token embeddings are intermediate and never stored.

---

## 8. Pretrained Embeddings and Vector DBs

- Pretrained embedding model = function
- Vector DB = storage + similarity search
- You still generate embeddings before storing them
- The embedding model itself is not stored in the vector DB

---

## 9. Common Misconceptions (Corrected)

❌ Store raw training text in vector DB for transformer training  
❌ Write learned token embeddings back into vector DB  
❌ Confuse token embeddings with document embeddings  
❌ Think vector DBs are required to train transformers  

---

## 10. Final Mental Model


Transformer model → learns language patterns
Vector database → retrieves relevant knowledge

They are **architecturally separate** and serve **different purposes**.

