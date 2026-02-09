# Model Switching in Vector Databases: Compatibility Guide

## The Core Question

**If we switch models in a vector database like ChromaDB, do existing document embeddings become useless?**

**Answer: Yes, absolutely. Switching models makes existing embeddings useless.**

This is a critical production gotcha that catches many developers.

---

## Why Embeddings Are Incompatible

### The Fundamental Problem

Different models create **completely different vector spaces**. The numbers are incomparable.

**Analogy**: Like switching from measuring in meters to measuring in pounds - the numbers mean completely different things.

### Example: Different Dimensions

```python
from chromadb.utils import embedding_functions

# Model A: MiniLM (384 dimensions)
model_a = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Model B: MPNet (768 dimensions)  
model_b = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

doc = "Machine learning is powerful"

emb_a = model_a([doc])[0]  # Shape: (384,)
emb_b = model_b([doc])[0]  # Shape: (768,)

# Can't even compute similarity - different dimensions!
```

### Example: Same Dimensions, Still Incompatible

Even if two models output the same dimension count, the vector spaces are **fundamentally different**:

```python
# Both output 384 dimensions
model_c = "all-MiniLM-L6-v2"  # 384 dims
model_d = "paraphrase-MiniLM-L3-v2"  # 384 dims

doc = "The cat sat on the mat"
emb_c = encode_with_model_c(doc)  # [0.12, -0.45, ...]
emb_d = encode_with_model_d(doc)  # [0.89, 0.23, ...]

# Dimensions match, but vectors point in completely different directions
similarity = cosine_similarity([emb_c], [emb_d])
# → 0.15 (basically random - no meaningful relationship)
```

**Better Analogy**: GPS coordinates vs street addresses. Both locate a place, but you can't mix them. "45.5° N, 122.6° W" and "123 Main St" might refer to the same location, but you can't do math between them.

---

## What Happens If You Mix Models

### Broken Search Scenario

```python
# Day 1: Index 1 million documents with Model A
collection = client.create_collection(
    name="docs",
    embedding_function=model_a  # MiniLM
)

for doc in million_docs:
    collection.add(documents=[doc], ids=[doc.id])

# Day 30: Switch to Model B
collection_new = client.get_collection(
    name="docs",
    embedding_function=model_b  # MPNet - WRONG!
)

# User searches
query = "artificial intelligence"
results = collection_new.query(
    query_texts=[query],  # Encoded with Model B
    n_results=10
)

# Results are GARBAGE because:
# - Query embedded with Model B
# - Documents embedded with Model A  
# - Comparing apples to oranges
```

### Concrete Broken Example

```python
# Original documents embedded with Model A
docs = [
    "Python programming tutorial",
    "Machine learning basics", 
    "Cooking pasta recipes"
]

# Stored embeddings (Model A space)
stored_embeddings_A = [
    [0.1, 0.8, 0.2, ...],  # Python doc
    [0.2, 0.7, 0.3, ...],  # ML doc
    [0.9, 0.1, 0.1, ...]   # Cooking doc
]

# Query with Model B (WRONG - different space)
query = "Python tutorial"
query_emb_B = [0.5, 0.3, 0.9, ...]  # From Model B

# Search results
similarities = cosine_similarity([query_emb_B], stored_embeddings_A)
# → [0.23, 0.61, 0.85]  

# WRONG! Cooking doc ranks highest instead of Python doc!
```

The search is completely broken because the vector spaces don't align.

---

## The Solution: Re-embed Everything

If you switch models, you **must re-embed all documents** from scratch.

### Migration Process

```python
# Step 1: Get all documents (text, not embeddings)
old_collection = client.get_collection("docs")
all_docs = old_collection.get()

documents = all_docs['documents']
ids = all_docs['ids']
metadatas = all_docs['metadatas']

# Step 2: Delete old collection
client.delete_collection("docs")

# Step 3: Create new collection with new model
new_collection = client.create_collection(
    name="docs",
    embedding_function=model_b  # New model
)

# Step 4: Re-add all documents (they'll be re-embedded automatically)
new_collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

# Now search works correctly
```

### Migration Costs

**Example: 10 million documents**
- **Re-embedding time**: Hours to days (depending on compute)
- **API costs**: If using OpenAI, potentially $1000s
- **Downtime**: Your search is unavailable during re-indexing
- **Storage**: Need space for both old and new temporarily

---

## Exceptions: When You CAN Switch Models

### 1. Same Model Family with Dimension Control

Some models are explicitly designed to be compatible:

```python
# OpenAI's embedding models (v3 family)
# You can use dimension parameter to make them compatible

emb_small = openai.embeddings.create(
    input="text",
    model="text-embedding-3-small",
    dimensions=1536  # Native size
)

emb_large = openai.embeddings.create(
    input="text", 
    model="text-embedding-3-large",
    dimensions=1536  # Reduced from native 3072
)

# Now they're in the same vector space and comparable
```

**Reference**: OpenAI Embeddings Documentation (2024)

### 2. Careful Model Fine-tuning

If you fine-tune from a checkpoint, embeddings **might** remain compatible:

```python
# Base model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Light fine-tuning on your data
finetuned_model = train(base_model, your_data, epochs=1)

# Old embeddings might still work (not guaranteed)
# Safer to re-embed, but emergency fallback exists
```

**Risk**: Even small fine-tuning can shift the vector space enough to degrade quality significantly.

---

## Best Practices

### 1. Always Store Original Text

**Critical**: Store the original document text, not just embeddings.

```python
# CORRECT
collection.add(
    documents=[doc.text],  # Store text - enables re-embedding
    ids=[doc.id],
    metadatas=[{"title": doc.title}]
)

# WRONG - don't do this
collection.add(
    ids=[doc.id],
    embeddings=[pre_computed_embedding]  # Can't re-embed later!
)
```

If you only store embeddings, you're locked into that model forever.

### 2. Version Your Model

Track which model created which embeddings:

```python
collection.add(
    documents=[doc],
    ids=[doc.id],
    metadatas=[{
        "model": "all-MiniLM-L6-v2",
        "model_version": "v1.0",
        "embedded_at": "2024-02-07"
    }]
)
```

This enables partial migrations and debugging.

### 3. Test Before Full Migration

```python
# Create test collection with new model
test_collection = client.create_collection(
    name="docs_test",
    embedding_function=new_model
)

# Sample subset of documents
sample_docs = random.sample(all_docs, 1000)
test_collection.add(documents=sample_docs, ...)

# Run evaluation queries
for query in test_queries:
    old_results = old_collection.query(query_texts=[query])
    new_results = test_collection.query(query_texts=[query])
    
    # Compare quality metrics (Recall@K, MRR, etc.)
    quality_improvement = evaluate_results(old_results, new_results)

# Only proceed if new model is measurably better
if quality_improvement > threshold:
    proceed_with_migration()
```

### 4. Gradual Migration Strategy (Advanced)

For large production deployments:

```python
# Maintain both collections temporarily
old_collection = client.get_collection("docs_v1")
new_collection = client.create_collection(
    name="docs_v2", 
    embedding_function=new_model
)

# Gradual re-embedding in background
batch_size = 1000
for i in range(0, total_docs, batch_size):
    batch = get_docs(i, i+batch_size)
    new_collection.add(documents=batch, ...)
    
    # Monitor progress
    log_progress(i, total_docs)

# A/B test search quality
route_percentage_to_new_collection(10%)  # Start with 10%

# Gradually increase traffic to new collection
# Once confident, switch 100% to v2
# Delete v1 after confidence period
```

### 5. Hybrid Search as Temporary Fallback

If you must search across different models temporarily:

```python
# Keyword search (BM25) doesn't depend on embeddings
results_keyword = keyword_search(query)

# Semantic search on new embeddings only
results_semantic = new_collection.query(query_texts=[query])

# Combine results (rerank or merge)
final_results = merge_and_rerank(results_keyword, results_semantic)
```

---

## Real-World Migration Example

### Case Study: OpenAI ada-002 → text-embedding-3-small

**Setup**:
- 10 million documents
- 1536 dimensions each
- Storage: 10M × 1536 × 4 bytes = ~61 GB

**Migration Costs**:
- Re-embedding cost: 10M tokens × $0.00002/token = **$200**
- Re-embedding time: **~8 hours** on GPU cluster
- Downtime: **4-6 hours** (overlap strategy)
- Quality improvement: **10-20% better retrieval** (measured on test set)

**Was it worth it?**
- For high-value applications: Yes
- For low-budget projects: Maybe not
- Depends on: traffic, revenue per query, error costs

**Reference**: Production system reports (OpenAI Community Forums, 2024)

---

## Decision Framework: Should You Switch Models?

### Calculate Total Cost

```
Total Cost = 
    Re-embedding API costs +
    Compute costs +
    Engineering time +
    Downtime costs
```

### Calculate Expected Benefit

```
Expected Benefit = 
    Quality improvement (%) × 
    Number of queries × 
    Value per improved query
```

### Example Calculation

```
Documents: 1M
Queries/month: 100K
Current quality (Recall@10): 75%
New model quality (Recall@10): 85%
Value per query: $0.10

Cost:
- Re-embedding: $20 (1M tokens @ OpenAI)
- Engineering: $2000 (2 days work)
- Total: $2020

Benefit per month:
- Quality improvement: 10%
- Improved queries: 100K × 10% = 10K
- Monthly value: 10K × $0.10 = $1000

ROI: Break even after 2 months
Annual benefit: $1000 × 12 = $12,000
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Storing Only Embeddings
```python
# Can't re-embed later!
collection.add(ids=[id], embeddings=[emb])
```

### ❌ Mistake 2: Assuming Same Dimensions = Compatible
```python
# Different models, same dims - NOT compatible
model_a_384d != model_b_384d
```

### ❌ Mistake 3: Not Testing on Real Queries
```python
# Generic benchmarks don't reflect YOUR use case
# Always test on your actual queries
```

### ❌ Mistake 4: Switching Without Cost Analysis
```python
# "New model is better" ≠ worth the migration cost
# Calculate ROI first
```

### ❌ Mistake 5: No Rollback Plan
```python
# What if new model is worse?
# Keep old collection until confident
```

---

## Summary: Key Takeaways

1. **Switching models = re-embed everything** - no shortcuts, no compatibility

2. **Different vector spaces** - models create incomparable embeddings, even with same dimensions

3. **Always store original text** - enables future re-embedding

4. **Test before migrating** - measure quality improvement on YOUR queries

5. **Calculate ROI** - re-embedding is expensive; ensure benefit justifies cost

6. **Gradual migration** - for production systems, migrate incrementally

7. **Version your models** - track what created what

8. **Rare exceptions exist** - same model family with dimension control (OpenAI v3)

9. **Hybrid search helps** - combine keyword + semantic during transition

10. **Plan model selection carefully** - switching later is costly

---

## The Bottom Line

**Model switching breaks all existing embeddings. You must re-embed from scratch.**

The only safe approach:
1. Store original text ✓
2. Test new model thoroughly ✓
3. Calculate migration ROI ✓
4. Re-embed all documents ✓
5. Validate search quality ✓

Choose your initial model carefully - changing it later is expensive and time-consuming.

---

## References

- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (ACL)
- OpenAI Embeddings Documentation (2024)
- Production system reports from OpenAI Community Forums (2024)
- ChromaDB Documentation on embedding functions

---

*Document created: February 2026*
