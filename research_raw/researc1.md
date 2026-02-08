# RAG System: Embeddings and Chunking - Complete Guide

## Core Concepts

### What are Embeddings?

**Embeddings** = Converting text into numerical vectors (arrays of numbers)

Example:
- "The cat sleeps" → [0.23, -0.45, 0.67, ..., 0.12] (768 numbers)
- "A feline rests" → [0.25, -0.43, 0.65, ..., 0.10] (similar vector)

**Purpose**: Enable semantic similarity matching (find similar meaning, not just keywords)

---

## Types of Embeddings

### 1. Token/Word Embedding
- **What**: Individual words or subwords
- **Example**: "bank" → [0.8, 0.1, 0.5]
- **When**: Building block for all other embeddings

### 2. Sentence Embedding
- **What**: Entire sentences as single vectors
- **Example**: "The cat sleeps" → [0.45, 0.62, 0.18]
- **When**: Comparing sentence-level meaning

### 3. Document Embedding
- **What**: Entire documents as single vectors
- **Status**: ❌ Rarely used in modern RAG (information gets diluted)

---

## Key Distinction: Sentence Embedding vs Chunking

### Sentence Embedding
- **Type**: The MODEL/PROCESS that converts text → vector
- **Tool**: SentenceTransformer, BERT, etc.
- **Can embed**: ANY text length (words, sentences, paragraphs, documents)

### Chunking
- **Type**: DATA PREPARATION strategy (splitting text)
- **Purpose**: Divide long documents into manageable pieces
- **Example**: Split 50-page doc → 100 chunks

**Relationship**: 
```
Text → Chunking (split) → Sentence Embedding (vectorize) → Store
```

**Important**: "Sentence embedding models" can embed ANY text, not just sentences!

---

## Why Chunking Instead of Sentence Splitting?

### Problem with Sentence Splitting

**Example:**
```
Sentence 1: "It was revolutionary."
Sentence 2: "The model achieved 95% accuracy."
Sentence 3: "This approach changed everything."
```

**Query**: "What was revolutionary?"

**Retrieved**: "It was revolutionary."

**Problem**: ❌ "It" = what? No context!

### Solution: Chunking (Group Sentences)
```
Chunk: "It was revolutionary. The model achieved 95% accuracy. 
This approach changed everything."
```

**Retrieved**: Full chunk with complete context ✅

### Four Reasons for Chunking

1. **Pronoun Resolution**: "It", "This", "They" need context
2. **Complete Thoughts**: Multi-sentence ideas stay together
3. **Semantic Coherence**: Narrative flow preserved
4. **Better Embeddings**: More text = richer representation

**Rule of Thumb**: 3-4 sentences per chunk optimal

---

## Chunking Strategies

### 1. Fixed-Size Chunking (Most Common)
```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks
```

**Pros**: Simple, predictable
**Cons**: Can split mid-sentence
**Use**: General purpose

---

### 2. Sentence-Based Chunking
```python
import nltk

def sentence_chunk(text, sentences_per_chunk=3):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks
```

**Pros**: Respects sentence boundaries, coherent
**Cons**: Variable chunk sizes
**Use**: Clean, well-formatted text

---

### 3. Recursive Character Splitting (LangChain Default)
```python
# Tries separators in order: paragraph → sentence → word
separators = ['\n\n', '\n', '. ', ' ']
```

**Pros**: Preserves semantic boundaries, balanced
**Cons**: More complex
**Use**: Production systems (recommended)

---

### 4. Semantic Chunking (Advanced)
```python
# Split when topic changes (measured by embedding similarity)
# Computationally expensive but highest quality
```

**Pros**: Respects topic boundaries, best quality
**Cons**: Slow, variable sizes
**Use**: High-quality systems, diverse content

---

## Optimal Chunk Sizes

### Token-Based Sizing (Recommended)

| Model | Max Tokens | Recommended Chunk | Overlap |
|-------|-----------|------------------|---------|
| BERT-base | 512 | 256-384 tokens | 50-100 |
| Sentence-BERT | 512 | 128-256 tokens | 20-50 |
| Text-embedding-ada-002 | 8191 | 512-1024 tokens | 100-200 |

**Formula**:
- Chunk size = 50-75% of model's max context
- Overlap = 10-20% of chunk size

**Why Overlap?**

Without overlap:
```
Chunk 1: "...the main cause of climate change"
Chunk 2: "is carbon emissions from..."
```
❌ Thought is split!

With overlap:
```
Chunk 1: "...the main cause of climate change is carbon emissions..."
Chunk 2: "...climate change is carbon emissions from factories..."
```
✅ Complete thought in both chunks!

---

## Fine-Tuning Explained

### What is Fine-Tuning?

**Pre-training** (done by Google/OpenAI):
- Model learns general language understanding
- Trained on massive text (Wikipedia, books, web)

**Fine-tuning** (optional, done by you):
- Adapt model for specific task (sentence similarity)
- Trained on sentence pairs with similarity labels

### Fine-Tuning Process
```python
# Training data: sentence pairs with similarity scores
training_data = [
    ("AI is fascinating", "Artificial intelligence is interesting", 0.9),
    ("I love dogs", "Pizza is delicious", 0.1),
]

# Fine-tune model
model = SentenceTransformer('bert-base-uncased')
model.fit(training_data, epochs=3)
model.save('./fine-tuned-model')
```

**Result**: Model learns to make similar sentences closer in vector space

### When to Fine-Tune?

**YES if**:
- Domain-specific docs (medical, legal, technical)
- You have >1000 training pairs
- Quality is critical

**NO if**:
- General documents
- Limited data (<1000 docs)
- Quick prototype

---

## Storage: What Goes Where?

### MongoDB (Source of Truth)
```javascript
{
  "_id": "doc_001",
  "title": "Introduction to AI",
  "content": "Full document text...",
  "category": "education",
  "date": "2024-01-15"
}
```

### ChromaDB (Vector Database)
```python
{
  "id": "doc_001_chunk_0",
  "embedding": [0.23, -0.45, 0.67, ...],  # 768 numbers
  "metadata": {
    "parent_doc_id": "doc_001",
    "title": "Introduction to AI",
    "category": "education",
    "chunk_id": 0,
    "total_chunks": 5
  },
  "document": "First paragraph of text..."  # Original chunk text
}
```

### Model Files (Separate)
```
fine-tuned-model/
├── config.json
├── pytorch_model.bin  # 440 MB (model parameters)
└── tokenizer files
```

**Key Point**: 
- Model parameters ≠ Embeddings
- Model = The "function" that creates embeddings
- Embeddings = The "results" you cache in VectorDB

---

## Document Embedding vs Chunk Embedding

### Why Document Embeddings Are Problematic

**Problem**: Information dilution
```
50-page textbook:
- Chapter 1: History (10 pages)
- Chapter 2: Programming (15 pages)
- Chapter 3: Data structures (10 pages)
- Chapter 4: Algorithms (15 pages)

Document embedding: [0.34, 0.56, -0.23, ...]  # Averages ALL chapters
```

**Query**: "What is a binary search tree?" (Chapter 3, page 23)

**Issue**: 
- Answer is 1 page out of 50
- Document embedding averages 50 pages
- Specific info diluted by 49 irrelevant pages
- Match score is weak ❌

### Chunk-Based Solution
```
Chapter 3 split into 5 chunks:
- Chunk 1: Arrays and lists
- Chunk 2: Binary search trees ← MATCHES!
- Chunk 3: Hash tables
- Chunk 4: Graphs
- Chunk 5: Heaps
```

**Query**: "What is a binary search tree?"

**Retrieved**: Chunk 2 - precise answer ✅

---

## Standard RAG Architecture

### What's Stored in VectorDB: Chunks ONLY
```
VectorDB Contents:
├─ Chunk 1 embedding + metadata + text
├─ Chunk 2 embedding + metadata + text
├─ Chunk 3 embedding + metadata + text
├─ ... (thousands of chunks)
└─ Chunk N embedding + metadata + text

Document embeddings: NONE ❌
```

**Industry Standard**: 94% of RAG systems use chunk-only embeddings

**Frameworks**: LangChain, LlamaIndex, Haystack all default to chunks

---

## Complete RAG Pipeline

### Phase 1: Model Preparation (One-Time)
```python
# Option 1: Use pre-trained model (no fine-tuning)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Option 2: Fine-tune (optional)
training_pairs = create_training_data()  # From your domain
model.fit(training_pairs, epochs=3)
model.save('./fine-tuned-model')
```

### Phase 2: Document Processing
```python
from pymongo import MongoClient
import chromadb

# 1. Fetch documents from MongoDB
client = MongoClient('mongodb://localhost:27017/')
documents = client['my_db']['documents'].find({})

# 2. Chunk each document
all_chunks = []
for doc in documents:
    chunks = chunk_document(doc['content'], chunk_size=512, overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            'id': f"{doc['_id']}_chunk_{i}",
            'text': chunk,
            'metadata': {
                'doc_id': doc['_id'],
                'title': doc['title'],
                'chunk_id': i
            }
        })

# 3. Generate embeddings (one per chunk)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([c['text'] for c in all_chunks])

# 4. Store in ChromaDB
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.create_collection("chunks")

collection.add(
    ids=[c['id'] for c in all_chunks],
    embeddings=embeddings.tolist(),
    metadatas=[c['metadata'] for c in all_chunks],
    documents=[c['text'] for c in all_chunks]
)
```

### Phase 3: Query
```python
def rag_query(question):
    # 1. Embed question
    query_embedding = model.encode([question])[0]
    
    # 2. Search chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    # 3. Return top chunks
    return results['documents'][0]

# Usage
answer = rag_query("What is a binary search tree?")
# Returns: 3 most relevant chunks
```

---

## Key Takeaways

### Embeddings
1. **Token/Word**: Individual words → vectors
2. **Sentence**: Sentences → vectors (but can embed ANY text length)
3. **Document**: Full docs → vectors (❌ rarely used in RAG)

### Chunking
1. **Purpose**: Split long docs into context-rich pieces
2. **Why not sentences**: Too short, lack context, pronoun issues
3. **Optimal size**: 256-512 tokens with 10-20% overlap
4. **Strategy**: Recursive character splitting (LangChain default)

### RAG Architecture
1. **Storage**: Chunk embeddings ONLY (no document embeddings)
2. **Fine-tuning**: Optional but recommended for domain-specific content
3. **VectorDB**: Stores embeddings + metadata + original text
4. **MongoDB**: Keeps original full documents

### Mental Model
```
Documents (MongoDB)
    ↓
Chunking (split into pieces)
    ↓
Sentence Embedding Model (vectorize chunks)
    ↓
VectorDB (store chunk embeddings)
    ↓
Query → Retrieve relevant chunks → Return to user
```

---

## Common Misconceptions Clarified

❌ "Sentence embeddings can only embed sentences"
✅ Sentence embedding models can embed ANY text length

❌ "We need both document and chunk embeddings"
✅ Modern RAG uses chunk embeddings ONLY

❌ "Chunking = sentence splitting"
✅ Chunking groups multiple sentences for context

❌ "Fine-tuning is required"
✅ Fine-tuning is optional (recommended for domain-specific use)

❌ "Embeddings are stored in the model"
✅ Embeddings are stored in VectorDB; model stores parameters

---

## Recommended Starting Point
```python
# Simple, production-ready RAG setup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=['\n\n', '\n', '. ', ' ']
)
chunks = splitter.split_documents(documents)

# 2. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([c.page_content for c in chunks])

# 3. Store in VectorDB
chroma = chromadb.Client()
collection = chroma.create_collection("my_docs")
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings.tolist(),
    documents=[c.page_content for c in chunks]
)

# 4. Query
def search(query):
    query_emb = model.encode([query])[0]
    return collection.query(query_embeddings=[query_emb.tolist()], n_results=3)
```

**This is the standard RAG pattern used by 90%+ of production systems.**

---

## References

- BERT Paper: https://arxiv.org/abs/1810.04805
- Sentence-BERT: https://arxiv.org/abs/1908.10084
- LangChain Documentation: https://python.langchain.com/docs/modules/data_connection/
- LlamaIndex Chunking Guide: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
- Semantic Chunking: https://github.com/FullStackRetrieval-com/RetrievalTutorials