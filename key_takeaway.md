# NLP Lesson Key Takeaways

## Model Performance
* Transformer-based architectures outperform traditional NLP methods (e.g., TF-IDF, basic word embeddings)
* However, we need to balance use case requirements with computing costs when selecting technologies

## Embeddings vs Model Parameters
* **Token embedding layer weights** are learnable model parameters that are updated during training
* **Generated embeddings** (vector representations of text produced at inference) are outputs, not parameters
* During training, the token embedding layer learns optimal representations; at inference, these learned weights transform input tokens into vector representations

## Vector Database & RAG Systems
* Embeddings are generated using a trained model at inference time, then stored in a vector database
* **Chunking** is the standard approach in RAG systems (more common than document-level embeddings)
* Chunking provides contextual information across multiple sentences, which is why it's preferred over sentence-level embeddings
* Document-level embeddings are less commonly used, though some systems employ them for initial retrieval before chunk-level precision

## Model Sharing
* To share a custom trained model, you need to provide:
  - **Model architecture** (the structure and layers)
  - **Trained weights** (the learned parameters, including the token embedding layer)
* Vector database contents (embeddings) don't need to be shared with the model itself, as they are outputs