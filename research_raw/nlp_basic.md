# NLP Basics Learning Summary

## Key Concepts Validation

### Tokenization
- **Correct**: Tokenization is text preparation, analogous to data cleaning and feature engineering for tabular data
- **NLTK**: Used for educational purposes, not in production
- **For LLMs**: Simpler tokenizers are typically used for LLM preparation

### spaCy Usage
- **Clarification**: spaCy is widely used in production for various NLP tasks:
  - Named entity recognition
  - Dependency parsing
  - Text preprocessing pipelines
- Not just for "specific purposes" but a robust production tool
- For LLM preparation, simpler tokenizers are preferred

### Normalization (Stemming & Lemmatization)
- **Correct**: NOT used for preparing data for LLMs
- **Where it's used**: 
  - Search engines
  - Error correction systems
  - Log analysis
  - Rule-based NLP systems
- **Why not for LLMs**: Modern LLMs learn these patterns from raw text themselves

### N-grams
- **Traditional NLP**: Used for text search and classical approaches
- **For LLMs**: 
  - **Important clarification**: Modern LLM tokenizers (BPE, WordPiece) use subword n-grams internally during tokenization
  - N-grams are embedded in how LLMs tokenize, not added as a separate step
- **Timing**: Can be used before or during vectorization, depending on approach
  - For TF-IDF: n-gram ranges specified during vectorization

### TF-IDF
- **Correct**: Used as baseline NLP model for classical machine learning
- **Common use cases**:
  - Text classification with logistic regression
  - Naive Bayes models
  - Traditional ML approaches

## Overall Assessment
Mental model is solid for understanding the distinction between traditional NLP and modern LLM-based approaches.