# ChromaDB Basics

This repository demonstrate the basic usage of Chromadb. Chromadb is the basic vector database design for small to medium scale LLM processing.

This repository supports conda environment with pip and also support a separate pure uv implementation.

> **Note: Pytorch also stop supporting Intel Mac**
> **Please run on Linux container for both uv and conda users.**

## UV Deployment
- The file `.python-version` locks in Python 3.11
- `pyproject.toml` specifies the requirements.
- `uv.lock` contain the exact version and it dependencies.
- You can regenerate `uv.lock` with `pyproject.toml`

Use the following command to generate the `venv` environment:
```bash
uv sync
```

## Conda Deployment
Use the following command to create the conda environment:
```bash
conda env create -f environment.yml
```

## Running the Notebook
Attached the environment to the notebook and runs each cell.

- [ChromaDB Basics](notebooks/chroma_basic.ipynb)
- [ChromaDB Complete Guide](notebooks/chromadb_complete_guide.ipynb)
- [Embedding](notebooks/embedding.ipynb)

### Hugging Face API Token
You will need hugging face API token to download the default embedding. Please create a dotenv file with the follow:

```text
HF_TOKEN='hf_xxxx'
OPENAI_API_KEY='sk-xxx'
```

### OpenAI API
OpenAI API is required to run the code with openai embedding. Please change to other embeddings if you do not have OpenAI API token.

## Documents and Guides
This README.md and [NLP Key Takeaways](NLP_Key_Takeaways.md) is in the root folder. The rest of detailed guides (mostly LLM generated) are in the docs folder.
