# ChromaDB Basics

This repository demonstrate the basic usage of Chromadb. Chromadb is the basic vector database design for small to medium scale LLM processing.

This repository supports conda environment with pip and also support a separate pure uv implementation.

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

## OpenAI API
OpenAI API is required to run the code 
