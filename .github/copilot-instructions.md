# `copilot-instructions` — ChromaDB Basics

Purpose: Give AI coding agents concise, actionable guidance to be productive in this repo.

- **Big picture:** This repository is primarily a hands-on demo of ChromaDB usage driven by a Jupyter notebook (`chroma_basic.ipynb`). It contains small helper artifacts (`main.py`, `pyproject.toml`, `environment.yml`) but no packaged library. Most experiments and canonical examples live in the notebook cells.

- **Key files to read first:**
  - [chroma_basic.ipynb](chroma_basic.ipynb) — primary examples for creating collections, adding documents, querying, and embedding backends (OpenAI, SentenceTransformers).
  - [pyproject.toml](pyproject.toml) and [environment.yml](environment.yml) — dependency lists and environment setup.
  - [main.py](main.py) — trivial entry point; not used for experiments.

- **Runtime & env patterns:**
  - Two supported environment workflows: Conda (`conda env create -f environment.yml`) and `uv sync` (see README). Use whichever the user prefers.
  - OpenAI usage expects an env var: `OPENAI_TOKEN` (not checked in). Notable code: notebook cells set `os.environ["OPENAI_API_KEY"] = openai_token` after loading `.env`.
  - Persistent Chroma DB paths are created under `./data/` (examples: `./data/my_chroma_db`, `./data/openai_chroma_db`). Prefer these folder patterns when changing or adding persistent examples.

- **Embedding backends observed:**
  - OpenAI embedding function: `embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")` (notebook).
  - SentenceTransformer: `paraphrase-MiniLM-L6-v2` via `embedding_functions.SentenceTransformerEmbeddingFunction` (notebook).

- **Project-specific conventions & patterns:**
  - Notebook-first: Additions, experiments, and examples should be added as new cells in `chroma_basic.ipynb` rather than creating multiple scattered scripts.
  - Use `client = chromadb.PersistentClient(path="./data/...")` for reproducible examples. Ephemeral examples may use `chromadb.Client()`.
  - When adding examples that use OpenAI, load keys from `.env` using `python-dotenv` as shown in the notebook; do not hard-code keys.
  - Avoid changing notebook cell IDs — edits should be limited to cell source content.

- **Build / run / debug commands:**
  - Create conda env: `conda env create -f environment.yml`
  - For uv workflow: `uv sync` (per README) to create the locked venv.
  - Run the notebook interactively in Jupyter / VS Code; attach the created env/kernel.
  - Quick script run: `python main.py` (prints a hello message).

- **Integration & external dependencies:**
  - `chromadb` for vector DB (core of examples).
  - `openai` package and `OPENAI_TOKEN` for using OpenAI embeddings.
  - `sentence-transformers`, `torch` for local embedding examples.

- **When changing examples, prefer:**
  - Add a new notebook cell demonstrating the change and include a short markdown explanation of why. Keep cells self-contained (imports + small snippet) so reviewers can run a single cell.
  - If adding a new script, update `README.md` with a one-line runnable example and mention required env vars.

- **What not to do:**
  - Don’t commit API keys or credentials.
  - Don’t reorganize into a package unless the user asks — this repo is intentionally a tutorial/demo set.

If anything here is unclear or you want additional detail (for instance, more examples of queries or a suggested test harness), tell me which part you want expanded and I'll iterate.
