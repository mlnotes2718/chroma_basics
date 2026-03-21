project_name := "chroma-basics"

# Returns 'uv' if the command is installed, otherwise 'conda'
env_type := `[ -n "${CONDA_PREFIX:-}" ] && echo "conda" || echo "uv"`

# Default: List commands
default:
    @just --list

# Setup the environment
setup:
    @echo "🚀 Using {{env_type}} environment..."
    @if [ "{{env_type}}" = "uv" ]; then \
        uv sync; \
        uv run pre-commit install; \
    else \
        pre-commit install --hook-type pre-commit --hook-type pre-push; \
    fi

## Start of development commands

# Check environment health
health:
    @echo "🩺 Checking environment health..."
    @if [ "{{env_type}}" = "uv" ]; then \
        uv pip check; \
        uv sync; \
    else \
        conda doctor; \
    fi

precommit:
    @echo "🔍  ({{env_type}}) Running pre-commit..."
    @if [ "{{env_type}}" = "uv" ]; then \
        uv run pre-commit run --all-files; \
    else \
        pre-commit run --all-files; \
    fi


# Remove build, cache, and coverage artifacts
clean:
    @echo "🧹 Cleaning up project..."
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf htmlcov
    rm -rf .mypy_cache
    rm -rf .ruff_cache
    rm -rf .hypothesis
    rm -rf ./data
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✨ Cleaned!"


# Run all checks
run: health clean
