# LocalASR Agent Instructions

## Scope

This contract applies to the repository. More specific nested instructions override it.

## Environment and proportional verification

- Python >= 3.11; use `uv`, not conda or bare pip.
- Exploration, analysis, docs, and config-only work do not automatically require tests or Ruff.
- For small code changes, use the nearest relevant test or focused smoke check.
- Use Ruff on changed Python files only when useful.
- Use `uv run pytest -x -q` for cross-module, regression-prone, or release work when full coverage is relevant.
- Do not mass-format the repository; the full-tree Ruff baseline has existing drift.

## Architecture and safety

- Preserve the local-first privacy boundary and the original/refined transcript distinction.
- Desktop capture/GUI dependencies and compute-node dependencies are intentionally separate.
- Keep credentials and private transcript/audio content out of source, logs, and documentation.

## Working style

Complete clear tasks autonomously. Report only validation that materially supports the result.
