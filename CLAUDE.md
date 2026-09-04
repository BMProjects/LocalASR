# LocalASR Project Contract

## Purpose and environment

- Local-first speech recognition for subtitles, dictation, and meeting notes.
- Python >= 3.11; manage dependencies with `uv` only.
- The desktop may use audio/GUI extras; the compute node deliberately has no audio stack.

## Proportional verification

- Exploration, analysis, docs, and config-only work: inspect the result; no automatic tests or Ruff.
- Small code changes: run the nearest relevant test or focused smoke check.
- Use Ruff on changed Python files only when useful.
- Run `uv run pytest -x -q` for cross-module, regression-prone, or release work when full coverage is relevant.
- Do not mass-format the repository; the current full-tree Ruff baseline is not clean.

## Product invariants

- Audio and transcript content remain local unless the user explicitly changes an endpoint.
- A refined transcript is a proposal; preserve the original and the fidelity checks.
- Keep the engine boundary compatible with the documented OpenAI-style audio/chat APIs.
- Never expose node tokens or record credentials in docs, tests, logs, or commits.

## Working style

- Complete clear tasks autonomously within the requested scope.
- Use project context and judgment instead of adding approval gates for routine edits.
- Report validation that materially supports the result.
