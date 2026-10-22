# Contributing to paperBoy

Thanks for your interest. paperBoy is a small, focused tool and we'd like to keep
it that way. This page covers how to set up, test, and send changes.

## What paperBoy is (and is not)

paperBoy is a **terminal-first research digest**: fetch -> rank -> summarize ->
PDF brief. Before you build, please keep these guardrails in mind:

- **Terminal-first. No web frontend, ever.** It's a CLI + config file + PDF/
  Markdown output. PRs that add a web UI, a hosted SaaS, or an onboarding flow
  are out of scope and will be declined.
- **The LLM summarizes and explains. It never predicts, advises, or recommends
  actions.** This is a research digest, not an advisor. Features that turn it
  into a recommender or predictor are out of scope.
- **Honesty in the repo.** Don't fake capabilities. Mark stubs as stubs. Don't
  add benchmark numbers that weren't measured on real hardware.

If you have an idea that stretches these lines, open an issue first so we can
talk about it before you spend time on a PR.

## Dev setup (Windows / PowerShell)

Python 3.11+ is required. We use the stdlib `venv` + `pip` (no `uv`).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`. (If you'd rather not use the extra, a plain
`pip install -e .` followed by `pip install pytest` works too.)

The optional `local` extra (`pip install -e ".[local]"`) adds the GPU serving
path (torch, transformers, bitsandbytes, FastAPI). Ollama users never need it.

On macOS / Linux the same commands work with the POSIX venv paths
(`.venv/bin/python`).

## Running the CLI

```powershell
.\.venv\Scripts\paperboy.exe --help
.\.venv\Scripts\paperboy.exe version
.\.venv\Scripts\paperboy.exe doctor      # checks the configured backend is reachable
```

See `examples/` for a ready-to-run config and a short walkthrough.

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Please add or update tests for any behavior you change. Keep logic in importable
modules (not in the CLI) so it stays testable.

## Code conventions

- **Type-hint public functions.** Keep signatures clear.
- **Keep the CLI thin.** `cli.py` should parse args and call into importable
  modules; put real logic in modules like `rank.py`, `render.py`, `sources/`,
  `backends/`. This keeps it testable.
- **Keep dependencies lean.** Prefer the stdlib and well-known libraries.
  Justify every new dependency in your PR description.
- **ASCII-only in console output.** Strings printed to the terminal must stay
  ASCII. The Windows code page renders em-dashes and other non-ASCII as `?`.
- **Commit messages.** Imperative subject line, and note the phase where it
  helps (e.g. "Phase 5: add CONTRIBUTING and examples"). Keep the body wrapped.

## Backends

LLM backends are pluggable and live in `src/paperboy/backends/`:

- `ollama` - the free, no-GPU default (and the easiest to run).
- `local` - the local quantized GPU path (the performance showcase).
- `hosted` - reserved, not implemented (paperBoy deliberately runs on your own hardware).

A backend implements the `Backend` ABC in `backends/base.py` and is wired into
the `get_backend` factory. If you add a backend, keep the same interface and add
a test.

## Issues and pull requests

1. Open an issue describing the bug or proposal (especially for anything that
   touches the guardrails above).
2. Fork the repo and create a topic branch off `main`.
3. Make your change, add tests, run `pytest` locally.
4. Open a pull request against `main` with a clear description and the reasoning
   for any new dependency.

Small, focused PRs get reviewed fastest. Thanks for contributing.
