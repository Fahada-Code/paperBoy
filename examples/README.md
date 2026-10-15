# paperBoy examples

A quick, copy-paste walkthrough to go from a fresh install to your first PDF
digest. Commands are PowerShell (Windows); on macOS / Linux use the POSIX venv
paths (`.venv/bin/paperboy`).

## Prerequisites

paperBoy's default backend is [Ollama](https://ollama.com) (free, no GPU). Once
Ollama is installed and running, pull the two models the example config uses:

```powershell
ollama pull llama3.2:3b         # summarizer
ollama pull nomic-embed-text    # embedding model for ranking
```

And install paperBoy (see CONTRIBUTING.md for the full dev setup):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 1. Copy the example config

paperBoy reads `config.yaml` from the current directory by default. Copy the
example into your project root:

```powershell
Copy-Item .\examples\config.yaml .\config.yaml
```

Open `config.yaml` and edit the `topics` list to match what you actually want to
keep up with.

## 2. Check the backend is reachable

```powershell
.\.venv\Scripts\paperboy.exe doctor
```

Expected output (Ollama up, model reachable):

```
Backend: ollama  model: llama3.2:3b  host: http://localhost:11434
OK - backend responded: 'pong'
```

If this fails, make sure Ollama is running and that you pulled the model named
in `config.yaml`.

## 3. Run the digest

```powershell
.\.venv\Scripts\paperboy.exe run
```

This fetches recent arXiv papers in your configured categories, ranks them
against your topics with embeddings, summarizes the top `top_n`, and writes the
brief. You'll see progress like:

```
Fetching up to 60 papers from arXiv ['cs.LG', 'cs.CL', 'cs.DC']...
  fetched 60 papers.
Ranking against 5 topics (embeddings)...
  selected top 8.
Summarizing...
  [1/8] <paper title>...
  ...
Digest written: digests\paperboy-2026-10-15.pdf
```

The result is a **PDF brief** under `.\digests\`. Open it and you're done.

## Useful flags

```powershell
# fewer papers, Markdown instead of PDF
.\.venv\Scripts\paperboy.exe run --top 5 --format markdown

# point at a config somewhere else
.\.venv\Scripts\paperboy.exe run --config .\examples\config.yaml

# summarize ad-hoc text (argument or piped stdin)
.\.venv\Scripts\paperboy.exe summarize "a long paragraph to condense"
```

Run `.\.venv\Scripts\paperboy.exe --help` to see every command.
