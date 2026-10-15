# Recording the paperBoy demo cast

This is a step-by-step script for recording the asciinema demo that the README embeds.

## Why this is a manual step

An agent can't produce a genuine cast. A real recording needs Fahad's actual terminal, the
RTX 4050, a running Ollama with the models pulled, and a live arXiv fetch. The whole point
is that it's real output, not a staged transcript. So this file is a script for a human to
run on the right machine, not something to automate.

Record it on the machine that already has: Ollama running, `llama3.2:3b` and
`nomic-embed-text` pulled, and paperBoy installed in a venv.

## One-time setup

```bash
# install asciinema if you don't have it
pip install asciinema        # or: brew install asciinema  /  apt install asciinema

# make sure the models are already pulled so there is NO cold-load wait on camera
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# warm the model once (off camera) so the first summarize call is fast in the recording
paperboy summarize "warm up the model so the cast has no 75s cold load"
```

## Prep the terminal (before you hit record)

- Clear the screen: `clear` (or `cls` on Windows).
- Use a readable font size (16-18pt) and a reasonably small window
  (about 90x28 chars) so the SVG stays legible when embedded.
- Make sure `config.yaml` exists and has 3-4 good topics (copy from `config.example.yaml`).
- Keep it short: aim for 60-90 seconds total. Don't type slowly or over-explain.
- Use a short, clean shell prompt if you can (long paths clutter the frame).

## Record

Start the recording with an idle time limit so long pauses (model thinking) are trimmed:

```bash
asciinema rec paperboy-demo.cast --idle-time-limit 2 --title "paperBoy demo"
```

Then run this exact sequence. Pause a beat between commands so the viewer can read:

```bash
# 1. show the CLI surface (fast)
paperboy --help

# 2. confirm the backend is healthy (fast; model is already warm)
paperboy doctor

# 3. the money shot: the full pipeline producing a real PDF
paperboy run --top 3

# 4. show the artifact that was written
ls digests/
```

On Windows use `.\.venv\Scripts\paperboy.exe` in place of `paperboy`, and `dir digests\`
in place of `ls digests/`.

Notes for a tight cast:
- `--top 3` keeps the summarize loop short (3 LLM calls) so `run` finishes on camera.
- `doctor` and `--help` are near-instant and establish what the tool is.
- Stop recording right after the digest path prints and you've shown the file. Type `exit`
  (or Ctrl-D) to end the cast.

## Upload and embed

Option A - host on asciinema.org (simplest):

```bash
asciinema upload paperboy-demo.cast
```

Copy the URL it prints (e.g. `https://asciinema.org/a/XXXXXX`). Then add to the top of the
README, just under the title/badge:

```markdown
[![asciicast](https://asciinema.org/a/XXXXXX.svg)](https://asciinema.org/a/XXXXXX)
```

That renders a clickable SVG thumbnail that opens the full player.

Option B - keep the cast in-repo (no external host):

1. Commit the file, e.g. `docs/paperboy-demo.cast`.
2. Convert to an animated SVG with svg-term-cli and commit that too:
   ```bash
   npm install -g svg-term-cli
   svg-term --in docs/paperboy-demo.cast --out docs/paperboy-demo.svg --window
   ```
3. Embed the SVG in the README:
   ```markdown
   ![paperBoy demo](docs/paperboy-demo.svg)
   ```

Option A is recommended for the Phase 5 release; it needs no build step and the player lets
viewers copy/paste commands straight out of the cast.

## Final check

- Watch the cast back before uploading (`asciinema play paperboy-demo.cast`).
- Confirm no secrets, API keys, or private paths are visible in the frame.
- Confirm the PDF path really printed and the `ls`/`dir` showed the file.
