# Fine-tuning the relevance judge (QLoRA, local)

paperBoy's relevance gate (`filter.py`) uses a local LLM to keep/drop papers. This
doc covers fine-tuning a small local model into a sharper, cheaper judge with
QLoRA, entirely on your own GPU (RTX 4050, 6 GB) - no hosted API.

> **Read this first.** Fine-tuning needs data. On a few dozen labels it overfits
> and won't beat the stock judge. Collect **several hundred** labels, ideally with
> **gold `paperboy rate` labels**, before you trust a tuned model. Always compare
> the tuned judge to the stock model on held-out data.

## 0. Collect data

Every `paperboy run` logs gate verdicts (weak labels). Bulk-generate more in one
sitting, and add your gold judgments:

```powershell
.\.venv\Scripts\paperboy.exe collect-labels --pool 150   # weak labels from the gate
.\.venv\Scripts\paperboy.exe rate                        # gold labels (interactive)
.\.venv\Scripts\paperboy.exe labels                      # how many you have
```

## 1. Export the training set

```powershell
.\.venv\Scripts\paperboy.exe export-train
# -> data/train/judge.train.jsonl, data/train/judge.val.jsonl
```

Each row is `{prompt, completion, label, source}`. The prompt states the task +
your topics + the paper; the completion is `VERDICT/REASON`. Gold ratings override
weak gate labels for the same paper.

Validate the data with no GPU / no ML deps:

```powershell
python -m paperboy.training.train_judge --train data\train\judge.train.jsonl --dry-run
```

## 2. Install the training deps

```powershell
pip install -e ".[train,local]"
# torch must come from the CUDA index (same as Phase 3 serving):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 3. Train (QLoRA)

```powershell
python -m paperboy.training.train_judge `
    --train data\train\judge.train.jsonl --val data\train\judge.val.jsonl `
    --base Qwen/Qwen2.5-1.5B-Instruct --out data\adapters\judge-lora
```

Why it fits 6 GB: 4-bit NF4 base weights (~1 GB), LoRA adapters only, gradient
checkpointing, paged 8-bit optimizer, batch 1 + gradient accumulation. A 1.5B base
is comfortable; a 7B base is tight - raise `--base` only if memory allows. These
are the same bitsandbytes/transformers mechanics as Phase 3's serving engine.

## 4. Merge, convert to GGUF, import into Ollama

Ollama runs GGUF models, so merge the adapter into the base, convert, and register:

```powershell
# merge LoRA into the base and save a full HF model (small helper script or peft merge_and_unload)
# then convert with llama.cpp:
python convert_hf_to_gguf.py data\adapters\judge-merged --outfile judge.gguf --outtype q4_K_M
```

Create a `Modelfile`:

```
FROM ./judge.gguf
PARAMETER temperature 0
```

```powershell
ollama create paperboy-judge -f Modelfile
```

## 5. Use it as the judge

Point the gate at the fine-tuned model (summaries stay on your main model):

```yaml
# config.yaml
judge_model: "paperboy-judge"
```

> **Deploy note.** The fine-tune is trained on a *no-few-shot* prompt (see
> `training/dataset.py`), since the model has internalized the task. `filter.py`
> currently always sends its few-shot prompt; to get the token savings, add a
> branch that drops the few-shot when a fine-tuned `judge_model` is set. Until
> then the tuned judge still works (it just also sees the few-shot).

## 6. Prove it helps

Before trusting the tuned judge, compare it to the stock model on held-out labels
(the same honesty bar the learned re-ranker was held to - which it failed). Re-run
`export-train`, keep the val split aside, and compare verdict accuracy. Ship only
if it clearly wins.
