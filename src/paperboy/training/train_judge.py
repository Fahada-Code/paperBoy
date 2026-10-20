"""QLoRA fine-tune of a small local model into paperBoy's relevance judge.

Run as a script (not imported - keeps the heavy ML stack out of the CLI):

    # validate the exported data with no GPU / no ML deps:
    python -m paperboy.training.train_judge --train data/train/judge.train.jsonl --dry-run

    # real run (needs the [train] + [local] extras and a CUDA torch):
    python -m paperboy.training.train_judge \
        --train data/train/judge.train.jsonl --val data/train/judge.val.jsonl \
        --base Qwen/Qwen2.5-1.5B-Instruct --out data/adapters/judge-lora

Why these choices on a 6 GB RTX 4050: 4-bit NF4 quantization (bitsandbytes) keeps
the base weights ~1 GB; LoRA trains only small adapters; gradient checkpointing and
a paged 8-bit optimizer keep activation/optimizer memory down; a 1.5B base fits
comfortably (a 7B is tight - bump --base only if memory allows). This reuses the
exact bitsandbytes/transformers stack from Phase 3 serving.

HONESTY: fine-tuning needs real data. On a few dozen weak labels this will overfit
and won't beat the stock judge - collect several hundred labels (ideally gold
`rate` ones) first. Always compare the tuned judge to the stock model before
trusting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune the paperBoy relevance judge.")
    ap.add_argument("--train", required=True, help="Path to judge.train.jsonl (from `paperboy export-train`).")
    ap.add_argument("--val", default="", help="Optional judge.val.jsonl for eval during training.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF base model id.")
    ap.add_argument("--out", default="data/adapters/judge-lora", help="Where to save the LoRA adapter.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1, help="Per-device batch size (keep small on 6 GB).")
    ap.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true", help="Validate the data and exit (no GPU/ML deps).")
    return ap.parse_args(argv)


def _dry_run(train: list[dict], val: list[dict]) -> None:
    pos = sum(r.get("label", 0) for r in train)
    gold = sum(1 for r in train if r.get("source") == "rating")
    print(f"train={len(train)} (relevant {pos} / not {len(train) - pos}; gold {gold})  val={len(val)}")
    if not train:
        print("ERROR: no training rows.", file=sys.stderr)
        raise SystemExit(1)
    ex = train[0]
    print("\n--- example[0] prompt ---")
    print(ex["prompt"])
    print("--- example[0] completion ---")
    print(ex["completion"])
    if len(train) < 200:
        print(f"\nWARNING: only {len(train)} rows - expect overfitting. Collect more (gold) labels "
              "and compare the tuned judge to the stock model before trusting it.", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    train = _load(args.train)
    val = _load(args.val) if args.val else []

    if args.dry_run:
        _dry_run(train, val)
        return

    # Heavy imports only for a real run, so --help/--dry-run need no ML stack.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        print(
            f"Missing training deps ({exc}). Install:  pip install -e \".[train,local]\"  "
            "(torch from the CUDA index, see docs/finetuning.md).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(row: dict) -> dict:
        # Render as the model's chat template; mask the prompt so loss is on the
        # completion only (we train the judge to produce the verdict, not echo input).
        msgs = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_only = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(full, truncation=True, max_length=args.max_seq)["input_ids"]
        n_prompt = len(tokenizer(prompt_only, truncation=True, max_length=args.max_seq)["input_ids"])
        labels = [-100] * min(n_prompt, len(ids)) + ids[min(n_prompt, len(ids)):]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds_train = Dataset.from_list(train).map(_format, remove_columns=Dataset.from_list(train).column_names)
    ds_val = (
        Dataset.from_list(val).map(_format, remove_columns=Dataset.from_list(val).column_names)
        if val
        else None
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    def _collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch" if ds_val is not None else "no",
            report_to=[],
        ),
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=_collate,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}. Next: merge + convert to GGUF and import into "
          "Ollama, then set judge_model in config. See docs/finetuning.md.")


if __name__ == "__main__":
    main()
