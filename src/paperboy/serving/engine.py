"""A small GPU inference engine with dynamic batching.

This is paperBoy's own serving layer (Phase 3) — the part that makes the
"I built an inference server" story real rather than deferring to Ollama. It:

  * loads an open HF model with optional 4-bit quantization (bitsandbytes NF4),
  * runs a background worker that groups concurrent requests into a single
    padded batch (dynamic batching) before hitting the GPU,
  * reports enough timing/VRAM to benchmark batch-on vs batch-off.

It is *dynamic* (request-level) batching, not *continuous* (iteration-level)
batching like vLLM — an honest, achievable optimization that still shows real
throughput gains.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class Request:
    prompt: str
    system: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.2
    # filled in by the engine:
    event: threading.Event = field(default_factory=threading.Event)
    result: str = ""
    n_prompt_tokens: int = 0
    n_gen_tokens: int = 0
    queued_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None

    @property
    def latency(self) -> float:
        return self.finished_at - self.queued_at


class InferenceEngine:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        quantize: str = "4bit",  # "4bit" | "none"
        max_batch_size: int = 8,
        max_wait_ms: int = 10,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.quantize = quantize
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched causal generation.
        self.tokenizer.padding_side = "left"

        self.model = self._load_model()
        self.model.eval()

        self._queue: queue.Queue[Request] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="paperboy-engine", daemon=True)
        self._worker.start()

    def _load_model(self):
        kwargs: dict = {"device_map": {"": 0} if self.device == "cuda" else None}
        if self.quantize == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    # ---- public API -----------------------------------------------------

    def submit(self, req: Request) -> Request:
        """Enqueue a request and block until it's done (or errored)."""
        req.queued_at = time.perf_counter()
        self._queue.put(req)
        req.event.wait()
        if req.error:
            raise RuntimeError(req.error)
        return req

    def vram_mb(self) -> float:
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)

    def close(self) -> None:
        self._stop.set()

    # ---- batching worker ------------------------------------------------

    def _collect_batch(self) -> list[Request]:
        """Wait for the first request (staying responsive to shutdown), then
        briefly gather more so they run as one batch."""
        first: Request | None = None
        while first is None:
            if self._stop.is_set():
                return []
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

        batch = [first]
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._run_batch(batch)
            except Exception as exc:  # surface to every waiter in the batch
                for r in batch:
                    r.error = str(exc)
                    r.finished_at = time.perf_counter()
                    r.event.set()

    def _build_text(self, req: Request) -> str:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _run_batch(self, batch: list[Request]) -> None:
        texts = [self._build_text(r) for r in batch]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        input_len = enc["input_ids"].shape[1]

        # One generation config for the batch; sampling only if any temp > 0.
        max_new = max(r.max_new_tokens for r in batch)
        temp = max(r.temperature for r in batch)
        gen_kwargs = dict(max_new_tokens=max_new, pad_token_id=self.tokenizer.pad_token_id)
        if temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp)
        else:
            gen_kwargs.update(do_sample=False)

        out = self.model.generate(**enc, **gen_kwargs)
        gen = out[:, input_len:]
        finished = time.perf_counter()

        for r, row, in_ids in zip(batch, gen, enc["input_ids"]):
            text = self.tokenizer.decode(row, skip_special_tokens=True).strip()
            n_gen = int((row != self.tokenizer.pad_token_id).sum().item())
            n_prompt = int((in_ids != self.tokenizer.pad_token_id).sum().item())
            r.result = text
            r.n_gen_tokens = n_gen
            r.n_prompt_tokens = n_prompt
            r.finished_at = finished
            r.event.set()
