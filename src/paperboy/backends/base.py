"""The backend contract.

A backend is anything that can turn a prompt into text. Concrete backends
(Ollama, local GPU, hosted API) implement exactly one method, `generate()`.
Higher-level operations like `summarize()` are composed on top here, so every
backend gets them for free and behaves identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

SUMMARY_SYSTEM = (
    "You are a precise technical summarizer for a software engineer's research digest. "
    "You explain and condense; you never give advice, predictions, or recommendations. "
    "Be concrete, neutral, and faithful to the source. No preamble, no 'in summary'. "
    "Only use facts stated in the source. Never invent numbers, names, model names, or "
    "results, and never output placeholders like '[insert ...]'. If a specific figure is "
    "not given, omit it rather than guessing."
)


class BackendError(RuntimeError):
    """Raised when a backend can't fulfill a request (unreachable, bad model, etc.)."""


class Backend(ABC):
    """An LLM text-generation backend."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
        """Return the model's completion for `prompt`.

        `temperature` lets callers trade creativity for faithfulness; None uses the
        backend/model default.
        """

    def embed(self, text: str, *, prefix: str = "") -> list[float]:
        """Return an embedding vector for `text`. Backends that can't embed raise.

        `prefix` lets retrieval models (e.g. nomic) distinguish queries from
        documents; backends that don't need it ignore it.
        """
        raise BackendError(f"{type(self).__name__} does not support embeddings.")

    def embed_many(self, texts: list[str], *, prefix: str = "") -> list[list[float]]:
        """Embed several texts. Default loops over embed(); backends with a batch
        endpoint override this, since per-request overhead dominates small texts."""
        return [self.embed(t, prefix=prefix) for t in texts]

    def summarize(self, text: str, *, max_words: int = 150, topic: str | None = None) -> str:
        """Summarize `text` in at most `max_words` words. Composed on top of generate().

        If `topic` is given (the reader's best-matching interest, as judged by the
        ranker), the summary ends with one short sentence on how the work relates to
        it - explanation of relevance, not advice. The model is told to admit when
        the connection is weak rather than invent one, keeping relevance honest.
        """
        prompt = (
            f"Summarize the following in at most {max_words} words. "
            f"Lead with the single most important point."
        )
        if topic:
            prompt += (
                f' This work was retrieved as a possible match for the reader\'s topic: "{topic}". '
                "End with one short sentence on how it relates to that topic - but if the "
                "connection is weak or tangential, say so plainly instead of overstating it."
            )
        prompt += f"\n\n{text}"
        # Low temperature: a summarizer should be faithful, not creative.
        return self.generate(prompt, system=SUMMARY_SYSTEM, temperature=0.2).strip()
