"""Tests for the LLM relevance gate (no network: a fake generate fn)."""

from paperboy.backends import BackendError
from paperboy.filter import _verdict, filter_relevant
from paperboy.sources import Paper


def _paper(pid: str, title: str) -> Paper:
    return Paper(arxiv_id=pid, title=title, abstract="abs", url="", published="2026-01-01")


def test_verdict_parses_keep_and_drop():
    assert _verdict("VERDICT: KEEP\nREASON: on topic") == (True, "on topic")
    assert _verdict("VERDICT: DROP\nREASON: unrelated") == (False, "unrelated")


def test_verdict_fails_open_when_unclear():
    # No parseable verdict -> keep (don't silently discard real papers).
    assert _verdict("I am not sure about this one.")[0] is True


def test_verdict_reason_overrides_contradictory_keep():
    # Model said KEEP but the reason plainly says irrelevant -> trust the reason.
    keep, reason = _verdict("VERDICT: KEEP\nREASON: Irrelevant to specified topics")
    assert keep is False
    assert "irrelevant" in reason.lower()


def test_filter_keeps_only_approved_and_sets_reason():
    def fake_generate(prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        # Drop the baking paper, keep the rest. Key on a title word that can't
        # appear in the topics or the few-shot examples baked into the prompt.
        if "sourdough" in prompt.lower():
            return "VERDICT: DROP\nREASON: only shares ML words"
        return "VERDICT: KEEP\nREASON: matches serving"

    gpu = _paper("gpu", "GPU inference serving")
    cook = _paper("cook", "Sourdough fermentation kinetics")
    kept = filter_relevant([gpu, cook], topics=["gpu serving"], generate=fake_generate)
    assert [p.arxiv_id for p in kept] == ["gpu"]
    assert kept[0].relevance_reason == "matches serving"


def test_filter_fails_open_on_backend_error():
    def boom(prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        raise BackendError("backend down")

    p = _paper("x", "anything")
    kept = filter_relevant([p], topics=["t"], generate=boom)
    assert kept == [p]  # kept despite the error


def test_filter_noop_without_topics():
    def boom(*a, **k):
        raise AssertionError("generate should not be called without topics")

    papers = [_paper("a", "a"), _paper("b", "b")]
    assert filter_relevant(papers, topics=[], generate=boom) == papers


def test_tuned_mode_sends_training_prompt_without_fewshot():
    from paperboy.training.dataset import build_prompt

    seen = []

    def fake_generate(prompt: str, *, system=None, temperature: float = 0.0) -> str:
        seen.append((prompt, system))
        return "VERDICT: KEEP\nREASON: on topic"

    p = _paper("a", "GPU inference serving")
    filter_relevant([p], topics=["gpu serving"], generate=fake_generate, tuned=True)
    prompt, system = seen[0]
    assert prompt == build_prompt(["gpu serving"], p.title, p.abstract)  # matches training
    assert "Example A" not in prompt and system is None


def test_default_mode_keeps_fewshot():
    seen = []

    def fake_generate(prompt: str, *, system=None, temperature: float = 0.0) -> str:
        seen.append(prompt)
        return "VERDICT: KEEP\nREASON: on topic"

    filter_relevant([_paper("a", "x")], topics=["t"], generate=fake_generate)
    assert "Example A" in seen[0]
