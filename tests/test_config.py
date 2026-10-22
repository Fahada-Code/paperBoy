"""Tests for config loading and defaults."""

from paperboy.config import load_config


def test_missing_file_uses_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.backend.kind == "ollama"
    assert cfg.backend.model == "llama3.2:3b"
    assert cfg.backend.timeout == 300
    assert cfg.top_n == 8
    assert cfg.topics == []
    assert cfg.arxiv.enabled is True
    assert cfg.llm_filter is True
    assert cfg.judge_model == ""  # empty = judge uses the summary model
    assert cfg.judge_tuned is False


def test_judge_model_loads(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("judge_model: qwen2.5:14b\nllm_filter: false\njudge_tuned: true\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.judge_model == "qwen2.5:14b"
    assert cfg.judge_tuned is True
    assert cfg.llm_filter is False


def test_partial_config_overrides_only_given_keys(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "top_n: 3\n"
        "topics:\n  - gpu serving\n"
        "backend:\n  model: qwen2.5:7b\n"
        "sources:\n  arxiv:\n    categories: [cs.DC]\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.top_n == 3
    assert cfg.topics == ["gpu serving"]
    assert cfg.backend.model == "qwen2.5:7b"
    assert cfg.backend.kind == "ollama"  # untouched default
    assert cfg.backend.timeout == 300  # untouched default
    assert cfg.arxiv.categories == ["cs.DC"]
