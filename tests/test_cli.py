"""End-to-end CLI tests with a fake backend and a fake arXiv (no network, no GPU).

These exercise the command wiring (fetch -> rank -> gate -> summarize -> render,
plus the label / history / metrics files) that the unit tests can't see.
"""

import json

import pytest
from typer.testing import CliRunner

from paperboy import cli
from paperboy.backends.base import Backend
from paperboy.sources import Paper

runner = CliRunner()


class FakeBackend(Backend):
    """Embeds by keyword, keeps papers whose title mentions 'gpu'."""

    def generate(self, prompt, *, system=None, temperature=None):
        if "VERDICT" in prompt:
            title = prompt.rsplit("Title:", 1)[1].split("\n")[0].lower()
            return "VERDICT: KEEP\nREASON: on topic" if "gpu" in title else "VERDICT: DROP\nREASON: off topic"
        return "A short faithful summary."

    def embed(self, text, *, prefix=""):
        t = text.lower()
        return [1.0 if "gpu" in t else 0.1, 1.0 if "cooking" in t else 0.1]


def _papers():
    return [
        Paper(arxiv_id=f"2610.{i:05d}v1", title=t, abstract="abs", url=f"http://x/{i}", published="")
        for i, t in enumerate(["GPU serving A", "GPU serving B", "Cooking pasta", "GPU kernels C"])
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_backend", lambda cfg: FakeBackend())
    monkeypatch.setattr(cli, "fetch_arxiv", lambda cats, n: _papers())
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "topics: ['gpu serving']\n"
        f"data_dir: '{(tmp_path / 'data').as_posix()}'\n"
        f"output:\n  format: markdown\n  dir: '{(tmp_path / 'out').as_posix()}'\n",
        encoding="utf-8",
    )
    return tmp_path, str(cfg)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_writes_digest_labels_metrics_and_history(env):
    tmp, cfg = env
    result = runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    assert result.exit_code == 0, result.output

    digest = next((tmp / "out").glob("digest-*.md")).read_text(encoding="utf-8")
    assert "GPU serving A" in digest and "Cooking" not in digest

    gate = [e for e in _lines(tmp / "data" / "labels.jsonl") if e["kind"] == "gate"]
    assert {e["verdict"] for e in gate} == {"keep", "drop"}  # negatives logged too

    run = _lines(tmp / "data" / "runs.jsonl")[-1]
    assert run["status"] == "ok" and run["selected"] == 2 and run["gate_dropped"] == 1

    assert len(json.loads((tmp / "data" / "seen.json").read_text(encoding="utf-8"))) == 2


def test_second_run_skips_delivered_papers(env):
    tmp, cfg = env
    runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    result = runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    assert result.exit_code == 0, result.output
    assert "skipped 2 already delivered" in result.output
    run = _lines(tmp / "data" / "runs.jsonl")[-1]
    assert run["skipped_seen"] == 2 and run["selected"] == 1  # only GPU kernels C is left

    again = runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2", "--include-seen"])
    assert "skipped" not in again.output


def test_failed_run_is_logged_with_stage(env, monkeypatch):
    tmp, cfg = env

    def boom(cats, n):
        import httpx

        raise httpx.ConnectError("down")

    monkeypatch.setattr(cli, "fetch_arxiv", boom)
    result = runner.invoke(cli.app, ["run", "-c", cfg])
    assert result.exit_code == 1
    run = _lines(tmp / "data" / "runs.jsonl")[-1]
    assert run["status"] == "failed" and run["failed_stage"] == "fetch"


def test_stats_after_runs(env):
    _, cfg = env
    runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    result = runner.invoke(cli.app, ["stats", "-c", cfg])
    assert result.exit_code == 0
    assert "All 1 runs: ok 1" in result.output


def test_rate_review_records_gold_and_reveals_judge(env):
    tmp, cfg = env
    runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    result = runner.invoke(cli.app, ["rate", "-c", cfg, "-r", "2"], input="g\nb\n")
    assert result.exit_code == 0, result.output
    assert "judge said" in result.output
    ratings = [e for e in _lines(tmp / "data" / "labels.jsonl") if e["kind"] == "rating"]
    assert [r["verdict"] for r in ratings] == ["good", "bad"]


def test_eval_judge_scores_against_labels(env):
    _, cfg = env
    runner.invoke(cli.app, ["run", "-c", cfg, "--top", "2"])
    result = runner.invoke(cli.app, ["eval-judge", "-c", cfg, "--split", "all"])
    assert result.exit_code == 0, result.output
    assert "acc 100%" in result.output  # same fake judge that produced the labels
    assert "not true accuracy" in result.output  # warns there are no gold labels


def test_doctor_checks_each_model(env, monkeypatch):
    tmp, cfg = env
    result = runner.invoke(cli.app, ["doctor", "-c", cfg])
    assert result.exit_code == 0, result.output
    assert "OK   summary model" in result.output and "OK   embeddings" in result.output

    class NoEmbed(FakeBackend):
        def embed(self, text, *, prefix=""):
            from paperboy.backends import BackendError

            raise BackendError("embedding model missing")

    monkeypatch.setattr(cli, "get_backend", lambda c: NoEmbed())
    result = runner.invoke(cli.app, ["doctor", "-c", cfg])
    assert result.exit_code == 1
    assert "FAIL embeddings" in result.output
