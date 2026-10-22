"""Tests for per-run metrics (timings, status, roundtrip, aggregation)."""

from paperboy.metrics import RunMetrics, append_run, load_runs, summarize_runs


def test_marks_time_each_stage_and_total():
    m = RunMetrics()
    m.mark("fetch")
    m.mark("rank")
    m.finish("ok")
    assert set(m.seconds) == {"fetch", "rank", "total"}
    assert m.status == "ok" and m.failed_stage == ""


def test_failed_run_records_the_stage_it_died_in():
    m = RunMetrics()
    m.mark("fetch")
    m.mark("gate")
    m.finish("failed")
    assert m.status == "failed" and m.failed_stage == "gate"


def test_roundtrip_skips_private_fields_and_corrupt_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    m = RunMetrics(fetched=60, selected=5)
    m.finish("ok")
    append_run(path, m)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    runs = load_runs(path)
    assert len(runs) == 1
    assert runs[0]["fetched"] == 60 and runs[0]["status"] == "ok"
    assert not any(k.startswith("_") for k in runs[0])


def test_summarize_runs_ignores_failed_runs_in_means():
    runs = [
        {"status": "ok", "selected": 6, "seconds": {"total": 100.0}},
        {"status": "empty", "selected": 0, "seconds": {"total": 50.0}},
        {"status": "failed", "selected": 0, "seconds": {"total": 1.0}},
    ]
    agg = summarize_runs(runs)
    assert agg["runs"] == 3
    assert agg["status"] == {"ok": 1, "empty": 1, "failed": 1}
    assert agg["mean_seconds"]["total"] == 75.0
    assert agg["mean_selected"] == 3.0
