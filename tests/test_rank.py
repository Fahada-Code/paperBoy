"""Tests for the relevance ranker (pure logic, no network or GPU)."""

from paperboy.rank import _cosine, _dedup, rank_papers
from paperboy.sources import Paper


def _paper(pid: str, title: str = "t", abstract: str = "a") -> Paper:
    return Paper(arxiv_id=pid, title=title, abstract=abstract, url="", published="2026-01-01")


def test_cosine_basics():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([], [1, 2]) == 0.0
    assert _cosine([0, 0], [1, 2]) == 0.0


def test_dedup_strips_version_suffix():
    papers = [_paper("2401.001v1"), _paper("2401.001v2"), _paper("2401.002v1")]
    out = _dedup(papers)
    assert [p.arxiv_id for p in out] == ["2401.001v1", "2401.002v1"]


def test_rank_without_topics_returns_most_recent():
    papers = [_paper(f"id{i}") for i in range(5)]
    # No embed calls should happen when there are no topics.
    out = rank_papers(papers, topics=[], embed=lambda *a, **k: 1 / 0, top_n=3)
    assert len(out) == 3
    assert out[0].arxiv_id == "id0"


def test_rank_orders_by_similarity_to_topics():
    # Fake embeddings: map keywords to orthogonal-ish vectors.
    def fake_embed(text: str, *, prefix: str = "") -> list[float]:
        t = text.lower()
        return [1.0 if "gpu" in t else 0.0, 1.0 if "cooking" in t else 0.0]

    relevant = _paper("gpu", title="GPU inference", abstract="gpu batching")
    off_topic = _paper("food", title="cooking pasta", abstract="cooking recipes")
    out = rank_papers([off_topic, relevant], topics=["gpu serving"], embed=fake_embed, top_n=2)
    assert out[0].arxiv_id == "gpu"
    assert out[0].score > out[1].score


def test_rank_scores_by_best_topic_not_centroid_average():
    # Two orthogonal topics. A paper matching ONE topic perfectly should score
    # 1.0 (best match), not ~0.71 (cosine to the averaged centroid of both).
    def fake_embed(text: str, *, prefix: str = "") -> list[float]:
        t = text.lower()
        return [1.0 if "gpu" in t else 0.0, 1.0 if "cook" in t else 0.0]

    gpu = _paper("gpu", title="gpu inference", abstract="gpu")
    out = rank_papers([gpu], topics=["gpu serving", "cooking"], embed=fake_embed, top_n=5)
    assert out[0].score == 1.0
    assert out[0].matched_topic == "gpu serving"  # records WHICH topic matched best


def test_rank_min_score_drops_weak_matches():
    def fake_embed(text: str, *, prefix: str = "") -> list[float]:
        return [1.0 if "gpu" in text.lower() else 0.0, 0.0]

    gpu = _paper("gpu", title="gpu inference", abstract="gpu")
    food = _paper("food", title="cooking pasta", abstract="recipes")  # scores 0.0
    out = rank_papers([gpu, food], topics=["gpu"], embed=fake_embed, top_n=5, min_score=0.5)
    assert [p.arxiv_id for p in out] == ["gpu"]
