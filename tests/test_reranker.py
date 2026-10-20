"""Tests for the learned re-ranker (pure logic, no backend/GPU)."""

import math

from paperboy.reranker import (
    Reranker,
    accuracy,
    best_cosine_threshold,
    extract_features,
    train_reranker,
    train_test_split,
)


def test_extract_features_fixed_shape_and_values():
    f = extract_features([0.8, 0.2])
    assert len(f) == 5
    mx, mean, mn, std, gap = f
    assert mx == 0.8 and mn == 0.2
    assert mean == 0.5
    assert math.isclose(std, 0.3)
    assert math.isclose(gap, 0.6)  # max minus second-highest


def test_extract_features_single_topic_gap_is_zero():
    f = extract_features([0.7])
    assert f == [0.7, 0.7, 0.7, 0.0, 0.0]  # std and gap undefined -> 0


def test_extract_features_empty():
    assert extract_features([]) == [0.0, 0.0, 0.0, 0.0, 0.0]


def _toy_examples():
    # Relevant papers: sharp single match (high max, high gap). Irrelevant: broad
    # low overlap (low max). The re-ranker should separate them.
    pos = [extract_features([0.8, 0.2, 0.1]) for _ in range(15)]
    neg = [extract_features([0.35, 0.33, 0.30]) for _ in range(15)]
    return [(f, 1) for f in pos] + [(f, 0) for f in neg]


def test_train_separates_a_toy_problem():
    model = train_reranker(_toy_examples())
    assert model.n_train == 30
    assert model.score(extract_features([0.8, 0.2, 0.1])) > 0.5
    assert model.score(extract_features([0.35, 0.33, 0.30])) < 0.5


def test_score_is_a_probability():
    model = train_reranker(_toy_examples())
    for feats in ([0.9, 0.9, 0.9, 0.9, 0.9], [-5, -5, -5, -5, -5], [0, 0, 0, 0, 0]):
        p = model.score(feats)
        assert 0.0 <= p <= 1.0  # stable sigmoid, never overflows


def test_save_load_roundtrip(tmp_path):
    model = train_reranker(_toy_examples())
    path = tmp_path / "data" / "reranker.json"  # parent created on save
    model.save(path)
    loaded = Reranker.load(path)
    feats = extract_features([0.8, 0.2, 0.1])
    assert math.isclose(loaded.score(feats), model.score(feats))
    assert loaded.n_train == model.n_train


def test_train_rejects_empty():
    try:
        train_reranker([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty training set")


def test_train_test_split_is_deterministic_and_disjoint():
    examples = [([float(i)], i % 2) for i in range(20)]
    tr1, te1 = train_test_split(examples, test_frac=0.25, seed=0)
    tr2, te2 = train_test_split(examples, test_frac=0.25, seed=0)
    assert (tr1, te1) == (tr2, te2)  # deterministic
    assert len(te1) == 5 and len(tr1) == 15
    keys = {tuple(f) for f, _ in te1}
    assert not any(tuple(f) in keys for f, _ in tr1)  # disjoint


def test_train_test_split_tiny_keeps_all_in_train():
    examples = [([0.0], 0), ([1.0], 1), ([2.0], 0)]
    tr, te = train_test_split(examples)
    assert te == [] and len(tr) == 3


def test_best_cosine_threshold_separates():
    # max-cosine >= 0.5 is relevant; the chosen threshold should classify perfectly.
    examples = [([0.9], 1), ([0.8], 1), ([0.3], 0), ([0.2], 0)]
    thr = best_cosine_threshold(examples)
    assert 0.3 < thr <= 0.8
    assert accuracy(lambda f, t=thr: f[0] >= t, examples) == 1.0
