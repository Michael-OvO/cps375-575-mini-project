import pytest

from pipeline import leakage


def test_check_sources():
    leakage.check_sources([{"name": "math_train", "dataset": "DigitalLearningGmbH/MATH-lighteval",
                            "split": "train", "count": 10},
                           {"name": "code", "dataset": "teacher-generated", "split": "generated", "count": 5}])
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "bad", "dataset": "google-research-datasets/mbpp",
                                "split": "train", "count": 1}])
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "bad", "dataset": "DigitalLearningGmbH/MATH-lighteval",
                                "split": "test", "count": 1}])


def test_overlap_filter():
    test_text = "Find the number of ordered pairs of positive integers whose product equals one hundred."
    f = leakage.OverlapFilter([test_text])
    assert f.is_leak("Find the number of ordered pairs of positive integers whose product equals one hundred!")
    assert not f.is_leak("Compute the area of a triangle with base four and height nine in square units.")
    assert not f.is_leak("short text")


def test_load_overlap_filter_disabled_without_flag(monkeypatch):
    monkeypatch.setattr(leakage, "ALLOW_TEST_OVERLAP_CHECK", False)
    assert leakage.load_overlap_filter() is None
