import pytest

from pipeline import leakage
from pipeline.config import ALLOWED_TRAIN_DATASETS


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


@pytest.mark.parametrize("dataset", [
    "mbpp",
    "Muennighoff/mbpp",
    "MATH-500",
    "google/ifeval",
    "some-unknown/dataset",
])
def test_check_sources_rejects_non_allowlisted_datasets(dataset):
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "x", "dataset": dataset, "split": "train"}])


def test_check_sources_rejects_excluded_keyword_in_source_name():
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "mbpp_copy", "dataset": "allenai/tulu-3-sft-mixture",
                                "split": "train"}])


def test_check_sources_rejects_wrong_split_case():
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "x", "dataset": "allenai/tulu-3-sft-mixture", "split": "Train"}])


def test_check_sources_rejects_missing_split_key():
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "x", "dataset": "allenai/tulu-3-sft-mixture"}])


def test_check_sources_rejects_missing_dataset_key():
    with pytest.raises(ValueError):
        leakage.check_sources([{"name": "x", "split": "train"}])


@pytest.mark.parametrize("dataset", ALLOWED_TRAIN_DATASETS)
def test_check_sources_accepts_allowlisted_datasets(dataset):
    split = "generated" if dataset == "teacher-generated" else "train"
    leakage.check_sources([{"name": "ok", "dataset": dataset, "split": split}])


def test_overlap_filter():
    test_text = "Find the number of ordered pairs of positive integers whose product equals one hundred."
    f = leakage.OverlapFilter([test_text])
    assert f.is_leak("Find the number of ordered pairs of positive integers whose product equals one hundred!")
    assert not f.is_leak("Compute the area of a triangle with base four and height nine in square units.")
    assert not f.is_leak("short text")


def test_short_verbatim_test_item_is_flagged():
    item = "alpha bravo charlie delta echo"  # 5 tokens, below n=8
    f = leakage.OverlapFilter([item])
    assert f.is_leak(item)


def test_short_test_item_wrapped_in_template_is_flagged():
    item = "alpha bravo charlie delta echo"  # 5 tokens
    template = " ".join(f"template{i}" for i in range(60))
    f = leakage.OverlapFilter([item])
    assert f.is_leak(f"{template} {item} {template}")


def test_long_test_item_embedded_in_unrelated_text_is_flagged():
    item_words = [f"tok{i}" for i in range(20)]  # 20 tokens
    pad = [f"pad{i}" for i in range(120)]  # 120 unrelated tokens
    f = leakage.OverlapFilter([" ".join(item_words)])
    assert f.is_leak(" ".join(pad[:60] + item_words + pad[60:]))


def test_long_test_item_with_one_word_changed_is_flagged():
    item_words = [f"tok{i}" for i in range(20)]
    changed = list(item_words)
    changed[10] = "zzchanged"
    f = leakage.OverlapFilter([" ".join(item_words)])
    assert f.is_leak(" ".join(changed))


def test_item_coverage_rule_flags_embedded_item_with_late_edit():
    # The edit sits near the end, so over half of the item's 8-grams survive, but the
    # candidate is so long that its own n-gram share stays under the threshold.
    item_words = [f"tok{i}" for i in range(20)]
    edited = list(item_words)
    edited[15] = "zzchanged"
    pad = [f"pad{i}" for i in range(120)]
    f = leakage.OverlapFilter([" ".join(item_words)])
    assert f.is_leak(" ".join(pad[:60] + edited + pad[60:]))


def test_unrelated_text_of_similar_length_is_not_flagged():
    item_words = [f"tok{i}" for i in range(20)]
    other = [f"word{i}x" for i in range(20)]
    f = leakage.OverlapFilter([" ".join(item_words)])
    assert not f.is_leak(" ".join(other))


def test_load_overlap_filter_disabled_without_flag(monkeypatch):
    monkeypatch.setattr(leakage, "ALLOW_TEST_OVERLAP_CHECK", False)
    assert leakage.load_overlap_filter() is None


def test_eval_texts_include_humaneval_and_ifbench(monkeypatch):
    seen = []

    def fake_load(name, *args, split=None, **kw):
        seen.append(name)

        class D(dict):
            pass

        d = D()
        for c in ("problem", "prompt", "text"):
            d[c] = [f"{name} sample text item"]
        return d

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    texts = leakage.eval_texts()
    assert "openai/openai_humaneval" in seen and "allenai/IFBench_test" in seen
    assert any("openai_humaneval" in t for t in texts)
