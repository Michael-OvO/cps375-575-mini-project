"""Result mapping only: fake eval logs, no dataset is loaded and no eval is run."""

from types import SimpleNamespace as NS

from pipeline import proxy_eval


def _log(task, accuracy=None, dataset=None):
    scores = [NS(metrics={"accuracy": NS(value=accuracy)})] if accuracy is not None else []
    return NS(eval=NS(task=task, dataset=NS(name=dataset)), results=NS(scores=scores) if scores else None)


def test_scores_mapped_by_task_name_not_position(capsys):
    out = proxy_eval.scores_by_task([_log("inspect_evals/humaneval", 0.4), _log("math_rest", 0.7)])
    assert out == {"math_rest": 0.7, "humaneval": 0.4}
    assert "WARNING" not in capsys.readouterr().out


def test_dataset_name_fallback_and_missing_score_warns(capsys):
    out = proxy_eval.scores_by_task([_log("math", 0.5, dataset="math_rest"), _log("humaneval")])
    assert out == {"math_rest": 0.5, "humaneval": None}
    assert "'humaneval' produced NO score" in capsys.readouterr().out


def test_absent_task_is_none_with_warning(capsys):
    out = proxy_eval.scores_by_task([_log("something_else", 0.9)])
    assert out == {"math_rest": None, "humaneval": None}
    text = capsys.readouterr().out
    assert "unknown task" in text and "'math_rest' produced NO score" in text
