import json

from pipeline import data_release as dr


def _setup(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "discover").mkdir(parents=True)
    for name in ("math_dev.jsonl", "math_train_solutions.jsonl", "x.jsonl.done", "x.jsonl.usage.json",
                 "oci_tasks.jsonl.report.json"):
        (data / name).write_text('{"a": 1}\n')
    (data / "discover" / "gen.log").write_text("log")
    for run in ("v4", "v4_smoke"):
        (runs / run).mkdir(parents=True)
        (runs / run / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (runs / "v4" / "manifest.json").write_text("{}")
    (runs / "rl1" / "pools").mkdir(parents=True)
    (runs / "rl1" / "pools" / "math.jsonl").write_text('{"a": 1}\n')
    monkeypatch.setattr(dr, "DATA", data)
    monkeypatch.setattr(dr, "RUNS", runs)


def test_layout_routes_files_and_skips_noise(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    paths = set(dr.layout())
    assert {"eval/math_dev.jsonl", "sources/math_train_solutions.jsonl", "sources/oci_tasks.jsonl.report.json",
            "mixtures/v4/train.jsonl", "mixtures/v4/manifest.json", "rl/pools/math.jsonl",
            "README.md", "CHANGELOG.md"} <= paths
    assert not any("done" in p or "usage" in p or "smoke" in p or "discover" in p for p in paths)
    assert "sources/math_dev.jsonl" not in paths


def test_manifest_and_changed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    files = {p: f for p, f in dr.layout().items() if p.startswith("mixtures/v4/train")}
    local = dr.manifest(files)
    assert local["mixtures/v4/train.jsonl"]["rows"] == 2
    remote = json.loads(json.dumps(local))
    remote["old/file.jsonl"] = {"sha256": "x", "bytes": 1}
    assert dr.changed(local, remote) == ([], ["old/file.jsonl"])
    remote["mixtures/v4/train.jsonl"]["sha256"] = "different"
    assert dr.changed(local, remote)[0] == ["mixtures/v4/train.jsonl"]
