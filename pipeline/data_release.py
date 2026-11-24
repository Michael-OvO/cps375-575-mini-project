"""Release our data to the private Hugging Face dataset repo (rules: docs/data/VERSIONING.md).

    python -m pipeline.data_release plan                       # what would change; uploads nothing
    python -m pipeline.data_release push -m "add v5 mixture"   # one commit with every changed file
    python -m pipeline.data_release tag v5 -m "trained v5 SFT" # freeze the current main as a version

The repo id comes from --repo or $HF_DATA_REPO. `push` refuses to upload unless the repo is private.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from pipeline.config import DATA, ROOT, RUNS

CARD = ROOT / "docs" / "data" / "README.md"
CHANGELOG = ROOT / "docs" / "data" / "CHANGELOG.md"
EVAL_FILES = {"math_dev.jsonl", "code_dev.jsonl", "if_dev.jsonl", "if_dev_new.jsonl", "rl_traps.jsonl"}
_SOURCE = re.compile(r".+\.(jsonl|stats\.json|report\.json)$")


def layout() -> dict[str, Path]:
    """Repo path -> local file. eval/: never train on it. sources/: inputs to select.py, which drops dev items.
    mixtures/<run>/: finished SFT mixtures, dev-filtered and leakage-checked. rl/pools/: rl1 prompt pools."""
    out: dict[str, Path] = {"README.md": CARD, "CHANGELOG.md": CHANGELOG}
    for f in sorted(DATA.iterdir()):
        if not f.is_file():
            continue  # discover/, logs and pilot archives stay local
        if f.name in EVAL_FILES:
            out[f"eval/{f.name}"] = f
        elif _SOURCE.match(f.name):
            out[f"sources/{f.name}"] = f
    for f in sorted(RUNS.glob("*/train.jsonl")):
        run = f.parent
        if not re.fullmatch(r"v\d+", run.name):
            continue  # rl1, v4_smoke and similar
        out[f"mixtures/{run.name}/train.jsonl"] = f
        if (run / "manifest.json").exists():
            out[f"mixtures/{run.name}/manifest.json"] = run / "manifest.json"
    for f in sorted((RUNS / "rl1" / "pools").glob("*")):
        out[f"rl/pools/{f.name}"] = f
    return out


def describe(path: Path) -> dict:
    h, rows, last = hashlib.sha256(), 0, b"\n"
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            rows += chunk.count(b"\n")
            last = chunk[-1:]
    rows += last != b"\n"  # a final line without a newline still counts
    out = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
    if path.suffix == ".jsonl":
        out["rows"] = rows
    return out


def manifest(files: dict[str, Path]) -> dict[str, dict]:
    return {p: describe(f) for p, f in sorted(files.items())}


def changed(local: dict[str, dict], remote: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(added or modified paths, paths on the hub that are no longer local). Removals are reported, never applied."""
    upd = [p for p, d in local.items() if remote.get(p, {}).get("sha256") != d["sha256"]]
    gone = [p for p in remote if p not in local]
    return upd, gone


def _remote_manifest(api, repo: str) -> dict:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    try:
        path = api.hf_hub_download(repo, "MANIFEST.json", repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}
    return json.loads(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "push", "tag"])
    ap.add_argument("tag_name", nargs="?")
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--repo", default=os.environ.get("HF_DATA_REPO"))
    args = ap.parse_args()
    if not args.repo:
        raise SystemExit("set --repo or HF_DATA_REPO (cps375-team/cps375-data)")
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()

    if args.cmd == "tag":
        if not args.tag_name or not args.message:
            raise SystemExit("usage: tag <name> -m <what was trained on it>")
        api.create_tag(args.repo, tag=args.tag_name, tag_message=args.message, repo_type="dataset")
        print(f"tagged {args.repo}@{args.tag_name}")
        return

    files = layout()
    local = manifest(files)
    remote = _remote_manifest(api, args.repo)
    upd, gone = changed(local, remote)
    size = sum(local[p]["bytes"] for p in upd)
    print(f"{len(local)} files tracked; {len(upd)} to upload ({size / 1e6:.1f} MB)")
    for p in upd:
        print("  +", p)
    for p in gone:
        print("  ? on hub, not local (left alone):", p)
    if args.cmd == "plan":
        return
    if not args.message:
        raise SystemExit("push needs -m <message>; also add an entry to docs/data/CHANGELOG.md")
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit(f"{args.repo} is not private; refusing to upload")
    if not upd:
        print("nothing changed")
        return
    ops = [CommitOperationAdd(p, str(files[p])) for p in upd]
    ops.append(CommitOperationAdd("MANIFEST.json", json.dumps(local, indent=1).encode()))
    info = api.create_commit(args.repo, operations=ops, commit_message=args.message, repo_type="dataset")
    print(f"pushed {info.oid[:10]}: {args.message}")


if __name__ == "__main__":
    main()
