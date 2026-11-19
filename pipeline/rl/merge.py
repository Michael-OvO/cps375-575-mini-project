"""Checkpoint merge candidates (spec §9). Adapters are combined in ΔW space (Σ c_i·B_i·A_i, never by averaging A
and B separately) and cut back to rank r exactly: QR of the stacked factors, then an SVD of the small core, so the
result is the best rank-r approximation and loads as an ordinary adapter with the original keys and dtypes."""

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from pipeline import jsonl
from pipeline.config import RUNS, STUDENT

A_SUF, B_SUF = ".lora_A.weight", ".lora_B.weight"


def lora_modules(sd: dict) -> list[str]:
    return sorted(k[: -len(A_SUF)] for k in sd if k.endswith(A_SUF) and k[: -len(A_SUF)] + B_SUF in sd)


def combine_factors(Bs: list[torch.Tensor], As: list[torch.Tensor], coefs: list[float],
                    r: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    Bc = torch.cat([c * b.float() for c, b in zip(coefs, Bs)], dim=1)  # out x (k r)
    Ac = torch.cat([a.float() for a in As], dim=0)  # (k r) x in
    Qb, Rb = torch.linalg.qr(Bc)
    Qa, Ra = torch.linalg.qr(Ac.T)  # Ac = Ra^T Qa^T
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    s = S[:r].sqrt()
    B = (Qb @ U[:, :r]) * s
    A = s[:, None] * (Vh[:r] @ Qa.T)
    err = (S[r:].square().sum() / S.square().sum().clamp_min(1e-30)).sqrt().item()
    return B, A, err


def merge_dirs(inputs: list[Path], coefs: list[float], out: Path) -> dict:
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in inputs]
    assert all(c == cfgs[0] for c in cfgs), "adapter configs differ"
    r = int(cfgs[0]["r"])
    sds = [load_file(str(d / "adapter_model.safetensors")) for d in inputs]
    assert all(set(s) == set(sds[0]) for s in sds), "adapter keys differ"
    mods = lora_modules(sds[0])
    lora_keys = {m + suf for m in mods for suf in (A_SUF, B_SUF)}
    out_sd, errs = {}, {}
    for k in set(sds[0]) - lora_keys:
        assert all(torch.equal(s[k], sds[0][k]) for s in sds), f"non-LoRA tensor differs: {k}"
        out_sd[k] = sds[0][k]
    for m in mods:
        B, A, errs[m] = combine_factors([s[m + B_SUF] for s in sds], [s[m + A_SUF] for s in sds], coefs, r)
        out_sd[m + B_SUF] = B.to(sds[0][m + B_SUF].dtype).contiguous()
        out_sd[m + A_SUF] = A.to(sds[0][m + A_SUF].dtype).contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(cfgs[0], indent=2))
    return {"modules": len(errs), "max_rel_err": max(errs.values()), "mean_rel_err": sum(errs.values()) / len(errs)}


def coefs(kind: str, k: int, lam: float | None) -> list[float]:
    if kind == "tail":
        return [1.0 / k] * k
    if kind == "interp":
        assert k == 2 and lam is not None, "interp takes [v4, rl1] and --lam"
        return [1.0 - lam, lam]
    raise ValueError(kind)


def export(state_path: str, out: Path) -> str:
    """Load a weights/ checkpoint, export it in PEFT format, and download every file into out."""
    import tinker

    client = tinker.ServiceClient().create_training_client_from_state(state_path).result()
    ext = client.save_weights_external(f"export-{out.name}", ttl_seconds=7 * 24 * 3600).result().path
    urls = tinker.RestClient().get_external_weights_urls(ext).result().urls
    out.mkdir(parents=True, exist_ok=True)
    for rel, url in urls.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, out / rel)
    return ext


def upload(d: Path, name: str, rank: int = 32) -> dict:
    """Experimental upload (tinker 0.39.0), then a sampler checkpoint from the uploaded weights."""
    import tinker
    from tinker import types

    rest = tinker.RestClient()
    start = rest.external_weights_upload_start(STUDENT, types.LoraConfig(rank=rank)).result()
    for fname, url in start.urls.items():
        req = urllib.request.Request(url, data=(d / fname).read_bytes(), method="PUT")
        urllib.request.urlopen(req).read()
    path = rest.external_weights_upload_finish(start.upload_id).result().tinker_path
    client = tinker.ServiceClient().create_training_client_from_state(path).result()
    sampler = client.save_weights_for_sampler(name, ttl_seconds=21 * 24 * 3600).result().path
    row = {"name": name, "dir": str(d), "external_path": path, "sampler_path": sampler}
    jsonl.append(RUNS / "rl1" / "merges.jsonl", row)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--state", required=True)
    e.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--kind", choices=["tail", "interp"], required=True)
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--lam", type=float, default=None)
    c.add_argument("--out", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--dir", required=True)
    u.add_argument("--name", required=True)
    a = p.parse_args()
    if a.cmd == "export":
        print(export(a.state, Path(a.out)))
    elif a.cmd == "combine":
        ins = [Path(x) for x in a.inputs]
        print(json.dumps(merge_dirs(ins, coefs(a.kind, len(ins), a.lam), Path(a.out)), indent=1))
    else:
        print(json.dumps(upload(Path(a.dir), a.name), indent=1))


if __name__ == "__main__":
    main()
