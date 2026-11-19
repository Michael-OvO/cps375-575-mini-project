import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from pipeline.rl import merge as M


def _adapter(tmp, name, B, A, extra=None):
    d = tmp / name
    d.mkdir()
    sd = {"base.layers.0.q.lora_A.weight": A.to(torch.bfloat16), "base.layers.0.q.lora_B.weight": B.to(torch.bfloat16)}
    sd |= extra or {}
    save_file(sd, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": A.shape[0], "lora_alpha": 32}))
    return d


def test_single_input_is_reproduced_exactly():
    g = torch.Generator().manual_seed(0)
    B, A = torch.randn(64, 4, generator=g), torch.randn(4, 48, generator=g)
    B2, A2, err = M.combine_factors([B], [A], [1.0], 4)
    assert err < 1e-5 and torch.allclose(B2 @ A2, B @ A, atol=1e-4)


def test_average_is_best_rank_r_approximation():
    g = torch.Generator().manual_seed(1)
    Bs = [torch.randn(64, 4, generator=g) for _ in range(3)]
    As = [torch.randn(4, 48, generator=g) for _ in range(3)]
    B2, A2, err = M.combine_factors(Bs, As, [1 / 3] * 3, 4)
    target = sum(b @ a for b, a in zip(Bs, As)) / 3
    U, S, Vh = torch.linalg.svd(target)
    best = (U[:, :4] * S[:4]) @ Vh[:4]
    assert torch.allclose(B2 @ A2, best, atol=1e-3)
    assert err == pytest.approx((S[4:].square().sum() / S.square().sum()).sqrt().item(), rel=1e-3)


def test_coefs():
    assert M.coefs("tail", 4, None) == [0.25] * 4
    assert M.coefs("interp", 2, 0.6) == pytest.approx([0.4, 0.6])
    assert M.coefs("interp", 2, 1.2) == pytest.approx([-0.2, 1.2])


def test_merge_dirs_keeps_keys_dtypes_config(tmp_path):
    g = torch.Generator().manual_seed(2)
    keep = {"base.norm.weight": torch.ones(8)}
    a = _adapter(tmp_path, "a", torch.randn(16, 2, generator=g), torch.randn(2, 8, generator=g), keep)
    b = _adapter(tmp_path, "b", torch.randn(16, 2, generator=g), torch.randn(2, 8, generator=g), keep)
    rep = M.merge_dirs([a, b], [0.5, 0.5], tmp_path / "out")
    out = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    src = load_file(str(a / "adapter_model.safetensors"))
    assert set(out) == set(src) and all(out[k].dtype == src[k].dtype and out[k].shape == src[k].shape for k in src)
    assert json.loads((tmp_path / "out" / "adapter_config.json").read_text()) == {"r": 2, "lora_alpha": 32}
    assert rep["modules"] == 1 and 0 <= rep["max_rel_err"] < 1


def test_non_lora_tensors_must_agree(tmp_path):
    g = torch.Generator().manual_seed(3)
    a = _adapter(tmp_path, "a", torch.randn(16, 2, generator=g), torch.randn(2, 8, generator=g), {"x": torch.ones(2)})
    b = _adapter(tmp_path, "b", torch.randn(16, 2, generator=g), torch.randn(2, 8, generator=g), {"x": torch.zeros(2)})
    with pytest.raises(AssertionError):
        M.merge_dirs([a, b], [0.5, 0.5], tmp_path / "out")
