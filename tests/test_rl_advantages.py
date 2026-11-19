import numpy as np
import pytest

from pipeline.rl import advantages as A


def test_centered_and_constant():
    assert A.centered([1, 0, 0, 1]) == [0.5, -0.5, -0.5, 0.5]
    assert A.is_constant([1, 1, 1]) and not A.is_constant([1, 0])


def test_group_estimator_is_unbiased_for_scaled_gradient():
    """E[(1/G) sum_i (r_i - rbar) grad log pi(a_i)] = ((G-1)/G) grad J on a softmax bandit."""
    rng = np.random.default_rng(0)
    theta = np.array([0.2, -0.1, 0.4])
    r = np.array([1.0, 0.0, 0.5])
    pi = np.exp(theta) / np.exp(theta).sum()
    grad_j = np.array([pi[a] * r[a] for a in range(3)]) - pi * (pi @ r)
    G, n = 8, 200_000
    acts = rng.choice(3, size=(n, G), p=pi)
    rew = r[acts]
    adv = rew - rew.mean(axis=1, keepdims=True)
    onehot = np.eye(3)[acts]  # (n, G, 3)
    glog = onehot - pi  # grad log pi(a) = e_a - pi
    est = (adv[..., None] * glog).mean(axis=1).mean(axis=0)
    np.testing.assert_allclose(est, (G - 1) / G * grad_j, atol=2e-3)


def test_scaler_equalizes_domains_and_uses_previous_steps_only():
    s = A.DomainScaler({"math": (0.4, 600.0), "if": (0.2, 300.0)})
    c_math, c_if = s.coef("math"), s.coef("if")
    assert c_math * 0.4 * 600 == pytest.approx(c_if * 0.2 * 300)
    s.update("math", [0.5, -0.5, 0.5, -0.5], 800.0)
    assert s.coef("math") < c_math  # larger S and L now -> smaller coefficient next step
    t = A.DomainScaler.from_state(s.state())
    assert t.coef("math") == pytest.approx(s.coef("math"))


def test_scaler_ignores_empty_updates():
    s = A.DomainScaler({"code": (0.3, 200.0)})
    before = s.coef("code")
    s.update("code", [], 0.0)
    assert s.coef("code") == before


def test_token_advantages_with_teacher_term_and_mask():
    out = A.token_advantages(0.5, 3, teacher_logratio=[0.1, -0.2, 0.0], lam=1.0, mask=[0.0, 1.0, 1.0])
    assert out == pytest.approx([0.5, 0.5 + 0.2, 0.5])
    assert A.token_advantages(-0.25, 2) == [-0.25, -0.25]
