"""Advantages for rl1 (spec §4). Group-centered rewards (no std normalization, no length normalization);
constant groups carry zero advantage and are dropped. Per-domain coefficient c_d = 1/(S_d L_d), normalized so only
ratios matter, from an EMA over previous steps only, so each step's estimate is c_d ((G-1)/G) grad J_d, unbiased."""

import math


def centered(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def is_constant(rewards: list[float], tol: float = 1e-9) -> bool:
    return max(rewards) - min(rewards) < tol


class DomainScaler:
    def __init__(self, init: dict[str, tuple[float, float]], alpha: float = 0.1):
        self.alpha = alpha
        self.s = {d: float(v[0]) for d, v in init.items()}
        self.l = {d: float(v[1]) for d, v in init.items()}

    def coef(self, d: str) -> float:
        prods = [self.s[k] * self.l[k] for k in self.s]
        return (sum(prods) / len(prods)) / (self.s[d] * self.l[d])

    def update(self, d: str, centered_nonconst: list[float], mean_len: float) -> None:
        if not centered_nonconst or mean_len <= 0:
            return
        s_new = math.sqrt(sum(a * a for a in centered_nonconst) / len(centered_nonconst))
        if s_new <= 0:
            return
        a = self.alpha
        self.s[d] = (1 - a) * self.s[d] + a * s_new
        self.l[d] = (1 - a) * self.l[d] + a * mean_len

    def state(self) -> dict:
        return {"alpha": self.alpha, "s": dict(self.s), "l": dict(self.l)}

    @classmethod
    def from_state(cls, state: dict) -> "DomainScaler":
        out = cls({d: (state["s"][d], state["l"][d]) for d in state["s"]}, state["alpha"])
        return out


def token_advantages(adv: float, n_tokens: int, teacher_logratio: list[float] | None = None, lam: float = 0.0,
                     mask: list[float] | None = None) -> list[float]:
    """Per completion token: the (already scaled) task advantage, minus lam*(log p_student - log q_teacher)
    on unmasked tokens when the teacher term is on."""
    out = [adv] * n_tokens
    if teacher_logratio is not None and lam:
        m = mask or [1.0] * n_tokens
        out = [a - lam * lr * w for a, lr, w in zip(out, teacher_logratio, m)]
    return out
