"""SPDC 纠缠光子教学演示 —— 物理核心。

本模块只依赖 numpy，负责三件事：

1. 描述一个双光子偏振态（Bell 态 + 白噪声混入）；
2. 按 Born 规则给出任意测量角度下的四重符合概率，并按该概率采样；
3. 提供 CHSH 统计、局域隐变量（LHV）对照模型、HOM 干涉曲线。

关于"随机"的诚实说明
--------------------
这里的随机数来自 numpy 的伪随机发生器。真实实验里的量子随机性在这里是
被 *模拟* 出来的，不是被 *产生* 出来的。任何单个结果都是不可预测的
（对使用者而言），但整套数据的统计性质由量子态严格决定。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ket",
    "projector",
    "BELL_STATES",
    "TwoPhotonState",
    "chsh_for_visibility",
    "chsh_visibility_threshold",
    "ChshAccumulator",
    "lhv_correlation",
    "lhv_chsh",
    "lhv_max_chsh",
    "hom_coincidence",
    "hom_visibility_from_purity",
]

# 单光子基：|H> = (1, 0), |V> = (0, 1)
_H = np.array([1.0, 0.0], dtype=complex)
_V = np.array([0.0, 1.0], dtype=complex)


# --------------------------------------------------------------------------
# 单光子：偏振测量
# --------------------------------------------------------------------------
def ket(theta_deg: float) -> np.ndarray:
    """线偏振态 |θ> = cosθ |H> + sinθ |V>。θ 用角度制。"""
    t = np.deg2rad(float(theta_deg))
    return np.array([np.cos(t), np.sin(t)], dtype=complex)


def projector(theta_deg: float) -> np.ndarray:
    """|θ><θ|，偏振片（分析器）沿 θ 方向的投影算符。"""
    k = ket(theta_deg)
    return np.outer(k, k.conj())


def _orthogonal(theta_deg: float) -> np.ndarray:
    """与 θ 正交的偏振方向，对应"没通过"的那一路。"""
    return ket(float(theta_deg) + 90.0)


# --------------------------------------------------------------------------
# 双光子偏振态
# --------------------------------------------------------------------------
# 基矢顺序：|HH>, |HV>, |VH>, |VV>
# 下标 = 2 * (信号位) + (闲频位)，所以 0 = (+,+)，3 = (-,-)
BELL_STATES = {
    "Φ+  (|HH⟩+|VV⟩)/√2": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0),
    "Φ-  (|HH⟩-|VV⟩)/√2": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2.0),
    "Ψ+  (|HV⟩+|VH⟩)/√2": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2.0),
    "Ψ-  (|HV⟩-|VH⟩)/√2": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2.0),
}


class TwoPhotonState:
    """双光子偏振态。

    用一个纯态 |ψ> 加上白噪声近似真实源：ρ = V|ψ><ψ| + (1-V)·I/4。

    V 是偏振干涉可见度（0 到 1）。真实实验中 V 永远到不了 1；当
    V 低于 1/√2 ≈ 0.707 时，CHSH 就再也超不过 2 —— 这是场景④的核心。
    """

    def __init__(self, psi=None, visibility: float = 1.0, name: str = ""):
        if psi is None:
            psi = BELL_STATES["Φ+  (|HH⟩+|VV⟩)/√2"]
        psi = np.asarray(psi, dtype=complex).reshape(4)
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("态矢不能为零")
        self.psi = psi / norm
        self.visibility = float(np.clip(visibility, 0.0, 1.0))
        self.name = name

    # ---- 概率 -------------------------------------------------------------
    def amplitudes(self, theta_signal: float, theta_idler: float) -> np.ndarray:
        """四个联合结果的振幅 <ab|ψ>，顺序 (++), (+-), (-+), (--)。"""
        ks = (ket(theta_signal), _orthogonal(theta_signal))
        ki = (ket(theta_idler), _orthogonal(theta_idler))
        amps = np.empty(4, dtype=complex)
        for a in (0, 1):
            for b in (0, 1):
                amps[2 * a + b] = np.vdot(np.kron(ks[a], ki[b]), self.psi)
        return amps

    def probabilities(self, theta_signal: float, theta_idler: float) -> np.ndarray:
        """四个联合结果的概率，顺序 (++), (+-), (-+), (--)。"""
        p = np.abs(self.amplitudes(theta_signal, theta_idler)) ** 2
        p = self.visibility * p + (1.0 - self.visibility) / 4.0
        total = p.sum()
        return p / total if total > 0 else np.full(4, 0.25)

    def correlation(self, theta_signal: float, theta_idler: float) -> float:
        """关联函数 E(a,b) = P(++) + P(--) - P(+-) - P(-+)。"""
        p = self.probabilities(theta_signal, theta_idler)
        return float(p[0] + p[3] - p[1] - p[2])

    # ---- 采样 -------------------------------------------------------------
    def sample(self, n: int, theta_signal: float, theta_idler: float, rng=None):
        """按 Born 规则采样 n 对，返回 (a, b)，取值均为 ±1。"""
        rng = np.random.default_rng() if rng is None else rng
        p = self.probabilities(theta_signal, theta_idler)
        idx = rng.choice(4, size=int(n), p=p)
        a = np.where(idx < 2, 1, -1)
        b = np.where(idx % 2 == 0, 1, -1)
        return a, b

    def counts(self, n: int, theta_signal: float, theta_idler: float, rng=None) -> np.ndarray:
        """采样 n 对，返回四个计数 (N++, N+-, N-+, N--)。"""
        rng = np.random.default_rng() if rng is None else rng
        p = self.probabilities(theta_signal, theta_idler)
        idx = rng.choice(4, size=int(n), p=p)
        return np.bincount(idx, minlength=4).astype(float)


# --------------------------------------------------------------------------
# CHSH
# --------------------------------------------------------------------------
CHSH_VISIBILITY_THRESHOLD = 1.0 / np.sqrt(2.0)  # ≈ 0.7071


def chsh_visibility_threshold() -> float:
    """能让 CHSH 超过 2 的最小偏振可见度。"""
    return float(CHSH_VISIBILITY_THRESHOLD)


def chsh_for_visibility(visibility):
    """理想角度下 S = 2√2·V（V 低于 0.7071 时就掉到 2 以下）。

    标量和数组都收；数组按元素算（画 S–V 曲线时要用）。
    """
    return 2.0 * np.sqrt(2.0) * np.asarray(visibility, dtype=float)


def _E_from_counts(c: np.ndarray) -> float:
    total = float(np.sum(c))
    if total <= 0:
        return 0.0
    return float((c[0] + c[3] - c[1] - c[2]) / total)


class ChshAccumulator:
    """实时累积 CHSH 统计。

    四个测量设置 (a,b)、(a,b')、(a',b)、(a',b') 各累积一堆符合计数，
    随时可以读出当前的 S 和它的统计误差。
    """

    DEFAULT_ANGLES = {"a": 0.0, "a'": 45.0, "b": 22.5, "b'": 67.5}

    def __init__(self, state: TwoPhotonState, angles: dict | None = None, seed=None):
        self.state = state
        self.angles = dict(self.DEFAULT_ANGLES)
        if angles:
            self.angles.update({k: float(v) for k, v in angles.items()})
        self.rng = np.random.default_rng(seed)
        self.counts = {k: np.zeros(4) for k in ("ab", "ab'", "a'b", "a'b'")}

    # ---- 设置 -------------------------------------------------------------
    def pair_settings(self):
        """返回 [(名字, 信号角, 闲频角), ...]。"""
        A = self.angles
        return [
            ("ab", A["a"], A["b"]),
            ("ab'", A["a"], A["b'"]),
            ("a'b", A["a'"], A["b"]),
            ("a'b'", A["a'"], A["b'"]),
        ]

    # ---- 累积 -------------------------------------------------------------
    def add(self, n: int) -> None:
        """每个设置各测 n 对。"""
        for name, ts, ti in self.pair_settings():
            self.counts[name] += self.state.counts(n, ts, ti, rng=self.rng)

    def reset(self) -> None:
        self.counts = {k: np.zeros(4) for k in self.counts}

    # ---- 读数 -------------------------------------------------------------
    def correlations(self) -> dict:
        return {name: _E_from_counts(self.counts[name]) for name, _, _ in self.pair_settings()}

    def total_pairs(self) -> int:
        return int(sum(c.sum() for c in self.counts.values()))

    def chsh(self):
        """返回 (S, 标准差)。标准差由每个 E 的 (1-E²)/N 合成。"""
        E = self.correlations()
        var = 0.0
        for name, _, _ in self.pair_settings():
            n = float(self.counts[name].sum())
            if n > 0:
                var += (1.0 - E[name] ** 2) / n
        S = E["ab"] - E["ab'"] + E["a'b"] + E["a'b'"]
        return float(S), float(np.sqrt(var))

    def summary(self) -> dict:
        S, sigma = self.chsh()
        return {
            "S": S,
            "sigma": sigma,
            "violated": bool(S - 2.0 > 0),
            "significant": bool(S - 2.0 > 2.0 * sigma) if sigma > 0 else False,
            "pairs": self.total_pairs(),
            "E": self.correlations(),
        }


# --------------------------------------------------------------------------
# 经典局域隐变量对照模型
# --------------------------------------------------------------------------
def lhv_correlation(theta_a, theta_b):
    """最经典的局域隐变量模型给出的关联函数。

    双方共享一个随机方向 λ，各自输出 sign(cos(θ-λ))。
    结果是一条三角波：E = 1 - 2|Δ|/π，而不是量子的 cos(2Δ)。

    它满足 |E| ≤ 1、且任意角度组合下 |S| ≤ 2 —— 这就是经典世界的样子。
    """
    d = np.abs((np.asarray(theta_a, dtype=float) - np.asarray(theta_b, dtype=float)) % 180.0)
    d = np.minimum(d, 180.0 - d)
    return 1.0 - 2.0 * d / 180.0


def lhv_chsh(angles: dict) -> float:
    """给定四个角度，算这个经典模型能得到的 S。"""
    return float(
        lhv_correlation(angles["a"], angles["b"])
        - lhv_correlation(angles["a"], angles["b'"])
        + lhv_correlation(angles["a'"], angles["b"])
        + lhv_correlation(angles["a'"], angles["b'"])
    )


def lhv_max_chsh(step: float = 10.0):
    """遍历所有角度，找这个经典模型能达到的最大 S。

    返回 (S_max, 对应的四个角度)。无论怎么调，它都不会超过 2.000。
    """
    g = np.arange(0.0, 180.0, float(step))
    a = g[:, None, None, None]
    ap = g[None, :, None, None]
    b = g[None, None, :, None]
    bp = g[None, None, None, :]
    S = (
        lhv_correlation(a, b)
        - lhv_correlation(a, bp)
        + lhv_correlation(ap, b)
        + lhv_correlation(ap, bp)
    )
    flat = int(np.argmax(S))
    idx = np.unravel_index(flat, S.shape)
    best = {
        "a": float(g[idx[0]]),
        "a'": float(g[idx[1]]),
        "b": float(g[idx[2]]),
        "b'": float(g[idx[3]]),
    }
    return float(S[idx]), best


# --------------------------------------------------------------------------
# HOM 干涉
# --------------------------------------------------------------------------
def hom_coincidence(delay_fs, coherence_fs: float = 200.0, indistinguishability: float = 1.0):
    """HOM 符合计数随延时的变化（示意模型）。

    两个光子完全不可分辨时，延时为零处符合计数掉到 0（"HOM 凹陷"）；
    完全可分辨时凹陷消失，符合计数保持 1/2。

    这里把"不可分辨"参数化为一个高斯凹陷，深度 = indistinguishability，
    宽度 = coherence_fs。真实装置里深度由光谱纯度、偏振、空间模匹配共同决定。
    """
    d = np.asarray(delay_fs, dtype=float)
    vis = float(np.clip(indistinguishability, 0.0, 1.0))
    return 0.5 * (1.0 - vis * np.exp(-(d / float(coherence_fs)) ** 2))


def hom_visibility_from_purity(purity: float) -> float:
    """把光谱纯度（Schmidt 纯度）折算成 HOM 凹陷深度。

    单模压缩源里，HOM 可见度约等于施密特模数倒数所定义的纯度。
    这里用最简单的一阶关系，仅作教学示意。
    """
    return float(np.clip(purity, 0.0, 1.0))
