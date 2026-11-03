"""精确的 Fock 空间线性光学演化。

设计原则：**不做概率近似**。
- 分束器用 SU(2) 的精确矩阵元，相位全保留；
- 损耗用"与真空端口做分束器 + 投影"实现，这在数学上就是振幅阻尼通道，
  不是"按概率随机删掉一个光子"那种会丢相位的做法。

光子数 ≤ 4、模式数 ≤ 8 时，用字典存占据数是够快也够清楚的。
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np

__all__ = [
    "FockState",
    "Mixture",
    "vacuum",
    "fock",
    "single_photon",
    "bs_theta",
    "apply_bs",
    "apply_phase",
    "apply_loss",
    "evolve",
    "single_photon_dist",
    "pair_dist",
]

Occ = tuple


# --------------------------------------------------------------------------
# 数学工具
# --------------------------------------------------------------------------
def bs_theta(reflectivity: float) -> float:
    """把分束器反射率 R 换成角度 θ。

    约定：a† → cosθ·a† + sinθ·b†，b† → -sinθ·a† + cosθ·b†，
    于是 R = sin²θ。R = 0.5 就是 50:50，θ = 45°。
    """
    r = float(np.clip(reflectivity, 0.0, 1.0))
    return float(np.arcsin(np.sqrt(r)))


def _bs_elements(n: int, m: int, c: float, s: float):
    """输入 |n,m⟩ 经过分束器后，各输出 |n_i, n_j⟩ 的系数。

    把 (c a† + s b†)^n (-s a† + c b†)^m / sqrt(n! m!) 展开再收集项，
    得到的就是 SU(2) 的精确矩阵元。
    """
    out = []
    for k in range(n + 1):
        for l in range(m + 1):
            ni = k + l
            nj = n + m - ni
            coeff = (
                comb(n, k)
                * comb(m, l)
                * (c ** (k + m - l))
                * (s ** (n - k))
                * ((-s) ** l)
                * np.sqrt(factorial(ni) * factorial(nj) / (factorial(n) * factorial(m)))
            )
            out.append((ni, nj, coeff))
    return out


# --------------------------------------------------------------------------
# 态
# --------------------------------------------------------------------------
class FockState:
    """一个纯态：占据数元组 -> 复振幅。"""

    def __init__(self, amps: dict, n_modes: int | None = None):
        cleaned = {tuple(k): complex(v) for k, v in amps.items()}
        cleaned = {k: v for k, v in cleaned.items() if abs(v) > 1e-14}
        if n_modes is None:
            n_modes = len(next(iter(cleaned))) if cleaned else 0
        self.n_modes = int(n_modes)
        self.amps = cleaned

    # ---- 基本量 ---------------------------------------------------------
    @property
    def norm2(self) -> float:
        return float(sum(abs(a) ** 2 for a in self.amps.values()))

    def normalized(self) -> "FockState":
        n2 = self.norm2
        if n2 <= 0:
            raise ValueError("零态无法归一化")
        k = 1.0 / np.sqrt(n2)
        return FockState({occ: a * k for occ, a in self.amps.items()}, self.n_modes)

    def dist(self) -> dict:
        """输出分布：占据数 -> 概率。"""
        return {occ: float(abs(a) ** 2) for occ, a in self.amps.items()}

    def prob(self, occ: Occ) -> float:
        return float(abs(self.amps.get(tuple(occ), 0.0)) ** 2)

    def inner(self, other: "FockState") -> complex:
        if self.n_modes != other.n_modes:
            raise ValueError("模式数不一致")
        return complex(sum(np.conj(a) * other.amps[occ] for occ, a in self.amps.items() if occ in other.amps))

    def photons(self) -> int:
        return int(sum(occ) for occ in self.amps)

    def __repr__(self) -> str:
        return f"FockState(n_modes={self.n_modes}, dim={len(self.amps)}, n={self.photons()})"


class Mixture:
    """混合态：若干纯态的统计混合。

    用来做两件事：
    1. 模式失配——把"完全不可分辨"和"完全可分辨"按可见度加权；
    2. 可分辨光子的输出——那是若干 Fock 基态的经典混合。
    """

    def __init__(self, components):
        self.components = [(float(w), st) for w, st in components if w > 0]

    @property
    def weight(self) -> float:
        return float(sum(w * st.norm2 for w, st in self.components))

    def normalized(self) -> "Mixture":
        tot = self.weight
        if tot <= 0:
            raise ValueError("零混合态无法归一化")
        return Mixture([(w / tot, st) for w, st in self.components])

    def dist(self) -> dict:
        tot = self.weight
        out: dict = {}
        if tot <= 0:
            return out
        for w, st in self.components:
            for occ, p in st.dist().items():
                out[occ] = out.get(occ, 0.0) + w * p / tot
        return out

    def fidelity_pure(self, psi: "FockState") -> float:
        """与一个纯态的重叠 ⟨ψ|ρ|ψ⟩。"""
        tot = self.weight
        if tot <= 0:
            return 0.0
        f = 0.0
        for w, st in self.components:
            f += w * abs(st.inner(psi)) ** 2
        return float(f / tot)

    def __repr__(self) -> str:
        return f"Mixture(components={len(self.components)}, weight={self.weight:.4f})"


# --------------------------------------------------------------------------
# 基本态
# --------------------------------------------------------------------------
def vacuum(n_modes: int) -> FockState:
    return FockState({(0,) * n_modes: 1.0}, n_modes)


def fock(occ) -> FockState:
    occ = tuple(int(x) for x in occ)
    return FockState({occ: 1.0}, len(occ))


def single_photon(n_modes: int, mode: int) -> FockState:
    occ = [0] * n_modes
    occ[mode] = 1
    return FockState({tuple(occ): 1.0}, n_modes)


# --------------------------------------------------------------------------
# 操作
# --------------------------------------------------------------------------
def apply_bs(state: FockState, i: int, j: int, theta: float) -> FockState:
    """在模式 i, j 之间做分束器，角度 theta（R = sin²θ）。"""
    c, s = np.cos(theta), np.sin(theta)
    out: dict = {}
    for occ, amp in state.amps.items():
        n, m = occ[i], occ[j]
        if n == 0 and m == 0:
            out[occ] = out.get(occ, 0.0) + amp
            continue
        base = list(occ)
        for ni, nj, coeff in _bs_elements(n, m, c, s):
            base[i] = ni
            base[j] = nj
            key = tuple(base)
            out[key] = out.get(key, 0.0) + amp * coeff
    return FockState(out, state.n_modes)


def apply_phase(state: FockState, mode: int, phi: float) -> FockState:
    """模式 mode 上加相移 φ。"""
    out = {occ: amp * np.exp(1j * phi * occ[mode]) for occ, amp in state.amps.items()}
    return FockState(out, state.n_modes)


def apply_loss(state: FockState, mode: int, eta: float) -> FockState:
    """透过率 η 的损耗通道。

    实现方式：接一个真空辅助模，做透射率 η 的分束器，再把辅助模投影回真空。
    这精确等价于振幅阻尼，保留相干性。
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    if eta > 1.0 - 1e-12:
        return state
    if eta <= 1e-12:
        # 全吸收：光子全没了，只剩真空分量
        n = state.n_modes
        keep = {}
        for occ, amp in state.amps.items():
            if all(x == 0 for x in occ):
                keep[occ] = amp
        return FockState(keep if keep else {(0,) * n: 0.0}, n)

    n = state.n_modes
    ext = FockState({occ + (0,): a for occ, a in state.amps.items()}, n + 1)
    theta = float(np.arccos(np.sqrt(eta)))
    ext = apply_bs(ext, mode, n, theta)
    kept = {occ[:-1]: a for occ, a in ext.amps.items() if occ[-1] == 0}
    return FockState(kept, n)


def evolve(state: FockState, ops) -> FockState:
    """按顺序执行一串操作。

    ops 里每一项是：
        ("bs", i, j, theta)
        ("phase", mode, phi)
        ("loss", mode, eta)
    """
    cur = state
    for op in ops:
        kind = op[0]
        if kind == "bs":
            cur = apply_bs(cur, op[1], op[2], op[3])
        elif kind == "phase":
            cur = apply_phase(cur, op[1], op[2])
        elif kind == "loss":
            cur = apply_loss(cur, op[1], op[2])
        else:
            raise ValueError(f"未知操作 {kind!r}")
    return cur


# --------------------------------------------------------------------------
# 可分辨光子：独立演化
# --------------------------------------------------------------------------
def single_photon_dist(ops, n_modes: int, start_mode: int) -> dict:
    """单个光子从 start_mode 出发，走完线路后的分布。"""
    st = single_photon(n_modes, start_mode)
    st = evolve(st, ops)
    return st.dist()


def pair_dist(ops, n_modes: int, mode_a: int, mode_b: int) -> dict:
    """两个**可分辨**光子的联合分布（各自独立，互不干涉）。

    返回占据数元组 -> 概率。丢光子的情形不会出现在返回值里。
    """
    da = single_photon_dist(ops, n_modes, mode_a)
    db = single_photon_dist(ops, n_modes, mode_b)

    out: dict = {}
    for oa, pa in da.items():
        for ob, pb in db.items():
            occ = tuple(x + y for x, y in zip(oa, ob))
            out[occ] = out.get(occ, 0.0) + pa * pb
    return out
