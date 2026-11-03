"""三个线路：HOM、MZI、两级干涉网络。

每个线路都返回 dict，里面既有结果也有中间量，方便报告里直接引用。

噪声模型
--------
- 损耗       : 透过率 η，用真空端口分束器精确实现
- 模式失配   : 可见度 V 插值，ρ = V·ρ_不可分辨 + (1-V)·ρ_完全可分辨
- 分束比失衡 : 反射率偏离 0.5
- 相位误差   : 相移器偏置 δφ

关于 V 插值的一点声明
--------------------
这个插值对**符合概率**是严格正确的（HOM 凹陷就是这个式子），
对**断后保真度**是一个标准近似——真实的部分可分辨态还有相干项，
这里用经典混合代替了。报告里提到的那几条结论在这个近似下成立。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fock import bs_theta, evolve, fock, pair_dist, single_photon

__all__ = ["NoiseParams", "hom", "mzi", "two_stage", "mix_dist"]


@dataclass
class NoiseParams:
    """一组噪声参数。"""

    visibility: float = 1.0        # 模式不可分辨度（HOM 凹陷深度）
    eta: float = 1.0               # 每模透过率
    reflectivity: float = 0.5      # 分束器反射率
    phase_error_deg: float = 0.0   # 相移器偏置

    def scaled(self, **kwargs) -> "NoiseParams":
        d = dict(
            visibility=self.visibility,
            eta=self.eta,
            reflectivity=self.reflectivity,
            phase_error_deg=self.phase_error_deg,
        )
        d.update(kwargs)
        return NoiseParams(**d)


def mix_dist(d_ind: dict, d_dis: dict, visibility: float) -> dict:
    """按可见度把"不可分辨"和"可分辨"两个分布混合起来。"""
    v = float(np.clip(visibility, 0.0, 1.0))
    keys = set(d_ind) | set(d_dis)
    return {k: v * d_ind.get(k, 0.0) + (1.0 - v) * d_dis.get(k, 0.0) for k in keys}


# --------------------------------------------------------------------------
# ① HOM
# --------------------------------------------------------------------------
def hom(p: NoiseParams) -> dict:
    """两个不可分辨光子同时进入分束器。

    理想情形（V=1, η=1, R=0.5）两个光子总是从同一侧出来，符合计数为 0。
    任何噪声都会把这个 0 抬起来 —— 所以符合概率本身就是噪声计。
    """
    theta = bs_theta(p.reflectivity)
    ops = [
        ("loss", 0, p.eta),
        ("loss", 1, p.eta),
        ("bs", 0, 1, theta),
    ]

    psi_ind = evolve(fock((1, 1)), ops)
    p_ind = psi_ind.prob((1, 1))
    p_bunch_ind = float(psi_ind.prob((2, 0)) + psi_ind.prob((0, 2)))

    d_dis = pair_dist(ops, 2, 0, 1)
    p_dis = float(d_dis.get((1, 1), 0.0))
    p_bunch_dis = float(d_dis.get((2, 0), 0.0) + d_dis.get((0, 2), 0.0))

    v = float(np.clip(p.visibility, 0.0, 1.0))
    p_coinc = v * p_ind + (1.0 - v) * p_dis
    p_bunch = v * p_bunch_ind + (1.0 - v) * p_bunch_dis

    return {
        "p_coinc": p_coinc,
        "p_coinc_indistinguishable": p_ind,
        "p_coinc_distinguishable": p_dis,
        "p_bunch": p_bunch,
        "theta_deg": float(np.rad2deg(theta)),
        "p_coinc_ideal": 0.0,
    }


# --------------------------------------------------------------------------
# ② MZI
# --------------------------------------------------------------------------
def mzi(p: NoiseParams, phase_deg: float) -> dict:
    """单光子马赫-曾德尔：两个分束器 + 一个相移。

    约定端口 0 是"亮端口"：理想输出 P0 = sin²(φ/2)，端口 1 是互补的 cos²(φ/2)。
    """
    theta = bs_theta(p.reflectivity)
    phi = np.deg2rad(float(phase_deg) + float(p.phase_error_deg))
    ops = [
        ("loss", 0, p.eta),
        ("bs", 0, 1, theta),
        ("phase", 1, phi),
        ("bs", 0, 1, theta),
    ]

    psi = evolve(single_photon(2, 0), ops)
    p0 = psi.prob((1, 0))
    p1 = psi.prob((0, 1))
    lost = max(0.0, 1.0 - p0 - p1)

    ideal_p0 = float(np.sin(phi / 2.0) ** 2)

    return {
        "p0": p0,
        "p1": p1,
        "p_lost": lost,
        "ideal_p0": ideal_p0,
        "deviation": p0 - ideal_p0,
        "phi_deg": float(np.rad2deg(phi)),
    }


# --------------------------------------------------------------------------
# ③ 两级干涉网络
# --------------------------------------------------------------------------
def two_stage(p: NoiseParams, phase_deg: float) -> dict:
    """第一级两个光子干涉，第二级再过一个相移 + 分束器。

    光子对在第一级被"压缩"，第二级会把这个已经被扰动过的分布再扭曲一次 ——
    用来观察噪声怎么随级数累积。
    """
    theta = bs_theta(p.reflectivity)
    phi = np.deg2rad(float(phase_deg) + float(p.phase_error_deg))
    ops = [
        ("loss", 0, p.eta),
        ("loss", 1, p.eta),
        ("bs", 0, 1, theta),
        ("phase", 1, phi),
        ("bs", 0, 1, theta),
    ]

    psi_ind = evolve(fock((1, 1)), ops)
    d_ind = psi_ind.dist()
    d_dis = pair_dist(ops, 2, 0, 1)
    d_mix = mix_dist(d_ind, d_dis, p.visibility)

    # 无噪声的理想输出，作为参照
    p_ideal = NoiseParams(visibility=1.0, eta=1.0, reflectivity=0.5, phase_error_deg=0.0)
    d_ideal = two_stage_dist_ideal(p_ideal, phase_deg)

    return {
        "dist": d_mix,
        "dist_ideal": d_ideal,
        "p_coinc": float(d_mix.get((1, 1), 0.0)),
        "p_coinc_ideal": float(d_ideal.get((1, 1), 0.0)),
    }


def two_stage_dist_ideal(p: NoiseParams, phase_deg: float) -> dict:
    """两级网络的理想（无噪声）输出分布，供参照用。"""
    theta = bs_theta(p.reflectivity)
    phi = np.deg2rad(float(phase_deg) + float(p.phase_error_deg))
    ops = [
        ("loss", 0, p.eta),
        ("loss", 1, p.eta),
        ("bs", 0, 1, theta),
        ("phase", 1, phi),
        ("bs", 0, 1, theta),
    ]
    psi = evolve(fock((1, 1)), ops)
    return psi.dist()
