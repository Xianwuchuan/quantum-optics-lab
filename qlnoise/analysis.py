"""参数扫描与误差预算。

报告里最有用的一张表是 `error_budget()`：把每个噪声源单独恶化到"典型实验室
水平"，看指标掉多少，然后排序。预算有限的时候先修排在前面那个。
"""

from __future__ import annotations

import numpy as np

from .circuits import NoiseParams, hom, mzi, two_stage
from .metrics import distribution_fidelity

__all__ = [
    "sweep_hom_visibility",
    "sweep_hom_loss",
    "sweep_hom_imbalance",
    "sweep_mzi_phase",
    "sweep_two_stage_visibility",
    "error_budget",
]

# 典型实验室水平的单参数恶化量（只恶化一个，其余保持完美）
TYPICAL_DEGRADATION = {
    "visibility": 0.97,        # HOM 可见度 97%
    "eta": 0.95,               # 每模 0.22 dB 损耗
    "reflectivity": 0.505,     # 分束器 1% 失衡
    "phase_error_deg": 0.5,    # 相移器偏置 0.5°
}


def sweep_hom_visibility(values, eta: float = 1.0, reflectivity: float = 0.5):
    xs = np.asarray(values, dtype=float)
    ys = [hom(NoiseParams(visibility=v, eta=eta, reflectivity=reflectivity))["p_coinc"] for v in xs]
    return xs, np.asarray(ys)


def sweep_hom_loss(values, visibility: float = 1.0, reflectivity: float = 0.5):
    xs = np.asarray(values, dtype=float)
    ys = [hom(NoiseParams(visibility=visibility, eta=e, reflectivity=reflectivity))["p_coinc"] for e in xs]
    return xs, np.asarray(ys)


def sweep_hom_imbalance(deltas, visibility: float = 1.0, eta: float = 1.0):
    xs = np.asarray(deltas, dtype=float)
    ys = [
        hom(NoiseParams(visibility=visibility, eta=eta, reflectivity=0.5 + d))["p_coinc"]
        for d in xs
    ]
    return xs, np.asarray(ys)


def sweep_mzi_phase(phase_deg, params: NoiseParams | None = None):
    p = params or NoiseParams()
    xs = np.asarray(phase_deg, dtype=float)
    ys, ideal, dev = [], [], []
    for x in xs:
        r = mzi(p, float(x))
        ys.append(r["p0"])
        ideal.append(r["ideal_p0"])
        dev.append(r["deviation"])
    return xs, np.asarray(ys), np.asarray(ideal), np.asarray(dev)


def sweep_two_stage_visibility(values, phase_deg: float = 45.0, eta: float = 1.0):
    xs = np.asarray(values, dtype=float)
    fids, coinc = [], []
    for v in xs:
        r = two_stage(NoiseParams(visibility=v, eta=eta), phase_deg)
        fids.append(distribution_fidelity(r["dist"], r["dist_ideal"]))
        coinc.append(r["p_coinc"])
    return xs, np.asarray(fids), np.asarray(coinc)


def error_budget(base: NoiseParams | None = None):
    """单参数误差预算：同时看"保真度侧"和"计数率侧"。

    每个噪声源单独恶化到典型值，其余保持完美：

    - 保真度侧：HOM 符合概率的抬升量（理想值严格是 0）
    - 计数率侧：双光子被同时探测到的概率变化

    这两列排序通常完全不同 —— 这正是不能用单一指标做预算的原因。
    """
    base = base or NoiseParams()
    ref = hom(base)
    p_ref = float(ref["p_coinc"])
    t_ref = float(ref["p_bunch"] + ref["p_coinc"])

    rows = []
    for key, bad in TYPICAL_DEGRADATION.items():
        r = hom(base.scaled(**{key: bad}))
        p = float(r["p_coinc"])
        t = float(r["p_bunch"] + r["p_coinc"])
        rows.append(
            {
                "param": key,
                "typical": bad,
                "p_coinc": p,
                "d_coinc": p - p_ref,
                "p_detect": t,
                "d_rate": t - t_ref,
            }
        )

    rows.sort(key=lambda r: abs(r["d_coinc"]), reverse=True)
    return rows
