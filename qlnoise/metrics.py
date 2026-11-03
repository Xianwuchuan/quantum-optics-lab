"""度量：分布距离、断后保真度、成功概率。

说明：这里所谓的"保真度"是**输出分布的重叠**（Bhattacharyya 系数），
不是量子态的 Uhlmann 保真度。原因是线路层的噪声最后都落到"测出来是什么"
上，分布层面的度量更贴近实验能验证的量。需要态保真度时用
`Mixture.fidelity_pure()`。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize",
    "tvd",
    "bhattacharyya",
    "distribution_fidelity",
    "postselect",
    "excess_coincidence",
]


def normalize(d: dict) -> dict:
    tot = float(sum(d.values()))
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in d.items()}


def tvd(d1: dict, d2: dict) -> float:
    """两个分布的总变差距离，取值 0–1。"""
    keys = set(d1) | set(d2)
    return 0.5 * float(sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys))


def bhattacharyya(d1: dict, d2: dict) -> float:
    keys = set(d1) | set(d2)
    return float(sum(np.sqrt(max(d1.get(k, 0.0), 0.0) * max(d2.get(k, 0.0), 0.0)) for k in keys))


def distribution_fidelity(d1: dict, d2: dict) -> float:
    """分布重叠，0–1。两个分布相同时为 1。"""
    return bhattacharyya(normalize(d1), normalize(d2))


def postselect(dist: dict, predicate) -> tuple[dict, float]:
    """按条件筛选输出，返回（归一化后的条件分布, 成功概率）。"""
    kept = {k: v for k, v in dist.items() if predicate(k)}
    p_success = float(sum(kept.values()))
    if p_success <= 0:
        return {}, 0.0
    return {k: v / p_success for k, v in kept.items()}, p_success


def excess_coincidence(p: float, p_ideal: float = 0.0) -> float:
    """符合概率超出理想值的部分，直接当作噪声计。"""
    return float(p - p_ideal)
