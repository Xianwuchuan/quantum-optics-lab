"""qlnoise —— 光学量子线路的噪声分析（线路层）。

    from qlnoise import NoiseParams, hom, mzi, two_stage

    hom(NoiseParams(visibility=0.95))["p_coinc"]

模块划分
--------
fock.py      精确 Fock 空间演化（分束器、相移、损耗）
circuits.py  三个线路
metrics.py   分布距离、断后保真度、成功概率
analysis.py  参数扫描与误差预算
"""

from .circuits import NoiseParams, hom, mzi, two_stage
from .fock import FockState, Mixture, evolve, fock, single_photon
from .metrics import (
    distribution_fidelity,
    postselect,
    tvd,
)

__all__ = [
    "NoiseParams",
    "hom",
    "mzi",
    "two_stage",
    "FockState",
    "Mixture",
    "evolve",
    "fock",
    "single_photon",
    "distribution_fidelity",
    "postselect",
    "tvd",
]
