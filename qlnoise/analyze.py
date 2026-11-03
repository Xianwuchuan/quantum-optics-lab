"""跑完全部分析，生成 4 张配图，并把关键数字打到终端。

    python analyze.py                 # 输出到 qlnoise/figs
    python analyze.py -o out          # 指定目录
"""

from __future__ import annotations

import argparse
import os
import sys

# 允许直接 `python qlnoise/analyze.py` 运行：把仓库根目录放进搜索路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qlnoise import NoiseParams, hom, mzi  # noqa: E402
from qlnoise import analysis  # noqa: E402

C_Q = "#c0392b"
C_C = "#2c3e50"
C_G = "#7f8c8d"


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _grid(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 自检：拿解析式对一遍
# --------------------------------------------------------------------------
def selfcheck_hom() -> float:
    """R=0.5、对称损耗时，符合概率应严格等于 (1-V)·η²/2。"""
    worst = 0.0
    for v in np.linspace(0.0, 1.0, 21):
        for e in (0.6, 0.75, 0.85, 0.95, 1.0):
            got = hom(NoiseParams(visibility=v, eta=e))["p_coinc"]
            ref = (1.0 - v) * e ** 2 / 2.0
            worst = max(worst, abs(got - ref))
    return worst


def selfcheck_mzi() -> float:
    """无噪声时 P1 应严格等于 sin²(φ/2)。"""
    worst = 0.0
    for ph in np.linspace(0.0, 360.0, 73):
        r = mzi(NoiseParams(), ph)
        worst = max(worst, abs(r["p0"] - r["ideal_p0"]))
    return worst


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def fig_hom(figdir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    vs = np.linspace(0, 1, 51)
    for eta, ls in ((1.00, "-"), (0.95, "--"), (0.90, ":")):
        _, ys = analysis.sweep_hom_visibility(vs, eta=eta)
        ax1.plot(vs, ys, ls, color=C_Q if eta == 1.0 else C_G, lw=1.8,
                 label=f"数值 η={eta:.2f}")
        ax1.plot(vs, (1 - vs) * eta ** 2 / 2, color=C_C, lw=0.9, alpha=0.6)
    ax1.set_xlabel("模式可见度 V")
    ax1.set_ylabel("符合概率")
    ax1.set_title("HOM：符合概率就是噪声计（实线=解析式）", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    _grid(ax1)

    ds = np.linspace(-0.05, 0.05, 41)
    _, ys = analysis.sweep_hom_imbalance(ds)
    ax2.plot(ds * 100, ys, color=C_Q, lw=1.8)
    ax2.set_xlabel("分束器反射率偏差 δR (%)")
    ax2.set_ylabel("符合概率")
    ax2.set_title("分束器失衡同样会抬起符合计数", fontsize=11)
    _grid(ax2)

    return _save(fig, os.path.join(figdir, "01_hom.png"))


def fig_mzi(figdir: str) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    phases = np.linspace(0, 360, 361)

    _, _, ideal, _ = analysis.sweep_mzi_phase(phases, NoiseParams())
    ax.plot(phases, ideal, color=C_C, lw=1.6, label="理想 sin²(φ/2)")

    for dphi, ls in ((0.0, "-"), (5.0, "--")):
        _, y, _, _ = analysis.sweep_mzi_phase(
            phases, NoiseParams(phase_error_deg=dphi, eta=0.95, reflectivity=0.51)
        )
        ax.plot(phases, y, ls, color=C_Q if dphi == 0 else "#e67e22", lw=1.4,
                label=f"含噪声 δφ={dphi:.0f}°, η=0.95, R=0.51")

    ax.set_xlabel("相对相位 φ (°)")
    ax.set_ylabel("输出端 1 的概率")
    ax.set_title("MZI：相位误差表现为条纹偏移，损耗表现为整体压低", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _grid(ax)
    return _save(fig, os.path.join(figdir, "02_mzi.png"))


def fig_two_stage(figdir: str) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    vs = np.linspace(0.5, 1.0, 51)
    _, fids, _ = analysis.sweep_two_stage_visibility(vs, phase_deg=45.0)
    ax.plot(vs, fids, color=C_Q, lw=1.8)
    ax.set_xlabel("模式可见度 V")
    ax.set_ylabel("输出分布重叠度")
    ax.set_ylim(0, 1.02)
    ax.set_title("两级网络：同样的 V，第二级再吃掉一部分", fontsize=11)
    _grid(ax)
    return _save(fig, os.path.join(figdir, "03_two_stage.png"))


def fig_budget(figdir: str, rows) -> str:
    names = [PARAM_CN.get(r["param"], r["param"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))

    ax1.barh(names[::-1], [r["d_coinc"] for r in rows][::-1], color=C_Q)
    ax1.set_xlabel("符合概率增量（理想值为 0）")
    ax1.set_title("保真度侧：噪声把符合计数抬起来多少", fontsize=11)

    ax2.barh(names[::-1], [r["d_rate"] for r in rows][::-1], color=C_C)
    ax2.set_xlabel("双光子探测率变化")
    ax2.set_title("计数率侧：噪声吃掉多少光子", fontsize=11)

    _grid(ax1)
    _grid(ax2)
    return _save(fig, os.path.join(figdir, "04_budget.png"))


# --------------------------------------------------------------------------
# 终端输出
# --------------------------------------------------------------------------
PARAM_CN = {
    "visibility": "模式可见度 V",
    "eta": "透过率 η",
    "reflectivity": "分束器反射率 R",
    "phase_error_deg": "相移偏置 δφ",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="光学量子线路噪声分析")
    # 默认输出到脚本旁边的 figs/，不随工作目录变
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("-o", "--out", default=os.path.join(here, "figs"),
                    help="输出目录（默认 qlnoise/figs）")
    args = ap.parse_args()

    setup_font()
    os.makedirs(args.out, exist_ok=True)

    print(f"\n输出目录：{os.path.abspath(args.out)}")
    print(f"  HOM 自检最大偏差：{selfcheck_hom():.2e}")
    print(f"  MZI 自检最大偏差：{selfcheck_mzi():.2e}")

    rows = analysis.error_budget()
    paths = [
        fig_hom(args.out),
        fig_mzi(args.out),
        fig_two_stage(args.out),
        fig_budget(args.out, rows),
    ]
    print("\n已生成：")
    for p in paths:
        print(f"  [ok] {p}")

    print("\nHOM 参考值")
    for v in (1.00, 0.97, 0.90):
        p = hom(NoiseParams(visibility=v))["p_coinc"]
        print(f"  V={v:.2f}  符合概率 {p:.5f}")

    print("\n误差预算（每个噪声源单独恶化，其余保持完美）")
    print(f"  {'噪声源':<14}{'典型值':>8}{'符合概率增量':>14}{'探测率变化':>14}")
    for r in rows:
        name = PARAM_CN.get(r["param"], r["param"])
        print(f"  {name:<14}{r['typical']:>8g}{r['d_coinc']:>+14.5f}{r['d_rate']:>+14.4f}")
    print()


if __name__ == "__main__":
    main()
