"""matplotlib 图形：静态版和 Streamlit 版共用。

所有图都返回 matplotlib.figure.Figure，调用方自己决定是 savefig 还是 st.pyplot。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import physics
import source

__all__ = [
    "use_chinese_font",
    "plot_phase_matching",
    "plot_correlation",
    "plot_chsh_history",
    "plot_hom",
    "plot_polar_map",
]

_C_QUANTUM = "#c0392b"
_C_CLASSIC = "#2c3e50"
_C_IDLER = "#2e86c1"
_C_SIGNAL = "#d35400"


def use_chinese_font() -> None:
    """中文标签不出方块。没有中文字体时自动退回 DejaVu Sans（会显示方框）。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def grid(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)


# --------------------------------------------------------------------------
# 场景① 相位匹配 / 出射锥
# --------------------------------------------------------------------------
def plot_phase_matching(lam_p_um: float = 0.405, lam_min_um: float = 0.65, lam_max_um: float = 1.00):
    use_chinese_font()
    curve = source.tuning_curve(
        lam_p_um=lam_p_um, lam_min_um=lam_min_um, lam_max_um=lam_max_um, n_points=300
    )
    screen = source.cone_screen(
        lam_p_um=lam_p_um, lam_min_um=lam_min_um, lam_max_um=lam_max_um, n_points=120, n_azimuth=48
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    if curve["lam_s"].size:
        ax1.plot(curve["lam_s"] * 1e3, curve["theta_s"], "o-", ms=2.5, lw=1.2,
                 color=_C_SIGNAL, label="信号光 θs")
        ax1.plot(curve["lam_i"] * 1e3, curve["theta_i"], "o-", ms=2.5, lw=1.2,
                 color=_C_IDLER, label="闲频光 θi")
        ax1.legend(loc="upper center", frameon=False)
    else:
        ax1.text(0.5, 0.5, "该参数下无物理解", ha="center", va="center", transform=ax1.transAxes)

    ax1.axvline(2 * lam_p_um * 1e3, ls=":", lw=1, color="0.5")
    ax1.text(2 * lam_p_um * 1e3, ax1.get_ylim()[1] * 0.9, " 简并点", fontsize=9, color="0.4")
    ax1.set_xlabel("波长 (nm)")
    ax1.set_ylabel("相对泵浦的出射角 (°)")
    ax1.set_title(f"调谐曲线（θ_pm = {curve['theta_pm']:.1f}°）", fontsize=11)
    grid(ax1)

    if screen["xs_s"].size:
        ax2.scatter(screen["xs_s"], screen["ys_s"], s=2, c=_C_SIGNAL, alpha=0.5, label="信号光")
        ax2.scatter(screen["xs_i"], screen["ys_i"], s=2, c=_C_IDLER, alpha=0.5, label="闲频光")
        ax2.legend(loc="upper right", markerscale=5, frameon=False, fontsize=9)
    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.axvline(0, color="0.85", lw=0.8)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("屏上位置 (°)")
    ax2.set_ylabel("屏上位置 (°)")
    ax2.set_title("垂直于泵浦的屏上看到的环", fontsize=11)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 场景② 关联函数
# --------------------------------------------------------------------------
def plot_polar_map(visibility: float = 1.0):
    """两个探测器方向的符合计数极坐标图。"""
    use_chinese_font()
    state = physics.TwoPhotonState(visibility=visibility)
    ang = np.linspace(0, 180, 181)
    counts = np.array([state.probabilities(0.0, a)[0] for a in ang])

    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(2 * ang), counts / counts.max(), color=_C_QUANTUM, lw=1.8)
    ax.set_title("符合计数随相对夹角变化", fontsize=11, pad=12)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7)
    fig.tight_layout()
    return fig


def plot_correlation(visibility: float = 1.0):
    use_chinese_font()
    delta = np.linspace(0, 180, 361)
    e_quantum = visibility * np.cos(np.deg2rad(2.0 * delta))
    e_classic = physics.lhv_correlation(0.0, delta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(delta, e_quantum, color=_C_QUANTUM, lw=2,
             label=f"量子：V·cos(2Δ)，V = {visibility:.2f}")
    ax1.plot(delta, e_classic, "--", color=_C_CLASSIC, lw=1.6, label="经典局域模型：三角波")
    ax1.axhline(0, color="0.8", lw=0.8)
    ax1.set_xlabel("两个偏振片的相对夹角 Δ (°)")
    ax1.set_ylabel("关联函数 E(a,b)")
    ax1.set_ylim(-1.15, 1.15)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    ax1.set_title("量子关联是余弦，经典关联只能到三角波", fontsize=11)
    grid(ax1)

    v = np.linspace(0, 1, 201)
    ax2.plot(v, physics.chsh_for_visibility(v), color=_C_QUANTUM, lw=2, label="S = 2√2·V")
    ax2.axhline(2.0, color=_C_CLASSIC, ls="--", lw=1.4, label="经典上限 S = 2")
    ax2.axhline(2 * np.sqrt(2), color="0.6", ls=":", lw=1.2, label="Tsirelson 界 2√2 ≈ 2.83")
    ax2.axvline(physics.chsh_visibility_threshold(), color="0.6", ls=":", lw=1.0)
    ax2.text(physics.chsh_visibility_threshold() + 0.01, 1.55, "V = 1/√2", fontsize=8, color="0.4")
    ax2.set_xlabel("偏振可见度 V")
    ax2.set_ylabel("CHSH 值 S")
    ax2.set_ylim(0, 3.1)
    ax2.legend(loc="lower right", frameon=False, fontsize=8)
    ax2.set_title("可见度掉到 0.707 以下，就再也违反不了 Bell", fontsize=11)
    grid(ax2)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 场景③ CHSH 累积
# --------------------------------------------------------------------------
def plot_chsh_history(history_n, history_s, history_sigma=None):
    use_chinese_font()
    n = np.asarray(history_n, dtype=float)
    s = np.asarray(history_s, dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    if history_sigma is not None and len(history_sigma) == len(s):
        sig = np.asarray(history_sigma, dtype=float)
        ax.fill_between(n, s - sig, s + sig, color=_C_QUANTUM, alpha=0.18, lw=0)
    ax.plot(n, s, "o-", ms=3, lw=1.4, color=_C_QUANTUM, label="实测 S")
    ax.axhline(2.0, color=_C_CLASSIC, ls="--", lw=1.4, label="经典上限 S = 2")
    ax.axhline(2 * np.sqrt(2), color="0.6", ls=":", lw=1.2, label="Tsirelson 界 2√2")
    ax.set_xscale("log")
    ax.set_xlabel("累积光子对数 N")
    ax.set_ylabel("CHSH 值 S")
    ax.set_ylim(0, 3.2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("S 随统计量收敛：N 太小的时候什么都说不准", fontsize=11)
    grid(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 场景⑤ HOM
# --------------------------------------------------------------------------
def plot_hom(purities=(1.0, 0.7, 0.4), coherence_fs: float = 200.0):
    use_chinese_font()
    delay = np.linspace(-600, 600, 481)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for p in purities:
        ax.plot(delay, physics.hom_coincidence(delay, coherence_fs, p), lw=1.8,
                label=f"光谱纯度 {p:.1f}")
    ax.axhline(0.5, color="0.7", ls=":", lw=1.0)
    ax.text(-590, 0.515, "完全可分辨时的符合水平 1/2", fontsize=8, color="0.4")
    ax.set_xlabel("两路光程差 Δτ (fs)")
    ax.set_ylabel("符合计数（归一化）")
    ax.set_ylim(-0.05, 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("HOM 凹陷深度 = 两个光子有多不可分辨", fontsize=11)
    grid(ax)
    fig.tight_layout()
    return fig
