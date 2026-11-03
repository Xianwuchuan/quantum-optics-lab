"""跑完全部分析，生成 report/report.md 和配图。

    python analyze.py                 # 输出到 ./report
    python analyze.py -o out          # 指定目录

报告里的每个数字都是这里现算的，不是手填的。
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
from qlnoise import analysis, metrics  # noqa: E402

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
# 报告
# --------------------------------------------------------------------------
PARAM_CN = {
    "visibility": "模式可见度 V",
    "eta": "透过率 η",
    "reflectivity": "分束器反射率 R",
    "phase_error_deg": "相移器偏置 δφ",
}


def _table(rows, header) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_report(outdir: str) -> str:
    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    p1 = fig_hom(figdir)
    p2 = fig_mzi(figdir)
    p3 = fig_two_stage(figdir)

    rows = analysis.error_budget()
    p4 = fig_budget(figdir, rows)

    err_hom = selfcheck_hom()
    err_mzi = selfcheck_mzi()

    # ---- 数字表 --------------------------------------------------------
    hom_rows = []
    for v in (1.00, 0.99, 0.97, 0.95, 0.90, 0.80):
        r = hom(NoiseParams(visibility=v))
        hom_rows.append([f"{v:.2f}", f"{r['p_coinc']:.5f}", f"{r['p_bunch']:.5f}"])

    loss_rows = []
    for e in (1.00, 0.98, 0.95, 0.90, 0.80):
        r = hom(NoiseParams(eta=e, visibility=0.97))
        loss_rows.append([f"{e:.2f}", f"{r['p_coinc']:.5f}", f"{(1 - 0.97) * e ** 2 / 2:.5f}"])

    mzi_rows = []
    for dphi in (0.0, 0.5, 1.0, 2.0, 5.0):
        r = mzi(NoiseParams(phase_error_deg=dphi, eta=0.95, reflectivity=0.51), 90.0)
        mzi_rows.append([f"{dphi:.1f}", f"{r['p0']:.4f}", f"{r['deviation']:+.4f}", f"{r['p_lost']:.4f}"])

    vis_rows = []
    for v in (1.00, 0.97, 0.95, 0.90):
        _, fids, _ = analysis.sweep_two_stage_visibility([v], phase_deg=45.0)
        vis_rows.append([f"{v:.2f}", f"{float(fids[0]):.4f}"])

    budget_rows = [
        [PARAM_CN.get(r["param"], r["param"]), f"{r['typical']:g}",
         f"{r['d_coinc']:+.5f}", f"{r['d_rate']:+.4f}"]
        for r in rows
    ]

    top_fid = rows[0]
    top_rate = max(rows, key=lambda r: abs(r["d_rate"]))
    md = f"""# 光学量子线路的噪声分析（线路层）

> 所有数字由 `analyze.py` 现算生成，不是手填的。
> 全部计算在 Fock 空间精确进行：分束器用 SU(2) 精确矩阵元，损耗用真空端口
> 分束器实现（等价于振幅阻尼，保相干），没有做概率近似。

---

## 1. 摘要

三个线路、四类噪声、一张误差预算表。核心结论：

1. **损耗和模式失配的作用完全不同。** 损耗是乘性的：它被抬高的符合概率按
   $\\eta^2$ 缩小，条件保真度基本不动，代价落在计数率上；模式失配是一阶的，
   直接破坏干涉。
2. **HOM 的符合概率本身就是噪声计。** 理想值是 0，任何噪声都会把它抬起来。
   在这个意义上，HOM 不需要额外的噪声表征手段。
3. **保真度和计数率的预算排序完全不同。** 保真度侧排第一的是
   `{PARAM_CN.get(top_fid["param"], top_fid["param"])}`；计数率侧排第一的是
   `{PARAM_CN.get(top_rate["param"], top_rate["param"])}`。用单一指标做预算，
   会漏掉一半的物理。

## 2. 方法

| 噪声 | 数学实现 | 性质 |
|---|---|---|
| 透过率 η | 与真空端口做 BS，再把辅助模投影回真空 | 精确振幅阻尼，保相位 |
| 模式可见度 V | $\\rho = V\\rho_{{\\text{{ind}}}} + (1-V)\\rho_{{\\text{{dis}}}}$ | 对符合概率严格，对保真度是标准近似 |
| 分束比失衡 | 反射率 $0.5+\\delta$ 直接进 BS 矩阵 | 酉矩阵偏离 |
| 相位误差 | 相移器偏置 $\\delta\\phi$ | 条纹偏移 |

状态用占据数字典表示，光子数 ≤ 4、模式数 ≤ 8 时维度很小，全部是精确计算。

## 3. 自检

| 检查项 | 解析式 | 最大偏差 |
|---|---|---|
| HOM 符合概率 | $(1-V)\\eta^2/2$ | {err_hom:.2e} |
| MZI 输出 | $\\sin^2(\\phi/2)$ | {err_mzi:.2e} |

偏差在数值精度范围内，说明 Fock 演化这一层是对的。
**如果这两行不为零，后面的所有结论都不要信。**

## 4. HOM：符合概率就是噪声计

![HOM](figs/01_hom.png)

{_table(hom_rows, ["可见度 V", "符合概率", "聚束概率"])}

理想情况下两个光子总是从同一侧出来，符合计数严格为 0。可见度掉到 0.97，
符合概率就到 1.5% 量级——这已经足以让后选择门的保真度明显下降。

损耗的影响是乘性的：

{_table(loss_rows, ["透过率 η（V=0.97）", "符合概率", "解析式 (1-V)η²/2"])}

注意符合概率随 η 平方下降，但**这个下降不是保真度下降**，而是成功率下降。
这是后面误差预算那条结论的根据。

## 5. MZI：相位误差 vs 分束比失衡

![MZI](figs/02_mzi.png)

{_table(mzi_rows, ["相位偏置 δφ (°)", "P0", "偏离理想值", "丢失概率"])}

两种噪声在这里的表现不一样：相位偏置让条纹**平移**，分束比失衡让条纹**变浅**，
损耗让整条曲线**压低**。三者在单条曲线上很难分开，这也是实验上要分别标定的原因。

## 6. 两级网络：噪声会累积

![两级](figs/03_two_stage.png)

{_table(vis_rows, ["可见度 V", "输出分布重叠度"])}

同样的可见度，过第二级之后保真度又掉一截。级数越多，对 V 的要求越苛刻——
这就是为什么可扩展的光子线路对源的纯度要求近乎苛刻。

## 7. 误差预算：两列，两种排序

每个噪声源单独恶化到典型实验室水平，其余保持完美：

{_table(budget_rows, ["噪声源", "典型值", "符合概率增量", "双光子探测率变化"])}

![预算](figs/04_budget.png)

左图是保真度侧，右图是计数率侧。摆在一起结论就出来了：**可见度只影响保真度，
透过率只影响计数率，两者根本不在同一个轴上**。任何把它们合并成一个数的模型，
都会同时丢掉一半信息。

## 8. 结论

1. **先修 `{PARAM_CN.get(top_fid["param"], top_fid["param"])}`。** 它单独就能把符合概率
   抬得最高；而 `{PARAM_CN.get(top_rate["param"], top_rate["param"])}` 的影响全落在计数率上，
   两者不能互相抵扣。如果几项同时存在，总量不是简单相加——耦合项要用完整的四参数扫描。
2. **不要把损耗当成退相干。** 在这个模型里，损耗只降低成功率，不降低条件保真度；
   把它折进一个 depolarizing 参数会高估它对保真度的破坏，同时低估它对计数率的破坏。
3. **HOM 可以直接当噪声计用。** 符合概率的抬升量就是噪声的累积效果，
   不需要另做层析。

## 9. 局限

- 可见度用经典混合插值，忽略了部分可分辨态里的相干项；对符合概率严格，对
  多光子线路的保真度是近似。
- 没有建模源的多对发射和探测器暗计数（属于光源/探测层）。
- 没有做到纠错层，所以不能给出"这个损耗水平能不能容错"的结论。
- 第三个线路是两级干涉网络，**不是 CNOT**：后选择 CNOT 需要 NS 门的三分束器
  分解，参数未经验证，本轮不写。

---

*生成：`python analyze.py`*
"""

    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="光学量子线路噪声分析")
    # 默认输出到 qlnoise/report，不随工作目录变
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("-o", "--out", default=os.path.join(here, "report"),
                    help="输出目录（默认 qlnoise/report）")
    args = ap.parse_args()

    setup_font()
    os.makedirs(args.out, exist_ok=True)

    print(f"\n输出目录：{os.path.abspath(args.out)}")
    print(f"  HOM 自检最大偏差：{selfcheck_hom():.2e}")
    print(f"  MZI 自检最大偏差：{selfcheck_mzi():.2e}")

    path = build_report(args.out)
    print(f"\n报告已写入 {path}\n")


if __name__ == "__main__":
    main()
