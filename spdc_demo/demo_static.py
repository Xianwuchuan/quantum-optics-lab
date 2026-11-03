"""静态版：一次生成全部教学图，不依赖 Streamlit。

    python demo_static.py                 # 输出到 ./figs
    python demo_static.py -o out --visibility 0.9

生成的 PNG 适合直接贴进 PPT 或讲义。
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # 无窗口环境也能跑

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figures  # noqa: E402
import physics  # noqa: E402
import source  # noqa: E402


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    # 用 ASCII 前缀：Windows 控制台默认 GBK，打 "✓" 会直接抛 UnicodeEncodeError
    print(f"  [ok] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPDC 纠缠光子教学演示（静态版）")
    # 默认输出到脚本自己的 figs/，不随工作目录变
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("-o", "--out", default=os.path.join(here, "figs"),
                        help="输出目录（默认 spdc_demo/figs）")
    parser.add_argument("--visibility", type=float, default=1.0, help="偏振可见度 V")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lam-p", type=float, default=405.0, help="泵浦波长 (nm)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lam_p = args.lam_p / 1000.0

    figures.use_chinese_font()

    print(f"\n泵浦波长 {args.lam_p:.0f} nm")

    # ---- 场景① --------------------------------------------------------
    try:
        theta_pm = source.phase_matching_angle(lam_p)
        print(f"  相位匹配角 θ_pm = {theta_pm:.2f}°")
    except ValueError as exc:
        print(f"  ! 相位匹配角算不出来：{exc}")

    print("\n[1] 相位匹配与出射锥")
    _save(figures.plot_phase_matching(lam_p_um=lam_p), os.path.join(args.out, "01_phase_matching.png"))

    # ---- 场景② --------------------------------------------------------
    print("\n[2] 关联函数")
    _save(figures.plot_correlation(args.visibility), os.path.join(args.out, "02_correlation.png"))
    _save(figures.plot_polar_map(args.visibility), os.path.join(args.out, "03_polar_map.png"))

    # ---- 场景③ --------------------------------------------------------
    print("\n[3] CHSH 累积")
    state = physics.TwoPhotonState(visibility=args.visibility)
    acc = physics.ChshAccumulator(state, seed=args.seed)
    hist_n, hist_s, hist_sig = [], [], []
    rng_chunks = [100, 400, 500, 1000, 3000, 5000, 10000, 30000]
    for chunk in rng_chunks:
        acc.add(chunk)
        s, sig = acc.chsh()
        hist_n.append(acc.total_pairs())
        hist_s.append(s)
        hist_sig.append(sig)
        print(f"  N = {acc.total_pairs():>7,}   S = {s:.3f} ± {sig:.3f}")

    _save(
        figures.plot_chsh_history(hist_n, hist_s, hist_sig),
        os.path.join(args.out, "04_chsh_convergence.png"),
    )

    lhv_max, best = physics.lhv_max_chsh(step=10.0)
    print(f"  经典局域模型最大 S = {lhv_max:.3f}（角度遍历）")
    print(f"  量子理论最大 S = {2 * np.sqrt(2):.3f}")

    # ---- 场景⑤ --------------------------------------------------------
    print("\n[4] HOM 干涉")
    _save(figures.plot_hom(), os.path.join(args.out, "05_hom.png"))

    print(f"\n完成，图已写入 {os.path.abspath(args.out)}\n")


if __name__ == "__main__":
    main()
