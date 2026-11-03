"""SPDC 纠缠光子 —— 交互式教学演示（Streamlit 版）。

运行：
    streamlit run app.py

五个场景：
    ① 光子对从哪来    相位匹配与出射锥
    ② 随机 vs 关联    单次不可预测，关联可预测
    ③ CHSH            定量地跨过经典上限
    ④ 现实不完美      暗计数、效率、多对发射把 S 拉回 2 以下
    ⑤ HOM 干涉        不可分辨性的直接体现

提醒：屏幕上的一切都是模型推演，不是实验测量。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import figures
import physics
import source

st.set_page_config(page_title="SPDC 纠缠光子教学演示", layout="wide")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _verdict(s: float, sigma: float) -> None:
    """根据 S 和误差给出结论。"""
    if sigma <= 0:
        st.info("还没有数据。")
        return
    if s - 2.0 > 2.0 * sigma:
        st.success(f"S = {s:.3f} ± {sigma:.3f} —— 超出经典上限 {((s - 2.0) / sigma):.1f} 个标准差。")
    elif s > 2.0:
        st.warning(f"S = {s:.3f} ± {sigma:.3f} —— 看着超过 2，但统计量还不够，别下结论。")
    else:
        st.error(f"S = {s:.3f} ± {sigma:.3f} —— 没有超过经典上限。")


@st.cache_data(show_spinner=False)
def _lhv_max(step: float = 10.0):
    s, angles = physics.lhv_max_chsh(step)
    return s, angles


def imperfect_chsh(visibility, eta, dark_cps, mu, window_ns, pair_rate):
    """现象学的"现实世界"模型（示意，不是严格推导）。

    真实符合率  : R_true = R_pair · η²
    单路计数率  : R_s = R_pair · η + 暗计数
    偶然符合    : R_acc = 2·τ·R_s1·R_s2 + 多对发射贡献
    有效可见度  : V_eff = V · R_true/(R_true + R_acc)
    """
    r_true = pair_rate * eta ** 2
    r_s = pair_rate * eta + dark_cps
    r_acc = 2.0 * window_ns * 1e-9 * r_s * r_s + pair_rate * mu * eta ** 2
    snr = r_true / (r_true + r_acc) if (r_true + r_acc) > 0 else 0.0
    v_eff = visibility * snr
    return {
        "r_true": r_true,
        "r_acc": r_acc,
        "snr": snr,
        "v_eff": v_eff,
        "S": 2.0 * np.sqrt(2.0) * v_eff,
        "accidentals_ratio": r_acc / r_true if r_true > 0 else float("inf"),
    }


# --------------------------------------------------------------------------
# 侧边栏
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("全局设置")
    visibility = st.slider("偏振可见度 V", 0.50, 1.00, 1.00, 0.01,
                           help="真实源永远到不了 1。V < 0.707 时 CHSH 必然低于 2。")
    seed = st.number_input("随机种子", min_value=0, max_value=10 ** 6, value=42, step=1,
                           help="固定种子 → 结果可复现。这也是伪随机的一个特征。")
    st.divider()
    st.caption(
        "所有数字都是模型算出来的，不是实验室测出来的。"
        "真实装置的产率、光谱、抖动，需要你自己标定。"
    )

state = physics.TwoPhotonState(visibility=visibility)

st.title("SPDC 纠缠光子：从光子对到 Bell 不等式")
st.caption(
    "泵浦光打进 BBO 晶体，一个光子变成一对纠缠光子。"
    "下面五个场景，从「它怎么产生的」一路走到「为什么经典世界解释不了它」。"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 光子对从哪来", "② 随机 vs 关联", "③ CHSH", "④ 现实不完美", "⑤ HOM 干涉"]
)


# ==========================================================================
# 场景 ①
# ==========================================================================
with tab1:
    st.subheader("能量守恒 + 动量守恒 = 一对光子")
    st.markdown(
        "泵浦光子（蓝紫，405 nm）在非线性晶体里湮灭，产生两个光子。"
        "它们必须同时满足两个守恒律："
    )
    st.latex(r"\frac{1}{\lambda_p}=\frac{1}{\lambda_s}+\frac{1}{\lambda_i}"
             r"\qquad\qquad \vec{k}_p=\vec{k}_s+\vec{k}_i")
    st.markdown(
        "第二个式子叫**相位匹配**。因为晶体有双折射，转动晶体角度可以调出一个"
        "「相位匹配角」θ_pm，让这两个式子同时成立。共线（简并）时两个光子波长都是 810 nm；"
        "一旦偏离简并，它们就只能以某个角度成对飞出 —— 这就是屏幕上那圈光环的来历。"
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        lam_p = st.slider("泵浦波长 (nm)", 380, 450, 405, 5) / 1000.0
    with c2:
        try:
            theta_pm = source.phase_matching_angle(lam_p)
            st.metric("相位匹配角 θ_pm", f"{theta_pm:.2f}°",
                      help="type-I：泵浦走 e 光，信号/闲频走 o 光")
        except ValueError as exc:
            theta_pm = float("nan")
            st.error(f"算不出相位匹配角：{exc}")
    with c3:
        st.markdown(
            f"简并波长 = 2 × {lam_p * 1e3:.0f} nm = **{2 * lam_p * 1e3:.0f} nm**；"
            "偏离简并的光子对以某个角度成对飞出。"
        )

    fig1 = figures.plot_phase_matching(lam_p_um=lam_p)
    _show(fig1)

    st.info(
        "**这里哪部分是真的**：折射率用 BBO 的 Sellmeier 方程算，相位匹配角和出射角是"
        "真解出来的，不是画上去的。405 nm 泵浦时 θ_pm ≈ 28.8°，和实验手册一致。\n\n"
        "**哪部分是近似的**：没有算泵浦光束的发散和晶体长度带来的接受角，"
        "所以这里的环是理想细线；真实实验里它有一定宽度和亮度分布。"
    )


# ==========================================================================
# 场景 ②
# ==========================================================================
with tab2:
    st.subheader("单次看是抛硬币，并排看是照镜子")
    st.markdown(
        "把信号光一路接到偏振片 + 探测器，闲频光一路同样处理。"
        "转动任意一边的偏振片，**单路计数率几乎不变**（一直在 50% 附近）；"
        "但两路的**符合**却强烈依赖两个偏振片的相对夹角。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        th_s = st.slider("信号侧偏振片角度 (°)", 0, 180, 0, 1)
    with c2:
        th_i = st.slider("闲频侧偏振片角度 (°)", 0, 180, 0, 1)
    with c3:
        n_pairs = st.slider("本次测量对数", 50, 2000, 400, 50)

    if st.button("测量", key="btn_tab2", type="primary"):
        rng = np.random.default_rng(int(seed))
        a, b = state.sample(n_pairs, th_s, th_i, rng=rng)
        p = state.probabilities(th_s, th_i)

        st.session_state["tab2"] = {
            "a": a, "b": b, "p": p, "th_s": th_s, "th_i": th_i,
        }

    data = st.session_state.get("tab2")
    if data is None:
        st.info("点「测量」开始。")
    else:
        a, b, p = data["a"], data["b"], data["p"]
        n = len(a)

        m1, m2, m3 = st.columns(3)
        m1.metric("信号通道通过率", f"{(a > 0).mean():.1%}", help="单看这一路，永远在 50% 附近")
        m2.metric("闲频通道通过率", f"{(b > 0).mean():.1%}")
        agree = (a == b).mean()
        m3.metric("两路一致率", f"{agree:.1%}", help="角度对齐时接近 100%")

        # 点击序列
        n_show = min(n, 300)
        seq = np.zeros((2, n_show))
        seq[0, np.where(a[:n_show] > 0)[0]] = 1.0
        seq[1, np.where(b[:n_show] > 0)[0]] = 1.0

        fig, ax = plt.subplots(figsize=(11, 2.2))
        ax.imshow(seq, aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=(-0.5, n_show - 0.5, 1.5, -0.5))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["信号", "闲频"])
        ax.set_xlabel("光子对序号")
        ax.set_title("两路的点击序列：单看是噪声，并排看是同步", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        _show(fig)

        # 四重符合柱状图
        counts = np.bincount(np.where(a > 0, 0, 2) + np.where(b > 0, 0, 1), minlength=4)
        labels = ["++", "+-", "-+", "--"]
        figc, axc = plt.subplots(figsize=(4.6, 3.2))
        axc.bar(labels, counts, color=["#2e86c1", "#95a5a6", "#95a5a6", "#c0392b"])
        axc.set_ylabel("符合计数")
        axc.set_title("四种联合结果的计数", fontsize=10)
        figc.tight_layout()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            _show(figc)
        with col_b:
            st.markdown(
                f"测量角度：信号 **{data['th_s']}°**，闲频 **{data['th_i']}°**。\n\n"
                f"理论概率：(++) {p[0]:.3f}　(+-) {p[1]:.3f}　(-+) {p[2]:.3f}　(--) {p[3]:.3f}\n\n"
                "注意：**你没法预测下一次落在哪一格**。理论只告诉你四个格子的相对高度，"
                "每一次点击仍然是随机的。这就是量子随机性的样子。"
            )

    st.warning(
        "这里有个容易踩的坑：如果代码里直接写 `if 信号通过: 闲频也通过`，"
        "那也能得到完美关联 —— 但那是把量子预测当输入喂进去，属于循环论证。"
        "本演示是先用态矢量算四个概率，再按概率采样，没有预设答案。"
    )


# ==========================================================================
# 场景 ③
# ==========================================================================
with tab3:
    st.subheader("四个角度，一个数字，跨过 2")
    st.markdown(
        "CHSH 把四个不同角度组合下的关联函数拼起来："
    )
    st.latex(r"S=E(a,b)-E(a,b')+E(a',b)+E(a',b')"
             r"\qquad |S|_{\mathrm{cl}}\le 2,\quad |S|_{\mathrm{q}}\le 2\sqrt{2}")
    st.markdown(
        "局域隐变量理论要求 S 不超过 2；量子力学允许到 2√2 ≈ 2.828。"
        "下面的默认角度 (0°, 45°, 22.5°, 67.5°) 就是最优的那组。"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ang_a = st.slider("a (°)", 0.0, 180.0, 0.0, 0.5, format="%.1f")
    with c2:
        ang_ap = st.slider("a' (°)", 0.0, 180.0, 45.0, 0.5, format="%.1f")
    with c3:
        ang_b = st.slider("b (°)", 0.0, 180.0, 22.5, 0.5, format="%.1f")
    with c4:
        ang_bp = st.slider("b' (°)", 0.0, 180.0, 67.5, 0.5, format="%.1f")

    angles = {"a": ang_a, "a'": ang_ap, "b": ang_b, "b'": ang_bp}

    # 可见度变了要重建累积器
    if st.session_state.get("acc_v") != visibility:
        st.session_state["acc"] = physics.ChshAccumulator(state, angles=angles, seed=int(seed))
        st.session_state["acc_v"] = visibility
        st.session_state["hist_n"] = []
        st.session_state["hist_s"] = []
        st.session_state["hist_sig"] = []

    acc: physics.ChshAccumulator = st.session_state["acc"]

    # 角度改了要重建（但保留可见度）
    if st.session_state.get("acc_angles") != angles:
        acc.angles.update(angles)
        st.session_state["acc_angles"] = dict(angles)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("每组累积 500 对", key="btn_add500"):
            acc.add(500)
            s, sig = acc.chsh()
            st.session_state["hist_n"].append(acc.total_pairs())
            st.session_state["hist_s"].append(s)
            st.session_state["hist_sig"].append(sig)
    with b2:
        if st.button("每组累积 5000 对", key="btn_add5000"):
            acc.add(5000)
            s, sig = acc.chsh()
            st.session_state["hist_n"].append(acc.total_pairs())
            st.session_state["hist_s"].append(s)
            st.session_state["hist_sig"].append(sig)
    with b3:
        if st.button("重置", key="btn_reset"):
            acc.reset()
            st.session_state["hist_n"] = []
            st.session_state["hist_s"] = []
            st.session_state["hist_sig"] = []

    s_now, sig_now = acc.chsh()
    m1, m2, m3 = st.columns(3)
    m1.metric("S", f"{s_now:.3f}")
    m2.metric("统计误差 σ", f"{sig_now:.3f}")
    m3.metric("累积对数", f"{acc.total_pairs():,}")
    _verdict(s_now, sig_now)

    if st.session_state["hist_n"]:
        fig3 = figures.plot_chsh_history(
            st.session_state["hist_n"], st.session_state["hist_s"], st.session_state["hist_sig"]
        )
        _show(fig3)

    with st.expander("对照：经典局域隐变量模型能做什么"):
        s_cls = physics.lhv_chsh(angles)
        s_max, best = _lhv_max()
        cc1, cc2 = st.columns(2)
        cc1.metric("经典模型在你这组角度下的 S", f"{s_cls:.3f}")
        cc2.metric("经典模型遍历所有角度的最大 S", f"{s_max:.3f}")
        ba, bap = best["a"], best["a'"]
        bb, bbp = best["b"], best["b'"]
        st.markdown(
            "这个模型假设双方共享一个随机方向 λ，各自输出 sign(cos(θ−λ))，"
            f"它在最优角度（a={ba:.0f}°, a'={bap:.0f}°, "
            f"b={bb:.0f}°, b'={bbp:.0f}°）也只能到 {s_max:.3f}。\n\n"
            "它不是「算得不够好」，而是被一条数学定理卡死了。"
        )


# ==========================================================================
# 场景 ④
# ==========================================================================
with tab4:
    st.subheader("真实装置为什么很难违反 Bell")
    st.markdown(
        "实验室里 S 很少能接近 2.83。原因不是量子力学错了，而是有各种现实损耗："
        "探测器不是 100% 效率、有暗计数、光源偶尔一次发两对、符合窗口开得太宽。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        eta = st.slider("单光子探测效率 η", 0.05, 1.00, 0.60, 0.01)
    with c2:
        dark = st.slider("暗计数 (cps)", 0, 5000, 200, 50)
    with c3:
        mu = st.slider("每次泵浦的多对发射 μ", 0.0, 0.20, 0.02, 0.005)

    c4, c5 = st.columns(2)
    with c4:
        win_ns = st.slider("符合窗口 τ (ns)", 0.1, 10.0, 1.0, 0.1)
    with c5:
        rate = st.slider("光子对产生率 (对/s)", 1e3, 1e7, 1e6, 1e3, format="%.0f")

    res = imperfect_chsh(visibility, eta, dark, mu, win_ns, rate)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("真实符合率", f"{res['r_true']:,.0f} /s")
    m2.metric("偶然符合率", f"{res['r_acc']:,.0f} /s")
    m3.metric("信噪比", f"{res['snr']:.3f}")
    m4.metric("S", f"{res['S']:.3f}", delta=f"{res['S'] - 2.0:+.3f} vs 2")

    fig4, ax4 = plt.subplots(figsize=(8.4, 3.4))
    labels = ["理想 S=2√2", "当前设置", "经典上限"]
    vals = [2 * np.sqrt(2), res["S"], 2.0]
    colors = ["#bdc3c7", "#c0392b" if res["S"] > 2 else "#e67e22", "#2c3e50"]
    ax4.bar(labels, vals, color=colors)
    ax4.axhline(2.0, color="#2c3e50", ls="--", lw=1)
    ax4.set_ylim(0, 3.1)
    ax4.set_ylabel("CHSH 值 S")
    ax4.set_title("现实损耗对 S 的影响（现象学模型）", fontsize=11)
    figures.grid(ax4)
    fig4.tight_layout()
    _show(fig4)

    if res["S"] > 2:
        st.success("这组参数还能违反 Bell 不等式。")
    else:
        st.error(
            "这组参数已经违反不了 Bell 不等式了 —— 但量子力学没错，"
            "是统计量不够/损耗太大。这正是实验物理学家要死磕的地方。"
        )

    st.info(
        "**这个模型是示意性的**。它把损耗、暗计数、多对发射压成一个信噪比，"
        "再线性折成可见度。真实分析要考虑泊松统计、探测器死时间、"
        "以及「探测效率漏洞」「定域性漏洞」「自由选择漏洞」分别怎么排除。"
    )


# ==========================================================================
# 场景 ⑤
# ==========================================================================
with tab5:
    st.subheader("两个光子撞上分束器")
    st.markdown(
        "两个完全不可分辨的光子同时从两侧进入 50:50 分束器，它们**总是从同一侧出来**，"
        "符合计数掉到零 —— 这就是 Hong-Ou-Mandel 凹陷。"
        "把其中一个延迟一点，或者让它们的光谱不再相同，凹陷就会变浅。"
    )

    c1, c2 = st.columns(2)
    with c1:
        purity = st.slider("光谱纯度（模式匹配度）", 0.0, 1.0, 1.0, 0.01)
    with c2:
        coh = st.slider("相干时间尺度 (fs)", 50, 500, 200, 10)

    fig5, ax5 = plt.subplots(figsize=(8.6, 4.0))
    delay = np.linspace(-600, 600, 481)
    ax5.plot(delay, physics.hom_coincidence(delay, coh, purity), lw=2, color="#c0392b")
    ax5.fill_between(delay, 0.5, physics.hom_coincidence(delay, coh, purity),
                     color="#c0392b", alpha=0.12, lw=0)
    ax5.axhline(0.5, color="0.7", ls=":", lw=1)
    ax5.set_xlabel("两路光程差 Δτ (fs)")
    ax5.set_ylabel("符合计数（归一化）")
    ax5.set_ylim(-0.05, 0.6)
    ax5.set_title(f"凹陷深度 = {purity:.2f}", fontsize=11)
    figures.grid(ax5)
    fig5.tight_layout()
    _show(fig5)

    st.markdown(
        "凹陷深度直接反映两个光子有多难区分。光谱越纯、模式越匹配，凹陷越深。"
        "这也是判断一个源能不能用来做高质量干涉实验的最直接指标。"
    )
    st.info(
        "这里的曲线是**示意模型**：用一个高斯凹陷代替真实的光谱干涉。"
        "严格的算法要从联合谱振幅 JSA 出发做 Schmidt 分解，"
        "再算 HOM 可见度 —— 那是下一个层次的模拟。"
    )
