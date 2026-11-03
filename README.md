# Quantum

两个独立的小项目，共用一个原则：屏幕上的数字都是模型算出来的，不是实验测量。

| 项目 | 做什么 | 入口 |
|---|---|---|
| [`spdc_demo/`](spdc_demo/) | SPDC 纠缠光子对的五场景教学演示 | `python spdc_demo/demo_static.py` 或 `streamlit run app.py` |
| [`qlnoise/`](qlnoise/) | 光学量子线路的噪声分析（线路层）+ 误差预算 | `python qlnoise/analyze.py` |

## 跑起来

```bash
pip install -r spdc_demo/requirements.txt

# 静态版：5 张 PNG 写到 spdc_demo/figs/
python spdc_demo/demo_static.py

# 交互版：五场景 Streamlit 界面
(cd spdc_demo && streamlit run app.py)

# 噪声分析：报告 + 4 张图写到 qlnoise/report/
python qlnoise/analyze.py
```

两个脚本的输出目录都固定在各自项目内，跟你在哪个目录下运行无关。
`qlnoise` 只需要 numpy + matplotlib，不依赖 streamlit。

## 当前状态（2026-11-03 实跑验证）

环境：Windows 11 / Python 3.12.8 / numpy 2.4.4 / matplotlib 3.11.0 / streamlit 1.66.1。

**spdc_demo** —— 已验证：
- 相位匹配角 θ_pm = 28.82°（405 nm 泵浦）
- CHSH 收敛：N = 200,000 对时 S = 2.829 ± 0.006；经典局域模型遍历角度最大 2.000
- 交互版：单路通过率 50.5% / 50.5%，两路一致率 100%；每组累积 5000 对点三次
  得 S = 2.840 ± 0.011
- 五个 tab 用 `streamlit.testing.v1.AppTest` 无头跑过，无异常

**qlnoise** —— 已验证：
- 自检偏差 HOM 1.67e-16、MZI 2.22e-16（解析式对得上，演化层可信）
- 报告 9 节 + 4 张图生成正常
- 误差预算的两列排序不同：保真度侧第一是模式可见度 V，计数率侧第一是透过率 η

**已知问题**：
- `qlnoise` 报告 §6 的两级网络在 45° 相位下扫 V，曲线几乎是水平线
  （V = 0.90 时保真度 0.9997），和标题「噪声会累积」不符。演示参数待重做。
- `qlnoise` §5 表的「偏离理想值」把相位偏置算进了参考相位，这一列实际只反映
  损耗和分束比失衡。
- `spdc_demo/app.py` 用的 `use_container_width` 已被 Streamlit 标记弃用
  （2025-12-31 后移除，替代为 `width="stretch"`）。

## 目录

```
Quantum/
├── spdc_demo/
│   ├── physics.py       双光子偏振态、Born 采样、CHSH、LHV 对照、HOM
│   ├── source.py        BBO 色散、相位匹配角、调谐曲线、出射锥
│   ├── figures.py       matplotlib 图形（静态版/交互版共用）
│   ├── app.py           Streamlit 五场景界面
│   ├── demo_static.py   命令行出图
│   └── README.md        每个场景讲什么、哪里是近似
└── qlnoise/
    ├── fock.py          精确 Fock 演化：BS 矩阵元、相移、损耗
    ├── circuits.py      HOM / MZI / 两级网络
    ├── metrics.py       分布重叠、TVD、后选择、符合概率增量
    ├── analysis.py      参数扫描 + 误差预算
    ├── analyze.py       报告生成器（CLI）
    └── README.md        噪声模型、报告结构、局限
```

## 建议的阅读顺序

1. `spdc_demo/physics.py` 的 `TwoPhotonState.probabilities()` —— 四个联合概率怎么算。
2. `spdc_demo/physics.py` 的 `ChshAccumulator` —— S 和误差棒怎么来。
3. `spdc_demo/physics.py` 的 `lhv_*` —— 经典模型为什么死在 2。
4. `qlnoise/fock.py` 的 `apply_bs` / `apply_loss` —— 噪声为什么是精确的幺正/非幺正演化。
5. `qlnoise/analysis.py` 的 `error_budget()` —— 预算表怎么排出来的。

关键设计：**先算概率再采样**（spdc_demo），**不把噪声揉成一个参数**（qlnoise）。
2.828 和预算的两列排序都是推出来的，不是凑出来的。

## 边界

两个项目都是模型推演，不是实验。`spdc_demo` 的随机数是伪随机采样，
`qlnoise` 的噪声是参数化模型。各自的近似和「不做什么」写在项目 README 里。
