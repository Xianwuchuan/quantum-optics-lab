# SPDC 纠缠光子：教学演示

按量子力学给出的概率采样 SPDC 光子对的测量结果：从「光子对怎么来的」讲到
「为什么经典模型解释不了」。

> 不产生纠缠光子，只模拟测量统计。屏幕上的数字都是模型推演的，不是实验测量。

## 跑起来

```bash
pip install -r requirements.txt

streamlit run app.py      # 交互版（在 spdc_demo/ 目录下）
python demo_static.py     # 静态版，出 5 张 PNG 到 spdc_demo/figs/
```

静态版的输出目录固定在脚本旁边的 `figs/`，在哪个目录下运行都一样；要换位置用
`-o`。命令行参数：`-o/--out`、`--visibility`、`--seed`、`--lam-p`（泵浦波长，nm）。

验证环境：Windows 11 / Python 3.12.8 / numpy 2.4.4 / matplotlib 3.11.0 / streamlit 1.66.1。

## 基准输出（seed=42，V=1.0）

| 量 | 值 |
|---|---|
| 相位匹配角 θ_pm（405 nm 泵浦） | 28.82° |
| CHSH，N = 400 对 | S = 2.820 ± 0.141 |
| CHSH，N = 40,000 对 | S = 2.844 ± 0.014 |
| CHSH，N = 200,000 对 | S = 2.829 ± 0.006 |
| 经典局域模型（遍历角度）最大 S | 2.000 |
| 量子上限（Tsirelson） | 2.828 |
| 单路通过率 / 两路一致率 | 50.5% / 100% |

N 是四个设置的总对数，误差按每个 E 各自对数合成 (1−E²)/N。

## 五个场景

| 场景 | 一句话 |
|---|---|
| ① 相位匹配 | 光环怎么来的 |
| ② 随机 vs 关联 | 单次不可预测，关联可预测 |
| ③ CHSH | 定量跨过经典上限 |
| ④ 现实不完美 | 损耗把 S 拉回 2 以下 |
| ⑤ HOM | 不可分辨性的直接体现 |

### ① 相位匹配

```
1/λ_p = 1/λ_s + 1/λ_i       能量守恒
k_p   = k_s   + k_i         动量守恒 = 相位匹配
```

type-I：泵浦走 e 光，信号/闲频走 o 光，BBO 双折射给出 θ_pm。405 nm 泵浦时
θ_pm ≈ 28.8°，共线简并两个光子都是 810 nm；偏离简并就成对以一定角度飞出，
这就是屏上的光环。

折射率用 Sellmeier（Eimerl 1987）算，角度是真解出来的；没算泵浦发散和晶体
长度，所以环是理想细线。

### ② 随机 vs 关联

转任意一边的偏振片，单路通过率总在 50%；两路符合则强烈依赖相对夹角。

没有「信号通过就让闲频通过」的写法——那是循环论证。这里是先算态矢量的四个
联合概率，再按概率采样。

### ③ CHSH

```
S = E(a,b) − E(a,b') + E(a',b) + E(a',b')
经典 ≤ 2        量子 ≤ 2√2 ≈ 2.828
```

默认角度 (0°, 45°, 22.5°, 67.5°) 就是最优那组。S 随累积收敛，误差由各 E 的
(1−E²)/N 合成。

对照模型共享随机方向 λ，输出 sign(cos(θ−λ))，遍历角度也只能到 2.000——不是
算得不够好，是被定理卡死的。

### ④ 现实不完美

滑块调 η、暗计数、多对发射 μ、符合窗口 τ、产率，看 S 怎么掉。

**现象学模型**：损耗压成一个信噪比再线性折成可见度，S = 2√2·V·SNR。严格分析
要管泊松统计、死时间，以及三类漏洞怎么关。

### ⑤ HOM

不可分辨的光子进 50:50 分束器，总从同一侧出来，符合计数凹到零；延时和光谱纯度
把凹陷填浅。这是判断源质量最直接的指标。

**示意模型**：高斯凹陷代替真实光谱干涉，严格做法要从 JSA 做 Schmidt 分解。

## 当库用

`physics.py` 和 `source.py` 不依赖 UI，在 `spdc_demo/` 目录下可以直接 import：

```python
from physics import TwoPhotonState, ChshAccumulator, lhv_max_chsh
from source import phase_matching_angle, tuning_curve

state = TwoPhotonState(visibility=0.97)
p = state.probabilities(0.0, 22.5)          # 四个联合概率
s, sigma = ChshAccumulator(state).chsh()
theta = phase_matching_angle(0.405)         # 28.82
```

## 文件结构

```
spdc_demo/
├── physics.py       双光子态、Born 采样、CHSH、LHV 对照、HOM
├── source.py        BBO 色散、相位匹配、调谐曲线、出射锥
├── figures.py       matplotlib 图形（两版共用）
├── app.py           Streamlit 五场景界面
├── demo_static.py   命令行出 PNG
└── requirements.txt
```

## 再往前一步

1. **JSA**：SVD 出 Schmidt 模，算光谱纯度和 heralded purity。
2. **标定**：换成本晶体的 Sellmeier 拟合。
3. **传播与探测**：色散、滤波、抖动卷积、死时间。
