"""BBO 非线性晶体的色散、type-I SPDC 相位匹配与出射锥。

物理链条
--------
泵浦光子 (405 nm) 在 BBO 里湮灭，产生信号光 + 闲频光：

    能量守恒：1/λ_p = 1/λ_s + 1/λ_i
    动量守恒：k_p = k_s + k_i        （相位匹配）

type-I 相位匹配下，泵浦是 e 光、信号和闲频都是 o 光。因为双折射，
可以找到一个"相位匹配角" θ_pm，使得共线简并时 n_e(θ_pm, λ_p) = n_o(λ_s)。
偏离简并时，每对波长有自己的出射角，于是屏幕上就是一圈圈光环。

本模块用 Sellmeier 方程算折射率，系数取 Eimerl et al. (1987) 的 BBO 参数：

    n² = A + B/(λ² - C) - D·λ²      （λ 单位 µm）

注意：这些是文献值，用来做课堂示意足够，但不同批次晶体、温度、掺杂都有
偏差；要拿它做定量预测，必须用你自己晶体的实测 Sellmeier 拟合。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "n_ordinary",
    "n_extraordinary",
    "n_extraordinary_at_angle",
    "phase_matching_angle",
    "degenerate_wavelength",
    "tuning_curve",
    "cone_screen",
    "wavelength_to_energy_ev",
]

# BBO（β-BaB2O4），Eimerl 1987
_BBO_O = (2.7359, 0.01878, 0.01822, 0.01354)  # A, B, C, D —— o 光
_BBO_E = (2.3753, 0.01224, 0.01667, 0.01516)  # A, B, C, D —— e 光

# 适用波长范围（µm）。超出这个区间公式本身会失效。
_LAM_MIN = 0.22
_LAM_MAX = 1.80

HC_EV_UM = 1.239841984  # hc = 1.23984 eV·µm


# --------------------------------------------------------------------------
# 色散
# --------------------------------------------------------------------------
def _sellmeier(coeffs, lam_um):
    A, B, C, D = coeffs
    lam = np.asarray(lam_um, dtype=float)
    if np.any(lam < _LAM_MIN) or np.any(lam > _LAM_MAX):
        raise ValueError(f"波长需在 {_LAM_MIN}–{_LAM_MAX} µm 之间（Sellmeier 公式的适用范围）")
    n2 = A + B / (lam ** 2 - C) - D * lam ** 2
    return np.sqrt(n2)


def n_ordinary(lam_um):
    """BBO 的 o 光折射率。λ 单位 µm。"""
    return _sellmeier(_BBO_O, lam_um)


def n_extraordinary(lam_um):
    """BBO 的 e 光主折射率（光轴方向）。"""
    return _sellmeier(_BBO_E, lam_um)


def n_extraordinary_at_angle(lam_um, theta_deg):
    """e 光在光轴夹角 θ 下的折射率（双折射椭球）。

    1/n²(θ) = cos²θ/n_o² + sin²θ/n_e²
    """
    no = float(n_ordinary(lam_um))
    ne = float(n_extraordinary(lam_um))
    t = np.deg2rad(float(theta_deg))
    return 1.0 / np.sqrt((np.cos(t) / no) ** 2 + (np.sin(t) / ne) ** 2)


# --------------------------------------------------------------------------
# 相位匹配
# --------------------------------------------------------------------------
def degenerate_wavelength(lam_p_um: float) -> float:
    """简并点：信号和闲频波长相等时，都等于泵浦波长的两倍。"""
    return 2.0 * float(lam_p_um)


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """找 f 在 [lo, hi] 上的零点。要求两端异号。"""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("区间两端同号，无法二分")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def phase_matching_angle(lam_p_um: float = 0.405, lam_s_um: float | None = None) -> float:
    """type-I 相位匹配角 θ_pm（度）。

    条件：泵浦（e 光，与光轴成 θ）的折射率等于信号光（o 光）的折射率，
    这样共线时 Δk = 0。默认取简并点 λ_s = 2λ_p。

    BBO 被 405 nm 泵浦时，典型结果大约是 28.8°。
    """
    lam_p = float(lam_p_um)
    lam_s = degenerate_wavelength(lam_p) if lam_s_um is None else float(lam_s_um)
    target = float(n_ordinary(lam_s))

    def f(theta_deg: float) -> float:
        return n_extraordinary_at_angle(lam_p, theta_deg) - target

    # θ=0 时 n_e < n_o，θ=90° 时 n_e 最小，所以 f 随 θ 单调递减
    return _bisect(f, 0.0, 90.0)


def _cone_angle(lam_s: float, lam_i: float, theta_pm_deg: float, lam_p: float) -> float:
    """给定一对波长，解出信号光的出射角 θ_s（度）。

    横向：k_s sinθ_s = k_i sinθ_i
    纵向：k_s cosθ_s + k_i cosθ_i = k_p
    """
    A = float(n_ordinary(lam_s)) / lam_s
    B = float(n_ordinary(lam_i)) / lam_i
    C = n_extraordinary_at_angle(lam_p, theta_pm_deg) / lam_p

    def f(theta: float) -> float:
        s = A * np.sin(theta) / B
        s = float(np.clip(s, 0.0, 1.0))
        return A * np.cos(theta) + B * np.sqrt(max(0.0, 1.0 - s * s)) - C

    lo, hi = 0.0, 0.5 * np.pi
    if f(lo) * f(hi) > 0:
        raise ValueError("该波长对没有物理解")
    return float(np.rad2deg(_bisect(f, lo, hi)))


def tuning_curve(
    lam_p_um: float = 0.405,
    lam_min_um: float | None = None,
    lam_max_um: float | None = None,
    n_points: int = 240,
    theta_pm_deg: float | None = None,
) -> dict:
    """固定晶体角度，扫描信号光波长，算出对应出射角。

    返回 dict，含 lam_s / lam_i / theta_s / theta_i 四个等长数组（单位 µm 和度）。
    算不出来的波长点会被跳过（物理上不出光）。
    """
    lam_p = float(lam_p_um)
    if theta_pm_deg is None:
        theta_pm_deg = phase_matching_angle(lam_p)
    lam_min = 0.55 if lam_min_um is None else float(lam_min_um)
    lam_max = 0.80 if lam_max_um is None else float(lam_max_um)
    lam_min = max(lam_min, lam_p * 1.001)

    lam_s_list = np.linspace(lam_min, lam_max, int(n_points))
    out = {"lam_s": [], "lam_i": [], "theta_s": [], "theta_i": [], "theta_pm": theta_pm_deg}

    for lam_s in lam_s_list:
        inv_i = 1.0 / lam_p - 1.0 / lam_s
        if inv_i <= 0:
            continue
        lam_i = 1.0 / inv_i
        if lam_i < _LAM_MIN or lam_i > _LAM_MAX:
            continue
        try:
            ts = _cone_angle(lam_s, lam_i, theta_pm_deg, lam_p)
        except ValueError:
            continue
        A = float(n_ordinary(lam_s)) / lam_s
        B = float(n_ordinary(lam_i)) / lam_i
        s_i = A * np.sin(np.deg2rad(ts)) / B
        if abs(s_i) > 1.0:
            continue
        ti = np.rad2deg(np.arcsin(s_i))
        out["lam_s"].append(float(lam_s))
        out["lam_i"].append(float(lam_i))
        out["theta_s"].append(ts)
        out["theta_i"].append(float(ti))

    for key in ("lam_s", "lam_i", "theta_s", "theta_i"):
        out[key] = np.asarray(out[key], dtype=float)
    return out


def cone_screen(
    lam_p_um: float = 0.405,
    n_azimuth: int = 72,
    n_points: int = 200,
    lam_min_um: float | None = None,
    lam_max_um: float | None = None,
) -> dict:
    """把出射锥画到一块垂直于泵浦的屏上。

    返回 dict，含 xs_s / ys_s / xs_i / ys_i / lam_s / lam_i，
    坐标单位是度（相对泵浦方向的张角），每对光子出现在相反方位角上。
    """
    curve = tuning_curve(
        lam_p_um=lam_p_um,
        lam_min_um=lam_min_um,
        lam_max_um=lam_max_um,
        n_points=n_points,
    )
    if curve["lam_s"].size == 0:
        empty = np.array([])
        return {"xs_s": empty, "ys_s": empty, "xs_i": empty, "ys_i": empty,
                "lam_s": empty, "lam_i": empty, "theta_pm": curve["theta_pm"]}

    az = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    ts = curve["theta_s"][:, None]
    ti = curve["theta_i"][:, None]
    xs_s = (ts * np.cos(az)).ravel()
    ys_s = (ts * np.sin(az)).ravel()
    xs_i = (-ti * np.cos(az)).ravel()
    ys_i = (-ti * np.sin(az)).ravel()
    rep = curve["lam_s"].size
    return {
        "xs_s": xs_s,
        "ys_s": ys_s,
        "xs_i": xs_i,
        "ys_i": ys_i,
        "lam_s": np.repeat(curve["lam_s"], len(az)),
        "lam_i": np.repeat(curve["lam_i"], len(az)),
        "theta_pm": curve["theta_pm"],
        "n_pairs": rep,
    }


def wavelength_to_energy_ev(lam_um):
    """光子能量（eV）。"""
    return HC_EV_UM / np.asarray(lam_um, dtype=float)
