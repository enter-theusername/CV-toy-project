# gel_core.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import cv2

# ---------- 基础增强 ----------
import cv2
import numpy as np

import numpy as np
import cv2

def auto_white_balance(
    img_bgr: np.ndarray,
    clip_limit: float = 2.0,     # 为兼容旧签名保留，但在 autoscale 中不使用
    tile_size: int = 8,          # 为兼容旧签名保留，但在 autoscale 中不使用
    exposure: float = 1.0,
    percent_low: float = 0.5,
    percent_high: float = 99.5,
    per_channel: bool = False,
    gamma: float | None = None
) -> np.ndarray:
    """
    ImageLab 风格的 Autoscale（自动对比度拉伸）

    灰度图：
        - 依据 [percent_low, percent_high] 的百分位，线性拉伸至 [0, 255]
        - 再乘以 exposure（建议 0.5~2.0）
        - 可选 gamma 校正（对归一化到 [0,1] 的值应用 out = out ** (1/gamma)）

    彩色图：
        - per_channel = False（默认）：全局统一缩放（对 BGR 全体像素做统一百分位拉伸）
            * 尽量保持通道比例（不改变色彩倾向）
        - per_channel = True：逐通道缩放（各通道分别按百分位拉伸）
            * 对比度最大化，但可能改变色彩
        - 然后统一乘 exposure；可选 gamma

    说明：
        - 本实现不再做灰度平场校正、CLAHE 或 Shades-of-Gray，以贴近 ImageLab Autoscale 的线性拉伸本意。
        - clip_limit/tile_size 参数仅为兼容历史签名保留，函数中不使用。

    参数:
        exposure: 亮度乘法，建议范围 0.5 ~ 2.0
        percent_low/percent_high: 百分位阈值（建议 0.1~2 / 98~99.9 之间微调）
        per_channel: 是否按通道各自 autoscale
        gamma: 若给定（>0），对归一化后的结果执行 gamma 校正（out = out ** (1/gamma)）

    返回:
        拉伸后的 BGR uint8 图像
    """
    # --- 保护性限定 ---
    exposure = float(np.clip(exposure, 0.5, 2.0))
    percent_low = float(np.clip(percent_low, 0.0, 50.0))
    percent_high = float(np.clip(percent_high, 50.0, 100.0))
    if percent_high <= percent_low:
        percent_low, percent_high = 0.5, 99.5
    if gamma is not None:
        gamma = max(1e-6, float(gamma))

    # 输入转 float32
    if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        # -------- 灰度图分支 --------
        g = img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]
        g = g.astype(np.float32)

        lo = np.percentile(g, percent_low)
        hi = np.percentile(g, percent_high)
        if hi <= lo + 1e-6:
            # 动态范围太小，直接做曝光乘法并裁剪
            out = np.clip(g / 255.0 * exposure, 0.0, 1.0)
        else:
            out = (g - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0)
            if gamma is not None:
                out = np.power(out, 1.0 / gamma)
            out = np.clip(out * exposure, 0.0, 1.0)

        g8 = (out * 255.0 + 0.5).astype(np.uint8)
        return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

    # -------- 彩色图分支 --------
    img = img_bgr.astype(np.float32)

    if per_channel:
        # 每个通道独立 autoscale
        out = np.empty_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c]
            lo = np.percentile(ch, percent_low)
            hi = np.percentile(ch, percent_high)
            if hi <= lo + 1e-6:
                chn = np.clip(ch / 255.0, 0.0, 1.0)
            else:
                chn = (ch - lo) / (hi - lo)
                chn = np.clip(chn, 0.0, 1.0)
            out[:, :, c] = chn
    else:
        # 全局统一 autoscale（对所有通道共同求阈值）
        lo = np.percentile(img, percent_low)
        hi = np.percentile(img, percent_high)
        if hi <= lo + 1e-6:
            out = np.clip(img / 255.0, 0.0, 1.0)
        else:
            out = (img - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0)

    if gamma is not None:
        out = np.power(out, 1.0 / gamma)

    out = np.clip(out * exposure, 0.0, 1.0)
    out_u8 = (out * 255.0 + 0.5).astype(np.uint8)
    return out_u8




# ---------- 胶块检测 ----------
def detect_gel_regions(img_bgr: np.ndarray, expected: int = 2,
                       thr_block: int = 51, thr_C: int = 10,
                       morph_ksize: int = 11) -> List[Tuple[int, int, int, int]]:
    """
    返回每块胶的边界框 (x, y, w, h)，按从左到右排序。
    参数可调：自适应阈值 blockSize/C、闭运算核大小等。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if (img_bgr.ndim == 3 and img_bgr.shape[2] == 3) else img_bgr
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    block = thr_block if thr_block % 2 == 1 else thr_block + 1
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, thr_C
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    close = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.02 * W * H:
            continue
        if w < 0.15 * W or h < 0.1 * H:
            continue
        cand.append((x, y, w, h))
    cand = sorted(cand, key=lambda b: b[2] * b[3], reverse=True)[:expected + 2]
    cand = sorted(cand, key=lambda b: b[0])
    picked = []
    for b in cand:
        cx = b[0] + b[2] / 2
        if all(abs(cx - (p[0] + p[2] / 2)) > min(b[2], p[2]) * 0.5 for p in picked):
            picked.append(b)
        if len(picked) >= expected:
            break
    out = []
    for x, y, w, h in picked:
        x2 = max(0, x - 10)
        y2 = max(0, y - 10)
        x3 = min(W - 1, x + w + 10)
        y3 = min(H - 1, y + h + 10)
        out.append((x2, y2, max(1, x3 - x2), max(1, y3 - y2)))
    return out

# ---------- 泳道划分（直立/等宽） ----------
def lanes_by_projection(gray: np.ndarray, n: int, smooth_px: int = 31,
                        min_sep_ratio: float = 1.2) -> List[Tuple[int, int]]:
    """
    垂直方向强度投影 → 取峰为中心 → 中点切分。
    峰值不足自动回退等宽。返回每条泳道的 (x1, x2)。
    """
    H, W = gray.shape
    inv = 255 - gray
    prof = inv.mean(axis=0)
    k = max(3, int(smooth_px) if int(smooth_px) % 2 == 1 else int(smooth_px) + 1)
    prof_s = np.convolve(prof, np.ones(k) / k, mode='same')
    min_sep = W / (n * min_sep_ratio)
    idx = np.argsort(prof_s)[::-1]
    centers = []
    for i in idx:
        if len(centers) >= n:
            break
        if all(abs(int(i) - c) >= min_sep for c in centers):
            centers.append(int(i))
    centers = sorted(centers)
    if len(centers) < 2:
        step = W / n
        return [(int(i * step), int((i + 1) * step)) for i in range(n)]
    bounds = [0]
    for i in range(len(centers) - 1):
        bounds.append(int((centers[i] + centers[i + 1]) / 2))
    bounds.append(W)
    if len(bounds) - 1 < n:
        step = W / n
        return [(int(i * step), int((i + 1) * step)) for i in range(n)]
    return [(max(0, bounds[i]), min(W, bounds[i + 1])) for i in range(n)]

def lanes_uniform(gray: np.ndarray, n: int, left_pad: int = 0, right_pad: int = 0) -> List[Tuple[int, int]]:
    H, W = gray.shape
    L = max(1, W - left_pad - right_pad)
    step = L / n
    return [(int(left_pad + i * step), int(left_pad + (i + 1) * step)) for i in range(n)]

# ---------- 标准带检测与拟合 ----------
def detect_bands_along_y(gray_lane: np.ndarray, y0: int = 0, y1: int | None = None) -> List[int]:
    H, W = gray_lane.shape
    if y1 is None:
        y1 = H
    y0 = max(0, y0)
    y1 = min(H, y1)
    roi = gray_lane[y0:y1, :]
    inv = 255 - roi
    prof = inv.mean(axis=1)
    k = max(5, int(max(3, roi.shape[0] * 0.01)) | 1)  # odd
    prof_s = np.convolve(prof, np.ones(k) / k, mode='same')
    win = max(7, int(max(3, roi.shape[0] * 0.02)) | 1)
    half = win // 2
    peaks = []
    thr = float(prof_s.mean() + 0.5 * prof_s.std())
    for i in range(half, roi.shape[0] - half):
        if prof_s[i] >= thr and prof_s[i] == prof_s[i - half:i + half + 1].max():
            peaks.append(i + y0)
    min_sep = max(10, int(roi.shape[0] * 0.02))
    filtered = []
    for p in peaks:
        if all(abs(p - q) >= min_sep for q in filtered):
            filtered.append(p)
    return filtered

def fit_log_mw_to_y(y_positions: List[int], ladder_sizes: List[float]) -> tuple[float, float]:
    """拟合 y = a*log10(MW) + b"""
    x = np.log10(np.array(ladder_sizes, dtype=np.float64))
    y = np.array(y_positions, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def y_from_mw(mw: float, a: float, b: float) -> float:
    return a * np.log10(mw) + b

# ---------- 渲染（直立矩形） ----------
def render_annotation(gel_bgr: np.ndarray,
                      lanes: List[Tuple[int, int]],
                      ladder_peaks_y: List[int],
                      ladder_labels: List[float],
                      a: float, b: float,
                      tick_labels: List[float],
                      yaxis_side: str = 'left') -> np.ndarray:
    """
    渲染说明：
    - 左侧白色标签栏（不覆盖原图），把分子量标注画在白色栏上。
    - 标注位置来自“检测到的峰真实 y (ladder_peaks_y)”，文本来自“输入 ladder_labels”，
      两者通过“自顶向下 (y 小 -> 大) ↔ 大->小 (kDa)”一一配对（取两者长度较小部分）。
    - 文字仅显示“数字”（不带单位），并右对齐到一条“短横线”的左边；
      短横线仅在白色标签栏内，不延伸到原图。
    """
    H, W, _ = gel_bgr.shape
    label_panel_w = 80
    canvas = np.full((H, W + label_panel_w, 3), 255, dtype=np.uint8)
    canvas[:, label_panel_w:label_panel_w + W] = gel_bgr

    # 泳道（绿色）
    if lanes is not None:
        for (l, r) in lanes:
            cv2.rectangle(canvas, (label_panel_w + int(l), 0),
                          (label_panel_w + int(r), H - 1),
                          (0, 255, 0), 1)

    # 分子量标注（仅数字 + 短横线）
    if ladder_peaks_y and ladder_labels:
        ys = sorted([int(round(float(y))) for y in ladder_peaks_y])
        lbs = sorted([float(x) for x in ladder_labels], reverse=True)
        K = min(len(ys), len(lbs))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.5, 1
        color_text, color_tick = (0, 0, 0), (0, 0, 0)
        margin_right, tick_len, tick_gap = 2, 12, 3
        x2 = label_panel_w - 1 - margin_right
        x1 = max(2, x2 - tick_len + 1)

        for y, mw in zip(ys[:K], lbs[:K]):
            if 0 <= y < H:
                y_draw = int(np.clip(y, 12, H - 5))
                cv2.line(canvas, (x1, y_draw), (x2, y_draw), color_tick, 1, cv2.LINE_AA)
                text = f"{mw:g}"
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x_text = max(2, x1 - tick_gap - tw)
                cv2.putText(canvas, text, (x_text, y_draw + th // 2 - 1),
                            font, font_scale, color_text, thickness, cv2.LINE_AA)

    return canvas



# ---------- 1D 峰检测（prominence） ----------
def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k); k = max(1, k + (k % 2 == 0))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode='same')

def find_peaks_1d(signal: np.ndarray, min_distance: int = 8, min_prominence: float = 5.0) -> Tuple[List[int], List[float]]:
    """
    返回峰位置索引 & 峰的prominence（显著性）。
    signal 较大代表条带更深（建议用 255-gray 或行平均）
    """
    s = _moving_avg(signal.astype(np.float32), 7)
    peaks = []
    prom = []
    N = len(s)
    for i in range(1, N - 1):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            # 向左右找“谷”
            j = i - 1
            left_min = s[i]
            while j > 0 and s[j] <= s[j + 1]:
                left_min = min(left_min, s[j])
                j -= 1
            j = i + 1
            right_min = s[i]
            while j < N - 1 and s[j] <= s[j - 1]:
                right_min = min(right_min, s[j])
                j += 1
            p = s[i] - max(left_min, right_min)
            if p >= min_prominence:
                peaks.append(i)
                prom.append(float(p))
    # 距离去重（保留prominence更大）
    filtered = []
    for idx, p in sorted(zip(peaks, prom), key=lambda t: t[1], reverse=True):
        if all(abs(idx - j) >= min_distance for j, _ in filtered):
            filtered.append((idx, p))
    filtered.sort(key=lambda t: t[0])
    return [i for i, _ in filtered], [p for _, p in filtered]

# ---------- 基线校正 ----------
def baseline_subtract_1d(signal: np.ndarray, win: int = 51) -> np.ndarray:
    s = signal.astype(np.float32, copy=False)
    w = int(win)
    if w < 3:
        return np.clip(s - s.min(), 0, None)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float32) / float(w)
    base = np.convolve(s, kernel, mode='same')
    corr = s - base
    corr -= corr.min()
    return corr

# ---------- 二次曲线亚像素微调 ----------
def refine_peaks_quadratic(signal: np.ndarray, peaks: List[int]) -> List[float]:
    s = signal.astype(np.float32, copy=False)
    N = len(s)
    out = []
    for p in peaks:
        if 1 <= p <= N - 2:
            y0, y1, y2 = float(s[p-1]), float(s[p]), float(s[p+1])
            denom = (y0 - 2.0*y1 + y2)
            if abs(denom) < 1e-6:
                out.append(float(p))
            else:
                delta = 0.5*(y0 - y2) / denom
                out.append(float(p) + float(np.clip(delta, -1.0, 1.0)))
        else:
            out.append(float(p))
    return out

# ---------- 拟合质量评估 ----------
def eval_fit_quality(y_positions: List[float], ladder_sizes: List[float],
                     a: float, b: float, H: int,
                     r2_min: float = 0.97,
                     rmse_frac_max: float = 0.02,
                     rmse_abs_min_px: float = 5.0) -> tuple[bool, float, float]:
    x = np.log10(np.array(ladder_sizes, dtype=np.float64))
    y = np.array(y_positions, dtype=np.float64)
    y_pred = a * x + b
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2) + 1e-6)
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(resid**2)))
    rmse_limit = max(rmse_abs_min_px, rmse_frac_max * max(1, H))
    ok = (r2 >= r2_min) and (rmse <= rmse_limit) and (a < -0.1)
    return ok, r2, rmse

# ---------- 2) 标准带峰检测（垂直/通用） ----------
def detect_bands_along_y_prominence(gray_lane: np.ndarray, y0: int = 0, y1: int | None = None,
                                    min_distance: int = 10, min_prominence: float = 5.0
                                    ) -> Tuple[List[float], List[float]]:
    H, W = gray_lane.shape
    if y1 is None: y1 = H
    y0 = max(0, y0); y1 = min(H, y1)
    # 1) 生成 profile
    inv_mean = (255 - gray_lane[y0:y1, :]).mean(axis=1)
    # 2) 大窗口基线校正（窗口 ~ 5% 高度，至少 51）
    win = max(51, int(0.05 * (y1 - y0)) | 1)
    prof_corr = baseline_subtract_1d(inv_mean, win=win)
    # 3) 找峰（在校正后的 profile 上）
    peaks_int, prom = find_peaks_1d(prof_corr, min_distance=min_distance, min_prominence=min_prominence)
    # 4) 亚像素微调
    peaks_sub = refine_peaks_quadratic(prof_corr, peaks_int)
    peaks_sub = [p + y0 for p in peaks_sub]  # 回到全局 y
    return peaks_sub, prom

# ---------- 3) 基于基准带优先的匹配 + 加权鲁棒拟合 ----------
def match_ladder_best(
    peaks_y: List[int],
    ladder_labels: List[float],
    peak_weights: List[float] | None = None,
    min_pairs: int = 3
) -> Tuple[List[int], List[int]]:
    """
    从峰列表中选取“最有显著性”的数个峰用于拟合：
    - 用 peak_weights 作为显著性（prominence）评分，选择 Top-K（K = min(len(ladder_labels), len(peaks_y))）。
    - 选择后再按 y 从小到大排序，使之与“分子量从大->小”的标志物一一对应。
    - 若可用配对数 < min_pairs，返回空（交由上层放弃拟合）。

    返回:
      (选中峰的索引列表[相对原 peaks_y]，选中标志物索引列表[相对“按大->小排序后的 ladder_labels”])
    """
    # 基本校验
    if peaks_y is None or ladder_labels is None:
        return [], []
    if len(peaks_y) == 0 or len(ladder_labels) == 0:
        return [], []

    # 标志物按 大->小 排序（仅用于确定K及返回索引域）
    L_sorted = sorted(ladder_labels, reverse=True)
    M = len(L_sorted)
    N = len(peaks_y)
    K = min(M, N)

    if K < max(1, int(min_pairs)):
        return [], []

    # 显著性权重（prominence）
    # 若未提供或长度不匹配，则退化为等权
    import numpy as np
    if peak_weights is None or len(peak_weights) != N:
        Wp = np.ones(N, dtype=np.float64)
    else:
        Wp = np.array(peak_weights, dtype=np.float64)
        # 清理 NaN/inf
        if not np.all(np.isfinite(Wp)):
            finite = Wp[np.isfinite(Wp)]
            max_finite = float(finite.max()) if finite.size > 0 else 1.0
            Wp = np.nan_to_num(Wp, nan=0.0, posinf=max_finite, neginf=0.0)

    # 1) 先按显著性从大到小选 Top-K；若显著性相同则优先取 y 更小（更靠上）的峰
    #   关键: reverse=True 下，key=(权重, -y) => 权重大者优先；同权时 y 小者(-y 大)优先
    order_by_prom = sorted(
        range(N),
        key=lambda i: (float(Wp[i]), -float(peaks_y[i])),
        reverse=True
    )
    topk_idx = order_by_prom[:K]

    # 2) 为与“分子量 大->小”的标志物顺序对应，将选中的峰按 y 从小到大排序
    sel_peak_idx = sorted(topk_idx, key=lambda i: float(peaks_y[i]))
    sel_label_idx = list(range(len(sel_peak_idx)))  # 对应 L_sorted 的 0..K-1

    # 3) 最低配对数检查
    if len(sel_peak_idx) < max(1, int(min_pairs)):
        return [], []

    return sel_peak_idx, sel_label_idx

def fit_log_mw_irls(y_positions: List[int], ladder_sizes: List[float],
                    weights: List[float] | None = None, iters: int = 5) -> Tuple[float, float]:
    """
    y = a*log10(MW) + b 的 IRLS (Huber) 加权鲁棒拟合。
    """
    x = np.log10(np.array(ladder_sizes, dtype=np.float64))
    y = np.array(y_positions, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.array(weights, dtype=np.float64)
    w = w / (w.max() + 1e-6)

    a, b = 0.0, y.mean()
    for _ in range(iters):
        A = np.vstack([x, np.ones_like(x)]).T
        Aw = A * w[:, None]
        yw = y * w
        a, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        r = y - (a * x + b)
        mad = np.median(np.abs(r)) + 1e-6
        huber = 1.0 / np.maximum(1.0, np.abs(r) / (1.345 * mad))
        w = w * huber
        w = w / (w.max() + 1e-6)
    return float(a), float(b)

# ---------- 4) 斜线（线性）分道：逐行跟踪 + 线性拟合 ----------
# （下略：与原实现一致，为篇幅起见不再改动，保持接口兼容）
# ...（此处保留您现有的 lanes_slanted / detect_bands_along_y_slanted / render_annotation_slanted 实现）...


# ---------- 4) 斜线（线性）分道：逐行跟踪 + 线性拟合 ----------
# --- gel_core 3.py: 替换原 lanes_slanted ---

# gel_core 2.py  （只粘贴 lanes_slanted 的完整新版本，其他函数不动）
def lanes_slanted(
    gray: np.ndarray, n: int,
    smooth_px: int = 31, min_sep_ratio: float = 1.2,
    search_half: int = 12, max_step_px: int = 3, smooth_y: int = 9,
    # 代价图权重
    w_grad: float = 0.9, w_int: float = 0.2,
    # DP / 走廊
    corridor_frac: float = 0.60, lambda_ref: float = 1e-3, gamma_step: float = 0.5,
    # 顶/底忽略区
    top_ignore_frac: float = 0.07, bot_ignore_frac: float = 0.06,
    # 锚点
    anchor_count: int = 5, anchor_band_px: int = 11,
    # valley 惩罚
    valley_penalty_q: float = 0.15,
    # === 新增：接收白平衡后的 BGR，用于更强调“边缘”的梯度分量；以及是否做等距混合 ===
    wb_bgr: np.ndarray | None = None,
    enable_uniform_blend: bool = False,
    uniform_blend_top_y: int | None = None,
    uniform_blend_bot_y: int | None = None
) -> np.ndarray:
    """
    斜线分道（DP-Lane+ 增强版）：
    - 代价图中的“边缘项”优先使用白平衡后的图像 wb_bgr 计算（Scharr x），让边缘更显著；
    - 去掉默认“顶/底等距混合”（可通过 enable_uniform_blend 重新开启）；
    - DP 顶部行对“参考线偏移惩罚”采用渐入（ramp），避免把各分隔线的起始 x 锁死一致。
    """
    H, W = gray.shape

    # 兜底：极窄 ROI -> 等宽
    if W < max(4, n):
        bounds = np.zeros((H, n + 1), dtype=np.int32)
        bounds[:, 0] = 0
        step = W / max(1, n)
        for i in range(1, n):
            bounds[:, i] = int(round(i * step))
        bounds[:, -1] = W
        return bounds

    # ---------------- 工具函数（与原实现相同/略有补充） ----------------
    def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k < 3:
            return x.astype(np.float32, copy=False)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=np.float32) / float(k)
        x = x.astype(np.float32, copy=False)
        return np.convolve(x, kernel, mode='same')

    def _interp_with_extrap(yq: np.ndarray, yk: np.ndarray, vk: np.ndarray) -> np.ndarray:
        yq = yq.astype(np.float32, copy=False)
        yk = yk.astype(np.float32, copy=False)
        vk = vk.astype(np.float32, copy=False)
        out = np.interp(yq, yk, vk).astype(np.float32)
        if len(yk) >= 2:
            m0 = (vk[1] - vk[0]) / max(1e-6, (yk[1] - yk[0]))
            mask_top = yq < yk[0]
            out[mask_top] = vk[0] + m0 * (yq[mask_top] - yk[0])
            m1 = (vk[-1] - vk[-2]) / max(1e-6, (yk[-1] - yk[-2]))
            mask_bot = yq > yk[-1]
            out[mask_bot] = vk[-1] + m1 * (yq[mask_bot] - yk[-1])
        return out

    inv = 255 - gray

    # ---------------- 0) 预处理/代价图：增强“边缘” ----------------
    # 基于白平衡后的 BGR（若提供）计算更强的水平梯度；否则沿用原逻辑
    if wb_bgr is not None and isinstance(wb_bgr, np.ndarray) and wb_bgr.ndim == 3 and wb_bgr.shape[:2] == gray.shape:
        wb_gray = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wb_blur = cv2.GaussianBlur(wb_gray, (3, 3), 0)
        # 更敏感的 Scharr 水平梯度，强调“竖直边界”
        gx_wb = np.abs(cv2.Scharr(wb_blur, cv2.CV_32F, 1, 0))
        gx = gx_wb / (gx_wb.max() + 1e-6)
        # 强度项仍来源于去趋势后的 inv（避免穿过深色条带）
        inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)
    else:
        inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)
        sobel_x = cv2.Sobel(inv_blur, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(sobel_x)
        gx = gx / (gx.max() + 1e-6)

    k = int(smooth_px)
    if k % 2 == 0:
        k += 1
    if k < 3:
        k = 3
    inv_smooth = cv2.blur(inv_blur, (k, 1)).astype(np.float32)       # 横向平滑
    row_med = np.median(inv_smooth, axis=1, keepdims=True)           # 行级去趋势
    inv_smooth = inv_smooth - row_med
    inv_smooth -= inv_smooth.min()
    inv_smooth /= (inv_smooth.max() + 1e-6)

    # 代价：靠近强边界 => 代价小；穿过深色条带 => 代价大
    cost = (w_grad * (1.0 - gx) + w_int * inv_smooth).astype(np.float32)

    # ---------------- 1) 顶/底忽略区 ----------------
    top_ig = int(np.clip(round(H * top_ignore_frac), 0, H // 3))
    bot_ig = int(np.clip(round(H * bot_ignore_frac), 0, H // 3))
    y0_anchors = top_ig
    y1_anchors = H - bot_ig
    if y1_anchors - y0_anchors < 20:
        y0_anchors, y1_anchors = 0, H  # 极端退回全高

    # ---------------- 2) 按行找“种子分隔”并评估（与原逻辑一致，略） ----------------
    # ...（此处保持你原来的 _row_seed_separators 实现，不变）...

    def _row_seed_separators(
        y: int, min_sep_ratio_local: float,
        int_mul_k_max: int = 4, int_mul_tol: float = 0.18,
        cv_max: float = 0.25, drop_outlier_tol: float = 0.45
    ) -> tuple[bool, np.ndarray]:
        c = cost[y, :].astype(np.float32)
        s = _smooth1d(-c, 7)
        N = len(s)
        mins: list[int] = []
        for i in range(1, N - 1):
            if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
                mins.append(i)
        order = sorted(mins, key=lambda i: c[i])  # cost 越小越好
        picks: list[int] = []
        min_sep = W / (n * max(1e-3, min_sep_ratio_local))
        for idx in order:
            if all(abs(idx - p) >= min_sep for p in picks):
                picks.append(idx)
            if len(picks) >= n - 1:
                break
        if len(picks) < n - 1:
            step = W / n
            targets = [(j + 1) * step for j in range(n - 1)]
            for t in targets:
                r = int(max(2, round(0.08 * step)))  # 8% 宽度搜索半径
                l = int(max(0, round(t) - r)); rgt = int(min(W - 1, round(t) + r))
                if l >= rgt: continue
                j = l + int(np.argmin(c[l:rgt + 1]))
                if all(abs(j - p) >= 0.6 * min_sep for p in picks):
                    picks.append(j)
            picks = sorted(picks)[:(n - 1)]
        if len(picks) < n - 1:
            return False, np.array([], dtype=np.int32)

        picks = sorted(picks)
        seps = np.array([0] + picks + [W], dtype=np.int32)
        widths = np.diff(seps).astype(np.float32)
        mu = float(widths.mean()); sd = float(widths.std(ddof=1)) if len(widths) > 1 else 0.0
        if mu <= 1e-6: return False, np.array([], dtype=np.int32)
        cv = sd / mu if mu > 0 else 1.0
        bad = np.abs(widths - mu) / mu > drop_outlier_tol
        if np.any(bad): return False, np.array([], dtype=np.int32)

        if len(picks) != n - 1 or cv > cv_max:
            new_edges = [0]
            for w, a, b in zip(widths, seps[:-1], seps[1:]):
                ratio = w / mu
                k_best = 1
                for k2 in range(2, int_mul_k_max + 1):
                    if abs(ratio - k2) <= int_mul_tol:
                        k_best = k2; break
                if k_best == 1:
                    new_edges.append(b)
                else:
                    stepw = (b - a) / k_best
                    for kk in range(1, k_best):
                        new_edges.append(int(round(a + kk * stepw)))
                    new_edges.append(b)
            new_edges = np.array(sorted(set(new_edges)), dtype=np.int32)
            new_edges[0] = 0; new_edges[-1] = W
            if len(new_edges) - 1 != n:
                step = W / n
                uniform = np.array([int(round((j + 1) * step)) for j in range(n - 1)], dtype=np.int32)
                seps2 = np.array([0] + list(uniform) + [W], dtype=np.int32)
            else:
                seps2 = new_edges
            widths2 = np.diff(seps2).astype(np.float32)
            mu2 = float(widths2.mean()); sd2 = float(widths2.std(ddof=1)) if len(widths2) > 1 else 0.0
            cv2v = sd2 / mu2 if mu2 > 0 else 1.0
            if len(seps2) == n + 1 and cv2v <= max(cv_max, 0.28):
                return True, seps2[1:-1].astype(np.int32)
            return False, np.array([], dtype=np.int32)

        return True, np.array(picks, dtype=np.int32)

    # 候选行
    rng_top = max(0, y0_anchors)
    rng_bot = min(H - 1, y1_anchors - 1)
    candidate_rows = np.linspace(rng_top + 2, rng_bot - 2, num=max(9, int(0.15 * H)), dtype=np.int32)

    seed_ok = False; y_seed = int((rng_top + rng_bot) // 2); seps_seed = None
    best_cv = 1e9; best_picks = None; best_y = y_seed
    for yy in candidate_rows:
        ok, picks = _row_seed_separators(int(yy), min_sep_ratio_local=min_sep_ratio,
                                         int_mul_k_max=4, int_mul_tol=0.18,
                                         cv_max=0.25, drop_outlier_tol=0.45)
        if ok and (len(picks) == n - 1):
            widths = np.diff(np.array([0] + list(picks) + [W], dtype=np.int32)).astype(np.float32)
            mu = float(widths.mean()); sd = float(widths.std(ddof=1)) if len(widths) > 1 else 0.0
            cv = sd / mu if mu > 0 else 1.0
            if cv < best_cv:
                best_cv, best_picks, best_y = cv, picks, int(yy)
            if cv <= 0.15:
                break
    if best_picks is not None:
        seed_ok, y_seed, seps_seed = True, best_y, best_picks
    else:
        step = W / n
        seps_seed = np.array([int(round((i + 1) * step)) for i in range(n - 1)], dtype=np.int32)
        seed_ok, y_seed = True, int((rng_top + rng_bot) // 2)

    # ---------------- 3) 锚点（与原逻辑一致） ----------------
    M = max(3, int(anchor_count))
    ys_anchor = np.linspace(y0_anchors, y1_anchors - 1, M).astype(np.int32)
    band = max(5, int(anchor_band_px)); band += (band % 2 == 0)
    half = band // 2

    def _pick_centers_band(yc: int) -> list[int]:
        y_top = max(0, yc - half)
        y_bot = min(H, yc + half + 1)
        prof = inv_blur[y_top:y_bot, :].mean(axis=0).astype(np.float32)
        prof = _smooth1d(prof, k)
        pmin, pmax = float(prof.min()), float(prof.max())
        rng = max(1e-6, pmax - pmin)
        z = (prof - pmin) / rng
        if z.size >= 3:
            dz = np.abs(np.gradient(z, edge_order=1))
        elif z.size == 2:
            d = abs(float(z[1]) - float(z[0])); dz = np.array([d, d], dtype=np.float32)
        else:
            dz = np.zeros_like(z, dtype=np.float32)
        valley = 1.0 / (1.0 + dz)
        score = z - float(valley_penalty_q) * valley
        idx = np.argsort(score)[::-1]
        centers: list[int] = []
        min_sep = W / (n * max(1e-3, min_sep_ratio))
        thr = z.mean() - 0.15 * z.std()
        for i in idx:
            i = int(i)
            if score[i] < thr: break
            if all(abs(i - c) >= min_sep for c in centers):
                centers.append(i)
            if len(centers) >= n: break
        if len(centers) < n or z.size < max(2, n):
            step2 = max(1.0, W / max(1, n))
            centers = [int((j + 0.5) * step2) for j in range(n)]
        centers.sort()
        return centers

    centers_list = [_pick_centers_band(int(yy)) for yy in ys_anchor]

    if seed_ok and seps_seed is not None and len(seps_seed) == n - 1:
        seps_full = np.array([0] + list(seps_seed) + [W], dtype=np.int32)
        centers_seed = [int(round(0.5 * (seps_full[i] + seps_full[i + 1]))) for i in range(n)]
        ys_aug = np.append(ys_anchor, y_seed).astype(np.int32)
        centers_aug = centers_list + [centers_seed]
        order = np.argsort(ys_aug)
        ys_anchor = ys_aug[order]
        centers_list = [centers_aug[i] for i in order]
        M = len(ys_anchor)

    seps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    gaps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    for m in range(M):
        c = centers_list[m]
        for i in range(n - 1):
            seps_anchor[m, i] = 0.5 * (c[i] + c[i + 1])
            gaps_anchor[m, i] = c[i + 1] - c[i]

    # ---------------- 4) 参考分隔曲线 & 走廊 ----------------
    ys_all = np.arange(H, dtype=np.float32)
    yk = ys_anchor.astype(np.float32)
    xref_all: list[np.ndarray] = []
    halfw_all: list[np.ndarray] = []
    for i in range(n - 1):
        x_ref = _interp_with_extrap(ys_all, yk, seps_anchor[:, i])
        gap_i = _interp_with_extrap(ys_all, yk, gaps_anchor[:, i])
        half_w = np.maximum(2.0, 0.5 * np.abs(gap_i) * float(corridor_frac))
        xref_all.append(x_ref.astype(np.float32))
        halfw_all.append(half_w.astype(np.float32))

    # ---------------- 4.5) 顶/底“等距混合” -> 默认关闭 ----------------
    if enable_uniform_blend:
        y_blend_top = uniform_blend_top_y
        y_blend_bot = uniform_blend_bot_y
        if y_blend_top is None:
            y_start_for_blend = max(1, int(top_ig))
            y_blend_top = int(max(1, max(y_start_for_blend, round(0.08 * H))))
        if y_blend_bot is None:
            y_end_for_blend = H - max(1, int(bot_ig))
            y_blend_bot = int(min(H - 1, min(y_end_for_blend, round(0.92 * H))))
        step_uni = W / float(max(1, n))
        if y_blend_top > 1:
            alpha_top = np.linspace(1.0, 0.0, y_blend_top, dtype=np.float32)
            for i in range(n - 1):
                x_uni = (i + 1) * step_uni
                xref_all[i][:y_blend_top] = alpha_top * x_uni + (1.0 - alpha_top) * xref_all[i][:y_blend_top]
        if y_blend_bot < H - 1:
            L = H - y_blend_bot
            alpha_bot = np.linspace(1.0, 0.0, L, dtype=np.float32)
            y_slice = np.arange(y_blend_bot, H, dtype=np.int32)
            for i in range(n - 1):
                x_uni = (i + 1) * step_uni
                seg = xref_all[i][y_slice]
                xref_all[i][y_slice] = alpha_bot * x_uni + (1.0 - alpha_bot) * seg

    # ---------------- 5) DP：逐条分隔线（左->右），不交叉 ----------------
    bounds = np.zeros((H, n + 1), dtype=np.int32)
    bounds[:, 0] = 0; bounds[:, -1] = W

    y_start = max(1, int(top_ig))
    y_end = H - max(1, int(bot_ig))

    def _dp_one_separator(i_sep: int, left_guard: np.ndarray | None) -> np.ndarray:
        x_ref = xref_all[i_sep]
        half_w = halfw_all[i_sep]

        xL = np.maximum(0, np.floor(x_ref - half_w)).astype(np.int32)
        xR = np.minimum(W - 1, np.ceil(x_ref + half_w)).astype(np.int32)

        # 左护栏：避免与左侧分隔交叉
        if left_guard is not None:
            xL = np.maximum(xL, left_guard + 1)

        # 右上界：不超过下一条参考线 - 1
        if i_sep < n - 2:
            xR = np.minimum(xR, np.floor(xref_all[i_sep + 1] - 1).astype(np.int32))

        # 顶/底忽略段：动态收窄走廊
        if y_start > 1:
            d_top = np.maximum(1, np.minimum(4, np.rint(0.3 * half_w[:y_start]).astype(np.int32)))
            xL[:y_start] = np.maximum(0, np.floor(x_ref[:y_start] - d_top)).astype(np.int32)
            xR[:y_start] = np.minimum(W - 1, np.ceil(x_ref[:y_start] + d_top)).astype(np.int32)
        if y_end < H - 1:
            d_bot = np.maximum(1, np.minimum(4, np.rint(0.3 * half_w[y_end:]).astype(np.int32)))
            xL[y_end:] = np.maximum(0, np.floor(x_ref[y_end:] - d_bot)).astype(np.int32)
            xR[y_end:] = np.minimum(W - 1, np.ceil(x_ref[y_end:] + d_bot)).astype(np.int32)

        bad = (xR - xL < 2)
        if np.any(bad):
            fix = np.where(bad)[0]
            for y in fix:
                xc = int(round(x_ref[y]))
                xl = max(0, min(W - 3, xc - 1))
                xL[y], xR[y] = xl, xl + 2

        widths = (xR - xL + 1).astype(np.int32)
        maxw = int(widths.max()); INF = 1e9

        C = np.full((H, maxw), INF, dtype=np.float32)
        for y in range(H):
            w = widths[y]
            C[y, :w] = cost[y, xL[y]:xR[y] + 1]

        E = np.full_like(C, INF)
        P = np.zeros(C.shape, dtype=np.int16)

        # 顶部初始化 —— 关键改动：参考线惩罚“渐入”，避免把起始 x 锁死一致
        y0_dp = max(1, y_start)
        ramp_len = max(1, min(y0_dp, int(0.03 * H)))  # ~3% 高度的线性渐入
        for y in range(0, y0_dp):
            w = widths[y]
            xs = xL[y] + np.arange(w, dtype=np.int32)
            # 线性从 0 -> lambda_ref 的权重
            w_ref = (0.0 if ramp_len <= 1 else (float(y) / float(ramp_len - 1))) * float(lambda_ref)
            ref_pen = w_ref * (xs.astype(np.float32) - x_ref[y]) ** 2
            E[y, :w] = C[y, :w] + ref_pen

        s = int(max(1, max_step_px))
        for y in range(1, H):
            w = widths[y]
            w_prev = widths[y - 1]
            xs_now = (xL[y] + np.arange(w)).astype(np.int32)
            # 常规行开始应用完整 lambda_ref
            ref_pen = float(lambda_ref) * (xs_now.astype(np.float32) - x_ref[y]) ** 2
            for j in range(w):
                xj = xs_now[j]
                k0 = max(0, (xj - s) - xL[y - 1])
                k1 = min(w_prev - 1, (xj + s) - xL[y - 1])
                if k0 <= k1:
                    xs_prev = (xL[y - 1] + np.arange(k0, k1 + 1)).astype(np.int32)
                    step_pen = float(gamma_step) * np.abs(xs_prev - xj).astype(np.float32)
                    cand = E[y - 1, k0:k1 + 1] + step_pen
                    kk = int(np.argmin(cand))
                    E[y, j] = C[y, j] + ref_pen[j] + cand[kk]
                    P[y, j] = k0 + kk
                else:
                    E[y, j] = C[y, j] + ref_pen[j] + INF / 4

        seam = np.zeros(H, dtype=np.int32)
        j = int(np.argmin(E[H - 1, :widths[H - 1]]))
        seam[H - 1] = xL[H - 1] + j
        for y in range(H - 1, 0, -1):
            j = int(P[y, j])
            seam[y - 1] = xL[y - 1] + j

        if smooth_y is not None and smooth_y >= 3:
            sy = int(smooth_y)
            if sy % 2 == 0:
                sy += 1
            kernel = np.ones(sy, dtype=np.float32) / sy
            sm = np.convolve(seam.astype(np.float32), kernel, mode='same')
            seam = np.clip(np.rint(sm), 0, W - 1).astype(np.int32)

        return seam

    left_guard = None
    seps = []
    for i in range(n - 1):
        seam = _dp_one_separator(i, left_guard)
        seps.append(seam)
        left_guard = seam
    for i in range(1, n):
        bounds[:, i] = seps[i - 1]

    # ---------------- 6) 健壮性收尾 & 行内约束 ----------------
    if n >= 2:
        mid_x = np.median(bounds[:, 1:n], axis=0)
        if float(np.max(mid_x) - np.min(mid_x)) <= 1.0:
            step = W / float(n)
            for i in range(1, n):
                x_line = int(round(i * step))
                bounds[:, i] = np.clip(x_line, 0, W - 1)

    margin = 2
    for y in range(H):
        row = bounds[y, :].astype(np.int32, copy=True)
        for i in range(1, n):             # 左->右，至少 margin
            row[i] = max(row[i], row[i - 1] + margin)
        row[-1] = W
        for i in range(n - 1, 0, -1):     # 右->左，至少 margin
            row[i] = min(row[i], row[i + 1] - margin)
        bounds[y, :] = np.clip(row, 0, W)
    return bounds



# ---------- 5) 使用斜线边界的标准带检测 ----------
def detect_bands_along_y_slanted(gray: np.ndarray, bounds: np.ndarray, lane_index: int,
                                 y0: int = 0, y1: int | None = None,
                                 min_distance: int = 10, min_prominence: float = 5.0
                                 ) -> Tuple[List[float], List[float]]:
    H, W = gray.shape
    if y1 is None: y1 = H
    y0 = max(0, y0); y1 = min(H, y1)

    L = bounds[y0:y1, lane_index]
    R = bounds[y0:y1, lane_index + 1]
    inv = 255 - gray

    # 1) 行均值 profile
    prof = np.array([(inv[y0 + i, L[i]:max(L[i] + 1, R[i])].mean())
                     for i in range(y1 - y0)], dtype=np.float32)

    # 2) 基线校正（窗口 ~ 5% 高度，至少 51）
    win = max(51, int(0.05 * (y1 - y0)) | 1)
    prof_corr = baseline_subtract_1d(prof, win=win)

    # 3) 找峰
    peaks_int, prom = find_peaks_1d(prof_corr, min_distance=min_distance,
                                    min_prominence=min_prominence)

    # 4) 亚像素微调
    peaks_sub = refine_peaks_quadratic(prof_corr, peaks_int)
    peaks_sub = [p + y0 for p in peaks_sub]

    return peaks_sub, prom



# ---------- 6) 斜线渲染（绿：边界折线；蓝：标准带；红：刻度） ----------
def render_annotation_slanted(gel_bgr: np.ndarray,
                              bounds: np.ndarray,
                              ladder_peaks_y: List[int],
                              ladder_labels: List[float],
                              a: float, b: float,
                              tick_labels: List[float],
                              yaxis_side: str = 'left') -> np.ndarray:
    """
    渲染说明：
    - 左侧白色标签栏（不覆盖原图），把分子量标注画在白色栏上。
    - 标注位置来自“检测到的峰真实 y (ladder_peaks_y)”，文本来自“输入 ladder_labels”，
      两者通过“自顶向下 ↔ 大->小”的顺序一一配对（取两者长度较小部分）。
    - 文字仅显示“数字”，并右对齐到一条“短横线”的左边；短横线仅在白色标签栏内。
    """
    H, W, _ = gel_bgr.shape
    label_panel_w = 80
    canvas = np.full((H, W + label_panel_w, 3), 255, dtype=np.uint8)
    canvas[:, label_panel_w:label_panel_w + W] = gel_bgr

    # 绿色边界折线
    if bounds is not None and bounds.ndim == 2 and bounds.shape[0] == H:
        for i in range(1, bounds.shape[1] - 1):
            pts = np.stack([bounds[:, i] + label_panel_w, np.arange(H)], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

    # 分子量标注（仅数字 + 短横线）
    if ladder_peaks_y and ladder_labels:
        ys = sorted([int(round(float(y))) for y in ladder_peaks_y])
        lbs = sorted([float(x) for x in ladder_labels], reverse=True)
        K = min(len(ys), len(lbs))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.5, 1
        color_text, color_tick = (0, 0, 0), (0, 0, 0)
        margin_right, tick_len, tick_gap = 2, 12, 3
        x2 = label_panel_w - 1 - margin_right
        x1 = max(2, x2 - tick_len + 1)

        for y, mw in zip(ys[:K], lbs[:K]):
            if 0 <= y < H:
                y_draw = int(np.clip(y, 12, H - 5))
                cv2.line(canvas, (x1, y_draw), (x2, y_draw), color_tick, 1, cv2.LINE_AA)
                text = f"{mw:g}"
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x_text = max(2, x1 - tick_gap - tw)
                cv2.putText(canvas, text, (x_text, y_draw + th // 2 - 1),
                            font, font_scale, color_text, thickness, cv2.LINE_AA)

    return canvas
