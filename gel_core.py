# gel_core.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
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
    #if lanes is not None:
    #    for (l, r) in lanes:
    #        cv2.rectangle(canvas, (label_panel_w + int(l), 0),
    #                      (label_panel_w + int(r), H - 1),
    #                      (0, 255, 0), 1)

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
# === [新增] 自适应 prominence 阈值（MAD） ===
def robust_prominence_threshold(signal: np.ndarray, k: float = 3.0, floor: float = 5.0) -> float:
    """
    基于MAD的鲁棒阈值：sigma_hat = 1.4826 * MAD(signal - median)
    返回 max(floor, k * sigma_hat)
    """
    s = signal.astype(np.float32, copy=False)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) + 1e-6
    sigma = 1.4826 * mad
    return float(max(floor, k * sigma))

# ====== 滑动窗口法：从 1D profile 检出条带区域 ======
def _bands_from_profile_sliding(
    prof: np.ndarray,
    win_h: int = 13,     # 条带窗口高度（odd）
    ctx_h: int = 15,     # 上/下背景窗口高度
    gap: int = 3,        # 条带窗口与背景窗口之间的空隙
    diff_thr: float = 10.0,  # 条带相对上下背景的最小对比度阈值（单位=profile强度）
    min_height: int = 9,     # 条带区域的最小高度
    merge_gap: int = 0       # 条带中心连续断点容忍（通常不需要）
) -> tuple[list[float], list[float]]:
    """
    输入：按行（y）求平均后的 1D 强度 prof（深色=大），返回
    - centers: 每条条带的中心 y（float）
    - scores : 每条条带的对比度评分（mean_in - max(mean_top, mean_bot) 的区域均值）
    """
    s = prof.astype(np.float32, copy=False)
    N = len(s)
    if N < 5:
        return [], []

    # 规范 odd 窗口
    win_h = int(max(3, win_h))
    if win_h % 2 == 0: win_h += 1
    half = win_h // 2
    ctx_h = int(max(3, ctx_h))
    gap = int(max(0, gap))
    min_height = int(max(win_h, min_height))

    # 前缀和，O(1) 求任意窗口平均
    csum = np.zeros(N + 1, dtype=np.float32)
    csum[1:] = np.cumsum(s)

    def win_mean(a: int, b: int) -> float:
        # 闭区间 [a, b]，越界自动裁剪；若窗口为空返回 NaN
        aa = max(0, min(N - 1, a))
        bb = max(0, min(N - 1, b))
        if bb < aa: return float('nan')
        return float((csum[bb + 1] - csum[aa]) / max(1, (bb - aa + 1)))

    # 扫描每个中心 c 的判别结果
    ok = np.zeros(N, dtype=bool)
    score_center = np.zeros(N, dtype=np.float32)

    for c in range(N):
        # 条带窗口
        a_in = c - half
        b_in = c + half
        m_in = win_mean(a_in, b_in)

        # 上背景窗口：紧贴条带窗口上边再向上偏移 gap
        b_top = a_in - gap - 1
        a_top = b_top - ctx_h + 1
        m_top = win_mean(a_top, b_top)

        # 下背景窗口：紧贴条带窗口下边再向下偏移 gap
        a_bot = b_in + gap + 1
        b_bot = a_bot + ctx_h - 1
        m_bot = win_mean(a_bot, b_bot)

        if not (np.isfinite(m_in) and np.isfinite(m_top) and np.isfinite(m_bot)):
            continue

        d_top = float(m_in - m_top)
        d_bot = float(m_in - m_bot)
        d_use = min(d_top, d_bot)

        if d_top >= diff_thr and d_bot >= diff_thr:
            ok[c] = True
            score_center[c] = d_use

    # 合并连续中心点为区域
    centers: list[float] = []
    scores: list[float] = []

    i = 0
    while i < N:
        if not ok[i]:
            i += 1
            continue
        j = i
        last = i
        while j + 1 < N and (ok[j + 1] or (merge_gap > 0 and (j + 1 - last) <= merge_gap)):
            if ok[j + 1]:
                last = j + 1
            j += 1
        seg_a = max(0, i - half)
        seg_b = min(N - 1, j + half)
        height = seg_b - seg_a + 1
        if height >= min_height:
            # 以区域内对比度加权的中心 y（近似）
            yy = np.arange(i, j + 1, dtype=np.float32)
            ww = score_center[i:j + 1].astype(np.float32) + 1e-6
            yc = float(np.sum(yy * ww) / np.sum(ww))
            centers.append(yc)
            # 区域评分：区域内 score_center 均值
            scores.append(float(np.mean(score_center[i:j + 1])))
        i = j + 1

    return centers, scores


# ====== 替换：直立/等宽泳道的标准带检测（滑动窗口版） ======
def detect_bands_along_y_prominence(
    gray_lane: np.ndarray, y0: int = 0, y1: int = None,
    min_distance: int = 10,      # ← 重新解释为“最小条带高度 px”
    min_prominence: float = 5.0  # ← 重新解释为“条带-上下背景最小对比度”
) -> Tuple[List[float], List[float]]:
    H, W = gray_lane.shape
    if y1 is None: y1 = H
    y0 = max(0, y0); y1 = min(H, y1)

    # 1D profile：条带越深值越大
    inv_mean = (255 - gray_lane[y0:y1, :]).mean(axis=1).astype(np.float32)

    # 参数（与 GUI 传入保持兼容）
    win_h   = max(7, min_distance // 2 * 2 + 1)    # 以 min_distance 推一个条带窗口，保证 odd
    ctx_h   = max(9, int(0.8 * win_h))             # 背景窗口不小于条带窗口的 0.8
    gap     = max(2, int(round(0.15 * win_h)))     # 条带与背景的间隙
    diff_thr = float(min_prominence)               # 与上下背景的最小对比度
    min_h   = max(min_distance, win_h)             # 最小条带高度

    peaks_local, scores = _bands_from_profile_sliding(
        inv_mean, win_h=win_h, ctx_h=ctx_h, gap=gap,
        diff_thr=diff_thr, min_height=min_h, merge_gap=0
    )
    # 回到全局 y
    peaks = [float(p) + y0 for p in peaks_local]
    return peaks, scores


# ====== 替换：斜线泳道（按 bounds 切列后做滑动窗口） ======
def detect_bands_along_y_slanted(
    gray: np.ndarray, bounds: np.ndarray, lane_index: int,
    y0: int = 0, y1: int = None,
    min_distance: int = 10,      # ← 重新解释为“最小条带高度 px”
    min_prominence: float = 5.0  # ← 重新解释为“条带-上下背景最小对比度”
) -> Tuple[List[float], List[float]]:
    H, W = gray.shape
    if y1 is None: y1 = H
    y0 = max(0, y0); y1 = min(H, y1)

    # 动态列边界 → 行均值 profile（深色=大）
    inv = 255 - gray
    L = bounds[y0:y1, lane_index]
    R = bounds[y0:y1, lane_index + 1]
    prof = np.array([inv[y0 + i, L[i]:max(L[i] + 1, R[i])].mean()
                     for i in range(y1 - y0)], dtype=np.float32)

    # 参数（与 GUI 传入保持兼容）
    win_h   = max(7, min_distance // 2 * 2 + 1)
    ctx_h   = max(9, int(0.8 * win_h))
    gap     = max(2, int(round(0.15 * win_h)))
    diff_thr = float(min_prominence)
    min_h   = max(min_distance, win_h)

    peaks_local, scores = _bands_from_profile_sliding(
        prof, win_h=win_h, ctx_h=ctx_h, gap=gap,
        diff_thr=diff_thr, min_height=min_h, merge_gap=0
    )
    peaks = [float(p) + y0 for p in peaks_local]
    return peaks, scores
# === [替换] 3) 基于RANSAC+顺序约束的鲁棒匹配（保留原函数名/签名） ===
def match_ladder_best(
    peaks_y: List[int],
    ladder_labels: List[float],
    peak_weights: List[float] | None = None,
    min_pairs: int = 3
) -> Tuple[List[int], List[int]]:
    """
    返回：选中“峰索引列表[相对原peaks_y]”、选中“标签索引列表[相对按大->小排序后的 ladder_labels]”
    策略：RANSAC 随机两点拟合 -> 顺序约束下容差匹配 -> 以权重求和为目标
    回退：若无有效模型则回退到‘Top-K by prominence’的旧策略
    """
    if not peaks_y or not ladder_labels:
        return [], []

    # 预处理
    y = np.array(peaks_y, dtype=np.float64)
    order_y = np.argsort(y)           # y升序（上->下）
    y = y[order_y]
    N = len(y)

    L_sorted = sorted(ladder_labels, reverse=True)  # kDa: 大->小
    x = np.log10(np.array(L_sorted, dtype=np.float64))
    M = len(x)

    # 权重
    if peak_weights is None or len(peak_weights) != len(peaks_y):
        W = np.ones(N, dtype=np.float64)
    else:
        W_full = np.array(peak_weights, dtype=np.float64)
        W = W_full[order_y]
        if not np.all(np.isfinite(W)):
            finite = W[np.isfinite(W)]
            W = np.nan_to_num(W, nan=0.0, posinf=float(finite.max() if finite.size else 1.0), neginf=0.0)

    # RANSAC参数
    trials = min(400, max(60, 20 * min(N, M)))
    tol_px = max(8.0, 0.08 * (y.max() - y.min() + 1e-6))  # 高度相关容差
    rng = np.random.default_rng(20250827)

    best_pairs: list[tuple[int, int]] = []
    best_score = -1.0
    best_ab = None

    def eval_model(a: float, b: float) -> tuple[list[tuple[int, int]], float]:
        # 预测每个标签的 y，并在顺序约束下与 y 匹配
        y_pred = a * x + b  # 升序/降序不必强制，此处按标签顺序 j=0..M-1（kDa大->小，y应从小->大）
        pairs = []
        score = 0.0
        i = 0
        for j in range(M):
            yp = y_pred[j]
            # 找到第一个 >= yp - tol 的峰
            while i < N and y[i] < yp - tol_px:
                i += 1
            if i < N and abs(y[i] - yp) <= tol_px:
                pairs.append((i, j))
                score += float(W[i])
                i += 1
        return pairs, score

    # RANSAC循环（随机两两对应拟合直线）
    if N >= 2 and M >= 2:
        for _ in range(trials):
            i1, i2 = sorted(rng.choice(N, size=2, replace=False).tolist())
            j1, j2 = sorted(rng.choice(M, size=2, replace=False).tolist())
            if x[j2] == x[j1]:
                continue
            a = (y[i2] - y[i1]) / (x[j2] - x[j1])
            b = y[i1] - a * x[j1]
            if not np.isfinite(a) or not np.isfinite(b) or a >= -0.05:  # 经验：a 应为负，且不能太接近0
                continue
            pairs, score = eval_model(a, b)
            if len(pairs) >= max(2, int(min_pairs)) and score > best_score:
                best_score, best_pairs, best_ab = score, pairs, (a, b)

    # 若RANSAC成功，做一次IRLS精炼并重筛
    if best_pairs and len(best_pairs) >= max(2, int(min_pairs)):
        sel_i = [p for p, _ in best_pairs]
        sel_j = [q for _, q in best_pairs]
        y_used = [float(y[i]) for i in sel_i]
        lbl_used = [float(L_sorted[j]) for j in sel_j]
        w_used = [float(W[i]) for i in sel_i]
        a_fit, b_fit = fit_log_mw_irls(y_used, lbl_used, w_used, iters=6)

        # 用精炼模型重新筛选对
        pairs2, score2 = eval_model(a_fit, b_fit)
        if len(pairs2) >= len(best_pairs):
            best_pairs = pairs2
        # 将索引还原到原 peaks_y 顺序空间
        sel_i_fin = [order_y[int(i)] for i, _ in best_pairs]
        sel_j_fin = [int(j) for _, j in best_pairs]
        return sel_i_fin, sel_j_fin

    # --- 回退：旧策略 Top-K by prominence ---
    # 注意：这里复用你原本的逻辑，保证行为兼容
    K = min(N, M)
    order_by_prom = sorted(range(N), key=lambda i: (float(W[i]), -float(y[i])), reverse=True)
    topk_idx = order_by_prom[:K]
    sel_peak_idx = sorted(topk_idx, key=lambda i: float(y[i]))
    sel_label_idx = list(range(len(sel_peak_idx)))
    # 还原到原 peaks_y 的索引
    sel_peak_idx = [int(order_y[int(i)]) for i in sel_peak_idx]
    return sel_peak_idx, sel_label_idx

def fit_log_mw_irls(
    y_positions: List[int],
    ladder_sizes: List[float],
    weights: List[float] | None = None,
    iters: int = 6,
    huber_k: float = 1.345,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    鲁棒线性回归（Huber IRLS）：拟合 y = a*log10(MW) + b
    修复点：
    1) 加权最小二乘按“√w”行缩放；
    2) Huber 权重更新为 w_eff = w0 * huber，其中 huber = min(1, k*s/|r|)；
    3) 数值健壮性：x-范围过小/样本过少时做保底处理。
    """
    x = np.log10(np.array(ladder_sizes, dtype=np.float64))
    y = np.array(y_positions, dtype=np.float64)

    if x.size < 2 or y.size < 2 or x.size != y.size:
        # 保底：返回“温和负斜率”与均值截距
        a0 = -0.5
        b0 = float(y.mean()) - a0 * float(x.mean() if x.size else 0.0)
        return float(a0), float(b0)

    # 初始权重
    if weights is None:
        w0 = np.ones_like(y, dtype=np.float64)
    else:
        w0 = np.array(weights, dtype=np.float64)
        # 非法值 → 0
        w0 = np.nan_to_num(w0, nan=0.0, posinf=np.max(w0[np.isfinite(w0)]) if np.any(np.isfinite(w0)) else 1.0, neginf=0.0)
    # 归一化到 [eps, 1]
    w0 = w0 / (np.max(w0) + eps)
    w0 = np.clip(w0, eps, 1.0)

    # 初值：普通最小二乘
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        a, b = 0.0, float(y.mean())

    # IRLS 主循环
    w_eff = w0.copy()
    for _ in range(max(1, int(iters))):
        ws = np.sqrt(np.clip(w_eff, eps, 1.0))  # √w
        Aw = A * ws[:, None]
        yw = y * ws
        try:
            a, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        except Exception:
            break

        r = y - (a * x + b)
        # 鲁棒尺度（MAD）
        mad = np.median(np.abs(r - np.median(r))) + eps
        # Huber 权重：min(1, k*s/|r|)
        hub = np.minimum(1.0, (huber_k * mad) / (np.abs(r) + eps))
        # 更新有效权重（保持到 [eps,1]）
        w_eff = np.clip(w0 * hub, eps, 1.0)

    return float(a), float(b)

# ---------- 4) 斜线（线性）分道：逐行跟踪 + 线性拟合 ----------
# （下略：与原实现一致，为篇幅起见不再改动，保持接口兼容）
# ...（此处保留您现有的 lanes_slanted / detect_bands_along_y_slanted / render_annotation_slanted 实现）...


# ---------- 4) 斜线（线性）分道：逐行跟踪 + 线性拟合 ----------
# --- gel_core 3.py: 替换原 lanes_slanted ---

from typing import Optional  # ← 放到 gel_core.py 顶部的 import 区

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
    # 可选：WB 后 BGR
    wb_bgr: Optional[np.ndarray] = None,
    # 可选：顶/底等距混合（兼容保留）
    enable_uniform_blend: bool = False,
    uniform_blend_top_y: Optional[int] = None,
    uniform_blend_bot_y: Optional[int] = None,
) -> np.ndarray:
    """
    斜线分道（DP-Lane+ 加速版，兼容 Python 3.8/3.9）
    - 重点：DP 过渡按偏移 d 向量化，减少 Python 内层循环，显著提速；
    - 返回：int32 的 H × (n+1) bounds，非交叉、首列=0、末列=W。
    """
    H, W = gray.shape
    # 窄 ROI 退化为等宽
    if W < max(4, n):
        bounds = np.zeros((H, n + 1), dtype=np.int32)
        bounds[:, 0] = 0
        step = W / max(1, n)
        for i in range(1, n):
            bounds[:, i] = int(round(i * step))
        bounds[:, -1] = W
        return bounds

    # —— 工具函数 —— #
    def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k < 3: return x.astype(np.float32, copy=False)
        if k % 2 == 0: k += 1
        ker = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(x.astype(np.float32, copy=False), ker, mode='same')

    def _interp_with_extrap(yq: np.ndarray, yk: np.ndarray, vk: np.ndarray) -> np.ndarray:
        yq = yq.astype(np.float32, copy=False)
        yk = yk.astype(np.float32, copy=False)
        vk = vk.astype(np.float32, copy=False)
        out = np.interp(yq, yk, vk).astype(np.float32)
        if len(yk) >= 2:
            m0 = (vk[1] - vk[0]) / max(1e-6, (yk[1] - yk[0]))
            mask_top = yq < yk[0]; out[mask_top] = vk[0] + m0 * (yq[mask_top] - yk[0])
            m1 = (vk[-1] - vk[-2]) / max(1e-6, (yk[-1] - yk[-2]))
            mask_bot = yq > yk[-1]; out[mask_bot] = vk[-1] + m1 * (yq[mask_bot] - yk[-1])
        return out

    # —— 0) 预处理 / 代价图 —— #
    inv = 255 - gray
    if wb_bgr is not None and isinstance(wb_bgr, np.ndarray) and wb_bgr.ndim == 3 and wb_bgr.shape[:2] == gray.shape:
        wb_gray = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wb_blur = cv2.GaussianBlur(wb_gray, (3, 3), 0)
        gx_wb = np.abs(cv2.Scharr(wb_blur, cv2.CV_32F, 1, 0))
        gx = gx_wb / (gx_wb.max() + 1e-6)
        inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)
    else:
        inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)
        sobel_x = cv2.Sobel(inv_blur, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(sobel_x); gx = gx / (gx.max() + 1e-6)

    k = int(smooth_px);  k = (k + 1) if (k % 2 == 0) else max(3, k)
    inv_smooth = cv2.blur(inv_blur, (k, 1)).astype(np.float32)
    row_med = np.median(inv_smooth, axis=1, keepdims=True)
    inv_smooth = inv_smooth - row_med
    inv_smooth -= inv_smooth.min()
    inv_smooth /= (inv_smooth.max() + 1e-6)
    cost = (w_grad * (1.0 - gx) + w_int * inv_smooth).astype(np.float32)

    # 纵向平滑（anti-flicker）
    sy = 5
    if sy >= 3:
        if sy % 2 == 0: sy += 1
        cost = cv2.blur(cost, (1, sy)).astype(np.float32)

    # —— 1) 顶/底忽略区 —— #
    top_ig = int(np.clip(round(H * top_ignore_frac), 0, H // 3))
    bot_ig = int(np.clip(round(H * bot_ignore_frac), 0, H // 3))
    y0_anchors = top_ig;  y1_anchors = H - bot_ig
    if y1_anchors - y0_anchors < 20:
        y0_anchors, y1_anchors = 0, H

    # —— 2) 候选行找“种子分隔” —— #
    def _row_seed_separators(
        y: int, min_sep_ratio_local: float,
        int_mul_k_max: int = 4, int_mul_tol: float = 0.18,
        cv_max: float = 0.25, drop_outlier_tol: float = 0.45
    ) -> tuple[bool, np.ndarray]:
        c = cost[y, :].astype(np.float32)
        s1 = _smooth1d(-c, 7)
        N1 = len(s1)
        mins = [i for i in range(1, N1 - 1) if s1[i] >= s1[i - 1] and s1[i] >= s1[i + 1]]
        order = sorted(mins, key=lambda i: c[i])
        picks = []
        min_sep = W / (n * max(1e-3, min_sep_ratio_local))
        for idx in order:
            if all(abs(idx - p) >= min_sep for p in picks):
                picks.append(idx)
            if len(picks) >= n - 1: break
        if len(picks) < n - 1:
            step = W / n
            targets = [(j + 1) * step for j in range(n - 1)]
            for t in targets:
                r = int(max(2, round(0.08 * step)))
                l = int(max(0, round(t) - r)); rgt = int(min(W - 1, round(t) + r))
                if l >= rgt: continue
                j = l + int(np.argmin(c[l:rgt + 1]))
                if all(abs(j - p) >= 0.6 * min_sep for p in picks):
                    picks.append(j)
        picks = sorted(picks)[:(n - 1)]
        if len(picks) < n - 1:
            return False, np.array([], dtype=np.int32)
        seps = np.array([0] + picks + [W], dtype=np.int32)
        widths = np.diff(seps).astype(np.float32)
        mu = float(widths.mean()); sd = float(widths.std(ddof=1)) if len(widths) > 1 else 0.0
        if mu <= 1e-6: return False, np.array([], dtype=np.int32)
        cv = sd / mu if mu > 0 else 1.0
        bad = np.abs(widths - mu) / mu > drop_outlier_tol
        if np.any(bad): return False, np.array([], dtype=np.int32)
        if len(picks) != n - 1 or cv > cv_max:
            new_edges = [0]
            for w_, a, b in zip(widths, seps[:-1], seps[1:]):
                ratio = w_ / mu; k_best = 1
                for k2 in range(2, int_mul_k_max + 1):
                    if abs(ratio - k2) <= int_mul_tol: k_best = k2; break
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

    rng_top = max(0, y0_anchors); rng_bot = min(H - 1, y1_anchors - 1)
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
            cvw = sd / mu if mu > 0 else 1.0
            if cvw < best_cv: best_cv, best_picks, best_y = cvw, picks, int(yy)
            if cvw <= 0.15: break
    if best_picks is not None:
        seed_ok, y_seed, seps_seed = True, best_y, best_picks
    else:
        step = W / n
        seps_seed = np.array([int(round((i + 1) * step)) for i in range(n - 1)], dtype=np.int32)
        seed_ok, y_seed = True, int((rng_top + rng_bot) // 2)

    # —— 3) 锚点中心 —— #
    M = max(3, int(anchor_count))
    ys_anchor = np.linspace(y0_anchors, y1_anchors - 1, M).astype(np.int32)
    band = max(5, int(anchor_band_px));  band += (band % 2 == 0)
    half_band = band // 2
    def _pick_centers_band(yc: int) -> list[int]:
        y_top = max(0, yc - half_band); y_bot = min(H, yc + half_band + 1)
        prof = inv_blur[y_top:y_bot, :].mean(axis=0).astype(np.float32)
        prof = _smooth1d(prof, k)
        pmin, pmax = float(prof.min()), float(prof.max())
        rng = max(1e-6, pmax - pmin)
        z = (prof - pmin) / rng
        if z.size >= 3: dz = np.abs(np.gradient(z, edge_order=1))
        elif z.size == 2: d = abs(float(z[1]) - float(z[0])); dz = np.array([d, d], dtype=np.float32)
        else: dz = np.zeros_like(z, dtype=np.float32)
        valley = 1.0 / (1.0 + dz); score = z - float(valley_penalty_q) * valley
        idx = np.argsort(score)[::-1]
        centers = []; min_sep = W / (n * max(1e-3, min_sep_ratio))
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
        centers.sort();  return centers

    centers_list = [_pick_centers_band(int(yy)) for yy in ys_anchor]
    if seed_ok and seps_seed is not None and len(seps_seed) == n - 1:
        seps_full = np.array([0] + list(seps_seed) + [W], dtype=np.int32)
        centers_seed = [int(round(0.5 * (seps_full[i] + seps_full[i + 1]))) for i in range(n)]
        ys_aug = np.append(ys_anchor, y_seed).astype(np.int32)
        centers_aug = centers_list + [centers_seed]
        order = np.argsort(ys_aug)
        ys_anchor = ys_aug[order]; centers_list = [centers_aug[i] for i in order]; M = len(ys_anchor)

    seps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    gaps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    for m in range(M):
        c = centers_list[m]
        for i in range(n - 1):
            seps_anchor[m, i] = 0.5 * (c[i] + c[i + 1])
            gaps_anchor[m, i] = c[i + 1] - c[i]

    # —— 4) 参考分隔曲线 & 走廊 —— #
    ys_all = np.arange(H, dtype=np.float32); yk = ys_anchor.astype(np.float32)
    xref_all: list[np.ndarray] = []; halfw_all: list[np.ndarray] = []
    for i in range(n - 1):
        x_ref = _interp_with_extrap(ys_all, yk, seps_anchor[:, i])
        gap_i = _interp_with_extrap(ys_all, yk, gaps_anchor[:, i])
        half_w = np.maximum(2.0, 0.5 * np.abs(gap_i) * float(corridor_frac))
        xref_all.append(x_ref.astype(np.float32))
        halfw_all.append(half_w.astype(np.float32))

    # 4.5) 顶/底等距混合（可选）
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

    # —— 5) DP（向量化过渡），禁止交叉 —— #
    bounds = np.zeros((H, n + 1), dtype=np.int32); bounds[:, 0] = 0; bounds[:, -1] = W
    y_start = max(1, int(top_ig)); y_end = H - max(1, int(bot_ig))

    def _dp_one_separator(i_sep: int, left_guard: Optional[np.ndarray]) -> np.ndarray:
        x_ref = xref_all[i_sep]; half_w = halfw_all[i_sep]
        xL = np.maximum(0, np.floor(x_ref - half_w)).astype(np.int32)
        xR = np.minimum(W - 1, np.ceil(x_ref + half_w)).astype(np.int32)
        if left_guard is not None: xL = np.maximum(xL, left_guard + 1)
        if i_sep < n - 2: xR = np.minimum(xR, np.floor(xref_all[i_sep + 1] - 1).astype(np.int32))

        # 顶/底动态收窄
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
            for yy in fix:
                xc = int(round(x_ref[yy])); xl = max(0, min(W - 3, xc - 1))
                xL[yy], xR[yy] = xl, xl + 2

        widths = (xR - xL + 1).astype(np.int32)
        maxw = int(widths.max()); INF = np.float32(1e9)
        C = np.full((H, maxw), INF, dtype=np.float32)
        for yy in range(H):
            w = widths[yy]; C[yy, :w] = cost[yy, xL[yy]:xR[yy] + 1]
        E = np.full_like(C, INF);  P = np.full((H, maxw), -1, dtype=np.int16)

        # 顶部初始化（参考线渐入）
        y0_dp = max(1, y_start); ramp_len = max(1, min(y0_dp, int(0.03 * H)))
        for yy in range(0, y0_dp):
            w = widths[yy]
            xs = xL[yy] + np.arange(w, dtype=np.int32)
            w_ref = (0.0 if ramp_len <= 1 else (float(yy) / float(ramp_len - 1))) * float(lambda_ref)
            ref_pen = w_ref * (xs.astype(np.float32) - x_ref[yy]) ** 2
            E[yy, :w] = C[yy, :w] + ref_pen.astype(np.float32)
            P[yy, :w] = 0

        s = int(max(1, max_step_px))
        for yy in range(1, H):
            w = widths[yy]; w_prev = widths[yy - 1]
            xs_now = (xL[yy] + np.arange(w)).astype(np.int32)
            ref_pen = (float(lambda_ref) * (xs_now.astype(np.float32) - x_ref[yy]) ** 2).astype(np.float32)
            base = int(xL[yy] - xL[yy - 1])
            slope_ref = float(x_ref[yy] - x_ref[yy - 1])

            min_cand = np.full(w, INF, dtype=np.float32)
            min_src = np.full(w, -1, dtype=np.int32)
            for d in range(-s, s + 1):
                const_pen = float(gamma_step) * abs(d) + float(0.6) * abs((-d) - slope_ref)
                shift = base + d
                j0 = max(0, -shift); j1 = min(w, w_prev - shift)
                if j1 <= j0: continue
                cand = E[yy - 1, j0 + shift:j1 + shift] + np.float32(const_pen)
                cur = min_cand[j0:j1]; mask = cand < cur
                if np.any(mask):
                    min_cand[j0:j1][mask] = cand[mask]
                    min_src[j0:j1][mask] = (j0 + shift + np.nonzero(mask)[0]).astype(np.int32)

            E[yy, :w] = C[yy, :w] + ref_pen + min_cand
            P[yy, :w] = min_src[:w].astype(np.int16)

        # 回溯
        seam = np.zeros(H, dtype=np.int32)
        j = int(np.argmin(E[H - 1, :widths[H - 1]]))
        seam[H - 1] = xL[H - 1] + j
        for yy in range(H - 1, 0, -1):
            j_prev = int(P[yy, j])
            if j_prev < 0 or j_prev >= widths[yy - 1]:
                j_prev = np.clip(j, 0, widths[yy - 1] - 1)
            j = j_prev
            seam[yy - 1] = xL[yy - 1] + j

        # 可选平滑
        if smooth_y is not None and smooth_y >= 3:
            sy2 = int(smooth_y); sy2 += (sy2 % 2 == 0)
            ker = np.ones(sy2, dtype=np.float32) / sy2
            sm = np.convolve(seam.astype(np.float32), ker, mode='same')
            seam = np.clip(np.rint(sm), 0, W - 1).astype(np.int32)
        return seam

    left_guard: Optional[np.ndarray] = None
    seps = []
    for i in range(n - 1):
        seam = _dp_one_separator(i, left_guard)
        seps.append(seam);  left_guard = seam
    for i in range(1, n):
        bounds[:, i] = seps[i - 1]

    # —— 6) 健壮性收尾 & 行内约束 —— #
    if n >= 2:
        mid_x = np.median(bounds[:, 1:n], axis=0)
        if float(np.max(mid_x) - np.min(mid_x)) <= 1.0:
            step = W / float(n)
            for i in range(1, n):
                bounds[:, i] = np.clip(int(round(i * step)), 0, W - 1)
        margin = 2
        for yy in range(H):
            row = bounds[yy, :].astype(np.int32, copy=True)
            for i in range(1, n): row[i] = max(row[i], row[i - 1] + margin)
            row[-1] = W
            for i in range(n - 1, 0, -1): row[i] = min(row[i], row[i + 1] - margin)
            bounds[yy, :] = np.clip(row, 0, W)

    # 顶/底垂直锁定 + 收尾一次行内约束
    top_lock = int(max(0, min(20, H - 1)))
    if top_lock > 0:
        y_src0 = top_lock; y_src1 = min(H - 1, top_lock + 4)
        for i in range(1, n):
            x_fix = int(round(np.median(bounds[y_src0:y_src1 + 1, i])))
            bounds[:top_lock, i] = x_fix
    bot_lock = int(max(0, min(20, H - 1)))
    if bot_lock > 0:
        y_src1 = max(0, H - bot_lock - 1); y_src0 = max(0, y_src1 - 4)
        for i in range(1, n):
            x_fix = int(round(np.median(bounds[y_src0:y_src1 + 1, i])))
            bounds[H - bot_lock:, i] = x_fix
    margin = 2
    for yy in range(H):
        row = bounds[yy, :].astype(np.int32, copy=True)
        for i in range(1, n): row[i] = max(row[i], row[i - 1] + margin)
        row[-1] = W
        for i in range(n - 1, 0, -1): row[i] = min(row[i], row[i + 1] - margin)
        bounds[yy, :] = np.clip(row, 0, W)
    return bounds





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
    #if bounds is not None and bounds.ndim == 2 and bounds.shape[0] == H:
    #    for i in range(1, bounds.shape[1] - 1):
    #        pts = np.stack([bounds[:, i] + label_panel_w, np.arange(H)], axis=1).astype(np.int32)
    #        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

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
