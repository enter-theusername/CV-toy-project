# gel_core.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import cv2

# ---------- 基础增强 ----------
def auto_white_balance(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    灰度图：直接 CLAHE 增强后转 BGR
    彩色图：Gray-world + LAB/CLAHE(L通道)，返回 BGR
    """
    if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        g = img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
        g2 = clahe.apply(g)
        return cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
    img_f = img_bgr.astype(np.float32)
    mean_bgr = img_f.mean(axis=(0, 1))
    gray = float(mean_bgr.mean())
    scale = gray / (mean_bgr + 1e-6)
    wb = np.clip(img_f * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

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
def render_annotation(gel_bgr: np.ndarray, lanes: List[Tuple[int, int]],
                      ladder_peaks_y: List[int], ladder_labels: List[float],
                      a: float, b: float,
                      tick_labels: List[float], yaxis_side: str = 'left') -> np.ndarray:
    """
    说明：为了与现有 GUI 兼容，形参 ladder_labels 保留但当前未用于绘制；
    蓝色横线 = ladder_peaks_y；红色刻度来自 tick_labels（在拟合通过时绘制）。
    """
    H, W, _ = gel_bgr.shape
    canvas = gel_bgr.copy()
    # 泳道（绿色）
    for (l, r) in lanes:
        cv2.rectangle(canvas, (l, 0), (r, H - 1), (0, 255, 0), 1)
    # 标准带横线（蓝色）
    for y in ladder_peaks_y:
        cv2.line(canvas, (0, y), (W - 1, y), (255, 0, 0), 1)
    # Y 轴刻度（红色）
    axis_x = 0 if yaxis_side == 'left' else W - 1
    for mw in tick_labels:
        y = int(round(y_from_mw(mw, a, b)))
        if 0 <= y < H:
            x2 = axis_x + (20 if yaxis_side == 'left' else -20)
            cv2.line(canvas, (axis_x, y), (x2, y), (0, 0, 255), 2)
            txt_x = (axis_x + 25) if yaxis_side == 'left' else (axis_x - 120)
            cv2.putText(canvas, f"{mw} kDa", (txt_x, max(15, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
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
    需求版：从“大分子量”开始自顶向下顺序匹配；若缺少标志物，则舍去末端（小分子量端）的若干个。
    返回（选中的 peaks 索引列表（相对原 peaks_y）、选中的 ladder 索引列表（相对“按大->小排序后的”标志物））。

    说明：
    - peaks_y 是图像 y 坐标，越小越靠上（对应分子量越大）。
    - ladder_labels 会在内部被按 大->小 排序；返回的 ladder 索引基于此排序后的列表。
    - 不再做“全局最优子序列搜索”，而是简单且稳定的“从上到下”一一配对，保证满足你的业务约束。
    """
    # 保护性处理
    if peaks_y is None or ladder_labels is None:
        return [], []
    if len(peaks_y) == 0 or len(ladder_labels) == 0:
        return [], []

    # 1) 标志物按 大->小 排序（记录其在"排序后数组"中的索引 0..M-1）
    L_sorted = sorted(ladder_labels, reverse=True)
    M = len(L_sorted)

    # 2) 峰按 y 从小到大（自顶向下）排序，保留原 peaks_y 的索引
    peaks_idx_sorted = sorted(range(len(peaks_y)), key=lambda i: float(peaks_y[i]))
    N = len(peaks_idx_sorted)

    # 3) 决定可配对数量 K：
    #    - 若峰更少：舍去末端的小分子量标志物（只取前 N 个标志物）
    #    - 若标志物更少：只取前 M 个峰
    K = min(M, N)

    # 若不足最小配对数，直接返回空
    if K < max(1, int(min_pairs)):
        return [], []

    # 4) 一一对应（从上到下 vs 从大到小）：第 j 个峰 ↔ 第 j 个标志物
    sel_peak_idx = peaks_idx_sorted[:K]          # 这些是相对原 peaks_y 的索引
    sel_label_idx = list(range(K))               # 这些是相对“已排序 L_sorted”的索引 0..K-1

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
import numpy as np
import cv2

def lanes_slanted(
    gray: np.ndarray, n: int,
    smooth_px: int = 31, min_sep_ratio: float = 1.2,
    search_half: int = 12,  # 兼容保留，不使用
    max_step_px: int = 3, smooth_y: int = 9,
    # 代价图权重
    w_grad: float = 0.7, w_int: float = 0.3,
    # DP 约束
    corridor_frac: float = 0.60, lambda_ref: float = 1e-3, gamma_step: float = 0.5,
    # 顶/底忽略（避开加样孔、眩光）
    top_ignore_frac: float = 0.07, bot_ignore_frac: float = 0.06,
    # 多锚点（沿高方向 M 个切片，每片厚 anchor_band_px）
    anchor_count: int = 5, anchor_band_px: int = 11,
    # 抑制“异常宽/深谷”（如中缝）
    valley_penalty_q: float = 0.15
) -> np.ndarray:
    """
    斜线分道（DP-Lane+）：
    新增“按行选种子分隔（基于明暗）→ 整幅 DP”的稳健起始策略，
    并保留顶/底退火与动态窄走廊，避免分隔线在顶/底端偏向一侧结束。
    """
    H, W = gray.shape

    # ---------- 兜底：极端窄 ROI 直接等宽 ----------
    if W < max(4, n):
        bounds = np.zeros((H, n + 1), dtype=np.int32)
        bounds[:, 0] = 0
        step = W / max(1, n)
        for i in range(1, n):
            bounds[:, i] = int(round(i * step))
        bounds[:, -1] = W
        return bounds

    # ---------- 工具 ----------
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
        """线性插值 + 两端线性外推"""
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

    # ---------- 0) 预处理/代价图 ----------
    inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)
    gx = np.abs(cv2.Sobel(inv_blur, cv2.CV_32F, 1, 0, ksize=3))
    gx = gx / (gx.max() + 1e-6)  # 竖向边（越大越像分隔）

    k = int(smooth_px)
    if k % 2 == 0:
        k += 1
    if k < 3:
        k = 3
    inv_smooth = cv2.blur(inv_blur, (k, 1)).astype(np.float32)   # 横向平滑
    row_med = np.median(inv_smooth, axis=1, keepdims=True)       # 行级去趋势
    inv_smooth = inv_smooth - row_med
    inv_smooth -= inv_smooth.min()
    inv_smooth /= (inv_smooth.max() + 1e-6)

    # 代价：靠近强边（gx 大 → cost 小），避开条带（inv_smooth 大 → cost 大）
    cost = (w_grad * (1.0 - gx) + w_int * inv_smooth).astype(np.float32)

    # ---------- 1) 顶/底忽略区 ----------
    top_ig = int(np.clip(round(H * top_ignore_frac), 0, H // 3))
    bot_ig = int(np.clip(round(H * bot_ignore_frac), 0, H // 3))
    y0_anchors = top_ig
    y1_anchors = H - bot_ig
    if y1_anchors - y0_anchors < 20:
        y0_anchors, y1_anchors = 0, H  # 极端回退

    # ---------- 2) “按行选种子”：在一条 y 上得到 n-1 个分隔 ----------
    def _row_seed_separators(y: int,
                             min_sep_ratio_local: float,
                             int_mul_k_max: int = 4,
                             int_mul_tol: float = 0.18,
                             cv_max: float = 0.25,
                             drop_outlier_tol: float = 0.45) -> tuple[bool, np.ndarray]:
        """
        在第 y 行，按 cost 的“谷”寻找 n-1 个分隔位置（x 索引）。
        - 若得到的列宽度近似等宽（CV<=cv_max），返回 True + 分隔；
        - 若某些宽度 ≈ k * 平均宽度（k=2..int_mul_k_max，容差 int_mul_tol），则把该宽均分补点后再验收；
        - 若宽度中存在明显离群（|w - mu|/mu > drop_outlier_tol），判不合格。
        """
        c = cost[y, :].astype(np.float32)
        # 寻找局部极小值：在 -c 上找峰
        s = _smooth1d(-c, 7)
        N = len(s)
        mins: list[int] = []
        for i in range(1, N - 1):
            if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
                mins.append(i)
        # 贪心挑选 n-1 个代价最小的“谷”，并保持最小间隔
        order = sorted(mins, key=lambda i: c[i])  # cost 越小越好
        picks: list[int] = []
        min_sep = W / (n * max(1e-3, min_sep_ratio_local))
        for idx in order:
            if all(abs(idx - p) >= min_sep for p in picks):
                picks.append(idx)
            if len(picks) >= n - 1:
                break
        # 若不足，退化为接近等距的低代价点
        if len(picks) < n - 1:
            step = W / n
            targets = [(j + 1) * step for j in range(n - 1)]
            # 在每个 target 邻域找局部最小
            for t in targets:
                r = int(max(2, round(0.08 * step)))  # 8% 宽度的搜寻半径
                l = int(max(0, round(t) - r)); rgt = int(min(W - 1, round(t) + r))
                if l >= rgt: 
                    continue
                j = l + int(np.argmin(c[l:rgt + 1]))
                # 保持最小间隔
                if all(abs(j - p) >= 0.6 * min_sep for p in picks):
                    picks.append(j)
            picks = sorted(picks)[:(n - 1)]
        if len(picks) < n - 1:
            return False, np.array([], dtype=np.int32)

        picks = sorted(picks)
        # 列宽
        seps = np.array([0] + picks + [W], dtype=np.int32)
        widths = np.diff(seps).astype(np.float32)
        mu = float(widths.mean()); sd = float(widths.std(ddof=1)) if len(widths) > 1 else 0.0
        if mu <= 1e-6:
            return False, np.array([], dtype=np.int32)
        cv = sd / mu if mu > 0 else 1.0

        # 若存在明显离群宽度，判不合格（整行丢弃）
        bad = np.abs(widths - mu) / mu > drop_outlier_tol
        if np.any(bad):
            return False, np.array([], dtype=np.int32)

        # 整数倍均分补点（把过宽区间拆分）
        # 只在 pick 数不足或 CV 超阈值时尝试
        if len(picks) != n - 1 or cv > cv_max:
            new_edges = [0]
            for w, a, b in zip(widths, seps[:-1], seps[1:]):
                ratio = w / mu
                k_best = 1
                # 找是否接近 2..K 的整数倍
                for k in range(2, int_mul_k_max + 1):
                    if abs(ratio - k) <= int_mul_tol:
                        k_best = k
                        break
                if k_best == 1:
                    new_edges.append(b)
                else:
                    stepw = (b - a) / k_best
                    for kk in range(1, k_best):
                        new_edges.append(int(round(a + kk * stepw)))
                    new_edges.append(b)
            new_edges = np.array(sorted(set(new_edges)), dtype=np.int32)
            # 避免重复/越界
            new_edges[0] = 0; new_edges[-1] = W
            # 若分隔数仍不对，按最近等距微调
            if len(new_edges) - 1 != n:
                step = W / n
                uniform = np.array([int(round((j + 1) * step)) for j in range(n - 1)], dtype=np.int32)
                seps2 = np.array([0] + list(uniform) + [W], dtype=np.int32)
            else:
                seps2 = new_edges
            widths2 = np.diff(seps2).astype(np.float32)
            mu2 = float(widths2.mean()); sd2 = float(widths2.std(ddof=1)) if len(widths2) > 1 else 0.0
            cv2 = sd2 / mu2 if mu2 > 0 else 1.0
            if len(seps2) == n + 1 and cv2 <= max(cv_max, 0.28):  # 放宽一点以通过筛选
                return True, seps2[1:-1].astype(np.int32)
            # 否则回退原 picks，继续交由外层挑其他 y
            return False, np.array([], dtype=np.int32)

        # 直接合格
        return True, np.array(picks, dtype=np.int32)

    # 在可用高度内抽样若干行，寻找“近似等分”的种子分隔
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
            # 记录最佳（以列宽 CV 为准）
            widths = np.diff(np.array([0] + list(picks) + [W], dtype=np.int32)).astype(np.float32)
            mu = float(widths.mean()); sd = float(widths.std(ddof=1)) if len(widths) > 1 else 0.0
            cv = sd / mu if mu > 0 else 1.0
            if cv < best_cv:
                best_cv, best_picks, best_y = cv, picks, int(yy)
                if cv <= 0.15:  # 已足够均匀，提前接受
                    break

    if best_picks is not None:
        seed_ok, y_seed, seps_seed = True, best_y, best_picks
    else:
        # 兜底：用几何等距作为种子
        step = W / n
        seps_seed = np.array([int(round((i + 1) * step)) for i in range(n - 1)], dtype=np.int32)
        seed_ok, y_seed = True, int((rng_top + rng_bot) // 2)

    # ---------- 3) 构建锚点 ----------
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
            if score[i] < thr:
                break
            if all(abs(i - c) >= min_sep for c in centers):
                centers.append(i)
            if len(centers) >= n:
                break
        if len(centers) < n or z.size < max(2, n):
            step = max(1.0, W / max(1, n))
            centers = [int((j + 0.5) * step) for j in range(n)]
        centers.sort()
        return centers

    centers_list = [_pick_centers_band(int(yy)) for yy in ys_anchor]

    # —— 将“种子分隔”转换为“种子中心”，并插入为额外锚点 —— #
    if seed_ok and seps_seed is not None and len(seps_seed) == n - 1:
        seps_full = np.array([0] + list(seps_seed) + [W], dtype=np.int32)
        centers_seed = [int(round(0.5 * (seps_full[i] + seps_full[i + 1]))) for i in range(n)]
        # 将 y_seed 插入 ys_anchor 的有序位置
        ys_aug = np.append(ys_anchor, y_seed).astype(np.int32)
        centers_aug = centers_list + [centers_seed]
        order = np.argsort(ys_aug)
        ys_anchor = ys_aug[order]
        centers_list = [centers_aug[i] for i in order]
        M = len(ys_anchor)

    # 由相邻中心中点得到锚点分隔（seps_anchor: M×(n-1)）
    seps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    gaps_anchor = np.zeros((M, n - 1), dtype=np.float32)
    for m in range(M):
        c = centers_list[m]
        for i in range(n - 1):
            seps_anchor[m, i] = 0.5 * (c[i] + c[i + 1])
            gaps_anchor[m, i] = c[i + 1] - c[i]

    # ---------- 4) 参考分隔曲线 & 走廊（插值 + 外推） ----------
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

    # ---------- 4.5) 顶/底：等距种子 → 数据驱动（退火混合） ----------
    # 顶端
    y_start_for_blend = max(1, int(top_ig))
    y_blend_top = int(max(1, max(y_start_for_blend, round(0.08 * H))))
    # 底端
    y_end_for_blend = H - max(1, int(bot_ig))
    y_blend_bot = int(min(H - 1, min(y_end_for_blend, round(0.92 * H))))

    if y_blend_top > 1 or y_blend_bot < H - 1:
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

    # ---------- 5) DP：逐条分隔线（左→右），不交叉 ----------
    bounds = np.zeros((H, n + 1), dtype=np.int32)
    bounds[:, 0] = 0; bounds[:, -1] = W

    y_start = max(1, int(top_ig))           # 顶端忽略带上界
    y_end   = H - max(1, int(bot_ig))       # 底端忽略带下界

    def _dp_one_separator(i_sep: int, left_guard: np.ndarray | None) -> np.ndarray:
        x_ref = xref_all[i_sep]
        half_w = halfw_all[i_sep]

        # 初始走廊：参考线 ± half_w
        xL = np.maximum(0, np.floor(x_ref - half_w)).astype(np.int32)
        xR = np.minimum(W - 1, np.ceil(x_ref + half_w)).astype(np.int32)

        # 左护栏（不交叉）
        if left_guard is not None:
            xL = np.maximum(xL, left_guard + 1)
        # 右侧上界：不超过下一条参考线 - 1
        if i_sep < n - 2:
            xR = np.minimum(xR, np.floor(xref_all[i_sep + 1] - 1).astype(np.int32))

        # 顶/底忽略带内：动态窄走廊（±d，d 随 half_w）
        if y_start > 1:
            d_top = np.maximum(1, np.minimum(4, np.rint(0.3 * half_w[:y_start]).astype(np.int32)))
            xL[:y_start] = np.maximum(0, np.floor(x_ref[:y_start] - d_top)).astype(np.int32)
            xR[:y_start] = np.minimum(W - 1, np.ceil (x_ref[:y_start] + d_top)).astype(np.int32)
        if y_end < H - 1:
            d_bot = np.maximum(1, np.minimum(4, np.rint(0.3 * half_w[y_end:]).astype(np.int32)))
            xL[y_end:] = np.maximum(0, np.floor(x_ref[y_end:] - d_bot)).astype(np.int32)
            xR[y_end:] = np.minimum(W - 1, np.ceil (x_ref[y_end:] + d_bot)).astype(np.int32)

        # 每行至少 3px 宽
        bad = (xR - xL < 2)
        if np.any(bad):
            fix = np.where(bad)[0]
            for y in fix:
                xc = int(round(x_ref[y]))
                xl = max(0, min(W - 3, xc - 1))
                xL[y], xR[y] = xl, xl + 2

        widths = (xR - xL + 1).astype(np.int32)
        maxw = int(widths.max())
        INF = 1e9

        # 局部 cost
        C = np.full((H, maxw), INF, dtype=np.float32)
        for y in range(H):
            w = widths[y]
            C[y, :w] = cost[y, xL[y]:xR[y] + 1]

        # 动规表
        E = np.full_like(C, INF)
        P = np.zeros(C.shape, dtype=np.int16)

        # 初始化（含顶端忽略段）
        y0_dp = max(1, y_start)
        for y in range(0, y0_dp):
            w = widths[y]
            xs = xL[y] + np.arange(w, dtype=np.int32)
            E[y, :w] = C[y, :w] + lambda_ref * (xs.astype(np.float32) - x_ref[y])**2

        s = int(max(1, max_step_px))
        for y in range(1, H):
            w = widths[y]
            w_prev = widths[y - 1]
            xs_now = (xL[y] + np.arange(w)).astype(np.int32)
            ref_pen = lambda_ref * (xs_now.astype(np.float32) - x_ref[y])**2
            for j in range(w):
                xj = xs_now[j]
                k0 = max(0, (xj - s) - xL[y - 1])
                k1 = min(w_prev - 1, (xj + s) - xL[y - 1])
                if k0 <= k1:
                    xs_prev = (xL[y - 1] + np.arange(k0, k1 + 1)).astype(np.int32)
                    step_pen = gamma_step * np.abs(xs_prev - xj).astype(np.float32)
                    cand = E[y - 1, k0:k1 + 1] + step_pen
                    kk = int(np.argmin(cand))
                    E[y, j] = C[y, j] + ref_pen[j] + cand[kk]
                    P[y, j] = k0 + kk
                else:
                    E[y, j] = C[y, j] + ref_pen[j] + INF / 4

        # 回溯
        seam = np.zeros(H, dtype=np.int32)
        j = int(np.argmin(E[H - 1, :widths[H - 1]]))
        seam[H - 1] = xL[H - 1] + j
        for y in range(H - 1, 0, -1):
            j = int(P[y, j])
            seam[y - 1] = xL[y - 1] + j

        # 纵向轻度平滑
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

    # ---------- 6) 健壮性收尾 ----------
    if n >= 2:
        mid_x = np.median(bounds[:, 1:n], axis=0)
        if float(np.max(mid_x) - np.min(mid_x)) <= 1.0:
            step = W / float(n)
            for i in range(1, n):
                x_line = int(round(i * step))
                bounds[:, i] = np.clip(x_line, 0, W - 1)

    # 行内严格不交叉 + 最小间距
    margin = 2
    for y in range(H):
        row = bounds[y, :].astype(np.int32, copy=True)
        for i in range(1, n):          # 左→右
            row[i] = max(row[i], row[i - 1] + margin)
        row[-1] = W
        for i in range(n - 1, 0, -1):  # 右→左
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
def render_annotation_slanted(gel_bgr: np.ndarray, bounds: np.ndarray,
                              ladder_peaks_y: List[int], ladder_labels: List[float],
                              a: float, b: float, tick_labels: List[float],
                              yaxis_side: str = 'left') -> np.ndarray:
    H, W, _ = gel_bgr.shape
    canvas = gel_bgr.copy()
    # 边界折线（直线拟合后逐行绘制，视觉上是折线/近似直线）
    for i in range(1, bounds.shape[1] - 1):
        pts = np.stack([bounds[:, i], np.arange(H)], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
    # 标准带横线（蓝色）
    for y in ladder_peaks_y:
        cv2.line(canvas, (0, y), (W - 1, y), (255, 0, 0), 1)
    # Y 轴刻度（红色）
    axis_x = 0 if yaxis_side == 'left' else W - 1
    for mw in tick_labels:
        y = int(round(a * np.log10(mw) + b))
        if 0 <= y < H:
            x2 = axis_x + (20 if yaxis_side == 'left' else -20)
            cv2.line(canvas, (axis_x, y), (x2, y), (0, 0, 255), 2)
            txt_x = (axis_x + 25) if yaxis_side == 'left' else (axis_x - 120)
            cv2.putText(canvas, f"{mw} kDa", (txt_x, max(15, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas
