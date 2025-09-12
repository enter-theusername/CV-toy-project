# app_desktop_gui_roi.py
# ==== 新增：用于加载图标 & Windows 任务栏分组ID ====
import sys, os
import ctypes

# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import json
from typing import Optional
from gel_core import (
    auto_white_balance, detect_gel_regions,
    lanes_uniform,
    render_annotation,     # 直立矩形渲染
    # 新增：斜线直线模式
    lanes_slanted,
    detect_bands_along_y_slanted,
    detect_bands_along_y_prominence,
    render_annotation_slanted,
    match_ladder_best,
    build_piecewise_log_mw_model,
    predict_y_from_mw_piecewise,
    eval_fit_quality_piecewise,
    fit_log_mw_irls,
    eval_fit_quality
)
from roi_editor import ROIEditorCanvas


# ===== 新增：右侧交互编辑画布（用于显示标注结果并可拖动红色箭头） =====
class RightAnnoCanvas(tk.Canvas):
    """
    在 Canvas 上显示最终标注底图（不含红箭头），并把红箭头作为可拖动的矢量元素叠加。
    - 自动等比缩放以适应画布，内部维护「图像坐标 <-> 画布坐标」的映射。
    - 仅允许垂直拖动；X 在拖动过程中会根据 lanes/bounds 在该 Y 的几何中心自动更新。
    - 可导出：把当前箭头叠加回像素图用于保存。
    """
    def __init__(self, parent):
        super().__init__(parent, bg="#222", highlightthickness=0)
        self.bind("<Configure>", self._on_resize)

        # 基础图
        self.base_img_bgr: np.ndarray | None = None   # 最终底图（已含Y轴/白面板等，但不含红箭头）
        self.base_img_tk = None                       # 缩放后的Tk图
        self.img_item = None                          # Canvas image item id

        # 缩放与偏移（将图像等比放入画布）
        self.s = 1.0
        self.ox = 0
        self.oy = 0

        # 与几何/映射相关的数据（生成/拖动箭头需要）
        self.gel_size = (0, 0)  # (Hg, Wg) 仅凝胶区域尺寸
        self.bounds: np.ndarray | None = None
        self.lanes: list[tuple[int, int]] | None = None
        self.a = 1.0
        self.b = 0.0
        self.fit_ok = False
        self.nlanes = 0
        self.ladder_lane = 1
        self.yaxis_side = "left"
        self.panel_top_h = 0
        self.panel_w = 0
        self.x_offset = 0
        self.arrow_style: str = "tri"
        # 箭头集合
        self.arrows: list[dict] = []  # dict: {id, lane_idx, y_img, mw, color}

        # 拖动状态
        self._drag = {"id": None, "y0_canvas": 0, "dy_img": 0}

        # ====== 新增：可调矩形框（红）+ 顶部黄色文字 ======
        # 方框集合：每一项是
        # { "rect_id": int, "text_id": int, "handles": {"nw":id,"ne":id,"se":id,"sw":id},
        #   "x_img": float, "y_img": float, "w": float, "h": float }
        self.boxes: list[dict] = []

        # 方框拖拽/缩放状态
        # mode ∈ {"move", "resize"}；corner ∈ {"nw","ne","se","sw", None}
        self._box_drag = {
            "box": None,
            "mode": None,
            "corner": None,
            "start_img": (0.0, 0.0),
            "orig": (0.0, 0.0, 0.0, 0.0),
        }

        # 用于快速计算灰度和（源于 set_scene 传入的 gel_bgr）
        self.gel_bgr: np.ndarray | None = None
        self.gel_gray: np.ndarray | None = None
        self.boxes_enabled: bool = False
        self.bind("<ButtonPress-1>",  self._on_canvas_press,   add="+")
        self.bind("<B1-Motion>",      self._on_canvas_drag,    add="+")
        self.bind("<ButtonRelease-1>", self._on_canvas_release, add="+")


    # ---------- 公有API ----------
    # === 在 RightAnnoCanvas 类中（app 24.py）新增 ===
    def update_base_image(self, base_img_bgr: np.ndarray):
        """
        仅更新底图，不改动箭头集合；用于“显示绿线”瞬时切换。
        """
        if base_img_bgr is None or not isinstance(base_img_bgr, np.ndarray) or base_img_bgr.size == 0:
            return
        self.base_img_bgr = base_img_bgr
        # 重新缩放/绘制底图，并按当前 scale/offset 复位箭头
        self._render_base_image()
        self._redraw_arrows()
        self._redraw_boxes() 
    def _mw_to_y_gel(self, mw: float) -> float:
        """
        把 MW（kDa）映射到“凝胶坐标系”的 Y，优先用分段模型；否则退回线性 y = a*log10(mw)+b。
        """
        try:
            v = float(mw)
            if not (np.isfinite(v) and v > 0):
                return float('nan')
            if self.calib_model is not None:
                xk = self.calib_model['xk']; yk = self.calib_model['yk']
                yv = predict_y_from_mw_piecewise([v], xk, yk)[0]
                return float(yv)
            # 退回线性
            return float(self.a * np.log10(v) + self.b)
        except Exception:
            return float('nan')



    def set_scene(
        self,
        base_img_bgr,            # np.ndarray
        gel_bgr,                 # np.ndarray
        bounds,                  # np.ndarray 或 None
        lanes,                   # list[(int,int)] 或 None
        a: float, b: float, fit_ok: bool,
        nlanes: int, ladder_lane: int,
        yaxis_side: str,
        lane_marks,              # list[list[float]] 或 None
        panel_top_h: int = 0,
        panel_w: int = None,     # ★ 显式传入左侧白板宽度
        calib_model: dict = None,# ★ 分段标定模型 {'xk','yk'}
        reset_positions: bool = False  # ★★★ False 时保留“已拖动箭头/方框”的相对凝胶坐标
    ):
        """
        设置底图与几何信息，并按 lane_marks 生成可拖拽红箭头。
        新增：当 reset_positions=False 时，同时“缓存并复原”已存在的红色方框位置（以凝胶坐标为准）。
        """
        # ===== ① 提前抓取“已拖动箭头”的相对凝胶坐标缓存（仅当 reset_positions=False 时） =====
        moved_cache = {}
        boxes_cache = {}  # ★ 新增：方框缓存 key=(lane_idx,mw)-> {x_gel,y_gel,w,h}
        if not reset_positions:
            def _key(li: int, mwv: float):
                return (int(li), float(mwv))
            prev_panel_top = getattr(self, "panel_top_h", 0)
            prev_x_offset  = getattr(self, "x_offset", 0)

            # 1.1 箭头缓存（与原逻辑一致）
            for a_meta in getattr(self, "arrows", []) or []:
                if not a_meta or not a_meta.get("moved", False):
                    continue
                try:
                    li = int(a_meta.get("lane_idx", -1))
                    mw = float(a_meta.get("mw", float("nan")))
                except Exception:
                    continue
                if li < 0 or not np.isfinite(mw):
                    continue
                x_img = float(a_meta.get("x_img", prev_x_offset))
                y_img = float(a_meta.get("y_img", prev_panel_top))
                x_gel = x_img - prev_x_offset
                y_gel = y_img - prev_panel_top
                moved_cache[_key(li, mw)] = {"x_gel": x_gel, "y_gel": y_gel}

            # 1.2 ★ 新增：方框缓存（以绑定箭头 key 为索引）
            if getattr(self, "boxes_enabled", False) and getattr(self, "boxes", None):
                for box in self.boxes:
                    try:
                        arrow = box.get("arrow") or {}
                        k = self._arrow_key(arrow)
                        if not k:
                            continue
                        # 优先取已存的 x_gel/y_gel；无则由 x_img/y_img 与旧偏移换算
                        x_gel = float(box.get("x_gel", box.get("x_img", 0.0) - prev_x_offset))
                        y_gel = float(box.get("y_gel", box.get("y_img", 0.0) - prev_panel_top))
                        w = float(box.get("w", 0.0))
                        h = float(box.get("h", 0.0))
                        boxes_cache[k] = {"x_gel": x_gel, "y_gel": y_gel, "w": w, "h": h}
                    except Exception:
                        continue

        # ===== ② 清空画布元素与缓存 =====
        self.delete("all")
        self.img_item = None
        self.base_img_tk = None
        self.arrows.clear()

        # 清空方框集合
        self._delete_all_boxes()
        self.boxes.clear()
        self._box_drag.update({"box": None, "mode": None, "corner": None})

        # ===== ③ 基础数据 =====
        self.base_img_bgr = base_img_bgr
        self.gel_bgr = gel_bgr.copy() if isinstance(gel_bgr, np.ndarray) else None
        self.gel_gray = cv2.cvtColor(self.gel_bgr, cv2.COLOR_BGR2GRAY) if self.gel_bgr is not None else None
        self.gel_size = gel_bgr.shape[:2] if isinstance(gel_bgr, np.ndarray) else (0, 0)  # (Hg, Wg)
        self.bounds = bounds
        self.lanes = lanes
        self.a, self.b = float(a), float(b)
        self.fit_ok = bool(fit_ok)
        self.nlanes = int(nlanes)
        self.ladder_lane = int(ladder_lane)
        self.yaxis_side = str(yaxis_side or "left").lower()
        self.panel_top_h = int(panel_top_h)
        Ht, Wt = self.base_img_bgr.shape[:2]
        Hg, Wg = self.gel_size

        # 面板宽（显式传入优先）
        if panel_w is not None:
            self.panel_w = max(0, int(panel_w))
        else:
            self.panel_w = max(0, Wt - Wg)
        self.x_offset = self.panel_w if self.yaxis_side == "left" else 0

        # ★ 分段模型：要求 xk, yk 长度>=2
        self.calib_model = None
        try:
            if calib_model and isinstance(calib_model, dict):
                xk = np.asarray(calib_model.get('xk', []), dtype=np.float64)
                yk = np.asarray(calib_model.get('yk', []), dtype=np.float64)
                if xk.size >= 2 and yk.size >= 2 and xk.size == yk.size:
                    self.calib_model = {'xk': xk, 'yk': yk}
        except Exception:
            self.calib_model = None

        # ===== ④ 渲染底图（不含红箭头/方框） =====
        self._render_base_image()
        self._redraw_arrows()
        self._redraw_boxes()

        # ===== ⑤ 生成/合并箭头 =====
        if self.fit_ok and lane_marks:
            # 若 reset_positions=True，则不传缓存，强制默认位置；否则可用 moved_cache
            self._create_arrows_from_marks(lane_marks, pos_cache=(moved_cache if not reset_positions else {}))

        # ===== ⑥ 根据“方框开关 & 箭头集合”恢复/生成方框 =====
        if self.boxes_enabled and self.arrows:
            self._delete_all_boxes()  # 防复用
            for meta in self.arrows:
                k = self._arrow_key(meta)
                if k and boxes_cache and k in boxes_cache:
                    # ★ 命中缓存：按凝胶坐标复原
                    self._create_box_for_arrow(meta, preset=boxes_cache[k])
                else:
                    # 默认：沿用原有“基于箭头尖端”的初始策略
                    self._create_box_for_arrow(meta)


    def set_arrow_style(self, style: str):
        """
        设置右侧箭头样式，并强制【重建】每个箭头的多边形，以确保：
        - 即时刷新，不需拖动；
        - 不受 Tkinter 对 polygon 顶点数变化的刷新行为影响。
        支持同义输入：'tri'/'triangle'；'star'/'★'；'diag_ne'/'↗'/'diag'.
        """
        s = (style or "").strip().lower()
        if s in ("triangle", "tri", "arrow", "→"):
            new_style = "tri"
        elif s in ("star", "★"):
            new_style = "star"
        elif s in ("diag_ne", "diag", "↗"):
            new_style = "diag_ne"
        else:
            new_style = "tri"

        # 若样式未变化，仅做一次重绘（轻量）
        changed = (getattr(self, "arrow_style", "tri") != new_style)
        self.arrow_style = new_style

        if not self.arrows:
            # 没有箭头也要刷新底图（以防未来逻辑引用），但此处快速返回
            try:
                self.update_idletasks()
            except Exception:
                pass
            return

        # 逐个【删除并重建】polygon，保证顶点数变化后立即生效
        r, g, b = (255, 64, 64)
        fill = f"#{r:02x}{g:02x}{b:02x}"
        rebuilt_ids = []
        for a in self.arrows:
            try:
                # 当前位置（若无 x_img 则按当前 y 动态求泳道左界）
                y_img = float(a.get("y_img", 0.0))
                if "x_img" in a:
                    x_img = float(a["x_img"])
                else:
                    lane_idx = int(a.get("lane_idx", 0))
                    xg = self._lane_left_x_at_y_gel(int(round(y_img - self.panel_top_h)), lane_idx)
                    x_img = float(self.x_offset + xg)

                # 先删除旧 item
                old_id = int(a["id"])
                try:
                    self.delete(old_id)
                except Exception:
                    pass

                # 以新样式计算点位并新建 polygon
                pts = self._arrow_canvas_points(x_img, y_img)
                new_id = self.create_polygon(pts, fill=fill, outline="#ff0000", width=1, smooth=False)

                # 重新绑定拖拽事件（闭包里用当前 meta）
                self.tag_bind(new_id, "<ButtonPress-1>", lambda e, m=a: self._on_drag_start(e, m))
                self.tag_bind(new_id, "<B1-Motion>",     lambda e, m=a: self._on_drag_move(e, m))
                self.tag_bind(new_id, "<ButtonRelease-1>", lambda e, m=a: self._on_drag_end(e, m))

                a["id"] = new_id
                rebuilt_ids.append(new_id)
            except Exception:
                continue

        # 提升 z 顺序，确保箭头位于底图之上
        try:
            for iid in rebuilt_ids:
                self.tag_raise(iid)
        except Exception:
            pass

        # 强制一次 UI 刷新
        try:
            self.update_idletasks()
        except Exception:
            pass


    def _shape_points_img(self, style: str, x_img: float, y_img: float) -> list[tuple[float, float]]:
        """
        返回“图像坐标系”下用于绘制的多边形顶点序列（[(x,y), ...]）。
        - 'tri'：水平向右的三角箭头
        - 'star'：五角星（中心在 x_img,y_img）
        - 'diag_ne'：↗ 箭头的【三角头部】（箭杆改由 3px 线段绘制，不再用矩形）
        """
        import math
        style = (style or "").lower()

        if style == "star":
            R, r = 5.0/self.s, 2.0/self.s
            cx, cy = float(x_img), float(y_img)
            pts = []
            for i in range(10):
                ang = math.radians(-90 + i * 36)
                rad = R if i % 2 == 0 else r
                pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
            return pts

        if style == "diag_ne":
            # ↗ 三角头（与 tri 的几何一致，但方向改为 NE）
            head_len, head_half = 15/self.s,3/self.s
            tip = (float(x_img), float(y_img))
            # 单位方向向量（NE）
            dx, dy = 1.0, -1.0
            inv = (dx*dx + dy*dy) ** -0.5
            ux, uy = dx*inv, dy*inv
            # 垂直单位向量（顺时针 90°）
            px, py = -uy, ux

            base_cx = tip[0] - ux * (head_len * 0.5)
            base_cy = tip[1] - uy * (head_len * 0.5)
            p1 = (base_cx + px * head_half, base_cy + py * head_half)
            p2 = (base_cx - px * head_half, base_cy - py * head_half)
            pmid=(base_cx,base_cy)
            p3=(base_cx-10/self.s,base_cy+10/self.s)
            return [p1,pmid,p3,pmid,p2, tip]

        # 默认水平向右三角头
        w_px, h_px =15/self.s,3/self.s
        tip = (float(x_img), float(y_img))
        p1 = (tip[0] - w_px * 0.5, tip[1] + h_px)
        p2 = (tip[0] - w_px * 0.5, tip[1] - h_px)
        return [p1, p2, tip]
        


    def render_to_image(self) -> np.ndarray | None:
        """把当前箭头叠加到底图像素，返回BGR图（用于导出）。"""
        if self.base_img_bgr is None:
            return None
        img = self.base_img_bgr.copy()
        H, W = img.shape[:2]

        for a in self.arrows:
            y = int(np.clip(a["y_img"], 0, H - 1))
            if "x_img" in a:
                x = int(np.clip(a["x_img"], 0, W - 1))
            else:
                # 默认 X = 左分界线（支持斜界线/等宽两种）
                lane_idx = a["lane_idx"]
                xl_gel = self._lane_left_x_at_y_gel(y - self.panel_top_h, lane_idx)
                x = int(np.clip(self.x_offset + xl_gel, 0, W - 1))

            # 根据当前样式生成“图像坐标”顶点
            pts_img = self._shape_points_img(self.arrow_style, float(x), float(y))
            pts = np.array(pts_img, dtype=np.int32)

            # 红色填充 + 白色描边
            cv2.fillPoly(img, [pts], (0, 0, 255))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        return img
    # ---------- 内部：几何/绘制 ----------
    def _on_resize(self, _evt=None):
        # 画布变化时重绘底图与箭头
        if self.base_img_bgr is None:
            return
        self._render_base_image()
        self._redraw_arrows()
        self._redraw_boxes()

    def _render_base_image(self):
        """按画布尺寸等比缩放底图，并绘制到Canvas。"""
        H, W = self.base_img_bgr.shape[:2]
        cw = max(1, self.winfo_width())
        ch = max(1, self.winfo_height())
        self.s = min(cw / W, ch / H)
        new_w = max(1, int(round(W * self.s)))
        new_h = max(1, int(round(H * self.s)))
        self.ox = (cw - new_w) // 2
        self.oy = (ch - new_h) // 2

        rgb = cv2.cvtColor(
            cv2.resize(self.base_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB
        )
        from PIL import Image, ImageTk
        pil = Image.fromarray(rgb)
        self.base_img_tk = ImageTk.PhotoImage(pil)  # 强引用，防GC

        try:
            if self.img_item is None:
                self.img_item = self.create_image(self.ox, self.oy, image=self.base_img_tk, anchor="nw")
            else:
                self.coords(self.img_item, self.ox, self.oy)
                self.itemconfig(self.img_item, image=self.base_img_tk)
        except tk.TclError:
            self.img_item = self.create_image(self.ox, self.oy, image=self.base_img_tk, anchor="nw")

        # ★ 保证底图在最底层，避免组合重绘瑕疵
        try:
            self.tag_lower(self.img_item)
        except Exception:
            pass

    def _to_canvas(self, x_img: float, y_img: float) -> tuple[float, float]:
        return self.ox + x_img * self.s, self.oy + y_img * self.s

    def _to_img(self, x_canvas: float, y_canvas: float) -> tuple[float, float]:
        return (x_canvas - self.ox) / self.s, (y_canvas - self.oy) / self.s

    def _lane_center_x_at_y_gel(self, y_gel: int, lane_idx: int) -> int:
        """返回在凝胶坐标系(y_gel)下，lane_idx 的中心X（单位：像素，相对凝胶左边界）。"""
        Hg, Wg = self.gel_size
        y = int(np.clip(y_gel, 0, max(0, Hg - 1)))
        if self.bounds is not None and isinstance(self.bounds, np.ndarray) and self.bounds.ndim == 2 and self.bounds.shape[0] >= Hg:
            L = int(self.bounds[y, lane_idx])
            R = int(self.bounds[y, lane_idx + 1])
            xc = int(round((L + R) / 2.0))
        elif self.lanes is not None and 0 <= lane_idx < len(self.lanes):
            l, r = self.lanes[lane_idx]
            xc = int(round((l + r) / 2.0))
        else:
            step = Wg / max(1, self.nlanes)
            xc = int(round((lane_idx + 0.5) * step))
        return int(np.clip(xc, 0, max(0, Wg - 1)))
    def _lane_left_x_at_y_gel(self, y_gel: int, lane_idx: int) -> int:
        """
        返回在“凝胶坐标系”(y_gel)下，第 lane_idx 条泳道的左侧分界线 X（相对凝胶左边界，单位：px）。
        - 当使用斜率模型(bounds)时：取 bounds[y, lane_idx]；
        - 当使用等宽/竖直模型(lanes)时：取 lanes[lane_idx][0]；
        - 当两者都无时：按等分估计左边界 lane_idx * (Wg / nlanes)。
        """
        Hg, Wg = self.gel_size
        y = int(np.clip(y_gel, 0, max(0, Hg - 1)))
        if (
            self.bounds is not None and isinstance(self.bounds, np.ndarray)
            and self.bounds.ndim == 2 and self.bounds.shape[0] >= Hg
            and 0 <= lane_idx < self.bounds.shape[1] - 1
        ):
            xl = int(self.bounds[y, lane_idx])
        elif self.lanes is not None and 0 <= lane_idx < len(self.lanes):
            xl = int(self.lanes[lane_idx][0])
        else:
            step = Wg / max(1, self.nlanes)
            xl = int(round(lane_idx * step))
        return int(np.clip(xl, 0, max(0, Wg - 1)))
    def _create_arrows_from_marks(self, lane_marks: list[list[float]], pos_cache: dict | None = None):
        """
        依据 lane_marks 在非标准道生成箭头（默认 X 在泳道左分界线）。
        —— 改造点：若 pos_cache 中命中“已拖动”的箭头（以 lane_idx+mw 为键），
        则以缓存位置还原，并且 moved=True，以便后续重绘/组合时持续保留。
        """
        # 真实泳道数
        if self.bounds is not None and isinstance(self.bounds, np.ndarray):
            real_nlanes = max(0, self.bounds.shape[1] - 1)
        elif self.lanes is not None:
            real_nlanes = len(self.lanes)
        else:
            real_nlanes = self.nlanes

        skip_idx = max(0, self.ladder_lane - 1)
        target_lane_idx = [i for i in range(real_nlanes) if i != skip_idx]
        use_k = min(len(target_lane_idx), len(lane_marks))

        Hg, Wg = self.gel_size
        H_img, W_img = self.base_img_bgr.shape[:2]

        def _key(li: int, mwv: float) -> tuple[int, float]:
            return (int(li), float(mwv))

        for k in range(use_k):
            arr = lane_marks[k] or []
            lane_idx = target_lane_idx[k]
            for mw in arr:
                try:
                    v = float(mw)
                except Exception:
                    continue
                if not (np.isfinite(v) and v > 0):
                    continue

                # 默认位置（基于拟合将 mw 映射至 y_gel；x 取该 y 的“泳道左分界线”）
                y_gel_default = int(round(self._mw_to_y_gel(v)))
                y_img_default = self.panel_top_h + y_gel_default
                xl_gel = self._lane_left_x_at_y_gel(y_gel_default, lane_idx)
                x_img_default = self.x_offset + xl_gel

                # 若 pos_cache 命中“已拖动箭头”，按缓存位置还原，并显式标记 moved=True
                x_img, y_img = x_img_default, y_img_default
                moved_flag = False
                if pos_cache:
                    entry = pos_cache.get(_key(lane_idx, v))
                    if entry:
                        x_img = self.x_offset + float(entry.get("x_gel", xl_gel))
                        y_img = self.panel_top_h + float(entry.get("y_gel", y_gel_default))
                        moved_flag = True  # ★ 关键：保留“已拖动”状态

                # 夹取到图像范围内
                x_img = float(np.clip(x_img, 0, W_img - 1))
                y_img = float(np.clip(y_img, 0, H_img - 1))

                # 创建箭头（★ 携带 moved_flag）
                self._add_arrow(x_img, y_img, lane_idx, v, moved=moved_flag)


    def _arrow_canvas_points(self, x_img: float, y_img: float) -> list[float]:
        """
        返回 Canvas 坐标系下的多边形顶点（flat list）。
        会根据 self.arrow_style 选择不同形状。
        """
        pts_img = self._shape_points_img(self.arrow_style, x_img, y_img)
        pts_canvas: list[float] = []
        for (x, y) in pts_img:
            cx, cy = self._to_canvas(x, y)
            pts_canvas.extend([cx, cy])
        return pts_canvas


    def _add_arrow(self, x_img: float, y_img: float, lane_idx: int, mw: float,
                color: tuple[int, int, int] = (255, 64, 64),
                moved: bool = False):
        """
        在Canvas上新增一个箭头并绑定拖拽事件；支持保留 moved 状态。
        """
        r, g, b = (255, 64, 64)
        pts = self._arrow_canvas_points(x_img, y_img)
        # ★ outline="" 彻底取消轮廓，避免 1px 栅格化残点
        aid=self.create_polygon(pts, fill="#FF0000", outline="#ff0000", width=1, smooth=False)
        meta = {
            "id": aid,
            "lane_idx": int(lane_idx),
            "x_img": float(x_img),
            "y_img": float(y_img),
            "mw": float(mw),
            "moved": bool(moved),
        }
        self.arrows.append(meta)
        # 绑定拖拽
        self.tag_bind(aid, "<ButtonPress-1>",  lambda e, a=meta: self._on_drag_start(e, a))
        self.tag_bind(aid, "<B1-Motion>",      lambda e, a=meta: self._on_drag_move(e, a))
        self.tag_bind(aid, "<ButtonRelease-1>",lambda e, a=meta: self._on_drag_end(e, a))
        # ★ 保证箭头在底图之上
        try: self.tag_raise(aid)
        except Exception: pass

    def _redraw_arrows(self):
        """画布变化时，按照最新缩放/偏移重画箭头形状。"""
        if not self.arrows:
            return
        for a in self.arrows:
            lane_idx = a["lane_idx"]
            y_img = float(a["y_img"])
            # 若未显式记录 x_img，则动态取该 y 的左分界线（支持斜界线）
            if "x_img" in a:
                x_img = float(a["x_img"])
            else:
                xg = self._lane_left_x_at_y_gel(int(round(y_img - self.panel_top_h)), lane_idx)
                x_img = float(self.x_offset + xg)
            # 限制可见范围
            H, W = self.base_img_bgr.shape[:2]
            x_img = float(np.clip(x_img, 0, W - 1))
            pts = self._arrow_canvas_points(x_img, y_img)




    # 在 RightAnnoCanvas 类内新增此工具方法（位置不拘）
    def _arrow_key(self, arrow_meta: dict):
        """
        生成可稳定识别箭头的绑定键： (lane_idx, mw:float)
        lane_idx 与 mw 共同唯一标识一枚箭头。
        """
        try:
            li = int(arrow_meta.get("lane_idx", -1))
            mw = float(arrow_meta.get("mw", float("nan")))
            if li < 0 or not np.isfinite(mw):
                return None
            return (li, mw)
        except Exception:
            return None

# === 方框：创建/重绘/事件/灰度和 ===
    def _create_box_for_arrow(self, arrow_meta: dict, preset: dict=None):
        """
        基于箭头位置创建红色可调矩形框及顶部黄色文字。
        新增：
        - 支持 preset={'x_gel','y_gel','w','h'} 以“凝胶坐标”为准恢复方框位置；
        - 在 box 中同时记录 x_img/y_img 与 x_gel/y_gel，其中 x_gel/y_gel 为跨场景复原的权威坐标；
        - 记录 bind_key 用于与箭头关联（箭头被删除则本框不再恢复）。
        """
        if self.base_img_bgr is None:
            return

        H, W = self.base_img_bgr.shape[:2]
        # ——— 计算初始/恢复位置（像素坐标系 & 凝胶坐标系） ———
        if preset and all(k in preset for k in ("x_gel","y_gel","w","h")):
            # 以 preset 的凝胶坐标为准，换算到当前像素坐标
            x_gel = float(preset["x_gel"])
            y_gel = float(preset["y_gel"])
            w0 = float(preset["w"]); h0 = float(preset["h"])
            bx = float(self.x_offset + x_gel)
            by = float(self.panel_top_h + y_gel)
            # 限制到图像范围
            bx = float(np.clip(bx, 0, max(0, W - w0)))
            by = float(np.clip(by, 0, max(0, H - h0)))
        else:
            # 沿用原逻辑：将方框放在箭头尖端左侧
            x_img = float(arrow_meta.get("x_img", 0.0))
            y_img = float(arrow_meta.get("y_img", 0.0))
            w0, h0 = 40.0, 24.0
            margin = 6.0
            tip_x = x_img
            bx = tip_x - margin - w0 + 40  # 左上 x（保持你原先修正）
            by = y_img - h0 / 2.0
            bx = float(np.clip(bx, 0, max(0, W - w0)))
            by = float(np.clip(by, 0, max(0, H - h0)))
            # 对应的凝胶坐标
            x_gel = bx - self.x_offset
            y_gel = by - self.panel_top_h

        rect_id = self.create_rectangle(0, 0, 0, 0, outline="#ff4545", width=2)
        handles = {
            "nw": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "ne": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "se": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "sw": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
        }
        text_id = self.create_text(0, 0, text="", fill="#ffd800", anchor="s")

        box = {
            "rect_id": rect_id,
            "text_id": text_id,
            "handles": handles,
            # —— 双坐标记录 ——（x_gel/y_gel 为权威，x_img/y_img 用于绘制与导出）
            "x_gel": float(x_gel), "y_gel": float(y_gel),
            "x_img": float(bx),    "y_img": float(by),
            "w": float(w0), "h": float(h0),
            # 绑定箭头元信息 & 绑定键
            "arrow": arrow_meta,
            "bind_key": self._arrow_key(arrow_meta),
        }
        self.boxes.append(box)

        # 初次绘制
        self._draw_box(box)

        # 绑定事件：矩形整体拖拽
        self.tag_bind(rect_id, "<ButtonPress-1>",  lambda e, b=box: self._on_box_press(e, b, mode="move",   corner=None))
        self.tag_bind(rect_id, "<B1-Motion>",      lambda e, b=box: self._on_box_drag(e, b))
        self.tag_bind(rect_id, "<ButtonRelease-1>",lambda e, b=box: self._on_box_release(e, b))

        # 手柄拖拽（四角缩放）
        self.tag_bind(handles["nw"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="nw"))
        self.tag_bind(handles["ne"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="ne"))
        self.tag_bind(handles["se"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="se"))
        self.tag_bind(handles["sw"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="sw"))
        for c in ("nw","ne","se","sw"):
            hid = handles[c]
            self.tag_bind(hid, "<B1-Motion>",       lambda e, b=box: self._on_box_drag(e, b))
            self.tag_bind(hid, "<ButtonRelease-1>", lambda e, b=box: self._on_box_release(e, b))

    def _redraw_boxes(self):
        """画布尺寸/底图变化时，重绘所有方框与文字。"""
        if not self.boxes_enabled or not self.boxes:
            return
        for box in self.boxes:
            self._draw_box(box)

    def set_boxes_enabled(self, enabled: bool):
        """
        外部开关：是否显示/生成红色方框。
        - True：基于当前所有箭头“按当下位置”一次性生成方框；
        - False：删除所有方框。
        """
        enabled = bool(enabled)
        if enabled and self.boxes_enabled and self.boxes:
            # 已经启用且已有方框，无需重复生成
            return
        if not enabled:
            self._delete_all_boxes()
            self.boxes_enabled = False
            return

        # 启用：按当前箭头生成一次
        self._delete_all_boxes()
        if self.arrows:
            for meta in self.arrows:
                self._create_box_for_arrow(meta)
        self.boxes_enabled = True
        self._redraw_boxes()

    def _delete_all_boxes(self):
        """删除画布上所有已创建的方框及其手柄与文字，并清空集合。"""
        try:
            for box in self.boxes:
                try:
                    self.delete(box.get("rect_id"))
                except Exception:
                    pass
                try:
                    self.delete(box.get("text_id"))
                except Exception:
                    pass
                try:
                    for hid in (box.get("handles") or {}).values():
                        self.delete(hid)
                except Exception:
                    pass
        finally:
            self.boxes.clear()
            self._box_drag.update({"box": None, "mode": None, "corner": None})

    def _draw_box(self, box: dict):
        """
        根据当前缩放/偏移，将 box 的图像坐标绘制为 Canvas 坐标，并更新黄色文字（灰度和）。
        新增：以 box['x_gel','y_gel'] + 当前 x_offset/panel_top_h 计算 x_img/y_img，并回填。
        """
        # 先由凝胶坐标换算像素坐标（权威 -> 表现）
        x = float(self.x_offset + float(box.get("x_gel", box.get("x_img", 0.0) - self.x_offset)))
        y = float(self.panel_top_h + float(box.get("y_gel", box.get("y_img", 0.0) - self.panel_top_h)))
        w = float(box.get("w", 0.0)); h = float(box.get("h", 0.0))

        # 回填像素坐标，供度量/兼容路径使用
        box["x_img"], box["y_img"] = x, y

        # 转为 Canvas 坐标
        x1, y1 = self._to_canvas(x, y)
        x2, y2 = self._to_canvas(x + w, y + h)

        # 更新矩形
        self.coords(box["rect_id"], x1, y1, x2, y2)

        # 更新手柄
        hs = 2
        corners = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "se": (x2, y2),
            "sw": (x1, y2),
        }
        for k, (cx, cy) in corners.items():
            self.coords(box["handles"][k], cx - hs, cy - hs, cx + hs, cy + hs)

        # 计算灰度和，更新文本位置
        val = self._gray_sum_in_box(box)
        txt = f"{int(val)}"
        cx = (x1 + x2) / 2.0
        ty = y1 - 4
        self.coords(box["text_id"], cx, ty)
        self.itemconfig(box["text_id"], text=txt)


    def _on_box_press(self, evt, box: dict, mode: str, corner: str | None):
        """开始拖拽/缩放方框。"""
        # 记录起点（转换到图像坐标）
        ix, iy = self._to_img(evt.x, evt.y)
        self._box_drag["box"] = box
        self._box_drag["mode"] = mode
        self._box_drag["corner"] = corner
        self._box_drag["start_img"] = (float(ix), float(iy))
        self._box_drag["orig"] = (box["x_img"], box["y_img"], box["w"], box["h"])

    def _on_box_drag(self, evt, box: dict):
        """
        拖拽过程：平移或按角缩放。
        新增：任何位置更新后，除更新 x_img/y_img 外，同时更新 x_gel/y_gel= x_img - x_offset, y_img - panel_top_h，
            以保证跨场景复原的独立位置记录。
        """
        if self._box_drag["box"] is not box:
            return
        ix, iy = self._to_img(evt.x, evt.y)
        sx, sy = self._box_drag["start_img"]
        dx, dy = float(ix - sx), float(iy - sy)
        x0, y0, w0, h0 = self._box_drag["orig"]
        H, W = self.base_img_bgr.shape[:2]
        min_w, min_h = 6.0, 6.0

        if self._box_drag["mode"] == "move":
            nx = np.clip(x0 + dx, 0, max(0, W - w0))
            ny = np.clip(y0 + dy, 0, max(0, H - h0))
            box["x_img"], box["y_img"] = float(nx), float(ny)
        else:
            corner = self._box_drag["corner"]
            if corner == "nw":
                nx1 = np.clip(x0 + dx, 0, x0 + w0 - min_w)
                ny1 = np.clip(y0 + dy, 0, y0 + h0 - min_h)
                box["w"] = (x0 + w0) - nx1
                box["h"] = (y0 + h0) - ny1
                box["x_img"], box["y_img"] = float(nx1), float(ny1)
            elif corner == "ne":
                nx2 = np.clip(x0 + w0 + dx, x0 + min_w, W)
                ny1 = np.clip(y0 + dy, 0, y0 + h0 - min_h)
                box["w"] = nx2 - x0
                box["h"] = (y0 + h0) - ny1
                box["x_img"], box["y_img"] = float(x0), float(ny1)
            elif corner == "se":
                nx2 = np.clip(x0 + w0 + dx, x0 + min_w, W)
                ny2 = np.clip(y0 + h0 + dy, y0 + min_h, H)
                box["w"] = nx2 - x0
                box["h"] = ny2 - y0
                box["x_img"], box["y_img"] = float(x0), float(y0)
            elif corner == "sw":
                nx1 = np.clip(x0 + dx, 0, x0 + w0 - min_w)
                ny2 = np.clip(y0 + h0 + dy, y0 + min_h, H)
                box["w"] = (x0 + w0) - nx1
                box["h"] = ny2 - y0
                box["x_img"], box["y_img"] = float(nx1), float(y0)

        # ★ 同步更新凝胶坐标（权威记录）
        box["x_gel"] = float(box["x_img"] - self.x_offset)
        box["y_gel"] = float(box["y_img"] - self.panel_top_h)

        # 重绘
        self._draw_box(box)

    def _on_box_release(self, _evt, _box: dict):
        """结束拖拽/缩放。"""
        self._box_drag.update({"box": None, "mode": None, "corner": None})


    def _gray_sum_and_count_in_box(self, box: dict) -> tuple[float, int]:
            """
            计算 box 与凝胶区域的交叠部分在 gel_gray 上的“灰度和”与“像素数量”。
            - 灰度和使用 (255 - gray) 的和（即“暗度积分”）；
            - count 是交叠区域的像素数；
            - 若无交叠，返回 (0.0, 0)。
            """
            if self.gel_gray is None or self.gel_gray.size == 0:
                return 0.0, 0
            Hg, Wg = self.gel_size

            # 将方框映射到“凝胶坐标系”
            x1_img, y1_img = float(box["x_img"]), float(box["y_img"])
            x2_img, y2_img = x1_img + float(box["w"]), y1_img + float(box["h"])
            gx1 = int(round(x1_img - self.x_offset))
            gy1 = int(round(y1_img - self.panel_top_h))
            gx2 = int(round(x2_img - self.x_offset))
            gy2 = int(round(y2_img - self.panel_top_h))

            # 与凝胶区域求交裁剪
            gx1 = max(0, min(Wg, gx1))
            gx2 = max(0, min(Wg, gx2))
            gy1 = max(0, min(Hg, gy1))
            gy2 = max(0, min(Hg, gy2))
            if gx2 <= gx1 or gy2 <= gy1:
                return 0.0, 0

            roi = self.gel_gray[gy1:gy2, gx1:gx2]
            count = int(roi.size)
            s = float(np.sum(255 - roi))
            return s, count
    
    def _gray_sum_in_box(self, box: dict) -> float:
        """
        （保持向后兼容）返回灰度和 = sum(255 - gray)。
        现已委托给 _gray_sum_and_count_in_box。
        """
        s, _ = self._gray_sum_and_count_in_box(box)
        return s

    def get_box_metrics(self) -> list[dict]:
        """
        导出“每个箭头对应方框”的度量信息，返回列表，每项包含：
          - lane_idx        : 0-based 真实泳道序号
          - lane_no         : 1-based 真实泳道序号（便于人读）
          - mw              : 分子量（kDa）
          - sum_intensity   : 灰度和（sum(255 - gray)）
          - pixel_count     : 参与统计像素数
          - mean_intensity  : 每像素平均（sum / count），若 count=0 则为 0
          - x, y, w, h      : 方框在“最终导出图像坐标系”中的位置与尺寸
          - arrow_x, arrow_y: 箭头尖端在图像坐标系的位置（如可得）
        说明：仅统计方框与“凝胶区域”的交叠部分。
        """
        out = []
        if not self.boxes:
            return out

        for box in self.boxes:
            s, n = self._gray_sum_and_count_in_box(box)
            mean = (s / n) if n > 0 else 0.0
            arrow = box.get("arrow") or {}
            lane_idx = arrow.get("lane_idx", None)
            y_img = float(arrow.get("y_img", np.nan))
            x_img = float(arrow.get("x_img", np.nan))
            out.append({
                "lane_idx": int(lane_idx) if lane_idx is not None else None,
                "lane_no": (int(lane_idx) + 1) if lane_idx is not None else None,
                "mw": float(arrow.get("mw")) if arrow.get("mw") is not None else None,
                "sum_intensity": float(s),
                "pixel_count": int(n),
                "mean_intensity": float(mean),
                "x": float(box.get("x_img", 0.0)),
                "y": float(box.get("y_img", 0.0)),
                "w": float(box.get("w", 0.0)),
                "h": float(box.get("h", 0.0)),
                "arrow_x": x_img,
                "arrow_y": y_img,
            })
        return out



# ========== 画布级命中与拖拽（支持“按住方框内部移动”） ==========

    def _point_in_box_img(self, box: dict, ix: float, iy: float) -> bool:
        """判断图像坐标 (ix, iy) 是否落在 box 的图像坐标矩形内。"""
        x1, y1 = float(box["x_img"]), float(box["y_img"])
        x2, y2 = x1 + float(box["w"]), y1 + float(box["h"])
        return (x1 <= ix <= x2) and (y1 <= iy <= y2)

    def _on_canvas_press(self, evt):
        """
        当点击画布空白或底图时，检测是否落在某个方框内部；
        若是，则启动该方框的 move 拖拽（不干扰已有 item 级绑定）。
        """
        # 若已经在进行方框拖拽/缩放，则不重复进入
        if self._box_drag["box"] is not None:
            return

        # 如果点击到了已有 item（如矩形边、手柄、箭头），交给原有 item 事件处理
        cur = self.find_withtag("current")
        if cur:
            cur_id = int(cur[0])
            # 命中任一方框的矩形/手柄/文字 —— 交给原有 item 绑定
            for b in self.boxes:
                if cur_id == b["rect_id"] or cur_id == b["text_id"] or cur_id in b["handles"].values():
                    return
            # 命中任一箭头 —— 交给箭头自身的绑定
            for a in self.arrows:
                if cur_id == int(a["id"]):
                    return

        # 画布坐标 -> 图像坐标
        ix, iy = self._to_img(evt.x, evt.y)

        # 从上到下优先命中“顶层方框”（后创建的在上层）
        for b in reversed(self.boxes):
            if self._point_in_box_img(b, ix, iy):
                self._box_drag["box"] = b
                self._box_drag["mode"] = "move"
                self._box_drag["corner"] = None
                self._box_drag["start_img"] = (float(ix), float(iy))
                self._box_drag["orig"] = (b["x_img"], b["y_img"], b["w"], b["h"])
                # 提升 z 顺序：矩形 < 手柄 < 文本
                try:
                    self.tag_raise(b["rect_id"])
                    for hid in b["handles"].values():
                        self.tag_raise(hid)
                    self.tag_raise(b["text_id"])
                except Exception:
                    pass
                break

    def _on_canvas_drag(self, evt):
        """当由画布级命中触发了方框移动时，复用已有 _on_box_drag 逻辑。"""
        box = self._box_drag.get("box")
        mode = self._box_drag.get("mode")
        if box is None or mode is None:
            return
        # 仅处理通过画布命中的移动（缩放仍由手柄 item 绑定处理）
        if mode == "move":
            self._on_box_drag(evt, box)

    def _on_canvas_release(self, evt):
        """结束画布级触发的方框拖拽。"""
        box = self._box_drag.get("box")
        if box is None:
            return
        self._on_box_release(evt, box)

    def reset_arrows_to_default(self):
        """
        将当前画布上的所有箭头位置重置为默认位置：
        - X：该 y 对应泳道“左分界线”；
        - Y：按拟合 y = a*log10(mw) + b 映射的 y_gel，再加 panel_top_h；
        同时把 'moved' 标记清零；若启用了红色方框，则按新箭头重建方框。
        """
        if not getattr(self, "arrows", None) or self.base_img_bgr is None:
            return
        Hg, Wg = self.gel_size
        H_img, W_img = self.base_img_bgr.shape[:2]
        ymin = self.panel_top_h
        ymax = self.panel_top_h + max(0, Hg - 1)

        for a in self.arrows:
            try:
                lane_idx = int(a.get("lane_idx", 0))
                mw = float(a.get("mw", float("nan")))
                if not (np.isfinite(mw) and mw > 0):
                    continue
                # 计算默认 y/x（相对当前 scene 的 panel_top_h 与 x_offset）
                y_gel = int(round(self._mw_to_y_gel(mw)))
                y_img = float(np.clip(self.panel_top_h + y_gel, ymin, ymax))
                xl_gel = self._lane_left_x_at_y_gel(y_gel, lane_idx)
                x_img = float(np.clip(self.x_offset + xl_gel, 0, W_img - 1))

                a["x_img"] = x_img
                a["y_img"] = y_img
                a["moved"] = False  # 清零移动标记

                # 重绘箭头形状
                pts = self._arrow_canvas_points(x_img, y_img)
                self.coords(a["id"], *pts)
            except Exception:
                continue

        # 若当前启用了“显示红色方框”：按最新箭头重建方框
        if getattr(self, "boxes_enabled", False):
            try:
                self._delete_all_boxes()
                for meta in self.arrows:
                    self._create_box_for_arrow(meta)
            except Exception:
                pass



    def _on_drag_start(self, evt, arrow_meta: dict):
        self._drag["id"] = arrow_meta["id"]
        self._drag["y0_canvas"] = evt.y


    def _on_drag_move(self, evt, arrow_meta: dict):
        if self._drag["id"] != arrow_meta["id"]:
            return
        # 画布坐标 → 图像坐标
        x_img, y_img = self._to_img(evt.x, evt.y)

        # 边界：Y 在 gel 高度内；X 在整图宽度内
        H, W = self.base_img_bgr.shape[:2]
        Hg, Wg = self.gel_size
        ymin = self.panel_top_h
        ymax = self.panel_top_h + Hg - 1
        x_img = float(np.clip(x_img, 0, W - 1))
        y_img = float(np.clip(y_img, ymin, ymax))

        # 保存并重绘
        arrow_meta["x_img"] = x_img
        arrow_meta["y_img"] = y_img
        arrow_meta["moved"] = True

        pts = self._arrow_canvas_points(x_img, y_img)
        self.coords(arrow_meta["id"], *pts)

        # ★ 抬高并请求一次 UI 刷新，减少残影概率
        try:
            self.tag_raise(arrow_meta["id"])
            self.update_idletasks()
        except Exception:
            pass



class App(tk.Tk):


    def _ensure_win_appid(self, appid: str):
        """Windows: 设置 AppUserModelID，保证任务栏分组与图标一致（需尽早调用）。"""
        try:
            if os.name == "nt":
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
        except Exception:
            pass

    def _load_app_icon(self):
        """
        统一加载应用图标：
        - Windows：优先用 gelannot.ico（影响窗口左上角图标，也会影响任务栏显示）
        - 其他平台：退回用 PNG 通过 iconphoto 设置
        同时兼容 PyInstaller 打包路径（sys._MEIPASS）
        """
        try:
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
            ico = os.path.join(base_dir, "gelannot.ico")
            png = os.path.join(base_dir, "gelannot.png")

            if os.name == "nt" and os.path.exists(ico):
                # 标准方式：标题栏 + 任务栏均使用 ICO
                self.iconbitmap(ico)
            elif os.path.exists(png):
                from PIL import Image, ImageTk
                self.iconphoto(True, ImageTk.PhotoImage(Image.open(png)))
        except Exception:
            # 不阻断主程序
            pass



    def __init__(self):
        super().__init__()
        self.title("Electrophoresis image visual processing")
        self.geometry("1000x720")
        
        self._ensure_win_appid("JunYe.GelAnnotator.1")
        self._load_app_icon()

        # 数据
        self.orig_bgr: np.ndarray | None = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.gi = 1  # 当前胶索引（1-based）
        self.render_cache = {}
        self.LEFT_WIDTH = 300

        # —— 标签/标注数据 —— #
        self.var_label_rows = tk.IntVar(value=1)       # （保留）标签行数
        self.var_labels_on = tk.BooleanVar(value=True) # 是否附加标签面板（白底）
        self.custom_labels: list[list[str]] = []       # （保留）上方白底文本标签二维表

        # 新增：每列（泳道）元信息
        self.lane_names: list[str] = []               # 每列显示的“列名”
        self.lane_marks: list[list[float]] = []       # 每列要标注的分子量（kDa）

        # —— 自适应图片显示注册表（右侧两个预览） —— #
        # widget -> {"img": np.ndarray, "last": (w,h)}
        self._autofit_store: dict[tk.Label, dict] = {}

        self.stash: list[dict] = []

        # 左右UI
        self._build_left()
        self._build_right()
        self._init_ladder_presets()

    # ===== 标准分子量集合：持久化与初始化 =====
    def _preset_store_path(self) -> Path:
        # ~/.gel_annotator/ladder_presets.json
        base = Path.home() / ".gel_annotator"
        base.mkdir(parents=True, exist_ok=True)
        return base / "ladder_presets.json"

    def _load_ladder_presets(self) -> list[dict]:
        """返回 [{'name': str, 'values': [float], 'note': str}, ...] """
        p = self._preset_store_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                items = data.get("items", []) if isinstance(data, dict) else data
                # 基本清洗
                out = []
                for it in items:
                    name = str(it.get("name", "")).strip()
                    vals = [float(v) for v in it.get("values", []) if isinstance(v, (int,float))]
                    note = str(it.get("note", "")).strip()
                    if name and vals:
                        out.append({"name": name, "values": vals, "note": note})
                # 去重（以名称为准，后者覆盖前者）
                seen = {}
                for it in out:
                    seen[it["name"]] = it
                return list(seen.values())
            except Exception:
                pass
        # 不存在或损坏：写入一个默认集合
        default_items = [{
            "name": "10 Usuals:(10–180 kDa)",
            "values": [180,130,95,65,52,41,31,25,17,10],
            "note": "Default"
        }]
        self._save_ladder_presets(default_items)
        return default_items

    def _save_ladder_presets(self, items: list[dict]):
        p = self._preset_store_path()
        data = {"version": 1, "items": items}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _init_ladder_presets(self):
        """一次性初始化变量、加载并刷新 UI"""
        self._ladder_items: list[dict] = self._load_ladder_presets()   # 内存集合
        self.var_preset = getattr(self, "var_preset", tk.StringVar(value=""))
        self._refresh_preset_ui()

    def _refresh_preset_ui(self):
        """刷新下拉框与信息标签"""
        names = [it["name"] for it in getattr(self, "_ladder_items", [])]
        if hasattr(self, "cb_preset"):
            self.cb_preset["values"] = names
        # 如果当前选项已被删除，则清空
        cur = self.var_preset.get().strip()
        if cur not in names:
            self.var_preset.set("")
            if hasattr(self, "lbl_preset_info"):
                self.lbl_preset_info.configure(text="No standard selected")

    def _apply_preset_to_entry(self, name: str):
        """把指定集合写回到文本框，并在下方显示详情"""
        items = getattr(self, "_ladder_items", [])
        match = next((it for it in items if it["name"] == name), None)
        if not match:
            return
        # 写回原有输入框（保持兼容渲染逻辑）
        vals = match["values"]
        self.ent_marker.delete(0, tk.END)
        self.ent_marker.insert(0, ", ".join(f"{v:g}" for v in vals))
        # 显示信息 —— 中文改为英文
        note = match.get("note", "").strip()
        info = f"Set: {name} | Count: {len(vals)} | Values: {', '.join(f'{v:g}' for v in vals)}"
        if note:
            info += f"\nNote: {note}"
        if hasattr(self, "lbl_preset_info"):
            self.lbl_preset_info.configure(text=info)


    def on_select_preset(self, *_):
        name = self.var_preset.get().strip()
        if name:
            self._apply_preset_to_entry(name)

    def open_preset_manager(self):
        """查看/新建/编辑/删除 标准分子量集合"""
        win = tk.Toplevel(self)
        win.title("Manage")
        win.transient(self); win.grab_set()

        frm = ttk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左：列表
        left = ttk.Frame(frm); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Set list").pack(anchor="w")
        lb = tk.Listbox(left, width=26, height=16)
        lb.pack(fill=tk.Y, expand=False, pady=(4,8))
        def _reload_list(select_name: str|None=None):
            lb.delete(0, tk.END)
            for it in self._ladder_items:
                lb.insert(tk.END, it["name"])
            if select_name:
                try:
                    idx = [it["name"] for it in self._ladder_items].index(select_name)
                    lb.selection_set(idx); lb.see(idx)
                except Exception:
                    pass

        # 右：编辑区
        right = ttk.Frame(frm); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Name").grid(row=0, column=0, sticky="w")
        ent_name = ttk.Entry(right); ent_name.grid(row=0, column=1, sticky="ew", padx=(6,0))
        ttk.Label(right, text="Molecular weights (supporting , ; space / Chinese comma, sorted from large to small)").grid(row=1, column=0, columnspan=2, sticky="w", pady=(8,0))
        txt_vals = tk.Text(right, height=5); txt_vals.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4,0))
        ttk.Label(right, text="Note").grid(row=3, column=0, sticky="w", pady=(8,0))
        txt_note = tk.Text(right, height=4); txt_note.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(4,0))

        right.columnconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.rowconfigure(4, weight=1)

        def _parse_vals(raw: str) -> list[float]:
            s = (raw or "").replace("，", ",").replace("；", ";").replace("、", ",")
            for ch in [":", "：", ";", " ", "\t", "\n", "\r", ";"]:
                s = s.replace(ch, ",")
            out = []
            for t in s.split(","):
                t = t.strip()
                if not t: continue
                try:
                    v = float(t)
                    if np.isfinite(v) and v > 0:
                        out.append(v)
                except Exception:
                    pass
            return out

        def _fill_editor(it: dict|None):
            ent_name.delete(0, tk.END)
            txt_vals.delete("1.0", tk.END)
            txt_note.delete("1.0", tk.END)
            if not it: return
            ent_name.insert(0, it["name"])
            txt_vals.insert("1.0", ", ".join(f"{v:g}" for v in it["values"]))
            if it.get("note"):
                txt_note.insert("1.0", it["note"])

        def _current_item() -> dict|None:
            sel = lb.curselection()
            if not sel: return None
            idx = sel[0]
            if 0 <= idx < len(self._ladder_items):
                return self._ladder_items[idx]
            return None

        def on_list_select(_evt=None):
            _fill_editor(_current_item())

        lb.bind("<<ListboxSelect>>", on_list_select)

        # 按钮区
        btns = ttk.Frame(right); btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10,0))
        def do_new():
            _fill_editor({"name": "", "values": [], "note": ""})
            ent_name.focus_set()

        def do_save():
            name = ent_name.get().strip()
            vals = _parse_vals(txt_vals.get("1.0", "end"))
            note = txt_note.get("1.0", "end").strip()
            if not name:
                messagebox.showwarning("Notice", "Fill in the name please"); return
            if not vals:
                messagebox.showwarning("Notice", "Please enter at least one valid molecular weight value"); return
            # 去重：同名覆盖
            new_items = []
            replaced = False
            for it in self._ladder_items:
                if it["name"] == name:
                    new_items.append({"name": name, "values": vals, "note": note})
                    replaced = True
                else:
                    new_items.append(it)
            if not replaced:
                new_items.append({"name": name, "values": vals, "note": note})
            self._ladder_items = new_items
            self._save_ladder_presets(self._ladder_items)
            self._refresh_preset_ui()
            _reload_list(select_name=name)
            # 同步到主界面（若正选中）
            self.var_preset.set(name)
            self._apply_preset_to_entry(name)

        def do_delete():
            it = _current_item()
            if not it: return
            if not messagebox.askyesno("Confirm", f"Delete set「{it['name']}」?"):
                return
            self._ladder_items = [x for x in self._ladder_items if x["name"] != it["name"]]
            self._save_ladder_presets(self._ladder_items)
            self._refresh_preset_ui()
            _reload_list()
            _fill_editor(None)
            # 若删除的是当前选中项，清空显示
            if self.var_preset.get().strip() == it["name"]:
                self.var_preset.set("")
                if hasattr(self, "lbl_preset_info"):
                    self.lbl_preset_info.configure(text="No standard set selected.")

        ttk.Button(btns, text="New", command=do_new).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="Save", command=do_save).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="Delete", command=do_delete).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        _reload_list()

    def open_right_table_editor(self):
        """
        打开“右侧表格”编辑窗口：
        - 直接粘贴 Excel 多行多列内容（Tab 分隔，多行换行）
        - 支持选项：首行表头、显示网格、对齐方式、字号（像素字高）
        - 确认后仅做“组合”（不重复核心运算），与底注逻辑一致
        """
        win = tk.Toplevel(self)
        win.title("Edit right-side table (paste from Excel)")
        win.transient(self); win.grab_set()
        win.resizable(True, True)

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frm, text="Paste here (Excel -> Copy; Click here -> Ctrl+V):", justify="left").pack(anchor="w")
        txt = tk.Text(frm, height=12, width=64, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        # 预填
        try:
            pre = getattr(self, "right_table_text", "") or ""
            if pre:
                txt.insert("1.0", pre)
        except Exception:
            pass

        # 选项区
        opt = ttk.LabelFrame(frm, text="Options")
        opt.pack(fill=tk.X, pady=(6, 0))

        var_header = tk.BooleanVar(value=True)
        var_grid   = tk.BooleanVar(value=True)
        var_align  = tk.StringVar(value="center")
        # 字号（像素字高）
        design = self._get_design_params()
        var_cap   = tk.IntVar(value=int(design.get("bottom_cap_px", 28)))

        # 回显历史配置
        prev_cfg = getattr(self, "right_table_opts", None) or {}
        if "has_header" in prev_cfg: var_header.set(bool(prev_cfg.get("has_header")))
        if "grid" in prev_cfg:       var_grid.set(bool(prev_cfg.get("grid")))
        if "align" in prev_cfg:      var_align.set(str(prev_cfg.get("align") or "center"))
        if "cap_px" in prev_cfg and isinstance(prev_cfg["cap_px"], (int, float)):
            var_cap.set(int(prev_cfg["cap_px"]))

        row1 = ttk.Frame(opt); row1.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(row1, text="First row is header (gray)", variable=var_header).pack(side=tk.LEFT)
        ttk.Checkbutton(row1, text="Show grid lines", variable=var_grid).pack(side=tk.LEFT, padx=(12, 0))

        row2 = ttk.Frame(opt); row2.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row2, text="Align").pack(side=tk.LEFT)
        cb_align = ttk.Combobox(row2, textvariable=var_align, state="readonly", values=["left", "center", "right"], width=8)
        cb_align.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row2, text="Font height (px)").pack(side=tk.LEFT)
        sp_cap = ttk.Spinbox(row2, textvariable=var_cap, from_=10, to=60, increment=1, width=6)
        sp_cap.pack(side=tk.LEFT, padx=(6, 0))

        # 按钮区
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))

        def do_clear():
            txt.delete("1.0", "end")

        def do_cancel():
            win.destroy()

        def do_ok():
            try:
                self.right_table_text = txt.get("1.0", "end")
            except Exception:
                self.right_table_text = ""
            self.right_table_opts = {
                "has_header": bool(var_header.get()),
                "grid": bool(var_grid.get()),
                "align": (var_align.get() or "center"),
                "cap_px": int(var_cap.get()),
            }
            win.destroy()
            # 组合（若缓存可复用）——失败则回退完整渲染
            try:
                self.recompose_using_cache()
            except Exception:
                try: self.render_current()
                except Exception: pass

        ttk.Button(btns, text="Clear", command=do_clear).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel", command=do_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Confirm", command=do_ok).pack(side=tk.RIGHT, padx=(0, 6))

        win.bind("<Escape>", lambda e: (do_cancel(), "break"))

    def _attach_right_table_panel(
        self,
        img_bgr: np.ndarray,
        table_text: str,
        has_header: bool = True,
        show_grid: bool = True,
        align: str = "center",  # "left" / "center" / "right"
        cap_px: int = None,     # 用户可见的“Font height (px)”；在 cap_policy 下可能被上限约束
        cell_pad_x: int = 12,
        cell_pad_y: int = 8,
        line_color: tuple = (0, 0, 0),  # BGR
        gap_px: int = 30,
        panel_top_h: int = 0,
        gel_height_px: int = None,      # 参考高度优先使用胶高；否则 H_img - panel_top_h
        cap_policy: str = "max_auto"    # "max_auto"（默认）|"auto_only"|"free"
    ) -> np.ndarray:
        """
        右侧白底表格列（从 Excel 粘贴文本解析）：
        - shrink-to-fit：表格过高则自动等比缩小“字号+内边距”，直到完整适配可用高度；
        - cap_policy 控制“用户设置 cap_px 与按高度自动计算的上限”的关系：
            · "max_auto"（默认）：字号 <= 按高度计算的 auto_cap；用户设置只能调小，不能放大；
            · "auto_only"：完全忽略 cap_px，始终用 auto_cap；
            · "free"：尊重 cap_px（与前一版相同）。
        """
        import numpy as np, cv2
        from PIL import Image, ImageDraw, ImageFont

        # ---------- 基础 ----------
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return img_bgr
        H_img, W_img = img_bgr.shape[:2]
        panel_top_h = max(0, int(panel_top_h))
        body_avail_h = max(0, H_img - panel_top_h)  # 表格主体可用高度（顶对齐）

        # ---------- 解析粘贴文本 ----------
        raw = (table_text or "").replace("\r\n", "\n").replace("\r", "\n")
        rows_raw = raw.split("\n")
        rows = []
        for ln in rows_raw:
            sline = ln
            if "\t" in sline:
                cells = sline.split("\t")
            else:
                s2 = sline.replace("，", ",").replace("；", ";").replace("、", ",")
                if "," in s2:
                    cells = s2.split(",")
                elif ";" in s2:
                    cells = s2.split(";")
                elif "|" in s2:
                    cells = s2.split("|")
                else:
                    cells = [s2]
            rows.append([c.strip() for c in cells])
        rows = [r for r in rows if any((c.strip() for c in r))]
        if not rows:
            gap = np.full((H_img, max(1, gap_px), 3), 255, dtype=np.uint8)
            return np.concatenate([img_bgr, gap], axis=1)
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r.extend([""] * (max_cols - len(r)))

        # ---------- 初始字号：按“高度/1000”为基准，叠加 cap_policy 上限 ----------
        ref_h = int(gel_height_px) if isinstance(gel_height_px, (int, float)) and gel_height_px else int(body_avail_h)
        ref_h = max(1, ref_h)
        s_ref = max(0.35, float(ref_h) / 1000.0)  # 下限避免过小
        design = self._get_design_params()         # bottom_cap_px ≈ 28
        base_cap = int(design.get("bottom_cap_px", 28))
        auto_cap = max(10, int(round(base_cap * s_ref)))  # “按高度自动”字号上限

        # 处理 cap_policy
        if str(cap_policy).lower() == "auto_only":
            cap_init = auto_cap
        elif str(cap_policy).lower() == "max_auto":
            user_cap = int(cap_px) if isinstance(cap_px, (int, float)) and cap_px else auto_cap
            cap_init = min(int(user_cap), int(auto_cap))   # << 关键：不允许超过自动上限
        else:  # "free"
            cap_init = int(cap_px) if isinstance(cap_px, (int, float)) and cap_px else auto_cap

        # 内边距随 s 缩放（不提供外部放大开关，因此直接按 auto）
        pad_x_init = max(6, int(round(cell_pad_x * s_ref)))
        pad_y_init = max(4, int(round(cell_pad_y * s_ref)))

        # ---------- 最小限（可按需调整） ----------
        MIN_CAP   = 12
        MIN_PAD_X = 4
        MIN_PAD_Y = 3
        # ---------- 字体引擎（Pillow 优先，回退 OpenCV） ----------
        font_path = self._find_font_ttf()
        use_pil = font_path is not None

        def _measure_table(cap_px_cur: int, pad_x_cur: int, pad_y_cur: int, want_pil: bool):
            """测量列宽/行高，返回 (col_widths, row_heights, grid_w, grid_h, engine, ctx)。"""
            if want_pil:
                try:
                    font = ImageFont.truetype(font_path, size=max(12, int(cap_px_cur)))
                except Exception:
                    return _measure_table(cap_px_cur, pad_x_cur, pad_y_cur, False)
                tmp = Image.new("RGB", (8, 8), "white")
                drw = ImageDraw.Draw(tmp)
                def meas(text: str):
                    t = text if text else " "
                    bbox = drw.textbbox((0, 0), t, font=font)
                    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
                col_widths = [0] * max_cols
                row_heights = [0] * len(rows)
                for i, r in enumerate(rows):
                    max_h = 0
                    for j, cell in enumerate(r):
                        tw, th = meas(cell)
                        col_widths[j] = max(col_widths[j], tw + 2 * pad_x_cur)
                        max_h = max(max_h, th + 2 * pad_y_cur)
                    row_heights[i] = max(1, max_h)
                grid_w = sum(col_widths) + (1 if show_grid else 0)
                grid_h = sum(row_heights) + (1 if show_grid else 0)
                return col_widths, row_heights, grid_w, grid_h, "pil", font
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = self._font_scale_for_cap_height_px(int(cap_px_cur), font=font, thickness=2)
                def meas_cv(text: str):
                    t = text if text else " "
                    (tw, th), _ = cv2.getTextSize(t, font, scale, 2)
                    return max(1, tw), max(1, th)
                col_widths = [0] * max_cols
                row_heights = [0] * len(rows)
                for i, r in enumerate(rows):
                    max_h = 0
                    for j, cell in enumerate(r):
                        tw, th = meas_cv(cell)
                        col_widths[j] = max(col_widths[j], tw + 2 * pad_x_cur)
                        max_h = max(max_h, th + 2 * pad_y_cur)
                    row_heights[i] = max(1, max_h)
                grid_w = sum(col_widths) + (1 if show_grid else 0)
                grid_h = sum(row_heights) + (1 if show_grid else 0)
                return col_widths, row_heights, grid_w, grid_h, "cv", scale

        # ---------- shrink-to-fit：过高时循环缩小（字号+内边距） ----------
        cap_cur  = int(cap_init)
        padx_cur = int(pad_x_init)
        pady_cur = int(pad_y_init)
        engine   = "pil" if use_pil else "cv"
        ctx      = None

        MAX_ITERS = 8
        for _ in range(MAX_ITERS):
            col_w, row_h, grid_w, grid_h, engine, ctx = _measure_table(cap_cur, padx_cur, pady_cur, engine == "pil")
            if grid_h <= body_avail_h:
                break  # 已适配
            f = float(body_avail_h) / float(grid_h)
            f = max(0.50, min(0.98, f))  # 收缩但不过猛
            new_cap  = max(MIN_CAP,   int(round(cap_cur  * f)))
            new_padx = max(MIN_PAD_X, int(round(padx_cur * f)))
            new_pady = max(MIN_PAD_Y, int(round(pady_cur * f)))
            if new_cap == cap_cur and new_padx == padx_cur and new_pady == pady_cur:
                if cap_cur > MIN_CAP:
                    cap_cur -= 1; continue
                if pady_cur > MIN_PAD_Y:
                    pady_cur -= 1; continue
                if padx_cur > MIN_PAD_X:
                    padx_cur -= 1; continue
                break
            cap_cur, padx_cur, pady_cur = new_cap, new_padx, new_pady

        # 最终尺寸
        col_w, row_h, grid_w, grid_h, engine, ctx = _measure_table(cap_cur, padx_cur, pady_cur, engine == "pil")

        # ---------- 绘制表格 ----------
        if engine == "pil":
            font = ctx  # PIL Font
            panel_img = Image.new("RGB", (grid_w, grid_h), "white")
            draw = ImageDraw.Draw(panel_img)
            header_rows = 1 if has_header and len(rows) >= 1 else 0
            if header_rows == 1:
                y_top = 0
                hdr_h = row_h[0]
                draw.rectangle([(0, y_top), (grid_w - 1, y_top + hdr_h - 1)], fill=(240, 240, 240))
            if show_grid:
                y = 0
                draw.line([(0, y), (grid_w - 1, y)], fill=(0, 0, 0), width=1)
                for h in row_h:
                    y += h
                    draw.line([(0, y), (grid_w - 1, y)], fill=(0, 0, 0), width=1)
                x = 0
                draw.line([(x, 0), (x, grid_h - 1)], fill=(0, 0, 0), width=1)
                for w in col_w:
                    x += w
                    draw.line([(x, 0), (x, grid_h - 1)], fill=(0, 0, 0), width=1)
            def _x_for_cell(tw: int, col_left: int, col_wid: int):
                if align == "left":  return col_left + padx_cur
                if align == "right": return col_left + max(0, col_wid - tw - padx_cur)
                return col_left + max(0, (col_wid - tw) // 2)
            tmp = Image.new("RGB", (8, 8), "white")
            drw = ImageDraw.Draw(tmp)
            def _meas_pil(text: str):
                t = text if text else " "
                bb = drw.textbbox((0, 0), t, font=font)
                return max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
            y_cursor = 0
            for i, r in enumerate(rows):
                x_cursor = 0
                for j, cell in enumerate(r):
                    tw, th = _meas_pil(cell)
                    tx = _x_for_cell(tw, x_cursor, col_w[j])
                    ty = y_cursor + max(0, (row_h[i] - th) // 2)
                    if header_rows and i == 0:
                        draw.text((tx, ty), cell or " ", fill=(0, 0, 0), font=font,
                                stroke_width=1, stroke_fill=(0, 0, 0))
                    else:
                        draw.text((tx, ty), cell or " ", fill=(0, 0, 0), font=font)
                    x_cursor += col_w[j]
                y_cursor += row_h[i]
            panel_bgr = cv2.cvtColor(np.array(panel_img), cv2.COLOR_RGB2BGR)
        else:
            font_cv = cv2.FONT_HERSHEY_SIMPLEX
            scale_cv = ctx
            panel_bgr = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
            header_rows = 1 if has_header and len(rows) >= 1 else 0
            if header_rows == 1:
                y_top = 0
                hdr_h = row_h[0]
                cv2.rectangle(panel_bgr, (0, y_top), (grid_w - 1, y_top + hdr_h - 1), (240, 240, 240), thickness=-1)
            if show_grid:
                y = 0
                cv2.line(panel_bgr, (0, y), (grid_w - 1, y), (0, 0, 0), 1)
                for h in row_h:
                    y += h
                    cv2.line(panel_bgr, (0, y), (grid_w - 1, y), (0, 0, 0), 1)
                x = 0
                cv2.line(panel_bgr, (x, 0), (x, grid_h - 1), (0, 0, 0), 1)
                for w in col_w:
                    x += w
                    cv2.line(panel_bgr, (x, 0), (x, grid_h - 1), (0, 0, 0), 1)
            def _x_for_cell_cv(tw: int, col_left: int, col_wid: int):
                if align == "left":  return col_left + padx_cur
                if align == "right": return col_left + max(0, col_wid - tw - padx_cur)
                return col_left + max(0, (col_wid - tw) // 2)
            y_cursor = 0
            for i, r in enumerate(rows):
                x_cursor = 0
                for j, cell in enumerate(r):
                    t = cell if cell else " "
                    (tw, th), base = cv2.getTextSize(t, font_cv, scale_cv, 2)
                    tx = _x_for_cell_cv(tw, x_cursor, col_w[j])
                    ty = y_cursor + max(0, (row_h[i] + th) // 2)
                    try:
                        cv2.putText(panel_bgr, t, (tx, ty), font_cv, scale_cv, line_color, 2, cv2.LINE_AA)
                    except Exception:
                        cv2.putText(panel_bgr, "?", (tx, ty), font_cv, scale_cv, line_color, 2, cv2.LINE_AA)
                    x_cursor += col_w[j]
                y_cursor += row_h[i]

        # ---------- 顶对齐 & 与主图并排 ----------
        Hp, Wp = panel_bgr.shape[:2]
        if Hp < body_avail_h:
            pad = np.full((body_avail_h - Hp, Wp, 3), 255, dtype=np.uint8)
            panel_bgr = np.vstack([panel_bgr, pad])
        elif Hp > body_avail_h:
            panel_bgr = panel_bgr[:body_avail_h, :, :]  # 兜底裁切

        top_pad_img = np.full((panel_top_h, Wp, 3), 255, dtype=np.uint8) if panel_top_h > 0 else None
        column_full = panel_bgr if top_pad_img is None else np.vstack([top_pad_img, panel_bgr])

        gap = np.full((H_img, max(1, gap_px), 3), 255, dtype=np.uint8)
        out = np.concatenate([img_bgr, gap, column_full], axis=1)
        return out


    # -------------------- UI：左侧 -------------------- #
    def _build_left(self):
        # Fixed-width container
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left_container.pack_propagate(False)
        left_container.configure(width=self.LEFT_WIDTH)

        # Canvas + vertical scrollbar
        self.left_canvas = tk.Canvas(
            left_container, bg=self.cget("bg") if hasattr(self, "cget") else "#F0F0F0",
            highlightthickness=0, borderwidth=0, width=self.LEFT_WIDTH
        )
        vbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=vbar.set)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        left = ttk.Frame(self.left_canvas)
        self._left_window_item = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            self.left_canvas.itemconfigure(self._left_window_item, width=self.LEFT_WIDTH)
        left.bind("<Configure>", _on_frame_configure)

        def _on_container_configure(event):
            self.left_canvas.configure(width=self.LEFT_WIDTH)
        left_container.bind("<Configure>", _on_container_configure)

        # Mouse wheel binding
        def _on_mousewheel_win_mac(e):
            step = -1 if e.delta > 0 else 1
            self.left_canvas.yview_scroll(step, "units")
        def _on_wheel_linux_up(e): self.left_canvas.yview_scroll(-1, "units")
        def _on_wheel_linux_down(e): self.left_canvas.yview_scroll(1, "units")
        def _bind_wheel(_):
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel_win_mac)
            self.left_canvas.bind_all("<Button-4>", _on_wheel_linux_up)
            self.left_canvas.bind_all("<Button-5>", _on_wheel_linux_down)
        def _unbind_wheel(_):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")
        self.left_canvas.bind("<Enter>", _bind_wheel)
        self.left_canvas.bind("<Leave>", _unbind_wheel)

        # ---- File ----
        f_file = ttk.LabelFrame(left, text="Open image")
        f_file.pack(fill=tk.X, pady=6)
        ttk.Button(f_file, text="Open image...", command=self.open_image).pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(f_file, text="Reset angle (horizontal)", command=self.reset_angle).pack(fill=tk.X, padx=6, pady=4)

        # ---- Lane detection ----
        f_lane = ttk.LabelFrame(left, text="Lane detection")
        f_lane.pack(fill=tk.X, pady=6)
        self.var_nlanes = tk.IntVar(value=15)
        self._spin(f_lane, "Number of lanes", self.var_nlanes, 1, 40)
        self.var_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(f_lane, text="Auto (with slope)", variable=self.var_mode, value="auto").pack(anchor="w", padx=6)
        ttk.Radiobutton(f_lane, text="Uniform", variable=self.var_mode, value="uniform").pack(anchor="w", padx=6)
        self.var_smooth = tk.IntVar(value=31)
        self.var_sep = tk.DoubleVar(value=1.2)
        self.var_lpad = tk.IntVar(value=0)
        self.var_rpad = tk.IntVar(value=0)

        # ---- Ladder / Axis ----
        f_marker = ttk.LabelFrame(left, text="Ladder / Axis")
        f_marker.pack(fill=tk.X, pady=6)
        self.var_ladder_lane = tk.IntVar(value=1)
        self._spin(f_marker, "Ladder lane index", self.var_ladder_lane, 1, 40)
        ttk.Label(f_marker, text="Ladder MW (kDa, high → low):", wraplength=self.LEFT_WIDTH-24, justify="left")\
            .pack(anchor="w", padx=6)
        self.ent_marker = ttk.Entry(f_marker); self.ent_marker.insert(0, "180,130,95,65,52,41,31,25,17,10")
        self.ent_marker.pack(fill=tk.X, padx=6, pady=2)
        row = ttk.Frame(f_marker); row.pack(fill=tk.X, padx=6, pady=(6,2))
        ttk.Label(row, text="Preset").pack(side=tk.LEFT)
        self.var_preset = getattr(self, "var_preset", tk.StringVar(value=""))
        self.cb_preset = ttk.Combobox(row, textvariable=self.var_preset, state="readonly", width=22)
        self.cb_preset.pack(side=tk.LEFT, padx=(6,6))
        self.cb_preset.bind("<<ComboboxSelected>>", self.on_select_preset)
        ttk.Button(row, text="Manage presets…", command=self.open_preset_manager).pack(side=tk.RIGHT)

        self.var_show_green = tk.BooleanVar(value=False)
        self.var_axis = tk.StringVar(value="left")
        self.var_show_boxes = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f_marker, text="Show intensity box",
            variable=self.var_show_boxes, command=self.on_toggle_show_boxes
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Button(f_marker, text="Reset all shapes", command=self.reset_all_arrows).pack(fill=tk.X, padx=6, pady=4)
        row2 = ttk.Frame(f_marker); row2.pack(fill=tk.X, padx=6, pady=(2,2))
        ttk.Label(row2, text="Arrow shape").pack(side=tk.LEFT)
        self.var_arrow_style = getattr(self, "var_arrow_style", tk.StringVar(value="Arrow (→)"))
        self.cb_arrow_style = ttk.Combobox(
            row2, textvariable=self.var_arrow_style, state="readonly",
            values=["Arrow (→)", "Diagonal ↗", "Star (★)"], width=18
        )
        self.cb_arrow_style.pack(side=tk.RIGHT)
        self.cb_arrow_style.bind("<<ComboboxSelected>>", self.on_change_arrow_style)

        # ---- Actions ----
        f_action = ttk.Frame(left)
        f_action.pack(fill=tk.X, pady=10)
        ttk.Button(f_action, text="Render current gel", command=self.render_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="Export annotated image", command=self.export_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="Export intensities (CSV)", command=self.export_arrow_box_metrics)\
            .pack(fill=tk.X, padx=6, pady=4)

        # ---- Stash / Compose（中文改为英文）----
        f_stash = ttk.LabelFrame(left, text="Stash / Compose")
        f_stash.pack(fill=tk.X, pady=6)
        # >> 为“暂存当前截取”保留句柄（英文）
        self.btn_stash_current = ttk.Button(f_stash, text="Stash current crop", command=self.stash_current_gel)
        self.btn_stash_current.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Button(f_stash, text="Clear stash", command=self.clear_stash).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(f_stash, text="Compose stashed (with table & footnote)", command=self.compose_stash)\
            .pack(fill=tk.X, padx=6, pady=(2, 6))

        # ---- Custom labels ----
        f_lab = ttk.LabelFrame(left, text="Custom labels")
        f_lab.pack(fill=tk.X, pady=6)
        # >> 为“编辑列名 & MWs...”保留句柄
        self.btn_edit_labels = ttk.Button(f_lab, text="Edit lane names & MWs...", command=self.open_labels_editor)
        self.btn_edit_labels.pack(fill=tk.X, padx=6, pady=(6, 6))
        ttk.Button(f_lab, text="Edit right-side table...", command=self._edit_right_table_auto)\
            .pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(f_lab, text="Edit bottom note...", command=self._edit_bottom_note_auto)\
            .pack(fill=tk.X, padx=6, pady=6)

        # ---- White balance / autoscale ----
        f_wb = ttk.LabelFrame(left, text="White balance / Autoscale")
        f_wb.pack(fill=tk.X, pady=6)
        self.var_wb_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_wb, text="Enable white balance", variable=self.var_wb_on).pack(anchor="w", padx=6)
        self.var_wb_exposure = tk.DoubleVar(value=1.0)
        self.var_wb_p_low = tk.DoubleVar(value=0.5)
        self.var_wb_p_high = tk.DoubleVar(value=99.5)
        self.var_wb_per_channel = tk.BooleanVar(value=False)
        self.var_gamma_on = tk.BooleanVar(value=False)
        self.var_gamma_val = tk.DoubleVar(value=1.0)
        self._spin(f_wb, "Exposure (0.5–2.0)", self.var_wb_exposure, 0.5, 2.0, step=0.05)
        self._spin(f_wb, "Low percentile (0–50)", self.var_wb_p_low, 0.0, 50.0, step=0.1)
        self._spin(f_wb, "High percentile (50–100)", self.var_wb_p_high, 50.0, 100.0, step=0.1)

        def _toggle_gamma_state():
            self.sp_gamma.configure(state=("normal" if self.var_gamma_on.get() else "disabled"))
        ttk.Checkbutton(
            f_wb, text="Enable gamma",
            variable=self.var_gamma_on, command=_toggle_gamma_state
        ).pack(anchor="w", padx=6)
        frm_g = ttk.Frame(f_wb); frm_g.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(frm_g, text="gamma (>0)", wraplength=self.LEFT_WIDTH-40, justify="left").pack(side=tk.LEFT)
        self.sp_gamma = ttk.Spinbox(frm_g, textvariable=self.var_gamma_val,
                                    from_=0.2, to=3.0, increment=0.05, width=8, state="disabled")
        self.sp_gamma.pack(side=tk.RIGHT)

        # 刷新按钮状态（拼接预览禁用两按钮）
        try:
            self._refresh_buttons_state()
        except Exception:
            pass


    # -------------------- UI：右侧（显示） -------------------- #

    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.pw_main = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_main.pack(fill=tk.BOTH, expand=True)

        top_grp = ttk.LabelFrame(self.pw_main, text="Gel editor (drag corners & edges; wheel to zoom; right-drag to pan)")
        self.roi_editor = ROIEditorCanvas(top_grp)
        self.roi_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        bottom_container = ttk.Frame(self.pw_main)
        self.pw_bottom = tk.PanedWindow(bottom_container, orient=tk.HORIZONTAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_bottom.pack(fill=tk.BOTH, expand=True)

        self.lbl_roi_wb = ttk.LabelFrame(self.pw_bottom, text="Gel - WB crop")
        self.canvas_roi_wb = tk.Label(self.lbl_roi_wb, bg="#222")
        self.canvas_roi_wb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.lbl_anno = ttk.LabelFrame(self.pw_bottom, text="Annotated result")
        self.canvas_anno = RightAnnoCanvas(self.lbl_anno)
        self.canvas_anno.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.pw_bottom.add(self.lbl_roi_wb, minsize=120)
        self.pw_bottom.add(self.lbl_anno, minsize=120)

        self.pw_main.add(top_grp, minsize=180)
        self.pw_main.add(bottom_container, minsize=160)

        self.after(150, self._init_paned_positions)

        self._bind_autofit(self.canvas_roi_wb)
        self._bind_autofit(self.canvas_anno)

    def open_bottom_note_editor(self):
        """
        打开一个顶层窗口用于多行编辑“底部备注”。
        - 预填已有 self.bottom_note_text；
        - 点击“确定”仅做模块化组合（不重复渲染核心），并刷新右侧；
        - 点击“取消”不保存；
        - 支持“清空后保存”。
        """
        win = tk.Toplevel(self)
        win.title("edit foot note")
        win.transient(self); win.grab_set()
        win.resizable(True, True)

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(frm, text="Note content:", justify="left").pack(anchor="w")
        txt = tk.Text(frm, height=10, width=60, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        # 预填
        try:
            txt.insert("1.0", getattr(self, "bottom_note_text", "") or "")
        except Exception:
            pass

        # 按钮区
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X)

        def do_ok():
            # 保存并快速组合
            try:
                self.bottom_note_text = txt.get("1.0", "end")
            except Exception:
                self.bottom_note_text = ""
            win.destroy()
            try:
                self.recompose_using_cache()
            except Exception:
                # 回退完整渲染
                try: self.render_current()
                except Exception: pass

        def do_clear():
            txt.delete("1.0", "end")

        def do_cancel():
            win.destroy()

        ttk.Button(btns, text="Clear", command=do_clear).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel", command=do_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Confirm", command=do_ok).pack(side=tk.RIGHT, padx=(0,6))

        # 回车=确定，Esc=取消
        #win.bind("<Return>", lambda e: (do_ok(), "break"))
        win.bind("<Escape>", lambda e: (do_cancel(), "break"))


    def on_change_arrow_style(self, *_):
        """
        左侧下拉切换箭头样式 -> 通知右侧画布即时重绘
        """
        txt = (self.var_arrow_style.get() or "").strip().lower()
        if "star" in txt or "★" in txt:
            code = "star"
        elif "diag" in txt or "↗" in txt:
            code = "diag_ne"
        else:
            code = "tri"
        if hasattr(self, "canvas_anno") and self.canvas_anno is not None:
            try:
                self.canvas_anno.set_arrow_style(code)
            except Exception:
                pass

    def _scale_geometry_for_factor(self, bounds, lanes, s: float):
        """
        按比例因子 s 缩放 lanes/bounds：
        - bounds: shape=(Hg, nlanes+1)，仅缩放 X（四舍五入为 int32）
        - lanes : [(l,r), ...]      ，l/r 同比例缩放为 int
        返回 (bounds_s, lanes_s)
        """
        import numpy as np
        bounds_s, lanes_s = None, None
        if isinstance(bounds, np.ndarray) and bounds.ndim == 2:
            try:
                x = (bounds.astype(np.float64) * float(s))
                bounds_s = np.rint(x).astype(np.int32)
            except Exception:
                bounds_s = None
        if isinstance(lanes, list) and lanes:
            try:
                lanes_s = [(int(round(l * s)), int(round(r * s))) for (l, r) in lanes]
            except Exception:
                lanes_s = None
        return bounds_s, lanes_s
    def _attach_fixed_axis_panel(
        self,
        gel_bgr: np.ndarray,
        ladder_peaks_y: list,      # 各刻度对应的 y（相对 gel 顶，单位像素）
        ladder_labels: list,       # 各刻度的文本（float -> 将转为整数）
        yaxis_side: str = "left",
        cap_px: int = None,        # 字高像素，默认 _get_design_params()['axis_cap_px']
        tick_len: int = 8,
        pad_left: int = 10,
        pad_right: int = 10,
        pad_text_gap: int = 6,
    ) -> tuple[np.ndarray, int]:
        """
        在 gel_bgr 左/右侧追加固定像素字号的白色面板，绘制刻度短线与文本（文本强制整数，无小数）。
        返回 (合成图, 面板宽 panel_w)。
        """
        import numpy as np, cv2
        if gel_bgr is None or not isinstance(gel_bgr, np.ndarray) or gel_bgr.size == 0:
            return gel_bgr, 0
        H, W = gel_bgr.shape[:2]
        design = self._get_design_params()
        cap_px = int(cap_px) if isinstance(cap_px, (int, float)) and cap_px else int(design.get("axis_cap_px", 16))
        # 字体度量
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        scale = self._font_scale_for_cap_height_px(cap_px, font=font, thickness=thickness)

        # 统一转整数文本（去小数）
        def _fmt_int(v) -> str:
            try:
                return str(int(round(float(v))))
            except Exception:
                return str(v)

        labels = [_fmt_int(x) for x in (ladder_labels or [])]
        ys = [int(round(float(y))) for y in (ladder_peaks_y or []) if isinstance(y, (int, float))]
        if len(labels) != len(ys):
            n = min(len(labels), len(ys))
            labels, ys = labels[:n], ys[:n]

        # 计算最大文本宽度
        max_w = 0
        sizes = []
        for t in labels:
            (tw, th), base = cv2.getTextSize(t if t else " ", font, scale, thickness)
            sizes.append((tw, th))
            max_w = max(max_w, tw)

        panel_w = pad_left + tick_len + pad_text_gap + max_w + pad_right
        panel = np.full((H, panel_w, 3), 255, dtype=np.uint8)

        # 绘制
        side = str(yaxis_side or "left").lower()
        for i, t in enumerate(labels):
            y = int(np.clip(ys[i], 0, H - 1))
            (tw, th) = sizes[i]
            if side == "right":
                x0 = 0
                x1 = tick_len
                tx = x1 + pad_text_gap
            else:
                x1 = panel_w - 1
                x0 = x1 - tick_len
                tx = x0 - pad_text_gap - tw
                tx = max(0, tx)
            cv2.line(panel, (x0, y), (x1, y), (0, 0, 0), 1, cv2.LINE_AA)
            ty = int(np.clip(y + th // 2, th, H - 1))
            cv2.putText(panel, t if t else " ", (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        out = np.concatenate([gel_bgr, panel], axis=1) if side == "right" else np.concatenate([panel, gel_bgr], axis=1)
        return out, int(panel_w)

    def _set_composed_preview(self, base_img: np.ndarray, display_img: np.ndarray):
        """
        进入“拼接整体预览”模式：
        - base_img：未附加（或仅按当前值附加）的拼接底图（用于后续再次叠加表格/底注）
        - display_img：当前显示的拼接图（右侧画布底图）
        """
        setattr(self, "_is_composed_preview", True)
        setattr(self, "compose_preview", {"base": base_img, "display": display_img})
        # >>> 进入拼接预览后，自动禁用“编辑列名 & MWs…”与“暂存当前截取”
        try:
            self._refresh_buttons_state()
        except Exception:
            pass

    def _unset_composed_preview(self):
        """
        退出“拼接整体预览”模式。
        """
        setattr(self, "_is_composed_preview", False)
        setattr(self, "compose_preview", {"base": None, "display": None})
        # >>> 退出拼接预览后，恢复按钮可用
        try:
            self._refresh_buttons_state()
        except Exception:
            pass
    def _refresh_buttons_state(self):
        """
        根据是否处于“拼接整体预览”状态，统一启/禁用左侧的部分按钮：
        - 处于预览：禁用 “Edit lane names & MWs...” 与 “暂存当前截取”
        - 非预览：恢复上述按钮为 normal
        """
        in_mosaic = bool(getattr(self, "_is_composed_preview", False))
        state = "disabled" if in_mosaic else "normal"

        # “编辑列名 & MWs...”
        try:
            if hasattr(self, "btn_edit_labels") and self.btn_edit_labels is not None:
                self.btn_edit_labels.configure(state=state)
        except Exception:
            pass

        # “暂存当前截取”
        try:
            if hasattr(self, "btn_stash_current") and self.btn_stash_current is not None:
                self.btn_stash_current.configure(state=state)
        except Exception:
            pass


    def _apply_mosaic_annotations(self, mosaic_base: np.ndarray) -> np.ndarray:
        """
        在拼接后的“基础图”上，按【拼接专用】right_table/bottom_note 进行叠加，
        返回可直接用于显示/导出的整图（不改箭头、无重算）。
        """
        self._ensure_mosaic_vars()
        if mosaic_base is None or not isinstance(mosaic_base, np.ndarray) or mosaic_base.size == 0:
            return mosaic_base

        out = mosaic_base

        # —— 右侧表格（拼接专用变量） ——
        rtxt = (self.mosaic_right_table_text or "").strip()
        if rtxt:
            cfg = (self.mosaic_right_table_opts or {})
            out = self._attach_right_table_panel(
                out,
                rtxt,
                has_header=bool(cfg.get("has_header", True)),
                show_grid=bool(cfg.get("grid", True)),
                align=str(cfg.get("align", "center")),
                cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                panel_top_h=0,
                gel_height_px=out.shape[0],  # 以整图高度为参考，保证收缩适配
            )

        # —— 底部备注（拼接专用变量） ——
        note_raw = self.mosaic_bottom_note_text or ""
        if note_raw.strip():
            out = self._attach_bottom_note_panel(out, note_raw, allow_expand_width=True)

        return out



    def stash_current_gel(self):
        """
        暂存当前截取的胶图（对齐策略：仅在“检测条带数 == 标准条带数(>=2)”时才参与对齐）。
        """
        from tkinter import messagebox
        import numpy as np
        rc = getattr(self, "render_cache", None) or {}
        gel_bgr = rc.get("gel_bgr", None)
        if gel_bgr is None or not hasattr(gel_bgr, "shape"):
            messagebox.showwarning("Notice", "No rendered gel available. Please render the current crop first.")
            return

        H, W = gel_bgr.shape[:2]
        nlanes = int(self.var_nlanes.get())
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
        yaxis_side = str(self.var_axis.get() or "left").lower()

        def _parse_list(s: str):
            s = (s or "").replace("，", ",")
            out = []
            for t in s.split(","):
                t = t.strip()
                if not t:
                    continue
                try:
                    out.append(float(t))
                except Exception:
                    pass
            return out

        # 标准序列（仅用于“相等数量判断”与侧标绘制；不用来对齐）
        tick_labels = rc.get("tick_labels", None)
        if not tick_labels:
            tick_labels = _parse_list(self.ent_marker.get()) or [180,130,95,65,52,41,31,25,17,10]
        tick_labels = [float(x) for x in tick_labels if np.isfinite(x) and x > 0]
        expected_count = len(tick_labels)

        bounds = rc.get("bounds", None)
        lanes = rc.get("lanes", None)

        # === 实际检测条带（仅用检测值；不使用拟合） ===
        y_top, y_bot = None, None
        peaks_list = []
        try:
            gray = cv2.cvtColor(gel_bgr, cv2.COLOR_BGR2GRAY)
            if isinstance(bounds, np.ndarray):
                peaks, prom = detect_bands_along_y_slanted(
                    gray, bounds, lane_index=ladder_lane-1,
                    y0=0, y1=None, min_distance=30, min_prominence=1,
                )
            elif isinstance(lanes, list) and lanes:
                l, r = lanes[ladder_lane-1]
                sub = gray[:, l:r]
                peaks, prom = detect_bands_along_y_prominence(
                    sub, y0=0, y1=None, min_distance=30, min_prominence=1
                )
            else:
                peaks, prom = [], []
            peaks = [float(p) for p in (peaks or []) if np.isfinite(p) and 0 <= float(p) <= H-1]
            peaks_list = [int(round(p)) for p in peaks]
        except Exception:
            peaks_list = []

        detected_count = len(peaks_list)
        # —— 仅当“检测数量 == 标准数量（且标准≥2）”才参与对齐 —— #
        align_valid = (expected_count >= 2) and (detected_count == expected_count)
        if align_valid:
            y_top = float(min(peaks_list))
            y_bot = float(max(peaks_list))
        else:
            y_top, y_bot = None, None

        # —— lane_names / lane_marks / 几何 —— #
        lane_names_snapshot = list(self.lane_names or [])
        lane_marks_snapshot = [list(x or []) for x in (self.lane_marks or [])]
        bounds_snapshot = None
        if isinstance(bounds, np.ndarray):
            try:
                bounds_snapshot = bounds.astype(np.int32).copy()
            except Exception:
                bounds_snapshot = bounds.copy()
        lanes_snapshot = None
        if isinstance(lanes, list):
            try:
                lanes_snapshot = [(int(l), int(r)) for (l, r) in lanes]
            except Exception:
                lanes_snapshot = lanes

        # —— 右侧画布箭头 —— #
        try:
            arrows_manual = self._snapshot_arrows_from_canvas()
        except Exception:
            arrows_manual = []

        item = {
            "gel_bgr": gel_bgr.copy(),
            "meta": {
                "nlanes": int(nlanes),
                "ladder_lane": int(ladder_lane),
                "yaxis_side": yaxis_side,
                "tick_labels": tick_labels,  # 仅用于显示（侧标）；不参与对齐判断
                "calib_model": rc.get("calib_model", None),  # 保留但不用于对齐
                "fit_ok": bool(rc.get("fit_ok", False)),
                "mode": ("slanted" if isinstance(bounds, np.ndarray) else ("uniform" if lanes else "unknown")),
                "bounds": bounds_snapshot,
                "lanes": lanes_snapshot,
                "lane_names": lane_names_snapshot,
                "lane_marks": lane_marks_snapshot,
                "arrows_manual": arrows_manual,
            },
            "align": {
                "policy": "exact_match",  # 规则：检测数必须等于标准数
                "expected_count": int(expected_count),
                "detected_count": int(detected_count),
                "valid": bool(align_valid),  # 仅在“数量相等且≥2”才 True
                "peaks_y": [int(p) for p in peaks_list],
                "y_top": (float(y_top) if align_valid else None),
                "y_bot": (float(y_bot) if align_valid else None),
            }
        }
        self.stash.append(item)
        from tkinter import messagebox
        messagebox.showinfo("Complete", f"Stashed 1 image (total {len(self.stash)}).")


    def clear_stash(self):
        """清空暂存队列（内存）。"""
        from tkinter import messagebox
        try:
            self._soft_reset_for_operation()
        except Exception:
            pass
        self.stash.clear()
        messagebox.showinfo("Complete", "Cleared the stash.")

    def compose_stash(self):
        """
        拼接已暂存（新的“基准锚定”策略）：

        规则（满足你的新要求）：
        1) 选定一张“基准图”（ref）：优先选择“检测数量==标准数量（且标准≥2）”的第一张；
        若没有可对齐的，则用暂存队列中的第 1 张作为基准。
        2) 基准图 **不缩放**（s_ref = 1.0），整体拼接高度 H_final = 基准图原始高度 H0_ref。
        3) 对“可对齐”的其他图片 i：
        - 计算其标准道峰值跨度 span_i 与基准的 span_ref；
        - 等比缩放因子 s_i = span_ref / span_i；
        - 缩放后按“上沿峰值 y_top”与基准“上沿峰值 y_top_ref”对齐（仅上下补白，不再改变尺寸）；
        - 最终上下补白/裁剪，使整块高度 = H0_ref。
        4) 对“不可对齐”的图片（invalid）：
        - 不做峰值对齐，直接等比缩放到高度 H0_ref（宽度按比例变化），顶对齐（不额外补白）。
        5) 横向拼接时，左侧统一追加“侧边轴面板”（取基准图来源），顶部统一追加“全局顶栏泳道标注”；
        右侧表格/底部备注为“绝对像素叠加”，并将表格顶部对齐到“顶栏下沿”。

        好处：
        - 不再出现“整体高度被少数图块拉大很多”的情况，最终高度稳定 = 基准图高度；
        - ROI 很窄时，若仍检测到完整峰值，仍可按基准图准确对齐（缩放因子来自基准跨度）。
        """
        from tkinter import messagebox
        import numpy as np, cv2

        if not getattr(self, "stash", None):
            messagebox.showwarning("Notice", "Nothing stashed yet. Please 'Stash current crop' first.")
            return
        if len(self.stash) < 1:
            messagebox.showwarning("Notice", "Not enough items in stash (at least 1 required).")
            return

        items = self.stash

        # --- 0) 收集原始高度/对齐信息 ---
        H0_list, spans, ytop, ybot, valid_mask = [], [], [], [], []
        for it in items:
            H0, W0 = it["gel_bgr"].shape[:2]
            H0_list.append((H0, W0))
            a = it.get("align", {}) or {}
            ok = bool(a.get("valid", False)) and (a.get("y_top") is not None) and (a.get("y_bot") is not None) \
                and (float(a["y_bot"]) > float(a["y_top"]))
            if ok:
                valid_mask.append(True)
                spans.append(float(a["y_bot"]) - float(a["y_top"]))
                ytop.append(float(a["y_top"]))
                ybot.append(float(a["y_bot"]))
            else:
                valid_mask.append(False)
                spans.append(None)
                ytop.append(None)
                ybot.append(None)

        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]

        # --- 1) 选择“基准图 ref_idx” ---
        if valid_indices:
            ref_idx = valid_indices[0]  # 第一张可对齐的，作为基准
        else:
            ref_idx = 0  # 没有可对齐的则用第一张
        ref_item = items[ref_idx]
        H0_ref, W0_ref = H0_list[ref_idx]
        # 基准图不缩放
        s_ref = 1.0
        # 基准图的“峰值对齐参数”（若不可对齐则用整幅）
        if valid_mask[ref_idx]:
            span_ref = float(spans[ref_idx])
            ytop_ref = float(ytop[ref_idx])
            ybot_ref = float(ybot[ref_idx])
        else:
            span_ref = float(H0_ref - 1)
            ytop_ref = 0.0
            ybot_ref = float(H0_ref - 1)

        # --- 2) 第一轮：各块主缩放 ---
        scaled_blocks, per_item_scale = [], []
        for idx, it in enumerate(items):
            gel = it["gel_bgr"]
            H0, W0 = H0_list[idx]

            if idx == ref_idx:
                # 基准图不缩放
                img_scaled = gel
                si = 1.0
                # 兼容：若基准图可对齐，记录对齐后的“在自身坐标系”的 y_top1/y_bot1
                if valid_mask[idx]:
                    y_top1 = ytop[idx] * si
                    y_bot1 = ybot[idx] * si
                else:
                    y_top1 = 0.0
                    y_bot1 = float(H0 - 1)
            else:
                if valid_mask[idx] and span_ref > 1e-6 and spans[idx] and spans[idx] > 1e-6:
                    # 可对齐：按“基准跨度”缩放
                    si = float(span_ref / float(spans[idx]))
                    new_w = max(1, int(round(W0 * si)))
                    new_h = max(1, int(round(H0 * si)))
                    interp = cv2.INTER_AREA if si < 1.0 else cv2.INTER_LINEAR
                    img_scaled = cv2.resize(gel, (new_w, new_h), interpolation=interp)
                    # 映射后的峰值位置（用于之后与基准的上沿对齐）
                    y_top1 = float(ytop[idx]) * si
                    y_bot1 = float(ybot[idx]) * si
                else:
                    # 不可对齐：直接把整块高度缩放到与基准高度一致（不做峰值对齐）
                    si = float(H0_ref) / max(1e-6, float(H0))
                    new_w = max(1, int(round(W0 * si)))
                    new_h = H0_ref  # 直接等于基准高度
                    interp = cv2.INTER_AREA if si < 1.0 else cv2.INTER_LINEAR
                    img_scaled = cv2.resize(gel, (new_w, new_h), interpolation=interp)
                    y_top1 = 0.0
                    y_bot1 = float(new_h - 1)

            scaled_blocks.append({
                "img": img_scaled,
                "H": int(img_scaled.shape[0]),
                "W": int(img_scaled.shape[1]),
                "s": float(si),
                "y_top1": float(y_top1),
                "y_bot1": float(y_bot1),
                # 下游用到的元数据
                "panel_top_h": 0,  # 单块不再加顶栏
                "nlanes": int(it["meta"].get("nlanes", 0)),
                "ladder_lane": int(it["meta"].get("ladder_lane", 1)),
                "yaxis_side": str(it["meta"].get("yaxis_side", "left")).lower(),
                "lane_names": it["meta"].get("lane_names", []) or [],
                "arrows_manual": it["meta"].get("arrows_manual", []) or [],
                "calib_model": it["meta"].get("calib_model", None),
                "tick_labels": it["meta"].get("tick_labels", []) or [],
            })
            # 缩放 lanes/bounds（供箭头/顶栏使用）
            bounds_i = it["meta"].get("bounds", None)
            lanes_i  = it["meta"].get("lanes",  None)
            bounds_s, lanes_s = self._scale_geometry_for_factor(bounds_i, lanes_i, scaled_blocks[-1]["s"])
            scaled_blocks[-1]["bounds_s"] = bounds_s
            scaled_blocks[-1]["lanes_s"]  = lanes_s

            per_item_scale.append(float(si))

        # --- 3) 第二轮：按基准“上沿 y_top_ref”对齐，并统一高度为 H0_ref ---
        def _fit_to_height(img, target_h):
            """底/裁，确保高度=target_h（白色补底）"""
            if img is None or img.size == 0:
                return img
            h, w = img.shape[:2]
            if h == target_h:
                return img
            if h < target_h:
                pad = target_h - h
                return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            return img[:target_h, :, :]

        # 基准图的对齐参数
        y_top_ref_total = float(ytop[ref_idx] * s_ref) if valid_mask[ref_idx] else 0.0

        for i, b in enumerate(scaled_blocks):
            img = b["img"]
            H = b["H"]

            if i == ref_idx:
                # 基准图：无需额外 pad，最终高度就是 H0_ref
                img2 = _fit_to_height(img, H0_ref)
                pad_top = 0
                pad_bot = max(0, H0_ref - img2.shape[0])
                s_post = 1.0
            else:
                if valid_mask[i] and valid_mask[ref_idx]:
                    # 有效对齐：按上沿对齐（只做上下补白，不改变内容尺寸）
                    # 我们希望：pad_top + y_top1 == y_top_ref_total
                    pad_top = int(np.ceil(y_top_ref_total - b["y_top1"]))
                    pad_top = max(0, pad_top)
                    # 临时补白到 >= y_top_ref_total 的顶部对齐高度，再在底部补白/裁切到 H0_ref
                    if pad_top > 0:
                        img = cv2.copyMakeBorder(img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    # 统一目标高度
                    img2 = _fit_to_height(img, H0_ref)
                    pad_bot = max(0, H0_ref - img2.shape[0])
                    s_post = 1.0
                else:
                    # 不可对齐：第一轮已等比缩放到 H0_ref，这里只保证高度=H0_ref
                    img2 = _fit_to_height(img, H0_ref)
                    pad_top = 0
                    pad_bot = max(0, H0_ref - img2.shape[0])
                    s_post = 1.0

            b.update({
                "img": img2,
                "H": int(img2.shape[0]),
                "W": int(img2.shape[1]),
                "pad_top": int(pad_top),
                "pad_bot": int(pad_bot),
                "s_post": float(s_post),
                # 便于箭头恢复
                "y_top_total": float(b["y_top1"] + pad_top),
                "y_bot_total": float(b["y_bot1"] + pad_top),
            })

        # --- 4) 横向拼接（统一高度 = H0_ref） ---
        gap = 30
        columns, x_offsets, x_cursor = [], [], 0
        for i, b in enumerate(scaled_blocks):
            if i > 0:
                columns.append(np.full((H0_ref, gap, 3), 255, dtype=np.uint8))
                x_cursor += gap
            columns.append(b["img"])
            x_offsets.append(x_cursor)
            x_cursor += b["W"]

        mosaic_base = np.concatenate(columns, axis=1) if columns else None
        if mosaic_base is None:
            messagebox.showerror("Error", "Compose failed: no valid image.")
            return

        # --- 5) 左侧轴面板（取基准图来源，绝对像素） ---
        axis_w_global = 0
        try:
            b_ref       = scaled_blocks[ref_idx]
            si_ref      = float(b_ref["s"])
            s_post_ref  = float(b_ref.get("s_post", 1.0))
            pad_top_ref = int(b_ref.get("pad_top", 0))
            panel_top_ref = int(b_ref.get("panel_top_h", 0))
            tick_labels_ref = items[ref_idx]["meta"].get("tick_labels", []) or b_ref.get("tick_labels", []) or []

            # 若有基准的 piecewise 模型且有 tick_labels，用模型定位刻度；否则退化为用上下沿两点
            try:
                from gel_core import predict_y_from_mw_piecewise
            except Exception:
                predict_y_from_mw_piecewise = None

            ladder_peaks_mosaic, ladder_labels_use = [], []
            if items[ref_idx]["meta"].get("calib_model", None) and tick_labels_ref and predict_y_from_mw_piecewise:
                cm = items[ref_idx]["meta"]["calib_model"]
                xk = np.asarray(cm["xk"]); yk = np.asarray(cm["yk"])
                ys_gel = predict_y_from_mw_piecewise(tick_labels_ref, xk, yk)
                for y_gel, lab in zip(ys_gel, tick_labels_ref):
                    y_img = (pad_top_ref + panel_top_ref + float(y_gel) * si_ref) * s_post_ref
                    ladder_peaks_mosaic.append(int(round(y_img)))
                    ladder_labels_use.append(lab)
            else:
                # 退化：用上沿与下沿两个刻度（若基准不可对齐，则用 0 与 H0_ref-1，并给空标签）
                if valid_mask[ref_idx]:
                    y_top_img = (pad_top_ref + panel_top_ref + float(ytop[ref_idx]) * si_ref) * s_post_ref
                    y_bot_img = (pad_top_ref + panel_top_ref + float(ybot[ref_idx]) * si_ref) * s_post_ref
                else:
                    y_top_img = 0.0
                    y_bot_img = float(H0_ref - 1)
                ladder_peaks_mosaic = [int(round(y_top_img)), int(round(y_bot_img))]
                if tick_labels_ref and len(tick_labels_ref) >= 2:
                    ladder_labels_use = [tick_labels_ref[0], tick_labels_ref[-1]]
                else:
                    ladder_labels_use = ["", ""]

            mosaic_with_axis, axis_w = self._attach_fixed_axis_panel(
                gel_bgr=mosaic_base,
                ladder_peaks_y=ladder_peaks_mosaic,
                ladder_labels=ladder_labels_use,
                yaxis_side="left",
                cap_px=self._get_design_params().get("axis_cap_px", 16)
            )
            mosaic_base = mosaic_with_axis
            axis_w_global = int(axis_w or 0)
        except Exception:
            axis_w_global = 0

        # --- 6) 全局顶栏（与单图一致，绝对像素） ---
        top_panel_h_global = 0
        try:
            mosaic_base, top_panel_h_global = self._attach_lane_labels_global(
                mosaic_base,
                scaled_blocks=scaled_blocks,
                x_offsets=x_offsets,
                axis_w_left=axis_w_global,
                top_cap_px=int(self._get_design_params().get("top_cap_px", 26))
            )
        except Exception:
            top_panel_h_global = 0

        # --- 7) 右表/底注叠加（绝对像素；表格顶部对齐到“顶栏下沿”） ---
        mosaic_display = self._apply_mosaic_annotations(
            mosaic_base,
            absolute_pixels=True,
            panel_top_h_override=int(top_panel_h_global)
        )

        # --- 8) 刷新右侧画布（不重置箭头），并恢复箭头 ---
        if hasattr(self, "canvas_anno") and self.canvas_anno is not None:
            try:
                self.canvas_anno.set_scene(
                    base_img_bgr=mosaic_display,
                    gel_bgr=np.zeros_like(mosaic_display),
                    bounds=None, lanes=None,
                    a=0.0, b=0.0, fit_ok=True,
                    nlanes=0, ladder_lane=1,
                    yaxis_side="left",
                    lane_marks=[],
                    panel_top_h=0, panel_w=0,
                    calib_model=None,
                    reset_positions=False
                )
            except Exception:
                pass

        # 恢复箭头（考虑左轴宽度 + 顶栏高度；每块缩放 s 和 s_post）
        try:
            from gel_core import predict_y_from_mw_piecewise
        except Exception:
            predict_y_from_mw_piecewise = None

        if hasattr(self, "canvas_anno") and self.canvas_anno is not None:
            can = self.canvas_anno
            Hm, Wm = mosaic_display.shape[:2]

            for i, it in enumerate(items):
                b       = scaled_blocks[i]
                si      = float(b["s"])
                s_post  = float(b.get("s_post", 1.0))
                x_left  = int(x_offsets[i])
                pad_top = int(b.get("pad_top", 0))
                panel_top = int(b.get("panel_top_h", 0))
                bounds_s = b["bounds_s"]; lanes_s = b["lanes_s"]
                nlanes_i = int(b["nlanes"]); ladder_lane_i = int(b["ladder_lane"])

                # 真实泳道数
                if isinstance(bounds_s, np.ndarray):
                    real_n = max(0, bounds_s.shape[1] - 1)
                elif isinstance(lanes_s, list) and lanes_s:
                    real_n = len(lanes_s)
                else:
                    real_n = nlanes_i

                # 1) 有手工箭头 snapshot 则优先恢复（绝对位置）
                manual = it["meta"].get("arrows_manual", []) or b.get("arrows_manual", []) or []
                if manual:
                    for ar in manual:
                        try:
                            li = int(ar.get("lane_idx", -1))
                            mw = float(ar.get("mw", float("nan")))
                            x_gel = float(ar.get("x_gel", float("nan")))
                            y_gel = float(ar.get("y_gel", float("nan")))
                            if li < 0 or not (np.isfinite(mw) and mw > 0):
                                continue
                            if not (np.isfinite(x_gel) and np.isfinite(y_gel)):
                                continue
                            x_img = float(axis_w_global + x_left + (x_gel * si) * s_post)
                            y_img = float(top_panel_h_global + (pad_top + panel_top + y_gel * si) * s_post)
                            x_img = float(np.clip(x_img, 0, Wm - 1))
                            y_img = float(np.clip(y_img, 0, Hm - 1))
                            can._add_arrow(x_img, y_img, lane_idx=li, mw=mw, moved=True)
                        except Exception:
                            continue
                    continue  # 下一块

                # 2) 否则按模型初位（若有）
                lane_marks_i = it["meta"].get("lane_marks", []) or []
                cm = it["meta"].get("calib_model", None) if predict_y_from_mw_piecewise else None
                skip_idx = max(0, ladder_lane_i - 1)
                nonladder_idx = [li for li in range(real_n) if li != skip_idx]
                use_k = min(len(nonladder_idx), len(lane_marks_i))
                for k in range(use_k):
                    li = nonladder_idx[k]
                    for mw in (lane_marks_i[k] or []):
                        try:
                            v = float(mw)
                            if not (np.isfinite(v) and v > 0):
                                continue
                            if cm and "xk" in cm and "yk" in cm and predict_y_from_mw_piecewise:
                                y_gel = float(predict_y_from_mw_piecewise(
                                    [v], np.asarray(cm["xk"]), np.asarray(cm["yk"])
                                )[0])
                            else:
                                continue
                            y_img = float(top_panel_h_global + (pad_top + panel_top + (y_gel * si)) * s_post)

                            # x：取左分界线（bounds_s/lanes_s 已按 si 缩放；再乘 s_post）
                            if isinstance(bounds_s, np.ndarray) and bounds_s.ndim == 2 and bounds_s.shape[0] > 0:
                                y_row_pre = int(np.clip(round(y_gel * si), 0, bounds_s.shape[0] - 1))
                                xl = int(bounds_s[y_row_pre, li])
                            elif isinstance(lanes_s, list) and lanes_s and 0 <= li < len(lanes_s):
                                xl = int(lanes_s[li][0])
                            else:
                                Wg_si = int(round(it["gel_bgr"].shape[1] * si))
                                step = Wg_si / max(1, real_n)
                                xl = int(round(li * step))
                            x_img = float(axis_w_global + x_left + (xl * s_post))
                            x_img = float(np.clip(x_img, 0, Wm - 1))
                            y_img = float(np.clip(y_img, 0, Hm - 1))
                            can._add_arrow(x_img, y_img, lane_idx=li, mw=v, moved=True)
                        except Exception:
                            continue

            try:
                can._redraw_arrows()
            except Exception:
                pass
            try:
                if bool(self.var_show_boxes.get()):
                    can.set_boxes_enabled(True)
            except Exception:
                pass

        # --- 9) 缓存“用于注解的基图”和“顶栏高度”（供轻量刷新使用） ---
        compose_items = []
        for i, it in enumerate(items):
            compose_items.append({
                "meta": dict(it.get("meta", {})),
                "align": dict(it.get("align", {})),
                "scale_s": float(per_item_scale[i]),
                "pad_top": int(scaled_blocks[i].get("pad_top", 0)),
                "pad_bottom": int(scaled_blocks[i].get("pad_bot", 0)),
            })

        if not hasattr(self, "render_cache") or self.render_cache is None:
            self.render_cache = {}

        self.mosaic_base_last = mosaic_base.copy()
        self.mosaic_top_panel_h_last = int(top_panel_h_global)
        self.render_cache["compose_meta"] = {
            "version": 14,  # 版本+1：锚定基准图策略
            "anchor_policy": "fixed_baseline",  # 记录策略
            "ref_index": int(ref_idx),
            "left_axis_w": int(axis_w_global),
            "top_panel_h": int(top_panel_h_global),
            "H_final": int(H0_ref),
            "items": compose_items,
        }

        self._set_composed_preview(base_img=mosaic_base, display_img=mosaic_display)
        
    
    
    def _attach_lane_labels_global(
        self,
        mosaic_bgr: np.ndarray,
        scaled_blocks: list,
        x_offsets: list,
        axis_w_left: int = 0,
        # ↓ 这几个是“绝对像素顶栏”的基础垫距，但字号/线宽/收缩等规则完全复用单图风格
        top_cap_px: int = 26,
        top_pad_px: int = 8,
        bottom_pad_px: int = 10,
        txt_pad_px: int = 3
    ) -> tuple[np.ndarray, int]:
        """
        在整张拼接图的“最顶部”一次性追加“泳道标注顶栏”（竖排、风格与单图一致）。
        关键点：
        - 对每个块，按单图规则计算：s=max(0.35, Hg_eff/1000)，thick≈2*s，cap_px≈26*s；
        若旋转后宽度将超过该道宽，则按单图逻辑等比收缩（不改变厚度）。
        - 文本小图按“横排绘制→逆时针旋转90°→轻裁边”流程，风格与单图一致。
        - 全局顶栏统一“基线对齐”：所有标签的底沿落在同一基线（与单图每块的“行底沿”视觉一致）。
        - 标准道（ladder_lane）若该行有任一文本，则绘制 "Marker"（黑色）。
        返回：(叠加后的整图, 顶栏高度 top_panel_h)。
        """
        import numpy as np, cv2

        if mosaic_bgr is None or mosaic_bgr.size == 0:
            return mosaic_bgr, 0

        H, W_total = mosaic_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # === 1) 收集每块的道中心与道宽（以“块内部坐标”为准），用于“按列宽收缩” ===
        centers_per_block: list[list[int]] = []
        widths_per_block:  list[list[int]] = []
        for b in scaled_blocks:
            bounds_s = b["bounds_s"]; lanes_s = b["lanes_s"]
            centers_mid, widths = [], []
            if isinstance(bounds_s, np.ndarray) and bounds_s.ndim == 2 and bounds_s.shape[0] > 0:
                yc = min(bounds_s.shape[0]-1, max(0, b["H"]//2))
                real_n = max(0, bounds_s.shape[1]-1)
                for i in range(real_n):
                    L = int(bounds_s[yc, i]); R = int(bounds_s[yc, i+1])
                    centers_mid.append(int(round((L+R)/2.0)))
                    widths.append(max(1, R-L))
            elif isinstance(lanes_s, list) and lanes_s:
                for (L, R) in lanes_s:
                    L, R = int(L), int(R)
                    centers_mid.append(int(round((L+R)/2.0)))
                    widths.append(max(1, R-L))
            else:
                # 兜底：按块总宽等分（极少进入）
                Wg = int(round(b["W"]))
                real_n = max(1, int(b.get("nlanes", 1)))
                step = max(1.0, Wg / real_n)
                for i in range(real_n):
                    L = int(round(i*step)); R = int(round((i+1)*step))
                    centers_mid.append(int(round((L+R)/2.0)))
                    widths.append(max(1, R-L))
            centers_per_block.append(centers_mid)
            widths_per_block.append(widths)

        # === 2) 预渲染：逐块逐道生成“旋转文本小图”，完全复用单图风格的度量与收缩规则 ===
        # 保存条目：(bi, x_center_mosaic:int, rot_img, w_rot, h_rot, top_pad_i, bottom_pad_i, is_marker:bool)
        rot_items = []
        max_h_rot_across = 0
        max_top_pad_across = 0
        max_bottom_pad_across = 0

        for bi, b in enumerate(scaled_blocks):
            # 单图风格：以“有效胶高”计算 s（等效于单图：s = Hg / 1000）
            # 这里的有效胶高：纯胶体高度（缩放后）再乘以二次缩放 s_post
            body_h_eff = float(b.get("body_h", b["H"])) * float(b.get("s_post", 1.0))
            s = max(0.35, body_h_eff / 1000.0)

            thick_i  = max(1, int(round(2 * s)))
            cap_px_i = max(10, int(round(26 * s)))
            scale_i  = self._font_scale_for_cap_height_px(cap_px_i, font=font, thickness=thick_i)

            # 单图风格：顶/底垫距随 s 缩放
            top_pad_i    = max(4, int(round(8  * s)))
            bottom_pad_i = max(4, int(round(10 * s)))

            centers_mid = centers_per_block[bi]
            widths_mid  = widths_per_block[bi]
            si_post     = float(b.get("s_post", 1.0))
            x_left      = int(x_offsets[bi])
            names       = (b.get("lane_names") or [])
            nlanes_i    = int(b["nlanes"])
            ladder_lane = max(1, int(b["ladder_lane"]))
            skip_idx    = ladder_lane - 1

            # 非标准道（真实道索引）：names 与非标准道一一对应
            nonladder_idx = [i for i in range(len(centers_mid)) if i != skip_idx]
            use_k = min(len(nonladder_idx), len(names))
            any_text = any((str(n or "").strip() for n in names))

            # 工具：生成“竖排小图”（横排绘制→旋转90°→轻裁边），并在“超过列宽”时按单图规则收缩
            def render_vertical_text_fitting(text: str, lane_w_mosaic: int):
                t = (text or "").strip()
                if not t:
                    return None, 0, 0

                # 先用“初始 scale_i”度量横排的 tw/th
                (tw, th), base = cv2.getTextSize(t, font, scale_i, thick_i)

                # 单图规则：如果“旋转后的宽度≈横排的高度 th” 会超过道宽，则按比例减小 scale
                # （注意：厚度 thick_i 不随缩放变化 —— 与单图保持一致）
                allow_w = max(1, lane_w_mosaic - 2*txt_pad_px)
                scale_fit = float(scale_i)
                if th > allow_w:
                    scale_fit = max(0.35, scale_i * (allow_w / (th + 1e-6)))
                    (tw, th), base = cv2.getTextSize(t, font, scale_fit, thick_i)

                # 绘制横排小图（边距 txt_pad_px）
                w_horiz = max(1, tw + 2*txt_pad_px)
                h_horiz = max(1, th + base + 2*txt_pad_px)
                img = np.full((h_horiz, w_horiz, 3), 255, dtype=np.uint8)
                org = (txt_pad_px, txt_pad_px + th)
                try:
                    cv2.putText(img, t, org, font, scale_fit, (0, 0, 0), thick_i, cv2.LINE_AA)
                except Exception:
                    cv2.putText(img, "?", org, font, scale_fit, (0, 0, 0), thick_i, cv2.LINE_AA)

                # 逆时针旋转 90°，并轻裁边（与单图一致）
                rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
                ys, xs = np.where(gray < 252)
                if ys.size > 0:
                    y1 = max(0, ys.min() - 1); y2 = min(rot.shape[0]-1, ys.max() + 1)
                    x1 = max(0, xs.min() - 1); x2 = min(rot.shape[1]-1, xs.max() + 1)
                    rot = rot[y1:y2+1, x1:x2+1]
                h_rot, w_rot = rot.shape[:2]
                return rot, w_rot, h_rot

            # 渲染非标准道标签
            for k in range(use_k):
                li = nonladder_idx[k]
                text = str(names[k] or "").strip()
                if not text:
                    continue

                # 该道的“拼接图中的宽度”：道宽 * s_post
                lane_w_mosaic = int(round(widths_mid[li] * si_post))
                rot_img, w_rot, h_rot = render_vertical_text_fitting(text, lane_w_mosaic)
                if rot_img is None:
                    continue

                # 道中心在“拼接图中的X”：左轴偏移 + 块x偏移 + center * s_post
                x_center_mosaic = int(round(axis_w_left + x_left + centers_mid[li] * si_post))

                rot_items.append((bi, x_center_mosaic, rot_img, w_rot, h_rot, top_pad_i, bottom_pad_i, False))
                max_h_rot_across      = max(max_h_rot_across, h_rot)
                max_top_pad_across    = max(max_top_pad_across, top_pad_i)
                max_bottom_pad_across = max(max_bottom_pad_across, bottom_pad_i)

            # 若该行有任意文本，标准道画“Marker”（黑色）
            if any_text and 0 <= skip_idx < len(centers_mid):
                lane_w_mosaic = int(round(widths_mid[skip_idx] * si_post))
                rot_img, w_rot, h_rot = render_vertical_text_fitting("Marker", lane_w_mosaic)
                if rot_img is not None:
                    x_center_mosaic = int(round(axis_w_left + x_left + centers_mid[skip_idx] * si_post))
                    rot_items.append((bi, x_center_mosaic, rot_img, w_rot, h_rot, top_pad_i, bottom_pad_i, True))
                    max_h_rot_across      = max(max_h_rot_across, h_rot)
                    max_top_pad_across    = max(max_top_pad_across, top_pad_i)
                    max_bottom_pad_across = max(max_bottom_pad_across, bottom_pad_i)

        # 若没有任何要绘制的文本，直接返回
        if not rot_items:
            return mosaic_bgr, 0

        # === 3) 统一顶栏：基线对齐（与单图“每块行底沿”视觉一致） ===
        # 统一使用“全局最大”的 top_pad / 文本高 / bottom_pad 组合确定面板高度
        panel_top_pad    = int(max_top_pad_across)
        panel_text_h_max = int(max_h_rot_across)
        panel_bottom_pad = int(max_bottom_pad_across)
        top_panel_h = int(max(1, panel_top_pad + panel_text_h_max + panel_bottom_pad))
        panel = np.full((top_panel_h, W_total, 3), 255, dtype=np.uint8)

        # 统一基线：所有标签的底沿都落在 (panel_top_pad + panel_text_h_max) 这一“行基线”
        baseline_y = panel_top_pad + panel_text_h_max

        for (bi, x_center_mosaic, rot_img, w_rot, h_rot, top_pad_i, bottom_pad_i, is_marker) in rot_items:
            # 横向：以 x_center 居中，防止溢出
            x_left = int(np.clip(x_center_mosaic - w_rot // 2, 2, W_total - w_rot - 2))
            # 纵向：底沿对齐到统一基线（而非使用各自 bottom_pad_i），保证一条直线
            y_top = int(np.clip(baseline_y - h_rot, 0, top_panel_h - h_rot))

            # 粘贴（含边界裁切）
            y1, y2 = y_top, y_top + h_rot
            x1, x2 = x_left, x_left + w_rot
            if y2 <= y1 or x2 <= x1:
                continue
            sub_h = min(h_rot, panel.shape[0] - y1)
            sub_w = min(w_rot, panel.shape[1] - x1)
            if sub_h <= 0 or sub_w <= 0:
                continue
            panel[y1:y1 + sub_h, x1:x1 + sub_w] = rot_img[:sub_h, :sub_w, :]

        # === 4) 叠加到原图顶部 ===
        out = np.vstack([panel, mosaic_bgr])
        return out, int(top_panel_h)



    def _apply_mosaic_annotations(
        self,
        mosaic_base: np.ndarray,
        absolute_pixels: bool = False,
        panel_top_h_override: int = None  # ★ 新增：右表顶部与胶对齐（顶栏下沿）
    ) -> np.ndarray:
        """
        在拼接后的基础图上叠加：
        - 右侧表格（可选择 absolute_pixels=True：绝对像素字号与内边距；不 shrink）；
        - 底部备注（保持绝对像素字号）。
        panel_top_h_override：若提供，则将表格的“面板顶部偏移”设为该值（通常 = 顶栏高度），
                            保证表格顶部与“胶体可视区域上边缘”对齐。
        """
        self._ensure_mosaic_vars()
        if mosaic_base is None or not isinstance(mosaic_base, np.ndarray) or mosaic_base.size == 0:
            return mosaic_base
        out = mosaic_base

        # —— 右侧表格 ——
        rtxt = (self.mosaic_right_table_text or "").strip()
        if rtxt:
            cfg = (self.mosaic_right_table_opts or {})
            out = self._attach_right_table_panel(
                out,
                rtxt,
                has_header=bool(cfg.get("has_header", True)),
                show_grid=bool(cfg.get("grid", True)),
                align=str(cfg.get("align", "center")),
                cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                panel_top_h=int(panel_top_h_override or 0),        # ★ 关键：对齐到顶栏下沿
                gel_height_px=out.shape[0],
                absolute_pixels=bool(absolute_pixels)              # ★ 走绝对像素路径（不 shrink）
            )

        # —— 底部备注 ——
        note_raw = self.mosaic_bottom_note_text or ""
        if note_raw.strip():
            out = self._attach_bottom_note_panel(out, note_raw, allow_expand_width=True)

        return out

    def _hook_right_table_auto_refresh(self):
        """
        将“右侧表格输入框 / 底注输入框”的文本变化与“拼接注解轻量刷新”绑定（去抖）。
        适配 tk.Text 或 tk.Entry。控件命名若不同，可在此函数里按你的实际名称微调。
        """
        import tkinter as tk

        def _bind_text_widget(widget, setter):
            if widget is None:
                return
            # tk.Text：使用 <<Modified>> 事件
            if isinstance(widget, tk.Text):
                def _on_modified(evt=None):
                    try:
                        txt = widget.get("1.0", "end").strip()
                        setter(txt)
                        self._refresh_mosaic_annotations()
                        widget.edit_modified(False)
                    except Exception:
                        pass
                try:
                    widget.bind("<<Modified>>", _on_modified)
                except Exception:
                    pass
            # tk.Entry：使用 <KeyRelease>
            elif isinstance(widget, tk.Entry):
                def _on_key(evt=None):
                    try:
                        setter(widget.get())
                        self._refresh_mosaic_annotations()
                    except Exception:
                        pass
                try:
                    widget.bind("<KeyRelease>", _on_key)
                except Exception:
                    pass
        # 根据你项目里的实际控件名进行映射（若名称不同，请在这里改成你真实的变量）
        table_widget = getattr(self, "txt_right_table", None)     # 假定：右侧表格输入框（tk.Text）
        note_widget  = getattr(self, "txt_bottom_note", None)     # 假定：底注输入框（tk.Text 或 tk.Entry）

        # 统一的 setter：同步“拼接专用变量”，供 _apply_mosaic_annotations 读取
        def _set_table_text(v: str):
            self.mosaic_right_table_text = (v or "")

        def _set_note_text(v: str):
            self.mosaic_bottom_note_text = (v or "")

        _bind_text_widget(table_widget, _set_table_text)
        _bind_text_widget(note_widget, _set_note_text)


    def _refresh_mosaic_annotations(self, debounce_ms: int = 150):
        """
        仅刷新拼接图的注解层（右表/底注），不重新跑对齐与拼接。
        要求：compose_stash 已经跑过（缓存了 mosaic_base_last / mosaic_top_panel_h_last）。
        """
        import numpy as np

        # 去抖
        if not hasattr(self, "_pending_refresh_job"):
            self._pending_refresh_job = None
        if self._pending_refresh_job is not None:
            try:
                self.after_cancel(self._pending_refresh_job)
            except Exception:
                pass
            self._pending_refresh_job = None

        def _do():
            base = getattr(self, "mosaic_base_last", None)
            top_h = int(getattr(self, "mosaic_top_panel_h_last", 0))
            if base is None or (not isinstance(base, np.ndarray)) or base.size == 0:
                # 没有缓存，退化为一次完整拼接
                try:
                    self.compose_stash()
                except Exception:
                    pass
                return
            try:
                disp = self._apply_mosaic_annotations(
                    base.copy(),
                    absolute_pixels=True,
                    panel_top_h_override=top_h
                )
                # 更新右侧画布底图（不重置箭头）
                if hasattr(self, "canvas_anno") and self.canvas_anno is not None:
                    try:
                        self.canvas_anno.set_scene(
                            base_img_bgr=disp,
                            gel_bgr=np.zeros_like(disp),
                            bounds=None, lanes=None,
                            a=0.0, b=0.0, fit_ok=True,
                            nlanes=0, ladder_lane=1,
                            yaxis_side="left",
                            lane_marks=[],
                            panel_top_h=0, panel_w=0,
                            calib_model=None,
                            reset_positions=False
                        )
                        try:
                            self.canvas_anno._redraw_arrows()
                        except Exception:
                            pass
                    except Exception:
                        pass
                # 同步预览记忆（可选）
                try:
                    self._set_composed_preview(base_img=base, display_img=disp)
                except Exception:
                    pass
            finally:
                self._pending_refresh_job = None

        try:
            self._pending_refresh_job = self.after(int(max(0, debounce_ms)), _do)
        except Exception:
            _do()



    def _attach_right_table_panel(
        self,
        img_bgr: np.ndarray,
        table_text: str,
        has_header: bool = True,
        show_grid: bool = True,
        align: str = "center",   # "left" / "center" / "right"
        cap_px: int = None,      # 字高像素（绝对）
        cell_pad_x: int = 12,    # 单元内边距（绝对）
        cell_pad_y: int = 8,
        line_color: tuple = (0, 0, 0),   # BGR
        gap_px: int = 30,
        panel_top_h: int = 0,
        gel_height_px: int = None,       # 仅自适应模式参考高度
        cap_policy: str = "max_auto",    # 兼容原参数
        absolute_pixels: bool = False    # ★ 新增：True 则使用绝对像素，不做 shrink/self-scaling
    ) -> np.ndarray:
        """
        右侧白底表格列（从 Excel 粘贴文本解析）。
        - absolute_pixels=True 时：
        · cap_px/内边距按传入“绝对像素”使用；
        · 不进行 shrink-to-fit；
        · panel 顶对齐，高度不足则在底部补白；过高则裁切（兜底）。
        - absolute_pixels=False 时，保留原有 shrink-to-fit 行为（兼容旧逻辑）。
        """
        import numpy as np, cv2
        from PIL import Image, ImageDraw, ImageFont

        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return img_bgr

        H_img, W_img = img_bgr.shape[:2]
        panel_top_h = max(0, int(panel_top_h))
        body_avail_h = max(0, H_img - panel_top_h)

        # 解析粘贴文本
        raw = (table_text or "").replace("\r\n", "\n").replace("\r", "\n")
        rows_raw = raw.split("\n")
        rows = []
        for ln in rows_raw:
            sline = ln
            if "\t" in sline:
                cells = sline.split("\t")
            else:
                s2 = sline.replace("，", ",").replace("；", ";").replace("、", ",")
                if "," in s2:   cells = s2.split(",")
                elif ";" in s2: cells = s2.split(";")
                else:           cells = [s2]
            rows.append([c.strip() for c in cells])
        rows = [r for r in rows if any((c.strip() for c in r))]
        if not rows:
            gap = np.full((H_img, max(1, gap_px), 3), 255, dtype=np.uint8)
            return np.concatenate([img_bgr, gap], axis=1)

        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r.extend([""] * (max_cols - len(r)))

        # 字体与测量
        font_path = self._find_font_ttf()
        use_pil = font_path is not None

        # 统一度量函数（根据 absolute_pixels 决定字号与内边距）
        if absolute_pixels:
            # 绝对像素：固定 cap/内边距，不做自适应缩放
            cap_px_abs = int(cap_px) if isinstance(cap_px, (int, float)) and cap_px else int(self._get_design_params().get("bottom_cap_px", 28))
            pad_x_abs  = int(cell_pad_x)
            pad_y_abs  = int(cell_pad_y)

            if use_pil:
                try:
                    font = ImageFont.truetype(font_path, size=max(12, cap_px_abs))
                except Exception:
                    use_pil = False

            def measure_table_fixed():
                if use_pil:
                    tmp = Image.new("RGB", (8, 8), "white")
                    drw = ImageDraw.Draw(tmp)
                    def meas(text: str):
                        t = text if text else " "
                        bbox = drw.textbbox((0, 0), t, font=font)
                        return max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])
                else:
                    font_cv = cv2.FONT_HERSHEY_SIMPLEX
                    scale_cv = self._font_scale_for_cap_height_px(cap_px_abs, font=font_cv, thickness=2)
                    def meas(text: str):
                        t = text if text else " "
                        (tw, th), _ = cv2.getTextSize(t, font_cv, scale_cv, 2)
                        return max(1, tw), max(1, th)

                col_w = [0]*max_cols
                row_h = [0]*len(rows)
                for i, r in enumerate(rows):
                    max_h = 0
                    for j, cell in enumerate(r):
                        tw, th = meas(cell)
                        col_w[j] = max(col_w[j], tw + 2*pad_x_abs)
                        max_h = max(max_h, th + 2*pad_y_abs)
                    row_h[i] = max(1, max_h)
                grid_w = sum(col_w) + (1 if show_grid else 0)
                grid_h = sum(row_h) + (1 if show_grid else 0)
                return col_w, row_h, grid_w, grid_h

            col_w, row_h, grid_w, grid_h = measure_table_fixed()

            # 绘制（绝对像素）
            if use_pil:
                font_p = font
                panel_img = Image.new("RGB", (grid_w, grid_h), "white")
                draw = ImageDraw.Draw(panel_img)

                header_rows = 1 if has_header and len(rows) >= 1 else 0
                if header_rows == 1:
                    y_top = 0
                    hdr_h = row_h[0]
                    draw.rectangle([(0, y_top), (grid_w-1, y_top+hdr_h-1)], fill=(240,240,240))

                if show_grid:
                    y = 0
                    draw.line([(0, y), (grid_w-1, y)], fill=(0,0,0), width=1)
                    for h in row_h:
                        y += h
                        draw.line([(0, y), (grid_w-1, y)], fill=(0,0,0), width=1)
                    x = 0
                    draw.line([(x, 0), (x, grid_h-1)], fill=(0,0,0), width=1)
                    for w in col_w:
                        x += w
                        draw.line([(x, 0), (x, grid_h-1)], fill=(0,0,0), width=1)

                # 文本
                def _x_for_cell(tw, col_left, col_wid):
                    if align == "left":  return col_left + pad_x_abs
                    if align == "right": return col_left + max(0, col_wid - tw - pad_x_abs)
                    return col_left + max(0, (col_wid - tw)//2)

                tmp = Image.new("RGB", (8,8), "white")
                drw = ImageDraw.Draw(tmp)
                def meas_pil(text: str):
                    t = text if text else " "
                    bb = drw.textbbox((0,0), t, font=font_p)
                    return max(1, bb[2]-bb[0]), max(1, bb[3]-bb[1])

                y_cursor = 0
                for i, r in enumerate(rows):
                    x_cursor = 0
                    for j, cell in enumerate(r):
                        tw, th = meas_pil(cell)
                        tx = _x_for_cell(tw, x_cursor, col_w[j])
                        ty = y_cursor + max(0, (row_h[i]-th)//2)
                        if (has_header and i==0):
                            draw.text((tx, ty), cell or " ", fill=(0,0,0), font=font_p, stroke_width=1, stroke_fill=(0,0,0))
                        else:
                            draw.text((tx, ty), cell or " ", fill=(0,0,0), font=font_p)
                        x_cursor += col_w[j]
                    y_cursor += row_h[i]

                panel_bgr = cv2.cvtColor(np.array(panel_img), cv2.COLOR_RGB2BGR)
            else:
                # OpenCV 绘制
                font_cv = cv2.FONT_HERSHEY_SIMPLEX
                scale_cv = self._font_scale_for_cap_height_px(cap_px_abs, font=font_cv, thickness=2)
                panel_bgr = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

                header_rows = 1 if has_header and len(rows) >= 1 else 0
                if header_rows == 1:
                    y_top = 0; hdr_h = row_h[0]
                    cv2.rectangle(panel_bgr, (0,y_top), (grid_w-1, y_top+hdr_h-1), (240,240,240), thickness=-1)

                if show_grid:
                    y = 0
                    cv2.line(panel_bgr, (0,y), (grid_w-1,y), (0,0,0), 1)
                    for h in row_h:
                        y += h
                        cv2.line(panel_bgr, (0,y), (grid_w-1,y), (0,0,0), 1)
                    x = 0
                    cv2.line(panel_bgr, (x,0), (x,grid_h-1), (0,0,0), 1)
                    for w in col_w:
                        x += w
                        cv2.line(panel_bgr, (x,0), (x,grid_h-1), (0,0,0), 1)

                def _x_for_cell_cv(tw, col_left, col_wid):
                    if align == "left":  return col_left + pad_x_abs
                    if align == "right": return col_left + max(0, col_wid - tw - pad_x_abs)
                    return col_left + max(0, (col_wid - tw)//2)

                y_cursor = 0
                for i, r in enumerate(rows):
                    x_cursor = 0
                    for j, cell in enumerate(r):
                        t = cell if cell else " "
                        (tw, th), base = cv2.getTextSize(t, font_cv, scale_cv, 2)
                        tx = _x_for_cell_cv(tw, x_cursor, col_w[j])
                        ty = y_cursor + max(0, (row_h[i] + th)//2)
                        try:
                            cv2.putText(panel_bgr, t, (tx, ty), font_cv, scale_cv, line_color, 2, cv2.LINE_AA)
                        except Exception:
                            cv2.putText(panel_bgr, "?", (tx, ty), font_cv, scale_cv, line_color, 2, cv2.LINE_AA)
                        x_cursor += col_w[j]
                    y_cursor += row_h[i]

            # 顶对齐到主体高度
            Hp, Wp = panel_bgr.shape[:2]
            if Hp < body_avail_h:
                pad = np.full((body_avail_h - Hp, Wp, 3), 255, dtype=np.uint8)
                panel_bgr = np.vstack([panel_bgr, pad])
            elif Hp > body_avail_h:
                panel_bgr = panel_bgr[:body_avail_h, :, :]

            top_pad_img = np.full((panel_top_h, Wp, 3), 255, dtype=np.uint8) if panel_top_h > 0 else None
            column_full = panel_bgr if top_pad_img is None else np.vstack([top_pad_img, panel_bgr])

            gap = np.full((H_img, max(1, gap_px), 3), 255, dtype=np.uint8)
            out = np.concatenate([img_bgr, gap, column_full], axis=1)
            return out

        # —— 非 absolute（老逻辑保持不变）——
        # 下面保留你原有自适应/收缩实现（略，为节省篇幅，你可以直接沿用旧版 _attach_right_table_panel）
        # 为保证完整性，这里调用原接口：回退到旧行为
        return super(App, self)._attach_right_table_panel(
            img_bgr, table_text, has_header, show_grid, align, cap_px,
            cell_pad_x, cell_pad_y, line_color, gap_px, panel_top_h, gel_height_px, cap_policy
        )



    def _snapshot_arrows_from_canvas(self) -> list[dict]:
        """
        从右侧交互画布（RightAnnoCanvas）读取所有箭头，转换为“凝胶坐标系”的绝对位置，返回列表：
        [
            { "lane_idx": int, "mw": float, "x_gel": float, "y_gel": float, "moved": bool },
            ...
        ]
        说明：
        - x_gel = x_img - canvas.x_offset
        - y_gel = y_img - canvas.panel_top_h
        """
        import numpy as np
        out = []
        can = getattr(self, "canvas_anno", None)
        if not can or not getattr(can, "arrows", None):
            return out
        x_off = float(getattr(can, "x_offset", 0.0))
        y_off = float(getattr(can, "panel_top_h", 0.0))
        for a in can.arrows:
            try:
                li = int(a.get("lane_idx", -1))
                mw = float(a.get("mw", float("nan")))
                x_img = float(a.get("x_img", float("nan")))
                y_img = float(a.get("y_img", float("nan")))
                if li < 0 or not (np.isfinite(mw) and mw > 0): 
                    continue
                if not (np.isfinite(x_img) and np.isfinite(y_img)):
                    continue
                x_gel = float(x_img - x_off)
                y_gel = float(y_img - y_off)
                out.append({
                    "lane_idx": li,
                    "mw": float(mw),
                    "x_gel": float(x_gel),
                    "y_gel": float(y_gel),
                    "moved": bool(a.get("moved", True))  # 没标记也当成 moved=True，确保被保留
                })
            except Exception:
                continue
        return out

    # -------------------- 自适应显示：注册 / 刷新 -------------------- #
    def _bind_autofit(self, widget: tk.Label):
        if widget not in self._autofit_store:
            self._autofit_store[widget] = {"img": None, "last": (0, 0)}
        widget.bind("<Configure>", lambda e, w=widget: self._render_autofit(w), add="+")
        parent = widget.nametowidget(widget.winfo_parent())
        parent.bind("<Configure>", lambda e, w=widget: self._render_autofit(w), add="+")

    def _set_autofit_image(self, widget: tk.Label, img_bgr: np.ndarray | None):
        if widget not in self._autofit_store:
            self._bind_autofit(widget)
        self._autofit_store[widget]["img"] = img_bgr
        self._render_autofit(widget)

        # app 23.py —— 在 App 类中新增一个辅助方法：根据 overlay/bounds/lanes 绘制绿线与刻度
    def _font_scale_for_cap_height_px(self, cap_px: int, font=cv2.FONT_HERSHEY_SIMPLEX, thickness: int = 1) -> float:
        """
        给定目标像素字高（cap_px），返回 cv2.putText 的 fontScale。
        做法：对样例文本（取 "Ag"）进行二分搜索，使 getTextSize 返回的高度 ≈ cap_px。
        """
        cap_px = max(1, int(cap_px))
        lo, hi = 0.1, 5.0
        target = cap_px

        def height_for(scale: float) -> int:
            (_, h), _ = cv2.getTextSize("Ag", font, scale, thickness)
            return max(1, int(h))

        for _ in range(20):
            mid = (lo + hi) / 2.0
            h = height_for(mid)
            if h < target:
                lo = mid
            else:
                hi = mid
        return hi
    def _standardize_gel_size(self, gel_bgr: np.ndarray, target_width_px: int) -> np.ndarray:
        """
        将 WB 后的凝胶图等比缩放到 target_width_px 像素宽（高度随比例变动）。
        """
        if gel_bgr is None or gel_bgr.size == 0:
            return gel_bgr
        H, W = gel_bgr.shape[:2]
        target_width_px = max(200, int(target_width_px))  # 下限防呆
        if W == target_width_px:
            return gel_bgr
        scale = target_width_px / float(W)
        new_w = target_width_px
        new_h = max(1, int(round(H * scale)))
        return cv2.resize(gel_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    def _get_design_params(self) -> dict:
        """
        统一读取 UI/图像文字相关的设计参数（像素字高）。
        """
        design_gel_width_px = getattr(self, "var_design_gel_width_px", None)
        top_cap_px = getattr(self, "var_top_font_cap_px", None)
        bottom_cap_px = getattr(self, "var_bottom_font_cap_px", None)

        # 原图宽度标准化（保持不变）
        design_gel_width_px = int(design_gel_width_px.get()) if design_gel_width_px else 1200

        # 顶部列名面板：26（你当前已较清晰）
        top_cap_px = int(top_cap_px.get()) if top_cap_px else 26

        # ★ 底部备注：统一走设计参数；默认 28（显著大且清晰）
        bottom_cap_px = int(bottom_cap_px.get()) if bottom_cap_px else 28

        # 侧边刻度：保持 16
        axis_cap_px = 16

        return {
            "design_gel_width_px": max(200, design_gel_width_px),
            "top_cap_px": max(8, top_cap_px),
            "bottom_cap_px": max(8, bottom_cap_px),
            "axis_cap_px": max(8, axis_cap_px),
        }
    def _draw_overlays_on_core(
        self,
        img_core: np.ndarray,
        gel_bgr: np.ndarray,
        overlay: dict | None,
        bounds: np.ndarray | None,
        lanes: list[tuple[int, int]] | None,
        a: float, b: float, fit_ok: bool,
        tick_labels: list[float],
        yaxis_side: str,
        show_green: bool | None = None,
        y_offset: int = 0,
    ) -> np.ndarray:
        """
        仅叠加“绿线（分隔线）”，不再在 App 侧绘制任何刻度/数字。
        这样可避免与 render_annotation/_slanted 底图中的刻度重复。
        """
        if img_core is None or img_core.size == 0:
            return img_core

        img = img_core.copy()
        H, W_total = img.shape[:2]
        Hg, Wg = gel_bgr.shape[:2]

        # 轴侧与白板宽度（决定 X 偏移）
        panel_w = (overlay.get("panel_w") if overlay else max(0, W_total - Wg))
        side = (overlay.get("yaxis_side") if overlay else str(yaxis_side or "left").lower())
        x_offset = 0 if side == "right" else panel_w

        # === 绿线（分隔线） ===
        boundaries = (overlay.get("boundaries") if overlay else None)
        if not boundaries:
            boundaries = []
            if isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[0] == Hg:
                # 斜率模型：逐列随 y 变动
                for i in range(1, bounds.shape[1] - 1):
                    xcol = (bounds[:, i].astype(np.int32) + int(x_offset))
                    xcol = np.clip(xcol, 0, W_total - 1)
                    y = y_offset + np.arange(Hg, dtype=np.int32)
                    pts = np.stack([xcol, y], axis=1)
                    boundaries.append(pts)
            elif lanes:
                # 等宽模型：竖直直线
                xs = [lanes[i][0] for i in range(1, len(lanes))]
                for x in xs:
                    xg = int(np.clip(x_offset + int(x), 0, W_total - 1))
                    y = y_offset + np.arange(Hg, dtype=np.int32)
                    pts = np.stack([np.full((Hg,), xg, dtype=np.int32), y], axis=1)
                    boundaries.append(pts)

        # ★ 仅控制绿线开关，不再画刻度
        do_green = bool(self.var_show_green.get()) if show_green is None else bool(show_green)
        if do_green:
            for pts in boundaries or []:
                cv2.polylines(
                    img, [pts.astype(np.int32)],
                    isClosed=False, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA
                )

        return img


    def on_toggle_show_boxes(self):
        """
        勾选/取消“显示红色方框”时的回调：
        - 勾选：让右侧交互画布基于当前箭头一次性生成方框；
        - 取消：删除所有方框。
        """
        if not hasattr(self, "canvas_anno") or self.canvas_anno is None:
            return
        try:
            self.canvas_anno.set_boxes_enabled(bool(self.var_show_boxes.get()))
        except Exception:
            pass

    def _render_autofit(self, widget: tk.Label):
        store = self._autofit_store.get(widget)
        if not store:
            return
        img_bgr = store.get("img")
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            widget.configure(image="")
            widget.image = None
            return
        tw = max(1, widget.winfo_width())
        th = max(1, widget.winfo_height())
        H, W = img_bgr.shape[:2]
        if W < 1 or H < 1:
            return
        scale = min(tw / W, th / H)
        if abs(scale - 1.0) < 1e-3:
            rgb_disp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            new_w, new_h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            rgb_disp = cv2.cvtColor(cv2.resize(img_bgr, (new_w, new_h), interpolation=interp), cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil = Image.fromarray(rgb_disp)
        tkimg = ImageTk.PhotoImage(pil)
        widget.configure(image=tkimg)
        widget.image = tkimg

    def _find_font_ttf(self) -> str | None:
        """
        返回一个可用于 Pillow 的 TTF/TTC/OTF 字体文件路径。
        ★ 优先尝试 Arial（满足你希望的统一风格）；找不到再回退到系统常见 CJK/通用字体。
        """
        import os, sys, glob

        # 缓存，避免反复扫盘
        cache_attr = "_font_ttf_cache_path"
        if hasattr(self, cache_attr) and getattr(self, cache_attr):
            return getattr(self, cache_attr)

        candidates: list[str] = []
        plat = sys.platform.lower()

        def add(paths: list[str]):  # 仅加入存在的路径或通配匹配的文件
            for p in paths:
                if any(ch in p for ch in "*?[]"):
                    for q in glob.glob(p):
                        if os.path.isfile(q):
                            candidates.append(q)
                else:
                    if os.path.isfile(p):
                        candidates.append(p)

        # ---- ① 强优先：Arial 系 ----
        if plat.startswith("win"):
            windir = os.environ.get("WINDIR", r"C:\Windows")
            fontdir = os.path.join(windir, "Fonts")
            add([
                os.path.join(fontdir, "arial.ttf"),                # Arial
                os.path.join(fontdir, "arialbd.ttf"),              # Arial Bold
                os.path.join(fontdir, "ariali.ttf"),               # Arial Italic
                os.path.join(fontdir, "arialbi.ttf"),              # Arial Bold Italic
                os.path.join(fontdir, "arialuni.ttf"),             # Arial Unicode MS（更全字库）
            ])
        elif plat == "darwin":
            add([
                "/Library/Fonts/Arial.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/Library/Fonts/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode MS.ttf",
            ])
        else:
            # Linux 常见安装位置（含 msttcorefonts）
            add([
                "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
            ])

        # ---- ② 回退：常见 CJK/通用字体（保证中文/兼容性）----
        if plat.startswith("win"):
            windir = os.environ.get("WINDIR", r"C:\Windows")
            fontdir = os.path.join(windir, "Fonts")
            add([
                os.path.join(fontdir, "msyh.ttc"),   # Microsoft YaHei
                os.path.join(fontdir, "simsun.ttc"), # SimSun
            ])
        elif plat == "darwin":
            add(glob.glob("/System/Library/Fonts/*PingFang*.ttc"))
            add([
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/PingFangSC.ttc",
                "/System/Library/Fonts/Supplemental/Songti.ttc",
            ])
        else:
            add([
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ])

        # 用户字体目录
        home = os.path.expanduser("~")
        add(glob.glob(os.path.join(home, ".fonts", "**", "*.tt*")))
        add(glob.glob(os.path.join(home, ".local", "share", "fonts", "**", "*.tt*")))

        font_path = candidates[0] if candidates else None
        setattr(self, cache_attr, font_path)
        return font_path

    def _init_paned_positions(self):
        try:
            self.update_idletasks()
            total_h = max(1, self.pw_main.winfo_height())
            self.pw_main.sash_place(0, 0, int(total_h * 0.58))
            total_w = max(1, self.pw_bottom.winfo_width())
            self.pw_bottom.sash_place(0, int(total_w * 0.5), 0)
        except Exception:
            pass

    # -------------------- 逻辑 -------------------- #
    def _spin(self, parent, text, var, from_, to, step=1):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(
            f, text=text, wraplength=getattr(self, "LEFT_WIDTH", 300) - 40, justify="left"
        ).pack(side=tk.LEFT)
        sp = ttk.Spinbox(f, textvariable=var, from_=from_, to=to, increment=step, width=8)
        sp.pack(side=tk.RIGHT)

    # -------------------- 文件/检测 -------------------- #

    def export_arrow_box_metrics(self):
            """
            导出“每个箭头对应方框”的灰度统计为 CSV：
              - lane_no（真实泳道 1-based）与 lane_label（若可从列名映射得到）
              - mw（kDa）
              - sum_intensity / pixel_count / mean_intensity
              - x, y, w, h（方框在最终导出图像坐标系中的坐标/尺寸）
            要求：
              - 右侧画布需已渲染，且【显示红色方框】已勾选并生成方框。
            """
            # 基本可用性检查
            if not hasattr(self, "canvas_anno") or self.canvas_anno is None:
                messagebox.showwarning("Notice", "Right-side annotation canvas is unavailable. Please render first.")
                return
            # 需要已有方框（由“显示红色方框”开关生成）
            if not getattr(self.canvas_anno, "boxes", None):
                messagebox.showwarning("Notice", "No bounding box detected. Please check 'Show red box' and render before exporting.")
                return

            # 取度量
            try:
                metrics = self.canvas_anno.get_box_metrics()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to retrieve bounding box metrics:{e}")
                return
            if not metrics:
                messagebox.showwarning("Notice", "No measurment data to export")
                return

            # 构造泳道标签映射（从真实泳道 -> 非标准道位置 -> lane_names）
            # 真实泳道数量
            can = self.canvas_anno
            if isinstance(getattr(can, "bounds", None), np.ndarray):
                real_nlanes = max(0, can.bounds.shape[1] - 1)
            elif isinstance(getattr(can, "lanes", None), list) and can.lanes:
                real_nlanes = len(can.lanes)
            else:
                real_nlanes = int(getattr(can, "nlanes", 0)) or int(self.var_nlanes.get())

            skip_idx = max(0, int(getattr(can, "ladder_lane", 1)) - 1)  # 标准道(0-based)
            nonladder_idx = [i for i in range(real_nlanes) if i != skip_idx]
            laneidx_to_nonladder_pos = {li: pos for pos, li in enumerate(nonladder_idx)}

            # 组织输出行
            rows = []
            for m in metrics:
                lane_no = m.get("lane_no")
                lane_label = ""
                if lane_no is not None:
                    li0 = int(lane_no) - 1  # 0-based
                    pos = laneidx_to_nonladder_pos.get(li0, None)
                    if pos is not None and pos < len(self.lane_names):
                        # 仅非标准道有标签；若没有对应文本则给空串
                        lane_label = (self.lane_names[pos] or "").strip()

                rows.append({
                    "lane_no": lane_no,
                    "lane_label": lane_label,
                    "mw_kDa": m.get("mw"),
                    "sum_intensity": m.get("sum_intensity"),
                    "pixel_count": m.get("pixel_count"),
                    "mean_intensity": m.get("mean_intensity"),
                    "x": m.get("x"),
                    "y": m.get("y"),
                    "w": m.get("w"),
                    "h": m.get("h"),
                    "arrow_x": m.get("arrow_x"),
                    "arrow_y": m.get("arrow_y"),
                })

            # 文件名建议
            default_name = "arrow_boxes_metrics.csv"
            try:
                if getattr(self, "image_path", None):
                    p = Path(self.image_path)
                    default_name = f"{p.stem}_arrow_boxes_metrics.csv"
            except Exception:
                pass

            # 选择保存路径
            fpath = filedialog.asksaveasfilename(
                title="Failed to retrieve arrow box grayscale (CSV)",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV", "*.csv")]
            )
            if not fpath:
                return

            # 写出 CSV（Excel 友好：UTF-8 BOM）
            import csv
            headers = [
                "lane_no", "lane_label", "mw_kDa",
                "sum_intensity", "pixel_count", "mean_intensity",
                "x", "y", "w", "h",
                "arrow_x", "arrow_y",
            ]
            try:
                with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write into csv：{e}")
                return

            messagebox.showinfo("Complete", f"Exported CSV：{fpath}\nwith {len(rows)} entries.")



    def open_image(self):
        self._deep_reset_for_new_image()
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("Image", "*.jpg;*.jpeg;*.png;*.tif;*.tiff"), ("All", "*.*")]
        )
        if not path:
            return
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Error", "Failed to read the image file.")
            return
        try:
            bgr_comp, est_bytes = self._compress_image_near_size(
                bgr,
                target_bytes=1_024_000,
                tol_ratio=0.08,
                min_quality=40,
                max_quality=95,
                min_side_px=720,
                max_downscale_rounds=4
            )
            self.orig_bgr = bgr_comp
            self.image_path = path
            self.image_load_est_bytes = int(est_bytes)
        except Exception:
            self.orig_bgr = bgr
            self.image_path = path
            self.image_load_est_bytes = int(getattr(bgr, "nbytes", 0))

        # —— 清空（单图）注释 —— #
        try:
            self.bottom_note_text = ""
        except Exception:
            setattr(self, "bottom_note_text", "")

        self.lane_names = []
        self.lane_marks = []
        self.render_cache = {}
        self._clear_right_canvas()
        self.gi = 1
        self.boxes = []
        self.roi_editor.set_image(self.orig_bgr)

        # —— 清空（单图）右侧表格 —— #
        self.right_table_text = ""
        self.right_table_opts = {}

        # —— 关键：清空“拼接专用”变量，确保新工程互不影响 —— #
        self._ensure_mosaic_vars()
        self.mosaic_right_table_text = ""
        self.mosaic_right_table_opts = {}
        self.mosaic_bottom_note_text = ""

        self.run_detect()




    def run_detect(self):
        self._clear_all_caches_and_overlays()
        if self.orig_bgr is None:
            return
        block = self.var_block.get()
        if block % 2 == 0:
            block += 1
        morph = self.var_morph.get()
        if morph % 2 == 0:
            morph += 1
        self.boxes = detect_gel_regions(
            self.orig_bgr,
            expected=int(self.var_expected.get()),
            thr_block=int(block),
            thr_C=int(self.var_thrC.get()),
            morph_ksize=int(morph)
        )
        if not self.boxes:
            #messagebox.showwarning("提示", "未检测到胶块，请调整参数或确认图片。")
            self.roi_editor.set_roi(None)
            return
        # 默认选第一块
        self.gi = 1
        self.roi_editor.set_roi(self.boxes[0])
    def reset_all_arrows(self):
        """
        一键将右侧画布的所有箭头重置为默认位置。
        """
        if not hasattr(self, "canvas_anno") or self.canvas_anno is None:
            messagebox.showwarning("Notice", "Right-side annotation canvas is unavailable. Please render first.")
            return
        try:
            if not getattr(self.canvas_anno, "arrows", None):
                messagebox.showinfo("Notice", "No arrow to reset currently.")
                return
            self.canvas_anno.reset_arrows_to_default()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset arrows: {e}")
    def _clear_right_canvas(self):
        """
        清空右侧交互画布的底图、箭头、方框等叠加元素，避免跨 ROI/图像污染。
        """
        can = getattr(self, "canvas_anno", None)
        if not can:
            return
        try:
            can.delete("all")
        except Exception:
            pass
        try:
            if hasattr(can, "arrows"):
                can.arrows.clear()
            if hasattr(can, "boxes"):
                can.boxes.clear()
            # 同时清空底图相关状态，强制后续 set_scene 重建
            can.base_img_bgr = None
            can.base_img_tk = None
            can.img_item = None
        except Exception:
            pass

    def _clear_all_caches_and_overlays(self,
                                    preserve_inputs: bool = True, 
                                    reset_composed: bool = True):
        """
        软初始化入口（增强版）：
        - preserve_inputs=True：保留用户输入（lane_names、lane_marks、bottom_note_text、
        right_table_text/opts 等），仅清空运行态（右侧画布/拼接态/render 缓存），
        用于“编辑/渲染/清空暂存”等不希望丢当前编辑内容的场景。
        - preserve_inputs=False：连同上述输入一并清空，用于“读取新图片”的全新开始。
        - reset_composed=True：退出拼接模式并清理一次性标记。
        """
        # 1) 清空渲染缓存
        try:
            self.render_cache = {}
        except Exception:
            pass

        # 2) 清空右侧交互画布（底图/箭头/方框）
        self._clear_right_canvas()

        # 3) 退出拼接模式并清理一次性标记
        if reset_composed:
            try:
                self._unset_composed_preview()
            except Exception:
                pass
            try:
                # 避免下一次 recompose_using_cache 误触发 reset_positions
                setattr(self, "_just_left_composed_preview", False)
            except Exception:
                pass

        # 4) 拼接专用变量（一般不影响单图；若希望更“干净”可清空）
        try:
            self._ensure_mosaic_vars()
            # 若保留输入，则只在读取新图片时清空
            if not preserve_inputs:
                self.mosaic_right_table_text = ""
                self.mosaic_right_table_opts = {}
                self.mosaic_bottom_note_text = ""
        except Exception:
            pass

        # 5) 用户输入：根据 preserve_inputs 决定是否保留
        if not preserve_inputs:
            try:
                self.bottom_note_text = ""
            except Exception:
                setattr(self, "bottom_note_text", "")
            try:
                self.right_table_text = ""
                self.right_table_opts = {}
            except Exception:
                setattr(self, "right_table_text", "")
                setattr(self, "right_table_opts", {})
            # 列名/分子量清空（新图从零开始）
            self.lane_names = []
            self.lane_marks = []
            # 也可视化开关归位
            try: self.var_show_boxes.set(False)
            except Exception: pass
            try: self.var_show_green.set(False)
            except Exception: pass

    def _soft_reset_for_operation(self):
        """操作前的轻量级软初始化：保留用户输入，仅清理运行态。"""
        try:
            self._clear_all_caches_and_overlays(preserve_inputs=True, reset_composed=True)
        except Exception:
            pass

    def _deep_reset_for_new_image(self):
        """读取新图片前的深度重置：不保留任何输入，彻底回到初始状态。"""
        try:
            self._clear_all_caches_and_overlays(preserve_inputs=False, reset_composed=True)
        except Exception:
            pass


    def _edit_right_table_auto(self):
        """
        “编辑右侧表格”按钮的自动分发入口：
        - 在拼接整体预览中：打开拼接专用编辑器（确认后直接改右侧预览底图）
        - 否则：沿用原有单图编辑器（确认后走 recompose_using_cache/render_current）
        """
        if bool(getattr(self, "_is_composed_preview", False)):
            self.open_right_table_editor_for_mosaic()
        else:
            self.open_right_table_editor()

    def _edit_bottom_note_auto(self):
        """
        “编辑底注”按钮的自动分发入口（逻辑同上）。
        """
        if bool(getattr(self, "_is_composed_preview", False)):
            self.open_bottom_note_editor_for_mosaic()
        else:
            self.open_bottom_note_editor()

    def open_right_table_editor_for_mosaic(self):
        """
        仅在“拼接整体预览”中生效的右侧表格编辑器（Confirm 才更新）：
        - 与 open_right_table_editor 相同的弹窗交互；
        - 只有在点击 Confirm 时，才写入 mosaic_right_table_text / mosaic_right_table_opts，
        并触发一次轻量刷新（_refresh_mosaic_annotations）。
        """
        import tkinter as tk
        from tkinter import ttk

        # 仅在拼接模式下启用；否则回退到单图编辑器
        if not bool(getattr(self, "_is_composed_preview", False)) or not getattr(self, "compose_preview", None):
            return self.open_right_table_editor()

        self._ensure_mosaic_vars()

        win = tk.Toplevel(self)
        win.title("Edit right-side table (mosaic)")
        win.transient(self); win.grab_set()
        win.resizable(True, True)

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frm, text="Paste here (Excel -> Copy; Click here -> Ctrl+V):",
                justify="left").pack(anchor="w")

        txt = tk.Text(frm, height=12, width=64, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        # 预填：拼接专用变量
        pre = getattr(self, "mosaic_right_table_text", "") or ""
        if pre:
            txt.insert("1.0", pre)

        # 选项区
        opt = ttk.LabelFrame(frm, text="Options")
        opt.pack(fill=tk.X, pady=(6, 0))

        var_header = tk.BooleanVar(value=True)
        var_grid   = tk.BooleanVar(value=True)
        var_align  = tk.StringVar(value="center")
        design     = self._get_design_params()
        var_cap    = tk.IntVar(value=int(design.get("bottom_cap_px", 28)))

        # 回显历史配置
        prev_cfg = getattr(self, "mosaic_right_table_opts", {}) or {}
        if "has_header" in prev_cfg: var_header.set(bool(prev_cfg.get("has_header")))
        if "grid" in prev_cfg:       var_grid.set(bool(prev_cfg.get("grid")))
        if "align" in prev_cfg:      var_align.set(str(prev_cfg.get("align") or "center"))
        if "cap_px" in prev_cfg and isinstance(prev_cfg["cap_px"], (int, float)):
            var_cap.set(int(prev_cfg["cap_px"]))

        row1 = ttk.Frame(opt); row1.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(row1, text="First row is header (gray)", variable=var_header).pack(side=tk.LEFT)
        ttk.Checkbutton(row1, text="Show grid lines", variable=var_grid).pack(side=tk.LEFT, padx=(12, 0))

        row2 = ttk.Frame(opt); row2.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row2, text="Align").pack(side=tk.LEFT)
        cb_align = ttk.Combobox(row2, textvariable=var_align, state="readonly",
                                values=["left", "center", "right"], width=8)
        cb_align.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row2, text="Font height (px)").pack(side=tk.LEFT)
        sp_cap = ttk.Spinbox(row2, textvariable=var_cap, from_=10, to=60, increment=1, width=6)
        sp_cap.pack(side=tk.LEFT, padx=(6, 0))

        # 按钮区
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))

        def do_clear():
            txt.delete("1.0", "end")
            # 不自动刷新；等 Confirm

        def do_cancel():
            win.destroy()

        def do_ok():
            # 仅在确认时写入数据并刷新
            try:
                self.mosaic_right_table_text = txt.get("1.0", "end")
            except Exception:
                self.mosaic_right_table_text = ""

            self.mosaic_right_table_opts = {
                "has_header": bool(var_header.get()),
                "grid": bool(var_grid.get()),
                "align": (var_align.get() or "center"),
                "cap_px": int(var_cap.get()),
            }

            win.destroy()

            # 轻量刷新（不重跑对齐/拼接；表格顶部自动对齐“顶栏下沿”）
            try:
                self._refresh_mosaic_annotations(debounce_ms=0)
            except Exception:
                pass

        ttk.Button(btns, text="Clear",   command=do_clear ).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel",  command=do_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Confirm", command=do_ok    ).pack(side=tk.RIGHT, padx=(0, 6))

        win.bind("<Escape>", lambda e: (do_cancel(), "break"))


    def open_bottom_note_editor_for_mosaic(self):
        """
        仅在“拼接整体预览”中生效的底注编辑器（Confirm 才更新）：
        - 只有点击 Confirm 时，才写入 mosaic_bottom_note_text，并触发一次轻量刷新。
        """
        import tkinter as tk
        from tkinter import ttk

        if not bool(getattr(self, "_is_composed_preview", False)) or not getattr(self, "compose_preview", None):
            return self.open_bottom_note_editor()

        self._ensure_mosaic_vars()

        win = tk.Toplevel(self)
        win.title("edit foot note (mosaic)")
        win.transient(self); win.grab_set()
        win.resizable(True, True)

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frm, text="Note content:", justify="left").pack(anchor="w")

        txt = tk.Text(frm, height=10, width=60, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        # 预填
        txt.insert("1.0", getattr(self, "mosaic_bottom_note_text", "") or "")

        # 按钮区
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X)

        def do_clear():
            txt.delete("1.0", "end")
            # 不自动刷新；等 Confirm

        def do_cancel():
            win.destroy()

        def do_ok():
            try:
                self.mosaic_bottom_note_text = txt.get("1.0", "end")
            except Exception:
                self.mosaic_bottom_note_text = ""

            win.destroy()

            # 轻量刷新（不重跑对齐/拼接；与右侧表格一起叠加）
            try:
                self._refresh_mosaic_annotations(debounce_ms=0)
            except Exception:
                pass

        ttk.Button(btns, text="Clear",   command=do_clear ).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel",  command=do_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Confirm", command=do_ok    ).pack(side=tk.RIGHT, padx=(0, 6))

        win.bind("<Escape>", lambda e: (do_cancel(), "break"))



    def on_toggle_show_green(self):
        """
        仅切换绿线显示，不做任何重算。
        若缓存不存在（例如首次，还未 render_current），则回退到 render_current。
        """
        cache = getattr(self, "render_cache", None) or {}
        img_no = cache.get("annotated_base_no_green")
        img_yes = cache.get("annotated_base_with_green")
        if img_no is None or img_yes is None:
            # 首次或缓存失效时，执行一次完整渲染构建缓存
            try:
                self.render_current()
            except Exception:
                pass
            return
    
        target = img_yes if bool(self.var_show_green.get()) else img_no
        # 快速替换右侧交互画布底图（箭头保持）
        if hasattr(self, "canvas_anno") and hasattr(self.canvas_anno, "update_base_image"):
            try:
                self.canvas_anno.update_base_image(target)
            except Exception:
                pass

    def switch_gel(self, delta):
        self._clear_all_caches_and_overlays()
        if not self.boxes:
            return
        self.gi = (self.gi - 1 + delta) % len(self.boxes) + 1  # 1..N 环绕
        self.roi_editor.set_roi(self.boxes[self.gi - 1])

    def reset_roi_from_detect(self):
        self._clear_all_caches_and_overlays()
        if not self.boxes:
            return
        idx = max(1, min(self.gi, len(self.boxes))) - 1
        self.roi_editor.set_roi(self.boxes[idx])

    def _show_np_on_label(self, widget: tk.Label, img_bgr: np.ndarray):
        self._set_autofit_image(widget, img_bgr)

    # -------- 旋转对齐裁剪（按旋转 ROI） -------- #
    @staticmethod
    def _rotate_bound_with_M(image_bgr: np.ndarray, angle_deg: float):
        """
        旋转保持完整：扩张画布，返回 (rot_img, M)；M 为 2x3 仿射矩阵（已含平移）
        """
        (h0, w0) = image_bgr.shape[:2]
        c = (w0 / 2.0, h0 / 2.0)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        nW = int(round((h0 * sin) + (w0 * cos)))
        nH = int(round((h0 * cos) + (w0 * sin)))
        # 平移修正：把图像中心移到新画布中心
        M[0, 2] += (nW / 2.0) - c[0]
        M[1, 2] += (nH / 2.0) - c[1]
        rot = cv2.warpAffine(image_bgr, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rot, M

    @staticmethod
    def _affine_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
        """
        把点 (x,y) 应用 2x3 仿射矩阵 M。
        """
        nx = M[0,0]*x + M[0,1]*y + M[0,2]
        ny = M[1,0]*x + M[1,1]*y + M[1,2]
        return nx, ny

    def _compress_image_near_size(
        self,
        bgr: np.ndarray,
        target_bytes: int = 1_024_000,   # 约 1MB
        tol_ratio: float = 0.08,         # 目标大小的容差（±8%）
        min_quality: int = 40,
        max_quality: int = 95,
        min_side_px: int = 720,          # 防止过度缩小
        max_downscale_rounds: int = 4
    ) -> tuple[np.ndarray, int]:
        """
        将 BGR 图像有损压缩到 ~target_bytes（默认 ~1MB）并返回 (压缩后BGR, 估算字节数)。
        策略：
          1) 先在当前分辨率下对 JPEG 质量做二分搜索，力求 <= target_bytes*(1+tol) 且尽量高质量；
          2) 若最低质量仍超标，则按比例缩小分辨率后重试（最多 max_downscale_rounds 次）；
          3) 最终用 cv2.imdecode 解码回 BGR，以便后续算法统一处理。
        说明：
          - 这一步是“读取后即做压缩”，会引入 JPEG 有损；如需无损保真，请告知改为仅缩放或使用 PNG。
        """
        if not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return bgr, 0

        H, W = bgr.shape[:2]
        img = bgr
        target_hi = int(target_bytes * (1.0 + tol_ratio))

        def _encode_with_quality(src: np.ndarray, q: int):
            ok, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            if not ok:
                raise RuntimeError("JPEG coding failed")
            return buf

        def _binary_search_quality(src: np.ndarray):
            lo, hi = min_quality, max_quality
            best_buf = None
            best_q = None
            # 质量二分：尽量逼近但不超过 target_hi；若超过，则降低质量
            for _ in range(9):
                if lo > hi:
                    break
                mid = (lo + hi) // 2
                buf = _encode_with_quality(src, mid)
                sz = int(buf.size)
                if sz <= target_hi:
                    best_buf = buf
                    best_q = mid
                    lo = mid + 1  # 还能更高质量
                else:
                    hi = mid - 1  # 需要更小
            # 若不存在<=target_hi 的结果，则返回最小质量的编码作为兜底
            if best_buf is not None:
                return best_buf, best_q
            return _encode_with_quality(src, min_quality), min_quality

        # 尝试：最多若干轮“缩放 + 质量搜索”
        for _round in range(max_downscale_rounds + 1):
            # 质量搜索（当前分辨率）
            buf, used_q = _binary_search_quality(img)
            cur_sz = int(buf.size)
            if cur_sz <= target_hi:
                # 命中目标，解码为 BGR 返回
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return (out if out is not None else img), cur_sz

            # 仍超标：若还允许继续缩放，则估算缩放比例，缩小再试
            if _round >= max_downscale_rounds:
                # 达到缩放轮次上限，直接用当前最小质量结果
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return (out if out is not None else img), cur_sz

            # 估算缩放比例：文件大小 ~ 像素数 * 压缩率，粗略按 sqrt 比例缩小边长
            scale = max(0.4, min(0.95, (target_bytes / float(cur_sz)) ** 0.5 * 0.97))
            new_w = max(min_side_px, int(W * scale))
            new_h = max(min_side_px, int(H * scale))
            if new_w == W and new_h == H:
                # 尺寸已经很小，避免死循环，直接返回当前结果
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return (out if out is not None else img), cur_sz
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            H, W = img.shape[:2]

        # 理论不会到此；兜底返回原图


# 在 App 类中新增（位置不拘，和其它小工具方法放一起即可）
    def _ensure_mosaic_vars(self):
        """确保拼接专用变量已存在。"""
        if not hasattr(self, "mosaic_right_table_text"):
            self.mosaic_right_table_text = ""
        if not hasattr(self, "mosaic_right_table_opts"):
            self.mosaic_right_table_opts = {}
        if not hasattr(self, "mosaic_bottom_note_text"):
            self.mosaic_bottom_note_text = ""


    def render_current(self):
        self._soft_reset_for_operation()
        try: self._unset_composed_preview()
        except Exception: pass
        if self.orig_bgr is None:
            return
        # 1) 旋转对齐并裁剪 ROI
        rroi = self.roi_editor.get_rotated_roi()
        if rroi is None:
            from tkinter import messagebox
            messagebox.showwarning("Notice", "Please adjust or create the ROI frame in the canvas first")
            return
        cx, cy, w, h, angle_ccw = rroi
        rot_img, M = self._rotate_bound_with_M(self.orig_bgr, -angle_ccw)
        def _affine_point(M_, x, y):
            return (M_[0,0]*x + M_[0,1]*y + M_[0,2], M_[1,0]*x + M_[1,1]*y + M_[1,2])
        cx2, cy2 = _affine_point(M, cx, cy)
        x0 = int(round(cx2 - w/2.0)); y0 = int(round(cy2 - h/2.0))
        x1 = x0 + int(round(w)); y1 = y0 + int(round(h))
        H2, W2 = rot_img.shape[:2]
        x0 = max(0, min(x0, W2 - 1)); y0 = max(0, min(y0, H2 - 1))
        x1 = max(x0 + 1, min(x1, W2)); y1 = max(y0 + 1, min(y1, H2))
        gel = rot_img[y0:y1, x0:x1].copy()

        # 2) WB
        gamma_val = float(self.var_gamma_val.get())
        gamma = float(gamma_val) if self.var_gamma_on.get() else None
        gel_bgr = auto_white_balance(
            gel,
            exposure=float(self.var_wb_exposure.get()),
            percent_low=float(self.var_wb_p_low.get()),
            percent_high=float(self.var_wb_p_high.get()),
            per_channel=bool(self.var_wb_per_channel.get()),
            gamma=gamma
        ) if self.var_wb_on.get() else gel

        # 2.5) 标准宽度
        design = self._get_design_params()
        gel_bgr = self._standardize_gel_size(gel_bgr, design["design_gel_width_px"])
        gel_gray = cv2.cvtColor(gel_bgr, cv2.COLOR_BGR2GRAY)

        # 3) 解析标准带数字
        def parse_list(s: str):
            s = (s or "").replace("，", ",")
            out = []
            for t in s.split(","):
                t = t.strip()
                if not t: continue
                try: out.append(float(t))
                except: pass
            return out
        ladder_labels_all = parse_list(self.ent_marker.get()) or [180,130,100,70,55,40,35,25,15,10]
        tick_labels = ladder_labels_all  # 仅在拟合通过时绘制刻度

        # 4) 分道
        nlanes = int(self.var_nlanes.get())
        mode = self.var_mode.get()
        if mode == "uniform":
            lanes = lanes_uniform(gel_gray, nlanes, int(self.var_lpad.get()), int(self.var_rpad.get()))
            bounds = None
        else:
            bounds = lanes_slanted(
                gel_gray, nlanes,
                smooth_px=int(self.var_smooth.get()),
                min_sep_ratio=float(self.var_sep.get()),
                search_half=12, max_step_px=5, smooth_y=9,
                wb_bgr=gel_bgr,
                enable_uniform_blend=False
            )
            lanes = None

# 5) 标准道检测 + 分段线性标定
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
        y0_roi, y1_roi = 0, None

        # 解析“标准序列”并确定目标数量 K
        def parse_list(s: str):
            s = (s or "").replace("，", ",")
            out = []
            for t in s.split(","):
                t = t.strip()
                if not t: continue
                try: out.append(float(t))
                except: pass
            return out
        ladder_labels_all = parse_list(self.ent_marker.get()) or [180,130,100,70,55,40,35,25,15,10]
        K_target = len([x for x in ladder_labels_all if isinstance(x, (int,float)) and x>0])

        if bounds is not None:
            # 斜率泳道：传入 target_count=标准序列长度
            peaks, prom = detect_bands_along_y_slanted(
                gel_gray, bounds, lane_index=ladder_lane-1,
                y0=y0_roi, y1=y1_roi, min_distance=30, min_prominence=1,
                target_count=K_target  # ★ 新增
            )
        else:
            # 竖直/等宽泳道：传入 target_count=标准序列长度
            lx, rx = lanes[ladder_lane-1]
            sub = gel_gray[:, lx:rx]
            peaks, prom = detect_bands_along_y_prominence(
                sub, y0=y0_roi, y1=y1_roi, min_distance=30, min_prominence=1,
                target_count=K_target  # ★ 新增
            )

        # 用于绘制：与检测结果“等长”的标签，并且 peaks 已按高度排序
        ladder_peaks_for_draw  = [int(round(p)) for p in peaks]
        ladder_labels_for_draw = sorted(ladder_labels_all, reverse=True)[:len(ladder_peaks_for_draw)]

        # —— 后续标定流程保持不变（piecewise+质量评估+渲染 …）——
        import numpy as np
        ys_sorted = sorted(int(round(float(p))) for p in peaks)
        lbs_sorted = sorted([float(x) for x in ladder_labels_all], reverse=True)
        K = min(len(ys_sorted), len(lbs_sorted))
        if len(prom) == len(peaks) and K >= 1:
            order_by_y = np.argsort(np.array(peaks, dtype=np.float64))
            w_sorted   = [float(prom[i]) for i in order_by_y]
            w_used     = w_sorted[:K]
        else:
            w_used = None

        fit_ok = False
        a, b = 1.0, 0.0
        calib_model = None
        if K >= 2:
            y_used   = [float(ys_sorted[i]) for i in range(K)]
            lbl_used = [float(lbs_sorted[i]) for i in range(K)]
            model = build_piecewise_log_mw_model(y_used, lbl_used)
            xk = model.get('xk', np.array([], dtype=np.float64))
            yk = model.get('yk', np.array([], dtype=np.float64))
            H_roi = gel_gray.shape[0]
            ok, r2, rmse = eval_fit_quality_piecewise(
                y_used, lbl_used, xk, yk, H=H_roi,
                r2_min=0.5, rmse_frac_max=0.02, rmse_abs_min_px=80.0
            )
            if ok and xk.size >= 2:
                calib_model = {'xk': xk, 'yk': yk}
                fit_ok = True
                a_fit, b_fit = fit_log_mw_irls(y_used, lbl_used, w_used, iters=6)
                a, b = a_fit, b_fit

        # 6) 核心底图（侧标沿检测真 y；刻度在拟合通过时绘制）
        if bounds is not None:
            res = render_annotation_slanted(
                gel_bgr, bounds, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, (tick_labels if fit_ok else []),
                yaxis_side=self.var_axis.get()
            )
        else:
            res = render_annotation(
                gel_bgr, lanes, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, (tick_labels if fit_ok else []),
                yaxis_side=self.var_axis.get()
            )
        annotated_core = res if not (isinstance(res, tuple) and len(res) == 2) else res[0]

        # 7) 核心（无绿线）
        annotated_core_no_green = annotated_core
        self.render_cache = {
            "core_no_green": annotated_core_no_green,
            "gi": int(self.gi),
            "gel_bgr": gel_bgr,
            "bounds": bounds,
            "lanes": lanes,
            "a": float(a), "b": float(b),
            "fit_ok": bool(fit_ok),
            "tick_labels": tick_labels if fit_ok else [],
            "yaxis_side": self.var_axis.get(),
            "calib_model": calib_model,
            "annotated_base_no_green": None,
            "annotated_base_with_green": None,
        }

        # 8) 顶部列名白板（可选）
        annotated_final_base = annotated_core_no_green
        panel_top_h = 0
        if self.var_labels_on.get():
            nlanes_val = int(self.var_nlanes.get())
            ladder_lane_val = max(1, min(int(self.var_ladder_lane.get()), nlanes_val))
            n_nonladder = max(0, nlanes_val - 1)
            names_seq = (self.lane_names or [])
            names_use = (names_seq + [""] * n_nonladder)[:n_nonladder]
            labels_table = [names_use]
            if labels_table and any(((t or "").strip() for t in labels_table[0])):
                H0 = annotated_final_base.shape[0]
                annotated_final_base = self._attach_labels_panel(
                    img_bgr=annotated_final_base,
                    lanes=lanes, bounds=bounds,
                    labels_table=labels_table,
                    gel_bgr=gel_bgr,
                    ladder_lane=ladder_lane_val,
                    yaxis_side=self.var_axis.get()
                )
                panel_top_h = annotated_final_base.shape[0] - H0

        # 计算左白板宽度（仅在加头部面板后计算，避免被右侧拼接影响）
        panel_w_val = max(0, annotated_final_base.shape[1] - gel_bgr.shape[1])

        # 9) 绿线（仅绿线）
        annotated_final_with_green = self._draw_overlays_on_core(
            img_core=annotated_final_base, gel_bgr=gel_bgr,
            overlay=None, bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            tick_labels=self.render_cache["tick_labels"],
            yaxis_side=self.var_axis.get(),
            show_green=True,
            y_offset=panel_top_h
        )

        # 10) 底部备注
        note_raw = getattr(self, "bottom_note_text", "") or ""
        annotated_final_base_with_note = self._attach_bottom_note_panel(
            annotated_final_base, note_raw, allow_expand_width=True
        )
        annotated_final_with_green_with_note = self._attach_bottom_note_panel(
            annotated_final_with_green, note_raw, allow_expand_width=True
        )

        # >>> Right-side table：在添加底注之后拼接到右侧
        rtxt = (getattr(self, "right_table_text", "") or "").strip()
        if rtxt:
            cfg = (getattr(self, "right_table_opts", None) or {})
            annotated_final_base_with_note = self._attach_right_table_panel(
                annotated_final_base_with_note,
                rtxt,
                has_header=bool(cfg.get("has_header", True)),
                show_grid=bool(cfg.get("grid", True)),
                align=str(cfg.get("align", "center")),
                cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                panel_top_h=panel_top_h
            )
            annotated_final_with_green_with_note = self._attach_right_table_panel(
                annotated_final_with_green_with_note,
                rtxt,
                has_header=bool(cfg.get("has_header", True)),
                show_grid=bool(cfg.get("grid", True)),
                align=str(cfg.get("align", "center")),
                cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                panel_top_h=panel_top_h
            )

        # 11) 左侧 ROI 预览
        self._set_autofit_image(self.canvas_roi_wb, gel_bgr)

        # 12) 右侧交互画布 —— 显式传入 calib_model（左板宽 panel_w_val 不受右侧表格影响）
        lane_marks_input = self.lane_marks or []
        ladder_lane_val = max(1, min(int(self.var_ladder_lane.get()), int(self.var_nlanes.get())))
        self.canvas_anno.set_scene(
            base_img_bgr=annotated_final_base_with_note,
            gel_bgr=gel_bgr,
            bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            nlanes=int(self.var_nlanes.get()),
            ladder_lane=ladder_lane_val,
            yaxis_side=self.var_axis.get(),
            lane_marks=lane_marks_input,
            panel_top_h=panel_top_h,
            panel_w=panel_w_val,     # 只统计左白板宽度
            calib_model=calib_model,
            reset_positions=True
        )
        if bool(self.var_show_boxes.get()):
            try: self.canvas_anno.set_boxes_enabled(True)
            except Exception: pass
        if bool(self.var_show_green.get()):
            self.canvas_anno.update_base_image(annotated_final_with_green_with_note)

        # 13) 更新缓存导出图（已包含底注与右侧表格）
        self.render_cache.update({
            "annotated_base_no_green": annotated_final_base_with_note,
            "annotated_base_with_green": annotated_final_with_green_with_note,
        })

        if not fit_ok:
            from tkinter import messagebox
            messagebox.showinfo(
                "Notice",
                "The Y-axis molecular weight scale was not plotted this time (fitting failed quality control), and no draggable arrow is generated on the right side."
            )

    def recompose_using_cache(self):
        
        try: self._unset_composed_preview()
        except Exception: pass

        rc = getattr(self, "render_cache", None) or {}
        core = rc.get("core_no_green", None)
        if core is None or not isinstance(core, np.ndarray) or core.size == 0:
            try:
                self.render_current()
            except Exception:
                pass
            return

        gel_bgr = rc.get("gel_bgr", None)
        bounds = rc.get("bounds", None)
        lanes = rc.get("lanes", None)
        a = rc.get("a", 1.0)
        b = rc.get("b", 0.0)
        fit_ok = bool(rc.get("fit_ok", False))
        tick_lbls = rc.get("tick_labels", [])
        yaxis_side = rc.get("yaxis_side", "left")
        calib_model = rc.get("calib_model", None)

        # 1) 顶部列名（如启用）
        base_img = core
        panel_top_h = 0
        if self.var_labels_on.get():
            nlanes_val = int(self.var_nlanes.get())
            ladder_lane_val = max(1, min(int(self.var_ladder_lane.get()), nlanes_val))
            n_nonladder = max(0, nlanes_val - 1)
            names_seq = (self.lane_names or [])
            names_use = (names_seq + [""] * n_nonladder)[:n_nonladder]
            labels_table = [names_use]
            if labels_table and any(((t or "").strip() for t in labels_table[0])):
                H0 = base_img.shape[0]
                base_img = self._attach_labels_panel(
                    img_bgr=base_img,
                    lanes=lanes, bounds=bounds,
                    labels_table=labels_table,
                    gel_bgr=gel_bgr,
                    ladder_lane=ladder_lane_val,
                    yaxis_side=yaxis_side
                )
                panel_top_h = base_img.shape[0] - H0

        # 面板宽（左白板；此时尚未拼右侧表格）
        try:
            panel_w_val = max(0, base_img.shape[1] - gel_bgr.shape[1])
        except Exception:
            panel_w_val = 0

        # 2) 绿线（仅绿线）
        want_green = bool(self.var_show_green.get())
        if want_green:
            with_green = self._draw_overlays_on_core(
                img_core=base_img, gel_bgr=gel_bgr,
                overlay=None, bounds=bounds, lanes=lanes,
                a=a, b=b, fit_ok=fit_ok,
                tick_labels=tick_lbls,
                yaxis_side=yaxis_side,
                show_green=True,
                y_offset=panel_top_h
            )
        else:
            with_green = None

        # 3) 底部备注
        note_raw = getattr(self, "bottom_note_text", "") or ""
        base_with_note = self._attach_bottom_note_panel(base_img, note_raw, allow_expand_width=True)
        if want_green and with_green is not None:
            with_green_note = self._attach_bottom_note_panel(with_green, note_raw, allow_expand_width=True)
        else:
            with_green_note = None

        # >>> Right-side table：在添加底注之后拼接到右侧
        rtxt = (getattr(self, "right_table_text", "") or "").strip()
        if rtxt:
            cfg = (getattr(self, "right_table_opts", None) or {})
            base_with_note = self._attach_right_table_panel(
                base_with_note,
                rtxt,
                has_header=bool(cfg.get("has_header", True)),
                show_grid=bool(cfg.get("grid", True)),
                align=str(cfg.get("align", "center")),
                cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                panel_top_h=panel_top_h
            )
            if want_green and with_green_note is not None:
                with_green_note = self._attach_right_table_panel(
                    with_green_note,
                    rtxt,
                    has_header=bool(cfg.get("has_header", True)),
                    show_grid=bool(cfg.get("grid", True)),
                    align=str(cfg.get("align", "center")),
                    cap_px=int(cfg.get("cap_px", self._get_design_params().get("bottom_cap_px", 28))),
                    panel_top_h=panel_top_h
                )

        # 4) 刷新右侧交互画布（panel_w 只给左白板，不含右侧表格宽度）
        ladder_lane_val = max(1, min(int(self.var_ladder_lane.get()), int(self.var_nlanes.get())))
        self.canvas_anno.set_scene(
            base_img_bgr=base_with_note,
            gel_bgr=gel_bgr,
            bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            nlanes=int(self.var_nlanes.get()),
            ladder_lane=ladder_lane_val,
            yaxis_side=yaxis_side,
            lane_marks=(self.lane_marks or []),
            panel_top_h=panel_top_h,
            panel_w=panel_w_val,
            calib_model=calib_model,
        )
        if bool(self.var_show_boxes.get()):
            try: self.canvas_anno.set_boxes_enabled(True)
            except Exception: pass
        if want_green and with_green_note is not None:
            self.canvas_anno.update_base_image(with_green_note)

        # 5) 更新可导出的缓存
        self.render_cache.update({
            "annotated_base_no_green": base_with_note,
            "annotated_base_with_green": with_green_note or base_with_note,
        })



    # -------- 保留：上方白底文本标签面板（与现有实现相同，略） -------- #
    def _attach_labels_panel(
        self,
        img_bgr: np.ndarray,
        lanes: list[tuple[int, int]] | None,
        bounds: np.ndarray | None,
        labels_table: list[list[str]],
        gel_bgr: np.ndarray,
        ladder_lane: int,
        yaxis_side: str
    ) -> np.ndarray:
        """
        顶部白底竖排泳道标注：随“胶区高度 Hg”按 1000px 基准缩放。
        """
        import numpy as np, cv2
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        H, W_total = img_bgr.shape[:2]
        Hg, Wg = gel_bgr.shape[:2]

        # === 基于胶区高度的缩放因子 ===
        s = max(0.35, float(Hg) / 1000.0)

        panel_w  = max(0, W_total - Wg)
        x_offset = panel_w if (str(yaxis_side).lower() == "left") else 0
        rows = len(labels_table)
        if rows == 0:
            return img_bgr
        cols_use = len(labels_table[0]) if rows else 0
        if cols_use == 0:
            return img_bgr

        # 真实道数与中点
        if bounds is not None and isinstance(bounds, np.ndarray):
            real_nlanes = max(0, bounds.shape[1] - 1)
        elif lanes is not None:
            real_nlanes = len(lanes)
        else:
            real_nlanes = cols_use + 1

        centers_mid_all, widths_all, lefts_all, rights_all = [], [], [], []
        if (bounds is not None and isinstance(bounds, np.ndarray)
            and bounds.ndim == 2 and bounds.shape[1] >= real_nlanes + 1):
            yc = int(min(bounds.shape[0] - 1, max(0, H // 2)))
            for i in range(real_nlanes):
                L = int(bounds[yc, i]); R = int(bounds[yc, i + 1])
                centers_mid_all.append(int(round((L + R) / 2.0)))
                lefts_all.append(L); rights_all.append(R)
                widths_all.append(max(1, R - L))
        elif lanes is not None:
            for (l, r) in lanes[:real_nlanes]:
                L, R = int(l), int(r)
                centers_mid_all.append(int(round((L + R) / 2.0)))
                lefts_all.append(L); rights_all.append(R)
                widths_all.append(max(1, R - L))
        else:
            step = max(1.0, Wg / max(1, real_nlanes))
            for i in range(real_nlanes):
                L = int(round(i * step))
                R = int(round((i + 1) * step))
                centers_mid_all.append(int(round((L + R) / 2.0)))
                lefts_all.append(L); rights_all.append(R)
                widths_all.append(max(1, R - L))

        centers_mid_all = [int(x_offset + np.clip(c, 0, Wg - 1)) for c in centers_mid_all]
        lefts_all  = [int(x_offset + max(0, L)) for L in lefts_all]
        rights_all = [int(x_offset + min(Wg, R)) for R in rights_all]
        widths_all = [max(1, r - l) for l, r in zip(lefts_all, rights_all)]

        skip_idx = max(0, int(ladder_lane) - 1)  # 标准道（0-based）
        target_lane_idx = [i for i in range(real_nlanes) if i != skip_idx]

        # === 文本与版式（统一按 s 缩放）===
        font      = cv2.FONT_HERSHEY_SIMPLEX
        thick     = max(1, int(round(2 * s)))
        cap_px    = max(10, int(round(26 * s)))      # 原基准=26px -> 随 s
        h_margin  = max(4,  int(round(8  * s)))
        top_pad   = max(4,  int(round(8  * s)))
        bot_pad   = max(4,  int(round(10 * s)))
        row_gap   = max(2,  int(round(6  * s)))
        txt_pad   = max(1,  int(round(3  * s)))
        _ROT_FLAG = cv2.ROTATE_90_COUNTERCLOCKWISE

        base_scale = self._font_scale_for_cap_height_px(cap_px, font=font, thickness=thick)

        def render_rotated_text_img(text: str, lane_idx: int):
            text = (text or "").strip()
            if not text:
                return None, 0, 0, 0.0
            lane_w = max(1, widths_all[lane_idx] - 2 * h_margin)

            scale = base_scale
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
            # 旋转后高度≈th，若超过列宽则等比缩小
            if th > lane_w:
                scale = max(0.35, scale * (lane_w / (th + 1e-6)))
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

            w_horiz = max(1, tw + 2 * txt_pad)
            h_horiz = max(1, th + baseline + 2 * txt_pad)
            img = np.full((h_horiz, w_horiz, 3), 255, dtype=np.uint8)
            org = (txt_pad, txt_pad + th)
            cv2.putText(img, text, org, font, scale, (0, 0, 0), thick, cv2.LINE_AA)

            rot = cv2.rotate(img, _ROT_FLAG)
            rot_gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
            nz = np.where(rot_gray < 252)
            if nz[0].size > 0:
                y_min, y_max = int(nz[0].min()), int(nz[0].max())
                x_min, x_max = int(nz[1].min()), int(nz[1].max())
                y_min = max(0, y_min - 1); y_max = min(rot.shape[0] - 1, y_max + 1)
                x_min = max(0, x_min - 1); x_max = min(rot.shape[1] - 1, x_max + 1)
                rot = rot[y_min:y_max + 1, x_min:x_max + 1]
            rot = cv2.copyMakeBorder(rot, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            h_rot, w_rot = rot.shape[:2]
            return rot, w_rot, h_rot, float(scale)

        # 行测量与缓存
        rows_used_idx, row_heights = [], []
        cell_cache: dict[tuple[int, int], tuple[np.ndarray | None, int, int, float]] = {}

        STD_TEXT = "Marker"
        for r in range(rows):
            row_has_text = any((labels_table[r][c] or "").strip() for c in range(cols_use))
            if not row_has_text:
                row_heights.append(0)
                continue
            max_h_rot = 0
            for c in range(min(cols_use, len(target_lane_idx))):
                text = str(labels_table[r][c] or "").strip()
                if not text: 
                    continue
                li = target_lane_idx[c]
                cell = render_rotated_text_img(text, li)
                cell_cache[(r, li)] = cell
                _, w_rot, h_rot, _ = cell
                max_h_rot = max(max_h_rot, h_rot)

            if 0 <= skip_idx < real_nlanes:
                cell_std = render_rotated_text_img(STD_TEXT, skip_idx)
                cell_cache[(r, skip_idx)] = cell_std
                _, w_rot_s, h_rot_s, _ = cell_std
                max_h_rot = max(max_h_rot, h_rot_s)

            row_heights.append(max_h_rot + 2)
            rows_used_idx.append(r)

        if not rows_used_idx:
            return img_bgr

        panel_h = top_pad + sum(row_heights[r] for r in rows_used_idx) \
                + (len(rows_used_idx) - 1) * row_gap + bot_pad
        panel = np.full((panel_h, W_total, 3), 255, dtype=np.uint8)

        y_cursor = top_pad
        for r in range(rows):
            if r not in rows_used_idx:
                continue
            rh = row_heights[r]
            # 非标准道
            for c in range(min(cols_use, len(target_lane_idx))):
                text = str(labels_table[r][c] or "").strip()
                if not text: 
                    continue
                li = target_lane_idx[c]
                rot_img, w_rot, h_rot, _ = cell_cache.get((r, li), (None, 0, 0, 0.0))
                if rot_img is None or w_rot <= 0 or h_rot <= 0:
                    continue
                lane_center = centers_mid_all[li]
                x_left = int(np.clip(lane_center - w_rot // 2, 2, W_total - w_rot - 2))
                y_top = int(y_cursor + (rh - h_rot))
                y1 = max(0, y_top); y2 = min(panel_h, y_top + h_rot)
                x1 = max(0, x_left); x2 = min(W_total, x_left + w_rot)
                if y2 > y1 and x2 > x1:
                    sub_h, sub_w = y2 - y1, x2 - x1
                    panel[y1:y2, x1:x2] = rot_img[0:sub_h, 0:sub_w]

            # 标准道
            if 0 <= skip_idx < real_nlanes and any((labels_table[r][c] or "").strip() for c in range(cols_use)):
                rot_img_s, w_rot_s, h_rot_s, _ = cell_cache.get((r, skip_idx), (None, 0, 0, 0.0))
                if rot_img_s is not None and w_rot_s > 0 and h_rot_s > 0:
                    lane_center_s = centers_mid_all[skip_idx]
                    x_left_s = int(np.clip(lane_center_s - w_rot_s // 2, 2, W_total - w_rot_s - 2))
                    y_top_s = int(y_cursor + (rh - h_rot_s))
                    y1 = max(0, y_top_s); y2 = min(panel_h, y_top_s + h_rot_s)
                    x1 = max(0, x_left_s); x2 = min(W_total, x_left_s + w_rot_s)
                    if y2 > y1 and x2 > x1:
                        sub_h, sub_w = y2 - y1, x2 - x1
                        panel[y1:y2, x1:x2] = rot_img_s[0:sub_h, 0:sub_w]

            y_cursor += rh + row_gap

        return np.vstack([panel, img_bgr])

    def _expand_tabs_for_cv2(self, text: str, tab_size: int = 4) -> str:
            """
            由于 cv2.putText 不支持 \t，这里把 Tab 展开为若干空格，使列对齐更可控。
            采用“按字符列”计数的近似方案（与等宽字体一致），满足对齐诉求且实现简单稳健。
            """
            if not text:
                return ""
            out = []
            col = 0
            for ch in text:
                if ch == '\t':
                    spaces = tab_size - (col % tab_size)
                    out.append(' ' * spaces)
                    col += spaces
                else:
                    out.append(ch)
                    # \n / \r 视为换行，重置列计数
                    if ch in ('\n', '\r'):
                        col = 0
                    else:
                        col += 1
            return ''.join(out)

   
    def _attach_bottom_note_panel(
        self,
        img_bgr: np.ndarray,
        note_text: str,
        line_gap: int = 6,
        top_pad: int = 10,
        bot_pad: int = 10,
        left_pad: int = 12,
        right_pad: int = 12,
        allow_expand_width: bool = True
    ) -> np.ndarray:
        """
        底部白底备注：随“当前图像高度 H_img”按 1000px 基准缩放。
        （若你更希望跟随胶区高度，可在此改为用缓存中的胶高。）
        """
        import numpy as np, cv2
        from PIL import Image, ImageDraw, ImageFont
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return img_bgr

        raw = (note_text or "")
        lines_raw = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        def _expand_tabs_for_cv2(text: str, tab_size: int = 4) -> str:
            if not text:
                return ""
            out, col = [], 0
            for ch in text:
                if ch == '\t':
                    spaces = tab_size - (col % tab_size)
                    out.append(' ' * spaces)
                    col += spaces
                else:
                    out.append(ch)
                    if ch in ('\n', '\r'):
                        col = 0
                    else:
                        col += 1
            return ''.join(out)

        lines = [_expand_tabs_for_cv2(ln, tab_size=4) for ln in lines_raw]
        if not any((ln.strip() for ln in lines)):
            return img_bgr

        H_img, W_img = img_bgr.shape[:2]

        # === 基于当前图像高度的缩放（H_img/1000）===
        s = max(0.35, float(H_img) / 1000.0)

        # 统一缩放基础设计参数
        design = self._get_design_params()
        base_cap = int(design.get("bottom_cap_px", 28))
        cap_px   = max(10, int(round(base_cap * s)))          # 字高
        top_pad  = max(4,  int(round(10 * s)))
        bot_pad  = max(4,  int(round(10 * s)))
        left_pad = max(6,  int(round(12 * s)))
        right_pad= max(6,  int(round(12 * s)))
        line_gap = max(2,  int(round(6 * s)))
        thickness= 2

        font_path = self._find_font_ttf()

        # Pillow 首选（支持中英混排），失败则回退 OpenCV
        if font_path:
            try:
                size_px = max(12, cap_px)
                font = ImageFont.truetype(font_path, size=size_px)

                # 逐行测量
                tmp = Image.new("RGB", (8, 8), "white")
                drw = ImageDraw.Draw(tmp)
                line_heights, line_widths = [], []
                for ln in lines:
                    text = ln if ln else " "
                    bbox = drw.textbbox((0, 0), text, font=font)
                    tw, th = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    line_heights.append(max(1, th))
                    line_widths.append(max(1, tw))

                max_line_w = max(line_widths) if line_widths else 0
                panel_h = top_pad + bot_pad
                if line_heights:
                    panel_h += sum(line_heights) + max(0, (len(line_heights) - 1) * line_gap)

                needed_w = left_pad + max_line_w + right_pad
                new_W = max(W_img, needed_w) if allow_expand_width else W_img
                if new_W > W_img:
                    pad = np.full((H_img, new_W - W_img, 3), 255, dtype=np.uint8)
                    img_main = np.concatenate([img_bgr, pad], axis=1)
                else:
                    img_main = img_bgr

                panel_img = Image.new("RGB", (new_W, panel_h), "white")
                drw = ImageDraw.Draw(panel_img)

                y = top_pad
                for i, ln in enumerate(lines):
                    text = ln if ln else " "
                    x = left_pad
                    drw.text((x, y), text, fill=(0, 0, 0), font=font)
                    y += line_heights[i] + line_gap

                panel_bgr = cv2.cvtColor(np.array(panel_img), cv2.COLOR_RGB2BGR)
                return np.vstack([img_main, panel_bgr])
            except Exception:
                pass

        # 回退：OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._font_scale_for_cap_height_px(cap_px, font=font, thickness=thickness)

        line_sizes = [cv2.getTextSize(ln if ln else " ", font, scale, thickness)[0] for ln in lines]
        line_heights = [sz[1] for sz in line_sizes]
        line_widths  = [sz[0] for sz in line_sizes]
        max_line_w = max(line_widths) if line_widths else 0

        panel_h = top_pad + bot_pad
        if line_heights:
            panel_h += sum(line_heights) + max(0, (len(line_heights) - 1) * line_gap)

        needed_w = left_pad + max_line_w + right_pad
        new_W = max(W_img, needed_w) if allow_expand_width else W_img
        if new_W > W_img:
            pad = np.full((H_img, new_W - W_img, 3), 255, dtype=np.uint8)
            img_main = np.concatenate([img_bgr, pad], axis=1)
        else:
            img_main = img_bgr

        panel = np.full((panel_h, new_W, 3), 255, dtype=np.uint8)
        y = top_pad
        for i, ln in enumerate(lines):
            text = ln if ln else " "
            (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
            x = left_pad
            y_baseline = y + th
            try:
                cv2.putText(panel, text, (x, y_baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
            except Exception:
                cv2.putText(panel, "?", (x, y_baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y += line_heights[i] + line_gap

        return np.vstack([img_main, panel])

    
    # -------------------- 其他 -------------------- #
    def export_current(self):
        """
        导出当前右侧“标注结果”：
        - 优先从右侧交互式 Canvas（self.canvas_anno）合成：包含用户拖动后的箭头位置（若有）。
        - 若交互式 Canvas 不可用或未渲染，则根据“显示绿线（分隔线）”勾选框状态，从缓存中选择【带/不带绿线】的底图导出。
        - 继续兼容旧键：若新键缺失，回退到 annotated_base / annotated_final / annotated。
        """
        # 1) 优先：交互式 Canvas -> 所见即所得（含红箭头，底图随勾选切换）
        img_to_save = None
        if hasattr(self, "canvas_anno") and hasattr(self.canvas_anno, "render_to_image"):
            try:
                img_to_save = self.canvas_anno.render_to_image()
            except Exception:
                img_to_save = None

        # 2) 回退：根据勾选框状态，从缓存选【有/无绿线】底图
        if img_to_save is None:
            if not getattr(self, "render_cache", None):
                messagebox.showwarning("Notice", "Please render the current gel block first.")
                return
            want_green = bool(self.var_show_green.get())
            key = "annotated_base_with_green" if want_green else "annotated_base_no_green"
            img_to_save = self.render_cache.get(key)

            # 再次回退：兼容旧键
            if img_to_save is None:
                # 旧版仅存“annotated_base”（通常为无绿线）
                img_to_save = self.render_cache.get("annotated_base")
            if img_to_save is None:
                # 更老版本键
                img_to_save = self.render_cache.get("annotated_final") or self.render_cache.get("annotated")

            if img_to_save is None:
                messagebox.showwarning("Notice", "No exportable image found")
                return

        # 3) 健壮性检查
        if not isinstance(img_to_save, np.ndarray) or img_to_save.size == 0:
            messagebox.showerror("Error", "Invalid image to export")
            return

        # 4) 生成默认文件名
        try:
            gi = int(self.render_cache.get("gi", getattr(self, "gi", 1)))
        except Exception:
            gi = getattr(self, "gi", 1) or 1
        default_name = f"gel{gi}_annotated.png"
        try:
            if getattr(self, "image_path", None):
                p = Path(self.image_path)
                default_name = f"{p.stem}_annotated.png"
        except Exception:
            pass

        # 5) 保存对话框
        fpath = filedialog.asksaveasfilename(
            title="Saving annotated image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png")]
        )
        if not fpath:
            return

        # 6) PNG 编码并写入（tofile 适配 Windows 中文路径）
        ok, buf = cv2.imencode(".png", img_to_save)
        if not ok:
            messagebox.showerror("Error", "PNG encoding failed, export unsuccessful.")
            return
        try:
            buf.tofile(fpath)
        except Exception:
            try:
                with open(fpath, "wb") as f:
                    f.write(buf.tobytes())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write into file:{e}")
                return

        messagebox.showinfo("Complete", f"Exported:{fpath}")

    # ----- 新增/改造：角度重置 & 列名+分子量编辑器 ----- #
    def reset_angle(self):
        """
        将右侧 ROI 编辑器中的“校准线”角度重置为 0°（水平），并立即重绘。
        """
        self._clear_all_caches_and_overlays()
        if getattr(self, "roi_editor", None) is None:
            return
        # 优先调用 ROIEditorCanvas 的公有方法（若无则直接设置属性）
        if hasattr(self.roi_editor, "reset_angle") and callable(self.roi_editor.reset_angle):
            self.roi_editor.reset_angle()
        else:
            try:
                self.roi_editor.cal_angle_deg = 0.0
                self.roi_editor.redraw()
            except Exception:
                pass

    def open_labels_editor(self):
        """
        编辑“每列：列名 + 分子量列表”的简化录入。
        - 提交后不再全量渲染，而是基于缓存核心图快速组合顶部面板与备注；
        - 依旧支持空行（不标注）。
        """
        #self._soft_reset_for_operation()
        win = tk.Toplevel(self)
        win.title("Edit column names and molecular weights (one column per line; empty line = no label).")
        win.transient(self); win.grab_set()
        frm = ttk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(
            frm,
            text="Example:\nLane 1: 70,55,40\nSample A 100 70 35\nB; 60;30;10\n (*First item = column name; empty line = no label; the rest are molecular weights in kDa)",
            justify="left"
        ).pack(anchor="w")
        txt = tk.Text(frm, height=10, width=60, wrap="none")
        txt.pack(fill=tk.X, pady=6)

        def _fmt_num(x: float) -> str:
            return f"{x:g}"

        # 预填：严格回显（包含空行）
        pre_lines = []
        N = max(len(self.lane_names), len(self.lane_marks))
        for i in range(N):
            name = (self.lane_names[i] if i < len(self.lane_names) else "").strip()
            marks = self.lane_marks[i] if i < len(self.lane_marks) else []
            ms = ", ".join(_fmt_num(v) for v in (marks or []))
            line = (name + (": " + ms if ms else "")).strip()
            pre_lines.append(line)
        if pre_lines:
            txt.insert("1.0", "\n".join(pre_lines))

        def parse_lane_meta(raw: str) -> tuple[list[str], list[list[float]]]:
            lines = (raw or "").splitlines()  # 不丢弃空行
            names: list[str] = []
            marks: list[list[float]] = []
            def _split_tokens(s: str) -> list[str]:
                s = (s.replace("，", ",").replace("；", ";")
                     .replace("、", ",").replace("\t", ",")
                     .replace(":", ",").replace("：", ","))
                s = s.replace(";", ",")
                toks = [t.strip() for t in s.split(",")]
                return [t for t in toks if t]
            for ln in lines:
                if not ln.strip():  # 空行 ⇒ 不标注列
                    names.append("")
                    marks.append([])
                    continue
                toks = _split_tokens(ln)
                if not toks:
                    names.append("")
                    marks.append([])
                    continue
                name = toks[0]
                nums: list[float] = []
                for t in toks[1:]:
                    try:
                        v = float(t)
                        if np.isfinite(v) and v > 0:
                            nums.append(v)
                    except Exception:
                        pass
                names.append(name)
                marks.append(nums)
            return names, marks

        def do_ok():
            raw = txt.get("1.0", "end")
            names, marks = parse_lane_meta(raw)
            self.lane_names = names
            self.lane_marks = marks
            win.destroy()
            try:
                self.recompose_using_cache()
            except Exception:
                # 回退完整渲染
                try: self.render_current()
                except Exception: pass

        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btns, text="Confirm", command=do_ok).pack(side=tk.RIGHT)


# ===== 在 App 类定义结束后新增：全局字体初始化 =====
# ===== 在 App 类定义结束后 =====
def setup_global_fonts(root):
    """
    Force Tk/ttk named fonts to use Arial (fallback to Helvetica/Segoe UI/Tk default if Arial not installed).
    Changing named fonts takes effect on existing widgets immediately.
    """
    import tkinter.font as tkfont
    try:
        families = {f.lower() for f in tkfont.families(root)}
        def pick(*names):
            for n in names:
                if n and n.lower() in families:
                    return n
            return None

        # ★ 强制优先 Arial（若无则回退）
        base_family = pick('Arial') or pick('Helvetica', 'Segoe UI') \
                       or tkfont.nametofont('TkDefaultFont').cget('family')

        # ★ UI 基础字号由 10 → 11（略微放大）
        base_size = 11

        for name in ('TkDefaultFont', 'TkTextFont', 'TkFixedFont',
                     'TkMenuFont', 'TkHeadingFont', 'TkIconFont', 'TkTooltipFont'):
            try:
                f = tkfont.nametofont(name)
                f.configure(family=base_family, size=base_size)
            except Exception:
                pass

        # ttk 默认字体
        try:
            from tkinter import ttk
            ttk.Style().configure('.', font=(base_family, base_size))
        except Exception:
            pass
    except Exception:
        pass

if __name__ == "__main__":
    app = App()
    # ★ 全局字体：宋体优先，退化到 Arial（对现有控件实时生效）
    setup_global_fonts(app)
    app.mainloop()
