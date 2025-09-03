# app_desktop_gui_roi.py
# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import json
from gel_core import (
    auto_white_balance, detect_gel_regions,
    lanes_by_projection, lanes_uniform,
    detect_bands_along_y,  # 旧的直线模式备用
    render_annotation,     # 直立矩形渲染
    # 新增：斜线直线模式
    lanes_slanted,
    detect_bands_along_y_slanted,
    detect_bands_along_y_prominence,
    render_annotation_slanted,
    match_ladder_best,
    fit_log_mw_irls,
    eval_fit_quality
)
from roi_editor import ROIEditorCanvas


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

    def set_scene(
        self,
        base_img_bgr: np.ndarray,
        gel_bgr: np.ndarray,
        bounds: np.ndarray | None,
        lanes: list[tuple[int, int]] | None,
        a: float, b: float, fit_ok: bool,
        nlanes: int, ladder_lane: int,
        yaxis_side: str,
        lane_marks: list[list[float]] | None,
        panel_top_h: int = 0,
        panel_w: int | None = None,      # ★ 新增：显式传入左侧刻度白板宽度
    ):
        """设置底图与几何信息，并按 lane_marks 生成可拖拽红箭头。"""
        # 清空画布元素与缓存
        self.delete("all")
        self.img_item = None
        self.base_img_tk = None
        self.arrows.clear()

        # 清空方框集合
        self._delete_all_boxes()  # ← 统一删除（安全起见）
        self.boxes.clear()
        self._box_drag.update({"box": None, "mode": None, "corner": None})

        # 基础数据
        self.base_img_bgr = base_img_bgr
        self.gel_bgr = gel_bgr.copy() if isinstance(gel_bgr, np.ndarray) else None
        self.gel_gray = cv2.cvtColor(self.gel_bgr, cv2.COLOR_BGR2GRAY) if self.gel_bgr is not None else None
        self.gel_size = gel_bgr.shape[:2]  # (Hg, Wg)
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

        # ★ 关键：panel_w 使用“显式传入值”，否则才退回到 Wt - Wg 的估算
        if panel_w is not None:
            self.panel_w = max(0, int(panel_w))
        else:
            self.panel_w = max(0, Wt - Wg)

        self.x_offset = self.panel_w if self.yaxis_side == "left" else 0

        # 渲染底图
        self._render_base_image()

        # 生成箭头
        if self.fit_ok and lane_marks:
            self._create_arrows_from_marks(lane_marks)

        # ★ 若当前启用了“显示方框”，则基于最新箭头一次性生成
        if self.boxes_enabled and self.arrows:
            self._delete_all_boxes()  # 防止残留复用
            for meta in self.arrows:
                self._create_box_for_arrow(meta)


    def render_to_image(self) -> np.ndarray | None:
        """把当前箭头叠加到底图像素，返回BGR图（用于导出）。"""
        if self.base_img_bgr is None:
            return None
        img = self.base_img_bgr.copy()
        H, W = img.shape[:2]

        # 三角箭头（像素坐标系，未缩放的固定尺寸）
        w, h = 40, 10
        for a in self.arrows:
            y = int(np.clip(a["y_img"], 0, H - 1))
            if "x_img" in a:
                x = int(np.clip(a["x_img"], 0, W - 1))
            else:
                lane_idx = a["lane_idx"]
                xc_gel = self._lane_center_x_at_y_gel(y - self.panel_top_h, lane_idx)
                x = int(np.clip(self.x_offset + xc_gel, 0, W - 1))
            tip = (x , y)
            p1 = (tip[0] - w // 2, int(np.clip(tip[1] + h, 0, H - 1)))
            p2 = (tip[0] - w // 2, int(np.clip(tip[1] - h, 0, H - 1)))
            pts = np.array([p1, p2, tip], dtype=np.int32)
            cv2.fillPoly(img, [pts], (0, 0, 255))
            cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255),
                          thickness=1, lineType=cv2.LINE_AA)
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
        self.base_img_tk = ImageTk.PhotoImage(pil)  # 保持强引用，防止GC

        # 尝试复用旧的 image item，如已失效则创建新的
        try:
            if self.img_item is None:
                self.img_item = self.create_image(self.ox, self.oy, image=self.base_img_tk, anchor="nw")
            else:
                self.coords(self.img_item, self.ox, self.oy)
                self.itemconfig(self.img_item, image=self.base_img_tk)
        except tk.TclError:
            # 旧 id 已无效，重建
            self.img_item = self.create_image(self.ox, self.oy, image=self.base_img_tk, anchor="nw")

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

    def _create_arrows_from_marks(self, lane_marks: list[list[float]]):
        """依据 lane_marks 在非标准道生成箭头。"""
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
                y_gel = int(round(self.a * np.log10(v) + self.b))
                y_img = self.panel_top_h + y_gel
                xc_gel = self._lane_center_x_at_y_gel(y_gel, lane_idx)
                x_img = self.x_offset + xc_gel
                self._add_arrow(x_img, y_img, lane_idx, v)

    def _arrow_canvas_points(self, x_img: float, y_img: float) -> list[float]:
        """返回画布坐标下的三角形顶点序列（flat list）。"""
        # 以像素坐标为准的箭头尺寸（未缩放）
        w_px, h_px = 40, 10
        tip_img = (x_img, y_img)
        p1_img = (tip_img[0] - w_px // 2, y_img + h_px)
        p2_img = (tip_img[0] - w_px // 2, y_img - h_px)
        pts_img = [p1_img, p2_img, tip_img]
        pts_canvas = []
        for (x, y) in pts_img:
            cx, cy = self._to_canvas(x, y)
            pts_canvas.extend([cx, cy])
        return pts_canvas



    def _add_arrow(self, x_img: float, y_img: float, lane_idx: int, mw: float, color: tuple[int, int, int] = (255, 64, 64)):
            """在Canvas上新增一个箭头并绑定拖拽事件，同时创建配套红色可调方框+黄色灰度和。"""
            # Tk 颜色（BGR -> #RRGGBB）
            r, g, b = (255, 64, 64)
            pts = self._arrow_canvas_points(x_img, y_img)
            aid = self.create_polygon(pts, fill=f"#{r:02x}{g:02x}{b:02x}", outline="#ffffff", width=1, smooth=False)
            meta = {"id": aid, "lane_idx": lane_idx, "x_img": float(x_img), "y_img": float(y_img), "mw": float(mw)}
            self.arrows.append(meta)

            # 仅绑定该箭头的拖拽事件（竖向拖拽）
            self.tag_bind(aid, "<ButtonPress-1>", lambda e, a=meta: self._on_drag_start(e, a))
            self.tag_bind(aid, "<B1-Motion>",     lambda e, a=meta: self._on_drag_move(e, a))
            self.tag_bind(aid, "<ButtonRelease-1>", lambda e, a=meta: self._on_drag_end(e, a))

            # === 新增：为该箭头同步创建一个红色可调矩形框 + 顶部黄色文字 ===
            #self._create_box_for_arrow(meta)

    def _redraw_arrows(self):
        """画布变化时，按照最新缩放/偏移重画箭头形状。"""
        for a in self.arrows:
            lane_idx = a["lane_idx"]
            y_img = a["y_img"]
            if "x_img" in a:
                x_img = float(a["x_img"])
            else:
                xc_gel = self._lane_center_x_at_y_gel(int(round(y_img - self.panel_top_h)), lane_idx)
                x_img = float(self.x_offset + xc_gel)
            # 限制在整张图可见区域（你也可改成仅限 gel 区域）
            H, W = self.base_img_bgr.shape[:2]
            x_img = float(np.clip(x_img, 0, W - 1))
            pts = self._arrow_canvas_points(x_img, y_img)
            self.coords(a["id"], *pts)


# === 方框：创建/重绘/事件/灰度和 ===
    def _create_box_for_arrow(self, arrow_meta: dict):
        """
        基于箭头位置，创建默认尺寸的红色可调矩形框及顶部黄色文字。
        初始策略：在箭头尖端左侧，宽40px、高24px，纵向居中于箭头。
        """
        if self.base_img_bgr is None:
            return

        # 箭头尖端在像素坐标（未缩放）
        x_img = float(arrow_meta.get("x_img", 0.0))
        y_img = float(arrow_meta.get("y_img", 0.0))

        # 初始尺寸与位置（放在箭头尖端左侧）
        w0, h0 = 40.0, 24.0
        margin = 6.0
        # 箭头的“tip”在 _arrow_canvas_points 里是 (x_img-20, y_img)
        tip_x = x_img
        bx = tip_x - margin - w0 +40  # 左上 x
        by = y_img - h0 / 2.0      # 左上 y

        # 限制在整图范围内
        H, W = self.base_img_bgr.shape[:2]
        bx = float(np.clip(bx, 0, max(0, W - w0)))
        by = float(np.clip(by, 0, max(0, H - h0)))

        rect_id = self.create_rectangle(0, 0, 0, 0, outline="#ff4545", width=2)
        # 四角手柄（小方块）
        hs = 5
        handles = {
            "nw": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "ne": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "se": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
            "sw": self.create_rectangle(0, 0, 0, 0, outline="#ff4545", fill="#ff4545", width=1),
        }
        text_id = self.create_text(0, 0, text="", fill="#ffd800", anchor="s")  # s=底对齐，显示在框上方

        box = {
            "rect_id": rect_id,
            "text_id": text_id,
            "handles": handles,
            "x_img": bx, "y_img": by, "w": w0, "h": h0,
            "arrow": arrow_meta
        }
        self.boxes.append(box)

        # 初次定位/绘制
        self._draw_box(box)

        # 绑定事件：矩形整体拖拽
        self.tag_bind(rect_id, "<ButtonPress-1>",  lambda e, b=box: self._on_box_press(e, b, mode="move", corner=None))
        self.tag_bind(rect_id, "<B1-Motion>",      lambda e, b=box: self._on_box_drag(e, b))
        self.tag_bind(rect_id, "<ButtonRelease-1>", lambda e, b=box: self._on_box_release(e, b))

        # 手柄拖拽（四角缩放）
        self.tag_bind(handles["nw"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="nw"))
        self.tag_bind(handles["ne"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="ne"))
        self.tag_bind(handles["se"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="se"))
        self.tag_bind(handles["sw"], "<ButtonPress-1>", lambda e, b=box: self._on_box_press(e, b, mode="resize", corner="sw"))

        for c in ("nw","ne","se","sw"):
            hid = handles[c]
            self.tag_bind(hid, "<B1-Motion>",      lambda e, b=box: self._on_box_drag(e, b))
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
        """根据当前缩放/偏移，将 box 的图像坐标绘制为 Canvas 坐标，并更新黄色文字（灰度和）。"""
        x, y, w, h = box["x_img"], box["y_img"], box["w"], box["h"]
        x1, y1 = self._to_canvas(x, y)
        x2, y2 = self._to_canvas(x + w, y + h)
        # 更新矩形
        self.coords(box["rect_id"], x1, y1, x2, y2)

        # 更新手柄（四角小方块）
        hs = 2
        corners = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "se": (x2, y2),
            "sw": (x1, y2),
        }
        for k, (cx, cy) in corners.items():
            self.coords(box["handles"][k], cx - hs, cy - hs, cx + hs, cy + hs)

        # 计算灰度和，并更新文字在矩形上方居中（anchor="s"）
        val = self._gray_sum_in_box(box)
        txt = f"{int(val)}"
        cx = (x1 + x2) / 2.0
        ty = y1 - 4  # 略高于顶部
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
        """拖拽过程：平移或按角缩放。"""
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
            # 按角缩放：根据 corner 决定哪个角固定
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

    # ---------- 交互：仅允许垂直拖动 ----------
    def _on_drag_start(self, evt, arrow_meta: dict):
        self._drag["id"] = arrow_meta["id"]
        self._drag["y0_canvas"] = evt.y


    def _on_drag_move(self, evt, arrow_meta: dict):
        if self._drag["id"] != arrow_meta["id"]:
            return
        # 画布坐标 → 图像坐标
        x_img, y_img = self._to_img(evt.x, evt.y)

        # 边界：Y 仍限制在 gel 有效高度（不进入上方白板），X 限制在整图宽
        H, W = self.base_img_bgr.shape[:2]
        Hg, Wg = self.gel_size
        ymin = self.panel_top_h
        ymax = self.panel_top_h + Hg - 1
        x_img = float(np.clip(x_img, 0, W - 1))
        y_img = float(np.clip(y_img, ymin, ymax))

        # 保存坐标并重绘箭头
        arrow_meta["x_img"] = x_img
        arrow_meta["y_img"] = y_img
        pts = self._arrow_canvas_points(x_img, y_img)
        self.coords(arrow_meta["id"], *pts)




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Electrophoresis image visual processing")
        self.geometry("1000x720")

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
        # 显示信息
        note = match.get("note", "").strip()
        info = f"条目：{name}｜条数：{len(vals)}｜值：{', '.join(f'{v:g}' for v in vals)}"
        if note:
            info += f"\n备注：{note}"
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
        ttk.Label(
            f_file,
            text=("Tip: In the right canvas, drag ROI corners/edges to crop; wheel to zoom; right-drag to pan; "
                "arrow keys for fine adjustments.\n"
                "ROI now follows the calibration line angle for accurate selection on tilted gels."),
            wraplength=self.LEFT_WIDTH-24,
            justify="left"
        ).pack(anchor="w", padx=6, pady=(0,6))

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
        ttk.Checkbutton(f_marker, text="Show green separators", variable=self.var_show_green,
                        command=self.on_toggle_show_green).pack(anchor="w", padx=6, pady=2)

        self.var_show_boxes = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f_marker, text="Show red boxes (generate from arrows when checked)",
            variable=self.var_show_boxes, command=self.on_toggle_show_boxes
        ).pack(anchor="w", padx=6, pady=2)

        # ---- Actions ----
        f_action = ttk.Frame(left)
        f_action.pack(fill=tk.X, pady=10)
        ttk.Button(f_action, text="Render current ROI", command=self.render_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="Export annotated image", command=self.export_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="Export arrow-box intensities (CSV)", command=self.export_arrow_box_metrics)\
            .pack(fill=tk.X, padx=6, pady=4)

        # ---- Custom labels (column name + per-column MWs) ----
        f_lab = ttk.LabelFrame(left, text="Custom labels (per column name + MWs)")
        f_lab.pack(fill=tk.X, pady=6)
        ttk.Button(f_lab, text="Edit column names & MWs...", command=self.open_labels_editor)\
            .pack(fill=tk.X, padx=6, pady=(6, 6))

        # ---- Bottom note ----
        f_note = ttk.LabelFrame(left, text="Bottom note")
        f_note.pack(fill=tk.X, pady=6)
        if not hasattr(self, "bottom_note_text"):
            self.bottom_note_text = ""
        ttk.Button(f_note, text="Edit bottom note...", command=self.open_bottom_note_editor)\
            .pack(fill=tk.X, padx=6, pady=6)

        # ---- White balance / autoscale ----
        f_wb = ttk.LabelFrame(left, text="White balance / Autoscale (ROI level, optional)")
        f_wb.pack(fill=tk.X, pady=6)
        self.var_wb_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_wb, text="Enable white balance (linear percentile stretch)",
                        variable=self.var_wb_on).pack(anchor="w", padx=6)

        self.var_wb_exposure = tk.DoubleVar(value=1.0)
        self.var_wb_p_low = tk.DoubleVar(value=0.5)
        self.var_wb_p_high = tk.DoubleVar(value=99.5)
        self.var_wb_per_channel = tk.BooleanVar(value=False)
        self.var_gamma_on = tk.BooleanVar(value=False)
        self.var_gamma_val = tk.DoubleVar(value=1.0)

        self._spin(f_wb, "Exposure (0.5–2.0)", self.var_wb_exposure, 0.5, 2.0, step=0.05)
        self._spin(f_wb, "Low percentile (0–50)", self.var_wb_p_low, 0.0, 50.0, step=0.1)
        self._spin(f_wb, "High percentile (50–100)", self.var_wb_p_high, 50.0, 100.0, step=0.1)
        ttk.Checkbutton(
            f_wb, text="Per-channel (max contrast, may change colors)",
            variable=self.var_wb_per_channel
        ).pack(anchor="w", padx=6, pady=(2,4))

        def _toggle_gamma_state():
            self.sp_gamma.configure(state=("normal" if self.var_gamma_on.get() else "disabled"))
        ttk.Checkbutton(
            f_wb, text="Enable gamma (out = out ** (1/gamma))",
            variable=self.var_gamma_on, command=_toggle_gamma_state
        ).pack(anchor="w", padx=6)

        frm_g = ttk.Frame(f_wb); frm_g.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(frm_g, text="gamma (>0)", wraplength=self.LEFT_WIDTH-40, justify="left").pack(side=tk.LEFT)
        self.sp_gamma = ttk.Spinbox(frm_g, textvariable=self.var_gamma_val,
                                    from_=0.2, to=3.0, increment=0.05, width=8, state="disabled")
        self.sp_gamma.pack(side=tk.RIGHT)

    # -------------------- UI：右侧（显示） -------------------- #

    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.pw_main = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_main.pack(fill=tk.BOTH, expand=True)

        top_grp = ttk.LabelFrame(self.pw_main, text="Original / ROI editor (drag corners & edges; wheel to zoom; right-drag to pan)")
        self.roi_editor = ROIEditorCanvas(top_grp)
        self.roi_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        bottom_container = ttk.Frame(self.pw_main)
        self.pw_bottom = tk.PanedWindow(bottom_container, orient=tk.HORIZONTAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_bottom.pack(fill=tk.BOTH, expand=True)

        self.lbl_roi_wb = ttk.LabelFrame(self.pw_bottom, text="ROI - WB crop")
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

    def _normalize_labels(self, labels: list[list[str]], nlanes: int) -> list[list[str]]:
        """
        将 labels 规整为 rows × nlanes 的二维表：
        - 每行不足 nlanes 用空串补齐；
        - 超出则裁剪到 nlanes；
        - labels 为空时返回 []。
        """
        if not labels:
            return []
        out: list[list[str]] = []
        for row in labels:
            row = list(row) if row else []
            if len(row) < nlanes:
                row = row + [""] * (nlanes - len(row))
            else:
                row = row[:nlanes]
            out.append(row)
        return out

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

        try:
            self.bottom_note_text = ""
        except Exception:
            setattr(self, "bottom_note_text", "")
        self.lane_names = []
        self.lane_marks = []
        self.render_cache = {}
        self.gi = 1
        self.boxes = []
        self.roi_editor.set_image(self.orig_bgr)
        self.run_detect()



    def run_detect(self):
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
        if not self.boxes:
            return
        self.gi = (self.gi - 1 + delta) % len(self.boxes) + 1  # 1..N 环绕
        self.roi_editor.set_roi(self.boxes[self.gi - 1])

    def reset_roi_from_detect(self):
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


    def render_current(self):
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

        # 2.5) 标准凝胶宽度
        design = self._get_design_params()
        gel_bgr = self._standardize_gel_size(gel_bgr, design["design_gel_width_px"])
        gel_gray = cv2.cvtColor(gel_bgr, cv2.COLOR_BGR2GRAY)

        # 3) 解析标准带
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
        tick_labels = ladder_labels_all  # 仅在拟合通过时交给底图绘刻度

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

        # 5) 标准道检测 + 拟合
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
        y0_roi, y1_roi = 0, None
        if bounds is not None:
            peaks, prom = detect_bands_along_y_slanted(
                gel_gray, bounds, lane_index=ladder_lane-1,
                y0=y0_roi, y1=y1_roi, min_distance=20, min_prominence=10.0
            )
        else:
            lx, rx = lanes[ladder_lane-1]
            sub = gel_gray[:, lx:rx]
            peaks, prom = detect_bands_along_y_prominence(
                sub, y0=y0_roi, y1=y1_roi, min_distance=20, min_prominence=10.0
            )
        ladder_peaks_for_draw = [int(round(p)) for p in peaks]
        ladder_labels_for_draw = sorted(ladder_labels_all, reverse=True)[:len(ladder_peaks_for_draw)]
        sel_p_idx, sel_l_idx = match_ladder_best(peaks, ladder_labels_all, prom, min_pairs=3)
        fit_ok = False; a, b = 1.0, 0.0; r2 = rmse = None
        if len(sel_p_idx) >= 2:
            y_used = [float(peaks[i]) for i in sel_p_idx]
            lbl_used = [sorted(ladder_labels_all, reverse=True)[j] for j in sel_l_idx]
            w_used = [float(prom[i]) for i in sel_p_idx]
            a_fit, b_fit = fit_log_mw_irls(y_used, lbl_used, w_used, iters=6)
            H_roi = gel_gray.shape[0]
            ok, r2, rmse = eval_fit_quality(y_used, lbl_used, a_fit, b_fit, H=H_roi,
                                            r2_min=0.5, rmse_frac_max=0.02, rmse_abs_min_px=80.0)
            if ok: a, b, fit_ok = a_fit, b_fit, True

        # 6) 核心底图（底图函数负责峰位标注；刻度在拟合通过时由底图绘制）
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

        # ★ 在“底部备注扩宽”之前，计算真实 panel_w 并保留
        #    此时宽度仍然是：panel_w + Wg
        panel_w_val = max(0, annotated_final_base.shape[1] - gel_bgr.shape[1])

        # 9) 绿线（仅绿线）
        annotated_final_with_green = self._draw_overlays_on_core(
            img_core=annotated_final_base, gel_bgr=gel_bgr,
            overlay=None, bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            tick_labels=self.render_cache["tick_labels"],  # 不在该函数里绘制
            yaxis_side=self.var_axis.get(),
            show_green=True,
            y_offset=panel_top_h
        )

        # 10) 底部备注（可能会向右扩白）
        note_raw = getattr(self, "bottom_note_text", "") or ""
        annotated_final_base_with_note = self._attach_bottom_note_panel(
            annotated_final_base, note_raw, allow_expand_width=True
        )
        annotated_final_with_green_with_note = self._attach_bottom_note_panel(
            annotated_final_with_green, note_raw, allow_expand_width=True
        )

        # 11) 左侧 ROI 预览
        self._set_autofit_image(self.canvas_roi_wb, gel_bgr)

        # 12) 右侧交互画布 —— ★ 显式传入 panel_w_val，避免箭头随右侧扩白漂移
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
            panel_w=panel_w_val,                # ★ 关键
        )
        if bool(self.var_show_boxes.get()):
            try: self.canvas_anno.set_boxes_enabled(True)
            except Exception: pass
        if bool(self.var_show_green.get()):
            self.canvas_anno.update_base_image(annotated_final_with_green_with_note)

        # 13) 更新缓存两份导出图
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
        """
        不重复跑检测/分道/拟合：用缓存核心图 + 几何参数，
        仅做：顶部列名（可选） -> 绿线（可选，注意 y_offset） -> 底部备注。
        刻度/峰标注完全保留底图本身，不在这里重复绘制。
        """
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

        # 1) 顶部列名（如开启）
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

        # ★ 在“底部备注扩宽”之前，计算真实 panel_w
        panel_w_val = 0
        try:
            panel_w_val = max(0, base_img.shape[1] - gel_bgr.shape[1])
        except Exception:
            panel_w_val = 0

        # 2) 绿线（仅绿线；不重复画刻度）
        want_green = bool(self.var_show_green.get())
        if want_green:
            with_green = self._draw_overlays_on_core(
                img_core=base_img, gel_bgr=gel_bgr,
                overlay=None, bounds=bounds, lanes=lanes,
                a=a, b=b, fit_ok=fit_ok,
                tick_labels=tick_lbls,  # 不在该函数中使用
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

        # 4) 刷新右侧交互画布 —— ★ 显式传入 panel_w_val
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
            panel_w=panel_w_val,               # ★ 关键
        )
        if bool(self.var_show_boxes.get()):
            try: self.canvas_anno.set_boxes_enabled(True)
            except Exception: pass
        if want_green and with_green_note is not None:
            self.canvas_anno.update_base_image(with_green_note)

        # 5) 更新缓存导出图
        self.render_cache.update({
            "annotated_base_no_green": base_with_note,
            "annotated_base_with_green": with_green_note or base_with_note,
        })



    # -------- 新增：按“每列分子量列表”绘制红色箭头 -------- #
    def _overlay_lane_marks(
        self,
        annotated_img: np.ndarray,
        gel_bgr: np.ndarray,
        bounds: np.ndarray | None,
        lanes: list[tuple[int, int]] | None,
        a: float, b: float, fit_ok: bool,
        nlanes: int,
        ladder_lane: int,
        yaxis_side: str
    ) -> np.ndarray:
        """
        将 self.lane_marks（编辑器的每行）顺序映射到“非标准道”各列，仅这些列绘制红色箭头。
        规则：缺则空，多则省。Y 轴在左时自动做水平偏移。
        """
        img = annotated_img.copy()
        if (not fit_ok) or (not isinstance(img, np.ndarray)) or img.size == 0:
            return img

        H, W_total = img.shape[:2]
        Hg, Wg = gel_bgr.shape[:2]
        if Hg != H:
            Hg = H  # 兜底

        # 白面板偏移：仅当 Y 轴在左
        panel_w = max(0, W_total - Wg)
        x_offset = panel_w if (str(yaxis_side).lower() == "left") else 0

        # 给定 y、lane i（0-based）→ 该行两分隔线中点（加上 x_offset）
        def lane_center_x_at_y(y_int: int, lane_idx: int) -> int:
            y_clamp = int(np.clip(y_int, 0, Hg - 1))
            if bounds is not None and isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[0] >= Hg:
                L = int(bounds[y_clamp, lane_idx])
                R = int(bounds[y_clamp, lane_idx + 1])
                xc = int(round((L + R) / 2.0))
            elif lanes is not None and 0 <= lane_idx < len(lanes):
                l, r = lanes[lane_idx]
                xc = int(round((l + r) / 2.0))
            else:
                step = Wg / max(1, nlanes)
                xc = int(round((lane_idx + 0.5) * step))
            return int(x_offset + np.clip(xc, 0, Wg - 1))

        # 目标列索引：所有非标准道，按左到右
        skip_idx = max(0, int(ladder_lane) - 1)  # 标准道(0-based)
        if bounds is not None and isinstance(bounds, np.ndarray):
            real_nlanes = max(0, bounds.shape[1] - 1)
        elif lanes is not None:
            real_nlanes = len(lanes)
        else:
            real_nlanes = nlanes
        target_lane_idx = [i for i in range(real_nlanes) if i != skip_idx]

        # 小实心红箭头
        def draw_arrow(xc: int, y: int, color=(0, 0, 255)):
            tip = (int(xc), int(np.clip(y, 0, H - 1)))
            w, h = 40, 10
            p1 = (tip[0] - w // 2, int(np.clip(tip[1] + h, 0, H - 1)))
            p2 = (tip[0] - w // 2, int(np.clip(tip[1] - h, 0, H - 1)))
            pts = np.array([p1, p2, tip], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        marks_seq = self.lane_marks or []  # 编辑器“每行”的分子量列表
        # 仅取前 len(target_lane_idx) 个，缺则空，多则省
        use_k = min(len(target_lane_idx), len(marks_seq))
        for k in range(use_k):
            arr = marks_seq[k] or []
            lane_idx = target_lane_idx[k]
            for mw in arr:
                if not (isinstance(mw, (int, float)) and np.isfinite(mw) and mw > 0):
                    continue
                y = int(round(a * np.log10(float(mw)) + b))
                xc = lane_center_x_at_y(y, lane_idx)
                draw_arrow(xc, y)

        # 若 marks_seq 比非标准道少：自动空出；比之多：已被省略
        return img

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
        在 img_bgr 上方追加“白底黑字”的标签面板（竖排+列中心水平居中，纵向向下对齐）。
        本版修复：字符被裁半的问题——包含 baseline、高度/边距放宽，并为旋转后位图添加安全留白。
        """
        import numpy as np, cv2
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        H, W_total = img_bgr.shape[:2]
        Hg, Wg = gel_bgr.shape[:2]

        panel_w = max(0, W_total - Wg)
        x_offset = panel_w if (str(yaxis_side).lower() == "left") else 0

        rows = len(labels_table)
        if rows == 0:
            return img_bgr
        cols_use = len(labels_table[0]) if rows else 0
        if cols_use == 0:
            return img_bgr

        # 真实泳道数
        if bounds is not None and isinstance(bounds, np.ndarray):
            real_nlanes = max(0, bounds.shape[1] - 1)
        elif lanes is not None:
            real_nlanes = len(lanes)
        else:
            real_nlanes = cols_use + 1  # 至少比使用列多1（含标准道）

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
        lefts_all = [int(x_offset + max(0, L)) for L in lefts_all]
        rights_all = [int(x_offset + min(Wg, R)) for R in rights_all]
        widths_all = [max(1, r - l) for l, r in zip(lefts_all, rights_all)]

        skip_idx = max(0, int(ladder_lane) - 1)  # 标准道（0-based）
        target_lane_idx = [i for i in range(real_nlanes) if i != skip_idx]

        # 字体参数（字号更大 + 加粗）
        design = self._get_design_params()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 2  # 加粗
        cap_px = int(design["top_cap_px"])
        base_scale = self._font_scale_for_cap_height_px(cap_px, font=font, thickness=thick)
        min_scale = 0.35

        # 版面参数（适度放宽）
        h_margin = 8                      # 列内安全边距
        top_pad, bot_pad = 8, 10
        row_gap = 6
        text_bottom_margin = 0            # 原先为 2，容易顶到行边导致视觉裁切，这里置 0
        _ROT_FLAG = cv2.ROTATE_90_COUNTERCLOCKWISE

        def render_rotated_text_img(text: str, lane_idx: int):
            """
            生成整段文本的水平位图 -> 旋转 90° -> 紧致裁剪(保留安全边距)
            关键：在水平位图阶段把 baseline 计入高度，避免 descender 被裁掉。
            """
            text = (text or "").strip()
            if not text:
                return None, 0, 0, 0.0

            lane_w = max(1, widths_all[lane_idx] - 2 * h_margin)

            # 先用目标字高对应的 scale
            scale = base_scale
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
            # 如果旋转后高度(≈ 水平th)会超出列宽，则按比例收缩
            if th > lane_w:
                scale = max(min_scale, scale * (lane_w / (th + 1e-6)))
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

            # --- 关键修复点：高度包含 baseline，并增加内边距 ---
            txt_pad = 3
            w_horiz = max(1, tw + 2 * txt_pad)
            h_horiz = max(1, th + baseline + 2 * txt_pad)

            img = np.full((h_horiz, w_horiz, 3), 255, dtype=np.uint8)
            # 基线位置：左下基线
            org = (txt_pad, txt_pad + th)
            cv2.putText(img, text, org, font, scale, (0, 0, 0), thick, cv2.LINE_AA)

            # 旋转 90° 得到竖排
            rot = cv2.rotate(img, _ROT_FLAG)

            # 紧致裁剪但留安全边界，避免把抗锯齿边缘吃掉
            rot_gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
            nz = np.where(rot_gray < 252)  # 适当放宽阈值
            if nz[0].size > 0:
                y_min, y_max = int(nz[0].min()), int(nz[0].max())
                x_min, x_max = int(nz[1].min()), int(nz[1].max())
                # 安全扩展 1px
                y_min = max(0, y_min - 1); y_max = min(rot.shape[0] - 1, y_max + 1)
                x_min = max(0, x_min - 1); x_max = min(rot.shape[1] - 1, x_max + 1)
                rot = rot[y_min:y_max + 1, x_min:x_max + 1]

            # 再加一圈 1px 白边，避免贴边观感
            rot = cv2.copyMakeBorder(rot, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            h_rot, w_rot = rot.shape[:2]
            return rot, w_rot, h_rot, float(scale)

        # 行测量与缓存
        rows_used_idx, row_heights = [], []
        cell_cache: dict[tuple[int, int], tuple[np.ndarray | None, int, int, float]] = {}
        STD_TEXT = "Standard"

        for r in range(rows):
            row_has_nonstd = any((labels_table[r][c] or "").strip() for c in range(cols_use))
            if not row_has_nonstd:
                row_heights.append(0)
                continue

            max_h_rot = 0
            # 非标准道
            for c in range(min(cols_use, len(target_lane_idx))):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                li = target_lane_idx[c]
                cell = render_rotated_text_img(text, li)
                cell_cache[(r, li)] = cell
                _, w_rot, h_rot, _ = cell
                max_h_rot = max(max_h_rot, h_rot)

            # 标准道（仅在该行存在任何文本时绘制“Standard”）
            if 0 <= skip_idx < real_nlanes:
                cell_std = render_rotated_text_img(STD_TEXT, skip_idx)
                cell_cache[(r, skip_idx)] = cell_std
                _, w_rot_s, h_rot_s, _ = cell_std
                max_h_rot = max(max_h_rot, h_rot_s)

            # 再加 2px 行安全高度，避免顶/底贴边
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
                # 纵向：行内向下对齐；取消上抬 margin，避免顶边裁切
                y_top = int(y_cursor + (rh - h_rot))

                y1 = max(0, y_top); y2 = min(panel_h, y_top + h_rot)
                x1 = max(0, x_left); x2 = min(W_total, x_left + w_rot)
                if y2 > y1 and x2 > x1:
                    sub_h, sub_w = y2 - y1, x2 - x1
                    panel[y1:y2, x1:x2] = rot_img[0:sub_h, 0:sub_w]

            # 标准道（条件绘制）
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
        追加底部白底备注，多行左对齐。优先使用支持 CJK 的 TTF/TTC。
        找不到字体时回退到 OpenCV（仅英文可靠），但不会抛异常。
        —— 本版改动：
        1) 读取设计参数 bottom_cap_px（默认 28）作为“目标字高/字号”；
        2) Pillow 分支用 bottom_cap_px 作为像素字号；
        3) OpenCV 分支用 bottom_cap_px 计算 fontScale；
        4) 两分支统一 thickness=2，缩放后更清晰。
        """
        import numpy as np, cv2
        from PIL import Image, ImageDraw, ImageFont

        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return img_bgr

        raw = (note_text or "")
        lines_raw = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        # 展开 \t 以便对齐
        tab_size = 4
        lines = [self._expand_tabs_for_cv2(ln, tab_size=tab_size) for ln in lines_raw]
        if not any((ln.strip() for ln in lines)):
            return img_bgr

        H, W = img_bgr.shape[:2]
        design = self._get_design_params()
        cap_px = int(design.get("bottom_cap_px", 28))  # ★ 统一字号

        font_path = self._find_font_ttf()

        # —— 首选：Pillow + 可用 TTF（支持中文）
        if font_path:
            try:
                size_px = max(12, cap_px)  # 直接作为像素字号
                font = ImageFont.truetype(font_path, size=size_px)

                # 逐行测量
                tmp = Image.new("RGB", (8, 8), "white")
                drw = ImageDraw.Draw(tmp)
                line_heights, line_widths = [], []
                for ln in lines:
                    text = ln if ln else " "
                    bbox = drw.textbbox((0, 0), text, font=font)
                    tw, th = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                    line_heights.append(max(1, th))
                    line_widths.append(max(1, tw))

                max_line_w = max(line_widths) if line_widths else 0
                panel_h = top_pad + bot_pad
                if line_heights:
                    panel_h += sum(line_heights) + max(0, (len(line_heights) - 1) * line_gap)

                needed_w = left_pad + max_line_w + right_pad
                new_W = max(W, needed_w) if allow_expand_width else W

                # 宽度不足则右侧扩白
                if new_W > W:
                    pad = np.full((H, new_W - W, 3), 255, dtype=np.uint8)
                    img_main = np.concatenate([img_bgr, pad], axis=1)
                else:
                    img_main = img_bgr

                panel_img = Image.new("RGB", (new_W, panel_h), "white")
                drw = ImageDraw.Draw(panel_img)

                y = top_pad
                for i, ln in enumerate(lines):
                    text = ln if ln else " "
                    x = left_pad
                    # 用描边模拟 thickness=2 的对比度（可选）
                    drw.text((x, y), text, fill=(0, 0, 0), font=font)
                    y += line_heights[i] + line_gap

                panel_bgr = cv2.cvtColor(np.array(panel_img), cv2.COLOR_RGB2BGR)
                return np.vstack([img_main, panel_bgr])
            except Exception:
                pass

        # —— 回退：OpenCV Hershey
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 用 bottom_cap_px 推算 fontScale
        scale = self._font_scale_for_cap_height_px(cap_px, font=font, thickness=2)
        thickness = 2  # ★ 与顶端面板统一

        # 估算行高与最大宽
        line_sizes = [cv2.getTextSize(ln if ln else " ", font, scale, thickness)[0] for ln in lines]
        line_heights = [sz[1] for sz in line_sizes]
        line_widths = [sz[0] for sz in line_sizes]
        max_line_w = max(line_widths) if line_widths else 0

        panel_h = top_pad + bot_pad
        if line_heights:
            panel_h += sum(line_heights) + max(0, (len(line_heights) - 1) * line_gap)

        needed_w = left_pad + max_line_w + right_pad
        new_W = max(W, needed_w) if allow_expand_width else W

        if new_W > W:
            pad = np.full((H, new_W - W, 3), 255, dtype=np.uint8)
            img_main = np.concatenate([img_bgr, pad], axis=1)
        else:
            img_main = img_bgr

        panel = np.full((panel_h, new_W, 3), 255, dtype=np.uint8)

        y = top_pad
        for i, ln in enumerate(lines):
            text = ln if ln else " "
            (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
            x = left_pad
            y_baseline = y + th  # 基线
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
                s = s.replace(";", ",").replace(" ", ",")
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
