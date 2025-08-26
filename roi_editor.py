# roi_editor.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageTk
import cv2
import math


@dataclass
class RBox:
    """旋转矩形：中心坐标 + 宽高（像素，图像坐标系），角度由外部基准线提供。"""
    cx: float
    cy: float
    w: float
    h: float

    def as_center_size(self) -> Tuple[float, float, float, float]:
        return float(self.cx), float(self.cy), float(self.w), float(self.h)

    def to_aabb(self, angle_deg: float) -> Tuple[int, int, int, int]:
        """把旋转矩形在给定角度下的外接轴对齐框（AABB）返回为 x,y,w,h（int）。"""
        corners = RBox._corners(self.cx, self.cy, self.w, self.h, angle_deg)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x0 = int(math.floor(min(xs)))
        y0 = int(math.floor(min(ys)))
        x1 = int(math.ceil(max(xs)))
        y1 = int(math.ceil(max(ys)))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    @staticmethod
    def _corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[Tuple[float, float]]:
        """返回按顺时针顺序的四个角点（图像坐标）。"""
        hw, hh = w / 2.0, h / 2.0
        # 局部坐标（u-v）
        pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        rad = math.radians(angle_deg)
        cosA, sinA = math.cos(rad), math.sin(rad)
        out = []
        for u, v in pts:
            x = cx + (u * cosA - v * sinA)
            y = cy + (u * sinA + v * cosA)
            out.append((x, y))
        return out


class ROIEditorCanvas(ttk.Frame):
    """
    可缩放/平移的图像画布，支持交互式拖拽编辑“随基准线旋转”的矩形 ROI：
    - 左键拖动角/边修改 ROI（在 ROI 局部坐标系内缩放），左键拖动内部移动 ROI
    - 左键空白处拖拽新建 ROI（创建的是与基准线同角度的旋转矩形）
    - 右键/中键平移画布；滚轮缩放（以光标为中心）
    - 方向键/Shift+方向键：以像素级微调 ROI 中心
    扩展：新增“校准横线”以手动校准倾斜（±15°）
    - 永久显示；支持拖动（平移）与端点旋转（保持长度不变）
    - 旋转角度限制在 [-15°, +15°]
    - 通过 get_angle_deg() 提供当前角度（用于下游裁剪后的“摆正”）
    """
    HANDLE_SIZE = 5            # ROI 角点可见更清晰
    EDGE_HANDLE_SIZE = 5       # 边中点句柄
    CORNER_LEN = 18            # ROI L 形角标长度（保留视觉风格）
    LINE_HANDLE_SZ = 6         # 校准线端点/中心手柄可见尺寸
    LINE_COLOR = "#00d1ff"
    ROI_COLOR = "#20f28b"

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, background="#1e1e1e", highlightthickness=0, cursor="tcross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 图像与显示
        self.img_bgr: Optional[np.ndarray] = None
        self.img_pil: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.min_scale = 0.05
        self.max_scale = 16.0
        self.offset_x = 0.0  # 画布坐标系中图像左上角位置
        self.offset_y = 0.0

        # ROI：旋转矩形（角度使用 cal_angle_deg）
        self.rbox: Optional[RBox] = None

        # 校准线（以图像坐标存储：中心+长度+角度）
        self.cal_cx: float = 0.0
        self.cal_cy: float = 0.0
        self.cal_len: float = 0.0
        self.cal_angle_deg: float = 0.0  # [-15,+15]

        # 状态
        self.dragging = False
        self.drag_mode = None  # "move" | "new" | "resize-<tag>" | "pan" | "line-move" | "line-rot-p1" | "line-rot-p2"
        self.drag_start = (0, 0)  # canvas coords
        self.orig_rbox: Optional[RBox] = None
        self.orig_line = None  # (cx,cy,len,angle) 在旋转/拖动校准线时保存初始值

        # 事件绑定
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-2>", self._on_mid_down)   # 有些平台中键平移
        self.canvas.bind("<B2-Motion>", self._on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_mid_up)
        self.canvas.bind("<Button-3>", self._on_right_down) # 右键平移
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)

        # 滚轮（Win/Mac/Linux 兼容）
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)     # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux) # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux) # Linux down

        # 键盘微调
        self.canvas.focus_set()
        self.canvas.bind("<Up>",    lambda e: self._nudge(0, -1, e))
        self.canvas.bind("<Down>",  lambda e: self._nudge(0,  1, e))
        self.canvas.bind("<Left>",  lambda e: self._nudge(-1, 0, e))
        self.canvas.bind("<Right>", lambda e: self._nudge( 1, 0, e))

        # 信息栏
        self.info_var = tk.StringVar(value="")
        self.lbl_info = ttk.Label(self, textvariable=self.info_var, foreground="#ddd")
        self.lbl_info.pack(anchor="w", padx=6, pady=4)

    # ---------- 公有 API ----------
    def set_image(self, img_bgr: np.ndarray):
        self.img_bgr = img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(rgb)
        # 初始缩放以适配窗口
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self._fit_to_window()
        # 初始化校准线（位于图上方 18% 处，长度为宽度的 60%，角度 0）
        H, W = img_bgr.shape[0], img_bgr.shape[1]
        self.cal_cx = W / 2.0
        self.cal_cy = max(10.0, H * 0.18)
        self.cal_len = max(50.0, W * 0.6)
        self.cal_angle_deg = 0.0
        self.redraw()

    def set_roi(self, box: Tuple[int, int, int, int] | None):
        """
        兼容旧接口：输入轴对齐框 (x,y,w,h) 或 None。
        内部转换为旋转矩形的“中心+宽高”，角度由 cal_angle_deg 决定。
        """
        if box is None:
            self.rbox = None
        else:
            x, y, w, h = map(int, box)
            cx = x + w / 2.0
            cy = y + h / 2.0
            self.rbox = RBox(cx, cy, max(1.0, float(w)), max(1.0, float(h)))
            self._clamp_rbox_basic()
        self.redraw()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """保持兼容：返回当前“旋转 ROI”的外接轴对齐框 (x,y,w,h)。"""
        if self.rbox is None:
            return None
        return self.rbox.to_aabb(self.cal_angle_deg)

    def get_rotated_roi(self):
        """
        返回 (cx, cy, w, h, angle_deg_ccw)
        - cx,cy,w,h：图像坐标/像素，w、h 为 ROI 局部宽高
        - angle_deg_ccw：对外统一为 OpenCV 语义（逆时针 CCW 为正）
        """
        if self.rbox is None:
            return None
        cx, cy, w, h = self.rbox.as_center_size()
        # 画布内部若使用 CW 为正，这里导出成 CCW 为正：取反
        angle_deg_ccw = float(-self.cal_angle_deg)
        return float(cx), float(cy), float(w), float(h), angle_deg_ccw

    def get_angle_deg(self) -> float:
        """返回当前校准线角度（度），范围 [-15, +15]。正值=顺时针（向右上倾斜）。"""
        return float(self.cal_angle_deg)

    # ---------- 坐标转换 ----------
    def img_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        return ((cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale)

    # ---------- 事件处理 ----------
    def _on_resize(self, _):
        self.redraw()

    def _on_left_down(self, e):
        self.canvas.focus_set()
        if self.img_pil is None:
            return
        cx, cy = e.x, e.y

        # ① 优先命中“校准线”手柄（中心/两端）或近线拖动
        tag_line = self._hit_test_line_handles(cx, cy)
        if tag_line:
            self.dragging = True
            self.drag_start = (cx, cy)
            self.orig_line = (self.cal_cx, self.cal_cy, self.cal_len, self.cal_angle_deg)
            if tag_line in ("line_c", "line_seg"):
                self.drag_mode = "line-move"
            elif tag_line == "line_p1":
                self.drag_mode = "line-rot-p1"
            elif tag_line == "line_p2":
                self.drag_mode = "line-rot-p2"
            return

        # ② ROI 操作
        if self.rbox:
            tag = self._hit_test_rbox_handles(cx, cy)
            if tag:
                # 拖拽边/角（在 ROI 局部坐标系下）
                self.drag_mode = f"resize-{tag}"
                self.dragging = True
                self.drag_start = (cx, cy)
                self.orig_rbox = RBox(self.rbox.cx, self.rbox.cy, self.rbox.w, self.rbox.h)
                return

            # 点在旋转 ROI 内 → 移动
            if self._point_in_rbox_canvas(cx, cy, self.rbox, self.cal_angle_deg):
                self.drag_mode = "move"
                self.dragging = True
                self.drag_start = (cx, cy)
                self.orig_rbox = RBox(self.rbox.cx, self.rbox.cy, self.rbox.w, self.rbox.h)
                return

        # ③ 空白处 → 新建 ROI（初始创建为与基准线同角度的矩形）
        self.drag_mode = "new"
        self.dragging = True
        self.drag_start = (cx, cy)
        ix, iy = self.canvas_to_img(cx, cy)
        # 初始给 1x1，后续拖动时计算外接框 → 转为旋转矩形
        self.rbox = RBox(float(ix), float(iy), 1.0, 1.0)
        self.redraw()
    def reset_angle(self):
        """
        将校准线旋转角设为 0°（水平），并重绘。
        仅修改角度，不改变校准线的中心位置与长度。
        """
        self.cal_angle_deg = 0.0
        self.redraw()

    def _on_left_drag(self, e):
        if not self.dragging or self.img_pil is None:
            return
        cx, cy = e.x, e.y
        dx, dy = cx - self.drag_start[0], cy - self.drag_start[1]
        H = self.img_bgr.shape[0]
        W = self.img_bgr.shape[1]

        # —— 校准线拖动/旋转 —— #
        if self.drag_mode in ("line-move", "line-rot-p1", "line-rot-p2"):
            if self.orig_line is None:
                return
            ocx, ocy, olen, oang = self.orig_line
            if self.drag_mode == "line-move":
                # 画布位移 → 图像位移
                mx = dx / self.scale
                my = dy / self.scale
                self.cal_cx = float(np.clip(ocx + mx, 0, W))
                self.cal_cy = float(np.clip(ocy + my, 0, H))
            else:
                # 旋转：保持长度，改变角度（以中心为轴）
                ix, iy = self.canvas_to_img(cx, cy)
                vx = ix - ocx
                vy = iy - ocy
                if self.drag_mode == "line-rot-p1":
                    # 把 p1 的拖动同样映射到“中心→p2”方向，消除端点差异
                    vx, vy = -vx, -vy

                # 屏幕坐标（y 向下）：atan2(vy, vx) 的正角是【顺时针】
                # 统一到 (-90°, +90°]，再限幅到 ±15°
                if abs(vx) + abs(vy) < 1e-9:
                    theta = 0.0
                else:
                    theta = math.degrees(math.atan2(vy, vx))      # [-180, 180)
                    theta = ((theta + 180.0) % 360.0) - 180.0     # [-180, 180)
                    if theta > 90.0:
                        theta -= 180.0
                    elif theta <= -90.0:
                        theta += 180.0
                    theta = max(-45.0, min(45.0, theta))          # 限幅
                # 不再取反：保持“顺时针为正”
                self.cal_angle_deg = float(theta)
                self.cal_cx, self.cal_cy = ocx, ocy
                self.cal_len = olen
            self.redraw()
            return

        # —— ROI 操作 —— #
        if self.rbox is None and self.drag_mode != "new":
            return

        if self.drag_mode == "move":
            mx = dx / self.scale
            my = dy / self.scale
            self.rbox.cx = self.orig_rbox.cx + mx
            self.rbox.cy = self.orig_rbox.cy + my
            self._clamp_rbox_basic()

        elif self.drag_mode and self.drag_mode.startswith("resize-"):
            tag = self.drag_mode.split("-", 1)[1]
            ix, iy = self.canvas_to_img(cx, cy)
            self._resize_rbox_to_pointer(tag, ix, iy)

        elif self.drag_mode == "new":
            # 新建：按下起点为锚定角；另一角随鼠标沿“旋转坐标系”变化
            ax, ay = self.canvas_to_img(*self.drag_start)  # 锚点（固定角）
            bx, by = self.canvas_to_img(cx, cy)            # 当前鼠标

            rad = math.radians(self.cal_angle_deg)
            cosA, sinA = math.cos(rad), math.sin(rad)
            du = (bx - ax) * cosA + (by - ay) * sinA   # 局部 u（宽）方向
            dv = -(bx - ax) * sinA + (by - ay) * cosA  # 局部 v（高）方向

            ul, ur = (min(0.0, du), max(0.0, du))
            vt, vb = (min(0.0, dv), max(0.0, dv))

            u_c = (ul + ur) / 2.0
            v_c = (vt + vb) / 2.0
            w = max(1.0, ur - ul)
            h = max(1.0, vb - vt)

            dcx = u_c * cosA - v_c * sinA
            dcy = u_c * sinA + v_c * cosA
            cx_new = ax + dcx
            cy_new = ay + dcy

            self.rbox = RBox(cx_new, cy_new, w, h)
            self._clamp_rbox_basic()

        self.redraw()



    def _on_left_up(self, _):
        self.dragging = False
        self.drag_mode = None
        self.orig_rbox = None
        self.orig_line = None

    def _on_right_down(self, e):
        self.dragging = True
        self.drag_mode = "pan"
        self.drag_start = (e.x, e.y)

    def _on_right_drag(self, e):
        if not self.dragging or self.drag_mode != "pan":
            return
        dx = e.x - self.drag_start[0]
        dy = e.y - self.drag_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self.drag_start = (e.x, e.y)
        self.redraw()

    def _on_right_up(self, _):
        self.dragging = False
        self.drag_mode = None

    def _on_mid_down(self, e):  # 有些鼠标中键也用来平移
        self._on_right_down(e)

    def _on_mid_drag(self, e):
        self._on_right_drag(e)

    def _on_mid_up(self, e):
        self._on_right_up(e)

    def _on_mousewheel(self, e):
        if self.img_pil is None:
            return
        step = 1.1 if e.delta > 0 else 1 / 1.1
        self._zoom_at(step, e.x, e.y)

    def _on_mousewheel_linux(self, e):
        step = 1.1 if e.num == 4 else 1 / 1.1
        self._zoom_at(step, e.x, e.y)

    def _nudge(self, dx, dy, e):
        if self.rbox is None:
            return
        k = 10 if (e.state & 0x0001 or e.state & 0x0004) else 1  # Shift/Ctrl 都加速
        self.rbox.cx += dx * k
        self.rbox.cy += dy * k
        self._clamp_rbox_basic()
        self.redraw()

    # ---------- 内部工具 ----------
    def _fit_to_window(self):
        if self.img_pil is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        W, H = self.img_pil.size
        sx = cw / W
        sy = ch / H
        s = min(sx, sy) * 0.98
        self.scale = max(self.min_scale, min(self.max_scale, s))
        # 居中
        self.offset_x = (cw - W * self.scale) / 2
        self.offset_y = (ch - H * self.scale) / 2

    def _zoom_at(self, factor: float, cx: float, cy: float):
        """
        以鼠标位置为锚点缩放；若正在拖拽（move/resize/new/line-*），
        同步更新 drag_start，使拖拽连续不跳变。
        """
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - old_scale) < 1e-6:
            return

        # 1) 记录缩放锚点（鼠标）对应的图像坐标
        ix_anchor, iy_anchor = self.canvas_to_img(cx, cy)

        # 2) 若正在拖拽，记录 drag_start 对应的图像坐标（作为“拖拽锚点”）
        dragging = bool(self.dragging and self.drag_mode)
        if dragging:
            ds_cx, ds_cy = self.drag_start  # 旧的 canvas 坐标
            ds_ix, ds_iy = self.canvas_to_img(ds_cx, ds_cy)
        else:
            ds_ix = ds_iy = None

        # 3) 应用缩放，并调整 offset 以保证“鼠标下的图像点”不动
        self.scale = new_scale
        nx, ny = self.img_to_canvas(ix_anchor, iy_anchor)  # 新缩放下，该点的 canvas 坐标
        self.offset_x += (cx - nx)
        self.offset_y += (cy - ny)

        # 4) 若正在拖拽，把原 drag_start 对应的“同一图像点”映射回新的 canvas 坐标，保持拖拽连续性
        if dragging and ds_ix is not None:
            nds_cx, nds_cy = self.img_to_canvas(ds_ix, ds_iy)
            self.drag_start = (nds_cx, nds_cy)

        # 5) 重绘
        self.redraw()


    def _clamp_rbox_basic(self):
        """基础约束：中心留在图内；宽高范围做保守限制。"""
        if self.img_pil is None or self.rbox is None:
            return
        W, H = self.img_pil.size
        self.rbox.cx = float(np.clip(self.rbox.cx, 0, W - 1))
        self.rbox.cy = float(np.clip(self.rbox.cy, 0, H - 1))
        self.rbox.w = float(np.clip(self.rbox.w, 1.0, W))
        self.rbox.h = float(np.clip(self.rbox.h, 1.0, H))

    # ---------- ROI 命中/判定 ----------
    def _hit_test_rbox_handles(self, cx: float, cy: float) -> Optional[str]:
        """命中旋转矩形的 8 个句柄：lt/rt/rb/lb/l/t/r/b"""
        if self.rbox is None:
            return None
        # 角点与边中点（图像坐标）
        corners = RBox._corners(self.rbox.cx, self.rbox.cy, self.rbox.w, self.rbox.h, self.cal_angle_deg)
        (lt, rt, rb, lb) = corners
        mids = [
            ((lt[0] + rt[0]) / 2.0, (lt[1] + rt[1]) / 2.0),  # t
            ((rt[0] + rb[0]) / 2.0, (rt[1] + rb[1]) / 2.0),  # r
            ((rb[0] + lb[0]) / 2.0, (rb[1] + lb[1]) / 2.0),  # b
            ((lb[0] + lt[0]) / 2.0, (lb[1] + lt[1]) / 2.0),  # l
        ]
        tags_pts = [
            ("lt", lt), ("rt", rt), ("rb", rb), ("lb", lb),
            ("t", mids[0]), ("r", mids[1]), ("b", mids[2]), ("l", mids[3]),
        ]
        hs = self.HANDLE_SIZE
        for tag, (ix, iy) in tags_pts:
            hx, hy = self.img_to_canvas(ix, iy)
            if abs(cx - hx) <= hs and abs(cy - hy) <= hs:
                return tag
        return None

    def _point_in_rbox_canvas(self, cx: float, cy: float, rbox: RBox, angle_deg: float) -> bool:
        ix, iy = self.canvas_to_img(cx, cy)
        # 转到 ROI 局部坐标
        rad = math.radians(angle_deg)
        cosA, sinA = math.cos(rad), math.sin(rad)
        dx = ix - rbox.cx
        dy = iy - rbox.cy
        # 逆旋转（-angle）
        u = dx * cosA + dy * sinA
        v = -dx * sinA + dy * cosA
        return (abs(u) <= rbox.w / 2.0) and (abs(v) <= rbox.h / 2.0)

    # ---------- 校准线命中测试 ----------
    def _hit_test_line_handles(self, cx: float, cy: float) -> Optional[str]:
        if self.img_pil is None:
            return None
        (p1x, p1y), (p2x, p2y) = self._line_endpoints_img()
        cix, ciy = self.cal_cx, self.cal_cy
        items = [
            ("line_p1", p1x, p1y),
            ("line_p2", p2x, p2y),
            ("line_c",  cix, ciy),
        ]
        for tag, ix, iy in items:
            hx, hy = self.img_to_canvas(ix, iy)
            if abs(cx - hx) <= self.LINE_HANDLE_SZ and abs(cy - hy) <= self.LINE_HANDLE_SZ:
                return tag
        # 近线（拖到线身上 → 平移）
        dist = self._dist_point_to_segment_canvas(cx, cy, (p1x, p1y), (p2x, p2y))
        if dist <= self.LINE_HANDLE_SZ + 2:
            return "line_seg"
        return None

    def _line_endpoints_img(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rad = math.radians(self.cal_angle_deg)
        dx = math.cos(rad) * (self.cal_len / 2.0)
        dy = math.sin(rad) * (self.cal_len / 2.0)
        p1 = (self.cal_cx - dx, self.cal_cy - dy)
        p2 = (self.cal_cx + dx, self.cal_cy + dy)
        return p1, p2

    def _dist_point_to_segment_canvas(self, cx: float, cy: float,
                                      p1_img: Tuple[float, float], p2_img: Tuple[float, float]) -> float:
        x1, y1 = self.img_to_canvas(*p1_img)
        x2, y2 = self.img_to_canvas(*p2_img)
        vx, vy = x2 - x1, y2 - y1
        wx, wy = cx - x1, cy - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            return math.hypot(wx, wy)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        projx = x1 + t * vx
        projy = y1 + t * vy
        return math.hypot(cx - projx, cy - projy)

    # ---------- 旋转 ROI 缩放逻辑（在局部坐标系内） ----------
    def _resize_rbox_to_pointer(self, tag: str, ix: float, iy: float):
        """
        在 ROI 局部坐标系内进行缩放：
        - 角点句柄（lt/rt/rb/lb）：锚点为“对角点”；
        - 边句柄（l/r/t/b）：锚点为“对侧边整条线”，即该边的位置保持不动（宽/高单向改变）。
        鼠标点与锚点共同决定新的中心与尺寸。
        """
        if self.orig_rbox is None:
            return

        # 1) 世界坐标 -> ROI 局部坐标（u,v），局部坐标系定义：
        #    u 轴沿矩形宽方向（向右为 +），v 轴沿高方向（向下为 +）
        rad = math.radians(self.cal_angle_deg)
        cosA, sinA = math.cos(rad), math.sin(rad)

        def to_local(x, y, cx0, cy0):
            dx = x - cx0
            dy = y - cy0
            u = dx * cosA + dy * sinA
            v = -dx * sinA + dy * cosA
            return u, v

        # 当前鼠标在“原始矩形中心”为基准的局部坐标
        u_mouse, v_mouse = to_local(ix, iy, self.orig_rbox.cx, self.orig_rbox.cy)

        # 原始半宽/半高
        w2_0 = self.orig_rbox.w / 2.0
        h2_0 = self.orig_rbox.h / 2.0

        # 2) 计算“锚点/锚边”的局部坐标边界（ul/ur/vt/vb 分别为左/右/上/下边在线的 u 或 v 值）
        # 对于角点：锚点是对角点 → 固定一个点 (u_anchor, v_anchor)；
        # 对于边：锚边是一条线 → 对侧边的位置固定（例如左边拖动时，锚右边 u = +w2_0 不动）。
        # 然后用  “锚点/锚边 + 鼠标点” 共同决定新的边界，再求新中心与尺寸。
        # 初始边界取自原始矩形
        ul, ur = -w2_0, +w2_0   # 左/右边的 u
        vt, vb = -h2_0, +h2_0   # 上/下边的 v

        # 防止出现 0 或负的尺寸
        MIN_HALF = 0.5

        if tag in ("lt", "rt", "rb", "lb"):
            # 角点拖拽：锚点为对角点
            if tag == "lt":
                # 锚：rb → ( +w2_0, +h2_0 )；鼠标定义新 (ul, vt)
                ul = min(u_mouse, ur - 2*MIN_HALF)
                vt = min(v_mouse, vb - 2*MIN_HALF)
            elif tag == "rt":
                # 锚：lb → ( -w2_0, +h2_0 )；鼠标定义新 (ur, vt)
                ur = max(u_mouse, ul + 2*MIN_HALF)
                vt = min(v_mouse, vb - 2*MIN_HALF)
            elif tag == "rb":
                # 锚：lt → ( -w2_0, -h2_0 )；鼠标定义新 (ur, vb)
                ur = max(u_mouse, ul + 2*MIN_HALF)
                vb = max(v_mouse, vt + 2*MIN_HALF)
            elif tag == "lb":
                # 锚：rt → ( +w2_0, -h2_0 )；鼠标定义新 (ul, vb)
                ul = min(u_mouse, ur - 2*MIN_HALF)
                vb = max(v_mouse, vt + 2*MIN_HALF)

        elif tag in ("l", "r", "t", "b"):
            # 边拖拽：对侧边整条线位置固定；另一维度尺寸不变
            if tag == "l":
                # 固定右边：ur = +w2_0；改变 ul
                ul = min(u_mouse, ur - 2*MIN_HALF)
                # 高度不变
                vt, vb = -h2_0, +h2_0
            elif tag == "r":
                # 固定左边：ul = -w2_0；改变 ur
                ur = max(u_mouse, ul + 2*MIN_HALF)
                vt, vb = -h2_0, +h2_0
            elif tag == "t":
                # 固定下边：vb = +h2_0；改变 vt
                vt = min(v_mouse, vb - 2*MIN_HALF)
                ul, ur = -w2_0, +w2_0
            elif tag == "b":
                # 固定上边：vt = -h2_0；改变 vb
                vb = max(v_mouse, vt + 2*MIN_HALF)
                ul, ur = -w2_0, +w2_0
        else:
            return

        # 3) 由新边界求新的中心（局部）与半宽/半高
        u_c = (ul + ur) / 2.0
        v_c = (vt + vb) / 2.0
        w2_new = max(MIN_HALF, (ur - ul) / 2.0)
        h2_new = max(MIN_HALF, (vb - vt) / 2.0)

        # 4) 局部中心 -> 世界坐标中心
        dcx = u_c * cosA - v_c * sinA
        dcy = u_c * sinA + v_c * cosA
        cx_new = self.orig_rbox.cx + dcx
        cy_new = self.orig_rbox.cy + dcy

        # 5) 写回并约束
        self.rbox = RBox(cx_new, cy_new, 2.0 * w2_new, 2.0 * h2_new)
        self._clamp_rbox_basic()


    # ---------- 绘制 ----------
    def redraw(self):
        self.canvas.delete("all")
        if self.img_pil is None:
            return

        # 绘制图像
        W, H = self.img_pil.size
        disp_w = int(max(1, round(W * self.scale)))
        disp_h = int(max(1, round(H * self.scale)))
        img = self.img_pil.resize((disp_w, disp_h), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

        # ROI 绘制（旋转）
        if self.rbox:
            corners = RBox._corners(self.rbox.cx, self.rbox.cy, self.rbox.w, self.rbox.h, self.cal_angle_deg)
            pts_canvas = [self.img_to_canvas(x, y) for (x, y) in corners]
            # 外框
            self.canvas.create_polygon(
                *sum(([x, y] for (x, y) in pts_canvas), []),
                outline=self.ROI_COLOR, width=2, fill=""
            )
            # 角 L 形标（沿边方向）
            L = self.CORNER_LEN
            def draw_L(p0, p1, p3):  # 以 p0 为角点，p1/p3 为相邻边方向
                # 取单位方向向量
                vx = p1[0] - p0[0]; vy = p1[1] - p0[1]
                ux = p3[0] - p0[0]; uy = p3[1] - p0[1]
                vlen = max(1e-6, math.hypot(vx, vy))
                ulen = max(1e-6, math.hypot(ux, uy))
                vx, vy = vx / vlen, vy / vlen
                ux, uy = ux / ulen, uy / ulen
                self.canvas.create_line(p0[0], p0[1], p0[0] + vx * L, p0[1] + vy * L, fill=self.ROI_COLOR, width=1)
                self.canvas.create_line(p0[0], p0[1], p0[0] + ux * L, p0[1] + uy * L, fill=self.ROI_COLOR, width=1)
            draw_L(pts_canvas[0], pts_canvas[1], pts_canvas[3])  # lt
            draw_L(pts_canvas[1], pts_canvas[2], pts_canvas[0])  # rt
            draw_L(pts_canvas[2], pts_canvas[3], pts_canvas[1])  # rb
            draw_L(pts_canvas[3], pts_canvas[0], pts_canvas[2])  # lb

            # 句柄（角点+边中点）
            hs = self.HANDLE_SIZE
            lt, rt, rb, lb = pts_canvas
            mids = [
                ((lt[0] + rt[0]) / 2.0, (lt[1] + rt[1]) / 2.0),  # t
                ((rt[0] + rb[0]) / 2.0, (rt[1] + rb[1]) / 2.0),  # r
                ((rb[0] + lb[0]) / 2.0, (rb[1] + lb[1]) / 2.0),  # b
                ((lb[0] + lt[0]) / 2.0, (lb[1] + lt[1]) / 2.0),  # l
            ]
            for hx, hy in [lt, rt, rb, lb] + mids:
                self.canvas.create_oval(hx - hs, hy - hs, hx + hs, hy + hs, outline="#111", fill="#fcf802")

        # 校准线（最后绘制，始终可见）
        (p1x, p1y), (p2x, p2y) = self._line_endpoints_img()
        lp1 = self.img_to_canvas(p1x, p1y)
        lp2 = self.img_to_canvas(p2x, p2y)
        lc = self.img_to_canvas(self.cal_cx, self.cal_cy)
        self.canvas.create_line(lp1[0], lp1[1], lp2[0], lp2[1], fill=self.LINE_COLOR, width=2)
        # 端点圆 & 中心方块
        s = self.LINE_HANDLE_SZ
        self.canvas.create_oval(lp1[0] - s, lp1[1] - s, lp1[0] + s, lp1[1] + s, outline="#111", fill=self.LINE_COLOR)
        self.canvas.create_oval(lp2[0] - s, lp2[1] - s, lp2[0] + s, lp2[1] + s, outline="#111", fill=self.LINE_COLOR)
        self.canvas.create_rectangle(lc[0] - s, lc[1] - s, lc[0] + s, lc[1] + s, outline="#111", fill=self.LINE_COLOR)

        # 信息
        if self.rbox:
            bx, by, bw, bh = self.rbox.to_aabb(self.cal_angle_deg)
            cx, cy, w, h = self.rbox.as_center_size()
            self.info_var.set(
                f"ROI: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}  "
                f"AABB(x,y,w,h)={bx},{by},{bw},{bh}   "
                f"校准角度: {self.cal_angle_deg:+.1f}°（±45°）"
            )
        else:
            self.info_var.set(
                f"无 ROI：左键拖动可新建。  校准角度: {self.cal_angle_deg:+.1f}°（±45°）"
            )
