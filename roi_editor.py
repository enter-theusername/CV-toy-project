# roi_editor.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageTk
import cv2
import math


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def clamp(self, W: int, H: int):
        self.x = max(0, min(self.x, W - 1))
        self.y = max(0, min(self.y, H - 1))
        self.w = max(1, min(self.w, W - self.x))
        self.h = max(1, min(self.h, H - self.y))


class ROIEditorCanvas(ttk.Frame):
    """
    一个可缩放/平移的图像画布，支持交互式拖拽编辑矩形 ROI：
    - 左键拖动角/边修改 ROI，左键拖动内部移动 ROI
    - 左键空白处拖拽新建 ROI
    - 右键拖动画布平移
    - 滚轮缩放（以光标为中心）
    - 方向键/Shift+方向键 像素级微调

    扩展：新增“校准横线”以手动校准倾斜（±15°）
    - 永久显示；支持拖动（平移）与端点旋转（保持长度不变）
    - 旋转角度限制在 [-15°, +15°]
    - 通过 get_angle_deg() 提供当前角度（用于下游裁剪后“摆正”）
    """
    HANDLE_SIZE = 5        # ROI 角点可见更清晰
    CORNER_LEN = 18        # ROI L 形角标长度
    LINE_HANDLE_SZ = 6     # 校准线端点/中心手柄可见尺寸
    LINE_COLOR = "#00d1ff"

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

        # ROI
        self.box: Optional[Box] = None

        # 校准线（以图像坐标存储：中心+长度+角度）
        self.cal_cx: float = 0.0
        self.cal_cy: float = 0.0
        self.cal_len: float = 0.0
        self.cal_angle_deg: float = 0.0  # [-15,+15]

        # 状态
        self.dragging = False
        self.drag_mode = None  # "move" | "new" | "resize-<tag>" | "pan" | "line-move" | "line-rot-p1" | "line-rot-p2"
        self.drag_start = (0, 0)  # canvas coords
        self.orig_box: Optional[Box] = None
        self.orig_line = None     # (cx,cy,len,angle) 在旋转/拖动校准线时保存初始值

        # 事件绑定
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-2>", self._on_mid_down)      # 有些平台中键平移
        self.canvas.bind("<B2-Motion>", self._on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_mid_up)
        self.canvas.bind("<Button-3>", self._on_right_down)    # 右键平移
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)
        # 滚轮（Win/Mac/Linux 兼容）
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)  # Linux down

        # 键盘微调
        self.canvas.focus_set()
        self.canvas.bind("<Up>",    lambda e: self._nudge(0, -1, e))
        self.canvas.bind("<Down>",  lambda e: self._nudge(0,  1, e))
        self.canvas.bind("<Left>",  lambda e: self._nudge(-1, 0, e))
        self.canvas.bind("<Right>", lambda e: self._nudge(1,  0, e))

        # 信息栏
        self.info_var = tk.StringVar(value="")
        self.lbl_info = ttk.Label(self, textvariable=self.info_var, foreground="#ddd")
        self.lbl_info.pack(anchor="w", padx=6, pady=4)

    # --------------- 公有 API ---------------
    def set_image(self, img_bgr: np.ndarray):
        self.img_bgr = img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(rgb)
        # 初始缩放以适配窗口
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self._fit_to_window()
        # 初始化校准线（位于图上方 20% 处，长度为宽度的 60%，角度 0）
        H, W = img_bgr.shape[0], img_bgr.shape[1]
        self.cal_cx = W / 2.0
        self.cal_cy = max(10.0, H * 0.18)
        self.cal_len = max(50.0, W * 0.6)
        self.cal_angle_deg = 0.0
        self.redraw()

    def set_roi(self, box: Tuple[int, int, int, int] | None):
        if box is None:
            self.box = None
        else:
            self.box = Box(*map(int, box))
        self._clamp_box()
        self.redraw()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.box.as_tuple() if self.box else None

    def get_angle_deg(self) -> float:
        """返回当前校准线角度（度），范围 [-15, +15]。正值=顺时针（向右上倾斜）。"""
        return float(self.cal_angle_deg)

    # --------------- 坐标转换 ---------------
    def img_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        return ((cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale)

    # --------------- 事件处理 ---------------
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
            if tag_line == "line_c" or tag_line == "line_seg":
                self.drag_mode = "line-move"
            elif tag_line == "line_p1":
                self.drag_mode = "line-rot-p1"
            elif tag_line == "line_p2":
                self.drag_mode = "line-rot-p2"
            return

        # ② ROI 操作
        if self.box:
            tag = self._hit_test_handles(cx, cy)
            if tag:
                # 拖拽边/角
                self.drag_mode = f"resize-{tag}"
                self.dragging = True
                self.drag_start = (cx, cy)
                self.orig_box = Box(self.box.x, self.box.y, self.box.w, self.box.h)
                return

            # 点在 ROI 内 → 移动
            if self._point_in_box_canvas(cx, cy, self.box):
                self.drag_mode = "move"
                self.dragging = True
                self.drag_start = (cx, cy)
                self.orig_box = Box(self.box.x, self.box.y, self.box.w, self.box.h)
                return

        # ③ 空白处 → 新建 ROI
        self.drag_mode = "new"
        self.dragging = True
        self.drag_start = (cx, cy)
        ix, iy = self.canvas_to_img(cx, cy)
        self.box = Box(int(ix), int(iy), 1, 1)
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
                # 被拖动端点的向量（相对中心）
                vx = ix - ocx
                vy = iy - ocy
                if self.drag_mode == "line-rot-p2":
                    # 端点 p2：直接用当前向量
                    pass
                else:
                    # 端点 p1：与 p2 相反方向
                    vx, vy = -vx, -vy
                ang = math.degrees(math.atan2(vy, vx))  # [-180,180]
                # 限制到 [-15, +15]
                ang = max(-30.0, min(30.0, ang))
                self.cal_angle_deg = float(ang)
                # 中心不变，长度固定
                self.cal_cx, self.cal_cy = ocx, ocy
                self.cal_len = olen

            self.redraw()
            return

        # —— ROI 操作 —— #
        if self.box is None:
            return
        if self.drag_mode == "move":
            mx = int(round(dx / self.scale))
            my = int(round(dy / self.scale))
            self.box.x = self.orig_box.x + mx
            self.box.y = self.orig_box.y + my
            self.box.clamp(W, H)

        elif self.drag_mode and self.drag_mode.startswith("resize-"):
            tag = self.drag_mode.split("-", 1)[1]
            ix, iy = self.canvas_to_img(cx, cy)
            ix = int(round(ix))
            iy = int(round(iy))
            b = Box(self.orig_box.x, self.orig_box.y, self.orig_box.w, self.orig_box.h)
            if "l" in tag:
                new_x = min(max(0, ix), b.x + b.w - 1)
                new_w = b.x + b.w - new_x
                b.x, b.w = new_x, max(1, new_w)
            if "r" in tag:
                new_w = max(1, ix - b.x)
                b.w = new_w
            if "t" in tag:
                new_y = min(max(0, iy), b.y + b.h - 1)
                new_h = b.y + b.h - new_y
                b.y, b.h = new_y, max(1, new_h)
            if "b" in tag:
                new_h = max(1, iy - b.y)
                b.h = new_h
            b.clamp(W, H)
            self.box = b

        elif self.drag_mode == "new":
            ix0, iy0 = self.canvas_to_img(*self.drag_start)
            ix1, iy1 = self.canvas_to_img(cx, cy)
            x0, y0 = int(round(min(ix0, ix1))), int(round(min(iy0, iy1)))
            x1, y1 = int(round(max(ix0, ix1))), int(round(max(iy0, iy1)))
            self.box = Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            self._clamp_box()

        self.redraw()

    def _on_left_up(self, _):
        self.dragging = False
        self.drag_mode = None
        self.orig_box = None
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
        if self.box is None:
            return
        k = 10 if (e.state & 0x0001 or e.state & 0x0004) else 1  # Shift/Ctrl 都加速
        self.box.x += dx * k
        self.box.y += dy * k
        self._clamp_box()
        self.redraw()

    # --------------- 内部工具 ---------------
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
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - old_scale) < 1e-6:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        self.scale = new_scale
        nx, ny = self.img_to_canvas(ix, iy)
        self.offset_x += (cx - nx)
        self.offset_y += (cy - ny)
        self.redraw()

    def _clamp_box(self):
        if self.img_pil is None or self.box is None:
            return
        W, H = self.img_pil.size
        self.box.clamp(W, H)

    # ROI 句柄命中测试
    def _hit_test_handles(self, cx: float, cy: float) -> Optional[str]:
        if self.box is None:
            return None
        x, y, w, h = self.box.as_tuple()
        pts = {
            "lt": (x, y), "rt": (x + w, y),
            "rb": (x + w, y + h), "lb": (x, y + h),
            "l": (x, y + h // 2), "t": (x + w // 2, y),
            "r": (x + w, y + h // 2), "b": (x + w // 2, y + h),
        }
        for tag, (ix, iy) in pts.items():
            hx, hy = self.img_to_canvas(ix, iy)
            if abs(cx - hx) <= self.HANDLE_SIZE and abs(cy - hy) <= self.HANDLE_SIZE:
                return tag
        return None

    def _point_in_box_canvas(self, cx: float, cy: float, box: Box) -> bool:
        x0, y0 = self.img_to_canvas(box.x, box.y)
        x1, y1 = self.img_to_canvas(box.x + box.w, box.y + box.h)
        return (x0 <= cx <= x1) and (y0 <= cy <= y1)

    # 校准线命中测试（端点/中心/近线）
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
        # 使用点到线段距离（以“画布像素”为单位）
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
        # 点到线段距离
        vx, vy = x2 - x1, y2 - y1
        wx, wy = cx - x1, cy - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            return math.hypot(wx, wy)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        projx = x1 + t * vx
        projy = y1 + t * vy
        return math.hypot(cx - projx, cy - projy)

    # --------------- 绘制 ---------------
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

        # ROI 基础
        if self.box:
            x, y, w, h = self.box.as_tuple()
            x0, y0 = self.img_to_canvas(x, y)
            x1, y1 = self.img_to_canvas(x + w, y + h)

            # ROI 框与角标
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#20f28b", width=2)

            L = self.CORNER_LEN
            # 左上
            self.canvas.create_line(x0, y0, x0 + L, y0, fill="#20f28b", width=1)
            self.canvas.create_line(x0, y0, x0, y0 + L, fill="#20f28b", width=1)
            # 右上
            self.canvas.create_line(x1, y0, x1 - L, y0, fill="#20f28b", width=1)
            self.canvas.create_line(x1, y0, x1, y0 + L, fill="#20f28b", width=1)
            # 右下
            self.canvas.create_line(x1, y1, x1 - L, y1, fill="#20f28b", width=1)
            self.canvas.create_line(x1, y1, x1, y1 - L, fill="#20f28b", width=1)
            # 左下
            self.canvas.create_line(x0, y1, x0 + L, y1, fill="#20f28b", width=1)
            self.canvas.create_line(x0, y1, x0, y1 - L, fill="#20f28b", width=1)

            # 句柄（四角+四边中点）
            hs = self.HANDLE_SIZE
            handles = [
                ("lt", x0, y0), ("rt", x1, y0), ("rb", x1, y1), ("lb", x0, y1),
                ("l", x0, (y0 + y1) / 2), ("t", (x0 + x1) / 2, y0),
                ("r", x1, (y0 + y1) / 2), ("b", (x0 + x1) / 2, y1),
            ]
            for tag, hx, hy in handles:
                self.canvas.create_oval(hx - hs, hy - hs, hx + hs, hy + hs,
                                        outline="#111", fill="#fcf802")

        # 校准线（最后绘制，始终可见）
        (p1x, p1y), (p2x, p2y) = self._line_endpoints_img()
        lp1 = self.img_to_canvas(p1x, p1y)
        lp2 = self.img_to_canvas(p2x, p2y)
        lc  = self.img_to_canvas(self.cal_cx, self.cal_cy)

        self.canvas.create_line(lp1[0], lp1[1], lp2[0], lp2[1], fill=self.LINE_COLOR, width=2)
        # 端点圆 & 中心方块
        s = self.LINE_HANDLE_SZ
        self.canvas.create_oval(lp1[0] - s, lp1[1] - s, lp1[0] + s, lp1[1] + s, outline="#111", fill=self.LINE_COLOR)
        self.canvas.create_oval(lp2[0] - s, lp2[1] - s, lp2[0] + s, lp2[1] + s, outline="#111", fill=self.LINE_COLOR)
        self.canvas.create_rectangle(lc[0] - s, lc[1] - s, lc[0] + s, lc[1] + s, outline="#111", fill=self.LINE_COLOR)

        # 信息
        if self.box:
            bx, by, bw, bh = self.box.as_tuple()
            self.info_var.set(f"ROI: x={bx}, y={by}, w={bw}, h={bh} | 校准角度: {self.cal_angle_deg:+.1f}°（±15°）")
        else:
            self.info_var.set(f"无 ROI：左键拖动可新建。 | 校准角度: {self.cal_angle_deg:+.1f}°（±15°）")
