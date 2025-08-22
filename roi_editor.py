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
    """

    HANDLE_SIZE = 5          # 角点可见更清晰
    CORNER_LEN = 18          # L 形角标长度


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

        # 状态
        self.dragging = False
        self.drag_mode = None  # "move" | "new" | "resize-<tag>" | "pan"
        self.drag_start = (0, 0)  # canvas coords
        self.orig_box: Optional[Box] = None

        # 事件绑定
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<Button-2>", self._on_mid_down)  # 有些平台中键平移
        self.canvas.bind("<B2-Motion>", self._on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_mid_up)

        self.canvas.bind("<Button-3>", self._on_right_down)  # 右键平移
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)

        # 滚轮（Win/Mac/Linux 兼容）
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)       # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)   # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)   # Linux down

        # 键盘微调
        self.canvas.focus_set()
        self.canvas.bind("<Up>", lambda e: self._nudge(0, -1, e))
        self.canvas.bind("<Down>", lambda e: self._nudge(0, 1, e))
        self.canvas.bind("<Left>", lambda e: self._nudge(-1, 0, e))
        self.canvas.bind("<Right>", lambda e: self._nudge(1, 0, e))

        # 信息栏
        self.info_var = tk.StringVar(value="")
        self.lbl_info = ttk.Label(self, textvariable=self.info_var, foreground="#ddd")
        self.lbl_info.pack(anchor="w", padx=6, pady=4)

    # ------------ 公有 API ------------
    def set_image(self, img_bgr: np.ndarray):
        self.img_bgr = img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(rgb)
        # 初始缩放以适配窗口
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self._fit_to_window()
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

    # ------------ 坐标转换 ------------
    def img_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        return ((cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale)

    # ------------ 事件处理 ------------
    def _on_resize(self, _):
        self.redraw()

    def _on_left_down(self, e):
        self.canvas.focus_set()
        if self.img_pil is None:
            return
        cx, cy = e.x, e.y
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
        # 空白处 → 新建
        self.drag_mode = "new"
        self.dragging = True
        self.drag_start = (cx, cy)
        ix, iy = self.canvas_to_img(cx, cy)
        self.box = Box(int(ix), int(iy), 1, 1)
        self.redraw()

    def _on_left_drag(self, e):
        if not self.dragging or self.img_pil is None or self.box is None:
            return
        cx, cy = e.x, e.y
        dx, dy = cx - self.drag_start[0], cy - self.drag_start[1]
        H = self.img_bgr.shape[0]
        W = self.img_bgr.shape[1]

        if self.drag_mode == "move":
            # 画布位移转像素位移
            mx = int(round(dx / self.scale))
            my = int(round(dy / self.scale))
            self.box.x = self.orig_box.x + mx
            self.box.y = self.orig_box.y + my
            self.box.clamp(W, H)
        elif self.drag_mode and self.drag_mode.startswith("resize-"):
            tag = self.drag_mode.split("-", 1)[1]
            # 将当前拖动点转换到图像坐标
            ix, iy = self.canvas_to_img(cx, cy)
            ix = int(round(ix))
            iy = int(round(iy))
            b = Box(self.orig_box.x, self.orig_box.y, self.orig_box.w, self.orig_box.h)

            if "l" in tag:
                # 左边/左上/左下
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
        # Windows: e.delta 正负 120 的倍数；Mac: 可能是较小数值
        step = 1.1 if e.delta > 0 else 1 / 1.1
        self._zoom_at(step, e.x, e.y)

    def _on_mousewheel_linux(self, e):
        # Linux: Button-4 上滚、Button-5 下滚
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

    # ------------ 内部工具 ------------
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
        # 保持光标指向的图像点不动：调整 offset
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

    def _hit_test_handles(self, cx: float, cy: float) -> Optional[str]:
        if self.box is None:
            return None
        # 计算 8 个句柄：四角(lt, rt, rb, lb) + 四边(l, t, r, b)
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
                # 边的 tag 用 l/t/r/b，角用 lt/rt/rb/lb
                return tag
        return None

    def _point_in_box_canvas(self, cx: float, cy: float, box: Box) -> bool:
        x0, y0 = self.img_to_canvas(box.x, box.y)
        x1, y1 = self.img_to_canvas(box.x + box.w, box.y + box.h)
        return (x0 <= cx <= x1) and (y0 <= cy <= y1)

    # ------------ 绘制 ------------

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

        if not self.box:
            self.info_var.set("无 ROI：左键拖动可新建。")
            return

        # ROI 基础
        x, y, w, h = self.box.as_tuple()
        x0, y0 = self.img_to_canvas(x, y)
        x1, y1 = self.img_to_canvas(x + w, y + h)

        # 半透明区域
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#20f28b", width=2)
        #self.canvas.create_rectangle(x0, y0, x1, y1, outline="")

        # L 形角标 (四角)
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

        # 圆形手柄（四角 + 四边中点）
        hs = self.HANDLE_SIZE
        handles = [
            ("lt", x0, y0), ("rt", x1, y0), ("rb", x1, y1), ("lb", x0, y1),
            ("l", x0, (y0 + y1) / 2), ("t", (x0 + x1) / 2, y0),
            ("r", x1, (y0 + y1) / 2), ("b", (x0 + x1) / 2, y1),
        ]
        for tag, hx, hy in handles:
            self.canvas.create_oval(hx - hs, hy - hs, hx + hs, hy + hs,
                                    outline="#111", fill="#fcf802")

        # 信息
        self.info_var.set(f"ROI: x={x}, y={y}, w={w}, h={h}")
