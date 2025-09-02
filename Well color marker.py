import os
import math
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import colorsys
import json
from typing import List, Tuple, Optional


def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def ensure_int_tuple(xy):
    return int(round(xy[0])), int(round(xy[1]))


class PlateColorAnalyzerApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("孔板颜色分析器 - Plate Color Analyzer (Tk)")
        self.geometry("1280x800")
        self.minsize(1100, 700)

        # ---- 状态数据 ----
        self.image_paths: List[str] = []
        self.current_index: int = -1
        self.original_image: Optional[Image.Image] = None
        self.transformed_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None

        self.angle_var = tk.DoubleVar(value=0.0)
        self.flip_h_var = tk.BooleanVar(value=False)
        self.flip_v_var = tk.BooleanVar(value=False)

        self.rows_var = tk.IntVar(value=8)
        self.cols_var = tk.IntVar(value=12)

        self.roi_shape_var = tk.StringVar(value="circle")  # "circle" or "square"
        self.roi_size_var = tk.IntVar(value=15)

        self.rect_p1 = None
        self.rect_p2 = None
        self.setting_rect_mode = False

        # 颜色格式
        self.fmt_rgb255_var = tk.BooleanVar(value=True)
        self.fmt_rgb01_var = tk.BooleanVar(value=False)
        self.fmt_hex_var = tk.BooleanVar(value=True)
        self.fmt_hsv_var = tk.BooleanVar(value=False)
        self.fmt_hsl_var = tk.BooleanVar(value=False)
        self.fmt_gray_var = tk.BooleanVar(value=False)

        self.font_size_var = tk.IntVar(value=14)
        self.use_relative_rect_var = tk.BooleanVar(value=True)
        self.merge_csv_var = tk.BooleanVar(value=False)
        self.export_dir = tk.StringVar(value=os.getcwd())
        self.canvas_bg = "#1e1e1e"

        # 画布交互：缩放与拖动
        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)
        self._pan_active = False
        self._pan_start = (0, 0)

        self.preview_points_cache: List[Tuple[float, float, str]] = []

        # GUI 布局
        self._build_ui()

        # 绑定事件
        # 左键点击作为设置矩形用；拖拽与缩放使用中键/右键/滚轮
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # —— 缩放（Windows/Mac）——
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        # —— 缩放（Linux 常见事件）——
        self.canvas.bind("<Button-4>", lambda e: self.on_zoom(e, wheel_delta=1))
        self.canvas.bind("<Button-5>", lambda e: self.on_zoom(e, wheel_delta=-1))

        # —— 拖拽平移：中键或右键 —— 
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)


    # ---------------- UI ----------------
    def _build_ui(self):
        # 左侧：滚动的控制面板
        left_holder = ttk.Frame(self)
        left_holder.pack(side=tk.LEFT, fill=tk.Y)

        # 用 Canvas + Scrollbar 实现可滚动面板
        self.ctrl_canvas = tk.Canvas(left_holder, borderwidth=0, highlightthickness=0)
        self.ctrl_vsb = ttk.Scrollbar(left_holder, orient="vertical", command=self.ctrl_canvas.yview)
        self.ctrl_canvas.configure(yscrollcommand=self.ctrl_vsb.set)

        self.ctrl_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ctrl_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        # 真正承载控件的 Frame
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas, padding=10)
        self._ctrl_window = self.ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")

        # 面板尺寸与滚动区域联动
        def _on_ctrl_configure(event=None):
            self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))
            # 让内部面板宽度跟随左侧容器宽度（减去滚动条）
            try:
                width = left_holder.winfo_width() - self.ctrl_vsb.winfo_width()
                if width > 50:
                    self.ctrl_canvas.itemconfig(self._ctrl_window, width=width)
            except Exception:
                pass

        self.ctrl_frame.bind("<Configure>", _on_ctrl_configure)
        left_holder.bind("<Configure>", _on_ctrl_configure)

        # 面板滚轮滚动（Windows/Mac）
        self.ctrl_frame.bind_all("<MouseWheel>", self._on_ctrl_mousewheel)
        # 面板滚轮滚动（Linux）
        self.ctrl_frame.bind_all("<Button-4>", lambda e: self.ctrl_canvas.yview_scroll(-1, "units"))
        self.ctrl_frame.bind_all("<Button-5>", lambda e: self.ctrl_canvas.yview_scroll(+1, "units"))

        # 右侧：图像画布区域
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ===== 以下把原先在右侧的控件，全部放到 self.ctrl_frame 内 =====
        # 文件与导航
        frm_files = ttk.LabelFrame(self.ctrl_frame, text="图片与导出", padding=10)
        frm_files.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frm_files, text="加载图片…", command=self.load_images).pack(fill=tk.X)

        nav = ttk.Frame(frm_files); nav.pack(fill=tk.X, pady=5)
        ttk.Button(nav, text="上一张", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(nav, text="下一张", command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # 新增：适配到窗口（重置缩放与居中）
        ttk.Button(frm_files, text="适配到窗口", command=self.fit_canvas_to_image).pack(fill=tk.X, pady=(4, 0))

        ttk.Button(frm_files, text="选择导出目录…", command=self.choose_export_dir).pack(fill=tk.X)
        ttk.Label(frm_files, textvariable=self.export_dir, foreground="#666").pack(fill=tk.X, pady=(4, 0))

        # 变换
        frm_tf = ttk.LabelFrame(self.ctrl_frame, text="旋转 / 翻转", padding=10)
        frm_tf.pack(fill=tk.X, pady=(0, 10))
        row_tf = ttk.Frame(frm_tf); row_tf.pack(fill=tk.X)
        ttk.Label(row_tf, text="旋转角度 (°)：").pack(side=tk.LEFT)
        ent_angle = ttk.Entry(row_tf, textvariable=self.angle_var, width=8)
        ent_angle.pack(side=tk.LEFT, padx=5)
        ttk.Button(row_tf, text="应用", command=self.apply_transform_and_redraw).pack(side=tk.LEFT)
        flips = ttk.Frame(frm_tf); flips.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(flips, text="水平翻转", variable=self.flip_h_var, command=self.apply_transform_and_redraw).pack(side=tk.LEFT)
        ttk.Checkbutton(flips, text="垂直翻转", variable=self.flip_v_var, command=self.apply_transform_and_redraw).pack(side=tk.LEFT)

        # 矩形与网格
        frm_grid = ttk.LabelFrame(self.ctrl_frame, text="矩形与孔板网格", padding=10)
        frm_grid.pack(fill=tk.X, pady=(0, 10))
        rr = ttk.Frame(frm_grid); rr.pack(fill=tk.X)
        ttk.Label(rr, text="行数：").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.rows_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(rr, text="列数：").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.cols_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        btns_rect = ttk.Frame(frm_grid); btns_rect.pack(fill=tk.X, pady=5)
        ttk.Button(btns_rect, text="设置矩形（画布点两次）", command=self.start_set_rect).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns_rect, text="清除矩形", command=self.clear_rect).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(frm_grid, text="预览网格", command=self.redraw).pack(fill=tk.X)

        # 采样设置
        frm_roi = ttk.LabelFrame(self.ctrl_frame, text="采样 ROI", padding=10)
        frm_roi.pack(fill=tk.X, pady=(0, 10))
        shapes = ttk.Frame(frm_roi); shapes.pack(fill=tk.X)
        ttk.Radiobutton(shapes, text="圆形", variable=self.roi_shape_var, value="circle", command=self.redraw).pack(side=tk.LEFT)
        ttk.Radiobutton(shapes, text="方形", variable=self.roi_shape_var, value="square", command=self.redraw).pack(side=tk.LEFT)
        srow = ttk.Frame(frm_roi); srow.pack(fill=tk.X, pady=5)
        ttk.Label(srow, text="ROI 尺寸（像素）：").pack(side=tk.LEFT)
        ttk.Entry(srow, textvariable=self.roi_size_var, width=8).pack(side=tk.LEFT, padx=5)
        fs = ttk.Frame(frm_roi); fs.pack(fill=tk.X)
        ttk.Label(fs, text="标注字号：").pack(side=tk.LEFT)
        ttk.Entry(fs, textvariable=self.font_size_var, width=8).pack(side=tk.LEFT, padx=5)

        # 颜色格式
        frm_fmt = ttk.LabelFrame(self.ctrl_frame, text="导出颜色格式", padding=10)
        frm_fmt.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frm_fmt, text="RGB (0-255)", variable=self.fmt_rgb255_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="RGB (0-1)", variable=self.fmt_rgb01_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HEX (#RRGGBB)", variable=self.fmt_hex_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HSV", variable=self.fmt_hsv_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HSL", variable=self.fmt_hsl_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="灰度", variable=self.fmt_gray_var).pack(anchor=tk.W)

        # 批量设置
        frm_batch = ttk.LabelFrame(self.ctrl_frame, text="批量设置", padding=10)
        frm_batch.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frm_batch, text="使用相对坐标用于批量", variable=self.use_relative_rect_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_batch, text="合并导出一个CSV", variable=self.merge_csv_var).pack(anchor=tk.W)
        bb = ttk.Frame(frm_batch); bb.pack(fill=tk.X, pady=5)
        ttk.Button(bb, text="导出当前", command=self.export_current).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(bb, text="批量导出全部", command=self.batch_export).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # 配置保存/读取
        frm_cfg = ttk.LabelFrame(self.ctrl_frame, text="配置", padding=10)
        frm_cfg.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frm_cfg, text="保存配置…", command=self.save_config).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(frm_cfg, text="加载配置…", command=self.load_config).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)



    # ---------------- 文件与图像 ----------------
    def load_images(self):
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.current_index = 0
        self.load_current_image()

    def choose_export_dir(self):
        d = filedialog.askdirectory(title="选择导出目录")
        if d:
            self.export_dir.set(d)

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_current_image()

    def load_current_image(self):
        if not self.image_paths or self.current_index < 0:
            return
        path = self.image_paths[self.current_index]
        try:
            self.original_image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片：\n{path}\n{e}")
            return
        # 每次更换图片时，变换后的矩形点应清空（避免跨图误用）
        self.rect_p1 = None
        self.rect_p2 = None
        self.apply_transform_and_redraw()
        self.status_var.set(f"已加载：{os.path.basename(path)} [{self.current_index+1}/{len(self.image_paths)}]")

    def apply_transform_and_redraw(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()

        # 翻转
        if self.flip_h_var.get():
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v_var.get():
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 旋转（expand=True 保留完整内容）
        angle = self.angle_var.get() % 360
        if abs(angle) > 1e-6:
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        self.transformed_image = img
        self.fit_canvas_to_image()
        self.redraw()

    # ---------------- 画布显示与交互 ----------------
    def fit_canvas_to_image(self):
        if self.transformed_image is None:
            return
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        iw, ih = self.transformed_image.size
        scale = min(cw / iw, ch / ih)
        self.canvas_scale = scale
        ox = (cw - iw * scale) / 2
        oy = (ch - ih * scale) / 2
        self.canvas_offset = (ox, oy)
        self.redraw()

    def image_to_canvas(self, x, y):
        sx = x * self.canvas_scale + self.canvas_offset[0]
        sy = y * self.canvas_scale + self.canvas_offset[1]
        return sx, sy

    def canvas_to_image(self, sx, sy):
        x = (sx - self.canvas_offset[0]) / self.canvas_scale
        y = (sy - self.canvas_offset[1]) / self.canvas_scale
        return x, y

    def start_set_rect(self):
        if self.transformed_image is None:
            return
        self.setting_rect_mode = True
        self.rect_p1 = None
        self.rect_p2 = None

    def clear_rect(self):
        self.rect_p1 = None
        self.rect_p2 = None
        self.setting_rect_mode = False
        self.redraw()

    def on_canvas_click(self, event):
        if not self.setting_rect_mode or self.transformed_image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        if self.rect_p1 is None:
            self.rect_p1 = (x, y)
        else:
            self.rect_p2 = (x, y)
            self.setting_rect_mode = False
            self.redraw()

    def on_canvas_motion(self, event):
        if self.transformed_image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        iw, ih = self.transformed_image.size
        if 0 <= x < iw and 0 <= y < ih:
            self.status_var.set(f"坐标：({int(x)}, {int(y)})")
        if self.setting_rect_mode and self.rect_p1 is not None:
            # 临时预览矩形
            self.redraw(temp_p2=(x, y))

    def _draw_image_on_canvas(self):
        self.canvas.delete("all")
        if self.transformed_image is None:
            return
        iw, ih = self.transformed_image.size
        disp = self.transformed_image.resize(
            (int(iw * self.canvas_scale), int(ih * self.canvas_scale)),
            Image.BILINEAR
        )
        self.display_image = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.canvas_offset[0], self.canvas_offset[1], anchor=tk.NW, image=self.display_image)
    def _compute_grid_points(self) -> List[Tuple[float, float, str]]:
        """返回所有孔的 (x, y, label) 列表，按行优先。"""
        pts = []
        if self.rect_p1 is None or self.rect_p2 is None:
            return pts
        rows = max(1, self.rows_var.get())
        cols = max(1, self.cols_var.get())

        x0 = min(self.rect_p1[0], self.rect_p2[0])
        y0 = min(self.rect_p1[1], self.rect_p2[1])
        x1 = max(self.rect_p1[0], self.rect_p2[0])
        y1 = max(self.rect_p1[1], self.rect_p2[1])

        dx = 0 if cols == 1 else (x1 - x0) / (cols - 1)
        dy = 0 if rows == 1 else (y1 - y0) / (rows - 1)

        for r in range(rows):
            row_letter = chr(ord('A') + r)
            for c in range(cols):
                x = x0 + c * dx
                y = y0 + r * dy
                label = f"{row_letter}{c+1}"
                pts.append((x, y, label))
        return pts

    def redraw(self, temp_p2=None):
        self._draw_image_on_canvas()
        if self.transformed_image is None:
            return

        # 矩形与网格
        p1 = self.rect_p1
        p2 = self.rect_p2 if temp_p2 is None else temp_p2
        if p1 is not None and p2 is not None:
            # 画选择矩形（按当前缩放转换到画布）
            x0, y0 = p1
            x1, y1 = p2
            sx0, sy0 = self.image_to_canvas(x0, y0)
            sx1, sy1 = self.image_to_canvas(x1, y1)
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline="#00FF99", width=2)

            # 预览网格与标注（仅最终定型后）
            if self.rect_p1 is not None and self.rect_p2 is not None and temp_p2 is None:
                pts = self._compute_grid_points()
                self.preview_points_cache = pts

                roi_size = max(1, self.roi_size_var.get())
                # ——关键变更：ROI 半径按缩放绘制，保持与原图像素比例一致——
                rpx_canvas = (roi_size / 2.0) * self.canvas_scale

                font_color = "#00FFC2"
                for (x, y, label) in pts:
                    sx, sy = self.image_to_canvas(x, y)

                    # ROI边界（按缩放半径）
                    if self.roi_shape_var.get() == "circle":
                        self.canvas.create_oval(sx - rpx_canvas, sy - rpx_canvas,
                                                sx + rpx_canvas, sy + rpx_canvas,
                                                outline="#FFD200")
                    else:
                        self.canvas.create_rectangle(sx - rpx_canvas, sy - rpx_canvas,
                                                     sx + rpx_canvas, sy + rpx_canvas,
                                                     outline="#FFD200")
                    # 点中心
                    self.canvas.create_oval(sx - 2, sy - 2, sx + 2, sy + 2, fill="#FF4081", outline="")

                    # 标注（字号也随缩放系数变化，原有逻辑保留）
                    self.canvas.create_text(
                        sx, sy - rpx_canvas - 6,
                        text=label, fill=font_color,
                        font=("Arial", max(8, int(self.font_size_var.get() * self.canvas_scale * 0.75))),
                        anchor=tk.S
                    )
    def on_zoom(self, event, wheel_delta: Optional[int] = None):
        """
        鼠标滚轮缩放。以光标所在点为中心缩放，保持该图像点在画布上的位置不变。
        - Windows/Mac: event.delta 正负
        - Linux: 通过 wheel_delta 参数传入 +1/-1
        """
        if self.transformed_image is None:
            return

        # 计算滚轮方向
        delta = 0
        if wheel_delta is not None:
            delta = wheel_delta
        elif hasattr(event, "delta"):
            delta = 1 if event.delta > 0 else -1
        if delta == 0:
            return

        # 当前缩放与限制
        old_scale = self.canvas_scale
        step = 1.1
        new_scale = old_scale * (step if delta > 0 else 1.0 / step)
        new_scale = max(0.05, min(20.0, new_scale))  # 缩放范围

        # 以光标处的图像坐标为锚点
        ix, iy = self.canvas_to_image(event.x, event.y)

        # 更新偏移，使缩放后该图像点仍落在光标位置
        self.canvas_scale = new_scale
        self.canvas_offset = (event.x - ix * new_scale, event.y - iy * new_scale)
        self.redraw()

    def on_pan_start(self, event):
        """中键或右键按下开始平移"""
        self._pan_active = True
        self._pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        """中键或右键拖动进行平移"""
        if not self._pan_active:
            return
        sx0, sy0 = self._pan_start
        dx, dy = event.x - sx0, event.y - sy0
        ox, oy = self.canvas_offset
        self.canvas_offset = (ox + dx, oy + dy)
        self._pan_start = (event.x, event.y)
        self.redraw()

    def _on_ctrl_mousewheel(self, event):
        """
        左侧操作面板滚动：防止滚轮在面板上时触发图像缩放。
        仅滚动面板，不传递到画布。
        """
        if event.widget is self.canvas:
            return  # 让画布自己的缩放处理
        # Windows/Mac
        if hasattr(event, "delta") and event.delta != 0:
            self.ctrl_canvas.yview_scroll(-1 if event.delta > 0 else +1, "units")
    # ---------------- 计算与导出 ----------------
    def _sample_color_at(self, img: Image.Image, x: float, y: float, roi_shape: str, roi_size: int) -> Optional[Tuple[float,float,float]]:
        """在 img 上以 (x,y) 为中心，按 ROI 计算平均 RGB(0-255)。返回 (r,g,b) 或 None."""
        w, h = img.size
        half = roi_size / 2.0
        left = int(math.floor(x - half))
        upper = int(math.floor(y - half))
        right = int(math.ceil(x + half))
        lower = int(math.ceil(y + half))

        # 与图像边界裁剪
        left = max(0, left); upper = max(0, upper)
        right = min(w, right); lower = min(h, lower)
        if right <= left or lower <= upper:
            return None

        crop = img.crop((left, upper, right, lower))
        arr = np.asarray(crop, dtype=np.float32)  # HxWx3

        if roi_shape == "circle":
            hh, ww = arr.shape[:2]
            yy, xx = np.ogrid[0:hh, 0:ww]
            cx = (x - left)
            cy = (y - upper)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (half) ** 2
            # 避免无像素
            if not np.any(mask):
                return None
            sel = arr[mask]
        else:
            sel = arr.reshape(-1, arr.shape[-1])

        if sel.size == 0:
            return None
        mean = sel.mean(axis=0)
        return (float(mean[0]), float(mean[1]), float(mean[2]))

    def _build_color_columns(self, rgb255):
        r255, g255, b255 = rgb255
        cols = {}
        # 归一化
        r01, g01, b01 = r255/255.0, g255/255.0, b255/255.0
        if self.fmt_rgb255_var.get():
            cols["R"] = int(round(r255)); cols["G"] = int(round(g255)); cols["B"] = int(round(b255))
        if self.fmt_rgb01_var.get():
            cols["R01"] = round(r01, 4); cols["G01"] = round(g01, 4); cols["B01"] = round(b01, 4)
        if self.fmt_hex_var.get():
            cols["HEX"] = rgb_to_hex(r255, g255, b255)
        if self.fmt_hsv_var.get():
            h, s, v = colorsys.rgb_to_hsv(r01, g01, b01)  # h[0..1], s,v [0..1]
            cols["H"] = round(h * 360.0, 2); cols["S"] = round(s, 4); cols["V"] = round(v, 4)
        if self.fmt_hsl_var.get():
            h, l, s = colorsys.rgb_to_hls(r01, g01, b01)  # 注意 HLS 顺序
            cols["Hsl_H"] = round(h * 360.0, 2); cols["Hsl_S"] = round(s, 4); cols["Hsl_L"] = round(l, 4)
        if self.fmt_gray_var.get():
            # 相对亮度加权
            gray = 0.2126*r255 + 0.7152*g255 + 0.0722*b255
            cols["Gray"] = int(round(gray))
        return cols

    def _annotate_image(self, img: Image.Image, points: List[Tuple[float,float,str]], roi_shape: str, roi_size: int, font_size: int) -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        rpx = roi_size / 2.0
        # 简单矢量字体（PIL 内置）
        for (x, y, label) in points:
            if roi_shape == "circle":
                bbox = (x-rpx, y-rpx, x+rpx, y+rpx)
                draw.ellipse(bbox, outline=(255,210,0), width=2)
            else:
                bbox = (x-rpx, y-rpx, x+rpx, y+rpx)
                draw.rectangle(bbox, outline=(255,210,0), width=2)
            # 中心点
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(255,64,129))
            # 标注（居中，放在 ROI 上方）
            tx, ty = x, y - rpx - 6
            draw.text((tx, ty), label, fill=(0,255,194), anchor="ms", stroke_width=1, stroke_fill=(0,0,0))
        return out

    def export_current(self):
        if self.transformed_image is None or not self.image_paths:
            messagebox.showwarning("提示", "请先加载图片。")
            return
        if self.rect_p1 is None or self.rect_p2 is None:
            messagebox.showwarning("提示", "请先设置矩形对角点。")
            return

        try:
            self._export_single(self.current_index)
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _export_single(self, index: int, merged_writer: Optional[csv.DictWriter]=None, merged_file: Optional[any]=None):
        # 载入图像并应用当前变换
        path = self.image_paths[index]
        img0 = Image.open(path).convert("RGB")
        img = self._apply_current_transform_to_image(img0)

        # 计算本图的矩形（如果使用相对坐标则反解）
        if self.use_relative_rect_var.get():
            # 将当前 rect_p1,p2 先换成相对坐标（基于当前显示图的宽高），再应用到本图 img.size
            # 注意：我们把 rect_p1/p2 存为当前“变换后图”的绝对像素。对其它图：先求相对，再乘本图尺寸。
            cur_w, cur_h = self.transformed_image.size
            rel_p1 = (self.rect_p1[0]/cur_w, self.rect_p1[1]/cur_h)
            rel_p2 = (self.rect_p2[0]/cur_w, self.rect_p2[1]/cur_h)
            iw, ih = img.size
            p1 = (rel_p1[0]*iw, rel_p1[1]*ih)
            p2 = (rel_p2[0]*iw, rel_p2[1]*ih)
        else:
            # 使用绝对坐标：仅当各图尺寸与当前图一致时才合理
            p1 = self.rect_p1
            p2 = self.rect_p2
            iw, ih = img.size
            cw, ch = self.transformed_image.size
            if (int(iw)!=int(cw)) or (int(ih)!=int(ch)):
                raise ValueError("当前使用绝对坐标，但批量中的图片尺寸与当前图不一致。请切换为“使用相对坐标用于批量”。")

        # 构造一个临时上下文来计算网格
        rows = max(1, self.rows_var.get())
        cols = max(1, self.cols_var.get())
        x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
        dx = 0 if cols == 1 else (x1 - x0) / (cols - 1)
        dy = 0 if rows == 1 else (y1 - y0) / (rows - 1)

        # 采样并汇总
        roi_shape = self.roi_shape_var.get()
        roi_size = max(1, self.roi_size_var.get())
        points = []
        records = []

        for r in range(rows):
            row_letter = chr(ord('A') + r)
            for c in range(cols):
                x = x0 + c * dx
                y = y0 + r * dy
                label = f"{row_letter}{c+1}"
                rgb = self._sample_color_at(img, x, y, roi_shape, roi_size)
                if rgb is None:
                    continue
                points.append((x, y, label))
                row = {
                    "filename": os.path.basename(path),
                    "row": row_letter,
                    "col": c+1,
                    "label": label,
                    "x": int(round(x)),
                    "y": int(round(y)),
                }
                row.update(self._build_color_columns(rgb))
                records.append(row)

        # 导出标注图
        annotated = self._annotate_image(img, points, roi_shape, roi_size, self.font_size_var.get())
        base = os.path.splitext(os.path.basename(path))[0]
        out_img_path = os.path.join(self.export_dir.get(), f"{base}_annotated.png")
        annotated.save(out_img_path)

        # 导出CSV（若 merged_writer 提供，则写入合并文件，否则单独写文件）
        if merged_writer is not None:
            # merged_writer 的字段集需要统一，确保有全部列
            for rec in records:
                merged_writer.writerow(rec)
        else:
            # 单文件写出
            if records:
                out_csv_path = os.path.join(self.export_dir.get(), f"{base}_data.csv")
                # 收集所有字段（基础 + 选中的颜色格式列）
                fieldnames = ["filename", "row", "col", "label", "x", "y"]
                # 从首行推断颜色列（与当前选择一致）
                extra_cols = [k for k in records[0].keys() if k not in fieldnames]
                fieldnames += extra_cols
                with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for rec in records:
                        writer.writerow(rec)

        return len(records), len(points), out_img_path

    def _apply_current_transform_to_image(self, img: Image.Image) -> Image.Image:
        # 复制当前变换逻辑：翻转 -> 旋转
        if self.flip_h_var.get():
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v_var.get():
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        angle = self.angle_var.get() % 360
        if abs(angle) > 1e-6:
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        return img

    def batch_export(self):
        if not self.image_paths:
            messagebox.showwarning("提示", "请先加载至少一张图片。")
            return
        if self.rect_p1 is None or self.rect_p2 is None:
            messagebox.showwarning("提示", "请先在当前图上设置矩形对角点。")
            return
        # 合并CSV模式
        merged_path = os.path.join(self.export_dir.get(), "batch_data.csv") if self.merge_csv_var.get() else None
        merged_file = None
        merged_writer = None
        fieldnames = None
        total_points = 0
        total_records = 0
        try:
            if merged_path:
                merged_file = open(merged_path, "w", newline="", encoding="utf-8-sig")
                # 先不知道字段，写时动态收集（用首图写出头）
                merged_writer = None

            for idx, _ in enumerate(self.image_paths):
                # 先单独计算一次，得到 records 与字段
                path = self.image_paths[idx]
                img0 = Image.open(path).convert("RGB")
                img = self._apply_current_transform_to_image(img0)

                # 计算矩形在该图上的位置
                if self.use_relative_rect_var.get():
                    cur_w, cur_h = self.transformed_image.size
                    rel_p1 = (self.rect_p1[0]/cur_w, self.rect_p1[1]/cur_h)
                    rel_p2 = (self.rect_p2[0]/cur_w, self.rect_p2[1]/cur_h)
                    iw, ih = img.size
                    p1 = (rel_p1[0]*iw, rel_p1[1]*ih)
                    p2 = (rel_p2[0]*iw, rel_p2[1]*ih)
                else:
                    p1 = self.rect_p1
                    p2 = self.rect_p2
                    iw, ih = img.size
                    cw, ch = self.transformed_image.size
                    if (int(iw)!=int(cw)) or (int(ih)!=int(ch)):
                        raise ValueError("当前使用绝对坐标，但批量中的图片尺寸与当前图不一致。请切换为“使用相对坐标用于批量”。")

                # 构建点与采样
                rows = max(1, self.rows_var.get())
                cols = max(1, self.cols_var.get())
                x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
                dx = 0 if cols == 1 else (x1 - x0) / (cols - 1)
                dy = 0 if rows == 1 else (y1 - y0) / (rows - 1)
                roi_shape = self.roi_shape_var.get()
                roi_size = max(1, self.roi_size_var.get())

                # 采样
                local_records = []
                points = []
                for r in range(rows):
                    row_letter = chr(ord('A') + r)
                    for c in range(cols):
                        x = x0 + c * dx
                        y = y0 + r * dy
                        label = f"{row_letter}{c+1}"
                        rgb = self._sample_color_at(img, x, y, roi_shape, roi_size)
                        if rgb is None:
                            continue
                        points.append((x,y,label))
                        rec = {
                            "filename": os.path.basename(path),
                            "row": row_letter,
                            "col": c+1,
                            "label": label,
                            "x": int(round(x)),
                            "y": int(round(y)),
                        }
                        rec.update(self._build_color_columns(rgb))
                        local_records.append(rec)

                # 导出标注图
                annotated = self._annotate_image(img, points, roi_shape, roi_size, self.font_size_var.get())
                base = os.path.splitext(os.path.basename(path))[0]
                out_img_path = os.path.join(self.export_dir.get(), f"{base}_annotated.png")
                annotated.save(out_img_path)

                # 写合并CSV或单独CSV
                if self.merge_csv_var.get():
                    if local_records:
                        if merged_writer is None:
                            # 初始化合并 writer 的列
                            fieldnames = ["filename", "row", "col", "label", "x", "y"]
                            extra_cols = [k for k in local_records[0].keys() if k not in fieldnames]
                            fieldnames += extra_cols
                            merged_writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
                            merged_writer.writeheader()
                        for rec in local_records:
                            merged_writer.writerow(rec)
                else:
                    if local_records:
                        out_csv_path = os.path.join(self.export_dir.get(), f"{base}_data.csv")
                        # 字段统一
                        fieldnames = ["filename", "row", "col", "label", "x", "y"]
                        extra_cols = [k for k in local_records[0].keys() if k not in fieldnames]
                        fieldnames += extra_cols
                        with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                            w = csv.DictWriter(f, fieldnames=fieldnames)
                            w.writeheader()
                            for rec in local_records:
                                w.writerow(rec)

                total_points += len(points)
                total_records += len(local_records)

            self.status_var.set(f"批量导出完成：{len(self.image_paths)} 张图片，采样点 {total_points}，记录 {total_records}。")
            messagebox.showinfo("完成", f"批量导出完成。\n总记录数：{total_records}\n导出目录：\n{self.export_dir.get()}")
        except Exception as e:
            messagebox.showerror("批量导出失败", str(e))
        finally:
            if merged_file:
                merged_file.close()

    # ---------------- 配置保存/加载 ----------------
    def save_config(self):
        if self.transformed_image is None:
            messagebox.showwarning("提示", "请先加载图片（用于记录矩形相对坐标）。")
            return
        path = filedialog.asksaveasfilename(
            title="保存配置为 JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        cfg = {
            "angle": self.angle_var.get(),
            "flip_h": self.flip_h_var.get(),
            "flip_v": self.flip_v_var.get(),
            "rows": self.rows_var.get(),
            "cols": self.cols_var.get(),
            "roi_shape": self.roi_shape_var.get(),
            "roi_size": self.roi_size_var.get(),
            "font_size": self.font_size_var.get(),
            "fmt": {
                "rgb255": self.fmt_rgb255_var.get(),
                "rgb01": self.fmt_rgb01_var.get(),
                "hex": self.fmt_hex_var.get(),
                "hsv": self.fmt_hsv_var.get(),
                "hsl": self.fmt_hsl_var.get(),
                "gray": self.fmt_gray_var.get(),
            },
            "use_relative_rect": self.use_relative_rect_var.get(),
            "merge_csv": self.merge_csv_var.get(),
            "export_dir": self.export_dir.get(),
        }
        # 记录矩形（用相对坐标，便于跨图）
        if self.rect_p1 is not None and self.rect_p2 is not None:
            w, h = self.transformed_image.size
            cfg["rect_rel"] = [
                [self.rect_p1[0]/w, self.rect_p1[1]/h],
                [self.rect_p2[0]/w, self.rect_p2[1]/h]
            ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"配置已保存：{os.path.basename(path)}")

    def load_config(self):
        path = filedialog.askopenfilename(
            title="加载配置 JSON",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.angle_var.set(cfg.get("angle", 0.0))
            self.flip_h_var.set(cfg.get("flip_h", False))
            self.flip_v_var.set(cfg.get("flip_v", False))
            self.rows_var.set(cfg.get("rows", 8))
            self.cols_var.set(cfg.get("cols", 12))
            self.roi_shape_var.set(cfg.get("roi_shape", "circle"))
            self.roi_size_var.set(cfg.get("roi_size", 15))
            self.font_size_var.set(cfg.get("font_size", 14))
            fmt = cfg.get("fmt", {})
            self.fmt_rgb255_var.set(fmt.get("rgb255", True))
            self.fmt_rgb01_var.set(fmt.get("rgb01", False))
            self.fmt_hex_var.set(fmt.get("hex", True))
            self.fmt_hsv_var.set(fmt.get("hsv", False))
            self.fmt_hsl_var.set(fmt.get("hsl", False))
            self.fmt_gray_var.set(fmt.get("gray", False))
            self.use_relative_rect_var.set(cfg.get("use_relative_rect", True))
            self.merge_csv_var.set(cfg.get("merge_csv", False))
            self.export_dir.set(cfg.get("export_dir", self.export_dir.get()))

            # 如果当前已有图，则恢复矩形（相对坐标）
            if self.transformed_image is not None and "rect_rel" in cfg:
                w, h = self.transformed_image.size
                rp1, rp2 = cfg["rect_rel"]
                self.rect_p1 = (rp1[0]*w, rp1[1]*h)
                self.rect_p2 = (rp2[0]*w, rp2[1]*h)

            self.apply_transform_and_redraw()
            self.status_var.set(f"配置已加载：{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败：{e}")

    # ---------------- 主程序入口 ----------------


if __name__ == "__main__":
    app = PlateColorAnalyzerApp()
    app.mainloop()
