# app_desktop_gui_roi.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from gel_core import (
    auto_white_balance, detect_gel_regions,
    lanes_by_projection, lanes_uniform,
    detect_bands_along_y,  # 旧的直线模式备用
    render_annotation,  # 直立矩形渲染
    # 新增：斜线直线模式
    lanes_slanted,
    detect_bands_along_y_slanted,
    detect_bands_along_y_prominence,
    render_annotation_slanted,
    match_ladder_best,
    fit_log_mw_irls,
    eval_fit_quality  # <-- 仍沿用
)
from roi_editor import ROIEditorCanvas

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("电泳图可视化处理（图形化裁剪版）")
        self.geometry("1200x800")

        # 数据
        self.orig_bgr: np.ndarray | None = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.gi = 1  # 当前胶索引（1-based）
        self.render_cache = {}
        self.LEFT_WIDTH = 300
        
        self.var_label_rows = tk.IntVar(value=1)       # 标签行数
        self.var_labels_on  = tk.BooleanVar(value=True) # 是否附加标签面板
        self.custom_labels: list[list[str]] = []        # 标签二维表

        # 左右UI
        self._build_left()
        self._build_right()

    # ----------------------- UI -----------------------
    def _build_left(self):
    # 外层容器：固定宽度，防止被内容撑大
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left_container.pack_propagate(False)           # 禁止跟随子控件自动扩展
        left_container.configure(width=self.LEFT_WIDTH)

        # Canvas + 垂直滚动条
        self.left_canvas = tk.Canvas(
            left_container, bg=self.cget("bg") if hasattr(self, "cget") else "#F0F0F0",
            highlightthickness=0, borderwidth=0, width=self.LEFT_WIDTH
        )
        vbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=vbar.set)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 内部 Frame：真正放控件的地方
        left = ttk.Frame(self.left_canvas)
        # 把内部 Frame 嵌进 Canvas
        self._left_window_item = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        # 关键：内部 Frame 大小变化时，更新 scrollregion，并让内部 Frame 宽度等于 Canvas 宽度
        def _on_frame_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            # 让内部 frame 宽度等于 canvas，避免出现水平滚动
            self.left_canvas.itemconfigure(self._left_window_item, width=self.LEFT_WIDTH)

        left.bind("<Configure>", _on_frame_configure)

        # 容器尺寸变化时，保持 canvas 固定宽度
        def _on_container_configure(event):
            self.left_canvas.configure(width=self.LEFT_WIDTH)

        left_container.bind("<Configure>", _on_container_configure)

        # 绑定鼠标滚轮（Win/Mac/Linux）
        def _on_mousewheel_win_mac(e):
            # e.delta：120 的倍数（Win）；Mac 可能是小值，统一归一
            step = -1 if e.delta > 0 else 1
            self.left_canvas.yview_scroll(step, "units")

        def _on_wheel_linux_up(e):   self.left_canvas.yview_scroll(-1, "units")
        def _on_wheel_linux_down(e): self.left_canvas.yview_scroll(1, "units")

        # 当鼠标进入/离开左栏时再绑定/解绑，这样不会影响右侧画布的缩放滚轮
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

        # ------- 以下把你现有的左栏控件都放到 left 这个 Frame 里 -------
        # （原来是 left = ttk.Frame(self); left.pack(...)）
        # 下面代码保持你原有的结构不变，只把 parent 从原来的 `left` 改为这里的 `left`

        # 文件打开
        f_file = ttk.LabelFrame(left, text="步骤1：打开图片")
        f_file.pack(fill=tk.X, pady=6)
        ttk.Button(f_file, text="打开图片...", command=self.open_image).pack(fill=tk.X, padx=6, pady=6)

        # 胶块检测
        f_det = ttk.LabelFrame(left, text="步骤2：自动检测胶块（原图）")
        f_det.pack(fill=tk.X, pady=6)
        self.var_expected = tk.IntVar(value=2)
        self.var_block = tk.IntVar(value=51)
        self.var_thrC = tk.IntVar(value=10)
        self.var_morph = tk.IntVar(value=11)
        self._spin(f_det, "期望胶块数量", self.var_expected, 1, 8)
        self._spin(f_det, "阈值块大小(奇数)", self.var_block, 15, 151, step=2)
        self._spin(f_det, "阈值偏移C", self.var_thrC, -20, 20)
        self._spin(f_det, "闭运算核大小(奇数)", self.var_morph, 3, 31, step=2)
        ttk.Button(f_det, text="运行胶块检测", command=self.run_detect).pack(fill=tk.X, padx=6, pady=6)

        # 图形化裁剪（ROI）
        f_roi = ttk.LabelFrame(left, text="步骤3：图形化裁剪（在右侧画布中拖动）")
        f_roi.pack(fill=tk.X, pady=6)
        nav = ttk.Frame(f_roi); nav.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(nav, text="⬅ 上一块", command=lambda: self.switch_gel(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,3))
        ttk.Button(nav, text="下一块 ➡", command=lambda: self.switch_gel(1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3,0))
        ttk.Button(f_roi, text="用自动检测结果重置当前 ROI", command=self.reset_roi_from_detect).pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(
            f_roi,
            text="提示：在右侧画布中左键拖动角点/边界即可精细裁剪；滚轮缩放，右键平移，方向键微调。",
            wraplength=self.LEFT_WIDTH-24,  # 关键：避免过宽
            justify="left"
        ).pack(anchor="w", padx=6, pady=(0,6))

        # 白平衡
        f_wb = ttk.LabelFrame(left, text="白平衡（ROI 级别，可选）")
        f_wb.pack(fill=tk.X, pady=6)
        self.var_wb_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_wb, text="启用白平衡", variable=self.var_wb_on).pack(anchor="w", padx=6)
        self.var_clip = tk.DoubleVar(value=2.0)
        self.var_tile = tk.IntVar(value=8)
        self._spin(f_wb, "CLAHE clipLimit", self.var_clip, 1.0, 5.0, step=0.1)
        self._spin(f_wb, "CLAHE tileSize", self.var_tile, 4, 16)

        # 分道参数
        f_lane = ttk.LabelFrame(left, text="分道参数")
        f_lane.pack(fill=tk.X, pady=6)
        self.var_nlanes = tk.IntVar(value=16)
        self._spin(f_lane, "泳道数量", self.var_nlanes, 1, 40)
        self.var_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(f_lane, text="自动（含斜率）", variable=self.var_mode, value="auto").pack(anchor="w", padx=6)
        ttk.Radiobutton(f_lane, text="等宽", variable=self.var_mode, value="uniform").pack(anchor="w", padx=6)
        self.var_smooth = tk.IntVar(value=31)
        self.var_sep = tk.DoubleVar(value=1.2)
        self._spin(f_lane, "投影平滑窗口(px,奇数)", self.var_smooth, 5, 101, step=2)
        self._spin(f_lane, "峰间最小间隔系数", self.var_sep, 1.0, 2.0, step=0.05)
        self.var_lpad = tk.IntVar(value=0)
        self.var_rpad = tk.IntVar(value=0)
        self._spin(f_lane, "等宽-左边距(px)", self.var_lpad, 0, 2000)
        self._spin(f_lane, "等宽-右边距(px)", self.var_rpad, 0, 2000)

        # 标准带/刻度
        f_marker = ttk.LabelFrame(left, text="标准带/刻度")
        f_marker.pack(fill=tk.X, pady=6)
        self.var_ladder_lane = tk.IntVar(value=1)
        self._spin(f_marker, "标准道序号", self.var_ladder_lane, 1, 40)
        ttk.Label(f_marker, text="标准分子量(kDa，大→小)：", wraplength=self.LEFT_WIDTH-24, justify="left").pack(anchor="w", padx=6)
        self.ent_marker = ttk.Entry(f_marker); self.ent_marker.insert(0, "180,130,100,70,55,40,35,25,15,10")
        self.ent_marker.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(f_marker, text="Y轴刻度(kDa，留空=同标准)：", wraplength=self.LEFT_WIDTH-24, justify="left").pack(anchor="w", padx=6)
        self.ent_ticks = ttk.Entry(f_marker); self.ent_ticks.insert(0, "100,70,55,40,35,25,15,10")
        self.ent_ticks.pack(fill=tk.X, padx=6, pady=2)
        self.var_top = tk.IntVar(value=0)
        self.var_bot = tk.IntVar(value=0)
        self._spin(f_marker, "仅在此行以下找带(top, px)", self.var_top, 0, 30000)
        self._spin(f_marker, "仅在此行以上(bottom, px; 0=底部)", self.var_bot, 0, 30000)
        self.var_axis = tk.StringVar(value="left")
        ttk.Radiobutton(f_marker, text="Y轴在左", variable=self.var_axis, value="left").pack(anchor="w", padx=6)
        ttk.Radiobutton(f_marker, text="Y轴在右", variable=self.var_axis, value="right").pack(anchor="w", padx=6)

        # 操作按钮
        f_action = ttk.Frame(left)
        f_action.pack(fill=tk.X, pady=10)
        ttk.Button(f_action, text="渲染当前胶块", command=self.render_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="导出当前结果", command=self.export_current).pack(fill=tk.X, padx=6, pady=4)
        # 自定义标签（按列排列）
        # 自定义标签（简化版）
        f_lab = ttk.LabelFrame(left, text="自定义标签")
        f_lab.pack(fill=tk.X, pady=6)
        
        # 仅保留一个按钮
        ttk.Button(f_lab, text="编辑标签...", command=self.open_labels_editor).pack(fill=tk.X, padx=6, pady=(6, 6))



    def _normalize_labels(self, labels: list[list[str]], nlanes: int) -> list[list[str]]:
        """将 labels 规整为 rows x nlanes 的二维表，缺省补空，超出裁剪。"""
        if not labels:
            return []
        out = []
        for row in labels:
            row = list(row) if row else []
            if len(row) < nlanes:
                row = row + [""] * (nlanes - len(row))
            else:
                row = row[:nlanes]
            out.append(row)
        return out

    def open_labels_editor(self):
        """
        简化标签编辑：仅允许用户输入一行文本（单行 Entry），使用逗号或 Tab 分隔。
        回车键被禁用（防止换行），输入内容仅支持逗号或 Tab。
        """
        cols = max(1, int(self.var_nlanes.get()))
        win = tk.Toplevel(self)
        win.title("编辑标签（仅一行，逗号或 Tab 分隔）")
        win.transient(self)
        win.grab_set()

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frm, text="请输入标签（仅一行，逗号或 Tab 分隔）", justify="left").pack(anchor="w")

        ent = ttk.Entry(frm, width=60)
        ent.pack(fill=tk.X, pady=6)

        # 禁用回车键，防止换行
        def on_return(event):
            return "break"  # 阻止默认行为

        ent.bind("<Return>", on_return)

        def do_ok():
            raw = ent.get().strip().replace("，", ",")
            # 按 Tab 拆分，如果只有一段且包含逗号，再按逗号拆
            parts = [p.strip() for p in raw.split("\t")]
            if len(parts) == 1 and "," in parts[0]:
                parts = [p.strip() for p in parts[0].split(",")]
            table = [parts]  # 单行 → 多列
            self.custom_labels = self._normalize_labels(table, cols)
            win.destroy()
            try:
                self.render_current()
            except Exception:
                pass

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btns, text="确定", command=do_ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 6))




    def _attach_labels_panel(
        self,
        img_bgr: np.ndarray,
        lanes: list[tuple[int, int]] | None,
        bounds: np.ndarray | None,
        labels_table: list[list[str]]
    ) -> np.ndarray:
        """
        在 img_bgr 上方追加一条“白底黑字”的标签面板。
        - 文本竖排（逐字符竖向绘制）
        - 每列（泳道）x 位置：相邻两条分隔线在图像高度中线处的中点
          * 斜线分道：x = (bounds[H//2, i] + bounds[H//2, i+1]) / 2
          * 等宽/手动：x = (l + r) / 2
        - 面板在图像“上方”（返回 np.vstack([panel, img_bgr])）
        """
        import numpy as np, cv2

        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        H, W = img_bgr.shape[:2]
        rows = len(labels_table)
        if rows == 0:
            return img_bgr
        cols = len(labels_table[0]) if rows else 0
        if cols == 0:
            return img_bgr

        # —— 1) 计算每个泳道的中心与宽度（用于字号自适配与水平居中）——
        centers: list[int] = []
        widths: list[int] = []
        if bounds is not None and isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] >= cols + 1:
            yc = int(min(bounds.shape[0] - 1, max(0, H // 2)))  # 用中线，避免端点偏移
            for i in range(cols):
                L = int(bounds[yc, i])
                R = int(bounds[yc, i + 1])
                centers.append(int(round((L + R) / 2)))
                widths.append(max(1, R - L))
        elif lanes is not None:
            for (l, r) in lanes[:cols]:
                centers.append(int(round((l + r) / 2)))
                widths.append(max(1, r - l))
        else:
            # 兜底：等距
            step = W / max(1, cols)
            for i in range(cols):
                centers.append(int(round((i + 0.5) * step)))
                widths.append(int(step))

        # —— 2) 预估每列可用宽度，并为竖排字符选择合适的字号（按列自适应）——
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.7
        thick = 1
        v_char_gap = 2  # 竖排字符之间的竖直间距（像素）
        h_margin = 3    # 列左右留白（像素）

        per_col_scale: list[float] = []
        per_col_char_w: list[int] = []
        per_col_char_h: list[int] = []

        for c in range(cols):
            max_w = max(10, widths[c] - 2 * h_margin)  # 文本列的最大允许宽度
            scale = base_scale
            # 用 'W' 近似最大字符宽（保守一些）
            (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            if tw_char > max_w:
                scale = max(0.4, scale * (max_w / (tw_char + 1e-6)))
                (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            per_col_scale.append(scale)
            per_col_char_w.append(max(1, int(tw_char)))
            per_col_char_h.append(max(1, int(th_char)))

        # —— 3) 计算面板总高度：逐行取“该行最长标签”的高度之最大值 —— 
        top_pad, bot_pad, row_gap = 8, 10, 6
        row_heights: list[int] = []
        rows_used_idx: list[int] = []  # 仅渲染有文本的行

        for r in range(rows):
            has_text = any((t or "").strip() for t in labels_table[r])
            if not has_text:
                row_heights.append(0)
                continue
            max_h = 0
            for c in range(cols):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                ch_h = per_col_char_h[c]
                # 竖排高度 = 字符个数 * (char_h + gap) - gap
                h_need = len(text) * (ch_h + v_char_gap) - v_char_gap
                max_h = max(max_h, h_need)
            # 最小给一行高度，避免边界挤压
            max_h = max(max_h, int(0.8 * max(per_col_char_h)))
            row_heights.append(max_h)
            rows_used_idx.append(r)

        if not rows_used_idx:
            # 全为空
            return img_bgr

        panel_h = top_pad + sum(row_heights[r] for r in rows_used_idx) + (len(rows_used_idx) - 1) * row_gap + bot_pad
        panel = np.full((panel_h, W, 3), 255, dtype=np.uint8)  # 白底

        # —— 4) 绘制（逐行区块，自上而下；列按泳道中心对齐，文本竖排）——
        y_cursor = top_pad
        for r in range(rows):
            if r not in rows_used_idx:
                continue
            rh = row_heights[r]
            # 在该行区块顶部开始绘制每个泳道的标签
            for c in range(cols):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                scale = per_col_scale[c]
                ch_w = per_col_char_w[c]
                ch_h = per_col_char_h[c]

                # 列的水平起点：以泳道中点为中心进行字宽居中
                x = int(np.clip(centers[c] - ch_w // 2, 2, W - ch_w - 2))
                # 逐字符向下排布；y 使用 baseline，因此先加一个字符高度
                y = y_cursor + ch_h
                for ch in text:
                    cv2.putText(panel, ch, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
                    y += ch_h + v_char_gap

            y_cursor += rh + row_gap

        # 边框
        cv2.rectangle(panel, (0, 0), (W - 1, panel_h - 1), (0, 0, 0), 1)

        # —— 5) 拼接到图像“上方” —— 
        return np.vstack([panel, img_bgr])

    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        grp = ttk.LabelFrame(right, text="原图 / 图形化裁剪（直接拖动边框/角点；滚轮缩放；右键平移）")
        grp.pack(fill=tk.BOTH, expand=True)
        self.roi_editor = ROIEditorCanvas(grp)
        self.roi_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.BOTH, expand=True)
        self.lbl_roi_wb = ttk.LabelFrame(bottom, text="ROI - WB 后裁剪图")
        self.lbl_roi_wb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.lbl_anno = ttk.LabelFrame(bottom, text="标注结果")
        self.lbl_anno.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.canvas_roi_wb = tk.Label(self.lbl_roi_wb, bg="#222")
        self.canvas_roi_wb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas_anno = tk.Label(self.lbl_anno, bg="#222")
        self.canvas_anno.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ----------------------- 逻辑 -----------------------
    def _spin(self, parent, text, var, from_, to, step=1):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(
            f, text=text, wraplength=getattr(self, "LEFT_WIDTH", 300) - 40, justify="left"
        ).pack(side=tk.LEFT)
        sp = ttk.Spinbox(f, textvariable=var, from_=from_, to=to, increment=step, width=8)
        sp.pack(side=tk.RIGHT)


    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择电泳图",
            filetypes=[("Image", "*.jpg;*.jpeg;*.png;*.tif;*.tiff"), ("All", "*.*")]
        )
        if not path:
            return
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("错误", "无法读取图片")
        else:
            self.orig_bgr = bgr
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
            messagebox.showwarning("提示", "未检测到胶块，请调整参数或确认图片。")
            self.roi_editor.set_roi(None)
            return
        # 默认选第一块
        self.gi = 1
        self.roi_editor.set_roi(self.boxes[0])

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
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        tw = widget.winfo_width() or 600
        th = widget.winfo_height() or 300
        scale = min(tw / W, th / H, 1.0)
        if scale < 1.0:
            rgb_disp = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb_disp = rgb
        from PIL import Image, ImageTk
        pil = Image.fromarray(rgb_disp)
        tkimg = ImageTk.PhotoImage(pil)
        widget.configure(image=tkimg)
        widget.image = tkimg

    def render_current(self):
        if self.orig_bgr is None:
            return
        roi = self.roi_editor.get_roi()
        if roi is None:
            messagebox.showwarning("提示", "请先在画布中框选或调整 ROI。")
            return

        x, y, w, h = roi
        gel = self.orig_bgr[y:y + h, x:x + w]

        # WB
        gel_bgr = auto_white_balance(
            gel, clip_limit=float(self.var_clip.get()), tile_size=int(self.var_tile.get())
        ) if self.var_wb_on.get() else gel

        gel_gray = cv2.cvtColor(gel_bgr, cv2.COLOR_BGR2GRAY)

        # 解析标签
        def parse_list(s: str):
            s = (s or "").replace("，", ",")
            out = []
            for t in s.split(","):
                t = t.strip()
                if not t:
                    continue
                try:
                    out.append(float(t))
                except:
                    pass
            return out

        ladder_labels_all = parse_list(self.ent_marker.get()) or [180, 130, 100, 70, 55, 40, 35, 25, 15, 10]
        tick_labels = parse_list(self.ent_ticks.get()) or ladder_labels_all

        # 分道
        nlanes = int(self.var_nlanes.get())
        mode = self.var_mode.get()
        if mode == "uniform":
            lanes = lanes_uniform(gel_gray, nlanes, int(self.var_lpad.get()), int(self.var_rpad.get()))
            bounds = None
        else:  # auto（含斜率）
            bounds = lanes_slanted(
                gel_gray, nlanes,
                smooth_px=int(self.var_smooth.get()),
                min_sep_ratio=float(self.var_sep.get()),
                search_half=12, max_step_px=5, smooth_y=9
            )
            lanes = None

        # 标准道
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
        top = int(self.var_top.get())
        bot = int(self.var_bot.get())
        y0 = top
        y1 = (gel_gray.shape[0] - bot) if bot > 0 else None

        # ---------- 1) 先“标记所有可见标记物（峰）” ----------
        if bounds is not None:
            peaks, prom = detect_bands_along_y_slanted(
                gel_gray, bounds, lane_index=ladder_lane - 1,
                y0=y0, y1=y1, min_distance=10, min_prominence=30.0
            )
        else:
            lx, rx = lanes[ladder_lane - 1]
            sub = gel_gray[:, lx:rx]
            peaks, prom = detect_bands_along_y_prominence(
                sub, y0=y0, y1=y1, min_distance=10, min_prominence=5.0
            )

        # 一律：用于绘制的峰 = 所有检测到的峰（取整）
        ladder_peaks_for_draw = [int(round(p)) for p in peaks]
        # 提前准备（保持接口兼容；实际渲染函数并不使用 ladder_labels_for_draw）
        ladder_labels_for_draw = sorted(ladder_labels_all, reverse=True)[:len(ladder_peaks_for_draw)]

        # ---------- 2) 再进行匹配与拟合（在所有峰中找最佳子序列） ----------
        sel_p_idx, sel_l_idx = match_ladder_best(peaks, ladder_labels_all, prom, min_pairs=3)

        fit_ok = False
        a, b = 1.0, 0.0  # 占位；仅当 fit_ok 时才用于刻度绘制
        r2, rmse = None, None

        if len(sel_p_idx) >= 2:
            y_used = [float(peaks[i]) for i in sel_p_idx]  # 亚像素 y
            lbl_used = [sorted(ladder_labels_all, reverse=True)[j] for j in sel_l_idx]
            w_used = [float(prom[i]) for i in sel_p_idx]

            # IRLS 拟合
            a_fit, b_fit = fit_log_mw_irls(y_used, lbl_used, w_used, iters=6)

            # 质量门控
            H_roi = gel_gray.shape[0]
            ok, r2, rmse = eval_fit_quality(
                y_used, lbl_used, a_fit, b_fit, H=H_roi,
                r2_min=0.97, rmse_frac_max=0.02, rmse_abs_min_px=20.0
            )
            if ok:
                a, b = a_fit, b_fit
                fit_ok = True
            else:
                # 给出提示，但不影响“所有峰”的绘制
                from tkinter import messagebox
                messagebox.showwarning(
                    "提示",
                    f"标准道分子量拟合偏差较大（R²={r2:.3f}, RMSE={rmse:.1f}px），已放弃拟合。\n"
                    "请检查：标准道序号、分子量列表、泳道数/边界与 ROI。"
                )
        else:
            # 可用于稳健拟合的点不足 → 仍然输出“所有峰”的图像，但不画kDa刻度
            from tkinter import messagebox
            messagebox.showinfo("提示", "可用于稳健拟合的标准带不足：已输出图像，但 Y 轴刻度仅在拟合成功时显示。")

        # ---------- 3) 渲染（fit_ok=False 时不画 kDa 刻度，仅画蓝线） ----------
        if bounds is not None:
            annotated = render_annotation_slanted(
                gel_bgr, bounds, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, tick_labels if fit_ok else [],
                yaxis_side=self.var_axis.get()
            )
        else:
            annotated = render_annotation(
                gel_bgr, lanes, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, tick_labels if fit_ok else [],
                yaxis_side=self.var_axis.get()
            )
        # —— 在标注图下方追加“白底黑字”标签面板（可选） ——
        
        annotated_final = annotated
        if self.var_labels_on.get():
            table = self._normalize_labels(self.custom_labels, nlanes)
            # 只要存在非空文本，就附加
            if table and any(any((t or "").strip() for t in row) for row in table):
                annotated_final = self._attach_labels_panel(annotated, lanes, bounds, table)


        self._show_np_on_label(self.canvas_roi_wb, gel_bgr)
        self._show_np_on_label(self.canvas_anno, annotated_final)


        # 缓存（导出用）
        self.render_cache = {
            
            "gi": int(self.gi),
            "roi": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "gel_bgr": gel_bgr,
            "annotated": annotated,
            "annotated_final": annotated_final,
            "lanes": [(int(l), int(r)) for (l, r) in lanes] if lanes is not None else None,
            "bounds": bounds.tolist() if bounds is not None else None,
            "ladder_lane": int(ladder_lane),
            "ladder_peaks_y": [int(v) for v in ladder_peaks_for_draw],  # 现在始终为“所有峰”
            "fit": {
                "a": float(a), "b": float(b), "valid": bool(fit_ok),
                "r2": float(r2) if r2 is not None else None,
                "rmse_px": float(rmse) if rmse is not None else None
            },
            "ladder_labels_used_desc": [float(v) for v in ladder_labels_for_draw],
            "tick_labels": [float(t) for t in tick_labels],
            "white_balance": {"enabled": bool(self.var_wb_on.get()),
                              "clipLimit": float(self.var_clip.get()),
                              "tileSize": int(self.var_tile.get())},
            "labels": table if self.var_labels_on.get() else []
        }
        # —— 自定义标签（按列排列）——


        if not fit_ok:
            messagebox.showinfo("提示", "本次未绘制 Y 轴分子量刻度（拟合未通过质控）。")

    def export_current(self):
        if not self.render_cache:
            messagebox.showwarning("提示", "请先渲染当前胶块。")
            return
        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return
        d = Path(out_dir)
        gi = self.render_cache["gi"]

        # 保存图片
        cv2.imencode(".png", self.render_cache["gel_bgr"])[1].tofile(str(d / f"gel{gi}_cropped.png"))
        # 改这里：导出带标签面板的图
        img_to_save = self.render_cache.get("annotated_final", self.render_cache.get("annotated"))
        cv2.imencode(".png", img_to_save)[1].tofile(str(d / f"gel{gi}_annotated.png"))


        # summary.json 合并
        js_path = d / "summary.json"
        try:
            summary = json.loads(js_path.read_text("utf-8")) if js_path.exists() else {"gels": []}
        except:
            summary = {"gels": []}
        summary["gels"] = [g for g in summary.get("gels", []) if g.get("index") != gi]
        summary["gels"].append({
            "index": gi,
            "box": [self.render_cache["roi"]["x"], self.render_cache["roi"]["y"],
                    self.render_cache["roi"]["w"], self.render_cache["roi"]["h"]],
            "lanes": self.render_cache["lanes"],
            "bounds": self.render_cache["bounds"],
            "ladder_lane": self.render_cache["ladder_lane"],
            "ladder_peaks_y": self.render_cache["ladder_peaks_y"],
            "fit": self.render_cache["fit"],
            "ladder_labels_used_desc": self.render_cache["ladder_labels_used_desc"],
            "tick_labels": self.render_cache["tick_labels"],
            "white_balance": self.render_cache["white_balance"],
            "labels": self.render_cache.get("labels", [])

        })
        js_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        messagebox.showinfo("完成", f"已导出：gel{gi}_cropped.png, gel{gi}_annotated.png, summary.json")

if __name__ == "__main__":
    App().mainloop()
