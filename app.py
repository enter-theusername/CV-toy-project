# app_desktop_gui_roi.py
# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from gel_core import (
    auto_white_balance, detect_gel_regions,
    lanes_by_projection, lanes_uniform,
    detect_bands_along_y,  # 旧的直线模式备用
    render_annotation,      # 直立矩形渲染
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("电泳图可视化处理（图形化裁剪版）")
        self.geometry("1000x720")

        # 数据
        self.orig_bgr: np.ndarray | None = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.gi = 1  # 当前胶索引（1-based）
        self.render_cache = {}
        self.LEFT_WIDTH = 300

        self.var_label_rows = tk.IntVar(value=1)     # 标签行数
        self.var_labels_on = tk.BooleanVar(value=True)  # 是否附加标签面板
        self.custom_labels: list[list[str]] = []  # 标签二维表

        # —— 自适应图片展示注册表（右侧两个预览） ——
        # widget -> {"img": np.ndarray, "last": (w,h)}
        self._autofit_store: dict[tk.Label, dict] = {}

        # 左右UI
        self._build_left()
        self._build_right()

    # ---------------- UI ----------------
    def _build_left(self):
        # 外层容器：固定宽度，防止被内容撑大
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left_container.pack_propagate(False)  # 禁止跟随子控件自动扩展
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
        # 把内部 Frame 嵌入 Canvas
        self._left_window_item = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        # 内部 Frame 大小变化时，更新 scrollregion，并让内部 Frame 宽度等于 Canvas 宽度
        def _on_frame_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            self.left_canvas.itemconfigure(self._left_window_item, width=self.LEFT_WIDTH)
        left.bind("<Configure>", _on_frame_configure)

        # 容器尺寸变化时，保持 canvas 固定宽度
        def _on_container_configure(event):
            self.left_canvas.configure(width=self.LEFT_WIDTH)
        left_container.bind("<Configure>", _on_container_configure)

        # 绑定鼠标滚轮（Win/Mac/Linux）
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

        # ---- 以下把你现有的左栏控件都放到 left 这个 Frame 里 ----

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

        # ⭐ 新增：重置旋转角度按钮（让基准线回到水平）
        ttk.Button(f_roi, text="重置旋转角度（水平）", command=self.reset_angle).pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(
            f_roi,
            text="提示：在右侧画布中左键拖动角点/边界即可精细裁剪；滚轮缩放，右键平移，方向键微调。\n现在 ROI 会随基准线角度旋转，便于在倾斜时准确圈选。",
            wraplength=self.LEFT_WIDTH-24,
            justify="left"
        ).pack(anchor="w", padx=6, pady=(0,6))

        # ===== 白平衡 / Autoscale（ROI 级别，可选） =====
        f_wb = ttk.LabelFrame(left, text="白平衡 / Autoscale（ROI 级别，可选）")
        f_wb.pack(fill=tk.X, pady=6)
        self.var_wb_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_wb, text="启用白平衡（线性百分位拉伸）", variable=self.var_wb_on).pack(anchor="w", padx=6)
        # 新增参数：exposure / percent_low / percent_high / per_channel / gamma
        self.var_wb_exposure = tk.DoubleVar(value=1.0)
        self.var_wb_p_low = tk.DoubleVar(value=0.5)
        self.var_wb_p_high = tk.DoubleVar(value=99.5)
        self.var_wb_per_channel = tk.BooleanVar(value=False)
        self.var_gamma_on = tk.BooleanVar(value=False)
        self.var_gamma_val = tk.DoubleVar(value=1.0)
        self._spin(f_wb, "曝光 exposure (0.5–2.0)", self.var_wb_exposure, 0.5, 2.0, step=0.05)
        self._spin(f_wb, "低百分位 percent_low (0–50)", self.var_wb_p_low, 0.0, 50.0, step=0.1)
        self._spin(f_wb, "高百分位 percent_high (50–100)", self.var_wb_p_high, 50.0, 100.0, step=0.1)
        ttk.Checkbutton(
            f_wb, text="逐通道 per_channel（最大对比度，可能改变色彩）",
            variable=self.var_wb_per_channel
        ).pack(anchor="w", padx=6, pady=(2,4))

        def _toggle_gamma_state():
            self.sp_gamma.configure(state=("normal" if self.var_gamma_on.get() else "disabled"))
        ttk.Checkbutton(
            f_wb, text="启用 gamma 校正（out = out ** (1/gamma)）",
            variable=self.var_gamma_on, command=_toggle_gamma_state
        ).pack(anchor="w", padx=6)
        frm_g = ttk.Frame(f_wb); frm_g.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(frm_g, text="gamma (>0)", wraplength=self.LEFT_WIDTH-40, justify="left").pack(side=tk.LEFT)
        self.sp_gamma = ttk.Spinbox(frm_g, textvariable=self.var_gamma_val,
                                    from_=0.2, to=3.0, increment=0.05, width=8,
                                    state="disabled")
        self.sp_gamma.pack(side=tk.RIGHT)

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

        # 自定义标签（简化版）
        f_lab = ttk.LabelFrame(left, text="自定义标签")
        f_lab.pack(fill=tk.X, pady=6)
        ttk.Button(f_lab, text="编辑标签...", command=self.open_labels_editor).pack(fill=tk.X, padx=6, pady=(6, 6))
    def reset_angle(self):
        """
        将右侧 ROI 编辑器中的“校准线”角度重置为 0°（水平），并立即重绘。
        """
        if getattr(self, "roi_editor", None) is None:
            return
        # 优先调用 ROIEditorCanvas 提供的公有方法（若无则直接设置属性）
        if hasattr(self.roi_editor, "reset_angle") and callable(self.roi_editor.reset_angle):
            self.roi_editor.reset_angle()
        else:
            try:
                self.roi_editor.cal_angle_deg = 0.0
                self.roi_editor.redraw()
            except Exception:
                pass


    def open_labels_editor(self):
        """简化标签编辑：支持从 Excel 中复制的纵向内容（多行），自动将换行、Tab、中文逗号统一处理为英文逗号。"""
        cols = max(1, int(self.var_nlanes.get()))
        win = tk.Toplevel(self)
        win.title("编辑标签（支持从 Excel 粘贴）")
        win.transient(self)
        win.grab_set()
        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(frm, text="请输入标签（支持从 Excel 粘贴，自动按逗号分隔）", justify="left").pack(anchor="w")
        txt = tk.Text(frm, height=4, width=60, wrap="none")
        txt.pack(fill=tk.X, pady=6)
        def on_return(event):
            return "break"
        txt.bind("<Return>", on_return)
        def do_ok():
            raw = txt.get("1.0", "end").strip()
            raw = raw.replace("，", ",").replace("\t", ",").replace("\n", ",")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
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

    # ---------------- 右侧（显示） ----------------
    def _build_right(self):
        # 整体右侧容器
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # —— 一级：上下可拖动 —— #
        self.pw_main = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_main.pack(fill=tk.BOTH, expand=True)

        # 顶部：ROI 编辑
        top_grp = ttk.LabelFrame(self.pw_main, text="原图 / 图形化裁剪（直接拖动边框/角点；滚轮缩放；右键平移）")
        self.roi_editor = ROIEditorCanvas(top_grp)
        self.roi_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 底部：左右可拖动（两个预览）
        bottom_container = ttk.Frame(self.pw_main)
        self.pw_bottom = tk.PanedWindow(bottom_container, orient=tk.HORIZONTAL, sashwidth=6, sashrelief="raised", opaqueresize=False)
        self.pw_bottom.pack(fill=tk.BOTH, expand=True)

        # 左预览
        self.lbl_roi_wb = ttk.LabelFrame(self.pw_bottom, text="ROI - WB 后裁剪图")
        self.canvas_roi_wb = tk.Label(self.lbl_roi_wb, bg="#222")
        self.canvas_roi_wb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 右预览
        self.lbl_anno = ttk.LabelFrame(self.pw_bottom, text="标注结果")
        self.canvas_anno = tk.Label(self.lbl_anno, bg="#222")
        self.canvas_anno.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 把左右预览加入下层分隔
        self.pw_bottom.add(self.lbl_roi_wb, minsize=120)
        self.pw_bottom.add(self.lbl_anno, minsize=120)

        # 把顶部/底部加入主分隔
        self.pw_main.add(top_grp, minsize=180)
        self.pw_main.add(bottom_container, minsize=160)

        # 设置初始分隔位置
        self.after(150, self._init_paned_positions)

        # 两个 Label 的自适应刷新
        self._bind_autofit(self.canvas_roi_wb)
        self._bind_autofit(self.canvas_anno)

    # ---------- 自适应显示：注册 / 刷新 ----------
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

    def _init_paned_positions(self):
        try:
            self.update_idletasks()
            total_h = max(1, self.pw_main.winfo_height())
            self.pw_main.sash_place(0, 0, int(total_h * 0.58))
            total_w = max(1, self.pw_bottom.winfo_width())
            self.pw_bottom.sash_place(0, int(total_w * 0.5), 0)
        except Exception:
            pass

    # ---------------- 逻辑 ----------------
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
            self.image_path = path
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
        self._set_autofit_image(widget, img_bgr)

    # --------- 旋转对齐裁剪（按旋转 ROI） ---------
    @staticmethod
    def _rotate_bound_with_M(image_bgr: np.ndarray, angle_deg: float):
        """旋转保持完整：扩张画布，返回 (rot_img, M)；M 为 2x3 仿射矩阵（已含平移）"""
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
        """把点 (x,y) 应用 2x3 仿射矩阵 M。"""
        nx = M[0,0]*x + M[0,1]*y + M[0,2]
        ny = M[1,0]*x + M[1,1]*y + M[1,2]
        return nx, ny

    def render_current(self):
        if self.orig_bgr is None:
            return

        # 1) 只从 ROI 编辑器拿“旋转矩形 + OpenCV 角度”（CCW为正）
        rroi = self.roi_editor.get_rotated_roi()
        if rroi is None:
            from tkinter import messagebox
            messagebox.showwarning("提示", "请先在画布中框选或调整 ROI。")
            return
        cx, cy, w, h, angle_ccw = rroi  # angle_ccw：逆时针为正

        # 2) 先对整幅原图按 -angle 摆正（使 ROI 局部坐标与最终图像轴对齐）
        rot_img, M = self._rotate_bound_with_M(self.orig_bgr, -angle_ccw)

        # 3) 用同一仿射 M 把 ROI 中心映射到旋转后的图
        def _affine_point(M_, x, y):
            return (M_[0,0]*x + M_[0,1]*y + M_[0,2],
                    M_[1,0]*x + M_[1,1]*y + M_[1,2])

        cx2, cy2 = _affine_point(M, cx, cy)

        # 4) 在摆正后的图上按 (w,h) 做轴对齐裁剪（以中心为准）
        x0 = int(round(cx2 - w / 2.0))
        y0 = int(round(cy2 - h / 2.0))
        x1 = x0 + int(round(w))
        y1 = y0 + int(round(h))

        H2, W2 = rot_img.shape[:2]
        x0 = max(0, min(x0, W2 - 1))
        y0 = max(0, min(y0, H2 - 1))
        x1 = max(x0 + 1, min(x1, W2))
        y1 = max(y0 + 1, min(y1, H2))

        gel = rot_img[y0:y1, x0:x1].copy()

        # 5) WB（放在摆正/裁剪之后）
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

        gel_gray = cv2.cvtColor(gel_bgr, cv2.COLOR_BGR2GRAY)

        # 6) 解析标签
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

        # 7) 分道
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
                # 新增：默认不做顶/底“等距混合”，避免把起始 x 锁死一致
                enable_uniform_blend=False
            )
            lanes = None

        # 8) 标准道/找峰
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
        top = int(self.var_top.get())
        bot = int(self.var_bot.get())
        y0_roi = top
        y1_roi = (gel_gray.shape[0] - bot) if bot > 0 else None

        if bounds is not None:
            peaks, prom = detect_bands_along_y_slanted(
                gel_gray, bounds, lane_index=ladder_lane - 1,
                y0=y0_roi, y1=y1_roi, min_distance=10, min_prominence=50.0
            )
        else:
            lx, rx = lanes[ladder_lane - 1]
            sub = gel_gray[:, lx:rx]
            peaks, prom = detect_bands_along_y_prominence(
                sub, y0=y0_roi, y1=y1_roi, min_distance=10, min_prominence=50.0
            )

        ladder_peaks_for_draw = [int(round(p)) for p in peaks]
        ladder_labels_for_draw = sorted(ladder_labels_all, reverse=True)[:len(ladder_peaks_for_draw)]

        # 9) 拟合
        sel_p_idx, sel_l_idx = match_ladder_best(peaks, ladder_labels_all, prom, min_pairs=3)
        fit_ok = False
        a, b = 1.0, 0.0
        r2, rmse = None, None
        if len(sel_p_idx) >= 2:
            y_used = [float(peaks[i]) for i in sel_p_idx]
            lbl_used = [sorted(ladder_labels_all, reverse=True)[j] for j in sel_l_idx]
            w_used = [float(prom[i]) for i in sel_p_idx]
            a_fit, b_fit = fit_log_mw_irls(y_used, lbl_used, w_used, iters=6)
            H_roi = gel_gray.shape[0]
            ok, r2, rmse = eval_fit_quality(
                y_used, lbl_used, a_fit, b_fit, H=H_roi,
                r2_min=0.97, rmse_frac_max=0.02, rmse_abs_min_px=20.0
            )
            if ok:
                a, b = a_fit, b_fit
                fit_ok = True
            else:
                from tkinter import messagebox
                messagebox.showwarning(
                    "提示",
                    f"标准道拟合偏差较大（R²={r2:.3f}, RMSE={rmse:.1f}px），已放弃拟合。\n"
                    "请检查标准道序号、分子量表、泳道与 ROI。"
                )
        else:
            from tkinter import messagebox
            messagebox.showinfo("提示", "用于稳健拟合的标准带不足：已输出图像，但 Y 轴刻度仅在拟合成功时显示。")

        # 10) 标注渲染
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

        # 11) 附加标签面板（可选）
        annotated_final = annotated
        if self.var_labels_on.get():
            table = self._normalize_labels(self.custom_labels, nlanes)
            if table and any(any((t or "").strip() for t in row) for row in table):
                annotated_final = self._attach_labels_panel(annotated, lanes, bounds, table)

        # 12) 显示（自适应）
        self._set_autofit_image(self.canvas_roi_wb, gel_bgr)
        self._set_autofit_image(self.canvas_anno, annotated_final)

        # 13) 缓存导出
        self.render_cache = {
            "gi": int(self.gi),
            "gel_bgr": gel_bgr,
            "annotated": annotated,
            "annotated_final": annotated_final
        }

        if not fit_ok:
            from tkinter import messagebox
            messagebox.showinfo("提示", "本次未绘制 Y 轴分子量刻度（拟合未通过质控）。")



    def _attach_labels_panel(
        self,
        img_bgr: np.ndarray,
        lanes: list[tuple[int, int]] | None,
        bounds: np.ndarray | None,
        labels_table: list[list[str]]
    ) -> np.ndarray:
        """
        在 img_bgr 上方追加“白底黑字”的标签面板（与原实现相同，略）。
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
        # 1) 泳道中心与宽度
        centers: list[int] = []
        widths: list[int] = []
        if bounds is not None and isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] >= cols + 1:
            yc = int(min(bounds.shape[0] - 1, max(0, H // 2)))
            for i in range(cols):
                L = int(bounds[yc, i]); R = int(bounds[yc, i + 1])
                centers.append(int(round((L + R) / 2)))
                widths.append(max(1, R - L))
        elif lanes is not None:
            for (l, r) in lanes[:cols]:
                centers.append(int(round((l + r) / 2)))
                widths.append(max(1, r - l))
        else:
            step = W / max(1, cols)
            for i in range(cols):
                centers.append(int(round((i + 0.5) * step)))
                widths.append(int(step))
        # 2) 列宽→字号
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.7
        thick = 1
        v_char_gap = 2
        h_margin = 3
        per_col_scale: list[float] = []
        per_col_char_w: list[int] = []
        per_col_char_h: list[int] = []
        for c in range(cols):
            max_w = max(10, widths[c] - 2 * h_margin)
            scale = base_scale
            (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            if tw_char > max_w:
                scale = max(0.4, scale * (max_w / (tw_char + 1e-6)))
                (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            per_col_scale.append(scale)
            per_col_char_w.append(max(1, int(tw_char)))
            per_col_char_h.append(max(1, int(th_char)))
        # 3) 面板高度
        top_pad, bot_pad, row_gap = 8, 10, 6
        row_heights: list[int] = []
        rows_used_idx: list[int] = []
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
                h_need = len(text) * (ch_h + v_char_gap) - v_char_gap
                max_h = max(max_h, h_need)
            max_h = max(max_h, int(0.8 * max(per_col_char_h)))
            row_heights.append(max_h)
            rows_used_idx.append(r)
        if not rows_used_idx:
            return img_bgr
        panel_h = top_pad + sum(row_heights[r] for r in rows_used_idx) + (len(rows_used_idx) - 1) * row_gap + bot_pad
        panel = np.full((panel_h, W, 3), 255, dtype=np.uint8)
        # 4) 绘制
        y_cursor = top_pad
        for r in range(rows):
            if r not in rows_used_idx:
                continue
            rh = row_heights[r]
            for c in range(cols):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                scale = per_col_scale[c]
                ch_w = per_col_char_w[c]
                ch_h = per_col_char_h[c]
                x = int(np.clip(centers[c] - ch_w // 2, 2, W - ch_w - 2))
                y = y_cursor + ch_h
                for ch in text:
                    cv2.putText(panel, ch, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
                    y += ch_h + v_char_gap
            y_cursor += rh + row_gap
        cv2.rectangle(panel, (0, 0), (W - 1, panel_h - 1), (0, 0, 0), 1)
        return np.vstack([panel, img_bgr])

    def export_current(self):
        if not self.render_cache:
            messagebox.showwarning("提示", "请先渲染当前胶块。")
            return
        img_to_save = self.render_cache.get("annotated_final", self.render_cache.get("annotated"))
        if img_to_save is None:
            messagebox.showwarning("提示", "未找到可导出的标注图像。")
            return
        gi = self.render_cache.get("gi", 1)
        default_name = f"gel{gi}_annotated.png"
        try:
            if getattr(self, "image_path", None):
                p = Path(self.image_path)
                default_name = f"{p.stem}_annotated.png"
        except Exception:
            pass
        fpath = filedialog.asksaveasfilename(
            title="保存带标注的图片",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png")]
        )
        if not fpath:
            return
        ok, buf = cv2.imencode(".png", img_to_save)
        if not ok:
            messagebox.showerror("错误", "PNG 编码失败，未能导出。")
            return
        buf.tofile(fpath)
        messagebox.showinfo("完成", f"已导出：{fpath}")


if __name__ == "__main__":
    App().mainloop()
