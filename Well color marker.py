import os
import math
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import colorsys
import json
from typing import List, Tuple, Optional, Dict


def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def ensure_int_tuple(xy):
    return int(round(xy[0])), int(round(xy[1]))


class PlateColorAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plate Color Analyzer (Tk)")
        self.geometry("1000x600")
        self.minsize(800, 400)

        # ---- State ----
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

        # Color format
        self.fmt_rgb255_var = tk.BooleanVar(value=False)
        self.fmt_rgb01_var = tk.BooleanVar(value=True)
        self.fmt_hex_var = tk.BooleanVar(value=False)
        self.fmt_hsv_var = tk.BooleanVar(value=False)
        self.fmt_hsl_var = tk.BooleanVar(value=False)
        self.fmt_gray_var = tk.BooleanVar(value=True)
        self.font_size_var = tk.IntVar(value=14)

        self.use_relative_rect_var = tk.BooleanVar(value=True)
        self.merge_csv_var = tk.BooleanVar(value=False)
        self.export_dir = tk.StringVar(value=os.getcwd())
        self.canvas_bg = "#1e1e1e"

        # Canvas interaction: zoom & pan
        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)
        self._pan_active = False
        self._pan_start = (0, 0)

        # Preview points (current image)
        self.preview_points_cache: List[Tuple[float, float, str]] = []

        # ---- Per-image rectangle (relative 0~1, based on transformed image size) & draggable points ----
        # Custom rectangle: path -> ((rx1, ry1), (rx2, ry2))
        self.custom_rects: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        # Default rectangle (from the 1st image)
        self.default_rect_rel: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

        # Per-image per-well custom points (relative 0~1): { path: { label: (rx, ry), ... }, ... }
        self.custom_points: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # ⭐ Global unified positions (template from the 1st image)
        self.use_global_points_var = tk.BooleanVar(value=False)
        # Global template: { label: (rx, ry) } in relative coords
        self.global_points_template: Dict[str, Tuple[float, float]] = {}

        # Drag state
        self.dragging_label: Optional[str] = None  # e.g., "A1"
        self._drag_pick_radius_canvas = 10        # canvas px radius for picking
        self._drag_active = False

        # ---- Status bar ----
        self.status_var = tk.StringVar(value="Ready: please load images...")

        # Build UI
        self._build_ui()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)       # drag ROI
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)  # end drag
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", lambda e: self.on_zoom(e, wheel_delta=1))   # Linux
        self.canvas.bind("<Button-5>", lambda e: self.on_zoom(e, wheel_delta=-1))  # Linux
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)

    # -------------------- UI --------------------
    def _build_ui(self):
        # Left: scrollable control panel
        left_holder = ttk.Frame(self)
        left_holder.pack(side=tk.LEFT, fill=tk.Y)

        # Canvas + Scrollbar for scrollable panel
        self.ctrl_canvas = tk.Canvas(left_holder, borderwidth=0, highlightthickness=0)
        self.ctrl_vsb = ttk.Scrollbar(left_holder, orient="vertical", command=self.ctrl_canvas.yview)
        self.ctrl_canvas.configure(yscrollcommand=self.ctrl_vsb.set)
        self.ctrl_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ctrl_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        # Frame inside the scrollable canvas
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas, padding=10)
        self._ctrl_window = self.ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")

        # Sync scrollregion and width
        def _on_ctrl_configure(event=None):
            self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))
            try:
                width = left_holder.winfo_width() - self.ctrl_vsb.winfo_width()
                if width > 50:
                    self.ctrl_canvas.itemconfig(self._ctrl_window, width=width)
            except Exception:
                pass

        self.ctrl_frame.bind("<Configure>", _on_ctrl_configure)
        left_holder.bind("<Configure>", _on_ctrl_configure)

        # Scroll wheel on panel (Windows/Mac)
        self.ctrl_frame.bind_all("<MouseWheel>", self._on_ctrl_mousewheel)
        # Scroll wheel on panel (Linux)
        self.ctrl_frame.bind_all("<Button-4>", lambda e: self.ctrl_canvas.yview_scroll(-1, "units"))
        self.ctrl_frame.bind_all("<Button-5>", lambda e: self.ctrl_canvas.yview_scroll(+1, "units"))

        # Right: canvas + status bar
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_label = ttk.Label(right, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Image canvas
        self.canvas = tk.Canvas(right, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ===== Controls in self.ctrl_frame =====
        # Images & Export
        frm_files = ttk.LabelFrame(self.ctrl_frame, text="Images & Export", padding=10)
        frm_files.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frm_files, text="Load Images...", command=self.load_images).pack(fill=tk.X)
        ttk.Button(frm_files, text="Load Folder...", command=self.load_folder).pack(fill=tk.X)
        nav = ttk.Frame(frm_files); nav.pack(fill=tk.X, pady=5)
        ttk.Button(nav, text="Prev", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(nav, text="Next", command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        ttk.Button(frm_files, text="Fit to Window", command=self.fit_canvas_to_image).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(frm_files, text="Choose Export Folder...", command=self.choose_export_dir).pack(fill=tk.X)
        ttk.Label(frm_files, textvariable=self.export_dir, foreground="#666").pack(fill=tk.X, pady=(4, 0))

        # Rotate / Flip
        frm_tf = ttk.LabelFrame(self.ctrl_frame, text="Rotate / Flip", padding=10)
        frm_tf.pack(fill=tk.X, pady=(0, 10))
        row_tf = ttk.Frame(frm_tf); row_tf.pack(fill=tk.X)
        ttk.Label(row_tf, text="Angle (°):").pack(side=tk.LEFT)
        ent_angle = ttk.Entry(row_tf, textvariable=self.angle_var, width=8)
        ent_angle.pack(side=tk.LEFT, padx=5)
        ttk.Button(row_tf, text="Apply", command=self.apply_transform_and_redraw).pack(side=tk.LEFT)
        flips = ttk.Frame(frm_tf); flips.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(flips, text="Flip Horizontally", variable=self.flip_h_var, command=self.apply_transform_and_redraw).pack(side=tk.LEFT)
        ttk.Checkbutton(flips, text="Flip Vertically", variable=self.flip_v_var, command=self.apply_transform_and_redraw).pack(side=tk.LEFT)

        # Rectangle & Plate Grid
        frm_grid = ttk.LabelFrame(self.ctrl_frame, text="ROI Rectangle & Plate Grid", padding=10)
        frm_grid.pack(fill=tk.X, pady=(0, 10))
        rr = ttk.Frame(frm_grid); rr.pack(fill=tk.X)
        ttk.Label(rr, text="Rows:").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.rows_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(rr, text="Cols:").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.cols_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        btns_rect = ttk.Frame(frm_grid); btns_rect.pack(fill=tk.X, pady=5)
        ttk.Button(btns_rect, text="Set Rectangle (click twice on canvas)", command=self.start_set_rect).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns_rect, text="Clear Rectangle", command=self.clear_rect).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(frm_grid, text="Preview Grid", command=self.redraw).pack(fill=tk.X)

        # Sampling ROI
        frm_roi = ttk.LabelFrame(self.ctrl_frame, text="Sampling ROI", padding=10)
        frm_roi.pack(fill=tk.X, pady=(0, 10))
        shapes = ttk.Frame(frm_roi); shapes.pack(fill=tk.X)
        ttk.Radiobutton(shapes, text="Circle", variable=self.roi_shape_var, value="circle", command=self.redraw).pack(side=tk.LEFT)
        ttk.Radiobutton(shapes, text="Square", variable=self.roi_shape_var, value="square", command=self.redraw).pack(side=tk.LEFT)
        srow = ttk.Frame(frm_roi); srow.pack(fill=tk.X, pady=5)
        ttk.Label(srow, text="ROI size (px):").pack(side=tk.LEFT)
        ttk.Entry(srow, textvariable=self.roi_size_var, width=8).pack(side=tk.LEFT, padx=5)
        fs = ttk.Frame(frm_roi); fs.pack(fill=tk.X)
        ttk.Label(fs, text="Label font size:").pack(side=tk.LEFT)
        ttk.Entry(fs, textvariable=self.font_size_var, width=8).pack(side=tk.LEFT, padx=5)

        # Export Color Formats
        frm_fmt = ttk.LabelFrame(self.ctrl_frame, text="Export Color Formats", padding=10)
        frm_fmt.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frm_fmt, text="RGB (0-255)", variable=self.fmt_rgb255_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="RGB (0-1)", variable=self.fmt_rgb01_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HEX (#RRGGBB)", variable=self.fmt_hex_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HSV", variable=self.fmt_hsv_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="HSL", variable=self.fmt_hsl_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_fmt, text="Grayscale", variable=self.fmt_gray_var).pack(anchor=tk.W)

        # Batch
        frm_batch = ttk.LabelFrame(self.ctrl_frame, text="Batch", padding=10)
        frm_batch.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frm_batch, text="Use relative rectangle for batch", variable=self.use_relative_rect_var).pack(anchor=tk.W)
        ttk.Checkbutton(frm_batch, text="Merge all to one CSV", variable=self.merge_csv_var).pack(anchor=tk.W)
        ttk.Checkbutton(
            frm_batch,
            text="Unify well positions (use 1st image as template)",
            variable=self.use_global_points_var,
            command=self._on_toggle_use_global_points
        ).pack(anchor=tk.W)

        bb = ttk.Frame(frm_batch); bb.pack(fill=tk.X, pady=5)
        ttk.Button(bb, text="Export current", command=self.export_current).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(bb, text="Batch export all", command=self.batch_export).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Configuration
        frm_cfg = ttk.LabelFrame(self.ctrl_frame, text="Configuration", padding=10)
        frm_cfg.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frm_cfg, text="Save config...", command=self.save_config).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(frm_cfg, text="Load config...", command=self.load_config).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    # -------------- Relative/absolute conversions --------------
    def _rect_to_rel(self, p1: Tuple[float, float], p2: Tuple[float, float], w: int, h: int):
        return ((p1[0] / w, p1[1] / h), (p2[0] / w, p2[1] / h))

    def _rel_to_abs(self, rel_rect: Tuple[Tuple[float, float], Tuple[float, float]], w: int, h: int):
        (rx1, ry1), (rx2, ry2) = rel_rect
        return (rx1 * w, ry1 * h), (rx2 * w, ry2 * h)

    def _load_rect_for_current_image(self):
        """Restore self.rect_p1/p2 from custom/default rectangle (absolute px on transformed image)."""
        if self.transformed_image is None or not self.image_paths or self.current_index < 0:
            return
        path = self.image_paths[self.current_index]
        w, h = self.transformed_image.size
        if path in self.custom_rects:
            p1, p2 = self._rel_to_abs(self.custom_rects[path], w, h)
        elif self.default_rect_rel is not None:
            p1, p2 = self._rel_to_abs(self.default_rect_rel, w, h)
        else:
            p1 = p2 = None
        self.rect_p1 = p1
        self.rect_p2 = p2

    # -------------------- File & Image --------------------
    def _clear_loaded_image_state(self):
        """Clear current image and related state."""
        try:
            if self.original_image is not None:
                self.original_image.close()
        except Exception:
            pass
        try:
            if (self.transformed_image is not None) and (self.transformed_image is not self.original_image):
                self.transformed_image.close()
        except Exception:
            pass

        self.original_image = None
        self.transformed_image = None
        self.display_image = None

        self.preview_points_cache = []
        self.rect_p1 = None
        self.rect_p2 = None

        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)
        self._pan_active = False
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        if hasattr(self, "status_var"):
            try:
                self.status_var.set("Cache cleared. Ready to load images...")
            except Exception:
                pass

    def load_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.current_index = 0
        self.load_current_image()

    def load_folder(self):
        """Pick a folder and read supported images (no recursion)."""
        d = filedialog.askdirectory(title="Select image folder")
        if not d:
            return
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        try:
            all_names = os.listdir(d)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot access the folder:\n{d}\n{e}")
            return
        files = []
        for name in all_names:
            full = os.path.join(d, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                files.append(full)
        files.sort()
        if not files:
            messagebox.showwarning("Tip", "No supported image files found in this folder.")
            return

        self._clear_loaded_image_state()
        self.image_paths = files
        self.current_index = 0
        try:
            self.export_dir.set(d)
        except Exception:
            pass
        self.load_current_image()

    def choose_export_dir(self):
        d = filedialog.askdirectory(title="Choose export folder")
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
            messagebox.showerror("Error", f"Cannot open image:\n{path}\n{e}")
            return

        # Rectangle restored in apply_transform_and_redraw() via _load_rect_for_current_image()
        self.rect_p1 = None
        self.rect_p2 = None
        self.apply_transform_and_redraw()
        if hasattr(self, "status_var"):
            try:
                self.status_var.set(f"Loaded: {os.path.basename(path)} [{self.current_index+1}/{len(self.image_paths)}]")
            except Exception:
                pass
    def apply_transform_and_redraw(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        # Flips
        if self.flip_h_var.get():
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v_var.get():
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Rotation
        angle = self.angle_var.get() % 360
        if abs(angle) > 1e-6:
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        self.transformed_image = img

        # Restore rectangle for current image
        self._load_rect_for_current_image()

        # Fit & redraw
        self.fit_canvas_to_image()
        self.redraw()

    # -------------------- Canvas display & interactions --------------------
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
        # Clear current image's custom rect & custom points
        path = self.image_paths[self.current_index] if (self.image_paths and self.current_index >= 0) else None
        if path and path in self.custom_rects:
            del self.custom_rects[path]
        if path and path in self.custom_points:
            del self.custom_points[path]
        # If first image, also clear default rect
        if self.current_index == 0:
            self.default_rect_rel = None
        # (Do not auto-clear global_points_template)

        self.rect_p1 = None
        self.rect_p2 = None
        self.setting_rect_mode = False
        # Try restoring from default for current image
        self._load_rect_for_current_image()
        self.redraw()

    # ---------- Global unified wells: toggle ----------
    def _on_toggle_use_global_points(self):
        """If turned on and template empty, try copy once from the 1st image custom points."""
        if self.use_global_points_var.get():
            if not self.global_points_template:
                first_path = self.image_paths[0] if self.image_paths else None
                if first_path and first_path in self.custom_points:
                    self.global_points_template = dict(self.custom_points[first_path])
        self.redraw()

    def _get_current_path_size(self):
        if self.transformed_image is None or not self.image_paths or self.current_index < 0:
            return None, 0, 0
        return self.image_paths[self.current_index], *self.transformed_image.size

    def _compute_points_with_overrides(self, p1, p2, rows, cols, path: Optional[str], iw: int, ih: int) -> List[Tuple[float, float, str]]:
        """
        Base on grid from rectangle, then override by:
          1) per-image custom_points[path][label] (highest priority)
          2) global template (if enabled), otherwise keep grid positions
        """
        pts: List[Tuple[float, float, str]] = []
        if p1 is None or p2 is None:
            return pts

        x0 = min(p1[0], p2[0])
        y0 = min(p1[1], p2[1])
        x1 = max(p1[0], p2[0])
        y1 = max(p1[1], p2[1])
        dx = 0 if cols == 1 else (x1 - x0) / (cols - 1)
        dy = 0 if rows == 1 else (y1 - y0) / (rows - 1)

        for r in range(rows):
            row_letter = chr(ord('A') + r)
            for c in range(cols):
                x = x0 + c * dx
                y = y0 + r * dy
                label = f"{row_letter}{c+1}"
                pts.append((x, y, label))

        overrides = self.custom_points.get(path, {}) if path else {}
        use_global = self.use_global_points_var.get() and bool(self.global_points_template)

        for i, (x, y, label) in enumerate(pts):
            if label in overrides:
                rx, ry = overrides[label]
                pts[i] = (rx * iw, ry * ih, label)
            elif use_global and label in self.global_points_template:
                grx, gry = self.global_points_template[label]
                pts[i] = (grx * iw, gry * ih, label)
        return pts

    def _compute_grid_points(self) -> List[Tuple[float, float, str]]:
        """Return all well centers as (x, y, label) list (row-major) with overrides applied."""
        if self.rect_p1 is None or self.rect_p2 is None:
            return []
        rows = max(1, self.rows_var.get())
        cols = max(1, self.cols_var.get())
        path, iw, ih = self._get_current_path_size()
        if path is None:
            return []
        return self._compute_points_with_overrides(self.rect_p1, self.rect_p2, rows, cols, path, iw, ih)

    def on_canvas_click(self, event):
        # If setting rectangle: collect two clicks
        if self.setting_rect_mode and self.transformed_image is not None:
            x, y = self.canvas_to_image(event.x, event.y)
            if self.rect_p1 is None:
                self.rect_p1 = (x, y)
            else:
                self.rect_p2 = (x, y)
                self.setting_rect_mode = False
                # Save custom rectangle (relative)
                if self.image_paths and self.current_index >= 0 and self.transformed_image is not None:
                    w, h = self.transformed_image.size
                    rel = self._rect_to_rel(self.rect_p1, self.rect_p2, w, h)
                    path = self.image_paths[self.current_index]
                    self.custom_rects[path] = rel
                    if self.current_index == 0:
                        self.default_rect_rel = rel
                self.redraw()
            return

        # Else: try to start dragging a well
        if self.transformed_image is None:
            return
        pts = self._compute_grid_points()
        if not pts:
            return
        sx, sy = event.x, event.y
        min_d2 = float('inf')
        pick_label = None
        for (x, y, label) in pts:
            px, py = self.image_to_canvas(x, y)
            d2 = (px - sx) ** 2 + (py - sy) ** 2
            if d2 < min_d2:
                min_d2 = d2
                pick_label = label
        roi_px_canvas = (max(1, self.roi_size_var.get()) / 2.0) * self.canvas_scale
        pick_radius = max(self._drag_pick_radius_canvas, roi_px_canvas * 0.8)
        if min_d2 <= pick_radius ** 2 and pick_label is not None:
            self.dragging_label = pick_label
            self._drag_active = True
            self.status_var.set(f"Dragging: {pick_label} (hold LMB to move, release to finish)")

    def on_drag_move(self, event):
        """While dragging: update the selected well center (stored in relative coords)."""
        if not self._drag_active or self.dragging_label is None or self.transformed_image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        iw, ih = self.transformed_image.size
        x = min(max(0.0, x), iw - 1e-6)
        y = min(max(0.0, y), ih - 1e-6)

        path = self.image_paths[self.current_index]
        rel = (x / iw, y / ih)
        if path not in self.custom_points:
            self.custom_points[path] = {}
        self.custom_points[path][self.dragging_label] = rel

        # If on first image and unified is ON, update template simultaneously
        first_path = self.image_paths[0] if self.image_paths else None
        if self.use_global_points_var.get() and path == first_path:
            self.global_points_template[self.dragging_label] = rel

        self.redraw()
        self.status_var.set(f"Dragging: {self.dragging_label} -> ({int(x)}, {int(y)})")

    def on_drag_end(self, event):
        if self._drag_active:
            self._drag_active = False
            self.status_var.set("Drag ended.")
        self.dragging_label = None

    def on_canvas_motion(self, event):
        if self.transformed_image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        iw, ih = self.transformed_image.size
        if 0 <= x < iw and 0 <= y < ih:
            self.status_var.set(f"Coord: ({int(x)}, {int(y)})")
        if self.setting_rect_mode and self.rect_p1 is not None:
            # Temporary preview for rectangle
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

    def redraw(self, temp_p2=None):
        self._draw_image_on_canvas()
        if self.transformed_image is None:
            return
        p1 = self.rect_p1
        p2 = self.rect_p2 if temp_p2 is None else temp_p2
        if p1 is not None and p2 is not None:
            x0, y0 = p1
            x1, y1 = p2
            sx0, sy0 = self.image_to_canvas(x0, y0)
            sx1, sy1 = self.image_to_canvas(x1, y1)
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline="#00FF99", width=2)

            if self.rect_p1 is not None and self.rect_p2 is not None and temp_p2 is None:
                pts = self._compute_grid_points()
                self.preview_points_cache = pts
                roi_size = max(1, self.roi_size_var.get())
                rpx_canvas = (roi_size / 2.0) * self.canvas_scale
                font_color = "#00FFC2"
                for (x, y, label) in pts:
                    sx, sy = self.image_to_canvas(x, y)
                    if self.roi_shape_var.get() == "circle":
                        self.canvas.create_oval(sx - rpx_canvas, sy - rpx_canvas,
                                                sx + rpx_canvas, sy + rpx_canvas,
                                                outline="#FFD200")
                    else:
                        self.canvas.create_rectangle(sx - rpx_canvas, sy - rpx_canvas,
                                                     sx + rpx_canvas, sy + rpx_canvas,
                                                     outline="#FFD200")
                    # center dot
                    self.canvas.create_oval(sx - 2, sy - 2, sx + 2, sy + 2, fill="#FF4081", outline="")
                    # label
                    self.canvas.create_text(
                        sx, sy - rpx_canvas - 6,
                        text=label, fill=font_color,
                        font=("Arial", max(8, int(self.font_size_var.get() * self.canvas_scale * 0.75))),
                        anchor=tk.S
                    )

    def on_zoom(self, event, wheel_delta: Optional[int] = None):
        """
        Mouse wheel zoom centered at the cursor.
        - Windows/Mac: event.delta (+/-)
        - Linux: wheel_delta argument (+1/-1)
        """
        if self.transformed_image is None:
            return
        delta = 0
        if wheel_delta is not None:
            delta = wheel_delta
        elif hasattr(event, "delta"):
            delta = 1 if event.delta > 0 else -1
        if delta == 0:
            return

        old_scale = self.canvas_scale
        step = 1.1
        new_scale = old_scale * (step if delta > 0 else 1.0 / step)
        new_scale = max(0.05, min(20.0, new_scale))

        ix, iy = self.canvas_to_image(event.x, event.y)
        self.canvas_scale = new_scale
        self.canvas_offset = (event.x - ix * new_scale, event.y - iy * new_scale)
        self.redraw()

    def on_pan_start(self, event):
        """Start panning with middle/right mouse button."""
        self._pan_active = True
        self._pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        """Pan while dragging middle/right mouse button."""
        if not self._pan_active:
            return
        sx0, sy0 = self._pan_start
        dx, dy = event.x - sx0, event.y - sy0
        ox, oy = self.canvas_offset
        self.canvas_offset = (ox + dx, oy + dy)
        self._pan_start = (event.x, event.y)
        self.redraw()

    def _on_ctrl_mousewheel(self, event):
        """Scroll the left control panel; do not pass to the canvas."""
        if event.widget is self.canvas:
            return
        if hasattr(event, "delta") and event.delta != 0:
            self.ctrl_canvas.yview_scroll(-1 if event.delta > 0 else +1, "units")

    # -------------------- Sampling & Export --------------------
    def _sample_color_at(self, img: Image.Image, x: float, y: float, roi_shape: str, roi_size: int) -> Optional[Tuple[float, float, float]]:
        """Average RGB(0-255) within the ROI centered at (x, y)."""
        w, h = img.size
        half = roi_size / 2.0
        left = int(math.floor(x - half))
        upper = int(math.floor(y - half))
        right = int(math.ceil(x + half))
        lower = int(math.ceil(y + half))
        left = max(0, left); upper = max(0, upper)
        right = min(w, right); lower = min(h, lower)
        if right <= left or lower <= upper:
            return None
        crop = img.crop((left, upper, right, lower))
        arr = np.asarray(crop, dtype=np.float32)
        if roi_shape == "circle":
            hh, ww = arr.shape[:2]
            yy, xx = np.ogrid[0:hh, 0:ww]
            cx = (x - left)
            cy = (y - upper)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (half) ** 2
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
        r01, g01, b01 = r255 / 255.0, g255 / 255.0, b255 / 255.0
        if self.fmt_rgb255_var.get():
            cols["R"] = int(round(r255)); cols["G"] = int(round(g255)); cols["B"] = int(round(b255))
        if self.fmt_rgb01_var.get():
            cols["R01"] = round(r01, 4); cols["G01"] = round(g01, 4); cols["B01"] = round(b01, 4)
        if self.fmt_hex_var.get():
            cols["HEX"] = rgb_to_hex(r255, g255, b255)
        if self.fmt_hsv_var.get():
            h, s, v = colorsys.rgb_to_hsv(r01, g01, b01)
            cols["H"] = round(h * 360.0, 2); cols["S"] = round(s, 4); cols["V"] = round(v, 4)
        if self.fmt_hsl_var.get():
            h, l, s = colorsys.rgb_to_hls(r01, g01, b01)  # HLS order
            cols["Hsl_H"] = round(h * 360.0, 2); cols["Hsl_S"] = round(s, 4); cols["Hsl_L"] = round(l, 4)
        if self.fmt_gray_var.get():
            gray = 0.2126 * r255 + 0.7152 * g255 + 0.0722 * b255
            cols["Gray"] = int(round(gray))
        return cols

    def _annotate_image(self, img: Image.Image, points: List[Tuple[float, float, str]], roi_shape: str, roi_size: int, font_size: int) -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        rpx = roi_size / 2.0
        for (x, y, label) in points:
            bbox = (x - rpx, y - rpx, x + rpx, y + rpx)
            if roi_shape == "circle":
                draw.ellipse(bbox, outline=(255, 210, 0), width=2)
            else:
                draw.rectangle(bbox, outline=(255, 210, 0), width=2)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 64, 129))
            tx, ty = x, y - rpx - 6
            draw.text((tx, ty), label, fill=(0, 255, 194), anchor="ms", stroke_width=1, stroke_fill=(0, 0, 0))
        return out

    def export_current(self):
        if self.transformed_image is None or not self.image_paths:
            messagebox.showwarning("Tip", "Please load images first.")
            return
        if self.rect_p1 is None or self.rect_p2 is None:
            messagebox.showwarning("Tip", "Please set the rectangle corners first.")
            return
        try:
            self._export_single(self.current_index)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_single(self, index: int, merged_writer: Optional[csv.DictWriter] = None, merged_file: Optional[any] = None):
        path = self.image_paths[index]
        img0 = Image.open(path).convert("RGB")
        img = self._apply_current_transform_to_image(img0)

        # Rectangle (relative preferred)
        if self.use_relative_rect_var.get():
            rel = None
            if path in self.custom_rects:
                rel = self.custom_rects[path]
            elif self.default_rect_rel is not None:
                rel = self.default_rect_rel
            else:
                raise ValueError("No rectangle found. Set it on the first image or per image.")
            iw, ih = img.size
            p1, p2 = self._rel_to_abs(rel, iw, ih)
        else:
            p1 = self.rect_p1
            p2 = self.rect_p2
            if p1 is None or p2 is None:
                raise ValueError("Rectangle not set. Set it or switch to relative rectangle for batch.")
            iw, ih = img.size
            cw, ch = self.transformed_image.size
            if (int(iw) != int(cw)) or (int(ih) != int(ch)):
                raise ValueError("Absolute rectangle mode requires same size across images. Use relative mode instead.")

        # Points with overrides
        rows = max(1, self.rows_var.get())
        cols = max(1, self.cols_var.get())
        iw, ih = img.size
        pts = self._compute_points_with_overrides(p1, p2, rows, cols, path, iw, ih)

        roi_shape = self.roi_shape_var.get()
        roi_size = max(1, self.roi_size_var.get())
        records = []
        points_for_annotate = []
        for (x, y, label) in pts:
            rgb = self._sample_color_at(img, x, y, roi_shape, roi_size)
            if rgb is None:
                continue
            points_for_annotate.append((x, y, label))
            row_letter = label[0]
            col_index = int(label[1:])
            row_dict = {
                "filename": os.path.basename(path),
                "row": row_letter,
                "col": col_index,
                "label": label,
                "x": int(round(x)),
                "y": int(round(y)),
            }
            row_dict.update(self._build_color_columns(rgb))
            records.append(row_dict)

        annotated = self._annotate_image(img, points_for_annotate, roi_shape, roi_size, self.font_size_var.get())
        base = os.path.splitext(os.path.basename(path))[0]
        out_img_path = os.path.join(self.export_dir.get(), f"{base}_annotated.png")
        annotated.save(out_img_path)

        if merged_writer is not None:
            for rec in records:
                merged_writer.writerow(rec)
        else:
            if records:
                out_csv_path = os.path.join(self.export_dir.get(), f"{base}_data.csv")
                fieldnames = ["filename", "row", "col", "label", "x", "y"]
                extra_cols = [k for k in records[0].keys() if k not in fieldnames]
                fieldnames += extra_cols
                with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for rec in records:
                        writer.writerow(rec)
        return len(records), len(points_for_annotate), out_img_path

    def _apply_current_transform_to_image(self, img: Image.Image) -> Image.Image:
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
            messagebox.showwarning("Tip", "Please load at least one image.")
            return
        if not self.use_relative_rect_var.get():
            messagebox.showwarning("Tip", "It is recommended to enable 'Use relative rectangle for batch'.")
        if (self.default_rect_rel is None) and (not any(p in self.custom_rects for p in self.image_paths)):
            messagebox.showwarning("Tip", "No default rectangle (1st image) or per-image rectangle found.")
            return

        merged_path = os.path.join(self.export_dir.get(), "batch_data.csv") if self.merge_csv_var.get() else None
        merged_file = None
        merged_writer = None
        total_points = 0
        total_records = 0
        try:
            if merged_path:
                merged_file = open(merged_path, "w", newline="", encoding="utf-8-sig")
                merged_writer = None
            for idx, _ in enumerate(self.image_paths):
                path = self.image_paths[idx]
                img0 = Image.open(path).convert("RGB")
                img = self._apply_current_transform_to_image(img0)

                # Rectangle
                if self.use_relative_rect_var.get():
                    if path in self.custom_rects:
                        rel = self.custom_rects[path]
                    elif self.default_rect_rel is not None:
                        rel = self.default_rect_rel
                    else:
                        raise ValueError(f"No rectangle found for image {os.path.basename(path)}.")
                    iw, ih = img.size
                    p1, p2 = self._rel_to_abs(rel, iw, ih)
                else:
                    p1 = self.rect_p1
                    p2 = self.rect_p2
                    if p1 is None or p2 is None:
                        raise ValueError("Rectangle not set. Set it or switch to relative rectangle for batch.")
                    iw, ih = img.size
                    cw, ch = self.transformed_image.size
                    if (int(iw) != int(cw)) or (int(ih) != int(ch)):
                        raise ValueError("Absolute rectangle mode requires same size across images. Use relative mode instead.")

                # Points
                rows = max(1, self.rows_var.get())
                cols = max(1, self.cols_var.get())
                iw, ih = img.size
                pts = self._compute_points_with_overrides(p1, p2, rows, cols, path, iw, ih)

                roi_shape = self.roi_shape_var.get()
                roi_size = max(1, self.roi_size_var.get())
                local_records = []
                points = []
                for (x, y, label) in pts:
                    rgb = self._sample_color_at(img, x, y, roi_shape, roi_size)
                    if rgb is None:
                        continue
                    points.append((x, y, label))
                    row_letter = label[0]
                    col_index = int(label[1:])
                    rec = {
                        "filename": os.path.basename(path),
                        "row": row_letter,
                        "col": col_index,
                        "label": label,
                        "x": int(round(x)),
                        "y": int(round(y)),
                    }
                    rec.update(self._build_color_columns(rgb))
                    local_records.append(rec)

                # Annotated image
                base = os.path.splitext(os.path.basename(path))[0]
                out_img_path = os.path.join(self.export_dir.get(), f"{base}_annotated.png")
                annotated = self._annotate_image(img, points, roi_shape, roi_size, self.font_size_var.get())
                annotated.save(out_img_path)

                # CSV
                if self.merge_csv_var.get():
                    if local_records:
                        if merged_writer is None:
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

            self.status_var.set(f"Batch export finished: {len(self.image_paths)} images, points {total_points}, records {total_records}.")
            messagebox.showinfo("Done", f"Batch export finished.\nTotal records: {total_records}\nExport folder:\n{self.export_dir.get()}")
        except Exception as e:
            messagebox.showerror("Batch export failed", str(e))
        finally:
            if merged_file:
                merged_file.close()

    # -------------------- Configuration save/load --------------------
    def save_config(self):
        if self.transformed_image is None:
            messagebox.showwarning("Tip", "Please load an image first (to record rectangle in relative coords).")
            return
        path = filedialog.asksaveasfilename(
            title="Save config as JSON",
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
            # Unified wells toggle & template
            "use_global_points": self.use_global_points_var.get(),
        }

        if self.rect_p1 is not None and self.rect_p2 is not None:
            w, h = self.transformed_image.size
            cfg["rect_rel"] = [
                [self.rect_p1[0] / w, self.rect_p1[1] / h],
                [self.rect_p2[0] / w, self.rect_p2[1] / h]
            ]

        if self.default_rect_rel is not None:
            cfg["default_rect_rel"] = [
                [self.default_rect_rel[0][0], self.default_rect_rel[0][1]],
                [self.default_rect_rel[1][0], self.default_rect_rel[1][1]],
            ]
        if self.custom_rects:
            custom_rects_rel = {}
            for pth, ((rx1, ry1), (rx2, ry2)) in self.custom_rects.items():
                custom_rects_rel[pth] = [[rx1, ry1], [rx2, ry2]]
            cfg["custom_rects_rel"] = custom_rects_rel

        if self.custom_points:
            custom_points_rel = {}
            for pth, m in self.custom_points.items():
                custom_points_rel[pth] = {label: [rx, ry] for label, (rx, ry) in m.items()}
            cfg["custom_points_rel"] = custom_points_rel

        if self.global_points_template:
            cfg["global_points_rel"] = {label: [rx, ry] for label, (rx, ry) in self.global_points_template.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Config saved: {os.path.basename(path)}")

    def load_config(self):
        path = filedialog.askopenfilename(
            title="Load config JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
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

            # Compatibility: rect_rel only affects current display
            if self.transformed_image is not None and "rect_rel" in cfg:
                w, h = self.transformed_image.size
                rp1, rp2 = cfg["rect_rel"]
                self.rect_p1 = (rp1[0] * w, rp1[1] * h)
                self.rect_p2 = (rp2[0] * w, rp2[1] * h)

            # Rectangles & points
            if "default_rect_rel" in cfg:
                a, b = cfg["default_rect_rel"]
                self.default_rect_rel = ((a[0], a[1]), (b[0], b[1]))
            else:
                self.default_rect_rel = None
            self.custom_rects.clear()
            for pth, v in cfg.get("custom_rects_rel", {}).items():
                a, b = v
                self.custom_rects[pth] = ((a[0], a[1]), (b[0], b[1]))

            self.custom_points.clear()
            for pth, m in cfg.get("custom_points_rel", {}).items():
                self.custom_points[pth] = {label: (float(rx), float(ry)) for label, (rx, ry) in m.items()}

            # Unified wells
            self.use_global_points_var.set(cfg.get("use_global_points", False))
            self.global_points_template = {
                label: (float(rx), float(ry)) for label, (rx, ry) in cfg.get("global_points_rel", {}).items()
            }

            self.apply_transform_and_redraw()
            self.status_var.set(f"Config loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

# -------------------- Main --------------------
if __name__ == "__main__":
    app = PlateColorAnalyzerApp()
    app.mainloop()
