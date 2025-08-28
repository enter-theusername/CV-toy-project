# app_desktop_gui_roi.py
# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
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
        panel_top_h: int = 0
    ):
        """设置底图与几何信息，并按 lane_marks 生成可拖动红箭头。"""
        # 清空画布元素，并重置图像 item 与缓存，防止使用已删除的 item id
        self.delete("all")
        self.img_item = None           # ★★ 关键修复：防止后续用已删除的 item id
        self.base_img_tk = None        # 推荐一并重置，避免旧引用
        self.arrows.clear()

        # 基础数据
        self.base_img_bgr = base_img_bgr
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
        self.panel_w = max(0, Wt - Wg)
        self.x_offset = self.panel_w if self.yaxis_side == "left" else 0

        # 渲染底图
        self._render_base_image()

        # 生成箭头（拟合失败则不生成）
        if self.fit_ok and lane_marks:
            self._create_arrows_from_marks(lane_marks)

    def render_to_image(self) -> np.ndarray | None:
        """把当前箭头叠加到底图像素，返回BGR图（用于导出）。"""
        if self.base_img_bgr is None:
            return None
        img = self.base_img_bgr.copy()
        H, W = img.shape[:2]

        # 三角箭头（像素坐标系，未缩放的固定尺寸）
        w, h = 20, 5
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
        w_px, h_px = 20, 5
        tip_img = (x_img - 20, y_img)
        p1_img = (tip_img[0] - w_px // 2, y_img + h_px)
        p2_img = (tip_img[0] - w_px // 2, y_img - h_px)
        pts_img = [p1_img, p2_img, tip_img]
        pts_canvas = []
        for (x, y) in pts_img:
            cx, cy = self._to_canvas(x, y)
            pts_canvas.extend([cx, cy])
        return pts_canvas

    def _add_arrow(self, x_img: float, y_img: float, lane_idx: int, mw: float, color: tuple[int, int, int] = (255, 64, 64)):
        """在Canvas上新增一个箭头并绑定拖动事件。"""
        # Tk 颜色（BGR -> #RRGGBB）
        r, g, b = (255, 64, 64)
        pts = self._arrow_canvas_points(x_img, y_img)
        aid = self.create_polygon(pts, fill=f"#{r:02x}{g:02x}{b:02x}", outline="#ffffff", width=1, smooth=False)
        meta = {"id": aid, "lane_idx": lane_idx, "x_img": float(x_img),"y_img": float(y_img), "mw": float(mw)}
        self.arrows.append(meta)

        # 绑定仅用于该箭头的拖动事件
        self.tag_bind(aid, "<ButtonPress-1>", lambda e, a=meta: self._on_drag_start(e, a))
        self.tag_bind(aid, "<B1-Motion>",     lambda e, a=meta: self._on_drag_move(e, a))
        self.tag_bind(aid, "<ButtonRelease-1>", lambda e, a=meta: self._on_drag_end(e, a))

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
        self.title("电泳图可视化处理（图形化裁剪版）")
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
            "name": "常用10条(10–180 kDa)",
            "values": [180,130,95,65,52,41,31,25,17,10],
            "note": "与应用内默认相同，顺序大→小"
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
                self.lbl_preset_info.configure(text="（未选择标准集合）")

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
        win.title("管理标准集合")
        win.transient(self); win.grab_set()

        frm = ttk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左：列表
        left = ttk.Frame(frm); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="集合列表").pack(anchor="w")
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
        ttk.Label(right, text="名称").grid(row=0, column=0, sticky="w")
        ent_name = ttk.Entry(right); ent_name.grid(row=0, column=1, sticky="ew", padx=(6,0))
        ttk.Label(right, text="分子量（支持 , ; 空格/中文逗号，顺序大→小）").grid(row=1, column=0, columnspan=2, sticky="w", pady=(8,0))
        txt_vals = tk.Text(right, height=5); txt_vals.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4,0))
        ttk.Label(right, text="备注").grid(row=3, column=0, sticky="w", pady=(8,0))
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
                messagebox.showwarning("提示", "请填写名称"); return
            if not vals:
                messagebox.showwarning("提示", "请至少填写一个有效分子量数值"); return
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
            if not messagebox.askyesno("确认", f"确定要删除集合「{it['name']}」吗？"):
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
                    self.lbl_preset_info.configure(text="（未选择标准集合）")

        ttk.Button(btns, text="新建", command=do_new).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="保存", command=do_save).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="删除", command=do_delete).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

        _reload_list()

    # -------------------- UI：左侧 -------------------- #
    def _build_left(self):
        # 外层容器：固定宽度，防止被内容撑大
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left_container.pack_propagate(False)
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
        # ⭐ 新增：重置旋转角度按键（让基准线回到水平）
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
        self.var_show_green = tk.BooleanVar(value=True)
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
        self.ent_marker = ttk.Entry(f_marker); self.ent_marker.insert(0, "180,130,95,65,52,41,31,25,17,10")
        self.ent_marker.pack(fill=tk.X, padx=6, pady=2)
        row = ttk.Frame(f_marker); row.pack(fill=tk.X, padx=6, pady=(6,2))
        ttk.Label(row, text="选择集合").pack(side=tk.LEFT)
        self.var_preset = getattr(self, "var_preset", tk.StringVar(value=""))
        self.cb_preset = ttk.Combobox(row, textvariable=self.var_preset, state="readonly", width=22)
        self.cb_preset.pack(side=tk.LEFT, padx=(6,6))
        self.cb_preset.bind("<<ComboboxSelected>>", self.on_select_preset)
        ttk.Button(row, text="管理集合…", command=self.open_preset_manager).pack(side=tk.RIGHT)
        # 信息标签（多行）
        #self.lbl_preset_info = ttk.Label(f_marker, text="（未选择标准集合）", justify="left")
        #self.lbl_preset_info.pack(fill=tk.X, padx=6, pady=(4,2))

        self.var_axis = tk.StringVar(value="left")
        ttk.Checkbutton(f_marker,text="显示绿线（分隔线）",variable=self.var_show_green,command=self.on_toggle_show_green ).pack(anchor="w", padx=6, pady=2) # <--- 轻量切换


        # 操作按钮
        f_action = ttk.Frame(left)
        f_action.pack(fill=tk.X, pady=10)
        ttk.Button(f_action, text="渲染当前胶块", command=self.render_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(f_action, text="导出当前结果", command=self.export_current).pack(fill=tk.X, padx=6, pady=4)

        # 自定义标签（改为“列名 + 分子量标注”）
        f_lab = ttk.LabelFrame(left, text="自定义标签（每列：列名 + 本列分子量）")
        f_lab.pack(fill=tk.X, pady=6)
        ttk.Button(f_lab, text="编辑列名与分子量...", command=self.open_labels_editor).pack(fill=tk.X, padx=6, pady=(6, 6))

    # -------------------- UI：右侧（显示） -------------------- #

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
        self.canvas_anno = RightAnnoCanvas(self.lbl_anno)
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
        show_green: bool | None = None
    ) -> np.ndarray:
        if img_core is None or img_core.size == 0:
            return img_core
        img = img_core.copy()
        H, W_total = img.shape[:2]
        Hg, Wg = gel_bgr.shape[:2]
        panel_w = (overlay.get("panel_w") if overlay else max(0, W_total - Wg))
        side = (overlay.get("yaxis_side") if overlay else str(yaxis_side or "left").lower())
        x_offset = 0 if side == "right" else panel_w

        # === 1) 绿线：分隔线折线 ===
        boundaries = (overlay.get("boundaries") if overlay else None) if overlay else None
        if not boundaries:
            # 回退：根据 bounds 或 lanes 重新构造
            boundaries = []
            if isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[0] == Hg:
                for i in range(1, bounds.shape[1]-1):
                    xcol = (bounds[:, i].astype(np.int32) + int(x_offset))
                    xcol = np.clip(xcol, 0, W_total-1)
                    pts = np.stack([xcol, np.arange(Hg, dtype=np.int32)], axis=1)
                    boundaries.append(pts)
            elif lanes:
                xs = [lanes[i][0] for i in range(1, len(lanes))]
                for x in xs:
                    xg = int(np.clip(x_offset + int(x), 0, W_total-1))
                    pts = np.stack([np.full((Hg,), xg, dtype=np.int32),
                                    np.arange(Hg, dtype=np.int32)], axis=1)
                    boundaries.append(pts)
        # 绘制
        do_green = bool(self.var_show_green.get()) if show_green is None else bool(show_green)
        if do_green:
            for pts in boundaries or []:
                cv2.polylines(
                    img, [pts.astype(np.int32)],
                    isClosed=False, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA
                )


        # === 2) 刻度短横线/文字：App 层重绘，支持左右轴 ===
        #if fit_ok and tick_labels:
        #    lbs = sorted([float(x) for x in tick_labels if np.isfinite(x) and x > 0], reverse=True)
        #    font = cv2.FONT_HERSHEY_SIMPLEX
        #    fscale, thick = 0.5, 1
        #    tick_len, gap = 12, 3
        #    margin = 2
        #    for mw in lbs:
        #        y = int(round(a * np.log10(float(mw)) + b))
        #        y_draw = int(np.clip(y, 12, H-5))
        #        if side == "right":
        #            # 右轴：短横线靠右白板左缘；文字在其右侧
        #            x1 = int(Wg + margin)
        #            x2 = int(min(W_total-1, x1 + tick_len - 1))
        #            cv2.line(img, (x1, y_draw), (x2, y_draw), (0, 0, 0), 1, cv2.LINE_AA)
        #            text = f"{mw:g}"
        #            (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
        #            x_text = int(min(W_total - 2 - tw, x2 + gap))
        #            cv2.putText(img, text, (x_text, y_draw + th//2 - 1), font, fscale, (0,0,0), thick, cv2.LINE_AA)
        #        else:
        #            # 左轴：短横线靠白板右缘；文字在其左侧
        #            x2 = int(panel_w - 1 - margin)
        #            x1 = int(max(2, x2 - tick_len + 1))
        #            cv2.line(img, (x1, y_draw), (x2, y_draw), (0, 0, 0), 1, cv2.LINE_AA)
        #            text = f"{mw:g}"
        #            (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
        #            x_text = int(max(2, x1 - gap - tw))
        #            cv2.putText(img, text, (x_text, y_draw + th//2 - 1), font, fscale, (0,0,0), thick, cv2.LINE_AA)

        return img


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

    def render_current(self):
        if self.orig_bgr is None:
            return

        # 1) 读取旋转ROI并校正、裁切（保持原逻辑）
        rroi = self.roi_editor.get_rotated_roi()
        if rroi is None:
            from tkinter import messagebox
            messagebox.showwarning("提示", "请先在画布中框选或调整 ROI。")
            return
        cx, cy, w, h, angle_ccw = rroi
        rot_img, M = self._rotate_bound_with_M(self.orig_bgr, -angle_ccw)
        def _affine_point(M_, x, y):
            return (M_[0,0]*x + M_[0,1]*y + M_[0,2], M_[1,0]*x + M_[1,1]*y + M_[1,2])
        cx2, cy2 = _affine_point(M, cx, cy)
        x0 = int(round(cx2 - w/2.0)); y0 = int(round(cy2 - h/2.0))
        x1 = x0 + int(round(w));      y1 = y0 + int(round(h))
        H2, W2 = rot_img.shape[:2]
        x0 = max(0, min(x0, W2 - 1)); y0 = max(0, min(y0, H2 - 1))
        x1 = max(x0 + 1, min(x1, W2)); y1 = max(y0 + 1, min(y1, H2))
        gel = rot_img[y0:y1, x0:x1].copy()

        # 2) WB（保持原逻辑）
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

        # 3) 解析标记（保持原逻辑）
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
        tick_labels = ladder_labels_all

        # 4) 分道（保持原逻辑）
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

        # 5) 标准带检测与拟合（保持原逻辑）
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
                                            r2_min=0.97, rmse_frac_max=0.02, rmse_abs_min_px=20.0)
            if ok:
                a, b = a_fit, b_fit
                fit_ok = True
            else:
                from tkinter import messagebox
                messagebox.showwarning(
                    "提示",
                    f"标准道拟合偏差较大（R²={r2:.3f}, RMSE={rmse:.1f}px），已放弃拟合。\n请检查标准道序号、分子量表、泳道与 ROI。"
                )
        else:
            from tkinter import messagebox
            messagebox.showinfo("提示", "用于稳健拟合的标准带不足：已输出图像，但 Y 轴分子量刻度仅在拟合成功时显示。")

        # 6) 核心标注底图（不包含绿线）
        if bounds is not None:
            res = render_annotation_slanted(
                gel_bgr, bounds, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, tick_labels if fit_ok else [],
                yaxis_side=self.var_axis.get()
            )
        else:
            res = render_annotation(
                gel_bgr, lanes, ladder_peaks_for_draw, ladder_labels_for_draw,
                a, b, tick_labels if fit_ok else [],
                yaxis_side=self.var_axis.get()
            )
        annotated_core = res if not (isinstance(res, tuple) and len(res) == 2) else res[0]
        overlay = None

        # 7) “不带绿线”的核心图
        annotated_core_no_green = self._draw_overlays_on_core(
            img_core=annotated_core, gel_bgr=gel_bgr,
            overlay=overlay, bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            tick_labels=tick_labels if fit_ok else [],
            yaxis_side=self.var_axis.get(),
            show_green=False
        )

        # 8) 上方“列名”白板（如开启），得到最终“无绿线”底图
        annotated_final_base = annotated_core_no_green
        panel_top_h = 0
        if self.var_labels_on.get():
            nlanes = int(self.var_nlanes.get())
            ladder_lane = max(1, min(int(self.var_ladder_lane.get()), nlanes))
            n_nonladder = max(0, nlanes - 1)
            names_seq = (self.lane_names or [])
            names_use = (names_seq + [""] * n_nonladder)[:n_nonladder]
            labels_table = [names_use]  # 仅一行：列名
            if labels_table and any((t or "").strip() for t in labels_table[0]):
                H0 = annotated_core_no_green.shape[0]
                annotated_final_base = self._attach_labels_panel(
                    img_bgr=annotated_core_no_green,
                    lanes=lanes, bounds=bounds,
                    labels_table=labels_table,
                    gel_bgr=gel_bgr,
                    ladder_lane=ladder_lane,
                    yaxis_side=self.var_axis.get()
                )
                panel_top_h = annotated_final_base.shape[0] - H0

        # 9) 在“无绿线最终底图”基础上，再生成一份“带绿线”的版本
        annotated_final_with_green = self._draw_overlays_on_core(
            img_core=annotated_final_base, gel_bgr=gel_bgr,
            overlay=overlay, bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            tick_labels=tick_labels if fit_ok else [],
            yaxis_side=self.var_axis.get(),
            show_green=True
        )

        # 10) 左侧预览（WB 后 ROI 裁剪图）
        self._set_autofit_image(self.canvas_roi_wb, gel_bgr)

        # 11) 右侧交互画布初始化：用“无绿线”底图；若勾选，则再瞬时切到“有绿线”
        lane_marks_input = self.lane_marks or []
        ladder_lane = max(1, min(int(self.var_ladder_lane.get()), int(self.var_nlanes.get())))
        self.canvas_anno.set_scene(
            base_img_bgr=annotated_final_base,  # 先用无绿线
            gel_bgr=gel_bgr,
            bounds=bounds, lanes=lanes,
            a=a, b=b, fit_ok=fit_ok,
            nlanes=int(self.var_nlanes.get()),
            ladder_lane=ladder_lane,
            yaxis_side=self.var_axis.get(),
            lane_marks=lane_marks_input,
            panel_top_h=panel_top_h
        )
        if bool(self.var_show_green.get()):
            # 切换为“带绿线”
            self.canvas_anno.update_base_image(annotated_final_with_green)

        # 12) 缓存：两版底图 + 其他信息
        self.render_cache = {
            "gi": int(self.gi),
            "gel_bgr": gel_bgr,
            "annotated_base": annotated_final_base,            # 兼容旧键：无绿线
            "annotated_base_no_green": annotated_final_base,   # 新：无绿线
            "annotated_base_with_green": annotated_final_with_green,  # 新：有绿线
            "fit_ok": fit_ok
        }
        if not fit_ok:
            from tkinter import messagebox
            messagebox.showinfo("提示", "本次未绘制 Y 轴分子量刻度（拟合未通过质控），右侧不生成可拖动箭头。")

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
            tip = (int(xc)-20, int(np.clip(y, 0, H - 1)))
            w, h = 20, 5
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
        在 img_bgr 上方追加“白底黑字”的标签面板（列名），仅为非标准道绘制。
        labels_table 的列数 = 非标准道数量（或更少），按左到右顺延到真实的“非标准道”列。
        空文本列保持空白；多出的列名将被省略。
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

        # 真实泳道数量
        if bounds is not None and isinstance(bounds, np.ndarray):
            real_nlanes = max(0, bounds.shape[1] - 1)
        elif lanes is not None:
            real_nlanes = len(lanes)
        else:
            real_nlanes = cols_use + 1  # 至少比使用的列多 1（含被排除的标准道）

        # 计算每个“真实泳道”的几何中心（取中间高度处）
        centers_mid_all: list[int] = []
        widths_all: list[int] = []
        if bounds is not None and isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] >= real_nlanes + 1:
            yc = int(min(bounds.shape[0] - 1, max(0, H // 2)))
            for i in range(real_nlanes):
                L = int(bounds[yc, i]); R = int(bounds[yc, i + 1])
                centers_mid_all.append(int(round((L + R) / 2)))
                widths_all.append(max(1, R - L))
        elif lanes is not None:
            for (l, r) in lanes[:real_nlanes]:
                centers_mid_all.append(int(round((l + r) / 2)))
                widths_all.append(max(1, r - l))
        else:
            step = max(1.0, Wg / max(1, real_nlanes))
            for i in range(real_nlanes):
                centers_mid_all.append(int(round((i + 0.5) * step)))
                widths_all.append(int(step))

        # 转到最终图像坐标（考虑左轴偏移）
        centers_mid_all = [int(x_offset + np.clip(c, 0, Wg - 1)) for c in centers_mid_all]

        # 非标准道索引（目标位置）
        skip_idx = max(0, int(ladder_lane) - 1)
        target_lane_idx = [i for i in range(real_nlanes) if i != skip_idx]

        # 字号依据目标列宽决定（按目标列）
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale, thick = 0.7, 1
        v_char_gap, h_margin = 2, 3

        # 按目标列宽获得每列建议字号与字符高宽
        per_col_scale, per_col_char_w, per_col_char_h = [], [], []
        for dst_i in range(min(cols_use, len(target_lane_idx))):
            col_lane = target_lane_idx[dst_i]
            max_w = max(10, widths_all[col_lane] - 2 * h_margin)
            scale = base_scale
            (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            if tw_char > max_w:
                scale = max(0.4, scale * (max_w / (tw_char + 1e-6)))
                (tw_char, th_char), _ = cv2.getTextSize("W", font, scale, thick)
            per_col_scale.append(scale)
            per_col_char_w.append(max(1, int(tw_char)))
            per_col_char_h.append(max(1, int(th_char)))

        # 面板高度（仅统计含非空文本的行）
        top_pad, bot_pad, row_gap = 8, 10, 6
        row_heights, rows_used_idx = [], []
        for r in range(rows):
            has_text = any((labels_table[r][c] or "").strip() for c in range(cols_use))
            if not has_text:
                row_heights.append(0)
                continue
            max_h = 0
            for c in range(min(cols_use, len(target_lane_idx))):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                ch_h = per_col_char_h[min(c, len(per_col_char_h)-1)]
                h_need = len(text) * (ch_h + v_char_gap) - v_char_gap
                max_h = max(max_h, h_need)
            max_h = max(max_h, int(0.8 * (max(per_col_char_h) if per_col_char_h else 12)))
            row_heights.append(max_h)
            rows_used_idx.append(r)
        if not rows_used_idx:
            return img_bgr

        panel_h = top_pad + sum(row_heights[r] for r in rows_used_idx) + (len(rows_used_idx) - 1) * row_gap + bot_pad
        panel = np.full((panel_h, W_total, 3), 255, dtype=np.uint8)

        # 绘制（竖排，按“非标准道”的目标列位置）
        y_cursor = top_pad
        for r in range(rows):
            if r not in rows_used_idx:
                continue
            rh = row_heights[r]
            for c in range(min(cols_use, len(target_lane_idx))):
                text = str(labels_table[r][c] or "").strip()
                if not text:
                    continue
                scale = per_col_scale[min(c, len(per_col_scale)-1)]
                ch_w = per_col_char_w[min(c, len(per_col_char_w)-1)]
                ch_h = per_col_char_h[min(c, len(per_col_char_h)-1)]
                x_center = centers_mid_all[target_lane_idx[c]]
                x = int(np.clip(x_center - ch_w // 2, 2, W_total - ch_w - 2))
                y = y_cursor + ch_h
                for ch in text:
                    cv2.putText(panel, ch, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
                    y += ch_h + v_char_gap
            y_cursor += rh + row_gap

        #cv2.rectangle(panel, (0, 0), (W_total - 1, panel_h - 1), (0, 0, 0), 1)
        return np.vstack([panel, img_bgr])



    # -------------------- 其他 -------------------- #
    def export_current(self):
        """
        导出当前右侧“标注结果”：
        - 优先从右侧交互式 Canvas（self.canvas_anno）合成：包含用户拖动后的位置（若有箭头）。
        - 若交互式 Canvas 不可用或未渲染，则回退导出不含红箭头的底图（render_cache["annotated_base"]）。
        - 继续兼容旧键：若没有 annotated_base，会尝试 annotated_final / annotated。
        """
        # 1) 优先从交互画布取“所见即所得”的合成结果
        img_to_save = None
        if hasattr(self, "canvas_anno") and hasattr(self.canvas_anno, "render_to_image"):
            try:
                img_to_save = self.canvas_anno.render_to_image()
            except Exception:
                img_to_save = None

        # 2) 回退：导出底图（不含红箭头）
        if img_to_save is None:
            if not getattr(self, "render_cache", None):
                messagebox.showwarning("提示", "请先渲染当前胶块。")
                return
            # 新版键：仅底图（不含红箭头）
            img_to_save = self.render_cache.get("annotated_base")
            # 兼容旧版键：annotated_final（含叠加内容）、annotated（不一定含白板）
            if img_to_save is None:
                img_to_save = self.render_cache.get("annotated_final") or self.render_cache.get("annotated")

            if img_to_save is None:
                messagebox.showwarning("提示", "未找到可导出的图像。")
                return

        # 3) 基本健壮性检查
        if not isinstance(img_to_save, np.ndarray) or img_to_save.size == 0:
            messagebox.showerror("错误", "导出图像无效。")
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
                # 使用源文件名作为前缀
                default_name = f"{p.stem}_annotated.png"
        except Exception:
            pass

        # 5) 保存对话框
        fpath = filedialog.asksaveasfilename(
            title="保存带标注的图片",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png")]
        )
        if not fpath:
            return

        # 6) PNG 编码并写入（tofile 适配 Windows 中文路径）
        ok, buf = cv2.imencode(".png", img_to_save)
        if not ok:
            messagebox.showerror("错误", "PNG 编码失败，未能导出。")
            return

        try:
            buf.tofile(fpath)   # Windows 下避免编码问题
        except Exception:
            try:
                with open(fpath, "wb") as f:
                    f.write(buf.tobytes())
            except Exception as e:
                messagebox.showerror("错误", f"写入文件失败：{e}")
                return

        messagebox.showinfo("完成", f"已导出：{fpath}")

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
        - 每行 = 一列（泳道）；允许空行（表示该列不标注）。
        - 第一项视为列名（文本），其余项解析为分子量（数值，>0）。
        - 支持分隔符：中文/英文逗号、空格、分号、Tab、冒号。
        - 完成后：仅非标准道会按顺序显示列名与红箭头；缺则空，多则省略。
        """
        win = tk.Toplevel(self)
        win.title("编辑列名与分子量（每行一列；空行=不标注）")
        win.transient(self)
        win.grab_set()

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(
            frm,
            text="示例：\nLane 1: 70,55,40\n样本A\t100 70 35\nB; 60;30;10\n（第一项=列名；空行=该列不标注；其后为分子量kDa）",
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
            pre_lines.append(line)  # 保留空行
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

            # 不对齐到泳道数；保持用户输入的行序，
            # 渲染阶段按“非标准道数量”做 截断/补空。
            return names, marks

        def do_ok():
            raw = txt.get("1.0", "end")
            names, marks = parse_lane_meta(raw)
            self.lane_names = names
            self.lane_marks = marks
            win.destroy()
            try:
                self.render_current()
            except Exception:
                pass

        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btns, text="确定", command=do_ok).pack(side=tk.RIGHT)


if __name__ == "__main__":
    App().mainloop()