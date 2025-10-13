from typing import Any, List, Optional, Tuple
import tkinter as tk


class ExampleGUI:
    """
    Compact grid renderer for an Environment with `level_matrix`.

    Contract
    --------
    - Expects `env.level_matrix` as `List[List[List[Any]]]`.
    - Calls to `update()` fully redraw the view (idempotent).
    - Headless if no Tk root is attached.

    Responsibilities
    ----------------
    - Paint a rectilinear grid.
    - Draw agents (objects with a `name` attribute) as labeled discs.
    - Keep per-label colors stable via a deterministic cache.

    Non-Responsibilities
    --------------------
    - No timers, no input, no simulation control, no layout decisions beyond cell size.

    Accessibility
    -------------
    - High-contrast grid and labels.
    - Short labels to reduce clutter at small cell sizes.
    """

    def __init__(self, env: Any, root: Optional[tk.Misc] = None, *, cell_px: int = 40) -> None:
        """
        Initialize the view.

        Parameters
        ----------
        env : Any
            Environment exposing `level_matrix`.
        root : Optional[tk.Misc]
            Tk root or Toplevel. If None, run headless.
        cell_px : int
            Pixel size for square cells. Clamped to a sane minimum.
        """
        self.env = env
        self.root: Optional[tk.Misc] = root
        self.cell_px = max(8, int(cell_px))
        self.canvas: Optional[tk.Canvas] = None
        self._color_cache: dict[str, str] = {}

        if self.root is not None:
            self._ensure_canvas()
            self.update()

    # ---------- Public API ----------
    def update(self) -> None:
        """
        Redraw the entire matrix. Safe to call frequently.

        Notes
        -----
        - Full repaint keeps logic simple and avoids stale artifacts.
        - For very large grids, consider dirty-rect optimizations upstream.
        """
        if self.canvas is None:
            return

        matrix: List[List[List[Any]]] = self.env.level_matrix
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0

        self._resize_canvas(cols, rows)
        self.canvas.delete("all")

        # Grid
        for r in range(rows):
            for c in range(cols):
                x1, y1, x2, y2 = self._cell_bounds(c, r)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#2A2A33", fill="#17171F")

        # Agents
        for r in range(rows):
            for c in range(cols):
                cell = matrix[r][c]
                # Draw any object that exposes a non-empty string `name`
                for obj in cell:
                    if obj is None:
                        continue
                    label = getattr(obj, "name", None)
                    if isinstance(label, str) and label:
                        self._draw_agent(c, r, label)

        self.canvas.update_idletasks()

    def set_root(self, root: tk.Misc) -> None:
        """
        Attach a Tk root after construction. Enables rendering if previously headless.
        """
        self.root = root
        self._ensure_canvas()
        self.update()

    # ---------- Internals ----------
    def _ensure_canvas(self) -> None:
        if self.canvas is not None or self.root is None:
            return
        rows = len(self.env.level_matrix)
        cols = len(self.env.level_matrix[0]) if rows else 0
        w = max(1, cols * self.cell_px)
        h = max(1, rows * self.cell_px)
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#101014", highlightthickness=0)
        self.canvas.pack(fill="both", expand=False)

    def _resize_canvas(self, cols: int, rows: int) -> None:
        if self.canvas is None:
            return
        self.canvas.config(width=max(1, cols * self.cell_px), height=max(1, rows * self.cell_px))

    def _cell_bounds(self, col: int, row: int) -> Tuple[int, int, int, int]:
        x1 = col * self.cell_px
        y1 = row * self.cell_px
        x2 = x1 + self.cell_px
        y2 = y1 + self.cell_px
        return x1, y1, x2, y2

    def _draw_agent(self, col: int, row: int, label: str) -> None:
        """
        Render a filled circle with a compact label.

        Label policy
        ------------
        - Up to 4 characters to remain readable at small sizes.
        """
        x1, y1, x2, y2 = self._cell_bounds(col, row)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        r = self.cell_px * 0.35

        color = self._color_for(label)
        assert self.canvas is not None
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="#ECECF1", width=1)

        short = (label or "A")[:4]
        self.canvas.create_text(
            cx, cy, text=short, fill="#0B0B0D",
            font=("TkDefaultFont", max(8, int(self.cell_px * 0.28)), "bold")
        )

    def _color_for(self, key: str) -> str:
        """
        Return a deterministic pastel color for a given key.

        Determinism
        -----------
        - Based on Python's hash of the label; cached for stability within a run.
        """
        hit = self._color_cache.get(key)
        if hit:
            return hit
        h = abs(hash(key)) % 360
        r, g, b = self._hsl_to_rgb(h / 360.0, 0.45, 0.72)
        hexcol = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        self._color_cache[key] = hexcol
        return hexcol

    # Small HSL→RGB utility (no external dependencies)
    def _hsl_to_rgb(self, h: float, s: float, l: float):
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        return r, g, b
