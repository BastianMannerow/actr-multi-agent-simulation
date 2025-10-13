import tkinter as tk
from tkinter import font, filedialog
import random
from PIL import Image, ImageDraw, ImageFont


class StepLogWindow:
    """
    A dedicated, scrollable visualization window for simulation step logs.

    Purpose
    -------
    - Visualize per-agent event timelines in a grid (event types × timestamps).
    - Keep a persistent left column for row labels (event types).
    - Provide synchronized vertical scrolling for labels and data.
    - Offer a JUMP field to trigger navigation to a specific production.
    - Export the current view as a pixel-perfect PNG via PIL.

    Parameters
    ----------
    master : tk.Misc | None
        Parent widget. Creates a Toplevel if provided, otherwise a Tk root.
    tracer : Any | None
        Object exposing `known_agents: Set[str]` and `records: List[dict]]`.
        Records must contain: "timestamp", "type", "event", "agent_name".
    simulation : Any | None
        Object exposing `start_jump(name: str)` to navigate to a production.
    title : str
        Window title.

    Notes
    -----
    - Event grid columns are unique timestamps, rows are unique event types.
    - Cell height adapts to the maximum line count of multi-line event texts.
    - Colors are assigned per event type from a soft randomized pastel palette.
    - Export uses a combined PIL canvas so the PNG mirrors the on-screen grid.
    """

    def __init__(self, master=None, tracer=None, simulation=None, title="Stepper Log"):
        self.tracer     = tracer
        self.simulation = simulation

        # Window: root or child toplevel
        self.window = tk.Toplevel(master) if master else tk.Tk()
        self.window.title(title)
        self.window.configure(bg="#2e1111")

        # Grid: left listbox, right plot area (labels + data canvases)
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=0)
        self.window.grid_columnconfigure(1, weight=1)

        # Top control row: production JUMP + PNG export
        self.input_frame = tk.Frame(self.window, bg="#2e1111")
        self.input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 10))
        self.input_frame.grid_columnconfigure(1, weight=1)

        # UI labels remain in German by design; this does not affect code semantics.
        tk.Label(self.input_frame, text="Produktion:", bg="#2e1111", fg="white")\
          .grid(row=0, column=0, padx=5)
        self.jump_entry = tk.Entry(self.input_frame)
        self.jump_entry.grid(row=0, column=1, padx=5, sticky="ew")
        tk.Button(self.input_frame, text="JUMP", command=self.on_jump)\
          .grid(row=0, column=2, padx=5)
        tk.Button(self.input_frame, text="Download PNG", command=self.on_download)\
          .grid(row=0, column=3, padx=5)

        # Agent list on the left
        self.listbox = tk.Listbox(self.window, width=20, bg="#2e1111", fg="white")
        self.listbox.grid(row=1, column=0, sticky="ns", padx=(5, 0), pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_agent_select)

        # Plot area on the right: fixed label canvas + scrollable data canvas + scrollbars
        self.plot_frame = tk.Frame(self.window, bg="#2e1111")
        self.plot_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 5), pady=5)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(1, weight=1)

        # Left canvas for persistent row labels
        self.label_canvas = tk.Canvas(self.plot_frame, background="#2e1111", highlightthickness=0)
        # Right canvas for the scrollable grid content
        self.data_canvas  = tk.Canvas(self.plot_frame, background="#2e1111")
        # Shared vertical scrollbar + data-only horizontal scrollbar
        self.vsb = tk.Scrollbar(self.plot_frame, orient="vertical", command=self._on_vertical_scroll)
        self.hsb = tk.Scrollbar(self.plot_frame, orient="horizontal", command=self.data_canvas.xview)

        # Wire scroll commands
        self.label_canvas.configure(yscrollcommand=self.vsb.set)
        self.data_canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        # Layout canvases and scrollbars
        self.label_canvas.grid(row=0, column=0, sticky="ns")
        self.data_canvas .grid(row=0, column=1, sticky="nsew")
        self.vsb         .grid(row=0, column=2, sticky="ns")
        self.hsb         .grid(row=1, column=1, sticky="ew")

        # Screen resolution for sizing heuristics
        self.screen_w = self.window.winfo_screenwidth()
        self.screen_h = self.window.winfo_screenheight()

        # Cell metrics: width targets ~7 columns visible; height at least one text line
        self.cell_w       = max(self.screen_w // 7, 40)
        self.base_cell_h  = max(self.screen_h // 20, 40)

        # Font for measurement and drawing
        self.font = font.nametofont("TkDefaultFont")

        # Event-type → pastel color cache
        self.color_map = {}

        # PIL image buffer for export
        self.pil_image = None

        # Redraw on canvas resize to keep grid consistent with viewport
        self.data_canvas.bind('<Configure>', lambda e: self.redraw_current())

        # Initial state
        self.current_agent = None
        self.refresh_agent_list()

    def _on_vertical_scroll(self, *args):
        """
        Synchronize vertical scrolling between the label and the data canvases.

        Tkinter limitation: two canvases cannot share the same scrollbar without
        manually forwarding the scroll commands to both.
        """
        self.label_canvas.yview(*args)
        self.data_canvas .yview(*args)

    def refresh_agent_list(self):
        """
        Rebuild the agent sidebar from `tracer.known_agents` and keep selection.

        Behavior
        --------
        - Preserves the previous selection if still present.
        - Selects the first agent if the previous selection is gone.
        - Clears selection if no agents are available.
        """
        names = sorted(self.tracer.known_agents) if self.tracer else []
        prev  = self.current_agent
        self.listbox.delete(0, tk.END)
        for n in names:
            self.listbox.insert(tk.END, n)
        if prev in names:
            i = names.index(prev)
            self.listbox.selection_set(i)
            self.current_agent = prev
        elif names:
            self.listbox.selection_set(0)
            self.current_agent = names[0]
        else:
            self.current_agent = None

    def on_agent_select(self, event):
        """
        Handle agent selection changes from the sidebar and trigger a redraw.
        """
        sel = self.listbox.curselection()
        if not sel:
            return
        self.current_agent = self.listbox.get(sel[0])
        self.redraw_current()

    def redraw_current(self):
        """
        Redraw the grid for the currently selected agent if any.
        """
        if not self.current_agent:
            return
        self.show_agent_logs(self.current_agent)

    def show_agent_logs(self, agent_name: str):
        """
        Render the timeline grid for a given agent and prepare a PIL export image.

        Pipeline
        --------
        1) Filter tracer records by `agent_name`.
        2) Extract unique, sorted event types (rows) and timestamps (columns).
        3) Compute cell metrics:
           - Row label width from type text metrics.
           - Cell height from font line space and multi-line event content.
        4) Reset canvases and set scrollregions.
        5) Draw:
           - Left label column with event types (Tk + PIL).
           - Top timestamp headers (Tk + PIL).
           - Grid cells with type-based colors and event texts (Tk + PIL).
        6) Store the composed PIL image for `on_download()`.

        Constraints
        -----------
        - Assumes records contain comparable numeric timestamps.
        - Uses a default PIL bitmap font for portability.
        - Colors are deterministic per run only if `random` is seeded externally.
        """
        # 1) Select records for the chosen agent
        recs = [r for r in (self.tracer.records if self.tracer else []) if r["agent_name"] == agent_name]
        if not recs:
            return

        # 2) Unique types (rows) and timestamps (columns)
        types = sorted({r["type"] for r in recs})
        times = sorted({r["timestamp"] for r in recs})
        margin_y = 30  # top margin for headers

        # 3) Label metrics and adaptive cell height based on wrapped content
        max_type_w = max((self.font.measure(str(t)) for t in types), default=0)
        line_h     = self.font.metrics("linespace")

        event_texts = [str(r["event"]) for r in recs if r.get("event") is not None]
        max_lines   = max((t.count('\n') + 1 for t in event_texts), default=1)
        cell_h      = max(self.base_cell_h, max_lines * line_h + 4)

        cols    = len(times)
        rows    = len(types)
        total_h = margin_y + rows * cell_h + 20
        total_w = cols * self.cell_w + 20
        label_w = max_type_w + 20  # padding around type labels

        # 4) Reset canvases and declare scrollable extents
        self.label_canvas.delete("all")
        self.data_canvas .delete("all")
        self.label_canvas.config(scrollregion=(0, 0, label_w, total_h))
        self.data_canvas .config(scrollregion=(0, 0, total_w, total_h))

        # Prepare PIL export surface spanning both label and data regions
        img  = Image.new("RGB", (label_w + total_w, total_h), "#2e1111")
        draw = ImageDraw.Draw(img)
        pil_font = ImageFont.load_default()

        # 5.1) Row labels (left, persistent, white)
        for i, t in enumerate(types):
            y = margin_y + i * cell_h + cell_h // 2
            # Tk canvas
            self.label_canvas.create_text(label_w // 2, y,
                                          text=str(t), fill="white",
                                          font=self.font)
            # PIL mirror
            draw.text((5, y - line_h // 2), str(t), fill="white", font=pil_font)

        # 5.2) Column headers (timestamps) across the top
        for j, ts in enumerate(times):
            x = j * self.cell_w + self.cell_w // 2
            ts_str = f"{ts:.2f}"
            # Tk canvas
            self.data_canvas.create_text(x, margin_y // 2,
                                         text=ts_str, fill="white",
                                         font=self.font)
            # PIL mirror (centered by approximate text width)
            draw.text((label_w + x - self.font.measure(ts_str) // 2, 5),
                      ts_str, fill="white", font=pil_font)

        # 5.3) Cells and event payloads
        for r in recs:
            i = types.index(r["type"])
            j = times.index(r["timestamp"])
            x1 = j * self.cell_w
            y1 = margin_y + i * cell_h
            x2 = x1 + self.cell_w
            y2 = y1 + cell_h

            typ = r["type"]
            if typ not in self.color_map:
                # Soft pastel assignment; seed externally to stabilize across runs.
                r_col = random.randint(180, 255)
                g_col = random.randint(180, 255)
                b_col = random.randint(180, 255)
                self.color_map[typ] = f'#{r_col:02x}{g_col:02x}{b_col:02x}'
            fill = self.color_map[typ]

            # Tk rectangle
            self.data_canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="white")

            evt = r.get("event")
            if evt is not None:
                # Tk multi-line text, wrapped within the cell
                self.data_canvas.create_text(x1 + self.cell_w / 2,
                                             y1 + cell_h / 2,
                                             text=str(evt), fill="black",
                                             font=self.font,
                                             width=self.cell_w - 4)

            # PIL rectangle
            draw.rectangle([label_w + x1, y1, label_w + x2, y2], fill=fill, outline="white")

            if evt is not None:
                # PIL multi-line text, vertically stacked and horizontally centered
                lines = str(evt).split('\n')
                for k, line in enumerate(lines):
                    wy = y1 + k * line_h + 2
                    lw = draw.textlength(line, font=pil_font)
                    wx = label_w + x1 + (self.cell_w - lw) / 2
                    draw.text((wx, wy), line, fill="black", font=pil_font)

        # 6) Keep the composed image for download
        self.pil_image = img

    def on_download(self):
        """
        Save the last rendered PIL image as a PNG via a standard file dialog.

        No-op if nothing was rendered yet.
        """
        if not self.pil_image:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png")])
        if path:
            self.pil_image.save(path)

    def log(self, message: str = None):
        """
        External hook: refresh agent list and redraw current selection.

        This can be called after the tracer ingests new records.
        """
        self.refresh_agent_list()
        if self.current_agent:
            self.redraw_current()

    def on_jump(self):
        """
        Trigger simulation-level jump to a production name entered in the field.

        Requires `simulation.start_jump(str)` to be present. Silent if empty.
        """
        prod_name = self.jump_entry.get().strip()
        if prod_name and self.simulation:
            self.simulation.start_jump(prod_name)
