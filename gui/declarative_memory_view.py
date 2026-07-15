"""Declarative-memory graph tab for the running agent inspector."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.analysis_graphs import (
    ZoomableGraphicsView,
    build_declarative_memory_scene,
)
from gui.llm_export import declarative_memory_payload
from simulation.inspection.declarative_memory import (
    DeclarativeMemoryInspector,
    DeclarativeMemorySnapshot,
)


class DeclarativeMemoryInspectorTab(QWidget):
    """Render runtime memory without performing graph work in the GUI thread."""

    LIVE_MIN_INTERVAL_SECONDS = 0.25
    LIVE_RENDER_BUDGET_SECONDS = 0.16
    LIVE_GUI_DUTY_CYCLE = 0.10
    LIVE_ABSOLUTE_CHUNK_LIMIT = 180
    LIVE_ABSOLUTE_COMPLEXITY_LIMIT = 900

    PAUSED_FULL_CHUNK_LIMIT = 500
    PAUSED_FULL_COMPLEXITY_LIMIT = 1_500
    PAUSED_PAGE_SIZE = 100
    PAUSED_PAGE_EDGE_LIMIT = 240

    LIVE_WARNING = (
        "Live declarative-memory rendering has been paused because the graph "
        "complexity exceeds real-time GUI performance. Pause the simulation "
        "to load the current graph."
    )

    def __init__(self, parent=None, *, task_manager=None) -> None:
        super().__init__(parent)
        self.task_manager = task_manager
        self._task_id = f"runtime-declarative-memory:{id(self)}"
        self._request_generation = 0
        self._agent_name: str | None = None
        self._agent_identity: int | None = None
        self._current_agent: Any | None = None
        self._signature: tuple[Any, ...] | None = None
        self._requested_signature: tuple[Any, ...] | None = None
        self._next_live_update_at = 0.0
        self._live_suspended = False
        self._last_render_seconds = 0.0
        self._last_chunk_count = 0
        self._last_complexity = 0
        self._total_chunk_count = 0
        self._page_start = 0
        self._paged_mode = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        toolbar = QHBoxLayout()
        self.info_label = QLabel("No runtime agent selected")
        self.info_label.setObjectName("muted")
        self.page_label = QLabel("")
        self.page_label.setObjectName("muted")
        fit_button = QPushButton("Fit View")
        png_button = QPushButton("Export PNG")
        svg_button = QPushButton("Export SVG")
        llm_button = QPushButton("Export for LLM")
        toolbar.addWidget(self.info_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.page_label)
        toolbar.addWidget(fit_button)
        toolbar.addWidget(png_button)
        toolbar.addWidget(svg_button)
        toolbar.addWidget(llm_button)
        layout.addLayout(toolbar)

        self.live_notice_label = QLabel(self.LIVE_WARNING)
        self.live_notice_label.setWordWrap(True)
        self.live_notice_label.setStyleSheet(
            "QLabel { color: #fde68a; background: #422006; "
            "border: 1px solid #a16207; border-radius: 5px; padding: 7px; }"
        )
        self.live_notice_label.hide()
        layout.addWidget(self.live_notice_label)

        self.graph = ZoomableGraphicsView(self)
        layout.addWidget(self.graph, 1)

        fit_button.clicked.connect(self.graph.reset_zoom)
        png_button.clicked.connect(lambda: self.graph.export_dialog("png"))
        svg_button.clicked.connect(lambda: self.graph.export_dialog("svg"))
        llm_button.clicked.connect(self.graph.export_for_llm_dialog)
        self._update_page_controls()

    @property
    def live_updates_suspended(self) -> bool:
        return self._live_suspended

    def update_agent(
        self,
        agent: Any | None,
        *,
        automatic_running: bool = False,
        force: bool = False,
    ) -> None:
        if agent is None:
            self._reset_runtime_state()
            self.info_label.setText("No runtime agent selected")
            snapshot = DeclarativeMemoryInspector.inspect_agent(None)
            title = "Declarative Memory"
            self.graph.setScene(
                build_declarative_memory_scene(snapshot, title=title)
            )
            self.graph.set_llm_export_data(
                declarative_memory_payload(snapshot, title=title),
                default_name="declarative_memory",
            )
            return

        agent_name = str(getattr(agent, "name", "Agent"))
        agent_identity = id(agent)
        changed_agent = (
            agent_name != self._agent_name
            or agent_identity != self._agent_identity
        )
        if changed_agent:
            self._agent_name = agent_name
            self._agent_identity = agent_identity
            self._current_agent = agent
            self._signature = None
            self._requested_signature = None
            self._next_live_update_at = 0.0
            self._live_suspended = False
            self._last_render_seconds = 0.0
            self._last_chunk_count = 0
            self._last_complexity = 0
            self._page_start = 0
            self._paged_mode = False
            self.live_notice_label.hide()

        now = perf_counter()
        if automatic_running:
            self._paged_mode = False
            self._update_page_controls()
            if self._live_suspended:
                self.live_notice_label.setText(self.LIVE_WARNING)
                self.live_notice_label.show()
                return
            if not force and now < self._next_live_update_at:
                return
            if (
                self.task_manager is not None
                and self.task_manager.is_running(self._task_id)
            ):
                return
            _, chunk_count, estimated_edges = (
                DeclarativeMemoryInspector.estimate_agent_graph_complexity(
                    agent,
                    detailed_chunk_limit=self.LIVE_ABSOLUTE_CHUNK_LIMIT,
                )
            )
            self._total_chunk_count = chunk_count
            estimated_complexity = chunk_count + estimated_edges
            if self._live_preflight_exceeds_budget(
                chunk_count, estimated_complexity
            ):
                self._suspend_live_updates()
                return
            snapshot_factory = lambda: DeclarativeMemoryInspector.inspect_agent(
                agent
            )
            page_key = (0, chunk_count)
        else:
            self._live_suspended = False
            self._next_live_update_at = 0.0
            self.live_notice_label.hide()
            _, chunk_count, estimated_edges = (
                DeclarativeMemoryInspector.estimate_agent_graph_complexity(
                    agent,
                    detailed_chunk_limit=self.PAUSED_FULL_CHUNK_LIMIT,
                )
            )
            self._total_chunk_count = chunk_count
            estimated_complexity = chunk_count + estimated_edges
            self._paged_mode = (
                chunk_count > self.PAUSED_FULL_CHUNK_LIMIT
                or estimated_complexity > self.PAUSED_FULL_COMPLEXITY_LIMIT
            )
            if self._paged_mode:
                if changed_agent or self._page_start >= chunk_count:
                    self._page_start = max(
                        0, chunk_count - self.PAUSED_PAGE_SIZE
                    )
                self._page_start = max(
                    0,
                    min(
                        self._page_start,
                        max(0, chunk_count - self.PAUSED_PAGE_SIZE),
                    ),
                )
                start = self._page_start
                snapshot_factory = lambda: (
                    DeclarativeMemoryInspector.inspect_agent_window(
                        agent,
                        chunk_offset=start,
                        chunk_limit=self.PAUSED_PAGE_SIZE,
                        max_edges=self.PAUSED_PAGE_EDGE_LIMIT,
                    )
                )
                page_key = (
                    start,
                    min(chunk_count, start + self.PAUSED_PAGE_SIZE),
                )
            else:
                self._page_start = 0
                snapshot_factory = lambda: DeclarativeMemoryInspector.inspect_agent(
                    agent
                )
                page_key = (0, chunk_count)
            self._update_page_controls()

        request_signature = (
            agent_identity,
            page_key,
            chunk_count,
            automatic_running,
        )
        if (
            request_signature == self._requested_signature
            and not force
            and not changed_agent
        ):
            return
        self._requested_signature = request_signature
        self._request_generation += 1
        request_generation = self._request_generation
        title = f"Current Declarative Memory — {agent_name}"

        def job(progress):
            started = perf_counter()
            progress(8, f"Reading declarative memory · {agent_name}")
            snapshot = snapshot_factory()
            progress(36, "Inferring memory relationships")
            signature = (
                page_key,
                tuple(snapshot.memories),
                tuple(
                    (
                        chunk.chunk_id,
                        chunk.label,
                        tuple(sorted(chunk.slots.items())),
                        tuple(chunk.traces),
                        chunk.activation,
                    )
                    for chunk in snapshot.chunks
                ),
                tuple(
                    (
                        edge.source_id,
                        edge.target_id,
                        edge.label,
                        edge.relation,
                    )
                    for edge in snapshot.edges
                ),
            )
            progress(58, "Routing declarative-memory graph")
            scene = build_declarative_memory_scene(snapshot, title=title)
            progress(92, "Preparing LLM export")
            payload = declarative_memory_payload(snapshot, title=title)
            return {
                "scene": scene,
                "snapshot": snapshot,
                "signature": signature,
                "payload": payload,
                "render_seconds": max(perf_counter() - started, 0.001),
                "page_key": page_key,
                "chunk_count": chunk_count,
                "agent_identity": agent_identity,
                "generation": request_generation,
                "automatic_running": automatic_running,
            }

        def apply(result: dict[str, Any]) -> None:
            if (
                result["generation"] != self._request_generation
                or result["agent_identity"] != self._agent_identity
            ):
                return
            snapshot = result["snapshot"]
            signature = result["signature"]
            render_seconds = result["render_seconds"]
            if signature != self._signature or changed_agent:
                self.graph.setScene(result["scene"])
                self.graph.set_llm_export_data(
                    result["payload"],
                    default_name=f"{agent_name}_declarative_memory",
                )
                self._signature = signature
                if changed_agent:
                    self.graph.reset_zoom()
            complexity = (
                len(snapshot.chunks)
                + len(snapshot.edges)
                + len(snapshot.operations)
            )
            self._last_render_seconds = render_seconds
            self._last_chunk_count = len(snapshot.chunks)
            self._last_complexity = complexity
            suffix = (
                f" · showing {page_key[0] + 1:,}–{page_key[1]:,}"
                if self._paged_mode and chunk_count
                else ""
            )
            self.info_label.setText(
                f"{agent_name} · {len(snapshot.memories)} retrieval-linked memories · "
                f"{chunk_count:,} query-matching chunks{suffix} · "
                f"{render_seconds * 1000:.0f} ms"
            )
            if result["automatic_running"]:
                self._schedule_next_live_update(render_seconds)
                if (
                    render_seconds > self.LIVE_RENDER_BUDGET_SECONDS
                    or complexity > self.LIVE_ABSOLUTE_COMPLEXITY_LIMIT
                ):
                    self._suspend_live_updates()

        if self.task_manager is None:
            apply(job(lambda _value, _stage=None: None))
        else:
            self.task_manager.submit(
                self._task_id,
                f"Declarative Memory Graph · {agent_name}",
                job,
                apply,
            )

    def _live_preflight_exceeds_budget(
        self,
        chunk_count: int,
        estimated_complexity: int | None = None,
    ) -> bool:
        if chunk_count > self.LIVE_ABSOLUTE_CHUNK_LIMIT:
            return True
        current_complexity = max(
            chunk_count,
            estimated_complexity
            if estimated_complexity is not None
            else chunk_count,
        )
        if current_complexity > self.LIVE_ABSOLUTE_COMPLEXITY_LIMIT:
            return True
        if self._last_render_seconds <= 0.0 or self._last_chunk_count <= 0:
            return False
        chunk_growth = chunk_count / max(self._last_chunk_count, 1)
        complexity_growth = current_complexity / max(
            self._last_complexity, 1
        )
        growth = max(1.0, chunk_growth, complexity_growth)
        predicted = self._last_render_seconds * growth**1.35
        return predicted > self.LIVE_RENDER_BUDGET_SECONDS

    def _schedule_next_live_update(self, render_seconds: float) -> None:
        interval = max(
            self.LIVE_MIN_INTERVAL_SECONDS,
            render_seconds / self.LIVE_GUI_DUTY_CYCLE,
        )
        self._next_live_update_at = perf_counter() + interval

    def _suspend_live_updates(self) -> None:
        self._live_suspended = True
        self.live_notice_label.setText(self.LIVE_WARNING)
        self.live_notice_label.show()

    def _update_page_controls(self) -> None:
        if self._paged_mode and self._total_chunk_count:
            end = min(
                self._total_chunk_count,
                self._page_start + self.PAUSED_PAGE_SIZE,
            )
            self.page_label.setText(
                f"{self._page_start + 1:,}–{end:,} of "
                f"{self._total_chunk_count:,}"
            )
        else:
            self.page_label.setText("")

    def _reset_runtime_state(self) -> None:
        self._request_generation += 1
        if self.task_manager is not None:
            self.task_manager.invalidate(self._task_id)
        self._agent_name = None
        self._agent_identity = None
        self._current_agent = None
        self._signature = None
        self._requested_signature = None
        self._next_live_update_at = 0.0
        self._live_suspended = False
        self._last_render_seconds = 0.0
        self._last_chunk_count = 0
        self._last_complexity = 0
        self._total_chunk_count = 0
        self._page_start = 0
        self._paged_mode = False
        self.live_notice_label.hide()
        self._update_page_controls()
