"""Live visualization of a production-jump prerequisite path."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from gui.analysis_graphs import (
    ZoomableGraphicsView,
    build_jump_progress_scene,
)
from gui.llm_export import jump_payload
from simulation.inspection.source_analysis import AgentStaticAnalysis


class JumpProgressDialog(QDialog):
    """Modeless window that highlights jump progress after every ACT-R event."""

    def __init__(
        self,
        *,
        analysis: AgentStaticAnalysis,
        tracer: Any,
        agent_name: str,
        target_production: str,
        start_record_index: int,
        parent=None,
        task_manager=None,
    ) -> None:
        super().__init__(parent)
        self.analysis = analysis
        self.tracer = tracer
        self.agent_name = agent_name
        self.target_production = target_production
        self.start_record_index = start_record_index
        self.task_manager = task_manager
        self._task_id = f"jump-graph:{id(self)}"
        self._render_generation = 0
        self._scan_index = start_record_index
        self._fired_cache: list[str] = []
        self._rendered_fired_count = -1
        self._view_initialized = False
        self.setWindowTitle(
            f"Jump Progress — {agent_name} → {target_production}"
        )
        self.resize(1180, 620)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.status_label = QLabel("")
        header.addWidget(self.status_label)
        header.addStretch(1)
        fit_button = QPushButton("Fit View")
        png_button = QPushButton("Export PNG")
        svg_button = QPushButton("Export SVG")
        llm_button = QPushButton("Export for LLM")
        close_button = QPushButton("Close")
        header.addWidget(fit_button)
        header.addWidget(png_button)
        header.addWidget(svg_button)
        header.addWidget(llm_button)
        header.addWidget(close_button)
        layout.addLayout(header)

        self.graph = ZoomableGraphicsView(self)
        layout.addWidget(self.graph, 1)
        fit_button.clicked.connect(self.graph.reset_zoom)
        png_button.clicked.connect(lambda: self.graph.export_dialog("png"))
        svg_button.clicked.connect(lambda: self.graph.export_dialog("svg"))
        llm_button.clicked.connect(self.graph.export_for_llm_dialog)
        close_button.clicked.connect(self.close)
        self.refresh()

    def refresh(self) -> None:
        fired = self._fired_productions()
        if len(fired) != self._rendered_fired_count:
            self._schedule_graph_render(list(fired))
        transition_path = self.analysis.transition_path_to_production(
            self.target_production
        )
        if transition_path is None:
            self.status_label.setText(
                "Static path unavailable — monitoring runtime events only."
            )
            return
        fired_folded = [name.casefold() for name in fired]
        completed = 0
        consumed = 0
        for transition in transition_path:
            if transition.kind == "adapter":
                trigger = (transition.trigger_production or "").casefold()
                if trigger and trigger in fired_folded[:consumed]:
                    completed += 1
                    continue
                break
            while consumed < len(fired_folded):
                current = fired_folded[consumed]
                consumed += 1
                if (transition.production_name or "").casefold() == current:
                    completed += 1
                    break
            else:
                break
        if completed >= len(transition_path):
            self.status_label.setText("Target production fired.")
        else:
            next_transition = transition_path[completed]
            prefix = "adapter" if next_transition.kind == "adapter" else "production"
            self.status_label.setText(
                f"{completed}/{len(transition_path)} transitions reached · "
                f"next {prefix}: {next_transition.label}"
            )

    def _schedule_graph_render(self, fired: list[str]) -> None:
        if (
            self.task_manager is not None
            and self.task_manager.is_running(self._task_id)
        ):
            return
        self._render_generation += 1
        generation = self._render_generation

        def job(progress):
            progress(15, "Preparing Production Jump path")
            scene = build_jump_progress_scene(
                self.analysis, self.target_production, fired
            )
            progress(90, "Preparing LLM export")
            return {
                "scene": scene,
                "payload": jump_payload(
                    self.analysis, self.target_production, fired
                ),
                "fired_count": len(fired),
                "generation": generation,
            }

        def apply(result):
            if result["generation"] != self._render_generation:
                return
            self.graph.setScene(result["scene"])
            self.graph.set_llm_export_data(
                result["payload"],
                default_name=(
                    f"{self.agent_name}_{self.target_production}_jump_progress"
                ),
            )
            self._rendered_fired_count = result["fired_count"]
            if not self._view_initialized:
                self.graph.reset_zoom()
                self._view_initialized = True

        if self.task_manager is None:
            apply(job(lambda _value, _stage=None: None))
        else:
            self.task_manager.submit(
                self._task_id,
                f"Production Jump Graph · {self.agent_name}",
                job,
                apply,
            )

    def _fired_productions(self) -> list[str]:
        records = getattr(self.tracer, "records", ())
        end = len(records)
        while self._scan_index < end:
            record = records[self._scan_index]
            self._scan_index += 1
            if str(record.get("agent_name", "")) != self.agent_name:
                continue
            if str(record.get("type", "")).upper() != "PROCEDURAL":
                continue
            event = str(record.get("event", "")).strip()
            prefix = "RULE FIRED:"
            if event.upper().startswith(prefix):
                self._fired_cache.append(event[len(prefix) :].strip())
        return self._fired_cache
