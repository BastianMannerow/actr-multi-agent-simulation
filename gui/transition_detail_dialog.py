"""Explainable details for one production or adapter transition."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class TransitionDetailDialog(QDialog):
    """Present one or several bundled transitions without crowding the graph."""

    def __init__(self, payload: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        transitions = list(payload.get("transitions") or [])
        if not transitions and payload.get("payload_type") == "state_transition_explanation":
            transitions = [payload]
        self._transitions = transitions
        caption = str(payload.get("caption") or (transitions[0].get("code") if transitions else "Transition"))
        self.setWindowTitle(f"{caption} · Explainability")
        self.resize(780, 680)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(18, 18, 18, 18)
        self._outer.setSpacing(12)

        self._selector: QComboBox | None = None
        if len(transitions) > 1:
            selector = QComboBox(self)
            for transition in transitions:
                selector.addItem(
                    f"{transition.get('code', '')} · {transition.get('title', 'Transition')}"
                )
            selector.currentIndexChanged.connect(self._render_transition)
            self._selector = selector
            self._outer.addWidget(selector)

        self._content = QWidget(self)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(12)
        self._outer.addWidget(self._content, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        self._outer.addWidget(buttons)
        self._render_transition(0)

    def _render_transition(self, index: int) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._transitions:
            self._content_layout.addWidget(QLabel("No transition details are available."))
            return
        transition = self._transitions[max(0, min(index, len(self._transitions) - 1))]

        heading = QLabel(
            f"{transition.get('code', '')} · {transition.get('title', 'Transition')}"
        )
        heading.setObjectName("sectionTitle")
        heading.setWordWrap(True)
        self._content_layout.addWidget(heading)

        summary = QLabel(str(transition.get("summary", "")))
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._content_layout.addWidget(summary)

        form_frame = QFrame(self)
        form_frame.setObjectName("panel")
        form = QFormLayout(form_frame)
        form.setContentsMargins(12, 12, 12, 12)
        form.addRow("Type", _selectable(transition.get("kind", "")))
        form.addRow("Source state", _selectable(transition.get("source_state", "")))
        form.addRow("Target state", _selectable(transition.get("target_state", "")))
        if transition.get("trigger_production"):
            form.addRow("Triggered after", _selectable(transition["trigger_production"]))
        if transition.get("utility") is not None:
            form.addRow("Utility", _selectable(transition["utility"]))
        if transition.get("reward") is not None:
            form.addRow("Reward", _selectable(transition["reward"]))
        self._content_layout.addWidget(form_frame)

        modules = transition.get("modules") or []
        module_lines = []
        for item in modules:
            modes = ", ".join(item.get("modes") or [])
            module_lines.append(
                f"• {item.get('module', 'Module')} — {item.get('buffer', '')} ({modes})"
            )
        self._content_layout.addWidget(
            _section("Involved ACT-R modules", "\n".join(module_lines) or "Goal")
        )
        self._content_layout.addWidget(
            _section("Conditions / guard", transition.get("guard", ""))
        )
        self._content_layout.addWidget(
            _section("Effects / actions", transition.get("actions", ""))
        )
        self._content_layout.addStretch(1)


def _selectable(value: Any) -> QLabel:
    label = QLabel(str(value))
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _section(title: str, content: Any) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    heading = QLabel(title)
    heading.setStyleSheet("font-weight: 600;")
    layout.addWidget(heading)
    browser = QTextBrowser()
    browser.setPlainText(str(content or "—"))
    browser.setMinimumHeight(100)
    layout.addWidget(browser)
    return page
