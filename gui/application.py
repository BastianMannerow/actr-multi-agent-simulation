"""QApplication construction and global application configuration."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication

from gui.resources import application_icon_path
from gui.styles import APP_STYLESHEET


def _configure_windows_taskbar_identity() -> None:
    """Assign a stable Windows AppUserModelID for the taskbar icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ACTR.MultiAgentSimulation"
        )
    except Exception:
        pass


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the active QApplication or create and configure one."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        icon_path = application_icon_path()
        if icon_path.exists():
            existing.setWindowIcon(QIcon(str(icon_path)))
        return existing

    _configure_windows_taskbar_identity()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("ACT-R Multi-Agent Simulation")
    app.setOrganizationName("ACT-R Simulation Framework")
    icon_path = application_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)
    return app
