"""Non-blocking background work with one global progress/ETA indicator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from PyQt6.QtCore import QCoreApplication, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar


ProgressCallback = Callable[[int, str | None], None]
TaskCallable = Callable[[ProgressCallback], Any]


class _WorkerSignals(QObject):
    started = pyqtSignal(str, str, int)
    progress = pyqtSignal(str, int, str, int)
    succeeded = pyqtSignal(str, object, float, int)
    failed = pyqtSignal(str, str, float, int)


class _Worker(QRunnable):
    def __init__(
        self,
        *,
        task_id: str,
        title: str,
        generation: int,
        job: TaskCallable,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.title = title
        self.generation = generation
        self.job = job
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        started = perf_counter()
        self.signals.started.emit(
            self.task_id, self.title, self.generation
        )

        def progress(value: int, stage: str | None = None) -> None:
            bounded = max(0, min(100, int(value)))
            self.signals.progress.emit(
                self.task_id,
                bounded,
                str(stage or self.title),
                self.generation,
            )

        try:
            progress(1, self.title)
            result = self.job(progress)
            _move_qobjects_to_gui_thread(result)
            progress(100, "Finishing")
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            self.signals.failed.emit(
                self.task_id,
                f"{type(exc).__name__}: {exc}",
                perf_counter() - started,
                self.generation,
            )
            return
        self.signals.succeeded.emit(
            self.task_id,
            result,
            perf_counter() - started,
            self.generation,
        )


@dataclass(slots=True)
class _TaskState:
    title: str
    generation: int
    started_at: float
    progress: int = 0
    stage: str = ""


class TaskProgressWidget(QFrame):
    """Compact bottom-left progress indicator shared by all graph views."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("backgroundTaskProgress")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 8, 1)
        layout.setSpacing(8)
        self.label = QLabel("", self)
        self.label.setMinimumWidth(340)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(170)
        self.progress.setFixedHeight(11)
        layout.addWidget(self.progress)
        layout.addWidget(self.label)
        self.hide()

    def update_task(
        self,
        *,
        title: str,
        stage: str,
        progress: int,
        eta_seconds: float | None,
        queued: int,
    ) -> None:
        self.progress.setValue(progress)
        eta = (
            f" · about {max(0.0, eta_seconds):.1f} s remaining"
            if eta_seconds is not None
            else " · estimating time…"
        )
        queue_text = f" · {queued} queued" if queued else ""
        detail = stage if stage and stage != title else title
        self.label.setText(f"Loading: {detail}{eta}{queue_text}")
        self.show()

    def show_completion(self, text: str) -> None:
        self.progress.setValue(100)
        self.label.setText(text)
        self.show()


class BackgroundTaskManager(QObject):
    """Serialize expensive GUI preparation outside the GUI thread.

    A single worker is intentional: graph layout is CPU-heavy and should not
    compete with the running simulation. Repeated requests with the same task id
    supersede older results; obsolete work may finish, but its result is ignored.
    """

    task_failed = pyqtSignal(str)

    def __init__(self, progress_widget: TaskProgressWidget, parent=None) -> None:
        super().__init__(parent)
        self.progress_widget = progress_widget
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.signals = _WorkerSignals(self)
        self.signals.started.connect(self._started)
        self.signals.progress.connect(self._progress)
        self.signals.succeeded.connect(self._succeeded)
        self.signals.failed.connect(self._failed)
        self._generations: dict[str, int] = {}
        self._callbacks: dict[tuple[str, int], Callable[[Any], None]] = {}
        self._errors: dict[tuple[str, int], Callable[[str], None] | None] = {}
        self._states: dict[tuple[str, int], _TaskState] = {}
        self._workers: dict[tuple[str, int], _Worker] = {}
        self._duration_history: dict[str, float] = {}
        self._active_key: tuple[str, int] | None = None
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.progress_widget.hide)
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(200)
        self._eta_timer.timeout.connect(self._update_widget)

    def submit(
        self,
        task_id: str,
        title: str,
        job: TaskCallable,
        on_success: Callable[[Any], None],
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> int:
        generation = self._generations.get(task_id, 0) + 1
        self._generations[task_id] = generation
        key = (task_id, generation)
        self._callbacks[key] = on_success
        self._errors[key] = on_error
        self._states[key] = _TaskState(
            title=title,
            generation=generation,
            started_at=perf_counter(),
            stage=title,
        )
        worker = _Worker(
            task_id=task_id,
            title=title,
            generation=generation,
            job=job,
            signals=self.signals,
        )
        self._workers[key] = worker
        self._hide_timer.stop()
        self.pool.start(worker)
        self._update_widget()
        return generation

    def invalidate(self, task_id: str) -> None:
        """Invalidate results and remove workers that have not started yet."""
        self._generations[task_id] = self._generations.get(task_id, 0) + 1
        for key in [key for key in self._states if key[0] == task_id]:
            worker = self._workers.get(key)
            removed = False
            if worker is not None:
                try:
                    removed = bool(self.pool.tryTake(worker))
                except (AttributeError, RuntimeError):
                    removed = False
            if not removed:
                continue
            self._workers.pop(key, None)
            self._states.pop(key, None)
            self._callbacks.pop(key, None)
            self._errors.pop(key, None)
        self._update_widget()

    def is_running(self, task_id: str) -> bool:
        return any(
            key[0] == task_id and self.is_current(key[0], key[1])
            for key in self._states
        )

    def is_current(self, task_id: str, generation: int) -> bool:
        return self._generations.get(task_id) == generation

    def _started(self, task_id: str, title: str, generation: int) -> None:
        key = (task_id, generation)
        state = self._states.get(key)
        if state is None:
            return
        state.started_at = perf_counter()
        state.title = title
        self._active_key = key
        if not self._eta_timer.isActive():
            self._eta_timer.start()
        self._update_widget()

    def _progress(
        self,
        task_id: str,
        value: int,
        stage: str,
        generation: int,
    ) -> None:
        key = (task_id, generation)
        state = self._states.get(key)
        if state is None:
            return
        state.progress = value
        state.stage = stage
        self._active_key = key
        self._update_widget()

    def _succeeded(
        self,
        task_id: str,
        result: Any,
        duration: float,
        generation: int,
    ) -> None:
        key = (task_id, generation)
        state = self._states.pop(key, None)
        self._workers.pop(key, None)
        callback = self._callbacks.pop(key, None)
        self._errors.pop(key, None)
        if state is not None:
            previous = self._duration_history.get(state.title)
            self._duration_history[state.title] = (
                duration if previous is None else previous * 0.65 + duration * 0.35
            )
        current = self.is_current(task_id, generation)
        if current and callback is not None:
            callback(result)
        self._finish_visible_task(
            f"Loaded: {state.title if state is not None else task_id}"
            if current
            else "Obsolete graph result discarded"
        )

    def _failed(
        self,
        task_id: str,
        message: str,
        duration: float,
        generation: int,
    ) -> None:
        del duration
        key = (task_id, generation)
        state = self._states.pop(key, None)
        self._workers.pop(key, None)
        self._callbacks.pop(key, None)
        error_callback = self._errors.pop(key, None)
        if self.is_current(task_id, generation):
            if error_callback is not None:
                error_callback(message)
            self.task_failed.emit(message)
        self._finish_visible_task(
            f"Could not load {state.title if state is not None else task_id}: {message}"
        )

    def _finish_visible_task(self, message: str) -> None:
        self._active_key = None
        pending = self._current_states()
        if pending:
            self._active_key = pending[0][0]
            self._update_widget()
            return
        self._eta_timer.stop()
        try:
            self.progress_widget.show_completion(message)
            self._hide_timer.start(1600)
        except RuntimeError:
            self._eta_timer.stop()

    def _current_states(self) -> list[tuple[tuple[str, int], _TaskState]]:
        values = [
            (key, state)
            for key, state in self._states.items()
            if self.is_current(key[0], key[1])
        ]
        return sorted(values, key=lambda item: item[1].started_at)

    def _update_widget(self) -> None:
        current = self._current_states()
        if not current:
            return
        key, state = (
            next(
                (item for item in current if item[0] == self._active_key),
                current[0],
            )
        )
        self._active_key = key
        elapsed = max(0.0, perf_counter() - state.started_at)
        eta: float | None = None
        if 2 <= state.progress < 100 and elapsed > 0.05:
            eta = elapsed * (100 - state.progress) / state.progress
        else:
            estimate = self._duration_history.get(
                state.title, self._default_duration(state.title)
            )
            eta = max(0.0, estimate - elapsed)
        try:
            self.progress_widget.update_task(
                title=state.title,
                stage=state.stage,
                progress=state.progress,
                eta_seconds=eta,
                queued=max(0, len(current) - 1),
            )
        except RuntimeError:
            # The owning window may have been closed while a background graph
            # task was finishing. The result is obsolete and must not crash Qt.
            self._eta_timer.stop()

    @staticmethod
    def _default_duration(title: str) -> float:
        lowered = title.casefold()
        if "detailed state" in lowered:
            return 4.0
        if "declarative" in lowered:
            return 3.5
        if "state graph" in lowered:
            return 2.2
        if "buffer" in lowered:
            return 1.4
        if "jump" in lowered:
            return 0.8
        return 2.0


def _move_qobjects_to_gui_thread(value: Any) -> None:
    """Move worker-created scenes/QObjects before they enter the GUI thread."""
    application = QCoreApplication.instance()
    if application is None:
        return
    gui_thread = application.thread()
    if isinstance(value, QObject):
        items_bounds = getattr(value, "itemsBoundingRect", None)
        set_scene_rect = getattr(value, "setSceneRect", None)
        if callable(items_bounds) and callable(set_scene_rect):
            try:
                set_scene_rect(items_bounds().adjusted(-24, -24, 24, 24))
            except Exception:
                pass
        value.moveToThread(gui_thread)
        return
    if isinstance(value, dict):
        for item in value.values():
            _move_qobjects_to_gui_thread(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _move_qobjects_to_gui_thread(item)
