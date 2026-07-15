"""Headless, process-isolated execution of multiple ACT-R simulations."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation.config.models import SimulationConfig


@dataclass(slots=True)
class MultiRunScenario:
    """One configurable scenario that can be repeated in a batch."""

    name: str
    repetitions: int = 1
    scheduling: str = "parallel"
    speed_factor: float = -1.0
    end_condition: str = "simulation_time"
    end_value: str = "10.0"
    config: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Every scenario needs a name.")
        if self.repetitions < 1:
            raise ValueError(f"Scenario '{self.name}' needs at least one repetition.")
        if self.scheduling not in {"parallel", "sequential"}:
            raise ValueError(f"Unknown scheduling mode in '{self.name}'.")
        if self.end_condition == "simulation_time":
            if float(self.end_value) <= 0:
                raise ValueError(f"Simulation time in '{self.name}' must be greater than zero.")
        elif self.end_condition == "production":
            if not self.end_value.strip():
                raise ValueError(f"Scenario '{self.name}' needs a production name.")
        else:
            raise ValueError(f"Unknown end condition in '{self.name}'.")
        if self.config.human_agent_enabled:
            raise ValueError("Human-controlled agents are not supported in Multi Simulation Run.")
        self.config.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repetitions": self.repetitions,
            "scheduling": self.scheduling,
            "speed_factor": self.speed_factor,
            "end_condition": self.end_condition,
            "end_value": self.end_value,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiRunScenario":
        return cls(
            name=str(payload.get("name", "Scenario")),
            repetitions=max(1, int(payload.get("repetitions", 1))),
            scheduling=str(payload.get("scheduling", "parallel")),
            speed_factor=float(payload.get("speed_factor", -1.0)),
            end_condition=str(payload.get("end_condition", "simulation_time")),
            end_value=str(payload.get("end_value", "10.0")),
            config=SimulationConfig.from_dict(payload.get("config")),
        )


@dataclass(slots=True)
class MultiRunBatch:
    scenarios: list[MultiRunScenario]
    output_path: str
    max_workers: int = 0
    max_events_per_run: int = 100_000

    def validate(self) -> None:
        if not self.scenarios:
            raise ValueError("Add at least one scenario.")
        if not self.output_path.strip():
            raise ValueError("Choose an output ZIP before starting the batch.")
        if self.max_workers < 0:
            raise ValueError("Worker count cannot be negative.")
        if self.max_events_per_run < 1:
            raise ValueError("The event safety limit must be at least 1.")
        for scenario in self.scenarios:
            scenario.validate()

    def expanded_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        run_number = 0
        for scenario_index, scenario in enumerate(self.scenarios, start=1):
            for repetition in range(1, scenario.repetitions + 1):
                run_number += 1
                tasks.append(
                    {
                        "run_number": run_number,
                        "scenario_index": scenario_index,
                        "scenario_name": scenario.name,
                        "repetition": repetition,
                        "scheduling": scenario.scheduling,
                        "speed_factor": scenario.speed_factor,
                        "end_condition": scenario.end_condition,
                        "end_value": scenario.end_value,
                        "config": scenario.config.without_human_agent().to_dict(),
                        "max_events": self.max_events_per_run,
                    }
                )
        return tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "output_path": self.output_path,
            "max_workers": self.max_workers,
            "max_events_per_run": self.max_events_per_run,
            "recommended_workers": recommended_worker_count(
                sum(s.repetitions for s in self.scenarios)
            ),
        }


def recommended_worker_count(task_count: int | None = None) -> int:
    """Choose a conservative CPU- and memory-sensitive process count."""
    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_limit = max(1, logical_cpus - 1)
    available_bytes = _available_memory_bytes()
    memory_limit = (
        max(1, int(available_bytes / (1.25 * 1024**3)))
        if available_bytes is not None
        else cpu_limit
    )
    limit = min(cpu_limit, memory_limit)
    if task_count is not None:
        limit = min(limit, max(1, int(task_count)))
    return max(1, limit)


def execute_multi_run_task(task: dict[str, Any], temp_root: str) -> dict[str, Any]:
    """Execute one simulation in a child process and export its full history."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    run_number = int(task["run_number"])
    started_wall = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "run_number": run_number,
        "scenario_index": task.get("scenario_index"),
        "scenario_name": task.get("scenario_name"),
        "repetition": task.get("repetition"),
        "scheduling": task.get("scheduling"),
        "speed_factor": task.get("speed_factor"),
        "end_condition": task.get("end_condition"),
        "end_value": task.get("end_value"),
        "environment_mode": (task.get("config") or {}).get("environment_mode", "virtual"),
        "status": "running",
        "target_reached": False,
        "event_count": 0,
        "simulation_time": 0.0,
        "error": None,
        "traceback": None,
        "history_path": None,
        "started_at_utc": started_at,
    }
    simulation = None
    try:
        from simulation.runtime.simulation import Simulation
        from simulation.runtime.tracer import Tracer

        config = SimulationConfig.from_dict(task.get("config")).without_human_agent()
        config.execution_mode = "single"
        config.speed_factor = float(task.get("speed_factor", -1.0))
        config.validate()
        tracer = Tracer()
        simulation = Simulation(tracer)
        simulation.start_simulation(config)

        max_events = max(1, int(task.get("max_events", 100_000)))
        end_condition = str(task.get("end_condition", "simulation_time"))
        end_value = str(task.get("end_value", "10.0"))
        target_time = float(end_value) if end_condition == "simulation_time" else None
        target_production = end_value.strip() if end_condition == "production" else None
        previous_time = 0.0

        for event_index in range(max_events):
            before_record_count = len(tracer.records)
            execution = simulation._execute_next_event()  # shared runtime executor
            result["event_count"] = event_index + 1
            current_time = float(simulation.global_sim_time)
            result["simulation_time"] = current_time
            new_records = tracer.records[before_record_count:]

            if end_condition == "simulation_time" and target_time is not None:
                if current_time >= target_time:
                    result["status"] = "completed"
                    result["target_reached"] = True
                    break
            elif target_production and any(
                _record_matches_production(record, target_production)
                for record in new_records
            ):
                result["status"] = "completed"
                result["target_reached"] = True
                break

            if not simulation.agent_list:
                result["status"] = "finished_before_condition"
                break

            _pace_simulation(
                simulated_delta=max(0.0, current_time - previous_time),
                speed_factor=float(task.get("speed_factor", -1.0)),
            )
            previous_time = current_time

            if execution is None and simulation.run_state == "finished":
                result["status"] = "finished_before_condition"
                break
        else:
            result["status"] = "safety_limit"

    except BaseException as exc:  # child failures must not abort the batch
        result["status"] = "crashed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if simulation is not None and getattr(simulation, "initialized", False):
            try:
                history_path = Path(temp_root) / f"run_{run_number:04d}_history.zip"
                simulation.export_history(history_path)
                result["history_path"] = str(history_path)
                result["simulation_time"] = float(simulation.global_sim_time)
            except BaseException as export_exc:
                export_message = f"History export failed: {type(export_exc).__name__}: {export_exc}"
                result["error"] = (
                    f"{result['error']}; {export_message}"
                    if result.get("error")
                    else export_message
                )
                if result["status"] == "completed":
                    result["status"] = "completed_history_error"
        result["duration_seconds"] = time.perf_counter() - started_wall
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def bundle_multi_run_results(
    batch: MultiRunBatch,
    results: list[dict[str, Any]],
) -> Path:
    """Create one archive containing all histories, errors, and summaries."""
    destination = Path(batch.output_path).expanduser()
    if destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(results, key=lambda row: int(row.get("run_number", 0)))
    public_results: list[dict[str, Any]] = []
    for row in ordered:
        public = dict(row)
        history_path = public.pop("history_path", None)
        run_number = int(public.get("run_number", 0))
        public["history_in_archive"] = (
            f"runs/run_{run_number:04d}/history/"
            if history_path and Path(history_path).exists()
            else None
        )
        public_results.append(public)
    manifest = {
        "schema_version": 1,
        "application": "ACT-R Multi Simulation Run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(ordered),
        "completed": sum(row.get("status") == "completed" for row in ordered),
        "crashed": sum(row.get("status") == "crashed" for row in ordered),
        "status_counts": _status_counts(ordered),
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        archive.writestr(
            "batch_configuration.json",
            json.dumps(batch.to_dict(), indent=2, ensure_ascii=False, default=str),
        )
        archive.writestr(
            "summary.json",
            json.dumps(public_results, indent=2, ensure_ascii=False, default=str),
        )
        archive.writestr("summary.csv", _summary_csv(public_results))
        archive.writestr("README.txt", _batch_readme())

        for row in ordered:
            run_number = int(row.get("run_number", 0))
            prefix = f"runs/run_{run_number:04d}"
            safe_row = dict(row)
            history_path = safe_row.pop("history_path", None)
            archive.writestr(
                f"{prefix}/status.json",
                json.dumps(safe_row, indent=2, ensure_ascii=False, default=str),
            )
            if row.get("traceback"):
                archive.writestr(f"{prefix}/error.txt", str(row["traceback"]))
            if history_path and Path(history_path).exists():
                with zipfile.ZipFile(history_path, "r") as history_archive:
                    for member in history_archive.infolist():
                        if member.is_dir():
                            continue
                        archive.writestr(
                            f"{prefix}/history/{member.filename}",
                            history_archive.read(member.filename),
                        )
    return destination


def create_batch_temp_directory() -> str:
    return tempfile.mkdtemp(prefix="actr_multi_run_")


def cleanup_batch_temp_directory(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _pace_simulation(*, simulated_delta: float, speed_factor: float) -> None:
    if speed_factor < 0 or simulated_delta <= 0:
        return
    delay = simulated_delta * (100.0 / speed_factor)
    if delay > 0:
        time.sleep(delay)


def _record_matches_production(record: dict[str, Any], target: str) -> bool:
    if str(record.get("type", "")).upper() != "PROCEDURAL":
        return False
    event = str(record.get("event", "")).strip()
    prefix = "RULE FIRED:"
    fired = event[len(prefix) :].strip() if event.upper().startswith(prefix) else event
    return fired.casefold() == target.casefold()


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def _status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _summary_csv(results: list[dict[str, Any]]) -> str:
    fields = [
        "run_number",
        "scenario_name",
        "repetition",
        "scheduling",
        "speed_factor",
        "end_condition",
        "end_value",
        "environment_mode",
        "status",
        "target_reached",
        "event_count",
        "simulation_time",
        "duration_seconds",
        "error",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue()


def _batch_readme() -> str:
    return """ACT-R multi-simulation archive
================================

manifest.json
    Batch-level counts and status summary.
batch_configuration.json
    All scenarios, simulation settings, scheduling modes, and limits.
summary.json / summary.csv
    One row per run, including crashes and stop conditions.
runs/run_XXXX/status.json
    Detailed status and timing for the individual run.
runs/run_XXXX/history/
    The complete regular simulation-history export for that run.
runs/run_XXXX/error.txt
    Traceback when the run crashed. Other runs continue independently.
"""
