"""Export a complete, portable simulation history archive."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from simulation.discovery.agent_discovery import AgentDiscovery
from simulation.inspection.declarative_memory import DeclarativeMemoryInspector


class SimulationHistoryExporter:
    """Write timelines, buffer changes, configuration, and final state to ZIP."""

    SCHEMA_VERSION = 3

    def export(self, path: str | Path, simulation: Any) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".zip":
            destination = destination.with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)

        records = list(
            getattr(getattr(simulation, "interceptor", None), "records", [])
        )
        history_recorder = getattr(simulation, "buffer_history", None)
        all_buffer_histories = (
            history_recorder.all_histories()
            if history_recorder is not None
            else {}
        )
        agents = list(getattr(simulation, "agent_list", []))
        spatial_agents = list(getattr(simulation, "spatial_agents", agents))
        config = getattr(simulation, "config", None)
        config_payload = (
            config.to_dict()
            if config is not None and hasattr(config, "to_dict")
            else {}
        )

        try:
            plugins = [info.to_dict() for info in AgentDiscovery().discover()]
        except Exception as exc:
            plugins = [{"discovery_error": f"{type(exc).__name__}: {exc}"}]

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "application": "ACT-R Demo Simulation",
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "simulation_time": float(
                getattr(simulation, "global_sim_time", 0.0)
            ),
            "execution_mode": getattr(simulation, "execution_mode", None),
            "run_state": getattr(simulation, "run_state", None),
            "environment_mode": config_payload.get("environment_mode"),
            "environment_label": config_payload.get("environment_label"),
            "virtual_level": config_payload.get("virtual_level"),
            "event_count": len(records),
            "agent_count": len(spatial_agents),
            "cognitive_agent_count": len(agents),
            "human_agent_count": sum(
                bool(getattr(agent, "is_human_controlled", False))
                for agent in spatial_agents
            ),
            "agents": [self._serialize_agent(agent) for agent in spatial_agents],
            "formats": {
                "json": "Structured metadata, complete arrays, and current state",
                "jsonl": "Append-friendly event and buffer histories",
                "csv": "Tabular analysis in spreadsheet and statistics tools",
            },
        }

        with zipfile.ZipFile(
            destination,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            self._write_json(archive, "manifest.json", manifest)
            self._write_json(archive, "configuration.json", config_payload)
            self._write_json(archive, "agent_plugins.json", plugins)
            archive.writestr("README.txt", self._readme_text())

            self._write_json(archive, "events/all_events.json", records)
            self._write_jsonl(archive, "events/all_events.jsonl", records)
            self._write_csv(
                archive,
                "events/all_events.csv",
                records,
                ["sequence", "timestamp", "agent_name", "type", "event"],
            )

            agent_names = {
                str(record.get("agent_name", ""))
                for record in records
                if record.get("agent_name")
            } | set(all_buffer_histories)
            agent_names.update(
                str(getattr(agent, "name", ""))
                for agent in spatial_agents
                if getattr(agent, "name", None)
            )

            for agent_name in sorted(agent_names):
                slug = self._slug(agent_name)
                agent_records = [
                    record
                    for record in records
                    if str(record.get("agent_name", "")) == agent_name
                ]
                self._write_json(
                    archive,
                    f"agents/{slug}/timeline.json",
                    agent_records,
                )
                self._write_jsonl(
                    archive,
                    f"agents/{slug}/timeline.jsonl",
                    agent_records,
                )
                self._write_csv(
                    archive,
                    f"agents/{slug}/timeline.csv",
                    agent_records,
                    [
                        "sequence",
                        "timestamp",
                        "agent_name",
                        "type",
                        "event",
                    ],
                )

                buffers = all_buffer_histories.get(agent_name, {})
                current: dict[str, Any] = {}
                buffer_catalog: list[dict[str, Any]] = []
                for buffer_name, entries in sorted(buffers.items()):
                    buffer_slug = self._slug(buffer_name)
                    flattened = [
                        self._flatten_buffer_entry(entry)
                        for entry in entries
                    ]
                    self._write_json(
                        archive,
                        f"agents/{slug}/buffers/{buffer_slug}.json",
                        entries,
                    )
                    self._write_jsonl(
                        archive,
                        f"agents/{slug}/buffers/{buffer_slug}.jsonl",
                        entries,
                    )
                    self._write_csv(
                        archive,
                        f"agents/{slug}/buffers/{buffer_slug}.csv",
                        flattened,
                        [
                            "sequence",
                            "timestamp",
                            "agent_name",
                            "buffer_name",
                            "reason",
                            "change",
                            "event_type",
                            "event",
                            "buffer_class",
                            "buffer_state",
                            "empty",
                            "content",
                            "module_attributes",
                        ],
                    )
                    if entries:
                        snapshot = entries[-1].get("snapshot") or {}
                        current[buffer_name] = snapshot
                        buffer_catalog.append(
                            {
                                "name": buffer_name,
                                "class": snapshot.get("buffer_class"),
                                "history_entries": len(entries),
                            }
                        )
                self._write_json(
                    archive,
                    f"agents/{slug}/buffers/current.json",
                    current,
                )
                self._write_json(
                    archive,
                    f"agents/{slug}/buffers/catalog.json",
                    buffer_catalog,
                )

            self._write_json(
                archive,
                "final_state/environment.json",
                self._serialize_environment(
                    getattr(simulation, "game_environment", None)
                ),
            )
            self._write_json(
                archive,
                "final_state/productions.json",
                self._serialize_productions(agents),
            )
            self._write_json(
                archive,
                "final_state/declarative_memory.json",
                {
                    str(getattr(agent, "name", "")): {
                        "memories": snapshot.memories,
                        "chunks": [asdict(chunk) for chunk in snapshot.chunks],
                        "edges": [asdict(edge) for edge in snapshot.edges],
                    }
                    for agent in agents
                    for snapshot in [DeclarativeMemoryInspector.inspect_agent(agent)]
                },
            )

        return destination

    @staticmethod
    def _serialize_agent(agent: Any) -> dict[str, Any]:
        adapter = getattr(agent, "actr_adapter", None)
        model = getattr(agent, "actr_agent", None)
        return {
            "name": str(getattr(agent, "name", "")),
            "type": str(getattr(agent, "actr_agent_type_name", "")),
            "actr_time": float(getattr(agent, "actr_time", 0.0)),
            "model_class": (
                f"{type(model).__module__}.{type(model).__name__}"
                if model is not None
                else None
            ),
            "human_controlled": bool(
                getattr(agent, "is_human_controlled", False)
            ),
            "adapter_class": (
                f"{type(adapter).__module__}.{type(adapter).__name__}"
                if adapter is not None
                else None
            ),
        }

    @staticmethod
    def _flatten_buffer_entry(entry: dict[str, Any]) -> dict[str, Any]:
        snapshot = entry.get("snapshot") or {}
        chunks = snapshot.get("chunks", [])
        return {
            "sequence": entry.get("sequence"),
            "timestamp": entry.get("timestamp"),
            "agent_name": entry.get("agent_name"),
            "buffer_name": entry.get("buffer_name"),
            "reason": entry.get("reason"),
            "change": entry.get("change"),
            "event_type": entry.get("event_type"),
            "event": entry.get("event"),
            "buffer_class": snapshot.get("buffer_class"),
            "buffer_state": snapshot.get("state"),
            "empty": snapshot.get("empty"),
            "content": json.dumps(
                chunks, ensure_ascii=False, sort_keys=True
            ),
            "module_attributes": json.dumps(
                snapshot.get("module_attributes", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    @staticmethod
    def _serialize_environment(environment: Any) -> dict[str, Any]:
        matrix = getattr(environment, "level_matrix", None)
        if not matrix:
            return {"backend": None, "width": 0, "height": 0, "cells": []}
        cells = []
        for row_index, row in enumerate(matrix):
            for column_index, cell in enumerate(row):
                occupants = []
                for item in cell:
                    occupants.append(
                        {
                            "name": getattr(item, "name", None),
                            "class": (
                                f"{type(item).__module__}."
                                f"{type(item).__name__}"
                            ),
                            "display_name": getattr(item, "display_name", None),
                            "symbol": getattr(item, "symbol", None),
                            "blocks_movement": bool(getattr(item, "blocks_movement", False)),
                            "repr": repr(item),
                        }
                    )
                if occupants:
                    cells.append(
                        {
                            "row": row_index,
                            "column": column_index,
                            "occupants": occupants,
                        }
                    )
        return {
            "backend": getattr(environment, "backend_name", None),
            "width": len(matrix[0]) if matrix and matrix[0] else 0,
            "height": len(matrix),
            "cells": cells,
        }

    @staticmethod
    def _serialize_productions(agents: list[Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for agent in agents:
            model = getattr(agent, "actr_agent", None)
            productions = (
                getattr(model, "productions", {})
                if model is not None
                else {}
            )
            rows = []
            try:
                iterable = productions.items()
            except AttributeError:
                iterable = []
            for name, definition in iterable:
                rows.append(
                    {
                        "name": str(name),
                        "utility": (
                            definition.get("utility")
                            if isinstance(definition, dict)
                            else None
                        ),
                        "reward": (
                            definition.get("reward")
                            if isinstance(definition, dict)
                            else None
                        ),
                    }
                )
            result[str(getattr(agent, "name", ""))] = rows
        return result

    @staticmethod
    def _write_json(
        archive: zipfile.ZipFile, name: str, payload: Any
    ) -> None:
        archive.writestr(
            name,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )

    @staticmethod
    def _write_jsonl(
        archive: zipfile.ZipFile,
        name: str,
        rows: Iterable[dict[str, Any]],
    ) -> None:
        content = "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n"
            for row in rows
        )
        archive.writestr(name, content)

    @staticmethod
    def _write_csv(
        archive: zipfile.ZipFile,
        name: str,
        rows: Iterable[dict[str, Any]],
        fieldnames: list[str],
    ) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        archive.writestr(name, buffer.getvalue())

    @staticmethod
    def _slug(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return normalized.strip("._") or "unnamed"

    @staticmethod
    def _readme_text() -> str:
        return """ACT-R simulation history export
================================

manifest.json
    Archive schema, export metadata, runtime state, and agent list.
configuration.json
    Complete GUI/runtime configuration used to build the simulation.
agent_plugins.json
    Dynamically discovered model and adapter files, classes, and errors.
events/
    Complete global timeline in JSON, JSON Lines, and CSV.
agents/<agent>/timeline.*
    Per-agent event timeline in JSON, JSON Lines, and CSV.
agents/<agent>/buffers/
    One JSON, JSONL, and CSV history per pyactr buffer. current.json stores
    the final snapshot of every buffer and catalog.json lists buffer classes.
final_state/environment.json
    Final grid occupancy.
final_state/productions.json
    Production names and available utility/reward metadata.
final_state/declarative_memory.json
    Runtime declarative memories, chunks, time traces, activations, and inferred links.

JSON preserves complete arrays and nested snapshots. JSONL supports streaming
processing. CSV provides flattened rows for Excel, R, pandas, or statistical
software. All timestamps are simulation time.
"""
