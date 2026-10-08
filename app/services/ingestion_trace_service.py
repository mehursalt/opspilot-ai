"""Trace service for document ingestion pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class IngestionStep:
    name: str
    status: str
    start_ts: str
    end_ts: str = ""
    duration_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class IngestionTraceService:
    """Write ingestion pipeline runs as JSONL."""

    def __init__(self) -> None:
        self.logs_dir = Path("logs")
        self.trace_file = self.logs_dir / "ingestion_traces.jsonl"

    def start_run(self, file_path: str) -> dict[str, Any]:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "file_path": file_path,
            "status": "RUNNING",
            "steps": [],
        }

    def start_step(self, run: dict[str, Any], name: str, detail: dict[str, Any] | None = None) -> IngestionStep:
        step = IngestionStep(
            name=name,
            status="RUNNING",
            start_ts=datetime.now().isoformat(timespec="milliseconds"),
            detail=detail or {},
        )
        run.setdefault("steps", []).append(step)
        return step

    def finish_step(self, step: IngestionStep, detail: dict[str, Any] | None = None) -> None:
        start = datetime.fromisoformat(step.start_ts)
        end = datetime.now()
        step.status = "SUCCESS"
        step.end_ts = end.isoformat(timespec="milliseconds")
        step.duration_ms = int((end - start).total_seconds() * 1000)
        if detail:
            step.detail.update(detail)

    def fail_step(self, step: IngestionStep, error: Exception | str) -> None:
        start = datetime.fromisoformat(step.start_ts)
        end = datetime.now()
        step.status = "ERROR"
        step.end_ts = end.isoformat(timespec="milliseconds")
        step.duration_ms = int((end - start).total_seconds() * 1000)
        step.error = str(error)

    def finish_run(self, run: dict[str, Any], status: str = "SUCCESS", error: str = "") -> dict[str, Any]:
        run["status"] = status
        run["end_ts"] = datetime.now().isoformat(timespec="seconds")
        run["error"] = error
        payload = {
            **run,
            "steps": [asdict(step) if isinstance(step, IngestionStep) else step for step in run.get("steps", [])],
        }
        self._append(payload)
        return payload

    def _append(self, payload: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


ingestion_trace_service = IngestionTraceService()
