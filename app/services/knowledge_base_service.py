"""Knowledge-base file management service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.vector_index_service import vector_index_service
from app.services.vector_store_manager import vector_store_manager


class KnowledgeBaseService:
    """List, reindex and delete files in the uploads knowledge base."""

    def __init__(self) -> None:
        self.upload_dir = Path("uploads")
        self.trace_file = Path("logs") / "ingestion_traces.jsonl"
        self.allowed_extensions = {".txt", ".md", ".markdown"}

    def summary(self) -> dict[str, Any]:
        files = self.list_files()["files"]
        success_count = sum(1 for item in files if item["index_status"] == "SUCCESS")
        error_count = sum(1 for item in files if item["index_status"] == "ERROR")
        return {
            "file_count": len(files),
            "indexed_count": success_count,
            "error_count": error_count,
            "untracked_count": len(files) - success_count - error_count,
            "total_size": sum(item["size"] for item in files),
        }

    def list_files(self) -> dict[str, Any]:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        traces = self.list_traces(limit=1000)["traces"]
        latest_by_file: dict[str, dict[str, Any]] = {}
        for trace in traces:
            filename = Path(str(trace.get("file_path", ""))).name
            if filename and filename not in latest_by_file:
                latest_by_file[filename] = trace

        files: list[dict[str, Any]] = []
        for path in sorted(self.upload_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file() or path.suffix.lower() not in self.allowed_extensions:
                continue
            stat = path.stat()
            trace = latest_by_file.get(path.name)
            split_step = self._find_step(trace, "split_document") if trace else None
            index_step = self._find_step(trace, "index_vectors") if trace else None
            files.append(
                {
                    "filename": path.name,
                    "extension": path.suffix.lower().lstrip("."),
                    "size": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "index_status": trace.get("status", "NOT_TRACKED") if trace else "NOT_TRACKED",
                    "chunks": self._step_detail(split_step, "chunks", 0),
                    "stored": self._step_detail(index_step, "stored", False),
                    "duration_ms": self._trace_duration(trace),
                    "last_indexed_at": trace.get("end_ts", trace.get("ts", "")) if trace else "",
                    "error": trace.get("error", "") if trace else "",
                }
            )
        return {
            "files": files,
            "summary": {
                "file_count": len(files),
                "total_size": sum(item["size"] for item in files),
            },
        }

    def list_traces(self, filename: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        if not self.trace_file.exists():
            return {"traces": traces}

        safe_name = Path(filename).name if filename else ""
        lines = self.trace_file.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            trace_name = Path(str(trace.get("file_path", ""))).name
            if safe_name and trace_name != safe_name:
                continue
            trace["filename"] = trace_name
            traces.append(trace)
            if len(traces) >= max(1, limit):
                break
        return {"traces": traces}

    def reindex_file(self, filename: str) -> dict[str, Any]:
        path = self._resolve_upload_file(filename)
        vector_index_service.index_single_file(str(path))
        return {
            "filename": path.name,
            "status": "SUCCESS",
            "message": "reindex finished",
        }

    def get_file_path(self, filename: str) -> Path:
        """Return a safe local path for an uploaded knowledge file."""
        return self._resolve_upload_file(filename)

    def delete_file(self, filename: str) -> dict[str, Any]:
        path = self._resolve_upload_file(filename)
        normalized_path = path.as_posix()
        vector_store_manager.delete_by_source(normalized_path)
        path.unlink()
        return {
            "filename": path.name,
            "status": "SUCCESS",
            "message": "file and vectors deleted",
        }

    def _resolve_upload_file(self, filename: str) -> Path:
        safe_name = Path(filename).name
        if not safe_name:
            raise ValueError("文件名不能为空")
        path = (self.upload_dir / safe_name).resolve()
        upload_root = self.upload_dir.resolve()
        if path.parent != upload_root:
            raise ValueError("只能操作 uploads 目录下的文件")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {safe_name}")
        if path.suffix.lower() not in self.allowed_extensions:
            raise ValueError("不支持的文件类型")
        return path

    @staticmethod
    def _find_step(trace: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
        if not trace:
            return None
        for step in trace.get("steps", []):
            if step.get("name") == name:
                return step
        return None

    @staticmethod
    def _step_detail(step: dict[str, Any] | None, key: str, default: Any) -> Any:
        if not step:
            return default
        return step.get("detail", {}).get(key, default)

    @staticmethod
    def _trace_duration(trace: dict[str, Any] | None) -> int:
        if not trace:
            return 0
        return sum(int(step.get("duration_ms", 0)) for step in trace.get("steps", []))


knowledge_base_service = KnowledgeBaseService()
