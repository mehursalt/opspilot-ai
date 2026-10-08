"""Vector indexing service.

This module reads local Markdown/text documents, splits them into chunks and
stores them in the vector database. The indexing path now records an ingestion
trace so the RAG pipeline can be inspected after uploads or batch indexing.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.ingestion_trace_service import ingestion_trace_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """Result summary for directory indexing."""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        self.success_count += 1

    def increment_fail_count(self):
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """Index local documents into the configured vector store."""

    def __init__(self):
        self.upload_path = "./uploads"
        logger.info("Vector index service initialized")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory does not exist or is invalid: {target_path}")

            result.directory_path = str(dir_path)
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning("No supported files found in directory: {}", target_path)
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info("Start indexing directory: {}, files={}", target_path, len(files))

            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info("File indexed successfully: {}", file_path.name)
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error("File indexing failed: {}, error={}", file_path.name, e)

            result.success = result.fail_count == 0
            result.end_time = datetime.now()
            logger.info(
                "Directory indexing finished: total={}, success={}, failed={}",
                result.total_files,
                result.success_count,
                result.fail_count,
            )
            return result

        except Exception as e:
            logger.error("Directory indexing failed: {}", e)
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {file_path}")

        logger.info("Start indexing file: {}", path)
        run = ingestion_trace_service.start_run(str(path))
        current_step = None

        try:
            current_step = ingestion_trace_service.start_step(
                run,
                "read_file",
                {"path": str(path)},
            )
            content = path.read_text(encoding="utf-8")
            ingestion_trace_service.finish_step(current_step, {"chars": len(content)})
            logger.info("File read: {}, chars={}", path, len(content))

            normalized_path = path.as_posix()
            current_step = ingestion_trace_service.start_step(
                run,
                "delete_old_vectors",
                {"source": normalized_path},
            )
            vector_store_manager.delete_by_source(normalized_path)
            ingestion_trace_service.finish_step(current_step)

            current_step = ingestion_trace_service.start_step(
                run,
                "split_document",
                {"source": normalized_path},
            )
            documents = document_splitter_service.split_document(content, normalized_path)
            ingestion_trace_service.finish_step(current_step, {"chunks": len(documents)})
            logger.info("Document split finished: {} -> {} chunks", file_path, len(documents))

            current_step = ingestion_trace_service.start_step(
                run,
                "index_vectors",
                {"source": normalized_path, "chunks": len(documents)},
            )
            if documents:
                vector_store_manager.add_documents(documents)
                ingestion_trace_service.finish_step(
                    current_step,
                    {"chunks": len(documents), "stored": True},
                )
                logger.info("File indexing finished: {}, chunks={}", file_path, len(documents))
            else:
                ingestion_trace_service.finish_step(
                    current_step,
                    {"chunks": 0, "stored": False},
                )
                logger.warning("File is empty or cannot be split: {}", file_path)

            ingestion_trace_service.finish_run(run, "SUCCESS")

        except Exception as e:
            if current_step and current_step.status == "RUNNING":
                ingestion_trace_service.fail_step(current_step, str(e))
            ingestion_trace_service.finish_run(run, "ERROR", str(e))
            logger.error("File indexing failed: {}, error={}", file_path, e)
            raise RuntimeError(f"File indexing failed: {e}") from e


vector_index_service = VectorIndexService()
