# ============================================================
# FILE: pipeline_monitor.py
# PART 1: CORE CONFIGURATION, IMPORTS & PIPELINE STATUS SYSTEM
# PROJECT: Ferry Demand Forecasting & Predictive Decision Support
# ============================================================

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
import uuid

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

PIPELINE_OUTPUT_DIR = OUTPUT_DIR / "pipeline_monitoring"

PIPELINE_LOG_DIR = PIPELINE_OUTPUT_DIR / "logs"

PIPELINE_REPORT_DIR = PIPELINE_OUTPUT_DIR / "reports"

PIPELINE_STATE_FILE = PIPELINE_OUTPUT_DIR / "pipeline_state.json"

PIPELINE_HISTORY_FILE = PIPELINE_OUTPUT_DIR / "pipeline_history.jsonl"

PIPELINE_SUMMARY_FILE = PIPELINE_REPORT_DIR / "pipeline_summary.json"

PIPELINE_ERROR_FILE = PIPELINE_REPORT_DIR / "pipeline_errors.json"

PIPELINE_EVENTS_FILE = PIPELINE_LOG_DIR / "pipeline_events.jsonl"

PIPELINE_LOG_FILE = PIPELINE_LOG_DIR / "pipeline_monitor.log"


# ============================================================
# 2. PIPELINE DIRECTORY INITIALIZATION
# ============================================================

def initialize_pipeline_directories() -> None:
    """
    Create the directories required by the pipeline monitoring system.
    """

    required_directories = [
        OUTPUT_DIR,
        PIPELINE_OUTPUT_DIR,
        PIPELINE_LOG_DIR,
        PIPELINE_REPORT_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. STANDARD PIPELINE STATUS DEFINITIONS
# ============================================================

class PipelineStatus(str, Enum):
    """
    Standardized execution statuses for pipeline files and tasks.
    """

    PENDING = "PENDING"

    RUNNING = "RUNNING"

    SUCCESS = "SUCCESS"

    WARNING = "WARNING"

    FAILED = "FAILED"

    SKIPPED = "SKIPPED"

    CANCELLED = "CANCELLED"

    BLOCKED = "BLOCKED"


# ============================================================
# 4. STATUS COLORS FOR DASHBOARD DISPLAY
# ============================================================

PIPELINE_STATUS_COLORS = {
    PipelineStatus.PENDING.value: "#9CA3AF",

    PipelineStatus.RUNNING.value: "#3B82F6",

    PipelineStatus.SUCCESS.value: "#16A34A",

    PipelineStatus.WARNING.value: "#EAB308",

    PipelineStatus.FAILED.value: "#DC2626",

    PipelineStatus.SKIPPED.value: "#6B7280",

    PipelineStatus.CANCELLED.value: "#9333EA",

    PipelineStatus.BLOCKED.value: "#EA580C",
}


PIPELINE_STATUS_ICONS = {
    PipelineStatus.PENDING.value: "⚪",

    PipelineStatus.RUNNING.value: "🔵",

    PipelineStatus.SUCCESS.value: "🟢",

    PipelineStatus.WARNING.value: "🟡",

    PipelineStatus.FAILED.value: "🔴",

    PipelineStatus.SKIPPED.value: "⚪",

    PipelineStatus.CANCELLED.value: "🟣",

    PipelineStatus.BLOCKED.value: "🟠",
}


# ============================================================
# 5. PIPELINE SEVERITY LEVELS
# ============================================================

class PipelineSeverity(str, Enum):
    """
    Severity levels used for pipeline events and diagnostics.
    """

    INFO = "INFO"

    LOW = "LOW"

    MEDIUM = "MEDIUM"

    HIGH = "HIGH"

    CRITICAL = "CRITICAL"


# ============================================================
# 6. PIPELINE EXECUTION CONFIGURATION
# ============================================================

PIPELINE_CONFIG = {
    "project_name": "Ferry Demand Forecasting",

    "project_version": "1.0.0",

    "monitor_version": "1.0.0",

    "timezone": "UTC",

    "status_colors_enabled": True,

    "record_execution_history": True,

    "record_error_tracebacks": True,

    "record_execution_duration": True,

    "record_output_artifacts": True,

    "validate_output_artifacts": True,

    "persist_state_after_each_event": True,

    "maximum_error_message_length": 4000,

    "maximum_traceback_length": 12000,

    "maximum_history_records": 10000,

    "maximum_event_message_length": 2000,

    "fail_on_missing_required_output": True,
}


# ============================================================
# 7. THREAD-SAFE PIPELINE MONITORING LOCK
# ============================================================

_PIPELINE_LOCK = threading.RLock()


# ============================================================
# 8. PIPELINE RUNTIME CONTEXT
# ============================================================

PIPELINE_RUN_ID = str(uuid.uuid4())

PIPELINE_START_TIME = datetime.now(timezone.utc)

PIPELINE_PROCESS_ID = os.getpid()

PIPELINE_RUNTIME_CONTEXT = {
    "run_id": PIPELINE_RUN_ID,

    "project_name": PIPELINE_CONFIG["project_name"],

    "project_version": PIPELINE_CONFIG["project_version"],

    "monitor_version": PIPELINE_CONFIG["monitor_version"],

    "started_at": PIPELINE_START_TIME.isoformat(),

    "process_id": PIPELINE_PROCESS_ID,

    "python_version": sys.version.split()[0],

    "platform": platform.platform(),

    "working_directory": str(Path.cwd()),

    "project_directory": str(BASE_DIR),
}


# ============================================================
# 9. UTC TIMESTAMP HELPER
# ============================================================

def get_pipeline_timestamp() -> str:
    """
    Return a timezone-aware UTC timestamp in ISO 8601 format.
    """

    return datetime.now(timezone.utc).isoformat()


# ============================================================
# 10. SAFE JSON SERIALIZATION
# ============================================================

def pipeline_json_serializer(value: Any) -> Any:
    """
    Convert common Python objects into JSON-compatible values.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Exception):
        return str(value)

    if isinstance(value, set):
        return sorted(value, key=str)

    if isinstance(value, tuple):
        return list(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError, AttributeError):
            pass

    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass

    return str(value)


# ============================================================
# 11. SAFE STATUS NORMALIZATION
# ============================================================

def normalize_pipeline_status(
    status: Any,
) -> str:
    """
    Normalize status values into the supported pipeline vocabulary.
    """

    if isinstance(status, PipelineStatus):
        return status.value

    normalized_status = str(status).strip().upper()

    supported_statuses = {
        item.value for item in PipelineStatus
    }

    if normalized_status not in supported_statuses:
        raise ValueError(
            f"Unsupported pipeline status: {status!r}. "
            f"Expected one of: {sorted(supported_statuses)}"
        )

    return normalized_status


# ============================================================
# 12. SAFE SEVERITY NORMALIZATION
# ============================================================

def normalize_pipeline_severity(
    severity: Any,
) -> str:
    """
    Normalize event severity into a supported severity level.
    """

    if isinstance(severity, PipelineSeverity):
        return severity.value

    normalized_severity = str(severity).strip().upper()

    supported_severities = {
        item.value for item in PipelineSeverity
    }

    if normalized_severity not in supported_severities:
        raise ValueError(
            f"Unsupported severity: {severity!r}. "
            f"Expected one of: {sorted(supported_severities)}"
        )

    return normalized_severity


# ============================================================
# END OF PART 1
# NEXT: LOGGING ENGINE, PERSISTENT PIPELINE STATE & EVENT RECORDS
# ============================================================

# ============================================================
# PART 2: LOGGING ENGINE & PERSISTENT PIPELINE STATE
# ============================================================

# ============================================================
# 13. LOGGING ENGINE INITIALIZATION
# ============================================================

def initialize_pipeline_logging() -> logging.Logger:
    """
    Initialize console and file logging for pipeline monitoring.
    """

    initialize_pipeline_directories()

    logger = logging.getLogger("ferry_pipeline_monitor")

    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        PIPELINE_LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.info("Pipeline logging initialized.")

    logger.debug(
        "Pipeline runtime context: %s",
        json.dumps(
            PIPELINE_RUNTIME_CONTEXT,
            default=pipeline_json_serializer,
        ),
    )

    return logger


LOGGER = initialize_pipeline_logging()


# ============================================================
# 14. SAFE JSON FILE WRITER
# ============================================================

def write_pipeline_json(
    file_path: Path,
    data: Any,
) -> None:
    """
    Write JSON data atomically to reduce the risk of partial files.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = file_path.with_name(
        f".{file_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                default=pipeline_json_serializer,
            )

            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            file_path,
        )

    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

        raise


# ============================================================
# 15. APPEND-ONLY JSONL EVENT WRITER
# ============================================================

def append_pipeline_jsonl(
    file_path: Path,
    record: dict[str, Any],
) -> None:
    """
    Append one JSON record per line for execution history and events.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_record = json.dumps(
        record,
        ensure_ascii=False,
        default=pipeline_json_serializer,
    )

    with _PIPELINE_LOCK:
        with file_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(serialized_record + "\n")

            file.flush()


# ============================================================
# 16. PIPELINE EVENT RECORDING
# ============================================================

def record_pipeline_event(
    event_type: str,
    message: str,
    *,
    severity: str = PipelineSeverity.INFO.value,
    task_name: Optional[str] = None,
    file_name: Optional[str] = None,
    status: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Record a structured event for dashboard visibility and debugging.
    """

    normalized_severity = normalize_pipeline_severity(
        severity
    )

    normalized_status = (
        normalize_pipeline_status(status)
        if status is not None
        else None
    )

    maximum_message_length = int(
        PIPELINE_CONFIG["maximum_event_message_length"]
    )

    event_record = {
        "event_id": str(uuid.uuid4()),

        "run_id": PIPELINE_RUN_ID,

        "timestamp": get_pipeline_timestamp(),

        "event_type": str(event_type),

        "severity": normalized_severity,

        "message": str(message)[:maximum_message_length],

        "task_name": task_name,

        "file_name": file_name,

        "status": normalized_status,

        "details": details or {},
    }

    with _PIPELINE_LOCK:
        append_pipeline_jsonl(
            PIPELINE_EVENTS_FILE,
            event_record,
        )

    log_message = (
        f"{event_record['event_type']} | "
        f"{event_record['message']}"
    )

    if normalized_severity in {
        PipelineSeverity.CRITICAL.value,
        PipelineSeverity.HIGH.value,
    }:
        LOGGER.error(log_message)

    elif normalized_severity == PipelineSeverity.MEDIUM.value:
        LOGGER.warning(log_message)

    else:
        LOGGER.info(log_message)

    return event_record


# ============================================================
# 17. INITIAL PIPELINE STATE
# ============================================================

def create_initial_pipeline_state() -> dict[str, Any]:
    """
    Create the initial state for a new pipeline execution.
    """

    return {
        "run_id": PIPELINE_RUN_ID,

        "project": PIPELINE_CONFIG["project_name"],

        "project_version": PIPELINE_CONFIG["project_version"],

        "monitor_version": PIPELINE_CONFIG["monitor_version"],

        "status": PipelineStatus.PENDING.value,

        "started_at": PIPELINE_START_TIME.isoformat(),

        "updated_at": get_pipeline_timestamp(),

        "finished_at": None,

        "duration_seconds": None,

        "process_id": PIPELINE_PROCESS_ID,

        "environment": {
            "python_version": sys.version.split()[0],

            "platform": platform.platform(),
        },

        "summary": {
            "total_tasks": 0,

            "pending": 0,

            "running": 0,

            "success": 0,

            "warning": 0,

            "failed": 0,

            "skipped": 0,

            "cancelled": 0,

            "blocked": 0,
        },

        "tasks": {},

        "events": [],

        "errors": [],

        "artifacts": [],

        "metadata": {},
    }


# ============================================================
# 18. GLOBAL PIPELINE STATE
# ============================================================

PIPELINE_STATE: dict[str, Any] = (
    create_initial_pipeline_state()
)


# ============================================================
# 19. PERSIST PIPELINE STATE
# ============================================================

def persist_pipeline_state() -> None:
    """
    Persist the current pipeline state for later inspection.
    """

    with _PIPELINE_LOCK:
        PIPELINE_STATE["updated_at"] = (
            get_pipeline_timestamp()
        )

        state_snapshot = json.loads(
            json.dumps(
                PIPELINE_STATE,
                default=pipeline_json_serializer,
            )
        )

        write_pipeline_json(
            PIPELINE_STATE_FILE,
            state_snapshot,
        )


# ============================================================
# 20. RECALCULATE PIPELINE SUMMARY
# ============================================================

def recalculate_pipeline_summary() -> dict[str, int]:
    """
    Calculate task totals from the current registered task records.
    """

    status_counts = {
        status.value.lower(): 0
        for status in PipelineStatus
    }

    with _PIPELINE_LOCK:
        tasks = list(
            PIPELINE_STATE["tasks"].values()
        )

    for task in tasks:
        task_status = str(
            task.get(
                "status",
                PipelineStatus.PENDING.value,
            )
        ).lower()

        if task_status in status_counts:
            status_counts[task_status] += 1

    summary = {
        "total_tasks": len(tasks),

        **status_counts,
    }

    with _PIPELINE_LOCK:
        PIPELINE_STATE["summary"] = summary

    return summary


# ============================================================
# 21. RECORD PIPELINE ERROR
# ============================================================

def record_pipeline_error(
    *,
    task_name: str,
    file_name: str,
    exception: BaseException,
    phase: str = "EXECUTION",
    line_number: Optional[int] = None,
    additional_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Store a structured error record with its traceback and context.
    """

    error_id = str(uuid.uuid4())

    exception_message = str(exception)

    maximum_error_length = int(
        PIPELINE_CONFIG["maximum_error_message_length"]
    )

    maximum_traceback_length = int(
        PIPELINE_CONFIG["maximum_traceback_length"]
    )

    traceback_text = "".join(
        traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        )
    )

    error_record = {
        "error_id": error_id,

        "run_id": PIPELINE_RUN_ID,

        "timestamp": get_pipeline_timestamp(),

        "task_name": task_name,

        "file_name": file_name,

        "phase": phase,

        "exception_type": type(exception).__name__,

        "message": exception_message[:maximum_error_length],

        "line_number": line_number,

        "traceback": (
            traceback_text[:maximum_traceback_length]
            if PIPELINE_CONFIG["record_error_tracebacks"]
            else None
        ),

        "context": additional_context or {},
    }

    with _PIPELINE_LOCK:
        PIPELINE_STATE["errors"].append(
            error_record
        )

        append_pipeline_jsonl(
            PIPELINE_HISTORY_FILE,
            {
                "record_type": "ERROR",
                **error_record,
            },
        )

        write_pipeline_json(
            PIPELINE_ERROR_FILE,
            {
                "run_id": PIPELINE_RUN_ID,
                "updated_at": get_pipeline_timestamp(),
                "errors": PIPELINE_STATE["errors"],
            },
        )

    record_pipeline_event(
        "TASK_ERROR",
        f"{task_name}: {type(exception).__name__}: "
        f"{exception_message}",
        severity=PipelineSeverity.HIGH.value,
        task_name=task_name,
        file_name=file_name,
        status=PipelineStatus.FAILED.value,
        details={
            "error_id": error_id,
            "phase": phase,
            "line_number": line_number,
        },
    )

    return error_record


# ============================================================
# END OF PART 2
# NEXT: TASK REGISTRATION, EXECUTION TRACKING & ARTIFACT VALIDATION
# ============================================================
# ============================================================
# PART 3: TASK REGISTRATION, EXECUTION TRACKING & VALIDATION
# ============================================================

# ============================================================
# 22. REGISTER A PIPELINE TASK
# ============================================================

def register_pipeline_task(
    task_name: str,
    file_name: str,
    *,
    description: str = "",
    dependencies: Optional[list[str]] = None,
    required_outputs: Optional[list[str | Path]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Register a pipeline task before executing it.
    """

    if not task_name or not task_name.strip():
        raise ValueError("Task name cannot be empty.")

    if not file_name or not file_name.strip():
        raise ValueError("File name cannot be empty.")

    normalized_task_name = task_name.strip()

    normalized_file_name = file_name.strip()

    task_record = {
        "task_id": str(uuid.uuid4()),

        "task_name": normalized_task_name,

        "file_name": normalized_file_name,

        "description": description,

        "status": PipelineStatus.PENDING.value,

        "registered_at": get_pipeline_timestamp(),

        "started_at": None,

        "finished_at": None,

        "duration_seconds": None,

        "dependencies": list(dependencies or []),

        "required_outputs": [
            str(path)
            for path in (required_outputs or [])
        ],

        "generated_outputs": [],

        "error_id": None,

        "error_message": None,

        "warning_messages": [],

        "metadata": metadata or {},
    }

    with _PIPELINE_LOCK:
        if normalized_task_name in PIPELINE_STATE["tasks"]:
            raise ValueError(
                f"Pipeline task already registered: "
                f"{normalized_task_name}"
            )

        PIPELINE_STATE["tasks"][
            normalized_task_name
        ] = task_record

    recalculate_pipeline_summary()

    persist_pipeline_state()

    record_pipeline_event(
        "TASK_REGISTERED",
        f"Registered pipeline task: {normalized_task_name}",
        task_name=normalized_task_name,
        file_name=normalized_file_name,
        status=PipelineStatus.PENDING.value,
    )

    return task_record


# ============================================================
# 23. GET A REGISTERED TASK
# ============================================================

def get_pipeline_task(
    task_name: str,
) -> Optional[dict[str, Any]]:
    """
    Return a copy of a registered task record.
    """

    with _PIPELINE_LOCK:
        task = PIPELINE_STATE["tasks"].get(task_name)

        if task is None:
            return None

        return json.loads(
            json.dumps(
                task,
                default=pipeline_json_serializer,
            )
        )


# ============================================================
# 24. UPDATE TASK STATUS
# ============================================================

def update_pipeline_task_status(
    task_name: str,
    status: str | PipelineStatus,
    *,
    message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Update a task status and persist the change.
    """

    normalized_status = normalize_pipeline_status(
        status
    )

    with _PIPELINE_LOCK:
        task = PIPELINE_STATE["tasks"].get(task_name)

        if task is None:
            raise KeyError(
                f"Pipeline task is not registered: {task_name}"
            )

        previous_status = task["status"]

        task["status"] = normalized_status

        task["updated_at"] = get_pipeline_timestamp()

        if message:
            task["status_message"] = str(message)

        if metadata:
            task["metadata"].update(metadata)

        if normalized_status == PipelineStatus.RUNNING.value:
            if task["started_at"] is None:
                task["started_at"] = get_pipeline_timestamp()

        if normalized_status in {
            PipelineStatus.SUCCESS.value,
            PipelineStatus.WARNING.value,
            PipelineStatus.FAILED.value,
            PipelineStatus.SKIPPED.value,
            PipelineStatus.CANCELLED.value,
            PipelineStatus.BLOCKED.value,
        }:
            task["finished_at"] = get_pipeline_timestamp()

            if task["started_at"]:
                start_time = datetime.fromisoformat(
                    task["started_at"]
                )

                finish_time = datetime.fromisoformat(
                    task["finished_at"]
                )

                task["duration_seconds"] = max(
                    0.0,
                    (finish_time - start_time).total_seconds(),
                )

        task_snapshot = json.loads(
            json.dumps(
                task,
                default=pipeline_json_serializer,
            )
        )

    recalculate_pipeline_summary()

    persist_pipeline_state()

    record_pipeline_event(
        "TASK_STATUS_CHANGED",
        (
            f"{task_name}: {previous_status} "
            f"-> {normalized_status}"
        ),
        task_name=task_name,
        file_name=task_snapshot["file_name"],
        status=normalized_status,
        details={
            "previous_status": previous_status,
            "message": message,
        },
    )

    return task_snapshot


# ============================================================
# 25. REGISTER A TASK WARNING
# ============================================================

def add_pipeline_warning(
    task_name: str,
    warning_message: str,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Attach a warning to a task without automatically failing it.
    """

    if not warning_message or not warning_message.strip():
        raise ValueError(
            "Warning message cannot be empty."
        )

    with _PIPELINE_LOCK:
        task = PIPELINE_STATE["tasks"].get(task_name)

        if task is None:
            raise KeyError(
                f"Pipeline task is not registered: {task_name}"
            )

        if task["status"] in {
            PipelineStatus.FAILED.value,
            PipelineStatus.CANCELLED.value,
            PipelineStatus.BLOCKED.value,
        }:
            raise RuntimeError(
                f"Cannot add a warning to a task in "
                f"{task['status']} status."
            )

        warning_record = {
            "timestamp": get_pipeline_timestamp(),

            "message": warning_message.strip(),

            "metadata": metadata or {},
        }

        task["warning_messages"].append(
            warning_record
        )

        task["updated_at"] = get_pipeline_timestamp()

        if task["status"] == PipelineStatus.SUCCESS.value:
            task["status"] = PipelineStatus.WARNING.value

        task_snapshot = json.loads(
            json.dumps(
                task,
                default=pipeline_json_serializer,
            )
        )

    recalculate_pipeline_summary()

    persist_pipeline_state()

    record_pipeline_event(
        "TASK_WARNING",
        warning_message,
        severity=PipelineSeverity.MEDIUM.value,
        task_name=task_name,
        file_name=task_snapshot["file_name"],
        status=task_snapshot["status"],
        details=metadata or {},
    )

    return task_snapshot


# ============================================================
# 26. VALIDATE REQUIRED OUTPUT FILES
# ============================================================

def validate_pipeline_outputs(
    task_name: str,
    *,
    additional_outputs: Optional[list[str | Path]] = None,
) -> dict[str, Any]:
    """
    Check whether the registered task's required output files exist.
    """

    task = get_pipeline_task(task_name)

    if task is None:
        raise KeyError(
            f"Pipeline task is not registered: {task_name}"
        )

    required_outputs = [
        Path(path)
        for path in task["required_outputs"]
    ]

    required_outputs.extend(
        Path(path)
        for path in (additional_outputs or [])
    )

    unique_outputs = list(
        dict.fromkeys(required_outputs)
    )

    output_results = []

    for output_path in unique_outputs:
        resolved_path = (
            output_path
            if output_path.is_absolute()
            else BASE_DIR / output_path
        )

        exists = resolved_path.is_file()

        file_size = (
            resolved_path.stat().st_size
            if exists
            else None
        )

        output_results.append(
            {
                "path": str(resolved_path),

                "exists": exists,

                "size_bytes": file_size,

                "valid": exists and file_size > 0,
            }
        )

    missing_outputs = [
        item["path"]
        for item in output_results
        if not item["exists"]
    ]

    invalid_outputs = [
        item["path"]
        for item in output_results
        if item["exists"] and not item["valid"]
    ]

    validation_result = {
        "task_name": task_name,

        "checked_at": get_pipeline_timestamp(),

        "total_outputs": len(output_results),

        "all_valid": (
            not missing_outputs
            and not invalid_outputs
        ),

        "outputs": output_results,

        "missing_outputs": missing_outputs,

        "invalid_outputs": invalid_outputs,
    }

    with _PIPELINE_LOCK:
        task_record = PIPELINE_STATE["tasks"][task_name]

        task_record["output_validation"] = (
            validation_result
        )

        task_record["generated_outputs"] = [
            item["path"]
            for item in output_results
            if item["valid"]
        ]

        task_record["updated_at"] = (
            get_pipeline_timestamp()
        )

    persist_pipeline_state()

    record_pipeline_event(
        "OUTPUT_VALIDATION",
        (
            f"Validated outputs for {task_name}: "
            f"{'PASS' if validation_result['all_valid'] else 'FAIL'}"
        ),
        severity=(
            PipelineSeverity.INFO.value
            if validation_result["all_valid"]
            else PipelineSeverity.HIGH.value
        ),
        task_name=task_name,
        file_name=task["file_name"],
        details=validation_result,
    )

    return validation_result


# ============================================================
# 27. CHECK TASK DEPENDENCIES
# ============================================================

def check_pipeline_dependencies(
    task_name: str,
) -> dict[str, Any]:
    """
    Check whether all dependencies have completed successfully.
    """

    task = get_pipeline_task(task_name)

    if task is None:
        raise KeyError(
            f"Pipeline task is not registered: {task_name}"
        )

    dependency_results = []

    for dependency_name in task["dependencies"]:
        dependency = get_pipeline_task(
            dependency_name
        )

        if dependency is None:
            dependency_results.append(
                {
                    "task_name": dependency_name,

                    "status": "NOT_REGISTERED",

                    "satisfied": False,
                }
            )

            continue

        dependency_status = dependency["status"]

        dependency_satisfied = (
            dependency_status
            == PipelineStatus.SUCCESS.value
        )

        dependency_results.append(
            {
                "task_name": dependency_name,

                "status": dependency_status,

                "satisfied": dependency_satisfied,
            }
        )

    unsatisfied_dependencies = [
        result
        for result in dependency_results
        if not result["satisfied"]
    ]

    return {
        "task_name": task_name,

        "checked_at": get_pipeline_timestamp(),

        "dependencies_satisfied": (
            len(unsatisfied_dependencies) == 0
        ),

        "total_dependencies": len(dependency_results),

        "unsatisfied_dependencies": (
            unsatisfied_dependencies
        ),

        "dependency_results": dependency_results,
    }


# ============================================================
# END OF PART 3
# NEXT: TASK EXECUTION CONTEXT, ERROR CAPTURE & AUTO STATUS
# ============================================================
# ============================================================
# PART 4: TASK EXECUTION CONTEXT & AUTOMATIC ERROR CAPTURE
# ============================================================

# ============================================================
# 28. EXTRACT ERROR LINE NUMBER
# ============================================================

def extract_exception_line_number(
    exception: BaseException,
) -> Optional[int]:
    """
    Extract the most relevant traceback line number.
    """

    traceback_object = exception.__traceback__

    if traceback_object is None:
        return None

    traceback_entries = traceback.extract_tb(
        traceback_object
    )

    if not traceback_entries:
        return None

    return traceback_entries[-1].lineno


# ============================================================
# 29. EXECUTION RESULT CONTAINER
# ============================================================

@dataclass
class PipelineExecutionResult:
    """
    Structured result returned after executing a pipeline task.
    """

    task_name: str

    file_name: str

    status: str

    started_at: str

    finished_at: str

    duration_seconds: float

    success: bool

    warnings: list[str] = field(default_factory=list)

    error_message: Optional[str] = None

    error_type: Optional[str] = None

    error_line_number: Optional[int] = None

    generated_outputs: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the execution result into a dictionary.
        """

        return asdict(self)


# ============================================================
# 30. TASK EXECUTION CONTEXT MANAGER
# ============================================================

@contextmanager
def track_pipeline_task(
    task_name: str,
    *,
    validate_outputs: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Track a task's lifecycle, execution duration and exceptions.

    Usage:

        with track_pipeline_task("dataset_analysis") as context:
            # Execute the task here.
            pass
    """

    task = get_pipeline_task(task_name)

    if task is None:
        raise KeyError(
            f"Pipeline task is not registered: {task_name}"
        )

    dependency_check = check_pipeline_dependencies(
        task_name
    )

    if not dependency_check["dependencies_satisfied"]:
        unsatisfied_names = [
            item["task_name"]
            for item in dependency_check[
                "unsatisfied_dependencies"
            ]
        ]

        update_pipeline_task_status(
            task_name,
            PipelineStatus.BLOCKED,
            message=(
                "Required dependencies are incomplete: "
                + ", ".join(unsatisfied_names)
            ),
        )

        raise RuntimeError(
            f"Cannot execute {task_name}. "
            f"Unsatisfied dependencies: {unsatisfied_names}"
        )

    started_at = get_pipeline_timestamp()

    start_clock = time.perf_counter()

    execution_context: dict[str, Any] = {
        "task_name": task_name,

        "file_name": task["file_name"],

        "started_at": started_at,

        "status": PipelineStatus.RUNNING.value,

        "warnings": [],

        "metadata": {},

        "generated_outputs": [],

        "error": None,
    }

    update_pipeline_task_status(
        task_name,
        PipelineStatus.RUNNING,
        message="Task execution started.",
    )

    record_pipeline_event(
        "TASK_EXECUTION_STARTED",
        f"Execution started for {task_name}.",
        task_name=task_name,
        file_name=task["file_name"],
        status=PipelineStatus.RUNNING,
    )

    try:
        yield execution_context

        if validate_outputs:
            output_validation = validate_pipeline_outputs(
                task_name
            )

            execution_context["generated_outputs"] = [
                item["path"]
                for item in output_validation["outputs"]
                if item["valid"]
            ]

            if not output_validation["all_valid"]:
                missing_outputs = (
                    output_validation["missing_outputs"]
                )

                invalid_outputs = (
                    output_validation["invalid_outputs"]
                )

                output_error = (
                    "Required output validation failed. "
                    f"Missing: {missing_outputs}. "
                    f"Invalid or empty: {invalid_outputs}."
                )

                if PIPELINE_CONFIG[
                    "fail_on_missing_required_output"
                ]:
                    raise RuntimeError(output_error)

                execution_context["warnings"].append(
                    output_error
                )

        task_warnings = execution_context["warnings"]

        final_status = (
            PipelineStatus.WARNING.value
            if task_warnings
            else PipelineStatus.SUCCESS.value
        )

        execution_context["status"] = final_status

        update_pipeline_task_status(
            task_name,
            final_status,
            message=(
                "Task completed with warnings."
                if task_warnings
                else "Task completed successfully."
            ),
            metadata=execution_context["metadata"],
        )

        for warning_message in task_warnings:
            add_pipeline_warning(
                task_name,
                warning_message,
            )

    except Exception as exception:
        error_line_number = extract_exception_line_number(
            exception
        )

        error_record = record_pipeline_error(
            task_name=task_name,
            file_name=task["file_name"],
            exception=exception,
            phase="TASK_EXECUTION",
            line_number=error_line_number,
            additional_context={
                "started_at": started_at,
            },
        )

        execution_context["error"] = error_record

        execution_context["status"] = (
            PipelineStatus.FAILED.value
        )

        try:
            with _PIPELINE_LOCK:
                current_task = PIPELINE_STATE[
                    "tasks"
                ].get(task_name)

                if current_task is not None:
                    current_task["error_id"] = (
                        error_record["error_id"]
                    )

                    current_task["error_message"] = (
                        error_record["message"]
                    )

            update_pipeline_task_status(
                task_name,
                PipelineStatus.FAILED,
                message=str(exception),
            )

        except Exception:
            LOGGER.exception(
                "Unable to persist failure status for %s.",
                task_name,
            )

        raise

    finally:
        finished_at = get_pipeline_timestamp()

        duration_seconds = round(
            time.perf_counter() - start_clock,
            4,
        )

        execution_context["finished_at"] = finished_at

        execution_context["duration_seconds"] = (
            duration_seconds
        )

        with _PIPELINE_LOCK:
            current_task = PIPELINE_STATE[
                "tasks"
            ].get(task_name)

            if current_task is not None:
                current_task["execution_context"] = {
                    "started_at": started_at,

                    "finished_at": finished_at,

                    "duration_seconds": duration_seconds,
                }

        persist_pipeline_state()

        append_pipeline_jsonl(
            PIPELINE_HISTORY_FILE,
            {
                "record_type": "TASK_EXECUTION",
                "run_id": PIPELINE_RUN_ID,
                "task_name": task_name,
                "file_name": task["file_name"],
                "status": execution_context["status"],
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_seconds,
                "error": execution_context["error"],
                "warnings": execution_context["warnings"],
            },
        )

        record_pipeline_event(
            "TASK_EXECUTION_FINISHED",
            (
                f"{task_name} finished with status "
                f"{execution_context['status']} "
                f"in {duration_seconds:.4f} seconds."
            ),
            severity=(
                PipelineSeverity.HIGH.value
                if execution_context["status"]
                == PipelineStatus.FAILED.value
                else PipelineSeverity.INFO.value
            ),
            task_name=task_name,
            file_name=task["file_name"],
            status=execution_context["status"],
            details={
                "duration_seconds": duration_seconds,
            },
        )


# ============================================================
# 31. EXECUTE A TASK THROUGH A CALLBACK
# ============================================================

def execute_pipeline_task(
    task_name: str,
    task_function: Callable[..., Any],
    *,
    validate_outputs: bool = True,
    task_kwargs: Optional[dict[str, Any]] = None,
) -> PipelineExecutionResult:
    """
    Execute a callable while capturing task status and exceptions.
    """

    task = get_pipeline_task(task_name)

    if task is None:
        raise KeyError(
            f"Pipeline task is not registered: {task_name}"
        )

    started_at = get_pipeline_timestamp()

    start_clock = time.perf_counter()

    warnings: list[str] = []

    generated_outputs: list[str] = []

    metadata: dict[str, Any] = {}

    error_message = None

    error_type = None

    error_line_number = None

    execution_status = PipelineStatus.SUCCESS.value

    try:
        with track_pipeline_task(
            task_name,
            validate_outputs=validate_outputs,
        ) as context:

            result = task_function(
                **(task_kwargs or {})
            )

            if isinstance(result, dict):
                metadata["result_summary"] = {
                    str(key): value
                    for key, value in result.items()
                    if isinstance(
                        value,
                        (str, int, float, bool, type(None)),
                    )
                }

            warnings = context["warnings"]

            generated_outputs = context[
                "generated_outputs"
            ]

            metadata.update(
                context["metadata"]
            )

            execution_status = context["status"]

    except Exception as exception:
        execution_status = PipelineStatus.FAILED.value

        error_message = str(exception)

        error_type = type(exception).__name__

        error_line_number = extract_exception_line_number(
            exception
        )

    finished_at = get_pipeline_timestamp()

    duration_seconds = round(
        time.perf_counter() - start_clock,
        4,
    )

    return PipelineExecutionResult(
        task_name=task_name,
        file_name=task["file_name"],
        status=execution_status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        success=(
            execution_status
            == PipelineStatus.SUCCESS.value
        ),
        warnings=warnings,
        error_message=error_message,
        error_type=error_type,
        error_line_number=error_line_number,
        generated_outputs=generated_outputs,
        metadata=metadata,
    )


# ============================================================
# END OF PART 4
# NEXT: PIPELINE REGISTRY, EXECUTION HISTORY & DASHBOARD DATA
# ============================================================
# ============================================================
# PART 5: PIPELINE REGISTRY, EXECUTION HISTORY & DASHBOARD DATA
# ============================================================

# ============================================================
# 32. REGISTER THE COMPLETE PROJECT PIPELINE
# ============================================================

def register_default_pipeline() -> dict[str, Any]:
    """
    Register the major stages of the ferry demand forecasting pipeline.

    Task dependencies represent execution order. Update these task
    definitions when the actual project architecture is finalized.
    """

    pipeline_definitions = [
        {
            "task_name": "dataset_validation",
            "file_name": "check_dataset.py",
            "description": "Validate the input dataset and its structure.",
            "dependencies": [],
            "required_outputs": [],
        },
        {
            "task_name": "dataset_analysis",
            "file_name": "analyze_dataset.py",
            "description": "Analyze dataset quality, demand patterns and forecasting readiness.",
            "dependencies": ["dataset_validation"],
            "required_outputs": [],
        },
        {
            "task_name": "feature_engineering",
            "file_name": "feature_engineering.py",
            "description": "Prepare forecasting features and target series.",
            "dependencies": ["dataset_analysis"],
            "required_outputs": [],
        },
        {
            "task_name": "anomaly_detection",
            "file_name": "anomaly_detection.py",
            "description": "Detect unusual patterns in the available observations.",
            "dependencies": ["feature_engineering"],
            "required_outputs": [],
        },
        {
            "task_name": "forecasting",
            "file_name": "forecasting.py",
            "description": "Generate short-term ferry ticket demand forecasts.",
            "dependencies": ["feature_engineering"],
            "required_outputs": [],
        },
        {
            "task_name": "model_evaluation",
            "file_name": "model_evaluation.py",
            "description": "Evaluate forecasts using time-aware validation.",
            "dependencies": ["forecasting"],
            "required_outputs": [],
        },
        {
            "task_name": "forecast_monitoring",
            "file_name": "forecast_monitoring.py",
            "description": "Monitor forecast errors and performance changes.",
            "dependencies": ["model_evaluation"],
            "required_outputs": [],
        },
        {
            "task_name": "report_generation",
            "file_name": "report_generation.py",
            "description": "Generate final reports and exportable artifacts.",
            "dependencies": ["model_evaluation"],
            "required_outputs": [],
        },
    ]

    registered_tasks = []

    for definition in pipeline_definitions:
        task_name = definition["task_name"]

        existing_task = get_pipeline_task(task_name)

        if existing_task is not None:
            registered_tasks.append(existing_task)
            continue

        task_record = register_pipeline_task(
            task_name=task_name,
            file_name=definition["file_name"],
            description=definition["description"],
            dependencies=definition["dependencies"],
            required_outputs=definition["required_outputs"],
        )

        registered_tasks.append(task_record)

    return {
        "registered_count": len(registered_tasks),
        "tasks": registered_tasks,
    }


# ============================================================
# 33. GET PIPELINE TASK STATUS TABLE DATA
# ============================================================

def get_pipeline_status_table() -> list[dict[str, Any]]:
    """
    Return task records in a dashboard-friendly order.
    """

    with _PIPELINE_LOCK:
        tasks = list(
            PIPELINE_STATE["tasks"].values()
        )

    tasks.sort(
        key=lambda task: task.get(
            "registered_at",
            "",
        )
    )

    table_rows = []

    for task in tasks:
        status = task.get(
            "status",
            PipelineStatus.PENDING.value,
        )

        table_rows.append(
            {
                "Task": task["task_name"],

                "File": task["file_name"],

                "Status": status,

                "Status Icon": PIPELINE_STATUS_ICONS.get(
                    status,
                    "⚪",
                ),

                "Duration (seconds)": task.get(
                    "duration_seconds"
                ),

                "Started At": task.get(
                    "started_at"
                ),

                "Finished At": task.get(
                    "finished_at"
                ),

                "Dependencies": ", ".join(
                    task.get("dependencies", [])
                ),

                "Warnings": len(
                    task.get("warning_messages", [])
                ),

                "Error": task.get(
                    "error_message"
                ),
            }
        )

    return table_rows


# ============================================================
# 34. GET PIPELINE EXECUTION SUMMARY
# ============================================================

def get_pipeline_execution_summary() -> dict[str, Any]:
    """
    Calculate high-level execution statistics for the dashboard.
    """

    summary = recalculate_pipeline_summary()

    with _PIPELINE_LOCK:
        tasks = list(
            PIPELINE_STATE["tasks"].values()
        )

    completed_statuses = {
        PipelineStatus.SUCCESS.value,
        PipelineStatus.WARNING.value,
        PipelineStatus.FAILED.value,
        PipelineStatus.SKIPPED.value,
        PipelineStatus.CANCELLED.value,
        PipelineStatus.BLOCKED.value,
    }

    completed_tasks = [
        task
        for task in tasks
        if task.get("status") in completed_statuses
    ]

    total_duration = sum(
        float(task.get("duration_seconds") or 0.0)
        for task in completed_tasks
    )

    successful_tasks = summary.get("success", 0)

    warning_tasks = summary.get("warning", 0)

    failed_tasks = summary.get("failed", 0)

    total_tasks = summary.get("total_tasks", 0)

    completed_count = len(completed_tasks)

    completion_percentage = (
        round(
            completed_count / total_tasks * 100,
            2,
        )
        if total_tasks
        else 0.0
    )

    success_percentage = (
        round(
            successful_tasks / total_tasks * 100,
            2,
        )
        if total_tasks
        else 0.0
    )

    if failed_tasks > 0:
        overall_status = PipelineStatus.FAILED.value

    elif summary.get("blocked", 0) > 0:
        overall_status = PipelineStatus.BLOCKED.value

    elif warning_tasks > 0:
        overall_status = PipelineStatus.WARNING.value

    elif summary.get("running", 0) > 0:
        overall_status = PipelineStatus.RUNNING.value

    elif total_tasks > 0 and completed_count == total_tasks:
        overall_status = PipelineStatus.SUCCESS.value

    else:
        overall_status = PipelineStatus.PENDING.value

    return {
        "run_id": PIPELINE_RUN_ID,

        "project_name": PIPELINE_CONFIG["project_name"],

        "overall_status": overall_status,

        "overall_status_color": PIPELINE_STATUS_COLORS.get(
            overall_status,
            "#9CA3AF",
        ),

        "overall_status_icon": PIPELINE_STATUS_ICONS.get(
            overall_status,
            "⚪",
        ),

        "total_tasks": total_tasks,

        "completed_tasks": completed_count,

        "completion_percentage": completion_percentage,

        "success_percentage": success_percentage,

        "successful_tasks": successful_tasks,

        "warning_tasks": warning_tasks,

        "failed_tasks": failed_tasks,

        "pending_tasks": summary.get("pending", 0),

        "running_tasks": summary.get("running", 0),

        "skipped_tasks": summary.get("skipped", 0),

        "cancelled_tasks": summary.get("cancelled", 0),

        "blocked_tasks": summary.get("blocked", 0),

        "total_recorded_task_duration_seconds": round(
            total_duration,
            4,
        ),

        "generated_at": get_pipeline_timestamp(),
    }


# ============================================================
# 35. GET RECENT PIPELINE EVENTS
# ============================================================

def get_recent_pipeline_events(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return recent events from the current pipeline run.
    """

    if limit < 1:
        raise ValueError(
            "Event limit must be greater than zero."
        )

    if not PIPELINE_EVENTS_FILE.exists():
        return []

    recent_events = []

    try:
        with PIPELINE_EVENTS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    event = json.loads(
                        stripped_line
                    )

                    if event.get("run_id") == PIPELINE_RUN_ID:
                        recent_events.append(event)

                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Ignoring malformed pipeline event record."
                    )

    except OSError:
        LOGGER.exception(
            "Unable to read pipeline event history."
        )

        return []

    return recent_events[-limit:][::-1]


# ============================================================
# 36. GET PIPELINE ERROR HISTORY
# ============================================================

def get_pipeline_error_history() -> list[dict[str, Any]]:
    """
    Return errors recorded during the current pipeline run.
    """

    with _PIPELINE_LOCK:
        errors = list(
            PIPELINE_STATE.get("errors", [])
        )

    return errors[::-1]


# ============================================================
# 37. GENERATE PIPELINE SUMMARY REPORT
# ============================================================

def generate_pipeline_summary_report() -> dict[str, Any]:
    """
    Generate and persist a summary of the current pipeline execution.
    """

    execution_summary = get_pipeline_execution_summary()

    status_table = get_pipeline_status_table()

    error_history = get_pipeline_error_history()

    recent_events = get_recent_pipeline_events(
        limit=100
    )

    report = {
        "report_generated_at": get_pipeline_timestamp(),

        "runtime_context": PIPELINE_RUNTIME_CONTEXT,

        "execution_summary": execution_summary,

        "task_status_table": status_table,

        "error_count": len(error_history),

        "errors": error_history,

        "recent_events": recent_events,
    }

    with _PIPELINE_LOCK:
        PIPELINE_STATE["summary_report_generated_at"] = (
            report["report_generated_at"]
        )

    write_pipeline_json(
        PIPELINE_SUMMARY_FILE,
        report,
    )

    persist_pipeline_state()

    return report


# ============================================================
# END OF PART 5
# NEXT: PIPELINE RECOVERY, HISTORY LOADING & FINAL REPORTS
# ============================================================
# ============================================================
# PART 6: PIPELINE RECOVERY, HISTORY LOADING & FINAL REPORTS
# ============================================================

# ============================================================
# 38. LOAD EXISTING PIPELINE STATE SAFELY
# ============================================================

def load_pipeline_state() -> dict[str, Any]:
    """
    Load a previously saved pipeline state when available.

    Invalid or unreadable state files do not crash the application.
    The current in-memory state remains available as a fallback.
    """

    global PIPELINE_STATE

    if not PIPELINE_STATE_FILE.exists():
        LOGGER.info(
            "No previous pipeline state found. Using current state."
        )

        return PIPELINE_STATE

    try:
        with PIPELINE_STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved_state = json.load(file)

        if not isinstance(saved_state, dict):
            raise ValueError(
                "Saved pipeline state must be a JSON object."
            )

        if not isinstance(
            saved_state.get("tasks", {}),
            dict,
        ):
            raise ValueError(
                "Saved pipeline tasks must be a dictionary."
            )

        if not isinstance(
            saved_state.get("errors", []),
            list,
        ):
            raise ValueError(
                "Saved pipeline errors must be a list."
            )

        with _PIPELINE_LOCK:
            PIPELINE_STATE.update(saved_state)

        LOGGER.info(
            "Previous pipeline state loaded successfully."
        )

        return PIPELINE_STATE

    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        LOGGER.warning(
            "Unable to load previous pipeline state: %s",
            exc,
        )

        return PIPELINE_STATE


# ============================================================
# 39. RECOVER INTERRUPTED TASKS
# ============================================================

def recover_interrupted_tasks() -> dict[str, int]:
    """
    Mark tasks left RUNNING by an interrupted process.

    A task that was running in an earlier process cannot safely
    be assumed to have completed successfully.
    """

    recovered_count = 0

    with _PIPELINE_LOCK:
        tasks = list(
            PIPELINE_STATE.get(
                "tasks",
                {},
            ).items()
        )

    for task_name, task in tasks:
        if task.get("status") != PipelineStatus.RUNNING.value:
            continue

        task["status"] = PipelineStatus.WARNING.value

        task["finished_at"] = get_pipeline_timestamp()

        task["error_message"] = (
            "The previous execution may have been interrupted. "
            "Completion was not verified."
        )

        task.setdefault(
            "warning_messages",
            [],
        ).append(
            "Interrupted execution detected during recovery."
        )

        with _PIPELINE_LOCK:
            PIPELINE_STATE["tasks"][task_name] = task

        recovered_count += 1

        record_pipeline_event(
            event_type="TASK_RECOVERED",
            task_name=task_name,
            status=PipelineStatus.WARNING.value,
            message=(
                "Recovered an unfinished task from a previous execution."
            ),
            severity=PipelineSeverity.MEDIUM.value,
        )

    if recovered_count:
        persist_pipeline_state()

        LOGGER.warning(
            "Recovered %s interrupted pipeline task(s).",
            recovered_count,
        )

    return {
        "recovered_tasks": recovered_count,
    }


# ============================================================
# 40. READ PIPELINE HISTORY
# ============================================================

def load_pipeline_history(
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Read the latest saved pipeline execution history records.
    """

    if limit < 1:
        raise ValueError(
            "History limit must be greater than zero."
        )

    if not PIPELINE_HISTORY_FILE.exists():
        return []

    history_records = []

    try:
        with PIPELINE_HISTORY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                    if isinstance(record, dict):
                        history_records.append(record)

                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Ignoring malformed pipeline history record."
                    )

    except OSError:
        LOGGER.exception(
            "Unable to read pipeline execution history."
        )

        return []

    return history_records[-limit:][::-1]


# ============================================================
# 41. PIPELINE HEALTH CHECK
# ============================================================

def run_pipeline_health_check() -> dict[str, Any]:
    """
    Check whether monitoring directories and essential files
    are accessible.

    This check reports monitoring infrastructure health.
    It does not execute forecasting or machine-learning tasks.
    """

    checks = []

    # Check output directories.
    for directory_path in (
        PIPELINE_OUTPUT_DIR,
        PIPELINE_LOG_DIR,
        PIPELINE_REPORT_DIR,
    ):
        try:
            directory_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not directory_path.is_dir():
                raise OSError(
                    "Expected a directory but found another file type."
                )

            checks.append(
                {
                    "check": f"Directory: {directory_path.name}",
                    "status": "SUCCESS",
                    "message": "Directory is available.",
                }
            )

        except OSError as exc:
            checks.append(
                {
                    "check": f"Directory: {directory_path.name}",
                    "status": "FAILED",
                    "message": str(exc),
                }
            )

    # Check whether the configured input dataset exists.
    dataset_path = BASE_DIR / "data" / "Toronto Island Ferry Tickets (1).csv"

    if dataset_path.is_file():
        checks.append(
            {
                "check": "Input dataset",
                "status": "SUCCESS",
                "message": "Configured dataset file exists.",
            }
        )
    else:
        checks.append(
            {
                "check": "Input dataset",
                "status": "WARNING",
                "message": (
                    "Configured dataset file was not found. "
                    "Confirm its path and filename."
                ),
            }
        )

    failed_checks = sum(
        check["status"] == "FAILED"
        for check in checks
    )

    warning_checks = sum(
        check["status"] == "WARNING"
        for check in checks
    )

    if failed_checks:
        overall_status = PipelineStatus.FAILED.value

    elif warning_checks:
        overall_status = PipelineStatus.WARNING.value

    else:
        overall_status = PipelineStatus.SUCCESS.value

    return {
        "checked_at": get_pipeline_timestamp(),

        "overall_status": overall_status,

        "overall_status_color": PIPELINE_STATUS_COLORS.get(
            overall_status,
            "#9CA3AF",
        ),

        "total_checks": len(checks),

        "successful_checks": sum(
            check["status"] == "SUCCESS"
            for check in checks
        ),

        "warning_checks": warning_checks,

        "failed_checks": failed_checks,

        "checks": checks,
    }


# ============================================================
# 42. EXPORT PIPELINE REPORT AS CSV
# ============================================================

def export_pipeline_status_csv() -> Path:
    """
    Export current task statuses to a CSV file.

    Uses the standard library so this monitoring module does not
    require pandas solely for exporting the status table.
    """

    import csv

    csv_path = PIPELINE_REPORT_DIR / (
        f"pipeline_status_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )

    rows = get_pipeline_status_table()

    fieldnames = [
        "Task",
        "File",
        "Status",
        "Status Icon",
        "Duration (seconds)",
        "Started At",
        "Finished At",
        "Dependencies",
        "Warnings",
        "Error",
    ]

    try:
        PIPELINE_REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

        LOGGER.info(
            "Pipeline status CSV exported: %s",
            csv_path,
        )

        record_pipeline_event(
            event_type="REPORT_EXPORTED",
            task_name="pipeline_monitor",
            status=PipelineStatus.SUCCESS.value,
            message="Pipeline status CSV exported successfully.",
            severity=PipelineSeverity.INFO.value,
            metadata={
                "file_path": str(csv_path),
            },
        )

        return csv_path

    except OSError as exc:
        LOGGER.exception(
            "Failed to export pipeline status CSV."
        )

        record_pipeline_error(
            task_name="pipeline_monitor",
            exception=exc,
            severity=PipelineSeverity.HIGH.value,
        )

        raise


# ============================================================
# 43. FINALIZE THE CURRENT PIPELINE RUN
# ============================================================

def finalize_pipeline_run() -> dict[str, Any]:
    """
    Generate the final report and mark the current run as finalized.
    """

    report = generate_pipeline_summary_report()

    summary = report["execution_summary"]

    with _PIPELINE_LOCK:
        PIPELINE_STATE["run_finalized_at"] = (
            get_pipeline_timestamp()
        )

        PIPELINE_STATE["run_final_status"] = (
            summary["overall_status"]
        )

    persist_pipeline_state()

    record_pipeline_event(
        event_type="PIPELINE_FINALIZED",
        task_name="pipeline_monitor",
        status=summary["overall_status"],
        message="Pipeline monitoring run finalized.",
        severity=PipelineSeverity.INFO.value,
        metadata={
            "total_tasks": summary["total_tasks"],
            "completed_tasks": summary["completed_tasks"],
            "failed_tasks": summary["failed_tasks"],
            "warning_tasks": summary["warning_tasks"],
        },
    )

    LOGGER.info(
        "Pipeline run finalized with status: %s",
        summary["overall_status"],
    )

    return report


# ============================================================
# END OF PART 6
# NEXT: STREAMLIT PIPELINE MONITORING DASHBOARD COMPONENTS
# ============================================================
# ============================================================
# PART 7: STREAMLIT PIPELINE MONITORING DASHBOARD COMPONENTS
# ============================================================

# ============================================================
# 44. STREAMLIT IMPORT HELPER
# ============================================================

def get_streamlit_module():
    """
    Import Streamlit only when the dashboard is requested.

    This allows pipeline_monitor.py to be imported by other Python
    modules without requiring Streamlit at import time.
    """

    try:
        import streamlit as st
        return st

    except ImportError as exc:
        raise ImportError(
            "Streamlit is required to display the pipeline dashboard. "
            "Install it using your project's requirements.txt."
        ) from exc


# ============================================================
# 45. DISPLAY STATUS BADGE
# ============================================================

def render_pipeline_status_badge(
    status: str,
    label: str | None = None,
) -> None:
    """
    Display a colored status badge in Streamlit.
    """

    st = get_streamlit_module()

    normalized_status = normalize_pipeline_status(
        status
    )

    color = PIPELINE_STATUS_COLORS.get(
        normalized_status,
        "#9CA3AF",
    )

    icon = PIPELINE_STATUS_ICONS.get(
        normalized_status,
        "⚪",
    )

    display_label = label or normalized_status

    st.markdown(
        f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 13px;
            border-radius: 20px;
            background-color: {color}20;
            border: 1px solid {color};
            color: {color};
            font-weight: 600;
            font-size: 14px;
            margin: 4px 0;
        ">
            <span>{icon}</span>
            <span>{display_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 46. DISPLAY PIPELINE SUMMARY METRICS
# ============================================================

def render_pipeline_summary_metrics() -> dict[str, Any]:
    """
    Render top-level pipeline monitoring metrics.
    """

    st = get_streamlit_module()

    summary = get_pipeline_execution_summary()

    st.subheader("Pipeline Execution Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tasks",
        summary["total_tasks"],
    )

    col2.metric(
        "Successful",
        summary["successful_tasks"],
    )

    col3.metric(
        "Warnings",
        summary["warning_tasks"],
    )

    col4.metric(
        "Failed",
        summary["failed_tasks"],
    )

    st.progress(
        min(
            max(
                summary["completion_percentage"] / 100,
                0.0,
            ),
            1.0,
        ),
        text=(
            f"Execution progress: "
            f"{summary['completion_percentage']:.1f}%"
        ),
    )

    render_pipeline_status_badge(
        summary["overall_status"]
    )

    return summary


# ============================================================
# 47. DISPLAY TASK STATUS TABLE
# ============================================================

def render_pipeline_task_table() -> None:
    """
    Display the status of registered pipeline tasks.
    """

    st = get_streamlit_module()

    st.subheader("Pipeline Task Status")

    rows = get_pipeline_status_table()

    if not rows:
        st.info(
            "No pipeline tasks have been registered yet."
        )
        return

    import pandas as pd

    dataframe = pd.DataFrame(rows)

    status_options = [
        "All",
        *sorted(
            dataframe["Status"].dropna().unique().tolist()
        ),
    ]

    selected_status = st.selectbox(
        "Filter tasks by status",
        status_options,
        key="pipeline_monitor_status_filter",
    )

    if selected_status != "All":
        dataframe = dataframe[
            dataframe["Status"] == selected_status
        ]

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 48. DISPLAY PIPELINE HEALTH CHECKS
# ============================================================

def render_pipeline_health_checks() -> dict[str, Any]:
    """
    Display monitoring infrastructure and dataset-path checks.
    """

    st = get_streamlit_module()

    st.subheader("System Health Checks")

    health_report = run_pipeline_health_check()

    render_pipeline_status_badge(
        health_report["overall_status"],
        label="Health: " + health_report["overall_status"],
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Passed",
        health_report["successful_checks"],
    )

    col2.metric(
        "Warnings",
        health_report["warning_checks"],
    )

    col3.metric(
        "Failed",
        health_report["failed_checks"],
    )

    for check in health_report["checks"]:
        status = check["status"]

        if status == "SUCCESS":
            st.success(
                f"{check['check']}: {check['message']}"
            )

        elif status == "WARNING":
            st.warning(
                f"{check['check']}: {check['message']}"
            )

        else:
            st.error(
                f"{check['check']}: {check['message']}"
            )

    return health_report


# ============================================================
# 49. DISPLAY RECENT PIPELINE EVENTS
# ============================================================

def render_recent_pipeline_events(
    limit: int = 20,
) -> None:
    """
    Display recent monitoring events for the current run.
    """

    st = get_streamlit_module()

    st.subheader("Recent Pipeline Events")

    events = get_recent_pipeline_events(
        limit=limit
    )

    if not events:
        st.info(
            "No events recorded for the current pipeline run."
        )
        return

    import pandas as pd

    event_rows = []

    for event in events:
        event_rows.append(
            {
                "Timestamp": event.get("timestamp"),
                "Event": event.get("event_type"),
                "Task": event.get("task_name"),
                "Status": event.get("status"),
                "Severity": event.get("severity"),
                "Message": event.get("message"),
            }
        )

    st.dataframe(
        pd.DataFrame(event_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 50. DISPLAY PIPELINE ERROR DETAILS
# ============================================================

def render_pipeline_error_details() -> None:
    """
    Show recorded errors with useful debugging information.
    """

    st = get_streamlit_module()

    st.subheader("Error Diagnostics")

    errors = get_pipeline_error_history()

    if not errors:
        st.success(
            "No pipeline errors have been recorded in the current state."
        )
        return

    st.warning(
        f"{len(errors)} recorded error(s) are available for review."
    )

    for index, error in enumerate(errors, start=1):
        task_name = error.get(
            "task_name",
            "Unknown task",
        )

        exception_type = error.get(
            "exception_type",
            "Unknown exception",
        )

        message = error.get(
            "message",
            "No error message available.",
        )

        with st.expander(
            f"{index}. {task_name} — {exception_type}"
        ):
            st.write(
                "**Recorded at:**",
                error.get("timestamp", "Unknown"),
            )

            st.write(
                "**Severity:**",
                error.get("severity", "UNKNOWN"),
            )

            st.write(
                "**Error message:**",
                message,
            )

            if error.get("file_name"):
                st.write(
                    "**Source file:**",
                    error["file_name"],
                )

            if error.get("line_number") is not None:
                st.write(
                    "**Line number:**",
                    error["line_number"],
                )

            traceback_text = error.get(
                "traceback",
                "",
            )

            if traceback_text:
                st.code(
                    traceback_text,
                    language="text",
                )


# ============================================================
# 51. PIPELINE REPORT DOWNLOAD
# ============================================================

def render_pipeline_report_downloads() -> None:
    """
    Provide JSON and CSV report downloads in Streamlit.
    """

    st = get_streamlit_module()

    st.subheader("Export Monitoring Reports")

    report_col, csv_col = st.columns(2)

    with report_col:
        if st.button(
            "Generate JSON Summary Report",
            key="generate_pipeline_json_report",
        ):
            try:
                report = generate_pipeline_summary_report()

                json_content = json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                    default=pipeline_json_serializer,
                )

                st.download_button(
                    label="Download JSON Report",
                    data=json_content,
                    file_name="pipeline_summary_report.json",
                    mime="application/json",
                    key="download_pipeline_json_report",
                )

                st.success(
                    "JSON summary report generated."
                )

            except Exception as exc:
                LOGGER.exception(
                    "Unable to generate JSON pipeline report."
                )

                st.error(
                    f"Report generation failed: {exc}"
                )

    with csv_col:
        if st.button(
            "Generate CSV Status Report",
            key="generate_pipeline_csv_report",
        ):
            try:
                csv_path = export_pipeline_status_csv()

                with csv_path.open(
                    "rb"
                ) as csv_file:
                    csv_content = csv_file.read()

                st.download_button(
                    label="Download CSV Report",
                    data=csv_content,
                    file_name=csv_path.name,
                    mime="text/csv",
                    key="download_pipeline_csv_report",
                )

                st.success(
                    "CSV status report generated."
                )

            except Exception as exc:
                LOGGER.exception(
                    "Unable to export pipeline status CSV."
                )

                st.error(
                    f"CSV export failed: {exc}"
                )


# ============================================================
# 52. COMPLETE PIPELINE MONITORING DASHBOARD
# ============================================================

def render_pipeline_monitor_dashboard() -> None:
    """
    Render the complete reusable pipeline monitoring interface.

    Call this function from the main Streamlit application when
    the user opens the Pipeline Monitoring section.
    """

    st = get_streamlit_module()

    st.title(
        "Pipeline Monitoring & Diagnostics"
    )

    st.caption(
        "Monitor task execution, system health, warnings, "
        "failures and execution reports."
    )

    st.info(
        "Task statuses reflect recorded monitoring events. "
        "Registering a task does not mean that the task has run."
    )

    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.write(
            "Review task status and investigate issues "
            "before rerunning affected stages."
        )

    with top_right:
        if st.button(
            "Refresh Dashboard",
            key="refresh_pipeline_dashboard",
            use_container_width=True,
        ):
            st.rerun()

    st.divider()

    render_pipeline_summary_metrics()

    st.divider()

    task_tab, health_tab, events_tab, errors_tab, reports_tab = (
        st.tabs(
            [
                "Task Status",
                "System Health",
                "Recent Events",
                "Error Diagnostics",
                "Reports",
            ]
        )
    )

    with task_tab:
        render_pipeline_task_table()

    with health_tab:
        render_pipeline_health_checks()

    with events_tab:
        render_recent_pipeline_events()

    with errors_tab:
        render_pipeline_error_details()

    with reports_tab:
        render_pipeline_report_downloads()


# ============================================================
# END OF PART 7
# NEXT: FINAL INITIALIZATION, SAFE STARTUP & MODULE COMPLETION
# ============================================================
# ============================================================
# PART 8: FINAL INITIALIZATION, SAFE STARTUP & MODULE COMPLETION
# ============================================================

# ============================================================
# 53. INITIALIZE PIPELINE MONITORING
# ============================================================

def initialize_pipeline_monitor(
    register_default_tasks: bool = True,
    recover_previous_tasks: bool = False,
) -> dict[str, Any]:
    """
    Initialize monitoring directories, register pipeline tasks,
    and run a basic infrastructure health check.

    Parameters
    ----------
    register_default_tasks : bool
        Register the default forecasting pipeline task definitions.

    recover_previous_tasks : bool
        Recover unfinished tasks from the current in-memory state.

        This does not automatically load a previous run's task states.
        Loading previous state is an explicit operation because a
        previous run may have used different code or configuration.
    """

    initialize_pipeline_directories()

    initialization_messages = []

    # Register the configured project pipeline.
    if register_default_tasks:
        registration_result = register_default_pipeline()

        initialization_messages.append(
            f"Registered {registration_result['registered_count']} "
            "default pipeline tasks."
        )

    # Optionally recover tasks already marked as running.
    if recover_previous_tasks:
        recovery_result = recover_interrupted_tasks()

        initialization_messages.append(
            f"Recovered {recovery_result['recovered_tasks']} "
            "interrupted task(s)."
        )

    # Check monitoring infrastructure.
    health_report = run_pipeline_health_check()

    # Record the initialization event.
    record_pipeline_event(
        event_type="PIPELINE_MONITOR_INITIALIZED",
        task_name="pipeline_monitor",
        status=health_report["overall_status"],
        message="Pipeline monitoring initialization completed.",
        severity=(
            PipelineSeverity.HIGH.value
            if health_report["failed_checks"] > 0
            else PipelineSeverity.INFO.value
        ),
        metadata={
            "register_default_tasks": register_default_tasks,
            "recover_previous_tasks": recover_previous_tasks,
            "health_status": health_report["overall_status"],
        },
    )

    persist_pipeline_state()

    LOGGER.info(
        "Pipeline monitor initialized. Health status: %s",
        health_report["overall_status"],
    )

    return {
        "initialized": health_report["failed_checks"] == 0,

        "initialized_at": get_pipeline_timestamp(),

        "run_id": PIPELINE_RUN_ID,

        "health_report": health_report,

        "messages": initialization_messages,
    }


# ============================================================
# 54. SAFE TASK REGISTRATION HELPER
# ============================================================

def register_project_task(
    task_name: str,
    file_name: str,
    description: str = "",
    dependencies: list[str] | None = None,
    required_outputs: list[str] | None = None,
) -> dict[str, Any]:
    """
    Register a project task with basic input validation.
    """

    if not task_name or not task_name.strip():
        raise ValueError(
            "Task name cannot be empty."
        )

    if not file_name or not file_name.strip():
        raise ValueError(
            "Task file name cannot be empty."
        )

    return register_pipeline_task(
        task_name=task_name.strip(),
        file_name=file_name.strip(),
        description=description.strip(),
        dependencies=dependencies or [],
        required_outputs=required_outputs or [],
    )


# ============================================================
# 55. CHECK WHETHER A TASK CAN START
# ============================================================

def can_pipeline_task_start(
    task_name: str,
) -> dict[str, Any]:
    """
    Check task existence, dependency readiness and current status.

    This function does not execute the task.
    """

    task = get_pipeline_task(task_name)

    if task is None:
        return {
            "can_start": False,
            "task_name": task_name,
            "reason": "Task is not registered.",
            "status": PipelineStatus.BLOCKED.value,
        }

    current_status = task.get(
        "status",
        PipelineStatus.PENDING.value,
    )

    if current_status == PipelineStatus.RUNNING.value:
        return {
            "can_start": False,
            "task_name": task_name,
            "reason": "Task is already marked as running.",
            "status": current_status,
        }

    dependencies_ready, dependency_message = (
        check_pipeline_dependencies(task_name)
    )

    if not dependencies_ready:
        return {
            "can_start": False,
            "task_name": task_name,
            "reason": dependency_message,
            "status": PipelineStatus.BLOCKED.value,
        }

    return {
        "can_start": True,
        "task_name": task_name,
        "reason": "Task dependencies are satisfied.",
        "status": current_status,
    }


# ============================================================
# 56. GET A DEBUGGING CHECKLIST FOR A TASK
# ============================================================

def get_pipeline_debugging_checklist(
    task_name: str,
) -> dict[str, Any]:
    """
    Build a practical debugging checklist for a registered task.
    """

    task = get_pipeline_task(task_name)

    if task is None:
        return {
            "task_name": task_name,
            "found": False,
            "checklist": [
                "Confirm that the task name is spelled correctly.",
                "Register the task before executing it.",
                "Verify that the expected Python file exists.",
            ],
        }

    task_file = BASE_DIR / task.get(
        "file_name",
        "",
    )

    checklist = [
        "Confirm the task's input dataset exists.",
        "Check that all required Python packages are installed.",
        "Review the task's dependencies and their recorded statuses.",
        "Inspect the task's most recent error message, if any.",
        "Verify that expected output files were created.",
        "Check the execution logs for the relevant timestamp.",
    ]

    if not task_file.is_file():
        checklist.insert(
            0,
            f"Expected Python file was not found: {task_file}",
        )

    if task.get("status") == PipelineStatus.FAILED.value:
        checklist.insert(
            0,
            "Inspect the recorded exception and traceback first.",
        )

    if task.get("status") == PipelineStatus.BLOCKED.value:
        checklist.insert(
            0,
            "Resolve the failed or incomplete dependencies first.",
        )

    return {
        "task_name": task_name,
        "found": True,
        "file_path": str(task_file),
        "current_status": task.get("status"),
        "checklist": checklist,
    }


# ============================================================
# 57. DISPLAY A TASK DEBUGGING CHECKLIST
# ============================================================

def render_pipeline_debugging_checklist(
    task_name: str,
) -> None:
    """
    Display the debugging checklist for a selected pipeline task.
    """

    st = get_streamlit_module()

    result = get_pipeline_debugging_checklist(
        task_name
    )

    st.subheader(
        f"Debugging Checklist: {task_name}"
    )

    if not result["found"]:
        st.warning(
            "This task is not registered."
        )

    else:
        st.write(
            f"**Current status:** "
            f"{result.get('current_status', 'UNKNOWN')}"
        )

        st.write(
            f"**Expected file:** "
            f"{result.get('file_path', 'Unknown')}"
        )

    for index, item in enumerate(
        result["checklist"],
        start=1,
    ):
        st.checkbox(
            item,
            key=f"pipeline_debug_{task_name}_{index}",
        )


# ============================================================
# 58. FINAL PIPELINE MONITOR MODULE INFORMATION
# ============================================================

def get_pipeline_monitor_info() -> dict[str, Any]:
    """
    Return monitoring module metadata and current runtime details.
    """

    return {
        "module_name": "pipeline_monitor",

        "project_name": PIPELINE_CONFIG["project_name"],

        "run_id": PIPELINE_RUN_ID,

        "started_at": PIPELINE_START_TIME,

        "python_version": sys.version.split()[0],

        "platform": platform.platform(),

        "process_id": os.getpid(),

        "base_directory": str(BASE_DIR),

        "output_directory": str(PIPELINE_OUTPUT_DIR),

        "log_file": str(PIPELINE_LOG_FILE),

        "state_file": str(PIPELINE_STATE_FILE),

        "history_file": str(PIPELINE_HISTORY_FILE),

        "summary_file": str(PIPELINE_SUMMARY_FILE),
    }


# ============================================================
# 59. MODULE EXPORTS
# ============================================================

__all__ = [
    "PipelineStatus",
    "PipelineSeverity",
    "PipelineExecutionResult",
    "PIPELINE_CONFIG",
    "PIPELINE_STATE",
    "LOGGER",
    "initialize_pipeline_monitor",
    "register_pipeline_task",
    "register_project_task",
    "get_pipeline_task",
    "update_pipeline_task_status",
    "add_pipeline_warning",
    "validate_pipeline_outputs",
    "check_pipeline_dependencies",
    "track_pipeline_task",
    "execute_pipeline_task",
    "get_pipeline_status_table",
    "get_pipeline_execution_summary",
    "get_recent_pipeline_events",
    "get_pipeline_error_history",
    "generate_pipeline_summary_report",
    "load_pipeline_state",
    "recover_interrupted_tasks",
    "load_pipeline_history",
    "run_pipeline_health_check",
    "export_pipeline_status_csv",
    "finalize_pipeline_run",
    "can_pipeline_task_start",
    "get_pipeline_debugging_checklist",
    "render_pipeline_status_badge",
    "render_pipeline_summary_metrics",
    "render_pipeline_task_table",
    "render_pipeline_health_checks",
    "render_recent_pipeline_events",
    "render_pipeline_error_details",
    "render_pipeline_report_downloads",
    "render_pipeline_debugging_checklist",
    "render_pipeline_monitor_dashboard",
    "get_pipeline_monitor_info",
]


# ============================================================
# END OF pipeline_monitor.py
# ============================================================