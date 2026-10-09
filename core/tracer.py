"""
Trace logging – structured, append-only event log for the entire reasoning pipeline.

Every LLM call, tool use, checker result, belief mutation, and sublation is
written here.  The Sākṣī (witness) reads only from this log.

Format: JSON-Lines (one JSON object per line) written to traces/YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class TraceEvent(str, Enum):
    LLM_CALL_START    = "llm_call_start"
    LLM_CALL_END      = "llm_call_end"
    CLAIM_PROPOSED    = "claim_proposed"
    CHECKER_RUN       = "checker_run"
    BELIEF_CREATED    = "belief_created"
    BELIEF_SUBLATED   = "belief_sublated"
    TOOL_CALL         = "tool_call"
    TOOL_RESULT       = "tool_result"
    ABSTAIN           = "abstain"
    FINAL_OUTPUT      = "final_output"
    WITNESS_FLAG      = "witness_flag"     # written by Sākṣī, read-only path
    ERROR             = "error"


# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Append-only JSONL logger.

    Usage:
        logger = TraceLogger()
        run_id = logger.new_run()
        logger.log(run_id, TraceEvent.LLM_CALL_START, {"model": "gpt-4o", ...})
    """

    def __init__(self, trace_dir: str | Path = "traces") -> None:
        self._dir = Path(trace_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._structlog = structlog.get_logger("sattvic.trace")

    def _file_for_today(self) -> Path:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"{today}.jsonl"

    def new_run(self, query: Optional[str] = None) -> str:
        """Generate a fresh run ID and log the start event."""
        run_id = str(uuid.uuid4())
        self.log(run_id, TraceEvent.LLM_CALL_START, {
            "query": query or "",
            "note": "run_start",
        })
        return run_id

    def log(
        self,
        run_id: str,
        event: TraceEvent,
        payload: dict[str, Any],
        *,
        step: Optional[int] = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts":     datetime.now(tz=timezone.utc).isoformat(),
            "run_id": run_id,
            "event":  event.value,
            "step":   step,
            **payload,
        }
        # Write to JSONL file
        with self._file_for_today().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        # Also emit to structlog (console in dev)
        self._structlog.info(event.value, run_id=run_id, **payload)

    def read_run(self, run_id: str, *, days_back: int = 7) -> list[dict[str, Any]]:
        """Return all trace records for a given run_id from recent files."""
        records = []
        for i in range(days_back):
            from datetime import timedelta
            day = (datetime.now(tz=timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            path = self._dir / f"{day}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("run_id") == run_id:
                        records.append(rec)
        return sorted(records, key=lambda r: r["ts"])


# ---------------------------------------------------------------------------
# Global singleton (override in tests)
# ---------------------------------------------------------------------------

_default_trace_dir = os.environ.get("SATTVIC_TRACE_DIR", "traces")
tracer: TraceLogger = TraceLogger(trace_dir=_default_trace_dir)
