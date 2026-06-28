"""
Telemetry event logger — the data foundation for all other modules.

Every synthesized input event flows through this logger, producing JSONL output
suitable for academic analysis. The output format is designed so downstream
anomaly-detection training pipelines can consume it directly.

Output: one JSON object per line, no trailing comma.
All float timestamps at microsecond precision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Event type taxonomy
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    MOUSE_MOVE = "MOUSE_MOVE"
    MOUSE_CLICK_DOWN = "MOUSE_CLICK_DOWN"
    MOUSE_CLICK_UP = "MOUSE_CLICK_UP"
    MOUSE_DRAG_START = "MOUSE_DRAG_START"
    MOUSE_DRAG_MOVE = "MOUSE_DRAG_MOVE"
    MOUSE_DRAG_END = "MOUSE_DRAG_END"
    KEY_DOWN = "KEY_DOWN"
    KEY_UP = "KEY_UP"
    GAZE_POINT = "GAZE_POINT"           # simulated eye-gaze fixation
    SCREEN_SCAN = "SCREEN_SCAN"         # visual search event
    REACTION_TIME = "REACTION_TIME"      # cognitive delay measured
    COGNITIVE_DELAY = "COGNITIVE_DELAY"  # any modeled delay
    SCENARIO_STATE_CHANGE = "SCENARIO_STATE_CHANGE"
    SESSION_META = "SESSION_META"
    INTERRUPTION_START = "INTERRUPTION_START"
    INTERRUPTION_END = "INTERRUPTION_END"
    CAMERA_ROTATE = "CAMERA_ROTATE"      # simulated camera rotation
    MISCLICK = "MISCLICK"                # intentional off-target click
    AFK = "AFK"                          # tab-out / away-from-keyboard
    ERROR = "ERROR"                      # scenario-level error


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """A single recorded input or cognitive event."""
    session_id: str
    timestamp: float                          # monotonic seconds since session start
    wall_time: float                          # absolute unix timestamp
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=lambda: ["synthetic"])
    persona: str = ""
    session_phase: str = ""
    affective_state: str = ""
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict:
        """Serialize to dictionary for JSONL output."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


# ---------------------------------------------------------------------------
# Convenience constructors for specific event types
# ---------------------------------------------------------------------------

def make_mouse_move(
    session_id: str, timestamp: float, wall_time: float,
    x: float, y: float, velocity: float, acceleration: float,
    phase: str, persona: str, session_phase: str, affective_state: str,
    elapsed_ms: float,
) -> TelemetryEvent:
    return TelemetryEvent(
        session_id=session_id, timestamp=timestamp, wall_time=wall_time,
        event_type=EventType.MOUSE_MOVE,
        data={"x": round(x, 2), "y": round(y, 2),
              "velocity": round(velocity, 2), "acceleration": round(acceleration, 2),
              "phase": phase},
        persona=persona, session_phase=session_phase, affective_state=affective_state,
        elapsed_ms=elapsed_ms,
    )


def make_click(
    event_type: EventType, session_id: str, timestamp: float,
    wall_time: float, x: float, y: float, button: str,
    persona: str, session_phase: str, affective_state: str, elapsed_ms: float,
    hold_duration_ms: float = 0.0, micro_drift_px: float = 0.0,
    is_misclick: bool = False,
) -> TelemetryEvent:
    labels = ["synthetic"]
    if is_misclick:
        labels.append("misclick")
    return TelemetryEvent(
        session_id=session_id, timestamp=timestamp, wall_time=wall_time,
        event_type=event_type,
        data={"x": round(x, 2), "y": round(y, 2), "button": button,
              "hold_duration_ms": round(hold_duration_ms, 3),
              "micro_drift_px": round(micro_drift_px, 3)},
        labels=labels, persona=persona,
        session_phase=session_phase, affective_state=affective_state,
        elapsed_ms=elapsed_ms,
    )


def make_key_event(
    event_type: EventType, session_id: str, timestamp: float,
    wall_time: float, key: str, is_correction: bool,
    persona: str, session_phase: str, affective_state: str, elapsed_ms: float,
) -> TelemetryEvent:
    labels = ["synthetic"]
    if is_correction:
        labels.append("correction")
    return TelemetryEvent(
        session_id=session_id, timestamp=timestamp, wall_time=wall_time,
        event_type=event_type,
        data={"key": key, "is_correction": is_correction},
        labels=labels, persona=persona,
        session_phase=session_phase, affective_state=affective_state,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------

class TelemetryLogger:
    """
    Buffered JSONL telemetry recorder.

    Events are held in an in-memory buffer and auto-flushed when the buffer
    exceeds `buffer_size`.  Call `flush()` manually at session boundaries or
    important checkpoints.  Call `close()` when the session ends.
    """

    def __init__(
        self,
        output_dir: Path | str,
        session_id: str,
        buffer_size: int = 1000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.buffer_size = buffer_size

        self._buffer: list[TelemetryEvent] = []
        self._file_path = self.output_dir / f"{session_id}.jsonl"
        self._fh = open(self._file_path, "a", encoding="utf-8")
        self._event_count = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: TelemetryEvent) -> None:
        """Record a single telemetry event."""
        self._buffer.append(event)
        self._event_count += 1
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def log_many(self, events: list[TelemetryEvent]) -> None:
        """Record multiple events at once."""
        self._buffer.extend(events)
        self._event_count += len(events)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write all buffered events to disk."""
        if not self._buffer:
            return
        lines = []
        for evt in self._buffer:
            lines.append(json.dumps(evt.as_dict(), ensure_ascii=False, cls=_NumpyEncoder))
        self._fh.write("\n".join(lines) + "\n")
        self._fh.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining events, write session summary, close file."""
        self.flush()
        summary = self.session_summary()
        summary_line = json.dumps({"SESSION_SUMMARY": summary}, ensure_ascii=False)
        self._fh.write(summary_line + "\n")
        self._fh.flush()
        self._fh.close()

    def export_dataframe(self) -> "pd.DataFrame":
        """Load the current session's JSONL into a pandas DataFrame."""
        import pandas as pd
        self.flush()
        if not self._file_path.exists():
            return pd.DataFrame()
        df = pd.read_json(self._file_path, lines=True)
        # The last line is the summary dict, drop it
        if "SESSION_SUMMARY" in df.columns:
            df = df[df["SESSION_SUMMARY"].isna()]
        return df

    def session_summary(self) -> dict:
        """Compute aggregate statistics for the session."""
        duration_s = time.time() - self._start_time
        type_counts: dict[str, int] = {}
        for evt in self._buffer:
            key = evt.event_type.value
            type_counts[key] = type_counts.get(key, 0) + 1
        # Note: counts only reflect buffered events; for a complete count
        # use the in-memory _event_count
        return {
            "session_id": self.session_id,
            "duration_s": round(duration_s, 3),
            "total_events": self._event_count,
            "event_type_counts": type_counts,
            "output_file": str(self._file_path),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def file_path(self) -> Path:
        return self._file_path
