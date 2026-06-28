"""
Session & Scenario Scheduler — orchestrates multi-session data generation.

Responsibilities:
  - Select persona + scenario per session (weighted random, avoiding repeats)
  - Maintain cross-session memory (JSON log of past sessions)
  - Apply circadian (time-of-day) influence on reaction times
  - Probabilistic session anomalies (10% short check-in, 5% skip)
  - Drive session execution and collect results
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import numpy as np

from core.config import SchedulerConfig, PersonaType


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Configuration for a single session run."""
    session_id: str
    persona: PersonaType
    scenario_name: str
    start_time: datetime
    anomaly_type: Optional[Literal["short_checkin", "skip"]] = None
    rng_seed: int = 0
    hour_of_day: float = 14.0


@dataclass
class SessionRecord:
    """Record of a completed session for cross-session memory."""
    session_id: str
    scenario: str
    persona: str
    start_time: str       # ISO format
    duration_s: float
    events_count: int
    anomaly_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Circadian Model
# ---------------------------------------------------------------------------

class CircadianModel:
    """
    Time-of-day influence on cognitive performance.

    Quadratic function: minimum RT at 10:00 (peak alertness),
    maximum at 03:00 (circadian trough).
    """

    @staticmethod
    def compute_rt_multiplier(hour: float) -> float:
        """
        Returns a multiplier on reaction time (1.0 = baseline at 10:00).

        Parameters
        ----------
        hour : Hour of day (0–24, can be fractional).

        Returns
        -------
        Multiplier in [1.0, ~1.4].
        """
        # Shift so 10:00 is the zero point
        shifted = (hour - 10.0 + 24.0) % 24.0
        # Max at 17 hours from 10:00 = 03:00
        max_shift = 17.0
        mult = 1.0 + 0.4 * (shifted / max_shift) ** 2
        return mult

    @staticmethod
    def compute_error_multiplier(hour: float) -> float:
        """Error rate multiplier — same shape, slightly different scale."""
        shifted = (hour - 10.0 + 24.0) % 24.0
        max_shift = 17.0
        return 1.0 + 0.5 * (shifted / max_shift) ** 2


# ---------------------------------------------------------------------------
# Session Memory
# ---------------------------------------------------------------------------

class SessionMemory:
    """
    Persistent cross-session memory.

    Stores a JSON file with recent session records to:
      - Avoid repeating the same persona+scenario back-to-back
      - Track how many times each scenario has been run
      - Support diversity analysis of generated data
    """

    def __init__(self, memory_file: Path | str) -> None:
        self._path = Path(memory_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[SessionRecord] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                self._records = [
                    SessionRecord(**r) for r in data.get("sessions", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump({
                "sessions": [asdict(r) for r in self._records],
                "updated": datetime.now().isoformat(),
            }, f, indent=2)

    def record_session(self, record: SessionRecord) -> None:
        self._records.append(record)
        # Keep only recent history
        max_history = 20
        if len(self._records) > max_history:
            self._records = self._records[-max_history:]
        self._save()

    def recent_scenarios(self, k: int = 5) -> list[str]:
        """Return the last k scenario names (most recent first)."""
        return [r.scenario for r in self._records[-k:]]

    def recent_personas(self, k: int = 3) -> list[str]:
        """Return the last k persona names (most recent first)."""
        return [r.persona for r in self._records[-k:]]

    def session_count_for_scenario(self, scenario: str) -> int:
        """How many times this scenario has been run total."""
        return sum(1 for r in self._records if r.scenario == scenario)

    def total_sessions(self) -> int:
        return len(self._records)

    def flush(self) -> None:
        self._save()


# ---------------------------------------------------------------------------
# Session Scheduler
# ---------------------------------------------------------------------------

class SessionScheduler:
    """
    Top-level orchestration of multi-session telemetry generation.

    Usage:
        sched = SessionScheduler(config, memory, rng)
        for session_cfg in sched.generate_sessions(count=10):
            run_session(session_cfg)
    """

    # Available scenarios (extend this list as new scenarios are added)
    AVAILABLE_SCENARIOS = [
        "chaos_altar_dragon_bones",
    ]

    # Persona selection weights
    DEFAULT_PERSONA_WEIGHTS = {
        PersonaType.FAST_ACCURATE: 0.35,
        PersonaType.SLOW_METHODICAL: 0.35,
        PersonaType.DISTRACTED_ERROR_PRONE: 0.30,
    }

    def __init__(
        self,
        config: SchedulerConfig,
        memory: SessionMemory,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.memory = memory
        self._rng = rng

        # Persona weights (from config or defaults)
        weights = config.persona_weights
        self._persona_types = [PersonaType(k) for k in weights.keys()]
        self._persona_weights = [weights[pt.value] for pt in self._persona_types]

        # Normalize
        total = sum(self._persona_weights)
        if total > 0:
            self._persona_weights = [w / total for w in self._persona_weights]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_next_session(self) -> Optional[SessionConfig]:
        """
        Select persona + scenario for the next session, applying anomaly lottery.

        Returns None if the session should be skipped.
        """
        # 1. Anomaly lottery
        if self._rng.random() < self.config.anomaly_skip_prob:
            return None  # skip this session entirely

        # 2. Pick persona (avoid last-used)
        recent_personas = self.memory.recent_personas(self.config.avoid_recent_personas)
        eligible_personas = [
            pt for pt in self._persona_types
            if pt.value not in recent_personas
        ]
        if not eligible_personas:
            eligible_personas = list(self._persona_types)

        # Re-weight eligible personas
        persona = self._rng.choice(eligible_personas)

        # 3. Pick scenario (avoid recent repeats)
        recent_scenarios = self.memory.recent_scenarios(self.config.avoid_recent_scenarios)
        eligible_scenarios = [
            s for s in self.AVAILABLE_SCENARIOS if s not in recent_scenarios
        ]
        if not eligible_scenarios:
            eligible_scenarios = list(self.AVAILABLE_SCENARIOS)
        scenario = self._rng.choice(eligible_scenarios)

        # 4. Anomaly: short check-in
        anomaly_type = None
        if self._rng.random() < self.config.anomaly_short_checkin_prob:
            anomaly_type = "short_checkin"

        # 5. Time of day (randomized across sessions for diversity)
        # Sample from a realistic play-time distribution:
        # peak at 18:00-22:00 (evening), secondary peak at 10:00-14:00 (weekend)
        hour_of_day = self._sample_play_time()

        return SessionConfig(
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            persona=persona,
            scenario_name=scenario,
            start_time=datetime.now(),
            anomaly_type=anomaly_type,
            rng_seed=self._rng.integers(0, 2**31),
            hour_of_day=hour_of_day,
        )

    def generate_sessions(self, count: int) -> list[SessionConfig]:
        """Generate `count` session configurations."""
        sessions = []
        for _ in range(count):
            cfg = self.select_next_session()
            if cfg is not None:
                sessions.append(cfg)
            else:
                # Record the skip in memory
                self.memory.record_session(SessionRecord(
                    session_id=f"skip_{uuid.uuid4().hex[:8]}",
                    scenario="none",
                    persona="none",
                    start_time=datetime.now().isoformat(),
                    duration_s=0.0,
                    events_count=0,
                    anomaly_type="skip",
                ))
        return sessions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_play_time(self) -> float:
        """
        Sample a realistic hour of day for play sessions.

        Mixture: 60% evening (18-23), 25% afternoon (12-18),
        10% morning (8-12), 5% late night (23-3).
        """
        mode = self._rng.choice(
            ["evening", "afternoon", "morning", "late_night"],
            p=[0.60, 0.25, 0.10, 0.05],
        )
        ranges = {
            "evening": (18.0, 23.0),
            "afternoon": (12.0, 18.0),
            "morning": (8.0, 12.0),
            "late_night": (23.0, 27.0),  # wrap-around handled below
        }
        lo, hi = ranges[mode]
        h = self._rng.uniform(lo, hi) % 24.0
        return h
