"""
Cognitive Delay Engine — models human reaction times, personality, fatigue, and affect.

Sub-models:
  ExWaldRT          — Ex-Gaussian + inverse Gaussian reaction time distribution
  UserPersonaModel  — personality type that modulates all timing/error parameters
  SessionPhaseManager — WARMUP→GROOVE→PLATEAU→FATIGUE progression with micro-breaks
  AffectiveStateMachine — hidden Markov model over 5 emotional states
  InterruptionModel — probabilistic real-world distractions
  CognitiveEngine    — public facade ticking all sub-models
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy import stats

from core.config import (
    CognitiveConfig, PersonaParams, PhaseTiming, HMMTransitionMatrix,
    PersonaType, SessionPhase, AffectiveState, InterruptionConfig,
    ExWaldParams, DEFAULT_PERSONAS,
)


# ---------------------------------------------------------------------------
# Enums re-exported for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "ExWaldRT", "UserPersonaModel", "SessionPhaseManager",
    "AffectiveStateMachine", "InterruptionModel", "CognitiveEngine",
    "PersonaType", "SessionPhase", "AffectiveState", "Interruption",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Interruption:
    """A real-world distraction event."""
    interruption_type: str   # "cursor_freeze" | "gaze_drift" | "keystroke_burst"
    duration_s: float
    key_count: int = 0       # only for keystroke_burst
    start_time: float = 0.0


# ---------------------------------------------------------------------------
# 1. Ex-Wald Reaction Time Distribution
# ---------------------------------------------------------------------------

class ExWaldRT:
    """
    Reaction time sampled from:
        RT = Wald(mu, sigma) + Exponential(lambda)

    - Wald (inverse Gaussian): models decision time + motor preparation
      with positive drift toward a threshold.
    - Exponential tail: rare long delays from distraction or hesitation.

    Clamped to [rt_min_ms, rt_max_ms].
    """

    def __init__(self, params: ExWaldParams, rng: np.random.Generator) -> None:
        self.mu = params.mu
        self.sigma = params.sigma
        self.lambda_ = params.lambda_
        self.rt_min = params.rt_min_ms
        self.rt_max = params.rt_max_ms
        self._rng = rng

    def sample(self) -> float:
        """Return a reaction time sample in milliseconds."""
        # Wald component
        # scipy.stats.wald takes mean= parameter directly (not mu/sigma like R)
        wald_sample = stats.wald.rvs(loc=0, scale=self.sigma, random_state=self._rng)
        wald_sample = wald_sample * self.mu / (self.mu + self.sigma)  # rescale

        # Exponential tail
        exp_sample = self._rng.exponential(1.0 / self.lambda_) if self.lambda_ > 0 else 0.0

        rt = wald_sample + exp_sample
        return float(np.clip(rt, self.rt_min, self.rt_max))

    def sample_batch(self, n: int) -> np.ndarray:
        """Return n reaction time samples."""
        return np.array([self.sample() for _ in range(n)])

    def pdf(self, t: float) -> float:
        """Probability density at time t (ms) — approximate, for analysis."""
        # Simplified: treat as ex-Gaussian
        exgauss = stats.exponnorm
        return float(exgauss.pdf(t, K=self.lambda_, loc=self.mu, scale=self.sigma))


# ---------------------------------------------------------------------------
# 2. User Persona Model
# ---------------------------------------------------------------------------

class UserPersonaModel:
    """
    Loads persona-specific parameters from YAML or uses built-in defaults.

    Each persona modulates:
      - Reaction time distribution parameters
      - Click hold times
      - Typo / error rate
      - Micro-break frequency
      - Fatigue multiplier
      - Distracter probability
      - Misclick rate
      - Tremor amplitude scale
      - Camera rotation frequency
      - AFK probability
    """

    def __init__(
        self,
        persona_type: PersonaType,
        config: CognitiveConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.type = persona_type

        # Load from config or use defaults
        if config and persona_type.value in config.personas:
            params = config.personas[persona_type.value]
        elif persona_type.value in DEFAULT_PERSONAS:
            params = DEFAULT_PERSONAS[persona_type.value]
        else:
            params = DEFAULT_PERSONAS["fast_accurate"]

        self.params = params
        self._rng = rng or np.random.default_rng()

        # Expose commonly-used values as properties
        self.rt_dist = ExWaldRT(params.exwald, self._rng)
        self.hold_time_mean = params.hold_time_mean_ms
        self.hold_time_std = params.hold_time_std_ms
        self.typo_rate = params.typo_rate
        self.micro_break_freq = params.micro_break_frequency_per_min
        self.fatigue_rt_mult = params.fatigue_rt_multiplier
        self.distracter_prob = params.distracter_probability
        self.misclick_rate = params.misclick_rate
        self.tremor_scale = params.tremor_amplitude_scale
        self.camera_rotation_freq = params.camera_rotation_frequency
        self.afk_prob_per_min = params.afk_probability_per_minute


# ---------------------------------------------------------------------------
# 3. Session Phase Manager
# ---------------------------------------------------------------------------

class SessionPhaseManager:
    """
    Deterministic phase progression based on elapsed session time:

        WARMUP (0–3 min)
          ↓
        GROOVE (3–15 min)  ←────────────────┐
          ↓                                   │
        PLATEAU (15–30 min)                   │
          ↓  (micro-breaks every 2–5 min)     │
          ├── MICRO_BREAK (3–8 s) → RECOVERY ─┘
          ↓
        FATIGUE (30+ min) — increasing errors, tremor, RT slowdown
    """

    def __init__(self, config: PhaseTiming, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng

        self._current_phase: SessionPhase = SessionPhase.WARMUP
        self._elapsed_s: float = 0.0

        # Next micro-break timing (randomised within configured interval)
        self._next_micro_break_s: float = self._rng.uniform(*config.micro_break_interval_s)
        self._micro_break_remaining_s: float = 0.0
        self._recovery_remaining_s: float = 0.0
        self._in_post_break_recovery: bool = False

    @property
    def current_phase(self) -> SessionPhase:
        return self._current_phase

    def update(self, dt_s: float) -> SessionPhase:
        """
        Advance the phase state machine by dt_s seconds.

        Returns the new (or unchanged) phase.
        """
        self._elapsed_s += dt_s

        # Handle micro-break and recovery countdowns
        if self._current_phase == SessionPhase.MICRO_BREAK:
            self._micro_break_remaining_s -= dt_s
            if self._micro_break_remaining_s <= 0:
                self._current_phase = SessionPhase.RECOVERY
                self._recovery_remaining_s = self._rng.uniform(
                    *self.config.recovery_duration_s
                )
                self._in_post_break_recovery = True
            return self._current_phase

        if self._current_phase == SessionPhase.RECOVERY:
            self._recovery_remaining_s -= dt_s
            if self._recovery_remaining_s <= 0:
                self._current_phase = SessionPhase.GROOVE
                self._in_post_break_recovery = False
                # Schedule next micro-break
                self._next_micro_break_s = (
                    self._elapsed_s + self._rng.uniform(*self.config.micro_break_interval_s)
                )
            return self._current_phase

        # Phase transitions by elapsed time
        if self._elapsed_s < self.config.warmup_duration_s:
            self._current_phase = SessionPhase.WARMUP
        elif self._elapsed_s < (self.config.warmup_duration_s + self.config.groove_duration_s):
            self._current_phase = SessionPhase.GROOVE
        elif self._elapsed_s < self.config.fatigue_onset_s:
            self._current_phase = SessionPhase.PLATEAU
        else:
            self._current_phase = SessionPhase.FATIGUE

        # Check for micro-break during PLATEAU or FATIGUE
        if self._current_phase in (SessionPhase.PLATEAU, SessionPhase.FATIGUE):
            if self._elapsed_s >= self._next_micro_break_s:
                self._current_phase = SessionPhase.MICRO_BREAK
                self._micro_break_remaining_s = self._rng.uniform(
                    *self.config.micro_break_duration_s
                )

        return self._current_phase

    def reset(self) -> None:
        """Reset to beginning of session."""
        self._current_phase = SessionPhase.WARMUP
        self._elapsed_s = 0.0
        self._next_micro_break_s = self._rng.uniform(*self.config.micro_break_interval_s)

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s


# ---------------------------------------------------------------------------
# 4. Affective State Machine (HMM)
# ---------------------------------------------------------------------------

class AffectiveStateMachine:
    """
    5-state hidden Markov model over affective states:

        NEUTRAL ⇄ FOCUSED ⇄ BORED ⇄ FRUSTRATED ⇄ RUSHING

    Each persona has its own transition matrix.  Transition probabilities
    are dynamically modulated by:
      - Recent error rate (↑ errors → ↑ FRUSTRATED)
      - Time on task (↑ time → ↑ BORED)
      - Session phase (FATIGUE → ↑ RUSHING)
    """

    STATE_ORDER = [
        AffectiveState.NEUTRAL,
        AffectiveState.FOCUSED,
        AffectiveState.BORED,
        AffectiveState.FRUSTRATED,
        AffectiveState.RUSHING,
    ]

    def __init__(
        self,
        transition_matrix: HMMTransitionMatrix,
        rng: np.random.Generator,
    ) -> None:
        self._base_matrix = np.array(transition_matrix.as_matrix())
        self._rng = rng
        self._current_state = AffectiveState.NEUTRAL
        self._state_index = 0

    @property
    def current_state(self) -> AffectiveState:
        return self._current_state

    def step(
        self,
        recent_error_rate: float = 0.0,
        time_on_task_s: float = 0.0,
        session_phase: SessionPhase = SessionPhase.GROOVE,
    ) -> AffectiveState:
        """
        Sample next affective state from dynamically-modulated transition matrix.

        Parameters
        ----------
        recent_error_rate : Fraction of recent actions that were errors (0–1).
        time_on_task_s : Elapsed session time in seconds.
        session_phase : Current session phase for context.
        """
        # Clone base matrix
        tm = self._base_matrix.copy()

        # Modulate: errors → shift some NEUTRAL/FOCUSED mass to FRUSTRATED
        if recent_error_rate > 0.05:
            # Increase transitions TO frustrated
            for i in range(5):
                shift = recent_error_rate * 0.3
                tm[i, 3] = min(tm[i, 3] + shift, 0.95)
                # Renormalize row
                tm[i] = tm[i] / tm[i].sum()

        # Modulate: long time on task → shift to BORED
        if time_on_task_s > 1200:  # 20+ minutes
            boredom_factor = min((time_on_task_s - 1200) / 3600, 0.3)
            for i in range(5):
                tm[i, 2] = min(tm[i, 2] + boredom_factor, 0.95)
                tm[i] = tm[i] / tm[i].sum()

        # Modulate: FATIGUE phase → shift to RUSHING
        if session_phase == SessionPhase.FATIGUE:
            for i in range(5):
                tm[i, 4] = min(tm[i, 4] + 0.1, 0.95)
                tm[i] = tm[i] / tm[i].sum()

        # Sample next state
        probs = tm[self._state_index]
        self._state_index = self._rng.choice(5, p=probs)
        self._current_state = self.STATE_ORDER[self._state_index]

        return self._current_state

    def reset(self) -> None:
        self._current_state = AffectiveState.NEUTRAL
        self._state_index = 0


# ---------------------------------------------------------------------------
# 5. Interruption Model
# ---------------------------------------------------------------------------

class InterruptionModel:
    """
    Probabilistically generates real-world interruptions.

    Types:
      - CURSOR_FREEZE: cursor stops 2–15 s (checking phone, reading)
      - GAZE_DRIFT: cursor drifts to screen edge 1–3 s (notification check)
      - KEYSTROKE_BURST: 3–20 keystrokes with backspaces 5–30 s (replying to msg)
    """

    def __init__(
        self,
        config: InterruptionConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self._rng = rng
        self._base_rate = config.base_rate_per_minute
        self._type_weights = [
            config.cursor_freeze_ratio,
            config.gaze_drift_ratio,
            config.keystroke_burst_ratio,
        ]
        self._types = ["cursor_freeze", "gaze_drift", "keystroke_burst"]

    def sample(
        self,
        session_phase: SessionPhase,
        affective_state: AffectiveState,
        persona_multiplier: float = 1.0,
        dt_s: float = 10.0,
    ) -> Optional[Interruption]:
        """
        Sample whether an interruption occurs in the next dt_s seconds.

        Returns an Interruption or None.
        """
        # Phase-based rate modulation
        phase_mult = {
            SessionPhase.WARMUP: 0.5,
            SessionPhase.GROOVE: 0.3,
            SessionPhase.PLATEAU: 0.6,
            SessionPhase.MICRO_BREAK: 0.0,
            SessionPhase.FATIGUE: 1.5,
            SessionPhase.RECOVERY: 0.8,
        }.get(session_phase, 1.0)

        # Affect-based rate modulation
        affect_mult = {
            AffectiveState.NEUTRAL: 1.0,
            AffectiveState.FOCUSED: 0.3,
            AffectiveState.BORED: 1.8,
            AffectiveState.FRUSTRATED: 1.2,
            AffectiveState.RUSHING: 0.4,
        }.get(affective_state, 1.0)

        # Probability of at least one interruption in dt_s
        rate_per_s = (self._base_rate / 60.0) * phase_mult * affect_mult * persona_multiplier
        prob = 1.0 - math.exp(-rate_per_s * dt_s)

        if self._rng.random() > prob:
            return None

        # Choose interruption type
        itype = self._rng.choice(self._types, p=self._type_weights)

        # Sample duration based on type
        if itype == "cursor_freeze":
            dur = self._rng.uniform(*self.config.cursor_freeze_duration_s)
            return Interruption(itype, dur)
        elif itype == "gaze_drift":
            dur = self._rng.uniform(*self.config.gaze_drift_duration_s)
            return Interruption(itype, dur)
        else:  # keystroke_burst
            dur = self._rng.uniform(*self.config.keystroke_burst_duration_s)
            n_keys = self._rng.integers(*self.config.keystroke_burst_keys)
            return Interruption(itype, dur, key_count=int(n_keys))


# ---------------------------------------------------------------------------
# 6. Cognitive Engine — Public Facade
# ---------------------------------------------------------------------------

class CognitiveEngine:
    """
    Top-level cognitive model wiring together:

      - UserPersonaModel    (who is this user?)
      - SessionPhaseManager (where are they in the session?)
      - AffectiveStateMachine (how are they feeling?)
      - InterruptionModel   (are they distracted?)
      - CircadianModel      (what time of day is it?)

    All timing and behavioral modifiers flow through this single interface
    so scenarios only need to call `cognitive.update()` and
    `cognitive.get_reaction_delay_ms()`.
    """

    # Phase → RT multiplier mapping
    PHASE_RT_MULT: dict[SessionPhase, float] = {
        SessionPhase.WARMUP: 1.25,
        SessionPhase.GROOVE: 1.0,
        SessionPhase.PLATEAU: 1.05,
        SessionPhase.MICRO_BREAK: 2.0,
        SessionPhase.FATIGUE: 1.35,
        SessionPhase.RECOVERY: 1.10,
    }

    PHASE_ERROR_MULT: dict[SessionPhase, float] = {
        SessionPhase.WARMUP: 1.3,
        SessionPhase.GROOVE: 1.0,
        SessionPhase.PLATEAU: 1.1,
        SessionPhase.MICRO_BREAK: 0.0,
        SessionPhase.FATIGUE: 2.0,
        SessionPhase.RECOVERY: 1.15,
    }

    AFFECT_RT_MULT: dict[AffectiveState, float] = {
        AffectiveState.NEUTRAL: 1.0,
        AffectiveState.FOCUSED: 0.85,
        AffectiveState.BORED: 1.2,
        AffectiveState.FRUSTRATED: 0.9,
        AffectiveState.RUSHING: 0.75,
    }

    AFFECT_ERROR_MULT: dict[AffectiveState, float] = {
        AffectiveState.NEUTRAL: 1.0,
        AffectiveState.FOCUSED: 0.7,
        AffectiveState.BORED: 1.1,
        AffectiveState.FRUSTRATED: 2.0,
        AffectiveState.RUSHING: 2.5,
    }

    def __init__(
        self,
        persona_type: PersonaType,
        config: CognitiveConfig | None = None,
        rng: np.random.Generator | None = None,
        hour_of_day: float = 14.0,  # default: 2 PM
    ) -> None:
        self._rng = rng or np.random.default_rng()
        self._config = config or CognitiveConfig()

        # Sub-models
        self.persona = UserPersonaModel(persona_type, self._config, self._rng)
        self.phase_manager = SessionPhaseManager(
            self._config.phases, self._rng
        )
        self.affect_hmm = AffectiveStateMachine(
            self._config.hmm_default if not config else config.hmm_default,
            self._rng,
        )
        self.interruption_model = InterruptionModel(
            InterruptionConfig(), self._rng,
        )

        # State tracking
        self._hour_of_day = hour_of_day
        self._recent_errors: list[bool] = []  # sliding window
        self._error_window_size = 20
        self._update_count = 0
        self._last_phase = self.phase_manager.current_phase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        dt_s: float,
        error_occurred: bool = False,
    ) -> None:
        """
        Advance all sub-models by dt_s seconds.

        Call this once per scenario "tick" (e.g., after each action).
        """
        self._update_count += 1

        # Track errors in sliding window
        self._recent_errors.append(error_occurred)
        if len(self._recent_errors) > self._error_window_size:
            self._recent_errors.pop(0)

        # Update phase
        old_phase = self.phase_manager.current_phase
        self.phase_manager.update(dt_s)

        # Update affective state (every ~10 seconds of simulation time)
        if self._update_count % 10 == 0:
            recent_error_rate = sum(self._recent_errors) / max(len(self._recent_errors), 1)
            self.affect_hmm.step(
                recent_error_rate=recent_error_rate,
                time_on_task_s=self.phase_manager.elapsed_s,
                session_phase=self.phase_manager.current_phase,
            )

    def get_reaction_delay_ms(self) -> float:
        """
        Sample a reaction time delay (ms) modulated by persona, phase, affect,
        and circadian rhythm.
        """
        base_rt = self.persona.rt_dist.sample()

        # Phase modulation
        phase_mult = self.PHASE_RT_MULT.get(self.phase_manager.current_phase, 1.0)

        # Affect modulation
        affect_mult = self.AFFECT_RT_MULT.get(self.affect_hmm.current_state, 1.0)

        # Circadian modulation
        circadian_mult = self._circadian_multiplier()

        # Fatigue: progressively scale RT in FATIGUE phase
        fatigue_mult = 1.0
        if self.phase_manager.current_phase == SessionPhase.FATIGUE:
            time_in_fatigue = self.phase_manager.elapsed_s - self._config.phases.fatigue_onset_s
            fatigue_frac = min(time_in_fatigue / 1800.0, 1.0)  # ramp over 30 min
            fatigue_mult = 1.0 + fatigue_frac * (self.persona.fatigue_rt_mult - 1.0)

        return base_rt * phase_mult * affect_mult * circadian_mult * fatigue_mult

    def get_post_action_delay_ms(self) -> float:
        """
        Post-completion delay (100–500 ms) — the pause after finishing an action
        before moving to the next one.  Shorter when RUSHING, longer when BORED.
        """
        base = self._rng.uniform(100.0, 500.0)
        affect_mult = {
            AffectiveState.NEUTRAL: 1.0, AffectiveState.FOCUSED: 0.8,
            AffectiveState.BORED: 1.5, AffectiveState.FRUSTRATED: 0.6,
            AffectiveState.RUSHING: 0.5,
        }.get(self.affect_hmm.current_state, 1.0)
        return base * affect_mult

    def get_micro_break(self) -> Optional[float]:
        """If currently in a micro-break, return remaining duration (s)."""
        if self.phase_manager.current_phase == SessionPhase.MICRO_BREAK:
            return self.phase_manager._micro_break_remaining_s
        return None

    def get_interruption(self, dt_s: float = 10.0) -> Optional[Interruption]:
        """Sample whether an interruption occurs."""
        return self.interruption_model.sample(
            self.phase_manager.current_phase,
            self.affect_hmm.current_state,
            persona_multiplier={
                PersonaType.FAST_ACCURATE: 0.6,
                PersonaType.SLOW_METHODICAL: 0.8,
                PersonaType.DISTRACTED_ERROR_PRONE: 2.0,
            }.get(self.persona.type, 1.0),
            dt_s=dt_s,
        )

    def record_error(self) -> None:
        """Record that an error occurred (for affective state modulation)."""
        self._recent_errors.append(True)
        if len(self._recent_errors) > self._error_window_size:
            self._recent_errors.pop(0)

    def should_misclick(self) -> bool:
        """Whether the next click should be a misclick."""
        rate = self.persona.misclick_rate
        rate *= self.PHASE_ERROR_MULT.get(self.phase_manager.current_phase, 1.0)
        rate *= self.AFFECT_ERROR_MULT.get(self.affect_hmm.current_state, 1.0)
        return self._rng.random() < rate

    def should_rotate_camera(self) -> bool:
        """Whether the camera should be rotated at this moment."""
        base_prob = self.persona.camera_rotation_freq
        if self.affect_hmm.current_state == AffectiveState.BORED:
            base_prob *= 1.5
        return self._rng.random() < base_prob

    def should_go_afk(self, dt_s: float = 60.0) -> bool:
        """Whether to simulate AFK/tab-out in the next dt_s seconds."""
        rate_per_s = self.persona.afk_prob_per_min / 60.0
        if self.affect_hmm.current_state == AffectiveState.BORED:
            rate_per_s *= 2.0
        prob = 1.0 - math.exp(-rate_per_s * dt_s)
        return self._rng.random() < prob

    @property
    def current_phase(self) -> SessionPhase:
        return self.phase_manager.current_phase

    @property
    def current_affect(self) -> AffectiveState:
        return self.affect_hmm.current_state

    @property
    def elapsed_seconds(self) -> float:
        return self.phase_manager.elapsed_s

    @property
    def recent_error_rate(self) -> float:
        if not self._recent_errors:
            return 0.0
        return sum(self._recent_errors) / len(self._recent_errors)

    def reset(self) -> None:
        """Reset all sub-models for a new session."""
        self.phase_manager.reset()
        self.affect_hmm.reset()
        self._recent_errors.clear()
        self._update_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _circadian_multiplier(self) -> float:
        """
        Time-of-day multiplier on reaction time.

        Quadratic: minimum at 10:00 (×1.0), maximum at 03:00 (×1.4).
        """
        h = self._hour_of_day
        # Shift so 10:00 is the center
        shifted = (h - 10.0 + 24.0) % 24.0
        # Parabola: 0 at 10:00, max at 03:00 (shifted = 17)
        max_shift = 17.0
        mult = 1.0 + 0.4 * (shifted / max_shift) ** 2
        return mult
