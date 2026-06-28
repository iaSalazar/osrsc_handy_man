"""Tests for the cognitive delay engine."""

import numpy as np
import pytest

from core.config import PersonaType, CognitiveConfig, DEFAULT_PERSONAS
from core.cognitive import (
    CognitiveEngine, UserPersonaModel, SessionPhaseManager,
    AffectiveStateMachine, SessionPhase, AffectiveState,
)


class TestUserPersonaModel:
    def test_load_fast_accurate(self):
        rng = np.random.default_rng(42)
        persona = UserPersonaModel(PersonaType.FAST_ACCURATE, rng=rng)
        assert persona.type == PersonaType.FAST_ACCURATE
        assert persona.typo_rate < 0.02
        assert persona.misclick_rate < 0.03

    def test_load_slow_methodical(self):
        rng = np.random.default_rng(42)
        persona = UserPersonaModel(PersonaType.SLOW_METHODICAL, rng=rng)
        assert persona.type == PersonaType.SLOW_METHODICAL
        assert persona.typo_rate < 0.01
        assert persona.misclick_rate < 0.02

    def test_load_distracted(self):
        rng = np.random.default_rng(42)
        persona = UserPersonaModel(PersonaType.DISTRACTED_ERROR_PRONE, rng=rng)
        assert persona.type == PersonaType.DISTRACTED_ERROR_PRONE
        assert persona.typo_rate > 0.02
        assert persona.misclick_rate > 0.03

    def test_rt_samples_positive(self):
        rng = np.random.default_rng(42)
        persona = UserPersonaModel(PersonaType.FAST_ACCURATE, rng=rng)
        for _ in range(100):
            rt = persona.rt_dist.sample()
            assert rt > 0
            assert rt < 10000  # well within max


class TestSessionPhaseManager:
    def test_initial_phase(self):
        rng = np.random.default_rng(42)
        from core.config import PhaseTiming
        spm = SessionPhaseManager(PhaseTiming(), rng)
        assert spm.current_phase == SessionPhase.WARMUP

    def test_transitions_to_groove(self):
        rng = np.random.default_rng(42)
        from core.config import PhaseTiming
        # Use valid values and advance past warmup
        pt = PhaseTiming(warmup_duration_s=30.0, groove_duration_s=120.0,
                         plateau_duration_s=300.0, fatigue_onset_s=600.0)
        spm = SessionPhaseManager(pt, rng)
        # Still in warmup
        assert spm.current_phase == SessionPhase.WARMUP
        spm.update(0.5)  # tiny time
        assert spm.current_phase == SessionPhase.WARMUP
        spm.update(30.0)  # past warmup
        assert spm.current_phase in (SessionPhase.GROOVE, SessionPhase.PLATEAU)

    def test_fatigue_eventually(self):
        rng = np.random.default_rng(42)
        from core.config import PhaseTiming
        pt = PhaseTiming(warmup_duration_s=30.0, groove_duration_s=120.0,
                         plateau_duration_s=300.0, fatigue_onset_s=600.0,
                         micro_break_interval_s=(9999.0, 99999.0))  # no micro-breaks
        spm = SessionPhaseManager(pt, rng)
        spm.update(700.0)  # past fatigue onset
        assert spm.current_phase == SessionPhase.FATIGUE


class TestCognitiveEngine:
    def test_reaction_delay_positive(self):
        rng = np.random.default_rng(42)
        ce = CognitiveEngine(PersonaType.FAST_ACCURATE, rng=rng)
        ce.update(60.0)  # advance well into GROOVE
        for _ in range(50):
            rt = ce.get_reaction_delay_ms()
            assert rt > 20
            assert rt < 10000

    def test_phase_progression(self):
        rng = np.random.default_rng(42)
        ce = CognitiveEngine(PersonaType.SLOW_METHODICAL, rng=rng)
        assert ce.current_phase == SessionPhase.WARMUP
        ce.update(300.0)  # past warmup (180s)
        assert ce.current_phase in (SessionPhase.GROOVE, SessionPhase.PLATEAU)

    def test_affect_starts_neutral(self):
        rng = np.random.default_rng(42)
        ce = CognitiveEngine(PersonaType.FAST_ACCURATE, rng=rng)
        assert ce.current_affect == AffectiveState.NEUTRAL

    def test_should_misclick_rate(self):
        rng = np.random.default_rng(42)
        ce = CognitiveEngine(PersonaType.DISTRACTED_ERROR_PRONE, rng=rng)
        # Over many trials, misclick rate should be roughly persona rate
        trials = 1000
        misclicks = sum(1 for _ in range(trials) if ce.should_misclick())
        rate = misclicks / trials
        # Distracted persona has ~5% base misclick rate
        assert 0.01 < rate < 0.15  # wide range due to randomness

    def test_should_rotate_camera(self):
        rng = np.random.default_rng(42)
        ce = CognitiveEngine(PersonaType.FAST_ACCURATE, rng=rng)
        # Just verify it returns a bool without error
        result = ce.should_rotate_camera()
        assert isinstance(result, bool)
