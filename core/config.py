"""
Central configuration management via Pydantic models.

All numeric parameters for every submodule are validated at load time.
Sensible defaults are provided so config files are optional during development.

Usage:
    cfg = AppConfig()                          # all defaults
    cfg = AppConfig.from_yaml("config.yaml")   # load from file
    cfg = AppConfig.from_dict({"mouse": {...}}) # partial override
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PersonaType(str, Enum):
    FAST_ACCURATE = "fast_accurate"
    SLOW_METHODICAL = "slow_methodical"
    DISTRACTED_ERROR_PRONE = "distracted_error_prone"


class SessionPhase(str, Enum):
    WARMUP = "WARMUP"
    GROOVE = "GROOVE"
    PLATEAU = "PLATEAU"
    MICRO_BREAK = "MICRO_BREAK"
    FATIGUE = "FATIGUE"
    RECOVERY = "RECOVERY"


class AffectiveState(str, Enum):
    NEUTRAL = "NEUTRAL"
    FOCUSED = "FOCUSED"
    BORED = "BORED"
    FRUSTRATED = "FRUSTRATED"
    RUSHING = "RUSHING"


class ScenarioState(str, Enum):
    STATE_BANKING = "STATE_BANKING"
    STATE_TRAVELING = "STATE_TRAVELING"
    STATE_OFFERING = "STATE_OFFERING"
    STATE_RETURNING = "STATE_RETURNING"
    STATE_IDLE = "STATE_IDLE"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class FittsParams(BaseModel):
    """Fitts's law coefficients (ms). Published values: a=30-50, b=50-100."""
    a: float = Field(default=40.0, ge=10.0, le=200.0,
                     description="Intercept — minimum movement time (ms)")
    b: float = Field(default=70.0, ge=20.0, le=200.0,
                     description="Slope — ms per bit of difficulty")
    ballistic_ratio: float = Field(default=0.75, ge=0.5, le=0.95,
                                   description="Fraction of movement in ballistic phase")
    corrective_std_px: float = Field(default=3.0, ge=0.5, le=15.0,
                                     description="Endpoint scatter std dev (pixels)")
    overshoot_probability: float = Field(default=0.15, ge=0.0, le=0.5,
                                         description="Probability of overshooting target")
    overshoot_fraction: float = Field(default=0.15, ge=0.05, le=0.4,
                                      description="How far past target overshoot goes (fraction of D)")


class TremorConfig(BaseModel):
    """Physiological tremor injection — 2 frequency bands."""
    band1_hz_min: float = Field(default=8.0, ge=4.0, le=15.0)
    band1_hz_max: float = Field(default=12.0, ge=8.0, le=18.0)
    band1_amplitude_px: float = Field(default=1.2, ge=0.1, le=5.0,
                                      description="Amplitude in pixels at 96 DPI")
    band2_hz_min: float = Field(default=20.0, ge=15.0, le=30.0)
    band2_hz_max: float = Field(default=40.0, ge=30.0, le=50.0)
    band2_amplitude_px: float = Field(default=0.4, ge=0.05, le=2.0)
    gaussian_noise_std: float = Field(default=0.3, ge=0.0, le=2.0,
                                      description="Additional white noise on position")


class SaccadeConfig(BaseModel):
    """Saccadic attention shift and visual search parameters."""
    distracter_probability: float = Field(default=0.12, ge=0.0, le=0.5,
                                          description="Prob of drifting toward distracter before target")
    distracter_angle_deg: float = Field(default=25.0, ge=5.0, le=90.0,
                                        description="Max angular deviation for distracter drift")
    distracter_distance_ratio: float = Field(default=0.4, ge=0.1, le=0.8,
                                             description="How far along the path the distracter appears")
    micro_pause_probability: float = Field(default=0.03, ge=0.0, le=0.15,
                                           description="Brief mid-movement pause probability")
    micro_pause_duration_ms: tuple[float, float] = Field(default=(50.0, 200.0))


class ClickDynamics(BaseModel):
    """Click timing and mechanics."""
    hold_time_mean_ms: float = Field(default=95.0, ge=40.0, le=200.0)
    hold_time_std_ms: float = Field(default=15.0, ge=5.0, le=60.0)
    hold_time_min_ms: float = Field(default=50.0, ge=20.0, le=100.0)
    micro_drift_probability: float = Field(default=0.03, ge=0.0, le=0.10,
                                           description="Prob of cursor drifting during click hold")
    micro_drift_px: float = Field(default=2.5, ge=0.5, le=8.0,
                                  description="Std dev of micro-drift during hold")
    double_click_interval_ms: tuple[float, float] = Field(default=(60.0, 140.0))
    release_asymmetry: float = Field(default=0.3, ge=0.0, le=1.0,
                                     description="How asymmetric the release is (0=symmetric)")


class HIDConfig(BaseModel):
    """HID report pacing — mimics real USB polling."""
    report_rate_hz: int = Field(default=500, ge=125, le=1000,
                                description="Base HID report rate")
    missing_report_probability: float = Field(default=0.005, ge=0.0, le=0.05,
                                              description="Probability of a dropped HID report")
    timestamp_jitter_ms: float = Field(default=1.5, ge=0.0, le=10.0,
                                       description="Gaussian jitter on report timestamps")
    inter_report_jitter_ratio: float = Field(default=0.15, ge=0.0, le=0.5,
                                             description="Jitter as fraction of base interval")


class MouseConfig(BaseModel):
    """Full mouse motor control configuration."""
    fitts: FittsParams = Field(default_factory=FittsParams)
    tremor: TremorConfig = Field(default_factory=TremorConfig)
    saccade: SaccadeConfig = Field(default_factory=SaccadeConfig)
    click: ClickDynamics = Field(default_factory=ClickDynamics)
    hid: HIDConfig = Field(default_factory=HIDConfig)
    misclick_probability: float = Field(default=0.025, ge=0.0, le=0.10,
                                        description="Prob of clicking off-target")
    misclick_offset_px: float = Field(default=35.0, ge=10.0, le=80.0,
                                      description="Std dev of misclick offset")


class ScanConfig(BaseModel):
    """Visual perception (color-bot) configuration."""
    min_confidence_hsv: float = Field(default=0.65, ge=0.3, le=0.95)
    min_confidence_template: float = Field(default=0.70, ge=0.3, le=0.95)
    hsv_hue_tolerance: int = Field(default=10, ge=2, le=30,
                                   description="± degrees on H channel for HSV matching")
    hsv_sat_tolerance: int = Field(default=40, ge=10, le=100)
    hsv_val_tolerance: int = Field(default=40, ge=10, le=100)
    noise_brightness_std: float = Field(default=5.0, ge=0.0, le=20.0,
                                        description="Std dev of brightness jitter (%)")
    noise_hue_shift_deg: float = Field(default=2.0, ge=0.0, le=10.0,
                                       description="Max random hue shift (degrees)")
    noise_blur_kernel: int = Field(default=2, ge=0, le=5,
                                   description="Max Gaussian blur kernel size (px)")
    distractor_count: int = Field(default=3, ge=1, le=6)
    distractor_time_ms: tuple[float, float] = Field(default=(150.0, 400.0))
    hover_dwell_min_s: float = Field(default=1.5, ge=0.5, le=5.0)
    hover_dwell_max_s: float = Field(default=3.0, ge=1.0, le=8.0)
    hover_micro_gesture_radius: float = Field(default=8.0, ge=2.0, le=20.0)
    ocr_fallback_enabled: bool = Field(default=False)
    synthetic_position_noise_px: float = Field(default=3.0, ge=0.0, le=15.0,
                                               description="Gaussian noise on synthetic positions")


class ExWaldParams(BaseModel):
    """Ex-Wald (inverse Gaussian + exponential) reaction time distribution."""
    mu: float = Field(default=220.0, ge=50.0, le=500.0,
                      description="Wald distribution mean (ms)")
    sigma: float = Field(default=40.0, ge=10.0, le=150.0,
                         description="Wald distribution shape")
    lambda_: float = Field(default=0.015, ge=0.001, le=0.1,
                           description="Exponential tail rate", alias="lambda")
    rt_min_ms: float = Field(default=50.0, ge=20.0, le=200.0)
    rt_max_ms: float = Field(default=5000.0, ge=1000.0, le=30000.0)


class PersonaParams(BaseModel):
    """Parameters defining a specific user persona."""
    persona_type: PersonaType
    exwald: ExWaldParams = Field(default_factory=ExWaldParams)
    hold_time_mean_ms: float = Field(default=95.0, ge=40.0, le=200.0)
    hold_time_std_ms: float = Field(default=15.0, ge=5.0, le=60.0)
    typo_rate: float = Field(default=0.01, ge=0.0, le=0.10)
    micro_break_frequency_per_min: float = Field(default=0.4, ge=0.0, le=2.0)
    fatigue_rt_multiplier: float = Field(default=1.2, ge=1.0, le=2.0,
                                         description="RT multiplier at peak fatigue")
    distracter_probability: float = Field(default=0.12, ge=0.0, le=0.5)
    misclick_rate: float = Field(default=0.025, ge=0.0, le=0.10)
    tremor_amplitude_scale: float = Field(default=1.0, ge=0.5, le=3.0,
                                          description="Multiplier on base tremor amplitudes")
    camera_rotation_frequency: float = Field(default=0.15, ge=0.0, le=0.5,
                                             description="Prob of rotating camera per state transition")
    afk_probability_per_minute: float = Field(default=0.02, ge=0.0, le=0.1)


class PhaseTiming(BaseModel):
    """Session phase transition timing (seconds from session start)."""
    warmup_duration_s: float = Field(default=180.0, ge=30.0, le=600.0)
    groove_duration_s: float = Field(default=720.0, ge=120.0, le=1800.0)
    plateau_duration_s: float = Field(default=900.0, ge=300.0, le=3600.0)
    micro_break_interval_s: tuple[float, float] = Field(default=(120.0, 300.0))
    micro_break_duration_s: tuple[float, float] = Field(default=(3.0, 8.0))
    recovery_duration_s: tuple[float, float] = Field(default=(30.0, 60.0))
    fatigue_onset_s: float = Field(default=1800.0, ge=600.0, le=7200.0)


class HMMTransitionMatrix(BaseModel):
    """5×5 transition matrix for affective states (rows sum to 1)."""
    neutral: list[float] = Field(default=[0.70, 0.15, 0.08, 0.04, 0.03],
                                 description="From NEUTRAL to [N, F, B, FR, RU]")
    focused: list[float] = Field(default=[0.05, 0.80, 0.05, 0.05, 0.05])
    bored: list[float] = Field(default=[0.10, 0.10, 0.65, 0.10, 0.05])
    frustrated: list[float] = Field(default=[0.05, 0.10, 0.10, 0.65, 0.10])
    rushing: list[float] = Field(default=[0.15, 0.10, 0.05, 0.10, 0.60])

    @field_validator("*")
    @classmethod
    def rows_sum_to_one(cls, v: list[float]) -> list[float]:
        if abs(sum(v) - 1.0) > 0.05:
            raise ValueError(f"Transition row must sum to ~1.0, got {sum(v)}")
        return v

    def as_matrix(self) -> list[list[float]]:
        return [self.neutral, self.focused, self.bored, self.frustrated, self.rushing]


class CognitiveConfig(BaseModel):
    """Cognitive delay engine configuration."""
    personas: dict[str, PersonaParams] = Field(default_factory=dict)
    phases: PhaseTiming = Field(default_factory=PhaseTiming)
    hmm_default: HMMTransitionMatrix = Field(default_factory=HMMTransitionMatrix)
    circadian_enabled: bool = Field(default=True)


class InterKeyDelayParams(BaseModel):
    """Two-component log-normal mixture for inter-key delays."""
    component_weights: list[float] = Field(default=[0.65, 0.35])
    component_means_ms: list[float] = Field(default=[140.0, 450.0],
                                            description="Log-space means for fast and slow modes")
    component_stds_ms: list[float] = Field(default=[40.0, 150.0])


class TypoConfig(BaseModel):
    """Typo generation parameters."""
    base_rate: float = Field(default=0.015, ge=0.0, le=0.10)
    transposition_ratio: float = Field(default=0.40, ge=0.0, le=1.0)
    adjacency_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    skip_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    correction_overshoot_mean: float = Field(default=1.5, ge=0.0, le=5.0,
                                             description="Mean extra backspaces during correction")
    correction_speed_ratio: float = Field(default=0.70, ge=0.3, le=1.0,
                                          description="Correction typing speed relative to normal")


class KeyboardConfig(BaseModel):
    """Keyboard synthesis configuration."""
    inter_key: InterKeyDelayParams = Field(default_factory=InterKeyDelayParams)
    typo: TypoConfig = Field(default_factory=TypoConfig)
    modifier_overlap_ms: float = Field(default=35.0, ge=10.0, le=100.0,
                                       description="Modifier key held after target key (ms)")


class InterruptionConfig(BaseModel):
    """Real-world interruption parameters."""
    base_rate_per_minute: float = Field(default=0.15, ge=0.0, le=2.0)
    cursor_freeze_ratio: float = Field(default=0.40, ge=0.0, le=1.0)
    gaze_drift_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    keystroke_burst_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    cursor_freeze_duration_s: tuple[float, float] = Field(default=(2.0, 15.0))
    gaze_drift_duration_s: tuple[float, float] = Field(default=(1.0, 3.0))
    keystroke_burst_keys: tuple[int, int] = Field(default=(3, 20))
    keystroke_burst_duration_s: tuple[float, float] = Field(default=(5.0, 30.0))


class SchedulerConfig(BaseModel):
    """Session and scenario scheduler configuration."""
    session_count_per_run: int = Field(default=10, ge=1, le=10000)
    persona_weights: dict[str, float] = Field(
        default={"fast_accurate": 0.35, "slow_methodical": 0.35, "distracted_error_prone": 0.30}
    )
    anomaly_short_checkin_prob: float = Field(default=0.10, ge=0.0, le=0.5)
    anomaly_skip_prob: float = Field(default=0.05, ge=0.0, le=0.3)
    memory_file_path: str = Field(default="data/session_memory.json")
    max_session_history: int = Field(default=20, ge=5, le=100)
    avoid_recent_scenarios: int = Field(default=5, ge=1, le=10)
    avoid_recent_personas: int = Field(default=3, ge=1, le=5)


class ScenarioSpecificConfig(BaseModel):
    """Scenario-specific overrides."""
    state_duration_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    template_ids: dict[str, list[str]] = Field(default_factory=dict)
    distractor_count_override: Optional[int] = Field(default=None)
    cycle_count: int = Field(default=10, ge=1, le=1000)
    bones_per_inventory: int = Field(default=28, ge=1, le=28)
    bone_save_probability: float = Field(default=0.50, ge=0.0, le=1.0)
    camera_rotation_prob_during_travel: float = Field(default=0.15, ge=0.0, le=0.5)


class AppConfig(BaseModel):
    """Top-level configuration container."""
    mouse: MouseConfig = Field(default_factory=MouseConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    keyboard: KeyboardConfig = Field(default_factory=KeyboardConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    interruption: InterruptionConfig = Field(default_factory=InterruptionConfig)
    scenario: ScenarioSpecificConfig = Field(default_factory=ScenarioSpecificConfig)

    data_dir: str = Field(default="data")
    output_dir: str = Field(default="output")
    random_seed: Optional[int] = Field(default=None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Load configuration from a dictionary (partial overrides allowed)."""
        # Flatten nested keys so users can provide partial overrides
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Default persona presets — research-informed values
# ---------------------------------------------------------------------------

DEFAULT_PERSONAS: dict[str, PersonaParams] = {
    "fast_accurate": PersonaParams(
        persona_type=PersonaType.FAST_ACCURATE,
        exwald=ExWaldParams(mu=180.0, sigma=30.0, lambda_=0.02),
        hold_time_mean_ms=90.0,
        hold_time_std_ms=10.0,
        typo_rate=0.008,
        micro_break_frequency_per_min=0.3,
        fatigue_rt_multiplier=1.10,
        distracter_probability=0.08,
        misclick_rate=0.015,
        tremor_amplitude_scale=0.8,
        camera_rotation_frequency=0.10,
        afk_probability_per_minute=0.01,
    ),
    "slow_methodical": PersonaParams(
        persona_type=PersonaType.SLOW_METHODICAL,
        exwald=ExWaldParams(mu=350.0, sigma=60.0, lambda_=0.008),
        hold_time_mean_ms=120.0,
        hold_time_std_ms=20.0,
        typo_rate=0.005,
        micro_break_frequency_per_min=0.5,
        fatigue_rt_multiplier=1.05,
        distracter_probability=0.05,
        misclick_rate=0.01,
        tremor_amplitude_scale=1.2,
        camera_rotation_frequency=0.20,
        afk_probability_per_minute=0.03,
    ),
    "distracted_error_prone": PersonaParams(
        persona_type=PersonaType.DISTRACTED_ERROR_PRONE,
        exwald=ExWaldParams(mu=250.0, sigma=80.0, lambda_=0.03),
        hold_time_mean_ms=100.0,
        hold_time_std_ms=25.0,
        typo_rate=0.03,
        micro_break_frequency_per_min=0.8,
        fatigue_rt_multiplier=1.30,
        distracter_probability=0.25,
        misclick_rate=0.05,
        tremor_amplitude_scale=1.5,
        camera_rotation_frequency=0.25,
        afk_probability_per_minute=0.05,
    ),
}
