"""
Motor Control Engine — realistic human mouse movement synthesis.

Models the full pipeline from Fittsʼs law ballistic movement through
physiological tremor injection to click dynamics and HID report pacing.

Architecture
------------
FittsLaw (static math)
  └─ TremorEngine (per-sample additive noise, 8–12 Hz + 20–40 Hz)
       └─ SaccadeModel (ballistic → distracter → corrective waypoint generation)
            └─ ClickModel (log-normal hold times, micro-drift, asymmetric release)
                 └─ HIDPacer (125–1000 Hz report timing with jitter & drops)
                      └─ MouseEmulator (public facade)

References
----------
- Fitts, P. M. (1954). "The information capacity of the human motor system
  in controlling the amplitude of movement."
- Flash, T. & Hogan, N. (1985). "The coordination of arm movements: an
  experimentally confirmed mathematical model."  (minimum-jerk)
- Pointergeist/PHC-mouse-movement-gen — sigma log-normal model
- Pydoll mouse humanization — Bezier + Fitts + tremor
- Silphe — tremor 4–12 Hz, smooth-pursuit, corrective sub-movements
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
from scipy import stats

from core.config import MouseConfig, FittsParams, TremorConfig, SaccadeConfig
from core.config import ClickDynamics, HIDConfig


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MouseSample:
    """A single cursor position sample at HID-report resolution."""
    timestamp: float          # seconds from movement start
    x: float                  # absolute screen x
    y: float                  # absolute screen y
    velocity: float           # px/s
    acceleration: float       # px/s²
    jerk: float               # px/s³
    phase: str                # "ballistic" | "corrective" | "tremor" | "dwell" | "distracter"


@dataclass
class Waypoint:
    """A key point along a movement trajectory."""
    x: float
    y: float
    phase: str                # "start" | "distracter" | "overshoot" | "target" | "corrective_end"
    time_fraction: float      # 0.0 to 1.0 — when this waypoint is reached


@dataclass
class ClickEvent:
    """A complete click action with realistic dynamics."""
    x: float
    y: float
    button: str               # "left" | "right" | "middle"
    down_timestamp: float
    up_timestamp: float
    hold_duration_ms: float
    micro_drift_px: float     # total displacement during hold
    pre_click_x: float        # position just before click
    pre_click_y: float
    post_click_x: float       # position just after release
    post_click_y: float
    samples: list[MouseSample] = field(default_factory=list)
    is_misclick: bool = False


# ---------------------------------------------------------------------------
# 1. Fitts's Law
# ---------------------------------------------------------------------------

class FittsLaw:
    """
    Fitts's law: MT = a + b * log₂(1 + D/W)

    Where:
        MT = movement time (ms)
        D  = distance to target (px)
        W  = target width (px)
        a  = intercept (minimum reaction + mechanical time)
        b  = slope (inverse of information processing rate)

    The effective width correction (Shannon formulation) accounts for
    actual endpoint spread — narrower spread = effectively wider target.
    """

    def __init__(self, params: FittsParams, rng: np.random.Generator) -> None:
        self.a = params.a
        self.b = params.b
        self.ballistic_ratio = params.ballistic_ratio
        self.corrective_std_px = params.corrective_std_px
        self.overshoot_probability = params.overshoot_probability
        self.overshoot_fraction = params.overshoot_fraction
        self._rng = rng

    def movement_time_ms(self, distance_px: float, target_width_px: float) -> float:
        """Return predicted movement time in milliseconds."""
        if distance_px < 0.5:
            return self.a  # negligible distance — just reaction time
        # Clamp W to avoid division by zero or negative log
        w = max(target_width_px, 1.0)
        index_of_difficulty = math.log2(1.0 + distance_px / w)
        mt = self.a + self.b * index_of_difficulty
        # Add small Gaussian jitter (±5%) to avoid perfectly deterministic timing
        mt *= self._rng.normal(1.0, 0.025)
        return max(mt, self.a * 0.8)

    def ballistic_time_ms(self, distance_px: float, target_width_px: float) -> float:
        """Duration of the ballistic (fast, less accurate) phase."""
        total = self.movement_time_ms(distance_px, target_width_px)
        return total * self.ballistic_ratio

    def corrective_time_ms(self, distance_px: float, target_width_px: float) -> float:
        """Duration of the corrective (slow, precise) phase."""
        total = self.movement_time_ms(distance_px, target_width_px)
        return total * (1.0 - self.ballistic_ratio)

    def endpoint_noise(self, target_x: float, target_y: float) -> tuple[float, float]:
        """Sample final endpoint from 2D Gaussian centered on target."""
        ex = self._rng.normal(target_x, self.corrective_std_px)
        ey = self._rng.normal(target_y, self.corrective_std_px)
        return ex, ey

    def should_overshoot(self) -> bool:
        """Whether this particular movement overshoots the target."""
        return self._rng.random() < self.overshoot_probability

    def overshoot_point(
        self, from_x: float, from_y: float, to_x: float, to_y: float,
    ) -> tuple[float, float]:
        """Compute a point beyond the target for the overshoot."""
        dx = to_x - from_x
        dy = to_y - from_y
        return (
            to_x + dx * self.overshoot_fraction,
            to_y + dy * self.overshoot_fraction,
        )


# ---------------------------------------------------------------------------
# 2. Tremor Engine
# ---------------------------------------------------------------------------

class TremorEngine:
    """
    Multi-frequency physiological tremor injection.

    Produces pixel-level additive noise from two frequency bands:
      - Band 1: 8–12 Hz  (physiological tremor — main component)
      - Band 2: 20–40 Hz (harmonics / enhanced physiological tremor)

    Each band uses a sinusoidal oscillator with randomised phase and
    small amplitude modulation over time to avoid pure-periodic artifacts.
    """

    def __init__(self, config: TremorConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng

        # Randomise actual frequencies within configured bands
        self._f1 = rng.uniform(config.band1_hz_min, config.band1_hz_max)
        self._f2 = rng.uniform(config.band2_hz_min, config.band2_hz_max)
        self._amp1 = config.band1_amplitude_px
        self._amp2 = config.band2_amplitude_px
        self._noise_std = config.gaussian_noise_std

        # Random phase offsets so each movement starts at a different tremor phase
        self._phase1 = rng.uniform(0.0, 2.0 * math.pi)
        self._phase2 = rng.uniform(0.0, 2.0 * math.pi)

        # Amplitude modulation — slow drift of tremor intensity
        self._mod_phase1 = rng.uniform(0.0, 2.0 * math.pi)
        self._mod_phase2 = rng.uniform(0.0, 2.0 * math.pi)

    def apply(self, x: float, y: float, t: float) -> tuple[float, float]:
        """
        Apply tremor displacement at time `t` (seconds from movement start).

        Returns (x + dx, y + dy).
        """
        # Amplitude modulation — slow 0.5–1.5 Hz modulation
        mod1 = 0.7 + 0.3 * math.sin(2.0 * math.pi * 0.7 * t + self._mod_phase1)
        mod2 = 0.7 + 0.3 * math.sin(2.0 * math.pi * 1.1 * t + self._mod_phase2)

        # Band 1: ~8–12 Hz, horizontal and vertical independently phased
        dx1 = self._amp1 * mod1 * math.sin(2.0 * math.pi * self._f1 * t + self._phase1)
        dy1 = self._amp1 * mod1 * math.cos(2.0 * math.pi * self._f1 * t + self._phase1 + 0.7)

        # Band 2: ~20–40 Hz, smaller amplitude, slightly different axis
        dx2 = self._amp2 * mod2 * math.sin(2.0 * math.pi * self._f2 * t + self._phase2)
        dy2 = self._amp2 * mod2 * math.cos(2.0 * math.pi * self._f2 * t + self._phase2 - 0.5)

        # White noise floor
        dx_noise = self._rng.normal(0.0, self._noise_std)
        dy_noise = self._rng.normal(0.0, self._noise_std)

        return (x + dx1 + dx2 + dx_noise, y + dy1 + dy2 + dy_noise)


# ---------------------------------------------------------------------------
# 3. Saccade Model — trajectory generation
# ---------------------------------------------------------------------------

class SaccadeModel:
    """
    Multi-phase mouse trajectory generator.

    Phase 1 — Ballistic:
        High initial velocity, bell-shaped (minimum-jerk) velocity profile.
        Power-law acceleration, skewed right (acceleration faster than deceleration).

    Phase 2 — Distracter (probabilistic):
        Cursor drifts toward a random peripheral location, then corrects
        back toward target.  Models saccadic attention shift — the eyes
        briefly flick to a distracter before fixating on the target.

    Phase 3 — Corrective:
        Low-velocity approach once within ~20 px of target.
        Sub-movement refinement with endpoint scatter.

    The trajectory is represented as a set of Waypoints; interpolation
    between waypoints uses cubic Bezier curves for smooth, natural curves.
    """

    def __init__(self, config: SaccadeConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng

    def generate(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        target_width: float,
        overshoot: bool = False,
        overshoot_point: Optional[tuple[float, float]] = None,
    ) -> list[Waypoint]:
        """
        Generate waypoints for a single movement.

        Parameters
        ----------
        from_x, from_y : Start position.
        to_x, to_y : Target center.
        target_width : Target width in pixels (for Fitts endpoint scatter).
        overshoot : Whether to overshoot the target before correcting.
        overshoot_point : Explicit overshoot coordinates (from FittsLaw).

        Returns
        -------
        List of Waypoints in time order.
        """
        waypoints: list[Waypoint] = []
        tf = 0.0  # cumulative time fraction

        # --- Waypoint 0: start ---
        waypoints.append(Waypoint(from_x, from_y, "start", tf))

        dx = to_x - from_x
        dy = to_y - from_y
        dist = math.hypot(dx, dy)

        # --- Distracter drift (probabilistic) ---
        has_distracter = self._rng.random() < self.config.distracter_probability
        distracter_wp: Optional[Waypoint] = None

        if has_distracter and dist > 50:
            # Pick a random angular deviation (±distracter_angle_deg)
            base_angle = math.atan2(dy, dx)
            deviation = math.radians(self._rng.uniform(
                -self.config.distracter_angle_deg, self.config.distracter_angle_deg
            ))
            distracter_angle = base_angle + deviation
            distracter_dist = dist * self.config.distracter_distance_ratio
            dx_dist = from_x + distracter_dist * math.cos(distracter_angle)
            dy_dist = from_y + distracter_dist * math.sin(distracter_angle)
            tf += self._rng.uniform(0.25, 0.45)
            distracter_wp = Waypoint(dx_dist, dy_dist, "distracter", tf)
            waypoints.append(distracter_wp)

        # --- Overshoot (probabilistic) ---
        overshoot_wp: Optional[Waypoint] = None
        if overshoot and overshoot_point:
            tf += self._rng.uniform(0.15, 0.25)
            overshoot_wp = Waypoint(
                overshoot_point[0], overshoot_point[1], "overshoot", tf
            )
            waypoints.append(overshoot_wp)

        # --- Target approach ---
        tf = min(tf + self._rng.uniform(0.3, 0.5), 0.95)
        waypoints.append(Waypoint(to_x, to_y, "target", tf))

        # --- Corrective sub-movement endpoint (Gaussian scatter) ---
        tf = 1.0
        # Scatter std proportional to target width, clamped
        scatter_std = max(target_width * 0.08, 1.5)
        final_x = self._rng.normal(to_x, scatter_std)
        final_y = self._rng.normal(to_y, scatter_std)
        waypoints.append(Waypoint(final_x, final_y, "corrective_end", tf))

        return waypoints

    def waypoints_to_bezier_curve(
        self,
        waypoints: list[Waypoint],
        num_samples: int,
    ) -> list[tuple[float, float, float]]:
        """
        Convert waypoints to smooth cubic Bezier-interpolated samples.

        Returns list of (x, y, time_fraction).
        """
        if len(waypoints) < 2:
            return [(waypoints[0].x, waypoints[0].y, 0.0)]

        samples: list[tuple[float, float, float]] = []

        for i in range(len(waypoints) - 1):
            wp_a = waypoints[i]
            wp_b = waypoints[i + 1]
            t_start = wp_a.time_fraction
            t_end = wp_b.time_fraction
            segment_duration = t_end - t_start
            if segment_duration <= 0:
                continue

            n_seg = max(int(num_samples * segment_duration), 3)

            # Control points for cubic Bezier — create natural curvature
            # by offsetting control points perpendicular to the direct path
            dx = wp_b.x - wp_a.x
            dy = wp_b.y - wp_a.y
            seg_dist = math.hypot(dx, dy) or 1.0

            # Perpendicular unit vector
            perp_x = -dy / seg_dist
            perp_y = dx / seg_dist

            # Control point offset — scales with distance and randomness
            cp_offset = seg_dist * self._rng.uniform(0.1, 0.35)
            cp_sign = 1.0 if self._rng.random() > 0.5 else -1.0

            cp1_x = wp_a.x + dx * 0.25 + perp_x * cp_offset * cp_sign
            cp1_y = wp_a.y + dy * 0.25 + perp_y * cp_offset * cp_sign
            cp2_x = wp_a.x + dx * 0.75 - perp_x * cp_offset * cp_sign * 0.7
            cp2_y = wp_a.y + dy * 0.75 - perp_y * cp_offset * cp_sign * 0.7

            # Evaluate cubic Bezier: B(t) = (1-t)³P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³P₃
            for j in range(n_seg):
                t_local = j / (n_seg - 1) if n_seg > 1 else 0.0
                t_global = t_start + t_local * segment_duration

                t_ = t_local
                mt = 1.0 - t_
                mt2 = mt * mt
                mt3 = mt2 * mt
                t2 = t_ * t_
                t3 = t2 * t_

                bx = mt3 * wp_a.x + 3.0 * mt2 * t_ * cp1_x + 3.0 * mt * t2 * cp2_x + t3 * wp_b.x
                by = mt3 * wp_a.y + 3.0 * mt2 * t_ * cp1_y + 3.0 * mt * t2 * cp2_y + t3 * wp_b.y

                samples.append((bx, by, t_global))

        return samples


# ---------------------------------------------------------------------------
# 4. Click Dynamics
# ---------------------------------------------------------------------------

class ClickModel:
    """
    Realistic mouse click generation.

    Features:
      - Log-normal hold time distribution (80–120 ms mean typical)
      - Asymmetric release (finger lifts at different speed than press)
      - Probabilistic micro-drift during hold (cursor slides 1–5 px)
      - Pre-click micro-pause (cursor stabilizes briefly before click)
    """

    def __init__(self, config: ClickDynamics, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng
        # Pre-compute log-normal parameters from mean/std
        self._hold_sigma = math.sqrt(
            math.log(1.0 + (config.hold_time_std_ms / config.hold_time_mean_ms) ** 2)
        )
        self._hold_mu = math.log(config.hold_time_mean_ms) - 0.5 * self._hold_sigma ** 2

    def sample_hold_time_ms(self) -> float:
        """Sample click hold duration from log-normal distribution."""
        hold = self._rng.lognormal(self._hold_mu, self._hold_sigma)
        return max(hold, self.config.hold_time_min_ms)

    def generate(
        self,
        x: float,
        y: float,
        button: str = "left",
        movement_start_time: float = 0.0,
        is_misclick: bool = False,
    ) -> ClickEvent:
        """
        Generate a complete click event with realistic timing and dynamics.

        Parameters
        ----------
        x, y : Click location.
        button : Mouse button.
        movement_start_time : Base timestamp for the movement this click ends.
        is_misclick : Whether this click is intentionally off-target.

        Returns
        -------
        ClickEvent with all timing and drift data.
        """
        hold_ms = self.sample_hold_time_ms()
        hold_s = hold_ms / 1000.0

        # Micro-drift during hold
        has_drift = self._rng.random() < self.config.micro_drift_probability
        drift_px = 0.0
        drift_x = x
        drift_y = y

        if has_drift:
            drift_px = abs(self._rng.normal(0.0, self.config.micro_drift_px))
            drift_angle = self._rng.uniform(0.0, 2.0 * math.pi)
            drift_x = x + drift_px * math.cos(drift_angle)
            drift_y = y + drift_px * math.sin(drift_angle)

        # Asymmetric release: cursor moves slightly after release
        release_drift = self._rng.normal(0.0, drift_px * self.config.release_asymmetry)

        # Timestamps
        now = movement_start_time
        pre_pause_ms = self._rng.uniform(20.0, 60.0)  # stabilization before click
        down_ts = now + pre_pause_ms / 1000.0
        up_ts = down_ts + hold_s

        return ClickEvent(
            x=x, y=y,
            button=button,
            down_timestamp=down_ts,
            up_timestamp=up_ts,
            hold_duration_ms=hold_ms,
            micro_drift_px=drift_px,
            pre_click_x=x, pre_click_y=y,
            post_click_x=x + release_drift, post_click_y=y + release_drift,
            is_misclick=is_misclick,
        )


# ---------------------------------------------------------------------------
# 5. HID Report Pacer
# ---------------------------------------------------------------------------

class HIDPacer:
    """
    Realistic USB HID report timing.

    Mimics the non-uniform polling of real hardware:
      - Base interval determined by report rate (e.g. 2 ms at 500 Hz).
      - Inter-report intervals exponentially distributed around the base.
      - Gaussian timestamp jitter.
      - Bernoulli missing reports (~0.5%).
    """

    def __init__(self, config: HIDConfig, rng: np.random.Generator) -> None:
        self.base_interval_s = 1.0 / config.report_rate_hz
        self.missing_prob = config.missing_report_probability
        self.jitter_std_s = config.timestamp_jitter_ms / 1000.0
        self.inter_jitter_ratio = config.inter_report_jitter_ratio
        self._rng = rng

    def generate_timestamps(self, duration_s: float) -> np.ndarray:
        """
        Generate HID report timestamps covering `duration_s` seconds.

        Returns array of timestamps (seconds from 0).
        """
        if duration_s <= 0:
            return np.array([0.0])

        # Expected number of reports
        n_expected = max(int(duration_s / self.base_interval_s), 2)

        # Inter-report intervals from exponential distribution
        # Scale so mean equals base_interval
        intervals = self._rng.exponential(self.base_interval_s, size=n_expected)

        # Add Gaussian jitter
        intervals += self._rng.normal(0.0, self.jitter_std_s, size=n_expected)
        intervals = np.clip(intervals, self.base_interval_s * 0.2, self.base_interval_s * 3.0)

        # Cumulative timestamps
        timestamps = np.cumsum(intervals)

        # Trim to duration
        timestamps = timestamps[timestamps <= duration_s]

        # Randomly drop reports
        if len(timestamps) > 2:
            drop_mask = self._rng.random(len(timestamps)) < self.missing_prob
            # Never drop first or last
            drop_mask[0] = False
            if len(timestamps) > 1:
                drop_mask[-1] = False
            timestamps = timestamps[~drop_mask]

        return timestamps


# ---------------------------------------------------------------------------
# 6. Mouse Emulator — public facade
# ---------------------------------------------------------------------------

class MouseEmulator:
    """
    Public API for generating realistic mouse input.

    Wires together FittsLaw, TremorEngine, SaccadeModel, ClickModel,
    and HIDPacer into a single `move_to()` / `click_at()` / `drag()` interface.

    Tracks an internal `current_position` so callers don't need to manage
    state across calls.
    """

    def __init__(
        self,
        config: MouseConfig,
        rng: np.random.Generator,
        persona_amplitude_scale: float = 1.0,
        persona_misclick_rate: float = 0.025,
    ) -> None:
        self.config = config
        self._rng = rng

        # Sub-engines
        self.fitts = FittsLaw(config.fitts, rng)
        self.tremor = TremorEngine(config.tremor, rng)
        self.saccade = SaccadeModel(config.saccade, rng)
        self.click_model = ClickModel(config.click, rng)
        self.hid = HIDPacer(config.hid, rng)

        # Persona-driven modulation
        self._persona_amplitude_scale = persona_amplitude_scale
        self._persona_misclick_rate = persona_misclick_rate

        # Internal state
        self._x: float = 0.0
        self._y: float = 0.0
        self._time: float = 0.0  # cumulative movement time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_position(self) -> tuple[float, float]:
        return (self._x, self._y)

    def reset_position(self, x: float, y: float) -> None:
        """Set cursor position without generating events."""
        self._x = x
        self._y = y

    def move_to(
        self,
        to_x: float,
        to_y: float,
        target_width: float | None = None,
        from_x: float | None = None,
        from_y: float | None = None,
    ) -> list[MouseSample]:
        """
        Generate a full human-like mouse movement from current position to (to_x, to_y).

        Parameters
        ----------
        to_x, to_y : Target coordinates.
        target_width : Width of the target element in pixels (for Fitts scaling).
        from_x, from_y : Override start position (default: current internal position).

        Returns
        -------
        List of MouseSample at HID report resolution.
        """
        if from_x is not None:
            self._x = from_x
        if from_y is not None:
            self._y = from_y

        start_x, start_y = self._x, self._y
        dx = to_x - start_x
        dy = to_y - start_y
        distance = math.hypot(dx, dy)

        # Negligible movement — just return current position with slight tremor
        if distance < 2.0:
            return [MouseSample(
                timestamp=self._time, x=to_x, y=to_y,
                velocity=0.0, acceleration=0.0, jerk=0.0, phase="dwell",
            )]

        tw = target_width if target_width is not None else 20.0

        # Movement timing from Fitts
        mt_ms = self.fitts.movement_time_ms(distance, tw)
        mt_s = mt_ms / 1000.0

        # Overshoot
        overshoot = self.fitts.should_overshoot()
        os_pt = self.fitts.overshoot_point(start_x, start_y, to_x, to_y) if overshoot else None

        # Generate waypoints
        waypoints = self.saccade.generate(
            start_x, start_y, to_x, to_y, tw,
            overshoot=overshoot, overshoot_point=os_pt,
        )

        # HID timestamps
        hid_ts = self.hid.generate_timestamps(mt_s)

        # Interpolate waypoints to Bezier curve, then sample at HID timestamps
        n_bezier = len(hid_ts) * 3  # oversample then pick
        bezier_pts = self.saccade.waypoints_to_bezier_curve(waypoints, n_bezier)

        if not bezier_pts:
            return []

        samples = self._sample_trajectory(bezier_pts, hid_ts, start_x, start_y, to_x, to_y, mt_s)

        # Micro-pause (probabilistic)
        if self._rng.random() < self.config.saccade.micro_pause_probability:
            pause_dur = self._rng.uniform(*self.config.saccade.micro_pause_duration_ms) / 1000.0
            pause_samples = self._generate_dwell(samples[-1].x, samples[-1].y, pause_dur)
            for ps in pause_samples:
                ps.phase = "dwell"
            samples.extend(pause_samples)

        # Update internal state
        if samples:
            self._x = samples[-1].x
            self._y = samples[-1].y
            self._time = samples[-1].timestamp

        return samples

    def click_at(
        self,
        x: float,
        y: float,
        button: str = "left",
        target_width: float | None = None,
    ) -> ClickEvent:
        """
        Move to (x, y) and click.

        Returns a ClickEvent containing all movement samples leading up to
        and including the click.
        """
        # Check for misclick
        is_misclick = self._rng.random() < self._persona_misclick_rate
        click_x, click_y = x, y
        if is_misclick:
            click_x = x + self._rng.normal(0.0, self.config.misclick_offset_px)
            click_y = y + self._rng.normal(0.0, self.config.misclick_offset_px)

        # Move to click location
        move_samples = self.move_to(click_x, click_y, target_width)

        # Generate click
        click = self.click_model.generate(
            click_x, click_y, button,
            movement_start_time=self._time,
            is_misclick=is_misclick,
        )
        click.samples = move_samples
        click.pre_click_x = self._x
        click.pre_click_y = self._y

        # Update position after click (account for release drift)
        self._x = click.post_click_x
        self._y = click.post_click_y

        return click

    def double_click(
        self,
        x: float,
        y: float,
        button: str = "left",
        target_width: float | None = None,
    ) -> list[ClickEvent]:
        """Two rapid clicks with realistic inter-click interval."""
        # Inter-click interval — usually 60-140 ms
        interval_ms = self._rng.uniform(*self.config.click.double_click_interval_ms)

        click1 = self.click_at(x, y, button, target_width)

        # Short dwell between clicks
        self._x = x
        self._y = y

        click2 = self.click_model.generate(
            x, y, button,
            movement_start_time=click1.up_timestamp + interval_ms / 1000.0,
        )
        click2.samples = []

        return [click1, click2]

    def drag(
        self,
        to_x: float,
        to_y: float,
        from_x: float | None = None,
        from_y: float | None = None,
        button: str = "left",
    ) -> tuple[ClickEvent, list[MouseSample], ClickEvent]:
        """
        Click down at current position, move to (to_x, to_y), release.

        Returns (press_event, drag_samples, release_event).
        """
        sx = from_x if from_x is not None else self._x
        sy = from_y if from_y is not None else self._y

        press = self.click_model.generate(sx, sy, button, self._time)
        drag_samples = self.move_to(to_x, to_y)
        release = self.click_model.generate(to_x, to_y, button, self._time)

        return press, drag_samples, release

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_trajectory(
        self,
        bezier_pts: list[tuple[float, float, float]],
        hid_timestamps: np.ndarray,
        start_x: float, start_y: float,
        to_x: float, to_y: float,
        total_duration_s: float,
    ) -> list[MouseSample]:
        """
        Sample the Bezier curve at HID timestamps, applying tremor and
        computing kinematics.
        """
        samples: list[MouseSample] = []
        prev_x, prev_y = start_x, start_y
        prev_vx, prev_vy = 0.0, 0.0
        prev_t = 0.0

        # Sort bezier points by time fraction
        bp_sorted = sorted(bezier_pts, key=lambda p: p[2])

        for hid_t in hid_timestamps:
            # Find the two bezier points bracketing this timestamp
            time_frac = hid_t / total_duration_s if total_duration_s > 0 else 1.0
            time_frac = min(time_frac, 1.0)

            # Interpolate bezier position at time_frac
            bx, by = self._interpolate_bezier_at(bp_sorted, time_frac)

            # Apply tremor
            tx, ty = self.tremor.apply(bx, by, hid_t)

            # Kinematics
            dt = hid_t - prev_t if hid_t > prev_t else 0.001
            vx = (tx - prev_x) / dt
            vy = (ty - prev_y) / dt
            velocity = math.hypot(vx, vy)

            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            acceleration = math.hypot(ax, ay)

            # Jerk: rate of change of acceleration (simplified)
            jerk = abs(acceleration) / dt if dt > 0 else 0.0

            # Determine phase
            if time_frac < 0.15:
                phase = "ballistic"
            elif time_frac < 0.85:
                phase = "ballistic"
            else:
                phase = "corrective"

            samples.append(MouseSample(
                timestamp=hid_t,
                x=tx, y=ty,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                phase=phase,
            ))

            prev_x, prev_y = tx, ty
            prev_vx, prev_vy = vx, vy
            prev_t = hid_t

        return samples

    @staticmethod
    def _interpolate_bezier_at(
        bp_sorted: list[tuple[float, float, float]],
        t: float,
    ) -> tuple[float, float]:
        """Linear interpolation between the two bracketing bezier points."""
        if t <= bp_sorted[0][2]:
            return bp_sorted[0][0], bp_sorted[0][1]
        if t >= bp_sorted[-1][2]:
            return bp_sorted[-1][0], bp_sorted[-1][1]

        for i in range(len(bp_sorted) - 1):
            t_a = bp_sorted[i][2]
            t_b = bp_sorted[i + 1][2]
            if t_a <= t <= t_b:
                if t_b - t_a < 1e-9:
                    return bp_sorted[i][0], bp_sorted[i][1]
                alpha = (t - t_a) / (t_b - t_a)
                x = bp_sorted[i][0] + alpha * (bp_sorted[i + 1][0] - bp_sorted[i][0])
                y = bp_sorted[i][1] + alpha * (bp_sorted[i + 1][1] - bp_sorted[i][1])
                return x, y

        return bp_sorted[-1][0], bp_sorted[-1][1]

    def _generate_dwell(
        self, x: float, y: float, duration_s: float,
    ) -> list[MouseSample]:
        """Generate samples representing cursor dwell (stationary with tremor)."""
        ts = self.hid.generate_timestamps(duration_s)
        samples = []
        for t in ts:
            tx, ty = self.tremor.apply(x, y, t)
            samples.append(MouseSample(
                timestamp=t, x=tx, y=ty,
                velocity=0.0, acceleration=0.0, jerk=0.0, phase="dwell",
            ))
        return samples
