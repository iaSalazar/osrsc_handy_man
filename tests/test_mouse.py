"""Tests for the motor control engine."""

import numpy as np
import pytest

from core.config import MouseConfig, FittsParams
from core.mouse import MouseEmulator, FittsLaw, TremorEngine, ClickModel, HIDPacer


class TestFittsLaw:
    def test_monotonicity(self):
        """Movement time should increase with distance and decrease with target width."""
        rng = np.random.default_rng(42)
        fl = FittsLaw(FittsParams(), rng)

        # Same width, larger distance = longer movement
        t1 = fl.movement_time_ms(50, 20)
        t2 = fl.movement_time_ms(200, 20)
        assert t2 > t1

        # Same distance, larger target = shorter movement
        t3 = fl.movement_time_ms(200, 10)
        t4 = fl.movement_time_ms(200, 50)
        assert t4 < t3

    def test_negligible_distance(self):
        rng = np.random.default_rng(42)
        fl = FittsLaw(FittsParams(a=40), rng)
        t = fl.movement_time_ms(0.1, 20)
        assert t >= 30  # close to intercept

    def test_endpoint_noise(self):
        rng = np.random.default_rng(42)
        fl = FittsLaw(FittsParams(), rng)
        x, y = fl.endpoint_noise(100, 100)
        assert abs(x - 100) < 30  # within 10 sigma
        assert abs(y - 100) < 30


class TestTremorEngine:
    def test_output_is_noisy(self):
        rng = np.random.default_rng(42)
        from core.config import TremorConfig
        te = TremorEngine(TremorConfig(), rng)

        x, y = te.apply(100, 100, 0.0)
        # Tremor should add non-zero displacement
        x2, y2 = te.apply(100, 100, 0.5)
        # Position should vary over time
        assert (x != x2) or (y != y2)


class TestClickModel:
    def test_hold_time_in_range(self):
        rng = np.random.default_rng(42)
        from core.config import ClickDynamics
        cm = ClickModel(ClickDynamics(), rng)

        for _ in range(100):
            ht = cm.sample_hold_time_ms()
            assert ht >= 40  # above minimum

    def test_generate_click(self):
        rng = np.random.default_rng(42)
        from core.config import ClickDynamics
        cm = ClickModel(ClickDynamics(), rng)

        click = cm.generate(100, 200, "left")
        assert click.button == "left"
        assert click.down_timestamp < click.up_timestamp
        assert click.hold_duration_ms > 0


class TestHIDPacer:
    def test_timestamps_in_range(self):
        rng = np.random.default_rng(42)
        from core.config import HIDConfig
        hp = HIDPacer(HIDConfig(report_rate_hz=500), rng)

        ts = hp.generate_timestamps(1.0)
        assert len(ts) > 0
        assert ts[0] >= 0
        assert ts[-1] <= 1.1  # small tolerance

    def test_empty_duration(self):
        rng = np.random.default_rng(42)
        from core.config import HIDConfig
        hp = HIDPacer(HIDConfig(), rng)

        ts = hp.generate_timestamps(0.0)
        assert len(ts) == 1
        assert ts[0] == 0.0


class TestMouseEmulator:
    def test_move_to_generates_samples(self):
        rng = np.random.default_rng(42)
        config = MouseConfig()
        emu = MouseEmulator(config, rng)

        samples = emu.move_to(500, 500, target_width=40, from_x=100, from_y=100)
        assert len(samples) > 5  # should generate multiple samples
        # Final position should be near target
        assert abs(samples[-1].x - 500) < 50
        assert abs(samples[-1].y - 500) < 50

    def test_negligible_movement(self):
        rng = np.random.default_rng(42)
        config = MouseConfig()
        emu = MouseEmulator(config, rng)

        samples = emu.move_to(100, 100, from_x=100, from_y=100)
        assert len(samples) == 1  # just the dwell

    def test_click_generates_event(self):
        rng = np.random.default_rng(42)
        config = MouseConfig()
        emu = MouseEmulator(config, rng)
        emu._x, emu._y = 200, 200

        click = emu.click_at(400, 400, target_width=20)
        assert click.samples  # movement samples
        assert click.hold_duration_ms > 0

    def test_position_updated_after_move(self):
        rng = np.random.default_rng(42)
        config = MouseConfig()
        emu = MouseEmulator(config, rng)

        emu.move_to(300, 300, from_x=0, from_y=0)
        x, y = emu.current_position
        assert abs(x - 300) < 50
        assert abs(y - 300) < 50
