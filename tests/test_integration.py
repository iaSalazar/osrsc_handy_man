"""Integration tests — verify the full pipeline works end-to-end."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.config import AppConfig, PersonaType
from core.mouse import MouseEmulator
from core.cognitive import CognitiveEngine
from core.scan import VisualScanner
from core.keyboard import KeyboardEmulator
from core.logger import TelemetryLogger


class TestFullPipeline:
    """Test the Chaos Altar scenario in synthetic mode."""

    def test_scenario_runs_without_errors(self):
        """A single-cycle synthetic run should complete without exceptions."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        rng = np.random.default_rng(42)
        config = AppConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            mouse = MouseEmulator(config.mouse, rng)
            cognitive = CognitiveEngine(PersonaType.FAST_ACCURATE, config.cognitive, rng)
            scanner = VisualScanner(config.scan, rng)
            keyboard = KeyboardEmulator(config.keyboard, rng, persona_typo_rate=0.01)
            logger = TelemetryLogger(Path(tmpdir), "test_integration")

            from scenarios.chaos_altar_dragon_bones import ChaosAltarDragonBonesScenario

            scenario = ChaosAltarDragonBonesScenario(
                mouse=mouse, scanner=scanner, cognitive=cognitive,
                keyboard=keyboard, logger=logger, rng=rng,
                config=config, cycles=1, synthetic=True,
            )

            summary = scenario.run_session()

            # Verify summary
            assert summary["cycles_completed"] == 1
            assert summary["total_bones_used"] == 28
            assert summary["scenario"] == "chaos_altar_dragon_bones"

            # Verify output file exists and has events
            output_file = Path(tmpdir) / "test_integration.jsonl"
            assert output_file.exists()
            assert output_file.stat().st_size > 0

            # Parse events
            with open(output_file) as f:
                lines = [json.loads(l) for l in f if l.strip() and "SESSION_SUMMARY" not in l]

            assert len(lines) > 1000  # should have many events

            # Verify event structure
            first = lines[0]
            assert "session_id" in first
            assert "timestamp" in first
            assert "event_type" in first
            assert "data" in first
            assert "labels" in first
            assert "synthetic" in first["labels"]

            # Verify all 5 states are visited
            states = set()
            for l in lines:
                for lbl in l.get("labels", []):
                    if lbl.startswith("state:"):
                        states.add(lbl)
            assert len(states) >= 5  # all scenario states

    def test_multi_persona_runs(self):
        """All three personas should complete a half-cycle without error."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        for persona_type in [
            PersonaType.FAST_ACCURATE,
            PersonaType.SLOW_METHODICAL,
            PersonaType.DISTRACTED_ERROR_PRONE,
        ]:
            rng = np.random.default_rng(42)
            config = AppConfig()

            with tempfile.TemporaryDirectory() as tmpdir:
                mouse = MouseEmulator(config.mouse, rng)
                cognitive = CognitiveEngine(persona_type, config.cognitive, rng)
                scanner = VisualScanner(config.scan, rng)
                keyboard = KeyboardEmulator(config.keyboard, rng,
                                            persona_typo_rate=cognitive.persona.typo_rate)
                logger = TelemetryLogger(Path(tmpdir), f"test_{persona_type.value}")

                from scenarios.chaos_altar_dragon_bones import ChaosAltarDragonBonesScenario

                scenario = ChaosAltarDragonBonesScenario(
                    mouse=mouse, scanner=scanner, cognitive=cognitive,
                    keyboard=keyboard, logger=logger, rng=rng,
                    config=config, cycles=1, synthetic=True,
                )

                summary = scenario.run_session()
                assert summary["cycles_completed"] == 1
                assert logger.event_count > 0


class TestLogger:
    """Test the telemetry logger."""

    def test_log_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir), "test_log")
            from core.logger import TelemetryEvent, EventType

            evt = TelemetryEvent(
                session_id="test_log", timestamp=0.0, wall_time=1234567890.0,
                event_type=EventType.MOUSE_MOVE,
                data={"x": 100, "y": 200},
            )
            logger.log(evt)
            logger.close()

            output_file = Path(tmpdir) / "test_log.jsonl"
            assert output_file.exists()
            with open(output_file) as f:
                lines = f.readlines()
            assert len(lines) >= 1

    def test_buffer_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir), "test_buffer", buffer_size=10)
            from core.logger import TelemetryEvent, EventType

            for i in range(25):
                evt = TelemetryEvent(
                    session_id="test_buffer", timestamp=float(i),
                    wall_time=1234567890.0 + i,
                    event_type=EventType.MOUSE_MOVE,
                    data={"x": i, "y": i},
                )
                logger.log(evt)

            # Should have auto-flushed at least once
            assert logger.event_count == 25
            logger.close()

    def test_export_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir), "test_df")
            from core.logger import TelemetryEvent, EventType

            for i in range(5):
                evt = TelemetryEvent(
                    session_id="test_df", timestamp=float(i),
                    wall_time=1234567890.0 + i,
                    event_type=EventType.MOUSE_MOVE,
                    data={"x": i, "y": i * 10},
                )
                logger.log(evt)
            logger.flush()

            df = logger.export_dataframe()
            assert len(df) == 5
            assert "data" in df.columns
            logger.close()
