"""
Grand Exchange Bank Test Scenario.

Opens and closes the Grand Exchange bank repeatedly using only template
matching — no hardcoded coordinates.  This is the first scenario you
should run after capturing GE bank templates.

Usage:
    python3 -c "
    from scenarios.test_ge_bank import run_ge_bank_test
    run_ge_bank_test(cycles=5, persona='fast_accurate')
    "

Template Requirements
---------------------
Before running, capture these templates (see TEMPLATE_LIST below):

  1. ge_bank_booth          — The GE bank booth (closed state, clickable)
  2. bank_interface_open    — The bank interface title/header (confirms bank is open)
  3. bank_close_button      — The X button to close the bank interface

Each template needs variants at:
  - 4 yaw angles: N (0°), E (90°), S (180°), W (270°)
  - 2 pitch: default (middle), high (bird's-eye)
  - 2 zoom: mid, out

  → Minimum 16 variants per template, 48 total screenshots for the set.

After templates are registered via `telem-emu capture register`, this
script will find the GE bank at any camera angle within the captured range.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import AppConfig, PersonaType
from core.mouse import MouseEmulator
from core.cognitive import CognitiveEngine
from core.scan import VisualScanner
from core.keyboard import KeyboardEmulator
from core.logger import TelemetryLogger
from interactions.bank_interaction import BankInteraction


# ======================================================================
# TEMPLATE LIST — what you need to capture
# ======================================================================

TEMPLATE_LIST = """
┌──────────────────────────────────────────────────────────────────┐
│  GRAND EXCHANGE BANK — Screenshot Capture Checklist               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TEMPLATE 1: ge_bank_booth                                       │
│  ─────────────────────────                                       │
│  What: The brown/tan bank booth with the window/counter.          │
│        Include the entire clickable area of the booth.            │
│  States: default (closed, not hovered)                            │
│  Variants needed:                                                │
│    • Yaw: 0° (N), 90° (E), 180° (S), 270° (W)                   │
│    • Pitch: default, high                                        │
│    • Zoom: mid, out                                              │
│    → 4 × 2 × 2 = 16 minimum                                     │
│                                                                  │
│  TEMPLATE 2: bank_interface_open                                 │
│  ─────────────────────────────                                   │
│  What: The bank interface title bar or a distinctive element      │
│        that's only visible when the bank is open.                 │
│        Good choices: "Bank of Gielinor" text, the deposit         │
│        inventory button, or the item slot grid background.        │
│  States: open                                                     │
│  Variants needed: 2 angles × 1 pitch × 1 zoom = 4 minimum       │
│                                                                  │
│  TEMPLATE 3: bank_close_button                                   │
│  ──────────────────────────                                      │
│  What: The red X button in the top-right corner of the bank       │
│        interface.                                                 │
│  States: visible                                                  │
│  Variants needed: 2 angles × 1 pitch × 1 zoom = 4 minimum       │
│                                                                  │
│  OPTIONAL (for right-click interaction):                          │
│  TEMPLATE 4: menu_bank_booth                                     │
│  ──────────────────────────                                      │
│  What: The "Bank" text option in the right-click context menu     │
│        that appears when you right-click the booth.               │
│  Variants: 2 angles × 1 zoom = 2 minimum                         │
│                                                                  │
│  TOTAL MINIMUM: ~26 screenshots                                   │
│  RECOMMENDED: ~50-60 screenshots for robust matching              │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  CAPTURE PROTOCOL:                                                │
│  1. Set RuneLite to Fixed mode (classic layout) at 1920×1080     │
│  2. Disable GPU plugin / 117 HD (use default renderer)            │
│  3. For each variant: take a PNG screenshot                      │
│  4. Name files: ge_bank_booth_default_y0_default_mid.png etc.    │
│  5. Use telem-emu capture register to crop and save each variant  │
└──────────────────────────────────────────────────────────────────┘
"""


# ======================================================================
# Test Scenario
# ======================================================================

class GEBankTestScenario:
    """
    Repeatedly opens and closes the GE bank for template validation.

    This is a validation/test scenario — it doesn't do any actual banking.
    Use it to:
      1. Verify your templates are correctly captured
      2. Tune HSV ranges and confidence thresholds
      3. Check that camera rotation + variant selection works
    """

    def __init__(
        self,
        mouse: MouseEmulator,
        scanner: VisualScanner,
        cognitive: CognitiveEngine,
        keyboard: KeyboardEmulator,
        logger: TelemetryLogger,
        rng: np.random.Generator,
        config: AppConfig,
    ) -> None:
        self.bank = BankInteraction(
            mouse, scanner, cognitive, keyboard, logger, rng, config,
        )
        self.cognitive = cognitive
        self.logger = logger
        self._rng = rng
        self._start_time = time.time()

    def run(self, cycles: int = 5) -> dict:
        """
        Run `cycles` open→close bank interactions.

        Each cycle:
          1. Random camera rotation (30% chance)
          2. Open GE bank
          3. Brief pause (as if looking at bank contents)
          4. Close bank
          5. Inter-cycle idle (checking minimap, hovering, etc.)
        """
        successes = 0
        failures = 0

        for cycle in range(cycles):
            self._log(f"cycle_{cycle+1}_start", {"cycle": cycle + 1, "total_cycles": cycles})

            # Tick cognitive state
            self.cognitive.update(dt_s=self._rng.uniform(8.0, 15.0))

            # Brief idle between cycles (except first)
            if cycle > 0:
                idle_s = self._rng.uniform(1.0, 4.0)
                self.cognitive.update(dt_s=idle_s)

            # --- Open bank ---
            success = self.bank.open_bank(location="grand_exchange")

            if not success:
                failures += 1
                self._log(f"cycle_{cycle+1}_failed", {"reason": "bank_not_found"})
                continue

            # --- Simulate looking at bank contents ---
            browse_s = self._rng.uniform(1.5, 3.5)
            self.cognitive.update(dt_s=browse_s)

            # Optional: hover over random bank slot (simulating browsing)
            if self._rng.random() < 0.5:
                # Move mouse to a random position in the bank area
                bank_x = self._rng.integers(400, 1200)
                bank_y = self._rng.integers(200, 700)
                self.bank.mouse.move_to(bank_x, bank_y)
                self.bank._hover_at(bank_x, bank_y, "browsing bank")

            # --- Close bank ---
            self.bank.close_bank()

            successes += 1
            self._log(f"cycle_{cycle+1}_complete", {"success": True})

        summary = {
            "scenario": "ge_bank_test",
            "cycles": cycles,
            "successes": successes,
            "failures": failures,
            "persona": self.cognitive.persona.type.value,
            "phase": self.cognitive.current_phase.value,
        }
        self._log("test_complete", summary)
        self.logger.flush()

        return summary

    def _log(self, event: str, data: dict | None = None) -> None:
        from core.logger import TelemetryEvent, EventType
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=time.time() - self._start_time,
            wall_time=time.time(),
            event_type=EventType.SESSION_META,
            data={"scenario": "ge_bank_test", "event": event, **(data or {})},
            labels=["synthetic", "ge_bank_test"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=(time.time() - self._start_time) * 1000,
        ))


# ======================================================================
# Convenience entry point
# ======================================================================

def run_ge_bank_test(
    cycles: int = 5,
    persona: str = "fast_accurate",
    output_dir: str = "output",
    seed: int | None = None,
    synthetic: bool = True,
) -> dict:
    """
    Run the GE bank test scenario.

    Parameters
    ----------
    cycles : Number of open→close cycles.
    persona : Persona to use.
    output_dir : Directory for telemetry JSONL output.
    seed : RNG seed for reproducibility.
    synthetic : Use synthetic scan mode (no screen capture needed).

    Returns
    -------
    Summary dict.
    """
    import uuid

    rng = np.random.default_rng(seed)
    config = AppConfig()
    persona_type = PersonaType(persona)
    session_id = f"ge_bank_test_{uuid.uuid4().hex[:8]}"

    mouse = MouseEmulator(config.mouse, rng,
        persona_amplitude_scale={
            PersonaType.FAST_ACCURATE: 0.8,
            PersonaType.SLOW_METHODICAL: 1.2,
            PersonaType.DISTRACTED_ERROR_PRONE: 1.5,
        }.get(persona_type, 1.0),
    )
    cognitive = CognitiveEngine(persona_type, config.cognitive, rng)
    scanner = VisualScanner(config.scan, rng)
    keyboard = KeyboardEmulator(config.keyboard, rng,
        persona_typo_rate=cognitive.persona.typo_rate,
    )
    logger = TelemetryLogger(Path(output_dir), session_id)

    scenario = GEBankTestScenario(
        mouse=mouse, scanner=scanner, cognitive=cognitive,
        keyboard=keyboard, logger=logger, rng=rng, config=config,
    )

    return scenario.run(cycles=cycles)


# ======================================================================
# Direct run
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Grand Exchange Bank — Template Capture Checklist")
    print("=" * 60)
    print(TEMPLATE_LIST)
    print("=" * 60)
    print("\nRunning synthetic-mode test (no templates needed)...")
    print("This will generate telemetry with synthetic positions.\n")

    summary = run_ge_bank_test(cycles=3, persona="fast_accurate", seed=42)
    print(f"\nDone. Summary: {summary}")
    print(f"Output: output/ge_bank_test_*.jsonl")
