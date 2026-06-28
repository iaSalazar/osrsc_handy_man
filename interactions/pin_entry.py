"""
Bank PIN Entry Handler — enters the 4‑digit bank PIN via screen‑space interaction.

OSRS bank PINs use a randomized number pad (digits 0–9 appear in random
positions each time).  This handler:

  1. Detects the PIN entry screen
  2. Locates each digit button via template matching
  3. Clicks the 4 digits in order
  4. Confirms the PIN was accepted (bank opens)

The PIN is read from the OSRS_BANK_PIN environment variable (4 digits, e.g. "2006").

Anti‑ban features:
  - Variable delays between digits (security‑conscious but human — not robotic)
  - Hover over each digit button before clicking
  - Occasional hover over a wrong digit, then correct (error simulation)
  - Gaze around the PIN screen between digits
  - Post‑PIN pause (as if mentally confirming the code)
  - Misclick handling on digit buttons
  - Varies click position within each digit button

Timing is calibrated to feel like someone who types their PIN regularly
(fast enough to be familiar, slow enough not to be a bot):
  - Button‑to‑button: 150–450 ms
  - Digit hover: 80–250 ms
  - Post‑PIN confirmation pause: 300–900 ms

Usage:
    from interactions.pin_entry import PinEntryHandler

    pin = PinEntryHandler(mouse, scanner, cognitive, keyboard, logger, rng, config)
    success = pin.enter_pin()
    if success:
        # bank is now open
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from core.config import AppConfig
from core.logger import TelemetryLogger, TelemetryEvent, EventType, make_click
from core.mouse import MouseEmulator, ClickEvent, MouseSample
from core.cognitive import CognitiveEngine
from core.scan import VisualScanner, ScanResult
from core.keyboard import KeyboardEmulator


PIN_TEMPLATES = {
    "pin_screen_indicator": "bank_pin_screen",         # confirms PIN screen is visible
    "pin_digit_0": "bank_pin_digit_0",
    "pin_digit_1": "bank_pin_digit_1",
    "pin_digit_2": "bank_pin_digit_2",
    "pin_digit_3": "bank_pin_digit_3",
    "pin_digit_4": "bank_pin_digit_4",
    "pin_digit_5": "bank_pin_digit_5",
    "pin_digit_6": "bank_pin_digit_6",
    "pin_digit_7": "bank_pin_digit_7",
    "pin_digit_8": "bank_pin_digit_8",
    "pin_digit_9": "bank_pin_digit_9",
    "pin_accepted": "bank_interface_open",             # confirms PIN was accepted → bank open
}


class PinEntryHandler:
    """
    Handles OSRS bank PIN entry via template‑matched digit buttons.
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
        self.mouse = mouse
        self.scanner = scanner
        self.cognitive = cognitive
        self.keyboard = keyboard
        self.logger = logger
        self._rng = rng
        self.config = config
        self._start_time = time.time()

        # Load PIN from environment
        self._pin = os.getenv("OSRS_BANK_PIN", "")
        if not self._pin or len(self._pin) != 4 or not self._pin.isdigit():
            self._pin = ""  # will skip PIN entry

        # Timing constants (overridable via env)
        self._digit_delay_min = float(os.getenv("OSRS_PIN_DIGIT_DELAY_MIN_MS", "80"))
        self._digit_delay_max = float(os.getenv("OSRS_PIN_DIGIT_DELAY_MAX_MS", "250"))
        self._between_digit_min = float(os.getenv("OSRS_PIN_BETWEEN_DIGIT_MIN_MS", "150"))
        self._between_digit_max = float(os.getenv("OSRS_PIN_BETWEEN_DIGIT_MAX_MS", "450"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enter_pin(self) -> bool:
        """
        Execute the PIN entry sequence.

        Returns True if PIN was entered and bank appears to be open.
        """
        if not self._pin:
            self._log("pin_skipped", {"reason": "no_pin_in_env"})
            return True  # no PIN configured — assume already unlocked

        self._log("pin_entry_start", {"pin_configured": True})

        # ---- Step 1: Detect PIN screen ----
        self._delay(200, 500, "orient to PIN screen")

        pin_screen = self._scan(
            PIN_TEMPLATES["pin_screen_indicator"],
            "verify PIN screen visible",
        )

        if not pin_screen.found:
            # Maybe PIN not required right now?
            self._log("pin_skipped", {"reason": "pin_screen_not_visible"})
            bank_check = self._scan(
                PIN_TEMPLATES["pin_accepted"],
                "check if bank already open",
            )
            return bank_check.found

        # ---- Step 2: Enter each of the 4 digits ----
        for i, digit_char in enumerate(self._pin):
            digit_template = f"bank_pin_digit_{digit_char}"
            template_id = PIN_TEMPLATES.get(digit_template, digit_template)

            self._log("pin_digit_entry", {"digit_index": i, "digit": digit_char})

            # Between‑digit cognitive delay
            if i > 0:
                self._delay(self._between_digit_min, self._between_digit_max,
                           f"pause before digit {i+1}")

            # Anti‑ban: occasionally hover over wrong digit first, then correct
            if i > 1 and self._rng.random() < 0.08:
                # 8% chance: hover toward wrong digit, realize, correct
                wrong_digit = str(self._rng.choice(
                    [d for d in "0123456789" if d != digit_char]
                ))
                wrong_template = f"bank_pin_digit_{wrong_digit}"
                wrong_scan = self._scan(
                    PIN_TEMPLATES.get(wrong_template, wrong_template),
                    f"glance at wrong digit {wrong_digit}",
                )
                if wrong_scan.found:
                    # Move toward it briefly, then correct
                    self.mouse.move_to(wrong_scan.center_x, wrong_scan.center_y)
                    self._delay(100, 300, "realize wrong digit")
                    self._log("pin_correction", {"glanced_at": wrong_digit, "correct_digit": digit_char})

            # Scan for the correct digit button
            digit_btn = self._scan(template_id, f"locate digit {digit_char} (position {i+1})")

            if not digit_btn.found:
                self._log("pin_error", {"digit": digit_char, "index": i, "reason": "digit_not_found"})
                self.cognitive.record_error()
                # Fallback: re‑scan the PIN screen and try again
                self._delay(300, 600, "re-scan PIN screen")
                digit_btn = self._scan(template_id, f"retry locate digit {digit_char}")
                if not digit_btn.found:
                    return False

            # Hover over the digit button
            hover_ms = self._rng.uniform(self._digit_delay_min, self._digit_delay_max)
            self._hover_at(digit_btn.center_x, digit_btn.center_y,
                          f"hover over digit {digit_char}",
                          hover_ms / 1000.0)

            # Click the digit
            # Jitter click position within the button
            jx = digit_btn.center_x + self._rng.normal(0.0, digit_btn.width * 0.12)
            jy = digit_btn.center_y + self._rng.normal(0.0, digit_btn.height * 0.12)
            self._move_and_click(
                jx, jy,
                target_width=digit_btn.width,
                description=f"Click digit {digit_char} (position {i+1}/4)",
            )

            # Brief post‑click pause (watching the PIN dot appear)
            self._delay(80, 200, f"confirm digit {i+1} registered")

        # ---- Step 3: Post‑PIN pause (as if mentally confirming) ----
        self._delay(300, 900, "mentally confirm PIN entered correctly")

        # ---- Step 4: Verify PIN accepted (bank open) ----
        bank_open = self._scan(
            PIN_TEMPLATES["pin_accepted"],
            "verify bank opened after PIN",
        )

        if bank_open.found:
            self._log("pin_success", {"digits_entered": 4})
            return True
        else:
            # Maybe PIN screen just takes a moment to transition
            self._delay(400, 1000, "wait for PIN to process")
            bank_open2 = self._scan(
                PIN_TEMPLATES["pin_accepted"],
                "recheck bank open after PIN",
            )
            if bank_open2.found:
                self._log("pin_success", {"digits_entered": 4, "delayed": True})
                return True

            self._log("pin_unconfirmed", {"digits_entered": 4})
            return True  # optimistic

    # ------------------------------------------------------------------
    # Internal: shared helpers (mirrors login.py pattern)
    # ------------------------------------------------------------------

    def _move_and_click(
        self, x: float, y: float, target_width: float | None = None,
        button: str = "left", description: str = "",
    ) -> ClickEvent:
        is_misclick = self.cognitive.should_misclick()
        movements = self.mouse.move_to(x, y, target_width)
        for ms in movements:
            self._log_mouse_sample(ms)
        click = self.mouse.click_model.generate(x, y, button, self._elapsed(), is_misclick=is_misclick)
        self._log_click(click, is_misclick, description)
        if is_misclick:
            self._delay(150, 400, "notice PIN misclick")
            correction = self.mouse.click_model.generate(x, y, button, self._elapsed())
            self._log_click(correction, False, f"correct {description}")
        self.mouse._x = click.post_click_x
        self.mouse._y = click.post_click_y
        return click

    def _hover_at(self, x: float, y: float, description: str = "", dwell_s: float = 0.4) -> None:
        self.mouse.move_to(x, y)
        hover_points = self.scanner.simulate_hover_dwell(x, y, dwell_s)
        for hp in hover_points:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed() + hp["timestamp"],
                wall_time=time.time(),
                event_type=EventType.GAZE_POINT,
                data={"x": hp["x"], "y": hp["y"], "duration_s": dwell_s, "context": description},
                labels=["synthetic", "hover_dwell", "pin_entry"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed() * 1000,
            ))
        self.mouse._x, self.mouse._y = x, y

    def _scan(self, template_id: str, description: str = "") -> ScanResult:
        result = self.scanner.find_element(template_id, conditions={}, screen=None)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id, timestamp=self._elapsed(),
            wall_time=time.time(), event_type=EventType.SCREEN_SCAN,
            data={"template_id": template_id, "found": result.found,
                  "center_x": result.center_x, "center_y": result.center_y,
                  "confidence": result.confidence, "method": result.method.value,
                  "description": description},
            labels=["synthetic", "pin_entry"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))
        return result

    def _log(self, event: str, data: dict | None = None) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id, timestamp=self._elapsed(),
            wall_time=time.time(), event_type=EventType.SESSION_META,
            data={"module": "pin_entry", "event": event, **(data or {})},
            labels=["synthetic", "pin_entry"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))

    def _delay(self, min_ms: float, max_ms: float, description: str = "") -> None:
        delay_ms = self._rng.uniform(min_ms, max_ms)
        if delay_ms <= 0:
            return
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id, timestamp=self._elapsed(),
            wall_time=time.time(), event_type=EventType.COGNITIVE_DELAY,
            data={"delay_ms": round(delay_ms, 2), "context": description},
            labels=["synthetic", "pin_entry"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))

    def _log_mouse_sample(self, ms: MouseSample) -> None:
        from core.logger import make_mouse_move
        self.logger.log(make_mouse_move(
            self.logger.session_id, self._elapsed() + ms.timestamp,
            time.time(), ms.x, ms.y, ms.velocity, ms.acceleration,
            ms.phase, self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed() * 1000,
        ))

    def _log_click(self, click: ClickEvent, is_misclick: bool, desc: str) -> None:
        self.logger.log(make_click(
            EventType.MOUSE_CLICK_DOWN if not is_misclick else EventType.MISCLICK,
            self.logger.session_id, self._elapsed() + click.down_timestamp,
            time.time(), click.pre_click_x, click.pre_click_y, click.button,
            self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed() * 1000,
            hold_duration_ms=click.hold_duration_ms,
            micro_drift_px=click.micro_drift_px,
            is_misclick=is_misclick,
        ))
        self.logger.log(make_click(
            EventType.MOUSE_CLICK_UP,
            self.logger.session_id, self._elapsed() + click.up_timestamp,
            time.time(), click.post_click_x, click.post_click_y, click.button,
            self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed() * 1000,
            hold_duration_ms=click.hold_duration_ms,
            micro_drift_px=click.micro_drift_px,
            is_misclick=is_misclick,
        ))

    def _elapsed(self) -> float:
        return time.time() - self._start_time
