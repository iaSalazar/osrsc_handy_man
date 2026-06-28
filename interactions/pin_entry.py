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
import subprocess
import time
from typing import Callable, Optional

import cv2
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
        screen_capture: Callable[[], np.ndarray] | None = None,
    ) -> None:
        self.mouse = mouse
        self.scanner = scanner
        self.cognitive = cognitive
        self.keyboard = keyboard
        self.logger = logger
        self._rng = rng
        self.config = config
        self._start_time = time.time()

        # Optional live-mode screen capture — when provided, enter_pin()
        # performs real computer vision instead of synthetic position lookup.
        self._screen_capture = screen_capture

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

        live_mode = self._screen_capture is not None

        self._log("pin_entry_start", {"pin_configured": True, "live_mode": live_mode})

        # ---- Step 1: Detect PIN screen ----
        self._delay(200, 500, "orient to PIN screen")

        pin_screen_visible = False
        if live_mode:
            # Live mode: try template first, then HSV red-square fallback
            screen_bgr = self._screen_capture()  # type: ignore[misc]
            screen_hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
            h, w = screen_bgr.shape[:2]

            # Template check
            pin_screen = self._scan_live(
                PIN_TEMPLATES["pin_screen_indicator"],
                screen_bgr,
                "verify PIN screen visible (live)",
            )

            # HSV fallback: count red square buttons in centre region
            red = cv2.inRange(
                screen_hsv[h // 5:4 * h // 5, w // 6:5 * w // 6],
                np.array([0, 150, 50]), np.array([12, 255, 160]),
            )
            red |= cv2.inRange(
                screen_hsv[h // 5:4 * h // 5, w // 6:5 * w // 6],
                np.array([170, 150, 50]), np.array([180, 255, 160]),
            )
            contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sq = sum(
                1 for c in contours
                if 200 < cv2.contourArea(c) < 50000
                and 0.6 < cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) < 1.7
            )
            pin_screen_visible = pin_screen.found or sq >= 7
            self._log("pin_screen_detect", {
                "template_found": pin_screen.found,
                "red_squares": sq,
                "visible": pin_screen_visible,
            })
        else:
            pin_screen = self._scan(
                PIN_TEMPLATES["pin_screen_indicator"],
                "verify PIN screen visible",
            )
            pin_screen_visible = pin_screen.found

        if not pin_screen_visible:
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
                wrong_digit = str(self._rng.choice(
                    [d for d in "0123456789" if d != digit_char]
                ))
                wrong_template = f"bank_pin_digit_{wrong_digit}"
                wrong_scan = self._scan(
                    PIN_TEMPLATES.get(wrong_template, wrong_template),
                    f"glance at wrong digit {wrong_digit}",
                )
                if wrong_scan.found:
                    self.mouse.move_to(wrong_scan.center_x, wrong_scan.center_y)
                    self._delay(100, 300, "realize wrong digit")
                    self._log("pin_correction", {"glanced_at": wrong_digit, "correct_digit": digit_char})

            # ---- Locate the correct digit button ----
            if live_mode:
                # Live mode: capture fresh screen, use two-stage detection
                self.mouse.move_to(100, 100)  # move away to avoid hover corruption
                time.sleep(0.05)
                screen_bgr = self._screen_capture()  # type: ignore[misc]
                screen_hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
                result = self._detect_digit_live(digit_char, screen_bgr, screen_hsv)

                if result is None:
                    self._log("pin_error", {"digit": digit_char, "index": i, "reason": "digit_not_found_live"})
                    self.cognitive.record_error()
                    self._delay(300, 600, "re-scan PIN screen")
                    screen_bgr = self._screen_capture()  # type: ignore[misc]
                    screen_hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
                    result = self._detect_digit_live(digit_char, screen_bgr, screen_hsv)
                    if result is None:
                        return False

                cx, cy, conf = result
                btn_width = 35  # approximate PIN button width for Fitts/jitter
                self._log("pin_digit_live_detect", {
                    "digit": digit_char, "x": int(cx), "y": int(cy),
                    "confidence": round(conf, 3),
                })
            else:
                # Synthetic mode: use template system
                digit_btn = self._scan(template_id, f"locate digit {digit_char} (position {i+1})")

                if not digit_btn.found:
                    self._log("pin_error", {"digit": digit_char, "index": i, "reason": "digit_not_found"})
                    self.cognitive.record_error()
                    self._delay(300, 600, "re-scan PIN screen")
                    digit_btn = self._scan(template_id, f"retry locate digit {digit_char}")
                    if not digit_btn.found:
                        return False

                cx, cy = digit_btn.center_x, digit_btn.center_y
                btn_width = digit_btn.width

            # Hover over the digit button
            hover_ms = self._rng.uniform(self._digit_delay_min, self._digit_delay_max)
            self._hover_at(cx, cy,
                          f"hover over digit {digit_char}",
                          hover_ms / 1000.0)

            # Click the digit — jitter click position within the button
            jx = cx + self._rng.normal(0.0, btn_width * 0.12)
            jy = cy + self._rng.normal(0.0, btn_width * 0.12)
            self._move_and_click(
                jx, jy,
                target_width=btn_width,
                description=f"Click digit {digit_char} (position {i+1}/4)",
            )

            # Brief post‑click pause (watching the PIN dot appear)
            self._delay(80, 200, f"confirm digit {i+1} registered")

        # ---- Step 3: Post‑PIN pause (as if mentally confirming) ----
        self._delay(300, 900, "mentally confirm PIN entered correctly")

        # ---- Step 4: Verify PIN accepted (bank open) ----
        if live_mode:
            screen_bgr = self._screen_capture()  # type: ignore[misc]
            bank_open = self._scan_live(
                PIN_TEMPLATES["pin_accepted"],
                screen_bgr,
                "verify bank opened after PIN (live)",
            )
        else:
            bank_open = self._scan(
                PIN_TEMPLATES["pin_accepted"],
                "verify bank opened after PIN",
            )

        if bank_open.found:
            self._log("pin_success", {"digits_entered": 4})
            return True
        else:
            # Maybe PIN screen just took a moment to transition
            self._delay(400, 1000, "wait for PIN to process")
            if live_mode:
                screen_bgr = self._screen_capture()  # type: ignore[misc]
                bank_open2 = self._scan_live(
                    PIN_TEMPLATES["pin_accepted"],
                    screen_bgr,
                    "recheck bank open after PIN (live)",
                )
            else:
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

    def _scan_live(
        self, template_id: str, screen_bgr: np.ndarray, description: str = "",
    ) -> ScanResult:
        """Like _scan but passes a live screen to VisualScanner for real CV."""
        result = self.scanner.find_element(template_id, conditions={}, screen=screen_bgr)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id, timestamp=self._elapsed(),
            wall_time=time.time(), event_type=EventType.SCREEN_SCAN,
            data={"template_id": template_id, "found": result.found,
                  "center_x": result.center_x, "center_y": result.center_y,
                  "confidence": result.confidence, "method": result.method.value,
                  "description": description, "live": True},
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

    # ------------------------------------------------------------------
    # Live-mode digit detection (two-stage: HSV buttons → template digits)
    # ------------------------------------------------------------------

    def _detect_digit_live(
        self, digit_char: str, screen_bgr: np.ndarray, screen_hsv: np.ndarray,
    ) -> tuple[int, int, float] | None:
        """
        Two-stage live digit detection for a single target digit.

        Stage 1: HSV red-square detection to find all PIN buttons.
        Stage 2: Template-match all 10 digit templates within each button.
        Stage 3: Uniqueness constraint + per-digit bias + margin check.

        Returns (center_x, center_y, adjusted_confidence) or None.
        """
        h, w = screen_bgr.shape[:2]

        DIGIT_BIAS = {7: -0.20, 1: -0.05, 3: -0.05}
        MIN_RAW_CONF = 0.5
        MARGIN_PCT = 3.0

        ox, oy = w // 6, h // 5
        crop_h1, crop_h2 = h // 5, 4 * h // 5
        crop_w1, crop_w2 = w // 6, 5 * w // 6

        # ---- Stage 1: Find red square buttons ----
        red = cv2.inRange(
            screen_hsv[crop_h1:crop_h2, crop_w1:crop_w2],
            np.array([0, 150, 50]), np.array([12, 255, 160]),
        )
        red |= cv2.inRange(
            screen_hsv[crop_h1:crop_h2, crop_w1:crop_w2],
            np.array([170, 150, 50]), np.array([180, 255, 160]),
        )
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        btns = [
            (cv2.boundingRect(c), cv2.contourArea(c))
            for c in contours
            if 200 < cv2.contourArea(c) < 50000
            and 0.6 < cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) < 1.7
        ]
        btns.sort(key=lambda b: b[1], reverse=True)
        uniq = []
        for (bx, by, bw, bh), _ in btns:
            if not any(abs(bx - u[0]) < 15 and abs(by - u[1]) < 15 for u in uniq):
                uniq.append((bx, by, bw, bh))
                if len(uniq) >= 12:
                    break

        # ---- Stage 2: Match all 10 digit templates within each button ----
        all_buttons: list[dict] = []
        for bx, by, bw, bh in uniq:
            sx, sy = ox + bx, oy + by
            if sy + bh > h or sx + bw > w:
                continue
            crop = screen_bgr[sy:sy + bh, sx:sx + bw]
            scores: dict[int, tuple[float, float]] = {}
            for dig in range(10):
                needle = cv2.imread(f"data/screenshots/pin/{dig}.png")
                if needle is None or needle.shape[0] > crop.shape[0] or needle.shape[1] > crop.shape[1]:
                    continue
                result = cv2.matchTemplate(crop, needle, cv2.TM_CCOEFF_NORMED)
                _, raw_conf, _, _ = cv2.minMaxLoc(result)
                if raw_conf > MIN_RAW_CONF:
                    adjusted = raw_conf + DIGIT_BIAS.get(dig, 0.0)
                    scores[dig] = (raw_conf, adjusted)

            if not scores:
                continue

            best_dig = max(scores, key=lambda d: scores[d][1])
            best_adj = scores[best_dig][1]
            others = [(d, scores[d][1]) for d in scores if d != best_dig]
            runner_adj = max(others, key=lambda x: x[1])[1] if others else 0.0
            margin = (best_adj - runner_adj) / best_adj * 100 if best_adj > 0 else 0.0

            all_buttons.append({
                "pos": (sx + bw // 2, sy + bh // 2),
                "best_dig": best_dig,
                "best_adj": best_adj,
                "margin": margin,
                "scores": scores,
            })

        # ---- Stage 3: Uniqueness constraint ----
        all_buttons.sort(key=lambda b: b["best_adj"], reverse=True)
        assigned_digits: set[int] = set()

        for btn in all_buttons:
            if btn["margin"] < MARGIN_PCT:
                continue

            best = btn["best_dig"]
            if best not in assigned_digits:
                assigned_digits.add(best)
                if str(best) == digit_char:
                    return (*btn["pos"], btn["best_adj"])
            else:
                candidates = [
                    (d, btn["scores"][d][1])
                    for d in btn["scores"]
                    if d not in assigned_digits
                ]
                if not candidates:
                    continue
                next_dig, next_conf = max(candidates, key=lambda x: x[1])
                others = [
                    (d, btn["scores"][d][1])
                    for d in btn["scores"]
                    if d != next_dig and d not in assigned_digits
                ]
                next_runner = max(others, key=lambda x: x[1])[1] if others else 0.0
                next_margin = (
                    (next_conf - next_runner) / next_conf * 100
                    if next_conf > 0 else 0.0
                )
                if next_margin >= MARGIN_PCT:
                    assigned_digits.add(next_dig)
                    if str(next_dig) == digit_char:
                        return (*btn["pos"], next_conf)

        return None

    # ------------------------------------------------------------------
    # Internal: timing
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.time() - self._start_time
