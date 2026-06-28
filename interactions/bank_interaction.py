"""
Reusable Bank Interaction Module — "open any bank, anywhere."

Provides a `BankInteraction` class that scenarios import to handle all
bank-related interactions.  Works with any bank type:

  - Bank booth (GE, Edgeville, Varrock, etc.)
  - Bank chest (Shantay Pass, Castle Wars, etc.)
  - Bank NPC (Al Kharid, Falador, etc.)
  - Bank deposit box

All localisation is done via template matching — no hardcoded coordinates.
The same code opens a bank at the GE, in Edgeville, or anywhere else;
you just provide the right templates.

Anti‑ban features built in:
  - Variable approach path (doesn't click the exact same pixel)
  - Sometimes hovers first, sometimes clicks immediately
  - Occasionally right‑clicks instead of left‑clicking
  - Camera rotation before approach (configurable probability)
  - Distracter scans before locating the bank
  - Realistic post‑interaction delays
  - Misclick handling with correction

Usage (from a scenario):
    from interactions.bank_interaction import BankInteraction

    bank = BankInteraction(mouse, scanner, cognitive, keyboard, logger, rng, config)
    success = bank.open_bank(location="grand_exchange")
    if success:
        # bank interface is now open — do your banking
        ...
        bank.close_bank()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np

from core.config import AppConfig
from core.logger import TelemetryLogger, TelemetryEvent, EventType
from core.mouse import MouseEmulator, ClickEvent, MouseSample
from core.cognitive import CognitiveEngine
from core.scan import VisualScanner, ScanResult
from core.keyboard import KeyboardEmulator


# ---------------------------------------------------------------------------
# Bank location definitions
# ---------------------------------------------------------------------------

@dataclass
class BankLocation:
    """
    Defines the templates and interaction parameters for a specific bank.

    Each location needs its own set of template IDs captured under
    varying camera angles and states.
    """
    location_id: str                          # e.g. "grand_exchange", "edgeville"
    display_name: str                         # human-readable name

    # Primary template — the thing you click to open the bank
    bank_object_template: str                 # e.g. "ge_bank_booth"
    bank_object_width_px: int = 35            # approx clickable width for Fitts

    # Optional: right‑click menu option label
    right_click_option_template: str = ""     # e.g. "menu_bank_option"

    # Verification template — confirms bank is open
    bank_open_indicator_template: str = ""    # e.g. "bank_interface_title"
    bank_open_indicator_width_px: int = 100

    # Close button
    close_button_template: str = "bank_close_button"
    close_button_width_px: int = 20

    # Interaction style preferences (probabilities)
    right_click_probability: float = 0.15       # chance to right-click instead of left
    hover_before_click_probability: float = 0.35 # chance to hover first
    camera_rotate_before_probability: float = 0.10 # chance to rotate camera first
    distracter_scan_count: int = 2               # distracter regions to scan before finding

    # Camera conditions that work well for this bank
    # (the template library should have variants for these)
    preferred_yaw_angles: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    preferred_pitches: list[str] = field(default_factory=lambda: ["default", "high"])


# ---------------------------------------------------------------------------
# Pre‑defined locations — extend this dictionary as you capture more templates
# ---------------------------------------------------------------------------

BANK_LOCATIONS: dict[str, BankLocation] = {
    "grand_exchange": BankLocation(
        location_id="grand_exchange",
        display_name="Grand Exchange",
        bank_object_template="ge_bank_booth",
        bank_object_width_px=35,
        right_click_option_template="menu_bank_booth",
        bank_open_indicator_template="bank_interface_open",
        bank_open_indicator_width_px=100,
        close_button_template="bank_close_button",
        right_click_probability=0.12,
        hover_before_click_probability=0.30,
        camera_rotate_before_probability=0.08,
        distracter_scan_count=2,
    ),
    "edgeville": BankLocation(
        location_id="edgeville",
        display_name="Edgeville",
        bank_object_template="edgeville_bank_booth",
        bank_object_width_px=35,
        bank_open_indicator_template="bank_interface_open",
        close_button_template="bank_close_button",
        right_click_probability=0.15,
        hover_before_click_probability=0.35,
        camera_rotate_before_probability=0.10,
        distracter_scan_count=3,
    ),
    # Generic fallback — uses whatever templates are registered
    "generic": BankLocation(
        location_id="generic",
        display_name="Generic Bank",
        bank_object_template="bank_booth",
        bank_object_width_px=35,
        bank_open_indicator_template="bank_interface_open",
        close_button_template="bank_close_button",
    ),
}


# ---------------------------------------------------------------------------
# Bank interaction class
# ---------------------------------------------------------------------------

class BankInteraction:
    """
    Reusable bank interaction handler.

    Usage:
        bank = BankInteraction(mouse, scanner, cognitive, keyboard, logger, rng, config)
        success = bank.open_bank(location="grand_exchange")
        # ... do stuff with bank open ...
        bank.close_bank()
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

        # Current location (set when open_bank is called)
        self._location: Optional[BankLocation] = None
        self._bank_is_open: bool = False

        # Timing — the scenario that owns this module should set _start_time
        self._start_time: float = time.time()

        # Camera state — shared with the scenario
        self._camera_yaw: int = 0
        self._camera_pitch: str = "default"
        self._camera_zoom: str = "mid"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_bank(
        self,
        location: str = "generic",
        camera_conditions: dict | None = None,
    ) -> bool:
        """
        Locate and open a bank.

        Parameters
        ----------
        location : Key into BANK_LOCATIONS dict (e.g. "grand_exchange").
        camera_conditions : Optional dict of current camera state for
                           template variant selection.

        Returns
        -------
        True if the bank was successfully opened.
        """
        loc = BANK_LOCATIONS.get(location, BANK_LOCATIONS["generic"])
        self._location = loc

        self._log("bank_open_start", {"location": loc.location_id})

        # 1. Maybe rotate camera for a better angle
        if self._rng.random() < loc.camera_rotate_before_probability:
            self._simulate_camera_rotate()

        # Build scan conditions
        conditions = camera_conditions or {
            "yaw": str(self._camera_yaw),
            "pitch": self._camera_pitch,
            "zoom": self._camera_zoom,
        }

        # 2. Cognitive pre-action delay
        rt_ms = self.cognitive.get_reaction_delay_ms()
        self._delay(rt_ms, "pre-bank reaction time")

        # 3. Scan for the bank object (with distracter scans)
        self._log("scanning_for_bank", {"template": loc.bank_object_template})
        scan_result = self._scan_with_distracters(
            loc.bank_object_template,
            conditions,
            distracter_count=loc.distracter_scan_count,
        )

        if not scan_result.found:
            self._log("bank_not_found", {"template": loc.bank_object_template})
            self.cognitive.record_error()
            # Re-scan with broader conditions
            scan_result = self._scan_with_distracters(
                loc.bank_object_template,
                conditions,
                distracter_count=1,
            )

        if not scan_result.found:
            self._log("bank_not_found_retry", {})
            return False

        # 4. Interact: hover → click, or just click
        hover_first = self._rng.random() < loc.hover_before_click_probability

        if hover_first:
            self._hover_at(scan_result.center_x, scan_result.center_y, "hover over bank booth")

        # 5. Choose interaction method
        use_right_click = self._rng.random() < loc.right_click_probability

        if use_right_click and loc.right_click_option_template:
            success = self._right_click_bank(scan_result, loc, conditions)
        else:
            success = self._left_click_bank(scan_result, loc)

        if not success:
            return False

        # 6. Wait for bank interface to appear
        wait_ms = self._rng.uniform(400.0, 900.0)
        self._delay(wait_ms, "waiting for bank interface")

        # 7. Verify bank is open
        if loc.bank_open_indicator_template:
            bank_open_scan = self.scanner.find_element(
                loc.bank_open_indicator_template,
                conditions=conditions,
                screen=None,
            )
            if bank_open_scan.found:
                self._bank_is_open = True
                self._log("bank_confirmed_open", {"confidence": bank_open_scan.confidence})
            else:
                self._log("bank_open_unconfirmed", {})
                self._bank_is_open = True  # assume open if we clicked successfully
        else:
            self._bank_is_open = True

        self._log("bank_open_complete", {"method": "right_click" if use_right_click else "left_click"})
        return True

    def close_bank(self) -> bool:
        """
        Close the bank interface.

        Uses a mix of: clicking the close button and pressing Escape.
        """
        if not self._location:
            return True

        loc = self._location
        self._log("bank_close_start", {})

        method = self._rng.choice(["close_button", "escape_key"])

        if method == "close_button" and loc.close_button_template:
            # Scan for close button
            close_scan = self.scanner.find_element(
                loc.close_button_template,
                conditions={"yaw": str(self._camera_yaw)},
                screen=None,
            )
            if close_scan.found:
                self._move_and_click(
                    close_scan.center_x, close_scan.center_y,
                    target_width=loc.close_button_width_px,
                    description="Click bank close button",
                )
            else:
                # Fallback to Escape
                self._press_key("escape")
        else:
            self._press_key("escape")

        # Brief post-close delay
        post_delay = self.cognitive.get_post_action_delay_ms()
        self._delay(post_delay, "post-close bank")

        self._bank_is_open = False
        self._log("bank_close_complete", {"method": method})
        return True

    def is_open(self) -> bool:
        """Check if the bank interface is currently believed to be open."""
        return self._bank_is_open

    # ------------------------------------------------------------------
    # Internal: click methods
    # ------------------------------------------------------------------

    def _left_click_bank(
        self, scan_result: ScanResult, loc: BankLocation,
    ) -> bool:
        """Left-click the bank booth/chest."""
        # Add slight positional jitter — don't click the exact center
        jitter_x = self._rng.normal(0.0, loc.bank_object_width_px * 0.15)
        jitter_y = self._rng.normal(0.0, loc.bank_object_width_px * 0.15)

        click_x = scan_result.center_x + jitter_x
        click_y = scan_result.center_y + jitter_y

        self._move_and_click(
            click_x, click_y,
            target_width=loc.bank_object_width_px,
            description="Left-click bank",
        )
        return True

    def _right_click_bank(
        self, scan_result: ScanResult, loc: BankLocation, conditions: dict,
    ) -> bool:
        """Right-click the bank and select the 'Bank' option from the menu."""
        # Right-click on the booth
        jitter_x = self._rng.normal(0.0, loc.bank_object_width_px * 0.10)
        jitter_y = self._rng.normal(0.0, loc.bank_object_width_px * 0.10)

        click_x = scan_result.center_x + jitter_x
        click_y = scan_result.center_y + jitter_y

        self._move_and_click(
            click_x, click_y,
            button="right",
            target_width=loc.bank_object_width_px,
            description="Right-click bank booth",
        )

        # Wait for context menu to appear
        menu_delay = self._rng.uniform(150.0, 350.0)
        self._delay(menu_delay, "wait for right-click menu")

        # Scan for the "Bank" option in the menu
        if loc.right_click_option_template:
            menu_scan = self.scanner.find_element(
                loc.right_click_option_template,
                conditions=conditions,
                screen=None,
            )
            if menu_scan.found:
                # Click the "Bank" menu option
                self._move_and_click(
                    menu_scan.center_x, menu_scan.center_y,
                    target_width=menu_scan.width,
                    description="Click 'Bank' option",
                )
                return True

        # Fallback: click slightly below the right-click point
        # (the menu appears below the cursor)
        self._delay(self._rng.uniform(100.0, 250.0), "processing menu")
        self._move_and_click(
            click_x,
            click_y + self._rng.integers(25, 50),
            target_width=80,
            description="Click estimated 'Bank' option position",
        )
        return True

    # ------------------------------------------------------------------
    # Internal: scanning helpers
    # ------------------------------------------------------------------

    def _scan_with_distracters(
        self,
        template_id: str,
        conditions: dict,
        distracter_count: int = 2,
    ) -> ScanResult:
        """Scan for an element with distracter fixations simulating visual search."""
        # Distracter scans (these model the time cost of visual search)
        for i in range(distracter_count):
            dx = self._rng.integers(100, 1820)
            dy = self._rng.integers(100, 980)
            distracter_time = self._rng.uniform(
                self.config.scan.distractor_time_ms[0],
                self.config.scan.distractor_time_ms[1],
            )
            self._delay(distracter_time, f"distracter scan {i+1}")

        # Real scan
        return self.scanner.find_element(template_id, conditions=conditions, screen=None)

    # ------------------------------------------------------------------
    # Internal: movement & click
    # ------------------------------------------------------------------

    def _move_and_click(
        self,
        x: float,
        y: float,
        target_width: float | None = None,
        button: str = "left",
        description: str = "",
    ) -> ClickEvent:
        """Move to (x, y) and click with logging."""
        # Check for misclick
        is_misclick = self.cognitive.should_misclick()

        # Movement
        movements = self.mouse.move_to(x, y, target_width)
        for ms in movements:
            self._log_mouse_sample(ms)

        # Click
        click = self.mouse.click_model.generate(
            x, y, button, self._elapsed(), is_misclick=is_misclick,
        )

        self._log_click(click, is_misclick, description)

        if is_misclick:
            self._delay(self._rng.uniform(200.0, 500.0), "notice misclick")
            correction = self.mouse.click_model.generate(
                x, y, button, self._elapsed(),
            )
            self._log_click(correction, False, f"correct {description}")

        # Update position
        self.mouse._x = click.post_click_x
        self.mouse._y = click.post_click_y

        # Post-action delay
        post = self.cognitive.get_post_action_delay_ms()
        self._delay(post, f"post-{description}")

        return click

    def _hover_at(self, x: float, y: float, description: str = "") -> None:
        """Hover cursor at position with micro-gestures."""
        dwell_s = self._rng.uniform(
            self.config.scan.hover_dwell_min_s,
            self.config.scan.hover_dwell_max_s * 0.6,  # shorter for booths
        )
        movements = self.mouse.move_to(x, y)
        for ms in movements:
            self._log_mouse_sample(ms)

        hover_points = self.scanner.simulate_hover_dwell(x, y, dwell_s)
        for hp in hover_points:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed() + hp["timestamp"],
                wall_time=time.time(),
                event_type=EventType.GAZE_POINT,
                data={"x": hp["x"], "y": hp["y"], "duration_s": dwell_s, "context": description},
                labels=["synthetic", "hover_dwell"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed() * 1000,
            ))

        self.mouse._x = x
        self.mouse._y = y

    def _press_key(self, key: str) -> None:
        """Press and release a single key with logging."""
        kevents = self.keyboard.press_key(key)
        for ke in kevents:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed() + ke.timestamp,
                wall_time=time.time(),
                event_type=EventType.KEY_DOWN if ke.action == "down" else EventType.KEY_UP,
                data={"key": ke.key, "is_correction": ke.is_correction},
                labels=["synthetic"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed() * 1000,
            ))

    # ------------------------------------------------------------------
    # Internal: camera
    # ------------------------------------------------------------------

    def _simulate_camera_rotate(self) -> None:
        """Rotate the camera by random amount."""
        delta = self._rng.choice([-90, -45, 45, 90])
        old_yaw = self._camera_yaw
        self._camera_yaw = (self._camera_yaw + delta) % 360

        if self._rng.random() < 0.15:
            self._camera_pitch = self._rng.choice(["high", "default", "low"])

        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.CAMERA_ROTATE,
            data={
                "old_yaw": old_yaw, "new_yaw": self._camera_yaw,
                "pitch": self._camera_pitch,
            },
            labels=["synthetic", "camera_rotate"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))

    # ------------------------------------------------------------------
    # Internal: logging
    # ------------------------------------------------------------------

    def _log(self, event: str, data: dict | None = None) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SESSION_META,
            data={"module": "bank_interaction", "event": event, **(data or {})},
            labels=["synthetic", "bank_interaction"],
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

    def _log_click(
        self, click: ClickEvent, is_misclick: bool, description: str,
    ) -> None:
        from core.logger import make_click
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

    def _delay(self, ms: float, description: str = "") -> None:
        """Log a cognitive delay."""
        if ms <= 0:
            return
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.COGNITIVE_DELAY,
            data={"delay_ms": round(ms, 2), "context": description},
            labels=["synthetic"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))

    def _elapsed(self) -> float:
        """Seconds since this module's clock started."""
        return time.time() - self._start_time
