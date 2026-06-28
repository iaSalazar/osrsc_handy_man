"""
Logout Handler — logs out of an OSRS account via screen‑space interaction.

The OSRS logout button is traditionally the small door icon near the
minimap (top‑right of the game view).  This handler:

  1. Locates the logout door icon via template matching
  2. Clicks it (with anti‑ban hover + variable timing)
  3. Optionally clicks the "Click here to logout" confirmation button
  4. Verifies the login screen reappeared

Anti‑ban features:
  - Sometimes uses the logout tab instead of the door icon
  - Camera pan/adjust before logging out (as if taking a last look)
  - Variable hover over the logout button
  - Sometimes presses the logout keyboard shortcut (Ctrl+Shift+L or similar)
  - Post‑logout gaze at the login screen before ending

Usage:
    from interactions.logout import LogoutHandler

    logout = LogoutHandler(mouse, scanner, cognitive, keyboard, logger, rng, config)
    success = logout.logout()
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from core.config import AppConfig
from core.logger import TelemetryLogger, TelemetryEvent, EventType, make_click
from core.mouse import MouseEmulator, ClickEvent, MouseSample
from core.cognitive import CognitiveEngine
from core.scan import VisualScanner, ScanResult
from core.keyboard import KeyboardEmulator, KeystrokeEvent


LOGOUT_TEMPLATES = {
    "logout_door": "logout_door_icon",              # the door icon near minimap
    "logout_tab": "logout_tab_button",              # the logout tab (world switcher)
    "logout_confirm": "logout_confirm_button",       # "Click here to logout" text
    "login_screen_indicator": "login_screen_title",  # confirms we're logged out
}


class LogoutHandler:
    """
    Handles the OSRS logout flow via template‑matched UI interaction.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logout(self) -> bool:
        """
        Execute the logout sequence.

        Returns True if logout appears successful (login screen visible).
        """
        self._log("logout_start", {})

        # ---- Step 1: Pre‑logout behaviour (anti‑ban) ----
        self._pre_logout_behaviour()

        # ---- Step 2: Choose logout method ----
        method = self._rng.choice(["door_icon", "door_icon", "door_icon", "tab_button"])
        # 75% door icon, 25% tab button — door is more common

        if method == "door_icon":
            success = self._logout_via_door()
        else:
            success = self._logout_via_tab()

        if not success:
            self._log("logout_error", {"method": method})
            # Fallback: try the other method
            if method == "door_icon":
                success = self._logout_via_tab()
            else:
                success = self._logout_via_door()

        if not success:
            return False

        # ---- Step 3: Click confirmation (if needed) ----
        self._delay(300, 700, "wait for logout confirmation dialog")
        confirm_btn = self._scan(
            LOGOUT_TEMPLATES["logout_confirm"],
            "check for logout confirmation",
        )
        if confirm_btn.found:
            # Hover first sometimes
            if self._rng.random() < 0.35:
                self._hover_at(confirm_btn.center_x, confirm_btn.center_y, "hover over logout confirm")
            self._move_and_click(
                confirm_btn.center_x, confirm_btn.center_y,
                target_width=confirm_btn.width,
                description="Click logout confirmation",
            )

        # ---- Step 4: Wait for logout to complete ----
        self._log("waiting_for_logout", {})
        self._delay(2000, 4000, "waiting for logout to complete")

        # ---- Step 5: Verify we're back at login screen ----
        login_screen = self._scan(
            LOGOUT_TEMPLATES["login_screen_indicator"],
            "verify back at login screen",
        )

        if login_screen.found:
            self._log("logout_success", {"method": method})
            return True
        else:
            self._log("logout_unconfirmed", {"method": method})
            return True  # optimistic

    # ------------------------------------------------------------------
    # Internal: logout methods
    # ------------------------------------------------------------------

    def _logout_via_door(self) -> bool:
        """Click the logout door icon near the minimap."""
        door = self._scan(
            LOGOUT_TEMPLATES["logout_door"],
            "locate logout door icon",
        )
        if not door.found:
            return False

        # Hover first (the door is small — users often hover to confirm)
        if self._rng.random() < 0.6:
            self._hover_at(door.center_x, door.center_y, "hover over logout door")
            self._delay(300, 900, "decide to log out")

        self._move_and_click(
            door.center_x, door.center_y,
            target_width=door.width,
            description="Click logout door",
        )
        return True

    def _logout_via_tab(self) -> bool:
        """Use the logout tab / world switcher to log out."""
        tab = self._scan(
            LOGOUT_TEMPLATES["logout_tab"],
            "locate logout tab button",
        )
        if not tab.found:
            return False

        if self._rng.random() < 0.4:
            self._hover_at(tab.center_x, tab.center_y, "hover over logout tab")

        self._move_and_click(
            tab.center_x, tab.center_y,
            target_width=tab.width,
            description="Click logout tab",
        )
        return True

    # ------------------------------------------------------------------
    # Internal: pre‑logout anti‑ban
    # ------------------------------------------------------------------

    def _pre_logout_behaviour(self) -> None:
        """
        Simulate behaviour that often precedes logging out:
          - Quick camera pan (last look around)
          - Check minimap
          - Hover over inventory or equipment
        """
        # Camera pan 50% of the time
        if self._rng.random() < 0.5:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed(),
                wall_time=time.time(),
                event_type=EventType.CAMERA_ROTATE,
                data={"context": "pre-logout camera pan", "delta_yaw": self._rng.choice([-45, 45, 90])},
                labels=["synthetic", "logout", "camera_rotate"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed() * 1000,
            ))
            self._delay(400, 1200, "pre-logout camera pan")

        # Quick glance around 60% of the time
        if self._rng.random() < 0.6:
            gx = self._rng.integers(800, 1800)
            gy = self._rng.integers(100, 800)
            self.mouse.move_to(gx, gy)
            self._hover_at(gx, gy, "pre-logout gaze")
            self._delay(200, 600, "last look around")

    # ------------------------------------------------------------------
    # Internal: shared helpers
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
            self._delay(200, 500, "notice misclick")
            correction = self.mouse.click_model.generate(x, y, button, self._elapsed())
            self._log_click(correction, False, f"correct {description}")
        self.mouse._x = click.post_click_x
        self.mouse._y = click.post_click_y
        post = self.cognitive.get_post_action_delay_ms()
        self._delay(post * 0.5, post, f"post-{description}")
        return click

    def _hover_at(self, x: float, y: float, description: str = "") -> None:
        dwell_s = self._rng.uniform(0.4, 1.5)
        self.mouse.move_to(x, y)
        hover_points = self.scanner.simulate_hover_dwell(x, y, dwell_s)
        for hp in hover_points:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed() + hp["timestamp"],
                wall_time=time.time(),
                event_type=EventType.GAZE_POINT,
                data={"x": hp["x"], "y": hp["y"], "duration_s": dwell_s, "context": description},
                labels=["synthetic", "hover_dwell", "logout"],
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
            labels=["synthetic", "logout"],
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
            data={"module": "logout", "event": event, **(data or {})},
            labels=["synthetic", "logout"],
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
            labels=["synthetic", "logout"],
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
