"""
Login Handler — logs into an OSRS account using screen‑space interaction.

All UI elements (username field, password field, buttons) are located via
template matching — no hardcoded coordinates.

Anti‑ban features:
  - Username typed at variable speed with possible typos + corrections
  - Password typed slightly slower (deliberate, no typos for passwords)
  - Tab-key navigation between fields (sometimes)
  - Hover delays over buttons before clicking
  - Gaze scans around the login screen (news panel, world select, etc.)
  - Variable cognitive delays between each step
  - Misclick handling when clicking buttons
  - Camera-like scanning of the login screen before starting

Credentials are read from environment variables:
  OSRS_USERNAME, OSRS_PASSWORD

Usage (from a scenario):
    from interactions.login import LoginHandler

    login = LoginHandler(mouse, scanner, cognitive, keyboard, logger, rng, config)
    success = login.login()
    if success:
        # ... proceed to game ...
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
from core.keyboard import KeyboardEmulator, KeystrokeEvent


# ---------------------------------------------------------------------------
# Template IDs for the login screen
# ---------------------------------------------------------------------------

LOGIN_TEMPLATES = {
    # Screen 1: Post‑logout / disconnected
    "disconnected_screen_indicator": "login_disconnected_screen",  # confirms we're on the disconnected screen
    "disconnected_continue_btn": "login_disconnected_continue",    # button to proceed to main login

    # Screen 2: Main login screen
    "login_screen_indicator": "login_screen_title",        # confirms we're on the main login screen
    "existing_user_button": "login_existing_user_btn",     # "Existing User" button
    "click_to_play_button": "login_click_to_play_btn",     # "Click here to play" red button

    # Optional — only needed for full‑typing mode (when credentials aren't cached)
    "username_field": "login_username_field",
    "password_field": "login_password_field",

    # Post‑login verification
    "game_loaded_indicator": "game_view_minimap",          # confirms we're in‑game
    "world_selector": "login_world_selector",              # world select button (optional)
}


class LoginHandler:
    """
    Handles the full OSRS login flow via template‑matched UI interaction.

    Flow:
      1. Verify we're on the login screen
      2. Click "Existing User" (if needed)
      3. Click username field → type username
      4. Click password field → type password
      5. Click "Click here to play"
      6. Wait for game to load → verify in-game
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

        # Credentials from env (optional — only needed if username isn't cached)
        self._username = os.getenv("OSRS_USERNAME", "")
        self._password = os.getenv("OSRS_PASSWORD", "")

        # Whether we need to type credentials (False = username is cached by game client)
        self._needs_typing = bool(self._username and self._password)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self, quick: bool = True) -> bool:
        """
        Execute the full login sequence — handles both post‑logout screens.

        Flow:
          Screen 1 (if present): "Disconnected" → click continue
          Screen 2: Main login → click "Existing User" → click "Play Now"

        In quick mode (default): assumes username is cached by the game client.
        Just clicks the buttons — no typing.
        """
        self._log("login_start", {"mode": "quick" if quick else "full"})

        # ================================================================
        # Screen 1: Handle post‑logout / disconnected screen
        # ================================================================
        disconnected = self._scan(
            LOGIN_TEMPLATES["disconnected_screen_indicator"],
            "check for disconnected screen",
        )

        if disconnected.found:
            self._log("screen_detected", {"screen": "disconnected"})

            # Brief orienting pause
            self._delay(400, 1000, "orient to disconnected screen")

            # Find and click the continue/return button
            continue_btn = self._scan(
                LOGIN_TEMPLATES["disconnected_continue_btn"],
                "locate continue button on disconnected screen",
            )

            if continue_btn.found:
                # Hover then click
                if self._rng.random() < 0.5:
                    self._hover_at(continue_btn.center_x, continue_btn.center_y,
                                   "hover over continue button")

                self._move_and_click(
                    continue_btn.center_x, continue_btn.center_y,
                    target_width=continue_btn.width,
                    description="Click continue (disconnected screen)",
                )

                # Wait for main login screen to appear
                self._delay(800, 2000, "wait for main login screen after disconnected")
            else:
                self._log("login_warning", {"step": "continue_button_not_found"})
                # Click center of the button area as fallback
                self._move_and_click(
                    disconnected.center_x,
                    disconnected.center_y + self._rng.integers(50, 120),
                    target_width=150,
                    description="Click estimated continue position",
                )
                self._delay(800, 2000, "wait for transition")

        # ================================================================
        # Screen 2: Main login screen
        # ================================================================
        self._delay(300, 800, "orient to login screen")

        # Verify we're on the main login screen
        login_screen = self._scan(
            LOGIN_TEMPLATES["login_screen_indicator"],
            "verify login screen visible",
        )

        if not login_screen.found:
            # Maybe already in game?
            game_check = self._scan(
                LOGIN_TEMPLATES["game_loaded_indicator"],
                "check if already in game",
            )
            if game_check.found:
                self._log("login_skipped", {"reason": "already_in_game"})
                return True

            # Try once more — maybe the screen is still transitioning
            self._delay(500, 1500, "wait and retry login screen detection")
            login_screen = self._scan(
                LOGIN_TEMPLATES["login_screen_indicator"],
                "retry verify login screen",
            )
            if not login_screen.found:
                self._log("login_error", {"step": "login_screen_not_found"})
                return False

        self._log("screen_detected", {"screen": "main_login"})

        # ---- Gaze around (anti‑ban) ----
        self._gaze_around_login_screen()

        # ---- Click "Existing User" ----
        existing_btn = self._scan(
            LOGIN_TEMPLATES["existing_user_button"],
            "locate Existing User button",
        )
        if existing_btn.found:
            if self._rng.random() < 0.5:
                self._hover_at(existing_btn.center_x, existing_btn.center_y,
                               "hover over Existing User")
                self._delay(100, 400, "decide to click Existing User")

            self._move_and_click(
                existing_btn.center_x, existing_btn.center_y,
                target_width=existing_btn.width,
                description="Click Existing User",
            )
            self._delay(300, 800, "wait after clicking Existing User")
        else:
            self._log("login_warning", {"step": "existing_user_button_not_found"})

        # ---- Type credentials (only in full mode) ----
        if not quick and self._needs_typing:
            self._fill_field(
                LOGIN_TEMPLATES["username_field"],
                self._username,
                field_name="username",
                typo_enabled=True,
                use_tab_to_next=True,
            )
            self._fill_field(
                LOGIN_TEMPLATES["password_field"],
                self._password,
                field_name="password",
                typo_enabled=False,
                use_tab_to_next=False,
            )
            self._delay(400, 1200, "review login details")

        # ---- Pause then click Play ----
        self._delay(200, 600, "check login screen ready")

        play_btn = self._scan(
            LOGIN_TEMPLATES["click_to_play_button"],
            "locate Click to Play button",
        )

        if not play_btn.found:
            self._log("login_error", {"step": "play_button_not_found"})
            return False

        if self._rng.random() < 0.4:
            self._hover_at(play_btn.center_x, play_btn.center_y, "hover over Play button")
            self._delay(100, 400, "decide to click play")

        self._move_and_click(
            play_btn.center_x, play_btn.center_y,
            target_width=play_btn.width,
            description="Click Here to Play",
        )

        # ================================================================
        # Wait for game load
        # ================================================================
        self._log("waiting_for_game_load", {})
        load_wait_s = self._rng.uniform(3.0, 8.0)
        self._delay(load_wait_s * 1000, load_wait_s * 1000 + 2000, "game loading")

        # Verify in-game
        game_check = self._scan(
            LOGIN_TEMPLATES["game_loaded_indicator"],
            "verify game loaded",
        )

        if game_check.found:
            self._log("login_success", {"load_time_s": load_wait_s,
                       "mode": "quick" if quick else "full"})
            return True
        else:
            self._log("login_unconfirmed", {"load_time_s": load_wait_s})
            return True  # optimistic

    # ------------------------------------------------------------------
    # Internal: field filling
    # ------------------------------------------------------------------

    def _fill_field(
        self,
        template_id: str,
        text: str,
        field_name: str = "",
        typo_enabled: bool = True,
        use_tab_to_next: bool = True,
    ) -> None:
        """
        Click a text field, type text into it, with full anti‑ban behaviour.
        """
        # Scan for the field
        field = self._scan(template_id, f"locate {field_name} field")

        if not field.found:
            self._log("field_not_found", {"field": field_name})
            return

        # Move to field and click to focus it
        self._move_and_click(
            field.center_x, field.center_y,
            target_width=field.width,
            description=f"Click {field_name} field",
        )

        # Brief pause — some users pause before typing
        self._delay(200, 600, f"prepare to type {field_name}")

        # Clear existing text if any (Ctrl+A then type over, or just click)
        if self._rng.random() < 0.3:
            # Some users clear the field first
            ctrl_a = self.keyboard.press_hotkey(["ctrl"], "a")
            for ke in ctrl_a:
                self._log_keystroke(ke)

        # Type the text
        typo_rate = self.cognitive.persona.typo_rate if typo_enabled else 0.0
        keystrokes = self.keyboard.type_text(
            text,
            typo_enabled=typo_enabled,
            typo_rate_override=typo_rate,
        )
        for ke in keystrokes:
            self._log_keystroke(ke)

        # Tab to next field (sometimes — not always)
        if use_tab_to_next and self._rng.random() < 0.5:
            self._delay(100, 300, "pause before tab")
            tab_events = self.keyboard.press_key("tab")
            for ke in tab_events:
                self._log_keystroke(ke)

        self._delay(150, 400, f"post-{field_name} pause")

    # ------------------------------------------------------------------
    # Internal: gaze simulation
    # ------------------------------------------------------------------

    def _gaze_around_login_screen(self) -> None:
        """
        Simulate scanning the login screen — looking at news panel,
        world selector, random areas.  Pure anti‑ban.
        """
        gaze_points = [
            (float(self._rng.integers(100, 500)), float(self._rng.integers(200, 600)), "look at news panel"),
            (float(self._rng.integers(600, 900)), float(self._rng.integers(100, 300)), "glance at world selector"),
            (float(self._rng.integers(300, 700)), float(self._rng.integers(400, 700)), "scan login form"),
        ]

        # Only use 1-2 gaze points (not all three — that would be too consistent)
        n_gaze = min(2, len(gaze_points))
        indices = self._rng.choice(len(gaze_points), size=n_gaze, replace=False)
        for idx in indices:
            x, y, desc = gaze_points[int(idx)]
            self.mouse.move_to(float(x), float(y))
            dwell = self._rng.uniform(0.3, 1.2)
            hover_points = self.scanner.simulate_hover_dwell(float(x), float(y), dwell)
            for hp in hover_points:
                self.logger.log(TelemetryEvent(
                    session_id=self.logger.session_id,
                    timestamp=self._elapsed() + hp["timestamp"],
                    wall_time=time.time(),
                    event_type=EventType.GAZE_POINT,
                    data={"x": hp["x"], "y": hp["y"], "duration_s": dwell, "context": desc},
                    labels=["synthetic", "gaze", "login"],
                    persona=self.cognitive.persona.type.value,
                    session_phase=self.cognitive.current_phase.value,
                    affective_state=self.cognitive.current_affect.value,
                    elapsed_ms=self._elapsed() * 1000,
                ))
            self.mouse._x, self.mouse._y = float(x), float(y)

    # ------------------------------------------------------------------
    # Internal: movement, scan, timing
    # ------------------------------------------------------------------

    def _move_and_click(
        self, x: float, y: float, target_width: float | None = None,
        button: str = "left", description: str = "", hover_first: bool = False,
    ) -> ClickEvent:
        """Move to (x, y) and click with full anti‑ban logging."""
        is_misclick = self.cognitive.should_misclick()

        if hover_first:
            self._hover_at(x, y, f"hover before {description}")

        movements = self.mouse.move_to(x, y, target_width)
        for ms in movements:
            self._log_mouse_sample(ms)

        click = self.mouse.click_model.generate(
            x, y, button, self._elapsed(), is_misclick=is_misclick,
        )
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
        dwell_s = self._rng.uniform(0.5, 1.5)
        self.mouse.move_to(x, y)
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
        self.mouse._x, self.mouse._y = x, y

    def _scan(self, template_id: str, description: str = "") -> ScanResult:
        result = self.scanner.find_element(template_id, conditions={}, screen=None)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SCREEN_SCAN,
            data={
                "template_id": template_id, "found": result.found,
                "center_x": result.center_x, "center_y": result.center_y,
                "confidence": result.confidence, "method": result.method.value,
                "description": description,
            },
            labels=["synthetic", "login"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))
        return result

    def _delay(self, min_ms: float, max_ms: float, description: str = "") -> None:
        delay_ms = self._rng.uniform(min_ms, max_ms)
        if delay_ms <= 0:
            return
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.COGNITIVE_DELAY,
            data={"delay_ms": round(delay_ms, 2), "context": description},
            labels=["synthetic", "login"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed() * 1000,
        ))

    def _log(self, event: str, data: dict | None = None) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SESSION_META,
            data={"module": "login", "event": event, **(data or {})},
            labels=["synthetic", "login"],
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

    def _log_keystroke(self, ke: KeystrokeEvent) -> None:
        from core.logger import make_key_event
        self.logger.log(make_key_event(
            EventType.KEY_DOWN if ke.action == "down" else EventType.KEY_UP,
            self.logger.session_id, self._elapsed() + ke.timestamp,
            time.time(), ke.key, ke.is_correction,
            self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed() * 1000,
        ))

    def _elapsed(self) -> float:
        return time.time() - self._start_time
