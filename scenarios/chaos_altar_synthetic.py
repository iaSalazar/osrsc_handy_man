"""
Chaos Altar Dragon Bones — Synthetic Mode Scenario.

This is the ORIGINAL synthetic-mode scenario for bulk training data generation.
It uses the full core stack (MouseEmulator, CognitiveEngine, VisualScanner,
KeyboardEmulator, TelemetryLogger) with stored bounding boxes and Gaussian
position noise — NO screen capture, NO mouse movement.

For live mode, use ChaosAltarScenario in chaos_altar_dragon_bones.py instead.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import AppConfig, ScenarioState, PersonaType
from core.logger import (
    TelemetryLogger, TelemetryEvent, EventType,
    make_mouse_move, make_click, make_key_event,
)
from core.mouse import MouseEmulator, ClickEvent, MouseSample
from core.cognitive import CognitiveEngine, Interruption
from core.scan import VisualScanner, ScanResult
from core.keyboard import KeyboardEmulator, KeystrokeEvent


# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

BONES_PER_INVENTORY = 28
BONE_SAVE_PROBABILITY = 0.50
DEFAULT_CYCLES = 10

TEMPLATE_IDS = {
    "bank_chest": "bank_chest",
    "bank_interface_open": "bank_interface_open",
    "dragon_bones_bank": "dragon_bones_bank",
    "dragon_bones_inventory": "dragon_bones_inventory",
    "close_button": "close_button",
    "minimap_altar_icon": "minimap_altar_icon",
    "wilderness_gate": "wilderness_gate",
    "wilderness_gate_open": "wilderness_gate_open",
    "chaos_altar_distant": "chaos_altar_distant",
    "chaos_altar_medium": "chaos_altar_medium",
    "chaos_altar_close": "chaos_altar_close",
    "offering_animation": "offering_animation",
    "prayer_xp_drop": "prayer_xp_drop",
    "inventory_full": "inventory_full",
    "teleport_item": "teleport_item",
    "run_energy_orb": "run_energy_orb",
    "minimap_bank_icon": "minimap_bank_icon",
}


# ---------------------------------------------------------------------------
# Camera state tracking
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    yaw: int = 0
    pitch: str = "default"
    zoom: str = "mid"

    def to_conditions(self) -> dict:
        return {"yaw": str(self.yaw), "pitch": self.pitch, "zoom": self.zoom}

    def rotate(self, rng: np.random.Generator) -> dict:
        delta = rng.choice([-90, -45, 45, 90])
        self.yaw = (self.yaw + delta) % 360
        if rng.random() < 0.2:
            self.pitch = rng.choice(["high", "default", "low"])
        if rng.random() < 0.15:
            self.zoom = rng.choice(["in", "mid", "out"])
        return self.to_conditions()


# ---------------------------------------------------------------------------
# Scenario class
# ---------------------------------------------------------------------------

class ChaosAltarDragonBonesScenario:
    """
    Full scenario state machine — SYNTHETIC MODE.

    Uses stored template bounding boxes with Gaussian position noise.
    No screen capture, no mouse movement. Generates JSONL telemetry
    for anomaly detector training.
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
        cycles: int = DEFAULT_CYCLES,
        synthetic: bool = True,
    ) -> None:
        self.mouse = mouse
        self.scanner = scanner
        self.cognitive = cognitive
        self.keyboard = keyboard
        self.logger = logger
        self._rng = rng
        self.config = config
        self.num_cycles = cycles
        self.synthetic = synthetic

        from interactions.bank_interaction import BankInteraction
        self.bank = BankInteraction(
            mouse, scanner, cognitive, keyboard, logger, rng, config,
        )

        self.current_state: ScenarioState = ScenarioState.STATE_BANKING
        self.cycle_count = 0
        self.bones_offered_this_cycle = 0
        self.bones_saved_total = 0
        self.total_bones_used = 0
        self.camera = CameraState()
        self._active_interruption: Optional[Interruption] = None
        self._session_start_s: float = 0.0
        self._state_entry_times: dict[ScenarioState, float] = {}

    # ==================================================================
    # Main loop
    # ==================================================================

    def run_session(self) -> dict:
        self._session_start_s = time.time()
        self._log_meta("session_start", {
            "scenario": "chaos_altar_dragon_bones",
            "cycles": self.num_cycles,
            "persona": self.cognitive.persona.type.value,
            "synthetic": self.synthetic,
        })

        for cycle in range(self.num_cycles):
            self.cycle_count = cycle

            self._transition_to(ScenarioState.STATE_BANKING)
            self._run_banking()

            self._transition_to(ScenarioState.STATE_TRAVELING)
            self._run_traveling()

            self._transition_to(ScenarioState.STATE_OFFERING)
            self._run_offering()

            self._transition_to(ScenarioState.STATE_RETURNING)
            self._run_returning()

            self._transition_to(ScenarioState.STATE_IDLE)
            self._run_idle()

            if self._is_short_checkin():
                self._log_meta("short_checkin_end", {"cycles_completed": cycle + 1})
                break

        self._transition_to(ScenarioState.STATE_IDLE)
        self._log_meta("session_end", self._session_summary())
        self.logger.flush()
        return self._session_summary()

    # ==================================================================
    # State: Banking
    # ==================================================================

    def _run_banking(self) -> None:
        self._log_state("banking_start")
        success = self.bank.open_bank(
            location="generic",
            camera_conditions=self.camera.to_conditions(),
        )
        if not success:
            self._log_state("banking_error", {"error": "bank_not_found"})
            self.cognitive.record_error()
            return

        self._interact_with_element(
            TEMPLATE_IDS["dragon_bones_bank"],
            click_type="right",
            hover_first=True,
            description="Right-click dragon bones in bank",
        )

        self._cognitive_delay(200, 500, "select withdraw-all option")
        bones_result = self.scanner.find_element(
            TEMPLATE_IDS["dragon_bones_bank"],
            conditions=self.camera.to_conditions(),
            screen=None,
        )
        if bones_result.found:
            menu_y = bones_result.center_y + self._rng.integers(30, 60)
            self._move_and_click(
                bones_result.center_x + self._rng.integers(-5, 5),
                menu_y,
                description="Click Withdraw-All",
            )

        self._cognitive_delay(300, 800, "verify inventory full")
        self._scan_element(
            TEMPLATE_IDS["inventory_full"], "verify inventory has 28 bones"
        )
        self.bank.close_bank()
        self.total_bones_used += BONES_PER_INVENTORY
        self._log_state("banking_complete", {"bones_withdrawn": BONES_PER_INVENTORY})

    # ==================================================================
    # State: Traveling
    # ==================================================================

    def _run_traveling(self) -> None:
        self._log_state("traveling_start")
        if self.cognitive.should_rotate_camera():
            self._rotate_camera()

        self._interact_with_element(
            TEMPLATE_IDS["minimap_altar_icon"],
            click_type="left",
            description="Click minimap: Chaos Altar",
        )

        travel_phases = self._rng.integers(2, 5)
        for i in range(travel_phases):
            travel_time_s = self._rng.uniform(2.0, 5.0)
            self._simulate_time_passage(travel_time_s, f"travel phase {i+1}/{travel_phases}")
            if self._rng.random() < 0.3:
                self._scan_element(
                    TEMPLATE_IDS["minimap_altar_icon"],
                    f"wilderness safety check {i+1}"
                )
            if self._rng.random() < 0.2:
                self._hover_over_element(
                    TEMPLATE_IDS["run_energy_orb"],
                    "check run energy",
                )
            if self.cognitive.should_go_afk():
                self._simulate_afk()

        self._interact_with_element(
            TEMPLATE_IDS["wilderness_gate"],
            click_type="left",
            hover_first=True,
            description="Click wilderness gate",
        )
        self._cognitive_delay(300, 700, "wait for gate to open")
        self._scan_element(TEMPLATE_IDS["wilderness_gate_open"], "verify gate opened")
        self._scan_element(TEMPLATE_IDS["chaos_altar_medium"], "spot altar at medium range")

        if self._rng.random() < 0.25:
            self._rotate_camera()

        self._interact_with_element(
            TEMPLATE_IDS["chaos_altar_close"],
            click_type="left",
            description="Move to altar (close range)",
        )
        self._log_state("traveling_complete")

    # ==================================================================
    # State: Offering
    # ==================================================================

    def _run_offering(self) -> None:
        self._log_state("offering_start")
        self.bones_offered_this_cycle = 0

        for bone_idx in range(BONES_PER_INVENTORY):
            micro_break = self.cognitive.get_micro_break()
            if micro_break is not None:
                self._simulate_micro_break(micro_break)

            interruption = self.cognitive.get_interruption(dt_s=5.0)
            if interruption is not None:
                self._handle_interruption(interruption)

            self.cognitive.update(dt_s=self._rng.uniform(1.5, 3.5))

            bones_inv = self._scan_element(
                TEMPLATE_IDS["dragon_bones_inventory"],
                f"locate bones in inventory (bone {bone_idx+1}/28)",
            )
            if not bones_inv.found:
                self._log_state("offering_error", {"error": "bones_not_found"})
                continue

            self._move_and_click(
                bones_inv.center_x, bones_inv.center_y,
                description=f"Click dragon bones ({bone_idx+1}/28)",
            )
            rt = self.cognitive.get_reaction_delay_ms()
            self._cognitive_delay(rt * 0.5, rt, "processing: bones selected")

            altar = self._scan_element(
                TEMPLATE_IDS["chaos_altar_close"],
                f"locate altar for offering (bone {bone_idx+1}/28)",
            )
            if not altar.found:
                if self._rng.random() < 0.5:
                    self._rotate_camera()
                    altar = self._scan_element(
                        TEMPLATE_IDS["chaos_altar_close"],
                        "re-scan altar after camera rotate",
                    )

            if altar.found:
                self._move_and_click(
                    altar.center_x, altar.center_y,
                    description=f"Click altar ({bone_idx+1}/28)",
                )

            anim_wait_ms = self._rng.uniform(400.0, 800.0)
            self._cognitive_delay(anim_wait_ms, anim_wait_ms + 200.0, f"offering animation ({bone_idx+1}/28)")
            self._scan_element(TEMPLATE_IDS["offering_animation"], "confirm offering animating")

            bone_saved = self._rng.random() < BONE_SAVE_PROBABILITY
            if bone_saved:
                self.bones_saved_total += 1

            self.bones_offered_this_cycle += 1

            if self._rng.random() < 0.4:
                self._scan_element(
                    TEMPLATE_IDS["prayer_xp_drop"],
                    f"check XP drop ({bone_idx+1}/28)",
                )

            if self._rng.random() < 0.3:
                hover_target = self._rng.choice([
                    TEMPLATE_IDS["dragon_bones_inventory"],
                    TEMPLATE_IDS["chaos_altar_close"],
                ])
                self._hover_over_element(hover_target, "idle hover during offering")

            error_occurred = (not bones_inv.found or not altar.found)
            self.cognitive.update(dt_s=self._rng.uniform(2.0, 4.0), error_occurred=error_occurred)

        self._log_state("offering_complete", {
            "bones_offered": self.bones_offered_this_cycle,
            "bones_saved_this_cycle": sum(
                1 for _ in range(self.bones_offered_this_cycle)
                if self._rng.random() < BONE_SAVE_PROBABILITY
            ),
        })

    # ==================================================================
    # State: Returning
    # ==================================================================

    def _run_returning(self) -> None:
        self._log_state("returning_start")
        method = self._rng.choice(["teleport", "run"])
        if method == "teleport":
            self._interact_with_element(
                TEMPLATE_IDS["teleport_item"],
                click_type="left",
                hover_first=True,
                description="Click teleport item",
            )
            teleport_wait_s = self._rng.uniform(2.0, 4.0)
            self._simulate_time_passage(teleport_wait_s, "teleport animation")
            self._scan_element(TEMPLATE_IDS["minimap_bank_icon"], "verify arrived at bank")
        else:
            self._interact_with_element(
                TEMPLATE_IDS["minimap_bank_icon"],
                click_type="left",
                description="Click minimap: bank",
            )
            run_phases = self._rng.integers(1, 3)
            for i in range(run_phases):
                self._simulate_time_passage(
                    self._rng.uniform(3.0, 6.0),
                    f"return run phase {i+1}",
                )
        self._log_state("returning_complete", {"method": method})

    # ==================================================================
    # State: Idle
    # ==================================================================

    def _run_idle(self) -> None:
        idle_duration_s = self._rng.uniform(3.0, 10.0)
        if self.cognitive.should_go_afk():
            self._simulate_afk()
            idle_duration_s = max(idle_duration_s, 15.0)
        if self.cognitive.should_rotate_camera():
            self._rotate_camera()

        hover_targets = [
            TEMPLATE_IDS["prayer_xp_drop"],
            TEMPLATE_IDS["run_energy_orb"],
            TEMPLATE_IDS["dragon_bones_inventory"],
        ]
        target = self._rng.choice(hover_targets)
        self._hover_over_element(target, "idle hover")

        if self._rng.random() < 0.3:
            self._scan_element(TEMPLATE_IDS["minimap_altar_icon"], "idle minimap check")

        self._simulate_time_passage(idle_duration_s, "between-cycle idle")
        self.cognitive.update(dt_s=idle_duration_s)
        self._log_state("idle_complete", {"idle_duration_s": idle_duration_s})

    # ==================================================================
    # Interaction helpers
    # ==================================================================

    def _interact_with_element(
        self,
        template_id: str,
        click_type: str = "left",
        hover_first: bool = False,
        description: str = "",
    ) -> Optional[ClickEvent]:
        rt_ms = self.cognitive.get_reaction_delay_ms() if self._rng.random() > 0.3 else 0.0
        if rt_ms > 0:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed(),
                wall_time=time.time(),
                event_type=EventType.REACTION_TIME,
                data={"delay_ms": rt_ms, "context": description},
                labels=["synthetic", f"state:{self.current_state.value}"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed_ms(),
            ))

        inter = self.cognitive.get_interruption(dt_s=5.0)
        if inter is not None:
            self._handle_interruption(inter)

        result = self._scan_element(template_id, description)
        if not result.found:
            self.cognitive.record_error()
            return None

        if hover_first or self._rng.random() < 0.3:
            self._hover_at_position(result.center_x, result.center_y)

        return self._move_and_click(
            result.center_x, result.center_y,
            target_width=result.width,
            description=description,
        )

    def _move_and_click(
        self,
        x: float,
        y: float,
        target_width: float | None = None,
        button: str = "left",
        description: str = "",
    ) -> ClickEvent:
        is_misclick = self.cognitive.should_misclick()
        movements = self.mouse.move_to(x, y, target_width)

        for ms in movements:
            self.logger.log(make_mouse_move(
                self.logger.session_id, self._elapsed() + ms.timestamp,
                time.time(), ms.x, ms.y, ms.velocity, ms.acceleration,
                ms.phase, self.cognitive.persona.type.value,
                self.cognitive.current_phase.value,
                self.cognitive.current_affect.value,
                self._elapsed_ms(),
            ))

        click = self.mouse.click_model.generate(
            x, y, button, self._elapsed(),
            is_misclick=is_misclick,
        )

        self.logger.log(make_click(
            EventType.MOUSE_CLICK_DOWN if not is_misclick else EventType.MISCLICK,
            self.logger.session_id, self._elapsed() + click.down_timestamp,
            time.time(), click.pre_click_x, click.pre_click_y, button,
            self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed_ms(),
            hold_duration_ms=click.hold_duration_ms,
            micro_drift_px=click.micro_drift_px,
            is_misclick=is_misclick,
        ))

        self.logger.log(make_click(
            EventType.MOUSE_CLICK_UP,
            self.logger.session_id, self._elapsed() + click.up_timestamp,
            time.time(), click.post_click_x, click.post_click_y, button,
            self.cognitive.persona.type.value,
            self.cognitive.current_phase.value,
            self.cognitive.current_affect.value,
            self._elapsed_ms(),
            hold_duration_ms=click.hold_duration_ms,
            micro_drift_px=click.micro_drift_px,
            is_misclick=is_misclick,
        ))

        if is_misclick:
            self._cognitive_delay(200, 500, "notice misclick")
            correction = self.mouse.click_model.generate(x, y, button, self._elapsed())
            self.logger.log(make_click(
                EventType.MOUSE_CLICK_DOWN,
                self.logger.session_id, self._elapsed() + correction.down_timestamp,
                time.time(), x, y, button,
                self.cognitive.persona.type.value,
                self.cognitive.current_phase.value,
                self.cognitive.current_affect.value,
                self._elapsed_ms(),
            ))

        self.mouse._x = click.post_click_x
        self.mouse._y = click.post_click_y

        post_delay = self.cognitive.get_post_action_delay_ms()
        self._cognitive_delay(post_delay * 0.5, post_delay, f"post-action ({description})")
        return click

    def _scan_element(
        self, template_id: str, description: str = ""
    ) -> ScanResult:
        result = self.scanner.find_element(
            template_id,
            conditions=self.camera.to_conditions(),
            screen=None,
        )
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SCREEN_SCAN,
            data={
                "template_id": template_id,
                "found": result.found,
                "center_x": result.center_x,
                "center_y": result.center_y,
                "confidence": result.confidence,
                "method": result.method.value,
                "scan_duration_ms": result.scan_duration_ms,
                "description": description,
                "camera_yaw": self.camera.yaw,
                "camera_pitch": self.camera.pitch,
                "camera_zoom": self.camera.zoom,
            },
            labels=["synthetic", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))
        return result

    def _hover_over_element(self, template_id: str, description: str = "") -> None:
        result = self._scan_element(template_id, description)
        if result.found:
            self._hover_at_position(result.center_x, result.center_y)

    def _hover_at_position(self, x: float, y: float) -> None:
        dwell_s = self._rng.uniform(
            self.config.scan.hover_dwell_min_s,
            self.config.scan.hover_dwell_max_s,
        )
        movements = self.mouse.move_to(x, y)
        for ms in movements:
            self.logger.log(make_mouse_move(
                self.logger.session_id, self._elapsed() + ms.timestamp,
                time.time(), ms.x, ms.y, ms.velocity, ms.acceleration,
                ms.phase, self.cognitive.persona.type.value,
                self.cognitive.current_phase.value,
                self.cognitive.current_affect.value,
                self._elapsed_ms(),
            ))
        dwell_points = self.scanner.simulate_hover_dwell(x, y, dwell_s)
        for dp in dwell_points:
            self.logger.log(TelemetryEvent(
                session_id=self.logger.session_id,
                timestamp=self._elapsed() + dp["timestamp"],
                wall_time=time.time(),
                event_type=EventType.GAZE_POINT,
                data={"x": dp["x"], "y": dp["y"], "duration_s": dwell_s},
                labels=["synthetic", "hover_dwell", f"state:{self.current_state.value}"],
                persona=self.cognitive.persona.type.value,
                session_phase=self.cognitive.current_phase.value,
                affective_state=self.cognitive.current_affect.value,
                elapsed_ms=self._elapsed_ms(),
            ))
        self.mouse._x = x
        self.mouse._y = y

    def _rotate_camera(self) -> None:
        old_conditions = self.camera.to_conditions()
        new_conditions = self.camera.rotate(self._rng)
        drag_start_x = self.mouse._x
        drag_start_y = self.mouse._y
        drag_end_x = drag_start_x + self._rng.integers(-200, 200)
        drag_end_y = drag_start_y + self._rng.integers(-100, 100)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.CAMERA_ROTATE,
            data={
                "old_conditions": old_conditions,
                "new_conditions": new_conditions,
                "drag_from": [drag_start_x, drag_start_y],
                "drag_to": [drag_end_x, drag_end_y],
            },
            labels=["synthetic", "camera_rotate", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _simulate_afk(self) -> None:
        afk_duration_s = self._rng.uniform(5.0, 30.0)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.AFK,
            data={"duration_s": afk_duration_s, "type": "tab_out"},
            labels=["synthetic", "afk", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))
        self._simulate_time_passage(afk_duration_s, "AFK / tabbed out")
        self._scan_element(TEMPLATE_IDS["chaos_altar_close"], "re-orientation after AFK")
        self._scan_element("minimap_altar_icon", "re-orientation: check minimap")

    def _simulate_micro_break(self, duration_s: float) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.INTERRUPTION_START,
            data={"type": "micro_break", "duration_s": duration_s},
            labels=["synthetic", "micro_break", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))
        self._simulate_time_passage(duration_s, "micro-break")
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.INTERRUPTION_END,
            data={"type": "micro_break"},
            labels=["synthetic", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _handle_interruption(self, inter: Interruption) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.INTERRUPTION_START,
            data={
                "type": inter.interruption_type,
                "duration_s": inter.duration_s,
                "key_count": inter.key_count,
            },
            labels=["synthetic", "interruption", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))
        if inter.interruption_type == "keystroke_burst" and inter.key_count > 0:
            fake_text = "".join(self._rng.choice(
                list("abcdefghijklmnopqrstuvwxyz "), size=inter.key_count
            ))
            keystrokes = self.keyboard.type_text(fake_text, typo_enabled=True)
            for ks in keystrokes:
                self.logger.log(make_key_event(
                    EventType.KEY_DOWN if ks.action == "down" else EventType.KEY_UP,
                    self.logger.session_id, self._elapsed() + ks.timestamp,
                    time.time(), ks.key, ks.is_correction,
                    self.cognitive.persona.type.value,
                    self.cognitive.current_phase.value,
                    self.cognitive.current_affect.value,
                    self._elapsed_ms(),
                ))
        self._simulate_time_passage(inter.duration_s, f"interruption: {inter.interruption_type}")
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.INTERRUPTION_END,
            data={"type": inter.interruption_type},
            labels=["synthetic", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))
        self._scan_element(TEMPLATE_IDS["chaos_altar_close"], "re-orientation after interruption")

    def _press_escape(self) -> None:
        key_events = self.keyboard.press_key("escape")
        for ke in key_events:
            self.logger.log(make_key_event(
                EventType.KEY_DOWN if ke.action == "down" else EventType.KEY_UP,
                self.logger.session_id, self._elapsed() + ke.timestamp,
                time.time(), ke.key, False,
                self.cognitive.persona.type.value,
                self.cognitive.current_phase.value,
                self.cognitive.current_affect.value,
                self._elapsed_ms(),
            ))

    # ==================================================================
    # Timing & meta helpers
    # ==================================================================

    def _cognitive_delay(
        self, min_ms: float, max_ms: float, description: str = ""
    ) -> None:
        delay_ms = self._rng.uniform(min_ms, max_ms)
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.COGNITIVE_DELAY,
            data={"delay_ms": delay_ms, "description": description},
            labels=["synthetic", f"state:{self.current_state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _simulate_time_passage(self, duration_s: float, description: str = "") -> None:
        self.cognitive.update(dt_s=duration_s)

    def _transition_to(self, state: ScenarioState) -> None:
        old_state = self.current_state
        self.current_state = state
        self._state_entry_times[state] = self._elapsed()
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SCENARIO_STATE_CHANGE,
            data={
                "from_state": old_state.value if old_state else "none",
                "to_state": state.value,
                "cycle": self.cycle_count,
                "camera_yaw": self.camera.yaw,
            },
            labels=["synthetic", f"state:{state.value}"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _log_meta(self, event: str, data: dict) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SESSION_META,
            data={"event": event, **data},
            labels=["synthetic", "meta"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _log_state(self, event: str, data: dict | None = None) -> None:
        self.logger.log(TelemetryEvent(
            session_id=self.logger.session_id,
            timestamp=self._elapsed(),
            wall_time=time.time(),
            event_type=EventType.SESSION_META,
            data={"state": self.current_state.value, "event": event, **(data or {})},
            labels=["synthetic", f"state:{self.current_state.value}", "state_event"],
            persona=self.cognitive.persona.type.value,
            session_phase=self.cognitive.current_phase.value,
            affective_state=self.cognitive.current_affect.value,
            elapsed_ms=self._elapsed_ms(),
        ))

    def _is_short_checkin(self) -> bool:
        return (
            self.cycle_count >= 1
            and self._elapsed() < 60.0
            and self._rng.random() < 0.05
        )

    def _elapsed(self) -> float:
        return time.time() - self._session_start_s

    def _elapsed_ms(self) -> float:
        return self._elapsed() * 1000.0

    def _session_summary(self) -> dict:
        return {
            "scenario": "chaos_altar_dragon_bones",
            "cycles_completed": self.cycle_count + 1,
            "total_bones_used": self.total_bones_used,
            "bones_saved_total": self.bones_saved_total,
            "persona": self.cognitive.persona.type.value,
            "final_phase": self.cognitive.current_phase.value,
            "final_affect": self.cognitive.current_affect.value,
            "camera_rotations": self.camera.yaw,
            "synthetic_mode": self.synthetic,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_chaos_altar_session(
    config: AppConfig,
    persona_type: PersonaType = PersonaType.FAST_ACCURATE,
    session_id: str = "",
    cycles: int = DEFAULT_CYCLES,
    output_dir: str = "output",
    synthetic: bool = True,
    seed: int | None = None,
) -> dict:
    import uuid
    rng = np.random.default_rng(seed)

    if not session_id:
        session_id = f"chaos_altar_{uuid.uuid4().hex[:8]}"

    from core.mouse import MouseEmulator
    from core.cognitive import CognitiveEngine
    from core.scan import VisualScanner
    from core.keyboard import KeyboardEmulator
    from core.logger import TelemetryLogger

    mouse = MouseEmulator(
        config.mouse, rng,
        persona_amplitude_scale={
            PersonaType.FAST_ACCURATE: 0.8,
            PersonaType.SLOW_METHODICAL: 1.2,
            PersonaType.DISTRACTED_ERROR_PRONE: 1.5,
        }.get(persona_type, 1.0),
        persona_misclick_rate={
            PersonaType.FAST_ACCURATE: 0.015,
            PersonaType.SLOW_METHODICAL: 0.01,
            PersonaType.DISTRACTED_ERROR_PRONE: 0.05,
        }.get(persona_type, 0.025),
    )

    cognitive = CognitiveEngine(
        persona_type=persona_type,
        config=config.cognitive,
        rng=rng,
        hour_of_day=rng.uniform(8.0, 23.0),
    )

    scanner = VisualScanner(config.scan, rng)

    keyboard = KeyboardEmulator(
        config.keyboard, rng,
        persona_typo_rate=cognitive.persona.typo_rate,
    )

    logger = TelemetryLogger(Path(output_dir), session_id)

    scenario = ChaosAltarDragonBonesScenario(
        mouse=mouse,
        scanner=scanner,
        cognitive=cognitive,
        keyboard=keyboard,
        logger=logger,
        rng=rng,
        config=config,
        cycles=cycles,
        synthetic=synthetic,
    )

    return scenario.run_session()
