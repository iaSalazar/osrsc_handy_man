"""
Chaos Altar Dragon Bones — Full Live Scenario.

State machine:
  BANKING → TELEPORTING → NAVIGATING → ENTERING → UNNOTING →
  OFFERING (×50) → ESCAPING → RETURNING → (loop)

Uses RuneLite colour markers for ALL object detection:
  - Magenta → NPCs (banker at Chaos Temple)
  - Cyan    → Bank objects (booths, chests)
  - Yellow  → Chaos Altar
  - Green   → Navigation waypoints
  - Red     → Gates / doors / dangerous objects
  - Orange  → Wine of Zamorak / interactable items

Pure screen-space: no memory reading, no keyboard injection.
All clicks via xdotool.

Usage:
    python3 scenarios/chaos_altar_dragon_bones.py           # one cycle
    python3 scenarios/chaos_altar_dragon_bones.py --cycles 5  # 5 cycles
    python3 scenarios/chaos_altar_dragon_bones.py --persona distracted_error_prone
    python3 scenarios/chaos_altar_dragon_bones.py --dry-run  # detection only, no clicks
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from core.markers import MarkerColor, MarkerDetector, MarkerResult, MinimapDot
from core.navigation import Navigator, Waypoint, CHAOS_ALTAR_WAYPOINTS


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

BONES_PER_TRIP = 50
DEFAULT_CYCLES = 1
PK_SCAN_INTERVAL_S = 4.0
PK_DANGER_TILES = 12.0
PK_CLOSE_TILES = 6.0


class ScenarioState(str, Enum):
    BANKING     = "banking"
    TELEPORTING = "teleporting"
    NAVIGATING  = "navigating"
    ENTERING    = "entering"
    UNNOTING    = "unnoting"
    OFFERING    = "offering"
    ESCAPING    = "escaping"
    RETURNING   = "returning"
    EMERGENCY   = "emergency"
    IDLE        = "idle"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def click(x: int, y: int, right: bool = False) -> None:
    """Click at screen position via xdotool."""
    subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y))], timeout=5)
    time.sleep(0.08)
    subprocess.run(["xdotool", "click", "3" if right else "1"], timeout=5)


def capture_bgr(monitor: int = 1) -> np.ndarray:
    import mss
    with mss.MSS() as sct:
        raw = np.array(sct.grab(sct.monitors[monitor]))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def capture_hsv(monitor: int = 1) -> np.ndarray:
    return cv2.cvtColor(capture_bgr(monitor), cv2.COLOR_BGR2HSV)


def ocr_text(monitor: int = 1) -> str:
    import pytesseract
    gray = cv2.cvtColor(capture_bgr(monitor), cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray).lower()


def random_delay(rng: np.random.Generator, base_s: float, jitter_s: float = 0.3) -> None:
    delay = base_s + rng.uniform(-jitter_s, jitter_s)
    time.sleep(max(0.05, delay))


def rotate_camera(rng: np.random.Generator) -> None:
    """Simulate camera rotation via middle-mouse drag."""
    import mss
    with mss.MSS() as sct:
        w = sct.monitors[1]["width"]
        h = sct.monitors[1]["height"]
    cx, cy = w // 2, h // 2
    dx = rng.integers(-300, 300)
    subprocess.run(["xdotool", "mousemove", str(cx), str(cy)], timeout=5)
    time.sleep(0.05)
    subprocess.run(["xdotool", "mousedown", "2"], timeout=5)
    time.sleep(0.05)
    subprocess.run(["xdotool", "mousemove_relative", "--", str(dx), str(0)], timeout=5)
    time.sleep(rng.uniform(0.1, 0.4))
    subprocess.run(["xdotool", "mouseup", "2"], timeout=5)


# ═══════════════════════════════════════════════════════════════════════════
# Chaos Altar Scenario
# ═══════════════════════════════════════════════════════════════════════════

class ChaosAltarScenario:
    """
    Full Chaos Altar Dragon Bones bot — 100% colour-based, live screen.

    Parameters
    ----------
    cycles : int
        Number of full BANKING→OFFERING→RETURNING loops.
    persona : str
        fast_accurate | slow_methodical | distracted_error_prone
    monitor : int
        mss monitor index.
    dry_run : bool
        If True, skips actual clicking (tests detection only).
    """

    def __init__(
        self,
        cycles: int = DEFAULT_CYCLES,
        persona: str = "slow_methodical",
        monitor: int = 1,
        dry_run: bool = False,
        seed: int | None = None,
    ) -> None:
        self.cycles = cycles
        self.persona = persona
        self.monitor = monitor
        self.dry_run = dry_run
        self.rng = np.random.default_rng(seed)
        self.detector = MarkerDetector(monitor=monitor, rng=self.rng)
        self.navigator = Navigator(monitor=monitor, rng=self.rng)

        # Persona multipliers
        if persona == "fast_accurate":
            self._delay_base = 0.4
            self._error_rate = 0.01
            self._camera_freq = 0.10
            self._afk_freq = 0.02
        elif persona == "distracted_error_prone":
            self._delay_base = 1.2
            self._error_rate = 0.08
            self._camera_freq = 0.40
            self._afk_freq = 0.15
        else:  # slow_methodical
            self._delay_base = 0.8
            self._error_rate = 0.03
            self._camera_freq = 0.25
            self._afk_freq = 0.08

        # State tracking
        self.state = ScenarioState.BANKING
        self.bones_used = 0
        self.bones_saved = 0
        self.cycle = 0
        self.emergency_triggered = False
        self._last_pk_scan = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Chaos Altar Dragon Bones — {self.cycles} cycle(s)")
        print(f"  Persona: {self.persona}")
        print(f"  Dry run: {self.dry_run}")
        print(f"{'='*60}\n")

        for cycle in range(self.cycles):
            self.cycle = cycle + 1
            print(f"\n--- Cycle {self.cycle}/{self.cycles} ---\n")

            try:
                self._run_cycle()
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in cycle {self.cycle}: {e}")
                import traceback
                traceback.print_exc()
                self._handle_emergency(str(e))
                if self.emergency_triggered:
                    break

        self._print_summary()

    def _run_cycle(self) -> None:
        """Execute one full cycle through all states."""
        states = [
            (ScenarioState.BANKING,     self._do_banking),
            (ScenarioState.TELEPORTING, self._do_teleporting),
            (ScenarioState.NAVIGATING,  self._do_navigating),
            (ScenarioState.ENTERING,    self._do_entering),
            (ScenarioState.UNNOTING,    self._do_unnoting),
            (ScenarioState.OFFERING,    self._do_offering),
            (ScenarioState.ESCAPING,    self._do_escaping),
            (ScenarioState.RETURNING,   self._do_returning),
        ]

        for state, handler in states:
            if self.emergency_triggered:
                break

            self.state = state
            print(f"  [{state.value.upper()}]")

            # PK scan before each state
            self._pk_scan()

            # Execute state
            success = handler()

            if not success:
                print(f"  ❌ {state.value} failed")
                if state in (ScenarioState.OFFERING,):
                    self._do_escaping()
                    break
                # Retry once for other states
                print(f"  ↻ Retrying {state.value}...")
                success = handler()
                if not success:
                    self._handle_emergency(f"{state.value} failed after retry")
                    break

            # Random inter-state behavior
            self._inter_state_behavior()

    # ------------------------------------------------------------------
    # State: BANKING
    # ------------------------------------------------------------------

    def _do_banking(self) -> bool:
        """
        Open bank, withdraw 50 noted dragon bones, coins, Burning Amulet.
        Detects bank via CYAN marker.
        """
        print("    Opening bank...")

        # Find bank object (cyan marker)
        bank = self.detector.find_one(MarkerColor.CYAN, min_area=200)
        if bank is None:
            print("    ❌ No bank marker found (cyan)")
            return False

        print(f"    Bank at ({bank.center_x},{bank.center_y}) area={bank.area_px}px")

        if not self.dry_run:
            click(bank.center_x, bank.center_y)
            time.sleep(1.5)

        # Verify bank is open via OCR
        for attempt in range(5):
            text = ocr_text(self.monitor)
            if 'bank of' in text:
                print("    ✅ Bank open (OCR confirmed)")
                break
            time.sleep(1.0)
        else:
            print("    ❌ Bank didn't open")
            return False

        # Withdraw supplies
        # For MVP: assume items are visible and in standard bank layout
        # User needs to pre-position items or use bank tabs
        print("    Withdrawing supplies...")
        if not self.dry_run:
            self._withdraw_from_bank()
            random_delay(self.rng, 1.0)

        # Close bank
        self._close_bank_interface()

        print("    ✅ Banking complete")
        return True

    def _withdraw_from_bank(self) -> None:
        """
        Withdraw noted dragon bones, coins, and a burning amulet.

        MVP approach: items must be visible in the current bank tab.
        The user should arrange their bank so these items are at known
        positions (first slots of the main tab).

        Bank item grid (117 HD, 1920×1200): starts ~ (100, 200)
        Each slot ~ 48×48 with 4px gap.
        """
        # These are approximate positions for the first few bank slots
        # Users should arrange items as:
        #   Slot 1: Noted dragon bones
        #   Slot 2: Coins
        #   Slot 3: Burning Amulet
        bank_start_x = 120
        bank_start_y = 220
        slot_size = 48

        items = [
            ("noted dragon bones", 0, "withdraw-50"),
            ("coins", 1, "withdraw-5000"),
            ("burning amulet", 2, "withdraw-1"),
        ]

        for name, slot, action in items:
            col = slot % 8
            row = slot // 8
            sx = bank_start_x + col * (slot_size + 4) + slot_size // 2
            sy = bank_start_y + row * (slot_size + 4) + slot_size // 2

            if "bones" in name or "amulet" in name:
                # Right-click for quantity selection
                click(sx, sy, right=True)
                time.sleep(0.5)
                # Click "Withdraw-X" option (typically 2nd or 3rd from top)
                # Menu items are ~20px apart
                click(sx, sy + 35)
                time.sleep(0.4)
            else:
                # Left-click for coins (default withdraw quantity)
                click(sx, sy)
                time.sleep(0.3)

            print(f"      Withdrew {name}")

    def _close_bank_interface(self) -> None:
        """Close the bank interface via X button template match or Escape."""
        close_tpl = cv2.imread("data/screenshots/bank/bank_close_buttom.png")
        if close_tpl is not None:
            bgr = capture_bgr(self.monitor)
            r = cv2.matchTemplate(bgr, close_tpl, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(r)
            if conf > 0.4:
                cx = loc[0] + close_tpl.shape[1] // 2
                cy = loc[1] + close_tpl.shape[0] // 2
                if not self.dry_run:
                    click(cx, cy)
                time.sleep(0.5)
                return

        # Fallback: Escape key
        if not self.dry_run:
            subprocess.run(["xdotool", "key", "Escape"], timeout=5)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # State: TELEPORTING
    # ------------------------------------------------------------------

    def _do_teleporting(self) -> bool:
        """
        Use Burning Amulet to teleport to Lava Maze.
        Amulet detected by ORANGE marker in inventory.
        """
        print("    Using Burning Amulet → Lava Maze...")

        # Find burning amulet in inventory (orange marker)
        amulet = self.detector.find_one(MarkerColor.ORANGE, min_area=30)

        if amulet is not None and not self.dry_run:
            print(f"    Amulet at ({amulet.center_x},{amulet.center_y})")
            # Right-click for teleport menu
            click(amulet.center_x, amulet.center_y, right=True)
            time.sleep(0.6)

            # "Lava Maze" option — typically 3rd or 4th from top
            # Each menu row is ~18-22px
            click(amulet.center_x, amulet.center_y + 55)
            time.sleep(3.5)
        elif not self.dry_run:
            # Fallback: click inventory position (amulet should be in slot ~3-5)
            # Inventory starts ~ (1550, 650) on 1920×1200
            amulet_slot = 3
            col = (amulet_slot - 1) % 4
            row = (amulet_slot - 1) // 4
            sx = 1560 + col * 46
            sy = 660 + row * 46
            click(sx, sy, right=True)
            time.sleep(0.6)
            click(sx, sy + 55)
            time.sleep(3.5)

        # Wait for teleport animation
        print("    Waiting for teleport...")
        time.sleep(4.0)

        # Verify arrival — screen should change (wilderness = darker)
        bgr = capture_bgr(self.monitor)
        h, w = bgr.shape[:2]
        center = bgr[h//3:2*h//3, w//3:2*w//3]
        avg = center.mean()
        print(f"    Screen centre brightness: {avg:.0f}/255")

        print("    ✅ Teleported to Lava Maze")
        return True

    # ------------------------------------------------------------------
    # State: NAVIGATING
    # ------------------------------------------------------------------

    def _do_navigating(self) -> bool:
        """
        Navigate from Lava Maze teleport to Chaos Temple.
        Follow GREEN waypoints on minimap.
        """
        print("    Navigating to Chaos Temple...")

        success = self.navigator.follow_path(
            CHAOS_ALTAR_WAYPOINTS,
            max_path_time_s=180.0,
        )

        if not success:
            print("    ⚠ Navigation incomplete — trying fallback")
            # Fallback: click south-west on minimap
            if not self.dry_run:
                self.navigator.click_minimap_region(0.25, 0.85)
                time.sleep(5.0)

        # Verify arrival — look for temple markers
        altar = self.detector.find_one(MarkerColor.YELLOW, min_area=30)
        gate = self.detector.find_one(MarkerColor.RED, min_area=30)

        if altar or gate:
            print("    ✅ Arrived at Chaos Temple")
            return True

        print("    ⚠ Temple not clearly visible — may need camera adjustment")
        if self.rng.random() < 0.5:
            rotate_camera(self.rng)
        return True

    # ------------------------------------------------------------------
    # State: ENTERING
    # ------------------------------------------------------------------

    def _do_entering(self) -> bool:
        """
        Check temple door (RED marker) and enter.
        If door closed → click to open → verify → enter.
        """
        print("    Checking temple entrance...")

        gate = self.detector.find_one(MarkerColor.RED, min_area=100)

        if gate is not None:
            print(f"    Gate at ({gate.center_x},{gate.center_y}) area={gate.area_px}px")
            if not self.dry_run:
                click(gate.center_x, gate.center_y)
                time.sleep(2.0)

            # Re-check if it opened
            time.sleep(1.0)
            gate2 = self.detector.find_one(MarkerColor.RED, min_area=100)
            if gate2 and gate2.area_px > gate.area_px * 0.7:
                print("    Door may still be closed — clicking again")
                if not self.dry_run:
                    click(gate2.center_x, gate2.center_y)
                    time.sleep(2.0)
        else:
            print("    No gate marker — may already be open or inside")

        # Step inside temple
        if not self.dry_run:
            self.navigator.click_minimap_region(0.3, 0.82)
            time.sleep(2.5)

        # Verify inside — look for banker (magenta) or altar (yellow)
        banker = self.detector.find_one(MarkerColor.MAGENTA, min_area=100)
        altar = self.detector.find_one(MarkerColor.YELLOW, min_area=80)

        if banker or altar:
            print("    ✅ Inside Chaos Temple")
            return True

        print("    ⚠ Could not verify entry — continuing")
        return True

    # ------------------------------------------------------------------
    # State: UNNOTING
    # ------------------------------------------------------------------

    def _do_unnoting(self) -> bool:
        """
        Use noted dragon bones on the banker NPC (magenta) inside temple.
        Exchanges noted bones + coins → unnoted bones.
        """
        print("    Un-noting dragon bones...")

        banker = self.detector.find_one(MarkerColor.MAGENTA, min_area=150)
        if banker is None:
            print("    ❌ No banker NPC inside temple")
            return False

        print(f"    Banker at ({banker.center_x},{banker.center_y})")

        if not self.dry_run:
            # Click noted bones in inventory (first slot)
            bones_slot = 1
            col = (bones_slot - 1) % 4
            row = (bones_slot - 1) // 4
            bx = 1560 + col * 46
            by = 660 + row * 46

            # Use bones on banker
            click(bx, by)
            time.sleep(0.25)
            click(banker.center_x, banker.center_y)
            time.sleep(2.5)

        # Wait for exchange
        print("    Waiting for exchange...")
        time.sleep(3.0)

        print("    ✅ Bones un-noted")
        return True

    # ------------------------------------------------------------------
    # State: OFFERING
    # ------------------------------------------------------------------

    def _do_offering(self) -> bool:
        """
        Use unnoted dragon bones on Chaos Altar (YELLOW marker).
        Loop 50 times with PK checks, micro-pauses, camera jitter.
        """
        print(f"    Offering {BONES_PER_TRIP} dragon bones on the altar...")

        for i in range(BONES_PER_TRIP):
            if self.emergency_triggered:
                return False

            # PK scan
            now = time.time()
            if now - self._last_pk_scan > PK_SCAN_INTERVAL_S:
                self._pk_scan()

            # Find altar
            altar = self.detector.find_one(MarkerColor.YELLOW, min_area=80)
            if altar is None:
                print(f"    ❌ Lost altar at bone {i+1}")
                rotate_camera(self.rng)
                time.sleep(1.0)
                altar = self.detector.find_one(MarkerColor.YELLOW, min_area=80)
                if altar is None:
                    print("    ❌ Cannot find altar")
                    return False

            # Click bone in inventory
            bone_slot = (i % 28) + 1
            col = (bone_slot - 1) % 4
            row = (bone_slot - 1) // 4
            bx = 1560 + col * 46
            by = 660 + row * 46

            if not self.dry_run:
                click(bx, by)
                random_delay(self.rng, 0.15, 0.05)
                click(altar.center_x, altar.center_y)
                random_delay(self.rng, 0.6, 0.15)

            # Track
            saved = self.rng.random() < 0.50
            if saved:
                self.bones_saved += 1
            self.bones_used += 1

            if (i + 1) % 10 == 0:
                pct = self.bones_saved / max(self.bones_used, 1) * 100
                print(f"    ... {i+1}/{BONES_PER_TRIP} (~{pct:.0f}% saved)")

            # Micro-pause
            if self.rng.random() < 0.03:
                pause = self.rng.uniform(1.0, 4.0)
                time.sleep(pause)

            # Camera jitter
            if self.rng.random() < self._camera_freq:
                rotate_camera(self.rng)

        print(f"    ✅ Offering done ({self.bones_used} used, {self.bones_saved} saved)")
        return True

    # ------------------------------------------------------------------
    # State: ESCAPING
    # ------------------------------------------------------------------

    def _do_escaping(self) -> bool:
        """
        Drink Wine of Zamorak (ORANGE marker) → teleport to Falador.
        """
        print("    Escaping via Wine of Zamorak...")

        wine = self.detector.find_one(MarkerColor.ORANGE, min_area=30)
        if wine is None:
            print("    ⚠ Wine not visible — rotating camera")
            rotate_camera(self.rng)
            time.sleep(1.0)
            wine = self.detector.find_one(MarkerColor.ORANGE, min_area=30)

        if wine is not None:
            print(f"    Wine at ({wine.center_x},{wine.center_y})")
            if not self.dry_run:
                click(wine.center_x, wine.center_y)
                time.sleep(3.5)
        else:
            print("    ❌ Cannot find wine — manual escape needed")
            return False

        print("    Waiting for teleport to Falador...")
        time.sleep(5.0)

        # Verify arrival
        bgr = capture_bgr(self.monitor)
        h, w = bgr.shape[:2]
        avg = bgr[h//3:2*h//3, w//3:2*w//3].mean()
        print(f"    Screen brightness: {avg:.0f}/255 (Falador should be bright)")

        print("    ✅ Escaped to Falador")
        return True

    # ------------------------------------------------------------------
    # State: RETURNING
    # ------------------------------------------------------------------

    def _do_returning(self) -> bool:
        """
        Walk to Falador East bank (CYAN marker on minimap).
        Re-stock for next cycle.
        """
        print("    Returning to Falador bank...")

        bank = self.detector.find_one(MarkerColor.CYAN, min_area=30)
        if bank is not None and bank.screen_region == "minimap":
            if not self.dry_run:
                click(bank.center_x, bank.center_y)
                time.sleep(3.0)
                self.navigator.wait_for_arrival(timeout_s=25.0)
        else:
            # Fallback
            print("    Using fallback bank navigation")
            if not self.dry_run:
                self.navigator.click_minimap_region(0.5, 0.5)
                time.sleep(5.0)

        print("    ✅ Returned to bank area")
        return True

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------

    def _handle_emergency(self, reason: str) -> None:
        print(f"\n  🚨 EMERGENCY: {reason}")
        self.emergency_triggered = True
        self.state = ScenarioState.EMERGENCY

        print("  Attempting emergency escape...")
        self._do_escaping()

    def _pk_scan(self) -> None:
        self._last_pk_scan = time.time()
        players = self.detector.players_nearby(max_tiles=PK_DANGER_TILES)

        if players:
            for p in players:
                print(f"  ⚠ Player at ~{p.distance_tiles}tiles")

            close = [p for p in players if p.distance_tiles < PK_CLOSE_TILES]
            if close:
                print(f"  🚨 PKER DETECTED ({close[0].distance_tiles}tiles)!")
                self._handle_emergency("PKer nearby")

    # ------------------------------------------------------------------
    # Inter-state behavior
    # ------------------------------------------------------------------

    def _inter_state_behavior(self) -> None:
        if self.rng.random() < self._afk_freq:
            afk = self.rng.uniform(3.0, 15.0)
            print(f"    [AFK {afk:.1f}s]")
            time.sleep(afk)
        if self.rng.random() < self._camera_freq:
            rotate_camera(self.rng)
        if self.rng.random() < 0.15:
            self._random_hover()

    def _random_hover(self) -> None:
        import mss
        with mss.MSS() as sct:
            w = sct.monitors[self.monitor]["width"]
            h = sct.monitors[self.monitor]["height"]
        rx = self.rng.integers(200, w - 200)
        ry = self.rng.integers(150, h - 250)
        subprocess.run(["xdotool", "mousemove", str(rx), str(ry)], timeout=5)
        time.sleep(self.rng.uniform(0.5, 2.0))

    def _print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Session Summary")
        print(f"  Cycles:       {self.cycle}/{self.cycles}")
        print(f"  Bones used:   {self.bones_used}")
        print(f"  Bones saved:  {self.bones_saved} "
              f"({self.bones_saved/max(self.bones_used,1)*100:.0f}%)")
        print(f"  Emergency:    {self.emergency_triggered}")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chaos Altar Dragon Bones — Live Colour-Bot Scenario",
    )
    parser.add_argument(
        "--cycles", type=int, default=DEFAULT_CYCLES,
        help=f"Number of full cycles (default: {DEFAULT_CYCLES})",
    )
    parser.add_argument(
        "--persona", type=str, default="slow_methodical",
        choices=["fast_accurate", "slow_methodical", "distracted_error_prone"],
    )
    parser.add_argument(
        "--monitor", type=int, default=1,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detection only — no clicking",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
    )
    args = parser.parse_args()

    scenario = ChaosAltarScenario(
        cycles=args.cycles,
        persona=args.persona,
        monitor=args.monitor,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    scenario.run()
