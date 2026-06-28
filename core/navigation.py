"""
Minimap Navigation — waypoint-based walking using Ground Markers.

Uses RuneLite Ground Markers (green fill) as navigation waypoints on the
minimap.  Clicks waypoints and waits for arrival before proceeding.

Pure screen-space: detects coloured waypoints on the minimap, clicks them,
monitors position change to detect arrival.

Usage:
    from core.navigation import Navigator

    nav = Navigator()
    nav.walk_to_waypoint("lava_maze_entrance")
    nav.follow_path(["wp_1", "wp_2", "wp_3", "chaos_temple"])
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.markers import MarkerColor, MarkerDetector, MarkerResult


# ═══════════════════════════════════════════════════════════════════════════
# Waypoint definition
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    """A named navigation waypoint."""
    name: str
    color: MarkerColor = MarkerColor.GREEN
    # Approximate minimap position relative to game view (0.0–1.0)
    # These are fallbacks when marker detection fails
    minimap_ratio_x: float = 0.5
    minimap_ratio_y: float = 0.5
    wait_after_arrival_s: float = 1.5


# ═══════════════════════════════════════════════════════════════════════════
# Navigator
# ═══════════════════════════════════════════════════════════════════════════

class Navigator:
    """
    Minimap-based navigation using RuneLite markers as waypoints.

    The navigator:
      1. Detects coloured markers on the minimap
      2. Clicks them to initiate walking
      3. Monitors for arrival (marker position converges or flag disappears)
      4. Proceeds to the next waypoint

    All timing is human-like with randomization.
    """

    def __init__(
        self,
        monitor: int = 1,
        rng: np.random.Generator | None = None,
        minimap_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._monitor = monitor
        self._rng = rng or np.random.default_rng()
        self._detector = MarkerDetector(monitor=monitor, rng=self._rng)
        self._minimap_bounds = minimap_bounds

        # State tracking
        self._last_click_pos: Optional[tuple[int, int]] = None
        self._last_click_time: float = 0.0
        self._arrival_threshold_px: int = 8  # how close flag must be to centre to count as "arrived"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def walk_to_marker(
        self,
        color: MarkerColor,
        min_area: int = 50,
        timeout_s: float = 30.0,
        check_interval_s: float = 1.5,
    ) -> bool:
        """
        Find a marker of the given colour on the minimap, click it,
        and wait until arrival.

        Returns True if arrived, False if timed out.
        """
        marker = self._detector.find_one(
            color, min_area=min_area, min_confidence=0.1,
        )

        if marker is None:
            print(f"  [nav] No {color.value} marker found on minimap")
            return False

        # Click the marker on the minimap
        cx, cy = marker.center_x, marker.center_y
        self._click(cx, cy)
        self._last_click_pos = (cx, cy)
        self._last_click_time = time.time()

        # Randomize click offset slightly
        jx = int(cx + self._rng.integers(-3, 4))
        jy = int(cy + self._rng.integers(-3, 4))
        self._click(jx, jy)

        print(f"  [nav] Walking to {color.value} marker at ({cx},{cy})")

        # Wait for arrival
        elapsed = 0.0
        while elapsed < timeout_s:
            time.sleep(check_interval_s)
            elapsed += check_interval_s

            if self._has_arrived():
                print(f"  [nav] Arrived at {color.value} marker ({elapsed:.1f}s)")
                time.sleep(self._rng.uniform(0.8, 2.0))  # post-arrival settle
                return True

            # Check if we got stuck — flag hasn't moved much
            if elapsed > 8.0 and not self._is_moving():
                print(f"  [nav] Re-clicking (stuck)")
                self._click(cx, cy)

        print(f"  [nav] Timeout walking to {color.value} marker")
        return False

    def walk_to_waypoint(
        self,
        waypoint: Waypoint,
        timeout_s: float = 30.0,
    ) -> bool:
        """Walk to a named waypoint."""
        print(f"  [nav] → {waypoint.name}")
        return self.walk_to_marker(
            color=waypoint.color,
            timeout_s=timeout_s,
        )

    def follow_path(
        self,
        waypoints: list[Waypoint],
        max_path_time_s: float = 300.0,
    ) -> bool:
        """
        Follow a sequence of waypoints in order.

        Returns True if all waypoints reached, False if any timed out.
        """
        path_start = time.time()

        for i, wp in enumerate(waypoints):
            remaining = max_path_time_s - (time.time() - path_start)
            if remaining < 10:
                print(f"  [nav] Path time budget exceeded at waypoint {i+1}/{len(waypoints)}")
                return False

            timeout = min(remaining, 45.0)
            if not self.walk_to_waypoint(wp, timeout_s=timeout):
                return False

            # Inter-waypoint pause (human-like)
            pause = self._rng.uniform(0.5, 2.0)
            time.sleep(pause)

        print(f"  [nav] ✅ Path complete ({len(waypoints)} waypoints)")
        return True

    def click_minimap_region(
        self,
        ratio_x: float,
        ratio_y: float,
    ) -> None:
        """
        Click a position on the minimap by ratio (0.0–1.0 from top-left).
        Useful as a fallback when markers aren't detected.
        """
        bounds = self._get_minimap_bounds()
        mx, my, mw, mh = bounds
        cx = mx + int(mw * ratio_x)
        cy = my + int(mh * ratio_y)
        self._click(cx, cy)
        self._last_click_pos = (cx, cy)
        self._last_click_time = time.time()

    def wait_for_arrival(
        self,
        timeout_s: float = 20.0,
        check_interval_s: float = 1.2,
    ) -> bool:
        """Wait until the player reaches their clicked destination."""
        elapsed = 0.0
        while elapsed < timeout_s:
            time.sleep(check_interval_s)
            elapsed += check_interval_s
            if self._has_arrived():
                return True
        return False

    def is_player_moving(self) -> bool:
        """Check if the player is currently moving (flag visible on minimap)."""
        return self._is_moving()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _click(self, x: int, y: int) -> None:
        subprocess.run(
            ["xdotool", "mousemove", str(x), str(y)],
            timeout=5,
        )
        time.sleep(0.08)
        subprocess.run(["xdotool", "click", "1"], timeout=5)

    def _get_minimap_bounds(self) -> tuple[int, int, int, int]:
        if self._minimap_bounds is not None:
            return self._minimap_bounds

        import mss
        with mss.MSS() as sct:
            w = sct.monitors[self._monitor]["width"]
            h = sct.monitors[self._monitor]["height"]

        # 117 HD stretched on 1920×1200: minimap is top-right
        mm_w = int(w * 0.125)   # ~240px
        mm_h = int(h * 0.20)    # ~240px
        mm_x = w - mm_w - 30
        mm_y = 30
        return (mm_x, mm_y, mm_w, mm_h)

    def _has_arrived(self) -> bool:
        """
        Check if the player has arrived at the clicked destination.

        Strategy: capture the minimap, check if the destination flag
        (small X or flashing marker) has disappeared, indicating arrival.
        Also checks if enough time has passed since the last click.
        """
        elapsed = time.time() - self._last_click_time

        # Minimum walk time — even 1 tile takes ~0.6s
        if elapsed < 1.5:
            return False

        # After 15s, assume we arrived (prevents infinite loops)
        if elapsed > 15.0:
            return True

        # Check if the click position flag is still visible
        # The flag is a small red X that appears where you clicked on the minimap
        if self._last_click_pos is not None:
            screen = self._capture_bgr()
            mm_bounds = self._get_minimap_bounds()
            mx, my, mw, mh = mm_bounds

            # Flag position relative to minimap
            flag_rx = self._last_click_pos[0] - mx
            flag_ry = self._last_click_pos[1] - my

            # Only check if flag was on the minimap
            if 0 <= flag_rx < mw and 0 <= flag_ry < mh:
                # Look for red X flag — small red region near click point
                flag_roi = screen[
                    max(my, self._last_click_pos[1] - 12):min(my + mh, self._last_click_pos[1] + 12),
                    max(mx, self._last_click_pos[0] - 12):min(mx + mw, self._last_click_pos[0] + 12),
                ]
                if flag_roi.size > 0:
                    hsv = cv2.cvtColor(flag_roi, cv2.COLOR_BGR2HSV)
                    red = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
                    red |= cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
                    red_px = np.count_nonzero(red)
                    # If no red flag visible, we arrived
                    if red_px < 5:
                        return True
                    # Flag still visible — still walking
                    return False

        # Fallback: after reasonable time, check if position is stable
        if elapsed > 6.0:
            return True

        return False

    def _is_moving(self) -> bool:
        """Check if the red destination flag is visible on the minimap."""
        if self._last_click_pos is None:
            return False

        screen = self._capture_bgr()
        mm_bounds = self._get_minimap_bounds()
        mx, my, mw, mh = mm_bounds

        # Check minimap area for any red flag
        mm_roi = screen[my:my+mh, mx:mx+mw]
        hsv = cv2.cvtColor(mm_roi, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))

        # Count red pixels — a flag is ~15-40 red pixels
        red_count = np.count_nonzero(red)
        return red_count > 10

    def _capture_bgr(self) -> np.ndarray:
        import mss
        with mss.MSS() as sct:
            raw = np.array(sct.grab(sct.monitors[self._monitor]))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


# ═══════════════════════════════════════════════════════════════════════════
# Waypoint library — common paths
# ═══════════════════════════════════════════════════════════════════════════

# Chaos Altar waypoints (from Lava Maze teleport to Chaos Temple)
# These are named positions that correspond to Ground Markers on the minimap.
# The actual positions are detected at runtime via colour markers.
# The ratios are fallback approximations.

CHAOS_ALTAR_WAYPOINTS = [
    Waypoint(name="lava_maze_arrival",      color=MarkerColor.GREEN,
             minimap_ratio_x=0.5, minimap_ratio_y=0.6),
    Waypoint(name="south_west_path_1",      color=MarkerColor.GREEN,
             minimap_ratio_x=0.4, minimap_ratio_y=0.7),
    Waypoint(name="south_west_path_2",      color=MarkerColor.GREEN,
             minimap_ratio_x=0.3, minimap_ratio_y=0.8),
    Waypoint(name="chaos_temple_entrance",  color=MarkerColor.RED,
             minimap_ratio_x=0.2, minimap_ratio_y=0.85),
]

# Falador teleport → bank waypoints
FALADOR_BANK_WAYPOINTS = [
    Waypoint(name="falador_arrival",        color=MarkerColor.GREEN,
             minimap_ratio_x=0.6, minimap_ratio_y=0.5),
    Waypoint(name="falador_east_bank",      color=MarkerColor.CYAN,
             minimap_ratio_x=0.5, minimap_ratio_y=0.4),
]
