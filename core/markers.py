"""
RuneLite Marker Detection — find Object Markers, Ground Markers, and NPC
Indicators by their fill colour.

Pure screen-space colour detection.  No memory reading, no client hooking.

Usage:
    from core.markers import MarkerDetector, MarkerColor

    md = MarkerDetector()
    bankers = md.find(MarkerColor.MAGENTA)       # NPCs
    altars  = md.find(MarkerColor.YELLOW)        # Chaos Altar
    gates   = md.find(MarkerColor.RED)           # Wilderness gate
    waypts  = md.find(MarkerColor.GREEN)         # Ground markers
    wine    = md.find(MarkerColor.ORANGE)        # Wine of Zamorak
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Colour scheme
# ═══════════════════════════════════════════════════════════════════════════

class MarkerColor(str, Enum):
    """
    Standard marker colour scheme for RuneLite plugins.

    Configure these in RuneLite:
      - NPC Indicators  → Magenta fill (#FF00FF)
      - Object Markers  → Yellow for altar, Red for gates, Orange for items
      - Ground Markers  → Green for navigation waypoints
    """
    MAGENTA = "magenta"   # #FF00FF — NPCs / bankers
    CYAN    = "cyan"      # #00FFFF — bank objects (booths, chests)
    YELLOW  = "yellow"    # #FFFF00 — altars / interactive objects
    GREEN   = "green"     # #00FF00 — navigation waypoints
    RED     = "red"       # #FF0000 — dangerous objects / gates / doors
    ORANGE  = "orange"    # #FF8800 — teleport items / interactable objects


# HSV ranges tuned for RuneLite marker fills on 117 HD.
# Each tuple is (lower, upper) in OpenCV HSV space [0-180, 0-255, 0-255].
#
# RuneLite fills are semi-transparent, so we use broad saturation/value
# ranges to catch the colour blended with the game world behind it.

HSV_RANGES: dict[MarkerColor, tuple[np.ndarray, np.ndarray]] = {
    MarkerColor.MAGENTA: (
        np.array([140, 100, 60],  dtype=np.uint8),
        np.array([165, 255, 255], dtype=np.uint8),
    ),
    MarkerColor.CYAN: (
        np.array([80,  100, 60],  dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    ),
    MarkerColor.YELLOW: (
        np.array([22,  100, 60],  dtype=np.uint8),
        np.array([38,  255, 255], dtype=np.uint8),
    ),
    MarkerColor.GREEN: (
        np.array([40,  100, 40],  dtype=np.uint8),
        np.array([75,  255, 255], dtype=np.uint8),
    ),
    MarkerColor.RED: (
        np.array([0,   100, 60],  dtype=np.uint8),
        np.array([12,  255, 255], dtype=np.uint8),
    ),
    MarkerColor.ORANGE: (
        np.array([5,   100, 60],  dtype=np.uint8),
        np.array([20,  255, 255], dtype=np.uint8),
    ),
}

# Red needs a second range to wrap around 180°
HSV_RANGES_WRAP: dict[MarkerColor, tuple[np.ndarray, np.ndarray]] = {
    MarkerColor.RED: (
        np.array([170, 100, 60],  dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Detection result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarkerResult:
    """A detected marker on screen."""
    color: MarkerColor
    x: int
    y: int
    center_x: int
    center_y: int
    width: int
    height: int
    area_px: int
    confidence: float          # 0.0–1.0 based on colour saturation purity
    screen_region: str = ""     # "minimap", "game_view", "inventory", etc.


@dataclass
class MinimapDot:
    """A dot detected on the minimap."""
    x: int
    y: int
    color: str                 # "white" (player), "blue" (NPC), "green" (friend)
    radius: int
    distance_tiles: float = 0.0  # estimated distance from centre


# ═══════════════════════════════════════════════════════════════════════════
# Marker Detector
# ═══════════════════════════════════════════════════════════════════════════

class MarkerDetector:
    """
    Detects RuneLite markers on screen by their fill colour.

    Parameters
    ----------
    monitor : int
        mss monitor index (1 = primary).
    rng : np.random.Generator or None
        Shared RNG for jitter.
    """

    def __init__(
        self,
        monitor: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._monitor = monitor
        self._rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(
        self,
        color: MarkerColor,
        screen: np.ndarray | None = None,
        min_area: int = 100,
        max_area: int | None = None,
        game_mask: bool = True,
    ) -> list[MarkerResult]:
        """
        Find all markers of a given colour on screen.

        Parameters
        ----------
        color : MarkerColor
            Which colour to search for.
        screen : np.ndarray or None
            HSV screen image.  If None, captures live.
        min_area : int
            Minimum contour area in pixels.
        max_area : int or None
            Maximum contour area (filters full-screen noise).
        game_mask : bool
            If True, restrict search to the game viewport (excludes
            minimap border, chatbox edges, etc. to reduce false positives).

        Returns
        -------
        List of MarkerResult sorted by area descending (largest first).
        """
        if screen is None:
            screen = self._capture_hsv()

        h, w = screen.shape[:2]

        # Primary HSV range
        lower, upper = HSV_RANGES[color]
        mask = cv2.inRange(screen, lower, upper)

        # Wrap-around range (e.g., red wraps 170°–180°)
        if color in HSV_RANGES_WRAP:
            wl, wu = HSV_RANGES_WRAP[color]
            mask |= cv2.inRange(screen, wl, wu)

        # Game-view mask — restrict to the playable area
        if game_mask:
            game_mask_arr = np.zeros((h, w), dtype=np.uint8)
            # Typical OSRS 117 HD layout on 1920×1200:
            # Game view is roughly x: 50–1870, y: 50–1050
            # (leaves out minimap, chatbox, side panels)
            margin = 50
            game_mask_arr[margin:h-margin, margin:w-margin] = 255
            # But also allow minimap region (top-right ~1650-1900, 50-300)
            game_mask_arr[30:300, 1600:w-30] = 255
            # And chatbox region (bottom ~1000-1180, centre)
            game_mask_arr[h-220:h-30, 100:w-100] = 255
            mask = cv2.bitwise_and(mask, mask, mask=game_mask_arr)

        # Morphological close — bridge small gaps in the marker fill
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        screen_area = h * w
        max_a = max_area or (screen_area // 4)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_a:
                continue

            x, y, bw, bh = cv2.boundingRect(c)

            # Filter by aspect ratio — markers are roughly square/blobby
            aspect = bw / max(bh, 1)
            if aspect < 0.2 or aspect > 5.0:
                continue

            # Compute confidence from colour saturation in this region
            region_hsv = screen[y:y+bh, x:x+bw]
            region_mask = mask[y:y+bh, x:x+bw]
            fill_ratio = np.count_nonzero(region_mask) / max(region_mask.size, 1)
            confidence = float(fill_ratio) if fill_ratio < 1.0 else float(min(area / 5000, 1.0))

            # Determine screen region
            region = self._classify_region(x + bw//2, y + bh//2, w, h)

            results.append(MarkerResult(
                color=color,
                x=int(x), y=int(y),
                center_x=int(x + bw // 2),
                center_y=int(y + bh // 2),
                width=int(bw), height=int(bh),
                area_px=int(area),
                confidence=confidence,
                screen_region=region,
            ))

        # Sort by area descending — largest marker is usually the primary target
        results.sort(key=lambda r: r.area_px, reverse=True)
        return results

    def find_one(
        self,
        color: MarkerColor,
        screen: np.ndarray | None = None,
        min_area: int = 100,
        min_confidence: float = 0.15,
    ) -> Optional[MarkerResult]:
        """
        Find the best (largest) marker of a given colour.

        Returns None if nothing found above min_confidence.
        """
        results = self.find(color, screen=screen, min_area=min_area)
        for r in results:
            if r.confidence >= min_confidence:
                return r
        return None

    def find_all_colors(
        self,
        colors: list[MarkerColor],
        screen: np.ndarray | None = None,
    ) -> dict[MarkerColor, list[MarkerResult]]:
        """
        Find markers for multiple colours in a single screen capture.

        More efficient than calling find() multiple times (only captures once).
        """
        if screen is None:
            screen = self._capture_hsv()

        return {c: self.find(c, screen=screen) for c in colors}

    # ------------------------------------------------------------------
    # Minimap player detection
    # ------------------------------------------------------------------

    def find_minimap_dots(
        self,
        screen: np.ndarray | None = None,
        minimap_bounds: tuple[int, int, int, int] | None = None,
    ) -> list[MinimapDot]:
        """
        Detect player dots (white) and other entities on the minimap.

        The minimap in OSRS shows:
          - White dots: other players (PKers!)
          - Blue dots:  NPCs
          - Green dots: friends / clan members
          - Yellow dot: yourself (usually at centre)

        Parameters
        ----------
        screen : np.ndarray or None
            BGR screen image.
        minimap_bounds : (x, y, w, h) or None
            If None, tries to auto-detect minimap position (top-right corner,
            ~250×250 pixels on 1920×1200).

        Returns
        -------
        List of MinimapDot found.
        """
        if screen is None:
            screen = self._capture_bgr()

        h, w = screen.shape[:2]

        # Auto-detect minimap bounds if not provided
        if minimap_bounds is None:
            # On 1920×1200 with 117 HD stretched, minimap is roughly
            # top-right corner, ~240×240 pixels
            mm_x = w - 270
            mm_y = 30
            mm_w = 240
            mm_h = 240
            minimap_bounds = (mm_x, mm_y, mm_w, mm_h)

        mx, my, mw, mh = minimap_bounds
        mm_roi = screen[my:my+mh, mx:mx+mw]

        # Convert to grayscale for dot detection
        gray = cv2.cvtColor(mm_roi, cv2.COLOR_BGR2GRAY)

        # White dots on dark minimap background
        # Threshold for bright dots (players, NPCs)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Remove the white orb border around minimap (>90% of pixels in edge rows)
        edge_mask = np.ones_like(white_mask) * 255
        edge_mask[10:mh-10, 10:mw-10] = 0
        white_mask = cv2.bitwise_and(white_mask, white_mask, mask=cv2.bitwise_not(edge_mask))

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dots = []
        mm_center = (mx + mw // 2, my + mh // 2)
        minimap_tile_radius = mw // 2  # rough: minimap shows ~18 tiles across

        for c in contours:
            area = cv2.contourArea(c)
            if area < 3 or area > 80:  # dots are small, 3-80 px²
                continue

            # Get circle centre
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < 1.5 or radius > 8:
                continue

            abs_x = mx + int(cx)
            abs_y = my + int(cy)

            # Estimate distance in tiles from centre (yourself)
            dx_px = abs_x - mm_center[0]
            dy_px = abs_y - mm_center[1]
            px_from_centre = math.sqrt(dx_px**2 + dy_px**2)
            tiles = (px_from_centre / minimap_tile_radius) * 9  # ~9 tiles to edge

            dots.append(MinimapDot(
                x=abs_x, y=abs_y,
                color="white",
                radius=int(radius),
                distance_tiles=round(tiles, 1),
            ))

        return dots

    def players_nearby(
        self,
        max_tiles: float = 12.0,
        screen: np.ndarray | None = None,
    ) -> list[MinimapDot]:
        """
        Check for nearby players (PK threat detection).

        Returns players within max_tiles distance of centre.
        """
        dots = self.find_minimap_dots(screen=screen)
        return [d for d in dots if d.distance_tiles <= max_tiles]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_hsv(self) -> np.ndarray:
        import mss
        with mss.MSS() as sct:
            raw = np.array(sct.grab(sct.monitors[self._monitor]))
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def _capture_bgr(self) -> np.ndarray:
        import mss
        with mss.MSS() as sct:
            raw = np.array(sct.grab(sct.monitors[self._monitor]))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _classify_region(self, cx: int, cy: int, screen_w: int, screen_h: int) -> str:
        """Classify where on screen a marker is located."""
        # Minimap: top-right ~250×250
        if cx > screen_w - 300 and cy < 300:
            return "minimap"
        # Chatbox: bottom ~150px
        if cy > screen_h - 200:
            return "chatbox"
        # Inventory: right side panel
        if cx > screen_w - 300 and cy > 300 and cy < screen_h - 250:
            return "inventory"
        return "game_view"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience — hex-to-HSV helper
# ═══════════════════════════════════════════════════════════════════════════

def hex_to_hsv_range(hex_color: str, tolerance: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a hex colour (e.g. "#FF00FF") to an HSV range for cv2.inRange.

    Useful for testing new marker colours without editing HSV_RANGES.

    Parameters
    ----------
    hex_color : str
        Hex colour string like "#FF00FF" or "FF00FF".
    tolerance : int
        ± range around the detected hue/saturation.

    Returns
    -------
    (lower, upper) as uint8 numpy arrays.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # Convert RGB to HSV (OpenCV expects 0-180, 0-255, 0-255)
    rgb = np.uint8([[[b, g, r]]])  # OpenCV is BGR
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)[0, 0]

    lower = np.array([
        max(hsv[0] - tolerance, 0),
        max(hsv[1] - 40, 50),
        max(hsv[2] - 80, 40),
    ], dtype=np.uint8)

    upper = np.array([
        min(hsv[0] + tolerance, 180),
        min(hsv[1] + 40, 255),
        min(hsv[2] + 80, 255),
    ], dtype=np.uint8)

    return lower, upper
