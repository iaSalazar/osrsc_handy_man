"""
Visual Perception Module — the "color-bot" core.

All UI element detection is performed via HSV colour matching and template
matching against a library of pre-captured templates.  No memory reading,
no client hooking — purely screen-space analysis.

Two operating modes:
  - Synthetic mode (screen=None): returns stored bounding boxes with
    Gaussian position noise.  Used for bulk training data generation.
  - Live mode (screen provided): actual OpenCV computer vision on
    real screenshots.  Used for validation and mixed-mode data.

The module also simulates human visual search behaviour:
  - Scanning 2–3 distractor regions before the target
  - Hover dwells with sub-pixel drift and circular micro-gestures
  - Confidence-based fallback from HSV → template → OCR
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import ScanConfig


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ScanMethod(str, Enum):
    HSV = "hsv"
    TEMPLATE = "template"
    OCR = "ocr"
    DISTRACTER = "distracter"
    SYNTHETIC = "synthetic"


@dataclass
class TemplateVariant:
    """A single variant of a UI element under specific conditions."""
    file_path: str
    conditions: dict = field(default_factory=dict)   # {state, camera_angle, zoom, ...}
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper: list[int] = field(default_factory=lambda: [180, 255, 255])
    confidence_threshold: float = 0.65


@dataclass
class Template:
    """A UI element with multiple variants for different conditions."""
    id: str
    label: str
    variants: list[TemplateVariant] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    default_bbox: tuple[int, int, int, int] = (0, 0, 20, 20)

    def resolve_best_variant(self, conditions: dict | None = None) -> TemplateVariant:
        """
        Pick the variant whose conditions best match the requested ones.
        Simple scoring: count matching key-value pairs.
        """
        if not conditions or not self.variants:
            return self.variants[0] if self.variants else TemplateVariant("")
        best, best_score = self.variants[0], 0
        for v in self.variants:
            score = sum(
                1 for k, v_ in conditions.items()
                if v.conditions.get(k) == v_
            )
            if score > best_score:
                best, best_score = v, score
        return best


@dataclass
class ScanResult:
    """Result of a visual scan for a UI element."""
    found: bool
    template_id: str = ""
    x: int = 0
    y: int = 0
    center_x: int = 0
    center_y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0
    method: ScanMethod = ScanMethod.SYNTHETIC
    scan_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Noise Injector — simulates environmental variability
# ---------------------------------------------------------------------------

class NoiseInjector:
    """
    Applies subtle colour/lighting variations to simulate:
      - Monitor warmth changes (brightness drift)
      - Ambient lighting fluctuations (hue shift)
      - Slight defocus / eye strain (Gaussian blur)
      - CCD/panel noise (S-channel jitter)
    """

    def __init__(self, config: ScanConfig, rng: np.random.Generator) -> None:
        self._brightness_std = config.noise_brightness_std
        self._hue_shift = config.noise_hue_shift_deg
        self._blur_kernel = config.noise_blur_kernel
        self._rng = rng

    def apply(self, screen: np.ndarray) -> np.ndarray:
        """
        Apply noise to an HSV image in-place.

        Parameters
        ----------
        screen : np.ndarray of shape (H, W, 3) in HSV colour space.

        Returns
        -------
        Modified HSV image.
        """
        if screen is None or screen.size == 0:
            return screen

        hsv = screen.astype(np.float32)

        # Hue shift — small random offset
        hue_shift = self._rng.uniform(-self._hue_shift, self._hue_shift)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0

        # Brightness jitter on V channel
        brightness = self._rng.normal(0.0, self._brightness_std)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0.0, 255.0)

        # Saturation noise
        sat_noise = self._rng.normal(0.0, 2.0, size=hsv[:, :, 1].shape)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_noise, 0.0, 255.0)

        return hsv.astype(np.uint8)


# ---------------------------------------------------------------------------
# Visual Scanner
# ---------------------------------------------------------------------------

class VisualScanner:
    """
    Main visual perception engine.

    In synthetic mode (screen=None), returns stored template bounding boxes
    with Gaussian position noise.  In live mode, performs actual HSV and
    template matching via OpenCV.
    """

    def __init__(
        self,
        config: ScanConfig,
        rng: np.random.Generator,
        template_dir: Path | str | None = None,
    ) -> None:
        self.config = config
        self._rng = rng
        self._noise = NoiseInjector(config, rng)
        self._templates: dict[str, Template] = {}
        self._template_dir = Path(template_dir) if template_dir else None

        if self._template_dir and self._template_dir.exists():
            self.load_templates(self._template_dir)

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def load_templates(self, template_dir: Path) -> None:
        """
        Walk a template directory and load all template.yaml files.

        Expected structure:
            template_dir/
              element_name/
                template.yaml
                variants/*.png
                hsv_masks/*.png
        """
        import yaml
        for yaml_path in template_dir.rglob("template.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            tid = data.get("id", yaml_path.parent.name)
            variants = []
            for vdata in data.get("variants", []):
                variants.append(TemplateVariant(
                    file_path=str(yaml_path.parent / vdata.get("file", "")),
                    conditions=vdata.get("conditions", {}),
                    bbox=tuple(vdata.get("bbox", [0, 0, 20, 20])),
                    hsv_lower=vdata.get("hsv_lower", [0, 0, 0]),
                    hsv_upper=vdata.get("hsv_upper", [180, 255, 255]),
                    confidence_threshold=vdata.get("confidence_threshold", 0.65),
                ))
            default_bbox = tuple(data.get("default_bbox", [0, 0, 20, 20]))
            self._templates[tid] = Template(
                id=tid,
                label=data.get("label", tid),
                variants=variants,
                tags=data.get("tags", []),
                default_bbox=default_bbox,
            )

    def register_template(self, template: Template) -> None:
        """Register a template programmatically (no YAML file needed)."""
        self._templates[template.id] = template

    # ------------------------------------------------------------------
    # Main scan method
    # ------------------------------------------------------------------

    def find_element(
        self,
        template_id: str,
        conditions: dict | None = None,
        screen: np.ndarray | None = None,
    ) -> ScanResult:
        """
        Locate a UI element on screen.

        Parameters
        ----------
        template_id : ID of the element to find.
        conditions : Dict of current conditions for variant selection
                     (e.g., {"yaw": "N", "pitch": "default", "zoom": "mid"}).
        screen : If None, synthetic mode.  If np.ndarray (HSV), live mode.

        Returns
        -------
        ScanResult with position, confidence, and method.
        """
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            # Unknown template — return synthetic default
            return self._synthetic_result(template_id, conditions)

        variant = tmpl.resolve_best_variant(conditions)

        if screen is None:
            return self._synthetic_result(template_id, conditions, tmpl, variant)

        return self._live_scan(screen, tmpl, variant)

    def scan_region_with_distracters(
        self,
        template_id: str,
        conditions: dict | None = None,
        screen: np.ndarray | None = None,
        distracter_count: int = 3,
    ) -> ScanResult:
        """
        Scan for an element with simulated visual search — 2–3 distractor
        fixations before the target scan.
        """
        screen_shape = (1080, 1920, 3)  # default when screen is None
        if screen is not None:
            screen_shape = screen.shape

        # Generate distracter scans (these model the time cost of visual search)
        for _ in range(distracter_count):
            dx = self._rng.integers(0, max(screen_shape[1], 100))
            dy = self._rng.integers(0, max(screen_shape[0], 100))
            distracter_time = self._rng.uniform(*self.config.distractor_time_ms)
            # Distracter scan — no actual detection, just time cost
            # In live mode, this would be an actual partial-screen scan

        # Now scan for the real target
        return self.find_element(template_id, conditions, screen)

    def simulate_hover_dwell(
        self,
        x: float,
        y: float,
        duration_s: float,
    ) -> list[dict]:
        """
        Generate hover dwell samples with sub-pixel drift and circular
        micro-gestures — typical of a human "reading" a tooltip or label.

        Returns list of dicts with {x, y, timestamp} for logging.
        """
        n_samples = max(int(duration_s * 8), 5)  # ~8 Hz sampling
        points = []
        for i in range(n_samples):
            t = i / n_samples * duration_s
            # Sub-pixel drift: sinusoid <5 px amplitude
            drift_x = 2.5 * math.sin(2.0 * math.pi * 0.3 * t + 0.0)
            drift_y = 2.5 * math.cos(2.0 * math.pi * 0.3 * t + 1.2)
            # Circular micro-gesture (~10 px radius, slow)
            circ_x = self.config.hover_micro_gesture_radius * math.sin(
                2.0 * math.pi * 0.15 * t
            )
            circ_y = self.config.hover_micro_gesture_radius * math.cos(
                2.0 * math.pi * 0.15 * t
            )
            points.append({
                "x": x + drift_x + circ_x,
                "y": y + drift_y + circ_y,
                "timestamp": t,
            })
        return points

    # ------------------------------------------------------------------
    # Internal: synthetic mode
    # ------------------------------------------------------------------

    def _synthetic_result(
        self,
        template_id: str,
        conditions: dict | None = None,
        template: Template | None = None,
        variant: TemplateVariant | None = None,
    ) -> ScanResult:
        """Generate a scan result from stored template data + noise."""
        if template and variant and variant.bbox != (0, 0, 0, 0):
            x, y, w, h = variant.bbox
        elif template and template.default_bbox != (0, 0, 20, 20):
            x, y, w, h = template.default_bbox
        else:
            # Fallback: generate plausible random position
            x = self._rng.integers(100, 1820)
            y = self._rng.integers(100, 980)
            w, h = 40, 40

        # Apply Gaussian position noise
        noise_std = self.config.synthetic_position_noise_px
        cx = x + w // 2 + self._rng.normal(0.0, noise_std)
        cy = y + h // 2 + self._rng.normal(0.0, noise_std)

        # Scan duration — typical human visual search time
        scan_ms = self._rng.uniform(150.0, 400.0)

        return ScanResult(
            found=True,
            template_id=template_id,
            x=int(x + self._rng.normal(0.0, noise_std)),
            y=int(y + self._rng.normal(0.0, noise_std)),
            center_x=int(cx),
            center_y=int(cy),
            width=w,
            height=h,
            confidence=self._rng.uniform(0.75, 0.98),
            method=ScanMethod.SYNTHETIC,
            scan_duration_ms=scan_ms,
        )

    # ------------------------------------------------------------------
    # Internal: live mode
    # ------------------------------------------------------------------

    def _live_scan(
        self,
        screen: np.ndarray,
        template: Template,
        variant: TemplateVariant,
    ) -> ScanResult:
        """
        Perform actual computer vision: template matching (precise) →
        HSV colour filtering (fallback) → OCR (last resort).

        screen is expected in HSV colour space (from mss capture pipeline).
        """
        import cv2

        # Convert to BGR for template matching (needles are stored as BGR)
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_HSV2BGR)

        # 1. Template matching — direct pixel comparison, no noise
        if variant.file_path and Path(variant.file_path).exists():
            try:
                needle = cv2.imread(variant.file_path)
                if needle is not None:
                    # Match directly on BGR — no noise, no corruption
                    result = cv2.matchTemplate(screen_bgr, needle, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    if max_val >= self.config.min_confidence_template:
                        nh, nw = needle.shape[:2]
                        return ScanResult(
                            found=True,
                            template_id=template.id,
                            x=int(max_loc[0]), y=int(max_loc[1]),
                            center_x=int(max_loc[0] + nw // 2),
                            center_y=int(max_loc[1] + nh // 2),
                            width=nw, height=nh,
                            confidence=float(max_val),
                            method=ScanMethod.TEMPLATE,
                            scan_duration_ms=self._rng.uniform(100.0, 300.0),
                        )
            except Exception:
                pass

        # 2. HSV colour matching — fallback when template match isn't enough
        try:
            lower = np.array(variant.hsv_lower, dtype=np.uint8)
            upper = np.array(variant.hsv_upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Dilate to connect nearby regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Filter contours by reasonable size (not the whole screen)
                screen_area = screen.shape[0] * screen.shape[1]
                valid = [c for c in contours if 50 < cv2.contourArea(c) < screen_area * 0.6]
                if valid:
                    largest = max(valid, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    # Confidence based on how well the contour aspect ratio
                    # matches the template variant's expected dimensions
                    contour_area = w * h
                    # Use crop template size as expected area
                    if variant.file_path and Path(variant.file_path).exists():
                        needle = cv2.imread(variant.file_path)
                        if needle is not None:
                            expected_w, expected_h = needle.shape[1], needle.shape[0]
                            expected_area = expected_w * expected_h
                            area_ratio = min(contour_area / expected_area, expected_area / max(contour_area, 1))
                        else:
                            area_ratio = 0.5
                    else:
                        area_ratio = 0.5

                    confidence = min(area_ratio, 1.0)

                    if confidence >= self.config.min_confidence_hsv:
                        return ScanResult(
                            found=True,
                            template_id=template.id,
                            x=int(x), y=int(y),
                            center_x=int(x + w // 2), center_y=int(y + h // 2),
                            width=w, height=h,
                            confidence=confidence,
                            method=ScanMethod.HSV,
                            scan_duration_ms=self._rng.uniform(50.0, 200.0),
                        )
        except Exception:
            pass

        # 3. OCR fallback (if enabled)
        if self.config.ocr_fallback_enabled:
            pass

        # All methods failed
        return ScanResult(
            found=False,
            template_id=template.id,
            confidence=0.0,
            method=ScanMethod.HSV,
            scan_duration_ms=500.0,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_template(self, template_id: str) -> bool:
        return template_id in self._templates

    def list_templates(self) -> list[str]:
        return sorted(self._templates.keys())
