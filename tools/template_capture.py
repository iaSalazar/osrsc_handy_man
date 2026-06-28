"""
UI Template Capture Tool — builds the template library for the colour-bot.

Operates on pre-captured screenshots (PNG).  Supports:
  - Interactive bounding-box selection for UI elements
  - Automatic HSV mask generation from the selected region
  - Variant registration (state, camera angle, zoom level)
  - Batch processing of screenshot directories
  - Template validation and thumbnail generation

Usage:
    python -m tools.template_capture register <element> <screenshot> [--state ...]
    python -m tools.template_capture batch <directory> [--auto-detect]
    python -m tools.template_capture list
    python -m tools.template_capture validate <element>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image


class TemplateCaptureTool:
    """
    Manages the template library on disk.

    Directory structure created:
        data/templates/<element_name>/
            template.yaml
            variants/
                <element>_<state>_<yaw>_<pitch>_<zoom>.png
            hsv_masks/
                <element>_<state>_<yaw>_<pitch>_<zoom>_mask.png
    """

    def __init__(self, templates_dir: Path | str) -> None:
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Register a single element from a screenshot
    # ------------------------------------------------------------------

    def register_from_screenshot(
        self,
        element_name: str,
        screenshot_path: Path | str,
        bbox: tuple[int, int, int, int],
        state: str = "default",
        yaw: int = 0,
        pitch: str = "default",
        zoom: str = "mid",
        resolution: tuple[int, int] = (1920, 1080),
        dpi_scaling: float = 1.0,
    ) -> Path:
        """
        Register a new template variant from a screenshot region.

        Parameters
        ----------
        element_name : Identifier for this UI element (e.g., "chaos_altar").
        screenshot_path : Path to the full screenshot PNG.
        bbox : (x, y, width, height) of the element in the screenshot.
        state : UI state (e.g., "open", "closed", "enabled", "disabled").
        yaw : Camera yaw angle in degrees (0=N, 90=E, 180=S, 270=W).
        pitch : Camera pitch ("high", "default", "low").
        zoom : Zoom level ("in", "mid", "out").
        resolution : Screen resolution of the screenshot.
        dpi_scaling : Display scaling factor (1.0 = 100%).

        Returns
        -------
        Path to the updated template.yaml.
        """
        element_dir = self.templates_dir / element_name
        variants_dir = element_dir / "variants"
        masks_dir = element_dir / "hsv_masks"
        variants_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Load and crop screenshot
        img = Image.open(screenshot_path)
        x, y, w, h = bbox
        cropped = img.crop((x, y, x + w, y + h))

        # Save variant image
        variant_filename = (
            f"{element_name}_{state}_y{yaw}_{pitch}_{zoom}_{resolution[0]}x{resolution[1]}.png"
        )
        variant_path = variants_dir / variant_filename
        cropped.save(variant_path)

        # Generate HSV mask
        hsv_lower, hsv_upper = self._auto_hsv_range(np.array(cropped))
        mask_img = self._generate_mask_image(np.array(cropped), hsv_lower, hsv_upper)
        mask_path = masks_dir / variant_filename.replace(".png", "_mask.png")
        Image.fromarray(mask_img).save(mask_path)

        # Update or create template.yaml
        template_yaml = element_dir / "template.yaml"
        template_data = self._load_template_yaml(template_yaml, element_name)

        # Add variant
        variant_entry = {
            "file": f"variants/{variant_filename}",
            "mask": f"hsv_masks/{variant_filename.replace('.png', '_mask.png')}",
            "conditions": {
                "state": state,
                "yaw": str(yaw),
                "pitch": pitch,
                "zoom": zoom,
                "resolution": f"{resolution[0]}x{resolution[1]}",
            },
            "bbox": [x, y, w, h],
            "hsv_lower": [int(v) for v in hsv_lower],
            "hsv_upper": [int(v) for v in hsv_upper],
            "confidence_threshold": 0.65,
        }

        # Avoid duplicate variants
        existing = template_data.get("variants", [])
        existing = [v for v in existing if v.get("file") != variant_entry["file"]]
        existing.append(variant_entry)
        template_data["variants"] = existing

        # Set default bbox if not set
        if "default_bbox" not in template_data:
            template_data["default_bbox"] = [x, y, w, h]

        with open(template_yaml, "w") as f:
            yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

        return template_yaml

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_register(
        self,
        screenshot_dir: Path | str,
        auto_detect: bool = False,
    ) -> list[Path]:
        """
        Process all PNG screenshots in a directory.

        If auto_detect is True, use edge detection to suggest element regions.
        Otherwise, prompts for manual bbox per screenshot (CLI mode).
        """
        screenshot_dir = Path(screenshot_dir)
        results = []
        for png in sorted(screenshot_dir.glob("*.png")):
            # Parse filename for metadata hints
            # Expected: element_state_yaw_pitch_zoom.png
            stem = png.stem
            parts = stem.split("_")
            element_name = parts[0] if parts else "unknown"
            state = parts[1] if len(parts) > 1 else "default"

            print(f"Processing: {png.name} → element='{element_name}' state='{state}'")

            if auto_detect:
                regions = self.auto_detect_regions(png)
                for i, region in enumerate(regions):
                    bbox = region["bbox"]
                    print(f"  Region {i+1}: {bbox} (confidence: {region['confidence']:.2f})")
            else:
                print("  Enter bbox as: x,y,w,h (or 'skip')")
                # In CLI mode, this would use click.prompt()

        return results

    # ------------------------------------------------------------------
    # Listing & validation
    # ------------------------------------------------------------------

    def list_elements(self) -> list[dict]:
        """List all registered elements and their variant counts."""
        elements = []
        if not self.templates_dir.exists():
            return elements

        for elem_dir in sorted(self.templates_dir.iterdir()):
            if not elem_dir.is_dir():
                continue
            yaml_path = elem_dir / "template.yaml"
            if not yaml_path.exists():
                continue
            data = self._load_template_yaml(yaml_path, elem_dir.name)
            elements.append({
                "name": elem_dir.name,
                "label": data.get("label", elem_dir.name),
                "variant_count": len(data.get("variants", [])),
                "tags": data.get("tags", []),
                "states": list(set(
                    v.get("conditions", {}).get("state", "default")
                    for v in data.get("variants", [])
                )),
            })
        return elements

    def validate_element(self, element_name: str) -> list[str]:
        """Check a template element for consistency issues."""
        warnings = []
        element_dir = self.templates_dir / element_name
        yaml_path = element_dir / "template.yaml"

        if not yaml_path.exists():
            return [f"ERROR: No template.yaml found for '{element_name}'"]

        data = self._load_template_yaml(yaml_path, element_name)
        variants = data.get("variants", [])

        if not variants:
            warnings.append(f"WARNING: '{element_name}' has no variants")

        for i, v in enumerate(variants):
            vpath = element_dir / v.get("file", "")
            if not vpath.exists():
                warnings.append(f"ERROR: Variant {i} file missing: {v.get('file')}")

            # Check HSV range validity
            lower = v.get("hsv_lower", [])
            upper = v.get("hsv_upper", [])
            if len(lower) != 3 or len(upper) != 3:
                warnings.append(f"WARNING: Variant {i} has invalid HSV range")

        return warnings

    # ------------------------------------------------------------------
    # Auto-detection (simple edge-based)
    # ------------------------------------------------------------------

    def auto_detect_regions(
        self,
        screenshot_path: Path | str,
        min_area: int = 400,
    ) -> list[dict]:
        """
        Use edge detection + contour finding to suggest UI element regions.

        Returns list of {bbox: (x, y, w, h), confidence: float}.
        """
        try:
            import cv2
        except ImportError:
            return []

        img = cv2.imread(str(screenshot_path))
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= min_area:
                # Simple confidence: larger, squarer regions score higher
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                confidence = min(area / 10000.0, 1.0) * (0.5 + 0.5 * aspect)
                regions.append({
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": round(confidence, 3),
                })

        # Sort by confidence descending
        regions.sort(key=lambda r: r["confidence"], reverse=True)
        return regions[:20]  # top 20 candidates

    # ------------------------------------------------------------------
    # Thumbnail grid
    # ------------------------------------------------------------------

    def generate_thumbnail_grid(
        self, element_name: str, output_path: Path | str,
    ) -> None:
        """Create a grid image of all variants for documentation."""
        element_dir = self.templates_dir / element_name
        variants_dir = element_dir / "variants"

        if not variants_dir.exists():
            print(f"No variants found for '{element_name}'")
            return

        images = []
        for png in sorted(variants_dir.glob("*.png")):
            try:
                img = Image.open(png)
                img.thumbnail((200, 200))
                images.append(img)
            except Exception:
                pass

        if not images:
            return

        # Arrange in a grid
        cols = min(4, len(images))
        rows = (len(images) + cols - 1) // cols
        cell_w = max(img.width for img in images) + 10
        cell_h = max(img.height for img in images) + 10

        grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            grid.paste(img, (c * cell_w + 5, r * cell_h + 5))

        grid.save(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_template_yaml(self, yaml_path: Path, element_name: str) -> dict:
        """Load existing template.yaml or create skeleton."""
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {
            "id": element_name,
            "label": element_name.replace("_", " ").title(),
            "tags": [],
            "default_bbox": [0, 0, 20, 20],
            "variants": [],
        }

    @staticmethod
    def _auto_hsv_range(
        cropped: np.ndarray,
        margin: float = 0.15,
    ) -> tuple[list[int], list[int]]:
        """
        Compute HSV mask range from the cropped region.

        Takes the 5th–95th percentile of each HSV channel and expands by margin.
        """
        try:
            import cv2
            hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        except ImportError:
            return ([0, 0, 0], [180, 255, 255])

        lower = []
        upper = []
        for ch in range(3):
            channel = hsv[:, :, ch].ravel()
            lo = float(np.percentile(channel, 5))
            hi = float(np.percentile(channel, 95))
            spread = (hi - lo) * margin
            lower.append(max(lo - spread, 0))
            upper.append(min(hi + spread, 255 if ch > 0 else 180))

        return (lower, upper)

    @staticmethod
    def _generate_mask_image(
        cropped: np.ndarray,
        lower: list[int],
        upper: list[int],
    ) -> np.ndarray:
        """Generate a binary mask image showing what the HSV range selects."""
        try:
            import cv2
            hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            return mask
        except ImportError:
            return np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)
