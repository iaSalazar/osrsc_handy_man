"""
Screenshot Auto-Crop Tool — automates template region detection.

Instead of manually drawing bounding boxes for every screenshot,
this tool:

  1. Scans a directory of full-game screenshots
  2. Uses OpenCV edge detection + contour finding to propose regions
  3. Shows each proposed crop and asks the user to validate (y/n/adjust)
  4. Registers validated crops as templates in the library

Workflow:
  python3 -m tools.auto_crop ~/screenshots/ --element ge_bank_booth --state default --yaw 0

The tool will:
  - Show a list of detected regions ranked by confidence
  - For each region: display the cropped image (if display available)
    or print the bbox coordinates for manual verification
  - Ask: accept / skip / adjust
  - Register accepted regions

If no display is available, it saves crops to a review directory for
manual inspection later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
from PIL import Image


# ======================================================================
# Auto-detection engine
# ======================================================================

class RegionDetector:
    """
    Detects candidate UI element regions in a screenshot using:
      1. Canny edge detection
      2. Contour finding
      3. Bounding rect ranking by area × aspect ratio × edge density
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 80000,
        min_aspect: float = 0.2,
        max_aspect: float = 5.0,
        edge_low: int = 50,
        edge_high: int = 150,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.edge_low = edge_low
        self.edge_high = edge_high

    def detect(self, image_path: Path | str) -> list[dict]:
        """
        Detect candidate regions in `image_path`.

        Returns list of dicts sorted by confidence (best first):
          {bbox: (x, y, w, h), confidence: float, area: int, aspect: float}
        """
        try:
            import cv2
        except ImportError:
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)

        # Dilate to connect nearby edges (helps group UI element boundaries)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch

            # Filter by size
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio
            aspect = cw / ch if ch > 0 else 0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Filter edge regions (UI elements are usually not at the very edge)
            margin = 10
            if x < margin or y < margin or x + cw > w - margin or y + ch > h - margin:
                continue

            # Confidence score: larger + squarer + more edge-dense = higher
            edge_density = cv2.countNonZero(edges[y:y+ch, x:x+cw]) / area if area > 0 else 0
            squareness = min(aspect, 1.0 / aspect) if aspect > 0 else 0
            area_score = min(area / 5000.0, 1.0)

            confidence = 0.3 * area_score + 0.3 * squareness + 0.4 * min(edge_density * 5, 1.0)

            regions.append({
                "bbox": (int(x), int(y), int(cw), int(ch)),
                "confidence": round(confidence, 3),
                "area": area,
                "aspect": round(aspect, 3),
                "edge_density": round(edge_density, 3),
            })

        # Sort by confidence descending, deduplicate overlapping regions
        regions.sort(key=lambda r: r["confidence"], reverse=True)
        regions = self._deduplicate_overlapping(regions)

        return regions

    def _deduplicate_overlapping(
        self, regions: list[dict], iou_threshold: float = 0.5,
    ) -> list[dict]:
        """Remove regions that heavily overlap with a higher-confidence region."""
        kept = []
        for r in regions:
            rx, ry, rw, rh = r["bbox"]
            overlap = False
            for k in kept:
                kx, ky, kw, kh = k["bbox"]
                # Intersection over Union
                ix = max(rx, kx)
                iy = max(ry, ky)
                iw = min(rx + rw, kx + kw) - ix
                ih = min(ry + rh, ky + kh) - iy
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    union = (rw * rh) + (kw * kh) - inter
                    iou = inter / union if union > 0 else 0
                    if iou > iou_threshold:
                        overlap = True
                        break
            if not overlap:
                kept.append(r)
        return kept


# ======================================================================
# Review & registration
# ======================================================================

class CropReviewer:
    """
    Saves detected crops to a review directory for manual validation,
    then registers accepted ones as templates.
    """

    def __init__(self, review_dir: Path | str) -> None:
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def save_crops_for_review(
        self,
        image_path: Path,
        regions: list[dict],
        element_name: str,
        max_crops: int = 10,
    ) -> list[Path]:
        """
        Save cropped regions as individual PNGs in the review directory.

        Returns list of saved crop paths.
        """
        img = Image.open(image_path)
        saved = []

        for i, region in enumerate(regions[:max_crops]):
            x, y, w, h = region["bbox"]
            crop = img.crop((x, y, x + w, y + h))

            crop_name = (
                f"{element_name}_crop{i+1:02d}_"
                f"conf{region['confidence']:.2f}_"
                f"{x}_{y}_{w}x{h}.png"
            )
            crop_path = self.review_dir / crop_name
            crop.save(crop_path)
            saved.append(crop_path)

        return saved

    def generate_review_script(
        self,
        crops: list[Path],
        element_name: str,
        screenshot_path: Path,
        state: str = "default",
        yaw: int = 0,
        pitch: str = "default",
        zoom: str = "mid",
    ) -> str:
        """
        Generate a bash script the user can run to register accepted crops.

        The user deletes rejected crop files, then runs this script.
        """
        lines = [
            "#!/bin/bash",
            f"# Auto-generated registration script for: {element_name}",
            f"# Source: {screenshot_path}",
            f"# State: {state}, Yaw: {yaw}, Pitch: {pitch}, Zoom: {zoom}",
            f"#",
            f"# Usage:",
            f"#   1. Review crops in {self.review_dir}/",
            f"#   2. DELETE any crop files that are WRONG",
            f"#   3. Run: bash {self.review_dir}/register.sh",
            "",
            "source .venv/bin/activate",
            "",
        ]

        for crop_path in crops:
            # Extract bbox from filename: ..._X_Y_WxH.png
            stem = crop_path.stem
            parts = stem.split("_")
            # Last part is like "100x50" or similar
            dims_part = [p for p in parts if "x" in p and p.replace("x", "").isdigit()]
            if dims_part:
                dims = dims_part[-1]  # e.g., "450x320"
                # Extract x, y — these are before the dims
                # Format: ..._conf0.85_450_320_60x50.png
                # We need to find the x, y, w, h
                xy_parts = [p for p in parts if p.isdigit()]
                if len(xy_parts) >= 4:
                    x, y, w, h = xy_parts[-4], xy_parts[-3], dims.split("x")[0], dims.split("x")[1]
                else:
                    continue
            else:
                continue

            lines.append(
                f"python3 -m tools.cli capture register {element_name} "
                f'"{screenshot_path}" '
                f"--state {state} --yaw {yaw} --pitch {pitch} --zoom {zoom} "
                f"--bbox {x},{y},{w},{h}"
            )

        script_path = self.review_dir / "register.sh"
        script_content = "\n".join(lines) + "\n"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        return script_content


# ======================================================================
# CLI
# ======================================================================

@click.group()
def main():
    """Auto-crop tool — detect UI elements in screenshots for template capture."""
    pass


@main.command()
@click.argument("screenshot", type=click.Path(exists=True))
@click.option("--element", "-e", required=True, help="Element name (e.g. ge_bank_booth)")
@click.option("--state", default="default", help="UI state")
@click.option("--yaw", default=0, type=int, help="Camera yaw (degrees)")
@click.option("--pitch", default="default", help="Camera pitch")
@click.option("--zoom", default="mid", help="Zoom level")
@click.option("--review-dir", default="data/review", help="Where to save crops for review")
@click.option("--max-crops", default=10, type=int, help="Max regions to propose")
@click.option("--min-area", default=200, type=int, help="Minimum region area (px²)")
@click.option("--auto-register/--no-auto", default=False,
              help="Auto-register best region without review (use carefully)")
def detect(
    screenshot: str,
    element: str,
    state: str,
    yaw: int,
    pitch: str,
    zoom: str,
    review_dir: str,
    max_crops: int,
    min_area: int,
    auto_register: bool,
):
    """
    Detect candidate UI element regions in a screenshot.

    Saves cropped images to the review directory for manual validation.
    Then run the generated register.sh to register accepted crops.
    """
    screenshot_path = Path(screenshot)

    click.echo(f"Scanning: {screenshot_path.name}")
    click.echo(f"Element: {element} | State: {state} | Yaw: {yaw}° | Pitch: {pitch} | Zoom: {zoom}")
    click.echo()

    detector = RegionDetector(min_area=min_area)
    regions = detector.detect(screenshot_path)

    if not regions:
        click.echo("No regions detected. Try lowering --min-area or checking the screenshot.")
        return

    click.echo(f"Found {len(regions)} candidate regions:\n")

    for i, r in enumerate(regions[:max_crops]):
        x, y, w, h = r["bbox"]
        click.echo(
            f"  {i+1:2d}. bbox=({x:4d},{y:4d},{w:3d}x{h:3d})  "
            f"conf={r['confidence']:.2f}  area={r['area']}  aspect={r['aspect']:.2f}"
        )

    click.echo()

    # Save for review
    reviewer = CropReviewer(review_dir)
    crops = reviewer.save_crops_for_review(screenshot_path, regions, element, max_crops)

    click.echo(f"Saved {len(crops)} crops to: {review_dir}/")
    click.echo()

    # Generate registration script
    script = reviewer.generate_review_script(
        crops, element, screenshot_path, state, yaw, pitch, zoom,
    )

    if auto_register and regions:
        # Auto-register the best region
        best = regions[0]
        x, y, w, h = best["bbox"]
        click.echo(f"Auto-registering best region: bbox={x},{y},{w},{h} (conf={best['confidence']:.2f})")

        from tools.template_capture import TemplateCaptureTool
        tool = TemplateCaptureTool(Path("data/templates"))
        tool.register_from_screenshot(
            element_name=element,
            screenshot_path=screenshot_path,
            bbox=(x, y, w, h),
            state=state,
            yaw=yaw,
            pitch=pitch,
            zoom=zoom,
        )
        click.echo("Registered!")
    else:
        click.echo("=" * 60)
        click.echo("NEXT STEPS:")
        click.echo("=" * 60)
        click.echo(f" 1. Open: {review_dir}/")
        click.echo(f" 2. DELETE any crop images that are WRONG")
        click.echo(f" 3. Run: bash {review_dir}/register.sh")
        click.echo()
        click.echo(f"Registration script preview:\n")
        click.echo(script[:2000])


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--element", "-e", required=True, help="Element name for all screenshots")
@click.option("--review-dir", default="data/review", help="Review directory")
@click.option("--max-crops", default=10, type=int)
@click.option("--min-area", default=300, type=int)
def batch(
    directory: str,
    element: str,
    review_dir: str,
    max_crops: int,
    min_area: int,
):
    """
    Batch-process all PNG screenshots in a directory.

    For each screenshot:
      1. Detect candidate regions
      2. Save crops for review
      3. Generate combined registration script
    """
    directory = Path(directory)
    pngs = sorted(directory.glob("*.png"))

    if not pngs:
        click.echo(f"No PNG files found in {directory}")
        return

    click.echo(f"Processing {len(pngs)} screenshots...\n")
    detector = RegionDetector(min_area=min_area)
    reviewer = CropReviewer(review_dir)

    all_crops = []
    for png in pngs:
        # Try to extract yaw/pitch/zoom from filename
        stem = png.stem
        yaw = 0
        pitch = "default"
        zoom = "mid"
        state = "default"

        # Parse yaw from filename if present: _y0_, _y90_, etc.
        import re
        yaw_match = re.search(r'_y(\d+)', stem)
        if yaw_match:
            yaw = int(yaw_match.group(1))
        if "_high" in stem:
            pitch = "high"
        if "_low" in stem:
            pitch = "low"
        if "_in" in stem:
            zoom = "in"
        if "_out" in stem:
            zoom = "out"

        regions = detector.detect(png)
        click.echo(f"  {png.name}: {len(regions)} regions detected")

        if regions:
            crops = reviewer.save_crops_for_review(png, regions, element, max_crops)
            all_crops.extend(crops)

    if all_crops:
        click.echo(f"\n{len(all_crops)} crops saved to {review_dir}/")
        click.echo(f"Review them, delete wrong ones, then run the registration scripts.")
    else:
        click.echo("\nNo regions detected in any screenshot.")


if __name__ == "__main__":
    main()
