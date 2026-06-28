"""
Live-Mode Test Utility — verify templates against real screenshots.

Quickly test whether your captured templates can find UI elements in
actual game screenshots.  No game client needed — works on PNG files.

Usage:
    # Test a single template against a screenshot
    python3 tools/live_test.py scan login_existing_user_btn data/screenshots/login/login_screen.png

    # Test all login templates against a login screenshot
    python3 tools/live_test.py login data/screenshots/login/login_screen.png

    # Test bank templates
    python3 tools/live_test.py bank data/screenshots/bank/ge_bank_y0.png

    # Interactive mode — show matches with bounding boxes drawn on the image
    python3 tools/live_test.py scan ge_bank_booth data/screenshots/bank/ge.png --show

    # List all registered templates
    python3 tools/live_test.py list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np


# ======================================================================
# Template -> Screenshot matcher
# ======================================================================

class LiveTester:
    """
    Runs the VisualScanner in live mode against a static screenshot file.
    """

    def __init__(self, templates_dir: str = "data/templates") -> None:
        from core.config import ScanConfig
        from core.scan import VisualScanner

        self._rng = np.random.default_rng()
        self.config = ScanConfig()
        self.scanner = VisualScanner(
            self.config, self._rng,
            template_dir=Path(templates_dir),
        )

    def test_scan(
        self,
        template_id: str,
        screenshot_path: Path,
        show: bool = False,
    ) -> dict:
        """
        Try to find `template_id` in `screenshot_path`.

        Returns a dict with the result, confidence, position, etc.
        """
        import cv2

        img = cv2.imread(str(screenshot_path))
        if img is None:
            return {"error": f"Could not load: {screenshot_path}"}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        result = self.scanner.find_element(
            template_id,
            conditions={},
            screen=hsv,
        )

        info = {
            "template_id": template_id,
            "found": result.found,
            "position": (result.center_x, result.center_y),
            "bbox": (result.x, result.y, result.width, result.height),
            "confidence": round(result.confidence, 3),
            "method": result.method.value,
            "scan_duration_ms": round(result.scan_duration_ms, 2),
            "screenshot": str(screenshot_path),
        }

        if show and result.found:
            self._draw_result(img, result)
        elif show:
            click.echo("  (not found — nothing to show)")

        return info

    def test_all_templates(
        self,
        screenshot_path: Path,
        show: bool = False,
    ) -> list[dict]:
        """Try every registered template against a screenshot."""
        results = []
        for tid in self.scanner.list_templates():
            result = self.test_scan(tid, screenshot_path, show=False)
            results.append(result)
        results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return results

    def _draw_result(self, img: np.ndarray, result) -> None:
        """Draw a bounding box on the image and save it for review."""
        import cv2

        if not result.found:
            return

        x, y, w, h = result.x, result.y, result.width, result.height
        cx, cy = result.center_x, result.center_y

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # Label
        label = f"{result.template_id} ({result.confidence:.2f})"
        cv2.putText(img, label, (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save annotated image
        out_path = Path("data/review") / f"annotated_{result.template_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        click.echo(f"  Annotated image saved: {out_path}")


# ======================================================================
# Template set definitions — groups of templates for each UI
# ======================================================================

LOGIN_TEMPLATES = [
    # Screen 1: disconnected
    "login_disconnected_screen",
    "login_disconnected_continue",
    # Screen 2: main login
    "login_screen_title",
    "login_existing_user_btn",
    "login_click_to_play_btn",
    # Optional: typing fields
    "login_username_field",
    "login_password_field",
    # Post-login
    "game_view_minimap",
]

BANK_TEMPLATES = [
    "ge_bank_booth",
    "bank_interface_open",
    "bank_close_button",
]

PIN_TEMPLATES = [
    "bank_pin_screen",
] + [f"bank_pin_digit_{d}" for d in range(10)]

LOGOUT_TEMPLATES = [
    "logout_door_icon",
    "logout_tab_button",
    "logout_confirm_button",
    "login_screen_title",
]

TEMPLATE_SETS = {
    "login": LOGIN_TEMPLATES,
    "bank": BANK_TEMPLATES,
    "pin": PIN_TEMPLATES,
    "logout": LOGOUT_TEMPLATES,
    "all": LOGIN_TEMPLATES + BANK_TEMPLATES + PIN_TEMPLATES + LOGOUT_TEMPLATES,
}


# ======================================================================
# CLI
# ======================================================================

@click.group()
def main():
    """Live-mode test — verify templates against real screenshots."""


@main.command()
@click.argument("template_id")
@click.argument("screenshot", type=click.Path(exists=True))
@click.option("--templates-dir", default="data/templates", help="Template library path")
@click.option("--show/--no-show", default=False, help="Draw bounding boxes on the image")
def scan(template_id: str, screenshot: str, templates_dir: str, show: bool):
    """
    Test a single TEMPLATE_ID against a SCREENSHOT.

    Example:
      python3 tools/live_test.py scan ge_bank_booth data/screenshots/bank/ge.png --show
    """
    tester = LiveTester(templates_dir)

    if not tester.scanner.has_template(template_id):
        click.echo(f"Template '{template_id}' not registered.")
        click.echo(f"Registered templates: {tester.scanner.list_templates()}")
        return

    result = tester.test_scan(template_id, Path(screenshot), show=show)

    if "error" in result:
        click.echo(f"ERROR: {result['error']}")
        return

    if result["found"]:
        click.echo(f"✅ FOUND  {template_id}")
        click.echo(f"   Position: ({result['position'][0]}, {result['position'][1]})")
        click.echo(f"   Bbox:     {result['bbox']}")
        click.echo(f"   Confidence: {result['confidence']:.3f}")
        click.echo(f"   Method:   {result['method']}")
    else:
        click.echo(f"❌ NOT FOUND  {template_id}")
        click.echo(f"   Confidence: {result['confidence']:.3f}")
        click.echo(f"   Method: {result['method']}")
        click.echo(f"   Try: lower confidence threshold, capture more variants, or check HSV ranges")


@main.command()
@click.argument("set_name")
@click.argument("screenshot", type=click.Path(exists=True))
@click.option("--templates-dir", default="data/templates", help="Template library path")
def test_set(set_name: str, screenshot: str, templates_dir: str):
    """
    Test a SET of templates against a SCREENSHOT.

    SET_NAME is one of: login, bank, pin, logout, all

    Example:
      python3 tools/live_test.py test-set login data/screenshots/login/login_screen.png
    """
    if set_name not in TEMPLATE_SETS:
        click.echo(f"Unknown set: {set_name}")
        click.echo(f"Available sets: {list(TEMPLATE_SETS.keys())}")
        return

    template_ids = TEMPLATE_SETS[set_name]
    tester = LiveTester(templates_dir)

    registered = tester.scanner.list_templates()
    to_test = [t for t in template_ids if t in registered]
    missing = [t for t in template_ids if t not in registered]

    if missing:
        click.echo(f"⚠  Missing templates (not registered yet):")
        for t in missing:
            click.echo(f"   - {t}")
        click.echo()

    if not to_test:
        click.echo("No templates to test. Register templates first:")
        click.echo("  python3 -m tools.auto_crop detect <screenshot> --element <name>")
        return

    click.echo(f"Testing {len(to_test)} templates against: {screenshot}\n")

    found = 0
    for tid in to_test:
        result = tester.test_scan(tid, Path(screenshot), show=False)
        status = "✅" if result["found"] else "❌"
        conf = result.get("confidence", 0)
        pos = result.get("position", (0, 0))
        click.echo(f"  {status} {tid:<35} conf={conf:.3f}  pos=({pos[0]}, {pos[1]})")
        if result["found"]:
            found += 1

    click.echo(f"\n{found}/{len(to_test)} found  ({missing and f'{len(missing)} missing' or ''})")


@main.command()
@click.option("--templates-dir", default="data/templates", help="Template library path")
def list_templates(templates_dir: str):
    """List all registered templates with variant counts."""
    from tools.template_capture import TemplateCaptureTool
    tool = TemplateCaptureTool(templates_dir)
    elements = tool.list_elements()

    if not elements:
        click.echo("No templates registered.")
        click.echo("Capture screenshots, then run:")
        click.echo("  python3 -m tools.auto_crop detect <screenshot> --element <name>")
        return

    click.echo(f"{'Element':<35} {'Variants':<10} {'States':<30} {'Tags'}")
    click.echo("-" * 90)
    for el in elements:
        states = ", ".join(el["states"][:3])
        tags = ", ".join(el.get("tags", [])[:3])
        click.echo(f"{el['name']:<35} {el['variant_count']:<10} {states:<30} {tags}")


@main.command()
@click.argument("screenshot", type=click.Path(exists=True))
@click.option("--templates-dir", default="data/templates", help="Template library path")
@click.option("--top", default=5, type=int, help="Show top N matches")
def explore(screenshot: str, templates_dir: str, top: int):
    """
    Scan a SCREENSHOT against ALL registered templates and show best matches.

    Useful when you're not sure what's in a screenshot — it shows you
    what templates match best.
    """
    tester = LiveTester(templates_dir)

    if not tester.scanner.list_templates():
        click.echo("No templates registered.")
        return

    click.echo(f"Scanning {screenshot} against all templates...\n")
    results = tester.test_all_templates(Path(screenshot), show=False)

    click.echo(f"Top {min(top, len(results))} matches:\n")

    shown = 0
    for r in results:
        if r["found"] and shown < top:
            click.echo(
                f"  {r['template_id']:<35} "
                f"conf={r['confidence']:.3f}  "
                f"pos=({r['position'][0]}, {r['position'][1]})  "
                f"method={r['method']}"
            )
            shown += 1

    if shown == 0:
        click.echo("  Nothing found.")
        click.echo("  Templates may need more variants or adjusted HSV ranges.")


if __name__ == "__main__":
    main()
