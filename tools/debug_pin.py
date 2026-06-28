"""
PIN Entry Debug Tool — step-by-step validation of each digit.

Usage:
    python3 tools/debug_pin.py

This will:
  1. Login (if needed)
  2. Open bank (find banker, right-click, select "Bank")
  3. For each PIN digit: capture screen, show ALL detected digits with
     confidence scores, highlight the target digit, and WAIT for you to
     press Enter before clicking.
  4. After each click, re-scan (OSRS shuffles digit positions after each
     digit is entered) and repeat.

Press Enter to approve and click. Type 'skip' to skip clicking (will
re-scan instead). Type 'quit' to abort. Press Ctrl+C anytime to abort.
"""

import subprocess
import time
import mss
import cv2
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def click(x, y, right=False):
    subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y))], timeout=5)
    time.sleep(0.08)
    subprocess.run(["xdotool", "click", "3" if right else "1"], timeout=5)

def capture():
    with mss.MSS() as sct:
        raw = np.array(sct.grab(sct.monitors[1]))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

# ---------------------------------------------------------------------------
# Login (same proven logic from run_live.py)
# ---------------------------------------------------------------------------

def _detect_login_state(bgr, gray, hsv):
    h, w = bgr.shape[:2]

    # ---- Check: disconnected screen ----
    disc_tmpl = cv2.imread("data/screenshots/login/you_were_disconected.png")
    disc_tmpl_conf = 0.0
    if disc_tmpl is not None:
        r = cv2.matchTemplate(bgr, disc_tmpl, cv2.TM_CCOEFF_NORMED)
        _, disc_tmpl_conf, _, _ = cv2.minMaxLoc(r)
    ok_btn = cv2.imread("data/screenshots/login/disconnected_ok_button.png")
    ok_conf = 0.0
    if ok_btn is not None:
        r = cv2.matchTemplate(bgr, ok_btn, cv2.TM_CCOEFF_NORMED)
        _, ok_conf, _, _ = cv2.minMaxLoc(r)
    if any([disc_tmpl_conf > 0.6, ok_conf > 0.7]):
        return 'disconnected'

    # ---- Check: red "Click here to play" button ----
    red_tmpl = cv2.imread("data/screenshots/login/play_now_red_button.png")
    red_tmpl_conf = 0.0
    if red_tmpl is not None:
        r = cv2.matchTemplate(bgr, red_tmpl, cv2.TM_CCOEFF_NORMED)
        _, red_tmpl_conf, _, _ = cv2.minMaxLoc(r)
    cy1, cy2 = int(h*0.42), int(h*0.65)
    cx1, cx2 = int(w*0.30), int(w*0.70)
    crop = hsv[cy1:cy2, cx1:cx2]
    red_mask = cv2.inRange(crop, np.array([0,70,70]), np.array([12,255,255]))
    red_mask |= cv2.inRange(crop, np.array([170,70,70]), np.array([180,255,255]))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_blob = False
    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if area > 3000 and 1.8 < aspect < 5.0:
            red_blob = True
            break
    if red_tmpl_conf > 0.6 or red_blob:
        return 'red_button'

    # ---- Check: Play Now / Welcome screen ----
    play_tmpl = cv2.imread("data/screenshots/login/login_play_now_text.png")
    play_tmpl_conf = 0.0
    if play_tmpl is not None:
        r = cv2.matchTemplate(bgr, play_tmpl, cv2.TM_CCOEFF_NORMED)
        _, play_tmpl_conf, _, _ = cv2.minMaxLoc(r)

    if play_tmpl_conf > 0.7:
        return 'play_now'

    return 'in_game'


def do_login():
    """Login flow — returns True once in-game."""
    print("\n" + "="*60)
    print("STEP 1: LOGIN")
    print("="*60)

    for attempt in range(15):
        bgr = capture()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        state = _detect_login_state(bgr, gray, hsv)
        print(f"  State: {state} (attempt {attempt+1})")

        if state == 'in_game':
            print("  ✅ In-game — login complete\n")
            return True

        if state == 'disconnected':
            ok_btn = cv2.imread("data/screenshots/login/disconnected_ok_button.png")
            if ok_btn is not None:
                r = cv2.matchTemplate(bgr, ok_btn, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                cx = loc[0] + ok_btn.shape[1]//2
                cy = loc[1] + ok_btn.shape[0]//2
                print(f"  [1/3] DISCONNECTED — clicking OK at ({cx},{cy}) conf={conf:.2f}")
                click(cx, cy)
                time.sleep(4)
                continue

        if state == 'play_now':
            play_tmpl = cv2.imread("data/screenshots/login/login_play_now_text.png")
            if play_tmpl is not None:
                r = cv2.matchTemplate(bgr, play_tmpl, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                cx = loc[0] + play_tmpl.shape[1]//2
                cy = loc[1] + play_tmpl.shape[0]//2
                print(f"  [2/3] PLAY NOW — clicking at ({cx},{cy}) conf={conf:.2f}")
                click(cx, cy)
                time.sleep(4)
                continue

        if state == 'red_button':
            red_tmpl = cv2.imread("data/screenshots/login/play_now_red_button.png")
            cx, cy = None, None
            if red_tmpl is not None:
                r = cv2.matchTemplate(bgr, red_tmpl, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                if conf > 0.6:
                    cx = loc[0] + red_tmpl.shape[1]//2
                    cy = loc[1] + red_tmpl.shape[0]//2
                    print(f"  [3/3] RED BUTTON — clicking at ({cx},{cy}) conf={conf:.2f}")
            if cx is None:
                h, w = bgr.shape[:2]
                crop = hsv[h//3:2*h//3, w//3:2*w//3]
                red_mask = cv2.inRange(crop, np.array([0,50,40]), np.array([20,255,255]))
                red_mask |= cv2.inRange(crop, np.array([170,50,40]), np.array([180,255,255]))
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 5000]
                if big:
                    (bx,by,bw,bh), _ = max(big, key=lambda b: b[1])
                    cx, cy = w//3 + bx + bw//2, h//3 + by + bh//2
                    print(f"  [3/3] RED BUTTON (HSV fallback) — clicking at ({cx},{cy})")
            if cx is not None:
                click(cx, cy)
                time.sleep(4)
                continue

        print(f"  Waiting... (state={state})")
        time.sleep(2)

    print("  ❌ Login failed")
    return False


# ---------------------------------------------------------------------------
# Open Bank
# ---------------------------------------------------------------------------

def open_bank():
    """Find banker NPC, right-click, select 'Bank'. Returns True if bank/PIN screen appeared."""
    print("="*60)
    print("STEP 2: OPEN BANK")
    print("="*60)

    bgr = capture()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Find magenta NPC (banker)
    magenta = cv2.inRange(hsv, np.array([140,100,60]), np.array([165,255,255]))
    game = np.zeros_like(magenta)
    game[50:950, 250:1700] = magenta[50:950, 250:1700]
    contours, _ = cv2.findContours(game, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 200]

    if not big:
        print("  ❌ No banker found (magenta NPC not visible)")
        return False

    (bx, by, bw, bh), area = max(big, key=lambda b: b[1])
    cx, cy = bx + bw//2, by + bh//2
    print(f"  Banker at ({cx},{cy}) area={area:.0f}")

    # Right-click the banker
    print(f"  Right-clicking banker...")
    click(cx, cy, True)
    time.sleep(0.8)

    # Find "Bank" option
    bgr = capture()
    needle = cv2.imread("data/screenshots/bank/bank_banker_option.png")
    found_menu = False
    if needle is not None:
        search = bgr[cy:cy+130, cx-120:cx+120]
        if search.shape[0] > 0 and search.shape[1] > 0:
            result = cv2.matchTemplate(search, needle, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(result)
            if conf > 0.5:
                mx = cx - 120 + loc[0] + needle.shape[1]//2
                my = cy + loc[1] + needle.shape[0]//2
                print(f"  'Bank' template at ({mx},{my}) conf={conf:.2f}")
                click(mx, my)
                found_menu = True

    if not found_menu:
        # OCR fallback
        import pytesseract
        menu_region = bgr[cy:cy+160, cx-140:cx+140]
        menu_gray = cv2.cvtColor(menu_region, cv2.COLOR_BGR2GRAY)
        menu_text = pytesseract.image_to_string(menu_gray).lower()
        print(f"  Menu OCR: {menu_text[:80]}")
        if 'bank' in menu_text:
            click(cx, cy + 55)
            found_menu = True

    if not found_menu:
        print("  ❌ 'Bank' option not found in menu")
        return False

    print("  Waiting for bank/PIN screen...")
    time.sleep(2)
    return True


# ---------------------------------------------------------------------------
# PIN Debug — interactive step-by-step
# ---------------------------------------------------------------------------

def detect_pin_digits(bgr, hsv, h, w):
    """
    Scan for PIN digit buttons using the two-stage approach:
      1. HSV red-square detection (wider window — w//6 not w//4)
      2. Template-match all 10 digits within each button crop
      3. Per-digit bias + margin check + uniqueness constraint

    Returns:
      resolved: dict[str, tuple[int, int, float]] — digit -> (cx, cy, adj_conf)
      all_buttons: list of button dicts for display
      rows: grid layout for display
    """
    DIGIT_BIAS = {7: -0.20, 1: -0.05, 3: -0.05}
    MIN_RAW_CONF = 0.5
    MARGIN_PCT = 3.0

    ox, oy = w // 6, h // 5
    crop_h1, crop_h2 = h // 5, 4 * h // 5
    crop_w1, crop_w2 = w // 6, 5 * w // 6

    # ---- Stage 1: Find red square buttons ----
    red = cv2.inRange(
        hsv[crop_h1:crop_h2, crop_w1:crop_w2],
        np.array([0, 150, 50]), np.array([12, 255, 160]),
    )
    red |= cv2.inRange(
        hsv[crop_h1:crop_h2, crop_w1:crop_w2],
        np.array([170, 150, 50]), np.array([180, 255, 160]),
    )
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    btns = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours
            if 200 < cv2.contourArea(c) < 50000
            and 0.6 < cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) < 1.7]
    btns.sort(key=lambda b: b[1], reverse=True)

    uniq = []
    for (bx, by, bw, bh), _ in btns:
        if not any(abs(bx - u[0]) < 15 and abs(by - u[1]) < 15 for u in uniq):
            uniq.append((bx, by, bw, bh))
            if len(uniq) >= 12:
                break

    print(f"  Found {len(uniq)} red button regions")

    # ---- Stage 2: Match all digits within each button crop ----
    all_buttons = []
    for bx, by, bw, bh in uniq:
        sx, sy = ox + bx, oy + by
        if sy + bh > h or sx + bw > w:
            continue
        crop = bgr[sy:sy + bh, sx:sx + bw]
        scores = {}
        for dig in range(10):
            n = cv2.imread(f"data/screenshots/pin/{dig}.png")
            if n is None or n.shape[0] > crop.shape[0] or n.shape[1] > crop.shape[1]:
                continue
            result = cv2.matchTemplate(crop, n, cv2.TM_CCOEFF_NORMED)
            _, raw_conf, _, _ = cv2.minMaxLoc(result)
            if raw_conf > MIN_RAW_CONF:
                adjusted = raw_conf + DIGIT_BIAS.get(dig, 0.0)
                scores[dig] = (raw_conf, adjusted)

        if not scores:
            continue

        best_dig = max(scores, key=lambda d: scores[d][1])
        best_adj = scores[best_dig][1]
        others = [(d, scores[d][1]) for d in scores if d != best_dig]
        runner_adj = max(others, key=lambda x: x[1])[1] if others else 0.0
        margin = (best_adj - runner_adj) / best_adj * 100 if best_adj > 0 else 0.0

        all_buttons.append({
            "pos": (sx + bw // 2, sy + bh // 2),
            "bx": bx, "by": by, "bw": bw, "bh": bh,
            "best_dig": best_dig,
            "best_adj": best_adj,
            "best_raw": scores[best_dig][0],
            "margin": margin,
            "scores": scores,
        })

    # ---- Stage 3: Uniqueness constraint ----
    all_buttons.sort(key=lambda b: b["best_adj"], reverse=True)
    assigned_digits: set[int] = set()
    resolved: dict[str, tuple[int, int, float]] = {}

    for btn in all_buttons:
        if btn["margin"] < MARGIN_PCT:
            continue
        best = btn["best_dig"]
        if best not in assigned_digits:
            assigned_digits.add(best)
            resolved[str(best)] = (*btn["pos"], btn["best_adj"])
        else:
            candidates = [(d, btn["scores"][d][1]) for d in btn["scores"]
                          if d not in assigned_digits]
            if not candidates:
                continue
            next_dig, next_conf = max(candidates, key=lambda x: x[1])
            others = [(d, btn["scores"][d][1]) for d in btn["scores"]
                      if d != next_dig and d not in assigned_digits]
            next_runner = max(others, key=lambda x: x[1])[1] if others else 0.0
            next_margin = (next_conf - next_runner) / next_conf * 100 if next_conf > 0 else 0.0
            if next_margin >= MARGIN_PCT:
                assigned_digits.add(next_dig)
                resolved[str(next_dig)] = (*btn["pos"], next_conf)

    # ---- Build grid for display ----
    uniq.sort(key=lambda b: (b[1], b[0]))
    rows = []
    cur_row, cur_y = [], None
    for bx, by, bw, bh in uniq:
        if cur_y is None or abs(by - cur_y) < 45:
            cur_row.append((bx, by, bw, bh))
            cur_y = by if cur_y is None else cur_y
        else:
            rows.append(sorted(cur_row, key=lambda b: b[0]))
            cur_row, cur_y = [(bx, by, bw, bh)], by
    if cur_row:
        rows.append(sorted(cur_row, key=lambda b: b[0]))

    return resolved, all_buttons, rows, ox, oy


def debug_pin():
    """Interactive PIN entry — user confirms each digit before clicking."""
    pin = os.getenv("OSRS_BANK_PIN", "2006")
    print("\n" + "="*60)
    print(f"STEP 3: PIN ENTRY — target PIN: {pin}")
    print("="*60)
    print("For each digit, I'll show what I detect. Press:")
    print("  Enter  = approve & click")
    print("  's'    = skip (re-scan without clicking)")
    print("  'q'    = quit")
    print()

    bgr = capture()
    h, w = bgr.shape[:2]

    for i, target_digit in enumerate(pin):
        print(f"\n--- Digit {i+1}/4: looking for '{target_digit}' ---")

        # Move mouse away before capture (hover changes button appearance)
        subprocess.run(["xdotool", "mousemove", "100", "100"], timeout=5)
        time.sleep(0.08)

        # Re-scan (critical: OSRS shuffles digit positions after each click)
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        resolved, all_buttons, rows, ox, oy = detect_pin_digits(bgr, hsv, h, w)

        # ---- Show grid layout ----
        print()
        print("  PIN pad layout:")
        for ri, row in enumerate(rows):
            row_str = ""
            for ci, (bx, by, bw, bh) in enumerate(row):
                sx, sy = ox + bx, oy + by
                # Find matching button data
                match = None
                for btn in all_buttons:
                    if abs(btn["bx"] - bx) < 5 and abs(btn["by"] - by) < 5:
                        match = btn
                        break
                if match:
                    d, c, m = match["best_dig"], match["best_raw"], match["margin"]
                    target = " ★★" if str(d) == target_digit else ""
                    ambig = "?" if m < 3.0 else ""
                    row_str += f"| {d}:{c:.2f} Δ{m:.0f}%{ambig}{target} "
                else:
                    row_str += "| ??? "
            print(f"  Row {ri}: {row_str}|")

        # ---- Check if target digit was resolved ----
        if target_digit in resolved:
            cx, cy, conf = resolved[target_digit]
            print(f"\n  >>> '{target_digit}' resolved at ({cx},{cy}) adj_conf={conf:.3f}")
            choice = input(f"  Click? [Enter=y / s=skip / q=quit]: ").strip().lower()
            if choice == 'q':
                return
            if choice == 's':
                print("  ⏭ Skipping — will re-scan")
                continue
            print(f"  👆 Clicking '{target_digit}' at ({cx},{cy})")
            click(cx, cy)
            print(f"  ⏳ Waiting for digit shuffle...")
            time.sleep(0.7)
        else:
            print(f"\n  ⚠ TARGET DIGIT '{target_digit}' NOT FOUND in resolved set!")
            print(f"  Resolved digits: {sorted(resolved.keys())}")
            choice = input("  [s=skip re-scan / q=quit]: ").strip().lower()
            if choice == 'q':
                return
            continue

    # After all 4 digits entered
    print("\n" + "="*60)
    print("All 4 digits entered! Waiting for bank to open...")
    print("="*60)
    time.sleep(2)

    # Verify
    import pytesseract
    bgr = capture()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).lower()
    if 'bank of' in text and 'enter your pin' not in text:
        print("  ✅ Bank open!")
    else:
        print(f"  ⚠ State uncertain. OCR text: {text[:100]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("PIN DEBUG TOOL")
    print("==============")
    print("Make sure OSRS/RuneLite is visible on monitor 1.")
    print("Press Ctrl+C at any time to abort.\n")

    # Step 1: Login
    if not do_login():
        print("Login failed — aborting")
        sys.exit(1)

    time.sleep(1)

    # Step 2: Open bank
    if not open_bank():
        print("Bank open failed — aborting")
        sys.exit(1)

    time.sleep(0.5)

    # Step 3: PIN entry (interactive)
    debug_pin()

    print("\nDone!")
