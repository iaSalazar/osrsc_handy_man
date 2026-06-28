"""
Live-Mode Test Runner — proven methods for each interaction.

Usage:
    python3 tools/run_live.py login
    python3 tools/run_live.py bank
    python3 tools/run_live.py logout
    python3 tools/run_live.py close
    python3 tools/run_live.py all
    python3 tools/run_live.py monitors

IMPORTANT: Takes over mouse/keyboard. Press Ctrl+C to abort.
"""

import subprocess, time, mss, cv2, numpy as np, os, pytesseract, sys
from pathlib import Path

def click(x, y, right=False):
    subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y))], timeout=5)
    time.sleep(0.1)
    subprocess.run(["xdotool", "click", "3" if right else "1"], timeout=5)

def capture():
    with mss.MSS() as sct:
        raw = np.array(sct.grab(sct.monitors[1]))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

def ocr_text():
    gray = cv2.cvtColor(capture(), cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray).lower()

# ═══════════════════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# State detection helpers
# ------------------------------------------------------------------

def _detect_login_state(bgr, gray, hsv):
    """
    Determine which login screen we're on using OCR + template matching.

    Returns one of: 'disconnected', 'play_now', 'red_button', 'in_game', 'unknown'

    CRITICAL: 'in_game' is the TERMINAL state — only returned when we're
    certain NO login screen elements are present.  Brightness alone is NOT
    enough (the red-button login screen has a bright background image).
    """
    h, w = bgr.shape[:2]
    txt = pytesseract.image_to_string(gray).lower()

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
    disc_ocr = 'disconnected' in txt or 'you were' in txt

    disc_score = (disc_ocr, disc_tmpl_conf > 0.6, ok_conf > 0.7)
    if any(disc_score):
        return 'disconnected'

    # ---- Check: red "Click here to play" button ----
    red_tmpl = cv2.imread("data/screenshots/login/play_now_red_button.png")
    red_tmpl_conf = 0.0
    red_tmpl_loc = (0, 0)
    if red_tmpl is not None:
        r = cv2.matchTemplate(bgr, red_tmpl, cv2.TM_CCOEFF_NORMED)
        _, red_tmpl_conf, _, red_tmpl_loc = cv2.minMaxLoc(r)
    # Red blob in TIGHT centre region.  The login red button is:
    #   - large (>3000 px)
    #   - horizontally wide (aspect w/h between 1.8 and 5.0)
    #   - dead-centre of screen
    # Game-world red (armor, items, HP orb) fails aspect or position.
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
    red_ocr = 'click here to play' in txt or 'click here' in txt

    red_score = (red_ocr, red_tmpl_conf > 0.6, red_blob)
    if any(red_score):
        return 'red_button'

    # ---- Check: Play Now / Welcome screen ----
    welcome_tmpl = cv2.imread("data/screenshots/login/welcome_to_runescape.png")
    welcome_tmpl_conf = 0.0
    if welcome_tmpl is not None:
        r = cv2.matchTemplate(bgr, welcome_tmpl, cv2.TM_CCOEFF_NORMED)
        _, welcome_tmpl_conf, _, _ = cv2.minMaxLoc(r)
    play_tmpl = cv2.imread("data/screenshots/login/login_play_now_text.png")
    play_tmpl_conf = 0.0
    if play_tmpl is not None:
        r = cv2.matchTemplate(bgr, play_tmpl, cv2.TM_CCOEFF_NORMED)
        _, play_tmpl_conf, _, _ = cv2.minMaxLoc(r)
    welcome_ocr = 'welcome' in txt or 'play now' in txt

    play_score = (welcome_ocr, welcome_tmpl_conf > 0.6, play_tmpl_conf > 0.7)
    if any(play_score):
        return 'play_now'

    # ---- In-game: default terminal state ----
    # If NONE of the login screens matched above, we're in-game.
    # No brightness check — the game world varies from dark caves to bright
    # areas.  The ONLY reliable signal is the ABSENCE of login UI.
    return 'in_game'


def _wait_for_state(target_state, timeout_s=20.0, poll_s=2.0):
    """
    Poll until the login screen transitions to target_state.

    Waits an initial delay before polling — the game needs time to
    render the next screen after a click.
    """
    # Initial render wait — don't even check for the first 3 seconds
    time.sleep(3.0)

    start = time.time()
    while time.time() - start < timeout_s:
        time.sleep(poll_s)
        bgr = capture()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        current = _detect_login_state(bgr, gray, hsv)
        print(f"    [poll {time.time()-start:.0f}s] {current}")
        if current == target_state:
            return True
        # If we overshot to the next state, that's also fine
        if target_state == 'play_now' and current in ('red_button', 'in_game'):
            return True
        if target_state == 'red_button' and current == 'in_game':
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# LOGIN — 3-state flow with OCR + template validation
# ═══════════════════════════════════════════════════════════════

def login():
    print("=== LOGIN ===")

    for attempt in range(15):
        bgr = capture()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        state = _detect_login_state(bgr, gray, hsv)
        print(f"  State: {state} (attempt {attempt+1})")

        # ---- IN-GAME → done ----
        if state == 'in_game':
            print("  ✅ In-game — login complete")
            return

        # ---- DISCONNECTED → click OK ----
        if state == 'disconnected':
            ok_btn = cv2.imread("data/screenshots/login/disconnected_ok_button.png")
            if ok_btn is not None:
                r = cv2.matchTemplate(bgr, ok_btn, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                cx, cy = loc[0] + ok_btn.shape[1]//2, loc[1] + ok_btn.shape[0]//2
                print(f"  [1/3] DISCONNECTED — OK at ({cx},{cy}) conf={conf:.2f}")
                click(cx, cy)
                if _wait_for_state('play_now', timeout_s=12):
                    continue
                print("  ⚠ Did not transition to Play Now — re-scanning")
                continue

        # ---- PLAY NOW → click "Play Now" ----
        if state == 'play_now':
            play_tmpl = cv2.imread("data/screenshots/login/login_play_now_text.png")
            if play_tmpl is not None:
                r = cv2.matchTemplate(bgr, play_tmpl, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                cx, cy = loc[0] + play_tmpl.shape[1]//2, loc[1] + play_tmpl.shape[0]//2
                print(f"  [2/3] PLAY NOW — button at ({cx},{cy}) conf={conf:.2f}")
                click(cx, cy)
                if _wait_for_state('red_button', timeout_s=12):
                    continue
                print("  ⚠ Did not transition to red button — re-scanning")
                continue

        # ---- RED BUTTON → click it ----
        if state == 'red_button':
            red_tmpl = cv2.imread("data/screenshots/login/play_now_red_button.png")
            cx, cy = None, None
            if red_tmpl is not None:
                r = cv2.matchTemplate(bgr, red_tmpl, cv2.TM_CCOEFF_NORMED)
                _, conf, _, loc = cv2.minMaxLoc(r)
                if conf > 0.6:
                    cx, cy = loc[0] + red_tmpl.shape[1]//2, loc[1] + red_tmpl.shape[0]//2
                    print(f"  [3/3] RED BUTTON — at ({cx},{cy}) conf={conf:.2f}")
            if cx is None:
                # Fallback: HSV red blob
                h, w = bgr.shape[:2]
                crop = hsv[h//3:2*h//3, w//3:2*w//3]
                red_mask = cv2.inRange(crop, np.array([0,50,40]), np.array([20,255,255]))
                red_mask |= cv2.inRange(crop, np.array([170,50,40]), np.array([180,255,255]))
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 5000]
                if big:
                    (bx,by,bw,bh), _ = max(big, key=lambda b: b[1])
                    cx, cy = w//3 + bx + bw//2, h//3 + by + bh//2
                    print(f"  [3/3] RED BUTTON (HSV) — at ({cx},{cy})")
            if cx is not None:
                click(cx, cy)
                if _wait_for_state('in_game', timeout_s=15):
                    print("  ✅ In-game — login complete")
                    return
                print("  ⚠ Did not transition to in-game — re-scanning")
                continue

        # ---- UNKNOWN — wait and retry ----
        print(f"  Waiting... (state={state})")
        time.sleep(2)

    print("  ❌ Login failed")

# ═══════════════════════════════════════════════════════════════
# BANK OPEN + PIN + CLOSE
# ═══════════════════════════════════════════════════════════════

def bank():
    print("=== BANK ===")
    pin = os.getenv("OSRS_BANK_PIN", "2006")

    # ---- Step 0: Detect what state we're in ----
    text = ocr_text()
    bgr = capture()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Count red square buttons (PIN pad)
    red = cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([0,150,50]), np.array([12,255,160]))
    red |= cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([170,150,50]), np.array([180,255,160]))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq = sum(1 for c in contours if 200<cv2.contourArea(c)<50000 and 0.6<cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3]<1.7)

    bank_open = 'bank of' in text and 'enter your pin' not in text
    pin_open = 'enter your pin' in text or sq >= 7
    in_game = not bank_open and not pin_open

    print(f"  State: bank_open={bank_open} pin_open={pin_open} in_game={in_game} (sq={sq})")

    # ---- CASE 1: Bank already open (no PIN) → done ----
    if bank_open:
        print("  ✅ Bank already open")
        return

    # ---- CASE 2: PIN screen open → enter PIN first ----
    if pin_open:
        print(f"  🔢 PIN screen open — entering {pin}")
        _enter_pin(pin, bgr, hsv, h, w)
        # Verify transition
        time.sleep(1)
        text = ocr_text()
        if 'bank of' in text and 'enter your pin' not in text:
            print("  ✅ Bank open after PIN")
            return
        # PIN might have been wrong — retry once
        print("  ⚠ PIN may have failed — retrying")
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _enter_pin(pin, bgr, hsv, h, w)
        time.sleep(1)
        text = ocr_text()
        if 'bank of' in text and 'enter your pin' not in text:
            print("  ✅ Bank open after PIN retry")
            return

    # ---- CASE 3: In-game, no PIN → find banker and open bank ----
    if not bank_open and not pin_open:
        # Find magenta NPC
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        magenta = cv2.inRange(hsv, np.array([140,100,60]), np.array([165,255,255]))
        game = np.zeros_like(magenta); game[50:950,250:1700] = magenta[50:950,250:1700]
        contours, _ = cv2.findContours(game, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c)>200]

        if not big:
            print("  ❌ No banker")
            return
        (bx,by,bw,bh), _ = max(big, key=lambda b: b[1])
        cx, cy = bx+bw//2, by+bh//2
        print(f"  Banker at ({cx},{cy})")

        # Right-click
        click(cx, cy, True)
        time.sleep(0.8)

        # Find "Bank" option — try template first, then OCR
        bgr = capture()
        needle = cv2.imread("data/screenshots/bank/bank_banker_option.png")
        found_menu = False
        if needle is not None:
            search = bgr[cy:cy+130, cx-120:cx+120]
            result = cv2.matchTemplate(search, needle, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(result)
            if conf > 0.5:
                bx, by = cx-120+loc[0]+needle.shape[1]//2, cy+loc[1]+needle.shape[0]//2
                print(f"  'Bank' template at ({bx},{by}) conf={conf:.2f}")
                click(bx, by)
                found_menu = True

        if not found_menu:
            # OCR fallback: read menu text, find "Bank"
            menu_region = bgr[cy:cy+160, cx-140:cx+140]
            menu_gray = cv2.cvtColor(menu_region, cv2.COLOR_BGR2GRAY)
            menu_text = pytesseract.image_to_string(menu_gray).lower()
            print(f"  Menu OCR: {menu_text[:60]}")
            if 'bank' in menu_text:
                # Click approximate position — "Bank" is usually 3rd-5th option
                # Each menu row ~18px, first option starts ~10px below click
                click(cx, cy + 55)
                found_menu = True

        if not found_menu:
            print("  ❌ 'Bank' option not found")
            return

        time.sleep(2)

        # After clicking Bank, check if PIN screen appeared
        text = ocr_text()
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([0,150,50]), np.array([12,255,160]))
        red |= cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([170,150,50]), np.array([180,255,160]))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sq = sum(1 for c in contours if 200<cv2.contourArea(c)<50000 and 0.6<cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3]<1.7)

        if sq >= 7 or 'enter your pin' in text:
            print(f"  🔢 PIN screen appeared — entering {pin}")
            _enter_pin(pin, bgr, hsv, h, w)
            time.sleep(1)

    # ---- Final verification ----
    text = ocr_text()
    if 'bank of' in text and 'enter your pin' not in text:
        print("  ✅ Bank open!")
    else:
        print(f"  ⚠ Bank state uncertain: {text[:80]}")


def _enter_pin(pin_str, bgr, hsv, h, w):
    """Enter a PIN on the bank PIN screen. Re-scans after each digit.

    Uses a two-stage approach:
      1. HSV red-square detection to find all PIN buttons
      2. Template-match all 10 digit templates within each button crop
      3. Uniqueness constraint + per-digit bias + margin check

    Improvements over the old version:
      - Wider search window (w//6 instead of w//4) to capture left column
      - Mouse moved away before capture (hover changes button appearance)
      - Per-digit bias penalizes promiscuous templates (digit 7)
      - Margin requirement: best match must beat runner-up by >= 3%
      - Uniqueness: each digit 0-9 appears at most once per scan
    """
    # Per-digit bias — penalize templates that are false-positive magnets
    # Digit 7 matches everything at ~0.73; 1 and 3 also slightly promiscuous
    DIGIT_BIAS = {7: -0.20, 1: -0.05, 3: -0.05}
    MIN_RAW_CONF = 0.5
    MARGIN_PCT = 3.0

    # Wider search window — old w//4 cut off the left column of buttons
    ox, oy = w // 6, h // 5
    crop_h1, crop_h2 = h // 5, 4 * h // 5
    crop_w1, crop_w2 = w // 6, 5 * w // 6

    for digit in pin_str:
        # ---- Move mouse away before capture ----
        # Hovering over a PIN button hides its digit (OSRS highlight effect),
        # which corrupts the template match. Park the cursor in a safe corner.
        subprocess.run(["xdotool", "mousemove", "100", "100"], timeout=5)
        time.sleep(0.08)

        # ---- Re-scan after each click (buttons rearrange) ----
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ---- Stage 1: Find all red square buttons via HSV ----
        red = cv2.inRange(
            hsv[crop_h1:crop_h2, crop_w1:crop_w2],
            np.array([0, 150, 50]), np.array([12, 255, 160]),
        )
        red |= cv2.inRange(
            hsv[crop_h1:crop_h2, crop_w1:crop_w2],
            np.array([170, 150, 50]), np.array([180, 255, 160]),
        )
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        btns = [
            (cv2.boundingRect(c), cv2.contourArea(c))
            for c in contours
            if 200 < cv2.contourArea(c) < 50000
            and 0.6 < cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) < 1.7
        ]
        btns.sort(key=lambda b: b[1], reverse=True)
        uniq = []
        for (bx, by, bw, bh), _ in btns:
            if not any(abs(bx - u[0]) < 15 and abs(by - u[1]) < 15 for u in uniq):
                uniq.append((bx, by, bw, bh))
                if len(uniq) >= 12:
                    break

        # ---- Stage 2: Match all 10 digit templates within each button crop ----
        # Each button gets scored against every digit; we track the best match
        # AND the runner-up so we can compute a confidence margin.
        all_buttons = []  # list of dicts with pos, best_dig, margin, scores
        for bx, by, bw, bh in uniq:
            sx, sy = ox + bx, oy + by
            if sy + bh > h or sx + bw > w:
                continue
            crop = bgr[sy:sy + bh, sx:sx + bw]
            scores = {}  # dig (int) -> (raw_conf, adjusted_conf)
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

            # Best digit for this button (by adjusted confidence)
            best_dig = max(scores, key=lambda d: scores[d][1])
            best_adj = scores[best_dig][1]
            # Runner-up (excluding the best digit)
            others = [(d, scores[d][1]) for d in scores if d != best_dig]
            runner_adj = max(others, key=lambda x: x[1])[1] if others else 0.0
            margin = (best_adj - runner_adj) / best_adj * 100 if best_adj > 0 else 0.0

            all_buttons.append({
                "pos": (sx + bw // 2, sy + bh // 2),
                "best_dig": best_dig,
                "best_adj": best_adj,
                "margin": margin,
                "scores": scores,
            })

        # ---- Stage 3: Uniqueness constraint ----
        # Each digit 0-9 appears exactly once on the PIN pad. Process buttons
        # from highest to lowest confidence. If a button's best digit is already
        # assigned, fall back to its next-best (unassigned) digit.
        all_buttons.sort(key=lambda b: b["best_adj"], reverse=True)
        assigned_digits: set[int] = set()
        resolved: dict[str, tuple[int, int, float]] = {}  # digit_str -> (cx, cy, adj_conf)

        for btn in all_buttons:
            if btn["margin"] < MARGIN_PCT:
                continue  # too ambiguous

            best = btn["best_dig"]
            if best not in assigned_digits:
                assigned_digits.add(best)
                resolved[str(best)] = (*btn["pos"], btn["best_adj"])
            else:
                # Duplicate — this button's best digit is already claimed.
                # Try next-best unassigned digit with adequate margin.
                candidates = [
                    (d, btn["scores"][d][1])
                    for d in btn["scores"]
                    if d not in assigned_digits
                ]
                if not candidates:
                    continue
                next_dig, next_conf = max(candidates, key=lambda x: x[1])
                # Check margin for this next-best candidate
                others = [
                    (d, btn["scores"][d][1])
                    for d in btn["scores"]
                    if d != next_dig and d not in assigned_digits
                ]
                next_runner = max(others, key=lambda x: x[1])[1] if others else 0.0
                next_margin = (
                    (next_conf - next_runner) / next_conf * 100
                    if next_conf > 0 else 0.0
                )
                if next_margin >= MARGIN_PCT:
                    assigned_digits.add(next_dig)
                    resolved[str(next_dig)] = (*btn["pos"], next_conf)

        # ---- Stage 4: Click the target digit ----
        if digit in resolved:
            cx, cy, conf = resolved[digit]
            print(f"    {digit} → ({cx},{cy}) adj_conf={conf:.2f}")
            click(cx, cy)
            time.sleep(0.6)
        else:
            print(f"    ❌ Digit {digit} not found — resolved: {sorted(resolved.keys())}")

# ═══════════════════════════════════════════════════════════════
# CLOSE BANK
# ═══════════════════════════════════════════════════════════════

def close():
    print("=== CLOSE ===")
    bgr = capture()
    close = cv2.imread("data/screenshots/bank/bank_close_buttom.png")
    r = cv2.matchTemplate(bgr, close, cv2.TM_CCOEFF_NORMED)
    _, conf, _, loc = cv2.minMaxLoc(r)
    if conf > 0.4:
        cx, cy = loc[0]+close.shape[1]//2, loc[1]+close.shape[0]//2
        print(f"  Close at ({cx},{cy}) conf={conf:.2f}")
        click(cx, cy)
        print("  ✅ Closed!")
    else:
        print(f"  Not found ({conf:.2f})")

# ═══════════════════════════════════════════════════════════════
# LOGOUT
# ═══════════════════════════════════════════════════════════════

def logout():
    print("=== LOGOUT ===")

    # Step 1: Click X
    bgr = capture()
    x_img = cv2.imread("data/screenshots/logout/logout_cross.png")
    r = cv2.matchTemplate(bgr, x_img, cv2.TM_CCOEFF_NORMED)
    _, c, _, l = cv2.minMaxLoc(r)
    cx, cy = l[0]+x_img.shape[1]//2, l[1]+x_img.shape[0]//2
    print(f"  X at ({cx},{cy}) conf={c:.2f}")
    if c > 0.4: click(cx, cy)
    time.sleep(0.8)

    # Step 2: Click red logout
    bgr = capture()
    red = cv2.imread("data/screenshots/logout/logout_buttom.png")
    r = cv2.matchTemplate(bgr, red, cv2.TM_CCOEFF_NORMED)
    _, c, _, l = cv2.minMaxLoc(r)
    cx, cy = l[0]+red.shape[1]//2, l[1]+red.shape[0]//2
    print(f"  Red at ({cx},{cy}) conf={c:.2f}")
    if c > 0.4: click(cx, cy)
    print("  ✅ Logged out!")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "monitors":
        from core.actuator import LiveActuator
        for m in LiveActuator.list_monitors():
            print(f"  Monitor {m['index']}: {m['width']}x{m['height']} at ({m['left']},{m['top']}) — {m['description']}")
    elif cmd == "login":
        login()
    elif cmd == "bank":
        bank()
    elif cmd == "close":
        close()
    elif cmd == "logout":
        logout()
    elif cmd == "all":
        login()
        time.sleep(2)
        bank()
        time.sleep(2)
        close()
        time.sleep(1)
        logout()
    else:
        print(f"Unknown: {cmd}")
