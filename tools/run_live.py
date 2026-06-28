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

def login():
    print("=== LOGIN ===")

    for attempt in range(8):
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]

        # Check for red button first (state 2)
        crop = hsv[h//3:2*h//3, w//3:2*w//3]
        red = cv2.inRange(crop, np.array([0,50,40]), np.array([20,255,255]))
        red |= cv2.inRange(crop, np.array([170,50,40]), np.array([180,255,255]))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 5000]

        if big:
            (bx,by,bw,bh), _ = max(big, key=lambda b: b[1])
            cx, cy = w//3 + bx + bw//2, h//3 + by + bh//2
            print(f"  Red button at ({cx},{cy}) — clicking")
            click(cx, cy)
            time.sleep(5)
            continue

        # Check for Play Now text (state 1)
        play = cv2.imread("data/screenshots/login/login_play_now_text.png")
        r = cv2.matchTemplate(bgr, play, cv2.TM_CCOEFF_NORMED)
        _, pc, _, pl = cv2.minMaxLoc(r)

        if pc > 0.7:
            px, py = pl[0]+play.shape[1]//2, pl[1]+play.shape[0]//2
            print(f"  Play Now at ({px},{py}) conf={pc:.2f} — clicking")
            click(px, py)
            time.sleep(5)
            continue

        # Check if in-game
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bright = gray[h//3:2*h//3, w//3:2*w//3].mean()
        if bright > 70 and pc < 0.5:
            print(f"  ✅ In-game (bright={bright:.0f})")
            return

        print(f"  Waiting... ({attempt+1})")
        time.sleep(3)

    print("  ❌ Login failed")

# ═══════════════════════════════════════════════════════════════
# BANK OPEN + PIN + CLOSE
# ═══════════════════════════════════════════════════════════════

def bank():
    print("=== BANK ===")
    pin = os.getenv("OSRS_BANK_PIN", "2006")

    # Check if already open
    text = ocr_text()
    if 'bank of' in text and 'enter your pin' not in text:
        print("  Already open")
    else:
        # Find magenta NPC
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        magenta = cv2.inRange(hsv, np.array([140,100,60]), np.array([165,255,255]))
        game = np.zeros_like(magenta); game[50:950,250:1700] = magenta[50:950,250:1700]
        contours, _ = cv2.findContours(game, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c)>200]

        if not big: print("  ❌ No banker"); return
        (bx,by,bw,bh), _ = max(big, key=lambda b: b[1])
        cx, cy = bx+bw//2, by+bh//2
        print(f"  Banker at ({cx},{cy})")

        # Right-click
        click(cx, cy, True); time.sleep(0.8)

        # Find "Bank" via template
        bgr = capture()
        needle = cv2.imread("data/screenshots/bank/bank_banker_option.png")
        search = bgr[cy:cy+130, cx-120:cx+120]
        result = cv2.matchTemplate(search, needle, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(result)

        if conf > 0.5:
            bx, by = cx-120+loc[0]+needle.shape[1]//2, cy+loc[1]+needle.shape[0]//2
            print(f"  'Bank' at ({bx},{by}) conf={conf:.2f}")
            click(bx, by)
            time.sleep(2)
        else:
            print(f"  'Bank' not found ({conf:.2f})"); return

    # Handle PIN if needed
    for pin_attempt in range(2):
        time.sleep(1)
        text = ocr_text()
        bgr = capture()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]

        # Count red PIN buttons
        red = cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([0,150,50]), np.array([12,255,160]))
        red |= cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([170,150,50]), np.array([180,255,160]))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sq = sum(1 for c in contours if 200<cv2.contourArea(c)<50000 and 0.6<cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3]<1.7)

        has_pin = sq >= 7 or 'enter your pin' in text

        if has_pin:
            print(f"  🔢 PIN ({sq} btns) — entering {pin}")
            for d in pin:
                # Re-scan after each click
                bgr = capture()
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                red = cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([0,150,50]), np.array([12,255,160]))
                red |= cv2.inRange(hsv[h//4:3*h//4, w//4:3*w//4], np.array([170,150,50]), np.array([180,255,160]))
                contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                btns = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours
                        if 200<cv2.contourArea(c)<50000 and 0.6<cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3]<1.7]
                btns.sort(key=lambda b: b[1], reverse=True)
                uniq = []
                for (bx,by,bw,bh), _ in btns:
                    if not any(abs(bx-u[0])<15 and abs(by-u[1])<15 for u in uniq):
                        uniq.append((bx,by,bw,bh))
                        if len(uniq)>=12: break

                dmap = {}
                for bx, by, bw, bh in uniq:
                    crop = bgr[by:by+bh, bx:bx+bw]
                    scores = []
                    for dig in range(10):
                        n = cv2.imread(f"data/screenshots/pin/{dig}.png")
                        if n is None or n.shape[0]>crop.shape[0]: continue
                        _, c, _, _ = cv2.minMaxLoc(cv2.matchTemplate(crop, n, cv2.TM_CCOEFF_NORMED))
                        scores.append((dig, c))
                    if scores:
                        best = max(scores, key=lambda x: x[1])
                        dmap[str(best[0])] = (bx+bw//2, by+bh//2)

                if d in dmap:
                    print(f"    {d} → ({dmap[d][0]},{dmap[d][1]})")
                    click(dmap[d][0], dmap[d][1])
                    time.sleep(0.6)
            time.sleep(1)
        else:
            break

    # Verify state
    text = ocr_text()
    if 'bank of' in text and 'enter your pin' not in text:
        print("  ✅ Bank open!")

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
