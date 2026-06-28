# Screenshot Capture Plan — OSRS Template Library

## Quick Start

**Total screenshots needed (MVP):** ~60-80  
**Total screenshots for robust matching:** ~120-150  
**Capture time estimate:** 20-30 minutes per session  

**Capture settings:**
- RuneLite **Fixed mode** (classic), 1920×1080
- GPU plugin / 117 HD **disabled** (default renderer)
- Minimal plugins (just the basics)
- PNG format, lossless

**Naming convention:**
```
{element}_{state}_y{yaw}_{pitch}_{zoom}.png
```
Example: `ge_bank_booth_default_y0_default_mid.png`

---

## Template Capture Matrix

### For each element, capture these camera variants:

| Parameter | Values | When to use |
|-----------|--------|-------------|
| **Yaw** | 0° (N), 90° (E), 180° (S), 270° (W) | Rotate camera with middle-mouse drag |
| **Pitch** | "high" (bird's eye), "default" (~45°) | Scroll wheel up/down |
| **Zoom** | "mid" (default zoom), "out" (zoomed out) | Scroll wheel |

**Base set per element:** 4 yaw × 2 pitch × 2 zoom = 16 variants  
**Reduced set** (simple UI elements): 2 yaw × 1 pitch × 1 zoom = 2 variants  

---

## Phase 1: Login Screen (Templates: 6, Screenshots: ~28)

### 1.1 `login_screen_title`
**What:** The "Old School RuneScape" title text, or the "Welcome to Old School RuneScape" banner.  
**Purpose:** Confirms we're on the login screen.  
**Variants:** 4 yaw × 1 pitch × 1 zoom = **4 screenshots**  
*(The login screen doesn't rotate — just capture at different yaw settings)*

### 1.2 `login_existing_user_btn`
**What:** The "Existing User" button (orange/brown).  
**Purpose:** Click to switch to username/password entry mode.  
**States:** `default` (visible, not hovered)  
**Variants:** 2 yaw × 1 pitch × 1 zoom = **2 screenshots**

### 1.3 `login_username_field`
**What:** The username/email input field (white box).  
**Purpose:** Locate where to click before typing username.  
**Variants:** 2 yaw × 1 pitch × 1 zoom = **2 screenshots**

### 1.4 `login_password_field`
**What:** The password input field (white box, below username).  
**Purpose:** Locate where to click before typing password.  
**Variants:** 2 yaw × 1 pitch × 1 zoom = **2 screenshots**

### 1.5 `login_click_to_play_btn`
**What:** The red "Click here to play" button.  
**Purpose:** Submit login credentials.  
**States:** `default` (visible, not hovered)  
**Variants:** 2 yaw × 1 pitch × 1 zoom = **2 screenshots**

### 1.6 `game_view_minimap`
**What:** The minimap in the top-right corner of the game view.  
**Purpose:** Confirms we're in-game after login.  
**Variants:** 4 yaw × 2 pitch × 2 zoom = **16 screenshots**  
*(This is shared with other scenarios)*

**Login subtotal: ~28 screenshots**

---

## Phase 2: Grand Exchange Bank (Templates: 3, Screenshots: ~24)

### 2.1 `ge_bank_booth`
**What:** The GE bank booth — the brown/tan counter window with bars.  
**Include:** The entire clickable area of the booth.  
**States:** `default` (closed, not hovered)  
**Variants:** 4 yaw × 2 pitch × 2 zoom = **16 screenshots**  
**Tip:** Stand ~3-5 tiles away so the full booth is visible.

### 2.2 `bank_interface_open`
**What:** Any distinctive element only visible when the bank interface is open.  
**Best options:**
- The "Bank of Gielinor" title text
- The "Deposit inventory" button icon
- The bank item grid background pattern
- The quantity selector buttons  
**States:** `open`  
**Variants:** 4 yaw × 1 pitch × 1 zoom = **4 screenshots**

### 2.3 `bank_close_button`
**What:** The red X button in the top-right corner of the bank interface.  
**States:** `visible`  
**Variants:** 4 yaw × 1 pitch × 1 zoom = **4 screenshots**

**GE Bank subtotal: ~24 screenshots**

---

## Phase 3: Logout (Templates: 3, Screenshots: ~18)

### 3.1 `logout_door_icon`
**What:** The small door icon near the minimap (top-right of game view).  
**Purpose:** The primary logout button.  
**Variants:** 4 yaw × 2 pitch × 2 zoom = **16 screenshots**  
**Tip:** This is a small element — make sure it's clearly cropped.

### 3.2 `logout_tab_button`
**What:** The logout tab icon (world switcher / logout panel button).  
**Purpose:** Alternative logout method.  
**Variants:** 2 yaw × 1 pitch × 1 zoom = **2 screenshots**  
*(Reduced set — tab icons don't change much with angle)*

### 3.3 `logout_confirm_button`
**What:** The "Click here to logout" text/button in the logout panel.  
**Variants:** 1 yaw × 1 pitch × 1 zoom = **1 screenshot**  
*(This is a UI element — camera angle irrelevant)*

**Logout subtotal: ~19 screenshots**

---

## Phase 4: Bank PIN Screen (Templates: 12, Screenshots: ~24)

### 4.1 `bank_pin_screen`
**What:** The PIN entry interface — the panel with the "Bank PIN" title and number pad.  
**Purpose:** Detects when the PIN screen is visible.  
**Variants:** 1 yaw × 1 pitch × 1 zoom = **1 screenshot**  
*(The PIN screen is a fixed UI overlay — angle doesn't matter)*

### 4.2–4.11 `bank_pin_digit_0` through `bank_pin_digit_9`
**What:** Each individual digit button (0–9) on the PIN pad.  
**Important:** The OSRS PIN pad randomizes digit positions each time.  
**Strategy:** Capture each digit **in 2 different positions** (open PIN screen twice, note where digits moved).  
**Variants:** 10 digits × 2 positions × 1 angle = **20 screenshots**

### 4.12 `bank_interface_open` (reuse from Phase 2)
Already captured — confirms PIN was accepted → bank is open.

**PIN subtotal: ~21 screenshots**

---

## Phase 5: Chaos Altar Area (Templates: 12, Screenshots: ~60)

### 5.1 `chaos_altar_distant`
**What:** The Chaos Altar building visible from 20+ tiles away.  
**Purpose:** Initial approach — locating the altar on screen while traveling.  
**Variants:** 4 yaw × 1 pitch × 2 zoom = **8 screenshots**

### 5.2 `chaos_altar_medium`
**What:** The altar at 5–20 tiles — clearly visible on screen.  
**Variants:** 4 yaw × 2 pitch × 2 zoom = **16 screenshots**

### 5.3 `chaos_altar_close`
**What:** The altar at <5 tiles — clickable range.  
**Variants:** 4 yaw × 2 pitch × 2 zoom = **16 screenshots**

### 5.4 `wilderness_gate`
**What:** The wilderness gate/door (closed state).  
**Variants:** 4 yaw × 1 pitch × 1 zoom = **4 screenshots**

### 5.5 `wilderness_gate_open`
**What:** The gate in open state.  
**Variants:** 4 yaw × 1 pitch × 1 zoom = **4 screenshots**

### 5.6 `dragon_bones_inventory`
**What:** A dragon bones item in the inventory (noted or unnoted).  
**Variants:** 2 zoom × 2 yaw = **4 screenshots**

### 5.7 `dragon_bones_bank`
**What:** Dragon bones item in the bank interface.  
**Variants:** 2 positions × 2 yaw = **4 screenshots**

### 5.8 `offering_animation`
**What:** The altar during the offering animation (character kneeling/offering).  
**Capture:** Mid-animation frame.  
**Variants:** 2 angles × 1 zoom = **2 screenshots**

### 5.9 `prayer_xp_drop`
**What:** The XP drop text that appears when a bone is offered.  
**Variants:** 1 angle = **1 screenshot**

### 5.10 `teleport_item`
**What:** The teleport item in inventory (ring of dueling, glory, etc.).  
**Variants:** 2 zoom = **2 screenshots**

### 5.11 `run_energy_orb`
**What:** The run energy orb next to the minimap.  
**Variants:** 4 yaw × 2 pitch = **8 screenshots**

### 5.12 `bank_chest` (Edgeville/Chaos Altar area)
If using a different bank than GE. Same pattern as `ge_bank_booth`.

**Chaos Altar subtotal: ~69 screenshots**

---

## Directory Structure After Capture

```
data/templates/
├── login_screen_title/
│   ├── template.yaml
│   ├── variants/
│   │   ├── login_screen_title_default_y0_default_mid.png
│   │   ├── login_screen_title_default_y90_default_mid.png
│   │   └── ...
│   └── hsv_masks/
│       └── ...
├── login_existing_user_btn/
│   ├── template.yaml
│   └── variants/
├── login_username_field/
├── login_password_field/
├── login_click_to_play_btn/
├── game_view_minimap/
├── ge_bank_booth/
│   ├── template.yaml
│   └── variants/
│       ├── ge_bank_booth_default_y0_default_mid.png
│       ├── ge_bank_booth_default_y0_default_out.png
│       ├── ge_bank_booth_default_y0_high_mid.png
│       ├── ge_bank_booth_default_y0_high_out.png
│       ├── ge_bank_booth_default_y90_default_mid.png
│       └── ... (16 variants total)
├── bank_interface_open/
├── bank_close_button/
├── logout_door_icon/
├── logout_tab_button/
├── logout_confirm_button/
├── bank_pin_screen/
├── bank_pin_digit_0/
├── bank_pin_digit_1/
│   └── ... (0–9)
├── chaos_altar_distant/
├── chaos_altar_medium/
├── chaos_altar_close/
├── wilderness_gate/
├── wilderness_gate_open/
├── dragon_bones_inventory/
├── dragon_bones_bank/
├── offering_animation/
├── prayer_xp_drop/
├── teleport_item/
└── run_energy_orb/
```

---

## Registering Templates

After capturing each screenshot, register it with:

```bash
source .venv/bin/activate

# Example: register GE bank booth at yaw 0 (North), default pitch, mid zoom
python3 -m tools.cli capture register ge_bank_booth \
  ~/screenshots/ge_bank_booth_default_y0_default_mid.png \
  --state default --yaw 0 --pitch default --zoom mid \
  --bbox 450,320,60,50

# Example: register PIN digit 5
python3 -m tools.cli capture register bank_pin_digit_5 \
  ~/screenshots/pin_digit_5_pos1.png \
  --state visible --yaw 0 --pitch default --zoom mid \
  --bbox 800,400,35,35
```

The `--bbox` is `x,y,width,height` of the element in the screenshot.  
Use an image viewer (GIMP, eog, etc.) to find the pixel coordinates.

---

## Quick-Start Capture Order

1. **Session 1 — Login + GE Bank (45 min)**
   - Stand at GE bank booth area
   - Capture `ge_bank_booth` at all 16 variants
   - Open bank, capture `bank_interface_open` and `bank_close_button`
   - Log out, capture login screen elements
   - Log back in, capture `game_view_minimap`

2. **Session 2 — PIN + Logout (30 min)**
   - Open bank to trigger PIN screen
   - Capture `bank_pin_screen` and all 10 digits (reopen PIN for 2nd position)
   - Enter PIN, capture `bank_interface_open` again (post-PIN variant)
   - Capture `logout_door_icon` at multiple angles
   - Capture `logout_tab_button` and `logout_confirm_button`

3. **Session 3 — Chaos Altar (45 min)**
   - Travel to Chaos Altar area
   - Capture altar at distant, medium, close range at all camera angles
   - Capture wilderness gate (open + closed)
   - Capture dragon bones in inventory and bank
   - Capture offering animation and XP drop

---

## Total Screenshot Count Summary

| Phase | Elements | Min Variants | Screenshots |
|-------|----------|-------------|-------------|
| Login | 6 | 2-16 | ~28 |
| GE Bank | 3 | 4-16 | ~24 |
| Logout | 3 | 1-16 | ~19 |
| PIN Entry | 12 | 1-20 | ~21 |
| Chaos Altar | 12 | 1-16 | ~69 |

**Total MVP: ~161 screenshots**  
**Relaxed minimum (just login + bank + PIN): ~73 screenshots**

---

## Notes

- **HSV masks** are auto-generated when you register a template — you don't need to capture separate mask images.
- **Same element, different location:** If you want to use the same bank at Edgeville AND GE, capture `bank_booth` at both locations (or use separate template IDs like `ge_bank_booth` and `edgeville_bank_booth`).
- **Lighting:** OSRS doesn't have dynamic lighting, but some areas have different ambient colors. The `NoiseInjector` in `scan.py` handles subtle variations. If an area has significantly different coloring (e.g., dark cave vs sunny field), capture separate variants.
- **RuneLite themes:** If you use the Resource Packs plugin or custom themes, capture templates with your actual theme. The HSV ranges will be different for each theme.
