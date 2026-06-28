# Screenshot Directory

Drop your full-screen PNG screenshots here, organized by UI element.

## Directory Structure

```
data/screenshots/
├── login/           ← login screen screenshots go here
│   ├── login_screen_y0.png
│   ├── login_screen_y90.png
│   └── ...
├── bank/            ← bank booth/chest screenshots
│   ├── ge_bank_booth_y0_default_mid.png
│   ├── ge_bank_booth_y90_default_mid.png
│   └── ...
├── pin/             ← bank PIN screen screenshots
├── logout/          ← logout button screenshots
├── chaos_altar/     ← Chaos Altar area screenshots
└── ...
```

## Workflow

### 1. Take screenshots
Use RuneLite in Fixed mode, 1920×1080, default renderer.
Save as PNG in the appropriate subdirectory.

### 2. Auto-crop to detect elements
```bash
source .venv/bin/activate

# Single screenshot
python3 -m tools.auto_crop detect data/screenshots/login/login_screen_y0.png \
  --element login_existing_user_btn --yaw 0

# Batch a whole directory
python3 -m tools.auto_crop batch data/screenshots/login/ \
  --element login_screen_title
```

### 3. Review crops
Open `data/review/` — delete any wrong crops, keep the good ones.

### 4. Register templates
```bash
bash data/review/register.sh
```

### 5. Test in live mode
```bash
python3 tools/live_test.py login data/screenshots/login/login_screen_y0.png
```

Templates are now in `data/templates/{element}/` and ready for live‑mode scenarios.
