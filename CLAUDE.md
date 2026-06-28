# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Synthetic Human Telemetry Emulator — generates realistic, labeled streams of mouse/keyboard input events for training behavioral anomaly detection models. All UI perception is **screen-space only** (HSV color matching + template matching + OCR). No memory reading, no client hooking. Target application: Old School RuneScape via RuneLite.

## Architecture

```
core/               ← Engine modules (no game-specific logic)
  config.py         ← Pydantic config with Field validation — all parameters live here
  logger.py         ← JSONL telemetry logger (buffered writes, 16 event types)
  mouse.py          ← FittsLaw → Tremor → SaccadeModel → ClickModel → HIDPacer → MouseEmulator
  cognitive.py      ← ExWaldRT → PersonaModel → PhaseManager → AffectiveHMM → CognitiveEngine
  scan.py           ← Template matching (primary) → HSV color match (fallback) → OCR
  keyboard.py       ← InterKeyDelayModel → TypoGenerator → KeyboardEmulator
  scheduler.py      ← CircadianModel → SessionMemory → SessionScheduler
  actuator.py       ← LiveActuator — bridges MouseEmulator to real mouse via pynput (+ monitor offset)

interactions/       ← Reusable UI handlers (template-matched, no hardcoded coords)
  login.py          ← Two-screen flow: disconnected → main login → Play Now
  logout.py         ← X button → red logout confirm
  bank_interaction.py ← Open any bank (booth/chest/NPC), close, multi-location
  pin_entry.py      ← Enter 4-digit bank PIN (reads OSRS_BANK_PIN from env)

scenarios/          ← Full workflow state machines
  chaos_altar_dragon_bones.py  ← BANKING → TRAVELING → OFFERING → RETURNING → IDLE

tools/
  auto_crop.py      ← Edge-detection region finder for building template library
  template_capture.py ← Register screenshots as templates (HSV mask gen, variant mgmt)
  live_test.py      ← Test individual templates against screenshots
  run_live.py       ← Live-mode test runner (login/bank/logout against real game)
  cli.py            ← Click CLI: telem-emu generate/capture/config

data/
  screenshots/      ← User-dropped PNGs organized by element (login/, bank/, pin/, chaos_altar/)
  templates/        ← Registered templates with template.yaml + variants/ + hsv_masks/
  human_priors/     ← YAML behavioral priors (reaction times, tremor, cognitive params)
```

## Two Operating Modes

**Synthetic mode** (bulk data generation): `screen=None` passed to `VisualScanner.find_element()`. Returns stored bounding boxes with Gaussian position noise. No screen capture, no mouse movement. Generates JSONL telemetry for detector training.

**Live mode** (real game): actual screen capture via `mss` + `pynput` for mouse/keyboard. `LiveActuator` applies monitor offsets for multi-monitor setups. Template matching runs against live screen frames.

## Anti-Ban Design

All timing/behavior decisions flow through `CognitiveEngine`. The actuator only executes — it never adds its own timing. Key anti‑ban layers:

| Layer | Module | What it controls |
|---|---|---|
| Motor | `mouse.py` | Fitts-law trajectories, tremor (8-12 + 20-40 Hz), Bezier curves, click dynamics, HID jitter |
| Cognitive | `cognitive.py` | Persona-modulated RT (ex-Wald), session phases (warmup→groove→fatigue), 5-state affective HMM, micro-breaks, real-world interruptions |
| Visual | `scan.py` | Distracter scans before target, hover dwells with micro-gestures, confidence-based re-orientation |
| Scenario | `interactions/` | Misclick probability, camera rotation, AFK simulation, variable interaction methods (left vs right click, tab vs click) |

**Persona modulation**: `fast_accurate` (low typo/error, fast RT), `slow_methodical` (deliberate, frequent camera checks), `distracted_error_prone` (high error rate, AFK-prone, amplified tremor). Persona affects every layer — RT distribution, tremor amplitude, typo rate, misclick probability, distracter scans, camera rotation frequency.

## Key Commands

```bash
# Development
source .venv/bin/activate
python3 -m pytest tests/ -q              # all tests (29)
python3 -m pytest tests/test_mouse.py -q  # single file

# Live testing (takes over mouse/keyboard — press Ctrl+C to abort)
python3 tools/run_live.py monitors         # list detected monitors
python3 tools/run_live.py login            # test login flow
python3 tools/run_live.py bank             # test bank + PIN
python3 tools/run_live.py logout           # test logout
python3 tools/run_live.py all              # full chain
python3 tools/run_live.py login --persona distracted_error_prone --monitor 1

# Template management
python3 -m tools.auto_crop detect screenshot.png --element ge_bank_booth
python3 -m tools.auto_crop batch data/screenshots/login/ --element login
python3 tools/live_test.py test-set login data/screenshots/login/screen.png
python3 tools/live_test.py explore screenshot.png  # test all templates against a screenshot
```

## Template Workflow

Templates are tight crops of UI elements. The bot uses `cv2.matchTemplate` (TM_CCOEFF_NORMED) to find them on any screen, regardless of position.

1. User takes full-screen PNG screenshots
2. User crops out individual UI elements (tight — just the button/booth/digit with 5-10px padding)
3. Drop crops in `data/screenshots/{element_type}/`
4. Run `auto_crop detect` or manually register via `cli.py capture register`
5. Registration auto-generates HSV masks from 5th-95th percentile color ranges
6. Run `live_test.py scan` to verify template matches screenshot
7. Templates stored in `data/templates/{element}/` with `template.yaml` + `variants/` + `hsv_masks/`

**Same element at different camera angles** → add multiple variants under the same template ID with different `conditions.yaw` values. The scanner selects the best-matching variant via `Template.resolve_best_variant(conditions)`.

**Template matching works across monitors/resolutions** as long as the game renders at the same pixel dimensions. 117 HD, stretched mode, etc. are fine — just be consistent.

## Module Dependency Order

```
config.py (no deps)
  → logger.py (no deps)
    → mouse.py (config + rng)
      → cognitive.py (config + rng + scipy)
        → scan.py (config + rng + cv2)
          → keyboard.py (config + rng)
            → scheduler.py (config + rng + memory)
              → interactions/ (all core modules)
                → scenarios/ (interactions + all core)
                  → actuator.py (mouse + keyboard + pynput)
                    → tools/run_live.py (actuator + all modules)
```

Import path: `from core.mouse import MouseEmulator`, `from interactions.login import LoginHandler`. Old `from scenarios.login import ...` still works via backward-compat re-exports.

## Credentials

Credentials in `.env` (gitignored). `.env.example` shows the format:
```bash
export OSRS_USERNAME="your_username"
export OSRS_PASSWORD="your_password"
export OSRS_BANK_PIN="2006"    # 4 digits
```

Login flow assumes username is cached by the game client — it only clicks "Play Now". Full typing mode (`login(quick=False)`) available when `OSRS_USERNAME`/`OSRS_PASSWORD` are set and credentials aren't cached.

## RNG Determinism

All randomness flows through a single `numpy.random.Generator` instance passed through the module tree. Seed it once for reproducible sessions. Never call `random.random()` or bare `np.random.*` — always use the shared `self._rng`.
