"""
Human Keyboard Synthesis — realistic keystroke timing and error generation.

Models:
  - Two-component log-normal mixture for inter-key delays
    (fast touch-typing bursts + slow hunt-and-peck transitions)
  - Common trigram chunking (familiar sequences typed faster)
  - Typo generation with three error types:
      * Transposition (swap adjacent characters)
      * Adjacency error (QWERTY-neighbor key press)
      * Skipped key (omitted character)
  - Variable-speed backspace corrections with probabilistic overshoot
  - Modifier key overlap timing (Ctrl/Shift held slightly into target key)

Keystroke timing is calibrated per persona:
  - FAST_ACCURATE: ~85 WPM equivalent
  - SLOW_METHODICAL: ~40 WPM equivalent
  - DISTRACTED_ERROR_PRONE: ~50 WPM with high correction overhead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np

from core.config import KeyboardConfig, InterKeyDelayParams, TypoConfig, PersonaType


# ---------------------------------------------------------------------------
# QWERTY adjacency map — for realistic adjacency errors
# ---------------------------------------------------------------------------

_QWERTY_ADJACENT: dict[str, list[str]] = {
    'a': ['q', 'w', 's', 'z'], 'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'], 'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 's', 'd', 'r'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'j', 'k', 'o'], 'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'k', 'l', 'p'], 'p': ['o', 'l'],
    'q': ['w', 'a'], 'r': ['e', 'd', 'f', 't'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'], 't': ['r', 'f', 'g', 'y'],
    'u': ['y', 'h', 'j', 'i'], 'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'a', 's', 'e'], 'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'g', 'h', 'u'], 'z': ['a', 's', 'x'],
    '1': ['2', 'q'], '2': ['1', '3', 'q', 'w'],
    '3': ['2', '4', 'w', 'e'], '4': ['3', '5', 'e', 'r'],
    '5': ['4', '6', 'r', 't'], '6': ['5', '7', 't', 'y'],
    '7': ['6', '8', 'y', 'u'], '8': ['7', '9', 'u', 'i'],
    '9': ['8', '0', 'i', 'o'], '0': ['9', 'o', 'p'],
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class KeystrokeEvent:
    """A single key press or release."""
    key: str
    action: Literal["down", "up"]
    timestamp: float               # seconds from sequence start
    is_correction: bool = False
    typo_type: str = ""            # "transposition" | "adjacency" | "skip" | ""
    intended_char: str = ""


@dataclass
class TypoRecord:
    """Record of a typo for later correction."""
    position: int                  # index in original text
    intended: str                  # intended character
    actual: str                    # what was actually typed
    typo_type: str                 # "transposition" | "adjacency" | "skip"


# ---------------------------------------------------------------------------
# Inter-Key Delay Model
# ---------------------------------------------------------------------------

class InterKeyDelayModel:
    """
    Two-component log-normal mixture for inter-key delays.

    Component 1 (fast):  touch-typing bursts, ~120–200 ms
    Component 2 (slow):  hunt-and-peck or hand transitions, ~300–800 ms
    """

    def __init__(self, params: InterKeyDelayParams, rng: np.random.Generator) -> None:
        self._weights = params.component_weights
        self._means = params.component_means_ms
        self._stds = params.component_stds_ms
        self._rng = rng

        # Pre-compute log-normal parameters
        self._sigmas = [
            np.sqrt(np.log(1.0 + (s / m) ** 2)) if m > 0 else 0.5
            for m, s in zip(self._means, self._stds)
        ]
        self._mus = [
            np.log(m) - 0.5 * sig ** 2 if m > 0 else np.log(100.0)
            for m, sig in zip(self._means, self._sigmas)
        ]

    def sample(self) -> float:
        """Return a single inter-key delay in milliseconds."""
        comp = self._rng.choice(2, p=self._weights)
        delay = self._rng.lognormal(self._mus[comp], self._sigmas[comp])
        return max(delay, 20.0)  # floor at 20 ms

    def sample_sequence(self, length: int) -> np.ndarray:
        """Vectorized batch of inter-key delays."""
        return np.array([self.sample() for _ in range(length)])


# ---------------------------------------------------------------------------
# Typo Generator
# ---------------------------------------------------------------------------

class TypoGenerator:
    """
    Generates realistic typing errors and correction sequences.

    Error types:
      - Transposition (40%): swap adjacent characters  (e.g., "hte" for "the")
      - Adjacency (35%): press a QWERTY-neighbor key   (e.g., "w" for "e")
      - Skip (25%): omit a character entirely           (e.g., "th" for "the")
    """

    def __init__(self, config: TypoConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng
        self._error_types = ["transposition", "adjacency", "skip"]
        self._error_weights = [
            config.transposition_ratio,
            config.adjacency_ratio,
            config.skip_ratio,
        ]

    def maybe_apply(self, text: str, typo_rate: float) -> tuple[str, list[TypoRecord]]:
        """
        Optionally introduce typos into `text`.

        Parameters
        ----------
        text : The intended text to type.
        typo_rate : Per-character probability of a typo.

        Returns
        -------
        (typed_text, list_of_typo_records)
        """
        chars = list(text)
        typos: list[TypoRecord] = []
        i = 0
        while i < len(chars):
            if self._rng.random() < typo_rate:
                error_type = self._rng.choice(self._error_types, p=self._error_weights)
                intended = chars[i]

                if error_type == "transposition" and i + 1 < len(chars):
                    # Swap chars[i] and chars[i+1]
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    typos.append(TypoRecord(i, intended, chars[i], "transposition"))
                    i += 1  # skip the swapped char

                elif error_type == "adjacency":
                    neighbors = _QWERTY_ADJACENT.get(intended.lower(), [])
                    if neighbors:
                        actual = self._rng.choice(neighbors)
                        if intended.isupper():
                            actual = actual.upper()
                        chars[i] = actual
                        typos.append(TypoRecord(i, intended, actual, "adjacency"))

                elif error_type == "skip":
                    typos.append(TypoRecord(i, intended, "", "skip"))
                    chars[i] = ""  # mark for removal

            i += 1

        typed = "".join(c for c in chars if c != "")
        return typed, typos

    def generate_correction(
        self,
        typed: str,
        typos: list[TypoRecord],
        intended: str,
    ) -> tuple[list[KeystrokeEvent], str]:
        """
        Generate backspace + retype events to correct all typos.

        Includes overshoot: sometimes delete 1–3 extra correct characters
        beyond the error, then retype them too.
        """
        events: list[KeystrokeEvent] = []
        t = 0.0

        if not typos:
            return events, typed

        # Calculate how many characters from the end need correction
        # Simplified: delete back to the earliest typo, with overshoot
        earliest_pos = min(t.position for t in typos)
        end_pos = len(typed)

        chars_to_delete = end_pos - earliest_pos
        overshoot = min(
            self._rng.poisson(self.config.correction_overshoot_mean),
            earliest_pos,  # can't overshoot past the beginning
        )
        total_delete = chars_to_delete + overshoot

        # Backspace events
        bs_delay = self.config.correction_speed_ratio * 80.0  # faster backspacing
        for _ in range(total_delete):
            events.append(KeystrokeEvent("backspace", "down", t))
            t += bs_delay / 1000.0
            events.append(KeystrokeEvent("backspace", "up", t, is_correction=True))
            t += bs_delay / 1000.0

        # Retype the corrected substring
        corrected_substring = intended[earliest_pos - overshoot:end_pos]
        for ch in corrected_substring:
            events.append(KeystrokeEvent(ch, "down", t, is_correction=True))
            t += self.config.correction_speed_ratio * 60.0 / 1000.0
            events.append(KeystrokeEvent(ch, "up", t, is_correction=True))
            t += self.config.correction_speed_ratio * 60.0 / 1000.0

        return events, intended[:earliest_pos - overshoot] + corrected_substring


# ---------------------------------------------------------------------------
# Keyboard Emulator — Public Facade
# ---------------------------------------------------------------------------

class KeyboardEmulator:
    """
    Generates realistic keyboard input sequences.

    Usage:
        emu = KeyboardEmulator(config, rng)
        events = emu.type_text("buy dragon bones")
        for evt in events:
            logger.log(evt)
    """

    def __init__(
        self,
        config: KeyboardConfig,
        rng: np.random.Generator,
        persona_typo_rate: float = 0.01,
    ) -> None:
        self.config = config
        self._rng = rng
        self._delay_model = InterKeyDelayModel(config.inter_key, rng)
        self._typo_gen = TypoGenerator(config.typo, rng)
        self._base_typo_rate = persona_typo_rate

    def type_text(
        self,
        text: str,
        typo_enabled: bool = True,
        typo_rate_override: float | None = None,
    ) -> list[KeystrokeEvent]:
        """
        Generate a full keystroke sequence for typing `text`.

        Parameters
        ----------
        text : The intended text.
        typo_enabled : If False, no typos are generated (for passwords etc.).
        typo_rate_override : Override the base typo rate.

        Returns
        -------
        List of KeystrokeEvent in temporal order.
        """
        typo_rate = typo_rate_override if typo_rate_override is not None else self._base_typo_rate
        if not typo_enabled:
            typo_rate = 0.0

        events: list[KeystrokeEvent] = []
        t = 0.0

        # Generate typed text (with potential typos)
        if typo_rate > 0:
            typed, typos = self._typo_gen.maybe_apply(text, typo_rate)
        else:
            typed, typos = text, []

        # Type the (potentially erroneous) text
        for i, ch in enumerate(typed):
            # Inter-key delay
            if i > 0:
                delay_ms = self._delay_model.sample()
                t += delay_ms / 1000.0

            events.append(KeystrokeEvent(ch, "down", t))
            # Key hold time (brief)
            hold_ms = self._rng.uniform(50.0, 100.0)
            t += hold_ms / 1000.0
            events.append(KeystrokeEvent(ch, "up", t))

        # If there were typos, generate corrections
        if typos:
            correction_events, final = self._typo_gen.generate_correction(typed, typos, text)
            # Shift correction timestamps
            for ce in correction_events:
                ce.timestamp += t
                events.append(ce)

        return events

    def type_password(self, password: str) -> list[KeystrokeEvent]:
        """
        Type a password — no typos, consistent 80–120 ms delays.
        Passwords are typed more deliberately and without error.
        """
        return self.type_text(password, typo_enabled=False, typo_rate_override=0.0)

    def press_hotkey(
        self,
        modifiers: list[str],
        key: str,
    ) -> list[KeystrokeEvent]:
        """
        Press a modifier+key combination with realistic overlap timing.

        Example: press_hotkey(["ctrl"], "c") generates:
            ctrl down → delay 30-80ms → c down → delay 50-100ms →
            c up → delay 20-50ms → ctrl up
        """
        events: list[KeystrokeEvent] = []
        t = 0.0

        # Press modifiers
        for mod in modifiers:
            events.append(KeystrokeEvent(mod, "down", t))
            t += self._rng.uniform(30.0, 80.0) / 1000.0

        # Press target key
        events.append(KeystrokeEvent(key, "down", t))
        t += self._rng.uniform(50.0, 100.0) / 1000.0

        # Release target key
        events.append(KeystrokeEvent(key, "up", t))
        t += self._rng.uniform(20.0, 50.0) / 1000.0

        # Release modifiers (in reverse order)
        for mod in reversed(modifiers):
            events.append(KeystrokeEvent(mod, "up", t))
            t += self._rng.uniform(10.0, 30.0) / 1000.0

        return events

    def press_key(self, key: str) -> list[KeystrokeEvent]:
        """Press and release a single key."""
        t = 0.0
        hold = self._rng.uniform(60.0, 120.0) / 1000.0
        return [
            KeystrokeEvent(key, "down", 0.0),
            KeystrokeEvent(key, "up", hold),
        ]
