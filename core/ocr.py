"""
OCR helper — reads text from screen to verify state and read dynamic content.

Used alongside template matching: templates say WHERE, OCR says WHAT.
Works on UI text even with 117 HD (menus, PIN pad, chat use standard fonts).
"""

import mss, cv2, numpy as np, pytesseract
from typing import Optional

class OCR:
    def __init__(self, monitor: int = 1):
        self._monitor = monitor

    def read_screen(self) -> tuple[str, list[dict]]:
        """Read all text on screen. Returns (full_text, word_list)."""
        with mss.MSS() as sct:
            raw = np.array(sct.grab(sct.monitors[self._monitor]))
        gray = cv2.cvtColor(cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip() and int(data['conf'][i]) > 25:
                words.append({
                    'word': data['text'][i].strip(),
                    'conf': int(data['conf'][i]),
                    'x': data['left'][i] + data['width'][i] // 2,
                    'y': data['top'][i] + data['height'][i] // 2,
                })
        return text.lower(), words

    def find_word(self, keyword: str) -> Optional[dict]:
        """Find a word on screen, return its position and confidence."""
        _, words = self.read_screen()
        for w in words:
            if keyword.lower() in w['word'].lower():
                return w
        # Fuzzy: check partial matches for OCR errors
        for w in words:
            if len(w['word']) >= 2 and w['word'][:2] in keyword.lower():
                return w
        return None

    def has_text(self, *keywords: str) -> bool:
        """Check if ANY keyword appears on screen."""
        text, _ = self.read_screen()
        return any(kw.lower() in text for kw in keywords)

    def find_in_region(self, x: int, y: int, w: int, h: int) -> str:
        """Read text from a specific screen region. Returns text."""
        with mss.MSS() as sct:
            raw = np.array(sct.grab(sct.monitors[self._monitor]))
        gray = cv2.cvtColor(cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
        region = gray[max(0,y):min(gray.shape[0],y+h), max(0,x):min(gray.shape[1],x+w)]
        return pytesseract.image_to_string(region).strip()

    def verify_state(self, template_name: str, *keywords: str) -> tuple[bool, str]:
        """Verify screen state using OCR keywords. Returns (is_match, text_found)."""
        text, _ = self.read_screen()
        for kw in keywords:
            if kw.lower() in text:
                return True, kw
        return False, text[:100] if text else "(nothing)"
