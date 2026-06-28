"""
Reusable interaction handlers — login, logout, bank, PIN entry.

Each module handles one type of UI interaction using template‑matched
screen‑space perception.  Scenarios import these and compose them into
full workflows.

Usage:
    from interactions.login import LoginHandler
    from interactions.logout import LogoutHandler
    from interactions.bank_interaction import BankInteraction
    from interactions.pin_entry import PinEntryHandler
"""

from interactions.bank_interaction import BankInteraction, BankLocation, BANK_LOCATIONS
from interactions.login import LoginHandler
from interactions.logout import LogoutHandler
from interactions.pin_entry import PinEntryHandler

__all__ = [
    "BankInteraction", "BankLocation", "BANK_LOCATIONS",
    "LoginHandler", "LogoutHandler", "PinEntryHandler",
]
