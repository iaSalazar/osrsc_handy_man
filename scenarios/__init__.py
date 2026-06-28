"""User workflow scenario scripts.

Scenarios compose interaction handlers (from interactions/) into
full task sequences.  They are state machines driven by screen‑space
visual perception.

For backward compatibility, re‑exports from interactions/ are provided.
New code should import directly from interactions/.
"""

# Backward-compat re-exports (old import paths still work)
from interactions.bank_interaction import BankInteraction, BankLocation, BANK_LOCATIONS
from interactions.login import LoginHandler
from interactions.logout import LogoutHandler
from interactions.pin_entry import PinEntryHandler
