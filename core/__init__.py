"""
Synthetic Human Telemetry Emulator — core engine modules.

All public APIs exposed through this package for clean imports.
Imports are lazy to avoid circular dependency issues during testing.
"""

__version__ = "0.1.0"

# Config (no dependencies)
from core.config import (
    AppConfig, MouseConfig, ScanConfig, CognitiveConfig,
    KeyboardConfig, SchedulerConfig, ScenarioSpecificConfig,
    PersonaType, SessionPhase, AffectiveState, ScenarioState,
    DEFAULT_PERSONAS,
)

# Foundation modules
from core.logger import TelemetryLogger, TelemetryEvent, EventType

# Motor & cognitive (import order: mouse before cognitive for type refs)
from core.mouse import MouseEmulator, MouseSample, ClickEvent, FittsLaw
from core.cognitive import CognitiveEngine, Interruption
from core.scan import VisualScanner, Template, ScanResult, ScanMethod
from core.keyboard import KeyboardEmulator, KeystrokeEvent
from core.scheduler import (
    SessionScheduler, SessionConfig, SessionMemory,
    SessionRecord, CircadianModel,
)
