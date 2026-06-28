"""Shared test fixtures."""

import numpy as np
import pytest
from pathlib import Path
from tempfile import mkdtemp


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def temp_dir() -> Path:
    """Temporary directory cleaned after test."""
    import shutil
    path = Path(mkdtemp(prefix="telem_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def project_root() -> Path:
    """Path to the project root directory."""
    return Path(__file__).parent.parent
