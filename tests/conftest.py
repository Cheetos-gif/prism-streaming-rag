import sys
from pathlib import Path

import pytest

# Add project root to sys.path so pytest can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """A TestClient over the real app, shared by the API-level tests."""
    from fastapi.testclient import TestClient

    from controller.main import app

    return TestClient(app)
