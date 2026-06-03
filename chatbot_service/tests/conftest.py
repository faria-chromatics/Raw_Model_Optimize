import os

# Must be set before any app imports so Settings() picks up mock mode
os.environ["MODEL_MODE"] = "mock"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
