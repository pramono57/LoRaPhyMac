from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from loraphymac.hardware.hardware import SX1276, SX1302, apply_hw_ptx_bounds  # noqa: E402


def _entity_with_hw_ptx_bounds(entity_id, hardware):
    """Build a test entity using the same APC precondition as production."""
    entity = SimpleNamespace(id=entity_id, hw=hardware)
    apply_hw_ptx_bounds(entity)
    return entity


@pytest.fixture
def node_entity():
    return _entity_with_hw_ptx_bounds("N1", SX1276())


@pytest.fixture
def relay_entity():
    return _entity_with_hw_ptx_bounds("R1", SX1302())
