from __future__ import annotations

import ast
from pathlib import Path


METRICS_DIR = Path(__file__).resolve().parents[2] / "src" / "loraphymac" / "metrics"


def _imports_statistics_facade(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "loraphymac.metrics.statistics":
            return True
        if isinstance(node, ast.Import):
            if any(alias.name == "loraphymac.metrics.statistics" for alias in node.names):
                return True
    return False


def test_read_only_metric_modules_do_not_import_statistics_facade():
    family_modules = sorted(
        path
        for path in METRICS_DIR.glob("*.py")
        if path.name not in {"__init__.py", "statistics.py"}
    )

    violations = [path.name for path in family_modules if _imports_statistics_facade(path)]

    assert violations == []
