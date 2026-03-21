from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.append(str(path))
