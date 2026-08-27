#!/usr/bin/env python3
"""Pipeline B online runtime wrapper。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.runtime_online import main, write_report

if __name__ == "__main__":
    main("B")
    write_report("B")
