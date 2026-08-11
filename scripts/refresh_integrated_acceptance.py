#!/usr/bin/env python3
"""Re-run native export checks and refresh the integrated manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from urban4.integrated_generation import OUTPUT
from urban4.native_acceptance import refresh_manifest


if __name__ == "__main__":
    print(json.dumps(refresh_manifest(OUTPUT), indent=2))
