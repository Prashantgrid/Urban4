#!/usr/bin/env python3
"""Run the Urban4 normal-operation interface fixed point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from urban4.integrated_generation import OUTPUT
from urban4.service_coupling import run_nominal_interface_closure


if __name__ == "__main__":
    print(json.dumps(run_nominal_interface_closure(OUTPUT), indent=2))
