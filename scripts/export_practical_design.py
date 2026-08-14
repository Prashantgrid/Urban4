#!/usr/bin/env python3
"""Export practical design envelopes and screening equipment sizes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban4.practical_sizing import export_practical_design_outputs

OUTPUT = ROOT / "outputs" / "verified_design_v1.6.0"

if __name__ == "__main__":
    result = export_practical_design_outputs(ROOT, OUTPUT)
    print(json.dumps(result, indent=2))
