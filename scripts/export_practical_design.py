#!/usr/bin/env python3
"""Export practical design envelopes and screening equipment sizes."""
from __future__ import annotations

import json
from pathlib import Path

from urban4.practical_sizing import export_practical_design_outputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "publication_ready_v1.6.0"

if __name__ == "__main__":
    result = export_practical_design_outputs(ROOT, OUTPUT)
    print(json.dumps(result, indent=2))
