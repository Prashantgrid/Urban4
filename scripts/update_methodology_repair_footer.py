#!/usr/bin/env python3
"""Overlay the implemented RP1 failure path on the supplied methodology SVG.

The supplied figure is retained as the visual source.  Its glyphs are outlined,
so this script replaces only the obsolete Phase-III footer with live SVG text
and then uses Inkscape to produce the matching vector PDF.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SVG = PROJECT / "figures" / "fig01_three_phase_methodology.svg"
PDF = PROJECT / "figures" / "fig01_three_phase_methodology.pdf"
START = "<!-- URBAN4_RP1_FOOTER_START -->"
END = "<!-- URBAN4_RP1_FOOTER_END -->"


def main() -> None:
    source = SVG.read_text(encoding="utf-8")
    source = re.sub(
        rf"{re.escape(START)}.*?{re.escape(END)}\s*",
        "",
        source,
        flags=re.DOTALL,
    )
    overlay = f"""
{START}
<g id="urban4-rp1-footer">
  <rect x="0" y="1410" width="1473.75" height="75" fill="#ffffff"/>
  <rect x="92" y="1418" width="905" height="57" rx="8" ry="8"
        fill="#f8f3fa" stroke="#7441ad" stroke-width="2"/>
  <text x="544" y="1438" text-anchor="middle"
        font-family="Times New Roman, Nimbus Roman, serif" font-size="14"
        font-weight="bold" fill="#7441ad">NO → RP1 applies one logged action, then repeats the complete coupled solve</text>
  <text x="544" y="1461" text-anchor="middle"
        font-family="Times New Roman, Nimbus Roman, serif" font-size="12.5"
        fill="#3f4650">interface audit → damping → bounded control → next catalogue component → Phase-II return</text>
  <rect x="1014" y="1418" width="443" height="57" rx="8" ry="8"
        fill="#fff4f2" stroke="#d94b3d" stroke-width="2"/>
  <text x="1235" y="1438" text-anchor="middle"
        font-family="Times New Roman, Nimbus Roman, serif" font-size="13"
        font-weight="bold" fill="#b3342b">ALL ACTIONS EXHAUSTED</text>
  <text x="1235" y="1458" text-anchor="middle"
        font-family="Times New Roman, Nimbus Roman, serif" font-size="11.5"
        fill="#3f4650">FAILED_NO_FEASIBLE_REPAIR · retain ledger · no export</text>
</g>
{END}
"""
    source = source.replace("</svg>", overlay + "</svg>")
    SVG.write_text(source, encoding="utf-8")
    subprocess.run(
        ["inkscape", str(SVG), "--export-type=pdf", f"--export-filename={PDF}"],
        check=True,
    )


if __name__ == "__main__":
    main()
