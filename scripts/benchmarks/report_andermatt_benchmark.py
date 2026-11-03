#!/usr/bin/env python3
"""Build the two-page report for the frozen 2026-11-03 accuracy experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/andermatt_benchmark")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/andermatt_topotherm")
    args = parser.parse_args()
    out = args.results
    ratios = pd.read_csv(out / "table_xii_comparison.csv").set_index("measure")
    absolute = pd.read_csv(out / "absolute_metrics.csv").set_index("network")
    spatial = pd.read_csv(out / "spatial_accuracy.csv").set_index(["network", "tolerance_m"])
    errors = json.loads((out / "route_distance_errors.json").read_text())
    solver = json.loads((out / "topotherm_run.json").read_text())
    audit = json.loads((out / "solution_audit.json").read_text())
    case = json.loads((args.data / "case.json").read_text())
    b = case["domain_bounds_2056"]
    area = (b[2]-b[0])*(b[3]-b[1])/1e6
    from matplotlib.font_manager import FontProperties, findfont
    for name, weight, style in [("BenchSerif", "normal", "normal"), ("BenchSerif-Bold", "bold", "normal"),
                                 ("BenchSerif-Italic", "normal", "italic")]:
        fontpath = findfont(FontProperties(family="DejaVu Serif", weight=weight, style=style))
        pdfmetrics.registerFont(TTFont(name, fontpath))
    pdfmetrics.registerFontFamily("BenchSerif", normal="BenchSerif", bold="BenchSerif-Bold", italic="BenchSerif-Italic")
    navy = colors.HexColor("#19374D")
    styles = {
        "title": ParagraphStyle("title", fontName="BenchSerif-Bold", fontSize=17, leading=20, textColor=navy, spaceAfter=5),
        "sub": ParagraphStyle("sub", fontName="BenchSerif", fontSize=9, leading=11, textColor=colors.HexColor("#555F68"), spaceAfter=10),
        "h": ParagraphStyle("h", fontName="BenchSerif-Bold", fontSize=11, leading=13, textColor=navy, spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="BenchSerif", fontSize=10, leading=12.6, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="BenchSerif", fontSize=8.6, leading=10.7, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="BenchSerif", fontSize=9.2, leading=11.2),
        "head": ParagraphStyle("head", fontName="BenchSerif-Bold", fontSize=9, leading=10.5, textColor=colors.white),
    }
    story = []
    def p(text, kind="body"):
        story.append(Paragraph(text, styles[kind]))
    width = A4[0] - 36*mm
    def table(headers, rows, widths):
        items = [[Paragraph(str(x), styles["head"]) for x in headers]]
        items += [[Paragraph(str(x), styles["cell"]) for x in row] for row in rows]
        t = Table(items, colWidths=[width*x for x in widths], hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),navy), ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F0F4F6"),colors.white]),
            ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#B9C5CD")),
        ]))
        story.append(t)
        story.append(Spacer(1,5))
    p("Andermatt district-heating routing accuracy", "title")
    p("Urban4 and Topotherm against mapped installed routes • Executed 3 November 2026", "sub")
    p("<b>Measured result: a routing tradeoff.</b> Urban4 covers more of the installed network and better reproduces its overall extent. Topotherm produces a smaller network whose routes align more closely with installed corridors. Neither method wins every measure.")
    p("Controlled case", "h")
    p(f"Both tools received the same <b>{case['customers']} observed transfer points</b>, northern local inlet, OSM road candidates and service attachments. Each point has a <b>50-kW scenario load</b> and 1,600 equivalent full-load hours; these are not measured demands. The common window covers {area:.3f} km² and contains {int(absolute.loc['Installed','evaluation_buildings'])} evaluation buildings. Active cadastral routes are held out except for the declared inlet. The upstream Göschenen plant/transmission route is excluded.")
    p("Table XII measures, recomputed with one common protocol", "h")
    rows = []
    specs = [("length_per_area", "Route length / area (km/km²)",3),
             ("length_per_connection", "Route length / connection (m/point)",2),
             ("connection_spacing", "Connection spacing (m) [control]",2),
             ("building_to_network_distance", "Building-to-route distance (m)",2),
             ("branch_length", "Mean branch length (m)",2),
             ("terminal_share", "Terminal share",4)]
    for key, label, digits in specs:
        row = ratios.loc[key]
        rows.append([label, f"{row.installed:.{digits}f}", f"{row.urban4_to_installed:.3f}", f"{row.topotherm_to_installed:.3f}"])
    table(["Measure", "Installed absolute", "Urban4 / installed", "Topotherm / installed"], rows, [.45,.19,.18,.18])
    p("Ratios equal to 1 indicate agreement. Connection spacing is fixed by the supplied inputs. The first two ratios are two normalisations of the same route length. Topology descriptors use common 0.5-m quantisation, noding and degree-2 suppression; they describe inferred 2D geometry, not verified hydraulic connectivity.", "small")
    p("Direct spatial accuracy", "h")
    rows = []
    for tol in [10,25,50]:
        u,t = spatial.loc[("Urban4",tol)],spatial.loc[("Topotherm",tol)]
        rows.append([f"{tol} m", f"{u.installed_route_covered_pct:.1f}% / {t.installed_route_covered_pct:.1f}%",
                     f"{u.generated_route_matching_pct:.1f}% / {t.generated_route_matching_pct:.1f}%"])
    table(["Tolerance", "Installed length covered\nUrban4 / Topotherm", "Generated length matching\nUrban4 / Topotherm"],rows,[.16,.42,.42])
    eu,et=errors["Urban4"],errors["Topotherm"]
    p(f"Mean generated-to-installed route error is <b>{eu['generated_to_installed']['mean_m']:.2f} m for Urban4 versus {et['generated_to_installed']['mean_m']:.2f} m for Topotherm</b> (95th percentiles {eu['generated_to_installed']['p95_m']:.2f} and {et['generated_to_installed']['p95_m']:.2f} m). Reverse installed-to-generated means are {eu['installed_to_generated']['mean_m']:.2f} and {et['installed_to_generated']['mean_m']:.2f} m. Distances use length-weighted midpoint samples at at most 5-m spacing.", "small")

    story.append(PageBreak())
    p("Installed and generated local routes", "title")
    from PIL import Image as PILImage
    imgpath = out / "Andermatt_DH_Routes_Comparison.png"
    with PILImage.open(imgpath) as im:
        w,h = im.size
    story.append(Image(str(imgpath),width=width,height=width*h/w))
    p("Unique clipped route lengths: installed 10.86 km; Urban4 9.56 km; Topotherm 6.75 km. Grey overlays show installed routes in the model panels. The complete OSM candidate universe covers 90.6% of mapped reference length within 25 m, indicating a road-corridor representation limit.","small")
    p("Execution and checks", "h")
    p(f"Topotherm 0.6.0 (pinned commit ec108e94) used its single-timestep forced-connection model with default pipe regressions, Pyomo 6.10.1 and HiGHS 1.15.1. On the common 411-node / 551-edge graph it reached the requested <b>1% MIP gap in {solver['runtime_s']:.2f} s</b> (actual gap {100*audit['relative_objective_gap']:.6f}%). All 85 points are connected. Independent solution replay passed flow balance, capacity, integrality, connectivity and graph checks; six geometry/contraction tests passed. This is a tolerance-certified solution, rather than a zero-gap proof.","small")
    p("Supported manuscript claim", "h")
    p("In this conditional Andermatt routing case, Urban4 covered 68.1% of installed route length within 25 m, compared with 62.7% for Topotherm; the reverse fractions were 74.4% and 86.7%. Mean generated-route location errors were 28.0 and 15.0 m. Urban4 better reproduced mapped extent and building proximity, whereas Topotherm better localised its more compact network. These findings quantify a case-specific routing tradeoff conditional on supplied connections and scenario loads.")
    p("Scope and manuscript integration", "h")
    p("This is a <b>new frozen 85-point case</b>. The former Table XII Andermatt row used 50 points; its exact input sample and extraction script are unavailable. Replace that row and related discussion, or label the cases separately. Do not attach these scores to the old row. The cadastral transfer-point layer is not assumed exhaustive. This experiment does not validate customer selection, measured heat demand, physical topology, hydraulics or accuracy in the other three sectors.","small")
    p("Reproducibility and primary sources", "h")
    p('Code, input hashes, frozen GIS data, solver log, edge decisions, all scores and replay instructions: <b>docs/andermatt_topotherm_benchmark.md</b> in the Urban4 benchmark branch. The separate comparison PDF preserves vector route geometry.',"small")
    p('<link href="https://opendata.swiss/de/dataset/fernwarme-leitungen" color="#19374D">Cadastral reference: Lisag AG / Geoinformation Kanton Uri, Fernwärme Leitungen</link> (open resource terms). Roads/buildings: <link href="https://www.openstreetmap.org/copyright" color="#19374D">© OpenStreetMap contributors, ODbL</link>. Model: <link href="https://github.com/jylambert/topotherm/tree/ec108e94b2e3f823a154afc47531156b1c85539f" color="#19374D">official Topotherm pinned source</link>. Retrieval and execution: 3 November 2026.',"small")
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#CBD3DA"));canvas.setLineWidth(.4)
        canvas.line(18*mm,15.5*mm,A4[0]-18*mm,15.5*mm)
        canvas.setFont("BenchSerif",8);canvas.setFillColor(colors.HexColor("#66717A"))
        canvas.drawString(18*mm,11*mm,"Andermatt • conditional local-distribution routing benchmark")
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
    target=out/"Andermatt_DH_Routing_Accuracy_Benchmark.pdf"
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=21*mm,
                          title="Andermatt district-heating routing accuracy",author="Urban4 benchmark",pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    main()
