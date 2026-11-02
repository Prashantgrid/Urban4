#!/usr/bin/env python3
"""Rebuild Fig. 7 from retained results and an unchanged vector map panel.

All right-hand plots are regenerated from CSVs, without a network rebuild.
Scenario codes remain stable as data keys; visible labels are descriptive.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import u4style as S

S.apply()
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results', type=Path, default=root / 'results/heat_power_case')
ap.add_argument('--map-source', type=Path, default=Path(__file__).parent / 'data/heat_power_map_reference.pdf')
ap.add_argument('--out', type=Path, default=root / 'figures')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
res = pd.read_csv(args.results / 'heat_power_states.csv')
var = pd.read_csv(args.results / 'heat_power_variant_states.csv')
cap = pd.read_csv(args.results / 'dh_hydraulic_capability.csv')
summ = json.loads((args.results / 'heat_power_summary.json').read_text())
t = np.sort(res.t_a_c.unique())[::-1]

def series(sc, col):
    return res[res.scenario.eq(sc)].set_index('t_a_c').reindex(t)[col].to_numpy(dtype=float)

def vseries(f, col):
    return var[var.ratio_mode.eq('measured') & var.renovation_f.eq(f)].set_index('t_a_c').reindex(t)[col].to_numpy(dtype=float)

c0, c1, c2, c3 = S.HEAT, S.POWER, S.SHARED, '#2A9D8F'
fig = plt.figure(figsize=(S.TEXT_W, 4.85))
fig.text(.004, .975, '(a) Heat-pump conversion at −14 °C', fontsize=8, fontweight='bold', va='top')
fig.text(.004, .945, 'COP model; existing demand at the reference peak', fontsize=6.5, va='top', color=S.MUTED)
axb = fig.add_axes([.555, .685, .43, .257])
axc = fig.add_axes([.555, .385, .43, .235], sharex=axb)
axd = fig.add_axes([.555, .075, .43, .235], sharex=axb)

axb.plot(t, series('S1_HP', 'added_electric_mw'), color=c1, label='Heat-pump conversion: COP')
axb.plot(t, vseries(1., 'added_electric_mw'), color=c1, lw=.9, ls=(0, (3, 1.6)), label='Heat-pump conversion: ratio 0.40')
axb.plot(t, vseries(.6, 'added_electric_mw'), color=c1, lw=.8, ls=(0, (1, 1.3)), label='Ratio 0.40 + 40% less heat demand')
axb.plot(t, series('S2_GKS_outage', 'added_electric_mw'), color=c2, label='CHP outage + electric backup')
axb.plot(t, series('S0_DH', 'added_electric_mw'), color=c0, label='Existing DH / CHP outage + reinforced link')
axb.set_ylabel('Added load (MW)')
axb.set_ylim(-1, 38); axb.set_yticks([0, 10, 20, 30])
S.panel_title(axb, '(b) Added electrical load')
axb.legend(loc='upper left', fontsize=6.1, handlelength=1.8, labelspacing=.25, borderaxespad=.15,
           title='Same lines in (b) and (c)', title_fontsize=6, alignment='left')

axc.step(t, series('S0_DH', 'scr_overloaded_trafos'), where='mid', color=c0, lw=2.6)
axc.step(t, series('S1_HP', 'scr_overloaded_trafos'), where='mid', color=c1)
axc.step(t, vseries(1., 'overloaded_areas'), where='mid', color=c1, lw=.9, ls=(0, (3, 1.6)))
axc.step(t, vseries(.6, 'overloaded_areas'), where='mid', color=c1, lw=.8, ls=(0, (1, 1.3)))
axc.step(t, series('S2_GKS_outage', 'scr_overloaded_trafos'), where='mid', color=c2, lw=1.1)
axc.axvspan(-2.5, -14.5, color='#EEF0F2', zorder=0, lw=0)
axc.text(-8.2, 25.2, 'Heat pumps, COP:\nAC power flow does not converge', ha='center', va='top', fontsize=5.8, color=S.MUTED)
axc.set_ylabel('Overloaded areas'); axc.set_ylim(-1, 26.5); axc.set_yticks([0, 5, 10, 15, 20])
S.panel_title(axc, '(c) Overloaded transformer areas (of 220)')

d = cap[cap['case'].eq('H2_only')].sort_values('phi')
d = d[d.phi >= summ['h2_only_max_fraction'] - .03]
axd.plot(t, series('S0_DH', 'dh_v_max_m_s'), color=c0, label='Existing district heating')
axd.plot(15 - d.phi * 29, d.v_max_m_s, color=c2, lw=.9, ls=(0, (1, 1.2)), label='CHP outage, no flow limit')
axd.plot(t, series('S2_GKS_outage', 'dh_v_max_m_s'), color=c2, label='CHP outage, existing heat link')
axd.plot(t, series('S3_GKS_outage_DN400', 'dh_v_max_m_s'), color=c3, ls=(0, (3, 1.6)), label='CHP outage, reinforced heat link')
axd.axhline(2, color=S.INK, lw=.7, ls=(0, (1, 1.5)))
axd.text(14.6, 2.06, '2.0 m/s limit', fontsize=6, va='bottom')
ts = summ['h2_only_threshold_c']
axd.plot([ts], [2], 'o', ms=4.5, mfc=c2, mec='white', mew=.6, zorder=5)
axd.annotate(f'about {ts:.0f} °C'.replace('-', '−'), (ts, 2), xytext=(4, -4), textcoords='offset points', ha='left', va='top', fontsize=6.2)
axd.set_ylabel('Flow speed (m/s)'); axd.set_ylim(0, 4.1); axd.set_yticks([0, 1, 2])
S.panel_title(axd, '(d) Highest flow speed in the DH network')
axd.legend(loc='upper left', fontsize=6, handlelength=1.8, labelspacing=.15, borderaxespad=.12)
axd.set_xlabel('Outdoor temperature (°C)')
for ax in (axb, axc, axd):
    ax.set_xlim(15.5, -14.5); ax.set_xticks([15, 10, 5, 0, -5, -10, -14])
for ax in (axb, axc):
    plt.setp(ax.get_xticklabels(), visible=False)

with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    fig.savefig(temp / 'plots.pdf', dpi=400)
    doc = fitz.open(temp / 'plots.pdf')
    source = fitz.open(args.map_source)
    pg = source[0]
    for rect in pg.search_for('DN300 section (limits S2)'):
        pg.add_redact_annot(rect, fill=(1, 1, 1))
    link_rects = pg.search_for('Link pipe between plants')
    for rect in link_rects:
        pg.add_redact_annot(rect, fill=(1, 1, 1))
    pg.apply_redactions(images=0, graphics=0)
    pg.insert_text((19.36, 243.), 'DN300 heat-link section', fontname='tiro', fontsize=6.4, color=(.106, .137, .173))
    for rect in link_rects:
        pg.insert_text((rect.x0, rect.y1-1.55), 'Heat link between plants', fontname='tiro', fontsize=6.4, color=(.106, .137, .173))
    doc[0].show_pdf_page(fitz.Rect(1., 31., 237., 323.), source, 0,
                          clip=fitz.Rect(0, 10., 231.5, 265.), keep_proportion=True)
    out = args.out / 'Fig07_Heat_Power_Case.pdf'
    doc.save(out, garbage=4, deflate=True)
    doc[0].get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72)).save(args.out / 'Fig07_Heat_Power_Case.png')
    doc.close(); source.close()
plt.close(fig)
print(out)
