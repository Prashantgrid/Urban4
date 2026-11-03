#!/usr/bin/env python3
"""Vector comparison map for the frozen Andermatt accuracy experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/andermatt_benchmark")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/andermatt_topotherm")
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif","DejaVu Serif"],
                         "font.size":8,"pdf.fonttype":42,"svg.fonttype":"none"})
    boundary = gpd.read_file(args.data / "common_inputs.gpkg", layer="boundary")
    reference = gpd.read_file(args.data / "common_inputs.gpkg", layer="installed_reference")
    customers = gpd.read_file(args.data / "common_inputs.gpkg", layer="customers")
    source = gpd.read_file(args.data / "common_inputs.gpkg", layer="source")
    buildings = gpd.read_file(args.results / "routes.gpkg", layer="evaluation_buildings")
    metrics = pd.read_csv(args.results / "absolute_metrics.csv").set_index("network")
    spatial = pd.read_csv(args.results / "spatial_accuracy.csv")
    minx,miny,maxx,maxy = boundary.total_bounds
    fig,axes = plt.subplots(1,3,figsize=(7.15,4.35),layout=None)
    panels = [("Installed",None,"#364052"),("Urban4","urban4","#D98900"),("Topotherm","topotherm","#2474D2")]
    for ax,(label,layer,color) in zip(axes,panels):
        buildings.plot(ax=ax,facecolor="#E7E8E9",edgecolor="#D4D6D8",linewidth=.15,zorder=0)
        reference.plot(ax=ax,color="#9BA2AD" if layer else color,linewidth=.5 if layer else .8,zorder=1)
        if layer:
            gpd.read_file(args.results / "routes.gpkg",layer=layer).clip(boundary).plot(ax=ax,color=color,linewidth=.95,zorder=3)
        customers.plot(ax=ax,color="#313B48",markersize=3.5,zorder=4)
        source.plot(ax=ax,color="#1F6D4F",marker="*",markersize=70,edgecolor="white",linewidth=.35,zorder=5)
        ax.set_xlim(minx-20,maxx+20);ax.set_ylim(miny-20,maxy+30);ax.set_aspect("equal");ax.axis("off")
        title = {"Installed":"(a) Mapped installed routes","Urban4":"(b) Urban4","Topotherm":"(c) Topotherm 0.6.0"}[label]
        ax.set_title(title,fontsize=9,fontweight="bold",pad=19)
        ax.text(.5,1.012,f'{metrics.loc[label,"route_km"]:.2f} km',transform=ax.transAxes,ha="center",va="bottom",fontsize=8)
        if layer:
            row=spatial.loc[(spatial.network==label)&(spatial.tolerance_m==25)].iloc[0]
            ax.text(.5,-.025,f'Generated routes within 25 m: {row.generated_route_matching_pct:.1f}%',
                    transform=ax.transAxes,ha="center",va="top",fontsize=7)
        else:
            ax.text(.5,-.025,"85 observed transfer points",transform=ax.transAxes,ha="center",va="top",fontsize=7)
        y=miny+60;x=minx+55
        ax.plot([x,x+200],[y,y],color="#28313B",lw=1.4)
        ax.text(x+100,y+25,"200 m",fontsize=6.5,ha="center")
        ax.annotate("N",xy=(maxx-65,maxy-120),xytext=(maxx-65,maxy-260),ha="center",va="center",
                    fontsize=7,arrowprops={"arrowstyle":"-|>","lw":.7,"color":"#28313B"})
    handles=[Line2D([0],[0],color="#6D7785",lw=1,label="Installed routes"),
             Line2D([0],[0],color="#D98900",lw=1.2,label="Urban4"),
             Line2D([0],[0],color="#2474D2",lw=1.2,label="Topotherm"),
             Line2D([0],[0],color="#313B48",marker="o",ms=3,lw=0,label="Transfer point"),
             Line2D([0],[0],color="#1F6D4F",marker="*",ms=7,lw=0,label="Incoming feed")]
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,.055),fontsize=7)
    fig.text(.5,.018,"Same local inlet, customers, loads and OSM road candidates. CRS: EPSG:2056. Cadastral data: Lisag AG / Kanton Uri; buildings: OpenStreetMap contributors.",
             ha="center",fontsize=5.8,color="#535A64")
    fig.subplots_adjust(left=.015,right=.985,top=.91,bottom=.17,wspace=.08)
    stem=args.results/"Andermatt_DH_Routes_Comparison"
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    print(stem.with_suffix('.pdf'))


if __name__ == "__main__":
    main()
