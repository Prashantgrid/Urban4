#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib as mpl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'municipal_scale_v2.5.0'
FIG=ROOT/'figures'
INTEG=ROOT/'outputs'/'integrated_service_resolved'

mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','Nimbus Roman'],'font.size':10.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
C={'E':'#D84A3A','MV':'#7E2948','W':'#2474D2','S':'#354052','H':'#E69F00'}

def draw_geom(ax,df,color,lw,ls='-',alpha=.87,zorder=2):
    segments=[]
    for r in df.itertuples(index=False):
        try:g=json.loads(r.geometry_json)
        except:continue
        if not g or len(g)<2:continue
        segments.append([(float(p[0]),float(p[1])) for p in g])
    if segments:
        lc=LineCollection(segments,colors=color,linewidths=lw,linestyles=ls,alpha=alpha,zorder=zorder)
        ax.add_collection(lc)

def main():
    b=pd.read_csv(ROOT/'outputs'/'building_sector_demands.csv')
    e=pd.read_csv(OUT/'electricity_links_municipal.csv')
    w=pd.read_csv(OUT/'drinking_water_links.csv')
    s=pd.read_csv(OUT/'wastewater_links.csv')
    h=pd.read_csv(OUT/'district_heating_corridors_municipal.csv')
    ws=pd.read_csv(OUT/'drinking_water_station_municipal.csv').iloc[0]
    wz=pd.read_csv(OUT/'drinking_water_zone_boundaries.csv')
    ww=pd.read_csv(OUT/'wastewater_pump_stations_municipal.csv')
    hs=pd.read_csv(OUT/'district_heating_sources_municipal.csv')
    water_native=json.load(open(OUT/'drinking_water_native_manifest.json'))
    sewer_native=json.load(open(OUT/'wastewater_station_native_manifest.json'))
    heat_audit=pd.read_csv(OUT/'district_heating_connectivity_audit.csv').iloc[0]
    fig,axes=plt.subplots(2,2,figsize=(8.3,7.45),sharex=True,sharey=True)
    axes=axes.ravel()
    # Common extent and evidence background make the four panels directly comparable.
    sx0,sy0=10.2115927,50.0257927
    xs=pd.concat([b.lon,pd.Series([ws.source_coordinate_lon,*hs.lon.tolist(),sx0]),ww.lon]); ys=pd.concat([b.lat,pd.Series([ws.source_coordinate_lat,*hs.lat.tolist(),sy0]),ww.lat])
    dx=xs.max()-xs.min();dy=ys.max()-ys.min(); xlim=(xs.min()-.02*dx,xs.max()+.02*dx); ylim=(ys.min()-.03*dy,ys.max()+.03*dy)
    for ax in axes:
        ax.scatter(b.lon,b.lat,s=.55,c='#B8BDC4',alpha=.23,linewidths=0,zorder=0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect(1/0.6428); ax.axis('off')

    # Electricity panel.
    ax=axes[0]
    draw_geom(ax,e[e.voltage_level.eq('LV')],C['E'],.24,alpha=.35,zorder=1)
    draw_geom(ax,e[e.voltage_level.eq('MV')],C['MV'],.72,'--',alpha=.70,zorder=2)
    en=pd.read_csv(INTEG/'electricity_nodes.csv'); er=en[en.node_type.astype(str).str.contains('source|external',case=False,regex=True)]
    if er.empty: ex,ey=10.2265821,50.047744
    else: ex,ey=float(er.iloc[0].lon),float(er.iloc[0].lat)
    ax.scatter(ex,ey,s=95,marker='s',c='#C7362F',edgecolors='white',linewidths=1.0,zorder=8)
    ax.annotate('E1',(ex,ey),xytext=(5,5),textcoords='offset points',fontsize=7.5,fontweight='bold')
    ax.set_title('(a) Electricity\n105 MV/LV sites; radial MV/LV topology',fontsize=10.2,loc='left',pad=3)

    # Drinking-water panel: show mains, not 22,641 property services.
    ax=axes[1]
    draw_geom(ax,w[w.link_type.eq('distribution_main')],C['W'],.62,alpha=.72,zorder=2)
    for r in wz.itertuples():
        ax.scatter(r.lon,r.lat,s=82,marker='o',c='#00A6D6',edgecolors='white',linewidths=.85,zorder=8)
        ax.annotate(r.pressure_zone_id,(r.lon,r.lat),xytext=(4,4),textcoords='offset points',fontsize=6.8,fontweight='bold')
    ax.set_title(f"(b) Drinking water\nEPANET: {water_native['peak_minimum_service_pressure_m']:.1f}–{water_native['average_maximum_service_pressure_m']:.1f} m; {water_native['peak_maximum_velocity_m_s']:.2f} m s$^{{-1}}$ peak",fontsize=10.2,loc='left',pad=3)

    # Wastewater panel: full retained main/manhole spatial layer plus 13 stations.
    ax=axes[2]
    draw_geom(ax,s[s.link_type.isin(['gravity_main','force_main'])],C['S'],.48,(0,(2.5,1.5)),alpha=.66,zorder=2)
    ax.scatter(sx0,sy0,s=105,marker='D',c='#5B2F83',edgecolors='white',linewidths=.9,zorder=9)
    ax.annotate('S0',(sx0,sy0),xytext=(5,-9),textcoords='offset points',fontsize=7.2,fontweight='bold',color='#3E205B')
    for r in ww.itertuples():
        ax.scatter(r.lon,r.lat,s=62,marker='^',c='#7B4AB5',edgecolors='white',linewidths=.55,zorder=8)
    ax.set_title(f"(c) Wastewater\nSWMM: {sewer_native['continuity_error_percent']:.3f}% continuity; {sewer_native['flooding_loss_percent']:.3f}% flooding",fontsize=10.2,loc='left',pad=3)

    # District-heating panel.
    ax=axes[3]
    draw_geom(ax,h,C['H'],1.12,alpha=.90,zorder=4)
    for r in hs.itertuples():
        face='#F2A900' if r.map_id=='H1' else '#FFD36A'
        ax.scatter(r.lon,r.lat,s=155,marker='*',c=face,edgecolors='#5C4200',linewidths=.85,zorder=10)
        ax.annotate(r.map_id,(r.lon,r.lat),xytext=(6,6),textcoords='offset points',fontsize=8.0,fontweight='bold',color='#6E4A00')
    ax.set_title(f"(d) District heating\n841 substations; {100*heat_audit.source_reachability_fraction:.0f}% source reachable",fontsize=10.2,loc='left',pad=3)

    handles=[
      Line2D([0],[0],color=C['E'],lw=1.5,label='LV'),Line2D([0],[0],color=C['MV'],lw=1.5,ls='--',label='MV'),Line2D([0],[0],color=C['W'],lw=1.8,label='Water main'),Line2D([0],[0],color=C['S'],lw=1.6,ls=(0,(2.5,1.5)),label='Sewer main'),Line2D([0],[0],color=C['H'],lw=2.0,label='Heat corridor'),
      Line2D([0],[0],marker='o',color='none',markerfacecolor='#00A6D6',markersize=7,label='WZ boundary'),Line2D([0],[0],marker='D',color='none',markerfacecolor='#5B2F83',markersize=7,label='Treatment works'),Line2D([0],[0],marker='^',color='none',markerfacecolor='#7B4AB5',markersize=7,label='13 pumpworks'),Line2D([0],[0],marker='*',color='none',markerfacecolor='#F2A900',markersize=10,label='Heat sources')]
    fig.legend(handles=handles,ncol=9,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,fontsize=7.4,columnspacing=.75,handlelength=1.35)
    fig.subplots_adjust(left=.012,right=.995,top=.965,bottom=.075,wspace=.035,hspace=.12)
    for ext in ['pdf','svg','png']:
        fig.savefig(FIG/f'fig07_schweinfurt_municipal_scale_v2_5_0.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
        fig.savefig(OUT/f'fig07_schweinfurt_municipal_scale_v2_5_0.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

if __name__=='__main__': main()
