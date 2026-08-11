#!/usr/bin/env python3
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
FIG=ROOT/'figures'
C={'E':'#D84A3A','W':'#2474D2','S':'#354052','H':'#E69F00','common':'#2E8B57','ind':'#7A3E9D','ink':'#18212B'}
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','Nimbus Roman'],'font.size':8.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def main():
    bm=pd.read_csv(OUT/'benchmark_case_metrics.csv')
    seeds=pd.read_csv(OUT/'clustering_seed_sensitivity.csv')
    ab=pd.read_csv(OUT/'common_ledger_ablation.csv')
    ids=['dense_urban','semi_urban','rural_peripheral']
    names=['Innenstadt\n(dense)','Bergl/Gartenstadt\n(semi-urban)','Dittelbrunn\n(peripheral)']
    sectors=[('electricity','E'),('drinking_water','W'),('wastewater','S'),('district_heating','H')]
    x=np.arange(3)
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.05))
    # a route per 1000 buildings
    ax=axs[0,0]
    width=.19
    for j,(sec,key) in enumerate(sectors):
        vals=[]
        for cid in ids:
            row=bm[(bm.case_id==cid)&(bm.sector==sec)].iloc[0]
            vals.append(float(row.network_length_per_1000_buildings_km))
        ax.bar(x+(j-1.5)*width,vals,width,color=C[key],label=sec.replace('_',' ').title())
    ax.set_ylabel('Route (km / 1,000 buildings)'); ax.set_title('(a) Route normalized by registered buildings',loc='left',fontweight='bold')
    ax.legend(ncol=2,fontsize=6.8,frameon=False,loc='upper left')
    # b mean path
    ax=axs[0,1]
    for j,(sec,key) in enumerate(sectors):
        vals=[]
        for cid in ids:
            row=bm[(bm.case_id==cid)&(bm.sector==sec)].iloc[0]
            vals.append(float(row.average_service_route_km))
        ax.bar(x+(j-1.5)*width,vals,width,color=C[key])
    ax.set_ylabel('Mean anchor-terminal path (km)'); ax.set_title('(b) Direction-neutral service path',loc='left',fontweight='bold')
    # c seed sensitivity
    ax=axs[1,0]
    data=[seeds[seeds.case_id==cid].nearest_zone_spacing_p95_km.to_numpy() for cid in ids]
    bp=ax.boxplot(data,positions=x,widths=.55,patch_artist=True,showfliers=False)
    for patch in bp['boxes']: patch.set_facecolor('#D7DFE8'); patch.set_edgecolor('#53657A')
    for med in bp['medians']: med.set_color(C['ink']); med.set_linewidth(1.2)
    ax.set_ylabel('P95 nearest-zone spacing (km)'); ax.set_title('(c) 20-seed planning-zone spread',loc='left',fontweight='bold')
    # d aggregation crosswalk need
    ax=axs[1,1]
    a=ab.set_index('case_id').loc[ids]
    common=100-a['common_ledger_exact_zone_interfaces_percent'].to_numpy()
    independent=100-a['independent_sector_clustering_exact_coincidence_percent'].to_numpy()
    ax.bar(x-.18,common,.36,color=C['common'],label='Common ledger')
    ax.bar(x+.18,independent,.36,color=C['ind'],label='Independent clustering')
    ax.set_ylabel('Zone supports needing spatial crosswalk (%)')
    ax.set_title('(d) Aggregation-support consistency',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=6.9,loc='upper left')
    ax.text(.02,.95,'Building IDs remain exact in both cases',transform=ax.transAxes,va='top',fontsize=7.0,color='#555555')
    for ax in axs.flat:
        ax.set_xticks(x,names,fontsize=7.3); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(hspace=.38,wspace=.28,left=.09,right=.985,bottom=.10,top=.96)
    for ext in ('pdf','svg','png'):
        fig.savefig(FIG/f'Fig06_Morphology_and_Ledger_Robustness.{ext}',dpi=320,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

if __name__=='__main__': main()
