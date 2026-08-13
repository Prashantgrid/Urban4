#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
COL = {'water':'#2474D2','power':'#D84A3A','green':'#2E7D32','slate':'#354052','gold':'#E69F00','purple':'#7B4AB5'}
CITY_PEAK_MW = 40.962

mpl.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Times','Nimbus Roman'],
    'font.size':8.4,'axes.labelsize':8.4,'axes.titlesize':9.0,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
})

def _finish(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(FIG/f'{stem}.{ext}', dpi=320, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)

def leak_figure():
    d = pd.read_csv(OUT/'hydraulic_leak_coupling'/'hydraulic_leak_timeseries.csv')
    # Use the accepted state immediately before the leak, not the first
    # controller warm-up step.
    base = d.loc[~d.leak_active.astype(bool)].iloc[-1]
    final = d.iloc[-1]
    dp_kw=(final.pump_electrical_power_mw-base.pump_electrical_power_mw)*1000
    dp_pct=100*(final.pump_electrical_power_mw/base.pump_electrical_power_mw-1)
    city_share=100*(final.pump_electrical_power_mw-base.pump_electrical_power_mw)/CITY_PEAK_MW
    dv=(final.pump_bus_voltage_pu-base.pump_bus_voltage_pu)

    fig,axs=plt.subplots(1,2,figsize=(7.15,2.55))
    ax=axs[0]; ax2=ax.twinx()
    ax.plot(d.elapsed_minute,d.leak_flow_l_s,lw=1.7,color=COL['water'],label='Leak flow')
    ax2.plot(d.elapsed_minute,d.pump_electrical_power_mw*1000,lw=1.7,color=COL['power'],label='Pump active power')
    ax.axvline(30,lw=.8,ls='--',color='0.45')
    ax.set_xlabel('Elapsed time (min)'); ax.set_ylabel('Leak flow (L s$^{-1}$)'); ax2.set_ylabel('Pump power (kW)')
    ax.set_title('(a) Hydraulic disturbance increases pump duty',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
    ax.text(.03,.95,f'$\\Delta P_{{pump}}$ = {dp_kw:.1f} kW ({dp_pct:.1f}%)',transform=ax.transAxes,va='top',fontsize=8.1,
            bbox=dict(boxstyle='round,pad=.18',fc='white',ec='0.75',alpha=.92))
    lines=ax.get_lines()[:1]+ax2.get_lines()[:1]; ax.legend(lines,[x.get_label() for x in lines],loc='lower right',frameon=False,fontsize=7.6)

    ax=axs[1]; ax2=ax.twinx()
    ax.plot(d.elapsed_minute,d.critical_node_pressure_m,lw=1.7,color=COL['water'],label='Minimum nodal pressure')
    ax.axhline(20,lw=.85,ls='--',color=COL['power'],label='20 m screen')
    dv_series=(d.pump_bus_voltage_pu-base.pump_bus_voltage_pu)*1e4
    ax2.plot(d.elapsed_minute,dv_series,lw=1.5,color=COL['slate'],label='$\\Delta V_{PB046}$')
    ax.axvline(30,lw=.8,ls='--',color='0.45')
    ax.set_xlabel('Elapsed time (min)'); ax.set_ylabel('Minimum nodal pressure (m)'); ax2.set_ylabel('$\\Delta V_{PB046}$ ($10^{-4}$ p.u.)')
    ax.set_title('(b) Cross-sector consequence is scale-limited',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
    ax.text(.03,.95,f'Pump increment = {city_share:.3f}% of city peak\nFinal $\\Delta V_{{PB046}}$ = {dv*1e4:.2f}×10$^{{-4}}$ p.u.',
            transform=ax.transAxes,va='top',fontsize=7.8,bbox=dict(boxstyle='round,pad=.18',fc='white',ec='0.75',alpha=.92))
    lines=[ax.lines[0],ax.lines[1],ax2.lines[0]]; ax.legend(lines,[x.get_label() for x in lines],loc='lower left',frameon=False,fontsize=7.3)
    fig.subplots_adjust(wspace=.43)
    _finish(fig,'Fig08_Hydraulic_Leak_Response')


def electrical_figure():
    case=OUT/'network_originating_contingency_v2.7.0'
    manifest=json.loads((case/'manifest.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(case/'network_originating_outage_summary.csv')
    selection=manifest['selection']
    base=summary.iloc[0]; outage=summary.iloc[1]

    footprint_labels=['Buses','Static load','Transformers','Powered interfaces']
    footprint_values=[
        100*selection['deenergized_bus_fraction'],
        100*selection['unserved_static_load_fraction'],
        100*selection['deenergized_transformer_count']/selection['base_transformer_count'],
        100*selection['deenergized_powered_interface_count']/selection['base_powered_interface_count'],
    ]
    footprint_counts=[
        f"{selection['deenergized_bus_count']}/{selection['base_bus_count']}",
        f"{selection['unserved_static_load_mw']:.2f} MW",
        f"{selection['deenergized_transformer_count']}/{selection['base_transformer_count']}",
        f"{selection['deenergized_powered_interface_count']}/{selection['base_powered_interface_count']}",
    ]

    fig,axs=plt.subplots(1,2,figsize=(7.15,2.62))
    ax=axs[0]
    y=np.arange(len(footprint_labels))
    bars=ax.barh(y,footprint_values,color=[COL['slate'],COL['power'],COL['purple'],COL['gold']],height=.58)
    ax.set_yticks(y,footprint_labels); ax.invert_yaxis(); ax.set_xlim(0,34)
    ax.set_xlabel('Source-disconnected share of accepted model (%)')
    ax.set_title('(a) The selected feeder has a network-scale footprint',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True)
    for bar,value,count in zip(bars,footprint_values,footprint_counts):
        ax.text(value+.6,bar.get_y()+bar.get_height()/2,f'{value:.1f}%  ({count})',va='center',fontsize=7.5)

    ax=axs[1]; ax2=ax.twinx(); x=np.arange(2); width=.34
    p_bars=ax.bar(x-width/2,[base.minimum_pressure_m,outage.minimum_pressure_m],width,
                  color=COL['water'],label='Minimum pressure')
    d_bars=ax2.bar(x+width/2,[base.delivered_water_percent,outage.delivered_water_percent],width,
                   color=COL['green'],label='Delivered demand')
    ax.axhline(20,lw=.85,ls='--',color=COL['power'],label='20 m screen')
    ax.axhline(0,lw=.7,color='0.35')
    ax.set_xticks(x,['Accepted base','Waterworks-feeder outage'])
    ax.set_ylabel('Minimum nodal pressure (m)'); ax2.set_ylabel('Delivered demand (%)')
    ax.set_ylim(-72,55); ax2.set_ylim(0,112)
    ax.set_title('(b) Main-waterworks outage collapses service',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    for bar,value in zip(p_bars,[base.minimum_pressure_m,outage.minimum_pressure_m]):
        ax.text(bar.get_x()+bar.get_width()/2,value+(3 if value>=0 else -3),f'{value:.1f} m',ha='center',
                va='bottom' if value>=0 else 'top',fontsize=7.4,color=COL['water'])
    for bar,value in zip(d_bars,[base.delivered_water_percent,outage.delivered_water_percent]):
        ax2.text(bar.get_x()+bar.get_width()/2,value+2.5,f'{value:.3f}%',ha='center',va='bottom',fontsize=7.4,color=COL['green'])
    fig.subplots_adjust(wspace=.38)
    _finish(fig,'Fig09_Electrical_Supply_Disturbance')

if __name__=='__main__':
    leak_figure(); electrical_figure()
