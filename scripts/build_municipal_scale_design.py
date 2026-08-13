#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, shutil, sys
import numpy as np
import pandas as pd
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban4 import schweinfurt_base as base
from urban4.service_topologies import attach_buildings, choose_heat_dn
from urban4.municipal_native import (
    generate_accepted_municipal_electricity_network,
    generate_accepted_municipal_heat_network,
    generate_accepted_municipal_water_network,
    generate_accepted_municipal_wastewater_station_model,
)

OUT=ROOT/'outputs'/'municipal_scale_v2.5.0'
OUT.mkdir(parents=True,exist_ok=True)
INTEG=ROOT/'outputs'/'integrated_service_resolved'
PRINC=ROOT/'outputs'/'water_sewer_principle_candidate'
case=json.load(open(ROOT/'cases'/'schweinfurt.json'))
off=case['official_anchors']

MOTOR=[1.3,1.5,2.2,3,4,5.5,7.5,11,15,18.5,22,30,37,45,55,75,90,110,132,160,200,250,315,355,400,500,560,630]
TRF=[0.63,0.8,1.0,1.25,1.6,2.0,2.5,3.15,4.0]

def next_size(x,cat):
    for c in cat:
        if c>=x-1e-12:return float(c)
    return float(cat[-1])

def pump_kw(q_lps,head_m,eta_h=.72,eta_m=.94,rho=998.):
    return rho*9.81*(q_lps/1000)*head_m/(eta_h*eta_m)/1000

# ---------- Common evidence / road ----------
road=base.build_road_graph(base.load_elements('local_roads.json'))
# The municipality-scale experiment must use the frozen Phase-I evidence table,
# not an intermediate solver workspace.  A v2.3.0 packaging regression pointed
# this script at a stale, incomplete integrated-service ledger and made a clean
# rerun disagree with the released municipal outputs.
LEDGER_PATH=ROOT/'outputs'/'building_sector_demands.csv'
ledger=pd.read_csv(LEDGER_PATH)
required_ledger_columns={
    'building_id','lon','lat','service_eligible','wastewater_connected',
    'heat_eligible','wastewater_sanitary_m3_year',
    'wastewater_service_peak_lps','heat_candidate_mwh_year',
}
missing_ledger_columns=sorted(required_ledger_columns.difference(ledger.columns))
if missing_ledger_columns:
    raise KeyError(f'municipal evidence ledger missing columns: {missing_ledger_columns}')
if ledger.building_id.astype(str).duplicated().any():
    raise ValueError('municipal evidence ledger contains duplicate building IDs')

# ---------- Electricity: retain topology; calibrate actual site capacity scale ----------
tr=pd.read_csv(INTEG/'electricity_transformers.csv')
links=pd.read_csv(INTEG/'electricity_links.csv')
target=float(off['electricity_mv_lv_installed_capacity_mva'])
raw=tr.sn_mva/tr.sn_mva.sum()*target
sel=np.array([TRF[int(np.argmin(np.abs(np.array(TRF)-x)))] for x in raw],dtype=float)
# Upgrade the most-loaded sites until the public aggregate is closed to within one catalogue step.
while sel.sum()<target:
    opts=[]
    for i,s in enumerate(sel):
        j=TRF.index(float(s))
        if j<len(TRF)-1:
            inc=TRF[j+1]-s
            opts.append((float(tr.assigned_peak_mw.iloc[i]/max(s,1e-9))/inc,i,inc))
    if not opts: break
    _,i,inc=max(opts); sel[i]+=inc
tr2=tr.copy()
tr2['municipal_selected_capacity_mva']=sel
tr2['municipal_peak_loading_percent']=100*tr2.assigned_peak_mw/(0.96*tr2.municipal_selected_capacity_mva)
tr2['capacity_evidence']='catalogue allocation calibrated to published 216.71 MVA total; site locations remain synthetic'
electricity_capacity_total=float(sel.sum())
tr2.to_csv(OUT/'electricity_transformers_municipal.csv',index=False)
# Cable family labels and practical circuit representation.
links2=links.copy()
links2['cable_family']=np.where(links2.voltage_level.eq('MV'),'NA2XS2Y 12/20-kV Al-equivalent','NAYY/NA2XY 0.6/1-kV Al-equivalent')
links2['municipal_circuit_groups']=np.where(links2.voltage_level.eq('LV'),np.ceil(links2.parallel_circuits/2).astype(int),1)
links2['conductors_per_group']=np.where(links2.voltage_level.eq('LV'),np.minimum(links2.parallel_circuits,2),links2.parallel_circuits)
links2['design_note']=np.where((links2.voltage_level.eq('LV'))&(links2.parallel_circuits>2),'represented as multiple outgoing feeder circuits rather than >2 conductors in one feeder record','retained')
links2.to_csv(OUT/'electricity_links_municipal.csv',index=False)
links2,electricity_native_manifest=generate_accepted_municipal_electricity_network(
    OUT,INTEG/'electricity_pandapower.json',tr2,links2
)

# ---------- Drinking water: zone-first construction, native solve, bounded repair ----------
_,_,_,water_manifest=generate_accepted_municipal_water_network(
    OUT,road,ledger,
    wholesale_delivery_m3_year=float(off['drinking_water_wholesale_delivery_m3']),
)
water_zones=pd.read_csv(OUT/'drinking_water_zone_boundaries.csv')
qavg_total=float(off['drinking_water_total_delivery_m3'])/365/86400*1000
water_peak_factor=float(case['demand_model']['drinking_water_peak_factor'])
qpeak_total=water_peak_factor*qavg_total
source_zone=water_zones.sort_values('pressure_zone_id').iloc[0]
head=float(water_zones.selected_boundary_head_m.max()-source_zone.ground_elevation_m)
per_pump_q=qavg_total
p_duty=pump_kw(per_pump_q,head,.72,.94)
water_motor=next_size(1.10*p_duty,MOTOR)
water_peak_duty_units=int(math.ceil(water_peak_factor-1e-12))
water_installed_units=water_peak_duty_units+1
water_station=pd.DataFrame([{
    'map_id':'W1','facility':'Wasserwerk Schweinfurt source/booster station','source_role':'aggregated municipal source station; individual wells are not separately motor-rated',
    'annual_delivery_m3':off['drinking_water_total_delivery_m3'],'direct_retail_m3':off['drinking_water_annual_m3'],'wholesale_boundary_m3':off['drinking_water_wholesale_delivery_m3'],
    'average_total_flow_lps':qavg_total,'design_peak_flow_lps':qpeak_total,'design_peak_factor':water_peak_factor,'design_head_m':head,
    'pump_arrangement':f'{water_installed_units} identical VFD pumps: {water_peak_duty_units} duty at design peak + 1 standby',
    'per_pump_design_flow_lps':per_pump_q,'per_pump_operating_power_kw':p_duty,'selected_motor_kw':water_motor,
    'installed_units':water_installed_units,'installed_nameplate_kw':water_installed_units*water_motor,'firm_nameplate_kw_with_one_unavailable':water_peak_duty_units*water_motor,
    'main_length_km':water_manifest['main_length_km'],'pressure_zones':water_manifest['pressure_zones'],
    'source_coordinate_lon':source_zone.lon,'source_coordinate_lat':source_zone.lat,
    'native_average_pressure_range_m':f"{water_manifest['average_minimum_service_pressure_m']:.2f}-{water_manifest['average_maximum_service_pressure_m']:.2f}",
    'native_peak_pressure_range_m':f"{water_manifest['peak_minimum_service_pressure_m']:.2f}-{water_manifest['peak_maximum_service_pressure_m']:.2f}",
    'native_peak_maximum_velocity_m_s':water_manifest['peak_maximum_velocity_m_s'],
    'status':'accepted municipality-scale EPANET network; pump curves/NPSH/surge remain manufacturer-design checks'
}])
water_station.to_csv(OUT/'drinking_water_station_municipal.csv',index=False)

# ---------- Wastewater: 13 municipal pumpworks, practical intermittent duties ----------
st=pd.read_csv(PRINC/'wastewater_lift_station_compounds.csv')
# allocate registered-building peak sanitary inflow to nearest published-count station compound
sx=np.array([base.xy_km((x,y)) for x,y in zip(st.lon,st.lat)])
sewer_ledger=ledger[ledger.wastewater_connected.astype(bool)].copy()
bx=np.array([base.xy_km((x,y)) for x,y in zip(sewer_ledger.lon,sewer_ledger.lat)])
nearest=[]
for i in range(0,len(bx),3000):
    d=((bx[i:i+3000,None,:]-sx[None,:,:])**2).sum(2)
    nearest.extend(np.argmin(d,axis=1))
ledger2=sewer_ledger.copy(); ledger2['station_index']=nearest
agg=ledger2.groupby('station_index').agg(buildings=('building_id','count'),annual_sanitary_m3=('wastewater_sanitary_m3_year','sum'),design_peak_lps=('wastewater_service_peak_lps','sum'))
st=st.join(agg)
# force-main length distribution closes the official 13 km inventory
weights=np.sqrt(st.transition_count.clip(lower=1)); st['force_main_km']=float(off['wastewater_force_main_km'])*weights/weights.sum()
outfall=(10.2115927,50.0257927); zout=base.proxy_elevation_m(outfall)
rows=[]
for i,r in st.iterrows():
    q_catch=float(r.design_peak_lps) if pd.notna(r.design_peak_lps) else 50.0 # zero-building compound treated as storm/relief station
    role='sanitary/combined lift' if pd.notna(r.design_peak_lps) else 'storm/relief lift (no registered-building sanitary catchment)'
    # Municipal sewage pumps cycle from a wet well; on-state flow must also preserve a cleansing velocity.
    # Use DN100 as the minimum raw-sewage pressure-main class and 0.6 m/s as the minimum screening velocity.
    q_clean_dn100_lps=0.6*math.pi*(0.100**2)/4*1000
    q=max(q_catch,q_clean_dn100_lps)
    # choose force main around 0.8-1.0 m/s, never below DN100 for municipal raw sewage screening
    required=math.sqrt(4*(q/1000)/(math.pi*1.0))*1000
    dns=[100,125,150,200,250,300,350,400,500]
    dn=next((d for d in dns if d>=required),500)
    vel=(q/1000)/(math.pi*(dn/1000)**2/4)
    z=base.proxy_elevation_m((float(r.lon),float(r.lat)))
    static=max(0.0,zout-z)+2.0
    tdh=max(6.0,static+0.8*float(r.force_main_km))
    duty=pump_kw(q,tdh,.62,.92)
    motor=next_size(max(1.15*duty,3.0),MOTOR) # municipal raw-sewage floor; not property sump scale
    rows.append({
        'map_id':f'S{i+1}','station_id':r.station_id,'lon':r.lon,'lat':r.lat,'station_role':role,
        'registered_buildings':0 if pd.isna(r.buildings) else int(r.buildings),'annual_sanitary_m3':0 if pd.isna(r.annual_sanitary_m3) else r.annual_sanitary_m3,
        'catchment_peak_flow_lps':q_catch,'design_pump_flow_lps':q,'force_main_dn_mm':dn,'force_main_velocity_m_s':vel,'force_main_length_km':r.force_main_km,
        'screening_tdh_m':tdh,'operating_power_kw':duty,'selected_motor_kw':motor,
        'active_swmm_pump_objects':1,'installed_pump_units':2,'standby_pump_units':1,
        'arrangement':'2 identical pumps (1 duty + 1 standby)','installed_nameplate_kw':2*motor,
        'evidence_role':'13 station count and 13-km pressure-main total are public inputs; locations and individual duties are synthetic'
    })
ww=pd.DataFrame(rows)
ww.to_csv(OUT/'wastewater_pump_stations_municipal.csv',index=False)
# actual treatment facility public scale
wwtp=pd.DataFrame([{
    'map_id':'S0','facility':'Klaerwerk Schweinfurt','design_population_equivalent':250000,'current_load_PE_range':'160000-220000',
    'dry_weather_m3_day':19000,'wet_weather_m3_day':80000,'annual_flow_m3':9000000,
    'average_electricity_kwh_day':10500,'average_gross_process_power_kw':10500/24,
    'onsite_chp':'2 x 600 kW BHKW','onsite_generation_capacity_kw':1200,
    '2025_self_generation_statement':'100% own-energy generation reported by operator',
    'role':'gross process load and onsite generation reported separately; not one pump'
}])
wwtp.to_csv(OUT/'wastewater_treatment_facility_municipal.csv',index=False)
for f in ['wastewater_nodes.csv','wastewater_links.csv','wastewater_service_connections.csv','wastewater_lift_station_compounds.csv']:
    shutil.copy2(PRINC/f,OUT/f)
# The retained 6,179-manhole graph is the spatial design layer.  Its 815
# sanitary catchments are now consolidated into exactly 13 explicit
# wet-well/pump/force-main systems and executed in SWMM.
_,_,native_ww_pumps,wastewater_native_manifest=generate_accepted_municipal_wastewater_station_model(
    OUT,road,PRINC
)
native_ww_pumps=native_ww_pumps.set_index('station_id')
terminal_map=pd.read_csv(OUT/'wastewater_station_subcatchment_mapping.csv')
terminal_agg=terminal_map.groupby('station_index').agg(
    registered_buildings=('building_count','sum'),
    annual_sanitary_m3=('annual_m3','sum'),
)
for i,row in ww.iterrows():
    native=native_ww_pumps.loc[row.station_id]
    ww.at[i,'registered_buildings']=int(terminal_agg.loc[i,'registered_buildings']) if i in terminal_agg.index else 0
    ww.at[i,'annual_sanitary_m3']=float(terminal_agg.loc[i,'annual_sanitary_m3']) if i in terminal_agg.index else 0.0
    ww.at[i,'catchment_peak_flow_lps']=float(native.design_flow_m3s)*1000.0
    ww.at[i,'design_pump_flow_lps']=float(native.design_flow_m3s)*1000.0
    ww.at[i,'force_main_dn_mm']=float(native.force_main_dn_mm)
    ww.at[i,'force_main_velocity_m_s']=float(native.force_main_velocity_m_s)
    ww.at[i,'screening_tdh_m']=float(native.design_head_m)
    duty=pump_kw(float(native.design_flow_m3s)*1000.0,float(native.design_head_m),.62,.92)
    motor=next_size(max(1.15*duty,3.0),MOTOR)
    ww.at[i,'operating_power_kw']=duty
    ww.at[i,'selected_motor_kw']=motor
    ww.at[i,'installed_nameplate_kw']=2*motor
    ww.at[i,'station_role']='sanitary lift' if ww.at[i,'annual_sanitary_m3']>0 else 'storm/relief lift'
ww.to_csv(OUT/'wastewater_pump_stations_municipal.csv',index=False)

# ---------- District heat: documented multi-source municipal network with real temperature envelopes ----------
att=attach_buildings(ledger[ledger.heat_eligible.astype(bool)],road)
xy=np.array([base.xy_km((x,y)) for x,y in zip(att.lon,att.lat)])
att['cell_x']=np.floor(xy[:,0]/0.25).astype(int);att['cell_y']=np.floor(xy[:,1]/0.25).astype(int)
cellheat=att.groupby(['cell_x','cell_y']).heat_candidate_mwh_year.sum()/6.25
att=att.join(cellheat.rename('heat_density_mwh_ha'),on=['cell_x','cell_y'])

source_defs=case['district_heating']['documented_supply_boundaries']
source_nodes={}; source_dist={}; source_paths={}
idx=base.NodeIndex(road.nodes)
for sd in source_defs:
    node=idx.nearest(tuple(sd['coordinate']))
    source_nodes[sd['id']]=node
    dist,paths=nx.single_source_dijkstra(road,node,weight='length_km')
    source_dist[sd['id']]=dist; source_paths[sd['id']]=paths
for sid in source_nodes:
    att[f'distance_{sid}_km']=att.road_node.map(lambda n,sid=sid:source_dist[sid].get(n,1e6))
att['distance_advantage_H2_km']=att['distance_H1_km']-att['distance_H2_km']
att['assigned_source_distance_km']=att[['distance_H1_km','distance_H2_km']].min(axis=1)

heat_full_load_hours=float(case['district_heating']['equivalent_full_load_hours'])

def assign_hydraulic_sources(df, h2_peak_share_cap):
    '''Keep H1 as the primary source and use H2 only as a bounded northern peak-support injection.'''
    x=df.copy(); x['hydraulic_source_id']='H1'
    eligible=x[x.distance_advantage_H2_km>0].sort_values('distance_advantage_H2_km',ascending=False)
    proxy_peak=x.heat_candidate_mwh_year/heat_full_load_hours
    cap=float(proxy_peak.sum())*h2_peak_share_cap
    used=0.0; ids=[]
    for i,r in eligible.iterrows():
        p=float(r.heat_candidate_mwh_year/heat_full_load_hours)
        if used+p>cap and ids: continue
        ids.append(i); used+=p
        if used>=cap*0.98: break
    x.loc[ids,'hydraulic_source_id']='H2'
    x['assigned_source_distance_km']=x.apply(lambda r:r[f"distance_{r.hydraulic_source_id}_km"],axis=1)
    return x

# Public length is a calibration input. Search a small predeclared policy set and retain the closest candidate.
target_n=int(off['district_heat_sales_contracts']); target_len=float(off['district_heat_route_km']); target_energy=float(off['district_heat_sales_mwh_year'])
h2_proxy_cap=float(case['district_heating']['h2_proxy_peak_share_cap'])
# mandatory intertie between documented H1/H2 injection boundaries
intertie_path=nx.shortest_path(road,source_nodes['H1'],source_nodes['H2'],weight='length_km')
intertie_edges=[]; intertie_len=0.0
for u,v in zip(intertie_path[:-1],intertie_path[1:]):
    key=(u,v) if u<=v else (v,u); intertie_edges.append(key); intertie_len+=float(road[u][v]['length_km'])

best=None
search_rows=[]
for thr in case['district_heating']['heat_inventory_search_thresholds_mwh_ha']:
  cand=att[att.heat_density_mwh_ha>=float(thr)].copy()
  if len(cand)<target_n: continue
  for alpha in case['district_heating']['heat_inventory_search_score_exponents']:
    cand['route_score']=(cand.heat_candidate_mwh_year**float(alpha))/(cand.assigned_source_distance_km+cand.service_length_km+0.05)
    sel0=assign_hydraulic_sources(cand.nlargest(target_n,'route_score').copy(),h2_proxy_cap)
    used=set(intertie_edges)
    for row in sel0.itertuples():
        sid=row.hydraulic_source_id; path=source_paths[sid][row.road_node]
        for u,v in zip(path[:-1],path[1:]): used.add((u,v) if u<=v else (v,u))
    route=sum(float(road[u][v]['length_km']) for u,v in used)
    service=float(sel0.service_length_km.sum()); comp=route+service
    shares=sel0.hydraulic_source_id.value_counts().to_dict()
    score=abs(comp-target_len)/target_len
    search_rows.append({'threshold_mwh_ha':thr,'score_exponent':alpha,'route_km':route,'service_km':service,'comparable_network_km':comp,'relative_error':(comp-target_len)/target_len,'H1_customers':shares.get('H1',0),'H2_customers':shares.get('H2',0),'H2_proxy_peak_share_cap':h2_proxy_cap})
    # both documented boundaries must participate in the hydraulic candidate
    if shares.get('H1',0)<25 or shares.get('H2',0)<25: continue
    if best is None or score<best[0]: best=(score,thr,alpha,sel0,used,route,service)
if best is None: raise RuntimeError('No two-source district-heating candidate found')
_,best_thr,best_alpha,sel,used,route_length,service_length=best
pd.DataFrame(search_rows).to_csv(OUT/'district_heating_inventory_search.csv',index=False)
sel=sel.copy()
sel['annual_heat_mwh']=sel.heat_candidate_mwh_year/sel.heat_candidate_mwh_year.sum()*target_energy
sel['peak_heat_mw']=sel.annual_heat_mwh/float(case['district_heating']['equivalent_full_load_hours'])

# Apply published Schweinfurt temperature envelopes as design classes.
profiles=case['district_heating']['published_temperature_classes']
def temp_class(density):
    if density>=1200:return 'HT120'
    if density>=900:return 'MT110'
    return 'LT85'
sel['temperature_class']=sel.heat_density_mwh_ha.map(temp_class)
sel['design_supply_temperature_c']=sel.temperature_class.map(lambda c:profiles[c]['max_supply_c'])
sel['minimum_supply_temperature_c']=sel.temperature_class.map(lambda c:profiles[c]['min_supply_c'])
sel['design_return_temperature_c']=sel.temperature_class.map(lambda c:profiles[c]['max_return_c'])
sel['design_delta_t_k']=sel.design_supply_temperature_c-sel.design_return_temperature_c
sel['design_pressure_class']=sel.temperature_class.map(lambda c:profiles[c]['pressure_class'])
sel['design_max_handover_dp_bar']=sel.temperature_class.map(lambda c:profiles[c]['max_handover_dp_bar'])
sel['design_mass_flow_kg_s']=sel.peak_heat_mw*1e6/(4180*sel.design_delta_t_k)

# collect edge duties, mass flows, temperature requirement and source coverage
edge_peak={}; edge_mass={}; edge_temp={}; edge_pn={}; edge_sources={}; edge_role={}
pn_rank={'PN10':10,'PN16':16,'PN25':25}
for row in sel.itertuples():
    sid=row.hydraulic_source_id; path=source_paths[sid][row.road_node]
    for u,v in zip(path[:-1],path[1:]):
        key=(u,v) if u<=v else (v,u)
        edge_peak[key]=edge_peak.get(key,0.0)+float(row.peak_heat_mw)
        edge_mass[key]=edge_mass.get(key,0.0)+float(row.design_mass_flow_kg_s)
        edge_temp[key]=max(edge_temp.get(key,0.0),float(row.design_supply_temperature_c))
        cur=edge_pn.get(key,'PN10')
        edge_pn[key]=row.design_pressure_class if pn_rank[row.design_pressure_class]>pn_rank[cur] else cur
        edge_sources.setdefault(key,set()).add(sid); edge_role[key]='distribution'
# intertie is sized for H2 assigned peak as contingency transfer from the main system
h2=sel[sel.hydraulic_source_id.eq('H2')]
h2_peak=float(h2.peak_heat_mw.sum()); h2_mass=float(h2.design_mass_flow_kg_s.sum())
for key in intertie_edges:
    edge_peak[key]=max(edge_peak.get(key,0.0),h2_peak)
    edge_mass[key]=max(edge_mass.get(key,0.0),h2_mass)
    edge_temp[key]=max(edge_temp.get(key,0.0),110.0)
    edge_pn[key]='PN25'
    edge_sources.setdefault(key,set()).update(['H1','H2']); edge_role[key]='source_intertie'

hrows=[]
for j,key in enumerate(sorted(used),1):
    u,v=key; dat=road[u][v]
    mass=edge_mass.get(key,0.0)
    dn=choose_heat_dn(max(mass,0.01),maximum_velocity_m_s=1.5,maximum_pressure_gradient_pa_m=100.0)
    geom=dat.get('geometry',[[u[0],u[1]],[v[0],v[1]]])
    if not isinstance(geom,list): geom=[[u[0],u[1]],[v[0],v[1]]]
    hrows.append({'corridor_id':f'H_R{j:05d}','from_lon':u[0],'from_lat':u[1],'to_lon':v[0],'to_lat':v[1],'length_km':dat['length_km'],'design_peak_mw_th':edge_peak.get(key,0.0),'design_mass_flow_kg_s':mass,'required_supply_temperature_c':edge_temp.get(key,85.0),'design_return_temperature_c':50.0,'required_pressure_class':edge_pn.get(key,'PN16'),'diameter_mm':dn,'edge_role':edge_role.get(key,'distribution'),'source_coverage':'+'.join(sorted(edge_sources.get(key,set()))),'geometry_json':json.dumps(geom,separators=(',',':'))})
hcorr=pd.DataFrame(hrows); hcorr.to_csv(OUT/'district_heating_corridors_municipal.csv',index=False)
hsvc=sel[['building_id','lon','lat','road_lon','road_lat','service_length_km','annual_heat_mwh','peak_heat_mw','heat_density_mwh_ha','hydraulic_source_id','assigned_source_distance_km','temperature_class','design_supply_temperature_c','minimum_supply_temperature_c','design_return_temperature_c','design_delta_t_k','design_pressure_class','design_max_handover_dp_bar','design_mass_flow_kg_s']].copy();hsvc.to_csv(OUT/'district_heating_services_municipal.csv',index=False)

# source-level hydraulic screening and documented plant-side evidence
source_rows=[]
for sd in source_defs:
    sid=sd['id']; ss=sel[sel.hydraulic_source_id.eq(sid)]
    peak=float(ss.peak_heat_mw.sum()); mdot=float(ss.design_mass_flow_kg_s.sum()); q_m3s=mdot/985.0; q_m3h=q_m3s*3600
    critical=float(ss[f'distance_{sid}_km'].max()) if len(ss) else 0.0
    dh_screen=case['practical_design_envelopes']['district_heating']
    pressure_gradient=float(dh_screen.get('source_pump_screening_pressure_gradient_pa_m',50.0))
    local_loss=float(dh_screen.get('local_loss_factor',1.3))
    terminal_dp=max(float(x) for x in dh_screen.get('terminal_differential_pressure_bar',[0.3,0.5]))
    system_dp_bar=terminal_dp+(2*critical*1000*pressure_gradient*local_loss)/1e5
    head_m=system_dp_bar*1e5/(985*9.81)
    full_power=985*9.81*q_m3s*head_m/(.78*.94)/1000
    # two 50%-flow duty pumps + one standby; apply the declared motor reserve and municipal floor
    per_duty=full_power/2
    margin=float(dh_screen.get('motor_nameplate_margin_fraction',0.15))
    motor=next_size(max((1.0+margin)*per_duty,45.0),MOTOR)
    source_rows.append({'map_id':sid,'source_name':sd['name'],'lon':sd['coordinate'][0],'lat':sd['coordinate'][1],'source_role':sd['role'],'address':sd['address'],'plant_side_steam_bar':sd['plant_side_steam_bar'],'plant_side_steam_c':sd['plant_side_steam_c'],'documented_fuel_input_mw_total':sum(sd['documented_fuel_input_mw']),'documented_fuel_input_units_mw':'+'.join(str(x) for x in sd['documented_fuel_input_mw']),'capacity_interpretation':sd['capacity_note'],'assigned_hydraulic_customers':len(ss),'assigned_peak_mw_th':peak,'assigned_annual_heat_gwh':float(ss.annual_heat_mwh.sum()/1000),'design_mass_flow_kg_s':mdot,'design_volume_flow_m3h':q_m3h,'critical_one_way_path_km':critical,'screening_pressure_gradient_pa_m':pressure_gradient,'screening_local_loss_factor':local_loss,'terminal_dp_bar':terminal_dp,'screening_system_dp_bar':system_dp_bar,'maximum_screening_pump_lift_bar':float(dh_screen.get('maximum_circulation_pump_lift_bar',10.0)),'pump_lift_screen_pass':bool(system_dp_bar<=float(dh_screen.get('maximum_circulation_pump_lift_bar',10.0))),'full_peak_circulation_power_kw':full_power,'pump_arrangement':'3 identical VFD circulation pumps; 2 x 50% duty at peak + 1 standby','per_duty_pump_operating_kw':per_duty,'motor_nameplate_margin_fraction':margin,'selected_motor_kw_each':motor,'installed_pump_nameplate_kw':3*motor,'firm_nameplate_kw_with_one_unavailable':2*motor,'status':'documented supply boundary with synthetic hydraulic allocation; plant dispatch/nameplate heat export not inferred'})
heat_sources=pd.DataFrame(source_rows); heat_sources.to_csv(OUT/'district_heating_sources_municipal.csv',index=False)
# Execute the municipality-scale supply network in pandapipes.  The absolute
# slack is a numerical reference only; release uses pressure-drop, velocity,
# pressure-gradient and source-pump differential-pressure limits.
hcorr,heat_native_manifest=generate_accepted_municipal_heat_network(
    OUT,hcorr,hsvc,heat_sources
)
# backwards-compatible alias for tools expecting old filename
heat_sources.to_csv(OUT/'district_heating_source_municipal.csv',index=False)

# Official profile catalogue used by the synthetic temperature-class adapter.
prof_rows=[]
for cid,pf in profiles.items():
    prof_rows.append({'temperature_class':cid,'basis':pf['basis'],'max_supply_c':pf['max_supply_c'],'min_supply_c':pf['min_supply_c'],'max_return_c':pf['max_return_c'],'dhw_return_short_c':'/'.join(str(x) for x in pf['dhw_return_short_c']),'min_handover_dp_bar':'-'.join(str(x) for x in pf['min_handover_dp_bar']),'max_handover_dp_bar':pf['max_handover_dp_bar'],'pressure_class':pf['pressure_class']})
pd.DataFrame(prof_rows).to_csv(OUT/'district_heating_temperature_profiles.csv',index=False)
sheet_urls=case['district_heating'].get('published_operating_sheet_urls',{})
published_area_profiles=pd.DataFrame([
    {'area':'B1/B2','sheet_date':'01.05.2021','max_possible_c':130.0,'max_supply_c':120.0,'min_supply_c':70.0,'max_normal_return_c':50.0,'short_dhw_return_c':'60/65','min_handover_dp_bar':'0.3-0.5','max_handover_dp_bar':4.0,'pressure_class':'PN10','evidence_role':'observed operating envelope','source_url':sheet_urls.get('B1/B2','')},
    {'area':'F1-Sued','sheet_date':'01.10.2023','max_possible_c':130.0,'max_supply_c':120.0,'min_supply_c':70.0,'max_normal_return_c':50.0,'short_dhw_return_c':'60/65','min_handover_dp_bar':'0.3-0.5','max_handover_dp_bar':9.0,'pressure_class':'PN25','evidence_role':'observed operating envelope','source_url':sheet_urls.get('F1-Sued','')},
    {'area':'Maintal','sheet_date':'06.02.2025','max_possible_c':110.0,'max_supply_c':110.0,'min_supply_c':70.0,'max_normal_return_c':50.0,'short_dhw_return_c':'60/65','min_handover_dp_bar':'0.3-0.5','max_handover_dp_bar':4.0,'pressure_class':'PN16','evidence_role':'observed operating envelope','source_url':sheet_urls.get('Maintal','')},
    {'area':'V6','sheet_date':'01.05.2021','max_possible_c':95.0,'max_supply_c':85.0,'min_supply_c':65.0,'max_normal_return_c':50.0,'short_dhw_return_c':'60/65','min_handover_dp_bar':'0.3-0.5','max_handover_dp_bar':2.0,'pressure_class':'PN16','evidence_role':'observed operating envelope','source_url':sheet_urls.get('V6','')}
])
published_area_profiles.to_csv(OUT/'district_heating_published_area_profiles.csv',index=False)

# Reachability/connectivity audit: one active hot-water graph, two documented injection nodes.
HG=nx.Graph(); HG.add_edges_from(used)
components=list(nx.connected_components(HG)); source_node_set=set(source_nodes.values())
source_components=sum(1 for comp in components if comp & source_node_set)
all_reachable=True
for r in sel.itertuples():
    if not nx.has_path(HG,r.road_node,source_nodes[r.hydraulic_source_id]): all_reachable=False; break
reach=pd.DataFrame([{'active_network_components':len(components),'components_with_documented_source':source_components,'documented_supply_boundaries':len(source_defs),'customer_substations':len(sel),'source_reachable_substations':len(sel) if all_reachable else None,'source_reachability_fraction':1.0 if all_reachable else 0.0,'H1_customers':int((sel.hydraulic_source_id=='H1').sum()),'H2_customers':int((sel.hydraulic_source_id=='H2').sum()),'H2_proxy_peak_share_cap':h2_proxy_cap,'H2_assigned_peak_share':float(sel.loc[sel.hydraulic_source_id.eq('H2'),'peak_heat_mw'].sum()/sel.peak_heat_mw.sum()),'intertie_length_km':intertie_len,'route_km':route_length,'service_km':service_length,'comparable_network_km':route_length+service_length,'inventory_relative_error':(route_length+service_length-target_len)/target_len,'selected_heat_density_threshold_mwh_ha':best_thr,'selected_route_score_exponent':best_alpha,'HT120_substations':int((sel.temperature_class=='HT120').sum()),'MT110_substations':int((sel.temperature_class=='MT110').sum()),'LT85_substations':int((sel.temperature_class=='LT85').sum())}])
reach.to_csv(OUT/'district_heating_connectivity_audit.csv',index=False)

# ---------- summary ----------
summary=pd.DataFrame([
 {'sector':'Electricity','municipal_topology':'105 MV/LV sites; radial pandapower network accepted','demand_or_flow':'40.962 MW coincident peak; 156.851 GWh/a','source_or_pumps':'20-kV external-grid boundary; no generator invented','practical_rating':f"{electricity_capacity_total:.2f} MVA portfolio; Vmin {electricity_native_manifest['minimum_voltage_pu']:.3f} pu; max line/transformer loading {electricity_native_manifest['maximum_line_loading_percent']:.1f}/{electricity_native_manifest['maximum_transformer_loading_percent']:.1f}%", 'public_scale':'216.71 MVA; 319.4 km MV cable; 809.12 km LV cable'},
 {'sector':'Drinking water','municipal_topology':f"4 pressure zones; {water_manifest['main_length_km']:.1f}-km generated main network; EPANET accepted",'demand_or_flow':f'{qavg_total:.1f}/{qpeak_total:.1f} L/s average/peak total delivery','source_or_pumps':f'W1 waterworks plus explicit controlled zone boundaries; {water_installed_units} source pumps including standby','practical_rating':f"{water_installed_units} x {water_motor:.0f} kW motors; native pressure {water_manifest['peak_minimum_service_pressure_m']:.1f}-{water_manifest['average_maximum_service_pressure_m']:.1f} m; peak velocity {water_manifest['peak_maximum_velocity_m_s']:.2f} m/s",'public_scale':'5.6 million m3/a total delivery; 341 km mains'},
 {'sector':'Wastewater','municipal_topology':'6,179-manhole spatial layer plus accepted 815-catchment/13-station SWMM hydraulic layer','demand_or_flow':f"{wastewater_native_manifest['sanitary_inflow_m3_year']/1e6:.3f} million m3/a registered sanitary inflow; WWTP 19,000/80,000 m3/d dry/wet",'source_or_pumps':'13 wet wells, 13 active SWMM pump objects, 26 installed duty/standby pump units, 13 force mains + treatment works','practical_rating':f"{ww.selected_motor_kw.min():.0f}-{ww.selected_motor_kw.max():.0f} kW station motors; SWMM continuity {wastewater_native_manifest['continuity_error_percent']:.3f}%, flooding {wastewater_native_manifest['flooding_loss_percent']:.3f}%",'public_scale':'13 pumpworks; 236 km gravity sewers plus 13 km pressure mains; 9 million m3/a WWTP'},
 {'sector':'District heating','municipal_topology':'two documented injection boundaries linked by one source-reachable pandapipes-accepted network','demand_or_flow':f'{sel.peak_heat_mw.sum():.3f} MWth peak; 87.5 GWh/a','source_or_pumps':'H1 primary GKS + H2 documented peak/backup injection; source-level duty/assist/standby circulation screening','practical_rating':f"{heat_native_manifest['maximum_native_velocity_m_s']:.2f} m/s; {heat_native_manifest['maximum_catalogue_pressure_gradient_pa_m']:.1f} Pa/m; {heat_native_manifest['maximum_source_screening_differential_pressure_bar']:.2f} bar critical dp; {100*heat_native_manifest['annual_heat_loss_fraction']:.1f}% heat loss",'public_scale':f'{target_len:.1f} km network; 87.5 GWh/a; 841 contract equivalents'}
])
summary.to_csv(OUT/'municipal_scale_summary.csv',index=False)
manifest={'version':'2.5.0','purpose':'accepted municipality-scale practical topology with deterministic build-solve-repair-release logic','input_building_ledger':str(LEDGER_PATH.relative_to(ROOT)),'service_eligible_buildings':int(ledger.service_eligible.astype(bool).sum()),'registered_wastewater_buildings':len(sewer_ledger),'electricity_native_acceptance':electricity_native_manifest,'heat_sources':heat_sources.to_dict(orient='records'),'heat_connectivity':reach.iloc[0].to_dict(),'heat_native_acceptance':heat_native_manifest,'water_station':water_station.iloc[0].to_dict(),'water_native_acceptance':water_manifest,'wastewater_native_acceptance':wastewater_native_manifest,'wastewater_station_count':len(ww),'electricity_transformer_capacity_mva':float(tr2.municipal_selected_capacity_mva.sum()),'release_gate_passed':bool(electricity_native_manifest['release_gate_passed'] and water_manifest['release_gate_passed'] and wastewater_native_manifest['release_gate_passed'] and heat_native_manifest['release_gate_passed'] and reach.iloc[0].source_reachability_fraction==1.0),'caveat':'Known public source, temperature, pressure, count and aggregate-scale evidence are inputs. Confidential installed edge geometry, exact named supply-area boundaries, plant dispatch and nameplates are not reconstructed.'}
json.dump(manifest,open(OUT/'manifest.json','w'),indent=2,default=float)
print(summary.to_string(index=False))
print('\nHeat sources:',heat_sources.to_string(index=False))
print('\nWastewater motors:',ww[['map_id','design_pump_flow_lps','screening_tdh_m','force_main_dn_mm','selected_motor_kw']].to_string(index=False))
