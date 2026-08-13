import pandas as pd
import geopandas as gpd
import pickle
from core.common import cpath
from services.geocode import crs_ca
from services.emissions import compute_on_road_emissions

with open(cpath('model_cache_dir', 'distancesj.pickle'), 'rb') as pfile:
    distancesj = pickle.load(pfile)
tmp=pd.read_csv('./rdrs_geocodes_crindt - OD flows by material type.csv',skiprows=1).clean_names()

tmp2=tmp.filt(lambda x: x.fillna(False).ismrf).assign(o_rdrsid=lambda x: 'RD'+x.o.astype(int).astype(str),d_rdrsid=lambda x: 'RD'+x.d.astype(int).astype(str))

tmp3=tmp2.join(distancesj.set_index(['o_rdrsid','d_rdrsid']),on=['o_rdrsid','d_rdrsid'],how='left')

tmp4=(compute_on_road_emissions(gpd.GeoDataFrame(tmp3,geometry='geometry_clip').set_crs(crs_ca).assign(
            clip_distance=lambda x: x.geometry_clip.length).assign(trips=1,EMFAC_class='T7 Tractor Class 8'),model_year=2021))

# %%
tmp4.groupby(['o_rdrsid','d_rdrsid']).agg({
    'origin_mrf':'first',
    'destination':'first',
    'dest_type':'first',
    'cardboard':'first',
    'cd':'first',
    'ferrous':'first',
    'glass':'first',
    'hdpe':'first',
    'mixed_metal':'first',
    'mixed_plastic':'first',
    'msw':'first',
    'nonferrous':'first',
    'organics':'first',
    'other':'first',
    'paper':'first',
    'pet':'first',
    'singlestream':'first',
    'clip_distance':'sum',
    'trips':'first',
    'EMFAC_class':'first',
    'vmt':'sum',
    'emiss_pm25':'sum',
    'emiss_nox':'sum',
    'emiss_co':'sum',
    'emiss_co2':'sum',
    'emiss_n2o':'sum',
    'emiss_ghg':'sum',
    'fuel_consumption':'sum',
    'geometry_merged_orig':'first',
    'geometry_merged_dest':'first'
}).to_csv('tmp.csv')

optdat=pd.read_csv('rdrs_geocodes_crindt - SAMPLE DATA.csv')

mats=['cardboard','cd','ferrous','glass','hdpe','mixed_metal','mixed_plastic','msw',
      'nonferrous','organics','other','paper','pet','singlestream']
optdat_long = (
    optdat.melt(
        id_vars=[col for col in optdat.columns 
                if col not in mats], 
        value_vars=mats,
        var_name='material', value_name='tonssent')
    .filt(lambda x: ~x.tonssent.isna() & (x.tonssent > 0))
    .assign(
        trips=lambda x: x.tonssent / 20,  # Assuming 20 tons per trip

    )
)
optdat_long.to_csv('optdat_long.csv',index=False)

used_rdrsid=pd.concat([
    optdat_long[['o_rdrsid']].rename(columns={'o_rdrsid':'rdrs_id'}),
    optdat_long[['d_rdrsid']].rename(columns={'d_rdrsid':'rdrs_id'}),
    ])

# %%
import importlib
importlib.reload(importlib.import_module('RDRS'))
from layers.rdrs import read_rdrs_entities_swis_expanded, read_swis
from core.common import case_when
swis=read_swis()
rdrs_ent=read_rdrs_entities_swis_expanded()
swis_cap=(
    rdrs_ent.join(swis[0],on='swis_number')
    [['rdrs_id','reporting_entity_activity','swis_number','operationalstatus_act','throughput_act','throughputunits_act','capacity_act','capacityunits_act']]
    .sort_values(['rdrs_id','capacity_act','throughput_act'],ascending=[True,False,False]).filt(lambda x: ~x.throughput_act.isna() | ~x.capacity_act.isna())
    .groupby(['rdrs_id','swis_number']).first()
    .assign(
        capacity_model_raw=lambda x: case_when(
            x.capacityunits_act.fillna('').str.match('.*per'), x.capacity_act,
            x.throughputunits_act.fillna('').str.match('.*per'), x.throughput_act
        ),
        capacityunits_model=lambda x: case_when(
            x.capacityunits_act.fillna('').str.match('.*per'), x.capacityunits_act,
            x.throughputunits_act.fillna('').str.match('.*per'), x.throughputunits_act,
        ),
        capacity_src=lambda x: case_when(
            x.capacityunits_act.fillna('').str.match('.*per'), 'CAPACITY',
            x.throughputunits_act.fillna('').str.match('.*per'), 'THROUGHPUT',
        ),
        capacity_amt=lambda x: x.capacityunits_model.fillna('').str.replace(r'^(.*) per (\w+)$',r'\1',regex=True).str.lower(),
        capacity_per=lambda x: x.capacityunits_model.fillna('').str.replace(r'^(.*) per (\w+)$',r'\2',regex=True).str.lower(),
    )
)

#%%

rdrs_capacity=(
    used_rdrsid.groupby('rdrs_id').first()[[]].reset_index()
    .join(swis_cap.reset_index().set_index('rdrs_id'),on='rdrs_id')
)

# legacy version takes first swis record
rdrs_capacity.groupby('rdrs_id').first().to_csv('rdrs_capacity.csv',index=False)

# new version reports all swis records per RDRSid
rdrs_capacity.to_csv('rdrs_capacity_expanded.csv',index=False)
# %%
