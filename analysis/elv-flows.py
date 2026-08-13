#!/usr/bin/env python
# coding: utf-8

import typer
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry
from IPython.display import display
import janitor 
import os
from importlib import reload
from layers.appliance import compute_diverted_appliances
from services.geocode import crs_ca, crs_ll
from core.common import cpath, ureg, case_when, wdisplay, wldisplay
import layers.appliance
from layers.rdrs import collect_rdrs_ods, geocode_entities, aggregate_all_geometries, merge_aggregated_geometries
from services.geocode import get_ca_port_endpoints
from layers.appliance import (
    get_used_zips, read_zip_car_mapping_mon, plot_spatial_mapping, 
    read_lf_car_mon, read_car_data_raw, read_diverted_appliances_mon,
    read_car_shred_mapping_mon, read_class1_landfill_endpoints, read_car_landfill1_mapping_mon, read_ctmsr_landfill_endpoints
)
from functools import reduce

from services.routing import compute_nearest_neighbor_mapping


app = typer.Typer()

lb_to_tonne = ureg('lb').to('tonne').magnitude

# material composition estimates
def read_material_composition(mcfile='material_composition.csv'):
    df_mc = pd.read_csv(cpath('appliance_data', mcfile), index_col=0).clean_names()

    cols = {
        'refrigerator': 'rf',
        'freezer': 'fz',
        'central_heater': 'ch',
        'central_cooling': 'cc',
        'wall_cooling': 'ww',
        'evaporative_cooler': 'evcl',
        'water_heater': 'wh',
        'stove': 'st',
        'oven': 'ov',
        'microwave': 'mw',
        'dishwasher': 'dw',
        'washer': 'cw',
        'dryer': 'cd',
        'white_goods': 'wg'
    }
    # flip the composition dataframe so we can join it. The material names are dirty so we strip them and clean them up
    df_mc2 = df_mc.rename(columns=cols).T.rename(columns=lambda x: x.strip()).clean_names()
    return df_mc2

# for appliances that have a discard column...
def expand_by_discrd(df, at):
    discrd_key = at + 'discrd'
    age_key = 'd' + at + 'age'
    ret = (
        # filter for items that have nonzero discards
        df.filt(lambda x: x[discrd_key] > 0)
        .assign(
            appltype=at,                          # set the appliance type
            disp=lambda x: x[discrd_key] * x.wt,  # scale by pop weight
            # create the age values: 3 values = {0,1} if it belongs to age[123] or not
            ages=lambda x: x.apply(lambda row: [row.wt * (row[age_key] == (i + 1)) for i in range(3)], axis=1) 
            )
        .rename(columns={discrd_key: 'discrd', age_key: 'age'})
        [['servzip', 'appltype', 'discrd', 'disp', 'age', 'ages']]
    )
    
    # Expand the 'ages' column into separate columns
    ret = (
        ret.join(pd.DataFrame(ret['ages'].tolist(), 
                              index=ret.index, 
                              columns=[f'age{(i+1)}' for i in range(3)]))
        .drop(columns='ages')
    )
    return ret


# for appliance that only have an age column
def expand_by_age(df,at):
    # df.loc[df["dccage"] <= 3, "dccage"] = 1
    # df.loc[df["dccage"] > 3, "dccage"] = 0
    # df['ccdisp'] = df["dccage"]*df["wt"]
    age_key='d'+at+'age'
    ret= (
        # filter for items that have nonzero discards
        df.filt(lambda x: x[age_key]<4)
        .assign(
            appltype=at,                     # set the appliance type
            disp=lambda x: x.wt,             # scale by pop weight
            # create the age values: 3 values = {0,1} if it belongs to age[123] or not
            ages=lambda x: x.apply(lambda row: [row.wt*(row[age_key]==(i+1)) for i in range(3)],axis=1) 
            )
        .rename(columns={age_key:'age'})
        [['servzip','wt','appltype','disp','age','ages']]
    )
    
    # Expand the 'ages' column into separate columns
    ret = (
        ret.join(pd.DataFrame(ret['ages'].tolist(), 
                              index=ret.index, 
                              columns=[f'age{(i+1)}' for i in range(3)]))
        .drop(columns='ages')
    )
    return ret


# Raw survey data 
def read_survey_data(df_mc2):
    dfr = pd.read_csv(cpath('appliance_data','Final19_SW_CleanedSurvey.csv'), low_memory=False).clean_names()
    dfr = dfr[['servzip','rfdiscrd','drfage','fzdiscrd','dfzage','dchage','dccage','dwwage','devclage',
            'dwhage','dstage','dovage','dmwage','ddwage','dcwage','dcdage','wt']]
    dfr['wt'] = dfr['wt'].replace(',', '')
    dfr['wt'] = pd.to_numeric(dfr['wt'])

    import re
    all_types=[re.sub(r'd(.*)age',r'\1',col) for col in dfr.columns if re.match(r'.*age$',col)]


    # FIXME:
    # recycling_rate = 1.  # 100% recycling rate for now, need to get this from JDS work
    recycling_rate = 0.64  # 64% from JDS work, need citation

    df_long=(
        pd.concat(
            # read data for appliances that have discrd column
            [expand_by_discrd(dfr,at) for at in ['rf','fz']]

            # read data for appliances that don't have discrd column
            +[expand_by_age(dfr, at) for at in list(set(all_types)-set(['rf','fz']))]
        )
        # .filt(lambda x: x.age1>0)
        .join(df_mc2,on='appltype')
        .assign(
            recycled   = lambda x: x.disp * recycling_rate,                                 # total recycled
            weight     = lambda x: x.recycled * x.weight * lb_to_tonne * recycling_rate,    # total weight
            ferrous    = lambda x: x.weight * x.ferrous,                                    # material weights...
            nonferrous = lambda x: x.weight * x.nonferrous,
            hazardous  = lambda x: x.weight * x.hazardous,
            other      = lambda x: x.weight * x.other,
            total      = lambda x: x.ferrous + x.nonferrous + x.hazardous + x.other, # for checks

            # assume one trip per appliance; LDT for everything except evcl and mw
            lda_trips  = lambda x: case_when( x.appltype.isin(['evcl','mw']), x.recycled,    # car (LDA) if evcl or mw
                                            True, 0),                                    # light-duty truck otherwise (so no trips)
            lda_weight = lambda x: case_when( x.appltype.isin(['evcl','mw']), x.weight,  # car (LDA) if evcl or mw
                                            True, 0),                                    # light-duty truck otherwise (so no weight)
            ldt_trips  = lambda x: case_when( ~x.appltype.isin(['evcl','mw']), x.recycled,   # light-duty truck if not evcl or mw
                                            True, 0),                                    # car (LDA) otherwise (so not trips)
            ldt_weight = lambda x: case_when( ~x.appltype.isin(['evcl','mw']), x.weight, # car (LDA) if evcl or mw
                                            True, 0),                                    # light-duty truck otherwise (so no weight)

        )
        .rename(columns={'servzip':'zip_code'})
    )
    return df_long


###########################
# Table 1 final: ZIP TO CAR
def compute_zip_to_car(df_long):

    # this will read the most recent CAR data directly from 
    # DTSC's ARCGIS MAP: https://dtsc.maps.arcgis.com/apps/instant/nearby/index.html?appid=57fd38f812d4467e9768ba6974d5db8c
    # (referenced from: https://dtsc.ca.gov/certified-appliance-recycler-car-program/#CARmap)
    # url = "https://services3.arcgis.com/Oy2JTCD10wkoelxS/arcgis/rest/services/CAR_coord_240410Map/FeatureServer/0/query?f=json&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=FID&outFields=*&outSR=102100&spatialRel=esriSpatialRelIntersects"
    # car=read_car_data(url)
    car_mon=read_car_data_raw().rename(columns={'car':'car_id'})
    
    used_zips=get_used_zips(df_long)
    assert(len(used_zips.filt(lambda x: x.geom_zip.isna()))==0)

    # map zips to CAR sites
    zip_car_mapping=compute_nearest_neighbor_mapping('zip_to_car',used_zips,'geom_zip',car_mon,'geometry_car','zip_code',radius=100,max_options=3)

    (fig,ax,p)=plot_spatial_mapping(
        zip_car_mapping.set_geometry('geom_zip').set_crs(crs_ll).set_geometry('geometry_car').set_crs(crs_ll)
        ,'geom_zip','geometry_car','car_id','ZIP to CAR: OSM Mapping')


    zip_car_mapping=zip_car_mapping.rename(
        columns={
            'car_id':'car',
            'business_name':'destination',
            'business_location':'destination_address',
            'geometry_car':'geometry_car'
    })[['zip_code','geom_zip','car','destination','destination_address','geometry_car']]

    table1_zip_to_car=(
        df_long
        .assign(zip_code=lambda x: x.zip_code.astype(str))
        .groupby(['zip_code']).sum().drop(columns=['appltype'])
        .join(zip_car_mapping.set_index('zip_code'))
        .reset_index()
    )

    return table1_zip_to_car

###########################
# DIVERTED APPLIANCES (TO ADD TO CAR INFLOWS)

def compute_landfill_car_flows(diverted_apps):

    (ent,flw)=collect_rdrs_ods(diverted_apps)

    # hacky way to use the geocode_entities to get the lat lon for these RDRS sites
    null_gdf = lambda geomname: gpd.GeoDataFrame({'rdrs_id':[],geomname:[]},geometry=geomname).set_crs(crs_ll)
    landfills=(
        geocode_entities(merge_aggregated_geometries(*aggregate_all_geometries(ent,
                                                    null_gdf('geometry_merged'),
                                                    null_gdf('geometry_merged'),
                                                    null_gdf('geometry_merged')))
        )
        .rename(columns={'geometry_merged':'geometry_landfill'})
        .set_geometry('geometry_landfill').to_crs(crs_ll)
    )

    # GET THE CAR ENTITIES
    car_mon=read_car_data_raw().rename(columns={'car':'car_id'})

    # MAP LANDFILLS TO CARS
    landfill_car_mapping=compute_nearest_neighbor_mapping('landfill_to_car',
                                                            landfills,'geometry_landfill',
                                                            car_mon,'geometry_car',
                                                            'rdrs_id',
                                                            radius=300,max_options=3)

    # PLOT THE MAPPING
    (fig,ax,p)=plot_spatial_mapping(landfill_car_mapping.set_geometry('geometry_landfill').set_crs(crs_ll)
                                            .set_geometry('geometry_car').set_crs(crs_ll)
                        ,'geometry_landfill','geometry_car','car_id','Landfill to CAR: OSM Mapping')

    # Compute landfill to CAR flows from diverted_apps
    lf_car_flows=(
        landfill_car_mapping.join(
            diverted_apps.groupby('o_rdrsid')[['tonssent']].sum()
            ,on='rdrs_id'
        )
        .rename(
            columns={
                'tonssent':'white_goods',
                'car_id':'car'
                })
    )
    return lf_car_flows


###########################
# TABLE 2: CAR to shredder

def compute_car_to_shredder(table1_zip_to_car,lf_car_flows,compare=True):

    # compute mapping from CAR to shredders
    car_mon=read_car_data_raw().rename(columns={'car':'car_id'})


    # GET THE SHREDDER ENTITIES
    # For now we pull the specific shredders from mon's list
    # FIXME: but we really just need the raw dataset
    car_shred_mapping_mon=read_car_shred_mapping_mon() 

    shred_ent = (
        car_shred_mapping_mon.groupby(['shredder']).first()
        .drop(columns=['car','business_n','origin_address','origin_x','origin_y','near_fid'
                    ,'near_dist','geometry_car']) # get rid of CAR column
        .set_geometry('geometry_shred').set_crs(crs_ll)
    )


    car_shred_mapping=compute_nearest_neighbor_mapping(
        'car_to_shred'
        ,car_mon.rename(columns={'business_location':'origin_address'}),'geometry_car',
        shred_ent.set_geometry('geometry_shred').set_crs(crs_ll),'geometry_shred',
        'car_id',radius=500,max_options=3)

    (fig,ax,p)=plot_spatial_mapping(
        car_shred_mapping
        .set_geometry('geometry_car').set_crs(crs_ll).set_geometry('geometry_shred').set_crs(crs_ll)
        ,'geometry_car','geometry_shred','rdrs','CAR to SHRED(RDRS): OSM Mapping')

    # add on white_goods from landfill1 to CAR
    assert(len(table1_zip_to_car.filt(lambda x: x.car.isna()))==0)
    all_car_inflows=(
        # take total CAR inflows from survey
        table1_zip_to_car
        .assign(car=lambda x: x.car.astype(int))

        # total by car
        .groupby(['car']).sum(numeric_only=True).reset_index()

        # join on white goods from landfill1
        .join(lf_car_flows.groupby('car')[['white_goods']].sum(),on='car',how='outer').fillna(0)
    )

    table2_car_to_shred=(
        all_car_inflows
        
        # join the car_to_shredder_mapping
        .join(car_shred_mapping.set_index('car_id'),on='car',rsuffix='_csm')

        # compute the weights
        .assign(
            origin_x=lambda x: x.geometry_car.apply(lambda point: round(point.x,8)),
            origin_y=lambda x: x.geometry_car.apply(lambda point: round(point.y,8)),
            destination_x=lambda x: x.geometry_shred.apply(lambda point: round(point.x,8)),
            destination_y=lambda x: x.geometry_shred.apply(lambda point: round(point.y,8)),
            weight_total=lambda x:(x.weight+x.white_goods.fillna(0))*0.994,
            trips=lambda x: x.weight_total/20
        )
    )
    return table2_car_to_shred, all_car_inflows

###########################
### TABLE 2b - CAR to class 1 landfills
def compute_car_to_landfill1(all_car_inflows,df_mc2):

    car_mon=read_car_data_raw().rename(columns={'car':'car_id'})
    cl1lf=read_class1_landfill_endpoints()

    car_landfill1_mapping=compute_nearest_neighbor_mapping(
        'car_to_landfill1'
        ,car_mon.to_crs(crs_ll),'geometry_car',
        cl1lf.set_geometry('geometry_landfill1').set_crs(crs_ll),'geometry_landfill1',
        'car_id',radius=500,max_options=3)

    (fig,ax,p)=plot_spatial_mapping(
        car_landfill1_mapping
        .set_geometry('geometry_car').set_crs(crs_ll).set_geometry('geometry_landfill1').set_crs(crs_ll)
        ,'geometry_car','geometry_landfill1','car_id','CAR to LANDFILL1(RDRS): OSM Mapping')

    cl1_mon=read_car_landfill1_mapping_mon()

    (fig,ax,p)=plot_spatial_mapping(
        cl1_mon
        .set_geometry('geometry_car').set_crs(crs_ll).set_geometry('geometry_landfill1').set_crs(crs_ll)
        ,'geometry_car','geometry_landfill1','car','CAR to LANDFILL1: ArcGIS Pro Mapping (Monica)')

    table2b_car_to_landfill1=(
        all_car_inflows
        .join(car_landfill1_mapping.set_index('car_id'),on='car',rsuffix='_mapping')
        .assign(
            weight_haz=lambda x: (x.hazardous + x.white_goods*df_mc2.loc['wg','hazardous']),
            trips_haz=lambda x: x.weight_haz/20
        )
        .rename(columns={'business_location':'origin_address'})
        [['car','origin_address','weight_haz','trips_haz','landfill','destination_address']]
    )
    return table2b_car_to_landfill1



########################
#### TABLE 3: SHREDDER TO PORTS
def compute_shredder_to_port(table2_car_to_shred,df_mc2):

    all_shred_inflows=(
        gpd.GeoDataFrame(
            table2_car_to_shred
            .groupby(['rdrs','shredder','geometry_shred']).sum(numeric_only=True).reset_index()
            .rename(columns={'rdrs':'rdrsid'})
            # [['rdrsid','shredder','ferrous','nonferrous','trips','geometry_shred']]
            ,geometry='geometry_shred'
            ,crs=crs_ll)
    )

    caports_ent = get_ca_port_endpoints()
    caports_use=(
        caports_ent.filt(lambda x: ~(x.n1=='LOS ANGELES'))   # since POLA and POLB are basically co-located, we'll just use POLB, so we drop LA here
        .rename(columns={'geometry_merged':'geometry_port'})
        .set_geometry('geometry_port').set_crs(crs_ca).to_crs(crs_ll)
    )

    shred_port_mapping=compute_nearest_neighbor_mapping(
        'shred_to_port'
        ,all_shred_inflows
        .rename(columns={'index_right':'index_right_old'})    # can't have index_right here because we do a spatial join later
        .to_crs(crs_ll),'geometry_shred',
        caports_use.set_geometry('geometry_port').set_crs(crs_ll),'geometry_port',
        'rdrsid',radius=500,max_options=3)

    (fig,ax,p)=plot_spatial_mapping(
        shred_port_mapping
        .set_geometry('geometry_shred').set_crs(crs_ll).set_geometry('geometry_port').set_crs(crs_ll)
        ,'geometry_shred','geometry_port','rdrs_id','SHREAD to CAPORT: OSM Nearest Neighbor Mapping')

    table3_shred_port =(
        all_shred_inflows
        .join(shred_port_mapping.set_index('rdrsid').add_col_suffix('_port'),on='rdrsid')
        .assign(
            ferrous_weight=lambda x: x.ferrous + x.white_goods*df_mc2.loc['wg','ferrous'],
            ferrous_trips =lambda x: x.ferrous_weight/20,
            nonferrous_weight=lambda x: x.nonferrous + x.white_goods*df_mc2.loc['wg','nonferrous'],
            nonferrous_trips =lambda x: x.nonferrous_weight/20
        )
        
    )
    return table3_shred_port, all_shred_inflows

###############################
## Table 3b - Shredder to landfills
def compute_shred_to_landfillctmsr(all_shred_inflows,df_mc2):

    landfills_ctmsr=read_ctmsr_landfill_endpoints().rename(columns={'rdrsid':'d_rdrsid'})

    shred_lfc_mapping=compute_nearest_neighbor_mapping(
        'shred_to_landfill_ctmsr'
        ,all_shred_inflows.drop(columns=['index_right'],errors='ignore').to_crs(crs_ll),'geometry_shred',
       landfills_ctmsr.set_geometry('geometry_landfillctmsr').set_crs(crs_ll),'geometry_landfillctmsr',
        'rdrsid',radius=500,max_options=3)


    (fig,ax,p)=plot_spatial_mapping(
        shred_lfc_mapping
        .set_geometry('geometry_shred').set_crs(crs_ll).set_geometry('geometry_landfillctmsr').set_crs(crs_ll)
        ,'geometry_shred','geometry_landfillctmsr','rdrs_id','SHRED to LANDFILLCTMSR: OSM Nearest Neighbor Mapping')

    table3b_shred_landfillctmsr =(
        all_shred_inflows
        .join(shred_lfc_mapping.set_index('rdrsid').add_col_suffix('_lfc'),on='rdrsid')
        .assign(
            other_weight=lambda x: x.other + x.white_goods * df_mc2.loc['wg','other'],
            other_trips =lambda x: x.other_weight/20,
        )
    )
    return table3b_shred_landfillctmsr

import os

@app.command()
def list_cache():
    cache_dir = cpath('model_cache_dir')
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('mapping.pickle')]
    for cache_file in cache_file:
        print(cache_file)

@app.command()
def run(mcfile: str = 'material_composition.csv'):

    # Part A: processes RDRS landfill outflows as diverted appliances
    df_mc2 = read_material_composition(mcfile)
    df_long = read_survey_data(df_mc2)
    assert(np.abs(df_long.weight.sum() - df_long.total.sum()) < 1e-8) # ensure our transformations are correct

    # save age data of appliances with refrigerants:
    age_expansion = (
        [(df_long.filt(lambda x: x.appltype==at)
          [['zip_code','age1','age2','age3']]
          .rename(columns={'age1':f'd{at}age1','age2':f'd{at}age2','age3':f'd{at}age3'})
          .groupby('zip_code').sum()) 
          for at in ['rf','fz','cc','ww']]
    )
    age_data = reduce(lambda left, right: pd.merge(left, right, on='zip_code', how='left'), age_expansion).fillna(0)
    age_data.to_csv(cpath('output_appliance','OD_Appliances_Metals_age.csv'))

    # Collection Activity to appliance recyclers (CAR)
    table1_zip_to_car = compute_zip_to_car(df_long)
    (
        table1_zip_to_car[['zip_code', 'car', 'destination', 'destination_address', 
                            'lda_trips', 'ldt_trips', 'lda_weight', 'ldt_weight']]
                            .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table1.csv'))
    )

    # COMPUTE DIVERTED APPLIANCES FROM RDRS DATA
    diverted_apps=compute_diverted_appliances()
    lf_car_flows = compute_landfill_car_flows(diverted_apps)   # FIXME: create option to remove this flow

    # CAR outputs to shredders
    table2_car_to_shred, all_car_inflows = compute_car_to_shredder(table1_zip_to_car, lf_car_flows)
    (
        table2_car_to_shred[['car', 'origin_address', 'origin_x', 'origin_y', 'trips', 'shredder', 
                             'destination_address', 'destination_x', 'destination_y']]
        .assign(car=lambda x: x.car.astype(int))
        .reset_index(drop=True)
        .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table2.csv'))
    )

    # CAR outputs to class 1 landfills
    table2b_car_to_landfill1 = compute_car_to_landfill1(all_car_inflows,df_mc2)
    (
        table2b_car_to_landfill1.assign(car=lambda x: x.car.astype(int)).reset_index(drop=True)
        .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table2b.csv'))
    )

    # Shredder outputs to ports
    table3_shred_port, all_shred_inflows = compute_shredder_to_port(table2_car_to_shred,df_mc2)
    (
        table3_shred_port.rename(columns={'n1_port': 'port'})
        [['shredder', 'ferrous_trips', 'nonferrous_trips', 'port']]
        .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table3.csv'))
    )

    # Shedder outputs to landfills handling CTMSR
    table3b_shred_landfillctmsr = compute_shred_to_landfillctmsr(all_shred_inflows,df_mc2)
    (
        table3b_shred_landfillctmsr.rename(columns={'landfill_lfc': 'landfill_ctmsr'})
        [['shredder', 'other_weight', 'other_trips', 'landfill_ctmsr']]
        .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table3b.csv'))
    )

    # FIXME: do a mass balance check.

if __name__ == "__main__":
    app()

