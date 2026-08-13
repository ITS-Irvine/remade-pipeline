from dataclasses import dataclass, field

from services.routing import compute_nearest_neighbor_mapping, has_maritime_leg, has_maritime_leg_v2
from config import cli_field
from config.base_config import BaseModelConfig
from config.path_template import PathTemplate
from core.units import ureg, unify_pint_units
from functools import reduce
from pathlib import Path
import pickle
import re
from typing import Tuple, Union
from typing_extensions import OrderedDict
import pandas as pd
import geopandas as gpd
import numpy as np
import pandera as pa
import shapely.geometry
import typer
from services.geocode import GeocodeById, Geocoder, MergeGeocoder, NullGeocoder, crs_ca, crs_ll, get_dismantler_endpoints, read_rdrs_manual_geocodes, read_unify_geo, zip_to_id_mapping
from layers.base import ModelLayer
from layers.rdrs import collect_rdrs_ods, load_rdrs_flow
import core.common # also loads units and common unit registry
from core.common import cpath, cachier, disable_index_checks
import logging
from IPython.display import display
from plotnine import ggplot, aes, geom_map, labs
import contextily as ctx
import pandera.pandas as pa
import pygris as pyg
from pandera.typing.pandas import DataFrame

from core.model_types import (
    _AggregatedSchemaWithNulls, AnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows, AnnualizedLayeredGroupedFlowIdSchemaWithFlows, ApplFlowSchema, ApplianceCase, ApplianceMaterialCompositionSchema, ApplianceZipGenerationSchema, ApplianceZipToCarByApplianceAndAgeSchema, _CountySchema, _CountySchemaWithGeocodes, ElvFlowSchema, 
    EndpointAggregation, ExportDestinationProportions, LayerFlowSchema,
    ODPairSchema, EndpointSchema, AnnualizedLayeredGroupedFlowIdSchema, GroupedFlowIdSchemaWithODNulls, GeocodedEndpointSchema, 
    GeocodedEndpointSchemaWithNulls, LayerEndpointSchema, RDRSMaterialSchema, RawRDRSData, RefrCase, pd
)
from core.model_types import unique_flow_agg,unique_flow_cols

from pandera.typing import DataFrame
import pandera as pa
import typer

from core.model_types import AnnualizedLayeredGroupedFlowIdSchemaWithFlows, schema_to_cols
from utils.logging_config import redirect_stdout_to_logging, set_log_level

app = typer.Typer()

# load settings, the second and third lines here support reloading
from config.settings import settings
# import config
# reload(config)

import core.common
from core.common import case_when, cpath, material_mapping_to_dict, read_material_mapping, safe_int_convert, wdisplay

import pandas as pd
import geopandas as gpd
import os # for OS independent paths
import osrm # note: https://github.com/ustroetz/python-osrm/issues/59
from osrm import Point, simple_route
import janitor
import matplotlib as mpl
import polyline
import shapely
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt

import re
import warnings
import pickle
from IPython.display import display
from services.geocode import (backfill_regions, crs_ca, crs_ll, get_zcta_endpoints, get_ca_port_endpoints)

from layers.appliance import read_car_endpoints, read_class1_landfill_endpoints, read_ctmsr_landfill_endpoints, read_shredder_endpoints



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def load_rdrs_appliance_flows(
#     rdrs_flow
# ):

#     rdrs_flow.groupby(['landfill'])[['white_goods']].sum()
#     df['white_goods'] = df['white_goods']
#     df.to_csv(cpath('output_appliance','LandfillOutflows_grouped.csv'))
#     df1 = pd.read_csv(cpath('appliance_data','landfills_collectors.csv'), encoding='unicode_escape').clean_names()
#     df1 = df1.rename(columns={'i»¿landfill': 'landfill'})
#     merged_df = df.merge(df1, on='landfill').fillna(0)
#     df['white_goods'].sum()

kg_to_tonnes = ureg('kg').to('tonnes').magnitude

trip_type_mat_class_map={
        'VEHICLE':'White Goods:Mixed White Goods:Mixed White Goods',
        'hazardous':'HAZ:HAZ:HAZ',
        'ferrous_metals':'Metal:Scrap Metal:Ferrous Scrap',
        'non_ferrous_metals':'Metal:Scrap Metal:Nonferrous Scrap - Aluminum',
        'Ferrous':'Metal:Scrap Metal:Ferrous Scrap',
        'NonFerrous':'Metal:Scrap Metal:Nonferrous Scrap - Aluminum',
        'other':'Mixed:Mixed Residuals:Processing Residuals',
        'ctmsr':'Mixed:Mixed Residuals:Processing Residuals'
        # 'white_goods_weight':'White Goods:Mixed White Goods:Mixed White Goods',
}


def read_dmv_data(
    dmv_data_file:     Path,
    cars_dataset_file: Path,
    cars_makes_file:   Path,
    cache:             bool = True
    ):
    """
    Reads and processes DMV data. Caches the processed data to avoid reprocessing.
    
    Parameters:
    - cache: If True, use cached data if available. If False, reprocess the data.
    """
    cache_file = cpath('model_cache_dir','dmv2021_final.pkl')

    if cache and os.path.exists(cache_file):
        logger.info("Loading cached DMV data")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logger.info("Junked Vehicles: Reading DMV data")

    # the dmv-augmented dataset is the DMV data junked vehicle dataset extended with data obtained from the
    # NHTSA's Vehicle Product Information Catalog (vPIC) API. Not all DMV VINs have augmented vPIC data.
    # fn=cpath('dmv_data','dmv-augmented.csv-2024-02-14.zip')
    dmv=pd.read_csv(dmv_data_file,low_memory=False).clean_names().strip_col_names()

    # we'll focus on 2021

    logger.info("Junked Vehicles: Junked Vehicles for 2021")
    dmv2021=dmv.filt(lambda x: x.source.str.match(r'.*Y2021'))

    # estimate the GVWR for each scrapped vehicle
    # we use unladen weight if it's provided
    # otherwise, we use the GVWR from the note field
    # if that's not available, we read it from the gross_vehicle_weight_rating_from field, which specifies it as a vehicle class
    dmv2021_gvwr=(dmv2021
        .assign(unladen_weight=lambda x: pd.to_numeric(x.unladen_weight.replace(r'^\s*$', np.nan, regex=True), errors='coerce'))
        .assign(gvwr_calc=lambda x: x.note.str.replace(r'^.*GVWR\s*[=:]\s*([\d,;]+).*$',r'\1',regex=True).str.replace('[,;]','',regex=True).apply(safe_int_convert))
        .assign(
                gvwr_calc2_low=lambda x: np.where(x.gross_vehicle_weight_rating_from.isna(),
                                                x.gross_vehicle_weight_rating_from,
                                                x.gross_vehicle_weight_rating_from.str.replace(r'^.*Class.*?:\s*([\d;]+)\s*(-\s*([\d;]+)|lb or less).*$',r'\1',regex=True)
                                                    .str.replace('[,;]','',regex=True).apply(safe_int_convert)
                                                ),
                gvwr_calc2_hi =lambda x: np.where(x.gross_vehicle_weight_rating_from.isna(),
                                                x.gross_vehicle_weight_rating_from,
                                                x.gross_vehicle_weight_rating_from.str.replace(r'^.*Class.*?:\s*([\d;]+)\s*(-\s*([\d;]+)|lb or less).*$',r'\3',regex=True)
                                                    .str.replace('[,;]','',regex=True).apply(safe_int_convert)
                                                )
                )
        .assign(
                # if we only got a low value, use that for hi too
                gvwr_calc2_hi =lambda x: np.where(~x.gvwr_calc2_low.isna() & x.gvwr_calc2_hi.isna(),
                                                x.gvwr_calc2_low,
                                                x.gvwr_calc2_hi),
                gvwr_calc2 = lambda x: x.gvwr_calc2_low + (x.gvwr_calc2_hi-x.gvwr_calc2_low)/2,
                gvwr_merge_good = lambda x: x.unladen_weight.combine_first(x.gvwr_calc).combine_first(x.gvwr_calc2)
                )
        )
    tmp=dmv2021_gvwr.filt(lambda x: ~x.gvwr_merge_good.isna()) # records we successfully inferred a weight from
    tmpx=dmv2021_gvwr.filt(lambda x: x.gvwr_merge_good.isna()) # 
    tmpx_bc=tmpx.filt(lambda x: ~x.body_class.isna())
    tmpx_nbc=tmpx.filt(lambda x: x.body_class.isna())

    # at this point, it seems we've only inferred a weight for about 1/3 of the records

    # augment weight estimate using the cars dataset, which was found on kaggle: https://www.kaggle.com/datasets/usefashrfi/car-specification-dataset
    # the dmv data has two make fields: the complete make_a field and the incomplete make field pulled from vPIC
    # here we a mapping between the dmv make_a field, which is make abbrevations that are quite messy
    # and a consistnt of makes in the cars dataset that also align with the incomplete make field in the dmv data
    # FIXME: consider this EPA sources: https://www.epa.gov/automotive-trends/explore-automotive-trends-data#DetailedData
    logger.info("Junked Vehicles: Merging augmented vehicle data with the cars dataset")
    all_makes=pd.read_csv(cars_makes_file) # do some validation on this

    # now we subset to cars only.
    dmv2021_cars=(
        dmv2021_gvwr.filt(lambda x: ~x.body_class.fillna('').astype(str).str.match('Trailer|Off-road|Motorcycle|Incomplete|Truck-Tractor|Motorhome|NEV|Bus|Truck'))
        .join(all_makes.set_index('make_a')[['make_merge']],on='make_a',how='left')

        # we're creating a set of merged columns for indexing
        .assign(
            makex=lambda x: x.make.fillna(x.make_merge),
            modelx=lambda x: x.model.fillna(''),
            body_classx=lambda x: x.body_class.fillna(''),
            seriesx=lambda x: x.series.fillna(''),
            fuel_type_primaryx=lambda x: x.fuel_type_primary.fillna(''),
            year_modelx=lambda x: x.model_year.fillna(x.year_model)
        )
    )

    # heaviest cars (for filtering largest unladen weight)
    # https://www.klipnik.com/used-car-buyers-guides/heaviest-cars-suvs-trucks/

    # use 10000 for gasoline vehicles

    # truck tractor
    # https://www.prettymotors.com/how-much-does-a-semi-truck-tractor-weigh/

    # use 30000 lbs as the cutoff for a truck tractor



    # now grab a set of unique make/model/body_class/series/fuel_type/year_model combinations
    logger.info("Junked Vehicles: Using averages for missing curb weight")
    vkey=['makex','modelx','body_classx','seriesx','fuel_type_primaryx','year_modelx']
    dmv2021_ucars=(
        dmv2021_cars.copy()    # copy to reduce fragmentation
        .groupby(vkey).count()
        .assign(mmcnt=lambda x: x.groupby(['makex', 'modelx'])['vin'].transform('sum'))
        .sort_values(['mmcnt','vin'],ascending=False)
        [['mmcnt','vin']]
    )

    # dump these unique combinations to a csv file; this is used to do a supervised mapping
    # between the DMV data and the cars dataset done in a spreadsheet
    (
        dmv2021_ucars
        .to_csv('make_model_class_series_fuel_year2.csv')
        )

    ## << LOTS OF MANUAL WORK DONE HERE TO PRODUCE MAPPING FILE >>

    # read in the mapping file
    mm=(
        pd.read_csv(cars_dataset_file, low_memory=False)
        .clean_names()
        .drop_columns(regex='unnamed_')
    )

    # now we merge the mapping file back into the DMV data
    dmv2021_cars_aug=(
        dmv2021_cars.copy()  # reduce fragmentation
        .join(mm.drop_columns(regex='vin')
            .assign(
                makex=lambda x: x.makex.fillna(''),
                modelx=lambda x: x.modelx.fillna(''),
                seriesx=lambda x: x.seriesx.fillna(''),
                fuel_type_primaryx=lambda x: x.fuel_type_primaryx.fillna(''),
                body_classx=lambda x: x.body_classx.fillna(''),
                year_modelx=lambda x: x.year_modelx.fillna(''),
                curb_weight_txt=lambda x: x.curb_weight,
                curb_weight=lambda x: x.curb_weight_txt.str.replace(r'([\d\.]+)[\+-]*\s+.*', r'\1', regex=True).astype(float),
                curb_weight_u=lambda x: x.curb_weight.astype('pint[lbs]'),
            ).set_index(vkey)
            ,on=vkey
            ,how='left')
        .assign(
            # this is the vehicle descriptor VIN without the check digit or specific vehicle id
            # we'll use this to compute means
            unladen_weight_u=lambda x: x.unladen_weight.astype('pint[lbs]'),
            strip_vin=lambda x: x.vin.str[:8]
        )
    )

    # At this point, roughly 17% of records still don't have a curb weight
    # for this, we'll use the average curb weight for the make/model/body_class/series/fuel_type/year_model
    avgs_strip_vin=(
        dmv2021_cars_aug.groupby('strip_vin').agg({'curb_weight':['count','mean','min','max']})
        .flatten_columns()
        .add_col_suffix('_strip_vin')
    )

    # then fall back to averages by DMV make and year
    avgs_make_year=(
        dmv2021_cars_aug.groupby(['make_a','year_model'])
        .agg({'curb_weight_u':['count','mean','min','max']})
        .flatten_columns()
        .add_col_suffix('_make_year')
    )

    # then fall back to averages by year
    avgs_year=(
        dmv2021_cars_aug.groupby(['year_model'])
        .agg({'curb_weight_u':['count','mean','min','max']})
        .flatten_columns()
        .add_col_suffix('_year')
    )

    # then averages by make
    avgs_make=(
        dmv2021_cars_aug.groupby(['make_a'])
        .agg({'curb_weight_u':['count','mean','min','max']})
        .flatten_columns()
        .add_col_suffix('_make')
    )

    # then overall averages (we won't have body class or GVWR for these items)
    avgs=(
        dmv2021_cars_aug
        .agg({'curb_weight_u':['count','mean','min','max']})
        .flatten_columns()
        .add_col_suffix('_all')
    )



    dmv2021_cars_aug2=(
        dmv2021_cars_aug.copy() # reduce fragmentation
        .join(avgs_strip_vin,on='strip_vin')
        .join(avgs_make_year,on=['make_a','year_model'])
        .join(avgs_year,on=['year_model'])
        .join(avgs_make,on=['make_a'])
        .assign(
            curb_weight_u=lambda x: (
                x.curb_weight_u.pint.to('lb')
                .combine_first(x.unladen_weight_u.pint.to('lb'))
                .combine_first(x.curb_weight_u_mean_make_year.pint.to('lb'))
                .combine_first(x.curb_weight_u_mean_year.pint.to('lb'))
                .combine_first(x.curb_weight_u_mean_make.pint.to('lb'))
                .fillna(avgs.loc['mean','curb_weight_u_all'].to('lb'))
            )#.astype('pint[lbs]') # UNIT
        )

        # tweak zips
        .rename(columns={'ro_zip':'zip_code'})
        .assign(zip_code=lambda x: x.zip_code.astype(str).str.zfill(5))
    )

    dmv2021_final=dmv2021_cars_aug2.copy() # copy defragments

    assert dmv2021_final.curb_weight_u.notna().all(), "NA values still exist for vehicle curb weights"

    if cache:
        logger.info("Caching processed DMV data")
        with open(cache_file, 'wb') as f:
            pickle.dump(dmv2021_final, f)

    return dmv2021_final


def read_material_composition(mcfile:Path):
    vcomp=(
        pd.read_csv(mcfile)
        .clean_names().strip_col_names()
        .drop_columns(regex='to_prior|unnamed|.*x$')
        .pivot_longer(index='year',names_to=['material'],values_to='composition')
    )
    return vcomp

#%%
def read_recovery_rates(mrrfile:Path):
    mrcov=(
        pd.read_csv(mrrfile).clean_names().strip_col_names()
        .drop_columns(regex='to_prior|unnamed|.*x$')
        .pivot_longer(index='year',names_to=['material'],values_to='recovery')
    )
    return mrcov

@pa.check_types()
def compute_dismantling(
    dism_in, 
    full_recov_rates
    ) -> DataFrame[AnnualizedLayeredGroupedFlowIdSchemaWithFlows]:
    # map dismantlers to shredders
    # dmv=dmv2021_final
    # used_zips=get_dmv_zips(dmv).rename(columns={'geometry_merged':'geometry_zip'})
    # map zips to CAR sites
    assert('d_geometry' in dism_in.columns)
    assert('year_modelx' in dism_in.columns)
    dism_proc=(
        dism_in.assign(
            year_model_lu=lambda x: (
                x.year_modelx
                .mask(x.year_modelx<full_recov_rates.year.min(),full_recov_rates.year.min())
                .mask(x.year_modelx>full_recov_rates.year.max(),full_recov_rates.year.max())
                )
        )
    )
    
    mat_recovered = (
        dism_proc.merge(full_recov_rates,left_on='year_model_lu',right_on='year')
        .assign(
            wt_sent=lambda x: (x.curb_weight_u*x.recyc_rate).pint.to('short_ton')
        )
        .groupby(['d_id','d_geometry','material'])
        [['wt_sent']].sum()
        .reset_index()
        .rename(columns={'d_id':'o_id','d_geometry':'o_geometry'})
    )

    # grab all endpoint data
    all_endpoints=[]
    def capture_endpoints(ep):
        all_endpoints.append(ep)
        return ep

    used_dism=gpd.GeoDataFrame(
        mat_recovered[['o_id','o_geometry']].drop_duplicates()
        ,geometry='o_geometry'
        ,crs=crs_ll
    )

    ### SHREDDERS
    # read Monica's shredder entities
    # shr=pd.read_csv(cpath('processed_data_appliance','Supporting Files/Recyclers_Shredder_Near_w_rdrs.csv'), encoding='unicode_escape').clean_names()

    # # merge them
    # shrent=(
    #     shr.groupby('rdrs').first()[['shredder','destination_address','destination_x','destination_y']]
    #     .join(rdrs_ent_allxxx.set_index('rdrs_id'),how='left')
    #     # there are some locations without RDRS
    #     .assign(geometry_merged=lambda x: x.geometry_merged.where(~x.geometry_merged.isna(),
    #                                                 gpd.points_from_xy(x.destination_x,x.destination_y)))
    # )
    shrent=(
        capture_endpoints(read_shredder_endpoints()).reset_index()
        .rename(columns={'id':'d_id'})
        .rename_geometry('d_geometry').set_crs(crs_ll)
    )

    dism_shred_mapping=compute_nearest_neighbor_mapping(
        'dism_to_shred',
        used_dism,'o_geometry',
        shrent,'d_geometry',
        'o_id',radius=400,max_options=3)
    
    mm=material_mapping_to_dict(read_material_mapping())
    
    pf = 1/(20 * ureg('ton')) # FIXME:UNIT: make this unit aware

    # join results
    dism_to_shred=(
        mat_recovered
        .filt(lambda x: x.material.isin(['ferrous_metals','non_ferrous_metals']))
        .merge(dism_shred_mapping,on='o_id',suffixes=('_dism',''))
        .groupby(['o_id','d_id','o_geometry','d_geometry','material'])
        .agg({'wt_sent':'sum'})
        .reset_index()
        [['o_id','d_id','o_geometry','d_geometry','material','wt_sent']] # geom needed downstream
        
        
        .assign(
            layer='EOL Vehicles', # FIXME: from class?
            material_stream=lambda x: x.material.map(trip_type_mat_class_map),
            material_grouping=lambda x: x.material_stream.map(mm),
            material_type='FIXME: DISM TO SHRED',
            material_category='FIXME: DISM TO SHRED',
            material_subcategory='FIXME: DISM TO SHRED',
            ttype='veh_dism_to_shredders',
            year=2021,
            EMFAC_class='T7 Tractor Class 8',
            trips=lambda x: (x.wt_sent * pf).pint.magnitude
        )
    )


    # deal with hazardous
    cl1lf=(
        capture_endpoints(read_class1_landfill_endpoints())
        .reset_index()
        .rename(columns={'id':'d_id'})
        .rename_geometry('d_geometry')
        .to_crs(crs_ll))
    
    dism_cl1lf_mapping=compute_nearest_neighbor_mapping(
        'dism_to_cl1lf_elv',
        used_dism,'o_geometry',
        cl1lf,
        'd_geometry','d_id',radius=400,max_options=3)  

    pf = 1/(20*ureg('ton'))
    
    # FIXME: these seem to be small numbers of trips, so maybe a different
    # truck type is needed
    dism_to_cl1lf=(
        mat_recovered
        .filt(lambda x: x.material.isin(['hazardous']))
        .merge(dism_cl1lf_mapping,on='o_id',suffixes=('_dism',''))
        .groupby(['o_id','d_id','o_geometry','d_geometry','material'])
        .agg({'wt_sent':'sum'})
        .reset_index()
        [['o_id','d_id','o_geometry','d_geometry','material','wt_sent']] # geom needed downstream
        
        .assign(
            layer='EOL Vehicles', # FIXME: from class?
            material_stream=lambda x: x.material.map(trip_type_mat_class_map),
            material_grouping=lambda x: x.material_stream.map(mm),
            material_type='FIXME: DISM TO CLASS 1 LANDFILL',
            material_category='FIXME: DISM TO CLASS 1 LANDFILL',
            material_subcategory='FIXME: DISM TO CLASS 1 LANDFILL',
            ttype='veh_dism_to_cl1lf',
            year=2021,
            EMFAC_class='T7 Tractor Class 8',
            trips=lambda x: (x.wt_sent * pf).pint.magnitude
        )
    )
    
    ret_ep = pd.concat(all_endpoints)
    assert(ret_ep.index.is_unique)
    
    return (
        ret_ep,
        pd.concat(disable_index_checks(
            [dism_to_shred,dism_to_cl1lf]
        ),ignore_index=True)
        # [schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows) + 
        #  ['o_geometry','d_geometry'] # keep geometries for next phase
        #  ]
    )


# %%
@pa.check_types()
def compute_shredding(
    shred_in,
    asr_frac=0.075,
    export_destinations:DataFrame[ExportDestinationProportions]|None=None
    ) -> DataFrame[AnnualizedLayeredGroupedFlowIdSchemaWithFlows]:


    # grab all endpoint data
    all_endpoints=[]
    def capture_endpoints(ep):
        all_endpoints.append(ep)
        return ep


    all_shred_inflows=(
        gpd.GeoDataFrame(
            shred_in
            .groupby(['d_id','material_grouping','d_geometry']).sum(numeric_only=True).reset_index()
            .rename(columns={'d_id':'o_id', 'd_geometry':'o_geometry'})
            # [['rdrsid','shredder','ferrous','nonferrous','trips','geometry_shred']]
            ,geometry='o_geometry'
            ,crs=crs_ll)
    )

    caports_ent = capture_endpoints(get_ca_port_endpoints())
    caports_use=(
        caports_ent.filt(lambda x: ~(x.n1=='LOS ANGELES'))   # since POLA and POLB are basically co-located, we'll just use POLB, so we drop LA here
        .rename(columns={'geometry':'d_geometry'})
        .set_geometry('d_geometry').to_crs(crs_ll)
    )
    
            
    # FIXME: assume 20 tons per trip, but we can change this later
    pf = 1/(20*ureg('tons')) # FIXME:UNIT: make this unit aware

    generated_shredded_metals=(
        all_shred_inflows
        .assign(
            layer='EOL Vehicles', # FIXME: from class?
            wt_sent=lambda x: x.wt_sent*(1-asr_frac),
            ttype='shred_to_port', # FIXME: should be shred_to_export
            material_stream=lambda x: x.material_grouping.map(trip_type_mat_class_map),
            EMFAC_class='T7 POLA Class 8',
            trips = lambda x: (x.wt_sent * pf).pint.magnitude, 
            year=2021, #FIXME: hardcode
        )
        # [schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows) + 
        #  ['o_geometry','d_geometry'] # keep geometries for next phase
        #  ]
    )
    
    if export_destinations is None:
        logger.info("No export destinations provided for ELV material types; sending exports to nearest port")

        shred_port_mapping=compute_nearest_neighbor_mapping(
            'shred_to_port_elv'
            ,all_shred_inflows[['o_id','o_geometry']],'o_geometry',
            caports_use.set_geometry('d_geometry').set_crs(crs_ll).reset_index()
            .rename(columns={'id':'d_id'}),'d_geometry',
            'o_id',radius=500,max_options=3)
        
        shred_to_export =(
            generated_shredded_metals
            .merge(shred_port_mapping,on='o_id',suffixes=('_shred',''))
            # [schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows) + 
            #  ['o_geometry','d_geometry'] # keep geometries for next phase
            #  ]
        )
        
    else:
        export_materials = set([
            m for m in
            export_destinations.material_grouping.unique()
            if m in generated_shredded_metals.material_grouping.unique()
        ])
        logger.info(f"Mapping exports to destinations using provided proportions for {export_materials}")
        shred_to_export=(
            generated_shredded_metals
            .merge(export_destinations[['material_grouping','d_id','proportion']],
                   left_on='material_grouping',right_on='material_grouping',how='left')
            .assign(wt_sent=lambda x: x.wt_sent * x.proportion)
            [schema_to_cols(LayerFlowSchema)]
        )

    # NOW DO FRACTION TO CTMSR
    landfills_ctmsr=(
        capture_endpoints(read_ctmsr_landfill_endpoints())
        .reset_index().rename(columns={'id':'d_id'})
        .rename_geometry('d_geometry')
    )

    shred_lfc_mapping=compute_nearest_neighbor_mapping(
        'shred_to_landfill_ctmsr_elv'
        ,all_shred_inflows[['o_id','o_geometry']],'o_geometry',
        landfills_ctmsr.set_geometry('d_geometry').set_crs(crs_ll),'d_geometry',
        'o_id',radius=500,max_options=3)

    mm=material_mapping_to_dict(read_material_mapping())
    
    pf = 1/20 # FIXME:UNIT: make this unit aware
    
    table4_shred_landfillctmsr =(
        all_shred_inflows
        .merge(shred_lfc_mapping,on='o_id',suffixes=('_shred',''))
        .groupby(['o_id','d_id','o_geometry','d_geometry']).agg({'wt_sent':'sum'}).reset_index()
        .assign(
            layer='EOL Vehicles', # FIXME: from class?
            wt_sent =lambda x: x.wt_sent*asr_frac,
            trips = lambda x: x.wt_sent.pint.to('tons').pint.magnitude * pf,
            ttype='shred_to_landfillctmsr_elv',
            material='ctmsr',
            material_category='FIXME: shredder residuals',
            material_subcategory='FIXME: shredder residuals',
            material_type='FIXME: shredder residuals',
            material_stream=lambda x: x.material.map(trip_type_mat_class_map),
            material_grouping=lambda x: x.material_stream.map(mm),
            EMFAC_class='T7 Tractor Class 8',
            year=2021, #FIXME: hardcode
        )
        # [schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows) + 
        #  ['o_geometry','d_geometry'] # keep geometries for next phase
        #  ]
    )

    ret_ep = pd.concat(all_endpoints)
    assert(ret_ep.index.is_unique)

    return (
        ret_ep,
        pd.concat([shred_to_export,table4_shred_landfillctmsr],ignore_index=True)
        # [schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows) + 
        #  ['o_geometry','d_geometry'] # keep geometries for next phase
        #  ]
    )

from config.cli_field import cli_field

@dataclass
class ELVModelConfig(BaseModelConfig, prefix="elv"):
    
    elv_data_dir: PathTemplate = field(default='.')
    dmv_data_dir: PathTemplate = field(default='.')
    fleet_db_dir: PathTemplate = field(default='.')

    # ── CLI-exposed fields ────────────────────────────────────────────────────
        
    vehicle_compositions:     PathTemplate = cli_field(
        'ELV material composition - vehicle_composition.csv',
        help    = "CSV file containing vehicle composition data by model year."
    )
    
    material_recovery_rates:  PathTemplate = cli_field(
        'ELV material composition - material_recovery_rates.csv',
        help    = "CSV file containing material recovery rates by material type."
    )
        
    dmv_junked_vehicles_file: PathTemplate = cli_field(
        '.',
        help    = "File to load DMV Junked vehicles from"
    )
        
    cars_dataset_file:       PathTemplate = cli_field(
        '.',
        help    = "File to load augmented vehicle make/model data from"
    )
        
    cars_makes_file:         PathTemplate = cli_field(
        '.',
        help    = "File to map from DMV makes to cars dataset makes"
    )
    
    elv_dismantler_file:     PathTemplate = cli_field(
        '.',
        help    = "File to load vehicle dismantlers from"
    )
    
    vehicles_per_acre: float = cli_field(
        150,
        help    = "Number of vehicles per acre for vehicles dismantler parcels"
    )
    
    illegal_vehicles_fraction: float = field(
        default=0.3,
        #help    = "Fraction of total discarded vehicles assumed to be illegally disposed of"
    )
    
    mexico_export_fraction: float = field(
        default=0.05,
        #help    = "Fraction of total discarded vehicles assumed to be disposed of in Mexico"
    )
    
    model_year: int = cli_field(
        2021,
        help    = "Model year for ELV calculations",
        # metavar = "MODEL_YEAR_ELV"
    )
    run_model: bool = cli_field(
        True,
        help    = "Whether to run the ELV flow model before loading flows",
        # metavar = "RUN_MODEL_ELV"
    )

    # ── Internal fields ───────────────────────────────────────────────────────
    matmap_file:         str          = field(default="")
    name:                str          = field(default="ELV")
    zip_dism_cache_file: PathTemplate = field(default=None)

    # # ── Runtime dependency — set in code, not CLI or TOML ────────────────────
    # geocoder: Geocoder = field(default=None)


class ELVLayer(ModelLayer):
    name: str = "ELV"
    @pa.check_types()
    def __init__(self,
                config: ELVModelConfig,
                geocoder: Geocoder = None,
                export_proportions_layer: ModelLayer = None
                ):
        super().__init__(config.name)
        self.read_material_mappings(config.matmap_file)
        self.model_year                = config.model_year
        self.geocoder                  = geocoder
        self.export_proportions_layer  = export_proportions_layer
        self.elv_data_dir              = config.elv_data_dir
        self.fleet_db_dir              = config.fleet_db_dir
        self.material_recovery_rates   = config.material_recovery_rates
        self.vehicle_compositions      = config.vehicle_compositions
        self.dmv_junked_vehicles_file  = config.dmv_junked_vehicles_file
        self.elv_dismantler_file       = config.elv_dismantler_file
        self.cars_dataset_file         = config.cars_dataset_file
        self.cars_makes_file           = config.cars_makes_file
        self.vehicles_per_acre         = config.vehicles_per_acre
        self.illegal_vehicles_fraction = config.illegal_vehicles_fraction
        self.mexico_export_fraction    = config.mexico_export_fraction
        self.zip_dism_cache_file       = config.zip_dism_cache_file
        self.elvflw                    = None
        self.endpoints                 = None
        
        # # let's fix up any templates
        # for f in ['elv_data_dir','fleet_db_dir',
        #           'material_recovery_rates','vehicle_compositions']:
        #     # updates self.[f] by treating it as a format template string
        #     # and assigning values from self # FIXME: could be an injection threat?
        #     setattr(self,f,getattr(self,f).format(**self.__dict__))
        

        # Load ELV data here as needed
        # forces flow/ent initialization
        if config.run_model:
            self.endpoints, self.elvflw=self.run_elv_flow_model()
        # self.get_flows()
        
    @pa.check_types
    def get_geocoder(self) -> Geocoder:
        if not self.geocoder:
            geocoders = OrderedDict()

            # vetted geocodes            
            geocoders['rdrs_man']  = GeocodeById(read_rdrs_manual_geocodes(),'Manual geocodes')
            geocoders['unify_geo'] = GeocodeById(read_unify_geo(settings.get('pipeline').unify_geo_file),'Unified geocodes')

            # ── Port Geocoder
            # Geocode California ports by ID. Used for maritime export routing.
            geocoders["caports"] = GeocodeById(get_ca_port_endpoints(),'California ports')
            
            # ── ZIP Code Geocoder
            # ZCTA (ZIP Code Tabulation Area) geocoding, needed by both appliance
            # and ELV layers for resolving ZIP-based facility locations.
            geocoders['zips'] = GeocodeById(get_zcta_endpoints(),'ZCTA endpoints')

            # ── ELV Dismantler Geocoder
            # Geocode end-of-life vehicle dismantler locations. Only loaded when
            # the ELV layer is active.
            geocoders['dismantler'] = GeocodeById(
                get_dismantler_endpoints(self.elv_dismantler_file),
                'Dismantler endpoints')

            # ── Null Geocoder (Safety Net)
            # Catch any remaining entities with invalid/empty geometries that
            # no other geocoder resolved. Must be last in the stack.
            geocoders['null_geocodes'] = NullGeocoder(
                select=lambda df: ~df.set_geometry('geometry').geometry.is_valid
            )
            
            self.geocoder = MergeGeocoder(geocoders, require_complete=True)
            
        # FIXME: ELV Layer never uses geocoder?

        return self.geocoder
    
    @pa.check_types()
    def read_material_mappings(self,matmap_file) -> DataFrame[RDRSMaterialSchema]:
        """Read material mappings for Appliance layer."""
        self.matmap = pd.read_excel(matmap_file).clean_names()
        return self.matmap

    @pa.check_types()
    def get_endpoints(self) -> DataFrame[LayerEndpointSchema]:
        """Get the geocoded endpoints of Appliance layer."""
        
        assert(self.elvflw is not None)

        return self.endpoints[schema_to_cols(LayerEndpointSchema)]

    @pa.check_types()
    def get_flows(self) -> DataFrame[LayerFlowSchema]:
        """Get the flows from ELV layer."""
        if self.elvflw is None:
            self.endpoints, self.elvflw=self.run_elv_flow_model()

            assert(len(self.elvflw)>1)
            # FIXME: this assertion shouldn't fail
            assert(len(self.elvflw.filt(lambda x: x.o_id.isna() | x.d_id.isna()))==0)

            # Check for failed material mappings and log warnings if any
            failed_elv_mat_mapping=self.elvflw.filt(lambda x: x.material_grouping.isna())
            if len(failed_elv_mat_mapping) > 0:
                logger.warning(f"ELV material mapping failed on {len(failed_elv_mat_mapping)} rows")
                logger.warning(f"...failed material classes {failed_elv_mat_mapping.MAT_class.unique()}")

            # Clean ELV flows by filling missing groupings and filtering out zero flows
            self.elvflw=(
                self.elvflw.assign(material_grouping=lambda x: x.material_grouping.fillna('MISSING')) # FIXME: do we really want to do this?
                .filt(lambda x: x.wt_sent>0) # remove any flows that are zero
            )

        # Implement loading and returning appliance flows
        return self.elvflw[schema_to_cols(LayerFlowSchema)]

    def get_materials(self) -> DataFrame[RDRSMaterialSchema]:
        """Get the materials handled by ELV layer."""
        # Implement logic to return materials for appliance layer
        return self.matmap
    
    
    


    # %%
    @pa.check_types()
    def run_elv_flow_model(
        self,
        # export_proportions_layer: ModelLayer = None,
        # vcfile: str = 'ELV material composition - vehicle_composition.csv',
        # mrrfile: str = 'ELV material composition - material_recovery_rates.csv'
    ) -> Tuple[DataFrame[LayerEndpointSchema], DataFrame[LayerFlowSchema]]:
        
        export_proportions_layer: ModelLayer=self.export_proportions_layer
        vcfile: str = Path(self.vehicle_compositions)
        mrrfile: str = Path(self.material_recovery_rates)
        
        logger.info("Running ELV flow model")
        
        # grab all endpoint data
        all_endpoints=[]
        def capture_endpoints(ep):
            all_endpoints.append(ep)
            return ep
        
        # get EMFAC data so we know the range of model years we can distribute to
        @cachier(wait_for_calc_timeout=5)
        def read_emfac_veh_zip_pop():
            logger.infox("Running ELV flow model: Reading EMFAC vehicle zip populations (not cached)")
            return(
                
                pd.concat([pd.read_csv(Path(self.fleet_db_dir) / f,header=12) for f in 
                        ['fleetdb-90000-92500-2021.zip', 'fleetdb-92501-95000-2021.zip', 'fleetdb-95001-95500-2021.zip', 
                            'fleetdb-95500-96000-2021.zip', 'fleetdb-96001-96xxx-2021.zip']])
                .clean_names()
                
                # FIXME: confirm how many we're dropping
                .filt(lambda x: ~x.zip_code.str.match('Scrubbed|Unknown'))
                .filt(lambda x: ~x.model_year.str.match('Scrubbed|Unknown'))
                .assign(model_year=lambda x: pd.to_numeric(x.model_year,errors='coerce'))
            )
        with redirect_stdout_to_logging(logging.INFOX):
            logger.info("Running ELV flow model: Reading EMFAC vehicle zip populations")
            emfac_veh_zip_pop = read_emfac_veh_zip_pop(cachier__verbose=True)


        dmv2021_final = read_dmv_data(
            self.dmv_junked_vehicles_file, 
            self.cars_dataset_file, 
            self.cars_makes_file)

        # Standardize the DMV retirement data
        logger.infox("Running ELV flow model: cleaning DMV junked vehicles")
        dmv_gen_load=(
            dmv2021_final
                .assign(
                    wt_sent=lambda x: x.curb_weight_u.pint.to('short_ton'),
                    material='VEHICLE',
                    material_category='FIXME: VEHICLE',
                    material_subcategory='FIXME: VEHICLE',
                    material_type='FIXME: VEHICLE'
                    )
            # .merge(capture_endpoints(get_zcta_endpoints()).reset_index().assign(zip_code=lambda x: x.id.str.replace('^ZIP','',regex=True))
            #     ,left_on='zip_code'
            #     ,right_on='zip_code'
            #     ,how='left'
            #     )
            
            # remove bad records (no year_modelx)
            .filt(lambda x: x.year_modelx.notna())
            
            # remove VIN duplicates!
            .filt(lambda x: ~x.duplicated(subset=['vin'],keep='first'))
        )
        # remove bad zip matches by weight
        # bad = dmv_gen.id.isna()
        # removed = dmv_gen.loc[bad]
        # kept=dmv_gen.loc[~bad]
        # if len(removed)>0:
            
        logger.infox("Running ELV flow model: cleaning DMV junked vehicles (2)")
        z2id=zip_to_id_mapping()
        zz=pd.DataFrame({'zip1':z2id.keys(),'zip2':z2id.values()}).assign(zip=lambda x: x.zip1.str.replace('ZIP','').astype(int))
        dmv_gen_all=(
            dmv_gen_load
            .assign(o_id_orig=lambda x: 'ZIP'+x.zip_code)  # we're going to ignore this
            .assign(year_modelx_orig=lambda x: x.year_modelx,
                    year_modelx=lambda x: x.year_modelx.clip(
                        lower=emfac_veh_zip_pop.model_year.min(),
                        upper=emfac_veh_zip_pop.model_year.max()
                        ),
                    )
        )
        # FIXME: we're ignoring the fact that a (small) chunk of the original zips are out of state zips
        # FIXME: we might instead use these to infer the fraction of legit junked vehicles that are
        # FIXME: moved out of state for dismantling
        dmv_gen=dmv_gen_all
        
        dmv_genx=(
            dmv_gen.groupby(['year_modelx'])
            .agg({'vin':'count','wt_sent':'sum'}).reset_index()
        )
        
        # make sure we're consistent
        assert(len(dmv_gen) == dmv_genx.vin.sum())
        
        # We need to map the vehicles retired by zips to the dismantlers. We will
        # use a gravity model to do this, but first we need to read in the
        # dismantler endpoints and assign them capacities based on their area.
        logger.infox("Running ELV flow model: Loading dismantlers")
        dism=capture_endpoints(get_dismantler_endpoints(self.elv_dismantler_file))
        
        veh_per_acre=self.vehicles_per_acre # FIXME: units

        dism_with_caps = dism.assign(
            parcel_area=lambda x: x.set_geometry('geometry_agg').to_crs(crs_ca).geometry_agg.area*ureg('m^2').to('acre'),
            veh_capacity=lambda x: x.parcel_area*veh_per_acre
        ).set_geometry('geometry').to_crs(crs_ll)
        
        
        # model3 = CapacitatedGravityModel(
        #     alpha=1.0,
        #     beta=3.0,
        #     convergence_threshold=1e-6,
        #     max_iterations=100
        # )
        
        logger.infox("Running ELV flow model: Computing vehicle zip distributions")
        # create the origins dataframe with unique index
        cars_gdf=(
            # gpd.GeoDataFrame(dmv_genx,crs=crs_ll)
            dmv_genx
            .set_index(['year_modelx']) # FIXME: set constraint on no duplicates for id,year_modelx
            .assign(demand=lambda x: x.vin)
            # .set_crs(crs_ll)
        )
        
        zcta_ep=capture_endpoints(get_zcta_endpoints()).reset_index().assign(zip_code=lambda x: x.id.str.replace('^ZIP','',regex=True))
        
        ### now we're going to redistribute zips according to EMFAC model year distributions
        assert(~emfac_veh_zip_pop.model_year.isna().any() & ~emfac_veh_zip_pop.zip_code.isna().any())
        veh_zip_dist=(
            emfac_veh_zip_pop
            
            # remap zips to known zips
            .assign(o_id=lambda x: ('ZIP'+x.zip_code).map(z2id))
            .assign(zip_code_orig=lambda x: x.zip_code,
                    zip_code=lambda x: x.o_id.str.replace('ZIP',''))
            .groupby(['o_id','zip_code','model_year'])[['vehicle_population']].sum().reset_index()
            .assign(prop_of_my=lambda x: x.vehicle_population/x.groupby('model_year').vehicle_population.transform('sum'))
        )
        
        # we're going to redistribute the DMV data according to the EMFAC model year distributions. 
        # This is because the DMV data may not be representative of the actual distribution of vehicles by 
        # model year in each zip code since the last owner is likely the dismantler and not the original owner. 
        # The EMFAC data provides a more accurate distribution of vehicles by model year for each zip code, 
        # which we can use to adjust our demand estimates.
        logger.infox("Running ELV flow model: Assigning junked vehicles to zips")
        cars_gdf_exp=(
            # get the total demand by model year from the DMV data
            cars_gdf.reset_index()
            
            # now merge with the EMFAC data to get the proportion of vehicles by model year for each zip code
            .merge(
                veh_zip_dist
                #.assign(my_pop=lambda x: x.groupby('model_year').vehicle_population.transform('sum'))
                .add_col_suffix('_dist')
                ,left_on='year_modelx',right_on='model_year_dist',how='left'
                )
            
            # estimate demand source for each model year
            .assign(demand_src=lambda x: x.demand * x.prop_of_my_dist)
            
            # compute the rank of each zip by the total amount of demand
            .assign(zip_rank=lambda x: x.groupby(['year_modelx']).demand_src.transform('rank',ascending=False))
            .sort_values(['year_modelx','zip_rank'])
            .assign(
                # integer portion from this zip
                dem_int=lambda x: np.floor(x.demand_src),

                # fractional remainder for each zip
                dem_frac=lambda x: x.demand_src-x.dem_int,
                
                # remaining total fractional demand from this zip to distribute
                my_dem_int_rem=lambda x: x.demand - x.groupby('year_modelx').dem_int.transform('sum'),
                
                )
            .sort_values(['year_modelx','dem_frac'],ascending=[True,False])
            .assign(
                # now rank the demand fractions at each zip from largest to smallest
                dem_rank=lambda x: x.groupby('year_modelx').dem_frac.transform('rank',ascending=False,method='first'),
                
                # assign one to each zip until we've covered the total remaining demand from this zip
                dem_xtra=lambda x: np.where(x.dem_rank<= x.my_dem_int_rem,1,0),
                
                # compute the total demand from the integer and extra portions
                demandx=lambda x: x.dem_int + x.dem_xtra
            )
            
            # now recreate in proper format
            .rename(columns={'o_id_dist':'o_id'})
            [['o_id','zip_code_dist','year_modelx','demandx']]
            
            # keep only nonzero demand
            .filt(lambda x: x.demandx>0)        
            
            .groupby(['o_id','year_modelx']).agg({'demandx':'sum'})
            .reset_index()
            .assign(zip_code_dist=lambda x: x.o_id.str.replace('ZIP',''))
            .merge(zcta_ep
                ,left_on='zip_code_dist'
                ,right_on='zip_code'
                ,suffixes=('','_zip')
                ,how='left'
                )
            .rename(columns={'demandx':'demand'})
            .set_index(['o_id','year_modelx'])
            [['geometry','demand']]
            .pipe(lambda df: gpd.GeoDataFrame(df,geometry='geometry',crs=crs_ll))
        )
        
        assert cars_gdf.demand.sum() == cars_gdf_exp.demand.sum()
        
        # create the destinations dataframe with unique index
        cap_mult = 4
        def capfn(df):
            cap=df.veh_capacity*cap_mult
            return cap.mask(cap > cap.quantile(0.95), cap.quantile(0.95))
        
        dism_with_caps_adj=(
            gpd.GeoDataFrame(dism_with_caps,crs=crs_ll)
            .assign(capacity=lambda x: capfn(x))
            
            # okay, so some dismantlers are assigned to the same parcel, we'll deal
            # with this by dividing the capacity between them equally
            .rename(columns={'capacity':'capacity_raw'})
            .assign(capacity=lambda x: (
                x.groupby('geometry_agg').capacity_raw.transform('max')
                / x.groupby('geometry_agg').capacity_raw.transform('count')
                )
            )
            .set_crs(crs_ll)
        )

        
        capgrp=['d_id']
        demgrp=['o_id','year_modelx']
        
        # cf=Path('xxn.pkl')
        cf=self.zip_dism_cache_file
        if cf.is_file():
            logger.info(f"Reading cached assignment from {cf}")
            xxn=pd.read_pickle(cf)
        else:
            logger.info(f"Performing rank-based, capacity constrained vehicle zip to dismantler assignment")
            # attach potential dism, compute dist, and rank distance by id_zip, model_yearx combo
            i1=(
                cars_gdf_exp.to_crs(crs_ca)
                .sjoin(dism_with_caps_adj.to_crs(crs_ca)
                    .assign(geom_hold=lambda x: x.geometry)
                    ,predicate='dwithin',distance=250000,how='left',lsuffix='zip',rsuffix='dism')
                .rename(columns={'id':'d_id'})
                .assign(dist=lambda x: x.geometry.distance(x.geom_hold))
                .reset_index()
                .assign(drank=lambda x: x.groupby(['o_id','year_modelx']).dist.transform('rank','first'))
                .sort_values(['o_id','year_modelx','dist'])
                
                # i2
                .assign(mindrank=lambda x: x.groupby('d_id').drank.transform('min'))
                [['o_id','year_modelx','demand','d_id','capacity','dist','drank','mindrank']]
                .assign(
                    icap=lambda x: x.capacity.astype(int),
                    dem_1=lambda x: x.demand,
                    cap_1=lambda x: x.icap,
                )
                .reset_index(drop=True)
                
                # do sort here
                .sort_values(['mindrank','d_id','dist'])

            )
            i1dem_sum=i1.groupby(demgrp).demand.first().sum()
            xx=i1.filt(lambda x: x.drank==1)
            ranks=[]
        
            for i in range(i1.drank.max().astype(int)):
                irank=i+1
                dem=f'dem_{irank}'
                demx=f'demx_{irank}'
                nextdem=f'dem_{irank+1}'
                nextcap=f'cap_{irank+1}'
                cap=f'cap_{irank}'
                # flow=f'flow_{irank}'
                flow='flow'
                cumdem_dism=f'cumdem_dism_{irank}'
                cumdem_zy=f'cumdem_zip_year_{irank}'
                
                logger.info(f"Assigning rank {irank} of {i1.drank.max().astype(int)}")

                xxn=(
                    xx
                    .add_column_if_missing('flow_tot',0)
                    .assign(**{
                        # restrict demand to appropriate rank
                        demx: lambda x: x[dem].mask(x.drank != irank,0),
                        
                        # compute greedy assignment of cumulative demand
                        cumdem_dism:lambda x: x.groupby(capgrp)[demx].transform('cumsum'),
                        cumdem_zy:lambda x: x.groupby(demgrp)[demx].transform('cumsum'),

                        # apply capacity constraints
                        flow:lambda x: (
                            x[demx]
                            # if the last assignment to this dism is over capacity this one will be
                            .mask(x.groupby(capgrp)[cumdem_dism].shift(1).fillna(0)>x[cap],0)

                            # if the last assignment was under capacity and this one will be over, cap to the difference
                            .mask((x.groupby(capgrp)[cumdem_dism].shift(1).fillna(0)<=x[cap])
                                &(x[cumdem_dism]>x[cap]),x[cap]-x.groupby(capgrp)[cumdem_dism].shift(1).fillna(0))

                            # now to demand constraints
                            .mask(x.groupby(demgrp)[cumdem_zy].shift(1).fillna(0)>x[demx],0)
                            .mask((x.groupby(demgrp)[cumdem_zy].shift(1).fillna(0)<x[demx])
                                &(x[cumdem_zy]>x[demx]),x[demx]-x.groupby(demgrp)[cumdem_zy].shift(1).fillna(0))
                        ),
                        
                        'm_dism_cap_over': lambda x: x.groupby(capgrp)[cumdem_dism].shift(1).fillna(0)>x[cap],
                        'm_dism_cap_bndy': lambda x: (
                            (x.groupby(capgrp)[cumdem_dism].shift(1).fillna(0)<x[cap]) &(x[cumdem_dism]>x[cap])
                        ),
                        
                        'm_zy_dem_over': lambda x: x.groupby(demgrp)[cumdem_zy].shift(1).fillna(0)>x[demx],
                        'm_zy_dem_bndy': lambda x: (
                            (x.groupby(demgrp)[cumdem_zy].shift(1).fillna(0)<x[demx])&(x[cumdem_zy]>x[demx])
                        ),

                        'cumflw_dism':lambda x: x.groupby(capgrp)[flow].transform('cumsum'),
                        'cumflw_zip_year':lambda x: x.groupby(demgrp)[flow].transform('cumsum'),
                        
                        'flow_tot':lambda x: x.flow_tot + x[flow],
                        
                        nextcap: lambda x: x[cap]-x.groupby(capgrp)[flow].transform('sum'),
                        nextdem: lambda x: x[dem]-x.groupby(demgrp)[flow].transform('sum'),
                    })
                )
                assert(~(xxn[dem]<0).any())
                assert(~(xxn[cap]<0).any())
                
                sol=xxn[demgrp+capgrp+['demand','icap','drank',flow]]
                ranks.append(sol)
                
                kk=xxn[demgrp+capgrp+['demand','icap',dem,cap,nextdem,nextcap]]
            
                logger.info(f'Assigned {sum([df[flow].sum() for df in ranks])} of {i1.groupby(demgrp).demand.first().sum()}')
                
                if sum([df[flow].sum() for df in ranks]) >= i1dem_sum:
                    logger.info(f'Assignment complete')
                    break
                # checks
                # xxn.groupby(demgrp).agg({'demand':'first','dem_1':'first','flow_1':'sum','dem_2':'first','flow_2':'sum','dem_3':'first'}).sum()
                # xxn.groupby(capgrp).agg({'cap_1':'first','flow_1':'sum','cap_2':'first','flow_2':'sum','cap_3':'first'}).sum()
                xx=(
                    i1.filt(lambda x: x.drank==irank+1)
                    .merge(kk[capgrp+[cap,nextcap]].drop_duplicates(),on=capgrp,how='left',suffixes=('','_dup')).assign(**{nextcap:lambda x: x[nextcap].fillna(x[cap].fillna(x.icap)).astype(int)})
                    .merge(kk[demgrp+[dem,nextdem]].drop_duplicates(),on=demgrp,how='left',suffixes=('','_dup')).assign(**{nextdem:lambda x: x[nextdem].fillna(x[dem].fillna(x.demand)).astype(int)})
                )
                # xx=xx.drop(columns=xx.filter(regex=f'(_{irank-1}|_dup)$')) # get rid of older iterations
                xx=xx.drop(columns=xx.filter(regex=f'(_dup)$')) # get rid of older iterations
                
            logger.info(f'Caching assignment to {cf}')
            xxn=pd.concat(ranks).filt(lambda x: x.flow>0)
            xxn.to_pickle(cf)
            
        check= (
            xxn.groupby(capgrp).agg({'icap':'first','flow':'sum'})
            .assign(veh_capacity=lambda x: x.icap,cap2=lambda x: capfn(x))
            .filt(lambda x: x.flow>x.cap2)
        )
        if not check.empty:
            logger.warning(f'{len(check)} DISM exceed capacity')
            display(check)
        
        all_ep=pd.concat(all_endpoints).drop(columns=['zips_contained']).drop_duplicates()
        zip_to_dism_assigned=(
            xxn
            .set_flags(allows_duplicate_labels=True)
            .merge(all_ep[['geometry','n1']].add_col_prefix('o_').set_flags(allows_duplicate_labels=True),left_on='o_id',right_index=True,how='left')
            .merge(all_ep[['geometry','n1']].add_col_prefix('d_').set_flags(allows_duplicate_labels=True),left_on='d_id',right_index=True,how='left')
            .rename(columns={'flow':'allocated_cars'})[['o_id','year_modelx','d_id','allocated_cars','icap']]
        )
        
        # run the gravity model to allocate cars to dismantlers
        # table1_zip_to_dism = model3.allocate(
        #         cars_gdf, # geodataframe[Index, 'demand', 'point_geometry']
        #         dism_with_caps_adj,  # geodataframe[Index, 'capacity', 'geometry'
        #         max_distance_km=200
        #     ).rename(
        #         columns={'id_origin':'o_id',
        #                     'year_modelx_origin':'year_modelx',
        #                     'id_dest':'d_id',
        #                     'flow':'allocated_cars'}
        #     )

        # confirm no capacity violations
        capv=(
            zip_to_dism_assigned
            .groupby('d_id')[['allocated_cars']].sum().reset_index()
            .merge(dism_with_caps_adj.reset_index()[['id','veh_capacity']],
                left_on='d_id',right_on='id')
            .assign(allowed_cap=lambda x: capfn(x))
            .filt(lambda x: x.allocated_cars>x.allowed_cap)
            )
        if not capv.empty:
            logger.warning(f"{len(capv)} capacity violations")
            display(capv)
            logger.warning("...allowing for now")
        
        # we need to assign specific cars to dismantlers to match the capacitated
        # gravity model solution. we join the original dmv list only the
        # zip,dism,model_yearx marginal assignment
        dmv_gen_adj=(
            # listing of vehicles generated by o_id (zip), and adjusted model_year
            dmv_gen.copy()
        )
        
        # at this point zip_to_dism_assigned has the total flows for (zip, modelyear) combos to each dism
        # We need to map the adjusted list of junked vehicles 
        zip_to_dism_assigned_exploded=(
            zip_to_dism_assigned
            .filt(lambda x: x.allocated_cars>0)
            .reset_index(drop=True) # need to reset to make sure we properly loc each o,y,d triplet
            .pipe(lambda df:(
                # should confirm unique?
                df.loc[df.index.repeat(df.allocated_cars.astype(int))]
                .copy()
            ))
            .assign(
                car_idx=lambda x: x.groupby(demgrp + capgrp).cumcount()+1,
                # create an index ordering the assignment by model year
                # we're going to use this to allocate them to o+d combos
                my_idx=lambda x: x.groupby('year_modelx').cumcount()+1,
            )
            .reset_index(drop=True)
        )

        zip_to_dism_discrete=(
            dmv_gen_adj
            [['o_id_orig','year_modelx','year_modelx_orig','makex','modelx','vin','curb_weight_u']]
            .assign(
                # create an index ordering the assignment by model year
                # we're going to use this to allocate o+d combos from zip_to_dism_assigned_exploded
                my_idx=lambda x: x.groupby('year_modelx').cumcount()+1
            )
            .merge(
                zip_to_dism_assigned_exploded,
                left_on=['year_modelx','my_idx'],
                right_on=['year_modelx','my_idx'],how='left')
            .sort_values(demgrp + capgrp)
            # .filt(lambda x: x['index'].notna())
        )
        assert (zip_to_dism_discrete.groupby(demgrp+capgrp)[['vin']].count()
                .merge(zip_to_dism_assigned,left_index=True,right_on=['o_id','year_modelx','d_id'])
                .filt(lambda x: x.vin!=x.allocated_cars).empty), (
                    "DMV Assignment marginals failed to align"
                )
        assert zip_to_dism_discrete.vin.is_unique, "DMV Assignment has duplicate VINs"
                
        
        # table1_zip_to_dism = compute_zip_to_dism(dmv_gen,dism)
        # table1_zip_to_dism.to_csv(cpath('output_elv','zip_to_dism.csv'),index=False)

        # ---------------------------------
        # NEXT is DISMANTLER to SHREDDER
        
        # at this point, dism_assign allocates all vehicles in our junked vehicle dataset
        # to a (zip, dismantler) pair. we can apply the material compositions and recovery
        # rates to this

        # read in compositions and recovery rates
        composition = read_material_composition(vcfile)
        # vcomp = read_material_composition('ELV material composition - vehicle_composition.csv')
        recov_rates = read_recovery_rates(mrrfile)

        # convert compositions and rates into material-specific recycling rates of vehicle tonnage
        full_recov_rates=(
            composition.join(recov_rates.filt(lambda x: x.year==2020)
                            .drop_columns('year').set_index(['material'])
                            ,on='material')
            .assign(recyc_rate=lambda x: x.composition*x.recovery)
        )
        
        # assume 10 cars per trip to dismantler, FIXME: assess assumption
        pf_zip_to_dism = 1/10
        
        zip_to_dism=(
            zip_to_dism_discrete
            .groupby(['o_id','d_id'])
            .agg({
                'allocated_cars':'count',
                'curb_weight_u':'sum',
            })
            .reset_index() # get o,d back
            .assign(
                layer='ELV',
                material_subcategory='FIXME: VEHICLE',
                material_category='FIXME: VEHICLE',
                material_type='FIXME: VEHICLE',
                material_stream='FIXME: VEHICLE',
                material='VEHICLE',
                material_grouping='VEHICLE', # FIXME: standardize using mapping
                ttype='zip_to_dism',
                year=2021,
                EMFAC_class='T7 Tractor Class 8',
                trips=lambda x: np.ceil(x.allocated_cars * pf_zip_to_dism),
                wt_sent=lambda x: x.curb_weight_u.pint.to('short_ton'),  # convert to tons
            )
            [schema_to_cols(LayerFlowSchema)]
        )

        eptmp=pd.concat(all_endpoints).reset_index().drop_duplicates(subset='id').set_index('id')
        (dism_ep, dism_flows) = compute_dismantling(
            zip_to_dism_discrete
            .merge(eptmp[['geometry']].add_col_prefix('o_'),left_on='o_id',right_index=True)
            .merge(eptmp[['geometry']].add_col_prefix('d_'),left_on='d_id',right_index=True)
            , full_recov_rates)
        
        capture_endpoints(dism_ep)
        
        export_props:DataFrame[ExportDestinationProportions] = None
        if export_proportions_layer is not None:
            export_props=(
                # get all of the maritime (export) flows from the proportions layer
                export_proportions_layer.get_flows_with_endpoints()
                .filt(lambda x: x.pipe(has_maritime_leg_v2))
                # group them by material and destination and sum the weights
                .groupby(['material_grouping', 'd_id']).agg(wt_sent=('wt_sent', 'sum'))
                .reset_index()
                # now compute the proportions of each material going to each destination by weight
                .assign(proportion=lambda df: (
                    df.groupby('material_grouping')['wt_sent']
                    .transform(lambda x: x / x.sum())
                ).pint.magnitude)
            )

        (shred_ep,shred_flows) = compute_shredding(
            dism_flows
            .filt(lambda x: x.material_grouping.isin(['Ferrous','NonFerrous']))
            ,export_destinations=export_props
            )
        capture_endpoints(shred_ep)
        
        def compute_mexico_exports(
            total_mexico_exports: int,
            veh_zip_dist:         DataFrame,
            dmv_gen_my:           DataFrame
        ) -> Tuple[DataFrame[LayerEndpointSchema],DataFrame[LayerFlowSchema]]:

            # grab all endpoint data
            all_endpoints=[]
            def capture_endpoints(ep):
                all_endpoints.append(ep)
                return ep
            
            socal_cty=(
                pyg.counties(state='06',cache=True,year=self.model_year).clean_names()
                .filt(lambda x: x.name.isin([
                    'Los Angeles', 'Orange', 'San Diego', 'San Bernardino',
                    'Riverside', 'Ventura', 'Santa Barbara', 'San Luis Obispo',
                    'Imperial', 'Kern']))
            )
            zcta_endpoints=capture_endpoints(get_zcta_endpoints())
            socal_zcta=(
                zcta_endpoints[['geometry','geometry_agg']].set_geometry('geometry_agg').to_crs(crs_ca)
                .set_flags(allows_duplicate_labels=True)
                .sjoin(socal_cty.to_crs(crs_ca)[['name','geometry']]
                       .set_flags(allows_duplicate_labels=True)
                       ,predicate='intersects',how='inner')
                .pipe(lambda df: df[~df.index.duplicated(keep='first')])
                .set_flags(allows_duplicate_labels=False)
            )
            outflows=(
                veh_zip_dist
                .merge(socal_zcta,left_on='o_id',right_index=True,how='inner')
                # compute the proportion of the vehicle population assigned to
                # come from each zip code within each model year
                .assign(
                    prop_of_my_sc=lambda x: (
                        x.vehicle_population
                        / x.groupby('model_year').vehicle_population.transform('sum')
                    )
                )
                # now attach the mexico exports distributed by model year
                .merge(
                    dmv_gen_my.assign(year_modelx=lambda x: x.year_modelx.astype(int),
                                      gen_veh=lambda x: x.vin/x.vin.sum()*total_mexico_exports),
                    left_on='model_year',
                    right_on='year_modelx',
                    how='left'
                )
                .assign(
                    # the total model year flow from the origin zip
                    fflow=lambda x: x.gen_veh * x.prop_of_my_sc,
                    
                    # we don't want fractional flows, so compute the integer portion
                    iflow=lambda x: x.fflow.astype(int),
                    
                    # this is the remainder that still needs to be assigned
                    flowrem=lambda x: x.fflow-x.iflow
                )
                .filter(regex='o_id|model_year|flow|geometry') # keep the relevant columns
                .assign(
                    # rank the remainder from highest to lowest
                    myrank=lambda x: x.flowrem.rank(method='first',ascending=False),
                    
                    # assign the remainder for all ranks less than or equal to the 
                    # remaining flow to assign
                    extflow=lambda x: (
                        x.myrank.mask(x.myrank<=x.flowrem.sum(),1)
                        .mask(x.myrank>x.flowrem.sum(),0)
                    ),
                    
                    flow = lambda x: x.iflow + x.extflow,
                )
                [['o_id','model_year','geometry','flow']]
                .filt(lambda x: x.flow>0)
            )
            
            # assert that we didn't lose anything; however, we expect to lose
            # the fractional portion of the total flow so we floor the total exports
            assert np.isclose(outflows.flow.sum(),np.floor(total_mexico_exports))
            
            # flows should now have the mexico-bound vehicles for each model
            # year from each zip code. We're going to assume they go through
            # otay mesa or calexico
            land_port_endpoints = capture_endpoints(read_land_port_endpoints())
            
            zip_to_mex_mapping=compute_nearest_neighbor_mapping(
                'zip_to_mex',
                (
                    outflows[['o_id','geometry']].drop_duplicates(subset=['o_id'],keep='first')
                    .pipe(gpd.GeoDataFrame).rename_geometry('o_geometry')
                )
                ,'o_geometry',
                land_port_endpoints[['geometry']].reset_index().add_col_prefix('d_'),'d_geometry',
                'o_id',radius=400,max_options=3)

            # create o_id,model_year -> d_id flows
            pf = 1/10 # 1 trip for 8 cars
            flows=(
                outflows.merge(
                    zip_to_mex_mapping,
                    left_on='o_id',right_on='o_id',how='left'
                )
                [['o_id','d_id','model_year','flow']]
                .merge(
                    dmv_gen_my.assign(
                        year_modelx=lambda x: x.year_modelx.astype(int),
                        wt_sent=lambda x: (x.wt_sent / x.vin).pint.to('ton')
                        )
                    [['year_modelx','wt_sent']]
                    ,left_on='model_year',right_on='year_modelx',how='left'
                )
                .groupby(['o_id','d_id']).agg({'flow':'sum','wt_sent':'sum'})
                .assign(
                    year=self.model_year,
                    layer=self.name,
                    ttype='veh_to_mex',
                    material_stream='FIXME:VEHICLES',
                    material_grouping='FIXME:VEHICLES',
                    EMFAC_class='T7 Tractor Class 8', # FIXME!
                    trips = lambda x: x.flow * pf,
                )
                .reset_index()
                [schema_to_cols(LayerFlowSchema)]
            )
            used_endpoints=(
                pd.concat(all_endpoints)
                .filt(
                    lambda x: x.index.isin(flows.o_id.to_list() + flows.d_id.to_list())
                )
                [schema_to_cols(LayerEndpointSchema)]
            )
            return used_endpoints, flows
        
        # total exports is 
        num_legit_junked:int       = len(dmv2021_final)
        total_junked:float         = np.floor(num_legit_junked/(1-self.illegal_vehicles_fraction)).astype(int)
        total_mexico_exports:float = total_junked * self.mexico_export_fraction
        total_unknown:float        = total_junked - total_mexico_exports
        (mexico_ep,mexico_flows)   = compute_mexico_exports(total_mexico_exports, veh_zip_dist, dmv_genx)
        capture_endpoints(mexico_ep)
        

        # # CAR outputs to class 1 landfills
        # table2b_car_to_landfill1 = compute_car_to_landfill1(all_car_inflows,df_mc2)
        # (
        #     table2b_car_to_landfill1.assign(car=lambda x: x.car.astype(int)).reset_index(drop=True)
        #     .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table2b.csv'))
        # )

        # # Shredder outputs to ports
        # table3_shred_port, all_shred_inflows = compute_shredder_to_port(table2_car_to_shred,df_mc2)
        # (
        #     table3_shred_port.rename(columns={'n1_port': 'port'})
        #     [['shredder', 'ferrous_trips', 'nonferrous_trips', 'port']]
        #     .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table3.csv'))
        # )

        # # Shedder outputs to landfills handling CTMSR
        # table3b_shred_landfillctmsr = compute_shred_to_landfillctmsr(all_shred_inflows,df_mc2)
        # (
        #     table3b_shred_landfillctmsr.rename(columns={'landfill_lfc': 'landfill_ctmsr'})
        #     [['shredder', 'other_weight', 'other_trips', 'landfill_ctmsr']]
        #     .to_csv(cpath('output_appliance', 'OD_Appliances_Metals_table3b.csv'))
        # )

        # FIXME: do a mass balance check.
        
        all_endpoints = (
            pd.concat([
                # make sure all of the dataframes have the same crs before joining
                df.assign(geometry_agg=lambda x: (
                    x.geometry_agg.to_crs(crs_ll)
                    if x.geometry_agg.crs is not None 
                    else x.geometry_agg.set_crs(crs_ll)))
                .reset_index() # releases id
                for df in all_endpoints
            ],ignore_index=True)
            .drop_duplicates(subset='id',keep='first')
            .set_index('id')
        )
        assert(all_endpoints.index.is_unique)

        all_flows = (
            pd.concat(unify_pint_units([zip_to_dism,dism_flows, shred_flows, mexico_flows]), 
                      ignore_index=True)
            .assign(layer=self.name)
        )
        
        return (all_endpoints[schema_to_cols(LayerEndpointSchema)], 
                all_flows[schema_to_cols(LayerFlowSchema)]
                )
# %%
def read_land_port_endpoints() -> DataFrame[LayerEndpointSchema]:
    logger.info("Reading LAND PORTS")
    df = (
        read_rdrs_manual_geocodes()
        .filt(lambda x: (
            (x.facility_flags.str.upper().str.match(r'.*PORT'))
        ))
        .add_column_if_missing('geometry_agg',None)
    )
    return df[schema_to_cols(LayerEndpointSchema)]
