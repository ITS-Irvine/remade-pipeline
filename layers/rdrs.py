import os
import re
from typing_extensions import OrderedDict
import pandas as pd
import geopandas as gpd
import warnings
import janitor
import pint
# from services.geocode import collect_external_geocodes, geocode_here, geocode_nominatim, get_ca_cty, get_us_state, get_world_boundary
from services.geocode import (
    GeocodeById, Geocoder, MergeGeocoder, NullGeocoder, backfill_regions, crs_ll, get_ca_cty, get_ca_port_endpoints, get_county_endpoints, get_state_endpoints, get_world_endpoints,
    county_endpoint_prefix, read_rdrs_manual_geocodes, read_unify_geo, state_endpoint_prefix, country_endpoint_prefix, unknown_endpoint_prefix)
from layers.base import ModelLayer
from core.common import case_when, cpath, filt, add_col_suffix, strip_col_names, wdisplay, cachier
from IPython.display import display

import pygris as pyg

import shapely.geometry

from config.settings import settings
from config.base_config import BaseModelConfig
from core.model_types import *

import logging
from utils.logging_config import log_errors
logger = logging.getLogger(__name__)


# Define crs_ll (assuming it's a coordinate reference system)
crs_ll = "EPSG:4326"

# Define model_year (assuming it's a specific year for the model)

@cachier(wait_for_calc_timeout=5)
@pa.check_types
def load_rdrs_flow(
        model_year:int=2021
        ) -> DataFrame[RawRDRSData]:
    logger.info("Loading RDRS Flow data")
    rdrs_flowx_s=None
    with warnings.catch_warnings():
        # The RDRS excel file lacks a style, which causes an annoying warning. We filter this out.
        warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))

        rdrs_flowx_s=(
            pd.read_excel(cpath('rdrs_data_nda','Outflows summed by Quarter, Sender, Stream, Destination, and Material Type.xlsx'))
            .clean_names()
            .filt(lambda x: ~x.focal_rd.isna()) # this removes the total row and summaries
            .assign(
                # split out quarter and year
                year=lambda x: x.year_quarter.str.replace(r'\s+.*','',regex=True).apply(
                    pd.to_numeric, errors='coerce'
                ),
                quarter=lambda x: x.year_quarter.str.replace(r'^.*?\s+','',regex=True),
                # orig_rd=lambda x: f'{}'
            )
        )
        if model_year is not None:
            rdrs_flowx_s=rdrs_flowx_s.filt(lambda x: x.year==model_year)  # ONLY WORK WITH model_year flows
    
    logger.info("Loading RDRS Flow data: Extracting RDRS IDs")
    rdrs_flowx_s['focalfacility']=rdrs_flowx_s['focal_org']+'; '+rdrs_flowx_s['focal_entity']+'; RD'+rdrs_flowx_s['focal_rd'].astype(int).astype(str)
    rdrs_flowx_s[['o_n1','o_n2','o_id']]=rdrs_flowx_s['focalfacility'].str.split('; ', expand=True)
    rdrs_flowx_s[['d_n1','d_n2','d_id']]=rdrs_flowx_s['destination_org;_entity;_rd'].str.split('; ', expand=True)

    return rdrs_flowx_s.assign(
        ttype=lambda x: 'rdrs',
        wt_sent=lambda x: x.tonssent.astype('pint[short_ton]')
    )

@pa.check_types
@log_errors
def collect_rdrs_ods(
        rdrs_flowx_a: DataFrame[RawRDRSData]
    ) -> tuple[DataFrame[RDRSFullEndpointSchema], DataFrame[DetailedFlowIdSchema]]:

    logger.info("Collecting RDRS flow origins and destinations")
    # collect origins; these all have an RDRSID
    origs=(
        rdrs_flowx_a.groupby(['o_id','o_n1','o_n2']).first()
        [['focal_activity']].reset_index()
        .assign(
            src='RDRS_ORIG',
            aggregation=EndpointAggregation.LOCATION.value
            )
        .rename(columns={
            'focal_activity':'facility_type'
        })
    )

    # collect destinations that have an RDRSID
    dests_rd=(
        rdrs_flowx_a.filt(
            lambda x: ~x.d_id.isna()
        ).groupby(['d_id','d_n1','d_n2']).first()
        [['destination_activity_type','destination_county','destination_state','destination_country']].reset_index()
        .assign(
            src='RDRS_DEST',
            aggregation=EndpointAggregation.LOCATION.value
            )
        .rename(columns={
            'destination_activity_type':'facility_type'
        })
    )

    # collect destinations that do NOT have an RDRSID, we will use the (county, state, country) 
    # triplet to identify these
    logger.info("Creating mock RDRS IDs (nRDRS) for non-RDRS endpoints")
    nord = (
        rdrs_flowx_a.filt(lambda x: x.d_id.isna())
        [['destination_county','destination_state','destination_country']]
        .drop_duplicates()
    )

    # Here we're going to standardize on standard county, state, and country endpoints
    counties=get_ca_cty()
    nord_cty     = nord.filt(lambda x: x.destination_county.isin(counties.county)) # county endpoints
    nord_state   = nord.filt(lambda x: x.destination_county.str.match('Out of CA')) # state endpoints
    nord_country = nord.filt(lambda x: x.destination_county.str.match('Out of USA')) # country endpoints
    nord_unk = nord.filt(lambda x: x.destination_country.str.match('Unknown')) # country endpoints
    nord_other   = nord.filt(lambda x: ~x.index.isin(
        list(nord_cty.index) + list(nord_state.index) + list(nord_country.index) + list(nord_unk.index)))
    
    assert(len(nord_other)==0)
    
    dests_cty = (
        nord_cty.merge(
            get_county_endpoints().reset_index()
            ,left_on='destination_county',right_on='county',how='left')
    )
    assert(not dests_cty.id.isna().any())
    
    dests_state = (
        nord_state.merge(
            get_state_endpoints().reset_index()
            ,left_on='destination_state',right_on='state',how='left')
    )
    assert(not dests_state.id.isna().any())

    dests_country = (
        nord_country.merge(
            get_world_endpoints().reset_index()
            ,left_on='destination_country',right_on='country',how='left')
    )
    assert(not dests_country.id.isna().any())
    
    dests_unk = (
        nord_unk
        .reset_index().drop(columns=['index'])
        .reset_index()
        .assign(
            id=lambda x: x.reset_index().apply(lambda row: unknown_endpoint_prefix+str(row["index"]+1).zfill(5), axis=1),
            n1='UNK',
            n2='UNK',
            facility_type=None,
            facility_flags="UNKNOWN",
            aggregation=EndpointAggregation.NONE.value,
        )
        .drop(columns=['index'])
    )
    
    dests_nrd=(
        pd.concat([dests_cty,dests_state,dests_country,dests_unk],ignore_index=True)
        .assign(src='RDRS_DEST_NRD')
    )
    
    # dests_nrd=(
    #     # select all dests without an RDRS id at this point
    #     nord
    #     .assign(
    #         id=lambda x: x.reset_index().apply(lambda row: 'nRD'+str(row["index"]).zfill(5), axis=1),
    #         n1=lambda x: x.destination_county,
    #         n2=lambda x: x.destination_state
    #     )
    #     .assign(
    #         src='RDRS_DEST_NRD',
    #         aggregation=lambda x: case_when(
    #             x.destination_county.str.match('Out of USA'), EndpointAggregation.COUNTRY.value,
    #             x.destination_county.str.match('Out of CA'), EndpointAggregation.STATE.value,
    #             x.destination_county.isin(counties.county), EndpointAggregation.COUNTY.value,
    #             x.destination_country.isin(['Unknown']), EndpointAggregation.NONE.value,
    #             True, None
    #         )
    #     )
    # )

    # merge all destinations togther.
    logger.info("Merging all RDRS entities")
    rdrs_flow_ent_s=(
        pd.concat([
            origs.rename(columns={
                'o_id':'id',
                'o_n1':'n1',
                'o_n2':'n2'
            }),
            dests_rd.rename(columns={
                'd_id':'id',
                'd_n1':'n1',
                'd_n2':'n2'
            }),
            dests_nrd.set_flags(allows_duplicate_labels=True)
        ])
        .sort_values(['id','destination_county']).assign(
            gcnt=lambda x: x.groupby('id').n1.transform('size')
        )
        .assign(
            # make sure facility flags is a string, even if null
            facility_flags=lambda x: x.facility_flags.fillna('')
        )
        .groupby(['id']).agg({
            'n1': lambda x: '|'.join(x.unique()),
            'n2': lambda x: '|'.join(x.unique()),
            'facility_type': 'first',
            'facility_flags': lambda x: '|'.join(x.unique()),
            'destination_county': 'first',
            'destination_state': 'first',
            'destination_country': 'first',
            'aggregation':'first',
            'gcnt':'first',
            'src':'first'
        })
        .rename(columns={
            'destination_county':'county',
            'destination_state':'state',
            'destination_country':'country'
        })
        .set_flags(allows_duplicate_labels=False)
    )
    logger.info(f'{len(rdrs_flow_ent_s)} total endpoints')

    logging.info("Backfilling nRDRS endpoints onto flow data")
    # Update destinations in rdrs_flow
    rdrs_flow_s=(
        rdrs_flowx_a
        .join(rdrs_flow_ent_s
            .reset_index()
            # we only want to add override values for these
            .filt(lambda x: x.id.str.match(f'^({county_endpoint_prefix}|{state_endpoint_prefix}|{country_endpoint_prefix}|{unknown_endpoint_prefix})'))
            .set_index(['county','state','country'])[['id']].add_col_suffix('_nrd')
            ,on=['destination_county','destination_state','destination_country'])
        .assign(
            d_n1=lambda x: case_when(x.d_id.isna(),x.destination_county,
                                            True,x.d_n1),
            d_n2=lambda x: case_when(x.d_id.isna(),x.destination_state,
                                            True,x.d_n2),
            d_id=lambda x: case_when(x.d_id.isna(),x.id_nrd,
                                            True,x.d_id)
        )
    )

    # Summarize the origin and destination characteristics in the flow records.

    # display(rdrs_flow_s[['o_id','d_id']].describe())
    if na_count := rdrs_flow_s.o_id.isna().sum():
        raise ValueError(f"{na_count} NAs for o_id") 
    if na_count := rdrs_flow_s.d_id.isna().sum():
        display(rdrs_flow_s.filt(lambda x: x.d_id.isna()))
        raise ValueError(f"{na_count} NAs for d_id") 
    logger.info(f"{len(pd.concat([
        rdrs_flow_s.set_flags(allows_duplicate_labels=True).o_id,
        rdrs_flow_s.set_flags(allows_duplicate_labels=True).d_id
        ]).unique())} unique RDRSIDs")
    if len(dups:=(rdrs_flow_s.filt(lambda x: x['destination_org;_entity;_rd'].isna()).fillna("")
            .groupby(['destination_org;_entity;_rd','destination_county','destination_state','destination_country','d_id'])
            [['quarter']].count()
            .reset_index()
            .groupby(['destination_county','destination_state','destination_country'])[['d_id']].count()
            .filt(lambda x: x.d_id>1)
        )) > 0:
        raise ValueError(f"{len(dups)} duplicated nRDRSIDs")

    return (rdrs_flow_ent_s,rdrs_flow_s)



@pa.check_types
def get_rdrs_flow_for_year(
    rdrs_flow: DataFrame[DetailedFlowIdSchema],
    model_year: int,
    matmap: DataFrame[RDRSMaterialSchema],
    layer_name: str
    ) -> DataFrame: #[AnnualizedLayeredGroupedFlowIdSchemaWithFlows]:

    payload_factor=pint.Quantity(1/20.0,'1/short_ton')

    logger.info(f"Aggregating RDRS flows")
    logger.info(f"Aggregating RDRS flows: converting truck trips using payload factor of {payload_factor:.3f}")

    ret = (
        rdrs_flow
        .join(matmap.set_index(['material_category','material_subcategory','material_type']),
             on=['material_category','material_subcategory','material_type'])
        .rename(columns={'grouping4':'material_grouping'})
        .groupby([
            'year','o_n1','o_n2','o_id',
            # 'origin_county','origin_state','origin_country',
            'd_n1','d_n2','d_id',
            'destination_county','destination_state','destination_country',
            'material_stream',
            'material_grouping'
        ]).agg({'wt_sent':'sum'})
        .reset_index()
        .filt(lambda x: x.year==model_year)
        .assign(
            EMFAC_class='T7 Tractor Class 8',
            trips=lambda x: x.wt_sent * payload_factor, # FIXME:UNIT
            layer=layer_name,
            ttype='rdrs'
        )
    )
    logger.info(f"Aggregating RDRS flows: {len(ret)} flow records")
    return(ret)


#%%
def is_rdrs_entity(df):
    return df.id.str.match('^RD')
def is_rdrs_swis_entity(df):
    return ~df.swis_number.isna()
def has_swis_geocode(df):
    return ~df.geometry_swis.isna()
def has_man_geocode(df):
    return ~df.geometry_man.isna()
def is_rdrs_nswis_entity(df):
    return df.id.str.match('^RD') & df.swis_number.isna()
def is_xx_entity(df):
    return df.id.str.match('^XX')
def is_car_entity(df):
    return df.id.str.match('^CAR')
def is_zip_entity(df):
    return df.id.str.match('^ZIP')
def is_port_entity(df):
    return df.id.str.match('^PORT')
def is_recl_entity(df):
    return df.id.str.match('^RCL')

from config.cli_field import cli_field

@dataclass
class RDRSModelConfig(BaseModelConfig, prefix="rdrs"):

    # ── Internal fields ───────────────────────────────────────────────────────
    matmap_file: str = field(default="")
    name:        str = field(default="RDRS")

    # ── CLI-exposed fields ────────────────────────────────────────────────────
    model_year: int = cli_field(
        2021,
        help    = "Model year for RDRS calculations",
        metavar = "MODEL_YEAR"
    )

    # # ── Runtime dependency — set in code, not CLI or TOML ────────────────────
    # geocoder: Geocoder = field(default=None)


class RDRSLayer(ModelLayer):
    name = 'RDRS'
    def __init__(self,
                 config: RDRSModelConfig, 
                 geocoder: Geocoder = None
                 ):
        """Initialize the RDRS layer with given data."""
        super().__init__(self.name)
        self.name       = config.name
        self.model_year = config.model_year
        self.geocoder   = geocoder

        self.rdrs_flows = None
        self.rdrs_flow_ent = None

        self.matmap = self.read_material_mappings(config.matmap_file)

        # Does the work to read flows and entities into standard forms
        self.read_flows()
        

    @pa.check_types
    def read_material_mappings(self,matmap_file) -> DataFrame[RDRSMaterialSchema]:
        """Read material mappings for RDRS layer."""
        self.matmap = pd.read_excel(matmap_file).clean_names()
        return self.matmap

    @pa.check_types
    def get_endpoints(self) -> DataFrame[LayerEndpointSchema]:
        """Get the geocoded endpoints of RDRS layer."""
        assert(self.rdrs_flows is not None)
        # return self.rdrs_data[['o_rdrsid', 'd_rdrsid']].drop_duplicates()
        return (
            self.rdrs_flow_ent
            .add_column_if_missing('geometry_agg',None)
            [schema_to_cols(LayerEndpointSchema)]
        )
        
    @pa.check_types
    def get_geocoder(self) -> Geocoder:
        if not self.geocoder:
            geocoders = OrderedDict()
            geocoders['rdrs_man']  = GeocodeById(read_rdrs_manual_geocodes(),'Manual geocodes')
            # FIXME: temporary template instantiation
            ff=settings.get('model').unify_geo_file.format(**settings.get('model').to_dict())
            geocoders['unify_geo'] = GeocodeById(read_unify_geo(ff),'Unified geocodes')
            
            # ── Administrative Boundary Geocoders
            # Fallback geocoders that resolve entities to county, state, or country
            # centroids when no more specific geocode is available.
            geocoders['counties']      = GeocodeById(get_county_endpoints(),'County endpoints')
            geocoders['states']        = GeocodeById(get_state_endpoints(),'State endpoints')
            geocoders['countries']     = GeocodeById(get_world_endpoints(),'Country endpoints')

            # ── Port Geocoder
            # Geocode California ports by ID. Used for maritime export routing.
            geocoders["caports"] = GeocodeById(get_ca_port_endpoints(),'California ports')
            
            # ── Null Geocoder (Safety Net)
            # Catch any remaining entities with invalid/empty geometries that
            # no other geocoder resolved. Must be last in the stack.
            geocoders['null_geocodes'] = NullGeocoder(
                select=lambda df: ~df.set_geometry('geometry').geometry.is_valid
            )

            self.geocoder = MergeGeocoder(geocoders,require_complete=True)
        return self.geocoder

    @pa.check_types
    def read_flows(self):
        logger.info("Reading RDRS Flow Data")
        self.rdrs_flows=(
            load_rdrs_flow(model_year=self.model_year)
         
            # Required for FlowSchema   
            .add_column_if_missing('layer',self.name)
            
            # FIXME: hard code fix for bogus destination
            # RDRS has a location labeled as Ashmore and Cartier Islands, which are uninhabited reefs
            # in the north of Australia. Just assume Australia for now
            .assign(destination_country=lambda x: x.destination_country.mask(x.destination_country.str.match('Ashmore'),'Australia'))
        )

        logger.info("Reading RDRS Flow Data...collecting")
        # takes flows, identifies all endpoints, remaps flow endpoints onto the identified endpoints
        (self.rdrs_flow_ent,self.rdrs_flows)=collect_rdrs_ods(self.rdrs_flows)

        # index has to be named 'id' for EndpointSchema
        # FIXME: this is enforced by the EndpointSchema if type checking is enabled
        self.rdrs_flow_ent.index.rename('id',inplace=True)
        
        # remove known bad flows
        is_bad_endpoint=self.rdrs_flow_ent.index.str.match('^UNK')
        if is_bad_endpoint.any():
            bad_endpoints = list(self.rdrs_flow_ent[is_bad_endpoint].index)
            logger.info(f"Removing predefined {len(bad_endpoints)} from RDRS flows: {bad_endpoints}")
            self.rdrs_flow_ent=self.rdrs_flow_ent[~is_bad_endpoint]
            is_bad_flow = self.rdrs_flows.o_id.isin(bad_endpoints) | self.rdrs_flows.d_id.isin(bad_endpoints)
            if is_bad_flow.any():
                logger.info(f"RDRS flows: removing {is_bad_flow.sum()} flow records involving bad endpoints")
                removed_flows_be = self.rdrs_flows.filt(lambda x: is_bad_flow)
                self.rdrs_flows = self.rdrs_flows.filt(lambda x: ~is_bad_flow)
                rem_flows_tot=removed_flows_be.wt_sent.sum()
                all_flows_tot=self.rdrs_flows.wt_sent.sum()
                logger.warning(f'{len(removed_flows_be)} affected flows removed totaling {rem_flows_tot}/{all_flows_tot} = {
                    (rem_flows_tot/all_flows_tot).magnitude*100:.1f}% of mass')

        # geocode all endpoints using the passed geocoder
        logger.info("Reading RDRS Flow Data...geocoding")
        self.rdrs_flow_ent = self.get_geocoder().geocode(self.rdrs_flow_ent)
        
        is_zero_flow = self.rdrs_flows.wt_sent == 0
        if is_zero_flow.any():
            logger.info(f"RDRS flows: removing {is_zero_flow.sum()} flow records with zero wt_sent")
            self.rdrs_flows = self.rdrs_flows.filt(lambda x: ~is_zero_flow)
        
        null_geoms = self.rdrs_flow_ent.geometry.isna() | ~self.rdrs_flow_ent.geometry.is_valid
        strict = False
        if null_geoms.any():
            if not strict:
                logger.warning(f'{sum(null_geoms)} entities without geocodes...dropping')
                display(self.rdrs_flow_ent[null_geoms])
                all_flows_tot=self.rdrs_flows.wt_sent.sum()
                self.rdrs_flow_ent = self.rdrs_flow_ent[~null_geoms]
                null_ents = null_geoms[null_geoms].index.tolist()
                self.rdrs_flows_removed= self.rdrs_flows.filt(lambda x: x.o_id.isin(null_ents) | x.d_id.isin(null_ents))
                rem_flows_tot=self.rdrs_flows_removed.wt_sent.sum()
                logger.warning(f'{len(self.rdrs_flows_removed)} affected flows removed totaling {rem_flows_tot}/{all_flows_tot} = {rem_flows_tot/all_flows_tot*100:.1f}%')
                self.rdrs_flows = self.rdrs_flows.filt(
                    lambda x: ~x.o_id.isin(null_ents) 
                            & ~x.d_id.isin(null_ents)
                )
            else:
                logger.error(f'{len(null_geoms)} entities without geocodes...stopping')
                wdisplay(null_geoms)
                raise(f'{len(null_geoms)} entities without geocodes')

        oos_origin = (
            ~self.rdrs_flows
            .join(self.rdrs_flow_ent[['geometry']].add_col_prefix('o_'),on='o_id',how='left')
            .set_geometry('o_geometry').to_crs(crs_ll)
            .intersects(pyg.states().set_index('STUSPS').to_crs(crs_ll).loc['CA'].geometry)
        )
        if oos_origin.any():
            if not strict:
                logger.warning(f'{oos_origin.sum()} flows with out-of-state origins...dropping')
                display(self.rdrs_flows[oos_origin])
                all_flows_tot=self.rdrs_flows.wt_sent.sum()
                self.rdrs_flows_removed=self.rdrs_flows[oos_origin]
                self.rdrs_flows = self.rdrs_flows[~oos_origin]
                rem_flows_tot=self.rdrs_flows_removed.wt_sent.sum()
                logger.warning(f'{len(self.rdrs_flows_removed)} affected flows removed totaling {rem_flows_tot}/{all_flows_tot} = {rem_flows_tot/all_flows_tot*100:.1f}%')
            else:
                logger.error(f'{oos_origin.sum()} flows with out-of-state origins...stopping')
                wdisplay(self.rdrs_flows[oos_origin])
                raise(f'{oos_origin.sum()} flows with out-of-state origins')


        # map every point to a county, state, and country
        logger.info("Reading RDRS Flow Data...backfilling regions")
        self.rdrs_flow_ent = backfill_regions(self.rdrs_flow_ent)

        # trims the flow data to the model year and applies material mapping
        logger.info("Reading RDRS Flow Data...material mappings")
        self.rdrs_flows = get_rdrs_flow_for_year(self.rdrs_flows,self.model_year,self.matmap,self.name)
        # rename columns to match FlowSchema
        self.rdrs_flows=self.rdrs_flows.rename(
            columns={
                'o_rdrsid':'o_id',
                'd_rdrsid':'d_id',
                'grouping4':'material_grouping'})

    @pa.check_types
    def get_flows(self) -> DataFrame[LayerFlowSchema]:
        """Get the flows from RDRS layer."""
        if self.rdrs_flows is None:
            self.read_flows()
        assert(self.rdrs_flows is not None)
        
        # logger.info(f"RDRS flows: removing {zero_flows.sum()} zero flows and {oos_origin.sum()} out-of-state origins")

        return (
            self.rdrs_flows
            # FIXME: downstream checks will potentially fail if we keep nonzero flows
            # .filt(lambda x: ~zero_flows & ~oos_origin)
            .merge(self.get_endpoints().reset_index().add_prefix('o_'),left_on='o_id',right_on='o_id')
            .merge(self.get_endpoints().reset_index().add_prefix('d_'),left_on='d_id',right_on='d_id')
            [schema_to_cols(LayerFlowSchema)]
        )

    def get_materials(self) -> DataFrame[RDRSMaterialSchema]:
        """Get the materials handled by RDRS layer."""
        return self.matmap
    
