

from pathlib import Path
from typing_extensions import OrderedDict

import pandas as pd
import re
from services.geocode import Geocoder, MergeGeocoder, NullGeocoder, crs_ll, extract_geometry, geocode_it, get_world_boundary
from geopy.geocoders import Nominatim
from layers.base import ModelLayer
import janitor
import pygris as pyg
import pickle
from shapely.geometry import shape
from geopy.extra.rate_limiter import RateLimiter
from IPython.display import display

from core.common import case_when, cpath, wdisplay, ldisplay, cachier
from config.base_config import BaseModelConfig
from config.path_template import PathTemplate
from core.model_types import *
from config.settings import settings

import logging
from utils.logging_config import log_errors
logger = logging.getLogger(__name__)


from config.cli_field import cli_field

@dataclass
class CTModelConfig(BaseModelConfig, prefix="ct"):

    model_year: int = cli_field(
        2021,
        help    = "Model year for CTDEEP calculations",
        metavar = "MODEL_YEAR_CT"
    )

    ct_data_dir:   Path = field(default="")

    ct_flows_file: PathTemplate = cli_field(
        None,
        help    = "CT Flows",
    )

    matmap_file:   str  = field(default="")
    name:          str  = field(default="Connecticut")


class CTLayer(ModelLayer):
    def __init__(self, 
                 config: CTModelConfig,
                 geocoder: Geocoder = None
                 ):
        """Initialize the CT layer with given data."""
        super().__init__(config.name)
        self.model_year = config.model_year
        
        self.flows = None
        self.flow_ent = None

        self.geocoder = geocoder

        self.use_ct_cache = True
        self.ct_flows_file=config.ct_flows_file #settings['model_files']['ct_flows']

        # self.matmap = self.read_material_mappings(matmap_file)
        self.matmap = self.read_material_mappings(self.ct_flows_file)
        
        # forces flow/ent initialization
        self.get_flows()
                
    @pa.check_types
    def get_geocoder(self) -> Geocoder:
        if not self.geocoder:
            geocoders: OrderedDict[str,Geocoder] = OrderedDict()
            
            geocoders['null_geocodes'] = NullGeocoder(
                select=lambda df: ~df.set_geometry('geometry').geometry.is_valid
            )

            # ── Merge Geocoder
            # Combine all geocoders into a single MergeGeocoder that applies
            # them in insertion order, using the first successful result.
            self.geocoder = MergeGeocoder(geocoders,require_complete=True)
            
        # FIXME: CT layer doesn't ever actually use its geocoder
        
        return self.geocoder

    @pa.check_types
    def read_material_mappings(self,matmap_file) -> DataFrame:# FIXME:[RDRSMaterialSchema]:
        """Read material mappings for CT layer."""
        # self.matmap = pd.read_excel(matmap_file).clean_names()
        mm1 = pd.read_excel(matmap_file, sheet_name='recycling-mappings').clean_names()[['material','category']]
        mm2 = pd.read_excel(matmap_file, sheet_name='recycling-mappings-summary').clean_names()[['category_list','grouping4']]
        self.matmap = (
            mm1.merge(mm2,
                      left_on=('category'),
                      right_on=('category_list'),
                      how='left')
        )
        return self.matmap
    
    @pa.check_types
    def get_endpoints(self) -> DataFrame[LayerEndpointSchema]:
        """Get the geocoded endpoints of CT layer."""
        if self.flow_ent is None:
            (self.flows,self.flow_ent) = CTLayer.read_flows(self.ct_flows_file)
        # return self.rdrs_data[['o_rdrsid', 'd_rdrsid']].drop_duplicates()
        return self.flow_ent[schema_to_cols(LayerEndpointSchema)]

    @staticmethod
    @cachier(wait_for_calc_timeout=5)
    @pa.check_types
    def read_flows(layer_name:str,
                   ct_file:Path=None,
                   ) -> DataFrame: # FIXME return type
        logger.info("Reading CT Flow Data")

        # for sht in [
        ####################################################
        sht = 'Received by CT SW Facilities'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',
            'receiving_facility_name_': 'd_n1',
            'receiving_facility_city': 'd_city',
            'receiving_facility_state': 'd_state',
            'full_address': 'd_n2',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'receiving_facility_description': 'd_type',

            'origin_facility_city_ia': 'o_city',
            'origin_facility_state_ia': 'o_state',
            'origin_facility_description_ia': 'o_type',

            'specialdescription_if_applicable': 'material_ct_detail'
        }
        sw_in=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                o_n1 = lambda x: case_when(x.origin_facility_if_applicable.notna(), x.origin_facility_if_applicable,
                                            x.origin_townname.notna(), x.origin_townname,
                                            True, None),
                o_n2 = lambda x: case_when(x.origin_facility_if_applicable.isna() & x.origin_townname.notna(), "[TOWN]",
                                            ~x.origin_facility_if_applicable.isna(), x.origin_facility_address1_ia+' '+x.o_city+' '+x.o_state,
                                            True, None),

                # if comes from a town, set the o_type
                o_type = lambda x: case_when(x.origin_facility_if_applicable.isna() & x.origin_townname.notna(), "TOWN",
                                                True, x.o_type),

                # if it's a town, we set the state to CT for these flows
                o_state = lambda x: case_when((x.o_type=='TOWN') & x.o_state.isna(), 'CT',
                                                True, x.o_state),

                ttype = 'sw_in',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Collection',

            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['^o_','^d_','^tons_','^ttype','EMFAC_class','material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = sw_in.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = sw_in.columns.difference(tons_columns)
        sw_in=(
            sw_in.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(sw_in.head(10))
        
        ####################################################
    #     'Outgoing from SW Facilities': {},
        sht = 'Outgoing from SW Facilities'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'origin_facility_name': 'o_n1',
            'full_address1': 'o_n2',
            'latitude': 'o_lat',
            'longitude': 'o_lon',
            'origin_facility_description': 'o_type',
            'origin_facility_city': 'o_city',
            'origin_facility_state': 'o_state',

            'receiving_facility_name': 'd_n1',
            'receiving_facility_city': 'd_city',
            'receiving_facility_state': 'd_state',
            'full_address2': 'd_n2',
            'latitude_1': 'd_lat',
            'longitude_1': 'd_lon',
            'receiving_facility_type': 'd_type',

            'itemdescription': 'material_ct_detail',
            'tons_quantitysent': 'wt_sent'
        }
        sw_out=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'sw_out',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Recycling/Composting',
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype','^o_','^d_', 'EMFAC', 'material_stream'])
            )
        )
        # omit=sw_out.filter(regex='orig_fac')
        # keep=
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(sw_out.head(10))


        ####################################################
        #     'SW Outgoing from Transfer': {},
        sht = 'SW Outgoing from Transfer'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'tbl003_facility_name': 'o_n1',
            'tbl003_facility_address1': 'o_street',
            'tbl003_facility_city': 'o_city',
            'tbl003_facility_state': 'o_state',
            # 'latitude': 'o_lat',
            # 'longitude': 'o_lon',
            'tbl005_facilitytype_description': 'o_type',

            'tbl003_facility_1_name': 'd_n1',
            'tbl003_facility_1_address1': 'd_street',
            'tbl003_facility_1_city': 'd_city',
            'tbl003_facility_1_state': 'd_state',
            # 'latitude_1': 'd_lat',
            # 'longitude_1': 'd_lon',
            'tbl005_facilitytype_1_description': 'd_type',

            'specialdescription': 'material_ct_detail',
            'tons_quantitysent': 'wt_sent'
        }
        sw_trans_out=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'sw_trans_out',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Recycling/Composting',
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype','^o_','^d_', 'EMFAC', 'material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = sw_trans_out.filter(regex='^sumof').columns
        # and keep the rest of the columns
        id_vars      = sw_trans_out.columns.difference(tons_columns)
        sw_trans_out=(
            sw_trans_out.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^sumof','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
        )
        # FIXME: This ^^^ is broken. Flows are specified in sumof* columns that need to be split out
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(sw_trans_out.head(10))


        ####################################################
    #     'SWReceived by Transfer Station': {},
        sht = 'SWReceived by Transfer Station '
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            # 'origin_townname': 'o_n1',
            # 'tbl003_facility_address1': 'o_street',
            # 'tbl003_facility_city': 'o_city',
            # 'tbl003_facility_state': 'o_state',
            # 'latitude': 'o_lat',
            # 'longitude': 'o_lon',
            # 'tbl005_facilitytype_description': 'o_type',

            'receving_transfer_station_name': 'd_n1',
            'full_address': 'd_n2',
            'receving_transfer_station_city': 'd_city',
            'receving_transfer_station_state': 'd_state',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'receving_transfer_station_facilitytype': 'd_type',
            'specialdescription_if_applicable': 'material_ct_detail'
        }
        sw_trans_in=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                o_n1 = lambda x: x.origin_townname,
                o_n2 = '[TOWN]',
                o_city = lambda x: x.origin_townname,
                o_type = 'TOWN',
                o_state = 'CT', # all from connecticut
                ttype = 'sw_trans_in',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Collection',
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['^tons_','^o_','^d_','ttype', 'EMFAC', 'material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = sw_trans_in.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = sw_trans_in.columns.difference(tons_columns)
        sw_trans_in=(
            sw_trans_in.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(sw_trans_in.head(10))



        ####################################################
    #     'Received by Recycling Facility': {},
        sht = 'Received by Recycling Facility'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            # 'origin_town_name_': 'o_n1',
            # 'tbl003_facility_address1': 'o_street',
            # 'tbl003_facility_city': 'o_city',
            # 'tbl003_facility_state': 'o_state',
            # 'latitude': 'o_lat',
            # 'longitude': 'o_lon',
            'origin_facility_type_if_applicable': 'o_type',

            'recycling_facility_name': 'd_n1',
            'full_address': 'd_n2',
            'recycling_facility_city': 'd_city',
            'recycling_facility_state': 'd_state',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'recycling_facility_type': 'd_type',
            'itemdescription': 'material_ct_detail'
        }
        rf_in=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                o_n1 = lambda x: case_when(x.origin_facility_name_if_applicable.notna(), x.origin_facility_name_if_applicable,
                                            x.origin_town_name_.notna(), x.origin_town_name_,
                                            True, None),
                o_n2 = lambda x: case_when(x.origin_facility_name_if_applicable.isna() & x.origin_town_name_.notna(), "[TOWN]",
                                            x.origin_facility_name_if_applicable.notna(), None,
                                            True, None),
                o_city = lambda x: x.origin_town_name_,

                # if comes from a town, set the o_type
                o_type = lambda x: case_when(x.o_type.notna(), x.o_type,
                                                x.origin_facility_name_if_applicable.isna() & x.origin_town_name_.notna(), "TOWN",
                                                True, x.o_type),
                ttype = 'rf_in',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Collection',
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['^tons_','^o_','^d_','ttype', 'EMFAC', 'material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = rf_in.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = rf_in.columns.difference(tons_columns)
        rf_in=(
            rf_in.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )

            .filt(lambda x: x.wt_sent>0)
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(rf_in.head(10))




        ####################################################
    #     'Marketed by Recycling Facility': {},
        sht = 'Marketed by Recycling Facility'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'recycling_facility_name': 'o_n1',
            'full_address1': 'o_n2',
            'latitude': 'o_lat',
            'longitude': 'o_lon',
            'recycling_facility_type': 'o_type',
            'recycling_facility_state': 'o_state',

            'end_market_name': 'd_n1',
            'end_market_city': 'd_city',
            'end_market_state': 'd_state',
            'full_address2': 'd_n2',
            'latitude_1': 'd_lat',
            'longitude_1': 'd_lon',
            'end_market_type': 'd_type',

            'itemsubtypedescription': 'material_ct_detail',
            'tons_marketed': 'wt_sent'
        }
        rf_out=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'rf_out',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = lambda x: case_when(x.d_type.fillna('').str.match('End User'), 'End Use',
                                                      x.d_type.fillna('').str.match('Recycling Facility'), 'Recycling/Composting',
                                                      x.d_type.fillna('').str.match('Scrap Metal Processor'), 'Recycling/Composting',
                                                      x.d_type.fillna('').str.match('Marketing Group'), 'End Use',
                                                      x.d_type.fillna('').str.match('Broker'), 'Brokering/Transporting',
                                                      x.d_type.fillna('').str.match('VRF'), 'Solid Waste for Disposal',
                                                      x.d_type.fillna('').str.match('Composting'), 'Recycling/Composting',
                                                      x.d_type.fillna('').str.match('Waste Oil Transporter'), 'Brokering/Transporting',
                                                      x.d_type.fillna('').str.match('Multi-Town Transfer Station'), 'Brokering/Transporting',
                                                      x.d_type.fillna('').str.match('Landfill'), 'Solid Waste for Disposal',
                                                      True, 'UNKNOWN'),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype', 'EMFAC_class', 'material_stream'])
            )
        )
        # omit=rf_out.filter(regex='orig_fac')
        # keep=
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(rf_out.head(10))



        ####################################################
    #     'Disposed as Recycling Residual': {},
        sht = 'Disposed as Recycling Residual'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'recycling_facility_name_': 'o_n1',
            'full_address1': 'o_n2',
            'latitude': 'o_lat',
            'longitude': 'o_lon',
            'recycling_facility_type': 'o_type',
            'recycling_facility_city': 'o_city',
            'recycling_facility_state': 'o_state',

            'disposal_facility_name': 'd_n1',
            'disposal_facility_city': 'd_city',
            'disposal_facility_state': 'd_state',
            'full_address2': 'd_n2',
            'latitude_1': 'd_lat',
            'longitude_1': 'd_lon',
            'disposal_facility_type': 'd_type',

            'itemsubtypedescription': 'material_ct_detail',
            'tons_residual_disposed': 'wt_sent'
        }
        rf_out_disp=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'rf_out_disp',
                material_ct = 'Recycling Residual',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Designated Waste for Disposal',
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype', 'material_ct','^o_','^d_', 'EMFAC_class', 'material_stream'])
            )
        )
        # omit=rf_out_disp.filter(regex='orig_fac')
        # keep=
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(rf_out_disp.head(10))




        ####################################################
    #     'VRF Received': {},
        sht = 'VRF Received'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            # 'origin_facility_name_if_applicable_': 'o_n1',
            # 'full_address2': 'o_n2',
            # 'latitude': 'o_lat',
            # 'longitude': 'o_lon',
            'origin_facility_type_if_applicable_': 'o_type',
            'origin_facility_city_if_applicable_': 'o_city',
            'origin_facility_state_if_applicable_': 'o_state',

            'vrf_facility_name': 'd_n1',
            'full_address1': 'd_n2',
            'vrf_facility_city': 'd_city',
            'vrf_facility_state': 'd_state',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'vrf_facility_type': 'd_type',

            'itemdescription': 'material_ct_detail'
        }
        vrf_in=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                # make the townname null if town not selected
                origin_townname = lambda x: x.origin_townname.mask(x.origin_townname=='Town not Selected',None),

                # name 1 is facility, then town name, then city
                o_n1 = lambda x: (x.origin_facility_name_if_applicable_
                                    .combine_first(x.origin_townname)
                                    .combine_first(x.o_city)),
                o_n2 = lambda x: x.full_address2,

                # use city, then origin_townname
                o_city = lambda x: x.o_city.combine_first(x.origin_townname),
                o_state = lambda x: case_when(x.o_state.notna(), x.o_state,
                                                x.origin_townname.notna(), 'CT', # if the down
                                                True, None),

                # if comes from a town, set the o_type
                o_type = lambda x: case_when(x.o_type.notna(), x.o_type,
                                                x.origin_townname.notna(), "TOWN",
                                                True, x.o_type),
                ttype = 'vrf_in',

                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Designated Waste for Disposal'
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['^tons_','ttype','^o_','^d_', 'EMFAC_class','material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = vrf_in.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = vrf_in.columns.difference(tons_columns)
        vrf_in=(
            vrf_in.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )

            .filt(lambda x: x.wt_sent>0)
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(vrf_in.head(10))




        ####################################################
    #     'VRF Marketed': {},
        sht = 'VRF Marketed'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'vrf_facility_name': 'o_n1',
            'unnamed_5': 'o_n2',
            'latitude': 'o_lat',
            'longitude': 'o_lon',
            'vrf_facility_type': 'o_type',
            'vrf_facility_city': 'o_city',
            'vrf_facility_state': 'o_state',

            'end_market_name_': 'd_n1',
            'end_market_city': 'd_city',
            'end_market_state': 'd_state',
            'full_address2': 'd_n2',
            'latitude_1': 'd_lat',
            'longitude_1': 'd_lon',
            'end_market_type': 'd_type',

            'itemsubtypedescription': 'material_ct_detail',
            'tons_marketed': 'wt_sent'
        }
        vrf_out=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'vrf_out',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'End Use',
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype','^o_','^d_','EMFAC_class','material_stream'])
            )
        )
        # omit=rf_out.filter(regex='orig_fac')
        # keep=
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(vrf_out.head(10))




        ####################################################
    #     'VRF Disposed': {},
        sht = 'VRF Disposed'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'vrf_facility_name': 'o_n1',
            'full_address1': 'o_n2',
            'latitude': 'o_lat',
            'longitude': 'o_lon',
            'vrf_facility_type': 'o_type',
            'vrf_facility_city': 'o_city',
            'vrf_facility_state': 'o_state',

            'disposal_facility_name': 'd_n1',
            'disposal_facility_city': 'd_city',
            'disposal_facility_state': 'd_state',
            'full_address2': 'd_n2',
            # 'latitude_1': 'd_lat',
            # 'longitude_1': 'd_lon',
            'disposal_facility_type': 'd_type',

            'specialdescription': 'material_ct_detail',
        }
        vrf_out_disp=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                ttype = 'vrf_out_disp',
                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Designated Waste for Disposal'
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype', 'material_ct', '^tons_','^o_','^d_', 'material_stream', 'EMFAC_class'])
            )
        )

        # want to melt the tonnage
        tons_columns = vrf_out_disp.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = vrf_out_disp.columns.difference(tons_columns)
        vrf_out_disp=(
            vrf_out_disp.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )

            .filt(lambda x: x.wt_sent>0)
        )

        # omit=rf_out_disp.filter(regex='orig_fac')
        # keep=
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(vrf_out_disp.head(10))



        #     'SUPPLEMENTAL Muni Rep Recycling': {},
        sht = 'SUPPLEMENTAL Muni Rep Recycling'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'origin_reporting_townname': 'o_n1',

            'receiving_recycling_facility_name_': 'd_n1',
            'receiving_recycling_facility_city': 'd_city',
            'receiving_recycling_facility_state': 'd_state',
            'full_address': 'd_n2',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'receiving_recycling_facility_type': 'd_type',

            'itemdescription': 'material_ct_detail',
        }
        muni_rep_in=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                o_n2= '[TOWN]',
                o_state= 'CT',
                o_country= 'United States',
                o_type= 'TOWN',
                ttype= 'muni_rep_in',

                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Recycling/Composting'
            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype', 'material_ct', '^tons_','^o_','^d_', 'EMFAC_class', 'material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = muni_rep_in.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = muni_rep_in.columns.difference(tons_columns)
        muni_rep_in=(
            muni_rep_in.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )

            .filt(lambda x: x.wt_sent>0)
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(muni_rep_in.head(10))

        #     'SUPPLEMENTAL Muni Rep Disposed': {},
        sht = 'SUPPLEMENTAL Muni Rep Disposed'
        logger.info(f"Reading CT flow sheet: {sht}")
        name_map={
            'reportyear': 'model_year',

            'townname': 'o_n1',

            'name': 'd_n1',
            'city': 'd_city',
            'state': 'd_state',
            'full_address': 'd_n2',
            'latitude': 'd_lat',
            'longitude': 'd_lon',
            'description': 'd_type',

        }
        muni_rep_in_disp=(
            pd.read_excel(ct_file, sheet_name=sht)
            .clean_names()
            .rename(columns=name_map)
            .assign(
                o_n2= '[TOWN]',
                o_state= 'CT',
                o_country= 'United States',
                o_type= 'TOWN',
                ttype= 'muni_rep_in_disp',

                EMFAC_class = 'T7 Tractor Class 8',
                material_stream = 'Solid Waste for Disposal'

            )
            .filter(
                regex='|'.join(list(name_map.values()) + ['ttype', 'material_ct', '^tons_','^o_','^d_', 'EMFAC_class', 'material_stream'])
            )
        )
        # want to melt the tonnage
        tons_columns = muni_rep_in_disp.filter(regex='^tons_').columns
        # and keep the rest of the columns
        id_vars      = muni_rep_in_disp.columns.difference(tons_columns)
        muni_rep_in_disp=(
            muni_rep_in_disp.melt(
                id_vars = id_vars,
                value_vars = tons_columns,
                var_name = 'material_ct',
                value_name = 'wt_sent'    # to align with RDRS 
            )
            .assign(
                # the material will be given as 'tons_<mat>' and we only want the <mat> portion
                material_ct = lambda x: x.material_ct.str.replace(r'^tons_','',regex=True),
                wt_sent = lambda x: x.wt_sent.fillna(0).astype('pint[short_ton]')
            )

            .filt(lambda x: x.wt_sent>0)
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            wdisplay(muni_rep_in_disp.head(10))




        ###########################################
        ###########################################
        all_flows=pd.concat([
            sw_in,sw_out,
            sw_trans_out,sw_trans_in,
            rf_in,rf_out,rf_out_disp,
            vrf_in, vrf_out, vrf_out_disp,
            muni_rep_in, muni_rep_in_disp
        ])

        all_flows


        ## OBTAIN ENDPOINTS FROM FLOWS
        o_list=(
            all_flows.filter(regex='^o_|ttype')
            # .assign(
            #     o_lon=lambda x: x.o_lon.fillna(-9999),
            #     o_lat=lambda x: x.o_lat.fillna(-9999)
            #     )

            # rename o_cols to just be endpoint cols (no o_/d_)
            .rename(columns=dict(
                zip(all_flows.filter(regex='^o_').columns,
                [re.sub(r'o_','',c) for c in all_flows.filter(regex='^o_').columns])
                ))
            # .groupby(['n1','type','ttype','lon','lat']).agg({'n2':'nunique'})
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            ldisplay(o_list)
        d_list=(
            all_flows.filter(regex='^d_|ttype')
            # .assign(
            #     # dummies for counts
            #     # d_lon=lambda x: x.d_lon.fillna(-9999),
            #     # d_lat=lambda x: x.d_lat.fillna(-9999),
            #     # d_n2 =lambda x: x.d_n2.fillna('NA')
            # )
            # rename d_cols to just be endpoint cols (no o_/d_)
            .rename(columns=dict(
                zip(all_flows.filter(regex='^d_').columns,
                [re.sub(r'd_','',c) for c in all_flows.filter(regex='^d_').columns])
                ))
            # .groupby(['n1','type','ttype','lon','lat']).agg({'n2':'nunique'})
        )
        if logger.getEffectiveLevel() <= logging.DEBUG:
            ldisplay(d_list)

        endpoints = (
            pd.concat([
                o_list.reset_index(),
                d_list.reset_index()
            ])
            .assign(
                city = lambda x: x.city.mask(x.city=='Town not Selected',None),
            )
            .add_column_if_missing('aggregation', None)
        )

        # omit unidentified sites in CT
        omit=endpoints.filt(lambda x: x.n1.fillna('UNIDENTIFIED').str.match('UNIDENTIFIED') & (x.state=='CT'))
        endpoints=(
            endpoints[~endpoints.n1.isin(omit.n1.unique())]
            .add_column_if_missing('geometry_src', None)
        )

        ep_calc=(
            endpoints
            .groupby(['n1'])
            .agg({
                'n2':'nunique',
                'lat':'nunique',
                'lon':'nunique'
                })
        )
        err_geo=ep_calc.filt(lambda x: (x.lat>1) | (x.lon>1))
        if len(err_geo):
            logger.error(f"{len(err_geo)} Named CT endpoints with differing locations")
            display(
                endpoints.filt(lambda x: x.n1.isin(err_geo.reset_index().n1))
                # .groupby('n1')
                )
            strict = False
            if not strict:
                logger.warning(f"...will use first lat lon")
                endpoints=(
                    endpoints.groupby(
                        ['n1']
                    )
                    .first()
                    .reset_index()
                    [['n1', 'n2', 'street', 'city', 'state', 'type', 'ttype', 'lat', 'lon', 'aggregation']]
                )
            else:
                raise ValueError(f"{len(err_geo)} Named CT endpoints with differing locations")
            
        # backfill null ll by n1 matches
        good_geocodes = (
            endpoints.groupby('n1')
            .agg({'lat':'first','lon':'first','city':'first','state':'first'})
            .filt(lambda x: x.lat.notna() & x.lon.notna())
        )
        endpoints = (
            endpoints.merge(
                good_geocodes.add_col_suffix('_good')
                ,left_on=['n1']
                ,right_on=['n1']
                ,how='left'
            )
            .assign(
                lat=lambda x: x.lat.combine_first(x.lat_good),
                lon=lambda x: x.lon.combine_first(x.lon_good),
                geometry_src = lambda x: case_when(x.lat.notna() & x.lon.notna(), 'CT GEOCODES',
                                                   True, None),
                aggregation = lambda x: case_when(x.geometry_src=='CT GEOCODES',EndpointAggregation.LOCATION.value,
                                                  True, x.aggregation)
            )
            .drop_columns(regex='_good$')
        )

        ep_calc=(
            endpoints
            .groupby(['n1','type','ttype'])
            .agg({
                'n2':'nunique',
                'lat':'nunique',
                'lon':'nunique'
                })
        )
        err_geo=ep_calc.filt(lambda x: (x.lat>1) | (x.lon>1))
        if len(err_geo):
            logger.error(f"{len(err_geo)} Named CT endpoints with differing locations")
            display(
                endpoints.filt(lambda x: x.n1.isin(err_geo.reset_index().n1))
                # .groupby('n1')
                )
            raise ValueError(f"{len(err_geo)} Named CT endpoints with differing locations")
            

        # identify towns that look like states
        have_network=False
        if have_network:
            us_states = (
                pyg.states(year=2021, cb=True, cache=True).clean_names()
                .rename_geometry('geometry_agg')
                .assign(
                    name_up=lambda x: x.name.str.upper()
                )
                [['name_up','geometry_agg','stusps']]
            )
        else:
            us_states=(
                gpd.read_file(cpath('gis_data','cb_2018_us_state_500k.zip'))
                .clean_names()
                .rename_geometry('geometry_agg')
                .assign(
                    name_up=lambda x: x.name.str.upper()
                )
                [['name_up','geometry_agg','stusps']]
            )

        # join the states by name on n1 and overwrite the info and geometry if there's a match
        if 'geometry_agg' not in endpoints.columns:
            endpoints['geometry_agg'] = None
        if 'geometry' not in endpoints.columns:
            endpoints['geometry'] = None
        endpoints=(
            gpd.GeoDataFrame(
                endpoints.merge(
                    us_states.assign(
                            type_targ='TOWN', # we're looking to capture items labeled as TOWNs
                            n2='[STATE]',
                            type='STATE',
                            state=lambda x: x.stusps
                        )
                        .add_col_suffix('_st'),
                    left_on=['n1','type'],
                    right_on=['name_up_st','type_targ_st'],
                    how='left'
                )
                ,geometry='geometry_agg',crs=us_states.crs)
            .assign(
                # use the state override if we have a match
                n2=lambda x: x.n2_st.combine_first(x.n2),
                type=lambda x: x.type_st.combine_first(x['type']),
                state=lambda x: x.state_st.combine_first(x.state),
                geometry_agg=lambda x: x.geometry_agg.combine_first(x.geometry_agg_st),
                geometry=lambda x: x['geometry'].combine_first(x.geometry_agg.representative_point()),
                geometry_src=lambda x: case_when(
                    x.geometry_src.notna(),x.geometry_src,
                    x['geometry'].notna(), 'CT STATE LOOKUP',
                ),
                aggregation = lambda x: case_when(x.geometry_src=='CT STATE LOOKUP',EndpointAggregation.STATE.value,
                                                  True, x.aggregation)
            )
            .set_geometry('geometry')
            .assign(
                lon=lambda x: x.lon.combine_first(x['geometry'].x),
                lat=lambda x: x.lat.combine_first(x['geometry'].y)
                )
        )
        endpoints=endpoints.drop(columns=endpoints.filter(regex='_st$').columns)


        ## fill in town geocodes
        town_states = endpoints.filt(lambda x: (x['type']=='TOWN'))
        all_tiger=(
            pd.concat([pyg.places(st,cb=True, year=2021, cache=True).assign(state_abbr=st).rename_geometry('geometry_agg')
                        for st in [p for p in town_states.state.unique() if not pd.isna(p)]])
            .clean_names()
        )
 
        # here we join the tiger place data on town names to get the geometry
        endpoints = (
            gpd.GeoDataFrame(
                endpoints.merge(
                    all_tiger[['name','state_abbr','geometry_agg']].assign(
                        name_up=lambda x: x.name.str.upper(),
                        type='TOWN'  # to force merge only on town types
                        ),
                    left_on=['n1','state'],
                    right_on=['name_up','state_abbr'],
                    suffixes=('','_tig'),
                    how='left'
                    )
                ,geometry='geometry_agg',crs=all_tiger.crs
                )
            .assign(
                geometry = lambda x: x.geometry_agg.representative_point(),
                # lon = lambda x: case_when(x.lon.isna() & x.geometry.notna(), x.geometry.x,
                #                 True, x.lat),
                # lat = lambda x: case_when(x.lat.isna() & x.geometry.notna(), x.geometry.y,
                #                 True, x.lat),
                lon = lambda x: x.apply(lambda row: row.geometry.x if not pd.isna(row.geometry) else row.lon, axis=1),
                lat = lambda x: x.apply(lambda row: row.geometry.y if not pd.isna(row.geometry) else row.lat, axis=1),

                geometry_src=lambda x: case_when(
                    x.geometry_src.notna(), x.geometry_src,
                    x['geometry'].notna(), 'CT PLACE LOOKUP'
                ),
                aggregation=lambda x: case_when(
                    x.geometry_src=='CT PLACE LOOKUP', EndpointAggregation.CITY.value,
                    True, x.aggregation
                )

            )
        )
        endpoints=endpoints.drop(columns=endpoints.filter(regex='_tig$').columns)

        to_nom_geo_f = 'ct_to_nom_geo'
        to_nom_geo = None
        try:
            with open(cpath('model_cache_dir', f'{to_nom_geo_f}.pickle'), 'rb') as pfile:
                to_nom_geo = pickle.load(pfile)
            logger.info("Read CT nominatim data from cache")
        except FileNotFoundError:
            logger.info("CT nominatim cache mssing...recalculating")

        if to_nom_geo is None:

            # fill in the remaining with nominatim
            nom_local=False
            if not nom_local:
                rl_delay=1
                geolocator = Nominatim(user_agent="remade_uci")
            else:
                rl_delay=0
                geolocator = Nominatim(user_agent="remade_uci",
                                    domain='localhost:8080')
                
            valid_states=us_states.stusps.unique()
            to_nom=(
                pd.concat([
                    endpoints.filt(lambda x: (x['type']=='TOWN') & x.state.notna() & x.lat.isna())
                    [['n1','state']],
                    endpoints.filt(lambda x: x.city.notna() & ~x.city.fillna('').str.match('Town not Selected') & x.state.notna() & x.lat.isna())
                    [['city','state']].rename(columns={'city':'n1'})
                ],ignore_index=True)
                .filt(lambda x: x.state.isin(valid_states))
                .groupby(['n1','state']).count()[[]].reset_index()
            )
            geocode = RateLimiter(geocode_it(geolocator, len(to_nom),verbose=True), 
                                min_delay_seconds=rl_delay)


            to_nom_geo=gpd.GeoDataFrame(
                to_nom
                .assign(
                    location = lambda x: (x.n1 + ', ' +x.state).apply(geocode),
                    geometry_agg = lambda x: x.location.apply(extract_geometry)
                )
                , geometry='geometry_agg', crs=crs_ll
            )
            logger.info("CT Nominatim data read. Caching...")
            to_nom_geo.to_pickle(cpath('model_cache_dir', f'{to_nom_geo_f}.pickle'))

        # geocode [n1,state] combos that match nominatim city lookup
        endpoints=(
            endpoints.merge(
                to_nom_geo.add_col_suffix('_nom'),
                left_on=['n1','state'],
                right_on=['n1_nom','state_nom'],
                suffixes=('','_nom'),
                how='left'
            )
            .assign(
                geometry_agg=lambda x: x.geometry_agg.combine_first(x.geometry_agg_nom),
                geometry=lambda x: x['geometry'].combine_first(x.geometry_agg.representative_point()),
                geometry_src="NOM PLACE LOOKUP",
                aggregation=EndpointAggregation.CITY.value
            )
            .set_geometry('geometry')
            .assign(
                lon=lambda x: x.lon.combine_first(x['geometry'].x),
                lat=lambda x: x.lat.combine_first(x['geometry'].y)
                )
            .drop(columns=['n1_nom','state_nom','location_nom','geometry_agg_nom'])
        )


        ### Backfill on [city,state] combos that match nom city lookup if geocode failed to this point
        endpoints=(
            endpoints.merge(
                to_nom_geo.add_col_suffix('_nom'),
                left_on=['city','state'],
                right_on=['n1_nom','state_nom'],
                suffixes=('','_nom'),
                how='left'
            )
            .assign(
                geometry_agg=lambda x: x.geometry_agg.combine_first(x.geometry_agg_nom),
                geometry=lambda x: x['geometry'].combine_first(x.geometry_agg.representative_point()),
                geometry_src='NOM CITY/STATE LOOKUP',
                aggregation=EndpointAggregation.CITY.value
            )
            .set_geometry('geometry')
            .assign(
                lon=lambda x: x.lon.combine_first(x['geometry'].x),
                lat=lambda x: x.lat.combine_first(x['geometry'].y)
                )
            .drop(columns=['n1_nom','state_nom','location_nom','geometry_agg_nom'])
        )

        # get supplemental geocodes
        geo_ext=pd.read_excel(ct_file, sheet_name='extra-manual-geocodes').clean_names()
        endpoints=(
            endpoints
            .add_column_if_missing('country',None)
            .merge(
                geo_ext[['id','addr','country','lat','lon']]
                .add_col_suffix('_ext')
                ,right_on=('id_ext')
                ,left_on=('n1')
                ,how='left'
            )
            .assign(
                n2=lambda x: x.n2.combine_first(x.addr_ext),
                country=lambda x: x.country.combine_first(x.country_ext),
                lat=lambda x: x.lat.combine_first(x.lat_ext),
                lon=lambda x: x.lon.combine_first(x.lon_ext)
            )
            .drop_columns(regex='_ext$')
            
        )

        # match to country if no geom but country is set
        endpoints=(
            endpoints
            .assign(ctrytmp=lambda x: x.country.fillna('UNK').str.upper())
            .merge(
                get_world_boundary().assign(
                    country=lambda x: x.country.fillna('').str.upper(),
                    n2='[COUNTRY]'
                    )[['n2','country','geometry']]
                .set_geometry('geometry')
                .to_crs(crs_ll)
                .add_col_suffix('_ctry')
                ,right_on='country_ctry'
                ,left_on='ctrytmp'
                ,how='left')
            .assign(
                n2=lambda x: x.n2.combine_first(x.n2_ctry),
                lat=lambda x: x.lat.combine_first(x.geometry_ctry.y),
                lon=lambda x: x.lon.combine_first(x.geometry_ctry.x),
            )
            .drop_columns(regex='_ctry$')
        )

        ### Create geometries if we have lat lon and they haven't been created yet
        endpoints=(
            endpoints
            .assign(
                geotmp=lambda x: gpd.points_from_xy(x.lon,x.lat),
                geotmp2=lambda x: case_when(x.geotmp.is_valid, x.geotmp,
                                            True, None),
                geometry_src=lambda x: case_when(
                    x.geometry.isna() & x.geotmp.is_valid, 'CT lat/lon',
                    True, x.geometry_src
                ),
                aggregation=lambda x: case_when(
                    x.geometry.isna() & x.geotmp.is_valid, 
                        case_when(x.n2.fillna('').str.match(r'.*\[(CITY|TOWN)\]'), EndpointAggregation.CITY.value,
                                  x.n2.fillna('').str.match(r'.*\[(STATE)\]'),     EndpointAggregation.STATE.value,
                                  x.n2.fillna('').str.match(r'.*\[(COUNTRY)\]'),   EndpointAggregation.COUNTRY.value,
                                  True, x.aggregation),
                    True, x.aggregation
                ),
                geometry=lambda x: x.geometry.combine_first(x.geotmp2),
            )
            .drop_columns(regex='^geotmp')
        )

        ### FIXME: MAKE SURE n2 has been defined
        endpoints=(
            endpoints.assign(n2=lambda x: x.n2.fillna('UNKNOWN'))
        )

        # now do some checks


        ep_calc=(
            endpoints
            .groupby(['n1'])
            .agg({
                'lat':'nunique',
                'lon':'nunique'
                })
        )
            
        warn_geo=ep_calc.filt(lambda x: ((x.lat==0) | (x.lon==0) ))

        # # last chance geocoding
        # nom_local=False
        # if not nom_local:
        #     rl_delay=1
        #     geolocator = Nominatim(user_agent="remade_uci")
        # else:
        #     rl_delay=0
        #     geolocator = Nominatim(user_agent="remade_uci",
        #                         domain='localhost:8080')
            
        # valid_states=us_states.stusps.unique()
        # to_nom2=(
        #     pd.concat([
        #         endpoints.filt(lambda x: x.n1.isin(warn_geo.index) & x.n2.notna())
        #         [['n1','n2']],
        #     ],ignore_index=True)
        #     .groupby(['n1','n2']).count()[[]].reset_index()
        # )
        # geocode = RateLimiter(geocode_it(geolocator, len(to_nom2),verbose=True), 
        #                     min_delay_seconds=rl_delay)

        # to_nom_geo2=gpd.GeoDataFrame(
        #     to_nom2
        #     .assign(
        #         location = lambda x: (x.n2).apply(geocode),
        #         geometry_agg = lambda x: x.location.apply(extract_geometry)
        #     )
        #     , geometry='geometry_agg', crs=crs_ll
        # )

        # endpoints=(
        #     endpoints.merge(
        #         to_nom_geo2.add_col_suffix('_nom'),
        #         left_on=['n1'],
        #         right_on=['n1_nom'],
        #         suffixes=('','_nom'),
        #         how='left'
        #     ))










        ep_calc=(
            endpoints
            .groupby(['n1'])
            .agg({
                'lat':'nunique',
                'lon':'nunique'
                })
        )
            
        warn_geo=ep_calc.filt(lambda x: ((x.lat==0) | (x.lon==0) ))

        if len(warn_geo):
            logger.warning(f"{len(warn_geo)} Named CT endpoints with no geocode")
            ldisplay(
                endpoints.filt(lambda x: x.n1.isin(warn_geo.reset_index().n1))
                .groupby(['n1']).agg({
                    'n2': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None,
                    'street':'first',
                    'city':'first',
                    'state': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None,
                    'type':'first',
                    'lat':'nunique',
                    'lon':'nunique',
                    'ttype': lambda x: ', '.join(map(str, x.unique()))
                })
                .reset_index()
            )

        # REMOVE ALL ENDPOINTS WITHOUT A GEOCODE
        stlu=pyg.states(year=2021,cb=True,cache=True).clean_names().set_index('stusps')[['name']].to_dict()['name']
        endpoints_final = (
            endpoints.filt(lambda x: ~x.n1.isin(warn_geo.reset_index().n1))
            .add_column_if_missing('county',None)
            .add_column_if_missing('state',None)
            .add_column_if_missing('country',None)
            .assign(
                state=lambda x: x.apply(lambda row: stlu.get(row.state), axis=1),
            )
            .reset_index()
            .assign(
                id=lambda x: case_when(x.state=='Connecticut','CT',
                                       True,'nCT')+(x["index"]+1).astype(str).str.zfill(5),
                # county='UNKNOWN',
                # state=lambda x: x.state.fillna('UNKNOWN'),
                # country='UNKNOWN'
            )
            .drop(columns='index')
            .rename(columns={'type':'facility_type'}) # required for model_type
            .assign(
                facility_flags='' # required for model_type
            )
        )

        ep_calc=(
            endpoints_final
            .groupby(['id','facility_type','ttype'])
            .agg({
                'n2':'nunique',
                'lat':'nunique',
                'lon':'nunique'
                })
        )

        warn_names=ep_calc.filt(lambda x: (x.n2>1))
        if len(warn_names):
            logger.warning(f"{len(warn_names)} Named CT endpoints with differing details")
            display(
                endpoints_final.filt(lambda x: x.n1.isin(warn_names.reset_index().n1))
                .groupby(['id']).agg({
                    'n2':'nunique',
                })
                .reset_index()
            )

        ep_calc=(
            endpoints_final
            .groupby(['id'])
            .agg({
                'n1':'nunique',
                'n2':'nunique',
                'lat':'nunique',
                'lon':'nunique'
                })
        )

        warn_names=ep_calc.filt(lambda x: (x.lat>1))
        if len(warn_names):
            logger.warning(f"{len(warn_names)} Named CT endpoints with differing geocodes")
            display(
                endpoints_final.filt(lambda x: x.n1.isin(warn_names.reset_index().n1))
                .groupby(['id']).agg({
                    'lat':'nunique',
                    'lon':'nunique'
                })
                .reset_index()
            )

            strict = False
            if not strict:
                logger.warning(f"...will use first lat lon")
                endpoints_final=(
                    endpoints_final.groupby(
                        ['n1']
                    )
                    .first()
                    .reset_index()
                    [['n1', 'n2', 'street', 'city', 'state', 'facility_type', 'ttype', 'lat', 'lon']]
                )
            else:
                raise ValueError(f"{len(err_geo)} Named CT endpoints with differing locations")
        
        # (re-)geocode geoms
        
        
        flow_ent=gpd.GeoDataFrame(
            endpoints_final
            .assign(geometry = lambda x: gpd.points_from_xy(x.lon,x.lat))
            ,geometry='geometry',crs=crs_ll
        ).set_index('id')


        # really make sure we have countries
        flow_ent=(
            flow_ent.to_crs(crs_ll)
            .sjoin_nearest(get_world_boundary().clean_names()[['country','geometry_agg']]
                   .add_col_suffix('_ct').set_geometry('geometry_agg_ct').to_crs(crs_ll)
                   ,how='left')
            .assign(country=lambda x: x.country.combine_first(x.country_ct))
            .drop_columns(regex='index_right|_ct$')
        )

        # really make sure we have states
        flow_ent=(
            flow_ent.to_crs(crs_ll)
            .sjoin(pyg.states(year=2021,cb=True,cache=True).clean_names()[['stusps','name','geometry']]
                   .add_col_suffix('_st').set_geometry('geometry_st').to_crs(crs_ll)
                   ,predicate='covered_by',how='left')
            .assign(state=lambda x: x.state.combine_first(x.name_st)
                    .combine_first(x.state.mask(~x.country.str.match('United States of America'),
                                                'Out of USA')))
            .drop_columns(regex='index_right|_st$')
        )
        # really make sure we have counties
        flow_ent=(
            flow_ent.to_crs(crs_ll)
            .sjoin(pyg.counties(year=2021,cb=True,cache=True).clean_names()[['name','geometry']]
                   .add_col_suffix('_cty').set_geometry('geometry_cty').to_crs(crs_ll)
                   ,predicate='covered_by',how='left')
            .assign(county=lambda x: x.county.combine_first(x.name_cty)
                    .combine_first(x.state.mask(~x.country.str.match('United States of America'),
                                                'Out of USA')))
            .drop_columns(regex='index_right|_cty$')
        )

        pf=1/20 # FIXME:UNIT
        flows=(
            all_flows.merge(
                endpoints_final.add_col_prefix('o_')
                ,left_on=['o_n1']
                ,right_on=['o_n1']
                ,suffixes=('_orig_o','')
                # ,suffixes=('','')
                ,how='left'
            )
            .merge(
                endpoints_final.add_col_prefix('d_')
                ,left_on=['d_n1']
                ,right_on=['d_n1']
                # ,suffixes=('','')
                ,suffixes=('_orig_d','')
                ,how='left'
            )
            .drop_columns(regex='_orig_d|_orig_o')
            .rename(columns={
                'material_ct_detail': 'material_type',
                'material_ct':'material'
            })
            # .assign(
            #     material=lambda x: if 
            # )
            .assign(
                # FIXME: need mappings
                material_category= 'TBD',
                material_subcategory= 'TBD',
                material_type=lambda x: x.material_type.combine_first(x.material).fillna('UNKNOWN')
            )
        )
        flows = (
            flows.filt(
                lambda x: x.o_id.notna() & x.d_id.notna()
            )
            .assign(
                layer=layer_name,
                year=2021,
                material=lambda x: x.material.fillna('UNKNOWN'),
                trips=lambda x: x.wt_sent * pf  # FIXME:UNIT: payload factor should be material-type specific
            )
        )
        return (flows,flow_ent)
            
    
    @pa.check_types
    def get_flows(self) -> DataFrame[LayerFlowSchema]:
        """Get the flows from CT layer."""
        if self.flows is None:
            (self.flows,self.flow_ent) = CTLayer.read_flows(
                self.name,
                self.ct_flows_file
                )
        assert(self.flows is not None)


        # rename columns to match FlowSchema
        ret = (
            self.flows
            .merge(self.matmap
                   .filt(lambda x: x.material.notna())
                   ,right_on=('material')
                   ,left_on=('material_type')
                   ,suffixes=('','_matgrp')
                   ,how='left')
            .assign(material=lambda x: x.grouping4.combine_first(x.material))
            # .drop_columns(regex='_matgrp|^category$|grouping4|category_list')
            .rename(columns={'grouping4':'material_grouping'})

            # FIXME: downstream checks will fail if don't remove zero flows
            .filt(lambda x: x.wt_sent>ureg('0 tons')) # FIXME: CRITICAL should be 0 tons
        )

        # group to material
        flwgrp=schema_to_cols(AnnualizedLayeredGroupedFlowIdSchema)
        sumcols=list(set(schema_to_cols(AnnualizedLayeredGroupedFlowIdSchemaWithFlows))
                     -set(schema_to_cols(AnnualizedLayeredGroupedFlowIdSchema)))
        firstcols=(set(ret.columns)
                   -set(flwgrp)
                   -set(sumcols)
                  )
        ret = (
            ret.groupby(flwgrp)
            .agg(
                {
                # **{col: 'first' for col in firstcols}, 
                 **{col: 'sum' for col in sumcols}}
            )
            .reset_index()
            .merge(self.get_endpoints().reset_index().add_prefix('o_'),left_on='o_id',right_on='o_id')
            .merge(self.get_endpoints().reset_index().add_prefix('d_'),left_on='d_id',right_on='d_id')
            [schema_to_cols(LayerFlowSchema)]
        )
        return ret

