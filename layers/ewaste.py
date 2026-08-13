import os
from pathlib import Path
import re
from typing_extensions import OrderedDict
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import janitor
import pint
from layers.appliance import read_ewaste_landfill_endpoints
from services.geocode import GeocodeById, GeocodeFromLatLon, Geocoder, MergeGeocoder, backfill_regions, crs_ca, crs_ll, fips_ca, get_ca_cty, get_census_tract_population, get_county_endpoints, get_state_endpoints, get_us_state, get_world_boundary, get_world_endpoints, read_rdrs_manual_geocodes, read_unify_geo
from layers.base import ModelLayer
from core.common import case_when, cpath, filt, add_col_suffix, strip_col_names, wdisplay
from config import cli_field
from config.settings import settings
from config.base_config import BaseModelConfig
from config.path_template import PathTemplate
from core.units import ureg
from IPython.display import display

import shapely.geometry
import pygris as pyg


from core.model_types import *

import logging
from core.units import create_unit_assignment, create_unit_transformation
from services.routing import compute_nearest_neighbor_mapping
from utils.logging_config import log_errors
logger = logging.getLogger(__name__)


# Define crs_ll (assuming it's a coordinate reference system)
crs_ll = "EPSG:4326"

    
def read_ewaste_landfills_endpoints() -> DataFrame[GeocodedEndpointSchema]:
    """Load e-waste landfill endpoints from manually curated RDRS geocodes.

    Filters RDRS manual geocodes to only facilities of type
    ``'E-WASTE LANDFILL'``, adding a null ``geometry_agg`` column for
    schema conformance.

    Returns
    -------
    DataFrame[GeocodedEndpointSchema]
        E-waste landfill endpoints with columns conforming to
        ``GeocodedEndpointSchema``. ``geometry_agg`` is ``None`` for all
        rows.

    Notes
    -----
    - FIXME: ``facility_type`` values should be standardized (currently
      relies on exact match ``'E-WASTE LANDFILL'``).
    - Results are not cached; each call re-reads from
      :func:`read_rdrs_manual_geocodes`.

    See Also
    --------
    read_rdrs_manual_geocodes : Source data for RDRS manual geocodes.
    read_ewaste_landfill_endpoints : Appliance module's e-waste landfill
        endpoint loader (used in :func:`compute_residual_flows`).
    """
    df = (
        read_rdrs_manual_geocodes()
        .filt(lambda x: (
            (x.facility_type=='E-WASTE LANDFILL')
            |(x.facility_flags.str.upper().str.match(r'.*E[\s_]*WASTE[\s_]*LANDFILL'))
            )) # FIXME: standardize facility_type values
        .add_column_if_missing('geometry_agg',None)
    )
    return df[schema_to_cols(GeocodedEndpointSchema)]


def load_ewaste_endpoints(
    uwed_endpoints_file: Path,
    model_year: int = 2021
    ) -> DataFrame[LayerEndpointSchema]: #[GeocodedNamedEndpointSchema]:
    """Load and merge all e-waste endpoint entities from multiple sources.

    Combines four endpoint sources into a single deduplicated DataFrame:

    1. **UWED endpoints** — e-waste collector/recycler facilities from the
       processed UWED CSV, with facility types prefixed by ``'E-waste '``.
    2. **Census tract geometries** — California census tracts as spatial
       endpoints with representative point coordinates.
    3. **County endpoints** — California county centroids (used as
       aggregation targets for export flows).
    4. **E-waste landfills** — RDRS-manually-geocoded landfill endpoints.

    Duplicate IDs across sources are resolved by keeping the first
    occurrence (``groupby('id').first()``).

    Parameters
    ----------
    model_year : int, default 2021
        Census year for tract and boundary data.

    Returns
    -------
    DataFrame
        Merged e-waste endpoints indexed by ``id``, with columns ``n1``,
        ``n2``, ``lat``, ``lon``, ``aggregation``, ``facility_type``,
        and source-specific fields. Not strictly validated against a
        single schema (commented-out annotation suggests
        ``GeocodedNamedEndpointSchema``).

    Notes
    -----
    - UWED duplicates (by design from matching) are resolved via
      ``groupby('id').first()``.
    - Census tracts receive ``id`` prefixed with ``'TRACT'`` and
      ``aggregation = TRACT``.
    - FIXME: ``aggregation`` assignments on tracts and UWED endpoints
      may be out of place (noted in source comments).
    - County endpoints are included without ``geometry`` or
      ``geometry_agg`` columns (these are dropped before concatenation).

    See Also
    --------
    read_ewaste_landfills_rdrs : E-waste landfill source.
    get_county_endpoints : County endpoint source.
    load_ewaste_flow : Loads the corresponding flow data.
    """
    tract_geo = (
        pyg.tracts(state=fips_ca, year=model_year, cb=True, cache=True).clean_names()
        .rename(columns={'geometry':'geometry_agg'})
        .assign(
            id=lambda x: 'TRACT'+x.geoid,
            n1=lambda x: x.geoid,
            n2=lambda x: x.namelsad,
            # geometry conversions here to avoid
            lat=lambda x: x.set_geometry('geometry_agg').to_crs(crs_ca).geometry.representative_point().to_crs(crs_ll).y,
            lon=lambda x: x.set_geometry('geometry_agg').to_crs(crs_ca).geometry.representative_point().to_crs(crs_ll).x,
            aggregation=EndpointAggregation.TRACT.value # FIXME: seems out of place
        )
        .pipe(pd.DataFrame)
    )
    
    # get counties, which will be endpoints of exports if we aggregate to that level.
    counties = (
        get_county_endpoints().to_crs(crs_ll).pipe(pd.DataFrame).drop(columns=['geometry','geometry_agg'])
        .reset_index()
    )
    
    # get uwed endpoints
    # f=cpath("ewaste_data",f"ewaste-processed-data - endpoints.csv")
    uwed= (
        pd.read_csv(uwed_endpoints_file)
        .clean_names()
        # .rename(columns={'ewasteid':'id'})
        .assign(
            n1=lambda x: x.lu_name.fillna(x['id']),
            aggregation=EndpointAggregation.LOCATION.value, # FIXME: seems out of place
            facility_type=lambda x: case_when(x['type'].notna(), 'E-waste '+x.type,
                                              True, x.type)
        )
        .rename(columns={'addr_merge':'n2'})
        
        # there are duplicates (by design due to matching), we keep the first
        .groupby('id').first()
        .reset_index()
        # .assign(  # and encode a geometry for them
        #     geometry=lambda x: x.apply(lambda row: shapely.geometry.Point(row.lon, row.lat), axis=1),
        #     geometry_src='UWED'
        # )
        # .pipe(lambda df: gpd.GeoDataFrame(df,geometry='geometry'))
        # [['id','county','state','country','src','geometry','geometry_src','aggregation']]
    )
    
    landfills=read_ewaste_landfills_endpoints().reset_index()
    
    return (
        pd.concat([
            df.set_flags(allows_duplicate_labels=True)  # temporarily allow duplicate columns across sources
            for df in [landfills,uwed,tract_geo,counties]
            
            ])
        # drop any duplicate IDs across sources, keeping the first occurrence (e.g., from UWED)
        # FIXME: should be careful with this
        .groupby('id').first()
        .set_flags(allows_duplicate_labels=False)       # disallow duplicates after concatenation
    )
    

def load_ewaste_flow(
    ewaste_flow_file: Path,
    model_year: int
    ) -> DataFrame[EwasteFlowSchema]:
    """Load e-waste flow records from the processed UWED flows CSV.

    Reads the UWED flows file and filters to only shipping records
    (``aq_type`` matching ``'^Shipped'``), renaming ``ewasteid`` to
    ``id``.

    Parameters
    ----------
    model_year : int
        Model year (currently unused in loading, but passed for
        consistency with other layer loaders).

    Returns
    -------
    DataFrame[EwasteFlowSchema]
        Shipping-only e-waste flow records with ``id`` column.

    Notes
    -----
    - Only records where ``aq_type`` starts with ``'Shipped'`` are
      retained; other acquisition types (e.g., ``'Stored'``,
      ``'Recycled'``) are excluded.
    - The ``model_year`` parameter is accepted but not used for
      filtering; year filtering occurs later in
      :meth:`EwasteLayer.read_flows`.

    See Also
    --------
    load_ewaste_endpoints : Loads the corresponding endpoint data.
    EwasteLayer.read_flows : Full flow processing pipeline.
    """
    # f=cpath("ewaste_data",f"ewaste-processed-data - flows.csv")
    return (
        pd.read_csv(ewaste_flow_file)
        .clean_names()
        # only keep shipping records
        .filt(lambda x: x.aq_type.str.match(r'^Shipped'))
        .rename(columns={'ewasteid':'id'})
        # [['id','county','state','country','src','geometry','geometry_src','aggregation']]
    )

from config.cli_field import cli_field

@dataclass
class EwasteModelConfig(BaseModelConfig, prefix="ewaste"):

    # ── Internal fields ───────────────────────────────────────────────────────
    matmap_file: str = field(default="")
    name:        str = field(default="Ewaste")

    # ── CLI-exposed fields ────────────────────────────────────────────────────
    model_year: int = cli_field(
        2021,
        help    = "Model year for Ewaste calculations",
        metavar = "MODEL_YEAR"
    )

    ewaste_data_dir: PathTemplate = cli_field(
        '.',
        help    = "Directory holding Ewaste data files"
    )
    uwed_endpoints_file: PathTemplate = cli_field(
        None,
        help    = "File holding UWED endpoint information"
    )
    ewaste_flow_file: PathTemplate = cli_field(
        None,
        help    = "File holding e-waste flow data"
    )
    
    ewaste_wt_pf_u:     pint.Quantity = field(default=None)
    ewaste_res_wr_pf_u: pint.Quantity = field(default=None)



class EwasteLayer(ModelLayer):
    """E-waste material flow layer for the REMADE pipeline model.

    Models the generation, collection, export, and residual disposal of
    electronic waste in California. The layer processes three flow types:

    - **Transfer flows** — e-waste shipped from collection points to
      recyclers/processors (from UWED data).
    - **Export flows** — "shadow" e-waste (67% of total generation)
      distributed proportionally by census tract population and routed
      to export destinations (ports), optionally aggregated to the
      county level.
    - **Residual flows** — processing residuals from
      collector/recyclers trucked to the nearest Class 1 landfill.

    Parameters
    ----------
    matmap_file : str
        Path to the material mapping Excel file.
    model_year : int
        Year for census data and flow processing.
    geocoder : Geocoder
        Geocoder instance for assigning geometries to endpoints.
    name : str, default "Ewaste"
        Layer name.

    Attributes
    ----------
    model_year : int
        Census/model year.
    ewaste_flows : DataFrame or None
        Accumulated flow records (built incrementally by
        ``read_flows``).
    ewaste_flow_ent : DataFrame or None
        Endpoint entities used in flows.
    matmap : DataFrame[RDRSMaterialSchema]
        Material mapping table.
    ewaste_payload_factor_tons_to_trips : float
        Tons-to-trips conversion factor for e-waste transfers
        (1/10, assuming ~25 tons/trip).
    ewaste_residuals_payload_factor_tons_to_trips : float
        Tons-to-trips conversion factor for residual flows
        (1/20, assuming full loads).
    geocoder : Geocoder
        Geocoder instance.
    Ewaste_flows_removed : DataFrame
        Flows removed due to null geocodes or out-of-state origins
        (set during ``read_flows``).

    Notes
    -----
    - Initialization calls :meth:`read_flows`, which performs the full
      pipeline: loading, geocoding, export computation, residual
      computation, material mapping, and aggregation.
    - The custom pint unit ``'edevice'`` (8.3349 lb) is defined to
      convert device counts to weight.
    - FIXMEs in source: ``resid_rate`` is hardcoded, specific landfill
      IDs are hardcoded, ``MAT_class`` for residuals may be incorrect.

    See Also
    --------
    ModelLayer : Abstract base class.
    ApplianceLayer : Appliance recycling layer (similar structure).
    """
    
    def __init__(self, 
                 config: EwasteModelConfig, 
                 geocoder: Geocoder = None
                 ):
        """Initialize the Ewaste layer and load all flow data.

        Parameters
        ----------
        matmap_file : str
            Path to the material mapping Excel file.
        model_year : int
            Year for census data and flow processing.
        geocoder : Geocoder
            Geocoder instance for endpoint geocoding.
        name : str, default "Ewaste"
            Layer name (overrides base class default).

        Notes
        -----
        - Calls :meth:`read_material_mappings` and :meth:`read_flows`
          during initialization, so the layer is fully populated after
          construction.
        """
        super().__init__(config.name)
        self.name                = config.name
        self.model_year          = config.model_year
        self.ewaste_data_dir     = config.ewaste_data_dir
        self.uwed_endpoints_file = config.uwed_endpoints_file
        self.ewaste_flow_file    = config.ewaste_flow_file

        self.ewaste_wt_pf_u      = config.ewaste_wt_pf_u
        self.ewaste_res_wr_pf_u  = config.ewaste_res_wr_pf_u
        
        self.geocoder            = geocoder
        
        self.ewaste_flows        = None
        self.ewaste_flow_ent     = None
        self.read_material_mappings(config.matmap_file)
        
        # claude sonnet 4.6
        # see ai-chats/ewaste-payload-factor.md
        # FIXME: review
        self.ewaste_payload_factor_tons_to_trips = pint.Quantity(1/10.0, '1/ton')           # Assuming 25 tons per trip as a reasonable estimate
        self.ewaste_residuals_payload_factor_tons_to_trips = pint.Quantity(1/20.0, '1/ton') # Assuming full loads

        # Does the work to read flows and entities into standard forms
        self.read_flows()
        
    @pa.check_types
    def get_geocoder(self) -> Geocoder:
        if self.geocoder is None:
            geocoders = OrderedDict()
            
            geocoders['rdrs_man']  = GeocodeById(read_rdrs_manual_geocodes(),'Manual geocodes')
            # FIXME: temporary template instantiation
            ff=settings.get('model').unify_geo_file.format(**settings.get('model').to_dict())
            geocoders['unify_geo'] = GeocodeById(read_unify_geo(ff),'Unified geocodes')

            geocoders['counties']      = GeocodeById(get_county_endpoints(),'County endpoints')

            geocoders['latlon'] = GeocodeFromLatLon(lat_col='lat', lon_col='lon', name='LatLon')

            
            self.geocoder = MergeGeocoder(geocoders,require_complete=True)
        return self.geocoder

    @pa.check_types
    def read_material_mappings(self,matmap_file) -> DataFrame[RDRSMaterialSchema]:
        """Read material category mappings from an Excel file.

        Loads the material mapping spreadsheet and applies
        :func:`janitor.clean_names` for column name normalization.

        Parameters
        ----------
        matmap_file : str
            Path to the material mapping Excel file.

        Returns
        -------
        DataFrame[RDRSMaterialSchema]
            Material mapping table with cleaned column names.

        Notes
        -----
        - The result is stored as ``self.matmap`` for use in flow
          processing.
        - Docstring incorrectly references "Appliance layer" — this is
          the Ewaste layer.

        See Also
        --------
        get_materials : Returns the stored material mapping.
        """
        self.matmap = pd.read_excel(matmap_file).clean_names()
        return self.matmap

    def get_materials(self) -> DataFrame[RDRSMaterialSchema]:
        """Return the material mapping table for the Ewaste layer.

        Returns
        -------
        DataFrame[RDRSMaterialSchema]
            Material mapping table loaded by
            :meth:`read_material_mappings`.

        Notes
        -----
        - Docstring incorrectly references "Appliance layer" — this is
          the Ewaste layer.
        """
        # FIXME: Implement logic to return materials for ewaste layer
        return self.matmap

    @pa.check_types
    def get_endpoints(self) -> DataFrame[LayerEndpointSchema]:
        """Return geocoded endpoint entities for the Ewaste layer.

        Returns the endpoint entities built during :meth:`read_flows`,
        ensuring ``geometry_agg`` is present (set to ``None`` if missing)
        and selecting only columns in ``LayerEndpointSchema``.

        Returns
        -------
        DataFrame[LayerEndpointSchema]
            E-waste endpoint entities with full geocoding and region
            metadata.

        Raises
        ------
        AssertionError
            If ``self.ewaste_flows`` is ``None`` (i.e., ``read_flows``
            has not been called).

        See Also
        --------
        load_ewaste_endpoints : Source endpoint data.
        EwasteLayer.read_flows : Builds and geocodes the endpoints.
        """
        assert(self.ewaste_flows is not None)
        return (
            self.ewaste_flow_ent.add_column_if_missing('geometry_agg',None)
            [schema_to_cols(LayerEndpointSchema)]
        )
    
    @pa.check_types
    def compute_exports(self):
        """Compute export flows for "shadow" e-waste distributed by census tract population.

        Models the ~67% of California's e-waste that is exported abroad.
        Shadow e-waste is distributed across census tracts proportional
        to population, then routed to export destination endpoints
        (facilities with non-US state codes) based on their share of
        existing UWED export tonnage.

        When ``agg_to_county=True`` (default), tract-level origins are
        spatially joined to county boundaries and aggregated, producing
        county-to-export-destination flows instead of
        tract-to-export-destination flows.

        The computed flows are appended to ``self.ewaste_flows``.

        Returns
        -------
        DataFrame
            Export flow records. If ``agg_to_county=True``, county-level
            flows; otherwise, tract-level flows.

        Raises
        ------
        AssertionError
            If ``self.ewaste_flows`` is ``None``.

        Notes
        -----
        - Total export tonnage: ``0.67 × 1.05e6 metric tons × 1000``,
          converted from kg to short tons via pint.
        - Export destinations are identified as endpoints whose
          ``facility_state`` is not a valid US state USPS code.
        - All export flows use ``EMFAC_class='T7 POLA Class 8'`` and
          ``ttype='ewaste_export'``.
        - ``agg_to_county`` is hardcoded to ``True``; the tract-level
          branch exists but is not currently used.
        - County aggregation uses ``sjoin`` with ``predicate='covers'``
          in CA Albers projection.

        See Also
        --------
        get_census_tract_population : Population data for proportional
            distribution.
        compute_nearest_neighbor_mapping : Used for residual flow routing.
        """
        assert(self.ewaste_flows is not None)
        # The balance, which is ~67% of 1.05 million metric tons (or 0.70 MT) is
        # exported. For simplicity, we assume that this shadow e-waste
        # originates at the centroid of each of California’s 8,057 census tracts
        # and that it is proportional to the population of each census tract.
        # From there, it is trucked to the ports of Los Angeles and Long Beach
        # and shipped abroad.
        # FIXME:UNIT
        to_distribute = 0.67 * 1.05e6 * 1000 * ureg('kg').to('tons')  # convert to tons
        
        from pygris.data import get_census

        # Get 2021 Census tract geometries for a specific state (e.g., California, FIPS code '06')
        # Use cb=True for smaller, simplified cartographic boundary files
        ewep=load_ewaste_endpoints(
            uwed_endpoints_file=self.uwed_endpoints_file,
            model_year=self.model_year
            )
        tract_geo = (
            ewep.filt(lambda x: x.index.str.match('^TRACT'))
            .pipe(gpd.GeoDataFrame, geometry='geometry_agg', crs=crs_ll)
        )

        # Get 2021 ACS 5-year population data for the same state and join it
        pop_data = get_census_tract_population()

        # Join the population data to the geometry and compute the ewaste
        # generated from the tract by proportion of total population
        tract_data = tract_geo.join(pop_data.set_index('geoid'), on="n1")
        total_pop = tract_data['b01003_001e'].sum()
        # FIXME:UNIT
        tract_data['ewaste_gen'] = (
            ((tract_data['b01003_001e'] / total_pop) * to_distribute)
            .astype(f'pint[{str(to_distribute.units)}]')
        )


        # The resulting GeoDataFrame has the geometry and the population data
        # print(tract_data[['GEOID', 'B01003_001E','ewaste_gen']].head())
        # we want to compute the fractions of exported waste going to export
        # locations, which we define as any destination not matching a 
        # two-letter state code
        tmp_dests=(
            # self.ewaste_flows.join(self.ewaste_flow_ent.add_col_prefix('d_'),on='d_id')
            self.ewaste_flows.join(
                # UWED has 2 letter state codes if in US, so filter vs known codes and keep the rest as exports
                ewep.filt(lambda x: ~x.facility_state.isin(get_us_state().stusps) & x.facility_state.notna())[[]]
                , on='d_id'
                , how='inner')
            .groupby(['d_id']).sum(numeric_only=True)
            .assign(frac=lambda x: x.wt_sent.pint.magnitude/x.wt_sent.pint.magnitude.sum())
            .filt(lambda x: x.frac>0)
            .reset_index()
        )
        # apply the export proportions to the ewaste generated in each tract
        # to generate flows
        tract_flows=(
            tract_data
            .assign(
                o_id=lambda x: x.index,        
            )
            .join(tmp_dests[['d_id','frac']],how='cross')
            .assign(
                layer=self.name,
                EMFAC_class='T7 POLA Class 8', # default to POLA class 8 for all flows, we can refine this later if needed
                year=self.model_year,
                ttype='ewaste_export',
                material_stream='ShadowEwaste',
                MAT_class='Ewaste:Ewaste:Ewaste',
                o_wt_sent=lambda x: x.ewaste_gen, 
                wt_sent=lambda x: (x.frac * x.o_wt_sent),
                trips=lambda x: (x.wt_sent * self.ewaste_payload_factor_tons_to_trips).pint.magnitude, # FIXME: use config
            )
            # remove endpoints with zero flows
            .filt(lambda x: x.wt_sent > 0)
        )
        
        agg_to_county=True
        # aggregate to county
        if agg_to_county:
            cty_flows=(
                # get the county geometries
                get_county_endpoints()
                .set_flags(allows_duplicate_labels=True)
                .set_geometry('geometry_agg').to_crs(crs_ca)
                .sjoin(tract_flows.set_flags(allows_duplicate_labels=True)
                       .set_geometry('geometry_agg').to_crs(crs_ca)
                    ,predicate='covers',how='left')
                .filt(lambda x: x.o_id.notna()).rename(columns={'o_id':'o_id_tract'})
                .reset_index().rename(columns={'id':'o_id'})
                .groupby(['layer','year','EMFAC_class','ttype','material_stream','MAT_class','o_id','d_id'])
                .agg({'wt_sent':'sum','trips':'sum'})
                .reset_index()
                .set_flags(allows_duplicate_labels=False)
            )
            self.ewaste_flows=pd.concat([
                self.ewaste_flows,
                cty_flows[list(set(schema_to_cols(LayerFlowSchema)) - set(['material_grouping'])) + ['MAT_class']]
                ],ignore_index=True)
            return cty_flows
        else:            
            self.ewaste_flows=pd.concat([
                self.ewaste_flows,
                tract_flows[list(set(schema_to_cols(LayerFlowSchema)) - set(['material_grouping'])) + ['MAT_class']]
                ],ignore_index=True)
            return tract_flows
        
    @pa.check_types
    def compute_residual_flows(self):
        """Compute residual waste flows from recyclers to the nearest Class 1 landfill.

        Models the residual waste generated by e-waste
        collector/recyclers after processing. For each recycler, computes
        net inbound tonnage (inbound minus outbound), applies a residual
        rate (``resid_rate = 0.02218065``) to determine residual tonnage,
        and routes it to the nearest e-waste landfill within a 500-mile
        radius using :func:`compute_nearest_neighbor_mapping`.

        The computed flows are appended to ``self.ewaste_flows``.

        Returns
        -------
        DataFrame
            Residual flow records with columns ``o_id``, ``d_id``,
            ``wt_sent``, ``trips``, ``layer``, ``EMFAC_class``,
            ``year``, ``ttype``, ``material_stream``, and ``MAT_class``.

        Raises
        ------
        AssertionError
            If ``self.ewaste_flows`` is ``None``.
        AssertionError
            If not all e-waste landfill IDs from
            :func:`read_ewaste_landfill_endpoints` are found in
            ``self.ewaste_flow_ent``.

        Notes
        -----
        - FIXME: ``resid_rate`` is hardcoded to ``0.02218065``; should
          be calculated to hit the 1,756,800 ton target from the e-waste
          analysis.
        - FIXME: Specific landfill IDs are loaded from the Appliance
          module's :func:`read_ewaste_landfill_endpoints` rather than
          being parameterized.
        - FIXME: ``MAT_class`` for residuals is ``'Ewaste:Ewaste:Ewaste'``
          — may need a distinct classification for residuals.
        - Residual flows use ``EMFAC_class='T7 Tractor Class 8'`` and
          ``ttype='ewaste_residuals_to_landfill'``.
        - The ``ewaste_residuals_payload_factor_tons_to_trips`` (1/20)
          is used for trip estimation (assumes full loads).

        See Also
        --------
        compute_nearest_neighbor_mapping : Nearest-neighbor routing.
        read_ewaste_landfill_endpoints : Landfill endpoint IDs.
        compute_exports : Export flow computation (complementary).
        """
        assert(self.ewaste_flows is not None)

        recyc=(
            self.ewaste_flows
            .merge(
                self.ewaste_flow_ent.reset_index()[['id','n1','facility_type','geometry']].add_col_prefix('d_'),left_on='d_id',right_on='d_id')
            .filt(lambda x: x.d_facility_type=='E-waste Collector/Recycler')
            [['d_id',
              'd_n1',
              'd_geometry',
              'd_facility_type']].drop_duplicates()
        )
        resid_rate=0.02218065 #FIXME: calculate this to hit the 1,756,800 target from the e-waste analysis
        resid_outflows=(
            recyc
            .merge(self.ewaste_flows[['d_id','wt_sent']].add_col_suffix('_in'),left_on='d_id',right_on='d_id_in',how='left')
            .groupby(['d_id']).agg({'d_n1':'first',
                                    'd_geometry':'first',
                                    'd_facility_type':'first',
                                    'wt_sent_in':'sum'}).reset_index()
            .merge(self.ewaste_flows[['o_id','wt_sent']].add_col_suffix('_out'),left_on='d_id',right_on='o_id_out',how='left')
            .groupby(['d_id']).agg({'d_n1':'first',
                                    'd_geometry':'first',
                                    'd_facility_type':'first',
                                    'wt_sent_in':'first',
                                    'wt_sent_out':'sum'}).reset_index()
            .fillna(0)
            .assign(
                net_wt_in=lambda x: x.wt_sent_in - x.wt_sent_out,
                resid_wt_out=lambda x: x.net_wt_in*resid_rate
                )
            .filt(lambda x: x.net_wt_in>0)
            .rename(columns={
                'd_id':'o_id',
                'd_n1':'o_n1',
                'd_geometry':'o_geometry',
                'd_facility_type':'o_facility_type'
            })
        )
        
        # FIXME: hardcoding specific IDs, should
        ewaste_lf_ids=list(read_ewaste_landfill_endpoints().index)
        lf=(
            self.ewaste_flow_ent.reset_index().filt(lambda x: x.id.isin(ewaste_lf_ids))
            .add_col_prefix('d_')
        )
        # make sure we found them all
        assert(lf.d_id.to_list() == ewaste_lf_ids)
        
        # now we want to identify the nearest neighbor for each of these and assign the resid_tons_out to that
        mm=(
            compute_nearest_neighbor_mapping(
                'ewaste_to_landfill_ctmsr',
                resid_outflows.set_geometry('o_geometry').set_crs(crs_ll),'o_geometry',
                lf.set_geometry('d_geometry').set_crs(crs_ll)[['d_id','d_geometry']],'d_geometry',
                'o_id',
                500 # mi radius
                )
            .assign(
                layer=self.name,
                EMFAC_class='T7 Tractor Class 8', # default to POLA class 8 for all flows, we can refine this later if needed
                year=self.model_year,
                ttype='ewaste_residuals_to_landfill',
                material_stream='UWED',
                MAT_class='Ewaste:Ewaste:Ewaste',  #FIXME: residuals?
                wt_sent=lambda x:  x.resid_wt_out.astype('pint[short_ton]'),
                trips=lambda x: (x.wt_sent * self.ewaste_residuals_payload_factor_tons_to_trips).pint.magnitude # FIXME: use config
            )
        )
        self.ewaste_flows=pd.concat([
            self.ewaste_flows,
            mm[list(set(schema_to_cols(LayerFlowSchema)) - set(['material_grouping'])) + ['MAT_class']]
            ],ignore_index=True)
        return mm
        
    @pa.check_types
    def read_flows(self):
        """Load, geocode, validate, and aggregate all e-waste flow data.

        Executes the full e-waste flow processing pipeline:

        1. **Load raw flows** — reads UWED shipping records and converts
           device counts and weights to short tons using pint unit
           arithmetic (custom ``'edevice'`` unit = 8.3349 lb).
        2. **Compute exports** — distributes shadow e-waste across census
           tracts and routes to export destinations.
        3. **Collect OD pairs** — identifies unique endpoints and remaps
           flow records onto them via :func:`collect_ewaste_ods`.
        4. **Build endpoint entities** — joins all referenced endpoint
           IDs (origins, destinations, landfills) against
           :func:`load_ewaste_endpoints`.
        5. **Geocode endpoints** — applies the configured ``Geocoder``
           to assign geometries.
        6. **Validate geocodes** — drops endpoints with null or invalid
           geometries and their associated flows, logging tonnage impact.
        7. **Validate origins** — drops flows with out-of-state (non-CA)
           origins, logging tonnage impact.
        8. **Backfill regions** — assigns county/state/country to all
           endpoints via :func:`backfill_regions`.
        9. **Compute residual flows** — routes recycler residuals to
           nearest landfills.
        10. **Map materials** — joins ``MAT_class`` against the material
            mapping table and splits into component columns.
        11. **Aggregate** — groups by flow key columns and sums
            ``wt_sent`` and ``trips``.
        12. **Drop zero flows** — removes any flows with zero or NaN
            tonnage.

        Raises
        ------
        AssertionError
            If any aggregation key column (``EMFAC_class``, ``year``,
            ``layer``, ``ttype``, ``material_stream``,
            ``material_grouping``, ``o_id``, ``d_id``) contains NaN
            values after material mapping.

        Notes
        -----
        - The ``strict`` flag (hardcoded to ``False``) controls whether
          null-geometry and out-of-state errors are fatal or result in
          row drops with warnings.
        - Removed flows are stored in ``self.Ewaste_flows_removed`` for
          audit purposes.
        - The custom pint unit ``'edevice'`` is defined as
          ``8.334860486 * pound`` based on mean device weights from the
          UWED analysis.
        - ``collect_ewaste_ods`` is called but not defined in this
          module — it is expected to be available in the calling scope.

        See Also
        --------
        load_ewaste_flow : Raw flow data loader.
        load_ewaste_endpoints : Endpoint data loader.
        compute_exports : Export flow computation.
        compute_residual_flows : Residual flow computation.
        backfill_regions : Region assignment for endpoints.
        """
        logger.info("Reading Ewaste Flow Data")
        # Define the custom unit 'edevice' with the conversion factor inferred
        # from mean device weights and estimated device compositions in 2018
        # see "Final UWED Flows.ipynb" for the calculation
        ureg.define('edevice = 8.334860486 * pound = edev')

        units = {
            'aq_ed_count': 'edevice',
            'aq_ed_lbs': 'lb'
        }

        self.ewaste_flows=(
            load_ewaste_flow(ewaste_flow_file=self.ewaste_flow_file,
                             model_year=self.model_year)
            .assign(**create_unit_assignment(units))
            .rename(columns={'aq_ed_lbs':'ed_wt'})
         
            # Required for FlowSchema   
            .add_column_if_missing('layer',self.name)
            .assign(
                ed_count_to_wt=lambda x: x.aq_ed_count.pint.to('tonnes'),  # convert counts to tons using a factor from UWED
                ed_wt=lambda x: x.ed_wt.pint.to('tonnes'),                 # just to be consistent
                wt_sent  = lambda x: (x.ed_count_to_wt + x.ed_wt).pint.to('short_ton'),
                EMFAC_class='T7 Tractor Class 8',
                trips=lambda x: (x.wt_sent * self.ewaste_payload_factor_tons_to_trips).pint.magnitude, # FIXME: use config
                year=self.model_year,
                ttype='ewaste_transfer',
                material_stream='UWED',
                MAT_class='Ewaste:Ewaste:Ewaste'
            )
            [list(set(schema_to_cols(LayerFlowSchema)) - set(['material_grouping'])) + ['MAT_class']]
        )

        # we're adding export compositions after endpoints have regions backfilled
        # so we know what country they're exporting to
        self.compute_exports()
        

        logger.info("Reading Ewaste Flow Data...collecting")
        # takes flows, identifies all endpoints, remaps flow endpoints onto the identified endpoints
        # (self.Ewaste_flow_ent,self.ewaste_flows)=collect_ewaste_ods(self.ewaste_flows)
        
        ewaste_lf_ids=list(read_ewaste_landfill_endpoints().index)
        all_used = pd.concat([
            self.ewaste_flows[['o_id']].rename(columns={'o_id':'id'}),
            self.ewaste_flows[['d_id']].rename(columns={'d_id':'id'}),
            read_ewaste_landfill_endpoints().reset_index()[['id']]
            ],
            ignore_index=True).drop_duplicates()
            
        self.ewaste_flow_ent = (
            load_ewaste_endpoints(
                uwed_endpoints_file=self.uwed_endpoints_file,
                model_year=self.model_year
                )
            .join(all_used.set_index('id'),how='inner')
            .add_column_if_missing('county',pd.NA)
            .add_column_if_missing('state',pd.NA)
            .add_column_if_missing('country',pd.NA)
            .add_column_if_missing('aggregation',EndpointAggregation.NONE.value)
        )

        # geocode all endpoints using the passed geocoder
        logger.info("Reading Ewaste Flow Data...geocoding")
        self.ewaste_flow_ent = self.get_geocoder().geocode(
            self.ewaste_flow_ent.drop(columns=['geometry','geometry_agg'])
            )
        
        # zero_flows = self.ewaste_flows.wt_sent == 0
        # if zero_flows.any():
        #     logger.info(f"Ewaste flows: removing {zero_flows.sum()} flow records with zero wt_sent")
        #     self.ewaste_flows = self.ewaste_flows.filt(lambda x: ~zero_flows)
        
        null_geoms = self.ewaste_flow_ent.geometry.isna() | ~self.ewaste_flow_ent.geometry.is_valid
        strict = False
        if null_geoms.any():
            if not strict:
                logger.warning(f'{sum(null_geoms)} entities without geocodes...dropping')
                display(self.ewaste_flow_ent[null_geoms])
                all_flows_tot=self.ewaste_flows.wt_sent.sum()
                self.ewaste_flow_ent = self.ewaste_flow_ent[~null_geoms]
                null_ents = null_geoms[null_geoms].index.tolist()
                self.Ewaste_flows_removed= self.ewaste_flows.filt(lambda x: x.o_id.isin(null_ents) | x.d_id.isin(null_ents))
                rem_flows_tot=self.Ewaste_flows_removed.wt_sent.sum()
                logger.warning(f'{len(self.Ewaste_flows_removed)} affected flows removed totaling {rem_flows_tot}/{all_flows_tot} = {rem_flows_tot/all_flows_tot*100:.1f}%')
                self.ewaste_flows = self.ewaste_flows.filt(
                    lambda x: ~x.o_id.isin(null_ents) 
                            & ~x.d_id.isin(null_ents)
                )
            else:
                logger.error(f'{len(null_geoms)} entities without geocodes...stopping')
                wdisplay(null_geoms)
                raise(f'{len(null_geoms)} entities without geocodes')

        oos_origin = (
            ~self.ewaste_flows
            .join(self.ewaste_flow_ent[['geometry']].add_col_prefix('o_'),on='o_id',how='left')
            .set_geometry('o_geometry').to_crs(crs_ll)
            .intersects(pyg.states(year=self.model_year,cache=True).set_index('STUSPS').to_crs(crs_ll).loc['CA'].geometry)
        )
        if oos_origin.any():
            if not strict:
                logger.warning(f'{oos_origin.sum()} flows with out-of-state origins...dropping')
                display(self.ewaste_flows[oos_origin])
                all_flows_tot=self.ewaste_flows.wt_sent.sum()
                self.ewaste_flows_removed=self.ewaste_flows[oos_origin]
                self.ewaste_flows = self.ewaste_flows[~oos_origin]
                rem_flows_tot=self.ewaste_flows_removed.wt_sent.sum()
                logger.warning(f'{len(self.ewaste_flows_removed)} affected flows removed totaling {rem_flows_tot}/{all_flows_tot} = {rem_flows_tot/all_flows_tot*100:.1f}%')
            else:
                logger.error(f'{oos_origin.sum()} flows with out-of-state origins...stopping')
                wdisplay(self.ewaste_flows[oos_origin])
                raise(f'{oos_origin.sum()} flows with out-of-state origins')


        # map every point to a county, state, and country
        logger.info("Reading Ewaste Flow Data...backfilling regions")
        self.ewaste_flow_ent = backfill_regions(self.ewaste_flow_ent)
        
        
        # now that everything's geocode, compute residual flows to landfills
        self.compute_residual_flows()
        
        
        logger.info("Mapping Ewaste flow material and vehicle classes")
        
        self.ewaste_flows = (
            self.ewaste_flows
            .join(
                pd.concat([
                    self.matmap
                    # add HAZ to material mappings
                    ,pd.DataFrame(
                        {'material_category':['Ewaste'],'material_subcategory':['Ewaste'],'material_type':['Ewaste'],'grouping4':['Ewaste']
                        ,'include':[True]})
                    ]
                    ,ignore_index=True)
                .assign(idx=lambda x: x.material_category+':'+x.material_subcategory+':'+x.material_type).set_index('idx')[['grouping4']]
                ,on='MAT_class')
            # .assign(
            #     car=lambda x: x.car.fillna(0).astype(int)
            # )
            # split the MAT_class into its components
            .pipe(
                lambda df: df.assign(
                    **df.MAT_class.str.split(':',expand=True).rename(
                        columns={0:'material_category',1:'material_subcategory',2:'material_type'}
                    )))
            
            .rename(columns={'grouping4':'material_grouping'})
        )

        # trims the flow data to the model year and applies material mapping
        logger.info("Reading Ewaste Flow Data...material mappings")
        # self.ewaste_flows = get_ewaste_flow_for_year(self.ewaste_flows,self.model_year,self.matmap,self.name)
        # rename columns to match FlowSchema
        
        # agg
        logger.info("Reading Ewaste Flow Data...aggregating")
        aggcols=['EMFAC_class','year','layer','ttype','material_stream','material_grouping','o_id','d_id']
        assert(not self.ewaste_flows[aggcols].isna().any().any()) # make sure agg cols are all available
        self.ewaste_flows = (
            self.ewaste_flows
            .groupby(aggcols,as_index=False)
            .agg({'wt_sent':'sum','trips':'sum'})
        )

        assert self.ewaste_flows.wt_sent.notna().all(), "NA wt_sent in read Ewaste flows"
        zero_flows=self.ewaste_flows.wt_sent == 0
        if zero_flows.any():
            logger.warning(f"{zero_flows.sum()} flows with zero wt_sent...dropping")
            self.ewaste_flows = self.ewaste_flows[~zero_flows]


    @pa.check_types
    def get_flows(self) -> DataFrame[LayerFlowSchema]: #DataFrame[AnnualizedLayeredGroupedFlowIdSchemaWithFlows]:
        """Get the flows from Ewaste layer."""
        if self.ewaste_flows is None:
            self.read_flows()
        assert(self.ewaste_flows is not None)

        return (
            self.ewaste_flows
            # FIXME: downstream checks will potentially fail if we keep nonzero flows
            # .filt(lambda x: ~zero_flows & ~oos_origin)
            .merge(self.get_endpoints().reset_index().add_prefix('o_'),left_on='o_id',right_on='o_id')
            .merge(self.get_endpoints().reset_index().add_prefix('d_'),left_on='d_id',right_on='d_id')
            [schema_to_cols(LayerFlowSchema)]
        )

    # def get_materials(self) -> DataFrame[EwasteMaterialSchema]:
    #     """Get the materials handled by Ewaste layer."""
    #     return self.matmap
    
