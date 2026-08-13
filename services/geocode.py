from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Callable, List, Union
import pandas as pd
import geopandas as gpd
from osrm import simple_route
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge
from core.common import cpath, case_when, shapely_point_to_osrm_point
import numpy as np
from config.settings import settings
from core.geocode_lookups import *

import pandera.pandas as pa
from pandera.typing import DataFrame
from pandera.typing.geopandas import GeoDataFrame

from core.model_types import (
    _RegionMappedEndpointSchema, _AggregatedSchemaWithNulls, 
    _CountryEndpointSchema, _CountySchema, DataFrame, EndpointAggregation, EndpointSchema, LayerEndpointSchema, GeocodedEndpointSchema, 
    GeocodedEndpointSchemaWithNamedNulls, GeocodedEndpointSchemaWithNulls, GeocodedNamedEndpointSchemaWithNulls, LayerEndpointSchema, 
    NamedEndpointSchemaWithNulls, _StateEndpointSchema, gpd, logger, np, schema_to_cols, 
    strict_check_types, strict_check_types_with_report)
import pygris as pyg
from cachier import cachier

import logging
from utils.logging_config import log_errors
logger = logging.getLogger(__name__)


# Coordinate Reference Systems
crs_ca = 3310  # this is a California Albers projection that is decent for showing the whole state
crs_usa = 3857
crs_ll = 4326
crs_ggl = 900913  # used by TIGER data; this is the google projections (900913)
crs_world = crs_ll

fips_ca=state_to_fips['CA']


null_gdf = lambda geomname: gpd.GeoDataFrame({'rdrs_id':[],geomname:[]},geometry=geomname).set_crs(crs_ll)



def clean_street_address(addr):
    """Clean street address strings for improved Nominatim geocoding accuracy.

    Removes common address components that Nominatim struggles to interpret,
    including unit/suite designators, trailing ZIP codes, and street suffix
    abbreviations.

    Parameters
    ----------
    addr : str or NaN
        Raw street address string.

    Returns
    -------
    str or NaN
        Cleaned address with unit numbers, trailing ZIPs, and street suffixes
        removed. Returns ``NaN`` unchanged.

    Notes
    -----
    The following patterns are stripped:

    - Unit/suite designators: ``UNIT``, ``SUITE``, ``STE`` + number
    - Trailing comma-separated ZIP codes
    - Street suffixes: ``AVE``, ``AVENUE``, ``BLVD``, ``ROAD``, ``RD``,
      ``DR``, ``DRIVE``, ``PL``, ``PLACE``, ``WAY``, ``WY``, ``LANE``,
      ``LN``, ``STREET``, ``ST``
    - FIXME: this is mostly legacy

    Examples
    --------
    >>> clean_street_address("123 Main St, Suite 400, 90210")
    '123 Main'
    >>> clean_street_address(None)  # returns NaN
    """
    if pd.isna(addr):
        return(addr)
    else:
        ret=addr
        ret=re.sub(r'(,\s*)?\b(UNIT|SUITE|STE\.?)\s+[\dA-Z]+','',ret,flags=re.IGNORECASE)
        ret=re.sub(r',\s*\d+\s*$','',ret,flags=re.IGNORECASE)
        ret=re.sub(r'\b(AVE(NUE)?|BLVD|ROAD|RD|(DR(IVE)?)|PL(ACE)?|WA?Y|LANE|LN|STREET|ST)\b\.?','',ret,flags=re.IGNORECASE)
        # print(f'ADDR {addr}=>{ret}')
        return(ret)



def geocode_here(luv_a):
    """Geocode entities using the HERE V7 geocoding service.

    Queries the HERE V7 API for each entity's ``geocode_name`` field,
    with rate limiting (50 ms minimum delay between requests). Results
    are cached to ``model_cache_dir/luhere.pickle`` for incremental
    processing on subsequent runs.

    Parameters
    ----------
    luv_a : DataFrame
        Entities with ``rdrs_id`` and ``geocode_name`` columns.

    Returns
    -------
    GeoDataFrame
        Columns ``rdrs_id``, ``here`` (raw geopy response), and
        ``geometry_here`` (Shapely Point). Empty if HERE geocoding is
        disabled (current default).

    Notes
    -----
    - **Currently disabled** — the ``if False`` branch is always taken,
      so this function returns an empty GeoDataFrame with a log message.
    - When enabled, previously cached geocodes are loaded first; only
      uncached entities are sent to the API.
    - Rate limiting uses :class:`geopy.extra.rate_limiter.RateLimiter`
      with ``min_delay_seconds=0.05``.
    - FIXME: this is mostly legacy

    See Also
    --------
    geocode_nominatim : Nominatim-based geocoder (alternative).
    collect_external_geocodes : Merges HERE and Nominatim results.
    """
    logger.info(f"...HereV7 geocoding ({len(luv_a)} entities)")

    from geopy.geocoders import HereV7
    apphere=HereV7(apikey=settings.here_connection.api_key)
    from geopy.extra.rate_limiter import RateLimiter
    geocode = RateLimiter(apphere.geocode, min_delay_seconds=0.05)

    luhere_s=gpd.GeoDataFrame({'geometry_here':gpd.GeoSeries()},columns=['rdrs_id','here','geometry_here']) # default is empty

    if False: # Not using HERE geocoder 
        try:
            luhere_cache=pd.read_pickle(cpath('model_cache_dir','luhere.pickle'))
            print(f"{len(luhere_s)} geocodes read from cache")
            to_read=luv_a.filt(lambda x: ~x.rdrs_id.isin(luhere_cache.rdrs_id.unique()))
        except FileNotFoundError:
            print("No cached geocodes")
            luhere_cache=pd.DataFrame() # empty DF: for pd.concat logit below, will be ignored
            to_read=luv_a
        
        if len(to_read)>0:
            print(f"{len(to_read)} geocodes to read...")
            # luhere_read=pd.DataFrame()
            luhere_read=(
                to_read
                .in_chain_immut(lambda df: print(f'Geocoding {len(df)} entities'))
                .assign(here=lambda x: x.geocode_name.apply(geocode))
                .in_chain_immut(lambda df: print(fr'Response on {len(df.filt(lambda x: ~x.here.isna()))} entities'))
            )
        else:
            luhere_read=pd.DataFrame()
        
        luhere_s=pd.concat([
            luhere_cache,
            luhere_read
        ])
        print("Done.")
        luhere_s.to_pickle('luhere.pickle')
    else:
        logger.info("...HereV7 geocoding (skipped)")
    return luhere_s



from geopy.geocoders import Nominatim
import shapely

def geocode_nominatim(luv_a):
    """Geocode entities using a local Nominatim instance.

    Queries a local Nominatim server (``localhost:8080``) for each entity's
    ``geocode_name`` field. Results are cached to
    ``model_cache_dir/lunom.pickle`` for incremental processing.

    Parameters
    ----------
    luv_a : DataFrame
        Entities with ``rdrs_id`` and ``geocode_name`` columns.

    Returns
    -------
    GeoDataFrame
        Columns ``rdrs_id``, ``nom`` (raw geopy response), and
        ``geometry_nom`` (Shapely Point). Empty if Nominatim geocoding
        is disabled or no rows remain to geocode.

    Notes
    -----
    - **Currently disabled** — the ``if False`` condition prevents actual
      API calls. Only cached results are returned.
    - Uses a custom ``user_agent`` (``'uci+remade'``) and connects via
      HTTP to a local Nominatim instance.
    - No rate limiting is applied (commented out), since the local server
      is assumed to handle the load.
    - FIXME: this is mostly legacy

    See Also
    --------
    geocode_here : HERE V7-based geocoder (alternative).
    collect_external_geocodes : Merges HERE and Nominatim results.
    """
    app=Nominatim(user_agent='uci+remade',domain="localhost:8080",scheme="http")
    from geopy.extra.rate_limiter import RateLimiter
    geocodenom = app.geocode #RateLimiter(app.geocode, min_delay_seconds=0.01)

    already_coded=[]
    logger.info(f"...Nominatim geocoding ({len(luv_a)} entities)")
    try:
        logger.info("...Nominatim geocoding: reading cache ")
        lunom_cache=pd.read_pickle(cpath('model_cache_dir','lunom.pickle'))
        logger.info(f"......{len(lunom_cache)} geocodes read from cache")
        to_read=luv_a.filt(lambda x: ~x.rdrs_id.isin(lunom_cache.rdrs_id.unique()))
    except FileNotFoundError:
        logger.info("...Nominatim geocoding: No cached geocodes")
        lunom_cache=pd.DataFrame() # empty DF: for pd.concat logit below, will be ignored
        to_read=luv_a

    if len(to_read)>0 and False:
        logger.info(f"...Nominatim geocoding: {len(to_read)} Rows left to geocode with Nominatim...")
        # luhere_read=pd.DataFrame()
        lunom_read=(
            to_read
            .in_chain_immut(lambda df: logger.info(f'...Nominatim geocoding: Geocoding {len(df)} entities'))
            .assign(nom=lambda x: x.geocode_name.apply(geocodenom))
            .in_chain_immut(lambda df: logger.info(f'...Nominatim geocoding: Response on {len(df.filt(lambda x: ~x.nom.isna()))} entities'))
        )
    else:
        logger.info(f"...Nominatim geocoding: No rows left to geocode with Nominatim")
        lunom_read=pd.DataFrame()

    lunom_s=pd.concat([
        lunom_cache,
        lunom_read
    ])
    logger.info("...Nominatim geocoding: Done and caching...")
    lunom_s.to_pickle(cpath('model_cache_dir','lunom.pickle'))
    logger.info("...Nominatim geocoding: Cached")
    return(lunom_s)


def collect_external_geocodes(luhere_a,lunom_a):
    """Merge and prioritize geocodes from HERE and Nominatim services.

    Joins HERE and Nominatim geocoding results on ``rdrs_id``, extracts
    Shapely Point geometries from raw geopy responses, and selects the
    best available geometry using a priority rule: HERE results take
    precedence over Nominatim.

    Parameters
    ----------
    luhere_a : DataFrame
        HERE geocoding results with ``rdrs_id`` and ``here`` (raw geopy
        response) columns.
    lunom_a : DataFrame
        Nominatim geocoding results with ``rdrs_id`` and ``nom`` (raw
        geopy response) columns.

    Returns
    -------
    GeoDataFrame
        Merged geocodes with columns:

        - ``geometry_here`` — Point from HERE (NaN if failed)
        - ``geometry_nom`` — Point from Nominatim (NaN if failed)
        - ``geometry_geo`` — Best available Point (HERE > Nominatim)
        - ``here_status`` — ``'SUCCESS'`` or ``'FAILED'``
        - ``nom_status`` — ``'SUCCESS'`` or ``'FAILED'``
        - ``geo_status`` — Status of the selected geometry source

    Notes
    -----
    - Geometry extraction uses ``row.here.point.longitude/latitude`` and
      ``row.nom.point.longitude/latitude`` from geopy's ``Location``
      objects.
    - The active geometry column is ``geometry_geo``, set in CRS
      ``EPSG:4326``.
    - A summary of ``geo_status`` value counts is printed to stdout.
    - FIXME: this is mostly legacy

    See Also
    --------
    geocode_here : Produces ``luhere_a``.
    geocode_nominatim : Produces ``lunom_a``.
    """
    logger.info("Collecting all geocodes from external services")

    # merge geocodes into a single dataframe
    lugeo=luhere_a.join(lunom_a.set_index('rdrs_id')[['nom']],on='rdrs_id',how='outer')
    lux_s=(
        gpd.GeoDataFrame(
            lugeo
            .assign(
                geometry_here=lambda x: gpd.GeoSeries(x.apply(
                    lambda row: (
                        shapely.geometry.Point(row.here.point.longitude,row.here.point.latitude)
                        if not pd.isna(row.here) else None
                    ),axis=1
                )).set_crs(crs_ll),
                here_status=lambda x: np.where(x.here.isna(),'FAILED','SUCCESS'),
                geometry_nom=lambda x: gpd.GeoSeries(x.apply(
                    lambda row: (
                        shapely.geometry.Point(row.nom.point.longitude,row.nom.point.latitude)
                        if not pd.isna(row.nom) else None
                    ),axis=1
                )).set_crs(crs_ll),
                nom_status=lambda x: np.where(x.nom.isna(),'FAILED','SUCCESS'),

                geometry_geo=lambda x: case_when(
                    x.here_status=='SUCCESS',x.geometry_here,
                    x.nom_status=='SUCCESS',x.geometry_nom,
                    True, x.geometry_nom #will be NA
                ),
                geo_status=lambda x: case_when(
                    x.here_status=='SUCCESS',x.here_status,
                    x.nom_status=='SUCCESS',x.nom_status,
                    True, x.nom_status #will be NA
                )
            )
        ).set_geometry('geometry_geo').set_crs(crs_ll)
    )
    print(lux_s.geo_status.value_counts())
    return lux_s


@pa.check_types
def backfill_regions(
        ent_all: DataFrame[GeocodedNamedEndpointSchemaWithNulls]
        ) -> DataFrame[LayerEndpointSchema]:
    """Backfill county, state, and country fields for all endpoint entities.

    Uses a cascading series of spatial joins to assign administrative region
    information:

    1. **California counties** — spatial join (``covered_by``) against Census
       county boundaries in CA Albers projection.
    2. **Census tracts** — fallback for CA census tract entities that don't map
       to a county via the county boundary join.
    3. **US states** — spatial join (``intersects``) against Census state
       boundaries for entities outside California.
    4. **Countries** — spatial join (``covered_by``) against world
       administrative boundaries for entities outside the US.
    5. **Nearest-country fallback** — ``sjoin_nearest`` within 100 km for
       entities not matched by the covered_by join.

    All existing county/state/country values are reset and recomputed to correct
    any pre-existing errors.

    Parameters
    ----------
    ent_all : DataFrame[GeocodedNamedEndpointSchemaWithNulls]
        Geocoded endpoint entities with ``geometry`` column. May have ``NaN``
        values in county/state/country columns.

    Returns
    -------
    DataFrame[LayerEndpointSchema]
        Entities with all county, state, and country fields populated.
        Non-California counties are labeled ``'Out of CA'``; non-US states are
        labeled ``'Out of US'``.

    Raises
    ------
    AssertionError
        If any TRACT-prefixed entity still has a null county after the census
        tract fallback step.
    AssertionError
        If any entity has a null county, state, or country after all backfill
        steps.
    AssertionError
        If the state spatial join produces multiple matches for a single entity
        (ambiguity that must be resolved manually).
    AssertionError
        If the nearest-country fallback produces empty results or unmatched
        entities.

    Notes
    -----
    - The function resets county/state/country to ``None`` before recomputing,
      so all values are authoritative.
    - The state join uses ``intersects`` rather than ``covered_by`` because some
      entities fall partially outside state boundaries.
    - Temporary join columns (``_cty``, ``_st``, ``_ct``, ``_ct2`` suffixes) are
      dropped before returning.
    - The motivation for this function derives from RDRS use of these
      aggregations for endpoints and also a need for county-level aggregations
      for anonymity

    See Also
    --------
    get_ca_cty : California county boundaries. get_us_state : US state
    boundaries. get_world_boundary : World country boundaries.
    """
    ca_cty=get_ca_cty()

    # select the entities that are missing county information (these will be the ones we need to backfill)
    # null_cty=ent_all.filt(lambda x: x.county.isna()).reset_index()
    # we're just going to backfill all of them, since it's not that expensive and this way we can fix any existing errors in the county/state/country fields too
    null_cty=ent_all.reset_index()
    cty_map=(
        null_cty[['id','geometry']].to_crs(crs_ca)
        .sjoin(ca_cty.set_geometry('geometry_agg').to_crs(crs_ca)
               ,how='left',predicate='covered_by')
    )
    ent_adj=(
        ent_all
        .assign(county=None,state=None,country=None) # reset these
        .join(cty_map.set_index('id').add_col_suffix('_cty'),on='id',how='left')
        .assign(
            county=lambda x: x.county.mask(x.county.isna(),x.county_cty)
            ,state=lambda x: x.state.mask(~x.county.isna(),'California')
            ,country=lambda x: x.country.mask(~x.county.isna(),'United States')
        )
    )
    
    # FIXME: hack to deal with census tract mapping
    # some CA census tracts don't map to California using the CA boundary but we 
    # know that they're in CA
    cty_tract_map=(
        pyg.tracts(year=2021,state=fips_ca,cache=True).clean_names()[['geoid','tractce','countyfp']]
        .merge(pyg.counties(year=2021,state=fips_ca,cache=True).clean_names()[['countyfp','name']].rename(columns={'name':'county'})
               ,left_on='countyfp',right_on='countyfp')
        .assign(id=lambda x: 'TRACT'+x.geoid)
    )
    ent_adj=(
        ent_adj
        .join(cty_tract_map.set_index('id').add_col_suffix('_tract_cty'),on='id',how='left')
        .assign(
            county=lambda x: x.county.mask(x.county.isna(),x.county_tract_cty)
            ,state=lambda x: x.state.mask(~x.county.isna(),'California')
            ,country=lambda x: x.country.mask(~x.county.isna(),'United States')
        )
    )
    # make sure we don't have any tracts with null county after this step
    assert ent_adj.filt(lambda x: x.index.str.match('TRACT') & x.county.isna()).index.size==0, (
        "Some TRACT entities still have null county after the census tract backfill step. "
    )
    
    us_state=get_us_state()
    # null states will be the rows that still have null country (after county and state geocodes)
    null_st=ent_adj.filt(lambda x: x.country.isna()).reset_index()
    st_map=(
        null_st[['id','geometry']].to_crs(crs_usa)
        .sjoin(us_state.set_geometry('geometry_agg').to_crs(crs_usa)
               ,how='left'
            #    ,predicate='covered_by' # this can omit some entities are partially covered by the state boundary
               ,predicate='intersects'
               )
    )
    # make sure we don't have multiple matches for the same entity (if this
    # fails, we need to do some more work to disambiguate the matches, e.g. by
    # looking at the county or doing a nearest neighbor join instead of
    # covered_by)
    assert (len(st_map) == 0)  or st_map.groupby('id').size().max()<=1, (
        "State spatial join produced multiple matches for some entities. "
    )
    ent_adj=(
            ent_adj
            .join(st_map.set_index('id').add_col_suffix('_st'),on='id',how='left')
            .assign(
                county=lambda x: x.county.mask(x.county.isna(),'Out of CA')
                ,state=lambda x: x.state.mask(x.state.isna(),x.state_st)
                ,country=lambda x: x.country.mask(~x.state.isna(),'United States')
            )
        )

    world=get_world_boundary()
    # null countries will be the rows that still have null county (after county and state)
    null_country=ent_adj.filt(lambda x: x.country.isna()).reset_index()
    if not null_country.empty:
        country_map=(
            null_country[['id','geometry']].to_crs(crs_ll)
            .sjoin(world.set_geometry('geometry_agg').to_crs(crs_ll)
                ,how='left',predicate='covered_by')
            .groupby('id').first().reset_index() # in case of multiple matches, just take the first
        )
        ent_adj=(
                ent_adj
                .join(country_map.set_index('id').add_col_suffix('_ct'),on='id',how='left')
                .assign(
                    county=lambda x: x.county.mask(x.county.isna(),'Out of CA')
                    ,state=lambda x: x.state.mask(x.state.isna(),'Out of US')
                    ,country=lambda x: x.country.mask(x.country.isna(),x.country_ct)
                )
            )

        # sometimes the country map covered_by predicate fails, we use dwithin as a backup
        null_country=ent_adj.filt(lambda x: x.country.isna()).reset_index()
        country_map=(
            gpd.sjoin_nearest(
                null_country[['id','geometry']].to_crs(crs_usa),
                world.set_geometry('geometry_agg').to_crs(crs_usa)
                ,how='left',max_distance=100000) # within 100km
            .to_crs(crs_ll)
            .groupby('id').first().reset_index() # in case of multiple matches, just take the first
        )
        assert(not country_map.empty)
        assert(not country_map.index_right.isna().any())
        ent_adj=(
                ent_adj
                .join(country_map.set_index('id').add_col_suffix('_ct2'),on='id',how='left')
                .assign(
                    county=lambda x: x.county.mask(x.county.isna(),'Out of CA')
                    ,state=lambda x: x.state.mask(x.state.isna(),'Out of US')
                    ,country=lambda x: x.country.mask(x.country.isna(),x.country_ct2)
                )
            )

    ent_adj=(
        ent_adj.drop(columns=ent_adj.filter(regex=f'_(cty|st|ct|ct2)$').columns)
    )

    # final checks to make sure we don't have any nulls in these fields (since we should have backfilled them all)    
    assert(~ent_adj.county.isna().any())
    assert(~ent_adj.state.isna().any())
    assert(~ent_adj.country.isna().any())

    return ent_adj


@cachier(wait_for_calc_timeout=5)
@pa.check_types
def get_ca_port_endpoints(
        regex=r'LOS ANGELES|LONG BEACH|OAKLAND',
        near_geoms:gpd.GeoDataFrame=None,
        dwithin:float=None
        ) -> DataFrame[LayerEndpointSchema]:
    """Load and geocode California port endpoint entities from the World Port Index.

    Reads the World Port Index shapefile, filters to major ports (large
    vessels, port of entry, entry permitted), then further filters by
    port name regex. Each port is assigned an ``rdrs_id`` (``PORT`` prefix),
    geocoded with county/state/country metadata, and conforms to
    ``FullGeocodedEndpointSchema``.

    Optionally filters ports to those within a distance of existing
    entity geometries, enabling proximity-based port selection.

    Parameters
    ----------
    regex : str, default r'LOS ANGELES|LONG BEACH|OAKLAND'
        Regex pattern to match port names. Only matching ports are included.
    near_geoms : GeoDataFrame, optional
        Reference geometries for proximity filtering. If provided with
        ``dwithin``, only ports within that distance of any reference
        geometry are included.
    dwithin : float, optional
        Maximum distance (in CRS units) for proximity filtering. Must be
        provided if ``near_geoms`` is given.

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        Port endpoint entities with full geocoding metadata, indexed by
        ``rdrs_id`` (renamed to ``id``).

    Notes
    -----
    - Port filtering criteria: ``max_vessel == 'L'``, ``portofentr == 'Y'``,
      ``entryother == 'Y'``.
    - County and state are assigned via ``dwithin`` spatial joins (1 km
      tolerance) against Census boundaries.
    - Multiple spatial join matches are resolved by ``groupby().first()``.
    - Results are cached via :mod:`cachier`.

    See Also
    --------
    get_ca_cty : County boundaries used for port region assignment.
    get_ca_boundary : California state boundary.
    """
    logger.info("Loading port data")

    world=get_world_boundary()
    ca_boundary=get_ca_boundary()
    ca_cty=get_ca_cty()
    states=pyg.states(year=2021,cache=True).clean_names()

    # world ports: https://hub.arcgis.com/datasets/1e7dfd86350e4e00b722b943a02cb14c/explore?showTable=true
    portsx=(
        # gpd.read_file(cpath('gis_data','World_Port_Index-shp.zip'))
        gpd.read_file(cpath('gis_data','World_Port_Index-shp.zip'))
        .clean_names()
        .join(world.set_index('iso_3166_1_'),on='country',rsuffix='_world')
    )

    # Limit to major ports
    ports_s=(
        portsx
        # .sjoin(us_state,how='inner')
        # .sjoin(ca_boundary,how='inner')
        .filt(lambda x: 
            True
            # & x.name.fillna('').str.match('United States') 
            # & ((x.port_name=='MARE ISLAND') | (x.port_name=='SAN DIEGO'))
            & (x.max_vessel=='L')
            # # & (x.chan_depth=='J')
            & (x.portofentr=='Y') 
            & (x.entryother=='Y') 
            # & (x.lift_100_=='Y')
            # & (x.cargo_depth=='L')
            )
    )

    # FIXME: this should go before creation of rdrs_ent_allxx
    logger.info(f"Creating port endpoints matching [{regex}]")

    # select California ports using a spatial join of all ports and the California boundary polygon.
    matchedports_n=(
        ports_s
        # limit to only the big ones!
        .filt(lambda x: x.port_name.fillna('').str.match(regex))
    )
    if near_geoms is not None and dwithin is not None:
        matchedports_s=(
            matchedports_n
            .to_crs(crs_usa)
            .sjoin_nearest(near_geoms.to_crs(crs_usa).add_col_suffix('_tmp').set_geometry('geometry_tmp')
                   ,max_distance=dwithin)
            .set_crs(crs_usa) # reset crs
            .drop_columns(regex='_tmp$')
            .reset_index(drop=True)
            )
    else:
        matchedports_s = matchedports_n

    # logger.info("Reading California county boundaries (to map Ports)")
    # ca_cty_s=(
    #     gpd.read_file(cpath('gis_data','ca-county-boundaries.zip!CA_Counties/CA_Counties_TIGER2016.shp')) # uses 3857
    #     .to_crs(crs_ll)
    #     .clean_names()
    #     .assign(
    #         centroid=lambda x: x.geometry.representative_point()
    #     )
    # )
    ports_ent_s=(
        matchedports_s
        .rename(columns={'country':'country_port'})
        
        # FIXME: rather than using backfill regions, we're going to explicitly do the mappings here to deal with some
        # edge cases. might want to consider this approach in backfill_regions
        .to_crs(crs_usa)
        .sjoin(ca_cty.set_geometry('geometry_agg').to_crs(crs_usa)[['geometry_agg','county']],how='left',predicate='dwithin',distance=1000).drop_columns(regex='index_right')
        .sjoin(states.to_crs(crs_usa)[['geometry','name']].rename(columns={'name':'state'}),how='left',predicate='dwithin',distance=1000).drop_columns(regex='index_right')
        # .sjoin(world.set_geometry('geometry_agg').to_crs(crs_usa)[['geometry_agg','country']],how='left',predicate='dwithin',distance=10000).drop_columns(regex='index_right')
        .assign(
            rdrs_id=lambda x: x.apply(lambda row: 'PORT'+str(row.fid).zfill(5), axis=1),
            n1=lambda x: x.port_name,
            n2=lambda x: 'PORT',
            county=lambda x: x.county,
            state=lambda x: x.state,
            country=lambda x: x.country_world,
            src='World Ports',
            geo_status='SUCCESS',
            geometry_src='World Ports',
            facility_type='PORT',
            facility_flags='',
            aggregation=EndpointAggregation.LOCATION.value
        )
        # since our sjoins might produce multiple matches
        .to_crs(crs_ll)
        .groupby('rdrs_id').first().reset_index()
        .set_crs(crs_ll)  # have to reset because of the grouping
        # .rename_geometry('geometry_merged')
        # FIXME: use df.set_flags(allows_duplicate_labels=False) throughout the codebase to catch these issues earlier
        .rename(columns={'id':'id_save'}) # make sure we don't stack ids in the index
        .rename(columns={'rdrs_id':'id'})
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
    )
    return ports_ent_s[schema_to_cols(LayerEndpointSchema)]


@cachier(wait_for_calc_timeout=5)
def get_ca_boundary():
    """Load the California state boundary polygon.

    Returns
    -------
    GeoDataFrame
        California state boundary in CRS ``EPSG:4326``, sourced from
        Census TIGER/Line state boundaries (2021).

    Notes
    -----
    - Results are cached via :mod:`cachier`.
    - Uses :func:`pygris.states` with ``statefp=='06'`` (California FIPS).
    - FIXME: type checking

    See Also
    --------
    get_us_state : All US state boundaries.
    get_ca_cty : California county boundaries.
    """
    logger.info("Reading California boundary")
    # return gpd.read_file(cpath('gis_data','ca-state-boundary.zip')).to_crs(crs_ll)
    return pyg.states(year=2021,cache=True).clean_names().filt(lambda x: x.statefp==fips_ca).to_crs(crs_ll)

ca_cty=None

@cachier(wait_for_calc_timeout=5)
@pa.check_types
@strict_check_types_with_report([
        _AggregatedSchemaWithNulls,
        _CountySchema
    ])
def get_ca_cty() -> Union[
        DataFrame[_AggregatedSchemaWithNulls],
        DataFrame[_CountySchema]
    ]:
    """Load California county boundary polygons with representative point geometries.

    Returns
    -------
    GeoDataFrame
        California counties with:

        - ``geometry`` — representative point (centroid-like) for each county
        - ``geometry_agg`` — full county boundary polygon
        - ``county`` — county name
        - ``geometry_src`` — ``'CENSUS County Boundaries'``

    Notes
    -----
    - Uses a module-level ``ca_cty`` singleton to avoid reloading.
    - Sourced from :func:`pygris.counties` with ``statefp=='06'``.
    - CRS is ``EPSG:4326``.
    - Results are cached via :mod:`cachier`.
    - Validated with :func:`strict_check_types_with_report` against both
      ``_AggregatedSchemaWithNulls`` and ``_CountySchema``.

    See Also
    --------
    get_ca_boundary : California state boundary.
    get_county_endpoints : County endpoint entities with population-weighted centroids.
    """
    global ca_cty
    if ca_cty is None:
        logger.info("Reading California counties")
        ca_cty=(
            # gpd.read_file(cpath('gis_data','ca-county-boundaries.zip!CA_Counties/CA_Counties_TIGER2016.shp')) # uses 3857
            pyg.counties(year=2021,cache=True,cb=True).clean_names()
            .filt(lambda x: x.statefp==fips_ca)
            .to_crs(crs_ll)
            .clean_names()
            .assign(
                centroid=lambda x: x.geometry.representative_point(),
                geometry_src='CENSUS County Boundaries',
                # facility_type='COUNTY',
            )
            .rename(columns={
                'name':'county',
                'geometry':'geometry_agg',
                'centroid':'geometry'
                })
            .set_geometry('geometry')
            .set_crs(crs_ll)
        )
    return ca_cty

# us_state=None

@pa.check_types
@strict_check_types_with_report([
        _AggregatedSchemaWithNulls,
        _StateEndpointSchema
    ])
def get_us_state(year:int = 2021) -> Union[
        DataFrame[_AggregatedSchemaWithNulls],
        DataFrame[_StateEndpointSchema]
    ]:
    """Load US state boundary polygons with representative point geometries.

    Parameters
    ----------
    year : int, default 2021
        Census year for TIGER/Line boundary data.

    Returns
    -------
    GeoDataFrame
        US states with:

        - ``geometry`` — representative point for each state
        - ``geometry_agg`` — full state boundary polygon
        - ``state`` — state name
        - ``geometry_src`` — ``'CENSUS State Boundaries'``

    Notes
    -----
    - Sourced from :func:`pygris.states`.
    - CRS is ``EPSG:4326``.
    - Validated with :func:`strict_check_types_with_report` against both
      ``_AggregatedSchemaWithNulls`` and ``_StateEndpointSchema``.

    See Also
    --------
    get_state_endpoints : State endpoint entities for geocoding.
    get_ca_boundary : California state boundary only.
    """
    return (
        pyg.states(year=year,cache=True)
            .to_crs(crs_ll)
            .clean_names()
            .assign(
                centroid=lambda x: x.geometry.representative_point(),
                geometry_src='CENSUS State Boundaries',
            )
            .rename(columns={
                'name':'state',
                'geometry':'geometry_agg',
                'centroid':'geometry'
                })
            .set_geometry('geometry')
            .set_crs(crs_ll)
        )

@cachier(wait_for_calc_timeout=5)
@pa.check_types
@strict_check_types_with_report([
        _AggregatedSchemaWithNulls,
        _CountryEndpointSchema
    ])
def get_world_boundary() -> Union[
        DataFrame[_AggregatedSchemaWithNulls],
        DataFrame[_CountryEndpointSchema]
    ]:
    """Load world country boundary polygons with representative point geometries.

    Returns
    -------
    GeoDataFrame
        World countries with:

        - ``geometry`` — representative point for each country
        - ``geometry_agg`` — full country boundary polygon
        - ``country`` — country name
        - ``geometry_src`` — ``'World Administrative Boundaries'``

    Notes
    -----
    - Sourced from ``world-administrative-boundaries.zip`` in ``gis_data``.
    - The source file has no CRS set; ``EPSG:4326`` is assigned explicitly.
    - CRS is ``EPSG:4326``.
    - Results are cached via :mod:`cachier`.
    - Validated with :func:`strict_check_types_with_report` against both
      ``_AggregatedSchemaWithNulls`` and ``_CountryEndpointSchema``.

    See Also
    --------
    get_world_endpoints : Country endpoint entities for geocoding.
    backfill_regions : Uses world boundaries for country assignment.
    """
    logger.info("Reading World boundary")
    world_boundary=(
        gpd.read_file(cpath('gis_data','world-administrative-boundaries.zip'))  # CRS not set, but it's in Lat Lon
        .clean_names()
        .set_crs(crs_ll)
        .assign(
            centroid=lambda x: x.apply(lambda row: row.geometry.representative_point(),axis=1),
            geometry_src='World Administrative Boundaries'
            )
        .rename(columns={
            'name':'country',
            'geometry':'geometry_agg',
            'centroid':'geometry'
            })
        .set_geometry('geometry')
        .set_crs(crs_ll)
    )
    return world_boundary

@pa.check_types
@cachier(wait_for_calc_timeout=5)
def get_zcta() -> GeoDataFrame:
    """Load California ZIP Code Tabulation Area (ZCTA) boundaries.

    Returns
    -------
    GeoDataFrame
        California ZCTA polygons in CRS ``EPSG:4326``, sourced from
        ``CA_ZCTA_2020`` shapefile.

    See Also
    --------
    get_zcta_endpoints : ZCTA endpoint entities with geocoding metadata.
    get_bad_zips : ZIP codes not covered by ZCTA.
    """
    # FIXME: consider: https://data.ca.gov/dataset/ca-zip-code-boundaries
    # return gpd.read_file(cpath('gis_data','CA_ZCTA_2020-fixed.zip')).clean_names().to_crs(crs_ll)
    return gpd.read_file(settings.get('model_paths').zcta_shapefile)


def get_bad_zips() -> GeoDataFrame:
    """Load ZIP codes that are not represented in the ZCTA shapefile.

    These are manually geocoded ZIP codes with latitude/longitude
    coordinates provided in a lookup CSV.

    Returns
    -------
    GeoDataFrame
        Manually geocoded ZIP codes with ``id``, ``lat``, ``lon``, and
        ``geometry`` (Point) columns in CRS ``EPSG:4326``.

    Raises
    ------
    AssertionError
        If any geometry is invalid.

    See Also
    --------
    get_zcta_endpoints : Augments ZCTA endpoints with these manual lookups.
    """
    bz= (
        pd.read_csv(cpath('gis_data','bad_zip_lookup - bad_zip_lookup.csv'))
        .clean_names()
        .filt(lambda x: x.lon.notna() & x.lat.notna())
        .assign(geometry=lambda x: gpd.points_from_xy(x.lon,x.lat))
        .pipe(gpd.GeoDataFrame,geometry='geometry',crs=crs_ll)
    )
    assert(bz.geometry.is_valid.all())
    return bz

@pa.check_types()
def get_zcta_missing() -> DataFrame[LayerEndpointSchema]:
    return (
        pd.read_pickle(cpath('gis_data','zcta_missing.pkl'))
        .assign(geometry_src='ZCTA missing')
    )
        

@cachier(wait_for_calc_timeout=5)
@pa.check_types
# @strict_check_types_with_report
def get_zcta_endpoints() -> LayerEndpointSchema:
    """Build ZCTA endpoint entities augmented with manually geocoded ZIP codes.

    Creates endpoint entities for all California ZCTAs, then augments
    them with ZIP codes from the manual lookup (``bad_zip_lookup``)
    that aren't contained within any ZCTA polygon. Each ZCTA is assigned
    a representative point geometry and backfilled with region metadata.

    Parameters
    ----------
    None (reads from fixed data sources).

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        ZCTA endpoint entities with:

        - ``id`` — ``ZIP`` + zero-padded GEOID
        - ``geometry`` — ZCTA representative point
        - ``geometry_agg`` — ZCTA boundary polygon
        - ``zips_contained`` — list of manual ZIP IDs contained within
          each ZCTA
        - Full region metadata from :func:`backfill_regions`

    Notes
    -----
    - Manual ZIP codes not contained in any ZCTA are added as standalone
      endpoints with a 100 m buffer as ``geometry_agg``.
    - The ``zips_contained`` column enables mapping from raw ZIP codes
      to ZCTA endpoint IDs.
    - Results are cached via :mod:`cachier`.

    See Also
    --------
    get_zcta : Raw ZCTA boundaries.
    get_bad_zips : Manual ZIP code lookup.
    zip_to_id_mapping : Builds the ZIP → endpoint ID dictionary from this output.
    backfill_regions : Assigns county/state/country metadata.
    """
    zcta = (
        get_zcta()
        # pyg.zctas(state="06",year=2021,cache=True).clean_names()
        .clean_names()
        .assign(
            id=lambda x: 'ZIP'+x.zip_code,
            n1=lambda x: x.id,
            n2=lambda x: x.po_name,
            src='ZCTA',
            geo_status='SUCCESS',
            facility_type='ZCTA',
            facility_flags='',
            geometry_src='ZCTA',
            geometry_zip=lambda x: x.geometry.to_crs(crs_ca).representative_point(), # FIXME: should use census data to get pop weighted centroid
            aggregation=EndpointAggregation.ZCTA.value
        )
        .rename_geometry('geometry_agg')
        .to_crs(crs_ll)
        .set_geometry('geometry_zip')
        .rename_geometry('geometry')
        .to_crs(crs_ll)
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        .pipe(lambda x: backfill_regions(x))
        [schema_to_cols(LayerEndpointSchema)]
        # [['n1','n2','src','geo_status','geometry_src','geometry_agg','geometry','aggregation']]
    )
    bz=get_bad_zips().drop_duplicates(keep='first')

    zcta_with_bz=(
        zcta.set_geometry('geometry_agg').to_crs(crs_ca)
        .set_flags(allows_duplicate_labels=True)

        .sjoin(bz.to_crs(crs_ca)[['id','geometry']].assign(geometry_right=lambda x: x.geometry).rename(columns={'id':'id_right'})
               ,predicate='contains',how='left')
        
        .reset_index()
        .groupby('id').first()
        .set_flags(allows_duplicate_labels=False)
    )
    zcta_with_bz_contained=(
        zcta_with_bz.reset_index().groupby('id')
        .agg({**dict.fromkeys(zcta_with_bz.filter(regex='^(?!.*_right)').columns,'first'),
              **{'id_right':lambda x: x.dropna().tolist()}
              })
        .rename(columns={'id_right':'zips_contained'})
    )
    bz_not_contained_by_zcta=(
        zcta.set_geometry('geometry_agg').to_crs(crs_ca)
        .sjoin(bz.to_crs(crs_ca)[['id','geometry']].assign(geometry_right=lambda x: x.geometry).rename(columns={'id':'id_right'})
               ,predicate='contains',how='right')
        .filt(lambda x: x.id.isna())
    )

    zcta_missing=get_zcta_missing()
    zcta_augmented_with_bz=(
        pd.concat([
            zcta_with_bz_contained.set_geometry('geometry_agg').set_crs(crs_ca).to_crs(crs_ll)
            .set_flags(allows_duplicate_labels=True)

            ,
            zcta_missing
            .set_flags(allows_duplicate_labels=True)
            ,
            bz_not_contained_by_zcta[['id_right','geometry']].rename(columns={'id_right':'id'})
            .set_flags(allows_duplicate_labels=True)
            .assign(geometry_src='ZCTA',
                    aggregation=EndpointAggregation.ZCTA.value,
                    facility_type='ZCTA',
                    facility_flags='manual ZCTA lookup',
                    n1=lambda x: x.id, n2=lambda x: x.id,
                    zips_contained=lambda x: x.apply(lambda row: [],axis=1))
            .set_index('id')
            .to_crs(crs_ll)
            .pipe(backfill_regions)
            .assign(geometry_agg=lambda x: x.geometry.to_crs(crs_ca).buffer(1000))
            .set_geometry('geometry_agg').to_crs(crs_ll)
            .set_geometry('geometry')
        ])
        .pipe(lambda df: df[~df.index.duplicated(keep='first')])
        .set_flags(allows_duplicate_labels=False)
        .assign(zips_contained=lambda x: [v if isinstance(v, list) else [] for v in x['zips_contained']])
    )
    return(zcta_augmented_with_bz)

def zip_to_id_mapping() -> dict[str, str]:
    """Build a dictionary mapping ZIP codes to their containing endpoint IDs.

    Creates a complete mapping from every ZIP code (both ZCTA IDs and
    manually looked-up ZIPs) to the endpoint entity ID that represents
    them. Each ZCTA endpoint maps to itself plus any manual ZIPs it
    contains.

    Returns
    -------
    dict
        Mapping from ZIP code string to endpoint ID string.

    Notes
    -----
    - ZCTA IDs are included in their own ``zips_contained`` list so that
      ZCTA-to-ZCTA mappings are also present.
    - Only non-NaN ZIP codes are included in the mapping.

    See Also
    --------
    get_zcta_endpoints : Produces the augmented ZCTA data this mapping is
        derived from.
    """
    zcta_augmented_with_bz=get_zcta_endpoints()
    zip_to_id = (
        zcta_augmented_with_bz.reset_index()
        # add self to zips_contained list so we can have a complete mapping
        .assign(
            zips_contained=lambda x: x.zips_contained + x[['id']].values.tolist()
        )
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        ['zips_contained']
        .explode()
        .reset_index()
        .filt(lambda x: x.zips_contained.notna())
        .set_index('zips_contained')['id']
        .to_dict()
    )
    return zip_to_id


### DISMANTLERS
def read_dismantler(dismantler_file:Path):
    """Load ELV dismantler facility data with parcel geometries.

    Reads dismantler facility data from CSV, creates Point geometries
    from latitude/longitude, normalizes ZIP codes to 5-digit format,
    and joins with parcel-level geometries from a separate shapefile.

    Parameters
    ----------
    dismantler_file : str, default cpath('elv_data','Dismantler list-crr.xlsx - Sheet 1.csv')
        Path to the dismantler facility CSV file.

    Returns
    -------
    GeoDataFrame
        Dismantler facilities with:

        - ``id`` — ``DISM`` + zero-padded index
        - ``geometry`` — Point from lat/lon
        - ``geometry_agg`` — Parcel polygon from shapefile
        - ``zip_code`` — Normalized 5-digit ZIP

    Notes
    -----
    - ZIP codes are truncated to 5 digits using regex.
    - Parcel geometries are loaded from ``all_dismantlers.zip``.
    - Facilities without lat/lon get ``None`` geometry.

    See Also
    --------
    get_dismantler_endpoints : Produces full endpoint entities from this data.
    """
    dl=gpd.GeoDataFrame(
        pd.read_csv(dismantler_file)
        .clean_names().strip_col_names()
        .assign(
            geometry=lambda x: gpd.GeoSeries(
                x.apply(
                    lambda row: (
                        shapely.geometry.Point(row.lon,row.lat)
                        if not pd.isna(row.lon) else None
                        )
                ,axis=1)
            ).set_crs(crs_ll),

            zip_code=lambda x: x.zip.str.replace(r'^(\d{5}).*', r'\1', regex=True),
        )
        .drop_columns(['rdrs'])
        ,geometry='geometry'
        ,crs=crs_ll        
    )
    
    dism_file=settings.get('model.elv').all_dismantlers_file.format(**settings.get('model.elv').to_dict())
    dism_parcel_geom=(
        gpd.read_file(dism_file)
    )
    dl3=(
        dl.reset_index()
        .merge(dism_parcel_geom.add_col_suffix('_dd'),left_on='index',right_on='myindex_dd',how='inner')
        .assign(
            id=lambda x: x.reset_index().apply(lambda row: 'DISM'+str(row["index"]).zfill(5), axis=1)
        )
        .rename(columns={'geometry_dd':'geometry_agg'})
    )
    return dl3

@cachier(wait_for_calc_timeout=5)
@pa.check_types
def get_dismantler_endpoints(dismantler_file:Path) -> DataFrame[LayerEndpointSchema]:
    """Build ELV dismantler endpoint entities with full geocoding metadata.

    Wraps :func:`read_dismantler` to produce endpoint entities conforming
    to ``FullGeocodedEndpointSchema``, with region metadata backfilled.

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        Dismantler endpoint entities with:

        - ``id`` — ``DISM`` + zero-padded index
        - ``n1`` — facility name (``'UNKNOWN'`` if missing)
        - ``n2`` — street address
        - ``facility_type`` — ``'DISMANTLER'``
        - Full region metadata from :func:`backfill_regions`

    Notes
    -----
    - Results are cached via :mod:`cachier`.
    - FIXME: check types

    See Also
    --------
    read_dismantler : Raw dismantler data loading.
    backfill_regions : County/state/country assignment.
    """
    dl=read_dismantler(dismantler_file)
    dl2=(
        dl
        .assign(
            n1=lambda x: x.name.fillna('UNKNOWN'),
            n2=lambda x: x.street,
            src='Dismantler List',
            geo_status='SUCCESS',
            geometry_src='Dismantler List',
            facility_type='DISMANTLER',
            facility_flags='',
            aggregation=EndpointAggregation.LOCATION.value
        )
        .set_geometry('geometry_agg')
        .to_crs(crs_ll)
        .set_geometry('geometry')
        .to_crs(crs_ll)
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        .pipe(lambda x: backfill_regions(x))
    )
    return dl2[schema_to_cols(LayerEndpointSchema)]


class Geocoder(ABC):
    """Abstract base class for endpoint geocoders.

    Defines the interface for geocoding endpoint entities. Subclasses
    must implement :meth:`_geocode` to provide geocoding logic.

    Parameters
    ----------
    name : str, default 'BaseGeocoder'
        Human-readable name for the geocoder (used in logging).
    select : callable, optional
        Filter function applied to entities before geocoding. Accepts a
        DataFrame and returns a boolean Series. If ``None``, all entities
        are selected.

    Attributes
    ----------
    name : str
        Geocoder name.
    select : callable or None
        Entity filter function.

    Methods
    -------
    geocode(entities)
        Public geocoding entry point with logging.
    _geocode(entities)
        Abstract method — subclass geocoding implementation.
    selected(df)
        Evaluate the select filter on a DataFrame.
    """

    def __init__(
            self,
            name:str = 'BaseGeocoder',
            select: Callable[[pd.Series],bool] = None, # optional filter to apply to entities before geocoding
            require_complete: bool = False             # set to True if resulting dataframe must have all geometries geocoded
            ):
        """Initialize the Geocoder."""
        self.select = select
        self.name = name
        self.require_complete = require_complete

    def selected(self, df):
        if self.select is None:
            return True
        else:
            return self.select(df)
        
    def geocode(
            self, 
            entities: DataFrame,                       # entities to map
        ) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Geocode a set of endpoints using this geocoder.

        Logs the geocoding operation and delegates to :meth:`_geocode`.

        Parameters
        ----------
        entities : DataFrame
            Endpoint entities to geocode.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with ``geometry``, ``geometry_src``, and
            ``aggregation`` columns populated where possible.
        """
        logger.info(f"Geocoding {len(entities)} entities with {self.name} geocoder")
        ent:DataFrame[GeocodedEndpointSchemaWithNulls] = self._geocode(entities)
        if self.require_complete==True:
            if not ent.geometry.is_valid.all() or ent.geometry.isna().any():
                raise RuntimeError(f'Null geometries produced in geocoder {self.name} when completeness required')
        
        return ent

    @abstractmethod
    def _geocode(
            self, 
            entities: DataFrame,                       # entities to map
        ) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Implement geocoding logic for a set of endpoints.

        Parameters
        ----------
        entities : DataFrame
            Endpoint entities to geocode.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with geocoding columns populated.
        """
        pass
    
class GeocodeFromLatLon(Geocoder):
    """Geocode entities directly from latitude/longitude columns.

    Creates Point geometries from specified lat/lon columns in the
    entity data. If the columns don't exist, returns entities with
    null geometries and a warning.

    Parameters
    ----------
    lat_col : str, default 'lat'
        Name of the latitude column.
    lon_col : str, default 'lon'
        Name of the longitude column.
    name : str, default 'LatLon'
        Geocoder name.
    select : callable, optional
        Entity filter function.

    Methods
    -------
    _geocode(entities)
        Create Point geometries from lat/lon columns.

    Notes
    -----
    - Aggregation is set to ``LOCATION`` only for Point geometries
      that pass the select filter.
    - If lat/lon columns are missing, a warning is logged and entities
      are returned with null geometries.
    """

    def __init__(
            self, 
            lat_col:str = 'lat',
            lon_col:str = 'lon',
            name:str = None,
            select: Callable[[pd.Series],bool] = None, # optional filter to apply to entities before geocoding
            require_complete: bool = True              # set to True if resulting dataframe must have all geometries geocoded
            ):
        """Initialize the Geocoder."""
        self.lat_col = lat_col
        self.lon_col = lon_col
        thename = f"{__class__.__name__} ({name})" if name else __class__.__name__
        super().__init__(name=thename,select=select,require_complete=require_complete)

    @pa.check_types
    def _geocode(
            self, 
            entities: DataFrame[EndpointSchema]
        ) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Create Point geometries from latitude/longitude columns.

        Parameters
        ----------
        entities : DataFrame[EndpointSchema]
            Entities with ``lat_col`` and ``lon_col`` columns.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with ``geometry`` set to Point(lon, lat) where
            coordinates are available.
        """
        gdf = (
            entities.copy()
            .add_column_if_missing('aggregation',EndpointAggregation.NONE.value)
            .add_column_if_missing('geometry_src', pd.NA)
            .add_column_if_missing('geometry', pd.NA)
        )
        if (self.lon_col not in gdf.columns) or (self.lat_col not in gdf.columns):
            logger.warning(f"{self.name} geocoder: Latitude and/or longitude columns [{self.lat_col}, {self.lon_col}] not found in entities. Returning entities with null geometries.")
            return gdf
        else:
            gdf = (
                gdf
                .assign(  # and encode a geometry for them
                    geometry=lambda x: x.apply(
                        lambda row: shapely.geometry.Point(row[self.lon_col], row[self.lat_col]), axis=1),
                    geometry_src=self.name,
                    
                    # FIXME: not sure we want to set aggregation here but needed for validation
                    aggregation=lambda x: case_when(
                        self.selected(x) & (x.set_geometry('geometry').geometry.geom_type == 'Point'), EndpointAggregation.LOCATION.value,
                        True, x.aggregation
                    ),
                )
                .pipe(lambda df: gpd.GeoDataFrame(df,geometry='geometry'))
                .set_crs(crs_ll)
            )
            return gdf


class GeocodeById(Geocoder):
    """Geocode entities by matching their ID to a pre-built geocoding lookup table.

    Joins entities against a reference DataFrame of geocoded endpoints
    on the ``id`` column, transferring geometry and name information
    from the lookup to the entities.

    Parameters
    ----------
    geocoder_data : DataFrame[GeocodedEndpointSchemaWithNulls]
        Pre-built geocoding lookup indexed by ``id``.
    select : callable, optional
        Entity filter function.

    Methods
    -------
    _geocode(entities)
        Join entities against the geocoding lookup.

    Notes
    -----
    - Existing geometries are preserved (not overwritten).
    - Name fields (``n1``, ``n2``) from the lookup are merged via
      ``combine_first`` to fill in missing names.
    - Aggregation is set to ``LOCATION`` for Point geometries that
      pass the select filter.
    """
    #@pa.check_types
    def __init__(
            self, 
            geocoder_data: DataFrame[GeocodedEndpointSchemaWithNulls],
            name:str = None,
            select: Callable[[pd.Series],bool] = None, # optional filter to apply to entities before geocoding
            require_complete: bool = True              # set to True if resulting dataframe must have all geometries geocoded
            ):
        """Initialize the Geocoder with a geocoding DataFrame."""
        thename = f"{__class__.__name__} ({name})" if name else __class__.__name__
        super().__init__(name=thename,select=select,require_complete=require_complete)
        self.geocoder_data = geocoder_data

    @pa.check_types
    def _geocode(
            self, 
            entities: DataFrame[EndpointSchema]
        ) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Geocode entities by joining on ID against the lookup table.

        Parameters
        ----------
        entities : DataFrame[EndpointSchema]
            Entities with ``id`` column matching the lookup index.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with geometry and names from the lookup where matched.
        """
        # if 'geometry' in entities.columns:
        #     raise ValueError("Entities already have geometry.")
        
        df = entities.copy().add_column_if_missing('geometry_src', pd.NA).add_column_if_missing('geometry', pd.NA)
        df = df.join(
            self.geocoder_data

            # some geocoders might provide n1/n2 info, some might not
            .add_column_if_missing('n1',None)
            .add_column_if_missing('n2',None)
            [['n1','n2','geometry']]
            .set_geometry('geometry')
            .to_crs(crs_ll).add_col_suffix('_geocoder'),
            on='id'
        )
        df = (df
              .add_column_if_missing('geometry_src',None)
              .assign(
                geometry=lambda x: case_when(
                    ~x.geometry.isna(), x.geometry,
                    self.selected(x) & ~x['geometry_geocoder'].isna(), x['geometry_geocoder'],
                    True, None
                ),
                geometry_src=lambda x: case_when(
                    ~x.geometry.isna(), x.geometry_src,
                    self.selected(x) & ~x['geometry_geocoder'].isna(), __class__.__name__,
                    True, x.geometry_src
                ),
                # FIXME: not sure we want to set aggregation here but needed for validation
                aggregation=lambda x: case_when(
                    self.selected(x) & (x.set_geometry('geometry').geometry.geom_type == 'Point'), EndpointAggregation.LOCATION.value,
                    True, x.aggregation
                ),

                # we're going to set n1 and n2 if they're missing
                n1 = lambda x: x.n1.combine_first(x.n1_geocoder),
                n2 = lambda x: x.n2.combine_first(x.n1_geocoder)
            )
        )
        df = df.drop(columns=['geometry_geocoder'])

        # reconstruct
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_ll)#[['aggregation','geometry']]
        return gdf
    
class GeocodeByCountyStateCountry(Geocoder):
    """Geocode entities by resolving their county, state, or country names to boundary centroids.

    Uses a cascading lookup against California county, US state, and
    world country boundaries. For each entity, the most specific
    available administrative region name is resolved to a representative
    point geometry and boundary polygon.

    Parameters
    ----------
    select : callable, optional
        Entity filter function.

    Attributes
    ----------
    ca_cty_s : GeoDataFrame
        California county boundaries (loaded at init).
    us_state_s : GeoDataFrame
        US state boundaries (loaded at init).
    world_s : GeoDataFrame
        World country boundaries with RDRS name mapping (loaded at init).

    Methods
    -------
    _geocode(entities)
        Resolve county/state/country names to geometries.

    Notes
    -----
    - Country names in the RDRS flow data may differ from the world
      boundary shapefile. A mapping (``country_name_map``) reconciles
      common discrepancies (e.g., ``'United Kingdom'`` vs.
      ``'U.K. of Great Britain and Northern Ireland'``).
    - The cascade order is: county → state → country. The first
      successful match wins.
    - Both ``geometry`` (representative point) and ``geometry_agg``
      (full boundary) are populated.
    """
    def __init__(
            self, 
            # geocoder_data: DataFrame[GeocodedEndpointSchemaWithNullGeometries],
            name:str = None,
            select: Callable[[pd.Series],bool] = None,  # optional filter to apply to entities before geocoding
            require_complete: bool = False              # set to True if resulting dataframe must have all geometries geocoded
    ):
        """Initialize the Geocoder with a geocoding DataFrame."""
        thename = f"{__class__.__name__} ({name})" if name else __class__.__name__
        super().__init__(name=thename,select=select,require_complete=require_complete)
        # self.geocoder_data = geocoder_data.set_index(['county','state','country'])
        logger.info("Reading California county boundaries (for region geocoding)")
        self.ca_cty_s=get_ca_cty()

        logger.info("Reading US state boundaries (for region geocoding)")
        self.us_state_s=get_us_state()

        logger.info("Reading country boundaries (for region geocoding)")
        self.world_s=get_world_boundary()

        # Some country names in the RDRS flow data differ from our country shapefile so we create a mapping.
        country_name_map=pd.DataFrame([
            ('United States','United States of America'),
            ('United Kingdom','U.K. of Great Britain and Northern Ireland'),
            ('Korea, South','Republic of Korea'),
            ('Korea, North',"Democratic People's Republic of Korea"),
            ('Russia',"Russian Federation"),
            ('Tanzania','United Republic of Tanzania'),
            ('Laos',"Lao People's Democratic Republic"),
            ('British Columbia','Canada')
        ],columns=['rdrs_name','world_name'])
        self.world_s=(
            self.world_s
            .rename(columns={'country':'o_country'})
            .join(
               country_name_map.set_index('world_name'),
               on='o_country'
            )
            .assign(
                country=lambda x: case_when(
                    ~x.rdrs_name.isna(), x.rdrs_name,
                    True, x.o_country
                )
            )
            .drop(columns=['rdrs_name'])
        )


    @pa.check_types
    def _geocode(
            self, 
            # entities needs to be namedendpoint since that includes county/state/country
            # needs to be withnulls since n1/n2/county/state/country might all be na
            # which is okay since the geocoder can just ignore them.
            entities: DataFrame[NamedEndpointSchemaWithNulls], 
            select: Callable[[pd.Series],bool] = None  # optional filter to apply to entities before geocoding
        ) -> DataFrame[GeocodedEndpointSchemaWithNamedNulls]:  # allow n1/n2 to be null
        """Resolve county/state/country names to representative point geometries.

        Parameters
        ----------
        entities : DataFrame[NamedEndpointSchemaWithNulls]
            Entities with ``county``, ``state``, and/or ``country`` columns.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNamedNulls]
            Entities with ``geometry``, ``geometry_agg``, and
            ``geometry_src`` populated from the most specific available
            region match.
        """
        # if 'geometry' in entities.columns:
        #     raise ValueError("Entities already have geometry.")
        
        df = (
            entities.copy()
            .add_column_if_missing('geometry_src', pd.NA)
            .add_column_if_missing('geometry', pd.NA)
            .add_column_if_missing('geometry_agg', pd.NA)
            .add_column_if_missing('aggregation', pd.NA)
        )

        gcs = {
            'county': self.ca_cty_s,
            'state': self.us_state_s,
            'country': self.world_s,
        }

        for typ, gc in gcs.items():
            df = df.join(
                gc.set_index(typ)[['geometry','geometry_agg']].to_crs(crs_ll),
                on=typ,
                rsuffix=f'_{typ}'
            )
            df = df.assign(
                geometry=lambda x: case_when(
                    ~x.geometry.isna(), x.geometry,
                    self.selected(x) & ~x[f'geometry_{typ}'].isna(), x[f'geometry_{typ}'],
                    True, None
                ),
            )
            df = df.assign(
                geometry_agg=lambda x: case_when(
                    ~x.geometry_agg.isna(), x.geometry_agg,
                    self.selected(x) & ~x[f'geometry_agg_{typ}'].isna(), x[f'geometry_agg_{typ}'],
                    True, None
                ),
            )
            df = df.assign(
                geometry_src=lambda x: case_when(
                    ~x.geometry_src.isna(), x.geometry_src,
                    self.selected(x) & ~x[f'geometry_agg_{typ}'].isna(), f'{__class__.__name__}__{typ}',
                    True, None
                ),
            )
            df = df.drop(columns=[
                f'geometry_{typ}',
                f'geometry_agg_{typ}'
                ])

        # reconstruct
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_ll)#[['aggregation','geometry']]
        return gdf
    

class MergeGeocoder(Geocoder):
    """Compose multiple geocoders into a priority-ordered stack.

    Applies geocoders in insertion order, with earlier geocoders taking
    precedence. Each geocoder's results are merged into the accumulating
    output, filling in geometries that previous geocoders couldn't resolve.

    Parameters
    ----------
    geocoder_dict : dict[str, Geocoder]
        Ordered mapping of geocoder names to instances. Applied in
        insertion order (use :class:`OrderedDict` for explicit ordering).
    select : callable, optional
        Entity filter function.

    Methods
    -------
    _geocode(entities)
        Apply all geocoders in sequence.

    Notes
    -----
    - Once a geometry is set by a higher-priority geocoder, it is not
      overwritten by subsequent geocoders.
    - Name fields (``n1``, ``n2``) are merged via ``combine_first``
      from each geocoder's output.
    - Raises :class:`ValueError` if entities already have a ``geometry``
      column (to prevent accidental overwrites).

    Raises
    ------
    ValueError
        If ``entities`` already contains a ``geometry`` column.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> geocoders = OrderedDict([
    ...     ('manual', GeocodeById(manual_data)),
    ...     ('county', GeocodeById(county_data)),
    ...     ('latlon', GeocodeFromLatLon()),
    ... ])
    >>> merged = MergeGeocoder(geocoders)
    >>> result = merged.geocode(entities)
    """
    def __init__(
            self, 
            geocoder_dict: dict[str, Geocoder],
            name:str = None,
            select: Callable[[pd.Series],bool] = None, # optional filter to apply to entities before geocoding
            require_complete: bool = True              # set to True if resulting dataframe must have all geometries geocoded
            ):
        """Initialize the Geocoder."""
        thename = f"{__class__.__name__} ({name})" if name else __class__.__name__
        super().__init__(name=thename,select=select,require_complete=require_complete)

        self.geocoders = geocoder_dict
        
        # we're going to disable require_complete in all child geocoders since the whole point of 
        # of a merge geocoder is that we're chaining geocoders to get complete geocoding done
        for (name, geocoder) in self.geocoders.items():
            geocoder.require_complete = False
        
        # for (name, geocoder) in self.geocoders.items():
        #     GeocodedEndpointSchemaWithNullGeometries.validate(geocoder)

    @pa.check_types
    def _geocode(self, entities: DataFrame[EndpointSchema]) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Apply all geocoders in sequence, merging results by priority.

        Parameters
        ----------
        entities : DataFrame[EndpointSchema]
            Entities without a ``geometry`` column.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with the best available geometry from the geocoder
            stack.

        Raises
        ------
        ValueError
            If ``entities`` already contains a ``geometry`` column.
        """

        if 'geometry' in entities.columns:
            raise ValueError("Entities already have geometry.")
        
        df = (
            entities.copy()
            .add_column_if_missing('geometry_src', pd.NA)
            .add_column_if_missing('geometry', pd.NA)
            .add_column_if_missing('aggregation', pd.NA)
        )
        logger.info("Geocoding entities using available geocoders.")
        for name, geocoder in self.geocoders.items():
            logger.infox(f"Processing geocoder '{name}'")
            df = df.join(
                gpd.GeoDataFrame(
                    # geocoders might encode n1 and n2
                    geocoder.geocode(df)[['n1','n2','geometry']]
                    .add_col_suffix(f'_{name}')
                    ,geometry=f'geometry_{name}',crs=crs_ll
                ),
                on='id',
                rsuffix=f'_{name}'
            )
            geom_col = f'geometry_{name}'
            n1_col = f'n1_{name}'
            n2_col = f'n2_{name}'
            df = df.assign(
                geometry_src=lambda x: case_when(
                    ~x.geometry.isna(), x.geometry_src,
                    self.selected(x) & ~x[geom_col].isna(), name,
                    True, None
                ),
                geometry=lambda x: case_when(
                    ~x.geometry.isna(), x.geometry,
                    self.selected(x) & ~x[geom_col].isna(), x[geom_col],
                    True, None
                ),
                n1=lambda x: x.n1.combine_first(x[n1_col]),
                n2=lambda x: x.n1.combine_first(x[n2_col])
            )
            logger.infox(f"Geocoding entities: processed geocoder '{name}'.")
            logger.trace(f"Dropping temporary geometry column '{geom_col}'.")
            df = df.drop(columns=[geom_col])

        # reconstruct
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_ll)
        return gdf
    
class NullGeocoder(Geocoder):
    """Identify and flag endpoints that cannot be geocoded.

    Assigns a NaN Point geometry and ``'NULL'`` geometry source to
    entities matching the select filter, marking them as unresolvable.
    Should be placed last in the :class:`MergeGeocoder` stack as a
    safety net.

    Parameters
    ----------
    select : callable, optional
        Filter identifying uncodable entities (e.g., entities with
        invalid geometries).

    Methods
    -------
    _geocode(entities)
        Flag matching entities with null geometries.

    Notes
    -----
    - Matching entities receive ``geometry = Point(NaN, NaN)``,
      ``geometry_src = 'NULL'``, and ``aggregation = NONE``.
    - Non-matching entities are left unchanged.
    """
    def __init__(
            self, 
            name:str = None,
            select: Callable[[pd.Series],bool] = None,  # optional filter to apply to entities before geocoding
            require_complete: bool = False              # set to True if resulting dataframe must have all geometries geocoded
            ):
        """Initialize the Geocoder."""
        thename = f"{__class__.__name__} ({name})" if name else __class__.__name__
        super().__init__(name=thename,select=select,require_complete=require_complete)

    @pa.check_types
    def _geocode(self, 
                entities: DataFrame[EndpointSchema]
            ) -> DataFrame[GeocodedEndpointSchemaWithNulls]:
        """Flag uncodable entities with null geometries.

        Parameters
        ----------
        entities : DataFrame[EndpointSchema]
            Entities to evaluate.

        Returns
        -------
        DataFrame[GeocodedEndpointSchemaWithNulls]
            Entities with null geometries and ``'NULL'`` source where
            the select filter matches.
        """
        df = (
            entities.copy()
            .add_column_if_missing('geometry_src', pd.NA)
            .add_column_if_missing('geometry', pd.NA)
            .add_column_if_missing('geometry_agg', pd.NA)
            .add_column_if_missing('aggregation', pd.NA)
            .assign(
                geometry=lambda x: case_when(
                    self.select(x), shapely.Point(np.nan,np.nan), # null point
                    True, x['geometry']), 
                geometry_src=lambda x: case_when(
                    self.select(x), 'NULL',
                    True, x.geometry_src),
                aggregation=lambda x: case_when(
                    self.select(x), EndpointAggregation.NONE.value,
                    True, x.aggregation)
            )
        )

        # reconstruct
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_ll)
        return gdf

@cachier(wait_for_calc_timeout=5)            
def geocode_it(geolocator_a: Nominatim, cnt: int, verbose:bool=False):
    """Create a geocoding callback function with progress tracking.

    Returns a closure that geocodes a single address string using the
    provided geolocator, with an internal counter for progress reporting.

    Parameters
    ----------
    geolocator_a : Nominatim
        Configured Nominatim geolocator instance.
    cnt : int
        Total number of entities to geocode (for progress percentage).
    verbose : bool, default False
        If True, print progress for each geocode call.

    Returns
    -------
    callable
        Function that accepts an address string and returns a
        :class:`geopy.Location` object or ``None``.

    Notes
    -----
    - Results are cached via :mod:`cachier`.
    - Requests GeoJSON geometry in the response (``geometry='geojson'``).
    - Uses ``nonlocal`` counter for progress tracking, so the returned
      function is **not thread-safe**.
    - FIXME: legacy

    Examples
    --------
    >>> geolocator = Nominatim(user_agent='my_app')
    >>> geocode_fn = geocode_it(geolocator, cnt=100, verbose=True)
    >>> result = geocode_fn("123 Main St, Los Angeles, CA")
    1 of 100 (1.0%): Geocoded 123 Main St, Los Angeles, CA -> ...
    """
    ctr=0
    def _geocode_it(x):
        nonlocal ctr
        ctr += 1
        l = geolocator_a.geocode(x, geometry='geojson')
        if verbose:
            print(f"{ctr} of {cnt} ({ctr/cnt*100:.1f}): Geocoded {x} -> {l}")
        return l
    return _geocode_it

# helper from kagi AI
# Extract geometry from GeoJSON feature and create a GeoSeries
def extract_geometry(geojson_response):
    """Extract a Shapely geometry from a geopy GeoJSON response.

    Parameters
    ----------
    geojson_response : geopy.Location or None
        Geopy response object with ``raw['geojson']`` attribute containing
        a GeoJSON geometry.

    Returns
    -------
    shapely.geometry.base.BaseGeometry or None
        Shapely geometry converted from the GeoJSON feature, or ``None``
        if the input is ``None``.

    See Also
    --------
    geocode_it : Produces the geopy responses this function processes.
    """
    if geojson_response is None:
        return None
    # The GeoJSON geometry is in the 'geojson' key, under 'features' list
    feature = geojson_response.raw['geojson']
    return shape(feature)  # Convert GeoJSON geometry to Shapely object


# from https://iifx.dev/en/articles/457774763/mastering-distance-calculations-in-geopandas-for-openstreetmap-data
from geopy.distance import geodesic
def calculate_geodesic_length(line_geometry):
    """Calculate the geodesic length of a LineString in meters.

    Computes the total length by summing geodesic distances between
    consecutive coordinate pairs using :func:`geopy.distance.geodesic`.

    Parameters
    ----------
    line_geometry : shapely.geometry.LineString
        LineString geometry in WGS84 (lon/lat) coordinates.

    Returns
    -------
    float or None
        Total geodesic length in meters, or ``None`` if the input is
        not a LineString.

    Notes
    -----
    - Coordinate order is swapped from Shapely (lon, lat) to geopy
      (lat, lon) convention.
    - This is the non-vectorized version; see
      :func:`calculate_geodesic_length_optimized` for better performance.

    See Also
    --------
    calculate_geodesic_length_optimized : Vectorized version using pyproj.
    """
    if not isinstance(line_geometry, shapely.geometry.LineString):
        return None # Or raise an error for non-LineString geometries

    total_length_m = 0
    coords = list(line_geometry.coords)
    for i in range(len(coords) - 1):
        point1 = (coords[i][1], coords[i][0]) # Geopy expects (latitude, longitude)
        point2 = (coords[i+1][1], coords[i+1][0])
        total_length_m += geodesic(point1, point2).m
    return total_length_m

import pyproj
import numpy as np
# Kagi AI suggestions to improve calculate_geodesic_length for speed
def calculate_geodesic_length_optimized(geoseries):
    """Calculate geodesic lengths for a GeoSeries of LineStrings using vectorized pyproj.

    Uses :class:`pyproj.Geod` for vectorized inverse geodesic computations,
    significantly faster than the point-by-point approach in
    :func:`calculate_geodesic_length`.

    Parameters
    ----------
    geoseries : GeoSeries
        GeoSeries of LineString geometries in WGS84 (``EPSG:4326``).

    Returns
    -------
    list of float or None
        Geodesic length in meters for each geometry. Returns ``0.0`` for
        non-LineString or single-point geometries, ``None`` for null
        geometries.

    Raises
    ------
    ValueError
        If the GeoSeries CRS is not WGS84 (``EPSG:4326``).

    Notes
    -----
    - Uses ``pyproj.Geod(ellps='WGS84')`` for the WGS84 ellipsoid.
    - Segment distances are computed in a single vectorized call per
      LineString via ``geod.inv()``.

    See Also
    --------
    calculate_geodesic_length : Non-vectorized version using geopy.
    """
    if geoseries.crs is None or geoseries.crs.to_epsg() != 4326:
        raise ValueError("GeoSeries must be in WGS84 (EPSG:4326)")

    geod = pyproj.Geod(ellps='WGS84')
    lengths = []

    for geom in geoseries.geometry:
        if not isinstance(geom, shapely.geometry.LineString) or len(geom.coords) < 2:
            lengths.append(0.0 if geom is not None else None)
            continue

        coords = np.array(geom.coords)
        lons = coords[:-1, 0]
        lats = coords[:-1, 1]
        lons_next = coords[1:, 0]
        lats_next = coords[1:, 1]

        # Vectorized calculation of all segments
        _, _, distances = geod.inv(lons, lats, lons_next, lats_next)
        lengths.append(float(np.sum(distances)))

    return lengths

import requests
import json
from itertools import chain

def bls_data():
    """Fetch Bureau of Labor Statistics (BLS) employment data for waste/remediation industries.

    Queries the BLS public API for State and Area Employment series
    (``SM`` dataset) filtered to waste management and remediation
    industries, across all US states. Results are grouped in batches
    of 50 series per API call with 10-second delays to respect rate
    limits.

    Returns
    -------
    DataFrame
        BLS time series data with employment values, series IDs, and
        area metadata joined from Census state boundaries.

    Notes
    -----
    - BLS API key is hardcoded (``2dff1620e7da4da09c3237d3ec851185``).
    - Industry filter: names matching ``'.*Waste.*Remediation'`` excluding
      ``'Administrative'``.
    - API calls are batched in groups of 50 series with 10-second delays.
    - Series IDs are constructed as ``SMU{geoid}{area_code}{industry_code}01``.
    - Area codes are mapped to Census GEOIDs via state FIPS codes.

    References
    ----------
    - BLS API documentation: https://www.bls.gov/developers/api_python.htm
    - BLS series format: https://www.bls.gov/help/hlpforma.htm#SM
    """
    bls_key=settings['bls_connection'].api_key
    ind=pd.read_csv('data/BLS/sm.industry.tsv',delimiter=r'\t',engine='python')
    area=pd.read_csv('data/BLS/sm.area.tsv',delimiter=r'\t',engine='python')
    datatype=pd.read_csv('data/BLS/sm.data.tsv',delimiter=r'\t',engine='python')
    st=pyg.states(year=2021,cache=True).clean_names()
    all_areas=(
        area.assign(st=lambda x: case_when(x.area_name.str.match(r'.*, \w{2}'),
                                           x.area_name.str.replace(r'.*,\s(\w{2})',r'\1',regex=True))
                                           )
        .filt(lambda x: x.st.notna())
        .merge(st[['stusps','geoid']],left_on=['st'],right_on=['stusps'])
        .assign(full_area=lambda x: x.geoid + x.area_code.astype(str))
    )

    group_size=50
    codes = (
        all_areas
        .join(ind.filt(lambda x: x.industry_name.str.match('.*Waste.*Remediation') 
                                & ~x.industry_name.str.match('Administrative')),how='cross')
        .assign(series=lambda x: 'SMU' + x.full_area.astype(str) + x.industry_code.astype(str) + '01')
        # .filt(lambda x: x.st.str.match('CA'))
        .reset_index(drop=True) # reset the index to get counts
        .assign(
            group=lambda x: x.index // group_size
        )

    )

    headers = {'Content-type': 'application/json'}

    import time
    all_series=[]
    for g, df in codes.groupby('group'):
        data = json.dumps({"seriesid": list(df.series),"startyear":"2010", "endyear":"2024"})
        p = requests.post(f'https://api.bls.gov/publicAPI/v2/timeseries/data/?registrationkey={bls_key}', data=data, headers=headers)
        json_data = json.loads(p.text)
        all_series += [
            pd.DataFrame(series['data'])
            .assign(seriesid = series['seriesID'])
            .groupby(['seriesid','year']).agg({'value':'max','footnotes': lambda x: list(chain.from_iterable(x)) })
            .reset_index()
            for series in json_data['Results']['series'] if len(series['data']) > 0
        ]
        time.sleep(10)
    all_seriesx=pd.concat(all_series).assign(
        full_area=lambda x: x.seriesid.str.replace(r'(\w{2})(\w)(\d{2})(\d{5})(\d{8})(\d{2})',r'\3\4',regex=True)
    ).join(all_areas.set_index('full_area'),on='full_area',how='left')


@pa.check_types
def read_unify_geo(unify_geo_file:Path) -> DataFrame[GeocodedEndpointSchema]:
    """Load Unify facility geocodes from the Unify Lat/Long spreadsheet.

    Reads the Unify facility coordinate file and creates Point geometries
    using a priority cascade of coordinate sources:

    1. JDs coordinates (``jdslat``/``jdslong``)
    2. RDRS coordinates (``rdrslat``/``rdrslong``)
    3. SWIS coordinates (``swislat``/``swislong``)

    Parameters
    ----------
    None (reads from fixed data source).

    Returns
    -------
    DataFrame[GeocodedEndpointSchema]
        Unify facility geocodes with:

        - ``id`` — Unify key field
        - ``geometry`` — Point from highest-priority coordinate source
        - ``geometry_src`` — ``'UnifyGeo'``
        - ``aggregation`` — ``EndpointAggregation.LOCATION``
        - ``facility_type`` — ``'RDRS_UNKNOWN'``

    Notes
    -----
    - Duplicate IDs are resolved by ``groupby().first()``.
    - Facilities with no coordinates at any priority level are filtered out.
    - CRS is ``EPSG:4326``.
    - FIXME: reconcile output types

    See Also
    --------
    read_rdrs_manual_geocodes : Manual RDRS geocodes (higher priority).
    """
    ret = (
        gpd.GeoDataFrame(
            # pd.read_excel(cpath('gis_data','UnifyLatLong03182024.xlsx'))
            pd.read_excel(unify_geo_file)
            .clean_names()
            .assign(
                geometry=lambda x: np.where(
                    ~x.jdslat.isna(),
                    x.apply(lambda row: shapely.geometry.Point(row.jdslong,row.jdslat),axis=1),
                    np.where(
                        ~x.rdrslat.isna(),
                        x.apply(lambda row: shapely.geometry.Point(row.rdrslong,row.rdrslat),axis=1),
                        np.where(
                            ~x.swislat.isna(),
                            x.apply(lambda row: shapely.geometry.Point(row.swislong,row.swislat),axis=1),
                            None
                        )
                    )
                ),
                geometry_src='UnifyGeo',
                aggregation=EndpointAggregation.LOCATION.value,
                facility_type='RDRS_UNKNOWN',
                facility_flags=''
            )
            ,geometry='geometry')
        .rename(columns={'key':'id'}) # to match GeocodedId schema
        # FIXME: figure out why there are duplicates
        .groupby('id').first()
        .filt(lambda x: ~x.geometry.isna()) # FIXME: temporary filter to remove NAs
        .set_crs(crs_ll)
    )
    return ret


@cachier(wait_for_calc_timeout=5)
@pa.check_types
def read_rdrs_manual_geocodes() -> DataFrame[LayerEndpointSchema]:
    """Load manually curated RDRS facility geocodes.

    Reads and concatenates two CSV files (internal RDRS geocodes and
    external geocodes), creates Point geometries from lon/lat, assigns
    facility types based on boolean flags, and backfills region metadata.

    Parameters
    ----------
    None (reads from fixed data sources).

    Returns
    -------
    DataFrame[LayerEndpointSchema]
        Manual RDRS geocodes with:

        - ``id`` — ``rdrs_id`` from source data
        - ``geometry`` — Point from lon/lat
        - ``geometry_src`` — ``'Manual RDRS'``
        - ``facility_type`` — inferred from flags (SHREDDER, CLASS1 LANDFILL,
          E-WASTE LANDFILL, CTMSR LANDFILL, MRF) or ``reporting_activity2``,
          falling back to ``'RDRS_UNKNOWN'``
        - ``facility_flags`` — ``'PROBABLY_OFFICE'`` if flagged
        - Full region metadata from :func:`backfill_regions`

    Notes
    -----
    - External geocodes have ``n2`` backfilled from ``addr`` when missing.
    - Facility type assignment uses a chain of ``.mask()`` calls, checking
      boolean flags in priority order.
    - Results are cached via :mod:`cachier`.
    - FIXME: reconcile output types

    See Also
    --------
    read_unify_geo : Unify facility geocodes (lower priority).
    backfill_regions : County/state/country assignment.
    """
    logger.info("Reading manual RDRS geocodes")
    
    def replace_true_with_col_name(row):
        return '|'.join(row.index[row == True].tolist())
    
    mgf=settings.get('model').rdrs_manual_geocodes_file.format(**settings.get('model').to_dict())
    xgf=settings.get('model').rdrs_external_geocodes_file.format(**settings.get('model').to_dict())

    ret = (
        gpd.GeoDataFrame(
            pd.concat(
                [
                    pd.read_csv(mgf).clean_names(),  # FIXME: use config
                    pd.read_csv(xgf).clean_names() # FIXME: use config
                    .clean_names()
                    .assign(
                        n2=lambda x: x.n2.fillna(x.addr)
                    )
                ],
                ignore_index=True
            )
            
            .assign(
                geometry=lambda x: x.apply(lambda row: shapely.geometry.Point(row.lon,row.lat),axis=1),
                geometry_src='Manual RDRS',
                aggregation=EndpointAggregation.LOCATION.value,
                facility_type=None,
                facility_flags=lambda x: x[x.filter(regex='_flag$')].apply(replace_true_with_col_name, axis=1)
            )
            .assign(
                facility_type=lambda x:(x.facility_type
                    .mask(x.facility_flags.str.contains('probably_office', case=False, na=False), 'OFFICE')
                    .mask(x.facility_flags.str.contains('shredder', case=False, na=False), 'SHREDDER')
                    .mask(x.facility_flags.str.contains('landfill', case=False, na=False), 'LANDFILL')
                    .mask(x.facility_flags.str.contains('mrf', case=False, na=False), 'MRF')
                    .fillna(x.reporting_activity2)
                    .fillna('RDRS_UNKNOWN')
                )
            )
            ,geometry='geometry')
        .rename(columns={'rdrs_id':'id'}) # to match GeocodedId schema
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        .set_geometry('geometry')
        .set_crs(crs_ll)
        .add_column_if_missing('geometry_agg',None)
        .pipe(backfill_regions)
    )
    return ret[schema_to_cols(LayerEndpointSchema)]
        
from pygris.data import get_census
# @cachier(wait_for_calc_timeout=5)
def get_census_tract_population(state='CA',year=2021):
    """Load census tract geometries merged with population data.

    Parameters
    ----------
    state : str, default 'CA'
        State abbreviation for tract selection.

    Returns
    -------
    GeoDataFrame
        Census tract polygons with population counts from ACS.

    See Also
    --------
    get_census_tract_population : Population data source.
    calculate_weighted_centroids : Uses this data for weighted centroid computation.
    """
    statecode=f'state:{pyg.states(cache=True,year=year).filt(lambda x: (x.STUSPS==state) | (x.NAME==state)).iloc[0]['STATEFP']}'
    assert(statecode)
    return get_census(
            dataset = f"acs/acs5", 
            year = year,
            variables = "B01003_001E", # Total Population
            params = {
                "for": "tract:*",
                "in": statecode,  # california
                "key": settings['census_connection']['api_key']
            },
            return_geoid = True,
        ).assign(
            B01003_001E = lambda x: x.B01003_001E.astype(int)
        ).clean_names()
        
def get_census_tracts_with_population(state=fips_ca):
    """Calculate population-weighted centroids for groups of census tracts.

    For each group (default: county), computes the population-weighted
    average of tract representative point coordinates, producing a
    centroid that reflects where people actually live rather than the
    geographic center.

    Parameters
    ----------
    df : GeoDataFrame
        Census tract data with ``b01003_001e`` (population) and
        ``geometry`` columns.
    groupby : str, default 'countyfp'
        Column to group tracts by for centroid calculation.

    Returns
    -------
    GeoDataFrame
        One row per group with:

        - ``geometry`` — population-weighted centroid Point
        - ``geometry_agg`` — original group boundary polygon
        - ``total_population`` — sum of tract populations in the group

    Notes
    -----
    - Representative points are computed in CA Albers projection
      (``crs_ca``) then converted back to WGS84.
    - The original group boundary is preserved as ``geometry_agg``.

    See Also
    --------
    get_census_tracts_with_population : Source data for this computation.
    get_county_endpoints : Uses these weighted centroids for county endpoints.
    """
    st_tracts=(
        pyg.tracts(state=state,cache=True).clean_names()
        .merge(get_census_tract_population(state=state),on='geoid')
    )
    return st_tracts
    

# Calculate tract centroids and extract coordinates
def calculate_weighted_centroids(df,groupby='countyfp'):
    return (
        df.to_crs(crs_ll)
        # Calculate centroid coordinates for each tract
        .assign(
            center_x=lambda x: x.to_crs(crs_ca).geometry.representative_point().to_crs(crs_ll).x,
            center_y=lambda x: x.to_crs(crs_ca).geometry.representative_point().to_crs(crs_ll).y
        )
        # Group by county and calculate weighted centroids
        .groupby('countyfp')
        .apply(lambda group: pd.Series({
            'pop_weighted_lon': np.average(group['center_x'], weights=group['b01003_001e']),
            'pop_weighted_lat': np.average(group['center_y'], weights=group['b01003_001e']),
            'total_population': group['b01003_001e'].sum()
        }))
        .reset_index()
        # Join back with original data to get geometries
        .merge(
            df[['countyfp', 'geometry']].drop_duplicates('countyfp'), 
            on='countyfp', 
            how='left'
        )
        # Rename original geometry and set new centroid as geometry
        .rename(columns={'geometry': 'geometry_agg'})
        .assign(
            geometry=lambda x: gpd.points_from_xy(x.pop_weighted_lon, x.pop_weighted_lat)
        )
        .pipe(gpd.GeoDataFrame, geometry='geometry', crs=crs_ll)
        .drop(columns=['pop_weighted_lon','pop_weighted_lat'])
    )

world_name_map = {
    "Lao People's Democratic Republic": 'Laos',
    "Republic of Korea": 'Korea, South',
    "U.K. of Great Britain and Northern Ireland": 'United Kingdom'
}

country_endpoint_prefix="CTRY_"  # used to define ids
state_endpoint_prefix="STATE_"  # used to define ids
unknown_endpoint_prefix='UNK_'  # used to define ids
county_endpoint_prefix="CTY_"  # used to define ids

@cachier(wait_for_calc_timeout=5)
def get_world_endpoints() -> DataFrame[LayerEndpointSchema]:
    """Build country-level endpoint entities from world boundary data.

    Creates one endpoint per country with a representative point geometry
    and the full country boundary as ``geometry_agg``.

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        Country endpoints with:

        - ``id`` — ``CTRY_`` + ISO3 code
        - ``n1``, ``n2`` — country name
        - ``county`` — ``'Out of USA'``
        - ``state`` — ``'Out of USA'``
        - ``country`` — common country name (full names replaced via
          ``world_name_map``)
        - ``facility_type`` — ``'COUNTRY'``
        - ``aggregation`` — ``EndpointAggregation.COUNTRY``

    Notes
    -----
    - Results are cached via :mod:`cachier`.
    - Countries with full official names (e.g., ``'Lao People's Democratic
      Republic'``) are replaced with common names via ``world_name_map``.

    See Also
    --------
    get_world_boundary : Source boundary data.
    get_state_endpoints : State-level endpoints.
    get_county_endpoints : County-level endpoints.
    """
    return (
        get_world_boundary()
        .reset_index()
        .assign(
            id = lambda x: country_endpoint_prefix+x.iso3.fillna('X'+x['index'].astype(str).apply(lambda x: x.zfill(3))),
            n1 = lambda x: x.country,
            n2 = lambda x: x.country,
            county = 'Out of USA',
            state  = 'Out of USA',
            
            # some countries are using full rather than common names
            country = lambda x: x.country.replace(world_name_map),
            facility_type = "COUNTRY",
            facility_flags = '',
            aggregation=EndpointAggregation.COUNTRY.value
        )
        # the iso3 is duplicated for a few countries:
        .assign(id=lambda x: x.id + x.groupby('id').cumcount().map(lambda x: f'_{x+1}' if x > 0 else ''))
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        [schema_to_cols(LayerEndpointSchema)]
    )
    
@cachier(wait_for_calc_timeout=5)
def get_state_endpoints() -> DataFrame[LayerEndpointSchema]:
    """Build US state-level endpoint entities from Census state boundaries.

    Creates one endpoint per US state with a representative point geometry
    and the full state boundary as ``geometry_agg``.

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        State endpoints with:

        - ``id`` — ``STATE_`` + state USPS abbreviation
        - ``n1`` — state name
        - ``n2`` — ``'United States'``
        - ``county`` — ``'Out of California'``
        - ``state`` — state name
        - ``country`` — ``'United States'``
        - ``facility_type`` — ``'STATE'``
        - ``aggregation`` — ``EndpointAggregation.STATE``

    Notes
    -----
    - Results are cached via :mod:`cachier`.

    See Also
    --------
    get_us_state : Source boundary data.
    get_world_endpoints : Country-level endpoints.
    get_county_endpoints : County-level endpoints.
    """
    return (
        get_us_state()
        .reset_index()
        .assign(
            id = lambda x: state_endpoint_prefix+x.stusps.fillna('X'+x['index'].astype(str).apply(lambda x: x.zfill(3))),
            county = 'Out of California',
            state  = lambda x: x.state,
            country = 'United States',
            n1 = lambda x: x.state,
            n2 = lambda x: x.country,
            facility_type = "STATE",
            facility_flags = '',
            aggregation=EndpointAggregation.STATE.value
        )
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        [schema_to_cols(LayerEndpointSchema)]
    )
    
@cachier(wait_for_calc_timeout=5)
def get_county_endpoints() -> DataFrame[LayerEndpointSchema]:
    """Build California county-level endpoint entities with population-weighted centroids.

    Creates one endpoint per California county with a population-weighted
    centroid as the active geometry and the full county boundary as
    ``geometry_agg``.

    Returns
    -------
    DataFrame[FullGeocodedEndpointSchema]
        County endpoints with:

        - ``id`` — ``CTY_`` + GEOID
        - ``n1`` — county name
        - ``n2`` — state name
        - ``county`` — county name
        - ``state`` — state name
        - ``country`` — ``'United States'``
        - ``geometry`` — population-weighted centroid Point
        - ``geometry_agg`` — county boundary polygon
        - ``facility_type`` — ``'COUNTY'``
        - ``aggregation`` — ``EndpointAggregation.COUNTY``

    Notes
    -----
    - Population-weighted centroids are computed by
      :func:`calculate_weighted_centroids` using ACS tract population data.
    - Results are cached via :mod:`cachier`.

    See Also
    --------
    get_ca_cty : Source county boundary data.
    calculate_weighted_centroids : Population-weighted centroid computation.
    get_state_endpoints : State-level endpoints.
    get_world_endpoints : Country-level endpoints.
    """
    return (
        get_ca_cty()
        .rename(columns={'geometry':'geometry_geo_centroid'})
        .merge(get_census_tracts_with_population(state='CA')[['countyfp','geometry','b01003_001e']]
               .pipe(calculate_weighted_centroids)
               .drop(columns=['geometry_agg']),
               on='countyfp')
        .reset_index()
        # at this point geometry_agg is the county boundary and geometry is the pop-weighted centroid (by tract)
        .assign(
            id = lambda x: county_endpoint_prefix+x.geoid,
            county = lambda x: x.county,
            state  = lambda x: x.state_name,
            country = 'United States',
            n1 = lambda x: x.county,
            n2 = lambda x: x.state,
            facility_type = "COUNTY",
            facility_flags = '',
            aggregation=EndpointAggregation.COUNTY.value
        )
        .set_flags(allows_duplicate_labels=False)
        .set_index('id')
        [schema_to_cols(LayerEndpointSchema)]
    )
    


# Kagi AI
import warnings
def gpd_concat(objs, *args, crs=None, **kwargs):
    """Concatenate GeoDataFrames while preserving CRS information.

    Wrapper around :func:`pd.concat` that infers and enforces a consistent
    CRS on the result, suppressing the spurious "CRS not set" warning
    that ``pd.concat`` produces when concatenating GeoDataFrames.

    Parameters
    ----------
    objs : list of GeoDataFrame
        GeoDataFrames to concatenate.
    crs : str or pyproj.CRS, optional
        Explicit CRS for the result. If ``None``, inferred from inputs.
    *args, **kwargs
        Additional arguments passed to :func:`pd.concat`.

    Returns
    -------
    GeoDataFrame or DataFrame
        Concatenated result with CRS set. Returns a plain DataFrame if
        no GeoDataFrames are in the input.

    Raises
    ------
    ValueError
        If input GeoDataFrames have conflicting CRS values and no
        explicit ``crs`` is provided.

    Notes
    -----
    - If all input GeoDataFrames share the same CRS, that CRS is forced
      onto the result.
    - The "CRS not set for some" warning from pandas is suppressed during
      concatenation.

    Examples
    --------
    >>> gdf1 = gpd.GeoDataFrame(..., crs="EPSG:4326")
    >>> gdf2 = gpd.GeoDataFrame(..., crs="EPSG:4326")
    >>> result = gpd_concat([gdf1, gdf2])  # CRS preserved
    """
    geodataframes = [df for df in objs if isinstance(df, gpd.GeoDataFrame)]

    if not geodataframes:
        return pd.concat(objs, *args, **kwargs)

    # Infer CRS if not explicitly provided
    if crs is None:
        crs_set = {df.crs for df in geodataframes if df.crs is not None}
        if len(crs_set) == 1:
            crs = crs_set.pop()
        elif len(crs_set) > 1:
            raise ValueError(
                f"Conflicting CRS in inputs: {crs_set}. "
                f"Pass an explicit crs= to override."
            )
        # else: all None — leave crs as None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CRS not set for some")
        result = pd.concat(objs, *args, **kwargs)

    if crs is not None and isinstance(result, gpd.GeoDataFrame):
        result = result.set_geometry(result.geometry.name, crs=crs)

    return result


def to_linestring(geom):
    """Ensure geometry is a single LineString."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        return LineString([c for line in geom.geoms for c in line.coords])
    raise TypeError(f"Unexpected geometry type: {type(geom)}")


def merge_lines(geom1, geom2):
    ls1 = to_linestring(geom1)
    ls2 = to_linestring(geom2)
    merged = linemerge([ls1, ls2])
    return to_linestring(merged)



## KAGI AI
from shapely.geometry import LineString
import numpy as np

def densify_linestring(geom, num_points=50):
    """Add interpolated points along a LineString to reduce projection distortion."""
    if geom.geom_type != 'LineString':
        return geom

    coords = list(geom.coords)
    if len(coords) < 2:
        return geom

    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i-1][0]
        dy = coords[i][1] - coords[i-1][1]
        distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))

    total = distances[-1]
    if total == 0:
        return geom

    # Interpolate evenly spaced points
    new_coords = []
    for d in np.linspace(0, total, num_points):
        # Find segment
        for i in range(1, len(distances)):
            if distances[i] >= d:
                seg_frac = (d - distances[i-1]) / (distances[i] - distances[i-1])
                x = coords[i-1][0] + seg_frac * (coords[i][0] - coords[i-1][0])
                y = coords[i-1][1] + seg_frac * (coords[i][1] - coords[i-1][1])
                new_coords.append((x, y))
                break

    return LineString(new_coords)
