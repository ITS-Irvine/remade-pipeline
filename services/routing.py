import functools
import itertools
from pathlib import Path
import pickle
from typing import Tuple
from IPython.display import display
import numpy as np
import osrm
import pandas as pd
import polyline
import pygris as pyg
from requests.exceptions import ConnectionError
import routingpy
from routingpy.routers import options
import searoute as sr
import logging
import hashlib

import shapely
import geopandas as gpd

import logging

from shapely.ops import linemerge

# from services.geocode import crs_ca, crs_ll, crs_usa
from core.model_types import gpd, logger, np
from layers.base import od_pair_group
from core.units import create_unit_assignment, get_pint_units, transform_units
from services.geocode import crs_ca, crs_ll, crs_usa, logger
logger = logging.getLogger(__name__)
from core.common import cpath, shapely_point_to_osrm_point
from core.units import ureg
from core.model_types import *
import pandera.pandas as pa
from cachier import cachier
from cachier.config import CacheEntry
from config.settings import settings

@pa.check_types
def has_maritime_leg(df:DataFrame, #[ODPairSchemaWithDetails],
                     endpoint_prefix:str='') -> pa.typing.Series[bool]:
    """Determine whether each row's endpoint requires a maritime shipping leg.

    A maritime leg is assumed when the endpoint is outside the contiguous US
    (i.e., the country is not the US, Mexico, or Canada) **or** the endpoint
    is in Hawaii, provided the row has a valid geometry for that endpoint.

    Parameters
    ----------
    df : DataFrame
        Origin-destination pair data with per-endpoint country, state, and
        geometry columns.  Expected columns vary based on *endpoint_prefix*:

        - ``{prefix}country`` — ISO or full country name
        - ``{prefix}state`` — US state name or abbreviation
        - ``geometry{suffix}`` — point/polygon geometry for the endpoint

    endpoint_prefix : str, default ''
        Column-name prefix identifying which endpoint to inspect:

        - ``'o_'`` → origin columns (uses ``geometry_orig``)
        - ``'d_'`` → destination columns (uses ``geometry_dest``)
        - ``''`` → unprefixed columns (uses ``geometry``)

    Returns
    -------
    Series[bool]
        True for rows where the endpoint is assumed to require a maritime
        leg.

    Notes
    -----
    - **Country matching** is case-insensitive and accepts ``"UNITED STATES"``,
      ``"US"``, ``"MEXICO"``, and ``"CANADA"`` as non-maritime.  All other
      countries are treated as maritime destinations.
    - **Hawaii** (``"Hawaii"`` or ``"HI"``) is treated as maritime regardless
      of country.
    - Missing ``country`` values default to ``"US"``; missing ``state``
      values default to ``"CA"`` (contiguous US), so neither NaN country nor
      NaN state alone triggers a maritime classification.
    - Rows with ``NaN`` geometry are excluded (``False``), even if the
      country/state criteria are met.

    Known limitations
    -----------------
    - Alaska, Puerto Rico, and Guam should also be treated as maritime but
      are not yet included in the state check.
    - International RDRS entities are not accounted for and may be
      misclassified.
    - Some rows have ``NaN`` geometries that should not exist; these are
      silently excluded rather than flagged as errors.
    - The geometry-suffix mapping (``_dest`` / ``_orig`` / ``''``) is a
      heuristic tied to the column naming convention.

    Examples
    --------
    >>> has_maritime_leg(df, endpoint_prefix='d_')
    0     True   # destination in Japan
    1     False  # destination in California
    2     True   # destination in Hawaii
    3     False  # destination in Mexico
    4     False  # NaN geometry (excluded)
    Name: geometry_dest, dtype: bool
    """
    # FIXME: hack to determine which geometry to use
    if endpoint_prefix==d_prefix:
        geom_suffix='_dest'
    elif endpoint_prefix==d_prefix:
        geom_suffix='_orig'
    else:
        geom_suffix=''
        
    # assume any WORLD destinations not Mexico or Canada have a maritime leg
    # FIXME: should also check for Intl RDRS entities
    dest_is_maritime = (
        # maritime if country is not us, mexico or canada
        (~df[f'{endpoint_prefix}country'].fillna('US').str.upper().str.match(r'UNITED STATES|US$|MEXICO|CANADA')
         
         # also if state is hawaii (FIXME: should include PR, guam, alaska (probably))
         | df[f'{endpoint_prefix}state'].fillna('CA').str.upper().str.match('Hawaii|HI$|GUAM|PUERTO RICO|ALASKA'))
        
        & ~df[f'geometry{geom_suffix}'].isna()  # FIXME: shouldn't really be any that are NA, but some are
    )
    return dest_is_maritime


class MaritimeLegInput(pa.DataFrameModel):
    d_country:  Series[str] = pa.Field(nullable=False)
    d_state:    Series[str] = pa.Field(nullable=False)
    d_geometry: GeoSeries   = pa.Field(nullable=False)
        
@pa.check_types
def has_maritime_leg_v2(df:DataFrame[MaritimeLegInput],
                     endpoint_prefix:str='') -> Series[bool]:
    """Determine whether each row's endpoint requires a maritime shipping leg.

    A maritime leg is assumed when the endpoint is outside the contiguous US
    (i.e., the country is not the US, Mexico, or Canada) **or** the endpoint
    is in Hawaii, provided the row has a valid geometry for that endpoint.

    Parameters
    ----------
    df : DataFrame
        Origin-destination pair data with per-endpoint country, state, and
        geometry columns.  Expected columns vary based on *endpoint_prefix*:

        - ``{prefix}country`` — ISO or full country name
        - ``{prefix}state`` — US state name or abbreviation
        - ``geometry{suffix}`` — point/polygon geometry for the endpoint

    endpoint_prefix : str, default ''
        Column-name prefix identifying which endpoint to inspect:

        - ``'o_'`` → origin columns (uses ``geometry_orig``)
        - ``'d_'`` → destination columns (uses ``geometry_dest``)
        - ``''`` → unprefixed columns (uses ``geometry``)

    Returns
    -------
    Series[bool]
        True for rows where the endpoint is assumed to require a maritime
        leg.

    Notes
    -----
    - **Country matching** is case-insensitive and accepts ``"UNITED STATES"``,
      ``"US"``, ``"MEXICO"``, and ``"CANADA"`` as non-maritime.  All other
      countries are treated as maritime destinations.
    - **Hawaii** (``"Hawaii"`` or ``"HI"``) is treated as maritime regardless
      of country.
    - Missing ``country`` values default to ``"US"``; missing ``state``
      values default to ``"CA"`` (contiguous US), so neither NaN country nor
      NaN state alone triggers a maritime classification.
    - Rows with ``NaN`` geometry are excluded (``False``), even if the
      country/state criteria are met.

    Known limitations
    -----------------
    - Alaska, Puerto Rico, and Guam should also be treated as maritime but
      are not yet included in the state check.
    - International RDRS entities are not accounted for and may be
      misclassified.
    - Some rows have ``NaN`` geometries that should not exist; these are
      silently excluded rather than flagged as errors.
    - The geometry-suffix mapping (``_dest`` / ``_orig`` / ``''``) is a
      heuristic tied to the column naming convention.

    Examples
    --------
    >>> has_maritime_leg(df, endpoint_prefix='d_')
    0     True   # destination in Japan
    1     False  # destination in California
    2     True   # destination in Hawaii
    3     False  # destination in Mexico
    4     False  # NaN geometry (excluded)
    Name: geometry_dest, dtype: bool
    """
    # assume any WORLD destinations not Mexico or Canada have a maritime leg
    # FIXME: should also check for Intl RDRS entities
    dest_is_maritime = (
        # maritime if country is not us, mexico or canada
        (
            ~df[f'd_country'].fillna('US').str.upper().str.match(r'UNITED STATES|US$|MEXICO|CANADA')
         
            # also if state is hawaii (FIXME: should include PR, guam, alaska (probably))
            | df[f'd_state'].fillna('CA').str.upper().str.match('Hawaii|HI$|GUAM|PUERTO RICO|ALASKA')
        )
        
        #& ~df[f'd_geometry'].isna()  # FIXME: shouldn't really be any that are NA, but some are
    )
    return dest_is_maritime

def _generate_searoutes_param_hasher(args, kwargs):
    """Compute a cache key for :func:`generate_searoutes` based on OD flow and entity data.

    Hashes only the columns that influence searoute computation (origin/destination
    IDs and geometries for flows; all columns for entities), producing a stable
    SHA-256 tuple that :mod:`cachier` uses to detect parameter changes.

    Parameters
    ----------
    args : tuple
        Positional arguments (unused).
    kwargs : dict
        Keyword arguments. Must contain:

        - ``od_flows_x`` — the OD flows DataFrame
        - ``all_ent`` — the geocoded entities DataFrame

    Returns
    -------
    tuple[str, str]
        Two-element tuple of SHA-256 hex digests: (flows_hash, entities_hash).

    See Also
    --------
    cachier : Caching library that consumes this hasher.
    """
    od_flows_x: DataFrame = kwargs['od_flows_x']
    all_ent: DataFrame[RegionSchema] = kwargs['all_ent']
    
    df_hash1 = hashlib.sha256(
        pd.util.hash_pandas_object(
            od_flows_x[[ # we're only interested in hashing on these fields
                'o_id','d_id',
                'geometry_orig',
                'geometry_dest'            ]]
        ).values.tobytes()
    ).hexdigest()
    df_hash2 = hashlib.sha256(
        pd.util.hash_pandas_object(
            all_ent
        ).values.tobytes()
    ).hexdigest()
    return (df_hash1, df_hash2)


@functools.cache
def route_by_endpoints(p1:shapely.geometry.Point,p2:shapely.geometry.Point) -> dict:
    """Compute an OSRM driving route between two points with automatic server fallback.

    Selects the appropriate OSRM instance based on which US states the endpoints
    fall within, then falls back through a priority list of servers if the
    preferred one fails:

    1. **local_ca** — Local OSRM instance covering California (port 5000).
    2. **local_ne** — Local OSRM instance covering the Northeast US (port 5001).
    3. **global** — Public OSM routing server (slow, rate-limited).

    Results are memoized with :func:`functools.cache` so repeated calls for the
    same point pair return instantly.

    Parameters
    ----------
    p1 : shgeo.Point
        Origin point (lon/lat).
    p2 : shgeo.Point
        Destination point (lon/lat).

    Returns
    -------
    dict
        Raw OSRM directions response including ``routes`` with duration, distance,
        and per-step geometry/annotations.

    Raises
    ------
    Exception
        If all available OSRM configurations fail or no configuration matches
        the endpoint locations.

    Notes
    -----
    - Endpoint states are resolved via a spatial join against US state polygons
      from :mod:`pygris`.
    - On ``RouterApiError`` from the global server, the function retries with
      ``local_ca`` regardless of endpoint location.
    - On ``ConnectionError``, the failed server is removed from the available
      pool and the next server is tried.
    - The global server uses a custom ``user_agent`` header to avoid basic
      bot-blocking.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> result = route_by_endpoints(Point(-118.24, 34.05), Point(-121.89, 37.33))
    >>> result['routes'][0]['distance']  # meters
    """

    pts=gpd.GeoDataFrame({'id':['p1','p2'],'geometry':[p1,p2]},geometry='geometry',crs=crs_ll)
    states = pyg.states(year=2021,cache=True).clean_names().to_crs(crs_ll)
    jj=pts.sjoin(states,predicate='covered_by',how='inner')
    used_states = set(jj.stusps.unique())
    osrm_ca = set(['CA'])
    osrm_ne = set(['NY','PA','CT','MA','RI','ME','NH','VT','NJ'])

    force_local_ca = False
    available = ['local_ca','local_ne','global']
    while True and len(available) > 0:

        def get_osrm_client(section:str):
            ss = settings.get(section)
            if ss is None:
                raise RuntimeError(f"No settings for {section} in get_osrm_client")
            request_url = ss.get('request_url') or None
            if request_url is None:
                raise RuntimeError(f"No request_url for {section} in get_osrm_client")
            rconfig = {}
            rconfig.update({'base_url':    ss.get('base_url')    or None})
            rconfig.update({'user_agent':  ss.get('user_agent')  or None})
            for k in list(rconfig.keys()):
                if rconfig[k] == None:
                    rconfig.pop(k,None)
            config = osrm.RequestConfig(request_url) # unclear if this is needed?
            client = routingpy.OSRM(**rconfig)
            return client

        if force_local_ca or used_states.intersection(osrm_ca) == used_states and 'local_ca' in available:
            # everything in osrm_ca
            client = get_osrm_client('routing.osrm.ca')
            last = 'local_ca'
        elif used_states.intersection(osrm_ne) == used_states and 'local_ne' in available:
            # everything in osrm_ne
            client = get_osrm_client('routing.osrm.ne')
            last = 'local_ne'
        elif 'global' in available:
            client = get_osrm_client('routing.osrm.global')
            last = 'global'
        else:
            logger.error(f"No available OSRM configurations for points {p1} and {p2}. No more fallbacks available.")
            raise Exception(f"No available OSRM configurations for points {p1} and {p2}. No more fallbacks available.")
        options.default_retry_over_query_limit=True

        try:
            res=client.directions(locations=[[p.x,p.y] for p in [p1,p2]],steps=True,annotations=True)
            return res.raw
        except routingpy.exceptions.RouterApiError as e:
            logger.warning(f"Error occurred while fetching route: {e}")
            if last == 'local_ca':
                logger.error(f"Global OSRM failed for points {p1} and {p2}. No more fallbacks available.")
                raise e
            else:
                logger.info(f"Retrying with local_ca OSRM configuration...")
                force_local_ca = True
        except ConnectionError as e:
            logger.warning(f"Connection error occurred while fetching route: {e} using '{last}'. Removing '{last}' from available options and retrying...")
            available.remove(last)
            if available and set(available) == {'global'}:
                logger.warning(f"Only global OSRM configuration available...lookups will be slow and may fail if rate limits are hit.")
    if len(available) == 0:
        logger.error(f"All OSRM configurations failed for points {p1} and {p2}. No more fallbacks available.")
        raise Exception(f"All OSRM configurations failed for points {p1} and {p2}.")


def safe_route_by_endpoints(p1:shapely.geometry.Point,p2:shapely.geometry.Point) -> dict:
    """Route between two endpoints, returning an infinite-distance fallback on failure.

    Wraps :func:`route_by_endpoints` with exception handling so that routing
    errors do not propagate upstream. When routing fails, a synthetic route
    with ``duration=inf`` and ``distance=inf`` is returned, allowing downstream
    code to detect and handle unroutable pairs without crashing.

    Parameters
    ----------
    p1 : shgeo.Point
        Origin point (lon/lat).
    p2 : shgeo.Point
        Destination point (lon/lat).

    Returns
    -------
    dict
        OSRM-style route dict. On success, the full OSRM response with
        ``routes`` key. On failure, ``{'routes': [{'duration': inf, 'distance': inf}]}``.

    See Also
    --------
    route_by_endpoints : Unwrapped routing with OSRM fallback logic.
    """
    try:
        return route_by_endpoints(p1,p2)
    except:
        return {'routes':[{
                    'duration': np.inf,
                    'distance': np.inf,
                }]}

@cachier(wait_for_calc_timeout=5,
         hash_func=_generate_searoutes_param_hasher
        )
@pa.check_types
def generate_searoutes(od_flows_x: DataFrame[ODFlows], 
                       all_ent: DataFrame[LayerEndpointSchema]
                      ) -> DataFrame[ODPairSchemaWithPortAndSearoute]:
                    #    ) -> Tuple[DataFrame[ODPairSchemaWithPortAndSearoute],
                    #               DataFrame[ODPairSchemaWithPortAndSearoute]]: # FIXME:CRITICAL
    """Generate maritime (sea) routes for OD flows pairs with international destinations.

    For each origin–destination pair where the destination requires a maritime
    leg (see :func:`has_maritime_leg`), this function:

    1. Identifies the closest California port to the origin via direct-line distance.
    2. Computes the on-road route from origin to that port using OSRM.
    3. Computes the sea route from the port to the international destination
       using the :mod:`searoute` library.
    4. Returns a DataFrame with port info, sea-route geometry, and distances.

    Results are cached via :mod:`cachier` using a custom parameter hasher
    (:func:`_generate_searoutes_param_hasher`).

    Parameters
    ----------
    od_flows_x : DataFrame[ODFlows]
        OD flow data with origin/destination IDs and geometries.
    all_ent : DataFrame[LayerEndpointSchema]
        All geocoded endpoint entities, including port entities (IDs starting
        with ``PORT``).

    Returns
    -------
    DataFrame[ODPairSchemaWithPortAndSearoute]
        OD pairs augmented with port assignment, on-road distance to port,
        sea-route geometry, and sea distance in km.

    Raises
    ------
    Exception
        If no port entities are found in ``all_ent``.

    Notes
    -----
    - Port selection minimizes the on-road distance from origin to port.
    - If no maritime flows exist, returns an empty DataFrame conforming to
      ``ODPairSchemaWithPortAndSearoute``.
    - The on-road route to the port is computed via OSRM, which introduces
      an external dependency that could be removed by reworking the logic
      to use the main on-road routing pipeline.

    See Also
    --------
    has_maritime_leg : Determines which destinations need maritime routing.
    get_maritime_routes : Joins maritime route data onto the full OD flows.
    """
    port_dests = None
    unique_ods = od_flows_x[od_pair_group].drop_duplicates()
    port_ents  = all_ent.filt(lambda x: x.index.str.match('^PORT')).reset_index().add_col_suffix('_port_info')
    
    logger.info(f"Generating searoutes for flows going outside North America")
    if len(port_ents)==0:
        raise Exception("No ports in entities passed to searoutes")
    port_dests_1 = (
        # for all flows, we grab the o-d pair
        unique_ods
        # we do an inner join (filter) for any destinations that have a maritime leg
        .join(
            all_ent
            .filt(lambda x: has_maritime_leg(x))
            .add_col_suffix('_dest_info')
            , on='d_id', how='inner')
        
        # similarly, we do an inner join if the origin state is one of the source
        # states (CA or CT---we include NEVADA TOO), again defaulting to CA
        .join(
            all_ent
            .filt(lambda x: 
                (x.state.fillna('CA').str.upper().str.match('^(CA$|CALIFORNIA|CT$|CONNECTICUT|NV$|NEVADA|AZ$|ARIZONA|OR$|OREGON)')) 
                & ~x.geometry.isna()
            )
            .add_col_suffix('_orig_info')
            , on='o_id', how='inner')
        
        .in_chain_immut(lambda df: logger.info(f"Generating searoutes for flows going outside North America: {len(df)} maritime flows out of {len(unique_ods)}"))
        
        
        # 
        .set_geometry('geometry_orig_info').to_crs(crs_ll)

        # at this point, we assume that the destinations are all outside of
        # North America and will be shipped via maritime routes 
        # # FIXME: vvv
        # we should confirm (perhaps with geometries) that the ultimate
        # destination is outside of North America Anyway
        
        # how we cross join all possible ports]
        .join(port_ents,how='cross')
        
        # compute direct line distances to sort them
        .assign(
            pdist_tmp=lambda x: ( 
                x.set_geometry('geometry_orig_info').to_crs(crs_usa)
                .geometry_orig_info.distance(
                    x.set_geometry('geometry_port_info').to_crs(crs_usa).geometry_port_info)
                ))
        # sort and keep the closest
        .sort_values('pdist_tmp')
        .groupby(['o_id','d_id']).head(1)
    )
    
    logger.info(f"Computing {port_dests_1.shape[0]} routes from origins to ports")
    assert(port_dests_1.geometry_orig_info.crs == crs_ll)
    assert(port_dests_1.geometry_port_info.crs == crs_ll)
    
    if not port_dests_1.empty:
        # we have destinations with maritime legs that need searooutes
        port_dests_2=(
            port_dests_1

            # Now we join all possible CA ports for the destination and compute the
            # over-road distances to all of them. from the origin to the port.
            # .join(port_ents, how='cross')
            .assign(
                # FIXME: SHOULD REMOVE THE OSRM DEPENDENCY HERE by reworking the
                # logic to work with the on-road routing.
                # # Lookup routes using OSRM,stores JSON result in route column
                route=lambda x: x.apply(
                    lambda row: (
                        [
                            {
                                'duration': rr['duration'],
                                'distance': rr['distance'],
                            }
                            for rr in safe_route_by_endpoints(
                                # shapely_point_to_osrm_point(row.geometry_orig_info), 
                                # shapely_point_to_osrm_point(row.geometry_port_info)
                                row.geometry_orig_info,
                                row.geometry_port_info
                                )['routes']
                        ]
                    ),
                    axis=1
                ),

                # extract total route distance for each row
                distance_u=lambda x: x.apply(lambda row: float(row.route[0]['distance']),
                                            axis=1).astype('pint[m]'),

                # extract total route duration for each row
                duration_u=lambda x: x.apply(lambda row: float(row.route[0]['duration']),
                                            axis=1).astype('pint[s]'),
                
            )
            # sort the o-ds by the to-port distance and select the first
            # we'll eventually use this as the over-road distance to the port
            .sort_values(['o_id', 'd_id', 'distance_u', 'duration_u'])
            .groupby(['o_id', 'd_id']).first()
            .reset_index()
        )
        logger.info(f"Computing {port_dests_2.groupby(['geometry_port_info','geometry_dest_info']).first()[[]].reset_index().shape[0]} searoutes from ports to destinations")
        searoute_pairs=(
            # select the unique port-destination pairs for which we need to compute searoutes
            port_dests_2.groupby(['geometry_port_info','geometry_dest_info']).first()[[]].reset_index()

            # next we compute the searoute from the port to the destination
            .assign(
                searte=lambda x: x.apply(
                    lambda row: dict(sr.searoute(row.geometry_port_info.coords[0], 
                                                 row.geometry_dest_info.coords[0]))
                    , axis=1),
                seadist_u=lambda x: x.apply(
                    lambda row: row.searte['properties']['length'] * ureg(row.searte['properties']['units']).to('km').magnitude
                    , axis=1).astype('pint[km]'),
                geometry_searte=lambda x: x.apply(
                    lambda row: shapely.geometry.shape(row.searte['geometry']), axis=1
                )
            )
            .set_geometry('geometry_searte').set_crs(crs_ll)
        )
        port_dests=(
            pd.merge(
                port_dests_2,
                searoute_pairs,
                right_on=['geometry_port_info','geometry_dest_info'],
                left_on=['geometry_port_info','geometry_dest_info'])
            [schema_to_cols(ODPairSchemaWithPortAndSearoute)]
        )
        
    else:
        # no searoutes, null-out the dataframe to match schema
        logger.info("No searoutes in provided OD flows")
        port_dests = validate_with_options(
            ODPairSchemaWithPortAndSearoute,
            port_dests_1,coerce=True,add_missing_columns=True
        )

    return port_dests


def generate_entxxsea(caports, rdrs_ent_allxxx):
    """Generate sea routes from California ports to international RDRS entities.

    Cross-joins California port entities with RDRS entities that have a maritime
    leg, computes the sea route for each port–destination pair, and selects
    the shortest sea route per (port, destination country) combination.

    Parameters
    ----------
    caports : GeoDataFrame
        California port entities with ``port_name`` and ``geometry_port`` columns.
    rdrs_ent_allxxx : GeoDataFrame
        All RDRS endpoint entities. Only rows with a maritime leg (per
        :func:`has_maritime_leg`) are routed.

    Returns
    -------
    GeoDataFrame
        One row per (port, destination country) with the shortest sea route,
        including ``searte``, ``seadist_km``, and ``geometry_searte``.

    Notes
    -----
    - Uses :func:`sr.searoute` for maritime path computation.
    - Distances are converted to km via :mod:`pint` unit handling.
    - CRS is assumed to be ``EPSG:4326`` (lon/lat).

    See Also
    --------
    generate_searoutes : Full pipeline version with OSRM on-road routing to port.
    """
    import shapely.geometry

    # caports = od_flows.filt(lambda x: x.geometry_src_dest == 'COUNTRY').drop_duplicates('o_rdrsid')
    # routes_through_port=rdrs_ent_allxxx.filt(lambda x: (x.geometry_src == 'COUNTRY'))
    routes_through_port=rdrs_ent_allxxx.filt(
        lambda x: (
            has_maritime_leg(x)
            # x.n2.fillna('').str.match('Out of USA') 
            # & ~x.geometry.isna()# FIXME: shouldn't really be any that are NA, but some are
            )
        )
    entxxsea = (
    # construct a crossed dataframe for ODs linking O={in california} and D={all ot
    # FIXMENOW: should be caports x ent_intl.ljoin(ports on country)
        caports.rename_geometry('geometry_port')
        .merge(routes_through_port.rename_geometry('geometry_dest'), how='cross')
        .assign(
            searte=lambda x: x.apply(
                lambda row: sr.searoute(row.geometry_port.coords[0], row.geometry_dest.coords[0]), axis=1),
            seadist_u=lambda x: x.apply(
                lambda row: row.searte['properties']['length'] * ureg(row.searte['properties']['units']).to('km').magnitude
                , axis=1).astype('pint[km]'),
            geometry_searte=lambda x: x.apply(
                lambda row: shapely.geometry.shape(row.searte['geometry']), axis=1).set_crs(crs_ll)
        )
        .sort_values('seadist_u')
        .groupby(['port_name', 'country_y']).first()
        .reset_index()
    )
    return entxxsea

@pa.check_types
def get_maritime_routes(od_flows_x: DataFrame[ODFlows],  #FIXME: type 
                        port_dests: DataFrame[ODPairSchemaWithPortAndSearoute]
                        ) -> DataFrame[ODPairSchemaWithPortAndSearouteFinal]: #FIXME: type:
    """Attach maritime route information to OD flows that have a maritime leg.

    Filters OD flows to those with a maritime destination, then left-joins
    the precomputed port/sea-route data from :func:`generate_searoutes`.
    Renames suffixed columns back to their canonical names for downstream
    consumption.

    Parameters
    ----------
    od_flows_x : DataFrame[ODFlows]
        Full OD flow data with origin/destination metadata.
    port_dests : DataFrame[ODPairSchemaWithPortAndSearoute]
        Maritime route data produced by :func:`generate_searoutes`, indexed
        by OD pair group columns.

    Returns
    -------
    DataFrame[ODPairSchemaWithPortAndSearouteFinal]
        OD flows with maritime leg data joined on, including port geometry,
        sea-route geometry, and sea distance. Non-maritime flows are excluded.

    See Also
    --------
    generate_searoutes : Produces the ``port_dests`` input.
    has_maritime_leg : Identifies which flows need maritime routing.
    """
    
    # grab od flows that have a maritime leg
    od_flows_with_maritime=od_flows_x.filt(lambda x: has_maritime_leg(x, endpoint_prefix='d_'))
    
    # split them into maritime and non-maritime
    
    # 
    mrte = (
        # For ods with a maritime leg (see above)
        od_flows_x.filt(lambda x: has_maritime_leg(x, endpoint_prefix='d_'))

        .join(port_dests.set_index(od_pair_group).add_col_suffix('_port'), on=od_pair_group, how='left')

        # .first()
        .rename(columns={'geometry_port_info_port': 'geometry_port',
                         'id_port_info_port': 'id_port',
                         'n1_port_info_port': 'name_port',
                         'searte_port': 'searte',
                         'geometry_searte_port': 'geometry_searte',
                         'seadist_u_port': 'seadist_u',
                         'distance_u_port': 'distance_u',
                         'duration_u_port': 'duration_u',
                         })
        .reset_index()
        [schema_to_cols(ODPairSchemaWithPortAndSearouteFinal)]
    )
    
    # if we don't have a searoute for any of the has_maritime leg ODs, we're out
    # of sync with the port_dests calculation
    # FIXME: should just collapse these instead of having two calcs---goes along with separating maritime legs
    # for now, we just through an error
    assert not mrte.id_port.isna().any(), "Maritime legs lack port_dests maritime routes. port_dests out of sync with has_maritime_leg"
    
    logger.info(f'{len(mrte)} ODs with a maritime leg')
    return( mrte)

@pa.check_types()
def split_multimodal(
    od_flows_x: DataFrame[ODFlows],
    all_ent:    DataFrame[LayerEndpointSchema]
) -> DataFrame[ODRoutableTourLeg]:
    od_flows_mm = (
        pd.concat([
            # get multi-modal-maritime
            od_flows_x.filt(lambda x: has_maritime_leg(x,endpoint_prefix='d_')).assign(
                o_id_tour = lambda x: x.o_id,
                d_id_tour = lambda x: x.d_id
            )
            .join(pd.DataFrame({'transport_mode':['on_road','maritime'],'leg':[1,2]}),how='cross')
            .assign(
                # mark the endpoints that need to be fixed
                o_id = lambda x: x.o_id.mask(x.transport_mode=='maritime','PORT_TO_FIND_MARITIME'),
                d_id = lambda x: x.d_id.mask(x.transport_mode=='on_road','PORT_TO_FIND_DRAYAGE')
            )
            ,
            # just on-road
            od_flows_x.filt(lambda x: ~has_maritime_leg(x,endpoint_prefix='d_')).assign(
                o_id_tour      = lambda x: x.o_id,
                d_id_tour      = lambda x: x.d_id,
                transport_mode = 'on_road',
                leg            = 1
            )
        ],ignore_index=True)
    )
    
    # we've split into maritime and on_road segments but still don't know the
    # transiting port, assign_maritime_tour_ports does that computation
    return assign_maritime_tour_ports(od_flows_mm, all_ent)
    
@pa.check_types
def assign_maritime_tour_ports(
    od_flows_mm: DataFrame[ODRoutableTourLeg], 
    all_ent: DataFrame[LayerEndpointSchema]
) -> DataFrame:
    # we have these tasks related to maritime routes
    # 1. determine the port of maritime od tour pairs (closest to the o)
    # 2. compute the maritime route for the p -> d leg
    # 3. assign the correct ods: 
    #    * for od tours with a maritime leg:
    #      * maritime = p -> d
    #      * on-road  = o -> p
    #    * for od tours without
    #      * on-road  = o -> d

    # first we group the unique od tour pairs
    port_dests = None
    od_tour_pair_group=['o_id_tour','d_id_tour']
    unique_maritime_ods = (
        od_flows_mm.filt(lambda x: x.transport_mode=='maritime')
        [['o_id_tour','d_id_tour'
          #,'o_id','d_id'
          ,'geometry_orig','geometry_dest']]
        .drop_duplicates()
    )
    port_ents  = all_ent.filt(lambda x: x.index.str.match('^PORT')).reset_index().add_col_suffix('_port_info')
    
    logger.info(f"Generating searoutes for flows going outside North America")
    if len(port_ents)==0:
        raise Exception("No ports in entities passed to searoutes")
    nearest_ports_to_consider=3
    unique_maritime_ods_with_nearest_ports = (
        # for all flows, we grab the o-d pair
        unique_maritime_ods
        .in_chain_immut(lambda df: logger.info(
            f"Generating searoutes for flows going outside North America: {len(df)} "
            f"maritime flows out of {len(unique_maritime_ods)}"))
        
        
        # 
        .set_geometry('geometry_orig').to_crs(crs_ll)

        # at this point, we assume that the destinations are all outside of
        # North America and will be shipped via maritime routes 
        # # FIXME: vvv
        # we should confirm (perhaps with geometries) that the ultimate
        # destination is outside of North America Anyway
        
        # now we cross join all possible ports]
        .join(port_ents,how='cross')
        
        # compute direct line distances to sort them
        .assign(
            pdist_tmp=lambda x: ( 
                x.set_geometry('geometry_orig').to_crs(crs_usa)
                .geometry_orig.distance(
                    x.set_geometry('geometry_port_info').to_crs(crs_usa).geometry_port_info)
                ))
        # sort and keep the closest
        .sort_values('pdist_tmp')
        .groupby(['o_id_tour','d_id_tour']).head(nearest_ports_to_consider)
    )
    
    logger.info(f"Computing {len(unique_maritime_ods_with_nearest_ports)} routes from origins to ports")

    assert(unique_maritime_ods_with_nearest_ports.geometry_orig.crs == crs_ll)
    assert(unique_maritime_ods_with_nearest_ports.geometry_port_info.crs == crs_ll)
    
    std_df=(
        unique_maritime_ods_with_nearest_ports
        .assign(
            o_id=lambda x: x.o_id_tour,
            d_id=lambda x: x.id_port_info,
            o_pt=lambda x: x.geometry_orig,
            geometry_dest=lambda x: x.geometry_port_info,
            d_pt=lambda x: x.geometry_dest
        )
        .filter(regex='^(?!.*_port_info)')
    )
    o_to_port_selected=(
        std_df        
        .merge(
            compute_on_road_routes(
                std_df
                [schema_to_cols(ODRoutablePts)]
                ,read_custom_cache=True
            )
            # keep the one with the smallest distance
            ,on=['o_pt','d_pt']
        )
        .filt(lambda x: x.groupby(['o_id_tour','d_id_tour']).distance_u.rank().pint.magnitude==1)
    )    
    
    od_flows_with_ports_assigned=(
        od_flows_mm.merge(
            o_to_port_selected.assign(leg=1).add_col_suffix('_drayage'),
            left_on=['o_id_tour','d_id_tour','leg'],
            right_on=[f'{c}_drayage' for c in ['o_id_tour','d_id_tour','leg']], 
            how='left')
    )
    
    # FIXME: could potentially add additional keys. e.g., a given material type
    # or EMFAC class may have restricted routing. For now we assume all trips 
    # for an od combo will travel have the same port and routing
    unique_route_key=['o_id_tour', 'd_id_tour'] 
    
    on_road_drayage_info = (
        od_flows_with_ports_assigned.loc[od_flows_with_ports_assigned['transport_mode'] == 'on_road']
        .groupby(unique_route_key)
        .agg({'d_id_drayage': 'first', 'geometry_dest_drayage': 'first'})
        .rename(columns={'d_id_drayage': 'port_id', 'geometry_dest_drayage': 'port_geometry'})
       .reset_index()  # ← move groupby keys back to regular columns
    )
    
    od_flows_with_drayage_info = (
        od_flows_with_ports_assigned#.drop(columns=['d_id_drayage'])  # ← remove before merge to avoid duplicates
        .merge(on_road_drayage_info, on=unique_route_key, how='left').assign(
            geometry_orig=lambda x: x['geometry_orig'].mask(
                x['o_id'].str.startswith('PORT_TO_FIND'),  # maritime leg's origin is the port
                x['port_geometry']
            ),
            geometry_dest=lambda x: x['geometry_dest'].mask(
                x['d_id'].str.startswith('PORT_TO_FIND'),
                x['port_geometry']
            ),
            # assign these after doing the geometries since the logic depends on "PORT_TO_FIND"
            o_id=lambda x: x['o_id'].mask(x['o_id'].str.startswith('PORT_TO_FIND'), x['port_id']),
            d_id=lambda x: x['d_id'].mask(x['d_id'].str.startswith('PORT_TO_FIND'), x['port_id'])
        )
        .drop(columns=['port_id', 'port_geometry'])
        .pipe(lambda x: x.drop(columns=x.filter(regex='_drayage$')))
    )
    
    # make sure all drayage legs have a port destination
    assert (
        od_flows_with_drayage_info
        .filt(lambda x: x.groupby(od_tour_pair_group).leg.transform('max')>1)
        .filt(lambda x: x.leg==1)
        .filt(lambda x: ~x.d_id.str.match('PORT')).empty
    ), "Non-port endpoints found for drayage leg of maritime tour"
    
    return od_flows_with_drayage_info

@pa.check_types()
def route_maritime(
    ods_to_route: DataFrame[ODRoutablePts],
    region:   DataFrame[RegionSchema],
) -> DataFrame[ODTourLegRouted]:
    assert ods_to_route.transport_mode.eq('maritime').all()
    
    od_pt_group = ['o_pt','d_pt']
    
    # now we've expanded the DF to include legs by all modes
    # we're going to compute the maritime routes now
    unique_maritime_ods_with_selected_port=(
        ods_to_route.filt(lambda x: x.transport_mode=='maritime')
        .groupby(od_pt_group)[[]].first().reset_index()
    )
    
    if len(unique_maritime_ods_with_selected_port) > 0:
        logger.info(f"Computing {len(unique_maritime_ods_with_selected_port)} searoutes from ports to destinations")
        # now compute the maritime leg routing
        searoute_pairs=(
            # select the unique port-destination pairs for which we need to compute searoutes
            unique_maritime_ods_with_selected_port

            # next we compute the searoute from the port to the destination
            .assign(
                searte=lambda x: x.apply(
                    lambda row: dict(sr.searoute(row.o_pt.coords[0], 
                                                 row.d_pt.coords[0]))
                    , axis=1),
                searte_dist_u=lambda x: x.apply(
                    lambda row: row.searte['properties']['length'] * ureg(row.searte['properties']['units']).to('km').magnitude
                    , axis=1).astype('pint[km]'),
                geometry_searte=lambda x: x.apply(
                    lambda row: shapely.geometry.shape(row.searte['geometry']), axis=1
                )
            )
            .set_geometry('geometry_searte').set_crs(crs_ll)
        )
        # just make sure
        assert not searoute_pairs.duplicated(subset=od_pt_group).any()

        # FOR SPEEDS; KAGI AI:
        # Vessel Type	    Typical Speed (knots)	Typical Speed (km/h)
        # Bulk carrier	    12–15 kn	            22–28 km/h
        # Container ship	18–25 kn	            33–46 km/h
        # Tanker	        12–16 kn	            22–30 km/h
        #
        # 4. Quick Heuristic (No Detailed Data)
        # If you only have the route length and vessel type, a practical rule of thumb:
        # 
        # Scenario	                             Effective Speed Assumption
        # Ideal conditions	                     Use nominal speed
        # Average open-ocean voyage	             Reduce nominal speed by 10–15%
        # Winter North Atlantic / rough regions	 Reduce nominal speed by 20–25%
        # Route with 1+ canal transit	         Add 10–16 hrs per canal
        # Route with 2+ port calls	             Add 6–12 hrs per port
        
        # for now, we'll just assume and average speed of 30 km/h
        ship_speed = ureg('30 km/hr')

        ret = validate_with_options(
            ODTourLegRouted,
            (
                ods_to_route
                .merge(searoute_pairs, on=od_pt_group,how='left')
                .add_col_if_missing('route',None)
                .add_col_if_missing('distance_u',None)
                .add_col_if_missing('geometry_full',None)
                .assign(
                    route=lambda x: x.route.mask(x.searte.notna(),x.searte),
                    distance_u=lambda x: (
                        x.distance_u.mask(x.searte.notna(),x.searte_dist_u)
                        .astype(f'pint[{str(searoute_pairs.searte_dist_u.pint.units)}]') # make sure the units are correct
                    ),
                    step_speed_u = lambda x: ship_speed,
                    duration_u=lambda x: x.distance_u / x.step_speed_u,
                    geometry_full=lambda x: x.geometry_full.mask(x.searte.notna(),x.geometry_searte),
                    
                    step_num = 1,
                    clip_num = 1,
                    geometry_clip = lambda x: x.geometry_full,
                    region = 'MARITIME',
                )            
            )
            ,coerce=True,add_missing_columns=True
        )
        
    else:
        # no searoutes, null-out the dataframe to match schema
        raise RuntimeError("Shouldn't call route_maritime without maritime legs")
        logger.info("No searoutes in provided OD flows")
        ret = validate_with_options(
            ODTourLegRouted,
            ods_to_route,
            coerce=True,add_missing_columns=True
        )

    return ret

# solution from kagi AI
from polyline_rs import decode_lonlat
from shapely.geometry import LineString
from shapely.ops import linemerge

def route_to_geom(route):
    lines = []
    for step in route[0]['steps']:
        coords = decode_lonlat(step['geometry_step_data'], 5)
        if len(coords) >= 2:
            lines.append(LineString(coords))  # already (lon, lat)!
    return linemerge(lines) if lines else None

@pa.check_types
def compute_on_road_routes(
        od_flows: DataFrame[ODRoutablePts],
        read_custom_cache=True,
        save_custom_cache=True,
        keep_extra_cols=False
        ) -> DataFrame[RouteSchemaPts]: # <- can't be RouteSchema because that has geometry_full, or we need to compute geometry_full here
    """Compute on-road OSRM routes for all unique OD pairs.

    For each unique origin–destination pair (considering maritime port
    substitutions), queries OSRM for driving directions including per-step
    geometry and annotations. Results are cached to a local pickle file so
    that subsequent runs only compute routes for new OD pairs.

    Parameters
    ----------
    od_flows : DataFrame[ODRoutable]
        OD flow data with ``geometry_orig``, ``geometry_dest``, and optional
        port/maritime columns. Flows with zero tonnage or missing geometries
        are excluded.
    read_custom_cache : bool, default True
        If True, load previously cached routes from
        ``model_cache_dir/distances_routes.pkl``.
    save_custom_cache : bool, default True
        If True, write newly computed routes to the same pickle file.

    Returns
    -------
    DataFrame[RouteSchema]
        OD pairs with OSRM route data, including ``route`` (raw JSON with
        steps), ``distance_u``, and ``duration_u``.

    Notes
    -----
    - For maritime flows, the on-road destination is the assigned port
      (``geometry_port``) rather than the final destination.
    - Routes are computed in batches of 100 to provide progress logging.
    - Each route's ``steps`` are extracted as a list of dicts with
      ``step_distance``, ``step_duration``, and ``geometry_step_data``
      (encoded polyline).
    - The function asserts that all wanted routes are present in the output.

    See Also
    --------
    route_by_endpoints : Single-pair OSRM routing with server fallback.
    split_on_road_routes_into_segments : Next stage that explodes routes into steps.
    """
    logger.info("On-road routes for all OD pairs using OSRM")

    logger.info("On-road routes for all OD pairs using OSRM: Re-computing distances")
    routable_keys=schema_to_cols(ODRoutablePts)
    distances_routes_all = (
        od_flows
        .groupby(routable_keys) # should include geoms in this
        .first()
        #.filt(lambda x: x.wt_sent > 0)
        .reset_index()
        .filt(lambda x: ~(x.o_pt.isna() | x.d_pt.isna()))
    )
    
    
    distances_routes_want = (
        distances_routes_all
        .set_geometry('o_pt').set_geometry('d_pt') # FIXME: legacy, was needed for .groupby().first() now replaced with drop duplicates
        [routable_keys].drop_duplicates()
    )
    
    distances_routes_have=None
    if read_custom_cache:
        cf = Path(cpath('model_cache_dir'),'distances_routes.pkl')
        if cf.is_file():
            distances_routes_have = pd.read_pickle(cf)
            
    if distances_routes_have is None:
        # no cache, get an empty version of distances_steps_want for what we have
        distances_routes_have = distances_routes_want.filt(lambda x: x.index==-1)

    # validate the _have df to make sure it matches the schema            
    distances_routes_have = validate_with_options(
        RouteSchemaPts
        ,distances_routes_have
        ,add_missing_columns=True,coerce=True
        ,ignore_non_null=True
    )

    distances_routes_need = (
        distances_routes_want.merge(distances_routes_have,on=routable_keys,how='left')
        .filt(lambda x: x.route.isna())[routable_keys]
    )

    logger.info(f'{len(distances_routes_want)} OD pairs to compute')            
    if not distances_routes_have.empty:
        logger.info(f'...Pre-loaded {len(distances_routes_want) - len(distances_routes_need)} needed from {len(distances_routes_have)} available in cache') 


    if not distances_routes_need.empty:
        logger.info(f'...Computing {len(distances_routes_need)} routes still needed')

        n = 100
        nrow = len(distances_routes_need)
        distances_routes_new = [
            distances_routes_need[i:i + n]
            .in_chain_immut(lambda df: logger.info(f"Looking up routes for {i}-{min(i + n,nrow)} of {nrow} {i / nrow * 100:.1f}%"))
            .assign(
                route=lambda x: x.apply(
                    lambda row: [
                        {
                            'duration': rr['duration'],
                            'distance': rr['distance'],
                            'steps': [{'step_distance': float(step['distance']),
                                        'step_duration': float(step['duration']),
                                        'geometry_step_data': step['geometry']}
                                        for step in sum([leg['steps'] for leg in rr['legs']], [])]
                        }
                        for rr in route_by_endpoints(row.o_pt, row.d_pt)['routes']
                    ],
                    axis=1
                ),
                distance_u=lambda x: x.apply(lambda row: float(row.route[0]['distance']), axis=1).astype('pint[m]'),
                duration_u=lambda x: x.apply(lambda row: float(row.route[0]['duration']), axis=1).astype('pint[s]')
            )
            .assign(geometry_full=lambda df: df['route'].apply(route_to_geom))
            .set_geometry('geometry_full')
            .set_crs(crs_ll)
            for i in range(0, len(distances_routes_need), n)
        ]
        distances_routes_new = pd.concat(distances_routes_new, ignore_index=True)
    else:
        distances_routes_new = distances_routes_have.filt(lambda x: x.index==-1)

    # FIXME: pint column coersion is dubious so convert the _u columns to base units before concat
    # FIXME: use units.unify_pint_units
    cols_to_keep=list(set(schema_to_cols(RouteSchemaPts)))
    common_units=get_pint_units(distances_routes_new[cols_to_keep]) | get_pint_units(distances_routes_have[cols_to_keep])
    # distances_steps_complete=(
    #     pd.concat([distances_steps_have[cols_to_keep].pipe(apply_units,common_units),
    #                distances_steps_new [cols_to_keep].pipe(apply_units,common_units),
    #               ],ignore_index=True)
    #     )
    distances_routes_complete=(
        pd.concat([distances_routes_have[cols_to_keep].pipe(transform_units,common_units),
                   distances_routes_new[cols_to_keep].pipe(transform_units,common_units)]
                  ,ignore_index=True)
    )

    # FIXME: THIS JOIN SHOULDN'T HAPPEN HERE. WE SHOULD DITCH THE DUPLICATES
    distances_routes_all = distances_routes_all.merge(
        distances_routes_complete[cols_to_keep] # - ['geometry_full']
        , on=routable_keys
        , how='left')
    assert(distances_routes_all.route.notna().all())
    assert(distances_routes_want.merge(distances_routes_all,on=routable_keys,how='left').filt(lambda x: x.route.isna()).empty)

    if save_custom_cache and not distances_routes_new.empty:
        logger.info(f"Caching {len(distances_routes_complete)} routes in distances_routes")
        cf = Path(cpath('model_cache_dir'),'distances_routes.pkl')
        distances_routes_complete.to_pickle(cf)
        
    distances_routes_all = (
        distances_routes_all
        .assign(geometry_full=lambda df: df.geometry_full.mask(df.geometry_full.isna(),df['route'].apply(route_to_geom)))
        .set_geometry('geometry_full')
        .set_crs(crs_ll)
    )

    if keep_extra_cols:
        ret = (
            od_flows.merge(
                distances_routes_all[schema_to_cols(RouteSchemaPts)],
                on=routable_keys,
                how='left',
                suffixes=('','_SHOULDNT_HAVE_COLLISIONS')
            )
        )
        assert ret.distance_u.notna().all()
        assert len(ret.filter(regex='_SHOULDNT_HAVE_COLLISIONS').columns)==0, 'Unexpected collisions merging routes'
        return ret
    else:
        return distances_routes_all

def split_on_road_routes_into_segments(
            distances_routes: DataFrame[RouteSchemaPts],
            read_custom_cache=True,
            save_custom_cache=True
        ) -> DataFrame[RouteStepsSchemaPts]:
    """Explode on-road routes into individual step segments with decoded geometries.

    Takes the route data from :func:`compute_on_road_routes` and:

    1. Decodes each route's polyline-encoded step geometries into coordinate lists.
    2. Explodes the steps array into one row per step.
    3. Converts each step's coordinates into a :class:`shapely.LineString`.
    4. Computes per-step speeds from distance and duration obtained from the router.
    5. Dissolves step geometries into a full-route ``geometry_full`` LineString. FIXME: does router return full geom?
    6. Assigns a ``step_num`` index within each OD pair group.

    Results are cached to ``model_cache_dir/distances_steps.pkl``, with
    incremental computation (only new routes are processed).

    Parameters
    ----------
    distances_routes : DataFrame[RouteSchema]
        Route data with ``route`` column containing OSRM step data.
    read_custom_cache : bool, default True
        Load previously cached step data from pickle.
    save_custom_cache : bool, default True
        Write newly computed step data to pickle.

    Returns
    -------
    DataFrame[RouteStepsSchema]
        One row per route step with decoded geometry, speed, and step number.

    Notes
    -----
    - Steps with zero line segments (single-point geometries) are filtered out.
    - Polyline coordinates are swapped from (lat, lon) to (lon, lat) for
      Shapely compatibility.
    - Zero-distance steps are assigned a speed of 0 to avoid division errors.
    - Processing is chunked in groups of ~1000 rows, keeping OD pair groups
      together within chunks.
    - A ``route_hash`` is computed from step polyline strings for deduplication
      and cache keying.

    See Also
    --------
    compute_on_road_routes : Produces the input route data.
    split_step_geometries_at_boundaries : Clips steps at regional boundaries.
    """

    logger.info("Splitting on-road routes into segments")
        
    # store_cols=od_pair_group + ['route_hash']    # cols to check against #FIXME: should it be ODRoutablePts (o_pt d_pt) instead of odpair?
    # store_cols=['geometry_orig','geometry_dest','route_hash']

    # store_join_cols defines the unique columns to save in the cache
    # these are the characteristics of a route step that we'll use to join
    # a cached record to one we need.
    store_join_cols = schema_to_cols(ODRoutablePts)
    store_cols      = store_join_cols #+ ['route_hash']
    
    # original, to eventially delete in favor of the schema derived one
    carried_cols=list(
        (set(schema_to_cols(RouteStepsSchemaPts) + schema_to_cols(RouteSchemaPts)) & set(distances_routes))
        -set(store_cols)
        )

    logger.info(f"Splitting route geometries into segments: generating keys")

    distances_steps_all = (
        distances_routes
        #.pipe(use_route_hash('geometry_step_data'))    # add a hash for the unique step geometry
        .drop_duplicates(subset=store_cols)
    )
    
    # so this defines the set of OD Routes that we want to split up and that
    # we want to find in the cache, if possibl (keyed by store_cols)
    distances_steps_want = (
        distances_steps_all.set_geometry('o_pt').set_geometry('d_pt')
        .groupby(store_cols).first()
        .reset_index() # we do this because we want to reference 'index' below
        [list(set(store_cols+carried_cols))]
        # .filt(lambda x: ~((x.geometry_step.isna() |x.geometry_step.is_empty) & (x.step_speed_u==0)))
    )
    logger.info(f"Splitting route geometries into segments: reading cache")
    
    
    distances_steps_have=None 
    if read_custom_cache:
        cf = Path(cpath('model_cache_dir'),'distances_steps.pkl')
        if cf.is_file():
            distances_steps_have = pd.read_pickle(cf)#.pipe(use_route_hash('geometry_step_data'))
    
    if distances_steps_have is None:
        # no cache, get an empty version of distances_steps_want for what we have
        distances_steps_have = distances_steps_want.filt(lambda x: x.index==-1)

    # validate the _have df to make sure it matches the schema            
    distances_steps_have = validate_with_options(
        RouteSchemaPts
        ,validate_with_options(
            RouteStepsSchemaPts
            ,distances_steps_have
            ,add_missing_columns=True,coerce=True
            ,ignore_non_null=True
        )
        ,add_missing_columns=True,coerce=True
        ,ignore_non_null=True
    )
    
    assert distances_steps_have.geometry_step.notna().all()

    logger.info(f"Splitting route geometries into segments: determining routes needing splits")
    distances_steps_need = (
        #distances_steps_want.merge(distances_steps_have,on=store_cols,how='left',suffixes=('','_want'))
        distances_steps_want.merge(distances_steps_have.ffill(),on=store_cols,how='left',suffixes=('','_have'))
        .filt(lambda x: x.geometry_step.isna() & ~x.geometry_step.apply(lambda x: x is None or x.is_empty))
    )
    assert distances_steps_need.filt(lambda x: (
        # make sure any matches providing an existing geometry_step actually match the geometry_full
        (x.geometry_step.notna()) & (x.geometry_full!=x.geometry_full_have))
    ).empty, (
        'Routes needing splitting have full geometry collisions -- logic bug'
    )
    distances_steps_need = distances_steps_need[store_cols + carried_cols]

    logger.info(f'{len(distances_steps_want)} OD pairs to segment')            
    if not distances_steps_have.empty:
        logger.info(f'...Pre-loaded {len(distances_steps_want) - len(distances_steps_need)} needed from {len(distances_steps_have)} available in cache') 


    if not distances_steps_need.empty:
        distances_steps_new = []
        i=0
        n = 1000
        nrow = len(distances_steps_need)
        logger.info(f"Splitting on-road routes into segments: Re-computing distances {n} at a time out of {nrow}")
        for cci, df in (
            # we want to split the distances frame into chunks of approximate n rows
            # but we want to keep the steps for each od_pair groups together in the same chunk
            # so we determine which n-size group the last element of the od_pair group is in
            # and use that index to group the od_pair groups
            distances_steps_need.reset_index(drop=True).reset_index().rename(columns={'index':'tindex'})
            .assign(cc=lambda x:x.groupby(schema_to_cols(ODRoutablePts)).tindex.transform('max'),
                    cci=lambda x: x.cc // n)
            .groupby('cci')
        ):

            distances_steps_new.append(
                df
                .in_chain_immut(lambda df: logger.info(f"Splitting on-road routes into segments: Decoding step geometries {i}-{min(i + n, nrow)} of {nrow} ({(i+len(df))/nrow*100:.1f}%)"))

                .assign(
                    # since we're interested in speed-derived emissions, we'll get average speeds by route steps
                    # these end up as an array of tuples in the steps column
                    steps=lambda x: (
                        x.apply(
                            lambda row:
                                [{'step_distance_m': float(step['step_distance']),
                                    'step_duration_s': float(step['step_duration']),
                                    'geometry_step_data': step['geometry_step_data']
                                    }
                                    for step in row.route[0]['steps']]
                            , axis=1)
                    )
                )

                # explode the step array into individual rows
                .in_chain_immut(lambda df: logger.info(f"Splitting on-road routes into segments: Exploding step geometries {i}-{min(i + n, nrow)} of {nrow} ({(i+len(df))/nrow*100:.1f}%)"))
                .explode('steps')

                # Extract how many line segments there are
                .assign(
                    geometry_segment_cnt=lambda x: x.apply(lambda row: len(row.steps['geometry_step_data'])-1, axis=1)
                )

                # for those steps with more than one segment extract the individual step geometries
                # (could just use step_distance?0 filter here?)
                .filt(lambda x: x.geometry_segment_cnt > 0)
                .in_chain_immut(lambda df: logger.info(f"Splitting on-road routes into segments: Converting step geometries {i}-{min(i + n, nrow)} of {nrow} ({(i+len(df))/nrow*100:.1f}%)"))
                .assign(
                    # geometry_step=lambda x: x.apply(lambda row: shapely.geometry.LineString(
                    #     # line string coords taken from the step data
                    #     [(c[1], c[0]) # step data is (lat,lon), we need to swap to (x,y)
                    #     for c in row.steps['geometry_step_data']]
                    # ), axis=1)
                    geometry_step=lambda x: (
                        x.steps
                        .apply(lambda y: [{'steps':[y]}]) # route_to_geom expects a "route format"
                        .apply(route_to_geom)
                    )
                )
            # for i in range(0, len(distances_routes), n)
            )
            i += n
            
        # pull them back together
        distances_steps_new = pd.concat(distances_steps_new, ignore_index=True)
    
        # conversions
        mps_to_kph = ureg('m/s').to('km/hr').magnitude
        mps_to_mph = ureg('m/s').to('mi/hr').magnitude

        # some final comps
        logger.info("Splitting on-road routes into segments: Exploding steps data into columns...")
        distances_steps_newx=(
            distances_steps_new.drop(columns='steps')
            # explode explode the columns in the steps field on join it
            .join(pd.DataFrame.from_records(distances_steps_new.steps))
            .filt(lambda x: ~x.route.isna())  # FIXME: we're ditching any bad routes
            .assign(geom_is_empty=lambda x: x.geometry_step.apply(lambda x: (x==None) or x.is_empty))
        )
        
        # make sure that any rows with null/empty geometry_step are because the 
        # step is a single point (which we don't need)
        assert (
            distances_steps_newx
            .filt(lambda x: x.geom_is_empty)
            .filt(lambda x: x.geometry_step_data.notna())
            .geometry_step_data.apply(decode_lonlat).apply(lambda x: list(set(x))).apply(len)==1
            ).all()
        
        # now clean and fill out things
        distances_steps_newxx=(
            distances_steps_newx
            .filt(lambda x: ~x.geom_is_empty)  # discard single point steps
            .assign(
                step_speed_u  = lambda x: (
                    (x.step_distance_m / x.step_duration_s)
                    .mask(x.step_duration_s==0, 0)          # will get rid of inf speeds where step_duration is 0
                    .astype('pint[m/s]')
                ),
                # step_speed_kph = lambda x: x.step_speed_u.pint.to('km / hr').pint.magnitude,
                # step_speed_mph = lambda x: x.step_speed_u.pint.to('mi / hr').pint.magnitude,
            )
            
            # drop the _m, _s, _mi, _km, _mph columns since we have the pint versions
            .pipe(lambda df: df.drop(columns=df.filter(like='_(m|s|mi|km|mph)$]').columns)) 
            
            # make route_hash
            # .pipe(use_route_hash('geometry_step_data'))
        )

        # create a single geometry for each route
        logger.info("Splitting on-road routes into segments: Creating full geometries...")
        distances_steps_new = gpd.GeoDataFrame(
            distances_steps_newxx
            .assign(
                # add a within path index
                step_num=lambda x: x.groupby(schema_to_cols(ODRoutablePts)).cumcount() + 1,
            )
            # .merge(
            #     gpd.GeoDataFrame(distances_steps_new,geometry='geometry_step',crs=crs_ll)
            #     .dissolve(by=schema_to_cols(ODRoutablePts))[['geometry_step']]
            #     .rename_geometry('geometry_full')
            #     .assign(
            #         geometry_full=lambda x: x.geometry_full.line_merge(),
            #         step_num=1   # we only want to merge onto rows where step_num==1
            #         )
            #     ,left_on=schema_to_cols(ODRoutablePts)+['step_num']
            #     ,right_on=schema_to_cols(ODRoutablePts)+['step_num']
            #     ,how='left'
            # )
            # [list(set(schema_to_cols(ODRoutablePts)+schema_to_cols(RouteStepsSchema))-set(od_pair_group))]
            ,geometry='geometry_step',crs=crs_ll
        )
        assert distances_steps_new.step_num.max()>1,(
               'Distances steps has no routes with more than one step. Likely a logic bug'
               )
            
    else:
        # get all valid rows or empty dataframe
        distances_steps_new = distances_steps_have.filt(lambda x: x.index==-1)


    # these are the columns we'll want to cache
    cols_to_keep=list(set(schema_to_cols(RouteSchemaPts)+schema_to_cols(RouteStepsSchemaPts)))

    # pint column coersion is dubious so convert the _u columns to base units before concat
    # get all needed columns
    # get the pint columns
    common_units_or=get_pint_units(distances_steps_new[cols_to_keep]) | get_pint_units(distances_steps_have[cols_to_keep])
    common_units_and=get_pint_units(distances_steps_new[cols_to_keep]) | get_pint_units(distances_steps_have[cols_to_keep])
    assert set(common_units_or.keys())==set(common_units_and.keys()), (
           "Distances steps_new and steps_have don't have a common set of pint columns"
    )
    common_units=common_units_or
    distances_steps_complete=(
        pd.concat([distances_steps_have[cols_to_keep].pipe(transform_units,common_units),
                   distances_steps_new [cols_to_keep].pipe(transform_units,common_units),
                  ],ignore_index=True)
        .filt(lambda x: ~((x.geometry_step.isna() |x.geometry_step.is_empty) & (x.step_speed_u==0)))
        .drop(columns=['route']) # no longer need route
        # only store on first step to reduce storate needed
        .assign(geometry_full=lambda x: x.geometry_full.mask(x.step_num>1,None))
        ) # FIXME: need to trim to distances_steps_all as in the _routes_ calcuation
    
    if save_custom_cache and not distances_steps_new.empty:
        logger.info(f"Splitting on-road routes into segments: Caching {len(distances_steps_complete)} route segments in distances_steps")
        cf = Path(cpath('model_cache_dir'),'distances_steps.pkl')
        distances_steps_complete.to_pickle(cf)

    return distances_steps_complete

# # custom hasher
# # see: https://github.com/python-cachier/cachier/issues/91
def _get_distances_param_hasher(args, kwargs):
    """Compute a cache key for :func:`get_distances` based on OD flow and region data.

    Hashes the subset of OD flow columns that affect distance computation
    (origin/destination IDs, geometries, and port info) plus the region
    geometries, producing a stable SHA-256 tuple for :mod:`cachier`.

    Parameters
    ----------
    args : tuple
        Positional arguments (unused).
    kwargs : dict
        Keyword arguments. Must contain:

        - ``od_flows_all_layers_w_maritime`` — the OD flows DataFrame
        - ``regions_full`` — the region boundaries DataFrame

    Returns
    -------
    tuple[str, str]
        Two-element tuple of SHA-256 hex digests: (flows_hash, regions_hash).

    See Also
    --------
    _generate_searoutes_param_hasher : Similar hasher for searoute caching.
    """
    od_flows_all_layers_w_maritime: DataFrame = kwargs['od_flows_all_layers_w_maritime']
    regions_full: DataFrame[RegionSchema] = kwargs['regions_full']
    
    df_hash1 = hashlib.sha256(
        pd.util.hash_pandas_object(
            od_flows_all_layers_w_maritime[[ # we're only interested in hashing on these fields
                'o_id','d_id',
                'geometry_orig',
                'geometry_dest',
                'id_port',
                # 'geometry_port',
            ]]
        ).values.tobytes()
    ).hexdigest()
    df_hash2 = hashlib.sha256(
        pd.util.hash_pandas_object(
            regions_full[['region','geometry']]
        ).values.tobytes()
    ).hexdigest()
    return (df_hash1, df_hash2)

@pa.check_types
def split_step_geometries_at_boundaries(
        distances_steps: DataFrame[RouteStepsSchemaPts] | None, 
        region:   DataFrame[RegionSchema],
        read_custom_cache: bool = True,
        save_custom_cache: bool = True
        ) -> DataFrame[RouteClipsSchemaPts]:
    """Clip route step geometries at CoAbDis regional boundaries.

    For each route step, performs a spatial join against CoAbDis region
    polygons and intersects the step geometry with each overlapping region.
    This produces one or more "clips" per step, each tagged with its region,
    enabling region-disaggregated emissions calculation.

    Steps that don't spatially intersect any region (due to floating-point
    precision or topology issues) are matched to the nearest region within
    100 km.

    Results are cached to ``model_cache_dir/distances_clips.pkl``, with
    incremental computation (only new clips are computed).

    Parameters
    ----------
    distances_clips_need : DataFrame[RouteStepsSchema] or None
        Route step data with ``geometry_step`` column. If None or empty,
        returns cached data or an empty validated DataFrame.
    region : DataFrame[RegionSchema]
        CoAbDis region boundaries with ``region`` name and ``geometry`` columns.
    read_custom_cache : bool, default True
        Load previously cached clip data from pickle.
    save_custom_cache : bool, default True
        Write newly computed clip data to pickle.

    Returns
    -------
    DataFrame[RouteClipsSchema]
        Route segments clipped at regional boundaries, with ``geometry_clip``,
        ``clip_num``, ``region``, and inherited route/step metadata.

    Notes
    -----
    - A ``route_hash`` is computed for deduplication and cache keying.
    - ``distance_u`` and ``duration_u`` are set to NaN for steps beyond the
      first (``step_num > 1``) to avoid double-counting in downstream
      aggregation.
    - ``geometry_step`` is set to None for clips beyond the first
      (``clip_num > 1``) since only the first clip retains the original
      step geometry.
    - Unmatched clips (no region from intersection) are backfilled via
      a nearest-region spatial join with a 100 km tolerance.
    - Processing is chunked in groups of ~5000 rows, keeping OD pair
      groups together within chunks.
    - Distance is converted from meters to miles via :mod:`pint`.

    Raises
    ------
    AssertionError
        If unmatched clips cannot be resolved to a nearest region, or if
        any clip has ``clip_num > 1`` in the unmatched set.

    See Also
    --------
    get_distances : Orchestrator that calls this function as stage 3.
    split_on_road_routes_into_segments : Stage 2 that produces the input steps.
    """
    
    logger.info(f"Splitting step geometries at RegionBoundaries")
    
    import hashlib
    
    od_pt_group = ['o_pt','d_pt']
    
    all_carried_cols=list(
        set(schema_to_cols(RouteSchemaPts)+schema_to_cols(RouteStepsSchemaPts))
         & set(distances_steps)
        )
    all_cols=(list(
        set(schema_to_cols(RouteStepsSchemaPts) 
            + schema_to_cols(RouteClipsSchemaPts) 
            + schema_to_cols(RouteSchemaPts))
        - set(['route']) # FIXME: should handle in schemas
    ))
    cols_to_keep=list(set(schema_to_cols(RouteSchemaPts)
                          +schema_to_cols(RouteClipsSchemaPts))
                      - set(['route']) # FIXME: should handle in schemas
                      )
    key_cols=[
        'step_num'
        # ,'route_hash'
            #   ,'geometry_step_hash'
              ]
    store_cols=od_pt_group + key_cols    # cols to check against
    carried_cols=list(set(all_carried_cols) - set(store_cols))
    
    logger.info(f"Splitting step geometries at RegionBoundaries: generating keys")
    distances_clips_all = (
        distances_steps
        # .pipe(use_route_hash('geometry_step_data'))    #needed for hashing
        # .pipe(add_geometry_step_hash)
        .drop_duplicates(subset=store_cols)
    )
    
    distances_clips_want = (
        distances_clips_all.drop_duplicates(subset=store_cols)[store_cols+carried_cols]
    )

    distances_clips_have=None
    if read_custom_cache:
        logger.info(f"Splitting step geometries at RegionBoundaries: reading from cache")
        cf = Path(cpath('model_cache_dir'),'distances_clips.pkl')
        if cf.is_file():
            distances_clips_have = (
                pd.read_pickle(cf)#.pipe(use_route_hash('geometry_step_data'))
                #.pipe(add_geometry_step_hash) #needed for hashing #FIXME: probably need something like this
            )
        
    if distances_clips_have is None:
        # no cache, get an empty version of distances_steps_want for what we have
        distances_clips_have = distances_clips_want.filt(lambda x: x.index==-1)

    # validate the _have df to make sure it matches the schema, this is mainly
    # for the case where we have a cache but it doesn't match the schema because
    # the schema has changed since the cache was created. FIXME: we should
    # probably version the cache so that we don't have to do this
    distances_clips_have = validate_with_options(
        ODRoutablePts
        ,validate_with_options(
            RouteClipsSchemaPts
            ,distances_clips_have
            ,add_missing_columns=True,coerce=True
            ,ignore_non_null=True
        )
        ,add_missing_columns=True,coerce=True
            ,ignore_non_null=True
    )

    logger.info(f'{len(distances_clips_want)} OD pairs to segment')            
        
    distances_clips_need = (
        # left join what we have to what we want
        distances_clips_want.merge(distances_clips_have,on=store_cols,how='left',suffixes=('','_have'))
        
        # filter for any rows where we don't have what we want
        .filt(lambda x: x.geometry_clip.isna())

        # this gives us what we need
    )
    assert distances_clips_need.filt(lambda x: (
        # make sure any matches providing an existing geometry_step actually match the geometry_full
        (x.geometry_clip.notna()) & (x.ffill().geometry_full!=x.geometry_full_have))
    ).empty, (
        'Route steps needing splitting have full geometry collisions -- logic bug'
    )
    distances_clips_need = distances_clips_need[store_cols + carried_cols]

    if not distances_clips_have.empty:
        logger.info(f'...Pre-loaded {len(distances_clips_want) - len(distances_clips_need)} needed from {len(distances_clips_have)} available in cache') 

    if not distances_clips_need.empty:

        n = 5000
        od_pair_step_group = od_pt_group + ['step_num']
        logger.info(f"Splitting needed step geometries at RegionBoundaries: Re-computing splits {n} at a time out of {len(distances_clips_need)}")
        distances_clips_new = []
        i=0
        for cci, df in (
            # we want to split the distances frame into chunks of approximate n rows
            # but we want to keep the steps for each od_pair groups together in the same chunk
            # so we determine which n-size group the last element of the od_pair group is in
            # and use that index to group the od_pair groups
            distances_clips_need.reset_index(drop=True).reset_index().rename(columns={'index':'tindex'})
            .assign(last_index_for_od_pair=lambda x:x.groupby(od_pt_group).tindex.transform('max'),
                    od_pair_processing_group=lambda x: x.last_index_for_od_pair // n)
            .groupby('od_pair_processing_group')
        ):
            print(f"Splitting needed step geometries at RegionBoundaries: Expanding step geometries {i}-{i + len(df)} of {len(distances_clips_need)}")
            logger.info(f"Splitting needed step geometries at RegionBoundaries: Start join {i}--{i+len(df)} of {len(distances_clips_need)} ({(i+len(df))/len(distances_clips_need)*100:.1f}%)")
            djx=gpd.sjoin(gpd.GeoDataFrame(df, geometry='geometry_step').set_crs(crs_ll).to_crs(crs_ca)
                            # sjoin will overwrite the geometry on the right side, 
                            # so we make a copy since we'll want the region geoms later
                            ,region.assign(geometry_region=lambda x: x.geometry).to_crs(crs_ca)
                            ,predicate='intersects', how='left')

            # after the join, we want to compute the portion of the route linestring
            # in each distinct region---use the intersection method
            logger.info(f"Splitting needed step geometries at RegionBoundaries: Completed join, doing clip {i}--{i+len(df)} of {len(distances_clips_need)} ({(i+len(df))/len(distances_clips_need)*100:.1f}%)")
            # .in_chain_immut(lambda df: wdisplay(df))
            djx=(
                djx
                .assign(
                    geometry_clip=lambda x: x.geometry_step.intersection(x.geometry_region.to_crs(crs_ca)).to_crs(crs_ca)
                    ,

                    # number each clip
                    clip_num=lambda x: x.groupby(od_pair_step_group).cumcount()+1
                )
                # .assign(
                #     # if there is no intersection, we just use geometry_step
                #     geometry_clip=lambda x: x.geometry_clip.fillna(x.geometry_step),
                # )
            )
            distances_clips_new.append(djx)
            i += len(df)
        distances_clips_new = pd.concat(distances_clips_new, ignore_index=True)

        # sometimes the clipping code produces unmatched segments. these will always have 
        # a single step and will have a null region. We're going to match these to the 
        # nearest region
        unmatched = distances_clips_new.filt(lambda x: x.region.isna() )
        
        if len(unmatched) > 0:
            logger.warning(f"{len(unmatched)} route clips have no region from overlaps --> mapping them to the nearest")
            assert unmatched.filt(lambda x: x.clip_num!=1).empty, ( # should never have clip_num > 1 in this set
                "Clip num != 1"
            )
            od_pair_clip_group = od_pair_step_group + ['clip_num']
            nearest_match = (
                unmatched
                .drop(columns='index_right')
                # pick clip if it exists and step if it doesn't
                .set_geometry('geometry_clip').to_crs(crs_ca)
                .set_geometry('geometry_step').to_crs(crs_ca)
                .assign(
                    tmp_geometry=lambda x: x.geometry_clip.combine_first(x.geometry_step)
                    )
                .set_geometry('tmp_geometry').set_crs(crs_ca)
                .sjoin(region.to_crs(crs_ca)
                        .assign(
                            geometry_region=lambda x: x.geometry
                            )
                        ,predicate='dwithin',distance=100000   # max 100km distance
                        )
                .assign(dist_tmp=lambda x: x.geometry_step.distance(x.geometry_region_right))
                # keep the most recent
                .sort_values('dist_tmp').groupby(od_pair_clip_group)
                .first()
                .reset_index()
            )
            assert(not nearest_match.region_right.isna().any())  # all should have a match
            distances_clips_new = (
                distances_clips_new.join(nearest_match.set_index(od_pair_clip_group)[['region_right']]
                                ,on=od_pair_clip_group
                                ,how='left')
                .assign(region=lambda x: x.region.combine_first(x.region_right))
            )
        
        assert(not distances_clips_new.region.isna().any())  # should be fixed now

        distances_clips_new = (distances_clips_new
                        # unmatched that are backfilled won't have geometry_clip defined
                        # so we simply copy the geometry_step
                        # FIXME: can this result in overlaps?
                        .assign(
                            geometry_clip=lambda x: x.geometry_clip.combine_first(x.geometry_step),
                            # clip_num=lambda x: x.clip_num.mask(x.clip_num==0,1),
                            geometry_step=lambda x: x.geometry_step.mask(x.clip_num>1,None),
                            )
                        # trip to what we need downstream
                        [all_cols]
                        )
    else:
        # get all valid rows or empty dataframe
        distances_clips_new = distances_clips_have.filt(lambda x: x.index==-1)

    # FIXME: pint column coersion is dubious so convert the _u columns to base units before concat
    
    
    common_units=get_pint_units(distances_clips_new[cols_to_keep]) | get_pint_units(distances_clips_have[cols_to_keep])
    distances_clips_complete=(
        pd.concat([distances_clips_have[cols_to_keep].pipe(transform_units,common_units),
                   distances_clips_new[cols_to_keep].pipe(transform_units,common_units)
                   ],ignore_index=True)
    )
    
    assert distances_clips_complete.o_pt.notna().all()
    
    if save_custom_cache and not distances_clips_new.empty:
        logger.info(f"Splitting needed step geometries at RegionBoundaries: Caching {len(distances_clips_complete)} route clips in distances_clips")
        cf = Path(cpath('model_cache_dir'),'distances_clips.pkl')
        distances_clips_complete.to_pickle(cf)


    return distances_clips_complete


@cachier(wait_for_calc_timeout=5,
        hash_func=_get_distances_param_hasher
        )
@pa.check_types
def get_distances(
        ods_to_route: DataFrame[ODRoutablePts], 
        regions_full: DataFrame[RegionSchema]
        ) -> DataFrame[ODTourLegRouted]: # FIXME: type
    """Compute full route distances with regional boundary clipping for all OD pairs.

    Orchestrates the three-stage routing pipeline:

    1. :func:`compute_on_road_routes` — OSRM route computation per OD pair.
    2. :func:`split_on_road_routes_into_segments` — Explode routes into steps.
    3. :func:`split_step_geometries_at_boundaries` — Clip steps at CoAbDis
       regional boundaries for disaggregated emissions calculation.

    For maritime flows, the on-road destination is the assigned port; the
    sea route is handled separately by :func:`generate_searoutes`.

    Results are cached via :mod:`cachier` using a custom parameter hasher
    that only considers OD pair identifiers, geometries, and port info,
    so cache hits are maximized when flow tonnage changes but routes don't.

    Parameters
    ----------
    od_flows_all_layers_w_maritime : DataFrame
        OD flows across all layers, with maritime route data joined on.
        Must include ``geometry_orig``, ``geometry_dest``, and optional
        port columns.
    regions_full : DataFrame[RegionSchema]
        CoAbDis (County/Air Basin/Air District) subarea boundaries used
        to segment route steps by emissions region.

    Returns
    -------
    DataFrame[RouteClipsSchema]
        Route segments clipped at regional boundaries, with per-clip
        geometry, region assignment, and distance/duration data.

    Notes
    -----
    - This is the primary entry point for distance computation in the pipeline.
    - The ``cachier`` decorator with ``wait_for_calc_timeout=5`` allows
      concurrent processes to wait for an in-progress computation rather
      than duplicating work.
    - OSRM stdout is redirected to the logger to avoid console spam.

    See Also
    --------
    compute_on_road_routes : Stage 1 — OSRM routing.
    split_on_road_routes_into_segments : Stage 2 — step explosion.
    split_step_geometries_at_boundaries : Stage 3 — regional clipping.
    """

    # Our `od_flows` dataframe is now ready to compute the on-road path for each
    # of our flows. Here, we use the OSRM to compute on-road shortest paths
    # between destinct OD pairs in the od_flows matrix. For those flows that
    # have a maritime leg, we use the selected port as the on-road destination.
    # Otherwise, the on-road destination is the merged geometry for the
    # associated destination endpoint.
    # 
    # Our query to OSRM asks for individual steps to be returned. This allows us
    # to compute speeds at a more disaggregate level to be used in emissions
    # calculations later.
    
    map_ods={ # FIXME: temporary
        'o_pt': lambda x: x.geometry_orig,
        'd_pt': lambda x: x.geometry_dest,
    }

    distances_routes = compute_on_road_routes(
        ods_to_route.assign(**map_ods),
        read_custom_cache=True,
        save_custom_cache=True)

    # The `distances_routes` dataframe has a single row for each distinct on-road
    # route. In order to compute speeds at the step level (here "steps" are akin
    # to distinct segments given in directions: "take the NB I-5 onramp for
    # 68m", "travel 20 miles on I-5 N", etc.). It is likely feasible to use OSM
    # ways to get finer resolution, but that is left for a later exercise. In
    # order to work with these steps, we need to explode the dataframe to
    # include them as distinct rows.

    distances_steps = split_on_road_routes_into_segments(distances_routes,
                                                         read_custom_cache=True,
                                                         save_custom_cache=True)

    # ### Accounting for emissions regions boundaries
    # 
    # The EMFAC emissions rates can be geographically disaggregated into County,
    # Air Basin, Air District (CoAbDis) subareas. Since geographic dispersion is
    # important localized air quality impacting health (e.g., criteria
    # pollutants), disaggregating as much as possible is important. As such, we
    # want to map our route steps onto `CoAbDis` subareas.

    # Now we do a spatial join between route segments and the `CoAbDis` shapes
    # to map them together. Since a single segment may overlap with multiple
    # `CoAbDis` shapes, we further subdivide them using an intersection
    # operation between the route step geometry and the `CoAbDis` shape.

    distances_clips = split_step_geometries_at_boundaries(distances_steps, 
                                                          regions_full,
                                                          read_custom_cache=True,
                                                          save_custom_cache=True)
    
    
    ret = ods_to_route.merge(distances_clips,on=schema_to_cols(ODRoutablePts),how='left')

    # we expect legs with colocated endpoints to not have been routed
    colocated = ret.o_pt == ret.d_pt
    no_steps  = ret.step_num.isna() | ret.clip_num.isna()
    if colocated.any() or no_steps.any():
        tofix = colocated | no_steps
        assert ret[tofix].geometry_clip.isna().all(), "Some colocated endpoints/unstepped routes unexpectedly routed"
        # create a zero distance linestring for colocated points to pass schema checks
        ret.loc[tofix,'geometry_clip']=[
           LineString([s, e]) for s, e in zip(ret.loc[tofix,'o_pt'], ret.loc[tofix,'d_pt'])
        ]
        ret.loc[tofix,'geometry_full']=ret.loc[tofix,'geometry_clip']
        ret.loc[tofix,'region']=(
            ret[tofix].set_geometry('o_pt').to_crs(crs_ca)
            .sjoin(regions_full.to_crs(crs_ca),predicate='covered_by',how='left')
            .region_right
        )
        if ret.step_num.isna().any():
            compute_on_road_routes(
                ret.filt(lambda x: x.step_num.isna()).groupby(['o_pt','d_pt'])[[]].first().reset_index(),
                read_custom_cache=True,
                save_custom_cache=True)
        ret.loc[tofix,'step_num'] = 1
        ret.loc[tofix,'clip_num'] = 1
        ret.loc[tofix,'distance_u'] = 0
        ret.loc[tofix,'duration_u'] = 0
        ret.loc[tofix,'step_speed_u'] = 0
        ret=ret.assign(
            step_num=lambda x: x.step_num.astype(int),
            clip_num=lambda x: x.clip_num.astype(int)
        )
        

    return ret



import hashlib

def add_route_hash(on_geom):
    def _hashit(df: DataFrame):
        return df.assign(route_hash=lambda x: x.route.apply(lambda r: hash(":".join([s[on_geom] for s in r[0]['steps']]))))
    return _hashit

def add_route_hash_hashlib(on_geom):
    
    def _hashit(df: DataFrame) -> DataFrame:
        steps_series = df['route'].str[0].str['steps']
        joined = steps_series.map(lambda steps: ":".join(s[on_geom] for s in steps))
        return df.assign(route_hash=joined.map(lambda s: hashlib.md5(s.encode()).hexdigest()))
    return _hashit

def add_route_hash_fast(on_geom):
    
    def _hashit(df: DataFrame) -> DataFrame:
        steps_series = df['route'].str[0].str['steps']
        return df.assign(route_hash=steps_series.map(
            lambda steps: hash((
                len(steps),                          # sequence length
                steps[0][on_geom],       # first
                steps[-1][on_geom],      # last
                steps[len(steps)//2][on_geom],  # middle
            ))
        ))
    return _hashit

use_route_hash=add_route_hash

def add_route_steps_hash(on_geom):
    def _hashit(df: DataFrame):
        return df.assign(route_hash=lambda x: x.route.apply(lambda r: hash(":".join([s[on_geom] for s in r[0]['steps']]))))
    return _hashit

use_route_steps_hash=add_route_steps_hash


def compute_nearest_neighbor_mapping(
    what: str,
    origins: gpd.GeoDataFrame,
    o_geom: str,
    destinations: gpd.GeoDataFrame,
    d_geom: str,
    join_key: str,
    radius: int = 100,
    max_options: int = 3,
    keep_options: int = 1,
) -> gpd.GeoDataFrame:
    """Compute nearest-neighbor mappings between origin and destination entities.

    For each origin entity, finds destination entities within a given radius,
    computes OSRM driving routes to each candidate, and selects the closest
    ones by road distance. Results are cached to a pickle file keyed by
    parameters.

    Parameters
    ----------
    what : str
        Descriptive name for the mapping (used in cache filename and logging).
    origins : GeoDataFrame
        Origin entities with geometry column specified by ``o_geom``.
    o_geom : str
        Name of the geometry column in ``origins`` to use for distance
        calculations.
    destinations : GeoDataFrame
        Destination entities with geometry column specified by ``d_geom``.
    d_geom : str
        Name of the geometry column in ``destinations`` to use for distance
        calculations.
    join_key : str
        Column name used to group origin entities (typically ``'id'`` or
        a ZIP-related key).
    radius : int, default 100
        Search radius in miles for initial candidate selection.
    max_options : int, default 3
        Maximum number of candidate destinations to route per origin.
    keep_options : int, default 1
        Number of top destinations to keep per origin (by road distance).

    Returns
    -------
    GeoDataFrame
        Origin–destination pairs with:

        - ``distance`` — straight-line distance (CRS units)
        - ``distance_u`` — OSRM road distance (meters)
        - ``duration_u`` — OSRM road duration (seconds)
        - ``route`` — raw OSRM route response
        - ``order`` — ranking within each origin group (1 = closest)

    Notes
    -----
    - Initial candidate selection uses ``sjoin`` with ``dwithin`` predicate
      in CA Albers projection (``crs_ca``).
    - Road distances are computed via :func:`osrm.simple_route`.
    - Results are cached to ``model_cache_dir/{what}_{radius}_{max_options}_{keep_options}_mapping.pickle``.
    - An assertion checks that all origins have at least one destination
      within the radius (currently disabled with FIXME).

    Raises
    ------
    AssertionError
        If any origin has no destination within the search radius (currently
        disabled — see FIXME in source).

    See Also
    --------
    osrm.simple_route : OSRM routing function used for road distance.
    """

    # If a cache exists for the given radius and max_options, use it
    fname = f'{what}_{radius}_{max_options}_{keep_options}_mapping.pickle'
    try:
        with open(cpath('model_cache_dir', fname), 'rb') as pfile:
            mapping = pickle.load(pfile)
            logger.info(f"Using cached version of {what}_mapping")
            return mapping
    except FileNotFoundError:


        # otherwise...
        logger.info(f"Recomputing {what}_mapping")

        # First, find all the destinations within the given radius of each origin
        mapping = (
            gpd.sjoin(
                origins
                # we expect duplicate labels after the join, since multiple
                # destinations can be within the radius of each origin
                .set_flags(allows_duplicate_labels=True)
                .set_geometry(o_geom).to_crs(crs_ca),
                (
                    destinations
                    .set_flags(allows_duplicate_labels=True) # have to allow duplicate labels here too to propogate through the join
                    .set_geometry(d_geom).to_crs(crs_ca)        # set the geometry to state plane for proper distance calcs
                    .assign(_geometry_dest=lambda x: x[d_geom]) # need to hold on to a copy of the destination geometry
                )
                , how='left', predicate='dwithin'
                , distance = radius * ureg('mi').to('m').magnitude
            )


            # d_geom will be gone after the join, we rename _geometry_dest so it's back
            .rename(columns={'_geometry_dest':d_geom})

            .in_chain_immut(lambda x: display(x))
        )

        # Next, keep only the closest max_options destinations for each origin
        mappingx = (
            mapping

            # handle nulls
            .assign(distance=lambda x: x.apply(lambda row: row[o_geom].distance(row[d_geom]) if row[o_geom] and row[d_geom] else 1e99,axis=1))

            # should handle out of state
            .sort_values([join_key,'distance'])
            .groupby(join_key).head(max_options)
            .reset_index()
            .set_geometry(o_geom).to_crs(crs_ll)
            .set_geometry(d_geom).to_crs(crs_ll)
            # .in_chain_immut(lambda x: display(x))
        )
        ## FIXME: Should raise if the following is > 0
        print(len(mapping.filt(lambda x: x[d_geom].isna())),f" dests not within {radius} mi of origin")
        if not (len(mapping.filt(lambda x: x[d_geom].isna()))==0):
            display(mapping.filt(lambda x: x[d_geom].isna()))
            assert(len(mapping.filt(lambda x: x[d_geom].isna()))==0) # FIXME: REENABLE THIS
        print(len(mapping.filt(lambda x: ~x[d_geom].isna())),f" lookups to perform")

        # now we compute the travel distance/time using OSRM for each candidate pair
        # sort them, and take the best
        mappingy=(
            mappingx
            .filt(lambda x: ~x[d_geom].isna()) # FIXME: we're losing some CARs here
            .assign(
                # look up the route for each ZIP centroid to candidate CAR
                route=lambda x: x.apply(
                    lambda row: (
                        [
                            {
                                'duration': rr['duration'],
                                'distance': rr['distance'],
                            }
                            for rr in
                            #simple_route(shapely_point_to_osrm_point(row[o_geom]), shapely_point_to_osrm_point(row[d_geom]))
                            safe_route_by_endpoints(row[o_geom],row[d_geom])
                            ['routes']
                        ]
                    ),
                    axis=1
                ),

                # extract total route distance for each row
                distance_u=lambda x: x.apply(lambda row: float(row.route[0]['distance']),
                                            axis=1).astype('pint[m]'),
                # extract total route duration for each row
                duration_u=lambda x: x.apply(lambda row: float(row.route[0]['duration']),
                                            axis=1).astype('pint[s]'),
            )

            # sort by distance
            .sort_values([join_key, 'distance_u', 'duration_u'])

            # compute ranking within each group
            .assign(
                order=lambda x: x.groupby(join_key).cumcount() + 1,
                # rank=lambda x: x.groupby(join_key).rank(method='first', ascending=True, numeric_only=True)
            )
            .filt(lambda x: x.order <= keep_options) # keep only the top N options
        )


        logger.info(f"Caching {fname}")
        mappingy.to_pickle(cpath('model_cache_dir', f'{fname}'))
        return mappingy