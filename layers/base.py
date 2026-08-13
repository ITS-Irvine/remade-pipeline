# With from __future__ import annotations, all annotations are stored as strings
# automatically (PEP 563), so forward references just work without quoting.
from __future__ import annotations  # put this at the top of the file

from abc import ABC, abstractmethod

import logging
from pathlib import Path
import pickle
from cachier import cachier
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

# loads the common model_types
# from layers.appliance import get_appliance_destinations, get_appliance_zips, get_reclaimer_destinations, load_appliance_flows, read_appliance_external_rdrs
from services.geocode import Geocoder, crs_ll
from core.common import disable_index_checks
from core.model_types import *
from core.model_types import AnnualizedLayeredGroupedFlowIdSchema
from core.model_types import _ODPairSchemaNames
from core.model_types import _ODPairSchemaGeometries
from core.model_types import _ODPairSchemaAggregations
from core.model_types import _ODPairSchemaFacilityTypes


logger = logging.getLogger(__name__)

class ModelLayer(ABC):
    """Abstract base class for pipeline model layers.

    Defines the interface that all material-flow layers (RDRS, Appliance,
    CT, Ewaste, ELV) must implement. Each layer provides endpoint entities
    (facilities with geographic locations) and origin–destination flow
    records (material movements between facilities).

    Parameters
    ----------
    name : str
        Human-readable layer name (e.g., ``'rdrs'``, ``'Appliances'``).

    Attributes
    ----------
    name : str
        Layer name, accessible via :meth:`get_name`.

    Methods
    -------
    get_name()
        Return the layer name.
    get_endpoints()
        Return geocoded endpoint entities (abstract).
    get_flows()
        Return origin–destination flow records (abstract).
    get_materials()
        Return materials handled by this layer.

    Notes
    -----
    Subclasses must implement :meth:`get_endpoints` and :meth:`get_flows`.
    The default :meth:`get_materials` returns an empty list; override in
    subclasses that track material categories.

    See Also
    --------
    RDRSLayer : Refrigerant distribution & recovery layer.
    ApplianceLayer : Appliance recycling layer.
    CTLayer : Connecticut-specific layer.
    EwasteLayer : Electronic waste layer.
    ELVLayer : End-of-life vehicle layer.
    """
    
     # keep track of instantiated layers to ensure they don't have the same name
    _used_names: set = set()

    def __init__(self, name):
        """Initialize the layer with a name.

        Parameters
        ----------
        name : str
            Human-readable layer name.
        """
        super().__init__()
        
        # safely assign name
        if name in self.__class__._used_names:
            raise ValueError(
                f"A ModelLayer with name '{name}' already exists. "
                f"All layer names must be unique. "
                f"Currently registered: {sorted(self.__class__._used_names)}"
            )
        self.__class__._used_names.add(name)
        self.name=name
        
        pass

    def get_name(self) -> str:
        """Return the layer name.

        Returns
        -------
        str
            The layer name set at initialization.
        """
        return self.name

    @abstractmethod
    def get_endpoints(self) -> DataFrame[LayerEndpointSchema]:
        """Return geocoded endpoint entities for this layer.

        Each endpoint represents a facility or location with geographic
        coordinates, administrative region metadata, and a unique ID.

        Returns
        -------
        DataFrame[LayerEndpointSchema]
            Endpoint entities with ``id`` index, ``geometry``, ``n1``,
            ``n2``, ``county``, ``state``, ``country``, ``facility_type``,
            and ``src`` columns.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        pass

    @abstractmethod
    def get_flows(self) -> DataFrame[LayerFlowSchema]:
        """Return origin–destination flow records for this layer.

        Each flow record represents a material movement from an origin
        endpoint to a destination endpoint, with tonnage, trip count,
        material grouping, and EMFAC vehicle class information.

        Returns
        -------
        DataFrame[AnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows]
            Flow records with ``o_id``, ``d_id``, ``wt_sent``, ``trips``,
            ``ttype``, ``material_grouping``, ``EMFAC_class``, and ``layer``
            columns.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        pass
    
    def get_flows_with_endpoints(self): # FIXME: types
        return ModelLayer.attach_flows_with_endpoints(
            self.get_flows(),
            self.get_endpoints()
        )

    def get_materials(self):
        """Return the materials handled by this layer.

        Returns
        -------
        list
            Material names or categories. Default is an empty list;
            override in subclasses that track material categories.
            
        Notes
        -----
        - FIXME: this isn't currently used consistently by the pipeline

        """
        return []
    
    @abstractmethod
    def get_geocoder(self) -> Geocoder:
        pass
    
    @staticmethod
    @pa.check_types
    def attach_flows_with_endpoints(
        flows:     DataFrame[LayerFlowSchema],
        endpoints: DataFrame[LayerEndpointSchema],
        ep_cols: List[str] = None,  # all cols by default
        o_prefix: str = 'o_',
        d_prefix: str = 'd_'
    ) -> DataFrame[LayerFlowSchema]: # FIXME: types 
        if ep_cols is None:
            ep_cols = list(endpoints.columns) 
        o_id=f'{o_prefix}id'
        d_id=f'{d_prefix}id'
        ret = (
            flows
            .merge(endpoints[ep_cols].add_col_prefix(o_prefix),left_on=o_id,right_index=True,how='left')
            .merge(endpoints[ep_cols].add_col_prefix(d_prefix),left_on=d_id,right_index=True,how='left')
        )
        # make sure they all matched
        firstcol = ep_cols[0]
        assert ret.filter(regex=f'^{o_prefix}(?!id$).*').notna().any().any(), "Missing attached origins"
        assert ret.filter(regex=f'^{d_prefix}(?!id$).*').notna().any().any(), "Missing attached destinations"
        return ret
    
    # make a static class of ModelLayer
    @classmethod
    @pa.check_types
    def merge_all_layer_flows(
        cls,
        layers:                     List[ModelLayer],
        merged_entities_all_layers: List[LayerEndpointSchema]
        ) -> LayerFlowSchema:
        logger.info(f"Merging OD flows for layers: [{', '.join([l.name for l in layers])}]")

        return ( 
            # Concatenate all layer flows into a single DataFrame
            pd.concat([layer.get_flows() for layer in layers], ignore_index=True)

            # Select only the aggregation keys and data columns
            [unique_flow_cols + list(unique_flow_agg.keys())]

            # Remove flow pairs with zero tons sent (no material moved)
            .filt(lambda x: x.wt_sent>0)

            # Aggregate duplicate OD pairs by summing tonnage and trips,
            # ensuring one row per unique OD combination
            .groupby(unique_flow_cols)
            .agg(unique_flow_agg)    # this will strip to o,d, and flow info
            .reset_index()

            # join origin and destination enditity metadata (name, facility type, location, geometry)
            .pipe(cls.attach_flows_with_endpoints,merged_entities_all_layers)

            # # Join origin entity metadata (name, facility type, location, geometry)
            # .join(merged_entities_all_layers.add_col_prefix('o_'),on='o_id',how='left')

            # # Join destination entity metadata (name, facility type, location, geometry)
            # .join(merged_entities_all_layers.add_col_prefix('d_'),on='d_id',how='left')
            
            # Select the final set of columns: OD keys, entity info, and flow data
            # FIXME: use a schema
            [  unique_flow_cols
                + schema_to_cols(_ODPairSchemaNames)
                + schema_to_cols(_ODPairSchemaGeometries)
                + schema_to_cols(_ODPairSchemaAggregations)  # county, state, country
                + schema_to_cols(_ODPairSchemaFacilityTypes) # facility_type
                + list(unique_flow_agg.keys())
            ]
            
            # Rename geometry columns for backward compatibility with cachier cache keys
            # FIXME: probably remove
            .rename(columns={
                'o_geometry':'geometry_orig',
                'd_geometry':'geometry_dest'
                })

            # IMPORTANT: the od_flow_index is used to keep track of distinct flows
            # that may have multi-modal legs
            .reset_index()        
            .rename(columns={'index':'od_flow_index'})
        )

    
    @classmethod
    def filter_to_layer(
        cls,
        flows: LayerFlowSchema
    ) -> LayerFlowSchema:
        return flows.filt(lambda x: x.layer == cls.name)
    
    
    #### some helpers to explicitly release a name (e.g. in tests or when rebuilding a pipeline):
    @classmethod
    def release(cls, name: str) -> None:
        """Explicitly free a layer name, e.g. in tests or pipeline resets."""
        cls._used_names.discard(name)

    @classmethod
    def release_all(cls) -> None:
        """Clear all registered names. Useful between test runs."""
        cls._used_names.clear()


    # FIXME: to complete after figuring out what to do with Emissions
    ## Saving
    def to_pickle(self, path: str | Path) -> None:
        """Save instance to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: str | Path) -> "ModelLayer":
        """Load instance from a pickle file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(obj).__name__}"
            )
        return obj        

# FIXME:create function aggregate_for_od_totals

def mask_for_od_totals(df):
    """Mask tonnage and trip values on non-first step/clip rows to avoid double-counting.

    In the route-segment emissions pipeline, each OD flow is exploded into
    multiple rows (one per step × clip combination). Tonnage and trips should
    only be counted once per OD pair — on the first step and first clip —
    to prevent double-counting when aggregating.

    This function sets ``wt_sent`` and ``trips`` to 0 for all rows where
    ``step_num != 1`` or ``clip_num != 1``, leaving the first step/clip
    row with the original values.

    Parameters
    ----------
    df : DataFrame
        Route-segment data with ``step_num``, ``clip_num``, ``wt_sent``,
        and ``trips`` columns.

    Returns
    -------
    DataFrame
        Same shape as input, with ``tonssent`` and ``trips`` set to 0
        on non-first step/clip rows. First step/clip rows retain their
        original values.

    Notes
    -----
    - Uses :meth:`Series.mask` rather than :meth:`Series.where` so that
      matching rows (non-first) are set to 0, while non-matching rows
      (first) are preserved.
    - The mask condition is ``(step_num != 1) | (clip_num != 1)``, meaning
      only rows where **both** ``step_num == 1`` **and** ``clip_num == 1``
      retain their tonnage/trip values.
    - FIXME: this needs typing

    See Also
    --------
    filter_for_od_totals : Alternative approach that filters to first
        step/clip rows instead of masking values.
    """
    return df.assign(
        wt_sent=lambda x: x.wt_sent.mask((x.step_num!=1) | (x.clip_num!=1),0),
        trips=lambda x: x.trips.mask((x.step_num!=1) | (x.clip_num!=1),0)
    )
    

def filter_for_od_totals(df):
    """Filter to only the first step/clip row per OD pair.

    Returns only rows where ``step_num == 1`` and ``clip_num == 1``,
    which represent the first segment of each route. This is useful
    when you need the actual OD-level records (one per OD pair) rather
    than the full exploded segment table.

    Parameters
    ----------
    df : DataFrame
        Route-segment data with ``step_num`` and ``clip_num`` columns.

    Returns
    -------
    DataFrame
        Subset containing only first-step, first-clip rows.

    Notes
    -----
    - Uses the custom ``.filt()`` accessor (equivalent to
      ``df[df.apply(lambda, axis=1)]``).
    - This is a **filtering** alternative to :func:`mask_for_od_totals`,
      which **masks** values instead. Choose based on downstream needs:

      - :func:`mask_for_od_totals` — keeps all rows, zeroes non-first values
        (preserves segment-level structure for emissions aggregation).
      - :func:`filter_for_od_totals` — drops non-first rows entirely
        (produces one row per OD pair).
    - FIXME: needs typing

    See Also
    --------
    mask_for_od_totals : Masking alternative that preserves all rows.
    """
    return df.filt(lambda x: (x.step_num==1) & (x.clip_num==1))


#@cachier(wait_for_calc_timeout=5)
@pa.check_types
def merge_all_endpoints(
        endpoints_lists: List[LayerEndpointSchema]
    ) -> DataFrame[LayerEndpointSchema]:
    """Merge endpoint entities from all layers into a single deduplicated GeoDataFrame.

    Concatenates endpoint entities from multiple layers, normalizes their
    CRS to WGS84 (``EPSG:4326``), and resolves duplicate entity IDs by
    merging rows that share the same ``id`` and ``geometry`` pair.

    Duplicate resolution strategy:

    1. **Assert uniqueness of id/geometry pairs** — confirms that each
       entity ID maps to at most one geometry (i.e., the same ID doesn't
       appear with different locations across layers).
    2. **Check for conflicting data** — computes the number of unique
       non-geometric values per id/geometry group; values > 1 indicate
       conflicting metadata across layers.
    3. **Merge duplicates** — joins conflicting string values with
       ``' /// '`` separator via :func:`safemerge`.

    Parameters
    ----------
    endpoints_lists : list of DataFrame[LayerEndpointSchema]
        Endpoint entities from each layer. Each must have ``id`` index,
        ``geometry``, ``geometry_agg``, and standard endpoint columns.

    Returns
    -------
    DataFrame[LayerEndpointSchema]
        Merged, deduplicated endpoint entities indexed by ``id`` in
        CRS ``EPSG:4326``.

    Raises
    ------
    AssertionError
        If any entity ID maps to more than one distinct geometry (i.e.,
        the same ID appears with different locations in different layers).
        This indicates a data integrity issue that must be resolved manually.

    Warns
    -----
    logger.warning
        If duplicate entity IDs are found during concatenation.
    logger.warning
        If duplicate entities have conflicting metadata values across layers.

    Notes
    -----
    - CRS normalization is applied to both ``geometry`` and ``geometry_agg``
      columns via the internal :func:`ensure_crs` helper.
    - Column selection is enforced against
      ``LayerEndpointSchema`` after concatenation,
      dropping any extra columns introduced by individual layers.
    - The ``safemerge`` aggregation function joins conflicting string
      values with ``' /// '`` separator, preserving all information
      rather than silently dropping data.
    - Caching is commented out (``#@cachier(wait_for_calc_timeout=5)``)
      but can be re-enabled for performance.

    Examples
    --------
    >>> rdrs_endpoints = rdrs_layer.get_endpoints()
    >>> appl_endpoints = appliance_layer.get_endpoints()
    >>> merged = merge_all_endpoints([rdrs_endpoints, appl_endpoints])
    >>> merged.index.is_unique
    True

    See Also
    --------
    ModelLayer.get_endpoints : Produces the per-layer endpoint data.
    LayerEndpointSchema : Schema for the output.
    """

    def ensure_crs(gdf, target_crs="EPSG:4326", geometry_name='geometry'):
        """Sets the CRS of a GeoDataFrame only if it is currently undefined."""
        curgeometry_name = gdf.geometry.name
        if gdf.set_geometry(geometry_name).crs is None:
            return gdf.set_geometry(geometry_name).set_crs(target_crs, allow_override=True).set_geometry(curgeometry_name)
        return gdf.set_geometry(geometry_name).to_crs(target_crs).set_geometry(curgeometry_name)

    # we concat all of the endpoints from all layers
    res=(
        pd.concat(
            disable_index_checks(
                [ensure_crs(ensure_crs(ep,crs_ll,'geometry'),crs_ll,'geometry_agg') for ep in endpoints_lists]
            ))
        [list(LayerEndpointSchema.to_schema().columns.keys())]
    )

    if not res.index.is_unique:
        logger.warning(f'{sum(res.index.duplicated())} Duplicated entities...attempting to clean')

        # see if we have collisions in the duplicates

        # first, confirm that we have unique id/geometry pairs
        assert(res.reset_index().assign(tmp=1).groupby(['id'])[['geometry']].nunique().max().max()<=1)

        # first, create unique rows for each id/geometry pair
        meal_test=(
            res
            # tag the duplicated columns
            .assign(tmp=lambda x: x.duplicated())
            .reset_index()

            # group by unique 'id'/'geometry' pairs
            .groupby(['id','geometry'])
            .agg({'tmp':'sum',
                **{col: 'nunique' for col in res.columns
                                            if col not in ['id','geometry','tmp','src']}})
            .filt(lambda x: x.tmp>0)

            # take the max of all cells
            .max().max()
        )

        if meal_test>1:
            logger.warning(f'We have {meal_test} duplicated entities with different ids...this should be investigated further')
            
        # assert(meal_test<2) # stop if any cells are 2 or more

        # otherwise, create a safe merge function (this won't be used, but just in case)
        safemerge = lambda x: ' /// '.join([str(x) for x in x.unique() if x is not None])

        # merge the result by unique id and geometry pairs
        res=(
            res
            .reset_index()
            # group by unique 'id'/'geometry' pairs
            .groupby(['id','geometry'])
            .agg(safemerge)
            .reset_index()
            .set_index('id')
            .set_geometry('geometry')
            .to_crs(crs_ll)
        )
    return res


od_pair_group=schema_to_cols(ODPairSchema)
"""list of str: Column names defining a unique origin–destination pair.

Used throughout the pipeline as the grouping key for OD-level operations
such as joining distances, maritime routes, and emissions data.
"""
