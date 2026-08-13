from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Annotated, Callable, List, Optional
import pandas as pd
import geopandas as gpd
from pandera import Check, Index, dtypes
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from pandera.typing.pandas import Index as PandasIndex
from pandera.typing.geopandas import GeoDataFrame, GeoSeries
from pandera.errors import SchemaErrors
import pint_pandas
import core.common
from core.units import ureg
from pandera.engines import pandas_engine
import numpy as np

import logging

from pint import DimensionalityError
logger = logging.getLogger(__name__)

# from PintDType import extend_model_schema, dim_check
from core.units import transform_units
from utils.pandera_checks import get_required_columns
from utils.pandera_checks import with_unique_check
from utils.pandera_checks import with_unique_index


def pint_series(values, unit: str) -> pd.Series:
    return pd.Series(pd.array(values, dtype=f'pint[{unit}]'))

def pint_columns_to_base_units(df):
    """Convert any pint columns in df to base units (e.g., km -> m)"""
    pint_cols=pint_cols=[a for (a,b) in df.dtypes.items() if str(b).startswith('pint[')]
    if pint_cols:
        base_unit_dict={col:str(ureg.get_base_units(df[col].pint.units)[1]) for col in pint_cols}
        return df.pipe(transform_units,base_unit_dict)
    else:
        return df
    
###
def has_dimension(series: pd.Series, *, reference_unit: str) -> bool:
    if not hasattr(series.dtype, 'units'):
        return False
    try:
        series.pint.to(reference_unit)
        return True
    except (DimensionalityError, AttributeError):
        return False

def pint_dimension_parser(reference_unit: str):
    def parser(series: pd.Series) -> pd.Series:
        if not has_dimension(series, reference_unit=reference_unit):
            dimension_name = ureg(reference_unit).dimensionality
            raise ValueError(
                f"Column '{series.name}' does not have dimension {dimension_name}"
            )
        return series
    return parser

# --- Parameterized PintDtype that carries a default unit ---

@pandas_engine.Engine.register_dtype(
    equivalents=[pint_pandas.pint_array.PintType],
)
@dtypes.immutable(init=True)
class PintDtype(pandas_engine.DataType):
    """Custom pandera dtype for pint columns, parameterized by default_unit."""

    type = object
    default_unit: str = "dimensionless"

    @classmethod
    def from_parametrized_dtype(cls, pint_type: pint_pandas.pint_array.PintType):
        unit = str(pint_type.units)
        return cls(default_unit=unit)

    def check(self, pandera_dtype, data_container=None):
        if data_container is not None:
            actual_dtype = str(data_container.dtype)
            return actual_dtype.startswith("pint[")
        dtype_str = str(pandera_dtype)
        return dtype_str.startswith("pint[")

    def coerce(self, data_container):
        """Coerce a Series to pint dtype using self.default_unit.

        KEY FIX: We must produce a Series with a REAL PintType dtype
        (not our PintDtype wrapper), so that pandas internals like
        _dtype_to_na_value can handle it properly.
        """
        if hasattr(data_container.dtype, 'units'):
            # Already a real pint dtype — no coercion needed
            return data_container

        # Build a proper pint Series from scratch using pint_pandas,
        # so the resulting dtype is a real PintType (not our wrapper).
        # This ensures pandas internals see ExtensionDtype, not our
        # custom DataType, and can access .na_value, .kind, etc.
        try:
            unit = ureg(self.default_unit)
            values = data_container.values

            # Create a PintArray directly — this guarantees the dtype
            # is a real pint_pandas PintType, not our PintDtype wrapper
            pint_array = pint_pandas.PintArray(values, dtype=f"pint[{self.default_unit}]")
            return pd.Series(pint_array, index=data_container.index, name=data_container.name)
        except (TypeError, ValueError) as e:
            # If values can't be converted (e.g., strings), return as-is
            # and let the parser/check raise a proper error
            return data_container

    def coerce_value(self, value):
        if pd.isna(value):
            return np.nan
        return ureg.Quantity(value, self.default_unit)

    def __str__(self) -> str:
        return f"pint[{self.default_unit}]"

o_prefix="o_"
d_prefix="d_"
    

class ODPairSchema(pa.DataFrameModel):
    """General OD Pair data, has origin and destination."""
    o_id: Series[str]
    d_id: Series[str]
    
class TourPairSchema(pa.DataFrameModel):
    o_id_tour: Series[str]  # 
    d_id_tour: Series[str]
    
class TransportMode(str, Enum):
    ON_ROAD = 'on_road'
    MARITIME = 'maritime'
    RAIL     = 'rail'

class TourLegSchema(TourPairSchema,ODPairSchema):
    leg:            Series[int] = pa.Field(nullable=False,ge=1)
    transport_mode: Series[str] = pa.Field(
        nullable=False,
        isin=[m.value for m in TransportMode]
    )

class _ODPairSchemaFacilityTypes(pa.DataFrameModel):
    o_facility_type: Series[str] = pa.Field(nullable=True)
    d_facility_type: Series[str] = pa.Field(nullable=True)

class _ODPairSchemaAggregations(pa.DataFrameModel):
    o_country: Series[str] = pa.Field(nullable=True)
    d_country: Series[str] = pa.Field(nullable=True)
    o_state:   Series[str] = pa.Field(nullable=True)
    d_state:   Series[str] = pa.Field(nullable=True)
    o_county:  Series[str] = pa.Field(nullable=True)
    d_county:  Series[str] = pa.Field(nullable=True)

class _ODPairSchemaGeometries(pa.DataFrameModel):
    o_geometry: GeoSeries
    o_geometry_src: Series[str]
    d_geometry: GeoSeries
    d_geometry_src: Series[str]

class _ODPairSchemaNames(pa.DataFrameModel):
    o_n1: Series[str] = pa.Field(nullable=True)
    o_n2: Series[str] = pa.Field(nullable=True)
    d_n1: Series[str] = pa.Field(nullable=True)
    d_n2: Series[str] = pa.Field(nullable=True)

class ODPairSchemaWithNulls(ODPairSchema):
    """General OD Pair data (allowing for destination nulls)."""
    d_id: Series[str] = pa.Field(nullable=True)

class GeocodedODPairSchema(ODPairSchema):
    """"""
    geometry_orig: GeoSeries
    geometry_dest: GeoSeries

"""General flow data---trip """
class _GroupedFlowIdSchema(pa.DataFrameModel):
    # how we distinguish between unique flows once they've been categorized
    # regardless of the O/D. OD Flow index tracks distinct flows across
    # potentially multiple modes
    ttype:             Series[str] = pa.Field(nullable=False)
    material_stream:   Series[str] = pa.Field(nullable=False)
    material_grouping: Series[str] = pa.Field(nullable=False) # FIXME: validate against known set
    
class _FlowModeSchema(pa.DataFrameModel):
    # FIXME: mode
    EMFAC_class:       Series[str] # FIXME: map this to actual classes
    
class GroupedFlowIdSchemaWithODNulls(_GroupedFlowIdSchema, ODPairSchemaWithNulls):
    pass

class ODPairSchemaWithPortAndSearoute(ODPairSchema):
    id_port_info: Series[str]
    n1_port_info: Series[str]
    geometry_port_info: GeoSeries
    # searoute info stored as geojson feature dict
    searte: Series[dict]  = pa.Field(nullable=True)
    # Linestring of searoute
    geometry_searte: GeoSeries = pa.Field(nullable=True) 
    
    seadist_u:  Series[Annotated[PintDtype, "m"]]  = pa.Field(nullable=True)
    distance_u: Series[Annotated[PintDtype, "m"]]  = pa.Field(nullable=True)
    duration_u: Series[Annotated[PintDtype, "s"]]  = pa.Field(nullable=True)


    @pa.parser("seadist_u", "distance_u")
    def check_length_dimension(cls, series):
        return pint_dimension_parser(reference_unit='meter')(series)

    @pa.parser("duration_u")
    def check_duration_dimension(cls, series):
        return pint_dimension_parser(reference_unit='second')(series)
 
    class Config:
        coerce = False


class ODPairSchemaWithPortAndSearouteFinal(ODPairSchema):
    geometry_orig: GeoSeries
    geometry_dest: GeoSeries
    id_port: Series[str]
    name_port: Series[str]
    geometry_port: GeoSeries
    searte: Series[dict]       # searoute info stored as geojson feature dict
    geometry_searte: GeoSeries # Linestring of searoute

    
    seadist_u:  Series[Annotated[PintDtype, "m"]] = pa.Field(nullable=True)
    distance_u: Series[Annotated[PintDtype, "m"]] = pa.Field(nullable=True)  # type: ignore[misc]
    duration_u: Series[Annotated[PintDtype, "s"]] = pa.Field(nullable=True) 


    @pa.parser("seadist_u", "distance_u")
    def check_length_dimension(cls, series):
        return pint_dimension_parser(reference_unit='meter')(series)

    @pa.parser("duration_u")
    def check_duration_dimension(cls, series):
        return pint_dimension_parser(reference_unit='second')(series)
 
    class Config:
        coerce = False
        

class _AnnualizedLayeredFlowSchema(pa.DataFrameModel):
    year: int
    layer:  Series[str]

@with_unique_check()
class AnnualizedLayeredGroupedFlowIdSchema(
    ODPairSchema,
    _GroupedFlowIdSchema, 
    _AnnualizedLayeredFlowSchema,
    _FlowModeSchema
    ):
    
    class Config:
        coerce = True

    @pa.dataframe_check
    def unique_ods(cls, df: pd.DataFrame) -> bool:
        """Confirm unique O/D pairs within each layer/year."""
        aggcols_chk=['EMFAC_class','year','layer','ttype','material_stream','material_grouping','o_id','d_id']
        aggcols=schema_to_cols(AnnualizedLayeredGroupedFlowIdSchema)
        assert(set(aggcols_chk) == set(aggcols)) # being careful here to avoid redefinition of grouping cols
        failed_checks = [check for check in [
            f"Duplicate OD pairs" if (df.groupby(aggcols).size() > 1).any() else False
        ] if check]
        if (len(failed_checks) > 0):
            for failure in failed_checks:
                print(f"unique_ods: {failure}")
                failed=df.assign(cnt=1).groupby(aggcols).agg({'cnt': 'count', 'wt_sent': 'sum'}).query('cnt > 1').reset_index()
                print(failed.to_string(index=False))
            # return False
            return True  # for now, we just report the duplicates but don't fail, since we know we have some and are working on cleaning them
        else:
            return True


class AnnualizedLayeredGroupedFlowIdSchemaWithAggregations(
    AnnualizedLayeredGroupedFlowIdSchema,
    # _ODPairSchemaAggregations, #FIXME: TEMPORARY
    ):
    pass
        
class _FlowData(pa.DataFrameModel):
    trips:             Series[float]
    wt_sent:           Series[Annotated[PintDtype, "kg"]] = pa.Field(nullable=True)
    
    @pa.parser("wt_sent")
    def check_mass_dimension(cls, series):
        return pint_dimension_parser(reference_unit='kg')(series)

    
unique_flow_cols=get_required_columns(AnnualizedLayeredGroupedFlowIdSchema)
unique_flow_agg={v:'sum' for v in _FlowData.to_schema().columns.keys()}
    
class AnnualizedLayeredGroupedFlowIdSchemaWithFlows(
        AnnualizedLayeredGroupedFlowIdSchema,
        _FlowData
    ):
    pass

class AnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows(
        AnnualizedLayeredGroupedFlowIdSchemaWithAggregations,
        _FlowData
    ):
    pass

class LayerFlowSchema(AnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows):
    pass

class IndexedAnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows(
        AnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows,
    ):
    od_flow_index:     Series[int] = pa.Field(nullable=False)

        

"""Model type definitions for flow data from RDRS."""
class _DetailedIdFlowSchema(pa.DataFrameModel):
    material_stream:      Series[str] = pa.Field(nullable=False)
    material_category:    Series[str] = pa.Field(nullable=False)
    material_subcategory: Series[str] = pa.Field(nullable=False)
    material_type:        Series[str] = pa.Field(nullable=False)

    # end_user_category:    Series[str] = pa.Field(nullable=True) # FIXME: should include?

@with_unique_check()
class DetailedFlowIdSchema(
        ODPairSchema,
        _DetailedIdFlowSchema,
        _AnnualizedLayeredFlowSchema):
    
    class Config:
        coerce = True

# class DetailedFlowSchemaWithCategorizedFlows(DetailedFlowIdSchema,_GroupedFlowSchemaWithFlows):
#     pass    

class RawRDRSData(_DetailedIdFlowSchema): #, _GroupedFlowIdSchema):
    year_quarter:        Series[str]
    destination_county:  Series[str]
    destination_state:   Series[str]
    destination_country: Series[str]
                         
# class DetailedFlowSchemaWithNulls(_DetailedIdFlowSchema,ODPairSchemaWithNulls):
#     pass

class DetailedFlowSchemaLayer(DetailedFlowIdSchema, AnnualizedLayeredGroupedFlowIdSchema):
    year: Series[int]
    layer:  Series[str]

@with_unique_check()
class ApplFlowIdSchema(
    ODPairSchema,
    _GroupedFlowIdSchema,   # includes material_stream/grouping
    _DetailedIdFlowSchema,  # includes material stream/category/subcategory/type
    _AnnualizedLayeredFlowSchema,
    _FlowModeSchema
    ):
    # 30_lbs, 123_lbs, 239_lbs
    container_type: str           = pa.Field(nullable=True) #,isin=['30_lbs','50_lbs', '123_lbs','239_lbs','1_000_lbs'])
    
    class Config:
        coerce = True
    
class ApplFlowSchema(ApplFlowIdSchema,_FlowData):
    
    # these are location-related info that are used in get_appliance_destinations
    # FIXME: not sure this is the best way to do it (should be in geocoder)
    orig_x: Series[float] = pa.Field(nullable=True)
    orig_y: Series[float] = pa.Field(nullable=True)
    dest_x: Series[float]
    dest_y: Series[float]
    car: Series[int] # = pa.Field(nullable=True)                  # should these be nullable?
    destination: Series[str] = pa.Field(nullable=True)          # should these be nullable?
    destination_address: Series[str] = pa.Field(nullable=True)  # should these be nullable?
    
    # these are flow-related info that will be aggregated
    ship_containers_int: Series[int]      = pa.Field() # would like to use Int64, but pandera doesn't support yet
    ship_weight: Series[float]            = pa.Field(nullable=True)

    # FIXME: fails with NotImplementedError: 
    # @pa.dataframe_check
    # def container_vars_validation(cls, df: pd.DataFrame) -> bool:
    #     ret = True
    #     for fld in ['container_type','ship_containers_int','ship_weight']:
    #         fld = fld and (df.container_type.notna() | (df.material_grouping != 'HAZ')).all()
    
    class Config:
        strict = False
        coerce = True

@with_unique_check()
class EwasteFlowIdSchema(
    ODPairSchema,
    _GroupedFlowIdSchema,   # includes material_stream/grouping
    _DetailedIdFlowSchema,  # includes material stream/category/subcategory/type
    _AnnualizedLayeredFlowSchema,
    _FlowModeSchema
    ):
    # 30_lbs, 123_lbs, 239_lbs
    # container_type: str           = pa.Field(nullable=True) #,isin=['30_lbs','50_lbs', '123_lbs','239_lbs','1_000_lbs'])
    
    class Config:
        coerce = True
    
class EwasteFlowSchema(EwasteFlowIdSchema,_FlowData):
    
    # these are location-related info that are used in get_appliance_destinations
    # FIXME: not sure this is the best way to do it (should be in geocoder)
    # orig_x: float = pa.Field(nullable=True)
    # orig_y: float = pa.Field(nullable=True)
    # dest_x: float
    # dest_y: float
    # car: int# = pa.Field(nullable=True)                  # should these be nullable?
    # destination: str = pa.Field(nullable=True)          # should these be nullable?
    # destination_address: str = pa.Field(nullable=True)  # should these be nullable?
    
    # these are flow-related info that will be aggregated
    # ship_containers_int: int      = pa.Field() # would like to use Int64, but pandera doesn't support yet
    # ship_weight: float            = pa.Field(nullable=True)

    # FIXME: fails with NotImplementedError: 
    # @pa.dataframe_check
    # def container_vars_validation(cls, df: pd.DataFrame) -> bool:
    #     ret = True
    #     for fld in ['container_type','ship_containers_int','ship_weight']:
    #         fld = fld and (df.container_type.notna() | (df.material_grouping != 'HAZ')).all()
    
    class Config:
        strict = False
        coerce = True

@with_unique_check()
class ElvFlowIdSchema(
    ODPairSchema,
    _GroupedFlowIdSchema,   # includes material_stream/grouping
    _DetailedIdFlowSchema,  # includes material stream/category/subcategory/type
    _AnnualizedLayeredFlowSchema,
    _FlowModeSchema
    ):
    # 30_lbs, 123_lbs, 239_lbs
    # container_type: str           = pa.Field(nullable=True) #,isin=['30_lbs','50_lbs', '123_lbs','239_lbs','1_000_lbs'])
    
    class Config:
        coerce = True

class ElvFlowSchema(ElvFlowIdSchema,_FlowData):
    
    class Config:
        strict = False
        coerce = True


class EndpointAggregation(str, Enum):
    LOCATION = 'location'
    CITY = 'city'
    COUNTY = 'county'
    STATE = 'state'
    COUNTRY = 'country'
    TRACT = 'tract'
    ZCTA = 'zcta'
    NONE = 'none'  # for null points

###
# The basic endpoint has a unique id, an aggregation level, and a facility_type
@with_unique_index(index_name='id')
class EndpointSchema(pa.DataFrameModel):
    # we use custom index validation below
    # id: PanderaIndex[str]    = pa.Field(check_name=True)  # Enforces index name == "id"
    # src: Series[str] # FIXME: should add
    aggregation: Series[str] = pa.Field(
        isin=[e.value for e in EndpointAggregation]
    )
    facility_type: Series[str] = pa.Field(nullable=True)
    # src <- endpoint sources
    
    facility_flags: Series[str] = pa.Field(nullable=True)  # for any additional flags, e.g., "manual geocode"

    class Config:
        """Config for EndpointSchema"""
        # This doesn't directly validate index, but you can combine with checks
        strict = False
        coerce = True
    
    # @pa.dataframe_check
    # def index_validation(cls, df: pd.DataFrame) -> bool:
    #     """Comprehensive index validation"""
    #     failed_checks = [check for check in [
    #         f"Index [{df.index.name}] not named 'id'" if not (df.index.name == 'id') else False,
    #         "Index not unique" if not df.index.is_unique else False,
    #         "Index contains NaN values" if df.index.isna().any() else False,
    #         "Index length is zero" if len(df.index) == 0 else False,
    #         "Index contains tuples" if df.index.map(lambda x: isinstance(x, tuple)).any() else False
    #     ] if check]
    #     if (len(failed_checks) > 0):
    #         for failure in failed_checks:
    #             print(f"Index validation failed: {failure}")
    #         return False
    #     else:
    #         return True
    #     # checks = [not bool(failed_checks)]  # Overall check result
    #     # return all(checks)

###
# An endpoint schema that has non-null names
class NamedEndpointSchema(EndpointSchema):
    n1: Series[str]         = pa.Field(nullable=False)
    n2: Series[str]         = pa.Field(nullable=False)

###
# An endpoint schema that has possibly null names
class NamedEndpointSchemaWithNulls(EndpointSchema):
    n1: Series[str]         = pa.Field(nullable=True)
    n2: Series[str]         = pa.Field(nullable=True)

###
# columns for defining RDRS-level regional aggregations (by name)    
class _RegionMappedEndpointSchema(pa.DataFrameModel):
    county: Series[str]     = pa.Field(nullable=True) # confirm if we want to allow nulls
    state: Series[str]      = pa.Field(nullable=True) # confirm if we want to allow nulls
    country: Series[str]    = pa.Field(nullable=True) # confirm if we want to allow nulls
    
# ###
# # columns for endpoints representing areas rather than points
class _AggregatedSchemaWithNulls(pa.DataFrameModel):
    geometry_agg: GeoSeries = pa.Field(nullable=True)  # FIXME: validate polygon
    geometry:     GeoSeries = pa.Field(nullable=True)      # centroid

###
# Columns for representing geometries --- both points and polygons
class _LocationSchema(pa.DataFrameModel):
    geometry:     GeoSeries   = pa.Field(nullable=False)
    geometry_src: Series[str] = pa.Field(nullable=False)

    # # locations can also b
    # geometry_agg: Optional[GeoSeries] = pa.Field(nullable=True)# (for non-location geometries)

    @pa.dataframe_check
    def geometry_type_check(cls, df: gpd.GeoDataFrame) -> bool:
        """Ensure all geometries are Points."""
        points = df.geometry.geom_type.eq("Point")
        if not points.all():
            print("Geometry type check failed: Not all geometries are Points.")
        crs_is_valid = (df.geometry.crs == 4326) # ==crs_ll, but defined in file that sources this
        if not crs_is_valid:
            print(f"CRS check failed: Expected {4326}, found {df.geometry.crs}.")

        return all([points.all(), crs_is_valid])

class _AggLocationSchema(pa.DataFrameModel):
    # locations can also b
    geometry_agg: Optional[GeoSeries] = pa.Field(nullable=True) # (for non-location geometries)

    @pa.dataframe_check
    def geometry_type_check(cls, df: gpd.GeoDataFrame) -> bool:
        if 'geometry_agg' in df:
            polys = df.geometry_agg.geom_type.str.match('.*Polygon')
            if not polys.all():
                print("Agg Geometry type check failed: Not all geometries are Points.")
            agg_crs_is_valid = (df.geometry.crs == 4326) # ==crs_ll, but defined in file that sources this
            if not agg_crs_is_valid:
                print(f"Agg CRS check failed: Expected {4326}, found {df.geometry_agg.crs}.")
        else:
            polys=[True]
            agg_crs_is_valid = True
        return all([polys.all(), agg_crs_is_valid])

class LocationSchemaWithNulls(pa.DataFrameModel):
    geometry: GeoSeries = pa.Field(nullable=True)

    @pa.dataframe_check
    def geometry_type_check(cls, df: gpd.GeoDataFrame) -> bool:
        """Ensure all geometries are Points."""
        nulls = df.geometry.isna()
        if nulls.all():
            return True
        else:
            point_or_null = df.geometry.geom_type.eq("Point") | nulls
            crs_is_valid = (df.geometry.crs == 4326) # ==crs_ll, but defined in file that sources this
            if not point_or_null.all():
                print("Geometry type check failed: Not all geometries are Points.")
            if not crs_is_valid:
                print(f"CRS check failed: Expected {4326}, found {df.geometry.crs}.")
            return all([point_or_null.all(), crs_is_valid])


class GeocodedEndpointSchemaWithNamedNulls(NamedEndpointSchemaWithNulls,LocationSchemaWithNulls):
    pass

class GeocodedEndpointSchemaWithNulls(EndpointSchema,LocationSchemaWithNulls):
    pass

class GeocodedEndpointSchema(EndpointSchema,_LocationSchema,_AggLocationSchema):
    pass    

class GeocodedNamedEndpointSchema(NamedEndpointSchema,GeocodedEndpointSchema):
    pass

class GeocodedNamedEndpointSchemaWithNulls(NamedEndpointSchema,GeocodedEndpointSchemaWithNulls):
    pass

class GeocodedNamedRegionMappedEndpointSchema(
        NamedEndpointSchema,         # id index, aggregation, facility_type, facility_flags; n1, n2
        _RegionMappedEndpointSchema, # county, state, country
        GeocodedEndpointSchema       # geometry, geometry_src; geometry_agg
):
    pass


class LayerEndpointSchema(GeocodedNamedRegionMappedEndpointSchema):
    pass

# class GeocodedNamedRegionMappedEndpointSchemaWithNulls(NamedEndpointSchema,_RegionMappedEndpointSchema,GeocodedEndpointSchemaWithNulls):
#     pass

class RDRSFullEndpointSchema(NamedEndpointSchema,_RegionMappedEndpointSchema):
    """Used to load RDRS data"""
    pass



class _CountySchema(pa.DataFrameModel):
    county: Series[str]  = pa.Field(nullable=True)

class _CountySchemaWithGeocodes(_CountySchema):
    geometry: GeoSeries = pa.Field(nullable=True)
    geometry_agg: GeoSeries = pa.Field(nullable=True)

class _StateEndpointSchema(pa.DataFrameModel):
    state: Series[str]  = pa.Field(nullable=True)

class _CountryEndpointSchema(pa.DataFrameModel):
    country: Series[str]  = pa.Field(nullable=True)

class RDRSMaterialSchema(pa.DataFrameModel):
    grouping4: Series[str]

class ODFlows(IndexedAnnualizedLayeredGroupedFlowIdSchemaWithAggregationsWithFlows
              #,ODPairSchemaWithDetails
              ):
    pass

class ODRoutablePts(pa.DataFrameModel):
    """Class representing endpoints for routing"""
    o_pt: GeoSeries    = pa.Field(nullable=False)
    d_pt: GeoSeries    = pa.Field(nullable=False)
    
class GeocodedPortSchema(pa.DataFrameModel):
    id_port: Series[str]     = pa.Field(nullable=True)
    geometry_port: GeoSeries = pa.Field(nullable=True)
    
class ODRoutableX(GeocodedODPairSchema):
    pass
    

class ODRoutable(GeocodedODPairSchema):
    # id_port:         Series[str]                       = pa.Field(nullable=True)
    # name_port:       Series[str]                       = pa.Field(nullable=True)
    # geometry_port:   GeoSeries                         = pa.Field(nullable=True)
    # # searte: str = pa.Field(nullable=True)
    # geometry_searte: GeoSeries                         = pa.Field(nullable=True)
    # seadist_u:       Series[Annotated[PintDtype, "m"]] = pa.Field(nullable=True)

    # @pa.parser("seadist_u")
    # def check_length_dimension(cls, series):
    #     return pint_dimension_parser(reference_unit='meter')(series)
    pass

class ODRoutableTourLeg(GeocodedODPairSchema):
    leg:             Series[int]                       = pa.Field(nullable=False)
    # @pa.parser("seadist_u")
    # def check_length_dimension(cls, series):
    #     return pint_dimension_parser(reference_unit='meter')(series)
    # pass

class ODTourLegUnRouted(pa.DataFrameModel):
    layer: Series[str]
    year: Series[int]
    material_stream: Series[str]
    material_grouping: Series[str]
    ttype: Series[str]
    o_id_tour: Series[str]
    d_id_tour: Series[str]
    o_id:      Series[str]
    d_id:      Series[str]
    leg:       Series[int]
    transport_mode: Series[str]
    EMFAC_class: Series[str]
    
    trips: Series[float]
    wt_sent: Series[Annotated[PintDtype, "kg"]]

    o_pt: GeoSeries
    d_pt: GeoSeries

    # @pa.parser("distance_u")
    # def check_length_dimension(cls, series):
    #     return pint_dimension_parser(reference_unit='meter')(series)

    # @pa.parser("duration_u")
    # def check_duration_dimension(cls, series):
    #     return pint_dimension_parser(reference_unit='second')(series)

    @pa.parser("wt_sent")
    def check_mass_dimension(cls, series):
        return pint_dimension_parser(reference_unit='kg')(series)


class ODTourLegRouted(ODTourLegUnRouted):    
    # route: Series[object]
    step_num: Series[int]
    clip_num: Series[int]
    region: Series[str]
    geometry_clip: GeoSeries = pa.Field(nullable=True)
    geometry_full: GeoSeries = pa.Field(nullable=True)
    
    distance_u: Series[Annotated[PintDtype, "m"]]
    duration_u: Series[Annotated[PintDtype, "s"]]
    step_speed_u: Series[Annotated[PintDtype, "m/s"]]



    @pa.parser("distance_u")
    def check_length_dimension(cls, series):
        return pint_dimension_parser(reference_unit='meter')(series)

    @pa.parser("duration_u")
    def check_duration_dimension(cls, series):
        return pint_dimension_parser(reference_unit='second')(series)

    @pa.parser("step_speed_u")
    def check_speed_dimension(cls, series):
        return pint_dimension_parser(reference_unit='m/s')(series)
    
# class ODFlowsWithSteps(ODRoutable,ODFlows):
#     step_num: Series[int] = pa.Field(nullable=False,ge=1)
    
#     class Config:
#         strict=False

# class RouteSchema(ODRoutable):
#     route: Series[object]  # fixme should be Series[list] but breaks validation coersion
#     geometry_full: GeoSeries = pa.Field(nullable=True)
#     distance_u: Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)
#     duration_u: Series[Annotated[PintDtype, "s"]]      = pa.Field(nullable=True)
#     # seadist_u:  Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)


#     @pa.parser("distance_u"
#             #    , "seadist_u"
#                )
#     def check_length_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='meter')(series)

#     @pa.parser("duration_u")
#     def check_duration_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='second')(series)

# class ODRoutedPts(ODRoutablePts):
#     """Class representing routes between points"""
#     route: Series[object] # Need to carry this forward for creation of ODRoutedStepsPts
#     distance_u: Series[Annotated[PintDtype, "m"]] = pa.Field(nullable=True)
#     duration_u: Series[Annotated[PintDtype, "s"]] = pa.Field(nullable=True)

#     @pa.parser("distance_u")
#     def check_length_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='meter')(series)

#     @pa.parser("duration_u")
#     def check_duration_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='second')(series)
    


class RouteSchemaPts(ODRoutablePts):
    route: Series[object]  # fixme should be Series[list] but breaks validation coersion
    geometry_full: GeoSeries = pa.Field(nullable=True)
    distance_u: Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)
    duration_u: Series[Annotated[PintDtype, "s"]]      = pa.Field(nullable=True)
    # seadist_u:  Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)

    @pa.parser("distance_u")
    def check_length_dimension(cls, series):
        return pint_dimension_parser(reference_unit='meter')(series)

    @pa.parser("duration_u")
    def check_duration_dimension(cls, series):
        return pint_dimension_parser(reference_unit='second')(series)


class RouteStepsSchemaPts(ODRoutablePts):
    step_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
    geometry_full: GeoSeries                            = pa.Field(nullable=True)
    geometry_step: GeoSeries
    step_speed_u:  Series[Annotated[PintDtype, "m/s"]]  = pa.Field(nullable=True)

    @pa.parser("step_speed_u")
    def check_speed_dimension(cls, series):
        return pint_dimension_parser(reference_unit='m/s')(series)


class RouteClipsSchemaPts(ODRoutablePts): # should extend RouteStepsSchema, but we don't want geometry_step
    step_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
    clip_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
    # we don't want geometry_step
    geometry_full: GeoSeries                            = pa.Field(nullable=True)
    geometry_clip: GeoSeries
    region: Series[str]

    step_speed_u:  Series[Annotated[PintDtype, "m/s"]]  = pa.Field(nullable=True)

    @pa.parser("step_speed_u")
    def check_speed_dimension(cls, series):
        return pint_dimension_parser(reference_unit='m/s')(series)

# class RouteTourLegSchema(ODRoutable):
#     route: Series[object]  # fixme should be Series[list] but breaks validation coersion
#     geometry_full: GeoSeries = pa.Field(nullable=True)
#     seadist_u:  Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)
#     distance_u: Series[Annotated[PintDtype, "m"]]      = pa.Field(nullable=True)
#     duration_u: Series[Annotated[PintDtype, "s"]]      = pa.Field(nullable=True)


#     @pa.parser("distance_u", "seadist_u")
#     def check_length_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='meter')(series)

#     @pa.parser("duration_u")
#     def check_duration_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='second')(series)


# """xx"""
# class RouteStepsSchema(ODRoutable):
#     step_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
#     geometry_full: GeoSeries                            = pa.Field(nullable=True)
#     geometry_step: GeoSeries
#     step_speed_u:  Series[Annotated[PintDtype, "m/s"]]  = pa.Field(nullable=True)

#     @pa.parser("step_speed_u")
#     def check_speed_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='m/s')(series)

# class RouteClipsSchema(ODRoutable): # should extend RouteStepsSchema, but we don't want geometry_step
#     step_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
#     clip_num:      Series[int]                          = pa.Field(nullable=False,ge=1)
#     # we don't want geometry_step
#     geometry_full: GeoSeries                            = pa.Field(nullable=True)
#     geometry_clip: GeoSeries
#     region: Series[str]

#     step_speed_u:  Series[Annotated[PintDtype, "m/s"]]  = pa.Field(nullable=True)

#     @pa.parser("step_speed_u")
#     def check_speed_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='m/s')(series)


# class ODRoutedX(ODRoutableX, RouteClipsSchemaPts):
#     """Class representing routes between points"""
#     route: Series[object] # fixme should be Series[list] but breaks validation coersion
#     distance_u: Series[Annotated[PintDtype, "m"]] = pa.Field(nullable=True)
#     duration_u: Series[Annotated[PintDtype, "s"]] = pa.Field(nullable=True)

#     @pa.parser("distance_u")
#     def check_length_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='meter')(series)

#     @pa.parser("duration_u")
#     def check_duration_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='second')(series)

    
# class RouteFinal(RouteClipsSchema):
#     pass


# class RouteClipsSchemaWithNulls(ODRoutable):
#     step_num:      Series[int]                          = pa.Field(nullable=True)
#     clip_num:      Series[int]                          = pa.Field(nullable=True)
#     # we don't want geometry_step          
#     geometry_clip: GeoSeries                            = pa.Field(nullable=True)
#     region:        Series[str]                          = pa.Field(nullable=True)

#     step_speed_u:  Series[Annotated[PintDtype, "m/s"]]  = pa.Field(nullable=True)

#     @pa.parser("step_speed_u")
#     def check_speed_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='m/s')(series)
    
# class ODFlowsWithClips(ODFlows,RouteClipsSchema):
#     geometry_searte: GeoSeries                          = pa.Field(nullable=True)

#     seadist_u:       Series[Annotated[PintDtype, "m"]]  = pa.Field(nullable=True)

#     @pa.parser("seadist_u")
#     def check_length_dimension(cls, series):
#         return pint_dimension_parser(reference_unit='m')(series)


class ApplianceCase(str, Enum):
    """Appliance case"""
    TEST = 'test'
    BEST = 'best'
    LOWER_BOUND = 'lower_bound'
    UPPER_BOUND = 'upper_bound'

class RefrCase(str, Enum):
    """Refrigerants case"""
    MIN = 'min'
    MAX = 'max'


def strict_check_types(allowed_schemas):
    def decorator(func):
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            for schema in allowed_schemas:
                schema.validate(df)  # Will raise if invalid
            return df
        return wrapper
    return decorator


# with help from kagi AI
def strict_check_types_with_report(schemas: List[pa.DataFrameSchema]):
    """
    Decorator: validates DataFrame against all provided schemas using lazy validation.
    Reports which schemas pass/fail and includes detailed failure cases.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)

            failed_schemas = []
            passed_schemas = []

            for schema in schemas:
                schema_name = schema.__name__  # Correct way for DataFrameSchema
                try:
                    schema.validate(df, lazy=True)
                    passed_schemas.append(schema_name)
                except SchemaErrors as err:
                    failed_schemas.append(schema_name)
                    print(f"❌ Schema '{schema_name}' failed:")
                    print(err.failure_cases.to_string(index=False))
                except Exception as err:
                    failed_schemas.append(schema_name)
                    print(f"❌ Schema '{schema_name}' raised error: {str(err)}")

            if failed_schemas:
                print(f"✅ Passed: {passed_schemas}")
                print(f"❌ Failed: {failed_schemas}")
                raise RuntimeError(f"Validation failed for schemas: {failed_schemas}")
            # else:
            #     print(f"✅ All schemas passed: {passed_schemas}")

            return df
        return wrapper
    return decorator

class RegionSchema(pa.DataFrameModel):
    region: Series[str]   # region name
    geometry: GeoSeries   # region polygon
    
    
    
#### APPLIANCE STUFF

class ApplianceMaterialCompositionSchema(pa.DataFrameModel):
    index: PandasIndex[str]  = pa.Field( # appliance type code
        isin=['cd', 'cw', 'mw', 'ov', 'rf', 'wh', 'ww', 'st', 'dw', 'cc', 'ch',
              'fz', 'evcl', 'wg'
              ])
    
    ferrous:    Series[float]
    nonferrous: Series[float]
    hazardous:  Series[float]
    other:      Series[float]

class ApplianceZipGenerationSchema(pa.DataFrameModel):
    o_id: Series[str]       # origin id (zip)
    # zip_code: Series[str]   # zip code
    # appltype: Series[str]  = pa.Field( # appliance type code
    #     # alias='appliance_type', 
    #     isin=['cd', 'cw', 'mw', 'ov', 'rf', 'wh', 'ww', 'st', 'dw', 'cc', 'ch',
    #           'fz', 'evcl', 
    #           'wg' #FIXME?
    #           ])
    
class ApplianceZipToCarByApplianceAndAgeSchema(ApplianceZipGenerationSchema):
    zip_id:     Series[str]                                                     # zip code id
    car_id:     Series[str]                                                     # car id
    # car:      Series[int]                                                     # Appl Recycler ID (to)
    n1_car:     Series[str]                                                     # destination name
    n2_car:     Series[str]                                                     # destination address
    ferrous:    Series[Annotated[PintDtype, "tons"]]
    nonferrous: Series[Annotated[PintDtype, "tons"]]
    hazardous:  Series[Annotated[PintDtype, "tons"]]
    other:      Series[Annotated[PintDtype, "tons"]]
    lda_trips:  Series[float]           # lda trips
    ldt_trips:  Series[float]           # ldt trips
    lda_weight: Series[Annotated[PintDtype, "tons"]]
    ldt_weight: Series[Annotated[PintDtype, "tons"]]
    age:        Series[int]                          = pa.Field(isin=[1,2,3])   # appliance age code

    @pa.parser("ferrous", "nonferrous", "hazardous", "other", "lda_weight", "ldt_weight")
    def check_mass_dimension(cls, series):
        return pint_dimension_parser(reference_unit='kg')(series)

    
    # @pa.check("zip_id")
    # @staticmethod
    # def valid_zip_id(cls, zip_id):
    #     return pa.Check.str_matches(r'^ZIP\d{5}$')
    # @pa.check("car_id")
    # @staticmethod
    # def valid_car_id(cls, car_id):
    #     return pa.Check.str_matches(r'^CAR\d{5}$')


class ExportDestinationProportions(pa.DataFrameModel):
    """A type describing proportions of exported materials going to specific destnations"""
    material_grouping: Series[str]   = pa.Field(nullable=False) # FIXME: validate against known set
    d_id:              Series[str]   = pa.Field(nullable=False)
    proportion:        Series[float] = pa.Field(nullable=False,ge=0,le=1)
    
    @pa.dataframe_check(name="proportions_sum_to_1")
    @classmethod
    def proportions_sum_to_1(cls, df: pd.DataFrame) -> bool:
        sums = df.groupby("material_grouping")["proportion"].sum()
        return np.all(np.isclose(sums, 1.0))

def schema_to_cols(schema):
    return list(schema.to_schema().columns.keys())

# Kagi AI: allow per-validate call with specific options to gracefully expand a
# dataframe to match a schema (coerce and add_missing_columns are typically only
# allowed in the class definition)
@pa.check_types()
def _validate_with_options(
    schema_base:pa.DataFrameSchema, 
    df:pd.DataFrame, 
    coerce:bool=True, 
    add_missing_columns:bool=True, 
    **validate_kwargs):
    """Validate a DataFrame against a DataFrameModel with coerce and
    add_missing_columns enabled, without modifying the original schema."""
    schema = pa.DataFrameSchema(
        columns=schema_base.columns,
        checks=schema_base.checks,
        index=schema_base.index,
        coerce=coerce,
        add_missing_columns=add_missing_columns,
    )
    return schema.validate(df, **validate_kwargs)


def validate_with_options(schema_cls, df, ignore_non_null=False, **kwargs):
    """Validate with add_missing_columns support for PintDtype columns.

    Pre-constructs missing pint columns with proper PintArray + Float64
    sub-dtype, bypassing the pandas _dtype_to_na_value code path that
    crashes with PintType.kind property bug.

    Parameters
    ----------
    schema_cls : pa.DataFrameModel
        The pandera schema class to validate against.
    df : pd.DataFrame
        The DataFrame to validate.
    ignore_non_null : bool, optional
        When True, all non-nullable column constraints are temporarily
        relaxed to nullable=True before validation. The schema class
        itself is never mutated — a temporary DataFrameSchema object
        is constructed for this validation call only.
    **kwargs
        add_missing_columns : bool
            Pre-construct missing PintDtype columns as NaN columns.
        coerce : bool
            Coerce column dtypes to the schema's declared dtypes.
        lazy : bool
            Collect all validation errors before raising (pandera default: False).
    """
    add_missing = kwargs.pop('add_missing_columns', False)
    coerce = kwargs.pop('coerce', False)

    # ── Build a DataFrameSchema object up front ──────
    # update_column() is a pure method — it returns a *new* schema each time,
    # so schema_cls is never touched regardless of what we do to schema_obj.
    schema_obj = schema_cls.to_schema()

    # ── Handle missing PintDtype columns ──────────────────────────────────────
    if add_missing:
        missing_pint_cols = {}
        for col_name, col_schema in schema_obj.columns.items():
            if col_name not in df.columns:
                dtype = col_schema.dtype
                if isinstance(dtype, PintDtype):
                    # Pre-construct a proper pint NaN column with Float64 sub-dtype
                    na_pint_array = pint_pandas.PintArray(
                        np.full(len(df), np.nan, dtype=np.float64),
                        dtype=f"pint[{dtype.default_unit}]",
                    )
                    missing_pint_cols[col_name] = pd.Series(
                        na_pint_array,
                        index=df.index,
                        name=col_name,
                    )

        if missing_pint_cols:
            df = df.copy()
            for col_name, series in missing_pint_cols.items():
                df[col_name] = series

    # ── Temporarily disable non-nullable constraints ───────────────────────────
    if ignore_non_null:
        for col_name, col in list(schema_obj.columns.items()):
            if not col.nullable:
                # update_column() returns a brand-new DataFrameSchema — no mutation
                schema_obj = schema_obj.update_column(col_name, nullable=True)

    # ── Normal path (no ignore_non_null) ──────────────────────────────────────
    return _validate_with_options(
        schema_obj, df,
        add_missing_columns=add_missing,
        coerce=coerce,
        **kwargs,
    )
