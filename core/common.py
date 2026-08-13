from typing import List

import core.units # defines application-wide registry and custom units
import pandas as pd
import geopandas as gpd
import shapely.geometry as shgeo
import numpy as np
import os
import config
import janitor
from config.settings import settings
import importlib


importlib.reload(config)
from pint import UnitRegistry
from IPython.display import display
import osrm
import pandera.pandas as pa
from cachier import cachier
import cachier.config as cachier_config
# import logging
# logger = logging.getLogger(__name__)

# inline filter
def filt(df, f):
    return df[f]
pd.DataFrame.filt = filt

def strip_col_names(df):
    # use rename(columns=dict) to support inline
    return df.rename(
        columns=dict(zip(df.columns, [col.strip() for col in df.columns]))
    )
pd.DataFrame.strip_col_names = strip_col_names

def add_col_suffix(df, suffix):
    # use rename(columns=dict) to support inline
    return df.rename(
        columns=dict(zip(df.columns, [col + suffix for col in df.columns]))
    )
pd.DataFrame.add_col_suffix = add_col_suffix

def add_col_prefix(df, prefix):
    # use rename(columns=dict) to support inline
    return df.rename(
        columns=dict(zip(df.columns, [prefix + col for col in df.columns]))
    )
pd.DataFrame.add_col_prefix = add_col_prefix

def add_column_if_missing(df, column_name, default_value):
    if column_name not in df.columns:
        df = df.assign(**{column_name: default_value})
    return df
pd.DataFrame.add_column_if_missing = add_column_if_missing
pd.DataFrame.add_col_if_missing = add_column_if_missing

def drop_columns(df, cols=None, regex=None):
    """
    Drop columns from a DataFrame by name or regex pattern.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        cols (list, optional): List of column names to drop.
        regex (str, optional): Regex pattern to match column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    if regex:
        cols = df.columns[df.columns.str.contains(regex, regex=True)].tolist()
    return df.drop(columns=cols)

pd.DataFrame.drop_columns = drop_columns

def keep_columns(df, cols=None, regex=None):
    """
    Drop columns from a DataFrame by name or regex pattern.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        cols (list, optional): List of column names to drop.
        regex (str, optional): Regex pattern to match column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    if regex:
        cols = df.columns[df.columns.str.contains(regex, regex=True)].tolist()
    return df[cols]

pd.DataFrame.keep_columns = keep_columns

def flatten_columns(df):
    """
    Flatten the columns of a DataFrame by joining multi-level columns with an underscore.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: DataFrame with flattened column names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns]
    return df
pd.DataFrame.flatten_columns = flatten_columns

# use of is_bad_geometry deals with the isna warning, so we turn it off
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
def is_bad_geometry(gds):
    return gds.isna() | gds.is_empty
gpd.GeoSeries.is_bad_geometry = is_bad_geometry

def switch_geometry(gdf, col):
    """
    A function to temporarily switch the active geometry in a gdf by renaming the geometry columns.
    
    This is needed if you're using plotnine for geom plots with non-standard geometry column names
    as it assumes `geometry` is the column name.
    """
    return gdf.rename(columns={'geometry': f'hold_geometry_for_{col}'}).rename(columns={col: 'geometry'}).set_geometry('geometry')
gpd.GeoDataFrame.switch_geometry = switch_geometry

def in_chain_immut(df, f):
    f(df)
    return df
pd.DataFrame.in_chain_immut = in_chain_immut

hidedisp = False
def wdisplay(df):
    if not hidedisp:
        with pd.option_context('display.max_columns', None):
            display(df)
def ldisplay(df):
    if not hidedisp:
        with pd.option_context('display.max_rows', None):
            display(df)
def wldisplay(df):
    if not hidedisp:
        with pd.option_context('display.max_columns', None):
            with pd.option_context('display.max_rows', None):
                display(df)

# r style case_when: https://stackoverflow.com/a/73973563
def case_when(*args, **kwargs):
    return np.select(
        condlist=args[::2],
        choicelist=args[1::2],
        default=kwargs.get('default', pd.NA)
    )

@pa.check_types
def shapely_point_to_osrm_point(pt: shgeo.Point) -> osrm.Point:
    ll = list(pt.coords)
    if len(ll) > 0:
        return osrm.Point(*ll[0])
    else:
        return None

### PATHS
def flatten_lists(ll):
    """Flatten a nested list of lists into a single list
       See: https://realpython.com/python-flatten-list/#:~:text=it%20a%20try!-,Using%20sum()%20to%20Concatenate%20Lists,-The%20built%2Din
    """
    return sum(ll, [])

def cpath(d, p=None, environment=None, section='model_paths'):
    """Converts a configuration-defined path to a safe os-independent path"""
    envsec=section
    if environment is not None:
        envsec=f'{environment}.{envsec}'
    return os.path.join(*(
        flatten_lists([i.split("/") for i in [settings[envsec][d], p] if i is not None])
    ))



# Define a function to convert to int, returning NaN for failures
# from chatgpt
def safe_int_convert(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return np.nan

def check_cache_exists(name,cache_format='pickle'):
    cache_file = cpath('model_cache_dir', f'{name}.{cache_format}')
    return os.path.exists(cache_file)


cachier_config.set_global_params(
    cache_dir=cpath('model_cache_dir','.')    
)


def read_material_mapping(ff=cpath('processed_data','MATCATS_MaterialMappings24037.xlsx')):
    return (
        pd.concat([
            pd.read_excel(ff, sheet_name='Sheet1').clean_names(),
            pd.DataFrame(
                {'material_category':['HAZ'],'material_subcategory':['HAZ'],'material_type':['HAZ'],'grouping4':['HAZ']
                     ,'include':[True]})
            ],
            ignore_index=True
        )
    )
    
def material_mapping_to_dict(mm):
    return dict(zip(mm.material_category+":"+mm.material_subcategory+":"+mm.material_type, mm.grouping4))



# use suppress_warning as function decorator or context manager
from contextlib import contextmanager
import warnings

@contextmanager
def suppress_warning(message='', category=UserWarning):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message, category)
        yield


@pa.check_types()
def disable_index_checks(dfl: List[pd.DataFrame]) -> List[pd.DataFrame]:
    return [df.set_flags(allows_duplicate_labels=True) for df in dfl ]

# check if obj is a List[str]
def is_list_str_instance(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)
