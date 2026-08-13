### UNIT STUFF
# units for conversions
import pint

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)  # makes sure pandas uses the same registry
# make this last!
import pint_pandas
import pandas as pd
pint_pandas.PintArray.ureg = ureg

m = ureg('1 m')
mi = ureg('1 mi')
hr = ureg('1 hr')
sec = ureg('1 sec')


def create_unit_assignment(units):
    return {
        col: lambda x, col=col, unit=unit: x[col].astype(f'pint[{unit}]')
        for col,unit in units.items()
    }


def create_unit_transformation(units):
    return {
        col: lambda x, col=col, unit=unit: x[col].pint.to(unit)
        for col,unit in units.items()
    }


def assign_units(df,units):
    adict = create_unit_assignment(units)
    return df.assign(**adict)


def transform_units(df,units):
    adict = create_unit_transformation(units)
    return df.assign(**adict)


import re
def display_with_units(df, display_dict):
    """
    Display the DataFrame with explicit units for specific columns.
    """
    display_df = df.copy()

    return (
        display_df.pipe(transform_units,display_dict)
        .pipe(
            lambda x: x.rename(columns={
                k:f'{re.sub(r'_u$','',k)}_{x[k].pint.units}' 
                for k in x.columns 
                if k in display_dict}
            )
        )
    )


def deunitize(df, *cols):
    return df.copy().assign(**{col: df[col].pint.magnitude for col in cols})

def safe_deunitize_all(df):
    return deunitize(df,*[c for c in df.columns if isinstance(df[c].dtype, pint_pandas.PintType)])
    
    

def ensure_pint_dtypes(df, pint_cols, unit_map):
    """Ensure pint columns have the correct dtype even if they're all NaN."""
    for c in pint_cols:
        if not isinstance(df[c].dtype, pint_pandas.PintType):
            df[c] = pd.array([float('nan')] * len(df), dtype='Float64').astype(
                f"pint[{unit_map[c]}]"
            )
    return df


def safe_describe(series: pd.Series) -> pd.Series:
    """Describe a pandas Series, stripping pint units to magnitudes if necessary.

    Calling ``.describe()`` directly on a pint-typed Series raises a TypeError
    because pandas cannot aggregate the extension dtype.  This function detects
    pint columns via the presence of a ``.units`` attribute on the dtype and
    extracts the underlying float magnitudes before describing.

    Args:
        series: A pandas Series, optionally carrying a pint extension dtype.

    Returns:
        Series of descriptive statistics (count, mean, std, min, quartiles,
        max).  Values are in the series' native pint units if applicable.
    """
    if hasattr(series.dtype, 'units'):  # pint dtype check
        return series.pint.magnitude.describe()
    return series.describe()



def unify_pint_units(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Align pint units across a list of DataFrames.

    1. Find columns common to all DataFrames.
    2. For each common column, take the unit from the first DataFrame
       that has a pint unit for that column.
    3. Apply transform_units(common_units, ...) to each DataFrame.

    Returns a new list of DataFrames with unified units.
    """
    # Columns present in every DataFrame
    common_cols = set.intersection(*(set(df.columns) for df in dfs))

    # Build common_units: first unit wins for each column
    common_units: dict[str, pint.Unit] = {}
    for col in common_cols:
        for df in dfs:
            units = get_pint_units(df[[col]])
            if units:  # non-empty → column has a pint unit
                common_units[col] = next(iter(units.values()))
                break

    # Apply transform_units to each DataFrame
    return [transform_units(df, common_units) for df in dfs]


def get_pint_units(df):
    """Returns a dict of the pint column units in df for use with apply_units"""
    pint_cols=pint_cols=[a for (a,b) in df.dtypes.items() if str(b).startswith('pint[')]
    return{col:str(df[col].pint.units) for col in pint_cols}