# ============================================================
# Utility
# ============================================================


import pandas as pd
import pandera.pandas as pa
from pandera.extensions import register_check_method


import logging
logger = logging.getLogger(__name__)

from typing import Callable, List, Optional


def get_required_columns(schema_class: type[pa.DataFrameModel]) -> List[str]:
    """Get list of required (non-nullable) column names from schema."""
    schema = schema_class.to_schema()
    return [
        col_name
        for col_name, col_schema in schema.columns.items()
        # if not col_schema.nullable
    ]


# KAGI AI:
def with_unique_check(columns: Optional[List[str]] = None):
    """
    Class decorator that adds uniqueness check on the class's own columns.

    Args:
        columns: Specific columns to check. If None, checks all columns
                defined in the decorated class (including inherited ones).
    """
    def decorator(cls):
        # Capture columns from this class at decoration time
        if columns is not None:
            captured_cols = columns
        else:
            captured_cols = get_required_columns(cls)

        @pa.dataframe_check
        def _check_unique_rows_over_columns(check_cls, df: pd.DataFrame) -> bool:
            """Check uniqueness on captured columns."""
            cols = [c for c in captured_cols if c in df.columns]

            if not cols:
                return True

            is_unique = ~df[cols].duplicated().any()

            if not is_unique:
                dups = df[df[cols].duplicated(keep=False)]
                logger.error(f"\n⚠️  Found {len(dups)} duplicate rows")
                logger.error(f"   Columns checked: {cols}")
                logger.error(f"\n{dups[cols].sort_values(cols).head(10).to_string()}")

            return is_unique

        # Add check to the class
        cls.check_unique = _check_unique_rows_over_columns
        return cls

    return decorator

def with_unique_index(index_name: Optional[str] = None):
    """
    Class decorator that adds comprehensive index validation:
        - Uniqueness
        - Absence of NaN values
        - Non-emptiness
        - No tuple values
        - Optionally, the index name

    Args:
        index_name: Expected index name to validate against. If None,
                    the index name is not checked.
    """
    def decorator(cls):

        @pa.dataframe_check
        def _check_unique_index(check_cls, df: pd.DataFrame) -> bool:
            """Comprehensive index validation check."""
            failed_checks = [check for check in [
                f"Index [{df.index.name}] not named '{index_name}'"
                    if index_name is not None and df.index.name != index_name else False,
                "Index not unique"
                    if not df.index.is_unique else False,
                "Index contains NaN values"
                    if df.index.isna().any() else False,
                "Index length is zero"
                    if len(df.index) == 0 else False,
                "Index contains tuples"
                    if df.index.map(lambda x: isinstance(x, tuple)).any() else False,
            ] if check]

            if failed_checks:
                for failure in failed_checks:
                    logger.error(f"⚠️  Index validation failed: {failure}")
                return False

            return True

        cls.check_unique_index = _check_unique_index
        return cls

    return decorator