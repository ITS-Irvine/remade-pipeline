from IPython.display import display
import pandas as pd
import numpy as np
import logging
from core.common import suppress_warning
from core.units import deunitize, safe_describe
from utils.logging_config import redirect_stdout_to_logging

from layers.base import ModelLayer, mask_for_od_totals

logger = logging.getLogger(__name__)

def validate_emissions_dataframe(
    emiss: pd.DataFrame,
    layers: list[ModelLayer]
    ) -> None: # FIXME: type
    # FIXME: pull expected cols from schema
    expected_cols = [
        'od_flow_index', 'EMFAC_class', 'year', 'layer', 'ttype',
        'material_stream', 'material_grouping', 'o_id', 'd_id', 'o_n1', 'o_n2',
        'd_n1', 'd_n2', 'geometry_orig', 'o_geometry_src', 'geometry_dest',
        'd_geometry_src', 'o_country', 'd_country', 'o_state', 'd_state',
        'o_county', 'd_county', 'o_facility_type', 'd_facility_type', 'trips',
        'wt_sent', 'step_num', 'region', 'geometry_orig_dist',
        'geometry_port', 'id_port', 'clip_num', 'geometry_clip', 'name_port',
        'geometry_searte', 'seadist_u', 'geometry_dest_dist', 'route',
        'geometry_full', 'duration_u', 'distance_u', 'step_speed_u',
        'speed_min_u', 'speed_max_u', 'clip_distance', 'speed_bin_u', 'vmt_u',
        'calendar_year', 'model_year', 'nox_runex_u', 'pm2_5_runex_u',
        'pm10_runex_u', 'co2_runex_u', 'ch4_runex_u', 'n2o_runex_u',
        'rog_runex_u', 'tog_runex_u', 'co_runex_u', 'sox_runex_u',
        'nh3_runex_u', 'pm10_pmbw_u', 'pm2_5_pmbw_u', 'fuel_consumption_r_u',
        'energy_consumption_r_u', 'total_vmt_u', 'fuel_consumption_u',
        'energy_consumption_u', 'emiss_pm25_u', 'emiss_pm10_u', 'emiss_nox_u',
        'emiss_co_u', 'emiss_co2_u', 'emiss_ch4_u', 'emiss_n2o_u',
        'emiss_ghg_u'
    ]

    # ── Emissions Completeness Assertion
    # Verify that every routed flow with valid geometries and a region
    # has computed fuel consumption. Missing values indicate a bug.
    with suppress_warning('GeoSeries.notna', UserWarning):
        assert(
            emiss.filt(lambda x: (
                x.region.notna()
                & x.geometry_orig.notna() & ~x.geometry_orig.is_empty
                & x.geometry_dest.notna() & ~x.geometry_dest.is_empty
                & x.geometry_clip.notna() & ~x.geometry_clip.is_empty
                & x.fuel_consumption_u.isna()))
            .empty), "Some routed flows with valid geometries and region are missing fuel consumption values, indicating a potential issue in the emissions calculation pipeline."

    # ── No-nulls test by column ─────────────────────────────────────────
    for col in emiss.columns:
        n_null = emiss[col].isna().sum()
        if n_null == 0:
            logger.info(f"✓ Column '{col}': no nulls")
        else:
            logger.warning(f"✗ Column '{col}': {n_null} null values")

    # ── Emissions vs. Original Flow Verification
    # For each layer, verify that the sum of trips and tonnage in the
    # emissions DataFrame matches the original layer flows. Any discrepancy
    # indicates data loss or duplication during the pipeline.
    logger.info("Confirming emissions flows match layer flows")
    for layer in layers:
        emiss_sums=( 
                    # Sum trips and tonnage by material_grouping in emissions
                    emiss
                    .filt(lambda x: (x.layer==layer.name))
                    .pipe(mask_for_od_totals)
                    # .groupby(['ttype','material','EMFAC_class']) # FIXME: issues with diffs producing NAs
                    .groupby(['ttype','material_grouping'])
                    .agg({'trips':'sum','wt_sent':'sum'})
                )
        layer_sums=(
                    # Sum trips and tonnage by material_grouping in original flows
                    layer.get_flows()
                    # .groupby(['ttype','material','EMFAC_class'])
                    .groupby(['ttype','material_grouping'])
                    .agg({'trips':'sum','wt_sent':'sum'})
                )
        res=np.allclose(
            emiss_sums.pipe(deunitize,'wt_sent'),
            layer_sums.pipe(deunitize,'wt_sent'))

        if not res:
            estr=f'{layer.name} layer Flows for emissions differ from original flows'
            with redirect_stdout_to_logging():
                logger.error(estr)
                logger.error('emiss sums:')
                display(emiss_sums)
                logger.error('layer sums')
                display(layer_sums)
            raise ValueError(estr)
            
    # FIXME:TODO: Test units

def compare_emissions_dataframes(emiss: pd.DataFrame, emiss_cmp: pd.DataFrame) -> None: # FIXME:types
    """Compare two emission dataframes and log equivalence results."""

    expected_cols = [
        'od_flow_index', 'EMFAC_class', 'year', 'layer', 'ttype',
        'material_stream', 'material_grouping', 'o_id', 'd_id', 'o_n1', 'o_n2',
        'd_n1', 'd_n2', 'geometry_orig', 'o_geometry_src', 'geometry_dest',
        'd_geometry_src', 'o_country', 'd_country', 'o_state', 'd_state',
        'o_county', 'd_county', 'o_facility_type', 'd_facility_type', 'trips',
        'wt_sent', 'step_num', 'region', 'geometry_orig_dist',
        'geometry_port', 'id_port', 'clip_num', 'geometry_clip', 'name_port',
        'geometry_searte', 'seadist_u', 'geometry_dest_dist', 'route',
        'geometry_full', 'duration_u', 'distance_u', 'step_speed_u',
        'speed_min_u', 'speed_max_u', 'clip_distance', 'speed_bin_u', 'vmt_u',
        'calendar_year', 'model_year', 'nox_runex_u', 'pm2_5_runex_u',
        'pm10_runex_u', 'co2_runex_u', 'ch4_runex_u', 'n2o_runex_u',
        'rog_runex_u', 'tog_runex_u', 'co_runex_u', 'sox_runex_u',
        'nh3_runex_u', 'pm10_pmbw_u', 'pm2_5_pmbw_u', 'fuel_consumption_r_u',
        'energy_consumption_r_u', 'total_vmt_u', 'fuel_consumption_u',
        'energy_consumption_u', 'emiss_pm25_u', 'emiss_pm10_u', 'emiss_nox_u',
        'emiss_co_u', 'emiss_co2_u', 'emiss_ch4_u', 'emiss_n2o_u',
        'emiss_ghg_u'
    ]

    # ── 1. Full DataFrame equality ──────────────────────────────────────
    if emiss.equals(emiss_cmp):
        logger.info("✓ Full DataFrame equivalence: PASS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # Nothing left to do!
        return
    else:
        logger.warning("✗ Full DataFrame equivalence: FAIL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # ── 2. Shape comparison ─────────────────────────────────────────────
    if emiss.shape == emiss_cmp.shape:
        logger.info(f"✓ Shape equivalence: PASS ({emiss.shape})")
    else:
        logger.warning(
            f"✗ Shape mismatch: emiss={emiss.shape} vs emiss_cmp={emiss_cmp.shape}"
        )

    # ── 3. Column set comparison ────────────────────────────────────────
    cols_emiss = set(emiss.columns)
    cols_cmp = set(emiss_cmp.columns)

    if cols_emiss == cols_cmp:
        logger.info("✓ Column set equivalence: PASS")
    else:
        missing_in_cmp = cols_emiss - cols_cmp
        extra_in_cmp = cols_cmp - cols_emiss
        if missing_in_cmp:
            logger.warning(f"✗ Columns missing in emiss_cmp: {sorted(missing_in_cmp)}")
        if extra_in_cmp:
            logger.warning(f"✗ Extra columns in emiss_cmp: {sorted(extra_in_cmp)}")

    # ── 4. Expected columns presence ────────────────────────────────────
    missing_expected = set(expected_cols) - cols_emiss
    missing_expected_cmp = set(expected_cols) - cols_cmp
    if not missing_expected and not missing_expected_cmp:
        logger.info("✓ All expected columns present in both DataFrames: PASS")
    else:
        if missing_expected:
            logger.warning(f"✗ Expected columns missing from emiss: {sorted(missing_expected)}")
        if missing_expected_cmp:
            logger.warning(f"✗ Expected columns missing from emiss_cmp: {sorted(missing_expected_cmp)}")

    # ── 5. Column order comparison ──────────────────────────────────────
    common_cols = [c for c in emiss.columns if c in cols_cmp]
    common_cols_cmp = [c for c in emiss_cmp.columns if c in cols_emiss]
    if common_cols == common_cols_cmp:
        logger.info("✓ Column order equivalence: PASS")
    else:
        logger.warning("✗ Column order differs between DataFrames")

    # ── 6. Dtypes comparison (on common columns) ───────────────────────
    common = list(set(emiss.columns) & set(emiss_cmp.columns))
    dtype_mismatches = {
        c: (emiss[c].dtype, emiss_cmp[c].dtype)
        for c in common
        if emiss[c].dtype != emiss_cmp[c].dtype
    }
    if not dtype_mismatches:
        logger.info("✓ Dtype equivalence on all common columns: PASS")
    else:
        for col, (dt1, dt2) in dtype_mismatches.items():
            logger.warning(f"✗ Dtype mismatch on '{col}': emiss={dt1} vs emiss_cmp={dt2}")

    # ── 7. Index equivalence ────────────────────────────────────────────
    if emiss.index.equals(emiss_cmp.index):
        logger.info("✓ Index equivalence: PASS")
    else:
        logger.warning("✗ Index mismatch between DataFrames")

    # ── 8. Layer-level equivalence ──────────────────────────────────────
    if 'layer' in common:
        layers_emiss = set(emiss['layer'].unique())
        layers_cmp = set(emiss_cmp['layer'].unique())

        if layers_emiss == layers_cmp:
            logger.info(f"✓ Layer set equivalence: PASS (layers: {sorted(layers_emiss)})")
        else:
            missing_layers = layers_emiss - layers_cmp
            extra_layers = layers_cmp - layers_emiss
            if missing_layers:
                logger.warning(f"✗ Layers missing in emiss_cmp: {sorted(missing_layers)}")
            if extra_layers:
                logger.warning(f"✗ Extra layers in emiss_cmp: {sorted(extra_layers)}")

        # Per-layer full equivalence
        shared_layers = sorted(layers_emiss & layers_cmp)
        for layer in shared_layers:
            sub = emiss[emiss['layer'] == layer].reset_index(drop=True)
            sub_cmp = emiss_cmp[emiss_cmp['layer'] == layer].reset_index(drop=True)

            if sub.equals(sub_cmp):
                logger.info(f"✓ Layer '{layer}' full equivalence: PASS")
            else:
                logger.warning(f"✗ Layer '{layer}' full equivalence: FAIL")

                # Shape within layer
                if sub.shape != sub_cmp.shape:
                    logger.warning(
                        f"  └ Layer '{layer}' shape mismatch: {sub.shape} vs {sub_cmp.shape}"
                    )
                else:
                    # Column-by-column value comparison within this layer
                    for col in common:
                        s1 = sub[col]
                        s2 = sub_cmp[col]
                        try:
                            # Numeric: use allclose for float tolerance
                            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                                mask = ~(s1.isna() & s2.isna())
                                if mask.any():
                                    if np.allclose(
                                        s1[mask].astype(float),
                                        s2[mask].astype(float),
                                        rtol=1e-9, atol=1e-12,
                                        equal_nan=True,
                                    ):
                                        continue  # values match
                                    else:
                                        diff_mask = ~np.isclose(
                                            s1[mask].astype(float),
                                            s2[mask].astype(float),
                                            rtol=1e-9, atol=1e-12,
                                        ) & ~(s1[mask].isna() & s2[mask].isna())
                                        n_diff = diff_mask.sum()
                                        if n_diff > 0:
                                            max_rel = np.max(np.abs(
                                                (s1[mask][diff_mask].astype(float) - s2[mask][diff_mask].astype(float))
                                                / s2[mask][diff_mask].astype(float).replace(0, np.nan)
                                            ))
                                            logger.warning(
                                                f"  └ Layer '{layer}' col '{col}': "
                                                f"{n_diff} value diffs, max relative diff={max_rel:.2e}"
                                            )
                            else:
                                # Non-numeric: exact equality
                                if not (s1 == s2).all():
                                    if ((s1 == s2) | (s1.isna() & s2.isna())).all():
                                        natot=(s1.isna() & s2.isna()).sum()
                                        eqtot=(s1 == s2).sum()
                                        logger.warning(
                                            f"  └ Layer '{layer}' col '{col}': all [equal ({eqtot}) or both NA({natot})]"
                                        )
                                    else:
                                        n_diff = (s1 != s2).sum()
                                        logger.warning(
                                            f"  └ Layer '{layer}' col '{col}': {n_diff} value diffs"
                                        )
                        except Exception as e:
                            logger.warning(
                                f"  └ Layer '{layer}' col '{col}': comparison error - {e}"
                            )
    else:
        logger.warning("✗ 'layer' column not in both DataFrames; skipping layer-level tests")

    # ── 9. Numeric column summary statistics comparison ─────────────────
    numeric_cols = [c for c in common if pd.api.types.is_numeric_dtype(emiss[c])]
    if numeric_cols:
        stats_mismatch = []
        for col in numeric_cols:
            s1 = safe_describe(emiss[col])
            s2 = safe_describe(emiss_cmp[col])
            try:
                if not np.allclose(s1, s2, rtol=1e-9, atol=1e-12, equal_nan=True):
                    stats_mismatch.append(col)
            except ValueError as e:
                logger.warning(f"✗ can't apply np.allclose: {e}")

        if not stats_mismatch:
            logger.info("✓ Summary statistics (describe) match for all numeric columns: PASS")
        else:
            logger.warning(f"✗ Summary statistics differ for: {stats_mismatch}")

    # ── 10. Null count comparison ────────────────────────────────────────
    null_mismatches = {}
    for col in common:
        n1 = emiss[col].isna().sum()
        n2 = emiss_cmp[col].isna().sum()
        if n1 != n2:
            null_mismatches[col] = (n1, n2)

    if not null_mismatches:
        logger.info("✓ Null count equivalence on all common columns: PASS")
    else:
        for col, (n1, n2) in null_mismatches.items():
            logger.warning(f"✗ Null count mismatch on '{col}': emiss={n1} vs emiss_cmp={n2}")

    # ── 11. Emission columns per-layer sum comparison ───────────────────
    emiss_cols = [c for c in common if c.startswith('emiss_')]
    if emiss_cols and 'layer' in common:
        for layer in shared_layers:
            sub = emiss[emiss['layer'] == layer]
            sub_cmp = emiss_cmp[emiss_cmp['layer'] == layer]
            for col in emiss_cols:
                sum1 = sub[col].sum()
                sum2 = sub_cmp[col].sum()
                if pd.isna(sum1) and pd.isna(sum2):
                    continue
                if not np.isclose(sum1, sum2, rtol=1e-6):
                    rel_diff = abs(sum1.magnitude - sum2.magnitude) / max(abs(sum2.magnitude), 1e-15)
                    logger.warning(
                        f"✗ Layer '{layer}' col '{col}' sum: "
                        f"emiss={sum1:.6e} vs emiss_cmp={sum2:.6e} (rel diff={rel_diff:.2e})"
                    )
                else:
                    logger.info(
                        f"✓ Layer '{layer}' col '{col}' sum match: {sum1:.6e}"
                    )




# e=emiss.filt(lambda x: x.layer=='Appliance')
# e_cmp=emiss_cmp.filt(lambda x: x.layer=='Appliance')
# e_sums=e.groupby('ttype').sum(numeric_only=True).filter(regex='emiss')
# e_cmp_sums=e_cmp.groupby('ttype').sum(numeric_only=True).filter(regex='emiss')
# not_close=(
#     list(e_sums.merge(e_cmp_sums,right_index=True,left_index=True,how='left',suffixes=('','_cmp'))
#          .assign(diffx=lambda x: x.emiss_ghg_u-x.emiss_ghg_u_cmp)
#          .filter(regex='^(?!.*(emiss|_u))')
#          .filt(lambda x: ~np.isclose(x.diffx.pint.magnitude,0)).index)
# )
# e.filt(lambda x: x.ttype.isin(not_close[0:1])).groupby(['ttype','o_id','d_id']).sum(numeric_only=True).filter(regex='emiss')
# ['car_to_landfill1', 'car_to_shredder', 'shredder_to_export', 'shredder_to_landfill']