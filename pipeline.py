#!/usr/bin/env python
# coding: utf-8

### must be before pandas and pint to ensure a single unit registry is used
from dataclasses import dataclass, field, fields
import dataclasses

from click.core import ParameterSource

from core.common import (
    wdisplay, case_when, cpath)


### import logging first so it propogates to imported modules
from collections import OrderedDict
import logging
from layers.ewaste import EwasteLayer
from config.cli_field import cli_field
from config.path_template import PathTemplate
from core.units import deunitize, transform_units, unify_pint_units
from utils.logging_config import log_errors, redirect_stdout_to_logging, set_log_level, LoggerStream
logger = logging.getLogger(__name__)

### common config stuff
from config.settings import settings
import config

### support for to-disk per function caching
from cachier import cachier

### general typing
import typer
from typing import Any, Callable, Dict, List, Optional, Set
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

### general stuff
import re
import numpy as np
import warnings
import os # for OS independent paths

### data handling
import pandas as pd
import geopandas as gpd
import janitor

### model layers
from layers.base import ModelLayer, filter_for_od_totals, mask_for_od_totals, merge_all_endpoints, od_pair_group
from layers.appliance import (
    ApplianceLayer, read_car_endpoints, read_reclaimer_endpoints, read_appliance_external_rdrs)
from core.model_types import LayerFlowSchema, ODRoutablePts, ODTourLegRouted, RegionSchema, schema_to_cols
from core.model_types import unique_flow_agg,unique_flow_cols
from core.model_types import (
    _ODPairSchemaNames, _ODPairSchemaGeometries, _ODPairSchemaAggregations, _ODPairSchemaFacilityTypes
)

from layers.ct import CTLayer
from layers.rdrs import RDRSLayer

### model support
from services.emissions import (
    compute_on_road_emissions, compute_msw_collection_emissions, compute_total_emissions, convert_emiss_to_full, convert_to_od_emiss, 
    load_regions_extended)

from services.routing import generate_searoutes, get_maritime_routes, get_distances, route_maritime


### geocoding
from services.geocode import (
    GeocodeByCountyStateCountry, GeocodeById, MergeGeocoder, NullGeocoder, GeocodeFromLatLon, crs_ca, crs_ll, crs_usa, 
    get_county_endpoints, get_dismantler_endpoints, get_state_endpoints, get_world_endpoints,
    get_zcta_endpoints, get_ca_port_endpoints, gpd_concat, read_rdrs_manual_geocodes, read_unify_geo)

# introspection
from pathlib import Path
import inspect
import sys

# Suppress the importlib deprecation warning from pygris
warnings.filterwarnings(
    'ignore',
    category=DeprecationWarning,
    module='pygris.internal_data'
)
import pygris as pyg
import pygris.data as pygdat

### Plotting

import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis.plots import (plot_maritime_routes, plot_flows_by_material)
from plotnine import *

from IPython.display import display
from shapely.geometry import box
import contextily as ctx

### Testing/validating
from test.test import compare_emissions_dataframes, validate_emissions_dataframe

# Typer CLI
from rich.console import Console # for content width
import typer.rich_utils

typer.rich_utils.MAX_WIDTH = 120

app = typer.Typer(
    rich_markup_mode = "rich"
)
#model_year = 2021
#appliance_case = ApplianceCase.TEST
#refr_case = RefrCase.MIN
#loglevel='INFO'

from enum import Enum

class OutputOption(str, Enum):
    """Valid output artifacts the pipeline can produce."""
    EMISS            = "emiss"
    OD_EMISS         = "od_emiss"
    EMISS_TOT        = "emiss_tot"
    EMISS_CSV        = "emiss.csv"
    OD_EMISS_CSV     = "od_emiss.csv"
    ENTITIES         = "entities"
    ENTITIES_GJ      = "entities.geojson"
    PORTS_GJ         = "ports.geojson"
    COLLECTIONS_CSV  = "collections.csv"
    AGGREGATIONS_CSV = "aggregations.csv"

# Build a lookup dict once so validation is O(1)
_VALID_OUTPUTS: dict[str, OutputOption] = {o.value: o for o in OutputOption}


class LayerName(str, Enum):
    """Valid output artifacts the pipeline can produce."""
    RDRS            = "rdrs"
    ELV             = "elv"
    APPLIANCE       = "appliance"
    EWASTE          = "ewaste"
    CT              = "ct"
_VALID_LAYERS: dict[str, LayerName] = {o.value: o for o in LayerName}


from services.option_utils import inject_model_options, parse_commalist_callback
from config.base_config import BaseModelConfig, _model_registry


#### IMPORTANT: explicitly import the model configs that we want to instantiate 
#### as command-line arguments
from layers.rdrs import RDRSModelConfig
from layers.appliance import ApplianceModelConfig
from layers.elv import ELVModelConfig
from layers.ewaste import EwasteModelConfig
from layers.ct import CTModelConfig

from dataclasses import dataclass, field
from config.cli_field import cli_field

@dataclass
class RunConfig(BaseModelConfig, label="pipeline"):
    output_pipeline: PathTemplate      = cli_field(
        Path("output/pipeline"),
        help            = "Output directory (supports templates)",
        rich_help_panel = "Pipeline options"
    )
    strict:          bool = cli_field(
        False,
        help            = "Apply strict checks on results"
    )
    model_year:      Optional[int]     = cli_field(
        None,
        help            = "The model year to use for the pipeline — will propagate to all layers.",
        rich_help_panel = "Global overrides"
    )
    include_layers:   List[str]        = cli_field(
        "all",
        help            = f"Comma-delimited list of layer names. Valid choices: {', '.join(list(_VALID_LAYERS) + ['all'])}.",
        callback        = parse_commalist_callback(_VALID_LAYERS,allow_all='all'),
        rich_help_panel = "Model choice options"
    )
    omit_layers:     List[str]         = cli_field(
        "",
        help            = f"Comma-delimited list of layer names to omit. Valid choices: {', '.join(_VALID_LAYERS)}.",
        callback        = parse_commalist_callback(_VALID_LAYERS),
        rich_help_panel = "Model choice options"
    )
    use_cachier:     bool = cli_field(
        True,
        help            = "Use function caching (globally) with cachier where annotated. Can be overridden in code",
        rich_help_panel = "Model chioce option" #?
    )
    strict:          bool = cli_field(
        False,
        help            = "Raise exceptions with identified data inconsistencies rather than cleaning.",
        rich_help_panel = "Validation options"
    )
    compare_results: Optional[str]     = cli_field(
        None,
        help            = "Suffix of results data to compare to",
        rich_help_panel = "Validation options"
    )
    
    gis_dir:          PathTemplate  = field(default = Path('data/gis'))
    
    unify_geo_file:   PathTemplate  = cli_field(
        default         = None,
        help            = 'File with unified endpoint geographies',
        rich_help_panel = "Geocode options"
    )

@app.command()
@log_errors
@inject_model_options
def run(
    ctx: typer.Context,
    dry_run:    bool = typer.Option(
        False, "--dry-run",
        help="Print plan without executing"
        ,rich_help_panel="Model options"
        ),
    loglevel: str = typer.Option(
        "INFO", 
        help="The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)."
        ,rich_help_panel="Model options"
        ),
    output: List[str] = typer.Option( # MOVE TO OPTIONS
        ["emiss"],
        "--output", "-o",
        callback=parse_commalist_callback(_VALID_OUTPUTS),
        help=(
            "Output artifact(s) to create. Accepts repeated flags, comma-delimited "
            "values, or a mix of both. "
            f"Valid choices: {', '.join(_VALID_OUTPUTS)}."
        )
        ,rich_help_panel="Pipeline options"
        ),

    # ── Injected by @inject_model_options — DO NOT declare manually ──────────
    # (--appliance-appliance-case, --appliance-model-year, etc. appear in --help
    #  automatically for every registered submodel)

    model_configs: Dict[str, Any] = None   # populated by the decorator wrapper

):

    # ── Logging Configuration ──────────────────────────────────────────
    # Set the application log level from the CLI option.
    set_log_level(loglevel)    
    
    # ── Override with global parameters ──────────────────────────────────────
    
    global_params = [
        # add any future globals here
        "model_year", "matmap_file", 'model_cache_dir',
        "unify_geo_file"
    ]

    for param in global_params:
        value = locals().get(param)  # grab the defined value
        if value is not None:
            for model,cfg in model_configs.items(): #[appliance_cfg, elv_cfg]:
                if getattr(cfg, "_sources", {}).get(param) == "cli":
                    continue   # don't override a layer-specific cli argument
                setattr(cfg, param, value)
                cfg._sources[param] = "cli (global)"

    # Define the material categorization file that maps raw material names
    # to standardized material groups/categories used throughout the pipeline.
    processed_data_overrides: Dict[str:str] = {
        'matmap_file': cpath('processed_data','MATCATS_MaterialMappings24037.xlsx')
    }
    param = 'matmap_file'
    matmap_file = cpath('processed_data','MATCATS_MaterialMappings24037.xlsx')
    for param,value in processed_data_overrides.items():
        if value is not None:
            for model,cfg in model_configs.items(): #[appliance_cfg, elv_cfg]:
                if getattr(cfg, "_sources", {}).get(param) == "cli":
                    continue   # don't override a layer-specific cli argument
                setattr(cfg, param, value)
                cfg._sources[param] = "config file override (processed_data)"        


    # ── Print all fields with value and source ───────────────────────────────
    
    _SOURCE_LABEL = {
        ParameterSource.COMMANDLINE : "cli",
        ParameterSource.ENVIRONMENT : "env",
        ParameterSource.DEFAULT     : "default",
        ParameterSource.DEFAULT_MAP : "default map",
    }

    # Names belonging to submodel configs — exclude from top-level display

    def echo_heading(label=None,width=80,newline=True):
        trail_len=max((width - 2 - len(label)),0)
        typer.echo(rf'{"\n" if newline else ""}── {label} {"─" * (trail_len)}')

    _model_param_names = {
        f.name if not config_cls._cli_prefix else f"{config_cls._cli_prefix}_{f.name}"
        for label, config_cls in _model_registry
        for f in fields(config_cls)
    } | {"model_configs"}

    echo_heading(label="top-level options")
    for name, value in ctx.params.items():
        if name in _model_param_names:
            continue
        source = _SOURCE_LABEL.get(ctx.get_parameter_source(name), "unknown")
        typer.echo(f"  {name:<30} : {str(value):<30}  [{source}]")
        
    # ── Print all submodel configs ────────────────────────────────────────────
    for prefix, cfg in model_configs.items():
        sources = getattr(cfg, "_sources", {})
        # typer.echo(rf"── {prefix} config {'─' * (48 - len(prefix))}")
        echo_heading(label=f"{prefix} config")

        for f in fields(cfg):
            value   = getattr(cfg, f.name)
            source  = sources.get(f.name, "unknown")
            display = "<MISSING>" if isinstance(value, dataclasses._MISSING_TYPE) else str(value)
            typer.echo(f"  {f.name:<30} : {display:<30}  [{source}]")


    if dry_run:
        return
    
    if not model_configs['pipeline'].use_cachier:
        logger.info("Disabling cachier caching as requested")
        cachier.disable_cachier()

    
    # ── Initial setup ──────────────────────────────────────────────────
    outdir:             Path = model_configs['pipeline'].output_pipeline
    model_year:         int  = model_configs['pipeline'].model_year
    strict:             bool = model_configs['pipeline'].strict
    compare_results:    str  = model_configs['pipeline'].compare_results
    
    # ── Get outputs to include ─────────────────────────────────────────
    requested_outputs: set[OutputOption] = set(output)         # type: ignore[arg-type]
    include_layers:    set[LayerName]    = model_configs['pipeline'].include_layers
    
    if include_layers == 'all':
        include_layers = _VALID_LAYERS
        
    requested_layers:  set[LayerName]    = (
        set(include_layers) - set(model_configs['pipeline'].omit_layers)
    )
    
    if not requested_layers:
         # show help and stop
        typer.echo("No layers requested after applying include/omit filters.\n")
        typer.echo(ctx.get_help())
        raise typer.Exit(1)
    
    # some layers have dependencies on rdrs:
    dependent_on_rdrs = set([LayerName.ELV, LayerName.APPLIANCE])
    needs_rdrs=requested_layers.intersection(dependent_on_rdrs)
    needs_rdrs=False
    if needs_rdrs and LayerName.RDRS not in requested_layers:
        logger.info(f"Layers {[v.value for v in needs_rdrs]} require RDRS, adding RDRS layer to requested layers.")
        requested_layers.add(LayerName.RDRS)
        
    if not outdir.exists():
        logger.info(f"Creating output directory: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)

        
    logger.info(f"==== Executing pipeline for layers: {requested_layers}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── CREATE LAYERS ─────────────────────────────────────────────────────────
    ## * Create layer objects
    ## * Load layer data (flows + endpoints)
    layers: List[ModelLayer] = []
    
    # ── RDRS Layer ────────────────────────────────────────────────────
    # Load the Refrigerant Distribution & Recovery System layer, which
    # models flows of recovered refrigerants through the RDRS network.
    if LayerName.RDRS in requested_layers:
        rdrs_layer = RDRSLayer(          config = model_configs['rdrs'] )
        layers.append(rdrs_layer)
    else:
        rdrs_layer = None

    # ── Appliance Layer ───────────────────────────────────────────────
    # Load the Appliance recycling layer, which models flows of appliances
    # through reclaimers and recycling facilities. Supports configurable
    # appliance and refrigerant cases, and optional flow model execution.
    if LayerName.APPLIANCE in requested_layers:
        appliance_layer = ApplianceLayer(config = model_configs['appliance'],
                                         export_proportions_layer=rdrs_layer )
        layers.append(appliance_layer)

    # ── Connecticut Layer ─────────────────────────────────────────────
    # Load the Connecticut-specific layer for regional flow modeling.
    if LayerName.CT in requested_layers:
        ct_layer = CTLayer(              config = model_configs['ct'] )
        layers.append(ct_layer)

    # ── E-Waste Layer ─────────────────────────────────────────────────
    # Load the electronic waste layer, modeling flows of e-waste through
    # collection and processing facilities.
    if LayerName.EWASTE in requested_layers:
        ewaste_layer = EwasteLayer(      config = model_configs['ewaste'] )
        layers.append(ewaste_layer)
        
    # ── ELV Layer ─────────────────────────────────────────────────────
    # Load the End-of-Life Vehicle layer, modeling flows of vehicles
    # through dismantlers and processors.
    if LayerName.ELV in requested_layers:
        from layers.elv import ELVLayer
        elv_layer = ELVLayer(            config = model_configs['elv'],
                                         export_proportions_layer=rdrs_layer)
        layers.append(elv_layer)

    # ── Layer Summary Logging ─────────────────────────────────────────
    # Log the number of flow records and unique endpoints loaded for each layer.
    # FIXME: this will also instantiate model runs so is kind of awkward
    for layer in layers:
        logger.info(f'{layer.name} layer: Read {len(layer.get_flows())} flow records')
        logger.info(f'{layer.name} layer: Read {len(layer.get_endpoints())} unique endpoints')
        
    # FIXME: should be explicit about the model run(s).  
    # at this point, we have a collection of layers, each with a set of flows,
    # across multiple trip types, modes (north american on-road plus maritime
    # exports) between endpoints (origins to destinations). The

    # ── Merge All Endpoints ───────────────────────────────────────────
    # Collect endpoints from all layers, standardize their geocodes using
    # the geocoder stack, and merge into a single GeoDataFrame. Each endpoint
    # is tagged with its source layer for traceability.
    logger.info("Collecting all endpoint entities for all layers and standardizing geocodes")
    merged_entities_all_layers = merge_all_endpoints([
            layer.get_endpoints().set_geometry('geometry').to_crs(crs_ll)
            .add_column_if_missing('src',layer.name)
            for layer in layers
        ])
    
    # ── Duplicate Index Check ─────────────────────────────────────────
    # Verify that all entity IDs are unique across layers. Duplicate IDs
    # would cause incorrect joins later in the pipeline.
    if not merged_entities_all_layers.index.is_unique:
        display(merged_entities_all_layers[merged_entities_all_layers.duplicated(keep=False)])
        raise ValueError(f"{sum(merged_entities_all_layers.index.duplicated())} entities with duplicated indexes")

    # ── Debug Display & GeoJSON Export ────────────────────────────────
    # Display the merged entities at DEBUG level and write to GeoJSON
    # for external use. This is specifically used by the web interface
    if logger.getEffectiveLevel() <= logging.DEBUG:
       display(merged_entities_all_layers)
    
    if OutputOption.ENTITIES_GJ in requested_outputs:    
        (
            merged_entities_all_layers.drop(columns=['geometry_agg'])
            .to_file(outdir / 'rdrs_ent_allxxx.geojson',driver='GeoJSON')
        )

    # ── Pickle Export ─────────────────────────────────────────────────
    # Serialize merged entities to pickle for fast reloading in
    # post-pipeline analysis notebooks.
    suffix="_".join([l.get_name() for l in layers])
    merged_entities_all_layers.to_pickle(outdir / f'entities_{suffix}.pkl')

    # ── Merge OD Flows Across Layers ──────────────────────────────────
    # Concatenate flows from all layers, filter out zero-tonnage flows,
    # aggregate by unique OD pair columns, and re-join origin/destination
    # entity metadata (names, facility types, geographies, geometries).
                
    od_flows_all_layers: LayerFlowSchema = ModelLayer.merge_all_layer_flows(layers,merged_entities_all_layers)
        

    # ── Debug Display of Merged Flows ─────────────────────────────────
    if logger.getEffectiveLevel() <= logging.DEBUG:
        wdisplay(od_flows_all_layers)

    # ── Flow Consistency Verification ─────────────────────────────────
    # For each layer, verify that the merged OD flows match the original
    # layer flows in both row count and total tonnage/trips. This catches
    # data loss or duplication introduced during the merge.
    logger.info(f"Checking flow consistency for each layer {[l.name for l in layers]}")
    for layer in layers:
        lflows = layer.get_flows()
        odf = od_flows_all_layers.filt(lambda x: x.layer==layer.name)
        flow_len_check=len(odf)==len(lflows)
        if not flow_len_check:
            raise ValueError(f"Different number of flow records in '{layer.name}'"
                             f" flows [{len(lflows)}] than in od_flows for layer '{layer.name}' [{len(odf)}]")
        
        # Compare aggregated totals (tonnage and trips) between original and merged
        lflows_tots = (
            lflows.groupby(unique_flow_cols).agg({'wt_sent':'sum','trips':'sum'})
            .pipe(transform_units,{'wt_sent':'tons'})
            .pipe(deunitize,'wt_sent')
        )
        odf_tots    = (
            odf.groupby(unique_flow_cols).agg({'wt_sent':'sum','trips':'sum'})
            .pipe(transform_units,{'wt_sent':'tons'})
            .pipe(deunitize,'wt_sent')
        )
        # make sure flows totals are back        diff=np.abs(lflows_tots-odf_tots).sum().sum()<0.01
        if(not np.allclose(lflows_tots,odf_tots)):
            with redirect_stdout_to_logging(logging.ERROR):
                diff=(lflows_tots - odf_tots)
                raise ValueError(f"OD flows {layer.name} flows sums not matching the original "
                                 f"{layer.name} flows sum(diff)={diff}")

    logger.info(f"Merging OD flows: successfully merged {len(od_flows_all_layers)} records.")

    # ── Layer/Type/Grouping Summary Export ────────────────────────────
    # Write a summary CSV of unique layer × trip-type × material-grouping ×
    # EMFAC-class combinations for reference.
    logger.info(f"Merging OD flows: Saving layer trip-type grouping totals.")
    (
        # FIXME: use drop duplicates
        od_flows_all_layers.groupby(['layer','ttype','material_grouping','EMFAC_class']).first()[[]]
        # FIXME: circle back and use this for modal choice
        .reset_index().to_csv(outdir / 'layer-ttype-grouping.csv')
    )

    # ── Port Endpoint Augmentation ────────────────────────────────────
    # Add California port endpoints that aren't already in the merged entities.
    # These are needed for maritime routing. Ports within 50,000m of existing
    # entities are included.
    # FIXME: think about this
    all_ports=get_ca_port_endpoints(near_geoms=merged_entities_all_layers,dwithin=50000,regex='.*')
    use_ports=all_ports.filt(lambda x: ~x.index.isin(merged_entities_all_layers.index))

    # ── FIXME ─────────────────────────────────────────────────────────
    logger.info(f"Collecting all endpoint entities for all layers")
    merged_entities_all_layers=gpd_concat([merged_entities_all_layers,use_ports])

    # ── Post-Augmentation Uniqueness Check ────────────────────────────
    # Verify no duplicate entity IDs after adding ports.
    if not merged_entities_all_layers.index.is_unique:
        raise ValueError(f"Duplicate entities in the merged entities for all layers")

    logger.info(f"Collecting all endpoint entities for all layers: "
                f"{len(merged_entities_all_layers)} unique endpoint entities")

    # ── Maritime Route Generation ──────────────────────────────────────
    # Generate sea routes between California ports and international
    # destinations for flows that require maritime transport.
    
    from services.routing import split_multimodal,assign_maritime_tour_ports
    
    # expand od_flows (tours) into individual legs by mode the legs  
    # that were split will have dummy port nodes (in the dest for the drayage
    # leg, and in the orig for the maritime leg)
    od_flows_mm = (
        split_multimodal(od_flows_all_layers,merged_entities_all_layers)
        # FIXME: below currently needed for ODRoutablePt
        .assign(**{
            'o_pt':lambda x: x.geometry_orig,
            'd_pt':lambda x: x.geometry_dest
        })
    )
    
    # now we have a dataframe with multimodal trips

    # ── Load Regions for Route Segmentation ───────────────────────────
    # Load extended region boundaries (CoAbDis subareas) for segmenting
    # on-road routes by region. This is needed for region-specific
    # emission rate calculations.
    regions_full = load_regions_extended().to_crs(crs_ll) # for segmenting routes
    
    # # do maritime
    # routes_maritime = route_maritime(
    #     od_flows_mm.filt(lambda x: x.transport_mode=='maritime'),
    #     region)

    # ── On-Road Distance Computation ──────────────────────────────────
    # Compute on-road routes and distances for all unique OD pairs using
    # OSRM. Routes are segmented by region for emissions calculation.
    # stdout from OSRM is redirected to the logger.
    logger.info("Routing by mode")
    # FIXME: interim check
    
    assert (od_flows_mm.filt(lambda x: (x.transport_mode=='on_road') 
                             & (~x.d_id.str.match('^PORT')))
            .groupby(['layer','ttype','material_stream','material_grouping','o_id','d_id'])
            .count().filt(lambda x: x.od_flow_index>1).empty), "Unexpected duplicate ODs"

    # for generalizability, we define a dict of routers keyed by mode    
    routers: OrderedDict[
        str,
        Callable[[DataFrame[ODRoutablePts],DataFrame[RegionSchema]],
                 DataFrame[ODTourLegRouted]]
    ] = {
        'maritime': route_maritime,
        'on_road':  lambda x,y: get_distances(
            x,y
            , cachier__verbose    = True
            , cachier__skip_cache = True)
    }
    
    routed={}
    for transport_mode,mode_df in (
        od_flows_mm
        .groupby('transport_mode')
    ):
    
        with redirect_stdout_to_logging(logging.INFO): # captures stdout to logger
            routing_fn = routers[transport_mode]
            assert routing_fn is not None, f'No routing function for {transport_mode}'
            mode_routed = (
                routing_fn(
                    mode_df
                    .drop_duplicates(subset=schema_to_cols(ODRoutablePts)),
                    regions_full)
                [schema_to_cols(ODTourLegRouted)]       # cast to remove unneeded cols
                .assign(
                    transport_mode = transport_mode
                )
            )
            unique_group=od_pair_group + ['step_num','clip_num']
            if mode_routed.duplicated(subset=unique_group).any():
                # FIXME: should enforce this in the OD schema
                logger.warning(f"Routing for {transport_mode} has {mode_routed.duplicated(subset=unique_group).sum()} duplicated O-D pairs. Keeping only the first...")
                mode_routed=mode_routed.drop_duplicates(subset=unique_group)
            routed[transport_mode] = (
                mode_df
                .merge(mode_routed,on=schema_to_cols(ODRoutablePts) + ['transport_mode'],
                       how='left',suffixes=('','_route'))
            )
            assert (
                routed[transport_mode].geometry_clip.notna()
                #| (routed[transport_mode].o_pt == routed[transport_mode].d_pt) # colocated endpoints will have null geom
            ).all(),  f"Some routes are NA for mode {transport_mode}"
            assert (
                len(routed[transport_mode].groupby(schema_to_cols(ODRoutablePts) + ['transport_mode'])
                    [['od_flow_index']].count())
                ==
                len(mode_df.groupby(schema_to_cols(ODRoutablePts) + ['transport_mode'])[['od_flow_index']].count())
            ), (
                f"Attached routed df ({len(routed[transport_mode])}) different length than onrouted {len(mode_df)} for mode {transport_mode}"
            )
            
    # at this point, we should have everything routed and ready to join back 
    # onto the od_flows_mm
    od_flows_routed = (
        pd.concat(unify_pint_units(routed.values()),ignore_index=True)
        .sort_values([
            'layer','year','material_stream','material_grouping',
            'ttype','o_id_tour','d_id_tour','o_id','d_id','leg'])
    )


    # ── Route Plotting (Disabled) ─────────────────────────────────────
    # Optional visualization of non-maritime and maritime routes.
    if False:
        # Do some plots
        plot_non_maritime_routes(distances, regions_full)

        # Now we plot three pairs of ODs that do have a Maritime leg.
        plot_maritime_routes(distances, crs_ll)



    def override_port_trucks(df):
        return (
            df
            # add port info where appropriate
            .merge(merged_entities_all_layers
                   .filt(lambda x: x.index.str.match('^PORT'))
                   .add_col_prefix('p_'),
                   left_on='d_id',right_index=True,how='left')
            # Override trip type for RDRS flows that go through a port
            .assign(
                ttype=lambda x: x.ttype.mask(x.o_id.str.match('^PORT') & (x.layer=='rdrs'),'rdrs_export'), # FIXME: LayerName.RDRS?
                EMFAC_class=lambda x: x.EMFAC_class.mask(x.p_n1.notna(),'T7 POLA Class 8')
            )
            
            # Override EMFAC class for flows through Port of Oakland
            .assign(
                # USE POAK if we know it's oakland
                EMFAC_class=lambda x: x.EMFAC_class.mask(x.p_n1.str.match('^.*OAKLAND'),'T7 POAK Class 8')
            )
            .filter(regex='^(?!p_)')
        )

    # ── Post-Distance Flow Consistency Checks ───────────────────────────
    # Re-verify that joining distances didn't alter flow totals for any layer.
    for layer in layers:
        lflows = layer.get_flows()
        odf = od_flows_routed.filt(lambda x: (x.layer==layer.name) & (x.transport_mode=='on_road')).pipe(filter_for_od_totals)
        flow_len_check=len(odf)==len(lflows)
        if not flow_len_check:
            raise ValueError(f"Different number of flow records in '{layer.name}' flows [{len(lflows)}] than in od_flows for layer '{layer.name}' [{len(odf)}]")
        
        lflows_tots = (
            lflows.groupby(unique_flow_cols).agg({'wt_sent':'sum','trips':'sum'})
            .pipe(transform_units,{'wt_sent':'tons'})
            .pipe(deunitize,'wt_sent')
        )
        # have to compare on the basis of tour endpoints
        unique_flow_cols_adj=['EMFAC_class', 'year', 'layer', 'ttype', 'material_stream', 'material_grouping', 'o_id_tour', 'd_id_tour']
        odf_tots    = (
            odf.groupby(unique_flow_cols_adj).agg({'wt_sent':'sum','trips':'sum'})
            .pipe(transform_units,{'wt_sent':'tons'})
            .pipe(deunitize,'wt_sent')
        )
        # make sure flows totals are back
        if(not np.allclose(lflows_tots,odf_tots)):
            with redirect_stdout_to_logging(logging.ERROR):
                display(lflows_tots-odf_tots)
                raise ValueError(f"OD flows {layer.name} flows sums not matching the original {layer.name} flows sum(diff)={diff}")
        
        
    od_flows_w_dist=od_flows_routed.pipe(override_port_trucks)
        
    # -- Additional expectations
    if od_flows_w_dist.step_num.max()<2:
        raise ValueError(f"OD flows maximum step_num is {od_flows_w_dist.step_num.max()}; Must be an issue as we expect multiple steps in routes")

    # ── Appliance Flow Uniqueness Assertion
    # Verify that appliance flows have exactly one row per OD pair ×
    # trip type × material grouping × EMFAC class combination. Duplicate
    # rows would cause double-counting in emissions.
    assert (od_flows_w_dist
               .filt(lambda x: (x.layer=='Appliances') & (x.step_num.isna()))
               .assign(tmp=1)
               .groupby(['o_id','d_id','ttype','material_grouping','EMFAC_class']) # FIXME: use schema
               .count()
               .filt(lambda x: x.tmp>1)).empty, (
                   "Appliance flows have duplicate rows for the same OD pair × trip type × material grouping × EMFAC class combination."
                   )

    # ── Collection Emissions (RDRS Layer) ──────────────────────────────
    # Compute MSW collection emissions using EMFAC2021 emission rates
    # by county. Only computed when the RDRS layer is active, since
    # collection activity is specific to refrigerant recovery.
    # Since we do not have direct information on collection activity, we default
    # to using estimates of emissions from
    # [EMFAC2021](https://arb.ca.gov/emfac/emissions-inventory) for each CoAbDis
    # subarea.

    
    if LayerName.RDRS in requested_layers:
        logger.info("Computing collection emissions (California)")
        emfac_inv = compute_msw_collection_emissions(model_year)
        
        if OutputOption.COLLECTIONS_CSV in requested_outputs:
            emfac_inv.to_csv(outdir / 'collection_activity_by_cty.csv')
    else:
        logger.info("Skipping computation of collection emissions since RDRS layer is not included.")
        emfac_inv=None


    # ── Non-Collection Emissions ──────────────────────────────────────-
    # 
    # We calculate on-road emissions using EMFAC emission rates by CoAbDis
    # subarea and speed bin. We use the EMFAC vehicle classes for each flow
    # record to determine the emission rates. We assume that the share of fuel
    # types used by trucks is proportional to the EMFAC VMT by fuel type so
    # compute the emissions rates as the average of rates associated with each
    # fuel type weighted by EMFAC VMT for that fuel type.

    # ── Default Region Assignment
    # For flows with missing region (CoAbDis subarea), assign a default
    # region. The default is inferred from the most common region among
    # CT-prefixed entities, falling back to 'Los Angeles (MD)' if none found.
    # FIXME: force a default region for locations not mapped to coabdis regions
    if od_flows_w_dist.region.isna().any():
        logger.warning(f"Using default region {default_region} for {od_flows_w_dist.filt(lambda x: x.region.isna()).shape[0]} records with missing regions")
        default_regions=list(od_flows_w_dist.filt(
            lambda x: x.o_id.str.match(r'^n?CT')
            ).groupby(['region']).count().sort_values('o_n1').tail(1).index)
        if len(default_regions)>0:
            default_region=default_regions[0]
            if len(default_regions)>0:
                logger.warning(f"Using default region {default_region} for {od_flows_w_dist.filt(lambda x: x.region.isna()).shape[0]} records with missing regions")
        else:
            default_region='Los Angeles (MD)'
            logger.warning(f"No default region found from data, using default region {default_region} for {od_flows_w_dist.filt(lambda x: x.region.isna()).shape[0]} records with missing regions")
    else:
        # won't need to fill
        default_region=None        
        
    # ── Fill Missing Values
    # Fill NaN values in region, EMFAC class, and trips columns with
    # sensible defaults to ensure downstream emissions computation
    # doesn't fail on missing data.
    od_flows_w_dist=od_flows_w_dist.assign(
        region=lambda x: x.region.fillna(default_region),
        EMFAC_class=lambda x: x.EMFAC_class.fillna('T7 Tractor Class 8'), # FIXME: asserted defqult here
        trips=lambda x: x.trips.fillna(x.wt_sent/20)
    )
    assert(~od_flows_w_dist.region.isna().any())

    # ── Unrouted Flow Handling
    # Check for flows that couldn't be routed (no step_num). In non-strict
    # mode, these are silently omitted with a warning. In strict mode,
    # they cause an error.
    # FIXME: catch unrouted
    logger.info("Checking for unrouted flows")
    if od_flows_w_dist.step_num.isna().any():
        if not strict:
            logger.warning(f'{od_flows_w_dist.step_num.isna().sum()} OD Flows are unrouted. Omitting them for now')
            od_flows_w_dist = (
                od_flows_w_dist
                .filt(lambda x: ~x.step_num.isna()) # if step_num is null we have no route
                .assign(
                    # if step_nums were null, we've removed the nulls and can convert to int now
                    step_num=lambda x: x.step_num.astype(int),
                    clip_num=lambda x: x.clip_num.astype(int)
                )
            )
        else:
            display(od_flows_w_dist[od_flows_w_dist.step_num.isna()])
            raise ValueError(f'{od_flows_w_dist.step_num.isna().sum()} OD Flows are unrouted.')

    # ── On-Road Emissions Computation ──────────────────────────────────
    # Compute on-road emissions (PM2.5, PM10, NOx, CO, CO2, CH4, N2O, GHG)
    # for each route segment using EMFAC emission rates by region and
    # speed bin. Fuel consumption and energy use are also calculated.
    logger.info("Computing emissions")
    
    with redirect_stdout_to_logging(logging.INFO): # captures stdout to logger
        emiss = compute_on_road_emissions(
            od_flows_w_dist.filt(lambda x: x.transport_mode=='on_road')
            , model_year) #, cachier__verbose=True)
    
    logger.info("Validating emissions")
    validate_emissions_dataframe(emiss,layers)

    if compare_results:
        suffix="_".join([l.get_name() for l in layers])
        cfe=Path(outdir / f'emiss_{suffix}-{compare_results}.pkl')
        if cfe.is_file():
            emiss_cmp=pd.read_pickle(cfe)
            logger.info(f"Comparing results to {cfe}")
            compare_emissions_dataframes(emiss,emiss_cmp)
        else:
            logger.warning(f'Emissions comparison file {cfe} not found...skipping comparison')
    

    # ── Emissions Pickle Export ────────────────────────────────────────
    # Save emissions DataFrame to pickle for fast reloading.
    if OutputOption.EMISS in requested_outputs:
        suffix="_".join([l.get_name() for l in layers])
        logger.info(f"Saving full results to emiss_{suffix}.pkl")
        emiss.to_pickle(outdir / f'emiss_{suffix}.pkl')
        
    if OutputOption.EMISS_TOT in requested_outputs:

        logger.info("Expanding od_flows by mode")
        emiss_tot = convert_emiss_to_full(emiss)

        # validate no duplicates
        assert(
            not
            emiss_tot.groupby(['o_id_od','d_id_od','layer','ttype','material_stream','material_grouping','mode'])
            .agg({'year':'count'}).filt(lambda x: x.year>1).year
            .any()
        )
        logger.info(f"Saving full results to emiss_tot_{suffix}.pkl")
        emiss_tot.to_pickle(outdir / f'emiss_tot_{suffix}.pkl')

    
    if OutputOption.OD_EMISS in requested_outputs or OutputOption.OD_EMISS_CSV in requested_outputs:
        od_emiss=convert_to_od_emiss(emiss)
        for layer in od_emiss.layer.unique():
            logger.info(f"Saving layer {layer} od_emiss results")
            od_emiss.filt(lambda x: x.layer==layer).to_csv(outdir / f'od_emiss_{layer}.pkl')
            od_emiss.filt(lambda x: x.layer==layer).to_csv(outdir / f'od_emiss_{layer}.csv')
        logger.info(f"Saving all od_emiss results")
        od_emiss.to_pickle(outdir / f'od_emiss_{suffix}.pkl')
        od_emiss.to_csv(outdir / 'od_emiss_all_layers.csv')


    # ── Aggregations (Optional) ───────────────────────────────────────
    # The following block computes aggregated emissions summaries by
    # origin county, destination county, and within-county segments.
    # Only runs when --do-aggregations is enabled.
    
    if OutputOption.AGGREGATIONS_CSV in requested_outputs:
    
        # ── Total Emissions Computation ────────────────────────────────
        # Compute total emissions by county to county flowscombining on-road and collection emissions.
        logger.info("Computing total emissions")
        tot_emiss=compute_total_emissions(emiss, emfac_inv, model_year)

        if logger.getEffectiveLevel() <= logging.INFOX:
            for layer in tot_emiss.layer.unique():
                print(f'Layer {layer}')
                wdisplay(tot_emiss.filt(lambda x: x.layer==layer))

        # ── Aggregated Emissions CSV Export ───────────────────────────
        # Write per-layer and combined county-to-county flow emissions.
        logger.info("Writing aggregated emissions to output: c2c_flows_w_emissions")

        for l,df in tot_emiss.reset_index().groupby('layer'):
            df.to_csv(outdir / f'c2c_flows_w_emissions_{l}.csv')

        tot_emiss.reset_index().to_csv(outdir / 'c2c_flows_w_emissions.csv')

        # ── Origin County Aggregation ─────────────────────────────────
        # Aggregate emissions by origin county, material stream, and
        # material grouping. Computes total tonnage, trips, VMT, fuel/
        # energy consumption, and all pollutant emissions. Also calculates
        # average trip length (VMT / trips).
        logger.info("Computing aggregated flows and emissions to output: by origin county")

        agg_county_emiss_orig=(
            emiss
            .pipe(mask_for_od_totals)
            .groupby(['year']+['o_county']+['material_stream','material_grouping'])
            .agg({
                'wt_sent':'sum',
                'trips':'sum',
                'vmt':'sum',
                'fuel_consumption':'sum',
                'energy_consumption':'sum',
                'emiss_pm25':'sum',
                'emiss_pm10':'sum',
                'emiss_nox':'sum',
                'emiss_co':'sum',
                'emiss_co2':'sum',
                'emiss_ch4':'sum',
                'emiss_n2o':'sum',
                'emiss_ghg':'sum'
            })
            .assign(
                avg_triplen_mi=lambda x: x.vmt/x.trips
            )
            .reset_index()
            .add_col_suffix('_orig')
            .assign(state='California',country='United States')
            .rename(columns={'year_orig':'year','o_county_orig':'county','material_stream_orig':'material_stream',
                             'material_grouping_orig':'material_grouping'})
        )

        # ── Destination County Aggregation ────────────────────────────
        # Aggregate emissions by destination county (including state and
        # country for non-California destinations), material stream, and
        # material grouping.
        logger.info("Computing aggregated flows and emissions to output: by destination county")

        agg_county_emiss_dest=(
            emiss
            .pipe(mask_for_od_totals)
            .groupby(['year']+['d_county','d_state','d_country']+['material_stream','material_grouping'])
            .agg({
                'wt_sent':'sum',
                'trips':'sum',
                'vmt':'sum',
                'fuel_consumption':'sum',
                'energy_consumption':'sum',
                'emiss_pm25':'sum',
                'emiss_pm10':'sum',
                'emiss_nox':'sum',
                'emiss_co':'sum',
                'emiss_co2':'sum',
                'emiss_ch4':'sum',
                'emiss_n2o':'sum',
                'emiss_ghg':'sum'
            })
            .assign(
                avg_triplen_mi=lambda x: x.vmt/x.trips
            )
            .reset_index()
            .add_col_suffix('_dest')
            .rename(columns={'year_dest':'year',
                            'd_county_dest':'county',
                            'd_state_dest':'state',
                            'd_country_dest':'country',
                            'material_stream_dest':'material_stream',
                            'material_grouping_dest':'material_grouping'})
        )


        # ── Within-County Emissions Aggregation ───────────────────────
        # Compute emissions for route segments that occur within each county.
        # Uses a spatial join to assign each clipped route segment to its
        # overlapping county. Ton-miles (wt_sent × clip_distance) are
        # computed as a transport intensity metric.
        logger.info("Computing aggregated flows and emissions to output: within county emissions")

        us_counties=pyg.counties(year=2021,cache=True).clean_names()
        agg_county_emiss_within=(
            emiss
            .assign(
                wtmi=lambda x: x.wt_sent*x.clip_distance
            )
            # Fill missing county names via spatial join with US county polygons
            .assign(
                co_name=lambda x: (
                    x.add_col_if_missing('co_name',None)
                    .co_name.fillna(
                        x.set_geometry('geometry_clip').to_crs(crs_usa)
                        .sjoin(us_counties.add_col_prefix('co_').set_geometry('co_geometry').to_crs(crs_usa)
                               ,predicate='covered_by',how='left')
                        .co_name))
            )
            .groupby(['year']+['co_name']+['material_stream','material_grouping'])
            .agg({
                'wt_sent':'first',  # we count the total wt_sent traversing the county
                'trips':'first',     # ...and the total number of trips traversing the county
                'vmt':'sum',         # VMT makes sense
                'wtmi':'sum',       # as does wt-mi and everything below
                'fuel_consumption':'sum',
                'energy_consumption':'sum',
                'emiss_pm25':'sum',
                'emiss_pm10':'sum',
                'emiss_nox':'sum',
                'emiss_co':'sum',
                'emiss_co2':'sum',
                'emiss_ch4':'sum',
                'emiss_n2o':'sum',
                'emiss_ghg':'sum'
            })
            .assign(
                avg_triplen_mi=lambda x: x.vmt/x.trips
            )
            .reset_index()
            .add_col_suffix('_within')
            .rename(columns={'year_within':'year',
                            'co_name_within':'county',
                            'material_stream_within':'material_stream',
                            'material_grouping_within':'material_grouping'})
            .assign(state='California',country='United States')
        )

        # ── County Totals Join & Export ───────────────────────────────
        # Outer-join origin, destination, and within-county aggregations
        # on (year, county, state, country, material_stream, material_grouping).
        # Fill NaN with 0 (counties that only appear in one perspective).
        # Write combined county totals to CSV.
        logger.info("Writing county origin, destination, and within flows and emissions")
        county_totals=(
            agg_county_emiss_orig.set_index(['year','county','state','country','material_stream','material_grouping'])
            .join(agg_county_emiss_dest.set_index(['year','county','state','country','material_stream','material_grouping']),how='outer')
            .join(agg_county_emiss_within.set_index(['year','county','state','country','material_stream','material_grouping']),how='outer')
            .reset_index().fillna(0)
        )
        if logger.getEffectiveLevel() <= logging.INFOX:
            display(county_totals)
        county_totals.reset_index().to_csv(outdir / 'county_totals_w_emissions.csv')

    logger.info("All Done!")
    

# ── Cachier Cache Management Utilities ────────────────────────────────
# These functions discover, list, and clear cachier-cached function results
# across pipeline modules.

# Find all cachier-decorated functions in a module by checking for the
# `clear_cache` attribute that cachier adds to decorated functions.
def find_cachier_functions(module=None):
    if module is None:
        module = sys.modules[__name__]

    cachier_functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if hasattr(obj, 'clear_cache'):
            cachier_functions.append(obj)

    return cachier_functions

# Collect all cachier cache file paths from the specified modules
# (defaults to the main module plus all pipeline modules).
import layers.appliance, layers.ct, layers.rdrs, services.geocode, services.emissions, services.routing
def _get_cachier_files(modules=None):
    files = []
    mods = modules if modules is not None else [sys.modules[__name__],layers.appliance,layers.ct,layers.rdrs,services.geocode,services.emissions,services.routing] # FIXME cachier
    for mod in mods:
        funcs = find_cachier_functions(mod)
        for func in funcs:
            fp = Path(f'{func.cache_dpath()}{mod.__name__}.{func.__name__}')
            if fp.is_file():
                files.append(fp)
    return files

# Filter cachier cache file paths using a glob pattern (e.g., '*.cache').
import fnmatch
def _filter_cachier_files_by_glob(files, pattern):
    """Filter a list of Path objects using a glob pattern."""
    # Convert Path objects to strings and filter using fnmatch
    matching_files = [
        f for f in files 
        if fnmatch.fnmatch(str(f), pattern)
    ]
    return matching_files


# ── CLI Command: List Cache Files ─────────────────────────────────────
# Prints all cachier cache files found across pipeline modules.
@app.command()
def list_cachier_files():
    files = _get_cachier_files()
    print("\n".join([str(f) for f in files]))

# ── CLI Command: Remove Cache Files ──────────────────────────────────
# Clears cachier caches matching a glob pattern. By default runs in
# pretend mode (shows what would be deleted). Use --no-pretend to actually
# clear caches. Optionally filter by module name(s).
@app.command()
def rm_cachier_files(
    file:    str  = typer.Argument(help="Glob pattern to match files, e.g., '*.cache' or 'cache_?.dat'"),
    omit:    str  = typer.Option(None,help="Exclusion glob pattern to match files, e.g., '*.cache' or 'cache_?.dat'"),
    module:  str  = typer.Option(None, help="Comma-separated list of module names to filter by"),
    pretend: bool = typer.Option(True, help="Show commands that would be run")
):
    module_names = module.split(",") if module else []

    files: List[Path]                  = _get_cachier_files([sys.modules[mod] for mod in module_names] if module_names else None)
    filtered_files: Set[Path]          = set(_filter_cachier_files_by_glob(files,file))
    
    if omit:
        omit_filtered_files: Set[Path] = set(_filter_cachier_files_by_glob(filtered_files,omit))
        filtered_files -= set(omit_filtered_files)
    
    if pretend:
        print('Would remove the following cachier files with --no-pretend switch:')
        for f in sorted(list(filtered_files)):
            print(f)
    else:
        for f in sorted(list(filtered_files)):
            # Parse the cache filename to extract module and function names
            basename = os.path.basename(f)  # ".Geocode.get_ca_boundary"
            # clear leading dot if needed:
            basename = basename[1:] if basename.startswith('.') else basename  # "Geocode.get_ca_boundary"
            module_name, function_name = basename.split('.', 1)  # Split on first dot
        
            # Look up the function in the loaded module and clear its cache
            if module_name in sys.modules:
                module = sys.modules[module_name]

                if hasattr(module, function_name):
                    func = getattr(module, function_name)

                    if hasattr(func, 'clear_cache'):
                        func.clear_cache()
                        logger.info(f"Cleared cache for {module_name}.{function_name}")
                    else:
                        logger.warning(f"{module_name}.{function_name} is not decorated with @cachier...skipping")
                else:
                    logger.warning(f"Function {function_name} not found in {module_name}...skipping")
            else:
                logger.error(f"Module {module_name} not loaded")
                raise RuntimeError(f"Module {module_name} not loaded for {basename}")

if __name__ == "__main__":
    app()

