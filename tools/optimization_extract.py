import typer
from importlib import reload # module reloading

from layers.rdrs import backfill_regions, aggregate_all_geometries, backfill_regions, collect_rdrs_ods, geocode_entities, get_rdrs_flow_for_year, load_rdrs_flow, merge_aggregated_geometries, read_aggregated_geometries, read_rdrs_geocodes
import pandas as pd
import geopandas as gpd
from services.geocode import get_zcta
from services.routing import compute_nearest_neighbor_mapping

zcta=get_zcta()

myrdrs=(
    read_rdrs_geocodes()
    .filt(lambda x: ~x.geometry_rdrs.isna() & ~x.geometry_rdrs.is_empty)
    .assign(lon=lambda x: x.apply(lambda row: row.geometry_rdrs.x,axis=1),
            lat=lambda x: x.apply(lambda row: row.geometry_rdrs.y,axis=1))
)
tt=(
    pd.read_csv('rdrs_geocodes - MRF data.csv',skiprows=1).clean_names()
    .assign(rdrs_id=lambda x: x.triid.str.replace(r'.*(RD\d+).*',r'\1',regex=True))
    .filt(lambda x: ~x.rdrs_id.isna())
)

nn=compute_nearest_neighbor_mapping(
    'zip_to_mrf',
    zcta.assign(centroid=lambda x: x.geometry.representative_point()),'centroid',
    myrdrs.filt(lambda x: x.rdrs_id.isin(tt.rdrs_id.unique())),'geometry_rdrs',
    'GEOID20',
    radius=250,max_options=10,keep_options=10
)

nn[['GEOID20','centroid','rdrs_id','geometry_rdrs','distance_m','duration_s','order']].to_csv('zip_to_mrf.csv')