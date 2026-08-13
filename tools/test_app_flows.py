import pytest
import pandas as pd
import geopandas as gpd
from layers.appliance import compute_car_to_landfill1, compute_car_to_reclaimer_mapping, compute_car_to_shredder, compute_landfill_car_flows, compute_shred_to_landfillctmsr, compute_shredder_to_port, compute_zip_to_car
from layers.appliance import (
    compute_fluid_diversion,
)

@pytest.fixture
def sample_df_long():
    # Create a sample dataframe to simulate `df_long`
    data = {
        "zip_code": ["12345", "67890"],
        "appltype": ["rf", "fz"],
        "age": [1, 2],
        "disp": [100, 200],
        "wt": [50, 100],
        "ferrous": [0.5, 0.6],
        "nonferrous": [0.3, 0.4],
        "hazardous": [0.1, 0.2],
        "other": [0.1, 0.2],
        "total": [1.0, 1.4],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_diverted_apps():
    # Create a sample dataframe to simulate diverted appliances
    data = {
        "o_rdrsid": ["RD001", "RD002"],
        "tonssent": [10, 20],
    }
    return pd.DataFrame(data)

def test_compute_zip_to_car(sample_df_long):
    # Test the compute_zip_to_car function
    result = compute_zip_to_car(sample_df_long, use_mapping="osm")
    assert not result.empty
    assert "car" in result.columns

def test_compute_fluid_diversion(sample_df_long):
    # Test the compute_fluid_diversion function
    result = compute_fluid_diversion(sample_df_long)
    assert not result.empty
    assert "ship_weight_kg" in result.columns

def test_compute_landfill_car_flows(sample_diverted_apps):
    # Test the compute_landfill_car_flows function
    result = compute_landfill_car_flows(sample_diverted_apps, use_mapping="osm")
    assert not result.empty
    assert "white_goods" in result.columns

def test_compute_car_to_shredder(sample_df_long, sample_diverted_apps):
    # Test the compute_car_to_shredder function
    lf_car_flows = compute_landfill_car_flows(sample_diverted_apps, use_mapping="osm")
    result, all_car_inflows = compute_car_to_shredder(sample_df_long, lf_car_flows, use_mapping="osm")
    assert not result.empty
    assert "shredder" in result.columns

def test_compute_car_to_landfill1(sample_df_long):
    # Test the compute_car_to_landfill1 function
    all_car_inflows = sample_df_long.copy()
    result = compute_car_to_landfill1(all_car_inflows, df_mc2=pd.DataFrame(), use_mapping="osm")
    assert not result.empty
    assert "landfill" in result.columns

def test_compute_shredder_to_port(sample_df_long):
    # Test the compute_shredder_to_port function
    table2_car_to_shred = sample_df_long.copy()
    result, all_shred_inflows = compute_shredder_to_port(table2_car_to_shred, df_mc2=pd.DataFrame(), use_mapping="osm")
    assert not result.empty
    assert "port" in result.columns

def test_compute_shred_to_landfillctmsr(sample_df_long):
    # Test the compute_shred_to_landfillctmsr function
    all_shred_inflows = sample_df_long.copy()
    result = compute_shred_to_landfillctmsr(all_shred_inflows, df_mc2=pd.DataFrame(), use_mapping="osm")
    assert not result.empty
    assert "landfill_ctmsr" in result.columns

def test_compute_car_to_reclaimer():
    # Test the compute_car_to_reclaimer function
    result = compute_car_to_reclaimer_mapping()
    assert not result.empty
    assert "geometry_recl" in result.columns
