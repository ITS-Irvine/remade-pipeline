from itertools import permutations
import re

from cachier import cachier
import geopandas as gpd
import pandas as pd
import numpy as np
import janitor
import pygris as pyg
from services.geocode import crs_ll, get_world_boundary, logger, merge_lines, crs_usa
from layers.base import mask_for_od_totals
from core.common import case_when, cpath
from IPython.display import display
import core.common # MUST LOAD TO INSTANTIATE UNITS!
from core.common import wdisplay
from core.units import ureg
from core.model_types import PintDtype, RegionSchema, RouteClipsSchemaPts, pint_dimension_parser, unique_flow_cols
import pandera.pandas as pa
from typing import Annotated
from pandera.typing import Series, DataFrame

import logging

from core.units import assign_units, create_unit_assignment, create_unit_transformation
from utils.logging_config import redirect_stdout_to_logging
logger = logging.getLogger(__name__)

# https://epact.energy.gov/fuel-conversion-factors
energy_units = pd.read_csv(cpath('processed_data', 'units_fuel_energy.csv')).clean_names()
# FIXME: temporary due to google drive issue; move back to processed_data
# energy_units=pd.read_csv('./units_fuel_energy.csv').clean_names()

# Define gge based on Pint's existing therm unit to avoid redefining it
# From the CSV: 1 therm (CNG) = 0.7836990596 gge
# Therefore: 1 gge = 1/0.7836990596 therm ≈ 1.2761 therm
therm_row = energy_units[energy_units['unit'] == 'therm'].iloc[0]
therm_conversion = float(therm_row['conversion_factor'])
ureg.define(f"gge = {1/therm_conversion} therm")

# Define other fuel units in terms of gge
for row in energy_units.filt(lambda x: ~x.unit.isna() & (x.unit != 'therm')).itertuples():
    ureg.define(f"{row.unit} = {row.conversion_factor} gge")
ureg.define('dge = 1 gal_diesel')


emfac_er_unit = {
    'pm2_5_runex': 'g/mi',
    'pm10_runex': 'g/mi',
    'pm2_5_pmbw': 'g/mi',
    'pm10_pmbw': 'g/mi',
    'nox_runex': 'g/mi',
    'co_runex':'g/mi',
    'co2_runex':'g/mi',
    'ch4_runex':'g/mi',
    'n2o_runex':'g/mi',
}
emfac_ei_unit = {
    'pm2_5_totex': 'ton/yr',
    'pm10_totex': 'ton/yr',
    'nox_totex': 'ton/yr',
    'co_totex':'ton/yr',
    'co2_totex':'ton/yr',
    'ch4_totex':'ton/yr',
    'n2o_totex':'ton/yr',
}


emiss_unit = {
    'pm25': 'kilogram',
    'pm10': 'kilogram',
    'nox': 'tonne',
    'co':'tonne',
    'co2':'tonne',
    'ch4':'tonne',
    'n2o':'tonne',
    'ghg':'tonne'
}


# map fuel types to units for fuel consumption column (see units_fuel_energy.csv)
fuel_to_units_conversion = {
    'Diesel': 'gal_diesel',
    'Natural Gas': 'dge', # per EMFAC2021 documentation, CNG fuel consumption is given in DGE (diesel gallon equivalent)
    'Gasoline': 'gal_gas',
    'Plug-in Hybrid': 'gal_gas',
    'Electricity': 'kilowatt * hr', # FIXME?
}

# https://www.edf.org/sites/default/files/content/emission_equivalency_tool_documentation_methodology_23062022.pdf
# p1
compute_ar5_gwp_100 = lambda x: x.emiss_co2_u + x.emiss_ch4_u * 28 + x.emiss_n2o_u * 265
compute_ar6_gwp_100 = lambda x: x.emiss_co2_u + x.emiss_ch4_u * 27 + x.emiss_n2o_u * 273

@cachier(wait_for_calc_timeout=5)
def compute_msw_collection_emissions(model_year):
    """Compute annual MSW collection vehicle emissions from EMFAC inventory data.

    Reads the EMFAC2021 emissions inventory for solid waste collection vehicles
    (SWCV), computes fuel and energy consumption, converts emission totals to
    consistent units, and aggregates results by county and calendar year.

    Args:
        model_year: The calendar year for which to compute emissions.

    Returns:
        DataFrame aggregated by (county, calendar_year) containing VMT, trip
        counts, estimated tons sent, fuel/energy consumption, and per-pollutant
        emission totals (PM2.5, PM10, NOx, CO, CO2, CH4, N2O, GHG CO2e).
    """
    logger.info("Compute emissions: MSW Collection (from EMFAC)")

    # note, for EI swcv:
    # Units:  miles/year for CVMT and EVMT, trips/year for Trips, kWh/year for Energy Consumption, 
    #         tons/year for Emissions (total & totex), 1000 gallons/year for Fuel Consumption"
    emfac_inv = (
        pd.read_csv(cpath('emfac_data', 'EMFAC2021-EI-202xClass-swcv.csv'), skiprows=8)
        .clean_names()
        .pipe(lambda x:
            x.rename(columns={c: re.sub('pm2_5', 'pm25', c) for c in x.filter(regex='pm2_5').columns})
        )
    )
    fuel_units = emfac_inv.fuel.map(fuel_to_units_conversion) # used for fuel_consumption_u column below
    emfac_invx = (
        emfac_inv
        .assign(
            # FIXME: lifted estimate of 13 as average of this rough range per trip: https://scdhec.gov/environment/land-and-waste-landfills/how-landfills-work#:~:text=There%20are%2C%20of%20course%2C%20different,from%20about%20800%2D850%20homes.
            wt_sent=lambda x: ureg('13 short_ton') * x.trips,
        )
        .pipe(assign_units,{c: f'mi' for c in emfac_inv.filter(regex="^([ce]|total_)vmt$").columns})
        .pipe(assign_units,{c: f'kilowatt * hr' for c in emfac_inv.filter(regex="^energy_consumption$").columns})
        .pipe(assign_units,{c: f'tons' for c in emfac_inv.filter(regex="^(_total|_totex)$").columns})

        # fuel units are trickier, could do this with assign(,axis=1) but this is more efficient
        .pipe(lambda x: x.assign(fuel_consumption=pd.Series(
            [(fc * ureg(f'{u}')).to('gal_diesel') for fc, u 
             in zip(emfac_inv.fuel_consumption.astype(float) * 1000 # EMFAC unit is 1000 gals/year
                    , fuel_units)],
            index=x.index
        ).astype('pint[gal_diesel]'))) # everything converted to gal_diesel/mi (dge/mi) for consistency

        # EMFAC only reports electricity consumption for electric vehicles, so
        # we add the fuel consumption for other vehicles to that total
        .assign(
            energy_consumption=lambda x: x.energy_consumption + x.fuel_consumption.pint.to('kilowatt * hr')
        )

        .pipe(lambda df: df.rename(columns={f'{totpol}_total': f'emiss_{totpol}_u' for totpol in ['pm25', 'pm10']}))
        .pipe(lambda df: df.rename(columns={f'{totpol}_totex': f'emiss_{totpol}_u' for totpol in ['nox', 'co', 'co2', 'ch4', 'n2o']}))

        .assign(emiss_ghg_u=compute_ar6_gwp_100)
        
        .assign(county=lambda x: x.region.str.replace(r'\s+\(.*\)\s*$', '', regex=True))
        .drop(columns=['region', 'vehicle_category', 'model_year', 'speed', 'fuel', 'population', 'cvmt', 'evmt'])
        .rename(columns={'total_vmt': 'vmt'})
        .groupby(['county', 'calendar_year'])
        # FIXME: use schema
        [['wt_sent', 'trips', 'vmt', 'fuel_consumption', 'energy_consumption', 
          'emiss_pm25_u', 'emiss_pm10_u', 'emiss_nox_u', 'emiss_co_u', 'emiss_co2_u', 
          'emiss_ch4_u', 'emiss_n2o_u', 'emiss_ghg_u']]
        .sum()
        .reset_index()
    )
    return emfac_invx


# ## Metrics for county-county flows.
# 
# Here we compute aggregations of tons sent, trips, vmt, fuel & energy
# consumption, and various emissions associated with between-county flows for
# specific material stream/material category combinations. These are OD-based
# summaries.

# In[ ]:
def compute_total_emissions(emiss, emfac_inv, model_year):
    """Merge and aggregate on-road and (optionally) MSW collection emissions by OD pair.

    Validates the input emissions frame for completeness, zeroes out duplicated
    trip/tonnage counts on non-first route segments, appends maritime leg GHG
    emissions using a fixed intensity factor, and aggregates all fields to the
    OD-flow/material-stream grain.  When the input contains an ``rdrs`` layer,
    EMFAC-derived collection emissions are appended as a separate set of rows.

    Args:
        emiss: Per-segment emissions DataFrame produced by ``compute_emissions``,
            including routing metadata (step_num, clip_num, seadist_u, etc.).
        emfac_inv: County-level MSW collection emissions inventory produced by
            ``compute_msw_collection_emissions``.
        model_year: Calendar year used to filter ``emfac_inv`` to the correct
            annual totals.

    Returns:
        DataFrame aggregated to (year, layer, o_county, d_county, d_state,
        d_country, material_stream, material_grouping, EMFAC_class, ttype)
        containing summed tonnage, VMT, fuel/energy consumption, per-pollutant
        emissions, and combined on-road + maritime GHG totals.
    """

    maritime_gCO2e_tonne_km = 11.97

    logger.info("Merge all emissions: on road")

    # FIXME: get from schema?
    egroup=['year', 'layer'] + ['o_county', 'd_county', 'd_state', 'd_country'] + ['material_stream', 'material_grouping', 'EMFAC_class', 'ttype']
    for k in egroup:
        chk=emiss.filt(lambda x: x[k].isna())
        if(len(chk)>0):
            logger.error(f"emiss frame missing values for {k}")
            wdisplay(chk)
            assert(len(chk)==0)
            

    agg_od_emiss = (
        emiss
        .assign(
            # for the aggregation below, we only want to sum the trips per unique OD,mat stream,grouping4 group, not per step, 
            # so we zero trips everywere except the first step
            # FIXME: use mask_for_od totals
            wt_sent=lambda x: x.wt_sent.mask(~((x.step_num == 1)&(x.clip_num == 1)), 0),
            trips=lambda x: x.trips.mask(~((x.step_num == 1)&(x.clip_num == 1)), 0)
        )

        # ## Maritime emissions
        # 
        # To account for emissions associated with maritime legs, we use a factor of 8.3 `g CO2e/tonne-km` 
        # (tonne=1 metric ton). See 
        # [this reference](https://www.cn.ca/repository/popups/ghg/Carbon-Calculator-Emission-Factors#:~:text=Marine%20vessel%20shipping,IMO%20Greenhouse%20Gas%20Study%202020).
        # 
        # Here we compute the total ton-miles of maritime transport for `grouping4` material types and convert them 
        # into kg CO2e.
        .assign(
            tonmi_sea=lambda x: (x.wt_sent * x.seadist_u).pint.to('ton * mi').pint.magnitude,
            tonnekm_sea=lambda x: (x.wt_sent * x.seadist_u).pint.to('tonne * km').pint.magnitude,
            emiss_ghg_sea=lambda x: x.tonnekm_sea * maritime_gCO2e_tonne_km,  # *(ureg('g').to('kg').magnitude)
            emiss_ghg_all=lambda x: x.emiss_ghg + x.emiss_ghg_sea
        )
        .groupby(egroup)
        .agg({
            'wt_sent': 'sum',
            'trips': 'sum',
            'vmt': 'sum',
            'tonmi_sea': 'sum',
            'fuel_consumption': 'sum',
            'energy_consumption': 'sum',
            'emiss_pm25': 'sum',
            'emiss_pm10': 'sum',
            'emiss_nox': 'sum',
            'emiss_co': 'sum',
            'emiss_co2': 'sum',
            'emiss_ch4': 'sum',
            'emiss_n2o': 'sum',
            'emiss_ghg': 'sum',
            'emiss_ghg_sea': 'sum',
            'emiss_ghg_all': 'sum'
        })
        .reset_index()
        .assign(
            avg_triplen_mi=lambda x: x.vmt / x.trips,
            # FIXME: reset country names back to rdrs countries from world countries
            d_country=lambda x: x.d_country.mask(lambda y: y == 'United States of America', 'United States'),
            # origin_country     = lambda x: x.origin_country.mask(lambda y: y=='United States of America','United States')
        )
    )

    if 'rdrs' in emiss.layer.unique():
        logger.info("Including RDRS collection emissions")
        collection_emissions = (
            emfac_inv
            .filt(lambda x: x.calendar_year == model_year)
            .rename(columns={'county': 'o_county',
                            'calendar_year': 'year'})
            .assign(
                d_county=lambda x: x.o_county,
                d_state='California',
                d_country='United States',
                material_stream='Collection',
                material_grouping='MSW',
                EMFAC_class='T7 SWCV',
                ttype='collection',
                layer='rdrs',
                avg_triplen_mi=lambda x: x.vmt / x.trips
            )
        )

        tot_emiss = pd.concat([agg_od_emiss, collection_emissions]).fillna(0)
    else:
        tot_emiss=agg_od_emiss.fillna(0)
    
    return tot_emiss


### GENERATED BY Claude Sonnet 4.5 using zotgpt


# Complete CARB County and Air Basin Districts (COABDIS) Database
# Using official CARB naming convention: "County (Air Basin Code)"
carb_districts = [
    # GREAT BASIN VALLEYS AIR BASIN (GBV)
    {
        'name': 'Alpine County (Great Basin Valleys)',
        'coabdis': 'Alpine (GBV)',
        'county': 'Alpine',
        'air_basin': 'Great Basin Valleys',
        'lat': 38.5933, 'lon': -119.8186,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 0.5,
        'elevation': 2438,
        'avg_temp': 6.5,
        'coastal': False
    },
    {
        'name': 'Mono County (Great Basin Valleys)',
        'coabdis': 'Mono (GBV)',
        'county': 'Mono',
        'air_basin': 'Great Basin Valleys',
        'lat': 38.1987, 'lon': -118.8360,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 1.4,
        'elevation': 1463,
        'avg_temp': 10.0,
        'coastal': False
    },
    {
        'name': 'Inyo County (Great Basin Valleys)',
        'coabdis': 'Inyo (GBV)',
        'county': 'Inyo',
        'air_basin': 'Great Basin Valleys',
        'lat': 36.5833, 'lon': -117.6333,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 0.7,
        'elevation': 1219,
        'avg_temp': 15.0,
        'coastal': False
    },
    
    # LAKE COUNTY AIR BASIN (LC)
    {
        'name': 'Lake County',
        'coabdis': 'Lake (LC)',
        'county': 'Lake',
        'air_basin': 'Lake County',
        'lat': 39.0840, 'lon': -122.7591,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 19,
        'elevation': 404,
        'avg_temp': 15.0,
        'coastal': False
    },
    
    # MOUNTAIN COUNTIES AIR BASIN (MC)
    {
        'name': 'Calaveras County (Mountain Counties)',
        'coabdis': 'Calaveras (MC)',
        'county': 'Calaveras',
        'air_basin': 'Mountain Counties',
        'lat': 38.1888, 'lon': -120.5419,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 19,
        'elevation': 670,
        'avg_temp': 13.0,
        'coastal': False
    },
    {
        'name': 'Sierra County (Mountain Counties)',
        'coabdis': 'Sierra (MC)',
        'county': 'Sierra',
        'air_basin': 'Mountain Counties',
        'lat': 39.5769, 'lon': -120.5220,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 1.3,
        'elevation': 1463,
        'avg_temp': 10.5,
        'coastal': False
    },
    {
        'name': 'Plumas County (Mountain Counties)',
        'coabdis': 'Plumas (MC)',
        'county': 'Plumas',
        'air_basin': 'Mountain Counties',
        'lat': 39.9841, 'lon': -120.8319,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 2,
        'elevation': 1097,
        'avg_temp': 10.0,
        'coastal': False
    },
    {
        'name': 'Tuolumne County (Mountain Counties)',
        'coabdis': 'Tuolumne (MC)',
        'county': 'Tuolumne',
        'air_basin': 'Mountain Counties',
        'lat': 37.9827, 'lon': -119.9382,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 12,
        'elevation': 670,
        'avg_temp': 13.5,
        'coastal': False
    },
    {
        'name': 'Mariposa County (Mountain Counties)',
        'coabdis': 'Mariposa (MC)',
        'county': 'Mariposa',
        'air_basin': 'Mountain Counties',
        'lat': 37.4849, 'lon': -119.9663,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 4,
        'elevation': 610,
        'avg_temp': 14.5,
        'coastal': False
    },
    {
        'name': 'Nevada County (Mountain Counties)',
        'coabdis': 'Nevada (MC)',
        'county': 'Nevada',
        'air_basin': 'Mountain Counties',
        'lat': 39.3293, 'lon': -120.5322,
        'climate_zone': 'mountain',
        'urban_type': 'suburban',
        'population_density': 40,
        'elevation': 900,
        'avg_temp': 12.0,
        'coastal': False
    },
    {
        'name': 'Placer County (Mountain Counties)',
        'coabdis': 'Placer (MC)',
        'county': 'Placer',
        'air_basin': 'Mountain Counties',
        'lat': 39.0916, 'lon': -120.7401,
        'climate_zone': 'mountain',
        'urban_type': 'suburban',
        'population_density': 45,
        'elevation': 1829,
        'avg_temp': 9.5,
        'coastal': False
    },
    {
        'name': 'El Dorado County (Mountain Counties)',
        'coabdis': 'El Dorado (MC)',
        'county': 'El Dorado',
        'air_basin': 'Mountain Counties',
        'lat': 38.8375, 'lon': -120.4007,
        'climate_zone': 'mountain',
        'urban_type': 'suburban',
        'population_density': 31,
        'elevation': 950,
        'avg_temp': 12.5,
        'coastal': False
    },
    {
        'name': 'Amador County (Mountain Counties)',
        'coabdis': 'Amador (MC)',
        'county': 'Amador',
        'air_basin': 'Mountain Counties',
        'lat': 38.3471, 'lon': -120.5698,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 16,
        'elevation': 570,
        'avg_temp': 13.5,
        'coastal': False
    },
    
    # NORTH CENTRAL COAST AIR BASIN (NCC)
    {
        'name': 'Monterey County (North Central Coast)',
        'coabdis': 'Monterey (NCC)',
        'county': 'Monterey',
        'air_basin': 'North Central Coast',
        'lat': 36.6002, 'lon': -121.8947,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 50,
        'elevation': 25,
        'avg_temp': 14.0,
        'coastal': True
    },
    {
        'name': 'Santa Cruz County (North Central Coast)',
        'coabdis': 'Santa Cruz (NCC)',
        'county': 'Santa Cruz',
        'air_basin': 'North Central Coast',
        'lat': 37.0413, 'lon': -121.9584,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 234,
        'elevation': 30,
        'avg_temp': 13.5,
        'coastal': True
    },
    {
        'name': 'San Benito County (North Central Coast)',
        'coabdis': 'San Benito (NCC)',
        'county': 'San Benito',
        'air_basin': 'North Central Coast',
        'lat': 36.5761, 'lon': -121.0185,
        'climate_zone': 'bay_area',
        'urban_type': 'rural',
        'population_density': 24,
        'elevation': 60,
        'avg_temp': 15.5,
        'coastal': False
    },
    
    # NORTH COAST AIR BASIN (NC)
    {
        'name': 'Del Norte County (North Coast)',
        'coabdis': 'Del Norte (NC)',
        'county': 'Del Norte',
        'air_basin': 'North Coast',
        'lat': 41.7404, 'lon': -124.0888,
        'climate_zone': 'coastal',
        'urban_type': 'rural',
        'population_density': 7,
        'elevation': 15,
        'avg_temp': 11.5,
        'coastal': True
    },
    {
        'name': 'Humboldt County (North Coast)',
        'coabdis': 'Humboldt (NC)',
        'county': 'Humboldt',
        'air_basin': 'North Coast',
        'lat': 40.7450, 'lon': -123.8695,
        'climate_zone': 'coastal',
        'urban_type': 'rural',
        'population_density': 14,
        'elevation': 13,
        'avg_temp': 11.8,
        'coastal': True
    },
    {
        'name': 'Mendocino County (North Coast)',
        'coabdis': 'Mendocino (NC)',
        'county': 'Mendocino',
        'air_basin': 'North Coast',
        'lat': 39.4318, 'lon': -123.3510,
        'climate_zone': 'coastal',
        'urban_type': 'rural',
        'population_density': 9,
        'elevation': 50,
        'avg_temp': 13.5,
        'coastal': True
    },
    {
        'name': 'Trinity County (North Coast)',
        'coabdis': 'Trinity (NC)',
        'county': 'Trinity',
        'air_basin': 'North Coast',
        'lat': 40.6655, 'lon': -123.1254,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 1,
        'elevation': 700,
        'avg_temp': 12.5,
        'coastal': False
    },
    {
        'name': 'Sonoma County (North Coast)',
        'coabdis': 'Sonoma (NC)',
        'county': 'Sonoma',
        'air_basin': 'North Coast',
        'lat': 38.2500, 'lon': -123.0000,
        'climate_zone': 'coastal',
        'urban_type': 'rural',
        'population_density': 20,
        'elevation': 50,
        'avg_temp': 13.0,
        'coastal': True
    },
    
    # NORTHEAST PLATEAU AIR BASIN (NEP)
    {
        'name': 'Siskiyou County (Northeast Plateau)',
        'coabdis': 'Siskiyou (NEP)',
        'county': 'Siskiyou',
        'air_basin': 'Northeast Plateau',
        'lat': 41.5987, 'lon': -122.3917,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 2,
        'elevation': 823,
        'avg_temp': 10.5,
        'coastal': False
    },
    {
        'name': 'Modoc County (Northeast Plateau)',
        'coabdis': 'Modoc (NEP)',
        'county': 'Modoc',
        'air_basin': 'Northeast Plateau',
        'lat': 41.4503, 'lon': -120.7416,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 1,
        'elevation': 1341,
        'avg_temp': 8.5,
        'coastal': False
    },
    {
        'name': 'Lassen County (Northeast Plateau)',
        'coabdis': 'Lassen (NEP)',
        'county': 'Lassen',
        'air_basin': 'Northeast Plateau',
        'lat': 40.6632, 'lon': -120.5619,
        'climate_zone': 'mountain',
        'urban_type': 'rural',
        'population_density': 2,
        'elevation': 1341,
        'avg_temp': 9.5,
        'coastal': False
    },
    
    # SACRAMENTO VALLEY AIR BASIN (SV)
    {
        'name': 'Glenn County (Sacramento Valley)',
        'coabdis': 'Glenn (SV)',
        'county': 'Glenn',
        'air_basin': 'Sacramento Valley',
        'lat': 39.5985, 'lon': -122.3872,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 8,
        'elevation': 46,
        'avg_temp': 16.2,
        'coastal': False
    },
    {
        'name': 'Colusa County (Sacramento Valley)',
        'coabdis': 'Colusa (SV)',
        'county': 'Colusa',
        'air_basin': 'Sacramento Valley',
        'lat': 39.1793, 'lon': -122.2411,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 8,
        'elevation': 19,
        'avg_temp': 16.5,
        'coastal': False
    },
    {
        'name': 'Butte County (Sacramento Valley)',
        'coabdis': 'Butte (SV)',
        'county': 'Butte',
        'air_basin': 'Sacramento Valley',
        'lat': 39.6271, 'lon': -121.5947,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 65,
        'elevation': 54,
        'avg_temp': 16.0,
        'coastal': False
    },
    {
        'name': 'Shasta County (Sacramento Valley)',
        'coabdis': 'Shasta (SV)',
        'county': 'Shasta',
        'air_basin': 'Sacramento Valley',
        'lat': 40.5865, 'lon': -122.3917,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 18,
        'elevation': 160,
        'avg_temp': 16.5,
        'coastal': False
    },
    {
        'name': 'Placer County (Sacramento Valley)',
        'coabdis': 'Placer (SV)',
        'county': 'Placer',
        'air_basin': 'Sacramento Valley',
        'lat': 38.9042, 'lon': -121.0863,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 93,
        'elevation': 159,
        'avg_temp': 16.0,
        'coastal': False
    },
    {
        'name': 'Sacramento County (Sacramento Valley)',
        'coabdis': 'Sacramento (SV)',
        'county': 'Sacramento',
        'air_basin': 'Sacramento Valley',
        'lat': 38.5816, 'lon': -121.4944,
        'climate_zone': 'inland_valley',
        'urban_type': 'urban',
        'population_density': 593,
        'elevation': 8,
        'avg_temp': 16.2,
        'coastal': False
    },
    {
        'name': 'Yolo County (Sacramento Valley)',
        'coabdis': 'Yolo (SV)',
        'county': 'Yolo',
        'air_basin': 'Sacramento Valley',
        'lat': 38.6846, 'lon': -121.9018,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 78,
        'elevation': 16,
        'avg_temp': 16.5,
        'coastal': False
    },
    {
        'name': 'Tehama County (Sacramento Valley)',
        'coabdis': 'Tehama (SV)',
        'county': 'Tehama',
        'air_basin': 'Sacramento Valley',
        'lat': 40.0249, 'lon': -122.1864,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 7,
        'elevation': 88,
        'avg_temp': 16.8,
        'coastal': False
    },
    {
        'name': 'Sutter County (Sacramento Valley)',
        'coabdis': 'Sutter (SV)',
        'county': 'Sutter',
        'air_basin': 'Sacramento Valley',
        'lat': 39.0293, 'lon': -121.6922,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 60,
        'elevation': 19,
        'avg_temp': 16.0,
        'coastal': False
    },
    {
        'name': 'Yuba County (Sacramento Valley)',
        'coabdis': 'Yuba (SV)',
        'county': 'Yuba',
        'air_basin': 'Sacramento Valley',
        'lat': 39.2698, 'lon': -121.3673,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 48,
        'elevation': 18,
        'avg_temp': 16.2,
        'coastal': False
    },
    {
        'name': 'Solano County (Sacramento Valley)',
        'coabdis': 'Solano (SV)',
        'county': 'Solano',
        'air_basin': 'Sacramento Valley',
        'lat': 38.3782, 'lon': -121.7825,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 95,
        'elevation': 15,
        'avg_temp': 16.0,
        'coastal': False
    },
    
    # SAN DIEGO AIR BASIN (SD)
    {
        'name': 'San Diego County (San Diego)',
        'coabdis': 'San Diego (SD)',
        'county': 'San Diego',
        'air_basin': 'San Diego',
        'lat': 32.7157, 'lon': -117.1611,
        'climate_zone': 'mediterranean',
        'urban_type': 'urban',
        'population_density': 297,
        'elevation': 20,
        'avg_temp': 17.8,
        'coastal': True
    },
    
    # SAN FRANCISCO BAY AREA AIR BASIN (SF)
    {
        'name': 'Contra Costa County (San Francisco Bay Area)',
        'coabdis': 'Contra Costa (SF)',
        'county': 'Contra Costa',
        'air_basin': 'San Francisco Bay Area',
        'lat': 37.9193, 'lon': -121.9018,
        'climate_zone': 'bay_area',
        'urban_type': 'suburban',
        'population_density': 477,
        'elevation': 50,
        'avg_temp': 15.5,
        'coastal': True
    },
    {
        'name': 'Marin County (San Francisco Bay Area)',
        'coabdis': 'Marin (SF)',
        'county': 'Marin',
        'air_basin': 'San Francisco Bay Area',
        'lat': 38.0834, 'lon': -122.7633,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 198,
        'elevation': 50,
        'avg_temp': 14.0,
        'coastal': True
    },
    {
        'name': 'Alameda County (San Francisco Bay Area)',
        'coabdis': 'Alameda (SF)',
        'county': 'Alameda',
        'air_basin': 'San Francisco Bay Area',
        'lat': 37.6017, 'lon': -121.7195,
        'climate_zone': 'bay_area',
        'urban_type': 'urban',
        'population_density': 837,
        'elevation': 42,
        'avg_temp': 15.0,
        'coastal': True
    },
    {
        'name': 'San Francisco County (San Francisco Bay Area)',
        'coabdis': 'San Francisco (SF)',
        'county': 'San Francisco',
        'air_basin': 'San Francisco Bay Area',
        'lat': 37.7749, 'lon': -122.4194,
        'climate_zone': 'coastal',
        'urban_type': 'very_urban',
        'population_density': 7174,
        'elevation': 16,
        'avg_temp': 13.8,
        'coastal': True
    },
    {
        'name': 'Santa Clara County (San Francisco Bay Area)',
        'coabdis': 'Santa Clara (SF)',
        'county': 'Santa Clara',
        'air_basin': 'San Francisco Bay Area',
        'lat': 37.3541, 'lon': -121.9552,
        'climate_zone': 'bay_area',
        'urban_type': 'urban',
        'population_density': 558,
        'elevation': 25,
        'avg_temp': 15.5,
        'coastal': False
    },
    {
        'name': 'San Mateo County (San Francisco Bay Area)',
        'coabdis': 'San Mateo (SF)',
        'county': 'San Mateo',
        'air_basin': 'San Francisco Bay Area',
        'lat': 37.4337, 'lon': -122.4014,
        'climate_zone': 'coastal',
        'urban_type': 'urban',
        'population_density': 654,
        'elevation': 30,
        'avg_temp': 14.2,
        'coastal': True
    },
    {
        'name': 'Solano County (San Francisco Bay Area)',
        'coabdis': 'Solano (SF)',
        'county': 'Solano',
        'air_basin': 'San Francisco Bay Area',
        'lat': 38.2455, 'lon': -121.9544,
        'climate_zone': 'bay_area',
        'urban_type': 'suburban',
        'population_density': 126,
        'elevation': 10,
        'avg_temp': 15.8,
        'coastal': False
    },
    {
        'name': 'Napa County (San Francisco Bay Area)',
        'coabdis': 'Napa (SF)',
        'county': 'Napa',
        'air_basin': 'San Francisco Bay Area',
        'lat': 38.5025, 'lon': -122.2654,
        'climate_zone': 'bay_area',
        'urban_type': 'suburban',
        'population_density': 67,
        'elevation': 7,
        'avg_temp': 15.0,
        'coastal': False
    },
    {
        'name': 'Sonoma County (San Francisco Bay Area)',
        'coabdis': 'Sonoma (SF)',
        'county': 'Sonoma',
        'air_basin': 'San Francisco Bay Area',
        'lat': 38.5110, 'lon': -122.8016,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 84,
        'elevation': 48,
        'avg_temp': 14.5,
        'coastal': True
    },
    
    # MOJAVE DESERT AIR BASIN (MD)
    {
        'name': 'Los Angeles County (Mojave Desert)',
        'coabdis': 'Los Angeles (MD)',
        'county': 'Los Angeles',
        'air_basin': 'Mojave Desert',
        'lat': 34.5794, 'lon': -118.2382,
        'climate_zone': 'desert',
        'urban_type': 'suburban',
        'population_density': 25,
        'elevation': 700,
        'avg_temp': 16.5,
        'coastal': False
    },
    {
        'name': 'Kern County (Mojave Desert)',
        'coabdis': 'Kern (MD)',
        'county': 'Kern',
        'air_basin': 'Mojave Desert',
        'lat': 35.3733, 'lon': -118.4401,
        'climate_zone': 'desert',
        'urban_type': 'suburban',
        'population_density': 20,
        'elevation': 390,
        'avg_temp': 18.5,
        'coastal': False
    },
    {
        'name': 'San Bernardino County (Mojave Desert)',
        'coabdis': 'San Bernardino (MD)',
        'county': 'San Bernardino',
        'air_basin': 'Mojave Desert',
        'lat': 34.5794, 'lon': -117.2382,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 8,
        'elevation': 610,
        'avg_temp': 17.2,
        'coastal': False
    },
    {
        'name': 'Riverside County (Mojave Desert/MDAQMD)',
        'coabdis': 'Riverside (MD/MDAQMD)',
        'county': 'Riverside',
        'air_basin': 'Mojave Desert',
        'lat': 33.7350, 'lon': -116.2000,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 5,
        'elevation': 550,
        'avg_temp': 20.0,
        'coastal': False
    },
    {
        'name': 'Riverside County (Mojave Desert/SCAQMD)',
        'coabdis': 'Riverside (MD/SCAQMD)',
        'county': 'Riverside',
        'air_basin': 'Mojave Desert',
        'lat': 33.9000, 'lon': -116.0000,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 8,
        'elevation': 600,
        'avg_temp': 19.5,
        'coastal': False
    },
    
    # SAN JOAQUIN VALLEY AIR BASIN (SJV)
    {
        'name': 'Fresno County (San Joaquin Valley)',
        'coabdis': 'Fresno (SJV)',
        'county': 'Fresno',
        'air_basin': 'San Joaquin Valley',
        'lat': 36.7378, 'lon': -119.7871,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 61,
        'elevation': 94,
        'avg_temp': 17.8,
        'coastal': False
    },
    {
        'name': 'Madera County (San Joaquin Valley)',
        'coabdis': 'Madera (SJV)',
        'county': 'Madera',
        'air_basin': 'San Joaquin Valley',
        'lat': 37.1382, 'lon': -119.7693,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 24,
        'elevation': 82,
        'avg_temp': 16.8,
        'coastal': False
    },
    {
        'name': 'Kings County (San Joaquin Valley)',
        'coabdis': 'Kings (SJV)',
        'county': 'Kings',
        'air_basin': 'San Joaquin Valley',
        'lat': 36.0853, 'lon': -119.8198,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 23,
        'elevation': 73,
        'avg_temp': 17.5,
        'coastal': False
    },
    {
        'name': 'Kern County (San Joaquin Valley)',
        'coabdis': 'Kern (SJV)',
        'county': 'Kern',
        'air_basin': 'San Joaquin Valley',
        'lat': 35.3733, 'lon': -119.0187,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 41,
        'elevation': 122,
        'avg_temp': 18.2,
        'coastal': False
    },
    {
        'name': 'Tulare County (San Joaquin Valley)',
        'coabdis': 'Tulare (SJV)',
        'county': 'Tulare',
        'air_basin': 'San Joaquin Valley',
        'lat': 36.2079, 'lon': -118.9129,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 28,
        'elevation': 102,
        'avg_temp': 17.2,
        'coastal': False
    },
    {
        'name': 'Stanislaus County (San Joaquin Valley)',
        'coabdis': 'Stanislaus (SJV)',
        'county': 'Stanislaus',
        'air_basin': 'San Joaquin Valley',
        'lat': 37.6391, 'lon': -120.9969,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 130,
        'elevation': 26,
        'avg_temp': 16.8,
        'coastal': False
    },
    {
        'name': 'San Joaquin County (San Joaquin Valley)',
        'coabdis': 'San Joaquin (SJV)',
        'county': 'San Joaquin',
        'air_basin': 'San Joaquin Valley',
        'lat': 37.9577, 'lon': -121.2908,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 145,
        'elevation': 13,
        'avg_temp': 16.2,
        'coastal': False
    },
    {
        'name': 'Merced County (San Joaquin Valley)',
        'coabdis': 'Merced (SJV)',
        'county': 'Merced',
        'air_basin': 'San Joaquin Valley',
        'lat': 37.3022, 'lon': -120.4830,
        'climate_zone': 'inland_valley',
        'urban_type': 'rural',
        'population_density': 54,
        'elevation': 53,
        'avg_temp': 16.5,
        'coastal': False
    },
    
    # SOUTH CENTRAL COAST AIR BASIN (SCC)
    {
        'name': 'San Luis Obispo County (South Central Coast)',
        'coabdis': 'San Luis Obispo (SCC)',
        'county': 'San Luis Obispo',
        'air_basin': 'South Central Coast',
        'lat': 35.3102, 'lon': -120.7073,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 30,
        'elevation': 75,
        'avg_temp': 15.0,
        'coastal': True
    },
    {
        'name': 'Santa Barbara County (South Central Coast)',
        'coabdis': 'Santa Barbara (SCC)',
        'county': 'Santa Barbara',
        'air_basin': 'South Central Coast',
        'lat': 34.4208, 'lon': -119.6982,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 57,
        'elevation': 15,
        'avg_temp': 15.8,
        'coastal': True
    },
    {
        'name': 'Ventura County (South Central Coast)',
        'coabdis': 'Ventura (SCC)',
        'county': 'Ventura',
        'air_basin': 'South Central Coast',
        'lat': 34.3705, 'lon': -119.1391,
        'climate_zone': 'mediterranean',
        'urban_type': 'suburban',
        'population_density': 226,
        'elevation': 50,
        'avg_temp': 16.5,
        'coastal': True
    },
    
    # SOUTH COAST AIR BASIN (SC)
    {
        'name': 'Los Angeles County (South Coast)',
        'coabdis': 'Los Angeles (SC)',
        'county': 'Los Angeles',
        'air_basin': 'South Coast',
        'lat': 34.0522, 'lon': -118.2437,
        'climate_zone': 'mediterranean',
        'urban_type': 'very_urban',
        'population_density': 2419,
        'elevation': 93,
        'avg_temp': 18.5,
        'coastal': True
    },
    {
        'name': 'Orange County (South Coast)',
        'coabdis': 'Orange (SC)',
        'county': 'Orange',
        'air_basin': 'South Coast',
        'lat': 33.7175, 'lon': -117.8311,
        'climate_zone': 'mediterranean',
        'urban_type': 'very_urban',
        'population_density': 1656,
        'elevation': 46,
        'avg_temp': 17.8,
        'coastal': True
    },
    {
        'name': 'Riverside County (South Coast)',
        'coabdis': 'Riverside (SC)',
        'county': 'Riverside',
        'air_basin': 'South Coast',
        'lat': 33.9533, 'lon': -117.3962,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 127,
        'elevation': 246,
        'avg_temp': 19.2,
        'coastal': False
    },
    {
        'name': 'San Bernardino County (South Coast)',
        'coabdis': 'San Bernardino (SC)',
        'county': 'San Bernardino',
        'air_basin': 'South Coast',
        'lat': 34.1083, 'lon': -117.2898,
        'climate_zone': 'inland_valley',
        'urban_type': 'suburban',
        'population_density': 41,
        'elevation': 335,
        'avg_temp': 18.9,
        'coastal': False
    },
    
    # SALTON SEA AIR BASIN (SS)
    {
        'name': 'Riverside County (Salton Sea/Coachella Valley)',
        'coabdis': 'Riverside (SS)',
        'county': 'Riverside',
        'air_basin': 'Salton Sea',
        'lat': 33.8303, 'lon': -116.5453,
        'climate_zone': 'desert',
        'urban_type': 'suburban',
        'population_density': 36,
        'elevation': 140,
        'avg_temp': 24.5,
        'coastal': False
    },
    {
        'name': 'Imperial County (Salton Sea)',
        'coabdis': 'Imperial (SS)',
        'county': 'Imperial',
        'air_basin': 'Salton Sea',
        'lat': 32.8393, 'lon': -115.3630,
        'climate_zone': 'desert',
        'urban_type': 'rural',
        'population_density': 17,
        'elevation': -15,
        'avg_temp': 23.5,
        'coastal': False
    },
    
    # LAKE TAHOE AIR BASIN (LT)
    {
        'name': 'Placer County (Lake Tahoe)',
        'coabdis': 'Placer (LT)',
        'county': 'Placer',
        'air_basin': 'Lake Tahoe',
        'lat': 39.1981, 'lon': -120.1408,
        'climate_zone': 'mountain',
        'urban_type': 'suburban',
        'population_density': 40,
        'elevation': 1899,
        'avg_temp': 6.2,
        'coastal': False
    },
    {
        'name': 'El Dorado County (Lake Tahoe)',
        'coabdis': 'El Dorado (LT)',
        'county': 'El Dorado',
        'air_basin': 'Lake Tahoe',
        'lat': 38.9099, 'lon': -120.0324,
        'climate_zone': 'mountain',
        'urban_type': 'suburban',
        'population_density': 35,
        'elevation': 1899,
        'avg_temp': 6.2,
        'coastal': False
    },
]

# Verify all districts from the user's list are present
user_coabdis_list = ['Alpine (GBV)', 'Mono (GBV)', 'Inyo (GBV)', 'Lake (LC)',
       'Calaveras (MC)', 'Sierra (MC)', 'Plumas (MC)', 'Tuolumne (MC)',
       'Monterey (NCC)', 'Santa Cruz (NCC)', 'Del Norte (NC)',
       'Mariposa (MC)', 'Nevada (MC)', 'Placer (MC)', 'El Dorado (MC)',
       'Humboldt (NC)', 'Mendocino (NC)', 'Siskiyou (NEP)',
       'Trinity (NC)', 'Modoc (NEP)', 'Lassen (NEP)', 'Glenn (SV)',
       'Colusa (SV)', 'Butte (SV)', 'Shasta (SV)', 'Placer (SV)',
       'Sacramento (SV)', 'San Diego (SD)', 'Yolo (SV)',
       'Contra Costa (SF)', 'Marin (SF)', 'Alameda (SF)', 'Tehama (SV)',
       'Sutter (SV)', 'San Francisco (SF)', 'Santa Clara (SF)',
       'San Mateo (SF)', 'Los Angeles (MD)', 'Kern (MD)',
       'San Bernardino (MD)', 'Fresno (SJV)', 'Madera (SJV)',
       'Kings (SJV)', 'Kern (SJV)', 'Tulare (SJV)', 'Stanislaus (SJV)',
       'San Luis Obispo (SCC)', 'Santa Barbara (SCC)', 'Ventura (SCC)',
       'Los Angeles (SC)', 'San Joaquin (SJV)', 'Merced (SJV)',
       'Orange (SC)', 'Riverside (SC)', 'San Bernardino (SC)',
       'Riverside (SS)', 'Riverside (MD/MDAQMD)', 'Riverside (MD/SCAQMD)',
       'Solano (SV)', 'Solano (SF)', 'Yuba (SV)', 'Napa (SF)',
       'Sonoma (SF)', 'Imperial (SS)', 'San Benito (NCC)', 'Amador (MC)',
       'Placer (LT)', 'El Dorado (LT)', 'Sonoma (NC)']

class OnRoadEMFACTripSchema(RouteClipsSchemaPts, pa.DataFrameModel):
    region: Series[str] = pa.Field(isin=user_coabdis_list)
    EMFAC_class: Series[str]  # FIXME: validate 

class EmissionsSchema(pa.DataFrameModel):
    fuel_consumption_u:   Series[Annotated[PintDtype,'gal_diesel']]      = pa.Field(nullable=True)
    energy_consumption_u: Series[Annotated[PintDtype,'kilowatt * hour']] = pa.Field(nullable=True)
    emiss_pm25_u:         Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_pm10_u:         Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_nox_u:          Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_co_u:           Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_co2_u:          Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_ch4_u:          Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_n2o_u:          Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    emiss_ghg_u:          Series[Annotated[PintDtype,'g']]               = pa.Field(nullable=True)
    
    @pa.parser('emiss_pm25_u', 'emiss_pm10_u', 'emiss_nox_u', 'emiss_co_u', 
               'emiss_co2_u', 'emiss_ch4_u', 'emiss_n2o_u', 'emiss_ghg_u')
    def check_mass_dimension(cls, series):
        return pint_dimension_parser(reference_unit='g')(series)

    @pa.parser('fuel_consumption_u', 'energy_consumption_u')
    def check_energy_dimension(cls, series):
        return pint_dimension_parser(reference_unit='kilowatt * hour')(series)

class OnRoadEMFACTripEmissionsSchema(OnRoadEMFACTripSchema,EmissionsSchema):
    pass

generated_coabdis = [d['coabdis'] for d in carb_districts]

logger.trace(f"Total districts generated: {len(carb_districts)}")
logger.trace(f"Total districts expected: {len(user_coabdis_list)}")
logger.trace(f"\nAll districts match: {set(user_coabdis_list) == set(generated_coabdis)}")

# Check for missing or extra districts
missing = set(user_coabdis_list) - set(generated_coabdis)
extra = set(generated_coabdis) - set(user_coabdis_list)

if missing:
    logger.error(f"\nMissing districts: {missing}")
    raise Exception(f"\nMissing districts: {missing}")
if extra:
    logger.error(f"\nExtra districts: {extra}")
    raise Exception(f"\nExtra districts: {extra}")


# representative state characteristics
state_locations = {
    'Alabama': {
        'name': 'Alabama',
        'city': 'Birmingham',
        'lat': 33.5186, 'lon': -86.8104,
        'climate_zone': 'mediterranean',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 97,
        'elevation': 180,
        'avg_temp': 17.2,
        'coastal': True
    },
    'Alaska': {
        'name': 'Alaska',
        'city': 'Anchorage',
        'lat': 61.2181, 'lon': -149.9003,
        'climate_zone': 'mountain',  # Subarctic
        'urban_type': 'suburban',
        'population_density': 0.5,
        'elevation': 30,
        'avg_temp': 2.8,
        'coastal': True
    },
    'Arizona': {
        'name': 'Arizona',
        'city': 'Phoenix',
        'lat': 33.4484, 'lon': -112.0740,
        'climate_zone': 'desert',
        'urban_type': 'very_urban',
        'population_density': 230,
        'elevation': 330,
        'avg_temp': 24.0,
        'coastal': False
    },
    'Arkansas': {
        'name': 'Arkansas',
        'city': 'Little Rock',
        'lat': 34.7465, 'lon': -92.2896,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'suburban',
        'population_density': 58,
        'elevation': 100,
        'avg_temp': 16.7,
        'coastal': False
    },
    'Colorado': {
        'name': 'Colorado',
        'city': 'Denver',
        'lat': 39.7392, 'lon': -104.9903,
        'climate_zone': 'mountain',  # Semi-arid/high elevation
        'urban_type': 'urban',
        'population_density': 56,
        'elevation': 1609,
        'avg_temp': 10.6,
        'coastal': False
    },
    'Connecticut': {
        'name': 'Connecticut',
        'city': 'Hartford',
        'lat': 41.7658, 'lon': -72.6734,
        'climate_zone': 'bay_area',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 286,
        'elevation': 15,
        'avg_temp': 10.6,
        'coastal': True
    },
    'Delaware': {
        'name': 'Delaware',
        'city': 'Wilmington',
        'lat': 39.7391, 'lon': -75.5398,
        'climate_zone': 'coastal',
        'urban_type': 'suburban',
        'population_density': 189,
        'elevation': 30,
        'avg_temp': 13.3,
        'coastal': True
    },
    'Florida': {
        'name': 'Florida',
        'city': 'Miami',
        'lat': 25.7617, 'lon': -80.1918,
        'climate_zone': 'coastal',  # Tropical/subtropical
        'urban_type': 'very_urban',
        'population_density': 141,
        'elevation': 2,
        'avg_temp': 24.7,
        'coastal': True
    },
    'Georgia': {
        'name': 'Georgia',
        'city': 'Atlanta',
        'lat': 33.7490, 'lon': -84.3880,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'very_urban',
        'population_density': 185,
        'elevation': 320,
        'avg_temp': 16.7,
        'coastal': True
    },
    'Hawaii': {
        'name': 'Hawaii',
        'city': 'Honolulu',
        'lat': 21.3099, 'lon': -157.8581,
        'climate_zone': 'coastal',  # Tropical
        'urban_type': 'urban',
        'population_density': 86,
        'elevation': 5,
        'avg_temp': 25.0,
        'coastal': True
    },
    'Idaho': {
        'name': 'Idaho',
        'city': 'Boise',
        'lat': 43.6150, 'lon': -116.2023,
        'climate_zone': 'desert',  # Semi-arid
        'urban_type': 'suburban',
        'population_density': 22,
        'elevation': 823,
        'avg_temp': 11.7,
        'coastal': False
    },
    'Illinois': {
        'name': 'Illinois',
        'city': 'Chicago',
        'lat': 41.8781, 'lon': -87.6298,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'very_urban',
        'population_density': 231,
        'elevation': 181,
        'avg_temp': 10.6,
        'coastal': False
    },
    'Indiana': {
        'name': 'Indiana',
        'city': 'Indianapolis',
        'lat': 39.7684, 'lon': -86.1581,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'urban',
        'population_density': 188,
        'elevation': 220,
        'avg_temp': 11.7,
        'coastal': False
    },
    'Iowa': {
        'name': 'Iowa',
        'city': 'Des Moines',
        'lat': 41.5868, 'lon': -93.6250,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 56,
        'elevation': 245,
        'avg_temp': 10.0,
        'coastal': False
    },
    'Kansas': {
        'name': 'Kansas',
        'city': 'Wichita',
        'lat': 37.6872, 'lon': -97.3301,
        'climate_zone': 'inland_valley',  # Humid continental/semi-arid
        'urban_type': 'suburban',
        'population_density': 35,
        'elevation': 402,
        'avg_temp': 13.9,
        'coastal': False
    },
    'Kentucky': {
        'name': 'Kentucky',
        'city': 'Louisville',
        'lat': 38.2527, 'lon': -85.7585,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 113,
        'elevation': 142,
        'avg_temp': 14.4,
        'coastal': False
    },
    'Louisiana': {
        'name': 'Louisiana',
        'city': 'New Orleans',
        'lat': 29.9511, 'lon': -90.0715,
        'climate_zone': 'coastal',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 108,
        'elevation': 0,
        'avg_temp': 20.8,
        'coastal': True
    },
    'Maine': {
        'name': 'Maine',
        'city': 'Portland',
        'lat': 43.6591, 'lon': -70.2568,
        'climate_zone': 'coastal',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 17,
        'elevation': 20,
        'avg_temp': 7.8,
        'coastal': True
    },
    'Maryland': {
        'name': 'Maryland',
        'city': 'Baltimore',
        'lat': 39.2904, 'lon': -76.6122,
        'climate_zone': 'bay_area',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 244,
        'elevation': 10,
        'avg_temp': 13.9,
        'coastal': True
    },
    'Massachusetts': {
        'name': 'Massachusetts',
        'city': 'Boston',
        'lat': 42.3601, 'lon': -71.0589,
        'climate_zone': 'coastal',  # Humid continental
        'urban_type': 'very_urban',
        'population_density': 346,
        'elevation': 5,
        'avg_temp': 10.6,
        'coastal': True
    },
    'Michigan': {
        'name': 'Michigan',
        'city': 'Detroit',
        'lat': 42.3314, 'lon': -83.0458,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'urban',
        'population_density': 175,
        'elevation': 190,
        'avg_temp': 10.0,
        'coastal': False
    },
    'Minnesota': {
        'name': 'Minnesota',
        'city': 'Minneapolis',
        'lat': 44.9778, 'lon': -93.2650,
        'climate_zone': 'inland_valley',  # Humid continental (cold)
        'urban_type': 'urban',
        'population_density': 71,
        'elevation': 260,
        'avg_temp': 7.2,
        'coastal': False
    },
    'Mississippi': {
        'name': 'Mississippi',
        'city': 'Jackson',
        'lat': 32.2988, 'lon': -90.1848,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'suburban',
        'population_density': 63,
        'elevation': 85,
        'avg_temp': 18.3,
        'coastal': True
    },
    'Missouri': {
        'name': 'Missouri',
        'city': 'Kansas City',
        'lat': 39.0997, 'lon': -94.5786,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'urban',
        'population_density': 89,
        'elevation': 277,
        'avg_temp': 13.3,
        'coastal': False
    },
    'Montana': {
        'name': 'Montana',
        'city': 'Billings',
        'lat': 45.7833, 'lon': -108.5007,
        'climate_zone': 'mountain',  # Semi-arid
        'urban_type': 'suburban',
        'population_density': 3,
        'elevation': 950,
        'avg_temp': 7.8,
        'coastal': False
    },
    'Nebraska': {
        'name': 'Nebraska',
        'city': 'Omaha',
        'lat': 41.2565, 'lon': -95.9345,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 25,
        'elevation': 320,
        'avg_temp': 10.6,
        'coastal': False
    },
    'Nevada': {
        'name': 'Nevada',
        'city': 'Las Vegas',
        'lat': 36.1699, 'lon': -115.1398,
        'climate_zone': 'desert',
        'urban_type': 'urban',
        'population_density': 28,
        'elevation': 610,
        'avg_temp': 20.6,
        'coastal': False
    },
    'New Hampshire': {
        'name': 'New Hampshire',
        'city': 'Manchester',
        'lat': 42.9956, 'lon': -71.4548,
        'climate_zone': 'mountain',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 59,
        'elevation': 80,
        'avg_temp': 8.3,
        'coastal': True
    },
    'New Jersey': {
        'name': 'New Jersey',
        'city': 'Newark',
        'lat': 40.7357, 'lon': -74.1724,
        'climate_zone': 'bay_area',  # Humid subtropical
        'urban_type': 'very_urban',
        'population_density': 467,
        'elevation': 10,
        'avg_temp': 12.8,
        'coastal': True
    },
    'New Mexico': {
        'name': 'New Mexico',
        'city': 'Albuquerque',
        'lat': 35.0844, 'lon': -106.6504,
        'climate_zone': 'desert',  # Semi-arid
        'urban_type': 'suburban',
        'population_density': 7,
        'elevation': 1619,
        'avg_temp': 14.4,
        'coastal': False
    },
    'New York': {
        'name': 'New York',
        'city': 'New York City',
        'lat': 40.7128, 'lon': -74.0060,
        'climate_zone': 'bay_area',  # Humid subtropical
        'urban_type': 'very_urban',
        'population_density': 159,
        'elevation': 10,
        'avg_temp': 12.8,
        'coastal': True
    },
    'North Carolina': {
        'name': 'North Carolina',
        'city': 'Charlotte',
        'lat': 35.2271, 'lon': -80.8431,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 218,
        'elevation': 229,
        'avg_temp': 16.1,
        'coastal': True
    },
    'North Dakota': {
        'name': 'North Dakota',
        'city': 'Fargo',
        'lat': 46.8772, 'lon': -96.7898,
        'climate_zone': 'inland_valley',  # Humid continental (very cold)
        'urban_type': 'suburban',
        'population_density': 4,
        'elevation': 274,
        'avg_temp': 5.6,
        'coastal': False
    },
    'Ohio': {
        'name': 'Ohio',
        'city': 'Columbus',
        'lat': 39.9612, 'lon': -82.9988,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'urban',
        'population_density': 287,
        'elevation': 270,
        'avg_temp': 11.7,
        'coastal': False
    },
    'Oklahoma': {
        'name': 'Oklahoma',
        'city': 'Oklahoma City',
        'lat': 35.4676, 'lon': -97.5164,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'suburban',
        'population_density': 57,
        'elevation': 370,
        'avg_temp': 15.6,
        'coastal': False
    },
    'Oregon': {
        'name': 'Oregon',
        'city': 'Portland',
        'lat': 45.5152, 'lon': -122.6784,
        'climate_zone': 'coastal',  # Mediterranean/Oceanic
        'urban_type': 'urban',
        'population_density': 44,
        'elevation': 15,
        'avg_temp': 12.2,
        'coastal': True
    },
    'Pennsylvania': {
        'name': 'Pennsylvania',
        'city': 'Philadelphia',
        'lat': 39.9526, 'lon': -75.1652,
        'climate_zone': 'bay_area',  # Humid continental
        'urban_type': 'very_urban',
        'population_density': 286,
        'elevation': 12,
        'avg_temp': 13.3,
        'coastal': False
    },
    'Rhode Island': {
        'name': 'Rhode Island',
        'city': 'Providence',
        'lat': 41.8240, 'lon': -71.4128,
        'climate_zone': 'coastal',  # Humid continental
        'urban_type': 'urban',
        'population_density': 398,
        'elevation': 15,
        'avg_temp': 10.6,
        'coastal': True
    },
    'South Carolina': {
        'name': 'South Carolina',
        'city': 'Charleston',
        'lat': 32.7765, 'lon': -79.9311,
        'climate_zone': 'coastal',  # Humid subtropical
        'urban_type': 'suburban',
        'population_density': 173,
        'elevation': 6,
        'avg_temp': 18.9,
        'coastal': True
    },
    'South Dakota': {
        'name': 'South Dakota',
        'city': 'Sioux Falls',
        'lat': 43.5446, 'lon': -96.7311,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 4,
        'elevation': 435,
        'avg_temp': 7.8,
        'coastal': False
    },
    'Tennessee': {
        'name': 'Tennessee',
        'city': 'Nashville',
        'lat': 36.1627, 'lon': -86.7816,
        'climate_zone': 'inland_valley',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 167,
        'elevation': 182,
        'avg_temp': 15.0,
        'coastal': False
    },
    'Texas': {
        'name': 'Texas',
        'city': 'Houston',
        'lat': 29.7604, 'lon': -95.3698,
        'climate_zone': 'coastal',  # Humid subtropical
        'urban_type': 'very_urban',
        'population_density': 112,
        'elevation': 13,
        'avg_temp': 20.8,
        'coastal': True
    },
    'Utah': {
        'name': 'Utah',
        'city': 'Salt Lake City',
        'lat': 40.7608, 'lon': -111.8910,
        'climate_zone': 'desert',  # Semi-arid
        'urban_type': 'suburban',
        'population_density': 14,
        'elevation': 1288,
        'avg_temp': 11.1,
        'coastal': False
    },
    'Vermont': {
        'name': 'Vermont',
        'city': 'Burlington',
        'lat': 44.4759, 'lon': -73.2121,
        'climate_zone': 'mountain',  # Humid continental
        'urban_type': 'suburban',
        'population_density': 27,
        'elevation': 61,
        'avg_temp': 7.2,
        'coastal': False
    },
    'Virginia': {
        'name': 'Virginia',
        'city': 'Virginia Beach',
        'lat': 36.8529, 'lon': -75.9780,
        'climate_zone': 'coastal',  # Humid subtropical
        'urban_type': 'urban',
        'population_density': 218,
        'elevation': 4,
        'avg_temp': 15.6,
        'coastal': True
    },
    'Washington': {
        'name': 'Washington',
        'city': 'Seattle',
        'lat': 47.6062, 'lon': -122.3321,
        'climate_zone': 'coastal',  # Oceanic
        'urban_type': 'urban',
        'population_density': 113,
        'elevation': 53,
        'avg_temp': 11.1,
        'coastal': True
    },
    'West Virginia': {
        'name': 'West Virginia',
        'city': 'Charleston',
        'lat': 38.3498, 'lon': -81.6326,
        'climate_zone': 'mountain',  # Humid subtropical
        'urban_type': 'suburban',
        'population_density': 30,
        'elevation': 181,
        'avg_temp': 13.3,
        'coastal': False
    },
    'Wisconsin': {
        'name': 'Wisconsin',
        'city': 'Milwaukee',
        'lat': 43.0389, 'lon': -87.9065,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'urban',
        'population_density': 108,
        'elevation': 188,
        'avg_temp': 8.3,
        'coastal': False
    },
    'Wyoming': {
        'name': 'Wyoming',
        'city': 'Cheyenne',
        'lat': 41.1400, 'lon': -104.8202,
        'climate_zone': 'mountain',  # Semi-arid
        'urban_type': 'suburban',
        'population_density': 2,
        'elevation': 1848,
        'avg_temp': 7.2,
        'coastal': False
    },
    'District of Columbia': {
        'name': 'District of Columbia',
        'city': 'Washington',
        'lat': 38.9072, 'lon': -77.0369,
        'climate_zone': 'bay_area',  # Humid subtropical
        'urban_type': 'very_urban',
        'population_density': 4361,  # One of the highest in the US
        'elevation': 125,
        'avg_temp': 14.4,
        'coastal': True  # Near Chesapeake Bay/Potomac River
    },
    'Mexico': {
        'name': 'Mexico',
        'city': 'Mexico City',
        'lat': 19.4326, 'lon': -99.1332,
        'climate_zone': 'inland_valley',  # Subtropical highland
        'urban_type': 'very_urban',
        'population_density': 66,
        'elevation': 2240,  # High altitude capital
        'avg_temp': 16.5,
        'coastal': True
    },

    'Canada': {
        'name': 'Canada (East)',
        'city': 'Toronto',
        'lat': 43.6532, 'lon': -79.3832,
        'climate_zone': 'inland_valley',  # Humid continental
        'urban_type': 'very_urban',
        'population_density': 4,  # National average is very low
        'elevation': 76,
        'avg_temp': 9.4,
        'coastal': True
    }
}

import numpy as np
from scipy.spatial.distance import euclidean

from typing import Annotated, TypedDict, List

class AirDistrictCharacteristics(TypedDict):
    climate_zone: str
    urban_type: str
    population_density: float
    elevation: float
    avg_temp: float
    costal: bool

@pa.check_types
def select_best_carb_district_for_emissions(target_lat: float, 
                                            target_lon: float, 
                                             target_characteristics: AirDistrictCharacteristics,
                                             carb_districts=carb_districts):
    """
    Select the CARB district whose emissions profile best matches the target location.
    
    Parameters:
    -----------
    target_lat : float
        Latitude of target location
    target_lon : float
        Longitude of target location
    target_characteristics : dict
        Characteristics of target location:
        {
            'climate_zone': str,  # 'desert', 'mediterranean', 'mountain', 'coastal', etc.
            'urban_type': str,    # 'urban', 'suburban', 'rural'
            'population_density': float,  # people per sq km
            'elevation': float,   # meters
            'avg_temp': float,    # annual avg temperature °C
            'coastal': bool       # True if within 50km of coast
        }
    carb_districts : list of dict
        CARB districts with characteristics
    
    Returns:
    --------
    dict : Best matching CARB district with similarity score
    """
    
    def calculate_emissions_region_similarity_score(target_chars, district_chars):
        """Compute a scalar dissimilarity score between a target location and a CARB district.

        Penalties are applied for mismatches in climate zone (largest weight),
        urban classification, average temperature, elevation, coastal status,
        population density (log-scaled), and geographic distance (smallest weight).

        Args:
            target_chars: Attribute dict for the target location.
            district_chars: Attribute dict for a candidate CARB district, including
                ``lat`` and ``lon`` keys used for geographic distance.

        Returns:
            Non-negative float score; 0 indicates a perfect match on all criteria.
        """
        score = 0
        
        # Climate zone matching (most important for emissions)
        climate_match = {
            'desert': ['desert', 'semi_arid', 'inland_valley'],
            'mediterranean': ['mediterranean', 'coastal', 'bay_area'],
            'mountain': ['mountain', 'sierra', 'high_elevation'],
            'continental': ['inland_valley', 'mountain', 'desert'],
            'coastal': ['coastal', 'bay_area', 'mediterranean']
        }
        
        target_climate = target_chars.get('climate_zone', '')
        district_climate = district_chars.get('climate_zone', '')
        
        if district_climate == target_climate:
            score += 0  # Perfect match
        elif district_climate in climate_match.get(target_climate, []):
            score += 5  # Similar climate
        else:
            score += 15  # Different climate (heavy penalty)
        
        # Urban/rural classification (affects fleet mix and VMT)
        urban_types = {'rural': 0, 'suburban': 1, 'urban': 2, 'very_urban': 3}
        target_urban = urban_types.get(target_chars.get('urban_type', 'suburban'), 1)
        district_urban = urban_types.get(district_chars.get('urban_type', 'suburban'), 1)
        score += abs(target_urban - district_urban) * 3
        
        # Temperature similarity (affects evaporative emissions)
        temp_diff = abs(target_chars.get('avg_temp', 15) - 
                       district_chars.get('avg_temp', 15))
        score += temp_diff * 0.5
        
        # Elevation similarity (affects fuel combustion)
        elev_diff = abs(target_chars.get('elevation', 0) - 
                       district_chars.get('elevation', 0)) / 1000
        score += elev_diff * 2
        
        # Coastal vs inland (affects humidity, temperature swings)
        if target_chars.get('coastal', False) != district_chars.get('coastal', False):
            score += 5
        
        # Population density (log scale, affects congestion and fleet)
        target_pop = np.log10(target_chars.get('population_density', 100) + 1)
        district_pop = np.log10(district_chars.get('population_density', 100) + 1)
        score += abs(target_pop - district_pop) * 2
        
        # Geographic distance as minor factor
        dist_lat = abs(target_lat - district_chars.get('lat', target_lat))
        dist_lon = abs(target_lon - district_chars.get('lon', target_lon))
        geo_distance = np.sqrt(dist_lat**2 + dist_lon**2)
        score += geo_distance * 0.5  # Small weight
        
        return score
    
    # Score all districts
    best_score = float('inf')
    best_district = None
    
    for district in carb_districts:
        score = calculate_emissions_region_similarity_score(target_characteristics, district)
        
        if score < best_score:
            best_score = score
            best_district = district.copy()
    
    best_district['similarity_score'] = best_score
    return best_district

@cachier(wait_for_calc_timeout=5)
def read_emfac_rates(model_year,vehcat):
    """Load EMFAC2021 emission rates for specified vehicle categories and model year.

    Reads the EMFAC2021 emissions-rate (ER) CSV, cleans column names, and
    filters to the requested vehicle categories and calendar year.

    Args:
        model_year: Calendar year to retain from the EMFAC rates file.
        vehcat: Iterable of EMFAC vehicle category strings to include
            (e.g. ``['T7 Tractor Class 8']``).

    Returns:
        Filtered DataFrame of per-speed-bin emission rates (g/mi) and fuel/
        energy consumption factors for the requested categories and year.
    """
    logger.info("Reading EMFAC rates")
    rates = (
        pd.read_csv(cpath('emfac_data', 'EMFAC2021-ER-202xClass-all-subareas.csv'), skiprows=8)
        .clean_names()
        .filt(lambda x: (
            True
            # &(x.vehicle_category=='T7 Tractor Class 8')
            &(x.vehicle_category.isin(vehcat))

            # &(x.fuel=='Diesel')
            &(x.calendar_year == model_year)
        ))
    )
    # FROM EMFAC2021-ER-202xClass-all-subareas.csv documentation: Units:
    # * miles/day for CVMT and EVMT
    # * g/mile for RUNEX, PMBW and PMTW
    # * mph for Speed
    # * kWh/mile for Energy Consumption, 
    # * gallon/mile for Fuel Consumption. 
    # * PHEV calculated based on total VMT.

    fuel_units = rates.fuel.map(fuel_to_units_conversion) # used for fuel_consumption_u column below
    rr = (
        rates
        .pipe(assign_units,{c: f'mi' for c in rates.filter(regex="^([ce]|total_)vmt$").columns})
        .pipe(assign_units,{c: f'g / mi' for c in rates.filter(regex="_(runex|pmbw|pmtw)$").columns})
        .pipe(assign_units,{c: f'mi / hr' for c in rates.filter(regex="^speed$").columns})
        .pipe(assign_units,{c: f'kilowatt * hr / mi' for c in rates.filter(regex="^energy_consumption$").columns})
        
        # fuel units are trickier, could do this with assign(,axis=1) but this is more efficient
        .pipe(lambda x: x.assign(fuel_consumption=pd.Series(
            [(fc * ureg(f'{u} / mi')).to('gal_diesel / mi') for fc, u in zip(rates.fuel_consumption.astype(float), fuel_units)],
            index=x.index
        ).astype('pint[gal_diesel / mi]'))) # everything converted to gal_diesel/mi (dge/mi) for consistency
        
        # rename to _u to indicate units are attached
        .pipe(lambda df: df.rename(columns={
                c: f'{c}_u' for c 
                in df.filter(regex='(^([ce]|total_)vmt$)|(_(runex|pmbw|pmtw)$)|(^speed$)|(_consumption$)')
            })
        )
    )
    return rr

# @cachier(wait_for_calc_timeout=5)
@pa.check_types
def compute_on_road_emissions(
    od_flows_w_dist:DataFrame[OnRoadEMFACTripSchema], 
    model_year: int
    ) -> DataFrame[OnRoadEMFACTripEmissionsSchema]:
    """Calculate per-segment on-road vehicle emissions for OD flow route clips.

    The process proceeds in four stages:

    1. **Rate preparation** — Loads EMFAC emission rates for the requested model
       year and vehicle categories, then collapses multiple fuel-type rows into a
       single VMT-weighted average rate per (region, vehicle_category, speed) bin.

    2. **Speed-bin expansion** — Constructs a complete cross-product of
       (region, EMFAC_class, speed) combinations observed in ``od_flows_w_dist``
       and outer-joins it onto the averaged rates.  Any gaps introduced by the
       join are filled via ``ffill`` then ``bfill`` within each
       (region, vehicle_category) group, ensuring every speed bin has a rate.

    3. **Rate join with statewide fallback** — Reprojects clip geometries to a
       US metre CRS for accurate length calculation, snaps each segment's speed
       to the nearest 5 mph bin (clamped to the available rate range), and joins
       the regional rates.  Any remaining NaNs—caused by
       region/vehicle-class combinations absent from EMFAC—are filled with
       statewide averages (mean across all regions) for the matching
       vehicle_category and speed bin.  A ``RuntimeError`` is raised if any rows
       remain unmatched after both fill strategies.

    4. **Emission calculation** — Multiplies per-mile rates by segment VMT to
       produce fuel consumption (gal DGE), energy consumption (kWh), and
       per-pollutant emission masses (PM2.5, PM10, NOx, CO, CO2, CH4, N2O, and
       GHG CO2e using AR5 100-year GWPs: CH4 × 28, N2O × 265).  All outputs
       are converted to the units defined in ``emiss_unit``.

    Args:
        od_flows_w_dist: Validated route-clip DataFrame conforming to
            ``OnRoadEMFACTripSchema``, including clip geometry, segment speed,
            trip counts, tonnage, and EMFAC vehicle class assignment per row.
        model_year: Calendar year for which EMFAC emission rates are loaded.
            Also used to filter the rates to a single model-year aggregate.

    Returns:
        Input DataFrame augmented with clip distance (pint metres), speed-bin
        assignment, VMT (miles), fuel/energy consumption, and all emission
        columns, conforming to ``OnRoadEMFACTripEmissionsSchema``.

    Raises:
        AssertionError: If the expanded rate table contains any null speed values
            after ffill/bfill, indicating an unexpected gap in the rate grid.
        RuntimeError: If any OD flow rows remain without matched emission rates
            after both the regional join and the statewide-mean fallback.
    """
    logger.info("Compute emissions: Non-collection")

    logger.info("Compute emissions: Non-collection: Reading EMFAC rates")

    emfac_rates = read_emfac_rates(model_year, od_flows_w_dist.EMFAC_class.unique())
    grpcols = ['region', 'calendar_year', 'vehicle_category', 'model_year', 'speed_u']
    ratescols = [f'{c}_u' for c in ['nox_runex', 'pm2_5_runex', 'pm10_runex', 'co2_runex', 'ch4_runex', 'n2o_runex',
                    'rog_runex', 'tog_runex', 'co_runex', 'sox_runex', 'nh3_runex', 'pm10_pmbw',
                    'pm2_5_pmbw', 'fuel_consumption', 'energy_consumption']]

    def group_avg(gdf,subset=None):
        """Compute VMT-weighted average emission rates across fuel types within a group.

        Intended for use inside a ``groupby(...).apply()`` call that groups EMFAC
        rate rows by region, calendar year, vehicle category, model year, and speed
        bin—collapsing multiple fuel-type rows into a single blended rate.

        Args:
            gdf: Sub-DataFrame for one group, containing ``ratescols`` and
                ``total_vmt`` columns.

        Returns:
            Series of VMT-weighted average values for each column in ``ratescols``.
        """
        if subset is not None:
            gdf = gdf[subset]
        return(gdf.apply(lambda x: np.average(x, weights=gdf.total_vmt_u)))
    
    # emfac_rates_adj = emfac_rates.groupby(grpcols)[ratescols + ['total_vmt']].apply(group_avg).reset_index()
    logger.info("Compute emissions: Non-collection: Collapsing EMFAC rates across fuel types")
    emfac_rates_adj = (
        emfac_rates.groupby(grpcols)[ratescols + ['total_vmt_u']]
        .apply(group_avg).reset_index()
        # convert back to pint array dtype after groupby/apply (which converts to in-cell quantities)
        # see: https://pint-pandas.readthedocs.io/en/latest/user/common.html#units-in-cells
        .pint.convert_object_dtype()    
    )           

    # FIXME: check od_flows_w_dist for null speeds --- these are unmapped segments

    # FIXME: EMFAC doesn't necessarily have bins for all speeds <- this is a hack to deal with that
    # first, create a complete set of region, vehicle class, speed bin combinations

    logger.info("Compute emissions: Non-collection: Expanding EMFAC rates for missing bins")

    erexp = (od_flows_w_dist.groupby(['region', 'EMFAC_class']).first()[[]].reset_index()
                .join(pd.DataFrame({'speed_u': range(emfac_rates_adj.speed_u.pint.magnitude.min().astype(int), 
                                                     emfac_rates_adj.speed_u.pint.magnitude.max().astype(int), 5)})
                .pipe(assign_units,{'speed_u':'mi / hr'}), how='cross')
                )
    # then 
    emfac_rates_adj2 = (
        # do an outer join of these onto the emfac rates table
        emfac_rates_adj.reset_index(drop=True)
        .merge(erexp,
                  left_on=['region', 'vehicle_category', 'speed_u']
                  ,right_on=['region', 'EMFAC_class', 'speed_u']
                  , how='outer')
        .assign(
            vehicle_category=lambda x: x.vehicle_category.fillna(x.EMFAC_class)
        )
        
        .drop(columns=['EMFAC_class']) # done using this---keeping it will clash on merge below

        # clean up the speed bin categories to make sure they're complete by
        # forward filling then backward filling within each region/class group
        .pipe(lambda df: df.assign(
            **df.groupby(['region', 'vehicle_category'])[
                list(set(df.columns) - set(['region', 'vehicle_category']))
            ]
            .transform(lambda x: x.ffill().bfill())
        ))
    )
    
    assert not emfac_rates_adj2.speed_u.isna().any(), "Missing rates for some region, vehicle_cat, speed combinations"
    
    logger.info("Compute emissions: Non-collection: join rates and calulate emissions")

    # note, for ER all classes:
    # "Units: miles/day for CVMT and EVMT, g/mile for RUNEX, PMBW and PMTW, mph for Speed, 
    #         kWh/mile for Energy Consumption, gallon/mile for Fuel Consumption. 
    #         PHEV calculated based on total VMT."
    emissx = (
        od_flows_w_dist
        .join(emfac_rates_adj2.groupby(['region', 'vehicle_category'])[['speed_u']].min()
              .rename(columns={'speed_u': 'speed_min_u'}), on=['region', 'EMFAC_class'])
        .join(emfac_rates_adj2.groupby(['region', 'vehicle_category'])[['speed_u']].max()
              .rename(columns={'speed_u': 'speed_max_u'}), on=['region', 'EMFAC_class'])
        
        # FIXME: try set_geometry(clip_distance).to_crs(crs_usa) for better length calcs
        .set_geometry('geometry_clip')
        .assign(
            clip_distance=lambda x: x.geometry_clip.to_crs(crs_usa).length.astype('pint[m]'),  # meters

            # assign each step to a speed bin
            speed_bin_u=lambda x: (
                # snap to nearest 5 mph bin, but make sure we're within the min/max range for the vehicle class
                # calcs need to be done in magnitude space, then convert back to pint array dtype
                (np.floor((x.step_speed_u.pint.to('mi / hr').pint.magnitude + 0.0001) / 5) * 5)

                # hack to fix to int
                .fillna(0).astype(int)

                # make sure we're within bounds
                .clip(lower=x.speed_min_u.pint.to('mi / hr').pint.magnitude,
                      upper=x.speed_max_u.pint.to('mi / hr').pint.magnitude)
                
                .astype('pint[mi / hr]')
            ),
            vmt_u=lambda x: x.trips * x.clip_distance.pint.to('mi')
        )
        .join(emfac_rates_adj2.set_index(['region', 'speed_u', 'vehicle_category']), on=['region', 'speed_bin_u', 'EMFAC_class'])
    )    

    # FIXME: should consider *why* we're missing rates (most likely because the
    # vehicle class region combos are bogus)  
    # statewide averages across all regions, per vehicle class + speed bin
    # weighted by total_vmt
    statewide_means = (
        emfac_rates_adj2
        .filt(lambda x: x.total_vmt_u.notna()) # remove NAs for the weighted average
        .groupby(['vehicle_category', 'speed_u']) 
        .apply(lambda g: pd.Series({
            col: np.average(g[col], weights=g['total_vmt_u'])
            for col in ratescols
        }))
        # convert back to pint array dtype after groupby/apply (which converts to in-cell quantities)
        # see: https://pint-pandas.readthedocs.io/en/latest/user/common.html#units-in-cells
        .pint.convert_object_dtype()    
    )
    
    if statewide_means.isna().any().any():
        with redirect_stdout_to_logging(logger.ERROR):
            logger.error("Missing emission rates for some rows in statewide_means dataframe.")
            display(statewide_means[statewide_means.isna().any(axis=1)])
            raise RuntimeError("Missing emission rates for some rows in statewide_means dataframe.")
    
    # fill NaNs after the join via a second join on the broader key
    emissxx = (
        emissx
        .join(
            statewide_means,
            on=['EMFAC_class', 'speed_bin_u'],
            rsuffix='_sw'
        )
        .assign(
            model_year="Statewide",   # FIXME: need to set values here or they'll end up in the output 
            calendar_year=model_year,
            total_vmt_u = -1*ureg('mi'), # as NAs and break equivalance comparisons, really should compute
            speed_u=-1*ureg('mph'),      # as NAs and break equivalance comparisons
            **{
                col: lambda df, c=col: df[c].fillna(df[f'{c}_sw'])
                for col in ratescols
            })
        .drop(columns=[f'{col}_sw' for col in ratescols])
    )    

    # check to make sure all rate combinations are covered    
    missing_rate_combos=(
        emissxx.filt(lambda x: x.pm10_runex_u.isna())
        [['region','EMFAC_class']]
        .drop_duplicates()
    )
    if not missing_rate_combos.empty:
        with redirect_stdout_to_logging(logger.ERROR):
            logger.error("Missing emission rates for some rows in emissions dataframe.")
            display(missing_rate_combos)
            raise RuntimeError("Missing emission rates for some rows in emissions dataframe.")
    
    # okay, everything is complete, compute emissions
    emiss = (
        emissxx
        .rename(columns={
            # distinguish fuel/energy consumption *rates* columns from the *consumption* columns we're about to calculate
            'fuel_consumption_u': 'fuel_consumption_r_u',  # gal/mi (DGE)
            'energy_consumption_u': 'energy_consumption_r_u',  # kWh/mi
        })
        .assign(
            # fuel consumption given as gal per mi
            fuel_consumption_u=lambda x: x.fuel_consumption_r_u.fillna(0) * x.vmt_u,  # gal (DGE) /mi
            # NOTE: (FC is for gaseous fuels, EC is for electricity, but we want EC from FC added in too since we're blending types here)
            energy_consumption_u=lambda x: (x.energy_consumption_r_u + x.fuel_consumption_r_u.pint.to('kilowatt * hour / mi')) * x.vmt_u,  # kWh
            emiss_pm25_u=lambda x: (x.pm2_5_runex_u + x.pm2_5_pmbw_u
                                    # +x.pm2_5_pmtw_u   # TW not in rates?
                                    ) * x.vmt_u,
            emiss_pm10_u=lambda x: (x.pm10_runex_u + x.pm10_pmbw_u
                                    # +x.pm10_pmtw_u   # TW not in rates?
                                    ) * x.vmt_u,
            emiss_nox_u=lambda x: (x.nox_runex_u) * x.vmt_u,
            emiss_co_u=lambda x:  (x.co_runex_u)  * x.vmt_u,
            emiss_co2_u=lambda x: (x.co2_runex_u) * x.vmt_u,
            emiss_ch4_u=lambda x: (x.ch4_runex_u) * x.vmt_u,
            emiss_n2o_u=lambda x: (x.n2o_runex_u) * x.vmt_u,
            
            # GHG components already converted to mass units, just need to apply the GWP factors
            emiss_ghg_u=compute_ar6_gwp_100
        )

    )
    # everything except geometry_full should be NA
    assert emiss.drop(columns='geometry_full').notna().all().all(), "Some have unexpected NAs after emissions calculation"
    return emiss

@cachier(wait_for_calc_timeout=5)
@pa.check_types
def load_coabdis() -> DataFrame[RegionSchema]:
    """Load and clean California County/Air Basin/District (CoAbDis) geometries.

    Reads the CARB CoAbDis shapefile, standardises county and basin name
    formatting, dissolves sub-district polygons, and constructs EMFAC-compatible
    ``region`` strings of the form ``"County (BasinAbbrev)"`` or
    ``"County (BasinAbbrev/DistAQMD)"`` for split districts.
    
    This subarea logic is adapted from Bo Liu's work on the CARB LCT project (19RD026).

    Returns:
        GeoDataFrame conforming to ``RegionSchema`` with one row per
        CoAbDis polygon and a ``region`` column matching EMFAC subarea names.
    """

    logger.info("Reading CoAbDis...")

    coabdis=(
        gpd.read_file(cpath('gis_data','ca_co_ab_dis.zip')).to_crs(crs_ll)

        # clean up names
        .clean_names()


        .assign(
            co_name = lambda x: np.where(x.co_name.isna(), "Sacramento",
                                            x.co_name.str.title()),
            basin_name = lambda x: (
                case_when(x.coabdis_id == 69, "Sacramento Valley",
                            x.basin_name == 'SAN FRANCISCO BAY AREA', 'San Francisco',
                            x.basin_name == 'NORTHEAST PLATEAU', 'North East Plateau',
                            x.basin_name == 'SAN DIEGO COUNTY', 'San Diego',
                            True, x.basin_name.str.title()
                )
            )
        )
        .dissolve(by=['co_name','basin_name','dis_name'])
        .reset_index()

        .assign(
            count=lambda x: x.groupby(['co_name','basin_name']).co_num.transform('count'),
            region=lambda x: (
                case_when(x['count'] == 1, x.co_name + " (" + x.basin_name.str.replace(r"[^::A-Z::]","",regex=True) +")",
                            x['count'] > 1, (
                                x.co_name + " (" + x.basin_name.str.replace(r"[^::A-Z::]","",regex=True) +
                                "/" + x.dis_name.str.replace(r"[^::A-Z::]","",regex=True) + "AQMD" +")"
                            ),
                            True, "NOMATCH")
            )
        )
    )
    logger.info("Reading CoAbDis...Done")
    return coabdis

@cachier(wait_for_calc_timeout=5)
@pa.check_types
def load_regions_extended() -> DataFrame[RegionSchema]:
    """Build a full region GeoDataFrame covering California, other US states, and border countries.

    Extends the California CoAbDis layer with pseudo-district rows for each
    non-California US state and for Mexico and Canada.  Each out-of-state region
    is assigned the most emissions-profile-compatible CARB district via
    ``select_best_carb_district_for_emissions`` using a representative city
    location from ``state_locations``.

    Returns:
        GeoDataFrame combining California CoAbDis polygons with synthetic region
        rows for all other modelled geographies, conforming to ``RegionSchema``.
    """
    coabdis = load_coabdis()

    # To accommodate regions outside of california, we're going to create pseudo coabdis for each state
    regions_oos = (
        gpd.GeoDataFrame([{
            'co_name':row.name,
            'basin_name':row.name,
            'dis_name':select_best_carb_district_for_emissions(row.lat,row.lon,state_locations[row.name])['coabdis'] + ' - ' + row.stusps,
            'geometry':row.geometry,
            'co_num':999,
            'basa_id':999,
            'dist_type':'best match',
            'disa_id':999,
            'island_nam': None,
            'island_id': 999,
            'bay_split':None,
            'bay_id':0,
            'coabdis_id':999,
            'coabdis_ar':0,
            'count': 1,
            'region': select_best_carb_district_for_emissions(row.lat,row.lon,state_locations[row.name])['coabdis']
            }
            for row
            in pyg.states(year=2021,cache=True).clean_names()[['name','stusps','geometry']].to_crs(crs_ll)
               .filt(lambda x: ~x.name.str.match('.*(Samoa|Guam|Mariana|California|Virgin Islands|Puerto)'))
               .assign(lon=lambda x: x.geometry.representative_point().x,lat=lambda x: x.geometry.representative_point().y).itertuples()
            ]
            ,geometry='geometry',crs=crs_ll)
    )

    regions_ooc = (
        gpd.GeoDataFrame([{
            'co_name':row.country,
            'basin_name':row.country,
            'dis_name':select_best_carb_district_for_emissions(row.lat,row.lon,state_locations[row.country])['coabdis'] + ' - ' + row.country,
            'geometry':row.geometry_agg,
            'co_num':999,
            'basa_id':999,
            'dist_type':'best match',
            'disa_id':999,
            'island_nam': None,
            'island_id': 999,
            'bay_split':None,
            'bay_id':0,
            'coabdis_id':999,
            'coabdis_ar':0,
            'count': 1,
            'region': select_best_carb_district_for_emissions(row.lat,row.lon,state_locations[row.country])['coabdis']
            }
            for row
            in get_world_boundary().clean_names()[['country','geometry_agg']]
                .set_geometry('geometry_agg')
                .to_crs(crs_ll)
               .filt(lambda x: x.country.str.match('Mexico|Canada'))
               .assign(lon=lambda x: x.geometry_agg.representative_point().x,lat=lambda x: x.geometry_agg.representative_point().y).itertuples()
            ]
            ,geometry='geometry',crs=crs_ll)
    )

    coabdis_full=gpd.GeoDataFrame(
        pd.concat([coabdis,regions_oos, regions_ooc],ignore_index=True),
        geometry='geometry',crs=crs_ll)
    
    return coabdis_full


# FIXME: types
def convert_to_od_emiss(emiss: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-segment emissions to the origin–destination flow.

    Applies OD-total masking to avoid double-counting trips and tonnage across
    route segments, then groups by ``unique_flow_cols`` and sums or takes the
    first value of each field according to a fixed aggregation spec.  Geometry
    columns are retained as-is (first value).

    Args:
        emiss: Per-segment emissions DataFrame, typically the output of
            ``compute_emissions`` or ``convert_emiss_to_full``.

    Returns:
        OD-level DataFrame with one row per unique flow, containing summed VMT,
        fuel/energy consumption, emissions, and representative geometry columns.
        ``clip_distance_u`` is renamed to ``trip_distance_u``.
    """
    aggs={
        'od_flow_index':'first',
        'o_n1':'first',
        'o_n2':'first',
        'o_facility_type':'first',
        'o_state':'first',
        'o_county':'first',
        'o_country':'first',
        'd_n1':'first',
        'd_n2':'first',
        'd_facility_type':'first',
        'd_state':'first',
        'd_county':'first',
        'd_country':'first',
        'id_port':'first',
        'seadist_u':'first',
        'clip_num':'max',
        'distance_u':'first',
        'clip_distance':'sum',
        'wt_sent':'sum',
        'trips':'sum',
        'vmt_u':'sum',
        'fuel_consumption_u':'sum',
        'energy_consumption_u':'sum',
        'emiss_pm25_u':'sum',
        'emiss_pm10_u':'sum',
        'emiss_nox_u':'sum',
        'emiss_co_u':'sum',
        'emiss_co2_u':'sum',
        'emiss_ch4_u':'sum',
        'emiss_n2o_u':'sum',
        'emiss_ghg_u':'sum',
        'geometry_orig': 'first',
        'geometry_dest': 'first',
        'geometry_port': 'first',
        'geometry_full': 'first'
    }
    # remove missing columns
    for k in aggs.keys()-list(emiss.columns):
        logger.warning(f"Removing missing key [{k}] from aggregation")
        del aggs[k]
    od_emiss=(
        emiss
        .pipe(mask_for_od_totals)
        .set_geometry('geometry_clip')
        .to_crs(crs_ll)
        .assign(
            endpoints_class=lambda x: (
                case_when(x.o_n2.str.match('STATE|COUNTY|CITY'),'o_region_'+x.o_n2,True,'o_location')
                +","+case_when(x.d_n2.str.match('STATE|COUNTY|CITY'),'d_region_'+x.o_n2,True,'d_location')
            )
        )
        .groupby(unique_flow_cols)
        .agg(aggs)
        .rename(columns={'clip_distance_u':'trip_distance_u'})
        .reset_index()
    )
    return od_emiss


# FIXME: types
def convert_emiss_to_full(emiss: pd.DataFrame) -> pd.DataFrame:
    """Expand OD emissions into separate on-road and maritime mode rows.

    Splits the input into three logical subsets:

    * **No sea route** — purely on-road flows with no port leg.
    * **Export to port** — on-road legs that terminate at a port, with full
      geometry extended through the sea route.
    * **Maritime** — synthetic rows representing the sea leg, with all on-road
      emission columns nulled out and GHG computed from sea-distance × tonnage
      × a fixed maritime intensity factor (11.97 g CO2e / tonne·km).

    Unit columns are asserted or materialised via ``create_unit_assertion`` /
    ``create_unit_assignment`` to ensure pint compatibility before concatenation.

    Args:
        emiss: OD-level emissions DataFrame produced by ``convert_to_od_emiss``,
            containing sea-route geometry and distance columns where applicable.

    Returns:
        Combined DataFrame sorted by (od_flow_index, mode, step_num, clip_num)
        with a ``mode`` column distinguishing ``'onroad'`` from ``'maritime'``
        rows, and sea-route helper columns dropped.
    """
    maritime_gCO2e_tonne_km = 11.97 * ureg('g / (metric_ton * km)')
    searoute_cols=emiss.filter(regex='sea|port').columns
    emissadj=(
        emiss.copy()
        .assign(
            o_id_od=lambda x: x.o_id,
            d_id_od=lambda x: x.d_id,
            geometry_orig_od=lambda x: x.geometry_orig,
            geometry_dest_od=lambda x: x.geometry_dest,
        )
        # FIXME: we're asserting units here, they'd better be right!
        .pipe(lambda df: df.assign(
            **create_unit_assignment({f'emiss_{p}': u for p,u in emiss_unit.items() 
                                     if f'emiss_{p}' in (set(df.columns))}))
              )
    )
    no_searoute=(
        emissadj.filt(lambda x: x.id_port.isna())
        .assign(
            mode='onroad'
        )
        .drop(columns=searoute_cols)
    )
    export_to_port=(
        emissadj.filt(lambda x: x.id_port.notna())
        .assign(
            mode='onroad',
            d_id=lambda x: x.id_port,
            geometry_full=lambda x: x.apply(lambda row: merge_lines(row.geometry_full,row.geometry_searte),axis=1),
            )
        .drop(columns=searoute_cols)
    )
    export_maritime=(
        emissadj.filt(lambda x: x.id_port.notna() & (x.step_num==1))
        .assign(
            mode='maritime',
            EMFAC_class='Container Ship', # FIXME
            o_id=lambda x: x.id_port,
            geometry_clip=lambda x: x.geometry_searte,
            geometry_full=lambda x: x.apply(lambda row: merge_lines(row.geometry_full,row.geometry_searte),axis=1),
            )
        # clear all emissions columns
        .pipe(lambda d: d.assign(
            # would just like to do c: np.nan, but this breaks the pint column
            # type so we have to be explicit to keep it
            **{c: pd.Series(
                [float('nan')] * len(d),
                dtype=d[c].dtype,
                index=d.index
                ) for c in d.columns[d.columns.str.contains('emiss_')]
            }))
        .assign(
            # maritime ghg emissions
            emiss_ghg=lambda x: (
                (x.seadist_u * x.wt_sent).pint.to('tonne * km').pint.magnitude * maritime_gCO2e_tonne_km
            )
        )
        # materialize units
        .pipe(lambda df: df.assign(**create_unit_transformation({f'emiss_{p}': u for p,u in emiss_unit.items() if f'emiss_{p}' in set(df.columns)})))
        .drop(columns=searoute_cols)
    )
    
    # make sure columns align
    for a,b in list(permutations([no_searoute,export_to_port,export_maritime],2)):
        assert(not(set(a.columns)-set(b.columns)))
        
    ret=(
        pd.concat([no_searoute,export_to_port,export_maritime],ignore_index=True)
        .sort_values(['od_flow_index','mode','step_num','clip_num'])
    )
    return ret
