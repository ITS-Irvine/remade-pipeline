#%%
### common config stuff

from IPython.display import display

from layers.ct import CTLayer
import services.emissions
from config.settings import settings
import config

from core.common import case_when, cpath, ureg
from services.emissions import emiss_unit
import pandas as pd
import matplotlib.pyplot as plt
import pint
import pint_pandas  # Load pint integration for pandas. Units, yay!
import seaborn as sns
from services.geocode import NullGeocoder, crs_ll

from core.model_types import AnnualizedLayeredGroupedFlowIdSchema

from core.units import display_with_units
from core.units import create_unit_transformation

# %%
# let's be explicit about the units we want to use in our DataFrame

# Combine static and dynamic assignments into one dictionary
_assign_dict = {
    'wt_sent': lambda x: x.tonssent.astype('pint[ton]'),
    'dist': lambda x: x.distance_m.astype('pint[m]'),
    'clip_dist': lambda x: x.clip_distance.astype('pint[m]') # Fixed typo: 'clip_distance'
}
assign_dict=create_unit_transformation(_assign_dict)

# Add the emission columns using a dictionary comprehension
assign_dict.update({
    f'emiss_{pollutant}': lambda df, p=pollutant, u=unit: df[f'emiss_{p}'].astype(f'pint[{u}]') 
    for pollutant, unit in emiss_unit.items()
})

_display_dict = {
    'wt_sent':   'tonne',
    'dist':      'km',
    'clip_dist': 'km'
}

display_dict=create_unit_transformation(_display_dict)


for qty, unit in emiss_unit.items():
    assign_dict[f'emiss_{qty}'] = lambda df, u=unit: df[f'emiss_{qty}'].astype(f'pint[{u}]')

#%%
emiss = (
    pd.read_pickle(cpath('output_pipeline','emiss_RDRS_Appliance_Connecticut_Ewaste.pkl'))
 #   .assign(**assign_dict)
)

#%%
from layers.base import mask_for_od_totals, od_flows_agg
od_emiss=(
    emiss
    .pipe(mask_for_od_totals)
    .set_geometry('geometry_clip')
    .to_crs(crs_ll)
    .assign(
        distance_mi=lambda x: x.distance_m * ureg('m').to('mi').magnitude,
        endpoints_class=lambda x: (
            case_when(x.o_n2.str.match('STATE|COUNTY|CITY'),'o_region_'+x.o_n2,True,'o_location')
            +","+case_when(x.d_n2.str.match('STATE|COUNTY|CITY'),'d_region_'+x.o_n2,True,'d_location')
        )
    )
    .groupby(od_flows_agg)
    .agg({
        # FIXME: figure out the tonssent and trips aggregations
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
        'seadist_km':'first',
        'clip_num':'max',
        'distance_mi':'first',
        'clip_distance_mi':'sum',
        'tonssent':'sum',
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
        'emiss_ghg':'sum',
        'geometry_orig': 'first',
        'geometry_dest': 'first',
        'geometry_port': 'first',
        'geometry_full': 'first',
        **{k: 'sum' for k in assign_dict}
    })
    .rename(columns={'clip_distance_mi':'trip_distance_mi'})
    .reset_index()
)

#%%
endpoints=pd.read_pickle(cpath('output_pipeline','all_entities.pkl'))


#%%
# basic plots:

import matplotlib.pyplot as plt
import contextily as cx
import geopandas as gpd
from services.geocode import crs_usa, get_ca_boundary

# 1. Ensure it's a GeoDataFrame and project to Web Mercator (EPSG:3857)
gdf = (
    od_emiss.set_geometry('geometry_full').set_crs(crs_ll).to_crs(crs_usa)
    .filt(lambda x: ~x.layer.isin(['Connecticut']))
    # only show trips in CA
    .clip(get_ca_boundary().to_crs(crs_usa))
)


# 2. Define a scaling function for line thickness
# Adjust the multiplier (e.g., 5.0) to get the visual width you want
def get_scaled_linewidth(series, multiplier=5.0):
    # Normalize values between 0.1 and 1.0, then multiply
    norm = (series - series.min()) / (series.max() - series.min())
    return (norm * multiplier) + 0.5  # 0.5 is the minimum thickness

# 2. Get unique layers
layers = gdf['layer'].unique()

# 4. Iterate and plot
for layer in layers:
    print(f"Plotting layer: {layer}")
    
    # Create a new figure for each layer
    fig, ax = plt.subplots(figsize=(10, 10))

    # Subset data for this layer
    subset = gdf[gdf['layer'] == layer]
    
    # Calculate linewidths based on wt_sent
    # Note: Access .pint.magnitude if wt_sent is a pint-enabled column
    widths = get_scaled_linewidth(subset['wt_sent'].pint.magnitude)


    # Plot routes colored by material_grouping
    subset.plot(
        column='material_grouping', 
        ax=ax, 
        legend=True, 
        linewidth=widths
    )

    # Add contextily basemap
    cx.add_basemap(ax, crs=gdf.crs.to_string())

    ax.set_title(f'Routes for Layer: {layer}')
    ax.set_axis_off()

plt.tight_layout()
plt.show()

#%%
# let's have a look at the layers

print("In country vs exported flow characteristics by layer and material grouping")
print("Note that this is all flows, not just terminal flows (i.e., flows that leave our system)")
display(
    od_emiss.assign(
        export_status=lambda x: np.where(
            ~x.d_country.str.match('United States') | x.ttype.str.match('.*port'),
            'Exported','United States')
        )
    .groupby(['layer','material_grouping','export_status'])
    .agg({'trips':'sum','wt_sent':'sum','dist':'mean'})
    .pipe(display_with_units,display_dict)
    .assign(
        wt_export_fraction=lambda x: x.wt_sent_metric_ton/x.groupby(['layer','material_grouping'])['wt_sent_metric_ton'].transform('sum')
    )
)

#%%
# let's have a look at the mass balance:
mass_bal=(
    od_emiss.groupby(['layer','o_id']).agg({'wt_sent':'sum'}).pipe(display_with_units,display_dict).rename_axis(('layer','id'))
    .join(
        od_emiss.groupby(['layer','d_id']).agg({'wt_sent':'sum'}).pipe(display_with_units,display_dict).rename_axis(('layer','id')
        )
        .rename(columns={'wt_sent_metric_ton':'wt_rcvd_metric_ton'}),
        how='left'
    )
    .fillna(0)
    .assign(net_flow=lambda x: x.wt_rcvd_metric_ton - x.wt_sent_metric_ton)
    
)

# remove outliers for visualization purposes
frac = 0.00000001
lower_bound = mass_bal['net_flow'].quantile(frac)
upper_bound = mass_bal['net_flow'].quantile(1-frac)


# 1. Filter the data (keeping the unit-stripped magnitude)
filtered_data = mass_bal.reset_index().filt(lambda x:
    (x['net_flow'] > lower_bound) & 
    (x['net_flow'] < upper_bound) &
    (x.wt_sent_metric_ton > 0 ) &
    (x.wt_rcvd_metric_ton > 0 ) &
    (x.layer.isin(['Ewaste']))
)

# 2. Extract magnitudes for plotting
filtered_data['net_flow_mag'] = filtered_data['net_flow'].pint.magnitude

# 3. Plot using seaborn
plt.figure(figsize=(12, 7))
sns.histplot(
    data=filtered_data, 
    x='net_flow_mag', 
    hue='layer', 
    element='step', # 'step' or 'bars'
    common_norm=False, # Normalizes each group independently
    bins=50
)

plt.title('Distribution of Net Flows by Layer')
plt.xlabel('Net Flow (Metric Tons)')
plt.ylabel('Frequency')
plt.show()

#%%
# facility analysis
rdrs_class=pd.read_excel(cpath('processed_data','Master-facility-classifications.xlsx'),sheet_name='Master').clean_names()

# see Brian's work: https://docs.google.com/presentation/d/1YHhIcR7NbAmRTD46rII-kBgRrRnmjnKO/edit?slide=id.p3#slide=id.p3
type_codes = {
    0: 'Small Processors',
    1: 'Very Small MSW',
    2: 'Small C&D',
    3: 'Large Landfills',
    4: 'Large Composters',
    5: 'Extremely Small Processors',
    6: 'Small Plastic Recyclers',
    7: 'C&D Transfer',
    8: 'Small Composters',
    9: 'Small Transfer w/MSW and Organics',
    10: 'Very Large Transfer Working MSW',
}

rdrs_class = (
    rdrs_class
    .assign(
        fac_cat = lambda x: x.sck11.map(type_codes)
    )
)

ng=NullGeocoder(
        select=lambda df: ~df.set_geometry('geometry').geometry.is_valid
    )
matmap_file = cpath('processed_data','MATCATS_MaterialMappings24037.xlsx')
# CT classifications
ct_layer=CTLayer(
    matmap_file=matmap_file,
    model_year=2021,
    geocoder=ng
)

ct_class=ct_layer.get_endpoints().reset_index()[['id','facility_type']]#.assign(type=lambda x: pd.Categorical(x.type))

od_emiss=(
    od_emiss
    .merge(rdrs_class[['rdrs_id','fac_cat']].add_col_prefix('o_'),left_on='o_id',right_on='o_rdrs_id',how='left')
    .merge(rdrs_class[['rdrs_id','fac_cat']].add_col_prefix('d_'),left_on='d_id',right_on='d_rdrs_id',how='left')
    .merge(ct_class.add_col_prefix('o_ct_'),left_on='o_id',right_on='o_ct_id',how='left')
    .merge(ct_class.add_col_prefix('d_ct_'),left_on='d_id',right_on='d_ct_id',how='left')
    .assign(
        o_fac_cat=lambda x: (
            x.o_fac_cat.fillna(x.o_ct_facility_type)
            .fillna(
                x.o_fac_cat.mask(
                    x.o_id.str.match('^ZIP'), 'ZIP'
                )
                .mask(
                    x.o_id.str.match('^CTY'), 'COUNTY' # FIXME: split into collectors and processors
                )
                .mask(
                    x.o_id.str.match('^STA'), 'STATE'
                )
                .mask(
                    x.o_id.str.match('^CTRY'), 'COUNTRY'
                )
                .mask(
                    x.o_id.str.match('^PORT'), 'PORT'
                )
                .mask(
                    x.o_id.str.match('^UWED'), 'UWED' # FIXME: split into collectors and processors
                )
                .mask(
                    x.o_id.str.match('^CAR'), 'Appliance Recycler'
                )
            )
        ),

        d_fac_cat=lambda x: (
            x.d_fac_cat.fillna(x.d_ct_facility_type)
            .fillna(
                x.o_fac_cat.mask(
                    x.d_id.str.match('^ZIP'), 'ZIP'
                )
                .mask(
                    x.d_id.str.match('^CTY'), 'COUNTY' # FIXME: split into collectors and processors
                )
                .mask(
                    x.d_id.str.match('^STA'), 'STATE'
                )
                .mask(
                    x.d_id.str.match('^CTRY'), 'COUNTRY'
                )
                .mask(
                    x.d_id.str.match('^PORT'), 'PORT'
                )
                .mask(
                    x.d_id.str.match('^UWED'), 'UWED' # FIXME: split into collectors and processors
                )
                .mask(
                    x.d_id.str.match('^CAR'), 'Appliance Recycler'
                )
            )
        )
    )
)

#%%

# things to do
#
# * RDRS trip types as facility class to facility class flows
# * Trip length distributions by ttype

#%%

from plotnine import ggplot, aes, geom_histogram, facet_wrap, theme, ggtitle, after_stat, scale_y_continuous
from mizani.formatters import percent_format

for (layer), df in (
    od_emiss.pipe(display_with_units,display_dict).pipe(deunitize,'dist_kilometer')
    .filt(lambda x: ~x.layer.isin(['Connecticut']))
    .groupby(['layer'])
):

    p=(
        ggplot(df, aes(x='dist_kilometer'))
        + geom_histogram(bins=len(range(0, int(df['dist_kilometer'].max()) + 10, 25)),
                        alpha=0.7)
        + aes(y=after_stat('count / sum(count)'))
        + facet_wrap('~ttype', ncol=1, scales='free_y')  # Single column layout
        + theme(
            figure_size=(12, 20),
            legend_position='none'
        )
        + ggtitle(f'Trip Length Distribution by Material Grouping for {layer} layer ')
        + scale_y_continuous(labels=percent_format()) # Optional: formats y-axis as %

    )
    p.show()

#%%
# ELV plot issues
import contextily as ctx

layer='ELV'
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
N=10
df=(
    od_emiss.pipe(display_with_units,display_dict).pipe(deunitize,'dist_kilometer')
    .filt(lambda x: x.layer.isin(['ELV']) & x.ttype.isin(['zip_to_dism']))
    .nlargest(N, 'dist_kilometer') # Increased to 10 for better visualization
    .set_geometry('geometry_orig')
    .set_crs(crs_ll).to_crs(crs_usa)
    .set_geometry('geometry_dest')
    .set_crs(crs_ll).to_crs(crs_usa)
    .set_geometry('geometry_full')
    .set_crs(crs_ll).to_crs(crs_usa)
    
)

df.plot(ax=ax, linewidth=1, alpha=0.7, color='blue')


# 2. Add Origins (green markers)
df.drop_duplicates().set_geometry('geometry_orig').plot(ax=ax, color='green', markersize=50, label='Origin')

# 3. Add Labels for Origins
for x, y, label in zip(df.geometry_orig.x, df.geometry_orig.y, df.o_n1):
    ax.text(x, y, label, fontsize=9, color='darkgreen')

# 4. Add Destinations (assuming you have a geometry_dest column)
# If you don't have geometry_dest, ensure you have the correct destination column
df.drop_duplicates().set_geometry('geometry_dest').plot(ax=ax, color='red', markersize=50, label='Destination')

# 5. Add Labels for Destinations
for x, y, label in zip(df.geometry_dest.x, df.geometry_dest.y, df.d_n1):
    ax.text(x, y, label, fontsize=9, color='darkred')
    
get_ca_boundary().to_crs(crs_usa).plot(ax=ax,facecolor='none')

# Add the background basemap
ctx.add_basemap(ax=ax,crs=crs_usa)

plt.title(f'{N} longest zip to dismantler trips')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()


#%%
# facility to facility flow analysis:
# Create facility type pairs
import matplotlib.pyplot as plt
import importlib
import plotly.graph_objects as go


for layer, subset in od_emiss.groupby('layer'):
    # 1. Prepare data
    flow_data = (
        subset
        .groupby(['o_fac_cat', 'd_fac_cat'])
        .agg({'trips': 'sum'})
        .reset_index()
        .nlargest(20, 'trips') # Increased to 10 for better visualization
    )

    # 2. Create unique node labels and mapping
    all_facilities = list(set(flow_data['o_fac_cat']).union(set(flow_data['d_fac_cat'])))
    node_map = {facility: i for i, facility in enumerate(all_facilities)}

    # 3. Prepare Plotly-specific data structures
    sources = flow_data['o_fac_cat'].map(node_map).tolist()
    targets = flow_data['d_fac_cat'].map(node_map).tolist()
    values = flow_data['trips'].tolist()

    # 4. Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_facilities,
            color="skyblue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(title_text=f"{layer}: Freight Flow Between Facility Types", font_size=12)
    fig.show()

# %%
# plot top N distinations per layer
# 1. Get unique layers
N=10

for (layer, ttype), df in (
    od_emiss
    .filt(lambda x: ~x.layer.isin(['Connecticut']))
    .groupby(['layer','ttype'])
):
    # 2. Get the top 20 destinations for the current layer
    top_destinations = (
        df
        .groupby('d_id')
        .agg({'wt_sent': 'sum'})
        .pipe(deunitize, 'wt_sent')
        .nlargest(N, 'wt_sent')
        .join(endpoints[['n1', 'facility_type', 'geometry']], how='left')
        .set_geometry('geometry')
        .to_crs(crs_usa)
    )
    # 5. Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # 6. Plot points with size proportional to wt_sent
    # Adjust 's' multiplier to fit your visual preference
    top_destinations.plot(
        ax=ax, 
        markersize=top_destinations['wt_sent'] / top_destinations['wt_sent'].max() * 500, 
        color='red', 
        alpha=0.6,
        edgecolor='white'
    )
    
    # plot routes to dest
    routes = df.filt(lambda x: x.d_id.isin(top_destinations.index)).set_geometry('geometry_full').set_crs(crs_ll).to_crs(crs_usa).plot(ax=ax)
    
    # 7. Add labels using the 'n1' column
    # We iterate through the rows to place text at the geometry coordinates
    for x, y, label in zip(top_destinations.geometry.x, top_destinations.geometry.y, top_destinations['n1']):
        ax.text(x, y, label, fontsize=9, ha='center', va='bottom', color='black')
        
    get_ca_boundary().to_crs(crs_usa).plot(ax=ax,facecolor='none')

    # 7. Add basemap
    cx.add_basemap(ax, crs=top_destinations.crs.to_string(), source=cx.providers.CartoDB.Positron)

    ax.set_title(f'Top {N} Destinations for Layer/ttype: {layer}/{ttype}')
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# %%
# facility category modal efficiency
# Compare mode efficiency by facility types
# Create modal efficiency analysis
modal_efficiency = (
    od_emiss
    .groupby(['o_fac_cat', 'd_fac_cat', 'ttype'])
    .agg({'emiss_ghg': 'sum', 'tonssent': 'sum', 'trip_distance_mi': 'sum'})
    .reset_index()
    .assign(
        emissions_per_ton_mile=lambda x: x['emiss_ghg'] / (x['tonssent'] * x['trip_distance_mi'])
    )
)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 10 facility pairs by emissions
top_pairs = (
    modal_efficiency
    .assign(emiss_ghg=lambda x: x.emiss_ghg.astype('float64')) # avoid bug https://github.com/pandas-dev/pandas/issues/42816
    .nlargest(10, 'emiss_ghg')
)
top_pairs['pair'] = top_pairs['o_fac_cat'] + ' → ' + top_pairs['d_fac_cat']

# Emissions by facility pair
bars1 = ax1.barh(
    top_pairs['pair'], 
    top_pairs['emiss_ghg'],
    color='crimson', alpha=0.7
)
ax1.set_xlabel('Total GHG Emissions')
ax1.set_title('Top 10 Facility Pairs by Emissions')
ax1.bar_label(bars1, fmt='%.0f')

# Emissions efficiency by mode
mode_efficiency = modal_efficiency.groupby('ttype')['emissions_per_ton_mile'].mean().sort_values()

bars2 = ax2.barh(
    mode_efficiency.index.astype(str), 
    mode_efficiency.values.astype(float),
    color='forestgreen', alpha=0.7
)
ax2.set_xlabel('Emissions per Ton-Mile')
ax2.set_ylabel('Transportation Mode')
ax2.set_title('Emissions Efficiency by Mode')
# ax2.bar_label(bars2, fmt='%.2f', rotation=90)

plt.tight_layout()
plt.show()

# %%
# Distance decay analysis by ORIGIN facility type
# Create distance bins and analyze by facility types
import seaborn as sns
import numpy as np

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .filt(lambda x: x.layer==layer)
        .groupby(['o_fac_cat', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='o_fac_cat', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Origin Facility Type and Distance')
    plt.ylabel('Origin Facility Type')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

#%%
# Distance decay analysis by facility type
# Create distance bins and analyze by facility types
import seaborn as sns

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .filt(lambda x: x.layer==layer)
        .groupby(['d_fac_cat', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='d_fac_cat', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Distance Facility Type and Distance')
    plt.ylabel('Distance Facility Type')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
# Distance decay analysis by DESTINATION facility type
# Create distance bins and analyze by facility types
import seaborn as sns

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .filt(lambda x: x.layer==layer)
        .groupby(['d_fac_cat', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='d_fac_cat', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Destination Facility Type and Distance')
    plt.ylabel('Destination Facility Type')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# %%
# Distance decay analysis by trip type
# Create distance bins and analyze
import seaborn as sns

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .filt(lambda x: x.layer==layer)
        .groupby(['ttype', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='ttype', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Trip Type and Distance')
    plt.ylabel('Trip Type')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# %%
# Distance decay analysis by material stream
# Create distance bins and analyze
import seaborn as sns

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .filt(lambda x: x.layer==layer)
        .groupby(['material_stream', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='material_stream', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Material Stream and Distance')
    plt.ylabel('Material Stream')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
# Distance decay analysis by material stream and trip type
# Create distance bins and analyze
import seaborn as sns

# Create distance bins
od_emiss['distance_bin'] = pd.cut(
    od_emiss['trip_distance_mi'], 
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=['0-50', '50-100', '100-200', '200-500', '500+']
)

for layer in od_emiss.layer.unique():
    
    print(layer)

    # Create pivot table for heatmap
    heatmap_data = (
        od_emiss
        .assign(group=lambda x: x.material_stream+" + "+x.ttype)
        .filt(lambda x: x.layer==layer)
        .groupby(['group', 'distance_bin'])
        .agg({'trips': 'sum'})
        .reset_index()
        .pivot(index='group', columns='distance_bin', values='trips')
        .fillna(0)
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=',.0f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Trips'}
    )
    plt.title(f'Trip Distribution ({layer}) by Material Stream + Trip Type and Distance')
    plt.ylabel('Material Stream + Trip Type')
    plt.xlabel('Trip Distance (miles)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



#%%
# 0. trip length distribution
from plotnine import ggplot, aes, geom_histogram, facet_wrap, theme, ggtitle

(
    ggplot(od_emiss, aes(x='trip_distance_mi'))
    + geom_histogram(bins=len(range(0, int(od_emiss['trip_distance_mi'].max()) + 10, 10)),
                     alpha=0.7)
    + facet_wrap('~material_grouping', ncol=1, scales='free_y')  # Single column layout
    + theme(
        figure_size=(12, 20),
        legend_position='none'
    )
    + ggtitle('Trip Length Distribution by Material Grouping')
)

# %%

# 0.1 Modal Split Analysis by Haul Distance



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create trip length bins (e.g., every 100 miles)
bins = pd.cut(od_emiss['trip_distance_mi'], 
               bins=range(0, int(od_emiss['trip_distance_mi'].max()) + 100, 100),
               right=False)

# Group by haul distance bin and transportation type
distance_modal = (
    od_emiss
    .filt(lambda x: x.layer != 'Connecticut')
    .assign(distance_bin=bins)
    .groupby(['distance_bin', 'ttype'])
    .agg({'trips': 'sum', 'vmt': 'sum'})
    .reset_index()
)

# Calculate modal split (percentage of trips by mode within each distance bin)
distance_modal['modal_share'] = (
    distance_modal.groupby('distance_bin')['trips'].transform('sum')
)
distance_modal['modal_percentage'] = (distance_modal['trips'] / distance_modal['modal_share']) * 100

# Pivot for plotting
modal_pivot = distance_modal.pivot(index='distance_bin', 
                                     columns='ttype', 
                                     values='modal_percentage')
colors20 = plt.cm.tab20(np.linspace(0, 1, min(20, len(modal_pivot.columns))))

# Create stacked bar chart
ax = modal_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.8, color=colors20)
plt.title('Modal Split by Haul Distance')
plt.xlabel('Trip Length (miles)')
plt.ylabel('Percentage of Trips')
plt.legend(title='Trip Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 0.1b Material type Analysis by Haul Distance

import pandas as pd
import matplotlib.pyplot as plt

# Create trip length bins (e.g., every 100 miles)
bins = pd.cut(od_emiss['trip_distance_mi'], 
               bins=range(0, int(od_emiss['trip_distance_mi'].max()) + 100, 100),
               right=False)

# Group by haul distance bin and transportation type
distance_modal = (
    od_emiss
    .filt(lambda x: x.layer != 'Connecticut')
    .assign(distance_bin=bins)
    .groupby(['distance_bin', 'material_grouping'])
    .agg({'trips': 'sum', 'vmt': 'sum'})
    .reset_index()
)

# Calculate modal split (percentage of trips by mode within each distance bin)
distance_modal['modal_share'] = (
    distance_modal.groupby('distance_bin')['trips'].transform('sum')
)
distance_modal['modal_percentage'] = (distance_modal['trips'] / distance_modal['modal_share']) * 100

# Pivot for plotting

modal_pivot = distance_modal.pivot(index='distance_bin', 
                                     columns='material_grouping', 
                                     values='modal_percentage')
colors20 = plt.cm.tab20(np.linspace(0, 1, min(20, len(modal_pivot.columns))))

# Create stacked bar chart
ax = modal_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.8, color=colors20)
plt.title('Modal Split by Haul Distance')
plt.xlabel('Trip Length (miles)')
plt.ylabel('Percentage of Trips')
plt.legend(title='Material Grouping', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%
# 0.2 Value-to-Weight Analysis by Trip Length

import numpy as np

# Assuming you have tonssent (weight) and can calculate value
# You'll need to add a value column based on your data
# For example, if you have a value_per_ton column:

# od_emiss['value'] = od_emiss['tonssent'] * od_emiss['value_per_ton']

# Create trip length bins
od_emiss['distance_bin'] = pd.cut(od_emiss['trip_distance_mi'], 
                                 bins=range(0, int(od_emiss['trip_distance_mi'].max()) + 100, 100),
                                 right=False)

# Group by distance bin and calculate value-to-weight ratio
value_weight_analysis = (
    od_emiss
    .groupby('distance_bin')
    .agg({
        'tonssent': 'sum',           # Total weight
        'trips': 'count',              # Number of trips
        'trip_distance_mi': 'mean',    # Average distance
        # 'value': 'sum'               # Uncomment if you have value data
    })
    .reset_index()
)

# Calculate average tons per trip
value_weight_analysis['avg_tons_per_trip'] = \
    value_weight_analysis['tonssent'] / value_weight_analysis['trips']

# Display results
print(value_weight_analysis[[
    'distance_bin', 'trips', 'tonssent', 
    'avg_tons_per_trip', 'trip_distance_mi'
]])

# Plot average tons per trip by distance
ax = value_weight_analysis.plot(
    x='distance_bin', 
    y='avg_tons_per_trip', 
    kind='bar', 
    figsize=(10, 6)
)
plt.title('Average Freight Weight per Trip by Haul Distance')
plt.xlabel('Trip Length (miles)')
plt.ylabel('Average Tons per Trip')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Calculate Average Length of Haul by material grouping
# Formula: Total Ton-Miles / Total Tonnage
# 0.3 Average Length of Haul by Commodity Type


alh_analysis = (
    od_emiss
    .groupby('material_grouping')
    .agg({
        'tonssent': 'sum',                    # Total tonnage
        'vmt': 'sum',                           # Vehicle Miles Traveled
        'trips': 'count',                       # Number of trips
        'trip_distance_mi': ['mean', 'median']  # Average trip length
    })
    .round(2)
)

# Flatten column names
alh_analysis.columns = ['_'.join(col).strip() for col in alh_analysis.columns]

# Calculate Total Ton-Miles
alh_analysis['ton_miles'] = alh_analysis['trip_distance_mi_mean'] * alh_analysis['tonssent_sum']

# Calculate Average Length of Haul
alh_analysis['average_length_of_haul'] = alh_analysis['ton_miles'] / alh_analysis['tonssent_sum']

# Sort by average length of haul
alh_analysis = alh_analysis.sort_values('average_length_of_haul', ascending=False)

# Display results
print(alh_analysis[[
    'trips_count', 'tonssent_sum', 'vmt_sum', 
    'trip_distance_mi_mean', 'average_length_of_haul'
]])

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Average Length of Haul by material grouping
alh_analysis['average_length_of_haul'].plot(
    kind='bar', ax=ax1, color='skyblue'
)
ax1.set_title('Average Length of Haul by Material Grouping')
ax1.set_xlabel('Material Grouping')
ax1.set_ylabel('Average Length of Haul (miles)')
ax1.tick_params(axis='x', rotation=90)

# Ton-Miles by material grouping
alh_analysis['vmt_sum'].plot(
    kind='bar', ax=ax2, color='lightgreen'
)
ax2.set_title('Total Vehicle Miles Traveled by Material Grouping')
ax2.set_xlabel('Material Grouping')
ax2.set_ylabel('Vehicle Miles Traveled')
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


# %%
# 1. GHG Emissions per Ton-Mile by Material Stream
# This emphasizes the cost of appliance collection due to smaller loads
# Note: This is a simple average across all shipments of each material stream, not weighted by tonnage or distance.
# We could also do a weighted average by ton-miles to better reflect the overall impact of each material stream.
(
    emiss
    # .filt(lambda x: x.layer != 'Appliance')
    #  .filt(lambda x: x.EMFAC_class.str.match('T7'))
    .assign(ghg_per_dist=lambda df: 
        (df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])).pint.to('kg / tonne / km').pint.magnitude
        )
    .groupby('material_stream')['ghg_per_dist']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .plot(kind='bar', figsize=(12, 6), color='crimson')
)
plt.title('Top 10 Material Streams by GHG Emissions (kg CO2e) per Tonne-Kilometer (2021)')
plt.ylabel('Avg. GHG per Tonne-Kilometer (kg CO2e)')
plt.xlabel('Material Stream')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

#%%

# 1b. Weighted Average GHG Emissions per Tonne-Kilometer by Material Stream
# This calculates a tonne-kilometer-weighted average, giving more influence to larger/longer shipments.
(
    emiss
    .assign(
        ghg_per_dist=lambda df: (df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])).pint.to('kg / tonne / km').pint.magnitude,
        tonne_kms=lambda df: (df['wt_sent'] * df['clip_dist']).pint.to('tonne * km').pint.magnitude
    )
    .groupby('material_stream')
    # Weighted average: (sum of ghg_per_dist * tonne_kms) / sum of tonne_kms
    .apply(lambda x: (x['ghg_per_dist'] * x['tonne_kms']).sum() / x['tonne_kms'].sum())
    .rename('Weighted GHG per Tonne-Kilometer')
    .sort_values(ascending=False)
    .head(10)
    .plot(kind='bar', figsize=(12, 6), color='darkred')
)
plt.title('Top 10 Material Streams by Weighted Avg. GHG Emissions per Tonne-Kilometer (2021)')
plt.ylabel('Weighted Avg. GHG per Tonne-Kilometer (kg CO2e)')
plt.xlabel('Material Stream')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# %%
# 2. Modal Shift: Detailed Emission Intensity by EMFAC_class and Mode

# Calculate the average GHG per Tonne-Kilometer for on-road shipments, grouped by EMFAC_class
on_road_by_class = (
    emiss
    # .filt(lambda x: x.EMFAC_class.str.match('T[1-8]'))
    .assign(seadist_km=lambda df: df['seadist_km'].fillna(0))
    .query('seadist_km == 0')
    .assign(
        ghg_per_dist=lambda df: (df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])).pint.to('kg / tonne / km').pint.magnitude
    )
    .groupby('EMFAC_class')['ghg_per_dist']
    .mean()
)

# Calculate the average GHG per Tonne-Kilometer for maritime shipments
maritime_intensity = (
    emiss
    .assign(seadist_km=lambda df: df['seadist_km'].fillna(0))
    .query('seadist_km > 0')
    .assign(
        ghg_per_dist=lambda df: (df['emiss_ghg'] / (df['wt_sent'] * df['seadist_km'].astype('pint[km]'))).pint.to('kg / tonne / km').pint.magnitude
    )
    ['ghg_per_dist']
    .mean()
)

# Combine the results into a single DataFrame for plotting
# Add the maritime mode as a single value
on_road_by_class['Maritime'] = maritime_intensity
plot_data = on_road_by_class.reset_index()
plot_data.columns = ['EMFAC_class', 'ghg_per_dist']

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_data, 
    x='EMFAC_class', 
    y='ghg_per_dist',
    hue='EMFAC_class', # This creates a separate bar for each class
    dodge=False,        # Bars are not offset (not strictly necessary for one bar per category)
    palette='viridis'   # Optional: use a color palette for better visuals
)
plt.title('Avg. GHG Emissions per Tonne-Kilometer by Vehicle Class and Mode (2021)')
plt.ylabel('kg CO2e per Tonne-Kilometer')
plt.xlabel('Vehicle Class or Mode')
# The legend will show each class; you can remove it with plt.legend().remove() if it's too cluttered
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the results
print("Average GHG per Tonne-Kilometer by Vehicle Class (On-Road):")
print(on_road_by_class.drop('Maritime'))
print(f"\nMaritime GHG per Tonne-Kilometer: {maritime_intensity:.2f} kg CO2e / tonne-km")

# discussion: 
# The expected greenhouse gas (GHG) emission rate for Heavy Heavy-Duty Trucks
# (HHDT), such as the EMFAC T7 class, is defined by U.S. Environmental
# Protection Agency (EPA) regulations. The finalized Phase 3 standards set
# specific targets in grams of CO2 per ton-mile (g CO2/ton-mi) 2.

# For model year (MY) 2032, the EPA standard for "Combination Long Haul
# Tractors" (which corresponds to T7) is 80 grams of CO2 per ton-mile 3.

# To convert this to the more common unit of kg CO2e per tonne-kilometer:

# 1. Convert grams to kilograms: 80 g/ton-mi = 0.080 kg/ton-mi
# 2. Convert ton-mile to tonne-kilometer: 1 ton-mile = 1.609 tonne-kilometers
# 3. Calculate: 0.080 kg/ton-mi ÷ 1.609 tonne-km/ton-mi ≈ 0.05 kg CO2e per tonne-kilometer

# Therefore, the regulatory standard for new, efficient T7 trucks by 2032 is
# approximately 0.05 kg CO2e per tonne-km. Your observed value of 0.06 kg CO2e
# per tonne-km is very close to this target, indicating that the T7 trucks in
# your 2021 dataset are already operating with high freight efficiency, likely
# due to consistently high load factors

# refs
# 1. NRDC Document Bank: https://www.nrdc.org/sites/default/files/air_14011702a.pdf
# 2. EPA Heavy Truck Standards to Drive Down GHG Emissions: https://www.vnf.com/epa-heavy-truck-standards-to-drive-down-ghg-emissions
# 3. Greenhouse Gas Emissions Standards for Heavy-Duty Vehicles: https://www.federalregister.gov/documents/2024/04/22/2024-06809/greenhouse-gas-emissions-standards-for-heavy-duty-vehicles-phase-3


# %%
# 3. Emissions Hotspots by Origin County
(
    emiss
    .assign(ghg_kg=lambda df: df['emiss_ghg'].pint.to('kg').pint.magnitude)
    .groupby('o_county')['ghg_kg']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .plot(kind='bar', figsize=(12, 6), title='Top 10 Origin Counties by Total GHG Emissions (2021)')
)
plt.ylabel('Total GHG Emissions (kg CO2e)')
plt.xlabel('County')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

# 4. Tonnage vs Emissions Correlation (OD-Level)

# Step 1: Aggregate clip-level emissions to the od_flow_index level
# Sum all emission columns for each unique combination of the grouping variables.
emiss_agg = (emiss
    .groupby(['od_flow_index', 'layer', 'ttype', 'material_stream', 'material_grouping', 'EMFAC_class'])
    [[col for col in emiss.columns if col.startswith('emiss_')]]
    .sum()
    .reset_index()
)

# Step 2: Merge the aggregated emissions back with the OD-level data
# First, get the OD-level data (tonnage, trips, distance)
od_data = emiss[['od_flow_index', 'layer', 'ttype', 'material_stream', 'material_grouping', 'EMFAC_class', 'tonssent', 'trips', 'distance_mi']].drop_duplicates()

# Merge the aggregated emissions with the OD-level data
merged_data = od_data.merge(emiss_agg, on=['od_flow_index', 'layer', 'ttype', 'material_stream', 'material_grouping', 'EMFAC_class'])

# Step 3: Convert to Pint units and calculate ghg_per_vmt
merged_data_pint = (merged_data
    .assign(
        ghg_per_vmt=lambda df: (
            df['emiss_ghg'].astype('pint[tonne]').pint.to('kg').pint.magnitude * 
            df['tonssent'].astype('pint[ton]').pint.to('tonne').pint.magnitude
        ) / (
            df['tonssent'].astype('pint[ton]').pint.to('tonne').pint.magnitude * 
            df['distance_mi'].astype('pint[mile]').pint.to('km').pint.magnitude
        ),
        tonnage=lambda df: df['tonssent'].astype('pint[ton]').pint.to('tonne').pint.magnitude
    )
)


# Step 4: Create a series of scatterplots with different hue variables
variables_to_plot = ['ttype', 'material_grouping', 'EMFAC_class', 'layer']

for hue_var in variables_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_data_pint,
        x='tonnage', 
        y='ghg_per_vmt', 
        hue=hue_var,  
        alpha=0.6,
        palette='tab10' # Optional: ensures consistent colors across plots
    )
    plt.title(f'OD-Level: Tonnage vs GHG Emissions per VMT by {hue_var} (2021)')
    plt.xlabel('Tonnage Shipped (tonnes)')
    plt.ylabel('GHG per VMT (kg CO2e/km)')
    plt.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
# 6. Distance vs Emission Rate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=emiss.assign(
        ghg_per_dist=lambda df: (df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])).pint.to('kg / tonne / km').pint.magnitude,
        dist_km=lambda df: df['clip_dist'].pint.to('km').pint.magnitude
    ).reset_index(),
    x='dist_km', y='ghg_per_dist', alpha=0.5
)
plt.title('Trip Length (Clip) vs GHG per Tonne-Kilometer (2021)')
plt.xlabel('Clip Distance (kilometers)')
plt.ylabel('GHG per Tonne-Kilometer (kg CO2e)')
plt.show()

# %%
# 7. Material Grouping Contribution to Total Emissions (PM2.5, GHG, NOx)

# Define the pollutants to include
include_poll = ['pm25', 'ghg', 'nox']
# include_poll = ['pm25', 'nox']

# Create the list of column names
poll_cols = [f'emiss_{pollutant}' for pollutant in include_poll]

# Group the data and convert all emissions to kilograms
emiss_grouped = (
    emiss
    .groupby('material_grouping')[poll_cols]
    .sum()
    .assign(**{
        col: lambda df, c=col: df[c].pint.to('kg').pint.magnitude for col in poll_cols
    })
    .sort_values('emiss_ghg', ascending=False)
    .head(10)
)

# Create a stacked plot (subplots) for each pollutant
axs = emiss_grouped.plot(
    kind='bar', 
    subplots=True,                    # Create a separate plot for each column
    layout=(len(poll_cols), 1),        # Arrange plots in a single column (stacked)
    figsize=(12, 4 * len(poll_cols)), # Adjust height based on the number of plots
    sharex=True,                      # Share the x-axis for alignment
    legend=False                      # Legend is not needed for individual subplots
)

# Format the individual subplots
for idx, (ax, pollutant) in enumerate(zip(axs.flat, include_poll)):
    ax.set_ylabel(f'Total {pollutant.upper()}\nEmissions (kg)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)

# Set the overall title and adjust the layout
plt.suptitle('Top 10 Material Groupings by Total Emissions (2021)', y=0.98)
plt.tight_layout()
plt.show()

# %%
# 8. Stacked Regional Emission Intensity by Material Grouping (Top 20)
# Calculate GHG per Tonne-Kilometer
emiss_enhanced = (
    emiss
    .assign(ghg_per_dist=lambda df: (df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])).pint.to('kg / tonne / km').pint.magnitude)
)

# Group by region and material_grouping, calculate mean
regional_material_intensity = (
    emiss_enhanced
    .groupby(['region', 'material_grouping'])['ghg_per_dist']
    .mean()
    .unstack(fill_value=0)  # Reshape to wide format for plotting
)

# Calculate total intensity per region for ranking
regional_totals = regional_material_intensity.sum(axis=1).sort_values(ascending=False)

# Get top 20 regions
top_20_regions = regional_totals.head(20).index
data_top_20 = regional_material_intensity.loc[top_20_regions]

# Create stacked bar chart
ax = data_top_20.plot(kind='bar', stacked=True, figsize=(14, 8), width=0.8)
plt.title('Top 20 Regions by GHG Emissions per Tonne-Kilometer, Stacked by Material Grouping (2021)')
plt.ylabel('GHG Emissions per Tonne-Kilometer (kg CO2e)')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='Material Grouping', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
# 10. Port Activity and Emissions Footprint (Stacked by Layer)

# Define the variable to stack by
stack = 'material_grouping'

# First, calculate the total tonne-kms and ghg_kg for each port and stack variable
port_layer_impact = (
    emiss
    .assign(
        tonne_kms=lambda df: (df['wt_sent'] * df['clip_dist']).pint.to('tonne * km').pint.magnitude,
        ghg_kg=lambda df: df['emiss_ghg'].pint.to('kg').pint.magnitude
    )
    .groupby(['name_port', stack])[['tonne_kms', 'ghg_kg']]
    .sum()
    .reset_index()
)

# Create a pivot table to prepare the data for a stacked bar chart
pivot_data = port_layer_impact.pivot(index='name_port', columns=stack, values='tonne_kms').fillna(0)

# Sort the ports by their total emissions for the primary sort order
port_totals = port_layer_impact.groupby('name_port')['ghg_kg'].sum().sort_values(ascending=False)
top_ports = port_totals.head(10).index

# Filter the pivot table to the top 10 ports
pivot_top = pivot_data.loc[top_ports]

# Display the underlying data for the top ports
port_emissions = port_totals.head(10)
display(pd.DataFrame({
    'Tonne-Kilometers': pivot_top.sum(axis=1), 
    'GHG Emissions (kg CO2e)': port_emissions
}))

# --- Colormap Selection Logic ---
# Determine the number of unique categories
categories = pivot_top.columns
n_categories = len(categories)

# Select a qualitative colormap based on the number of categories
def choose_qualitative_cmap(n):
    if n <= 10:
        return 'tab10'
    elif n <= 20:
        return 'tab20'
    else:
        return 'Set3' # Set3 has 12 colors, but is good for many categories

color_map = choose_qualitative_cmap(n_categories)

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

# Create the stacked bar chart for tonne-kms with the selected colormap
pivot_top.plot(kind='bar', stacked=True, ax=ax1, alpha=0.8, width=0.7, colormap=color_map)

# Overlay a line plot for total GHG emissions
ax2.plot(port_emissions.index, port_emissions.values, color='red', marker='o', linewidth=2, label='Total GHG Emissions')

# Formatting
ax1.set_title(f'Top 10 Ports: Tonne-Kilometers (Stacked by {stack}) and Total GHG Emissions (2021)')
ax1.set_ylabel('Tonne-Kilometers')
ax2.set_ylabel('GHG Emissions (kg CO2e)', color='red')
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
line, label = ax2.get_legend_handles_labels()
ax1.legend(lines + line, labels + label, loc='upper right', bbox_to_anchor=(0.85, 0.85))

plt.tight_layout()
plt.show()


# %%
# Optional: Geographic Visualization of Transport Links with Background Map (CA Routes Only)
import geopandas as gpd
import contextily as ctx

# Filter for routes where both origin and destination are in California
ca_routes = emiss[
    (emiss['o_state'] == 'California') & 
    (emiss['d_state'] == 'California')
]

# Sample a few unique routes by od_flow_index
sampled_route_ids = ca_routes['od_flow_index'].drop_duplicates().sample(10)

# Get all link steps for the sampled routes
sample_gdf = gpd.GeoDataFrame(
    ca_routes[ca_routes['od_flow_index'].isin(sampled_route_ids)], 
    geometry='geometry_clip'
)

# Ensure the GeoDataFrame is in Web Mercator for the basemap
sample_gdf_web_mercator = sample_gdf.to_crs(epsg=3857)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
sample_gdf_web_mercator.plot(ax=ax, linewidth=1, alpha=0.7, color='blue')

# Add the background basemap
ctx.add_basemap(ax)

plt.title('Sample of Full Transport Flow Routes in California (2021)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()
# %%


# 11. Average Shipment Size by Layer and Material Grouping
(
    emiss
    .assign(tonnage_tonne=lambda df: df['wt_sent'].pint.to('tonne').pint.magnitude)
    .groupby(['layer', 'material_grouping'])['tonnage_tonne']
    .mean()
    .unstack()
    .plot(kind='bar', figsize=(14, 8), title='Average Shipment Size (Tonne) by Layer and Material Grouping (2021)')
)
plt.ylabel('Average Tonnage (tonnes)')
plt.xlabel('Layer')
plt.xticks(rotation=45)
plt.legend(title='Material Grouping', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# 12. Modal Split of Tonnage by Region
(
    emiss
    .assign(tonnage_tonne=lambda df: df['wt_sent'].pint.to('tonne').pint.magnitude)
    .groupby(['region', 'ttype'])['tonnage_tonne']
    .sum()
    .unstack(fill_value=0)
    .plot(kind='bar', stacked=True, figsize=(12, 7), title='Total Tonnage by Region, Stacked by Transport Mode (2021)')
)
plt.ylabel('Total Tonnage (tonnes)')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.show()

# %%
# 13. Distribution of Trip Distances by EMFAC_class
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=od_emiss.assign(dist_km=lambda df: df['trip_distance_mi']*ureg('mi').to('km').magnitude).reset_index(),
    x='EMFAC_class',
    y='dist_km'
)
plt.title('Distribution of Trip Distances by Vehicle Class (2021)')
plt.xlabel('EMFAC Vehicle Class')
plt.ylabel('Trip Distance (kilometers)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# 14. Tonnage Origin-Export by County
export_tonnage = (
    emiss
    .assign(tonnage_tonne=lambda df: df['wt_sent'].pint.to('tonne').pint.magnitude)
    .groupby('o_county')['tonnage_tonne']
    .sum()
    .sort_values(ascending=False)
)

export_tonnage.head(15).plot(kind='bar', figsize=(12, 6), title='Top 15 Counties by Total Exported Tonnage (2021)')
plt.ylabel('Total Exported Tonnage (tonnes)')
plt.xlabel('County')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# 15. Fuel Consumption per VMT by Layer and Speed Bin
(
    emiss
    .assign(fuel_per_vmt=lambda df: (df['fuel_consumption'].astype('pint[gallon]').pint.to('liter').pint.magnitude / 
                               (df['vmt'].astype('pint[mile]').pint.to('km').pint.magnitude)))
    .groupby(['layer', 'speed_bin'])['fuel_per_vmt']
    .mean()
    .unstack()
    .plot(kind='line', marker='o', figsize=(12, 7), title='Average Fuel Consumption per VMT by Layer and Speed (2021)')
)
plt.ylabel('Fuel Consumption (liters per km)')
plt.xlabel('Speed Bin (mph)')
plt.legend(title='Layer', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# 16. Proportion of Full Truckload (FTL) vs. Less-Than-Truckload (LTL) by Material Stream
# Define FTL as shipments over 18 short tons
truck_data = emiss[emiss['EMFAC_class'].str.match('T7')].assign(
    shipment_type=lambda df: df['tonssent'].apply(lambda x: 'FTL' if x > 18 else 'LTL')
).reset_index()

# Create a crosstab and plot
(
    pd.crosstab(truck_data['material_stream'], truck_data['shipment_type'], normalize='index')
    .plot(kind='bar', stacked=True, figsize=(14, 7), title='Proportion of FTL vs. LTL Shipments by Material Stream (2021)')
)
plt.ylabel('Proportion of Shipments')
plt.xlabel('Material Stream')
plt.xticks(rotation=60)
plt.legend(title='Shipment Type')
plt.tight_layout()
plt.show()


# %%
# 17. Tonnage Throughput by Port and Material Grouping
(
    emiss
    .filt(lambda x: x.layer != 'Connecticut')
    .assign(tonnage_tonne=lambda df: df['wt_sent'].pint.to('tonne').pint.magnitude)
    .groupby(['name_port', 'material_grouping'])['tonnage_tonne']
    .sum()
    .unstack(fill_value=0)
    # .sort_values('tonnage_tonne', ascending=False)
    .head(15)
    .plot(kind='bar', stacked=True, figsize=(14, 8), title='Top 15 Ports by Tonnage Throughput (CA only), Stacked by Material Grouping (2021)')
)
plt.ylabel('Total Tonnage (tonnes)')
plt.xlabel('Port')
plt.xticks(rotation=60)
plt.legend(title='Material Grouping', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
# 17b. Tonnage Throughput by Port and Layer Grouping
(
    emiss
    .filt(lambda x: x.layer != 'Connecticut')
    .assign(tonnage_tonne=lambda df: df['wt_sent'].pint.to('tonne').pint.magnitude)
    .groupby(['name_port', 'layer'])['tonnage_tonne']
    .sum()
    .unstack(fill_value=0)
    # .sort_values('tonnage_tonne', ascending=False)
    .head(15)
    .plot(kind='bar', stacked=True, figsize=(14, 8), title='Top 15 Ports by Tonnage Throughput (CA only), Stacked by Layer (2021)')
)
plt.ylabel('Total Tonnage (tonnes)')
plt.xlabel('Port')
plt.xticks(rotation=60)
plt.legend(title='Layer', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
# 18. Regional Trip Efficiency: Tonnage-Miles per VMT by EMFAC_class
(
    emiss
    .assign(
        ton_miles=lambda df: (df['wt_sent'].pint.to('tonne').pint.magnitude * 
                          df['dist'].pint.to('km').pint.magnitude),
        vmt=lambda df: df['vmt'].astype('pint[mile]'),
        trip_efficiency=lambda df: df['ton_miles'] / df['vmt'].pint.to('km').pint.magnitude
    )
    .groupby(['material_grouping', 'EMFAC_class'])['trip_efficiency']
    .mean()
    .unstack()
    .plot(kind='bar', figsize=(14, 8), title='Average Trip Efficiency (Tonne-Km per VMT) by Region and Vehicle Class (2021)')
)
plt.ylabel('Tonne-Kilometers of Freight per km of Vehicle Travel')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='EMFAC Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%


# 19. Calculate average GHG per Tonne-Kilometer by layer and transport type
ghg_by_layer_ttype = (
    emiss
    # Filter for on-road data only (maritime uses seadist_km)
    .assign(seadist_km=lambda df: df['seadist_km'].fillna(0))
    .query('seadist_km == 0')
    # Calculate GHG per unit of transport work
    .assign(
        ghg_per_dist=lambda df: (
            df['emiss_ghg'] / (df['wt_sent'] * df['clip_dist'])
        ).pint.to('kg / tonne / km').pint.magnitude
    )
    # Group by the two categorical variables and calculate the mean
    .groupby(['layer', 'ttype'])['ghg_per_dist']
    .mean()
    # Convert to a DataFrame for better table presentation
    .reset_index()
    # Rename the column for clarity
    .rename(columns={'ghg_per_dist': 'Avg. GHG per Tonne-Km (kg CO2e)'})
)

# Display the result as a table
ghg_by_layer_ttype

#%%
# Load calenviroscreen data
import geopandas as gpd
import numpy as np
ces = gpd.read_file('zip://' +cpath('calenviroscreen_data', 'calenviroscreen40shpf2021shp.zip')).clean_names()

# clean it up

# 2. Clean CalEnviroScreen Missing Values

# List of CalEnviroScreen variables that may contain -999 for missing data
ces_vars_with_missing = [
    'ciscore', 'ciscorep', 'ozone', 'pm2_5', 'dieselpm', 'traffic', 
    'asthma', 'poverty', 'unempl', 'drinkwat', 'lead', 
    'cleanup', 'gwthreat', 'hazwaste', 'impwatbod', 'solwaste'
]

# Replace -999 with NaN for proper missing data handling
ces = (
    ces
    .replace({var : -999.0 for var in ces_vars_with_missing}, np.nan)
)

# Verify replacement worked
for var in ces_vars_with_missing:
    if var in ces.columns:
        missing_count = ces[var].isna().sum()
        print(f"ces missing values after cleaning: {missing_count}")


# 3. Handle Remaining Missing Values

# Summary of missing values after cleaning
missing_summary = (
    ces
    .isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame('missing_count')
    .query('missing_count > 0')
)

print("Missing values by column after initial cleaning:")
print(missing_summary)


#%%

# 1. Total Tonnage and VMT of Freight Flows Through DACs vs. Non-DACs

# This analysis quantifies the total freight burden on DACs by comparing the sum
# of tonnage and VMT for all transport links that pass through DAC tracts versus
# non-DAC tracts.
# Calculate the 75th percentile threshold for ciscore
dac_threshold = ces['ciscore'].quantile(0.75)

# Create a new GeoDataFrame for DACs only, without modifying `ces`
dac_tracts = ces[ces['ciscore'] >= dac_threshold].copy()
dac_tracts['is_dac'] = True

# Convert emiss to a GeoDataFrame if it isn't already
emiss_gdf = gpd.GeoDataFrame(emiss, geometry='geometry_clip')
emiss_gdf = emiss_gdf.to_crs(dac_tracts.crs)

# Perform a spatial join
dac_joined = gpd.sjoin(emiss_gdf, dac_tracts[['geometry', 'is_dac']], how='inner', predicate='intersects')

# Aggregate by od_flow_index to avoid double-counting
summary = dac_joined.groupby('od_flow_index').agg({
    'tonssent': 'first',
    'vmt': 'sum',
    'is_dac': 'any'
}).reset_index()

# Sum the total for flows that pass through DACs vs. non-DACs (all other flows)
# We assume any flow not in `summary` did not pass through a DAC
total_tonnage = emiss_gdf['tonssent'].sum()
total_vmt = emiss_gdf['vmt'].sum()

ton_from_dac = summary['tonssent'].sum() if not summary.empty else 0
ton_not_from_dac = total_tonnage - ton_from_dac

vmt_in_dac = summary['vmt'].sum() if not summary.empty else 0
vmt_not_in_dac = total_vmt - vmt_in_dac

# Create a result DataFrame
result = pd.DataFrame({
    'Total Tonnage (tons)': [ton_from_dac, ton_not_from_dac],
    'Total VMT (miles)': [vmt_in_dac, vmt_not_in_dac]
}, index=['Through DACs', 'Not Through DACs'])

result

# %%

# 2. Proportion of Total Freight VMT Occurring Within DAC Tracts

# This analysis calculates what percentage of the state's total freight VMT
# occurs within the boundaries of DACs.
# Calculate the 75th percentile threshold for ciscore
dac_threshold = ces['ciscore'].quantile(0.75)

# Create a new GeoDataFrame for DACs only
dac_tracts = ces[ces['ciscore'] >= dac_threshold].copy()
dac_tracts['is_dac'] = True

# Convert emiss to a GeoDataFrame if it isn't already
emiss_gdf = gpd.GeoDataFrame(emiss, geometry='geometry_clip')
emiss_gdf = emiss_gdf.to_crs(dac_tracts.crs)

# Perform a spatial join to find links that intersect DAC tracts
joined = gpd.sjoin(emiss_gdf, dac_tracts[['geometry']], how='inner', predicate='intersects')

# Calculate the total VMT for all freight
# Calculate the VMT that occurs within DAC tracts
pct_vmt_in_dacs = (joined['vmt'].sum() / emiss_gdf['vmt'].sum()) * 100

print(f"Percentage of total VMT in DACs: {pct_vmt_in_dacs:.1f}%")
# %%

# 3. Average CalEnviroScreen Score of Tracts Impacted by Freight from Each Port

# This analysis identifies the origin ports and evaluates the average
# environmental burden of the communities through which their freight travels.

# Convert emiss to a GeoDataFrame if it isn't already
emiss_gdf = gpd.GeoDataFrame(emiss, geometry='geometry_clip')
emiss_gdf = emiss_gdf.to_crs(ces.crs)

# Perform the spatial join
joined_gdf = gpd.sjoin(emiss_gdf, ces[['geometry', 'ciscore']], how='inner', predicate='intersects')

# Group by the origin port and calculate the average ciscore of all intersected tracts
port_impact = joined_gdf.groupby('name_port')['ciscore'].mean().sort_values(ascending=False)

# Display the top 10 ports
port_impact.head(10)

#%%
# 4. Correlation Between Tract Pollution Burden and Freight VMT Through the Tract

# This analysis explores whether census tracts with higher pollution burdens
# (polburdsc) also experience higher levels of freight traffic.
# Calculate the 75th percentile threshold for ciscore
dac_threshold = ces['ciscore'].quantile(0.75)

# Create a copy of ces and add the is_dac flag
ces_with_dac = ces.copy()
ces_with_dac['is_dac'] = ces_with_dac['ciscore'] >= dac_threshold

# Convert emiss to a GeoDataFrame if it isn't already
emiss_gdf = gpd.GeoDataFrame(emiss, geometry='geometry_clip')
emiss_gdf = emiss_gdf.to_crs(ces_with_dac.crs)

# Perform the spatial join
joined_gdf = gpd.sjoin(emiss_gdf, ces_with_dac[['geometry', 'polburdsc']], how='inner', predicate='intersects')

# Group by the census tract (using its index) and sum the VMT for all freight that passes through it
tract_vmt = joined_gdf.groupby(joined_gdf.index_right)['vmt'].sum().rename('total_vmt')

# Join the VMT data back to the ces DataFrame
ces_with_vmt = ces_with_dac.join(tract_vmt, how='left').fillna(0)

# Calculate the correlation
correlation = ces_with_vmt['polburdsc'].corr(ces_with_vmt['total_vmt'])
print(f"Correlation between Pollution Burden Score and Freight VMT: {correlation:.3f}")

# Create a scatter plot
ces_with_vmt.plot.scatter(x='polburdsc', y='total_vmt', alpha=0.6, figsize=(10, 6))
plt.title('Pollution Burden Score vs. Freight VMT by Census Tract (2021)')
plt.xlabel('CalEnviroScreen Pollution Burden Score')
plt.ylabel('Total Freight VMT')
plt.show()


# %%
# 5. Map of Freight Flows Colored by the Average DAC Status of Intersected Tracts

# This visualization creates a map where freight flow links are colored based on
# the environmental justice characteristics of the communities they traverse.
# Calculate the 75th percentile threshold for ciscore

import geopandas as gpd
import contextily as ctx

# Calculate the 75th percentile threshold for ciscore
dac_threshold = ces['ciscore'].quantile(0.75)

# Create a new GeoDataFrame for DACs only
# Ensure both are in a projected CRS (e.g., EPSG:3857 or a state plane) for accurate length calculations
emiss_gdf = gpd.GeoDataFrame(emiss, geometry='geometry_clip').to_crs('EPSG:3857')
dac_tracts = ces[ces['ciscore'] >= dac_threshold].copy().to_crs('EPSG:3857')

# Step 1: Use sjoin to find all intersections between links and DAC tracts
# This creates one row for every link-tract intersection
joined = gpd.sjoin(emiss_gdf[['od_flow_index', 'geometry_clip']], 
                   dac_tracts[['geometry']].assign(geometry_dac=lambda x: x.geometry), 
                   how='inner', 
                   predicate='intersects')

#%%

# Step 2: For each intersection, calculate the actual length of the line within the polygon
# The intersection of a line and a polygon is a (Multi)LineString
# The right geometry is now named 'geometry_right' after the sjoin
joined['intersection_geom'] = joined.apply(lambda row: row['geometry_clip'].intersection(row['geometry_dac']), axis=1)
joined['intersection_length'] = joined['intersection_geom'].length

#%%

# Step 3: Group by the od_flow_index and sum the intersection lengths
# This gives the total length of each link that is within *any* DAC tract
total_dac_length = joined.groupby('od_flow_index')['intersection_length'].sum()

def calculate_dac_exposure(row, dac_length_series):
    """
    Calculate the proportion of a link's length that is within DACs.
    """
    total_length = row['geometry_clip'].length
    dac_length = dac_length_series.get(row['od_flow_index'], 0) # Get from the series, default 0
    return dac_length / total_length if total_length > 0 else 0.0

# Apply the function to create the dac_exposure column
dac_exposure_series = emiss_gdf.apply(lambda row: calculate_dac_exposure(row, total_dac_length), axis=1)
dac_exposure_series.name = 'dac_exposure'
emiss_with_exposure = emiss_gdf.join(dac_exposure_series)

#%%

# Step 4: Create the map (convert back to EPSG:4326 for the basemap)
emiss_with_exposure_web = emiss_with_exposure.to_crs('EPSG:4326')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
emiss_with_exposure_web.plot(
    column='dac_exposure', 
    ax=ax, 
    linewidth=2, 
    alpha=0.7, 
    legend=True, 
    legend_kwds={'label': "Proportion of Route in DACs", 'orientation': "horizontal"},
    cmap='coolwarm_r'
)
ctx.add_basemap(ax, crs=emiss_with_exposure_web.crs, source=ctx.providers.OpenStreetMap.Mapnik,ax=ax)
plt.title('Freight Flow Links Colored by Proportion in Disadvantaged Communities')
plt.show()



# %%
