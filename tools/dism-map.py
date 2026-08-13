#%%
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import polyline
import shapely
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import janitor

from plotnine import (
    ggplot,
    aes,
    arrow,
    geom_map,
    geom_text,
    geom_label,
    ggtitle,
    labs,
    facet_wrap,
    facet_grid,
    scale_fill_brewer,
    scale_x_continuous,
    scale_y_continuous,
    scale_size_continuous,
    coord_cartesian,
    element_rect,
    theme_void,
    theme,
    element_text,
    element_line,
    element_rect,
    theme_set,
    theme_void
)



from services.geocode import crs_ca, crs_ll
from core.common import (filt,add_col_suffix,display,cpath)

import fiona
from shapely.geometry import shape

# Function to safely read a file and handle invalid geometries from copilot
def safe_read_file(path):
    features = []
    crs = None
    with fiona.open(path) as src:
        crs = src.crs
        for feat in src:
            try:
                geom = shape(feat['geometry'])
                features.append({'geometry': geom, **feat['properties']})
            except Exception as e:
                print(f"Skipping invalid geometry: {e}")
    gdf = gpd.GeoDataFrame(features, crs=crs)
    return gdf


from pyogrio import set_gdal_config_options

def first_row_as_column_names(df):
    """
    Convert the first row of a DataFrame to column names.
    """
    df.columns = df.iloc[0]
    return df[1:].reset_index(drop=True)
pd.DataFrame.first_row_as_column_names = first_row_as_column_names
#%%
set_gdal_config_options({
    'SHAPE_RESTORE_SHX': 'NO',
})
#%%
# #par=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/Parcels_20250617/geo_export_351297f0-b7f5-4cff-a3e6-2d4a333eb57d.shx')
# par=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/Yolo_County_Tax_Parcels_Open_Data_')
# # %%
# lu=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/SC_LU.shp')
# # %%
# lu=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/Parcels_DelNorte/Parcels.shp')

# # %%
# lu=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/Parcels_2511541052090080241/69998d88-904d-4a7d-999e-845b7712db6d.gdb')

# # %%
# lu=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/monterey-parcels.kml')
# # %%
# lu=gpd.read_file('/vsizip/data/GIS/CA_Dismantler_SA/Dismantler/fresno-REGIONAL_PARCELS_VW_-4410040523645640654.kmz/doc.kml')

# # %%
# lu=gpd.read_file('data/GIS/CA_Dismantler_SA/Dismantler/doc2.kml')
# # parse html table in Description column to vals column
# lu2=lu.assign(vals=lambda x: x.apply(lambda row: pd.read_html(row.Description), axis=1))

# # convert vals column into single 1 row dataframe with parcel data as columns in vals2
# lu3=lu2.assign(vals2=lambda x: x.apply(lambda row: row.vals[0].transpose().reset_index(drop=True).first_row_as_column_names(),axis=1))

# # drop unnecessary cols
# lu4=lu3.drop(columns=['vals','Description'])

# # copilot-suggested approach to exploding dataframe in vals2 column into columsn in the lu dataframe
# lu5 = lu4.reset_index().rename(columns={'index': 'orig_idx'})
# exploded = pd.concat(
#     [df.assign(orig_idx=idx) for idx, df in zip(lu5['orig_idx'], lu5['vals2'])],
#     ignore_index=True
# )
# result = exploded.merge(
#     lu5.drop(columns=['vals2']),
#     on='orig_idx',
#     how='left'
# ).drop(columns=['orig_idx'])

# result2 = gpd.GeoDataFrame(result,geometry='geometry')

# result2.to_file('fresno-parcels')
# # %%
# lu=gpd.read_file('Dismantler/Parcels_-8243766860352516778_Placer/a7da1f31-dc6a-46ff-af16-17db1e3ddbc1.gdb')
# # %%
# lu=gpd.read_file('Dismantler/Parcels_Public_Shapefile.gdb/_ags_dataA79C5F4667AA46768815C48F82BA70A0.gdb')

# # %%
# lu=gpd.read_file('Dismantler/Assessor_Parcels_Land_2024_-6444669402860458995/572de876-b4fc-493b-b01e-4ef38b437055.gdb')
# # %%
# lu=gpd.read_file('Dismantler/Parcels 2024_202408220754079209/Parcels 2024.shp')

# # %%
# lu=safe_read_file('Dismantler/Parcels_-8514619513140089541/c01c1240-8267-4a95-a484-1941d0af97c3.gdb')

# # %%
# lu=gpd.read_file('Dismantler/Parcels_20250617/geo_export_351297f0-b7f5-4cff-a3e6-2d4a333eb57d.shp')

# # %%
# lu=safe_read_file('Dismantler/Parcels_2511541052090080241/69998d88-904d-4a7d-999e-845b7712db6d.gdb')

# # %%
# lu=safe_read_file('Dismantler/Parcels_20250617/geo_export_351297f0-b7f5-4cff-a3e6-2d4a333eb57d.shp')

# # %%
# lu=safe_read_file('Dismantler/Parcels_SJ/Parcels.shp')

# #%%
# # reading extended KML for monterey county
# # from fastkml import kml
# # import re

# # with open("Dismantler/monterey-parcels.kml", 'rt', encoding='utf-8') as f:
# #     doc = f.read()

# # # remove XML declaration if present as it leads to ValueError: Unicode strings with encoding declaration are not supported. Please use bytes input or XML fragments without declaration.
# # doc = re.sub(r'<\?xml.*?\?>', '', doc).strip()

# # k = kml.KML()
# # k.from_string(doc)

# # # Traverse the KML structure
# # features = list(k.features())
# # placemarks = []
# # for feature in features:
# #     for placemark in feature.features():
# #         placemarks.append(placemark)

# # for pm in placemarks:
# #     print(pm.name)
# #     print(pm.extended_data)  #
# #%%
# # gpd.io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# # fp="Dismantler/monterey-parcels.kml"

# # gdf_list = []
# # for layer in fiona.listlayers(fp):    
# #     gdf = gpd.read_file(fp, driver='LIBKML', layer=layer)
# #     gdf_list.append(gdf)

# # gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# #%%
# gdf=gpd.read_file(cpath('elv_data_dismantler','Dismantler/monterey-parcels.kml'), driver='LIBKML')
# #%%
# gdf.to_file('monterey-parcels.shp', driver='ESRI Shapefile')
# #################################################################
# #%%
holdem={}

# # %%
cldf=(
    pd.read_excel(cpath('elv_data_dismantler','CountyList_Dismantler_ParcelDataStatus.xlsx'),sheet_name='Sheet1').clean_names()
    .assign(
        county=lambda x: x.counties_.str.replace(' County','').str.strip(),
        use_file=lambda x: x.file.combine_first(x.file2),
        safe=lambda x: x.safe.fillna('')
        )
    .filt(lambda x: ~x.use_file.isna())
)

# %%

import pickle
import os

# Load if exists
if os.path.exists('dism_res.pkl'):
    with open('dism_res.pkl', 'rb') as f:
        res = pickle.load(f)
    print("Successfully loaded dataframes")
else:
    print("dism_res.pkl does not exist")
    res={}

# %%
dlo=(
    gpd.GeoDataFrame(
        pd.read_excel(cpath('elv_data','Dismantler list-crr.xlsx'),sheet_name='Sheet 1').clean_names()
        .assign(geometry=lambda x: gpd.points_from_xy(x.lon,x.lat))
        ,geometry='geometry',crs=crs_ll
    )
    # .filt(lambda x: ~x.license_number.isna())
    .filt(lambda x: ~x.street.isna())
    .assign(
        use_county=lambda x: x.branch_county.combine_first(x.main_county).str.strip()
    )
)
#%%
dlo2=(
    dlo.reset_index(names='myindex')
    # .filt(lambda x: x.parcels_est>1)
    .assign(pp=lambda x: (
        x.apply(lambda row: (
            [row.geometry] +
            [
                shapely.geometry.Point(o[1],o[0])
                # o
                for o in [
                    x.split(',') for x in [
                        z for z in [row.alt_pt,row.alt_pt2,row.alt_pt3,row.alt_pt4,row.alt_pt5,row.alt_pt6,row.alt_pt7,row.alt_pt8,
                                    row.alt_pt9,row.alt_pt10,row.alt_pt11,row.alt_pt12,row.alt_pt13,row.alt_pt14,row.alt_pt15,
                                    row.alt_pt16,row.alt_pt17,row.alt_pt18,row.alt_pt19 ] if not pd.isna(z)
                    ]
                ]
            ])
        ,axis=1)
        )
        )
    .explode('pp')
    .rename(columns={'geometry': 'main_geometry'})
    .rename(columns={'pp': 'geometry'})
    .set_geometry('geometry')
    .set_crs(crs_ll)
)

# %%
# FIXME: replace with pygris call for counties
gdf_cty=gpd.read_file('/vsizip/' + cpath('gis_data','ca-county-boundaries.zip/CA_Counties/CA_Counties_TIGER2016.shp'))
dl=gpd.sjoin(dlo2.to_crs(crs_ca),gdf_cty.to_crs(crs_ca)[['NAME','geometry']],how='left',predicate='intersects').rename(columns={'index_right': 'cty_index_right'})
assert(len(dl.filt(lambda x: x.cty_index_right.isna())) == 0), 'Some dismantler locations do not have a county assigned. Please check the data.'
#%%
import re
done=False
ctycol='NAME'

### DO OVERRIDES FIRST:
extra=gpd.read_file('/vsizip/' + cpath('gis_data','HandParcels.kmz/doc.kml')).assign(NAME='extra',file='HandParcels.kmz')
res['extra']=(
    gpd.sjoin(dl.to_crs(crs_ca),extra.to_crs(crs_ca)
            .assign(
                geom_parcel=lambda x: x.geometry,
                use_file='HandParcels.kmz',
                areacmp=lambda x: x.geometry.area
            )
            .rename(columns={ctycol:f'{ctycol}_right'})
            ,how='right',predicate='intersects')
            .rename(columns={'index_left':'index_match'})  # for use later to identify good matches
)

# remove any dismantler locations that are in the extra set
dlx=dl.filt(lambda x: ~x.myindex.isin(res['extra'].myindex.unique())).reset_index(drop=True)

todo=['Shasta'] #None
# todo=None
for cty,df in dlx.groupby(ctycol): #dl.filt(lambda x: x.use_county.isin(['Monterey'])).groupby('use_county'):
    print(cty)
    if (todo is None) or (cty not in res) or (cty in todo):
        try:
            print(f'Processing {cty}')
            use_file=cldf.set_index('county').loc[cty].use_file
            use_safe=cldf.set_index('county').loc[cty].safe=='safe'
            # print(use_file)
            # print(use_safe)
            uf=re.sub(r'^(/vsizip/)?',r'\1data/Dismantler/',use_file)
            if cty not in holdem:
                if use_safe:
                    gdf=safe_read_file(uf)
                else:
                    gdf=gpd.read_file(uf)

                holdem[cty]=gdf
            gdf=holdem[cty].assign(file=uf)

            res[cty]=(gpd.sjoin(df.to_crs(crs_ca),gdf.to_crs(crs_ca).assign(
                geom_parcel=lambda x: x.geometry,
                use_file=use_file,
                areacmp=lambda x: x.geometry.area
                )
                .rename(columns={
                    ctycol:f'{ctycol}_right', # this is to avoid collisions with the ctycol
                    })
                ,how='left',predicate='intersects')
                .rename(columns={'index_right':'index_match'})  # for use later to identify good matches
                )
            done=True

        except KeyError:
            use_file=None


# %%
all=(
    pd.concat(res.values())
    .groupby('index_match').first()  # we don't want to match parcels to multiple dismantlers, so we take the first match assignment
    .reset_index()
    .set_geometry('geometry').set_crs(crs_ca)
    .set_geometry('geom_parcel').set_crs(crs_ca).dissolve(by='myindex',
    aggfunc={
        'license_number':'first',
        'business_category':'first',
        'category':'first',
        'name':'first',
        'street':'first',
        'city':'first',
        'zip':'first',
        'NAME':'first',
        'geocode':'first',
        'exception':'first',
        'note':'first',
        'parcels_est':'first',
        'main_geometry':'first',
        'use_file':'first',
        'areacmp':'sum',
        'index_match':lambda x: [y for y in x.to_list() if not pd.isna(y)],
        'geometry':lambda x: x.to_list()
    })
    .rename(columns={'geometry': 'geom_pt_list'})
    .rename(columns={'geom_parcel': 'geometry'})
    .set_geometry('geometry')
    .to_crs(crs_ll)

    # remove objectionable rows
    .filt(lambda x: x.exception.isna())
)
#%%
stats=(
    dl.filt(lambda x: x.exception.isna())
    .groupby([ctycol,'myindex']).first().reset_index()
    .groupby(ctycol).count().reset_index()
    [[ctycol,'myindex']]
    .rename(columns={'myindex':'lindex'})
    .join(all.reset_index().groupby('NAME')
          .agg({'myindex':'nunique','license_number':'nunique','areacmp':'count'})
          [['myindex','license_number','areacmp']],on=ctycol,rsuffix='_all')
    .assign(diff=lambda x: x.lindex - x.areacmp.fillna(0))
)
display(stats)
# %%
# data=all.filt(lambda x: ~x.parcels_est.isna())# & x.name.fillna('').str.match('SRT AUTO'))
data=all.filt(lambda x: x.name.fillna('').str.match('SA RECYCLING'))
# data=all.filt(lambda x: (x.parcels_est!=1) & ~x.street.isna() & x.exception.isna())
# data=all.filt(lambda x: ~x.street.isna() & x.exception.isna() & (x.parcels_est.isna() + ((x.parcels_est!=1) & x.note.fillna('').str.match(r'.*check')) ))


ctyx='Orange'   #######################################################################
for cty,df in data.groupby(ctycol):
    if cty==ctyx:
        good=df.filt(lambda x: x.apply(lambda row: len(row.index_match)>0,axis=1))
        bad=df.filt(lambda x: x.apply(lambda row: len(row.index_match)==0,axis=1))
        ax=(ggplot())
        if cty in holdem:
            ax=ax+geom_map(holdem[cty].to_crs(crs_ll))
        if ( len(good) > 0 ):
            ax=ax+geom_map(good.to_crs(crs_ll),color='green')
        if ( len(bad) > 0 ):
            ax=ax+geom_map(bad.to_crs(crs_ll),color='red')
        plt.show(ax)

# # %%
# axs=[]
# src=bad
# color='red'
# use_crs=crs_ll
# buff=1000
# for i in range(len(src)): #range(5):#
#     pt=None
#     if(i==0):
#         pt=src.to_crs(crs_ca).head(1)
#     else:
#         pt=src.to_crs(crs_ca).tail(-i).head(1)
#     # pt=pt.to_crs(crs_ll).assign(addr=lambda x: x.branch_street.combine_first(x.main_street))
#     pt=pt.to_crs(crs_ll).assign(addr=lambda x: x.street)
#     axs.append(
#                {'pt': pt,
#          'p':   ggplot()
#                 +geom_map(holdem[ctyx].to_crs(crs_ca).filt(lambda x: x.intersects(src.to_crs(crs_ca).iloc[i].geometry.buffer(buff))).to_crs(use_crs))
#                 +geom_map(pt,color=color)
#                 # +ggtitle(f'{pt.iloc[0].business_name}: {pt.iloc[0].addr}')
#                 +ggtitle(f'{pt.iloc[0].name}: {pt.iloc[0].addr}')
#         })

# # %%
# for d in axs:
#     p=d['p']
#     mpl.rcParams['figure.dpi'] = 1000
#     fig = p.draw()
#     ax = fig.get_axes()[0]

#     ctx.add_basemap(ax, crs=use_crs,
#                     source='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoiY3JpbmR0IiwiYSI6ImNpdDZhdDZ2eTAxbzMyb3BndXFzZ3R5aHkifQ.CEdMDNgiCc-KyouaXUYWew',
#                     zoom_adjust=1,
#                     attribution_size=4)
#     display(fig)
#     print(d['pt'].iloc[0].geocode)

# %%
axs2=[]
src=good
color='green'
use_crs=crs_ll
buff=500
for i in range(len(src)):
    pt=None
    if(i==0):
        pt=src.to_crs(crs_ca).head(1)
    else:
        pt=src.to_crs(crs_ca).tail(-i).head(1)
    print(i+1,'of',len(src))
    display(pt)
    pt=pt.assign(addr=lambda x: x.street).set_geometry('main_geometry').set_crs(crs_ll).set_geometry('geometry').set_crs(crs_ca).to_crs(use_crs)
    p=ggplot()
    if ctyx in holdem:
        parcels=(
            holdem[ctyx].to_crs(crs_ca)
                    .filt(lambda x: x.intersects(src.switch_geometry('main_geometry').set_crs(crs_ll).to_crs(crs_ca).iloc[i].geometry.buffer(buff)))
        )
        clipped=(
            gpd.clip(
                parcels.to_crs(crs_ca)
                ,pt.to_crs(crs_ca).buffer(buff)
            )
            .to_crs(use_crs)
        )
        p = p +geom_map(clipped,alpha=0.4)
    p = (p+geom_map(pt.to_crs(use_crs)
                    ,color='green',fill='green',alpha=0.2)
            +geom_map(pt.switch_geometry('main_geometry').to_crs(use_crs),color=color,fill=color,size=5)
            # +geom_map(pt.switch_geometry('main_geometry').to_crs(use_crs),color='purple',fill=None,size=buff,linetype='dashed')
            +geom_map(pt.assign(cir=lambda x: x.switch_geometry('main_geometry').to_crs(crs_ca).buffer(buff).to_crs(use_crs))
                      .switch_geometry('cir').to_crs(use_crs)
                      ,color='purple',fill=None,linetype='dashed')
            +ggtitle(f'{pt.iloc[0]["name"]}:\n{pt.iloc[0].addr}')
        )
    axs2.append(
        {'pt': pt,
         'p' : p
        }
    )

# %%

# def forceAspect(ax,aspect=1):
#     im = ax.get_images()
#     extent =  im[0].get_extent()
#     ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

# for d in axs2:
for i in range(len(axs2)):
    d = axs2[i]
    p=d['p']
    mpl.rcParams['figure.dpi'] = 1000
    fig = p.draw()
    ax = fig.get_axes()[0]

    ctx.add_basemap(ax, crs=use_crs,
                    source='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoiY3JpbmR0IiwiYSI6ImNpdDZhdDZ2eTAxbzMyb3BndXFzZ3R5aHkifQ.CEdMDNgiCc-KyouaXUYWew',
                    zoom_adjust=1,
                    attribution_size=4)
    print(f'{i+1} of {len(axs2)}')
    display(fig)
    print(d['pt'].iloc[0].geocode)

# %%


### FOR NUMBER OF VEHICLES PER YARD:
# chatgpt: what is a reasonable assumption for the number of cars per acre that is typically stored in an auto junkyard



# A reasonable assumption for the number of cars per acre typically stored in an auto junkyard is 100 to 150 cars per acre.

# Low-density yards (with wide aisles, less stacking): ~80–100 cars/acre
# Average yards: ~100–150 cars/acre
# High-density yards (tight stacking, multi-level): up to 200+ cars/acre
# Source:

# Industry estimates and planning documents (e.g., EPA Brownfields Auto Salvage Yards)
# Local zoning and environmental reports
# Note: Actual numbers vary by yard layout, stacking, and regulations. For most planning or estimation purposes, 100–150 cars/acre is a practical range.


# FOR TURNOVER:

# ChatGPT: what is the typical annual vehicle turnover of a junkyard?

# A typical annual vehicle turnover for an auto junkyard (auto dismantler or salvage yard) is 1 to 2 times per year. This means that, on average, a junkyard will process and replace its entire inventory of vehicles once or twice annually.

# Low-turnover yards: 0.5–1 times per year (vehicles may stay 1–2 years)
# Average yards: 1–2 times per year (most common)
# High-turnover yards: 2–3+ times per year (very efficient, high-volume operations)
# Sources:

# EPA Brownfields Auto Salvage Yards
# Industry reports and state/local permitting documents
# Note: Actual turnover depends on yard size, business model (self-service vs. full-service), demand, and local regulations. For planning, 1–2 turnovers per year is a reasonable estimate.


# %%
# import fiona
# import os

# gdf = all.rename(columns={'NAME': 'county'})

# kml_path = 'all_counties.kml'  # Note the 'KML:' prefix

# # Remove the file if it exists to avoid appending to an old file
# if not os.path.exists('all_counties.kml'):
#     os.mkdir('dismantlers')

# with fiona.Env():
#     for county, df in gdf.groupby('county'):
#         df[['name', 'street', 'city', 'zip', 'county', 'geometry']].to_file(
#             f'dismantlers/{county}.kml',
#             driver='KML',
#             layer=county
#         )
# %%
# Exports to multi-layer KML using simplekml (readable by Google Earth)
import simplekml

gdf = all.rename(columns={'NAME': 'county'})

kml = simplekml.Kml()

for county, group in gdf.groupby('county'):
    print(county)
    display(group)
    folder = kml.newfolder(name=county)
    ggroup=group.explode('geometry').reset_index(drop=True)   # explode multipolygons
    for idx, row in ggroup.iterrows():
        # Assuming geometry is Point or Polygon
        geom = row.geometry
        if geom.geom_type == 'Point':
            pnt = folder.newpoint(
                name=row.get('name', ''),
                coords=[(geom.x, geom.y)]
            )
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            pol = folder.newpolygon(
                name=row.get('name', ''),
                outerboundaryis=list(geom.exterior.coords) if geom.geom_type == 'Polygon' else None
            )
        # Add more geometry types as needed
        # Add extended data
        desc = f"Street: {row.get('street', '')}<br>City: {row.get('city', '')}<br>Zip: {row.get('zip', '')}<br>County: {row.get('county', '')}"
        if geom.geom_type == 'Point':
            pnt.description = desc
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            pol.description = desc

kml.save("all_counties.kml")
# %%
import shutil
all.rename(columns={'NAME': 'county'})[['name', 'street', 'city', 'zip', 'county', 'geometry']].to_file('all_dismantlers')
shutil.make_archive("all_dismantlers", "zip", "all_dismantlers")
# %%





#%%
tmp=(
    zz.assign(zip=lambda x: x.o_rdrsid.str.replace('ZIP',''))
    .groupby(['zip','o_rdrsid','county_left']).agg({'gcnt':'first'})
    .sort_values('gcnt',ascending=False)
    .assign(
        frac=lambda x: x.gcnt/x.gcnt.sum(),
        ftot=lambda x: x.frac.cumsum()
    )
    .join(zctaxx.set_index('GEOID'),on='zip').reset_index()
    .set_geometry('geometry').to_crs(crs_ll)
    .rename(columns={'o_rdrsid':'Name'})[['Name','county_left','gcnt','frac','ftot','geometry']]
    .filt(lambda x: ~x.geometry.is_empty & ~x.geometry.isna())  # filter out empty geometries
)
import simplekml

# Assuming 'tmp' is your GeoDataFrame
kml = simplekml.Kml()

for idx, row in tmp.filt(lambda x: x.ftot<0.95).iterrows():
    # Get the polygon coordinates (assuming geometry is a Polygon or MultiPolygon)
    geom = row['geometry']
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        print(f'POLYGON {row["Name"]}')
        pol = kml.newpolygon(
            name=f'#{idx+1}: {row["Name"]}',
            outerboundaryis=coords
        )
    elif geom.geom_type == 'MultiPolygon':
        print(f'MULTIPOLYGON {row["Name"]}')
        cnt=0
        coords=None
        for poly in geom.geoms:
            tcoords = list(poly.exterior.coords)
            # only keep the biggest polygon
            if coords is None or len(coords) < len(tcoords):
                coords = tcoords
        pol = kml.newpolygon(
            name=f'#{idx+1}: {row["Name"]}',
            outerboundaryis=coords
        )
    else:
        continue  # Skip if not a polygon

    # Add description with other fields
    description = f"county_left: {row['county_left']}<br>vehicles retired in 2021: {row['gcnt']} ({row['frac']:.2%})"
    pol.description = description
    # Style: blue fill, blue outline
    pol.style.polystyle.color = simplekml.Color.changealphaint(50, simplekml.Color.blue)  # semi-transparent blue
    pol.style.linestyle.color = simplekml.Color.blue
    pol.style.linestyle.width = 2


# Save to file
kml.save("zcta_partial.kml")
# %%

import zipfile
import re
y=2021
ddf=[]
for ffn in ['fleetdb-90000-92500','fleetdb-92501-95000','fleetdb-95001-95500','fleetdb-95500-96000','fleetdb-96001-96xxx']:
    for y in range(2021,2023):
        fn=f'{ffn}-{y}.zip'
        with zipfile.ZipFile(fn) as z:
            ff=f'FleetDB-Zipcode-Selected-{y}-Agg-NoGVWR-Agg-Agg-Agg-All-Agg-Agg.csv'
            with z.open(ff) as f:
                ddf.append(pd.read_csv(f,skiprows=13).clean_names().assign(calyear=y))
df=pd.concat(ddf,ignore_index=True)
# %%
ddf=[]
y=2021
for ffn in ['P','T1_T2','T3_T4','T5_T6_T7_BS_BT_B_MH_MC']:
    fn=f'fleetdb/{y}/FleetDB-Zipcode-Selected-{y}-{ffn}-GVWR-All-All-Agg-All-All-ByCensusBlockGroupCode.csv.zip'
    ddf.append(pd.read_csv(fn,skiprows=13,compression='zip').clean_names().assign(calyear=y))
y=2022
for ffn in ['P','T1','T2','T3_T4','T5_T6_T7_BS_BT_B_MH_MC']:
    fn=f'fleetdb/{y}/FleetDB-Zipcode-Selected-{y}-{ffn}-GVWR-All-All-Agg-All-All-ByCensusBlockGroupCode.csv.zip'
    ddf.append(pd.read_csv(fn,skiprows=13,compression='zip').clean_names().assign(calyear=y))
df=pd.concat(ddf,ignore_index=True)
# %%
dfx=df.assign(model_year=lambda x: pd.to_numeric(x.model_year,errors='coerce'),
            age=lambda x: x.model_year-x.calyear-1)
key=['vehicle_category','gvwr_class','fuel_type','fuel_technology','county','mpo','sub_area','census_block_group_code','zip_code','age']; 
dfxx=dfx.groupby(key+['calyear'])[['vehicle_population']].sum()
dfxx2021=dfxx.reset_index().filt(lambda x: x.calyear==2021)
dfxx2022=dfxx.reset_index().filt(lambda x: x.calyear==2022)
dfxxj=dfxx2021.join(dfxx2022.set_index(key),on=key,rsuffix='_2022').assign(change=lambda x: x.vehicle_population_2022-x.vehicle_population)

# #%%
# (
#     ggplot()
#     +geom_bar