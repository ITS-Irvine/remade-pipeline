
##### FIXMEFIXME: this is random code stuck here for generating a polygon to grab western states for OSM
import geopandas as gpd
import importlib
import core.common
importlib.reload(common)
from services.geocode import crs_ll
import services.geocode
import janitor
states=gpd.read_file(cpath('gis_data','cb_2018_us_state_500k.zip')).clean_names()
display(states)
st=states.filt(lambda x: x.stusps.str.match('CA|NV|AZ|OR|WA')).assign(country=lambda x: 'US')
display(st)
from shapely.ops import unary_union
region=gpd.GeoDataFrame(st.groupby('country').agg({'geometry':unary_union}).reset_index(),geometry='geometry').set_crs(crs_ll)
display(region)
region.plot()

import geopandas as gpd

def write_poly(df, path, geometry_column = "geometry"):
    df = df.to_crs("EPSG:4326")

    df["aggregate"] = 0
    area = df.dissolve(by = "aggregate")[geometry_column].values[0]

    if not hasattr(area, "exterior"):
        print("Selected area is not connected -> Using convex hull.")
        area = area.convex_hull

    data = []
    data.append("polyfile")
    data.append("polygon")

    for coordinate in area.exterior.coords:
        data.append("    %e    %e" % coordinate)

    data.append("END")
    data.append("END")

    with open(path, "w+") as f:
        f.write("\n".join(data))
        
write_poly(region, "uswest.poly")

