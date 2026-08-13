import pandas as pd
import requests
import re
import janitor
import time
from core.common import wdisplay

def get_geocode(url):
    if not pd.isna(url) and re.match(r'^http',url):

        g=re.search(r'.*@([-\d\.]+),([-\d\.]+),\d+m/data.*',url)
        if ( g ):
            print (g.groups())
            return g.groups()
        else:
 
            resp=requests.get(url)
            g=re.search(r'.*@([-\d\.]+),([-\d\.]+),\d+m/data.*',resp.url)
            time.sleep(0.25)
            print (g.groups()) if g else None
            return g.groups() if g else None
    else:
        return None

df=(
    pd.read_csv('/Users/crindt/Downloads/25051_facility classification checks - all_ent.csv')
    .clean_names()
    .assign(mangeo=lambda x: x.apply(lambda row: get_geocode(row.alt_geocode), axis=1))
)
wdisplay(df)
