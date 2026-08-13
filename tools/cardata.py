#%%
import pandas as pd
import numpy as np
import geopandas as gpd
from core.common import cpath, ureg, case_when, wdisplay, wldisplay
import janitor
from difflib import get_close_matches, SequenceMatcher

# Force reload of match_dmv_cars if already loaded
import importlib
import match_dmv_cars
importlib.reload(match_dmv_cars)
from match_dmv_cars import match_dmv_to_cars

#%%
carsraw = pd.read_csv(cpath('elv-data','cars-dataset.csv')).clean_names()

#%%
cars = (
    carsraw
    .assign(
        cweight=lambda x: x.unladen_weight.str.replace(r'(\d+)[\+-]*\s+.*', r'\1', regex=True)
                            .str.replace(',', '').astype(float)
    )
    .filt(lambda x: ~pd.isna(x.cweight))

    # Expand comma-delimited production years into separate rows
    .assign(production_years=lambda x: x.production_years.str.split(','))
    .explode('production_years')
    .assign(production_years=lambda x: x.production_years.str.strip())
)

# %%
dmvraw = pd.read_csv('dmv-augmented.csv.zip').clean_names()

# Use the 'make' field if it is non-null, otherwise fall back to 'make_a'
dmvraw = dmvraw.assign(make_final=lambda x: np.where(x.make.notna(), x.make, x.make_a))

#%%
dmv_sample = dmvraw.sample(n=100, random_state=42)

#%%
# Perform matching
matched_results = match_dmv_to_cars(dmv_sample, cars)

# %%
