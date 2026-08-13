#%%
import pandas as pd

from layers.rdrs import load_rdrs_flow,get_rdrs_flow_for_year
import core.common
from core.common import cpath,wdisplay

#%%

rdrs_flow=load_rdrs_flow()
matmap=pd.read_excel(cpath('processed_data','MATCATS_MaterialMappings24037.xlsx')).clean_names()

flw=(
    get_rdrs_flow_for_year(rdrs_flow,2021,matmap)
    .groupby(['o_rdrsid','d_rdrsid','grouping4'])[['tonssent']].sum()
    .reset_index()
)
wdisplay(flw)
# %%
ents=(
    pd.read_csv('/Users/crindt/Downloads/25051_facility classification checks - all_ent (3).csv').clean_names()
    .assign(
        inflow=lambda x: x.inflow.fillna(0),
        outflow=lambda x: x.outflow.fillna(0)
    )
)
wdisplay(ents)
# %%
ostats=flw.join(ents.set_index('rdrs_id')[['status','compositional_category']],on='o_rdrsid',how='left')
wdisplay(ostats[['o_rdrsid','status','grouping4','compositional_category','tonssent']])
dstats=flw.join(ents.set_index('rdrs_id')[['status','compositional_category']],on='d_rdrsid',how='left')
wdisplay(dstats[['d_rdrsid','status','grouping4','compositional_category','tonssent']])


# %%
inout=(
    ostats.rename(columns={'o_rdrsid':'rdrsid'}).groupby(['rdrsid','status','grouping4','compositional_category'])[['tonssent']].sum().add_col_suffix('_out')
    .join(dstats.rename(columns={'d_rdrsid':'rdrsid'}).groupby(['rdrsid','status','grouping4','compositional_category'])[['tonssent']].sum().add_col_suffix('_in')
          ,how='outer'
          ,on=['rdrsid','status','grouping4','compositional_category'])
    .apply(lambda x: x.fillna(0))
    .assign(net=lambda x: x.tonssent_in-x.tonssent_out)
    .reset_index()
)
# %%
inoutx=(
    inout.groupby(['status','grouping4'])[['tonssent_out','tonssent_in']].sum().reset_index()
    .assign(tflows=lambda x: x.tonssent_out+x.tonssent_in)
    .sort_values(['tflows'],ascending=[False])
    # .filt(lambda x: ~x.status.str.match('.*Good'))
)
# %%
inoutx.groupby('status')[['tonssent_out','tonssent_in','tflows']].sum().assign(
    percent=lambda x: x.tflows/x.tflows.sum()*100
)
# %%
ldisplay(
    inoutx.assign(
        percent=lambda x: x.tflows/x.tflows.sum()*100
    )
    .sort_values(['status'],ascending=True)
    .filt(lambda x: ~x.status.str.match('.*Good'))
    .assign(
        percentofbad=lambda x: x.percent/x.percent.sum()*100
    )

)

# %%
wdisplay(
    ents.filt(lambda x: x.note.fillna('').str.upper().str.match(r'.*(OFFICE|COMMERCIAL|HOME|RESIDENCE|PRIVATE)'))
    .groupby('status')[['inflow','outflow']].sum()
    .assign(
        totflow=lambda x: x.inflow+x.outflow)
    .assign(percentofoverall=lambda x: x.totflow/inoutx.tflows.sum()*100)
    )
# %%
officeents=(
    ents
    .filt(lambda x: x.note.fillna('').str.upper().str.match(r'.*(OFFICE|COMMERCIAL|HOME|RESIDENCE|PRIVATE)'))
    .assign(totflow=lambda x: x.inflow+x.outflow)
    .filt(lambda x: x.totflow>0)
    .sort_values(['totflow'],ascending=False)
    .assign(
        mergegeocode=lambda x: x.alt_geocode.mask(x.alt_geocode.isna(),x.gmaps_link)
    )
    [['rdrs_id','totflow','mergegeocode']]
    )
officeents.to_csv('officeents.csv',index=False)
wdisplay(officeents)
# %%
