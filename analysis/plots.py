from collections import OrderedDict
import re
from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import shapely.geometry
import geopandas as gpd
from plotnine import facet_grid, ggplot, geom_map, aes, ggtitle, scale_size_continuous, theme
import contextily as ctx
import logging
from IPython.display import display
from services.geocode import crs_ca, crs_ggl, crs_ll, crs_usa
from core.common import case_when

from pint import UnitRegistry
ureg = UnitRegistry()
from core.model_types import *
import pygris as pyg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @pa.check_types
# def plot_non_maritime_routes(distances: DataFrame[ODFlowsWithClips], 
#                              coabdis, 
#                              use_crs=crs_ggl):
#     logger.info("DEPRECATED PLOT STUFF: NON-MARITIME")
#     pltdatx = (
#         distances
#         .sort_values(['o_id', 'd_id', 'step_num'])
#         .set_geometry('geometry_clip')
#         .reset_index(drop=True).explode('geometry_clip', index_parts=True)
#     )

#     # select 3 OD pairs with no maritime leg
#     select_pairs_ca = (
#         pltdatx
#         .filt(lambda x: x.geometry_searte.isna())  # limit to no searoutes
#         .groupby(['o_rdrsid', 'd_rdrsid'])  # get distinct OD pairs
#         .first()[[]]
#         .reset_index()
#         # incrementally count how many times each origin and destination have been seen
#         .assign(
#             ocnt=lambda x: x.groupby('o_rdrsid').cumcount(),
#             dcnt=lambda x: x.groupby('d_rdrsid').cumcount(),
#         )
#         # select pairs where the O and D are both being seen the first time
#         .filt(lambda x: (x.ocnt == 0) & (x.dcnt == 0))
#         # keep the top three
#         .head(3)
#     )

#     for grp, pltdat in pltdatx.join(
#         select_pairs_ca.set_index(['o_rdrsid', 'd_rdrsid']),
#         how='inner', on=['o_rdrsid', 'd_rdrsid']
#     ).groupby(['o_rdrsid', 'd_rdrsid']):
#         p = (ggplot()
#              + geom_map(pltdat
#                         .switch_geometry('geometry_clip').to_crs(use_crs),
#                         aes(color='region', fill='region'), size=2)
#              + geom_map(pltdat
#                         .assign(
#                             endpt=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_clip.coords[-1]), axis=1)
#                         )
#                         .rename(columns={'geometry': 'holdme', 'endpt': 'geometry'})
#                         .set_geometry('geometry').set_crs(crs_ll).to_crs(use_crs)
#                         , color='black', fill='black', size=.25)
#              + geom_map(pltdat
#                         .assign(
#                             endpt=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_step.coords[-1]), axis=1)
#                         )
#                         .rename(columns={'geometry': 'holdme', 'endpt': 'geometry'})
#                         .set_geometry('geometry').set_crs(crs_ll).to_crs(use_crs)
#                         , color='purple', fill=None, size=3)
#              )
#         last = pltdat.iloc[-1]
#         if last.geometry_searte is not None:
#             p = (p
#                  + geom_map(pltdat.tail(-1)
#                             .switch_geometry('geometry_searte').set_crs(crs_ll).to_crs(crs_ggl),
#                             color='blue',
#                             size=2.5
#                             )
#                  )
#         p = (p
#              + geom_map(gpd.clip(coabdis, shapely.geometry.box(
#                  *(pltdat.geometry_clip.total_bounds)
#              )).to_crs(use_crs), fill=None, color='blue', size=0.1)
#              + geom_map(pltdat.switch_geometry('geometry_coabdis').to_crs(use_crs),
#                         aes(color='region'),
#                         fill=None)
#              + geom_map(pltdat
#                         .switch_geometry('geometry_merged_orig').set_crs(crs_ll).to_crs(crs_ggl),
#                         color='green', fill='green', size=4)
#              + geom_map(pltdat
#                         .switch_geometry('geometry_merged_dest').set_crs(crs_ll).to_crs(crs_ggl),
#                         color='red', fill='red', size=4)
#              + ggtitle(f'In-California Route for flows between {grp[0]} and {grp[1]}')
#              )

#         fig = p.draw()
#         ax = fig.get_axes()[0]

#         ctx.add_basemap(ax, crs=use_crs,
#                         source=ctx.providers.CartoDB.Positron,
#                         attribution_size=4)
#         display(fig)

def plot_maritime_routes(distances, use_crs):
    logger.info("DEPRECATED PLOT STUFF: MARITIME")
    pltdatx = (
        distances
        .sort_values(['o_rdrsid', 'd_rdrsid', 'step_num'])
        .set_geometry('geometry_clip')
        .reset_index(drop=True).explode('geometry_clip', index_parts=True)
    )

    select_pairs_intl = (
        pltdatx
        .filt(lambda x: ~x.geometry_searte.isna())  # limit to no searoutes
        .groupby(['o_rdrsid', 'd_rdrsid'])  # get distinct OD pairs
        .first()[[]]
        .reset_index()
        # incrementally count how many times each origin and destination have been seen
        .assign(
            ocnt=lambda x: x.groupby('o_rdrsid').cumcount(),
            dcnt=lambda x: x.groupby('d_rdrsid').cumcount(),
        )
        # select pairs where the O and D are both being seen the first time
        .filt(lambda x: (x.ocnt == 0) & (x.dcnt == 0))
        # keep the top three
        .head(100)
        .tail(3)
    )

    proj=ccrs.Robinson(central_longitude=-117)

    for grp, pltdat in pltdatx.join(
        select_pairs_intl.set_index(['o_rdrsid', 'd_rdrsid']),
        how='inner', on=['o_rdrsid', 'd_rdrsid']
    ).groupby(['o_rdrsid', 'd_rdrsid']):
        fig = plt.figure()
        fig.suptitle(f'In-California and Maritime Routes between {grp[0]} and {grp[1]}', fontsize=12)
        ax = fig.add_subplot(1, 2, 1, projection=proj)
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)

        ax = (
            pltdat.switch_geometry('geometry_clip').to_crs(use_crs)
            .plot(ax=ax, linewidth=2, color='red', transform=proj)
        )

        ax = (pltdat.switch_geometry('geometry_merged_orig').set_crs(crs_ll).to_crs(use_crs)
              .plot(ax=ax, color='green', transform=proj))
        

        # Check for invalid geometries
        pltdat2 = pltdat.tail(1).assign(
            geometry_port=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_step.coords[-1]), axis=1)
        ).switch_geometry('geometry_port').set_crs(crs_ll).to_crs(use_crs)

        # Ensure geometries are valid
        pltdat2 = pltdat2[pltdat2.is_valid]

        # Check for missing or infinite values
        pltdat2 = pltdat2.dropna(subset=['geometry_port'])
        pltdat2 = pltdat2[pltdat2['geometry_port'].apply(lambda geom: geom.is_finite)]

        # Plot the data
        # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=-117)})
        ax=pltdat2.plot(ax=ax, color='purple', transform=ccrs.PlateCarree())

        # Add basemap
        ctx.add_basemap(ax, crs=use_crs, source=ctx.providers.CartoDB.Positron)

        # ax = (pltdat.tail(1)
        #       .assign(
        #           geometry_port=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_step.coords[-1]), axis=1)
        #       )
        #       .switch_geometry('geometry_port').set_crs(crs_ll).to_crs(use_crs)
        #       .plot(ax=ax, color='purple', transform=proj))

        ax = (
            pltdat.switch_geometry('geometry_clip').to_crs(use_crs)
            .plot(ax=ax, linewidth=2, color='red', transform=proj)
        )
        ax.set_title("In-California on-road route")

        ctx.add_basemap(ax, crs=use_crs,
                        source=ctx.providers.CartoDB.Positron,
                        attribution_size=4)

        # maritime
        ax2.stock_img()
        ax2 = (pltdat
               .assign(
                   endpt=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_step.coords[-1]), axis=1)
               )
               .switch_geometry('endpt').set_crs(crs_ll).to_crs(use_crs)
               .plot(ax=ax2, color='purple', linewidth=3,
                     transform=proj)
               )
        ax2 = (pltdat
               .assign(
                   endpt=lambda x: x.apply(lambda row: shapely.geometry.Point(row.geometry_searte.coords[-1]), axis=1)
               )
               .switch_geometry('endpt').set_crs(crs_ll).to_crs(use_crs)
               .plot(ax=ax2, color='red', linewidth=3,
                     transform=proj)
               )
        last = pltdat.iloc[-1]
        if last.geometry_searte is not None:
            ax2 = (pltdat.tail(-1)
                   .switch_geometry('geometry_searte').set_crs(crs_ll).to_crs(use_crs)
                   .plot(ax=ax2, color='blue', linewidth=2.5,
                         transform=proj)
                   )

        ax2.set_title("Maritime route (approximate)")

        display(fig)

def plot_onroad_flows(od_flows_w_dist, ca_boundary, use_crs):
    """
    Plot on-road flows of all California waste in 2021 according to the RDRS database.
    Color shows the material type and line thickness is proportional to the tonnage.
    """
    logger.info("DEPRECATED PLOT STUFF: ONROAD FLOWS AGGREGATED")
    pltdat = od_flows_w_dist.switch_geometry('geometry_clip').to_crs(use_crs)
    p = (ggplot()
         + geom_map(
             pltdat,
             aes(color='grouping4',
                 fill='grouping4',
                 size='tonssent'),
             alpha=0.5
         )
         + scale_size_continuous(range=[0.1, 5])
         + geom_map(ca_boundary.to_crs(use_crs), fill='none')
         + theme(figure_size=(12, 12))
         )
    fig = p.draw()
    ax = fig.get_axes()
    ctx.add_basemap(ax[0], crs=pltdat.crs.to_string(),
                    source=ctx.providers.CartoDB.Positron)
    return fig



def plot_onroad_flows_by_material(od_flows_w_dist, ca_boundary, use_crs):
    logger.info("DEPRECATED PLOT STUFF: ONROAD FLOWS BY GROUPING4")

    for mat, matdf in od_flows_w_dist.switch_geometry('geometry_clip').to_crs(use_crs).groupby('grouping4'):
        logger.info(f"...Plotting {mat}")
        p = (ggplot()
                + geom_map(matdf, aes(size='tonssent'), color='blue', fill='blue', alpha=0.5)
                + scale_size_continuous(range=[0.1, 3])
                + geom_map(ca_boundary.to_crs(use_crs), fill='none')
                + ggtitle(f'In-California flows of {mat}')
                )
        fig = p.draw()
        ax = fig.get_axes()
        ctx.add_basemap(ax[0], crs=matdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        display(fig)



def plot_emissions_by_material(emiss, model_year, ca_boundary):
    logger.info("DEPRECATED PLOT STUFF: CoAbDis EMISSIONS by material")

    pltdatx = (
        emiss
        .rename(columns={'grouping4': 'Material'})
        .groupby(['region', 'Material']).agg({
            'geometry_coabdis': 'first',
            'emiss_pm25': 'sum',
            'emiss_nox': 'sum',
            'emiss_ghg': 'sum'
        })
    )

    pltdat = gpd.GeoDataFrame(
        pltdatx
        .reset_index()
        .melt(id_vars=['region', 'Material', 'geometry_coabdis'],
                value_vars=['emiss_pm25', 'emiss_nox', 'emiss_ghg'],
                var_name='Pollutant',
                value_name='g_poll')
        .filt(lambda x: x.Material.isin(['Cardboard', 'Paper', 'PET']))
        .assign(kg_poll=lambda x: x.g_poll * ureg('g').to('kg').magnitude)
    ).switch_geometry('geometry_coabdis').set_crs(crs_ll).to_crs(crs_ca)

    for pol in ['emiss_pm25', 'emiss_nox', 'emiss_ghg']:
        p = (
            ggplot()
            + geom_map(
                pltdat
                .filt(lambda x: x.Pollutant == pol),
                aes(fill='kg_poll'), color='white')
            + facet_grid(['Pollutant', 'Material'], space='free')
            + geom_map(ca_boundary.to_crs(crs_ca), color='black', fill=None)
            + ggtitle(f"{pol} emissions in {model_year}")
            + theme(figure_size=(14, 4))
        )
        fig = p.draw()
        for ax in fig.get_axes():
            ctx.add_basemap(ax, crs=pltdat.crs.to_string(),
                            source=ctx.providers.CartoDB.Positron,
                            attribution_size=6)
        display(fig)


def classify_endpoints_ct(df):
    return df.assign(
        endpoints_class=lambda x:
        case_when(x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.o_state.str.match('Connect'),'Other '+x.o_n2,
                    x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.o_state.str.match('Connect'),'CT '+x.o_n2,
                    ~x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.o_state.str.match('Connect'),'Specific CT loc',
                    ~x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.o_state.str.match('Connect'),'Specific OOS loc',
                    True, 'What?')
        +" to "
        +case_when(x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.d_state.str.match('Connect'),'other'+x.d_n2,
                    x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.d_state.str.match('Connect'),'CT '+x.d_n2,
                    ~x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.d_state.str.match('Connect'),'Specific CT loc',
                    ~x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.d_state.str.match('Connect'),'Specific OOS loc',
                    True, 'What?')
    )

def classify_endpoints_ca(df):
    return df.assign(
        endpoints_class=lambda x:
        case_when(x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') 
                    & ~x.o_state.str.match('California|CA'),'Other '+x.o_n2,
                    x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.o_state.str.match('California|CA'),'CA '+x.o_n2,
                    ~x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.o_state.str.match('California|CA'),'Specific CA loc',
                    ~x.o_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.o_state.str.match('California|CA'),'Specific OOS loc',
                    True, 'What?')
        +" to "
        +case_when(x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.d_state.str.match('California|CA'),'other'+x.d_n2,
                    x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.d_state.str.match('California|CA'),'CA '+x.d_n2,
                    ~x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & x.d_state.str.match('California|CA'),'Specific CA loc',
                    ~x.d_n2.str.match(r'\[(STATE|COUNTY|CITY|TOWN)\]') & ~x.d_state.str.match('California|CA'),'Specific OOS loc',
                    True, 'What?')
    )


# default function that just returns its argument
_return_same = lambda x: x
_all_rows = lambda x: x.index==x.index

def plot_flows_by_criteria(
        emiss: DataFrame, # FIXME
        pregroup_filter: Callable[[DataFrame],DataFrame] = _all_rows,
        transform: Callable[[DataFrame],DataFrame] = _return_same,
        postgroup_filter: Callable[[DataFrame],DataFrame] = _all_rows,
        save_to:str=None,
        basemap=ctx.providers.CartoDB.Positron

    ):
    dat=(
        emiss
        .filt(pregroup_filter)
        .pipe(transform)
        .groupby(['o_id','d_id'])
        .agg({'o_n1':'first','d_n1':'first','endpoints_class':'first','clip_distance_mi':'sum','distance_m':'first','tonssent':'sum','geometry_full':'first'})
        .reset_index()
        .filt(postgroup_filter)
        .set_geometry('geometry_full').set_crs(crs_ll)
        .assign(wid=lambda x: 1+4*(x.tonssent-x.tonssent.min())/(x.tonssent.max()-x.tonssent.min()))
        .to_crs(crs_usa)
    )

    # Create clipping boundary (we don't want big states to extend the plot)
    # Get bounds of flows
    minx, miny, maxx, maxy = dat.total_bounds


    # Sort categories so CT comes first, then OOS
    is_categories = sorted([cat for cat in dat['endpoints_class'].unique() 
                            if re.match(r'.*(to.*(CA|CT))',str(cat))])
    oos_categories = sorted([cat for cat in dat['endpoints_class'].unique() 
                            if re.match(r'.*to.*OOS', str(cat))])
    other_categories = sorted([cat for cat in dat['endpoints_class'].unique() 
                            if not re.match(r'.*to.*(CA|CT)',str(cat)) 
                            and not re.match('.*to.*OOS', str(cat))])

    # Assign colors
    greens = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b']
    # blues = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']
    blues = ['#08306b', '#4292c6', '#6baed6', '#9ecae1']

    reds = ['#7f0000', '#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1']
    grays = ['#cccccc']

    # Extend colors if needed
    is_colors = blues[:len(is_categories)]
    oos_colors = reds[:len(oos_categories)]
    other_colors = grays * len(other_categories)

    all_colors = is_colors + oos_colors + other_colors

    ordered_cats = list( # remove any duplicate cats
        OrderedDict.fromkeys(is_categories + oos_categories + other_categories)
    )
    dat['endpoints_class'] = pd.Categorical(
        dat['endpoints_class'],
        categories=ordered_cats,
        ordered=True
    )

    # Create colormap
    custom_cmap = mpl.colors.ListedColormap(all_colors)

    ax=dat.plot(
        column='endpoints_class',
        cmap=custom_cmap,
        categorical=True,
        legend=True,
        legend_kwds={
            'title':'Route endpoints',
            'loc':'lower right'
            # 'bbox_to_anchor': (0.5, -0.1) # outside centered legend
            },
        linewidth=dat.wid
        )

    # add boundaries for states crossed
    ax=(
        pyg.states(year=2021).to_crs(crs_usa)
        .sjoin(dat.to_crs(crs_usa),predicate='intersects',how='inner')
        [['STUSPS','geometry']].drop_duplicates()
        .plot(ax=ax
            #   ,facecolor='#00000033'
              ,facecolor='none'
              ,edgecolor='#000000aa'
              ,linewidth=0.25)
    )
    # Add 10% padding
    x_padding = (maxx - minx) * 0.1
    y_padding = (maxy - miny) * 0.1
    ax.set_xlim(minx-x_padding,maxx+x_padding)
    ax.set_ylim(miny-y_padding,maxy+y_padding)

    # ticks show state plane coords, which isn't helpful, so remove them
    plt.xticks([])
    plt.yticks([])

    ctx.add_basemap(ax, crs=crs_usa, alpha=0.5, attribution='',source=basemap)
    if save_to is not None:
        plt.savefig(save_to,dpi=300)


def plot_flows_by_material(
        emiss: DataFrame, # FIXME
        pregroup_filter: Callable[[DataFrame],DataFrame] = _all_rows,
        transform: Callable[[DataFrame],DataFrame] = _return_same,
        postgroup_filter: Callable[[DataFrame],DataFrame] = _all_rows,
        save_to:str=None,
        cmap='Paired',
        basemap=ctx.providers.CartoDB.Positron,
        line_width_range=(1,4),
        figsize=(8,6),
        legend_kwds={
            'title':'Materials',
            'loc': 'center left',
            'bbox_to_anchor': (1, 0.5),
            'frameon': True,
            'fontsize': 10
            },

    ):
    dat=(
        emiss
        .filt(pregroup_filter)
        .pipe(transform)
        .groupby(['o_id','d_id','material_grouping'])
        .agg({'o_n1':'first','d_n1':'first','clip_distance_mi':'sum','distance_m':'first','tonssent':'sum','geometry_full':'first'})
        .reset_index()
        .filt(postgroup_filter)
        .set_geometry('geometry_full').set_crs(crs_ll)
        .assign(wid=lambda x: line_width_range[0]+line_width_range[1]*(x.tonssent-x.tonssent.min())/(x.tonssent.max()-x.tonssent.min()))
        .to_crs(crs_usa)
    )

    # Create clipping boundary (we don't want big states to extend the plot)
    # Get bounds of flows
    minx, miny, maxx, maxy = dat.total_bounds

    ax=dat.plot(
        column='material_grouping',
        categorical=True,
        cmap=cmap,
        legend=True,
        legend_kwds=legend_kwds,
        linewidth=dat.wid,
        figsize=figsize
        )

    # add boundaries for states crossed
    ax=(
        pyg.states(year=2021).to_crs(crs_usa)
        .sjoin(dat.to_crs(crs_usa),predicate='intersects',how='inner')
        [['STUSPS','geometry']].drop_duplicates()
        .plot(ax=ax
            #   ,facecolor='#00000033'
              ,facecolor='none'
              ,edgecolor='#000000aa'
              ,linewidth=0.25)
    )
    # Add 10% padding
    x_padding = (maxx - minx) * 0.1
    y_padding = (maxy - miny) * 0.1
    ax.set_xlim(minx-x_padding,maxx+x_padding)
    ax.set_ylim(miny-y_padding,maxy+y_padding)

    # ticks show state plane coords, which isn't helpful, so remove them
    plt.xticks([])
    plt.yticks([])

    ctx.add_basemap(ax, crs=crs_usa, alpha=0.5, attribution='',source=basemap)
    if save_to is not None:
        plt.savefig(save_to,dpi=300)

    return ax



def plot_flows_by_material_faceted_simple(
        emiss_in: DataFrame,
        facet_by: str = 'material_grouping',
        save_to: str = None,
        cmap='Paired',
        ncol: int = 3,
        basemap=ctx.providers.CartoDB.Positron
    ):
    
    # Data preparation
    dat = (
        emiss_in
        .groupby(['o_id','d_id','material_grouping','layer'])
        .agg({'clip_distance_mi':'sum','tonssent':'sum','geometry_full':'first'})
        .reset_index()
        .set_geometry('geometry_full').set_crs(crs_ll)
        .rename_geometry('geometry')
        .assign(
            wid=lambda x: 1+4*(x.tonssent-x.tonssent.min())/(x.tonssent.max()-x.tonssent.min())
        )
        .to_crs(crs_usa)
    )
    
    # States
    states = (
        pyg.states(year=2021).to_crs(crs_usa)
        .sjoin(dat, predicate='intersects', how='inner')
        [['STUSPS','geometry']].drop_duplicates()
    )
    
    # Replicate states for all facets
    facet_values = dat[facet_by].unique()
    states_all = pd.concat([
        states.assign(**{facet_by: val}) for val in facet_values
    ])
    
    # Calculate bounds
    minx, miny, maxx, maxy = dat.total_bounds
    padding = 0.1
    
    # Plot
    p = (
        ggplot()
        + geom_map(states_all, aes(group='STUSPS'), 
                   color='#00000066', fill=None, size=0.25)
        + geom_map(dat, aes(color='material_grouping', size='wid', group='material_grouping'))
        # + facet_wrap(f'~{facet_by}', ncol=ncol)
        # + scale_color_cmap(cmap)
        # + scale_size_continuous(range=(1, 5), guide=False)
        + coord_fixed(
            xlim=(minx*(1-padding), maxx*(1+padding)),
            ylim=(miny*(1-padding), maxy*(1+padding)),
            ratio=1
        )
        + theme_void()
        + theme(legend_position='right', strip_text=element_text(face='bold'))
    )
    
    if save_to:
        p.save(save_to, dpi=300)
    
    return p