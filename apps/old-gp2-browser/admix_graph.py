import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from geneview.palette._palettes import generate_colors_palette
from geneview.popgene._admixture import _draw_admixtureplot
from geneview.algorithm._cluster import hierarchical_cluster
from itertools import cycle
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable
from hold_data import blob_as_csv, get_gcloud_bucket

st.set_page_config(page_title = "Hidden Page", layout = 'wide')

'''Will update depending on changes needed for Admix Graph. This script does not display to any pages,
only used to create and save Admix Plot in order to avoid loading times. Utilizes functions from Geneview
package by Shujia Huang (https://github.com/ShujiaHuang/geneview)'''

# def generate_colors_palette(cmap="viridis", n_colors=10, alpha=1.0):
#     """Generate colors from matplotlib colormap; pass list to use exact colors"""
#     if isinstance(cmap, list):
#         colors = [list(to_rgba(color, alpha=alpha)) for color in cmap]
#     else:
#         scalar_mappable = ScalarMappable(cmap=cmap)
#         colors = scalar_mappable.to_rgba(range(n_colors), alpha=alpha).tolist()
#     return colors

# def _draw_admixtureplot(
#         data=None,
#         group_order=None,
#         linewidth=1.0,
#         palette="tab10",
#         xticklabels=None,
#         xticklabel_kws=None,
#         ylabel=None,
#         ylabel_kws=None,
#         hierarchical_kws=None,
#         ax=None
# ):
#     """Plot admixture.
#     Parameters
#     ----------
#     data : dict.
#         Input data structure. The key of `data` is the name of sub population and the
#         value of `data` is a pandas.DataFrame which contain the admixture output (.Q)
#         of the specify population. e.g:
#         data = {"sub-pop1": pandas.DataFrame("admixture result(.Q) for 'sub-pop1'"),
#                 "sub-pop2": pandas.DataFrame("admixture result(.Q) for 'sub-pop2'")}
#     group_order : vector of strings, optional
#         Specify the order of processing and plotting for the estimating sub populations.
#     """
#     if ax is None:
#         _, ax = plt.subplots(1, 1, figsize=(14, 2), facecolor="w", constrained_layout=True)

#     if hierarchical_kws is None:
#         hierarchical_kws = {"method": "average", "metric": "euclidean"}
#     if "axis" not in hierarchical_kws:
#         hierarchical_kws["axis"] = 0  # row axis (by sample) to use to calculate cluster.

#     if group_order is None:
#         group_order = list(set(data.keys()))

#     n = 0
#     for g in group_order:
#         n += len(data[g])
#         # hc = hierarchical_cluster(data=data[g], **hierarchical_kws)
#         # data[g] = hc.data.iloc[hc.reordered_index]  # re-order the data.

#     x = np.arange(n)
#     base_y = np.zeros(len(x))

#     column_names = data[group_order[0]].columns

#     # st.write(data)
#     # st.write(column_names)

#     colors = generate_colors_palette(cmap=palette, n_colors=len(column_names))
#     palette = cycle(colors)

#     for k in column_names:
#         c = next(palette)  # one color for one 'k'
#         print(c, k)
#         start_g_pos = 0
#         add_y = []
#         for g in group_order:  # keep group order
#             try:
#                 y = data[g][k]
#                 end_g_pos = start_g_pos + len(y)
#                 ax.bar(x[start_g_pos:end_g_pos], y, bottom=base_y[start_g_pos:end_g_pos],
#                        color=c, width=1.0, linewidth=0)

#                 start_g_pos = end_g_pos
#                 add_y.append(y)
#             except KeyError:
#                 raise KeyError("KeyError: '%s'. Missing in 'group_order'." % g)

#         base_y += np.array(pd.concat(add_y, axis=0))

#     g_pos = 0
#     xticks_pos = []
#     for g in group_order:  # make group order
#         g_size = len(data[g])
#         xticks_pos.append(g_pos + 0.5 * g_size)

#         g_pos += g_size
#         ax.axvline(x=g_pos, color="k", lw=linewidth)

#     ax.spines["left"].set_linewidth(linewidth)
#     ax.spines["top"].set_linewidth(linewidth)
#     ax.spines["right"].set_linewidth(linewidth)
#     ax.spines["bottom"].set_linewidth(linewidth)

#     ax.set_xlim([x[0], x[-1]])
#     ax.set_ylim([0, 1.0])
#     ax.tick_params(bottom=False, top=False, left=False, right=False)

#     if xticklabel_kws is None:
#         xticklabel_kws = {}

#     if xticklabels and (len(group_order) != len(xticklabels)):
#         raise ValueError("The size of 'xticklabels'(%d) is not the same with "
#                          "'group_order'(%d)." % (len(xticklabels), len(group_order)))

#     ax.set_xticks(xticks_pos)
#     ax.set_xticklabels(group_order if xticklabels is None else xticklabels,
#                        **xticklabel_kws)

#     if ylabel_kws is None:
#         ylabel_kws = {}

#     ax.set_ylabel(ylabel, **ylabel_kws)
#     ax.set_yticks([])

#     # ax.legend(column_names)

#     return ax

ref_admix = pd.read_csv('data/ref_panel_admixture_9.txt', sep='\s+')
# ref_panel_bucket_name = 'ref_panel'
# ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)
# ref_admix = blob_as_csv(ref_panel_bucket, 'ref_panel_admixture.txt')

st.dataframe(ref_admix.head())


admix_pop_info = ref_admix['ancestry']
st.dataframe(admix_pop_info.head())
# admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10']]
admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9']]
st.dataframe(admixture_output)

df = admixture_output
sample_info = ref_admix
popset = set(ref_admix['ancestry'])

data = {}
for g in popset:
    g_data = df[sample_info['ancestry'] == g].copy()
    # g_size = len(g_data)
    data[g] = g_data


f, ax = plt.subplots(1, 1, figsize=(14, 2), facecolor="w", constrained_layout=True, dpi=300)
ax = _draw_admixtureplot(data=data, ax=ax)

st.pyplot(f)

f.savefig('data/refpanel_admix_9.png')