import pandas as pd
import glob
import plotly.express as px

# !gcloud storage cp -r gs://gp2_working_eu/nicole/GBA1_metrics/ data/


def plot_clusters(df, x_col='theta', y_col='r', gtype_col='GT', title='SNP Plot', output_html=None):
    d3 = px.colors.qualitative.D3
    cmap = {'AA': d3[0], 'AB': d3[1], 'BB': d3[2], 'NC': d3[3]}
    smap = {'Control': 'circle', 'PD': 'diamond-open-dot'}

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=gtype_col,
        color_discrete_map=cmap,
        # symbol='phenotype',
        symbol_map=smap,
        title=title,
        width=650,
        height=497,
        labels={'r': 'R', 'theta': 'Theta'}
    )
    fig.update_layout(
        margin=dict(r=76, t=63, b=75),
        legend_title_text='Genotype'
    )

    if output_html is not None:
        fig.write_html(output_html)

    return fig

coord_path = 'data/metrics_coords.csv'
metrics_paths = glob.glob('data/GBA1_metrics/*')

coord_df = pd.DataFrame()

for metrics_path in metrics_paths:
    sm_df = pd.read_csv(metrics_path, dtype={'chromosome':str,'position':str})
    sm_df.loc[:,'variant_id'] = sm_df['chromosome'] + '_' + sm_df['position'] + '_' + sm_df['Ref'] + '_' + sm_df['Alt']

    sm_sub = sm_df.loc[0:0,:]
    sm_sub = sm_sub[['variant_id','snpID','chromosome','position','Ref','Alt']]
    
    variant_id = sm_sub['variant_id'][0]
    # plot clusters and write to html
    plot_clusters(sm_df, x_col='Theta', y_col='R', gtype_col='GT', title=f'{variant_id} Plot', output_html=f'data/plots/{variant_id}.html')
    
    coord_df = pd.concat([coord_df, sm_sub], ignore_index=True)

coord_df.loc[:,'variant_id'] = coord_df['chromosome'] + '_' + coord_df['position'] + '_' + coord_df['Ref'] + '_' + coord_df['Alt']
coord_df.to_csv(coord_path, header=True, index=False)

