import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import numpy as np
import pandas as pd
import os

def gen_umap_fig(data: np.array):
    fig = px.scatter(x=data[:, 0], y=data[:, 1])
    return fig


def gen_sample_fig():
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    return fig

print(os.getcwd())
data = pd.read_csv(r'../data/features.csv')[['vector1', 'vector2']].to_numpy()
layout = html.Div([
    html.Div([  # header

    ], className='header', id='app-header', style={'height': '100px'}),
    html.Div([  # navbar

    ], className='navbar', id='navbar', style={'height': '100px'}),
    html.Div([  # mainpart
        #         html.Div([
        # #             dcc.Dropdown()
        #         ], className='left_side', style={'height': '600px', 'float':'left', 'width':'20%'}),
        html.Div([
            dcc.Graph(id='graph', figure=gen_umap_fig(data))
        ], className='middle_side', style={'height': '600px', 'float': 'left', 'width': '60%'}),
        html.Div([
            #             dcc.Textarea(
            #             id='textarea',
            #             value='nothing',
            #             )
            dcc.Graph(id='image', figure=gen_sample_fig())

        ], className='rightside', style={'height': '600px', 'float': 'left', 'width': '40%', 'padding': '-50px'})

    ], className='main_part', style={'background-color': 'green'}),
    html.Div([  # footer

    ], className='footer')

], className='body')
