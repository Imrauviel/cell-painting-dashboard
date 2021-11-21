from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2

IMAGE_DIR_PATH = r'../resized_merged_images'
last_point = None
temp = {'points': [{'pointIndex': 0}]}


def gen_umap_fig(data: pd.DataFrame):
    fig = go.Figure(data=go.Scatter(x=data['vector1'],
                                    y=data['vector2'],
                                    mode='markers',
                                    text=data['name']))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(
        plot_bgcolor='#f2f2f2',
        font_family='Courier New',
        paper_bgcolor='#f2f2f2',
        xaxis=dict(showgrid=False,
                   showline=False,
                   zeroline=False),
        yaxis=dict(showgrid=False,
                   showline=False,
                   zeroline=False),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.update_traces(marker=dict(size=5,
                                  color='#ffd359'))
    return fig


def get_images(choosen_point=None):
    if choosen_point is None:
        file_name_1 = image_names[0][0]
        img1 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_1, cv2.IMREAD_GRAYSCALE)
        file_name_2 = image_names[1][0]
        img2 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_2, cv2.IMREAD_GRAYSCALE)
        return img1, img2, file_name_1, file_name_2
    global last_point, temp
    last_point = temp
    temp = choosen_point
    ids1 = last_point['points'][0]['pointIndex']
    file_name_1 = image_names[ids1][0]
    img1 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_1, cv2.IMREAD_GRAYSCALE)

    ids2 = choosen_point['points'][0]['pointIndex']
    file_name_2 = image_names[ids2][0]
    img2 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_2, cv2.IMREAD_GRAYSCALE)
    return img1, img2, file_name_1, file_name_2


def get_figure(image_1, image_2, name_1, name_2):
    fig = px.imshow(np.array([image_1, image_2]), facet_col=0, binary_string=True,
                    facet_col_spacing=0.02, width=800, height=800, labels={'facet_col': 'Image'},
                    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_traces(hovertemplate=None, hoverinfo='skip')
    fig.layout.annotations[0]['text'] = f'Image: {name_1}'
    fig.layout.annotations[1]['text'] = f'Image: {name_2}'
    fig.update_layout(
        plot_bgcolor='#f2f2f2',
        font_family='Courier New',
        paper_bgcolor='#f2f2f2',
        margin=dict(l=0, r=0, b=0, t=20))
    return fig


data = pd.read_csv(r'../data/features.csv')
image_names = data[['name']].values.tolist()
img1, img2, file_name_1, file_name_2 = get_images()
layout = html.Div([
    html.Div([

    ], className='header', id='app-header', style={'background': 'gray'}),
    html.Div([

    ], className='navbar', id='navbar', style={}),
    html.Div([
        html.Div([
            dcc.Graph(id='graph', figure=gen_umap_fig(data))
        ], className='middle_side', style={'height': '500px',
                                           'float': 'left',
                                           'width': '50%'}),

    ], className='main_part', style={}),
    html.Div([

        dcc.Graph(id='image', figure=get_figure(img1, img2, file_name_1, file_name_2))

    ], className='rightside', style={'float': 'left',
                                     'width': '50%',
                                     'height': '1000'}),
    html.Div([

    ], className='footer')

], className='body', style={
    'background-color': '#ebebeb',
    'margin-left': '60px',
    'margin-top': '60px'
})
