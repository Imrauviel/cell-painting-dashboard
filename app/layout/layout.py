from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2

IMAGE_DIR_PATH = r'../new'
last_point_index_1 = None
last_point_index_2 = None
point_index_in_cache_1 = 0

point_index_in_cache_2 = 0

def get_index(chosen_point):
    return chosen_point['points'][0]['pointIndex'] if chosen_point is not None else None


def gen_umap_fig(point_index_1=0, point_index_2=1):
    fig = go.Figure(data=go.Scatter(x=data['vector1'],
                                    y=data['vector2'],
                                    mode='markers',
                                    marker=dict(size=6,
                                                color='#ff6969'),
                                    text=data['name'],
                                    name='All images'))
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

    x_points = [data['vector1'][point_index_1],
                data['vector1'][point_index_2]]
    y_points = [data['vector2'][point_index_1],
                data['vector2'][point_index_2]]
    point_names = [data['name'][point_index_1],
                   data['name'][point_index_2]]

    fig.add_scatter(x=x_points,
                    y=y_points,
                    mode='markers',
                    text=point_names,
                    name='Selected image',
                    marker=dict(
                        size=9,
                        color="#000000"
                    ))
    return fig


def get_images(point_index_1=0, point_index_2=1):
    file_name_1 = image_names[point_index_1][0]
    img1 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_1, cv2.COLOR_RGB2BGR)

    file_name_2 = image_names[point_index_2][0]
    img2 = cv2.imread(IMAGE_DIR_PATH + '/' + file_name_2, cv2.COLOR_RGB2BGR)
    return img1, img2, file_name_1, file_name_2


def get_figure(image_1, image_2, name_1, name_2):
    fig = px.imshow(np.array([image_1, image_2]), facet_col=0, binary_string=True,
                    facet_col_spacing=0.02, width=800, height=400, labels={'facet_col': 'Image'},
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


def create_options(data):
    options = []
    for index, name in enumerate(data['name']):
        options.append({'label': name,
                        'value': index})
    return options


layout = html.Div([
    html.Div([

    ], className='header', id='app-header', style={'background': 'gray'}),
    html.Div([

    ], className='navbar', id='navbar', style={}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dropdown_image_1',
                options=create_options(data),
                multi=False,
                value=0,
                placeholder="Select first image",
                className='dropdown-image-1'
            ),
            dcc.Dropdown(
                id='dropdown_image_2',
                options=create_options(data),
                multi=False,
                value=1,
                placeholder="Select second image",
                className='dropdown-image-2'
            )
        ], className='dropdowns', style={}),
        html.Div([
            dcc.Graph(id='graph', config={
                "displayModeBar": False,
            }, figure=gen_umap_fig())
        ], className='middle-side', #style={'height': '500px',
                                           #'float': 'left',
                                           #'width': '50%'}
                            ),

    ], className='main_part', style={}),
    html.Div([
        html.Div([
            dcc.Graph(id='image',  config={
                "displayModeBar": False,
            }, figure=get_figure(img1, img2, file_name_1, file_name_2))
        ], className='img-graph')


    ], className='right-side', #style={'float': 'left',
    #                                  'width': '50%',
    #                                  'height': '1000'}
                                     ),
    html.Div([

    ], className='footer')

], className='body', style={
    'background-color': '#ebebeb',
    # 'margin-left': '60px',
    # 'margin-top': '60px'
})
