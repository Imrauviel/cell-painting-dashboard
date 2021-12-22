import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2

IMAGE_DIR_PATH = r'../processed/HepG2_Exp3_Plate1_FX9__2021-04-08T16_16_48'



#
# def get_index(chosen_point: dict):
#     return chosen_point['points'][0]['pointIndex'] if chosen_point is not None else None
#
#
# def merge_images(values, filename):
#     img = np.zeros((1080, 1080))
#     num_of_images = len(values)
#     for value in values:
#         channel = f'-ch{str(value)}s'
#         new = filename.replace('-s', channel)
#         image_channel = cv2.imread(IMAGE_DIR_PATH + '/' + new, cv2.COLOR_RGB2BGR)
#         img += image_channel / num_of_images
#     return img
#
#
# def adjust_gamma(image, gamma=0):
#     if gamma == 0:
#         return image
#     image = (255 * image).astype("uint8")
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     result = cv2.LUT(image, table)
#     return (result.astype(np.float)) / 255
#
#
# def gen_umap_fig(point_index_1=0, point_index_2=1):
#     fig = go.Figure(data=go.Scatter(x=data['vector1'],
#                                     y=data['vector2'],
#                                     mode='markers',
#                                     marker=dict(size=6,
#                                                 color='#ff6969'),
#                                     text=data['name'],
#                                     name='All images'))
#     fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#     fig.update_layout(
#         plot_bgcolor='#f2f2f2',
#         font_family='Courier New',
#         paper_bgcolor='#f2f2f2',
#         xaxis=dict(showgrid=False,
#                    showline=False,
#                    zeroline=False),
#         yaxis=dict(showgrid=False,
#                    showline=False,
#                    zeroline=False),
#         margin=dict(l=0, r=0, b=0, t=0),
#     )
#
#     x_points = [data['vector1'][point_index_1],
#                 data['vector1'][point_index_2]]
#     y_points = [data['vector2'][point_index_1],
#                 data['vector2'][point_index_2]]
#     point_names = [data['name'][point_index_1],
#                    data['name'][point_index_2]]
#
#     fig.add_scatter(x=x_points,
#                     y=y_points,
#                     mode='markers',
#                     text=point_names,
#                     name='Selected image',
#                     marker=dict(
#                         size=9,
#                         color="#000000"
#                     ))
#     return fig
#
#
# def get_images(point_index_1=0, point_index_2=1, values=None, gamma=0):
#     if values is None:
#         values = [1, 2, 3, 4]
#     file_name_1 = image_names[point_index_1][0]
#     img1 = merge_images(values, file_name_1)
#     img1 = adjust_gamma(img1, gamma)
#
#     file_name_2 = image_names[point_index_2][0]
#     img2 = merge_images(values, file_name_2)
#     img2 = adjust_gamma(img2, gamma)
#     return img1, img2, file_name_1, file_name_2
#
#
# def get_info(file_name):
#     info = data.loc[data['name'] == file_name]
#     for i, y in data.iterrows():
#         x = i
#         z = 1
#     return f'Row: {info["Row"]}'
#
#
# def get_figure(image_1, image_2, name_1, name_2):
#     fig = px.imshow(np.array([image_1, image_2]), facet_col=0, binary_string=True,
#                     facet_col_spacing=0.02, width=800, height=400, labels={'facet_col': 'Image'},
#                     )
#     fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#     fig.update_traces(hovertemplate=None, hoverinfo='skip')
#     fig.layout.annotations[0]['text'] = f'Image: {name_1}'
#     fig.layout.annotations[1]['text'] = f'Image: {name_2}'
#     fig.update_layout(
#         plot_bgcolor='#f2f2f2',
#         font_family='Courier New',
#         paper_bgcolor='#f2f2f2',
#         margin=dict(l=0, r=0, b=0, t=20))
#     return fig
#
#
# data = pd.read_csv(r'../data/features.csv')
# image_names = data[['name']].values.tolist()
#
# img1, img2, file_name_1, file_name_2 = get_images()

#
# def create_options(data):
#     options = []
#     for index, name in enumerate(data['name']):
#         options.append({'label': name,
#                         'value': index})
#     return options


layout = html.Div([
    html.Div([

    ], className='header', id='app-header', style={'background': 'gray'}),
    html.Div([

    ], className='navbar', id='navbar', style={}),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='graph', config={
                        # "displayModeBar": False,
                    })
                ], className='middle-side',),], md=12, lg=6),
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id='dropdown_image_1',
                        # options=create_options(data),
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
                ], className='dropdowns', style={}),], md=12, lg=6),
                html.Div([
                    html.Div([
                        dcc.Graph(id='image', config={
                            "displayModeBar": False,
                        })
                    ], className='img-graph')
                ])
            # dbc.Col([], md=6, lg=3),
        ]),
    ], className='main_part', style={}),

    html.Div([dcc.Checklist(
        id='channel_list',
        options=[
            {'label': 'Chanel 1',
             'value': 1},
            {'label': 'Chanel 2',
             'value': 2},
            {'label': 'Chanel 3',
             'value': 3},
            {'label': 'Chanel 4',
             'value': 4}
        ],
        value=[1, 2, 3, 4],
        labelStyle={'display': 'inline-block'}
    ),
        dcc.Slider(
            id='gamma_slider',
            min=-0.5,
            max=0.5,
            step=0.005,
            value=0,
            updatemode='drag',
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.P(
            id="image_info"
        )

    ], className='footer'),

], className='body', style={
    'background-color': '#ebebeb',

})
