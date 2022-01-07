import pickle
from typing import List, Optional, Dict, Tuple, Set

from dash import dash_table, html, dcc
from dash.dash import Dash
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering

import dash_bootstrap_components as dbc
from PIL import Image

from models.ImageModel import ImageModel

FEATURES_PKL = r'../data/features_dict.pkl'


class BackendUtilities(Dash):
    def __init__(self, images: Dict, title: str, path: str, csv_data: pd.DataFrame, external_stylesheets):
        super().__init__(title=title, external_stylesheets=external_stylesheets)
        self.images: Dict = images
        self._image_dir_path: str = path
        self._csv_data: pd.DataFrame = csv_data
        self._csv_data['Concentration'] = self._csv_data['Concentration'].astype(str)
        self._features: Optional[dict] = None

    def merge_images2(self, values: List[int], image_model: ImageModel) -> np.array:
        channels = []
        for value in values:
            channels.append(
                Image.fromarray(cv2.imread(self._image_dir_path + '/' + image_model.get_channel_image(value),
                                           cv2.IMREAD_GRAYSCALE)))
        for i in [1, 2, 3, 4]:
            if i not in values:
                channels.append(Image.fromarray(np.ones((1080, 1080), dtype=np.uint8)))

        result_image = Image.merge("CMYK", (channels[0], channels[1], channels[2], channels[3]))
        result_image = np.array(result_image.convert('RGB'))
        return result_image

    def generate_scatter_figure(self, point_index_1=0, point_index_2=1, color_by_group='None') -> go.Figure:
        if color_by_group != 'None':
            if isinstance(color_by_group, int):
                color_by_group = self._get_clusters(color_by_group)
            figure = px.scatter(self._csv_data,
                                x='Vector1',
                                y='Vector2',
                                color=color_by_group,
                                hover_name='Name',
                                hover_data={'Compound': True,
                                            'Concentration': True,
                                            'Vector1': False,
                                            'Vector2': False},
                                color_discrete_sequence=px.colors.qualitative.Bold,
                                )
        else:
            figure = px.scatter(self._csv_data,
                                x='Vector1',
                                y='Vector2',
                                hover_name='Name',
                                hover_data={'Compound': True,
                                            'Concentration': True,
                                            'Vector1': False,
                                            'Vector2': False},
                                color_discrete_sequence=px.colors.qualitative.Bold,

                                )
        figure.update_xaxes(showticklabels=False, visible=False).update_yaxes(showticklabels=False, visible=False)
        figure.update_layout(
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

        figure.add_scatter(
            x=[self._csv_data['Vector1'][point_index_1]],
            y=[self._csv_data['Vector2'][point_index_1]],
            mode='markers',
            hovertext=[self._csv_data['Name'][point_index_2]],
            name='Image 1',
            marker=dict(size=12,
                        color='#ffffff',
                        line=dict(width=3,
                                  color='#000000')
                        )
        )
        figure.add_scatter(
            x=[self._csv_data['Vector1'][point_index_2]],
            y=[self._csv_data['Vector2'][point_index_2]],
            mode='markers',
            hovertext=[self._csv_data['Name'][point_index_2]],
            name='Image 2',
            marker=dict(size=12,
                        color='#f7ff0a',
                        line=dict(width=3,
                                  color='#000000')
                        )
        )
        return figure

    @staticmethod
    def get_image_figure(image_1: np.array, image_2: np.array, name_1: str, name_2: str):
        fig = px.imshow(np.array([image_1, image_2]), facet_col=0,  # binary_string=True,
                        facet_col_spacing=0.02, width=800, height=400,
                        )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_traces(hovertemplate=None, hoverinfo='skip')
        fig.layout.annotations[0]['text'] = ""
        fig.layout.annotations[1]['text'] = ""
        fig.update_layout(
            plot_bgcolor='#f2f2f2',
            font_family='Courier New',
            paper_bgcolor='#f2f2f2',
            margin=dict(l=0, r=0, b=0, t=20))
        return fig

    def get_images(self, point_index_1: int = 0, point_index_2: int = 1, values: Optional[List[int]] = None,
                   gamma: int = 0) -> Tuple[np.array, np.array, ImageModel, ImageModel]:
        if values is None:
            values = [1, 2, 3, 4]
        image_model_1 = self.images[point_index_1]
        image_1 = self.merge_images2(values, image_model_1)
        image_1 = self._adjust_gamma(image_1, gamma)

        image_model_2 = self.images[point_index_2]
        image_2 = self.merge_images2(values, image_model_2)
        image_2 = self._adjust_gamma(image_2, gamma)
        return image_1, image_2, image_model_1, image_model_2

    def create_options(self) -> List[dict]:
        options: List[dict] = []
        for idx_value, image in self.images.items():
            options.append({'label': image.file_name,
                            'value': idx_value})
        return options

    def _get_clusters(self, n_clusters: int) -> Optional[List[str]]:
        if not self._features:
            file = open(FEATURES_PKL, "rb")
            self._features = pickle.load(file)
            file.close()
        if n_clusters:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            results = model.fit(X=[i.tolist()[0] for i in self._features.values()])
            return [str(i) for i in results.labels_]

    @staticmethod
    def _adjust_gamma(image: np.array, gamma: int = 0) -> np.array:
        if gamma == 0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_selected_points_info(self, selected_points: Optional[List[dict]]) -> dash_table.DataTable:
        table = dash_table.DataTable(
            id='selected_points',
            columns=[{"name": i.capitalize(), "id": i} for i in
                     ['file_name', 'index', 'vector_1', 'vector_2', 'row', 'column', 'f', 'well', 'compound',
                      'concentration']],
            export_format="csv",
            sort_action='native',
        )
        if selected_points:
            selected_points = selected_points['points']
            objects: List[dict] = [self.images[self.get_index(image)].__dict__ for image in selected_points]
            names_of_objects: Set[str] = list(set(image['file_name'] for image in objects))
            result: List[dict] = []
            for obj in objects:
                if obj['file_name'] in names_of_objects:
                    result.append(obj)
                    names_of_objects.remove(obj['file_name'])
            table.data = result
        return table

    def get_index(self, chosen_point: dict) -> Optional[int]:
        file_name: str = chosen_point['hovertext'] if chosen_point is not None else None
        if file_name:
            idx = self._csv_data[self._csv_data['Name'] == file_name].index.values.astype(int)[0]
            return idx

    def set_layout(self):
        layout = html.Div([

            html.Div([

            ], className='header', id='app-header', style={'background': 'gray'}),
            html.Div([

            ], className='navbar', id='navbar', style={}),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            dcc.Dropdown(
                                id='dropdown_color_select',
                                options=[
                                    {'label': 'None',
                                     'value': 'None'},
                                    {'label': 'Compound',
                                     'value': 'Compound'},
                                    {'label': 'Concentration',
                                     'value': 'Concentration'},
                                    {'label': '2 Clusters',
                                     'value': 2},
                                    {'label': '3 Clusters',
                                     'value': 3},
                                    {'label': '4 Clusters',
                                     'value': 4},
                                    {'label': '5 Clusters',
                                     'value': 5},
                                    {'label': '6 Clusters',
                                     'value': 6},
                                    {'label': '7 Clusters',
                                     'value': 7},
                                    {'label': '8 Clusters',
                                     'value': 8},
                                ],
                                multi=False,
                                placeholder="Select group color",

                            ), className='dropdown-wrapper',
                        ),

                        dcc.Graph(id='graph',
                                  config={"displaylogo": False,
                                          },
                                  )

                    ], width=6, lg=6, md=12),

                    dbc.Col([
                        html.Div([
                            dcc.Dropdown(
                                id='dropdown_image_1',
                                options=self.create_options(),
                                multi=False,
                                value=0,
                                placeholder="Select first image",
                                className='dropdown-image-1',
                                clearable=False
                            ),
                            dcc.Dropdown(
                                id='dropdown_image_2',
                                options=self.create_options(),
                                multi=False,
                                value=1,
                                placeholder="Select second image",
                                className='dropdown-image-2',
                                clearable=False
                            )
                        ], className='dropdowns', style={}),
                        html.Div([
                            dcc.Loading(id='image_loading',
                                        type='dot',
                                        fullscreen=False,
                                        color='red',
                                        children=[
                                            html.Div([
                                                dcc.Graph(id='image', config={
                                                    "displayModeBar": False,
                                                })
                                            ], className='img-graph')
                                        ])
                        ]),
                        html.Div([
                            html.P(
                                id="image_info",
                                className='image_info'
                            ),
                            html.P(
                                'some text',
                                id="image_info2",
                                className='image_info'
                            ),
                            # Info about second
                        ], className='info-wrapper'),
                        html.Div([
                            # title gamma correction
                            dcc.Slider(
                                id='gamma_slider',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.5,
                                updatemode='mouseup',  # optional drag
                                tooltip={"placement": "right",
                                         "always_visible": True},
                            ),
                            # title VISABLE CHannells
                            dcc.Checklist(
                                id='channel_list',
                                options=[
                                    {'label': ' Cell Nucleus',
                                     'value': 1},
                                    {'label': ' Endoplasmic reticulum and Nucleolus',
                                     'value': 2},
                                    {'label': ' Cytoskeleton, Golgi apparatus and membrane',
                                     'value': 3},
                                    {'label': ' Mitochondrion',
                                     'value': 4},
                                ],
                                value=[1, 2, 3, 4],
                                labelStyle={'display': 'block'}
                            ),
                        ], className='image-correction'),
                    ], width=6, lg=6, md=12),
                ]),
                html.Div(id='div_table',
                         children=[])
            ], className='footer'),
        ], className='body', style={
            'background-color': '#ebebeb',

        })
        self.layout = layout
