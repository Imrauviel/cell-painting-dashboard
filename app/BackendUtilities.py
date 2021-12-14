from typing import List, Optional, Dict, Tuple
from dash.dash import Dash
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import cv2
from dash import dcc, html, dash_table

from models.ImageModel import ImageModel


class BackendUtilities(Dash):
    def __init__(self, images: Dict, title: str, path: str, csv_data: pd.DataFrame):
        super().__init__(title=title)
        self.images: Dict = images
        self._image_dir_path: str = path
        self._csv_data: pd.DataFrame = csv_data

    def merge_images(self, values: List[int], image_model: ImageModel) -> np.array:
        result_image = np.zeros((1080, 1080))
        num_of_images = len(values)
        for value in values:
            image_channel = cv2.imread(self._image_dir_path + '/' + image_model.get_channel_image(value),
                                       cv2.COLOR_RGB2BGR)
            result_image += image_channel / num_of_images
        return result_image

    def generate_scatter_figure(self, point_index_1=0, point_index_2=1, color_by_group='None') -> go.Figure:
        if color_by_group != 'None':
            figure = px.scatter(self._csv_data,
                                x='Vector1',
                                y='Vector2',
                                color=color_by_group,
                                hover_name='Name',
                                hover_data={'Compound': True,
                                            'Concentration': True,
                                            'Vector1': False,
                                            'Vector2': False},
                                color_discrete_sequence=px.colors.qualitative.Bold

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
                                color_discrete_sequence=px.colors.qualitative.Bold

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
                        color='#ddff00',
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
                        color='#00ff11',
                        line=dict(width=3,
                                  color='#000000')
                        )
        )
        return figure

    @staticmethod
    def get_image_figure(image_1: np.array, image_2: np.array, name_1: str, name_2: str):
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

    def get_images(self, point_index_1: int = 0, point_index_2: int = 1, values: Optional[List[int]] = None,
                   gamma: int = 0) -> Tuple[np.array, np.array, ImageModel, ImageModel]:
        if values is None:
            values = [1, 2, 3, 4]
        image_model_1 = self.images[point_index_1]
        image_1 = self.merge_images(values, image_model_1)
        image_1 = self._adjust_gamma(image_1, gamma)

        image_model_2 = self.images[point_index_2]
        image_2 = self.merge_images(values, image_model_2)
        image_2 = self._adjust_gamma(image_2, gamma)
        return image_1, image_2, image_model_1, image_model_2

    def create_options(self) -> List[dict]:
        options: List[dict] = []
        for idx_value, image in self.images.items():
            options.append({'label': image.file_name,
                            'value': idx_value})
        return options

    @staticmethod
    def _adjust_gamma(image: np.array, gamma: int = 0) -> np.array:
        if gamma == 0:
            return image
        image = (255 * image).astype("uint8")
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(image, table)
        return result.astype(np.float) / 255

    def get_selected_points_info(self, selected_points: Optional[List[dict]]) -> dash_table.DataTable:
        table = dash_table.DataTable(
            id='selected_points',
            columns=[{"name": i.capitalize(), "id": i} for i in
                     ['file_name', 'index', 'vector_1', 'vector_2', 'row', 'column', 'f', 'well', 'compound',
                      'concentration']],
            export_format="csv",
            sort_action='native',
            # sort_mode='multi',
            page_current=0,
            page_size=10,
            filter_action="native",

        )
        if selected_points:
            # TODO wywalić powtorzenia
            selected_points = selected_points['points']
            objects: List[dict] = [self.images[self.get_index(image)].__dict__ for image in selected_points]
            table.data = objects
        return table

    def get_index(self, chosen_point: dict) -> Optional[int]:
        file_name: str = chosen_point['hovertext'] if chosen_point is not None else None
        if file_name:
            idx = self._csv_data[self._csv_data['Name'] == file_name].index.values.astype(int)[0]
            return idx

    def set_layout(self):
        layout = html.Div([
            # dcc.Loading(id='main_loading',
            #                            type='graph',
            #                            fullscreen=True,
            #                            color='red',
            #                            debug=False,
            #                            children=[
            html.Div([

            ], className='header', id='app-header', style={'background': 'gray'}),
            html.Div([

            ], className='navbar', id='navbar', style={}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='dropdown_image_1',
                        options=self.create_options(),
                        multi=False,
                        value=0,
                        placeholder="Select first image",
                        className='dropdown-image-1'
                    ),
                    dcc.Dropdown(
                        id='dropdown_image_2',
                        options=self.create_options(),
                        multi=False,
                        value=1,
                        placeholder="Select second image",
                        className='dropdown-image-2'
                    )
                ], className='dropdowns', style={}),
                html.Div([
                    dcc.Graph(id='graph', responsive=True)
                ], className='middle-side',
                ),

            ], className='main_part', style={}),
            html.Div([dcc.Loading(id='image_loading',
                                  type='dot',
                                  fullscreen=False,
                                  color='red',
                                  children=[
                                      html.Div([
                                          dcc.Graph(id='image', config={
                                              "displayModeBar": False,
                                          })
                                      ], className='img-graph')])

                      ], className='right-side',
                     ),
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
                    tooltip={"placement": "bottom",
                             "always_visible": False},
                ), dcc.Dropdown(
                    id='dropdown_color_select',
                    options=[
                        {'label': 'None',
                         'value': 'None'},
                        {'label': 'Compound',
                         'value': 'Compound'},
                        {'label': 'Concentration',
                         'value': 'Concentration'},
                    ],
                    multi=False,
                    placeholder="Select group color",

                ),
                html.P(
                    id="image_info"

                ),
                html.Div(id='div_table',
                         children=[])

            ], className='footer'),

            # ])
        ], className='body', style={
            'background-color': '#ebebeb',

        })
        self.layout = layout
