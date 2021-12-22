from dash.dependencies import Input, Output
import pandas as pd
from BackendUtilities import BackendUtilities
from models.ImageModel import ImageModel
import dash_bootstrap_components as dbc

IMAGES = dict()
IMAGE_DIR_PATH = r'..\HepG2_Exp3_Plate1_FX9__2021-04-08T16_16_48'
csv_data = pd.read_csv(r'../data/features.csv')
for idx, image_info in csv_data.iterrows():
    # TODO dodać index
    IMAGES[idx] = ImageModel(idx, image_info)

app = BackendUtilities(IMAGES, 'Cell app', IMAGE_DIR_PATH, csv_data, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.set_layout()
last_point = 0
temp = 1

first_image_cache = 0
second_image_cache = 1
chosen_index_cache = None


@app.callback([
    Output('div_table', 'children')],
    [Input('graph', 'selectedData')
     ]
)
def get_selected_images(selected_points):
    table = app.get_selected_points_info(selected_points)
    return [table]


@app.callback([
    Output('image', 'figure'),
    Output('graph', 'figure'),
    Output('dropdown_image_1', 'value'),
    Output('dropdown_image_2', 'value'),
    Output('image_info', 'children'),
]
    ,
    [Input('graph', 'clickData'),
     Input('dropdown_image_1', 'value'),
     Input('dropdown_image_2', 'value'),
     Input('channel_list', 'value'),
     Input('gamma_slider', 'value'),
     Input('dropdown_color_select', 'value'),
     ]
)
def update_by_scatter(chosen_point, drop_1, drop_2, values, gamma, color_group):
    global second_image_cache, first_image_cache, chosen_index_cache
    chosen_point_index = app.get_index(chosen_point['points'][0]) if chosen_point is not None else app.get_index(None)
    if first_image_cache != chosen_point_index and \
            chosen_point_index is not None and \
            chosen_index_cache != chosen_point_index:
        chosen_index_cache = chosen_point_index
        second_image_cache = first_image_cache
        first_image_cache = chosen_point_index

        img1, img2, image_model_1, image_model_2 = app.get_images(first_image_cache, second_image_cache, values, gamma)
        fig = app.get_image_figure(img1, img2, image_model_1.file_name, image_model_2.file_name)

        scatter_plot = app.generate_scatter_figure(first_image_cache, second_image_cache, color_group)
        return fig, scatter_plot, first_image_cache, second_image_cache, str(image_model_1),
    second_image_cache = drop_2
    first_image_cache = drop_1
    img1, img2, image_model_1, image_model_2 = app.get_images(drop_2, drop_1, values, gamma)
    fig = app.get_image_figure(img2, img1, image_model_2.file_name, image_model_1.file_name)
    scatter_plot = app.generate_scatter_figure(drop_1, drop_2, color_group)
    return fig, scatter_plot, drop_1, drop_2, str(image_model_1)


if __name__ == '__main__':
    app.run_server(port=1111, debug=True,
                   dev_tools_hot_reload=True, threaded=True)
