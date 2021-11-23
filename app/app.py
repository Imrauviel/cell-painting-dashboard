import dash
from dash.dependencies import Input, Output

from layout.layout import layout, get_figure, get_images, gen_umap_fig, get_index

last_point = 0
temp = 1
app = dash.Dash(__name__)
app.title = 'Cell app'

IMAGE_DIR_PATH = r'../resized_merged_images'
first_image_cache = 0
second_image_cache = 1
chosen_index_cache = None


@app.callback([
    Output('image', 'figure'),
    Output('graph', 'figure'),
    Output('dropdown_image_1', 'value'),
    Output('dropdown_image_2', 'value')
]
    ,
    [Input('graph', 'clickData'),
     Input('dropdown_image_1', 'value'),
     Input('dropdown_image_2', 'value')]
)
def update_by_scatter(chosen_point, drop_1, drop_2):
    global second_image_cache, first_image_cache, chosen_index_cache
    chosen_point_index = get_index(chosen_point)
    if first_image_cache != chosen_point_index and \
            chosen_point_index is not None and \
            chosen_index_cache != chosen_point_index:
        chosen_index_cache = chosen_point_index
        second_image_cache = first_image_cache
        first_image_cache = chosen_point_index

        img1, img2, file_name_1, file_name_2 = get_images(first_image_cache, second_image_cache)
        fig = get_figure(img1, img2, file_name_1, file_name_2)
        scatter_plot = gen_umap_fig(first_image_cache, second_image_cache)
        return fig, scatter_plot, first_image_cache, second_image_cache
    second_image_cache = drop_2
    first_image_cache = drop_1
    img1, img2, file_name_1, file_name_2 = get_images(drop_2, drop_1)
    fig = get_figure(img2, img1, file_name_2, file_name_1)
    scatter_plot = gen_umap_fig(drop_1, drop_2)
    return fig, scatter_plot, drop_1, drop_2


app.layout = layout
if __name__ == '__main__':
    app.run_server(port=1111, dev_tools_ui=True, debug=True,
                   dev_tools_hot_reload=True, threaded=True)
