import dash
from dash.dependencies import Input, Output

from layout.layout import layout, get_figure, get_images
import plotly.express as px
import pandas as pd
import numpy as np

last_point = 0
temp = 1
app = dash.Dash(__name__)
app.title = 'Cell app'
app.layout = layout

IMAGE_DIR_PATH = r'../resized_merged_images'


@app.callback(
    Output('image', 'figure')
    ,
    [Input('graph', 'clickData')]
)
def update_text(choosen_point):
    img1, img2, file_name_1, file_name_2 = get_images(choosen_point)
    fig = get_figure(img2, img1, file_name_2, file_name_1)
    return fig


# return gen_sample_fig()


if __name__ == '__main__':
    app.run_server(port=1111, dev_tools_ui=True, debug=True,
                   dev_tools_hot_reload=True, threaded=True)
