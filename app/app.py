import dash
from dash.dependencies import Input, Output
import cv2
from layout.layout import layout, gen_sample_fig
import plotly.express as px
import pandas as pd
import numpy as np
lastPoint = None
temp = None
app = dash.Dash(__name__)
app.title = 'Cell app'
app.layout = layout
data = pd.read_csv(r'../data/features.csv')[['name']].values.tolist()
IMAGE_DIR_PATH = r'../resized_merged_images'
@app.callback(
    Output('image', 'figure'),
    Input('graph', 'clickData')
)
def update_text(choosenPoint):
    global lastPoint, temp
    print('here')
    if choosenPoint:
        print(choosenPoint['points'][0]['pointIndex'])
        if temp is None:
            temp = choosenPoint
            ids = choosenPoint['points'][0]['pointIndex']
            path = data[ids]

            img = cv2.imread(IMAGE_DIR_PATH+'/'+path[0], cv2.IMREAD_GRAYSCALE)
            fig = px.imshow(img, binary_string=True)
            return fig
        else:
            lastPoint = temp
            temp = choosenPoint
            ids1 = lastPoint['points'][0]['pointIndex']
            path1 = data[ids1]
            img1 = cv2.imread(IMAGE_DIR_PATH+'/'+path1[0], cv2.IMREAD_GRAYSCALE)

            ids2 = choosenPoint['points'][0]['pointIndex']
            path2 = data[ids2]
            img2 = cv2.imread(IMAGE_DIR_PATH+'/'+path2[0], cv2.IMREAD_GRAYSCALE)

            fig = px.imshow(np.array([img2, img1]), facet_col=0, binary_string=True, facet_col_wrap=1,
                            facet_row_spacing=0.35, width=700, height=1000)
            return fig

    #             return (str(lastPoint['points'][0]['pointIndex'])+'\n'
    #             + str(choosenPoint['points'][0]['pointIndex']))
    return gen_sample_fig()


if __name__ == '__main__':
    app.run_server(port=1111, dev_tools_ui=True, debug=True,
                   dev_tools_hot_reload=True, threaded=True)
