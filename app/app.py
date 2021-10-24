import dash

from layout.layout import content

app = dash.Dash(__name__)
app.title = 'Cell app'
app.layout = content

if __name__ == '__main__':
    app.run_server(debug=True, port=1111)
