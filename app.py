from argparse import ArgumentParser
import plotly.graph_objs as go
import json
import dash
import plotly
import plotly.subplots
import plotly.graph_objs.layout as layout
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import flask
import pandas as pd
import time
import pathlib
import os

from tools.files import list_files
from tools.files import sorter
from tools.files import conditional_makedir
from tools.files import save_fiducials
from tools.files import load_data

from dash.dependencies import Output, Input, State
from textwrap import dedent as d
from datetime import datetime
from datetime import date
import os
import os.path
import numpy as np
import plotly.graph_objs as go

"""
Determine file paths
"""

# Define paths
paths = {'inputs' : '', 'output' : ''}
selected_pts = dict()

##################### DEFINE SERVER #####################
server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

##################### DEFINE APP #####################
app = dash.Dash(__name__, server=server, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.scripts.config.serve_locally = False
# dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

##################### DEFINE FIGURE #####################
TemplateHeader = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'SELECT']
signal = pd.DataFrame({TemplateHeader[i] : np.zeros(1000,) for i in range(len(TemplateHeader))})

fig = plotly.subplots.make_subplots(rows=13, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.01)
for i in range(len(TemplateHeader)):
    fig.append_trace({'x':np.arange(len(signal[TemplateHeader[i]])),'y': signal[TemplateHeader[i]],'name': TemplateHeader[i]},i+1,1)

fig['layout']['xaxis'].update(title='', range=[0, 500], autorange=False)
fig.update_traces(xaxis="x13")
fig.layout.xaxis13.spikemode = "across"
fig.layout.xaxis13.spikethickness = 1
fig.layout.xaxis13.rangeslider.visible = True
fig.update_traces(line=dict(width=1.5),)

##################### DEFINE LAYOUT #####################
app.layout = html.Div([
    # dcc.Store(id='button-storage', storage_type='local'),
    html.H1('TV QRS marker'),
    # html.H2('Select input path'),
    # html.Div([
    #     html.Div([
    #         dcc.Input(
    #             id='path',
    #             size=200,
    #         ),
    #     ], className="ten columns"),
    #     html.Div([
    #         html.Button('Select Folder', id='selectfolder'),
    #     ], className="two columns")
    # ]),
    html.H2('Select file'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[]
    ),
    html.Div([
        html.Div([
            dcc.Graph(
                id='basic-interactions',
                style={
                    'height': 1000,
                },
                figure={
                    'layout': {
                        'clickmode': 'event+select'
                    }
                },
            ),
        ], className="eight columns"),
        html.Div([
            html.H2('Control pannel'),
            html.Button('Delete Last element', id='button'),
            html.Button('Save Data', id='save'),
            html.Button('Plot Lines', id='plot_lines'),
            html.H2('Click Data'),
            dcc.Markdown(d("""Click on points in the graph.""")),
            html.Div(id='container-button-basic', children='-'),
            # html.Div([
            #     dash_table.DataTable(
            #         id='editable_table',
            #         columns=([{'id': 'fiducial', 'name': 'Fiducials'}]),
            #         data=[{"fiducial": 0}],
            #         editable=True
            #     ),
            # ]),
            html.Pre(id='click-data', style={'border': 'thin lightgrey solid','overflowX': 'scroll'}),
            dcc.Input(id="from", type="text", placeholder="from"),
            dcc.Input(id="to", type="text", placeholder="to"),
            html.Button('Rescale View', id='plot'),
        ], className="four columns"),
    ]),
], className="row")


##################### DEFINE CALLBACKS #####################
def get_lines(array: list or np.ndarray):
    return [{
        'type':'line', 'line':dict(color="#d62728", width=1.),
        'yref':'paper', 'y0':0, 'y1':1, 
        'xref':'x', 'x0':array[i], 'x1':array[i],
    } for i in range(len(array))]

@app.callback(
    Output('basic-interactions', 'figure'),
    # Output('click-data', 'children')],
    [Input('plot', 'n_clicks'),
    Input('plot_lines', 'n_clicks'),
    Input('my-dropdown', 'value'),], 
    [State("from", "value"), 
    State("to", "value"),])
def update_graph(plot_button, plot_lines, selected_dropdown_value, from_value, to_value): # , clicks_data, clickData
    # Keep track of what triggered whatever
    execution_instruction = dash.callback_context.triggered[0]['prop_id']

    # Output default
    js = []
    # Should always be active to make annotations
    if (selected_dropdown_value is not None):
        r, fn = os.path.split(selected_dropdown_value)
        code = fn.split('.')[0]
        if ('my-dropdown.value' in execution_instruction):
            signal = pd.read_csv(selected_dropdown_value, index_col=0)
            signal.columns = map(str.upper, signal.columns)
            for i in range(len(TemplateHeader)):
                name = fig.data[i].name
                loc = np.where(TemplateHeader == name)[0]
                if name == 'SELECT':
                    fig.data[i].y = np.zeros((signal.shape[0],))
                    fig.data[i].x = np.arange(len(fig.data[i].y))
                else:
                    fig.data[i].y = signal[name.upper()]
                    fig.data[i].x = np.arange(len(fig.data[i].y))
            
            # Set X range
            if signal.shape[0] == 60001:
                fig.layout.xaxis13.range = [0, 6000]
            elif signal.shape[0] == 120000:
                fig.layout.xaxis13.range = [0, 10000]
            else:
                if signal.shape[0] > 10000:
                    fig.layout.xaxis13.range = [0, 6000]
                else:
                    fig.layout.xaxis13.range = [0, signal.shape[0]]

            # Set lines
            shapes = get_lines(selected_pts[code])
            fig.layout.shapes = tuple([layout.Shape(**shapes[i]) for i in range(len(shapes))])
        elif ('plot_lines.n_clicks' in execution_instruction):
            # Add vertical lines
            shapes = get_lines(selected_pts[code])
            fig.layout.shapes = tuple([layout.Shape(**shapes[i]) for i in range(len(shapes))])
        elif ('plot.n_clicks' in execution_instruction):
            from_value = int(float(from_value)*1000)
            to_value = int(float(to_value)*1000)

            for j in range(13):
                min_val = int(np.min(fig.data[j]['y'][from_value:to_value]))
                max_val = int(np.max(fig.data[j]['y'][from_value:to_value]))
                
                if j == 0:
                    fig.layout['yaxis'].range = [min_val,max_val]
                else:
                    fig.layout['yaxis'+str(j+1)].range = [min_val,max_val]

    return fig
        

@app.callback(
    Output('click-data', 'children'),
    [Input('button', 'n_clicks'),
    Input('basic-interactions', 'clickData'),
    Input('my-dropdown', 'value')])
def display_click_data(delete_button, clickData, selected_dropdown_value):
    execution_instruction = dash.callback_context.triggered[0]['prop_id']

    # Output default
    js = []
    if selected_dropdown_value is not None:
        r, fn = os.path.split(selected_dropdown_value)
        code = fn.split('.')[0]
        if ('my-dropdown.value' in execution_instruction):
            if code not in selected_pts.keys():
                selected_pts[code] = []
        elif ('basic-interactions.clickData' in execution_instruction):
            selected_pts[code].append(clickData['points'][0]['x'])
        elif ('button.n_clicks' in execution_instruction) and (len(selected_pts[code]) != 0):
            selected_pts[code].pop()
        js = selected_pts[code]
    return json.dumps(js)


@app.callback(
    Output('container-button-basic', 'children'),
    [Input('save', 'n_clicks')])
def save_on_click(n_clicks):
    if len(selected_pts) != 0:
        save_fiducials(selected_pts,paths['output'])
    return "Saved at " + str(datetime.now())


##################### EXECUTE APP #####################
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help='input path', required=True)
    parser.add_argument('--output_path', type=str, help='output path', default='./output_{}.csv'.format(datetime.now().isoformat()))
    inputs = parser.parse_args()

    paths['inputs'] = inputs.path
    paths['output'] = inputs.output_path
    
    # Load previous output path
    if os.path.isfile(paths['output']):
        selected_pts = load_data(paths['output'])
        selected_pts.pop('0',None)

    app.layout['my-dropdown'].options = list_files(paths['inputs'])

    app.run_server()

