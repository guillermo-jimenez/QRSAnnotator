import plotly.graph_objs as go
import json
import dash
import plotly
import plotly.subplots
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import flask
import pandas as pd
import time
import pathlib
import os

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

path = pathlib.Path('/path/to/files')
output_file = pathlib.Path('/path/to/saved/Segmentation.txt')

def create_vertical_lines_tuple(fig, points, fn):
    shapes = []
    if fn in points:
        for p in points[fn]:
            for i in range(len(fig.data)):
                shapes.append(
                    go.layout.Shape(
                        type="line",
                        xref="x" + str(i+1),
                        yref="y" + str(i+1),
                        x0=p,
                        y0=fig.data[0].y.min(),
                        x1=p,
                        y1=fig.data[0].y.max(),
                    )
                )
    return tuple(shapes)



def list_files(path):
    opts = []
    # r=root, d=directories, f = files
    for r, di, f in os.walk(path):
        if pathlib.Path(r) == path:
            for file in f:
                if file.endswith('.txt'):
                    opts.append({'label': file[:-4], 'value': os.path.join(r, file)})
    return opts



def sorter(total_items):
    def sorter(item):
        if item['label'] != 'Fs':
            return (total_items**2+2)*int(item['label'].split('-')[0]) + (total_items+1)*int(item['label'].split('-')[1]) + int(item['label'].split('-')[2])
        else:
            return -1
    return sorter



def conditional_makedir(path):
    try: 
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
    except:
        list_path = os.path.normpath(path).split(os.sep)[1:]
        t = os.sep
        for p in list_path:
            t = os.path.join(t,p)
            if not os.path.exists(t):
                os.makedirs(t)



def save_fiducials(dictionary, path):
    with open(path, 'w') as f:
        for key in dictionary.keys():
            if len(dictionary[key]) != 0:
                f.write("%s,%s\n"%(key,str(dictionary[key]).replace('[','').replace(']','')))



# Data loader to un-clutter code    
def load_data(file):
    dic = dict()
    with open(file) as f:
        text = list(f)
    for line in text:
        line = line.replace(' ','').replace('\n','').replace(',,','').replace('[','').replace(']','')
        if line[-1] == ',': line = line[:-1]
        head = line.split(',')[0]
        tail = line.split(',')[1:]
        if tail == ['']:
            tail = []
        else:
            tail = np.asarray(tail).astype(int).tolist()
        
        dic[head] = tail
    return dic



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')


app = dash.Dash(__name__, 
                server=server, 
                external_stylesheets=external_stylesheets
               )

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.scripts.config.serve_locally = False
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

opts = list_files(path)
opts.sort(key=sorter(len(opts)))

TemplateHeader = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'SELECT']

# Signal template
signal = pd.DataFrame({
    'I'     : np.zeros(1000,),
    'II'    : np.zeros(1000,),
    'III'   : np.zeros(1000,),
    'AVR'   : np.zeros(1000,),
    'AVL'   : np.zeros(1000,),
    'AVF'   : np.zeros(1000,),
    'V1'    : np.zeros(1000,),
    'V2'    : np.zeros(1000,),
    'V3'    : np.zeros(1000,),
    'V4'    : np.zeros(1000,),
    'V5'    : np.zeros(1000,),
    'V6'    : np.zeros(1000,),
    'SELECT': np.zeros(1000,),
})


fig = plotly.subplots.make_subplots(rows=13, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.01)
for i in range(len(TemplateHeader)):
    fig.append_trace({'x':np.arange(len(signal[TemplateHeader[i]])),'y': signal[TemplateHeader[i]],'name': TemplateHeader[i]},i+1,1)



fig['layout']['xaxis'].update(title='', range=[0, 500], autorange=False)
fig.update_traces(xaxis="x13")
fig.layout.xaxis13.spikemode = "across"
fig.layout.xaxis13.spikethickness = 1
fig.layout.xaxis13.rangeslider.visible = True

fig.update_traces(
    line=dict(width=1.5),
)

if os.path.isfile(output_file):
    selected_pts = load_data(output_file)
    selected_pts.pop('0',None)
else:
    selected_pts = dict()

app.layout = html.Div([
    # dcc.Store(id='button-storage', storage_type='local'),
    html.H1('TV QRS marker'),
    dcc.Dropdown(
        id='my-dropdown',
        options=opts
    ),
    html.Div([
        dcc.Graph(
            id='basic-interactions',
            style={
                'height': 1200,
            },
            figure={
                'layout': {
                    'clickmode': 'event+select'
                }
            },
        ),
    ], className="six columns"),
    html.Div([
        html.Div([
            html.H1('Control pannel'),
            html.Button('Delete Last element', id='button'),
            html.Button('Save Data', id='save'),
            # html.Button('Rescale View', id='plot'),
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Div(id='container-button-basic', children='-'),
            html.Pre(id='click-data', style=styles['pre']),
            dcc.Input(id="from", type="text", placeholder="from"),
            dcc.Input(id="to", type="text", placeholder="to"),
            html.Button('Rescale View', id='plot'),
        ], className="five columns"),
    ]),
], className="row")




@app.callback(
    Output('basic-interactions', 'figure'),
    # Output('click-data', 'children')],
    [Input('plot', 'n_clicks'),
    Input('my-dropdown', 'value'),], 
    [State("from", "value"), 
    State("to", "value")])
def update_graph(plot_button, selected_dropdown_value, from_value, to_value): # , clicks_data, clickData
    # Keep track of what triggered whatever
    ctx = dash.callback_context
    if ctx.triggered:
        execution_instruction = ctx.triggered[0]['prop_id']
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
            
            # Reset Y range
            fig.update_yaxes(range=None)
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
    ctx = dash.callback_context
    if ctx.triggered:
        execution_instruction = ctx.triggered[0]['prop_id']
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
    save_fiducials(selected_pts,output_file)
    return "Saved at " + str(datetime.now())



app.run_server()



