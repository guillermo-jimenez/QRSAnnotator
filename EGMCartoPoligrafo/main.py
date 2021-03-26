from argparse import ArgumentParser

from functools import partial

import sys
import src
import sak
import sak.signal
import scipy as sp
import scipy.signal
import wfdb
import pandas as pd
import numpy as np
import glob
import os
import os.path
import torch
import dill
from pandas.core.frame import DataFrame
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Toggle, Button, RadioButtonGroup, BoxAnnotation, Band, Quad, Span
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure
from sak.signal import StandardHeader

# Argument parser
parser = ArgumentParser()
parser.add_argument("--basedir",     type=str, required=True)
parser.add_argument("--num_boxes",   type=int, default=150)
parser.add_argument("--num_sources", type=int, default=20)
parser.add_argument("--threshold",   type=float, default=None)
parser.add_argument("--title",       type=str, default="Project")
parser.add_argument("--model_name",  type=str, default="WNet5LevelsSelfAttentionDiceOnly_20201130125349")
args = parser.parse_args(sys.argv[1:])

# Hyperparameters
all_waves = ["local_P", "local_field", "far_field"]
all_filepaths = glob.glob(os.path.join(args.basedir,'Databases','Poligrafo','*.txt'))
tools = 'xbox_select,reset,ywheel_zoom,pan,box_zoom,undo,redo,save,crosshair,hover'

# Create annotation boxes
boxes_far_field = [[BoxAnnotation(left=0,right=0,fill_alpha=0.05,line_alpha=1,line_width=1.,fill_color="magenta") for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
boxes_local_field = [[BoxAnnotation(left=0,right=0,fill_alpha=0.05,line_alpha=1,line_width=1.,fill_color="green") for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
boxes_local_P = [[BoxAnnotation(left=0,right=0,fill_alpha=0.05,line_alpha=1,line_width=1.,fill_color="red") for _ in range(args.num_boxes)] for _ in range(args.num_sources)]

# Create delineation spans & mark as invisible
span_Pon    = [[Span(location=0,dimension='height',line_color='red',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
span_Poff   = [] # [[Span(location=0,dimension='height',line_color='red',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
span_QRSon  = [[Span(location=0,dimension='height',line_color='green',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
span_QRSoff = [[Span(location=0,dimension='height',line_color='green',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
span_Ton    = [[Span(location=0,dimension='height',line_color='magenta',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
span_Toff   = [] # [[Span(location=0,dimension='height',line_color='magenta',line_dash='dashed', line_width=2) for _ in range(args.num_boxes)] for _ in range(args.num_sources)]

for wave in ["P", "QRS", "T"]:
    for t in ["on", "off"]:
        span = eval(f"span_{wave}{t}")
        for i in range(args.num_sources):
            if i < len(span):
                for j in range(args.num_boxes):
                    span[i][j].visible = False
                
# Check different codes
file_correspondence = {}
for i,file in enumerate(all_filepaths):
    _, fname = os.path.split(file)
    fname, ext = os.path.splitext(fname)
    file_correspondence['/'.join([fname])] = file
files = np.array(list(file_correspondence))
files = [" "] + files.tolist()

# New segmentations
local_P = {}
local_field = {}
far_field = {}

for wave in all_waves:
    wavedic = eval(wave)
    if os.path.isfile(f"./{wave}.csv"):
        tmp = sak.load_data(f"./{wave}.csv")
        for k in tmp:
            knew = k.replace(" ", "")
            onoff = np.array(tmp[k])
            wavedic[knew] = [[on,off] for (on,off) in zip(onoff[::2],onoff[1::2])]

# Set up sources
current_data = [{}   for _ in range(args.num_sources)]
current_keys = [None for _ in range(args.num_sources)]
sources = [ColumnDataSource(data={"x": np.arange(100), "y": np.zeros((100,))}) for _ in range(args.num_sources)]
sources_static = [ColumnDataSource(data={"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}) for _ in range(args.num_sources)]
leads = [figure(plot_width=3000, plot_height=250, tools=tools, x_axis_type='auto', active_drag="xbox_select", active_scroll="ywheel_zoom") for i in range(args.num_sources)]
previous_local_P = [[] for _ in range(args.num_sources)] # For doing the correlation thing safely
previous_local_field = [[] for _ in range(args.num_sources)] # For doing the correlation thing safely
previous_far_field = [[] for _ in range(args.num_sources)] # For doing the correlation thing safely

# Retrieve source ids
id_map_sources = {s.id: i for i,s in enumerate(sources)}
id_map_sources_static = {s.id: i for i,s in enumerate(sources_static)}

# Set common lead parameters
for i in range(args.num_sources):
    leads[i].visible = False
    leads[i].line('x', 'y', line_width=1.5, source=sources_static[i], legend_field="label")
    leads[i].circle('x', 'y', size=1, source=sources[i], color=None, selection_color="orange")
    leads[i].xaxis.visible = False
    leads[i].yaxis.visible = False
    if i != 0:
        leads[i].x_range = leads[0].x_range
        leads[i].y_range = leads[0].y_range
    for j in range(args.num_boxes):
        leads[i].add_layout(boxes_far_field[i][j])
        leads[i].add_layout(boxes_local_P[i][j])
        leads[i].add_layout(boxes_local_field[i][j])

        # Add delineation spans to layout
        if i < len(span_Pon):
            leads[i].add_layout(span_Pon[i][j])
        if i < len(span_QRSon):
            leads[i].add_layout(span_QRSon[i][j])
        if i < len(span_Ton):
           leads[i].add_layout(span_Ton[i][j])
        if i < len(span_Poff):
            leads[i].add_layout(span_Poff[i][j])
        if i < len(span_QRSoff):
            leads[i].add_layout(span_QRSoff[i][j])
        if i < len(span_Toff):
            leads[i].add_layout(span_Toff[i][j])

# Define figure grid
grid = gridplot(leads, ncols=1, toolbar_location='above')

# Load delineation models
basepath = f'/media/guille/DADES/DADES/Delineator'
model_type = 'model_best'

# Load models
models = {}
for i in range(5):
    path = os.path.join(basepath, 'TrainedModels', args.model_name, f'fold_{i+1}', f'{model_type}.model')
    models[f'fold_{i+1}'] = torch.load(path, pickle_module=dill).eval().float()



# Set widgets
rangeslider = RangeSlider(start=0, end=2500, step=10, value=(0,2500), title="X range")
slider_threshold = Slider(start=0.5, end=1, step=0.01, value=0.90, title="Threshold for propagation")
file_selector = Select(value=" ", options=files)
referenceselector = Select(value="V3", options=StandardHeader.tolist())
waveselector = RadioButtonGroup(labels=["P", "LF", "FF"], active=0, height_policy="fit", width_policy="fixed", width=80, orientation="vertical")
textbox = PreText(text="New points:      \t[]")
retrievebutton = Button(label='Retrieve Segmentation')
storebutton = Button(label='St', height_policy="fit", width_policy="fixed", width=27, orientation="vertical", button_type="primary")
writebutton = Button(label='Write to File')
propagatebutton = Toggle(label='Pr', active=True, height_policy="fit", width_policy="fixed", width=27, orientation="vertical")#aspect_ratio=0.002)
delineatebutton = Button(label='Delineate')
undelineatebutton = Button(label='Undelineate')


# Set callbacks
referenceselector.on_change('value', partial(src.reference_change, file_selector=file_selector, 
                                             file_correspondence=file_correspondence, current_data=current_data,
                                             sources=sources, sources_static=sources_static, current_keys=current_keys))

file_selector.on_change('value', partial(src.file_change, args=args, file_correspondence=file_correspondence, 
                                         current_data=current_data, current_keys=current_keys, 
                                         sources=sources, sources_static=sources_static, 
                                         leads=leads, boxes_local_field=boxes_local_field, 
                                         boxes_local_P=boxes_local_P, 
                                         boxes_far_field=boxes_far_field, rangeslider=rangeslider, 
                                         textbox=textbox, all_waves=all_waves, waveselector=waveselector, 
                                         local_field=local_field, far_field=far_field, local_P=local_P, 
                                         previous_local_field=previous_local_field, 
                                         previous_far_field=previous_far_field, 
                                         previous_local_P=previous_local_P,
                                         referenceselector=referenceselector))#, cb_save_segmentation)
for i,source in enumerate(sources):
    source.selected.on_change('indices', partial(src.selection_change, i=i, all_waves=all_waves, file_selector=file_selector, 
                                                 sources=sources, waveselector=waveselector, leads=leads, 
                                                 boxes_far_field=boxes_far_field, boxes_local_field=boxes_local_field,
                                                 boxes_local_P=boxes_local_P, args=args, propagatebutton=propagatebutton,
                                                 previous_local_field=previous_local_field, previous_far_field=previous_far_field, 
                                                 previous_local_P=previous_local_P, slider_threshold=slider_threshold))#, cb_save_segmentation)
retrievebutton.on_click(partial(src.retrieve_segmentation, file_selector=file_selector, waveselector=waveselector, 
                                current_keys=current_keys, local_field=local_field, local_P=local_P, 
                                far_field=far_field, sources=sources))
storebutton.on_click(partial(src.save_segmentation, file_selector=file_selector, waveselector=waveselector, sources=sources, 
                             current_keys=current_keys, local_field=local_field, local_P=local_P, 
                             far_field=far_field, textbox=textbox, all_waves=all_waves))
writebutton.on_click(partial(src.write_segmentation, all_waves=all_waves, local_field=local_field, far_field=far_field, local_P=local_P))
rangeslider.on_change('value', partial(src.change_range, rangeslider=rangeslider, leads=leads))
waveselector.on_change('active', partial(src.wave_change, args=args, all_waves=all_waves, file_selector=file_selector, 
                                         local_field=local_field, far_field=far_field, 
                                         local_P=local_P, previous_local_P=previous_local_P, boxes_local_P=boxes_local_P, 
                                         sources=sources, current_keys=current_keys, 
                                         previous_local_field=previous_local_field, 
                                         previous_far_field=previous_far_field,
                                         boxes_local_field=boxes_local_field, 
                                         boxes_far_field=boxes_far_field,
                                         textbox=textbox))
delineatebutton.on_click(partial(src.predict, args=args, span_Pon=span_Pon, span_Poff=span_Poff, span_QRSon=span_QRSon, 
                                 span_QRSoff=span_QRSoff, span_Ton=span_Ton, span_Toff=span_Toff, models=models,
                                 file_selector=file_selector, file_correspondence=file_correspondence))
undelineatebutton.on_click(partial(src.remove_delineations, span_Pon=span_Pon, span_Poff=span_Poff, span_QRSon=span_QRSon, 
                                   span_QRSoff=span_QRSoff, span_Ton=span_Ton, span_Toff=span_Toff))

# set up layout
buttons = row(retrievebutton,writebutton,delineatebutton,undelineatebutton)
visor = row(grid,propagatebutton,storebutton,waveselector)
layout = column(file_selector,referenceselector,textbox,slider_threshold,rangeslider,buttons,visor)

# initialize
curdoc().add_root(layout)
curdoc().title = args.title


