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
from pandas.core.frame import DataFrame
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Toggle, Button, RadioButtonGroup, BoxAnnotation, Band, Quad
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure
from sak.signal import StandardHeader

# Argument parser
parser = ArgumentParser()
parser.add_argument("--basedir",     type=str, required=True)
parser.add_argument("--num_boxes",   type=int, default=20)
parser.add_argument("--num_sources", type=int, default=100)
parser.add_argument("--threshold",   type=float, default=None)
parser.add_argument("--title",       type=str, default="Project")
args = parser.parse_args(sys.argv[1:])

# Hyperparameters
all_waves = ["local_field", "far_field"]
all_filepaths = glob.glob(os.path.join(args.basedir,'Databases','*','Export','*ECG_Export*.txt'))
tools = 'xbox_select,reset,ywheel_zoom,pan,box_zoom,undo,redo,save,crosshair,hover'

# Create annotation boxes
boxes_far_field = [[BoxAnnotation(left=0,right=0,fill_alpha=0.05,fill_color="red") for _ in range(args.num_boxes)] for _ in range(args.num_sources)]
boxes_local_field = [[BoxAnnotation(left=0,right=0,fill_alpha=0.05,fill_color="green") for _ in range(args.num_boxes)] for _ in range(args.num_sources)]

# Check different codes
file_correspondence = {}
for file in all_filepaths:
    _, fname = os.path.split(file)
    fname, ext = os.path.splitext(fname)
    file_correspondence['/'.join([fname])] = file
files = list(file_correspondence) + [" "]

# New segmentations
local_field = {}
far_field = {}

for wave in all_waves:
    wavedic = eval(wave)
    if os.path.isfile(f"./{wave}.csv"):
        tmp = sak.load_data(f"./{wave}.csv")
        for k in tmp:
            onoff = np.array(tmp[k])
            wavedic[k] = [[on,off] for (on,off) in zip(onoff[::2],onoff[1::2])]

# Set up sources
current_data = [{}   for _ in range(args.num_sources)]
current_keys = [None for _ in range(args.num_sources)]
sources = [ColumnDataSource(data={"x": np.arange(100), "y": np.zeros((100,))}) for _ in range(args.num_sources)]
sources_static = [ColumnDataSource(data={"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}) for _ in range(args.num_sources)]
leads = [figure(plot_width=1500, plot_height=150, tools=tools, x_axis_type='auto', active_drag="xbox_select", active_scroll="ywheel_zoom") for _ in range(args.num_sources)]
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
        leads[i].add_layout(boxes_local_field[i][j])

grid = gridplot(leads, ncols=1, toolbar_location='right')

# Set widgets
rangeslider = RangeSlider(start=0, end=2500, step=10, value=(0,2500), title="X range")
file_selector = Select(value=" ", options=files)
waveselector = RadioButtonGroup(labels=all_waves, active=0)
textbox = PreText(text="New points:      \t[]")
retrievebutton = Button(label='Retrieve Segmentation')
storebutton = Button(label='Store Segmentation')
writebutton = Button(label='Write to File')
propagatebutton = Toggle(label='Propagate', active=True)


# Set callbacks
file_selector.on_change('value', partial(src.file_change, args=args, file_correspondence=file_correspondence, 
                                         current_data=current_data, current_keys=current_keys, 
                                         sources=sources, sources_static=sources_static, 
                                         leads=leads, boxes_local_field=boxes_local_field, 
                                         boxes_far_field=boxes_far_field, rangeslider=rangeslider, 
                                         textbox=textbox, all_waves=all_waves, waveselector=waveselector, 
                                         local_field=local_field, far_field=far_field,
                                         previous_local_field=previous_local_field, 
                                         previous_far_field=previous_far_field))#, cb_save_segmentation)
for i,source in enumerate(sources):
    source.selected.on_change('indices', partial(src.selection_change, i=i, all_waves=all_waves, file_selector=file_selector, 
                                                 sources=sources, waveselector=waveselector, leads=leads, 
                                                 boxes_far_field=boxes_far_field, boxes_local_field=boxes_local_field,
                                                 args=args, propagatebutton=propagatebutton,
                                                 previous_local_field=previous_local_field, 
                                                 previous_far_field=previous_far_field, ))#, cb_save_segmentation)
retrievebutton.on_click(partial(src.retrieve_segmentation, file_selector=file_selector, waveselector=waveselector, 
                                current_keys=current_keys, local_field=local_field, 
                                far_field=far_field, sources=sources))
storebutton.on_click(partial(src.save_segmentation, file_selector=file_selector, waveselector=waveselector, sources=sources, 
                             current_keys=current_keys, local_field=local_field, 
                             far_field=far_field, textbox=textbox))
writebutton.on_click(partial(src.write_segmentation,local_field=local_field,far_field=far_field))
rangeslider.on_change('value', partial(src.change_range, rangeslider=rangeslider, leads=leads))
waveselector.on_change('active', partial(src.wave_change, args=args, all_waves=all_waves, file_selector=file_selector, 
                                         local_field=local_field, far_field=far_field, 
                                         sources=sources, current_keys=current_keys, 
                                         previous_local_field=previous_local_field, 
                                         previous_far_field=previous_far_field,
                                         boxes_local_field=boxes_local_field, 
                                         boxes_far_field=boxes_far_field,
                                         textbox=textbox))

# set up layout
buttons = row(waveselector,retrievebutton,storebutton,writebutton,propagatebutton)
layout = column(file_selector,textbox,rangeslider,buttons,grid)

# initialize
curdoc().add_root(layout)
curdoc().title = args.title


