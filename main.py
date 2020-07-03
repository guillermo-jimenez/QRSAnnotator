from functools import lru_cache
from os.path import dirname, join, isfile, splitext, split
from tools.files import *

import pandas as pd
from pandas.core.frame import DataFrame
import glob

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure

DATA_DIR = '/media/guille/DADES/DADES/SoONew/RETAG'
TemplateHeader = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'SELECT']

def nix(val, lst):
    return [x for x in lst if x != val]

# Load previous output path
if isfile('./Segmentation.txt'):
    selected_pts = load_data('./SEGMENTATIONS.csv')
    selected_pts.pop('0',None)
if isfile('./SEGMENTATIONS_NEW.csv'):
    new_pts = load_data('./SEGMENTATIONS_NEW.csv')
else:
    new_pts = {}


@lru_cache()
def get_data(t1):
    fname = join(DATA_DIR, t1)
    data = pd.read_csv(fname, index_col=0)
    data.columns = map(str.upper, data.columns)
    data = data.dropna()
    data['sample'] = np.arange(data.shape[0])
    return data

# set up plots
source = ColumnDataSource(
    data=dict(
        sample=[], 
        I=[], I_returns=[],
        II=[], II_returns=[],
        III=[], III_returns=[],
        AVR=[], AVR_returns=[],
        AVL=[], AVL_returns=[],
        AVF=[], AVF_returns=[],
        V1=[], V1_returns=[],
        V2=[], V2_returns=[],
        V3=[], V3_returns=[],
        V4=[], V4_returns=[],
        V5=[], V5_returns=[],
        V6=[], V6_returns=[],
    )
)
source_static = ColumnDataSource(
    data=dict(
        sample=[], 
        I=[], I_returns=[],
        II=[], II_returns=[],
        III=[], III_returns=[],
        AVR=[], AVR_returns=[],
        AVL=[], AVL_returns=[],
        AVF=[], AVF_returns=[],
        V1=[], V1_returns=[],
        V2=[], V2_returns=[],
        V3=[], V3_returns=[],
        V4=[], V4_returns=[],
        V5=[], V5_returns=[],
        V6=[], V6_returns=[],
    )
)
tools = 'xbox_select,reset,ywheel_zoom,xzoom_in,pan,box_zoom,undo,redo,save,crosshair,hover'

# set 12 leads
leads = [figure(plot_width=1500, plot_height=80, tools=tools, x_axis_type='auto', active_drag="xbox_select") for i in range(12)]
for i in range(12):
    leads[i].line('sample', TemplateHeader[i], line_width=1.5, source=source_static)
    leads[i].circle('sample', TemplateHeader[i], size=1, source=source, color=None, selection_color="orange")
    leads[i].xaxis.visible = False
    leads[i].yaxis.visible = False
    if i != 0:
        leads[i].x_range = leads[0].x_range
        leads[i].y_range = leads[0].y_range

grid = gridplot(leads, ncols=1, toolbar_location='right')

# define callback functions
def file_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        TAG, ext = splitext(t1)
        IDs = ['-'.join([TAG,str(i)]) for i in range(10)]
        
        data = get_data(t1)
        source.data = data
        source_static.data = data
        source.selected.indices = []

        sel_pts = []
        for ID in IDs:
            if ID in selected_pts:
                source.selected.indices = np.concatenate((np.array(source.selected.indices),np.arange(selected_pts[ID][0],selected_pts[ID][1]))).astype(int).squeeze().tolist()
                sel_pts.append(selected_pts[ID])
        textboxnew.text = "New points:      \t" + str([])
        textboxold.text = "Original points: \t" + str(sel_pts)
        rangeslider.end = data.shape[0]

def selection_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        if isinstance(source.data, DataFrame):
            data = source.data
        else:
            data = get_data(t1)

        selected = np.sort(source.selected.indices).astype(int).squeeze().tolist()

        if selected:
            data = data.iloc[selected, :]

        # 1. Logging
        binary_mask = np.zeros((100000,),dtype=bool)
        selected = np.sort(source.selected.indices)
        if len(selected) != 0:
            binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)
        
        sel_pts = []
        for i in range(len(offsets)):
            sel_pts.append([onsets[i],offsets[i]])
        textboxnew.text = "New points:      \t" + str(sel_pts)

def retrieve_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        TAG, ext = splitext(t1)
        IDs = ['-'.join([TAG,str(i)]) for i in range(10)]
        source.selected.indices = []
        sel_pts = []
        for ID in IDs:
            if ID in selected_pts:
                source.selected.indices = np.concatenate((np.array(source.selected.indices),np.arange(selected_pts[ID][0],selected_pts[ID][1]))).astype(int).squeeze().tolist()
                sel_pts.append(selected_pts[ID])
        textboxold.text = "Original points: \t" + str(sel_pts)

def save_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        TAG, ext = splitext(t1)

        # 1. Generate binary mask
        binary_mask = np.zeros((100000,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)

        # 3. Save all as distinct fiducials
        sel_pts = []
        for i in range(len(onsets)):
            new_pts[TAG+'-'+str(i)] = [onsets[i],offsets[i]]
            sel_pts.append([onsets[i],offsets[i]])
        textboxnew.text = "Saved segmentation with points: \t" + str(sel_pts)

def write_segmentation():
    save_fiducials(new_pts,'./SEGMENTATIONS_NEW.csv')

def change_range(attr, old, new):
    low, high = rangeslider.value

    for i in range(12):
        leads[i].x_range.start = low
        leads[i].x_range.end = high

# Set widgets
rangeslider = RangeSlider(start=0, end=1000, step=1, value=(0,1000), title="X range")
file_selector = Select(value=None, options=nix(None,list_files_bokeh(DATA_DIR)))
textboxold = PreText(text="Original points: \t[]")
textboxnew = PreText(text="New points:      \t[]")
retrievebutton = Button(label='Retrieve Segmentation')
storebutton = Button(label='Store Segmentation')
writebutton = Button(label='Write to File')

# Set callbacks
file_selector.on_change('value', file_change)
source.selected.on_change('indices', selection_change)
retrievebutton.on_click(retrieve_segmentation)
storebutton.on_click(save_segmentation)
writebutton.on_click(write_segmentation)
rangeslider.on_change('value',change_range)


# set up layout
textboxes = row(textboxold,textboxnew)
buttons = row(retrievebutton,storebutton,writebutton,rangeslider)
layout = column(file_selector,textboxes,buttons,grid)

# initialize
# update()

curdoc().add_root(layout)
curdoc().title = "QRSAnnotator"