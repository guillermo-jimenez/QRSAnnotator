from functools import lru_cache
from os.path import dirname, join, isfile, splitext, split
from tools.files import *

import pandas as pd
from pandas.core.frame import DataFrame
import glob

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button, RadioButtonGroup
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure

DATA_DIR = '/media/guille/DADES/DADES/SoONew/RETAG'
TemplateHeader = ['I', 'II']

def nix(val, lst):
    return [x for x in lst if x != val]

# Load database
dataset             = ['A','B','C',]
dataset             = pd.read_csv('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/Dataset.csv', index_col=0)
dataset             = dataset.sort_index(axis=1)
labels              = np.asarray(list(dataset)) # In case no data augmentation is applied
description         = dataset.describe()
for key in description:
    dataset[key]    = (dataset[key] - description[key]['mean'])/description[key]['std']
files = np.sort(list(set(['_'.join(k.split('_')[:-1]) for k in dataset]))).tolist()

# Load previous segmentations
Pon = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/Pon.csv')
Poff = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/Poff.csv')
QRSon = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/QRSon.csv')
QRSoff = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/QRSoff.csv')
Ton = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/Ton.csv')
Toff = load_data('/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/Toff.csv')

# Convert to single data structures
Pwaves = {k: [[Pon[k][i], Poff[k][i]] for i in range(len(Pon[k]))] for k in Pon}
QRSwaves = {k: [[QRSon[k][i], QRSoff[k][i]] for i in range(len(QRSon[k]))] for k in QRSon}
Twaves = {k: [[Ton[k][i], Toff[k][i]] for i in range(len(Ton[k]))] for k in Ton}

# New segmentations
if isfile('./P_NEW.csv'):
    Pwaves_new = load_data('./P_NEW.csv')
else:
    Pwaves_new = {}
if isfile('./QRS_NEW.csv'):
    QRSwaves_new = load_data('./QRS_NEW.csv')
else:
    QRSwaves_new = {}
if isfile('./T_NEW.csv'):
    Twaves_new = load_data('./T_NEW.csv')
else:
    Twaves_new = {}


# set up plots
source = ColumnDataSource(
    data=dict(
        sample=[], 
        I=[], I_returns=[],
        II=[], II_returns=[]
    )
)
source_static = ColumnDataSource(
    data=dict(
        sample=[], 
        I=[], I_returns=[],
        II=[], II_returns=[]
    )
)
tools = 'xbox_select,reset,ywheel_zoom,pan,box_zoom,undo,redo,save,crosshair,hover'

# set 2 leads
leads = [figure(plot_width=1500, plot_height=200, tools=tools, x_axis_type='auto', active_drag="xbox_select") for i in range(2)]
for i in range(2):
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
        data0 = dataset[t1+'_0'].values
        data1 = dataset[t1+'_1'].values
        data = pd.DataFrame({'sample': np.arange(data0.size), 'I': data0, 'II': data1})
        source.data = data
        source_static.data = data
        
        # Retrieve segmentations
        textboxnew.text = "New points:      \t"
        if waveselector.active == 0:
            if t1 in Pwaves_new:
                if len(Pwaves_new[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Pwaves_new[t1+'_0']]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                if len(Pwaves[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Pwaves[t1+'_0']]).astype(int).squeeze().tolist()
        elif waveselector.active == 1:
            if t1 in QRSwaves_new:
                if len(QRSwaves_new[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRSwaves_new[t1+'_0']]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                if len(QRSwaves[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRSwaves[t1+'_0']]).astype(int).squeeze().tolist()
        elif waveselector.active == 2:
            if t1 in Twaves_new:
                if len(Twaves_new[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Twaves_new[t1+'_0']]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                if len(Twaves[t1+'_0']) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Twaves[t1+'_0']]).astype(int).squeeze().tolist()

        # Set xlim
        for i in range(2):
            leads[i].x_range.start = rangeslider.value[0]
            leads[i].x_range.end = rangeslider.value[1]


def selection_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        # Get data
        data0 = dataset[t1+'_0'].values
        data1 = dataset[t1+'_1'].values
        data = pd.DataFrame({'sample': np.arange(data0.size), 'I': data0, 'II': data1})

        # For some reason it needs this (?)
        selected = np.sort(source.selected.indices).astype(int).squeeze().tolist()
        if selected:
            data = data.iloc[selected, :]

        # 1. Logging
        binary_mask = np.zeros((225000,),dtype=bool)
        selected = np.sort(source.selected.indices)
        if len(selected) != 0:
            binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)
        
        sel_pts = []
        for i in range(len(offsets)):
            sel_pts.append([onsets[i],offsets[i]])
        textboxnew.text = "New points:      \t"

def retrieve_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        if waveselector.active == 0:
            if len(Pwaves[t1+'_0']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Pwaves[t1+'_0']]).astype(int).squeeze().tolist()
        elif waveselector.active == 1:
            if len(QRSwaves[t1+'_0']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRSwaves[t1+'_0']]).astype(int).squeeze().tolist()
        elif waveselector.active == 2:
            if len(Twaves[t1+'_0']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in Twaves[t1+'_0']]).astype(int).squeeze().tolist()

def save_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        # 1. Generate binary mask
        binary_mask = np.zeros((225000,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)

        # 3. Save all as distinct fiducials
        if waveselector.active == 0:
            Pwaves_new[t1+'_0'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            Pwaves_new[t1+'_1'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for P wave: \t"
        elif waveselector.active == 1:
            QRSwaves_new[t1+'_0'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            QRSwaves_new[t1+'_1'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for QRS wave: \t"
        elif waveselector.active == 2:
            Twaves_new[t1+'_0'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            Twaves_new[t1+'_1'] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for T wave: \t"
        

def write_segmentation():
    save_fiducials(Pwaves_new,'./P_NEW.csv')
    save_fiducials(QRSwaves_new,'./QRS_NEW.csv')
    save_fiducials(Twaves_new,'./T_NEW.csv')

def change_range(attrname, old, new):
    low, high = rangeslider.value

    for i in range(2):
        leads[i].x_range.start = low
        leads[i].x_range.end = high


# Set widgets
rangeslider = RangeSlider(start=140000, end=225000, step=10, value=(140000,150000), title="X range")
file_selector = Select(value=None, options=nix(None,files))
waveselector = RadioButtonGroup(labels=["P wave", "QRS wave", "T wave"], active=0)
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
waveselector.on_change('active', file_change)


# set up layout
buttons = row(waveselector,retrievebutton,storebutton,writebutton)
layout = column(file_selector,textboxnew,rangeslider,buttons,grid)

# initialize
# update()

curdoc().add_root(layout)
curdoc().title = "FullDelineator"