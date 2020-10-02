from functools import lru_cache
from os.path import dirname, join, isfile, splitext, split
from tools.files import *

import scipy as sp
import scipy.signal
import wfdb
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import glob

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button, RadioButtonGroup
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure

basedir = '/home/guille/DADES/DADES/Delineator/ludb'

# Standard header
StandardHeader = np.array(['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

# Check different codes
files = []

for file in glob.glob(join(basedir,'*.dat')):
    _, fname = split(file)
    fname, ext = splitext(fname)
    files.append('/'.join([fname]))

def nix(val, lst):
    return [x for x in lst if x != val]

# New segmentations
P = load_data('./P.csv') if isfile('./P.csv') else {}
QRS = load_data('./QRS.csv') if isfile('./QRS.csv') else {}
T = load_data('./T.csv') if isfile('./T.csv') else {}

current_data = [{}]

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
tools = 'xbox_select,reset,ywheel_zoom,pan,box_zoom,undo,redo,save,crosshair,hover'

# set 2 leads
leads = [figure(plot_width=1500, plot_height=200, tools=tools, x_axis_type='auto', active_drag="xbox_select") for i in range(StandardHeader.size)]
for i in range(StandardHeader.size):
    leads[i].line('sample', StandardHeader[i], line_width=1.5, source=source_static)
    leads[i].circle('sample', StandardHeader[i], size=1, source=source, color=None, selection_color="orange")
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
        signal,header = wfdb.rdsamp(join(basedir,t1))
        signal = sp.signal.filtfilt(*sp.signal.butter(4,   0.5/250., 'high'),signal.T).T
        signal = sp.signal.filtfilt(*sp.signal.butter(4, 125.0/250.,  'low'),signal.T).T
        sig_name = np.array(list(map(str.upper,header['sig_name'])))
        order = np.where(sig_name[:,None] == StandardHeader)[1]
        signal = signal[:,order]
        data = {StandardHeader[i]: signal[:,i] for i in range(StandardHeader.size)}
        data['sample'] = np.arange(signal.shape[0])

        current_data[0] = data

        source.data = data
        source_static.data = data
        
        # Retrieve segmentations
        textboxnew.text = "New points:      \t"
        if waveselector.active == 0:
            if t1+'###I' in P:
                if len(P[t1+'###I']) != 0:
                    onoff = list(zip(P[t1+'###I'][::2],P[t1+'###I'][1::2]))
                    source.selected.indices = np.concatenate([np.arange(on,off,dtype=int) for on,off in onoff]).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
        elif waveselector.active == 1:
            if t1+'###I' in QRS:
                if len(QRS[t1+'###I']) != 0:
                    onoff = list(zip(QRS[t1+'###I'][::2],QRS[t1+'###I'][1::2]))
                    source.selected.indices = np.concatenate([np.arange(on,off,dtype=int) for on,off in onoff]).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
        elif waveselector.active == 2:
            if t1+'###I' in T:
                if len(T[t1+'###I']) != 0:
                    onoff = list(zip(T[t1+'###I'][::2],T[t1+'###I'][1::2]))
                    source.selected.indices = np.concatenate([np.arange(on,off,dtype=int) for on,off in onoff]).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"

        # Set xlim
        for i in range(StandardHeader.size):
            leads[i].x_range.start = rangeslider.value[0]
            leads[i].x_range.end = rangeslider.value[1]


def selection_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        # Get data
        data = pd.DataFrame(current_data[0])

        # For some reason it needs this (?)
        selected = np.sort(source.selected.indices).astype(int).squeeze().tolist()
        if selected:
            data = data.iloc[selected, :]

        # 1. Logging
        binary_mask = np.zeros((current_data[0]['I'].size,),dtype=bool)
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
            if len(P[t1+'###I']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in P[t1+'###I']]).astype(int).squeeze().tolist()
        elif waveselector.active == 1:
            if len(QRS[t1+'###I']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRS[t1+'###I']]).astype(int).squeeze().tolist()
        elif waveselector.active == 2:
            if len(T[t1+'###I']) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in T[t1+'###I']]).astype(int).squeeze().tolist()

def save_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        # 1. Generate binary mask
        binary_mask = np.zeros((current_data[0]['I'].size,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)

        # 3. Save all as distinct fiducials
        if waveselector.active == 0:
            for lead in StandardHeader:
                P[t1+'###'+lead] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for P wave: \t"
        elif waveselector.active == 1:
            for lead in StandardHeader:
                QRS[t1+'###'+lead] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for QRS wave: \t"
        elif waveselector.active == 2:
            for lead in StandardHeader:
                T[t1+'###'+lead] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for T wave: \t"
        

def write_segmentation():
    save_fiducials(P,'./P.csv')
    save_fiducials(QRS,'./QRS.csv')
    save_fiducials(T,'./T.csv')

def change_range(attrname, old, new):
    low, high = rangeslider.value

    for i in range(StandardHeader.size):
        leads[i].x_range.start = low
        leads[i].x_range.end = high


# Set widgets
rangeslider = RangeSlider(start=0, end=500000, step=10, value=(0,3000), title="X range")
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