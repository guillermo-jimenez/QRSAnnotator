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
    for lead in StandardHeader:
        files.append('/'.join([fname])+'###{}'.format(lead))

def nix(val, lst):
    return [x for x in lst if x != val]

def get_fiducials(fname, lead):
    ann = wfdb.rdann(join(basedir,fname),'atr_{}'.format(lead.lower()))
    
    locP = np.where(np.array(ann.symbol) == 'p')[0]
    if len(locP) != 0:
        if locP[0]-1 < 0:
            locP = locP[1:]
        if locP[-1]+1 == len(ann.sample):
            locP = locP[:-1]
    p = []
    for j in range(locP.size):
        p.append(np.arange(ann.sample[locP[j]-1],ann.sample[locP[j]+1]))
    p = np.concatenate(p).tolist()

    locQRS = np.where(np.array(ann.symbol) == 'N')[0]
    if len(locQRS) != 0:
        if locQRS[0]-1 < 0:
            locQRS = locQRS[1:]
        if locQRS[-1]+1 == len(ann.sample):
            locQRS = locQRS[:-1]
    qrs = []
    for j in range(locQRS.size):
        qrs.append(np.arange(ann.sample[locQRS[j]-1],ann.sample[locQRS[j]+1]))
    qrs = np.concatenate(qrs).tolist()

    locT = np.where(np.array(ann.symbol) == 't')[0]
    if len(locT) != 0:
        if locT[0]-1 < 0:
            locT = locT[1:]
        if locT[-1]+1 == len(ann.sample):
            locT = locT[:-1]
    t = []
    for j in range(locT.size):
        t.append(np.arange(ann.sample[locT[j]-1],ann.sample[locT[j]+1]))
    t = np.concatenate(t).tolist()

    return p, qrs, t

# New segmentations
P = load_data('./P_ludb.csv') if isfile('./P_ludb.csv') else {}
QRS = load_data('./QRS_ludb.csv') if isfile('./QRS_ludb.csv') else {}
T = load_data('./T_ludb.csv') if isfile('./T_ludb.csv') else {}

current_data = [{}]

# set up plots
source = ColumnDataSource(
    data=dict(
        sample=[], 
        lead=[], lead_returns=[],
    )
)
source_static = ColumnDataSource(
    data=dict(
        sample=[], 
        lead=[], lead_returns=[],
    )
)
tools = 'xbox_select,reset,ywheel_zoom,pan,box_zoom,undo,redo,save,crosshair,hover'

# set 2 lead
lead = [figure(plot_width=1500, plot_height=200, tools=tools, x_axis_type='auto', active_drag="xbox_select")]
lead[0].line('sample', 'lead', line_width=1.5, source=source_static)
lead[0].circle('sample', 'lead', size=1, source=source, color=None, selection_color="orange")
lead[0].xaxis.visible = False
lead[0].yaxis.visible = False
grid = gridplot(lead, ncols=1, toolbar_location='right')

# define callback functions
def file_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        fname,flead = t1.split('###')
        signal,header = wfdb.rdsamp(join(basedir,fname),channels=[np.max(StandardHeader == flead.upper())])

        # Get leads
        p,qrs,t = get_fiducials(fname,flead)

        signal = sp.signal.filtfilt(*sp.signal.butter(4,   0.5/header['fs'], 'high'),signal.T).T
        signal = sp.signal.filtfilt(*sp.signal.butter(4, 125.0/header['fs'],  'low'),signal.T).T
        sig_name = np.array(list(map(str.upper,header['sig_name'])))
        data = {
            'lead': signal.squeeze(),
            'sample': np.arange(signal.shape[0]),
        }

        current_data[0] = data

        source.data = data
        source_static.data = data
        
        # Retrieve segmentations
        textboxnew.text = "New points:      \t"
        if waveselector.active == 0:
            if t1 in P:
                if len(P[t1]) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in P[t1]]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                source.selected.indices = p
                textboxnew.text = "Original points:      \t"
        elif waveselector.active == 1:
            if t1 in QRS:
                if len(QRS[t1]) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRS[t1]]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                source.selected.indices = qrs
                textboxnew.text = "Original points:      \t"
        elif waveselector.active == 2:
            if t1 in T:
                if len(T[t1]) != 0:
                    source.selected.indices = np.concatenate([np.arange(on,off) for on,off in T[t1]]).astype(int).squeeze().tolist()
                    textboxnew.text = "Loaded points:      \t"
            else:
                source.selected.indices = t
                textboxnew.text = "Original points:      \t"

        # Set xlim
        lead[0].x_range.start = rangeslider.value[0]
        lead[0].x_range.end = rangeslider.value[1]


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
        binary_mask = np.zeros((current_data[0]['lead'].size,),dtype=bool)
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
            if len(P[t1]) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in P[t1]]).astype(int).squeeze().tolist()
        elif waveselector.active == 1:
            if len(QRS[t1]) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in QRS[t1]]).astype(int).squeeze().tolist()
        elif waveselector.active == 2:
            if len(T[t1]) != 0:
                source.selected.indices = np.concatenate([np.arange(on,off) for on,off in T[t1]]).astype(int).squeeze().tolist()

def save_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        # 1. Generate binary mask
        binary_mask = np.zeros((current_data[0]['lead'].size,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = get_binary_on_off(binary_mask)

        # 3. Save all as distinct fiducials
        if waveselector.active == 0:
            P[t1] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for P wave: \t"
        elif waveselector.active == 1:
            QRS[t1] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for QRS wave: \t"
        elif waveselector.active == 2:
            T[t1] = [[onsets[i],offsets[i]] for i in range(len(offsets))]
            textboxnew.text = "Stored segmentation with points for T wave: \t"
        

def write_segmentation():
    save_fiducials(P,'./P_ludb.csv')
    save_fiducials(QRS,'./QRS_ludb.csv')
    save_fiducials(T,'./T_ludb.csv')

def change_range(attrname, old, new):
    low, high = rangeslider.value

    lead[0].x_range.start = low
    lead[0].x_range.end = high


# Set widgets
rangeslider = RangeSlider(start=0, end=500000, step=10, value=(0,5000), title="X range")
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