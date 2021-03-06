import sak
import sak.signal
import scipy as sp
import scipy.signal
import wfdb
import pandas as pd
import numpy as np
import glob
from functools import lru_cache
from os.path import dirname, join, isfile, splitext, split
from pandas.core.frame import DataFrame
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button, RadioButtonGroup, BoxAnnotation, Band, Quad
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure

# File path
basedir = '/media/guille/DADES/DADES/Delineator/SHAREEDB/'
num_boxes = 500
StandardHeader = np.array(['III', 'V3', 'V5'])
sample_from = 300000
sample_width = 200000
rs_width = 100

boxes_P   = [BoxAnnotation(left=0,right=0, fill_alpha=0.05, fill_color="red") for i in range(num_boxes)]
boxes_QRS = [BoxAnnotation(left=0,right=0, fill_alpha=0.05, fill_color="green") for i in range(num_boxes)]
boxes_T   = [BoxAnnotation(left=0,right=0, fill_alpha=0.05, fill_color="magenta") for i in range(num_boxes)]

# Check different codes
files = []
for file in glob.glob(join(basedir,'*.dat')):
    _, fname = split(file)
    fname, ext = splitext(fname)
    files.append('/'.join([fname]))

def nix(val, lst):
    return [x for x in lst if x != val]

# New segmentations
P = {}
QRS = {}
T = {}

for wave in ["P", "QRS", "T"]:
    wavedic = eval(wave)
    if isfile(f"./{wave}.csv"):
        tmp = sak.load_data(f"./{wave}.csv")
        for k in tmp:
            onoff = np.array(tmp[k])
            wavedic[k] = [[on,off] for (on,off) in zip(onoff[::2],onoff[1::2])]

# set up plots
current_data = [{}]
source = ColumnDataSource(
    data=dict(
        sample=[], 
        III=[], III_returns=[],
        V3=[], V3_returns=[],
        V5=[], V5_returns=[],
    )
)
source_static = ColumnDataSource(
    data=dict(
        sample=[], 
        III=[], III_returns=[],
        V3=[], V3_returns=[],
        V5=[], V5_returns=[],
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

    for j in range(num_boxes):
        leads[i].add_layout(boxes_P[j])
        leads[i].add_layout(boxes_QRS[j])
        leads[i].add_layout(boxes_T[j])

grid = gridplot(leads, ncols=1, toolbar_location='right')

# define callback functions
def file_change(attrname, old, new):
    if (new is not None) and (old != new):
        # Load signal
        signal,hea = wfdb.rdsamp(join(basedir,f'{new}'))

        # Crop a few hours
        signal = signal[sample_from:sample_from+sample_width,:]

        # Filter baseline wander
        signal = sp.signal.filtfilt(*sp.signal.butter(4,   0.75/hea['fs'], 'high'),signal.T).T
 
        # Include signals into data dict
        data = {lead: signal[:,i] for i,lead in enumerate(StandardHeader)}
        data['sample'] = np.arange(signal.shape[0])

        # Load current data
        current_data[0] = data
        source.data = data
        source_static.data = data

        # Redefine rangeslider
        rangeslider.end = max([signal.shape[0]//100+rs_width,30+rs_width])

        # Hide unfilled boxes
        for wave in ["P", "QRS", "T"]:
            boxes = eval(f"boxes_{wave}")
            for b in boxes:
                b.visible = False

        # Remove out-of-bounds segmentations
        for wave in ["P", "QRS", "T"]:
            wavedic = eval(wave)
            if f'{new}' in wavedic:
                onoff = np.array(wavedic[f'{new}'])
                if onoff.size == 0:
                    continue
                filt_onoff = (onoff >= 0).all(1) & (onoff < signal.shape[0]).all(1)
                if np.any(filt_onoff):
                    onoff = onoff[filt_onoff,:]
                    delete_locations = np.sort(np.argwhere(~filt_onoff)[:,0])

                    for ix in reversed(delete_locations):
                        wavedic[f'{new}'].pop(ix)


        # Retrieve segmentations
        if   waveselector.active == 0: wave = "P"
        elif waveselector.active == 1: wave = "QRS"
        elif waveselector.active == 2: wave = "T"
        else: raise ValueError(f"Not supposed to happen.")

        # Define the dict which will be used
        wavedic = eval(wave)

        textboxnew.text = "New points:      \t"
        if f'{new}' in wavedic:
            if len(wavedic[f'{new}']) != 0:
                onoff = np.array(wavedic[f'{new}'])
                onoff = [np.arange(on-sample_from,off-sample_from,dtype=int) for on,off in onoff]
                if len(onoff) > 1:
                    source.selected.indices = np.concatenate(onoff).squeeze().tolist()
                else:
                    source.selected.indices = onoff[0].squeeze().tolist()
                textboxnew.text = "Loaded points:      \t"
            else:
                source.selected.indices = []
        else:
            source.selected.indices = []

        # Set xlim
        low, high = rangeslider.value
        for i,lead in enumerate(StandardHeader):
            leads[i].x_range.start = low*100
            leads[i].x_range.end = (high-rs_width)*100

        # Show used boxes
        for wave in ["P", "QRS", "T"]:
            wavedic = eval(wave)
            if f'{new}' in wavedic:
                boxes = eval(f"boxes_{wave}")
                for i,(on,off) in enumerate(wavedic[f'{new}']):
                    boxes[i].left = on-sample_from
                    boxes[i].right = off-sample_from
                    boxes[i].visible = True



# define callback functions
def wave_change(attrname, old, new):
    t1 = file_selector.value
    if (new is not None) and (old != new):
        # Retrieve segmentations
        if   new == 0: wave = "P"
        elif new == 1: wave = "QRS"
        elif new == 2: wave = "T"
        else: raise ValueError(f"Not supposed to happen.")

        # Define the dict which will be used
        wavedic = eval(wave)

        textboxnew.text = "New points:      \t"
        if f'{t1}' in wavedic:
            if len(wavedic[f'{t1}']) != 0:
                onoff = [np.arange(on,off,dtype=int) for on,off in wavedic[f'{t1}']]
                if len(onoff) > 1:
                    source.selected.indices = np.concatenate(onoff).squeeze().tolist()
                else:
                    source.selected.indices = onoff[0].squeeze().tolist()
                textboxnew.text = "Loaded points:      \t"
            else:
                source.selected.indices = []
        else:
            source.selected.indices = []


def selection_change(attrname, old, new):
    t1 = file_selector.value
    if t1 is not None:
        # 1. Obtain binary mask
        binary_mask = np.zeros_like(current_data[0]['III'],dtype=bool)
        selected = np.array(new)
        if len(selected) != 0:
            binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = sak.signal.get_mask_boundary(binary_mask)
        
        if   waveselector.active == 0: wave = "P"
        elif waveselector.active == 1: wave = "QRS"
        elif waveselector.active == 2: wave = "T"
        else: ValueError("Not supposed to happen.")

        # 3. Hide old annotations
        boxes = eval(f"boxes_{wave}")
        for i in range(len(onsets),len(boxes)):
            boxes[i].visible = False

        # 4. Add glyphs
        for i,(on,off) in enumerate(zip(onsets,offsets)):
            for lead in leads:
                boxes[i].left = on
                boxes[i].right = off
                boxes[i].visible = True

def retrieve_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        if   waveselector.active == 0: wave = "P"
        elif waveselector.active == 1: wave = "QRS"
        elif waveselector.active == 2: wave = "T"
        else: ValueError("Not supposed to happen.")

        wavedic = eval(wave)
        if len(wavedic[t1]) != 0:
            source.selected.indices = np.concatenate([np.arange(on,off) for on,off in wavedic[t1]]).astype(int).squeeze().tolist()

def save_segmentation():
    t1 = file_selector.value
    if t1 is not None:
        # 1. Generate binary mask
        binary_mask = np.zeros((current_data[0]['III'].size,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2. Obtain onsets and offsets
        onsets, offsets = sak.signal.get_mask_boundary(binary_mask,aslist=False)
        onsets = (onsets).tolist()
        offsets = (offsets).tolist()

        # 3. Save all as distinct fiducials
        if   waveselector.active == 0: wave = "P"
        elif waveselector.active == 1: wave = "QRS"
        elif waveselector.active == 2: wave = "T"
        else: raise ValueError("Not supposed to happen")

        wavedic = eval(wave)
        wavedic[t1] = [[sample_from+on,sample_from+off] for on,off in zip(onsets,offsets)]
        textboxnew.text = f"Stored segmentation with points for {wave} wave: \t"
        

def write_segmentation():
    sak.save_data(P,'./P.csv')
    sak.save_data(QRS,'./QRS.csv')
    sak.save_data(T,'./T.csv')

def change_range(attrname, old, new):
    low, high = rangeslider.value

    for i,lead in enumerate(StandardHeader):
        leads[i].x_range.start = low*100
        leads[i].x_range.end = (high-rs_width)*100


# Set widgets
rangeslider = RangeSlider(start=0, end=10000, step=1, value=(0,rs_width+30), title="X range")
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
waveselector.on_change('active', wave_change)


# set up layout
buttons = row(waveselector,retrievebutton,storebutton,writebutton)
layout = column(file_selector,textboxnew,rangeslider,buttons,grid)

# initialize
# update()

curdoc().add_root(layout)
curdoc().title = "Shareedb annotator"