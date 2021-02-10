import sys
import sak
import sak.signal
import scipy as sp
import scipy.signal
import wfdb
import pandas as pd
import numpy as np
import cv2
import glob
import skimage
import skimage.morphology
import os
import os.path
from pandas.core.frame import DataFrame
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button, RadioButtonGroup, BoxAnnotation, Band, Quad
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure
from sak.signal import StandardHeader



# define callback functions
def file_change(attrname, old, new, args, file_correspondence, 
                current_data, current_keys, sources, sources_static, 
                leads, boxes_local_field, boxes_far_field, rangeslider, 
                textbox, all_waves, waveselector, local_field, far_field):
    if (new == " ") or (new is None) or (old == new):
        return None

    # Load signal
    signal,header = read_sample(file_correspondence[new])

    # Filter useless columns and sort
    signal = sort_columns(signal)
    signal = filter_bipolar(signal)
    signal = filter_reference(signal,header)
    col_names = list(signal)

    # Include signals into data dict
    data = [{"x": np.arange(signal.shape[0]), "y": signal[k].values, "label": np.full((signal.shape[0],),col_names[i])} for i,k in enumerate(signal)]

    # Load current data
    for i in range(args.num_sources):
        if i < signal.shape[1]:
            current_data[i] = data[i]
            current_keys[i] = col_names[i]
            sources[i].data = data[i]
            sources_static[i].data = data[i]
            leads[i].visible = True
    
            # Hide unfilled boxes
            for b in boxes_local_field[i]:
                b.visible = False
            for b in boxes_far_field[i]:
                b.visible = False
        else:
            sources[i].data = {"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}
            sources_static[i].data = {"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}
            current_keys[i] = None
            leads[i].visible = False
    
            # Hide unfilled boxes
            for b in boxes_local_field[i]:
                b.visible = False
            for b in boxes_far_field[i]:
                b.visible = False
    
    # Redefine rangeslider
    rangeslider.end = max([signal.shape[0],3072])

    # Remove out-of-bounds segmentations
    for wave in all_waves:
        wavedic = eval(wave)

        for k in signal:
            if f'{new}###{k}' in wavedic:
                onoff = np.array(wavedic[f'{new}###{k}'])
                if onoff.size == 0:
                    continue
                filt_onoff = (onoff >= 0).all(1) & (onoff < signal.shape[0]).all(1)
                if np.any(filt_onoff):
                    onoff = onoff[filt_onoff,:]
                    delete_locations = np.sort(np.argwhere(~filt_onoff)[:,0])

                    for ix in reversed(delete_locations):
                        wavedic[f'{new}###{k}'].pop(ix)

    # Retrieve segmentations
    wave = all_waves[waveselector.active]
    wavedic = eval(wave)

    # Load points into selector
    for i,k in enumerate(signal):
        if f'{new}###{k}' in wavedic:
            if len(wavedic[f'{new}###{k}']) != 0:
                onoff = np.array(wavedic[f'{new}###{k}'])
                onoff = [np.arange(on,off,dtype=int) for on,off in onoff]
                if len(onoff) > 1:
                    sources[i].selected.indices = np.concatenate(onoff).squeeze().tolist()
                else:
                    sources[i].selected.indices = onoff[0].squeeze().tolist()
                textbox.text = "Loaded points:      \t"
            else:
                sources[i].selected.indices = []
        else:
            sources[i].selected.indices = []

    # Set xlim
    for i,k in enumerate(signal):
        leads[i].x_range.start = rangeslider.value[0]
        leads[i].x_range.end = rangeslider.value[1]

    # Show used boxes
    for wave in all_waves:
        wavedic = eval(wave)

        for i,k in enumerate(signal):
            if f'{new}###{k}' in wavedic:
                boxes = eval(f"boxes_{wave}")
                for j,(on,off) in enumerate(wavedic[f'{new}###{k}']):
                    boxes[i][j].left = on
                    boxes[i][j].right = off
                    boxes[i][j].visible = True


# define callback functions
def wave_change(attrname, old, new, file_selector, local_field, far_field, sources, current_keys, textbox):
    fname = file_selector.value
    if (fname == " ") or (fname is None) or (new is None) or (old == new):
        return None
    
    # Retrieve segmentations
    if   new == 0: wave = "local_field"
    elif new == 1: wave = "far_field"
    else: raise ValueError(f"Not supposed to happen.")

    # Define the dict which will be used
    wavedic = eval(wave)

    for i,k in enumerate(current_keys):
        if k is None:
            continue
        if len(wavedic.get(f'{fname}###{k}', [])) != 0:
            onoff = [np.arange(on,off,dtype=int) for on,off in wavedic[f'{fname}###{k}']]
            if len(onoff) > 1:
                sources[i].selected.indices = np.concatenate(onoff).squeeze().tolist()
            else:
                sources[i].selected.indices = onoff[0].squeeze().tolist()
            textbox.text = "Loaded points:      \t"
        else:
            sources[i].selected.indices = []


def selection_change(attrname, old, new, i, file_selector, sources, waveselector, leads, boxes_far_field, boxes_local_field):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    # 1. Obtain binary mask of old and new points
    source = sources[i]
    set_new = set(new)
    set_old = set(old)
    if len(set_new.difference(set_old)) > 0:
        # Define list of new points
        new_pts = np.array(list(set_new.difference(set_old)))
        
        # Binary close masks
        tmp_mask = np.zeros_like(source.data["x"],dtype=bool)
        tmp_mask[new_pts] = True
        kernel_size = tmp_mask.size//10
        tmp_mask = np.pad(tmp_mask,kernel_size)
        tmp_mask = skimage.morphology.binary_closing(tmp_mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

        kernel_size = tmp_mask.size//10
        tmp_mask = np.pad(tmp_mask,kernel_size)
        tmp_mask = skimage.morphology.binary_closing(tmp_mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

        # Input new points in 
        source.selected.indices = list(set_old.union(set(np.where(tmp_mask)[0].tolist())))
    elif len(set_old.difference(set_new)) > 0:
        # Define list of deleted points
        erased_pts = np.array(list(set_old.difference(set_new)))

        # Binary close masks
        tmp_mask = np.zeros_like(source.data["x"],dtype=bool)
        tmp_mask[erased_pts] = True
        kernel_size = tmp_mask.size//10
        tmp_mask = np.pad(tmp_mask,kernel_size)
        tmp_mask = skimage.morphology.binary_closing(tmp_mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

        kernel_size = tmp_mask.size//10
        tmp_mask = np.pad(tmp_mask,kernel_size)
        tmp_mask = skimage.morphology.binary_closing(tmp_mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

        # Input new points in 
        source.selected.indices = list(set_old.difference(set(np.where(tmp_mask)[0].tolist())))

    # 2. Obtain onsets and offsets
    mask = np.zeros_like(source.data["x"],dtype=bool)
    selected = source.selected.indices
    if len(selected) != 0:
        mask[selected] = True
    onsets, offsets = sak.signal.get_mask_boundary(mask)

    # 3. Retrieve active wave for displaying
    if   waveselector.active == 0: wave = "local_field"
    elif waveselector.active == 1: wave = "far_field"
    else: ValueError("Not supposed to happen.")

    # 4. Hide old annotations
    boxes = eval(f"boxes_{wave}")
    for j in range(len(onsets),len(boxes[i])):
        boxes[i][j].visible = False

    # 5. Add glyphs
    for j,(on,off) in enumerate(zip(onsets,offsets)):
        for lead in leads:
            boxes[i][j].left = on
            boxes[i][j].right = off
            boxes[i][j].visible = True


def retrieve_segmentation(file_selector, waveselector, current_keys, local_field, far_field, sources):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    if   waveselector.active == 0: wave = "local_field"
    elif waveselector.active == 1: wave = "far_field"
    else: ValueError("Not supposed to happen.")

    wavedic = eval(wave)
    for i,k in enumerate(current_keys):
        if (k is not None) and (len(wavedic.get(f'{fname}###{k}',[])) != 0):
            sources[i].selected.indices = np.concatenate([np.arange(on,off) for on,off in wavedic[f'{fname}###{k}']]).astype(int).squeeze().tolist()
        else:
            sources[i].selected.indices = []



################################################### VOY POR AQUÍ ###################################################
def save_segmentation(file_selector, waveselector, sources, current_keys, local_field, far_field, textbox):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    # 1. Retrieve save dictionary
    if   waveselector.active == 0: wave = "local_field"
    elif waveselector.active == 1: wave = "far_field"
    else: raise ValueError("Not supposed to happen")
    wavedic = eval(wave)

    # 2. For every source available
    counter = 0
    for i,source in enumerate(sources):
        # 2.0. Skip if unused
        k = current_keys[i]
        if k is None:
            continue

        # 2.1. Generate binary mask
        binary_mask = np.zeros((source.data["x"].size,),dtype=bool)
        selected = np.sort(source.selected.indices).astype(int).squeeze()
        binary_mask[selected] = True

        # 2.2. Obtain onsets and offsets
        onsets, offsets = sak.signal.get_mask_boundary(binary_mask,aslist=False)
        onsets = onsets.tolist()
        offsets = offsets.tolist()

        # 2.3. Save fiducials
        if (len(onsets) != 0) and (len(offsets) != 0):
            wavedic[f"{fname}###{k}"] = [[on,off] for on,off in zip(onsets,offsets)]
            counter += 1
        elif f"{fname}###{k}" in wavedic:
            wavedic.pop(f"{fname}###{k}")

    if counter > 0:
        textbox.text = f"Stored segmentation for {wave}. \t"
    else:
        textbox.text = f"No segmentation found for {wave}. Skipping... \t"
        
def write_segmentation(local_field,far_field):
    sak.save_data(local_field,'./local_field.csv')
    sak.save_data(far_field,'./far_field.csv')

def change_range(attrname, old, new, rangeslider, leads):
    low, high = rangeslider.value

    for i,lead in enumerate(leads):
        lead.x_range.start = low
        lead.x_range.end = high

def nix(val, lst):
    return [x for x in lst if x != val]

def sort_columns(data):
    index_column = np.zeros((data.shape[1]),dtype=int)
    columns = np.array([col.split("(")[0].upper() for i,col in enumerate(data.columns)])
    filt_12Lead = (columns[:,None] == StandardHeader).sum(-1).astype(bool)
    index_column[filt_12Lead]  = np.where(columns[filt_12Lead][:,None] == StandardHeader)[1]
    index_column[~filt_12Lead] = np.arange(StandardHeader.size,data.shape[1])
    sorted_columns = data.columns[np.argsort(index_column)]

    return data[sorted_columns]

def read_sample(file):
    header = {}
    with open(file) as fp:
        for i,line in enumerate(fp):
            if i == 0:
                header["version"] = line.strip()
            elif i == 1:
                header["gain"] = float(line.strip().split("=")[1])
            elif i == 2:
                splitline = line.split()
                loc_equal = [i for i,elem in enumerate(splitline) if "=" in elem]

                for on,off in zip([-1] + loc_equal[:-1],loc_equal):
                    info = " ".join(splitline[on+1:off+1])
                    head,tail = info.split("=")
                    header[head] = tail
            elif i == 3:
                header["columns"] = [l.split("(")[0] for l in line.strip().split(" ") if l]
                header["column_ids"] = [l.split("(")[1].replace(")","") for l in line.strip().split(" ") if l]
            else:
                break
    
    signal = pd.read_fwf(file, sep=" ", skiprows=4, names=header["columns"])
    return signal,header


def filter_bipolar(data):
    bipolar_columns = [col for col in data.columns if (("-" in col) or col.split("(")[0].upper() in StandardHeader)]
    data_bipolar = data.filter(bipolar_columns)

    return data_bipolar


def filter_reference(data,header):
    if "Reference Channel" in header:
        reference_channel = header["Reference Channel"]
        
        filt_columns = [col for col in data.columns if (col.upper() not in StandardHeader) or (reference_channel in col)]
        
        return data.filter(filt_columns)
    else:
        return data

