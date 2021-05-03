from typing import Tuple

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
import torch
import tqdm
import os
import os.path
import math
import xml.etree.ElementTree as ET
from pandas.core.frame import DataFrame
from functools import partial
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, PreText, Select, Slider, RangeSlider, Button, RadioButtonGroup, BoxAnnotation, Band, Quad
from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool
from bokeh.plotting import figure
from sak.signal import StandardHeader

def predict_mask(signal, model, window_size=2048, stride=256, thr_dice=0.9, percentile=95, ptg_voting = 0.5, batch_size = 16, use_tqdm=False, normalize=False, norm_threshold=1e-6, filter=False):
    # Preprocess signal
    signal = np.copy(signal).squeeze()
    if signal.ndim == 0:
        return np.array([])
    elif signal.ndim == 1:
        signal = signal[:,None]
    elif signal.ndim == 2:
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
    else:
        raise ValueError("2 dims max allowed")
        
    # Pad if necessary
    if signal.shape[0] < window_size:
        signal = np.pad(signal,((0,math.ceil(signal.shape[0]/window_size)*window_size-signal.shape[0]),(0,0)),mode='edge')
    if (signal.shape[0]-window_size)%stride != 0:
        signal = np.pad(signal,((0,math.ceil((signal.shape[0]-window_size)/stride)*stride-(signal.shape[0]%window_size)),(0,0)),mode='edge')

    # Get dimensions
    N,L = signal.shape

    # (Optional) Normalize amplitudes
    if normalize:
        # Get signal when it's not flat zero
        norm_signal = signal[np.all(np.abs(np.diff(signal,axis=0,append=0)) >= norm_threshold,axis=1),:]

        # High pass filter normalized signal to avoid issues with baseline wander
        norm_signal = sp.signal.filtfilt(*sp.signal.butter(2, 0.5/250., 'high'),norm_signal, axis=0)

        # Compute amplitude for those segments
        amplitude = np.array(sak.signal.moving_lambda(
            norm_signal,
            256,
            partial(sak.signal.amplitude,axis=0),
            axis=0
        ))
        amplitude = amplitude[np.all(amplitude > norm_threshold,axis=1),]
        amplitude = np.percentile(amplitude, percentile, axis=0)

        # Apply normalization
        signal = signal/amplitude[None,:]

    # (Optional) Filter signal
    if filter:
        signal = sp.signal.filtfilt(*sp.signal.butter(2,   0.5/target_fs, 'high'),signal,axis=0)
        signal = sp.signal.filtfilt(*sp.signal.butter(2, 125.0/target_fs,  'low'),signal,axis=0)

    # Avoid issues with negative strides due to filtering:
    if np.any(np.array(signal.strides) < 0):
        signal = signal.copy()

    # Data structure for computing the segmentation
    windowed_signal = skimage.util.view_as_windows(signal,(window_size,1),(stride,1))

    # Flat batch shape
    new_shape = (windowed_signal.shape[0]*windowed_signal.shape[1],*windowed_signal.shape[2:])
    windowed_signal = np.reshape(windowed_signal,new_shape)

    # Exchange channel position
    windowed_signal = np.swapaxes(windowed_signal,1,2)

    # Output structures
    windowed_mask = np.zeros((windowed_signal.shape[0],3,windowed_signal.shape[-1]),dtype=float)
    
    # Compute segmentation for all leads independently
    with torch.no_grad():
        iterator = range(0,windowed_signal.shape[0],batch_size)
        if use_tqdm: 
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            inputs = {"x": torch.tensor(windowed_signal[i:i+batch_size]).cuda().float()}
            windowed_mask[i:i+batch_size] = model.cuda()(inputs)["sigmoid"].cpu().detach().numpy() > thr_dice

    # Retrieve mask as 1D
    counter = np.zeros((N), dtype=int)
    segmentation = np.zeros((3,N))
        
    # Iterate over windows
    for i in range(0,windowed_mask.shape[0],L):
        counter[(i//L)*stride:(i//L)*stride+window_size] += 1
        segmentation[:,(i//L)*stride:(i//L)*stride+window_size] += windowed_mask[i:i+L].sum(0)
    segmentation = ((segmentation/counter) >= (signal.shape[-1]*ptg_voting))

    return segmentation


def remove_delineations(span_Pon, span_Poff, span_QRSon, span_QRSoff, span_Ton, span_Toff):
    for i,wave in enumerate(["P", "QRS", "T"]):
        # for t in ["on", "off"]:
        # span = eval(f"span_{wave}{t}")
        for fid in ["on", "off"]:
            span = eval(f"span_{wave}{fid}")
            for j in range(len(span)):
                for k in range(len(span[j])):
                    span[j][k].visible = False


def Read_Polygraph(filename: str, delimiter: str = ',') -> Tuple[np.ndarray, list, float]:
    """Read_Polygraph - Reads the polygraph data (file delimited by spaces or
    commas, with a header containing basic information of the file being read.
      INPUTS:
        * file: relative or absolute path of the file to be read
        * delimiter: used delimiter. Usually a space (' ') or a comma (',').
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
          being M the number of samples and N the number of electrodes.
        * header: header of the file, in the shape of a cell of strings. Each
          element of the cell a string containing the name of an electrode.
        * Fs: Sampling frequency of the recording"""

    with open(filename, 'r') as f:
        header  = []
        Fs      = 0
        line    = f.readline().strip().split(' ')
        
        while not("[Data]" in line):
            if "Label:" in line:
                header.append(' '.join(line[1:]))

            if ("Sample" in line) and ("rate:" in line):
                Fs = float(line[2].strip("Hz"))
            
            line = f.readline().strip().split(' ')
            
        signal = np.loadtxt(f, delimiter=delimiter)

    return (signal,header,Fs)



def predict(basedir, span_Pon, span_Poff, span_QRSon, span_QRSoff, span_Ton, span_Toff, args, file_selector, file_correspondence, models):
    # Retrieve file name
    fname = file_selector.value

    # Get data
    signal,header,fs = Read_Polygraph(file_correspondence[fname])
    signal = pd.DataFrame.from_dict({k: signal[:,i] for i,k in enumerate(header)})
    signal = keep_12_lead(sort_columns(signal))
    signal = signal.values[:,:12]

    target_fs = 250.; down_factor = int(fs/target_fs);

    signal = sp.signal.decimate(signal.astype("float32"),down_factor,axis=0)

    # Obtain segmentation
    segmentation = 0
    for fold in models:
        segmentation += predict_mask(signal,models[fold],use_tqdm=False,window_size=512,stride=64,normalize=True,ptg_voting=0.5,percentile=25,filter=False)
    segmentation = segmentation >= 3

    for i,wave in enumerate(["P", "QRS", "T"]):
        on,off = sak.signal.get_mask_boundary(segmentation[i,:])
        # for t in ["on", "off"]:
        # fiducial = eval(t)
        # span = eval(f"span_{wave}{t}")
        for fid in ["on", "off"]:
            span = eval(f"span_{wave}{fid}")
            fiducial = eval(fid)
            for j in range(args.num_sources):
                for k in range(args.num_boxes):
                    if j < len(span):
                        if k < len(fiducial):
                            span[j][k].visible = True
                            span[j][k].location = fiducial[k]*down_factor
                        else:
                            span[j][k].visible = False




def get_tagged_points(basedir):
    all_studies_xml = glob.glob(os.path.join(basedir,'Databases','*','Export','[a-zA-Z]*.xml'))
    
    # Output structure
    tagged_points = {}
    
    # Iterate over studies
    for study_xml in all_studies_xml:
        # Get file path
        froot,fname = os.path.split(study_xml)
        
        # Get tree's root (all appended to it)
        root = ET.parse(study_xml).getroot()

        # Get table info
        xml_tags = root.find("Maps/TagsTable").findall("Tag")
        tag_correspondence = {tag.get('ID'): tag.get('Full_Name') for tag in xml_tags}

        # Points by map
        xml_maps =  root.find("Maps").findall("Map")

        for xml_map in xml_maps:
            xml_points = xml_map.find("CartoPoints").findall("Point")

            points = {}
            for xml_point in xml_points:
                tag = xml_point.find("Tags")
                if tag is not None:
                    path = os.path.join(froot,f"{xml_map.get('Name')}_P{xml_point.get('Id')}_ECG_Export.txt")
                    tagged_points[path] = tag_correspondence[tag.text]

    return tagged_points


# define callback functions
def file_change(attrname, old, new, args, file_correspondence, 
                current_data, current_keys, sources, sources_static, 
                leads, boxes_local_field, boxes_local_P, boxes_far_field, rangeslider, 
                textbox, all_waves, waveselector, local_field, far_field, local_P,
                previous_local_field, previous_far_field, previous_local_P, referenceselector):
    fname = new
    if (fname == " ") or (fname is None) or (old == new):
        return None

    # Load signal
    signal,header,fs = Read_Polygraph(file_correspondence[fname])
    signal = pd.DataFrame.from_dict({k: signal[:,i] for i,k in enumerate(header)})

    # Filter useless columns and sort
    signal = sort_columns(signal)
    # signal = filter_bipolar(signal)
    # signal = filter_reference(signal,header)
    col_names = list(signal)
    # print(col_names)
    # print(fname)

    # Include signals into data dict
    data = [{"x": np.arange(signal.shape[0]), "y": signal[k].values, "label": np.full((signal.shape[0],),col_names[i])} for i,k in enumerate(signal)]
    data[0]["y"] = data[0]["y"]/4 # Divide reference amplitude for better visualization

    # Load current data
    for i in range(args.num_sources):
        if i < signal.shape[1]:
            current_data[i] = data[i]
            current_keys[i] = col_names[i]
            sources[i].data = data[i]
            sources_static[i].data = data[i]
            leads[i].visible = True
    
            # Hide unfilled boxes
            for wave in all_waves:
                boxes = eval(f"boxes_{wave}")
                for b in boxes[i]:
                    b.visible = False
        else:
            sources[i].data = {"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}
            sources_static[i].data = {"x": np.arange(100), "y": np.zeros((100,)), "label": np.full((100,),"None")}
            current_keys[i] = None
            leads[i].visible = False
    
            # Hide unfilled boxes
            for wave in all_waves:
                boxes = eval(f"boxes_{wave}")
                for b in boxes[i]:
                    b.visible = False
    
    # Redefine rangeslider
    rangeslider.end = max([signal.shape[0],2500])

    # Remove out-of-bounds segmentations
    for wave in all_waves:
        wavedic = eval(wave)

        for k in signal:
            if k is not None:
                k = k.replace(" ","")
            onoff = np.array(wavedic.get(f'{fname}###{k}',[]))
            if onoff.size == 0:
                continue
            filt_onoff = (onoff >= 0).all(1) & (onoff < signal.shape[0]).all(1)
            if np.any(filt_onoff):
                onoff = onoff[filt_onoff,:]
                delete_locations = np.sort(np.argwhere(~filt_onoff)[:,0])

                for ix in reversed(delete_locations):
                    wavedic[f'{fname}###{k}'].pop(ix)


    # Retrieve segmentations
    wave = all_waves[waveselector.active]
    wavedic = eval(wave)
    previous = eval(f"previous_{wave}")

    # Input segmentation info for every wave
    for i,k in enumerate(signal):
        if k is not None:
            k = k.replace(" ", "")
        onoff = np.array(wavedic.get(f'{fname}###{k}', []))
        onoff = [np.arange(on,off+1,dtype=int) for on,off in onoff]
        if len(onoff) == 0:
            sources[i].selected.indices = []
        else:
            if len(onoff) == 1:
                tmp = onoff[0].squeeze().tolist()
            else:
                tmp = np.concatenate(onoff).squeeze().tolist()
            previous[i] = tmp
            sources[i].selected.indices = tmp
            print("Loaded points")

    # Set xlim
    for i,k in enumerate(signal):
        leads[i].x_range.start = rangeslider.value[0]
        leads[i].x_range.end = rangeslider.value[1]

    # Show used boxes
    # print(f'{fname}###{k}')
    for wave in all_waves:
        wavedic = eval(wave)
        boxes = eval(f"boxes_{wave}")

        for i in range(args.num_sources):
            k = current_keys[i]
            if k is not None:
                k = k.replace(" ","")
            
            for j in range(args.num_boxes):
                if j < len(wavedic.get(f'{fname}###{k}',[])):
                    (on,off) = wavedic[f'{fname}###{k}'][j]
                    boxes[i][j].left = on
                    boxes[i][j].right = off
                    boxes[i][j].visible = True
                else:
                    boxes[i][j].visible = False


def reference_change(attrname, old, new, file_selector, file_correspondence, current_data, sources, sources_static, current_keys):
    reference = new
    fname = file_selector.value
    if (fname == " ") or (fname is None) or (old == new):
        return None

    # Load signal
    signal,header,fs = Read_Polygraph(file_correspondence[fname])
    signal = pd.DataFrame.from_dict({k: signal[:,i] for i,k in enumerate(header)})

    # Filter useless columns and sort
    signal = sort_columns(signal)
    # signal = filter_bipolar(signal)
    # signal = filter_reference(signal,header,reference)
    col_names = list(signal)

    # Include signals into data dict
    data = [{"x": np.arange(signal.shape[0]), "y": signal[col_names[0]].values, "label": np.full((signal.shape[0],),col_names[0])}]
    data[0]["y"] = data[0]["y"]/4 # Divide reference amplitude for better visualization

    # Overwrite reference signal
    current_data[0] = data[0]
    current_keys[0] = col_names[0]
    sources[0].data = data[0]
    sources_static[0].data = data[0]


# define callback functions
def wave_change(attrname, old, new, args, all_waves, file_selector, local_field, far_field, local_P, previous_local_P, boxes_local_P, sources, current_keys, previous_local_field, previous_far_field, boxes_local_field, boxes_far_field, textbox):
    fname = file_selector.value
    if (fname == " ") or (fname is None) or (new is None) or (old == new):
        return None
    
    # Retrieve segmentations
    wave = all_waves[new]
    # if   new == 0: wave = "local_P"
    # if   new == 1: wave = "local_field"
    # elif new == 2: wave = "far_field"
    # else: raise ValueError(f"Not supposed to happen.")

    # Define the dict which will be used
    wavedic = eval(wave)
    previous = eval(f"previous_{wave}")

    for i,k in enumerate(current_keys):
        if k is None:
            continue
        k = k.replace(" ","")
        onoff = np.array(wavedic.get(f'{fname}###{k}', []))
        onoff = [np.arange(on,off+1,dtype=int) for on,off in onoff]
        if len(onoff) == 0:
            sources[i].selected.indices = []
        else:
            if len(onoff) == 1:
                tmp = onoff[0].squeeze().tolist()
            else:
                tmp = np.concatenate(onoff).squeeze().tolist()
            previous[i] = tmp
            sources[i].selected.indices = tmp
            print("Loaded points")

    # Show used boxes
    for wave in all_waves:
        # Retrieve wave and boxes
        wavedic = eval(wave)
        boxes = eval(f"boxes_{wave}")

        for i in range(args.num_sources):
            k = current_keys[i]
            if k is not None:
                k = k.replace(" ","")

            for j in range(args.num_boxes):
                if j < len(wavedic.get(f'{fname}###{k}', [])):
                    (on,off) = wavedic[f'{fname}###{k}'][j]
                    boxes[i][j].left = on
                    boxes[i][j].right = off
                    boxes[i][j].visible = True
                else:
                    boxes[i][j].visible = False



def selection_change(attrname, old, new, i, all_waves, file_selector, sources, waveselector, leads, boxes_far_field, boxes_local_field, boxes_local_P, args, propagatebutton, previous_local_field, previous_far_field, previous_local_P, slider_threshold):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    # 0. Get wave information
    wave = all_waves[waveselector.active]
    previous = eval(f"previous_{wave}")

    # 1. Obtain binary mask of old and new points
    source = sources[i]
    set_new = set(new)
    set_old = set(old)
    set_previous = set(previous[i])
    # print("\n\n\n\n\n\n\n")
    # print(f"new = {new}")
    # print(f"old = {old}")
    # print(f"previous = {previous}")
    # print(f"x = {source.data['x']}")
    # print(f"y = {source.data['y']}")
    if set_new != set_previous:
        if len(set_new.difference(set_old)) > 0:
            # Define list of new points
            new_pts = np.array(list(set_new.difference(set_old)))
            
            # Binary close masks
            mask = np.zeros_like(source.data["x"],dtype=bool)
            mask[new_pts] = True
            kernel_size = mask.size//10
            mask = np.pad(mask,kernel_size)
            mask = skimage.morphology.binary_closing(mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

            kernel_size = mask.size//10
            mask = np.pad(mask,kernel_size)
            mask = skimage.morphology.binary_closing(mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

            if (slider_threshold.value is not None) and (propagatebutton.active):
                has_been_executed = False
                while not has_been_executed:
                    #~~ Try to propagate new points by convolving with > 0.99% cross-correlation ~~#
                    # Retrieve signal
                    signal = np.copy(source.data["y"]).astype(float)

                    # Define fundamental
                    fundamental = np.copy(signal[mask].astype(float))#sak.signal.on_off_correction(np.copy(signal[mask].astype(float)))
                    ampl_fundamental = sak.signal.amplitude(fundamental)
                                
                    # Obtain windowing
                    windowed_signal  = skimage.util.view_as_windows(signal,fundamental.size).astype(float)

                    # Compute correlations
                    correlations  = np.zeros((windowed_signal.shape[0],))
                    amplitudes    = np.zeros((windowed_signal.shape[0],))
                    for j,window in enumerate(windowed_signal):
                        # Correct deviations w.r.t zero
                        try:
                            w = np.copy(window)# sak.signal.on_off_correction(np.copy(window))
                            c,_ = sak.signal.xcorr(fundamental,w,maxlags=0)
                            correlations[j] = c
                            ampl_w = sak.signal.amplitude(w)
                            amplitudes[j] = min([ampl_fundamental,ampl_w])/max([ampl_fundamental,ampl_w])
                        except:
                            # print("\n\n\n\n\n\n\n")
                            # print(f"new = {new}")
                            # print(f"old = {old}")
                            # print(f"indices = {source.selected.indices}")
                            # print(f"signal = {signal}")
                            # print(f"signal.shape = {signal.shape}")
                            # print(f"fundamental = {fundamental}")
                            # print(f"fundamental.shape = {fundamental.shape}")
                            # print(f"windowed_signal = {windowed_signal}")
                            # print(f"windowed_signal.shape = {windowed_signal.shape}")
                            # print(f"window = {window}")
                            # print(f"window.shape = {window.shape}")
                            break

                    # Predict mask
                    corr_mask = np.array(correlations) > slider_threshold.value
                    ampl_mask = sak.signal.sigmoid(100*(amplitudes-slider_threshold.value)) > slider_threshold.value
                    corr_mask = 0.5*corr_mask + 0.5*ampl_mask
                    corr_onsets = []
                    corr_offsets = []
                    for on,off in zip(*sak.signal.get_mask_boundary(corr_mask)):
                        if on != off: on += np.argmax(corr_mask[on:off])
                        else:         on += 0
                        mask[on:on+fundamental.size] = True

                    # Avoid crisis!
                    has_been_executed = True

            # Input new points in selected
            new_selected = list(set_old.union(set(np.where(mask)[0].tolist())))
            previous[i] = new_selected
            source.selected.indices = new_selected
        elif len(set_old.difference(set_new)) > 0:
            # Define list of deleted points
            erased_pts = np.array(list(set_old.difference(set_new)))

            # Binary close masks
            mask = np.zeros_like(source.data["x"],dtype=bool)
            mask[erased_pts] = True
            kernel_size = mask.size//10
            mask = np.pad(mask,kernel_size)
            mask = skimage.morphology.binary_closing(mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

            kernel_size = mask.size//10
            mask = np.pad(mask,kernel_size)
            mask = skimage.morphology.binary_closing(mask, np.ones((kernel_size,)))[kernel_size:-kernel_size]

            # Input new points in selected
            source.selected.indices = list(set_old.difference(set(np.where(mask)[0].tolist())))

    # 2. Obtain onsets and offsets
    on_off_mask = np.zeros_like(source.data["x"],dtype=bool)
    selected = source.selected.indices
    if len(selected) != 0:
        on_off_mask[selected] = True
    onsets, offsets = sak.signal.get_mask_boundary(on_off_mask)

    # 3. Retrieve active wave for displaying
    wave = all_waves[waveselector.active]
    # if   waveselector.active == 0: wave = "local_field"
    # elif waveselector.active == 1: wave = "far_field"
    # else: ValueError("Not supposed to happen.")

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


def retrieve_segmentation(file_selector, waveselector, current_keys, local_field, local_P, far_field, sources, all_waves):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    # Get selected wave
    wave = all_waves[waveselector.active]
    # if   waveselector.active == 0: wave = "local_field"
    # elif waveselector.active == 1: wave = "far_field"
    # else: ValueError("Not supposed to happen.")

    wavedic = eval(wave)
    for i,k in enumerate(current_keys):
        if k is not None:
            k = k.replace(" ","")
        if (k is not None) and (len(wavedic.get(f'{fname}###{k}',[])) != 0):
            sources[i].selected.indices = np.concatenate([np.arange(on,off+1) for on,off in wavedic[f'{fname}###{k}']]).astype(int).squeeze().tolist()
        else:
            sources[i].selected.indices = []



################################################### VOY POR AQUÍ ###################################################
def save_segmentation(file_selector, all_waves, waveselector, sources, current_keys, local_field, local_P, far_field, textbox):
    fname = file_selector.value
    if (fname == " ") or (fname is None):
        return None

    # 1. Retrieve save dictionary
    wave = all_waves[waveselector.active]
    wavedic = eval(wave)
    # if   waveselector.active == 0: wave = "local_field"
    # elif waveselector.active == 1: wave = "far_field"
    # else: raise ValueError("Not supposed to happen")

    # 2. For every source available
    counter = 0
    for i,source in enumerate(sources):
        # 2.0. Skip if unused
        k = current_keys[i]
        if k is None:
            continue
        k = k.replace(" ","")

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
            wavedic[f'{fname}###{k}'] = [[on,off] for on,off in zip(onsets,offsets)]
            counter += 1
        elif f'{fname}###{k}' in wavedic:
            wavedic.pop(f'{fname}###{k}')

    if counter > 0:
        print(f"Stored segmentation for {wave}. \t")
    else:
        print(f"No segmentation found for {wave}. Skipping... \t")
        
def write_segmentation(all_waves,local_field,far_field,local_P):
    for wave in all_waves:
        wavedic = eval(wave)
        sak.save_data(wavedic,f'./{wave}.csv')

def change_range(attrname, old, new, rangeslider, leads):
    low, high = rangeslider.value

    for i,lead in enumerate(leads):
        lead.x_range.start = low
        lead.x_range.end = high

def nix(val, lst):
    return [x for x in lst if x != val]

def sort_columns(data):
    columns       = np.array([col.split("(")[0]         for i,col in enumerate(data.columns)])
    columns_upper = np.array([col.split("(")[0].upper() for i,col in enumerate(data.columns)])
    filt_12Lead   = (columns_upper[:,None] == StandardHeader).sum(-1).astype(bool)
    sorted_columns = np.concatenate((columns[sak.argsort_as(columns_upper,StandardHeader)],columns[~filt_12Lead]))

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

    signal = pd.read_fwf(file, sep=" ", skiprows=3)
    signal.columns = [c.split("(")[0] for c in signal.columns]
    return signal,header


def filter_bipolar(data):
    bipolar_columns = [col for col in data.columns if (("-" in col) or col.split("(")[0].upper() in StandardHeader)]
    data_bipolar = data.filter(bipolar_columns)

    return data_bipolar


def keep_12_lead(data):
    columns       = np.array([col.split("(")[0]         for i,col in enumerate(data.columns)])
    columns_upper = np.array([col.split("(")[0].upper() for i,col in enumerate(data.columns)])
    filt_12Lead   = (columns_upper[:,None] == StandardHeader).sum(-1).astype(bool)
    sorted_columns = columns[sak.argsort_as(columns_upper,StandardHeader)]

    return data[sorted_columns]


def filter_reference(data,header,reference_channel: str = None):
    if reference_channel is not None:
        filt_columns = [col for col in data.columns if (col.upper() not in StandardHeader) or (reference_channel in col)]

        return data.filter(filt_columns)
    elif "Reference Channel" in header:
        reference_channel = header["Reference Channel"]
        
        filt_columns = [col for col in data.columns if (col.upper() not in StandardHeader) or (reference_channel in col)]
        
        return data.filter(filt_columns)
    else:
        return data

