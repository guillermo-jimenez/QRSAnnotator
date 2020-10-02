from typing import Tuple
import os
import os.path
import glob
import numpy as np

def get_binary_on_off(binary_mask: np.ndarray, axis=-1) -> Tuple[list,list]:
    binary_mask = binary_mask.astype(int)
    diff = np.diff(np.pad(binary_mask,((1,1),),'constant',constant_values=0),axis=axis)

    onsets = (np.where(diff ==  1.)[0]).tolist()
    offsets = (np.where(diff == -1.)[0] - 1).tolist()
    
    return onsets, offsets

def list_files(path: str, extension: str = '') -> list:
    opts = []
    for file in glob.glob(os.path.join(path,'*'+extension)):
        if os.path.isfile(file):
            fname, ext = os.path.splitext(file)
            root, fname = os.path.split(fname)
            opts.append({'label': fname, 'value': os.path.join(file)})
    return opts

def list_files_bokeh(path: str, extension: str = '') -> list:
    opts = []
    sort_order = []
    for file in glob.glob(os.path.join(path,'*'+extension)):
        if os.path.isfile(file):
            root, fname = os.path.split(file)
            opts.append(fname)
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

