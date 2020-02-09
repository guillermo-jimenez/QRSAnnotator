import os
import os.path
import numpy as np

def list_files(path):
    opts = []

    # r=root, d=directories, f = files
    for r, di, f in os.walk(path):
        if r == path:
            for file in f:
                if file.endswith('.txt'):
                    opts.append({'label': file[:-4], 'value': os.path.join(r, file)})

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
        t = '/'

        for p in list_path:
            t = os.path.join(t,p)
            
            if not os.path.exists(t):
                os.makedirs(t)

def save_fiducials(dictionary, path):
    with open(path, 'w') as f:
        for key in dictionary.keys():
            f.write("%s,%s\n"%(key,str(list(dictionary[key])).replace('[','').replace(']','')))

# Data loader to un-clutter code    
def load_data(file):
    dic = dict()

    with open(file) as f:
        text = list(f)

    for line in text:
        line = line.replace(' ','').replace('\n','').replace(',,','')
        if line[-1] == ',': line = line[:-1]
        head = line.split(',')[0]
        tail = line.split(',')[1:]
        if tail == ['']:
            tail = []
        else:
            tail = np.asarray(tail).astype(int).tolist()
        
        dic[head] = tail

    return dic

