import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from collections import Counter



def remove_row_h5(dset, row_idx):
    dset[row_idx : -1] = dset[row_idx + 1 :]

    dset.resize(dset.shape[0] - 1, axis=0)



IMG_DIR = 'data/images'
CROP_DIR = 'data/crops'
entries_to_delete = []
cleaned_dsets = []

for split in ['train', 'val', 'test']:

    filename = 'to_delete_' + split + '.txt'
    print(f'Opening file {filename}')
    with open(filename, 'r') as f:
        for line in f.readlines():
            entries_to_delete.append(line)

    with h5py.File('data/' + split + '.h5', "r") as h5_file:

        for k,v in h5_file.items():

            with h5py.File('data/' + split + '_clone.h5', "a") as cloned_file:

                cloned_file.create_dataset(k, data=v, dtype=v.dtype, chunks=True)

                dset = cloned_file[k]

                for entry in entries_to_delete:
                    try:
                        entry = int(entry)
                    except:
                        continue  
                    remove_row_h5(dset, entry)

        