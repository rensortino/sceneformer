import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from collections import Counter
import argparse

IMG_DIR = 'data/images'

def main(args):
    n_samples = {
        'train': args.train_samples,
        'val': args.val_samples,
        'test': args.test_samples
        }
    for split in ['train', 'val', 'test']:

        with h5py.File('data/' + split + '_cleaned.h5', "r") as h5_file:

            for k,v in h5_file.items():

                with h5py.File('data/' + split + '_cleaned_reduced.h5', "a") as cloned_file:

                    cloned_file.create_dataset(k, data=v[:int(n_samples[split])], dtype=v.dtype, chunks=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Reduction for prototyping')
    parser.add_argument("-t", "--train-samples", default="7000", help="Max number of train samples")
    parser.add_argument("-v", "--val-samples", default="700", help="Max number of val samples")
    parser.add_argument("-s", "--test-samples", default="700", help="Max number of test samples")
    args = parser.parse_args()

    main(args)