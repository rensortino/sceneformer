import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from shutil import copyfile


CROP_DIR = 'data/crops'
data = {}

for split in ['train', 'val', 'test']:

    split_dir = os.path.join(CROP_DIR, split)

    os.makedirs(split_dir, exist_ok=True)

    with h5py.File('data/' + split + '_clone.h5', "a") as h5_file:

        for k, v in h5_file.items():
            if k == 'image_paths':
                image_paths = list(v)
            else:
                data[k] = torch.IntTensor(np.asarray(v))

        with open('data/vocab.json', 'r') as v:
            vocab = json.load(v)
            obj_label = vocab['object_idx_to_name']

        for i, path in enumerate(image_paths):
            object_num = data['objects_per_image'][i]
            objects = data['object_names'][i]

            for j in range(object_num.item()): # for each valid object
                img_name, extension = path.split('\\')[1].split('.')
                crop_folder = path.split('\\')[0]
                crop_name = f'{img_name}_{j}.{extension}'
                crop_path = os.path.join(CROP_DIR, crop_folder, crop_name)
                
                if img_name == '3258':
                    x = 0
                    pass

                class_name = obj_label[objects[j]]

                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

                dst_path = os.path.join(split_dir, class_name, crop_name)

                copyfile(crop_path, dst_path)