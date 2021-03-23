import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from collections import Counter



IMG_DIR = 'data/images'
CROP_DIR = 'data/crops'
data = {}


def crop_coords_out_range(img, x0, x1, y0, y1):
    return  ( (y0.item() >= img.shape[0]) and (y1.item() >= img.shape[0]) ) or \
            ( (x0.item() >= img.shape[1]) and (x1.item() >= img.shape[1]) )

def crop_is_pad(x0, y0, w, h):
    return ( x0 == y0 == w == h == -1 )


counter = Counter()

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(os.path.join(CROP_DIR, 'VG_100K'), exist_ok=True)
os.makedirs(os.path.join(CROP_DIR, 'VG_100K_2'), exist_ok=True)


for split in ['train', 'val', 'test']:

    filename = 'to_delete_' + split + '.txt'
    print(f'Creating or opening file {filename}')

    image_idx_to_features = []

    with h5py.File('data/' + split + '.h5', "a") as h5_file:

        for k, v in h5_file.items():
            if k == 'image_paths':
                image_paths = list(v)
            else:
                data[k] = torch.IntTensor(np.asarray(v))

        for i, path in enumerate(h5_file['image_paths']):

            crops = []

            img_path = os.path.join(IMG_DIR, path)

            img = cv2.imread(img_path)

            boxes = data['object_boxes'][i]

            x0, y0, w, h = boxes.split(1, dim=1)

            x1 = x0 + w
            y1 = y0 + h 

            delete_image = False

            for j, box in enumerate(boxes):

                if crop_coords_out_range(img, x0[j], x1[j], y0[j], y1[j]):
                    counter.update({'oor': 1})
                    print(f'Appending image {img_path} to delete to file: {filename}')
                    with open(filename, 'a') as f:
                        f.write(f'{i}\t{path}\n')
                    delete_image = True
                    break          

                if crop_is_pad(x0[j], y0[j], w[j], h[j]): # creating pad rows up to max number of objects (=30)
                    break      
                
                crop = img[y0[j] : y1[j], x0[j] : x1[j], : ]
                crops.append(crop) 

            if not delete_image:
                crop_path, extension = path.split('.')
                extension = '.' + extension
                for c_idx, crop in enumerate(crops):
                    cv2.imwrite(os.path.join(CROP_DIR, crop_path + '_' + str(c_idx) + extension), crop)