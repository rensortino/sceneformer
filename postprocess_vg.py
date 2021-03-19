import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from torchvision.models import inception_v3
from torchvision import transforms
from collections import Counter



IMG_DIR = 'data/images'
data = {}
feature_extractor = inception_v3(pretrained=True)
model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def crop_coords_out_range(img, x0, x1, y0, y1):
    return  ( (y0.item() >= img.shape[0]) and (y1.item() >= img.shape[0]) ) or \
            ( (x0.item() >= img.shape[1]) and (x1.item() >= img.shape[1]) )

def crop_is_pad(x0, y0, w, h):
    return ( x0 == y0 == w == h == -1 )


counter = Counter()

for split in ['train', 'val', 'test']:

    image_idx_to_features = []

    with h5py.File('data/' + split + '.h5', "a") as h5_file:

        filename = 'to_delete_' + split + '.txt'
        print(f'Creating or opening file {filename}')
        with open(filename, 'a') as f:
            f.write(f'{split}:\n')

        if 'image_embeddings' in h5_file.keys():
            print('Deleting existing datset: ', 'image_embeddings')
            del h5_file['image_embeddings']

        for k, v in h5_file.items():
            if k == 'image_paths':
                image_paths = list(v)
            else:
                data[k] = torch.IntTensor(np.asarray(v))

        for i, path in enumerate(h5_file['image_paths'][42000:]):

            i += 42000

            img_path = os.path.join(IMG_DIR, path)

            img = cv2.imread(img_path)

            boxes = data['object_boxes'][i]

            x0, y0, w, h = boxes.split(1, dim=1)

            x1 = x0 + w
            y1 = y0 + h 

            pad = torch.tensor([])
            preprocessed_crops = []

            delete_image = False


            for j, box in enumerate(boxes):

                if crop_coords_out_range(img, x0[j], x1[j], y0[j], y1[j]):
                    counter.update({'oor': 1})
                    print(f'Appending image {img_path} to delete to file: {filename}')
                    with open(filename, 'a') as f:
                        f.write("%s\n" % i)
                    delete_image = True
                    break

                if crop_is_pad(x0[j], y0[j], w[j], h[j]): # creating pad rows up to max number of objects (=30)
                    pad_samples = len(boxes) - (j)
                    pad = torch.empty(pad_samples, 1000).fill_(-1)
                    break
                
                
                crop = img[y0[j] : y1[j], x0[j] : x1[j], : ]
                preprocessed = preprocess(Image.fromarray(crop))
                preprocessed_crops.append(preprocessed.unsqueeze(0))

            if delete_image:
                    continue
            if torch.cuda.is_available():
                preprocessed_crops = torch.cat(preprocessed_crops).to('cuda')
                pad = pad.to('cuda')
                model.to('cuda')  

            with torch.no_grad():     
                output = model(preprocessed_crops)

            image_idx_to_features.append(torch.cat((output, pad)))

            if i % 6000 == 0:
                name = 'image_embeddings'
                ary = torch.cat(image_idx_to_features)
                ary = np.asarray(ary.cpu())
                if i == 42000:
                    continue
                    # print('Creating datset: ', name, ary.shape, ary.dtype)
                    # with h5py.File('data/' + split + '.h5', "a") as h5_file:
                    # h5_file.create_dataset(name, data=ary, dtype=ary.dtype, maxshape=(None, 1000))
                else:
                    # print('Appending datset: ', name, ary.shape, ary.dtype)
                    # h5_file["image_embeddings"].resize((h5_file["image_embeddings"].shape[0] + ary.shape[0]), axis = 0)
                    # h5_file["image_embeddings"][-ary.shape[0]:] = ary
                    print('Creating datset: ', name, ary.shape, ary.dtype)
                    with h5py.File('data/' + split + '_embs_' + str(i) + '.h5', "w") as h5_file:
                        h5_file.create_dataset(name, data=ary, dtype=ary.dtype, maxshape=(None, 1000))
                image_idx_to_features = []
        
        name = 'image_embeddings'
        ary = torch.cat(image_idx_to_features)
        ary = np.asarray(ary.cpu())

        # print('Creating datset: ', name, ary.shape, ary.dtype)
        # h5_file.create_dataset(name, data=ary)
        print('Creating datset: ', name, ary.shape, ary.dtype)
        with h5py.File('data/' + split + '_embs_' + str(i) + '.h5', "w") as h5_file:
            h5_file.create_dataset(name, data=ary, dtype=ary.dtype, maxshape=(None, 1000))
