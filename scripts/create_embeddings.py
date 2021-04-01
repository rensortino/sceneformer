import h5py, json
import PIL, torch, os
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from shutil import copyfile

data = {}

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),    
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

for split in ['train', 'val', 'test']:

    with h5py.File('data/' + split + '_cleaned_reduced.h5', "a") as h5_file:

        cloned_file.create_dataset(k, data=v[:int(n_samples[split])], dtype=v.dtype, chunks=True)
