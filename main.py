import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import int_tuple, float_tuple, str_tuple
from utils import timeit, bool_flag, LossManager

from vg import VgSceneGraphDataset, vg_collate_fn
from model import OntoGAN
VG_DIR = os.path.expanduser('data')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=2, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)

parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--normalization', default='batch')

def build_model(args, vocab):
    if args.checkpoint_start_from is not None:
        checkpoint = torch.load(args.checkpoint_start_from)
        kwargs = checkpoint['model_kwargs']
        model = Sg2ImModel(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        kwargs = {
            'vocab': vocab,
            'image_size': args.image_size,
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim,
            'normalization': args.normalization,
            'mask_size': args.mask_size,
        }
        model = OntoGAN(**kwargs)
    return model, kwargs


def build_vg_dsets(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)
    
    return vocab, train_dset, val_dset


def build_loaders(args):
    if args.dataset == 'vg':
        vocab, train_dset, val_dset = build_vg_dsets(args)
        collate_fn = vg_collate_fn
    elif args.dataset == 'coco':
        vocab, train_dset, val_dset = build_coco_dsets(args)
        collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    
    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader


def main(args):
    print(args)
    # check_args(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    t, epoch = 0, 0
    while True:
        if t >= args.num_iterations:
            break
        epoch += 1
        print('Starting epoch %d' % epoch)
        
        for batch in train_loader:
            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            t += 1
            batch = [tensor.cuda() for tensor in batch]
            masks = None
            if len(batch) == 6:
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            elif len(batch) == 7:
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
            else:
                assert False
            predicates = triples[:, 1]

            with timeit('forward', args.timing):
                model_boxes = boxes
                model_masks = masks
                model_out = model(objs, triples, obj_to_img, triple_to_img)
                obj_vecs = model_out



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)