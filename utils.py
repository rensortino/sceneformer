#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import inspect
import subprocess
from contextlib import contextmanager

import torch

import PIL
import torch
import torchvision.transforms as T


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


def bool_flag(s):
  if s == '1':
    return True
  elif s == '0':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
  raise ValueError(msg % s)


def lineno():
  return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
  torch.cuda.synchronize()
  opts = [
      'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
  ]
  cmd = str.join(' ', opts)
  ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  output = ps.communicate()[0].decode('utf-8')
  output = output.split("\n")[1].split(":")
  consumed_mem = int(output[1].strip().split(" ")[0])
  return consumed_mem


@contextmanager
def timeit(msg, should_time=True):
  if should_time:
    torch.cuda.synchronize()
    t0 = time.time()
  yield
  if should_time:
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000.0
    print('%s: %.2f ms' % (msg, duration))


class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de


class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)


def unpack_var(v):
  if isinstance(v, torch.autograd.Variable):
    return v.data
  return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
  triples = unpack_var(triples)
  obj_data = [unpack_var(o) for o in obj_data]
  obj_to_img = unpack_var(obj_to_img)
  triple_to_img = unpack_var(triple_to_img)

  triples_out = []
  obj_data_out = [[] for _ in obj_data]
  obj_offset = 0
  N = obj_to_img.max() + 1
  for i in range(N):
    o_idxs = (obj_to_img == i).nonzero().view(-1)
    t_idxs = (triple_to_img == i).nonzero().view(-1)

    cur_triples = triples[t_idxs].clone()
    cur_triples[:, 0] -= obj_offset
    cur_triples[:, 2] -= obj_offset
    triples_out.append(cur_triples)

    for j, o_data in enumerate(obj_data):
      cur_o_data = None
      if o_data is not None:
        cur_o_data = o_data[o_idxs]
      obj_data_out[j].append(cur_o_data)

    obj_offset += o_idxs.size(0)

  return triples_out, obj_data_out

