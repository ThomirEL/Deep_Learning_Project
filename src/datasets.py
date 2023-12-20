#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
from string import ascii_letters
from PIL import Image

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from lvpyio import *
import matplotlib.pyplot as plt
from pprint import pprint
import os
import numpy as np
from skimage.transform import resize
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomHorizontalFlip


def load_buffers(buffer_size):
    # Get all .im7 files in directory
    dir_path = f"../../ML_project_data/{buffer_size}img/ImgPreproc(no_subtract)/"
    # Get all files in the directory
    all_files = os.listdir(dir_path)

    # Filter files that end with ".im7"
    im7_files = [file for file in all_files if file.endswith(".im7")]

    buffers = []    # list of buffers where buffer 0 is with 4 camera angles
    for file in im7_files:
        # Read the file
        buffer = read_buffer(dir_path + file)
        # Append to list of buffers
        buffers.append(buffer)
    return buffers

def load_dataset(buffers, camera_angle, params, shuffled=False, single=False, test=False):
    """Loads dataset and returns corresponding data loader."""

    
    # Instantiate appropriate dataset class
    if params.train_size == -1:
        dataset = BufferDataset(buffers, camera_angle, crop_size = params.random_crop_size, transform=params.random_flip)
    else:
        dataset = BufferDataset(buffers[:params.train_size], camera_angle, crop_size = params.random_crop_size, transform=params.random_flip)

    if test:
        dataset = BufferDataset(buffers, camera_angle, crop_size = 0, transform=0)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class BufferDataset(Dataset):
    def __init__(self, buffers, camera_angle, crop_size, transform):
        self.camera_angle = camera_angle
        self.buffers = buffers
        self.crop_size = crop_size
        frames = [frame[camera_angle] for frame in self.buffers]
        self.images = []
        for i in range(len(frames)):
            #for j in range(4):
            self.images.append(np.float32(frames[i].as_masked_array().data)[:512, :])
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def _random_crop(self, img):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """
        w, h = img.size
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        
        # Resize if dimensions are too small
        if min(w, h) < self.crop_size:
            img = resize(img, (self.crop_size, self.crop_size))
        
        # Random crop
        
        return(np.array(tvF.crop(img, i, j, self.crop_size, self.crop_size)))

    def _transform(self, img):
        return RandomHorizontalFlip(p=self.transform)(img)
        
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if idx == len(self.images)-1:
            next_image = self.images[idx - 1]
        else:
            next_image = self.images[idx + 1]
        if self.crop_size:
            image, next_image = Image.fromarray(image), Image.fromarray(next_image)
            image, next_image = self._random_crop(image), self._random_crop(next_image)
        image, next_image = tvF.to_tensor(image), tvF.to_tensor(next_image)
        if self.transform:
            image, next_image = self._transform(image), self._transform(next_image)
        return image, next_image
