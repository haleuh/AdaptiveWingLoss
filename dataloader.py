import sys
import os
import random
import glob
import torch
from skimage import io
from skimage import transform as ski_transform
from skimage.color import rgb2gray
import scipy.io as sio
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Lambda, Compose
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
from utils.utils import cv_crop, cv_rotate, draw_gaussian, transform, power_transform, shuffle_lr, fig2data, \
    generate_weight_map
from PIL import Image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float().div(255.0)}


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, img_list, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        item = self.img_list[idx]
        pil_image = Image.open(os.path.join(self.img_dir, item[1]))
        image = np.array(pil_image)
        box = item[3:7]
        left, top, right, bottom = box
        center = np.array([right - (right - left) / 2.0,
                  bottom - (bottom - top) / 2.0])
        center[1] = center[1] - (bottom - top) * 0.12
        scale = (right - left + bottom - top) / 195.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float().div(255.0)

        return image, center, scale


def get_dataset(val_img_dir, val_landmarks_dir, batch_size, num_landmarks=98):
    val_transforms = transforms.Compose([ToTensor()])

    val_dataset = FaceLandmarksDataset(val_img_dir, val_landmarks_dir,
                                       num_landmarks=num_landmarks,
                                       transform=val_transforms)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=6)
    data_loaders = {'val': val_dataloader}
    dataset_sizes = {'val': len(val_dataset)}
    return data_loaders, dataset_sizes
