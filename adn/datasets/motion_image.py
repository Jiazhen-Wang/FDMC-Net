import os
import os.path as path
import json
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from random import choice
from torch.utils.data import Dataset
from ..utils import read_dir

import torch.utils.data as data
import torch,torchvision
import h5py
import random
from torch.utils.data import Dataset
import glob, os
from PIL import Image
import numpy as np
import scipy.io as sio
from torchvision import transforms

def normal(in_image):
    value_max = np.max(in_image)
    value_min = np.min(in_image)
    return (in_image - value_min) / (value_max - value_min)


class MotionImage(Dataset):
    def __init__(self, a_dir, b_dir, random_flip=True, load_size=384, crop_size=256, crop_type="random"):
        super(MotionImage, self).__init__()
        self.deg_path = a_dir
        self.tar_path = b_dir
        deg_img1 = os.listdir(self.deg_path)
        deg_img1.sort()
        tar_img1 = os.listdir(self.tar_path)
        tar_img1.sort()
        self.deg_img = [os.path.join(self.deg_path, img1) for img1 in deg_img1]
        self.tar_img = [os.path.join(self.tar_path, img1) for img1 in tar_img1]
        self.transform = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Resize((256, 256)),])
        self.name = os.listdir(a_dir)
    def __len__(self):
        return len(self.deg_img)

    def get(self, index):


        img_clean = sio.loadmat(self.deg_img[index])['img']
        deg_img = normal(np.abs(img_clean))
        r = random.randint(0, len(self.deg_img) - 1)
        img_motion = sio.loadmat(self.tar_img[r])['img']
        tar_img = normal(np.abs(img_motion))

        #####for test
        # img_clean = sio.loadmat(self.deg_img[index])['img']
        # deg_img = normal(np.abs(img_clean))
        # img_motion = sio.loadmat(self.tar_img[index])['img']
        # tar_img = normal(np.abs(img_motion))
        data = {"data_name": self.name[index], "motion": self.transform(deg_img).float(), "clean": self.transform(tar_img).float()}
        return data

    def to_numpy(self, data, minmax=()):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        if minmax: data = self.denormalize(data, minmax)
        return data

    def __getitem__(self, index):

        return self.get(index)
