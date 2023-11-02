
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, feature_root=None, labels=None, transform=None, target_transform=None, mode='RGB'
                 ,root=None,return_target=False):
        imgs = make_dataset(image_list, labels)

        if len(imgs) == 0:
            raise(RuntimeError("No image found !"))
        if feature_root is not None:
            self.features = torch.load(feature_root)["features"]
        self.feature_root = feature_root

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.root = root
        self.return_target = return_target

    def __getitem__(self, index):
        path, target = self.imgs[index]
        path = os.path.join(self.root,path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.feature_root is None:
            return img, target
        elif self.return_target:
            return img,self.features[index],target
        else:
            return img,self.features[index]

    def __len__(self):
        return len(self.imgs)