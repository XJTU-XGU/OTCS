from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
from .celeba import CelebA
from .datasets import get_dataset
from .animal import ImageList

from torchvision import transforms

def get_celeba_dataset(root = "/home/sun_guxiang/dataset"):
    source_dataset = CelebA(root=root, split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(32),
                                     transforms.Resize(64),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), subset="even",
                                 download=True)

    source_dataset_test = CelebA(root=root, split='test',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(32),
                                     transforms.Resize(64),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), subset="all",
                                 download=True)

    target_dataset = CelebA(root=root, split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(64),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), subset="odd",
                                 download=True)

    return source_dataset,target_dataset,source_dataset_test

def get_animal_dataset(root="/data/guxiang/OT/data/animal_images/train",transform=None):
    if transform is None:
        transform = transforms.Compose(
            [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.ToTensor()]
        )
    source_dataset = ImageList(open("./data/source_balanced_list.txt").readlines(),
                                          transform=transform,root = root)
    target_dataset = ImageList(open("./data/target_balanced_list.txt").readlines(),
                               transform=transform, root=root)
    return source_dataset,target_dataset

def get_animal_dataset_test(root="/data/guxiang/OT/data/animal_images/test"):
    transform = transforms.Compose(
        [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    source_dataset_test = ImageList(open("./data/source_balanced_test_list.txt").readlines(),
                                          transform=transform,root = root,
                               feature_root="exp/OT/extracted_features/source_test_feat.pkl",
                                    return_target=True)
    target_dataset_test = ImageList(open("./data/target_balanced_test_list.txt").readlines(),
                               transform=transform, root=root,
                               feature_root="exp/OT/extracted_features/target_test_feat.pkl")
    return source_dataset_test,target_dataset_test

def get_animal_dataset_keypoints(root="/home/sun_guxiang/OT/data/train",transform=None):
    if transform is None:
        transform = transforms.Compose(
            [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.ToTensor()]
        )
    source_dataset_keypoint = ImageList(open("./data/source_keypoint_list.txt").readlines(),
                                          transform=transform,root = root)
    target_dataset_keypoint = ImageList(open("./data/target_keypoint_list.txt").readlines(),
                               transform=transform, root=root)
    return source_dataset_keypoint,target_dataset_keypoint

