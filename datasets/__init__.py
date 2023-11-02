import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN, MNIST, USPS
from datasets.celeba import CelebA
from torch.utils.data import Subset
import numpy as np


def get_dataset(args, config):
    if config.data.dataset == 'CIFAR10':
        if(config.data.random_flip):
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                    transforms.Resize(config.data.image_size),
                                    transforms.ToTensor()]))

        else:
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))

    elif config.data.dataset in ['CELEBA', 'CELEBA-even', 'CELEBA-odd']:
        root = "/home/sun_guxiang/OT/score_based_model/Tutorial/CelebA"
        subset = "all"
        if(config.data.dataset == "CELEBA-even"):
            subset = "even"
        elif(config.data.dataset == "CELEBA-odd"):
            subset = "odd"

        if config.data.random_flip:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)
        else:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]),
                             subset=subset,
                             download=True)

        test_dataset = CelebA(root=root, split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), subset=subset,
                              download=True)

    elif (config.data.dataset in ["CELEBA-32px", "CELEBA-32px-even", "CELEBA-32px-odd"]):
        root = "/home/sun_guxiang/OT/score_based_model/Tutorial/CelebA"
        subset = "all"
        if (config.data.dataset == "CELEBA-32px-even"):
            subset = "even"
        elif (config.data.dataset == "CELEBA-32px-odd"):
            subset = "odd"

        if config.data.random_flip:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(32),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)
        else:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(32),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)

        test_dataset = CelebA(root=root, split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(32),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), subset=subset,
                              download=True)


    elif(config.data.dataset == "CELEBA-8px"):
        root = "/home/sun_guxiang/OT/score_based_model/Tutorial/CelebA"
        if config.data.random_flip:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=root, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=root, split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(8),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)

    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == "MNIST":
        if config.data.random_flip:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        else:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        test_dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
    elif config.data.dataset == "USPS":
        if config.data.random_flip:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.Resize(config.data.image_size),
                               transforms.ToTensor()
                           ]))
        else:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(config.data.image_size),
                               transforms.ToTensor()
                           ]))
        test_dataset = USPS(root=os.path.join('datasets', 'USPS'),
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()
                            ]))
    elif config.data.dataset == "USPS-Pad":
        if config.data.random_flip:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        else:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        test_dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(20),  # resize and pad like MNIST
                                                transforms.Pad(4),
                                                transforms.Resize(config.data.image_size),
                                                transforms.ToTensor()
                                            ]))
    elif(config.data.dataset.upper() == "GAUSSIAN"):
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        if(config.data.isotropic):
            dim = config.data.dim
            rank = config.data.rank
            cov = np.diag( np.pad(np.ones((rank,)), [(0, dim - rank)]) )
            mean = np.zeros((dim,))
        else:
            cov = np.array(config.data.cov)
            mean = np.array(config.data.mean)
        
        shape = config.data.dataset.shape if hasattr(config.data.dataset, "shape") else None

        dataset = Gaussian(device=args.device, cov=cov, mean=mean, shape=shape)
        test_dataset = Gaussian(device=args.device, cov=cov, mean=mean, shape=shape)

    elif(config.data.dataset.upper() == "GAUSSIAN-HD"):
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        cov = np.load(config.data.cov_path)
        mean = np.load(config.data.mean_path)
        dataset = Gaussian(device=args.device, cov=cov, mean=mean)
        test_dataset = Gaussian(device=args.device, cov=cov, mean=mean)

    elif(config.data.dataset.upper() == "GAUSSIAN-HD-UNIT"):
        # This dataset is to be used when GAUSSIAN with the isotropic option is infeasible due to high dimensionality
        #   of the desired samples. If the dimension is too high, passing a huge covariance matrix is slow.
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        shape = config.data.shape if hasattr(config.data, "shape") else None
        dataset = Gaussian(device=args.device, mean=None, cov=None, shape=shape, iid_unit=True)
        test_dataset = Gaussian(device=args.device, mean=None, cov=None, shape=shape, iid_unit=True)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if len(X.size())>2:
        X = 2 * X - 1.
    return X.float()

def inverse_data_transform(config, X):
    if len(X.size()) > 2:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
