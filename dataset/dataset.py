import torch
from torch import nn, Tensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import (Compose, Normalize, ToTensor, Resize,
    RandomHorizontalFlip, RandomGrayscale, RandomApply, RandomResizedCrop)
from torch.utils.data import Dataset
import numpy as np

from typing import Callable, Optional, Tuple, Union

from dataset.redwine import get_redwine, get_redwine_trafo
from dataset.protein import get_protein, get_protein_trafo
from dataset.california import get_california, get_california_trafo
from dataset.enb import get_enb, get_enb_trafo
from dataset.navalprop import get_navalpro, get_navalpro_trafo
from dataset.imagenet import get_ImageNet
from dataset.snelson import get_snelson


def get_dataset(
        name: str, 
        path: str, 
        train: bool,
        **kwargs 
    ) -> Dataset:
    
    if name.lower()=="cifar10":
        def transform(train: bool) -> Callable:
            mu  = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
            trafo = [ToTensor(), Normalize(mu, std)]
            if train:
                trafo += [
                    RandomHorizontalFlip(p = 0.5),
                    RandomGrayscale(p = 0.3),
                    RandomApply(
                        nn.ModuleList([RandomResizedCrop(32, scale = (0.2, 1.0))]), 
                        p = 0.5)
                ]
            return Compose(trafo)
        
        data = CIFAR10(path, train=train, transform=transform(train), 
            download=True)
        
    elif name.lower()=="mnist":
        def transform() -> Callable:
            return Compose([ToTensor(), Normalize((0.5, ), (0.5,))])
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="fashionmnist":
        def transform() -> Callable:
            return Compose([ToTensor(), Normalize((0.5, ), (0.5,))])
        
        data = FashionMNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="mnist_small":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(14, antialias=True), 
                 Normalize((0.5, ), (0.5,))]
            )
        
        data = MNIST(path, train=train, transform=transform(), 
            download=True)
        
    elif name.lower()=="mnist_small2class":        
        def transform() -> Callable:
            return Compose(
                [ToTensor(), 
                 Resize(7, antialias=True), 
                 Normalize((0.5, ), (0.5,))]
            )
        
        # extract zeroth (0) and first (1) class
        data = MNIST(path, train=train, transform=None, 
            download=True)
        idcs = torch.logical_or(data.targets == 0, data.targets == 1)
        X, Y = data.data.numpy()[idcs, ...], data.targets[idcs] 
        data = DatasetGenerator(X, Y, transform=transform())

    elif name.lower()=="imagenet10":
        # preprocessing according to 
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        # The loaded data is resized to 256**3 and cropped to 224**3
        def transform(train: bool=True) -> Callable:
            trafo = [ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            if train:
                trafo += [
                    RandomHorizontalFlip(p=0.5)
                ]

            return Compose(trafo)
        if train:
            X, Y, _, _ =  get_ImageNet(path, n_class=10)
        else:
            _, _, X, Y =  get_ImageNet(path, n_class=10)
        data = DatasetGenerator(X, Y, transform=transform(), is_classification=True)

    elif name.lower()=="redwine":
        if train:
            X, Y, _, _ = get_redwine()
        else:
            _, _, X, Y = get_redwine()
        data = DatasetGenerator(X, Y, transform=get_redwine_trafo(train))

    elif name.lower()=="protein":
        if train:
            X, Y, _, _ = get_protein()
        else:
            _, _, X, Y = get_protein()
        data = DatasetGenerator(X, Y, transform=get_protein_trafo(train))

    elif name.lower()=="california":
        if train:
            X, Y, _, _ = get_california()
        else:
            _, _, X, Y = get_california()
        data = DatasetGenerator(X, Y, transform=get_california_trafo(train))

    elif name.lower()=="enb":
        if train:
            X, Y, _, _ = get_enb()
        else:
            _, _, X, Y = get_enb()
        data = DatasetGenerator(X, Y, transform=get_enb_trafo(train))

    elif name.lower()=="navalpro":
        if train:
            X, Y, _, _ = get_navalpro(path)
        else:
            _, _, X, Y = get_navalpro(path)
        data = DatasetGenerator(X, Y, transform=get_navalpro_trafo(train))

    elif name.lower()=="synthsinus":
        data_loaded = np.load(path)
        if train:
            X, Y = data_loaded["X_tr"], data_loaded["Y_tr"]
        else:
            X, Y = data_loaded["X_te"], data_loaded["Y_te"]
        data = DatasetGenerator(X, Y, transform=None, is_classification=False) 

    elif name.lower()=="snelson":
        X, Y = get_snelson(name, path, train)
        data = DatasetGenerator(X, Y, transform=None, is_classification=False) 

    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
        
    return data


class DatasetGenerator(Dataset):
    """ Create a dataset """
    def __init__(
            self, 
            X: Union[Tensor, np.ndarray], 
            Y: Union[Tensor, np.ndarray], 
            transform: Optional[Callable]=None,
            is_classification: bool=False 
        ):
        self.X         = X
        if len(Y.shape) > 1 or is_classification:
            self.Y         = Y
        else:
            self.Y = Y[...,None] # add target dimension
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.X[idx], self.Y[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.Y)