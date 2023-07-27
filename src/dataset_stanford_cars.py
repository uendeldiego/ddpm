import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt



def load_data_cars():
    return torchvision.datasets.StanfordCars(root='../data', download=False)


def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root="../data", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root="../data", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])
