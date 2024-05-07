import os
import glob
import PIL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#data = torchvision.datasets.StanfordCars(root="/home/irfan/Desktop/Data/car_data/", download=False)
#show_images(data)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_files  = glob.glob(img_dir+'/*')
        self.img_labels = [None]*len(self.img_files)#pd.read_csv(annotations_file)
        self.transform  = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_files[0]
        image = PIL.Image.open(img_path)
        label = 0#self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
    


def load_transformed_dataset(img_size=64,train_img_dir='./train',test_img_dir='./test'):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    
    train_data    = CustomImageDataset(None,train_img_dir,transform=data_transform)
    test_data     = CustomImageDataset(None,test_img_dir, transform=data_transform)
    
    #train = train_data#torchvision.datasets.StanfordCars(root=".", download=True,transform=data_transform)
    #test = test_data#torchvision.datasets.StanfordCars(root=".", download=True,transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train_data, test_data])
