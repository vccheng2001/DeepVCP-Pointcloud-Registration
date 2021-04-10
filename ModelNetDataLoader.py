import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import csv
import os
import numpy as np
import sys 


class ModelNet40DataLoader(Dataset):

    def __init__(self, root, category, transform=True):
        # root directory 
        self.root = root
        self.transform = transform
        # airplane, bathtub, ..
        self.category = category 
        # example: "data/modelnet40/airplane ""
        self.path = os.listdir(root+category)


    def __getitem__(self, index=None):
        # if index is specified
        if index:
            file_name = self.path[index]
        # else choose random file 
        else:
            file_name = random.choice(self.path)    
        # original pointcloud
        original = np.loadtxt(self.root+self.category+file_name, \
                              delimiter=",", dtype=np.float64)
        
        print(f"Original pointcloud shape: {original.shape}")
        if self.transform:
            # randomly transform pointcloud
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            
            transformed = np.dot(rot, original)

            return (original, transformed)

    def __len__(self):
        return len(self.path)

if __name__ == "__main__":
    root = './data/modelnet40_normal_resampled/'
    category = 'airplane/'
    data = ModelNet40DataLoader(root=root,category=category)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    for original, transformed in DataLoader:
        print(original.shape)
        print(transformed.shape)


