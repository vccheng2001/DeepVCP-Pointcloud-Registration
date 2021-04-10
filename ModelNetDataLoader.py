import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import csv
import os
import numpy as np
import sys 


class ModelNet40Dataset(Dataset):

    def __init__(self, root, category, rotate=True):
        # root directory 
        self.root = root
        # airplane, bathtub, bed, etc
        self.category = category 
        # specify random rotation
        self.rotate = rotate 
        # example: "data/modelnet40/airplane/"
        self.path = root+category
        # files in path
        self.data = os.listdir(self.path)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index=None):
        # if index is specified
        if index:
            file = self.data[index]
        else:
            file = np.random.choice(self.data) 
        
        # source pointcloud
        src = np.loadtxt(self.path+file, delimiter=",", dtype=np.float64)

        print(f"Source pointcloud shape: {src.shape}")
        if self.rotate:
            target = rotate(src)

        src = torch.from_numpy(src)
        target = torch.from_numpy(target)
        return (src, target)


def rotate(src):
    # randomly rotate pointcloud in y dir 
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0,             1,         0    ],
                    [-np.sin(theta),0, np.cos(theta)]])
    return np.dot(rot, src.reshape((3,-1)))
        

if __name__ == "__main__":
    root = './data/modelnet40_normal_resampled/'
    category = 'airplane/'
    data = ModelNet40Dataset(root=root,category=category)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for original, transformed in DataLoader:
        print(original.shape)
        print(transformed.shape)


