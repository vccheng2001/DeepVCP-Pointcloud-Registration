from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import csv
import os
import numpy as np
import sys 


class ModelNet40DataLoader(Dataset):
    def __init__(self, root, category, add_noise=True)
        self.root = root
        self.add_noise = add_noise
        # airplane, bathtub, ..
        self.category = category 
        # example: "data/modelnet40/airplane ""
        self.path = os.listdir(root+category)


    def __getitem__(self, index=None):
        # if index is specified
        if index:
            file_name = self.path[i]
        # else choose random file 
        else:
            file_name = random.choice(self.path)    
        # original pointcloud
        original = np.loadtxt(file_name, dtype=np.float64)
        
        if add_noise:
            # randomly transform pointcloud
            th = np.random.uniform(0, 2*pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            
            transformed = np.dot(rot, original)

            return (original, transformed)

    def __len__(self):
        return len(self.path)

if __name__ == "__main__":

    data = ModelNet40DataLoader('/data/modelnet40_normal_resampled/')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    for original, transformed in DataLoader:
        print(original.shape)
        print(transformed.shape)


