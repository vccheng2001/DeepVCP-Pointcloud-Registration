import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import csv
import os
import numpy as np
import sys 
from utils import * 

class ModelNet40Dataset(Dataset):

    def __init__(self, root, category, augment=True,rotate=True, split="train"):
        # root directory 
        self.root = root
        self.category = category 
        self.split = split
        self.augment = augment
        self.points = []
        self.normals = []
        self.labels = []

        # training file names 
        names = np.loadtxt(os.path.join(self.root, \
            f'modelnet40_{split}.txt'), dtype=np.str)

        # iterate through training files 
        for i, file in enumerate(names):
            # read point clouds
            txt_file= os.path.join(self.root, self.category, file) + '.txt'
            data = np.loadtxt(txt_file, delimiter=',', dtype=np.float64)

            points = data[:, :3]    # xyz
            normals = data[:, 3:]   # normals from origin

            # Add to list
            self.points.append(points)
            self.normals.append(normals)
            self.labels.append(file)

        print("# Total clouds", len(self.points))



    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # source pointcloud
        src_points, src_normals, src_file =  self.points[index], self.normals[index], self.labels[index]
        
        print('Source file name:', src_file)

        # Augment by rotating x, z axes
        if self.augment:
            # generate random angles for rotation matrices 
            theta_x = np.random.uniform(0, np.pi*2)
            theta_y = np.random.uniform(0, np.pi*2)
            theta_z = np.random.uniform(0, np.pi*2) 
 
            # Generate target point cloud by doing a series of random
            # rotations on source point cloud 
            target_points = src_points.copy()

            # rotate about x axis
            target_points[:,0] = rotateX(theta_x, target_points[:, 0])
            # rotate about y axis 
            target_points[:,1] = rotateY(theta_y, target_points[:, 1])
            # rotate about z axis
            target_points[:,2] = rotateZ(theta_z, target_points[:, 2])
 
        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points)
        # return source point cloud and transformed (target) point cloud 
        return (src_points, target_points)
                
if __name__ == "__main__":
    root = './data/modelnet40_normal_resampled/'
    category = 'airplane/'
    split="train"
    index=0
    data = ModelNet40Dataset(root=root,category=category,augment=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)
    for src, target in DataLoader:
        print('Source:', src, src.shape)
        print('Target:', target, target.shape)

        


