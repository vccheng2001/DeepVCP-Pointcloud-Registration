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
    def __init__(self, root, augment=True, rotate=True, full_dataset=True, split="train"):
        # root directory 
        self.root = root
        self.split = split
        self.augment = augment
        self.points = []
        self.normals = []
        self.labels = []
        self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]

        # training file names 
        if full_dataset == True:
            names = np.loadtxt(os.path.join(self.root, \
                f'modelnet10_{split}.txt'), dtype=np.str)
        else:
            names = np.loadtxt(os.path.join(self.root, \
                f'modelnet10_small_{split}.txt'), dtype=np.str)

        # iterate through training files 
        for i, file in enumerate(names):
            # read point clouds
            category, num = file.split('_0')
            txt_file= os.path.join(self.root, category, file) + '.txt'
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
        src_points, src_normals, src_file =  self.points[index].T, self.normals[index].T, self.labels[index]
        
        print('Processing file:', src_file)
        # Augment by rotating x, z axes
        if self.augment:
            # generate random angles for rotation matrices 
            theta_x = np.random.uniform(0, np.pi*2)
            theta_y = np.random.uniform(0, np.pi*2)
            theta_z = np.random.uniform(0, np.pi*2)

            # generate random translation
            translation_max = 1.0
            translation_min = -1.0
            t = (translation_max - translation_min) * torch.rand(3, 1) + translation_min
 
            # Generate target point cloud by doing a series of random
            # rotations on source point cloud 
            Rx = RotX(theta_x)
            Ry = RotY(theta_y)
            Rz = RotZ(theta_z)
            R = Rx @ Ry @ Rz

            # rotate source point cloud and normals
            target_points = R @ src_points
            target_normal = R @ src_normals

        src_points = torch.from_numpy(src_points)
        src_normals = torch.from_numpy(src_normals)
        target_points = torch.from_numpy(target_points) + t
        target_normal = torch.from_numpy(target_normal)

        R = torch.from_numpy(R)
        
        src_points = torch.cat((src_points, src_normals), dim = 0)
        target_points = torch.cat((target_points, target_normal), dim = 0)
        # return source point cloud and transformed (target) point cloud 
        return (src_points, target_points, R, t)

        
if __name__ == "__main__":
    root = './data/modelnet40_normal_resampled/'
    category = 'airplane/'
    split="train"
    index=0
    data = ModelNet40Dataset(root=root,category=category,augment=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)
    for src, target, R in DataLoader:
        print('Source:',  src.shape)
        print('Target:',  target.shape)
        print('R', R.shape)
        
