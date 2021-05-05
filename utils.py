''' Utility functions '''
import itertools
import torch
import numpy as np
import math 

# rotation about x axis
def RotX(theta):
    Rx = np.matrix([[ 1,            0           , 0     ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
    return Rx
  
# rotation about y axis
def RotY(theta):
    Ry =  np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1,          0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
    return Ry

# rotation about z axis
def RotZ(theta):
    Rz = np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
    return Rz

# euclidean distance between two points
def euclidean_dist(a,b):
    return torch.linalg.norm(a-b)
