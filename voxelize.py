import os
import numpy as np
import torch
from torch import nn
from utils import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# defines a voxels with a center, edge length, and vertices 
class Voxel():
    def __init__(self, center, length):
        self.center = center 
        self.cx = center[0]
        self.cy = center[1]
        self.cz = center[2]
        self.length = length # edge length
        self.center
        # 8 coordinates defining cube vertices
        self.vertices = np.asarray(cart_prod([[self.cx-length/2, self.cx+length/2],
                                  [self.cy-length/2, self.cy+length/2],
                                  [self.cz-length/2, self.cz+length/2]]))

''' Voxelizing neighborhood space 

    Given search radius r, divide search space into (2r/s + 1, 2r/s + 1, 2r/s + 1) voxels

    1) create bounding box (of type Voxel) around sphere with radius r
    2) voxelize space within bounding box
    3) measure Euclidean distance of each voxel center to origin
       if voxel center falls outside of search radius, reject

    @param xp_i: a transformed point in target point cloud xp
           search radius of neighborhood 
           voxel_len: edge length of each individual voxel
'''
def voxelize(xp_i, search_radius, voxel_len):
    # enclose search space in bounding box (think of it as a voxel of edge length 2r)
    search_space = Voxel(xp_i, 2*search_radius)     
    ss_vertices = search_space.vertices

    # min, max xyz coords of bounding box
    ss_minx, ss_miny, ss_minz  = np.min(ss_vertices, axis=0)
    ss_maxx, ss_maxy, ss_maxz = np.max(ss_vertices, axis=0)

    # number of voxels along each axes
    num_voxels = np.ceil((np.max(ss_vertices, axis=0) - np.min(ss_vertices, axis=0))/voxel_len)
    print(f"voxelize into {num_voxels} voxels along each axis ")
    # store voxels in sparse representation 
    voxels = {} 
    x,y,z = [], [], []

    # voxelize search space/cube 
    for xi in np.arange(ss_minx, ss_maxx, voxel_len):
        for yi in np.arange(ss_miny,ss_maxy,voxel_len):
            for zi in np.arange(ss_minz,ss_maxz,voxel_len):
                # define center of voxel
                cx,cy,cz = (xi+voxel_len/2, yi+voxel_len/2, zi+voxel_len/2)
                # create voxel object at center point with radius <voxel_len>
                vox = Voxel([cx,cy,cz], 2*voxel_len)
                # store voxel in dictionary
                voxels[vox] = 1
                
                # for graphing purposes only
                x.append(cx)
                y.append(cy)
                z.append(cz)


    # plot voxel centers
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=1)

    # plot bounding box vertices
    ax.scatter(ss_vertices[:,0], ss_vertices[:,1], ss_vertices[:,2], cmap='Greens', linewidth=1);
    plt.show()

    # measure distance from each voxel to center of search space
    # reject voxels whose distance >= search radius
    print(f'Search radius: {search_radius}\n')
    for vox in voxels.keys():
        dist = euclidean_dist(vox.center, search_space.center)
        if dist > search_radius: 
            print(f'Rejecting point {vox.center} which is {dist} from center')
            voxels[vox] = 0 

    voxels = {v for v in voxels.items() if v[1] != 1}
    return voxels


if __name__ == "__main__":
    # ** Using placeholder numbers for nwo

    r = 9   # search radius
    s = 2.5 # voxel size
    xp_i = [1,0.2,3] # ith point in transformed point cloud
    voxelize(xp_i, r, s)