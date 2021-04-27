''' Utility functions '''
import itertools
import numpy as np
import math 

# rotation about x axis
def rotateX(theta, obj):
    rX = np.matrix([[ 1,            0           , 0     ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
    return rX.dot(obj)
  
# rotation about y axis
def rotateY(theta, obj):
    rY =  np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1,          0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
    return rY.dot(obj)

# rotation about z axis
def rotateZ(theta, obj):
    rZ = np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
    return rZ.dot(obj)

# cartesian product of N arrays 
def cart_prod(arrs):
    return list(itertools.product(*arrs))

# euclidean distance between two points
def euclidean_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)
