import os
from utils import *
from torch.utils.data import Dataset, DataLoader

'''
Loads poses as np array
Each pose contains top 12 points of of HT only

 | t00 t01 t02 t03 | 
 | t10 t11 t12 t13 |
 | t20 t21 t22 t23 |
 |  0   0   0   1  | <- not included 

@param  file: path to 'poses.txt' file
@return poses: Nx12
'''
def load_poses(file):
    pose = np.loadtxt(file)
    assert(pose.shape[1] == 12)
    return pose


'''
Transforms src points using pose from 'pose.txt'
1) converts pose in pose.txt into a 4x4 HT
2) transforms src points 
@param src
       pose 
@return transformed: 
'''
def ht_transform(src, pose):
    # create HT matrix: 4x4 
    ht = np.vstack((pose.reshape(3,4), [0,0,0,1]))
    
    # appends col of 1s to make dim 4
    src = np.c_[src, np.ones(src.shape[0])]     # Nx3 -> Nx4

    # transform 
    transformed = ht @ src.T                    # 4x4 x 4xN = 4xN 
    print(ht.shape,'ht')
    print(src.shape,'src')
    transformed = transformed.T[:,:-1]          # keep N x 3 
    return transformed

''' 
Downsample point cloud to N points 
@params  src: original source point cloud
           N: number of points desired
@returns src: sampled point cloud
'''
def downsample(src, N=5000):
    num_src = src.shape[0]
    src_downsample_indices = np.arange(num_src)
    if num_src > N:
        src_downsample_indices = np.random.choice(num_src, N, replace=False)
    return src[src_downsample_indices,:]

class KITTIDataset(Dataset):
    def __init__(self, root, augment=True, rotate=True, split="train", N=5000):
        self.root = root
        self.split = split
        self.augment = augment
        self.N = N
        self.files = []
        self.points = []
        self.reflectances = []

        # path to pointclouds + poses (using sequence 00 for now)
        path = f"{self.root}/{split}/sequences/00/velodyne/"
        pose_path = f"{self.root}/{split}/sequences/"

        # load poses 
        poses = load_poses(pose_path + "poses.txt")

        for file in os.listdir(path):
            print(f'\nProcessing file:...', file)
            # get matching file 
            index = int(file.split(".")[0])

            # load point clouds (N x 4)
            src = np.fromfile(path + file, dtype=np.float32, count=-1).reshape([-1,4])
            print("Raw number of points...: ", src.shape)

            # downsample if num points > N
            src = downsample(src, self.N)                           # N x 4
            print('Num points after downsampling..', src.shape)

            # split into xyz, reflectances
            src_points = src[:, :3]                                 # N x 3
            src_reflectance = np.expand_dims(src[:,-1], axis=1)     # N x 1

            # make pose HT and transform points
            src_points = ht_transform(src_points, poses[index])
            print('After transform', src_points.shape)
            
            self.files.append(file)
            self.points.append(src_points)
            self.reflectances.append(src_reflectance)

        print('# Total clouds', len(self.points))


    def __len__(self):
        return len(self.points)


    def __getitem__(self, index):
        # source pointcloud 
        src_points = self.points[index].T                   # 3 x N
        src_reflectance = self.reflectances[index].T        # 1 x N
        print("Loading file: ", self.files[index])

        # data augmentation
        if self.augment:
            # generate random angles for rotation matrices 
            theta_x = np.random.uniform(0, np.pi*2)
            theta_y = np.random.uniform(0, np.pi*2)
            theta_z = np.random.uniform(0, np.pi*2)

            # generate random translation
            translation_max = 1.0
            translation_min = 0.0
            t = np.random.uniform(translation_min,translation_max, (3, 1))
 
            # Generate target point cloud by doing a series of random
            # rotations on source point cloud 
            Rx = RotX(theta_x)
            Ry = RotY(theta_y)
            Rz = RotZ(theta_z)
            R = Rx @ Ry @ Rz

            # rotate source point cloud
            target_points = R @ src_points + t
        
        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points)
        src_reflectance = torch.from_numpy(src_reflectance)
        R = torch.from_numpy(R)
        
        # return source point cloud and transformed (target) point cloud 
        # src, target: B x 3 x N, reflectance : B x 1 x N 
        return (src_points, target_points, R, t, src_reflectance)

if __name__ == "__main__":
    data = KITTIDataset(root='./data/KITTI', N=5000, augment=True, split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False) 
    for src, target, R, t, src_reflectance in DataLoader:
        print('Source:',  src.shape)                # B x 3 x N 
        print('Target:',  target.shape)             # B x 3 x N
        print('R', R.shape)                     
        print('Reflectance', src_reflectance.shape) # B x 1 x N 
