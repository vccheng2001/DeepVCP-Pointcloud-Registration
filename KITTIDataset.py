import os
from utils import *
from torch.utils.data import Dataset, DataLoader

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

        # path to pointclouds + poses
        path = f"{self.root}sequences/00/velodyne/"

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
