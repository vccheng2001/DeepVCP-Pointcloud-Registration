import os
from utils_kitti.pointcloud import make_point_cloud, estimate_normal
from utils_kitti.SE3 import *
from utils import *
from torch.utils.data import Dataset, DataLoader

class KITTIDataset(Dataset):
    def __init__(self, root, augment=True, rotate=True, split="train", N=5000):
        self.root = root
        self.split = split
        self.augment = augment
        self.points = []
        self.N = N
        self.files = []

        path = f"{self.root}/{split}/sequences/00/velodyne/"
        for i, file in enumerate(os.listdir(path)):
            print('file', file)
            points = np.fromfile(path+file, dtype=np.float32, count=-1).reshape([-1,4])
            self.points.append(points)
            self.files.append(file)

        print('# Total clouds', len(self.points))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # source pointcloud 
        src = self.points[index]
        print("Processing file: ", self.files[index])
        num_src = src.shape[0]          # num points 
        print('Number of points in raw cloud', num_src)

        # randomly subsample N points
        src_sel_ind = np.arange(num_src)
        if num_src > self.N:
            src_sel_ind = np.random.choice(num_src, self.N, replace=False)
        src = src[src_sel_ind, :]

        # split into xyz coords, reflectance 
        src_points= src[:, :3]                   # N x 3
        src_reflectance = src[:,-1]               # N x 1
        print('src_points', src_points.shape)
        
        # data augmentation
        if self.augment:
            # generate random angles for rotation matrices 
            theta_x = np.random.uniform(0, np.pi*2)
            theta_y = np.random.uniform(0, np.pi*2)
            theta_z = np.random.uniform(0, np.pi*2)

            # generate random translation
            translation_max = 1.0
            translation_min = 0.0
            t = (translation_max - translation_min) * torch.rand(3, 1) + translation_min
 
            # Generate target point cloud by doing a series of random
            # rotations on source point cloud 
            Rx = RotX(theta_x)
            Ry = RotY(theta_y)
            Rz = RotZ(theta_z)
            R = Rx @ Ry @ Rz

            src_points = src_points.T
            # rotate source point cloud
            target_points = R @ src_points
        
        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points)

        R = torch.from_numpy(R)
        
        # return source point cloud and transformed (target) point cloud 
        # src, target: B x 3 x N
        # reflectance : B x 1 x N 
        return (src_points, target_points, R, t, src_reflectance)

if __name__ == "__main__":
    data = KITTIDataset(root='./data/KITTI', N=5000, augment=True, split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False) 
    for src, target, R, t, src_reflectance in DataLoader:
        print('Source:',  src.shape)
        print('Target:',  target.shape)
        print('R', R.shape)
        print('Reflectance', src_reflectance.shape)
