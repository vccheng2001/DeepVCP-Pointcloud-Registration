import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import argparse 

from ModelNet40Dataset import ModelNet40Dataset
from KITTIDataset import KITTIDataset
from deepVCP import DeepVCP
from deepVCP_loss import deepVCP_loss

'''
Visualize pointcloud 
@param: list of point clouds, each is Nx3 
@output: visualization 
'''

VIS_PATH = "./vis/"

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="modelnet", help='dataset (specify modelnet or kitti)')
parser.add_argument('-m', '--model_path', default="final_model.pt", action = "store", type = str, help='specify path to load model')

args = parser.parse_args()
dataset = args.dataset
model_path = args.model_path

def draw(point_clouds): # N x 3
    pc_all = o3d.geometry.PointCloud()

    # draw both ground truth and predicted point cloud
    for i, points in enumerate(point_clouds):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        if (i == 0):
            pc.paint_uniform_color(np.array([[1], [0], [0]]))
        else:
            pc.paint_uniform_color(np.array([[0], [0], [1]]))
        pc = pc.voxel_down_sample(voxel_size = 0.01)
        pc_all += pc
        if i == 1:
            o3d.visualization.draw_geometries([pc_all])
    print("finished")
    print(pc)


    if not os.path.exists(VIS_PATH):
        os.makedirs(VIS_PATH)
    o3d.io.write_point_cloud(os.path.join(VIS_PATH,"vis.pcd"), pc_all)


def save_cloud():
    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # load model
    use_normal = False if dataset == "kitti" else True
    model = DeepVCP(use_normal=use_normal)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    batch_size = 1

    # dataset
    if dataset == "modelnet":
        root = '/home/zheruiz/datasets/modelnet40_normal_resampled/'
        test_data = ModelNet40Dataset(root=root, split='test')
    elif dataset == "kitti":
        root = '/data/dataset/'
        test_data = KITTIDataset(root=root, N=10000, augment=True, split="test")
    
    # set up dataloader on test set
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for n_batch, (src, target, R_gt, t_gt) in enumerate(test_loader):
            pointclouds = []
            transformation = []
            # mini batch
            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model(src, target, R_gt, t_init)

            loss, R_pred, t_pred = deepVCP_loss(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)
            # error metric for rigid body transformation
            # r_pred = R.from_matrix(R_pred.squeeze(0).cpu().detach().numpy())
            # r_pred_arr = r_pred.as_euler('xyz', degrees=True).reshape(1, 3)
            
            src_np = src[:, :3, :].cpu().detach().numpy().squeeze(0)
            target_np = target[:, :3, :].cpu().detach().numpy().squeeze(0)
            R_pred_np = R_pred.cpu().detach().numpy().reshape(3, 3)
            t_pred_np = t_pred.cpu().detach().numpy().reshape(3, 1)
            target_pred_np = R_pred_np @ src_np + t_pred_np

            # save the numpy array for visualization
            np.save(VIS_PATH + str(n_batch) + "_gt.npy", target_np.T)
            np.save(VIS_PATH + str(n_batch) + "_pred.npy", target_pred_np.T)
            print("Point cloud saved.")
            
            pointclouds.append(target_pred_np.T)
            pointclouds.append(target_np.T)
            # draw(pointclouds)

def main():
    save_cloud()
    curr_dir = os.path.dirname(__file__)
    if dataset == "kitti":
        path = os.path.join(curr_dir, "velodyne/")
        N = 10000
        point_clouds = []
        for file in os.listdir(path):
            print('Processing:', file)
            src = np.fromfile(path + file, dtype=np.float32, count=-1).reshape([-1,4])
            src_points = src[:, :3]                                 # N x 3
            point_clouds.append(src_points)
        draw(point_clouds)
    elif dataset == "modelnet":
        path = os.path.join(curr_dir, VIS_PATH)
        num_clouds = int(len([name for name in os.listdir(path)]) / 2)
        for file_id in range(num_clouds):
            point_clouds = []
            target_gt = np.load(path + str(file_id) + "_gt.npy").reshape([-1, 3])
            target_pred = np.load(path + str(file_id) + "_pred.npy").reshape([-1, 3])
            point_clouds.append(target_gt)
            point_clouds.append(target_pred)
            draw(point_clouds)



if __name__ == "__main__":    
    main()
