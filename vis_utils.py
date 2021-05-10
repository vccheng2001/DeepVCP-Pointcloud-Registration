import os
import open3d as o3d
import numpy as np

'''
Visualize pointcloud 
@param: list of point clouds, each is Nx3 
@output: visualization 
'''

VIS_PATH = "./vis"

def draw(point_clouds): # N x 3
    pc_all = o3d.geometry.PointCloud()

    # 
    for points in point_clouds:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc = pc.voxel_down_sample(voxel_size = 1)
        pc_all += pc
    o3d.visualization.draw_geometries([pc_all])

    if not os.path.exists(VIS_PATH):
        os.makedirs(VIS_PATH)
    o3d.io.write_point_cloud(os.path.join(VIS_PATH,"vis.pcd"), pc_all)


if __name__ == "__main__":    

    path = "./velodyne/"
    N = 5000
    point_clouds = []
    for file in os.listdir(path):
        print('Processing:', file)
        src = np.fromfile(path + file, dtype=np.float32, count=-1).reshape([-1,4])
        src_points = src[:, :3]                                 # N x 3
        point_clouds.append(src_points)
    draw(point_clouds)

