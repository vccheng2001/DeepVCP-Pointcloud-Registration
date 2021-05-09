import torch 
from torch import nn
from knn_cuda import KNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
obtain rotation and translation using single
value decomposition
@param x: Bx3xN keypoints of source point cloud 
       y: Bx3xN corresponding transformed points 
'''
def get_rigid_transform(x, y):
    B, _, N = x.shape
    # centroid of each point cloud: B x 3 x 1 
    centroid_x = torch.mean(x, dim=2, keepdim=True) 
    centroid_y = torch.mean(y, dim=2,  keepdim=True)

    # dist of each point from centroid: B x 3 x N 
    dist_x = torch.sub(x, centroid_x)
    dist_y = torch.sub(y, centroid_y)

    # covariance matrix H
    # (B x 3 x N) * (B x N x 3) => H: B x 3 x 3
    H = torch.matmul(dist_x, dist_y.permute(0,2,1))

    # Singular value decomposition of covariance matrix H = USV^T 
    # u:(B x 3 x 3), s:(B x 3), vt:(B x 3 x 3)
    u,s,v = torch.svd(H)
    uT = u.permute(0,2,1) # B x 3 x 3 
    # v = vT.permute(0,2,1)
    R = torch.matmul(v, uT)

    #determine whether we need to correct rotation matrix to ensure 
    #right-handed coordinate system
    Z = torch.eye(u.shape[-1]).to(device) #  B x 3 x 3
    Z = Z.unsqueeze(0).repeat(B,1,1)
    # for each batch, set last element on diagonal to d
    d = torch.sign(torch.det(torch.matmul(v, uT)))
    Z[:,-1,-1] = d

    # solve for translation: Bx3xN
    t = centroid_y + torch.matmul(-R, centroid_x)
    return R, t


'''deepVCP loss function: performs two svd 
    optimizations to refine transform, with outlier rejection

@param  x: Bx3xN source points 
        y_pred: Bx3xN transformed source points
        R_true: Bx3x3 ground truth rotation
        t_true: Bx3xN ground truth translation 
@return R2: Bx3x3 calculated rotation matrix 
        t2: Bx3xN calculated translation
'''
def svd_optimization(x, y_pred, R_true, t_true):

    # ground truth y_true: Bx3xN
    y_true = torch.matmul(R_true, x) + t_true
    y_pred = y_pred.double()
    B, _, N = y_pred.shape

    # first SVD to get rotation, translation
    R1, t1 = get_rigid_transform(x, y_pred) # R: Bx3x3, t: Bx3x1

    y_pred1 = torch.matmul(R1, x) + t1       # Bx3xN

    # get 1-nearest neighbor, outlier rejection
    knn = KNN(k=1, transpose_mode=False)
    
    dist, _ = knn(y_pred1, y_true)          # BxKxN
    dist = dist.to(device)

    # eliminate 20% outliers (keep 80% points with smallest 1-NN distance)
    num_inliers = int(N*0.8)
    inliers = torch.topk(dist, k=num_inliers, dim=-1,\
                     largest=False, sorted=True).indices
    inliers = inliers.repeat(1,3,1).to(device)

    y_pred1 = torch.gather(y_pred1, dim=-1, index = inliers) # Bx3xN'
    x1 = torch.gather(x, dim=-1, index = inliers)            # Bx3xN'

    # second SVD to refine rotation, translation
    R2, t2 = get_rigid_transform(x1, y_pred1)    
   
    # predicted y points based on R2, t2   
    y_pred2 = torch.matmul(R2, x1) + t2

    return R2, t2 , x1, y_pred2


   
'''
Combine L1 loss function with 
@param  x: BxNx3 source points 
        y_pred: BxNx3 transformed source points
        R_true: Bx3x3 ground truth rotation
        t_true: Bx3xN ground truth translation 
        alpha:  loss balancing factor
@return R: Bx3x3 calculated rotation matrix 
        t: Bx3xN calcualted translation
'''

def deepVCP_loss(x, y_pred, R_true, t_true, alpha):
    x = x.permute(0, 2, 1).double()
    y_pred = y_pred.permute(0, 2, 1).double()

    # l1 loss
    loss1 = nn.L1Loss(reduction="mean") 

    # svd loss
    R, t, x_inliers, y_pred_optimized = svd_optimization(x, y_pred, R_true, t_true)
    y_true_inliers = torch.matmul(R_true, x_inliers) + t_true

    loss2 = torch.abs(torch.mean(torch.sub(y_pred_optimized, y_true_inliers)))

    # combine loss
    loss = alpha * loss1(y_true_inliers, y_pred_optimized) + (1 - alpha) * loss2 
    print(f"Loss: {loss}")
    return loss, R, t

if __name__ == "__main__":
    THRESHOLD = 5
    N = 4
    B = 2
    alpha = 0.5
    torch.manual_seed(0)
    # print(f'Using batch size: {B}, number of keypoints: {N}')
    # original source keypoints: BxNx3
    x = torch.randn(B,3,N).to(device) 
    # print('x',x)    
    # output predicted points from previous layers of deepVCP
    y_pred = torch.randn(B,3,N).to(device)

    # 30 degree rotation 
    R_true = torch.Tensor([[[  0.7500000, -0.4330127,  0.5000000],
    [0.6495190,  0.6250000, -0.4330127],
    [-0.1250000,  0.6495190,  0.7500000 ]]])

    R_true = R_true.repeat(B,1,1).to(device)
    t_true = torch.zeros(B,3,1).repeat(1,1,N).to(device)

    # print("Ground truth R:", R_true)
    # print("Ground truth t:", t_true)


    # get deepVCP loss
    deepVCP_loss(x, y_pred, R_true, t_true, alpha)
