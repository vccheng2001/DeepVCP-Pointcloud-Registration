
import torch 
from knn_cuda import KNN
import numpy as np
from torch import nn
''' 
singular value decomposition step to estimate
rotation R given corresponding keypoint pairs {xi, yi}

@param  x: BxNx3 source points 
        y: BxNx3 transformed source points
@return R: Bx3x3 calculated rotation matrix 
        t: Bx1x3 calcualted translation
'''
def get_rigid_transform(x, y):

    # centroid of each point cloud: B x 1 x 3
    centroid_x = torch.mean(x, dim=1, keepdim=True) 
    centroid_y = torch.mean(y, dim=1,  keepdim=True)

    # dist of each point from centroid: B x N x 3
    dist_x = torch.sub(x, centroid_x)
    dist_y = torch.sub(y, centroid_y)
    # transposed dist x: permute to get B x 3 x N
    dist_x_T = dist_x.permute(0,2,1)

    # covariance matrix H
    # (B x 3 x N) * (B x N x 3) => H: B x 3 x 3
    H = torch.matmul(dist_x_T, dist_y)

    # Singular value decomposition of covariance matrix H = USV^T 
    # u:(B x 3 x 3), s:(B x 3), vt:(B x 3 x 3)
    u,s,vT = torch.linalg.svd(H)

    v = vT.permute(0,2,1)
    uT = u.permute(0,2,1)

    # determine whether we need to correct rotation matrix to ensure 
    # right-handed coordinate system
    Z = torch.eye(u.shape[-1]).to(device) #  B x 3 x 3
    Z = Z.unsqueeze(0).repeat(B,1,1)
    uT = u.permute(0,2,1)
    # for each batch, set last element on diagonal to d
    d = torch.sign(torch.det(torch.matmul(v, uT)))
    Z[:,-1,-1] = d

    # solve for rotation matrix: B x 3 x 3 
    R = torch.matmul(v, torch.matmul(Z, uT))
    # solve for translation: (Bx1x3)- (Bx3x3)*(Bx1x3) => t:(Bx1x3)
    t = torch.sub(centroid_y, torch.matmul(centroid_x, R))
    return R, t


'''deepVCP loss function: performs two svd 
    optimizations to refine transform, with outlier rejection

@param  x: BxNx3 source points 
        y_pred: BxNx3 transformed source points
        y_true: ground truth y 
        R_true: Bx3x3 ground truth rotation
        t_true: Bx1x3 ground truth translation 
@return R2: Bx3x3 calculated rotation matrix 
        t2: Bx1x3 calculated translation
'''

def svd_optimization(x, y_pred, y_true, R_true, t_true):

    # 1. first SVD to get rotation, translation
    R1, t1 = get_rigid_transform(x, y_pred)    
    # predicted y points based on R1,t1: B x N x 3       
    y_pred1 = torch.matmul(x,R1) + t1

    # 2. outlier rejection where K = 1
    knn = KNN(k=1, transpose_mode=True)
    # dist: B x N x K, index: B x N x K
    dist, index = knn(y_pred1.to(device), y_true.to(device))
    # filter out points whose distance > threshold (****)
    y_pred1 = y_pred1[dist[:,:,0] < THRESHOLD].unsqueeze(0)
    x1 = x[dist[:,:,0] < THRESHOLD].unsqueeze(0)

    # 3. second SVD to refine rotation, translation
    R2, t2 = get_rigid_transform(x1, y_pred1)    
    # predicted y points based on R2, t2     
    y_pred2 = torch.matmul(x,R2) + t2
    # return final rotation (Bx3x3), translation (Bx1x3)
    
    return R2, t2 


   
'''
Combine L1 loss function with 
@param  x: BxNx3 source points 
        y_pred: BxNx3 transformed source points
        y_true: ground truth y 
        R_true: Bx3x3 ground truth rotation
        t_true: Bx1x3 ground truth translation 
        alpha:  loss balancing factor
@return R: Bx3x3 calculated rotation matrix 
        t: Bx1x3 calcualted translation
'''

def deepVCP_loss(x, y_pred, y_true, R_true, t_true, alpha):
    # l1 loss
    loss1 = nn.L1Loss(reduction="mean") # sums and divides by N
    R, t = svd_optimization(x, y_pred, y_true, R_true, t_true)

    # svd loss
    print(f'Final Rotation: {R.shape}')
    print(f'Final Translation: {t.shape}')
    loss2 = torch.mean(torch.sub(y_pred, torch.matmul(x,R) + t))

    # combine loss
    loss = alpha * loss1(x,y_pred) + (1-alpha) * loss2 

    print(f"Loss: {loss}")
    return loss



# Testing
 
THRESHOLD = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 10
B = 5
alpha = 0.5
torch.manual_seed(0)
print(f'Using batch size: {B}, number of keypoints: {N}')


# original source keypoints: BxNx3
x = torch.randn(B,N,3).to(device)        
# output predicted points from previous layers of deepVCP
y_pred = torch.randn(B,N,3).to(device) 

# ground truth rotation: Bx3x3, ground truth translation: Bx1x3
R_true = torch.randn(B,3,3).to(device)   
t_true = torch.zeros(B,N,3).to(device)

# ground truth y: (BxNx3)@(Bx3x3) + (Bx1x3) => BxNx3
y_true = torch.matmul(x,R_true) + t_true  

# get deepVCP loss
loss = deepVCP_loss(x, y_pred, y_true, R_true, t_true, alpha)