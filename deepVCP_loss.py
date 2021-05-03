device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch 
from torch import nn
from knn_cuda import KNN

'''
obtain rotation and translation using single
value decomposition
@param x: keypoints of source point cloud 
       y: corresponding transformed points 
'''
def get_rigid_transform(x, y):
    print("Getting rigid transform...")
    print('x:', x.shape) # B x 3 x N 
    print('y:', y.shape)

    # centroid of each point cloud: B x 3 x 1 
    centroid_x = torch.mean(x, dim=2, keepdim=True) 
    centroid_y = torch.mean(y, dim=2,  keepdim=True)

    print('centroid_x', centroid_x)
    # dist of each point from centroid: B x 3 x N 
    dist_x = torch.sub(x, centroid_x)
    dist_y = torch.sub(y, centroid_y)

    # covariance matrix H
    # (B x 3 x N) * (B x N x 3) => H: B x 3 x 3
    H = torch.matmul(dist_x, dist_y.permute(0,2,1))

    # Singular value decomposition of covariance matrix H = USV^T 
    # u:(B x 3 x 3), s:(B x 3), vt:(B x 3 x 3)
    u,s,vT = torch.linalg.svd(H)
    uT = u.permute(0,2,1) # B x 3 x 3 
    v = vT.permute(0,2,1)
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
    t = t.repeat(1,1,N)

    return R, t


'''deepVCP loss function: performs two svd 
    optimizations to refine transform, with outlier rejection

@param  x: Bx3xN source points 
        y_pred: Bx3xN transformed source points
        y_true: ground truth y 
        R_true: Bx3x3 ground truth rotation
        t_true: Bx3xN ground truth translation 
@return R2: Bx3x3 calculated rotation matrix 
        t2: Bx3xN calculated translation
'''

def svd_optimization(x, y_pred, y_true, R_true, t_true):

    # 1. first SVD to get rotation, translation
    # R: Bx3x3, t: Bx3xN
    R1, t1 = get_rigid_transform(x, y_true)   

    # Bx3xN
    y_pred1 = torch.matmul(R1,x) + t1

    # 2. outlier rejection where K = 1
    knn = KNN(k=1, transpose_mode=False)
    dist, index = knn(y_pred1, y_true) 

    # filter out points whose distance > threshold 
    # y_pred1 = y_pred1[dist[:,0,:] < THRESHOLD].unsqueeze(1)
    # x1 = x[dist[:,:,0] < THRESHOLD].unsqueeze(1)

    # 3. second SVD to refine rotation, translation
    R2, t2 = get_rigid_transform(x1, y_true)    
   
    # predicted y points based on R2, t2   
    y_pred2 = torch.matmul(R2, x) + t2
    # return final rotation (Bx3x3), translation (Bx1x3)
    return R2, t2 

   
'''
Combine L1 loss function with 
@param  x: Bx3xN source points 
        y_pred: Bx3xN transformed source points
        y_true: ground truth y 
        R_true: Bx3x3 ground truth rotation
        t_true: Bx3xN ground truth translation 
        alpha:  loss balancing factor
@return R: Bx3x3 calculated rotation matrix 
        t: Bx3xN calcualted translation
'''

def deepVCP_loss(x, y_pred, y_true, R_true, t_true, alpha):
    # l1 loss
    loss1 = nn.L1Loss(reduction="mean") 

    # svd loss
    R, t = svd_optimization(x, y_pred, y_true, R_true, t_true)
    print(f'Final Rotation: {R}')    
    print(f'Final Translation: {t}') 
    yi = torch.matmul(R,x) + t
    
    loss2 = torch.mean(torch.sub(y_pred, yi))

    # combine loss
    loss = alpha * loss1(x,y_pred) + (1-alpha) * loss2 
    print(f"Loss: {loss}")
    return loss


THRESHOLD = 5
N = 4
B = 2
alpha = 0.5
torch.manual_seed(0)
print(f'Using batch size: {B}, number of keypoints: {N}')
# original source keypoints: BxNx3
x = torch.randn(B,3,N).to(device) 
print('x',x)    
# output predicted points from previous layers of deepVCP
y_pred = torch.randn(B,3,N).to(device)

# 30 degree rotation 
R_true = torch.Tensor([[[  0.7500000, -0.4330127,  0.5000000],
   [0.6495190,  0.6250000, -0.4330127],
  [-0.1250000,  0.6495190,  0.7500000 ]]])

R_true = R_true.repeat(B,1,1).to(device)
t_true = torch.zeros(B,3,1).repeat(1,1,N).to(device)

print("Ground truth R:", R_true)
print("Ground truth t:", t_true)

# ground truth y: (Bx3x3)@(Bx3xN) => Bx3xN
y_true = torch.matmul(R_true, x) + t_true

# get deepVCP loss
deepVCP_loss(x, y_pred, y_true, R_true, t_true, alpha)