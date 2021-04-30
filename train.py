from deepVCP import DeepVCP
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from ModelNet40Dataset import ModelNet40Dataset
from utils import *

from deep_feat_extraction import feat_extraction_layer

''' note: path to dataset is ./data/modelnet40_normal_resampled
    from https://modelnet.cs.princeton.edu/ '''

''' 
Define loss function 
@params
    y_true:     ground truth y
    x_pred:     predicted xi
    R:          rotation matrix
    T:          translation
    alpha:      loss weights 
'''
def loss_func(y_true, x_pred, R, T, alpha):
    # l1 loss
    loss1 = nn.L1Loss(reduction="mean") # sums and divides by N
    # single optimization iteration 
    loss2 = np.mean(abs(y_true - (R.dot(xi) + T)))
    return alpha * loss1 + (1-alpha) * loss2 

''' 
singular value decomposition step to estimate
rotation R given corresponding keypoint pairs {xi, yi}

@param  x: Nx3 source points 
        y: Nx3 transformed source points
@return R: calculated rotation matrix 

svd(E) = [U,S,V] 
E = USV^T
'''
def get_transformation(x, y):
    x = x.numpy()
    y = y.numpy()
    centroid_x = np.mean(x, axis=1).reshape(-1,1)
    centroid_y = np.mean(y, axis=1).reshape(-1,1)

    H = (x - centroid_x) @ (y - centroid_y).T
    U, S, V = np.linalg.svd(H)

    R = Vt.T @ U.T
    R = torch.from_numpy(R)
    t = -R @ centroid_x + centroid_y
    return R, t
    

def main():
    # hyper-parameters
    num_epochs = 50
    batch_size = 1
    lr = 0.001
    # loss balancing factor 
    alpha = 0.5

    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")


    # Load ModelNet40 data 
    print('Loading Model40 dataset ...')

    root = 'data/modelnet40_normal_resampled/'
    # only use airplane for now
    category = "airplane"
    shape_names = np.loadtxt(root+"modelnet40_shape_names.txt", dtype="str")


    train_data= ModelNet40Dataset(root=root, category=category, split='train')
    test_data = ModelNet40Dataset(root=root, category=category, split='test')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    num_train = len(train_data)
    num_test = len(test_data)
    print('Train dataset size: ', num_train)
    print('Test dataset size: ', num_test)

    
    # Initialize the model
    model = DeepVCP() # CHANGE THIS 
    model.to(device)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # begin train 
    model.train()

    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")

        running_loss = 0.0

        for n_batch, (src, target, R) in enumerate(train_loader):
            # mini batch
            src, target, R = src.to(device), target.to(device), R.to(device)
            print('Source:',  src.shape)
            print('Target:',  target.shape)
            print('R', R.shape)

            pred = model(src)
            print('output of model shape', pred.shape)
            # zero gradient 
            optim.zero_grad()
            loss = loss_func(y_true, x_pred, R, T, alpha)
            # backward pass
            loss.backward()
            # update parameters 
            optim.step()
            
            running_loss += loss.item()
            if (n_batch + 1) % 200 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
                running_loss = 0.0
    # save 
    print("Finished Training")
    torch.save(model.state_dict(), "model.pt")

    # begin test 
    model.eval()
    with torch.no_grad():
        for n_batch, (in_batch, label) in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)

            pred = model.test(in_batch)

            l2_err += loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

    print("Test L2 error:", l2_err)

if __name__ == "__main__":
    A = torch.tensor([[1,0,0], [0,1,0.], [0,0,3.]], dtype=torch.float)
    R0 = torch.tensor([[np.cos(90), -np.sin(90),0], [np.sin(90), np.cos(90),0],[0,0,1]], dtype=torch.float)
    B = (R0.mm(A.T)).T
    t0 = torch.tensor([3., 3.,4])
    print('R0,t0', R0, t0 )

    B += t0
    R, t =get_transformation(A,B)
    print('got R,t', R, t )
    A_aligned = (R.mm(A.T)).T + t
    rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
    