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

# singular value decomposition step to estimate
# relative transformation given corresponding keypoint pairs {xi, yi}
def svd():
    pass



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

        for n_batch, (in_batch, label) in enumerate(train_loader):
            # mini batch
            in_batch, label = in_batch.to(device), label.to(device)
            output_pts = model(in_batch, in_batch)
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
    main()