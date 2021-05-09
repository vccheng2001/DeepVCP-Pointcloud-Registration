import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import pickle
import argparse 
from utils import *

from deepVCP import DeepVCP
from ModelNet40Dataset import ModelNet40Dataset
from KITTIDataset import KITTIDataset
from deepVCP_loss import deepVCP_loss

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# setup train 
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="modelnet", help='dataset (specify modelnet or kitti)')
args = parser.parse_args()

def main():
    # hyper-parameters
    num_epochs = 10
    batch_size = 1
    lr = 0.001
    # loss balancing factor 
    alpha = 0.5

    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # dataset 
    if args.dataset == "modelnet":
        root = 'data/modelnet40_normal_resampled/'
        shape_names = np.loadtxt(root+"modelnet40_shape_names.txt", dtype="str")
        train_data= ModelNet40Dataset(root=root, augment=True, split='train')
        test_data = ModelNet40Dataset(root=root, augment=True,  split='test')
    else:
        root = '/data/dataset/'
        train_data= KITTIDataset(root=root, N=10000, augment=True, split="train")
        test_data = KITTIDataset(root=root, N=10000, augment=True, split="test")


    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


    num_train = len(train_data)
    num_test = len(test_data)
    print('Train dataset size: ', num_train)
    print('Test dataset size: ', num_test)

    use_normal = False if dataset == "kitti" else True

    # Initialize the model
    model = DeepVCP(use_normal=use_normal) 
    model.to(device)
    
    # Retrain
    model.load_state_dict(torch.load("final_model.pt"))

    # Define the optimizer
    optim = Adam(model.parameters(), lr=lr)

    # begin train 
    model.train()
    loss_epoch = []
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")

        running_loss = 0.0
        
        for n_batch, (src, target, R_gt, t_gt, ) in enumerate(train_loader):
            start_time = time.time()
            # mini batch
            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            #print('Source:',  src.shape)
            #print('Target:',  target.shape)
            #print('R', R_gt.shape)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model(src, target, R_gt, t_init)
            #print('src_keypts shape', src_keypts.shape)
            #print('target_vcp shape', target_vcp.shape)
            # zero gradient 
            optim.zero_grad()
            loss = deepVCP_loss(src_keypts.double(), target_vcp, R_gt, t_gt, alpha=0.5)
            # backward pass
            loss.backward()
            # update parameters 
            optim.step()
            
            running_loss += loss.item()
            print("--- %s seconds ---" % (time.time() - start_time))
            if (n_batch + 1) % 5 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
                running_loss = 0.0
        
        torch.save(model.state_dict(), "epoch_" + str(epoch) + "_model.pt")
        loss_epoch += [loss.item()]
    # save 
    print("Finished Training")
    torch.save(model.state_dict(), "final_model.pt")
    with open("training_loss.txt", "wb") as fp:   #Pickling
        pickle.dump(loss_epoch, fp)

    # begin test 
    model.eval()
    loss_test = []
    with torch.no_grad():
        for n_batch, (src, target, R_gt, t_gt) in enumerate(train_loader):
            # mini batch
            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model.test(src, target, R_gt, t_init)

            loss = deepVCP_loss(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)

            loss_test += [loss.item()]

    with open("test_loss.txt", "wb") as fp_test:   #Pickling
        pickle.dump(loss_test, fp_test)
    print("Test loss:", loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",  help="dataset (kitti or modelnet)")
    args = parser.parse_args()
    dataset = args.dataset 

    main()
