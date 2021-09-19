import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.spatial.transform import Rotation as R
import time
import pickle
import argparse 
from utils import *

import wandb
from deepVCP import DeepVCP
from ModelNet40Dataset import ModelNet40Dataset
from KITTIDataset import KITTIDataset
from deepVCP_loss import deepVCP_loss

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main(cfg):

    if cfg.logger:
        # start a new wandb run for experiment tracking
        tags = [cfg.dataset]
        wandb.init(project='deepvcp',
                entity='vccheng2001',
                config=cfg,
                tags=tags)
        wandb.run.name = f'{cfg.dataset}'




    # hyper-parameters
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    lr = cfg.learning_rate
    alpha = cfg.alpha

    print(f"Params: {cfg}")

    # check if cuda is available
    device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # Use multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    # dataset 
    if cfg.dataset == "modelnet":
        root = '/home/vivian_cheng/datasets/modelnet40_normal_resampled/'

        # root = '/home/local/SWNA/chenv022/Projects/datasets/modelnet40_normal_resampled/'
        shape_names = np.loadtxt(root+"modelnet10_shape_names.txt", dtype="str")
        train_data= ModelNet40Dataset(root=root, augment=True, full_dataset=cfg.full_dataset, split='train')
        test_data = ModelNet40Dataset(root=root, augment=True, full_dataset=cfg.full_dataset,  split='test')
    elif cfg.dataset == "kitti":
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

    if cfg.logger:
        # Log gradients and model parameters
        wandb.watch(model)

    # Retrain
    if retrain_path:
        print("Retrain on ", cfg.retrain_path)
        model.load_state_dict(torch.load(cfg.retrain_path))
    else:
        print("No retrain")

    # Define the optimizer
    optim = Adam(model.parameters(), lr=lr)

    # Begin training
    model.train()
    loss_epoch_avg = []
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0
        
        for n_batch, (src, target, R_gt, t_gt, ) in enumerate(train_loader):
            start_time = time.time()
            # mini batch
            src, target, R_gt, t_gt = src, target, R_gt, t_gt#src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model(src, target, R_gt, t_init)
            # print('src_keypts shape', src_keypts.shape)
            # print('target_vcp shape', target_vcp.shape)
            # zero gradient 
            optim.zero_grad()
            loss, R_pred, t_pred = deepVCP_loss(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)

            # error metric for rigid body transformation
            r_pred = R.from_matrix(R_pred.squeeze(0).cpu().detach().numpy())
            r_pred_arr = torch.tensor(r_pred.as_euler('xyz', degrees=True)).reshape(1, 3)
            r_gt = R.from_matrix(R_gt.squeeze(0).cpu().detach().numpy())
            r_gt_arr = torch.tensor(r_gt.as_euler('xyz', degrees=True)).reshape(1, 3)
            pdist = nn.PairwiseDistance(p = 2)
            

            train_rotation_error = pdist(r_pred_arr, r_gt_arr).item()
            train_translation_error = pdist(t_pred, t_gt).item()
            print("rotation error: ", train_rotation_error)
            print("translation error: ", train_translation_error)

            if cfg.logger:

                wandb.log({"train/rotation_error": train_rotation_error})
                wandb.log({"train/translation_error": train_translation_error})
                wandb.log({"train/loss": loss.item()})

            # backward pass
            loss.backward()
            # update parameters 
            optim.step()

            running_loss += loss.item()
            loss_epoch += [loss.item()]
            print("--- %s seconds ---" % (time.time() - start_time))
            if (n_batch + 1) % 5 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

                
                running_loss = 0.0

            
        
        torch.save(model.state_dict(), "epoch_" + str(epoch) + "_model.pt")
        loss_epoch_avg += [sum(loss_epoch) / len(loss_epoch)]
        with open("training_loss_" + str(epoch) + ".txt", "wb") as fp:   #Pickling
            pickle.dump(loss_epoch, fp)
        

    # save 
    print("Finished Training")
    torch.save(model.state_dict(), cfg.model_path)
    
    # begin test 
    model.eval()
    loss_test = []
    with torch.no_grad():
        for n_batch, (src, target, R_gt, t_gt) in enumerate(test_loader):
            # mini batch
            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model(src, target, R_gt, t_init)

            loss, R_pred, t_pred = deepVCP_loss(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)
            # error metric for rigid body transformation
            r_pred = R.from_matrix(R_pred.squeeze(0).cpu().detach().numpy())
            r_pred_arr = torch.tensor(r_pred.as_euler('xyz', degrees=True)).reshape(1, 3)
            r_gt = R.from_matrix(R_gt.squeeze(0).cpu().detach().numpy())
            r_gt_arr = torch.tensor(r_gt.as_euler('xyz', degrees=True)).reshape(1, 3)
            pdist = nn.PairwiseDistance(p = 2)
            
            test_rotation_error = pdist(r_pred_arr, r_gt_arr).item()
            test_translation_error = pdist(t_pred, t_gt).item()
            print("rotation error test: ", test_rotation_error)
            print("translation error test: ", test_translation_error)

            if cfg.logger:
                wandb.log({"test/loss": loss.item()})
                wandb.log({"test/rotation_error":test_rotation_error})
                wandb.log({"test/translation_error":test_translation_error})

            loss_test += [loss.item()]

    with open("test_loss.txt", "wb") as fp_test:   #Pickling
        pickle.dump(loss_test, fp_test)
    print("Test loss:", loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default="modelnet", help='dataset (specify modelnet or kitti)')
    parser.add_argument("-f", "--full_dataset", default=True, help="full dataset", action='store_true')
    parser.add_argument("-l", "--logger", default=True, help="use logger", action='store_true')
    parser.add_argument('-r', '--retrain_path', action = "store", type = str, help='specify a saved model to retrain on')
    parser.add_argument('-m', '--model_path', default="final_model.pt", action = "store", type = str, help='specify path to save final model')
    parser.add_argument('-ep', '--num_epochs', default=10, type = int, help='num epochs to train')
    parser.add_argument('-bs', '--batch_size', default=1, type = int, help='batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type = float, help='learning rate')
    parser.add_argument('-a', '--alpha', default=0.5,  type = float, help='loss balancing factor')
    cfg = parser.parse_args()

    main(cfg)
