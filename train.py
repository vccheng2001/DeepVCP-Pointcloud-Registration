import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

def main():
    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # define dataset and dataloader
    train_dataset = Dataset(mode='train')
    test_dataset  = Dataset(mode='test')
    train_loader  = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader   = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # hyper-parameters
    num_epochs = 50
    lr = 0.001
    num_train = len(train_dataset)
    
    # Initialize the model
    model = MLP()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss().to(device)
    
    optimizer = Adam(mlp.parameters(), lr=lr)

    # begin train 
    model.train()

    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")

        for n_batch, (in_batch, label) in enumerate(train_loader):
            # mini batch
            in_batch, label = in_batch.to(device), label.to(device)
            # init grads to 0
            optim.zero_grad() 
            model.zero_grad()     
            
            # FILL IN FORWARD PASS

            # FILL IN LOSS 

            # zero gradient 
            optim.zero_grad()
            # backward pass
            loss.backward()
            # update parameters 
            optim.step()

            if (n_batch + 1) % 200 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

    # save 
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