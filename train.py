import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import ModelNet40DataLoader

''' note: path to dataset is ./data/modelnet40_normal_resampled
    from https://modelnet.cs.princeton.edu/ '''

def main():
    # hyper-parameters
    num_epochs = 50
    batch_size = 16
    lr = 0.001
    num_train = len(train_dataset)

    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")


    # Load ModelNet40 data 
    print('Loading Model40 dataset ...')

    root = 'data/modelnet40_normal_resampled/'
    category = "airplane"

    train_data= ModelNet40DataLoader(root=root, category=category, split='train')
    test_data = ModelNet40DataLoader(root=data_path, category=category, split='train')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print('Train dataset size: ', len(train_dataset))
    print('Test dataset size: ', len(valid_dataset))

    
    # Initialize the model
    model = MLP() # CHANGE THIS 
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = Adam(mlp.parameters(), lr=lr)

    # begin train 
    model.train()

    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")

        running_loss = 0.0

        for n_batch, (in_batch, label) in enumerate(train_loader):
            # mini batch
            in_batch, label = in_batch.to(device), label.to(device)        
            # zero gradient 
            optim.zero_grad()
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