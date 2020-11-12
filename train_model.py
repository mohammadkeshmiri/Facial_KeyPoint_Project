# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

###################
## TODO: Define the Net in models.py

import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F

import torch.cuda

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import Net

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim

def train_net(n_epochs, model):
    # prepare the net for training

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    
    training_loss = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            images, key_pts = images.to(device), key_pts.to(device)
            
            # forward pass to get outputs
            output_pts = model(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += float(loss.item())
            training_loss.append(float(loss.item()))
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss))
                running_loss = 0.0

        if epoch % 5 == 4:
            saved_models()

    print('Finished Training')
    return training_loss

def saved_models():
    
    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)
    print('Model saved')

# TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

# load training data in batches
batch_size = 20
learningRate = 0.001
n_epochs = 30 # start small, and increase when you've decided on your model structure and hyperparams
net = Net()

model_dir = 'saved_models/'
model_name = 'keypoints_model.pt'

criterion = nn.MSELoss()
# # criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=learningRate)

torch.cuda.empty_cache()

train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

# train your network

training_loss = train_net(n_epochs, net)

epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()