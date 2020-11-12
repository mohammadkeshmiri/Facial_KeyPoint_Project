#import the usual resources
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

def accuracy(model, test_loader, pct_close):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_correct = 0
    n_items = 0
    for batch_i, data in enumerate(test_loader):
        # get the input images and their corresponding labels
        test_images = data['image']
        correct_key_pts = data['keypoints']

        n_items += len(correct_key_pts)

        # flatten pts
        correct_key_pts = correct_key_pts.view(correct_key_pts.size(0), -1)

        # convert variables to floats for regression loss
        correct_key_pts = correct_key_pts.type(torch.FloatTensor)
        test_images = test_images.type(torch.FloatTensor)

        test_images, correct_key_pts = test_images.to(device), correct_key_pts.to(device)

        # forward pass to get outputs
        predecited_key_pts = model(test_images)

        # #   n_items = len(data_y)
        # #   X = torch.Tensor(data_x)  # 2-d Tensor
        # #   Y = torch.Tensor(data_y)  # actual as 1-d Tensor
        # #   oupt = model(X)       # all predicted as 2-d Tensor
        # #   pred = oupt.view(n_items)  # all predicted as 1-d
        abbsolute_Error = torch.sum(torch.abs(predecited_key_pts - correct_key_pts))
        total_correct += abbsolute_Error.item()
        
    mean_absolute_error = (total_correct  / n_items)  # scalar
    model.train()
    return mean_absolute_error

model_dir = 'saved_models/'
model_name = 'keypoints_model_2.pt'
state_dict = torch.load(model_dir+model_name)

# Test on the test data

# load in the test data, using the dataset class
# AND apply the data_transform you defined above

## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                             root_dir='data/test/',
                                             transform=data_transform)

# load training data in batches
batch_size = 20

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=0)

criterion = nn.SmoothL1Loss()

# initialize tensor and lists to monitor test loss and accuracy
test_loss = torch.zeros(1)

net = Net()
net.load_state_dict(state_dict)

acc = accuracy(net, test_loader, 1)

print('Accuracy : {}\n'.format(acc))