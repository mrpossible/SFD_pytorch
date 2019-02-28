#!/usr/bin/env python
# coding: utf-8

# In[505]:


from __future__ import print_function

import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from numpy import array
from numpy import argmax
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms

import os,sys,cv2,random,datetime,time,math
import pandas as pd
import argparse
import numpy as np
from net_s3fd import *
from bbox import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from PIL import Image


# In[506]:


class CelebDataset(Dataset):
    """Dataset wrapping images and target labels
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """
    def __init__(self, csv_path, img_path, img_ext, transform=None):

        tmp_df = pd.read_csv(csv_path)
#         tmp_df['Label'] = tmp_df['Label'].astype(np.object)        
        
        d1 = tmp_df.dtypes.astype(str).to_dict()
        print(d1)
        
        assert tmp_df['Image_Name'].apply(lambda x: os.path.isfile(img_path + x)).all(),     "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['Image_Name']
#         self.y_train = self.mlb.fit_transform(tmp_df['Label'].str.split())
#         self.y_train = self.mlb.fit_transform(tmp_df['Label'].values).astype(np.float32)
#         self.y_train = self.mlb.fit_transform(tmp_df['Label'].str.split())
#         self.y_train = tmp_df.Label.values
#         self.y_train = self.mlb.fit_transform(self.y_train.str.split())
        self.y_train = tmp_df['Label'].values.astype(np.float32)
    ####
        print(self.y_train.dtype)
              
              
              
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.y_train)
        print('here here')
        print(self.y_train)
        print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        self.y_train = onehot_encoded
        print(onehot_encoded)
        print(self.y_train.shape)
        inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
        print(inverted)
        


    def __getitem__(self, index):
        img = cv2.imread(self.img_path + self.X_train[index])
        print(self.img_path)
        print(self.X_train[index])
        img = cv2.resize(img, (128,128))
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)

        #img = img.reshape((1,)+img.shape)
        img = torch.from_numpy(img).float()
        #img = Variable(torch.from_numpy(img).float(),volatile=True)

        #if self.transform is not None:
        #    img = self.transform(img)
        print(img)
        label = torch.from_numpy(self.y_train[index]).float()
        print(label)
        return img, label

    def __len__(self):
        return len(self.X_train.index)


# In[507]:


# mlb = MultiLabelBinarizer()
# mlb.fit_transform([(0,1), (1,2), (2,3), (3,4)])


# In[508]:


train_data = "/home/user/datasets/img_align_celeba/10_class/final_10_class_train.csv"
img_path = "/home/user/datasets/img_align_celeba/10_class/train/"
img_ext = ".jpg"
dset = CelebDataset(train_data,img_path,img_ext)
train_loader = DataLoader(dset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1, # 1 for CUDA
                          pin_memory =True # CUDA only
                         )
print('here')


# In[509]:


def save(model, optimizer, loss, filename):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
        }
    torch.save(save_dict, filename)


# In[510]:


def train_model(model, criterion, optimizer, num_classes, num_epochs = 100):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        model.train()
        running_loss = 0.0

        for i,(img,label) in enumerate(train_loader):
            img = img.view((1,)+img.shape[1:])
            if use_cuda:
                data, target = Variable(img.type(torch.FloatTensor).cuda()), Variable(torch.Tensor(label).type(torch.FloatTensor).cuda())
            else:
                data, target = Variable(img), Variable(torch.Tensor(label))
            target = target.view(num_classes,1)

            # Make sure to start with zero gradients otherwise, 
            # gradients from the previous run will affect your current run.

            optimizer.zero_grad()

            # forward pass - all the magic happens here.

            outputs = model(data)

            # our loss function calculates the loss between the outputs and 
            # targets vectors
            loss = criterion(outputs, target)

            # let's print some output so we know the network is doing something

            if i%50==0:
                print("Reached iteration ", i)
                running_loss += loss.item()

            # backward pass - the magic actually happens here.
            # and the optimizer step - where the parameters are updated based on 
            # the gradients calculated in the backward pass.
            
            loss.backward()
            optimizer.step()

            # let's keep a track of the loss through an epoch.

            running_loss += loss.item()
        if epoch % 10 == 0:
            save(model, optimizer, loss, 'faceRecog.saved.model')
        print(running_loss)


# In[511]:


# define the number of classes in your problem
# For gender recognition, I have 2 classes.

num_classes = 10


# In[512]:


# Initialize your model's network architecture by calling S3fd

myModel = s3fd(num_classes)


# In[513]:


# Load the pre-trained model.

loadedModel = torch.load('s3fd_convert.pth')


# In[514]:


# Our new model has different layers from the old model.

newModel = myModel.state_dict()


# In[515]:


# We only want to use the layers from the pre-trained model that are defined in
# our new model

pretrained_dict = {k: v for k, v in loadedModel.items() if k in newModel}


# In[516]:


# Let's update our model from weights extracted from the pre-trained model

newModel.update(pretrained_dict)


# In[517]:


# Load the state and we are good to go.

myModel.load_state_dict(newModel)


# In[518]:


use_cuda = True
myModel.eval()


# In[519]:


# Define your loss function to what you want.
# Play around with this.

criterion = nn.MSELoss()

# Turn the requires_grad off for all the parameters in your model.
# Remember you don't want to train and change the weights of any of the layers
# except the final FC layer.

for param in myModel.parameters():
    param.requires_grad = False

# This layer was already there but we can re-instantiate it.
# fc_1 layer by default will now contain requires_grad = True.
# This will be the only layer that actually learns weights from the data.

myModel.fc_1 = nn.Linear(1024, num_classes)


# In[520]:


# Define the optimization method, learning rate, momentum.
# You can use ADAM etc.
# Play around with this.
# Note that we need to send it the parameters to be optimized.
# We filter and only send it those parameters with requires_grad = True.
# (i.e. the FC Layer params)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()), lr=0.0001, momentum=0.9)

if use_cuda:
    myModel = myModel.type(torch.FloatTensor).cuda()


# In[521]:


# Call the training function defined above.
model_ft = train_model(myModel, criterion, optimizer, num_classes, num_epochs=100)


# In[ ]:





# In[ ]:




